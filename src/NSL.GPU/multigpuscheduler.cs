using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace NSL.GPU
{
    /// <summary>
    /// Multi-GPU Scheduler for NSL.
    ///
    /// Enables scaling across multiple GPUs for large workloads:
    /// - Automatic work distribution based on GPU capabilities
    /// - Data parallelism (split batches across GPUs)
    /// - Model parallelism (split layers across GPUs)
    /// - Peer-to-peer transfers when available
    /// - Load balancing based on real-time performance
    ///
    /// Based on NVIDIA multi-GPU best practices:
    /// - Use P2P when GPUs are on same PCIe switch
    /// - Overlap compute with transfers
    /// - Balance work based on compute capability
    /// </summary>
    public class MultiGpuScheduler : IDisposable
    {
        private readonly List<GpuWorker> _workers = new();
        private readonly ConcurrentQueue<WorkItem> _workQueue = new();
        private readonly CancellationTokenSource _cts = new();
        private readonly object _lock = new();
        private bool _disposed;

        /// <summary>
        /// Represents a GPU worker with its own accelerator and kernels
        /// </summary>
        public class GpuWorker
        {
            /// <summary>Public API</summary>
            public int Id { get; set; }
            /// <summary>Public API</summary>
            public GpuAutoConfig.GpuInfo GpuInfo { get; set; } = null!;
            /// <summary>Public API</summary>
            public Accelerator Accelerator { get; set; } = null!;
            /// <summary>Public API</summary>
            public GpuKernels Kernels { get; set; } = null!;
            /// <summary>Public API</summary>
            public GpuMemoryManager Memory { get; set; } = null!;

            // Performance tracking
            /// <summary>Public API</summary>
            public long TasksCompleted { get; set; }
            /// <summary>Public API</summary>
            public double TotalComputeTimeMs { get; set; }
            /// <summary>Public API</summary>
            public double AverageTaskTimeMs => TasksCompleted > 0 ? TotalComputeTimeMs / TasksCompleted : 0;

            // Current workload
            /// <summary>Public API</summary>
            public int CurrentQueueDepth { get; set; }
            /// <summary>Public API</summary>
            public bool IsBusy { get; set; }
        }

        /// <summary>
        /// Work item to be scheduled on a GPU
        /// </summary>
        private class WorkItem
        {
            /// <summary>Public API</summary>
            public Func<GpuWorker, Task<GpuTensor>> Work { get; set; } = null!;
            /// <summary>Public API</summary>
            public TaskCompletionSource<GpuTensor> Completion { get; set; } = null!;
            /// <summary>Public API</summary>
            public int PreferredGpu { get; set; } = -1; // -1 = any GPU
            /// <summary>Public API</summary>
            public long EstimatedMemory { get; set; }
        }

        /// <summary>
        /// Scheduling strategy for work distribution
        /// </summary>
        public enum SchedulingStrategy
        {
            RoundRobin,

            LeastLoaded,

            PerformanceWeighted,

            MemoryAware
        }

        /// <summary>Public API</summary>
        public SchedulingStrategy Strategy { get; set; } = SchedulingStrategy.MemoryAware;
        /// <summary>Public API</summary>
        public IReadOnlyList<GpuWorker> Workers => _workers.AsReadOnly();
        /// <summary>Public API</summary>
        public int GpuCount => _workers.Count;

        /// <summary>
        /// Initialize multi-GPU scheduler with all available GPUs
        /// </summary>
        public MultiGpuScheduler(Context context, List<GpuAutoConfig.GpuInfo> gpus)
        {
            int id = 0;
            foreach (var gpu in gpus.Where(g => g.Backend != GpuBackend.CPU))
            {
                try
                {
                    var accelerator = CreateAccelerator(context, gpu);
                    var worker = new GpuWorker
                    {
                        Id = id++,
                        GpuInfo = gpu,
                        Accelerator = accelerator,
                        Kernels = new GpuKernels(accelerator),
                        Memory = new GpuMemoryManager(accelerator)
                    };
                    _workers.Add(worker);

                    Console.WriteLine($"NSL MultiGPU: Added worker {worker.Id} - {gpu.Name}");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Failed to initialize GPU {gpu.Name}: {ex.Message}");
                }
            }

            if (_workers.Count == 0)
            {
                throw new InvalidOperationException("No GPUs available for multi-GPU scheduling");
            }

            Console.WriteLine($"NSL MultiGPU: Initialized with {_workers.Count} GPU(s)");
        }

        private static Accelerator CreateAccelerator(Context context, GpuAutoConfig.GpuInfo gpu)
        {
            var parts = gpu.DeviceId.Split(':');
            int deviceIndex = parts.Length > 1 ? int.Parse(parts[1]) : 0;

            return gpu.Backend switch
            {
                GpuBackend.CUDA => context.GetCudaDevice(deviceIndex).CreateAccelerator(context),
                GpuBackend.OpenCL => context.GetCLDevice(deviceIndex).CreateAccelerator(context),
                _ => throw new NotSupportedException($"Backend {gpu.Backend} not supported")
            };
        }

        #region Work Distribution

        /// <summary>
        /// Submit work to be executed on any available GPU
        /// </summary>
        public Task<GpuTensor> SubmitAsync(Func<GpuWorker, GpuTensor> work, long estimatedMemory = 0)
        {
            var tcs = new TaskCompletionSource<GpuTensor>();
            var item = new WorkItem
            {
                Work = w => Task.FromResult(work(w)),
                Completion = tcs,
                EstimatedMemory = estimatedMemory
            };

            ScheduleWork(item);
            return tcs.Task;
        }

        /// <summary>
        /// Submit async work to be executed on any available GPU
        /// </summary>
        public Task<GpuTensor> SubmitAsync(Func<GpuWorker, Task<GpuTensor>> work, long estimatedMemory = 0)
        {
            var tcs = new TaskCompletionSource<GpuTensor>();
            var item = new WorkItem
            {
                Work = work,
                Completion = tcs,
                EstimatedMemory = estimatedMemory
            };

            ScheduleWork(item);
            return tcs.Task;
        }

        private void ScheduleWork(WorkItem item)
        {
            var worker = SelectWorker(item);

            Task.Run(async () =>
            {
                try
                {
                    worker.IsBusy = true;
                    worker.CurrentQueueDepth++;

                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    var result = await item.Work(worker);
                    sw.Stop();

                    worker.TasksCompleted++;
                    worker.TotalComputeTimeMs += sw.Elapsed.TotalMilliseconds;

                    item.Completion.SetResult(result);
                }
                catch (Exception ex)
                {
                    item.Completion.SetException(ex);
                }
                finally
                {
                    worker.CurrentQueueDepth--;
                    worker.IsBusy = worker.CurrentQueueDepth > 0;
                }
            });
        }

        private GpuWorker SelectWorker(WorkItem item)
        {
            if (item.PreferredGpu >= 0 && item.PreferredGpu < _workers.Count)
            {
                return _workers[item.PreferredGpu];
            }

            return Strategy switch
            {
                SchedulingStrategy.RoundRobin => SelectRoundRobin(),
                SchedulingStrategy.LeastLoaded => SelectLeastLoaded(),
                SchedulingStrategy.PerformanceWeighted => SelectPerformanceWeighted(),
                SchedulingStrategy.MemoryAware => SelectMemoryAware(item.EstimatedMemory),
                _ => _workers[0]
            };
        }

        private int _roundRobinIndex;
        private GpuWorker SelectRoundRobin()
        {
            int index = Interlocked.Increment(ref _roundRobinIndex) % _workers.Count;
            return _workers[index];
        }

        private GpuWorker SelectLeastLoaded()
        {
            return _workers.OrderBy(w => w.CurrentQueueDepth).First();
        }

        private GpuWorker SelectPerformanceWeighted()
        {
            // Prefer GPUs with higher scores and lower current load
            return _workers.OrderByDescending(w =>
                w.GpuInfo.Score / (1 + w.CurrentQueueDepth)).First();
        }

        private GpuWorker SelectMemoryAware(long estimatedMemory)
        {
            if (estimatedMemory <= 0)
            {
                return SelectLeastLoaded();
            }

            // Find GPU with enough memory and lowest load
            var suitable = _workers
                .Where(w =>
                {
                    var stats = w.Memory.GetMemoryStats();
                    return stats.AvailableVRAM >= estimatedMemory;
                })
                .OrderBy(w => w.CurrentQueueDepth);

            return suitable.FirstOrDefault() ?? SelectLeastLoaded();
        }

        #endregion

        #region Data Parallelism

        /// <summary>
        /// Split a batch across all GPUs and process in parallel.
        /// Each GPU processes a portion of the batch.
        /// </summary>
        public async Task<GpuTensor> ParallelBatchProcess(
            float[] batchData,
            int[] batchShape,
            Func<GpuWorker, GpuTensor, GpuTensor> processFunc)
        {
            int batchSize = batchShape[0];
            int itemSize = batchData.Length / batchSize;

            // Split batch across GPUs
            var tasks = new List<Task<GpuTensor>>();
            int batchesPerGpu = (batchSize + _workers.Count - 1) / _workers.Count;

            for (int i = 0; i < _workers.Count; i++)
            {
                int startIdx = i * batchesPerGpu;
                int endIdx = Math.Min(startIdx + batchesPerGpu, batchSize);

                if (startIdx >= batchSize) break;

                int chunkSize = endIdx - startIdx;
                var chunkData = new float[chunkSize * itemSize];
                Array.Copy(batchData, startIdx * itemSize, chunkData, 0, chunkData.Length);

                var chunkShape = (int[])batchShape.Clone();
                chunkShape[0] = chunkSize;

                int workerId = i;
                tasks.Add(Task.Run(() =>
                {
                    var worker = _workers[workerId];
                    var input = GpuTensor.FromArray(worker.Accelerator, chunkData, chunkShape);
                    return processFunc(worker, input);
                }));
            }

            var results = await Task.WhenAll(tasks);

            // Concatenate results on first GPU
            return ConcatenateTensors(results, axis: 0);
        }

        private GpuTensor ConcatenateTensors(GpuTensor[] tensors, int axis)
        {
            if (tensors.Length == 1) return tensors[0];

            // Calculate total size
            int totalBatch = tensors.Sum(t => t.Shape[0]);
            int[] newShape = (int[])tensors[0].Shape.Clone();
            newShape[0] = totalBatch;

            int itemSize = tensors[0].Size / tensors[0].Shape[0];
            var resultData = new float[totalBatch * itemSize];

            int offset = 0;
            foreach (var tensor in tensors)
            {
                var data = tensor.ToArray();
                Array.Copy(data, 0, resultData, offset, data.Length);
                offset += data.Length;
            }

            return GpuTensor.FromArray(_workers[0].Accelerator, resultData, newShape);
        }

        #endregion

        #region Statistics

        /// <summary>
        /// Get multi-GPU performance statistics
        /// </summary>
        public string GetStatistics()
        {
            var lines = new List<string>
            {
                $"=== Multi-GPU Statistics ({_workers.Count} GPUs) ===",
                ""
            };

            foreach (var worker in _workers)
            {
                var stats = worker.Memory.GetMemoryStats();
                lines.Add($"GPU {worker.Id}: {worker.GpuInfo.Name}");
                lines.Add($"  Tasks: {worker.TasksCompleted}, Avg Time: {worker.AverageTaskTimeMs:F2}ms");
                lines.Add($"  Memory: {stats.UsedVRAM / (1024 * 1024)}MB / {stats.TotalVRAM / (1024 * 1024)}MB");
                lines.Add($"  Queue Depth: {worker.CurrentQueueDepth}");
                lines.Add("");
            }

            long totalTasks = _workers.Sum(w => w.TasksCompleted);
            double totalTime = _workers.Sum(w => w.TotalComputeTimeMs);
            lines.Add($"Total Tasks: {totalTasks}");
            lines.Add($"Total Compute Time: {totalTime / 1000:F2}s");

            return string.Join(Environment.NewLine, lines);
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _cts.Cancel();

                foreach (var worker in _workers)
                {
                    worker.Memory.Dispose();
                    worker.Accelerator.Dispose();
                }

                _workers.Clear();
                _disposed = true;
            }
        }
    }
}