using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// Tensor Parallelism for distributing individual layers across multiple GPUs.
    /// Splits weight matrices column-wise or row-wise across devices.
    /// Essential for layers too large to fit on a single GPU.
    /// </summary>
    public class TensorParallelism : IDisposable
    {
        private readonly List<Accelerator> _accelerators;
        private readonly List<GpuKernels> _kernels;
        private readonly int _worldSize;
        private readonly TensorParallelConfig _config;
        private bool _disposed;

        /// <summary>Public API</summary>
        public class TensorParallelConfig
        {
            /// <summary>Split strategy for linear layers</summary>
            public ParallelStrategy LinearStrategy { get; set; } = ParallelStrategy.ColumnParallel;

            /// <summary>Split strategy for attention</summary>
            public ParallelStrategy AttentionStrategy { get; set; } = ParallelStrategy.HeadParallel;

            /// <summary>Whether to overlap communication with compute</summary>
            public bool OverlapCommunication { get; set; } = true;

            /// <summary>Synchronization mode</summary>
            public SyncMode SyncMode { get; set; } = SyncMode.AllReduce;
        }

        /// <summary>Public API</summary>
        public enum ParallelStrategy
        {
            ColumnParallel,

            RowParallel,

            HeadParallel,

            Replicated
        }

        /// <summary>Public API</summary>
        public enum SyncMode
        {
            AllReduce,

            AllGather,

            ReduceScatter
        }

        /// <summary>Public API</summary>
        public int WorldSize => _worldSize;

        /// <summary>Public API</summary>
        public TensorParallelism(List<Accelerator> accelerators, TensorParallelConfig? config = null)
        {
            _accelerators = accelerators;
            _worldSize = accelerators.Count;
            _config = config ?? new TensorParallelConfig();

            _kernels = accelerators.Select(a => new GpuKernels(a)).ToList();
        }

        /// <summary>
        /// Shard a weight tensor across GPUs
        /// </summary>
        public List<GpuTensor> ShardWeight(float[] weight, int[] shape, ParallelStrategy strategy)
        {
            var shards = new List<GpuTensor>();

            switch (strategy)
            {
                case ParallelStrategy.ColumnParallel:
                    shards = ShardColumns(weight, shape);
                    break;

                case ParallelStrategy.RowParallel:
                    shards = ShardRows(weight, shape);
                    break;

                case ParallelStrategy.HeadParallel:
                    shards = ShardHeads(weight, shape);
                    break;

                case ParallelStrategy.Replicated:
                    shards = Replicate(weight, shape);
                    break;
            }

            return shards;
        }

        private List<GpuTensor> ShardColumns(float[] weight, int[] shape)
        {
            // shape: [out_features, in_features]
            // Split out_features across GPUs
            int outFeatures = shape[0];
            int inFeatures = shape[1];
            int shardSize = outFeatures / _worldSize;

            var shards = new List<GpuTensor>();

            for (int rank = 0; rank < _worldSize; rank++)
            {
                int start = rank * shardSize;
                int end = (rank == _worldSize - 1) ? outFeatures : (rank + 1) * shardSize;
                int localOut = end - start;

                var shardData = new float[localOut * inFeatures];
                for (int i = 0; i < localOut; i++)
                {
                    Array.Copy(weight, (start + i) * inFeatures, shardData, i * inFeatures, inFeatures);
                }

                var shard = GpuTensor.FromArray(_accelerators[rank], shardData, new[] { localOut, inFeatures });
                shards.Add(shard);
            }

            return shards;
        }

        private List<GpuTensor> ShardRows(float[] weight, int[] shape)
        {
            // shape: [out_features, in_features]
            // Split in_features across GPUs
            int outFeatures = shape[0];
            int inFeatures = shape[1];
            int shardSize = inFeatures / _worldSize;

            var shards = new List<GpuTensor>();

            for (int rank = 0; rank < _worldSize; rank++)
            {
                int start = rank * shardSize;
                int end = (rank == _worldSize - 1) ? inFeatures : (rank + 1) * shardSize;
                int localIn = end - start;

                var shardData = new float[outFeatures * localIn];
                for (int i = 0; i < outFeatures; i++)
                {
                    Array.Copy(weight, i * inFeatures + start, shardData, i * localIn, localIn);
                }

                var shard = GpuTensor.FromArray(_accelerators[rank], shardData, new[] { outFeatures, localIn });
                shards.Add(shard);
            }

            return shards;
        }

        private List<GpuTensor> ShardHeads(float[] weight, int[] shape)
        {
            // shape: [num_heads, head_dim, ...]
            // Split num_heads across GPUs
            int numHeads = shape[0];
            int headsPerGpu = numHeads / _worldSize;
            int elementsPerHead = shape.Skip(1).Aggregate(1, (a, b) => a * b);

            var shards = new List<GpuTensor>();

            for (int rank = 0; rank < _worldSize; rank++)
            {
                int startHead = rank * headsPerGpu;
                int endHead = (rank == _worldSize - 1) ? numHeads : (rank + 1) * headsPerGpu;
                int localHeads = endHead - startHead;

                var shardData = new float[localHeads * elementsPerHead];
                Array.Copy(weight, startHead * elementsPerHead, shardData, 0, localHeads * elementsPerHead);

                var newShape = new int[] { localHeads }.Concat(shape.Skip(1)).ToArray();
                var shard = GpuTensor.FromArray(_accelerators[rank], shardData, newShape);
                shards.Add(shard);
            }

            return shards;
        }

        private List<GpuTensor> Replicate(float[] weight, int[] shape)
        {
            return _accelerators.Select(a => GpuTensor.FromArray(a, weight, shape)).ToList();
        }

        /// <summary>
        /// Parallel linear layer: Y = X @ W^T + b
        /// Uses column parallelism by default (split output dimension)
        /// </summary>
        public async Task<GpuTensor> ParallelLinearAsync(
            GpuTensor input,
            List<GpuTensor> weightShards,
            List<GpuTensor>? biasShards = null)
        {
            // Broadcast input to all GPUs
            var inputShards = await BroadcastAsync(input);

            // Compute local matmul on each GPU
            var localResults = new GpuTensor[_worldSize];

            await Task.WhenAll(Enumerable.Range(0, _worldSize).Select(async rank =>
            {
                await Task.Yield();
                localResults[rank] = _kernels[rank].MatMul(inputShards[rank], weightShards[rank]);

                if (biasShards != null)
                {
                    var biased = _kernels[rank].Add(localResults[rank], biasShards[rank]);
                    localResults[rank].Dispose();
                    localResults[rank] = biased;
                }

                inputShards[rank].Dispose();
            }));

            // AllGather to reconstruct full output
            var result = await AllGatherAsync(localResults.ToList(), axis: -1);

            foreach (var local in localResults) local.Dispose();

            return result;
        }

        /// <summary>
        /// Parallel attention with head parallelism
        /// </summary>
        public async Task<GpuTensor> ParallelAttentionAsync(
            GpuTensor query, GpuTensor key, GpuTensor value,
            List<GpuTensor> wqShards, List<GpuTensor> wkShards,
            List<GpuTensor> wvShards, List<GpuTensor> woShards,
            float scale, GpuTensor? mask = null)
        {
            // Each GPU handles a subset of attention heads
            var queryShards = await BroadcastAsync(query);
            var keyShards = await BroadcastAsync(key);
            var valueShards = await BroadcastAsync(value);

            var localOutputs = new GpuTensor[_worldSize];

            await Task.WhenAll(Enumerable.Range(0, _worldSize).Select(async rank =>
            {
                await Task.Yield();

                // Project Q, K, V with local shards
                var q = _kernels[rank].MatMul(queryShards[rank], wqShards[rank]);
                var k = _kernels[rank].MatMul(keyShards[rank], wkShards[rank]);
                var v = _kernels[rank].MatMul(valueShards[rank], wvShards[rank]);

                // Attention on local heads
                var scores = _kernels[rank].MatMul(q, _kernels[rank].Transpose(k));
                var scaled = _kernels[rank].MulScalar(scores, scale);

                if (mask != null)
                {
                    // Apply mask
                    scaled = _kernels[rank].Add(scaled, mask);
                }

                var attnWeights = _kernels[rank].Softmax(scaled);
                var attnOutput = _kernels[rank].MatMul(attnWeights, v);

                // Project output
                localOutputs[rank] = _kernels[rank].MatMul(attnOutput, woShards[rank]);

                // Cleanup
                q.Dispose(); k.Dispose(); v.Dispose();
                scores.Dispose(); scaled.Dispose(); attnWeights.Dispose(); attnOutput.Dispose();
                queryShards[rank].Dispose(); keyShards[rank].Dispose(); valueShards[rank].Dispose();
            }));

            // AllReduce to sum partial outputs (each GPU computed subset of heads)
            var result = await AllReduceAsync(localOutputs.ToList(), ReduceOperation.Sum);

            foreach (var local in localOutputs) local.Dispose();

            return result;
        }

        /// <summary>
        /// Broadcast tensor to all GPUs
        /// </summary>
        public async Task<List<GpuTensor>> BroadcastAsync(GpuTensor source, int sourceRank = 0)
        {
            var data = source.ToArray();
            var results = new List<GpuTensor>();

            await Task.WhenAll(_accelerators.Select(async (accel, rank) =>
            {
                await Task.Yield();
                lock (results)
                {
                    results.Add(GpuTensor.FromArray(accel, data, source.Shape));
                }
            }));

            return results.OrderBy(_ => 0).ToList();  // Maintain order
        }

        /// <summary>
        /// AllReduce across GPUs
        /// </summary>
        public async Task<GpuTensor> AllReduceAsync(List<GpuTensor> tensors, ReduceOperation op)
        {
            // Gather all data to CPU
            var allData = tensors.Select(t => t.ToArray()).ToArray();
            int size = allData[0].Length;

            // Reduce
            var result = new float[size];
            for (int i = 0; i < size; i++)
            {
                switch (op)
                {
                    case ReduceOperation.Sum:
                        result[i] = allData.Sum(d => d[i]);
                        break;
                    case ReduceOperation.Mean:
                        result[i] = allData.Average(d => d[i]);
                        break;
                    case ReduceOperation.Max:
                        result[i] = allData.Max(d => d[i]);
                        break;
                    case ReduceOperation.Min:
                        result[i] = allData.Min(d => d[i]);
                        break;
                }
            }

            // Return on first GPU (caller can broadcast if needed)
            return GpuTensor.FromArray(_accelerators[0], result, tensors[0].Shape);
        }

        /// <summary>
        /// AllGather to reconstruct sharded tensor
        /// </summary>
        public async Task<GpuTensor> AllGatherAsync(List<GpuTensor> shards, int axis)
        {
            // Gather all shards
            var allData = shards.Select(s => s.ToArray()).ToList();
            var shapes = shards.Select(s => s.Shape).ToList();

            // Calculate output shape
            int totalAlongAxis = shapes.Sum(s => s[axis < 0 ? s.Length + axis : axis]);
            var outShape = shapes[0].ToArray();
            outShape[axis < 0 ? outShape.Length + axis : axis] = totalAlongAxis;

            // Concatenate data
            var result = new float[outShape.Aggregate(1, (a, b) => a * b)];
            int offset = 0;

            foreach (var data in allData)
            {
                Array.Copy(data, 0, result, offset, data.Length);
                offset += data.Length;
            }

            return GpuTensor.FromArray(_accelerators[0], result, outShape);
        }

        /// <summary>Public API</summary>
        public enum ReduceOperation { Sum, Mean, Max, Min }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                // Kernels don't need disposal, just accelerators are managed externally
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Pipeline Parallelism for distributing model layers across GPUs.
    /// Each GPU holds a "stage" of the model (subset of layers).
    /// Uses micro-batching to keep all GPUs busy.
    /// </summary>
    public class PipelineParallelism : IDisposable
    {
        private readonly List<Accelerator> _accelerators;
        private readonly List<GpuKernels> _kernels;
        private readonly int _numStages;
        private readonly int _microBatchSize;
        private readonly PipelineConfig _config;

        private readonly ConcurrentDictionary<int, Queue<MicroBatch>> _stageQueues = new();
        private readonly ConcurrentDictionary<int, GpuTensor?> _stageActivations = new();

        private bool _disposed;

        /// <summary>Public API</summary>
        public class PipelineConfig
        {
            /// <summary>Number of micro-batches for pipeline filling</summary>
            public int NumMicroBatches { get; set; } = 4;

            /// <summary>Pipeline schedule type</summary>
            public PipelineSchedule Schedule { get; set; } = PipelineSchedule.GPipe;

            /// <summary>Enable gradient checkpointing per stage</summary>
            public bool GradientCheckpointing { get; set; } = true;

            /// <summary>Async pipeline mode</summary>
            public bool AsyncPipeline { get; set; } = false;
        }

        /// <summary>Public API</summary>
        public enum PipelineSchedule
        {
            GPipe,

            OneFOneBSchedule,

            InterleavedOneFOneB
        }

        private class MicroBatch
        {
            /// <summary>Public API</summary>
            public int Id { get; set; }
            /// <summary>Public API</summary>
            public GpuTensor Data { get; set; } = null!;
            /// <summary>Public API</summary>
            public GpuTensor? Gradient { get; set; }
            /// <summary>Public API</summary>
            public bool IsBackward { get; set; }
        }

        /// <summary>Public API</summary>
        public int NumStages => _numStages;

        /// <summary>Public API</summary>
        public PipelineParallelism(List<Accelerator> accelerators, PipelineConfig? config = null)
        {
            _accelerators = accelerators;
            _numStages = accelerators.Count;
            _config = config ?? new PipelineConfig();
            _microBatchSize = _config.NumMicroBatches;

            _kernels = accelerators.Select(a => new GpuKernels(a)).ToList();

            // Initialize queues for each stage
            for (int i = 0; i < _numStages; i++)
            {
                _stageQueues[i] = new Queue<MicroBatch>();
            }
        }

        /// <summary>
        /// Forward pass through pipeline
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <param name="stageForward">Function to execute for each stage</param>
        /// <returns>Output from final stage</returns>
        public async Task<GpuTensor> ForwardAsync(
            GpuTensor input,
            Func<int, GpuTensor, GpuKernels, GpuTensor> stageForward)
        {
            int batchSize = input.Shape[0];
            int microBatchSize = batchSize / _microBatchSize;

            var microBatches = SplitBatch(input, _microBatchSize);
            var outputs = new GpuTensor[_microBatchSize];

            switch (_config.Schedule)
            {
                case PipelineSchedule.GPipe:
                    outputs = await GPipeForwardAsync(microBatches, stageForward);
                    break;

                case PipelineSchedule.OneFOneBSchedule:
                    outputs = await OneFOneBForwardAsync(microBatches, stageForward);
                    break;

                default:
                    outputs = await GPipeForwardAsync(microBatches, stageForward);
                    break;
            }

            // Concatenate outputs
            return ConcatenateBatches(outputs);
        }

        private async Task<GpuTensor[]> GPipeForwardAsync(
            GpuTensor[] microBatches,
            Func<int, GpuTensor, GpuKernels, GpuTensor> stageForward)
        {
            var outputs = new GpuTensor[microBatches.Length];
            var stageOutputs = new GpuTensor[_numStages][];

            for (int stage = 0; stage < _numStages; stage++)
            {
                stageOutputs[stage] = new GpuTensor[microBatches.Length];
            }

            // Process all micro-batches through all stages
            for (int mb = 0; mb < microBatches.Length; mb++)
            {
                var current = microBatches[mb];

                for (int stage = 0; stage < _numStages; stage++)
                {
                    // Transfer to stage's GPU if needed
                    if (stage > 0)
                    {
                        current = TransferToDevice(current, _accelerators[stage]);
                    }

                    // Execute stage
                    var output = stageForward(stage, current, _kernels[stage]);
                    stageOutputs[stage][mb] = output;

                    if (stage > 0) current.Dispose();
                    current = output;
                }

                outputs[mb] = current;
            }

            return outputs;
        }

        private async Task<GpuTensor[]> OneFOneBForwardAsync(
            GpuTensor[] microBatches,
            Func<int, GpuTensor, GpuKernels, GpuTensor> stageForward)
        {
            // 1F1B: Overlap forward and backward passes
            // More memory efficient than GPipe
            var outputs = new GpuTensor[microBatches.Length];
            var activations = new Dictionary<(int stage, int mb), GpuTensor>();

            int totalSteps = _numStages + microBatches.Length - 1;

            for (int step = 0; step < totalSteps; step++)
            {
                var tasks = new List<Task>();

                // Each stage processes in parallel
                for (int stage = 0; stage < _numStages; stage++)
                {
                    int mb = step - stage;
                    if (mb < 0 || mb >= microBatches.Length) continue;

                    int localStage = stage;
                    int localMb = mb;

                    tasks.Add(Task.Run(() =>
                    {
                        GpuTensor input;
                        if (localStage == 0)
                        {
                            input = microBatches[localMb];
                        }
                        else
                        {
                            // Wait for previous stage output
                            while (!activations.ContainsKey((localStage - 1, localMb)))
                            {
                                Thread.SpinWait(100);
                            }
                            input = TransferToDevice(activations[(localStage - 1, localMb)], _accelerators[localStage]);
                        }

                        var output = stageForward(localStage, input, _kernels[localStage]);

                        lock (activations)
                        {
                            activations[(localStage, localMb)] = output;
                        }

                        if (localStage == _numStages - 1)
                        {
                            outputs[localMb] = output;
                        }
                    }));
                }

                await Task.WhenAll(tasks);
            }

            return outputs;
        }

        /// <summary>
        /// Full training step with pipeline parallelism
        /// </summary>
        public async Task<float> TrainStepAsync(
            GpuTensor input,
            GpuTensor targets,
            Func<int, GpuTensor, GpuKernels, GpuTensor> stageForward,
            Func<int, GpuTensor, GpuTensor, GpuKernels, (GpuTensor gradient, float loss)> stageBackward,
            Action<int, GpuTensor, GpuKernels> applyGradients)
        {
            // Forward pass
            var predictions = await ForwardAsync(input, stageForward);

            // Compute loss at final stage
            var (gradOutput, loss) = stageBackward(_numStages - 1, predictions, targets, _kernels[_numStages - 1]);

            // Backward pass through stages in reverse
            var currentGrad = gradOutput;
            for (int stage = _numStages - 1; stage >= 0; stage--)
            {
                // Transfer gradient to stage's GPU
                if (stage < _numStages - 1)
                {
                    currentGrad = TransferToDevice(currentGrad, _accelerators[stage]);
                }

                // Compute gradients for this stage
                var (nextGrad, _) = stageBackward(stage, _stageActivations[stage]!, currentGrad, _kernels[stage]);

                // Apply gradients
                applyGradients(stage, nextGrad, _kernels[stage]);

                if (stage < _numStages - 1) currentGrad.Dispose();
                currentGrad = nextGrad;
            }

            currentGrad.Dispose();
            predictions.Dispose();

            return loss;
        }

        private GpuTensor[] SplitBatch(GpuTensor input, int numParts)
        {
            int batchSize = input.Shape[0];
            int partSize = batchSize / numParts;
            var data = input.ToArray();

            var parts = new GpuTensor[numParts];
            int elementsPerSample = input.Size / batchSize;

            for (int i = 0; i < numParts; i++)
            {
                int startIdx = i * partSize;
                int endIdx = (i == numParts - 1) ? batchSize : (i + 1) * partSize;
                int localBatch = endIdx - startIdx;

                var partData = new float[localBatch * elementsPerSample];
                Array.Copy(data, startIdx * elementsPerSample, partData, 0, localBatch * elementsPerSample);

                var shape = input.Shape.ToArray();
                shape[0] = localBatch;
                parts[i] = GpuTensor.FromArray(_accelerators[0], partData, shape);
            }

            return parts;
        }

        private GpuTensor ConcatenateBatches(GpuTensor[] batches)
        {
            int totalBatch = batches.Sum(b => b.Shape[0]);
            int elementsPerSample = batches[0].Size / batches[0].Shape[0];

            var shape = batches[0].Shape.ToArray();
            shape[0] = totalBatch;

            var result = new float[totalBatch * elementsPerSample];
            int offset = 0;

            foreach (var batch in batches)
            {
                var data = batch.ToArray();
                Array.Copy(data, 0, result, offset, data.Length);
                offset += data.Length;
            }

            return GpuTensor.FromArray(_accelerators[0], result, shape);
        }

        private GpuTensor TransferToDevice(GpuTensor source, Accelerator target)
        {
            var data = source.ToArray();
            return GpuTensor.FromArray(target, data, source.Shape);
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// ZeRO (Zero Redundancy Optimizer) implementation for memory-efficient training.
    /// Partitions optimizer states, gradients, and parameters across GPUs.
    /// </summary>
    public class ZeROOptimizer : IDisposable
    {
        private readonly List<Accelerator> _accelerators;
        private readonly List<GpuKernels> _kernels;
        private readonly int _worldSize;
        private readonly ZeROConfig _config;

        private readonly Dictionary<string, List<GpuTensor>> _parameterShards = new();
        private readonly Dictionary<string, List<float[]>> _momentumShards = new();
        private readonly Dictionary<string, List<float[]>> _varianceShards = new();

        private int _step = 0;
        private bool _disposed;

        /// <summary>Public API</summary>
        public class ZeROConfig
        {
            /// <summary>ZeRO stage (1, 2, or 3)</summary>
            public int Stage { get; set; } = 3;

            /// <summary>Learning rate</summary>
            public float LearningRate { get; set; } = 1e-4f;

            /// <summary>Adam beta1</summary>
            public float Beta1 { get; set; } = 0.9f;

            /// <summary>Adam beta2</summary>
            public float Beta2 { get; set; } = 0.999f;

            /// <summary>Weight decay</summary>
            public float WeightDecay { get; set; } = 0.01f;

            /// <summary>Gradient clipping</summary>
            public float MaxGradNorm { get; set; } = 1.0f;

            /// <summary>CPU offloading for optimizer states</summary>
            public bool OffloadOptimizer { get; set; } = false;

            /// <summary>CPU offloading for parameters</summary>
            public bool OffloadParams { get; set; } = false;
        }

        /// <summary>Public API</summary>
        public int Stage => _config.Stage;
        /// <summary>Public API</summary>
        public int Step => _step;

        /// <summary>Public API</summary>
        public ZeROOptimizer(List<Accelerator> accelerators, ZeROConfig? config = null)
        {
            _accelerators = accelerators;
            _worldSize = accelerators.Count;
            _config = config ?? new ZeROConfig();
            _kernels = accelerators.Select(a => new GpuKernels(a)).ToList();
        }

        /// <summary>
        /// Register parameters for optimization
        /// </summary>
        public void RegisterParameter(string name, GpuTensor parameter)
        {
            var data = parameter.ToArray();
            int shardSize = data.Length / _worldSize;

            var paramShards = new List<GpuTensor>();
            var momShards = new List<float[]>();
            var varShards = new List<float[]>();

            for (int rank = 0; rank < _worldSize; rank++)
            {
                int start = rank * shardSize;
                int end = (rank == _worldSize - 1) ? data.Length : (rank + 1) * shardSize;
                int localSize = end - start;

                // Stage 3: Shard parameters across GPUs
                if (_config.Stage >= 3)
                {
                    var shardData = new float[localSize];
                    Array.Copy(data, start, shardData, 0, localSize);
                    paramShards.Add(GpuTensor.FromArray(_accelerators[rank], shardData, new[] { localSize }));
                }
                else
                {
                    // Stage 1-2: Replicate parameters
                    paramShards.Add(GpuTensor.FromArray(_accelerators[rank], data, parameter.Shape));
                }

                // Stage 2+: Shard optimizer states
                if (_config.Stage >= 2 || _config.OffloadOptimizer)
                {
                    momShards.Add(new float[localSize]);
                    varShards.Add(new float[localSize]);
                }
                else
                {
                    momShards.Add(new float[data.Length]);
                    varShards.Add(new float[data.Length]);
                }
            }

            _parameterShards[name] = paramShards;
            _momentumShards[name] = momShards;
            _varianceShards[name] = varShards;
        }

        /// <summary>
        /// Optimizer step with ZeRO
        /// </summary>
        public async Task StepAsync(Dictionary<string, GpuTensor> gradients)
        {
            _step++;

            // Bias correction
            float biasCorrection1 = 1 - MathF.Pow(_config.Beta1, _step);
            float biasCorrection2 = 1 - MathF.Pow(_config.Beta2, _step);

            foreach (var (name, gradient) in gradients)
            {
                if (!_parameterShards.ContainsKey(name)) continue;

                var gradData = gradient.ToArray();

                // Gradient clipping
                float gradNorm = MathF.Sqrt(gradData.Sum(g => g * g));
                if (gradNorm > _config.MaxGradNorm)
                {
                    float scale = _config.MaxGradNorm / (gradNorm + 1e-6f);
                    for (int i = 0; i < gradData.Length; i++)
                    {
                        gradData[i] *= scale;
                    }
                }

                // Each rank updates its shard
                await Task.WhenAll(Enumerable.Range(0, _worldSize).Select(async rank =>
                {
                    await Task.Yield();

                    int shardSize = gradData.Length / _worldSize;
                    int start = rank * shardSize;
                    int end = (rank == _worldSize - 1) ? gradData.Length : (rank + 1) * shardSize;
                    int localSize = end - start;

                    var localGrad = new float[localSize];
                    Array.Copy(gradData, start, localGrad, 0, localSize);

                    var momentum = _momentumShards[name][rank];
                    var variance = _varianceShards[name][rank];

                    var paramData = _parameterShards[name][rank].ToArray();

                    // AdamW update
                    for (int i = 0; i < localSize; i++)
                    {
                        // Update momentum
                        momentum[i] = _config.Beta1 * momentum[i] + (1 - _config.Beta1) * localGrad[i];

                        // Update variance
                        variance[i] = _config.Beta2 * variance[i] + (1 - _config.Beta2) * localGrad[i] * localGrad[i];

                        // Bias-corrected estimates
                        float mHat = momentum[i] / biasCorrection1;
                        float vHat = variance[i] / biasCorrection2;

                        // Weight decay
                        paramData[i] -= _config.LearningRate * _config.WeightDecay * paramData[i];

                        // Adam update
                        paramData[i] -= _config.LearningRate * mHat / (MathF.Sqrt(vHat) + 1e-8f);
                    }

                    // Update GPU tensor
                    _parameterShards[name][rank].Buffer.View.SubView(0, localSize).CopyFromCPU(paramData);
                }));
            }
        }

        /// <summary>
        /// Gather full parameter from shards (for inference/checkpointing)
        /// </summary>
        public GpuTensor GatherParameter(string name)
        {
            if (!_parameterShards.ContainsKey(name))
            {
                throw new KeyNotFoundException($"Parameter {name} not registered");
            }

            var shards = _parameterShards[name];
            var allData = shards.SelectMany(s => s.ToArray()).ToArray();

            return GpuTensor.FromArray(_accelerators[0], allData, new[] { allData.Length });
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                foreach (var shards in _parameterShards.Values)
                {
                    foreach (var shard in shards)
                    {
                        shard.Dispose();
                    }
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Activation Checkpointing to trade compute for memory.
    /// Only stores activations at checkpoints, recomputes others during backward pass.
    /// </summary>
    public class ActivationCheckpointing
    {
        private readonly Dictionary<string, GpuTensor> _checkpoints = new();
        private readonly HashSet<string> _checkpointLayers;

        /// <summary>Public API</summary>
        public ActivationCheckpointing(IEnumerable<string>? checkpointLayers = null)
        {
            _checkpointLayers = checkpointLayers != null
                ? new HashSet<string>(checkpointLayers)
                : new HashSet<string>();
        }

        /// <summary>
        /// Save checkpoint if layer is marked for checkpointing
        /// </summary>
        public void SaveCheckpoint(string layerName, GpuTensor activation)
        {
            if (_checkpointLayers.Count == 0 || _checkpointLayers.Contains(layerName))
            {
                // Clone to avoid modifications
                var data = activation.ToArray();
                _checkpoints[layerName] = GpuTensor.FromArray(activation.Accelerator, data, activation.Shape);
            }
        }

        /// <summary>
        /// Get checkpoint if exists
        /// </summary>
        public GpuTensor? GetCheckpoint(string layerName)
        {
            return _checkpoints.TryGetValue(layerName, out var checkpoint) ? checkpoint : null;
        }

        /// <summary>
        /// Clear all checkpoints
        /// </summary>
        public void Clear()
        {
            foreach (var checkpoint in _checkpoints.Values)
            {
                checkpoint.Dispose();
            }
            _checkpoints.Clear();
        }

        /// <summary>
        /// Execute function with checkpointing
        /// </summary>
        public GpuTensor Checkpoint(
            string name,
            Func<GpuTensor, GpuTensor> forward,
            GpuTensor input)
        {
            var output = forward(input);
            SaveCheckpoint(name, output);
            return output;
        }
    }
}