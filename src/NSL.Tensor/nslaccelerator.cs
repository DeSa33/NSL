using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using System.Threading.Tasks;

// GPU backend - enabled via HAS_GPU define in .csproj when NSL.GPU is referenced
#if HAS_GPU
using NSL.GPU;
#endif

namespace NSL.Tensor
{
    /// NSL Runtime Accelerator - Unified CPU/GPU acceleration layer custom-built for NSL.
    /// Provides automatic device selection, kernel fusion, memory optimization, and auto-tuning.
    /// </summary>
    public sealed class NSLAccelerator : IDisposable
    {
        private static readonly Lazy<NSLAccelerator> _instance = new(() => new NSLAccelerator());
        /// <summary>Public API</summary>
        public static NSLAccelerator Instance => _instance.Value;

        private readonly NSLCpuBackend _cpu;
#if HAS_GPU
        private readonly NSLGpuBackend? _gpu;
#endif
        private readonly NSLMemoryManager _memory;
        private readonly NSLKernelCache _kernelCache;
        private readonly NSLProfiler _profiler;
        private readonly AcceleratorConfig _config;
        private bool _disposed;

#if HAS_GPU
        /// <summary>Public API</summary>
        public bool HasGpu => _gpu != null;
#else
        /// <summary>Public API</summary>
        public bool HasGpu => false;
#endif
        /// <summary>Public API</summary>
        public NSLCpuBackend Cpu => _cpu;
#if HAS_GPU
        /// <summary>Public API</summary>
        public NSLGpuBackend? Gpu => _gpu;
#endif
        /// <summary>Public API</summary>
        public NSLMemoryManager Memory => _memory;
        /// <summary>Public API</summary>
        public NSLProfiler Profiler => _profiler;

        /// <summary>Public API</summary>
        public NSLAccelerator(AcceleratorConfig? config = null)
        {
            _config = config ?? AcceleratorConfig.Default;
            _cpu = new NSLCpuBackend(_config);
            _memory = new NSLMemoryManager(_config);
            _kernelCache = new NSLKernelCache();
            _profiler = new NSLProfiler();

#if HAS_GPU
            try
            {
                _gpu = new NSLGpuBackend(_config);
            }
            catch
            {
                _gpu = null;
            }
#endif
        }

        /// Execute operation on optimal device based on tensor size and operation type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Execute(NSLOperation op, params Tensor[] inputs)
        {
            var device = SelectDevice(op, inputs);
            _profiler.StartOp(op.ToString());

            try
            {
                return device switch
                {
                    ComputeDevice.Cpu => ExecuteCpu(op, inputs),
                    ComputeDevice.Gpu => ExecuteGpu(op, inputs),
                    _ => ExecuteCpu(op, inputs)
                };
            }
            finally
            {
                _profiler.EndOp(op.ToString());
            }
        }

        private ComputeDevice SelectDevice(NSLOperation op, Tensor[] inputs)
        {
            if (!HasGpu) return ComputeDevice.Cpu;

            long totalElements = inputs.Sum(t => t.Data.LongLength);

            // GPU is better for large operations
            return op switch
            {
                NSLOperation.MatMul when totalElements > 4096 => ComputeDevice.Gpu,
                NSLOperation.Conv2d => ComputeDevice.Gpu,
                NSLOperation.Attention => ComputeDevice.Gpu,
                NSLOperation.Softmax when totalElements > 8192 => ComputeDevice.Gpu,
                NSLOperation.BatchNorm when totalElements > 4096 => ComputeDevice.Gpu,
                NSLOperation.LayerNorm when totalElements > 4096 => ComputeDevice.Gpu,
                _ when totalElements > _config.GpuThreshold => ComputeDevice.Gpu,
                _ => ComputeDevice.Cpu
            };
        }

        private Tensor ExecuteCpu(NSLOperation op, Tensor[] inputs)
        {
            return op switch
            {
                NSLOperation.Add => _cpu.Add(inputs[0], inputs[1]),
                NSLOperation.Sub => _cpu.Sub(inputs[0], inputs[1]),
                NSLOperation.Mul => _cpu.Mul(inputs[0], inputs[1]),
                NSLOperation.Div => _cpu.Div(inputs[0], inputs[1]),
                NSLOperation.MatMul => _cpu.MatMul(inputs[0], inputs[1]),
                NSLOperation.Transpose => _cpu.Transpose(inputs[0]),
                NSLOperation.ReLU => _cpu.ReLU(inputs[0]),
                NSLOperation.Sigmoid => _cpu.Sigmoid(inputs[0]),
                NSLOperation.Tanh => _cpu.Tanh(inputs[0]),
                NSLOperation.Softmax => _cpu.Softmax(inputs[0]),
                NSLOperation.LayerNorm => _cpu.LayerNorm(inputs[0], inputs.Length > 1 ? inputs[1] : null, inputs.Length > 2 ? inputs[2] : null),
                NSLOperation.Sum => _cpu.Sum(inputs[0]),
                NSLOperation.Mean => _cpu.Mean(inputs[0]),
                NSLOperation.Max => _cpu.Max(inputs[0]),
                NSLOperation.Min => _cpu.Min(inputs[0]),
                NSLOperation.Sqrt => _cpu.Sqrt(inputs[0]),
                NSLOperation.Exp => _cpu.Exp(inputs[0]),
                NSLOperation.Log => _cpu.Log(inputs[0]),
                NSLOperation.Pow => _cpu.Pow(inputs[0], inputs[1]),
                _ => throw new NotSupportedException($"Operation {op} not supported on CPU")
            };
        }

#if HAS_GPU
        private Tensor ExecuteGpu(NSLOperation op, Tensor[] inputs)
        {
            if (_gpu == null) return ExecuteCpu(op, inputs);

            return op switch
            {
                NSLOperation.Add => _gpu.Add(inputs[0], inputs[1]),
                NSLOperation.Sub => _gpu.Sub(inputs[0], inputs[1]),
                NSLOperation.Mul => _gpu.Mul(inputs[0], inputs[1]),
                NSLOperation.Div => _gpu.Div(inputs[0], inputs[1]),
                NSLOperation.MatMul => _gpu.MatMul(inputs[0], inputs[1]),
                NSLOperation.Conv2d => _gpu.Conv2d(inputs[0], inputs[1], inputs.Length > 2 ? inputs[2] : null),
                NSLOperation.Attention => _gpu.Attention(inputs[0], inputs[1], inputs[2]),
                NSLOperation.ReLU => _gpu.ReLU(inputs[0]),
                NSLOperation.Sigmoid => _gpu.Sigmoid(inputs[0]),
                NSLOperation.Tanh => _gpu.Tanh(inputs[0]),
                NSLOperation.Softmax => _gpu.Softmax(inputs[0]),
                NSLOperation.LayerNorm => _gpu.LayerNorm(inputs[0], inputs.Length > 1 ? inputs[1] : null, inputs.Length > 2 ? inputs[2] : null),
                NSLOperation.BatchNorm => _gpu.BatchNorm(inputs[0], inputs.Length > 1 ? inputs[1] : null, inputs.Length > 2 ? inputs[2] : null),
                _ => ExecuteCpu(op, inputs) // Fallback to CPU
            };
        }
#else
        private Tensor ExecuteGpu(NSLOperation op, Tensor[] inputs) => ExecuteCpu(op, inputs);
#endif

        /// Fused operation execution for improved performance.
        /// </summary>
        public Tensor ExecuteFused(FusedOperation fusion, params Tensor[] inputs)
        {
            _profiler.StartOp($"Fused:{fusion.Name}");
            try
            {
#if HAS_GPU
                return HasGpu && fusion.PreferGpu
                    ? _gpu!.ExecuteFused(fusion, inputs)
                    : _cpu.ExecuteFused(fusion, inputs);
#else
                return _cpu.ExecuteFused(fusion, inputs);
#endif
            }
            finally
            {
                _profiler.EndOp($"Fused:{fusion.Name}");
            }
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

#if HAS_GPU
            _gpu?.Dispose();
#endif
            _memory.Dispose();
            _kernelCache.Clear();
        }
    }

    /// Accelerator configuration.
    /// </summary>
    public class AcceleratorConfig
    {
        /// <summary>Public API</summary>
        public long GpuThreshold { get; set; } = 16384;
        /// <summary>Public API</summary>
        public int NumThreads { get; set; } = Environment.ProcessorCount;
        /// <summary>Public API</summary>
        public int CacheBlockSize { get; set; } = 64;
        /// <summary>Public API</summary>
        public long MemoryPoolSizeBytes { get; set; } = 1024L * 1024L * 1024L; // 1GB
        /// <summary>Public API</summary>
        public bool EnableProfiling { get; set; } = false;
        /// <summary>Public API</summary>
        public bool EnableAutoTuning { get; set; } = true;
        /// <summary>Public API</summary>
        public bool EnableKernelFusion { get; set; } = true;

        /// <summary>Public API</summary>
        public static AcceleratorConfig Default => new();

        /// <summary>Public API</summary>
        public static AcceleratorConfig HighPerformance => new()
        {
            GpuThreshold = 4096,
            MemoryPoolSizeBytes = 4L * 1024L * 1024L * 1024L,
            EnableProfiling = true,
            EnableAutoTuning = true,
            EnableKernelFusion = true
        };

        /// <summary>Public API</summary>
        public static AcceleratorConfig LowMemory => new()
        {
            GpuThreshold = 65536,
            MemoryPoolSizeBytes = 256L * 1024L * 1024L,
            EnableKernelFusion = false
        };
    }

    /// <summary>NSL accelerated operations.</summary>
    public enum NSLOperation
    {
        Add,
        Sub,
        Mul,
        Div,
        MatMul,
        Conv2d,
        MaxPool2d,
        AvgPool2d,
        Attention,
        FlashAttention,
        ReLU,
        LeakyReLU,
        GELU,
        SiLU,
        Sigmoid,
        Tanh,
        Softmax,
        LayerNorm,
        BatchNorm,
        GroupNorm,
        RMSNorm,
        Transpose,
        Reshape,
        Permute,
        Concat,
        Split,
        Sum,
        Mean,
        Max,
        Min,
        Argmax,
        Argmin,
        Sqrt,
        Exp,
        Log,
        Pow,
        Abs,
        Neg,
        Embedding,
        Linear,
        Dropout
    }

    /// <summary>Public API</summary>
    public enum ComputeDevice
    {
        Cpu,
        Gpu,
        Auto
    }

    /// NSL CPU Backend - Optimized SIMD operations for CPU execution.
    /// </summary>
    public sealed class NSLCpuBackend
    {
        private static readonly bool HasAvx512 = Avx512F.IsSupported;
        private static readonly bool HasAvx2 = Avx2.IsSupported;
        private static readonly bool HasFma = Fma.IsSupported;
        private static readonly bool HasSse = Sse.IsSupported;

        private readonly AcceleratorConfig _config;
        private readonly int _numThreads;

        /// <summary>Public API</summary>
        public NSLCpuBackend(AcceleratorConfig config)
        {
            _config = config;
            _numThreads = config.NumThreads;
        }

        /// <summary>API member</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Add(Tensor a, Tensor b)
        {
            var result = new double[a.Data.Length];
            VectorizedBinaryOp(a.Data, b.Data, result, (x, y) => x + y, Vector256.Add);
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>API member</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Sub(Tensor a, Tensor b)
        {
            var result = new double[a.Data.Length];
            VectorizedBinaryOp(a.Data, b.Data, result, (x, y) => x - y, Vector256.Subtract);
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>API member</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Mul(Tensor a, Tensor b)
        {
            var result = new double[a.Data.Length];
            VectorizedBinaryOp(a.Data, b.Data, result, (x, y) => x * y, Vector256.Multiply);
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>API member</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Div(Tensor a, Tensor b)
        {
            var result = new double[a.Data.Length];
            VectorizedBinaryOp(a.Data, b.Data, result, (x, y) => x / y, Vector256.Divide);
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor MatMul(Tensor a, Tensor b)
        {
            int m = (int)a.Shape[0];
            int k = a.Shape.Length > 1 ? (int)a.Shape[1] : 1;
            int n = b.Shape.Length > 1 ? (int)b.Shape[1] : 1;

            var result = new double[m * n];

            // Cache-blocked matrix multiplication with SIMD
            int blockSize = _config.CacheBlockSize;

            Parallel.For(0, (m + blockSize - 1) / blockSize, bi =>
            {
                int iStart = bi * blockSize;
                int iEnd = Math.Min(iStart + blockSize, m);

                for (int bk = 0; bk < k; bk += blockSize)
                {
                    int kEnd = Math.Min(bk + blockSize, k);

                    for (int bj = 0; bj < n; bj += blockSize)
                    {
                        int jEnd = Math.Min(bj + blockSize, n);

                        // Inner kernel with SIMD
                        for (int i = iStart; i < iEnd; i++)
                        {
                            for (int kk = bk; kk < kEnd; kk++)
                            {
                                double aVal = a.Data[i * k + kk];

                                if (HasAvx2)
                                {
                                    var aVec = Vector256.Create(aVal);
                                    int j = bj;

                                    for (; j <= jEnd - 4; j += 4)
                                    {
                                        var bVec = Vector256.Create(
                                            b.Data[kk * n + j],
                                            b.Data[kk * n + j + 1],
                                            b.Data[kk * n + j + 2],
                                            b.Data[kk * n + j + 3]);

                                        var rVec = Vector256.Create(
                                            result[i * n + j],
                                            result[i * n + j + 1],
                                            result[i * n + j + 2],
                                            result[i * n + j + 3]);

                                        if (HasFma)
                                        {
                                            rVec = Fma.MultiplyAdd(aVec, bVec, rVec);
                                        }
                                        else
                                        {
                                            rVec = Vector256.Add(rVec, Vector256.Multiply(aVec, bVec));
                                        }

                                        result[i * n + j] = rVec.GetElement(0);
                                        result[i * n + j + 1] = rVec.GetElement(1);
                                        result[i * n + j + 2] = rVec.GetElement(2);
                                        result[i * n + j + 3] = rVec.GetElement(3);
                                    }

                                    for (; j < jEnd; j++)
                                    {
                                        result[i * n + j] += aVal * b.Data[kk * n + j];
                                    }
                                }
                                else
                                {
                                    for (int j = bj; j < jEnd; j++)
                                    {
                                        result[i * n + j] += aVal * b.Data[kk * n + j];
                                    }
                                }
                            }
                        }
                    }
                }
            });

            return new Tensor(result, new long[] { m, n }, false);
        }

        /// <summary>Public API</summary>
        public Tensor Transpose(Tensor a)
        {
            if (a.Shape.Length != 2)
                throw new ArgumentException("Transpose requires 2D tensor");

            int rows = (int)a.Shape[0];
            int cols = (int)a.Shape[1];
            var result = new double[rows * cols];

            // Cache-oblivious transpose with blocking
            int blockSize = 32;

            Parallel.For(0, (rows + blockSize - 1) / blockSize, bi =>
            {
                int iStart = bi * blockSize;
                int iEnd = Math.Min(iStart + blockSize, rows);

                for (int bj = 0; bj < cols; bj += blockSize)
                {
                    int jEnd = Math.Min(bj + blockSize, cols);

                    for (int i = iStart; i < iEnd; i++)
                    {
                        for (int j = bj; j < jEnd; j++)
                        {
                            result[j * rows + i] = a.Data[i * cols + j];
                        }
                    }
                }
            });

            return new Tensor(result, new long[] { cols, rows }, false);
        }

        /// <summary>API member</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor ReLU(Tensor a)
        {
            var result = new double[a.Data.Length];
            var zero = Vector256<double>.Zero;

            int i = 0;
            if (HasAvx2)
            {
                for (; i <= a.Data.Length - 4; i += 4)
                {
                    var vec = Vector256.Create(a.Data[i], a.Data[i + 1], a.Data[i + 2], a.Data[i + 3]);
                    var relu = Vector256.Max(vec, zero);

                    result[i] = relu.GetElement(0);
                    result[i + 1] = relu.GetElement(1);
                    result[i + 2] = relu.GetElement(2);
                    result[i + 3] = relu.GetElement(3);
                }
            }

            for (; i < a.Data.Length; i++)
            {
                result[i] = Math.Max(0, a.Data[i]);
            }

            return new Tensor(result, a.Shape, false);
        }

        /// <summary>API member</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Sigmoid(Tensor a)
        {
            var result = new double[a.Data.Length];

            if (a.Data.Length > 4096)
            {
                Parallel.For(0, a.Data.Length, i =>
                {
                    result[i] = 1.0 / (1.0 + Math.Exp(-a.Data[i]));
                });
            }
            else
            {
                for (int i = 0; i < a.Data.Length; i++)
                {
                    result[i] = 1.0 / (1.0 + Math.Exp(-a.Data[i]));
                }
            }

            return new Tensor(result, a.Shape, false);
        }

        /// <summary>API member</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Tensor Tanh(Tensor a)
        {
            var result = new double[a.Data.Length];

            if (a.Data.Length > 4096)
            {
                Parallel.For(0, a.Data.Length, i =>
                {
                    result[i] = Math.Tanh(a.Data[i]);
                });
            }
            else
            {
                for (int i = 0; i < a.Data.Length; i++)
                {
                    result[i] = Math.Tanh(a.Data[i]);
                }
            }

            return new Tensor(result, a.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor Softmax(Tensor a, int axis = -1)
        {
            if (axis < 0) axis = a.Shape.Length + axis;

            var result = new double[a.Data.Length];
            int outerSize = 1, axisSize = (int)a.Shape[axis], innerSize = 1;

            for (int i = 0; i < axis; i++) outerSize *= (int)a.Shape[i];
            for (int i = axis + 1; i < a.Shape.Length; i++) innerSize *= (int)a.Shape[i];

            Parallel.For(0, outerSize, outer =>
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    // Find max for numerical stability
                    double maxVal = double.NegativeInfinity;
                    for (int j = 0; j < axisSize; j++)
                    {
                        int idx = (outer * axisSize + j) * innerSize + inner;
                        if (a.Data[idx] > maxVal) maxVal = a.Data[idx];
                    }

                    // Compute exp and sum
                    double sum = 0;
                    for (int j = 0; j < axisSize; j++)
                    {
                        int idx = (outer * axisSize + j) * innerSize + inner;
                        result[idx] = Math.Exp(a.Data[idx] - maxVal);
                        sum += result[idx];
                    }

                    // Normalize
                    double invSum = 1.0 / sum;
                    for (int j = 0; j < axisSize; j++)
                    {
                        int idx = (outer * axisSize + j) * innerSize + inner;
                        result[idx] *= invSum;
                    }
                }
            });

            return new Tensor(result, a.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor LayerNorm(Tensor input, Tensor? weight = null, Tensor? bias = null, double eps = 1e-5)
        {
            int lastDim = (int)input.Shape[^1];
            int numVectors = input.Data.Length / lastDim;
            var result = new double[input.Data.Length];

            Parallel.For(0, numVectors, v =>
            {
                int offset = v * lastDim;

                // Compute mean
                double mean = 0;
                for (int i = 0; i < lastDim; i++)
                {
                    mean += input.Data[offset + i];
                }
                mean /= lastDim;

                // Compute variance
                double variance = 0;
                for (int i = 0; i < lastDim; i++)
                {
                    double diff = input.Data[offset + i] - mean;
                    variance += diff * diff;
                }
                variance /= lastDim;

                double invStd = 1.0 / Math.Sqrt(variance + eps);

                // Normalize and apply affine transform
                for (int i = 0; i < lastDim; i++)
                {
                    double normalized = (input.Data[offset + i] - mean) * invStd;

                    if (weight != null)
                        normalized *= weight.Data[i];
                    if (bias != null)
                        normalized += bias.Data[i];

                    result[offset + i] = normalized;
                }
            });

            return new Tensor(result, input.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor Sum(Tensor a) => new Tensor(VectorizedSum(a.Data), false);
        /// <summary>Public API</summary>
        public Tensor Mean(Tensor a) => new Tensor(VectorizedSum(a.Data) / a.Data.Length, false);
        /// <summary>Public API</summary>
        public Tensor Max(Tensor a) => new Tensor(VectorizedMax(a.Data), false);
        /// <summary>Public API</summary>
        public Tensor Min(Tensor a) => new Tensor(VectorizedMin(a.Data), false);

        /// <summary>Public API</summary>
        public Tensor Sqrt(Tensor a)
        {
            var result = new double[a.Data.Length];
            VectorizedUnaryOp(a.Data, result, Math.Sqrt);
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor Exp(Tensor a)
        {
            var result = new double[a.Data.Length];
            VectorizedUnaryOp(a.Data, result, Math.Exp);
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor Log(Tensor a)
        {
            var result = new double[a.Data.Length];
            VectorizedUnaryOp(a.Data, result, Math.Log);
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor Pow(Tensor a, Tensor b)
        {
            var result = new double[a.Data.Length];
            for (int i = 0; i < a.Data.Length; i++)
            {
                result[i] = Math.Pow(a.Data[i], b.Data.Length == 1 ? b.Data[0] : b.Data[i]);
            }
            return new Tensor(result, a.Shape, false);
        }

        /// <summary>Public API</summary>
        public Tensor ExecuteFused(FusedOperation fusion, Tensor[] inputs)
        {
            // Execute fused operations sequentially on CPU
            var current = inputs[0];
            int inputIdx = 1;

            foreach (var op in fusion.Operations)
            {
                switch (op)
                {
                    case NSLOperation.Add:
                        current = Add(current, inputs[inputIdx++]);
                        break;
                    case NSLOperation.Mul:
                        current = Mul(current, inputs[inputIdx++]);
                        break;
                    case NSLOperation.ReLU:
                        current = ReLU(current);
                        break;
                    case NSLOperation.Sigmoid:
                        current = Sigmoid(current);
                        break;
                    // Add more as needed
                    default:
                        throw new NotSupportedException($"Fused operation {op} not supported");
                }
            }

            return current;
        }

        #region Vectorized Helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void VectorizedBinaryOp(double[] a, double[] b, double[] result,
            Func<double, double, double> scalarOp,
            Func<Vector256<double>, Vector256<double>, Vector256<double>> vectorOp)
        {
            int i = 0;

            if (HasAvx2 && a.Length >= 4)
            {
                for (; i <= a.Length - 4; i += 4)
                {
                    var va = Vector256.Create(a[i], a[i + 1], a[i + 2], a[i + 3]);
                    var vb = Vector256.Create(b.Length == 1 ? b[0] : b[i],
                                               b.Length == 1 ? b[0] : b[i + 1],
                                               b.Length == 1 ? b[0] : b[i + 2],
                                               b.Length == 1 ? b[0] : b[i + 3]);
                    var vr = vectorOp(va, vb);

                    result[i] = vr.GetElement(0);
                    result[i + 1] = vr.GetElement(1);
                    result[i + 2] = vr.GetElement(2);
                    result[i + 3] = vr.GetElement(3);
                }
            }

            for (; i < a.Length; i++)
            {
                result[i] = scalarOp(a[i], b.Length == 1 ? b[0] : b[i]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void VectorizedUnaryOp(double[] a, double[] result, Func<double, double> op)
        {
            if (a.Length > 4096)
            {
                Parallel.For(0, a.Length, i =>
                {
                    result[i] = op(a[i]);
                });
            }
            else
            {
                for (int i = 0; i < a.Length; i++)
                {
                    result[i] = op(a[i]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double VectorizedSum(double[] data)
        {
            double sum = 0;
            int i = 0;

            if (HasAvx2 && data.Length >= 4)
            {
                var vsum = Vector256<double>.Zero;

                for (; i <= data.Length - 4; i += 4)
                {
                    var v = Vector256.Create(data[i], data[i + 1], data[i + 2], data[i + 3]);
                    vsum = Vector256.Add(vsum, v);
                }

                sum = vsum.GetElement(0) + vsum.GetElement(1) + vsum.GetElement(2) + vsum.GetElement(3);
            }

            for (; i < data.Length; i++)
            {
                sum += data[i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double VectorizedMax(double[] data)
        {
            double max = double.NegativeInfinity;

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] > max) max = data[i];
            }

            return max;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double VectorizedMin(double[] data)
        {
            double min = double.PositiveInfinity;

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] < min) min = data[i];
            }

            return min;
        }

        #endregion
    }

#if HAS_GPU
    /// NSL GPU Backend - GPU-accelerated operations wrapper.
    /// </summary>
    public sealed class NSLGpuBackend : IDisposable
    {
        private readonly GpuAccelerator _gpu;
        private readonly AcceleratorConfig _config;
        private bool _disposed;

        /// <summary>Public API</summary>
        public NSLGpuBackend(AcceleratorConfig config)
        {
            _config = config;
            _gpu = new GpuAccelerator();
        }

        /// <summary>Public API</summary>
        public Tensor Add(Tensor a, Tensor b) => ExecuteElementwise(a, b, (ga, gb) => _gpu.Add(ga, gb));
        /// <summary>Public API</summary>
        public Tensor Sub(Tensor a, Tensor b) => ExecuteElementwise(a, b, (ga, gb) => _gpu.Sub(ga, gb));
        /// <summary>Public API</summary>
        public Tensor Mul(Tensor a, Tensor b) => ExecuteElementwise(a, b, (ga, gb) => _gpu.Mul(ga, gb));
        /// <summary>Public API</summary>
        public Tensor Div(Tensor a, Tensor b) => ExecuteElementwise(a, b, (ga, gb) => _gpu.Div(ga, gb));

        /// <summary>Public API</summary>
        public Tensor MatMul(Tensor a, Tensor b)
        {
            var ga = ToGpu(a);
            var gb = ToGpu(b);
            var result = _gpu.MatMul(ga, gb);
            return FromGpu(result, new[] { a.Shape[0], b.Shape.Length > 1 ? b.Shape[1] : 1 });
        }

        /// <summary>Public API</summary>
        public Tensor Conv2d(Tensor input, Tensor kernel, Tensor? bias)
        {
            var gInput = ToGpu(input);
            var gKernel = ToGpu(kernel);
            var gBias = bias != null ? ToGpu(bias) : null;

            var result = _gpu.Conv2d(gInput, gKernel, 1, 0);

            long outH = input.Shape[2] - kernel.Shape[2] + 1;
            long outW = input.Shape[3] - kernel.Shape[3] + 1;

            return FromGpu(result, new[] { input.Shape[0], kernel.Shape[0], outH, outW });
        }

        /// <summary>Public API</summary>
        public Tensor Attention(Tensor query, Tensor key, Tensor value)
        {
            var gQ = ToGpu(query);
            var gK = ToGpu(key);
            var gV = ToGpu(value);

            // Use ScaledDotProductAttention
            var result = _gpu.ScaledDotProductAttention(gQ, gK, gV);

            return FromGpu(result, query.Shape);
        }

        /// <summary>Public API</summary>
        public Tensor ReLU(Tensor a) => ExecuteUnary(a, ga => _gpu.ReLU(ga));
        /// <summary>Public API</summary>
        public Tensor Sigmoid(Tensor a) => ExecuteUnary(a, ga => _gpu.Sigmoid(ga));
        /// <summary>Public API</summary>
        public Tensor Tanh(Tensor a) => ExecuteUnary(a, ga => _gpu.Tanh(ga));

        /// <summary>Public API</summary>
        public Tensor Softmax(Tensor a)
        {
            var ga = ToGpu(a);
            var result = _gpu.Softmax(ga);
            return FromGpu(result, a.Shape);
        }

        /// <summary>Public API</summary>
        public Tensor LayerNorm(Tensor input, Tensor? weight, Tensor? bias)
        {
            var gInput = ToGpu(input);

            // Create default gamma (ones) and beta (zeros) if not provided
            int lastDim = (int)input.Shape[^1];
            var gWeight = weight != null ? ToGpu(weight) : _gpu.Ones(lastDim);
            var gBias = bias != null ? ToGpu(bias) : _gpu.Zeros(lastDim);

            var result = _gpu.LayerNorm(gInput, gWeight, gBias);

            return FromGpu(result, input.Shape);
        }

        /// <summary>Public API</summary>
        public Tensor BatchNorm(Tensor input, Tensor? weight, Tensor? bias)
        {
            var gInput = ToGpu(input);
            int channels = (int)input.Shape[1];

            // Create default parameters if not provided
            var gWeight = weight != null ? ToGpu(weight) : _gpu.Ones(channels);
            var gBias = bias != null ? ToGpu(bias) : _gpu.Zeros(channels);
            var gMean = _gpu.Zeros(channels);
            var gVar = _gpu.Ones(channels);

            var result = _gpu.BatchNorm(gInput, gWeight, gBias, gMean, gVar, training: false);

            return FromGpu(result, input.Shape);
        }

        /// <summary>Public API</summary>
        public Tensor ExecuteFused(FusedOperation fusion, Tensor[] inputs)
        {
            // For GPU, we can potentially generate fused CUDA kernels
            // For now, execute sequentially with GPU tensors staying on device
            var gpuInputs = inputs.Select(ToGpu).ToArray();
            var current = gpuInputs[0];
            int inputIdx = 1;

            foreach (var op in fusion.Operations)
            {
                switch (op)
                {
                    case NSLOperation.Add:
                        current = _gpu.Add(current, gpuInputs[inputIdx++]);
                        break;
                    case NSLOperation.Mul:
                        current = _gpu.Mul(current, gpuInputs[inputIdx++]);
                        break;
                    case NSLOperation.ReLU:
                        current = _gpu.ReLU(current);
                        break;
                    case NSLOperation.Sigmoid:
                        current = _gpu.Sigmoid(current);
                        break;
                    default:
                        throw new NotSupportedException($"Fused GPU operation {op} not supported");
                }
            }

            return FromGpu(current, inputs[0].Shape);
        }

        private Tensor ExecuteElementwise(Tensor a, Tensor b, Func<GpuTensor, GpuTensor, GpuTensor> op)
        {
            var ga = ToGpu(a);
            var gb = ToGpu(b);
            var result = op(ga, gb);
            return FromGpu(result, a.Shape);
        }

        private Tensor ExecuteUnary(Tensor a, Func<GpuTensor, GpuTensor> op)
        {
            var ga = ToGpu(a);
            var result = op(ga);
            return FromGpu(result, a.Shape);
        }

        private GpuTensor ToGpu(Tensor t)
        {
            // Convert double[] to float[] and transfer to GPU
            var floatData = new float[t.Data.Length];
            for (int i = 0; i < t.Data.Length; i++)
                floatData[i] = (float)t.Data[i];

            // Convert long[] shape to int[]
            var intShape = t.Shape.Select(s => (int)s).ToArray();
            return _gpu.ToGpu(floatData, intShape);
        }

        private Tensor FromGpu(GpuTensor gt, long[] shape)
        {
            // Get data from GPU and convert back to double[]
            var (floatData, _) = _gpu.ToCpu(gt);
            var doubleData = new double[floatData.Length];
            for (int i = 0; i < floatData.Length; i++)
                doubleData[i] = floatData[i];

            return new Tensor(doubleData, shape, false);
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _gpu.Dispose();
        }
    }
#endif // HAS_GPU

    /// NSL Memory Manager - Efficient memory pooling and allocation.
    /// </summary>
    public sealed class NSLMemoryManager : IDisposable
    {
        private readonly AcceleratorConfig _config;
        private readonly ConcurrentDictionary<int, ArrayPool<double>> _pools;
        private readonly ConcurrentDictionary<long, GCHandle> _pinnedMemory;
        private long _totalAllocated;
        private bool _disposed;

        /// <summary>Public API</summary>
        public long TotalAllocated => _totalAllocated;

        /// <summary>Public API</summary>
        public NSLMemoryManager(AcceleratorConfig config)
        {
            _config = config;
            _pools = new ConcurrentDictionary<int, ArrayPool<double>>();
            _pinnedMemory = new ConcurrentDictionary<long, GCHandle>();
        }

        /// Rent an array from the pool.
        /// </summary>
        public double[] Rent(int size)
        {
            int bucket = GetBucket(size);
            var pool = _pools.GetOrAdd(bucket, _ => ArrayPool<double>.Create());
            Interlocked.Add(ref _totalAllocated, size * sizeof(double));
            return pool.Rent(size);
        }

        /// Return an array to the pool.
        /// </summary>
        public void Return(double[] array)
        {
            int bucket = GetBucket(array.Length);
            if (_pools.TryGetValue(bucket, out var pool))
            {
                Interlocked.Add(ref _totalAllocated, -array.Length * sizeof(double));
                pool.Return(array, clearArray: false);
            }
        }

        /// Allocate pinned memory for zero-copy GPU transfer.
        /// </summary>
        public unsafe double* AllocatePinned(int size)
        {
            var array = new double[size];
            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            var id = Interlocked.Increment(ref _totalAllocated);
            _pinnedMemory[id] = handle;
            return (double*)handle.AddrOfPinnedObject();
        }

        /// Free pinned memory.
        /// </summary>
        public void FreePinned(long id)
        {
            if (_pinnedMemory.TryRemove(id, out var handle))
            {
                handle.Free();
            }
        }

        private int GetBucket(int size)
        {
            // Round up to power of 2 for bucketing
            int bucket = 64;
            while (bucket < size)
                bucket *= 2;
            return bucket;
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            foreach (var handle in _pinnedMemory.Values)
            {
                if (handle.IsAllocated)
                    handle.Free();
            }
            _pinnedMemory.Clear();
        }
    }

    /// Kernel cache for compiled/optimized kernels.
    /// </summary>
    public sealed class NSLKernelCache
    {
        private readonly ConcurrentDictionary<string, object> _cache;

        /// <summary>Public API</summary>
        public NSLKernelCache()
        {
            _cache = new ConcurrentDictionary<string, object>();
        }

        /// <summary>Public API</summary>
        public T GetOrCreate<T>(string key, Func<T> factory)
        {
            return (T)_cache.GetOrAdd(key, _ => factory()!);
        }

        /// <summary>Public API</summary>
        public void Clear()
        {
            _cache.Clear();
        }
    }

    /// Fused operation definition for kernel fusion.
    /// </summary>
    public class FusedOperation
    {
        /// <summary>Public API</summary>
        public string Name { get; }
        /// <summary>Public API</summary>
        public NSLOperation[] Operations { get; }
        /// <summary>Public API</summary>
        public bool PreferGpu { get; }

        /// <summary>Public API</summary>
        public FusedOperation(string name, NSLOperation[] operations, bool preferGpu = true)
        {
            Name = name;
            Operations = operations;
            PreferGpu = preferGpu;
        }

        // Common fused operations
        /// <summary>Public API</summary>
        public static FusedOperation LinearReLU = new("LinearReLU",
            new[] { NSLOperation.MatMul, NSLOperation.Add, NSLOperation.ReLU });

        /// <summary>Public API</summary>
        public static FusedOperation LinearSigmoid = new("LinearSigmoid",
            new[] { NSLOperation.MatMul, NSLOperation.Add, NSLOperation.Sigmoid });

        /// <summary>Public API</summary>
        public static FusedOperation AddMul = new("AddMul",
            new[] { NSLOperation.Add, NSLOperation.Mul });

        /// <summary>Public API</summary>
        public static FusedOperation MulAdd = new("MulAdd",
            new[] { NSLOperation.Mul, NSLOperation.Add });
    }

    /// NSL Profiler for performance monitoring.
    /// </summary>
    public sealed class NSLProfiler
    {
        private readonly ConcurrentDictionary<string, ProfileEntry> _entries;
        private readonly Stopwatch _globalTimer;
        private readonly ThreadLocal<Dictionary<string, Stopwatch>> _threadTimers;
        private bool _enabled;

        /// <summary>Public API</summary>
        public bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        /// <summary>Public API</summary>
        public NSLProfiler()
        {
            _entries = new ConcurrentDictionary<string, ProfileEntry>();
            _globalTimer = Stopwatch.StartNew();
            _threadTimers = new ThreadLocal<Dictionary<string, Stopwatch>>(() => new Dictionary<string, Stopwatch>());
        }

        /// <summary>Public API</summary>
        public void StartOp(string name)
        {
            if (!_enabled) return;

            var timers = _threadTimers.Value!;
            if (!timers.ContainsKey(name))
                timers[name] = new Stopwatch();

            timers[name].Restart();
        }

        /// <summary>Public API</summary>
        public void EndOp(string name)
        {
            if (!_enabled) return;

            var timers = _threadTimers.Value!;
            if (timers.TryGetValue(name, out var timer))
            {
                timer.Stop();
                var entry = _entries.GetOrAdd(name, _ => new ProfileEntry());
                entry.AddSample(timer.Elapsed.TotalMilliseconds);
            }
        }

        /// <summary>Public API</summary>
        public ProfileReport GetReport()
        {
            return new ProfileReport
            {
                TotalTime = _globalTimer.Elapsed,
                Entries = _entries.ToDictionary(kv => kv.Key, kv => kv.Value.Clone())
            };
        }

        /// <summary>Public API</summary>
        public void Reset()
        {
            _entries.Clear();
            _globalTimer.Restart();
        }
    }

    /// <summary>Public API</summary>
    public class ProfileEntry
    {
        private long _count;
        private double _totalMs;
        private double _minMs = double.MaxValue;
        private double _maxMs;
        private readonly object _lock = new();

        /// <summary>Public API</summary>
        public long Count => _count;
        /// <summary>Public API</summary>
        public double TotalMs => _totalMs;
        /// <summary>Public API</summary>
        public double AvgMs => _count > 0 ? _totalMs / _count : 0;
        /// <summary>Public API</summary>
        public double MinMs => _minMs == double.MaxValue ? 0 : _minMs;
        /// <summary>Public API</summary>
        public double MaxMs => _maxMs;

        /// <summary>Public API</summary>
        public void AddSample(double ms)
        {
            lock (_lock)
            {
                _count++;
                _totalMs += ms;
                if (ms < _minMs) _minMs = ms;
                if (ms > _maxMs) _maxMs = ms;
            }
        }

        /// <summary>Public API</summary>
        public ProfileEntry Clone()
        {
            lock (_lock)
            {
                return new ProfileEntry
                {
                    _count = _count,
                    _totalMs = _totalMs,
                    _minMs = _minMs,
                    _maxMs = _maxMs
                };
            }
        }
    }

    /// <summary>Public API</summary>
    public class ProfileReport
    {
        /// <summary>Public API</summary>
        public TimeSpan TotalTime { get; set; }
        /// <summary>Public API</summary>
        public Dictionary<string, ProfileEntry> Entries { get; set; } = new();

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"NSL Profiler Report (Total: {TotalTime.TotalSeconds:F2}s)");
            sb.AppendLine(new string('-', 80));
            sb.AppendLine($"{"Operation",-30} {"Count",10} {"Total(ms)",12} {"Avg(ms)",10} {"Min(ms)",10} {"Max(ms)",10}");
            sb.AppendLine(new string('-', 80));

            foreach (var kv in Entries.OrderByDescending(e => e.Value.TotalMs))
            {
                var e = kv.Value;
                sb.AppendLine($"{kv.Key,-30} {e.Count,10} {e.TotalMs,12:F2} {e.AvgMs,10:F4} {e.MinMs,10:F4} {e.MaxMs,10:F4}");
            }

            return sb.ToString();
        }
    }

    /// NSL JIT Compiler for dynamic kernel generation.
    /// </summary>
    public sealed class NSLJitCompiler
    {
        private readonly NSLKernelCache _cache;

        /// <summary>Public API</summary>
        public NSLJitCompiler(NSLKernelCache cache)
        {
            _cache = cache;
        }

        /// Compile a computation graph into an optimized execution plan.
        /// </summary>
        public CompiledGraph Compile(ComputationGraph graph)
        {
            var key = graph.GetHash();

            return _cache.GetOrCreate(key, () =>
            {
                // Optimize the graph
                var optimized = OptimizeGraph(graph);

                // Generate execution plan
                return new CompiledGraph(optimized);
            });
        }

        private ComputationGraph OptimizeGraph(ComputationGraph graph)
        {
            // Apply optimizations:
            // 1. Constant folding
            // 2. Dead code elimination
            // 3. Kernel fusion
            // 4. Memory optimization

            var optimized = graph.Clone();

            // Fuse adjacent compatible operations
            FuseOperations(optimized);

            return optimized;
        }

        private void FuseOperations(ComputationGraph graph)
        {
            // Find sequences like MatMul+Add+ReLU and replace with fused op
            var nodes = graph.GetTopologicalOrder();
            var fusionCandidates = new List<(int start, int end, FusedOperation fusion)>();

            for (int i = 0; i < nodes.Count - 2; i++)
            {
                // Check for Linear+ReLU pattern
                if (nodes[i].Op == NSLOperation.MatMul &&
                    nodes[i + 1].Op == NSLOperation.Add &&
                    nodes[i + 2].Op == NSLOperation.ReLU)
                {
                    fusionCandidates.Add((i, i + 2, FusedOperation.LinearReLU));
                }
            }

            // Apply fusions in reverse order to preserve indices
            foreach (var (start, end, fusion) in fusionCandidates.OrderByDescending(f => f.start))
            {
                graph.FuseNodes(start, end, fusion);
            }
        }
    }

    /// Computation graph for lazy evaluation and optimization.
    /// </summary>
    public class ComputationGraph
    {
        private readonly List<ComputeNode> _nodes = new();

        /// <summary>Public API</summary>
        public void AddNode(ComputeNode node)
        {
            _nodes.Add(node);
        }

        /// <summary>Public API</summary>
        public List<ComputeNode> GetTopologicalOrder() => _nodes.ToList();

        /// <summary>Public API</summary>
        public string GetHash()
        {
            var sb = new System.Text.StringBuilder();
            foreach (var node in _nodes)
            {
                sb.Append($"{node.Op}:{string.Join(",", node.InputIds)};");
            }
            return sb.ToString();
        }

        /// <summary>Public API</summary>
        public ComputationGraph Clone()
        {
            var clone = new ComputationGraph();
            foreach (var node in _nodes)
            {
                clone.AddNode(node.Clone());
            }
            return clone;
        }

        /// <summary>Public API</summary>
        public void FuseNodes(int start, int end, FusedOperation fusion)
        {
            // Replace nodes [start, end] with a single fused node
            var fusedNode = new ComputeNode
            {
                Id = _nodes[start].Id,
                Op = NSLOperation.Linear, // Placeholder
                IsFused = true,
                FusedOperation = fusion,
                InputIds = _nodes[start].InputIds.Concat(
                    _nodes.Skip(start + 1).Take(end - start)
                        .SelectMany(n => n.InputIds.Skip(1))).ToArray()
            };

            _nodes.RemoveRange(start, end - start + 1);
            _nodes.Insert(start, fusedNode);
        }
    }

    /// <summary>Public API</summary>
    public class ComputeNode
    {
        /// <summary>Public API</summary>
        public int Id { get; set; }
        /// <summary>Public API</summary>
        public NSLOperation Op { get; set; }
        /// <summary>Public API</summary>
        public int[] InputIds { get; set; } = Array.Empty<int>();
        /// <summary>Public API</summary>
        public int[] Shape { get; set; } = Array.Empty<int>();
        /// <summary>Public API</summary>
        public bool IsFused { get; set; }
        /// <summary>Public API</summary>
        public FusedOperation? FusedOperation { get; set; }

        /// <summary>Public API</summary>
        public ComputeNode Clone()
        {
            return new ComputeNode
            {
                Id = Id,
                Op = Op,
                InputIds = (int[])InputIds.Clone(),
                Shape = (int[])Shape.Clone(),
                IsFused = IsFused,
                FusedOperation = FusedOperation
            };
        }
    }

    /// <summary>Public API</summary>
    public class CompiledGraph
    {
        private readonly ComputationGraph _graph;

        /// <summary>Public API</summary>
        public CompiledGraph(ComputationGraph graph)
        {
            _graph = graph;
        }

        /// <summary>Public API</summary>
        public Tensor Execute(Dictionary<int, Tensor> inputs)
        {
            var values = new Dictionary<int, Tensor>(inputs);
            var accelerator = NSLAccelerator.Instance;

            foreach (var node in _graph.GetTopologicalOrder())
            {
                var nodeInputs = node.InputIds.Select(id => values[id]).ToArray();

                Tensor result;
                if (node.IsFused && node.FusedOperation != null)
                {
                    result = accelerator.ExecuteFused(node.FusedOperation, nodeInputs);
                }
                else
                {
                    result = accelerator.Execute(node.Op, nodeInputs);
                }

                values[node.Id] = result;
            }

            return values[_graph.GetTopologicalOrder().Last().Id];
        }
    }

    /// Auto-tuner for finding optimal kernel parameters.
    /// </summary>
    public sealed class NSLAutoTuner
    {
        private readonly ConcurrentDictionary<string, TuneResult> _tuneCache;

        /// <summary>Public API</summary>
        public NSLAutoTuner()
        {
            _tuneCache = new ConcurrentDictionary<string, TuneResult>();
        }

        /// Auto-tune a kernel for given input shapes.
        /// </summary>
        public TuneResult Tune(string kernelName, int[] shape, Func<TuneConfig, double> benchmark)
        {
            var key = $"{kernelName}:{string.Join(",", shape)}";

            return _tuneCache.GetOrAdd(key, _ =>
            {
                var configs = GenerateConfigs(kernelName, shape);
                TuneConfig? bestConfig = null;
                double bestTime = double.MaxValue;

                foreach (var config in configs)
                {
                    // Warmup
                    benchmark(config);

                    // Benchmark
                    double totalTime = 0;
                    const int trials = 5;

                    for (int i = 0; i < trials; i++)
                    {
                        totalTime += benchmark(config);
                    }

                    double avgTime = totalTime / trials;

                    if (avgTime < bestTime)
                    {
                        bestTime = avgTime;
                        bestConfig = config;
                    }
                }

                return new TuneResult
                {
                    Config = bestConfig ?? new TuneConfig(),
                    TimeMs = bestTime
                };
            });
        }

        private IEnumerable<TuneConfig> GenerateConfigs(string kernelName, int[] shape)
        {
            // Generate search space based on kernel type
            var blockSizes = new[] { 16, 32, 64, 128, 256 };
            var tileMs = new[] { 16, 32, 64, 128 };
            var tileNs = new[] { 16, 32, 64, 128 };
            var tileKs = new[] { 8, 16, 32 };

            if (kernelName.Contains("MatMul"))
            {
                foreach (var bs in blockSizes)
                    foreach (var tm in tileMs)
                        foreach (var tn in tileNs)
                            foreach (var tk in tileKs)
                            {
                                yield return new TuneConfig
                                {
                                    BlockSize = bs,
                                    TileM = tm,
                                    TileN = tn,
                                    TileK = tk
                                };
                            }
            }
            else
            {
                foreach (var bs in blockSizes)
                {
                    yield return new TuneConfig { BlockSize = bs };
                }
            }
        }
    }

    /// <summary>Public API</summary>
    public class TuneConfig
    {
        /// <summary>Public API</summary>
        public int BlockSize { get; set; } = 64;
        /// <summary>Public API</summary>
        public int TileM { get; set; } = 32;
        /// <summary>Public API</summary>
        public int TileN { get; set; } = 32;
        /// <summary>Public API</summary>
        public int TileK { get; set; } = 16;
        /// <summary>Public API</summary>
        public int UnrollFactor { get; set; } = 4;
        /// <summary>Public API</summary>
        public bool UseSharedMemory { get; set; } = true;
    }

    /// <summary>Public API</summary>
    public class TuneResult
    {
        /// <summary>Public API</summary>
        public TuneConfig Config { get; set; } = new();
        /// <summary>Public API</summary>
        public double TimeMs { get; set; }
    }
}