using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace NSL.GPU
{
    /// <summary>
    /// GPU backend types supported by NSL
    /// </summary>
    public enum GpuBackend
    {
        Auto,
        CUDA,
        OpenCL,
        CPU
    }

    /// <summary>
    /// Information about an available GPU device
    /// </summary>
    public class GpuDeviceInfo
    {
        /// <summary>Device name</summary>
        public string Name { get; init; } = "";
        /// <summary>Device index</summary>
        public int Index { get; init; }
        /// <summary>GPU backend type</summary>
        public GpuBackend Backend { get; init; }
        /// <summary>Total memory in bytes</summary>
        public long TotalMemory { get; init; }
        /// <summary>Number of multiprocessors/compute units</summary>
        public int ComputeUnits { get; init; }
        /// <summary>Maximum threads per block/workgroup</summary>
        public int MaxThreadsPerGroup { get; init; }
        /// <summary>Warp/wavefront size</summary>
        public int WarpSize { get; init; }
        /// <summary>Whether the device is currently selected</summary>
        public bool IsSelected { get; init; }

        /// <summary>
        /// Returns string representation of the device
        /// </summary>
        public override string ToString() =>
            $"{Name} ({Backend}) - {TotalMemory / (1024 * 1024)}MB, {ComputeUnits} compute units";
    }

    /// <summary>
    /// Central GPU accelerator manager for NSL.
    /// Provides GPU tensor operations 100-1000x faster than CPU.
    ///
    /// Usage:
    /// <code>
    /// using var gpu = new GpuAccelerator();
    /// var gpuTensor = gpu.ToGpu(cpuTensor);
    /// var result = gpu.MatMul(gpuTensor, weights);
    /// var cpuResult = gpu.ToCpu(result);
    /// </code>
    /// </summary>
    public class GpuAccelerator : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;
        private readonly HighPerformanceKernels _hpKernels;
        private bool _disposed;

        /// <summary>
        /// The active GPU backend
        /// </summary>
        public GpuBackend ActiveBackend { get; }

        /// <summary>
        /// Device information for the selected accelerator
        /// </summary>
        public GpuDeviceInfo DeviceInfo { get; }

        /// <summary>
        /// Whether a CUDA-capable GPU is available
        /// </summary>
        public static bool IsCudaAvailable
        {
            get
            {
                try
                {
                    using var context = Context.CreateDefault();
                    return context.GetCudaDevices().Count > 0;
                }
                catch
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Whether an OpenCL-capable GPU is available
        /// </summary>
        public static bool IsOpenCLAvailable
        {
            get
            {
                try
                {
                    using var context = Context.CreateDefault();
                    return context.GetCLDevices().Count > 0;
                }
                catch
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Create a GPU accelerator with automatic backend selection
        /// </summary>
        public GpuAccelerator() : this(GpuBackend.Auto, 0) { }

        /// <summary>
        /// Create a GPU accelerator with specific backend
        /// </summary>
        /// <param name="backend">The GPU backend to use</param>
        /// <param name="deviceIndex">Device index (0 for first GPU)</param>
        public GpuAccelerator(GpuBackend backend, int deviceIndex = 0)
        {
            _context = Context.Create(builder => builder
                .Default()
                .EnableAlgorithms()
                .Optimize(OptimizationLevel.O2)
                .Math(MathMode.Default));

            (_accelerator, ActiveBackend) = CreateAccelerator(backend, deviceIndex);
            DeviceInfo = GetDeviceInfo(_accelerator, ActiveBackend, deviceIndex);
            _kernels = new GpuKernels(_accelerator);
            _hpKernels = new HighPerformanceKernels(_accelerator);
        }

        /// <summary>
        /// Access high-performance optimized kernels.
        /// These provide significant speedups for large operations:
        /// - Tiled MatMul with shared memory (2-5x faster)
        /// - Fused attention (FlashAttention-style, 4x less memory)
        /// - Warp-level reductions (no shared memory overhead)
        /// - Fused LayerNorm (single pass Welford algorithm)
        /// </summary>
        public HighPerformanceKernels HighPerformance => _hpKernels;

        /// <summary>
        /// Access the underlying ILGPU Accelerator for advanced operations.
        /// </summary>
        public Accelerator Accelerator => _accelerator;

        /// <summary>
        /// Access the GPU kernels for custom operations.
        /// </summary>
        public GpuKernels Kernels => _kernels;

        /// <summary>
        /// List all available GPU devices
        /// </summary>
        public static List<GpuDeviceInfo> ListDevices()
        {
            var devices = new List<GpuDeviceInfo>();

            try
            {
                using var context = Context.CreateDefault();

                // CUDA devices
                var cudaDevices = context.GetCudaDevices();
                for (int i = 0; i < cudaDevices.Count; i++)
                {
                    var device = cudaDevices[i];
                    devices.Add(new GpuDeviceInfo
                    {
                        Name = device.Name,
                        Index = i,
                        Backend = GpuBackend.CUDA,
                        TotalMemory = device.MemorySize,
                        ComputeUnits = device.NumMultiprocessors,
                        MaxThreadsPerGroup = device.MaxNumThreadsPerGroup,
                        WarpSize = device.WarpSize
                    });
                }

                // OpenCL devices
                var clDevices = context.GetCLDevices();
                for (int i = 0; i < clDevices.Count; i++)
                {
                    var device = clDevices[i];
                    devices.Add(new GpuDeviceInfo
                    {
                        Name = device.Name,
                        Index = i,
                        Backend = GpuBackend.OpenCL,
                        TotalMemory = device.MemorySize,
                        ComputeUnits = device.NumMultiprocessors,
                        MaxThreadsPerGroup = device.MaxNumThreadsPerGroup,
                        WarpSize = device.WarpSize
                    });
                }

                // CPU fallback
                var cpuDevices = context.GetCPUDevices();
                for (int i = 0; i < cpuDevices.Count; i++)
                {
                    var device = cpuDevices[i];
                    devices.Add(new GpuDeviceInfo
                    {
                        Name = device.Name,
                        Index = i,
                        Backend = GpuBackend.CPU,
                        ComputeUnits = Environment.ProcessorCount,
                        MaxThreadsPerGroup = device.MaxNumThreadsPerGroup,
                        WarpSize = device.WarpSize
                    });
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Warning: Failed to enumerate GPU devices: {ex.Message}");
            }

            return devices;
        }

        #region Tensor Transfer Operations

        /// <summary>
        /// Transfer a CPU float array to GPU memory
        /// </summary>
        /// <param name="data">The data to transfer</param>
        /// <param name="shape">The shape of the tensor</param>
        /// <returns>A GPU tensor handle</returns>
        public GpuTensor ToGpu(float[] data, params int[] shape)
        {
            ThrowIfDisposed();
            return GpuTensor.FromArray(_accelerator, data, shape);
        }

        /// <summary>
        /// Transfer a CPU double array to GPU memory (auto-converts to float)
        /// </summary>
        /// <param name="data">The data to transfer</param>
        /// <param name="shape">The shape of the tensor</param>
        /// <returns>A GPU tensor handle</returns>
        public GpuTensor ToGpu(double[] data, params int[] shape)
        {
            ThrowIfDisposed();
            return GpuTensor.FromDoubleArray(_accelerator, data, shape);
        }

        /// <summary>
        /// Transfer a GPU tensor back to CPU memory
        /// </summary>
        /// <param name="gpuTensor">The GPU tensor to transfer</param>
        /// <returns>A tuple of (data, shape)</returns>
        public (float[] Data, int[] Shape) ToCpu(GpuTensor gpuTensor)
        {
            ThrowIfDisposed();
            return (gpuTensor.ToArray(), gpuTensor.Shape);
        }

        /// <summary>
        /// Create a GPU tensor filled with zeros
        /// </summary>
        public GpuTensor Zeros(params int[] shape)
        {
            ThrowIfDisposed();
            return GpuTensor.Zeros(_accelerator, shape);
        }

        /// <summary>
        /// Create a GPU tensor filled with ones
        /// </summary>
        public GpuTensor Ones(params int[] shape)
        {
            ThrowIfDisposed();
            return GpuTensor.Ones(_accelerator, shape);
        }

        /// <summary>
        /// Create a GPU tensor with random values from uniform distribution [0, 1)
        /// </summary>
        public GpuTensor Random(params int[] shape)
        {
            ThrowIfDisposed();
            return GpuTensor.Random(_accelerator, shape);
        }

        /// <summary>
        /// Create a GPU tensor with random values from normal distribution
        /// </summary>
        public GpuTensor RandomNormal(float mean = 0f, float std = 1f, params int[] shape)
        {
            ThrowIfDisposed();
            return GpuTensor.RandomNormal(_accelerator, mean, std, shape);
        }

        #endregion

        #region Element-wise Operations

        /// <summary>
        /// Element-wise addition: C = A + B
        /// </summary>
        public GpuTensor Add(GpuTensor a, GpuTensor b)
        {
            ThrowIfDisposed();
            ValidateShapes(a, b, "addition");
            return _kernels.Add(a, b);
        }

        /// <summary>
        /// Element-wise subtraction: C = A - B
        /// </summary>
        public GpuTensor Sub(GpuTensor a, GpuTensor b)
        {
            ThrowIfDisposed();
            ValidateShapes(a, b, "subtraction");
            return _kernels.Sub(a, b);
        }

        /// <summary>
        /// Element-wise multiplication (Hadamard product): C = A * B
        /// </summary>
        public GpuTensor Mul(GpuTensor a, GpuTensor b)
        {
            ThrowIfDisposed();
            ValidateShapes(a, b, "multiplication");
            return _kernels.Mul(a, b);
        }

        /// <summary>
        /// Element-wise division: C = A / B
        /// </summary>
        public GpuTensor Div(GpuTensor a, GpuTensor b)
        {
            ThrowIfDisposed();
            ValidateShapes(a, b, "division");
            return _kernels.Div(a, b);
        }

        /// <summary>
        /// Scalar addition: B = A + scalar
        /// </summary>
        public GpuTensor AddScalar(GpuTensor a, float scalar)
        {
            ThrowIfDisposed();
            return _kernels.AddScalar(a, scalar);
        }

        /// <summary>
        /// Scalar multiplication: B = A * scalar
        /// </summary>
        public GpuTensor MulScalar(GpuTensor a, float scalar)
        {
            ThrowIfDisposed();
            return _kernels.MulScalar(a, scalar);
        }

        /// <summary>
        /// Element-wise power: B = A^exponent
        /// </summary>
        public GpuTensor Pow(GpuTensor a, float exponent)
        {
            ThrowIfDisposed();
            return _kernels.Pow(a, exponent);
        }

        /// <summary>
        /// Element-wise square root: B = sqrt(A)
        /// </summary>
        public GpuTensor Sqrt(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Sqrt(a);
        }

        /// <summary>
        /// Element-wise exponential: B = exp(A)
        /// </summary>
        public GpuTensor Exp(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Exp(a);
        }

        /// <summary>
        /// Element-wise natural logarithm: B = log(A)
        /// </summary>
        public GpuTensor Log(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Log(a);
        }

        /// <summary>
        /// Element-wise absolute value: B = |A|
        /// </summary>
        public GpuTensor Abs(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Abs(a);
        }

        /// <summary>
        /// Element-wise negation: B = -A
        /// </summary>
        public GpuTensor Neg(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Neg(a);
        }

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Matrix multiplication: C = A @ B
        /// Supports batched matrix multiplication.
        ///
        /// For 2D matrices [M, K] @ [K, N] -> [M, N]
        /// For batched [B, M, K] @ [B, K, N] -> [B, M, N]
        /// </summary>
        public GpuTensor MatMul(GpuTensor a, GpuTensor b)
        {
            ThrowIfDisposed();

            if (a.Shape.Length < 2 || b.Shape.Length < 2)
                throw new ArgumentException("MatMul requires at least 2D tensors");

            // Get inner dimensions
            int aLastDim = a.Shape[^1];
            int bSecondLastDim = b.Shape.Length >= 2 ? b.Shape[^2] : b.Shape[0];

            if (aLastDim != bSecondLastDim)
                throw new ArgumentException(
                    $"MatMul dimension mismatch: {string.Join("x", a.Shape)} and {string.Join("x", b.Shape)}");

            return _kernels.MatMul(a, b);
        }

        /// <summary>
        /// Transpose the last two dimensions of a tensor
        /// </summary>
        public GpuTensor Transpose(GpuTensor a)
        {
            ThrowIfDisposed();

            if (a.Shape.Length < 2)
                throw new ArgumentException("Transpose requires at least 2D tensor");

            return _kernels.Transpose(a);
        }

        /// <summary>
        /// Outer product of two vectors: C = a ⊗ b
        /// </summary>
        public GpuTensor Outer(GpuTensor a, GpuTensor b)
        {
            ThrowIfDisposed();

            if (a.Shape.Length != 1 || b.Shape.Length != 1)
                throw new ArgumentException("Outer product requires 1D tensors");

            return _kernels.Outer(a, b);
        }

        #endregion

        #region Activation Functions

        /// <summary>
        /// ReLU activation: B = max(0, A)
        /// </summary>
        public GpuTensor ReLU(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.ReLU(a);
        }

        /// <summary>
        /// Leaky ReLU activation: B = max(alpha * A, A)
        /// </summary>
        public GpuTensor LeakyReLU(GpuTensor a, float alpha = 0.01f)
        {
            ThrowIfDisposed();
            return _kernels.LeakyReLU(a, alpha);
        }

        /// <summary>
        /// Sigmoid activation: B = 1 / (1 + exp(-A))
        /// </summary>
        public GpuTensor Sigmoid(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Sigmoid(a);
        }

        /// <summary>
        /// Tanh activation: B = tanh(A)
        /// </summary>
        public GpuTensor Tanh(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Tanh(a);
        }

        /// <summary>
        /// GELU activation (Gaussian Error Linear Unit)
        /// Used in transformers: B = A * 0.5 * (1 + erf(A / sqrt(2)))
        /// </summary>
        public GpuTensor GELU(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.GELU(a);
        }

        /// <summary>
        /// Swish activation: B = A * sigmoid(A)
        /// </summary>
        public GpuTensor Swish(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Swish(a);
        }

        /// <summary>
        /// Softmax activation over the last dimension
        /// </summary>
        public GpuTensor Softmax(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Softmax(a);
        }

        /// <summary>
        /// Log softmax for numerical stability in cross-entropy loss
        /// </summary>
        public GpuTensor LogSoftmax(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.LogSoftmax(a);
        }

        #endregion

        #region Reduction Operations

        /// <summary>
        /// Sum all elements of a tensor
        /// </summary>
        public float Sum(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Sum(a);
        }

        /// <summary>
        /// Sum along a specific axis
        /// </summary>
        public GpuTensor SumAxis(GpuTensor a, int axis, bool keepDims = false)
        {
            ThrowIfDisposed();
            return _kernels.SumAxis(a, axis, keepDims);
        }

        /// <summary>
        /// Mean of all elements
        /// </summary>
        public float Mean(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Mean(a);
        }

        /// <summary>
        /// Mean along a specific axis
        /// </summary>
        public GpuTensor MeanAxis(GpuTensor a, int axis, bool keepDims = false)
        {
            ThrowIfDisposed();
            return _kernels.MeanAxis(a, axis, keepDims);
        }

        /// <summary>
        /// Maximum element
        /// </summary>
        public float Max(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Max(a);
        }

        /// <summary>
        /// Maximum along a specific axis
        /// </summary>
        public GpuTensor MaxAxis(GpuTensor a, int axis, bool keepDims = false)
        {
            ThrowIfDisposed();
            return _kernels.MaxAxis(a, axis, keepDims);
        }

        /// <summary>
        /// Minimum element
        /// </summary>
        public float Min(GpuTensor a)
        {
            ThrowIfDisposed();
            return _kernels.Min(a);
        }

        /// <summary>
        /// Variance along a specific axis
        /// </summary>
        public GpuTensor Variance(GpuTensor a, int axis, bool keepDims = false)
        {
            ThrowIfDisposed();
            return _kernels.Variance(a, axis, keepDims);
        }

        /// <summary>
        /// Standard deviation along a specific axis
        /// </summary>
        public GpuTensor Std(GpuTensor a, int axis, bool keepDims = false)
        {
            ThrowIfDisposed();
            return Sqrt(_kernels.Variance(a, axis, keepDims));
        }

        #endregion

        #region Neural Network Layers

        /// <summary>
        /// Layer normalization over the last dimension
        /// Used in transformers: output = (x - mean) / sqrt(var + eps) * gamma + beta
        /// </summary>
        public GpuTensor LayerNorm(GpuTensor x, GpuTensor gamma, GpuTensor beta, float eps = 1e-5f)
        {
            ThrowIfDisposed();
            return _kernels.LayerNorm(x, gamma, beta, eps);
        }

        /// <summary>
        /// Batch normalization
        /// </summary>
        public GpuTensor BatchNorm(GpuTensor x, GpuTensor gamma, GpuTensor beta,
            GpuTensor? runningMean = null, GpuTensor? runningVar = null,
            float eps = 1e-5f, float momentum = 0.1f, bool training = true)
        {
            ThrowIfDisposed();
            return _kernels.BatchNorm(x, gamma, beta, runningMean, runningVar, eps, momentum, training);
        }

        /// <summary>
        /// Dropout layer (training mode)
        /// </summary>
        public GpuTensor Dropout(GpuTensor x, float p = 0.5f, bool training = true)
        {
            ThrowIfDisposed();
            if (!training || p == 0)
                return x;
            return _kernels.Dropout(x, p);
        }

        /// <summary>
        /// 2D Convolution
        /// </summary>
        public GpuTensor Conv2d(GpuTensor input, GpuTensor kernel, int stride = 1, int padding = 0)
        {
            ThrowIfDisposed();
            return _kernels.Conv2d(input, kernel, stride, padding);
        }

        /// <summary>
        /// 2D Max pooling
        /// </summary>
        public GpuTensor MaxPool2d(GpuTensor input, int kernelSize, int stride = -1)
        {
            ThrowIfDisposed();
            if (stride < 0) stride = kernelSize;
            return _kernels.MaxPool2d(input, kernelSize, stride);
        }

        #endregion

        #region Attention Operations

        /// <summary>
        /// Scaled dot-product attention (core of transformer architecture)
        ///
        /// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        ///
        /// Mathematical background:
        /// - Q (Query): What we're looking for
        /// - K (Key): What we're matching against
        /// - V (Value): What we retrieve
        /// - The scaling factor sqrt(d_k) prevents softmax saturation
        /// </summary>
        /// <param name="query">Query tensor [batch, seq_len, d_k]</param>
        /// <param name="key">Key tensor [batch, seq_len, d_k]</param>
        /// <param name="value">Value tensor [batch, seq_len, d_v]</param>
        /// <param name="mask">Optional attention mask</param>
        /// <param name="dropout">Dropout probability</param>
        /// <returns>Attention output [batch, seq_len, d_v]</returns>
        public GpuTensor ScaledDotProductAttention(
            GpuTensor query, GpuTensor key, GpuTensor value,
            GpuTensor? mask = null, float dropout = 0f)
        {
            ThrowIfDisposed();
            return _kernels.ScaledDotProductAttention(query, key, value, mask, dropout);
        }

        /// <summary>
        /// Multi-head attention (as used in transformers)
        ///
        /// Splits Q, K, V into multiple heads, applies attention to each,
        /// then concatenates and projects the results.
        /// </summary>
        public GpuTensor MultiHeadAttention(
            GpuTensor query, GpuTensor key, GpuTensor value,
            GpuTensor wQ, GpuTensor wK, GpuTensor wV, GpuTensor wO,
            int numHeads, GpuTensor? mask = null, float dropout = 0f)
        {
            ThrowIfDisposed();
            return _kernels.MultiHeadAttention(query, key, value, wQ, wK, wV, wO, numHeads, mask, dropout);
        }

        #endregion

        #region Consciousness Operators (NSL-specific)

        /// <summary>
        /// Holographic operator (◈) - GPU accelerated
        /// Creates distributed representations using attention mechanisms.
        ///
        /// Mathematical foundation:
        /// H(x) = Σᵢ αᵢ · vᵢ where αᵢ = softmax(xᵀWK · WQx / √d)
        ///
        /// This creates a holographic-like representation where information
        /// is distributed across the entire vector space, similar to how
        /// holograms store information in interference patterns.
        /// </summary>
        public GpuTensor Holographic(GpuTensor x)
        {
            ThrowIfDisposed();
            return _kernels.Holographic(x);
        }

        /// <summary>
        /// Tensor product operator (⊗) - GPU accelerated
        /// Computes outer product creating entangled representations.
        ///
        /// Mathematical foundation:
        /// For vectors a and b: (a ⊗ b)ᵢⱼ = aᵢ · bⱼ
        /// This creates a higher-dimensional representation encoding
        /// all pairwise interactions, analogous to quantum entanglement.
        /// </summary>
        public GpuTensor TensorProduct(GpuTensor a, GpuTensor b)
        {
            ThrowIfDisposed();
            return _kernels.TensorProduct(a, b);
        }

        /// <summary>
        /// Quantum branching operator (Ψ) - GPU accelerated
        /// Creates superposition states with probability amplitudes.
        ///
        /// Mathematical foundation:
        /// Ψ(x) = Σᵢ |αᵢ|² · |ψᵢ⟩ where Σᵢ |αᵢ|² = 1
        ///
        /// Simulates quantum superposition by maintaining multiple
        /// weighted states simultaneously.
        /// </summary>
        public GpuTensor QuantumBranch(GpuTensor x, int numBranches = 4)
        {
            ThrowIfDisposed();
            return _kernels.QuantumBranch(x, numBranches);
        }

        #endregion

        #region Memory Management

        /// <summary>
        /// Get current GPU memory usage in bytes
        /// </summary>
        public long GetMemoryUsage()
        {
            ThrowIfDisposed();
            // Note: ILGPU doesn't directly expose memory usage
            // This would need backend-specific implementation
            return 0;
        }

        /// <summary>
        /// Synchronize GPU operations (wait for completion)
        /// </summary>
        public void Synchronize()
        {
            ThrowIfDisposed();
            _accelerator.Synchronize();
        }

        #endregion

        #region Private Helpers

        private (Accelerator accelerator, GpuBackend backend) CreateAccelerator(GpuBackend requested, int deviceIndex)
        {
            if (requested == GpuBackend.Auto)
            {
                // Try CUDA first, then OpenCL, then CPU
                var cudaDevices = _context.GetCudaDevices();
                if (cudaDevices.Count > 0)
                {
                    var device = cudaDevices[Math.Min(deviceIndex, cudaDevices.Count - 1)];
                    return (device.CreateAccelerator(_context), GpuBackend.CUDA);
                }

                var clDevices = _context.GetCLDevices();
                if (clDevices.Count > 0)
                {
                    var device = clDevices[Math.Min(deviceIndex, clDevices.Count - 1)];
                    return (device.CreateAccelerator(_context), GpuBackend.OpenCL);
                }

                // Fall back to CPU
                var cpuDevice = _context.GetCPUDevice(0);
                return (cpuDevice.CreateAccelerator(_context), GpuBackend.CPU);
            }

            return requested switch
            {
                GpuBackend.CUDA =>
                    (_context.GetCudaDevice(deviceIndex).CreateAccelerator(_context), GpuBackend.CUDA),
                GpuBackend.OpenCL =>
                    (_context.GetCLDevice(deviceIndex).CreateAccelerator(_context), GpuBackend.OpenCL),
                GpuBackend.CPU =>
                    (_context.GetCPUDevice(deviceIndex).CreateAccelerator(_context), GpuBackend.CPU),
                _ => throw new ArgumentException($"Unknown GPU backend: {requested}")
            };
        }

        private static GpuDeviceInfo GetDeviceInfo(Accelerator accelerator, GpuBackend backend, int index)
        {
            return new GpuDeviceInfo
            {
                Name = accelerator.Name,
                Index = index,
                Backend = backend,
                TotalMemory = accelerator.MemorySize,
                ComputeUnits = accelerator.NumMultiprocessors,
                MaxThreadsPerGroup = accelerator.MaxNumThreadsPerGroup,
                WarpSize = accelerator.WarpSize,
                IsSelected = true
            };
        }

        private static void ValidateShapes(GpuTensor a, GpuTensor b, string operation)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
            {
                throw new ArgumentException(
                    $"Shape mismatch for {operation}: " +
                    $"[{string.Join(", ", a.Shape)}] vs [{string.Join(", ", b.Shape)}]");
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GpuAccelerator));
        }

        #endregion

        /// <summary>
        /// Dispose GPU resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _accelerator.Dispose();
                _context.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}