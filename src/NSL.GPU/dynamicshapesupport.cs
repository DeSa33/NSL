using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// Dynamic Shape Support for NSL GPU Operations.
    ///
    /// Handles tensors with variable dimensions at runtime:
    /// - Symbolic dimensions (e.g., batch_size=-1, seq_len=dynamic)
    /// - Shape inference from inputs
    /// - Automatic buffer resizing
    /// - Kernel dispatch based on actual shapes
    ///
    /// This allows models to handle:
    /// - Variable batch sizes
    /// - Variable sequence lengths
    /// - Dynamic image sizes
    /// </summary>
    public class DynamicShapeManager
    {
        private readonly Accelerator _accelerator;
        private readonly GpuMemoryManager _memoryManager;
        private readonly ConcurrentDictionary<string, ShapeInfo> _registeredShapes = new();
        private readonly ConcurrentDictionary<string, DynamicBuffer> _dynamicBuffers = new();

        /// <summary>
        /// Represents a shape that may contain dynamic dimensions
        /// </summary>
        public class ShapeInfo
        {
            /// <summary>Shape dimensions (-1 for dynamic)</summary>
            public int[] Dims { get; set; } = Array.Empty<int>();

            /// <summary>Names for each dimension (optional)</summary>
            public string[]? DimNames { get; set; }

            /// <summary>Whether this shape has any dynamic dimensions</summary>
            public bool IsDynamic => Dims.Any(d => d < 0);

            /// <summary>Number of static elements (product of non-dynamic dims)</summary>
            public int StaticElements => Dims.Where(d => d > 0).Aggregate(1, (a, b) => a * b);

            /// <summary>Indices of dynamic dimensions</summary>
            public int[] DynamicDimIndices => Dims
                .Select((d, i) => (d, i))
                .Where(x => x.d < 0)
                .Select(x => x.i)
                .ToArray();

            /// <summary>
            /// Resolve dynamic dimensions with actual values
            /// </summary>
            public int[] Resolve(Dictionary<string, int>? dimValues = null)
            {
                var resolved = (int[])Dims.Clone();

                if (dimValues != null && DimNames != null)
                {
                    for (int i = 0; i < Dims.Length; i++)
                    {
                        if (Dims[i] < 0 && DimNames[i] != null && dimValues.TryGetValue(DimNames[i], out int value))
                        {
                            resolved[i] = value;
                        }
                    }
                }

                return resolved;
            }

            /// <summary>Public API</summary>
            public override string ToString()
            {
                var dimStrs = Dims.Select((d, i) =>
                    d < 0 ? (DimNames?[i] ?? "?") : d.ToString());
                return $"[{string.Join(", ", dimStrs)}]";
            }
        }

        /// <summary>
        /// A buffer that automatically resizes for dynamic shapes
        /// </summary>
        private class DynamicBuffer : IDisposable
        {
            /// <summary>Public API</summary>
            public MemoryBuffer1D<float, Stride1D.Dense>? Buffer { get; set; }
            /// <summary>Public API</summary>
            public int CurrentCapacity { get; set; }
            /// <summary>Public API</summary>
            public int[] CurrentShape { get; set; } = Array.Empty<int>();
            /// <summary>Public API</summary>
            public DateTime LastUsed { get; set; } = DateTime.UtcNow;
            private readonly Accelerator _accelerator;
            private bool _disposed;

            /// <summary>Public API</summary>
            public DynamicBuffer(Accelerator accelerator)
            {
                _accelerator = accelerator;
            }

            /// <summary>
            /// Ensure buffer has sufficient capacity, reallocating if needed
            /// </summary>
            public void EnsureCapacity(int requiredElements, int[] newShape)
            {
                if (Buffer == null || CurrentCapacity < requiredElements)
                {
                    // Allocate with some headroom (1.5x) to reduce reallocations
                    int newCapacity = (int)(requiredElements * 1.5);
                    Buffer?.Dispose();
                    Buffer = _accelerator.Allocate1D<float>(newCapacity);
                    CurrentCapacity = newCapacity;
                }

                CurrentShape = newShape;
                LastUsed = DateTime.UtcNow;
            }

            /// <summary>Public API</summary>
            public void Dispose()
            {
                if (!_disposed)
                {
                    Buffer?.Dispose();
                    _disposed = true;
                }
            }
        }

        /// <summary>Public API</summary>
        public DynamicShapeManager(Accelerator accelerator, GpuMemoryManager? memoryManager = null)
        {
            _accelerator = accelerator;
            _memoryManager = memoryManager ?? new GpuMemoryManager(accelerator);
        }

        #region Shape Registration

        /// <summary>
        /// Register a named shape with optional dynamic dimensions
        /// </summary>
        public void RegisterShape(string name, int[] dims, string[]? dimNames = null)
        {
            _registeredShapes[name] = new ShapeInfo
            {
                Dims = dims,
                DimNames = dimNames
            };
        }

        /// <summary>
        /// Register common dynamic shapes
        /// </summary>
        public void RegisterCommonShapes()
        {
            // Common NLP shapes
            RegisterShape("embedding_input", new[] { -1, -1 }, new[] { "batch", "seq_len" });
            RegisterShape("hidden_state", new[] { -1, -1, -1 }, new[] { "batch", "seq_len", "hidden" });
            RegisterShape("attention_scores", new[] { -1, -1, -1, -1 }, new[] { "batch", "heads", "seq_q", "seq_k" });

            // Common vision shapes
            RegisterShape("image_batch", new[] { -1, -1, -1, -1 }, new[] { "batch", "channels", "height", "width" });
            RegisterShape("feature_map", new[] { -1, -1, -1, -1 }, new[] { "batch", "channels", "height", "width" });

            // Common output shapes
            RegisterShape("logits", new[] { -1, -1 }, new[] { "batch", "classes" });
            RegisterShape("sequence_logits", new[] { -1, -1, -1 }, new[] { "batch", "seq_len", "vocab" });
        }

        /// <summary>
        /// Get registered shape info
        /// </summary>
        public ShapeInfo? GetShape(string name)
        {
            return _registeredShapes.TryGetValue(name, out var shape) ? shape : null;
        }

        #endregion

        #region Shape Inference

        /// <summary>
        /// Infer output shape from operation and input shapes
        /// </summary>
        public int[] InferShape(string operation, params int[][] inputShapes)
        {
            return operation.ToLower() switch
            {
                "matmul" => InferMatMulShape(inputShapes[0], inputShapes[1]),
                "add" or "sub" or "mul" or "div" => InferBroadcastShape(inputShapes[0], inputShapes[1]),
                "relu" or "sigmoid" or "tanh" or "gelu" => inputShapes[0],
                "softmax" => inputShapes[0],
                "layernorm" or "batchnorm" => inputShapes[0],
                "transpose" => InferTransposeShape(inputShapes[0]),
                "reshape" => inputShapes[1], // Second arg is target shape
                "concat" => InferConcatShape(inputShapes, axis: 0),
                "conv2d" => InferConv2dShape(inputShapes[0], inputShapes[1]),
                "maxpool2d" or "avgpool2d" => InferPool2dShape(inputShapes[0], kernelSize: 2, stride: 2),
                _ => throw new ArgumentException($"Unknown operation: {operation}")
            };
        }

        private int[] InferMatMulShape(int[] a, int[] b)
        {
            if (a.Length == 1) return new[] { b[^1] };
            if (b.Length == 1) return new[] { a[0] };

            var result = new int[Math.Max(a.Length, b.Length)];

            // Batch dimensions broadcast
            for (int i = 0; i < result.Length - 2; i++)
            {
                int aIdx = a.Length - result.Length + i;
                int bIdx = b.Length - result.Length + i;
                int aDim = aIdx >= 0 ? a[aIdx] : 1;
                int bDim = bIdx >= 0 ? b[bIdx] : 1;
                result[i] = Math.Max(aDim, bDim);
            }

            result[^2] = a[^2];
            result[^1] = b[^1];

            return result;
        }

        private int[] InferBroadcastShape(int[] a, int[] b)
        {
            int maxDims = Math.Max(a.Length, b.Length);
            var result = new int[maxDims];

            for (int i = 0; i < maxDims; i++)
            {
                int aIdx = a.Length - maxDims + i;
                int bIdx = b.Length - maxDims + i;
                int aDim = aIdx >= 0 ? a[aIdx] : 1;
                int bDim = bIdx >= 0 ? b[bIdx] : 1;

                if (aDim != bDim && aDim != 1 && bDim != 1 && aDim != -1 && bDim != -1)
                {
                    throw new ArgumentException($"Cannot broadcast {aDim} with {bDim}");
                }

                result[i] = Math.Max(aDim, bDim);
            }

            return result;
        }

        private int[] InferTransposeShape(int[] input)
        {
            var result = new int[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = input[input.Length - 1 - i];
            }
            return result;
        }

        private int[] InferConcatShape(int[][] shapes, int axis)
        {
            var result = (int[])shapes[0].Clone();
            for (int i = 1; i < shapes.Length; i++)
            {
                result[axis] += shapes[i][axis];
            }
            return result;
        }

        private int[] InferConv2dShape(int[] input, int[] kernel,
            int strideH = 1, int strideW = 1, int padH = 0, int padW = 0)
        {
            // input: [N, C_in, H, W]
            // kernel: [C_out, C_in, kH, kW]
            int outH = (input[2] + 2 * padH - kernel[2]) / strideH + 1;
            int outW = (input[3] + 2 * padW - kernel[3]) / strideW + 1;
            return new[] { input[0], kernel[0], outH, outW };
        }

        private int[] InferPool2dShape(int[] input, int kernelSize, int stride)
        {
            int outH = (input[2] - kernelSize) / stride + 1;
            int outW = (input[3] - kernelSize) / stride + 1;
            return new[] { input[0], input[1], outH, outW };
        }

        #endregion

        #region Dynamic Tensor Operations

        /// <summary>
        /// Create a dynamic tensor that automatically resizes
        /// </summary>
        public DynamicTensor CreateDynamicTensor(string name, ShapeInfo shape)
        {
            var buffer = new DynamicBuffer(_accelerator);
            _dynamicBuffers[name] = buffer;

            return new DynamicTensor(_accelerator, name, shape, buffer);
        }

        /// <summary>
        /// Get or create a dynamic tensor
        /// </summary>
        public DynamicTensor GetOrCreateDynamic(string name, int[] initialShape)
        {
            if (!_dynamicBuffers.TryGetValue(name, out var buffer))
            {
                buffer = new DynamicBuffer(_accelerator);
                _dynamicBuffers[name] = buffer;
            }

            var shapeInfo = new ShapeInfo { Dims = initialShape };
            return new DynamicTensor(_accelerator, name, shapeInfo, buffer);
        }

        /// <summary>
        /// Cleanup unused dynamic buffers
        /// </summary>
        public void CleanupUnused(TimeSpan maxIdleTime)
        {
            var now = DateTime.UtcNow;
            var toRemove = _dynamicBuffers
                .Where(kvp => now - kvp.Value.LastUsed > maxIdleTime)
                .Select(kvp => kvp.Key)
                .ToList();

            foreach (var key in toRemove)
            {
                if (_dynamicBuffers.TryRemove(key, out var buffer))
                {
                    buffer.Dispose();
                }
            }
        }

        #endregion
    }

    /// <summary>
    /// A tensor with dynamic shape support.
    /// Automatically resizes underlying buffer when shape changes.
    /// </summary>
    public class DynamicTensor : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly DynamicShapeManager.ShapeInfo _shapeInfo;
        private MemoryBuffer1D<float, Stride1D.Dense>? _buffer;
        private int[] _currentShape;
        private int _currentCapacity;
        private bool _disposed;

        /// <summary>Public API</summary>
        public string Name { get; }
        /// <summary>Public API</summary>
        public int[] Shape => _currentShape;
        /// <summary>Public API</summary>
        public int Size => _currentShape.Aggregate(1, (a, b) => a * b);
        /// <summary>Public API</summary>
        public bool IsDynamic => _shapeInfo.IsDynamic;
        /// <summary>Public API</summary>
        public ArrayView<float> View => _buffer!.View;

        internal DynamicTensor(Accelerator accelerator, string name,
            DynamicShapeManager.ShapeInfo shapeInfo, object buffer)
        {
            _accelerator = accelerator;
            Name = name;
            _shapeInfo = shapeInfo;
            _currentShape = shapeInfo.Dims.Select(d => d < 0 ? 1 : d).ToArray();
        }

        /// <summary>
        /// Reshape to new dimensions, reallocating if needed
        /// </summary>
        public void Reshape(int[] newShape)
        {
            int newSize = newShape.Aggregate(1, (a, b) => a * b);

            if (_buffer == null || _currentCapacity < newSize)
            {
                // Allocate with headroom
                int newCapacity = (int)(newSize * 1.5);
                _buffer?.Dispose();
                _buffer = _accelerator.Allocate1D<float>(newCapacity);
                _currentCapacity = newCapacity;
            }

            _currentShape = newShape;
        }

        /// <summary>
        /// Set data with automatic reshape
        /// </summary>
        public void SetData(float[] data, int[] shape)
        {
            Reshape(shape);
            _buffer!.View.SubView(0, data.Length).CopyFromCPU(data);
        }

        /// <summary>
        /// Get data as array
        /// </summary>
        public float[] ToArray()
        {
            var result = new float[Size];
            _buffer!.View.SubView(0, Size).CopyToCPU(result);
            return result;
        }

        /// <summary>
        /// Convert to fixed GpuTensor (copies data)
        /// </summary>
        public GpuTensor ToGpuTensor()
        {
            var data = ToArray();
            return GpuTensor.FromArray(_accelerator, data, _currentShape);
        }

        /// <summary>
        /// Create from fixed GpuTensor
        /// </summary>
        public static DynamicTensor FromGpuTensor(Accelerator accelerator, GpuTensor tensor, string name = "dynamic")
        {
            var shapeInfo = new DynamicShapeManager.ShapeInfo { Dims = tensor.Shape };
            var dynamic = new DynamicTensor(accelerator, name, shapeInfo, new object());
            dynamic.SetData(tensor.ToArray(), tensor.Shape);
            return dynamic;
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _buffer?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Extensions for dynamic shape operations
    /// </summary>
    public static class DynamicShapeExtensions
    {
        /// <summary>
        /// Check if two shapes are compatible for broadcasting
        /// </summary>
        public static bool IsCompatible(this int[] shape1, int[] shape2)
        {
            int maxDims = Math.Max(shape1.Length, shape2.Length);

            for (int i = 0; i < maxDims; i++)
            {
                int idx1 = shape1.Length - maxDims + i;
                int idx2 = shape2.Length - maxDims + i;
                int dim1 = idx1 >= 0 ? shape1[idx1] : 1;
                int dim2 = idx2 >= 0 ? shape2[idx2] : 1;

                // Dynamic dims are always compatible
                if (dim1 < 0 || dim2 < 0) continue;

                // Must be equal or one must be 1 (broadcast)
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Compute total elements, treating dynamic dims as 1
        /// </summary>
        public static int StaticSize(this int[] shape)
        {
            return shape.Where(d => d > 0).Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Replace dynamic dimensions (-1) with actual values
        /// </summary>
        public static int[] ResolveDynamic(this int[] shape, int totalElements)
        {
            var result = (int[])shape.Clone();
            int dynamicIdx = Array.IndexOf(result, -1);

            if (dynamicIdx >= 0)
            {
                int staticProduct = result.Where(d => d > 0).Aggregate(1, (a, b) => a * b);
                result[dynamicIdx] = totalElements / staticProduct;
            }

            return result;
        }
    }
}