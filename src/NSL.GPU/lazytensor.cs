using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// Lazy tensor that defers execution until values are needed.
    ///
    /// Instead of executing operations immediately, LazyTensor records them
    /// in a computation graph. When the tensor's values are accessed (via
    /// ToArray() or Evaluate()), the entire graph is optimized and executed
    /// efficiently.
    ///
    /// Benefits:
    /// - Automatic operation fusion (eliminates intermediate tensors)
    /// - Graph-level optimization (dead code elimination, CSE)
    /// - Parallel execution of independent operations
    /// - Memory planning and reuse
    ///
    /// Usage:
    /// <code>
    /// // Create lazy tensors
    /// var x = LazyTensor.FromArray(accelerator, data, 32, 768);
    /// var w = LazyTensor.FromArray(accelerator, weights, 768, 768);
    ///
    /// // Operations are recorded, not executed
    /// var y = x.MatMul(w);
    /// var z = y.ReLU();
    /// var output = z.LayerNorm(gamma, beta);
    ///
    /// // Graph is optimized and executed when values are needed
    /// float[] result = output.ToArray();  // &lt;-- Execution happens here
    /// </code>
    ///
    /// This is similar to:
    /// - PyTorch's LazyTensor (lazy tensor mode)
    /// - TensorFlow's tf.function graph tracing
    /// - JAX's jit traced execution
    /// </summary>
    public class LazyTensor : IDisposable
    {
        private static ComputationGraph? _sharedGraph;
        private static readonly object _graphLock = new();

        private readonly Accelerator _accelerator;
        private readonly GraphNode _node;
        private readonly int[] _shape;
        private GpuTensor? _materializedTensor;
        private bool _isMaterialized;
        private bool _disposed;

        /// <summary>
        /// Shape of the tensor
        /// </summary>
        public int[] Shape => _shape;

        /// <summary>
        /// Total number of elements
        /// </summary>
        public int Size => _shape.Aggregate(1, (a, b) => a * b);

        /// <summary>
        /// Number of dimensions
        /// </summary>
        public int NDim => _shape.Length;

        /// <summary>
        /// Whether this tensor has been materialized (computed)
        /// </summary>
        public bool IsMaterialized => _isMaterialized;

        /// <summary>
        /// The underlying graph node
        /// </summary>
        internal GraphNode Node => _node;

        /// <summary>
        /// Get or create the shared computation graph
        /// </summary>
        private static ComputationGraph GetOrCreateGraph(Accelerator accelerator)
        {
            lock (_graphLock)
            {
                if (_sharedGraph == null)
                {
                    _sharedGraph = new ComputationGraph(accelerator);
                }
                return _sharedGraph;
            }
        }

        /// <summary>
        /// Clear the shared graph (call after execution to free memory)
        /// </summary>
        public static void ClearGraph()
        {
            lock (_graphLock)
            {
                _sharedGraph?.Dispose();
                _sharedGraph = null;
            }
        }

        private LazyTensor(Accelerator accelerator, GraphNode node, int[] shape)
        {
            _accelerator = accelerator;
            _node = node;
            _shape = (int[])shape.Clone();
        }

        #region Factory Methods

        /// <summary>
        /// Create a lazy tensor from a float array
        /// </summary>
        public static LazyTensor FromArray(Accelerator accelerator, float[] data, params int[] shape)
        {
            var graph = GetOrCreateGraph(accelerator);
            var gpuTensor = GpuTensor.FromArray(accelerator, data, shape);
            var node = graph.Constant($"const_{Guid.NewGuid():N}", gpuTensor);
            return new LazyTensor(accelerator, node, shape) { _materializedTensor = gpuTensor, _isMaterialized = true };
        }

        /// <summary>
        /// Create a lazy tensor from a GpuTensor
        /// </summary>
        public static LazyTensor FromGpuTensor(Accelerator accelerator, GpuTensor tensor, string? name = null)
        {
            var graph = GetOrCreateGraph(accelerator);
            var node = graph.Constant(name ?? $"tensor_{Guid.NewGuid():N}", tensor);
            return new LazyTensor(accelerator, node, tensor.Shape) { _materializedTensor = tensor, _isMaterialized = true };
        }

        /// <summary>
        /// Create an input placeholder (bound at execution time)
        /// </summary>
        public static LazyTensor Input(Accelerator accelerator, string name, params int[] shape)
        {
            var graph = GetOrCreateGraph(accelerator);
            var node = graph.Input(name, shape);
            return new LazyTensor(accelerator, node, shape);
        }

        /// <summary>
        /// Create a lazy tensor filled with zeros
        /// </summary>
        public static LazyTensor Zeros(Accelerator accelerator, params int[] shape)
        {
            var tensor = GpuTensor.Zeros(accelerator, shape);
            return FromGpuTensor(accelerator, tensor, "zeros");
        }

        /// <summary>
        /// Create a lazy tensor filled with ones
        /// </summary>
        public static LazyTensor Ones(Accelerator accelerator, params int[] shape)
        {
            var tensor = GpuTensor.Ones(accelerator, shape);
            return FromGpuTensor(accelerator, tensor, "ones");
        }

        /// <summary>
        /// Create a lazy tensor with random values
        /// </summary>
        public static LazyTensor Random(Accelerator accelerator, params int[] shape)
        {
            var tensor = GpuTensor.Random(accelerator, shape);
            return FromGpuTensor(accelerator, tensor, "random");
        }

        /// <summary>
        /// Create a lazy tensor with random normal values
        /// </summary>
        public static LazyTensor RandomNormal(Accelerator accelerator, float mean, float std, params int[] shape)
        {
            var tensor = GpuTensor.RandomNormal(accelerator, mean, std, shape);
            return FromGpuTensor(accelerator, tensor, "random_normal");
        }

        #endregion

        #region Operations

        /// <summary>
        /// Matrix multiplication
        /// </summary>
        public LazyTensor MatMul(LazyTensor other)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.MatMul(_node, other._node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Element-wise addition
        /// </summary>
        public LazyTensor Add(LazyTensor other)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Add(_node, other._node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        public LazyTensor Sub(LazyTensor other)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Sub(_node, other._node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Element-wise multiplication
        /// </summary>
        public LazyTensor Mul(LazyTensor other)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Mul(_node, other._node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Element-wise division
        /// </summary>
        public LazyTensor Div(LazyTensor other)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Div(_node, other._node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// ReLU activation
        /// </summary>
        public LazyTensor ReLU()
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.ReLU(_node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// GELU activation
        /// </summary>
        public LazyTensor GELU()
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.GELU(_node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Sigmoid activation
        /// </summary>
        public LazyTensor Sigmoid()
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Sigmoid(_node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Tanh activation
        /// </summary>
        public LazyTensor Tanh()
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Tanh(_node);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Softmax activation
        /// </summary>
        public LazyTensor Softmax(int axis = -1)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Softmax(_node, axis);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Layer normalization
        /// </summary>
        public LazyTensor LayerNorm(LazyTensor gamma, LazyTensor beta, float eps = 1e-5f)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.LayerNorm(_node, gamma._node, beta._node, eps);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// RMS normalization
        /// </summary>
        public LazyTensor RMSNorm(LazyTensor gamma, float eps = 1e-5f)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.RMSNorm(_node, gamma._node, eps);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Scaled dot-product attention
        /// </summary>
        public static LazyTensor Attention(LazyTensor query, LazyTensor key, LazyTensor value,
            LazyTensor? mask = null, float dropoutProb = 0f)
        {
            var graph = GetOrCreateGraph(query._accelerator);
            var node = graph.ScaledDotProductAttention(query._node, key._node, value._node,
                mask?._node, dropoutProb);
            return new LazyTensor(query._accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// FlashAttention-2 style memory-efficient attention
        /// </summary>
        public static LazyTensor FlashAttention(LazyTensor query, LazyTensor key, LazyTensor value,
            LazyTensor? mask = null, float dropoutProb = 0f, bool causal = false)
        {
            var graph = GetOrCreateGraph(query._accelerator);
            var node = graph.FlashAttention(query._node, key._node, value._node,
                mask?._node, dropoutProb, causal);
            return new LazyTensor(query._accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Transpose
        /// </summary>
        public LazyTensor Transpose(int dim0 = -2, int dim1 = -1)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Transpose(_node, dim0, dim1);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Reshape
        /// </summary>
        public LazyTensor Reshape(params int[] newShape)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Reshape(_node, newShape);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Sum reduction
        /// </summary>
        public LazyTensor Sum(int? axis = null, bool keepDims = false)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Sum(_node, axis, keepDims);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        /// <summary>
        /// Mean reduction
        /// </summary>
        public LazyTensor Mean(int? axis = null, bool keepDims = false)
        {
            var graph = GetOrCreateGraph(_accelerator);
            var node = graph.Mean(_node, axis, keepDims);
            return new LazyTensor(_accelerator, node, node.OutputShape);
        }

        #endregion

        #region Operators

        /// <summary>Public API</summary>
        public static LazyTensor operator +(LazyTensor a, LazyTensor b) => a.Add(b);
        /// <summary>Public API</summary>
        public static LazyTensor operator -(LazyTensor a, LazyTensor b) => a.Sub(b);
        /// <summary>Public API</summary>
        public static LazyTensor operator *(LazyTensor a, LazyTensor b) => a.Mul(b);
        /// <summary>Public API</summary>
        public static LazyTensor operator /(LazyTensor a, LazyTensor b) => a.Div(b);

        #endregion

        #region Materialization

        /// <summary>
        /// Force evaluation of the computation graph and return GPU tensor
        /// </summary>
        public GpuTensor Evaluate()
        {
            if (_isMaterialized && _materializedTensor != null)
            {
                return _materializedTensor;
            }

            // Mark this node as an output
            var graph = GetOrCreateGraph(_accelerator);
            graph.MarkOutput(_node);

            // Optimize and execute the graph
            graph.Optimize();
            var results = graph.Execute(new Dictionary<string, GpuTensor>());

            if (results.TryGetValue(_node, out var tensor))
            {
                _materializedTensor = tensor;
                _isMaterialized = true;
                return tensor;
            }

            throw new InvalidOperationException("Failed to materialize tensor");
        }

        /// <summary>
        /// Evaluate and copy results to CPU array
        /// </summary>
        public float[] ToArray()
        {
            return Evaluate().ToArray();
        }

        /// <summary>
        /// Evaluate with input bindings
        /// </summary>
        public GpuTensor Evaluate(Dictionary<string, GpuTensor> inputs)
        {
            var graph = GetOrCreateGraph(_accelerator);
            graph.MarkOutput(_node);
            graph.Optimize();
            var results = graph.Execute(inputs);

            if (results.TryGetValue(_node, out var tensor))
            {
                _materializedTensor = tensor;
                _isMaterialized = true;
                return tensor;
            }

            throw new InvalidOperationException("Failed to materialize tensor");
        }

        #endregion

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var status = _isMaterialized ? "materialized" : "lazy";
            return $"LazyTensor([{string.Join(", ", _shape)}], {status})";
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                // Don't dispose the tensor here as it may be shared
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Extension methods for easy lazy tensor creation
    /// </summary>
    public static class LazyTensorExtensions
    {
        /// <summary>
        /// Convert GpuTensor to LazyTensor for deferred execution
        /// </summary>
        public static LazyTensor ToLazy(this GpuTensor tensor, Accelerator accelerator, string? name = null)
        {
            return LazyTensor.FromGpuTensor(accelerator, tensor, name);
        }
    }
}