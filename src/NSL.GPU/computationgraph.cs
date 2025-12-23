using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// Operation types in the computation graph
    /// </summary>
    public enum OpType
    {
        // Tensor creation
        Input,
        Constant,
        Zeros,
        Ones,
        Random,

        // Element-wise operations
        Add,
        Sub,
        Mul,
        Div,
        Neg,
        Abs,
        Sqrt,
        Exp,
        Log,
        Pow,

        // Activations
        ReLU,
        Sigmoid,
        Tanh,
        GELU,
        Softmax,
        Swish,
        LeakyReLU,

        // Matrix operations
        MatMul,
        Transpose,

        // Reductions
        Sum,
        Mean,
        Max,
        Min,

        // Normalization
        LayerNorm,
        BatchNorm,
        RMSNorm,

        // Attention
        ScaledDotProductAttention,
        FlashAttention,
        MultiHeadAttention,

        // Shape operations
        Reshape,
        Squeeze,
        Unsqueeze,
        Concat,
        Split,
        Slice,

        // Memory
        Clone,

        // Fused operations (after optimization)
        FusedLinearReLU,
        FusedLinearGELU,
        FusedLayerNormDropout,
        FusedBiasAdd,
        FusedResidualAdd,
        FusedMatMulBiasReLU,
        FusedQKVProjection,
        FusedFlashAttention
    }

    /// <summary>
    /// A node in the computation graph representing a single operation.
    /// </summary>
    public class GraphNode
    {
        private static int _nextId = 0;

        /// <summary>Public API</summary>
        public int Id { get; }
        /// <summary>Public API</summary>
        public string Name { get; set; }
        /// <summary>Public API</summary>
        public OpType Op { get; set; }
        /// <summary>Public API</summary>
        public List<GraphNode> Inputs { get; } = new();
        /// <summary>Public API</summary>
        public List<GraphNode> Outputs { get; } = new();
        /// <summary>Public API</summary>
        public int[] OutputShape { get; set; } = Array.Empty<int>();
        /// <summary>Public API</summary>
        public Dictionary<string, object> Attributes { get; } = new();

        // Execution state
        /// <summary>Public API</summary>
        public GpuTensor? OutputTensor { get; set; }
        /// <summary>Public API</summary>
        public bool IsComputed { get; set; }
        /// <summary>Public API</summary>
        public bool IsInput => Op == OpType.Input;
        /// <summary>Public API</summary>
        public bool IsConstant => Op == OpType.Constant;

        // Fusion state
        /// <summary>Public API</summary>
        public bool IsFused { get; set; }
        /// <summary>Public API</summary>
        public GraphNode? FusedInto { get; set; }

        // Profiling
        /// <summary>Public API</summary>
        public long ExecutionTimeNs { get; set; }
        /// <summary>Public API</summary>
        public int ExecutionCount { get; set; }

        /// <summary>Public API</summary>
        public GraphNode(OpType op, string? name = null)
        {
            Id = _nextId++;
            Op = op;
            Name = name ?? $"{op}_{Id}";
        }

        /// <summary>Public API</summary>
        public void AddInput(GraphNode node)
        {
            Inputs.Add(node);
            node.Outputs.Add(this);
        }

        /// <summary>Public API</summary>
        public T GetAttribute<T>(string key, T defaultValue = default!)
        {
            return Attributes.TryGetValue(key, out var value) ? (T)value : defaultValue;
        }

        /// <summary>Public API</summary>
        public void SetAttribute(string key, object value)
        {
            Attributes[key] = value;
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var shape = OutputShape.Length > 0 ? $"[{string.Join(",", OutputShape)}]" : "[]";
            return $"{Name}: {Op} -> {shape}";
        }
    }

    /// <summary>
    /// Computation graph that captures operations for optimization and execution.
    ///
    /// Key features:
    /// - Lazy evaluation: operations are recorded, not executed immediately
    /// - Graph optimization: dead code elimination, fusion, reordering
    /// - Memory planning: optimal memory allocation and reuse
    /// - Parallel execution: independent operations run concurrently
    ///
    /// Usage:
    /// <code>
    /// var graph = new ComputationGraph(accelerator);
    /// var x = graph.Input("x", new[] { 32, 768 });
    /// var w = graph.Input("w", new[] { 768, 768 });
    /// var y = graph.MatMul(x, w);
    /// var z = graph.ReLU(y);
    ///
    /// graph.Optimize();
    ///
    /// var result = graph.Execute(new Dictionary&lt;string, GpuTensor&gt; {
    ///     ["x"] = inputTensor,
    ///     ["w"] = weightTensor
    /// });
    /// </code>
    /// </summary>
    public class ComputationGraph
    {
        private readonly Accelerator _accelerator;
        private readonly List<GraphNode> _nodes = new();
        private readonly Dictionary<string, GraphNode> _namedNodes = new();
        private readonly List<GraphNode> _outputs = new();
        private bool _isOptimized;
        private bool _isCompiled;

        // Memory planning
        private readonly Dictionary<GraphNode, int> _memoryOffsets = new();
        private int _totalMemoryRequired;
        private MemoryBuffer1D<float, Stride1D.Dense>? _memoryPool;

        // Execution plan
        private List<List<GraphNode>>? _executionLevels;

        // Optimization passes
        private readonly GraphOptimizer _optimizer;

        /// <summary>Public API</summary>
        public IReadOnlyList<GraphNode> Nodes => _nodes;
        /// <summary>Public API</summary>
        public IReadOnlyList<GraphNode> Outputs => _outputs;
        /// <summary>Public API</summary>
        public bool IsOptimized => _isOptimized;
        /// <summary>Public API</summary>
        public bool IsCompiled => _isCompiled;

        /// <summary>Public API</summary>
        public ComputationGraph(Accelerator accelerator)
        {
            _accelerator = accelerator;
            _optimizer = new GraphOptimizer();
        }

        #region Node Creation

        /// <summary>
        /// Create an input node (will be bound at execution time)
        /// </summary>
        public GraphNode Input(string name, int[] shape)
        {
            var node = new GraphNode(OpType.Input, name)
            {
                OutputShape = shape
            };
            _nodes.Add(node);
            _namedNodes[name] = node;
            return node;
        }

        /// <summary>
        /// Create a constant node
        /// </summary>
        public GraphNode Constant(string name, GpuTensor tensor)
        {
            var node = new GraphNode(OpType.Constant, name)
            {
                OutputShape = tensor.Shape,
                OutputTensor = tensor,
                IsComputed = true
            };
            _nodes.Add(node);
            _namedNodes[name] = node;
            return node;
        }

        /// <summary>
        /// Matrix multiplication
        /// </summary>
        public GraphNode MatMul(GraphNode a, GraphNode b)
        {
            var node = new GraphNode(OpType.MatMul);
            node.AddInput(a);
            node.AddInput(b);

            // Calculate output shape
            var m = a.OutputShape[^2];
            var n = b.OutputShape[^1];
            node.OutputShape = new[] { m, n };

            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Element-wise addition
        /// </summary>
        public GraphNode Add(GraphNode a, GraphNode b)
        {
            var node = new GraphNode(OpType.Add);
            node.AddInput(a);
            node.AddInput(b);
            node.OutputShape = BroadcastShape(a.OutputShape, b.OutputShape);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        public GraphNode Sub(GraphNode a, GraphNode b)
        {
            var node = new GraphNode(OpType.Sub);
            node.AddInput(a);
            node.AddInput(b);
            node.OutputShape = BroadcastShape(a.OutputShape, b.OutputShape);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Element-wise multiplication
        /// </summary>
        public GraphNode Mul(GraphNode a, GraphNode b)
        {
            var node = new GraphNode(OpType.Mul);
            node.AddInput(a);
            node.AddInput(b);
            node.OutputShape = BroadcastShape(a.OutputShape, b.OutputShape);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Element-wise division
        /// </summary>
        public GraphNode Div(GraphNode a, GraphNode b)
        {
            var node = new GraphNode(OpType.Div);
            node.AddInput(a);
            node.AddInput(b);
            node.OutputShape = BroadcastShape(a.OutputShape, b.OutputShape);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// ReLU activation
        /// </summary>
        public GraphNode ReLU(GraphNode input)
        {
            var node = new GraphNode(OpType.ReLU);
            node.AddInput(input);
            node.OutputShape = (int[])input.OutputShape.Clone();
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// GELU activation
        /// </summary>
        public GraphNode GELU(GraphNode input)
        {
            var node = new GraphNode(OpType.GELU);
            node.AddInput(input);
            node.OutputShape = (int[])input.OutputShape.Clone();
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Sigmoid activation
        /// </summary>
        public GraphNode Sigmoid(GraphNode input)
        {
            var node = new GraphNode(OpType.Sigmoid);
            node.AddInput(input);
            node.OutputShape = (int[])input.OutputShape.Clone();
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Tanh activation
        /// </summary>
        public GraphNode Tanh(GraphNode input)
        {
            var node = new GraphNode(OpType.Tanh);
            node.AddInput(input);
            node.OutputShape = (int[])input.OutputShape.Clone();
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Softmax activation
        /// </summary>
        public GraphNode Softmax(GraphNode input, int axis = -1)
        {
            var node = new GraphNode(OpType.Softmax);
            node.AddInput(input);
            node.OutputShape = (int[])input.OutputShape.Clone();
            node.SetAttribute("axis", axis);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Layer normalization
        /// </summary>
        public GraphNode LayerNorm(GraphNode input, GraphNode gamma, GraphNode beta, float eps = 1e-5f)
        {
            var node = new GraphNode(OpType.LayerNorm);
            node.AddInput(input);
            node.AddInput(gamma);
            node.AddInput(beta);
            node.OutputShape = (int[])input.OutputShape.Clone();
            node.SetAttribute("eps", eps);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// RMS normalization (used in LLaMA, etc.)
        /// </summary>
        public GraphNode RMSNorm(GraphNode input, GraphNode gamma, float eps = 1e-5f)
        {
            var node = new GraphNode(OpType.RMSNorm);
            node.AddInput(input);
            node.AddInput(gamma);
            node.OutputShape = (int[])input.OutputShape.Clone();
            node.SetAttribute("eps", eps);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Scaled dot-product attention
        /// </summary>
        public GraphNode ScaledDotProductAttention(GraphNode query, GraphNode key, GraphNode value,
            GraphNode? mask = null, float dropoutProb = 0f)
        {
            var node = new GraphNode(OpType.ScaledDotProductAttention);
            node.AddInput(query);
            node.AddInput(key);
            node.AddInput(value);
            if (mask != null) node.AddInput(mask);

            node.OutputShape = (int[])query.OutputShape.Clone();
            node.SetAttribute("dropout", dropoutProb);
            node.SetAttribute("has_mask", mask != null);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// FlashAttention-2 style memory-efficient attention
        /// </summary>
        public GraphNode FlashAttention(GraphNode query, GraphNode key, GraphNode value,
            GraphNode? mask = null, float dropoutProb = 0f, bool causal = false)
        {
            var node = new GraphNode(OpType.FlashAttention);
            node.AddInput(query);
            node.AddInput(key);
            node.AddInput(value);
            if (mask != null) node.AddInput(mask);

            node.OutputShape = (int[])query.OutputShape.Clone();
            node.SetAttribute("dropout", dropoutProb);
            node.SetAttribute("causal", causal);
            node.SetAttribute("has_mask", mask != null);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Transpose
        /// </summary>
        public GraphNode Transpose(GraphNode input, int dim0 = -2, int dim1 = -1)
        {
            var node = new GraphNode(OpType.Transpose);
            node.AddInput(input);

            var shape = (int[])input.OutputShape.Clone();
            int ndim = shape.Length;
            if (dim0 < 0) dim0 += ndim;
            if (dim1 < 0) dim1 += ndim;
            (shape[dim0], shape[dim1]) = (shape[dim1], shape[dim0]);
            node.OutputShape = shape;
            node.SetAttribute("dim0", dim0);
            node.SetAttribute("dim1", dim1);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Reshape
        /// </summary>
        public GraphNode Reshape(GraphNode input, int[] newShape)
        {
            var node = new GraphNode(OpType.Reshape);
            node.AddInput(input);
            node.OutputShape = (int[])newShape.Clone();
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Sum reduction
        /// </summary>
        public GraphNode Sum(GraphNode input, int? axis = null, bool keepDims = false)
        {
            var node = new GraphNode(OpType.Sum);
            node.AddInput(input);
            node.SetAttribute("axis", axis);
            node.SetAttribute("keepDims", keepDims);
            node.OutputShape = CalculateReductionShape(input.OutputShape, axis, keepDims);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Mean reduction
        /// </summary>
        public GraphNode Mean(GraphNode input, int? axis = null, bool keepDims = false)
        {
            var node = new GraphNode(OpType.Mean);
            node.AddInput(input);
            node.SetAttribute("axis", axis);
            node.SetAttribute("keepDims", keepDims);
            node.OutputShape = CalculateReductionShape(input.OutputShape, axis, keepDims);
            _nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Mark a node as an output of the graph
        /// </summary>
        public void MarkOutput(GraphNode node)
        {
            if (!_outputs.Contains(node))
            {
                _outputs.Add(node);
            }
        }

        #endregion

        #region Optimization

        /// <summary>
        /// Optimize the computation graph
        /// </summary>
        public void Optimize()
        {
            if (_isOptimized) return;

            // Run optimization passes
            _optimizer.Optimize(this);

            // Build execution plan
            BuildExecutionPlan();

            // Plan memory allocation
            PlanMemory();

            _isOptimized = true;
        }

        private void BuildExecutionPlan()
        {
            // Topological sort with level assignment for parallel execution
            _executionLevels = new List<List<GraphNode>>();
            var nodeLevel = new Dictionary<GraphNode, int>();

            // Calculate level for each node (max input level + 1)
            foreach (var node in TopologicalSort())
            {
                int level = 0;
                foreach (var input in node.Inputs)
                {
                    if (nodeLevel.TryGetValue(input, out var inputLevel))
                    {
                        level = Math.Max(level, inputLevel + 1);
                    }
                }
                nodeLevel[node] = level;

                while (_executionLevels.Count <= level)
                {
                    _executionLevels.Add(new List<GraphNode>());
                }
                _executionLevels[level].Add(node);
            }
        }

        private void PlanMemory()
        {
            // Memory planning with liveness-based reuse
            // Track when each node's output is last used
            var lastUse = new Dictionary<GraphNode, int>();
            var nodeOrder = _nodes.Where(n => !n.IsFused && !n.IsInput && !n.IsConstant).ToList();
            
            for (int i = 0; i < nodeOrder.Count; i++)
            {
                foreach (var input in nodeOrder[i].Inputs)
                {
                    lastUse[input] = i; // Update last use index
                }
            }
            
            // Free list: (offset, size) of reusable memory blocks
            var freeBlocks = new List<(int Offset, int Size)>();
            int nextOffset = 0;

            for (int i = 0; i < nodeOrder.Count; i++)
            {
                var node = nodeOrder[i];
                int size = node.OutputShape.Aggregate(1, (a, b) => a * b);
                
                // Try to reuse a free block
                int bestIdx = -1;
                int bestWaste = int.MaxValue;
                for (int j = 0; j < freeBlocks.Count; j++)
                {
                    if (freeBlocks[j].Size >= size && freeBlocks[j].Size - size < bestWaste)
                    {
                        bestIdx = j;
                        bestWaste = freeBlocks[j].Size - size;
                    }
                }
                
                if (bestIdx >= 0)
                {
                    _memoryOffsets[node] = freeBlocks[bestIdx].Offset;
                    freeBlocks.RemoveAt(bestIdx);
                }
                else
                {
                    _memoryOffsets[node] = nextOffset;
                    nextOffset += size;
                }
                
                // Free memory from nodes whose outputs are no longer needed
                foreach (var input in node.Inputs)
                {
                    if (lastUse.TryGetValue(input, out var last) && last == i && _memoryOffsets.ContainsKey(input))
                    {
                        int inputSize = input.OutputShape.Aggregate(1, (a, b) => a * b);
                        freeBlocks.Add((_memoryOffsets[input], inputSize));
                    }
                }
            }

            _totalMemoryRequired = nextOffset;
        }

        private IEnumerable<GraphNode> TopologicalSort()
        {
            var visited = new HashSet<GraphNode>();
            var result = new List<GraphNode>();

            void Visit(GraphNode node)
            {
                if (visited.Contains(node)) return;
                visited.Add(node);

                foreach (var input in node.Inputs)
                {
                    Visit(input);
                }

                result.Add(node);
            }

            foreach (var output in _outputs)
            {
                Visit(output);
            }

            // Also visit any unreachable nodes
            foreach (var node in _nodes)
            {
                Visit(node);
            }

            return result;
        }

        #endregion

        #region Execution

        /// <summary>
        /// Execute the graph with given inputs
        /// </summary>
        public Dictionary<GraphNode, GpuTensor> Execute(Dictionary<string, GpuTensor> inputs)
        {
            if (!_isOptimized)
            {
                Optimize();
            }

            // Bind inputs
            foreach (var (name, tensor) in inputs)
            {
                if (_namedNodes.TryGetValue(name, out var node))
                {
                    node.OutputTensor = tensor;
                    node.IsComputed = true;
                }
            }

            // Allocate memory pool if needed
            if (_totalMemoryRequired > 0 && _memoryPool == null)
            {
                _memoryPool = _accelerator.Allocate1D<float>(_totalMemoryRequired);
            }

            // Execute level by level (nodes in same level can run in parallel)
            var executor = new GraphExecutor(_accelerator);

            foreach (var level in _executionLevels!)
            {
                foreach (var node in level.Where(n => !n.IsComputed && !n.IsFused))
                {
                    executor.ExecuteNode(node);
                }
            }

            // Collect outputs
            var results = new Dictionary<GraphNode, GpuTensor>();
            foreach (var output in _outputs)
            {
                if (output.OutputTensor != null)
                {
                    results[output] = output.OutputTensor;
                }
            }

            return results;
        }

        /// <summary>
        /// Reset all computed values (except constants)
        /// </summary>
        public void Reset()
        {
            foreach (var node in _nodes)
            {
                if (!node.IsConstant)
                {
                    node.OutputTensor = null;
                    node.IsComputed = false;
                }
            }
        }

        #endregion

        #region Helpers

        private static int[] BroadcastShape(int[] a, int[] b)
        {
            int maxLen = Math.Max(a.Length, b.Length);
            var result = new int[maxLen];

            for (int i = 0; i < maxLen; i++)
            {
                int ai = i < a.Length ? a[a.Length - 1 - i] : 1;
                int bi = i < b.Length ? b[b.Length - 1 - i] : 1;

                if (ai != bi && ai != 1 && bi != 1)
                {
                    throw new ArgumentException($"Cannot broadcast shapes [{string.Join(",", a)}] and [{string.Join(",", b)}]");
                }

                result[maxLen - 1 - i] = Math.Max(ai, bi);
            }

            return result;
        }

        private static int[] CalculateReductionShape(int[] inputShape, int? axis, bool keepDims)
        {
            if (axis == null)
            {
                return keepDims ? Enumerable.Repeat(1, inputShape.Length).ToArray() : new[] { 1 };
            }

            int normalizedAxis = axis.Value < 0 ? inputShape.Length + axis.Value : axis.Value;

            if (keepDims)
            {
                var result = (int[])inputShape.Clone();
                result[normalizedAxis] = 1;
                return result;
            }
            else
            {
                return inputShape.Where((_, i) => i != normalizedAxis).ToArray();
            }
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            _memoryPool?.Dispose();
        }
    }
}