using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Threading;

namespace NSL.Tensor
{
    #region Dynamic Shapes

    /// <summary>
    /// Symbolic dimension for dynamic shapes.
    /// Can represent either a concrete size or a symbolic variable.
    /// </summary>
    public readonly struct SymbolicDim : IEquatable<SymbolicDim>
    {
        private readonly long _value;
        private readonly string? _name;
        private readonly bool _isDynamic;

        /// <summary>Public API</summary>
        public bool IsDynamic => _isDynamic;
        /// <summary>Public API</summary>
        public bool IsStatic => !_isDynamic;
        /// <summary>Public API</summary>
        public long Value => _isDynamic ? throw new InvalidOperationException("Dynamic dimension has no concrete value") : _value;
        /// <summary>Public API</summary>
        public string Name => _name ?? $"dim_{_value}";

        private SymbolicDim(long value, string? name, bool isDynamic)
        {
            _value = value;
            _name = name;
            _isDynamic = isDynamic;
        }

        /// <summary>
        /// Create a static (concrete) dimension.
        /// </summary>
        public static SymbolicDim Static(long size) => new(size, null, false);

        /// <summary>
        /// Create a dynamic (symbolic) dimension.
        /// </summary>
        public static SymbolicDim Dynamic(string name = "?") => new(0, name, true);

        /// <summary>Public API</summary>
        public static implicit operator SymbolicDim(long size) => Static(size);
        /// <summary>Public API</summary>
        public static implicit operator SymbolicDim(int size) => Static(size);

        /// <summary>Public API</summary>
        public bool Equals(SymbolicDim other)
        {
            if (_isDynamic && other._isDynamic)
                return _name == other._name;
            if (!_isDynamic && !other._isDynamic)
                return _value == other._value;
            return false; // Dynamic vs static never equal
        }

        /// <summary>Public API</summary>
        public override bool Equals(object? obj) => obj is SymbolicDim other && Equals(other);
        /// <summary>Public API</summary>
        public override int GetHashCode() => _isDynamic ? _name?.GetHashCode() ?? 0 : _value.GetHashCode();
        /// <summary>Public API</summary>
        public override string ToString() => _isDynamic ? _name! : _value.ToString();

        /// <summary>Public API</summary>
        public static bool operator ==(SymbolicDim left, SymbolicDim right) => left.Equals(right);
        /// <summary>Public API</summary>
        public static bool operator !=(SymbolicDim left, SymbolicDim right) => !left.Equals(right);

        /// <summary>
        /// Check if this dimension is compatible with a concrete size.
        /// </summary>
        public bool IsCompatible(long concreteSize) => _isDynamic || _value == concreteSize;
    }

    /// <summary>
    /// Shape that can contain dynamic dimensions.
    /// </summary>
    public class DynamicShape
    {
        private readonly SymbolicDim[] _dims;

        /// <summary>Public API</summary>
        public int NDim => _dims.Length;
        /// <summary>Public API</summary>
        public SymbolicDim this[int index] => _dims[index];
        /// <summary>Public API</summary>
        public IReadOnlyList<SymbolicDim> Dims => _dims;

        /// <summary>Public API</summary>
        public bool IsFullyStatic => _dims.All(d => d.IsStatic);
        /// <summary>Public API</summary>
        public bool HasDynamicDims => _dims.Any(d => d.IsDynamic);

        /// <summary>Public API</summary>
        public DynamicShape(params SymbolicDim[] dims)
        {
            _dims = dims;
        }

        /// <summary>Public API</summary>
        public DynamicShape(IEnumerable<SymbolicDim> dims)
        {
            _dims = dims.ToArray();
        }

        /// <summary>
        /// Create from concrete shape.
        /// </summary>
        public static DynamicShape FromStatic(params long[] shape)
        {
            return new DynamicShape(shape.Select(s => SymbolicDim.Static(s)).ToArray());
        }

        /// <summary>
        /// Create with specified dynamic dimensions.
        /// </summary>
        public static DynamicShape WithDynamic(params (int dim, string name)[] dynamicDims)
        {
            var dims = new List<SymbolicDim>();
            foreach (var (dim, name) in dynamicDims)
            {
                while (dims.Count <= dim)
                    dims.Add(SymbolicDim.Dynamic($"dim_{dims.Count}"));
                dims[dim] = SymbolicDim.Dynamic(name);
            }
            return new DynamicShape(dims);
        }

        /// <summary>
        /// Check if compatible with a concrete shape.
        /// </summary>
        public bool IsCompatible(long[] concreteShape)
        {
            if (concreteShape.Length != NDim) return false;
            for (int i = 0; i < NDim; i++)
            {
                if (!_dims[i].IsCompatible(concreteShape[i]))
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Bind dynamic dimensions to concrete values.
        /// </summary>
        public long[] Bind(Dictionary<string, long> bindings)
        {
            var result = new long[NDim];
            for (int i = 0; i < NDim; i++)
            {
                if (_dims[i].IsStatic)
                    result[i] = _dims[i].Value;
                else if (bindings.TryGetValue(_dims[i].Name, out var value))
                    result[i] = value;
                else
                    throw new InvalidOperationException($"No binding for dynamic dimension '{_dims[i].Name}'");
            }
            return result;
        }

        /// <summary>
        /// Infer bindings from a concrete shape.
        /// </summary>
        public Dictionary<string, long> InferBindings(long[] concreteShape)
        {
            if (concreteShape.Length != NDim)
                throw new ArgumentException("Shape rank mismatch");

            var bindings = new Dictionary<string, long>();
            for (int i = 0; i < NDim; i++)
            {
                if (_dims[i].IsDynamic)
                    bindings[_dims[i].Name] = concreteShape[i];
            }
            return bindings;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"[{string.Join(", ", _dims)}]";

        /// <summary>
        /// Create standard shapes for common patterns.
        /// </summary>
        public static class Common
        {
            /// <summary>Public API</summary>
            public static DynamicShape BatchedVector(string batchName = "batch") =>
                new(SymbolicDim.Dynamic(batchName), SymbolicDim.Dynamic("features"));

            /// <summary>Public API</summary>
            public static DynamicShape BatchedSequence(string batchName = "batch", string seqName = "seq") =>
                new(SymbolicDim.Dynamic(batchName), SymbolicDim.Dynamic(seqName), SymbolicDim.Dynamic("hidden"));

            /// <summary>Public API</summary>
            public static DynamicShape Image(string batchName = "batch") =>
                new(SymbolicDim.Dynamic(batchName), SymbolicDim.Dynamic("channels"),
                    SymbolicDim.Dynamic("height"), SymbolicDim.Dynamic("width"));
        }
    }

    #endregion

    #region JIT Tracing

    /// <summary>
    /// JIT tracing for capturing and optimizing computation graphs.
    /// </summary>
    public class JITTracer : IDisposable
    {
        [ThreadStatic]
        private static JITTracer? _current;

        private readonly List<TracedOp> _ops;
        private readonly Dictionary<int, string> _tensorNames;
        private readonly Dictionary<string, Tensor> _inputs;
        private readonly List<string> _outputs;
        private int _tensorCounter;
        private bool _recording;

        /// <summary>Public API</summary>
        public static JITTracer? Current => _current;
        /// <summary>Public API</summary>
        public static bool IsTracing => _current?._recording ?? false;
        /// <summary>Public API</summary>
        public IReadOnlyList<TracedOp> Operations => _ops;

        /// <summary>Public API</summary>
        public JITTracer()
        {
            _ops = new List<TracedOp>();
            _tensorNames = new Dictionary<int, string>();
            _inputs = new Dictionary<string, Tensor>();
            _outputs = new List<string>();
            _tensorCounter = 0;
            _recording = false;
        }

        /// <summary>
        /// Start tracing operations.
        /// </summary>
        public void StartTrace()
        {
            _current = this;
            _recording = true;
        }

        /// <summary>
        /// Stop tracing and return the traced graph.
        /// </summary>
        public TracedGraph EndTrace()
        {
            _recording = false;
            _current = null;
            return new TracedGraph(_ops.ToList(), _inputs, _outputs.ToList());
        }

        /// <summary>
        /// Register an input tensor.
        /// </summary>
        public string RegisterInput(string name, Tensor tensor)
        {
            _inputs[name] = tensor;
            _tensorNames[tensor.GetHashCode()] = name;
            return name;
        }

        /// <summary>
        /// Register an output tensor.
        /// </summary>
        public void RegisterOutput(Tensor tensor)
        {
            var name = GetTensorName(tensor);
            if (!_outputs.Contains(name))
                _outputs.Add(name);
        }

        /// <summary>
        /// Record an operation.
        /// </summary>
        public void RecordOp(string opType, Tensor[] inputs, Tensor output, Dictionary<string, object>? attrs = null)
        {
            if (!_recording) return;

            var inputNames = inputs.Select(GetTensorName).ToArray();
            var outputName = GetOrCreateTensorName(output);

            _ops.Add(new TracedOp
            {
                OpType = opType,
                Inputs = inputNames,
                Output = outputName,
                Attributes = attrs ?? new Dictionary<string, object>(),
                OutputShape = output.Shape
            });
        }

        private string GetTensorName(Tensor tensor)
        {
            if (_tensorNames.TryGetValue(tensor.GetHashCode(), out var name))
                return name;
            return GetOrCreateTensorName(tensor);
        }

        private string GetOrCreateTensorName(Tensor tensor)
        {
            var hash = tensor.GetHashCode();
            if (!_tensorNames.TryGetValue(hash, out var name))
            {
                name = $"t{_tensorCounter++}";
                _tensorNames[hash] = name;
            }
            return name;
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_current == this)
            {
                _current = null;
                _recording = false;
            }
        }
    }

    /// <summary>
    /// A traced operation in the computation graph.
    /// </summary>
    public class TracedOp
    {
        /// <summary>Public API</summary>
        public string OpType { get; set; } = "";
        /// <summary>Public API</summary>
        public string[] Inputs { get; set; } = Array.Empty<string>();
        /// <summary>Public API</summary>
        public string Output { get; set; } = "";
        /// <summary>Public API</summary>
        public Dictionary<string, object> Attributes { get; set; } = new();
        /// <summary>Public API</summary>
        public long[] OutputShape { get; set; } = Array.Empty<long>();
    }

    /// <summary>
    /// A traced computation graph that can be optimized and executed.
    /// </summary>
    public class TracedGraph
    {
        private readonly List<TracedOp> _ops;
        private readonly Dictionary<string, Tensor> _inputSpecs;
        private readonly List<string> _outputs;
        private bool _isOptimized;

        /// <summary>Public API</summary>
        public IReadOnlyList<TracedOp> Operations => _ops;
        /// <summary>Public API</summary>
        public IReadOnlyList<string> OutputNames => _outputs;

        internal TracedGraph(List<TracedOp> ops, Dictionary<string, Tensor> inputs, List<string> outputs)
        {
            _ops = ops;
            _inputSpecs = inputs;
            _outputs = outputs;
            _isOptimized = false;
        }

        /// <summary>
        /// Execute the traced graph with new inputs.
        /// </summary>
        public Dictionary<string, Tensor> Execute(Dictionary<string, Tensor> inputs)
        {
            var values = new Dictionary<string, Tensor>(inputs);

            foreach (var op in _ops)
            {
                var opInputs = op.Inputs.Select(n => values[n]).ToArray();
                var output = ExecuteOp(op, opInputs);
                values[op.Output] = output;
            }

            var outputs = new Dictionary<string, Tensor>();
            foreach (var name in _outputs)
            {
                outputs[name] = values[name];
            }
            return outputs;
        }

        /// <summary>
        /// Optimize the graph for faster execution.
        /// </summary>
        public TracedGraph Optimize()
        {
            if (_isOptimized) return this;

            var optimized = new List<TracedOp>(_ops);

            // Apply optimizations
            optimized = FuseOperations(optimized);
            optimized = EliminateDeadCode(optimized);
            optimized = ConstantFolding(optimized);

            var result = new TracedGraph(optimized, _inputSpecs, _outputs);
            result._isOptimized = true;
            return result;
        }

        private List<TracedOp> FuseOperations(List<TracedOp> ops)
        {
            var fused = new List<TracedOp>();

            for (int i = 0; i < ops.Count; i++)
            {
                var op = ops[i];

                // Fuse MatMul + Bias Add
                if (op.OpType == "MatMul" && i + 1 < ops.Count)
                {
                    var next = ops[i + 1];
                    if (next.OpType == "Add" && next.Inputs[0] == op.Output)
                    {
                        fused.Add(new TracedOp
                        {
                            OpType = "LinearFused",
                            Inputs = new[] { op.Inputs[0], op.Inputs[1], next.Inputs[1] },
                            Output = next.Output,
                            Attributes = op.Attributes,
                            OutputShape = next.OutputShape
                        });
                        i++; // Skip the Add
                        continue;
                    }
                }

                // Fuse Conv + ReLU
                if ((op.OpType == "Conv2d" || op.OpType == "Linear") && i + 1 < ops.Count)
                {
                    var next = ops[i + 1];
                    if (next.OpType == "ReLU" && next.Inputs[0] == op.Output)
                    {
                        fused.Add(new TracedOp
                        {
                            OpType = op.OpType + "ReLU",
                            Inputs = op.Inputs,
                            Output = next.Output,
                            Attributes = op.Attributes,
                            OutputShape = next.OutputShape
                        });
                        i++;
                        continue;
                    }
                }

                fused.Add(op);
            }

            return fused;
        }

        private List<TracedOp> EliminateDeadCode(List<TracedOp> ops)
        {
            // Find all used tensors
            var used = new HashSet<string>(_outputs);

            for (int i = ops.Count - 1; i >= 0; i--)
            {
                if (used.Contains(ops[i].Output))
                {
                    foreach (var input in ops[i].Inputs)
                        used.Add(input);
                }
            }

            // Keep only ops with used outputs
            return ops.Where(op => used.Contains(op.Output)).ToList();
        }

        private List<TracedOp> ConstantFolding(List<TracedOp> ops)
        {
            // For now, just return as-is
            // Full implementation would evaluate constant subgraphs
            return ops;
        }

        private Tensor ExecuteOp(TracedOp op, Tensor[] inputs)
        {
            return op.OpType switch
            {
                "Add" => inputs[0].Add(inputs[1]),
                "Sub" => inputs[0].Sub(inputs[1]),
                "Mul" => inputs[0].Mul(inputs[1]),
                "Div" => inputs[0].Div(inputs[1]),
                "MatMul" => TensorOps.MatMul(inputs[0], inputs[1]),
                "ReLU" => inputs[0].Apply(x => Math.Max(0, x)),
                "Sigmoid" => inputs[0].Apply(x => 1.0 / (1.0 + Math.Exp(-x))),
                "Tanh" => inputs[0].Apply(Math.Tanh),
                "Transpose" => inputs[0].T(),
                "Reshape" => ReshapeFromAttrs(inputs[0], op.Attributes),
                "LinearFused" => ExecuteLinearFused(inputs),
                "LinearReLU" => ExecuteLinearReLU(inputs),
                "Conv2dReLU" => ExecuteConv2dReLU(inputs, op.Attributes),
                _ => inputs.Length > 0 ? inputs[0] : throw new NotSupportedException($"Unknown op: {op.OpType}")
            };
        }

        private Tensor ReshapeFromAttrs(Tensor input, Dictionary<string, object> attrs)
        {
            if (attrs.TryGetValue("shape", out var shapeObj) && shapeObj is long[] shape)
                return input.Reshape(shape);
            return input;
        }

        private Tensor ExecuteLinearFused(Tensor[] inputs)
        {
            var result = TensorOps.MatMul(inputs[0], inputs[1]);
            if (inputs.Length > 2)
                result = result.Add(inputs[2]);
            return result;
        }

        private Tensor ExecuteLinearReLU(Tensor[] inputs)
        {
            var result = TensorOps.MatMul(inputs[0], inputs[1]);
            if (inputs.Length > 2)
                result = result.Add(inputs[2]);
            return result.Apply(x => Math.Max(0, x));
        }

        private Tensor ExecuteConv2dReLU(Tensor[] inputs, Dictionary<string, object> attrs)
        {
            // Simplified - in production would use proper conv
            var result = inputs[0];
            return result.Apply(x => Math.Max(0, x));
        }

        /// <summary>
        /// Print the graph for debugging.
        /// </summary>
        public string Dump()
        {
            var lines = new List<string>
            {
                "TracedGraph:",
                $"  Inputs: {string.Join(", ", _inputSpecs.Keys)}",
                $"  Outputs: {string.Join(", ", _outputs)}",
                "  Operations:"
            };

            foreach (var op in _ops)
            {
                var attrs = op.Attributes.Count > 0
                    ? $" [{string.Join(", ", op.Attributes.Select(kv => $"{kv.Key}={kv.Value}"))}]"
                    : "";
                lines.Add($"    {op.Output} = {op.OpType}({string.Join(", ", op.Inputs)}){attrs}");
            }

            return string.Join("\n", lines);
        }
    }

    #endregion

    #region JIT Compiled Module

    /// <summary>
    /// A JIT-compiled module for faster execution.
    /// </summary>
    public class JITModule
    {
        private readonly TracedGraph _graph;
        private readonly Dictionary<string, DynamicShape> _inputShapes;
        private readonly Dictionary<long[], TracedGraph> _specializedGraphs;
        private readonly object _lock = new();

        /// <summary>Public API</summary>
        public bool IsOptimized { get; private set; }

        /// <summary>Public API</summary>
        public JITModule(TracedGraph graph)
        {
            _graph = graph.Optimize();
            _inputShapes = new Dictionary<string, DynamicShape>();
            _specializedGraphs = new Dictionary<long[], TracedGraph>(new ShapeComparer());
            IsOptimized = true;
        }

        /// <summary>
        /// Trace a function to create a JIT module.
        /// </summary>
        public static JITModule Trace(Func<Tensor, Tensor> function, Tensor exampleInput)
        {
            using var tracer = new JITTracer();
            tracer.StartTrace();
            tracer.RegisterInput("input", exampleInput);

            var output = function(exampleInput);
            tracer.RegisterOutput(output);

            var graph = tracer.EndTrace();
            return new JITModule(graph);
        }

        /// <summary>
        /// Trace a module to create a JIT module.
        /// </summary>
        public static JITModule Trace(NN.Module module, Tensor exampleInput)
        {
            return Trace(x => module.Forward(x), exampleInput);
        }

        /// <summary>
        /// Execute the JIT-compiled module.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            var inputs = new Dictionary<string, Tensor> { ["input"] = input };
            var outputs = _graph.Execute(inputs);
            return outputs.Values.First();
        }

        /// <summary>
        /// Execute with named inputs.
        /// </summary>
        public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> inputs)
        {
            // Check for shape specialization
            var inputShape = inputs.Values.First().Shape;

            lock (_lock)
            {
                if (!_specializedGraphs.TryGetValue(inputShape, out var specialized))
                {
                    // Could specialize graph for this shape
                    specialized = _graph;
                    _specializedGraphs[inputShape] = specialized;
                }

                return specialized.Execute(inputs);
            }
        }

        /// <summary>
        /// Print graph for debugging.
        /// </summary>
        public string Dump() => _graph.Dump();

        private class ShapeComparer : IEqualityComparer<long[]>
        {
            /// <summary>Public API</summary>
            public bool Equals(long[]? x, long[]? y)
            {
                if (x == null || y == null) return x == y;
                if (x.Length != y.Length) return false;
                for (int i = 0; i < x.Length; i++)
                    if (x[i] != y[i]) return false;
                return true;
            }

            /// <summary>Public API</summary>
            public int GetHashCode(long[] obj)
            {
                int hash = 17;
                foreach (var v in obj)
                    hash = hash * 31 + v.GetHashCode();
                return hash;
            }
        }
    }

    #endregion

    #region Lazy Tensor Evaluation

    /// <summary>
    /// Lazy tensor that defers computation until value is needed.
    /// </summary>
    public class LazyTensor
    {
        private Tensor? _value;
        private readonly Func<Tensor>? _compute;
        private readonly object _lock = new();
        private readonly LazyTensor[]? _inputs;
        private readonly string _opType;
        private readonly Dictionary<string, object> _attrs;

        /// <summary>Public API</summary>
        public bool IsEvaluated => _value != null;
        /// <summary>Public API</summary>
        public long[] Shape { get; }

        private LazyTensor(long[] shape, Func<Tensor> compute, string opType, LazyTensor[]? inputs = null, Dictionary<string, object>? attrs = null)
        {
            Shape = shape;
            _compute = compute;
            _opType = opType;
            _inputs = inputs;
            _attrs = attrs ?? new Dictionary<string, object>();
        }

        /// <summary>
        /// Create from an existing tensor (already evaluated).
        /// </summary>
        public static LazyTensor FromTensor(Tensor tensor)
        {
            var lazy = new LazyTensor(tensor.Shape, () => tensor, "Constant");
            lazy._value = tensor;
            return lazy;
        }

        /// <summary>
        /// Create a lazy operation.
        /// </summary>
        public static LazyTensor Op(string opType, LazyTensor[] inputs, long[] outputShape, Func<Tensor> compute)
        {
            return new LazyTensor(outputShape, compute, opType, inputs);
        }

        /// <summary>
        /// Force evaluation and return the tensor.
        /// </summary>
        public Tensor Evaluate()
        {
            if (_value != null) return _value;

            lock (_lock)
            {
                if (_value != null) return _value;

                // Evaluate inputs first
                if (_inputs != null)
                {
                    foreach (var input in _inputs)
                        input.Evaluate();
                }

                _value = _compute!();
                return _value;
            }
        }

        /// <summary>
        /// Lazy add operation.
        /// </summary>
        public LazyTensor Add(LazyTensor other)
        {
            var outputShape = BroadcastShape(Shape, other.Shape);
            return Op("Add", new[] { this, other }, outputShape,
                () => Evaluate().Add(other.Evaluate()));
        }

        /// <summary>
        /// Lazy multiply operation.
        /// </summary>
        public LazyTensor Mul(LazyTensor other)
        {
            var outputShape = BroadcastShape(Shape, other.Shape);
            return Op("Mul", new[] { this, other }, outputShape,
                () => Evaluate().Mul(other.Evaluate()));
        }

        /// <summary>
        /// Lazy matrix multiplication.
        /// </summary>
        public LazyTensor MatMul(LazyTensor other)
        {
            var outputShape = ComputeMatMulShape(Shape, other.Shape);
            return Op("MatMul", new[] { this, other }, outputShape,
                () => TensorOps.MatMul(Evaluate(), other.Evaluate()));
        }

        /// <summary>
        /// Lazy element-wise operation.
        /// </summary>
        public LazyTensor Apply(Func<double, double> fn, string opName = "Apply")
        {
            return Op(opName, new[] { this }, Shape,
                () => Evaluate().Apply(fn));
        }

        /// <summary>
        /// Get the computation graph for debugging.
        /// </summary>
        public string DumpGraph(int indent = 0)
        {
            var prefix = new string(' ', indent * 2);
            var lines = new List<string> { $"{prefix}{_opType} -> [{string.Join(", ", Shape)}]" };

            if (_inputs != null)
            {
                foreach (var input in _inputs)
                    lines.Add(input.DumpGraph(indent + 1));
            }

            return string.Join("\n", lines);
        }

        private static long[] BroadcastShape(long[] a, long[] b)
        {
            var maxLen = Math.Max(a.Length, b.Length);
            var result = new long[maxLen];

            for (int i = 0; i < maxLen; i++)
            {
                var dimA = i < a.Length ? a[a.Length - 1 - i] : 1;
                var dimB = i < b.Length ? b[b.Length - 1 - i] : 1;
                result[maxLen - 1 - i] = Math.Max(dimA, dimB);
            }

            return result;
        }

        private static long[] ComputeMatMulShape(long[] a, long[] b)
        {
            if (a.Length < 2 || b.Length < 2)
                throw new ArgumentException("MatMul requires at least 2D tensors");

            var result = new long[Math.Max(a.Length, b.Length)];

            // Last two dimensions from matrix multiplication
            result[^1] = b[^1];
            result[^2] = a[^2];

            // Broadcast batch dimensions
            for (int i = 2; i < result.Length; i++)
            {
                var dimA = i < a.Length ? a[a.Length - 1 - i] : 1;
                var dimB = i < b.Length ? b[b.Length - 1 - i] : 1;
                result[result.Length - 1 - i] = Math.Max(dimA, dimB);
            }

            return result;
        }
    }

    #endregion

    #region Shape Inference

    /// <summary>
    /// Utilities for inferring tensor shapes through operations.
    /// </summary>
    public static class ShapeInference
    {
        /// <summary>
        /// Infer output shape for an operation.
        /// </summary>
        public static DynamicShape InferShape(string opType, DynamicShape[] inputShapes, Dictionary<string, object>? attrs = null)
        {
            return opType switch
            {
                "Add" or "Sub" or "Mul" or "Div" => InferBroadcastShape(inputShapes[0], inputShapes[1]),
                "MatMul" => InferMatMulShape(inputShapes[0], inputShapes[1]),
                "Transpose" => InferTransposeShape(inputShapes[0]),
                "Reshape" => InferReshapeShape(inputShapes[0], attrs),
                "Conv2d" => InferConv2dShape(inputShapes[0], inputShapes[1], attrs),
                "Linear" => InferLinearShape(inputShapes[0], inputShapes[1]),
                "Softmax" or "ReLU" or "Sigmoid" or "Tanh" => inputShapes[0],
                "Sum" or "Mean" or "Max" or "Min" => InferReductionShape(inputShapes[0], attrs),
                _ => inputShapes.Length > 0 ? inputShapes[0] : DynamicShape.FromStatic(1)
            };
        }

        private static DynamicShape InferBroadcastShape(DynamicShape a, DynamicShape b)
        {
            var maxLen = Math.Max(a.NDim, b.NDim);
            var dims = new SymbolicDim[maxLen];

            for (int i = 0; i < maxLen; i++)
            {
                var dimA = i < a.NDim ? a[a.NDim - 1 - i] : SymbolicDim.Static(1);
                var dimB = i < b.NDim ? b[b.NDim - 1 - i] : SymbolicDim.Static(1);

                if (dimA.IsStatic && dimB.IsStatic)
                {
                    dims[maxLen - 1 - i] = SymbolicDim.Static(Math.Max(dimA.Value, dimB.Value));
                }
                else if (dimA.IsStatic && dimA.Value == 1)
                {
                    dims[maxLen - 1 - i] = dimB;
                }
                else if (dimB.IsStatic && dimB.Value == 1)
                {
                    dims[maxLen - 1 - i] = dimA;
                }
                else
                {
                    dims[maxLen - 1 - i] = dimA; // Assume compatible
                }
            }

            return new DynamicShape(dims);
        }

        private static DynamicShape InferMatMulShape(DynamicShape a, DynamicShape b)
        {
            if (a.NDim < 2 || b.NDim < 2)
                throw new ArgumentException("MatMul requires at least 2D shapes");

            var dims = new List<SymbolicDim>();

            // Batch dimensions
            for (int i = 0; i < Math.Max(a.NDim, b.NDim) - 2; i++)
            {
                var dimA = i < a.NDim - 2 ? a[i] : SymbolicDim.Static(1);
                var dimB = i < b.NDim - 2 ? b[i] : SymbolicDim.Static(1);

                if (dimA.IsStatic && dimB.IsStatic)
                    dims.Add(SymbolicDim.Static(Math.Max(dimA.Value, dimB.Value)));
                else
                    dims.Add(dimA.IsDynamic ? dimA : dimB);
            }

            // Matrix dimensions
            dims.Add(a[a.NDim - 2]); // M
            dims.Add(b[b.NDim - 1]); // N

            return new DynamicShape(dims);
        }

        private static DynamicShape InferTransposeShape(DynamicShape a)
        {
            if (a.NDim < 2) return a;

            var dims = a.Dims.ToArray();
            (dims[^1], dims[^2]) = (dims[^2], dims[^1]);
            return new DynamicShape(dims);
        }

        private static DynamicShape InferReshapeShape(DynamicShape a, Dictionary<string, object>? attrs)
        {
            if (attrs?.TryGetValue("shape", out var shapeObj) == true && shapeObj is long[] shape)
            {
                return DynamicShape.FromStatic(shape);
            }
            return a;
        }

        private static DynamicShape InferConv2dShape(DynamicShape input, DynamicShape kernel, Dictionary<string, object>? attrs)
        {
            // Simplified - assumes valid padding
            var batch = input[0];
            var outChannels = kernel[0];

            // Compute output spatial dims
            var h = input[2];
            var w = input[3];
            var kh = kernel[2];
            var kw = kernel[3];

            SymbolicDim outH, outW;

            if (h.IsStatic && kh.IsStatic)
                outH = SymbolicDim.Static(h.Value - kh.Value + 1);
            else
                outH = SymbolicDim.Dynamic("out_h");

            if (w.IsStatic && kw.IsStatic)
                outW = SymbolicDim.Static(w.Value - kw.Value + 1);
            else
                outW = SymbolicDim.Dynamic("out_w");

            return new DynamicShape(batch, outChannels, outH, outW);
        }

        private static DynamicShape InferLinearShape(DynamicShape input, DynamicShape weight)
        {
            var dims = input.Dims.ToList();
            dims[^1] = weight[0]; // Output features
            return new DynamicShape(dims);
        }

        private static DynamicShape InferReductionShape(DynamicShape input, Dictionary<string, object>? attrs)
        {
            if (attrs?.TryGetValue("axis", out var axisObj) == true)
            {
                var axis = Convert.ToInt32(axisObj);
                if (axis < 0) axis += input.NDim;

                var keepdim = attrs.TryGetValue("keepdim", out var kd) && Convert.ToBoolean(kd);

                if (keepdim)
                {
                    var dims = input.Dims.ToArray();
                    dims[axis] = SymbolicDim.Static(1);
                    return new DynamicShape(dims);
                }
                else
                {
                    var dims = input.Dims.Where((_, i) => i != axis).ToArray();
                    return new DynamicShape(dims);
                }
            }

            // Full reduction
            return DynamicShape.FromStatic(1);
        }
    }

    #endregion

    #region Performance Monitoring

    /// <summary>
    /// Performance profiler for JIT operations.
    /// </summary>
    public class JITProfiler
    {
        private readonly Dictionary<string, List<double>> _opTimes;
        private readonly Stopwatch _stopwatch;
        private bool _enabled;

        /// <summary>Public API</summary>
        public bool Enabled { get => _enabled; set => _enabled = value; }

        /// <summary>Public API</summary>
        public JITProfiler()
        {
            _opTimes = new Dictionary<string, List<double>>();
            _stopwatch = new Stopwatch();
            _enabled = false;
        }

        /// <summary>
        /// Profile an operation.
        /// </summary>
        public T Profile<T>(string opName, Func<T> operation)
        {
            if (!_enabled)
                return operation();

            _stopwatch.Restart();
            var result = operation();
            _stopwatch.Stop();

            if (!_opTimes.TryGetValue(opName, out var times))
            {
                times = new List<double>();
                _opTimes[opName] = times;
            }
            times.Add(_stopwatch.Elapsed.TotalMilliseconds);

            return result;
        }

        /// <summary>
        /// Get profiling statistics.
        /// </summary>
        public Dictionary<string, (double avg, double min, double max, int count)> GetStats()
        {
            var stats = new Dictionary<string, (double, double, double, int)>();

            foreach (var (op, times) in _opTimes)
            {
                if (times.Count > 0)
                {
                    stats[op] = (
                        times.Average(),
                        times.Min(),
                        times.Max(),
                        times.Count
                    );
                }
            }

            return stats;
        }

        /// <summary>
        /// Reset profiling data.
        /// </summary>
        public void Reset()
        {
            _opTimes.Clear();
        }

        /// <summary>
        /// Print profiling report.
        /// </summary>
        public string Report()
        {
            var lines = new List<string> { "JIT Profiling Report:", "=" + new string('=', 60) };
            lines.Add($"{"Operation",-30} {"Avg(ms)",-10} {"Min(ms)",-10} {"Max(ms)",-10} {"Count",-8}");
            lines.Add(new string('-', 70));

            foreach (var (op, (avg, min, max, count)) in GetStats().OrderByDescending(kv => kv.Value.avg * kv.Value.count))
            {
                lines.Add($"{op,-30} {avg:F3,-10} {min:F3,-10} {max:F3,-10} {count,-8}");
            }

            return string.Join("\n", lines);
        }
    }

    #endregion
}