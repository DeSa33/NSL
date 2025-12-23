using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// Hot loop detection state
    /// </summary>
    public class LoopProfile
    {
        /// <summary>Public API</summary>
        public string LoopId { get; set; } = "";
        /// <summary>Public API</summary>
        public int ExecutionCount { get; set; }
        /// <summary>Public API</summary>
        public long TotalTimeNs { get; set; }
        /// <summary>Public API</summary>
        public long AverageTimeNs => ExecutionCount > 0 ? TotalTimeNs / ExecutionCount : 0;
        /// <summary>Public API</summary>
        public bool IsHot => ExecutionCount >= JITCompiler.HOT_THRESHOLD;
        /// <summary>Public API</summary>
        public bool IsCompiled { get; set; }
        /// <summary>Public API</summary>
        public Func<object[], object?>? CompiledAction { get; set; }

        // Loop body analysis
        /// <summary>Public API</summary>
        public List<OpType> Operations { get; } = new();
        /// <summary>Public API</summary>
        public List<int[]> TensorShapes { get; } = new();
        /// <summary>Public API</summary>
        public bool HasGpuOps { get; set; }
        /// <summary>Public API</summary>
        public bool IsFusible { get; set; }
    }

    /// <summary>
    /// Traced operation in a loop body
    /// </summary>
    public class TracedOp
    {
        /// <summary>Public API</summary>
        public OpType Op { get; set; }
        /// <summary>Public API</summary>
        public int[][] InputShapes { get; set; } = Array.Empty<int[]>();
        /// <summary>Public API</summary>
        public int[] OutputShape { get; set; } = Array.Empty<int>();
        /// <summary>Public API</summary>
        public Dictionary<string, object> Attributes { get; } = new();
    }

    /// <summary>
    /// JIT Compiler for NSL hot loops.
    ///
    /// Detects frequently executed loops and compiles them to optimized
    /// native code or fused GPU kernels.
    ///
    /// Compilation pipeline:
    /// 1. Profiling: Track loop execution counts and timing
    /// 2. Tracing: Capture operations in hot loops
    /// 3. Analysis: Determine if loop is fusible/optimizable
    /// 4. Compilation: Generate optimized code
    ///    - For GPU ops: Create fused kernel
    ///    - For CPU ops: Generate IL code
    /// 5. Caching: Store compiled loops for reuse
    ///
    /// Based on techniques from:
    /// - PyTorch TorchDynamo + TorchInductor
    /// - JAX's jit tracing and XLA compilation
    /// - LuaJIT's trace compiler
    /// - TVM's auto-scheduler
    /// </summary>
    public class JITCompiler
    {
        /// <summary>
        /// Number of executions before a loop is considered "hot"
        /// </summary>
        public const int HOT_THRESHOLD = 100;

        /// <summary>
        /// Number of executions before attempting recompilation
        /// </summary>
        public const int RECOMPILE_THRESHOLD = 10000;

        private readonly Accelerator _accelerator;
        private readonly Dictionary<string, LoopProfile> _profiles = new();
        private readonly Dictionary<string, CompiledLoop> _compiledLoops = new();
        private readonly GraphOptimizer _optimizer = new();

        private bool _enableTracing = true;
        private bool _enableAutoCompilation = true;
        private TracingContext? _currentTrace;

        /// <summary>Public API</summary>
        public JITCompiler(Accelerator accelerator)
        {
            _accelerator = accelerator;
        }

        #region Profiling

        /// <summary>
        /// Profile a loop iteration
        /// </summary>
        public void ProfileLoop(string loopId, long elapsedNs)
        {
            if (!_profiles.TryGetValue(loopId, out var profile))
            {
                profile = new LoopProfile { LoopId = loopId };
                _profiles[loopId] = profile;
            }

            profile.ExecutionCount++;
            profile.TotalTimeNs += elapsedNs;

            // Check if loop became hot
            if (!profile.IsCompiled && profile.IsHot && _enableAutoCompilation)
            {
                TryCompileLoop(loopId);
            }
        }

        /// <summary>
        /// Get profile for a loop
        /// </summary>
        public LoopProfile? GetProfile(string loopId)
        {
            return _profiles.TryGetValue(loopId, out var profile) ? profile : null;
        }

        /// <summary>
        /// Get all hot loops
        /// </summary>
        public IEnumerable<LoopProfile> GetHotLoops()
        {
            return _profiles.Values.Where(p => p.IsHot);
        }

        #endregion

        #region Tracing

        /// <summary>
        /// Start tracing operations in a loop body
        /// </summary>
        public void StartTrace(string loopId)
        {
            if (!_enableTracing) return;

            _currentTrace = new TracingContext
            {
                LoopId = loopId,
                StartTime = Stopwatch.GetTimestamp()
            };
        }

        /// <summary>
        /// Record an operation during tracing
        /// </summary>
        public void TraceOp(OpType op, int[][] inputShapes, int[] outputShape, Dictionary<string, object>? attrs = null)
        {
            if (_currentTrace == null) return;

            _currentTrace.Operations.Add(new TracedOp
            {
                Op = op,
                InputShapes = inputShapes,
                OutputShape = outputShape
            });

            if (IsGpuOp(op))
            {
                _currentTrace.HasGpuOps = true;
            }
        }

        /// <summary>
        /// End tracing and analyze the loop
        /// </summary>
        public void EndTrace()
        {
            if (_currentTrace == null) return;

            var elapsed = Stopwatch.GetTimestamp() - _currentTrace.StartTime;
            var elapsedNs = elapsed * 1_000_000_000 / Stopwatch.Frequency;

            // Update profile with traced info
            if (_profiles.TryGetValue(_currentTrace.LoopId, out var profile))
            {
                if (profile.Operations.Count == 0)
                {
                    profile.Operations.AddRange(_currentTrace.Operations.Select(o => o.Op));
                    profile.HasGpuOps = _currentTrace.HasGpuOps;
                    profile.IsFusible = AnalyzeFusibility(_currentTrace.Operations);
                }
            }

            ProfileLoop(_currentTrace.LoopId, elapsedNs);
            _currentTrace = null;
        }

        private bool AnalyzeFusibility(List<TracedOp> ops)
        {
            if (ops.Count < 2) return false;

            // Check for fusible patterns
            for (int i = 0; i < ops.Count - 1; i++)
            {
                var current = ops[i].Op;
                var next = ops[i + 1].Op;

                // MatMul + Activation is fusible
                if (current == OpType.MatMul && IsActivation(next)) return true;

                // MatMul + Add + Activation is fusible
                if (i < ops.Count - 2 && current == OpType.MatMul &&
                    next == OpType.Add && IsActivation(ops[i + 2].Op)) return true;

                // LayerNorm + anything is usually fusible
                if (current == OpType.LayerNorm) return true;

                // Attention patterns
                if (current == OpType.MatMul && next == OpType.Softmax) return true;
            }

            return false;
        }

        private bool IsActivation(OpType op)
        {
            return op == OpType.ReLU || op == OpType.GELU || op == OpType.Sigmoid ||
                   op == OpType.Tanh || op == OpType.Softmax || op == OpType.Swish ||
                   op == OpType.LeakyReLU;
        }

        private bool IsGpuOp(OpType op)
        {
            return op switch
            {
                OpType.Input or OpType.Constant or OpType.Reshape or OpType.Clone => false,
                _ => true
            };
        }

        #endregion

        #region Compilation

        /// <summary>
        /// Attempt to compile a hot loop
        /// </summary>
        public bool TryCompileLoop(string loopId)
        {
            if (!_profiles.TryGetValue(loopId, out var profile)) return false;
            if (profile.IsCompiled) return true;
            if (profile.Operations.Count == 0) return false;

            try
            {
                var compiled = CompileLoopOps(profile);
                if (compiled != null)
                {
                    _compiledLoops[loopId] = compiled;
                    profile.IsCompiled = true;
                    profile.CompiledAction = compiled.Execute;
                    return true;
                }
            }
            catch (Exception ex)
            {
                // Compilation failed, fall back to interpretation
                Console.WriteLine($"JIT compilation failed for {loopId}: {ex.Message}");
            }

            return false;
        }

        /// <summary>
        /// Execute a potentially compiled loop
        /// </summary>
        public bool TryExecuteCompiled(string loopId, object[] args, out object? result)
        {
            result = null;

            if (_compiledLoops.TryGetValue(loopId, out var compiled))
            {
                result = compiled.Execute(args);
                return true;
            }

            return false;
        }

        private CompiledLoop? CompileLoopOps(LoopProfile profile)
        {
            // Build a computation graph from traced operations
            var graph = new ComputationGraph(_accelerator);

            // Create input nodes for each unique input shape
            var inputs = new Dictionary<string, GraphNode>();
            int inputIdx = 0;

            // Simplified: For now, we create a fused kernel for common patterns
            if (profile.IsFusible && profile.HasGpuOps)
            {
                return CompileFusedGpuLoop(profile);
            }

            // For non-fusible loops, generate IL code
            return CompileILLoop(profile);
        }

        private CompiledLoop CompileFusedGpuLoop(LoopProfile profile)
        {
            var ops = profile.Operations;

            // Detect pattern and create appropriate fused kernel
            if (ContainsPattern(ops, OpType.MatMul, OpType.Add, OpType.ReLU))
            {
                return new CompiledLoop
                {
                    LoopId = profile.LoopId,
                    Type = CompiledLoopType.FusedGpuKernel,
                    Execute = args =>
                    {
                        var fusion = new OperatorFusion(_accelerator);
                        var input = (GpuTensor)args[0];
                        var weight = (GpuTensor)args[1];
                        var bias = (GpuTensor)args[2];
                        return fusion.FusedLinearRelu(input, weight, bias);
                    }
                };
            }

            if (ContainsPattern(ops, OpType.MatMul, OpType.Add, OpType.GELU))
            {
                return new CompiledLoop
                {
                    LoopId = profile.LoopId,
                    Type = CompiledLoopType.FusedGpuKernel,
                    Execute = args =>
                    {
                        var fusion = new OperatorFusion(_accelerator);
                        var input = (GpuTensor)args[0];
                        var weight = (GpuTensor)args[1];
                        var bias = (GpuTensor)args[2];
                        var hpKernels = new HighPerformanceKernels(_accelerator);
                        var linear = hpKernels.TiledMatMul(input, weight);
                        return fusion.FusedBiasGelu(linear, bias);
                    }
                };
            }

            if (ContainsPattern(ops, OpType.ScaledDotProductAttention) ||
                ContainsPattern(ops, OpType.MatMul, OpType.Softmax, OpType.MatMul))
            {
                return new CompiledLoop
                {
                    LoopId = profile.LoopId,
                    Type = CompiledLoopType.FusedGpuKernel,
                    Execute = args =>
                    {
                        var hpKernels = new HighPerformanceKernels(_accelerator);
                        var query = (GpuTensor)args[0];
                        var key = (GpuTensor)args[1];
                        var value = (GpuTensor)args[2];
                        return hpKernels.FlashAttention2(query, key, value);
                    }
                };
            }

            // Default: compile as sequence
            return CompileSequentialGpuOps(profile);
        }

        private CompiledLoop CompileSequentialGpuOps(LoopProfile profile)
        {
            // Create a graph and compile it
            return new CompiledLoop
            {
                LoopId = profile.LoopId,
                Type = CompiledLoopType.GpuGraph,
                Execute = args =>
                {
                    var graph = new ComputationGraph(_accelerator);
                    var executor = new GraphExecutor(_accelerator);

                    // Build graph from operations
                    GraphNode? current = null;
                    int argIdx = 0;

                    foreach (var op in profile.Operations)
                    {
                        var node = new GraphNode(op);

                        // Connect to previous node if not input
                        if (current != null && op != OpType.Input)
                        {
                            node.AddInput(current);
                        }

                        // Bind input tensors
                        if (op == OpType.Input || op == OpType.Constant)
                        {
                            if (argIdx < args.Length && args[argIdx] is GpuTensor tensor)
                            {
                                node.OutputTensor = tensor;
                                node.OutputShape = tensor.Shape;
                                node.IsComputed = true;
                                argIdx++;
                            }
                        }

                        current = node;
                    }

                    // Execute the final node
                    if (current != null)
                    {
                        executor.ExecuteNode(current);
                        return current.OutputTensor;
                    }

                    return null;
                }
            };
        }

        private CompiledLoop? CompileILLoop(LoopProfile profile)
        {
            // For CPU-bound loops, generate IL using System.Reflection.Emit
            // This is more complex and would require careful implementation

            // For now, return null to fall back to interpretation
            return null;
        }

        private bool ContainsPattern(List<OpType> ops, params OpType[] pattern)
        {
            for (int i = 0; i <= ops.Count - pattern.Length; i++)
            {
                bool match = true;
                for (int j = 0; j < pattern.Length; j++)
                {
                    if (ops[i + j] != pattern[j])
                    {
                        match = false;
                        break;
                    }
                }
                if (match) return true;
            }
            return false;
        }

        #endregion

        #region Statistics

        /// <summary>
        /// Get JIT compilation statistics
        /// </summary>
        public JITStats GetStats()
        {
            return new JITStats
            {
                TotalLoopsProfiled = _profiles.Count,
                HotLoops = _profiles.Values.Count(p => p.IsHot),
                CompiledLoops = _compiledLoops.Count,
                TotalExecutions = _profiles.Values.Sum(p => (long)p.ExecutionCount),
                TotalTimeNs = _profiles.Values.Sum(p => p.TotalTimeNs)
            };
        }

        /// <summary>
        /// Clear all profiles and compiled loops
        /// </summary>
        public void Reset()
        {
            _profiles.Clear();
            _compiledLoops.Clear();
        }

        #endregion
    }

    /// <summary>
    /// Type of compiled loop
    /// </summary>
    public enum CompiledLoopType
    {
        FusedGpuKernel,
        GpuGraph,
        ILMethod
    }

    /// <summary>
    /// A compiled loop ready for execution
    /// </summary>
    public class CompiledLoop
    {
        /// <summary>Public API</summary>
        public string LoopId { get; set; } = "";
        /// <summary>Public API</summary>
        public CompiledLoopType Type { get; set; }
        /// <summary>Public API</summary>
        public Func<object[], object?> Execute { get; set; } = _ => null;
    }

    /// <summary>
    /// JIT compilation statistics
    /// </summary>
    public class JITStats
    {
        /// <summary>Public API</summary>
        public int TotalLoopsProfiled { get; set; }
        /// <summary>Public API</summary>
        public int HotLoops { get; set; }
        /// <summary>Public API</summary>
        public int CompiledLoops { get; set; }
        /// <summary>Public API</summary>
        public long TotalExecutions { get; set; }
        /// <summary>Public API</summary>
        public long TotalTimeNs { get; set; }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"JIT Stats: {TotalLoopsProfiled} loops profiled, {HotLoops} hot, {CompiledLoops} compiled, " +
                   $"{TotalExecutions} executions, {TotalTimeNs / 1_000_000.0:F2}ms total time";
        }
    }

    /// <summary>
    /// Context for tracing a loop body
    /// </summary>
    internal class TracingContext
    {
        /// <summary>Public API</summary>
        public string LoopId { get; set; } = "";
        /// <summary>Public API</summary>
        public long StartTime { get; set; }
        /// <summary>Public API</summary>
        public List<TracedOp> Operations { get; } = new();
        /// <summary>Public API</summary>
        public bool HasGpuOps { get; set; }
    }
}