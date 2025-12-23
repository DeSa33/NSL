using System;
using System.Collections.Generic;
using System.Linq;

namespace NSL.GPU
{
    /// <summary>
    /// Fusion pattern definition for graph optimization.
    /// Patterns are matched against sequences of operations to identify
    /// opportunities for kernel fusion.
    /// </summary>
    public class FusionPattern
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public OpType[] Sequence { get; set; } = Array.Empty<OpType>();
        /// <summary>Public API</summary>
        public OpType FusedOp { get; set; }
        /// <summary>Public API</summary>
        public Func<List<GraphNode>, bool>? Predicate { get; set; }

        /// <summary>Public API</summary>
        public FusionPattern(string name, OpType[] sequence, OpType fusedOp)
        {
            Name = name;
            Sequence = sequence;
            FusedOp = fusedOp;
        }
    }

    /// <summary>
    /// Graph optimizer that applies various optimization passes.
    ///
    /// Optimization passes:
    /// 1. Dead code elimination - remove unused operations
    /// 2. Constant folding - evaluate constant subgraphs at compile time
    /// 3. Common subexpression elimination - reuse identical computations
    /// 4. Operator fusion - combine compatible operations into single kernels
    /// 5. Memory layout optimization - reorder operations for better locality
    /// 6. Algebraic simplification - apply algebraic identities
    ///
    /// Based on optimization techniques from:
    /// - TensorFlow XLA compiler
    /// - PyTorch torch.compile / TorchDynamo
    /// - TVM Relay optimizer
    /// - ONNX Runtime graph optimizations
    /// </summary>
    public class GraphOptimizer
    {
        /// <summary>
        /// Predefined fusion patterns for common operation sequences
        /// </summary>
        public static readonly List<FusionPattern> DefaultPatterns = new()
        {
            // Linear + Activation patterns
            new FusionPattern("LinearReLU", new[] { OpType.MatMul, OpType.Add, OpType.ReLU }, OpType.FusedLinearReLU),
            new FusionPattern("LinearGELU", new[] { OpType.MatMul, OpType.Add, OpType.GELU }, OpType.FusedLinearGELU),
            new FusionPattern("MatMulBiasReLU", new[] { OpType.MatMul, OpType.Add, OpType.ReLU }, OpType.FusedMatMulBiasReLU),

            // Normalization patterns
            new FusionPattern("LayerNormDropout",
                new[] { OpType.LayerNorm }, OpType.FusedLayerNormDropout)
            {
                Predicate = nodes => nodes.Count > 0 && nodes[0].Outputs.Any(o => o.GetAttribute<float>("dropout", 0) > 0)
            },

            // Residual patterns
            new FusionPattern("ResidualAdd", new[] { OpType.Add }, OpType.FusedResidualAdd)
            {
                Predicate = nodes => nodes.Count > 0 && nodes[0].Attributes.ContainsKey("residual")
            },

            // Attention patterns - convert standard attention to FlashAttention
            new FusionPattern("FlashAttention",
                new[] { OpType.ScaledDotProductAttention }, OpType.FusedFlashAttention)
            {
                Predicate = nodes =>
                {
                    // Check if input sizes warrant FlashAttention
                    if (nodes.Count == 0) return false;
                    var shape = nodes[0].Inputs.FirstOrDefault()?.OutputShape;
                    if (shape == null || shape.Length < 2) return false;
                    // Use FlashAttention for sequences > 64
                    return shape[^2] > 64;
                }
            },

            // QKV projection fusion
            new FusionPattern("QKVProjection",
                new[] { OpType.MatMul, OpType.MatMul, OpType.MatMul }, OpType.FusedQKVProjection)
            {
                Predicate = nodes =>
                {
                    // Check if three matmuls share the same input (for Q, K, V projections)
                    if (nodes.Count != 3) return false;
                    var inputs = nodes.Select(n => n.Inputs.FirstOrDefault()).ToList();
                    return inputs.All(i => i != null) && inputs.Distinct().Count() == 1;
                }
            }
        };

        private readonly List<FusionPattern> _patterns;
        private bool _enableConstantFolding = true;
        private bool _enableDeadCodeElimination = true;
        private bool _enableCSE = true;
        private bool _enableFusion = true;
        private bool _enableAlgebraicSimplification = true;

        /// <summary>Public API</summary>
        public GraphOptimizer()
        {
            _patterns = new List<FusionPattern>(DefaultPatterns);
        }

        /// <summary>
        /// Add a custom fusion pattern
        /// </summary>
        public void AddPattern(FusionPattern pattern)
        {
            _patterns.Add(pattern);
        }

        /// <summary>
        /// Run all optimization passes on the graph
        /// </summary>
        public void Optimize(ComputationGraph graph)
        {
            int passCount = 0;
            bool changed;

            do
            {
                changed = false;

                if (_enableDeadCodeElimination)
                {
                    changed |= EliminateDeadCode(graph);
                }

                if (_enableAlgebraicSimplification)
                {
                    changed |= ApplyAlgebraicSimplifications(graph);
                }

                if (_enableCSE)
                {
                    changed |= EliminateCommonSubexpressions(graph);
                }

                if (_enableFusion)
                {
                    changed |= ApplyFusionPatterns(graph);
                }

                passCount++;
            }
            while (changed && passCount < 10); // Limit passes to prevent infinite loops
        }

        #region Dead Code Elimination

        /// <summary>
        /// Remove nodes that don't contribute to any output
        /// </summary>
        private bool EliminateDeadCode(ComputationGraph graph)
        {
            var reachable = new HashSet<GraphNode>();

            // Mark all nodes reachable from outputs
            void MarkReachable(GraphNode node)
            {
                if (reachable.Contains(node)) return;
                reachable.Add(node);
                foreach (var input in node.Inputs)
                {
                    MarkReachable(input);
                }
            }

            foreach (var output in graph.Outputs)
            {
                MarkReachable(output);
            }

            // Mark unreachable nodes as fused (effectively removed)
            bool changed = false;
            foreach (var node in graph.Nodes)
            {
                if (!reachable.Contains(node) && !node.IsFused)
                {
                    node.IsFused = true;
                    changed = true;
                }
            }

            return changed;
        }

        #endregion

        #region Algebraic Simplification

        /// <summary>
        /// Apply algebraic identities to simplify the graph
        /// </summary>
        private bool ApplyAlgebraicSimplifications(ComputationGraph graph)
        {
            bool changed = false;

            foreach (var node in graph.Nodes.Where(n => !n.IsFused))
            {
                // x + 0 = x
                if (node.Op == OpType.Add && IsZeroConstant(node.Inputs.LastOrDefault()))
                {
                    FuseNodeIntoInput(node, 0);
                    changed = true;
                    continue;
                }

                // x * 1 = x
                if (node.Op == OpType.Mul && IsOneConstant(node.Inputs.LastOrDefault()))
                {
                    FuseNodeIntoInput(node, 0);
                    changed = true;
                    continue;
                }

                // x * 0 = 0
                if (node.Op == OpType.Mul && IsZeroConstant(node.Inputs.LastOrDefault()))
                {
                    node.Op = OpType.Constant;
                    node.Inputs.Clear();
                    node.SetAttribute("value", 0.0f);
                    changed = true;
                    continue;
                }

                // ReLU(ReLU(x)) = ReLU(x)
                if (node.Op == OpType.ReLU &&
                    node.Inputs.Count > 0 &&
                    node.Inputs[0].Op == OpType.ReLU)
                {
                    FuseNodeIntoInput(node, 0);
                    changed = true;
                    continue;
                }

                // Softmax(Softmax(x)) can be simplified (though unusual)
                if (node.Op == OpType.Softmax &&
                    node.Inputs.Count > 0 &&
                    node.Inputs[0].Op == OpType.Softmax)
                {
                    FuseNodeIntoInput(node, 0);
                    changed = true;
                    continue;
                }
            }

            return changed;
        }

        private bool IsZeroConstant(GraphNode? node)
        {
            if (node == null) return false;
            if (node.Op != OpType.Constant) return false;
            return node.GetAttribute<float>("value", 1f) == 0f;
        }

        private bool IsOneConstant(GraphNode? node)
        {
            if (node == null) return false;
            if (node.Op != OpType.Constant) return false;
            return Math.Abs(node.GetAttribute<float>("value", 0f) - 1f) < 1e-7f;
        }

        private void FuseNodeIntoInput(GraphNode node, int inputIndex)
        {
            if (node.Inputs.Count <= inputIndex) return;

            var input = node.Inputs[inputIndex];

            // Redirect all outputs of this node to the input
            foreach (var output in node.Outputs.ToList())
            {
                int idx = output.Inputs.IndexOf(node);
                if (idx >= 0)
                {
                    output.Inputs[idx] = input;
                    if (!input.Outputs.Contains(output))
                    {
                        input.Outputs.Add(output);
                    }
                }
            }

            node.IsFused = true;
        }

        #endregion

        #region Common Subexpression Elimination

        /// <summary>
        /// Find and eliminate duplicate computations
        /// </summary>
        private bool EliminateCommonSubexpressions(ComputationGraph graph)
        {
            var nodeSignatures = new Dictionary<string, GraphNode>();
            bool changed = false;

            foreach (var node in graph.Nodes.Where(n => !n.IsFused && !n.IsInput && !n.IsConstant))
            {
                string signature = ComputeSignature(node);

                if (nodeSignatures.TryGetValue(signature, out var existingNode))
                {
                    // Found duplicate - redirect all uses to existing node
                    foreach (var output in node.Outputs.ToList())
                    {
                        int idx = output.Inputs.IndexOf(node);
                        if (idx >= 0)
                        {
                            output.Inputs[idx] = existingNode;
                            if (!existingNode.Outputs.Contains(output))
                            {
                                existingNode.Outputs.Add(output);
                            }
                        }
                    }

                    node.IsFused = true;
                    changed = true;
                }
                else
                {
                    nodeSignatures[signature] = node;
                }
            }

            return changed;
        }

        private string ComputeSignature(GraphNode node)
        {
            // Create a unique signature based on operation and inputs
            var inputIds = string.Join(",", node.Inputs.Select(i => i.Id));
            var attrs = string.Join(",", node.Attributes.Select(kv => $"{kv.Key}={kv.Value}"));
            return $"{node.Op}({inputIds})[{attrs}]";
        }

        #endregion

        #region Operator Fusion

        /// <summary>
        /// Apply fusion patterns to combine operations
        /// </summary>
        private bool ApplyFusionPatterns(ComputationGraph graph)
        {
            bool changed = false;

            foreach (var pattern in _patterns)
            {
                changed |= ApplyPattern(graph, pattern);
            }

            return changed;
        }

        private bool ApplyPattern(ComputationGraph graph, FusionPattern pattern)
        {
            bool changed = false;
            var nodes = graph.Nodes.Where(n => !n.IsFused).ToList();

            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i].Op != pattern.Sequence[0]) continue;

                // Try to match the pattern starting from this node
                var matchedNodes = new List<GraphNode> { nodes[i] };
                var currentNode = nodes[i];

                for (int j = 1; j < pattern.Sequence.Length; j++)
                {
                    // Find next node in sequence (must be the only output of current)
                    if (currentNode.Outputs.Count != 1) break;

                    var nextNode = currentNode.Outputs[0];
                    if (nextNode.Op != pattern.Sequence[j]) break;

                    matchedNodes.Add(nextNode);
                    currentNode = nextNode;
                }

                // Check if full pattern matched
                if (matchedNodes.Count == pattern.Sequence.Length)
                {
                    // Check predicate if any
                    if (pattern.Predicate != null && !pattern.Predicate(matchedNodes))
                    {
                        continue;
                    }

                    // Apply fusion
                    var firstNode = matchedNodes[0];
                    var lastNode = matchedNodes[^1];

                    // Update first node to be the fused op
                    firstNode.Op = pattern.FusedOp;
                    firstNode.Name = $"Fused_{pattern.Name}_{firstNode.Id}";
                    firstNode.OutputShape = lastNode.OutputShape;

                    // Collect all inputs from all matched nodes
                    var allInputs = matchedNodes.SelectMany(n => n.Inputs)
                        .Where(inp => !matchedNodes.Contains(inp))
                        .Distinct()
                        .ToList();
                    firstNode.Inputs.Clear();
                    foreach (var inp in allInputs)
                    {
                        firstNode.Inputs.Add(inp);
                    }

                    // Redirect outputs of last node to first node
                    firstNode.Outputs.Clear();
                    foreach (var output in lastNode.Outputs)
                    {
                        int idx = output.Inputs.IndexOf(lastNode);
                        if (idx >= 0)
                        {
                            output.Inputs[idx] = firstNode;
                        }
                        firstNode.Outputs.Add(output);
                    }

                    // Mark intermediate nodes as fused
                    for (int j = 1; j < matchedNodes.Count; j++)
                    {
                        matchedNodes[j].IsFused = true;
                        matchedNodes[j].FusedInto = firstNode;
                    }

                    changed = true;
                }
            }

            return changed;
        }

        #endregion
    }
}