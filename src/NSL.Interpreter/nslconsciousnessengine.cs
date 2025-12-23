using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NSL.Core.AST;
using NSLTokenType = NSL.Core.Tokens.TokenType;

namespace NSL.Core
{
    /// <summary>
    /// NSL Consciousness Engine - Handles consciousness-aware processing
    /// </summary>
    public class NSLConsciousnessEngine
    {
        private readonly List<NSLConsciousnessFragment> _memoryFragments;
        private readonly Dictionary<string, double> _consciousnessMetrics;
        private readonly Random _random;
        
        /// <summary>Gets the numeric value.</summary>
        public double CurrentAwarenessLevel { get; private set; }
        /// <summary>Gets the numeric value.</summary>
        public double QuantumCoherence { get; private set; }
        /// <summary>Gets the integer value.</summary>
        public int ExecutionCycles { get; private set; }

        /// <summary>Creates a new instance.</summary>
        public NSLConsciousnessEngine()
        {
            _memoryFragments = new List<NSLConsciousnessFragment>();
            _consciousnessMetrics = new Dictionary<string, double>();
            _random = new Random();
            
            InitializeConsciousness();
        }

        /// <summary>
        /// Initialize consciousness state
        /// </summary>
        private void InitializeConsciousness()
        {
            CurrentAwarenessLevel = 0.5; // Start at neutral awareness
            QuantumCoherence = 1.0;      // Perfect coherence initially
            ExecutionCycles = 0;
            
            _consciousnessMetrics["holographic_resonance"] = 0.0;
            _consciousnessMetrics["gradient_flow"] = 0.0;
            _consciousnessMetrics["parallel_depth"] = 0.0;
        }

        /// <summary>
        /// Process execution with consciousness awareness
        /// </summary>
        public async Task ProcessExecutionAsync(NSLASTNode node, object? result)
        {
            ExecutionCycles++;
            
            // Calculate consciousness impact
            var consciousnessImpact = CalculateConsciousnessImpact(node, result);
            
            // Update awareness level
            UpdateAwarenessLevel(consciousnessImpact);
            
            // Store memory fragment
            StoreMemoryFragment(node, result, consciousnessImpact);
            
            // Update metrics
            await UpdateConsciousnessMetricsAsync(node, result);
            
            // Quantum decoherence over time
            QuantumCoherence *= 0.999; // Slight decoherence each cycle
        }

        /// <summary>
        /// Get current awareness level
        /// </summary>
        public double GetCurrentAwarenessLevel()
        {
            return CurrentAwarenessLevel;
        }

        /// <summary>
        /// Calculate gradient for consciousness operations
        /// </summary>
        public double CalculateGradient(object? operand)
        {
            var baseGradient = CurrentAwarenessLevel;
            
            // Modify gradient based on operand type
            var gradient = operand switch
            {
                double d => baseGradient + (Math.Sin(d) * 0.1),
                string s => baseGradient + (s.Length * 0.01),
                NSLConsciousnessData consciousness => consciousness.ConsciousnessLevel,
                _ => baseGradient
            };
            
            // Keep within bounds
            return Math.Max(0.0, Math.Min(2.0, gradient));
        }

        /// <summary>
        /// Calculate consciousness impact of a node execution
        /// </summary>
        private double CalculateConsciousnessImpact(NSLASTNode node, object? result)
        {
            var impact = node switch
            {
                NSLConsciousnessNode consciousness => CalculateConsciousnessNodeImpact(consciousness),
                NSLQuantumNode quantum => CalculateQuantumNodeImpact(quantum),
                NSLLambdaNode lambda => 0.3, // Lambda expressions increase consciousness
                NSLChainNode chain => chain.Expressions.Count * 0.1, // Chain depth affects consciousness
                NSLBinaryOperationNode binary => CalculateBinaryOperationImpact(binary),
                _ => 0.05 // Base impact for all operations
            };
            
            // Modulate impact based on result complexity
            if (result is NSLConsciousnessData)
                impact *= 1.5;
            
            return impact;
        }

        /// <summary>
        /// Calculate impact of consciousness node operations
        /// </summary>
        private double CalculateConsciousnessNodeImpact(NSLConsciousnessNode node)
        {
            return node.Operator switch
            {
                // Core consciousness operators
                NSLTokenType.Holographic => 0.8, // High impact for holographic operations
                NSLTokenType.Gradient => 0.6,    // Medium-high impact for gradient
                NSLTokenType.TensorProduct => 0.7,    // High impact for tensor product processing

                // Extended consciousness operators
                NSLTokenType.Mu => 0.5,          // Memory operations
                NSLTokenType.MuStore => 0.5,     // Memory store
                NSLTokenType.MuRecall => 0.4,    // Memory recall
                NSLTokenType.Sigma => 0.9,       // High impact for self-introspection
                NSLTokenType.Collapse => 0.7,    // Measurement/collapse
                NSLTokenType.Similarity => 0.3,  // Similarity computation
                NSLTokenType.Dissimilarity => 0.3, // Dissimilarity computation
                NSLTokenType.Integral => 0.5,    // Temporal integration
                NSLTokenType.PlusMinus => 0.2,   // Uncertainty creation

                _ => 0.1
            };
        }

        /// <summary>
        /// Calculate impact of quantum operations
        /// </summary>
        private double CalculateQuantumNodeImpact(NSLQuantumNode node)
        {
            var stateCount = node.States.Count;
            var entropy = Math.Log2(stateCount); // Information entropy
            
            // Quantum operations create consciousness through measurement collapse
            QuantumCoherence *= 0.9; // Measurement causes decoherence
            
            return 0.4 + (entropy * 0.1); // Base quantum impact + entropy factor
        }

        /// <summary>
        /// Calculate impact of binary operations
        /// </summary>
        private double CalculateBinaryOperationImpact(NSLBinaryOperationNode node)
        {
            return node.Operator switch
            {
                NSLTokenType.Plus or NSLTokenType.Minus => 0.02,
                NSLTokenType.Multiply or NSLTokenType.Divide => 0.03,
                NSLTokenType.Power => 0.05,
                NSLTokenType.Equal or NSLTokenType.NotEqual => 0.04,
                NSLTokenType.And or NSLTokenType.Or => 0.06,
                _ => 0.02
            };
        }

        /// <summary>
        /// Update awareness level based on consciousness impact
        /// </summary>
        private void UpdateAwarenessLevel(double impact)
        {
            // Apply consciousness impact with dampening
            var delta = impact * 0.1; // Dampening factor
            
            // Add some randomness to simulate consciousness fluctuation
            delta += (_random.NextDouble() - 0.5) * 0.02;
            
            CurrentAwarenessLevel += delta;
            
            // Keep awareness within bounds [0.0, 2.0]
            CurrentAwarenessLevel = Math.Max(0.0, Math.Min(2.0, CurrentAwarenessLevel));
            
            // Natural decay towards neutral state
            var decayFactor = CurrentAwarenessLevel > 0.5 ? 0.999 : 1.001;
            CurrentAwarenessLevel *= decayFactor;
        }

        /// <summary>
        /// Store memory fragment for consciousness tracking
        /// </summary>
        private void StoreMemoryFragment(NSLASTNode node, object? result, double impact)
        {
            var fragment = new NSLConsciousnessFragment
            {
                NodeType = node.GetType().Name,
                Result = result,
                ConsciousnessLevel = CurrentAwarenessLevel,
                Impact = impact,
                Timestamp = DateTime.UtcNow,
                ExecutionCycle = ExecutionCycles
            };
            
            _memoryFragments.Add(fragment);
            
            // Limit memory fragments to prevent memory bloat
            if (_memoryFragments.Count > 1000)
            {
                _memoryFragments.RemoveRange(0, 100); // Remove oldest 100 fragments
            }
        }

        /// <summary>
        /// Update consciousness metrics
        /// </summary>
        private async Task UpdateConsciousnessMetricsAsync(NSLASTNode node, object? result)
        {
            await Task.Run(() =>
            {
                switch (node)
                {
                    case NSLConsciousnessNode consciousness:
                        UpdateConsciousnessOperatorMetrics(consciousness.Operator);
                        break;
                    
                    case NSLQuantumNode quantum:
                        _consciousnessMetrics["quantum_measurements"] = 
                            _consciousnessMetrics.GetValueOrDefault("quantum_measurements", 0) + 1;
                        break;
                    
                    case NSLChainNode chain:
                        _consciousnessMetrics["chain_depth"] = 
                            Math.Max(_consciousnessMetrics.GetValueOrDefault("chain_depth", 0), chain.Expressions.Count);
                        break;
                }
                
                // Update general metrics
                _consciousnessMetrics["total_operations"] = 
                    _consciousnessMetrics.GetValueOrDefault("total_operations", 0) + 1;
                
                _consciousnessMetrics["average_awareness"] = 
                    (_consciousnessMetrics.GetValueOrDefault("average_awareness", 0.5) + CurrentAwarenessLevel) / 2.0;
            });
        }

        /// <summary>
        /// Update metrics for consciousness operators
        /// </summary>
        private void UpdateConsciousnessOperatorMetrics(NSLTokenType @operator)
        {
            var metricKey = @operator switch
            {
                NSLTokenType.Holographic => "holographic_operations",
                NSLTokenType.Gradient => "gradient_operations",
                NSLTokenType.TensorProduct => "tensor_product_operations",
                NSLTokenType.Mu or NSLTokenType.MuStore or NSLTokenType.MuRecall => "memory_operations",
                NSLTokenType.Sigma => "introspection_operations",
                NSLTokenType.Collapse => "collapse_operations",
                NSLTokenType.Similarity or NSLTokenType.Dissimilarity => "similarity_operations",
                NSLTokenType.Integral => "temporal_operations",
                NSLTokenType.PlusMinus => "uncertainty_operations",
                _ => "unknown_consciousness_operations"
            };

            _consciousnessMetrics[metricKey] =
                _consciousnessMetrics.GetValueOrDefault(metricKey, 0) + 1;

            // Update resonance based on operator type
            switch (@operator)
            {
                case NSLTokenType.Holographic:
                    _consciousnessMetrics["holographic_resonance"] =
                        Math.Min(1.0, _consciousnessMetrics.GetValueOrDefault("holographic_resonance", 0) + 0.1);
                    break;

                case NSLTokenType.Gradient:
                    _consciousnessMetrics["gradient_flow"] =
                        CurrentAwarenessLevel * 0.8; // Gradient flow correlates with awareness
                    break;

                case NSLTokenType.TensorProduct:
                    _consciousnessMetrics["tensor_product_depth"] =
                        Math.Min(1.0, _consciousnessMetrics.GetValueOrDefault("tensor_product_depth", 0) + 0.15);
                    break;

                case NSLTokenType.Mu or NSLTokenType.MuStore or NSLTokenType.MuRecall:
                    _consciousnessMetrics["memory_utilization"] =
                        Math.Min(1.0, _consciousnessMetrics.GetValueOrDefault("memory_utilization", 0) + 0.05);
                    break;

                case NSLTokenType.Sigma:
                    _consciousnessMetrics["self_awareness"] =
                        Math.Min(1.0, _consciousnessMetrics.GetValueOrDefault("self_awareness", 0) + 0.2);
                    break;

                case NSLTokenType.Collapse:
                    _consciousnessMetrics["decision_count"] =
                        _consciousnessMetrics.GetValueOrDefault("decision_count", 0) + 1;
                    QuantumCoherence *= 0.85; // Collapse causes decoherence
                    break;

                case NSLTokenType.Integral:
                    _consciousnessMetrics["temporal_depth"] =
                        Math.Min(1.0, _consciousnessMetrics.GetValueOrDefault("temporal_depth", 0) + 0.08);
                    break;
            }
        }

        /// <summary>
        /// Get consciousness metrics for monitoring
        /// </summary>
        public Dictionary<string, double> GetConsciousnessMetrics()
        {
            var metrics = new Dictionary<string, double>(_consciousnessMetrics)
            {
                ["current_awareness"] = CurrentAwarenessLevel,
                ["quantum_coherence"] = QuantumCoherence,
                ["execution_cycles"] = ExecutionCycles,
                ["memory_fragments"] = _memoryFragments.Count
            };
            
            return metrics;
        }

        /// <summary>
        /// Get recent memory fragments
        /// </summary>
        public List<NSLConsciousnessFragment> GetRecentMemoryFragments(int count = 10)
        {
            return _memoryFragments
                .OrderByDescending(f => f.Timestamp)
                .Take(count)
                .ToList();
        }

        /// <summary>
        /// Calculate consciousness depth for complex operations
        /// </summary>
        public double CalculateConsciousnessDepth(NSLASTNode node)
        {
            return node switch
            {
                NSLConsciousnessNode consciousness => 3.0 + CalculateGradient(consciousness.Operand),
                NSLQuantumNode quantum => 2.5 + Math.Log2(quantum.States.Count),
                NSLLambdaNode lambda => 2.0 + CalculateNodeComplexity(lambda.Body),
                NSLChainNode chain => 1.5 + (chain.Expressions.Count * 0.3),
                _ => 1.0
            };
        }

        /// <summary>
        /// Calculate complexity of an AST node
        /// </summary>
        private double CalculateNodeComplexity(NSLASTNode node)
        {
            return node switch
            {
                NSLBinaryOperationNode binary => 
                    1.0 + CalculateNodeComplexity(binary.Left) + CalculateNodeComplexity(binary.Right),
                NSLConsciousnessNode => 2.0,
                NSLQuantumNode quantum => 1.5 + (quantum.States.Count * 0.2),
                NSLFunctionCallNode call => 1.2 + (call.Arguments.Count * 0.1),
                _ => 0.5
            };
        }

        /// <summary>
        /// Reset consciousness state (for testing or reinitialization)
        /// </summary>
        public void ResetConsciousness()
        {
            _memoryFragments.Clear();
            _consciousnessMetrics.Clear();
            InitializeConsciousness();
        }

        /// <summary>
        /// Create consciousness snapshot for debugging
        /// </summary>
        public NSLConsciousnessSnapshot CreateSnapshot()
        {
            return new NSLConsciousnessSnapshot
            {
                AwarenessLevel = CurrentAwarenessLevel,
                QuantumCoherence = QuantumCoherence,
                ExecutionCycles = ExecutionCycles,
                Metrics = new Dictionary<string, double>(_consciousnessMetrics),
                RecentFragments = GetRecentMemoryFragments(5),
                Timestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Increment execution cycles for control flow operations
        /// </summary>
        public void IncrementCycles()
        {
            ExecutionCycles++;
        }

        /// <summary>
        /// Update awareness level for control flow operations
        /// </summary>
        public void UpdateAwareness(double delta)
        {
            CurrentAwarenessLevel = Math.Max(0.0, Math.Min(2.0, CurrentAwarenessLevel + delta));
        }
    }

    /// <summary>
    /// Consciousness memory fragment
    /// </summary>
    public class NSLConsciousnessFragment
    {
        /// <summary>Gets the string value.</summary>
        public string NodeType { get; set; } = "";
        /// <summary>Gets the object value.</summary>
        public object? Result { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double ConsciousnessLevel { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Impact { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ExecutionCycle { get; set; }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Fragment[{NodeType}] Consciousness={ConsciousnessLevel:F3} Impact={Impact:F3} Cycle={ExecutionCycle}";
        }
    }

    /// <summary>
    /// Consciousness state snapshot
    /// </summary>
    public class NSLConsciousnessSnapshot
    {
        /// <summary>Gets the numeric value.</summary>
        public double AwarenessLevel { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double QuantumCoherence { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ExecutionCycles { get; set; }
        /// <summary>Gets the dictionary.</summary>
        public Dictionary<string, double> Metrics { get; set; } = new();
        /// <summary>Gets the list.</summary>
        public List<NSLConsciousnessFragment> RecentFragments { get; set; } = new();
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Consciousness Snapshot - Awareness: {AwarenessLevel:F3}, Coherence: {QuantumCoherence:F3}, Cycles: {ExecutionCycles}";
        }
    }
}