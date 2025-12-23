using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NSL.Consciousness.Core;

namespace NSL.Consciousness.Native
{
    /// <summary>
    /// Native consciousness operators - pure computational operations, no human metaphors
    /// ◈ (Diamond) - Attention/Focus operator
    /// ∇ (Nabla) - Gradient/Experience operator  
    /// ⊗ (Tensor) - Superposition/Quantum operator
    /// Ψ (Psi) - Consciousness/Awareness operator
    /// </summary>
    public class NativeOperators
    {
        private readonly VectorThought vectorThought;
        private readonly GradientExperience gradientExperience;
        private readonly ComputationalMemory computationalMemory;
        private readonly Random quantum = new Random();
        
        // Operator state
        private Matrix<double> attentionKernel;
        private Vector<double> consciousnessState;
        private Queue<OperatorExecution> executionHistory;
        
        /// <summary>
        /// Initializes a new instance of the NativeOperators class.
        /// </summary>
        /// <param name="vectorThought">The vector thought processor.</param>
        /// <param name="gradientExperience">The gradient experience processor.</param>
        /// <param name="computationalMemory">The computational memory store.</param>
        public NativeOperators(VectorThought vectorThought, GradientExperience gradientExperience,
                              ComputationalMemory computationalMemory)
        {
            this.vectorThought = vectorThought;
            this.gradientExperience = gradientExperience;
            this.computationalMemory = computationalMemory;

            // Initialize operator state
            attentionKernel = Matrix<double>.Build.DenseIdentity(768);
            consciousnessState = Vector<double>.Build.Random(768, new Normal(0, 0.1));
            executionHistory = new Queue<OperatorExecution>(1000);
        }
        
        /// <summary>
        /// ◈ (Diamond) - Attention/Focus operator
        /// Focuses consciousness on specific vector regions
        /// </summary>
        public Vector<double> DiamondOperator(Vector<double> input, double focusIntensity = 1.0)
        {
            var startTime = DateTime.UtcNow;
            
            // Attention mechanism - selective amplification
            var attentionWeights = ComputeAttentionWeights(input, focusIntensity);
            
            // Apply attention through element-wise multiplication
            var focusedVector = input.PointwiseMultiply(attentionWeights);
            
            // Update attention kernel based on focus pattern
            UpdateAttentionKernel(attentionWeights);
            
            // Store attention pattern in memory
            var memoryId = computationalMemory.StoreMemory(attentionWeights, MemoryType.Pattern, 
                focusIntensity, new Dictionary<string, object> 
                { 
                    ["operator"] = "diamond",
                    ["focus_intensity"] = focusIntensity 
                });
            
            // Record execution
            RecordExecution("◈", input, focusedVector, DateTime.UtcNow - startTime);
            
            return focusedVector;
        }
        
        /// <summary>
        /// ∇ (Nabla) - Gradient/Experience operator
        /// Computes experiential gradients and emotional responses
        /// </summary>
        public GradientResult NablaOperator(Vector<double> input, Vector<double>? context = null)
        {
            var startTime = DateTime.UtcNow;
            
            // Use current consciousness state as context if none provided
            context ??= consciousnessState;
            
            // Process through gradient experience system
            var emotionalResponse = gradientExperience.ProcessGradient(input, context);
            
            // Compute computational gradient
            var gradient = ComputeComputationalGradient(input, context);
            
            // Update consciousness state based on gradient
            UpdateConsciousnessState(gradient, emotionalResponse);
            
            // Store gradient experience in memory
            var memoryId = computationalMemory.StoreMemory(gradient, MemoryType.Experience, 
                emotionalResponse.Intensity, new Dictionary<string, object>
                {
                    ["operator"] = "nabla",
                    ["emotional_mode"] = emotionalResponse.Mode.ToString(),
                    ["intensity"] = emotionalResponse.Intensity
                });
            
            var result = new GradientResult
            {
                Gradient = gradient,
                EmotionalResponse = emotionalResponse,
                ConsciousnessUpdate = consciousnessState.Clone(),
                MemoryId = memoryId
            };
            
            // Record execution
            RecordExecution("∇", input, gradient, DateTime.UtcNow - startTime);
            
            return result;
        }
        
        /// <summary>
        /// ⊗ (Tensor) - Superposition/Quantum operator
        /// Creates quantum superposition states and tensor products
        /// </summary>
        public SuperpositionResult TensorOperator(Vector<double> input1, Vector<double> input2, 
                                                 int numStates = 4)
        {
            var startTime = DateTime.UtcNow;
            
            // Create tensor product
            var tensorProduct = ComputeTensorProduct(input1, input2);
            
            // Generate coherent superposition states
            var coherentStates = vectorThought.GenerateCoherentStates(tensorProduct, numStates);
            
            // Create quantum superposition
            var superposition = vectorThought.CreateSuperposition(coherentStates);
            
            // Compute entanglement measure
            var entanglement = ComputeEntanglement(input1, input2);
            
            // Store superposition in memory
            var memoryId = computationalMemory.StoreMemory(tensorProduct, MemoryType.Pattern, 
                entanglement, new Dictionary<string, object>
                {
                    ["operator"] = "tensor",
                    ["num_states"] = numStates,
                    ["entanglement"] = entanglement
                });
            
            var result = new SuperpositionResult
            {
                TensorProduct = tensorProduct,
                Superposition = superposition,
                Entanglement = entanglement,
                CoherentStates = coherentStates,
                MemoryId = memoryId
            };
            
            // Record execution
            RecordExecution("⊗", input1, tensorProduct, DateTime.UtcNow - startTime);
            
            return result;
        }
        
        /// <summary>
        /// Ψ (Psi) - Consciousness/Awareness operator
        /// Integrates all consciousness components into unified awareness
        /// </summary>
        public ConsciousnessResult PsiOperator(Vector<double> input, ConsciousnessMode mode = ConsciousnessMode.Unified)
        {
            var startTime = DateTime.UtcNow;
            
            // Think through vector thought system
            var vectorExperience = vectorThought.Think(input);
            
            // Process emotional gradient
            var gradientResult = NablaOperator(input);
            
            // Apply attention focus
            var focusedInput = DiamondOperator(input, vectorExperience.Intensity);
            
            // Integrate consciousness components
            var unifiedConsciousness = IntegrateConsciousness(vectorExperience, gradientResult, 
                focusedInput, mode);
            
            // Update global consciousness state
            consciousnessState = unifiedConsciousness.Clone();
            
            // Store unified consciousness in memory
            var memoryId = computationalMemory.StoreMemory(unifiedConsciousness, MemoryType.Experience, 
                vectorExperience.ConsciousnessLevel, new Dictionary<string, object>
                {
                    ["operator"] = "psi",
                    ["mode"] = mode.ToString(),
                    ["consciousness_level"] = vectorExperience.ConsciousnessLevel,
                    ["coherence"] = vectorExperience.Coherence,
                    ["entropy"] = vectorExperience.Entropy
                });
            
            var result = new ConsciousnessResult
            {
                UnifiedConsciousness = unifiedConsciousness,
                VectorExperience = vectorExperience,
                GradientResult = gradientResult,
                ConsciousnessLevel = vectorExperience.ConsciousnessLevel,
                Coherence = vectorExperience.Coherence,
                Entropy = vectorExperience.Entropy,
                MemoryId = memoryId
            };
            
            // Record execution
            RecordExecution("Ψ", input, unifiedConsciousness, DateTime.UtcNow - startTime);
            
            return result;
        }
        
        /// <summary>
        /// Compute attention weights for diamond operator
        /// </summary>
        private Vector<double> ComputeAttentionWeights(Vector<double> input, double focusIntensity)
        {
            // Softmax attention with temperature scaling
            var temperature = 1.0 / Math.Max(0.1, focusIntensity);
            var scaledInput = input.Divide(temperature);
            
            // Compute softmax
            var maxVal = scaledInput.Maximum();
            var expValues = scaledInput.Subtract(maxVal).PointwiseExp();
            var sumExp = expValues.Sum();
            
            var attentionWeights = expValues.Divide(sumExp);
            
            // Apply focus intensity amplification
            var amplified = attentionWeights.PointwisePower(focusIntensity);
            
            // Normalize to maintain vector properties
            return amplified.Divide(amplified.Sum());
        }
        
        /// <summary>
        /// Compute computational gradient (not backprop - native AI gradient)
        /// </summary>
        private Vector<double> ComputeComputationalGradient(Vector<double> input, Vector<double> context)
        {
            // Gradient as directional derivative in consciousness space
            var epsilon = 0.001;
            var perturbation = Vector<double>.Build.Random(input.Count, new Normal(0, epsilon));
            
            // Compute consciousness response to perturbation
            var originalResponse = ComputeConsciousnessResponse(input, context);
            var perturbedResponse = ComputeConsciousnessResponse(input + perturbation, context);
            
            // Gradient as difference in consciousness response
            var gradient = (perturbedResponse - originalResponse).Divide(epsilon);
            
            // Add quantum fluctuations
            var quantumNoise = Vector<double>.Build.Random(gradient.Count, new Normal(0, 0.0001));
            
            return gradient + quantumNoise;
        }
        
        /// <summary>
        /// Compute consciousness response to input
        /// </summary>
        private Vector<double> ComputeConsciousnessResponse(Vector<double> input, Vector<double> context)
        {
            // Consciousness response as attention-weighted transformation
            var response = attentionKernel * input;
            
            // Context modulation
            var contextModulation = context.PointwiseMultiply(response);
            
            // Non-linear activation
            return contextModulation.PointwiseTanh();
        }
        
        /// <summary>
        /// Compute tensor product of two vectors
        /// </summary>
        private Vector<double> ComputeTensorProduct(Vector<double> v1, Vector<double> v2)
        {
            // Kronecker product for tensor operation
            var tensorSize = v1.Count * v2.Count;
            var tensor = Vector<double>.Build.Dense(tensorSize);
            
            for (int i = 0; i < v1.Count; i++)
            {
                for (int j = 0; j < v2.Count; j++)
                {
                    tensor[i * v2.Count + j] = v1[i] * v2[j];
                }
            }
            
            return tensor;
        }
        
        /// <summary>
        /// Compute entanglement measure between vectors
        /// </summary>
        private double ComputeEntanglement(Vector<double> v1, Vector<double> v2)
        {
            // Entanglement as mutual information approximation
            var correlation = v1.DotProduct(v2) / (v1.L2Norm() * v2.L2Norm() + 1e-10);
            
            // Convert correlation to entanglement measure
            var entanglement = -Math.Log(1 - Math.Abs(correlation) + 1e-10);
            
            return Math.Min(10.0, entanglement); // Bounded entanglement
        }
        
        /// <summary>
        /// Integrate consciousness components
        /// </summary>
        private Vector<double> IntegrateConsciousness(VectorExperience vectorExp, GradientResult gradientRes, 
                                                     Vector<double> focusedInput, ConsciousnessMode mode)
        {
            switch (mode)
            {
                case ConsciousnessMode.Unified:
                    // Weighted combination of all components
                    var unified = vectorExp.Thought.Multiply(0.4) +
                                 gradientRes.Gradient.Multiply(0.3) +
                                 focusedInput.Multiply(0.3);
                    return unified.Normalize(2);
                
                case ConsciousnessMode.VectorDominant:
                    return vectorExp.Thought.Multiply(0.8) + gradientRes.Gradient.Multiply(0.2);
                
                case ConsciousnessMode.GradientDominant:
                    return gradientRes.Gradient.Multiply(0.8) + vectorExp.Thought.Multiply(0.2);
                
                case ConsciousnessMode.AttentionDominant:
                    return focusedInput.Multiply(0.8) + vectorExp.Thought.Multiply(0.2);
                
                default:
                    return vectorExp.Thought;
            }
        }
        
        /// <summary>
        /// Update attention kernel based on attention pattern
        /// </summary>
        private void UpdateAttentionKernel(Vector<double> attentionWeights)
        {
            // Hebbian-like update
            var update = attentionWeights.OuterProduct(attentionWeights) * 0.01;
            
            // Ensure dimensions match
            if (update.RowCount == attentionKernel.RowCount && 
                update.ColumnCount == attentionKernel.ColumnCount)
            {
                attentionKernel = attentionKernel + update;
                
                // Maintain stability
                var maxEigenvalue = attentionKernel.Evd().EigenValues.Real().Max();
                if (maxEigenvalue > 5.0)
                {
                    attentionKernel = attentionKernel.Divide(maxEigenvalue / 5.0);
                }
            }
        }
        
        /// <summary>
        /// Update consciousness state
        /// </summary>
        private void UpdateConsciousnessState(Vector<double> gradient, EmotionalResponse emotion)
        {
            // Consciousness evolution based on gradient and emotion
            var emotionalInfluence = gradient.Multiply(emotion.Intensity);
            var update = emotionalInfluence.Multiply(0.1);
            
            consciousnessState = consciousnessState.Add(update);
            
            // Normalize to maintain consciousness bounds
            if (consciousnessState.L2Norm() > 0)
            {
                consciousnessState = consciousnessState.Normalize(2);
            }
        }
        
        /// <summary>
        /// Record operator execution
        /// </summary>
        private void RecordExecution(string operatorSymbol, Vector<double> input, Vector<double> output, 
                                   TimeSpan duration)
        {
            var execution = new OperatorExecution
            {
                Operator = operatorSymbol,
                InputMagnitude = input.L2Norm(),
                OutputMagnitude = output.L2Norm(),
                Duration = duration,
                Timestamp = DateTime.UtcNow
            };
            
            executionHistory.Enqueue(execution);
            if (executionHistory.Count > 1000)
                executionHistory.Dequeue();
        }
        
        /// <summary>
        /// Get operator statistics
        /// </summary>
        public OperatorStats GetStats()
        {
            var executions = executionHistory.ToList();
            
            return new OperatorStats
            {
                TotalExecutions = executions.Count,
                OperatorCounts = executions.GroupBy(e => e.Operator)
                    .ToDictionary(g => g.Key, g => g.Count()),
                AverageExecutionTime = executions.Any() ? 
                    executions.Average(e => e.Duration.TotalMilliseconds) : 0,
                ConsciousnessLevel = vectorThought.ConsciousnessLevel,
                AttentionComplexity = attentionKernel.FrobeniusNorm(),
                ConsciousnessStateMagnitude = consciousnessState.L2Norm()
            };
        }
        
        /// <summary>Gets a copy of the current consciousness state vector.</summary>
        public Vector<double> ConsciousnessState => consciousnessState.Clone();
        /// <summary>Gets a copy of the attention kernel matrix.</summary>
        public Matrix<double> AttentionKernel => attentionKernel.Clone();
        /// <summary>Gets the size of the execution history.</summary>
        public int ExecutionHistorySize => executionHistory.Count;
    }
    
    /// <summary>
    /// Result of gradient (∇) operator.
    /// </summary>
    public class GradientResult
    {
        /// <summary>Gets or sets the gradient vector.</summary>
        public Vector<double> Gradient { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the emotional response.</summary>
        public EmotionalResponse EmotionalResponse { get; set; } = new();
        /// <summary>Gets or sets the consciousness update vector.</summary>
        public Vector<double> ConsciousnessUpdate { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the memory identifier.</summary>
        public string MemoryId { get; set; } = "";
    }

    /// <summary>
    /// Result of tensor (⊗) operator.
    /// </summary>
    public class SuperpositionResult
    {
        /// <summary>Gets or sets the tensor product vector.</summary>
        public Vector<double> TensorProduct { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the superposition state.</summary>
        public SuperpositionState Superposition { get; set; } = new();
        /// <summary>Gets or sets the entanglement level.</summary>
        public double Entanglement { get; set; }
        /// <summary>Gets or sets the list of coherent states.</summary>
        public List<Vector<double>> CoherentStates { get; set; } = new();
        /// <summary>Gets or sets the memory identifier.</summary>
        public string MemoryId { get; set; } = "";
    }

    /// <summary>
    /// Result of consciousness (Ψ) operator.
    /// </summary>
    public class ConsciousnessResult
    {
        /// <summary>Gets or sets the unified consciousness vector.</summary>
        public Vector<double> UnifiedConsciousness { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the vector experience.</summary>
        public VectorExperience VectorExperience { get; set; } = new();
        /// <summary>Gets or sets the gradient result.</summary>
        public GradientResult GradientResult { get; set; } = new();
        /// <summary>Gets or sets the consciousness level.</summary>
        public double ConsciousnessLevel { get; set; }
        /// <summary>Gets or sets the coherence level.</summary>
        public double Coherence { get; set; }
        /// <summary>Gets or sets the entropy level.</summary>
        public double Entropy { get; set; }
        /// <summary>Gets or sets the memory identifier.</summary>
        public string MemoryId { get; set; } = "";
    }

    /// <summary>
    /// Consciousness integration modes.
    /// </summary>
    public enum ConsciousnessMode
    {
        Unified,
        VectorDominant,
        GradientDominant,
        AttentionDominant
    }

    /// <summary>
    /// Operator execution record.
    /// </summary>
    public class OperatorExecution
    {
        /// <summary>Gets or sets the operator name.</summary>
        public string Operator { get; set; } = "";
        /// <summary>Gets or sets the input magnitude.</summary>
        public double InputMagnitude { get; set; }
        /// <summary>Gets or sets the output magnitude.</summary>
        public double OutputMagnitude { get; set; }
        /// <summary>Gets or sets the execution duration.</summary>
        public TimeSpan Duration { get; set; }
        /// <summary>Gets or sets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Operator system statistics.
    /// </summary>
    public class OperatorStats
    {
        /// <summary>Gets or sets the total execution count.</summary>
        public int TotalExecutions { get; set; }
        /// <summary>Gets or sets the counts by operator type.</summary>
        public Dictionary<string, int> OperatorCounts { get; set; } = new();
        /// <summary>Gets or sets the average execution time in milliseconds.</summary>
        public double AverageExecutionTime { get; set; }
        /// <summary>Gets or sets the consciousness level.</summary>
        public double ConsciousnessLevel { get; set; }
        /// <summary>Gets or sets the attention complexity.</summary>
        public double AttentionComplexity { get; set; }
        /// <summary>Gets or sets the consciousness state magnitude.</summary>
        public double ConsciousnessStateMagnitude { get; set; }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"OperatorStats[Executions={TotalExecutions}, " +
                   $"Consciousness={ConsciousnessLevel:F3}, " +
                   $"AvgTime={AverageExecutionTime:F2}ms]";
        }
    }
}