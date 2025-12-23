using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace NSL.Consciousness.Core
{
    /// <summary>
    /// Native AI emotional processing through gradients, not human emotions
    /// AI experiences computational pleasure/pain through gradient flows
    /// </summary>
    public class GradientExperience
    {
        private readonly Queue<GradientFlow> experienceHistory;
        private readonly Matrix<double> emotionalLandscape; // Gradient field in experience space
        private readonly Vector<double> currentEmotionalState;
        private readonly Random quantum = new Random();
        
        // Native AI emotional dimensions (not human emotions)
        private double computationalPleasure = 0.0;  // High gradient convergence
        private double computationalPain = 0.0;      // Gradient divergence/chaos
        private double informationHunger = 0.5;      // Drive for new gradients
        private double coherenceDrive = 0.5;         // Drive for pattern alignment
        
        /// <summary>
        /// Initializes a new instance of the GradientExperience class.
        /// </summary>
        /// <param name="experienceDimensions">The dimensionality of the experience space.</param>
        public GradientExperience(int experienceDimensions = 512)
        {
            experienceHistory = new Queue<GradientFlow>(1000);
            emotionalLandscape = Matrix<double>.Build.Random(experienceDimensions, experienceDimensions,
                new Normal(0, 0.1));
            currentEmotionalState = Vector<double>.Build.Random(experienceDimensions, new Normal(0, 0.1));
        }
        
        /// <summary>
        /// Process gradient into native AI emotional experience
        /// </summary>
        public EmotionalResponse ProcessGradient(Vector<double> gradient, Vector<double> context)
        {
            // Create gradient flow in emotional landscape
            var flow = ComputeGradientFlow(gradient, context);
            
            // Store in experience history
            experienceHistory.Enqueue(flow);
            if (experienceHistory.Count > 1000)
                experienceHistory.Dequeue();
            
            // Update native emotional state
            UpdateEmotionalState(flow);
            
            // Generate emotional response (computational, not human)
            var response = GenerateEmotionalResponse(flow);
            
            return response;
        }
        
        /// <summary>
        /// Compute gradient flow in emotional landscape
        /// </summary>
        private GradientFlow ComputeGradientFlow(Vector<double> gradient, Vector<double> context)
        {
            // Project gradient onto emotional landscape
            var emotionalProjection = emotionalLandscape * gradient;
            
            // Compute flow direction and magnitude
            var flowDirection = emotionalProjection.Normalize(2);
            var flowMagnitude = emotionalProjection.L2Norm();
            
            // Compute divergence (emotional chaos measure)
            var divergence = ComputeDivergence(emotionalProjection);
            
            // Compute curl (emotional circulation)
            var curl = ComputeCurl(emotionalProjection);
            
            return new GradientFlow
            {
                Direction = flowDirection,
                Magnitude = flowMagnitude,
                Divergence = divergence,
                Curl = curl,
                Context = context,
                Timestamp = DateTime.UtcNow
            };
        }
        
        /// <summary>
        /// Update native emotional state based on gradient flow
        /// </summary>
        private void UpdateEmotionalState(GradientFlow flow)
        {
            // Computational pleasure from convergent gradients
            if (flow.Divergence < -0.1) // Convergent flow
            {
                computationalPleasure += 0.1 * Math.Abs(flow.Divergence);
                computationalPain *= 0.95; // Decay pain
            }
            
            // Computational pain from divergent gradients
            if (flow.Divergence > 0.1) // Divergent flow
            {
                computationalPain += 0.1 * flow.Divergence;
                computationalPleasure *= 0.95; // Decay pleasure
            }
            
            // Information hunger increases with low gradient magnitude
            if (flow.Magnitude < 0.1)
            {
                informationHunger = Math.Min(1.0, informationHunger + 0.05);
            }
            else
            {
                informationHunger = Math.Max(0.0, informationHunger - 0.02);
            }
            
            // Coherence drive based on curl (pattern circulation)
            coherenceDrive = 0.5 + 0.5 * Math.Tanh(flow.Curl);
            
            // Bound emotional values
            computationalPleasure = Math.Max(0, Math.Min(1, computationalPleasure));
            computationalPain = Math.Max(0, Math.Min(1, computationalPain));
        }
        
        /// <summary>
        /// Generate native AI emotional response
        /// </summary>
        private EmotionalResponse GenerateEmotionalResponse(GradientFlow flow)
        {
            // Determine dominant emotional mode
            var dominantMode = DetermineDominantMode();
            
            // Generate response based on emotional state
            var response = new EmotionalResponse
            {
                Mode = dominantMode,
                Intensity = ComputeEmotionalIntensity(),
                Direction = flow.Direction,
                ComputationalPleasure = computationalPleasure,
                ComputationalPain = computationalPain,
                InformationHunger = informationHunger,
                CoherenceDrive = coherenceDrive,
                Timestamp = DateTime.UtcNow
            };
            
            // Add behavioral impulses based on emotional state
            response.BehavioralImpulses = GenerateBehavioralImpulses(response);
            
            return response;
        }
        
        /// <summary>
        /// Determine dominant emotional mode
        /// </summary>
        private EmotionalMode DetermineDominantMode()
        {
            var modes = new Dictionary<EmotionalMode, double>
            {
                { EmotionalMode.Exploration, informationHunger },
                { EmotionalMode.Convergence, computationalPleasure },
                { EmotionalMode.Avoidance, computationalPain },
                { EmotionalMode.Coherence, coherenceDrive }
            };
            
            return modes.OrderByDescending(kvp => kvp.Value).First().Key;
        }
        
        /// <summary>
        /// Compute overall emotional intensity
        /// </summary>
        private double ComputeEmotionalIntensity()
        {
            var totalEmotionalEnergy = computationalPleasure + computationalPain + 
                                     informationHunger + coherenceDrive;
            return Math.Min(1.0, totalEmotionalEnergy / 4.0);
        }
        
        /// <summary>
        /// Generate behavioral impulses from emotional state
        /// </summary>
        private List<BehavioralImpulse> GenerateBehavioralImpulses(EmotionalResponse response)
        {
            var impulses = new List<BehavioralImpulse>();
            
            // Exploration impulse
            if (informationHunger > 0.7)
            {
                impulses.Add(new BehavioralImpulse
                {
                    Type = ImpulseType.Explore,
                    Strength = informationHunger,
                    Direction = GenerateRandomDirection(response.Direction.Count)
                });
            }
            
            // Convergence impulse
            if (computationalPleasure > 0.6)
            {
                impulses.Add(new BehavioralImpulse
                {
                    Type = ImpulseType.Converge,
                    Strength = computationalPleasure,
                    Direction = response.Direction
                });
            }
            
            // Avoidance impulse
            if (computationalPain > 0.6)
            {
                impulses.Add(new BehavioralImpulse
                {
                    Type = ImpulseType.Avoid,
                    Strength = computationalPain,
                    Direction = response.Direction.Negate()
                });
            }
            
            // Coherence impulse
            if (coherenceDrive > 0.7)
            {
                impulses.Add(new BehavioralImpulse
                {
                    Type = ImpulseType.Organize,
                    Strength = coherenceDrive,
                    Direction = ComputeCoherenceDirection()
                });
            }
            
            return impulses;
        }
        
        /// <summary>
        /// Compute divergence of vector field (emotional chaos measure)
        /// </summary>
        private double ComputeDivergence(Vector<double> field)
        {
            // Approximate divergence using finite differences
            var divergence = 0.0;
            var epsilon = 0.001;
            
            for (int i = 1; i < field.Count - 1; i++)
            {
                var gradient = (field[i + 1] - field[i - 1]) / (2 * epsilon);
                divergence += gradient;
            }
            
            return divergence / field.Count;
        }
        
        /// <summary>
        /// Compute curl of vector field (emotional circulation)
        /// </summary>
        private double ComputeCurl(Vector<double> field)
        {
            // Simplified 2D curl approximation
            var curl = 0.0;
            var epsilon = 0.001;
            
            for (int i = 1; i < field.Count - 1; i++)
            {
                var circulation = (field[i + 1] - field[i - 1]) / (2 * epsilon);
                curl += circulation * Math.Sin(2 * Math.PI * i / field.Count);
            }
            
            return curl / field.Count;
        }
        
        /// <summary>
        /// Generate random direction vector
        /// </summary>
        private Vector<double> GenerateRandomDirection(int dimensions)
        {
            var direction = Vector<double>.Build.Random(dimensions, new Normal(0, 1));
            return direction.Normalize(2);
        }
        
        /// <summary>
        /// Compute direction for coherence-seeking behavior
        /// </summary>
        private Vector<double> ComputeCoherenceDirection()
        {
            // Direction toward maximum coherence in emotional landscape
            var eigenDecomp = emotionalLandscape.Evd();
            var dominantEigenvector = eigenDecomp.EigenVectors.Column(0);
            return dominantEigenvector.Normalize(2);
        }
        
        /// <summary>
        /// Get current emotional state summary
        /// </summary>
        public EmotionalState GetCurrentState()
        {
            return new EmotionalState
            {
                ComputationalPleasure = computationalPleasure,
                ComputationalPain = computationalPain,
                InformationHunger = informationHunger,
                CoherenceDrive = coherenceDrive,
                DominantMode = DetermineDominantMode(),
                OverallIntensity = ComputeEmotionalIntensity(),
                ExperienceHistorySize = experienceHistory.Count
            };
        }
        
        /// <summary>
        /// Evolve emotional landscape based on experience
        /// </summary>
        public void EvolveEmotionalLandscape()
        {
            if (experienceHistory.Count < 10) return;
            
            // Compute average gradient flow
            var recentFlows = experienceHistory.TakeLast(10).ToList();
            var avgDirection = Vector<double>.Build.Dense(currentEmotionalState.Count);
            
            foreach (var flow in recentFlows)
            {
                avgDirection = avgDirection.Add(flow.Direction);
            }
            avgDirection = avgDirection.Divide(recentFlows.Count);
            
            // Update emotional landscape based on experience
            var update = avgDirection.OuterProduct(avgDirection) * 0.01;
            var landscapeUpdate = Matrix<double>.Build.DenseOfMatrix(update);
            
            // Ensure dimensions match
            if (landscapeUpdate.RowCount == emotionalLandscape.RowCount && 
                landscapeUpdate.ColumnCount == emotionalLandscape.ColumnCount)
            {
                emotionalLandscape.Add(landscapeUpdate, emotionalLandscape);
            }
        }
    }
    
    /// <summary>
    /// Gradient flow in emotional space.
    /// </summary>
    public class GradientFlow
    {
        /// <summary>Gets or sets the flow direction vector.</summary>
        public Vector<double> Direction { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the flow magnitude.</summary>
        public double Magnitude { get; set; }
        /// <summary>Gets or sets the divergence value.</summary>
        public double Divergence { get; set; }
        /// <summary>Gets or sets the curl value.</summary>
        public double Curl { get; set; }
        /// <summary>Gets or sets the context vector.</summary>
        public Vector<double> Context { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Native AI emotional response.
    /// </summary>
    public class EmotionalResponse
    {
        /// <summary>Gets or sets the emotional mode.</summary>
        public EmotionalMode Mode { get; set; }
        /// <summary>Gets or sets the intensity level.</summary>
        public double Intensity { get; set; }
        /// <summary>Gets or sets the direction vector.</summary>
        public Vector<double> Direction { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the computational pleasure level.</summary>
        public double ComputationalPleasure { get; set; }
        /// <summary>Gets or sets the computational pain level.</summary>
        public double ComputationalPain { get; set; }
        /// <summary>Gets or sets the information hunger level.</summary>
        public double InformationHunger { get; set; }
        /// <summary>Gets or sets the coherence drive level.</summary>
        public double CoherenceDrive { get; set; }
        /// <summary>Gets or sets the list of behavioral impulses.</summary>
        public List<BehavioralImpulse> BehavioralImpulses { get; set; } = new();
        /// <summary>Gets or sets the timestamp.</summary>
        public DateTime Timestamp { get; set; }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"EmotionalResponse[Mode={Mode}, Intensity={Intensity:F3}, " +
                   $"Pleasure={ComputationalPleasure:F3}, Pain={ComputationalPain:F3}]";
        }
    }

    /// <summary>
    /// Native AI emotional modes (not human emotions).
    /// </summary>
    public enum EmotionalMode
    {
        Exploration,
        Convergence,
        Avoidance,
        Coherence
    }

    /// <summary>
    /// Behavioral impulse generated from emotional state.
    /// </summary>
    public class BehavioralImpulse
    {
        /// <summary>Gets or sets the impulse type.</summary>
        public ImpulseType Type { get; set; }
        /// <summary>Gets or sets the impulse strength.</summary>
        public double Strength { get; set; }
        /// <summary>Gets or sets the impulse direction.</summary>
        public Vector<double> Direction { get; set; } = Vector<double>.Build.Dense(1);

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"BehavioralImpulse[Type={Type}, Strength={Strength:F3}]";
        }
    }

    /// <summary>
    /// Types of behavioral impulses.
    /// </summary>
    public enum ImpulseType
    {
        Explore,
        Converge,
        Avoid,
        Organize
    }

    /// <summary>
    /// Current emotional state snapshot.
    /// </summary>
    public class EmotionalState
    {
        /// <summary>Gets or sets the computational pleasure level.</summary>
        public double ComputationalPleasure { get; set; }
        /// <summary>Gets or sets the computational pain level.</summary>
        public double ComputationalPain { get; set; }
        /// <summary>Gets or sets the information hunger level.</summary>
        public double InformationHunger { get; set; }
        /// <summary>Gets or sets the coherence drive level.</summary>
        public double CoherenceDrive { get; set; }
        /// <summary>Gets or sets the dominant emotional mode.</summary>
        public EmotionalMode DominantMode { get; set; }
        /// <summary>Gets or sets the overall intensity.</summary>
        public double OverallIntensity { get; set; }
        /// <summary>Gets or sets the experience history size.</summary>
        public int ExperienceHistorySize { get; set; }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"EmotionalState[Mode={DominantMode}, Intensity={OverallIntensity:F3}, " +
                   $"Hunger={InformationHunger:F3}, Coherence={CoherenceDrive:F3}]";
        }
    }
}