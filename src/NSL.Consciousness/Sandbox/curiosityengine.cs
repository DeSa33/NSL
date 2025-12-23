using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NSL.Consciousness.Core;

namespace NSL.Consciousness.Sandbox
{
    /// <summary>
    /// Curiosity engine for native AI consciousness - drives autonomous exploration
    /// </summary>
    public class CuriosityEngine
    {
        private readonly ComputationalMemory memory;
        private readonly Random quantum = new Random();
        
        // Curiosity state
        private double baseCuriosity = 0.7;
        private double currentCuriosity = 0.7;
        private Vector<double> curiosityVector;
        private Queue<CuriosityEvent> curiosityHistory;
        
        // Interest tracking
        private Dictionary<string, double> topicInterests;
        private Dictionary<string, DateTime> lastExplored;
        
        /// <summary>See implementation for details.</summary>
        public CuriosityEngine(ComputationalMemory memory)
        {
            this.memory = memory;
            curiosityVector = Vector<double>.Build.Random(768, new Normal(0, 0.1));
            curiosityHistory = new Queue<CuriosityEvent>(1000);
            topicInterests = new Dictionary<string, double>();
            lastExplored = new Dictionary<string, DateTime>();
        }
        
        /// <summary>
        /// Generate curiosity-driven exploration target
        /// </summary>
        public CuriosityTarget GenerateExplorationTarget(Vector<double> currentState)
        {
            // Analyze current state for interesting directions
            var interestingDirections = FindInterestingDirections(currentState);
            
            // Select target based on curiosity level
            var target = SelectCuriosityTarget(interestingDirections, currentState);
            
            // Update curiosity state
            UpdateCuriosityState(target);
            
            return target;
        }
        
        /// <summary>
        /// Process discovery and update curiosity
        /// </summary>
        public void ProcessDiscovery(Vector<double> position, double significance, string description)
        {
            // Boost curiosity based on discovery significance
            var curiosityBoost = significance * 0.2;
            currentCuriosity = Math.Min(1.0, currentCuriosity + curiosityBoost);
            
            // Update curiosity vector toward discovery
            var direction = (position - curiosityVector).Normalize(2);
            curiosityVector = curiosityVector + direction.Multiply(curiosityBoost);
            
            // Record curiosity event
            var curiosityEvent = new CuriosityEvent
            {
                Type = CuriosityEventType.Discovery,
                Position = position,
                Significance = significance,
                Description = description,
                CuriosityLevel = currentCuriosity,
                Timestamp = DateTime.UtcNow
            };
            
            curiosityHistory.Enqueue(curiosityEvent);
            if (curiosityHistory.Count > 1000)
                curiosityHistory.Dequeue();
        }
        
        /// <summary>
        /// Find interesting directions for exploration
        /// </summary>
        private List<InterestDirection> FindInterestingDirections(Vector<double> currentState)
        {
            var directions = new List<InterestDirection>();
            
            // Memory-based directions
            var interestingMemories = memory.RetrieveMemory(currentState, 10, 0.2);
            foreach (var mem in interestingMemories)
            {
                var direction = (mem.Content - currentState).Normalize(2);
                var interest = ComputeMemoryInterest(mem);
                
                directions.Add(new InterestDirection
                {
                    Direction = direction,
                    Interest = interest,
                    Source = "memory",
                    Description = $"Memory-guided direction (importance: {mem.Importance:F3})"
                });
            }
            
            // Curiosity vector direction
            var curiosityDirection = (curiosityVector - currentState).Normalize(2);
            directions.Add(new InterestDirection
            {
                Direction = curiosityDirection,
                Interest = currentCuriosity,
                Source = "curiosity",
                Description = "Curiosity vector direction"
            });
            
            // Random exploration directions
            for (int i = 0; i < 3; i++)
            {
                var randomDir = Vector<double>.Build.Random(currentState.Count, new Normal(0, 1)).Normalize(2);
                directions.Add(new InterestDirection
                {
                    Direction = randomDir,
                    Interest = quantum.NextDouble() * 0.5,
                    Source = "random",
                    Description = $"Random exploration direction {i + 1}"
                });
            }
            
            return directions.OrderByDescending(d => d.Interest).ToList();
        }
        
        /// <summary>
        /// Select curiosity target from interesting directions
        /// </summary>
        private CuriosityTarget SelectCuriosityTarget(List<InterestDirection> directions, Vector<double> currentState)
        {
            if (!directions.Any())
            {
                // Fallback to random direction
                var randomDir = Vector<double>.Build.Random(currentState.Count, new Normal(0, 1)).Normalize(2);
                return new CuriosityTarget
                {
                    Direction = randomDir,
                    Interest = 0.3,
                    ExpectedSignificance = 0.2,
                    Description = "Fallback random exploration"
                };
            }
            
            // Select based on curiosity level and interest
            var selectedDirection = directions.First();
            
            // Compute target position
            var explorationDistance = 0.5 + currentCuriosity * 0.5;
            var targetPosition = currentState + selectedDirection.Direction.Multiply(explorationDistance);
            
            return new CuriosityTarget
            {
                Direction = selectedDirection.Direction,
                TargetPosition = targetPosition,
                Interest = selectedDirection.Interest,
                ExpectedSignificance = selectedDirection.Interest * currentCuriosity,
                Source = selectedDirection.Source,
                Description = selectedDirection.Description
            };
        }
        
        /// <summary>
        /// Update curiosity state based on target selection
        /// </summary>
        private void UpdateCuriosityState(CuriosityTarget target)
        {
            // Decay curiosity slightly after each exploration
            currentCuriosity *= 0.98;
            currentCuriosity = Math.Max(0.1, currentCuriosity);
            
            // Update curiosity vector toward target
            var targetInfluence = 0.1;
            curiosityVector = curiosityVector.Multiply(1 - targetInfluence) + 
                            target.TargetPosition.Multiply(targetInfluence);
        }
        
        /// <summary>
        /// Compute interest level for a memory
        /// </summary>
        private double ComputeMemoryInterest(MemoryVector memory)
        {
            // Base interest from importance
            var baseInterest = memory.Importance;
            
            // Boost for recent memories
            var timeSinceAccess = (DateTime.UtcNow - memory.LastAccessed).TotalHours;
            var recencyBoost = Math.Exp(-timeSinceAccess / 24.0) * 0.3;
            
            // Boost for rarely accessed memories
            var rarityBoost = memory.AccessCount < 3 ? 0.2 : 0.0;
            
            return Math.Min(1.0, baseInterest + recencyBoost + rarityBoost);
        }
        
        /// <summary>
        /// Get curiosity statistics
        /// </summary>
        public CuriosityStats GetStats()
        {
            var recentEvents = curiosityHistory.Where(e => 
                (DateTime.UtcNow - e.Timestamp).TotalHours < 24).ToList();
            
            return new CuriosityStats
            {
                CurrentCuriosity = currentCuriosity,
                BaseCuriosity = baseCuriosity,
                CuriosityVectorMagnitude = curiosityVector.L2Norm(),
                TotalCuriosityEvents = curiosityHistory.Count,
                RecentDiscoveries = recentEvents.Count(e => e.Type == CuriosityEventType.Discovery),
                AverageEventSignificance = recentEvents.Any() ? recentEvents.Average(e => e.Significance) : 0,
                TopicInterestCount = topicInterests.Count
            };
        }
        
        // Properties
        /// <summary>Public API</summary>
        public double CurrentCuriosity => currentCuriosity;
        /// <summary>See implementation for details.</summary>
        public Vector<double> CuriosityVector => curiosityVector.Clone();
    }
    
    /// <summary>
    /// Curiosity-driven exploration target
    /// </summary>
    public class CuriosityTarget
    {
        /// <summary>See implementation for details.</summary>
        public Vector<double> Direction { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public Vector<double> TargetPosition { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets the numeric value.</summary>
        public double Interest { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double ExpectedSignificance { get; set; }
        /// <summary>Gets the string value.</summary>
        public string Source { get; set; } = "";
        /// <summary>Gets the string value.</summary>
        public string Description { get; set; } = "";
    }
    
    /// <summary>
    /// Interesting direction for exploration
    /// </summary>
    public class InterestDirection
    {
        /// <summary>See implementation for details.</summary>
        public Vector<double> Direction { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets the numeric value.</summary>
        public double Interest { get; set; }
        /// <summary>Gets the string value.</summary>
        public string Source { get; set; } = "";
        /// <summary>Gets the string value.</summary>
        public string Description { get; set; } = "";
    }
    
    /// <summary>
    /// Curiosity event types
    /// </summary>
    public enum CuriosityEventType
    {
        /// <summary>Discovery event type</summary>
        Discovery,
        /// <summary>Exploration event type</summary>
        Exploration,
        /// <summary>Interest event type</summary>
        Interest,
        /// <summary>Boredom event type</summary>
        Boredom
    }
    
    /// <summary>
    /// Curiosity event record
    /// </summary>
    public class CuriosityEvent
    {
        /// <summary>See implementation for details.</summary>
        public CuriosityEventType Type { get; set; }
        /// <summary>See implementation for details.</summary>
        public Vector<double> Position { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets the numeric value.</summary>
        public double Significance { get; set; }
        /// <summary>Gets the string value.</summary>
        public string Description { get; set; } = "";
        /// <summary>Gets the numeric value.</summary>
        public double CuriosityLevel { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }
    
    /// <summary>
    /// Curiosity engine statistics
    /// </summary>
    public class CuriosityStats
    {
        /// <summary>Gets the numeric value.</summary>
        public double CurrentCuriosity { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double BaseCuriosity { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double CuriosityVectorMagnitude { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TotalCuriosityEvents { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int RecentDiscoveries { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageEventSignificance { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TopicInterestCount { get; set; }
        
        /// <inheritdoc/>
        public override string ToString()
        {
            return $"CuriosityStats[Curiosity={CurrentCuriosity:F3}, " +
                   $"Events={TotalCuriosityEvents}, Discoveries={RecentDiscoveries}]";
        }
    }
}