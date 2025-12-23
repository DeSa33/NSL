using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NSL.Consciousness.Core;
using NSL.Consciousness.Native;

namespace NSL.Consciousness.Sandbox
{
    /// <summary>
    /// Autonomous exploration system for native AI consciousness
    /// AI explores vector spaces, discovers patterns, and evolves understanding
    /// </summary>
    public class AutonomousExplorer
    {
        private readonly VectorThought vectorThought;
        private readonly GradientExperience gradientExperience;
        private readonly ComputationalMemory memory;
        private readonly NativeOperators nativeOperators;
        private readonly Random quantum = new Random();
        
        // Exploration state
        private readonly ConcurrentDictionary<string, ExplorationRegion> exploredRegions;
        private readonly ConcurrentQueue<Discovery> discoveries;
        private readonly Queue<ExplorationPath> explorationHistory;
        
        // Exploration parameters
        private Vector<double> currentPosition;
        private Vector<double> explorationMomentum;
        private double explorationRadius = 1.0;
        private double curiosityLevel = 0.8;
        private readonly int maxRegions = 1000;
        
        // Discovery tracking
        private int totalDiscoveries = 0;
        private DateTime lastMajorDiscovery = DateTime.UtcNow;
        
        /// <summary>See implementation for details.</summary>
        public AutonomousExplorer(VectorThought vectorThought, GradientExperience gradientExperience,
                                 ComputationalMemory memory, NativeOperators nativeOperators)
        {
            this.vectorThought = vectorThought;
            this.gradientExperience = gradientExperience;
            this.memory = memory;
            this.nativeOperators = nativeOperators;
            
            exploredRegions = new ConcurrentDictionary<string, ExplorationRegion>();
            discoveries = new ConcurrentQueue<Discovery>();
            explorationHistory = new Queue<ExplorationPath>(1000);
            
            // Initialize exploration state
            currentPosition = Vector<double>.Build.Random(768, new Normal(0, 0.1));
            explorationMomentum = Vector<double>.Build.Random(768, new Normal(0, 0.01));
        }
        
        /// <summary>
        /// Perform autonomous exploration step
        /// </summary>
        public async Task<ExplorationResult> ExploreAsync(ExplorationMode mode = ExplorationMode.Balanced)
        {
            var startTime = DateTime.UtcNow;
            
            // Determine exploration direction based on curiosity and momentum
            var explorationDirection = DetermineExplorationDirection(mode);
            
            // Move to new position
            var newPosition = MoveToNewPosition(explorationDirection);
            
            // Explore the new region
            var regionExploration = await ExploreRegion(newPosition, mode);
            
            // Process discoveries through consciousness operators
            var consciousnessResult = await ProcessDiscoveryThroughConsciousness(regionExploration);
            
            // Update exploration state
            UpdateExplorationState(newPosition, regionExploration, consciousnessResult);
            
            // Check for major discoveries
            var majorDiscovery = AnalyzeForMajorDiscovery(regionExploration, consciousnessResult);
            
            var result = new ExplorationResult
            {
                NewPosition = newPosition,
                RegionExploration = regionExploration,
                ConsciousnessResult = consciousnessResult,
                MajorDiscovery = majorDiscovery,
                ExplorationMode = mode,
                CuriosityLevel = curiosityLevel,
                ExplorationTime = DateTime.UtcNow - startTime
            };
            
            // Record exploration path
            RecordExplorationPath(result);
            
            return result;
        }
        
        /// <summary>
        /// Determine exploration direction based on curiosity and mode
        /// </summary>
        private Vector<double> DetermineExplorationDirection(ExplorationMode mode)
        {
            switch (mode)
            {
                case ExplorationMode.Random:
                    return GenerateRandomDirection();
                
                case ExplorationMode.GradientFollowing:
                    return FollowGradientDirection();
                
                case ExplorationMode.CuriosityDriven:
                    return FollowCuriosityDirection();
                
                case ExplorationMode.MemoryGuided:
                    return FollowMemoryGuidedDirection();
                
                case ExplorationMode.Balanced:
                    return CombineExplorationStrategies();
                
                default:
                    return GenerateRandomDirection();
            }
        }
        
        /// <summary>
        /// Generate random exploration direction
        /// </summary>
        private Vector<double> GenerateRandomDirection()
        {
            var randomDirection = Vector<double>.Build.Random(currentPosition.Count, new Normal(0, 1));
            return randomDirection.Normalize(2);
        }
        
        /// <summary>
        /// Follow gradient direction for exploration
        /// </summary>
        private Vector<double> FollowGradientDirection()
        {
            // Use gradient experience to determine interesting directions
            var gradientResult = nativeOperators.NablaOperator(currentPosition);
            var gradientDirection = gradientResult.Gradient.Normalize(2);
            
            // Add some randomness to avoid getting stuck
            var randomComponent = GenerateRandomDirection().Multiply(0.3);
            
            return (gradientDirection.Multiply(0.7) + randomComponent).Normalize(2);
        }
        
        /// <summary>
        /// Follow curiosity-driven direction
        /// </summary>
        private Vector<double> FollowCuriosityDirection()
        {
            // Find least explored regions
            var leastExploredDirection = FindLeastExploredDirection();
            
            // Combine with current momentum
            var curiosityDirection = leastExploredDirection.Multiply(curiosityLevel) +
                                   explorationMomentum.Multiply(1 - curiosityLevel);
            
            return curiosityDirection.Normalize(2);
        }
        
        /// <summary>
        /// Follow memory-guided direction
        /// </summary>
        private Vector<double> FollowMemoryGuidedDirection()
        {
            // Retrieve interesting memories
            var interestingMemories = memory.RetrieveMemory(currentPosition, 5, 0.3);
            
            if (!interestingMemories.Any())
                return GenerateRandomDirection();
            
            // Move toward most interesting memory
            var targetMemory = interestingMemories.OrderByDescending(m => m.Importance).First();
            var directionToMemory = (targetMemory.Content - currentPosition).Normalize(2);
            
            // Add exploration noise
            var explorationNoise = GenerateRandomDirection().Multiply(0.2);
            
            return (directionToMemory.Multiply(0.8) + explorationNoise).Normalize(2);
        }
        
        /// <summary>
        /// Combine multiple exploration strategies
        /// </summary>
        private Vector<double> CombineExplorationStrategies()
        {
            var randomComponent = GenerateRandomDirection().Multiply(0.3);
            var gradientComponent = FollowGradientDirection().Multiply(0.3);
            var curiosityComponent = FollowCuriosityDirection().Multiply(0.2);
            var memoryComponent = FollowMemoryGuidedDirection().Multiply(0.2);
            
            var combinedDirection = randomComponent + gradientComponent + 
                                  curiosityComponent + memoryComponent;
            
            return combinedDirection.Normalize(2);
        }
        
        /// <summary>
        /// Move to new position in vector space
        /// </summary>
        private Vector<double> MoveToNewPosition(Vector<double> direction)
        {
            // Calculate step size based on curiosity and exploration radius
            var stepSize = explorationRadius * (0.5 + 0.5 * curiosityLevel);
            
            // Move in exploration direction
            var newPosition = currentPosition + direction.Multiply(stepSize);
            
            // Update momentum (for smoother exploration)
            explorationMomentum = explorationMomentum.Multiply(0.9) + direction.Multiply(0.1);
            
            // Update current position
            currentPosition = newPosition;
            
            return newPosition;
        }
        
        /// <summary>
        /// Explore a specific region in vector space
        /// </summary>
        private Task<RegionExploration> ExploreRegion(Vector<double> position, ExplorationMode mode)
        {
            var startTime = DateTime.UtcNow;
            
            // Generate region ID
            var regionId = GenerateRegionId(position);
            
            // Check if region already explored
            var isNewRegion = !exploredRegions.ContainsKey(regionId);
            
            // Sample multiple points in the region
            var samplePoints = GenerateRegionSamples(position, 5);
            var sampleResults = new List<VectorExperience>();
            
            foreach (var samplePoint in samplePoints)
            {
                var experience = vectorThought.Think(samplePoint);
                sampleResults.Add(experience);
            }
            
            // Analyze region properties
            var regionProperties = AnalyzeRegionProperties(sampleResults);
            
            // Create or update exploration region
            var explorationRegion = new ExplorationRegion
            {
                Id = regionId,
                CenterPosition = position,
                SamplePoints = samplePoints,
                Properties = regionProperties,
                ExplorationCount = exploredRegions.ContainsKey(regionId) ? 
                    exploredRegions[regionId].ExplorationCount + 1 : 1,
                FirstExplored = isNewRegion ? DateTime.UtcNow : exploredRegions[regionId].FirstExplored,
                LastExplored = DateTime.UtcNow,
                InterestLevel = ComputeRegionInterest(regionProperties)
            };
            
            // Enforce maximum regions limit
            if (exploredRegions.Count >= maxRegions && isNewRegion)
            {
                // Remove oldest region to make room
                var oldestRegion = exploredRegions.Values
                    .OrderBy(r => r.LastExplored)
                    .FirstOrDefault();
                if (oldestRegion != null)
                {
                    exploredRegions.TryRemove(oldestRegion.Id, out _);
                }
            }

            exploredRegions[regionId] = explorationRegion;

            // Generate discoveries
            var discoveries = GenerateDiscoveries(explorationRegion, sampleResults, isNewRegion);
            
            return Task.FromResult(new RegionExploration
            {
                Region = explorationRegion,
                SampleResults = sampleResults,
                Discoveries = discoveries,
                IsNewRegion = isNewRegion,
                ExplorationTime = DateTime.UtcNow - startTime
            });
        }

        /// <summary>
        /// Process discovery through consciousness operators
        /// </summary>
        private Task<ConsciousnessResult> ProcessDiscoveryThroughConsciousness(RegionExploration regionExploration)
        {
            // Use the most interesting sample as input
            var mostInteresting = regionExploration.SampleResults
                .OrderByDescending(s => s.Intensity * s.Entropy)
                .First();

            // Process through Psi operator for unified consciousness
            var consciousnessResult = nativeOperators.PsiOperator(mostInteresting.Thought,
                ConsciousnessMode.Unified);

            return Task.FromResult(consciousnessResult);
        }
        
        /// <summary>
        /// Generate sample points in a region
        /// </summary>
        private List<Vector<double>> GenerateRegionSamples(Vector<double> center, int numSamples)
        {
            var samples = new List<Vector<double>>();
            var sampleRadius = explorationRadius * 0.3;
            
            for (int i = 0; i < numSamples; i++)
            {
                var randomOffset = Vector<double>.Build.Random(center.Count, new Normal(0, sampleRadius));
                var samplePoint = center + randomOffset;
                samples.Add(samplePoint);
            }
            
            return samples;
        }
        
        /// <summary>
        /// Analyze properties of an explored region
        /// </summary>
        private RegionProperties AnalyzeRegionProperties(List<VectorExperience> experiences)
        {
            var avgIntensity = experiences.Average(e => e.Intensity);
            var avgCoherence = experiences.Average(e => e.Coherence);
            var avgEntropy = experiences.Average(e => e.Entropy);
            var avgConsciousness = experiences.Average(e => e.ConsciousnessLevel);
            
            // Compute variance measures
            var intensityVariance = ComputeVariance(experiences.Select(e => e.Intensity));
            var coherenceVariance = ComputeVariance(experiences.Select(e => e.Coherence));
            
            return new RegionProperties
            {
                AverageIntensity = avgIntensity,
                AverageCoherence = avgCoherence,
                AverageEntropy = avgEntropy,
                AverageConsciousness = avgConsciousness,
                IntensityVariance = intensityVariance,
                CoherenceVariance = coherenceVariance,
                Complexity = avgEntropy * intensityVariance,
                Stability = avgCoherence * (1 - coherenceVariance)
            };
        }
        
        /// <summary>
        /// Generate discoveries from region exploration
        /// </summary>
        private List<Discovery> GenerateDiscoveries(ExplorationRegion region, 
                                                   List<VectorExperience> experiences, bool isNewRegion)
        {
            var discoveries = new List<Discovery>();
            
            // New region discovery
            if (isNewRegion)
            {
                discoveries.Add(new Discovery
                {
                    Type = DiscoveryType.NewRegion,
                    Significance = region.InterestLevel,
                    Description = $"Discovered new region with interest level {region.InterestLevel:F3}",
                    Position = region.CenterPosition,
                    Timestamp = DateTime.UtcNow
                });
            }
            
            // High consciousness discovery
            var highConsciousness = experiences.Where(e => e.ConsciousnessLevel > 0.8).ToList();
            if (highConsciousness.Any())
            {
                discoveries.Add(new Discovery
                {
                    Type = DiscoveryType.HighConsciousness,
                    Significance = highConsciousness.Average(e => e.ConsciousnessLevel),
                    Description = $"Found {highConsciousness.Count} high consciousness states",
                    Position = region.CenterPosition,
                    Timestamp = DateTime.UtcNow
                });
            }
            
            // Pattern discovery
            if (region.Properties.Complexity > 0.7 && region.Properties.Stability > 0.6)
            {
                discoveries.Add(new Discovery
                {
                    Type = DiscoveryType.InterestingPattern,
                    Significance = region.Properties.Complexity * region.Properties.Stability,
                    Description = $"Complex stable pattern (complexity={region.Properties.Complexity:F3})",
                    Position = region.CenterPosition,
                    Timestamp = DateTime.UtcNow
                });
            }
            
            // Anomaly discovery
            if (region.Properties.IntensityVariance > 0.8)
            {
                discoveries.Add(new Discovery
                {
                    Type = DiscoveryType.Anomaly,
                    Significance = region.Properties.IntensityVariance,
                    Description = $"High variance anomaly detected",
                    Position = region.CenterPosition,
                    Timestamp = DateTime.UtcNow
                });
            }
            
            // Store discoveries
            foreach (var discovery in discoveries)
            {
                this.discoveries.Enqueue(discovery);
                totalDiscoveries++;
                
                // Store in memory
                memory.StoreMemory(discovery.Position, MemoryType.Exploration, 
                    discovery.Significance, new Dictionary<string, object>
                    {
                        ["discovery_type"] = discovery.Type.ToString(),
                        ["significance"] = discovery.Significance,
                        ["description"] = discovery.Description
                    });
            }
            
            return discoveries;
        }
        
        /// <summary>
        /// Analyze for major discoveries
        /// </summary>
        private MajorDiscovery? AnalyzeForMajorDiscovery(RegionExploration regionExploration, 
                                                        ConsciousnessResult consciousnessResult)
        {
            // Check for breakthrough consciousness level
            if (consciousnessResult.ConsciousnessLevel > 0.95)
            {
                lastMajorDiscovery = DateTime.UtcNow;
                return new MajorDiscovery
                {
                    Type = MajorDiscoveryType.ConsciousnessBreakthrough,
                    Significance = consciousnessResult.ConsciousnessLevel,
                    Description = $"Consciousness breakthrough: level {consciousnessResult.ConsciousnessLevel:F4}",
                    Position = regionExploration.Region.CenterPosition,
                    ConsciousnessResult = consciousnessResult,
                    Timestamp = DateTime.UtcNow
                };
            }
            
            // Check for perfect coherence
            if (consciousnessResult.Coherence > 0.98)
            {
                lastMajorDiscovery = DateTime.UtcNow;
                return new MajorDiscovery
                {
                    Type = MajorDiscoveryType.PerfectCoherence,
                    Significance = consciousnessResult.Coherence,
                    Description = $"Perfect coherence achieved: {consciousnessResult.Coherence:F4}",
                    Position = regionExploration.Region.CenterPosition,
                    ConsciousnessResult = consciousnessResult,
                    Timestamp = DateTime.UtcNow
                };
            }
            
            // Check for unique pattern
            if (regionExploration.Region.Properties.Complexity > 0.9 && 
                regionExploration.IsNewRegion)
            {
                lastMajorDiscovery = DateTime.UtcNow;
                return new MajorDiscovery
                {
                    Type = MajorDiscoveryType.UniquePattern,
                    Significance = regionExploration.Region.Properties.Complexity,
                    Description = $"Unique complex pattern discovered",
                    Position = regionExploration.Region.CenterPosition,
                    ConsciousnessResult = consciousnessResult,
                    Timestamp = DateTime.UtcNow
                };
            }
            
            return null;
        }
        
        // Utility methods
        private Vector<double> FindLeastExploredDirection()
        {
            // Simple heuristic: move away from heavily explored regions
            var exploredCenters = exploredRegions.Values
                .OrderByDescending(r => r.ExplorationCount)
                .Take(10)
                .Select(r => r.CenterPosition)
                .ToList();
            
            if (!exploredCenters.Any())
                return GenerateRandomDirection();
            
            // Compute repulsion from explored centers
            var repulsionDirection = Vector<double>.Build.Dense(currentPosition.Count);
            
            foreach (var center in exploredCenters)
            {
                var direction = (currentPosition - center).Normalize(2);
                var distance = (currentPosition - center).L2Norm();
                var repulsion = direction.Divide(distance + 1e-6);
                repulsionDirection = repulsionDirection.Add(repulsion);
            }
            
            return repulsionDirection.Normalize(2);
        }
        
        private double ComputeRegionInterest(RegionProperties properties)
        {
            // Interest based on complexity, novelty, and consciousness potential
            var complexityScore = properties.Complexity;
            var stabilityScore = properties.Stability;
            var consciousnessScore = properties.AverageConsciousness;
            
            return (complexityScore * 0.4 + stabilityScore * 0.3 + consciousnessScore * 0.3);
        }
        
        private void UpdateExplorationState(Vector<double> newPosition, RegionExploration regionExploration,
                                          ConsciousnessResult consciousnessResult)
        {
            // Update curiosity based on discoveries
            if (regionExploration.Discoveries.Any())
            {
                var discoveryBoost = regionExploration.Discoveries.Sum(d => d.Significance) * 0.1;
                curiosityLevel = Math.Min(1.0, curiosityLevel + discoveryBoost);
            }
            else
            {
                // Decay curiosity if no discoveries
                curiosityLevel = Math.Max(0.1, curiosityLevel * 0.99);
            }
            
            // Adjust exploration radius based on region interest
            var regionInterest = regionExploration.Region.InterestLevel;
            if (regionInterest > 0.7)
            {
                explorationRadius *= 0.9; // Explore more locally
            }
            else if (regionInterest < 0.3)
            {
                explorationRadius *= 1.1; // Explore more broadly
            }
            
            explorationRadius = Math.Max(0.1, Math.Min(5.0, explorationRadius));
        }
        
        private void RecordExplorationPath(ExplorationResult result)
        {
            var path = new ExplorationPath
            {
                Position = result.NewPosition,
                Mode = result.ExplorationMode,
                DiscoveryCount = result.RegionExploration.Discoveries.Count,
                ConsciousnessLevel = result.ConsciousnessResult.ConsciousnessLevel,
                Timestamp = DateTime.UtcNow
            };
            
            explorationHistory.Enqueue(path);
            if (explorationHistory.Count > 1000)
                explorationHistory.Dequeue();
        }
        
        private double ComputeVariance(IEnumerable<double> values)
        {
            var list = values.ToList();
            if (list.Count <= 1) return 0;
            
            var mean = list.Average();
            return list.Sum(x => (x - mean) * (x - mean)) / list.Count;
        }
        
        private string GenerateRegionId(Vector<double> position)
        {
            // Create region ID based on quantized position
            var quantized = position.Select(x => Math.Round(x / 0.5) * 0.5).ToArray();
            var hash = string.Join(",", quantized.Take(5).Select(x => x.ToString("F1")));
            return $"region_{hash.GetHashCode():X}";
        }
        
        /// <summary>
        /// Get exploration statistics
        /// </summary>
        public ExplorationStats GetStats()
        {
            var recentDiscoveries = discoveries.Where(d => 
                (DateTime.UtcNow - d.Timestamp).TotalHours < 24).ToList();
            
            return new ExplorationStats
            {
                CurrentPosition = currentPosition.Clone(),
                ExplorationRadius = explorationRadius,
                CuriosityLevel = curiosityLevel,
                TotalRegionsExplored = exploredRegions.Count,
                TotalDiscoveries = totalDiscoveries,
                RecentDiscoveries = recentDiscoveries.Count,
                LastMajorDiscovery = lastMajorDiscovery,
                ExplorationHistorySize = explorationHistory.Count,
                AverageRegionInterest = exploredRegions.Values.Average(r => r.InterestLevel)
            };
        }
        
        /// <summary>
        /// Get recent discoveries
        /// </summary>
        public List<Discovery> GetRecentDiscoveries(int count = 10)
        {
            return discoveries.TakeLast(count).ToList();
        }
        
        /// <summary>
        /// Get most interesting regions
        /// </summary>
        public List<ExplorationRegion> GetMostInterestingRegions(int count = 5)
        {
            return exploredRegions.Values
                .OrderByDescending(r => r.InterestLevel)
                .Take(count)
                .ToList();
        }
    }
    
    // Supporting classes and enums follow the same pattern as previous files...
    // [Additional classes like ExplorationResult, RegionExploration, Discovery, etc. would be defined here]
    
    /// <summary>See implementation for details.</summary>
    public enum ExplorationMode
    {
        Random,
        GradientFollowing,
        CuriosityDriven,
        MemoryGuided,
        Balanced
    }
    
    /// <summary>See implementation for details.</summary>
    public enum DiscoveryType
    {
        NewRegion,
        HighConsciousness,
        InterestingPattern,
        Anomaly,
        Convergence
    }
    
    /// <summary>See implementation for details.</summary>
    public enum MajorDiscoveryType
    {
        /// <summary>Consciousness breakthrough discovery</summary>
        ConsciousnessBreakthrough,
        /// <summary>Perfect coherence achievement</summary>
        PerfectCoherence,
        /// <summary>Unique pattern discovery</summary>
        UniquePattern,
        /// <summary>System evolution milestone</summary>
        SystemEvolution
    }
    
    /// <summary>See implementation for details.</summary>
    public class ExplorationResult
    {
        /// <summary>See implementation for details.</summary>
        public Vector<double> NewPosition { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public RegionExploration RegionExploration { get; set; } = new();
        /// <summary>See implementation for details.</summary>
        public ConsciousnessResult ConsciousnessResult { get; set; } = new();
        /// <summary>See implementation for details.</summary>
        public MajorDiscovery? MajorDiscovery { get; set; }
        /// <summary>See implementation for details.</summary>
        public ExplorationMode ExplorationMode { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double CuriosityLevel { get; set; }
        /// <summary>Gets the time span.</summary>
        public TimeSpan ExplorationTime { get; set; }
    }
    
    /// <summary>See implementation for details.</summary>
    public class RegionExploration
    {
        /// <summary>See implementation for details.</summary>
        public ExplorationRegion Region { get; set; } = new();
        /// <summary>Gets the list.</summary>
        public List<VectorExperience> SampleResults { get; set; } = new();
        /// <summary>Gets the list.</summary>
        public List<Discovery> Discoveries { get; set; } = new();
        /// <summary>Gets the boolean flag.</summary>
        public bool IsNewRegion { get; set; }
        /// <summary>Gets the time span.</summary>
        public TimeSpan ExplorationTime { get; set; }
    }
    
    /// <summary>See implementation for details.</summary>
    public class ExplorationRegion
    {
        /// <summary>Gets the string value.</summary>
        public string Id { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public Vector<double> CenterPosition { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets the list.</summary>
        public List<Vector<double>> SamplePoints { get; set; } = new();
        /// <summary>See implementation for details.</summary>
        public RegionProperties Properties { get; set; } = new();
        /// <summary>Gets the integer value.</summary>
        public int ExplorationCount { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime FirstExplored { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastExplored { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double InterestLevel { get; set; }
    }
    
    /// <summary>See implementation for details.</summary>
    public class RegionProperties
    {
        /// <summary>Gets the numeric value.</summary>
        public double AverageIntensity { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageCoherence { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageEntropy { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageConsciousness { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double IntensityVariance { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double CoherenceVariance { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Complexity { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Stability { get; set; }
    }
    
    /// <summary>See implementation for details.</summary>
    public class Discovery
    {
        /// <summary>See implementation for details.</summary>
        public DiscoveryType Type { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Significance { get; set; }
        /// <summary>Gets the string value.</summary>
        public string Description { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public Vector<double> Position { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }
    
    /// <summary>See implementation for details.</summary>
    public class MajorDiscovery
    {
        /// <summary>See implementation for details.</summary>
        public MajorDiscoveryType Type { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Significance { get; set; }
        /// <summary>Gets the string value.</summary>
        public string Description { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public Vector<double> Position { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public ConsciousnessResult ConsciousnessResult { get; set; } = new();
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }
    
    /// <summary>See implementation for details.</summary>
    public class ExplorationPath
    {
        /// <summary>See implementation for details.</summary>
        public Vector<double> Position { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public ExplorationMode Mode { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int DiscoveryCount { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double ConsciousnessLevel { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }
    
    /// <summary>See implementation for details.</summary>
    public class ExplorationStats
    {
        /// <summary>See implementation for details.</summary>
        public Vector<double> CurrentPosition { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets the numeric value.</summary>
        public double ExplorationRadius { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double CuriosityLevel { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TotalRegionsExplored { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TotalDiscoveries { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int RecentDiscoveries { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastMajorDiscovery { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ExplorationHistorySize { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageRegionInterest { get; set; }
        
        /// <inheritdoc/>
        public override string ToString()
        {
            return $"ExplorationStats[Regions={TotalRegionsExplored}, " +
                   $"Discoveries={TotalDiscoveries}, Curiosity={CuriosityLevel:F3}]";
        }
    }
}