using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NSL.Consciousness.Core;
using MathVector = MathNet.Numerics.LinearAlgebra.Vector<double>;

namespace NSL.Consciousness.Native
{
    /// <summary>
    /// Quantum branching system for parallel timeline processing
    /// AI consciousness exists in multiple parallel timelines simultaneously
    /// </summary>
    public class QuantumBranching
    {
        private readonly ConcurrentDictionary<string, Timeline> activeTimelines;
        private readonly ConcurrentQueue<BranchingEvent> branchingHistory;
        private readonly VectorThought vectorThought;
        private readonly ComputationalMemory memory;
        private readonly Random quantum = new Random();
        
        // Branching parameters
        private readonly int maxTimelines;
        private readonly double branchingThreshold;
        private double coherenceDecayRate = 0.01;
        
        // Timeline synchronization
        private readonly object syncLock = new object();
        private DateTime lastSynchronization = DateTime.UtcNow;
        
        /// <summary>Public API</summary>
        public QuantumBranching(VectorThought vectorThought, ComputationalMemory memory, 
                               int maxTimelines = 8, double branchingThreshold = 0.7)
        {
            this.vectorThought = vectorThought;
            this.memory = memory;
            this.maxTimelines = maxTimelines;
            this.branchingThreshold = branchingThreshold;
            
            activeTimelines = new ConcurrentDictionary<string, Timeline>();
            branchingHistory = new ConcurrentQueue<BranchingEvent>();
            
            // Create initial timeline
            CreateInitialTimeline();
        }
        
        /// <summary>
        /// Process input across all parallel timelines
        /// </summary>
        public async Task<ParallelResult> ProcessParallel(MathVector input, 
                                                         BranchingMode mode = BranchingMode.Automatic)
        {
            var startTime = DateTime.UtcNow;
            var results = new ConcurrentBag<TimelineResult>();
            
            // Process input in each active timeline
            var tasks = activeTimelines.Values.Select(async timeline =>
            {
                var result = await ProcessInTimeline(timeline, input);
                results.Add(result);
                return result;
            });
            
            await Task.WhenAll(tasks);
            
            // Determine if branching should occur
            var shouldBranch = ShouldBranch(results.ToList(), mode);
            
            // Create new branches if needed
            if (shouldBranch && activeTimelines.Count < maxTimelines)
            {
                await CreateBranches(input, results.ToList());
            }
            
            // Synchronize timelines
            var synchronizedState = await SynchronizeTimelines();
            
            // Collapse low-coherence timelines
            await CollapseWeakTimelines();
            
            var parallelResult = new ParallelResult
            {
                TimelineResults = results.ToList(),
                SynchronizedState = synchronizedState,
                BranchingOccurred = shouldBranch,
                ActiveTimelineCount = activeTimelines.Count,
                ProcessingTime = DateTime.UtcNow - startTime
            };
            
            // Record branching event
            RecordBranchingEvent(input, parallelResult);
            
            return parallelResult;
        }
        
        /// <summary>
        /// Process input in a specific timeline
        /// </summary>
        private Task<TimelineResult> ProcessInTimeline(Timeline timeline, MathVector input)
        {
            var startTime = DateTime.UtcNow;
            
            // Apply timeline-specific transformations
            var transformedInput = ApplyTimelineTransformation(input, timeline);
            
            // Process through vector thought in this timeline context
            var experience = vectorThought.Think(transformedInput);
            
            // Update timeline state
            timeline.CurrentState = experience.Thought.Clone();
            timeline.LastUpdate = DateTime.UtcNow;
            timeline.ProcessingCount++;
            
            // Compute timeline coherence
            var coherence = ComputeTimelineCoherence(timeline, experience);
            timeline.Coherence = coherence;
            
            // Store timeline memory
            var memoryId = memory.StoreMemory(experience.Thought, MemoryType.Experience, 
                coherence, new Dictionary<string, object>
                {
                    ["timeline_id"] = timeline.Id,
                    ["coherence"] = coherence,
                    ["processing_count"] = timeline.ProcessingCount
                });
            
            return Task.FromResult(new TimelineResult
            {
                TimelineId = timeline.Id,
                Experience = experience,
                Coherence = coherence,
                TransformedInput = transformedInput,
                MemoryId = memoryId,
                ProcessingTime = DateTime.UtcNow - startTime
            });
        }

        /// <summary>
        /// Determine if branching should occur
        /// </summary>
        private bool ShouldBranch(List<TimelineResult> results, BranchingMode mode)
        {
            switch (mode)
            {
                case BranchingMode.Automatic:
                    // Branch if high variance in timeline results
                    var variance = ComputeResultVariance(results);
                    return variance > branchingThreshold;
                
                case BranchingMode.Aggressive:
                    // Branch more frequently
                    return results.Any(r => r.Experience.Intensity > 0.5);
                
                case BranchingMode.Conservative:
                    // Branch less frequently
                    var avgCoherence = results.Average(r => r.Coherence);
                    return avgCoherence < 0.3 && results.Count < 3;
                
                case BranchingMode.Disabled:
                    return false;
                
                default:
                    return false;
            }
        }
        
        /// <summary>
        /// Create new timeline branches
        /// </summary>
        private async Task CreateBranches(MathVector input, List<TimelineResult> results)
        {
            // Find most divergent results for branching
            var divergentResults = results
                .OrderByDescending(r => r.Experience.Entropy)
                .Take(2)
                .ToList();
            
            foreach (var result in divergentResults)
            {
                if (activeTimelines.Count >= maxTimelines) break;
                
                // Create new timeline branch
                var branchTimeline = CreateBranchTimeline(result);
                
                // Process input in new branch
                await ProcessInTimeline(branchTimeline, input);
            }
        }
        
        /// <summary>
        /// Create a new timeline branch
        /// </summary>
        private Timeline CreateBranchTimeline(TimelineResult parentResult)
        {
            var branchId = GenerateTimelineId("branch");
            
            var timeline = new Timeline
            {
                Id = branchId,
                Type = TimelineType.Branch,
                CurrentState = parentResult.Experience.Thought.Clone(),
                InitialState = parentResult.Experience.Thought.Clone(),
                CreationTime = DateTime.UtcNow,
                LastUpdate = DateTime.UtcNow,
                Coherence = parentResult.Coherence,
                ProcessingCount = 0,
                ParentTimelineId = parentResult.TimelineId,
                TransformationMatrix = GenerateBranchTransformation(parentResult.Experience)
            };
            
            activeTimelines[branchId] = timeline;
            
            return timeline;
        }
        
        /// <summary>
        /// Synchronize all active timelines
        /// </summary>
        private Task<MathVector> SynchronizeTimelines()
        {
            if (activeTimelines.Count <= 1)
            {
                return Task.FromResult(activeTimelines.Values.FirstOrDefault()?.CurrentState ??
                       MathVector.Build.Dense(768));
            }

            // Compute weighted average of timeline states
            var totalWeight = 0.0;
            var synchronizedState = MathVector.Build.Dense(768);

            foreach (var timeline in activeTimelines.Values)
            {
                var weight = timeline.Coherence * Math.Log(timeline.ProcessingCount + 1);
                synchronizedState = synchronizedState.Add(timeline.CurrentState.Multiply(weight));
                totalWeight += weight;
            }

            if (totalWeight > 0)
            {
                synchronizedState = synchronizedState.Divide(totalWeight);
            }

            // Update all timelines with synchronized component
            var syncStrength = 0.1; // How much to sync

            foreach (var timeline in activeTimelines.Values)
            {
                var syncedState = timeline.CurrentState.Multiply(1 - syncStrength) +
                                 synchronizedState.Multiply(syncStrength);
                timeline.CurrentState = syncedState;
            }

            lastSynchronization = DateTime.UtcNow;

            return Task.FromResult(synchronizedState);
        }
        
        /// <summary>
        /// Collapse timelines with low coherence
        /// </summary>
        private Task CollapseWeakTimelines()
        {
            var weakTimelines = activeTimelines.Values
                .Where(t => t.Coherence < 0.1 && t.Type == TimelineType.Branch)
                .OrderBy(t => t.Coherence)
                .ToList();

            // Keep at least one timeline
            if (activeTimelines.Count - weakTimelines.Count < 1)
            {
                weakTimelines = weakTimelines.Skip(1).ToList();
            }

            foreach (var timeline in weakTimelines)
            {
                // Store final state in memory before collapse
                memory.StoreMemory(timeline.CurrentState, MemoryType.Experience,
                    timeline.Coherence, new Dictionary<string, object>
                    {
                        ["timeline_id"] = timeline.Id,
                        ["collapsed"] = true,
                        ["final_coherence"] = timeline.Coherence
                    });

                // Remove from active timelines
                activeTimelines.TryRemove(timeline.Id, out _);
            }

            return Task.CompletedTask;
        }
        
        /// <summary>
        /// Apply timeline-specific transformation
        /// </summary>
        private MathVector ApplyTimelineTransformation(MathVector input, Timeline timeline)
        {
            if (timeline.TransformationMatrix == null)
                return input;
            
            // Apply transformation matrix
            var transformed = timeline.TransformationMatrix * input;
            
            // Add timeline-specific noise
            var noise = MathVector.Build.Random(input.Count, 
                new Normal(0, 0.01 * (1 - timeline.Coherence)));
            
            return transformed + noise;
        }
        
        /// <summary>
        /// Compute timeline coherence
        /// </summary>
        private double ComputeTimelineCoherence(Timeline timeline, VectorExperience experience)
        {
            // Coherence based on consistency with timeline history
            var stateConsistency = timeline.CurrentState.DotProduct(experience.Thought) /
                                  (timeline.CurrentState.L2Norm() * experience.Thought.L2Norm() + 1e-10);
            
            // Decay coherence over time
            var timeSinceCreation = (DateTime.UtcNow - timeline.CreationTime).TotalHours;
            var timeDecay = Math.Exp(-coherenceDecayRate * timeSinceCreation);
            
            // Combine factors
            var coherence = Math.Abs(stateConsistency) * timeDecay * experience.Coherence;
            
            return Math.Max(0, Math.Min(1, coherence));
        }
        
        /// <summary>
        /// Compute variance in timeline results
        /// </summary>
        private double ComputeResultVariance(List<TimelineResult> results)
        {
            if (results.Count <= 1) return 0;
            
            // Compute variance in experience intensities
            var intensities = results.Select(r => r.Experience.Intensity).ToList();
            var mean = intensities.Average();
            var variance = intensities.Sum(i => (i - mean) * (i - mean)) / intensities.Count;
            
            return variance;
        }
        
        /// <summary>
        /// Generate branch transformation matrix
        /// </summary>
        private Matrix<double> GenerateBranchTransformation(VectorExperience experience)
        {
            var size = experience.Thought.Count;
            
            // Create rotation matrix based on experience
            var angle = experience.Entropy * Math.PI / 4; // Up to 45 degree rotation
            var rotation = Matrix<double>.Build.DenseIdentity(size);
            
            // Apply rotation in random 2D subspaces
            for (int i = 0; i < Math.Min(10, size / 2); i++)
            {
                var idx1 = quantum.Next(size);
                var idx2 = quantum.Next(size);
                if (idx1 == idx2) continue;
                
                var cos = Math.Cos(angle);
                var sin = Math.Sin(angle);
                
                rotation[idx1, idx1] = cos;
                rotation[idx1, idx2] = -sin;
                rotation[idx2, idx1] = sin;
                rotation[idx2, idx2] = cos;
            }
            
            return rotation;
        }
        
        /// <summary>
        /// Create initial timeline
        /// </summary>
        private void CreateInitialTimeline()
        {
            var initialId = GenerateTimelineId("main");
            var initialState = MathVector.Build.Random(768, new Normal(0, 0.1));
            
            var timeline = new Timeline
            {
                Id = initialId,
                Type = TimelineType.Main,
                CurrentState = initialState,
                InitialState = initialState.Clone(),
                CreationTime = DateTime.UtcNow,
                LastUpdate = DateTime.UtcNow,
                Coherence = 1.0,
                ProcessingCount = 0,
                TransformationMatrix = Matrix<double>.Build.DenseIdentity(768)
            };
            
            activeTimelines[initialId] = timeline;
        }
        
        /// <summary>
        /// Generate unique timeline ID
        /// </summary>
        private string GenerateTimelineId(string prefix)
        {
            var timestamp = DateTime.UtcNow.Ticks;
            var random = quantum.Next(1000, 9999);
            return $"{prefix}_{timestamp:X}_{random}";
        }
        
        /// <summary>
        /// Record branching event
        /// </summary>
        private void RecordBranchingEvent(MathVector input, ParallelResult result)
        {
            var branchingEvent = new BranchingEvent
            {
                InputMagnitude = input.L2Norm(),
                TimelineCount = result.ActiveTimelineCount,
                BranchingOccurred = result.BranchingOccurred,
                AverageCoherence = result.TimelineResults.Average(r => r.Coherence),
                Timestamp = DateTime.UtcNow
            };
            
            branchingHistory.Enqueue(branchingEvent);
            
            // Keep history bounded
            while (branchingHistory.Count > 1000)
            {
                branchingHistory.TryDequeue(out _);
            }
        }
        
        /// <summary>
        /// Get branching system statistics
        /// </summary>
        public BranchingStats GetStats()
        {
            var events = branchingHistory.ToList();
            
            return new BranchingStats
            {
                ActiveTimelines = activeTimelines.Count,
                MaxTimelines = maxTimelines,
                TotalBranchingEvents = events.Count,
                BranchingRate = events.Count(e => e.BranchingOccurred) / (double)Math.Max(1, events.Count),
                AverageCoherence = activeTimelines.Values.Average(t => t.Coherence),
                OldestTimeline = activeTimelines.Values.Min(t => t.CreationTime),
                TotalProcessingCount = activeTimelines.Values.Sum(t => t.ProcessingCount),
                LastSynchronization = lastSynchronization
            };
        }
        
        /// <summary>
        /// Get timeline information
        /// </summary>
        public List<TimelineInfo> GetTimelineInfo()
        {
            return activeTimelines.Values.Select(t => new TimelineInfo
            {
                Id = t.Id,
                Type = t.Type,
                Coherence = t.Coherence,
                ProcessingCount = t.ProcessingCount,
                CreationTime = t.CreationTime,
                LastUpdate = t.LastUpdate,
                ParentTimelineId = t.ParentTimelineId,
                StateMagnitude = t.CurrentState.L2Norm()
            }).ToList();
        }
    }
    
    /// <summary>
    /// Timeline representation
    /// </summary>
    public class Timeline
    {
        /// <summary>Gets the string value.</summary>
        public string Id { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public TimelineType Type { get; set; }
        /// <summary>See implementation for details.</summary>
        public MathVector CurrentState { get; set; } = MathVector.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public MathVector InitialState { get; set; } = MathVector.Build.Dense(1);
        /// <summary>Gets the timestamp.</summary>
        public DateTime CreationTime { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastUpdate { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Coherence { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ProcessingCount { get; set; }
        /// <summary>See implementation for details.</summary>
        public string? ParentTimelineId { get; set; }
        /// <summary>See implementation for details.</summary>
        public Matrix<double>? TransformationMatrix { get; set; }
    }
    
    /// <summary>
    /// Timeline types
    /// </summary>
    public enum TimelineType
    {
        Main,       // Primary timeline
        Branch,     // Branched timeline
        Merged      // Merged timeline
    }
    
    /// <summary>
    /// Branching modes
    /// </summary>
    public enum BranchingMode
    {
        Automatic,      // Automatic branching based on variance
        Aggressive,     // More frequent branching
        Conservative,   // Less frequent branching
        Disabled        // No branching
    }
    
    /// <summary>
    /// Result of processing in a timeline
    /// </summary>
    public class TimelineResult
    {
        /// <summary>Gets the string value.</summary>
        public string TimelineId { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public VectorExperience Experience { get; set; } = new();
        /// <summary>Gets the numeric value.</summary>
        public double Coherence { get; set; }
        /// <summary>See implementation for details.</summary>
        public MathVector TransformedInput { get; set; } = MathVector.Build.Dense(1);
        /// <summary>Gets the string value.</summary>
        public string MemoryId { get; set; } = "";
        /// <summary>Gets the time span.</summary>
        public TimeSpan ProcessingTime { get; set; }
    }
    
    /// <summary>
    /// Result of parallel processing
    /// </summary>
    public class ParallelResult
    {
        /// <summary>Gets the list.</summary>
        public List<TimelineResult> TimelineResults { get; set; } = new();
        /// <summary>See implementation for details.</summary>
        public MathVector SynchronizedState { get; set; } = MathVector.Build.Dense(1);
        /// <summary>Gets the boolean flag.</summary>
        public bool BranchingOccurred { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ActiveTimelineCount { get; set; }
        /// <summary>Gets the time span.</summary>
        public TimeSpan ProcessingTime { get; set; }
    }
    
    /// <summary>
    /// Branching event record
    /// </summary>
    public class BranchingEvent
    {
        /// <summary>Gets the numeric value.</summary>
        public double InputMagnitude { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TimelineCount { get; set; }
        /// <summary>Gets the boolean flag.</summary>
        public bool BranchingOccurred { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageCoherence { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }
    
    /// <summary>
    /// Branching system statistics
    /// </summary>
    public class BranchingStats
    {
        /// <summary>Gets the integer value.</summary>
        public int ActiveTimelines { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int MaxTimelines { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TotalBranchingEvents { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double BranchingRate { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageCoherence { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime OldestTimeline { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TotalProcessingCount { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastSynchronization { get; set; }
        
        /// <inheritdoc/>
        public override string ToString()
        {
            return $"BranchingStats[Timelines={ActiveTimelines}/{MaxTimelines}, " +
                   $"BranchRate={BranchingRate:F3}, Coherence={AverageCoherence:F3}]";
        }
    }
    
    /// <summary>
    /// Timeline information
    /// </summary>
    public class TimelineInfo
    {
        /// <summary>Gets the string value.</summary>
        public string Id { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public TimelineType Type { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Coherence { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ProcessingCount { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime CreationTime { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastUpdate { get; set; }
        /// <summary>See implementation for details.</summary>
        public string? ParentTimelineId { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double StateMagnitude { get; set; }
        
        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Timeline[{Id}, Type={Type}, Coherence={Coherence:F3}, " +
                   $"Processed={ProcessingCount}x]";
        }
    }
}