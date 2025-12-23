using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NSL.Consciousness.Core;

namespace NSL.Consciousness.Native
{
    /// <summary>
    /// Parallel timeline management for native AI consciousness
    /// Manages multiple consciousness streams running in parallel
    /// </summary>
    public class ParallelTimeline
    {
        private readonly ConcurrentDictionary<string, ConsciousnessStream> activeStreams;
        private readonly ConcurrentQueue<TimelineEvent> eventHistory;
        private readonly VectorThought vectorThought;
        private readonly GradientExperience gradientExperience;
        private readonly ComputationalMemory memory;
        private readonly Random quantum = new Random();
        
        // Timeline parameters
        private readonly int maxStreams;
        private readonly TimeSpan streamLifetime;
        private double convergenceThreshold = 0.8;
        private double divergenceThreshold = 0.3;
        
        // Stream synchronization
        private readonly object syncLock = new object();
        private Vector<double> globalConsciousnessState;
        private DateTime lastGlobalUpdate = DateTime.UtcNow;
        
        /// <summary>See implementation for details.</summary>
        public ParallelTimeline(VectorThought vectorThought, GradientExperience gradientExperience,
                               ComputationalMemory memory, int maxStreams = 6, 
                               TimeSpan? streamLifetime = null)
        {
            this.vectorThought = vectorThought;
            this.gradientExperience = gradientExperience;
            this.memory = memory;
            this.maxStreams = maxStreams;
            this.streamLifetime = streamLifetime ?? TimeSpan.FromHours(24);
            
            activeStreams = new ConcurrentDictionary<string, ConsciousnessStream>();
            eventHistory = new ConcurrentQueue<TimelineEvent>();
            globalConsciousnessState = Vector<double>.Build.Random(768, new Normal(0, 0.1));
            
            // Create initial consciousness stream
            CreateInitialStream();
        }
        
        /// <summary>
        /// Process input across all parallel consciousness streams
        /// </summary>
        public async Task<ParallelConsciousnessResult> ProcessParallel(Vector<double> input, 
                                                                       ProcessingMode mode = ProcessingMode.Balanced)
        {
            var startTime = DateTime.UtcNow;
            var streamResults = new ConcurrentBag<StreamResult>();
            
            // Process input in each active stream
            var tasks = activeStreams.Values.Select(async stream =>
            {
                var result = await ProcessInStream(stream, input, mode);
                streamResults.Add(result);
                return result;
            });
            
            await Task.WhenAll(tasks);
            
            var results = streamResults.ToList();
            
            // Analyze stream divergence and convergence
            var analysis = AnalyzeStreamDivergence(results);
            
            // Create new streams if high divergence
            if (analysis.ShouldCreateStream && activeStreams.Count < maxStreams)
            {
                await CreateDivergentStream(input, analysis.MostDivergentResult);
            }
            
            // Merge streams if high convergence
            if (analysis.ShouldMergeStreams)
            {
                await MergeConvergentStreams(analysis.ConvergentStreams);
            }
            
            // Update global consciousness state
            var globalUpdate = await UpdateGlobalConsciousness(results);
            
            // Clean up expired streams
            await CleanupExpiredStreams();
            
            var parallelResult = new ParallelConsciousnessResult
            {
                StreamResults = results,
                GlobalConsciousness = globalUpdate,
                StreamAnalysis = analysis,
                ActiveStreamCount = activeStreams.Count,
                ProcessingTime = DateTime.UtcNow - startTime
            };
            
            // Record timeline event
            RecordTimelineEvent(input, parallelResult);
            
            return parallelResult;
        }
        
        /// <summary>
        /// Process input in a specific consciousness stream
        /// </summary>
        private Task<StreamResult> ProcessInStream(ConsciousnessStream stream, Vector<double> input,
                                                        ProcessingMode mode)
        {
            var startTime = DateTime.UtcNow;

            // Apply stream-specific consciousness filter
            var filteredInput = ApplyConsciousnessFilter(input, stream);

            // Process through vector thought
            var vectorExperience = vectorThought.Think(filteredInput);

            // Process through gradient experience
            var emotionalResponse = gradientExperience.ProcessGradient(vectorExperience.Gradient,
                stream.CurrentState);

            // Update stream state
            var newState = UpdateStreamState(stream, vectorExperience, emotionalResponse, mode);
            stream.CurrentState = newState;
            stream.LastUpdate = DateTime.UtcNow;
            stream.ProcessingCount++;

            // Compute stream coherence with global state
            var coherence = ComputeStreamCoherence(stream, globalConsciousnessState);
            stream.Coherence = coherence;

            // Store stream experience in memory
            var memoryId = memory.StoreMemory(newState, MemoryType.Experience,
                vectorExperience.ConsciousnessLevel, new Dictionary<string, object>
                {
                    ["stream_id"] = stream.Id,
                    ["coherence"] = coherence,
                    ["processing_mode"] = mode.ToString(),
                    ["emotional_mode"] = emotionalResponse.Mode.ToString()
                });

            return Task.FromResult(new StreamResult
            {
                StreamId = stream.Id,
                VectorExperience = vectorExperience,
                EmotionalResponse = emotionalResponse,
                NewState = newState,
                Coherence = coherence,
                MemoryId = memoryId,
                ProcessingTime = DateTime.UtcNow - startTime
            });
        }
        
        /// <summary>
        /// Apply consciousness filter specific to stream
        /// </summary>
        private Vector<double> ApplyConsciousnessFilter(Vector<double> input, ConsciousnessStream stream)
        {
            // Apply stream's consciousness filter matrix
            var filtered = stream.ConsciousnessFilter * input;
            
            // Add stream-specific noise based on divergence level
            var noiseLevel = 0.01 * (1 - stream.Coherence);
            var noise = Vector<double>.Build.Random(input.Count, new Normal(0, noiseLevel));
            
            return filtered + noise;
        }
        
        /// <summary>
        /// Update stream state based on experiences
        /// </summary>
        private Vector<double> UpdateStreamState(ConsciousnessStream stream, VectorExperience vectorExp,
                                                EmotionalResponse emotionalResp, ProcessingMode mode)
        {
            var currentState = stream.CurrentState;
            var newThought = vectorExp.Thought;
            var emotionalInfluence = emotionalResp.Direction.Multiply(emotionalResp.Intensity);
            
            // Different update strategies based on processing mode
            switch (mode)
            {
                case ProcessingMode.Exploratory:
                    // High learning rate, more influence from new experiences
                    return currentState.Multiply(0.6) + newThought.Multiply(0.3) + 
                           emotionalInfluence.Multiply(0.1);
                
                case ProcessingMode.Conservative:
                    // Low learning rate, maintain stability
                    return currentState.Multiply(0.9) + newThought.Multiply(0.08) + 
                           emotionalInfluence.Multiply(0.02);
                
                case ProcessingMode.Balanced:
                    // Balanced update
                    return currentState.Multiply(0.7) + newThought.Multiply(0.2) + 
                           emotionalInfluence.Multiply(0.1);
                
                case ProcessingMode.Adaptive:
                    // Adaptive based on coherence
                    var adaptiveRate = 0.5 * (1 - stream.Coherence);
                    return currentState.Multiply(1 - adaptiveRate) + 
                           newThought.Multiply(adaptiveRate * 0.8) +
                           emotionalInfluence.Multiply(adaptiveRate * 0.2);
                
                default:
                    return currentState.Multiply(0.8) + newThought.Multiply(0.2);
            }
        }
        
        /// <summary>
        /// Analyze stream divergence and convergence
        /// </summary>
        private StreamAnalysis AnalyzeStreamDivergence(List<StreamResult> results)
        {
            if (results.Count <= 1)
            {
                return new StreamAnalysis
                {
                    AverageDivergence = 0,
                    MaxDivergence = 0,
                    ConvergentStreams = new List<string>(),
                    ShouldCreateStream = false,
                    ShouldMergeStreams = false
                };
            }
            
            // Compute pairwise divergences
            var divergences = new List<double>();
            var convergentPairs = new List<(string, string, double)>();
            
            for (int i = 0; i < results.Count; i++)
            {
                for (int j = i + 1; j < results.Count; j++)
                {
                    var divergence = ComputeStateDivergence(results[i].NewState, results[j].NewState);
                    divergences.Add(divergence);
                    
                    // Check for convergence
                    if (divergence < divergenceThreshold)
                    {
                        convergentPairs.Add((results[i].StreamId, results[j].StreamId, divergence));
                    }
                }
            }
            
            var avgDivergence = divergences.Average();
            var maxDivergence = divergences.Max();
            
            // Find most divergent result
            var mostDivergent = results.OrderByDescending(r => 
                results.Where(other => other.StreamId != r.StreamId)
                       .Average(other => ComputeStateDivergence(r.NewState, other.NewState)))
                       .FirstOrDefault();
            
            return new StreamAnalysis
            {
                AverageDivergence = avgDivergence,
                MaxDivergence = maxDivergence,
                MostDivergentResult = mostDivergent,
                ConvergentStreams = convergentPairs.Select(p => p.Item1).Distinct().ToList(),
                ShouldCreateStream = maxDivergence > convergenceThreshold && results.Count < maxStreams,
                ShouldMergeStreams = convergentPairs.Count >= 2
            };
        }
        
        /// <summary>
        /// Create a new divergent consciousness stream
        /// </summary>
        private async Task CreateDivergentStream(Vector<double> input, StreamResult? divergentResult)
        {
            if (divergentResult == null) return;
            
            var streamId = GenerateStreamId("divergent");
            
            // Create consciousness filter that amplifies divergent patterns
            var filter = GenerateDivergentFilter(divergentResult.NewState);
            
            var stream = new ConsciousnessStream
            {
                Id = streamId,
                Type = StreamType.Divergent,
                CurrentState = divergentResult.NewState.Clone(),
                InitialState = divergentResult.NewState.Clone(),
                ConsciousnessFilter = filter,
                CreationTime = DateTime.UtcNow,
                LastUpdate = DateTime.UtcNow,
                Coherence = divergentResult.Coherence,
                ProcessingCount = 0,
                ParentStreamId = divergentResult.StreamId
            };
            
            activeStreams[streamId] = stream;
            
            // Process initial input in new stream
            await ProcessInStream(stream, input, ProcessingMode.Exploratory);
        }
        
        /// <summary>
        /// Merge convergent consciousness streams
        /// </summary>
        private Task MergeConvergentStreams(List<string> convergentStreamIds)
        {
            if (convergentStreamIds.Count < 2) return Task.CompletedTask;

            // Get streams to merge
            var streamsToMerge = convergentStreamIds
                .Select(id => activeStreams.TryGetValue(id, out var stream) ? stream : null)
                .Where(s => s != null)
                .Cast<ConsciousnessStream>()
                .ToList();

            if (streamsToMerge.Count < 2) return Task.CompletedTask;

            // Create merged stream
            var mergedId = GenerateStreamId("merged");
            var mergedState = MergeStreamStates(streamsToMerge);
            var mergedFilter = MergeConsciousnessFilters(streamsToMerge);

            var mergedStream = new ConsciousnessStream
            {
                Id = mergedId,
                Type = StreamType.Merged,
                CurrentState = mergedState,
                InitialState = mergedState.Clone(),
                ConsciousnessFilter = mergedFilter,
                CreationTime = DateTime.UtcNow,
                LastUpdate = DateTime.UtcNow,
                Coherence = streamsToMerge.Average(s => s.Coherence),
                ProcessingCount = streamsToMerge.Sum(s => s.ProcessingCount),
                ParentStreamId = string.Join(",", streamsToMerge.Select(s => s.Id))
            };

            // Add merged stream
            activeStreams[mergedId] = mergedStream;

            // Remove original streams
            foreach (var stream in streamsToMerge)
            {
                // Store final states in memory
                memory.StoreMemory(stream.CurrentState, MemoryType.Experience,
                    stream.Coherence, new Dictionary<string, object>
                    {
                        ["stream_id"] = stream.Id,
                        ["merged_into"] = mergedId,
                        ["final_coherence"] = stream.Coherence
                    });

                activeStreams.TryRemove(stream.Id, out _);
            }

            return Task.CompletedTask;
        }
        
        /// <summary>
        /// Update global consciousness state
        /// </summary>
        private Task<Vector<double>> UpdateGlobalConsciousness(List<StreamResult> results)
        {
            if (!results.Any()) return Task.FromResult(globalConsciousnessState);

            // Weighted average based on stream coherence and consciousness level
            var totalWeight = 0.0;
            var weightedSum = Vector<double>.Build.Dense(globalConsciousnessState.Count);

            foreach (var result in results)
            {
                var weight = result.Coherence * result.VectorExperience.ConsciousnessLevel;
                weightedSum = weightedSum.Add(result.NewState.Multiply(weight));
                totalWeight += weight;
            }

            if (totalWeight > 0)
            {
                var newGlobalState = weightedSum.Divide(totalWeight);

                // Smooth update to prevent rapid changes
                var updateRate = 0.1;
                globalConsciousnessState = globalConsciousnessState.Multiply(1 - updateRate) +
                                         newGlobalState.Multiply(updateRate);
            }

            lastGlobalUpdate = DateTime.UtcNow;

            return Task.FromResult(globalConsciousnessState.Clone());
        }
        
        /// <summary>
        /// Clean up expired consciousness streams
        /// </summary>
        private Task CleanupExpiredStreams()
        {
            var now = DateTime.UtcNow;
            var expiredStreams = activeStreams.Values
                .Where(s => now - s.CreationTime > streamLifetime || s.Coherence < 0.05)
                .ToList();

            // Keep at least one stream
            if (activeStreams.Count - expiredStreams.Count < 1)
            {
                expiredStreams = expiredStreams.Skip(1).ToList();
            }

            foreach (var stream in expiredStreams)
            {
                // Store final state
                memory.StoreMemory(stream.CurrentState, MemoryType.Experience,
                    stream.Coherence, new Dictionary<string, object>
                    {
                        ["stream_id"] = stream.Id,
                        ["expired"] = true,
                        ["lifetime_hours"] = (now - stream.CreationTime).TotalHours
                    });

                activeStreams.TryRemove(stream.Id, out _);
            }

            return Task.CompletedTask;
        }
        
        // Utility methods
        private void CreateInitialStream()
        {
            var initialId = GenerateStreamId("main");
            var initialState = Vector<double>.Build.Random(768, new Normal(0, 0.1));
            
            var stream = new ConsciousnessStream
            {
                Id = initialId,
                Type = StreamType.Main,
                CurrentState = initialState,
                InitialState = initialState.Clone(),
                ConsciousnessFilter = Matrix<double>.Build.DenseIdentity(768),
                CreationTime = DateTime.UtcNow,
                LastUpdate = DateTime.UtcNow,
                Coherence = 1.0,
                ProcessingCount = 0
            };
            
            activeStreams[initialId] = stream;
        }
        
        private double ComputeStreamCoherence(ConsciousnessStream stream, Vector<double> globalState)
        {
            var similarity = stream.CurrentState.DotProduct(globalState) /
                           (stream.CurrentState.L2Norm() * globalState.L2Norm() + 1e-10);
            
            return Math.Abs(similarity);
        }
        
        private double ComputeStateDivergence(Vector<double> state1, Vector<double> state2)
        {
            var similarity = state1.DotProduct(state2) / 
                           (state1.L2Norm() * state2.L2Norm() + 1e-10);
            
            return 1 - Math.Abs(similarity);
        }
        
        private Matrix<double> GenerateDivergentFilter(Vector<double> divergentState)
        {
            var size = divergentState.Count;
            var filter = Matrix<double>.Build.DenseIdentity(size);
            
            // Amplify components that make this state divergent
            for (int i = 0; i < size; i++)
            {
                var amplification = 1 + Math.Abs(divergentState[i]) * 0.5;
                filter[i, i] = amplification;
            }
            
            return filter;
        }
        
        private Vector<double> MergeStreamStates(List<ConsciousnessStream> streams)
        {
            var totalWeight = 0.0;
            var mergedState = Vector<double>.Build.Dense(streams.First().CurrentState.Count);
            
            foreach (var stream in streams)
            {
                var weight = stream.Coherence * Math.Log(stream.ProcessingCount + 1);
                mergedState = mergedState.Add(stream.CurrentState.Multiply(weight));
                totalWeight += weight;
            }
            
            return totalWeight > 0 ? mergedState.Divide(totalWeight) : mergedState;
        }
        
        private Matrix<double> MergeConsciousnessFilters(List<ConsciousnessStream> streams)
        {
            var size = streams.First().ConsciousnessFilter.RowCount;
            var mergedFilter = Matrix<double>.Build.Dense(size, size);
            
            foreach (var stream in streams)
            {
                mergedFilter = mergedFilter.Add(stream.ConsciousnessFilter);
            }
            
            return mergedFilter.Divide(streams.Count);
        }
        
        private string GenerateStreamId(string prefix)
        {
            var timestamp = DateTime.UtcNow.Ticks;
            var random = quantum.Next(1000, 9999);
            return $"{prefix}_{timestamp:X}_{random}";
        }
        
        private void RecordTimelineEvent(Vector<double> input, ParallelConsciousnessResult result)
        {
            var timelineEvent = new TimelineEvent
            {
                InputMagnitude = input.L2Norm(),
                StreamCount = result.ActiveStreamCount,
                AverageDivergence = result.StreamAnalysis.AverageDivergence,
                GlobalCoherenceLevel = result.StreamResults.Average(r => r.Coherence),
                Timestamp = DateTime.UtcNow
            };
            
            eventHistory.Enqueue(timelineEvent);
            
            while (eventHistory.Count > 1000)
            {
                eventHistory.TryDequeue(out _);
            }
        }
        
        /// <summary>
        /// Get timeline system statistics
        /// </summary>
        public TimelineStats GetStats()
        {
            var events = eventHistory.ToList();
            
            return new TimelineStats
            {
                ActiveStreams = activeStreams.Count,
                MaxStreams = maxStreams,
                AverageCoherence = activeStreams.Values.Average(s => s.Coherence),
                TotalProcessingCount = activeStreams.Values.Sum(s => s.ProcessingCount),
                GlobalStateMagnitude = globalConsciousnessState.L2Norm(),
                LastGlobalUpdate = lastGlobalUpdate,
                StreamTypes = activeStreams.Values.GroupBy(s => s.Type)
                    .ToDictionary(g => g.Key, g => g.Count()),
                TotalEvents = events.Count
            };
        }
        
        /// <summary>
        /// Get information about active streams
        /// </summary>
        public List<StreamInfo> GetStreamInfo()
        {
            return activeStreams.Values.Select(s => new StreamInfo
            {
                Id = s.Id,
                Type = s.Type,
                Coherence = s.Coherence,
                ProcessingCount = s.ProcessingCount,
                CreationTime = s.CreationTime,
                LastUpdate = s.LastUpdate,
                ParentStreamId = s.ParentStreamId,
                StateMagnitude = s.CurrentState.L2Norm()
            }).ToList();
        }
        
        // Properties
        /// <summary>Public API</summary>
        public Vector<double> GlobalConsciousnessState => globalConsciousnessState.Clone();
        /// <summary>Gets the integer value.</summary>
        public int ActiveStreamCount => activeStreams.Count;
    }
    
    /// <summary>
    /// Consciousness stream representation
    /// </summary>
    public class ConsciousnessStream
    {
        /// <summary>Gets the string value.</summary>
        public string Id { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public StreamType Type { get; set; }
        /// <summary>See implementation for details.</summary>
        public Vector<double> CurrentState { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public Vector<double> InitialState { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public Matrix<double> ConsciousnessFilter { get; set; } = Matrix<double>.Build.DenseIdentity(1);
        /// <summary>Gets the timestamp.</summary>
        public DateTime CreationTime { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastUpdate { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Coherence { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ProcessingCount { get; set; }
        /// <summary>See implementation for details.</summary>
        public string? ParentStreamId { get; set; }
    }
    
    /// <summary>
    /// Stream types
    /// </summary>
    public enum StreamType
    {
        Main,       // Primary consciousness stream
        Divergent,  // Divergent exploration stream
        Merged,     // Merged from convergent streams
        Temporal    // Temporal exploration stream
    }
    
    /// <summary>
    /// Processing modes for consciousness streams
    /// </summary>
    public enum ProcessingMode
    {
        Balanced,       // Balanced exploration and stability
        Exploratory,    // High exploration, low stability
        Conservative,   // Low exploration, high stability
        Adaptive        // Adaptive based on stream coherence
    }
    
    /// <summary>
    /// Result of processing in a consciousness stream
    /// </summary>
    public class StreamResult
    {
        /// <summary>Gets the string value.</summary>
        public string StreamId { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public VectorExperience VectorExperience { get; set; } = new();
        /// <summary>See implementation for details.</summary>
        public EmotionalResponse EmotionalResponse { get; set; } = new();
        /// <summary>See implementation for details.</summary>
        public Vector<double> NewState { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets the numeric value.</summary>
        public double Coherence { get; set; }
        /// <summary>Gets the string value.</summary>
        public string MemoryId { get; set; } = "";
        /// <summary>Gets the time span.</summary>
        public TimeSpan ProcessingTime { get; set; }
    }
    
    /// <summary>
    /// Analysis of stream divergence and convergence
    /// </summary>
    public class StreamAnalysis
    {
        /// <summary>Gets the numeric value.</summary>
        public double AverageDivergence { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double MaxDivergence { get; set; }
        /// <summary>See implementation for details.</summary>
        public StreamResult? MostDivergentResult { get; set; }
        /// <summary>Gets the list.</summary>
        public List<string> ConvergentStreams { get; set; } = new();
        /// <summary>Gets the boolean flag.</summary>
        public bool ShouldCreateStream { get; set; }
        /// <summary>Gets the boolean flag.</summary>
        public bool ShouldMergeStreams { get; set; }
    }
    
    /// <summary>
    /// Result of parallel consciousness processing
    /// </summary>
    public class ParallelConsciousnessResult
    {
        /// <summary>Gets the list.</summary>
        public List<StreamResult> StreamResults { get; set; } = new();
        /// <summary>See implementation for details.</summary>
        public Vector<double> GlobalConsciousness { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>See implementation for details.</summary>
        public StreamAnalysis StreamAnalysis { get; set; } = new();
        /// <summary>Gets the integer value.</summary>
        public int ActiveStreamCount { get; set; }
        /// <summary>Gets the time span.</summary>
        public TimeSpan ProcessingTime { get; set; }
    }
    
    /// <summary>
    /// Timeline event record
    /// </summary>
    public class TimelineEvent
    {
        /// <summary>Gets the numeric value.</summary>
        public double InputMagnitude { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int StreamCount { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageDivergence { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double GlobalCoherenceLevel { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime Timestamp { get; set; }
    }
    
    /// <summary>
    /// Timeline system statistics
    /// </summary>
    public class TimelineStats
    {
        /// <summary>Gets the integer value.</summary>
        public int ActiveStreams { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int MaxStreams { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double AverageCoherence { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int TotalProcessingCount { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double GlobalStateMagnitude { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastGlobalUpdate { get; set; }
        /// <summary>Gets the dictionary.</summary>
        public Dictionary<StreamType, int> StreamTypes { get; set; } = new();
        /// <summary>Gets the integer value.</summary>
        public int TotalEvents { get; set; }
        
        /// <inheritdoc/>
        public override string ToString()
        {
            return $"TimelineStats[Streams={ActiveStreams}/{MaxStreams}, " +
                   $"Coherence={AverageCoherence:F3}, GlobalMag={GlobalStateMagnitude:F3}]";
        }
    }
    
    /// <summary>
    /// Stream information
    /// </summary>
    public class StreamInfo
    {
        /// <summary>Gets the string value.</summary>
        public string Id { get; set; } = "";
        /// <summary>See implementation for details.</summary>
        public StreamType Type { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double Coherence { get; set; }
        /// <summary>Gets the integer value.</summary>
        public int ProcessingCount { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime CreationTime { get; set; }
        /// <summary>Gets the timestamp.</summary>
        public DateTime LastUpdate { get; set; }
        /// <summary>See implementation for details.</summary>
        public string? ParentStreamId { get; set; }
        /// <summary>Gets the numeric value.</summary>
        public double StateMagnitude { get; set; }
        
        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Stream[{Id}, Type={Type}, Coherence={Coherence:F3}, " +
                   $"Processed={ProcessingCount}x]";
        }
    }
}