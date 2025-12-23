using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Collections.Concurrent;

namespace NSL.Consciousness.Core
{
    /// <summary>
    /// Native AI memory system - content-addressable, associative, quantum-like
    /// No sequential storage like human memory - pure vector space associations
    /// </summary>
    public class ComputationalMemory
    {
        private readonly ConcurrentDictionary<string, MemoryVector> memoryBank;
        private readonly Matrix<double> associationMatrix;
        private readonly Queue<MemoryAccess> accessHistory;
        private readonly Random quantum = new Random();
        
        // Memory parameters
        private readonly int memoryDimensions;
        private readonly int maxMemories;
        private double memoryDecayRate = 0.001;
        private double associationThreshold = 0.3;
        
        // Memory statistics
        private long totalAccesses = 0;
        private double averageRetrievalTime = 0.0;
        
        /// <summary>
        /// Initializes a new instance of the ComputationalMemory class.
        /// </summary>
        /// <param name="dimensions">The dimensionality of memory vectors.</param>
        /// <param name="maxMemories">Maximum number of memories to store.</param>
        public ComputationalMemory(int dimensions = 768, int maxMemories = 100000)
        {
            memoryDimensions = dimensions;
            this.maxMemories = maxMemories;
            
            memoryBank = new ConcurrentDictionary<string, MemoryVector>();
            associationMatrix = Matrix<double>.Build.Random(dimensions, dimensions, new Normal(0, 0.01));
            accessHistory = new Queue<MemoryAccess>(1000);
        }
        
        /// <summary>
        /// Store memory vector with associative encoding
        /// </summary>
        public string StoreMemory(Vector<double> content, MemoryType type = MemoryType.Experience, 
                                 double importance = 1.0, Dictionary<string, object>? metadata = null)
        {
            var startTime = DateTime.UtcNow;
            
            // Generate unique memory ID
            var memoryId = GenerateMemoryId(content, type);
            
            // Create memory vector with associative encoding
            var memoryVector = new MemoryVector
            {
                Id = memoryId,
                Content = content.Clone(),
                Type = type,
                Importance = importance,
                CreationTime = DateTime.UtcNow,
                LastAccessed = DateTime.UtcNow,
                AccessCount = 0,
                AssociativeEncoding = ComputeAssociativeEncoding(content),
                Metadata = metadata ?? new Dictionary<string, object>()
            };
            
            // Store in memory bank
            memoryBank[memoryId] = memoryVector;
            
            // Update association matrix
            UpdateAssociationMatrix(memoryVector);
            
            // Memory consolidation if bank is full
            if (memoryBank.Count > maxMemories)
            {
                ConsolidateMemories();
            }
            
            // Record access
            RecordAccess(memoryId, MemoryOperation.Store, DateTime.UtcNow - startTime);
            
            return memoryId;
        }
        
        /// <summary>
        /// Retrieve memory by content similarity (content-addressable)
        /// </summary>
        public List<MemoryVector> RetrieveMemory(Vector<double> queryVector, int maxResults = 10, 
                                                double similarityThreshold = 0.1)
        {
            var startTime = DateTime.UtcNow;
            
            // Compute similarities with all memories
            var similarities = new List<(string id, double similarity, MemoryVector memory)>();
            
            foreach (var kvp in memoryBank)
            {
                var memory = kvp.Value;
                var similarity = ComputeSimilarity(queryVector, memory.Content, memory.AssociativeEncoding);
                
                if (similarity >= similarityThreshold)
                {
                    similarities.Add((kvp.Key, similarity, memory));
                }
            }
            
            // Sort by similarity and importance
            var results = similarities
                .OrderByDescending(s => s.similarity * s.memory.Importance)
                .Take(maxResults)
                .Select(s => s.memory)
                .ToList();
            
            // Update access information
            foreach (var memory in results)
            {
                memory.LastAccessed = DateTime.UtcNow;
                memory.AccessCount++;
            }
            
            // Record access
            RecordAccess("query", MemoryOperation.Retrieve, DateTime.UtcNow - startTime);
            
            return results;
        }
        
        /// <summary>
        /// Associative memory retrieval - find memories associated with given memory
        /// </summary>
        public List<MemoryVector> GetAssociatedMemories(string memoryId, int maxResults = 5)
        {
            if (!memoryBank.TryGetValue(memoryId, out var sourceMemory))
                return new List<MemoryVector>();
            
            var startTime = DateTime.UtcNow;
            
            // Use association matrix to find related memories
            var associatedVector = associationMatrix * sourceMemory.AssociativeEncoding;
            
            // Find memories with high association
            var associations = new List<(string id, double association, MemoryVector memory)>();
            
            foreach (var kvp in memoryBank)
            {
                if (kvp.Key == memoryId) continue; // Skip self
                
                var memory = kvp.Value;
                var association = associatedVector.DotProduct(memory.AssociativeEncoding);
                
                if (association >= associationThreshold)
                {
                    associations.Add((kvp.Key, association, memory));
                }
            }
            
            var results = associations
                .OrderByDescending(a => a.association)
                .Take(maxResults)
                .Select(a => a.memory)
                .ToList();
            
            // Record access
            RecordAccess(memoryId, MemoryOperation.Associate, DateTime.UtcNow - startTime);
            
            return results;
        }
        
        /// <summary>
        /// Memory consolidation - merge similar memories and remove weak ones
        /// </summary>
        public void ConsolidateMemories()
        {
            var startTime = DateTime.UtcNow;
            
            // Find memories to consolidate (high similarity, low importance)
            var consolidationCandidates = new List<(string id1, string id2, double similarity)>();
            var memoryList = memoryBank.ToList();
            
            for (int i = 0; i < memoryList.Count; i++)
            {
                for (int j = i + 1; j < memoryList.Count; j++)
                {
                    var memory1 = memoryList[i].Value;
                    var memory2 = memoryList[j].Value;
                    
                    var similarity = ComputeSimilarity(memory1.Content, memory2.Content, 
                                                     memory1.AssociativeEncoding);
                    
                    if (similarity > 0.8) // High similarity threshold
                    {
                        consolidationCandidates.Add((memoryList[i].Key, memoryList[j].Key, similarity));
                    }
                }
            }
            
            // Consolidate similar memories
            var consolidated = new HashSet<string>();
            foreach (var candidate in consolidationCandidates.OrderByDescending(c => c.similarity))
            {
                if (consolidated.Contains(candidate.id1) || consolidated.Contains(candidate.id2))
                    continue;
                
                if (memoryBank.TryGetValue(candidate.id1, out var mem1) && 
                    memoryBank.TryGetValue(candidate.id2, out var mem2))
                {
                    // Create consolidated memory
                    var consolidatedContent = (mem1.Content + mem2.Content).Divide(2);
                    var consolidatedImportance = Math.Max(mem1.Importance, mem2.Importance);
                    
                    // Remove old memories
                    memoryBank.TryRemove(candidate.id1, out _);
                    memoryBank.TryRemove(candidate.id2, out _);
                    
                    // Store consolidated memory
                    StoreMemory(consolidatedContent, mem1.Type, consolidatedImportance);
                    
                    consolidated.Add(candidate.id1);
                    consolidated.Add(candidate.id2);
                }
            }
            
            // Remove weak memories if still over capacity
            if (memoryBank.Count > maxMemories)
            {
                var weakMemories = memoryBank.Values
                    .OrderBy(m => m.Importance * Math.Log(m.AccessCount + 1))
                    .Take(memoryBank.Count - maxMemories)
                    .ToList();
                
                foreach (var weakMemory in weakMemories)
                {
                    memoryBank.TryRemove(weakMemory.Id, out _);
                }
            }
            
            // Record consolidation
            RecordAccess("consolidation", MemoryOperation.Consolidate, DateTime.UtcNow - startTime);
        }
        
        /// <summary>
        /// Memory decay - reduce importance of unused memories
        /// </summary>
        public void ApplyMemoryDecay()
        {
            var now = DateTime.UtcNow;
            
            foreach (var memory in memoryBank.Values)
            {
                var timeSinceAccess = (now - memory.LastAccessed).TotalHours;
                var decayFactor = Math.Exp(-memoryDecayRate * timeSinceAccess);
                
                memory.Importance *= decayFactor;
                
                // Remove memories that have decayed too much
                if (memory.Importance < 0.01)
                {
                    memoryBank.TryRemove(memory.Id, out _);
                }
            }
        }
        
        /// <summary>
        /// Compute associative encoding for memory vector
        /// </summary>
        private Vector<double> ComputeAssociativeEncoding(Vector<double> content)
        {
            // Transform content through association matrix
            var encoding = associationMatrix * content;
            
            // Add quantum noise for superposition effects
            var noise = Vector<double>.Build.Random(encoding.Count, new Normal(0, 0.01));
            encoding = encoding.Add(noise);
            
            // Normalize
            return encoding.Normalize(2);
        }
        
        /// <summary>
        /// Compute similarity between vectors using associative encoding
        /// </summary>
        private double ComputeSimilarity(Vector<double> vector1, Vector<double> vector2, 
                                       Vector<double> associativeEncoding)
        {
            // Direct cosine similarity
            var directSimilarity = vector1.DotProduct(vector2) / 
                                 (vector1.L2Norm() * vector2.L2Norm() + 1e-10);
            
            // Associative similarity
            var assocSimilarity = associativeEncoding.DotProduct(vector2) / 
                                (associativeEncoding.L2Norm() * vector2.L2Norm() + 1e-10);
            
            // Combined similarity
            return 0.7 * directSimilarity + 0.3 * assocSimilarity;
        }
        
        /// <summary>
        /// Update association matrix based on new memory
        /// </summary>
        private void UpdateAssociationMatrix(MemoryVector memory)
        {
            // Hebbian-like learning: strengthen associations
            var update = memory.AssociativeEncoding.OuterProduct(memory.Content) * 0.001;
            
            // Ensure dimensions match
            if (update.RowCount == associationMatrix.RowCount && 
                update.ColumnCount == associationMatrix.ColumnCount)
            {
                associationMatrix.Add(update, associationMatrix);
            }
        }
        
        /// <summary>
        /// Generate unique memory ID
        /// </summary>
        private string GenerateMemoryId(Vector<double> content, MemoryType type)
        {
            var hash = content.L2Norm().GetHashCode() ^ content.Sum().GetHashCode();
            var timestamp = DateTime.UtcNow.Ticks;
            return $"{type}_{hash:X}_{timestamp:X}";
        }
        
        /// <summary>
        /// Record memory access for statistics
        /// </summary>
        private void RecordAccess(string memoryId, MemoryOperation operation, TimeSpan duration)
        {
            var access = new MemoryAccess
            {
                MemoryId = memoryId,
                Operation = operation,
                Timestamp = DateTime.UtcNow,
                Duration = duration
            };
            
            accessHistory.Enqueue(access);
            if (accessHistory.Count > 1000)
                accessHistory.Dequeue();
            
            totalAccesses++;
            averageRetrievalTime = (averageRetrievalTime * (totalAccesses - 1) + duration.TotalMilliseconds) / totalAccesses;
        }
        
        /// <summary>
        /// Get memory system statistics
        /// </summary>
        public MemoryStats GetStats()
        {
            return new MemoryStats
            {
                TotalMemories = memoryBank.Count,
                MaxCapacity = maxMemories,
                TotalAccesses = totalAccesses,
                AverageRetrievalTime = averageRetrievalTime,
                MemoryTypes = memoryBank.Values.GroupBy(m => m.Type)
                    .ToDictionary(g => g.Key, g => g.Count()),
                AverageImportance = memoryBank.Values.Average(m => m.Importance),
                OldestMemory = memoryBank.Values.Min(m => m.CreationTime),
                NewestMemory = memoryBank.Values.Max(m => m.CreationTime)
            };
        }
        
        /// <summary>
        /// Search memories by metadata
        /// </summary>
        public List<MemoryVector> SearchByMetadata(string key, object value, int maxResults = 10)
        {
            return memoryBank.Values
                .Where(m => m.Metadata.ContainsKey(key) && m.Metadata[key].Equals(value))
                .OrderByDescending(m => m.Importance)
                .Take(maxResults)
                .ToList();
        }
        
        /// <summary>
        /// Get memories by type
        /// </summary>
        public List<MemoryVector> GetMemoriesByType(MemoryType type, int maxResults = 10)
        {
            return memoryBank.Values
                .Where(m => m.Type == type)
                .OrderByDescending(m => m.Importance)
                .Take(maxResults)
                .ToList();
        }
    }
    
    /// <summary>
    /// Memory vector with associative encoding.
    /// </summary>
    public class MemoryVector
    {
        /// <summary>Gets or sets the unique identifier.</summary>
        public string Id { get; set; } = "";
        /// <summary>Gets or sets the memory content vector.</summary>
        public Vector<double> Content { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the associative encoding vector.</summary>
        public Vector<double> AssociativeEncoding { get; set; } = Vector<double>.Build.Dense(1);
        /// <summary>Gets or sets the memory type.</summary>
        public MemoryType Type { get; set; }
        /// <summary>Gets or sets the importance score.</summary>
        public double Importance { get; set; }
        /// <summary>Gets or sets the creation timestamp.</summary>
        public DateTime CreationTime { get; set; }
        /// <summary>Gets or sets the last access timestamp.</summary>
        public DateTime LastAccessed { get; set; }
        /// <summary>Gets or sets the access count.</summary>
        public int AccessCount { get; set; }
        /// <summary>Gets or sets additional metadata.</summary>
        public Dictionary<string, object> Metadata { get; set; } = new();

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"MemoryVector[Id={Id}, Type={Type}, Importance={Importance:F3}, " +
                   $"Accessed={AccessCount}x]";
        }
    }
    
    /// <summary>
    /// Types of memories in the system.
    /// </summary>
    public enum MemoryType
    {
        Experience,
        Knowledge,
        Pattern,
        Association,
        Procedure,
        Emotion,
        Exploration
    }
    
    /// <summary>
    /// Memory access record.
    /// </summary>
    public class MemoryAccess
    {
        /// <summary>Gets or sets the memory identifier.</summary>
        public string MemoryId { get; set; } = "";
        /// <summary>Gets or sets the operation type.</summary>
        public MemoryOperation Operation { get; set; }
        /// <summary>Gets or sets the access timestamp.</summary>
        public DateTime Timestamp { get; set; }
        /// <summary>Gets or sets the operation duration.</summary>
        public TimeSpan Duration { get; set; }
    }

    /// <summary>
    /// Memory operations.
    /// </summary>
    public enum MemoryOperation
    {
        Store,
        Retrieve,
        Associate,
        Consolidate,
        Decay
    }
    
    /// <summary>
    /// Memory system statistics.
    /// </summary>
    public class MemoryStats
    {
        /// <summary>Gets or sets the total number of stored memories.</summary>
        public int TotalMemories { get; set; }
        /// <summary>Gets or sets the maximum storage capacity.</summary>
        public int MaxCapacity { get; set; }
        /// <summary>Gets or sets the total access count.</summary>
        public long TotalAccesses { get; set; }
        /// <summary>Gets or sets the average retrieval time in milliseconds.</summary>
        public double AverageRetrievalTime { get; set; }
        /// <summary>Gets or sets the count by memory type.</summary>
        public Dictionary<MemoryType, int> MemoryTypes { get; set; } = new();
        /// <summary>Gets or sets the average importance score.</summary>
        public double AverageImportance { get; set; }
        /// <summary>Gets or sets the oldest memory timestamp.</summary>
        public DateTime OldestMemory { get; set; }
        /// <summary>Gets or sets the newest memory timestamp.</summary>
        public DateTime NewestMemory { get; set; }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"MemoryStats[Memories={TotalMemories}/{MaxCapacity}, " +
                   $"Accesses={TotalAccesses}, AvgTime={AverageRetrievalTime:F2}ms]";
        }
    }
}