using System;
using System.Collections.Generic;
using System.Threading;

namespace NSL.Tensor
{
    /// <summary>
    /// LRU cache for tensor operations with memory-aware eviction.
    /// Inspired by pmx_utils TensorCache.
    /// Useful for caching attention computations, intermediate results, etc.
    /// </summary>
    public class TensorCache
    {
        private readonly int _maxEntries;
        private readonly long _maxMemoryBytes;
        private readonly Dictionary<string, CacheEntry> _cache;
        private readonly LinkedList<string> _accessOrder;
        private readonly ReaderWriterLockSlim _lock;
        private long _currentMemoryBytes;
        private long _hits;
        private long _misses;

        /// <summary>
        /// Create a new tensor cache.
        /// </summary>
        /// <param name="maxEntries">Maximum number of entries (default: 1000)</param>
        /// <param name="maxMemoryMB">Maximum memory in MB (default: 512)</param>
        public TensorCache(int maxEntries = 1000, int maxMemoryMB = 512)
        {
            _maxEntries = maxEntries;
            _maxMemoryBytes = maxMemoryMB * 1024L * 1024L;
            _cache = new Dictionary<string, CacheEntry>();
            _accessOrder = new LinkedList<string>();
            _lock = new ReaderWriterLockSlim();
            _currentMemoryBytes = 0;
            _hits = 0;
            _misses = 0;
        }

        /// <summary>
        /// Get a tensor from cache, or compute and cache it.
        /// </summary>
        /// <param name="key">Cache key</param>
        /// <param name="computeFunc">Function to compute tensor if not cached</param>
        /// <returns>Cached or computed tensor</returns>
        public Tensor GetOrCompute(string key, Func<Tensor> computeFunc)
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                if (_cache.TryGetValue(key, out var entry))
                {
                    // Cache hit - move to front of LRU list
                    _lock.EnterWriteLock();
                    try
                    {
                        _accessOrder.Remove(entry.Node);
                        _accessOrder.AddFirst(entry.Node);
                        Interlocked.Increment(ref _hits);
                        return entry.Tensor;
                    }
                    finally
                    {
                        _lock.ExitWriteLock();
                    }
                }

                // Cache miss - compute and store
                Interlocked.Increment(ref _misses);
                var tensor = computeFunc();

                _lock.EnterWriteLock();
                try
                {
                    Put(key, tensor);
                    return tensor;
                }
                finally
                {
                    _lock.ExitWriteLock();
                }
            }
            finally
            {
                _lock.ExitUpgradeableReadLock();
            }
        }

        /// <summary>
        /// Try to get a tensor from cache.
        /// </summary>
        public bool TryGet(string key, out Tensor? tensor)
        {
            _lock.EnterReadLock();
            try
            {
                if (_cache.TryGetValue(key, out var entry))
                {
                    tensor = entry.Tensor;
                    Interlocked.Increment(ref _hits);
                    return true;
                }
                tensor = null;
                Interlocked.Increment(ref _misses);
                return false;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <summary>
        /// Put a tensor in the cache.
        /// </summary>
        public void Put(string key, Tensor tensor)
        {
            var memSize = tensor.Data.Length * sizeof(double);

            _lock.EnterWriteLock();
            try
            {
                // If key exists, remove old entry
                if (_cache.TryGetValue(key, out var oldEntry))
                {
                    _currentMemoryBytes -= oldEntry.MemoryBytes;
                    _accessOrder.Remove(oldEntry.Node);
                    _cache.Remove(key);
                }

                // Evict entries if needed
                while ((_cache.Count >= _maxEntries || _currentMemoryBytes + memSize > _maxMemoryBytes)
                       && _accessOrder.Count > 0)
                {
                    EvictLRU();
                }

                // Add new entry
                var node = _accessOrder.AddFirst(key);
                _cache[key] = new CacheEntry(tensor, node, memSize);
                _currentMemoryBytes += memSize;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Remove a specific key from cache.
        /// </summary>
        public bool Remove(string key)
        {
            _lock.EnterWriteLock();
            try
            {
                if (_cache.TryGetValue(key, out var entry))
                {
                    _currentMemoryBytes -= entry.MemoryBytes;
                    _accessOrder.Remove(entry.Node);
                    _cache.Remove(key);
                    return true;
                }
                return false;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Clear all cached entries.
        /// </summary>
        public void Clear()
        {
            _lock.EnterWriteLock();
            try
            {
                _cache.Clear();
                _accessOrder.Clear();
                _currentMemoryBytes = 0;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        private void EvictLRU()
        {
            if (_accessOrder.Last == null) return;

            var lruKey = _accessOrder.Last.Value;
            if (_cache.TryGetValue(lruKey, out var entry))
            {
                _currentMemoryBytes -= entry.MemoryBytes;
                _cache.Remove(lruKey);
            }
            _accessOrder.RemoveLast();
        }

        /// <summary>Number of entries in cache</summary>
        public int Count
        {
            get
            {
                _lock.EnterReadLock();
                try { return _cache.Count; }
                finally { _lock.ExitReadLock(); }
            }
        }

        /// <summary>Current memory usage in bytes</summary>
        public long MemoryBytes => Interlocked.Read(ref _currentMemoryBytes);

        /// <summary>Current memory usage in MB</summary>
        public double MemoryMB => MemoryBytes / (1024.0 * 1024.0);

        /// <summary>Cache hit count</summary>
        public long Hits => Interlocked.Read(ref _hits);

        /// <summary>Cache miss count</summary>
        public long Misses => Interlocked.Read(ref _misses);

        /// <summary>Cache hit rate (0.0 to 1.0)</summary>
        public double HitRate
        {
            get
            {
                var total = Hits + Misses;
                return total > 0 ? (double)Hits / total : 0.0;
            }
        }

        /// <summary>Get cache statistics</summary>
        public CacheStats GetStats()
        {
            return new CacheStats(Count, MemoryBytes, Hits, Misses, HitRate);
        }

        private class CacheEntry
        {
            /// <summary>Public API</summary>
            public Tensor Tensor { get; }
            /// <summary>Public API</summary>
            public LinkedListNode<string> Node { get; }
            /// <summary>Public API</summary>
            public long MemoryBytes { get; }

            /// <summary>Public API</summary>
            public CacheEntry(Tensor tensor, LinkedListNode<string> node, long memoryBytes)
            {
                Tensor = tensor;
                Node = node;
                MemoryBytes = memoryBytes;
            }
        }
    }

    /// <summary>
    /// Cache statistics.
    /// </summary>
    public readonly struct CacheStats
    {
        /// <summary>Public API</summary>
        public int EntryCount { get; }
        /// <summary>Public API</summary>
        public long MemoryBytes { get; }
        /// <summary>Public API</summary>
        public long Hits { get; }
        /// <summary>Public API</summary>
        public long Misses { get; }
        /// <summary>Public API</summary>
        public double HitRate { get; }

        /// <summary>Public API</summary>
        public CacheStats(int entryCount, long memoryBytes, long hits, long misses, double hitRate)
        {
            EntryCount = entryCount;
            MemoryBytes = memoryBytes;
            Hits = hits;
            Misses = misses;
            HitRate = hitRate;
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"CacheStats(entries={EntryCount}, memory={MemoryBytes / 1024.0 / 1024.0:F2}MB, " +
                   $"hits={Hits}, misses={Misses}, hitRate={HitRate:P1})";
        }
    }
}