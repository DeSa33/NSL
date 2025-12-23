using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using ILGPU;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// GPU Memory Management System for NSL.
    ///
    /// Based on NVIDIA best practices and modern GPU memory architecture:
    /// - Memory pooling to reduce allocation overhead
    /// - VRAM usage monitoring and OOM prevention
    /// - Adaptive batch sizing based on available memory
    /// - Automatic garbage collection of unused tensors
    /// - Memory defragmentation support
    ///
    /// Key Concepts from GPU Architecture:
    /// - Dedicated VRAM is faster but limited (4-24GB typical)
    /// - Memory bandwidth is often the bottleneck (~192-1000 GB/s)
    /// - Coalesced memory access patterns are critical
    /// - Memory pooling reduces cudaMalloc overhead
    /// </summary>
    public class GpuMemoryManager : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly ConcurrentDictionary<long, MemoryPool> _pools = new();
        private readonly ConcurrentDictionary<IntPtr, AllocationInfo> _allocations = new();
        private readonly object _lock = new();
        private readonly Timer? _gcTimer;
        private bool _disposed;

        // Memory tracking
        private long _totalAllocated;
        private long _peakAllocated;
        private long _totalPooled;
        private int _allocationCount;
        private int _poolHits;
        private int _poolMisses;

        /// <summary>
        /// Memory pool configuration
        /// </summary>
        public class PoolConfig
        {
            /// <summary>Enable memory pooling (reuse allocations)</summary>
            public bool EnablePooling { get; set; } = true;

            /// <summary>Maximum memory to keep in pools (bytes)</summary>
            public long MaxPoolSizeBytes { get; set; } = 512 * 1024 * 1024; // 512MB default

            /// <summary>Enable periodic garbage collection of unused pooled memory</summary>
            public bool EnableAutoGC { get; set; } = true;

            /// <summary>GC interval in milliseconds</summary>
            public int GCIntervalMs { get; set; } = 30000; // 30 seconds

            /// <summary>Memory threshold (0-1) to trigger aggressive GC</summary>
            public float MemoryPressureThreshold { get; set; } = 0.85f;

            /// <summary>Reserve this percentage of VRAM for system use</summary>
            public float SystemReservePercent { get; set; } = 0.1f; // 10%

            /// <summary>Enable adaptive batch sizing based on memory</summary>
            public bool EnableAdaptiveBatching { get; set; } = true;
        }

        /// <summary>Public API</summary>
        public PoolConfig Config { get; }

        /// <summary>
        /// Memory statistics
        /// </summary>
        public class MemoryStats
        {
            /// <summary>Public API</summary>
            public long TotalVRAM { get; set; }
            /// <summary>Public API</summary>
            public long AvailableVRAM { get; set; }
            /// <summary>Public API</summary>
            public long UsedVRAM { get; set; }
            /// <summary>Public API</summary>
            public long AllocatedByNSL { get; set; }
            /// <summary>Public API</summary>
            public long PooledMemory { get; set; }
            /// <summary>Public API</summary>
            public long PeakAllocated { get; set; }
            /// <summary>Public API</summary>
            public int ActiveAllocations { get; set; }
            /// <summary>Public API</summary>
            public int PoolHits { get; set; }
            /// <summary>Public API</summary>
            public int PoolMisses { get; set; }
            /// <summary>Public API</summary>
            public float PoolHitRate => PoolHits + PoolMisses > 0
                ? (float)PoolHits / (PoolHits + PoolMisses)
                : 0;
            /// <summary>Public API</summary>
            public float MemoryPressure { get; set; }

            /// <summary>Public API</summary>
            public override string ToString() =>
                $"VRAM: {UsedVRAM / (1024 * 1024)}MB / {TotalVRAM / (1024 * 1024)}MB " +
                $"({MemoryPressure:P0} pressure), Pool: {PooledMemory / (1024 * 1024)}MB, " +
                $"Hit Rate: {PoolHitRate:P0}";
        }

        /// <summary>
        /// Information about a single allocation
        /// </summary>
        private class AllocationInfo
        {
            /// <summary>Public API</summary>
            public long Size { get; set; }
            /// <summary>Public API</summary>
            public DateTime AllocatedAt { get; set; }
            /// <summary>Public API</summary>
            public string? Tag { get; set; }
            /// <summary>Public API</summary>
            public bool FromPool { get; set; }
        }

        /// <summary>
        /// Memory pool for a specific size class
        /// </summary>
        private class MemoryPool
        {
            /// <summary>Public API</summary>
            public long SizeClass { get; }
            /// <summary>Public API</summary>
            public ConcurrentQueue<MemoryBuffer1D<float, Stride1D.Dense>> FloatBuffers { get; } = new();
            /// <summary>Public API</summary>
            public ConcurrentQueue<MemoryBuffer1D<sbyte, Stride1D.Dense>> ByteBuffers { get; } = new();
            /// <summary>Public API</summary>
            public long TotalPooledBytes => (FloatBuffers.Count * SizeClass * sizeof(float)) +
                                            (ByteBuffers.Count * SizeClass * sizeof(sbyte));
            /// <summary>Public API</summary>
            public DateTime LastAccess { get; set; } = DateTime.UtcNow;

            /// <summary>Public API</summary>
            public MemoryPool(long sizeClass)
            {
                SizeClass = sizeClass;
            }
        }

        /// <summary>Public API</summary>
        public GpuMemoryManager(Accelerator accelerator, PoolConfig? config = null)
        {
            _accelerator = accelerator;
            Config = config ?? new PoolConfig();

            if (Config.EnableAutoGC)
            {
                _gcTimer = new Timer(PerformGC, null, Config.GCIntervalMs, Config.GCIntervalMs);
            }
        }

        #region Memory Allocation

        /// <summary>
        /// Allocate a float buffer, using pool if available
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense> AllocateFloat(int size, string? tag = null)
        {
            if (size <= 0)
                throw new ArgumentException("Size must be positive", nameof(size));

            // Check memory pressure before allocation
            var stats = GetMemoryStats();
            if (stats.MemoryPressure > Config.MemoryPressureThreshold)
            {
                // Trigger GC to free pooled memory
                PerformGC(null);

                // If still under pressure, throw
                stats = GetMemoryStats();
                if (stats.AvailableVRAM < size * sizeof(float))
                {
                    throw new OutOfMemoryException(
                        $"Insufficient GPU memory. Requested: {size * sizeof(float) / (1024 * 1024)}MB, " +
                        $"Available: {stats.AvailableVRAM / (1024 * 1024)}MB");
                }
            }

            MemoryBuffer1D<float, Stride1D.Dense> buffer;
            var sizeClass = GetSizeClass(size);

            // Try to get from pool
            if (Config.EnablePooling && _pools.TryGetValue(sizeClass, out var pool))
            {
                if (pool.FloatBuffers.TryDequeue(out var pooledBuffer))
                {
                    Interlocked.Increment(ref _poolHits);
                    Interlocked.Add(ref _totalPooled, -sizeClass * sizeof(float));
                    pool.LastAccess = DateTime.UtcNow;

                    TrackAllocation(pooledBuffer.NativePtr, size * sizeof(float), tag, fromPool: true);
                    return pooledBuffer;
                }
            }

            // Allocate new buffer
            Interlocked.Increment(ref _poolMisses);
            buffer = _accelerator.Allocate1D<float>(size);

            TrackAllocation(buffer.NativePtr, size * sizeof(float), tag, fromPool: false);
            return buffer;
        }

        /// <summary>
        /// Allocate a byte buffer for INT8 operations
        /// </summary>
        public MemoryBuffer1D<sbyte, Stride1D.Dense> AllocateByte(int size, string? tag = null)
        {
            if (size <= 0)
                throw new ArgumentException("Size must be positive", nameof(size));

            var sizeClass = GetSizeClass(size);

            // Try to get from pool
            if (Config.EnablePooling && _pools.TryGetValue(sizeClass, out var pool))
            {
                if (pool.ByteBuffers.TryDequeue(out var pooledBuffer))
                {
                    Interlocked.Increment(ref _poolHits);
                    Interlocked.Add(ref _totalPooled, -sizeClass);
                    pool.LastAccess = DateTime.UtcNow;

                    TrackAllocation(pooledBuffer.NativePtr, size, tag, fromPool: true);
                    return pooledBuffer;
                }
            }

            // Allocate new
            Interlocked.Increment(ref _poolMisses);
            var buffer = _accelerator.Allocate1D<sbyte>(size);

            TrackAllocation(buffer.NativePtr, size, tag, fromPool: false);
            return buffer;
        }

        /// <summary>
        /// Return a buffer to the pool or dispose it
        /// </summary>
        public void Release(MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            if (buffer == null) return;

            UntrackAllocation(buffer.NativePtr);

            if (!Config.EnablePooling || _totalPooled >= Config.MaxPoolSizeBytes)
            {
                buffer.Dispose();
                return;
            }

            var sizeClass = GetSizeClass((int)buffer.Length);
            var pool = _pools.GetOrAdd(sizeClass, s => new MemoryPool(s));

            pool.FloatBuffers.Enqueue(buffer);
            pool.LastAccess = DateTime.UtcNow;
            Interlocked.Add(ref _totalPooled, sizeClass * sizeof(float));
        }

        /// <summary>
        /// Return a byte buffer to the pool
        /// </summary>
        public void Release(MemoryBuffer1D<sbyte, Stride1D.Dense> buffer)
        {
            if (buffer == null) return;

            UntrackAllocation(buffer.NativePtr);

            if (!Config.EnablePooling || _totalPooled >= Config.MaxPoolSizeBytes)
            {
                buffer.Dispose();
                return;
            }

            var sizeClass = GetSizeClass((int)buffer.Length);
            var pool = _pools.GetOrAdd(sizeClass, s => new MemoryPool(s));

            pool.ByteBuffers.Enqueue(buffer);
            pool.LastAccess = DateTime.UtcNow;
            Interlocked.Add(ref _totalPooled, sizeClass);
        }

        #endregion

        #region Adaptive Batch Sizing

        /// <summary>
        /// Calculate optimal batch size based on available memory and tensor dimensions.
        /// Prevents OOM by adapting to GPU memory constraints.
        /// </summary>
        /// <param name="sampleSizeBytes">Memory required per sample</param>
        /// <param name="preferredBatchSize">Desired batch size</param>
        /// <param name="minBatchSize">Minimum acceptable batch size</param>
        /// <returns>Optimal batch size that fits in memory</returns>
        public int GetOptimalBatchSize(long sampleSizeBytes, int preferredBatchSize, int minBatchSize = 1)
        {
            if (!Config.EnableAdaptiveBatching)
                return preferredBatchSize;

            var stats = GetMemoryStats();

            // Calculate available memory (with safety margin)
            long availableForBatch = (long)(stats.AvailableVRAM * (1.0 - Config.SystemReservePercent));

            // Account for intermediate buffers (roughly 3x for forward pass)
            long memoryPerSample = sampleSizeBytes * 3;

            // Guard against divide-by-zero (if sample size is 0 or very small)
            if (memoryPerSample <= 0 || availableForBatch <= 0)
                return preferredBatchSize;

            // Calculate max batch size that fits
            int maxBatchSize = (int)(availableForBatch / memoryPerSample);
            maxBatchSize = Math.Max(maxBatchSize, minBatchSize);

            // Return the smaller of preferred and max
            int optimalSize = Math.Min(preferredBatchSize, maxBatchSize);

            if (optimalSize < preferredBatchSize)
            {
                Console.WriteLine($"NSL Memory: Adjusted batch size {preferredBatchSize} -> {optimalSize} " +
                    $"(Available: {stats.AvailableVRAM / (1024 * 1024)}MB)");
            }

            return Math.Max(optimalSize, minBatchSize);
        }

        /// <summary>
        /// Estimate memory required for a model with given layer sizes.
        /// Useful for planning GPU memory usage before allocation.
        /// </summary>
        public long EstimateModelMemory(int[] layerSizes, int batchSize, bool includeGradients = true)
        {
            long total = 0;

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                // Weights
                long weights = (long)layerSizes[i] * layerSizes[i + 1] * sizeof(float);
                // Biases
                long biases = layerSizes[i + 1] * sizeof(float);
                // Activations
                long activations = (long)batchSize * layerSizes[i + 1] * sizeof(float);

                total += weights + biases + activations;

                if (includeGradients)
                {
                    // Gradient tensors mirror the weights and activations
                    total += weights + activations;
                }
            }

            return total;
        }

        /// <summary>
        /// Check if a model of given size can fit in GPU memory
        /// </summary>
        public bool CanFitModel(long modelSizeBytes)
        {
            var stats = GetMemoryStats();
            long availableBytes = (long)(stats.AvailableVRAM * (1.0 - Config.SystemReservePercent));
            return modelSizeBytes <= availableBytes;
        }

        #endregion

        #region Memory Statistics

        /// <summary>
        /// Get current memory statistics
        /// </summary>
        public MemoryStats GetMemoryStats()
        {
            long totalVRAM = GetTotalVRAM();
            long usedVRAM = GetUsedVRAM();
            long availableVRAM = totalVRAM - usedVRAM;

            return new MemoryStats
            {
                TotalVRAM = totalVRAM,
                UsedVRAM = usedVRAM,
                AvailableVRAM = availableVRAM,
                AllocatedByNSL = _totalAllocated,
                PooledMemory = _totalPooled,
                PeakAllocated = _peakAllocated,
                ActiveAllocations = _allocationCount,
                PoolHits = _poolHits,
                PoolMisses = _poolMisses,
                MemoryPressure = totalVRAM > 0 ? (float)usedVRAM / totalVRAM : 0
            };
        }

        private long GetTotalVRAM()
        {
            // ILGPU provides memory info through the accelerator
            return _accelerator.MemorySize;
        }

        private long GetUsedVRAM()
        {
            // Approximate based on our tracking
            // Note: This doesn't account for other processes using GPU memory
            return _totalAllocated;
        }

        private void TrackAllocation(IntPtr ptr, long size, string? tag, bool fromPool)
        {
            _allocations[ptr] = new AllocationInfo
            {
                Size = size,
                AllocatedAt = DateTime.UtcNow,
                Tag = tag,
                FromPool = fromPool
            };

            Interlocked.Add(ref _totalAllocated, size);
            Interlocked.Increment(ref _allocationCount);

            // Update peak
            long currentTotal = _totalAllocated;
            long currentPeak = _peakAllocated;
            while (currentTotal > currentPeak)
            {
                if (Interlocked.CompareExchange(ref _peakAllocated, currentTotal, currentPeak) == currentPeak)
                    break;
                currentPeak = _peakAllocated;
            }
        }

        private void UntrackAllocation(IntPtr ptr)
        {
            if (_allocations.TryRemove(ptr, out var info))
            {
                Interlocked.Add(ref _totalAllocated, -info.Size);
                Interlocked.Decrement(ref _allocationCount);
            }
        }

        #endregion

        #region Garbage Collection

        /// <summary>
        /// Force garbage collection of pooled memory
        /// </summary>
        public void ForceGC()
        {
            PerformGC(null);
        }

        private void PerformGC(object? state)
        {
            if (_disposed) return;

            lock (_lock)
            {
                var stats = GetMemoryStats();
                bool underPressure = stats.MemoryPressure > Config.MemoryPressureThreshold;
                var now = DateTime.UtcNow;

                foreach (var kvp in _pools)
                {
                    var pool = kvp.Value;

                    // Remove old pooled buffers or all if under memory pressure
                    var idleTime = now - pool.LastAccess;
                    bool shouldClear = underPressure || idleTime.TotalSeconds > 60;

                    if (shouldClear)
                    {
                        // Clear float buffers
                        while (pool.FloatBuffers.TryDequeue(out var buffer))
                        {
                            Interlocked.Add(ref _totalPooled, -pool.SizeClass * sizeof(float));
                            buffer.Dispose();
                        }

                        // Clear byte buffers
                        while (pool.ByteBuffers.TryDequeue(out var buffer))
                        {
                            Interlocked.Add(ref _totalPooled, -pool.SizeClass);
                            buffer.Dispose();
                        }
                    }
                }

                // Remove empty pools
                var emptyPools = _pools.Where(kvp =>
                    kvp.Value.FloatBuffers.IsEmpty && kvp.Value.ByteBuffers.IsEmpty).ToList();
                foreach (var kvp in emptyPools)
                {
                    _pools.TryRemove(kvp.Key, out _);
                }
            }
        }

        /// <summary>
        /// Clear all pooled memory immediately
        /// </summary>
        public void ClearPools()
        {
            lock (_lock)
            {
                foreach (var kvp in _pools)
                {
                    var pool = kvp.Value;

                    while (pool.FloatBuffers.TryDequeue(out var buffer))
                    {
                        buffer.Dispose();
                    }
                    while (pool.ByteBuffers.TryDequeue(out var buffer))
                    {
                        buffer.Dispose();
                    }
                }

                _pools.Clear();
                _totalPooled = 0;
            }
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Round up to nearest power of 2 size class for efficient pooling.
        /// Reduces fragmentation by using standard sizes.
        /// </summary>
        private static long GetSizeClass(int size)
        {
            // Minimum size class is 256 elements
            if (size <= 256) return 256;

            // Round up to nearest power of 2
            long sizeClass = 256;
            while (sizeClass < size)
            {
                sizeClass *= 2;
            }
            return sizeClass;
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _gcTimer?.Dispose();
                ClearPools();
                _disposed = true;
            }
        }
    }
}