using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

// GPU support requires ILGPU package - uncomment when available:
// #define HAS_ILGPU
#if HAS_ILGPU
using ILGPU;
using ILGPU.Runtime;
#endif

namespace NSL.Tensor
{
#if HAS_ILGPU
    #region GPU Memory Pool

    /// <summary>
    /// GPU memory buffer pool for efficient VRAM allocation and reuse.
    /// Minimizes GPU memory fragmentation and allocation overhead.
    /// </summary>
    public class GpuMemoryPool : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly ConcurrentDictionary<long, ConcurrentBag<MemoryBuffer1D<float, Stride1D.Dense>>> _floatBufferPools;
        private readonly ConcurrentDictionary<long, ConcurrentBag<MemoryBuffer1D<double, Stride1D.Dense>>> _doubleBufferPools;
        private readonly ConcurrentDictionary<long, ConcurrentBag<MemoryBuffer1D<int, Stride1D.Dense>>> _intBufferPools;

        // Pinned memory for async transfers
        private readonly ConcurrentDictionary<long, ConcurrentBag<PinnedMemoryHandle>> _pinnedMemoryPools;

        // Statistics
        private long _gpuAllocations;
        private long _gpuPoolHits;
        private long _gpuPoolMisses;
        private long _totalGpuBytesAllocated;
        private long _currentGpuBytesInUse;
        private long _peakGpuBytesInUse;
        private long _pinnedMemoryBytes;

        // Configuration
        private readonly long _maxGpuPoolBytes;
        private readonly long _maxPinnedBytes;
        private long _currentGpuPoolBytes;

        private bool _disposed;
        private readonly object _statsLock = new();

        /// <summary>
        /// Creates a new GPU memory pool.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator</param>
        /// <param name="maxPoolSizeMB">Maximum GPU pool size in MB (default: 2048)</param>
        /// <param name="maxPinnedMB">Maximum pinned host memory in MB (default: 512)</param>
        public GpuMemoryPool(Accelerator accelerator, int maxPoolSizeMB = 2048, int maxPinnedMB = 512)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _maxGpuPoolBytes = maxPoolSizeMB * 1024L * 1024L;
            _maxPinnedBytes = maxPinnedMB * 1024L * 1024L;

            _floatBufferPools = new ConcurrentDictionary<long, ConcurrentBag<MemoryBuffer1D<float, Stride1D.Dense>>>();
            _doubleBufferPools = new ConcurrentDictionary<long, ConcurrentBag<MemoryBuffer1D<double, Stride1D.Dense>>>();
            _intBufferPools = new ConcurrentDictionary<long, ConcurrentBag<MemoryBuffer1D<int, Stride1D.Dense>>>();
            _pinnedMemoryPools = new ConcurrentDictionary<long, ConcurrentBag<PinnedMemoryHandle>>();
        }

        #region GPU Buffer Allocation

        /// <summary>
        /// Rents a GPU float buffer from pool or allocates new.
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense> RentFloatBuffer(long length)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(GpuMemoryPool));

            var bucket = GetGpuBucket(length);
            var pool = _floatBufferPools.GetOrAdd(bucket, _ => new ConcurrentBag<MemoryBuffer1D<float, Stride1D.Dense>>());

            Interlocked.Increment(ref _gpuAllocations);

            if (pool.TryTake(out var buffer) && buffer.Length >= length)
            {
                Interlocked.Increment(ref _gpuPoolHits);
                Interlocked.Add(ref _currentGpuPoolBytes, -buffer.Length * sizeof(float));
                UpdateGpuBytesInUse(buffer.Length * sizeof(float));
                return buffer;
            }

            Interlocked.Increment(ref _gpuPoolMisses);
            var newBuffer = _accelerator.Allocate1D<float>(bucket);
            Interlocked.Add(ref _totalGpuBytesAllocated, bucket * sizeof(float));
            UpdateGpuBytesInUse(bucket * sizeof(float));
            return newBuffer;
        }

        /// <summary>
        /// Rents a GPU double buffer from pool or allocates new.
        /// </summary>
        public MemoryBuffer1D<double, Stride1D.Dense> RentDoubleBuffer(long length)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(GpuMemoryPool));

            var bucket = GetGpuBucket(length);
            var pool = _doubleBufferPools.GetOrAdd(bucket, _ => new ConcurrentBag<MemoryBuffer1D<double, Stride1D.Dense>>());

            Interlocked.Increment(ref _gpuAllocations);

            if (pool.TryTake(out var buffer) && buffer.Length >= length)
            {
                Interlocked.Increment(ref _gpuPoolHits);
                Interlocked.Add(ref _currentGpuPoolBytes, -buffer.Length * sizeof(double));
                UpdateGpuBytesInUse(buffer.Length * sizeof(double));
                return buffer;
            }

            Interlocked.Increment(ref _gpuPoolMisses);
            var newBuffer = _accelerator.Allocate1D<double>(bucket);
            Interlocked.Add(ref _totalGpuBytesAllocated, bucket * sizeof(double));
            UpdateGpuBytesInUse(bucket * sizeof(double));
            return newBuffer;
        }

        /// <summary>
        /// Rents a GPU int buffer from pool or allocates new.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> RentIntBuffer(long length)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(GpuMemoryPool));

            var bucket = GetGpuBucket(length);
            var pool = _intBufferPools.GetOrAdd(bucket, _ => new ConcurrentBag<MemoryBuffer1D<int, Stride1D.Dense>>());

            Interlocked.Increment(ref _gpuAllocations);

            if (pool.TryTake(out var buffer) && buffer.Length >= length)
            {
                Interlocked.Increment(ref _gpuPoolHits);
                Interlocked.Add(ref _currentGpuPoolBytes, -buffer.Length * sizeof(int));
                UpdateGpuBytesInUse(buffer.Length * sizeof(int));
                return buffer;
            }

            Interlocked.Increment(ref _gpuPoolMisses);
            var newBuffer = _accelerator.Allocate1D<int>(bucket);
            Interlocked.Add(ref _totalGpuBytesAllocated, bucket * sizeof(int));
            UpdateGpuBytesInUse(bucket * sizeof(int));
            return newBuffer;
        }

        /// <summary>
        /// Returns a GPU float buffer to the pool.
        /// </summary>
        public void ReturnFloatBuffer(MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            if (_disposed || buffer == null) return;

            if (Interlocked.Read(ref _currentGpuPoolBytes) >= _maxGpuPoolBytes)
            {
                buffer.Dispose();
                return;
            }

            var bucket = GetGpuBucket(buffer.Length);
            var pool = _floatBufferPools.GetOrAdd(bucket, _ => new ConcurrentBag<MemoryBuffer1D<float, Stride1D.Dense>>());

            pool.Add(buffer);
            Interlocked.Add(ref _currentGpuPoolBytes, buffer.Length * sizeof(float));
            UpdateGpuBytesInUse(-buffer.Length * sizeof(float));
        }

        /// <summary>
        /// Returns a GPU double buffer to the pool.
        /// </summary>
        public void ReturnDoubleBuffer(MemoryBuffer1D<double, Stride1D.Dense> buffer)
        {
            if (_disposed || buffer == null) return;

            if (Interlocked.Read(ref _currentGpuPoolBytes) >= _maxGpuPoolBytes)
            {
                buffer.Dispose();
                return;
            }

            var bucket = GetGpuBucket(buffer.Length);
            var pool = _doubleBufferPools.GetOrAdd(bucket, _ => new ConcurrentBag<MemoryBuffer1D<double, Stride1D.Dense>>());

            pool.Add(buffer);
            Interlocked.Add(ref _currentGpuPoolBytes, buffer.Length * sizeof(double));
            UpdateGpuBytesInUse(-buffer.Length * sizeof(double));
        }

        /// <summary>
        /// Returns a GPU int buffer to the pool.
        /// </summary>
        public void ReturnIntBuffer(MemoryBuffer1D<int, Stride1D.Dense> buffer)
        {
            if (_disposed || buffer == null) return;

            if (Interlocked.Read(ref _currentGpuPoolBytes) >= _maxGpuPoolBytes)
            {
                buffer.Dispose();
                return;
            }

            var bucket = GetGpuBucket(buffer.Length);
            var pool = _intBufferPools.GetOrAdd(bucket, _ => new ConcurrentBag<MemoryBuffer1D<int, Stride1D.Dense>>());

            pool.Add(buffer);
            Interlocked.Add(ref _currentGpuPoolBytes, buffer.Length * sizeof(int));
            UpdateGpuBytesInUse(-buffer.Length * sizeof(int));
        }

        #endregion

        #region Pinned Memory

        /// <summary>
        /// Rents pinned host memory for async GPU transfers.
        /// </summary>
        public PinnedMemoryHandle RentPinnedMemory(long sizeBytes)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(GpuMemoryPool));

            var bucket = GetGpuBucket(sizeBytes);
            var pool = _pinnedMemoryPools.GetOrAdd(bucket, _ => new ConcurrentBag<PinnedMemoryHandle>());

            if (pool.TryTake(out var handle) && handle.SizeBytes >= sizeBytes)
            {
                Interlocked.Add(ref _pinnedMemoryBytes, -handle.SizeBytes);
                return handle;
            }

            // Allocate new pinned memory
            return new PinnedMemoryHandle(bucket);
        }

        /// <summary>
        /// Returns pinned memory to the pool.
        /// </summary>
        public void ReturnPinnedMemory(PinnedMemoryHandle handle)
        {
            if (_disposed || handle == null) return;

            if (Interlocked.Read(ref _pinnedMemoryBytes) >= _maxPinnedBytes)
            {
                handle.Dispose();
                return;
            }

            var bucket = GetGpuBucket(handle.SizeBytes);
            var pool = _pinnedMemoryPools.GetOrAdd(bucket, _ => new ConcurrentBag<PinnedMemoryHandle>());

            pool.Add(handle);
            Interlocked.Add(ref _pinnedMemoryBytes, handle.SizeBytes);
        }

        #endregion

        #region Async Transfers

        /// <summary>
        /// Asynchronously copies data from CPU to GPU using pinned memory.
        /// </summary>
        public async Task<MemoryBuffer1D<float, Stride1D.Dense>> CopyToGpuAsync(float[] data, AcceleratorStream? stream = null)
        {
            var buffer = RentFloatBuffer(data.Length);
            var pinnedHandle = RentPinnedMemory(data.Length * sizeof(float));

            try
            {
                // Copy to pinned memory
                Marshal.Copy(data, 0, pinnedHandle.Pointer, data.Length);

                // Async copy to GPU
                var usedStream = stream ?? _accelerator.DefaultStream;
                buffer.View.CopyFromCPU(usedStream, data);

                await Task.Run(() => usedStream.Synchronize());

                return buffer;
            }
            finally
            {
                ReturnPinnedMemory(pinnedHandle);
            }
        }

        /// <summary>
        /// Asynchronously copies data from GPU to CPU using pinned memory.
        /// </summary>
        public async Task<float[]> CopyFromGpuAsync(MemoryBuffer1D<float, Stride1D.Dense> buffer, AcceleratorStream? stream = null)
        {
            var data = new float[buffer.Length];
            var pinnedHandle = RentPinnedMemory(buffer.Length * sizeof(float));

            try
            {
                var usedStream = stream ?? _accelerator.DefaultStream;
                buffer.View.CopyToCPU(usedStream, data);

                await Task.Run(() => usedStream.Synchronize());

                return data;
            }
            finally
            {
                ReturnPinnedMemory(pinnedHandle);
            }
        }

        #endregion

        #region Helper Methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long GetGpuBucket(long length)
        {
            if (length <= 0) return 256;

            // Round up to power of 2, minimum 256 elements
            long bucket = 256;
            while (bucket < length && bucket < long.MaxValue / 2)
            {
                bucket *= 2;
            }
            return bucket;
        }

        private void UpdateGpuBytesInUse(long delta)
        {
            var newValue = Interlocked.Add(ref _currentGpuBytesInUse, delta);
            lock (_statsLock)
            {
                if (newValue > _peakGpuBytesInUse)
                    _peakGpuBytesInUse = newValue;
            }
        }

        #endregion

        #region Memory Management

        /// <summary>
        /// Clears all pooled GPU buffers.
        /// </summary>
        public void Clear()
        {
            foreach (var pool in _floatBufferPools.Values)
            {
                while (pool.TryTake(out var buffer))
                    buffer.Dispose();
            }
            foreach (var pool in _doubleBufferPools.Values)
            {
                while (pool.TryTake(out var buffer))
                    buffer.Dispose();
            }
            foreach (var pool in _intBufferPools.Values)
            {
                while (pool.TryTake(out var buffer))
                    buffer.Dispose();
            }
            foreach (var pool in _pinnedMemoryPools.Values)
            {
                while (pool.TryTake(out var handle))
                    handle.Dispose();
            }

            Interlocked.Exchange(ref _currentGpuPoolBytes, 0);
            Interlocked.Exchange(ref _pinnedMemoryBytes, 0);
        }

        /// <summary>
        /// Defragments GPU memory by releasing and reallocating buffers.
        /// </summary>
        public void Defragment()
        {
            _accelerator.Synchronize();
            Clear();
            GC.Collect(2, GCCollectionMode.Aggressive);
            GC.WaitForPendingFinalizers();
        }

        /// <summary>
        /// Gets GPU memory usage info.
        /// </summary>
        public GpuMemoryInfo GetMemoryInfo()
        {
            return new GpuMemoryInfo(
                _accelerator.MemorySize,
                Interlocked.Read(ref _currentGpuBytesInUse),
                Interlocked.Read(ref _peakGpuBytesInUse),
                Interlocked.Read(ref _currentGpuPoolBytes),
                Interlocked.Read(ref _gpuAllocations),
                Interlocked.Read(ref _gpuPoolHits),
                Interlocked.Read(ref _gpuPoolMisses),
                Interlocked.Read(ref _pinnedMemoryBytes)
            );
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Clear();
        }
    }

    /// <summary>
    /// Handle for pinned host memory that can be used for async GPU transfers.
    /// </summary>
    public sealed class PinnedMemoryHandle : IDisposable
    {
        private GCHandle _handle;
        private byte[] _buffer;
        private bool _disposed;

        /// <summary>Public API</summary>
        public IntPtr Pointer { get; private set; }
        /// <summary>Public API</summary>
        public long SizeBytes { get; }

        /// <summary>Public API</summary>
        public PinnedMemoryHandle(long sizeBytes)
        {
            SizeBytes = sizeBytes;
            _buffer = new byte[sizeBytes];
            _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
            Pointer = _handle.AddrOfPinnedObject();
        }

        /// <summary>Public API</summary>
        public Span<T> AsSpan<T>() where T : unmanaged
        {
            return MemoryMarshal.Cast<byte, T>(_buffer.AsSpan());
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_handle.IsAllocated)
                _handle.Free();

            _buffer = null!;
            Pointer = IntPtr.Zero;
        }
    }

    /// <summary>
    /// GPU memory usage information.
    /// </summary>
    public readonly struct GpuMemoryInfo
    {
        /// <summary>Public API</summary>
        public long TotalBytes { get; }
        /// <summary>Public API</summary>
        public long UsedBytes { get; }
        /// <summary>Public API</summary>
        public long PeakBytes { get; }
        /// <summary>Public API</summary>
        public long PooledBytes { get; }
        /// <summary>Public API</summary>
        public long Allocations { get; }
        /// <summary>Public API</summary>
        public long PoolHits { get; }
        /// <summary>Public API</summary>
        public long PoolMisses { get; }
        /// <summary>Public API</summary>
        public long PinnedBytes { get; }

        /// <summary>Public API</summary>
        public double HitRate => Allocations > 0 ? (double)PoolHits / Allocations : 0.0;
        /// <summary>Public API</summary>
        public double UtilizationPercent => TotalBytes > 0 ? 100.0 * UsedBytes / TotalBytes : 0.0;

        /// <summary>Public API</summary>
        public GpuMemoryInfo(long total, long used, long peak, long pooled,
            long allocs, long hits, long misses, long pinned)
        {
            TotalBytes = total;
            UsedBytes = used;
            PeakBytes = peak;
            PooledBytes = pooled;
            Allocations = allocs;
            PoolHits = hits;
            PoolMisses = misses;
            PinnedBytes = pinned;
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"GPU Memory: {UsedBytes / 1024.0 / 1024.0:F1}MB used / {TotalBytes / 1024.0 / 1024.0:F1}MB total " +
                   $"({UtilizationPercent:F1}%), peak={PeakBytes / 1024.0 / 1024.0:F1}MB, pool={PooledBytes / 1024.0 / 1024.0:F1}MB, " +
                   $"hitRate={HitRate:P1}";
        }
    }

    /// <summary>
    /// Multi-GPU memory manager for distributed memory across devices.
    /// </summary>
    public class MultiGpuMemoryManager : IDisposable
    {
        private readonly List<GpuMemoryPool> _devicePools;
        private readonly List<Accelerator> _accelerators;
        private readonly ConcurrentDictionary<int, AcceleratorStream> _transferStreams;
        private int _currentDevice;
        private bool _disposed;

        /// <summary>
        /// Creates a multi-GPU memory manager.
        /// </summary>
        public MultiGpuMemoryManager(IEnumerable<Accelerator> accelerators, int maxPoolPerDeviceMB = 2048)
        {
            _accelerators = accelerators.ToList();
            _devicePools = _accelerators.Select(a => new GpuMemoryPool(a, maxPoolPerDeviceMB)).ToList();
            _transferStreams = new ConcurrentDictionary<int, AcceleratorStream>();
            _currentDevice = 0;
        }

        /// <summary>
        /// Number of available GPU devices.
        /// </summary>
        public int DeviceCount => _accelerators.Count;

        /// <summary>
        /// Gets the memory pool for a specific device.
        /// </summary>
        public GpuMemoryPool GetDevicePool(int deviceId)
        {
            if (deviceId < 0 || deviceId >= _devicePools.Count)
                throw new ArgumentOutOfRangeException(nameof(deviceId));
            return _devicePools[deviceId];
        }

        /// <summary>
        /// Rents a buffer from the device with most available memory.
        /// </summary>
        public (int deviceId, MemoryBuffer1D<float, Stride1D.Dense> buffer) RentFromBestDevice(long length)
        {
            // Find device with lowest utilization
            int bestDevice = 0;
            double lowestUtil = double.MaxValue;

            for (int i = 0; i < _devicePools.Count; i++)
            {
                var info = _devicePools[i].GetMemoryInfo();
                if (info.UtilizationPercent < lowestUtil)
                {
                    lowestUtil = info.UtilizationPercent;
                    bestDevice = i;
                }
            }

            return (bestDevice, _devicePools[bestDevice].RentFloatBuffer(length));
        }

        /// <summary>
        /// Rents a buffer using round-robin device selection.
        /// </summary>
        public (int deviceId, MemoryBuffer1D<float, Stride1D.Dense> buffer) RentRoundRobin(long length)
        {
            var device = Interlocked.Increment(ref _currentDevice) % _devicePools.Count;
            return (device, _devicePools[device].RentFloatBuffer(length));
        }

        /// <summary>
        /// Copies data between GPU devices.
        /// </summary>
        public async Task CopyBetweenDevicesAsync<T>(
            MemoryBuffer1D<T, Stride1D.Dense> source, int sourceDevice,
            MemoryBuffer1D<T, Stride1D.Dense> dest, int destDevice) where T : unmanaged
        {
            if (sourceDevice == destDevice)
            {
                // Same device, direct copy
                var stream = _accelerators[sourceDevice].DefaultStream;
                source.View.CopyTo(stream, dest.View);
                stream.Synchronize();
                return;
            }

            // Different devices - copy through host
            var hostBuffer = new T[source.Length];

            // Copy from source GPU to host
            source.View.CopyToCPU(_accelerators[sourceDevice].DefaultStream, hostBuffer);
            _accelerators[sourceDevice].Synchronize();

            // Copy from host to dest GPU
            dest.View.CopyFromCPU(_accelerators[destDevice].DefaultStream, hostBuffer);
            _accelerators[destDevice].Synchronize();
        }

        /// <summary>
        /// Gets combined memory info across all devices.
        /// </summary>
        public GpuMemoryInfo[] GetAllDeviceInfo()
        {
            return _devicePools.Select(p => p.GetMemoryInfo()).ToArray();
        }

        /// <summary>
        /// Clears all device pools.
        /// </summary>
        public void ClearAll()
        {
            foreach (var pool in _devicePools)
                pool.Clear();
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            foreach (var pool in _devicePools)
                pool.Dispose();

            foreach (var stream in _transferStreams.Values)
                stream.Dispose();
        }
    }

    /// <summary>
    /// Unified memory abstraction that automatically manages CPU/GPU placement.
    /// </summary>
    public class UnifiedMemory<T> : IDisposable where T : unmanaged
    {
        private T[]? _cpuData;
        private MemoryBuffer1D<T, Stride1D.Dense>? _gpuBuffer;
        private readonly Accelerator? _accelerator;
        private MemoryLocation _currentLocation;
        private bool _isDirty;
        private bool _disposed;

        /// <summary>Public API</summary>
        public int Length { get; }
        /// <summary>Public API</summary>
        public MemoryLocation CurrentLocation => _currentLocation;

        /// <summary>Public API</summary>
        public enum MemoryLocation { CPU, GPU, Both }

        /// <summary>Public API</summary>
        public UnifiedMemory(int length, Accelerator? accelerator = null)
        {
            Length = length;
            _accelerator = accelerator;
            _cpuData = new T[length];
            _currentLocation = MemoryLocation.CPU;
            _isDirty = false;
        }

        /// <summary>
        /// Gets CPU data, transferring from GPU if necessary.
        /// </summary>
        public T[] GetCpuData()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UnifiedMemory<T>));

            if (_currentLocation == MemoryLocation.GPU && _isDirty)
            {
                // Transfer from GPU
                _cpuData ??= new T[Length];
                _gpuBuffer!.View.CopyToCPU(_accelerator!.DefaultStream, _cpuData);
                _accelerator.Synchronize();
                _currentLocation = MemoryLocation.Both;
                _isDirty = false;
            }

            return _cpuData!;
        }

        /// <summary>
        /// Gets GPU buffer, transferring from CPU if necessary.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> GetGpuBuffer()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UnifiedMemory<T>));
            if (_accelerator == null) throw new InvalidOperationException("No GPU accelerator configured");

            if (_gpuBuffer == null)
            {
                _gpuBuffer = _accelerator.Allocate1D<T>(Length);
            }

            if (_currentLocation == MemoryLocation.CPU && _isDirty)
            {
                // Transfer to GPU
                _gpuBuffer.View.CopyFromCPU(_accelerator.DefaultStream, _cpuData!);
                _accelerator.Synchronize();
                _currentLocation = MemoryLocation.Both;
                _isDirty = false;
            }

            return _gpuBuffer;
        }

        /// <summary>
        /// Marks CPU data as modified.
        /// </summary>
        public void MarkCpuDirty()
        {
            _currentLocation = MemoryLocation.CPU;
            _isDirty = true;
        }

        /// <summary>
        /// Marks GPU data as modified.
        /// </summary>
        public void MarkGpuDirty()
        {
            _currentLocation = MemoryLocation.GPU;
            _isDirty = true;
        }

        /// <summary>
        /// Prefetches data to specified location.
        /// </summary>
        public void Prefetch(MemoryLocation location)
        {
            if (location == MemoryLocation.GPU)
                _ = GetGpuBuffer();
            else if (location == MemoryLocation.CPU)
                _ = GetCpuData();
        }

        /// <summary>
        /// Releases GPU memory, keeping only CPU copy.
        /// </summary>
        public void EvictFromGpu()
        {
            if (_gpuBuffer != null)
            {
                if (_currentLocation == MemoryLocation.GPU || _currentLocation == MemoryLocation.Both)
                {
                    // Ensure we have CPU copy
                    GetCpuData();
                }

                _gpuBuffer.Dispose();
                _gpuBuffer = null;
                _currentLocation = MemoryLocation.CPU;
            }
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _gpuBuffer?.Dispose();
            _cpuData = null;
        }
    }

    /// <summary>
    /// Memory-efficient tensor storage with automatic CPU/GPU offloading.
    /// </summary>
    public class OffloadableTensorStorage : IDisposable
    {
        private readonly ConcurrentDictionary<string, UnifiedMemory<float>> _tensors;
        private readonly Accelerator? _accelerator;
        private readonly long _gpuMemoryBudget;
        private readonly LinkedList<string> _accessOrder;
        private readonly object _accessLock = new();
        private long _currentGpuUsage;
        private bool _disposed;

        /// <summary>Public API</summary>
        public OffloadableTensorStorage(Accelerator? accelerator = null, long gpuMemoryBudgetMB = 4096)
        {
            _accelerator = accelerator;
            _gpuMemoryBudget = gpuMemoryBudgetMB * 1024 * 1024;
            _tensors = new ConcurrentDictionary<string, UnifiedMemory<float>>();
            _accessOrder = new LinkedList<string>();
        }

        /// <summary>
        /// Stores a tensor with automatic memory management.
        /// </summary>
        public void Store(string key, float[] data)
        {
            var memory = new UnifiedMemory<float>(data.Length, _accelerator);
            Array.Copy(data, memory.GetCpuData(), data.Length);
            memory.MarkCpuDirty();

            _tensors[key] = memory;
            UpdateAccessOrder(key);
        }

        /// <summary>
        /// Retrieves tensor for GPU computation, automatically managing memory.
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense>? GetForGpu(string key)
        {
            if (!_tensors.TryGetValue(key, out var memory))
                return null;

            UpdateAccessOrder(key);
            EnsureGpuBudget(memory.Length * sizeof(float));

            return memory.GetGpuBuffer();
        }

        /// <summary>
        /// Retrieves tensor for CPU computation.
        /// </summary>
        public float[]? GetForCpu(string key)
        {
            if (!_tensors.TryGetValue(key, out var memory))
                return null;

            UpdateAccessOrder(key);
            return memory.GetCpuData();
        }

        private void UpdateAccessOrder(string key)
        {
            lock (_accessLock)
            {
                _accessOrder.Remove(key);
                _accessOrder.AddFirst(key);
            }
        }

        private void EnsureGpuBudget(long requiredBytes)
        {
            while (_currentGpuUsage + requiredBytes > _gpuMemoryBudget)
            {
                // Evict least recently used
                string? toEvict;
                lock (_accessLock)
                {
                    if (_accessOrder.Count == 0) break;
                    toEvict = _accessOrder.Last?.Value;
                    if (toEvict != null)
                        _accessOrder.RemoveLast();
                }

                if (toEvict != null && _tensors.TryGetValue(toEvict, out var memory))
                {
                    var sizeBytes = memory.Length * sizeof(float);
                    memory.EvictFromGpu();
                    Interlocked.Add(ref _currentGpuUsage, -sizeBytes);
                }
            }
        }

        /// <summary>
        /// Prefetches tensors to GPU based on predicted access pattern.
        /// </summary>
        public void PrefetchToGpu(IEnumerable<string> keys)
        {
            foreach (var key in keys)
            {
                if (_tensors.TryGetValue(key, out var memory))
                {
                    EnsureGpuBudget(memory.Length * sizeof(float));
                    memory.Prefetch(UnifiedMemory<float>.MemoryLocation.GPU);
                }
            }
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            foreach (var memory in _tensors.Values)
                memory.Dispose();

            _tensors.Clear();
        }
    }

    #endregion
#endif // HAS_ILGPU

    /// <summary>
    /// Memory pool for efficient tensor memory allocation and reuse.
    /// Reduces allocation overhead and memory fragmentation for large models.
    /// </summary>
    public class MemoryPool : IDisposable
    {
        private static readonly Lazy<MemoryPool> _default = new(() => new MemoryPool());
        /// <summary>Public API</summary>
        public static MemoryPool Default => _default.Value;

        // Size buckets for pooling (powers of 2)
        private readonly ConcurrentDictionary<int, ConcurrentBag<double[]>> _floatPools;
        private readonly ConcurrentDictionary<int, ConcurrentBag<float[]>> _float32Pools;

        // Statistics
        private long _allocations;
        private long _poolHits;
        private long _poolMisses;
        private long _totalBytesAllocated;
        private long _currentBytesInUse;
        private long _peakBytesInUse;

        // Configuration
        private readonly long _maxPoolSizeBytes;
        private readonly int _maxBufferSize;
        private long _currentPoolBytes;

        private bool _disposed;
        private readonly object _statsLock = new();

        /// <summary>
        /// Creates a new memory pool.
        /// </summary>
        /// <param name="maxPoolSizeMB">Maximum pool size in MB (default: 1024)</param>
        /// <param name="maxBufferSizeMB">Maximum individual buffer size in MB (default: 256)</param>
        public MemoryPool(int maxPoolSizeMB = 1024, int maxBufferSizeMB = 256)
        {
            _maxPoolSizeBytes = maxPoolSizeMB * 1024L * 1024L;
            _maxBufferSize = maxBufferSizeMB * 1024 * 1024;
            _floatPools = new ConcurrentDictionary<int, ConcurrentBag<double[]>>();
            _float32Pools = new ConcurrentDictionary<int, ConcurrentBag<float[]>>();
        }

        #region Allocation Methods

        /// <summary>
        /// Rents a double array from the pool or allocates a new one.
        /// </summary>
        public double[] RentDouble(int length)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(MemoryPool));

            var bucket = GetBucket(length);
            var pool = _floatPools.GetOrAdd(bucket, _ => new ConcurrentBag<double[]>());

            Interlocked.Increment(ref _allocations);

            if (pool.TryTake(out var buffer) && buffer.Length >= length)
            {
                Interlocked.Increment(ref _poolHits);
                Interlocked.Add(ref _currentPoolBytes, -buffer.Length * sizeof(double));
                UpdateBytesInUse(buffer.Length * sizeof(double));
                return buffer;
            }

            Interlocked.Increment(ref _poolMisses);
            var newBuffer = new double[bucket];
            Interlocked.Add(ref _totalBytesAllocated, bucket * sizeof(double));
            UpdateBytesInUse(bucket * sizeof(double));
            return newBuffer;
        }

        /// <summary>
        /// Rents a float array from the pool or allocates a new one.
        /// </summary>
        public float[] RentFloat(int length)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(MemoryPool));

            var bucket = GetBucket(length);
            var pool = _float32Pools.GetOrAdd(bucket, _ => new ConcurrentBag<float[]>());

            Interlocked.Increment(ref _allocations);

            if (pool.TryTake(out var buffer) && buffer.Length >= length)
            {
                Interlocked.Increment(ref _poolHits);
                Interlocked.Add(ref _currentPoolBytes, -buffer.Length * sizeof(float));
                UpdateBytesInUse(buffer.Length * sizeof(float));
                return buffer;
            }

            Interlocked.Increment(ref _poolMisses);
            var newBuffer = new float[bucket];
            Interlocked.Add(ref _totalBytesAllocated, bucket * sizeof(float));
            UpdateBytesInUse(bucket * sizeof(float));
            return newBuffer;
        }

        /// <summary>
        /// Returns a double array to the pool for reuse.
        /// </summary>
        public void ReturnDouble(double[] buffer)
        {
            if (_disposed || buffer == null || buffer.Length > _maxBufferSize)
                return;

            // Don't pool if we're at capacity
            if (Interlocked.Read(ref _currentPoolBytes) >= _maxPoolSizeBytes)
                return;

            var bucket = GetBucket(buffer.Length);
            var pool = _floatPools.GetOrAdd(bucket, _ => new ConcurrentBag<double[]>());

            // Clear sensitive data
            Array.Clear(buffer);

            pool.Add(buffer);
            Interlocked.Add(ref _currentPoolBytes, buffer.Length * sizeof(double));
            UpdateBytesInUse(-buffer.Length * sizeof(double));
        }

        /// <summary>
        /// Returns a float array to the pool for reuse.
        /// </summary>
        public void ReturnFloat(float[] buffer)
        {
            if (_disposed || buffer == null || buffer.Length > _maxBufferSize)
                return;

            if (Interlocked.Read(ref _currentPoolBytes) >= _maxPoolSizeBytes)
                return;

            var bucket = GetBucket(buffer.Length);
            var pool = _float32Pools.GetOrAdd(bucket, _ => new ConcurrentBag<float[]>());

            Array.Clear(buffer);

            pool.Add(buffer);
            Interlocked.Add(ref _currentPoolBytes, buffer.Length * sizeof(float));
            UpdateBytesInUse(-buffer.Length * sizeof(float));
        }

        #endregion

        #region Bucket Management

        /// <summary>
        /// Gets the bucket size for a given length (next power of 2).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetBucket(int length)
        {
            if (length <= 0) return 1;

            // Round up to next power of 2
            int bucket = 1;
            while (bucket < length && bucket < int.MaxValue / 2)
            {
                bucket *= 2;
            }

            // Minimum bucket size
            return Math.Max(bucket, 64);
        }

        private void UpdateBytesInUse(long delta)
        {
            var newValue = Interlocked.Add(ref _currentBytesInUse, delta);
            lock (_statsLock)
            {
                if (newValue > _peakBytesInUse)
                    _peakBytesInUse = newValue;
            }
        }

        #endregion

        #region Memory Management

        /// <summary>
        /// Clears all pooled buffers to free memory.
        /// </summary>
        public void Clear()
        {
            foreach (var pool in _floatPools.Values)
            {
                while (pool.TryTake(out _)) { }
            }
            foreach (var pool in _float32Pools.Values)
            {
                while (pool.TryTake(out _)) { }
            }

            Interlocked.Exchange(ref _currentPoolBytes, 0);
        }

        /// <summary>
        /// Trims the pool to reduce memory usage.
        /// </summary>
        /// <param name="targetSizeMB">Target pool size in MB</param>
        public void Trim(int targetSizeMB = 256)
        {
            var targetBytes = targetSizeMB * 1024L * 1024L;

            while (Interlocked.Read(ref _currentPoolBytes) > targetBytes)
            {
                bool removed = false;

                foreach (var pool in _floatPools.Values)
                {
                    if (pool.TryTake(out var buffer))
                    {
                        Interlocked.Add(ref _currentPoolBytes, -buffer.Length * sizeof(double));
                        removed = true;
                        break;
                    }
                }

                if (!removed)
                {
                    foreach (var pool in _float32Pools.Values)
                    {
                        if (pool.TryTake(out var buffer))
                        {
                            Interlocked.Add(ref _currentPoolBytes, -buffer.Length * sizeof(float));
                            removed = true;
                            break;
                        }
                    }
                }

                if (!removed) break;
            }

            // Suggest GC for freed memory
            GC.Collect(2, GCCollectionMode.Optimized);
        }

        #endregion

        #region Statistics

        /// <summary>Total number of allocation requests</summary>
        public long Allocations => Interlocked.Read(ref _allocations);

        /// <summary>Number of allocations satisfied from pool</summary>
        public long PoolHits => Interlocked.Read(ref _poolHits);

        /// <summary>Number of allocations that required new allocation</summary>
        public long PoolMisses => Interlocked.Read(ref _poolMisses);

        /// <summary>Pool hit rate (0.0 to 1.0)</summary>
        public double HitRate => Allocations > 0 ? (double)PoolHits / Allocations : 0.0;

        /// <summary>Total bytes allocated since creation</summary>
        public long TotalBytesAllocated => Interlocked.Read(ref _totalBytesAllocated);

        /// <summary>Current bytes in use</summary>
        public long CurrentBytesInUse => Interlocked.Read(ref _currentBytesInUse);

        /// <summary>Peak bytes in use</summary>
        public long PeakBytesInUse => _peakBytesInUse;

        /// <summary>Current pool size in bytes</summary>
        public long PoolSizeBytes => Interlocked.Read(ref _currentPoolBytes);

        /// <summary>Gets memory pool statistics</summary>
        public MemoryPoolStats GetStats()
        {
            return new MemoryPoolStats(
                Allocations,
                PoolHits,
                PoolMisses,
                HitRate,
                TotalBytesAllocated,
                CurrentBytesInUse,
                PeakBytesInUse,
                PoolSizeBytes
            );
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Clear();
        }
    }

    /// <summary>
    /// Memory pool statistics.
    /// </summary>
    public readonly struct MemoryPoolStats
    {
        /// <summary>Public API</summary>
        public long Allocations { get; }
        /// <summary>Public API</summary>
        public long PoolHits { get; }
        /// <summary>Public API</summary>
        public long PoolMisses { get; }
        /// <summary>Public API</summary>
        public double HitRate { get; }
        /// <summary>Public API</summary>
        public long TotalBytesAllocated { get; }
        /// <summary>Public API</summary>
        public long CurrentBytesInUse { get; }
        /// <summary>Public API</summary>
        public long PeakBytesInUse { get; }
        /// <summary>Public API</summary>
        public long PoolSizeBytes { get; }

        /// <summary>Public API</summary>
        public MemoryPoolStats(long allocations, long poolHits, long poolMisses, double hitRate,
            long totalBytesAllocated, long currentBytesInUse, long peakBytesInUse, long poolSizeBytes)
        {
            Allocations = allocations;
            PoolHits = poolHits;
            PoolMisses = poolMisses;
            HitRate = hitRate;
            TotalBytesAllocated = totalBytesAllocated;
            CurrentBytesInUse = currentBytesInUse;
            PeakBytesInUse = peakBytesInUse;
            PoolSizeBytes = poolSizeBytes;
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"MemoryPoolStats(allocs={Allocations}, hitRate={HitRate:P1}, " +
                   $"current={CurrentBytesInUse / 1024.0 / 1024.0:F2}MB, " +
                   $"peak={PeakBytesInUse / 1024.0 / 1024.0:F2}MB, " +
                   $"pool={PoolSizeBytes / 1024.0 / 1024.0:F2}MB)";
        }
    }

    /// <summary>
    /// Gradient checkpointing for memory-efficient training.
    /// Trades compute for memory by recomputing activations during backward pass.
    /// </summary>
    public class GradientCheckpointing
    {
        private readonly Dictionary<string, Func<Tensor>> _checkpoints;
        private readonly HashSet<string> _activeCheckpoints;
        private bool _enabled;

        /// <summary>Public API</summary>
        public GradientCheckpointing()
        {
            _checkpoints = new Dictionary<string, Func<Tensor>>();
            _activeCheckpoints = new HashSet<string>();
            _enabled = true;
        }

        /// <summary>
        /// Enables or disables gradient checkpointing.
        /// </summary>
        public bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        /// <summary>
        /// Checkpoints an activation for later recomputation.
        /// </summary>
        /// <param name="name">Checkpoint name</param>
        /// <param name="activation">The activation tensor</param>
        /// <param name="recompute">Function to recompute the activation</param>
        /// <returns>The activation (possibly detached if checkpointing is enabled)</returns>
        public Tensor Checkpoint(string name, Tensor activation, Func<Tensor> recompute)
        {
            if (!_enabled)
                return activation;

            // Store recompute function
            _checkpoints[name] = recompute;
            _activeCheckpoints.Add(name);

            // Return detached tensor to save memory
            return activation.Detach();
        }

        /// <summary>
        /// Retrieves a checkpointed activation, recomputing if necessary.
        /// </summary>
        /// <param name="name">Checkpoint name</param>
        /// <returns>The activation tensor</returns>
        public Tensor? Retrieve(string name)
        {
            if (!_enabled || !_checkpoints.TryGetValue(name, out var recompute))
                return null;

            return recompute();
        }

        /// <summary>
        /// Clears a specific checkpoint.
        /// </summary>
        public void Clear(string name)
        {
            _checkpoints.Remove(name);
            _activeCheckpoints.Remove(name);
        }

        /// <summary>
        /// Clears all checkpoints.
        /// </summary>
        public void ClearAll()
        {
            _checkpoints.Clear();
            _activeCheckpoints.Clear();
        }

        /// <summary>
        /// Number of active checkpoints.
        /// </summary>
        public int Count => _activeCheckpoints.Count;
    }

    /// <summary>
    /// Memory-efficient tensor operations for large models.
    /// </summary>
    public static class MemoryEfficientOps
    {
        /// <summary>
        /// Performs matrix multiplication in chunks to reduce peak memory usage.
        /// </summary>
        /// <param name="a">First matrix [M, K]</param>
        /// <param name="b">Second matrix [K, N]</param>
        /// <param name="chunkSize">Maximum chunk size (default: 1024)</param>
        /// <returns>Result matrix [M, N]</returns>
        public static Tensor ChunkedMatMul(Tensor a, Tensor b, int chunkSize = 1024)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2)
                throw new ArgumentException("ChunkedMatMul requires 2D tensors");

            int m = (int)a.Shape[0];
            int k = (int)a.Shape[1];
            int n = (int)b.Shape[1];

            if (k != b.Shape[0])
                throw new ArgumentException("Matrix dimensions don't match for multiplication");

            var result = Tensor.Zeros(m, n);

            // Process in chunks along M dimension
            for (int i = 0; i < m; i += chunkSize)
            {
                int chunkM = Math.Min(chunkSize, m - i);

                // Extract chunk from A (rows i to i+chunkM)
                var aChunk = a.Slice(0, i, i + chunkM);

                // Multiply chunk
                var chunkResult = TensorOps.MatMul(aChunk, b);

                // Copy result
                for (int ci = 0; ci < chunkM; ci++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        result.Data[(i + ci) * n + j] = chunkResult.Data[ci * n + j];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Performs attention computation in chunks to reduce memory for long sequences.
        /// </summary>
        /// <param name="query">Query tensor [batch, seq, dim]</param>
        /// <param name="key">Key tensor [batch, seq, dim]</param>
        /// <param name="value">Value tensor [batch, seq, dim]</param>
        /// <param name="chunkSize">Query chunk size (default: 128)</param>
        /// <returns>Attention output</returns>
        public static Tensor ChunkedAttention(Tensor query, Tensor key, Tensor value, int chunkSize = 128)
        {
            // This implements Flash Attention-like chunked computation
            // to avoid materializing the full attention matrix

            int batchSize = (int)query.Shape[0];
            int seqLen = (int)query.Shape[1];
            int dim = (int)query.Shape[2];

            var result = Tensor.Zeros(batchSize, seqLen, dim);
            float scale = 1.0f / MathF.Sqrt(dim);

            // Process query in chunks
            for (int qStart = 0; qStart < seqLen; qStart += chunkSize)
            {
                int qEnd = Math.Min(qStart + chunkSize, seqLen);
                int qLen = qEnd - qStart;

                // For each query chunk, compute attention over all keys
                for (int batch = 0; batch < batchSize; batch++)
                {
                    // Running max and sum for online softmax
                    var runningMax = new double[qLen];
                    var runningSum = new double[qLen];
                    var output = new double[qLen * dim];
                    Array.Fill(runningMax, double.NegativeInfinity);

                    // Process key/value in chunks
                    for (int kvStart = 0; kvStart < seqLen; kvStart += chunkSize)
                    {
                        int kvEnd = Math.Min(kvStart + chunkSize, seqLen);
                        int kvLen = kvEnd - kvStart;

                        // Compute attention scores for this chunk
                        for (int qi = 0; qi < qLen; qi++)
                        {
                            double prevMax = runningMax[qi];
                            double newMax = prevMax;

                            // Find new max
                            for (int ki = 0; ki < kvLen; ki++)
                            {
                                double score = 0;
                                for (int d = 0; d < dim; d++)
                                {
                                    score += query.Data[batch * seqLen * dim + (qStart + qi) * dim + d] *
                                            key.Data[batch * seqLen * dim + (kvStart + ki) * dim + d];
                                }
                                score *= scale;
                                if (score > newMax) newMax = score;
                            }

                            // Update running statistics with new max
                            if (newMax > prevMax)
                            {
                                double expDiff = Math.Exp(prevMax - newMax);
                                runningSum[qi] *= expDiff;
                                for (int d = 0; d < dim; d++)
                                {
                                    output[qi * dim + d] *= expDiff;
                                }
                                runningMax[qi] = newMax;
                            }

                            // Add contribution from this KV chunk
                            for (int ki = 0; ki < kvLen; ki++)
                            {
                                double score = 0;
                                for (int d = 0; d < dim; d++)
                                {
                                    score += query.Data[batch * seqLen * dim + (qStart + qi) * dim + d] *
                                            key.Data[batch * seqLen * dim + (kvStart + ki) * dim + d];
                                }
                                score *= scale;

                                double weight = Math.Exp(score - runningMax[qi]);
                                runningSum[qi] += weight;

                                for (int d = 0; d < dim; d++)
                                {
                                    output[qi * dim + d] += weight *
                                        value.Data[batch * seqLen * dim + (kvStart + ki) * dim + d];
                                }
                            }
                        }
                    }

                    // Normalize and store result
                    for (int qi = 0; qi < qLen; qi++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            result.Data[batch * seqLen * dim + (qStart + qi) * dim + d] =
                                output[qi * dim + d] / runningSum[qi];
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Performs in-place gradient accumulation to save memory.
        /// </summary>
        public static void AccumulateGradients(Tensor target, Tensor gradient)
        {
            if (target.Data.Length != gradient.Data.Length)
                throw new ArgumentException("Tensor shapes must match for gradient accumulation");

            for (int i = 0; i < target.Data.Length; i++)
            {
                target.Data[i] += gradient.Data[i];
            }
        }

        /// <summary>
        /// Performs gradient scaling for mixed precision training.
        /// </summary>
        public static void ScaleGradients(Tensor gradient, double scale)
        {
            for (int i = 0; i < gradient.Data.Length; i++)
            {
                gradient.Data[i] *= scale;
            }
        }
    }

    /// <summary>
    /// Model parallelism utilities for distributing large models.
    /// </summary>
    public static class ModelParallelism
    {
        /// <summary>
        /// Splits a tensor across multiple devices/partitions.
        /// </summary>
        public static Tensor[] SplitTensor(Tensor tensor, int numSplits, int axis = 0)
        {
            if (axis < 0 || axis >= tensor.Shape.Length)
                throw new ArgumentException("Invalid axis for split");

            long axisSize = tensor.Shape[axis];
            long splitSize = (axisSize + numSplits - 1) / numSplits;

            var results = new Tensor[numSplits];

            for (int i = 0; i < numSplits; i++)
            {
                long start = i * splitSize;
                long end = Math.Min(start + splitSize, axisSize);

                if (start >= axisSize)
                {
                    // Empty split
                    var emptyShape = tensor.Shape.ToArray();
                    emptyShape[axis] = 0;
                    results[i] = Tensor.Zeros(emptyShape);
                }
                else
                {
                    results[i] = tensor.SliceAxis(axis, start, end);
                }
            }

            return results;
        }

        /// <summary>
        /// Concatenates tensors from multiple devices/partitions.
        /// </summary>
        public static Tensor ConcatTensors(Tensor[] tensors, int axis = 0)
        {
            if (tensors.Length == 0)
                throw new ArgumentException("No tensors to concatenate");

            if (tensors.Length == 1)
                return tensors[0].Clone();

            // Calculate output shape
            var outputShape = tensors[0].Shape.ToArray();
            long totalAxisSize = 0;

            foreach (var t in tensors)
            {
                totalAxisSize += t.Shape[axis];
            }
            outputShape[axis] = totalAxisSize;

            var result = Tensor.Zeros(outputShape);
            long offset = 0;

            foreach (var t in tensors)
            {
                // Copy tensor data at offset
                CopyToAxis(t, result, axis, (int)offset);
                offset += t.Shape[axis];
            }

            return result;
        }

        private static void CopyToAxis(Tensor source, Tensor dest, int axis, int offset)
        {
            // Simplified copy - works for common cases
            int[] indices = new int[source.Shape.Length];
            CopyRecursive(source, dest, axis, offset, indices, 0);
        }

        private static void CopyRecursive(Tensor source, Tensor dest, int axis, int offset, int[] indices, int dim)
        {
            if (dim == source.Shape.Length)
            {
                // Copy single element
                int srcIdx = 0, dstIdx = 0;
                int srcMult = 1, dstMult = 1;

                for (int i = source.Shape.Length - 1; i >= 0; i--)
                {
                    srcIdx += indices[i] * srcMult;
                    srcMult *= (int)source.Shape[i];

                    int dstDimIdx = (i == axis) ? indices[i] + offset : indices[i];
                    dstIdx += dstDimIdx * dstMult;
                    dstMult *= (int)dest.Shape[i];
                }

                dest.Data[dstIdx] = source.Data[srcIdx];
                return;
            }

            for (int i = 0; i < source.Shape[dim]; i++)
            {
                indices[dim] = i;
                CopyRecursive(source, dest, axis, offset, indices, dim + 1);
            }
        }
    }

    #region Mixed Precision Training

    /// <summary>
    /// Mixed precision training manager for FP16/FP32 hybrid training.
    /// Provides automatic loss scaling to prevent underflow in FP16 gradients.
    /// </summary>
    public class MixedPrecisionManager
    {
        private double _lossScale;
        private readonly double _initialScale;
        private readonly double _scaleGrowthFactor;
        private readonly double _scaleBackoffFactor;
        private readonly int _scaleGrowthInterval;
        private int _stepsSinceGrowth;
        private int _consecutiveSuccesses;
        private readonly int _maxScale;
        private readonly int _minScale;

        // Statistics
        private long _totalSteps;
        private long _overflowCount;
        private long _underflowCount;
        private long _scaleUpdates;

        /// <summary>Public API</summary>
        public double CurrentLossScale => _lossScale;
        /// <summary>Public API</summary>
        public long TotalSteps => _totalSteps;
        /// <summary>Public API</summary>
        public long OverflowCount => _overflowCount;
        /// <summary>Public API</summary>
        public double OverflowRate => _totalSteps > 0 ? (double)_overflowCount / _totalSteps : 0;

        /// <summary>
        /// Creates a mixed precision manager.
        /// </summary>
        /// <param name="initialScale">Initial loss scale (default: 65536)</param>
        /// <param name="growthFactor">Scale growth factor (default: 2.0)</param>
        /// <param name="backoffFactor">Scale reduction on overflow (default: 0.5)</param>
        /// <param name="growthInterval">Steps between scale increases (default: 2000)</param>
        public MixedPrecisionManager(
            double initialScale = 65536.0,
            double growthFactor = 2.0,
            double backoffFactor = 0.5,
            int growthInterval = 2000,
            int maxScale = 65536 * 1024,
            int minScale = 1)
        {
            _lossScale = initialScale;
            _initialScale = initialScale;
            _scaleGrowthFactor = growthFactor;
            _scaleBackoffFactor = backoffFactor;
            _scaleGrowthInterval = growthInterval;
            _maxScale = maxScale;
            _minScale = minScale;
            _stepsSinceGrowth = 0;
            _consecutiveSuccesses = 0;
        }

        /// <summary>
        /// Scales loss for backward pass.
        /// </summary>
        public double ScaleLoss(double loss)
        {
            return loss * _lossScale;
        }

        /// <summary>
        /// Unscales gradients after backward pass.
        /// </summary>
        public void UnscaleGradients(Tensor[] gradients)
        {
            double invScale = 1.0 / _lossScale;
            foreach (var grad in gradients)
            {
                if (grad?.Data == null) continue;
                for (int i = 0; i < grad.Data.Length; i++)
                {
                    grad.Data[i] *= invScale;
                }
            }
        }

        /// <summary>
        /// Checks gradients for overflow/underflow and updates scale accordingly.
        /// Returns true if gradients are valid (no overflow).
        /// </summary>
        public bool CheckAndUpdateScale(Tensor[] gradients)
        {
            _totalSteps++;

            bool hasOverflow = false;
            bool hasUnderflow = false;

            foreach (var grad in gradients)
            {
                if (grad?.Data == null) continue;

                foreach (var val in grad.Data)
                {
                    if (double.IsNaN(val) || double.IsInfinity(val))
                    {
                        hasOverflow = true;
                        break;
                    }
                    if (val != 0 && Math.Abs(val) < 1e-45)
                    {
                        hasUnderflow = true;
                    }
                }
                if (hasOverflow) break;
            }

            if (hasOverflow)
            {
                // Reduce scale on overflow
                _overflowCount++;
                _lossScale = Math.Max(_minScale, _lossScale * _scaleBackoffFactor);
                _consecutiveSuccesses = 0;
                _stepsSinceGrowth = 0;
                _scaleUpdates++;
                return false;
            }

            if (hasUnderflow)
            {
                _underflowCount++;
            }

            // Consider growing scale
            _consecutiveSuccesses++;
            _stepsSinceGrowth++;

            if (_stepsSinceGrowth >= _scaleGrowthInterval)
            {
                _lossScale = Math.Min(_maxScale, _lossScale * _scaleGrowthFactor);
                _stepsSinceGrowth = 0;
                _scaleUpdates++;
            }

            return true;
        }

        /// <summary>
        /// Resets the manager to initial state.
        /// </summary>
        public void Reset()
        {
            _lossScale = _initialScale;
            _stepsSinceGrowth = 0;
            _consecutiveSuccesses = 0;
            _totalSteps = 0;
            _overflowCount = 0;
            _underflowCount = 0;
            _scaleUpdates = 0;
        }

        /// <summary>
        /// Gets state dictionary for checkpointing.
        /// </summary>
        public Dictionary<string, double> GetStateDict()
        {
            return new Dictionary<string, double>
            {
                ["loss_scale"] = _lossScale,
                ["steps_since_growth"] = _stepsSinceGrowth,
                ["consecutive_successes"] = _consecutiveSuccesses
            };
        }

        /// <summary>
        /// Loads state dictionary from checkpoint.
        /// </summary>
        public void LoadStateDict(Dictionary<string, double> stateDict)
        {
            if (stateDict.TryGetValue("loss_scale", out var scale))
                _lossScale = scale;
            if (stateDict.TryGetValue("steps_since_growth", out var steps))
                _stepsSinceGrowth = (int)steps;
            if (stateDict.TryGetValue("consecutive_successes", out var successes))
                _consecutiveSuccesses = (int)successes;
        }
    }

    /// <summary>
    /// FP16 tensor storage with automatic conversion.
    /// Uses Half precision to reduce memory by 50%.
    /// </summary>
    public class HalfPrecisionStorage
    {
        private readonly ConcurrentDictionary<string, Half[]> _fp16Data;
        private readonly ConcurrentDictionary<string, int[]> _shapes;

        /// <summary>Public API</summary>
        public HalfPrecisionStorage()
        {
            _fp16Data = new ConcurrentDictionary<string, Half[]>();
            _shapes = new ConcurrentDictionary<string, int[]>();
        }

        /// <summary>
        /// Stores a tensor in FP16 format.
        /// </summary>
        public void Store(string key, Tensor tensor)
        {
            var fp16 = new Half[tensor.Data.Length];
            for (int i = 0; i < tensor.Data.Length; i++)
            {
                fp16[i] = (Half)tensor.Data[i];
            }
            _fp16Data[key] = fp16;
            _shapes[key] = (int[])tensor.Shape.Clone();
        }

        /// <summary>
        /// Retrieves tensor, converting from FP16 to FP32.
        /// </summary>
        public Tensor? Retrieve(string key)
        {
            if (!_fp16Data.TryGetValue(key, out var fp16) ||
                !_shapes.TryGetValue(key, out var shape))
                return null;

            var data = new double[fp16.Length];
            for (int i = 0; i < fp16.Length; i++)
            {
                data[i] = (float)fp16[i];
            }

            return new Tensor(data, shape.Select(x => (long)x).ToArray());
        }

        /// <summary>
        /// Gets memory usage in bytes.
        /// </summary>
        public long MemoryUsageBytes
        {
            get
            {
                long total = 0;
                foreach (var arr in _fp16Data.Values)
                {
                    total += arr.Length * 2; // 2 bytes per Half
                }
                return total;
            }
        }

        /// <summary>
        /// Clears all stored tensors.
        /// </summary>
        public void Clear()
        {
            _fp16Data.Clear();
            _shapes.Clear();
        }
    }

    #endregion

#if HAS_ILGPU
    #region GPU Gradient Checkpointing

    /// <summary>
    /// GPU-aware gradient checkpointing that manages GPU memory for large models.
    /// Implements activation checkpointing with automatic GPU offloading.
    /// </summary>
    public class GpuGradientCheckpointing : IDisposable
    {
        private readonly Dictionary<string, CheckpointEntry> _checkpoints;
        private readonly Accelerator? _accelerator;
        private readonly GpuMemoryPool? _memoryPool;
        private readonly long _gpuMemoryBudget;
        private long _currentGpuMemory;
        private bool _enabled;
        private bool _disposed;

        private class CheckpointEntry
        {
            /// <summary>Public API</summary>
            public Func<Tensor> RecomputeFunc { get; set; } = null!;
            /// <summary>Public API</summary>
            public MemoryBuffer1D<float, Stride1D.Dense>? GpuBuffer { get; set; }
            /// <summary>Public API</summary>
            public float[]? CpuBackup { get; set; }
            /// <summary>Public API</summary>
            public int[] Shape { get; set; } = null!;
            /// <summary>Public API</summary>
            public bool IsOnGpu { get; set; }
            /// <summary>Public API</summary>
            public long LastAccessTime { get; set; }
        }

        /// <summary>Public API</summary>
        public GpuGradientCheckpointing(
            Accelerator? accelerator = null,
            GpuMemoryPool? memoryPool = null,
            long gpuMemoryBudgetMB = 2048)
        {
            _accelerator = accelerator;
            _memoryPool = memoryPool;
            _gpuMemoryBudget = gpuMemoryBudgetMB * 1024 * 1024;
            _checkpoints = new Dictionary<string, CheckpointEntry>();
            _enabled = true;
        }

        /// <summary>Public API</summary>
        public bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        /// <summary>
        /// Checkpoints an activation with GPU memory management.
        /// </summary>
        public Tensor Checkpoint(string name, Tensor activation, Func<Tensor> recompute)
        {
            if (!_enabled)
                return activation;

            var entry = new CheckpointEntry
            {
                RecomputeFunc = recompute,
                Shape = (int[])activation.Shape.Clone(),
                LastAccessTime = Environment.TickCount64
            };

            var sizeBytes = activation.Data.Length * sizeof(float);

            // Try to keep on GPU if we have budget
            if (_accelerator != null && _currentGpuMemory + sizeBytes <= _gpuMemoryBudget)
            {
                var floatData = activation.Data.Select(d => (float)d).ToArray();

                if (_memoryPool != null)
                {
                    entry.GpuBuffer = _memoryPool.RentFloatBuffer(floatData.Length);
                }
                else
                {
                    entry.GpuBuffer = _accelerator.Allocate1D<float>(floatData.Length);
                }

                entry.GpuBuffer.View.CopyFromCPU(_accelerator.DefaultStream, floatData);
                entry.IsOnGpu = true;
                _currentGpuMemory += sizeBytes;
            }
            else
            {
                // Offload to CPU
                entry.CpuBackup = activation.Data.Select(d => (float)d).ToArray();
                entry.IsOnGpu = false;
            }

            _checkpoints[name] = entry;
            return activation.Detach();
        }

        /// <summary>
        /// Retrieves a checkpointed activation.
        /// </summary>
        public Tensor? Retrieve(string name, bool preferRecompute = false)
        {
            if (!_enabled || !_checkpoints.TryGetValue(name, out var entry))
                return null;

            entry.LastAccessTime = Environment.TickCount64;

            if (preferRecompute)
            {
                return entry.RecomputeFunc();
            }

            if (entry.IsOnGpu && entry.GpuBuffer != null && _accelerator != null)
            {
                // Retrieve from GPU
                var floatData = new float[entry.GpuBuffer.Length];
                entry.GpuBuffer.View.CopyToCPU(_accelerator.DefaultStream, floatData);
                _accelerator.Synchronize();

                var doubleData = floatData.Select(f => (double)f).ToArray();
                return Tensor.FromArray(doubleData, entry.Shape);
            }
            else if (entry.CpuBackup != null)
            {
                // Retrieve from CPU backup
                var doubleData = entry.CpuBackup.Select(f => (double)f).ToArray();
                return Tensor.FromArray(doubleData, entry.Shape);
            }

            // Fall back to recomputation
            return entry.RecomputeFunc();
        }

        /// <summary>
        /// Evicts least recently used checkpoints from GPU to CPU.
        /// </summary>
        public void EvictLRU(long bytesToFree)
        {
            if (_accelerator == null) return;

            var sorted = _checkpoints
                .Where(kv => kv.Value.IsOnGpu)
                .OrderBy(kv => kv.Value.LastAccessTime)
                .ToList();

            long freedBytes = 0;
            foreach (var kv in sorted)
            {
                if (freedBytes >= bytesToFree) break;

                var entry = kv.Value;
                if (entry.GpuBuffer != null)
                {
                    // Copy to CPU before freeing
                    entry.CpuBackup = new float[entry.GpuBuffer.Length];
                    entry.GpuBuffer.View.CopyToCPU(_accelerator.DefaultStream, entry.CpuBackup);
                    _accelerator.Synchronize();

                    var sizeBytes = entry.GpuBuffer.Length * sizeof(float);

                    if (_memoryPool != null)
                    {
                        _memoryPool.ReturnFloatBuffer(entry.GpuBuffer);
                    }
                    else
                    {
                        entry.GpuBuffer.Dispose();
                    }

                    entry.GpuBuffer = null;
                    entry.IsOnGpu = false;
                    _currentGpuMemory -= sizeBytes;
                    freedBytes += sizeBytes;
                }
            }
        }

        /// <summary>
        /// Clears a specific checkpoint.
        /// </summary>
        public void Clear(string name)
        {
            if (_checkpoints.TryGetValue(name, out var entry))
            {
                if (entry.GpuBuffer != null)
                {
                    var sizeBytes = entry.GpuBuffer.Length * sizeof(float);
                    if (_memoryPool != null)
                    {
                        _memoryPool.ReturnFloatBuffer(entry.GpuBuffer);
                    }
                    else
                    {
                        entry.GpuBuffer.Dispose();
                    }
                    _currentGpuMemory -= sizeBytes;
                }
                _checkpoints.Remove(name);
            }
        }

        /// <summary>
        /// Clears all checkpoints.
        /// </summary>
        public void ClearAll()
        {
            foreach (var entry in _checkpoints.Values)
            {
                if (entry.GpuBuffer != null)
                {
                    if (_memoryPool != null)
                    {
                        _memoryPool.ReturnFloatBuffer(entry.GpuBuffer);
                    }
                    else
                    {
                        entry.GpuBuffer.Dispose();
                    }
                }
            }
            _checkpoints.Clear();
            _currentGpuMemory = 0;
        }

        /// <summary>Public API</summary>
        public int Count => _checkpoints.Count;
        /// <summary>Public API</summary>
        public long GpuMemoryUsed => _currentGpuMemory;

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            ClearAll();
        }
    }

    #endregion
#endif // HAS_ILGPU

    #region Memory Profiler

    /// <summary>
    /// Memory profiler for tracking tensor allocations and identifying leaks.
    /// </summary>
    public class MemoryProfiler
    {
        private readonly ConcurrentDictionary<long, AllocationRecord> _allocations;
        private readonly ConcurrentDictionary<string, long> _allocationsBySource;
        private long _nextId;
        private long _totalAllocated;
        private long _totalFreed;
        private long _peakMemory;
        private long _currentMemory;
        private bool _enabled;
        private readonly object _lock = new();

        private class AllocationRecord
        {
            /// <summary>Public API</summary>
            public long Id { get; set; }
            /// <summary>Public API</summary>
            public long SizeBytes { get; set; }
            /// <summary>Public API</summary>
            public string Source { get; set; } = "";
            /// <summary>Public API</summary>
            public DateTime Timestamp { get; set; }
            /// <summary>Public API</summary>
            public string? StackTrace { get; set; }
        }

        /// <summary>Public API</summary>
        public MemoryProfiler(bool captureStackTraces = false)
        {
            _allocations = new ConcurrentDictionary<long, AllocationRecord>();
            _allocationsBySource = new ConcurrentDictionary<string, long>();
            CaptureStackTraces = captureStackTraces;
            _enabled = true;
        }

        /// <summary>Public API</summary>
        public bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        /// <summary>Public API</summary>
        public bool CaptureStackTraces { get; set; }

        /// <summary>
        /// Records a tensor allocation.
        /// </summary>
        public long RecordAllocation(long sizeBytes, string source = "unknown")
        {
            if (!_enabled) return -1;

            var id = Interlocked.Increment(ref _nextId);
            var record = new AllocationRecord
            {
                Id = id,
                SizeBytes = sizeBytes,
                Source = source,
                Timestamp = DateTime.UtcNow,
                StackTrace = CaptureStackTraces ? Environment.StackTrace : null
            };

            _allocations[id] = record;
            _allocationsBySource.AddOrUpdate(source, sizeBytes, (_, old) => old + sizeBytes);

            Interlocked.Add(ref _totalAllocated, sizeBytes);
            var newCurrent = Interlocked.Add(ref _currentMemory, sizeBytes);

            lock (_lock)
            {
                if (newCurrent > _peakMemory)
                    _peakMemory = newCurrent;
            }

            return id;
        }

        /// <summary>
        /// Records a tensor deallocation.
        /// </summary>
        public void RecordDeallocation(long id)
        {
            if (!_enabled || id < 0) return;

            if (_allocations.TryRemove(id, out var record))
            {
                Interlocked.Add(ref _totalFreed, record.SizeBytes);
                Interlocked.Add(ref _currentMemory, -record.SizeBytes);

                _allocationsBySource.AddOrUpdate(
                    record.Source,
                    0,
                    (_, old) => Math.Max(0, old - record.SizeBytes));
            }
        }

        /// <summary>
        /// Gets memory statistics.
        /// </summary>
        public MemoryProfileStats GetStats()
        {
            return new MemoryProfileStats
            {
                TotalAllocated = Interlocked.Read(ref _totalAllocated),
                TotalFreed = Interlocked.Read(ref _totalFreed),
                CurrentMemory = Interlocked.Read(ref _currentMemory),
                PeakMemory = _peakMemory,
                ActiveAllocations = _allocations.Count,
                AllocationsBySource = _allocationsBySource.ToDictionary(kv => kv.Key, kv => kv.Value)
            };
        }

        /// <summary>
        /// Gets potentially leaked allocations (older than specified age).
        /// </summary>
        public IEnumerable<(long Id, long SizeBytes, string Source, TimeSpan Age)> GetPotentialLeaks(TimeSpan minAge)
        {
            var cutoff = DateTime.UtcNow - minAge;
            return _allocations
                .Where(kv => kv.Value.Timestamp < cutoff)
                .Select(kv => (
                    kv.Key,
                    kv.Value.SizeBytes,
                    kv.Value.Source,
                    DateTime.UtcNow - kv.Value.Timestamp
                ))
                .OrderByDescending(x => x.SizeBytes);
        }

        /// <summary>
        /// Resets all profiling data.
        /// </summary>
        public void Reset()
        {
            _allocations.Clear();
            _allocationsBySource.Clear();
            _totalAllocated = 0;
            _totalFreed = 0;
            _peakMemory = 0;
            _currentMemory = 0;
        }
    }

    /// <summary>
    /// Memory profile statistics.
    /// </summary>
    public class MemoryProfileStats
    {
        /// <summary>Public API</summary>
        public long TotalAllocated { get; set; }
        /// <summary>Public API</summary>
        public long TotalFreed { get; set; }
        /// <summary>Public API</summary>
        public long CurrentMemory { get; set; }
        /// <summary>Public API</summary>
        public long PeakMemory { get; set; }
        /// <summary>Public API</summary>
        public int ActiveAllocations { get; set; }
        /// <summary>Public API</summary>
        public Dictionary<string, long> AllocationsBySource { get; set; } = new();

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"Memory Profile: current={CurrentMemory / 1024.0 / 1024.0:F2}MB, " +
                   $"peak={PeakMemory / 1024.0 / 1024.0:F2}MB, " +
                   $"allocated={TotalAllocated / 1024.0 / 1024.0:F2}MB, " +
                   $"freed={TotalFreed / 1024.0 / 1024.0:F2}MB, " +
                   $"active={ActiveAllocations}";
        }
    }

    #endregion

    #region Stream-Based Processing

    /// <summary>
    /// Stream processor for handling data that doesn't fit in memory.
    /// Processes tensors in chunks using streaming I/O.
    /// </summary>
    public class TensorStreamProcessor
    {
        private readonly int _chunkSize;
        private readonly MemoryPool _memoryPool;

        /// <summary>Public API</summary>
        public TensorStreamProcessor(int chunkSizeMB = 64, MemoryPool? memoryPool = null)
        {
            _chunkSize = chunkSizeMB * 1024 * 1024 / sizeof(float);
            _memoryPool = memoryPool ?? MemoryPool.Default;
        }

        /// <summary>
        /// Processes a large tensor file in chunks.
        /// </summary>
        public async IAsyncEnumerable<Tensor> StreamFromFileAsync(
            string filePath,
            int[] chunkShape,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true);
            using var reader = new BinaryReader(stream);

            var chunkElements = chunkShape.Aggregate(1, (a, b) => a * b);
            var buffer = _memoryPool.RentFloat(chunkElements);

            try
            {
                while (stream.Position < stream.Length && !cancellationToken.IsCancellationRequested)
                {
                    var bytesToRead = Math.Min((stream.Length - stream.Position), chunkElements * sizeof(float));
                    var elementsToRead = (int)(bytesToRead / sizeof(float));

                    for (int i = 0; i < elementsToRead; i++)
                    {
                        buffer[i] = reader.ReadSingle();
                    }

                    var data = new double[elementsToRead];
                    for (int i = 0; i < elementsToRead; i++)
                    {
                        data[i] = buffer[i];
                    }

                    // Adjust shape if we read fewer elements
                    var actualShape = (int[])chunkShape.Clone();
                    if (elementsToRead < chunkElements)
                    {
                        actualShape[0] = elementsToRead / (chunkElements / chunkShape[0]);
                    }

                    yield return new Tensor(data, actualShape.Select(x => (long)x).ToArray());

                    await Task.Yield(); // Allow other async operations
                }
            }
            finally
            {
                _memoryPool.ReturnFloat(buffer);
            }
        }

        /// <summary>
        /// Writes tensors to file in streaming fashion.
        /// </summary>
        public async Task StreamToFileAsync(
            string filePath,
            IAsyncEnumerable<Tensor> tensors,
            CancellationToken cancellationToken = default)
        {
            using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 65536, true);
            using var writer = new BinaryWriter(stream);

            await foreach (var tensor in tensors.WithCancellation(cancellationToken))
            {
                foreach (var val in tensor.Data)
                {
                    writer.Write((float)val);
                }
            }

            await stream.FlushAsync(cancellationToken);
        }

        /// <summary>
        /// Applies a function to each chunk of a tensor stream.
        /// </summary>
        public async IAsyncEnumerable<Tensor> MapAsync(
            IAsyncEnumerable<Tensor> source,
            Func<Tensor, Tensor> transform,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var tensor in source.WithCancellation(cancellationToken))
            {
                yield return transform(tensor);
            }
        }

        /// <summary>
        /// Reduces a tensor stream to a single value.
        /// </summary>
        public async Task<Tensor> ReduceAsync(
            IAsyncEnumerable<Tensor> source,
            Func<Tensor, Tensor, Tensor> reducer,
            Tensor? initial = null,
            CancellationToken cancellationToken = default)
        {
            Tensor? accumulator = initial;

            await foreach (var tensor in source.WithCancellation(cancellationToken))
            {
                if (accumulator == null)
                {
                    accumulator = tensor;
                }
                else
                {
                    accumulator = reducer(accumulator, tensor);
                }
            }

            return accumulator ?? Tensor.Zeros(1);
        }
    }

    #endregion
}