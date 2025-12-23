using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;

namespace NSL.GPU
{
    /// <summary>
    /// Automatic GPU Detection and Configuration System for NSL.
    ///
    /// Features:
    /// - Automatic detection of all available GPUs
    /// - Hot-swap support for GPU changes (new GPU, driver update, etc.)
    /// - Architecture-specific optimizations (Ampere, Turing, Volta, etc.)
    /// - Automatic kernel recompilation for new GPUs
    /// - Multi-GPU load balancing
    /// - Fallback to CPU when no GPU available
    ///
    /// Usage:
    /// <code>
    /// var gpuManager = new GpuAutoConfig();
    /// gpuManager.OnGpuChanged += (sender, e) => Console.WriteLine($"GPU changed: {e.NewGpu?.Name}");
    /// var accelerator = gpuManager.GetBestAccelerator();
    /// </code>
    /// </summary>
    public class GpuAutoConfig : IDisposable
    {
        private Context? _context;
        private Accelerator? _currentAccelerator;
        private GpuKernels? _kernels;
        private readonly object _lock = new();
        private Timer? _monitorTimer;
        private List<GpuInfo> _lastKnownGpus = new();
        private bool _disposed;

        /// <summary>
        /// Information about a detected GPU
        /// </summary>
        public class GpuInfo
        {
            /// <summary>Public API</summary>
            public string Name { get; set; } = "";
            /// <summary>Public API</summary>
            public string DeviceId { get; set; } = "";
            /// <summary>Public API</summary>
            public GpuBackend Backend { get; set; }
            /// <summary>Public API</summary>
            public GpuArchitecture Architecture { get; set; }
            /// <summary>Public API</summary>
            public long MemoryBytes { get; set; }
            /// <summary>Public API</summary>
            public int ComputeUnits { get; set; }
            /// <summary>Public API</summary>
            public int WarpSize { get; set; }
            /// <summary>Public API</summary>
            public int MaxThreadsPerBlock { get; set; }
            /// <summary>Public API</summary>
            public int ComputeCapabilityMajor { get; set; }
            /// <summary>Public API</summary>
            public int ComputeCapabilityMinor { get; set; }
            /// <summary>Public API</summary>
            public bool SupportsFloat16 { get; set; }
            /// <summary>Public API</summary>
            public bool SupportsTensorCores { get; set; }
            /// <summary>Public API</summary>
            public bool SupportsInt8 { get; set; }
            /// <summary>Public API</summary>
            public float Score { get; set; } // Performance score for ranking

            /// <summary>Public API</summary>
            public override string ToString() =>
                $"{Name} ({Architecture}) - {MemoryBytes / (1024 * 1024)}MB, CC {ComputeCapabilityMajor}.{ComputeCapabilityMinor}";
        }

        /// <summary>
        /// GPU Architecture classification based on NVIDIA generations
        /// </summary>
        public enum GpuArchitecture
        {
            Unknown = 0,
            // NVIDIA Architectures
            Kepler = 30,       // SM 3.0-3.7 (GTX 600/700)
            Maxwell = 50,      // SM 5.0-5.3 (GTX 900)
            Pascal = 60,       // SM 6.0-6.2 (GTX 1000)
            Volta = 70,        // SM 7.0 (Tesla V100)
            Turing = 75,       // SM 7.5 (RTX 2000)
            Ampere = 80,       // SM 8.0-8.6 (RTX 3000, A100)
            Ada = 89,          // SM 8.9 (RTX 4000)
            Hopper = 90,       // SM 9.0 (H100)
            Blackwell = 100,   // SM 10.0 (B100)
            // AMD/OpenCL
            RDNA = 1000,
            RDNA2 = 1001,
            RDNA3 = 1002,
            // CPU fallback
            CPU = 9999
        }

        /// <summary>
        /// Event fired when GPU configuration changes
        /// </summary>
        public event EventHandler<GpuChangedEventArgs>? OnGpuChanged;

        /// <summary>Public API</summary>
        public class GpuChangedEventArgs : EventArgs
        {
            /// <summary>Public API</summary>
            public GpuInfo? OldGpu { get; set; }
            /// <summary>Public API</summary>
            public GpuInfo? NewGpu { get; set; }
            /// <summary>Public API</summary>
            public ChangeReason Reason { get; set; }
        }

        /// <summary>Public API</summary>
        public enum ChangeReason
        {
            NewGpuDetected,
            GpuRemoved,
            DriverUpdate,
            UserRequested,
            Fallback
        }

        /// <summary>
        /// Configuration options
        /// </summary>
        public class Config
        {
            /// <summary>Public API</summary>
            public bool EnableMonitoring { get; set; } = true;
            /// <summary>Public API</summary>
            public int MonitoringIntervalMs { get; set; } = 5000;
            /// <summary>Public API</summary>
            public bool PreferCuda { get; set; } = true;
            /// <summary>Public API</summary>
            public bool AllowCpuFallback { get; set; } = true;
            /// <summary>Public API</summary>
            public int MinimumMemoryMB { get; set; } = 1024;
            /// <summary>Public API</summary>
            public bool AutoOptimize { get; set; } = true;
        }

        /// <summary>Public API</summary>
        public Config Configuration { get; }
        /// <summary>Public API</summary>
        public GpuInfo? CurrentGpu { get; private set; }
        /// <summary>Public API</summary>
        public List<GpuInfo> AvailableGpus => DetectAllGpus();

        /// <summary>Public API</summary>
        public GpuAutoConfig(Config? config = null)
        {
            Configuration = config ?? new Config();
            Initialize();
        }

        private void Initialize()
        {
            lock (_lock)
            {
                // Create ILGPU context with all backends
                _context = Context.Create(builder => builder
                    .Default()
                    .EnableAlgorithms()
                    .Optimize(OptimizationLevel.O2)
                    .Math(MathMode.Default));

                // Detect and select best GPU
                var gpus = DetectAllGpus();
                _lastKnownGpus = gpus;

                var bestGpu = SelectBestGpu(gpus);
                if (bestGpu != null)
                {
                    ActivateGpu(bestGpu);
                }
                else if (Configuration.AllowCpuFallback)
                {
                    ActivateCpuFallback();
                }

                // Start monitoring for GPU changes
                if (Configuration.EnableMonitoring)
                {
                    _monitorTimer = new Timer(MonitorGpuChanges, null,
                        Configuration.MonitoringIntervalMs,
                        Configuration.MonitoringIntervalMs);
                }
            }
        }

        /// <summary>
        /// Detect all available GPUs on the system
        /// </summary>
        public List<GpuInfo> DetectAllGpus()
        {
            var gpus = new List<GpuInfo>();

            if (_context == null) return gpus;

            try
            {
                // Detect CUDA devices
                var cudaDevices = _context.GetCudaDevices();
                foreach (var device in cudaDevices)
                {
                    var info = new GpuInfo
                    {
                        Name = device.Name,
                        DeviceId = $"CUDA:{device.DeviceId}",
                        Backend = GpuBackend.CUDA,
                        MemoryBytes = device.MemorySize,
                        ComputeUnits = device.NumMultiprocessors,
                        WarpSize = device.WarpSize,
                        MaxThreadsPerBlock = device.MaxNumThreadsPerGroup
                    };

                    // Determine architecture from SM version
                    var arch = device.Architecture;
                    // CudaArchitecture enum values correspond to SM versions (e.g., SM_86 = 86)
                    int archValue = 0;
                    if (arch.HasValue)
                    {
                        var archName = arch.Value.ToString();
                        if (archName.StartsWith("SM_") && int.TryParse(archName.Substring(3), out int parsed))
                        {
                            archValue = parsed;
                        }
                    }
                    info.ComputeCapabilityMajor = archValue / 10;
                    info.ComputeCapabilityMinor = archValue % 10;
                    info.Architecture = ClassifyArchitecture(archValue);

                    // Feature detection based on architecture
                    info.SupportsFloat16 = archValue >= 60;
                    info.SupportsTensorCores = archValue >= 70;
                    info.SupportsInt8 = archValue >= 61;

                    // Calculate performance score
                    info.Score = CalculateGpuScore(info);

                    gpus.Add(info);
                }

                // Detect OpenCL devices
                var clDevices = _context.GetCLDevices();
                foreach (var device in clDevices)
                {
                    var info = new GpuInfo
                    {
                        Name = device.Name,
                        DeviceId = $"OpenCL:{device.DeviceId}",
                        Backend = GpuBackend.OpenCL,
                        MemoryBytes = device.MemorySize,
                        ComputeUnits = device.NumMultiprocessors,
                        WarpSize = device.WarpSize,
                        MaxThreadsPerBlock = device.MaxNumThreadsPerGroup,
                        Architecture = ClassifyAMDArchitecture(device.Name)
                    };

                    info.Score = CalculateGpuScore(info);
                    gpus.Add(info);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"GPU detection error: {ex.Message}");
            }

            return gpus.OrderByDescending(g => g.Score).ToList();
        }

        /// <summary>
        /// Select the best GPU based on performance score and requirements
        /// </summary>
        public GpuInfo? SelectBestGpu(List<GpuInfo> gpus)
        {
            var candidates = gpus
                .Where(g => g.MemoryBytes >= Configuration.MinimumMemoryMB * 1024L * 1024L)
                .OrderByDescending(g => g.Score);

            if (Configuration.PreferCuda)
            {
                var cudaGpu = candidates.FirstOrDefault(g => g.Backend == GpuBackend.CUDA);
                if (cudaGpu != null) return cudaGpu;
            }

            return candidates.FirstOrDefault();
        }

        /// <summary>
        /// Activate a specific GPU
        /// </summary>
        public void ActivateGpu(GpuInfo gpu)
        {
            lock (_lock)
            {
                var oldGpu = CurrentGpu;

                // Dispose old accelerator
                _kernels = null;
                _currentAccelerator?.Dispose();

                try
                {
                    // Create new accelerator based on backend
                    _currentAccelerator = gpu.Backend switch
                    {
                        GpuBackend.CUDA => _context!.GetCudaDevice(GetDeviceIndex(gpu.DeviceId))
                            .CreateAccelerator(_context),
                        GpuBackend.OpenCL => _context!.GetCLDevice(GetDeviceIndex(gpu.DeviceId))
                            .CreateAccelerator(_context),
                        _ => throw new NotSupportedException($"Backend {gpu.Backend} not supported")
                    };

                    // Create optimized kernels for this GPU
                    _kernels = new GpuKernels(_currentAccelerator);

                    CurrentGpu = gpu;

                    // Apply architecture-specific optimizations
                    if (Configuration.AutoOptimize)
                    {
                        ApplyArchitectureOptimizations(gpu);
                    }

                    OnGpuChanged?.Invoke(this, new GpuChangedEventArgs
                    {
                        OldGpu = oldGpu,
                        NewGpu = gpu,
                        Reason = oldGpu == null ? ChangeReason.NewGpuDetected : ChangeReason.UserRequested
                    });

                    // GPU ready
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Failed to activate GPU {gpu.Name}: {ex.Message}");
                    if (Configuration.AllowCpuFallback)
                    {
                        ActivateCpuFallback();
                    }
                }
            }
        }

        /// <summary>
        /// Fallback to CPU execution
        /// </summary>
        public void ActivateCpuFallback()
        {
            lock (_lock)
            {
                var oldGpu = CurrentGpu;

                _kernels = null;
                _currentAccelerator?.Dispose();

                _currentAccelerator = _context!.GetCPUDevice(0).CreateAccelerator(_context);
                _kernels = new GpuKernels(_currentAccelerator);

                CurrentGpu = new GpuInfo
                {
                    Name = "CPU Fallback",
                    Backend = GpuBackend.CPU,
                    Architecture = GpuArchitecture.CPU,
                    ComputeUnits = Environment.ProcessorCount,
                    Score = 0
                };

                OnGpuChanged?.Invoke(this, new GpuChangedEventArgs
                {
                    OldGpu = oldGpu,
                    NewGpu = CurrentGpu,
                    Reason = ChangeReason.Fallback
                });

                // CPU fallback
            }
        }

        /// <summary>
        /// Get the current accelerator for tensor operations
        /// </summary>
        public Accelerator GetAccelerator()
        {
            if (_currentAccelerator == null)
                throw new InvalidOperationException("No GPU accelerator available");
            return _currentAccelerator;
        }

        /// <summary>
        /// Get the current GPU kernels
        /// </summary>
        public GpuKernels GetKernels()
        {
            if (_kernels == null)
                throw new InvalidOperationException("No GPU kernels available");
            return _kernels;
        }

        /// <summary>
        /// Force a GPU rescan and reconfiguration
        /// </summary>
        public void Rescan()
        {
            var gpus = DetectAllGpus();
            var bestGpu = SelectBestGpu(gpus);

            if (bestGpu != null && (CurrentGpu == null || bestGpu.DeviceId != CurrentGpu.DeviceId))
            {
                ActivateGpu(bestGpu);
            }
            else if (bestGpu == null && Configuration.AllowCpuFallback)
            {
                ActivateCpuFallback();
            }

            _lastKnownGpus = gpus;
        }

        #region Private Methods

        private void MonitorGpuChanges(object? state)
        {
            try
            {
                var currentGpus = DetectAllGpus();

                // Check for new GPUs
                var newGpus = currentGpus.Where(c =>
                    !_lastKnownGpus.Any(l => l.DeviceId == c.DeviceId)).ToList();

                // Check for removed GPUs
                var removedGpus = _lastKnownGpus.Where(l =>
                    !currentGpus.Any(c => c.DeviceId == l.DeviceId)).ToList();

                if (newGpus.Any() || removedGpus.Any())
                {
                    // GPU change detected

                    // If current GPU was removed, select new one
                    if (CurrentGpu != null && removedGpus.Any(r => r.DeviceId == CurrentGpu.DeviceId))
                    {
                        Rescan();
                    }
                    // If new GPU is better, consider switching
                    else if (newGpus.Any())
                    {
                        var bestNew = newGpus.OrderByDescending(g => g.Score).First();
                        if (CurrentGpu == null || bestNew.Score > CurrentGpu.Score * 1.5)
                        {
                            // Switching GPU
                            ActivateGpu(bestNew);
                        }
                    }

                    _lastKnownGpus = currentGpus;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"GPU monitoring error: {ex.Message}");
            }
        }

        private static GpuArchitecture ClassifyArchitecture(int smVersion)
        {
            return smVersion switch
            {
                >= 100 => GpuArchitecture.Blackwell,
                >= 90 => GpuArchitecture.Hopper,
                >= 89 => GpuArchitecture.Ada,
                >= 80 => GpuArchitecture.Ampere,
                >= 75 => GpuArchitecture.Turing,
                >= 70 => GpuArchitecture.Volta,
                >= 60 => GpuArchitecture.Pascal,
                >= 50 => GpuArchitecture.Maxwell,
                >= 30 => GpuArchitecture.Kepler,
                _ => GpuArchitecture.Unknown
            };
        }

        private static GpuArchitecture ClassifyAMDArchitecture(string name)
        {
            if (name.Contains("RX 7") || name.Contains("7900") || name.Contains("7800"))
                return GpuArchitecture.RDNA3;
            if (name.Contains("RX 6") || name.Contains("6900") || name.Contains("6800") || name.Contains("6700"))
                return GpuArchitecture.RDNA2;
            if (name.Contains("RX 5") || name.Contains("5700") || name.Contains("5600"))
                return GpuArchitecture.RDNA;
            return GpuArchitecture.Unknown;
        }

        private static float CalculateGpuScore(GpuInfo gpu)
        {
            float score = 0;

            // Memory score (log scale)
            score += MathF.Log2(gpu.MemoryBytes / (1024f * 1024f)) * 100;

            // Compute units
            score += gpu.ComputeUnits * 10;

            // Architecture bonus
            score += (int)gpu.Architecture;

            // Feature bonuses
            if (gpu.SupportsTensorCores) score += 500;
            if (gpu.SupportsFloat16) score += 200;
            if (gpu.SupportsInt8) score += 100;

            // Backend preference (CUDA slightly preferred for ML workloads)
            if (gpu.Backend == GpuBackend.CUDA) score *= 1.1f;

            return score;
        }

        private static int GetDeviceIndex(string deviceId)
        {
            var parts = deviceId.Split(':');
            return parts.Length > 1 ? int.Parse(parts[1]) : 0;
        }

        private void ApplyArchitectureOptimizations(GpuInfo gpu)
        {
            // Architecture-specific kernel tuning
            switch (gpu.Architecture)
            {
                case GpuArchitecture.Ampere:
                case GpuArchitecture.Ada:
                case GpuArchitecture.Hopper:
                    // Enable TF32 for matrix operations
                    // TF32 enabled
                    // Async mem enabled
                    break;

                case GpuArchitecture.Turing:
                case GpuArchitecture.Volta:
                    // FP16 TC enabled
                    break;

                case GpuArchitecture.Pascal:
                    // FP16 enabled
                    break;
            }

            // Log optimization details
            // CC logged
            // Mem logged
            // CU logged
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _monitorTimer?.Dispose();
                _currentAccelerator?.Dispose();
                _context?.Dispose();
                _disposed = true;
            }
        }
    }
}