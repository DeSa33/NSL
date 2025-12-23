# NSL.GPU API Reference

Complete API documentation for NSL.GPU components.

## Table of Contents

- [GpuAutoConfig](#gpuautoconfig)
- [GpuMemoryManager](#gpumemorymanager)
- [GpuKernels](#gpukernels)
- [QuantizationEngine](#quantizationengine)
- [Float16Ops](#float16ops)
- [OperatorFusion](#operatorfusion)
- [MultiGpuScheduler](#multigpuscheduler)
- [DistributedTraining](#distributedtraining)
- [DynamicShapeManager](#dynamicshapemanager)
- [ModelSerializer](#modelserializer)

---

## GpuAutoConfig

Automatic GPU detection and configuration with hot-swap support.

### Constructor

```csharp
public GpuAutoConfig(Config? config = null)
```

### Configuration Options

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EnableMonitoring` | bool | true | Monitor for GPU changes |
| `MonitoringIntervalMs` | int | 5000 | Check interval in ms |
| `PreferCuda` | bool | true | Prefer CUDA over OpenCL |
| `AllowCpuFallback` | bool | true | Fall back to CPU if no GPU |
| `MinimumMemoryMB` | int | 1024 | Minimum required VRAM |
| `AutoOptimize` | bool | true | Apply architecture optimizations |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `CurrentGpu` | GpuInfo? | Currently active GPU |
| `AvailableGpus` | List<GpuInfo> | All detected GPUs |
| `Configuration` | Config | Current configuration |

### Methods

```csharp
// Get the accelerator for tensor operations
Accelerator GetAccelerator()

// Get compiled GPU kernels
GpuKernels GetKernels()

// Force GPU rescan
void Rescan()

// Activate specific GPU
void ActivateGpu(GpuInfo gpu)

// Switch to CPU fallback
void ActivateCpuFallback()
```

### Events

```csharp
event EventHandler<GpuChangedEventArgs>? OnGpuChanged
```

### GpuInfo Properties

| Property | Type | Description |
|----------|------|-------------|
| `Name` | string | GPU name |
| `DeviceId` | string | Unique device ID |
| `Backend` | GpuBackend | CUDA/OpenCL/CPU |
| `Architecture` | GpuArchitecture | Kepler/Maxwell/etc |
| `MemoryBytes` | long | Total VRAM |
| `ComputeUnits` | int | CUDA cores/SMs |
| `SupportsTensorCores` | bool | Has Tensor Cores |
| `SupportsFloat16` | bool | Has FP16 support |
| `SupportsInt8` | bool | Has INT8 support |
| `Score` | float | Performance score |

---

## GpuMemoryManager

VRAM monitoring, memory pooling, and adaptive batch sizing.

### Constructor

```csharp
public GpuMemoryManager(Accelerator accelerator, PoolConfig? config = null)
```

### Configuration Options

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EnablePooling` | bool | true | Enable buffer pooling |
| `MaxPoolSizeBytes` | long | 512MB | Max pooled memory |
| `EnableAutoGC` | bool | true | Auto garbage collection |
| `GCIntervalMs` | int | 30000 | GC check interval |
| `MemoryPressureThreshold` | float | 0.85 | Trigger GC at this % |
| `SystemReservePercent` | float | 0.10 | Reserved for system |
| `EnableAdaptiveBatching` | bool | true | Auto batch sizing |

### Methods

```csharp
// Allocate float buffer (pooled if available)
MemoryBuffer1D<float, Stride1D.Dense> AllocateFloat(int size, string? tag = null)

// Allocate byte buffer for INT8
MemoryBuffer1D<sbyte, Stride1D.Dense> AllocateByte(int size, string? tag = null)

// Return buffer to pool
void Release(MemoryBuffer1D<float, Stride1D.Dense> buffer)

// Get memory statistics
MemoryStats GetMemoryStats()

// Calculate optimal batch size for available memory
int GetOptimalBatchSize(long sampleSizeBytes, int preferredBatchSize, int minBatchSize = 1)

// Estimate memory for model
long EstimateModelMemory(int[] layerSizes, int batchSize, bool includeGradients = true)

// Check if model fits in memory
bool CanFitModel(long modelSizeBytes)

// Force garbage collection
void ForceGC()

// Clear all pools
void ClearPools()
```

### MemoryStats Properties

| Property | Type | Description |
|----------|------|-------------|
| `TotalVRAM` | long | Total GPU memory |
| `AvailableVRAM` | long | Available memory |
| `UsedVRAM` | long | Currently used |
| `AllocatedByNSL` | long | NSL allocations |
| `PooledMemory` | long | Memory in pools |
| `PeakAllocated` | long | Peak usage |
| `ActiveAllocations` | int | Active buffers |
| `PoolHitRate` | float | Pool efficiency |
| `MemoryPressure` | float | Usage percentage |

---

## QuantizationEngine

INT8 quantization for memory-efficient inference.

### Constructor

```csharp
public QuantizationEngine(Accelerator accelerator)
```

### Methods

```csharp
// Calibrate quantization parameters
QuantParams CalibrateMinMax(float[] data, bool symmetric = true,
    bool perChannel = false, int numChannels = 1)

// Entropy-based calibration
QuantParams CalibrateEntropy(float[] data, int numBins = 2048)

// Quantize FP32 tensor to INT8
QuantizedTensor Quantize(GpuTensor tensor, QuantParams? existingParams = null)

// Dequantize INT8 back to FP32
GpuTensor Dequantize(QuantizedTensor quantized)

// INT8 matrix multiplication
GpuTensor Int8MatMul(QuantizedTensor a, QuantizedTensor b)
```

### QuantParams Properties

| Property | Type | Description |
|----------|------|-------------|
| `Scale` | float | Quantization scale |
| `ZeroPoint` | sbyte | Zero point offset |
| `IsSymmetric` | bool | Symmetric mode |
| `PerChannel` | bool | Per-channel mode |
| `ChannelScales` | float[]? | Per-channel scales |

---

## Float16Ops

FP16 operations for 2x memory efficiency.

### Constructor

```csharp
public Float16Ops(Accelerator accelerator, bool hasTensorCores = false)
```

### Methods

```csharp
// Convert FP32 tensor to FP16
Float16Tensor ToFloat16(GpuTensor tensor)

// Convert FP16 tensor to FP32
GpuTensor ToFloat32(Float16Tensor tensor)

// Create FP16 tensor from array
Float16Tensor FromArray(float[] data, params int[] shape)

// Extract FP16 data as float array
float[] ToArray(Float16Tensor tensor)

// Mixed-precision matmul
GpuTensor MixedPrecisionMatMul(Float16Tensor a, Float16Tensor b, GpuKernels kernels)

// CPU-side conversions
static ushort FloatToHalf(float value)
static float HalfToFloat(ushort value)

// Calculate memory savings
static (long fp32Bytes, long fp16Bytes, float savingsPercent)
    CalculateMemorySavings(int[] shape)
```

---

## OperatorFusion

Fused GPU kernels for reduced memory bandwidth.

### Constructor

```csharp
public OperatorFusion(Accelerator accelerator)
```

### Methods

```csharp
// Fused Linear + ReLU
GpuTensor FusedLinearRelu(GpuTensor input, GpuTensor weight, GpuTensor bias)

// Fused Linear + BatchNorm + ReLU
GpuTensor FusedLinearBNRelu(GpuTensor input, GpuTensor weight, GpuTensor bias,
    GpuTensor bnGamma, GpuTensor bnBeta, float eps = 1e-5f)

// Fused LayerNorm + Dropout
GpuTensor FusedLayerNormDropout(GpuTensor input, GpuTensor gamma, GpuTensor beta,
    float dropProb = 0.1f)

// Fused Bias + GELU
GpuTensor FusedBiasGelu(GpuTensor input, GpuTensor bias)

// Fused Scaled Dot-Product Attention
GpuTensor FusedAttention(GpuTensor query, GpuTensor key, GpuTensor value,
    GpuTensor? mask, float scale)
```

---

## MultiGpuScheduler

Multi-GPU work distribution.

### Constructor

```csharp
public MultiGpuScheduler(Context context, List<GpuAutoConfig.GpuInfo> gpus)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Strategy` | SchedulingStrategy | Work distribution mode |
| `Workers` | IReadOnlyList<GpuWorker> | GPU workers |
| `GpuCount` | int | Number of GPUs |

### Scheduling Strategies

- `RoundRobin` - Even distribution
- `LeastLoaded` - Send to least busy GPU
- `PerformanceWeighted` - Weight by GPU score
- `MemoryAware` - Consider available VRAM

### Methods

```csharp
// Submit work to any GPU
Task<GpuTensor> SubmitAsync(Func<GpuWorker, GpuTensor> work, long estimatedMemory = 0)

// Split batch across all GPUs
Task<GpuTensor> ParallelBatchProcess(float[] batchData, int[] batchShape,
    Func<GpuWorker, GpuTensor, GpuTensor> processFunc)

// Get statistics
string GetStatistics()
```

---

## DistributedTraining

Multi-node distributed training over network.

### Constructor

```csharp
public DistributedTraining(Accelerator accelerator, GpuKernels kernels,
    DistributedConfig config)
```

### Configuration

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `Rank` | int | 0 | Node rank (0=master) |
| `WorldSize` | int | 1 | Total nodes |
| `MasterAddress` | string | localhost | Master IP |
| `Port` | int | 29500 | Communication port |
| `UseGradientCompression` | bool | true | Compress to FP16 |
| `TimeoutMs` | int | 30000 | Connection timeout |

### Methods

```csharp
// Initialize distributed environment
Task InitializeAsync()

// Synchronization barrier
Task BarrierAsync()

// Sum gradients across all nodes
Task<GpuTensor> AllReduceAsync(GpuTensor localGradients, ReduceOp op = ReduceOp.Sum)

// Broadcast from root to all
Task<GpuTensor> BroadcastAsync(GpuTensor? tensor, int root = 0)

// Scatter chunks to nodes
Task<GpuTensor> ScatterAsync(GpuTensor? tensor, int root = 0)

// Gather from all to root
Task<GpuTensor?> GatherAsync(GpuTensor localTensor, int root = 0)
```

### ReduceOp Options

- `Sum` - Sum all gradients
- `Mean` - Average across nodes
- `Max` - Maximum value
- `Min` - Minimum value

---

## DynamicShapeManager

Handle variable-sized tensors at runtime.

### Constructor

```csharp
public DynamicShapeManager(Accelerator accelerator, GpuMemoryManager? memoryManager = null)
```

### Methods

```csharp
// Register named shape with dynamic dims (-1)
void RegisterShape(string name, int[] dims, string[]? dimNames = null)

// Register common NLP/vision shapes
void RegisterCommonShapes()

// Get registered shape
ShapeInfo? GetShape(string name)

// Infer output shape from operation
int[] InferShape(string operation, params int[][] inputShapes)

// Create dynamic tensor
DynamicTensor CreateDynamicTensor(string name, ShapeInfo shape)

// Get or create dynamic tensor
DynamicTensor GetOrCreateDynamic(string name, int[] initialShape)
```

### Supported Operations for Shape Inference

- `matmul`, `add`, `sub`, `mul`, `div`
- `relu`, `sigmoid`, `tanh`, `gelu`
- `softmax`, `layernorm`, `batchnorm`
- `transpose`, `reshape`, `concat`
- `conv2d`, `maxpool2d`, `avgpool2d`

---

## ModelSerializer

Native binary format (.nslm) for model persistence.

### Static Methods

```csharp
// Save model
static void Save(string path,
    Dictionary<string, (float[] Data, int[] Shape)> tensors,
    ModelMetadata? metadata = null)

// Load model
static (Dictionary<string, (float[] Data, int[] Shape)> Tensors, ModelMetadata Metadata)
    Load(string path)

// Save training checkpoint
static void SaveCheckpoint(string path,
    Dictionary<string, (float[] Data, int[] Shape)> modelState,
    Dictionary<string, (float[] Data, int[] Shape)>? optimizerState,
    int epoch, int step, float loss)

// Load checkpoint
static (Dictionary<string, (float[] Data, int[] Shape)> ModelState,
    Dictionary<string, (float[] Data, int[] Shape)>? OptimizerState,
    int Epoch, int Step, float Loss) LoadCheckpoint(string path)
```

### File Format

```
NSLM Binary Format:
├── Header (12 bytes)
│   ├── Magic: 0x4E534C4D ("NSLM")
│   ├── Version: uint32
│   └── Flags: uint32
├── Metadata
│   ├── Name, Architecture, Timestamp
│   └── Custom key-value pairs
├── Tensor Table
│   └── (Name, Shape, Dtype, Offset, Size) entries
├── Tensor Data (64-byte aligned)
└── Checksum (4 bytes)
```

---

## Quick Start Examples

### Basic GPU Operations

```csharp
using NSL.GPU;

// Auto-configure GPU
using var config = new GpuAutoConfig();
var kernels = config.GetKernels();

// Create tensors
var a = GpuTensor.FromArray(config.GetAccelerator(),
    new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
var b = GpuTensor.FromArray(config.GetAccelerator(),
    new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });

// Matrix multiplication
var c = kernels.MatMul(a, b);
Console.WriteLine($"Result: {string.Join(", ", c.ToArray())}");
```

### Memory-Efficient Inference

```csharp
// Setup
using var config = new GpuAutoConfig();
var accel = config.GetAccelerator();
var memManager = new GpuMemoryManager(accel);
var quantizer = new QuantizationEngine(accel);

// Load weights as INT8
var weights = /* load your weights */;
var quantParams = quantizer.CalibrateMinMax(weights);
var quantizedWeights = quantizer.Quantize(
    GpuTensor.FromArray(accel, weights, shape), quantParams);

// Adaptive batch sizing
int batchSize = memManager.GetOptimalBatchSize(
    sampleSizeBytes: inputSize * 4,
    preferredBatchSize: 64
);
```

### Distributed Training

```csharp
// On each node
var distConfig = new DistributedTraining.DistributedConfig
{
    Rank = int.Parse(Environment.GetEnvironmentVariable("RANK")!),
    WorldSize = int.Parse(Environment.GetEnvironmentVariable("WORLD_SIZE")!),
    MasterAddress = Environment.GetEnvironmentVariable("MASTER_ADDR")!
};

using var dist = new DistributedTraining(accelerator, kernels, distConfig);
await dist.InitializeAsync();

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    var gradients = ComputeGradients(batch);
    var syncedGradients = await dist.AllReduceAsync(gradients, ReduceOp.Mean);
    ApplyGradients(syncedGradients);
}
```
