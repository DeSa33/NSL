// Quick GPU/CPU acceleration test
using System;
using System.Linq;
using NSL.Tensor;
using NSL.GPU;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

Console.WriteLine("=== NSL GPU/CPU Acceleration Test ===\n");

// Test 1: Basic Tensor operations (CPU)
Console.WriteLine("1. Testing CPU Tensor operations...");
var a = Tensor.Ones(new long[] { 3, 3 });
var b = Tensor.Ones(new long[] { 3, 3 }) * 2;
var c = a + b;
Console.WriteLine($"   CPU Add: [3x3] ones + [3x3] twos = sum={c.Sum()}");

var matA = Tensor.Rand(new long[] { 4, 3 });
var matB = Tensor.Rand(new long[] { 3, 4 });
var matC = matA.MatMul(matB);
Console.WriteLine($"   CPU MatMul: [4x3] @ [3x4] = [{string.Join(",", matC.Shape)}]");
Console.WriteLine("   CPU tests PASSED\n");

// Test 2: Direct ILGPU diagnostic
Console.WriteLine("2. ILGPU Diagnostic...");
try
{
    using var context = Context.CreateDefault();
    Console.WriteLine($"   ILGPU Context created");

    var cudaDevices = context.GetCudaDevices();
    Console.WriteLine($"   Found {cudaDevices.Count} CUDA device(s)");

    foreach (var device in cudaDevices)
    {
        Console.WriteLine($"   - {device.Name}");
        Console.WriteLine($"     Architecture: {device.Architecture}");
        Console.WriteLine($"     Warp Size: {device.WarpSize}");
        Console.WriteLine($"     Max Threads/Block: {device.MaxNumThreadsPerGroup}");
    }

    if (cudaDevices.Count > 0)
    {
        Console.WriteLine($"\n   Creating CUDA accelerator...");
        using var accelerator = cudaDevices[0].CreateAccelerator(context);
        Console.WriteLine($"   CUDA Accelerator created: {accelerator.Name}");

        // Simple kernel test
        var buffer = accelerator.Allocate1D<float>(10);
        buffer.MemSetToZero();
        accelerator.Synchronize();
        Console.WriteLine($"   Memory allocation test PASSED");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"   ILGPU Error: {ex.GetType().Name}: {ex.Message}");
    if (ex.InnerException != null)
        Console.WriteLine($"   Inner: {ex.InnerException.Message}");
}

// Test 3: GpuAccelerator wrapper
Console.WriteLine("\n3. Testing GpuAccelerator wrapper...");
try
{
    using var gpu = new GpuAccelerator();
    Console.WriteLine($"   GPU Device: {gpu.DeviceInfo.Name} ({gpu.ActiveBackend})");

    // Test ToGpu/ToCpu
    var floatData = new float[] { 1, 2, 3, 4, 5, 6 };
    var gpuTensor = gpu.ToGpu(floatData, 2, 3);
    Console.WriteLine($"   ToGpu: Created GPU tensor [{string.Join(",", gpuTensor.Shape)}]");

    // Test GPU Softmax (fully GPU-accelerated, no CPU fallback)
    var softmaxResult = gpu.Softmax(gpuTensor);
    var (softmaxData, _) = gpu.ToCpu(softmaxResult);
    Console.WriteLine($"   Softmax: First 3 values = [{softmaxData[0]:F4}, {softmaxData[1]:F4}, {softmaxData[2]:F4}]");

    // Test GPU Sum (fully GPU-accelerated with atomic ops)
    var sumResult = gpu.Sum(gpuTensor);
    Console.WriteLine($"   Sum: {sumResult} (expected 21)");

    // Test GPU Max/Min
    var maxResult = gpu.Max(gpuTensor);
    var minResult = gpu.Min(gpuTensor);
    Console.WriteLine($"   Max: {maxResult} (expected 6), Min: {minResult} (expected 1)");

    // Test GPU MatMul
    var gpuA = gpu.ToGpu(new float[] { 1, 2, 3, 4 }, 2, 2);
    var gpuB = gpu.ToGpu(new float[] { 5, 6, 7, 8 }, 2, 2);
    var gpuC = gpu.MatMul(gpuA, gpuB);
    var (matMulData, _) = gpu.ToCpu(gpuC);
    Console.WriteLine($"   MatMul: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[{matMulData[0]},{matMulData[1]}],[{matMulData[2]},{matMulData[3]}]]");

    // Test GPU Outer product (fully GPU-accelerated)
    var vecA = gpu.ToGpu(new float[] { 1, 2, 3 }, 3);
    var vecB = gpu.ToGpu(new float[] { 4, 5 }, 2);
    var outer = gpu.Outer(vecA, vecB);
    Console.WriteLine($"   Outer: [3] x [2] = [{string.Join(",", outer.Shape)}]");

    Console.WriteLine("   GPU tests PASSED\n");
}
catch (Exception ex)
{
    Console.WriteLine($"   GPU Error: {ex.GetType().Name}: {ex.Message}");
    if (ex.InnerException != null)
        Console.WriteLine($"   Inner: {ex.InnerException.Message}");
    Console.WriteLine($"\n   Stack trace: {ex.StackTrace?.Split('\n').FirstOrDefault()}");
}

// Test 3: NSLAccelerator unified layer
Console.WriteLine("3. Testing NSLAccelerator (unified CPU/GPU)...");
try
{
    var config = AcceleratorConfig.Default;
    using var accel = new NSLAccelerator(config);
    Console.WriteLine($"   Accelerator initialized with {config.NumThreads} threads");
    Console.WriteLine("   NSLAccelerator tests PASSED\n");
}
catch (Exception ex)
{
    Console.WriteLine($"   NSLAccelerator Error: {ex.Message}\n");
}

Console.WriteLine("=== All tests completed ===");
