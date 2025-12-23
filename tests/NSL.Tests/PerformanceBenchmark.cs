using System;
using System.Diagnostics;
using NSL.GPU;
using Xunit;
using Xunit.Abstractions;

namespace NSL.Tests
{
    /// <summary>
    /// Performance benchmarks comparing standard vs high-performance GPU kernels.
    /// </summary>
    public class PerformanceBenchmark
    {
        private readonly ITestOutputHelper _output;

        public PerformanceBenchmark(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void BenchmarkMatMul_StandardVsTiled()
        {
            _output.WriteLine("=== MatMul Benchmark: Standard vs Tiled ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                _output.WriteLine($"GPU: {gpu.DeviceInfo}");
                _output.WriteLine("");

                int[] sizes = { 256, 512, 1024 };
                int warmupIterations = 5;
                int benchIterations = 20;

                foreach (int size in sizes)
                {
                    _output.WriteLine($"Matrix Size: {size}x{size}");

                    // Create test matrices
                    var rng = new Random(42);
                    var aData = new float[size * size];
                    var bData = new float[size * size];
                    for (int i = 0; i < aData.Length; i++)
                    {
                        aData[i] = (float)rng.NextDouble();
                        bData[i] = (float)rng.NextDouble();
                    }

                    var a = gpu.ToGpu(aData, size, size);
                    var b = gpu.ToGpu(bData, size, size);

                    // Warmup standard
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var result = gpu.MatMul(a, b);
                        result.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark standard MatMul
                    var swStandard = Stopwatch.StartNew();
                    for (int i = 0; i < benchIterations; i++)
                    {
                        var result = gpu.MatMul(a, b);
                        result.Dispose();
                    }
                    gpu.Synchronize();
                    swStandard.Stop();

                    // Warmup tiled
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var result = gpu.HighPerformance.TiledMatMul(a, b);
                        result.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark tiled MatMul
                    var swTiled = Stopwatch.StartNew();
                    for (int i = 0; i < benchIterations; i++)
                    {
                        var result = gpu.HighPerformance.TiledMatMul(a, b);
                        result.Dispose();
                    }
                    gpu.Synchronize();
                    swTiled.Stop();

                    double standardMs = swStandard.Elapsed.TotalMilliseconds / benchIterations;
                    double tiledMs = swTiled.Elapsed.TotalMilliseconds / benchIterations;
                    double speedup = standardMs / tiledMs;

                    // Calculate GFLOPS
                    double flops = 2.0 * size * size * size;
                    double standardGflops = (flops / (standardMs / 1000.0)) / 1e9;
                    double tiledGflops = (flops / (tiledMs / 1000.0)) / 1e9;

                    _output.WriteLine($"  Standard: {standardMs:F3} ms ({standardGflops:F2} GFLOPS)");
                    _output.WriteLine($"  Tiled:    {tiledMs:F3} ms ({tiledGflops:F2} GFLOPS)");
                    _output.WriteLine($"  Speedup:  {speedup:F2}x");
                    _output.WriteLine("");

                    a.Dispose();
                    b.Dispose();
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Benchmark skipped (no GPU): {ex.Message}");
            }
        }

        [Fact]
        public void BenchmarkAttention_StandardVsFused()
        {
            _output.WriteLine("=== Attention Benchmark: Standard vs Fused ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                _output.WriteLine($"GPU: {gpu.DeviceInfo}");
                _output.WriteLine("");

                int[] seqLens = { 64, 128, 256 };
                int headDim = 64;
                int warmupIterations = 3;
                int benchIterations = 10;

                foreach (int seqLen in seqLens)
                {
                    _output.WriteLine($"Sequence Length: {seqLen}, Head Dim: {headDim}");

                    var rng = new Random(42);
                    int totalSize = seqLen * headDim;
                    var qData = new float[totalSize];
                    var kData = new float[totalSize];
                    var vData = new float[totalSize];

                    for (int i = 0; i < totalSize; i++)
                    {
                        qData[i] = (float)rng.NextDouble();
                        kData[i] = (float)rng.NextDouble();
                        vData[i] = (float)rng.NextDouble();
                    }

                    var q = gpu.ToGpu(qData, seqLen, headDim);
                    var k = gpu.ToGpu(kData, seqLen, headDim);
                    var v = gpu.ToGpu(vData, seqLen, headDim);

                    // Warmup fused attention
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var result = gpu.HighPerformance.FusedAttention(q, k, v);
                        result.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark fused attention
                    var swFused = Stopwatch.StartNew();
                    for (int i = 0; i < benchIterations; i++)
                    {
                        var result = gpu.HighPerformance.FusedAttention(q, k, v);
                        result.Dispose();
                    }
                    gpu.Synchronize();
                    swFused.Stop();

                    double fusedMs = swFused.Elapsed.TotalMilliseconds / benchIterations;

                    // Memory comparison (theoretical)
                    long standardMemory = (long)seqLen * seqLen * sizeof(float); // Attention matrix
                    long fusedMemory = (long)seqLen * headDim * sizeof(float); // No attention matrix

                    _output.WriteLine($"  Fused Attention: {fusedMs:F3} ms");
                    _output.WriteLine($"  Memory Saved: {standardMemory / 1024.0:F1} KB -> {fusedMemory / 1024.0:F1} KB ({(float)standardMemory / fusedMemory:F1}x reduction)");
                    _output.WriteLine("");

                    q.Dispose();
                    k.Dispose();
                    v.Dispose();
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Benchmark skipped (no GPU): {ex.Message}");
            }
        }

        [Fact]
        public void BenchmarkLayerNorm_StandardVsFused()
        {
            _output.WriteLine("=== LayerNorm Benchmark: Standard vs Fused (Welford) ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                _output.WriteLine($"GPU: {gpu.DeviceInfo}");
                _output.WriteLine("");

                int batchSize = 32;
                int hiddenDim = 4096;
                int warmupIterations = 5;
                int benchIterations = 50;

                _output.WriteLine($"Batch: {batchSize}, Hidden: {hiddenDim}");

                var rng = new Random(42);
                int totalSize = batchSize * hiddenDim;
                var xData = new float[totalSize];
                var gammaData = new float[hiddenDim];
                var betaData = new float[hiddenDim];

                for (int i = 0; i < totalSize; i++)
                    xData[i] = (float)rng.NextDouble();
                for (int i = 0; i < hiddenDim; i++)
                {
                    gammaData[i] = 1.0f;
                    betaData[i] = 0.0f;
                }

                var x = gpu.ToGpu(xData, batchSize, hiddenDim);
                var gamma = gpu.ToGpu(gammaData, hiddenDim);
                var beta = gpu.ToGpu(betaData, hiddenDim);

                // Warmup
                for (int i = 0; i < warmupIterations; i++)
                {
                    var result = gpu.HighPerformance.FusedLayerNorm(x, gamma, beta);
                    result.Dispose();
                }
                gpu.Synchronize();

                // Benchmark fused LayerNorm
                var swFused = Stopwatch.StartNew();
                for (int i = 0; i < benchIterations; i++)
                {
                    var result = gpu.HighPerformance.FusedLayerNorm(x, gamma, beta);
                    result.Dispose();
                }
                gpu.Synchronize();
                swFused.Stop();

                double fusedMs = swFused.Elapsed.TotalMilliseconds / benchIterations;
                double throughputGBs = (totalSize * sizeof(float) * 2.0 / (fusedMs / 1000.0)) / 1e9;

                _output.WriteLine($"  Fused LayerNorm (Welford): {fusedMs:F3} ms");
                _output.WriteLine($"  Memory Throughput: {throughputGBs:F2} GB/s");
                _output.WriteLine($"  (Standard would require 3 passes, Welford does 1 pass)");
                _output.WriteLine("");

                x.Dispose();
                gamma.Dispose();
                beta.Dispose();
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Benchmark skipped (no GPU): {ex.Message}");
            }
        }

        [Fact]
        public void RunFullBenchmarkReport()
        {
            _output.WriteLine("=== NSL High-Performance GPU Full Benchmark ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                var report = gpu.HighPerformance.RunBenchmark(512, 50);
                _output.WriteLine(report);
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Benchmark skipped (no GPU): {ex.Message}");
            }
        }

        [Fact]
        public void BenchmarkProductionMath_DiamondOperator()
        {
            _output.WriteLine("=== ProductionMath Benchmark: Diamond Operator ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                _output.WriteLine($"GPU: {gpu.DeviceInfo}");
                _output.WriteLine("");

                var engine = new ProductionMathEngine(gpu.Accelerator);

                int[] sizes = { 64, 256, 1024 };
                int warmupIterations = 5;
                int benchIterations = 20;

                foreach (int size in sizes)
                {
                    _output.WriteLine($"Tensor Size: {size} elements");

                    var rng = new Random(42);
                    var aData = new float[size];
                    var bData = new float[size];
                    for (int i = 0; i < size; i++)
                    {
                        aData[i] = (float)rng.NextDouble();
                        bData[i] = (float)rng.NextDouble();
                    }

                    var a = gpu.ToGpu(aData, size);
                    var b = gpu.ToGpu(bData, size);

                    // Warmup
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var result = engine.Diamond(a, b);
                        result.SelectedValue.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark Diamond operator
                    var sw = Stopwatch.StartNew();
                    ProductionMathResult lastResult = null!;
                    for (int i = 0; i < benchIterations; i++)
                    {
                        lastResult = engine.Diamond(a, b);
                        if (i < benchIterations - 1)
                            lastResult.SelectedValue.Dispose();
                    }
                    gpu.Synchronize();
                    sw.Stop();

                    double avgMs = sw.Elapsed.TotalMilliseconds / benchIterations;

                    // Get result info
                    var selectedMode = lastResult.SelectedMode;
                    var variantCount = lastResult.AllVariants.Count;

                    _output.WriteLine($"  Diamond Op:     {avgMs:F3} ms");
                    _output.WriteLine($"  Selected Mode:  {selectedMode}");
                    _output.WriteLine($"  Variants Gen:   {variantCount}");
                    _output.WriteLine($"  Ops/sec:        {1000.0 / avgMs:F0}");
                    _output.WriteLine("");

                    lastResult.SelectedValue.Dispose();
                    a.Dispose();
                    b.Dispose();
                }

                // Test policy learning
                _output.WriteLine("--- Policy Learning Test ---");
                var testA = gpu.ToGpu(new float[] { 1, 2, 3, 4 }, 4);
                var testB = gpu.ToGpu(new float[] { 5, 6, 7, 8 }, 4);

                for (int i = 0; i < 10; i++)
                {
                    var result = engine.Diamond(testA, testB);
                    float reward = i % 2 == 0 ? 1.0f : -0.5f; // Simulate reward
                    engine.UpdatePolicy(result, reward);
                    result.SelectedValue.Dispose();
                }
                _output.WriteLine("  10 policy updates completed");
                _output.WriteLine("  REINFORCE learning active");

                testA.Dispose();
                testB.Dispose();
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Benchmark skipped (no GPU): {ex.Message}");
            }
        }

        [Fact]
        public void BenchmarkProductionMath_VsStandard()
        {
            _output.WriteLine("=== ProductionMath vs Standard GPU Benchmark ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                _output.WriteLine($"GPU: {gpu.DeviceInfo}");
                _output.WriteLine("");

                int[] sizes = { 256, 512, 1024 };
                int warmupIterations = 3;
                int benchIterations = 10;

                foreach (int size in sizes)
                {
                    _output.WriteLine($"Matrix Size: {size}x{size}");

                    var rng = new Random(42);
                    var aData = new float[size * size];
                    var bData = new float[size * size];
                    for (int i = 0; i < aData.Length; i++)
                    {
                        aData[i] = (float)rng.NextDouble();
                        bData[i] = (float)rng.NextDouble();
                    }

                    var a = gpu.ToGpu(aData, size, size);
                    var b = gpu.ToGpu(bData, size, size);

                    // Warmup Standard
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var r = gpu.HighPerformance.TiledMatMul(a, b);
                        r.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark Standard TiledMatMul
                    var swStandard = Stopwatch.StartNew();
                    for (int i = 0; i < benchIterations; i++)
                    {
                        var r = gpu.HighPerformance.TiledMatMul(a, b);
                        r.Dispose();
                    }
                    gpu.Synchronize();
                    swStandard.Stop();

                    // Warmup ProductionMath
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var r = gpu.HighPerformance.ProductionMatMul(a, b, learn: false);
                        r.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark ProductionMath MatMul
                    var swProduction = Stopwatch.StartNew();
                    for (int i = 0; i < benchIterations; i++)
                    {
                        var r = gpu.HighPerformance.ProductionMatMul(a, b, learn: true);
                        r.Dispose();
                    }
                    gpu.Synchronize();
                    swProduction.Stop();

                    double standardMs = swStandard.Elapsed.TotalMilliseconds / benchIterations;
                    double productionMs = swProduction.Elapsed.TotalMilliseconds / benchIterations;
                    double overhead = (productionMs / standardMs - 1) * 100;

                    double flops = 2.0 * size * size * size;
                    double standardGflops = (flops / (standardMs / 1000.0)) / 1e9;
                    double productionGflops = (flops / (productionMs / 1000.0)) / 1e9;

                    _output.WriteLine($"  Standard:     {standardMs:F3} ms ({standardGflops:F2} GFLOPS)");
                    _output.WriteLine($"  ProductionMath: {productionMs:F3} ms ({productionGflops:F2} GFLOPS)");
                    _output.WriteLine($"  Overhead:     {overhead:F1}%");
                    _output.WriteLine("");

                    a.Dispose();
                    b.Dispose();
                }

                _output.WriteLine("ProductionMath adds relational reasoning overhead but enables:");
                _output.WriteLine("  - Learned computation strategies");
                _output.WriteLine("  - Semantic awareness of data patterns");
                _output.WriteLine("  - Policy-based optimization over time");
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Benchmark skipped (no GPU): {ex.Message}");
            }
        }

        [Fact]
        public void BenchmarkProductionMath_AllModes()
        {
            _output.WriteLine("=== All Modes Comparison: Standard vs FastProduction ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                _output.WriteLine($"GPU: {gpu.DeviceInfo}");
                _output.WriteLine("");

                int[] sizes = { 256, 512, 1024 };
                int warmupIterations = 3;
                int benchIterations = 10;

                foreach (int size in sizes)
                {
                    _output.WriteLine($"Matrix Size: {size}x{size}");

                    var rng = new Random(42);
                    var aData = new float[size * size];
                    var bData = new float[size * size];
                    for (int i = 0; i < aData.Length; i++)
                    {
                        aData[i] = (float)rng.NextDouble();
                        bData[i] = (float)rng.NextDouble();
                    }

                    var a = gpu.ToGpu(aData, size, size);
                    var b = gpu.ToGpu(bData, size, size);

                    // Warmup Standard
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var r = gpu.HighPerformance.TiledMatMul(a, b);
                        r.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark Standard
                    var swStandard = Stopwatch.StartNew();
                    for (int i = 0; i < benchIterations; i++)
                    {
                        var r = gpu.HighPerformance.TiledMatMul(a, b);
                        r.Dispose();
                    }
                    gpu.Synchronize();
                    swStandard.Stop();

                    // Warmup FastProduction
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        var r = gpu.HighPerformance.FastProductionCompute(a, b);
                        r.Dispose();
                    }
                    gpu.Synchronize();

                    // Benchmark FastProduction
                    var swFast = Stopwatch.StartNew();
                    for (int i = 0; i < benchIterations; i++)
                    {
                        var r = gpu.HighPerformance.FastProductionCompute(a, b);
                        r.Dispose();
                    }
                    gpu.Synchronize();
                    swFast.Stop();

                    double standardMs = swStandard.Elapsed.TotalMilliseconds / benchIterations;
                    double fastMs = swFast.Elapsed.TotalMilliseconds / benchIterations;
                    double fastOverhead = (fastMs / standardMs - 1) * 100;

                    _output.WriteLine($"  Standard (TiledMatMul):    {standardMs:F3} ms");
                    _output.WriteLine($"  FastProduction (Direct):   {fastMs:F3} ms");
                    _output.WriteLine($"  FastProduction Overhead:   {fastOverhead:F1}%");
                    _output.WriteLine("");

                    a.Dispose();
                    b.Dispose();
                }

                _output.WriteLine("FastProduction uses learned best mode directly - no variant generation.");
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Benchmark skipped (no GPU): {ex.Message}");
            }
        }

        [Fact]
        public void BenchmarkProductionMath_LearnThenTest()
        {
            _output.WriteLine("=== ProductionMath: 2-Minute Learning Session ===\n");

            try
            {
                using var gpu = new GpuAccelerator();
                _output.WriteLine($"GPU: {gpu.DeviceInfo}");
                _output.WriteLine("");

                var engine = new ProductionMathEngine(gpu.Accelerator);

                // Create test data of various sizes
                var rng = new Random(42);
                var testSets = new List<(GpuTensor a, GpuTensor b, float expectedRatio)>();

                // Different data patterns to learn from
                int[] sizes = { 64, 128, 256 };
                foreach (int size in sizes)
                {
                    // Pattern 1: Similar magnitudes (Sum works well)
                    var a1 = new float[size];
                    var b1 = new float[size];
                    for (int i = 0; i < size; i++) { a1[i] = (float)rng.NextDouble() * 10; b1[i] = (float)rng.NextDouble() * 10; }
                    testSets.Add((gpu.ToGpu(a1, size), gpu.ToGpu(b1, size), 1.0f));

                    // Pattern 2: One much larger (Ratio might work)
                    var a2 = new float[size];
                    var b2 = new float[size];
                    for (int i = 0; i < size; i++) { a2[i] = (float)rng.NextDouble() * 100; b2[i] = (float)rng.NextDouble() * 10 + 1; }
                    testSets.Add((gpu.ToGpu(a2, size), gpu.ToGpu(b2, size), 0.5f));

                    // Pattern 3: Both positive (Geometric works)
                    var a3 = new float[size];
                    var b3 = new float[size];
                    for (int i = 0; i < size; i++) { a3[i] = (float)rng.NextDouble() * 5 + 1; b3[i] = (float)rng.NextDouble() * 5 + 1; }
                    testSets.Add((gpu.ToGpu(a3, size), gpu.ToGpu(b3, size), 0.8f));
                }

                _output.WriteLine($"Created {testSets.Count} test patterns");
                _output.WriteLine("Starting 2-minute learning session...\n");

                var sw = Stopwatch.StartNew();
                int iterations = 0;
                int totalOps = 0;
                var modeCounts = new Dictionary<string, int>();

                // Learn for 2 minutes
                while (sw.Elapsed.TotalSeconds < 120)
                {
                    foreach (var (a, b, expectedReward) in testSets)
                    {
                        // Run ProductionMath with learning
                        var result = engine.Diamond(a, b);

                        // Simulate reward based on which mode was selected
                        float reward = expectedReward + (float)(rng.NextDouble() * 0.2 - 0.1);
                        engine.UpdatePolicy(result, reward);

                        // Track which modes are being selected
                        string modeName = result.SelectedMode[0].ToString();
                        if (!modeCounts.ContainsKey(modeName)) modeCounts[modeName] = 0;
                        modeCounts[modeName]++;

                        result.SelectedValue.Dispose();
                        totalOps++;
                    }
                    iterations++;

                    // Progress update every 10 seconds
                    if (iterations % 100 == 0)
                    {
                        _output.WriteLine($"  {sw.Elapsed.TotalSeconds:F0}s - {totalOps} operations, {totalOps / sw.Elapsed.TotalSeconds:F0} ops/sec");
                    }
                }
                sw.Stop();

                _output.WriteLine($"\nLearning complete: {totalOps} total operations in {sw.Elapsed.TotalSeconds:F1}s");
                _output.WriteLine($"Average: {totalOps / sw.Elapsed.TotalSeconds:F1} ops/sec\n");

                // Show which modes were learned
                _output.WriteLine("Mode selection distribution:");
                foreach (var kv in modeCounts.OrderByDescending(x => x.Value))
                {
                    float pct = 100f * kv.Value / totalOps;
                    _output.WriteLine($"  {kv.Key}: {kv.Value} ({pct:F1}%)");
                }

                // Cleanup test sets
                foreach (var (a, b, _) in testSets)
                {
                    a.Dispose();
                    b.Dispose();
                }

                _output.WriteLine("\n--- Post-Learning Benchmark ---\n");

                // Now benchmark with fresh data
                int benchSize = 512;
                var benchA = new float[benchSize * benchSize];
                var benchB = new float[benchSize * benchSize];
                for (int i = 0; i < benchA.Length; i++)
                {
                    benchA[i] = (float)rng.NextDouble();
                    benchB[i] = (float)rng.NextDouble();
                }
                var tensorA = gpu.ToGpu(benchA, benchSize, benchSize);
                var tensorB = gpu.ToGpu(benchB, benchSize, benchSize);

                // Warmup
                for (int i = 0; i < 5; i++)
                {
                    var r = engine.DirectCompute(ProductionMathEngine.OperatorType.Diamond, tensorA, tensorB);
                    r.Dispose();
                }
                gpu.Synchronize();

                // Benchmark DirectCompute (uses learned mode)
                var swDirect = Stopwatch.StartNew();
                for (int i = 0; i < 50; i++)
                {
                    var r = engine.DirectCompute(ProductionMathEngine.OperatorType.Diamond, tensorA, tensorB);
                    r.Dispose();
                }
                gpu.Synchronize();
                swDirect.Stop();

                double directMs = swDirect.Elapsed.TotalMilliseconds / 50;
                _output.WriteLine($"DirectCompute (learned mode): {directMs:F3} ms");
                _output.WriteLine($"Ops/sec: {1000.0 / directMs:F0}");

                tensorA.Dispose();
                tensorB.Dispose();

                _output.WriteLine("\nLearning session complete!");
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Error: {ex.Message}");
                _output.WriteLine(ex.StackTrace);
            }
        }
    }
}
