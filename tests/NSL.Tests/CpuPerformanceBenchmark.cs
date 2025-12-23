using System;
using System.Diagnostics;
using System.Linq;
using NSL.Tensor;
using Xunit;
using Xunit.Abstractions;

namespace NSL.Tests
{
    /// <summary>
    /// CPU performance benchmarks comparing standard vs optimized implementations.
    /// These run on CPU so they work without a GPU.
    /// </summary>
    public class CpuPerformanceBenchmark
    {
        private readonly ITestOutputHelper _output;

        public CpuPerformanceBenchmark(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void BenchmarkVectorAdd_StandardVsAVX512()
        {
            _output.WriteLine("=== Vector Add Benchmark: Standard vs SIMD ===\n");
            _output.WriteLine($"CPU Features: {CpuInfo.Summary}");
            _output.WriteLine("");

            int[] sizes = { 10000, 100000, 1000000 };
            int warmupIterations = 10;
            int benchIterations = 100;

            foreach (int size in sizes)
            {
                _output.WriteLine($"Array Size: {size:N0} elements");

                var rng = new Random(42);
                var a = new double[size];
                var b = new double[size];
                var resultStandard = new double[size];
                var resultSIMD = new double[size];

                for (int i = 0; i < size; i++)
                {
                    a[i] = rng.NextDouble();
                    b[i] = rng.NextDouble();
                }

                // Warmup standard
                for (int i = 0; i < warmupIterations; i++)
                    StandardAdd(a, b, resultStandard);

                // Benchmark standard
                var swStandard = Stopwatch.StartNew();
                for (int i = 0; i < benchIterations; i++)
                    StandardAdd(a, b, resultStandard);
                swStandard.Stop();

                // Warmup SIMD
                for (int i = 0; i < warmupIterations; i++)
                    CpuTensorPrimitives.Add(a, b, resultSIMD);

                // Benchmark SIMD
                var swSIMD = Stopwatch.StartNew();
                for (int i = 0; i < benchIterations; i++)
                    CpuTensorPrimitives.Add(a, b, resultSIMD);
                swSIMD.Stop();

                double standardMs = swStandard.Elapsed.TotalMilliseconds / benchIterations;
                double simdMs = swSIMD.Elapsed.TotalMilliseconds / benchIterations;
                double speedup = standardMs / simdMs;

                double gbPerSec = (size * 3 * sizeof(double) / (simdMs / 1000.0)) / 1e9;

                _output.WriteLine($"  Standard:  {standardMs:F4} ms");
                _output.WriteLine($"  SIMD:      {simdMs:F4} ms");
                _output.WriteLine($"  Speedup:   {speedup:F2}x");
                _output.WriteLine($"  Bandwidth: {gbPerSec:F2} GB/s");
                _output.WriteLine("");

                // Verify correctness
                Assert.True(resultStandard.Zip(resultSIMD, (x, y) => Math.Abs(x - y) < 1e-10).All(x => x));
            }
        }

        [Fact]
        public void BenchmarkDotProduct_StandardVsSIMD()
        {
            _output.WriteLine("=== Dot Product Benchmark: Standard vs SIMD ===\n");

            int[] sizes = { 10000, 100000, 1000000 };
            int warmupIterations = 10;
            int benchIterations = 100;

            foreach (int size in sizes)
            {
                _output.WriteLine($"Array Size: {size:N0} elements");

                var rng = new Random(42);
                var a = new double[size];
                var b = new double[size];

                for (int i = 0; i < size; i++)
                {
                    a[i] = rng.NextDouble();
                    b[i] = rng.NextDouble();
                }

                // Warmup
                for (int i = 0; i < warmupIterations; i++)
                {
                    StandardDot(a, b);
                    CpuTensorPrimitives.Dot(a, b);
                }

                // Benchmark standard
                var swStandard = Stopwatch.StartNew();
                double resultStandard = 0;
                for (int i = 0; i < benchIterations; i++)
                    resultStandard = StandardDot(a, b);
                swStandard.Stop();

                // Benchmark SIMD
                var swSIMD = Stopwatch.StartNew();
                double resultSIMD = 0;
                for (int i = 0; i < benchIterations; i++)
                    resultSIMD = CpuTensorPrimitives.Dot(a, b);
                swSIMD.Stop();

                double standardMs = swStandard.Elapsed.TotalMilliseconds / benchIterations;
                double simdMs = swSIMD.Elapsed.TotalMilliseconds / benchIterations;
                double speedup = standardMs / simdMs;

                // FLOPS: 2 ops per element (multiply + add)
                double gflops = (size * 2.0 / (simdMs / 1000.0)) / 1e9;

                _output.WriteLine($"  Standard:  {standardMs:F4} ms");
                _output.WriteLine($"  SIMD:      {simdMs:F4} ms");
                _output.WriteLine($"  Speedup:   {speedup:F2}x");
                _output.WriteLine($"  GFLOPS:    {gflops:F2}");
                _output.WriteLine("");

                // Verify correctness
                Assert.True(Math.Abs(resultStandard - resultSIMD) / Math.Max(Math.Abs(resultStandard), 1) < 1e-10);
            }
        }

        [Fact]
        public void BenchmarkFMA_StandardVsSIMD()
        {
            _output.WriteLine("=== FMA (Fused Multiply-Add) Benchmark ===\n");

            int[] sizes = { 10000, 100000, 1000000 };
            int warmupIterations = 10;
            int benchIterations = 100;

            foreach (int size in sizes)
            {
                _output.WriteLine($"Array Size: {size:N0} elements");

                var rng = new Random(42);
                var a = new double[size];
                var b = new double[size];
                var c = new double[size];
                var resultStandard = new double[size];
                var resultSIMD = new double[size];

                for (int i = 0; i < size; i++)
                {
                    a[i] = rng.NextDouble();
                    b[i] = rng.NextDouble();
                    c[i] = rng.NextDouble();
                }

                // Warmup
                for (int i = 0; i < warmupIterations; i++)
                {
                    StandardFMA(a, b, c, resultStandard);
                    CpuTensorPrimitives.FusedMultiplyAdd(a, b, c, resultSIMD);
                }

                // Benchmark standard
                var swStandard = Stopwatch.StartNew();
                for (int i = 0; i < benchIterations; i++)
                    StandardFMA(a, b, c, resultStandard);
                swStandard.Stop();

                // Benchmark SIMD
                var swSIMD = Stopwatch.StartNew();
                for (int i = 0; i < benchIterations; i++)
                    CpuTensorPrimitives.FusedMultiplyAdd(a, b, c, resultSIMD);
                swSIMD.Stop();

                double standardMs = swStandard.Elapsed.TotalMilliseconds / benchIterations;
                double simdMs = swSIMD.Elapsed.TotalMilliseconds / benchIterations;
                double speedup = standardMs / simdMs;

                // FLOPS: 2 ops per element (multiply + add fused)
                double gflops = (size * 2.0 / (simdMs / 1000.0)) / 1e9;

                _output.WriteLine($"  Standard:  {standardMs:F4} ms");
                _output.WriteLine($"  SIMD:      {simdMs:F4} ms");
                _output.WriteLine($"  Speedup:   {speedup:F2}x");
                _output.WriteLine($"  GFLOPS:    {gflops:F2}");
                _output.WriteLine("");
            }
        }

        [Fact]
        public void BenchmarkMatMul_NaiveVsBlocked()
        {
            _output.WriteLine("=== Matrix Multiply Benchmark: Naive vs Cache-Blocked ===\n");

            int[] sizes = { 128, 256, 512 };
            int warmupIterations = 2;
            int benchIterations = 5;

            foreach (int size in sizes)
            {
                _output.WriteLine($"Matrix Size: {size}x{size}");

                var rng = new Random(42);
                var a = new double[size * size];
                var b = new double[size * size];
                var resultNaive = new double[size * size];
                var resultBlocked = new double[size * size];

                for (int i = 0; i < a.Length; i++)
                {
                    a[i] = rng.NextDouble();
                    b[i] = rng.NextDouble();
                }

                // Warmup
                for (int i = 0; i < warmupIterations; i++)
                {
                    NaiveMatMul(a, b, resultNaive, size);
                    BlockedMatMul(a, b, resultBlocked, size);
                }

                // Benchmark naive
                var swNaive = Stopwatch.StartNew();
                for (int i = 0; i < benchIterations; i++)
                    NaiveMatMul(a, b, resultNaive, size);
                swNaive.Stop();

                // Benchmark blocked
                var swBlocked = Stopwatch.StartNew();
                for (int i = 0; i < benchIterations; i++)
                    BlockedMatMul(a, b, resultBlocked, size);
                swBlocked.Stop();

                double naiveMs = swNaive.Elapsed.TotalMilliseconds / benchIterations;
                double blockedMs = swBlocked.Elapsed.TotalMilliseconds / benchIterations;
                double speedup = naiveMs / blockedMs;

                // FLOPS: 2*n^3 for matrix multiply
                double flops = 2.0 * size * size * size;
                double naiveGflops = (flops / (naiveMs / 1000.0)) / 1e9;
                double blockedGflops = (flops / (blockedMs / 1000.0)) / 1e9;

                _output.WriteLine($"  Naive:    {naiveMs:F2} ms ({naiveGflops:F2} GFLOPS)");
                _output.WriteLine($"  Blocked:  {blockedMs:F2} ms ({blockedGflops:F2} GFLOPS)");
                _output.WriteLine($"  Speedup:  {speedup:F2}x");
                _output.WriteLine("");
            }
        }

        [Fact]
        public void PrintCpuCapabilities()
        {
            _output.WriteLine("=== NSL CPU Capabilities Report ===\n");

            var features = CpuInfo.Features;

            _output.WriteLine("SIMD Extensions:");
            _output.WriteLine($"  SSE:      {features.HasSse}");
            _output.WriteLine($"  SSE2:     {features.HasSse2}");
            _output.WriteLine($"  SSE3:     {features.HasSse3}");
            _output.WriteLine($"  SSSE3:    {features.HasSsse3}");
            _output.WriteLine($"  SSE4.1:   {features.HasSse41}");
            _output.WriteLine($"  SSE4.2:   {features.HasSse42}");
            _output.WriteLine($"  AVX:      {features.HasAvx}");
            _output.WriteLine($"  AVX2:     {features.HasAvx2}");
            _output.WriteLine($"  FMA:      {features.HasFma}");
            _output.WriteLine($"  AVX-512F: {features.HasAvx512F}");
            _output.WriteLine("");

            _output.WriteLine("CPU Info:");
            _output.WriteLine($"  Processor Count: {features.ProcessorCount}");
            _output.WriteLine($"  Best Vector Size: {features.BestVectorSize} bytes ({features.DoublesPerVector} doubles)");
            _output.WriteLine($"  Cache Line Size: {features.CacheLineSize} bytes");
            _output.WriteLine("");

            _output.WriteLine("Thermal Awareness:");
            _output.WriteLine($"  {Avx512ThermalAwareness.GetStatusDescription()}");
            _output.WriteLine($"  Should Use AVX-512: {Avx512ThermalAwareness.ShouldUseAvx512}");
        }

        #region Helper Methods

        private static void StandardAdd(double[] a, double[] b, double[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] + b[i];
        }

        private static double StandardDot(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
                sum += a[i] * b[i];
            return sum;
        }

        private static void StandardFMA(double[] a, double[] b, double[] c, double[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] * b[i] + c[i];
        }

        private static void NaiveMatMul(double[] a, double[] b, double[] result, int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a[i * n + k] * b[k * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }
        }

        private static void BlockedMatMul(double[] a, double[] b, double[] result, int n)
        {
            const int blockSize = 32;

            // Clear result
            Array.Clear(result, 0, result.Length);

            for (int ii = 0; ii < n; ii += blockSize)
            {
                for (int jj = 0; jj < n; jj += blockSize)
                {
                    for (int kk = 0; kk < n; kk += blockSize)
                    {
                        int iMax = Math.Min(ii + blockSize, n);
                        int jMax = Math.Min(jj + blockSize, n);
                        int kMax = Math.Min(kk + blockSize, n);

                        for (int i = ii; i < iMax; i++)
                        {
                            for (int k = kk; k < kMax; k++)
                            {
                                double aik = a[i * n + k];
                                for (int j = jj; j < jMax; j++)
                                {
                                    result[i * n + j] += aik * b[k * n + j];
                                }
                            }
                        }
                    }
                }
            }
        }

        #endregion
    }
}
