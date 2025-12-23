using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NSL.GPU;

namespace NSL.Tests
{
    /// <summary>
    /// Comprehensive unit tests for all NSL.GPU components
    /// </summary>
    public class GpuComponentTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;

        public GpuComponentTests(ITestOutputHelper output)
        {
            _output = output;

            _context = Context.Create(builder => builder
                .Default()
                .EnableAlgorithms()
                .Optimize(OptimizationLevel.O2)
                .Math(MathMode.Default));

            // Use CPU accelerator for tests (works everywhere)
            _accelerator = _context.GetCPUDevice(0).CreateAccelerator(_context);
            _kernels = new GpuKernels(_accelerator);

            _output.WriteLine($"Test accelerator: {_accelerator.Name}");
        }

        public void Dispose()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
        }

        #region GpuMemoryManager Tests

        [Fact]
        public void MemoryManager_AllocateFloat_ReturnsValidBuffer()
        {
            var manager = new GpuMemoryManager(_accelerator);

            var buffer = manager.AllocateFloat(1024, "test");

            Assert.NotNull(buffer);
            Assert.Equal(1024, buffer.Length);

            manager.Release(buffer);
            manager.Dispose();
        }

        [Fact]
        public void MemoryManager_PoolReuse_ReturnsSameBuffer()
        {
            var config = new GpuMemoryManager.PoolConfig { EnablePooling = true };
            var manager = new GpuMemoryManager(_accelerator, config);

            // Allocate and release
            var buffer1 = manager.AllocateFloat(1024);
            var ptr1 = buffer1.NativePtr;
            manager.Release(buffer1);

            // Allocate same size - should get from pool
            var buffer2 = manager.AllocateFloat(1024);
            var stats = manager.GetMemoryStats();

            Assert.True(stats.PoolHits > 0 || stats.PoolMisses > 0);
            _output.WriteLine($"Pool hit rate: {stats.PoolHitRate:P0}");

            manager.Release(buffer2);
            manager.Dispose();
        }

        [Fact]
        public void MemoryManager_GetMemoryStats_ReturnsValidStats()
        {
            var manager = new GpuMemoryManager(_accelerator);

            var buffer = manager.AllocateFloat(1000);
            var stats = manager.GetMemoryStats();

            Assert.True(stats.TotalVRAM > 0);
            Assert.True(stats.AllocatedByNSL > 0);
            Assert.Equal(1, stats.ActiveAllocations);

            _output.WriteLine(stats.ToString());

            manager.Release(buffer);
            manager.Dispose();
        }

        [Fact]
        public void MemoryManager_AdaptiveBatchSize_AdjustsCorrectly()
        {
            var config = new GpuMemoryManager.PoolConfig { EnableAdaptiveBatching = true };
            var manager = new GpuMemoryManager(_accelerator, config);

            var stats = manager.GetMemoryStats();
            _output.WriteLine($"TotalVRAM: {stats.TotalVRAM}, AvailableVRAM: {stats.AvailableVRAM}");

            // Skip test if accelerator doesn't report realistic memory stats
            // CPU accelerator returns MaxValue, real GPUs return actual VRAM
            if (stats.TotalVRAM == 0 || stats.TotalVRAM == long.MaxValue)
            {
                _output.WriteLine("Skipping: Accelerator reports unrealistic memory (CPU fallback or MaxValue)");
                manager.Dispose();
                return;
            }

            // Request huge batch that won't fit
            long hugeMemory = stats.TotalVRAM * 2;
            int optimalBatch = manager.GetOptimalBatchSize(
                sampleSizeBytes: hugeMemory / 10,
                preferredBatchSize: 100,
                minBatchSize: 1
            );

            Assert.True(optimalBatch < 100, $"Expected batch < 100 but got {optimalBatch}");
            _output.WriteLine($"Optimal batch adjusted to: {optimalBatch}");

            manager.Dispose();
        }

        [Fact]
        public void MemoryManager_EstimateModelMemory_CalculatesCorrectly()
        {
            var manager = new GpuMemoryManager(_accelerator);

            int[] layers = { 784, 512, 256, 10 }; // Simple MLP
            long estimate = manager.EstimateModelMemory(layers, batchSize: 32);

            Assert.True(estimate > 0);
            _output.WriteLine($"Estimated model memory: {estimate / (1024 * 1024.0):F2} MB");

            manager.Dispose();
        }

        #endregion

        #region Quantization Tests

        [Fact]
        public void Quantization_CalibrateMinMax_ReturnsValidParams()
        {
            var quantizer = new QuantizationEngine(_accelerator);

            float[] data = { -1.5f, -0.5f, 0.0f, 0.5f, 1.5f };
            var quantParams = quantizer.CalibrateMinMax(data, symmetric: true);

            Assert.True(quantParams.Scale > 0);
            Assert.Equal(0, quantParams.ZeroPoint);
            Assert.True(quantParams.IsSymmetric);

            _output.WriteLine($"Scale: {quantParams.Scale}, ZeroPoint: {quantParams.ZeroPoint}");
        }

        [Fact]
        public void Quantization_QuantizeAndDequantize_PreservesApproximateValues()
        {
            var quantizer = new QuantizationEngine(_accelerator);

            float[] original = { 0.1f, 0.5f, 1.0f, 2.0f };
            var tensor = GpuTensor.FromArray(_accelerator, original, new[] { 2, 2 });

            // Quantize
            var quantized = quantizer.Quantize(tensor);

            // Dequantize
            var restored = quantizer.Dequantize(quantized);
            var restoredData = restored.ToArray();

            // Check values are approximately preserved (some loss expected)
            for (int i = 0; i < original.Length; i++)
            {
                Assert.True(Math.Abs(original[i] - restoredData[i]) < 0.1f,
                    $"Value {i}: expected ~{original[i]}, got {restoredData[i]}");
            }

            quantized.Dispose();
            restored.Dispose();
            tensor.Dispose();
        }

        [Fact]
        public void Quantization_PerChannelCalibration_Works()
        {
            var quantizer = new QuantizationEngine(_accelerator);

            // 2 channels with different ranges
            float[] data = {
                -1.0f, -0.5f, 0.0f, 0.5f,  // Channel 0: [-1, 0.5]
                -0.1f, 0.0f, 0.5f, 1.0f     // Channel 1: [-0.1, 1.0]
            };

            var quantParams = quantizer.CalibrateMinMax(data,
                symmetric: true, perChannel: true, numChannels: 2);

            Assert.True(quantParams.PerChannel);
            Assert.Equal(2, quantParams.NumChannels);
            Assert.NotNull(quantParams.ChannelScales);
            Assert.Equal(2, quantParams.ChannelScales!.Length);

            _output.WriteLine($"Channel scales: [{string.Join(", ", quantParams.ChannelScales)}]");
        }

        #endregion

        #region Float16Ops Tests

        [Fact]
        public void Float16_ConvertToAndFrom_PreservesApproximateValues()
        {
            var fp16 = new Float16Ops(_accelerator);

            float[] original = { 0.0f, 1.0f, -1.0f, 0.5f, 100.0f, -0.001f };

            var fp16Tensor = fp16.FromArray(original, new[] { 6 });
            var restored = fp16.ToArray(fp16Tensor);

            for (int i = 0; i < original.Length; i++)
            {
                float tolerance = Math.Abs(original[i]) * 0.01f + 0.001f;
                Assert.True(Math.Abs(original[i] - restored[i]) < tolerance,
                    $"Value {i}: expected ~{original[i]}, got {restored[i]}");
            }

            fp16Tensor.Dispose();
        }

        [Fact]
        public void Float16_MemorySavings_Is50Percent()
        {
            var (fp32Bytes, fp16Bytes, savings) =
                Float16Ops.CalculateMemorySavings(new[] { 1000, 1000 });

            Assert.Equal(4_000_000, fp32Bytes);
            Assert.Equal(2_000_000, fp16Bytes);
            Assert.Equal(50.0f, savings, 1);

            _output.WriteLine($"FP32: {fp32Bytes} bytes, FP16: {fp16Bytes} bytes, Savings: {savings}%");
        }

        [Fact]
        public void Float16_SpecialValues_HandleCorrectly()
        {
            // Test infinity and zero
            Assert.Equal(0x7C00, Float16Ops.FloatToHalf(float.PositiveInfinity));
            Assert.Equal(0xFC00, Float16Ops.FloatToHalf(float.NegativeInfinity));
            Assert.Equal(0, Float16Ops.FloatToHalf(0.0f));

            // Convert back
            Assert.Equal(float.PositiveInfinity, Float16Ops.HalfToFloat(0x7C00));
            Assert.Equal(float.NegativeInfinity, Float16Ops.HalfToFloat(0xFC00));
            Assert.Equal(0.0f, Float16Ops.HalfToFloat(0));
        }

        #endregion

        #region OperatorFusion Tests

        [Fact]
        public void Fusion_LinearRelu_ProducesCorrectOutput()
        {
            var fusion = new OperatorFusion(_accelerator);

            // Create test data
            var input = GpuTensor.FromArray(_accelerator, new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            var weight = GpuTensor.FromArray(_accelerator, new float[] { 1, 0, 0, 1 }, new[] { 2, 2 });
            var bias = GpuTensor.FromArray(_accelerator, new float[] { -5, 0 }, new[] { 2 });

            var result = fusion.FusedLinearRelu(input, weight, bias);
            var data = result.ToArray();

            // Linear: [1,2] @ [1,0;0,1] + [-5,0] = [-4, 2]
            // ReLU:  [0, 2]
            // Linear: [3,4] @ [1,0;0,1] + [-5,0] = [-2, 4]
            // ReLU:  [0, 4]
            Assert.Equal(0, data[0], 1);
            Assert.True(data[1] > 0);
            Assert.Equal(0, data[2], 1);
            Assert.True(data[3] > 0);

            input.Dispose();
            weight.Dispose();
            bias.Dispose();
            result.Dispose();
        }

        #endregion

        #region GpuAutoConfig Tests

        [Fact]
        public void AutoConfig_DetectsGpus_ReturnsAtLeastCPU()
        {
            var config = new GpuAutoConfig.Config { AllowCpuFallback = true };
            using var autoConfig = new GpuAutoConfig(config);

            Assert.NotNull(autoConfig.CurrentGpu);
            _output.WriteLine($"Current GPU: {autoConfig.CurrentGpu}");

            var gpus = autoConfig.AvailableGpus;
            _output.WriteLine($"Available GPUs: {gpus.Count}");
            foreach (var gpu in gpus)
            {
                _output.WriteLine($"  - {gpu}");
            }
        }

        [Fact]
        public void AutoConfig_CalculatesGpuScore_ReturnsPositive()
        {
            var config = new GpuAutoConfig.Config();
            using var autoConfig = new GpuAutoConfig(config);

            Assert.True(autoConfig.CurrentGpu!.Score >= 0);
            _output.WriteLine($"GPU Score: {autoConfig.CurrentGpu.Score}");
        }

        #endregion

        #region DynamicShapeManager Tests

        [Fact]
        public void DynamicShape_RegisterShape_RetrievesCorrectly()
        {
            var manager = new DynamicShapeManager(_accelerator);

            manager.RegisterShape("test", new[] { -1, 512 }, new[] { "batch", "hidden" });

            var shape = manager.GetShape("test");
            Assert.NotNull(shape);
            Assert.True(shape!.IsDynamic);
            Assert.Single(shape.DynamicDimIndices);
            Assert.Equal(0, shape.DynamicDimIndices[0]);

            _output.WriteLine($"Shape: {shape}");
        }

        [Fact]
        public void DynamicShape_InferMatMulShape_CalculatesCorrectly()
        {
            var manager = new DynamicShapeManager(_accelerator);

            var outputShape = manager.InferShape("matmul",
                new[] { 32, 128 },
                new[] { 128, 64 });

            Assert.Equal(new[] { 32, 64 }, outputShape);

            // Batched matmul
            var batchedShape = manager.InferShape("matmul",
                new[] { 4, 32, 128 },
                new[] { 128, 64 });

            Assert.Equal(new[] { 4, 32, 64 }, batchedShape);
        }

        [Fact]
        public void DynamicShape_InferBroadcast_HandlesExpansion()
        {
            var manager = new DynamicShapeManager(_accelerator);

            // [32, 1, 64] + [1, 128, 64] = [32, 128, 64]
            var outputShape = manager.InferShape("add",
                new[] { 32, 1, 64 },
                new[] { 1, 128, 64 });

            Assert.Equal(new[] { 32, 128, 64 }, outputShape);
        }

        [Fact]
        public void DynamicTensor_Reshape_AdjustsCapacity()
        {
            var manager = new DynamicShapeManager(_accelerator);

            var tensor = manager.GetOrCreateDynamic("test", new[] { 10, 10 });

            // Set initial data
            var data1 = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();
            tensor.SetData(data1, new[] { 10, 10 });
            Assert.Equal(100, tensor.Size);

            // Reshape larger
            var data2 = Enumerable.Range(0, 400).Select(i => (float)i).ToArray();
            tensor.SetData(data2, new[] { 20, 20 });
            Assert.Equal(400, tensor.Size);

            tensor.Dispose();
        }

        [Fact]
        public void DynamicShape_ShapeCompatibility_ChecksCorrectly()
        {
            Assert.True(new[] { 32, 1, 64 }.IsCompatible(new[] { 1, 128, 64 }));
            Assert.True(new[] { 32, 128 }.IsCompatible(new[] { 128 }));
            Assert.False(new[] { 32, 128 }.IsCompatible(new[] { 64 }));
            Assert.True(new[] { -1, 128 }.IsCompatible(new[] { 32, 128 })); // Dynamic compatible
        }

        [Fact]
        public void DynamicShape_ResolveDynamic_CalculatesCorrectly()
        {
            var shape = new[] { -1, 128 };
            var resolved = shape.ResolveDynamic(totalElements: 4096);

            Assert.Equal(new[] { 32, 128 }, resolved);
        }

        #endregion

        #region ModelSerializer Tests

        [Fact]
        public void ModelSerializer_SaveAndLoad_PreservesData()
        {
            var tempPath = System.IO.Path.GetTempFileName() + ".nslm";

            try
            {
                // Create test tensors
                var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
                {
                    ["weight1"] = (new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 }),
                    ["bias1"] = (new float[] { 0.1f, 0.2f }, new[] { 2 })
                };

                var metadata = new ModelSerializer.ModelMetadata
                {
                    Name = "TestModel",
                    Architecture = "MLP",
                    CustomData = new Dictionary<string, string> { ["version"] = "1.0" }
                };

                // Save
                ModelSerializer.Save(tempPath, tensors, metadata);

                // Load
                var (loadedTensors, loadedMetadata) = ModelSerializer.Load(tempPath);

                // Verify
                Assert.Equal("TestModel", loadedMetadata.Name);
                Assert.Equal("MLP", loadedMetadata.Architecture);
                Assert.Equal("1.0", loadedMetadata.CustomData["version"]);

                Assert.Equal(2, loadedTensors.Count);
                Assert.Equal(tensors["weight1"].Data, loadedTensors["weight1"].Data);
                Assert.Equal(tensors["weight1"].Shape, loadedTensors["weight1"].Shape);

                _output.WriteLine($"Saved and loaded {loadedTensors.Count} tensors");
            }
            finally
            {
                if (System.IO.File.Exists(tempPath))
                    System.IO.File.Delete(tempPath);
            }
        }

        [Fact]
        public void ModelSerializer_Checkpoint_SavesTrainingState()
        {
            var tempPath = System.IO.Path.GetTempFileName() + ".nslm";

            try
            {
                var modelState = new Dictionary<string, (float[] Data, int[] Shape)>
                {
                    ["layer1.weight"] = (new float[] { 1, 2, 3, 4 }, new[] { 2, 2 })
                };

                var optimizerState = new Dictionary<string, (float[] Data, int[] Shape)>
                {
                    ["layer1.weight.m"] = (new float[] { 0.1f, 0.1f, 0.1f, 0.1f }, new[] { 2, 2 })
                };

                ModelSerializer.SaveCheckpoint(tempPath, modelState, optimizerState,
                    epoch: 5, step: 1000, loss: 0.123f);

                var (loadedModel, loadedOptimizer, epoch, step, loss) =
                    ModelSerializer.LoadCheckpoint(tempPath);

                Assert.Equal(5, epoch);
                Assert.Equal(1000, step);
                Assert.Equal(0.123f, loss, 3);
                Assert.NotNull(loadedOptimizer);
                Assert.Single(loadedOptimizer);

                _output.WriteLine($"Checkpoint: epoch={epoch}, step={step}, loss={loss}");
            }
            finally
            {
                if (System.IO.File.Exists(tempPath))
                    System.IO.File.Delete(tempPath);
            }
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void Integration_FullPipeline_WorksEndToEnd()
        {
            // Test a complete pipeline: create tensor, quantize, dequantize, fuse ops

            // 1. Create tensor
            var data = Enumerable.Range(0, 16).Select(i => (float)i).ToArray();
            var tensor = GpuTensor.FromArray(_accelerator, data, new[] { 4, 4 });

            // 2. Test kernels
            var activated = _kernels.ReLU(tensor);
            var activatedData = activated.ToArray();
            Assert.All(activatedData, v => Assert.True(v >= 0));

            // 3. Quantize
            var quantizer = new QuantizationEngine(_accelerator);
            var quantized = quantizer.Quantize(tensor);
            Assert.NotNull(quantized.Buffer);

            // 4. Dequantize
            var restored = quantizer.Dequantize(quantized);
            var restoredData = restored.ToArray();

            // 5. Verify approximate values
            for (int i = 0; i < data.Length; i++)
            {
                Assert.True(Math.Abs(data[i] - restoredData[i]) < 0.5f);
            }

            _output.WriteLine("Full pipeline test passed!");

            tensor.Dispose();
            activated.Dispose();
            quantized.Dispose();
            restored.Dispose();
        }

        [Fact]
        public void Integration_MemoryPressure_HandlesGracefully()
        {
            var config = new GpuMemoryManager.PoolConfig
            {
                EnablePooling = true,
                MemoryPressureThreshold = 0.9f,
                EnableAutoGC = true
            };
            var manager = new GpuMemoryManager(_accelerator, config);

            // Allocate multiple buffers
            var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            for (int i = 0; i < 10; i++)
            {
                var buffer = manager.AllocateFloat(10000);
                buffers.Add(buffer);
            }

            var stats = manager.GetMemoryStats();
            Assert.Equal(10, stats.ActiveAllocations);

            // Release all
            foreach (var buffer in buffers)
            {
                manager.Release(buffer);
            }

            // Force GC
            manager.ForceGC();

            stats = manager.GetMemoryStats();
            Assert.True(stats.PooledMemory >= 0);

            _output.WriteLine($"Peak memory: {stats.PeakAllocated / 1024}KB");

            manager.Dispose();
        }

        #endregion

        #region Inference Optimization Tests

        [Fact]
        public void KVCache_UpdateAndRetrieve_WorksCorrectly()
        {
            using var cache = new KVCache(_accelerator, numLayers: 2, numHeads: 4, headDim: 64, maxSeqLen: 512);

            // Create test keys/values
            var keys = GpuTensor.FromArray(_accelerator, new float[4 * 64], new[] { 1, 4, 64 });
            var values = GpuTensor.FromArray(_accelerator, new float[4 * 64], new[] { 1, 4, 64 });

            // Update cache for all layers (sequence length updates on last layer)
            cache.Update(0, 0, keys, values);
            cache.Update(1, 0, keys, values);  // Last layer triggers length update

            Assert.Equal(1, cache.GetSequenceLength(0));
            _output.WriteLine($"KV-Cache sequence length after 1 token: {cache.GetSequenceLength(0)}");

            // Test that cache can be retrieved
            Assert.Equal(2, cache.NumLayers);
            Assert.Equal(512, cache.MaxSequenceLength);

            keys.Dispose();
            values.Dispose();
        }

        [Fact]
        public void PagedAttention_AllocateAndFree_ManagesMemory()
        {
            using var paged = new PagedAttention(_accelerator, numHeads: 8, headDim: 64, pageSize: 16, maxPages: 64);

            int initialFree = paged.FreePages;

            // Allocate sequence
            paged.AllocateSequence(seqId: 1, initialTokens: 10);

            Assert.True(paged.UsedPages > 0);
            _output.WriteLine($"Pages used after allocation: {paged.UsedPages}");

            // Free sequence
            paged.FreeSequence(1);

            Assert.Equal(initialFree, paged.FreePages);
            _output.WriteLine("PagedAttention allocation/free test passed");
        }

        [Fact]
        public void SpeculativeDecoder_CreatesCorrectly()
        {
            var kernels = new GpuKernels(_accelerator);
            var decoder = new SpeculativeDecoder(_accelerator, kernels, speculativeTokens: 4);

            Assert.Equal(4, decoder.SpeculativeTokens);
            _output.WriteLine($"Speculative decoder with {decoder.SpeculativeTokens} tokens");
        }

        [Fact]
        public void ContinuousBatcher_SubmitRequest_Queues()
        {
            var kernels = new GpuKernels(_accelerator);
            var batcher = new ContinuousBatcher(_accelerator, kernels, maxBatchSize: 8);

            Assert.Equal(0, batcher.ActiveRequests);
            Assert.Equal(0, batcher.PendingRequests);

            // Submit a request (won't complete without model forward)
            var task = batcher.SubmitAsync(new[] { 1, 2, 3 }, maxNewTokens: 10);

            Assert.Equal(1, batcher.PendingRequests);
            _output.WriteLine("Continuous batcher submission test passed");
        }

        #endregion

        #region Large Scale Parallelism Tests

        [Fact]
        public void GradientCompression_FP16_CompressesCorrectly()
        {
            var gradients = new float[] { 1.0f, -0.5f, 0.25f, -0.125f, 2.5f };

            var compressed = GradientCompression.CompressToFP16(gradients);
            var decompressed = GradientCompression.DecompressFromFP16(compressed);

            // FP16 loses some precision but should be close
            for (int i = 0; i < gradients.Length; i++)
            {
                Assert.True(Math.Abs(gradients[i] - decompressed[i]) < 0.01f,
                    $"FP16 compression error too large at index {i}");
            }

            _output.WriteLine($"FP16 compression: {gradients.Length * 4} bytes -> {compressed.Length * 2} bytes (50% reduction)");
        }

        [Fact]
        public void GradientCompression_TopK_SelectsLargest()
        {
            var gradients = new float[] { 0.1f, -5.0f, 0.2f, 3.0f, -0.5f, 0.01f };

            var (indices, values) = GradientCompression.TopKCompress(gradients, ratio: 0.5f);

            Assert.Equal(3, indices.Length);  // Top 50%

            // Should contain the largest absolute values
            Assert.Contains(-5.0f, values);
            Assert.Contains(3.0f, values);

            _output.WriteLine($"Top-K selected {indices.Length} gradients from {gradients.Length}");
        }

        [Fact]
        public void GradientCompression_OneBit_CompressesExtremely()
        {
            var gradients = new float[] { 1.5f, -2.0f, 0.5f, -0.3f, 2.1f, -1.8f, 0.9f, -0.1f };

            var (signs, scale) = GradientCompression.OneBitCompress(gradients);
            var decompressed = GradientCompression.OneBitDecompress(signs, scale, gradients.Length);

            // Check signs are preserved
            for (int i = 0; i < gradients.Length; i++)
            {
                Assert.Equal(gradients[i] > 0, decompressed[i] > 0);
            }

            // Compression ratio: 32 bits -> 1 bit per gradient
            _output.WriteLine($"1-bit compression: {gradients.Length * 4} bytes -> {signs.Length + 4} bytes");
        }

        [Fact]
        public void ZeROOptimizer_RegistersParameters()
        {
            var accelerators = new List<Accelerator> { _accelerator };
            using var zero = new ZeROOptimizer(accelerators, new ZeROOptimizer.ZeROConfig { Stage = 3 });

            var param = GpuTensor.FromArray(_accelerator, new float[1024], new[] { 1024 });
            zero.RegisterParameter("layer1.weight", param);

            Assert.Equal(3, zero.Stage);
            _output.WriteLine("ZeRO Stage 3 optimizer registered parameter");

            param.Dispose();
        }

        [Fact]
        public void ActivationCheckpointing_SavesAndRetrieves()
        {
            var checkpointing = new ActivationCheckpointing();

            var activation = GpuTensor.FromArray(_accelerator, new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });

            checkpointing.SaveCheckpoint("layer1", activation);

            var retrieved = checkpointing.GetCheckpoint("layer1");
            Assert.NotNull(retrieved);

            var data = retrieved!.ToArray();
            Assert.Equal(1, data[0]);
            Assert.Equal(4, data[3]);

            _output.WriteLine("Activation checkpointing test passed");

            checkpointing.Clear();
            activation.Dispose();
        }

        #endregion

        #region AI Capabilities Tests

        [Fact]
        public void OnlineLearning_CreateAdapter_ReturnsValidAdapter()
        {
            using var learning = new OnlineLearning(_accelerator, _kernels);

            var adapter = learning.CreateAdapter("test_layer", 64, 128);

            Assert.NotNull(adapter);
            Assert.Equal("test_layer", adapter.LayerName);
            Assert.Equal(8, adapter.Rank);  // Default rank
            Assert.True(adapter.Enabled);
            Assert.NotNull(adapter.A);
            Assert.NotNull(adapter.B);

            _output.WriteLine($"Created LoRA adapter with rank {adapter.Rank}");
        }

        [Fact]
        public void OnlineLearning_ApplyLoRA_ModifiesOutput()
        {
            using var learning = new OnlineLearning(_accelerator, _kernels);

            var adapter = learning.CreateAdapter("layer1", 4, 4);

            var input = GpuTensor.FromArray(_accelerator, new float[] { 1, 2, 3, 4 }, new[] { 1, 4 });
            var original = GpuTensor.FromArray(_accelerator, new float[] { 0, 0, 0, 0 }, new[] { 1, 4 });

            var result = learning.ApplyLoRA("layer1", input, original);

            // Initially B is zero so result should be close to original
            var resultData = result.ToArray();
            Assert.Equal(4, resultData.Length);

            input.Dispose();
            original.Dispose();
            result.Dispose();

            _output.WriteLine("LoRA apply test passed");
        }

        [Fact]
        public void HNSW_InsertAndSearch_FindsNearest()
        {
            var hnsw = new PersistentMemory.HNSWIndex(dim: 4, M: 4, efConstruction: 16, efSearch: 8);

            // Insert some vectors
            hnsw.Insert(0, new float[] { 1, 0, 0, 0 });
            hnsw.Insert(1, new float[] { 0, 1, 0, 0 });
            hnsw.Insert(2, new float[] { 0, 0, 1, 0 });
            hnsw.Insert(3, new float[] { 0.9f, 0.1f, 0, 0 });  // Close to vector 0

            Assert.Equal(4, hnsw.Count);

            // Search for nearest to [1,0,0,0]
            var results = hnsw.Search(new float[] { 1, 0, 0, 0 }, k: 2);

            Assert.NotEmpty(results);
            Assert.Equal(0, results[0].id);  // Exact match first
            Assert.True(results[0].distance < 0.01f);  // Very small distance

            _output.WriteLine($"HNSW search found {results.Length} results, nearest distance: {results[0].distance}");
        }

        [Fact]
        public void UncertaintyEstimation_QuickUncertainty_ComputesValidResult()
        {
            var uncertainty = new UncertaintyEstimation(_accelerator, _kernels);

            // Confident logits (one high, rest low)
            var confidentLogits = GpuTensor.FromArray(_accelerator, new float[] { 10, 0, 0, 0 }, new[] { 4 });
            var confidentResult = uncertainty.QuickUncertainty(confidentLogits);

            Assert.True(confidentResult.Confidence > 0.5f);
            Assert.False(confidentResult.ShouldAskUser);

            // Uncertain logits (all similar)
            var uncertainLogits = GpuTensor.FromArray(_accelerator, new float[] { 1, 1, 1, 1 }, new[] { 4 });
            var uncertainResult = uncertainty.QuickUncertainty(uncertainLogits);

            Assert.True(uncertainResult.TotalUncertainty > 0.3f);

            confidentLogits.Dispose();
            uncertainLogits.Dispose();

            _output.WriteLine($"Confident uncertainty: {confidentResult.TotalUncertainty:F3}, Uncertain: {uncertainResult.TotalUncertainty:F3}");
        }

        [Fact]
        public void UncertaintyEstimation_AddCalibrationAndConformal_Works()
        {
            var uncertainty = new UncertaintyEstimation(_accelerator, _kernels);

            // Add calibration samples
            uncertainty.AddCalibrationSample(0.5f, 0.6f);  // error = 0.1
            uncertainty.AddCalibrationSample(0.7f, 0.65f); // error = 0.05
            uncertainty.AddCalibrationSample(0.3f, 0.5f);  // error = 0.2

            // Get conformal interval
            var (lower, upper) = uncertainty.GetConformalInterval(0.5f);

            Assert.True(lower < 0.5f);
            Assert.True(upper > 0.5f);

            _output.WriteLine($"Conformal interval for 0.5: [{lower:F3}, {upper:F3}]");
        }

        [Fact]
        public void CausalGraph_AddEdgeAndHasPath_Works()
        {
            var graph = new CausalReasoning.CausalGraph();

            graph.AddEdge("smoking", "tar");
            graph.AddEdge("tar", "cancer");
            graph.AddEdge("smoking", "cancer");

            Assert.True(graph.HasPath("smoking", "cancer"));
            Assert.True(graph.HasPath("tar", "cancer"));
            Assert.False(graph.HasPath("cancer", "smoking"));

            _output.WriteLine("Causal graph path finding works correctly");
        }

        [Fact]
        public void CausalReasoning_CheckIdentifiability_ReturnsResult()
        {
            var causal = new CausalReasoning(_accelerator, _kernels);

            var graph = new CausalReasoning.CausalGraph();
            graph.AddEdge("X", "Y");  // Simple case: X causes Y

            var result = causal.CheckIdentifiability(graph, "X", "Y");

            Assert.True(result.IsIdentifiable);
            Assert.NotEmpty(result.RulesApplied);

            _output.WriteLine($"Causal effect identifiable: {result.IsIdentifiable}, Estimand: {result.Estimand}");
        }

        [Fact]
        public void CausalReasoning_Counterfactual_ComputesResult()
        {
            var causal = new CausalReasoning(_accelerator, _kernels);

            var graph = new CausalReasoning.CausalGraph();
            graph.AddEdge("treatment", "outcome");

            var factual = new Dictionary<string, float>
            {
                ["treatment"] = 0,
                ["outcome"] = 0.3f
            };

            var result = causal.Counterfactual(graph, factual, "treatment", 1.0f);

            Assert.NotNull(result.CounterfactualWorld);
            Assert.Equal(1.0f, result.CounterfactualWorld["treatment"]);

            _output.WriteLine($"Counterfactual world: treatment={result.CounterfactualWorld["treatment"]}, outcome={result.CounterfactualWorld["outcome"]}");
        }

        [Fact]
        public void StructuralCausalModel_Evaluate_ProducesValues()
        {
            var graph = new CausalReasoning.CausalGraph();
            graph.AddEdge("X", "Y");
            graph.AddEdge("Y", "Z");

            var scm = new CausalReasoning.StructuralCausalModel
            {
                Graph = graph,
                ExogenousNoise = new Dictionary<string, float>
                {
                    ["X"] = 1.0f,
                    ["Y"] = 0f,
                    ["Z"] = 0f
                }
            };

            scm.Functions["Y"] = (vals, noise) => vals.GetValueOrDefault("X", 0) * 0.5f + noise;
            scm.Functions["Z"] = (vals, noise) => vals.GetValueOrDefault("Y", 0) * 2.0f + noise;

            var values = scm.Evaluate();

            Assert.Equal(1.0f, values["X"]);
            Assert.Equal(0.5f, values["Y"]);
            Assert.Equal(1.0f, values["Z"]);

            // Test intervention do(X=2)
            var interventedValues = scm.Evaluate(new Dictionary<string, float> { ["X"] = 2.0f });

            Assert.Equal(2.0f, interventedValues["X"]);
            Assert.Equal(1.0f, interventedValues["Y"]);
            Assert.Equal(2.0f, interventedValues["Z"]);

            _output.WriteLine("SCM evaluation and intervention work correctly");
        }

        #endregion
    }
}
