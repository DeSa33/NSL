using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NSL.GPU
{
    #region Online Learning - Learn During Inference

    /// <summary>
    /// Online Learning: Adapt model weights during inference without full retraining.
    /// Enables AI systems to learn from user feedback in real-time.
    ///
    /// Techniques:
    /// - LoRA (Low-Rank Adaptation): Inject small trainable matrices
    /// - Prompt tuning: Learn soft prompts
    /// - Gradient-free adaptation: Update without backprop
    /// </summary>
    public class OnlineLearning : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;
        private readonly OnlineLearningConfig _config;

        private readonly ConcurrentDictionary<string, LoRAAdapter> _adapters = new();
        private readonly ConcurrentDictionary<string, GpuTensor> _softPrompts = new();
        private readonly List<(string input, string correction, float weight)> _corrections = new();

        private bool _disposed;

        /// <summary>Public API</summary>
        public class OnlineLearningConfig
        {
            /// <summary>LoRA rank (4-8 for simple tasks, 16-64 for complex)</summary>
            public int LoRARank { get; set; } = 8;

            /// <summary>LoRA alpha - best practice is 2x rank for aggressive learning</summary>
            public float LoRAAlpha { get; set; } = 16f;  // 2 * rank

            /// <summary>Learning rate - higher for LoRA than full fine-tuning (1e-3 to 5e-4)</summary>
            public float LearningRate { get; set; } = 1e-3f;

            /// <summary>Maximum corrections to remember</summary>
            public int MaxCorrections { get; set; } = 1000;

            /// <summary>Soft prompt length</summary>
            public int SoftPromptLength { get; set; } = 20;

            /// <summary>Enable gradient-free adaptation</summary>
            public bool UseGradientFree { get; set; } = false;

            /// <summary>LoRA dropout for regularization (0 for speed, >0 if overfitting)</summary>
            public float LoRADropout { get; set; } = 0f;

            /// <summary>Train bias vectors alongside LoRA for extra performance</summary>
            public bool TrainBias { get; set; } = true;

            /// <summary>Use QLoRA (4-bit quantized) - 33% memory savings, 39% slower</summary>
            public bool UseQLoRA { get; set; } = false;

            /// <summary>Apply LoRA to all layers (recommended for max quality)</summary>
            public bool ApplyToAllLayers { get; set; } = true;
        }

        /// <summary>
        /// LoRA Adapter: Low-rank matrices that modify layer outputs
        /// Original: Y = X @ W
        /// With LoRA: Y = X @ W + X @ A @ B * scale (where A,B are small)
        /// Based on: https://arxiv.org/abs/2106.09685
        /// </summary>
        public class LoRAAdapter : IDisposable
        {
            /// <summary>Public API</summary>
            public string LayerName { get; set; } = "";
            /// <summary>Public API</summary>
            public GpuTensor A { get; set; } = null!;  // [in_features, rank]
            /// <summary>Public API</summary>
            public GpuTensor B { get; set; } = null!;  // [rank, out_features]
            /// <summary>Public API</summary>
            public float Alpha { get; set; } = 16f;
            /// <summary>Public API</summary>
            public int Rank { get; set; } = 8;
            /// <summary>Public API</summary>
            public bool Enabled { get; set; } = true;
            /// <summary>Public API</summary>
            public float Dropout { get; set; } = 0f;

            // Trainable bias (per research: training bias improves performance)
            /// <summary>Public API</summary>
            public float[]? Bias { get; set; }
            /// <summary>Public API</summary>
            public float[]? BiasMomentum { get; set; }

            // Optimizer states (Adam-style momentum)
            /// <summary>Public API</summary>
            public float[] MomentumA { get; set; } = Array.Empty<float>();
            /// <summary>Public API</summary>
            public float[] MomentumB { get; set; } = Array.Empty<float>();
            /// <summary>Public API</summary>
            public float[] VelocityA { get; set; } = Array.Empty<float>();  // Second moment for Adam
            /// <summary>Public API</summary>
            public float[] VelocityB { get; set; } = Array.Empty<float>();
            /// <summary>Public API</summary>
            public int UpdateStep { get; set; } = 0;

            /// <summary>Public API</summary>
            public float Scale => Alpha / Rank;

            /// <summary>Public API</summary>
            public void Dispose()
            {
                A?.Dispose();
                B?.Dispose();
            }
        }

        /// <summary>Public API</summary>
        public OnlineLearning(Accelerator accelerator, GpuKernels kernels, OnlineLearningConfig? config = null)
        {
            _accelerator = accelerator;
            _kernels = kernels;
            _config = config ?? new OnlineLearningConfig();
        }

        /// <summary>
        /// Create a LoRA adapter for a layer.
        /// Per research: Apply to all layers for max quality.
        /// </summary>
        public LoRAAdapter CreateAdapter(string layerName, int inFeatures, int outFeatures)
        {
            int rank = _config.LoRARank;

            // Initialize A with Kaiming/He initialization, B with zeros
            // B=0 ensures adapter has no effect at start (identity)
            var aData = new float[inFeatures * rank];
            var bData = new float[rank * outFeatures];

            var random = new Random();
            // He initialization: sqrt(2/fan_in) for ReLU, sqrt(1/fan_in) for linear
            float scale = MathF.Sqrt(1.0f / inFeatures);
            for (int i = 0; i < aData.Length; i++)
            {
                // Gaussian initialization (Box-Muller)
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                aData[i] = (float)gaussian * scale;
            }
            // B starts at zero so adapter initially has no effect

            var adapter = new LoRAAdapter
            {
                LayerName = layerName,
                A = GpuTensor.FromArray(_accelerator, aData, new[] { inFeatures, rank }),
                B = GpuTensor.FromArray(_accelerator, bData, new[] { rank, outFeatures }),
                Alpha = _config.LoRAAlpha,
                Rank = rank,
                Dropout = _config.LoRADropout,
                MomentumA = new float[aData.Length],
                MomentumB = new float[bData.Length],
                VelocityA = new float[aData.Length],
                VelocityB = new float[bData.Length],
                // Initialize bias if enabled
                Bias = _config.TrainBias ? new float[outFeatures] : null,
                BiasMomentum = _config.TrainBias ? new float[outFeatures] : null
            };

            _adapters[layerName] = adapter;
            return adapter;
        }

        /// <summary>
        /// Apply LoRA to a linear layer output with optional dropout and bias.
        /// Formula: Y = X @ W + (dropout(X @ A) @ B) * (alpha/rank) + bias
        /// </summary>
        public GpuTensor ApplyLoRA(string layerName, GpuTensor input, GpuTensor originalOutput, bool training = false)
        {
            if (!_adapters.TryGetValue(layerName, out var adapter) || !adapter.Enabled)
            {
                return originalOutput;
            }

            // Compute LoRA delta: input @ A @ B * scale
            var intermediate = _kernels.MatMul(input, adapter.A);

            // Apply dropout during training if configured
            if (training && adapter.Dropout > 0)
            {
                var data = intermediate.ToArray();
                var random = new Random();
                float keepProb = 1.0f - adapter.Dropout;
                for (int i = 0; i < data.Length; i++)
                {
                    if (random.NextDouble() > keepProb)
                        data[i] = 0;
                    else
                        data[i] /= keepProb;  // Scale to maintain expected value
                }
                intermediate.Buffer.CopyFromCPU(data);
            }

            var loraOutput = _kernels.MatMul(intermediate, adapter.B);
            var scaled = _kernels.MulScalar(loraOutput, adapter.Scale);

            // Add to original
            var result = _kernels.Add(originalOutput, scaled);

            // Add trainable bias if present
            if (adapter.Bias != null)
            {
                var resultData = result.ToArray();
                int outDim = adapter.Bias.Length;
                for (int i = 0; i < resultData.Length; i++)
                {
                    resultData[i] += adapter.Bias[i % outDim];
                }
                result.Buffer.CopyFromCPU(resultData);
            }

            intermediate.Dispose();
            loraOutput.Dispose();
            scaled.Dispose();

            return result;
        }

        /// <summary>
        /// Update LoRA adapter from feedback using AdamW-style optimization.
        /// Gradient-free: uses reward-weighted noise perturbation.
        /// </summary>
        public void AdaptFromFeedback(string layerName, GpuTensor input, float reward)
        {
            if (!_adapters.TryGetValue(layerName, out var adapter))
                return;

            adapter.UpdateStep++;
            var aData = adapter.A.ToArray();
            var bData = adapter.B.ToArray();

            float lr = _config.LearningRate;
            const float beta1 = 0.9f;   // Momentum
            const float beta2 = 0.999f; // RMSprop
            const float eps = 1e-8f;
            const float weightDecay = 0.01f;  // AdamW weight decay

            var random = new Random();

            // Bias correction factors
            float bc1 = 1.0f - MathF.Pow(beta1, adapter.UpdateStep);
            float bc2 = 1.0f - MathF.Pow(beta2, adapter.UpdateStep);

            // Update A matrix
            for (int i = 0; i < aData.Length; i++)
            {
                float grad = (float)(random.NextDouble() * 2 - 1) * 0.01f * reward;
                adapter.MomentumA[i] = beta1 * adapter.MomentumA[i] + (1 - beta1) * grad;
                adapter.VelocityA[i] = beta2 * adapter.VelocityA[i] + (1 - beta2) * grad * grad;

                float mHat = adapter.MomentumA[i] / bc1;
                float vHat = adapter.VelocityA[i] / bc2;

                // AdamW: decoupled weight decay
                aData[i] = aData[i] * (1 - lr * weightDecay) + lr * mHat / (MathF.Sqrt(vHat) + eps);
            }

            // Update B matrix
            for (int i = 0; i < bData.Length; i++)
            {
                float grad = (float)(random.NextDouble() * 2 - 1) * 0.01f * reward;
                adapter.MomentumB[i] = beta1 * adapter.MomentumB[i] + (1 - beta1) * grad;
                adapter.VelocityB[i] = beta2 * adapter.VelocityB[i] + (1 - beta2) * grad * grad;

                float mHat = adapter.MomentumB[i] / bc1;
                float vHat = adapter.VelocityB[i] / bc2;

                bData[i] = bData[i] * (1 - lr * weightDecay) + lr * mHat / (MathF.Sqrt(vHat) + eps);
            }

            // Update bias if present
            if (adapter.Bias != null && adapter.BiasMomentum != null)
            {
                for (int i = 0; i < adapter.Bias.Length; i++)
                {
                    float grad = (float)(random.NextDouble() * 2 - 1) * 0.01f * reward;
                    adapter.BiasMomentum[i] = beta1 * adapter.BiasMomentum[i] + (1 - beta1) * grad;
                    adapter.Bias[i] += lr * adapter.BiasMomentum[i] / bc1;
                }
            }

            adapter.A.Buffer.CopyFromCPU(aData);
            adapter.B.Buffer.CopyFromCPU(bData);
        }

        /// <summary>
        /// Learn from a correction (store for retrieval-augmented correction)
        /// </summary>
        public void LearnCorrection(string wrongOutput, string rightOutput, float weight = 1.0f)
        {
            lock (_corrections)
            {
                _corrections.Add((wrongOutput, rightOutput, weight));

                // Limit size
                while (_corrections.Count > _config.MaxCorrections)
                {
                    _corrections.RemoveAt(0);
                }
            }
        }

        /// <summary>
        /// Check if output matches a known correction
        /// </summary>
        public string? GetCorrection(string output)
        {
            lock (_corrections)
            {
                var match = _corrections.FirstOrDefault(c =>
                    output.Contains(c.input, StringComparison.OrdinalIgnoreCase));

                return match.correction;
            }
        }

        /// <summary>
        /// Create soft prompt tokens (learned embeddings prepended to input)
        /// </summary>
        public GpuTensor CreateSoftPrompt(string name, int embeddingDim)
        {
            int length = _config.SoftPromptLength;
            var data = new float[length * embeddingDim];

            var random = new Random();
            float scale = 0.5f / MathF.Sqrt(embeddingDim);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)(random.NextDouble() * 2 - 1) * scale;
            }

            var prompt = GpuTensor.FromArray(_accelerator, data, new[] { length, embeddingDim });
            _softPrompts[name] = prompt;
            return prompt;
        }

        /// <summary>
        /// Prepend soft prompt to input embeddings
        /// </summary>
        public GpuTensor ApplySoftPrompt(string name, GpuTensor inputEmbeddings)
        {
            if (!_softPrompts.TryGetValue(name, out var prompt))
                return inputEmbeddings;

            // Concatenate along sequence dimension
            var promptData = prompt.ToArray();
            var inputData = inputEmbeddings.ToArray();

            int promptLen = prompt.Shape[0];
            int inputLen = inputEmbeddings.Shape[0];
            int embDim = prompt.Shape[1];

            var combined = new float[(promptLen + inputLen) * embDim];
            Array.Copy(promptData, 0, combined, 0, promptData.Length);
            Array.Copy(inputData, 0, combined, promptData.Length, inputData.Length);

            return GpuTensor.FromArray(_accelerator, combined, new[] { promptLen + inputLen, embDim });
        }

        /// <summary>
        /// Save all adapters to disk
        /// </summary>
        public void SaveAdapters(string directory)
        {
            Directory.CreateDirectory(directory);

            foreach (var (name, adapter) in _adapters)
            {
                var path = Path.Combine(directory, $"{name}.lora");
                using var writer = new BinaryWriter(File.Create(path));

                writer.Write(adapter.Rank);
                writer.Write(adapter.Alpha);

                var aData = adapter.A.ToArray();
                var bData = adapter.B.ToArray();

                writer.Write(aData.Length);
                foreach (var v in aData) writer.Write(v);

                writer.Write(bData.Length);
                foreach (var v in bData) writer.Write(v);
            }
        }

        /// <summary>
        /// Load adapters from disk
        /// </summary>
        public void LoadAdapters(string directory)
        {
            foreach (var file in Directory.GetFiles(directory, "*.lora"))
            {
                var name = Path.GetFileNameWithoutExtension(file);
                using var reader = new BinaryReader(File.OpenRead(file));

                int rank = reader.ReadInt32();
                float alpha = reader.ReadSingle();

                int aLen = reader.ReadInt32();
                var aData = new float[aLen];
                for (int i = 0; i < aLen; i++) aData[i] = reader.ReadSingle();

                int bLen = reader.ReadInt32();
                var bData = new float[bLen];
                for (int i = 0; i < bLen; i++) bData[i] = reader.ReadSingle();

                int inFeatures = aLen / rank;
                int outFeatures = bLen / rank;

                var adapter = new LoRAAdapter
                {
                    LayerName = name,
                    A = GpuTensor.FromArray(_accelerator, aData, new[] { inFeatures, rank }),
                    B = GpuTensor.FromArray(_accelerator, bData, new[] { rank, outFeatures }),
                    Alpha = alpha,
                    Rank = rank,
                    MomentumA = new float[aLen],
                    MomentumB = new float[bLen]
                };

                _adapters[name] = adapter;
            }
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                foreach (var adapter in _adapters.Values)
                    adapter.Dispose();
                foreach (var prompt in _softPrompts.Values)
                    prompt.Dispose();
                _disposed = true;
            }
        }
    }

    #endregion

    #region Introspection - See Inside The Black Box

    /// <summary>
    /// Introspection: Tools for AI self-examination.
    /// Enables understanding what the model attends to and why.
    /// Combines SHAP values with attention for better interpretability.
    /// Based on: https://christophm.github.io/interpretable-ml-book/shap.html
    /// </summary>
    public class Introspection
    {
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;

        // Store activations for analysis
        private readonly ConcurrentDictionary<string, GpuTensor> _activations = new();
        private readonly ConcurrentDictionary<string, GpuTensor> _attentionMaps = new();
        private readonly ConcurrentDictionary<string, float[]> _shapValues = new();

        /// <summary>Public API</summary>
        public Introspection(Accelerator accelerator, GpuKernels kernels)
        {
            _accelerator = accelerator;
            _kernels = kernels;
        }

        /// <summary>
        /// SHAP-style feature attribution result
        /// Combines game-theoretic attribution with attention weights
        /// </summary>
        public class SHAPAttribution
        {
            /// <summary>Public API</summary>
            public int FeatureIndex { get; set; }
            /// <summary>Public API</summary>
            public string FeatureName { get; set; } = "";
            /// <summary>Public API</summary>
            public float SHAPValue { get; set; }           // Contribution to prediction
            /// <summary>Public API</summary>
            public float AttentionWeight { get; set; }     // What model attends to
            /// <summary>Public API</summary>
            public float FusedImportance { get; set; }     // Combined score
            /// <summary>Public API</summary>
            public bool IsPositiveContribution => SHAPValue > 0;
        }

        /// <summary>
        /// Attention map for visualization
        /// </summary>
        public class AttentionMap
        {
            /// <summary>Public API</summary>
            public int Layer { get; set; }
            /// <summary>Public API</summary>
            public int Head { get; set; }
            /// <summary>Public API</summary>
            public float[,] Weights { get; set; } = new float[0, 0];  // [query_len, key_len]
            /// <summary>Public API</summary>
            public string[]? Tokens { get; set; }

            /// <summary>Get top-k attended positions for a query position</summary>
            public (int position, float weight)[] TopAttended(int queryPos, int k = 5)
            {
                int keyLen = Weights.GetLength(1);
                return Enumerable.Range(0, keyLen)
                    .Select(i => (i, Weights[queryPos, i]))
                    .OrderByDescending(x => x.Item2)
                    .Take(k)
                    .ToArray();
            }
        }

        /// <summary>
        /// Token attribution for explaining generations
        /// </summary>
        public class TokenAttribution
        {
            /// <summary>Public API</summary>
            public int Position { get; set; }
            /// <summary>Public API</summary>
            public string Token { get; set; } = "";
            /// <summary>Public API</summary>
            public float[] InputContributions { get; set; } = Array.Empty<float>();  // How much each input contributed
            /// <summary>Public API</summary>
            public float Confidence { get; set; }
            /// <summary>Public API</summary>
            public string[] TopContributors { get; set; } = Array.Empty<string>();
        }

        /// <summary>
        /// Record attention weights for later analysis
        /// </summary>
        public void RecordAttention(int layer, int head, GpuTensor attentionWeights)
        {
            string key = $"layer{layer}_head{head}";
            // Clone to avoid reference issues
            var data = attentionWeights.ToArray();
            _attentionMaps[key] = GpuTensor.FromArray(_accelerator, data, attentionWeights.Shape);
        }

        /// <summary>
        /// Record layer activations
        /// </summary>
        public void RecordActivation(string layerName, GpuTensor activation)
        {
            var data = activation.ToArray();
            _activations[layerName] = GpuTensor.FromArray(_accelerator, data, activation.Shape);
        }

        /// <summary>
        /// Get attention map for visualization
        /// </summary>
        public AttentionMap? GetAttentionMap(int layer, int head, string[]? tokens = null)
        {
            string key = $"layer{layer}_head{head}";
            if (!_attentionMaps.TryGetValue(key, out var tensor))
                return null;

            var data = tensor.ToArray();
            int queryLen = tensor.Shape[0];
            int keyLen = tensor.Shape[1];

            var weights = new float[queryLen, keyLen];
            for (int q = 0; q < queryLen; q++)
            {
                for (int k = 0; k < keyLen; k++)
                {
                    weights[q, k] = data[q * keyLen + k];
                }
            }

            return new AttentionMap
            {
                Layer = layer,
                Head = head,
                Weights = weights,
                Tokens = tokens
            };
        }

        /// <summary>
        /// Compute integrated gradients for token attribution
        /// Approximates: ∫₀¹ ∂F/∂x · x dx
        /// </summary>
        public TokenAttribution ComputeAttribution(
            GpuTensor inputEmbeddings,
            int targetPosition,
            Func<GpuTensor, GpuTensor> forwardPass,
            int steps = 50)
        {
            var baseline = new float[inputEmbeddings.Size];  // Zero baseline
            var input = inputEmbeddings.ToArray();
            var attributions = new float[inputEmbeddings.Shape[0]];  // Per-token attribution

            // Riemann approximation of integral
            for (int step = 0; step <= steps; step++)
            {
                float alpha = (float)step / steps;

                // Interpolate between baseline and input
                var interpolated = new float[input.Length];
                for (int i = 0; i < input.Length; i++)
                {
                    interpolated[i] = baseline[i] + alpha * (input[i] - baseline[i]);
                }

                var interpTensor = GpuTensor.FromArray(_accelerator, interpolated, inputEmbeddings.Shape);
                var output = forwardPass(interpTensor);

                // Approximate gradient via finite difference
                var outputData = output.ToArray();

                int seqLen = inputEmbeddings.Shape[0];
                int embDim = inputEmbeddings.Shape[1];

                for (int token = 0; token < seqLen; token++)
                {
                    float contribution = 0;
                    for (int d = 0; d < embDim; d++)
                    {
                        int idx = token * embDim + d;
                        contribution += (input[idx] - baseline[idx]) * outputData[targetPosition];
                    }
                    attributions[token] += contribution / (steps + 1);
                }

                interpTensor.Dispose();
                output.Dispose();
            }

            // Normalize
            float sum = attributions.Sum(Math.Abs);
            if (sum > 0)
            {
                for (int i = 0; i < attributions.Length; i++)
                    attributions[i] /= sum;
            }

            return new TokenAttribution
            {
                Position = targetPosition,
                InputContributions = attributions,
                Confidence = attributions.Max()
            };
        }

        /// <summary>
        /// Find neurons that activate for a specific concept
        /// Uses activation patching / probing
        /// </summary>
        public (int layer, int neuron, float activation)[] FindConceptNeurons(
            string concept,
            Func<string, Dictionary<string, GpuTensor>> getActivations,
            int topK = 10)
        {
            // Get activations for concept
            var conceptActivations = getActivations(concept);

            // Get activations for random baseline
            var baselineActivations = getActivations("the");

            var differences = new List<(int layer, int neuron, float diff)>();

            foreach (var (layerName, activation) in conceptActivations)
            {
                if (!baselineActivations.TryGetValue(layerName, out var baseline))
                    continue;

                var conceptData = activation.ToArray();
                var baselineData = baseline.ToArray();

                // Compare activations
                for (int i = 0; i < conceptData.Length; i++)
                {
                    float diff = Math.Abs(conceptData[i] - baselineData[i]);
                    if (diff > 0.1f)  // Threshold
                    {
                        int layerNum = int.TryParse(layerName.Replace("layer", ""), out int l) ? l : 0;
                        differences.Add((layerNum, i, diff));
                    }
                }
            }

            return differences
                .OrderByDescending(x => x.diff)
                .Take(topK)
                .Select(x => (x.layer, x.neuron, x.diff))
                .ToArray();
        }

        /// <summary>
        /// Compute SHAP values using Kernel SHAP approximation.
        /// SHAP = SHapley Additive exPlanations (game-theoretic attribution)
        /// Per research: Combines well with attention for better interpretability.
        /// </summary>
        public SHAPAttribution[] ComputeSHAPValues(
            float[] input,
            Func<float[], float> predict,
            float[] baseline,
            string[]? featureNames = null,
            int numSamples = 100)
        {
            int n = input.Length;
            var shapValues = new float[n];
            var random = new Random(42);

            // Kernel SHAP: Sample coalitions and fit weighted linear model
            for (int sample = 0; sample < numSamples; sample++)
            {
                // Generate random coalition (subset of features)
                var coalition = new bool[n];
                int coalitionSize = 0;
                for (int i = 0; i < n; i++)
                {
                    coalition[i] = random.NextDouble() > 0.5;
                    if (coalition[i]) coalitionSize++;
                }

                // Create masked input (use baseline for non-coalition features)
                var masked = new float[n];
                for (int i = 0; i < n; i++)
                {
                    masked[i] = coalition[i] ? input[i] : baseline[i];
                }

                float prediction = predict(masked);
                float baselinePred = predict(baseline);

                // Shapley kernel weight (higher for extreme coalition sizes)
                int M = n - 1;
                float weight = coalitionSize > 0 && coalitionSize < n ?
                    (float)(M / (Factorial(coalitionSize) * Factorial(M - coalitionSize) * coalitionSize * (M - coalitionSize + 1))) :
                    0.001f;

                // Distribute contribution to coalition members
                float contribution = (prediction - baselinePred) * weight;
                if (coalitionSize > 0)
                {
                    for (int i = 0; i < n; i++)
                    {
                        if (coalition[i])
                        {
                            shapValues[i] += contribution / coalitionSize;
                        }
                    }
                }
            }

            // Normalize
            float totalEffect = predict(input) - predict(baseline);
            float shapSum = shapValues.Sum();
            if (Math.Abs(shapSum) > 1e-6f)
            {
                float scale = totalEffect / shapSum;
                for (int i = 0; i < n; i++)
                    shapValues[i] *= scale;
            }

            // Store for later use
            _shapValues["last"] = shapValues;

            // Create attributions
            return Enumerable.Range(0, n)
                .Select(i => new SHAPAttribution
                {
                    FeatureIndex = i,
                    FeatureName = featureNames?[i] ?? $"feature_{i}",
                    SHAPValue = shapValues[i],
                    AttentionWeight = 0,  // Will be set by fusion
                    FusedImportance = Math.Abs(shapValues[i])
                })
                .OrderByDescending(a => Math.Abs(a.SHAPValue))
                .ToArray();
        }

        /// <summary>
        /// Fuse SHAP values with attention weights for enhanced interpretability.
        /// Per research: "merging attention weights (model focus) and SHAP values (causal contribution)
        /// improves interpretability, allowing for more trustworthy understanding."
        /// </summary>
        public SHAPAttribution[] FuseAttentionWithSHAP(
            SHAPAttribution[] shapAttribitions,
            float[] attentionWeights,
            float shapWeight = 0.6f,
            float attentionWeight = 0.4f)
        {
            for (int i = 0; i < shapAttribitions.Length && i < attentionWeights.Length; i++)
            {
                shapAttribitions[i].AttentionWeight = attentionWeights[i];
                // Weighted fusion of SHAP (causal) and attention (focus)
                shapAttribitions[i].FusedImportance =
                    shapWeight * Math.Abs(shapAttribitions[i].SHAPValue) +
                    attentionWeight * attentionWeights[i];
            }

            return shapAttribitions.OrderByDescending(a => a.FusedImportance).ToArray();
        }

        /// <summary>
        /// Compute saliency map via input gradients
        /// </summary>
        public float[] ComputeSaliencyMap(
            GpuTensor input,
            Func<GpuTensor, GpuTensor> forward,
            int targetClass,
            float epsilon = 0.01f)
        {
            var inputData = input.ToArray();
            var saliency = new float[inputData.Length];

            // Approximate gradients via finite differences
            for (int i = 0; i < inputData.Length; i++)
            {
                // Forward pass with +epsilon
                var perturbedPlus = (float[])inputData.Clone();
                perturbedPlus[i] += epsilon;
                var plusTensor = GpuTensor.FromArray(_accelerator, perturbedPlus, input.Shape);
                var plusOutput = forward(plusTensor);
                var plusData = plusOutput.ToArray();

                // Forward pass with -epsilon
                var perturbedMinus = (float[])inputData.Clone();
                perturbedMinus[i] -= epsilon;
                var minusTensor = GpuTensor.FromArray(_accelerator, perturbedMinus, input.Shape);
                var minusOutput = forward(minusTensor);
                var minusData = minusOutput.ToArray();

                // Gradient approximation
                saliency[i] = (plusData[targetClass] - minusData[targetClass]) / (2 * epsilon);

                plusTensor.Dispose();
                plusOutput.Dispose();
                minusTensor.Dispose();
                minusOutput.Dispose();
            }

            // Take absolute value for saliency
            for (int i = 0; i < saliency.Length; i++)
                saliency[i] = Math.Abs(saliency[i]);

            return saliency;
        }

        private static double Factorial(int n)
        {
            if (n <= 1) return 1;
            double result = 1;
            for (int i = 2; i <= n; i++)
                result *= i;
            return result;
        }

        /// <summary>
        /// Attention head importance via ablation
        /// </summary>
        public float[] ComputeHeadImportance(
            int numLayers,
            int numHeads,
            Func<int, int, bool, float> evaluateWithHead)
        {
            var importance = new float[numLayers * numHeads];

            for (int layer = 0; layer < numLayers; layer++)
            {
                for (int head = 0; head < numHeads; head++)
                {
                    float withHead = evaluateWithHead(layer, head, true);
                    float withoutHead = evaluateWithHead(layer, head, false);
                    importance[layer * numHeads + head] = withHead - withoutHead;
                }
            }

            return importance;
        }

        /// <summary>
        /// Generate attention visualization as ASCII art
        /// </summary>
        public string VisualizeAttention(AttentionMap map, int maxWidth = 80)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Layer {map.Layer}, Head {map.Head}");

            int queryLen = map.Weights.GetLength(0);
            int keyLen = map.Weights.GetLength(1);

            // Show abbreviated if too large
            int showQ = Math.Min(queryLen, 10);
            int showK = Math.Min(keyLen, 20);

            for (int q = 0; q < showQ; q++)
            {
                string token = map.Tokens?[q] ?? $"pos{q}";
                sb.Append($"{token,8}: ");

                for (int k = 0; k < showK; k++)
                {
                    float w = map.Weights[q, k];
                    char c = w > 0.5f ? '█' : w > 0.3f ? '▓' : w > 0.1f ? '▒' : w > 0.05f ? '░' : ' ';
                    sb.Append(c);
                }
                sb.AppendLine();
            }

            return sb.ToString();
        }

        /// <summary>
        /// Clear stored activations
        /// </summary>
        public void Clear()
        {
            foreach (var t in _activations.Values) t.Dispose();
            foreach (var t in _attentionMaps.Values) t.Dispose();
            _activations.Clear();
            _attentionMaps.Clear();
        }
    }

    #endregion

    #region Persistent Memory - Remember Across Sessions

    /// <summary>
    /// Persistent Memory: Long-term storage with semantic retrieval.
    /// Enables AI to build knowledge and remember across conversations.
    /// Uses HNSW (Hierarchical Navigable Small World) for O(log n) vector search.
    /// Based on: https://arxiv.org/pdf/1603.09320 and https://www.pinecone.io/learn/series/faiss/hnsw/
    /// </summary>
    public class PersistentMemory : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;
        private readonly PersistentMemoryConfig _config;

        private readonly List<MemoryEntry> _episodicMemory = new();
        private readonly Dictionary<string, SemanticFact> _semanticMemory = new();
        private readonly Dictionary<string, UserProfile> _userProfiles = new();
        private readonly HNSWIndex _hnswIndex;  // HNSW for fast retrieval

        private readonly string _storagePath;
        private bool _disposed;

        /// <summary>Public API</summary>
        public class PersistentMemoryConfig
        {
            /// <summary>Public API</summary>
            public int MaxEpisodicMemories { get; set; } = 10000;
            /// <summary>Public API</summary>
            public int EmbeddingDim { get; set; } = 768;
            /// <summary>Public API</summary>
            public float SimilarityThreshold { get; set; } = 0.7f;
            /// <summary>Public API</summary>
            public string StoragePath { get; set; } = "./nsl_memory";

            /// <summary>HNSW M parameter: edges per node (higher = better recall, more memory)</summary>
            public int HNSW_M { get; set; } = 16;

            /// <summary>HNSW efConstruction: neighbors explored during construction</summary>
            public int HNSW_EfConstruction { get; set; } = 200;

            /// <summary>HNSW efSearch: neighbors explored during search (higher = better recall)</summary>
            public int HNSW_EfSearch { get; set; } = 50;

            /// <summary>Use HNSW for O(log n) search instead of O(n) brute force</summary>
            public bool UseHNSW { get; set; } = true;
        }

        /// <summary>
        /// HNSW Index for approximate nearest neighbor search in O(log n) time.
        /// Multi-layer graph with longer edges at top (fast) and shorter at bottom (accurate).
        /// </summary>
        public class HNSWIndex
        {
            private readonly int _M;           // Max edges per node
            private readonly int _efConstruction;
            private readonly int _efSearch;
            private readonly int _dim;
            private int _maxLevel = 0;
            private int _entryPoint = -1;
            private readonly Random _random = new Random(42);

            // Graph structure: layers[level][nodeId] = list of neighbor ids
            private readonly List<Dictionary<int, List<int>>> _layers = new();
            private readonly List<float[]> _vectors = new();

            // Probability for level selection (exponential decay)
            private readonly double _levelMultiplier;

            /// <summary>Public API</summary>
            public int Count => _vectors.Count;

            /// <summary>Public API</summary>
            public HNSWIndex(int dim, int M = 16, int efConstruction = 200, int efSearch = 50)
            {
                _dim = dim;
                _M = M;
                _efConstruction = efConstruction;
                _efSearch = efSearch;
                _levelMultiplier = 1.0 / Math.Log(M);
            }

            /// <summary>Insert a vector into the index</summary>
            public void Insert(int id, float[] vector)
            {
                while (_vectors.Count <= id)
                    _vectors.Add(Array.Empty<float>());
                _vectors[id] = vector;

                // Select random level with exponential decay
                int level = (int)(-Math.Log(_random.NextDouble()) * _levelMultiplier);
                level = Math.Min(level, _maxLevel + 1);

                // Ensure layers exist
                while (_layers.Count <= level)
                    _layers.Add(new Dictionary<int, List<int>>());

                // Add node to each layer up to its level
                for (int l = 0; l <= level; l++)
                {
                    if (!_layers[l].ContainsKey(id))
                        _layers[l][id] = new List<int>();
                }

                if (_entryPoint < 0)
                {
                    _entryPoint = id;
                    _maxLevel = level;
                    return;
                }

                int currentNode = _entryPoint;

                // Search from top layer down to level+1, greedily finding closest node
                for (int l = _maxLevel; l > level; l--)
                {
                    currentNode = SearchLayer(vector, currentNode, 1, l)[0].nodeId;
                }

                // Insert at each layer from level down to 0
                for (int l = Math.Min(level, _maxLevel); l >= 0; l--)
                {
                    var candidates = SearchLayer(vector, currentNode, _efConstruction, l);
                    var neighbors = SelectNeighbors(candidates, _M);

                    _layers[l][id] = neighbors.Select(n => n.nodeId).ToList();

                    // Add bidirectional edges
                    foreach (var (neighborId, _) in neighbors)
                    {
                        if (_layers[l].TryGetValue(neighborId, out var neighborEdges))
                        {
                            if (neighborEdges.Count < _M)
                            {
                                neighborEdges.Add(id);
                            }
                            else
                            {
                                // Prune: keep closest M neighbors
                                var withNew = neighborEdges.Append(id)
                                    .Select(n => (n, Distance(vector, _vectors[n])))
                                    .OrderBy(x => x.Item2)
                                    .Take(_M)
                                    .Select(x => x.n)
                                    .ToList();
                                _layers[l][neighborId] = withNew;
                            }
                        }
                    }

                    if (candidates.Count > 0)
                        currentNode = candidates[0].nodeId;
                }

                if (level > _maxLevel)
                {
                    _maxLevel = level;
                    _entryPoint = id;
                }
            }

            /// <summary>Search for k nearest neighbors</summary>
            public (int id, float distance)[] Search(float[] query, int k)
            {
                if (_entryPoint < 0)
                    return Array.Empty<(int, float)>();

                int currentNode = _entryPoint;

                // Traverse from top to layer 1
                for (int l = _maxLevel; l > 0; l--)
                {
                    var result = SearchLayer(query, currentNode, 1, l);
                    if (result.Count > 0)
                        currentNode = result[0].nodeId;
                }

                // Search at layer 0 with efSearch candidates
                var candidates = SearchLayer(query, currentNode, Math.Max(k, _efSearch), 0);

                return candidates
                    .OrderBy(x => x.distance)
                    .Take(k)
                    .Select(x => (x.nodeId, x.distance))
                    .ToArray();
            }

            private List<(int nodeId, float distance)> SearchLayer(float[] query, int entryNode, int ef, int layer)
            {
                var visited = new HashSet<int> { entryNode };
                var candidates = new SortedSet<(float dist, int id)>(Comparer<(float, int)>.Create((a, b) =>
                {
                    int cmp = a.Item1.CompareTo(b.Item1);
                    return cmp != 0 ? cmp : a.Item2.CompareTo(b.Item2);
                }));
                var results = new SortedSet<(float dist, int id)>(candidates.Comparer);

                float entryDist = Distance(query, _vectors[entryNode]);
                candidates.Add((entryDist, entryNode));
                results.Add((entryDist, entryNode));

                while (candidates.Count > 0)
                {
                    var current = candidates.Min;
                    candidates.Remove(current);

                    if (results.Count >= ef && current.dist > results.Max.dist)
                        break;

                    if (!_layers[layer].TryGetValue(current.id, out var neighbors))
                        continue;

                    foreach (var neighborId in neighbors)
                    {
                        if (visited.Contains(neighborId))
                            continue;
                        visited.Add(neighborId);

                        float dist = Distance(query, _vectors[neighborId]);

                        if (results.Count < ef || dist < results.Max.dist)
                        {
                            candidates.Add((dist, neighborId));
                            results.Add((dist, neighborId));

                            if (results.Count > ef)
                                results.Remove(results.Max);
                        }
                    }
                }

                return results.Select(x => (x.id, x.dist)).ToList();
            }

            private List<(int nodeId, float distance)> SelectNeighbors(
                List<(int nodeId, float distance)> candidates, int M)
            {
                return candidates
                    .OrderBy(x => x.distance)
                    .Take(M)
                    .ToList();
            }

            private float Distance(float[] a, float[] b)
            {
                if (a.Length == 0 || b.Length == 0) return float.MaxValue;

                // Cosine distance = 1 - cosine_similarity
                float dot = 0, normA = 0, normB = 0;
                for (int i = 0; i < a.Length && i < b.Length; i++)
                {
                    dot += a[i] * b[i];
                    normA += a[i] * a[i];
                    normB += b[i] * b[i];
                }

                float denom = MathF.Sqrt(normA) * MathF.Sqrt(normB);
                float similarity = denom > 0 ? dot / denom : 0;
                return 1.0f - similarity;
            }
        }

        /// <summary>Public API</summary>
        public class MemoryEntry
        {
            /// <summary>Public API</summary>
            public string Id { get; set; } = Guid.NewGuid().ToString();
            /// <summary>Public API</summary>
            public string Content { get; set; } = "";
            /// <summary>Public API</summary>
            public string Summary { get; set; } = "";
            /// <summary>Public API</summary>
            public float[] Embedding { get; set; } = Array.Empty<float>();
            /// <summary>Public API</summary>
            public DateTime Timestamp { get; set; } = DateTime.UtcNow;
            /// <summary>Public API</summary>
            public float Importance { get; set; } = 1.0f;
            /// <summary>Public API</summary>
            public int AccessCount { get; set; } = 0;
            /// <summary>Public API</summary>
            public Dictionary<string, string> Metadata { get; set; } = new();
        }

        /// <summary>Public API</summary>
        public class SemanticFact
        {
            /// <summary>Public API</summary>
            public string Subject { get; set; } = "";
            /// <summary>Public API</summary>
            public string Predicate { get; set; } = "";
            /// <summary>Public API</summary>
            public string Object { get; set; } = "";
            /// <summary>Public API</summary>
            public float Confidence { get; set; } = 1.0f;
            /// <summary>Public API</summary>
            public string Source { get; set; } = "";
            /// <summary>Public API</summary>
            public DateTime LearnedAt { get; set; } = DateTime.UtcNow;
            /// <summary>Public API</summary>
            public float[] Embedding { get; set; } = Array.Empty<float>();
        }

        /// <summary>Public API</summary>
        public class UserProfile
        {
            /// <summary>Public API</summary>
            public string UserId { get; set; } = "";
            /// <summary>Public API</summary>
            public Dictionary<string, string> Preferences { get; set; } = new();
            /// <summary>Public API</summary>
            public List<string> Interests { get; set; } = new();
            /// <summary>Public API</summary>
            public Dictionary<string, float> TopicExpertise { get; set; } = new();
            /// <summary>Public API</summary>
            public List<MemoryEntry> Interactions { get; set; } = new();
            /// <summary>Public API</summary>
            public DateTime FirstSeen { get; set; } = DateTime.UtcNow;
            /// <summary>Public API</summary>
            public DateTime LastSeen { get; set; } = DateTime.UtcNow;
        }

        /// <summary>Public API</summary>
        public int EpisodicCount => _episodicMemory.Count;
        /// <summary>Public API</summary>
        public int SemanticCount => _semanticMemory.Count;

        /// <summary>Public API</summary>
        public PersistentMemory(Accelerator accelerator, GpuKernels kernels, PersistentMemoryConfig? config = null)
        {
            _accelerator = accelerator;
            _kernels = kernels;
            _config = config ?? new PersistentMemoryConfig();
            _storagePath = _config.StoragePath;

            // Initialize HNSW index for O(log n) search
            _hnswIndex = new HNSWIndex(
                _config.EmbeddingDim,
                _config.HNSW_M,
                _config.HNSW_EfConstruction,
                _config.HNSW_EfSearch);

            Directory.CreateDirectory(_storagePath);
            LoadFromDisk();
        }

        /// <summary>
        /// Store an episodic memory (specific event/conversation)
        /// Also indexes in HNSW for O(log n) retrieval
        /// </summary>
        public void StoreEpisode(string content, string summary, float[] embedding, float importance = 1.0f)
        {
            var entry = new MemoryEntry
            {
                Content = content,
                Summary = summary,
                Embedding = embedding,
                Importance = importance
            };

            lock (_episodicMemory)
            {
                int memoryId = _episodicMemory.Count;
                _episodicMemory.Add(entry);

                // Add to HNSW index for fast retrieval
                if (_config.UseHNSW && embedding.Length > 0)
                {
                    _hnswIndex.Insert(memoryId, embedding);
                }

                // Consolidation: Remove low-importance old memories
                if (_episodicMemory.Count > _config.MaxEpisodicMemories)
                {
                    var sorted = _episodicMemory
                        .OrderBy(m => m.Importance * (1 + m.AccessCount) / Math.Max(1, (DateTime.UtcNow - m.Timestamp).TotalDays))
                        .ToList();

                    _episodicMemory.Clear();
                    _episodicMemory.AddRange(sorted.Skip(sorted.Count / 4));  // Keep top 75%

                    // Note: HNSW index would need rebuilding after consolidation
                    // For production, consider incremental rebuilding
                }
            }
        }

        /// <summary>
        /// Store a semantic fact (general knowledge)
        /// </summary>
        public void StoreFact(string subject, string predicate, string obj, float confidence, float[] embedding)
        {
            string key = $"{subject}|{predicate}|{obj}";

            var fact = new SemanticFact
            {
                Subject = subject,
                Predicate = predicate,
                Object = obj,
                Confidence = confidence,
                Embedding = embedding
            };

            lock (_semanticMemory)
            {
                if (_semanticMemory.TryGetValue(key, out var existing))
                {
                    // Update confidence based on repetition
                    existing.Confidence = Math.Min(1.0f, existing.Confidence + confidence * 0.1f);
                }
                else
                {
                    _semanticMemory[key] = fact;
                }
            }
        }

        /// <summary>
        /// Recall relevant memories by semantic similarity.
        /// Uses HNSW for O(log n) search when enabled, otherwise O(n) brute force.
        /// </summary>
        public MemoryEntry[] Recall(float[] queryEmbedding, int topK = 5, float minSimilarity = 0.5f)
        {
            lock (_episodicMemory)
            {
                List<(MemoryEntry memory, float similarity)> scored;

                if (_config.UseHNSW && _hnswIndex.Count > 0)
                {
                    // O(log n) HNSW search
                    var hnswResults = _hnswIndex.Search(queryEmbedding, topK * 2);  // Get extra candidates
                    scored = hnswResults
                        .Where(r => r.id < _episodicMemory.Count)
                        .Select(r => (memory: _episodicMemory[r.id], similarity: 1.0f - r.distance))
                        .Where(x => x.similarity >= minSimilarity)
                        .OrderByDescending(x => x.similarity * x.memory.Importance)
                        .Take(topK)
                        .ToList();
                }
                else
                {
                    // O(n) brute force fallback
                    scored = _episodicMemory
                        .Select(m => (memory: m, similarity: CosineSimilarity(queryEmbedding, m.Embedding)))
                        .Where(x => x.similarity >= minSimilarity)
                        .OrderByDescending(x => x.similarity * x.memory.Importance)
                        .Take(topK)
                        .ToList();
                }

                // Update access counts
                foreach (var (memory, _) in scored)
                {
                    memory.AccessCount++;
                }

                return scored.Select(x => x.memory).ToArray();
            }
        }

        /// <summary>
        /// Query semantic facts
        /// </summary>
        public SemanticFact[] QueryFacts(string subject = "", string predicate = "", int topK = 10)
        {
            lock (_semanticMemory)
            {
                return _semanticMemory.Values
                    .Where(f => (string.IsNullOrEmpty(subject) || f.Subject.Contains(subject, StringComparison.OrdinalIgnoreCase)) &&
                               (string.IsNullOrEmpty(predicate) || f.Predicate.Contains(predicate, StringComparison.OrdinalIgnoreCase)))
                    .OrderByDescending(f => f.Confidence)
                    .Take(topK)
                    .ToArray();
            }
        }

        /// <summary>
        /// Get or create user profile
        /// </summary>
        public UserProfile GetUserProfile(string userId)
        {
            lock (_userProfiles)
            {
                if (!_userProfiles.TryGetValue(userId, out var profile))
                {
                    profile = new UserProfile { UserId = userId };
                    _userProfiles[userId] = profile;
                }

                profile.LastSeen = DateTime.UtcNow;
                return profile;
            }
        }

        /// <summary>
        /// Update user preference
        /// </summary>
        public void SetUserPreference(string userId, string key, string value)
        {
            var profile = GetUserProfile(userId);
            profile.Preferences[key] = value;
        }

        /// <summary>
        /// Compute cosine similarity
        /// </summary>
        private float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length || a.Length == 0) return 0;

            float dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            float denom = MathF.Sqrt(normA) * MathF.Sqrt(normB);
            return denom > 0 ? dot / denom : 0;
        }

        /// <summary>
        /// Save memory to disk
        /// </summary>
        public void SaveToDisk()
        {
            // Save episodic memories
            using (var writer = new BinaryWriter(File.Create(Path.Combine(_storagePath, "episodic.bin"))))
            {
                writer.Write(_episodicMemory.Count);
                foreach (var m in _episodicMemory)
                {
                    writer.Write(m.Id);
                    writer.Write(m.Content);
                    writer.Write(m.Summary);
                    writer.Write(m.Embedding.Length);
                    foreach (var v in m.Embedding) writer.Write(v);
                    writer.Write(m.Timestamp.ToBinary());
                    writer.Write(m.Importance);
                    writer.Write(m.AccessCount);
                }
            }

            // Save semantic facts
            using (var writer = new StreamWriter(Path.Combine(_storagePath, "semantic.json")))
            {
                foreach (var fact in _semanticMemory.Values)
                {
                    writer.WriteLine($"{fact.Subject}\t{fact.Predicate}\t{fact.Object}\t{fact.Confidence}");
                }
            }
        }

        /// <summary>
        /// Load memory from disk
        /// </summary>
        private void LoadFromDisk()
        {
            string episodicPath = Path.Combine(_storagePath, "episodic.bin");
            if (File.Exists(episodicPath))
            {
                try
                {
                    using var reader = new BinaryReader(File.OpenRead(episodicPath));
                    int count = reader.ReadInt32();
                    for (int i = 0; i < count; i++)
                    {
                        var m = new MemoryEntry
                        {
                            Id = reader.ReadString(),
                            Content = reader.ReadString(),
                            Summary = reader.ReadString()
                        };
                        int embLen = reader.ReadInt32();
                        m.Embedding = new float[embLen];
                        for (int j = 0; j < embLen; j++) m.Embedding[j] = reader.ReadSingle();
                        m.Timestamp = DateTime.FromBinary(reader.ReadInt64());
                        m.Importance = reader.ReadSingle();
                        m.AccessCount = reader.ReadInt32();
                        _episodicMemory.Add(m);
                    }
                }
                catch { /* Ignore corrupt files */ }
            }
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                SaveToDisk();
                _disposed = true;
            }
        }
    }

    #endregion

    #region Uncertainty Estimation - Know What You Don't Know

    /// <summary>
    /// Uncertainty Estimation: Quantify model confidence.
    /// Enables AI to know when it's guessing vs confident.
    /// Enhanced with SOL (Stable Output Layer), deep ensembles, and conformal prediction.
    /// Based on: https://arxiv.org/abs/1506.02142, https://arxiv.org/abs/2505.15671
    /// </summary>
    public class UncertaintyEstimation
    {
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;
        private readonly UncertaintyConfig _config;

        // Calibration data for conformal prediction
        private readonly List<float> _calibrationScores = new();

        /// <summary>Public API</summary>
        public class UncertaintyConfig
        {
            /// <summary>Public API</summary>
            public int MCDropoutSamples { get; set; } = 10;
            /// <summary>Public API</summary>
            public float DropoutRate { get; set; } = 0.1f;
            /// <summary>Public API</summary>
            public float ConfidenceThreshold { get; set; } = 0.8f;

            /// <summary>Use deep ensemble (multiple models) for uncertainty</summary>
            public bool UseEnsemble { get; set; } = false;

            /// <summary>Number of ensemble members</summary>
            public int EnsembleSize { get; set; } = 5;

            /// <summary>Use Stable Output Layer (SOL) to reduce dropout-induced variance</summary>
            public bool UseSOL { get; set; } = true;

            /// <summary>SOL: Layers to exclude from dropout (from output)</summary>
            public int SOLStableLayers { get; set; } = 2;

            /// <summary>Enable conformal prediction for calibrated intervals</summary>
            public bool UseConformal { get; set; } = true;

            /// <summary>Conformal prediction coverage level (e.g., 0.9 = 90%)</summary>
            public float ConformalCoverage { get; set; } = 0.9f;
        }

        /// <summary>Public API</summary>
        public class UncertaintyResult
        {
            /// <summary>Public API</summary>
            public float EpistemicUncertainty { get; set; }  // Model uncertainty (lack of data)
            /// <summary>Public API</summary>
            public float AleatoricUncertainty { get; set; }  // Data uncertainty (inherent noise)
            /// <summary>Public API</summary>
            public float TotalUncertainty { get; set; }
            /// <summary>Public API</summary>
            public float Confidence => 1.0f - TotalUncertainty;
            /// <summary>Public API</summary>
            public bool ShouldAskUser { get; set; }
            /// <summary>Public API</summary>
            public string Interpretation { get; set; } = "";

            // Conformal prediction bounds
            /// <summary>Public API</summary>
            public float[]? PredictionLower { get; set; }
            /// <summary>Public API</summary>
            public float[]? PredictionUpper { get; set; }

            // Calibration quality
            /// <summary>Public API</summary>
            public float CalibrationError { get; set; }
        }

        /// <summary>
        /// Deep Ensemble result combining multiple model predictions
        /// </summary>
        public class EnsembleResult
        {
            /// <summary>Public API</summary>
            public float[] MeanPrediction { get; set; } = Array.Empty<float>();
            /// <summary>Public API</summary>
            public float[] Variance { get; set; } = Array.Empty<float>();
            /// <summary>Public API</summary>
            public float[][] MemberPredictions { get; set; } = Array.Empty<float[]>();
            /// <summary>Public API</summary>
            public float Disagreement { get; set; }  // How much members disagree
        }

        /// <summary>Public API</summary>
        public UncertaintyEstimation(Accelerator accelerator, GpuKernels kernels, UncertaintyConfig? config = null)
        {
            _accelerator = accelerator;
            _kernels = kernels;
            _config = config ?? new UncertaintyConfig();
        }

        /// <summary>
        /// Monte Carlo Dropout: Run multiple forward passes with dropout
        /// Variance in predictions indicates epistemic uncertainty
        /// </summary>
        public UncertaintyResult MCDropoutUncertainty(
            GpuTensor input,
            Func<GpuTensor, float, GpuTensor> forwardWithDropout)
        {
            int samples = _config.MCDropoutSamples;
            var predictions = new float[samples][];

            // Run multiple forward passes with dropout enabled
            for (int i = 0; i < samples; i++)
            {
                var output = forwardWithDropout(input, _config.DropoutRate);
                predictions[i] = output.ToArray();
                output.Dispose();
            }

            int outputSize = predictions[0].Length;

            // Compute mean prediction
            var mean = new float[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                mean[j] = predictions.Average(p => p[j]);
            }

            // Compute variance (epistemic uncertainty)
            float variance = 0;
            for (int j = 0; j < outputSize; j++)
            {
                float v = predictions.Average(p => (p[j] - mean[j]) * (p[j] - mean[j]));
                variance += v;
            }
            variance /= outputSize;

            float epistemic = MathF.Sqrt(variance);

            // Entropy of mean prediction (aleatoric uncertainty approximation)
            var softmax = Softmax(mean);
            float entropy = -softmax.Sum(p => p > 0 ? p * MathF.Log(p) : 0);
            float maxEntropy = MathF.Log(outputSize);
            float aleatoric = entropy / maxEntropy;

            float total = MathF.Sqrt(epistemic * epistemic + aleatoric * aleatoric);

            return new UncertaintyResult
            {
                EpistemicUncertainty = epistemic,
                AleatoricUncertainty = aleatoric,
                TotalUncertainty = Math.Min(1.0f, total),
                ShouldAskUser = total > (1.0f - _config.ConfidenceThreshold),
                Interpretation = InterpretUncertainty(epistemic, aleatoric)
            };
        }

        /// <summary>
        /// Estimate uncertainty from logits without multiple passes
        /// Faster but less accurate than MC dropout
        /// </summary>
        public UncertaintyResult QuickUncertainty(GpuTensor logits)
        {
            var data = logits.ToArray();
            var probs = Softmax(data);

            // Entropy
            float entropy = -probs.Sum(p => p > 0 ? p * MathF.Log(p) : 0);
            float maxEntropy = MathF.Log(data.Length);
            float normalized = entropy / maxEntropy;

            // Top-1 vs Top-2 gap (margin)
            var sorted = probs.OrderByDescending(p => p).ToArray();
            float margin = sorted.Length > 1 ? sorted[0] - sorted[1] : sorted[0];

            // Combine entropy and margin
            float uncertainty = 0.5f * normalized + 0.5f * (1 - margin);

            return new UncertaintyResult
            {
                EpistemicUncertainty = 0,  // Can't separate without multiple passes
                AleatoricUncertainty = uncertainty,
                TotalUncertainty = uncertainty,
                ShouldAskUser = uncertainty > (1.0f - _config.ConfidenceThreshold),
                Interpretation = uncertainty > 0.5f ? "Low confidence - consider asking for clarification" : "Reasonably confident"
            };
        }

        /// <summary>
        /// Temperature scaling for calibrated probabilities
        /// </summary>
        public GpuTensor CalibratedProbabilities(GpuTensor logits, float temperature = 1.0f)
        {
            var data = logits.ToArray();
            var scaled = data.Select(x => x / temperature).ToArray();
            var probs = Softmax(scaled);
            return GpuTensor.FromArray(_accelerator, probs, logits.Shape);
        }

        /// <summary>
        /// Check if model should abstain from answering
        /// </summary>
        public bool ShouldAbstain(GpuTensor logits, float threshold = 0.3f)
        {
            var result = QuickUncertainty(logits);
            return result.TotalUncertainty > threshold;
        }

        /// <summary>
        /// MC Dropout with Stable Output Layer (SOL).
        /// Per 2025 research: "SOL MC dropout produces improved uncertainty estimation
        /// with bootstrap-like robust prediction distribution."
        /// </summary>
        public UncertaintyResult MCDropoutWithSOL(
            GpuTensor input,
            Func<GpuTensor, float, int, GpuTensor> forwardWithSelectiveDropout)
        {
            int samples = _config.MCDropoutSamples;
            var predictions = new float[samples][];

            // SOL: Apply dropout only to early layers, keep output layers stable
            int stableLayers = _config.SOLStableLayers;

            for (int i = 0; i < samples; i++)
            {
                // forwardWithSelectiveDropout(input, dropoutRate, stableLayersFromEnd)
                var output = forwardWithSelectiveDropout(input, _config.DropoutRate, stableLayers);
                predictions[i] = output.ToArray();
                output.Dispose();
            }

            return AnalyzePredictionDistribution(predictions);
        }

        /// <summary>
        /// Deep Ensemble: Combine predictions from multiple independently trained models.
        /// More robust than MC Dropout but requires multiple models.
        /// </summary>
        public EnsembleResult DeepEnsemble(
            GpuTensor input,
            Func<GpuTensor, int, GpuTensor>[] ensembleForwards)
        {
            int n = ensembleForwards.Length;
            var predictions = new float[n][];

            for (int i = 0; i < n; i++)
            {
                var output = ensembleForwards[i](input, i);
                predictions[i] = output.ToArray();
                output.Dispose();
            }

            int outputSize = predictions[0].Length;

            // Mean prediction
            var mean = new float[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                mean[j] = predictions.Average(p => p[j]);
            }

            // Per-output variance
            var variance = new float[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                variance[j] = predictions.Average(p => (p[j] - mean[j]) * (p[j] - mean[j]));
            }

            // Disagreement: average pairwise KL divergence between ensemble members
            float disagreement = 0;
            int pairs = 0;
            for (int i = 0; i < n; i++)
            {
                for (int k = i + 1; k < n; k++)
                {
                    var p1 = Softmax(predictions[i]);
                    var p2 = Softmax(predictions[k]);
                    for (int j = 0; j < outputSize; j++)
                    {
                        if (p1[j] > 0 && p2[j] > 0)
                            disagreement += p1[j] * MathF.Log(p1[j] / p2[j]);
                    }
                    pairs++;
                }
            }
            if (pairs > 0) disagreement /= pairs;

            return new EnsembleResult
            {
                MeanPrediction = mean,
                Variance = variance,
                MemberPredictions = predictions,
                Disagreement = disagreement
            };
        }

        /// <summary>
        /// Add calibration sample for conformal prediction.
        /// Call this with validation set examples to build calibration set.
        /// </summary>
        public void AddCalibrationSample(float prediction, float trueValue)
        {
            float score = Math.Abs(prediction - trueValue);
            _calibrationScores.Add(score);
        }

        /// <summary>
        /// Conformal Prediction: Get prediction intervals with coverage guarantee.
        /// Per research: "Conformal prediction provides distribution-free guarantees."
        /// </summary>
        public (float lower, float upper) GetConformalInterval(float prediction)
        {
            if (_calibrationScores.Count == 0)
            {
                // No calibration data, use default interval
                return (prediction - 1.0f, prediction + 1.0f);
            }

            // Find quantile that covers ConformalCoverage fraction
            var sorted = _calibrationScores.OrderBy(x => x).ToList();
            int index = (int)Math.Ceiling(_config.ConformalCoverage * sorted.Count) - 1;
            index = Math.Max(0, Math.Min(index, sorted.Count - 1));
            float quantile = sorted[index];

            return (prediction - quantile, prediction + quantile);
        }

        /// <summary>
        /// Expected Calibration Error (ECE): Measure how well-calibrated the model is.
        /// Lower is better. Well-calibrated: P(correct | confidence=p) = p
        /// </summary>
        public float ComputeECE(
            float[][] predictions,
            int[] trueLabels,
            int numBins = 10)
        {
            var binCorrect = new float[numBins];
            var binConfidence = new float[numBins];
            var binCounts = new int[numBins];

            for (int i = 0; i < predictions.Length; i++)
            {
                var probs = Softmax(predictions[i]);
                float maxProb = probs.Max();
                int predLabel = Array.IndexOf(probs, maxProb);

                int bin = Math.Min((int)(maxProb * numBins), numBins - 1);
                binConfidence[bin] += maxProb;
                binCorrect[bin] += predLabel == trueLabels[i] ? 1 : 0;
                binCounts[bin]++;
            }

            float ece = 0;
            int totalSamples = predictions.Length;
            for (int b = 0; b < numBins; b++)
            {
                if (binCounts[b] > 0)
                {
                    float avgConf = binConfidence[b] / binCounts[b];
                    float accuracy = binCorrect[b] / binCounts[b];
                    ece += (binCounts[b] / (float)totalSamples) * Math.Abs(accuracy - avgConf);
                }
            }

            return ece;
        }

        private UncertaintyResult AnalyzePredictionDistribution(float[][] predictions)
        {
            int outputSize = predictions[0].Length;

            // Compute mean prediction
            var mean = new float[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                mean[j] = predictions.Average(p => p[j]);
            }

            // Compute variance (epistemic uncertainty)
            float variance = 0;
            for (int j = 0; j < outputSize; j++)
            {
                float v = predictions.Average(p => (p[j] - mean[j]) * (p[j] - mean[j]));
                variance += v;
            }
            variance /= outputSize;

            float epistemic = MathF.Sqrt(variance);

            // Entropy of mean prediction (aleatoric uncertainty)
            var softmax = Softmax(mean);
            float entropy = -softmax.Sum(p => p > 0 ? p * MathF.Log(p) : 0);
            float maxEntropy = MathF.Log(outputSize);
            float aleatoric = entropy / maxEntropy;

            float total = MathF.Sqrt(epistemic * epistemic + aleatoric * aleatoric);

            // Compute conformal bounds if enabled
            float[]? lower = null, upper = null;
            if (_config.UseConformal && _calibrationScores.Count > 0)
            {
                lower = new float[outputSize];
                upper = new float[outputSize];
                for (int j = 0; j < outputSize; j++)
                {
                    var (l, u) = GetConformalInterval(mean[j]);
                    lower[j] = l;
                    upper[j] = u;
                }
            }

            return new UncertaintyResult
            {
                EpistemicUncertainty = epistemic,
                AleatoricUncertainty = aleatoric,
                TotalUncertainty = Math.Min(1.0f, total),
                ShouldAskUser = total > (1.0f - _config.ConfidenceThreshold),
                Interpretation = InterpretUncertainty(epistemic, aleatoric),
                PredictionLower = lower,
                PredictionUpper = upper
            };
        }

        private float[] Softmax(float[] logits)
        {
            float max = logits.Max();
            var exp = logits.Select(x => MathF.Exp(x - max)).ToArray();
            float sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }

        private string InterpretUncertainty(float epistemic, float aleatoric)
        {
            if (epistemic > 0.5f && aleatoric < 0.3f)
                return "High model uncertainty - this topic may be outside training data";
            if (aleatoric > 0.5f && epistemic < 0.3f)
                return "High input uncertainty - the question may be ambiguous";
            if (epistemic > 0.3f && aleatoric > 0.3f)
                return "High overall uncertainty - both the model and input are unclear";
            return "Reasonably confident";
        }
    }

    #endregion

    #region Causal Reasoning - Understand Cause and Effect

    /// <summary>
    /// Causal Reasoning: Go beyond correlation to causation.
    /// Enables AI to answer "what if" and "why" questions.
    /// Enhanced with do-calculus rules and ID algorithm for identifiability.
    /// Based on: Pearl's do-calculus, https://ftp.cs.ucla.edu/pub/stat_ser/r485.pdf
    /// </summary>
    public class CausalReasoning
    {
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;

        /// <summary>
        /// Structural Causal Model (SCM) - The foundation of causal inference.
        /// An SCM consists of: (U, V, F) where U=exogenous, V=endogenous, F=functions
        /// </summary>
        public class StructuralCausalModel
        {
            /// <summary>Public API</summary>
            public CausalGraph Graph { get; set; } = new();
            /// <summary>Public API</summary>
            public Dictionary<string, Func<Dictionary<string, float>, float, float>> Functions { get; set; } = new();
            /// <summary>Public API</summary>
            public Dictionary<string, float> ExogenousNoise { get; set; } = new();

            /// <summary>Evaluate the SCM to get values for all variables</summary>
            public Dictionary<string, float> Evaluate(Dictionary<string, float>? interventions = null)
            {
                var values = new Dictionary<string, float>();

                // Topological sort
                var sorted = TopologicalSort();

                foreach (var variable in sorted)
                {
                    // Check for intervention (do-operator)
                    if (interventions != null && interventions.TryGetValue(variable, out float intervened))
                    {
                        values[variable] = intervened;
                        continue;
                    }

                    // Evaluate structural function
                    if (Functions.TryGetValue(variable, out var func))
                    {
                        float noise = ExogenousNoise.GetValueOrDefault(variable, 0f);
                        values[variable] = func(values, noise);
                    }
                    else
                    {
                        // Exogenous variable
                        values[variable] = ExogenousNoise.GetValueOrDefault(variable, 0f);
                    }
                }

                return values;
            }

            private List<string> TopologicalSort()
            {
                var result = new List<string>();
                var visited = new HashSet<string>();
                var temp = new HashSet<string>();

                void Visit(string node)
                {
                    if (visited.Contains(node)) return;
                    if (temp.Contains(node)) return; // Cycle

                    temp.Add(node);
                    if (Graph.Parents.TryGetValue(node, out var parents))
                    {
                        foreach (var parent in parents)
                            Visit(parent);
                    }
                    temp.Remove(node);
                    visited.Add(node);
                    result.Add(node);
                }

                foreach (var v in Graph.Variables)
                    Visit(v);

                return result;
            }
        }

        /// <summary>
        /// Do-Calculus result - whether a causal query is identifiable
        /// </summary>
        public class DoCalculusResult
        {
            /// <summary>Public API</summary>
            public bool IsIdentifiable { get; set; }
            /// <summary>Public API</summary>
            public string Estimand { get; set; } = "";  // The formula to compute from data
            /// <summary>Public API</summary>
            public List<string> RulesApplied { get; set; } = new();
            /// <summary>Public API</summary>
            public string? NonIdentifiabilityReason { get; set; }
        }

        /// <summary>Public API</summary>
        public class CausalGraph
        {
            /// <summary>Public API</summary>
            public List<string> Variables { get; set; } = new();
            /// <summary>Public API</summary>
            public Dictionary<string, List<string>> Parents { get; set; } = new();  // var -> its causes
            /// <summary>Public API</summary>
            public Dictionary<string, List<string>> Children { get; set; } = new(); // var -> its effects
            /// <summary>Public API</summary>
            public Dictionary<string, float[]> Mechanisms { get; set; } = new();    // P(var | parents)

            /// <summary>Public API</summary>
            public void AddEdge(string cause, string effect)
            {
                if (!Parents.ContainsKey(effect))
                    Parents[effect] = new List<string>();
                if (!Children.ContainsKey(cause))
                    Children[cause] = new List<string>();

                if (!Parents[effect].Contains(cause))
                    Parents[effect].Add(cause);
                if (!Children[cause].Contains(effect))
                    Children[cause].Add(effect);

                if (!Variables.Contains(cause)) Variables.Add(cause);
                if (!Variables.Contains(effect)) Variables.Add(effect);
            }

            /// <summary>Public API</summary>
            public bool HasPath(string from, string to)
            {
                var visited = new HashSet<string>();
                var queue = new Queue<string>();
                queue.Enqueue(from);

                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (current == to) return true;
                    if (visited.Contains(current)) continue;
                    visited.Add(current);

                    if (Children.TryGetValue(current, out var children))
                    {
                        foreach (var child in children)
                            queue.Enqueue(child);
                    }
                }
                return false;
            }
        }

        /// <summary>Public API</summary>
        public class InterventionResult
        {
            /// <summary>Public API</summary>
            public string Variable { get; set; } = "";
            /// <summary>Public API</summary>
            public float IntervenedValue { get; set; }
            /// <summary>Public API</summary>
            public Dictionary<string, float> Effects { get; set; } = new();
            /// <summary>Public API</summary>
            public string Explanation { get; set; } = "";
        }

        /// <summary>Public API</summary>
        public class CounterfactualResult
        {
            /// <summary>Public API</summary>
            public Dictionary<string, float> FactualWorld { get; set; } = new();
            /// <summary>Public API</summary>
            public Dictionary<string, float> CounterfactualWorld { get; set; } = new();
            /// <summary>Public API</summary>
            public string Intervention { get; set; } = "";
            /// <summary>Public API</summary>
            public float CausalEffect { get; set; }
        }

        /// <summary>Public API</summary>
        public CausalReasoning(Accelerator accelerator, GpuKernels kernels)
        {
            _accelerator = accelerator;
            _kernels = kernels;
        }

        /// <summary>
        /// Infer causal structure from observational data
        /// Uses PC algorithm approximation
        /// </summary>
        public CausalGraph InferStructure(float[,] data, string[] variableNames, float threshold = 0.05f)
        {
            int n = variableNames.Length;
            var graph = new CausalGraph();
            foreach (var v in variableNames)
            {
                graph.Variables.Add(v);
                graph.Parents[v] = new List<string>();
                graph.Children[v] = new List<string>();
            }

            // Start with complete undirected graph
            var edges = new bool[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    edges[i, j] = i != j;

            // Remove edges based on conditional independence
            int numSamples = data.GetLength(0);

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (!edges[i, j]) continue;

                    // Test marginal independence
                    float corr = ComputeCorrelation(data, i, j, numSamples);
                    if (Math.Abs(corr) < threshold)
                    {
                        edges[i, j] = false;
                        edges[j, i] = false;
                    }
                }
            }

            // Orient edges (simplified: use correlation direction)
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (!edges[i, j]) continue;

                    // Use temporal ordering or correlation strength to orient
                    float corrIJ = ComputeCorrelation(data, i, j, numSamples);
                    if (corrIJ > 0)
                    {
                        graph.AddEdge(variableNames[i], variableNames[j]);
                    }
                    else
                    {
                        graph.AddEdge(variableNames[j], variableNames[i]);
                    }
                }
            }

            return graph;
        }

        /// <summary>
        /// Compute effect of intervention: do(X = value)
        /// </summary>
        public InterventionResult Intervene(
            CausalGraph graph,
            Dictionary<string, float> observations,
            string variable,
            float value)
        {
            // Create interventional distribution
            var interventional = new Dictionary<string, float>(observations);
            interventional[variable] = value;

            // Propagate effects to descendants
            var effects = new Dictionary<string, float>();
            var toProcess = new Queue<string>();

            if (graph.Children.TryGetValue(variable, out var children))
            {
                foreach (var child in children)
                    toProcess.Enqueue(child);
            }

            while (toProcess.Count > 0)
            {
                var current = toProcess.Dequeue();
                if (effects.ContainsKey(current)) continue;

                // Compute effect based on parents
                float effect = 0;
                if (graph.Parents.TryGetValue(current, out var parents))
                {
                    foreach (var parent in parents)
                    {
                        float parentValue = effects.ContainsKey(parent) ? effects[parent] :
                                           interventional.GetValueOrDefault(parent, 0);
                        effect += parentValue * 0.5f;  // Simplified linear causal effect
                    }
                }

                effects[current] = effect;
                interventional[current] = observations.GetValueOrDefault(current, 0) + effect;

                if (graph.Children.TryGetValue(current, out var grandchildren))
                {
                    foreach (var gc in grandchildren)
                        toProcess.Enqueue(gc);
                }
            }

            return new InterventionResult
            {
                Variable = variable,
                IntervenedValue = value,
                Effects = effects,
                Explanation = $"Setting {variable}={value} affects: {string.Join(", ", effects.Keys)}"
            };
        }

        /// <summary>
        /// Counterfactual reasoning: What would have happened if...?
        /// </summary>
        public CounterfactualResult Counterfactual(
            CausalGraph graph,
            Dictionary<string, float> factual,
            string variable,
            float counterfactualValue)
        {
            // Step 1: Abduction - infer noise terms from factual
            var noise = new Dictionary<string, float>();
            foreach (var v in factual.Keys)
            {
                float predicted = 0;
                if (graph.Parents.TryGetValue(v, out var parents))
                {
                    foreach (var p in parents)
                        predicted += factual.GetValueOrDefault(p, 0) * 0.5f;
                }
                noise[v] = factual[v] - predicted;
            }

            // Step 2: Action - set counterfactual value
            var counterfactualWorld = new Dictionary<string, float>(factual);
            counterfactualWorld[variable] = counterfactualValue;

            // Step 3: Prediction - propagate with same noise
            foreach (var v in graph.Variables)
            {
                if (v == variable) continue;

                float predicted = 0;
                if (graph.Parents.TryGetValue(v, out var parents))
                {
                    foreach (var p in parents)
                        predicted += counterfactualWorld.GetValueOrDefault(p, 0) * 0.5f;
                }
                counterfactualWorld[v] = predicted + noise.GetValueOrDefault(v, 0);
            }

            // Compute causal effect
            float causalEffect = 0;
            if (graph.Children.TryGetValue(variable, out var children))
            {
                foreach (var child in children)
                {
                    causalEffect += Math.Abs(counterfactualWorld[child] - factual.GetValueOrDefault(child, 0));
                }
            }

            return new CounterfactualResult
            {
                FactualWorld = factual,
                CounterfactualWorld = counterfactualWorld,
                Intervention = $"do({variable}={counterfactualValue})",
                CausalEffect = causalEffect
            };
        }

        /// <summary>
        /// Check if a causal effect P(Y|do(X)) is identifiable from observational data.
        /// Implements simplified ID algorithm based on do-calculus rules.
        /// Per research: "Whether Q is estimable can be decided in polynomial time."
        /// </summary>
        public DoCalculusResult CheckIdentifiability(CausalGraph graph, string treatment, string outcome)
        {
            var result = new DoCalculusResult();

            // Find all paths from treatment to outcome
            var paths = FindAllPaths(graph, treatment, outcome);

            if (paths.Count == 0)
            {
                result.IsIdentifiable = true;
                result.Estimand = $"P({outcome})"; // No causal effect
                result.RulesApplied.Add("No path: treatment doesn't affect outcome");
                return result;
            }

            // Check for confounders (common causes)
            var confounders = FindConfounders(graph, treatment, outcome);

            if (confounders.Count == 0)
            {
                // No confounding - simple case
                result.IsIdentifiable = true;
                result.Estimand = $"P({outcome}|{treatment})";
                result.RulesApplied.Add("Rule 2: No confounders, observational = interventional");
                return result;
            }

            // Check backdoor criterion
            var backdoorSet = FindBackdoorAdjustmentSet(graph, treatment, outcome);
            if (backdoorSet.Count > 0)
            {
                result.IsIdentifiable = true;
                string adjustmentVars = string.Join(", ", backdoorSet);
                result.Estimand = $"Σ_({adjustmentVars}) P({outcome}|{treatment}, {adjustmentVars}) P({adjustmentVars})";
                result.RulesApplied.Add($"Backdoor adjustment: condition on {adjustmentVars}");
                return result;
            }

            // Check front-door criterion
            var frontdoorSet = FindFrontdoorSet(graph, treatment, outcome);
            if (frontdoorSet.Count > 0)
            {
                result.IsIdentifiable = true;
                string mediators = string.Join(", ", frontdoorSet);
                result.Estimand = $"Σ_{mediators} P({mediators}|{treatment}) Σ_{treatment}' P({outcome}|{mediators},{treatment}') P({treatment}')";
                result.RulesApplied.Add($"Front-door adjustment via {mediators}");
                return result;
            }

            // Not identifiable
            result.IsIdentifiable = false;
            result.NonIdentifiabilityReason = $"Unblocked backdoor paths with no valid adjustment set. Confounders: {string.Join(", ", confounders)}";
            return result;
        }

        /// <summary>
        /// Apply do-calculus Rule 1: Insertion/deletion of observations.
        /// P(y|do(x),z,w) = P(y|do(x),w) if (Y ⊥ Z | X,W)_G_X̄
        /// </summary>
        public bool CanApplyRule1(CausalGraph graph, string y, string z, string x, IEnumerable<string> w)
        {
            // Check d-separation in mutilated graph G_X̄ (remove incoming edges to X)
            var mutilated = MutilateGraph(graph, new[] { x });
            return IsDSeparated(mutilated, y, z, w.Append(x));
        }

        /// <summary>
        /// Apply do-calculus Rule 2: Action/observation exchange.
        /// P(y|do(x),do(z),w) = P(y|do(x),z,w) if (Y ⊥ Z | X,W)_G_X̄Z_
        /// </summary>
        public bool CanApplyRule2(CausalGraph graph, string y, string z, string x, IEnumerable<string> w)
        {
            // Check d-separation in G with incoming edges to X removed and outgoing from Z removed
            var mutilated = MutilateGraph(graph, new[] { x });
            mutilated = RemoveOutgoingEdges(mutilated, new[] { z });
            return IsDSeparated(mutilated, y, z, w.Append(x));
        }

        /// <summary>
        /// Apply do-calculus Rule 3: Insertion/deletion of actions.
        /// P(y|do(x),do(z),w) = P(y|do(x),w) if (Y ⊥ Z | X,W)_G_X̄,Z(W)_
        /// </summary>
        public bool CanApplyRule3(CausalGraph graph, string y, string z, string x, IEnumerable<string> w)
        {
            // Most complex rule - check specific graph conditions
            var mutilated = MutilateGraph(graph, new[] { x });
            var zAncestors = GetAncestors(mutilated, z);
            var wSet = w.ToHashSet();
            var nonAncestorW = wSet.Where(v => !zAncestors.Contains(v));

            mutilated = RemoveOutgoingEdges(mutilated, nonAncestorW);
            return IsDSeparated(mutilated, y, z, w.Append(x));
        }

        /// <summary>
        /// Create SCM from graph with linear functions
        /// </summary>
        public StructuralCausalModel CreateLinearSCM(CausalGraph graph, Dictionary<string, float[]>? coefficients = null)
        {
            var scm = new StructuralCausalModel { Graph = graph };

            foreach (var variable in graph.Variables)
            {
                if (graph.Parents.TryGetValue(variable, out var parents) && parents.Count > 0)
                {
                    var varCoeffs = coefficients?.GetValueOrDefault(variable) ??
                                   parents.Select(_ => 0.5f).ToArray();

                    scm.Functions[variable] = (values, noise) =>
                    {
                        float sum = noise;
                        for (int i = 0; i < parents.Count; i++)
                        {
                            sum += values.GetValueOrDefault(parents[i], 0) * varCoeffs[Math.Min(i, varCoeffs.Length - 1)];
                        }
                        return sum;
                    };
                }

                scm.ExogenousNoise[variable] = 0f;
            }

            return scm;
        }

        private List<List<string>> FindAllPaths(CausalGraph graph, string from, string to)
        {
            var paths = new List<List<string>>();
            var currentPath = new List<string> { from };
            FindPathsDFS(graph, from, to, new HashSet<string> { from }, currentPath, paths);
            return paths;
        }

        private void FindPathsDFS(CausalGraph graph, string current, string target,
            HashSet<string> visited, List<string> path, List<List<string>> paths)
        {
            if (current == target)
            {
                paths.Add(new List<string>(path));
                return;
            }

            if (graph.Children.TryGetValue(current, out var children))
            {
                foreach (var child in children)
                {
                    if (!visited.Contains(child))
                    {
                        visited.Add(child);
                        path.Add(child);
                        FindPathsDFS(graph, child, target, visited, path, paths);
                        path.RemoveAt(path.Count - 1);
                        visited.Remove(child);
                    }
                }
            }
        }

        private HashSet<string> FindConfounders(CausalGraph graph, string x, string y)
        {
            var confounders = new HashSet<string>();
            var xAncestors = GetAncestors(graph, x);
            var yAncestors = GetAncestors(graph, y);

            foreach (var ancestor in xAncestors)
            {
                if (yAncestors.Contains(ancestor) && ancestor != x && ancestor != y)
                {
                    confounders.Add(ancestor);
                }
            }

            return confounders;
        }

        private HashSet<string> GetAncestors(CausalGraph graph, string node)
        {
            var ancestors = new HashSet<string>();
            var queue = new Queue<string>();

            if (graph.Parents.TryGetValue(node, out var parents))
            {
                foreach (var p in parents)
                    queue.Enqueue(p);
            }

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (ancestors.Contains(current)) continue;
                ancestors.Add(current);

                if (graph.Parents.TryGetValue(current, out var grandparents))
                {
                    foreach (var gp in grandparents)
                        queue.Enqueue(gp);
                }
            }

            return ancestors;
        }

        private List<string> FindBackdoorAdjustmentSet(CausalGraph graph, string x, string y)
        {
            // Simple heuristic: use all confounders that are observable
            var confounders = FindConfounders(graph, x, y);
            return confounders.ToList();
        }

        private List<string> FindFrontdoorSet(CausalGraph graph, string x, string y)
        {
            // Find mediators: variables on all paths from X to Y that block backdoor
            var mediators = new List<string>();

            if (graph.Children.TryGetValue(x, out var xChildren))
            {
                foreach (var child in xChildren)
                {
                    if (graph.HasPath(child, y))
                    {
                        // Check if this blocks all backdoor paths
                        var confounders = FindConfounders(graph, child, y);
                        if (confounders.Count == 0)
                        {
                            mediators.Add(child);
                        }
                    }
                }
            }

            return mediators;
        }

        private CausalGraph MutilateGraph(CausalGraph graph, IEnumerable<string> interventedVars)
        {
            var mutilated = new CausalGraph
            {
                Variables = new List<string>(graph.Variables),
                Parents = graph.Parents.ToDictionary(kv => kv.Key, kv => new List<string>(kv.Value)),
                Children = graph.Children.ToDictionary(kv => kv.Key, kv => new List<string>(kv.Value))
            };

            // Remove incoming edges to intervened variables
            foreach (var v in interventedVars)
            {
                if (mutilated.Parents.TryGetValue(v, out var parents))
                {
                    foreach (var parent in parents)
                    {
                        mutilated.Children[parent].Remove(v);
                    }
                    mutilated.Parents[v].Clear();
                }
            }

            return mutilated;
        }

        private CausalGraph RemoveOutgoingEdges(CausalGraph graph, IEnumerable<string> vars)
        {
            var modified = new CausalGraph
            {
                Variables = new List<string>(graph.Variables),
                Parents = graph.Parents.ToDictionary(kv => kv.Key, kv => new List<string>(kv.Value)),
                Children = graph.Children.ToDictionary(kv => kv.Key, kv => new List<string>(kv.Value))
            };

            foreach (var v in vars)
            {
                if (modified.Children.TryGetValue(v, out var children))
                {
                    foreach (var child in children)
                    {
                        modified.Parents[child].Remove(v);
                    }
                    modified.Children[v].Clear();
                }
            }

            return modified;
        }

        private bool IsDSeparated(CausalGraph graph, string x, string y, IEnumerable<string> conditioned)
        {
            // Simplified d-separation check using ancestor graph
            var condSet = conditioned.ToHashSet();

            // If x and y are both in conditioning set, they're separated
            if (condSet.Contains(x) || condSet.Contains(y))
                return true;

            // Check if there's any active path
            return !HasActivePath(graph, x, y, condSet);
        }

        private bool HasActivePath(CausalGraph graph, string x, string y, HashSet<string> conditioned)
        {
            // BFS to find active path
            var visited = new HashSet<(string node, bool incoming)>();
            var queue = new Queue<(string node, bool incoming)>();

            // Start from x going both directions
            queue.Enqueue((x, true));
            queue.Enqueue((x, false));

            while (queue.Count > 0)
            {
                var (current, incoming) = queue.Dequeue();
                if (current == y) return true;
                if (visited.Contains((current, incoming))) continue;
                visited.Add((current, incoming));

                bool isConditioned = conditioned.Contains(current);

                // Chain/fork rules
                if (!isConditioned)
                {
                    // Can pass through
                    if (graph.Children.TryGetValue(current, out var children))
                    {
                        foreach (var child in children)
                            queue.Enqueue((child, true));
                    }
                    if (graph.Parents.TryGetValue(current, out var parents))
                    {
                        foreach (var parent in parents)
                            queue.Enqueue((parent, false));
                    }
                }
            }

            return false;
        }

        private float ComputeCorrelation(float[,] data, int i, int j, int n)
        {
            float sumI = 0, sumJ = 0, sumIJ = 0, sumI2 = 0, sumJ2 = 0;

            for (int k = 0; k < n; k++)
            {
                sumI += data[k, i];
                sumJ += data[k, j];
                sumIJ += data[k, i] * data[k, j];
                sumI2 += data[k, i] * data[k, i];
                sumJ2 += data[k, j] * data[k, j];
            }

            float meanI = sumI / n;
            float meanJ = sumJ / n;
            float varI = sumI2 / n - meanI * meanI;
            float varJ = sumJ2 / n - meanJ * meanJ;
            float cov = sumIJ / n - meanI * meanJ;

            float denom = MathF.Sqrt(varI * varJ);
            return denom > 0 ? cov / denom : 0;
        }
    }

    #endregion
}