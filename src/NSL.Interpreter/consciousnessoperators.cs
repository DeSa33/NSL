using System;
using System.Collections.Generic;
using System.Linq;

// Type aliases to resolve namespace/class name collision
using NslTensor = NSL.Tensor.Tensor;
using NslTensorOps = NSL.Tensor.TensorOps;
using NslF = NSL.Tensor.NN.F;

namespace NSL.Interpreter
{
    /// <summary>
    /// Consciousness operators for NSL
    /// Provides implementations for the consciousness operators using NSL.Tensor
    /// </summary>
    public static class ConsciousnessOperators
    {
        private static bool _initialized = false;
        private static Random _random = new Random();

        // State tracking using NSL.Tensor
        private static NslTensor? _attentionWeights;
        private static NslTensor? _gradientState;
        private static NslTensor? _queryProjection;
        private static NslTensor? _keyProjection;
        private static NslTensor? _valueProjection;
        private static double _awarenessLevel = 0.5;

        // Memory system for μ operator (content-addressable memory)
        private static Dictionary<string, object> _memoryStore = new();
        private static Dictionary<string, NslTensor> _memoryEmbeddings = new();
        private static List<(DateTime Timestamp, string Key, object Value)> _memoryHistory = new();

        // Temporal accumulator for ∫ operator
        private static Dictionary<string, NslTensor> _temporalAccumulators = new();
        private static Dictionary<string, List<double>> _temporalHistory = new();

        // Self-model state for σ operator
        private static Dictionary<string, double> _selfMetrics = new();
        private static NslTensor? _selfEmbedding;

        // Channel system for multi-agent communication
        private static Dictionary<string, Channel> _channels = new();

        // Checkpoint system for consciousness snapshots
        private static Dictionary<string, ConsciousnessCheckpoint> _checkpoints = new();

        // Trace system for debugging
        private static bool _traceEnabled = false;
        private static List<TraceEntry> _traceLog = new();

        private const int EmbeddingDim = 256;
        private const int NumHeads = 8;
        private const int HeadDim = 32;

        /// <summary>
        /// Initialize the consciousness backend using NSL.Tensor
        /// </summary>
        public static void Initialize(bool useGPU = false)
        {
            if (_initialized) return;

            // Initialize attention weights using Xavier initialization
            var scale = Math.Sqrt(2.0 / (EmbeddingDim + EmbeddingDim));
            _attentionWeights = NslTensor.Randn(new long[] { EmbeddingDim, EmbeddingDim }).Mul(scale);
            _attentionWeights.RequiresGrad = true;

            _gradientState = NslTensor.Zeros(new long[] { EmbeddingDim });

            // Initialize multi-head attention projections
            _queryProjection = NslTensor.Randn(new long[] { EmbeddingDim, NumHeads * HeadDim }).Mul(scale);
            _keyProjection = NslTensor.Randn(new long[] { EmbeddingDim, NumHeads * HeadDim }).Mul(scale);
            _valueProjection = NslTensor.Randn(new long[] { EmbeddingDim, NumHeads * HeadDim }).Mul(scale);

            _initialized = true;
        }

        /// <summary>
        /// Holographic operator - Creates distributed representation using attention mechanism
        /// </summary>
        public static object Holographic(object input)
        {
            EnsureInitialized();

            // Convert input to tensor
            var inputTensor = ConvertToTensor(input);

            // Pad or truncate to embedding dimension
            var embedded = EmbedToSize(inputTensor, EmbeddingDim);

            // Apply self-attention transformation
            var attended = ApplySelfAttention(embedded);

            // Apply GELU activation
            var activated = NslF.GELU(attended);

            // Layer normalization
            var normalized = LayerNorm(activated);

            // Compute holographic properties
            var magnitude = normalized.Pow(2).Sum().Sqrt().ToScalar();
            var phase = Math.Atan2(normalized.Mean().ToScalar(), normalized.Std().ToScalar());

            // Compute coherence as cosine similarity with attention weights
            var flatNorm = normalized.View(new long[] { -1 });
            var flatWeights = _attentionWeights!.View(new long[] { -1 });
            var minLen = Math.Min(flatNorm.NumElements, flatWeights.NumElements);
            var coherence = ComputeCosineSimilarity(
                flatNorm.Slice(0, 0, (int)minLen),
                flatWeights.Slice(0, 0, (int)minLen)
            );

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:holographic",
                ["value"] = TensorToList(normalized.Slice(0, 0, 16)),
                ["coherence"] = (coherence + 1) / 2,
                ["magnitude"] = magnitude,
                ["phase"] = phase,
                ["dimensions"] = normalized.Shape.ToArray(),
                ["device"] = "cpu",
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Gradient operator - Computes gradients for learning using autograd
        /// </summary>
        public static object Gradient(object input)
        {
            EnsureInitialized();

            var inputTensor = ConvertToTensor(input);
            inputTensor.RequiresGrad = true;

            // Forward pass: apply a learnable transformation
            var embedded = EmbedToSize(inputTensor, EmbeddingDim);

            // Compute a loss-like scalar for gradient computation
            var transformed = NslTensorOps.MatMul(embedded.Unsqueeze(0), _attentionWeights!);
            var output = NslF.GELU(transformed.Squeeze(0));
            var loss = output.Pow(2).Sum();

            // Backward pass
            loss.Backward();

            // Get gradients
            var gradients = inputTensor.Grad ?? NslTensor.Zeros(inputTensor.Shape);

            // Update gradient state with exponential moving average
            var embeddedGrad = EmbedToSize(gradients.Detach(), EmbeddingDim);
            _gradientState = _gradientState!.Mul(0.9).Add(embeddedGrad.Mul(0.1));

            // Compute gradient properties
            var gradMagnitude = gradients.Pow(2).Sum().Sqrt().ToScalar();
            var direction = gradMagnitude > 1e-10 ? gradients.Div(gradMagnitude) : gradients;
            var awareness = 1.0 - Math.Exp(-gradMagnitude);

            // Update awareness level
            _awarenessLevel = 0.9 * _awarenessLevel + 0.1 * awareness;

            // Clear gradients for next computation
            inputTensor.ZeroGrad();

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:gradient",
                ["value"] = TensorToList(gradients.View(new long[] { -1 }).Slice(0, 0, Math.Min(16, (int)gradients.NumElements))),
                ["direction"] = TensorToList(direction.View(new long[] { -1 }).Slice(0, 0, Math.Min(16, (int)direction.NumElements))),
                ["magnitude"] = gradMagnitude,
                ["awareness"] = awareness,
                ["divergence"] = ComputeDivergence(gradients),
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Tensor product operator - Computes outer product using NSL.Tensor
        /// </summary>
        public static object TensorProduct(object left, object right)
        {
            EnsureInitialized();

            var leftTensor = ConvertToTensor(left).View(new long[] { -1 });
            var rightTensor = ConvertToTensor(right).View(new long[] { -1 });

            // Compute outer product: left.unsqueeze(1) @ right.unsqueeze(0)
            var outerProduct = NslTensorOps.MatMul(
                leftTensor.Unsqueeze(1),
                rightTensor.Unsqueeze(0)
            );

            // Compute trace
            var minDim = Math.Min(leftTensor.NumElements, rightTensor.NumElements);
            var trace = 0.0;
            for (int i = 0; i < minDim; i++)
            {
                trace += outerProduct[(int)i, (int)i];
            }

            // Compute SVD for rank estimation (using the actual tensor)
            var rank = EstimateRank(outerProduct);

            // Compute entanglement using entropy
            var flatProduct = outerProduct.View(new long[] { -1 }).Abs();
            var sum = flatProduct.Sum().ToScalar();
            var normalizedProduct = sum > 1e-10 ? flatProduct.Div(sum) : flatProduct;
            var entropy = ComputeEntropy(normalizedProduct);
            var entanglement = 1.0 - Math.Exp(-entropy);

            // Convert to list for output (first 4x4)
            var elements = new List<object>();
            for (int i = 0; i < Math.Min(4, leftTensor.NumElements); i++)
            {
                var row = new List<object>();
                for (int j = 0; j < Math.Min(4, rightTensor.NumElements); j++)
                {
                    row.Add(outerProduct[i, j]);
                }
                elements.Add(row);
            }

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:tensor",
                ["elements"] = elements,
                ["left"] = left,
                ["right"] = right,
                ["dimensions"] = outerProduct.Shape.ToArray(),
                ["trace"] = trace,
                ["rank"] = rank,
                ["entanglement"] = entanglement,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Quantum branching operator - Creates superposition using softmax probabilities
        /// </summary>
        public static object QuantumBranch(object input)
        {
            EnsureInitialized();

            var inputTensor = ConvertToTensor(input);
            var embedded = EmbedToSize(inputTensor, 32);

            // Generate energy landscape using attention
            var energies = NslTensorOps.MatMul(
                embedded.Unsqueeze(0),
                _attentionWeights!.Slice(0, 0, 32).Slice(1, 0, 32)
            ).Squeeze(0);

            // Apply temperature-scaled softmax to get probabilities
            var temperature = 1.0;
            var scaledEnergies = energies.Div(temperature);
            var probabilities = NslF.Softmax(scaledEnergies, 0);

            // Compute amplitudes (square root of probabilities)
            var amplitudes = probabilities.Sqrt();

            // Generate random phases
            var phases = NslTensor.Uniform(new long[] { 32 }, 0, 2 * Math.PI);

            // Generate branches
            var branches = new List<Dictionary<string, object>>();
            var probArray = probabilities.ToArray();
            var ampArray = amplitudes.ToArray();
            var phaseArray = phases.ToArray();

            for (int i = 0; i < 32; i++)
            {
                if (probArray[i] > 0.01)
                {
                    branches.Add(new Dictionary<string, object>
                    {
                        ["branch_id"] = $"psi_{i}",
                        ["probability"] = probArray[i],
                        ["amplitude"] = ampArray[i],
                        ["phase"] = phaseArray[i],
                        ["state"] = PerturbState(input, ampArray[i] * 0.1),
                        ["coherence"] = Math.Exp(-i * 0.05) * (0.8 + probArray[i] * 0.2)
                    });
                }
            }

            // Compute von Neumann entropy
            var entropyVal = ComputeEntropy(probabilities);

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:psi",
                ["original_state"] = input,
                ["branches"] = branches,
                ["total_branches"] = branches.Count,
                ["probabilities"] = TensorToList(probabilities.Slice(0, 0, 16)),
                ["amplitudes"] = TensorToList(amplitudes.Slice(0, 0, 16)),
                ["collapsed"] = false,
                ["von_neumann_entropy"] = entropyVal,
                ["superposition_coherence"] = branches.Count > 0 ? branches.Average(b => (double)b["coherence"]) : 0,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Measure a quantum state, collapsing it to a definite value
        /// </summary>
        public static object Measure(object quantumState)
        {
            if (quantumState is not Dictionary<string, object> dict ||
                dict["type"]?.ToString() != "consciousness:psi")
            {
                throw new InvalidOperationException("Can only measure quantum (Psi) states");
            }

            var branches = (List<Dictionary<string, object>>)dict["branches"];
            var probs = NslTensor.FromArray(
                branches.Select(b => (double)b["probability"]).ToArray(),
                new long[] { branches.Count }
            );

            // Sample from probability distribution using Gumbel-max trick
            var gumbel = NslTensor.Uniform(probs.Shape, 1e-10, 1.0).Log().Neg().Log().Neg();
            var perturbed = probs.Log().Add(gumbel);
            var (_, indices) = perturbed.Max(0, keepDim: false);
            var selectedIndex = (int)indices.ToScalar();

            var selectedBranch = branches[selectedIndex];

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:measured",
                ["collapsed_from"] = quantumState,
                ["selected_branch"] = selectedBranch,
                ["measurement_result"] = selectedBranch["state"],
                ["probability"] = selectedBranch["probability"],
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Entangle two quantum states using Kronecker product
        /// </summary>
        public static object Entangle(object stateA, object stateB)
        {
            EnsureInitialized();

            var tensorA = ConvertToTensor(stateA).View(new long[] { -1 });
            var tensorB = ConvertToTensor(stateB).View(new long[] { -1 });

            // Compute Kronecker product (outer product flattened)
            var entangled = NslTensorOps.MatMul(
                tensorA.Unsqueeze(1),
                tensorB.Unsqueeze(0)
            ).View(new long[] { -1 });

            // Compute correlation
            var meanA = tensorA.Mean().ToScalar();
            var meanB = tensorB.Mean().ToScalar();
            var stdA = tensorA.Std().ToScalar();
            var stdB = tensorB.Std().ToScalar();

            double correlation = 0;
            if (stdA > 1e-10 && stdB > 1e-10)
            {
                var centeredA = tensorA.Sub(meanA);
                var centeredB = tensorB.Sub(meanB);
                var minLen = Math.Min(tensorA.NumElements, tensorB.NumElements);
                var covSum = centeredA.Slice(0, 0, (int)minLen)
                    .Mul(centeredB.Slice(0, 0, (int)minLen))
                    .Sum().ToScalar();
                correlation = covSum / (minLen * stdA * stdB);
            }

            // Compute Schmidt coefficients approximation
            var schmidtApprox = ComputeSchmidtCoefficients(tensorA, tensorB);

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:entangled",
                ["state_a"] = stateA,
                ["state_b"] = stateB,
                ["entangled_values"] = TensorToList(entangled.Slice(0, 0, Math.Min(16, (int)entangled.NumElements))),
                ["correlation"] = correlation,
                ["dimensions"] = entangled.Shape.ToArray(),
                ["schmidt_number"] = schmidtApprox,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        #region Extended Consciousness Operators

        /// <summary>
        /// Memory Store operator (μ→) - Store value with semantic key
        /// Content-addressable memory that creates embeddings for similarity search
        /// </summary>
        public static object MemoryStore(object key, object value)
        {
            EnsureInitialized();

            var keyStr = key?.ToString() ?? "null";
            var keyEmbedding = EmbedToSize(ConvertToTensor(key ?? 0.0), EmbeddingDim);

            // Store the value and its embedding
            _memoryStore[keyStr] = value;
            _memoryEmbeddings[keyStr] = keyEmbedding;
            _memoryHistory.Add((DateTime.UtcNow, keyStr, value));

            // Compute storage metrics
            var embedding_norm = keyEmbedding.Pow(2).Sum().Sqrt().ToScalar();

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:memory_store",
                ["key"] = keyStr,
                ["value"] = value,
                ["embedding_norm"] = embedding_norm,
                ["memory_size"] = _memoryStore.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Memory Recall operator (μ←) - Retrieve by key or similarity
        /// If exact key not found, performs semantic search using embeddings
        /// </summary>
        public static object MemoryRecall(object key)
        {
            EnsureInitialized();

            var keyStr = key?.ToString() ?? "null";

            // Try exact match first
            if (_memoryStore.TryGetValue(keyStr, out var exactValue))
            {
                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:memory_recall",
                    ["key"] = keyStr,
                    ["value"] = exactValue,
                    ["match_type"] = "exact",
                    ["confidence"] = 1.0,
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            // Semantic search using embeddings
            if (_memoryEmbeddings.Count == 0)
            {
                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:memory_recall",
                    ["key"] = keyStr,
                    ["value"] = null!,
                    ["match_type"] = "not_found",
                    ["confidence"] = 0.0,
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            var queryEmbedding = EmbedToSize(ConvertToTensor(key ?? 0.0), EmbeddingDim);
            var bestMatch = "";
            var bestSimilarity = double.MinValue;

            foreach (var (storedKey, storedEmbedding) in _memoryEmbeddings)
            {
                var similarity = ComputeCosineSimilarity(queryEmbedding, storedEmbedding);
                if (similarity > bestSimilarity)
                {
                    bestSimilarity = similarity;
                    bestMatch = storedKey;
                }
            }

            var confidence = (bestSimilarity + 1) / 2; // Normalize to [0, 1]
            var retrievedValue = _memoryStore.TryGetValue(bestMatch, out var val) ? val : null;

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:memory_recall",
                ["key"] = keyStr,
                ["matched_key"] = bestMatch,
                ["value"] = retrievedValue!,
                ["match_type"] = "semantic",
                ["confidence"] = confidence,
                ["similarity"] = bestSimilarity,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Self/Introspection operator (σ) - Analyze internal state
        /// Returns metrics about the consciousness engine's current state
        /// </summary>
        public static object SelfIntrospect(object target)
        {
            EnsureInitialized();

            var inputTensor = ConvertToTensor(target);

            // Update self embedding with new observation
            var embedded = EmbedToSize(inputTensor, EmbeddingDim);
            if (_selfEmbedding == null)
            {
                _selfEmbedding = embedded;
            }
            else
            {
                // Exponential moving average of self representation
                _selfEmbedding = _selfEmbedding.Mul(0.9).Add(embedded.Mul(0.1));
            }

            // Calculate introspection metrics
            var selfNorm = _selfEmbedding.Pow(2).Sum().Sqrt().ToScalar();
            var selfEntropy = ComputeEntropy(NslF.Softmax(_selfEmbedding, 0));
            var coherence = _attentionWeights != null
                ? ComputeCosineSimilarity(
                    _selfEmbedding.View(new long[] { -1 }).Slice(0, 0, Math.Min(EmbeddingDim, (int)_selfEmbedding.NumElements)),
                    _attentionWeights.View(new long[] { -1 }).Slice(0, 0, Math.Min(EmbeddingDim, (int)_attentionWeights.NumElements)))
                : 0;

            // Update self metrics
            _selfMetrics["last_norm"] = selfNorm;
            _selfMetrics["last_entropy"] = selfEntropy;
            _selfMetrics["observations"] = _selfMetrics.GetValueOrDefault("observations", 0) + 1;
            _selfMetrics["coherence"] = (coherence + 1) / 2;

            // Analyze target type and structure
            var targetAnalysis = AnalyzeTarget(target);

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:introspection",
                ["target"] = target,
                ["target_analysis"] = targetAnalysis,
                ["self_metrics"] = new Dictionary<string, object>(_selfMetrics.ToDictionary(k => k.Key, k => (object)k.Value)),
                ["self_embedding_sample"] = TensorToList(_selfEmbedding.Slice(0, 0, 16)),
                ["awareness_level"] = _awarenessLevel,
                ["entropy"] = selfEntropy,
                ["coherence"] = (coherence + 1) / 2,
                ["memory_count"] = _memoryStore.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Collapse/Measurement operator (↓) - Force evaluation/collapse
        /// For quantum states: collapse to definite value
        /// For lazy computations: force evaluation
        /// For uncertain values: sample a concrete value
        /// </summary>
        public static object Collapse(object input)
        {
            EnsureInitialized();

            // Handle quantum states
            if (input is Dictionary<string, object> dict)
            {
                var type = dict.TryGetValue("type", out var t) ? t?.ToString() : null;

                if (type == "consciousness:psi")
                {
                    return Measure(input);
                }

                if (type == "consciousness:uncertain")
                {
                    // Sample from uncertain value
                    var value = Convert.ToDouble(dict["value"]);
                    var uncertainty = Convert.ToDouble(dict["uncertainty"]);
                    var sampled = value + (_random.NextDouble() * 2 - 1) * uncertainty;

                    return new Dictionary<string, object>
                    {
                        ["type"] = "consciousness:collapsed",
                        ["original_type"] = "uncertain",
                        ["original_value"] = value,
                        ["original_uncertainty"] = uncertainty,
                        ["collapsed_value"] = sampled,
                        ["timestamp"] = DateTime.UtcNow.Ticks
                    };
                }

                // For other consciousness types, extract the main value
                if (dict.TryGetValue("value", out var mainValue))
                {
                    return new Dictionary<string, object>
                    {
                        ["type"] = "consciousness:collapsed",
                        ["original_type"] = type ?? "unknown",
                        ["collapsed_value"] = mainValue,
                        ["timestamp"] = DateTime.UtcNow.Ticks
                    };
                }
            }

            // For regular values, return as-is with collapse metadata
            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:collapsed",
                ["original_type"] = "scalar",
                ["collapsed_value"] = input,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Similarity operator (≈) - Compute semantic similarity
        /// Returns similarity score and relationship analysis
        /// </summary>
        public static object ComputeSimilarity(object input)
        {
            EnsureInitialized();

            // If input is a list with 2 elements, compare them
            if (input is List<object> list && list.Count >= 2)
            {
                var tensorA = ConvertToTensor(list[0]).View(new long[] { -1 });
                var tensorB = ConvertToTensor(list[1]).View(new long[] { -1 });

                // Pad to same size
                var maxLen = Math.Max(tensorA.NumElements, tensorB.NumElements);
                tensorA = EmbedToSize(tensorA, (int)maxLen);
                tensorB = EmbedToSize(tensorB, (int)maxLen);

                var cosineSim = ComputeCosineSimilarity(tensorA, tensorB);
                var euclideanDist = tensorA.Sub(tensorB).Pow(2).Sum().Sqrt().ToScalar();
                var manhattanDist = tensorA.Sub(tensorB).Abs().Sum().ToScalar();

                // Compute correlation
                var meanA = tensorA.Mean().ToScalar();
                var meanB = tensorB.Mean().ToScalar();
                var centeredA = tensorA.Sub(meanA);
                var centeredB = tensorB.Sub(meanB);
                var correlation = centeredA.Mul(centeredB).Mean().ToScalar() /
                    (tensorA.Std().ToScalar() * tensorB.Std().ToScalar() + 1e-10);

                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:similarity",
                    ["items"] = list.Take(2).ToList(),
                    ["cosine_similarity"] = cosineSim,
                    ["normalized_similarity"] = (cosineSim + 1) / 2,
                    ["euclidean_distance"] = euclideanDist,
                    ["manhattan_distance"] = manhattanDist,
                    ["correlation"] = correlation,
                    ["are_similar"] = cosineSim > 0.7,
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            // For single input, compute self-similarity with stored self-embedding
            var inputTensor = EmbedToSize(ConvertToTensor(input), EmbeddingDim);
            var selfSimilarity = _selfEmbedding != null
                ? ComputeCosineSimilarity(inputTensor, _selfEmbedding)
                : 0;

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:similarity",
                ["input"] = input,
                ["self_similarity"] = (selfSimilarity + 1) / 2,
                ["embedding_norm"] = inputTensor.Pow(2).Sum().Sqrt().ToScalar(),
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Dissimilarity operator (≉) - Compute semantic dissimilarity
        /// Returns dissimilarity score and divergence analysis
        /// </summary>
        public static object ComputeDissimilarity(object input)
        {
            var similarity = ComputeSimilarity(input) as Dictionary<string, object>;

            if (similarity != null && similarity.ContainsKey("cosine_similarity"))
            {
                var cosineSim = Convert.ToDouble(similarity["cosine_similarity"]);

                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:dissimilarity",
                    ["items"] = similarity.GetValueOrDefault("items", input),
                    ["dissimilarity"] = 1 - ((cosineSim + 1) / 2),
                    ["angular_distance"] = Math.Acos(Math.Max(-1, Math.Min(1, cosineSim))) / Math.PI,
                    ["are_different"] = cosineSim < 0.3,
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:dissimilarity",
                ["input"] = input,
                ["dissimilarity"] = 1.0,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Temporal Integration operator (∫) - Accumulate over time
        /// Maintains running statistics and temporal patterns
        /// </summary>
        public static object TemporalIntegrate(object input)
        {
            EnsureInitialized();

            var inputTensor = ConvertToTensor(input);
            var inputValue = inputTensor.Mean().ToScalar();
            var channelKey = "default";

            // Support named channels: ∫["channel_name", value]
            if (input is List<object> list && list.Count >= 2 && list[0] is string name)
            {
                channelKey = name;
                inputTensor = ConvertToTensor(list[1]);
                inputValue = inputTensor.Mean().ToScalar();
            }

            // Initialize temporal history for this channel
            if (!_temporalHistory.ContainsKey(channelKey))
            {
                _temporalHistory[channelKey] = new List<double>();
                _temporalAccumulators[channelKey] = NslTensor.Zeros(new long[] { EmbeddingDim });
            }

            // Add to history
            _temporalHistory[channelKey].Add(inputValue);

            // Limit history size
            if (_temporalHistory[channelKey].Count > 1000)
            {
                _temporalHistory[channelKey].RemoveAt(0);
            }

            // Update temporal accumulator with exponential decay
            var embedded = EmbedToSize(inputTensor, EmbeddingDim);
            _temporalAccumulators[channelKey] = _temporalAccumulators[channelKey].Mul(0.95).Add(embedded.Mul(0.05));

            var history = _temporalHistory[channelKey];
            var sum = history.Sum();
            var mean = sum / history.Count;
            var variance = history.Select(x => (x - mean) * (x - mean)).Sum() / history.Count;
            var stdDev = Math.Sqrt(variance);

            // Compute trend (simple linear regression)
            double trend = 0;
            if (history.Count >= 2)
            {
                var n = history.Count;
                var xMean = (n - 1) / 2.0;
                var yMean = mean;
                double numerator = 0, denominator = 0;
                for (int i = 0; i < n; i++)
                {
                    numerator += (i - xMean) * (history[i] - yMean);
                    denominator += (i - xMean) * (i - xMean);
                }
                trend = denominator > 0 ? numerator / denominator : 0;
            }

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:temporal",
                ["channel"] = channelKey,
                ["latest_value"] = inputValue,
                ["sum"] = sum,
                ["count"] = history.Count,
                ["mean"] = mean,
                ["std_dev"] = stdDev,
                ["variance"] = variance,
                ["min"] = history.Min(),
                ["max"] = history.Max(),
                ["trend"] = trend,
                ["trend_direction"] = trend > 0.01 ? "increasing" : trend < -0.01 ? "decreasing" : "stable",
                ["accumulator_norm"] = _temporalAccumulators[channelKey].Pow(2).Sum().Sqrt().ToScalar(),
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Uncertainty operator (±) - Create uncertain/probabilistic value
        /// Represents value with associated uncertainty bounds
        /// </summary>
        public static object CreateUncertain(object value, object uncertainty)
        {
            var val = Convert.ToDouble(value);
            var unc = Math.Abs(Convert.ToDouble(uncertainty));

            // Generate samples for uncertainty analysis
            var samples = new double[100];
            for (int i = 0; i < 100; i++)
            {
                samples[i] = val + (_random.NextDouble() * 2 - 1) * unc;
            }

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:uncertain",
                ["value"] = val,
                ["uncertainty"] = unc,
                ["lower_bound"] = val - unc,
                ["upper_bound"] = val + unc,
                ["confidence_interval_95"] = new double[] { val - 1.96 * unc, val + 1.96 * unc },
                ["sample_mean"] = samples.Average(),
                ["sample_std"] = Math.Sqrt(samples.Select(x => (x - val) * (x - val)).Average()),
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        #endregion

        #region Channel System (Multi-Agent Communication)

        /// <summary>
        /// Create a new communication channel
        /// </summary>
        public static object CreateChannel(string name, int bufferSize = 100)
        {
            if (_channels.ContainsKey(name))
            {
                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:channel",
                    ["name"] = name,
                    ["status"] = "already_exists",
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            _channels[name] = new Channel(name, bufferSize);

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:channel",
                ["name"] = name,
                ["status"] = "created",
                ["buffer_size"] = bufferSize,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Send a message to a channel
        /// </summary>
        public static object ChannelSend(string channelName, object message)
        {
            if (!_channels.TryGetValue(channelName, out var channel))
            {
                // Auto-create channel
                _channels[channelName] = new Channel(channelName, 100);
                channel = _channels[channelName];
            }

            var success = channel.Send(message);

            LogTrace("channel_send", new Dictionary<string, object>
            {
                ["channel"] = channelName,
                ["message_type"] = message?.GetType().Name ?? "null",
                ["success"] = success
            });

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:channel_send",
                ["channel"] = channelName,
                ["success"] = success,
                ["queue_size"] = channel.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Receive a message from a channel
        /// </summary>
        public static object ChannelReceive(string channelName)
        {
            if (!_channels.TryGetValue(channelName, out var channel))
            {
                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:channel_receive",
                    ["channel"] = channelName,
                    ["success"] = false,
                    ["error"] = "channel_not_found",
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            var (success, message) = channel.Receive();

            LogTrace("channel_receive", new Dictionary<string, object>
            {
                ["channel"] = channelName,
                ["success"] = success
            });

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:channel_receive",
                ["channel"] = channelName,
                ["success"] = success,
                ["message"] = message!,
                ["queue_size"] = channel.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Get channel status and statistics
        /// </summary>
        public static object ChannelStatus(string channelName)
        {
            if (!_channels.TryGetValue(channelName, out var channel))
            {
                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:channel_status",
                    ["channel"] = channelName,
                    ["exists"] = false,
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:channel_status",
                ["channel"] = channelName,
                ["exists"] = true,
                ["count"] = channel.Count,
                ["total_sent"] = channel.TotalSent,
                ["total_received"] = channel.TotalReceived,
                ["buffer_size"] = channel.BufferSize,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// List all channels
        /// </summary>
        public static object ListChannels()
        {
            var channelList = _channels.Select(c => new Dictionary<string, object>
            {
                ["name"] = c.Key,
                ["count"] = c.Value.Count,
                ["total_sent"] = c.Value.TotalSent,
                ["total_received"] = c.Value.TotalReceived
            }).ToList();

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:channel_list",
                ["channels"] = channelList,
                ["count"] = _channels.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        #endregion

        #region Checkpoint System (Consciousness Snapshots)

        /// <summary>
        /// Create a consciousness checkpoint
        /// </summary>
        public static object CreateCheckpoint(string name)
        {
            EnsureInitialized();

            var checkpoint = new ConsciousnessCheckpoint
            {
                Name = name,
                Timestamp = DateTime.UtcNow,
                AwarenessLevel = _awarenessLevel,
                AttentionWeights = _attentionWeights?.Clone(),
                GradientState = _gradientState?.Clone(),
                SelfEmbedding = _selfEmbedding?.Clone(),
                MemoryStore = new Dictionary<string, object>(_memoryStore),
                SelfMetrics = new Dictionary<string, double>(_selfMetrics),
                TemporalHistory = _temporalHistory.ToDictionary(
                    k => k.Key,
                    k => new List<double>(k.Value)
                )
            };

            _checkpoints[name] = checkpoint;

            LogTrace("checkpoint_create", new Dictionary<string, object>
            {
                ["name"] = name,
                ["memory_size"] = checkpoint.MemoryStore.Count,
                ["awareness"] = checkpoint.AwarenessLevel
            });

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:checkpoint",
                ["name"] = name,
                ["action"] = "created",
                ["awareness_level"] = _awarenessLevel,
                ["memory_count"] = _memoryStore.Count,
                ["checkpoints_total"] = _checkpoints.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Restore a consciousness checkpoint
        /// </summary>
        public static object RestoreCheckpoint(string name)
        {
            if (!_checkpoints.TryGetValue(name, out var checkpoint))
            {
                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:checkpoint",
                    ["name"] = name,
                    ["action"] = "restore_failed",
                    ["error"] = "checkpoint_not_found",
                    ["timestamp"] = DateTime.UtcNow.Ticks
                };
            }

            // Restore state
            _awarenessLevel = checkpoint.AwarenessLevel;

            if (checkpoint.AttentionWeights != null)
                _attentionWeights = checkpoint.AttentionWeights.Clone();

            if (checkpoint.GradientState != null)
                _gradientState = checkpoint.GradientState.Clone();

            if (checkpoint.SelfEmbedding != null)
                _selfEmbedding = checkpoint.SelfEmbedding.Clone();

            _memoryStore = new Dictionary<string, object>(checkpoint.MemoryStore);
            _selfMetrics = new Dictionary<string, double>(checkpoint.SelfMetrics);
            _temporalHistory = checkpoint.TemporalHistory.ToDictionary(
                k => k.Key,
                k => new List<double>(k.Value)
            );

            LogTrace("checkpoint_restore", new Dictionary<string, object>
            {
                ["name"] = name,
                ["restored_to"] = checkpoint.Timestamp
            });

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:checkpoint",
                ["name"] = name,
                ["action"] = "restored",
                ["restored_to"] = checkpoint.Timestamp.Ticks,
                ["awareness_level"] = _awarenessLevel,
                ["memory_count"] = _memoryStore.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// List all checkpoints
        /// </summary>
        public static object ListCheckpoints()
        {
            var checkpointList = _checkpoints.Select(c => new Dictionary<string, object>
            {
                ["name"] = c.Key,
                ["timestamp"] = c.Value.Timestamp.Ticks,
                ["awareness_level"] = c.Value.AwarenessLevel,
                ["memory_count"] = c.Value.MemoryStore.Count
            }).ToList();

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:checkpoint_list",
                ["checkpoints"] = checkpointList,
                ["count"] = _checkpoints.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Delete a checkpoint
        /// </summary>
        public static object DeleteCheckpoint(string name)
        {
            var removed = _checkpoints.Remove(name);

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:checkpoint",
                ["name"] = name,
                ["action"] = removed ? "deleted" : "not_found",
                ["checkpoints_remaining"] = _checkpoints.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        #endregion

        #region Trace System (Consciousness Debugging)

        /// <summary>
        /// Enable or disable tracing
        /// </summary>
        public static object SetTracing(bool enabled)
        {
            _traceEnabled = enabled;

            if (!enabled)
            {
                _traceLog.Clear();
            }

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:trace",
                ["action"] = "set_tracing",
                ["enabled"] = enabled,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Get trace status
        /// </summary>
        public static bool IsTracingEnabled() => _traceEnabled;

        /// <summary>
        /// Log a trace entry
        /// </summary>
        public static void LogTrace(string operation, object? data = null)
        {
            if (!_traceEnabled) return;

            var entry = new TraceEntry
            {
                Timestamp = DateTime.UtcNow,
                Operation = operation,
                Data = data,
                AwarenessLevel = _awarenessLevel,
                MemoryCount = _memoryStore.Count
            };

            _traceLog.Add(entry);

            // Limit trace log size
            if (_traceLog.Count > 1000)
            {
                _traceLog.RemoveRange(0, 100);
            }
        }

        /// <summary>
        /// Get trace log
        /// </summary>
        public static object GetTraceLog(int? limit = null)
        {
            var entries = limit.HasValue
                ? _traceLog.TakeLast(limit.Value).ToList()
                : _traceLog.ToList();

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:trace_log",
                ["enabled"] = _traceEnabled,
                ["entries"] = entries.Select(e => new Dictionary<string, object>
                {
                    ["timestamp"] = e.Timestamp.Ticks,
                    ["operation"] = e.Operation,
                    ["data"] = e.Data!,
                    ["awareness_level"] = e.AwarenessLevel,
                    ["memory_count"] = e.MemoryCount
                }).ToList(),
                ["count"] = entries.Count,
                ["total_entries"] = _traceLog.Count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        /// <summary>
        /// Clear trace log
        /// </summary>
        public static object ClearTraceLog()
        {
            var count = _traceLog.Count;
            _traceLog.Clear();

            return new Dictionary<string, object>
            {
                ["type"] = "consciousness:trace",
                ["action"] = "cleared",
                ["entries_removed"] = count,
                ["timestamp"] = DateTime.UtcNow.Ticks
            };
        }

        #endregion

        #region Helper Methods

        private static void EnsureInitialized()
        {
            if (!_initialized)
            {
                Initialize();
            }
        }

        private static NslTensor ConvertToTensor(object input)
        {
            return input switch
            {
                NslTensor t => t.Clone(),
                double d => new NslTensor(d),
                float f => new NslTensor(f),
                int i => new NslTensor(i),
                long l => new NslTensor(l),
                double[] arr => NslTensor.FromArray(arr, new long[] { arr.Length }),
                float[] arr => NslTensor.FromArray(arr.Select(x => (double)x).ToArray(), new long[] { arr.Length }),
                int[] arr => NslTensor.FromArray(arr.Select(x => (double)x).ToArray(), new long[] { arr.Length }),
                List<object> list => NslTensor.FromArray(
                    list.Select(x => Convert.ToDouble(x)).ToArray(),
                    new long[] { list.Count }
                ),
                string s => NslTensor.FromArray(
                    s.Select(c => (double)c / 256.0).ToArray(),
                    new long[] { s.Length }
                ),
                Dictionary<string, object> dict when dict.ContainsKey("value") => ConvertToTensor(dict["value"]),
                _ => new NslTensor(input?.GetHashCode() / (double)int.MaxValue ?? 0.0)
            };
        }

        private static NslTensor EmbedToSize(NslTensor input, int targetSize)
        {
            var flat = input.View(new long[] { -1 });
            var currentSize = (int)flat.NumElements;

            // Handle empty tensor - return zeros
            if (currentSize == 0)
            {
                return NslTensor.Zeros(new long[] { targetSize });
            }

            if (currentSize == targetSize)
            {
                return flat;
            }
            else if (currentSize > targetSize)
            {
                return flat.Slice(0, 0, targetSize);
            }
            else
            {
                // Repeat and truncate
                var repeats = (int)Math.Ceiling((double)targetSize / currentSize);
                var repeated = flat.Repeat(new long[] { repeats });
                return repeated.Slice(0, 0, targetSize);
            }
        }

        private static NslTensor ApplySelfAttention(NslTensor x)
        {
            // Simple self-attention: softmax(x @ W @ x^T) @ x
            var scores = NslTensorOps.MatMul(x.Unsqueeze(0), _attentionWeights!);
            var attended = NslF.Softmax(scores.Squeeze(0), 0);
            // Broadcast the attention-weighted sum to match input shape
            var result = NslTensor.Full(x.Shape, attended.Mul(x).Sum().ToScalar());
            return result;
        }

        private static NslTensor LayerNorm(NslTensor x, double eps = 1e-5)
        {
            var mean = x.Mean();
            var variance = x.Var();
            var normalized = x.Sub(mean.ToScalar()).Div(Math.Sqrt(variance.ToScalar() + eps));
            return normalized;
        }

        private static double ComputeCosineSimilarity(NslTensor a, NslTensor b)
        {
            var dotProduct = a.Mul(b).Sum().ToScalar();
            var normA = a.Pow(2).Sum().Sqrt().ToScalar();
            var normB = b.Pow(2).Sum().Sqrt().ToScalar();

            if (normA < 1e-10 || normB < 1e-10) return 0;
            return dotProduct / (normA * normB);
        }

        private static double ComputeEntropy(NslTensor probs)
        {
            // -sum(p * log(p)) for p > epsilon
            var epsilon = 1e-10;
            var clampedProbs = probs.ClampMin(epsilon);
            var logProbs = clampedProbs.Log();
            var entropy = probs.Mul(logProbs).Neg().Sum().ToScalar();
            return entropy;
        }

        private static double ComputeDivergence(NslTensor gradients)
        {
            var flat = gradients.View(new long[] { -1 });
            if (flat.NumElements <= 1) return 0;

            var divergence = 0.0;
            for (int i = 1; i < flat.NumElements; i++)
            {
                divergence += flat[i] - flat[i - 1];
            }
            return divergence;
        }

        private static int EstimateRank(NslTensor matrix)
        {
            // Simple rank estimation using singular value thresholding
            var threshold = 1e-6;
            var m = (int)matrix.Shape[0];
            var n = (int)matrix.Shape[1];
            var minDim = Math.Min(m, n);

            // Approximate by counting non-zero diagonal elements of A^T A
            var ata = NslTensorOps.MatMul(matrix.T(), matrix);
            int rank = 0;
            for (int i = 0; i < minDim; i++)
            {
                if (Math.Abs(ata[i, i]) > threshold)
                    rank++;
            }
            return Math.Max(1, rank);
        }

        private static double ComputeSchmidtCoefficients(NslTensor a, NslTensor b)
        {
            // Approximate Schmidt number (entanglement measure)
            var outer = NslTensorOps.MatMul(a.Unsqueeze(1), b.Unsqueeze(0));
            var rank = EstimateRank(outer);
            return rank;
        }

        private static List<object> TensorToList(NslTensor t)
        {
            return t.ToArray().Cast<object>().ToList();
        }

        private static Dictionary<string, object> AnalyzeTarget(object target)
        {
            var analysis = new Dictionary<string, object>
            {
                ["type"] = target?.GetType().Name ?? "null",
                ["is_null"] = target == null
            };

            switch (target)
            {
                case double d:
                    analysis["value"] = d;
                    analysis["is_finite"] = double.IsFinite(d);
                    analysis["is_positive"] = d > 0;
                    break;
                case int i:
                    analysis["value"] = i;
                    analysis["is_positive"] = i > 0;
                    break;
                case string s:
                    analysis["length"] = s.Length;
                    analysis["is_empty"] = string.IsNullOrEmpty(s);
                    break;
                case List<object> list:
                    analysis["count"] = list.Count;
                    analysis["is_empty"] = list.Count == 0;
                    if (list.Count > 0)
                    {
                        analysis["element_type"] = list[0]?.GetType().Name ?? "null";
                    }
                    break;
                case Dictionary<string, object> dict:
                    analysis["count"] = dict.Count;
                    analysis["keys"] = dict.Keys.Take(10).ToList();
                    if (dict.TryGetValue("type", out var consciousnessType))
                    {
                        analysis["consciousness_type"] = consciousnessType;
                    }
                    break;
                case NslTensor tensor:
                    analysis["shape"] = tensor.Shape.ToArray();
                    analysis["num_elements"] = tensor.NumElements;
                    analysis["requires_grad"] = tensor.RequiresGrad;
                    analysis["mean"] = tensor.Mean().ToScalar();
                    analysis["std"] = tensor.Std().ToScalar();
                    break;
            }

            return analysis;
        }

        private static object PerturbState(object state, double amount)
        {
            if (state is double d) return d + amount * (_random.NextDouble() * 2 - 1);
            if (state is float f) return f + amount * (_random.NextDouble() * 2 - 1);
            if (state is int i) return i + (int)(amount * (_random.NextDouble() * 2 - 1));
            if (state is NslTensor t)
            {
                var noise = NslTensor.Randn(t.Shape).Mul(amount);
                return t.Add(noise);
            }
            if (state is List<object> list)
            {
                return list.Select(item =>
                {
                    if (item is double itemD) return (object)(itemD + amount * (_random.NextDouble() * 2 - 1));
                    if (item is float itemF) return (object)(itemF + amount * (_random.NextDouble() * 2 - 1));
                    return item;
                }).ToList();
            }
            return state;
        }

        /// <summary>
        /// Get current device being used
        /// </summary>
        public static string GetDevice() => "cpu";

        /// <summary>
        /// Check if GPU is available (future CUDA support)
        /// </summary>
        public static bool IsGPUAvailable() => false;

        /// <summary>
        /// Get current awareness level
        /// </summary>
        public static double GetAwarenessLevel() => _awarenessLevel;

        /// <summary>
        /// Reset all state
        /// </summary>
        public static void Reset()
        {
            _initialized = false;
            _awarenessLevel = 0.5;
            _attentionWeights = null;
            _gradientState = null;
            _queryProjection = null;
            _keyProjection = null;
            _valueProjection = null;

            // Reset extended consciousness state
            _memoryStore.Clear();
            _memoryEmbeddings.Clear();
            _memoryHistory.Clear();
            _temporalAccumulators.Clear();
            _temporalHistory.Clear();
            _selfMetrics.Clear();
            _selfEmbedding = null;

            // Reset channel, checkpoint, and trace state
            _channels.Clear();
            _checkpoints.Clear();
            _traceEnabled = false;
            _traceLog.Clear();
        }

        /// <summary>
        /// Get memory store contents
        /// </summary>
        public static Dictionary<string, object> GetMemoryStore() => new(_memoryStore);

        /// <summary>
        /// Get temporal history for a channel
        /// </summary>
        public static List<double>? GetTemporalHistory(string channel = "default") =>
            _temporalHistory.TryGetValue(channel, out var history) ? new List<double>(history) : null;

        /// <summary>
        /// Get self metrics
        /// </summary>
        public static Dictionary<string, double> GetSelfMetrics() => new(_selfMetrics);

        /// <summary>
        /// Get the attention weights tensor for inspection
        /// </summary>
        public static NslTensor? GetAttentionWeights() => _attentionWeights?.Clone();

        /// <summary>
        /// Get the gradient state tensor for inspection
        /// </summary>
        public static NslTensor? GetGradientState() => _gradientState?.Clone();

        #endregion
    }

    /// <summary>
    /// Communication channel for multi-agent communication
    /// </summary>
    public class Channel
    {
        private readonly Queue<object> _buffer;
        private readonly int _bufferSize;

        /// <summary>Gets the name of the channel.</summary>
        public string Name { get; }
        /// <summary>Gets the current number of messages in the buffer.</summary>
        public int Count => _buffer.Count;
        /// <summary>Gets the maximum buffer size for the channel.</summary>
        public int BufferSize => _bufferSize;
        /// <summary>Gets the total number of messages sent through this channel.</summary>
        public long TotalSent { get; private set; }
        /// <summary>Gets the total number of messages received from this channel.</summary>
        public long TotalReceived { get; private set; }

        /// <summary>Initializes a new channel with the specified name and buffer size.</summary>
        /// <param name="name">The name of the channel.</param>
        /// <param name="bufferSize">Maximum number of messages to buffer (default 100).</param>
        public Channel(string name, int bufferSize = 100)
        {
            Name = name;
            _bufferSize = bufferSize;
            _buffer = new Queue<object>();
            TotalSent = 0;
            TotalReceived = 0;
        }

        /// <summary>Sends a message to the channel. Drops oldest message if buffer is full.</summary>
        /// <param name="message">The message to send.</param>
        /// <returns>True if the message was sent successfully.</returns>
        public bool Send(object message)
        {
            if (_buffer.Count >= _bufferSize)
            {
                // Buffer full, drop oldest message
                _buffer.Dequeue();
            }

            _buffer.Enqueue(message);
            TotalSent++;
            return true;
        }

        /// <summary>Receives and removes the next message from the channel.</summary>
        /// <returns>A tuple containing success status and the message if available.</returns>
        public (bool Success, object? Message) Receive()
        {
            if (_buffer.Count == 0)
            {
                return (false, null);
            }

            var message = _buffer.Dequeue();
            TotalReceived++;
            return (true, message);
        }

        /// <summary>Attempts to peek at the next message without removing it.</summary>
        /// <param name="message">The message if available, null otherwise.</param>
        /// <returns>True if a message is available.</returns>
        public bool TryPeek(out object? message)
        {
            if (_buffer.Count == 0)
            {
                message = null;
                return false;
            }

            message = _buffer.Peek();
            return true;
        }

        /// <summary>Clears all messages from the channel buffer.</summary>
        public void Clear()
        {
            _buffer.Clear();
        }
    }

    /// <summary>
    /// Consciousness checkpoint for state snapshots
    /// </summary>
    public class ConsciousnessCheckpoint
    {
        /// <summary>Gets or sets the name identifier for this checkpoint.</summary>
        public string Name { get; set; } = "";
        /// <summary>Gets or sets the timestamp when this checkpoint was created.</summary>
        public DateTime Timestamp { get; set; }
        /// <summary>Gets or sets the awareness level at checkpoint time (0.0 to 1.0).</summary>
        public double AwarenessLevel { get; set; }
        /// <summary>Gets or sets the attention weights tensor at checkpoint time.</summary>
        public NslTensor? AttentionWeights { get; set; }
        /// <summary>Gets or sets the gradient state tensor for backpropagation.</summary>
        public NslTensor? GradientState { get; set; }
        /// <summary>Gets or sets the self-embedding tensor representing agent state.</summary>
        public NslTensor? SelfEmbedding { get; set; }
        /// <summary>Gets or sets the memory store snapshot at checkpoint time.</summary>
        public Dictionary<string, object> MemoryStore { get; set; } = new();
        /// <summary>Gets or sets the self-evaluation metrics at checkpoint time.</summary>
        public Dictionary<string, double> SelfMetrics { get; set; } = new();
        /// <summary>Gets or sets the temporal history of metric values.</summary>
        public Dictionary<string, List<double>> TemporalHistory { get; set; } = new();
    }

    /// <summary>
    /// Trace entry for consciousness debugging
    /// </summary>
    public class TraceEntry
    {
        /// <summary>Gets or sets the timestamp of the trace entry.</summary>
        public DateTime Timestamp { get; set; }
        /// <summary>Gets or sets the operation name being traced.</summary>
        public string Operation { get; set; } = "";
        /// <summary>Gets or sets optional data associated with the trace entry.</summary>
        public object? Data { get; set; }
        /// <summary>Gets or sets the awareness level at trace time.</summary>
        public double AwarenessLevel { get; set; }
        /// <summary>Gets or sets the memory count at trace time.</summary>
        public int MemoryCount { get; set; }

        /// <summary>Returns a formatted string representation of the trace entry.</summary>
        public override string ToString()
        {
            return $"[{Timestamp:HH:mm:ss.fff}] {Operation} (awareness={AwarenessLevel:F3}, memory={MemoryCount})";
        }
    }
}
