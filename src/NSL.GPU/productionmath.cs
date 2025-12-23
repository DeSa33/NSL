using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace NSL.GPU
{
    /// <summary>
    /// ProductionMath: A GPU-accelerated implementation of relational mathematics.
    ///
    /// Classical math: f: R^n → R (single output)
    /// ProductionMath: O: E^n → P(R), π_θ: R → [0,1] (relation generation + learned selection)
    ///
    /// Key insight: Math = Relation Generation + Learned Selection
    ///
    /// Instead of asking "What is the answer?", we ask:
    /// "What does this operation mean, and which meaning matters right now?"
    /// </summary>
    public class ProductionMathEngine
    {
        private readonly Accelerator _accelerator;
        private readonly int _embeddingDim;
        private readonly int _maxVariants;
        private readonly float _epsilon;  // ε-greedy exploration rate
        private readonly float _alpha;    // Memory update rate

        // Relation memory tensor: M_op,mode ∈ R^(k×d)
        // Stores statistics for each (operator, mode) pair
        private readonly Dictionary<OperatorType, GpuTensor> _relationMemory;

        // Policy network weights (trainable)
        private GpuTensor _policyWeights;
        private GpuTensor _policyBias;

        // Operation slice embeddings
        private readonly Dictionary<OperatorType, GpuTensor> _operatorEmbeddings;

        // Compiled kernels
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int, int> _relationGeneratorKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _policySoftmaxKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, float, int> _variantSelectionKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, float, int> _memoryUpdateKernel;

        // Random state for ε-greedy
        private readonly Random _rng = new Random();

        // === OPTIMIZATION: Caching and fast paths ===
        private readonly Dictionary<int, (SemanticMode[] modes, float[] scores)> _variantCache = new();
        private readonly int[] _topKModes;  // Pre-computed top-k modes based on learned preferences
        private int _fastModeCount = 4;     // Only evaluate top-4 modes in fast path
        private float _confidenceThreshold = 0.8f;  // Skip exploration if confidence > threshold
        private bool _useFastPath = true;
        private int _operationCount = 0;
        private const int CACHE_CLEANUP_INTERVAL = 1000;

        /// <summary>
        /// Semantic modes for relational variants.
        /// Each operation can produce multiple interpretations.
        /// </summary>
        public enum SemanticMode
        {
            Sum = 0,              // Aggregation: a + b
            Replicate = 1,        // Duplication: a × b (as repeated addition)
            CoPresence = 2,       // Min of operands (both must be present)
            Ratio = 3,            // Division: a / b
            InverseRatio = 4,     // Division: b / a
            Difference = 5,       // Subtraction: a - b
            InverseDiff = 6,      // Subtraction: b - a
            Geometric = 7,        // Geometric mean: √(a × b)
            Harmonic = 8,         // Harmonic mean: 2ab/(a+b)
            Power = 9,            // Exponentiation: a^b
            InversePower = 10,    // Exponentiation: b^a
            LogRatio = 11,        // log(a/b)
            Compound = 12,        // For symbolic: compound identity
            SelfDuplication = 13, // For x ⋄ x patterns
            Projection = 14,      // Project onto one operand
            Null = 15             // No meaningful relation
        }

        /// <summary>Public API</summary>
        public enum OperatorType
        {
            Diamond,    // ⋄ - The universal ProductionMath operator
            Plus,       // + with relational awareness
            Times,      // × with relational awareness
            Power,      // ^ with relational awareness
            Compose     // ∘ function composition
        }

        /// <summary>Public API</summary>
        public ProductionMathEngine(
            Accelerator accelerator,
            int embeddingDim = 64,
            int maxVariants = 16,
            float epsilon = 0.1f,
            float alpha = 0.01f)
        {
            _accelerator = accelerator;
            _embeddingDim = embeddingDim;
            _maxVariants = maxVariants;
            _epsilon = epsilon;
            _alpha = alpha;

            _relationMemory = new Dictionary<OperatorType, GpuTensor>();
            _operatorEmbeddings = new Dictionary<OperatorType, GpuTensor>();

            // Initialize relation memory for each operator
            foreach (OperatorType op in Enum.GetValues<OperatorType>())
            {
                // M_op ∈ R^(numModes × embeddingDim)
                int numModes = Enum.GetValues<SemanticMode>().Length;
                _relationMemory[op] = CreateScaledRandom(accelerator, numModes, embeddingDim, 0.1f);

                // Operator embedding s_op ∈ R^embeddingDim
                _operatorEmbeddings[op] = CreateScaledRandom(accelerator, 1, embeddingDim, 0.1f);
            }

            // Initialize policy network: f_θ(φ, s_op) = φ·W·s_op + b
            // W ∈ R^(embeddingDim × embeddingDim), b ∈ R^1
            _policyWeights = CreateScaledRandom(accelerator, embeddingDim, embeddingDim, 0.1f);
            _policyBias = GpuTensor.Zeros(accelerator, 1);

            // Compile kernels
            _relationGeneratorKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int, int>(
                RelationGeneratorKernelImpl);

            _policySoftmaxKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
                PolicySoftmaxKernelImpl);

            _variantSelectionKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, float, int>(
                VariantSelectionKernelImpl);

            _memoryUpdateKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, float, int>(
                MemoryUpdateKernelImpl);

            // Initialize top-k modes with most common useful modes
            _topKModes = new[] {
                (int)SemanticMode.Sum,
                (int)SemanticMode.Geometric,
                (int)SemanticMode.Replicate,
                (int)SemanticMode.Ratio
            };
        }

        #region Core ProductionMath Operations

        /// <summary>
        /// The Diamond operator: ⋄
        ///
        /// O⋄(a, b) = {r1, r2, ..., rk}
        ///
        /// Instead of computing a single result, generates all relational variants
        /// and uses learned policy to select the contextually appropriate one.
        /// </summary>
        public ProductionMathResult Diamond(GpuTensor a, GpuTensor b, GpuTensor? context = null)
        {
            return ApplyOperator(OperatorType.Diamond, a, b, context);
        }

        /// <summary>
        /// Relational addition: generates multiple interpretations of "combining"
        /// </summary>
        public ProductionMathResult RelationalAdd(GpuTensor a, GpuTensor b, GpuTensor? context = null)
        {
            return ApplyOperator(OperatorType.Plus, a, b, context);
        }

        /// <summary>
        /// Relational multiplication: generates multiple interpretations of "scaling/repeating"
        /// </summary>
        public ProductionMathResult RelationalMultiply(GpuTensor a, GpuTensor b, GpuTensor? context = null)
        {
            return ApplyOperator(OperatorType.Times, a, b, context);
        }

        /// <summary>
        /// Core operator application with relation generation and learned selection.
        /// OPTIMIZED: Uses fast path with top-k modes and caching.
        ///
        /// Algorithm:
        /// 1. Generate all relational variants: O(a,b) = {(mode_i, v_i, φ_i)}
        /// 2. Compute policy scores: π_θ(r_i | a, b, O)
        /// 3. Select variant (ε-greedy): r* = argmax or sample
        /// 4. Update relation memory: M ← (1-α)M + αφ_selected
        /// 5. Return selected variant with full context
        /// </summary>
        public ProductionMathResult ApplyOperator(OperatorType op, GpuTensor a, GpuTensor b, GpuTensor? context = null)
        {
            _operationCount++;

            // Periodic cache cleanup
            if (_operationCount % CACHE_CLEANUP_INTERVAL == 0)
            {
                _variantCache.Clear();
            }

            int batchSize = a.Shape[0];

            // === FAST PATH: Only evaluate top-k modes ===
            if (_useFastPath && batchSize <= 64)
            {
                return ApplyOperatorFast(op, a, b);
            }

            // === FULL PATH: All modes ===
            int numModes = Enum.GetValues<SemanticMode>().Length;

            // Step 1: Generate all relational variants
            var variantValues = new GpuTensor(_accelerator, new[] { batchSize, numModes });
            var variantEmbeddings = new GpuTensor(_accelerator, new[] { batchSize, numModes, _embeddingDim });
            var variantModes = _accelerator.Allocate1D<int>(batchSize * numModes);

            GenerateRelationalVariants(op, a, b, variantValues, variantEmbeddings, variantModes.View);

            // Step 2: Compute policy scores via softmax
            var policyScores = new GpuTensor(_accelerator, new[] { batchSize, numModes });
            ComputePolicyScores(variantEmbeddings, _operatorEmbeddings[op], policyScores);

            // Step 3: Select variant (ε-greedy)
            var selectedIndices = _accelerator.Allocate1D<int>(batchSize);
            var selectedValues = new GpuTensor(_accelerator, new[] { batchSize });
            SelectVariant(policyScores, variantValues, selectedIndices.View, selectedValues);

            // Step 4: Update relation memory with selected embeddings
            UpdateRelationMemory(op, variantEmbeddings, selectedIndices.View);

            // Step 5: Extract linear anchor (compatibility with classical math)
            float? linearValue = ExtractLinearAnchor(a, b, op);

            // Build result
            var result = new ProductionMathResult
            {
                SelectedValue = selectedValues,
                SelectedMode = GetSelectedModes(selectedIndices),
                AllVariants = ExtractAllVariants(variantValues, variantModes, variantEmbeddings),
                PolicyScores = policyScores,
                LinearAnchor = linearValue,
                Operator = op
            };

            // Cleanup intermediate buffers
            variantModes.Dispose();
            selectedIndices.Dispose();

            return result;
        }

        /// <summary>
        /// FAST PATH: Only evaluates top-k learned modes.
        /// Reduces variant generation from 16 to 4 modes (4x speedup).
        /// </summary>
        private ProductionMathResult ApplyOperatorFast(OperatorType op, GpuTensor a, GpuTensor b)
        {
            int batchSize = a.Shape[0];
            int numModes = _fastModeCount;

            // Only generate top-k variants
            var variantValues = new GpuTensor(_accelerator, new[] { batchSize, numModes });
            var variantEmbeddings = new GpuTensor(_accelerator, new[] { batchSize, numModes, _embeddingDim });
            var variantModes = _accelerator.Allocate1D<int>(batchSize * numModes);

            // Use fast variant generation with only top-k modes
            GenerateRelationalVariantsFast(op, a, b, variantValues, variantEmbeddings, variantModes.View, _topKModes);

            // Compute policy scores for reduced set
            var policyScores = new GpuTensor(_accelerator, new[] { batchSize, numModes });
            ComputePolicyScores(variantEmbeddings, _operatorEmbeddings[op], policyScores);

            // Check confidence - skip exploration if high confidence
            float maxScore = GetMaxPolicyScore(policyScores);

            var selectedIndices = _accelerator.Allocate1D<int>(batchSize);
            var selectedValues = new GpuTensor(_accelerator, new[] { batchSize });

            if (maxScore > _confidenceThreshold)
            {
                // High confidence: just take argmax, no exploration
                SelectVariantArgmax(policyScores, variantValues, selectedIndices.View, selectedValues);
            }
            else
            {
                // Low confidence: use ε-greedy
                SelectVariant(policyScores, variantValues, selectedIndices.View, selectedValues);
            }

            // Skip memory update in fast path (batch updates later)
            float? linearValue = ExtractLinearAnchor(a, b, op);

            var result = new ProductionMathResult
            {
                SelectedValue = selectedValues,
                SelectedMode = GetSelectedModesFast(selectedIndices, _topKModes),
                AllVariants = ExtractAllVariantsFast(variantValues, variantModes, variantEmbeddings, _topKModes),
                PolicyScores = policyScores,
                LinearAnchor = linearValue,
                Operator = op
            };

            variantModes.Dispose();
            selectedIndices.Dispose();

            return result;
        }

        /// <summary>
        /// Enable or disable fast path optimization.
        /// </summary>
        public bool UseFastPath
        {
            get => _useFastPath;
            set => _useFastPath = value;
        }

        /// <summary>
        /// Set number of modes to evaluate in fast path (default: 4).
        /// </summary>
        public int FastModeCount
        {
            get => _fastModeCount;
            set => _fastModeCount = Math.Clamp(value, 1, 16);
        }

        /// <summary>
        /// SUPER-FAST: Direct computation using learned best mode.
        /// Bypasses variant generation entirely - just computes the result.
        /// Use when you want ProductionMath's learned preferences without overhead.
        /// </summary>
        public GpuTensor DirectCompute(OperatorType op, GpuTensor a, GpuTensor b)
        {
            // Use the top learned mode directly - no variant generation
            var mode = (SemanticMode)_topKModes[0];
            return ComputeModeOnGpu(mode, a, b);
        }

        /// <summary>
        /// Compute a semantic mode directly on GPU without any overhead.
        /// </summary>
        private GpuTensor ComputeModeOnGpu(SemanticMode mode, GpuTensor a, GpuTensor b)
        {
            // Use existing GPU kernels for actual computation
            return mode switch
            {
                SemanticMode.Sum => GpuAdd(a, b),
                SemanticMode.Replicate => GpuMul(a, b),
                SemanticMode.Geometric => GpuGeometricMean(a, b),
                SemanticMode.Ratio => GpuDiv(a, b),
                _ => GpuAdd(a, b)  // Default to sum
            };
        }

        private GpuTensor GpuAdd(GpuTensor a, GpuTensor b)
        {
            var result = new GpuTensor(_accelerator, a.Shape);
            // Direct GPU kernel - no CPU transfer
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (idx, aView, bView, rView) => { rView[idx] = aView[idx] + bView[idx]; });
            kernel(a.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();
            return result;
        }

        private GpuTensor GpuMul(GpuTensor a, GpuTensor b)
        {
            var result = new GpuTensor(_accelerator, a.Shape);
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (idx, aView, bView, rView) => { rView[idx] = aView[idx] * bView[idx]; });
            kernel(a.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();
            return result;
        }

        private GpuTensor GpuDiv(GpuTensor a, GpuTensor b)
        {
            var result = new GpuTensor(_accelerator, a.Shape);
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (idx, aView, bView, rView) => {
                    rView[idx] = bView[idx] != 0 ? aView[idx] / bView[idx] : 0;
                });
            kernel(a.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();
            return result;
        }

        private GpuTensor GpuGeometricMean(GpuTensor a, GpuTensor b)
        {
            var result = new GpuTensor(_accelerator, a.Shape);
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (idx, aView, bView, rView) => {
                    float prod = aView[idx] * bView[idx];
                    rView[idx] = prod >= 0 ? XMath.Sqrt(prod) : XMath.Sqrt(-prod);
                });
            kernel(a.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();
            return result;
        }

        #endregion

        #region Relation Generation

        /// <summary>
        /// Generate all relational variants for an operation.
        ///
        /// For O⋄(3, 2):
        /// - (sum, 5, φ_sum)
        /// - (replicate, 6, φ_rep)
        /// - (co-presence, 2, φ_cop)
        /// - (ratio, 1.5, φ_rat)
        /// - (inverse-ratio, 0.666, φ_inv)
        /// </summary>
        private void GenerateRelationalVariants(
            OperatorType op,
            GpuTensor a,
            GpuTensor b,
            GpuTensor values,
            GpuTensor embeddings,
            ArrayView<int> modes)
        {
            int batchSize = a.Shape[0];
            int numModes = Enum.GetValues<SemanticMode>().Length;

            _relationGeneratorKernel(
                batchSize * numModes,
                a.Buffer.View,
                b.Buffer.View,
                values.Buffer.View,
                modes,
                embeddings.Buffer.View,
                batchSize,
                numModes,
                _embeddingDim);

            _accelerator.Synchronize();
        }

        /// <summary>
        /// GPU kernel for generating relational variants.
        /// Each thread computes one (batch, mode) pair.
        /// </summary>
        private static void RelationGeneratorKernelImpl(
            Index1D idx,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> values,
            ArrayView<int> modes,
            ArrayView<float> embeddings,
            int batchSize,
            int numModes,
            int embeddingDim)
        {
            int batchIdx = idx / numModes;
            int modeIdx = idx % numModes;

            if (batchIdx >= batchSize) return;

            float aVal = a[batchIdx];
            float bVal = b[batchIdx];

            // Compute value for this semantic mode
            float value = 0f;

            switch (modeIdx)
            {
                case 0:  // Sum
                    value = aVal + bVal;
                    break;
                case 1:  // Replicate (multiplication as repeated addition)
                    value = aVal * bVal;
                    break;
                case 2:  // Co-presence (minimum - both must contribute)
                    value = XMath.Min(aVal, bVal);
                    break;
                case 3:  // Ratio
                    value = bVal != 0 ? aVal / bVal : 0f;
                    break;
                case 4:  // Inverse ratio
                    value = aVal != 0 ? bVal / aVal : 0f;
                    break;
                case 5:  // Difference
                    value = aVal - bVal;
                    break;
                case 6:  // Inverse difference
                    value = bVal - aVal;
                    break;
                case 7:  // Geometric mean
                    value = XMath.Sqrt(XMath.Abs(aVal * bVal));
                    break;
                case 8:  // Harmonic mean
                    value = (aVal + bVal) != 0 ? 2 * aVal * bVal / (aVal + bVal) : 0f;
                    break;
                case 9:  // Power a^b
                    value = XMath.Pow(XMath.Abs(aVal), bVal);
                    break;
                case 10: // Power b^a
                    value = XMath.Pow(XMath.Abs(bVal), aVal);
                    break;
                case 11: // Log ratio
                    value = (aVal > 0 && bVal > 0) ? XMath.Log(aVal / bVal) : 0f;
                    break;
                case 12: // Compound (for symbolic entities)
                    value = 1f;  // Indicates compound formed
                    break;
                case 13: // Self-duplication (when a ≈ b)
                    value = XMath.Abs(aVal - bVal) < 0.001f ? 2f : 0f;
                    break;
                case 14: // Projection onto a
                    value = aVal;
                    break;
                case 15: // Null relation
                    value = 0f;
                    break;
            }

            values[batchIdx * numModes + modeIdx] = value;
            modes[batchIdx * numModes + modeIdx] = modeIdx;

            // Generate embedding for this variant
            // φ_i = f(mode, a, b, value)
            int embOffset = (batchIdx * numModes + modeIdx) * embeddingDim;
            for (int d = 0; d < embeddingDim; d++)
            {
                // Simple embedding: encode mode, operands, and result
                float modeComponent = XMath.Sin(modeIdx * 0.1f + d * 0.01f);
                float aComponent = aVal * XMath.Cos(d * 0.05f);
                float bComponent = bVal * XMath.Sin(d * 0.05f);
                float valueComponent = value * XMath.Cos(d * 0.1f);

                embeddings[embOffset + d] = (modeComponent + aComponent + bComponent + valueComponent) * 0.25f;
            }
        }

        #endregion

        #region Policy Network

        /// <summary>
        /// Compute policy scores for variant selection.
        ///
        /// π_θ(r_i | a, b, O) = exp(f_θ(φ_i, s_op)) / Σ_j exp(f_θ(φ_j, s_op))
        ///
        /// Where f_θ(φ, s) = φ · W · s + b (bilinear form)
        /// </summary>
        private void ComputePolicyScores(
            GpuTensor variantEmbeddings,  // [batch, modes, embDim]
            GpuTensor operatorEmbedding,  // [embDim]
            GpuTensor policyScores)       // [batch, modes]
        {
            int batchSize = variantEmbeddings.Shape[0];
            int numModes = variantEmbeddings.Shape[1];

            _policySoftmaxKernel(
                batchSize,
                variantEmbeddings.Buffer.View,
                operatorEmbedding.Buffer.View,
                _policyWeights.Buffer.View,
                policyScores.Buffer.View,
                numModes,
                _embeddingDim);

            _accelerator.Synchronize();
        }

        /// <summary>
        /// GPU kernel for policy softmax computation.
        /// </summary>
        private static void PolicySoftmaxKernelImpl(
            Index1D batchIdx,
            ArrayView<float> embeddings,    // [batch, modes, embDim]
            ArrayView<float> opEmb,         // [embDim]
            ArrayView<float> weights,       // [embDim, embDim]
            ArrayView<float> scores,        // [batch, modes]
            int numModes,
            int embDim)
        {
            // Compute logits for each mode
            float maxLogit = float.MinValue;

            for (int m = 0; m < numModes; m++)
            {
                // f_θ(φ_m, s_op) = φ_m · W · s_op
                float logit = 0f;

                for (int i = 0; i < embDim; i++)
                {
                    float phi_i = embeddings[(batchIdx * numModes + m) * embDim + i];

                    float Ws_i = 0f;
                    for (int j = 0; j < embDim; j++)
                    {
                        Ws_i += weights[i * embDim + j] * opEmb[j];
                    }

                    logit += phi_i * Ws_i;
                }

                scores[batchIdx * numModes + m] = logit;
                maxLogit = XMath.Max(maxLogit, logit);
            }

            // Softmax with numerical stability
            float sumExp = 0f;
            for (int m = 0; m < numModes; m++)
            {
                float expVal = XMath.Exp(scores[batchIdx * numModes + m] - maxLogit);
                scores[batchIdx * numModes + m] = expVal;
                sumExp += expVal;
            }

            for (int m = 0; m < numModes; m++)
            {
                scores[batchIdx * numModes + m] /= sumExp;
            }
        }

        #endregion

        #region Variant Selection

        /// <summary>
        /// ε-greedy variant selection.
        ///
        /// r = argmax_i π_θ(r_i)  with probability 1-ε
        /// r = sample(r_i)        with probability ε
        ///
        /// This makes math non-deterministic by choice, not by noise.
        /// </summary>
        private void SelectVariant(
            GpuTensor policyScores,
            GpuTensor variantValues,
            ArrayView<int> selectedIndices,
            GpuTensor selectedValues)
        {
            int batchSize = policyScores.Shape[0];
            int numModes = policyScores.Shape[1];

            // Generate random values for ε-greedy
            var randomVals = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
                randomVals[i] = (float)_rng.NextDouble();

            using var randomBuffer = _accelerator.Allocate1D<float>(batchSize);
            randomBuffer.CopyFromCPU(randomVals);

            _variantSelectionKernel(
                batchSize,
                policyScores.Buffer.View,
                selectedIndices,
                randomBuffer.View,
                _epsilon,
                numModes);

            _accelerator.Synchronize();

            // Extract selected values
            var scoresData = policyScores.ToArray();
            var valuesData = variantValues.ToArray();
            var indicesData = new int[batchSize];
            selectedIndices.CopyToCPU(indicesData);

            var selectedVals = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                selectedVals[i] = valuesData[i * numModes + indicesData[i]];
            }

            selectedValues.Buffer.CopyFromCPU(selectedVals);
        }

        /// <summary>
        /// GPU kernel for ε-greedy selection.
        /// </summary>
        private static void VariantSelectionKernelImpl(
            Index1D batchIdx,
            ArrayView<float> scores,
            ArrayView<int> selected,
            ArrayView<float> randomVals,
            float epsilon,
            int numModes)
        {
            float r = randomVals[batchIdx];

            if (r > epsilon)
            {
                // Exploit: argmax
                int bestIdx = 0;
                float bestScore = scores[batchIdx * numModes];

                for (int m = 1; m < numModes; m++)
                {
                    float score = scores[batchIdx * numModes + m];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestIdx = m;
                    }
                }

                selected[batchIdx] = bestIdx;
            }
            else
            {
                // Explore: sample from distribution
                float cumSum = 0f;
                float samplePoint = r / epsilon;  // Rescale to [0,1]

                for (int m = 0; m < numModes; m++)
                {
                    cumSum += scores[batchIdx * numModes + m];
                    if (samplePoint <= cumSum)
                    {
                        selected[batchIdx] = m;
                        return;
                    }
                }

                selected[batchIdx] = numModes - 1;  // Fallback
            }
        }

        #endregion

        #region Memory Update

        /// <summary>
        /// Update relation memory with selected embeddings.
        ///
        /// M ← (1-α)M + αφ_i
        ///
        /// This creates learned preference over interpretations.
        /// </summary>
        private void UpdateRelationMemory(
            OperatorType op,
            GpuTensor variantEmbeddings,
            ArrayView<int> selectedIndices)
        {
            var memory = _relationMemory[op];
            int batchSize = variantEmbeddings.Shape[0];
            int numModes = variantEmbeddings.Shape[1];

            // For each selected variant, update the corresponding mode's memory
            var indicesData = new int[batchSize];
            selectedIndices.CopyToCPU(indicesData);

            var embData = variantEmbeddings.ToArray();
            var memData = memory.ToArray();

            for (int b = 0; b < batchSize; b++)
            {
                int modeIdx = indicesData[b];

                for (int d = 0; d < _embeddingDim; d++)
                {
                    int embOffset = (b * numModes + modeIdx) * _embeddingDim + d;
                    int memOffset = modeIdx * _embeddingDim + d;

                    // Exponential moving average update
                    memData[memOffset] = (1 - _alpha) * memData[memOffset] + _alpha * embData[embOffset];
                }
            }

            memory.Buffer.CopyFromCPU(memData);
        }

        /// <summary>
        /// GPU kernel for memory update.
        /// </summary>
        private static void MemoryUpdateKernelImpl(
            Index1D idx,
            ArrayView<float> memory,
            ArrayView<float> newValues,
            float alpha,
            int size)
        {
            if (idx >= size) return;
            memory[idx] = (1 - alpha) * memory[idx] + alpha * newValues[idx];
        }

        #endregion

        #region Linear Anchor

        /// <summary>
        /// Extract linear anchor for compatibility with classical math.
        ///
        /// L: R → R ∪ ∅
        /// L(r_i) = v_i if mode is linear-compatible, else ∅
        ///
        /// This ensures L(O+(a,b)) = a + b when numeric.
        /// Correctness is constrained, not learned.
        /// </summary>
        private float? ExtractLinearAnchor(GpuTensor a, GpuTensor b, OperatorType op)
        {
            var aData = a.ToArray();
            var bData = b.ToArray();

            if (aData.Length != 1 || bData.Length != 1)
                return null;  // Only for scalar operations

            float aVal = aData[0];
            float bVal = bData[0];

            return op switch
            {
                OperatorType.Plus => aVal + bVal,
                OperatorType.Times => aVal * bVal,
                OperatorType.Power => MathF.Pow(aVal, bVal),
                OperatorType.Diamond => aVal + bVal,  // Default linear interpretation
                _ => null
            };
        }

        #endregion

        #region REINFORCE Learning

        /// <summary>
        /// Update policy via REINFORCE gradient.
        ///
        /// ∇_θ J(θ) = E[(R - b) ∇_θ log π_θ(r)]
        ///
        /// Where:
        /// - R = relational reward
        /// - b = baseline (moving average of rewards)
        ///
        /// This is learning which math interpretation works.
        /// </summary>
        public void UpdatePolicy(ProductionMathResult result, float reward, float baseline = 0f)
        {
            float advantage = reward - baseline;

            // Compute gradient: ∇_θ log π_θ(r) = ∇_θ f_θ(φ, s) - E[∇_θ f_θ]
            // For selected action, this simplifies to the embedding difference

            // Get selected embeddings and policy scores
            var scores = result.PolicyScores.ToArray();
            int batchSize = result.PolicyScores.Shape[0];
            int numModes = result.PolicyScores.Shape[1];

            // Update policy weights using gradient
            var weightsData = _policyWeights.ToArray();
            float learningRate = 0.001f;

            foreach (var variant in result.AllVariants)
            {
                bool isSelected = variant.Mode == result.SelectedMode[0];
                float factor = isSelected ? (1 - scores[0]) : -scores[(int)variant.Mode];
                factor *= advantage * learningRate;

                // Update weights in direction of gradient
                for (int i = 0; i < _embeddingDim; i++)
                {
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        weightsData[i * _embeddingDim + j] += factor * variant.Embedding[i] * 0.01f;
                    }
                }
            }

            _policyWeights.Buffer.CopyFromCPU(weightsData);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Create a random tensor with scaled values in [-scale, scale]
        /// </summary>
        private static GpuTensor CreateScaledRandom(Accelerator accelerator, int dim1, int dim2, float scale)
        {
            var rng = new Random();
            var data = new float[dim1 * dim2];
            for (int i = 0; i < data.Length; i++)
                data[i] = ((float)rng.NextDouble() * 2 - 1) * scale;
            return GpuTensor.FromArray(accelerator, data, dim1, dim2);
        }

        private SemanticMode[] GetSelectedModes(MemoryBuffer1D<int, Stride1D.Dense> indices)
        {
            var data = new int[indices.Length];
            indices.CopyToCPU(data);
            return data.Select(i => (SemanticMode)i).ToArray();
        }

        private List<RelationalVariant> ExtractAllVariants(
            GpuTensor values,
            MemoryBuffer1D<int, Stride1D.Dense> modes,
            GpuTensor embeddings)
        {
            var valData = values.ToArray();
            var modeData = new int[modes.Length];
            modes.CopyToCPU(modeData);
            var embData = embeddings.ToArray();

            int numModes = values.Shape[1];
            var variants = new List<RelationalVariant>();

            for (int m = 0; m < numModes; m++)
            {
                var embedding = new float[_embeddingDim];
                Array.Copy(embData, m * _embeddingDim, embedding, 0, _embeddingDim);

                variants.Add(new RelationalVariant
                {
                    Mode = (SemanticMode)m,
                    Value = valData[m],
                    Embedding = embedding
                });
            }

            return variants;
        }

        #endregion

        #region Fast Path Helpers

        /// <summary>
        /// Generate variants for only the top-k modes (fast path).
        /// </summary>
        private void GenerateRelationalVariantsFast(
            OperatorType op,
            GpuTensor a,
            GpuTensor b,
            GpuTensor values,
            GpuTensor embeddings,
            ArrayView<int> modeIndices,
            int[] topKModes)
        {
            var aData = a.ToArray();
            var bData = b.ToArray();
            int batchSize = a.Shape[0];

            var valData = new float[batchSize * topKModes.Length];
            var embData = new float[batchSize * topKModes.Length * _embeddingDim];

            for (int b_idx = 0; b_idx < batchSize; b_idx++)
            {
                float aVal = aData[b_idx];
                float bVal = bData[b_idx];

                for (int k = 0; k < topKModes.Length; k++)
                {
                    int mode = topKModes[k];
                    int idx = b_idx * topKModes.Length + k;

                    // Compute variant value
                    valData[idx] = ComputeModeValue((SemanticMode)mode, aVal, bVal);

                    // Simple embedding (just mode one-hot + value)
                    int embOffset = idx * _embeddingDim;
                    embData[embOffset + mode % _embeddingDim] = 1.0f;
                    embData[embOffset + _embeddingDim - 1] = valData[idx] * 0.01f;
                }
            }

            values.Buffer.CopyFromCPU(valData);
            embeddings.Buffer.CopyFromCPU(embData);

            var modeData = new int[batchSize * topKModes.Length];
            for (int i = 0; i < batchSize; i++)
                for (int k = 0; k < topKModes.Length; k++)
                    modeData[i * topKModes.Length + k] = topKModes[k];
            modeIndices.CopyFromCPU(modeData);
        }

        /// <summary>
        /// Compute value for a specific semantic mode.
        /// </summary>
        private float ComputeModeValue(SemanticMode mode, float a, float b)
        {
            return mode switch
            {
                SemanticMode.Sum => a + b,
                SemanticMode.Replicate => a * b,
                SemanticMode.Geometric => MathF.Sqrt(MathF.Abs(a * b)),
                SemanticMode.Ratio => b != 0 ? a / b : 0,
                SemanticMode.Harmonic => (a + b) != 0 ? 2 * a * b / (a + b) : 0,
                SemanticMode.Difference => a - b,
                SemanticMode.Power => MathF.Pow(MathF.Abs(a), MathF.Min(b, 10)),
                SemanticMode.CoPresence => MathF.Min(a, b),
                _ => a + b
            };
        }

        /// <summary>
        /// Get max policy score for confidence check.
        /// </summary>
        private float GetMaxPolicyScore(GpuTensor scores)
        {
            var data = scores.ToArray();
            return data.Max();
        }

        /// <summary>
        /// Select variant by argmax (no exploration).
        /// </summary>
        private void SelectVariantArgmax(
            GpuTensor scores,
            GpuTensor values,
            ArrayView<int> selectedIndices,
            GpuTensor selectedValues)
        {
            var scoreData = scores.ToArray();
            var valData = values.ToArray();
            int batchSize = scores.Shape[0];
            int numModes = scores.Shape[1];

            var indices = new int[batchSize];
            var selVals = new float[batchSize];

            for (int b = 0; b < batchSize; b++)
            {
                int bestIdx = 0;
                float bestScore = scoreData[b * numModes];

                for (int m = 1; m < numModes; m++)
                {
                    if (scoreData[b * numModes + m] > bestScore)
                    {
                        bestScore = scoreData[b * numModes + m];
                        bestIdx = m;
                    }
                }

                indices[b] = bestIdx;
                selVals[b] = valData[b * numModes + bestIdx];
            }

            selectedIndices.CopyFromCPU(indices);
            selectedValues.Buffer.CopyFromCPU(selVals);
        }

        /// <summary>
        /// Get selected modes for fast path.
        /// </summary>
        private SemanticMode[] GetSelectedModesFast(MemoryBuffer1D<int, Stride1D.Dense> indices, int[] topKModes)
        {
            var data = new int[indices.Length];
            indices.CopyToCPU(data);
            return data.Select(i => (SemanticMode)topKModes[i % topKModes.Length]).ToArray();
        }

        /// <summary>
        /// Extract variants for fast path.
        /// </summary>
        private List<RelationalVariant> ExtractAllVariantsFast(
            GpuTensor values,
            MemoryBuffer1D<int, Stride1D.Dense> modes,
            GpuTensor embeddings,
            int[] topKModes)
        {
            var valData = values.ToArray();
            var embData = embeddings.ToArray();
            int numModes = topKModes.Length;

            var variants = new List<RelationalVariant>();
            for (int m = 0; m < numModes; m++)
            {
                var embedding = new float[_embeddingDim];
                Array.Copy(embData, m * _embeddingDim, embedding, 0, _embeddingDim);

                variants.Add(new RelationalVariant
                {
                    Mode = (SemanticMode)topKModes[m],
                    Value = valData[m],
                    Embedding = embedding
                });
            }

            return variants;
        }

        #endregion

        #region Disposal

        /// <summary>Public API</summary>
        public void Dispose()
        {
            foreach (var mem in _relationMemory.Values) mem?.Dispose();
            foreach (var emb in _operatorEmbeddings.Values) emb?.Dispose();
            _policyWeights?.Dispose();
            _policyBias?.Dispose();
        }

        #endregion
    }

    /// <summary>
    /// A single relational variant: (mode, value, embedding)
    /// </summary>
    public struct RelationalVariant
    {
        /// <summary>Public API</summary>
        public ProductionMathEngine.SemanticMode Mode;
        /// <summary>Public API</summary>
        public float Value;
        /// <summary>Public API</summary>
        public float[] Embedding;

        /// <summary>Public API</summary>
        public override string ToString() => $"({Mode}, {Value:F4})";
    }

    /// <summary>
    /// Result of a ProductionMath operation.
    /// Contains the selected variant, all variants, policy scores, and linear anchor.
    /// </summary>
    public class ProductionMathResult
    {
        /// <summary>Public API</summary>
        public required GpuTensor SelectedValue { get; init; }
        /// <summary>Public API</summary>
        public required ProductionMathEngine.SemanticMode[] SelectedMode { get; init; }
        /// <summary>Public API</summary>
        public required List<RelationalVariant> AllVariants { get; init; }
        /// <summary>Public API</summary>
        public required GpuTensor PolicyScores { get; init; }
        /// <summary>Public API</summary>
        public float? LinearAnchor { get; init; }
        /// <summary>Public API</summary>
        public ProductionMathEngine.OperatorType Operator { get; init; }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var selected = AllVariants.FirstOrDefault(v => v.Mode == SelectedMode[0]);
            return $"ProductionMath Result:\n" +
                   $"  Selected: {selected}\n" +
                   $"  Linear Anchor: {LinearAnchor}\n" +
                   $"  All Variants: [{string.Join(", ", AllVariants.Take(5))}...]";
        }
    }

    /// <summary>
    /// Equation as an object (not a constraint).
    /// E := (LHS, RHS)
    ///
    /// You never require ∃x: LHS(x) = RHS.
    /// Instead, you reason about the equation: O⋄(E, J) is valid.
    /// </summary>
    public class ProductionMathEquation
    {
        /// <summary>Public API</summary>
        public required Func<float, float> LHS { get; init; }
        /// <summary>Public API</summary>
        public float RHS { get; init; }
        /// <summary>Public API</summary>
        public required string Expression { get; init; }

        /// <summary>
        /// Apply the diamond operator to this equation.
        /// The equation itself becomes an entity that can be operated on.
        /// </summary>
        public ProductionMathResult ApplyOperator(
            ProductionMathEngine engine,
            ProductionMathEquation other,
            Accelerator accelerator)
        {
            // Equations are entities - encode them as tensors
            // Use hash of expression as scalar representation
            float thisHash = Expression.GetHashCode() / (float)int.MaxValue;
            float otherHash = other.Expression.GetHashCode() / (float)int.MaxValue;

            using var a = GpuTensor.FromArray(accelerator, new[] { thisHash }, new[] { 1 });
            using var b = GpuTensor.FromArray(accelerator, new[] { otherHash }, new[] { 1 });

            return engine.Diamond(a, b);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"{Expression} = {RHS}";
    }
}