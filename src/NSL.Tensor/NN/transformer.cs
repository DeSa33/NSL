using System;
using System.Collections.Generic;

namespace NSL.Tensor.NN
{
    /// <summary>
    /// Multi-Head Attention layer
    /// </summary>
    public class MultiheadAttention : Module
    {
        private readonly int _embedDim;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly double _dropout;
        private readonly bool _batchFirst;
        private readonly bool _addBiasKv;
        private readonly bool _addZeroAttn;

        private Tensor _wQ, _wK, _wV, _wO;
        private Tensor? _bQ, _bK, _bV, _bO;
        private Tensor? _biasK, _biasV;

        /// <summary>Public API</summary>
        public MultiheadAttention(int embedDim, int numHeads, double dropout = 0, bool bias = true,
            bool addBiasKv = false, bool addZeroAttn = false, bool batchFirst = false)
        {
            if (embedDim % numHeads != 0)
                throw new ArgumentException("embed_dim must be divisible by num_heads");

            _embedDim = embedDim;
            _numHeads = numHeads;
            _headDim = embedDim / numHeads;
            _dropout = dropout;
            _batchFirst = batchFirst;
            _addBiasKv = addBiasKv;
            _addZeroAttn = addZeroAttn;

            // Initialize projection weights
            double scale = Math.Sqrt(1.0 / embedDim);
            _wQ = Tensor.Uniform(new long[] { embedDim, embedDim }, -scale, scale);
            _wK = Tensor.Uniform(new long[] { embedDim, embedDim }, -scale, scale);
            _wV = Tensor.Uniform(new long[] { embedDim, embedDim }, -scale, scale);
            _wO = Tensor.Uniform(new long[] { embedDim, embedDim }, -scale, scale);

            RegisterParameter("in_proj_weight_q", _wQ);
            RegisterParameter("in_proj_weight_k", _wK);
            RegisterParameter("in_proj_weight_v", _wV);
            RegisterParameter("out_proj_weight", _wO);

            if (bias)
            {
                _bQ = Tensor.Zeros(new long[] { embedDim });
                _bK = Tensor.Zeros(new long[] { embedDim });
                _bV = Tensor.Zeros(new long[] { embedDim });
                _bO = Tensor.Zeros(new long[] { embedDim });
                RegisterParameter("in_proj_bias_q", _bQ);
                RegisterParameter("in_proj_bias_k", _bK);
                RegisterParameter("in_proj_bias_v", _bV);
                RegisterParameter("out_proj_bias", _bO);
            }

            if (addBiasKv)
            {
                _biasK = Tensor.Zeros(new long[] { 1, 1, embedDim });
                _biasV = Tensor.Zeros(new long[] { 1, 1, embedDim });
                RegisterParameter("bias_k", _biasK);
                RegisterParameter("bias_v", _biasV);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // Self-attention
            var (output, _) = Forward(input, input, input, null, false);
            return output;
        }

        /// <summary>Public API</summary>
        public (Tensor output, Tensor? attnWeights) Forward(Tensor query, Tensor key, Tensor value,
            Tensor? keyPaddingMask = null, bool needWeights = false, Tensor? attnMask = null)
        {
            // Convert to [seq, batch, embed] if batch_first
            if (_batchFirst)
            {
                query = query.Transpose(0, 1);
                key = key.Transpose(0, 1);
                value = value.Transpose(0, 1);
            }

            var seqLenQ = (int)query.Shape[0];
            var batch = (int)query.Shape[1];
            var seqLenK = (int)key.Shape[0];

            // Project Q, K, V
            var Q = TensorOps.MatMul(query, _wQ.T());
            var K = TensorOps.MatMul(key, _wK.T());
            var V = TensorOps.MatMul(value, _wV.T());

            if (_bQ != null) Q = Q.Add(_bQ);
            if (_bK != null) K = K.Add(_bK);
            if (_bV != null) V = V.Add(_bV);

            // Add bias to K and V if specified
            if (_addBiasKv && _biasK != null && _biasV != null)
            {
                // Broadcast and concatenate bias
                var biasK = _biasK.Repeat(new long[] { 1, batch, 1 });
                var biasV = _biasV.Repeat(new long[] { 1, batch, 1 });
                K = TensorOps.Cat(new[] { K, biasK }, 0);
                V = TensorOps.Cat(new[] { V, biasV }, 0);
                seqLenK++;
            }

            // Reshape for multi-head attention: [seq, batch, embed] -> [batch, num_heads, seq, head_dim]
            Q = Q.Reshape(new long[] { seqLenQ, batch, _numHeads, _headDim }).Permute(new[] { 1, 2, 0, 3 });
            K = K.Reshape(new long[] { seqLenK, batch, _numHeads, _headDim }).Permute(new[] { 1, 2, 0, 3 });
            V = V.Reshape(new long[] { seqLenK, batch, _numHeads, _headDim }).Permute(new[] { 1, 2, 0, 3 });

            // Scaled dot-product attention
            double scale = 1.0 / Math.Sqrt(_headDim);
            var scores = TensorOps.MatMul(Q, K.Transpose(-2, -1)).Mul(scale);

            // Apply attention mask
            if (attnMask != null)
            {
                // Mask shape should be [seqLenQ, seqLenK] or [batch, num_heads, seqLenQ, seqLenK]
                scores = scores.Add(attnMask.Mul(-1e9));
            }

            // Apply key padding mask
            if (keyPaddingMask != null)
            {
                // keyPaddingMask shape: [batch, seqLenK]
                // Expand to [batch, 1, 1, seqLenK]
                var mask = keyPaddingMask.Unsqueeze(1).Unsqueeze(2);
                scores = scores.Add(mask.Mul(-1e9));
            }

            var attnWeights = scores.Softmax(-1);

            // Apply dropout
            if (_dropout > 0 && Training)
            {
                attnWeights = F.Dropout(attnWeights, _dropout, true);
            }

            // Compute output
            var attnOutput = TensorOps.MatMul(attnWeights, V);

            // Reshape back: [batch, num_heads, seqLenQ, head_dim] -> [seqLenQ, batch, embed]
            attnOutput = attnOutput.Permute(new[] { 2, 0, 1, 3 }).Reshape(new long[] { seqLenQ, batch, _embedDim });

            // Final projection
            var output = TensorOps.MatMul(attnOutput, _wO.T());
            if (_bO != null) output = output.Add(_bO);

            // Convert back to batch_first if needed
            if (_batchFirst)
            {
                output = output.Transpose(0, 1);
            }

            // Average attention weights across heads if needed
            Tensor? avgAttnWeights = null;
            if (needWeights)
            {
                avgAttnWeights = attnWeights.Mean(1, keepDim: false);
            }

            return (output, avgAttnWeights);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"MultiheadAttention(embed_dim={_embedDim}, num_heads={_numHeads}, dropout={_dropout}, batch_first={_batchFirst})";
    }

    /// <summary>
    /// Transformer Encoder Layer
    /// </summary>
    public class TransformerEncoderLayer : Module
    {
        private readonly MultiheadAttention _selfAttn;
        private readonly Linear _linear1;
        private readonly Linear _linear2;
        private readonly Module _norm1;
        private readonly Module _norm2;
        private readonly Dropout _dropout;
        private readonly Dropout _dropout1;
        private readonly Dropout _dropout2;
        private readonly Module _activation;
        private readonly bool _normFirst;

        /// <summary>Public API</summary>
        public TransformerEncoderLayer(int dModel, int nHead, int dimFeedforward = 2048,
            double dropout = 0.1, string activation = "relu", bool normFirst = false, bool batchFirst = false)
        {
            _selfAttn = new MultiheadAttention(dModel, nHead, dropout, batchFirst: batchFirst);
            _linear1 = new Linear(dModel, dimFeedforward);
            _linear2 = new Linear(dimFeedforward, dModel);
            _norm1 = new LayerNorm(dModel);
            _norm2 = new LayerNorm(dModel);
            _dropout = new Dropout(dropout);
            _dropout1 = new Dropout(dropout);
            _dropout2 = new Dropout(dropout);
            _activation = activation == "gelu" ? new GELU() : new ReLU();
            _normFirst = normFirst;

            RegisterModule("self_attn", _selfAttn);
            RegisterModule("linear1", _linear1);
            RegisterModule("linear2", _linear2);
            RegisterModule("norm1", _norm1);
            RegisterModule("norm2", _norm2);
            RegisterModule("dropout", _dropout);
            RegisterModule("dropout1", _dropout1);
            RegisterModule("dropout2", _dropout2);
            RegisterModule("activation", _activation);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return Forward(input, null, null);
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor src, Tensor? srcMask = null, Tensor? srcKeyPaddingMask = null)
        {
            Tensor x;
            if (_normFirst)
            {
                // Pre-norm architecture
                x = src.Add(SelfAttnBlock(_norm1.Forward(src), srcMask, srcKeyPaddingMask));
                x = x.Add(FFBlock(_norm2.Forward(x)));
            }
            else
            {
                // Post-norm architecture (original Transformer)
                x = _norm1.Forward(src.Add(SelfAttnBlock(src, srcMask, srcKeyPaddingMask)));
                x = _norm2.Forward(x.Add(FFBlock(x)));
            }
            return x;
        }

        private Tensor SelfAttnBlock(Tensor x, Tensor? attnMask, Tensor? keyPaddingMask)
        {
            var (attnOutput, _) = _selfAttn.Forward(x, x, x, keyPaddingMask, false, attnMask);
            return _dropout1.Forward(attnOutput);
        }

        private Tensor FFBlock(Tensor x)
        {
            x = _linear1.Forward(x);
            x = _activation.Forward(x);
            x = _dropout.Forward(x);
            x = _linear2.Forward(x);
            return _dropout2.Forward(x);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"TransformerEncoderLayer()";
    }

    /// <summary>
    /// Transformer Decoder Layer
    /// </summary>
    public class TransformerDecoderLayer : Module
    {
        private readonly MultiheadAttention _selfAttn;
        private readonly MultiheadAttention _multiheadAttn;
        private readonly Linear _linear1;
        private readonly Linear _linear2;
        private readonly Module _norm1;
        private readonly Module _norm2;
        private readonly Module _norm3;
        private readonly Dropout _dropout;
        private readonly Dropout _dropout1;
        private readonly Dropout _dropout2;
        private readonly Dropout _dropout3;
        private readonly Module _activation;
        private readonly bool _normFirst;

        /// <summary>Public API</summary>
        public TransformerDecoderLayer(int dModel, int nHead, int dimFeedforward = 2048,
            double dropout = 0.1, string activation = "relu", bool normFirst = false, bool batchFirst = false)
        {
            _selfAttn = new MultiheadAttention(dModel, nHead, dropout, batchFirst: batchFirst);
            _multiheadAttn = new MultiheadAttention(dModel, nHead, dropout, batchFirst: batchFirst);
            _linear1 = new Linear(dModel, dimFeedforward);
            _linear2 = new Linear(dimFeedforward, dModel);
            _norm1 = new LayerNorm(dModel);
            _norm2 = new LayerNorm(dModel);
            _norm3 = new LayerNorm(dModel);
            _dropout = new Dropout(dropout);
            _dropout1 = new Dropout(dropout);
            _dropout2 = new Dropout(dropout);
            _dropout3 = new Dropout(dropout);
            _activation = activation == "gelu" ? new GELU() : new ReLU();
            _normFirst = normFirst;

            RegisterModule("self_attn", _selfAttn);
            RegisterModule("multihead_attn", _multiheadAttn);
            RegisterModule("linear1", _linear1);
            RegisterModule("linear2", _linear2);
            RegisterModule("norm1", _norm1);
            RegisterModule("norm2", _norm2);
            RegisterModule("norm3", _norm3);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            throw new InvalidOperationException("Use Forward(tgt, memory, ...) instead");
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor tgt, Tensor memory, Tensor? tgtMask = null, Tensor? memoryMask = null,
            Tensor? tgtKeyPaddingMask = null, Tensor? memoryKeyPaddingMask = null)
        {
            Tensor x;
            if (_normFirst)
            {
                x = tgt.Add(SelfAttnBlock(_norm1.Forward(tgt), tgtMask, tgtKeyPaddingMask));
                x = x.Add(MultiheadAttnBlock(_norm2.Forward(x), memory, memoryMask, memoryKeyPaddingMask));
                x = x.Add(FFBlock(_norm3.Forward(x)));
            }
            else
            {
                x = _norm1.Forward(tgt.Add(SelfAttnBlock(tgt, tgtMask, tgtKeyPaddingMask)));
                x = _norm2.Forward(x.Add(MultiheadAttnBlock(x, memory, memoryMask, memoryKeyPaddingMask)));
                x = _norm3.Forward(x.Add(FFBlock(x)));
            }
            return x;
        }

        private Tensor SelfAttnBlock(Tensor x, Tensor? attnMask, Tensor? keyPaddingMask)
        {
            var (attnOutput, _) = _selfAttn.Forward(x, x, x, keyPaddingMask, false, attnMask);
            return _dropout1.Forward(attnOutput);
        }

        private Tensor MultiheadAttnBlock(Tensor x, Tensor memory, Tensor? attnMask, Tensor? keyPaddingMask)
        {
            var (attnOutput, _) = _multiheadAttn.Forward(x, memory, memory, keyPaddingMask, false, attnMask);
            return _dropout2.Forward(attnOutput);
        }

        private Tensor FFBlock(Tensor x)
        {
            x = _linear1.Forward(x);
            x = _activation.Forward(x);
            x = _dropout.Forward(x);
            x = _linear2.Forward(x);
            return _dropout3.Forward(x);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"TransformerDecoderLayer()";
    }

    /// <summary>
    /// Transformer Encoder - stack of encoder layers
    /// </summary>
    public class TransformerEncoder : Module
    {
        private readonly List<TransformerEncoderLayer> _layers;
        private readonly Module? _norm;

        /// <summary>Public API</summary>
        public TransformerEncoder(TransformerEncoderLayer encoderLayer, int numLayers, Module? norm = null)
        {
            _layers = new List<TransformerEncoderLayer>();
            for (int i = 0; i < numLayers; i++)
            {
                // Clone the layer configuration
                _layers.Add(encoderLayer);
                RegisterModule($"layers.{i}", encoderLayer);
            }
            _norm = norm;
            if (norm != null)
                RegisterModule("norm", norm);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return Forward(input, null, null);
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor src, Tensor? mask = null, Tensor? srcKeyPaddingMask = null)
        {
            var output = src;
            foreach (var layer in _layers)
            {
                output = layer.Forward(output, mask, srcKeyPaddingMask);
            }
            if (_norm != null)
                output = _norm.Forward(output);
            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"TransformerEncoder(num_layers={_layers.Count})";
    }

    /// <summary>
    /// Transformer Decoder - stack of decoder layers
    /// </summary>
    public class TransformerDecoder : Module
    {
        private readonly List<TransformerDecoderLayer> _layers;
        private readonly Module? _norm;

        /// <summary>Public API</summary>
        public TransformerDecoder(TransformerDecoderLayer decoderLayer, int numLayers, Module? norm = null)
        {
            _layers = new List<TransformerDecoderLayer>();
            for (int i = 0; i < numLayers; i++)
            {
                _layers.Add(decoderLayer);
                RegisterModule($"layers.{i}", decoderLayer);
            }
            _norm = norm;
            if (norm != null)
                RegisterModule("norm", norm);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            throw new InvalidOperationException("Use Forward(tgt, memory, ...) instead");
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor tgt, Tensor memory, Tensor? tgtMask = null, Tensor? memoryMask = null,
            Tensor? tgtKeyPaddingMask = null, Tensor? memoryKeyPaddingMask = null)
        {
            var output = tgt;
            foreach (var layer in _layers)
            {
                output = layer.Forward(output, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask);
            }
            if (_norm != null)
                output = _norm.Forward(output);
            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"TransformerDecoder(num_layers={_layers.Count})";
    }

    /// <summary>
    /// Full Transformer model
    /// </summary>
    public class Transformer : Module
    {
        private readonly TransformerEncoder _encoder;
        private readonly TransformerDecoder _decoder;
        private readonly int _dModel;
        private readonly int _nHead;

        /// <summary>Public API</summary>
        public Transformer(int dModel = 512, int nHead = 8, int numEncoderLayers = 6, int numDecoderLayers = 6,
            int dimFeedforward = 2048, double dropout = 0.1, string activation = "relu", bool normFirst = false, bool batchFirst = false)
        {
            _dModel = dModel;
            _nHead = nHead;

            var encoderLayer = new TransformerEncoderLayer(dModel, nHead, dimFeedforward, dropout, activation, normFirst, batchFirst);
            var encoderNorm = new LayerNorm(dModel);
            _encoder = new TransformerEncoder(encoderLayer, numEncoderLayers, encoderNorm);

            var decoderLayer = new TransformerDecoderLayer(dModel, nHead, dimFeedforward, dropout, activation, normFirst, batchFirst);
            var decoderNorm = new LayerNorm(dModel);
            _decoder = new TransformerDecoder(decoderLayer, numDecoderLayers, decoderNorm);

            RegisterModule("encoder", _encoder);
            RegisterModule("decoder", _decoder);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            throw new InvalidOperationException("Use Forward(src, tgt, ...) instead");
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor src, Tensor tgt, Tensor? srcMask = null, Tensor? tgtMask = null,
            Tensor? memoryMask = null, Tensor? srcKeyPaddingMask = null, Tensor? tgtKeyPaddingMask = null,
            Tensor? memoryKeyPaddingMask = null)
        {
            var memory = _encoder.Forward(src, srcMask, srcKeyPaddingMask);
            var output = _decoder.Forward(tgt, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask);
            return output;
        }

        /// <summary>
        /// Generate a causal mask for autoregressive decoding
        /// </summary>
        public static Tensor GenerateSquareSubsequentMask(int sz)
        {
            var mask = Tensor.Zeros(new long[] { sz, sz });
            for (int i = 0; i < sz; i++)
            {
                for (int j = i + 1; j < sz; j++)
                {
                    mask[i, j] = 1; // Will be multiplied by -inf later
                }
            }
            return mask;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Transformer(d_model={_dModel}, nhead={_nHead})";
    }

    /// <summary>
    /// Vision Transformer (ViT) patch embedding
    /// </summary>
    public class PatchEmbedding : Module
    {
        private readonly int _imgSize;
        private readonly int _patchSize;
        private readonly int _numPatches;
        private readonly Conv2d _proj;
        private readonly int _embedDim;

        /// <summary>Public API</summary>
        public int NumPatches => _numPatches;

        /// <summary>Public API</summary>
        public PatchEmbedding(int imgSize = 224, int patchSize = 16, int inChannels = 3, int embedDim = 768)
        {
            _imgSize = imgSize;
            _patchSize = patchSize;
            _numPatches = (imgSize / patchSize) * (imgSize / patchSize);
            _embedDim = embedDim;

            _proj = new Conv2d(inChannels, embedDim, patchSize, stride: patchSize);
            RegisterModule("proj", _proj);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, channels, height, width]
            var batch = (int)input.Shape[0];

            // Project patches: [batch, embed_dim, num_patches_h, num_patches_w]
            var x = _proj.Forward(input);

            // Flatten spatial dimensions: [batch, embed_dim, num_patches]
            x = x.Flatten(2, 3);

            // Transpose to [batch, num_patches, embed_dim]
            x = x.Transpose(1, 2);

            return x;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"PatchEmbedding(img_size={_imgSize}, patch_size={_patchSize}, embed_dim={_embedDim})";
    }

    /// <summary>
    /// Cross-Attention layer for cross-modal attention
    /// </summary>
    public class CrossAttention : Module
    {
        private readonly MultiheadAttention _attn;
        private readonly Module _norm1;
        private readonly Module _norm2;
        private readonly Dropout _dropout;

        /// <summary>Public API</summary>
        public CrossAttention(int embedDim, int numHeads, double dropout = 0.1, bool batchFirst = true)
        {
            _attn = new MultiheadAttention(embedDim, numHeads, dropout, batchFirst: batchFirst);
            _norm1 = new LayerNorm(embedDim);
            _norm2 = new LayerNorm(embedDim);
            _dropout = new Dropout(dropout);

            RegisterModule("attn", _attn);
            RegisterModule("norm1", _norm1);
            RegisterModule("norm2", _norm2);
            RegisterModule("dropout", _dropout);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            throw new InvalidOperationException("Use Forward(query, context) instead");
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor query, Tensor context, Tensor? mask = null)
        {
            var normalizedQuery = _norm1.Forward(query);
            var normalizedContext = _norm2.Forward(context);

            var (attnOutput, _) = _attn.Forward(normalizedQuery, normalizedContext, normalizedContext, attnMask: mask);

            return query.Add(_dropout.Forward(attnOutput));
        }

        /// <summary>Public API</summary>
        public override string ToString() => "CrossAttention()";
    }

    /// <summary>
    /// Rotary Position Embedding (RoPE) for modern transformers
    /// </summary>
    public class RotaryEmbedding : Module
    {
        private readonly int _dim;
        private readonly int _maxSeqLen;
        private Tensor _cosCache;
        private Tensor _sinCache;

        /// <summary>Public API</summary>
        public RotaryEmbedding(int dim, int maxSeqLen = 2048, double baseFreq = 10000)
        {
            _dim = dim;
            _maxSeqLen = maxSeqLen;

            // Precompute frequencies
            var freqs = new double[dim / 2];
            for (int i = 0; i < dim / 2; i++)
            {
                freqs[i] = 1.0 / Math.Pow(baseFreq, 2.0 * i / dim);
            }

            // Create position-frequency products
            _cosCache = Tensor.Zeros(new long[] { maxSeqLen, dim / 2 });
            _sinCache = Tensor.Zeros(new long[] { maxSeqLen, dim / 2 });

            for (int pos = 0; pos < maxSeqLen; pos++)
            {
                for (int i = 0; i < dim / 2; i++)
                {
                    double angle = pos * freqs[i];
                    _cosCache[pos, i] = Math.Cos(angle);
                    _sinCache[pos, i] = Math.Sin(angle);
                }
            }

            RegisterBuffer("cos_cache", _cosCache);
            RegisterBuffer("sin_cache", _sinCache);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, seq_len, num_heads, head_dim]
            var seqLen = (int)input.Shape[1];

            // Get cached sin/cos for this sequence length
            var cos = _cosCache.Slice(0, 0, seqLen);
            var sin = _sinCache.Slice(0, 0, seqLen);

            return ApplyRotaryEmb(input, cos, sin);
        }

        private Tensor ApplyRotaryEmb(Tensor x, Tensor cos, Tensor sin)
        {
            // Split into real and imaginary parts
            var x1 = x.Slice(-1, 0, _dim / 2);
            var x2 = x.Slice(-1, _dim / 2, _dim);

            // Apply rotation
            var rotated1 = x1.Mul(cos).Sub(x2.Mul(sin));
            var rotated2 = x1.Mul(sin).Add(x2.Mul(cos));

            return TensorOps.Cat(new[] { rotated1, rotated2 }, -1);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"RotaryEmbedding(dim={_dim}, max_seq_len={_maxSeqLen})";
    }

    /// <summary>
    /// ALiBi (Attention with Linear Biases) position encoding
    /// </summary>
    public class ALiBiPositionBias : Module
    {
        private readonly int _numHeads;
        private Tensor _slopes;

        /// <summary>Public API</summary>
        public ALiBiPositionBias(int numHeads)
        {
            _numHeads = numHeads;

            // Compute slopes
            var slopes = new double[numHeads];
            double ratio = Math.Pow(2, -8.0 / numHeads);
            double current = 1.0;
            for (int i = 0; i < numHeads; i++)
            {
                slopes[i] = current;
                current *= ratio;
            }

            _slopes = Tensor.FromArray(slopes, new long[] { numHeads });
            RegisterBuffer("slopes", _slopes);
        }

        /// <summary>Public API</summary>
        public Tensor GetBias(int seqLenQ, int seqLenK)
        {
            // Create relative position matrix
            var bias = Tensor.Zeros(new long[] { _numHeads, seqLenQ, seqLenK });

            for (int h = 0; h < _numHeads; h++)
            {
                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        // Causal bias (only attend to past)
                        if (j > i)
                            bias[h, i, j] = double.NegativeInfinity;
                        else
                            bias[h, i, j] = -_slopes[h] * (i - j);
                    }
                }
            }

            return bias;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var seqLen = (int)input.Shape[^2];
            return GetBias(seqLen, seqLen);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"ALiBiPositionBias(num_heads={_numHeads})";
    }

    /// <summary>
    /// Flash Attention-like efficient attention (simplified implementation)
    /// </summary>
    public class EfficientAttention : Module
    {
        private readonly int _embedDim;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly double _scale;
        private Linear _qProj, _kProj, _vProj, _outProj;

        /// <summary>Public API</summary>
        public EfficientAttention(int embedDim, int numHeads)
        {
            _embedDim = embedDim;
            _numHeads = numHeads;
            _headDim = embedDim / numHeads;
            _scale = 1.0 / Math.Sqrt(_headDim);

            _qProj = new Linear(embedDim, embedDim);
            _kProj = new Linear(embedDim, embedDim);
            _vProj = new Linear(embedDim, embedDim);
            _outProj = new Linear(embedDim, embedDim);

            RegisterModule("q_proj", _qProj);
            RegisterModule("k_proj", _kProj);
            RegisterModule("v_proj", _vProj);
            RegisterModule("out_proj", _outProj);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return Forward(input, input, input, null);
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor query, Tensor key, Tensor value, Tensor? mask = null)
        {
            var batch = (int)query.Shape[0];
            var seqLen = (int)query.Shape[1];

            // Project
            var Q = _qProj.Forward(query);
            var K = _kProj.Forward(key);
            var V = _vProj.Forward(value);

            // Reshape: [batch, seq, embed] -> [batch, num_heads, seq, head_dim]
            Q = Q.Reshape(new long[] { batch, seqLen, _numHeads, _headDim }).Transpose(1, 2);
            K = K.Reshape(new long[] { batch, (int)key.Shape[1], _numHeads, _headDim }).Transpose(1, 2);
            V = V.Reshape(new long[] { batch, (int)value.Shape[1], _numHeads, _headDim }).Transpose(1, 2);

            // Attention
            var scores = TensorOps.MatMul(Q, K.Transpose(-2, -1)).Mul(_scale);

            if (mask != null)
            {
                scores = scores.Add(mask.Mul(-1e9));
            }

            var attnWeights = scores.Softmax(-1);
            var output = TensorOps.MatMul(attnWeights, V);

            // Reshape back
            output = output.Transpose(1, 2).Reshape(new long[] { batch, seqLen, _embedDim });

            return _outProj.Forward(output);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"EfficientAttention(embed_dim={_embedDim}, num_heads={_numHeads})";
    }
}