using System;

namespace NSL.Tensor.NN
{
    #region Linear Layers

    /// <summary>
    /// Applies a linear transformation: y = xW^T + b
    /// </summary>
    public class Linear : Module
    {
        private readonly int _inFeatures;
        private readonly int _outFeatures;
        private readonly bool _useBias;
        private Tensor _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public int InFeatures => _inFeatures;
        /// <summary>Public API</summary>
        public int OutFeatures => _outFeatures;
        /// <summary>Public API</summary>
        public Tensor Weight => _weight;
        /// <summary>Public API</summary>
        public Tensor? Bias => _bias;

        /// <summary>Public API</summary>
        public Linear(int inFeatures, int outFeatures, bool bias = true)
        {
            _inFeatures = inFeatures;
            _outFeatures = outFeatures;
            _useBias = bias;

            // Kaiming uniform initialization
            double bound = Math.Sqrt(1.0 / inFeatures);
            _weight = Tensor.Uniform(new long[] { outFeatures, inFeatures }, -bound, bound);
            RegisterParameter("weight", _weight);

            if (bias)
            {
                _bias = Tensor.Uniform(new long[] { outFeatures }, -bound, bound);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, in_features] or [in_features]
            // weight: [out_features, in_features]
            // output: [batch, out_features] or [out_features]
            var output = TensorOps.MatMul(input, _weight.T());
            if (_bias != null)
                output = output.Add(_bias);
            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Linear(in_features={_inFeatures}, out_features={_outFeatures}, bias={_useBias})";
    }

    /// <summary>
    /// Bilinear transformation: y = x1^T A x2 + b
    /// </summary>
    public class Bilinear : Module
    {
        private readonly int _in1Features, _in2Features, _outFeatures;
        private readonly bool _useBias;
        private Tensor _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public Bilinear(int in1Features, int in2Features, int outFeatures, bool bias = true)
        {
            _in1Features = in1Features;
            _in2Features = in2Features;
            _outFeatures = outFeatures;
            _useBias = bias;

            double bound = Math.Sqrt(1.0 / in1Features);
            _weight = Tensor.Uniform(new long[] { outFeatures, in1Features, in2Features }, -bound, bound);
            RegisterParameter("weight", _weight);

            if (bias)
            {
                _bias = Tensor.Uniform(new long[] { outFeatures }, -bound, bound);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            throw new InvalidOperationException("Use Forward(Tensor input1, Tensor input2) instead");
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor input1, Tensor input2)
        {
            // Simplified bilinear: for each output feature k, compute x1^T W[k] x2
            var batchSize = input1.Shape[0];
            var result = Tensor.Zeros(new long[] { batchSize, _outFeatures });

            for (int b = 0; b < batchSize; b++)
            {
                for (int k = 0; k < _outFeatures; k++)
                {
                    double sum = 0;
                    for (int i = 0; i < _in1Features; i++)
                    {
                        for (int j = 0; j < _in2Features; j++)
                        {
                            sum += input1[b, i] * _weight[k, i, j] * input2[b, j];
                        }
                    }
                    result[b, k] = sum;
                }
            }

            if (_bias != null)
                result = result.Add(_bias);
            return result;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Bilinear(in1_features={_in1Features}, in2_features={_in2Features}, out_features={_outFeatures}, bias={_useBias})";
    }

    #endregion

    #region Convolutional Layers

    /// <summary>
    /// 1D Convolution layer
    /// </summary>
    public class Conv1d : Module
    {
        private readonly int _inChannels, _outChannels, _kernelSize;
        private readonly int _stride, _padding, _dilation;
        private readonly bool _useBias;
        private Tensor _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public Conv1d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, int dilation = 1, bool bias = true)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelSize = kernelSize;
            _stride = stride;
            _padding = padding;
            _dilation = dilation;
            _useBias = bias;

            double k = 1.0 / (inChannels * kernelSize);
            double bound = Math.Sqrt(k);
            _weight = Tensor.Uniform(new long[] { outChannels, inChannels, kernelSize }, -bound, bound);
            RegisterParameter("weight", _weight);

            if (bias)
            {
                _bias = Tensor.Uniform(new long[] { outChannels }, -bound, bound);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, in_channels, length]
            // weight: [out_channels, in_channels, kernel_size]
            // output: [batch, out_channels, output_length]

            var batch = (int)input.Shape[0];
            var inputLength = (int)input.Shape[2];
            var effectiveKernel = _dilation * (_kernelSize - 1) + 1;
            var outputLength = (inputLength + 2 * _padding - effectiveKernel) / _stride + 1;

            var output = Tensor.Zeros(new long[] { batch, _outChannels, outputLength });

            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < _outChannels; oc++)
                {
                    for (int ol = 0; ol < outputLength; ol++)
                    {
                        double sum = 0;
                        for (int ic = 0; ic < _inChannels; ic++)
                        {
                            for (int k = 0; k < _kernelSize; k++)
                            {
                                int il = ol * _stride - _padding + k * _dilation;
                                if (il >= 0 && il < inputLength)
                                {
                                    sum += input[b, ic, il] * _weight[oc, ic, k];
                                }
                            }
                        }
                        output[b, oc, ol] = sum + (_bias != null ? _bias[oc] : 0);
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Conv1d({_inChannels}, {_outChannels}, kernel_size={_kernelSize}, stride={_stride}, padding={_padding})";
    }

    /// <summary>
    /// 2D Convolution layer
    /// </summary>
    public class Conv2d : Module
    {
        private readonly int _inChannels, _outChannels;
        private readonly int _kernelH, _kernelW;
        private readonly int _strideH, _strideW;
        private readonly int _paddingH, _paddingW;
        private readonly int _dilationH, _dilationW;
        private readonly bool _useBias;
        private Tensor _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public Tensor Weight => _weight;
        /// <summary>Public API</summary>
        public Tensor? Bias => _bias;

        /// <summary>Public API</summary>
        public Conv2d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, int dilation = 1, bool bias = true)
            : this(inChannels, outChannels, kernelSize, kernelSize, stride, stride, padding, padding, dilation, dilation, bias) { }

        /// <summary>Public API</summary>
        public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int strideH = 1, int strideW = 1,
            int paddingH = 0, int paddingW = 0, int dilationH = 1, int dilationW = 1, bool bias = true)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelH = kernelH;
            _kernelW = kernelW;
            _strideH = strideH;
            _strideW = strideW;
            _paddingH = paddingH;
            _paddingW = paddingW;
            _dilationH = dilationH;
            _dilationW = dilationW;
            _useBias = bias;

            double k = 1.0 / (inChannels * kernelH * kernelW);
            double bound = Math.Sqrt(k);
            _weight = Tensor.Uniform(new long[] { outChannels, inChannels, kernelH, kernelW }, -bound, bound);
            RegisterParameter("weight", _weight);

            if (bias)
            {
                _bias = Tensor.Uniform(new long[] { outChannels }, -bound, bound);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, in_channels, height, width]
            // weight: [out_channels, in_channels, kernel_h, kernel_w]
            // output: [batch, out_channels, out_height, out_width]

            var batch = (int)input.Shape[0];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];

            var effectiveKernelH = _dilationH * (_kernelH - 1) + 1;
            var effectiveKernelW = _dilationW * (_kernelW - 1) + 1;
            var outputH = (inputH + 2 * _paddingH - effectiveKernelH) / _strideH + 1;
            var outputW = (inputW + 2 * _paddingW - effectiveKernelW) / _strideW + 1;

            var output = Tensor.Zeros(new long[] { batch, _outChannels, outputH, outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < _outChannels; oc++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            double sum = 0;
                            for (int ic = 0; ic < _inChannels; ic++)
                            {
                                for (int kh = 0; kh < _kernelH; kh++)
                                {
                                    for (int kw = 0; kw < _kernelW; kw++)
                                    {
                                        int ih = oh * _strideH - _paddingH + kh * _dilationH;
                                        int iw = ow * _strideW - _paddingW + kw * _dilationW;
                                        if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW)
                                        {
                                            sum += input[b, ic, ih, iw] * _weight[oc, ic, kh, kw];
                                        }
                                    }
                                }
                            }
                            output[b, oc, oh, ow] = sum + (_bias != null ? _bias[oc] : 0);
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Conv2d({_inChannels}, {_outChannels}, kernel_size=({_kernelH}, {_kernelW}), stride=({_strideH}, {_strideW}), padding=({_paddingH}, {_paddingW}))";
    }

    /// <summary>
    /// Transposed 2D Convolution (deconvolution)
    /// </summary>
    public class ConvTranspose2d : Module
    {
        private readonly int _inChannels, _outChannels;
        private readonly int _kernelH, _kernelW;
        private readonly int _strideH, _strideW;
        private readonly int _paddingH, _paddingW;
        private readonly int _outputPaddingH, _outputPaddingW;
        private readonly bool _useBias;
        private Tensor _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public ConvTranspose2d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, int outputPadding = 0, bool bias = true)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelH = _kernelW = kernelSize;
            _strideH = _strideW = stride;
            _paddingH = _paddingW = padding;
            _outputPaddingH = _outputPaddingW = outputPadding;
            _useBias = bias;

            double k = 1.0 / (inChannels * kernelSize * kernelSize);
            double bound = Math.Sqrt(k);
            _weight = Tensor.Uniform(new long[] { inChannels, outChannels, kernelSize, kernelSize }, -bound, bound);
            RegisterParameter("weight", _weight);

            if (bias)
            {
                _bias = Tensor.Uniform(new long[] { outChannels }, -bound, bound);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // Simplified transpose convolution
            var batch = (int)input.Shape[0];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];

            var outputH = (inputH - 1) * _strideH - 2 * _paddingH + _kernelH + _outputPaddingH;
            var outputW = (inputW - 1) * _strideW - 2 * _paddingW + _kernelW + _outputPaddingW;

            var output = Tensor.Zeros(new long[] { batch, _outChannels, outputH, outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int ic = 0; ic < _inChannels; ic++)
                {
                    for (int ih = 0; ih < inputH; ih++)
                    {
                        for (int iw = 0; iw < inputW; iw++)
                        {
                            double val = input[b, ic, ih, iw];
                            for (int oc = 0; oc < _outChannels; oc++)
                            {
                                for (int kh = 0; kh < _kernelH; kh++)
                                {
                                    for (int kw = 0; kw < _kernelW; kw++)
                                    {
                                        int oh = ih * _strideH - _paddingH + kh;
                                        int ow = iw * _strideW - _paddingW + kw;
                                        if (oh >= 0 && oh < outputH && ow >= 0 && ow < outputW)
                                        {
                                            output[b, oc, oh, ow] += val * _weight[ic, oc, kh, kw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (_bias != null)
            {
                for (int b = 0; b < batch; b++)
                    for (int oc = 0; oc < _outChannels; oc++)
                        for (int oh = 0; oh < outputH; oh++)
                            for (int ow = 0; ow < outputW; ow++)
                                output[b, oc, oh, ow] += _bias[oc];
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"ConvTranspose2d({_inChannels}, {_outChannels}, kernel_size={_kernelH}, stride={_strideH}, padding={_paddingH})";
    }

    #endregion

    #region Normalization Layers

    /// <summary>
    /// Batch Normalization for 1D inputs
    /// </summary>
    public class BatchNorm1d : Module
    {
        private readonly int _numFeatures;
        private readonly double _eps;
        private readonly double _momentum;
        private readonly bool _affine;
        private readonly bool _trackRunningStats;
        private Tensor? _weight;
        private Tensor? _bias;
        private Tensor _runningMean;
        private Tensor _runningVar;
        private long _numBatchesTracked;

        /// <summary>Public API</summary>
        public BatchNorm1d(int numFeatures, double eps = 1e-5, double momentum = 0.1, bool affine = true, bool trackRunningStats = true)
        {
            _numFeatures = numFeatures;
            _eps = eps;
            _momentum = momentum;
            _affine = affine;
            _trackRunningStats = trackRunningStats;

            if (affine)
            {
                _weight = Tensor.Ones(new long[] { numFeatures });
                _bias = Tensor.Zeros(new long[] { numFeatures });
                RegisterParameter("weight", _weight);
                RegisterParameter("bias", _bias);
            }

            _runningMean = Tensor.Zeros(new long[] { numFeatures });
            _runningVar = Tensor.Ones(new long[] { numFeatures });
            RegisterBuffer("running_mean", _runningMean);
            RegisterBuffer("running_var", _runningVar);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, features] or [batch, features, length]
            Tensor mean, variance;

            if (Training)
            {
                if (input.NDim == 2)
                {
                    mean = input.Mean(0, keepDim: true);
                    variance = input.Var(0, keepDim: true);
                }
                else
                {
                    // [batch, features, length] -> mean/var over batch and length
                    var flat = input.Transpose(1, 2).Reshape(new long[] { -1, _numFeatures });
                    mean = flat.Mean(0, keepDim: false);
                    variance = flat.Var(0, keepDim: false);
                }

                if (_trackRunningStats)
                {
                    _runningMean = _runningMean.Mul(1 - _momentum).Add(mean.Detach().Mul(_momentum));
                    _runningVar = _runningVar.Mul(1 - _momentum).Add(variance.Detach().Mul(_momentum));
                    _numBatchesTracked++;
                }
            }
            else
            {
                mean = _runningMean;
                variance = _runningVar;
            }

            // Normalize
            var output = input.Sub(mean).Div(variance.Add(_eps).Sqrt());

            if (_affine && _weight != null && _bias != null)
            {
                output = output.Mul(_weight).Add(_bias);
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"BatchNorm1d({_numFeatures}, eps={_eps}, momentum={_momentum}, affine={_affine})";
    }

    /// <summary>
    /// Batch Normalization for 2D inputs (images)
    /// </summary>
    public class BatchNorm2d : Module
    {
        private readonly int _numFeatures;
        private readonly double _eps;
        private readonly double _momentum;
        private readonly bool _affine;
        private readonly bool _trackRunningStats;
        private Tensor? _weight;
        private Tensor? _bias;
        private Tensor _runningMean;
        private Tensor _runningVar;

        /// <summary>Public API</summary>
        public BatchNorm2d(int numFeatures, double eps = 1e-5, double momentum = 0.1, bool affine = true, bool trackRunningStats = true)
        {
            _numFeatures = numFeatures;
            _eps = eps;
            _momentum = momentum;
            _affine = affine;
            _trackRunningStats = trackRunningStats;

            if (affine)
            {
                _weight = Tensor.Ones(new long[] { numFeatures });
                _bias = Tensor.Zeros(new long[] { numFeatures });
                RegisterParameter("weight", _weight);
                RegisterParameter("bias", _bias);
            }

            _runningMean = Tensor.Zeros(new long[] { numFeatures });
            _runningVar = Tensor.Ones(new long[] { numFeatures });
            RegisterBuffer("running_mean", _runningMean);
            RegisterBuffer("running_var", _runningVar);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, channels, height, width]
            var batch = (int)input.Shape[0];
            var height = (int)input.Shape[2];
            var width = (int)input.Shape[3];

            Tensor mean, variance;

            if (Training)
            {
                // Compute mean and variance per channel over batch, height, width
                mean = Tensor.Zeros(new long[] { _numFeatures });
                variance = Tensor.Zeros(new long[] { _numFeatures });
                int n = batch * height * width;

                for (int c = 0; c < _numFeatures; c++)
                {
                    double sum = 0, sumSq = 0;
                    for (int b = 0; b < batch; b++)
                        for (int h = 0; h < height; h++)
                            for (int w = 0; w < width; w++)
                            {
                                double val = input[b, c, h, w];
                                sum += val;
                                sumSq += val * val;
                            }
                    mean[c] = sum / n;
                    variance[c] = sumSq / n - mean[c] * mean[c];
                }

                if (_trackRunningStats)
                {
                    _runningMean = _runningMean.Mul(1 - _momentum).Add(mean.Detach().Mul(_momentum));
                    _runningVar = _runningVar.Mul(1 - _momentum).Add(variance.Detach().Mul(_momentum));
                }
            }
            else
            {
                mean = _runningMean;
                variance = _runningVar;
            }

            // Normalize
            var output = Tensor.Zeros(input.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < _numFeatures; c++)
                {
                    double m = mean[c];
                    double std = Math.Sqrt(variance[c] + _eps);
                    double scale = _affine && _weight != null ? _weight[c] : 1.0;
                    double shift = _affine && _bias != null ? _bias[c] : 0.0;

                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                            output[b, c, h, w] = (input[b, c, h, w] - m) / std * scale + shift;
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"BatchNorm2d({_numFeatures}, eps={_eps}, momentum={_momentum}, affine={_affine})";
    }

    /// <summary>
    /// Layer Normalization
    /// </summary>
    public class LayerNorm : Module
    {
        private readonly long[] _normalizedShape;
        private readonly double _eps;
        private readonly bool _elementwiseAffine;
        private Tensor? _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public LayerNorm(long[] normalizedShape, double eps = 1e-5, bool elementwiseAffine = true)
        {
            _normalizedShape = normalizedShape;
            _eps = eps;
            _elementwiseAffine = elementwiseAffine;

            if (elementwiseAffine)
            {
                _weight = Tensor.Ones(normalizedShape);
                _bias = Tensor.Zeros(normalizedShape);
                RegisterParameter("weight", _weight);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public LayerNorm(int normalizedShape, double eps = 1e-5, bool elementwiseAffine = true)
            : this(new long[] { normalizedShape }, eps, elementwiseAffine) { }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // Normalize over the last dimensions
            var numNormDims = _normalizedShape.Length;
            var startDim = input.NDim - numNormDims;

            // Calculate mean and variance over normalized dimensions
            var flatSize = 1L;
            for (int i = startDim; i < input.NDim; i++)
                flatSize *= input.Shape[i];

            var output = Tensor.Zeros(input.Shape);
            var leadingShape = 1L;
            for (int i = 0; i < startDim; i++)
                leadingShape *= input.Shape[i];

            var inputData = input.ToArray();
            var outputData = new double[inputData.Length];

            for (long outer = 0; outer < leadingShape; outer++)
            {
                // Calculate mean
                double sum = 0;
                for (long inner = 0; inner < flatSize; inner++)
                    sum += inputData[outer * flatSize + inner];
                double mean = sum / flatSize;

                // Calculate variance
                double varSum = 0;
                for (long inner = 0; inner < flatSize; inner++)
                {
                    double diff = inputData[outer * flatSize + inner] - mean;
                    varSum += diff * diff;
                }
                double variance = varSum / flatSize;
                double std = Math.Sqrt(variance + _eps);

                // Normalize and apply affine
                for (long inner = 0; inner < flatSize; inner++)
                {
                    double normalized = (inputData[outer * flatSize + inner] - mean) / std;
                    if (_elementwiseAffine && _weight != null && _bias != null)
                    {
                        normalized = normalized * _weight.ToArray()[inner] + _bias.ToArray()[inner];
                    }
                    outputData[outer * flatSize + inner] = normalized;
                }
            }

            return Tensor.FromArray(outputData, input.Shape);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"LayerNorm({string.Join(", ", _normalizedShape)}, eps={_eps}, elementwise_affine={_elementwiseAffine})";
    }

    /// <summary>
    /// Group Normalization
    /// </summary>
    public class GroupNorm : Module
    {
        private readonly int _numGroups;
        private readonly int _numChannels;
        private readonly double _eps;
        private readonly bool _affine;
        private Tensor? _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public GroupNorm(int numGroups, int numChannels, double eps = 1e-5, bool affine = true)
        {
            if (numChannels % numGroups != 0)
                throw new ArgumentException("numChannels must be divisible by numGroups");

            _numGroups = numGroups;
            _numChannels = numChannels;
            _eps = eps;
            _affine = affine;

            if (affine)
            {
                _weight = Tensor.Ones(new long[] { numChannels });
                _bias = Tensor.Zeros(new long[] { numChannels });
                RegisterParameter("weight", _weight);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, channels, *]
            var batch = (int)input.Shape[0];
            var channelsPerGroup = _numChannels / _numGroups;

            var output = input.Clone();

            // For simplicity, handle 4D input [batch, channels, height, width]
            if (input.NDim == 4)
            {
                var height = (int)input.Shape[2];
                var width = (int)input.Shape[3];

                for (int b = 0; b < batch; b++)
                {
                    for (int g = 0; g < _numGroups; g++)
                    {
                        // Calculate mean and variance for this group
                        double sum = 0, sumSq = 0;
                        int count = channelsPerGroup * height * width;

                        for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++)
                            for (int h = 0; h < height; h++)
                                for (int w = 0; w < width; w++)
                                {
                                    double val = input[b, c, h, w];
                                    sum += val;
                                    sumSq += val * val;
                                }

                        double mean = sum / count;
                        double variance = sumSq / count - mean * mean;
                        double std = Math.Sqrt(variance + _eps);

                        // Normalize
                        for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++)
                        {
                            double scale = _affine && _weight != null ? _weight[c] : 1.0;
                            double shift = _affine && _bias != null ? _bias[c] : 0.0;
                            for (int h = 0; h < height; h++)
                                for (int w = 0; w < width; w++)
                                    output[b, c, h, w] = (input[b, c, h, w] - mean) / std * scale + shift;
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"GroupNorm({_numGroups}, {_numChannels}, eps={_eps}, affine={_affine})";
    }

    /// <summary>
    /// Instance Normalization for 2D inputs
    /// </summary>
    public class InstanceNorm2d : Module
    {
        private readonly int _numFeatures;
        private readonly double _eps;
        private readonly bool _affine;
        private Tensor? _weight;
        private Tensor? _bias;

        /// <summary>Public API</summary>
        public InstanceNorm2d(int numFeatures, double eps = 1e-5, bool affine = false)
        {
            _numFeatures = numFeatures;
            _eps = eps;
            _affine = affine;

            if (affine)
            {
                _weight = Tensor.Ones(new long[] { numFeatures });
                _bias = Tensor.Zeros(new long[] { numFeatures });
                RegisterParameter("weight", _weight);
                RegisterParameter("bias", _bias);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, channels, height, width]
            var batch = (int)input.Shape[0];
            var height = (int)input.Shape[2];
            var width = (int)input.Shape[3];
            var output = Tensor.Zeros(input.Shape);

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < _numFeatures; c++)
                {
                    // Calculate mean and variance for this instance
                    double sum = 0, sumSq = 0;
                    int n = height * width;

                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                        {
                            double val = input[b, c, h, w];
                            sum += val;
                            sumSq += val * val;
                        }

                    double mean = sum / n;
                    double variance = sumSq / n - mean * mean;
                    double std = Math.Sqrt(variance + _eps);

                    double scale = _affine && _weight != null ? _weight[c] : 1.0;
                    double shift = _affine && _bias != null ? _bias[c] : 0.0;

                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                            output[b, c, h, w] = (input[b, c, h, w] - mean) / std * scale + shift;
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"InstanceNorm2d({_numFeatures}, eps={_eps}, affine={_affine})";
    }

    #endregion

    #region Dropout Layers

    /// <summary>
    /// Dropout layer
    /// </summary>
    public class Dropout : Module
    {
        private readonly double _p;
        private readonly bool _inplace;
        private static readonly Random _random = new Random();

        /// <summary>Public API</summary>
        public Dropout(double p = 0.5, bool inplace = false)
        {
            if (p < 0 || p > 1)
                throw new ArgumentException("Dropout probability must be between 0 and 1");
            _p = p;
            _inplace = inplace;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            if (!Training || _p == 0)
                return input;

            var mask = Tensor.Zeros(input.Shape);
            var scale = 1.0 / (1.0 - _p);
            var data = mask.ToArray();

            for (int i = 0; i < data.Length; i++)
                data[i] = _random.NextDouble() > _p ? scale : 0;

            mask = Tensor.FromArray(data, input.Shape);
            return input.Mul(mask);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Dropout(p={_p}, inplace={_inplace})";
    }

    /// <summary>
    /// 2D Dropout (drops entire channels)
    /// </summary>
    public class Dropout2d : Module
    {
        private readonly double _p;
        private static readonly Random _random = new Random();

        /// <summary>Public API</summary>
        public Dropout2d(double p = 0.5)
        {
            if (p < 0 || p > 1)
                throw new ArgumentException("Dropout probability must be between 0 and 1");
            _p = p;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            if (!Training || _p == 0)
                return input;

            // input: [batch, channels, height, width]
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var height = (int)input.Shape[2];
            var width = (int)input.Shape[3];

            var output = input.Clone();
            var scale = 1.0 / (1.0 - _p);

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    if (_random.NextDouble() < _p)
                    {
                        // Drop this channel
                        for (int h = 0; h < height; h++)
                            for (int w = 0; w < width; w++)
                                output[b, c, h, w] = 0;
                    }
                    else
                    {
                        // Scale
                        for (int h = 0; h < height; h++)
                            for (int w = 0; w < width; w++)
                                output[b, c, h, w] *= scale;
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Dropout2d(p={_p})";
    }

    /// <summary>
    /// Alpha Dropout (for SELU activations)
    /// </summary>
    public class AlphaDropout : Module
    {
        private readonly double _p;
        private static readonly Random _random = new Random();

        // SELU parameters
        private const double Alpha = 1.6732632423543772;
        private const double Scale = 1.0507009873554804;

        /// <summary>Public API</summary>
        public AlphaDropout(double p = 0.5)
        {
            _p = p;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            if (!Training || _p == 0)
                return input;

            double alphaPrime = -Alpha * Scale;
            double a = 1.0 / Math.Sqrt(_p + alphaPrime * alphaPrime * _p * (1 - _p));
            double b = -a * (1 - _p) * alphaPrime;

            var data = input.ToArray();
            for (int i = 0; i < data.Length; i++)
            {
                if (_random.NextDouble() < _p)
                    data[i] = alphaPrime;
            }

            var output = Tensor.FromArray(data, input.Shape);
            return output.Mul(a).Add(b);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"AlphaDropout(p={_p})";
    }

    #endregion

    #region Pooling Layers

    /// <summary>
    /// 1D Max Pooling
    /// </summary>
    public class MaxPool1d : Module
    {
        private readonly int _kernelSize;
        private readonly int _stride;
        private readonly int _padding;

        /// <summary>Public API</summary>
        public MaxPool1d(int kernelSize, int? stride = null, int padding = 0)
        {
            _kernelSize = kernelSize;
            _stride = stride ?? kernelSize;
            _padding = padding;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, channels, length]
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputLength = (int)input.Shape[2];
            var outputLength = (inputLength + 2 * _padding - _kernelSize) / _stride + 1;

            var output = Tensor.Zeros(new long[] { batch, channels, outputLength });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int ol = 0; ol < outputLength; ol++)
                    {
                        double maxVal = double.NegativeInfinity;
                        for (int k = 0; k < _kernelSize; k++)
                        {
                            int il = ol * _stride - _padding + k;
                            if (il >= 0 && il < inputLength)
                                maxVal = Math.Max(maxVal, input[b, c, il]);
                        }
                        output[b, c, ol] = maxVal;
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"MaxPool1d(kernel_size={_kernelSize}, stride={_stride}, padding={_padding})";
    }

    /// <summary>
    /// 2D Max Pooling
    /// </summary>
    public class MaxPool2d : Module
    {
        private readonly int _kernelH, _kernelW;
        private readonly int _strideH, _strideW;
        private readonly int _paddingH, _paddingW;

        /// <summary>Public API</summary>
        public MaxPool2d(int kernelSize, int? stride = null, int padding = 0)
            : this(kernelSize, kernelSize, stride ?? kernelSize, stride ?? kernelSize, padding, padding) { }

        /// <summary>Public API</summary>
        public MaxPool2d(int kernelH, int kernelW, int strideH, int strideW, int paddingH = 0, int paddingW = 0)
        {
            _kernelH = kernelH;
            _kernelW = kernelW;
            _strideH = strideH;
            _strideW = strideW;
            _paddingH = paddingH;
            _paddingW = paddingW;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, channels, height, width]
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];
            var outputH = (inputH + 2 * _paddingH - _kernelH) / _strideH + 1;
            var outputW = (inputW + 2 * _paddingW - _kernelW) / _strideW + 1;

            var output = Tensor.Zeros(new long[] { batch, channels, outputH, outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            double maxVal = double.NegativeInfinity;
                            for (int kh = 0; kh < _kernelH; kh++)
                            {
                                for (int kw = 0; kw < _kernelW; kw++)
                                {
                                    int ih = oh * _strideH - _paddingH + kh;
                                    int iw = ow * _strideW - _paddingW + kw;
                                    if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW)
                                        maxVal = Math.Max(maxVal, input[b, c, ih, iw]);
                                }
                            }
                            output[b, c, oh, ow] = maxVal;
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"MaxPool2d(kernel_size=({_kernelH}, {_kernelW}), stride=({_strideH}, {_strideW}), padding=({_paddingH}, {_paddingW}))";
    }

    /// <summary>
    /// 2D Average Pooling
    /// </summary>
    public class AvgPool2d : Module
    {
        private readonly int _kernelH, _kernelW;
        private readonly int _strideH, _strideW;
        private readonly int _paddingH, _paddingW;

        /// <summary>Public API</summary>
        public AvgPool2d(int kernelSize, int? stride = null, int padding = 0)
            : this(kernelSize, kernelSize, stride ?? kernelSize, stride ?? kernelSize, padding, padding) { }

        /// <summary>Public API</summary>
        public AvgPool2d(int kernelH, int kernelW, int strideH, int strideW, int paddingH = 0, int paddingW = 0)
        {
            _kernelH = kernelH;
            _kernelW = kernelW;
            _strideH = strideH;
            _strideW = strideW;
            _paddingH = paddingH;
            _paddingW = paddingW;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];
            var outputH = (inputH + 2 * _paddingH - _kernelH) / _strideH + 1;
            var outputW = (inputW + 2 * _paddingW - _kernelW) / _strideW + 1;

            var output = Tensor.Zeros(new long[] { batch, channels, outputH, outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            double sum = 0;
                            int count = 0;
                            for (int kh = 0; kh < _kernelH; kh++)
                            {
                                for (int kw = 0; kw < _kernelW; kw++)
                                {
                                    int ih = oh * _strideH - _paddingH + kh;
                                    int iw = ow * _strideW - _paddingW + kw;
                                    if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW)
                                    {
                                        sum += input[b, c, ih, iw];
                                        count++;
                                    }
                                }
                            }
                            output[b, c, oh, ow] = sum / count;
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"AvgPool2d(kernel_size=({_kernelH}, {_kernelW}), stride=({_strideH}, {_strideW}), padding=({_paddingH}, {_paddingW}))";
    }

    /// <summary>
    /// Adaptive Average Pooling 2D
    /// </summary>
    public class AdaptiveAvgPool2d : Module
    {
        private readonly int _outputH, _outputW;

        /// <summary>Public API</summary>
        public AdaptiveAvgPool2d(int outputSize)
        {
            _outputH = _outputW = outputSize;
        }

        /// <summary>Public API</summary>
        public AdaptiveAvgPool2d(int outputH, int outputW)
        {
            _outputH = outputH;
            _outputW = outputW;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];

            var output = Tensor.Zeros(new long[] { batch, channels, _outputH, _outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < _outputH; oh++)
                    {
                        for (int ow = 0; ow < _outputW; ow++)
                        {
                            // Calculate pooling region
                            int startH = (int)Math.Floor(oh * inputH / (double)_outputH);
                            int endH = (int)Math.Ceiling((oh + 1) * inputH / (double)_outputH);
                            int startW = (int)Math.Floor(ow * inputW / (double)_outputW);
                            int endW = (int)Math.Ceiling((ow + 1) * inputW / (double)_outputW);

                            double sum = 0;
                            int count = 0;
                            for (int ih = startH; ih < endH; ih++)
                            {
                                for (int iw = startW; iw < endW; iw++)
                                {
                                    sum += input[b, c, ih, iw];
                                    count++;
                                }
                            }
                            output[b, c, oh, ow] = sum / count;
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"AdaptiveAvgPool2d(output_size=({_outputH}, {_outputW}))";
    }

    /// <summary>
    /// Adaptive Max Pooling 2D
    /// </summary>
    public class AdaptiveMaxPool2d : Module
    {
        private readonly int _outputH, _outputW;

        /// <summary>Public API</summary>
        public AdaptiveMaxPool2d(int outputSize)
        {
            _outputH = _outputW = outputSize;
        }

        /// <summary>Public API</summary>
        public AdaptiveMaxPool2d(int outputH, int outputW)
        {
            _outputH = outputH;
            _outputW = outputW;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];

            var output = Tensor.Zeros(new long[] { batch, channels, _outputH, _outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < _outputH; oh++)
                    {
                        for (int ow = 0; ow < _outputW; ow++)
                        {
                            int startH = (int)Math.Floor(oh * inputH / (double)_outputH);
                            int endH = (int)Math.Ceiling((oh + 1) * inputH / (double)_outputH);
                            int startW = (int)Math.Floor(ow * inputW / (double)_outputW);
                            int endW = (int)Math.Ceiling((ow + 1) * inputW / (double)_outputW);

                            double maxVal = double.NegativeInfinity;
                            for (int ih = startH; ih < endH; ih++)
                                for (int iw = startW; iw < endW; iw++)
                                    maxVal = Math.Max(maxVal, input[b, c, ih, iw]);

                            output[b, c, oh, ow] = maxVal;
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"AdaptiveMaxPool2d(output_size=({_outputH}, {_outputW}))";
    }

    /// <summary>
    /// Global Average Pooling 2D
    /// </summary>
    public class GlobalAvgPool2d : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, channels, height, width]
            // output: [batch, channels]
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var height = (int)input.Shape[2];
            var width = (int)input.Shape[3];

            var output = Tensor.Zeros(new long[] { batch, channels });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double sum = 0;
                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                            sum += input[b, c, h, w];
                    output[b, c] = sum / (height * width);
                }
            }

            return output;
        }

        /// <summary>Public API</summary>
        public override string ToString() => "GlobalAvgPool2d()";
    }

    #endregion

    #region Embedding Layers

    /// <summary>
    /// Embedding layer - lookup table for discrete tokens
    /// </summary>
    public class Embedding : Module
    {
        private readonly int _numEmbeddings;
        private readonly int _embeddingDim;
        private readonly int? _paddingIdx;
        private Tensor _weight;

        /// <summary>Public API</summary>
        public int NumEmbeddings => _numEmbeddings;
        /// <summary>Public API</summary>
        public int EmbeddingDim => _embeddingDim;
        /// <summary>Public API</summary>
        public Tensor Weight => _weight;

        /// <summary>Public API</summary>
        public Embedding(int numEmbeddings, int embeddingDim, int? paddingIdx = null)
        {
            _numEmbeddings = numEmbeddings;
            _embeddingDim = embeddingDim;
            _paddingIdx = paddingIdx;

            _weight = Tensor.Randn(new long[] { numEmbeddings, embeddingDim });
            RegisterParameter("weight", _weight);

            if (paddingIdx.HasValue)
            {
                // Zero out padding embedding
                for (int i = 0; i < embeddingDim; i++)
                    _weight[paddingIdx.Value, i] = 0;
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: tensor of indices [*]
            // output: [*, embedding_dim]
            var inputShape = input.Shape;
            var outputShape = new long[inputShape.Length + 1];
            Array.Copy(inputShape, outputShape, inputShape.Length);
            outputShape[^1] = _embeddingDim;

            var output = Tensor.Zeros(outputShape);
            var inputData = input.ToArray();
            var flatOutput = output.Flatten();

            for (int i = 0; i < inputData.Length; i++)
            {
                int idx = (int)inputData[i];
                if (idx < 0 || idx >= _numEmbeddings)
                    throw new ArgumentException($"Index {idx} out of range for embedding with {_numEmbeddings} embeddings");

                for (int j = 0; j < _embeddingDim; j++)
                    flatOutput[i * _embeddingDim + j] = _weight[idx, j];
            }

            return flatOutput.Reshape(outputShape);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Embedding({_numEmbeddings}, {_embeddingDim})";
    }

    /// <summary>
    /// Positional Embedding for Transformers
    /// </summary>
    public class PositionalEmbedding : Module
    {
        private readonly int _maxLen;
        private readonly int _embeddingDim;
        private Tensor _pe;

        /// <summary>Public API</summary>
        public PositionalEmbedding(int maxLen, int embeddingDim)
        {
            _maxLen = maxLen;
            _embeddingDim = embeddingDim;

            // Create sinusoidal positional encodings
            _pe = Tensor.Zeros(new long[] { maxLen, embeddingDim });

            for (int pos = 0; pos < maxLen; pos++)
            {
                for (int i = 0; i < embeddingDim; i++)
                {
                    double angle = pos / Math.Pow(10000, 2.0 * (i / 2) / embeddingDim);
                    _pe[pos, i] = i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
                }
            }

            RegisterBuffer("pe", _pe);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // input: [batch, seq_len, embedding_dim]
            var seqLen = (int)input.Shape[1];
            var pe = _pe.Slice(0, 0, seqLen);
            return input.Add(pe);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"PositionalEmbedding(max_len={_maxLen}, embedding_dim={_embeddingDim})";
    }

    #endregion

    #region Activation Functions as Modules

    /// <summary>
    /// ReLU activation module
    /// </summary>
    public class ReLU : Module
    {
        private readonly bool _inplace;

        /// <summary>Public API</summary>
        public ReLU(bool inplace = false)
        {
            _inplace = inplace;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input) => input.ReLU();

        /// <summary>Public API</summary>
        public override string ToString() => $"ReLU(inplace={_inplace})";
    }

    /// <summary>
    /// Leaky ReLU activation module
    /// </summary>
    public class LeakyReLU : Module
    {
        private readonly double _negativeSlope;

        /// <summary>Public API</summary>
        public LeakyReLU(double negativeSlope = 0.01)
        {
            _negativeSlope = negativeSlope;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input) => input.LeakyReLU(_negativeSlope);

        /// <summary>Public API</summary>
        public override string ToString() => $"LeakyReLU(negative_slope={_negativeSlope})";
    }

    /// <summary>
    /// PReLU - Parametric ReLU
    /// </summary>
    public class PReLU : Module
    {
        private Tensor _weight;

        /// <summary>Public API</summary>
        public PReLU(int numParameters = 1, double init = 0.25)
        {
            _weight = Tensor.Full(new long[] { numParameters }, init);
            RegisterParameter("weight", _weight);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var positive = input.ReLU();
            var negative = input.Apply(x => Math.Min(x, 0));

            if (_weight.NumElements == 1)
                return positive.Add(negative.Mul(_weight[0]));
            else
                return positive.Add(negative.Mul(_weight));
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"PReLU(num_parameters={_weight.NumElements})";
    }

    /// <summary>
    /// ELU activation module
    /// </summary>
    public class ELU : Module
    {
        private readonly double _alpha;

        /// <summary>Public API</summary>
        public ELU(double alpha = 1.0)
        {
            _alpha = alpha;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x => x > 0 ? x : _alpha * (Math.Exp(x) - 1));
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"ELU(alpha={_alpha})";
    }

    /// <summary>
    /// SELU activation module
    /// </summary>
    public class SELU : Module
    {
        private const double Alpha = 1.6732632423543772;
        private const double Scale = 1.0507009873554804;

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x => Scale * (x > 0 ? x : Alpha * (Math.Exp(x) - 1)));
        }

        /// <summary>Public API</summary>
        public override string ToString() => "SELU()";
    }

    /// <summary>
    /// GELU activation module (Gaussian Error Linear Unit)
    /// </summary>
    public class GELU : Module
    {
        private readonly bool _approximate;

        /// <summary>Public API</summary>
        public GELU(bool approximate = false)
        {
            _approximate = approximate;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            if (_approximate)
            {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                double c = Math.Sqrt(2 / Math.PI);
                return input.Apply(x => 0.5 * x * (1 + Math.Tanh(c * (x + 0.044715 * x * x * x))));
            }
            else
            {
                // Exact GELU: x * Phi(x) where Phi is the CDF of standard normal
                return input.Apply(x => x * 0.5 * (1 + Erf(x / Math.Sqrt(2))));
            }
        }

        // Error function approximation
        private static double Erf(double x)
        {
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;

            int sign = x < 0 ? -1 : 1;
            x = Math.Abs(x);

            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

            return sign * y;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"GELU(approximate={_approximate})";
    }

    /// <summary>
    /// Sigmoid activation module
    /// </summary>
    public class Sigmoid : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input) => input.Sigmoid();

        /// <summary>Public API</summary>
        public override string ToString() => "Sigmoid()";
    }

    /// <summary>
    /// Tanh activation module
    /// </summary>
    public class Tanh : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input) => input.Tanh();

        /// <summary>Public API</summary>
        public override string ToString() => "Tanh()";
    }

    /// <summary>
    /// Softmax activation module
    /// </summary>
    public class Softmax : Module
    {
        private readonly int _dim;

        /// <summary>Public API</summary>
        public Softmax(int dim = -1)
        {
            _dim = dim;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input) => input.Softmax(_dim);

        /// <summary>Public API</summary>
        public override string ToString() => $"Softmax(dim={_dim})";
    }

    /// <summary>
    /// LogSoftmax activation module
    /// </summary>
    public class LogSoftmax : Module
    {
        private readonly int _dim;

        /// <summary>Public API</summary>
        public LogSoftmax(int dim = -1)
        {
            _dim = dim;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input) => input.LogSoftmax(_dim);

        /// <summary>Public API</summary>
        public override string ToString() => $"LogSoftmax(dim={_dim})";
    }

    /// <summary>
    /// Softplus activation: log(1 + exp(x))
    /// </summary>
    public class Softplus : Module
    {
        private readonly double _beta;
        private readonly double _threshold;

        /// <summary>Public API</summary>
        public Softplus(double beta = 1, double threshold = 20)
        {
            _beta = beta;
            _threshold = threshold;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x =>
            {
                if (_beta * x > _threshold)
                    return x; // Linear for large values to avoid overflow
                return Math.Log(1 + Math.Exp(_beta * x)) / _beta;
            });
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Softplus(beta={_beta}, threshold={_threshold})";
    }

    /// <summary>
    /// Softsign activation: x / (1 + |x|)
    /// </summary>
    public class Softsign : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x => x / (1 + Math.Abs(x)));
        }

        /// <summary>Public API</summary>
        public override string ToString() => "Softsign()";
    }

    /// <summary>
    /// Swish activation: x * sigmoid(x)
    /// </summary>
    public class Swish : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Mul(input.Sigmoid());
        }

        /// <summary>Public API</summary>
        public override string ToString() => "Swish()";
    }

    /// <summary>
    /// SiLU activation (same as Swish): x * sigmoid(x)
    /// </summary>
    public class SiLU : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Mul(input.Sigmoid());
        }

        /// <summary>Public API</summary>
        public override string ToString() => "SiLU()";
    }

    /// <summary>
    /// Mish activation: x * tanh(softplus(x))
    /// </summary>
    public class Mish : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x => x * Math.Tanh(Math.Log(1 + Math.Exp(x))));
        }

        /// <summary>Public API</summary>
        public override string ToString() => "Mish()";
    }

    /// <summary>
    /// Hardtanh activation
    /// </summary>
    public class Hardtanh : Module
    {
        private readonly double _minVal;
        private readonly double _maxVal;

        /// <summary>Public API</summary>
        public Hardtanh(double minVal = -1, double maxVal = 1)
        {
            _minVal = minVal;
            _maxVal = maxVal;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Clamp(_minVal, _maxVal);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Hardtanh(min_val={_minVal}, max_val={_maxVal})";
    }

    /// <summary>
    /// Hardswish activation
    /// </summary>
    public class Hardswish : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x =>
            {
                if (x <= -3) return 0;
                if (x >= 3) return x;
                return x * (x + 3) / 6;
            });
        }

        /// <summary>Public API</summary>
        public override string ToString() => "Hardswish()";
    }

    /// <summary>
    /// Hardsigmoid activation
    /// </summary>
    public class Hardsigmoid : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x =>
            {
                if (x <= -3) return 0;
                if (x >= 3) return 1;
                return (x + 3) / 6;
            });
        }

        /// <summary>Public API</summary>
        public override string ToString() => "Hardsigmoid()";
    }

    /// <summary>
    /// RReLU - Randomized Leaky ReLU
    /// </summary>
    public class RReLU : Module
    {
        private readonly double _lower;
        private readonly double _upper;
        private static readonly Random _random = new Random();

        /// <summary>Public API</summary>
        public RReLU(double lower = 0.125, double upper = 0.333)
        {
            _lower = lower;
            _upper = upper;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            if (Training)
            {
                return input.Apply(x =>
                {
                    if (x >= 0) return x;
                    double a = _lower + _random.NextDouble() * (_upper - _lower);
                    return a * x;
                });
            }
            else
            {
                double a = (_lower + _upper) / 2;
                return input.Apply(x => x >= 0 ? x : a * x);
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"RReLU(lower={_lower}, upper={_upper})";
    }

    /// <summary>
    /// CELU activation
    /// </summary>
    public class CELU : Module
    {
        private readonly double _alpha;

        /// <summary>Public API</summary>
        public CELU(double alpha = 1.0)
        {
            _alpha = alpha;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return input.Apply(x => Math.Max(0, x) + Math.Min(0, _alpha * (Math.Exp(x / _alpha) - 1)));
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"CELU(alpha={_alpha})";
    }

    /// <summary>
    /// GLU activation (Gated Linear Unit)
    /// </summary>
    public class GLU : Module
    {
        private readonly int _dim;

        /// <summary>Public API</summary>
        public GLU(int dim = -1)
        {
            _dim = dim;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // Split input in half along dim, apply sigmoid to second half
            var dim = _dim < 0 ? input.NDim + _dim : _dim;
            var size = (int)(input.Shape[dim] / 2);

            var a = input.Slice(dim, 0, size);
            var b = input.Slice(dim, size, 2 * size);

            return a.Mul(b.Sigmoid());
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"GLU(dim={_dim})";
    }

    #endregion

    #region Shape Manipulation Layers

    /// <summary>
    /// Flatten layer
    /// </summary>
    public class Flatten : Module
    {
        private readonly int _startDim;
        private readonly int _endDim;

        /// <summary>Public API</summary>
        public Flatten(int startDim = 1, int endDim = -1)
        {
            _startDim = startDim;
            _endDim = endDim;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var startDim = _startDim < 0 ? input.NDim + _startDim : _startDim;
            var endDim = _endDim < 0 ? input.NDim + _endDim : _endDim;

            // Calculate new shape
            var newShape = new long[startDim + 1 + (input.NDim - endDim - 1)];

            for (int i = 0; i < startDim; i++)
                newShape[i] = input.Shape[i];

            long flatSize = 1;
            for (int i = startDim; i <= endDim; i++)
                flatSize *= input.Shape[i];
            newShape[startDim] = flatSize;

            for (int i = endDim + 1; i < input.NDim; i++)
                newShape[startDim + 1 + (i - endDim - 1)] = input.Shape[i];

            return input.Reshape(newShape);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Flatten(start_dim={_startDim}, end_dim={_endDim})";
    }

    /// <summary>
    /// Unflatten layer
    /// </summary>
    public class Unflatten : Module
    {
        private readonly int _dim;
        private readonly long[] _unflattenedSize;

        /// <summary>Public API</summary>
        public Unflatten(int dim, long[] unflattenedSize)
        {
            _dim = dim;
            _unflattenedSize = unflattenedSize;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var dim = _dim < 0 ? input.NDim + _dim : _dim;

            var newShape = new long[input.NDim - 1 + _unflattenedSize.Length];

            for (int i = 0; i < dim; i++)
                newShape[i] = input.Shape[i];

            for (int i = 0; i < _unflattenedSize.Length; i++)
                newShape[dim + i] = _unflattenedSize[i];

            for (int i = dim + 1; i < input.NDim; i++)
                newShape[dim + _unflattenedSize.Length + (i - dim - 1)] = input.Shape[i];

            return input.Reshape(newShape);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Unflatten(dim={_dim}, unflattened_size=[{string.Join(", ", _unflattenedSize)}])";
    }

    /// <summary>
    /// Identity layer (passthrough)
    /// </summary>
    public class Identity : Module
    {
        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input) => input;

        /// <summary>Public API</summary>
        public override string ToString() => "Identity()";
    }

    #endregion
}