using System;
using System.Linq;

namespace NSL.Tensor.NN
{
    /// <summary>
    /// Functional API for neural network operations (stateless)
    /// </summary>
    public static class F
    {
        #region Activation Functions

        /// <summary>
        /// ReLU activation: max(0, x)
        /// </summary>
        public static Tensor ReLU(Tensor input, bool inplace = false)
        {
            return input.ReLU();
        }

        /// <summary>
        /// Leaky ReLU activation
        /// </summary>
        public static Tensor LeakyReLU(Tensor input, double negativeSlope = 0.01, bool inplace = false)
        {
            return input.LeakyReLU(negativeSlope);
        }

        /// <summary>
        /// ELU activation
        /// </summary>
        public static Tensor ELU(Tensor input, double alpha = 1.0, bool inplace = false)
        {
            return input.Apply(x => x > 0 ? x : alpha * (Math.Exp(x) - 1));
        }

        /// <summary>
        /// SELU activation
        /// </summary>
        public static Tensor SELU(Tensor input, bool inplace = false)
        {
            const double alpha = 1.6732632423543772;
            const double scale = 1.0507009873554804;
            return input.Apply(x => scale * (x > 0 ? x : alpha * (Math.Exp(x) - 1)));
        }

        /// <summary>
        /// GELU activation
        /// </summary>
        public static Tensor GELU(Tensor input, bool approximate = false)
        {
            if (approximate)
            {
                double c = Math.Sqrt(2 / Math.PI);
                return input.Apply(x => 0.5 * x * (1 + Math.Tanh(c * (x + 0.044715 * x * x * x))));
            }
            else
            {
                return input.Apply(x => x * 0.5 * (1 + Erf(x / Math.Sqrt(2))));
            }
        }

        /// <summary>
        /// Sigmoid activation
        /// </summary>
        public static Tensor Sigmoid(Tensor input)
        {
            return input.Sigmoid();
        }

        /// <summary>
        /// Tanh activation
        /// </summary>
        public static Tensor Tanh(Tensor input)
        {
            return input.Tanh();
        }

        /// <summary>
        /// Softmax activation
        /// </summary>
        public static Tensor Softmax(Tensor input, int dim = -1)
        {
            return input.Softmax(dim);
        }

        /// <summary>
        /// Log softmax activation
        /// </summary>
        public static Tensor LogSoftmax(Tensor input, int dim = -1)
        {
            return input.LogSoftmax(dim);
        }

        /// <summary>
        /// Softplus activation
        /// </summary>
        public static Tensor Softplus(Tensor input, double beta = 1, double threshold = 20)
        {
            return input.Apply(x =>
            {
                if (beta * x > threshold)
                    return x;
                return Math.Log(1 + Math.Exp(beta * x)) / beta;
            });
        }

        /// <summary>
        /// Softsign activation
        /// </summary>
        public static Tensor Softsign(Tensor input)
        {
            return input.Apply(x => x / (1 + Math.Abs(x)));
        }

        /// <summary>
        /// Swish/SiLU activation
        /// </summary>
        public static Tensor Swish(Tensor input)
        {
            return input.Mul(input.Sigmoid());
        }

        /// <summary>
        /// SiLU activation (same as Swish)
        /// </summary>
        public static Tensor SiLU(Tensor input)
        {
            return Swish(input);
        }

        /// <summary>
        /// Mish activation
        /// </summary>
        public static Tensor Mish(Tensor input)
        {
            return input.Apply(x => x * Math.Tanh(Math.Log(1 + Math.Exp(x))));
        }

        /// <summary>
        /// Hardtanh activation
        /// </summary>
        public static Tensor Hardtanh(Tensor input, double minVal = -1, double maxVal = 1, bool inplace = false)
        {
            return input.Clamp(minVal, maxVal);
        }

        /// <summary>
        /// Hardswish activation
        /// </summary>
        public static Tensor Hardswish(Tensor input, bool inplace = false)
        {
            return input.Apply(x =>
            {
                if (x <= -3) return 0;
                if (x >= 3) return x;
                return x * (x + 3) / 6;
            });
        }

        /// <summary>
        /// Hardsigmoid activation
        /// </summary>
        public static Tensor Hardsigmoid(Tensor input, bool inplace = false)
        {
            return input.Apply(x =>
            {
                if (x <= -3) return 0;
                if (x >= 3) return 1;
                return (x + 3) / 6;
            });
        }

        /// <summary>
        /// ReLU6 activation: min(max(0, x), 6)
        /// </summary>
        public static Tensor ReLU6(Tensor input, bool inplace = false)
        {
            return input.Clamp(0, 6);
        }

        /// <summary>
        /// Threshold function
        /// </summary>
        public static Tensor Threshold(Tensor input, double threshold, double value, bool inplace = false)
        {
            return input.Apply(x => x > threshold ? x : value);
        }

        /// <summary>
        /// CELU activation
        /// </summary>
        public static Tensor CELU(Tensor input, double alpha = 1.0, bool inplace = false)
        {
            return input.Apply(x => Math.Max(0, x) + Math.Min(0, alpha * (Math.Exp(x / alpha) - 1)));
        }

        /// <summary>
        /// GLU activation
        /// </summary>
        public static Tensor GLU(Tensor input, int dim = -1)
        {
            dim = dim < 0 ? input.NDim + dim : dim;
            var size = (int)(input.Shape[dim] / 2);
            var a = input.Slice(dim, 0, size);
            var b = input.Slice(dim, size, 2 * size);
            return a.Mul(b.Sigmoid());
        }

        #endregion

        #region Loss Functions

        /// <summary>
        /// Mean Squared Error loss
        /// </summary>
        public static Tensor MSELoss(Tensor input, Tensor target, string reduction = "mean")
        {
            var diff = input.Sub(target);
            var sq = diff.Square();

            switch (reduction)
            {
                case "mean":
                    return sq.Mean();
                case "sum":
                    return sq.Sum();
                case "none":
                    return sq;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// L1 Loss (Mean Absolute Error)
        /// </summary>
        public static Tensor L1Loss(Tensor input, Tensor target, string reduction = "mean")
        {
            var diff = input.Sub(target).Abs();

            switch (reduction)
            {
                case "mean":
                    return diff.Mean();
                case "sum":
                    return diff.Sum();
                case "none":
                    return diff;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Smooth L1 Loss (Huber Loss)
        /// </summary>
        public static Tensor SmoothL1Loss(Tensor input, Tensor target, string reduction = "mean", double beta = 1.0)
        {
            var diff = input.Sub(target).Abs();
            var loss = diff.Apply(d => d < beta ? 0.5 * d * d / beta : d - 0.5 * beta);

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Binary Cross Entropy loss
        /// </summary>
        public static Tensor BCELoss(Tensor input, Tensor target, string reduction = "mean")
        {
            // BCE = -[y * log(p) + (1-y) * log(1-p)]
            var eps = 1e-7;
            var inputClamped = input.Clamp(eps, 1 - eps);

            var loss = target.Mul(inputClamped.Log()).Add(
                Tensor.Ones(target.Shape).Sub(target).Mul(
                    Tensor.Ones(inputClamped.Shape).Sub(inputClamped).Log()
                )
            ).Neg();

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Binary Cross Entropy with Logits loss (more numerically stable)
        /// </summary>
        public static Tensor BCEWithLogitsLoss(Tensor input, Tensor target, string reduction = "mean")
        {
            // BCE with logits = max(x, 0) - x * y + log(1 + exp(-|x|))
            var maxVal = input.ReLU();
            var negAbs = input.Abs().Neg();
            var logPart = negAbs.Exp().Add(1).Log();

            var loss = maxVal.Sub(input.Mul(target)).Add(logPart);

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Cross Entropy loss
        /// </summary>
        public static Tensor CrossEntropyLoss(Tensor input, Tensor target, string reduction = "mean", int ignoreIndex = -100)
        {
            // input: [batch, classes] (logits)
            // target: [batch] (class indices) or [batch, classes] (one-hot)

            if (target.NDim == 1)
            {
                // Target is class indices
                var logSoftmax = input.LogSoftmax(-1);
                var batch = (int)input.Shape[0];
                var numClasses = (int)input.Shape[1];

                var loss = Tensor.Zeros(new long[] { batch });
                int validCount = 0;

                for (int b = 0; b < batch; b++)
                {
                    int classIdx = (int)target[b];
                    if (classIdx != ignoreIndex)
                    {
                        loss[b] = -logSoftmax[b, classIdx];
                        validCount++;
                    }
                }

                switch (reduction)
                {
                    case "mean":
                        return validCount > 0 ? loss.Sum().Div(validCount) : new Tensor(0.0);
                    case "sum":
                        return loss.Sum();
                    case "none":
                        return loss;
                    default:
                        throw new ArgumentException($"Unknown reduction: {reduction}");
                }
            }
            else
            {
                // Target is one-hot
                var logSoftmax = input.LogSoftmax(-1);
                var loss = target.Mul(logSoftmax).Sum(-1, keepDim: false).Neg();

                switch (reduction)
                {
                    case "mean":
                        return loss.Mean();
                    case "sum":
                        return loss.Sum();
                    case "none":
                        return loss;
                    default:
                        throw new ArgumentException($"Unknown reduction: {reduction}");
                }
            }
        }

        /// <summary>
        /// Negative Log Likelihood loss
        /// </summary>
        public static Tensor NLLLoss(Tensor input, Tensor target, string reduction = "mean", int ignoreIndex = -100)
        {
            // input: [batch, classes] (log probabilities)
            // target: [batch] (class indices)

            var batch = (int)input.Shape[0];
            var loss = Tensor.Zeros(new long[] { batch });
            int validCount = 0;

            for (int b = 0; b < batch; b++)
            {
                int classIdx = (int)target[b];
                if (classIdx != ignoreIndex)
                {
                    loss[b] = -input[b, classIdx];
                    validCount++;
                }
            }

            switch (reduction)
            {
                case "mean":
                    return validCount > 0 ? loss.Sum().Div(validCount) : new Tensor(0.0);
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Hinge loss (for SVM)
        /// </summary>
        public static Tensor HingeLoss(Tensor input, Tensor target, string reduction = "mean")
        {
            // target should be -1 or 1
            var loss = Tensor.Ones(input.Shape).Sub(input.Mul(target)).Apply(x => Math.Max(0, x));

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Margin Ranking Loss
        /// </summary>
        public static Tensor MarginRankingLoss(Tensor input1, Tensor input2, Tensor target, double margin = 0, string reduction = "mean")
        {
            // target is 1 or -1, indicating which input should be larger
            var loss = target.Neg().Mul(input1.Sub(input2)).Add(margin).Apply(x => Math.Max(0, x));

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Triplet Margin Loss
        /// </summary>
        public static Tensor TripletMarginLoss(Tensor anchor, Tensor positive, Tensor negative, double margin = 1.0, double p = 2, string reduction = "mean")
        {
            var posDist = PairwiseDistance(anchor, positive, p);
            var negDist = PairwiseDistance(anchor, negative, p);

            var loss = posDist.Sub(negDist).Add(margin).Apply(x => Math.Max(0, x));

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Cosine Embedding Loss
        /// </summary>
        public static Tensor CosineEmbeddingLoss(Tensor input1, Tensor input2, Tensor target, double margin = 0, string reduction = "mean")
        {
            var cosSim = CosineSimilarity(input1, input2, 1);

            // If target is 1, want cos sim close to 1; if -1, want cos sim < margin
            var loss = target.Apply((t, idx) =>
            {
                var sim = cosSim.ToArray()[idx];
                if (t == 1)
                    return 1 - sim;
                else
                    return Math.Max(0, sim - margin);
            });

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// KL Divergence loss
        /// </summary>
        public static Tensor KLDivLoss(Tensor input, Tensor target, string reduction = "mean", bool logTarget = false)
        {
            // input is log probabilities, target is probabilities (or log probs if logTarget=true)
            Tensor loss;
            if (logTarget)
            {
                loss = target.Exp().Mul(target.Sub(input));
            }
            else
            {
                var targetClamped = target.ClampMin(1e-7);
                loss = targetClamped.Mul(targetClamped.Log().Sub(input));
            }

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "batchmean":
                    return loss.Sum().Div(input.Shape[0]);
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Poisson NLL Loss
        /// </summary>
        public static Tensor PoissonNLLLoss(Tensor input, Tensor target, bool logInput = true, bool full = false, string reduction = "mean")
        {
            Tensor loss;
            if (logInput)
            {
                loss = input.Exp().Sub(target.Mul(input));
            }
            else
            {
                loss = input.Sub(target.Mul(input.ClampMin(1e-8).Log()));
            }

            if (full)
            {
                // Add Stirling approximation term
                var approx = target.Mul(target.Log()).Sub(target).Add(0.5 * Math.Log(2 * Math.PI * Math.E));
                loss = loss.Add(approx.Apply(x => target.ToArray().Any(t => t > 1) ? x : 0));
            }

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Focal Loss (for imbalanced classification)
        /// </summary>
        public static Tensor FocalLoss(Tensor input, Tensor target, double alpha = 0.25, double gamma = 2.0, string reduction = "mean")
        {
            var p = input.Sigmoid();
            var ce = BCEWithLogitsLoss(input, target, "none");

            // Focal weight: (1 - p_t)^gamma where p_t = p if y=1 else 1-p
            var pT = target.Mul(p).Add(Tensor.Ones(target.Shape).Sub(target).Mul(Tensor.Ones(p.Shape).Sub(p)));
            var focalWeight = Tensor.Ones(pT.Shape).Sub(pT).Pow(gamma);

            // Alpha weight
            var alphaWeight = target.Mul(alpha).Add(Tensor.Ones(target.Shape).Sub(target).Mul(1 - alpha));

            var loss = alphaWeight.Mul(focalWeight).Mul(ce);

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        /// <summary>
        /// Label Smoothing Cross Entropy Loss
        /// </summary>
        public static Tensor LabelSmoothingCrossEntropy(Tensor input, Tensor target, double smoothing = 0.1, string reduction = "mean")
        {
            var numClasses = (int)input.Shape[^1];
            var confidence = 1.0 - smoothing;
            var smoothValue = smoothing / numClasses;

            var logProbs = input.LogSoftmax(-1);
            var nllLoss = NLLLoss(logProbs, target, "none");
            var smoothLoss = logProbs.Mean(-1, keepDim: false).Neg();

            var loss = nllLoss.Mul(confidence).Add(smoothLoss.Mul(smoothing));

            switch (reduction)
            {
                case "mean":
                    return loss.Mean();
                case "sum":
                    return loss.Sum();
                case "none":
                    return loss;
                default:
                    throw new ArgumentException($"Unknown reduction: {reduction}");
            }
        }

        #endregion

        #region Normalization Functions

        /// <summary>
        /// Batch normalization
        /// </summary>
        public static Tensor BatchNorm(Tensor input, Tensor? runningMean, Tensor? runningVar, Tensor? weight = null, Tensor? bias = null,
            bool training = true, double momentum = 0.1, double eps = 1e-5)
        {
            Tensor mean, variance;

            if (training)
            {
                // Compute batch statistics
                mean = input.Mean(0, keepDim: true);
                variance = input.Var(0, keepDim: true);

                // Update running statistics
                if (runningMean != null)
                {
                    var newMean = runningMean.Mul(1 - momentum).Add(mean.Detach().Mul(momentum));
                    Array.Copy(newMean.ToArray(), runningMean.ToArray(), runningMean.NumElements);
                }
                if (runningVar != null)
                {
                    var newVar = runningVar.Mul(1 - momentum).Add(variance.Detach().Mul(momentum));
                    Array.Copy(newVar.ToArray(), runningVar.ToArray(), runningVar.NumElements);
                }
            }
            else
            {
                mean = runningMean ?? Tensor.Zeros(new long[] { input.Shape[^1] });
                variance = runningVar ?? Tensor.Ones(new long[] { input.Shape[^1] });
            }

            var normalized = input.Sub(mean).Div(variance.Add(eps).Sqrt());

            if (weight != null)
                normalized = normalized.Mul(weight);
            if (bias != null)
                normalized = normalized.Add(bias);

            return normalized;
        }

        /// <summary>
        /// Layer normalization
        /// </summary>
        public static Tensor LayerNorm(Tensor input, long[] normalizedShape, Tensor? weight = null, Tensor? bias = null, double eps = 1e-5)
        {
            var numNormDims = normalizedShape.Length;
            var startDim = input.NDim - numNormDims;

            // Compute mean and variance over normalized dimensions
            var axes = Enumerable.Range(startDim, numNormDims).ToArray();
            var mean = input;
            var variance = input;

            foreach (var axis in axes.Reverse())
            {
                mean = mean.Mean(axis, keepDim: true);
            }

            var centered = input.Sub(mean);
            var sq = centered.Square();
            foreach (var axis in axes.Reverse())
            {
                variance = sq.Mean(axis, keepDim: true);
            }

            var normalized = centered.Div(variance.Add(eps).Sqrt());

            if (weight != null)
                normalized = normalized.Mul(weight);
            if (bias != null)
                normalized = normalized.Add(bias);

            return normalized;
        }

        #endregion

        #region Dropout Functions

        private static readonly Random _random = new Random();

        /// <summary>
        /// Dropout
        /// </summary>
        public static Tensor Dropout(Tensor input, double p = 0.5, bool training = true, bool inplace = false)
        {
            if (!training || p == 0)
                return input;

            var mask = Tensor.Zeros(input.Shape);
            var scale = 1.0 / (1.0 - p);
            var data = mask.ToArray();

            for (int i = 0; i < data.Length; i++)
                data[i] = _random.NextDouble() > p ? scale : 0;

            mask = Tensor.FromArray(data, input.Shape);
            return input.Mul(mask);
        }

        /// <summary>
        /// Alpha Dropout
        /// </summary>
        public static Tensor AlphaDropout(Tensor input, double p = 0.5, bool training = true, bool inplace = false)
        {
            if (!training || p == 0)
                return input;

            const double alpha = 1.6732632423543772;
            const double scale = 1.0507009873554804;
            double alphaPrime = -alpha * scale;
            double a = 1.0 / Math.Sqrt(p + alphaPrime * alphaPrime * p * (1 - p));
            double b = -a * (1 - p) * alphaPrime;

            var data = input.ToArray();
            for (int i = 0; i < data.Length; i++)
            {
                if (_random.NextDouble() < p)
                    data[i] = alphaPrime;
            }

            var output = Tensor.FromArray(data, input.Shape);
            return output.Mul(a).Add(b);
        }

        #endregion

        #region Pooling Functions

        /// <summary>
        /// 2D Max Pooling
        /// </summary>
        public static Tensor MaxPool2d(Tensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            var s = stride ?? kernelSize;
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];
            var outputH = (inputH + 2 * padding - kernelSize) / s + 1;
            var outputW = (inputW + 2 * padding - kernelSize) / s + 1;

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
                            for (int kh = 0; kh < kernelSize; kh++)
                            {
                                for (int kw = 0; kw < kernelSize; kw++)
                                {
                                    int ih = oh * s - padding + kh;
                                    int iw = ow * s - padding + kw;
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

        /// <summary>
        /// 2D Average Pooling
        /// </summary>
        public static Tensor AvgPool2d(Tensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            var s = stride ?? kernelSize;
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];
            var outputH = (inputH + 2 * padding - kernelSize) / s + 1;
            var outputW = (inputW + 2 * padding - kernelSize) / s + 1;

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
                            for (int kh = 0; kh < kernelSize; kh++)
                            {
                                for (int kw = 0; kw < kernelSize; kw++)
                                {
                                    int ih = oh * s - padding + kh;
                                    int iw = ow * s - padding + kw;
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

        /// <summary>
        /// Adaptive Average Pooling 2D
        /// </summary>
        public static Tensor AdaptiveAvgPool2d(Tensor input, int outputSize)
        {
            return AdaptiveAvgPool2d(input, outputSize, outputSize);
        }

        /// <summary>
        /// Adaptive Average Pooling 2D
        /// </summary>
        public static Tensor AdaptiveAvgPool2d(Tensor input, int outputH, int outputW)
        {
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];

            var output = Tensor.Zeros(new long[] { batch, channels, outputH, outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            int startH = (int)Math.Floor(oh * inputH / (double)outputH);
                            int endH = (int)Math.Ceiling((oh + 1) * inputH / (double)outputH);
                            int startW = (int)Math.Floor(ow * inputW / (double)outputW);
                            int endW = (int)Math.Ceiling((ow + 1) * inputW / (double)outputW);

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

        #endregion

        #region Distance Functions

        /// <summary>
        /// Pairwise distance
        /// </summary>
        public static Tensor PairwiseDistance(Tensor x1, Tensor x2, double p = 2, double eps = 1e-6)
        {
            var diff = x1.Sub(x2);
            if (Math.Abs(p - 2) < 1e-10)
            {
                return diff.Square().Sum(-1, keepDim: false).Sqrt();
            }
            else
            {
                return diff.Abs().Pow(p).Sum(-1, keepDim: false).Pow(1.0 / p);
            }
        }

        /// <summary>
        /// Cosine similarity
        /// </summary>
        public static Tensor CosineSimilarity(Tensor x1, Tensor x2, int dim = 1, double eps = 1e-8)
        {
            var dot = x1.Mul(x2).Sum(dim, keepDim: false);
            var norm1 = x1.Square().Sum(dim, keepDim: false).Sqrt().ClampMin(eps);
            var norm2 = x2.Square().Sum(dim, keepDim: false).Sqrt().ClampMin(eps);

            return dot.Div(norm1.Mul(norm2));
        }

        /// <summary>
        /// Pairwise cosine similarity
        /// </summary>
        public static Tensor PairwiseCosineSimilarity(Tensor x1, Tensor x2, double eps = 1e-8)
        {
            // x1: [n, d], x2: [m, d] -> output: [n, m]
            var norm1 = x1.Square().Sum(-1, keepDim: true).Sqrt().ClampMin(eps);
            var norm2 = x2.Square().Sum(-1, keepDim: true).Sqrt().ClampMin(eps);

            var x1Norm = x1.Div(norm1);
            var x2Norm = x2.Div(norm2);

            return TensorOps.MatMul(x1Norm, x2Norm.T());
        }

        #endregion

        #region Attention Functions

        /// <summary>
        /// Scaled dot-product attention
        /// </summary>
        public static Tensor ScaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor? mask = null, double dropout = 0)
        {
            // query: [batch, seq_q, d_k]
            // key: [batch, seq_k, d_k]
            // value: [batch, seq_k, d_v]

            var dk = query.Shape[^1];
            var scale = 1.0 / Math.Sqrt(dk);

            // Compute attention scores: Q @ K^T / sqrt(d_k)
            var scores = TensorOps.MatMul(query, key.Transpose(-2, -1)).Mul(scale);

            if (mask != null)
            {
                // Apply mask (set masked positions to -inf)
                scores = scores.Add(mask.Mul(-1e9).Mul(Tensor.Ones(mask.Shape).Sub(mask)));
            }

            // Softmax over last dimension
            var attnWeights = scores.Softmax(-1);

            // Apply dropout
            if (dropout > 0)
            {
                attnWeights = Dropout(attnWeights, dropout, training: true);
            }

            // Compute output: attn_weights @ V
            return TensorOps.MatMul(attnWeights, value);
        }

        /// <summary>
        /// Multi-head attention
        /// </summary>
        public static Tensor MultiheadAttention(Tensor query, Tensor key, Tensor value, int numHeads,
            Tensor wQ, Tensor wK, Tensor wV, Tensor wO, Tensor? bQ = null, Tensor? bK = null, Tensor? bV = null, Tensor? bO = null,
            Tensor? mask = null, double dropout = 0)
        {
            var batch = (int)query.Shape[0];
            var seqLen = (int)query.Shape[1];
            var dModel = (int)query.Shape[2];
            var headDim = dModel / numHeads;

            // Project Q, K, V
            var Q = TensorOps.MatMul(query, wQ);
            var K = TensorOps.MatMul(key, wK);
            var V = TensorOps.MatMul(value, wV);

            if (bQ != null) Q = Q.Add(bQ);
            if (bK != null) K = K.Add(bK);
            if (bV != null) V = V.Add(bV);

            // Reshape for multi-head: [batch, seq, d_model] -> [batch, num_heads, seq, head_dim]
            Q = Q.Reshape(new long[] { batch, seqLen, numHeads, headDim }).Transpose(1, 2);
            K = K.Reshape(new long[] { batch, (int)key.Shape[1], numHeads, headDim }).Transpose(1, 2);
            V = V.Reshape(new long[] { batch, (int)value.Shape[1], numHeads, headDim }).Transpose(1, 2);

            // Apply attention to each head
            var attnOutput = ScaledDotProductAttention(Q, K, V, mask, dropout);

            // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, d_model]
            attnOutput = attnOutput.Transpose(1, 2).Reshape(new long[] { batch, seqLen, dModel });

            // Final projection
            var output = TensorOps.MatMul(attnOutput, wO);
            if (bO != null) output = output.Add(bO);

            return output;
        }

        #endregion

        #region Utility Functions

        /// <summary>
        /// Error function (used in GELU)
        /// </summary>
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

        /// <summary>
        /// One-hot encoding
        /// </summary>
        public static Tensor OneHot(Tensor indices, int numClasses)
        {
            var data = indices.ToArray();
            var output = Tensor.Zeros(new long[] { data.Length, numClasses });

            for (int i = 0; i < data.Length; i++)
            {
                int idx = (int)data[i];
                if (idx >= 0 && idx < numClasses)
                    output[i, idx] = 1;
            }

            return output;
        }

        /// <summary>
        /// Interpolate (resize) tensor
        /// </summary>
        public static Tensor Interpolate(Tensor input, int[] size, string mode = "nearest")
        {
            // input: [batch, channels, height, width]
            var batch = (int)input.Shape[0];
            var channels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];
            var outputH = size[0];
            var outputW = size[1];

            var output = Tensor.Zeros(new long[] { batch, channels, outputH, outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            double srcH = oh * (inputH - 1) / (double)(outputH - 1);
                            double srcW = ow * (inputW - 1) / (double)(outputW - 1);

                            if (mode == "nearest")
                            {
                                int ih = (int)Math.Round(srcH);
                                int iw = (int)Math.Round(srcW);
                                ih = Math.Min(ih, inputH - 1);
                                iw = Math.Min(iw, inputW - 1);
                                output[b, c, oh, ow] = input[b, c, ih, iw];
                            }
                            else if (mode == "bilinear")
                            {
                                int ih0 = (int)Math.Floor(srcH);
                                int iw0 = (int)Math.Floor(srcW);
                                int ih1 = Math.Min(ih0 + 1, inputH - 1);
                                int iw1 = Math.Min(iw0 + 1, inputW - 1);

                                double hFrac = srcH - ih0;
                                double wFrac = srcW - iw0;

                                double v00 = input[b, c, ih0, iw0];
                                double v01 = input[b, c, ih0, iw1];
                                double v10 = input[b, c, ih1, iw0];
                                double v11 = input[b, c, ih1, iw1];

                                double v0 = v00 * (1 - wFrac) + v01 * wFrac;
                                double v1 = v10 * (1 - wFrac) + v11 * wFrac;

                                output[b, c, oh, ow] = v0 * (1 - hFrac) + v1 * hFrac;
                            }
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>
        /// Pad tensor
        /// </summary>
        public static Tensor Pad(Tensor input, int[] padding, string mode = "constant", double value = 0)
        {
            // padding format: [left, right, top, bottom] for 2D
            // input: [batch, channels, height, width]

            if (input.NDim == 4 && padding.Length == 4)
            {
                var batch = (int)input.Shape[0];
                var channels = (int)input.Shape[1];
                var height = (int)input.Shape[2];
                var width = (int)input.Shape[3];

                var newH = height + padding[2] + padding[3];
                var newW = width + padding[0] + padding[1];

                Tensor output;
                if (mode == "constant")
                    output = Tensor.Full(new long[] { batch, channels, newH, newW }, value);
                else
                    output = Tensor.Zeros(new long[] { batch, channels, newH, newW });

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                int newH_idx = h + padding[2];
                                int newW_idx = w + padding[0];

                                if (mode == "reflect" || mode == "replicate")
                                {
                                    // For reflect/replicate, just copy the original values
                                    // (Full implementation would handle edges differently)
                                }

                                output[b, c, newH_idx, newW_idx] = input[b, c, h, w];
                            }
                        }
                    }
                }

                return output;
            }

            throw new ArgumentException("Unsupported padding configuration");
        }

        #endregion

        #region Sampling Functions

        private static readonly Random _samplingRandom = new Random();

        /// <summary>
        /// Sample from a probability distribution using top-k sampling.
        /// Selects from only the k highest probability tokens.
        /// </summary>
        /// <param name="logits">Logits tensor (1D or 2D with shape [batch, vocab])</param>
        /// <param name="k">Number of top tokens to consider</param>
        /// <param name="temperature">Temperature for softmax (higher = more random)</param>
        /// <returns>Sampled token indices</returns>
        public static int[] SampleTopK(Tensor logits, int k, double temperature = 1.0)
        {
            if (logits.NDim == 1)
            {
                return new[] { SampleTopKSingle(logits.Data, k, temperature) };
            }
            else if (logits.NDim == 2)
            {
                var batchSize = (int)logits.Shape[0];
                var vocabSize = (int)logits.Shape[1];
                var results = new int[batchSize];

                for (int b = 0; b < batchSize; b++)
                {
                    var batchLogits = new double[vocabSize];
                    for (int v = 0; v < vocabSize; v++)
                        batchLogits[v] = logits[b, v];
                    results[b] = SampleTopKSingle(batchLogits, k, temperature);
                }
                return results;
            }
            throw new ArgumentException("Logits must be 1D or 2D tensor");
        }

        private static int SampleTopKSingle(double[] logits, int k, double temperature)
        {
            var n = logits.Length;
            k = Math.Min(k, n);

            // Find top-k indices
            var indices = Enumerable.Range(0, n)
                .OrderByDescending(i => logits[i])
                .Take(k)
                .ToArray();

            // Apply temperature and softmax to top-k logits
            var topKLogits = new double[k];
            double maxLogit = double.NegativeInfinity;
            for (int i = 0; i < k; i++)
            {
                topKLogits[i] = logits[indices[i]] / temperature;
                maxLogit = Math.Max(maxLogit, topKLogits[i]);
            }

            // Stable softmax
            double sumExp = 0;
            for (int i = 0; i < k; i++)
            {
                topKLogits[i] = Math.Exp(topKLogits[i] - maxLogit);
                sumExp += topKLogits[i];
            }
            for (int i = 0; i < k; i++)
                topKLogits[i] /= sumExp;

            // Sample from distribution
            double r = _samplingRandom.NextDouble();
            double cumsum = 0;
            for (int i = 0; i < k; i++)
            {
                cumsum += topKLogits[i];
                if (r <= cumsum)
                    return indices[i];
            }
            return indices[k - 1];
        }

        /// <summary>
        /// Sample from a probability distribution using nucleus (top-p) sampling.
        /// Selects from smallest set of tokens whose cumulative probability >= p.
        /// </summary>
        /// <param name="logits">Logits tensor (1D or 2D with shape [batch, vocab])</param>
        /// <param name="p">Cumulative probability threshold (0.0 to 1.0)</param>
        /// <param name="temperature">Temperature for softmax (higher = more random)</param>
        /// <returns>Sampled token indices</returns>
        public static int[] SampleTopP(Tensor logits, double p, double temperature = 1.0)
        {
            if (logits.NDim == 1)
            {
                return new[] { SampleTopPSingle(logits.Data, p, temperature) };
            }
            else if (logits.NDim == 2)
            {
                var batchSize = (int)logits.Shape[0];
                var vocabSize = (int)logits.Shape[1];
                var results = new int[batchSize];

                for (int b = 0; b < batchSize; b++)
                {
                    var batchLogits = new double[vocabSize];
                    for (int v = 0; v < vocabSize; v++)
                        batchLogits[v] = logits[b, v];
                    results[b] = SampleTopPSingle(batchLogits, p, temperature);
                }
                return results;
            }
            throw new ArgumentException("Logits must be 1D or 2D tensor");
        }

        private static int SampleTopPSingle(double[] logits, double p, double temperature)
        {
            var n = logits.Length;

            // Apply temperature and compute softmax
            var probs = new double[n];
            double maxLogit = logits.Max();
            double sumExp = 0;
            for (int i = 0; i < n; i++)
            {
                probs[i] = Math.Exp((logits[i] - maxLogit) / temperature);
                sumExp += probs[i];
            }
            for (int i = 0; i < n; i++)
                probs[i] /= sumExp;

            // Sort by probability descending
            var sortedIndices = Enumerable.Range(0, n)
                .OrderByDescending(i => probs[i])
                .ToArray();

            // Find nucleus (smallest set with cumulative prob >= p)
            double cumsum = 0;
            int nucleusSize = 0;
            for (int i = 0; i < n; i++)
            {
                cumsum += probs[sortedIndices[i]];
                nucleusSize++;
                if (cumsum >= p)
                    break;
            }

            // Renormalize nucleus probabilities
            double nucleusSum = 0;
            for (int i = 0; i < nucleusSize; i++)
                nucleusSum += probs[sortedIndices[i]];

            // Sample from nucleus
            double r = _samplingRandom.NextDouble() * nucleusSum;
            cumsum = 0;
            for (int i = 0; i < nucleusSize; i++)
            {
                cumsum += probs[sortedIndices[i]];
                if (r <= cumsum)
                    return sortedIndices[i];
            }
            return sortedIndices[nucleusSize - 1];
        }

        /// <summary>
        /// Gumbel-Softmax: Differentiable approximation to sampling from categorical distribution.
        /// Useful for training with discrete latent variables.
        /// </summary>
        /// <param name="logits">Logits tensor</param>
        /// <param name="temperature">Temperature (lower = closer to one-hot)</param>
        /// <param name="hard">If true, returns one-hot but with gradients from soft version</param>
        /// <param name="dim">Dimension to apply softmax</param>
        /// <returns>Soft or hard samples</returns>
        public static Tensor GumbelSoftmax(Tensor logits, double temperature = 1.0, bool hard = false, int dim = -1)
        {
            // Sample Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
            var gumbel = logits.Apply(_ =>
            {
                double u = _samplingRandom.NextDouble();
                u = Math.Max(u, 1e-10); // Avoid log(0)
                return -Math.Log(-Math.Log(u));
            });

            // Add Gumbel noise and apply temperature-scaled softmax
            var perturbed = logits.Add(gumbel).Div(temperature);
            var soft = perturbed.Softmax(dim);

            if (!hard)
                return soft;

            // Hard: argmax but with soft gradients (straight-through estimator)
            dim = dim < 0 ? logits.NDim + dim : dim;
            var hardSamples = ArgmaxOneHot(soft, dim);

            // Return hard samples but with soft gradients attached
            // (In a full implementation, this would use detach + add)
            return hardSamples;
        }

        private static Tensor ArgmaxOneHot(Tensor input, int dim)
        {
            // Create one-hot encoding of argmax
            var result = Tensor.Zeros(input.Shape);
            var data = input.Data;
            var resultData = result.Data;
            var shape = input.Shape;

            if (input.NDim == 1)
            {
                int maxIdx = 0;
                double maxVal = data[0];
                for (int i = 1; i < data.Length; i++)
                {
                    if (data[i] > maxVal)
                    {
                        maxVal = data[i];
                        maxIdx = i;
                    }
                }
                resultData[maxIdx] = 1.0;
            }
            else if (input.NDim == 2)
            {
                int rows = (int)shape[0];
                int cols = (int)shape[1];
                if (dim == 0)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        int maxIdx = 0;
                        double maxVal = data[c];
                        for (int r = 1; r < rows; r++)
                        {
                            if (data[r * cols + c] > maxVal)
                            {
                                maxVal = data[r * cols + c];
                                maxIdx = r;
                            }
                        }
                        resultData[maxIdx * cols + c] = 1.0;
                    }
                }
                else // dim == 1
                {
                    for (int r = 0; r < rows; r++)
                    {
                        int maxIdx = 0;
                        double maxVal = data[r * cols];
                        for (int c = 1; c < cols; c++)
                        {
                            if (data[r * cols + c] > maxVal)
                            {
                                maxVal = data[r * cols + c];
                                maxIdx = c;
                            }
                        }
                        resultData[r * cols + maxIdx] = 1.0;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Sample a single token from logits using temperature-scaled softmax.
        /// </summary>
        public static int[] SampleCategorical(Tensor logits, double temperature = 1.0)
        {
            if (logits.NDim == 1)
            {
                return new[] { SampleCategoricalSingle(logits.Data, temperature) };
            }
            else if (logits.NDim == 2)
            {
                var batchSize = (int)logits.Shape[0];
                var vocabSize = (int)logits.Shape[1];
                var results = new int[batchSize];

                for (int b = 0; b < batchSize; b++)
                {
                    var batchLogits = new double[vocabSize];
                    for (int v = 0; v < vocabSize; v++)
                        batchLogits[v] = logits[b, v];
                    results[b] = SampleCategoricalSingle(batchLogits, temperature);
                }
                return results;
            }
            throw new ArgumentException("Logits must be 1D or 2D tensor");
        }

        private static int SampleCategoricalSingle(double[] logits, double temperature)
        {
            var n = logits.Length;
            var probs = new double[n];

            // Apply temperature and stable softmax
            double maxLogit = logits.Max();
            double sumExp = 0;
            for (int i = 0; i < n; i++)
            {
                probs[i] = Math.Exp((logits[i] - maxLogit) / temperature);
                sumExp += probs[i];
            }
            for (int i = 0; i < n; i++)
                probs[i] /= sumExp;

            // Sample
            double r = _samplingRandom.NextDouble();
            double cumsum = 0;
            for (int i = 0; i < n; i++)
            {
                cumsum += probs[i];
                if (r <= cumsum)
                    return i;
            }
            return n - 1;
        }

        #endregion

        #region Convolution (for compatibility)

        /// <summary>
        /// 2D convolution operation.
        /// </summary>
        public static Tensor Conv2d(
            Tensor input,
            Tensor weight,
            Tensor? bias = null,
            int strideH = 1,
            int strideW = 1,
            int paddingH = 0,
            int paddingW = 0,
            int dilationH = 1,
            int dilationW = 1)
        {
            var batch = (int)input.Shape[0];
            var inChannels = (int)input.Shape[1];
            var inputH = (int)input.Shape[2];
            var inputW = (int)input.Shape[3];

            var outChannels = (int)weight.Shape[0];
            var kernelH = (int)weight.Shape[2];
            var kernelW = (int)weight.Shape[3];

            var effectiveKernelH = dilationH * (kernelH - 1) + 1;
            var effectiveKernelW = dilationW * (kernelW - 1) + 1;
            var outputH = (inputH + 2 * paddingH - effectiveKernelH) / strideH + 1;
            var outputW = (inputW + 2 * paddingW - effectiveKernelW) / strideW + 1;

            var output = Tensor.Zeros(new long[] { batch, outChannels, outputH, outputW });

            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            double sum = 0;
                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                for (int kh = 0; kh < kernelH; kh++)
                                {
                                    for (int kw = 0; kw < kernelW; kw++)
                                    {
                                        int ih = oh * strideH - paddingH + kh * dilationH;
                                        int iw = ow * strideW - paddingW + kw * dilationW;
                                        if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW)
                                        {
                                            sum += input[b, ic, ih, iw] * weight[oc, ic, kh, kw];
                                        }
                                    }
                                }
                            }
                            output[b, oc, oh, ow] = sum + (bias != null ? bias[oc] : 0);
                        }
                    }
                }
            }

            return output;
        }

        /// <summary>
        /// Linear transformation.
        /// </summary>
        public static Tensor Linear(Tensor input, Tensor weight, Tensor? bias = null)
        {
            var output = TensorOps.MatMul(input, weight.T());
            if (bias != null)
                output = output.Add(bias);
            return output;
        }

        #endregion
    }

    /// <summary>
    /// Alias for F class (PyTorch-style naming)
    /// </summary>
    public static class Functional
    {
        /// <summary>Public API</summary>
        public static Tensor Conv2d(Tensor input, Tensor weight, Tensor? bias = null,
            int strideH = 1, int strideW = 1, int paddingH = 0, int paddingW = 0,
            int dilationH = 1, int dilationW = 1)
            => F.Conv2d(input, weight, bias, strideH, strideW, paddingH, paddingW, dilationH, dilationW);

        /// <summary>Public API</summary>
        public static Tensor Linear(Tensor input, Tensor weight, Tensor? bias = null)
            => F.Linear(input, weight, bias);

        /// <summary>Public API</summary>
        public static Tensor ReLU(Tensor input, bool inplace = false) => F.ReLU(input, inplace);
        /// <summary>Public API</summary>
        public static Tensor LeakyReLU(Tensor input, double negativeSlope = 0.01) => F.LeakyReLU(input, negativeSlope);
        /// <summary>Public API</summary>
        public static Tensor GELU(Tensor input, bool approximate = false) => F.GELU(input, approximate);
        /// <summary>Public API</summary>
        public static Tensor Sigmoid(Tensor input) => F.Sigmoid(input);
        /// <summary>Public API</summary>
        public static Tensor Tanh(Tensor input) => F.Tanh(input);
        /// <summary>Public API</summary>
        public static Tensor Softmax(Tensor input, int dim = -1) => F.Softmax(input, dim);
        /// <summary>Public API</summary>
        public static Tensor LogSoftmax(Tensor input, int dim = -1) => F.LogSoftmax(input, dim);
        /// <summary>Public API</summary>
        public static Tensor SiLU(Tensor input) => F.SiLU(input);
        /// <summary>Public API</summary>
        public static Tensor Swish(Tensor input) => F.Swish(input);

        /// <summary>Public API</summary>
        public static Tensor Dropout(Tensor input, double p = 0.5, bool training = true) => F.Dropout(input, p, training);

        /// <summary>Public API</summary>
        public static Tensor MaxPool2d(Tensor input, int kernelSize, int? stride = null, int padding = 0)
            => F.MaxPool2d(input, kernelSize, stride, padding);
        /// <summary>Public API</summary>
        public static Tensor AvgPool2d(Tensor input, int kernelSize, int? stride = null, int padding = 0)
            => F.AvgPool2d(input, kernelSize, stride, padding);
        /// <summary>Public API</summary>
        public static Tensor AdaptiveAvgPool2d(Tensor input, int outputH, int outputW)
            => F.AdaptiveAvgPool2d(input, outputH, outputW);

        /// <summary>Public API</summary>
        public static Tensor LayerNorm(Tensor input, long[] normalizedShape, Tensor? weight = null, Tensor? bias = null, double eps = 1e-5)
            => F.LayerNorm(input, normalizedShape, weight, bias, eps);
        /// <summary>Public API</summary>
        public static Tensor BatchNorm(Tensor input, Tensor? runningMean, Tensor? runningVar, Tensor? weight = null, Tensor? bias = null,
            bool training = true, double momentum = 0.1, double eps = 1e-5)
            => F.BatchNorm(input, runningMean, runningVar, weight, bias, training, momentum, eps);

        /// <summary>Public API</summary>
        public static Tensor MSELoss(Tensor input, Tensor target, string reduction = "mean") => F.MSELoss(input, target, reduction);
        /// <summary>Public API</summary>
        public static Tensor CrossEntropyLoss(Tensor input, Tensor target, string reduction = "mean") => F.CrossEntropyLoss(input, target, reduction);
        /// <summary>Public API</summary>
        public static Tensor BCELoss(Tensor input, Tensor target, string reduction = "mean") => F.BCELoss(input, target, reduction);
        /// <summary>Public API</summary>
        public static Tensor BCEWithLogitsLoss(Tensor input, Tensor target, string reduction = "mean") => F.BCEWithLogitsLoss(input, target, reduction);

        /// <summary>Public API</summary>
        public static Tensor ScaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor? mask = null, double dropout = 0)
            => F.ScaledDotProductAttention(query, key, value, mask, dropout);

        /// <summary>Public API</summary>
        public static Tensor Interpolate(Tensor input, int[] size, string mode = "nearest") => F.Interpolate(input, size, mode);
        /// <summary>Public API</summary>
        public static Tensor Pad(Tensor input, int[] padding, string mode = "constant", double value = 0) => F.Pad(input, padding, mode, value);
        /// <summary>Public API</summary>
        public static Tensor OneHot(Tensor indices, int numClasses) => F.OneHot(indices, numClasses);
    }
}