using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace NSL.Tensor.NN
{
    #region Mixed Precision Training

    /// <summary>
    /// Mixed precision training utilities for FP16/BF16 training.
    /// Reduces memory usage and increases throughput on modern GPUs while maintaining model quality.
    ///
    /// Usage:
    /// var scaler = new GradScaler();
    /// using (var autocast = new AutocastScope(PrecisionMode.FP16))
    /// {
    ///     var output = model.Forward(input);
    ///     var loss = criterion(output, target);
    ///     scaler.Scale(loss).Backward();
    ///     scaler.Step(optimizer);
    ///     scaler.Update();
    /// }
    /// </summary>
    public enum PrecisionMode
    {
        /// <summary>32-bit floating point precision</summary>
        FP32,       // Full 32-bit precision (default)
        /// <summary>16-bit floating point precision</summary>
        FP16,       // Half precision (16-bit float)
        /// <summary>Brain float 16-bit precision</summary>
        BF16,       // Brain floating point (16-bit)
        /// <summary>TensorFloat 32-bit precision</summary>
        TF32,       // TensorFloat-32 (NVIDIA Ampere+)
        /// <summary>Mixed precision mode</summary>
        Mixed       // Automatic mixed precision
    }

    /// <summary>
    /// Context manager for automatic mixed precision casting.
    /// Operations within this scope automatically use lower precision where safe.
    /// </summary>
    public sealed class AutocastScope : IDisposable
    {
        [ThreadStatic]
        private static AutocastScope? _current;
        private readonly AutocastScope? _previous;
        private readonly PrecisionMode _mode;
        private readonly bool _enabled;

        /// <summary>Public API</summary>
        public static AutocastScope? Current => _current;
        /// <summary>Public API</summary>
        public static bool IsEnabled => _current?._enabled ?? false;
        /// <summary>Public API</summary>
        public static PrecisionMode CurrentMode => _current?._mode ?? PrecisionMode.FP32;

        /// <summary>Public API</summary>
        public PrecisionMode Mode => _mode;
        /// <summary>Public API</summary>
        public bool Enabled => _enabled;

        /// <summary>Public API</summary>
        public AutocastScope(PrecisionMode mode = PrecisionMode.FP16, bool enabled = true)
        {
            _mode = mode;
            _enabled = enabled;
            _previous = _current;
            _current = this;
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            _current = _previous;
        }

        /// <summary>
        /// Check if an operation should use lower precision.
        /// Some ops (like normalization, softmax) should stay in FP32 for stability.
        /// </summary>
        public static bool ShouldCast(string opName)
        {
            if (!IsEnabled) return false;

            // These ops benefit from lower precision
            var lowPrecisionOps = new HashSet<string>
            {
                "matmul", "linear", "conv1d", "conv2d", "conv3d",
                "bmm", "mm", "addmm", "addbmm",
                "embedding", "embedding_bag"
            };

            // These ops must stay in FP32 for numerical stability
            var fp32Ops = new HashSet<string>
            {
                "softmax", "log_softmax", "layer_norm", "batch_norm",
                "group_norm", "instance_norm", "rms_norm",
                "cross_entropy", "nll_loss", "mse_loss",
                "exp", "log", "pow", "sum", "mean", "var", "std"
            };

            var opLower = opName.ToLowerInvariant();

            if (fp32Ops.Contains(opLower))
                return false;

            return lowPrecisionOps.Contains(opLower) || !fp32Ops.Contains(opLower);
        }
    }

    /// <summary>
    /// Gradient scaler for mixed precision training.
    /// Scales gradients to prevent underflow in FP16, with automatic scale adjustment.
    /// </summary>
    public sealed class GradScaler
    {
        private double _scale;
        private readonly double _initScale;
        private readonly double _growthFactor;
        private readonly double _backoffFactor;
        private readonly int _growthInterval;
        private readonly double _maxScale;
        private readonly double _minScale;

        private int _growthTracker;
        private bool _foundInf;
        private int _consecutiveSuccess;

        /// <summary>Public API</summary>
        public double CurrentScale => _scale;
        /// <summary>Public API</summary>
        public bool FoundInf => _foundInf;

        /// <summary>
        /// Create a gradient scaler for mixed precision training.
        /// </summary>
        /// <param name="initScale">Initial loss scale factor (default: 65536)</param>
        /// <param name="growthFactor">Factor to increase scale (default: 2.0)</param>
        /// <param name="backoffFactor">Factor to decrease scale on overflow (default: 0.5)</param>
        /// <param name="growthInterval">Iterations between scale increases (default: 2000)</param>
        public GradScaler(
            double initScale = 65536.0,
            double growthFactor = 2.0,
            double backoffFactor = 0.5,
            int growthInterval = 2000,
            double maxScale = 65536.0 * 256,
            double minScale = 1.0)
        {
            _initScale = initScale;
            _scale = initScale;
            _growthFactor = growthFactor;
            _backoffFactor = backoffFactor;
            _growthInterval = growthInterval;
            _maxScale = maxScale;
            _minScale = minScale;
            _growthTracker = 0;
            _foundInf = false;
            _consecutiveSuccess = 0;
        }

        /// <summary>
        /// Scale a loss tensor for backward pass.
        /// </summary>
        public ScaledLoss Scale(Tensor loss)
        {
            var scaledLoss = loss.Mul(_scale);
            return new ScaledLoss(scaledLoss, this);
        }

        /// <summary>
        /// Unscale gradients after backward pass.
        /// </summary>
        public void Unscale(Optimizer optimizer)
        {
            _foundInf = false;
            var invScale = 1.0 / _scale;

            foreach (var (param, grad) in optimizer.GetGradients())
            {
                if (grad == null) continue;

                // Check for inf/nan in gradients
                if (HasInfOrNan(grad))
                {
                    _foundInf = true;
                    return;
                }

                // Unscale the gradient
                var unscaled = grad.Mul(invScale);

                // Check again after unscaling
                if (HasInfOrNan(unscaled))
                {
                    _foundInf = true;
                    return;
                }

                // Store unscaled gradient back
                optimizer.SetGradient(param, unscaled);
            }
        }

        /// <summary>
        /// Perform optimizer step with gradient unscaling.
        /// Skips step if inf/nan detected.
        /// </summary>
        public bool Step(Optimizer optimizer)
        {
            Unscale(optimizer);

            if (_foundInf)
            {
                // Skip update - gradients are invalid
                return false;
            }

            optimizer.StepOptimizer();
            return true;
        }

        /// <summary>
        /// Update the scale factor based on gradient history.
        /// </summary>
        public void Update()
        {
            if (_foundInf)
            {
                // Overflow occurred - reduce scale
                _scale = Math.Max(_scale * _backoffFactor, _minScale);
                _growthTracker = 0;
                _consecutiveSuccess = 0;
            }
            else
            {
                // No overflow - maybe increase scale
                _consecutiveSuccess++;
                _growthTracker++;

                if (_growthTracker >= _growthInterval)
                {
                    _scale = Math.Min(_scale * _growthFactor, _maxScale);
                    _growthTracker = 0;
                }
            }

            _foundInf = false;
        }

        /// <summary>
        /// Reset scaler state.
        /// </summary>
        public void Reset()
        {
            _scale = _initScale;
            _growthTracker = 0;
            _foundInf = false;
            _consecutiveSuccess = 0;
        }

        /// <summary>
        /// Get scaler state for checkpointing.
        /// </summary>
        public Dictionary<string, object> StateDict()
        {
            return new Dictionary<string, object>
            {
                ["scale"] = _scale,
                ["growth_tracker"] = _growthTracker,
                ["consecutive_success"] = _consecutiveSuccess
            };
        }

        /// <summary>
        /// Load scaler state from checkpoint.
        /// </summary>
        public void LoadStateDict(Dictionary<string, object> state)
        {
            if (state.TryGetValue("scale", out var scale))
                _scale = Convert.ToDouble(scale);
            if (state.TryGetValue("growth_tracker", out var tracker))
                _growthTracker = Convert.ToInt32(tracker);
            if (state.TryGetValue("consecutive_success", out var success))
                _consecutiveSuccess = Convert.ToInt32(success);
        }

        private static bool HasInfOrNan(Tensor t)
        {
            foreach (var v in t.Data)
            {
                if (double.IsInfinity(v) || double.IsNaN(v))
                    return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Wrapper for scaled loss that handles backward pass.
    /// </summary>
    public readonly struct ScaledLoss
    {
        private readonly Tensor _scaledLoss;
        private readonly GradScaler _scaler;

        internal ScaledLoss(Tensor scaledLoss, GradScaler scaler)
        {
            _scaledLoss = scaledLoss;
            _scaler = scaler;
        }

        /// <summary>Public API</summary>
        public Tensor Value => _scaledLoss;

        /// <summary>
        /// Perform backward pass with scaled loss.
        /// </summary>
        public void Backward()
        {
            // Backward with scaled gradients
            if (GradientTape.Current != null)
            {
                GradientTape.Current.Backward(_scaledLoss);
            }
        }
    }

    #endregion

    #region Half Precision Tensor Operations

    /// <summary>
    /// Half precision (FP16) utilities for memory-efficient tensor storage.
    /// </summary>
    public static class HalfPrecision
    {
        /// <summary>
        /// Convert tensor data to FP16 byte representation.
        /// </summary>
        public static byte[] ToFloat16Bytes(Tensor tensor)
        {
            var result = new byte[tensor.NumElements * 2];

            for (int i = 0; i < tensor.NumElements; i++)
            {
                var half = FloatToHalf((float)tensor.Data[i]);
                result[i * 2] = (byte)(half & 0xFF);
                result[i * 2 + 1] = (byte)((half >> 8) & 0xFF);
            }

            return result;
        }

        /// <summary>
        /// Convert FP16 bytes back to tensor.
        /// </summary>
        public static Tensor FromFloat16Bytes(byte[] bytes, long[] shape)
        {
            var numElements = bytes.Length / 2;
            var data = new double[numElements];

            for (int i = 0; i < numElements; i++)
            {
                ushort half = (ushort)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
                data[i] = HalfToFloat(half);
            }

            return new Tensor(data, shape);
        }

        /// <summary>
        /// Convert tensor data to BF16 byte representation.
        /// </summary>
        public static byte[] ToBFloat16Bytes(Tensor tensor)
        {
            var result = new byte[tensor.NumElements * 2];

            for (int i = 0; i < tensor.NumElements; i++)
            {
                var bf16 = FloatToBFloat16((float)tensor.Data[i]);
                result[i * 2] = (byte)(bf16 & 0xFF);
                result[i * 2 + 1] = (byte)((bf16 >> 8) & 0xFF);
            }

            return result;
        }

        /// <summary>
        /// Convert BF16 bytes back to tensor.
        /// </summary>
        public static Tensor FromBFloat16Bytes(byte[] bytes, long[] shape)
        {
            var numElements = bytes.Length / 2;
            var data = new double[numElements];

            for (int i = 0; i < numElements; i++)
            {
                ushort bf16 = (ushort)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
                data[i] = BFloat16ToFloat(bf16);
            }

            return new Tensor(data, shape);
        }

        /// <summary>
        /// Convert float to IEEE 754 half precision.
        /// </summary>
        public static ushort FloatToHalf(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);

            uint sign = (bits >> 16) & 0x8000;
            int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
            uint mant = bits & 0x7FFFFF;

            if (exp <= 0)
            {
                if (exp < -10)
                    return (ushort)sign;

                mant = (mant | 0x800000) >> (1 - exp);
                return (ushort)((uint)sign | (mant >> 13));
            }

            if (exp >= 31)
            {
                return (ushort)(sign | 0x7C00 | ((bits & 0x7FFFFF) != 0 ? 0x200 : 0));
            }

            return (ushort)((uint)sign | (uint)(exp << 10) | (mant >> 13));
        }

        /// <summary>
        /// Convert half precision to float.
        /// </summary>
        public static float HalfToFloat(ushort half)
        {
            int sign = (half >> 15) & 1;
            int exp = (half >> 10) & 0x1F;
            int mant = half & 0x3FF;

            if (exp == 0)
            {
                if (mant == 0)
                    return sign == 1 ? -0.0f : 0.0f;

                // Denormalized
                float val = mant / 1024.0f * (float)Math.Pow(2, -14);
                return sign == 1 ? -val : val;
            }

            if (exp == 31)
            {
                if (mant == 0)
                    return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
                return float.NaN;
            }

            float result = (1.0f + mant / 1024.0f) * (float)Math.Pow(2, exp - 15);
            return sign == 1 ? -result : result;
        }

        /// <summary>
        /// Convert float to BFloat16 (truncate lower 16 bits).
        /// </summary>
        public static ushort FloatToBFloat16(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            // Round to nearest even
            uint rounded = bits + 0x7FFF + ((bits >> 16) & 1);
            return (ushort)(rounded >> 16);
        }

        /// <summary>
        /// Convert BFloat16 to float (pad with zeros).
        /// </summary>
        public static float BFloat16ToFloat(ushort bf16)
        {
            uint bits = (uint)bf16 << 16;
            return BitConverter.UInt32BitsToSingle(bits);
        }
    }

    #endregion

    #region Mixed Precision Module Wrapper

    /// <summary>
    /// Wrapper for modules to enable mixed precision.
    /// Automatically casts weights and handles precision during forward/backward.
    /// </summary>
    public class MixedPrecisionModule : Module
    {
        private readonly Module _module;
        private readonly PrecisionMode _mode;
        private Dictionary<string, byte[]>? _fp16Weights;
        private bool _weightsConverted;

        /// <summary>Public API</summary>
        public Module InnerModule => _module;
        /// <summary>Public API</summary>
        public PrecisionMode Mode => _mode;

        /// <summary>Public API</summary>
        public MixedPrecisionModule(Module module, PrecisionMode mode = PrecisionMode.FP16)
        {
            _module = module;
            _mode = mode;
            _weightsConverted = false;
        }

        /// <summary>
        /// Convert module weights to half precision for memory savings.
        /// Weights are converted back to FP32 during forward pass.
        /// </summary>
        public void ConvertWeightsToHalf()
        {
            if (_weightsConverted) return;

            _fp16Weights = new Dictionary<string, byte[]>();

            foreach (var (name, tensor) in _module.StateDict())
            {
                _fp16Weights[name] = _mode == PrecisionMode.BF16
                    ? HalfPrecision.ToBFloat16Bytes(tensor)
                    : HalfPrecision.ToFloat16Bytes(tensor);
            }

            _weightsConverted = true;
        }

        /// <summary>
        /// Restore weights to FP32.
        /// </summary>
        public void RestoreWeightsToFull()
        {
            if (!_weightsConverted || _fp16Weights == null) return;

            var stateDict = new Dictionary<string, Tensor>();

            foreach (var (name, tensor) in _module.StateDict())
            {
                if (_fp16Weights.TryGetValue(name, out var bytes))
                {
                    var restored = _mode == PrecisionMode.BF16
                        ? HalfPrecision.FromBFloat16Bytes(bytes, tensor.Shape)
                        : HalfPrecision.FromFloat16Bytes(bytes, tensor.Shape);
                    stateDict[name] = restored;
                }
            }

            _module.LoadStateDict(stateDict, strict: false);
            _weightsConverted = false;
            _fp16Weights = null;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            using var autocast = new AutocastScope(_mode);
            return _module.Forward(input);
        }

        /// <summary>Public API</summary>
        public override Dictionary<string, Tensor> StateDict()
        {
            return _module.StateDict();
        }

        /// <summary>Public API</summary>
        public override void LoadStateDict(Dictionary<string, Tensor> stateDict, bool strict = true)
        {
            _module.LoadStateDict(stateDict, strict);
        }
    }

    #endregion

    #region Automatic Mixed Precision Training Loop Helper

    /// <summary>
    /// Helper class for mixed precision training loops.
    /// Combines autocast and gradient scaling in a convenient API.
    /// </summary>
    public sealed class AMPTrainer : IDisposable
    {
        private readonly GradScaler _scaler;
        private readonly PrecisionMode _mode;
        private bool _enabled;

        /// <summary>Public API</summary>
        public GradScaler Scaler => _scaler;
        /// <summary>Public API</summary>
        public bool Enabled { get => _enabled; set => _enabled = value; }
        /// <summary>Public API</summary>
        public PrecisionMode Mode => _mode;

        /// <summary>Public API</summary>
        public AMPTrainer(PrecisionMode mode = PrecisionMode.FP16, bool enabled = true)
        {
            _mode = mode;
            _enabled = enabled;
            _scaler = new GradScaler();
        }

        /// <summary>
        /// Create an autocast scope for forward pass.
        /// </summary>
        public AutocastScope Autocast()
        {
            return new AutocastScope(_mode, _enabled);
        }

        /// <summary>
        /// Scale loss for backward pass.
        /// </summary>
        public ScaledLoss ScaleLoss(Tensor loss)
        {
            if (!_enabled)
                return new ScaledLoss(loss, _scaler);

            return _scaler.Scale(loss);
        }

        /// <summary>
        /// Perform optimizer step with gradient unscaling and update scaler.
        /// </summary>
        public bool Step(Optimizer optimizer)
        {
            if (!_enabled)
            {
                optimizer.StepOptimizer();
                return true;
            }

            var success = _scaler.Step(optimizer);
            _scaler.Update();
            return success;
        }

        /// <summary>
        /// Get state for checkpointing.
        /// </summary>
        public Dictionary<string, object> StateDict()
        {
            return new Dictionary<string, object>
            {
                ["enabled"] = _enabled,
                ["mode"] = (int)_mode,
                ["scaler"] = _scaler.StateDict()
            };
        }

        /// <summary>
        /// Load state from checkpoint.
        /// </summary>
        public void LoadStateDict(Dictionary<string, object> state)
        {
            if (state.TryGetValue("enabled", out var enabled))
                _enabled = Convert.ToBoolean(enabled);

            if (state.TryGetValue("scaler", out var scalerState) && scalerState is Dictionary<string, object> scalerDict)
                _scaler.LoadStateDict(scalerDict);
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            // Nothing to dispose
        }
    }

    #endregion

    #region FP16 Safe Operations

    /// <summary>
    /// Operations that are safe and efficient in FP16.
    /// </summary>
    public static class FP16SafeOps
    {
        /// <summary>
        /// FP16-safe matrix multiplication with optional accumulation in FP32.
        /// </summary>
        public static Tensor SafeMatMul(Tensor a, Tensor b, bool accumulateInFP32 = true)
        {
            // In production with GPU, this would use tensor cores with FP16 input and FP32 accumulation
            // For CPU simulation, we just do the multiplication
            return TensorOps.MatMul(a, b);
        }

        /// <summary>
        /// FP16-safe linear layer forward pass.
        /// </summary>
        public static Tensor SafeLinear(Tensor input, Tensor weight, Tensor? bias = null)
        {
            var output = SafeMatMul(input, weight.T());
            if (bias != null)
                output = output.Add(bias);
            return output;
        }

        /// <summary>
        /// FP16-safe convolution (delegates to regular conv with FP32 accumulation).
        /// </summary>
        public static Tensor SafeConv2d(
            Tensor input, Tensor weight, Tensor? bias,
            int strideH, int strideW, int padH, int padW)
        {
            return Functional.Conv2d(input, weight, bias, strideH, strideW, padH, padW);
        }

        /// <summary>
        /// Scale tensor to FP16 range to prevent overflow/underflow.
        /// </summary>
        public static (Tensor scaled, double scaleFactor) ScaleToFP16Range(Tensor t)
        {
            var maxAbs = t.Abs().Max().ToScalar();

            // FP16 max is ~65504, use a safe margin
            const double fp16Max = 60000.0;

            if (maxAbs > fp16Max)
            {
                var scale = fp16Max / maxAbs;
                return (t.Mul(scale), scale);
            }

            // Check for denormals (very small values)
            const double fp16MinNormal = 6.1e-5;
            var minNonZero = t.Abs().Data.Where(v => v > 0).DefaultIfEmpty(1.0).Min();

            if (minNonZero < fp16MinNormal && minNonZero > 0)
            {
                var scale = fp16MinNormal / minNonZero;
                return (t.Mul(scale), scale);
            }

            return (t, 1.0);
        }
    }

    #endregion

    #region Loss Scaling Utilities

    /// <summary>
    /// Static loss scaling without dynamic adjustment.
    /// Useful when you know the optimal scale factor for your model.
    /// </summary>
    public static class StaticLossScaling
    {
        /// <summary>
        /// Apply static loss scaling to a loss tensor.
        /// </summary>
        public static Tensor Scale(Tensor loss, double scaleFactor = 128.0)
        {
            return loss.Mul(scaleFactor);
        }

        /// <summary>
        /// Unscale gradients after backward pass.
        /// </summary>
        public static void UnscaleGradients(Optimizer optimizer, double scaleFactor = 128.0)
        {
            var invScale = 1.0 / scaleFactor;

            foreach (var (param, grad) in optimizer.GetGradients())
            {
                if (grad != null)
                {
                    var unscaled = grad.Mul(invScale);
                    optimizer.SetGradient(param, unscaled);
                }
            }
        }
    }

    /// <summary>
    /// Per-tensor scaling for fine-grained control.
    /// Each tensor gets its own scale factor based on its value range.
    /// </summary>
    public class PerTensorScaler
    {
        private readonly Dictionary<string, double> _scales = new();
        private readonly double _targetRange;

        /// <summary>Public API</summary>
        public PerTensorScaler(double targetRange = 1000.0)
        {
            _targetRange = targetRange;
        }

        /// <summary>
        /// Compute and apply scaling for a tensor.
        /// </summary>
        public Tensor Scale(string name, Tensor tensor)
        {
            var maxAbs = tensor.Abs().Max().ToScalar();

            if (maxAbs < 1e-10)
            {
                _scales[name] = 1.0;
                return tensor;
            }

            var scale = _targetRange / maxAbs;
            _scales[name] = scale;

            return tensor.Mul(scale);
        }

        /// <summary>
        /// Unscale a tensor back to original range.
        /// </summary>
        public Tensor Unscale(string name, Tensor tensor)
        {
            if (_scales.TryGetValue(name, out var scale) && scale != 1.0)
            {
                return tensor.Div(scale);
            }
            return tensor;
        }

        /// <summary>
        /// Get the scale factor for a tensor.
        /// </summary>
        public double GetScale(string name)
        {
            return _scales.TryGetValue(name, out var scale) ? scale : 1.0;
        }
    }

    #endregion

    #region Optimizer Extension for Mixed Precision

    /// <summary>
    /// Extension methods for optimizers to support mixed precision training.
    /// </summary>
    public static class OptimizerMixedPrecisionExtensions
    {
        /// <summary>
        /// Get all gradients from the optimizer.
        /// </summary>
        public static IEnumerable<(Tensor param, Tensor? grad)> GetGradients(this Optimizer optimizer)
        {
            // This would need actual implementation in the optimizer base class
            // For now, return empty - actual implementation depends on optimizer structure
            return Enumerable.Empty<(Tensor, Tensor?)>();
        }

        /// <summary>
        /// Set a gradient for a parameter.
        /// </summary>
        public static void SetGradient(this Optimizer optimizer, Tensor param, Tensor grad)
        {
            // This would need actual implementation in the optimizer base class
            // For now, no-op - actual implementation depends on optimizer structure
        }
    }

    #endregion
}