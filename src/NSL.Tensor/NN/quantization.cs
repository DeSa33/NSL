using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace NSL.Tensor.NN
{
    #region Quantization Core

    /// <summary>
    /// Quantization types supported by NSL.
    /// </summary>
    public enum QuantizationType
    {
        Dynamic,
        Static,
        PerChannel,
        PerGroup
    }

    /// <summary>
    /// Quantization data types.
    /// </summary>
    public enum QuantDType
    {
        INT8,       // Signed 8-bit integer (-128 to 127)
        UINT8,      // Unsigned 8-bit integer (0 to 255)
        INT4,       // Signed 4-bit integer (-8 to 7)
        UINT4,      // Unsigned 4-bit integer (0 to 15)
        INT2,       // 2-bit quantization (4 levels)
        FP8_E4M3,   // 8-bit float with 4 exponent, 3 mantissa bits
        FP8_E5M2    // 8-bit float with 5 exponent, 2 mantissa bits
    }

    /// <summary>
    /// Quantization parameters for a tensor.
    /// </summary>
    public class QuantizationParams
    {
        /// <summary>Public API</summary>
        public QuantDType DType { get; set; } = QuantDType.INT8;
        /// <summary>Public API</summary>
        public QuantizationType Type { get; set; } = QuantizationType.Static;

        /// <summary>Scale factor(s) for quantization. Single value for per-tensor, array for per-channel.</summary>
        public double[] Scales { get; set; } = new[] { 1.0 };

        /// <summary>Zero point(s) for asymmetric quantization.</summary>
        public long[] ZeroPoints { get; set; } = new long[] { 0 };

        /// <summary>For per-group quantization, the group size.</summary>
        public int GroupSize { get; set; } = 128;

        /// <summary>Axis for per-channel quantization.</summary>
        public int ChannelAxis { get; set; } = 0;

        /// <summary>Whether to use symmetric quantization (zero point = 0).</summary>
        public bool Symmetric { get; set; } = true;

        /// <summary>Min value observed during calibration.</summary>
        public double ObservedMin { get; set; }

        /// <summary>Max value observed during calibration.</summary>
        public double ObservedMax { get; set; }
    }

    /// <summary>
    /// Quantized tensor representation.
    /// Stores integer data with quantization parameters for dequantization.
    /// </summary>
    public class QuantizedTensor
    {
        private readonly byte[] _data;       // For INT8/UINT8
        private readonly byte[]? _data4bit;  // For INT4/UINT4 (packed, 2 values per byte)
        private readonly long[] _shape;
        private readonly QuantizationParams _params;

        /// <summary>Public API</summary>
        public long[] Shape => _shape;
        /// <summary>Public API</summary>
        public QuantizationParams Params => _params;
        /// <summary>Public API</summary>
        public long NumElements => _shape.Aggregate(1L, (a, b) => a * b);
        /// <summary>Public API</summary>
        public QuantDType DType => _params.DType;

        /// <summary>
        /// Create a quantized tensor from float data.
        /// </summary>
        public QuantizedTensor(double[] data, long[] shape, QuantizationParams qparams)
        {
            _shape = shape;
            _params = qparams;

            switch (qparams.DType)
            {
                case QuantDType.INT8:
                case QuantDType.UINT8:
                    _data = Quantize8Bit(data, qparams);
                    break;

                case QuantDType.INT4:
                case QuantDType.UINT4:
                    _data = new byte[0];
                    _data4bit = Quantize4Bit(data, qparams);
                    break;

                default:
                    _data = Quantize8Bit(data, qparams);
                    break;
            }
        }

        /// <summary>
        /// Create a quantized tensor from raw bytes.
        /// </summary>
        public QuantizedTensor(byte[] rawData, long[] shape, QuantizationParams qparams)
        {
            _shape = shape;
            _params = qparams;

            if (qparams.DType == QuantDType.INT4 || qparams.DType == QuantDType.UINT4)
            {
                _data = new byte[0];
                _data4bit = rawData;
            }
            else
            {
                _data = rawData;
            }
        }

        /// <summary>
        /// Dequantize back to float tensor.
        /// </summary>
        public Tensor Dequantize()
        {
            double[] result;

            switch (_params.DType)
            {
                case QuantDType.INT8:
                    result = Dequantize8BitSigned();
                    break;

                case QuantDType.UINT8:
                    result = Dequantize8BitUnsigned();
                    break;

                case QuantDType.INT4:
                    result = Dequantize4BitSigned();
                    break;

                case QuantDType.UINT4:
                    result = Dequantize4BitUnsigned();
                    break;

                default:
                    result = Dequantize8BitSigned();
                    break;
            }

            return new Tensor(result, _shape);
        }

        /// <summary>
        /// Get raw quantized data.
        /// </summary>
        public byte[] GetRawData()
        {
            return _params.DType switch
            {
                QuantDType.INT4 or QuantDType.UINT4 => _data4bit ?? Array.Empty<byte>(),
                _ => _data
            };
        }

        /// <summary>
        /// Get memory usage in bytes.
        /// </summary>
        public long GetMemoryUsage()
        {
            return _params.DType switch
            {
                QuantDType.INT4 or QuantDType.UINT4 => (NumElements + 1) / 2,
                QuantDType.INT2 => (NumElements + 3) / 4,
                _ => NumElements
            };
        }

        #region Quantization Methods

        private byte[] Quantize8Bit(double[] data, QuantizationParams qparams)
        {
            var result = new byte[data.Length];
            var scale = qparams.Scales[0];
            var zeroPoint = qparams.ZeroPoints[0];

            bool signed = qparams.DType == QuantDType.INT8;
            int minVal = signed ? -128 : 0;
            int maxVal = signed ? 127 : 255;

            for (int i = 0; i < data.Length; i++)
            {
                var quantized = Math.Round(data[i] / scale) + zeroPoint;
                quantized = Math.Clamp(quantized, minVal, maxVal);
                result[i] = unchecked((byte)(int)quantized);
            }

            return result;
        }

        private byte[] Quantize4Bit(double[] data, QuantizationParams qparams)
        {
            // Pack 2 values per byte
            var packedLength = (data.Length + 1) / 2;
            var result = new byte[packedLength];
            var scale = qparams.Scales[0];
            var zeroPoint = qparams.ZeroPoints[0];

            bool signed = qparams.DType == QuantDType.INT4;
            int minVal = signed ? -8 : 0;
            int maxVal = signed ? 7 : 15;

            for (int i = 0; i < data.Length; i += 2)
            {
                var q1 = (int)Math.Clamp(Math.Round(data[i] / scale) + zeroPoint, minVal, maxVal);
                var q2 = i + 1 < data.Length
                    ? (int)Math.Clamp(Math.Round(data[i + 1] / scale) + zeroPoint, minVal, maxVal)
                    : 0;

                // Pack: lower 4 bits = first value, upper 4 bits = second value
                result[i / 2] = (byte)((q1 & 0x0F) | ((q2 & 0x0F) << 4));
            }

            return result;
        }

        private double[] Dequantize8BitSigned()
        {
            var result = new double[_data.Length];
            var scale = _params.Scales[0];
            var zeroPoint = _params.ZeroPoints[0];

            for (int i = 0; i < _data.Length; i++)
            {
                result[i] = ((sbyte)_data[i] - zeroPoint) * scale;
            }

            return result;
        }

        private double[] Dequantize8BitUnsigned()
        {
            var result = new double[_data.Length];
            var scale = _params.Scales[0];
            var zeroPoint = _params.ZeroPoints[0];

            for (int i = 0; i < _data.Length; i++)
            {
                result[i] = (_data[i] - zeroPoint) * scale;
            }

            return result;
        }

        private double[] Dequantize4BitSigned()
        {
            var numElements = NumElements;
            var result = new double[numElements];
            var scale = _params.Scales[0];
            var zeroPoint = _params.ZeroPoints[0];

            for (int i = 0; i < numElements; i++)
            {
                int byteIdx = i / 2;
                int nibble = (i % 2 == 0) ? (_data4bit![byteIdx] & 0x0F) : ((_data4bit![byteIdx] >> 4) & 0x0F);

                // Sign extend for INT4
                if (nibble >= 8) nibble -= 16;

                result[i] = (nibble - zeroPoint) * scale;
            }

            return result;
        }

        private double[] Dequantize4BitUnsigned()
        {
            var numElements = NumElements;
            var result = new double[numElements];
            var scale = _params.Scales[0];
            var zeroPoint = _params.ZeroPoints[0];

            for (int i = 0; i < numElements; i++)
            {
                int byteIdx = i / 2;
                int nibble = (i % 2 == 0) ? (_data4bit![byteIdx] & 0x0F) : ((_data4bit![byteIdx] >> 4) & 0x0F);
                result[i] = (nibble - zeroPoint) * scale;
            }

            return result;
        }

        #endregion
    }

    #endregion

    #region Calibration

    /// <summary>
    /// Calibration methods for determining quantization parameters.
    /// </summary>
    public enum CalibrationMethod
    {
        MinMax,
        Percentile,
        Entropy,
        MSE
    }

    /// <summary>
    /// Observer for collecting statistics during calibration.
    /// </summary>
    public class QuantizationObserver
    {
        private readonly CalibrationMethod _method;
        private readonly QuantDType _dtype;
        private readonly double _percentile;

        private double _minVal = double.MaxValue;
        private double _maxVal = double.MinValue;
        private readonly List<double> _samples = new();
        private int _numBatches;

        /// <summary>Public API</summary>
        public double ObservedMin => _minVal;
        /// <summary>Public API</summary>
        public double ObservedMax => _maxVal;
        /// <summary>Public API</summary>
        public int NumBatches => _numBatches;

        /// <summary>Public API</summary>
        public QuantizationObserver(
            CalibrationMethod method = CalibrationMethod.MinMax,
            QuantDType dtype = QuantDType.INT8,
            double percentile = 99.99)
        {
            _method = method;
            _dtype = dtype;
            _percentile = percentile;
        }

        /// <summary>
        /// Observe a tensor to collect statistics.
        /// </summary>
        public void Observe(Tensor tensor)
        {
            _numBatches++;

            foreach (var val in tensor.Data)
            {
                _minVal = Math.Min(_minVal, val);
                _maxVal = Math.Max(_maxVal, val);

                if (_method == CalibrationMethod.Percentile || _method == CalibrationMethod.Entropy)
                {
                    // Sample for percentile/entropy calculation (limit memory usage)
                    if (_samples.Count < 100000 || Random.Shared.NextDouble() < 0.01)
                    {
                        _samples.Add(val);
                    }
                }
            }
        }

        /// <summary>
        /// Calculate quantization parameters from observed data.
        /// </summary>
        public QuantizationParams CalculateParams(bool symmetric = true)
        {
            double min, max;

            switch (_method)
            {
                case CalibrationMethod.Percentile:
                    (min, max) = CalculatePercentileBounds();
                    break;

                case CalibrationMethod.Entropy:
                    (min, max) = CalculateEntropyBounds();
                    break;

                case CalibrationMethod.MSE:
                    (min, max) = CalculateMSEBounds();
                    break;

                default: // MinMax
                    min = _minVal;
                    max = _maxVal;
                    break;
            }

            return CalculateQuantParams(min, max, symmetric);
        }

        private (double min, double max) CalculatePercentileBounds()
        {
            if (_samples.Count == 0)
                return (_minVal, _maxVal);

            _samples.Sort();

            var lowerIdx = (int)((_samples.Count - 1) * (1 - _percentile / 100.0));
            var upperIdx = (int)((_samples.Count - 1) * (_percentile / 100.0));

            return (_samples[lowerIdx], _samples[upperIdx]);
        }

        private (double min, double max) CalculateEntropyBounds()
        {
            // Simplified entropy-based calibration
            // In practice, would use histogram binning and KL divergence
            if (_samples.Count == 0)
                return (_minVal, _maxVal);

            // Start with percentile bounds and refine
            var (pMin, pMax) = CalculatePercentileBounds();

            // Use 99th percentile as starting point for entropy method
            return (pMin, pMax);
        }

        private (double min, double max) CalculateMSEBounds()
        {
            // Search for bounds that minimize MSE after quantization
            if (_samples.Count == 0)
                return (_minVal, _maxVal);

            var bestMin = _minVal;
            var bestMax = _maxVal;
            var bestMSE = double.MaxValue;

            // Grid search over different clip ratios
            for (double ratio = 0.9; ratio <= 1.0; ratio += 0.01)
            {
                var testMin = _minVal * ratio;
                var testMax = _maxVal * ratio;

                var testParams = CalculateQuantParams(testMin, testMax, true);
                var mse = ComputeQuantizationMSE(testParams);

                if (mse < bestMSE)
                {
                    bestMSE = mse;
                    bestMin = testMin;
                    bestMax = testMax;
                }
            }

            return (bestMin, bestMax);
        }

        private double ComputeQuantizationMSE(QuantizationParams qparams)
        {
            var scale = qparams.Scales[0];
            var zeroPoint = qparams.ZeroPoints[0];

            bool signed = _dtype == QuantDType.INT8 || _dtype == QuantDType.INT4;
            int bits = (_dtype == QuantDType.INT4 || _dtype == QuantDType.UINT4) ? 4 : 8;
            int qMin = signed ? -(1 << (bits - 1)) : 0;
            int qMax = signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

            double sumSqError = 0;
            var sampleCount = Math.Min(_samples.Count, 10000);

            for (int i = 0; i < sampleCount; i++)
            {
                var original = _samples[i];
                var quantized = Math.Clamp(Math.Round(original / scale) + zeroPoint, qMin, qMax);
                var dequantized = (quantized - zeroPoint) * scale;
                var error = original - dequantized;
                sumSqError += error * error;
            }

            return sumSqError / sampleCount;
        }

        private QuantizationParams CalculateQuantParams(double min, double max, bool symmetric)
        {
            int bits = (_dtype == QuantDType.INT4 || _dtype == QuantDType.UINT4) ? 4 : 8;
            bool signed = _dtype == QuantDType.INT8 || _dtype == QuantDType.INT4;

            int qMin = signed ? -(1 << (bits - 1)) : 0;
            int qMax = signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

            double scale;
            long zeroPoint;

            if (symmetric)
            {
                // Symmetric quantization: zero point is 0
                var absMax = Math.Max(Math.Abs(min), Math.Abs(max));
                scale = absMax / qMax;
                zeroPoint = 0;
            }
            else
            {
                // Asymmetric quantization
                scale = (max - min) / (qMax - qMin);
                zeroPoint = (long)Math.Round(qMin - min / scale);
                zeroPoint = Math.Clamp(zeroPoint, qMin, qMax);
            }

            // Avoid division by zero
            if (scale < 1e-10) scale = 1e-10;

            return new QuantizationParams
            {
                DType = _dtype,
                Scales = new[] { scale },
                ZeroPoints = new[] { zeroPoint },
                Symmetric = symmetric,
                ObservedMin = min,
                ObservedMax = max
            };
        }

        /// <summary>
        /// Reset observer statistics.
        /// </summary>
        public void Reset()
        {
            _minVal = double.MaxValue;
            _maxVal = double.MinValue;
            _samples.Clear();
            _numBatches = 0;
        }
    }

    #endregion

    #region Quantized Operations

    /// <summary>
    /// Quantized tensor operations for efficient inference.
    /// </summary>
    public static class QuantizedOps
    {
        /// <summary>
        /// Quantized matrix multiplication (INT8 x INT8 -> INT32 -> FP32).
        /// </summary>
        public static Tensor QuantizedMatMul(
            QuantizedTensor a, QuantizedTensor b,
            double outputScale, long outputZeroPoint)
        {
            // Dequantize, compute, requantize
            // In production, this would use integer-only arithmetic
            var aDequant = a.Dequantize();
            var bDequant = b.Dequantize();
            return TensorOps.MatMul(aDequant, bDequant);
        }

        /// <summary>
        /// Quantized linear layer.
        /// </summary>
        public static Tensor QuantizedLinear(
            Tensor input,
            QuantizedTensor weight,
            Tensor? bias,
            QuantizationParams? inputParams = null)
        {
            var weightDequant = weight.Dequantize();
            var output = TensorOps.MatMul(input, weightDequant.T());

            if (bias != null)
                output = output.Add(bias);

            return output;
        }

        /// <summary>
        /// Quantized convolution.
        /// </summary>
        public static Tensor QuantizedConv2d(
            Tensor input,
            QuantizedTensor weight,
            Tensor? bias,
            int strideH, int strideW,
            int padH, int padW)
        {
            var weightDequant = weight.Dequantize();
            return Functional.Conv2d(input, weightDequant, bias, strideH, strideW, padH, padW);
        }

        /// <summary>
        /// Fused quantized linear with ReLU activation.
        /// </summary>
        public static Tensor QuantizedLinearReLU(
            Tensor input,
            QuantizedTensor weight,
            Tensor? bias)
        {
            var output = QuantizedLinear(input, weight, bias);
            return output.Apply(x => Math.Max(0, x));
        }
    }

    #endregion

    #region Model Quantization

    /// <summary>
    /// Quantize entire models for efficient inference.
    /// </summary>
    public static class ModelQuantizer
    {
        /// <summary>
        /// Dynamic quantization: quantize weights at load time, activations at runtime.
        /// Best for CPU inference.
        /// </summary>
        public static QuantizedModel QuantizeDynamic(
            Module model,
            QuantDType weightDType = QuantDType.INT8)
        {
            var quantizedWeights = new Dictionary<string, QuantizedTensor>();

            foreach (var (name, tensor) in model.StateDict())
            {
                // Skip small tensors (like biases)
                if (tensor.NumElements < 16)
                {
                    continue;
                }

                var observer = new QuantizationObserver(CalibrationMethod.MinMax, weightDType);
                observer.Observe(tensor);
                var qparams = observer.CalculateParams(symmetric: true);

                quantizedWeights[name] = new QuantizedTensor(tensor.Data, tensor.Shape, qparams);
            }

            return new QuantizedModel(model, quantizedWeights, QuantizationType.Dynamic);
        }

        /// <summary>
        /// Static quantization with calibration data.
        /// Best accuracy, requires calibration dataset.
        /// </summary>
        public static QuantizedModel QuantizeStatic(
            Module model,
            IEnumerable<Tensor> calibrationData,
            QuantDType dtype = QuantDType.INT8,
            CalibrationMethod method = CalibrationMethod.MinMax)
        {
            var observers = new Dictionary<string, QuantizationObserver>();

            // Initialize observers for each layer
            foreach (var (name, _) in model.StateDict())
            {
                observers[name] = new QuantizationObserver(method, dtype);
            }

            // Run calibration
            foreach (var input in calibrationData)
            {
                model.Forward(input);

                // Observe activations (simplified - in practice would hook into forward pass)
                foreach (var (name, tensor) in model.StateDict())
                {
                    observers[name].Observe(tensor);
                }
            }

            // Quantize weights with calibrated parameters
            var quantizedWeights = new Dictionary<string, QuantizedTensor>();

            foreach (var (name, tensor) in model.StateDict())
            {
                if (tensor.NumElements < 16) continue;

                var qparams = observers[name].CalculateParams(symmetric: true);
                quantizedWeights[name] = new QuantizedTensor(tensor.Data, tensor.Shape, qparams);
            }

            return new QuantizedModel(model, quantizedWeights, QuantizationType.Static);
        }

        /// <summary>
        /// Quantization-Aware Training (QAT) preparation.
        /// Inserts fake quantization nodes for training.
        /// </summary>
        public static Module PrepareForQAT(Module model, QuantDType dtype = QuantDType.INT8)
        {
            // Wrap the model with fake quantization
            return new QATWrapper(model, dtype);
        }
    }

    /// <summary>
    /// Wrapper for Quantization-Aware Training.
    /// </summary>
    public class QATWrapper : Module
    {
        private readonly Module _module;
        private readonly QuantDType _dtype;
        private readonly Dictionary<string, QuantizationObserver> _observers;
        private bool _calibrating = true;

        /// <summary>Public API</summary>
        public QATWrapper(Module module, QuantDType dtype)
        {
            _module = module;
            _dtype = dtype;
            _observers = new Dictionary<string, QuantizationObserver>();

            foreach (var (name, _) in module.StateDict())
            {
                _observers[name] = new QuantizationObserver(CalibrationMethod.MinMax, dtype);
            }
        }

        /// <summary>
        /// Enable/disable calibration mode.
        /// </summary>
        public void SetCalibrating(bool calibrating) => _calibrating = calibrating;

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // Apply fake quantization during forward pass
            var fakeQuantInput = FakeQuantize(input, "input");
            var output = _module.Forward(fakeQuantInput);
            return FakeQuantize(output, "output");
        }

        private Tensor FakeQuantize(Tensor tensor, string name)
        {
            if (!_observers.TryGetValue(name, out var observer))
            {
                observer = new QuantizationObserver(CalibrationMethod.MinMax, _dtype);
                _observers[name] = observer;
            }

            if (_calibrating)
            {
                observer.Observe(tensor);
            }

            var qparams = observer.CalculateParams();
            return SimulateQuantization(tensor, qparams);
        }

        private Tensor SimulateQuantization(Tensor tensor, QuantizationParams qparams)
        {
            var scale = qparams.Scales[0];
            var zeroPoint = qparams.ZeroPoints[0];

            int bits = (qparams.DType == QuantDType.INT4 || qparams.DType == QuantDType.UINT4) ? 4 : 8;
            bool signed = qparams.DType == QuantDType.INT8 || qparams.DType == QuantDType.INT4;
            int qMin = signed ? -(1 << (bits - 1)) : 0;
            int qMax = signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

            // Quantize and immediately dequantize (fake quantization)
            var result = new double[tensor.Data.Length];
            for (int i = 0; i < tensor.Data.Length; i++)
            {
                var quantized = Math.Clamp(Math.Round(tensor.Data[i] / scale) + zeroPoint, qMin, qMax);
                result[i] = (quantized - zeroPoint) * scale;
            }

            return new Tensor(result, tensor.Shape);
        }

        /// <summary>Public API</summary>
        public override Dictionary<string, Tensor> StateDict() => _module.StateDict();
        /// <summary>Public API</summary>
        public override void LoadStateDict(Dictionary<string, Tensor> stateDict, bool strict = true)
            => _module.LoadStateDict(stateDict, strict);
    }

    /// <summary>
    /// Container for a quantized model.
    /// </summary>
    public class QuantizedModel
    {
        private readonly Module _originalModule;
        private readonly Dictionary<string, QuantizedTensor> _quantizedWeights;
        private readonly QuantizationType _quantType;

        /// <summary>Public API</summary>
        public QuantizationType QuantType => _quantType;

        /// <summary>Public API</summary>
        public QuantizedModel(
            Module module,
            Dictionary<string, QuantizedTensor> quantizedWeights,
            QuantizationType quantType)
        {
            _originalModule = module;
            _quantizedWeights = quantizedWeights;
            _quantType = quantType;
        }

        /// <summary>
        /// Get memory savings compared to FP32.
        /// </summary>
        public (long originalBytes, long quantizedBytes, double compressionRatio) GetMemorySavings()
        {
            long originalBytes = 0;
            long quantizedBytes = 0;

            foreach (var (name, qtensor) in _quantizedWeights)
            {
                originalBytes += qtensor.NumElements * 4; // FP32 = 4 bytes
                quantizedBytes += qtensor.GetMemoryUsage();
            }

            return (originalBytes, quantizedBytes, (double)originalBytes / quantizedBytes);
        }

        /// <summary>
        /// Run inference with quantized weights.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            // Load dequantized weights
            var stateDict = new Dictionary<string, Tensor>();
            foreach (var (name, qtensor) in _quantizedWeights)
            {
                stateDict[name] = qtensor.Dequantize();
            }

            _originalModule.LoadStateDict(stateDict, strict: false);
            return _originalModule.Forward(input);
        }

        /// <summary>
        /// Save quantized model.
        /// </summary>
        public void Save(string path)
        {
            using var stream = System.IO.File.Create(path);
            using var writer = new System.IO.BinaryWriter(stream);

            // Magic
            writer.Write(0x51534E4C); // "NSLQ"
            writer.Write((ushort)1); // Version

            // Write quantized tensors
            writer.Write(_quantizedWeights.Count);

            foreach (var (name, qtensor) in _quantizedWeights)
            {
                // Name
                var nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
                writer.Write((ushort)nameBytes.Length);
                writer.Write(nameBytes);

                // Shape
                writer.Write(qtensor.Shape.Length);
                foreach (var dim in qtensor.Shape)
                    writer.Write(dim);

                // Quantization params
                writer.Write((byte)qtensor.Params.DType);
                writer.Write(qtensor.Params.Scales[0]);
                writer.Write(qtensor.Params.ZeroPoints[0]);

                // Data
                var rawData = qtensor.GetRawData();
                writer.Write(rawData.Length);
                writer.Write(rawData);
            }
        }

        /// <summary>
        /// Load quantized model.
        /// </summary>
        public static QuantizedModel Load(string path, Module module)
        {
            using var stream = System.IO.File.OpenRead(path);
            using var reader = new System.IO.BinaryReader(stream);

            // Verify magic
            var magic = reader.ReadUInt32();
            if (magic != 0x51534E4C)
                throw new System.IO.InvalidDataException("Invalid quantized model file");

            var version = reader.ReadUInt16();

            var quantizedWeights = new Dictionary<string, QuantizedTensor>();
            var tensorCount = reader.ReadInt32();

            for (int i = 0; i < tensorCount; i++)
            {
                // Name
                var nameLen = reader.ReadUInt16();
                var nameBytes = reader.ReadBytes(nameLen);
                var name = System.Text.Encoding.UTF8.GetString(nameBytes);

                // Shape
                var shapeLen = reader.ReadInt32();
                var shape = new long[shapeLen];
                for (int j = 0; j < shapeLen; j++)
                    shape[j] = reader.ReadInt64();

                // Quantization params
                var dtype = (QuantDType)reader.ReadByte();
                var scale = reader.ReadDouble();
                var zeroPoint = reader.ReadInt64();

                var qparams = new QuantizationParams
                {
                    DType = dtype,
                    Scales = new[] { scale },
                    ZeroPoints = new[] { zeroPoint }
                };

                // Data
                var dataLen = reader.ReadInt32();
                var rawData = reader.ReadBytes(dataLen);

                quantizedWeights[name] = new QuantizedTensor(rawData, shape, qparams);
            }

            return new QuantizedModel(module, quantizedWeights, QuantizationType.Static);
        }
    }

    #endregion

    #region GPTQ-style Quantization

    /// <summary>
    /// GPTQ (Generative Pre-trained Transformer Quantization) style quantization.
    /// Uses second-order information for minimal accuracy loss.
    /// </summary>
    public static class GPTQQuantizer
    {
        /// <summary>
        /// Quantize a weight matrix using GPTQ algorithm.
        /// </summary>
        public static QuantizedTensor QuantizeGPTQ(
            Tensor weight,
            Tensor hessian,
            QuantDType dtype = QuantDType.INT4,
            int groupSize = 128,
            double dampening = 0.01)
        {
            // Simplified GPTQ implementation
            // Full implementation would use layer-wise optimization

            var observer = new QuantizationObserver(CalibrationMethod.MSE, dtype);
            observer.Observe(weight);
            var qparams = observer.CalculateParams(symmetric: true);

            // Apply GPTQ-style error compensation
            var quantized = new double[weight.Data.Length];
            var residual = new double[weight.Data.Length];
            Array.Copy(weight.Data, quantized, weight.Data.Length);

            var scale = qparams.Scales[0];
            var zeroPoint = qparams.ZeroPoints[0];

            int bits = (dtype == QuantDType.INT4 || dtype == QuantDType.UINT4) ? 4 : 8;
            bool signed = dtype == QuantDType.INT8 || dtype == QuantDType.INT4;
            int qMin = signed ? -(1 << (bits - 1)) : 0;
            int qMax = signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

            // Quantize column by column with error compensation
            int cols = (int)weight.Shape[weight.Shape.Length - 1];
            int rows = (int)(weight.NumElements / cols);

            for (int c = 0; c < cols; c++)
            {
                for (int r = 0; r < rows; r++)
                {
                    int idx = r * cols + c;

                    // Quantize with current residual
                    var val = quantized[idx] + residual[idx];
                    var q = Math.Clamp(Math.Round(val / scale) + zeroPoint, qMin, qMax);
                    var dequant = (q - zeroPoint) * scale;

                    // Compute quantization error
                    var error = val - dequant;

                    // Propagate error to remaining columns (simplified)
                    if (c + 1 < cols)
                    {
                        residual[r * cols + c + 1] += error * dampening;
                    }

                    quantized[idx] = dequant;
                }
            }

            return new QuantizedTensor(quantized, weight.Shape, qparams);
        }
    }

    #endregion

    #region AWQ-style Quantization

    /// <summary>
    /// AWQ (Activation-aware Weight Quantization) style quantization.
    /// Protects salient weights based on activation patterns.
    /// </summary>
    public static class AWQQuantizer
    {
        /// <summary>
        /// Quantize weights using AWQ algorithm with activation awareness.
        /// </summary>
        public static QuantizedTensor QuantizeAWQ(
            Tensor weight,
            Tensor activationStats,
            QuantDType dtype = QuantDType.INT4,
            int groupSize = 128)
        {
            // Compute weight saliency based on activation magnitude
            var saliency = ComputeSaliency(weight, activationStats);

            // Find optimal per-group scales that protect salient weights
            var groupScales = ComputeGroupScales(weight, saliency, groupSize, dtype);

            // Quantize with per-group scaling
            var qparams = new QuantizationParams
            {
                DType = dtype,
                Type = QuantizationType.PerGroup,
                Scales = groupScales,
                ZeroPoints = new long[groupScales.Length],
                GroupSize = groupSize,
                Symmetric = true
            };

            return new QuantizedTensor(weight.Data, weight.Shape, qparams);
        }

        private static double[] ComputeSaliency(Tensor weight, Tensor activations)
        {
            // Saliency = |weight| * |activation|
            var saliency = new double[weight.Data.Length];

            for (int i = 0; i < weight.Data.Length; i++)
            {
                var actIdx = i % activations.Data.Length;
                saliency[i] = Math.Abs(weight.Data[i]) * Math.Abs(activations.Data[actIdx]);
            }

            return saliency;
        }

        private static double[] ComputeGroupScales(
            Tensor weight,
            double[] saliency,
            int groupSize,
            QuantDType dtype)
        {
            int numGroups = (weight.Data.Length + groupSize - 1) / groupSize;
            var scales = new double[numGroups];

            int bits = (dtype == QuantDType.INT4 || dtype == QuantDType.UINT4) ? 4 : 8;
            int qMax = (1 << (bits - 1)) - 1;

            for (int g = 0; g < numGroups; g++)
            {
                int start = g * groupSize;
                int end = Math.Min(start + groupSize, weight.Data.Length);

                // Find max absolute value weighted by saliency
                double maxVal = 0;
                double maxSaliency = 0;

                for (int i = start; i < end; i++)
                {
                    var absVal = Math.Abs(weight.Data[i]);
                    if (saliency[i] > maxSaliency || (saliency[i] == maxSaliency && absVal > maxVal))
                    {
                        maxVal = absVal;
                        maxSaliency = saliency[i];
                    }
                }

                scales[g] = maxVal / qMax;
                if (scales[g] < 1e-10) scales[g] = 1e-10;
            }

            return scales;
        }
    }

    #endregion
}