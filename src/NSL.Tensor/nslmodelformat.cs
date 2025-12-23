using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NSL.Tensor
{
    #region NSL Model Format Core

    /// NSL Native Model Format (.nslm) - NSL's independent model interchange format.
    /// Replaces ONNX dependency with a native, efficient binary format optimized for NSL.
    ///
    /// Format Structure:
    /// ┌─────────────────────────────────────┐
    /// │ Magic (4 bytes): "NSLM"             │
    /// │ Version (2 bytes): Major.Minor      │
    /// │ Flags (2 bytes): Compression, etc   │
    /// │ Header Size (4 bytes)               │
    /// ├─────────────────────────────────────┤
    /// │ Header (JSON):                      │
    /// │   - Model metadata                  │
    /// │   - Input/Output specs              │
    /// │   - OpSet version                   │
    /// ├─────────────────────────────────────┤
    /// │ Graph Section:                      │
    /// │   - Node count                      │
    /// │   - Nodes (binary encoded)          │
    /// │   - Edge definitions                │
    /// ├─────────────────────────────────────┤
    /// │ Tensor Section:                     │
    /// │   - Tensor count                    │
    /// │   - Tensor headers                  │
    /// │   - Tensor data (optionally comp.)  │
    /// └─────────────────────────────────────┘
    /// </summary>
    public sealed class NSLModel : IDisposable
    {
        // Magic: "NSLM" in little-endian
        private const uint Magic = 0x4D4C534E;
        private const ushort FormatVersion = 0x0100; // 1.0

        // Format flags
        /// <summary>API member</summary>
        [Flags]
        public enum FormatFlags : ushort
        {
            None = 0,
            Compressed = 1 << 0,
            HalfPrecision = 1 << 1,
            Quantized = 1 << 2,
            Encrypted = 1 << 3,
            Streaming = 1 << 4,
            MemoryMapped = 1 << 5
        }

        private readonly Dictionary<string, Tensor> _weights;
        private readonly List<NSLNode> _nodes;
        private readonly List<string> _inputs;
        private readonly List<string> _outputs;
        private readonly Dictionary<string, TensorSpec> _tensorSpecs;
        private readonly NSLModelMetadata _metadata;

        /// <summary>Public API</summary>
        public IReadOnlyDictionary<string, Tensor> Weights => _weights;
        /// <summary>Public API</summary>
        public IReadOnlyList<NSLNode> Nodes => _nodes;
        /// <summary>Public API</summary>
        public IReadOnlyList<string> InputNames => _inputs;
        /// <summary>Public API</summary>
        public IReadOnlyList<string> OutputNames => _outputs;
        /// <summary>Public API</summary>
        public NSLModelMetadata Metadata => _metadata;

        /// <summary>Public API</summary>
        public NSLModel()
        {
            _weights = new Dictionary<string, Tensor>();
            _nodes = new List<NSLNode>();
            _inputs = new List<string>();
            _outputs = new List<string>();
            _tensorSpecs = new Dictionary<string, TensorSpec>();
            _metadata = new NSLModelMetadata();
        }

        #region Model Building

        /// Add an input specification to the model.
        /// </summary>
        public NSLModel AddInput(string name, long[] shape, NSLDataType dtype = NSLDataType.Float32)
        {
            _inputs.Add(name);
            _tensorSpecs[name] = new TensorSpec { Name = name, Shape = shape, DType = dtype };
            return this;
        }

        /// Add an output specification to the model.
        /// </summary>
        public NSLModel AddOutput(string name, long[] shape, NSLDataType dtype = NSLDataType.Float32)
        {
            _outputs.Add(name);
            _tensorSpecs[name] = new TensorSpec { Name = name, Shape = shape, DType = dtype };
            return this;
        }

        /// Add a node to the computation graph.
        /// </summary>
        public NSLModel AddNode(NSLNode node)
        {
            _nodes.Add(node);
            return this;
        }

        /// Add a weight tensor.
        /// </summary>
        public NSLModel AddWeight(string name, Tensor tensor)
        {
            _weights[name] = tensor;
            _tensorSpecs[name] = new TensorSpec
            {
                Name = name,
                Shape = tensor.Shape,
                DType = NSLDataType.Float32
            };
            return this;
        }

        /// Set model metadata.
        /// </summary>
        public NSLModel SetMetadata(string name, string producer, string version = "1.0.0")
        {
            _metadata.Name = name;
            _metadata.Producer = producer;
            _metadata.Version = version;
            _metadata.Created = DateTime.UtcNow;
            return this;
        }

        #endregion

        #region Serialization

        /// Save model to file.
        /// </summary>
        public void Save(string path, FormatFlags flags = FormatFlags.Compressed)
        {
            using var stream = File.Create(path);
            Save(stream, flags);
        }

        /// Save model to stream.
        /// </summary>
        public void Save(Stream stream, FormatFlags flags = FormatFlags.Compressed)
        {
            using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

            // Write magic and version
            writer.Write(Magic);
            writer.Write(FormatVersion);
            writer.Write((ushort)flags);

            // Prepare header
            var header = new NSLModelHeader
            {
                Metadata = _metadata,
                Inputs = _inputs.Select(n => _tensorSpecs.GetValueOrDefault(n) ?? new TensorSpec { Name = n }).ToList(),
                Outputs = _outputs.Select(n => _tensorSpecs.GetValueOrDefault(n) ?? new TensorSpec { Name = n }).ToList(),
                OpSetVersion = NSLOpSet.Version,
                NodeCount = _nodes.Count,
                /// <summary>Tensor attribute type</summary>
        TensorCount = _weights.Count
            };

            // Serialize header to JSON
            var headerJson = JsonSerializer.Serialize(header, new JsonSerializerOptions { WriteIndented = false });
            var headerBytes = Encoding.UTF8.GetBytes(headerJson);

            // Write header size and data
            writer.Write(headerBytes.Length);
            writer.Write(headerBytes);

            // Write graph nodes
            WriteGraphSection(writer, flags);

            // Write tensors
            WriteTensorSection(writer, flags);
        }

        private void WriteGraphSection(BinaryWriter writer, FormatFlags flags)
        {
            // Write node count
            writer.Write(_nodes.Count);

            foreach (var node in _nodes)
            {
                // Node ID
                WriteString(writer, node.Name);

                // Op type
                writer.Write((ushort)node.OpType);

                // Inputs
                writer.Write(node.Inputs.Count);
                foreach (var input in node.Inputs)
                    WriteString(writer, input);

                // Outputs
                writer.Write(node.Outputs.Count);
                foreach (var output in node.Outputs)
                    WriteString(writer, output);

                // Attributes
                WriteAttributes(writer, node.Attributes);
            }
        }

        private void WriteTensorSection(BinaryWriter writer, FormatFlags flags)
        {
            bool compress = flags.HasFlag(FormatFlags.Compressed);

            // Write tensor count
            writer.Write(_weights.Count);

            foreach (var (name, tensor) in _weights)
            {
                // Tensor name
                WriteString(writer, name);

                // Data type
                writer.Write((byte)NSLDataType.Float32);

                // Shape
                writer.Write(tensor.Shape.Length);
                foreach (var dim in tensor.Shape)
                    writer.Write(dim);

                // Data
                var floatData = tensor.Data.Select(d => (float)d).ToArray();
                var bytes = MemoryMarshal.Cast<float, byte>(floatData).ToArray();

                if (compress && bytes.Length > 256)
                {
                    using var ms = new MemoryStream();
                    using (var gz = new GZipStream(ms, CompressionLevel.Optimal, leaveOpen: true))
                        gz.Write(bytes);

                    var compressed = ms.ToArray();

                    // Only use compression if it actually saves space
                    if (compressed.Length < bytes.Length * 0.9)
                    {
                        writer.Write(true); // Compressed
                        writer.Write(bytes.Length); // Original size
                        writer.Write(compressed.Length);
                        writer.Write(compressed);
                    }
                    else
                    {
                        writer.Write(false); // Not compressed
                        writer.Write(bytes.Length);
                        writer.Write(bytes);
                    }
                }
                else
                {
                    writer.Write(false); // Not compressed
                    writer.Write(bytes.Length);
                    writer.Write(bytes);
                }
            }
        }

        private static void WriteString(BinaryWriter writer, string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s ?? "");
            writer.Write((ushort)bytes.Length);
            writer.Write(bytes);
        }

        private static void WriteAttributes(BinaryWriter writer, Dictionary<string, NSLAttribute> attributes)
        {
            writer.Write(attributes.Count);

            foreach (var (key, attr) in attributes)
            {
                WriteString(writer, key);
                writer.Write((byte)attr.Type);

                switch (attr.Type)
                {
                    case NSLAttributeType.Int:
                        writer.Write(attr.IntValue);
                        break;
                    case NSLAttributeType.Float:
                        writer.Write(attr.FloatValue);
                        break;
                    case NSLAttributeType.String:
                        WriteString(writer, attr.StringValue ?? "");
                        break;
                    case NSLAttributeType.IntArray:
                        writer.Write(attr.IntArrayValue?.Length ?? 0);
                        foreach (var v in attr.IntArrayValue ?? Array.Empty<long>())
                            writer.Write(v);
                        break;
                    case NSLAttributeType.FloatArray:
                        writer.Write(attr.FloatArrayValue?.Length ?? 0);
                        foreach (var v in attr.FloatArrayValue ?? Array.Empty<double>())
                            writer.Write(v);
                        break;
                    case NSLAttributeType.Tensor:
                        // Inline tensor
                        if (attr.TensorValue != null)
                        {
                            writer.Write(attr.TensorValue.Shape.Length);
                            foreach (var d in attr.TensorValue.Shape)
                                writer.Write(d);
                            writer.Write(attr.TensorValue.NumElements);
                            foreach (var v in attr.TensorValue.Data)
                                writer.Write((float)v);
                        }
                        else
                        {
                            writer.Write(0);
                        }
                        break;
                }
            }
        }

        #endregion

        #region Deserialization

        /// Load model from file.
        /// </summary>
        public static NSLModel Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Model file not found: {path}");

            using var stream = File.OpenRead(path);
            return Load(stream);
        }

        /// Load model from stream.
        /// </summary>
        public static NSLModel Load(Stream stream)
        {
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

            // Verify magic
            var magic = reader.ReadUInt32();
            if (magic != Magic)
                throw new InvalidDataException($"Invalid NSL model file (magic: 0x{magic:X8})");

            // Read version and flags
            var version = reader.ReadUInt16();
            var flags = (FormatFlags)reader.ReadUInt16();

            // Read header
            var headerSize = reader.ReadInt32();
            var headerBytes = reader.ReadBytes(headerSize);
            var headerJson = Encoding.UTF8.GetString(headerBytes);
            var header = JsonSerializer.Deserialize<NSLModelHeader>(headerJson)
                ?? throw new InvalidDataException("Failed to parse model header");

            var model = new NSLModel();

            // Copy metadata
            if (header.Metadata != null)
            {
                model._metadata.Name = header.Metadata.Name;
                model._metadata.Producer = header.Metadata.Producer;
                model._metadata.Version = header.Metadata.Version;
                model._metadata.Created = header.Metadata.Created;
                model._metadata.Custom = header.Metadata.Custom;
            }

            // Copy inputs/outputs
            foreach (var input in header.Inputs ?? Enumerable.Empty<TensorSpec>())
            {
                model._inputs.Add(input.Name);
                model._tensorSpecs[input.Name] = input;
            }
            foreach (var output in header.Outputs ?? Enumerable.Empty<TensorSpec>())
            {
                model._outputs.Add(output.Name);
                model._tensorSpecs[output.Name] = output;
            }

            // Read graph section
            ReadGraphSection(reader, model);

            // Read tensor section
            ReadTensorSection(reader, model, flags);

            return model;
        }

        private static void ReadGraphSection(BinaryReader reader, NSLModel model)
        {
            var nodeCount = reader.ReadInt32();

            for (int i = 0; i < nodeCount; i++)
            {
                var node = new NSLNode();

                // Node name
                node.Name = ReadString(reader);

                // Op type
                node.OpType = (NSLOpType)reader.ReadUInt16();

                // Inputs
                var inputCount = reader.ReadInt32();
                for (int j = 0; j < inputCount; j++)
                    node.Inputs.Add(ReadString(reader));

                // Outputs
                var outputCount = reader.ReadInt32();
                for (int j = 0; j < outputCount; j++)
                    node.Outputs.Add(ReadString(reader));

                // Attributes
                ReadAttributes(reader, node.Attributes);

                model._nodes.Add(node);
            }
        }

        private static void ReadTensorSection(BinaryReader reader, NSLModel model, FormatFlags flags)
        {
            var tensorCount = reader.ReadInt32();

            for (int i = 0; i < tensorCount; i++)
            {
                // Tensor name
                var name = ReadString(reader);

                // Data type
                var dtype = (NSLDataType)reader.ReadByte();

                // Shape
                var shapeLen = reader.ReadInt32();
                var shape = new long[shapeLen];
                for (int j = 0; j < shapeLen; j++)
                    shape[j] = reader.ReadInt64();

                // Data
                var isCompressed = reader.ReadBoolean();
                byte[] bytes;

                if (isCompressed)
                {
                    var originalSize = reader.ReadInt32();
                    var compressedSize = reader.ReadInt32();
                    var compressed = reader.ReadBytes(compressedSize);

                    bytes = new byte[originalSize];
                    using var ms = new MemoryStream(compressed);
                    using var gz = new GZipStream(ms, CompressionMode.Decompress);
                    gz.Read(bytes, 0, originalSize);
                }
                else
                {
                    var dataSize = reader.ReadInt32();
                    bytes = reader.ReadBytes(dataSize);
                }

                // Convert to tensor
                var data = ConvertBytesToDoubles(bytes, dtype);
                model._weights[name] = new Tensor(data, shape);
            }
        }

        private static string ReadString(BinaryReader reader)
        {
            var len = reader.ReadUInt16();
            var bytes = reader.ReadBytes(len);
            return Encoding.UTF8.GetString(bytes);
        }

        private static void ReadAttributes(BinaryReader reader, Dictionary<string, NSLAttribute> attributes)
        {
            var count = reader.ReadInt32();

            for (int i = 0; i < count; i++)
            {
                var key = ReadString(reader);
                var type = (NSLAttributeType)reader.ReadByte();
                var attr = new NSLAttribute { Type = type };

                switch (type)
                {
                    case NSLAttributeType.Int:
                        attr.IntValue = reader.ReadInt64();
                        break;
                    case NSLAttributeType.Float:
                        attr.FloatValue = reader.ReadDouble();
                        break;
                    case NSLAttributeType.String:
                        attr.StringValue = ReadString(reader);
                        break;
                    case NSLAttributeType.IntArray:
                        var intLen = reader.ReadInt32();
                        attr.IntArrayValue = new long[intLen];
                        for (int j = 0; j < intLen; j++)
                            attr.IntArrayValue[j] = reader.ReadInt64();
                        break;
                    case NSLAttributeType.FloatArray:
                        var floatLen = reader.ReadInt32();
                        attr.FloatArrayValue = new double[floatLen];
                        for (int j = 0; j < floatLen; j++)
                            attr.FloatArrayValue[j] = reader.ReadDouble();
                        break;
                    case NSLAttributeType.Tensor:
                        var shapeLen = reader.ReadInt32();
                        if (shapeLen > 0)
                        {
                            var shape = new long[shapeLen];
                            for (int j = 0; j < shapeLen; j++)
                                shape[j] = reader.ReadInt64();
                            var numElements = reader.ReadInt32();
                            var data = new double[numElements];
                            for (int j = 0; j < numElements; j++)
                                data[j] = reader.ReadSingle();
                            attr.TensorValue = new Tensor(data, shape);
                        }
                        break;
                }

                attributes[key] = attr;
            }
        }

        private static double[] ConvertBytesToDoubles(byte[] bytes, NSLDataType dtype)
        {
            return dtype switch
            {
                NSLDataType.Float16 => ConvertFloat16(bytes),
                NSLDataType.BFloat16 => ConvertBFloat16(bytes),
                NSLDataType.Float32 => MemoryMarshal.Cast<byte, float>(bytes).ToArray().Select(f => (double)f).ToArray(),
                NSLDataType.Float64 => MemoryMarshal.Cast<byte, double>(bytes).ToArray(),
                NSLDataType.Int8 => bytes.Select(b => (double)(sbyte)b).ToArray(),
                NSLDataType.Int16 => MemoryMarshal.Cast<byte, short>(bytes).ToArray().Select(s => (double)s).ToArray(),
                NSLDataType.Int32 => MemoryMarshal.Cast<byte, int>(bytes).ToArray().Select(i => (double)i).ToArray(),
                NSLDataType.Int64 => MemoryMarshal.Cast<byte, long>(bytes).ToArray().Select(l => (double)l).ToArray(),
                NSLDataType.UInt8 => bytes.Select(b => (double)b).ToArray(),
                NSLDataType.Bool => bytes.Select(b => b != 0 ? 1.0 : 0.0).ToArray(),
                _ => MemoryMarshal.Cast<byte, float>(bytes).ToArray().Select(f => (double)f).ToArray()
            };
        }

        private static double[] ConvertFloat16(byte[] bytes)
        {
            var result = new double[bytes.Length / 2];
            for (int i = 0; i < result.Length; i++)
            {
                var half = BitConverter.ToUInt16(bytes, i * 2);
                result[i] = HalfToFloat(half);
            }
            return result;
        }

        private static double[] ConvertBFloat16(byte[] bytes)
        {
            var result = new double[bytes.Length / 2];
            for (int i = 0; i < result.Length; i++)
            {
                var bf16 = BitConverter.ToUInt16(bytes, i * 2);
                var floatBits = (uint)bf16 << 16;
                result[i] = BitConverter.UInt32BitsToSingle(floatBits);
            }
            return result;
        }

        private static float HalfToFloat(ushort half)
        {
            int sign = (half >> 15) & 1;
            int exp = (half >> 10) & 0x1F;
            int mant = half & 0x3FF;

            if (exp == 0)
            {
                if (mant == 0) return sign == 1 ? -0.0f : 0.0f;
                float val = mant / 1024.0f * (float)Math.Pow(2, -14);
                return sign == 1 ? -val : val;
            }
            if (exp == 31)
                return mant == 0 ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;

            float result = (1.0f + mant / 1024.0f) * (float)Math.Pow(2, exp - 15);
            return sign == 1 ? -result : result;
        }

        #endregion

        #region Inference

        /// Run inference on the model.
        /// </summary>
        public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> inputs)
        {
            var tensors = new Dictionary<string, Tensor>(_weights);

            // Add inputs
            foreach (var (name, tensor) in inputs)
                tensors[name] = tensor;

            // Execute nodes in order
            foreach (var node in _nodes)
            {
                var result = ExecuteNode(node, tensors);
                if (result != null)
                {
                    foreach (var (name, tensor) in result)
                        tensors[name] = tensor;
                }
            }

            // Collect outputs
            var outputs = new Dictionary<string, Tensor>();
            foreach (var name in _outputs)
            {
                if (tensors.TryGetValue(name, out var tensor))
                    outputs[name] = tensor;
            }

            return outputs;
        }

        private Dictionary<string, Tensor>? ExecuteNode(NSLNode node, Dictionary<string, Tensor> tensors)
        {
            var nodeInputs = node.Inputs
                .Where(tensors.ContainsKey)
                .Select(n => tensors[n])
                .ToList();

            if (nodeInputs.Count == 0 && node.Inputs.Count > 0)
                return null;

            /// <summary>Tensor attribute type</summary>
        Tensor? output = NSLOpSet.Execute(node.OpType, nodeInputs, node.Attributes);

            if (output != null && node.Outputs.Count > 0)
                return new Dictionary<string, Tensor> { [node.Outputs[0]] = output };

            return null;
        }

        #endregion

        #region ONNX Conversion

        /// Import an ONNX model and convert to NSL format.
        /// </summary>
        public static NSLModel FromOnnx(OnnxModel onnx)
        {
            var model = new NSLModel();

            model.SetMetadata("converted_onnx", "NSL Converter", "1.0.0");

            // Copy inputs
            foreach (var input in onnx.InputNames)
                model._inputs.Add(input);

            // Copy outputs
            foreach (var output in onnx.OutputNames)
                model._outputs.Add(output);

            // Copy weights
            foreach (var (name, tensor) in onnx.Weights)
                model._weights[name] = tensor;

            return model;
        }

        /// Import ONNX file and save as NSL model.
        /// </summary>
        public static void ConvertOnnxToNsl(string onnxPath, string nslPath)
        {
            var onnx = OnnxModel.Load(onnxPath);
            var nsl = FromOnnx(onnx);
            nsl.Save(nslPath);
        }

        #endregion

        #region Validation

        /// Validate the model structure.
        /// </summary>
        public ModelValidationResult Validate()
        {
            var result = new ModelValidationResult();

            // Check inputs exist
            if (_inputs.Count == 0)
                result.Warnings.Add("Model has no defined inputs");

            // Check outputs exist
            if (_outputs.Count == 0)
                result.Warnings.Add("Model has no defined outputs");

            // Check all node inputs are available
            var availableTensors = new HashSet<string>(_weights.Keys);
            foreach (var input in _inputs)
                availableTensors.Add(input);

            foreach (var node in _nodes)
            {
                foreach (var input in node.Inputs)
                {
                    if (!availableTensors.Contains(input))
                        result.Errors.Add($"Node '{node.Name}' references undefined input '{input}'");
                }

                // Add node outputs to available tensors
                foreach (var output in node.Outputs)
                    availableTensors.Add(output);

                // Validate op type
                if (!NSLOpSet.IsSupported(node.OpType))
                    result.Warnings.Add($"Node '{node.Name}' uses unsupported op '{node.OpType}'");
            }

            // Check outputs are produced
            foreach (var output in _outputs)
            {
                if (!availableTensors.Contains(output))
                    result.Errors.Add($"Output '{output}' is never produced");
            }

            result.IsValid = result.Errors.Count == 0;
            return result;
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            _weights.Clear();
            _nodes.Clear();
        }
    }

    #endregion

    #region NSL OpSet - Native Operator Definitions

    /// NSL native operator types - replaces ONNX operators.
    /// </summary>
    public enum NSLOpType : ushort
    {
        // Basic arithmetic
        Add = 0x0001,
        Sub = 0x0002,
        Mul = 0x0003,
        Div = 0x0004,
        Neg = 0x0005,
        Abs = 0x0006,
        Pow = 0x0007,
        Sqrt = 0x0008,
        Exp = 0x0009,
        Log = 0x000A,

        // Comparison
        Equal = 0x0020,
        NotEqual = 0x0021,
        Less = 0x0022,
        LessOrEqual = 0x0023,
        Greater = 0x0024,
        GreaterOrEqual = 0x0025,

        // Reduction
        Sum = 0x0040,
        Mean = 0x0041,
        Max = 0x0042,
        Min = 0x0043,
        Prod = 0x0044,
        ArgMax = 0x0045,
        ArgMin = 0x0046,

        // Matrix operations
        MatMul = 0x0060,
        Gemm = 0x0061,
        Transpose = 0x0062,
        Dot = 0x0063,

        // Shape operations
        Reshape = 0x0080,
        Flatten = 0x0081,
        Squeeze = 0x0082,
        Unsqueeze = 0x0083,
        Concat = 0x0084,
        Split = 0x0085,
        Slice = 0x0086,
        Gather = 0x0087,
        Scatter = 0x0088,
        Tile = 0x0089,
        Expand = 0x008A,
        Pad = 0x008B,

        // Activations
        ReLU = 0x00A0,
        LeakyReLU = 0x00A1,
        PReLU = 0x00A2,
        ELU = 0x00A3,
        SELU = 0x00A4,
        Sigmoid = 0x00A5,
        Tanh = 0x00A6,
        Softmax = 0x00A7,
        LogSoftmax = 0x00A8,
        GELU = 0x00A9,
        SiLU = 0x00AA,
        Swish = 0x00AB,
        Mish = 0x00AC,
        HardSigmoid = 0x00AD,
        HardSwish = 0x00AE,
        Softplus = 0x00AF,

        // Convolution
        Conv1d = 0x00C0,
        Conv2d = 0x00C1,
        Conv3d = 0x00C2,
        ConvTranspose1d = 0x00C3,
        ConvTranspose2d = 0x00C4,
        ConvTranspose3d = 0x00C5,
        DepthwiseConv2d = 0x00C6,

        // Pooling
        MaxPool1d = 0x00E0,
        MaxPool2d = 0x00E1,
        MaxPool3d = 0x00E2,
        AvgPool1d = 0x00E3,
        AvgPool2d = 0x00E4,
        AvgPool3d = 0x00E5,
        GlobalMaxPool = 0x00E6,
        GlobalAvgPool = 0x00E7,
        AdaptiveAvgPool2d = 0x00E8,

        // Normalization
        BatchNorm = 0x0100,
        LayerNorm = 0x0101,
        InstanceNorm = 0x0102,
        GroupNorm = 0x0103,
        RMSNorm = 0x0104,

        // Attention
        MultiHeadAttention = 0x0120,
        ScaledDotProductAttention = 0x0121,
        FlashAttention = 0x0122,

        // Recurrent
        LSTM = 0x0140,
        GRU = 0x0141,
        RNN = 0x0142,

        // Embedding
        Embedding = 0x0160,
        EmbeddingBag = 0x0161,
        PositionalEncoding = 0x0162,
        RotaryEmbedding = 0x0163,

        // Linear
        Linear = 0x0180,
        Bilinear = 0x0181,

        // Dropout & Regularization
        Dropout = 0x01A0,
        AlphaDropout = 0x01A1,
        FeatureDropout = 0x01A2,

        // Loss functions
        MSELoss = 0x01C0,
        CrossEntropyLoss = 0x01C1,
        BCELoss = 0x01C2,
        BCEWithLogitsLoss = 0x01C3,
        NLLLoss = 0x01C4,
        CTCLoss = 0x01C5,
        KLDivLoss = 0x01C6,
        HuberLoss = 0x01C7,
        L1Loss = 0x01C8,
        SmoothL1Loss = 0x01C9,

        // Control flow
        Identity = 0x01E0,
        If = 0x01E1,
        Loop = 0x01E2,
        Where = 0x01E3,

        // Special
        Constant = 0x0200,
        ConstantOfShape = 0x0201,
        Shape = 0x0202,
        Size = 0x0203,
        Cast = 0x0204,
        Custom = 0xFFFF
    }

    /// NSL OpSet - defines and executes native operators.
    /// </summary>
    public static class NSLOpSet
    {
        /// <summary>Public API</summary>
        public const int Version = 1;

        private static readonly HashSet<NSLOpType> SupportedOps = new(Enum.GetValues<NSLOpType>());

        /// <summary>Public API</summary>
        public static bool IsSupported(NSLOpType op) => SupportedOps.Contains(op);

        /// Execute an NSL operator.
        /// </summary>
        public static Tensor? Execute(NSLOpType op, List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            if (inputs.Count == 0) return null;

            return op switch
            {
                // Arithmetic
                NSLOpType.Add => inputs.Count >= 2 ? inputs[0].Add(inputs[1]) : inputs[0],
                NSLOpType.Sub => inputs.Count >= 2 ? inputs[0].Sub(inputs[1]) : inputs[0],
                NSLOpType.Mul => inputs.Count >= 2 ? inputs[0].Mul(inputs[1]) : inputs[0],
                NSLOpType.Div => inputs.Count >= 2 ? inputs[0].Div(inputs[1]) : inputs[0],
                NSLOpType.Neg => inputs[0].Apply((Func<double, double>)(x => -x)),
                NSLOpType.Abs => inputs[0].Apply((Func<double, double>)Math.Abs),
                NSLOpType.Pow => ExecutePow(inputs, attrs),
                NSLOpType.Sqrt => inputs[0].Apply((Func<double, double>)Math.Sqrt),
                NSLOpType.Exp => inputs[0].Apply((Func<double, double>)Math.Exp),
                NSLOpType.Log => inputs[0].Apply((Func<double, double>)Math.Log),

                // Matrix ops
                NSLOpType.MatMul => inputs.Count >= 2 ? TensorOps.MatMul(inputs[0], inputs[1]) : inputs[0],
                NSLOpType.Gemm => ExecuteGemm(inputs, attrs),
                NSLOpType.Transpose => inputs[0].T(),
                NSLOpType.Dot => inputs.Count >= 2 ? TensorOps.MatMul(inputs[0], inputs[1]) : inputs[0],

                // Shape ops
                NSLOpType.Reshape => ExecuteReshape(inputs, attrs),
                NSLOpType.Flatten => inputs[0].Flatten(),
                NSLOpType.Squeeze => ExecuteSqueeze(inputs, attrs),
                NSLOpType.Unsqueeze => ExecuteUnsqueeze(inputs, attrs),
                NSLOpType.Concat => ExecuteConcat(inputs, attrs),

                // Activations
                NSLOpType.ReLU => inputs[0].Apply(x => Math.Max(0, x)),
                NSLOpType.LeakyReLU => ExecuteLeakyReLU(inputs, attrs),
                NSLOpType.Sigmoid => inputs[0].Apply(x => 1.0 / (1.0 + Math.Exp(-x))),
                NSLOpType.Tanh => inputs[0].Apply(Math.Tanh),
                NSLOpType.Softmax => ExecuteSoftmax(inputs, attrs),
                NSLOpType.LogSoftmax => ExecuteLogSoftmax(inputs, attrs),
                NSLOpType.GELU => inputs[0].Apply(x => x * 0.5 * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))),
                NSLOpType.SiLU => inputs[0].Apply(x => x / (1.0 + Math.Exp(-x))),
                NSLOpType.Swish => inputs[0].Apply(x => x / (1.0 + Math.Exp(-x))),
                NSLOpType.Softplus => inputs[0].Apply(x => Math.Log(1 + Math.Exp(x))),

                // Reductions
                NSLOpType.Sum => ExecuteReduceSum(inputs, attrs),
                NSLOpType.Mean => ExecuteReduceMean(inputs, attrs),
                NSLOpType.Max => inputs[0].Max(),
                NSLOpType.Min => inputs[0].Min(),

                // Normalization
                NSLOpType.LayerNorm => ExecuteLayerNorm(inputs, attrs),
                NSLOpType.BatchNorm => ExecuteBatchNorm(inputs, attrs),
                NSLOpType.RMSNorm => ExecuteRMSNorm(inputs, attrs),

                // Linear
                NSLOpType.Linear => ExecuteLinear(inputs, attrs),

                // Control
                NSLOpType.Identity => inputs[0],
                NSLOpType.Dropout => inputs[0], // Identity at inference
                NSLOpType.Where => ExecuteWhere(inputs, attrs),

                // Special
                NSLOpType.Constant => attrs.TryGetValue("value", out var v) ? v.TensorValue : inputs[0],
                NSLOpType.Cast => inputs[0], // Keep as is for now
                NSLOpType.Shape => new Tensor(inputs[0].Shape.Select(d => (double)d).ToArray(), new[] { (long)inputs[0].Shape.Length }),

                _ => inputs[0] // Fallback passthrough
            };
        }

        private static Tensor ExecutePow(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            double exponent = 2.0;
            if (attrs.TryGetValue("exponent", out var attr))
                exponent = attr.FloatValue;
            else if (inputs.Count >= 2)
                exponent = inputs[1].Data[0];

            return inputs[0].Apply(x => Math.Pow(x, exponent));
        }

        private static Tensor ExecuteGemm(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            var a = inputs[0];
            var b = inputs.Count > 1 ? inputs[1] : a;

            double alpha = 1.0, beta = 1.0;
            bool transA = false, transB = false;

            if (attrs.TryGetValue("alpha", out var alphaAttr)) alpha = alphaAttr.FloatValue;
            if (attrs.TryGetValue("beta", out var betaAttr)) beta = betaAttr.FloatValue;
            if (attrs.TryGetValue("transA", out var transAAttr)) transA = transAAttr.IntValue != 0;
            if (attrs.TryGetValue("transB", out var transBAttr)) transB = transBAttr.IntValue != 0;

            if (transA) a = a.T();
            if (transB) b = b.T();

            var result = TensorOps.MatMul(a, b);
            if (alpha != 1.0) result = result.Mul(alpha);

            if (inputs.Count > 2 && beta != 0.0)
            {
                var c = inputs[2];
                result = result.Add(c.Mul(beta));
            }

            return result;
        }

        private static Tensor ExecuteReshape(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            long[] shape;
            if (inputs.Count > 1)
            {
                shape = inputs[1].Data.Select(d => (long)d).ToArray();
            }
            else if (attrs.TryGetValue("shape", out var shapeAttr))
            {
                shape = shapeAttr.IntArrayValue ?? new long[] { -1 };
            }
            else
            {
                return inputs[0];
            }

            return inputs[0].Reshape(shape);
        }

        private static Tensor ExecuteSqueeze(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            if (attrs.TryGetValue("axes", out var axesAttr) && axesAttr.IntArrayValue != null)
            {
                var tensor = inputs[0];
                foreach (var axis in axesAttr.IntArrayValue.OrderByDescending(a => a))
                {
                    tensor = tensor.Squeeze((int)axis);
                }
                return tensor;
            }
            return inputs[0].Squeeze();
        }

        private static Tensor ExecuteUnsqueeze(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            int axis = 0;
            if (attrs.TryGetValue("axes", out var axesAttr) && axesAttr.IntArrayValue != null && axesAttr.IntArrayValue.Length > 0)
                axis = (int)axesAttr.IntArrayValue[0];
            else if (inputs.Count > 1)
                axis = (int)inputs[1].Data[0];

            return inputs[0].Unsqueeze(axis);
        }

        private static Tensor ExecuteConcat(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            int axis = 0;
            if (attrs.TryGetValue("axis", out var axisAttr))
                axis = (int)axisAttr.IntValue;

            return TensorOps.Cat(inputs.ToArray(), axis);
        }

        private static Tensor ExecuteLeakyReLU(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            double alpha = 0.01;
            if (attrs.TryGetValue("alpha", out var alphaAttr))
                alpha = alphaAttr.FloatValue;

            return inputs[0].Apply(x => x >= 0 ? x : alpha * x);
        }

        private static Tensor ExecuteSoftmax(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            int axis = -1;
            if (attrs.TryGetValue("axis", out var axisAttr))
                axis = (int)axisAttr.IntValue;

            var x = inputs[0];
            var max = x.Max().ToScalar();
            var exp = x.Apply(v => Math.Exp(v - max));
            var sum = exp.Sum().ToScalar();
            return exp.Div(sum);
        }

        private static Tensor ExecuteLogSoftmax(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            var softmax = ExecuteSoftmax(inputs, attrs);
            return softmax.Apply((Func<double, double>)Math.Log);
        }

        private static Tensor ExecuteReduceSum(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            if (attrs.TryGetValue("axes", out var axesAttr) && axesAttr.IntArrayValue != null)
            {
                var tensor = inputs[0];
                foreach (var axis in axesAttr.IntArrayValue.OrderByDescending(a => a))
                {
                    tensor = tensor.Sum((int)axis);
                }
                return tensor;
            }
            return inputs[0].Sum();
        }

        private static Tensor ExecuteReduceMean(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            if (attrs.TryGetValue("axes", out var axesAttr) && axesAttr.IntArrayValue != null)
            {
                var tensor = inputs[0];
                foreach (var axis in axesAttr.IntArrayValue.OrderByDescending(a => a))
                {
                    tensor = tensor.Mean((int)axis);
                }
                return tensor;
            }
            return inputs[0].Mean();
        }

        private static Tensor ExecuteLayerNorm(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            var x = inputs[0];
            double eps = 1e-5;
            if (attrs.TryGetValue("epsilon", out var epsAttr))
                eps = epsAttr.FloatValue;

            var mean = x.Mean().ToScalar();
            var variance = x.Var().ToScalar();
            var normalized = x.Apply(v => (v - mean) / Math.Sqrt(variance + eps));

            // Apply gamma and beta if provided
            if (inputs.Count > 1)
                normalized = normalized.Mul(inputs[1]);
            if (inputs.Count > 2)
                normalized = normalized.Add(inputs[2]);

            return normalized;
        }

        private static Tensor ExecuteBatchNorm(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            // Simplified batch norm for inference
            var x = inputs[0];
            double eps = 1e-5;
            if (attrs.TryGetValue("epsilon", out var epsAttr))
                eps = epsAttr.FloatValue;

            if (inputs.Count >= 5)
            {
                var gamma = inputs[1];
                var beta = inputs[2];
                var runningMean = inputs[3];
                var runningVar = inputs[4];

                // For now, simplified version
                var mean = runningMean.Mean().ToScalar();
                var variance = runningVar.Mean().ToScalar();

                return x.Apply(v => (v - mean) / Math.Sqrt(variance + eps));
            }

            return x;
        }

        private static Tensor ExecuteRMSNorm(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            var x = inputs[0];
            double eps = 1e-6;
            if (attrs.TryGetValue("epsilon", out var epsAttr))
                eps = epsAttr.FloatValue;

            // RMS = sqrt(mean(x^2))
            var squared = x.Apply(v => v * v);
            var rms = Math.Sqrt(squared.Mean().ToScalar() + eps);

            var normalized = x.Div(rms);

            // Apply weight if provided
            if (inputs.Count > 1)
                normalized = normalized.Mul(inputs[1]);

            return normalized;
        }

        private static Tensor ExecuteLinear(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            var x = inputs[0];
            var weight = inputs.Count > 1 ? inputs[1] : x;

            var result = TensorOps.MatMul(x, weight.T());

            // Add bias if provided
            if (inputs.Count > 2)
                result = result.Add(inputs[2]);

            return result;
        }

        private static Tensor ExecuteWhere(List<Tensor> inputs, Dictionary<string, NSLAttribute> attrs)
        {
            if (inputs.Count < 3) return inputs[0];

            var condition = inputs[0];
            var x = inputs[1];
            var y = inputs[2];

            var result = new double[condition.Data.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = condition.Data[i] != 0 ? x.Data[i % x.Data.Length] : y.Data[i % y.Data.Length];
            }

            return new Tensor(result, condition.Shape);
        }
    }

    #endregion

    #region Supporting Types

    /// NSL data types for tensors.
    /// </summary>
    public enum NSLDataType : byte
    {
        Float16 = 0,
        BFloat16 = 1,
        Float32 = 2,
        Float64 = 3,
        Int8 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        UInt8 = 8,
        UInt16 = 9,
        UInt32 = 10,
        UInt64 = 11,
        Bool = 12,
        Complex64 = 13,
        Complex128 = 14
    }

    /// Attribute types for node configuration.
    /// </summary>
    public enum NSLAttributeType : byte
    {
        Int = 0,
        Float = 1,
        String = 2,
        IntArray = 3,
        FloatArray = 4,
        /// <summary>Tensor attribute type</summary>
        Tensor = 5
    }

    /// Node attribute value.
    /// </summary>
    public class NSLAttribute
    {
        /// <summary>Public API</summary>
        public NSLAttributeType Type { get; set; }
        /// <summary>Public API</summary>
        public long IntValue { get; set; }
        /// <summary>Public API</summary>
        public double FloatValue { get; set; }
        /// <summary>Public API</summary>
        public string? StringValue { get; set; }
        /// <summary>Public API</summary>
        public long[]? IntArrayValue { get; set; }
        /// <summary>Public API</summary>
        public double[]? FloatArrayValue { get; set; }
        /// <summary>Public API</summary>
        public Tensor? TensorValue { get; set; }

        /// <summary>Public API</summary>
        public static NSLAttribute FromInt(long value) => new() { Type = NSLAttributeType.Int, IntValue = value };
        /// <summary>Public API</summary>
        public static NSLAttribute FromFloat(double value) => new() { Type = NSLAttributeType.Float, FloatValue = value };
        /// <summary>Public API</summary>
        public static NSLAttribute FromString(string value) => new() { Type = NSLAttributeType.String, StringValue = value };
        /// <summary>Public API</summary>
        public static NSLAttribute FromIntArray(long[] value) => new() { Type = NSLAttributeType.IntArray, IntArrayValue = value };
        /// <summary>Public API</summary>
        public static NSLAttribute FromFloatArray(double[] value) => new() { Type = NSLAttributeType.FloatArray, FloatArrayValue = value };
        /// <summary>Public API</summary>
        public static NSLAttribute FromTensor(Tensor value) => new() { Type = NSLAttributeType.Tensor, TensorValue = value };
    }

    /// Computation graph node.
    /// </summary>
    public class NSLNode
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public NSLOpType OpType { get; set; }
        /// <summary>Public API</summary>
        public List<string> Inputs { get; } = new();
        /// <summary>Public API</summary>
        public List<string> Outputs { get; } = new();
        /// <summary>Public API</summary>
        public Dictionary<string, NSLAttribute> Attributes { get; } = new();

        /// <summary>Public API</summary>
        public NSLNode() { }

        /// <summary>Public API</summary>
        public NSLNode(string name, NSLOpType opType)
        {
            Name = name;
            OpType = opType;
        }

        /// <summary>Public API</summary>
        public NSLNode WithInput(string input)
        {
            Inputs.Add(input);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLNode WithOutput(string output)
        {
            Outputs.Add(output);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLNode WithAttr(string key, long value)
        {
            Attributes[key] = NSLAttribute.FromInt(value);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLNode WithAttr(string key, double value)
        {
            Attributes[key] = NSLAttribute.FromFloat(value);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLNode WithAttr(string key, string value)
        {
            Attributes[key] = NSLAttribute.FromString(value);
            return this;
        }
    }

    /// Tensor specification for inputs/outputs.
    /// </summary>
    public class TensorSpec
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public long[] Shape { get; set; } = Array.Empty<long>();
        /// <summary>Public API</summary>
        public NSLDataType DType { get; set; } = NSLDataType.Float32;
    }

    /// Model metadata.
    /// </summary>
    public class NSLModelMetadata
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string Producer { get; set; } = "NSL";
        /// <summary>Public API</summary>
        public string Version { get; set; } = "1.0.0";
        /// <summary>Public API</summary>
        public DateTime Created { get; set; } = DateTime.UtcNow;
        /// <summary>Public API</summary>
        public Dictionary<string, string> Custom { get; set; } = new();
    }

    /// Model file header structure.
    /// </summary>
    internal class NSLModelHeader
    {
        /// <summary>Public API</summary>
        public NSLModelMetadata? Metadata { get; set; }
        /// <summary>Public API</summary>
        public List<TensorSpec>? Inputs { get; set; }
        /// <summary>Public API</summary>
        public List<TensorSpec>? Outputs { get; set; }
        /// <summary>Public API</summary>
        public int OpSetVersion { get; set; }
        /// <summary>Public API</summary>
        public int NodeCount { get; set; }
        /// <summary>Public API</summary>
        public int TensorCount { get; set; }
    }

    /// Model validation result.
    /// </summary>
    public class ModelValidationResult
    {
        /// <summary>Public API</summary>
        public bool IsValid { get; set; }
        /// <summary>Public API</summary>
        public List<string> Errors { get; } = new();
        /// <summary>Public API</summary>
        public List<string> Warnings { get; } = new();
    }

    #endregion

    #region Model Builder Helper

    /// Fluent builder for creating NSL models programmatically.
    /// </summary>
    public class NSLModelBuilder
    {
        private readonly NSLModel _model = new();
        private int _nodeCounter = 0;

        /// <summary>Public API</summary>
        public NSLModelBuilder SetMetadata(string name, string producer = "NSL", string version = "1.0.0")
        {
            _model.SetMetadata(name, producer, version);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder AddInput(string name, params long[] shape)
        {
            _model.AddInput(name, shape);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder AddOutput(string name, params long[] shape)
        {
            _model.AddOutput(name, shape);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder AddWeight(string name, Tensor tensor)
        {
            _model.AddWeight(name, tensor);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder AddNode(NSLOpType op, string[] inputs, string[] outputs, Dictionary<string, NSLAttribute>? attrs = null)
        {
            var node = new NSLNode($"node_{_nodeCounter++}", op);
            foreach (var input in inputs) node.Inputs.Add(input);
            foreach (var output in outputs) node.Outputs.Add(output);
            if (attrs != null)
            {
                foreach (var (k, v) in attrs)
                    node.Attributes[k] = v;
            }
            _model.AddNode(node);
            return this;
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder Linear(string input, string weight, string? bias, string output)
        {
            var inputs = bias != null ? new[] { input, weight, bias } : new[] { input, weight };
            return AddNode(NSLOpType.Linear, inputs, new[] { output });
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder MatMul(string a, string b, string output)
        {
            return AddNode(NSLOpType.MatMul, new[] { a, b }, new[] { output });
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder ReLU(string input, string output)
        {
            return AddNode(NSLOpType.ReLU, new[] { input }, new[] { output });
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder GELU(string input, string output)
        {
            return AddNode(NSLOpType.GELU, new[] { input }, new[] { output });
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder Softmax(string input, string output, int axis = -1)
        {
            return AddNode(NSLOpType.Softmax, new[] { input }, new[] { output },
                new Dictionary<string, NSLAttribute> { ["axis"] = NSLAttribute.FromInt(axis) });
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder LayerNorm(string input, string gamma, string beta, string output, double eps = 1e-5)
        {
            return AddNode(NSLOpType.LayerNorm, new[] { input, gamma, beta }, new[] { output },
                new Dictionary<string, NSLAttribute> { ["epsilon"] = NSLAttribute.FromFloat(eps) });
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder Add(string a, string b, string output)
        {
            return AddNode(NSLOpType.Add, new[] { a, b }, new[] { output });
        }

        /// <summary>Public API</summary>
        public NSLModelBuilder Reshape(string input, string output, long[] shape)
        {
            return AddNode(NSLOpType.Reshape, new[] { input }, new[] { output },
                new Dictionary<string, NSLAttribute> { ["shape"] = NSLAttribute.FromIntArray(shape) });
        }

        /// <summary>Public API</summary>
        public NSLModel Build()
        {
            var validation = _model.Validate();
            if (!validation.IsValid)
            {
                throw new InvalidOperationException(
                    $"Model validation failed:\n{string.Join("\n", validation.Errors)}");
            }
            return _model;
        }

        /// <summary>Public API</summary>
        public NSLModel BuildUnchecked() => _model;
    }

    #endregion

    #region Module to Model Conversion

    /// Extension methods for converting NSL modules to the model format.
    /// </summary>
    public static class ModuleToModelExtensions
    {
        /// Export a Module to NSL model format.
        /// </summary>
        public static NSLModel ToNSLModel(this NN.Module module, string name = "model")
        {
            var builder = new NSLModelBuilder();
            builder.SetMetadata(name, "NSL.Tensor", "1.0.0");

            var stateDict = module.StateDict();

            // Add all weights
            foreach (var (weightName, tensor) in stateDict)
            {
                builder.AddWeight(weightName, tensor);
            }

            return builder.BuildUnchecked();
        }

        /// Save a Module directly to .nslm file.
        /// </summary>
        public static void SaveNSLM(this NN.Module module, string path, string name = "model")
        {
            var model = module.ToNSLModel(name);
            model.Save(path);
        }

        /// Load weights from a .nslm file into a Module.
        /// </summary>
        public static void LoadNSLM(this NN.Module module, string path, bool strict = true)
        {
            var model = NSLModel.Load(path);
            module.LoadStateDict(model.Weights.ToDictionary(kv => kv.Key, kv => kv.Value), strict);
        }
    }

    #endregion
}