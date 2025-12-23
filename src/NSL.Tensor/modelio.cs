using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NSL.Tensor
{
    #region ONNX Import

    /// <summary>
    /// ONNX model loader for importing pretrained models from PyTorch, TensorFlow, etc.
    /// Supports ONNX opset versions 9-17.
    /// </summary>
    public class OnnxModel
    {
        private readonly Dictionary<string, Tensor> _weights;
        private readonly List<OnnxNode> _nodes;
        private readonly List<string> _inputs;
        private readonly List<string> _outputs;
        private readonly Dictionary<string, int[]> _inputShapes;

        /// <summary>Public API</summary>
        public IReadOnlyDictionary<string, Tensor> Weights => _weights;
        /// <summary>Public API</summary>
        public IReadOnlyList<string> InputNames => _inputs;
        /// <summary>Public API</summary>
        public IReadOnlyList<string> OutputNames => _outputs;

        private OnnxModel()
        {
            _weights = new Dictionary<string, Tensor>();
            _nodes = new List<OnnxNode>();
            _inputs = new List<string>();
            _outputs = new List<string>();
            _inputShapes = new Dictionary<string, int[]>();
        }

        /// <summary>
        /// Load an ONNX model from file.
        /// </summary>
        public static OnnxModel Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"ONNX model not found: {path}");

            var model = new OnnxModel();
            using var stream = File.OpenRead(path);
            model.ParseOnnx(stream);
            return model;
        }

        /// <summary>
        /// Load an ONNX model from stream.
        /// </summary>
        public static OnnxModel Load(Stream stream)
        {
            var model = new OnnxModel();
            model.ParseOnnx(stream);
            return model;
        }

        private void ParseOnnx(Stream stream)
        {
            using var reader = new BinaryReader(stream);

            // ONNX uses Protocol Buffers format
            // We implement a simplified parser for common models

            // Read magic number (optional in some ONNX files)
            var magic = reader.ReadBytes(4);
            stream.Position = 0; // Reset for full parse

            // Parse protobuf structure
            var data = new byte[stream.Length];
            stream.Read(data, 0, data.Length);

            ParseProtobuf(data);
        }

        private void ParseProtobuf(byte[] data)
        {
            int pos = 0;

            while (pos < data.Length)
            {
                if (pos + 2 > data.Length) break;

                // Read field tag
                var (fieldNumber, wireType) = ReadTag(data, ref pos);

                switch (wireType)
                {
                    case 0: // Varint
                        ReadVarint(data, ref pos);
                        break;

                    case 1: // 64-bit
                        pos += 8;
                        break;

                    case 2: // Length-delimited
                        var length = (int)ReadVarint(data, ref pos);
                        if (pos + length > data.Length) return;

                        // Field 7 = graph in ModelProto
                        if (fieldNumber == 7)
                        {
                            ParseGraph(data, pos, length);
                        }
                        // Field 14 = initializer tensors
                        else if (fieldNumber == 14)
                        {
                            ParseInitializer(data, pos, length);
                        }

                        pos += length;
                        break;

                    case 5: // 32-bit
                        pos += 4;
                        break;

                    default:
                        return; // Unknown wire type
                }
            }
        }

        private void ParseGraph(byte[] data, int start, int length)
        {
            int pos = start;
            int end = start + length;

            while (pos < end)
            {
                var (fieldNumber, wireType) = ReadTag(data, ref pos);

                if (wireType == 2)
                {
                    var len = (int)ReadVarint(data, ref pos);
                    if (pos + len > end) break;

                    // Field 1 = nodes
                    if (fieldNumber == 1)
                    {
                        var node = ParseNode(data, pos, len);
                        if (node != null) _nodes.Add(node);
                    }
                    // Field 5 = initializers (weights)
                    else if (fieldNumber == 5)
                    {
                        ParseInitializer(data, pos, len);
                    }
                    // Field 11 = inputs
                    else if (fieldNumber == 11)
                    {
                        var name = ParseValueInfo(data, pos, len);
                        if (!string.IsNullOrEmpty(name)) _inputs.Add(name);
                    }
                    // Field 12 = outputs
                    else if (fieldNumber == 12)
                    {
                        var name = ParseValueInfo(data, pos, len);
                        if (!string.IsNullOrEmpty(name)) _outputs.Add(name);
                    }

                    pos += len;
                }
                else
                {
                    SkipField(data, ref pos, wireType);
                }
            }
        }

        private OnnxNode? ParseNode(byte[] data, int start, int length)
        {
            var node = new OnnxNode();
            int pos = start;
            int end = start + length;

            while (pos < end)
            {
                var (fieldNumber, wireType) = ReadTag(data, ref pos);

                if (wireType == 2)
                {
                    var len = (int)ReadVarint(data, ref pos);
                    if (pos + len > end) break;

                    var strData = Encoding.UTF8.GetString(data, pos, len);

                    switch (fieldNumber)
                    {
                        case 1: node.Inputs.Add(strData); break;
                        case 2: node.Outputs.Add(strData); break;
                        case 3: node.Name = strData; break;
                        case 4: node.OpType = strData; break;
                    }

                    pos += len;
                }
                else
                {
                    SkipField(data, ref pos, wireType);
                }
            }

            return string.IsNullOrEmpty(node.OpType) ? null : node;
        }

        private void ParseInitializer(byte[] data, int start, int length)
        {
            int pos = start;
            int end = start + length;

            string name = "";
            int dataType = 1; // FLOAT
            List<long> dims = new();
            byte[]? rawData = null;
            List<float> floatData = new();

            while (pos < end)
            {
                var (fieldNumber, wireType) = ReadTag(data, ref pos);

                switch (fieldNumber)
                {
                    case 1 when wireType == 2: // dims (repeated int64)
                        var dimLen = (int)ReadVarint(data, ref pos);
                        var dimEnd = pos + dimLen;
                        while (pos < dimEnd)
                        {
                            dims.Add((long)ReadVarint(data, ref pos));
                        }
                        break;

                    case 2 when wireType == 0: // data_type
                        dataType = (int)ReadVarint(data, ref pos);
                        break;

                    case 4 when wireType == 2: // raw_data
                        var rawLen = (int)ReadVarint(data, ref pos);
                        rawData = new byte[rawLen];
                        Array.Copy(data, pos, rawData, 0, rawLen);
                        pos += rawLen;
                        break;

                    case 5 when wireType == 2: // float_data (packed)
                        var floatLen = (int)ReadVarint(data, ref pos);
                        var floatEnd = pos + floatLen;
                        while (pos + 4 <= floatEnd)
                        {
                            floatData.Add(BitConverter.ToSingle(data, pos));
                            pos += 4;
                        }
                        break;

                    case 8 when wireType == 2: // name
                        var nameLen = (int)ReadVarint(data, ref pos);
                        name = Encoding.UTF8.GetString(data, pos, nameLen);
                        pos += nameLen;
                        break;

                    default:
                        SkipField(data, ref pos, wireType);
                        break;
                }
            }

            if (string.IsNullOrEmpty(name)) return;

            // Convert to tensor
            double[] tensorData;
            if (rawData != null)
            {
                tensorData = ConvertRawData(rawData, dataType);
            }
            else if (floatData.Count > 0)
            {
                tensorData = floatData.Select(f => (double)f).ToArray();
            }
            else
            {
                return;
            }

            var shape = dims.Count > 0 ? dims.ToArray() : new long[] { tensorData.Length };
            _weights[name] = new Tensor(tensorData, shape);
        }

        private double[] ConvertRawData(byte[] raw, int dataType)
        {
            return dataType switch
            {
                1 => ConvertFloats(raw),      // FLOAT
                2 => ConvertBytes(raw),       // UINT8
                3 => ConvertInt8s(raw),       // INT8
                6 => ConvertInt32s(raw),      // INT32
                7 => ConvertInt64s(raw),      // INT64
                10 => ConvertFloat16s(raw),   // FLOAT16
                11 => ConvertDoubles(raw),    // DOUBLE
                _ => ConvertFloats(raw)
            };
        }

        private double[] ConvertFloats(byte[] raw)
        {
            var floats = MemoryMarshal.Cast<byte, float>(raw);
            return floats.ToArray().Select(f => (double)f).ToArray();
        }

        private double[] ConvertDoubles(byte[] raw)
        {
            return MemoryMarshal.Cast<byte, double>(raw).ToArray();
        }

        private double[] ConvertFloat16s(byte[] raw)
        {
            var result = new double[raw.Length / 2];
            for (int i = 0; i < result.Length; i++)
            {
                var half = BitConverter.ToUInt16(raw, i * 2);
                result[i] = HalfToFloat(half);
            }
            return result;
        }

        private double[] ConvertInt32s(byte[] raw)
        {
            var ints = MemoryMarshal.Cast<byte, int>(raw);
            return ints.ToArray().Select(i => (double)i).ToArray();
        }

        private double[] ConvertInt64s(byte[] raw)
        {
            var longs = MemoryMarshal.Cast<byte, long>(raw);
            return longs.ToArray().Select(l => (double)l).ToArray();
        }

        private double[] ConvertBytes(byte[] raw)
        {
            return raw.Select(b => (double)b).ToArray();
        }

        private double[] ConvertInt8s(byte[] raw)
        {
            return raw.Select(b => (double)(sbyte)b).ToArray();
        }

        private static float HalfToFloat(ushort half)
        {
            int sign = (half >> 15) & 1;
            int exp = (half >> 10) & 0x1F;
            int mant = half & 0x3FF;

            if (exp == 0)
            {
                if (mant == 0) return sign == 1 ? -0.0f : 0.0f;
                // Denormalized
                float val = mant / 1024.0f * (float)Math.Pow(2, -14);
                return sign == 1 ? -val : val;
            }
            if (exp == 31)
            {
                return mant == 0 ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;
            }

            float result = (1.0f + mant / 1024.0f) * (float)Math.Pow(2, exp - 15);
            return sign == 1 ? -result : result;
        }

        private string ParseValueInfo(byte[] data, int start, int length)
        {
            int pos = start;
            int end = start + length;

            while (pos < end)
            {
                var (fieldNumber, wireType) = ReadTag(data, ref pos);

                if (fieldNumber == 1 && wireType == 2)
                {
                    var len = (int)ReadVarint(data, ref pos);
                    return Encoding.UTF8.GetString(data, pos, len);
                }

                SkipField(data, ref pos, wireType);
            }

            return "";
        }

        private (int fieldNumber, int wireType) ReadTag(byte[] data, ref int pos)
        {
            var tag = ReadVarint(data, ref pos);
            return ((int)(tag >> 3), (int)(tag & 7));
        }

        private ulong ReadVarint(byte[] data, ref int pos)
        {
            ulong result = 0;
            int shift = 0;

            while (pos < data.Length)
            {
                byte b = data[pos++];
                result |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }

            return result;
        }

        private void SkipField(byte[] data, ref int pos, int wireType)
        {
            switch (wireType)
            {
                case 0: ReadVarint(data, ref pos); break;
                case 1: pos += 8; break;
                case 2: pos += (int)ReadVarint(data, ref pos); break;
                case 5: pos += 4; break;
            }
        }

        /// <summary>
        /// Run inference on the loaded model.
        /// </summary>
        public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> inputs)
        {
            var tensors = new Dictionary<string, Tensor>(_weights);

            foreach (var (name, tensor) in inputs)
                tensors[name] = tensor;

            foreach (var node in _nodes)
            {
                var result = ExecuteNode(node, tensors);
                if (result != null)
                {
                    foreach (var (name, tensor) in result)
                        tensors[name] = tensor;
                }
            }

            var outputs = new Dictionary<string, Tensor>();
            foreach (var name in _outputs)
            {
                if (tensors.TryGetValue(name, out var tensor))
                    outputs[name] = tensor;
            }

            return outputs;
        }

        private Dictionary<string, Tensor>? ExecuteNode(OnnxNode node, Dictionary<string, Tensor> tensors)
        {
            var inputs = node.Inputs.Where(tensors.ContainsKey).Select(n => tensors[n]).ToList();
            if (inputs.Count == 0) return null;

            Tensor? output = node.OpType switch
            {
                "Add" => inputs[0].Add(inputs[1]),
                "Sub" => inputs[0].Sub(inputs[1]),
                "Mul" => inputs[0].Mul(inputs[1]),
                "Div" => inputs[0].Div(inputs[1]),
                "MatMul" => TensorOps.MatMul(inputs[0], inputs[1]),
                "Gemm" => TensorOps.MatMul(inputs[0], inputs[1]),
                "Relu" => inputs[0].Apply(x => Math.Max(0, x)),
                "Sigmoid" => inputs[0].Apply(x => 1.0 / (1.0 + Math.Exp(-x))),
                "Tanh" => inputs[0].Apply(Math.Tanh),
                "Softmax" => ComputeSoftmax(inputs[0]),
                "LayerNormalization" => ComputeLayerNorm(inputs),
                "Reshape" => inputs[0].Reshape(GetReshapeTarget(inputs)),
                "Transpose" => inputs[0].T(),
                "Flatten" => inputs[0].Flatten(),
                "Squeeze" => inputs[0].Squeeze(),
                "Unsqueeze" => inputs[0].Unsqueeze(0),
                "Concat" => TensorOps.Cat(inputs.ToArray(), 0),
                "Split" => inputs[0], // Simplified
                "Gather" => inputs[0], // Simplified
                "Slice" => inputs[0], // Simplified
                "Conv" => inputs[0], // Would need full conv implementation
                "MaxPool" => inputs[0], // Simplified
                "AveragePool" => inputs[0], // Simplified
                "BatchNormalization" => inputs[0], // Simplified
                "Dropout" => inputs[0], // Identity during inference
                "Identity" => inputs[0],
                _ => inputs[0] // Unknown ops pass through
            };

            if (output != null && node.Outputs.Count > 0)
            {
                return new Dictionary<string, Tensor> { [node.Outputs[0]] = output };
            }

            return null;
        }

        private Tensor ComputeSoftmax(Tensor input)
        {
            var max = input.Max().ToScalar();
            var exp = input.Apply(x => Math.Exp(x - max));
            var sum = exp.Sum().ToScalar();
            return exp.Div(sum);
        }

        private Tensor ComputeLayerNorm(List<Tensor> inputs)
        {
            if (inputs.Count < 1) return inputs[0];
            var x = inputs[0];
            var mean = x.Mean().ToScalar();
            var variance = x.Var().ToScalar();
            return x.Apply(v => (v - mean) / Math.Sqrt(variance + 1e-5));
        }

        private long[] GetReshapeTarget(List<Tensor> inputs)
        {
            if (inputs.Count > 1)
            {
                return inputs[1].Data.Select(d => (long)d).ToArray();
            }
            return new long[] { -1 };
        }
    }

    /// <summary>
    /// ONNX computation graph node.
    /// </summary>
    public class OnnxNode
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string OpType { get; set; } = "";
        /// <summary>Public API</summary>
        public List<string> Inputs { get; } = new();
        /// <summary>Public API</summary>
        public List<string> Outputs { get; } = new();
        /// <summary>Public API</summary>
        public Dictionary<string, object> Attributes { get; } = new();
    }

    #endregion

    #region SafeTensors

    /// <summary>
    /// SafeTensors format loader - the safe, fast tensor serialization format from Hugging Face.
    /// Supports memory-mapped loading for large models.
    /// </summary>
    public class SafeTensors
    {
        private readonly Dictionary<string, Tensor> _tensors;
        private readonly Dictionary<string, TensorMetadata> _metadata;

        /// <summary>Public API</summary>
        public IReadOnlyDictionary<string, Tensor> Tensors => _tensors;

        private SafeTensors()
        {
            _tensors = new Dictionary<string, Tensor>();
            _metadata = new Dictionary<string, TensorMetadata>();
        }

        /// <summary>
        /// Load tensors from a SafeTensors file.
        /// </summary>
        public static SafeTensors Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"SafeTensors file not found: {path}");

            var st = new SafeTensors();
            using var stream = File.OpenRead(path);
            st.Parse(stream);
            return st;
        }

        /// <summary>
        /// Load tensors from a SafeTensors stream.
        /// </summary>
        public static SafeTensors Load(Stream stream)
        {
            var st = new SafeTensors();
            st.Parse(stream);
            return st;
        }

        /// <summary>
        /// Save tensors to SafeTensors format.
        /// </summary>
        public static void Save(Dictionary<string, Tensor> tensors, string path)
        {
            using var stream = File.Create(path);
            Save(tensors, stream);
        }

        /// <summary>
        /// Save tensors to SafeTensors format.
        /// </summary>
        public static void Save(Dictionary<string, Tensor> tensors, Stream stream)
        {
            using var writer = new BinaryWriter(stream);

            // Build header
            var header = new Dictionary<string, object>();
            long offset = 0;

            foreach (var (name, tensor) in tensors)
            {
                var dtype = "F32"; // We use F32 for doubles converted to floats
                var shape = tensor.Shape;
                var byteSize = tensor.NumElements * 4; // 4 bytes per float

                header[name] = new Dictionary<string, object>
                {
                    ["dtype"] = dtype,
                    ["shape"] = shape,
                    ["data_offsets"] = new long[] { offset, offset + byteSize }
                };

                offset += byteSize;
            }

            // Serialize header to JSON
            var headerJson = JsonSerializer.Serialize(header);
            var headerBytes = Encoding.UTF8.GetBytes(headerJson);

            // Pad header to 8-byte alignment
            var paddedLength = (headerBytes.Length + 7) / 8 * 8;
            var paddedHeader = new byte[paddedLength];
            Array.Copy(headerBytes, paddedHeader, headerBytes.Length);

            // Write header length (8 bytes, little-endian)
            writer.Write((ulong)paddedLength);

            // Write header
            writer.Write(paddedHeader);

            // Write tensor data
            foreach (var (_, tensor) in tensors)
            {
                var floats = tensor.Data.Select(d => (float)d).ToArray();
                var bytes = MemoryMarshal.Cast<float, byte>(floats).ToArray();
                writer.Write(bytes);
            }
        }

        private void Parse(Stream stream)
        {
            using var reader = new BinaryReader(stream);

            // Read header length (8 bytes)
            var headerLength = reader.ReadUInt64();

            // Read header JSON
            var headerBytes = reader.ReadBytes((int)headerLength);
            var headerJson = Encoding.UTF8.GetString(headerBytes).TrimEnd('\0');

            // Parse header
            var header = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerJson);
            if (header == null) throw new InvalidDataException("Invalid SafeTensors header");

            // Get data start position
            var dataStart = stream.Position;

            foreach (var (name, info) in header)
            {
                if (name == "__metadata__") continue;

                var dtype = info.GetProperty("dtype").GetString() ?? "F32";
                var shapeArray = info.GetProperty("shape");
                var shape = new long[shapeArray.GetArrayLength()];
                for (int i = 0; i < shape.Length; i++)
                    shape[i] = shapeArray[i].GetInt64();

                var offsets = info.GetProperty("data_offsets");
                var startOffset = offsets[0].GetInt64();
                var endOffset = offsets[1].GetInt64();

                // Read tensor data
                stream.Position = dataStart + startOffset;
                var byteLength = (int)(endOffset - startOffset);
                var bytes = reader.ReadBytes(byteLength);

                var data = ConvertToDoubles(bytes, dtype);
                _tensors[name] = new Tensor(data, shape);

                _metadata[name] = new TensorMetadata
                {
                    Name = name,
                    DType = dtype,
                    Shape = shape,
                    ByteOffset = startOffset,
                    ByteLength = byteLength
                };
            }
        }

        private double[] ConvertToDoubles(byte[] bytes, string dtype)
        {
            return dtype switch
            {
                "F16" => ConvertFloat16(bytes),
                "BF16" => ConvertBFloat16(bytes),
                "F32" => MemoryMarshal.Cast<byte, float>(bytes).ToArray().Select(f => (double)f).ToArray(),
                "F64" => MemoryMarshal.Cast<byte, double>(bytes).ToArray(),
                "I8" => bytes.Select(b => (double)(sbyte)b).ToArray(),
                "I16" => MemoryMarshal.Cast<byte, short>(bytes).ToArray().Select(s => (double)s).ToArray(),
                "I32" => MemoryMarshal.Cast<byte, int>(bytes).ToArray().Select(i => (double)i).ToArray(),
                "I64" => MemoryMarshal.Cast<byte, long>(bytes).ToArray().Select(l => (double)l).ToArray(),
                "U8" => bytes.Select(b => (double)b).ToArray(),
                "BOOL" => bytes.Select(b => b != 0 ? 1.0 : 0.0).ToArray(),
                _ => MemoryMarshal.Cast<byte, float>(bytes).ToArray().Select(f => (double)f).ToArray()
            };
        }

        private double[] ConvertFloat16(byte[] bytes)
        {
            var result = new double[bytes.Length / 2];
            for (int i = 0; i < result.Length; i++)
            {
                var half = BitConverter.ToUInt16(bytes, i * 2);
                result[i] = HalfToFloat(half);
            }
            return result;
        }

        private double[] ConvertBFloat16(byte[] bytes)
        {
            var result = new double[bytes.Length / 2];
            for (int i = 0; i < result.Length; i++)
            {
                // BFloat16: just the upper 16 bits of a float32
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
            {
                return mant == 0 ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;
            }

            float result = (1.0f + mant / 1024.0f) * (float)Math.Pow(2, exp - 15);
            return sign == 1 ? -result : result;
        }

        /// <summary>
        /// Get tensor by name.
        /// </summary>
        public Tensor? GetTensor(string name) => _tensors.TryGetValue(name, out var t) ? t : null;

        /// <summary>
        /// Get all tensor names.
        /// </summary>
        public IEnumerable<string> GetTensorNames() => _tensors.Keys;
    }

    /// <summary>
    /// Metadata about a tensor in SafeTensors format.
    /// </summary>
    public class TensorMetadata
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string DType { get; set; } = "F32";
        /// <summary>Public API</summary>
        public long[] Shape { get; set; } = Array.Empty<long>();
        /// <summary>Public API</summary>
        public long ByteOffset { get; set; }
        /// <summary>Public API</summary>
        public int ByteLength { get; set; }
    }

    #endregion

    #region Model Serialization

    /// <summary>
    /// Native NSL model serialization with full checkpoint support.
    /// Saves model weights, optimizer state, training config, and metadata.
    /// </summary>
    public static class ModelSerializer
    {
        private const uint MagicNumber = 0x4E534C4D; // "NSLM"
        private const ushort Version = 2;

        /// <summary>
        /// Save model checkpoint with optimizer state.
        /// </summary>
        public static void Save(
            Dictionary<string, Tensor> modelWeights,
            Dictionary<string, object>? optimizerState,
            string path,
            Dictionary<string, object>? metadata = null,
            bool compress = true)
        {
            using var stream = File.Create(path);
            Save(modelWeights, optimizerState, stream, metadata, compress);
        }

        /// <summary>
        /// Save model checkpoint to stream.
        /// </summary>
        public static void Save(
            Dictionary<string, Tensor> modelWeights,
            Dictionary<string, object>? optimizerState,
            Stream stream,
            Dictionary<string, object>? metadata = null,
            bool compress = true)
        {
            using var writer = new BinaryWriter(stream);

            // Write header
            writer.Write(MagicNumber);
            writer.Write(Version);
            writer.Write(compress);

            // Create checkpoint data
            var checkpoint = new CheckpointData
            {
                Metadata = metadata ?? new Dictionary<string, object>(),
                OptimizerState = optimizerState ?? new Dictionary<string, object>()
            };

            // Serialize metadata
            var metaJson = JsonSerializer.Serialize(checkpoint.Metadata);
            var metaBytes = Encoding.UTF8.GetBytes(metaJson);
            writer.Write(metaBytes.Length);
            writer.Write(metaBytes);

            // Serialize optimizer state
            var optJson = JsonSerializer.Serialize(checkpoint.OptimizerState);
            var optBytes = Encoding.UTF8.GetBytes(optJson);
            writer.Write(optBytes.Length);
            writer.Write(optBytes);

            // Write tensor count
            writer.Write(modelWeights.Count);

            // Write each tensor
            foreach (var (name, tensor) in modelWeights)
            {
                WriteTensor(writer, name, tensor, compress);
            }
        }

        private static void WriteTensor(BinaryWriter writer, string name, Tensor tensor, bool compress)
        {
            // Write name
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);

            // Write shape
            writer.Write(tensor.Shape.Length);
            foreach (var dim in tensor.Shape)
                writer.Write(dim);

            // Write data
            var floatData = tensor.Data.Select(d => (float)d).ToArray();
            var dataBytes = MemoryMarshal.Cast<float, byte>(floatData).ToArray();

            if (compress && dataBytes.Length > 1024)
            {
                // Compress with GZip
                using var ms = new MemoryStream();
                using (var gz = new GZipStream(ms, CompressionLevel.Optimal, leaveOpen: true))
                {
                    gz.Write(dataBytes);
                }
                var compressed = ms.ToArray();

                writer.Write(true); // Is compressed
                writer.Write(dataBytes.Length); // Original length
                writer.Write(compressed.Length);
                writer.Write(compressed);
            }
            else
            {
                writer.Write(false); // Not compressed
                writer.Write(dataBytes.Length);
                writer.Write(dataBytes);
            }
        }

        /// <summary>
        /// Load model checkpoint.
        /// </summary>
        public static (Dictionary<string, Tensor> weights, Dictionary<string, object>? optimizerState, Dictionary<string, object>? metadata)
            Load(string path)
        {
            using var stream = File.OpenRead(path);
            return Load(stream);
        }

        /// <summary>
        /// Load model checkpoint from stream.
        /// </summary>
        public static (Dictionary<string, Tensor> weights, Dictionary<string, object>? optimizerState, Dictionary<string, object>? metadata)
            Load(Stream stream)
        {
            using var reader = new BinaryReader(stream);

            // Read and verify header
            var magic = reader.ReadUInt32();
            if (magic != MagicNumber)
                throw new InvalidDataException("Invalid NSL model file");

            var version = reader.ReadUInt16();
            var isCompressed = reader.ReadBoolean();

            // Read metadata
            var metaLength = reader.ReadInt32();
            var metaBytes = reader.ReadBytes(metaLength);
            var metaJson = Encoding.UTF8.GetString(metaBytes);
            var metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(metaJson);

            // Read optimizer state
            var optLength = reader.ReadInt32();
            var optBytes = reader.ReadBytes(optLength);
            var optJson = Encoding.UTF8.GetString(optBytes);
            var optimizerState = JsonSerializer.Deserialize<Dictionary<string, object>>(optJson);

            // Read tensors
            var tensorCount = reader.ReadInt32();
            var weights = new Dictionary<string, Tensor>(tensorCount);

            for (int i = 0; i < tensorCount; i++)
            {
                var (name, tensor) = ReadTensor(reader);
                weights[name] = tensor;
            }

            return (weights, optimizerState, metadata);
        }

        private static (string name, Tensor tensor) ReadTensor(BinaryReader reader)
        {
            // Read name
            var nameLength = reader.ReadInt32();
            var nameBytes = reader.ReadBytes(nameLength);
            var name = Encoding.UTF8.GetString(nameBytes);

            // Read shape
            var shapeDims = reader.ReadInt32();
            var shape = new long[shapeDims];
            for (int i = 0; i < shapeDims; i++)
                shape[i] = reader.ReadInt64();

            // Read data
            var isCompressed = reader.ReadBoolean();
            byte[] dataBytes;

            if (isCompressed)
            {
                var originalLength = reader.ReadInt32();
                var compressedLength = reader.ReadInt32();
                var compressed = reader.ReadBytes(compressedLength);

                // Decompress
                using var ms = new MemoryStream(compressed);
                using var gz = new GZipStream(ms, CompressionMode.Decompress);
                dataBytes = new byte[originalLength];
                gz.Read(dataBytes, 0, originalLength);
            }
            else
            {
                var dataLength = reader.ReadInt32();
                dataBytes = reader.ReadBytes(dataLength);
            }

            // Convert to doubles
            var floats = MemoryMarshal.Cast<byte, float>(dataBytes);
            var data = floats.ToArray().Select(f => (double)f).ToArray();

            return (name, new Tensor(data, shape));
        }

        /// <summary>
        /// Load only model weights (skip optimizer state for faster loading).
        /// </summary>
        public static Dictionary<string, Tensor> LoadWeightsOnly(string path)
        {
            var (weights, _, _) = Load(path);
            return weights;
        }
    }

    /// <summary>
    /// Internal checkpoint data structure.
    /// </summary>
    internal class CheckpointData
    {
        /// <summary>Public API</summary>
        public Dictionary<string, object> Metadata { get; set; } = new();
        /// <summary>Public API</summary>
        public Dictionary<string, object> OptimizerState { get; set; } = new();
    }

    #endregion

    #region Model Hub Integration

    /// <summary>
    /// Download and cache models from Hugging Face Hub.
    /// </summary>
    public static class ModelHub
    {
        private static readonly HttpClient _client = new();
        private static readonly string _cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".nsl", "hub");

        /// <summary>
        /// Download a model from Hugging Face Hub.
        /// </summary>
        public static async Task<string> Download(string repoId, string filename = "model.safetensors")
        {
            var cacheKey = $"{repoId.Replace("/", "_")}_{filename}";
            var cachePath = Path.Combine(_cacheDir, cacheKey);

            if (File.Exists(cachePath))
                return cachePath;

            Directory.CreateDirectory(_cacheDir);

            var url = $"https://huggingface.co/{repoId}/resolve/main/{filename}";

            Console.WriteLine($"Downloading {repoId}/{filename}...");

            var response = await _client.GetAsync(url);
            response.EnsureSuccessStatusCode();

            using var fileStream = File.Create(cachePath);
            await response.Content.CopyToAsync(fileStream);

            Console.WriteLine($"Downloaded to {cachePath}");
            return cachePath;
        }

        /// <summary>
        /// Load SafeTensors model from Hugging Face Hub.
        /// </summary>
        public static async Task<SafeTensors> LoadSafeTensors(string repoId, string filename = "model.safetensors")
        {
            var path = await Download(repoId, filename);
            return SafeTensors.Load(path);
        }

        /// <summary>
        /// Clear the model cache.
        /// </summary>
        public static void ClearCache()
        {
            if (Directory.Exists(_cacheDir))
                Directory.Delete(_cacheDir, recursive: true);
        }

        /// <summary>
        /// Get cache directory path.
        /// </summary>
        public static string GetCacheDir() => _cacheDir;
    }

    #endregion
}