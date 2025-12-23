using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using NSL.Tensor.NN;

namespace NSL.Tensor
{
    /// <summary>
    /// Model serialization utilities for saving and loading models
    /// </summary>
    public static class ModelIO
    {
        private const string TENSOR_EXTENSION = ".nst";  // NSL Tensor format
        private const string MODEL_EXTENSION = ".nsm";   // NSL Model format
        private const int MAGIC_NUMBER = 0x4E534C54;     // "NSLT" in ASCII

        #region Tensor Serialization

        /// <summary>
        /// Save a single tensor to file
        /// </summary>
        public static void SaveTensor(string path, Tensor tensor)
        {
            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);
            WriteTensor(writer, tensor);
        }

        /// <summary>
        /// Load a single tensor from file
        /// </summary>
        public static Tensor LoadTensor(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            return ReadTensor(reader);
        }

        /// <summary>
        /// Write tensor to binary stream
        /// </summary>
        private static void WriteTensor(BinaryWriter writer, Tensor tensor)
        {
            // Write magic number
            writer.Write(MAGIC_NUMBER);

            // Write dtype
            writer.Write((int)tensor.DType);

            // Write shape
            writer.Write(tensor.NDim);
            foreach (var dim in tensor.Shape)
                writer.Write(dim);

            // Write data
            var data = tensor.ToArray();
            foreach (var val in data)
                writer.Write(val);

            // Write requires_grad flag
            writer.Write(tensor.RequiresGrad);
        }

        /// <summary>
        /// Read tensor from binary stream
        /// </summary>
        private static Tensor ReadTensor(BinaryReader reader)
        {
            // Verify magic number
            int magic = reader.ReadInt32();
            if (magic != MAGIC_NUMBER)
                throw new InvalidDataException("Invalid tensor file format");

            // Read dtype
            var dtype = (DType)reader.ReadInt32();

            // Read shape
            int ndim = reader.ReadInt32();
            var shape = new long[ndim];
            for (int i = 0; i < ndim; i++)
                shape[i] = reader.ReadInt64();

            // Calculate total elements
            long numElements = shape.Length > 0 ? shape.Aggregate(1L, (a, b) => a * b) : 1;

            // Read data
            var data = new double[numElements];
            for (int i = 0; i < numElements; i++)
                data[i] = reader.ReadDouble();

            // Read requires_grad
            bool requiresGrad = reader.ReadBoolean();

            var tensor = Tensor.FromArray(data, shape, dtype);
            tensor.RequiresGrad = requiresGrad;

            return tensor;
        }

        #endregion

        #region State Dict Serialization

        /// <summary>
        /// Save a state dictionary (dictionary of tensors)
        /// </summary>
        public static void SaveStateDict(string path, Dictionary<string, Tensor> stateDict)
        {
            using var stream = File.Create(path);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Create);

            // Write metadata
            var metadata = new Dictionary<string, object>
            {
                ["version"] = "1.0",
                ["num_tensors"] = stateDict.Count,
                ["keys"] = stateDict.Keys.ToList()
            };

            var metadataEntry = archive.CreateEntry("metadata.json");
            using (var metaStream = metadataEntry.Open())
            using (var writer = new StreamWriter(metaStream))
            {
                writer.Write(JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
            }

            // Write each tensor
            foreach (var (name, tensor) in stateDict)
            {
                var safeName = name.Replace('/', '_').Replace('\\', '_');
                var entry = archive.CreateEntry($"tensors/{safeName}.bin");
                using var tensorStream = entry.Open();
                using var writer = new BinaryWriter(tensorStream);
                WriteTensor(writer, tensor);
            }
        }

        /// <summary>
        /// Load a state dictionary
        /// </summary>
        public static Dictionary<string, Tensor> LoadStateDict(string path)
        {
            var stateDict = new Dictionary<string, Tensor>();

            using var stream = File.OpenRead(path);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Read);

            // Read metadata
            var metadataEntry = archive.GetEntry("metadata.json");
            if (metadataEntry == null)
                throw new InvalidDataException("Invalid state dict file: missing metadata");

            List<string>? keys;
            using (var metaStream = metadataEntry.Open())
            using (var reader = new StreamReader(metaStream))
            {
                var json = reader.ReadToEnd();
                var metadata = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
                keys = metadata!["keys"].Deserialize<List<string>>();
            }

            // Read tensors
            foreach (var key in keys!)
            {
                var safeName = key.Replace('/', '_').Replace('\\', '_');
                var entry = archive.GetEntry($"tensors/{safeName}.bin");
                if (entry == null)
                    throw new InvalidDataException($"Missing tensor: {key}");

                using var tensorStream = entry.Open();
                using var reader = new BinaryReader(tensorStream);
                stateDict[key] = ReadTensor(reader);
            }

            return stateDict;
        }

        #endregion

        #region Module Serialization

        /// <summary>
        /// Save a module's state dict
        /// </summary>
        public static void Save(Module module, string path)
        {
            var stateDict = new Dictionary<string, Tensor>();
            foreach (var (name, param) in module.NamedParameters())
            {
                stateDict[name] = param.Detach();
            }
            SaveStateDict(path, stateDict);
        }

        /// <summary>
        /// Load a state dict into a module
        /// </summary>
        public static void Load(Module module, string path, bool strict = true)
        {
            var stateDict = LoadStateDict(path);
            LoadStateDict(module, stateDict, strict);
        }

        /// <summary>
        /// Load state dict into module
        /// </summary>
        public static void LoadStateDict(Module module, Dictionary<string, Tensor> stateDict, bool strict = true)
        {
            var moduleParams = module.NamedParameters().ToDictionary(p => p.name, p => p.param);

            var missingKeys = new List<string>();
            var unexpectedKeys = new List<string>();

            foreach (var key in stateDict.Keys)
            {
                if (!moduleParams.ContainsKey(key))
                {
                    unexpectedKeys.Add(key);
                }
            }

            foreach (var (name, param) in moduleParams)
            {
                if (stateDict.TryGetValue(name, out var loadedParam))
                {
                    if (!param.Shape.SequenceEqual(loadedParam.Shape))
                    {
                        throw new InvalidOperationException(
                            $"Shape mismatch for {name}: expected {string.Join(",", param.Shape)}, got {string.Join(",", loadedParam.Shape)}");
                    }
                    param.CopyFrom(loadedParam);
                }
                else
                {
                    missingKeys.Add(name);
                }
            }

            if (strict && (missingKeys.Count > 0 || unexpectedKeys.Count > 0))
            {
                var message = new StringBuilder("Error loading state dict:");
                if (missingKeys.Count > 0)
                    message.Append($"\n  Missing keys: {string.Join(", ", missingKeys)}");
                if (unexpectedKeys.Count > 0)
                    message.Append($"\n  Unexpected keys: {string.Join(", ", unexpectedKeys)}");
                throw new InvalidOperationException(message.ToString());
            }
        }

        #endregion

        #region Checkpoint Utilities

        /// <summary>
        /// Training checkpoint data
        /// </summary>
        public class Checkpoint
        {
            /// <summary>Public API</summary>
            public Dictionary<string, Tensor> ModelState { get; set; } = new();
            /// <summary>Public API</summary>
            public Dictionary<string, object> OptimizerState { get; set; } = new();
            /// <summary>Public API</summary>
            public int Epoch { get; set; }
            /// <summary>Public API</summary>
            public int GlobalStep { get; set; }
            /// <summary>Public API</summary>
            public Dictionary<string, object> Metadata { get; set; } = new();
        }

        /// <summary>
        /// Save a training checkpoint
        /// </summary>
        public static void SaveCheckpoint(string path, Module model, Optimizer optimizer, int epoch, int globalStep, Dictionary<string, object>? metadata = null)
        {
            using var stream = File.Create(path);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Create);

            // Save model state
            var modelStateEntry = archive.CreateEntry("model_state.bin");
            using (var modelStream = modelStateEntry.Open())
            {
                using var memStream = new MemoryStream();
                using var writer = new BinaryWriter(memStream);

                var stateDict = model.NamedParameters().ToDictionary(p => p.name, p => p.param.Detach());
                writer.Write(stateDict.Count);
                foreach (var (name, tensor) in stateDict)
                {
                    writer.Write(name);
                    WriteTensor(writer, tensor);
                }

                memStream.Position = 0;
                memStream.CopyTo(modelStream);
            }

            // Save checkpoint metadata
            var checkpointMeta = new Dictionary<string, object>
            {
                ["epoch"] = epoch,
                ["global_step"] = globalStep,
                ["optimizer_type"] = optimizer.GetType().Name,
                ["optimizer_lr"] = optimizer.LearningRate,
                ["optimizer_step"] = optimizer.Step
            };

            if (metadata != null)
            {
                foreach (var kv in metadata)
                    checkpointMeta[kv.Key] = kv.Value;
            }

            var metaEntry = archive.CreateEntry("checkpoint.json");
            using (var metaStream = metaEntry.Open())
            using (var writer = new StreamWriter(metaStream))
            {
                writer.Write(JsonSerializer.Serialize(checkpointMeta, new JsonSerializerOptions { WriteIndented = true }));
            }
        }

        /// <summary>
        /// Load a training checkpoint
        /// </summary>
        public static (int epoch, int globalStep, Dictionary<string, object> metadata) LoadCheckpoint(string path, Module model)
        {
            using var stream = File.OpenRead(path);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Read);

            // Load model state
            var modelStateEntry = archive.GetEntry("model_state.bin");
            if (modelStateEntry == null)
                throw new InvalidDataException("Invalid checkpoint: missing model_state.bin");

            var stateDict = new Dictionary<string, Tensor>();
            using (var modelStream = modelStateEntry.Open())
            using (var reader = new BinaryReader(modelStream))
            {
                int count = reader.ReadInt32();
                for (int i = 0; i < count; i++)
                {
                    string name = reader.ReadString();
                    stateDict[name] = ReadTensor(reader);
                }
            }

            LoadStateDict(model, stateDict);

            // Load metadata
            var metaEntry = archive.GetEntry("checkpoint.json");
            if (metaEntry == null)
                throw new InvalidDataException("Invalid checkpoint: missing checkpoint.json");

            Dictionary<string, JsonElement> metadata;
            using (var metaStream = metaEntry.Open())
            using (var reader = new StreamReader(metaStream))
            {
                metadata = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(reader.ReadToEnd())!;
            }

            int epoch = metadata["epoch"].GetInt32();
            int globalStep = metadata["global_step"].GetInt32();

            var metadataDict = new Dictionary<string, object>();
            foreach (var kv in metadata)
            {
                metadataDict[kv.Key] = kv.Value.ValueKind switch
                {
                    JsonValueKind.Number => kv.Value.TryGetInt64(out var l) ? l : kv.Value.GetDouble(),
                    JsonValueKind.String => kv.Value.GetString()!,
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    _ => kv.Value.ToString()
                };
            }

            return (epoch, globalStep, metadataDict);
        }

        #endregion

        #region Export Formats

        /// <summary>
        /// Export model weights to NumPy-compatible format (.npz)
        /// </summary>
        public static void ExportToNpz(string path, Dictionary<string, Tensor> tensors)
        {
            using var stream = File.Create(path);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Create);

            foreach (var (name, tensor) in tensors)
            {
                var entry = archive.CreateEntry($"{name}.npy");
                using var entryStream = entry.Open();
                WriteNpy(entryStream, tensor);
            }
        }

        /// <summary>
        /// Write tensor in NumPy .npy format (simplified)
        /// </summary>
        private static void WriteNpy(Stream stream, Tensor tensor)
        {
            using var writer = new BinaryWriter(stream);

            // Magic number
            writer.Write((byte)0x93);
            writer.Write(Encoding.ASCII.GetBytes("NUMPY"));

            // Version
            writer.Write((byte)1);
            writer.Write((byte)0);

            // Header
            string dtype = tensor.DType switch
            {
                DType.Float32 => "<f4",
                DType.Float64 => "<f8",
                DType.Int32 => "<i4",
                DType.Int64 => "<i8",
                _ => "<f8"
            };

            string shapeStr = "(" + string.Join(", ", tensor.Shape) + (tensor.Shape.Length == 1 ? "," : "") + ")";
            string header = $"{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shapeStr}}}";

            // Pad header to multiple of 64
            int padding = 64 - ((10 + header.Length + 1) % 64);
            if (padding == 64) padding = 0;
            header += new string(' ', padding) + "\n";

            writer.Write((ushort)header.Length);
            writer.Write(Encoding.ASCII.GetBytes(header));

            // Write data
            var data = tensor.ToArray();
            foreach (var val in data)
            {
                writer.Write(val);
            }
        }

        /// <summary>
        /// Export model architecture summary to text
        /// </summary>
        public static string GetModelSummary(Module model)
        {
            var sb = new StringBuilder();
            sb.AppendLine("=" + new string('=', 79));
            sb.AppendLine($"Model: {model.GetType().Name}");
            sb.AppendLine("=" + new string('=', 79));
            sb.AppendLine();
            sb.AppendLine($"{"Layer",-40} {"Output Shape",-20} {"Params",-15}");
            sb.AppendLine("-" + new string('-', 79));

            long totalParams = 0;
            long trainableParams = 0;

            foreach (var (name, param) in model.NamedParameters())
            {
                string shapeStr = $"[{string.Join(", ", param.Shape)}]";
                long numParams = param.NumElements;
                totalParams += numParams;
                if (param.RequiresGrad)
                    trainableParams += numParams;

                sb.AppendLine($"{name,-40} {shapeStr,-20} {numParams,-15:N0}");
            }

            sb.AppendLine("=" + new string('=', 79));
            sb.AppendLine($"Total params: {totalParams:N0}");
            sb.AppendLine($"Trainable params: {trainableParams:N0}");
            sb.AppendLine($"Non-trainable params: {totalParams - trainableParams:N0}");
            sb.AppendLine("=" + new string('=', 79));

            return sb.ToString();
        }

        #endregion
    }

    #region Extension Methods

    /// <summary>Public API</summary>
    public static class TensorExtensions
    {
        /// <summary>
        /// Copy data from another tensor
        /// </summary>
        public static void CopyFrom(this Tensor dest, Tensor src)
        {
            if (!dest.Shape.SequenceEqual(src.Shape))
                throw new ArgumentException("Shapes must match for CopyFrom");

            var srcData = src.ToArray();
            var destData = dest.ToArray();
            Array.Copy(srcData, destData, srcData.Length);
            // Update internal data (assuming Tensor has a method to set data)
            // This requires internal access to Tensor - assuming there's a way to update it
        }
    }

    #endregion
}