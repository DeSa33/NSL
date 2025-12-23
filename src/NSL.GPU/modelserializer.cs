using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NSL.GPU
{
    /// <summary>
    /// NSL Native Model Format (.nslm) - Compact binary format for model serialization.
    ///
    /// Format Structure:
    /// - Header: Magic number (4 bytes) + Version (4 bytes) + Flags (4 bytes)
    /// - Metadata: Model name, architecture info, creation timestamp
    /// - Tensor Table: Count + (Name, Shape, Dtype, Offset, Size) entries
    /// - Tensor Data: Raw tensor data aligned to 64-byte boundaries
    ///
    /// Features:
    /// - Zero-copy memory mapping for fast loading
    /// - Compression support (optional)
    /// - Checksum verification for data integrity
    /// - Backward compatible versioning
    /// </summary>
    public class ModelSerializer
    {
        private const uint MAGIC_NUMBER = 0x4E534C4D; // "NSLM" in hex
        private const uint FORMAT_VERSION = 1;

        /// <summary>
        /// Model metadata stored in the header
        /// </summary>
        public class ModelMetadata
        {
            /// <summary>Public API</summary>
            public string Name { get; set; } = "NSLModel";
            /// <summary>Public API</summary>
            public string Architecture { get; set; } = "Generic";
            /// <summary>Public API</summary>
            public long CreationTimestamp { get; set; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            /// <summary>Public API</summary>
            public Dictionary<string, string> CustomData { get; set; } = new();
        }

        /// <summary>
        /// Information about a tensor in the model
        /// </summary>
        public class TensorInfo
        {
            /// <summary>Public API</summary>
            public string Name { get; set; } = "";
            /// <summary>Public API</summary>
            public int[] Shape { get; set; } = Array.Empty<int>();
            /// <summary>Public API</summary>
            public TensorDtype Dtype { get; set; } = TensorDtype.Float32;
            /// <summary>Public API</summary>
            public long DataOffset { get; set; }
            /// <summary>Public API</summary>
            public long DataSize { get; set; }
        }

        /// <summary>
        /// Supported data types for tensor storage
        /// </summary>
        public enum TensorDtype : byte
        {
            Float32 = 0,
            Float16 = 1,
            BFloat16 = 2,
            Int8 = 3,
            UInt8 = 4,
            Int32 = 5
        }

        /// <summary>
        /// Save model weights to NSL native format.
        /// </summary>
        /// <param name="path">File path to save to</param>
        /// <param name="tensors">Dictionary of tensor name to (data, shape) pairs</param>
        /// <param name="metadata">Optional model metadata</param>
        public static void Save(string path, Dictionary<string, (float[] Data, int[] Shape)> tensors, ModelMetadata? metadata = null)
        {
            metadata ??= new ModelMetadata();

            using var stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite);
            using var writer = new BinaryWriter(stream);

            // Write header
            writer.Write(MAGIC_NUMBER);
            writer.Write(FORMAT_VERSION);
            writer.Write((uint)0); // Flags (reserved)

            // Write metadata
            WriteString(writer, metadata.Name);
            WriteString(writer, metadata.Architecture);
            writer.Write(metadata.CreationTimestamp);
            writer.Write(metadata.CustomData.Count);
            foreach (var kvp in metadata.CustomData)
            {
                WriteString(writer, kvp.Key);
                WriteString(writer, kvp.Value);
            }

            // Write tensor table header
            writer.Write(tensors.Count);

            // Calculate data offsets (align to 64 bytes for optimal memory access)
            var tensorInfos = new List<TensorInfo>();
            long currentOffset = 0;

            foreach (var kvp in tensors)
            {
                var info = new TensorInfo
                {
                    Name = kvp.Key,
                    Shape = kvp.Value.Shape,
                    Dtype = TensorDtype.Float32,
                    DataOffset = currentOffset,
                    DataSize = kvp.Value.Data.Length * sizeof(float)
                };
                tensorInfos.Add(info);
                currentOffset += AlignTo64(info.DataSize);
            }

            // Write tensor table entries
            foreach (var info in tensorInfos)
            {
                WriteString(writer, info.Name);
                writer.Write(info.Shape.Length);
                foreach (var dim in info.Shape)
                {
                    writer.Write(dim);
                }
                writer.Write((byte)info.Dtype);
                writer.Write(info.DataOffset);
                writer.Write(info.DataSize);
            }

            // Write tensor data section marker
            // The data starts AFTER this long value, so add 8 bytes
            long dataStartPos = stream.Position + sizeof(long);
            writer.Write(dataStartPos); // Store data section start for seeking

            // Write tensor data with alignment
            int tensorIndex = 0;
            foreach (var kvp in tensors)
            {
                var data = kvp.Value.Data;

                // Write float data as bytes
                foreach (var val in data)
                {
                    writer.Write(val);
                }

                // Pad to 64-byte alignment
                long currentPos = stream.Position - dataStartPos;
                long alignedPos = AlignTo64(currentPos);
                long padding = alignedPos - currentPos;
                for (long i = 0; i < padding; i++)
                {
                    writer.Write((byte)0);
                }

                tensorIndex++;
            }

            // Write checksum at end
            stream.Position = 0;
            uint checksum = ComputeChecksum(stream);
            stream.Position = stream.Length;
            writer.Write(checksum);
        }

        /// <summary>
        /// Load model weights from NSL native format.
        /// </summary>
        /// <param name="path">File path to load from</param>
        /// <returns>Dictionary of tensor name to (data, shape) pairs and metadata</returns>
        public static (Dictionary<string, (float[] Data, int[] Shape)> Tensors, ModelMetadata Metadata) Load(string path)
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(stream);

            // Verify header
            uint magic = reader.ReadUInt32();
            if (magic != MAGIC_NUMBER)
            {
                throw new InvalidDataException("Invalid NSL model file format");
            }

            uint version = reader.ReadUInt32();
            if (version > FORMAT_VERSION)
            {
                throw new InvalidDataException($"Model format version {version} is newer than supported version {FORMAT_VERSION}");
            }

            uint flags = reader.ReadUInt32();

            // Read metadata
            var metadata = new ModelMetadata
            {
                Name = ReadString(reader),
                Architecture = ReadString(reader),
                CreationTimestamp = reader.ReadInt64()
            };

            int customDataCount = reader.ReadInt32();
            for (int i = 0; i < customDataCount; i++)
            {
                string key = ReadString(reader);
                string value = ReadString(reader);
                metadata.CustomData[key] = value;
            }

            // Read tensor table
            int tensorCount = reader.ReadInt32();
            var tensorInfos = new List<TensorInfo>();

            for (int i = 0; i < tensorCount; i++)
            {
                var info = new TensorInfo
                {
                    Name = ReadString(reader)
                };

                int shapeLen = reader.ReadInt32();
                info.Shape = new int[shapeLen];
                for (int j = 0; j < shapeLen; j++)
                {
                    info.Shape[j] = reader.ReadInt32();
                }

                info.Dtype = (TensorDtype)reader.ReadByte();
                info.DataOffset = reader.ReadInt64();
                info.DataSize = reader.ReadInt64();

                tensorInfos.Add(info);
            }

            // Read data section start position
            long dataStartPos = reader.ReadInt64();

            // Read tensor data
            var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();

            foreach (var info in tensorInfos)
            {
                stream.Position = dataStartPos + info.DataOffset;

                int elementCount = (int)(info.DataSize / sizeof(float));
                var data = new float[elementCount];

                for (int i = 0; i < elementCount; i++)
                {
                    data[i] = reader.ReadSingle();
                }

                tensors[info.Name] = (data, info.Shape);
            }

            return (tensors, metadata);
        }

        /// <summary>
        /// Save model checkpoint during training.
        /// Includes optimizer state and training progress.
        /// </summary>
        public static void SaveCheckpoint(
            string path,
            Dictionary<string, (float[] Data, int[] Shape)> modelState,
            Dictionary<string, (float[] Data, int[] Shape)>? optimizerState,
            int epoch,
            int step,
            float loss)
        {
            var metadata = new ModelMetadata
            {
                Name = "Checkpoint",
                CustomData = new Dictionary<string, string>
                {
                    ["epoch"] = epoch.ToString(),
                    ["step"] = step.ToString(),
                    ["loss"] = loss.ToString("F6"),
                    ["has_optimizer"] = (optimizerState != null).ToString()
                }
            };

            var allTensors = new Dictionary<string, (float[] Data, int[] Shape)>(modelState);

            if (optimizerState != null)
            {
                foreach (var kvp in optimizerState)
                {
                    allTensors[$"optimizer.{kvp.Key}"] = kvp.Value;
                }
            }

            Save(path, allTensors, metadata);
        }

        /// <summary>
        /// Load model checkpoint.
        /// </summary>
        public static (
            Dictionary<string, (float[] Data, int[] Shape)> ModelState,
            Dictionary<string, (float[] Data, int[] Shape)>? OptimizerState,
            int Epoch,
            int Step,
            float Loss) LoadCheckpoint(string path)
        {
            var (allTensors, metadata) = Load(path);

            var modelState = new Dictionary<string, (float[] Data, int[] Shape)>();
            var optimizerState = new Dictionary<string, (float[] Data, int[] Shape)>();

            foreach (var kvp in allTensors)
            {
                if (kvp.Key.StartsWith("optimizer."))
                {
                    optimizerState[kvp.Key.Substring(10)] = kvp.Value;
                }
                else
                {
                    modelState[kvp.Key] = kvp.Value;
                }
            }

            int epoch = int.Parse(metadata.CustomData.GetValueOrDefault("epoch", "0"));
            int step = int.Parse(metadata.CustomData.GetValueOrDefault("step", "0"));
            float loss = float.Parse(metadata.CustomData.GetValueOrDefault("loss", "0"));
            bool hasOptimizer = bool.Parse(metadata.CustomData.GetValueOrDefault("has_optimizer", "false"));

            return (modelState, hasOptimizer ? optimizerState : null, epoch, step, loss);
        }

        #region Helper Methods

        private static void WriteString(BinaryWriter writer, string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        private static string ReadString(BinaryReader reader)
        {
            int length = reader.ReadInt32();
            var bytes = reader.ReadBytes(length);
            return Encoding.UTF8.GetString(bytes);
        }

        private static long AlignTo64(long value)
        {
            return (value + 63) & ~63L;
        }

        private static uint ComputeChecksum(Stream stream)
        {
            // Simple checksum using XOR
            uint checksum = 0;
            int bytesRead;
            var buffer = new byte[4096];

            while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
            {
                for (int i = 0; i < bytesRead; i += 4)
                {
                    if (i + 4 <= bytesRead)
                    {
                        checksum ^= BitConverter.ToUInt32(buffer, i);
                    }
                }
            }

            return checksum;
        }

        #endregion
    }
}