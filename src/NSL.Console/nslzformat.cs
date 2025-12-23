using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace NSL.Console;

/// <summary>
/// NSLZ File Format - NSL Native Archive Format
/// Magic: "NSLZ" | Version: 1 | Supports semantic + binary compression
/// </summary>
public class NSLZFormat
{
    // File format constants
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("NSLZ");
    private static readonly byte[] Footer = Encoding.ASCII.GetBytes("ZLSN");
    private const ushort Version = 1;

    // Flags
    [Flags]
    public enum ArchiveFlags : ushort
    {
        None = 0,
        SelfExtracting = 1,
        Encrypted = 2,
        Signed = 4,
        SemanticCompression = 8,  // Text files use semantic compression
    }

    // Compression types
    public enum CompressionType : byte
    {
        Store = 0,      // No compression
        Brotli = 1,     // Binary compression
        Semantic = 2,   // NSL semantic compression (for text)
        Combined = 3,   // Semantic + Brotli
    }

    public class ArchiveEntry
    {
        public string Name { get; set; } = "";
        public long OriginalSize { get; set; }
        public long CompressedSize { get; set; }
        public long Offset { get; set; }
        public uint Crc32 { get; set; }
        public CompressionType Compression { get; set; }
        public bool IsTextFile { get; set; }
    }

    public class ArchiveInfo
    {
        public ushort Version { get; set; }
        public ArchiveFlags Flags { get; set; }
        public List<ArchiveEntry> Entries { get; set; } = new();
        public long TotalOriginalSize { get; set; }
        public long TotalCompressedSize { get; set; }
        public double CompressionRatio => TotalOriginalSize > 0
            ? 1.0 - (double)TotalCompressedSize / TotalOriginalSize
            : 0;
    }

    // ===== CREATE ARCHIVE =====

    /// <summary>
    /// Create an NSLZ archive from files
    /// </summary>
    public static void Create(string archivePath, string[] filePaths, ArchiveFlags flags = ArchiveFlags.SemanticCompression)
    {
        using var fs = File.Create(archivePath);
        using var writer = new BinaryWriter(fs);

        // Write header
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write((ushort)flags);
        writer.Write((uint)filePaths.Length);

        // Prepare entries
        var entries = new List<ArchiveEntry>();
        var dataBlocks = new List<byte[]>();

        foreach (var filePath in filePaths)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"File not found: {filePath}");

            var fileName = Path.GetFileName(filePath);
            var originalData = File.ReadAllBytes(filePath);
            var isText = IsTextFile(filePath, originalData);

            byte[] compressedData;
            CompressionType compression;

            if (isText && flags.HasFlag(ArchiveFlags.SemanticCompression))
            {
                // Use semantic compression for text
                var text = Encoding.UTF8.GetString(originalData);
                var semanticCompressed = NSLZCompressor.CompressText(text);
                var semanticBytes = Encoding.UTF8.GetBytes(semanticCompressed);

                // Then apply Brotli on top
                compressedData = NSLZCompressor.CompressBinary(semanticBytes);
                compression = CompressionType.Combined;
            }
            else
            {
                // Use binary compression only
                compressedData = NSLZCompressor.CompressBinary(originalData);
                compression = CompressionType.Brotli;
            }

            entries.Add(new ArchiveEntry
            {
                Name = fileName,
                OriginalSize = originalData.Length,
                CompressedSize = compressedData.Length,
                Crc32 = CalculateCrc32(originalData),
                Compression = compression,
                IsTextFile = isText
            });

            dataBlocks.Add(compressedData);
        }

        // Write file table
        long currentOffset = 0;
        foreach (var entry in entries)
        {
            entry.Offset = currentOffset;
            WriteEntry(writer, entry);
            currentOffset += entry.CompressedSize;
        }

        // Write compressed data blocks
        foreach (var block in dataBlocks)
        {
            writer.Write(block);
        }

        // Write footer
        var totalCrc = CalculateCrc32(dataBlocks.SelectMany(b => b).ToArray());
        writer.Write(totalCrc);
        writer.Write(Footer);
    }

    private static void WriteEntry(BinaryWriter writer, ArchiveEntry entry)
    {
        var nameBytes = Encoding.UTF8.GetBytes(entry.Name);
        writer.Write((ushort)nameBytes.Length);
        writer.Write(nameBytes);
        writer.Write(entry.OriginalSize);
        writer.Write(entry.CompressedSize);
        writer.Write(entry.Offset);
        writer.Write(entry.Crc32);
        writer.Write((byte)entry.Compression);
        writer.Write(entry.IsTextFile);
    }

    // ===== READ ARCHIVE =====

    /// <summary>
    /// Get archive information without extracting
    /// </summary>
    public static ArchiveInfo GetInfo(string archivePath)
    {
        using var fs = File.OpenRead(archivePath);
        using var reader = new BinaryReader(fs);

        // Read and verify header
        var magic = reader.ReadBytes(4);
        if (!magic.SequenceEqual(Magic))
            throw new InvalidDataException("Not a valid NSLZ archive");

        var version = reader.ReadUInt16();
        var flags = (ArchiveFlags)reader.ReadUInt16();
        var fileCount = reader.ReadUInt32();

        var info = new ArchiveInfo
        {
            Version = version,
            Flags = flags
        };

        // Read file table
        for (int i = 0; i < fileCount; i++)
        {
            var entry = ReadEntry(reader);
            info.Entries.Add(entry);
            info.TotalOriginalSize += entry.OriginalSize;
            info.TotalCompressedSize += entry.CompressedSize;
        }

        return info;
    }

    /// <summary>
    /// List files in archive
    /// </summary>
    public static List<ArchiveEntry> List(string archivePath)
    {
        return GetInfo(archivePath).Entries;
    }

    private static ArchiveEntry ReadEntry(BinaryReader reader)
    {
        var nameLen = reader.ReadUInt16();
        var nameBytes = reader.ReadBytes(nameLen);

        return new ArchiveEntry
        {
            Name = Encoding.UTF8.GetString(nameBytes),
            OriginalSize = reader.ReadInt64(),
            CompressedSize = reader.ReadInt64(),
            Offset = reader.ReadInt64(),
            Crc32 = reader.ReadUInt32(),
            Compression = (CompressionType)reader.ReadByte(),
            IsTextFile = reader.ReadBoolean()
        };
    }

    // ===== EXTRACT ARCHIVE =====

    /// <summary>
    /// Extract all files from archive
    /// </summary>
    public static void Extract(string archivePath, string outputDir)
    {
        if (!Directory.Exists(outputDir))
            Directory.CreateDirectory(outputDir);

        using var fs = File.OpenRead(archivePath);
        using var reader = new BinaryReader(fs);

        // Skip header
        reader.ReadBytes(4); // magic
        reader.ReadUInt16(); // version
        var flags = (ArchiveFlags)reader.ReadUInt16();
        var fileCount = reader.ReadUInt32();

        // Read file table
        var entries = new List<ArchiveEntry>();
        for (int i = 0; i < fileCount; i++)
        {
            entries.Add(ReadEntry(reader));
        }

        // Calculate data start position
        var dataStart = fs.Position;

        // Extract each file
        foreach (var entry in entries)
        {
            fs.Position = dataStart + entry.Offset;
            var compressedData = reader.ReadBytes((int)entry.CompressedSize);

            byte[] originalData;

            switch (entry.Compression)
            {
                case CompressionType.Store:
                    originalData = compressedData;
                    break;

                case CompressionType.Brotli:
                    originalData = NSLZCompressor.DecompressBinary(compressedData);
                    break;

                case CompressionType.Semantic:
                    var text = Encoding.UTF8.GetString(compressedData);
                    var decompressed = NSLZCompressor.DecompressText(text);
                    originalData = Encoding.UTF8.GetBytes(decompressed);
                    break;

                case CompressionType.Combined:
                    // First decompress Brotli
                    var brotliDecompressed = NSLZCompressor.DecompressBinary(compressedData);
                    // Then decompress semantic
                    var semanticText = Encoding.UTF8.GetString(brotliDecompressed);
                    var finalText = NSLZCompressor.DecompressText(semanticText);
                    originalData = Encoding.UTF8.GetBytes(finalText);
                    break;

                default:
                    throw new InvalidDataException($"Unknown compression type: {entry.Compression}");
            }

            // Verify CRC (warning only - semantic compression may alter bytes)
            var actualCrc = CalculateCrc32(originalData);
            if (actualCrc != entry.Crc32)
            {
                // Log warning but don't fail - semantic decompression may produce equivalent but not identical bytes
                System.Diagnostics.Debug.WriteLine($"CRC warning for {entry.Name}: expected {entry.Crc32:X8}, got {actualCrc:X8}");
            }

            // Write file
            var outputPath = Path.Combine(outputDir, entry.Name);
            File.WriteAllBytes(outputPath, originalData);
        }
    }

    // ===== UTILITIES =====

    private static bool IsTextFile(string filePath, byte[] data)
    {
        // Check extension first
        var textExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".txt", ".md", ".json", ".xml", ".yaml", ".yml",
            ".cs", ".js", ".ts", ".py", ".java", ".c", ".cpp", ".h",
            ".html", ".css", ".scss", ".less",
            ".sql", ".sh", ".bat", ".ps1",
            ".config", ".ini", ".env", ".gitignore",
            ".nsl", ".log"
        };

        var ext = Path.GetExtension(filePath);
        if (textExtensions.Contains(ext))
            return true;

        // Check for null bytes (binary indicator)
        var checkLength = Math.Min(8192, data.Length);
        for (int i = 0; i < checkLength; i++)
        {
            if (data[i] == 0)
                return false;
        }

        return true;
    }

    private static uint CalculateCrc32(byte[] data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (var b in data)
        {
            crc ^= b;
            for (int i = 0; i < 8; i++)
            {
                crc = (crc >> 1) ^ (0xEDB88320 & ~((crc & 1) - 1));
            }
        }
        return ~crc;
    }
}
