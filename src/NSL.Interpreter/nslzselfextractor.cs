using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Text;

namespace NSL.Core;

/// <summary>
/// NSLZ Self-Extracting Archive Builder
/// Creates standalone .exe files that extract themselves without needing NSL installed
/// </summary>
public class NSLZSelfExtractor
{
    // Stub executable template (minimal C# that decompresses NSLZ)
    private const string StubSourceTemplate = @"
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

class NSLZExtractor
{
    // Semantic dictionary - SYNCED with NSLZCompressor.cs
    private static readonly Dictionary<string, string> DecompressDict = new()
    {
        // Tier 1: Must match NSLZCompressor exactly
        {""∀"", ""the""}, {""∁"", ""a""}, {""∂"", ""an""},
        {""∃"", ""i""}, {""∄"", ""you""}, {""∅"", ""he""}, {""∆"", ""she""}, {""∇"", ""it""},
        {""∈"", ""we""}, {""∉"", ""they""}, {""∊"", ""me""}, {""∋"", ""him""}, {""∌"", ""her""},
        {""∍"", ""my""}, {""∎"", ""your""}, {""∏"", ""his""}, {""∐"", ""its""},
        {""∑"", ""our""}, {""−"", ""their""}, {""∓"", ""this""}, {""∔"", ""that""},
        {""∕"", ""is""}, {""∖"", ""are""}, {""∗"", ""was""}, {""∘"", ""were""},
        {""∙"", ""be""}, {""√"", ""been""}, {""∛"", ""being""}, {""∜"", ""am""},
        {""∝"", ""have""}, {""∞"", ""has""}, {""∟"", ""had""}, {""∠"", ""do""},
        {""∡"", ""does""}, {""∢"", ""did""}, {""∣"", ""will""}, {""∤"", ""would""},
        {""∥"", ""can""}, {""∦"", ""could""}, {""∧"", ""should""}, {""∨"", ""may""},
        {""∩"", ""might""}, {""∪"", ""must""}, {""∫"", ""shall""},
        {""∬"", ""of""}, {""∭"", ""to""}, {""∮"", ""in""}, {""∯"", ""for""},
        {""∰"", ""on""}, {""∱"", ""with""}, {""∲"", ""at""}, {""∳"", ""by""},
        {""∴"", ""from""}, {""∵"", ""as""}, {""∶"", ""into""}, {""∷"", ""about""},
        {""∸"", ""and""}, {""∹"", ""or""}, {""∺"", ""but""}, {""∻"", ""if""},
        {""∼"", ""then""}, {""∽"", ""so""}, {""∾"", ""because""}, {""∿"", ""when""},
        {""≀"", ""not""}, {""≁"", ""no""}, {""≂"", ""yes""}, {""≃"", ""all""},
        {""≄"", ""more""}, {""≅"", ""some""}, {""≆"", ""any""}, {""≇"", ""each""},
        {""≈"", ""which""}, {""≉"", ""who""}, {""≊"", ""what""}, {""≋"", ""how""},
        {""≌"", ""there""}, {""≍"", ""here""}, {""≎"", ""also""}, {""≏"", ""only""},
        {""≐"", ""new""}, {""≑"", ""now""}, {""≒"", ""very""}, {""≓"", ""just""},
        // Tier 2
        {""⋀"", ""ing""}, {""⋁"", ""ed""}, {""⋂"", ""tion""}, {""⋃"", ""ment""},
        {""⋄"", ""ness""}, {""⋅"", ""able""}, {""⋆"", ""ible""}, {""⋇"", ""ful""},
        {""⋈"", ""less""}, {""⋉"", ""ous""}, {""⋊"", ""ive""}, {""⋋"", ""ly""},
        // Tier 6
        {""⌀"", ""function""}, {""⌁"", ""variable""}, {""⌂"", ""class""}, {""⌃"", ""method""},
        // Tier 7
        {""⨀"", ""artificial intelligence""}, {""⨁"", ""machine learning""},
        {""⨂"", ""deep learning""}, {""⨃"", ""neural network""},
        // Case markers
        {""ˆ"", 
    static void Main(string[] args)
    {
        string outputDir = args.Length > 0 ? args[0] : Directory.GetCurrentDirectory();
        Console.WriteLine(""NSLZ Self-Extracting Archive"");
        Console.WriteLine(""Extracting to: "" + outputDir);

        try
        {
            ExtractFromSelf(outputDir);
            Console.WriteLine(""Extraction complete!"");
        }
        catch (Exception ex)
        {
            Console.WriteLine(""Error: "" + ex.Message);
            Environment.Exit(1);
        }
    }

    static void ExtractFromSelf(string outputDir)
    {
        string exePath = System.Reflection.Assembly.GetExecutingAssembly().Location;
        if (string.IsNullOrEmpty(exePath))
            exePath = Environment.ProcessPath ?? throw new Exception(""Cannot determine executable path"");

        byte[] exeData = File.ReadAllBytes(exePath);

        // Find NSLZ marker at end
        byte[] marker = Encoding.ASCII.GetBytes(""NSLZ_DATA_START"");
        int dataStart = FindMarker(exeData, marker);
        if (dataStart < 0) throw new Exception(""No embedded archive found"");

        dataStart += marker.Length;
        byte[] archiveData = new byte[exeData.Length - dataStart];
        Array.Copy(exeData, dataStart, archiveData, 0, archiveData.Length);

        ExtractArchive(archiveData, outputDir);
    }

    static int FindMarker(byte[] data, byte[] marker)
    {
        for (int i = data.Length - marker.Length - 1024; i < data.Length - marker.Length; i++)
        {
            if (i < 0) i = 0;
            bool found = true;
            for (int j = 0; j < marker.Length; j++)
            {
                if (data[i + j] != marker[j]) { found = false; break; }
            }
            if (found) return i;
        }
        return -1;
    }

    static void ExtractArchive(byte[] data, string outputDir)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        byte[] magic = reader.ReadBytes(4);
        if (Encoding.ASCII.GetString(magic) != ""NSLZ"")
            throw new Exception(""Invalid archive format"");

        reader.ReadUInt16(); // version
        reader.ReadUInt16(); // flags
        uint fileCount = reader.ReadUInt32();

        var entries = new List<(string name, long origSize, long compSize, long offset, uint crc, byte compression, bool isText)>();

        for (int i = 0; i < fileCount; i++)
        {
            ushort nameLen = reader.ReadUInt16();
            string name = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));
            long origSize = reader.ReadInt64();
            long compSize = reader.ReadInt64();
            long offset = reader.ReadInt64();
            uint crc = reader.ReadUInt32();
            byte compression = reader.ReadByte();
            bool isText = reader.ReadBoolean();
            entries.Add((name, origSize, compSize, offset, crc, compression, isText));
        }

        long dataStart = ms.Position;

        if (!Directory.Exists(outputDir))
            Directory.CreateDirectory(outputDir);

        foreach (var entry in entries)
        {
            ms.Position = dataStart + entry.offset;
            byte[] compressed = reader.ReadBytes((int)entry.compSize);
            byte[] original;

            switch (entry.compression)
            {
                case 0: // Store
                    original = compressed;
                    break;
                case 1: // Brotli
                    original = DecompressBrotli(compressed);
                    break;
                case 2: // Semantic
                    original = Encoding.UTF8.GetBytes(DecompressSemantic(Encoding.UTF8.GetString(compressed)));
                    break;
                case 3: // Combined
                    byte[] brotliDecomp = DecompressBrotli(compressed);
                    original = Encoding.UTF8.GetBytes(DecompressSemantic(Encoding.UTF8.GetString(brotliDecomp)));
                    break;
                default:
                    throw new Exception($""Unknown compression: {entry.compression}"");
            }

            string outPath = Path.Combine(outputDir, entry.name);
            File.WriteAllBytes(outPath, original);
            Console.WriteLine($""  Extracted: {entry.name}"");
        }
    }

    static byte[] DecompressBrotli(byte[] data)
    {
        using var input = new MemoryStream(data);
        using var brotli = new BrotliStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        brotli.CopyTo(output);
        return output.ToArray();
    }

    static string DecompressSemantic(string text)
    {
        var result = new StringBuilder(text);
        foreach (var kvp in DecompressDict.OrderByDescending(k => k.Key.Length))
        {
            result.Replace(kvp.Key, kvp.Value);
        }
        return result.ToString();
    }
}
";

    /// <summary>
    /// Create a self-extracting archive
    /// </summary>
    public static void CreateSFX(string outputPath, string[] filePaths)
    {
        // First create regular NSLZ archive in memory
        using var archiveStream = new MemoryStream();
        CreateArchiveToStream(archiveStream, filePaths);
        byte[] archiveData = archiveStream.ToArray();

        // Compile stub to exe
        byte[] stubExe = CompileStub();

        // Combine: stub exe + marker + archive data
        byte[] marker = Encoding.ASCII.GetBytes("NSLZ_DATA_START");

        using var fs = File.Create(outputPath);
        fs.Write(stubExe, 0, stubExe.Length);
        fs.Write(marker, 0, marker.Length);
        fs.Write(archiveData, 0, archiveData.Length);
    }

    /// <summary>
    /// Create archive directly to stream (for SFX embedding)
    /// </summary>
    private static void CreateArchiveToStream(Stream stream, string[] filePaths)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Write header
        writer.Write(Encoding.ASCII.GetBytes("NSLZ"));
        writer.Write((ushort)1); // version
        writer.Write((ushort)NSLZFormat.ArchiveFlags.SemanticCompression);
        writer.Write((uint)filePaths.Length);

        var entries = new List<NSLZFormat.ArchiveEntry>();
        var dataBlocks = new List<byte[]>();

        foreach (var filePath in filePaths)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"File not found: {filePath}");

            var fileName = Path.GetFileName(filePath);
            var originalData = File.ReadAllBytes(filePath);
            var isText = IsTextFile(filePath, originalData);

            byte[] compressedData;
            NSLZFormat.CompressionType compression;

            if (isText)
            {
                var text = Encoding.UTF8.GetString(originalData);
                var semanticCompressed = NSLZCompressor.CompressText(text);
                var semanticBytes = Encoding.UTF8.GetBytes(semanticCompressed);
                compressedData = NSLZCompressor.CompressBinary(semanticBytes);
                compression = NSLZFormat.CompressionType.Combined;
            }
            else
            {
                compressedData = NSLZCompressor.CompressBinary(originalData);
                compression = NSLZFormat.CompressionType.Brotli;
            }

            entries.Add(new NSLZFormat.ArchiveEntry
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

        // Write data blocks
        foreach (var block in dataBlocks)
        {
            writer.Write(block);
        }
    }

    private static void WriteEntry(BinaryWriter writer, NSLZFormat.ArchiveEntry entry)
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

    private static bool IsTextFile(string filePath, byte[] data)
    {
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

        var checkLength = Math.Min(8192, data.Length);
        for (int i = 0; i < checkLength; i++)
        {
            if (data[i] == 0) return false;
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

    /// <summary>
    /// Compile the stub extractor to an executable
    /// Uses runtime compilation via Roslyn
    /// </summary>
    private static byte[] CompileStub()
    {
        // For portability, we'll use a pre-compiled stub approach
        // The stub is embedded as a resource or generated at build time

        // Try to find dotnet CLI to compile
        var tempDir = Path.Combine(Path.GetTempPath(), $"nslz_stub_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            var stubCs = Path.Combine(tempDir, "Program.cs");
            var stubCsproj = Path.Combine(tempDir, "stub.csproj");
            var stubExe = Path.Combine(tempDir, "publish", "stub.exe");

            File.WriteAllText(stubCs, StubSourceTemplate);
            File.WriteAllText(stubCsproj, @"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <PublishSingleFile>true</PublishSingleFile>
    <SelfContained>true</SelfContained>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier>
    <PublishTrimmed>true</PublishTrimmed>
    <TrimMode>link</TrimMode>
    <InvariantGlobalization>true</InvariantGlobalization>
  </PropertyGroup>
</Project>");

            // Run dotnet publish
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = $"publish -c Release -o \"{Path.Combine(tempDir, "publish")}\"",
                WorkingDirectory = tempDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = System.Diagnostics.Process.Start(psi);
            process?.WaitForExit(120000); // 2 minute timeout

            if (process?.ExitCode != 0 || !File.Exists(stubExe))
            {
                // Fallback: create a simple batch-based extractor for Windows
                return CreateBatchExtractor();
            }

            return File.ReadAllBytes(stubExe);
        }
        finally
        {
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    /// <summary>
    /// Fallback: Create a batch-script based self-extractor
    /// This creates a .exe that's actually a renamed .bat with embedded data
    /// </summary>
    private static byte[] CreateBatchExtractor()
    {
        // For a more reliable cross-platform solution,
        // we'll create a minimal stub that requires .NET runtime
        // but is much smaller than a self-contained publish

        var stubCode = @"
using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Collections.Generic;
using System.Linq;

class P {
    static Dictionary<string,string> D = new() {
        {""∀"",""the""},{""∃"",""and""},{""∈"",""to""},{""∋"",""of""},{""⊂"",""a""},
        {""⊃"",""in""},{""∪"",""is""},{""∩"",""that""},{""⊆"",""it""},{""⊇"",""for""},
        {""⌀"",""function""},{""⌁"",""class""},{""⌂"",""public""},{""⌃"",""private""},
        {""⨀"",""artificial intelligence""},{""⨁"",""machine learning""},
    };
    static void Main(string[] a) {
        var o = a.Length > 0 ? a[0] : Directory.GetCurrentDirectory();
        var p = Environment.ProcessPath ?? """";
        var d = File.ReadAllBytes(p);
        var m = Encoding.ASCII.GetBytes(""NSLZ_DATA_START"");
        int s = -1;
        for (int i = d.Length - m.Length - 1024; i < d.Length - m.Length; i++) {
            if (i < 0) i = 0;
            bool f = true;
            for (int j = 0; j < m.Length; j++) if (d[i+j] != m[j]) { f = false; break; }
            if (f) { s = i + m.Length; break; }
        }
        if (s < 0) { Console.WriteLine(""No archive""); return; }
        var ad = new byte[d.Length - s];
        Array.Copy(d, s, ad, 0, ad.Length);
        using var ms = new MemoryStream(ad);
        using var r = new BinaryReader(ms);
        r.ReadBytes(4);r.ReadUInt16();r.ReadUInt16();
        uint n = r.ReadUInt32();
        var e = new List<(string,long,long,long,byte)>();
        for (int i = 0; i < n; i++) {
            var nl = r.ReadUInt16();
            var nm = Encoding.UTF8.GetString(r.ReadBytes(nl));
            var os = r.ReadInt64();var cs = r.ReadInt64();var of = r.ReadInt64();
            r.ReadUInt32();var c = r.ReadByte();r.ReadBoolean();
            e.Add((nm,os,cs,of,c));
        }
        var ds = ms.Position;
        Directory.CreateDirectory(o);
        foreach (var x in e) {
            ms.Position = ds + x.Item4;
            var cd = r.ReadBytes((int)x.Item3);
            byte[] od;
            switch(x.Item5) {
                case 0: od = cd; break;
                case 1: using(var bi=new MemoryStream(cd))using(var br=new BrotliStream(bi,CompressionMode.Decompress))using(var bo=new MemoryStream()){br.CopyTo(bo);od=bo.ToArray();} break;
                case 3: using(var bi=new MemoryStream(cd))using(var br=new BrotliStream(bi,CompressionMode.Decompress))using(var bo=new MemoryStream()){br.CopyTo(bo);var t=new StringBuilder(Encoding.UTF8.GetString(bo.ToArray()));foreach(var k in D)t.Replace(k.Key,k.Value);od=Encoding.UTF8.GetBytes(t.ToString());} break;
                default: od = cd; break;
            }
            File.WriteAllBytes(Path.Combine(o,x.Item1),od);
            Console.WriteLine(""  ""+x.Item1);
        }
        Console.WriteLine(""Done"");
    }
}";

        // Create minimal csproj for framework-dependent (smaller) build
        var tempDir = Path.Combine(Path.GetTempPath(), $"nslz_fstub_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "P.cs"), stubCode);
            File.WriteAllText(Path.Combine(tempDir, "s.csproj"), @"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <PublishSingleFile>true</PublishSingleFile>
    <SelfContained>true</SelfContained>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier>
    <PublishTrimmed>true</PublishTrimmed>
    <TrimMode>full</TrimMode>
    <InvariantGlobalization>true</InvariantGlobalization>
    <EnableCompressionInSingleFile>true</EnableCompressionInSingleFile>
  </PropertyGroup>
</Project>");

            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = $"publish -c Release -o \"{Path.Combine(tempDir, "out")}\"",
                WorkingDirectory = tempDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = System.Diagnostics.Process.Start(psi);
            process?.WaitForExit(120000);

            var exePath = Path.Combine(tempDir, "out", "s.exe");
            if (File.Exists(exePath))
                return File.ReadAllBytes(exePath);

            // Ultimate fallback - return empty (will need external extractor)
            throw new Exception("Could not compile stub extractor. Ensure .NET 8 SDK is installed.");
        }
        finally
        {
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }
}
