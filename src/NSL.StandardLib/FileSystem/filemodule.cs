using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NSL.StandardLib.FileSystem
{
    /// <summary>
    /// File system operations for NSL.
    ///
    /// Usage in NSL:
    /// <code>
    /// import fs
    ///
    /// // Read file
    /// let content = fs.read("data.txt")
    ///
    /// // Write file
    /// fs.write("output.txt", "Hello, World!")
    ///
    /// // Append to file
    /// fs.append("log.txt", "New entry\n")
    ///
    /// // Check existence
    /// if fs.exists("config.json") {
    ///     let config = json.load("config.json")
    /// }
    ///
    /// // List directory
    /// for file in fs.list("./data") {
    ///     println(file)
    /// }
    /// </code>
    /// </summary>
    public static class FileModule
    {
        #region Reading

        /// <summary>
        /// Read entire file as text
        /// </summary>
        public static string Read(string path)
        {
            return File.ReadAllText(path);
        }

        /// <summary>
        /// Read file asynchronously
        /// </summary>
        public static async Task<string> ReadAsync(string path)
        {
            return await File.ReadAllTextAsync(path);
        }

        /// <summary>
        /// Read file as bytes
        /// </summary>
        public static byte[] ReadBytes(string path)
        {
            return File.ReadAllBytes(path);
        }

        /// <summary>
        /// Read file as bytes asynchronously
        /// </summary>
        public static async Task<byte[]> ReadBytesAsync(string path)
        {
            return await File.ReadAllBytesAsync(path);
        }

        /// <summary>
        /// Read file as lines
        /// </summary>
        public static string[] ReadLines(string path)
        {
            return File.ReadAllLines(path);
        }

        /// <summary>
        /// Read file as lines asynchronously
        /// </summary>
        public static async Task<string[]> ReadLinesAsync(string path)
        {
            return await File.ReadAllLinesAsync(path);
        }

        /// <summary>
        /// Stream lines from a file (memory efficient for large files)
        /// </summary>
        public static IEnumerable<string> StreamLines(string path)
        {
            return File.ReadLines(path);
        }

        /// <summary>
        /// Read a portion of a file
        /// </summary>
        public static string ReadPart(string path, long offset, int length)
        {
            using var stream = File.OpenRead(path);
            stream.Seek(offset, SeekOrigin.Begin);
            var buffer = new byte[length];
            var bytesRead = stream.Read(buffer, 0, length);
            return Encoding.UTF8.GetString(buffer, 0, bytesRead);
        }

        #endregion

        #region Writing

        /// <summary>
        /// Write text to file (overwrites if exists)
        /// Automatically captures pre-edit state for history.
        /// </summary>
        public static void Write(string path, string content)
        {
            // Use atomic write with history capture
            FileHistory.Instance.AtomicWrite(path, content, "write");
        }

        /// <summary>
        /// Write text asynchronously
        /// </summary>
        public static async Task WriteAsync(string path, string content)
        {
            EnsureDirectory(path);
            await File.WriteAllTextAsync(path, content);
        }

        /// <summary>
        /// Write bytes to file
        /// </summary>
        public static void WriteBytes(string path, byte[] data)
        {
            EnsureDirectory(path);
            File.WriteAllBytes(path, data);
        }

        /// <summary>
        /// Write bytes asynchronously
        /// </summary>
        public static async Task WriteBytesAsync(string path, byte[] data)
        {
            EnsureDirectory(path);
            await File.WriteAllBytesAsync(path, data);
        }

        /// <summary>
        /// Write lines to file
        /// </summary>
        public static void WriteLines(string path, IEnumerable<string> lines)
        {
            EnsureDirectory(path);
            File.WriteAllLines(path, lines);
        }

        /// <summary>
        /// Append text to file
        /// Automatically captures pre-edit state for history.
        /// </summary>
        public static void Append(string path, string content)
        {
            FileHistory.Instance.SavePreEditState(path, "append");
            EnsureDirectory(path);
            File.AppendAllText(path, content);
        }

        /// <summary>
        /// Append text asynchronously
        /// </summary>
        public static async Task AppendAsync(string path, string content)
        {
            EnsureDirectory(path);
            await File.AppendAllTextAsync(path, content);
        }

        /// <summary>
        /// Append lines to file
        /// </summary>
        public static void AppendLines(string path, IEnumerable<string> lines)
        {
            EnsureDirectory(path);
            File.AppendAllLines(path, lines);
        }

        #endregion

        #region File Operations

        /// <summary>
        /// Check if file exists
        /// </summary>
        public static bool Exists(string path)
        {
            return File.Exists(path);
        }

        /// <summary>
        /// Check if path is a directory
        /// </summary>
        public static bool IsDirectory(string path)
        {
            return Directory.Exists(path);
        }

        /// <summary>
        /// Check if path is a file
        /// </summary>
        public static bool IsFile(string path)
        {
            return File.Exists(path);
        }

        /// <summary>
        /// Delete a file
        /// Automatically captures pre-edit state for history.
        /// </summary>
        public static void Delete(string path)
        {
            FileHistory.Instance.SavePreEditState(path, "delete");
            if (File.Exists(path))
                File.Delete(path);
        }

        /// <summary>
        /// Copy a file
        /// Captures destination pre-edit state if overwriting.
        /// </summary>
        public static void Copy(string source, string destination, bool overwrite = false)
        {
            if (overwrite)
                FileHistory.Instance.SavePreEditState(destination, "copy", $"from {source}");
            EnsureDirectory(destination);
            File.Copy(source, destination, overwrite);
        }

        /// <summary>
        /// Move/rename a file
        /// Captures both source and destination pre-edit states.
        /// </summary>
        public static void Move(string source, string destination, bool overwrite = false)
        {
            FileHistory.Instance.SavePreEditState(source, "move", $"to {destination}");
            if (overwrite)
                FileHistory.Instance.SavePreEditState(destination, "move", $"overwritten by {source}");
            EnsureDirectory(destination);
            File.Move(source, destination, overwrite);
        }

        /// <summary>
        /// Rename a file
        /// </summary>
        public static void Rename(string path, string newName)
        {
            var directory = Path.GetDirectoryName(path) ?? "";
            var newPath = Path.Combine(directory, newName);
            File.Move(path, newPath);
        }

        /// <summary>
        /// Get file info
        /// </summary>
        public static FileStats Stats(string path)
        {
            var info = new FileInfo(path);
            return new FileStats
            {
                Path = path,
                Name = info.Name,
                Extension = info.Extension,
                Size = info.Exists ? info.Length : 0,
                Created = info.CreationTimeUtc,
                Modified = info.LastWriteTimeUtc,
                Accessed = info.LastAccessTimeUtc,
                IsReadOnly = info.IsReadOnly,
                Exists = info.Exists
            };
        }

        /// <summary>
        /// Get file size in bytes
        /// </summary>
        public static long Size(string path)
        {
            return new FileInfo(path).Length;
        }

        /// <summary>
        /// Get file extension
        /// </summary>
        public static string Extension(string path)
        {
            return Path.GetExtension(path);
        }

        /// <summary>
        /// Get file name without extension
        /// </summary>
        public static string BaseName(string path)
        {
            return Path.GetFileNameWithoutExtension(path);
        }

        /// <summary>
        /// Get file name with extension
        /// </summary>
        public static string FileName(string path)
        {
            return Path.GetFileName(path);
        }

        /// <summary>
        /// Get directory name
        /// </summary>
        public static string? DirName(string path)
        {
            return Path.GetDirectoryName(path);
        }

        /// <summary>
        /// Join path components
        /// </summary>
        public static string Join(params string[] parts)
        {
            return Path.Combine(parts);
        }

        /// <summary>
        /// Resolve to absolute path
        /// </summary>
        public static string Resolve(string path)
        {
            return Path.GetFullPath(path);
        }

        /// <summary>
        /// Compute hash of a file
        /// </summary>
        public static string Hash(string path, string algorithm = "sha256")
        {
            using var stream = File.OpenRead(path);
            using var hasher = algorithm.ToLower() switch
            {
                "md5" => MD5.Create() as HashAlgorithm,
                "sha1" => SHA1.Create(),
                "sha256" => SHA256.Create(),
                "sha384" => SHA384.Create(),
                "sha512" => SHA512.Create(),
                _ => SHA256.Create()
            };

            var hash = hasher.ComputeHash(stream);
            return Convert.ToHexString(hash).ToLowerInvariant();
        }

        #endregion

        #region Directory Operations

        /// <summary>
        /// List files in a directory
        /// </summary>
        public static string[] List(string path, string pattern = "*", bool recursive = false)
        {
            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            return Directory.GetFiles(path, pattern, searchOption);
        }

        /// <summary>
        /// List directories in a directory
        /// </summary>
        public static string[] ListDirs(string path, string pattern = "*", bool recursive = false)
        {
            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            return Directory.GetDirectories(path, pattern, searchOption);
        }

        /// <summary>
        /// List all entries (files and directories) in a directory
        /// </summary>
        public static string[] ListAll(string path, string pattern = "*", bool recursive = false)
        {
            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            return Directory.GetFileSystemEntries(path, pattern, searchOption);
        }

        /// <summary>
        /// Create a directory
        /// </summary>
        public static void MkDir(string path, bool recursive = true)
        {
            if (recursive)
                Directory.CreateDirectory(path);
            else
                Directory.CreateDirectory(path);
        }

        /// <summary>
        /// Remove a directory
        /// </summary>
        public static void RmDir(string path, bool recursive = false)
        {
            if (Directory.Exists(path))
                Directory.Delete(path, recursive);
        }

        /// <summary>
        /// Copy a directory
        /// </summary>
        public static void CopyDir(string source, string destination, bool recursive = true)
        {
            Directory.CreateDirectory(destination);

            foreach (var file in Directory.GetFiles(source))
            {
                var destFile = Path.Combine(destination, Path.GetFileName(file));
                File.Copy(file, destFile, true);
            }

            if (recursive)
            {
                foreach (var dir in Directory.GetDirectories(source))
                {
                    var destDir = Path.Combine(destination, Path.GetFileName(dir));
                    CopyDir(dir, destDir, true);
                }
            }
        }

        /// <summary>
        /// Get current working directory
        /// </summary>
        public static string Cwd()
        {
            return Directory.GetCurrentDirectory();
        }

        /// <summary>
        /// Change current working directory
        /// </summary>
        public static void ChDir(string path)
        {
            Directory.SetCurrentDirectory(path);
        }

        /// <summary>
        /// Get temporary directory path
        /// </summary>
        public static string TempDir()
        {
            return Path.GetTempPath();
        }

        /// <summary>
        /// Create a temporary file and return its path
        /// </summary>
        public static string TempFile(string? extension = null)
        {
            var tempPath = Path.GetTempFileName();
            if (extension != null)
            {
                var newPath = Path.ChangeExtension(tempPath, extension);
                File.Move(tempPath, newPath);
                return newPath;
            }
            return tempPath;
        }

        #endregion

        #region Watching

        /// <summary>
        /// Watch a directory for changes
        /// </summary>
        public static FileSystemWatcher Watch(string path, string pattern = "*.*", Action<FileSystemEventArgs>? onChange = null)
        {
            var watcher = new FileSystemWatcher(path, pattern)
            {
                NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.DirectoryName,
                EnableRaisingEvents = true
            };

            if (onChange != null)
            {
                watcher.Changed += (_, e) => onChange(e);
                watcher.Created += (_, e) => onChange(e);
                watcher.Deleted += (_, e) => onChange(e);
                watcher.Renamed += (_, e) => onChange(e);
            }

            return watcher;
        }

        #endregion

        #region Helpers

        private static void EnsureDirectory(string filePath)
        {
            var dir = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
        }

        #endregion
    }

    /// <summary>
    /// File statistics
    /// </summary>
    public class FileStats
    {
        /// <summary>Full path</summary>
        public string Path { get; init; } = "";

        /// <summary>File name</summary>
        public string Name { get; init; } = "";

        /// <summary>File extension</summary>
        public string Extension { get; init; } = "";

        /// <summary>File size in bytes</summary>
        public long Size { get; init; }

        /// <summary>Creation time (UTC)</summary>
        public DateTime Created { get; init; }

        /// <summary>Last modified time (UTC)</summary>
        public DateTime Modified { get; init; }

        /// <summary>Last access time (UTC)</summary>
        public DateTime Accessed { get; init; }

        /// <summary>Whether file is read-only</summary>
        public bool IsReadOnly { get; init; }

        /// <summary>Whether file exists</summary>
        public bool Exists { get; init; }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"{Name} ({Size} bytes, modified {Modified:yyyy-MM-dd HH:mm:ss})";
        }
    }
}