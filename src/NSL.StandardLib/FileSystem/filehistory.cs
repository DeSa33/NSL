using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace NSL.StandardLib.FileSystem
{
    /// <summary>
    /// File Edit History System for NSL.
    /// 
    /// Provides reversible file edits by capturing pre-edit state before mutations.
    /// Supports both session-only (RAM) and persistent storage modes.
    /// 
    /// Key Invariants:
    /// - Pre-edit state saved BEFORE mutation (not after)
    /// - Ring buffer with configurable capacity (default 10)
    /// - Restore is always explicit, never automatic
    /// - Atomic writes prevent file corruption
    /// </summary>
    public class FileHistory
    {
        private static FileHistory? _instance;
        private static readonly object _lock = new();

        // Per-file history storage: normalized path -> ring buffer of entries
        private readonly Dictionary<string, FileHistoryRing> _histories = new();
        
        // Global configuration
        private FileHistoryConfig _config = new();

        // Singleton access
        /// <summary>Public API</summary>
        public static FileHistory Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_lock)
                    {
                        _instance ??= new FileHistory();
                    }
                }
                return _instance;
            }
        }

        private FileHistory() { }

        #region Configuration

        /// <summary>
        /// Get current configuration
        /// </summary>
        public FileHistoryConfig Config => _config;

        /// <summary>
        /// Update global configuration
        /// </summary>
        public void Configure(FileHistoryConfig config)
        {
            _config = config;
        }

        /// <summary>
        /// Set default capacity for all new files
        /// </summary>
        public void SetDefaultCapacity(int capacity)
        {
            _config.DefaultCapacity = Math.Max(1, capacity);
        }

        /// <summary>
        /// Enable or disable history globally
        /// </summary>
        public void SetEnabled(bool enabled)
        {
            _config.Enabled = enabled;
        }

        /// <summary>
        /// Set persistent mode (saves to disk)
        /// </summary>
        public void SetPersistent(bool persistent)
        {
            _config.Persistent = persistent;
        }

        #endregion

        #region Core Operations

        /// <summary>
        /// Save pre-edit state for a file. Call this BEFORE any mutation.
        /// Uses hybrid storage: snapshots every N edits, diffs in between.
        /// </summary>
        /// <param name="path">File path</param>
        /// <param name="operation">Name of the operation (e.g., "write", "replace", "insertAt")</param>
        /// <param name="summary">Optional summary of the edit</param>
        /// <param name="reason">Optional AI reasoning for why this edit was made</param>
        /// <param name="agent">Optional agent/tool name that made the edit</param>
        /// <param name="metadata">Optional additional context</param>
        /// <returns>True if state was saved, false if file doesn't exist or history disabled</returns>
        public bool SavePreEditState(string path, string operation, string? summary = null, 
            string? reason = null, string? agent = null, Dictionary<string, string>? metadata = null)
        {
            if (!_config.Enabled) return false;
            
            var normalizedPath = NormalizePath(path);
            
            // Only save if file exists (can't save pre-edit state of non-existent file)
            if (!File.Exists(normalizedPath)) return false;

            try
            {
                // Read current content
                var content = File.ReadAllText(normalizedPath);
                var contentBytes = Encoding.UTF8.GetByteCount(content);
                
                // Skip if file is too large (memory protection, default 10MB)
                if (contentBytes > _config.MaxFileSizeBytes)
                {
                    return false;
                }
                
                // For binary files, always use snapshot (no diffing)
                bool isBinary = IsBinaryContent(content);
                
                // Normalize line endings for consistent diffing (text files only)
                var normalizedContent = isBinary ? content : NormalizeLineEndings(content);
                var hash = ComputeHash(normalizedContent);

                // Get or create ring buffer for this file
                var ring = GetOrCreateRing(normalizedPath);
                var entries = ring.GetAll();
                
                // Determine if this should be a snapshot or diff (hybrid strategy)
                // Binary files always use snapshots
                bool useSnapshot = isBinary || ShouldUseSnapshot(entries);
                
                FileHistoryEntry entry;
                
                if (useSnapshot || _config.StorageStrategy == "snapshot")
                {
                    // Store full snapshot
                    entry = new FileHistoryEntry
                    {
                        Id = Guid.NewGuid().ToString("N")[..8],
                        Timestamp = DateTime.UtcNow,
                        Path = normalizedPath,
                        Content = normalizedContent,
                        Hash = hash,
                        SizeBytes = contentBytes,
                        StoredBytes = contentBytes,
                        Operation = operation,
                        Summary = summary,
                        IsSnapshot = true,
                        BaseHash = null,
                        Reason = reason,
                        Agent = agent,
                        Metadata = metadata
                    };
                }
                else
                {
                    // Find the most recent snapshot to diff against
                    var baseSnapshot = FindNearestSnapshot(entries);
                    if (baseSnapshot == null)
                    {
                        // No snapshot found, create one
                        entry = new FileHistoryEntry
                        {
                            Id = Guid.NewGuid().ToString("N")[..8],
                            Timestamp = DateTime.UtcNow,
                            Path = normalizedPath,
                            Content = normalizedContent,
                            Hash = hash,
                            SizeBytes = contentBytes,
                            StoredBytes = contentBytes,
                            Operation = operation,
                            Summary = summary,
                            IsSnapshot = true,
                            BaseHash = null,
                            Reason = reason,
                            Agent = agent,
                            Metadata = metadata
                        };
                    }
                    else
                    {
                        // Compute diff from base snapshot
                        var diff = ComputeDiff(baseSnapshot.Content, normalizedContent);
                        var diffBytes = Encoding.UTF8.GetByteCount(diff);
                        
                        // Only use diff if it's smaller than the full content
                        if (diffBytes < contentBytes * 0.8) // 80% threshold
                        {
                            entry = new FileHistoryEntry
                            {
                                Id = Guid.NewGuid().ToString("N")[..8],
                                Timestamp = DateTime.UtcNow,
                                Path = normalizedPath,
                                Content = diff,
                                Hash = hash,
                                SizeBytes = contentBytes,
                                StoredBytes = diffBytes,
                                Operation = operation,
                                Summary = summary,
                                IsSnapshot = false,
                                BaseHash = baseSnapshot.Hash,
                                Reason = reason,
                                Agent = agent,
                                Metadata = metadata
                            };
                        }
                        else
                        {
                            // Diff is too large, store snapshot instead
                            entry = new FileHistoryEntry
                            {
                                Id = Guid.NewGuid().ToString("N")[..8],
                                Timestamp = DateTime.UtcNow,
                                Path = normalizedPath,
                                Content = normalizedContent,
                                Hash = hash,
                                SizeBytes = contentBytes,
                                StoredBytes = contentBytes,
                                Operation = operation,
                                Summary = summary,
                                IsSnapshot = true,
                                BaseHash = null,
                                Reason = reason,
                                Agent = agent,
                                Metadata = metadata
                            };
                        }
                    }
                }

                ring.Add(entry);

                // Save to persistent storage if enabled
                if (_config.Persistent)
                {
                    SavePersistent(normalizedPath, ring);
                }

                return true;
            }
            catch
            {
                return false;
            }
        }
        
        /// <summary>
        /// Determine if we should use a snapshot based on hybrid strategy
        /// </summary>
        private bool ShouldUseSnapshot(List<FileHistoryEntry> entries)
        {
            if (_config.StorageStrategy != "hybrid") 
                return _config.StorageStrategy == "snapshot";
            
            // Count entries since last snapshot
            int countSinceSnapshot = 0;
            foreach (var entry in entries.AsEnumerable().Reverse())
            {
                if (entry.IsSnapshot) break;
                countSinceSnapshot++;
            }
            
            // Snapshot every N edits
            return countSinceSnapshot >= _config.SnapshotInterval - 1;
        }
        
        /// <summary>
        /// Find the nearest snapshot in history (searching backwards)
        /// </summary>
        private FileHistoryEntry? FindNearestSnapshot(List<FileHistoryEntry> entries)
        {
            for (int i = entries.Count - 1; i >= 0; i--)
            {
                if (entries[i].IsSnapshot)
                    return entries[i];
            }
            return null;
        }

        /// <summary>
        /// Get history entries for a file (newest first)
        /// </summary>
        public List<FileHistoryEntry> GetHistory(string path)
        {
            var normalizedPath = NormalizePath(path);
            
            // Load from persistent storage if needed
            if (_config.Persistent && !_histories.ContainsKey(normalizedPath))
            {
                LoadPersistent(normalizedPath);
            }

            if (_histories.TryGetValue(normalizedPath, out var ring))
            {
                return ring.GetAll().OrderByDescending(e => e.Timestamp).ToList();
            }

            return new List<FileHistoryEntry>();
        }

        /// <summary>
        /// Get history info for a file
        /// </summary>
        public FileHistoryInfo GetHistoryInfo(string path)
        {
            var normalizedPath = NormalizePath(path);
            var entries = GetHistory(normalizedPath);

            return new FileHistoryInfo
            {
                Path = normalizedPath,
                Count = entries.Count,
                Capacity = GetCapacity(normalizedPath),
                Persistent = _config.Persistent,
                Enabled = _config.Enabled,
                LastSavedAt = entries.FirstOrDefault()?.Timestamp,
                TotalSizeBytes = entries.Sum(e => e.SizeBytes)
            };
        }

        /// <summary>
        /// Restore file to a specific history entry by index (0 = most recent)
        /// Handles both snapshots and diffs (hybrid storage)
        /// </summary>
        /// <param name="path">File path</param>
        /// <param name="index">History index (0 = most recent pre-edit state)</param>
        /// <returns>True if restored successfully</returns>
        public bool Restore(string path, int index = 0)
        {
            var normalizedPath = NormalizePath(path);
            var entries = GetHistory(normalizedPath);

            if (index < 0 || index >= entries.Count)
            {
                return false;
            }

            var entry = entries[index];

            try
            {
                // Reconstruct content - either directly from snapshot or by applying diff
                string content;
                
                if (entry.IsSnapshot)
                {
                    content = entry.Content;
                }
                else
                {
                    // Need to find base snapshot and apply diff
                    content = ReconstructFromDiff(entries, entry);
                    if (content == null)
                    {
                        return false;
                    }
                }
                
                // Atomic write: write to temp, then replace
                var tempPath = normalizedPath + $".nsl_restore_{Guid.NewGuid():N}";
                File.WriteAllText(tempPath, content);

                // Verify content
                var written = File.ReadAllText(tempPath);
                if (ComputeHash(written) != entry.Hash)
                {
                    File.Delete(tempPath);
                    return false;
                }

                // Atomic replace
                File.Move(tempPath, normalizedPath, overwrite: true);
                return true;
            }
            catch
            {
                return false;
            }
        }
        
        /// <summary>
        /// Reconstruct content from a diff entry by finding base snapshot and applying diff
        /// </summary>
        private string? ReconstructFromDiff(List<FileHistoryEntry> entries, FileHistoryEntry diffEntry)
        {
            if (diffEntry.IsSnapshot)
                return diffEntry.Content;
                
            // Find the base snapshot
            FileHistoryEntry? baseSnapshot = null;
            foreach (var entry in entries)
            {
                if (entry.IsSnapshot && entry.Hash == diffEntry.BaseHash)
                {
                    baseSnapshot = entry;
                    break;
                }
            }
            
            if (baseSnapshot == null)
            {
                // Base snapshot not found - try to find any snapshot and reconstruct
                // This is a fallback for when the exact base is evicted
                foreach (var entry in entries)
                {
                    if (entry.IsSnapshot)
                    {
                        baseSnapshot = entry;
                        break;
                    }
                }
                
                if (baseSnapshot == null)
                    return null;
            }
            
            // Apply diff to reconstruct
            return ApplyDiff(baseSnapshot.Content, diffEntry.Content);
        }

        /// <summary>
        /// Restore file to the most recent pre-edit state
        /// </summary>
        public bool RestoreLatest(string path) => Restore(path, 0);

        /// <summary>
        /// Clear history for a specific file
        /// </summary>
        public void ClearHistory(string path)
        {
            var normalizedPath = NormalizePath(path);
            _histories.Remove(normalizedPath);

            if (_config.Persistent)
            {
                DeletePersistent(normalizedPath);
            }
        }

        /// <summary>
        /// Clear all history (session reset)
        /// </summary>
        public void ClearAll()
        {
            _histories.Clear();
        }

        #endregion

        #region Atomic Write Support

        /// <summary>
        /// Perform an atomic write with automatic history capture.
        /// This is the recommended way to write files in NSL.
        /// </summary>
        public bool AtomicWrite(string path, string content, string operation = "write", string? summary = null)
        {
            var normalizedPath = NormalizePath(path);

            // Save pre-edit state if file exists
            SavePreEditState(normalizedPath, operation, summary);

            try
            {
                // Ensure directory exists
                var dir = Path.GetDirectoryName(normalizedPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                // Write to temp file
                var tempPath = normalizedPath + $".nsl_tmp_{Guid.NewGuid():N}";
                File.WriteAllText(tempPath, content);

                // Atomic replace
                File.Move(tempPath, normalizedPath, overwrite: true);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Perform an atomic file operation with history capture.
        /// Use this for complex transformations.
        /// </summary>
        public bool AtomicTransform(string path, Func<string, string> transform, string operation, string? summary = null)
        {
            var normalizedPath = NormalizePath(path);

            if (!File.Exists(normalizedPath))
            {
                return false;
            }

            // Save pre-edit state
            SavePreEditState(normalizedPath, operation, summary);

            try
            {
                var content = File.ReadAllText(normalizedPath);
                var newContent = transform(content);

                // Write to temp
                var tempPath = normalizedPath + $".nsl_tmp_{Guid.NewGuid():N}";
                File.WriteAllText(tempPath, newContent);

                // Atomic replace
                File.Move(tempPath, normalizedPath, overwrite: true);
                return true;
            }
            catch
            {
                return false;
            }
        }

        #endregion

        #region Preview/Dry Run Mode

        /// <summary>
        /// Preview what a write would look like without actually writing.
        /// Returns a diff showing what would change.
        /// </summary>
        public PreviewResult Preview(string path, string newContent)
        {
            var normalizedPath = NormalizePath(path);
            var result = new PreviewResult { Path = normalizedPath };
            
            if (!File.Exists(normalizedPath))
            {
                result.IsNewFile = true;
                result.NewContent = newContent;
                result.NewLines = newContent.Split('\n').Length;
                result.Changes = new List<DiffChange> { 
                    new DiffChange { Type = "add", Content = newContent, LineCount = result.NewLines } 
                };
                return result;
            }
            
            var oldContent = File.ReadAllText(normalizedPath);
            var oldNormalized = NormalizeLineEndings(oldContent);
            var newNormalized = NormalizeLineEndings(newContent);
            
            result.OldContent = oldContent;
            result.NewContent = newContent;
            result.OldLines = oldNormalized.Split('\n').Length;
            result.NewLines = newNormalized.Split('\n').Length;
            
            // Compute diff
            var oldLines = oldNormalized.Split('\n');
            var newLines = newNormalized.Split('\n');
            var changes = new List<DiffChange>();
            
            int oldIdx = 0, newIdx = 0;
            while (oldIdx < oldLines.Length || newIdx < newLines.Length)
            {
                if (oldIdx >= oldLines.Length)
                {
                    changes.Add(new DiffChange { Type = "add", Line = newIdx + 1, Content = newLines[newIdx] });
                    newIdx++;
                }
                else if (newIdx >= newLines.Length)
                {
                    changes.Add(new DiffChange { Type = "remove", Line = oldIdx + 1, Content = oldLines[oldIdx] });
                    oldIdx++;
                }
                else if (oldLines[oldIdx] == newLines[newIdx])
                {
                    oldIdx++;
                    newIdx++;
                }
                else
                {
                    changes.Add(new DiffChange { Type = "remove", Line = oldIdx + 1, Content = oldLines[oldIdx] });
                    changes.Add(new DiffChange { Type = "add", Line = newIdx + 1, Content = newLines[newIdx] });
                    oldIdx++;
                    newIdx++;
                }
            }
            
            result.Changes = changes;
            result.LinesAdded = changes.Count(c => c.Type == "add");
            result.LinesRemoved = changes.Count(c => c.Type == "remove");
            
            return result;
        }

        #endregion

        #region Thrashing Detection

        /// <summary>
        /// Detect if a file is "thrashing" - oscillating between similar states.
        /// Returns a warning if the file has been restored or reverted multiple times.
        /// </summary>
        public ThrashingResult DetectThrashing(string path)
        {
            var normalizedPath = NormalizePath(path);
            var entries = GetHistory(normalizedPath);
            
            var result = new ThrashingResult { Path = normalizedPath };
            
            if (entries.Count < 3)
            {
                result.IsThrashing = false;
                return result;
            }
            
            // Check for duplicate hashes (same content appearing multiple times)
            var hashCounts = new Dictionary<string, int>();
            foreach (var entry in entries)
            {
                if (!hashCounts.ContainsKey(entry.Hash))
                    hashCounts[entry.Hash] = 0;
                hashCounts[entry.Hash]++;
            }
            
            // If any hash appears 3+ times, that's thrashing
            var repeatedHashes = hashCounts.Where(kv => kv.Value >= 3).ToList();
            if (repeatedHashes.Any())
            {
                result.IsThrashing = true;
                result.Message = $"File has oscillated to the same state {repeatedHashes.Max(kv => kv.Value)} times";
                result.RepeatedStates = repeatedHashes.Count;
            }
            
            // Check for rapid back-and-forth (A->B->A pattern)
            for (int i = 0; i < entries.Count - 2; i++)
            {
                if (entries[i].Hash == entries[i + 2].Hash && entries[i].Hash != entries[i + 1].Hash)
                {
                    result.IsThrashing = true;
                    result.Message = "File is oscillating between two states (A→B→A pattern detected)";
                    result.OscillationPattern = true;
                    break;
                }
            }
            
            // Suggest stable state if thrashing
            if (result.IsThrashing)
            {
                var mostCommonHash = hashCounts.OrderByDescending(kv => kv.Value).First().Key;
                var stableEntry = entries.FirstOrDefault(e => e.Hash == mostCommonHash);
                if (stableEntry != null)
                {
                    result.SuggestedStableIndex = entries.IndexOf(stableEntry);
                    result.SuggestedStableTimestamp = stableEntry.Timestamp;
                }
            }
            
            return result;
        }

        #endregion

        #region Per-File Configuration

        /// <summary>
        /// Configure history settings for a specific file or pattern
        /// </summary>
        public void ConfigureFile(string path, FileHistoryFileConfig config)
        {
            var normalizedPath = NormalizePath(path);
            var ring = GetOrCreateRing(normalizedPath);
            
            if (config.Capacity.HasValue)
                ring.SetCapacity(config.Capacity.Value);
            if (config.Enabled.HasValue)
                ring.Enabled = config.Enabled.Value;
            if (!string.IsNullOrEmpty(config.Strategy))
                ring.Strategy = config.Strategy;
        }

        /// <summary>
        /// Get default config based on file extension
        /// </summary>
        public FileHistoryFileConfig GetDefaultConfigForExtension(string extension)
        {
            extension = extension.ToLowerInvariant();
            
            // Large files / binaries - smaller capacity, snapshot only
            if (extension is ".exe" or ".dll" or ".pdf" or ".zip" or ".tar" or ".gz")
            {
                return new FileHistoryFileConfig { Capacity = 3, Strategy = "snapshot", Enabled = false };
            }
            
            // Config files - more history, important to track
            if (extension is ".json" or ".yaml" or ".yml" or ".xml" or ".config" or ".ini")
            {
                return new FileHistoryFileConfig { Capacity = 20, Strategy = "hybrid" };
            }
            
            // Source code - standard hybrid
            if (extension is ".cs" or ".py" or ".js" or ".ts" or ".java" or ".go" or ".rs" or ".cpp" or ".c" or ".h")
            {
                return new FileHistoryFileConfig { Capacity = 10, Strategy = "hybrid" };
            }
            
            // Log files - minimal history
            if (extension is ".log" or ".tmp" or ".cache")
            {
                return new FileHistoryFileConfig { Capacity = 2, Strategy = "snapshot" };
            }
            
            // Default
            return new FileHistoryFileConfig { Capacity = _config.DefaultCapacity, Strategy = _config.StorageStrategy };
        }

        /// <summary>
        /// Set capacity for a specific file
        /// </summary>
        public void SetCapacity(string path, int capacity)
        {
            var normalizedPath = NormalizePath(path);
            var ring = GetOrCreateRing(normalizedPath);
            ring.SetCapacity(Math.Max(1, capacity));
        }

        /// <summary>
        /// Get capacity for a specific file
        /// </summary>
        public int GetCapacity(string path)
        {
            var normalizedPath = NormalizePath(path);
            if (_histories.TryGetValue(normalizedPath, out var ring))
            {
                return ring.Capacity;
            }
            return _config.DefaultCapacity;
        }

        /// <summary>
        /// Disable history for a specific file
        /// </summary>
        public void DisableForFile(string path)
        {
            var normalizedPath = NormalizePath(path);
            var ring = GetOrCreateRing(normalizedPath);
            ring.Enabled = false;
        }

        /// <summary>
        /// Enable history for a specific file
        /// </summary>
        public void EnableForFile(string path)
        {
            var normalizedPath = NormalizePath(path);
            var ring = GetOrCreateRing(normalizedPath);
            ring.Enabled = true;
        }

        #endregion

        #region Private Helpers

        private FileHistoryRing GetOrCreateRing(string normalizedPath)
        {
            if (!_histories.TryGetValue(normalizedPath, out var ring))
            {
                ring = new FileHistoryRing(_config.DefaultCapacity);
                _histories[normalizedPath] = ring;
            }
            return ring;
        }

        private static string NormalizePath(string path)
        {
            // Absolute path, normalized separators, lowercase for Windows case-insensitivity
            var fullPath = Path.GetFullPath(path).Replace('/', '\\');
            if (OperatingSystem.IsWindows())
            {
                fullPath = fullPath.ToLowerInvariant();
            }
            return fullPath;
        }

        private static string ComputeHash(string content)
        {
            var bytes = Encoding.UTF8.GetBytes(content);
            var hash = SHA256.HashData(bytes);
            // Store full hash for integrity, display truncated
            return Convert.ToHexString(hash).ToLowerInvariant();
        }
        
        /// <summary>
        /// Truncate hash for display purposes
        /// </summary>
        private static string TruncateHash(string hash) => hash.Length > 16 ? hash[..16] : hash;
        
        /// <summary>
        /// Detect if content is likely binary (non-text)
        /// </summary>
        private static bool IsBinaryContent(string content)
        {
            // Check for null bytes or high ratio of non-printable chars
            int nonPrintable = 0;
            int checkLen = Math.Min(content.Length, 8000);
            for (int i = 0; i < checkLen; i++)
            {
                char c = content[i];
                if (c == '\0') return true;
                if (c < 32 && c != '\t' && c != '\n' && c != '\r') nonPrintable++;
            }
            return checkLen > 0 && (double)nonPrintable / checkLen > 0.1;
        }
        
        /// <summary>
        /// Normalize line endings to LF for consistent diffing
        /// </summary>
        private static string NormalizeLineEndings(string content)
        {
            return content.Replace("\r\n", "\n").Replace("\r", "\n");
        }
        
        /// <summary>
        /// Compute a line-based diff between two strings.
        /// Format: Each line is prefixed with + (add), - (remove), or = (keep)
        /// </summary>
        private static string ComputeDiff(string baseContent, string newContent)
        {
            var baseLines = baseContent.Split('\n');
            var newLines = newContent.Split('\n');
            
            var sb = new StringBuilder();
            
            // Simple LCS-based diff algorithm
            var lcs = ComputeLCS(baseLines, newLines);
            
            int baseIdx = 0, newIdx = 0, lcsIdx = 0;
            
            while (baseIdx < baseLines.Length || newIdx < newLines.Length)
            {
                if (lcsIdx < lcs.Count && baseIdx < baseLines.Length && newIdx < newLines.Length &&
                    baseLines[baseIdx] == lcs[lcsIdx] && newLines[newIdx] == lcs[lcsIdx])
                {
                    // Line is in both - keep
                    sb.AppendLine($"={baseLines[baseIdx]}");
                    baseIdx++;
                    newIdx++;
                    lcsIdx++;
                }
                else if (baseIdx < baseLines.Length && 
                         (lcsIdx >= lcs.Count || baseLines[baseIdx] != lcs[lcsIdx]))
                {
                    // Line removed from base
                    sb.AppendLine($"-{baseLines[baseIdx]}");
                    baseIdx++;
                }
                else if (newIdx < newLines.Length)
                {
                    // Line added in new
                    sb.AppendLine($"+{newLines[newIdx]}");
                    newIdx++;
                }
            }
            
            return sb.ToString();
        }
        
        /// <summary>
        /// Apply a diff to reconstruct content
        /// </summary>
        private static string ApplyDiff(string baseContent, string diff)
        {
            var baseLines = baseContent.Split('\n').ToList();
            var diffLines = diff.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            
            var result = new List<string>();
            int baseIdx = 0;
            
            foreach (var line in diffLines)
            {
                if (string.IsNullOrEmpty(line)) continue;
                
                var op = line[0];
                var content = line.Length > 1 ? line[1..] : "";
                
                switch (op)
                {
                    case '=':
                        // Keep line from base
                        result.Add(content);
                        baseIdx++;
                        break;
                    case '-':
                        // Skip line from base (removed)
                        baseIdx++;
                        break;
                    case '+':
                        // Add new line
                        result.Add(content);
                        break;
                    default:
                        // Unknown op, treat as content
                        result.Add(line);
                        break;
                }
            }
            
            return string.Join('\n', result);
        }
        
        /// <summary>
        /// Compute Longest Common Subsequence of lines
        /// </summary>
        private static List<string> ComputeLCS(string[] a, string[] b)
        {
            int m = a.Length, n = b.Length;
            var dp = new int[m + 1, n + 1];
            
            // Build LCS length table
            for (int i = 1; i <= m; i++)
            {
                for (int j = 1; j <= n; j++)
                {
                    if (a[i - 1] == b[j - 1])
                        dp[i, j] = dp[i - 1, j - 1] + 1;
                    else
                        dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
                }
            }
            
            // Backtrack to find LCS
            var lcs = new List<string>();
            int x = m, y = n;
            while (x > 0 && y > 0)
            {
                if (a[x - 1] == b[y - 1])
                {
                    lcs.Add(a[x - 1]);
                    x--;
                    y--;
                }
                else if (dp[x - 1, y] > dp[x, y - 1])
                {
                    x--;
                }
                else
                {
                    y--;
                }
            }
            
            lcs.Reverse();
            return lcs;
        }

        #endregion

        #region Persistent Storage

        private string GetHistoryDir()
        {
            var nslDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".nsl", "history"
            );
            Directory.CreateDirectory(nslDir);
            return nslDir;
        }

        private string GetHistoryFilePath(string normalizedPath)
        {
            var hash = ComputeHash(normalizedPath);
            return Path.Combine(GetHistoryDir(), $"{hash}.json");
        }

        private void SavePersistent(string normalizedPath, FileHistoryRing ring)
        {
            try
            {
                var historyFile = GetHistoryFilePath(normalizedPath);
                var lockFile = historyFile + ".lock";
                var data = new PersistentHistoryData
                {
                    Path = normalizedPath,
                    Capacity = ring.Capacity,
                    Enabled = ring.Enabled,
                    Entries = ring.GetAll()
                };
                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
                
                // Use file lock for multi-process safety
                using (var lockStream = new FileStream(lockFile, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
                {
                    // Atomic write: temp file then rename
                    var tempFile = historyFile + $".tmp_{Guid.NewGuid():N}";
                    File.WriteAllText(tempFile, json);
                    File.Move(tempFile, historyFile, overwrite: true);
                }
                
                // Clean up lock file (best effort)
                try { File.Delete(lockFile); } catch { }
            }
            catch
            {
                // Persistent storage failure - edit still proceeds safely
                // History just won't be persisted this time
            }
        }

        private void LoadPersistent(string normalizedPath)
        {
            try
            {
                var historyFile = GetHistoryFilePath(normalizedPath);
                if (File.Exists(historyFile))
                {
                    var lockFile = historyFile + ".lock";
                    
                    // Use file lock for multi-process safety
                    using (var lockStream = new FileStream(lockFile, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
                    {
                        var json = File.ReadAllText(historyFile);
                        var data = JsonSerializer.Deserialize<PersistentHistoryData>(json);
                        if (data != null && data.Path == normalizedPath)
                        {
                            var ring = new FileHistoryRing(data.Capacity)
                            {
                                Enabled = data.Enabled
                            };
                            foreach (var entry in data.Entries.OrderBy(e => e.Timestamp))
                            {
                                ring.Add(entry);
                            }
                            _histories[normalizedPath] = ring;
                        }
                    }
                    
                    // Clean up lock file (best effort)
                    try { File.Delete(lockFile); } catch { }
                }
            }
            catch
            {
                // Load failure - continue with empty history
            }
        }

        private void DeletePersistent(string normalizedPath)
        {
            try
            {
                var historyFile = GetHistoryFilePath(normalizedPath);
                var lockFile = historyFile + ".lock";
                
                using (var lockStream = new FileStream(lockFile, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
                {
                    if (File.Exists(historyFile))
                    {
                        File.Delete(historyFile);
                    }
                }
                
                try { File.Delete(lockFile); } catch { }
            }
            catch { }
        }

        #endregion
    }

    /// <summary>
    /// Ring buffer for file history entries
    /// </summary>
    public class FileHistoryRing
    {
        private readonly List<FileHistoryEntry> _entries = new();
        private int _capacity;

        /// <summary>Public API</summary>
        public int Capacity
        {
            get => _capacity;
            private set => _capacity = value;
        }

        /// <summary>Public API</summary>
        public bool Enabled { get; set; } = true;
        /// <summary>Public API</summary>
        public string Strategy { get; set; } = "hybrid";  // "snapshot", "diff", "hybrid"

        /// <summary>Public API</summary>
        public FileHistoryRing(int capacity)
        {
            _capacity = Math.Max(1, capacity);
        }

        /// <summary>Public API</summary>
        public void Add(FileHistoryEntry entry)
        {
            if (!Enabled) return;

            // Ring buffer: remove oldest if at capacity
            while (_entries.Count >= _capacity)
            {
                _entries.RemoveAt(0);
            }
            _entries.Add(entry);
        }

        /// <summary>Public API</summary>
        public List<FileHistoryEntry> GetAll()
        {
            return new List<FileHistoryEntry>(_entries);
        }

        /// <summary>Public API</summary>
        public void SetCapacity(int newCapacity)
        {
            _capacity = Math.Max(1, newCapacity);
            // Trim if needed
            while (_entries.Count > _capacity)
            {
                _entries.RemoveAt(0);
            }
        }

        /// <summary>Public API</summary>
        public void Clear()
        {
            _entries.Clear();
        }
    }

    /// <summary>
    /// A single history entry - can be snapshot or diff (hybrid storage)
    /// </summary>
    public class FileHistoryEntry
    {
        /// <summary>Public API</summary>
        public string Id { get; set; } = "";
        /// <summary>Public API</summary>
        public DateTime Timestamp { get; set; }
        /// <summary>Public API</summary>
        public string Path { get; set; } = "";
        /// <summary>Public API</summary>
        public string Content { get; set; } = "";  // Full content for snapshots, diff for diffs
        /// <summary>Public API</summary>
        public string Hash { get; set; } = "";     // Hash of the FULL content (for verification)
        /// <summary>Public API</summary>
        public long SizeBytes { get; set; }         // Size of original content
        /// <summary>Public API</summary>
        public long StoredBytes { get; set; }       // Actual stored bytes (smaller for diffs)
        /// <summary>Public API</summary>
        public string Operation { get; set; } = "";
        /// <summary>Public API</summary>
        public string? Summary { get; set; }
        /// <summary>Public API</summary>
        public bool IsSnapshot { get; set; } = true; // true = full snapshot, false = diff
        /// <summary>Public API</summary>
        public string? BaseHash { get; set; }        // Hash of base snapshot (for diffs)
        
        // Phase 3: Annotated edits - attach AI reasoning
        /// <summary>Public API</summary>
        public string? Reason { get; set; }          // Why this edit was made (AI reasoning)
        /// <summary>Public API</summary>
        public string? Agent { get; set; }           // Which agent/tool made the edit
        /// <summary>Public API</summary>
        public Dictionary<string, string>? Metadata { get; set; }  // Additional context

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var type = IsSnapshot ? "snap" : "diff";
            var reason = !string.IsNullOrEmpty(Reason) ? $" [{Reason}]" : "";
            return $"[{Id}] {Timestamp:yyyy-MM-dd HH:mm:ss} - {Operation}{reason} ({SizeBytes} bytes, {type})";
        }
    }
    
    /// <summary>
    /// Preview result for dry run mode
    /// </summary>
    public class PreviewResult
    {
        /// <summary>Public API</summary>
        public string Path { get; set; } = "";
        /// <summary>Public API</summary>
        public bool IsNewFile { get; set; }
        /// <summary>Public API</summary>
        public string? OldContent { get; set; }
        /// <summary>Public API</summary>
        public string? NewContent { get; set; }
        /// <summary>Public API</summary>
        public int OldLines { get; set; }
        /// <summary>Public API</summary>
        public int NewLines { get; set; }
        /// <summary>Public API</summary>
        public int LinesAdded { get; set; }
        /// <summary>Public API</summary>
        public int LinesRemoved { get; set; }
        /// <summary>Public API</summary>
        public List<DiffChange> Changes { get; set; } = new();
        
        /// <summary>Public API</summary>
        public string GetSummary()
        {
            if (IsNewFile) return $"New file: {NewLines} lines";
            return $"+{LinesAdded} -{LinesRemoved} lines ({Changes.Count} changes)";
        }
    }
    
    /// <summary>
    /// A single diff change
    /// </summary>
    public class DiffChange
    {
        /// <summary>Public API</summary>
        public string Type { get; set; } = "";  // "add", "remove"
        /// <summary>Public API</summary>
        public int Line { get; set; }
        /// <summary>Public API</summary>
        public string Content { get; set; } = "";
        /// <summary>Public API</summary>
        public int LineCount { get; set; } = 1;
    }
    
    /// <summary>
    /// Thrashing detection result
    /// </summary>
    public class ThrashingResult
    {
        /// <summary>Public API</summary>
        public string Path { get; set; } = "";
        /// <summary>Public API</summary>
        public bool IsThrashing { get; set; }
        /// <summary>Public API</summary>
        public string? Message { get; set; }
        /// <summary>Public API</summary>
        public int RepeatedStates { get; set; }
        /// <summary>Public API</summary>
        public bool OscillationPattern { get; set; }
        /// <summary>Public API</summary>
        public int? SuggestedStableIndex { get; set; }
        /// <summary>Public API</summary>
        public DateTime? SuggestedStableTimestamp { get; set; }
    }
    
    /// <summary>
    /// Per-file configuration
    /// </summary>
    public class FileHistoryFileConfig
    {
        /// <summary>Public API</summary>
        public int? Capacity { get; set; }
        /// <summary>Public API</summary>
        public bool? Enabled { get; set; }
        /// <summary>Public API</summary>
        public string? Strategy { get; set; }  // "snapshot", "diff", "hybrid"
    }

    /// <summary>
    /// History info summary
    /// </summary>
    public class FileHistoryInfo
    {
        /// <summary>Public API</summary>
        public string Path { get; set; } = "";
        /// <summary>Public API</summary>
        public int Count { get; set; }
        /// <summary>Public API</summary>
        public int Capacity { get; set; }
        /// <summary>Public API</summary>
        public bool Persistent { get; set; }
        /// <summary>Public API</summary>
        public bool Enabled { get; set; }
        /// <summary>Public API</summary>
        public DateTime? LastSavedAt { get; set; }
        /// <summary>Public API</summary>
        public long TotalSizeBytes { get; set; }
    }

    /// <summary>
    /// Global configuration for file history
    /// </summary>
    public class FileHistoryConfig
    {
        /// <summary>Public API</summary>
        public bool Enabled { get; set; } = true;
        /// <summary>Public API</summary>
        public int DefaultCapacity { get; set; } = 10;
        /// <summary>Public API</summary>
        public bool Persistent { get; set; } = true;
        /// <summary>Public API</summary>
        public string StorageStrategy { get; set; } = "hybrid"; // "snapshot", "diff", "hybrid"
        /// <summary>Public API</summary>
        public int SnapshotInterval { get; set; } = 5;  // For hybrid: snapshot every N edits
        /// <summary>Public API</summary>
        public long MaxFileSizeBytes { get; set; } = 10 * 1024 * 1024;  // 10MB default cap
        /// <summary>Public API</summary>
        public long MaxTotalMemoryBytes { get; set; } = 100 * 1024 * 1024;  // 100MB total session memory
    }

    /// <summary>
    /// Data structure for persistent storage
    /// </summary>
    internal class PersistentHistoryData
    {
        /// <summary>Public API</summary>
        public string Path { get; set; } = "";
        /// <summary>Public API</summary>
        public int Capacity { get; set; }
        /// <summary>Public API</summary>
        public bool Enabled { get; set; }
        /// <summary>Public API</summary>
        public List<FileHistoryEntry> Entries { get; set; } = new();
    }
}