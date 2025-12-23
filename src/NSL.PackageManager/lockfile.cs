using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NSL.PackageManager
{
    /// <summary>
    /// Manages the package lock file (nsl-package-lock.json) for deterministic builds
    /// </summary>
    public class LockFile
    {
        private readonly string _filePath;
        private LockFileData _data;

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            PropertyNameCaseInsensitive = true
        };

        /// <summary>Public API</summary>
        public const int CurrentLockFileVersion = 1;

        /// <summary>
        /// Create or load a lock file
        /// </summary>
        public LockFile(string filePath)
        {
            _filePath = filePath;
            _data = Load();
        }

        /// <summary>
        /// All locked packages
        /// </summary>
        public IReadOnlyDictionary<string, LockedPackage> Packages => _data.Packages;

        /// <summary>
        /// Lock file format version
        /// </summary>
        public int Version => _data.LockFileVersion;

        /// <summary>
        /// When the lock file was last modified
        /// </summary>
        public DateTime? LastModified => _data.LastModified;

        /// <summary>
        /// Set or update a package in the lock file
        /// </summary>
        public void SetPackage(string name, string version, Dictionary<string, string> dependencies)
        {
            _data.Packages[name] = new LockedPackage
            {
                Version = version,
                Resolved = $"local://{name}@{version}",
                Integrity = null, // Will be set when downloaded
                Dependencies = dependencies.Count > 0 ? new Dictionary<string, string>(dependencies) : null
            };
            _data.LastModified = DateTime.UtcNow;
        }

        /// <summary>
        /// Set package with integrity hash
        /// </summary>
        public void SetPackage(string name, string version, Dictionary<string, string> dependencies,
            string resolved, string? integrity)
        {
            _data.Packages[name] = new LockedPackage
            {
                Version = version,
                Resolved = resolved,
                Integrity = integrity,
                Dependencies = dependencies.Count > 0 ? new Dictionary<string, string>(dependencies) : null
            };
            _data.LastModified = DateTime.UtcNow;
        }

        /// <summary>
        /// Remove a package from the lock file
        /// </summary>
        public void RemovePackage(string name)
        {
            if (_data.Packages.Remove(name))
            {
                _data.LastModified = DateTime.UtcNow;
            }
        }

        /// <summary>
        /// Check if a package is locked
        /// </summary>
        public bool HasPackage(string name) => _data.Packages.ContainsKey(name);

        /// <summary>
        /// Get locked version for a package
        /// </summary>
        public string? GetLockedVersion(string name)
        {
            return _data.Packages.TryGetValue(name, out var pkg) ? pkg.Version : null;
        }

        /// <summary>
        /// Get a locked package
        /// </summary>
        public LockedPackage? GetPackage(string name)
        {
            return _data.Packages.TryGetValue(name, out var pkg) ? pkg : null;
        }

        /// <summary>
        /// Save the lock file
        /// </summary>
        public void Save()
        {
            var json = JsonSerializer.Serialize(_data, _jsonOptions);
            var directory = Path.GetDirectoryName(_filePath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            File.WriteAllText(_filePath, json);
        }

        /// <summary>
        /// Reload from disk
        /// </summary>
        public void Reload()
        {
            _data = Load();
        }

        /// <summary>
        /// Clear all packages
        /// </summary>
        public void Clear()
        {
            _data.Packages.Clear();
            _data.LastModified = DateTime.UtcNow;
        }

        /// <summary>
        /// Get packages in topological order (dependencies first)
        /// </summary>
        public List<string> GetInstallOrder()
        {
            var result = new List<string>();
            var visited = new HashSet<string>();
            var visiting = new HashSet<string>();

            void Visit(string name)
            {
                if (visited.Contains(name)) return;
                if (visiting.Contains(name))
                {
                    throw new InvalidOperationException($"Circular dependency detected involving {name}");
                }

                visiting.Add(name);

                if (_data.Packages.TryGetValue(name, out var pkg) && pkg.Dependencies != null)
                {
                    foreach (var dep in pkg.Dependencies.Keys)
                    {
                        Visit(dep);
                    }
                }

                visiting.Remove(name);
                visited.Add(name);
                result.Add(name);
            }

            foreach (var name in _data.Packages.Keys)
            {
                Visit(name);
            }

            return result;
        }

        /// <summary>
        /// Validate lock file integrity
        /// </summary>
        public LockFileValidation Validate()
        {
            var issues = new List<string>();

            // Check version
            if (_data.LockFileVersion != CurrentLockFileVersion)
            {
                issues.Add($"Lock file version mismatch. Expected {CurrentLockFileVersion}, got {_data.LockFileVersion}");
            }

            // Check for missing dependencies
            foreach (var (name, pkg) in _data.Packages)
            {
                if (pkg.Dependencies != null)
                {
                    foreach (var dep in pkg.Dependencies.Keys)
                    {
                        if (!_data.Packages.ContainsKey(dep))
                        {
                            issues.Add($"Package {name} depends on {dep} which is not in lock file");
                        }
                    }
                }

                if (string.IsNullOrEmpty(pkg.Version))
                {
                    issues.Add($"Package {name} has no version");
                }
            }

            // Check for circular dependencies
            try
            {
                GetInstallOrder();
            }
            catch (InvalidOperationException ex)
            {
                issues.Add(ex.Message);
            }

            return new LockFileValidation(issues.Count == 0, issues);
        }

        /// <summary>
        /// Compute a hash of the lock file for change detection
        /// </summary>
        public string ComputeHash()
        {
            var json = JsonSerializer.Serialize(_data.Packages, _jsonOptions);
            using var sha = System.Security.Cryptography.SHA256.Create();
            var hash = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(json));
            return Convert.ToHexString(hash).ToLowerInvariant();
        }

        /// <summary>
        /// Check if the lock file differs from manifest dependencies
        /// </summary>
        public LockFileDiff Diff(Dictionary<string, string> manifestDeps)
        {
            var added = new List<string>();
            var removed = new List<string>();
            var updated = new List<string>();

            foreach (var (name, constraint) in manifestDeps)
            {
                if (!_data.Packages.ContainsKey(name))
                {
                    added.Add(name);
                }
                else
                {
                    // Check if locked version satisfies constraint
                    var locked = _data.Packages[name];
                    if (VersionConstraint.TryParse(constraint, out var vc) &&
                        SemanticVersion.TryParse(locked.Version, out var sv))
                    {
                        if (!vc!.IsSatisfiedBy(sv!))
                        {
                            updated.Add(name);
                        }
                    }
                }
            }

            foreach (var name in _data.Packages.Keys)
            {
                if (!manifestDeps.ContainsKey(name))
                {
                    // Check if it's a transitive dependency
                    var isTransitive = _data.Packages.Values.Any(p =>
                        p.Dependencies != null && p.Dependencies.ContainsKey(name));
                    if (!isTransitive)
                    {
                        removed.Add(name);
                    }
                }
            }

            return new LockFileDiff(added, removed, updated);
        }

        private LockFileData Load()
        {
            if (!File.Exists(_filePath))
            {
                return new LockFileData
                {
                    LockFileVersion = CurrentLockFileVersion,
                    Packages = new Dictionary<string, LockedPackage>()
                };
            }

            var json = File.ReadAllText(_filePath);
            return JsonSerializer.Deserialize<LockFileData>(json, _jsonOptions)
                ?? new LockFileData
                {
                    LockFileVersion = CurrentLockFileVersion,
                    Packages = new Dictionary<string, LockedPackage>()
                };
        }
    }

    /// <summary>
    /// Lock file data structure
    /// </summary>
    internal class LockFileData
    {
        /// <summary>API member</summary>
        [JsonPropertyName("lockfileVersion")]
        public int LockFileVersion { get; set; } = 1;

        /// <summary>API member</summary>
        [JsonPropertyName("lastModified")]
        public DateTime? LastModified { get; set; }

        /// <summary>API member</summary>
        [JsonPropertyName("packages")]
        public Dictionary<string, LockedPackage> Packages { get; set; } = new();
    }

    /// <summary>
    /// A package entry in the lock file
    /// </summary>
    public class LockedPackage
    {
        /// <summary>API member</summary>
        [JsonPropertyName("version")]
        public string Version { get; set; } = "";

        /// <summary>API member</summary>
        [JsonPropertyName("resolved")]
        public string? Resolved { get; set; }

        /// <summary>API member</summary>
        [JsonPropertyName("integrity")]
        public string? Integrity { get; set; }

        /// <summary>API member</summary>
        [JsonPropertyName("dependencies")]
        public Dictionary<string, string>? Dependencies { get; set; }

        /// <summary>Public API</summary>
        public override string ToString() => $"{Version} ({Resolved})";
    }

    /// <summary>
    /// Result of lock file validation
    /// </summary>
    public class LockFileValidation
    {
        /// <summary>Public API</summary>
        public bool IsValid { get; }
        /// <summary>Public API</summary>
        public List<string> Issues { get; }

        /// <summary>Public API</summary>
        public LockFileValidation(bool isValid, List<string> issues)
        {
            IsValid = isValid;
            Issues = issues;
        }
    }

    /// <summary>
    /// Difference between lock file and manifest
    /// </summary>
    public class LockFileDiff
    {
        /// <summary>Public API</summary>
        public List<string> Added { get; }
        /// <summary>Public API</summary>
        public List<string> Removed { get; }
        /// <summary>Public API</summary>
        public List<string> Updated { get; }
        /// <summary>Public API</summary>
        public bool HasChanges => Added.Count > 0 || Removed.Count > 0 || Updated.Count > 0;

        /// <summary>Public API</summary>
        public LockFileDiff(List<string> added, List<string> removed, List<string> updated)
        {
            Added = added;
            Removed = removed;
            Updated = updated;
        }
    }
}