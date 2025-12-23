using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NSL.PackageManager
{
    /// <summary>
    /// Local file-based package registry that works offline.
    /// Stores packages in ~/.nsl/registry/ directory.
    /// </summary>
    public class LocalPackageRegistry : IPackageRegistry
    {
        private readonly string _registryPath;
        private readonly string _packagesPath;
        private readonly string _indexPath;
        private readonly object _indexLock = new();

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            WriteIndented = true
        };

        /// <summary>Public API</summary>
        public LocalPackageRegistry(string? registryPath = null)
        {
            _registryPath = registryPath ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "nsl", "registry"
            );
            _packagesPath = Path.Combine(_registryPath, "packages");
            _indexPath = Path.Combine(_registryPath, "index.json");

            Directory.CreateDirectory(_registryPath);
            Directory.CreateDirectory(_packagesPath);

            EnsureIndexExists();
        }

        /// <summary>
        /// Get all available versions for a package
        /// </summary>
        public Task<List<SemanticVersion>> GetVersionsAsync(string packageName)
        {
            var index = LoadIndex();
            if (!index.Packages.TryGetValue(NormalizePackageName(packageName), out var pkg))
                return Task.FromResult(new List<SemanticVersion>());

            var versions = pkg.Versions
                .Select(v => SemanticVersion.TryParse(v.Key, out var sv) ? sv : null)
                .Where(v => v != null)
                .Cast<SemanticVersion>()
                .OrderByDescending(v => v)
                .ToList();

            return Task.FromResult(versions);
        }

        /// <summary>
        /// Download a package
        /// </summary>
        public Task<byte[]> DownloadAsync(string packageName, string version)
        {
            var normalizedName = NormalizePackageName(packageName);
            var packageFile = Path.Combine(_packagesPath, normalizedName, $"{version}.nslpkg");

            if (!File.Exists(packageFile))
            {
                throw new PackageNotFoundException($"Package not found: {packageName}@{version}");
            }

            return File.ReadAllBytesAsync(packageFile);
        }

        /// <summary>
        /// Publish a package to the local registry
        /// </summary>
        public async Task PublishAsync(string packagePath, string? authToken = null)
        {
            if (!File.Exists(packagePath))
                throw new FileNotFoundException("Package file not found", packagePath);

            // Extract manifest from package
            var manifest = await ExtractManifestFromPackageAsync(packagePath);
            var normalizedName = NormalizePackageName(manifest.Name);

            // Create package directory
            var packageDir = Path.Combine(_packagesPath, normalizedName);
            Directory.CreateDirectory(packageDir);

            // Copy package file
            var targetPath = Path.Combine(packageDir, $"{manifest.Version}.nslpkg");
            File.Copy(packagePath, targetPath, overwrite: true);

            // Compute integrity hash
            var packageBytes = await File.ReadAllBytesAsync(targetPath);
            var integrity = ComputeIntegrity(packageBytes);

            // Update index
            lock (_indexLock)
            {
                var index = LoadIndex();

                if (!index.Packages.TryGetValue(normalizedName, out var pkg))
                {
                    pkg = new LocalPackageEntry
                    {
                        Name = manifest.Name,
                        Description = manifest.Description,
                        Keywords = manifest.Keywords,
                        Homepage = manifest.Homepage,
                        Repository = manifest.Repository,
                        Created = DateTime.UtcNow.ToString("o")
                    };
                    index.Packages[normalizedName] = pkg;
                }

                pkg.Versions[manifest.Version] = new LocalVersionEntry
                {
                    Version = manifest.Version,
                    License = manifest.License,
                    Authors = manifest.Authors,
                    Dependencies = manifest.Dependencies,
                    Integrity = integrity,
                    PublishedAt = DateTime.UtcNow.ToString("o")
                };
                pkg.LatestVersion = GetLatestVersionString(pkg.Versions.Keys);
                pkg.Modified = DateTime.UtcNow.ToString("o");
                pkg.Downloads = pkg.Downloads; // Keep existing count

                SaveIndex(index);
            }
        }

        /// <summary>
        /// Search for packages
        /// </summary>
        public Task<List<PackageSearchResult>> SearchAsync(string query, int limit = 20)
        {
            var index = LoadIndex();
            var queryLower = query.ToLowerInvariant();

            var results = index.Packages.Values
                .Where(p =>
                    p.Name.ToLowerInvariant().Contains(queryLower) ||
                    (p.Description?.ToLowerInvariant().Contains(queryLower) ?? false) ||
                    (p.Keywords?.Any(k => k.ToLowerInvariant().Contains(queryLower)) ?? false))
                .OrderByDescending(p => p.Downloads)
                .ThenBy(p => p.Name)
                .Take(limit)
                .Select(p => new PackageSearchResult
                {
                    Name = p.Name,
                    Description = p.Description,
                    Version = p.LatestVersion ?? "",
                    Keywords = p.Keywords ?? new List<string>(),
                    Downloads = p.Downloads,
                    Score = CalculateSearchScore(p, queryLower)
                })
                .ToList();

            return Task.FromResult(results);
        }

        /// <summary>
        /// Get detailed package info
        /// </summary>
        public Task<PackageInfo?> GetInfoAsync(string packageName)
        {
            var index = LoadIndex();
            var normalizedName = NormalizePackageName(packageName);

            if (!index.Packages.TryGetValue(normalizedName, out var pkg))
                return Task.FromResult<PackageInfo?>(null);

            var latestVersion = pkg.LatestVersion ?? "";
            var versionInfo = !string.IsNullOrEmpty(latestVersion) && pkg.Versions.TryGetValue(latestVersion, out var v)
                ? v : pkg.Versions.Values.FirstOrDefault();

            var info = new PackageInfo
            {
                Name = pkg.Name,
                Description = pkg.Description,
                LatestVersion = latestVersion,
                Homepage = pkg.Homepage,
                Repository = pkg.Repository,
                License = versionInfo?.License,
                Authors = versionInfo?.Authors ?? new List<string>(),
                Keywords = pkg.Keywords ?? new List<string>(),
                Dependencies = versionInfo?.Dependencies ?? new Dictionary<string, string>(),
                VersionCount = pkg.Versions.Count,
                Versions = pkg.Versions.Keys.ToList(),
                Created = pkg.Created,
                Modified = pkg.Modified,
                Downloads = pkg.Downloads
            };

            return Task.FromResult<PackageInfo?>(info);
        }

        /// <summary>
        /// Check if a package exists
        /// </summary>
        public Task<bool> ExistsAsync(string packageName)
        {
            var index = LoadIndex();
            var normalizedName = NormalizePackageName(packageName);
            return Task.FromResult(index.Packages.ContainsKey(normalizedName));
        }

        /// <summary>
        /// Get the latest version of a package
        /// </summary>
        public async Task<SemanticVersion?> GetLatestVersionAsync(string packageName)
        {
            var versions = await GetVersionsAsync(packageName);
            return versions.FirstOrDefault();
        }

        /// <summary>
        /// Increment download count for a package
        /// </summary>
        public void IncrementDownloads(string packageName)
        {
            lock (_indexLock)
            {
                var index = LoadIndex();
                var normalizedName = NormalizePackageName(packageName);

                if (index.Packages.TryGetValue(normalizedName, out var pkg))
                {
                    pkg.Downloads++;
                    SaveIndex(index);
                }
            }
        }

        /// <summary>
        /// List all packages in the registry
        /// </summary>
        public Task<List<PackageSearchResult>> ListAllAsync(int limit = 100)
        {
            var index = LoadIndex();

            var results = index.Packages.Values
                .OrderByDescending(p => p.Downloads)
                .ThenBy(p => p.Name)
                .Take(limit)
                .Select(p => new PackageSearchResult
                {
                    Name = p.Name,
                    Description = p.Description,
                    Version = p.LatestVersion ?? "",
                    Keywords = p.Keywords ?? new List<string>(),
                    Downloads = p.Downloads
                })
                .ToList();

            return Task.FromResult(results);
        }

        /// <summary>
        /// Delete a package version from the registry
        /// </summary>
        public Task UnpublishAsync(string packageName, string? version = null)
        {
            lock (_indexLock)
            {
                var index = LoadIndex();
                var normalizedName = NormalizePackageName(packageName);

                if (!index.Packages.TryGetValue(normalizedName, out var pkg))
                    throw new PackageNotFoundException($"Package not found: {packageName}");

                if (version == null)
                {
                    // Delete all versions
                    var packageDir = Path.Combine(_packagesPath, normalizedName);
                    if (Directory.Exists(packageDir))
                        Directory.Delete(packageDir, recursive: true);

                    index.Packages.Remove(normalizedName);
                }
                else
                {
                    // Delete specific version
                    if (!pkg.Versions.ContainsKey(version))
                        throw new PackageNotFoundException($"Version not found: {packageName}@{version}");

                    var packageFile = Path.Combine(_packagesPath, normalizedName, $"{version}.nslpkg");
                    if (File.Exists(packageFile))
                        File.Delete(packageFile);

                    pkg.Versions.Remove(version);

                    if (pkg.Versions.Count == 0)
                    {
                        index.Packages.Remove(normalizedName);
                        var packageDir = Path.Combine(_packagesPath, normalizedName);
                        if (Directory.Exists(packageDir))
                            Directory.Delete(packageDir, recursive: true);
                    }
                    else
                    {
                        pkg.LatestVersion = GetLatestVersionString(pkg.Versions.Keys);
                    }
                }

                SaveIndex(index);
            }

            return Task.CompletedTask;
        }

        #region Private Methods

        private void EnsureIndexExists()
        {
            if (!File.Exists(_indexPath))
            {
                SaveIndex(new RegistryIndex());
            }
        }

        private RegistryIndex LoadIndex()
        {
            try
            {
                var json = File.ReadAllText(_indexPath);
                return JsonSerializer.Deserialize<RegistryIndex>(json, _jsonOptions) ?? new RegistryIndex();
            }
            catch
            {
                return new RegistryIndex();
            }
        }

        private void SaveIndex(RegistryIndex index)
        {
            var json = JsonSerializer.Serialize(index, _jsonOptions);
            File.WriteAllText(_indexPath, json);
        }

        private static string NormalizePackageName(string name)
        {
            return name.ToLowerInvariant().Replace('/', '-');
        }

        private static string ComputeIntegrity(byte[] data)
        {
            using var sha512 = SHA512.Create();
            var hash = sha512.ComputeHash(data);
            return $"sha512-{Convert.ToBase64String(hash)}";
        }

        private static string? GetLatestVersionString(IEnumerable<string> versions)
        {
            return versions
                .Select(v => SemanticVersion.TryParse(v, out var sv) ? sv : null)
                .Where(v => v != null)
                .Cast<SemanticVersion>()
                .OrderByDescending(v => v)
                .FirstOrDefault()?.ToString();
        }

        private static double CalculateSearchScore(LocalPackageEntry pkg, string query)
        {
            double score = 0;

            if (pkg.Name.ToLowerInvariant() == query) score += 100;
            else if (pkg.Name.ToLowerInvariant().StartsWith(query)) score += 50;
            else if (pkg.Name.ToLowerInvariant().Contains(query)) score += 25;

            if (pkg.Keywords?.Any(k => k.ToLowerInvariant() == query) ?? false) score += 30;

            score += Math.Log10(pkg.Downloads + 1) * 5;

            return score;
        }

        private static async Task<PackageManifest> ExtractManifestFromPackageAsync(string packagePath)
        {
            using var archive = System.IO.Compression.ZipFile.OpenRead(packagePath);
            var manifestEntry = archive.GetEntry("nsl-package.json")
                ?? throw new InvalidOperationException("Package does not contain nsl-package.json");

            using var stream = manifestEntry.Open();
            using var reader = new StreamReader(stream);
            var json = await reader.ReadToEndAsync();

            return JsonSerializer.Deserialize<PackageManifest>(json, _jsonOptions)
                ?? throw new InvalidOperationException("Invalid package manifest");
        }

        #endregion

        #region Index DTOs

        private class RegistryIndex
        {
            /// <summary>API member</summary>
            [JsonPropertyName("version")]
            public int Version { get; set; } = 1;

            /// <summary>API member</summary>
            [JsonPropertyName("packages")]
            public Dictionary<string, LocalPackageEntry> Packages { get; set; } = new();
        }

        private class LocalPackageEntry
        {
            /// <summary>API member</summary>
            [JsonPropertyName("name")]
            public string Name { get; set; } = "";

            /// <summary>API member</summary>
            [JsonPropertyName("description")]
            public string? Description { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("latestVersion")]
            public string? LatestVersion { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("homepage")]
            public string? Homepage { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("repository")]
            public string? Repository { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("keywords")]
            public List<string>? Keywords { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("versions")]
            public Dictionary<string, LocalVersionEntry> Versions { get; set; } = new();

            /// <summary>API member</summary>
            [JsonPropertyName("created")]
            public string? Created { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("modified")]
            public string? Modified { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("downloads")]
            public long Downloads { get; set; }
        }

        private class LocalVersionEntry
        {
            /// <summary>API member</summary>
            [JsonPropertyName("version")]
            public string Version { get; set; } = "";

            /// <summary>API member</summary>
            [JsonPropertyName("license")]
            public string? License { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("authors")]
            public List<string> Authors { get; set; } = new();

            /// <summary>API member</summary>
            [JsonPropertyName("dependencies")]
            public Dictionary<string, string> Dependencies { get; set; } = new();

            /// <summary>API member</summary>
            [JsonPropertyName("integrity")]
            public string? Integrity { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("publishedAt")]
            public string? PublishedAt { get; set; }
        }

        #endregion
    }

    /// <summary>
    /// Interface for package registries (both local and remote)
    /// </summary>
    public interface IPackageRegistry
    {
        /// <summary>API member</summary>
        Task<List<SemanticVersion>> GetVersionsAsync(string packageName);
        /// <summary>API member</summary>
        Task<byte[]> DownloadAsync(string packageName, string version);
        /// <summary>API member</summary>
        Task PublishAsync(string packagePath, string? authToken = null);
        /// <summary>API member</summary>
        Task<List<PackageSearchResult>> SearchAsync(string query, int limit = 20);
        /// <summary>API member</summary>
        Task<PackageInfo?> GetInfoAsync(string packageName);
        /// <summary>API member</summary>
        Task<bool> ExistsAsync(string packageName);
        /// <summary>API member</summary>
        Task<SemanticVersion?> GetLatestVersionAsync(string packageName);
    }
}