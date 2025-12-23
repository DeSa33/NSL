using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NSL.PackageManager
{
    /// <summary>
    /// Client for communicating with remote NSL package registries
    /// </summary>
    public class PackageRegistry : IPackageRegistry
    {
        private readonly string _baseUrl;
        private readonly HttpClient _httpClient;
        private readonly Dictionary<string, PackageMetadataCache> _metadataCache = new();
        private readonly TimeSpan _cacheExpiration = TimeSpan.FromMinutes(5);

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        /// <summary>Public API</summary>
        public PackageRegistry(string baseUrl, HttpClient httpClient)
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _httpClient = httpClient;
            _httpClient.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json")
            );
            _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("nslpm/1.0");
        }

        /// <summary>
        /// Get all available versions for a package
        /// </summary>
        public async Task<List<SemanticVersion>> GetVersionsAsync(string packageName)
        {
            var metadata = await GetPackageMetadataAsync(packageName);
            if (metadata == null)
                return new List<SemanticVersion>();

            return metadata.Versions
                .Keys
                .Select(v => SemanticVersion.TryParse(v, out var sv) ? sv : null)
                .Where(v => v != null)
                .Cast<SemanticVersion>()
                .OrderByDescending(v => v)
                .ToList();
        }

        /// <summary>
        /// Download a package tarball
        /// </summary>
        public async Task<byte[]> DownloadAsync(string packageName, string version)
        {
            var metadata = await GetPackageMetadataAsync(packageName)
                ?? throw new PackageNotFoundException($"Package not found: {packageName}");

            if (!metadata.Versions.TryGetValue(version, out var versionInfo))
            {
                throw new PackageNotFoundException($"Version {version} not found for {packageName}");
            }

            var tarballUrl = versionInfo.Dist?.Tarball
                ?? $"{_baseUrl}/packages/{Uri.EscapeDataString(packageName)}/{version}/download";

            var response = await _httpClient.GetAsync(tarballUrl);
            response.EnsureSuccessStatusCode();

            var bytes = await response.Content.ReadAsByteArrayAsync();

            // Verify integrity if provided
            if (!string.IsNullOrEmpty(versionInfo.Dist?.Integrity))
            {
                var actualHash = ComputeIntegrity(bytes, versionInfo.Dist.Integrity);
                if (actualHash != versionInfo.Dist.Integrity)
                {
                    throw new PackageIntegrityException(
                        $"Integrity check failed for {packageName}@{version}. " +
                        $"Expected: {versionInfo.Dist.Integrity}, Got: {actualHash}"
                    );
                }
            }

            return bytes;
        }

        /// <summary>
        /// Publish a package to the registry
        /// </summary>
        public async Task PublishAsync(string packagePath, string? authToken = null)
        {
            if (!File.Exists(packagePath))
                throw new FileNotFoundException("Package file not found", packagePath);

            var packageBytes = await File.ReadAllBytesAsync(packagePath);
            var content = new MultipartFormDataContent();
            content.Add(new ByteArrayContent(packageBytes), "package", Path.GetFileName(packagePath));

            var request = new HttpRequestMessage(HttpMethod.Put, $"{_baseUrl}/publish")
            {
                Content = content
            };

            if (!string.IsNullOrEmpty(authToken))
            {
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", authToken);
            }

            var response = await _httpClient.SendAsync(request);

            if (!response.IsSuccessStatusCode)
            {
                var errorBody = await response.Content.ReadAsStringAsync();
                throw new PublishException($"Failed to publish package: {response.StatusCode} - {errorBody}");
            }
        }

        /// <summary>
        /// Search for packages
        /// </summary>
        public async Task<List<PackageSearchResult>> SearchAsync(string query, int limit = 20)
        {
            try
            {
                var url = $"{_baseUrl}/search?q={Uri.EscapeDataString(query)}&limit={limit}";
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                var searchResponse = JsonSerializer.Deserialize<SearchResponse>(json, _jsonOptions);

                return searchResponse?.Results ?? new List<PackageSearchResult>();
            }
            catch (HttpRequestException)
            {
                // Return empty results if registry is unreachable
                return new List<PackageSearchResult>();
            }
        }

        /// <summary>
        /// Get detailed package info
        /// </summary>
        public async Task<PackageInfo?> GetInfoAsync(string packageName)
        {
            var metadata = await GetPackageMetadataAsync(packageName);
            if (metadata == null)
                return null;

            var latestVersion = metadata.DistTags?.GetValueOrDefault("latest") ?? "";
            var versionInfo = !string.IsNullOrEmpty(latestVersion) && metadata.Versions.TryGetValue(latestVersion, out var v)
                ? v : metadata.Versions.Values.FirstOrDefault();

            return new PackageInfo
            {
                Name = metadata.Name,
                Description = metadata.Description,
                LatestVersion = latestVersion,
                Homepage = metadata.Homepage,
                Repository = metadata.Repository,
                License = versionInfo?.License,
                Authors = versionInfo?.Authors ?? new List<string>(),
                Keywords = metadata.Keywords ?? new List<string>(),
                Dependencies = versionInfo?.Dependencies ?? new Dictionary<string, string>(),
                VersionCount = metadata.Versions.Count,
                Versions = metadata.Versions.Keys.ToList(),
                Created = metadata.Time?.GetValueOrDefault("created"),
                Modified = metadata.Time?.GetValueOrDefault("modified"),
                Downloads = metadata.Downloads
            };
        }

        /// <summary>
        /// Check if a package exists
        /// </summary>
        public async Task<bool> ExistsAsync(string packageName)
        {
            try
            {
                var url = $"{_baseUrl}/packages/{Uri.EscapeDataString(packageName)}";
                var request = new HttpRequestMessage(HttpMethod.Head, url);
                var response = await _httpClient.SendAsync(request);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Get the latest version of a package
        /// </summary>
        public async Task<SemanticVersion?> GetLatestVersionAsync(string packageName)
        {
            var metadata = await GetPackageMetadataAsync(packageName);
            if (metadata == null)
                return null;

            var latestTag = metadata.DistTags?.GetValueOrDefault("latest");
            if (!string.IsNullOrEmpty(latestTag) && SemanticVersion.TryParse(latestTag, out var version))
                return version;

            var versions = await GetVersionsAsync(packageName);
            return versions.FirstOrDefault();
        }

        private async Task<RegistryPackageMetadata?> GetPackageMetadataAsync(string packageName)
        {
            // Check cache
            if (_metadataCache.TryGetValue(packageName, out var cached) &&
                DateTime.UtcNow - cached.FetchedAt < _cacheExpiration)
            {
                return cached.Metadata;
            }

            try
            {
                var url = $"{_baseUrl}/packages/{Uri.EscapeDataString(packageName)}";
                var response = await _httpClient.GetAsync(url);

                if (response.StatusCode == HttpStatusCode.NotFound)
                    return null;

                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                var metadata = JsonSerializer.Deserialize<RegistryPackageMetadata>(json, _jsonOptions);

                if (metadata != null)
                {
                    _metadataCache[packageName] = new PackageMetadataCache
                    {
                        Metadata = metadata,
                        FetchedAt = DateTime.UtcNow
                    };
                }

                return metadata;
            }
            catch (HttpRequestException ex)
            {
                throw new RegistryException($"Failed to fetch package metadata: {ex.Message}", ex);
            }
        }

        private static string ComputeIntegrity(byte[] data, string expectedIntegrity)
        {
            // Parse expected format (e.g., "sha512-...")
            var parts = expectedIntegrity.Split('-', 2);
            var algorithm = parts[0].ToLowerInvariant();

            System.Security.Cryptography.HashAlgorithm hasher = algorithm switch
            {
                "sha256" => System.Security.Cryptography.SHA256.Create(),
                "sha384" => System.Security.Cryptography.SHA384.Create(),
                "sha512" => System.Security.Cryptography.SHA512.Create(),
                _ => System.Security.Cryptography.SHA512.Create()
            };
            using var _ = hasher;

            var hash = hasher.ComputeHash(data);
            return $"{algorithm}-{Convert.ToBase64String(hash)}";
        }

        #region Response DTOs

        private class RegistryPackageMetadata
        {
            /// <summary>API member</summary>
            [JsonPropertyName("name")]
            public string Name { get; set; } = "";

            /// <summary>API member</summary>
            [JsonPropertyName("description")]
            public string? Description { get; set; }

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
            [JsonPropertyName("dist-tags")]
            public Dictionary<string, string>? DistTags { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("versions")]
            public Dictionary<string, VersionMetadata> Versions { get; set; } = new();

            /// <summary>API member</summary>
            [JsonPropertyName("time")]
            public Dictionary<string, string>? Time { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("downloads")]
            public long Downloads { get; set; }
        }

        private class VersionMetadata
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
            [JsonPropertyName("dist")]
            public DistInfo? Dist { get; set; }
        }

        private class DistInfo
        {
            /// <summary>API member</summary>
            [JsonPropertyName("tarball")]
            public string? Tarball { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("integrity")]
            public string? Integrity { get; set; }

            /// <summary>API member</summary>
            [JsonPropertyName("shasum")]
            public string? Shasum { get; set; }
        }

        private class SearchResponse
        {
            /// <summary>API member</summary>
            [JsonPropertyName("results")]
            public List<PackageSearchResult> Results { get; set; } = new();

            /// <summary>API member</summary>
            [JsonPropertyName("total")]
            public int Total { get; set; }
        }

        private class PackageMetadataCache
        {
            /// <summary>Public API</summary>
            public RegistryPackageMetadata Metadata { get; set; } = null!;
            /// <summary>Public API</summary>
            public DateTime FetchedAt { get; set; }
        }

        #endregion
    }

    /// <summary>
    /// Search result from package registry
    /// </summary>
    public class PackageSearchResult
    {
        /// <summary>API member</summary>
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        /// <summary>API member</summary>
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        /// <summary>API member</summary>
        [JsonPropertyName("version")]
        public string Version { get; set; } = "";

        /// <summary>API member</summary>
        [JsonPropertyName("keywords")]
        public List<string> Keywords { get; set; } = new();

        /// <summary>API member</summary>
        [JsonPropertyName("downloads")]
        public long Downloads { get; set; }

        /// <summary>API member</summary>
        [JsonPropertyName("score")]
        public double Score { get; set; }

        /// <summary>Public API</summary>
        public override string ToString() => $"{Name}@{Version} - {Description}";
    }

    /// <summary>
    /// Detailed package information
    /// </summary>
    public class PackageInfo
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string? Description { get; set; }
        /// <summary>Public API</summary>
        public string LatestVersion { get; set; } = "";
        /// <summary>Public API</summary>
        public string? Homepage { get; set; }
        /// <summary>Public API</summary>
        public string? Repository { get; set; }
        /// <summary>Public API</summary>
        public string? License { get; set; }
        /// <summary>Public API</summary>
        public List<string> Authors { get; set; } = new();
        /// <summary>Public API</summary>
        public List<string> Keywords { get; set; } = new();
        /// <summary>Public API</summary>
        public Dictionary<string, string> Dependencies { get; set; } = new();
        /// <summary>Public API</summary>
        public int VersionCount { get; set; }
        /// <summary>Public API</summary>
        public List<string> Versions { get; set; } = new();
        /// <summary>Public API</summary>
        public string? Created { get; set; }
        /// <summary>Public API</summary>
        public string? Modified { get; set; }
        /// <summary>Public API</summary>
        public long Downloads { get; set; }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"{Name}@{LatestVersion}");
            if (!string.IsNullOrEmpty(Description))
                sb.AppendLine($"  {Description}");
            if (!string.IsNullOrEmpty(License))
                sb.AppendLine($"  License: {License}");
            if (Authors.Any())
                sb.AppendLine($"  Authors: {string.Join(", ", Authors)}");
            if (!string.IsNullOrEmpty(Homepage))
                sb.AppendLine($"  Homepage: {Homepage}");
            sb.AppendLine($"  Downloads: {Downloads:N0}");
            sb.AppendLine($"  Versions: {VersionCount}");
            return sb.ToString();
        }
    }

    #region Exceptions

    /// <summary>Public API</summary>
    public class RegistryException : Exception
    {
        /// <summary>Public API</summary>
        public RegistryException(string message) : base(message) { }
        /// <summary>Public API</summary>
        public RegistryException(string message, Exception inner) : base(message, inner) { }
    }

    /// <summary>Public API</summary>
    public class PackageIntegrityException : Exception
    {
        /// <summary>Public API</summary>
        public PackageIntegrityException(string message) : base(message) { }
    }

    /// <summary>Public API</summary>
    public class PublishException : Exception
    {
        /// <summary>Public API</summary>
        public PublishException(string message) : base(message) { }
    }

    #endregion
}