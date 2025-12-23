using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NSL.PackageManager
{
    /// <summary>
    /// NSL Package Manager (nslpm) - Like pip but for NSL
    /// Manages package installation, removal, and dependency resolution
    /// </summary>
    public class NslPackageManager : IDisposable
    {
        private readonly string _projectRoot;
        private readonly string _packagesDir;
        private readonly string _cacheDir;
        private readonly string _globalPackagesDir;
        private readonly HttpClient? _httpClient;
        private readonly IPackageRegistry _registry;
        private readonly LocalPackageRegistry _localRegistry;
        private readonly LockFile _lockFile;
        private bool _disposed;
        private readonly bool _useLocalRegistry;

        /// <summary>Public API</summary>
        public const string PackagesDirectoryName = "nsl_packages";
        /// <summary>Public API</summary>
        public const string LockFileName = "nsl-package-lock.json";
        /// <summary>Public API</summary>
        public const string ManifestFileName = "nsl-package.json";

        /// <summary>
        /// Event raised when a package operation occurs
        /// </summary>
        public event EventHandler<PackageEventArgs>? PackageEvent;

        /// <summary>
        /// Create a package manager for a project
        /// </summary>
        /// <param name="projectRoot">Root directory of the project</param>
        /// <param name="registryUrl">Optional remote registry URL. If null, uses local registry.</param>
        /// <param name="useLocalRegistry">If true (default), uses local file-based registry. If false and registryUrl provided, uses remote.</param>
        public NslPackageManager(string projectRoot, string? registryUrl = null, bool useLocalRegistry = true)
        {
            _projectRoot = Path.GetFullPath(projectRoot);
            _packagesDir = Path.Combine(_projectRoot, PackagesDirectoryName);
            _cacheDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "nsl", "cache"
            );
            _globalPackagesDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "nsl", "packages"
            );

            // Always create local registry for local operations
            _localRegistry = new LocalPackageRegistry();
            _useLocalRegistry = useLocalRegistry || string.IsNullOrEmpty(registryUrl);

            if (_useLocalRegistry)
            {
                // Use local file-based registry (works offline, no server needed)
                _registry = _localRegistry;
                _httpClient = null;
            }
            else
            {
                // Use remote registry
                _httpClient = new HttpClient();
                _registry = new PackageRegistry(registryUrl!, _httpClient);
            }

            _lockFile = new LockFile(Path.Combine(_projectRoot, LockFileName));

            Directory.CreateDirectory(_packagesDir);
            Directory.CreateDirectory(_cacheDir);
            Directory.CreateDirectory(_globalPackagesDir);
        }

        /// <summary>
        /// Initialize a new NSL project
        /// </summary>
        public PackageManifest Init(string? name = null, string? version = null)
        {
            var manifestPath = Path.Combine(_projectRoot, ManifestFileName);
            if (File.Exists(manifestPath))
            {
                throw new InvalidOperationException("Project already initialized (nsl-package.json exists)");
            }

            var manifest = new PackageManifest
            {
                Name = name ?? Path.GetFileName(_projectRoot).ToLowerInvariant().Replace(' ', '-'),
                Version = version ?? "1.0.0",
                Description = "An NSL package",
                Main = "main.nsl"
            };

            manifest.Save(manifestPath);
            OnPackageEvent(PackageEventType.Initialized, manifest.Name, manifest.Version);

            return manifest;
        }

        /// <summary>
        /// Install a package and its dependencies
        /// </summary>
        public async Task<InstallResult> InstallAsync(string packageSpec, bool saveDependency = true, bool saveDev = false)
        {
            var (packageName, versionConstraint) = ParsePackageSpec(packageSpec);

            OnPackageEvent(PackageEventType.Installing, packageName, versionConstraint);

            try
            {
                // Resolve the best version
                var availableVersions = await _registry.GetVersionsAsync(packageName);
                var constraint = VersionConstraint.Parse(versionConstraint);
                var bestVersion = FindBestVersion(availableVersions, constraint)
                    ?? throw new PackageNotFoundException($"No version of {packageName} satisfies {versionConstraint}");

                // Check if already installed
                var installedVersion = GetInstalledVersion(packageName);
                if (installedVersion != null && installedVersion.ToString() == bestVersion.ToString())
                {
                    OnPackageEvent(PackageEventType.AlreadyInstalled, packageName, bestVersion.ToString());
                    return new InstallResult(true, packageName, bestVersion.ToString(), new List<string>());
                }

                // Download and install
                var packagePath = await DownloadPackageAsync(packageName, bestVersion.ToString());
                await InstallPackageFromPathAsync(packagePath, packageName);

                // Resolve and install dependencies
                var installedDeps = new List<string>();
                var manifest = LoadPackageManifest(packageName);
                foreach (var (depName, depConstraint) in manifest.Dependencies)
                {
                    var depResult = await InstallAsync($"{depName}@{depConstraint}", saveDependency: false);
                    if (depResult.Success)
                        installedDeps.Add(depResult.PackageName);
                }

                // Update project manifest
                if (saveDependency || saveDev)
                {
                    var projectManifest = LoadProjectManifest();
                    var deps = saveDev ? projectManifest.DevDependencies : projectManifest.Dependencies;
                    deps[packageName] = $"^{bestVersion}";
                    projectManifest.Save(Path.Combine(_projectRoot, ManifestFileName));
                }

                // Update lock file
                _lockFile.SetPackage(packageName, bestVersion.ToString(), manifest.Dependencies);
                _lockFile.Save();

                OnPackageEvent(PackageEventType.Installed, packageName, bestVersion.ToString());
                return new InstallResult(true, packageName, bestVersion.ToString(), installedDeps);
            }
            catch (Exception ex)
            {
                OnPackageEvent(PackageEventType.Failed, packageName, ex.Message);
                return new InstallResult(false, packageName, versionConstraint, new List<string>(), ex.Message);
            }
        }

        /// <summary>
        /// Install all dependencies from the manifest
        /// </summary>
        public async Task<List<InstallResult>> InstallAllAsync()
        {
            var results = new List<InstallResult>();
            var manifest = LoadProjectManifest();

            foreach (var (name, constraint) in manifest.Dependencies)
            {
                var result = await InstallAsync($"{name}@{constraint}", saveDependency: false);
                results.Add(result);
            }

            foreach (var (name, constraint) in manifest.DevDependencies)
            {
                var result = await InstallAsync($"{name}@{constraint}", saveDependency: false);
                results.Add(result);
            }

            return results;
        }

        /// <summary>
        /// Uninstall a package
        /// </summary>
        public void Uninstall(string packageName, bool removeDependency = true)
        {
            OnPackageEvent(PackageEventType.Uninstalling, packageName, null);

            var packageDir = Path.Combine(_packagesDir, packageName);
            if (!Directory.Exists(packageDir))
            {
                throw new PackageNotFoundException($"Package {packageName} is not installed");
            }

            // Remove package directory
            Directory.Delete(packageDir, recursive: true);

            // Update project manifest
            if (removeDependency)
            {
                var manifest = LoadProjectManifest();
                manifest.Dependencies.Remove(packageName);
                manifest.DevDependencies.Remove(packageName);
                manifest.Save(Path.Combine(_projectRoot, ManifestFileName));
            }

            // Update lock file
            _lockFile.RemovePackage(packageName);
            _lockFile.Save();

            OnPackageEvent(PackageEventType.Uninstalled, packageName, null);
        }

        /// <summary>
        /// Update a package to the latest version
        /// </summary>
        public async Task<InstallResult> UpdateAsync(string packageName)
        {
            Uninstall(packageName, removeDependency: false);
            return await InstallAsync($"{packageName}@latest");
        }

        /// <summary>
        /// Update all packages
        /// </summary>
        public async Task<List<InstallResult>> UpdateAllAsync()
        {
            var results = new List<InstallResult>();
            var manifest = LoadProjectManifest();

            foreach (var name in manifest.Dependencies.Keys.ToList())
            {
                var result = await UpdateAsync(name);
                results.Add(result);
            }

            return results;
        }

        /// <summary>
        /// List installed packages
        /// </summary>
        public List<InstalledPackage> List()
        {
            var packages = new List<InstalledPackage>();

            if (!Directory.Exists(_packagesDir))
                return packages;

            foreach (var dir in Directory.GetDirectories(_packagesDir))
            {
                var manifestPath = Path.Combine(dir, ManifestFileName);
                if (File.Exists(manifestPath))
                {
                    var manifest = PackageManifest.Load(manifestPath);
                    packages.Add(new InstalledPackage
                    {
                        Name = manifest.Name,
                        Version = manifest.Version,
                        Description = manifest.Description,
                        Path = dir,
                        Dependencies = manifest.Dependencies.Keys.ToList()
                    });
                }
            }

            return packages;
        }

        /// <summary>
        /// Search for packages in the registry
        /// </summary>
        public async Task<List<PackageSearchResult>> SearchAsync(string query, int limit = 20)
        {
            return await _registry.SearchAsync(query, limit);
        }

        /// <summary>
        /// Get package info from the registry
        /// </summary>
        public async Task<PackageInfo?> InfoAsync(string packageName)
        {
            return await _registry.GetInfoAsync(packageName);
        }

        /// <summary>
        /// Check for outdated packages
        /// </summary>
        public async Task<List<OutdatedPackage>> OutdatedAsync()
        {
            var outdated = new List<OutdatedPackage>();
            var installed = List();

            foreach (var pkg in installed)
            {
                var versions = await _registry.GetVersionsAsync(pkg.Name);
                if (versions.Any())
                {
                    var latestVersion = versions.Max();
                    var currentVersion = SemanticVersion.Parse(pkg.Version);
                    if (latestVersion > currentVersion)
                    {
                        outdated.Add(new OutdatedPackage
                        {
                            Name = pkg.Name,
                            CurrentVersion = pkg.Version,
                            LatestVersion = latestVersion.ToString()
                        });
                    }
                }
            }

            return outdated;
        }

        /// <summary>
        /// Pack the current project for publishing
        /// </summary>
        public string Pack(string? outputDir = null)
        {
            var manifest = LoadProjectManifest();
            var validation = manifest.Validate();
            if (!validation.IsValid)
            {
                throw new InvalidOperationException(
                    $"Invalid manifest:\n{string.Join("\n", validation.Errors)}"
                );
            }

            outputDir ??= _projectRoot;
            var packageFileName = $"{manifest.Name}-{manifest.Version}.nslpkg";
            var packagePath = Path.Combine(outputDir, packageFileName);

            using var archive = ZipFile.Open(packagePath, ZipArchiveMode.Create);

            // Add manifest
            archive.CreateEntryFromFile(
                Path.Combine(_projectRoot, ManifestFileName),
                ManifestFileName
            );

            // Add files matching patterns
            foreach (var pattern in manifest.Files)
            {
                var files = GetFilesMatchingPattern(_projectRoot, pattern);
                foreach (var file in files)
                {
                    var relativePath = Path.GetRelativePath(_projectRoot, file);
                    // Skip node_modules equivalent and other excludes
                    if (relativePath.StartsWith(PackagesDirectoryName) ||
                        relativePath.StartsWith(".git") ||
                        relativePath.StartsWith(".nsl"))
                        continue;

                    archive.CreateEntryFromFile(file, relativePath);
                }
            }

            OnPackageEvent(PackageEventType.Packed, manifest.Name, manifest.Version);
            return packagePath;
        }

        /// <summary>
        /// Publish package to registry
        /// </summary>
        public async Task PublishAsync(string? packagePath = null, string? authToken = null)
        {
            packagePath ??= Pack();

            var manifest = LoadProjectManifest();
            if (manifest.Private)
            {
                throw new InvalidOperationException("Cannot publish private package");
            }

            await _registry.PublishAsync(packagePath, authToken);
            OnPackageEvent(PackageEventType.Published, manifest.Name, manifest.Version);
        }

        /// <summary>
        /// Clean the package cache
        /// </summary>
        public void CleanCache()
        {
            if (Directory.Exists(_cacheDir))
            {
                Directory.Delete(_cacheDir, recursive: true);
                Directory.CreateDirectory(_cacheDir);
            }
        }

        /// <summary>
        /// Get the path to an installed package
        /// </summary>
        public string? GetPackagePath(string packageName)
        {
            var path = Path.Combine(_packagesDir, packageName);
            return Directory.Exists(path) ? path : null;
        }

        /// <summary>
        /// Resolve a package import path
        /// </summary>
        public string? ResolveImport(string importPath)
        {
            // Check if it's a relative import
            if (importPath.StartsWith("./") || importPath.StartsWith("../"))
            {
                return null; // Let the caller handle relative imports
            }

            // Parse package name from import
            var parts = importPath.Split('/');
            var packageName = parts[0];

            // Handle scoped packages (@scope/name)
            if (packageName.StartsWith('@') && parts.Length > 1)
            {
                packageName = $"{parts[0]}/{parts[1]}";
            }

            var packagePath = GetPackagePath(packageName);
            if (packagePath == null)
                return null;

            var manifest = LoadPackageManifest(packageName);

            // Check exports
            if (manifest.Exports.Count > 0)
            {
                var exportPath = parts.Skip(packageName.Contains('/') ? 2 : 1).Aggregate(".", Path.Combine);
                if (manifest.Exports.TryGetValue(exportPath, out var exportedFile))
                {
                    return Path.Combine(packagePath, exportedFile);
                }
            }

            // Fall back to main entry point
            return Path.Combine(packagePath, manifest.Main);
        }

        #region Private Methods

        private (string name, string version) ParsePackageSpec(string spec)
        {
            var atIndex = spec.LastIndexOf('@');
            if (atIndex > 0) // Ignore @ at position 0 (scoped packages)
            {
                return (spec[..atIndex], spec[(atIndex + 1)..]);
            }
            return (spec, "latest");
        }

        private SemanticVersion? FindBestVersion(List<SemanticVersion> versions, VersionConstraint constraint)
        {
            return versions
                .Where(v => constraint.IsSatisfiedBy(v))
                .OrderByDescending(v => v)
                .FirstOrDefault();
        }

        private SemanticVersion? GetInstalledVersion(string packageName)
        {
            var packageDir = Path.Combine(_packagesDir, packageName);
            var manifestPath = Path.Combine(packageDir, ManifestFileName);
            if (!File.Exists(manifestPath))
                return null;

            var manifest = PackageManifest.Load(manifestPath);
            return SemanticVersion.Parse(manifest.Version);
        }

        private async Task<string> DownloadPackageAsync(string packageName, string version)
        {
            var cacheFile = Path.Combine(_cacheDir, $"{packageName.Replace('/', '-')}@{version}.nslpkg");

            if (File.Exists(cacheFile))
            {
                return cacheFile;
            }

            var packageBytes = await _registry.DownloadAsync(packageName, version);
            await File.WriteAllBytesAsync(cacheFile, packageBytes);

            return cacheFile;
        }

        private async Task InstallPackageFromPathAsync(string packagePath, string packageName)
        {
            var targetDir = Path.Combine(_packagesDir, packageName.Replace('/', Path.DirectorySeparatorChar));

            if (Directory.Exists(targetDir))
            {
                Directory.Delete(targetDir, recursive: true);
            }

            Directory.CreateDirectory(targetDir);

            // Extract asynchronously to avoid blocking
            await Task.Run(() => ZipFile.ExtractToDirectory(packagePath, targetDir));
        }

        private PackageManifest LoadProjectManifest()
        {
            var manifestPath = Path.Combine(_projectRoot, ManifestFileName);
            if (!File.Exists(manifestPath))
            {
                throw new FileNotFoundException("Project manifest not found. Run 'nslpm init' first.");
            }
            return PackageManifest.Load(manifestPath);
        }

        private PackageManifest LoadPackageManifest(string packageName)
        {
            var packageDir = Path.Combine(_packagesDir, packageName.Replace('/', Path.DirectorySeparatorChar));
            return PackageManifest.LoadFromDirectory(packageDir);
        }

        private static IEnumerable<string> GetFilesMatchingPattern(string root, string pattern)
        {
            // Simple glob pattern matching
            var searchPattern = pattern.Replace("**/*", "*");
            searchPattern = searchPattern.Replace("**/", "");

            return Directory.EnumerateFiles(root, searchPattern, SearchOption.AllDirectories);
        }

        private void OnPackageEvent(PackageEventType type, string packageName, string? version)
        {
            PackageEvent?.Invoke(this, new PackageEventArgs(type, packageName, version));
        }

        #endregion

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _httpClient?.Dispose();
                _disposed = true;
            }
        }

        /// <summary>
        /// Get the local registry for direct access
        /// </summary>
        public LocalPackageRegistry LocalRegistry => _localRegistry;

        /// <summary>
        /// Check if using local registry
        /// </summary>
        public bool IsUsingLocalRegistry => _useLocalRegistry;
    }

    #region Supporting Types

    /// <summary>Public API</summary>
    public class InstallResult
    {
        /// <summary>Public API</summary>
        public bool Success { get; }
        /// <summary>Public API</summary>
        public string PackageName { get; }
        /// <summary>Public API</summary>
        public string Version { get; }
        /// <summary>Public API</summary>
        public List<string> InstalledDependencies { get; }
        /// <summary>Public API</summary>
        public string? Error { get; }

        /// <summary>Public API</summary>
        public InstallResult(bool success, string packageName, string version,
            List<string> installedDependencies, string? error = null)
        {
            Success = success;
            PackageName = packageName;
            Version = version;
            InstalledDependencies = installedDependencies;
            Error = error;
        }
    }

    /// <summary>Public API</summary>
    public class InstalledPackage
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string Version { get; set; } = "";
        /// <summary>Public API</summary>
        public string? Description { get; set; }
        /// <summary>Public API</summary>
        public string Path { get; set; } = "";
        /// <summary>Public API</summary>
        public List<string> Dependencies { get; set; } = new();
    }

    /// <summary>Public API</summary>
    public class OutdatedPackage
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string CurrentVersion { get; set; } = "";
        /// <summary>Public API</summary>
        public string LatestVersion { get; set; } = "";
    }

    /// <summary>Public API</summary>
    public enum PackageEventType
    {
        /// <summary>API member</summary>
        Initialized,
        /// <summary>API member</summary>
        Installing,
        /// <summary>API member</summary>
        Installed,
        /// <summary>API member</summary>
        AlreadyInstalled,
        /// <summary>API member</summary>
        Uninstalling,
        /// <summary>API member</summary>
        Uninstalled,
        /// <summary>API member</summary>
        Packed,
        /// <summary>API member</summary>
        Published,
        Failed
    }

    /// <summary>Public API</summary>
    public class PackageEventArgs : EventArgs
    {
        /// <summary>Public API</summary>
        public PackageEventType Type { get; }
        /// <summary>Public API</summary>
        public string PackageName { get; }
        /// <summary>Public API</summary>
        public string? Version { get; }

        /// <summary>Public API</summary>
        public PackageEventArgs(PackageEventType type, string packageName, string? version)
        {
            Type = type;
            PackageName = packageName;
            Version = version;
        }
    }

    /// <summary>Public API</summary>
    public class PackageNotFoundException : Exception
    {
        /// <summary>Public API</summary>
        public PackageNotFoundException(string message) : base(message) { }
    }

    #endregion
}