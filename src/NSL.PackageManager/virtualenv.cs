using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NSL.PackageManager
{
    /// <summary>
    /// NSL Virtual Environment - isolated package environments like Python's venv
    /// </summary>
    public class VirtualEnv
    {
        private readonly string _envPath;
        private readonly VirtualEnvConfig _config;

        /// <summary>Public API</summary>
        public const string ConfigFileName = "nsl-env.json";
        /// <summary>Public API</summary>
        public const string PackagesDirName = "packages";
        /// <summary>Public API</summary>
        public const string BinDirName = "bin";

        /// <summary>
        /// Path to the virtual environment
        /// </summary>
        public string Path => _envPath;

        /// <summary>
        /// Name of the virtual environment
        /// </summary>
        public string Name => _config.Name;

        /// <summary>
        /// Path to packages directory
        /// </summary>
        public string PackagesPath => System.IO.Path.Combine(_envPath, PackagesDirName);

        /// <summary>
        /// Path to bin directory
        /// </summary>
        public string BinPath => System.IO.Path.Combine(_envPath, BinDirName);

        /// <summary>
        /// Whether the environment is currently active
        /// </summary>
        public bool IsActive => Environment.GetEnvironmentVariable("NSL_VIRTUAL_ENV") == _envPath;

        private VirtualEnv(string envPath, VirtualEnvConfig config)
        {
            _envPath = envPath;
            _config = config;
        }

        /// <summary>
        /// Create a new virtual environment
        /// </summary>
        public static VirtualEnv Create(string path, string? name = null)
        {
            var envPath = System.IO.Path.GetFullPath(path);

            if (Directory.Exists(envPath))
            {
                throw new InvalidOperationException($"Directory already exists: {envPath}");
            }

            // Create directory structure
            Directory.CreateDirectory(envPath);
            Directory.CreateDirectory(System.IO.Path.Combine(envPath, PackagesDirName));
            Directory.CreateDirectory(System.IO.Path.Combine(envPath, BinDirName));

            // Create config
            var config = new VirtualEnvConfig
            {
                Name = name ?? System.IO.Path.GetFileName(envPath),
                Created = DateTime.UtcNow,
                NslVersion = GetNslVersion()
            };

            var configPath = System.IO.Path.Combine(envPath, ConfigFileName);
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(configPath, json);

            // Create activation scripts
            CreateActivationScripts(envPath, config.Name);

            return new VirtualEnv(envPath, config);
        }

        /// <summary>
        /// Load an existing virtual environment
        /// </summary>
        public static VirtualEnv Load(string path)
        {
            var envPath = System.IO.Path.GetFullPath(path);
            var configPath = System.IO.Path.Combine(envPath, ConfigFileName);

            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException($"Not a valid NSL virtual environment: {envPath}");
            }

            var json = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize<VirtualEnvConfig>(json)
                ?? throw new InvalidOperationException("Failed to load virtual environment config");

            return new VirtualEnv(envPath, config);
        }

        /// <summary>
        /// Check if a directory is a virtual environment
        /// </summary>
        public static bool IsVirtualEnv(string path)
        {
            var configPath = System.IO.Path.Combine(path, ConfigFileName);
            return File.Exists(configPath);
        }

        /// <summary>
        /// Get the currently active virtual environment
        /// </summary>
        public static VirtualEnv? GetActive()
        {
            var envPath = Environment.GetEnvironmentVariable("NSL_VIRTUAL_ENV");
            if (string.IsNullOrEmpty(envPath) || !IsVirtualEnv(envPath))
            {
                return null;
            }
            return Load(envPath);
        }

        /// <summary>
        /// Get environment variables for activation
        /// </summary>
        public Dictionary<string, string> GetActivationEnvironment()
        {
            var env = new Dictionary<string, string>
            {
                ["NSL_VIRTUAL_ENV"] = _envPath,
                ["NSL_VIRTUAL_ENV_NAME"] = _config.Name,
            };

            // Prepend bin directory to PATH
            var currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            env["PATH"] = $"{BinPath}{System.IO.Path.PathSeparator}{currentPath}";

            // Store original PATH for deactivation
            env["_OLD_NSL_PATH"] = currentPath;

            return env;
        }

        /// <summary>
        /// Get environment variables for deactivation
        /// </summary>
        public static Dictionary<string, string?> GetDeactivationEnvironment()
        {
            var env = new Dictionary<string, string?>
            {
                ["NSL_VIRTUAL_ENV"] = null,
                ["NSL_VIRTUAL_ENV_NAME"] = null,
            };

            // Restore original PATH
            var oldPath = Environment.GetEnvironmentVariable("_OLD_NSL_PATH");
            if (!string.IsNullOrEmpty(oldPath))
            {
                env["PATH"] = oldPath;
                env["_OLD_NSL_PATH"] = null;
            }

            return env;
        }

        /// <summary>
        /// Install a package into this virtual environment
        /// </summary>
        public void InstallPackage(string packagePath)
        {
            var manifest = PackageManifest.LoadFromDirectory(packagePath);
            var targetPath = System.IO.Path.Combine(PackagesPath, manifest.Name);

            if (Directory.Exists(targetPath))
            {
                Directory.Delete(targetPath, recursive: true);
            }

            CopyDirectory(packagePath, targetPath);

            // Create bin links if the package has scripts
            if (manifest.Scripts.TryGetValue("bin", out var binScript))
            {
                CreateBinLink(manifest.Name, binScript);
            }
        }

        /// <summary>
        /// Remove a package from this virtual environment
        /// </summary>
        public void RemovePackage(string packageName)
        {
            var packagePath = System.IO.Path.Combine(PackagesPath, packageName);

            if (!Directory.Exists(packagePath))
            {
                throw new PackageNotFoundException($"Package not found: {packageName}");
            }

            Directory.Delete(packagePath, recursive: true);

            // Remove bin links
            RemoveBinLink(packageName);
        }

        /// <summary>
        /// List installed packages in this virtual environment
        /// </summary>
        public List<InstalledPackage> ListPackages()
        {
            var packages = new List<InstalledPackage>();

            if (!Directory.Exists(PackagesPath))
            {
                return packages;
            }

            foreach (var dir in Directory.GetDirectories(PackagesPath))
            {
                var manifestPath = System.IO.Path.Combine(dir, NslPackageManager.ManifestFileName);
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
        /// Delete this virtual environment
        /// </summary>
        public void Delete()
        {
            if (IsActive)
            {
                throw new InvalidOperationException("Cannot delete active virtual environment. Deactivate first.");
            }

            Directory.Delete(_envPath, recursive: true);
        }

        private static void CreateActivationScripts(string envPath, string envName)
        {
            var binPath = System.IO.Path.Combine(envPath, BinDirName);

            // Windows batch script
            var activateBat = $@"@echo off
set ""NSL_VIRTUAL_ENV={envPath}""
set ""NSL_VIRTUAL_ENV_NAME={envName}""
set ""_OLD_NSL_PATH=%PATH%""
set ""PATH={binPath};%PATH%""
echo Virtual environment '{envName}' activated
";
            File.WriteAllText(System.IO.Path.Combine(binPath, "activate.bat"), activateBat);

            // Windows PowerShell script
            var activatePs1 = $@"$env:NSL_VIRTUAL_ENV = ""{envPath}""
$env:NSL_VIRTUAL_ENV_NAME = ""{envName}""
$env:_OLD_NSL_PATH = $env:PATH
$env:PATH = ""{binPath}"" + "";"" + $env:PATH
Write-Host ""Virtual environment '{envName}' activated""
";
            File.WriteAllText(System.IO.Path.Combine(binPath, "activate.ps1"), activatePs1);

            // Unix shell script
            var activateSh = $@"#!/bin/bash
export NSL_VIRTUAL_ENV=""{envPath}""
export NSL_VIRTUAL_ENV_NAME=""{envName}""
export _OLD_NSL_PATH=""$PATH""
export PATH=""{binPath}:$PATH""
echo ""Virtual environment '{envName}' activated""
";
            File.WriteAllText(System.IO.Path.Combine(binPath, "activate"), activateSh);

            // Deactivation scripts
            var deactivateBat = @"@echo off
set ""PATH=%_OLD_NSL_PATH%""
set ""NSL_VIRTUAL_ENV=""
set ""NSL_VIRTUAL_ENV_NAME=""
set ""_OLD_NSL_PATH=""
echo Virtual environment deactivated
";
            File.WriteAllText(System.IO.Path.Combine(binPath, "deactivate.bat"), deactivateBat);

            var deactivatePs1 = @"$env:PATH = $env:_OLD_NSL_PATH
$env:NSL_VIRTUAL_ENV = $null
$env:NSL_VIRTUAL_ENV_NAME = $null
$env:_OLD_NSL_PATH = $null
Write-Host ""Virtual environment deactivated""
";
            File.WriteAllText(System.IO.Path.Combine(binPath, "deactivate.ps1"), deactivatePs1);

            var deactivateSh = @"#!/bin/bash
export PATH=""$_OLD_NSL_PATH""
unset NSL_VIRTUAL_ENV
unset NSL_VIRTUAL_ENV_NAME
unset _OLD_NSL_PATH
echo ""Virtual environment deactivated""
";
            File.WriteAllText(System.IO.Path.Combine(binPath, "deactivate"), deactivateSh);
        }

        private void CreateBinLink(string packageName, string script)
        {
            var scriptPath = System.IO.Path.Combine(PackagesPath, packageName, script);
            var linkPath = System.IO.Path.Combine(BinPath, packageName);

            // Create a wrapper script
            if (OperatingSystem.IsWindows())
            {
                var wrapper = $@"@echo off
nsl ""{scriptPath}"" %*
";
                File.WriteAllText(linkPath + ".bat", wrapper);
            }
            else
            {
                var wrapper = $@"#!/bin/bash
nsl ""{scriptPath}"" ""$@""
";
                File.WriteAllText(linkPath, wrapper);
                // Make executable on Unix
            }
        }

        private void RemoveBinLink(string packageName)
        {
            var batPath = System.IO.Path.Combine(BinPath, packageName + ".bat");
            var shPath = System.IO.Path.Combine(BinPath, packageName);

            if (File.Exists(batPath)) File.Delete(batPath);
            if (File.Exists(shPath)) File.Delete(shPath);
        }

        private static void CopyDirectory(string sourceDir, string destDir)
        {
            Directory.CreateDirectory(destDir);

            foreach (var file in Directory.GetFiles(sourceDir))
            {
                var destFile = System.IO.Path.Combine(destDir, System.IO.Path.GetFileName(file));
                File.Copy(file, destFile, overwrite: true);
            }

            foreach (var dir in Directory.GetDirectories(sourceDir))
            {
                var destSubDir = System.IO.Path.Combine(destDir, System.IO.Path.GetFileName(dir));
                CopyDirectory(dir, destSubDir);
            }
        }

        private static string GetNslVersion()
        {
            // Get version from the NSL.PackageManager assembly
            var assembly = typeof(VirtualEnv).Assembly;
            var version = assembly.GetName().Version;
            return version?.ToString(3) ?? "1.0.0";
        }
    }

    internal class VirtualEnvConfig
    {
        /// <summary>API member</summary>
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        /// <summary>API member</summary>
        [JsonPropertyName("created")]
        public DateTime Created { get; set; }

        /// <summary>API member</summary>
        [JsonPropertyName("nslVersion")]
        public string NslVersion { get; set; } = "";

        /// <summary>API member</summary>
        [JsonPropertyName("pythonPath")]
        public string? PythonPath { get; set; }
    }
}