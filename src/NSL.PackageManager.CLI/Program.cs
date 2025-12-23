using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using NSL.PackageManager;

namespace NSL.PackageManager.CLI
{
    class Program
    {
        private static readonly ConsoleColor DefaultColor = Console.ForegroundColor;
        private static bool _verbose = false;

        static async Task<int> Main(string[] args)
        {
            if (args.Length == 0)
            {
                PrintUsage();
                return 0;
            }

            // Check for global flags
            var argList = args.ToList();
            _verbose = argList.Remove("--verbose") || argList.Remove("-v");
            var help = argList.Remove("--help") || argList.Remove("-h");

            if (help && argList.Count <= 1)
            {
                PrintUsage();
                return 0;
            }

            var command = argList[0].ToLowerInvariant();
            var commandArgs = argList.Skip(1).ToArray();

            try
            {
                return command switch
                {
                    "init" => await InitAsync(commandArgs),
                    "install" or "i" or "add" => await InstallAsync(commandArgs),
                    "uninstall" or "remove" or "rm" => Uninstall(commandArgs),
                    "update" or "upgrade" => await UpdateAsync(commandArgs),
                    "list" or "ls" => List(commandArgs),
                    "freeze" => Freeze(commandArgs),
                    "search" or "s" => await SearchAsync(commandArgs),
                    "info" or "show" => await InfoAsync(commandArgs),
                    "outdated" => await OutdatedAsync(commandArgs),
                    "pack" => Pack(commandArgs),
                    "publish" => await PublishAsync(commandArgs),
                    "cache" => Cache(commandArgs),
                    "tree" => await TreeAsync(commandArgs),
                    "validate" or "check" => Validate(commandArgs),
                    "config" => Config(commandArgs),
                    "version" or "--version" => Version(),
                    "run" => await RunScriptAsync(commandArgs),
                    "exec" or "x" => await ExecAsync(commandArgs),
                    "scripts" => ListScripts(),
                    "venv" => VenvCommand(commandArgs),
                    "hash" => Hash(commandArgs),
                    "debug" => Debug(),
                    _ => UnknownCommand(command)
                };
            }
            catch (Exception ex)
            {
                PrintError(ex.Message);
                if (_verbose && ex.StackTrace != null)
                {
                    Console.WriteLine(ex.StackTrace);
                }
                return 1;
            }
        }

        static Task<int> InitAsync(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            var name = args.Length > 0 ? args[0] : null;
            var version = args.Length > 1 ? args[1] : null;

            using var pm = new NslPackageManager(projectDir);

            pm.PackageEvent += OnPackageEvent;
            var manifest = pm.Init(name, version);

            PrintSuccess($"Initialized package: {manifest.Name}@{manifest.Version}");
            Console.WriteLine($"Created {NslPackageManager.ManifestFileName}");

            return Task.FromResult(0);
        }

        static async Task<int> InstallAsync(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            var argList = args.ToList();

            // Parse flags
            var saveDev = argList.Remove("--save-dev") || argList.Remove("-D");
            var dryRun = argList.Remove("--dry-run") || argList.Remove("-n");
            var upgrade = argList.Remove("--upgrade") || argList.Remove("-U");
            var quiet = argList.Remove("--quiet") || argList.Remove("-q");
            var force = argList.Remove("--force-reinstall") || argList.Remove("-f");

            // Check for requirements file
            var reqIndex = argList.IndexOf("-r");
            if (reqIndex < 0) reqIndex = argList.IndexOf("--requirements");
            string? requirementsFile = null;
            if (reqIndex >= 0 && reqIndex < argList.Count - 1)
            {
                requirementsFile = argList[reqIndex + 1];
                argList.RemoveAt(reqIndex + 1);
                argList.RemoveAt(reqIndex);
            }

            // Check for editable install
            var editIndex = argList.IndexOf("-e");
            if (editIndex < 0) editIndex = argList.IndexOf("--editable");
            string? editablePath = null;
            if (editIndex >= 0 && editIndex < argList.Count - 1)
            {
                editablePath = argList[editIndex + 1];
                argList.RemoveAt(editIndex + 1);
                argList.RemoveAt(editIndex);
            }

            // Handle editable install
            if (editablePath != null)
            {
                return await InstallEditableAsync(editablePath, projectDir);
            }

            var packages = argList.Where(a => !a.StartsWith("-")).ToList();

            // Load packages from requirements file if specified
            if (requirementsFile != null)
            {
                if (!File.Exists(requirementsFile))
                {
                    PrintError($"Requirements file not found: {requirementsFile}");
                    return 1;
                }

                var reqPackages = ParseRequirementsFile(requirementsFile);
                packages.AddRange(reqPackages);

                if (!quiet)
                {
                    Console.WriteLine($"Loaded {reqPackages.Count} packages from {requirementsFile}");
                }
            }

            using var pm = new NslPackageManager(projectDir);

            if (!quiet)
            {
                pm.PackageEvent += OnPackageEvent;
            }

            if (packages.Count == 0)
            {
                // Install all dependencies
                if (dryRun)
                {
                    Console.WriteLine("Would install all dependencies from manifest (dry-run)");
                    var manifestPath = Path.Combine(projectDir, NslPackageManager.ManifestFileName);
                    if (File.Exists(manifestPath))
                    {
                        var manifest = PackageManifest.Load(manifestPath);
                        foreach (var dep in manifest.Dependencies)
                        {
                            Console.WriteLine($"  Would install: {dep.Key}@{dep.Value}");
                        }
                        foreach (var dep in manifest.DevDependencies)
                        {
                            Console.WriteLine($"  Would install (dev): {dep.Key}@{dep.Value}");
                        }
                    }
                    else
                    {
                        Console.WriteLine("  No manifest found - nothing to install");
                    }
                    return 0;
                }

                Console.WriteLine("Installing dependencies...");
                var results = await pm.InstallAllAsync();

                var success = results.Count(r => r.Success);
                var failed = results.Count - success;

                if (failed > 0)
                {
                    PrintWarning($"Installed {success} packages, {failed} failed");
                    return 1;
                }

                PrintSuccess($"Installed {success} packages");
                return 0;
            }

            // Dry-run mode - just show what would be installed
            if (dryRun)
            {
                Console.WriteLine("Dry-run mode - would install:");
                foreach (var pkg in packages)
                {
                    var (name, version) = ParsePackageSpec(pkg);
                    Console.WriteLine($"  {name}{(version != null ? $"@{version}" : "")}");
                }
                Console.WriteLine("\nNo changes made (dry-run)");
                return 0;
            }

            // Install specific packages
            var anyFailed = false;
            foreach (var pkg in packages)
            {
                var result = await pm.InstallAsync(pkg, saveDependency: !saveDev, saveDev: saveDev);
                if (!result.Success)
                {
                    PrintError($"Failed to install {pkg}: {result.Error}");
                    anyFailed = true;
                }
            }

            return anyFailed ? 1 : 0;
        }

        /// <summary>
        /// Parse a requirements file (like pip's requirements.txt)
        /// Supports: package==version, package>=version, package (any version)
        /// Lines starting with # are comments
        /// </summary>
        static List<string> ParseRequirementsFile(string filePath)
        {
            var packages = new List<string>();
            var lines = File.ReadAllLines(filePath);

            foreach (var line in lines)
            {
                var trimmed = line.Trim();

                // Skip empty lines and comments
                if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                    continue;

                // Handle inline comments
                var commentIndex = trimmed.IndexOf('#');
                if (commentIndex > 0)
                    trimmed = trimmed.Substring(0, commentIndex).Trim();

                // Handle -r (recursive requirements) - skip for now
                if (trimmed.StartsWith("-r ") || trimmed.StartsWith("--requirements "))
                    continue;

                // Handle -e (editable) - skip for now
                if (trimmed.StartsWith("-e ") || trimmed.StartsWith("--editable "))
                    continue;

                // Parse version specifiers (==, >=, <=, ~=, !=)
                // Convert to nslpm format: package@constraint
                var pkg = trimmed;
                if (trimmed.Contains("=="))
                {
                    var parts = trimmed.Split("==", 2);
                    pkg = $"{parts[0].Trim()}@{parts[1].Trim()}";
                }
                else if (trimmed.Contains(">="))
                {
                    var parts = trimmed.Split(">=", 2);
                    pkg = $"{parts[0].Trim()}@>={parts[1].Trim()}";
                }
                else if (trimmed.Contains("~="))
                {
                    var parts = trimmed.Split("~=", 2);
                    pkg = $"{parts[0].Trim()}@~{parts[1].Trim()}";
                }

                packages.Add(pkg);
            }

            return packages;
        }

        /// <summary>
        /// Parse a package spec like "package@1.0.0" or "package"
        /// </summary>
        static (string name, string? version) ParsePackageSpec(string spec)
        {
            var atIndex = spec.IndexOf('@');
            if (atIndex > 0)
            {
                return (spec.Substring(0, atIndex), spec.Substring(atIndex + 1));
            }
            return (spec, null);
        }

        /// <summary>
        /// Install a package in editable/development mode (like pip install -e .)
        /// Creates a symlink instead of copying files
        /// </summary>
        static Task<int> InstallEditableAsync(string path, string projectDir)
        {
            var targetPath = Path.GetFullPath(path);

            if (!Directory.Exists(targetPath))
            {
                PrintError($"Directory not found: {targetPath}");
                return Task.FromResult(1);
            }

            var manifestPath = Path.Combine(targetPath, NslPackageManager.ManifestFileName);
            if (!File.Exists(manifestPath))
            {
                PrintError($"No {NslPackageManager.ManifestFileName} found in {targetPath}");
                return Task.FromResult(1);
            }

            var manifest = PackageManifest.Load(manifestPath);

            // Get packages directory
            var packagesDir = Path.Combine(projectDir, "nsl_packages");
            Directory.CreateDirectory(packagesDir);

            var linkPath = Path.Combine(packagesDir, manifest.Name);

            // Remove existing if any
            if (Directory.Exists(linkPath))
            {
                // Check if it's a symlink
                var info = new DirectoryInfo(linkPath);
                if (info.Attributes.HasFlag(FileAttributes.ReparsePoint))
                {
                    Directory.Delete(linkPath);
                }
                else
                {
                    PrintError($"Package already installed (not editable): {manifest.Name}");
                    Console.WriteLine($"Run 'nslpm uninstall {manifest.Name}' first");
                    return Task.FromResult(1);
                }
            }

            try
            {
                // Create symlink (directory junction on Windows)
                if (OperatingSystem.IsWindows())
                {
                    // Use mklink /J for directory junction
                    var startInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "cmd.exe",
                        Arguments = $"/c mklink /J \"{linkPath}\" \"{targetPath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    using var process = System.Diagnostics.Process.Start(startInfo);
                    process?.WaitForExit();

                    if (process?.ExitCode != 0)
                    {
                        // Fall back to directory info if mklink fails
                        var errorOutput = process?.StandardError.ReadToEnd();
                        PrintError($"Failed to create symlink: {errorOutput}");
                        return Task.FromResult(1);
                    }
                }
                else
                {
                    // Unix-style symlink
                    var startInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "ln",
                        Arguments = $"-s \"{targetPath}\" \"{linkPath}\"",
                        UseShellExecute = false,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    using var process = System.Diagnostics.Process.Start(startInfo);
                    process?.WaitForExit();

                    if (process?.ExitCode != 0)
                    {
                        PrintError("Failed to create symlink");
                        return Task.FromResult(1);
                    }
                }

                // Create a marker file to indicate this is an editable install
                var markerPath = Path.Combine(packagesDir, $".{manifest.Name}.editable");
                File.WriteAllText(markerPath, targetPath);

                PrintSuccess($"Installed {manifest.Name}@{manifest.Version} (editable)");
                Console.WriteLine($"  Linked: {linkPath} -> {targetPath}");
                Console.WriteLine("  Changes to source will take effect immediately");

                return Task.FromResult(0);
            }
            catch (Exception ex)
            {
                PrintError($"Failed to create editable install: {ex.Message}");
                return Task.FromResult(1);
            }
        }

        static int Uninstall(string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify packages to uninstall");
                return 1;
            }

            var projectDir = Directory.GetCurrentDirectory();
            using var pm = new NslPackageManager(projectDir);
            pm.PackageEvent += OnPackageEvent;

            var anyFailed = false;
            foreach (var pkg in args.Where(a => !a.StartsWith("-")))
            {
                try
                {
                    pm.Uninstall(pkg);
                }
                catch (PackageNotFoundException)
                {
                    PrintError($"Package not installed: {pkg}");
                    anyFailed = true;
                }
            }

            return anyFailed ? 1 : 0;
        }

        static async Task<int> UpdateAsync(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            using var pm = new NslPackageManager(projectDir);
            pm.PackageEvent += OnPackageEvent;

            var packages = args.Where(a => !a.StartsWith("-")).ToArray();

            if (packages.Length == 0)
            {
                Console.WriteLine("Updating all packages...");
                var results = await pm.UpdateAllAsync();
                var success = results.Count(r => r.Success);
                PrintSuccess($"Updated {success} packages");
                return 0;
            }

            foreach (var pkg in packages)
            {
                await pm.UpdateAsync(pkg);
            }

            return 0;
        }

        static int List(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            using var pm = new NslPackageManager(projectDir);

            var packages = pm.List();

            if (packages.Count == 0)
            {
                Console.WriteLine("No packages installed");
                return 0;
            }

            Console.WriteLine($"Installed packages ({packages.Count}):\n");

            foreach (var pkg in packages.OrderBy(p => p.Name))
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"  {pkg.Name}");
                Console.ForegroundColor = DefaultColor;
                Console.WriteLine($"@{pkg.Version}");

                if (!string.IsNullOrEmpty(pkg.Description) && _verbose)
                {
                    Console.WriteLine($"    {pkg.Description}");
                }

                if (pkg.Dependencies.Any() && _verbose)
                {
                    Console.WriteLine($"    deps: {string.Join(", ", pkg.Dependencies)}");
                }
            }

            return 0;
        }

        static async Task<int> SearchAsync(string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify a search query");
                return 1;
            }

            var query = string.Join(" ", args.Where(a => !a.StartsWith("-")));
            var projectDir = Directory.GetCurrentDirectory();
            using var pm = new NslPackageManager(projectDir);

            Console.WriteLine($"Searching for '{query}'...\n");

            var results = await pm.SearchAsync(query);

            if (results.Count == 0)
            {
                Console.WriteLine("No packages found");
                return 0;
            }

            foreach (var pkg in results)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"{pkg.Name}");
                Console.ForegroundColor = DefaultColor;
                Console.Write($"@{pkg.Version}");

                if (pkg.Downloads > 0)
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.Write($" ({pkg.Downloads:N0} downloads)");
                }

                Console.ForegroundColor = DefaultColor;
                Console.WriteLine();

                if (!string.IsNullOrEmpty(pkg.Description))
                {
                    Console.WriteLine($"  {pkg.Description}");
                }
            }

            return 0;
        }

        static async Task<int> InfoAsync(string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify a package name");
                return 1;
            }

            var packageName = args[0];
            var projectDir = Directory.GetCurrentDirectory();
            using var pm = new NslPackageManager(projectDir);

            var info = await pm.InfoAsync(packageName);

            if (info == null)
            {
                PrintError($"Package not found: {packageName}");
                return 1;
            }

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write($"{info.Name}");
            Console.ForegroundColor = DefaultColor;
            Console.WriteLine($"@{info.LatestVersion}");
            Console.WriteLine();

            if (!string.IsNullOrEmpty(info.Description))
                Console.WriteLine($"  {info.Description}");
            Console.WriteLine();

            Console.WriteLine($"  License:    {info.License ?? "Unknown"}");
            Console.WriteLine($"  Homepage:   {info.Homepage ?? "N/A"}");
            Console.WriteLine($"  Repository: {info.Repository ?? "N/A"}");
            Console.WriteLine($"  Downloads:  {info.Downloads:N0}");
            Console.WriteLine($"  Versions:   {info.VersionCount}");

            if (info.Authors.Any())
                Console.WriteLine($"  Authors:    {string.Join(", ", info.Authors)}");

            if (info.Keywords.Any())
                Console.WriteLine($"  Keywords:   {string.Join(", ", info.Keywords)}");

            if (info.Dependencies.Any())
            {
                Console.WriteLine();
                Console.WriteLine("  Dependencies:");
                foreach (var (dep, constraint) in info.Dependencies)
                {
                    Console.WriteLine($"    {dep}: {constraint}");
                }
            }

            if (_verbose && info.Versions.Any())
            {
                Console.WriteLine();
                Console.WriteLine("  All Versions:");
                Console.WriteLine($"    {string.Join(", ", info.Versions.Take(10))}");
                if (info.Versions.Count > 10)
                    Console.WriteLine($"    ... and {info.Versions.Count - 10} more");
            }

            Console.WriteLine();
            return 0;
        }

        static async Task<int> OutdatedAsync(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            using var pm = new NslPackageManager(projectDir);

            Console.WriteLine("Checking for outdated packages...\n");

            var outdated = await pm.OutdatedAsync();

            if (outdated.Count == 0)
            {
                PrintSuccess("All packages are up to date!");
                return 0;
            }

            Console.WriteLine($"{"Package",-30} {"Current",-15} {"Latest",-15}");
            Console.WriteLine(new string('-', 60));

            foreach (var pkg in outdated.OrderBy(p => p.Name))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write($"{pkg.Name,-30}");
                Console.ForegroundColor = DefaultColor;
                Console.Write($" {pkg.CurrentVersion,-15}");
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($" {pkg.LatestVersion,-15}");
                Console.ForegroundColor = DefaultColor;
            }

            Console.WriteLine();
            Console.WriteLine($"Run 'nslpm update' to update all packages");

            return 0;
        }

        static int Pack(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            var outputDir = args.Length > 0 ? args[0] : null;

            using var pm = new NslPackageManager(projectDir);
            pm.PackageEvent += OnPackageEvent;

            var packagePath = pm.Pack(outputDir);
            PrintSuccess($"Created: {packagePath}");

            return 0;
        }

        static async Task<int> PublishAsync(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();

            using var pm = new NslPackageManager(projectDir);
            pm.PackageEvent += OnPackageEvent;

            // Local registry doesn't need auth token
            if (pm.IsUsingLocalRegistry)
            {
                Console.WriteLine("Publishing to local registry...");
            }

            await pm.PublishAsync(authToken: null);
            PrintSuccess("Package published to local registry!");

            return 0;
        }

        static int Cache(string[] args)
        {
            var subcommand = args.Length > 0 ? args[0] : "info";
            var cacheDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "nsl", "cache"
            );

            switch (subcommand.ToLowerInvariant())
            {
                case "clean" or "clear" or "purge":
                    if (Directory.Exists(cacheDir))
                    {
                        var files = Directory.GetFiles(cacheDir);
                        foreach (var file in files)
                        {
                            File.Delete(file);
                        }
                        PrintSuccess($"Purged {files.Length} cached packages");
                    }
                    else
                    {
                        Console.WriteLine("Cache is already empty");
                    }
                    break;

                case "list" or "ls":
                    Console.WriteLine($"Cache directory: {cacheDir}\n");
                    if (Directory.Exists(cacheDir))
                    {
                        var files = Directory.GetFiles(cacheDir).OrderBy(f => f).ToArray();
                        if (files.Length == 0)
                        {
                            Console.WriteLine("No cached packages");
                        }
                        else
                        {
                            foreach (var file in files)
                            {
                                var info = new FileInfo(file);
                                Console.ForegroundColor = ConsoleColor.Cyan;
                                Console.Write($"  {Path.GetFileName(file)}");
                                Console.ForegroundColor = ConsoleColor.DarkGray;
                                Console.WriteLine($"  ({FormatSize(info.Length)})");
                                Console.ForegroundColor = DefaultColor;
                            }
                            Console.WriteLine();
                            Console.WriteLine($"Total: {files.Length} packages, {FormatSize(files.Sum(f => new FileInfo(f).Length))}");
                        }
                    }
                    else
                    {
                        Console.WriteLine("No cache directory");
                    }
                    break;

                case "dir":
                    Console.WriteLine(cacheDir);
                    break;

                case "remove" or "rm":
                    if (args.Length < 2)
                    {
                        PrintError("Please specify a package to remove from cache");
                        return 1;
                    }
                    var pattern = args[1];
                    if (Directory.Exists(cacheDir))
                    {
                        var matches = Directory.GetFiles(cacheDir, $"*{pattern}*");
                        if (matches.Length == 0)
                        {
                            Console.WriteLine($"No cached packages matching '{pattern}'");
                        }
                        else
                        {
                            foreach (var file in matches)
                            {
                                File.Delete(file);
                                Console.WriteLine($"  Removed: {Path.GetFileName(file)}");
                            }
                            PrintSuccess($"Removed {matches.Length} cached packages");
                        }
                    }
                    break;

                case "info":
                default:
                    Console.WriteLine($"Cache directory: {cacheDir}");
                    if (Directory.Exists(cacheDir))
                    {
                        var files = Directory.GetFiles(cacheDir);
                        var size = files.Sum(f => new FileInfo(f).Length);
                        Console.WriteLine($"Cached packages: {files.Length}");
                        Console.WriteLine($"Total size: {FormatSize(size)}");
                    }
                    else
                    {
                        Console.WriteLine("Cached packages: 0");
                        Console.WriteLine("Total size: 0 B");
                    }
                    break;
            }

            return 0;
        }

        static async Task<int> TreeAsync(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            var manifestPath = Path.Combine(projectDir, NslPackageManager.ManifestFileName);

            if (!File.Exists(manifestPath))
            {
                PrintError("No nsl-package.json found");
                return 1;
            }

            var manifest = PackageManifest.Load(manifestPath);
            using var pm = new NslPackageManager(projectDir);

            var resolver = new DependencyResolver(
                new LocalPackageRegistry()
            );

            Console.WriteLine($"{manifest.Name}@{manifest.Version}");

            var resolution = await resolver.ResolveAsync(manifest.Dependencies);

            if (!resolution.Success)
            {
                foreach (var error in resolution.Errors)
                {
                    PrintError(error);
                }
                return 1;
            }

            Console.WriteLine(resolution.GetDependencyTree());
            return 0;
        }

        static int Validate(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            var manifestPath = Path.Combine(projectDir, NslPackageManager.ManifestFileName);

            if (!File.Exists(manifestPath))
            {
                PrintError("No nsl-package.json found");
                return 1;
            }

            var manifest = PackageManifest.Load(manifestPath);
            var validation = manifest.Validate();

            if (validation.IsValid)
            {
                PrintSuccess("Package manifest is valid!");
            }
            else
            {
                PrintError("Validation failed:");
                foreach (var error in validation.Errors)
                {
                    Console.WriteLine($"  ✗ {error}");
                }
            }

            if (validation.Warnings.Any())
            {
                Console.WriteLine();
                PrintWarning("Warnings:");
                foreach (var warning in validation.Warnings)
                {
                    Console.WriteLine($"  ⚠ {warning}");
                }
            }

            // Also validate lock file if it exists
            var lockPath = Path.Combine(projectDir, NslPackageManager.LockFileName);
            if (File.Exists(lockPath))
            {
                Console.WriteLine();
                var lockFile = new LockFile(lockPath);
                var lockValidation = lockFile.Validate();

                if (lockValidation.IsValid)
                {
                    Console.WriteLine($"Lock file is valid ({lockFile.Packages.Count} packages)");
                }
                else
                {
                    PrintWarning("Lock file issues:");
                    foreach (var issue in lockValidation.Issues)
                    {
                        Console.WriteLine($"  ⚠ {issue}");
                    }
                }
            }

            return validation.IsValid ? 0 : 1;
        }

        static int Version()
        {
            Console.WriteLine("nslpm 1.2.0");
            Console.WriteLine("NSL Package Manager - Like pip but for NSL");
            Console.WriteLine("Using local file-based registry (works offline)");
            return 0;
        }

        /// <summary>
        /// Freeze - Output installed packages in requirements format (like pip freeze)
        /// </summary>
        static int Freeze(string[] args)
        {
            var projectDir = Directory.GetCurrentDirectory();
            using var pm = new NslPackageManager(projectDir);

            var packages = pm.List();

            if (packages.Count == 0)
            {
                // Output nothing - like pip freeze with no packages
                return 0;
            }

            // Check for output file option
            var outputFile = args.FirstOrDefault(a => !a.StartsWith("-"));
            var useStrictVersions = !args.Contains("--no-strict");

            var lines = new List<string>();
            foreach (var pkg in packages.OrderBy(p => p.Name))
            {
                // Output in requirements format: package==version
                lines.Add($"{pkg.Name}=={pkg.Version}");
            }

            if (outputFile != null)
            {
                File.WriteAllLines(outputFile, lines);
                PrintSuccess($"Wrote {packages.Count} packages to {outputFile}");
            }
            else
            {
                foreach (var line in lines)
                {
                    Console.WriteLine(line);
                }
            }

            return 0;
        }

        /// <summary>
        /// Config - Manage nslpm configuration (like pip config)
        /// </summary>
        static int Config(string[] args)
        {
            var subcommand = args.Length > 0 ? args[0].ToLowerInvariant() : "list";
            var subArgs = args.Skip(1).ToArray();

            var configPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".nsl", "config.json"
            );

            return subcommand switch
            {
                "list" => ConfigList(configPath),
                "get" => ConfigGet(configPath, subArgs),
                "set" => ConfigSet(configPath, subArgs),
                "unset" or "remove" => ConfigUnset(configPath, subArgs),
                "edit" => ConfigEdit(configPath),
                "path" => ConfigPath(configPath),
                _ => ConfigHelp()
            };
        }

        static int ConfigList(string configPath)
        {
            if (!File.Exists(configPath))
            {
                Console.WriteLine("No configuration file found.");
                Console.WriteLine($"Create one at: {configPath}");
                return 0;
            }

            var config = LoadConfig(configPath);

            Console.WriteLine($"Configuration ({configPath}):\n");

            foreach (var (key, value) in config.OrderBy(kvp => kvp.Key))
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"  {key}");
                Console.ForegroundColor = DefaultColor;
                Console.WriteLine($" = {value}");
            }

            return 0;
        }

        static int ConfigGet(string configPath, string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify a config key");
                return 1;
            }

            var key = args[0];
            var config = LoadConfig(configPath);

            if (config.TryGetValue(key, out var value))
            {
                Console.WriteLine(value);
                return 0;
            }

            PrintError($"Key not found: {key}");
            return 1;
        }

        static int ConfigSet(string configPath, string[] args)
        {
            if (args.Length < 2)
            {
                PrintError("Usage: nslpm config set <key> <value>");
                return 1;
            }

            var key = args[0];
            var value = string.Join(" ", args.Skip(1));

            var config = LoadConfig(configPath);
            config[key] = value;
            SaveConfig(configPath, config);

            PrintSuccess($"Set {key} = {value}");
            return 0;
        }

        static int ConfigUnset(string configPath, string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify a config key to remove");
                return 1;
            }

            var key = args[0];
            var config = LoadConfig(configPath);

            if (config.Remove(key))
            {
                SaveConfig(configPath, config);
                PrintSuccess($"Removed {key}");
                return 0;
            }

            PrintError($"Key not found: {key}");
            return 1;
        }

        static int ConfigEdit(string configPath)
        {
            // Ensure config file exists
            if (!File.Exists(configPath))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);
                File.WriteAllText(configPath, "{}");
            }

            // Try to open in default editor
            try
            {
                if (OperatingSystem.IsWindows())
                {
                    System.Diagnostics.Process.Start("notepad", configPath);
                }
                else if (OperatingSystem.IsMacOS())
                {
                    System.Diagnostics.Process.Start("open", configPath);
                }
                else
                {
                    var editor = Environment.GetEnvironmentVariable("EDITOR") ?? "nano";
                    System.Diagnostics.Process.Start(editor, configPath);
                }
                Console.WriteLine($"Opening {configPath}");
                return 0;
            }
            catch
            {
                Console.WriteLine($"Config file location: {configPath}");
                return 0;
            }
        }

        static int ConfigPath(string configPath)
        {
            Console.WriteLine(configPath);
            return 0;
        }

        static int ConfigHelp()
        {
            Console.WriteLine(@"
nslpm config - Configuration Management

Usage: nslpm config <command> [options]

Commands:
  list              Show all configuration values
  get <key>         Get a specific config value
  set <key> <val>   Set a config value
  unset <key>       Remove a config value
  edit              Open config file in editor
  path              Show config file path

Available Config Keys:
  cache.dir         Cache directory path
  registry.url      Package registry URL
  registry.local    Use local registry (true/false)
  install.timeout   Download timeout in seconds
  install.retries   Number of retry attempts

Examples:
  nslpm config list
  nslpm config set cache.dir /custom/cache
  nslpm config get registry.url
");
            return 0;
        }

        static Dictionary<string, string> LoadConfig(string path)
        {
            if (!File.Exists(path))
                return new Dictionary<string, string>();

            try
            {
                var json = File.ReadAllText(path);
                return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(json)
                    ?? new Dictionary<string, string>();
            }
            catch
            {
                return new Dictionary<string, string>();
            }
        }

        static void SaveConfig(string path, Dictionary<string, string> config)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var json = System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Hash - Compute hash of a package file (like pip hash)
        /// </summary>
        static int Hash(string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify a file to hash");
                Console.WriteLine("Usage: nslpm hash <file> [--algorithm sha256|sha512]");
                return 1;
            }

            var filePath = args[0];
            var algorithm = "sha256";

            var algIndex = Array.IndexOf(args, "--algorithm");
            if (algIndex >= 0 && algIndex < args.Length - 1)
            {
                algorithm = args[algIndex + 1].ToLowerInvariant();
            }
            else if (args.Contains("--sha512"))
            {
                algorithm = "sha512";
            }

            if (!File.Exists(filePath))
            {
                PrintError($"File not found: {filePath}");
                return 1;
            }

            try
            {
                using var stream = File.OpenRead(filePath);
                byte[] hashBytes;

                if (algorithm == "sha512")
                {
                    using var sha = System.Security.Cryptography.SHA512.Create();
                    hashBytes = sha.ComputeHash(stream);
                }
                else
                {
                    using var sha = System.Security.Cryptography.SHA256.Create();
                    hashBytes = sha.ComputeHash(stream);
                }

                var hashStr = Convert.ToHexString(hashBytes).ToLowerInvariant();
                Console.WriteLine($"--hash={algorithm}:{hashStr}");
                return 0;
            }
            catch (Exception ex)
            {
                PrintError($"Failed to compute hash: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Debug - Show debugging information (like pip debug)
        /// </summary>
        static int Debug()
        {
            Console.WriteLine("nslpm Debug Information\n");
            Console.WriteLine("=".PadRight(50, '='));

            // Version
            Console.WriteLine($"\nnslpm version: 1.2.0");
            Console.WriteLine($".NET version: {Environment.Version}");
            Console.WriteLine($"OS: {Environment.OSVersion}");
            Console.WriteLine($"Platform: {(Environment.Is64BitOperatingSystem ? "64-bit" : "32-bit")}");

            // Paths
            Console.WriteLine("\nPaths:");
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

            Console.WriteLine($"  Config:   {Path.Combine(userProfile, ".nsl", "config.json")}");
            Console.WriteLine($"  Cache:    {Path.Combine(localAppData, "nsl", "cache")}");
            Console.WriteLine($"  Registry: {Path.Combine(userProfile, ".nsl", "registry")}");
            Console.WriteLine($"  Packages: {Path.Combine(localAppData, "nsl", "packages")}");

            // Current project
            var projectDir = Directory.GetCurrentDirectory();
            var manifestPath = Path.Combine(projectDir, "nsl-package.json");
            Console.WriteLine($"\nCurrent directory: {projectDir}");
            Console.WriteLine($"Has manifest: {File.Exists(manifestPath)}");

            if (File.Exists(manifestPath))
            {
                try
                {
                    var manifest = PackageManifest.Load(manifestPath);
                    Console.WriteLine($"Project: {manifest.Name}@{manifest.Version}");
                }
                catch { }
            }

            // Check registry
            var registryPath = Path.Combine(userProfile, ".nsl", "registry", "index.json");
            Console.WriteLine($"\nRegistry:");
            Console.WriteLine($"  Local registry exists: {File.Exists(registryPath)}");

            if (File.Exists(registryPath))
            {
                try
                {
                    var indexContent = File.ReadAllText(registryPath);
                    var index = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(indexContent);
                    Console.WriteLine($"  Packages in registry: {index?.Count ?? 0}");
                }
                catch { }
            }

            // Cache info
            var cacheDir = Path.Combine(localAppData, "nsl", "cache");
            Console.WriteLine($"\nCache:");
            if (Directory.Exists(cacheDir))
            {
                var files = Directory.GetFiles(cacheDir);
                var size = files.Sum(f => new FileInfo(f).Length);
                Console.WriteLine($"  Cached packages: {files.Length}");
                Console.WriteLine($"  Total size: {FormatSize(size)}");
            }
            else
            {
                Console.WriteLine("  No cache directory");
            }

            Console.WriteLine();
            return 0;
        }

        static async Task<int> RunScriptAsync(string[] args)
        {
            if (args.Length == 0)
            {
                // Show available scripts
                return ListScripts();
            }

            var projectDir = Directory.GetCurrentDirectory();
            var scriptName = args[0];
            var scriptArgs = args.Skip(1).ToArray();

            try
            {
                var runner = new ScriptRunner(projectDir);
                Console.WriteLine($"\n> {scriptName}\n");

                var result = await runner.RunAsync(scriptName, scriptArgs);

                if (!result.Success)
                {
                    PrintError($"Script '{scriptName}' failed with exit code {result.ExitCode}");
                    return result.ExitCode;
                }

                return 0;
            }
            catch (FileNotFoundException)
            {
                PrintError("No nsl-package.json found. Run 'nslpm init' first.");
                return 1;
            }
        }

        static async Task<int> ExecAsync(string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify a command to execute");
                return 1;
            }

            var projectDir = Directory.GetCurrentDirectory();
            var command = string.Join(" ", args);

            try
            {
                var runner = new ScriptRunner(projectDir);
                Console.WriteLine($"\n> {command}\n");

                var result = await runner.ExecAsync(command);
                return result.ExitCode;
            }
            catch (FileNotFoundException)
            {
                PrintError("No nsl-package.json found. Run 'nslpm init' first.");
                return 1;
            }
        }

        static int ListScripts()
        {
            var projectDir = Directory.GetCurrentDirectory();

            try
            {
                var runner = new ScriptRunner(projectDir);
                var scripts = runner.ListScripts();

                if (scripts.Count == 0)
                {
                    Console.WriteLine("No scripts defined in nsl-package.json");
                    Console.WriteLine("\nAdd scripts to your package manifest:");
                    Console.WriteLine("  \"scripts\": {");
                    Console.WriteLine("    \"test\": \"nsl test.nsl\",");
                    Console.WriteLine("    \"build\": \"nsl build.nsl\"");
                    Console.WriteLine("  }");
                    return 0;
                }

                Console.WriteLine("Available scripts:\n");

                foreach (var script in scripts)
                {
                    Console.ForegroundColor = ConsoleColor.Cyan;
                    Console.Write($"  {script.Name}");
                    Console.ForegroundColor = DefaultColor;

                    if (script.HasPreScript || script.HasPostScript)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        var hooks = new List<string>();
                        if (script.HasPreScript) hooks.Add("pre");
                        if (script.HasPostScript) hooks.Add("post");
                        Console.Write($" [{string.Join(", ", hooks)}]");
                        Console.ForegroundColor = DefaultColor;
                    }

                    Console.WriteLine();

                    if (_verbose)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"    {script.Command}");
                        Console.ForegroundColor = DefaultColor;
                    }
                }

                Console.WriteLine("\nRun a script with: nslpm run <script-name>");
                return 0;
            }
            catch (FileNotFoundException)
            {
                PrintError("No nsl-package.json found. Run 'nslpm init' first.");
                return 1;
            }
        }

        static int VenvCommand(string[] args)
        {
            var subcommand = args.Length > 0 ? args[0].ToLowerInvariant() : "help";
            var subArgs = args.Skip(1).ToArray();

            return subcommand switch
            {
                "create" or "new" => VenvCreate(subArgs),
                "activate" => VenvActivate(subArgs),
                "deactivate" => VenvDeactivate(),
                "list" or "ls" => VenvList(),
                "delete" or "rm" => VenvDelete(subArgs),
                "info" => VenvInfo(),
                _ => VenvHelp()
            };
        }

        static int VenvCreate(string[] args)
        {
            var path = args.Length > 0 ? args[0] : ".venv";
            var name = args.Length > 1 ? args[1] : null;

            try
            {
                var venv = VirtualEnv.Create(path, name);
                PrintSuccess($"Created virtual environment: {venv.Name}");
                Console.WriteLine($"  Path: {venv.Path}");
                Console.WriteLine();
                Console.WriteLine("To activate:");

                if (OperatingSystem.IsWindows())
                {
                    Console.WriteLine($"  PowerShell: .\\{path}\\bin\\activate.ps1");
                    Console.WriteLine($"  CMD:        .\\{path}\\bin\\activate.bat");
                }
                else
                {
                    Console.WriteLine($"  source {path}/bin/activate");
                }

                return 0;
            }
            catch (Exception ex)
            {
                PrintError(ex.Message);
                return 1;
            }
        }

        static int VenvActivate(string[] args)
        {
            var path = args.Length > 0 ? args[0] : ".venv";

            if (!VirtualEnv.IsVirtualEnv(path))
            {
                PrintError($"Not a virtual environment: {path}");
                return 1;
            }

            // We can't actually activate from within C# - we need to tell the user how
            Console.WriteLine("To activate the virtual environment, run:");
            Console.WriteLine();

            if (OperatingSystem.IsWindows())
            {
                Console.WriteLine($"  PowerShell: .\\{path}\\bin\\activate.ps1");
                Console.WriteLine($"  CMD:        .\\{path}\\bin\\activate.bat");
            }
            else
            {
                Console.WriteLine($"  source {path}/bin/activate");
            }

            return 0;
        }

        static int VenvDeactivate()
        {
            var active = VirtualEnv.GetActive();
            if (active == null)
            {
                Console.WriteLine("No virtual environment is currently active");
                return 0;
            }

            Console.WriteLine("To deactivate the virtual environment, run:");
            Console.WriteLine();

            if (OperatingSystem.IsWindows())
            {
                Console.WriteLine("  PowerShell: deactivate.ps1");
                Console.WriteLine("  CMD:        deactivate.bat");
            }
            else
            {
                Console.WriteLine("  source deactivate");
            }

            return 0;
        }

        static int VenvList()
        {
            var searchPaths = new[]
            {
                ".venv",
                "venv",
                ".nsl-venv",
                "nsl-venv"
            };

            var found = new List<VirtualEnv>();
            var currentDir = Directory.GetCurrentDirectory();

            foreach (var searchPath in searchPaths)
            {
                var fullPath = Path.Combine(currentDir, searchPath);
                if (VirtualEnv.IsVirtualEnv(fullPath))
                {
                    found.Add(VirtualEnv.Load(fullPath));
                }
            }

            if (found.Count == 0)
            {
                Console.WriteLine("No virtual environments found in current directory");
                Console.WriteLine("\nCreate one with: nslpm venv create");
                return 0;
            }

            Console.WriteLine("Virtual environments:\n");

            foreach (var venv in found)
            {
                Console.ForegroundColor = venv.IsActive ? ConsoleColor.Green : ConsoleColor.Cyan;
                Console.Write($"  {venv.Name}");
                Console.ForegroundColor = DefaultColor;

                if (venv.IsActive)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write(" (active)");
                    Console.ForegroundColor = DefaultColor;
                }

                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"    {venv.Path}");
                Console.ForegroundColor = DefaultColor;
            }

            return 0;
        }

        static int VenvDelete(string[] args)
        {
            if (args.Length == 0)
            {
                PrintError("Please specify the virtual environment path to delete");
                return 1;
            }

            var path = args[0];

            if (!VirtualEnv.IsVirtualEnv(path))
            {
                PrintError($"Not a virtual environment: {path}");
                return 1;
            }

            try
            {
                var venv = VirtualEnv.Load(path);
                venv.Delete();
                PrintSuccess($"Deleted virtual environment: {path}");
                return 0;
            }
            catch (Exception ex)
            {
                PrintError(ex.Message);
                return 1;
            }
        }

        static int VenvInfo()
        {
            var active = VirtualEnv.GetActive();

            if (active == null)
            {
                Console.WriteLine("No virtual environment is currently active");
                return 0;
            }

            Console.WriteLine($"Active virtual environment:\n");
            Console.WriteLine($"  Name:     {active.Name}");
            Console.WriteLine($"  Path:     {active.Path}");
            Console.WriteLine($"  Packages: {active.PackagesPath}");
            Console.WriteLine($"  Bin:      {active.BinPath}");

            var packages = active.ListPackages();
            if (packages.Count > 0)
            {
                Console.WriteLine($"\nInstalled packages ({packages.Count}):");
                foreach (var pkg in packages.OrderBy(p => p.Name))
                {
                    Console.WriteLine($"  {pkg.Name}@{pkg.Version}");
                }
            }

            return 0;
        }

        static int VenvHelp()
        {
            Console.WriteLine(@"
nslpm venv - Virtual Environment Management

Usage: nslpm venv <command> [options]

Commands:
  create, new [path] [name]  Create a new virtual environment
  activate [path]            Show activation command
  deactivate                 Show deactivation command
  list, ls                   List virtual environments
  delete, rm <path>          Delete a virtual environment
  info                       Show active environment info

Examples:
  nslpm venv create                Create .venv in current directory
  nslpm venv create myenv myproj   Create myenv with name 'myproj'
  nslpm venv list                  List all environments
  nslpm venv info                  Show active environment details
");
            return 0;
        }

        static int UnknownCommand(string command)
        {
            PrintError($"Unknown command: {command}");
            Console.WriteLine("Run 'nslpm --help' for usage information");
            return 1;
        }

        static void PrintUsage()
        {
            Console.WriteLine(@"
nslpm - NSL Package Manager v1.2.0

Usage: nslpm <command> [options]

Commands:
  init [name] [version]     Initialize a new NSL project
  install, i, add [pkg...]  Install packages (or all from manifest)
  uninstall, remove, rm     Remove packages
  update, upgrade [pkg...]  Update packages
  list, ls                  List installed packages
  freeze [file]             Output installed packages in requirements format
  search, s <query>         Search for packages
  info, show <package>      Show package details
  outdated                  Check for outdated packages
  run <script> [args...]    Run a script from package.json
  exec, x <command>         Execute a command in project context
  scripts                   List available scripts
  pack [output-dir]         Pack project for publishing
  publish                   Publish package to local registry
  cache <subcommand>        Manage package cache (list|clean|dir|rm)
  tree                      Show dependency tree
  validate, check           Validate package manifest
  config <subcommand>       Configuration (list|get|set|unset|edit|path)
  hash <file>               Compute SHA256/SHA512 hash of a file
  debug                     Show debugging information
  venv <subcommand>         Manage virtual environments
  version, --version        Show version

Install Options:
  -D, --save-dev            Save as dev dependency
  -n, --dry-run             Don't actually install, just show what would happen
  -r, --requirements <file> Install from requirements file
  -e, --editable <path>     Install package in editable/dev mode (symlink)
  -U, --upgrade             Upgrade packages to latest version
  -q, --quiet               Suppress output
  -f, --force-reinstall     Reinstall even if already installed

Global Options:
  -v, --verbose             Verbose output
  -h, --help                Show help

Cache Subcommands:
  cache list                List cached packages with sizes
  cache clean               Remove all cached packages
  cache dir                 Show cache directory path
  cache rm <pattern>        Remove specific cached packages

Config Subcommands:
  config list               Show all config values
  config get <key>          Get a config value
  config set <key> <val>    Set a config value
  config unset <key>        Remove a config value
  config edit               Open config in editor
  config path               Show config file path

Registry:
  Packages are stored locally in ~/.nsl/registry/
  No remote server required - works completely offline!

Examples:
  nslpm init my-package
  nslpm install lodash
  nslpm install lodash@^1.0.0
  nslpm install -r requirements.nsl
  nslpm install -e .                      # Install current dir as editable
  nslpm install --dry-run axios           # Preview what would be installed
  nslpm install --save-dev test-framework
  nslpm freeze > requirements.nsl         # Export installed packages
  nslpm search http client
  nslpm info lodash
  nslpm outdated
  nslpm update
  nslpm hash package.nslpkg
  nslpm config set cache.dir /custom/path
");
        }

        static void OnPackageEvent(object? sender, PackageEventArgs e)
        {
            var color = e.Type switch
            {
                PackageEventType.Installed or PackageEventType.Published => ConsoleColor.Green,
                PackageEventType.Installing or PackageEventType.Uninstalling => ConsoleColor.Cyan,
                PackageEventType.Failed => ConsoleColor.Red,
                PackageEventType.AlreadyInstalled => ConsoleColor.DarkGray,
                _ => DefaultColor
            };

            var symbol = e.Type switch
            {
                PackageEventType.Installed => "✓",
                PackageEventType.Installing => "↓",
                PackageEventType.Uninstalled => "✗",
                PackageEventType.Uninstalling => "↑",
                PackageEventType.AlreadyInstalled => "=",
                PackageEventType.Published => "✓",
                PackageEventType.Failed => "✗",
                _ => "•"
            };

            Console.ForegroundColor = color;
            Console.Write($"  {symbol} ");
            Console.ForegroundColor = DefaultColor;

            var message = e.Type switch
            {
                PackageEventType.Installing => $"Installing {e.PackageName}",
                PackageEventType.Installed => $"Installed {e.PackageName}@{e.Version}",
                PackageEventType.AlreadyInstalled => $"Already installed {e.PackageName}@{e.Version}",
                PackageEventType.Uninstalling => $"Removing {e.PackageName}",
                PackageEventType.Uninstalled => $"Removed {e.PackageName}",
                PackageEventType.Packed => $"Packed {e.PackageName}@{e.Version}",
                PackageEventType.Published => $"Published {e.PackageName}@{e.Version}",
                PackageEventType.Failed => $"Failed: {e.Version}",
                PackageEventType.Initialized => $"Initialized {e.PackageName}@{e.Version}",
                _ => $"{e.Type}: {e.PackageName}"
            };

            Console.WriteLine(message);
        }

        static void PrintSuccess(string message)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ {message}");
            Console.ForegroundColor = DefaultColor;
        }

        static void PrintError(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"✗ {message}");
            Console.ForegroundColor = DefaultColor;
        }

        static void PrintWarning(string message)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"⚠ {message}");
            Console.ForegroundColor = DefaultColor;
        }

        static string FormatSize(long bytes)
        {
            string[] suffixes = { "B", "KB", "MB", "GB" };
            int i = 0;
            double size = bytes;
            while (size >= 1024 && i < suffixes.Length - 1)
            {
                size /= 1024;
                i++;
            }
            return $"{size:0.##} {suffixes[i]}";
        }
    }
}
