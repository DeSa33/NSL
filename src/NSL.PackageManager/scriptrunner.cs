using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSL.PackageManager
{
    /// <summary>
    /// Runs scripts defined in the package manifest
    /// </summary>
    public class ScriptRunner
    {
        private readonly string _projectRoot;
        private readonly PackageManifest _manifest;
        private readonly Dictionary<string, string> _environmentVariables;

        /// <summary>Public API</summary>
        public ScriptRunner(string projectRoot)
        {
            _projectRoot = Path.GetFullPath(projectRoot);
            _manifest = PackageManifest.LoadFromDirectory(_projectRoot);
            _environmentVariables = new Dictionary<string, string>();

            SetupEnvironment();
        }

        /// <summary>
        /// Available scripts
        /// </summary>
        public IReadOnlyDictionary<string, string> Scripts => _manifest.Scripts;

        /// <summary>
        /// Run a script by name
        /// </summary>
        public async Task<ScriptResult> RunAsync(string scriptName, string[]? args = null)
        {
            if (!_manifest.Scripts.TryGetValue(scriptName, out var command))
            {
                // Check for common lifecycle scripts
                var lifecycleCommand = GetLifecycleCommand(scriptName);
                if (lifecycleCommand == null)
                {
                    return new ScriptResult(
                        scriptName,
                        false,
                        -1,
                        "",
                        $"Script '{scriptName}' not found in package manifest"
                    );
                }
                command = lifecycleCommand;
            }

            // Append args to command
            if (args != null && args.Length > 0)
            {
                command = $"{command} {string.Join(" ", args.Select(EscapeArg))}";
            }

            // Run pre-script if exists
            var preScriptName = $"pre{scriptName}";
            if (_manifest.Scripts.ContainsKey(preScriptName))
            {
                var preResult = await RunAsync(preScriptName);
                if (!preResult.Success)
                {
                    return preResult;
                }
            }

            // Run the main script
            var result = await ExecuteCommandAsync(scriptName, command);

            // Run post-script if exists and main succeeded
            if (result.Success)
            {
                var postScriptName = $"post{scriptName}";
                if (_manifest.Scripts.ContainsKey(postScriptName))
                {
                    await RunAsync(postScriptName);
                }
            }

            return result;
        }

        /// <summary>
        /// Run an arbitrary command in the project context
        /// </summary>
        public async Task<ScriptResult> ExecAsync(string command)
        {
            return await ExecuteCommandAsync("exec", command);
        }

        /// <summary>
        /// List available scripts
        /// </summary>
        public List<ScriptInfo> ListScripts()
        {
            var scripts = new List<ScriptInfo>();

            foreach (var (name, command) in _manifest.Scripts)
            {
                scripts.Add(new ScriptInfo
                {
                    Name = name,
                    Command = command,
                    HasPreScript = _manifest.Scripts.ContainsKey($"pre{name}"),
                    HasPostScript = _manifest.Scripts.ContainsKey($"post{name}")
                });
            }

            return scripts.OrderBy(s => s.Name).ToList();
        }

        private async Task<ScriptResult> ExecuteCommandAsync(string scriptName, string command)
        {
            var isWindows = OperatingSystem.IsWindows();
            var shell = isWindows ? "cmd.exe" : "/bin/sh";
            var shellArgs = isWindows ? $"/c {command}" : $"-c \"{command}\"";

            var startInfo = new ProcessStartInfo
            {
                FileName = shell,
                Arguments = shellArgs,
                WorkingDirectory = _projectRoot,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            // Add environment variables
            foreach (var (key, value) in _environmentVariables)
            {
                startInfo.Environment[key] = value;
            }

            var stdout = new StringBuilder();
            var stderr = new StringBuilder();

            try
            {
                using var process = new Process { StartInfo = startInfo };

                process.OutputDataReceived += (_, e) =>
                {
                    if (e.Data != null)
                    {
                        stdout.AppendLine(e.Data);
                        Console.WriteLine(e.Data);
                    }
                };

                process.ErrorDataReceived += (_, e) =>
                {
                    if (e.Data != null)
                    {
                        stderr.AppendLine(e.Data);
                        Console.Error.WriteLine(e.Data);
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await process.WaitForExitAsync();

                return new ScriptResult(
                    scriptName,
                    process.ExitCode == 0,
                    process.ExitCode,
                    stdout.ToString(),
                    stderr.ToString()
                );
            }
            catch (Exception ex)
            {
                return new ScriptResult(
                    scriptName,
                    false,
                    -1,
                    "",
                    ex.Message
                );
            }
        }

        private void SetupEnvironment()
        {
            // Add nsl_packages/.bin to PATH
            var binPath = Path.Combine(_projectRoot, "nsl_packages", ".bin");
            var currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            _environmentVariables["PATH"] = $"{binPath}{Path.PathSeparator}{currentPath}";

            // Add NSL-specific variables
            _environmentVariables["NSL_PACKAGE_ROOT"] = _projectRoot;
            _environmentVariables["NSL_PACKAGE_NAME"] = _manifest.Name;
            _environmentVariables["NSL_PACKAGE_VERSION"] = _manifest.Version;

            // Add custom metadata as environment variables
            foreach (var (key, value) in _manifest.Metadata)
            {
                var envKey = $"NSL_{key.ToUpperInvariant().Replace('-', '_')}";
                _environmentVariables[envKey] = value?.ToString() ?? "";
            }
        }

        private string? GetLifecycleCommand(string scriptName)
        {
            return scriptName switch
            {
                "start" => $"nsl {_manifest.Main}",
                "test" => null, // No default test command
                "build" => null,
                "prepare" => null,
                _ => null
            };
        }

        private static string EscapeArg(string arg)
        {
            if (arg.Contains(' ') || arg.Contains('"'))
            {
                return $"\"{arg.Replace("\"", "\\\"")}\"";
            }
            return arg;
        }
    }

    /// <summary>
    /// Result of running a script
    /// </summary>
    public class ScriptResult
    {
        /// <summary>Public API</summary>
        public string ScriptName { get; }
        /// <summary>Public API</summary>
        public bool Success { get; }
        /// <summary>Public API</summary>
        public int ExitCode { get; }
        /// <summary>Public API</summary>
        public string Stdout { get; }
        /// <summary>Public API</summary>
        public string Stderr { get; }

        /// <summary>Public API</summary>
        public ScriptResult(string scriptName, bool success, int exitCode, string stdout, string stderr)
        {
            ScriptName = scriptName;
            Success = success;
            ExitCode = exitCode;
            Stdout = stdout;
            Stderr = stderr;
        }
    }

    /// <summary>
    /// Information about a script
    /// </summary>
    public class ScriptInfo
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string Command { get; set; } = "";
        /// <summary>Public API</summary>
        public bool HasPreScript { get; set; }
        /// <summary>Public API</summary>
        public bool HasPostScript { get; set; }
    }
}