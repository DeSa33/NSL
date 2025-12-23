using System;
using System.IO;
using NSL.Debugger;

namespace NSL.Debugger
{
    /// <summary>
    /// Entry point for the NSL Debug Adapter.
    /// Implements the Debug Adapter Protocol (DAP) for debugging NSL programs.
    /// </summary>
    public class Program
    {
        public static void Main(string[] args)
        {
            // Parse command line arguments
            var options = ParseArgs(args);

            if (options.ShowHelp)
            {
                ShowHelp();
                return;
            }

            if (options.ShowVersion)
            {
                Console.WriteLine("NSL Debug Adapter v1.0.0");
                return;
            }

            // Create and run the debug adapter
            var adapter = new NslDebugAdapter(
                Console.OpenStandardInput(),
                Console.OpenStandardOutput(),
                options
            );

            // Run the adapter (blocks until session ends)
            adapter.Run();
        }

        private static DebugAdapterOptions ParseArgs(string[] args)
        {
            var options = new DebugAdapterOptions();

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--help":
                    case "-h":
                        options.ShowHelp = true;
                        break;
                    case "--version":
                    case "-v":
                        options.ShowVersion = true;
                        break;
                    case "--trace":
                        options.TraceLevel = TraceLevel.Verbose;
                        break;
                    case "--log":
                        if (i + 1 < args.Length)
                        {
                            options.LogFile = args[++i];
                        }
                        break;
                    case "--port":
                        if (i + 1 < args.Length && int.TryParse(args[++i], out var port))
                        {
                            options.Port = port;
                            options.UseSocket = true;
                        }
                        break;
                }
            }

            return options;
        }

        private static void ShowHelp()
        {
            Console.WriteLine(@"
NSL Debug Adapter - Debug Adapter Protocol implementation for NSL

Usage: nsl-debug-adapter [options]

Options:
  --help, -h       Show this help message
  --version, -v    Show version information
  --trace          Enable verbose tracing
  --log <file>     Log debug output to file
  --port <number>  Use TCP socket instead of stdio

The debug adapter communicates via the Debug Adapter Protocol (DAP)
and can be used with VSCode, Vim, Emacs, and other DAP-compatible editors.

For more information, see: https://microsoft.github.io/debug-adapter-protocol/
");
        }
    }

    /// <summary>
    /// Debug adapter options
    /// </summary>
    public class DebugAdapterOptions
    {
        public bool ShowHelp { get; set; }
        public bool ShowVersion { get; set; }
        public TraceLevel TraceLevel { get; set; } = TraceLevel.None;
        public string? LogFile { get; set; }
        public bool UseSocket { get; set; }
        public int Port { get; set; } = 4711;
    }

    /// <summary>
    /// Trace level for debug output
    /// </summary>
    public enum TraceLevel
    {
        None,
        Messages,
        Verbose
    }
}
