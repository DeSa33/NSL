using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NSL.Console;

/// <summary>
/// Output formatting modes for human-readable display
/// Based on CLI UX best practices research
/// </summary>
public enum OutputMode
{
    /// <summary>Summary mode - concise output with icons (default for humans)</summary>
    Summary,
    /// <summary>Verbose mode - full detailed output</summary>
    Verbose,
    /// <summary>JSON mode - machine-readable (for AI)</summary>
    Json,
    /// <summary>Quiet mode - minimal output</summary>
    Quiet
}

/// <summary>
/// Formats NSL output for optimal human readability
/// Features:
/// - Summary/verbose modes
/// - Spinners for long operations
/// - Progress indicators
/// - Log file support
/// - Timing information
/// </summary>
public class OutputFormatter : IDisposable
{
    private readonly OutputMode _mode;
    private readonly bool _useColors;
    private readonly StreamWriter? _logWriter;
    private readonly string? _logPath;
    private readonly Stopwatch _stopwatch;
    private readonly List<string> _fullLog = new();
    private CancellationTokenSource? _spinnerCts;
    private Task? _spinnerTask;
    private string _currentSpinnerMessage = "";
    
    // Spinner frames (Braille pattern - smooth animation)
    private static readonly string[] SpinnerFrames = { "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" };
    private static readonly string[] SpinnerFramesAscii = { "|", "/", "-", "\\" };
    
    public OutputFormatter(OutputMode mode = OutputMode.Summary, bool enableLogging = false)
    {
        _mode = mode;
        _useColors = ColorOutput.UseColors;
        _stopwatch = Stopwatch.StartNew();
        
        if (enableLogging)
        {
            try
            {
                var logDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    ".nsl", "logs"
                );
                Directory.CreateDirectory(logDir);
                _logPath = Path.Combine(logDir, $"nsl-{DateTime.Now:yyyy-MM-dd-HHmmss}.log");
                _logWriter = new StreamWriter(_logPath, false, Encoding.UTF8);
                _logWriter.WriteLine($"NSL Log - {DateTime.Now:F}");
                _logWriter.WriteLine(new string('=', 60));
                _logWriter.WriteLine();
            }
            catch
            {
                // Logging is optional, don't fail if it doesn't work
            }
        }
    }
    
    /// <summary>
    /// Get the path to the current log file
    /// </summary>
    public string? LogPath => _logPath;
    
    /// <summary>
    /// Start a spinner with a message (for long operations)
    /// </summary>
    public void StartSpinner(string message)
    {
        if (_mode == OutputMode.Json || _mode == OutputMode.Quiet) return;
        
        _currentSpinnerMessage = message;
        _spinnerCts = new CancellationTokenSource();
        var token = _spinnerCts.Token;
        var frames = _useColors ? SpinnerFrames : SpinnerFramesAscii;
        
        _spinnerTask = Task.Run(async () =>
        {
            int frame = 0;
            while (!token.IsCancellationRequested)
            {
                var spinner = _useColors 
                    ? ColorOutput.Colorize(frames[frame], ColorConfig.Instance.Info)
                    : frames[frame];
                System.Console.Write($"\r{spinner} {_currentSpinnerMessage}...");
                frame = (frame + 1) % frames.Length;
                try { await Task.Delay(80, token); } catch { break; }
            }
        }, token);
    }
    
    /// <summary>
    /// Stop the spinner and show completion status
    /// </summary>
    public void StopSpinner(bool success = true, string? completedMessage = null)
    {
        if (_spinnerCts == null) return;
        
        _spinnerCts.Cancel();
        try { _spinnerTask?.Wait(100); } catch { }
        
        // Clear the spinner line
        System.Console.Write("\r" + new string(' ', _currentSpinnerMessage.Length + 10) + "\r");
        
        // Show completion status (transition from -ing to -ed)
        var msg = completedMessage ?? _currentSpinnerMessage.Replace("ing", "ed");
        if (success)
            ColorOutput.Success(msg);
        else
            ColorOutput.Error(msg);
        
        _spinnerCts = null;
        _spinnerTask = null;
    }
    
    /// <summary>
    /// Print a section header
    /// </summary>
    public void Section(string title)
    {
        if (_mode == OutputMode.Quiet) return;
        
        Log($"\n=== {title} ===");
        
        if (_mode == OutputMode.Json) return;
        
        System.Console.WriteLine();
        var line = new string('─', Math.Min(title.Length + 4, 60));
        System.Console.WriteLine(ColorOutput.Colorize($"┌{line}┐", ColorConfig.Instance.Info));
        System.Console.WriteLine(ColorOutput.Colorize($"│ {title.PadRight(line.Length - 2)} │", ColorConfig.Instance.Info));
        System.Console.WriteLine(ColorOutput.Colorize($"└{line}┘", ColorConfig.Instance.Info));
    }
    
    /// <summary>
    /// Print a key-value pair
    /// </summary>
    public void KeyValue(string key, object? value, bool important = false)
    {
        Log($"{key}: {value}");
        
        if (_mode == OutputMode.Quiet) return;
        if (_mode == OutputMode.Summary && !important) return;
        
        var keyColor = important ? ColorConfig.Instance.Warning : ColorConfig.Instance.Info;
        var valueStr = value?.ToString() ?? "null";
        
        System.Console.WriteLine($"  {ColorOutput.Colorize(key + ":", keyColor)} {valueStr}");
    }
    
    /// <summary>
    /// Print a list of items
    /// </summary>
    public void List(string title, IEnumerable<string> items, int maxItems = 5)
    {
        var itemList = new List<string>(items);
        Log($"{title}: {string.Join(", ", itemList)}");
        
        if (_mode == OutputMode.Quiet) return;
        
        System.Console.WriteLine($"  {ColorOutput.Colorize(title + ":", ColorConfig.Instance.Info)}");
        
        int shown = 0;
        foreach (var item in itemList)
        {
            if (_mode == OutputMode.Summary && shown >= maxItems)
            {
                var remaining = itemList.Count - maxItems;
                System.Console.WriteLine(ColorOutput.Colorize($"    ... and {remaining} more (use --verbose to see all)", ColorConfig.Instance.Hint));
                break;
            }
            System.Console.WriteLine($"    • {item}");
            shown++;
        }
    }
    
    /// <summary>
    /// Print execution result
    /// </summary>
    public void Result(object? value, long? executionMs = null)
    {
        var elapsed = executionMs ?? _stopwatch.ElapsedMilliseconds;
        Log($"Result: {value} (in {elapsed}ms)");
        
        if (_mode == OutputMode.Quiet) return;
        
        if (_mode == OutputMode.Json)
        {
            // JSON output handled elsewhere
            return;
        }
        
        // Show result with timing
        var timing = elapsed < 1000 
            ? $"{elapsed}ms" 
            : $"{elapsed / 1000.0:F2}s";
        
        System.Console.WriteLine();
        System.Console.WriteLine(ColorOutput.Colorize($"  Result: ", ColorConfig.Instance.Success) + 
            ColorOutput.Colorize(FormatValue(value), ColorConfig.Instance.Result));
        System.Console.WriteLine(ColorOutput.Colorize($"  Time:   {timing}", ColorConfig.Instance.Hint));
    }
    
    /// <summary>
    /// Print an error with optional code context
    /// </summary>
    public void Error(string message, string? code = null, int? line = null)
    {
        Log($"ERROR: {message}");
        
        if (_mode == OutputMode.Json)
        {
            System.Console.WriteLine($"{{\"error\": true, \"message\": \"{EscapeJson(message)}\"}}");
            return;
        }
        
        ColorOutput.Error(message);
        
        // Show code context in verbose mode or if we have line info
        if (!string.IsNullOrEmpty(code) && line.HasValue && _mode != OutputMode.Quiet)
        {
            ShowCodeContext(code, line.Value);
        }
    }
    
    /// <summary>
    /// Print a warning
    /// </summary>
    public void Warning(string message)
    {
        Log($"WARNING: {message}");
        if (_mode != OutputMode.Quiet && _mode != OutputMode.Json)
            ColorOutput.Warning(message);
    }
    
    /// <summary>
    /// Print success message
    /// </summary>
    public void Success(string message)
    {
        Log($"SUCCESS: {message}");
        if (_mode != OutputMode.Quiet && _mode != OutputMode.Json)
            ColorOutput.Success(message);
    }
    
    /// <summary>
    /// Print info message
    /// </summary>
    public void Info(string message)
    {
        Log($"INFO: {message}");
        if (_mode == OutputMode.Verbose)
            ColorOutput.Info(message);
    }
    
    /// <summary>
    /// Print a hint (only in verbose mode)
    /// </summary>
    public void Hint(string message)
    {
        Log($"HINT: {message}");
        if (_mode == OutputMode.Verbose)
            ColorOutput.Hint(message);
    }
    
    /// <summary>
    /// Show code context around an error line
    /// </summary>
    private void ShowCodeContext(string code, int errorLine)
    {
        var lines = code.Split('\n');
        var start = Math.Max(0, errorLine - 3);
        var end = Math.Min(lines.Length, errorLine + 2);
        
        System.Console.WriteLine();
        for (int i = start; i < end; i++)
        {
            var isErrorLine = i == errorLine - 1;
            var lineNum = ColorOutput.Colorize($"{i + 1,4} │ ", ColorConfig.Instance.LineNumber);
            var prefix = isErrorLine ? ColorOutput.Colorize("→ ", ColorConfig.Instance.Error) : "  ";
            var lineContent = isErrorLine
                ? ColorOutput.Colorize(lines[i].TrimEnd('\r'), ColorConfig.Instance.Error)
                : ColorOutput.HighlightSyntax(lines[i].TrimEnd('\r'));
            
            System.Console.WriteLine($"{prefix}{lineNum}{lineContent}");
        }
        System.Console.WriteLine();
    }
    
    /// <summary>
    /// Format a value for display
    /// </summary>
    private string FormatValue(object? value)
    {
        if (value == null) return "null";
        if (value is string s) return $"\"{s}\"";
        if (value is bool b) return b ? "true" : "false";
        if (value is IList<object?> list)
        {
            if (list.Count <= 5 || _mode == OutputMode.Verbose)
                return "[" + string.Join(", ", list.Select(FormatValue)) + "]";
            return $"[{FormatValue(list[0])}, {FormatValue(list[1])}, ... ({list.Count} items)]";
        }
        if (value is IDictionary<string, object?> dict)
        {
            if (dict.Count <= 3 || _mode == OutputMode.Verbose)
            {
                var pairs = dict.Select(kvp => $"{kvp.Key}: {FormatValue(kvp.Value)}");
                return "{" + string.Join(", ", pairs) + "}";
            }
            return $"{{... ({dict.Count} keys)}}";
        }
        return value.ToString() ?? "null";
    }
    
    /// <summary>
    /// Log to file
    /// </summary>
    private void Log(string message)
    {
        var timestamped = $"[{DateTime.Now:HH:mm:ss.fff}] {message}";
        _fullLog.Add(timestamped);
        _logWriter?.WriteLine(timestamped);
    }
    
    /// <summary>
    /// Show summary at the end
    /// </summary>
    public void ShowSummary(bool success, int? errorCount = null, int? warningCount = null)
    {
        if (_mode == OutputMode.Quiet || _mode == OutputMode.Json) return;
        
        var elapsed = _stopwatch.ElapsedMilliseconds;
        var timing = elapsed < 1000 ? $"{elapsed}ms" : $"{elapsed / 1000.0:F2}s";
        
        System.Console.WriteLine();
        System.Console.WriteLine(new string('─', 40));
        
        if (success)
        {
            ColorOutput.Success($"Completed in {timing}");
        }
        else
        {
            var summary = new StringBuilder();
            if (errorCount > 0) summary.Append($"{errorCount} error(s)");
            if (warningCount > 0)
            {
                if (summary.Length > 0) summary.Append(", ");
                summary.Append($"{warningCount} warning(s)");
            }
            ColorOutput.Error($"Failed with {summary} in {timing}");
        }
        
        // Show log file path hint
        if (_logPath != null)
        {
            ColorOutput.Hint($"Full log: {_logPath}");
        }
    }
    
    /// <summary>
    /// Get full log contents (for viewing later)
    /// </summary>
    public string GetFullLog() => string.Join("\n", _fullLog);
    
    private static string EscapeJson(string s) => 
        s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
    
    public void Dispose()
    {
        _spinnerCts?.Cancel();
        _logWriter?.Flush();
        _logWriter?.Dispose();
    }
}

/// <summary>
/// Simple progress bar for known-length operations
/// </summary>
public class ProgressBar
{
    private readonly int _total;
    private readonly string _title;
    private int _current;
    private readonly int _width;
    
    public ProgressBar(int total, string title, int width = 30)
    {
        _total = total;
        _title = title;
        _width = width;
        _current = 0;
    }
    
    public void Update(int current, string? status = null)
    {
        _current = current;
        var percent = _total > 0 ? (double)_current / _total : 0;
        var filled = (int)(percent * _width);
        var empty = _width - filled;
        
        var bar = new string('█', filled) + new string('░', empty);
        var percentStr = $"{percent * 100:F0}%".PadLeft(4);
        var statusStr = status != null ? $" {status}" : "";
        
        System.Console.Write($"\r{_title} {ColorOutput.Colorize(bar, ColorConfig.Instance.Success)} {percentStr}{statusStr}");
    }
    
    public void Complete(string? message = null)
    {
        Update(_total);
        System.Console.WriteLine();
        if (message != null)
            ColorOutput.Success(message);
    }
}
