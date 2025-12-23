using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace NSL.Console;

/// <summary>
/// Interactive console with readline-like features:
/// - Command history (up/down arrows)
/// - Tab completion for namespaces and functions
/// - Line editing (left/right arrows, home/end, delete/backspace)
/// </summary>
public class InteractiveConsole
{
    private readonly List<string> _history = new();
    private int _historyIndex = -1;
    private readonly string _historyFile;
    private readonly int _maxHistory = 500;
    
    // All known completions (namespaces and their functions)
    private readonly Dictionary<string, List<string>> _completions = new()
    {
        ["file"] = new() { "read", "write", "append", "exists", "delete", "copy", "move", "cwd", "home", "temp" },
        ["dir"] = new() { "exists", "list", "files", "dirs", "create", "delete", "copy", "move", "tree", "walk" },
        ["path"] = new() { "join", "dirname", "basename", "ext", "stem", "absolute", "normalize", "exists", "isFile", "isDir" },
        ["string"] = new() { "length", "upper", "lower", "trim", "split", "contains", "startsWith", "endsWith", "replace", "indexOf", "substring", "repeat", "reverse" },
        ["list"] = new() { "sum", "avg", "min", "max", "length", "reverse", "sort", "contains", "join", "range", "append", "prepend", "remove", "slice", "first", "last", "unique", "flatten" },
        ["json"] = new() { "parse", "stringify", "pretty", "valid" },
        ["yaml"] = new() { "parse", "stringify" },
        ["xml"] = new() { "parse", "query" },
        ["regex"] = new() { "match", "matches", "test", "replace", "split", "groups", "escape" },
        ["crypto"] = new() { "hash", "uuid", "random", "base64encode", "base64decode" },
        ["http"] = new() { "get", "post", "put", "delete", "download" },
        ["date"] = new() { "now", "utc", "parse", "format" },
        ["git"] = new() { "status", "branch", "branches", "log", "diff", "show", "root", "remote", "isRepo" },
        ["proc"] = new() { "list", "kill", "exists", "info" },
        ["net"] = new() { "ping", "lookup", "localIp", "isOnline", "ports" },
        ["env"] = new() { "get", "set", "all", "keys", "expand", "path", "home", "temp", "user", "os", "arch", "machine" },
        ["clip"] = new() { "copy", "paste" },
        ["zip"] = new() { "create", "extract", "list", "add" },
        ["diff"] = new() { "lines", "files", "patch" },
        ["template"] = new() { "render" },
        ["math"] = new() { "sin", "cos", "tan", "sqrt", "abs", "pow", "exp", "log", "floor", "ceil", "round", "min", "max", "random" },
        ["sys"] = new() { "exec", "shell", "which", "pid", "kill", "exit", "sleep" }
    };
    
    // Top-level functions and keywords
    private readonly List<string> _topLevel = new()
    {
        "fn", "let", "mut", "const", "if", "else", "for", "while", "return", "break", "continue",
        "true", "false", "null", "and", "or", "not", "in",
        "print", "println", "input", "len", "range", "typeof",
        "file", "dir", "path", "string", "list", "json", "yaml", "xml", "regex", "crypto",
        "http", "date", "git", "proc", "net", "env", "clip", "zip", "diff", "template", "math", "sys"
    };

    public InteractiveConsole()
    {
        _historyFile = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".nsl", "history"
        );
        LoadHistory();
    }

    /// <summary>
    /// Read a line with interactive features
    /// </summary>
    public string? ReadLine(string prompt)
    {
        ColorOutput.Prompt(prompt);
        
        var buffer = new StringBuilder();
        var cursorPos = 0;
        _historyIndex = _history.Count;
        var savedInput = "";

        while (true)
        {
            // Use KeyAvailable check to allow Windows messages to process
            // This fixes taskbar icon click not working
            while (!System.Console.KeyAvailable)
            {
                Thread.Sleep(10); // Small delay to prevent CPU spin
            }
            
            var key = System.Console.ReadKey(intercept: true);

            switch (key.Key)
            {
                case ConsoleKey.Enter:
                    System.Console.WriteLine();
                    var line = buffer.ToString();
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        AddToHistory(line);
                    }
                    return line;

                case ConsoleKey.Escape:
                    // Clear current input
                    ClearLine(prompt, buffer.Length, cursorPos);
                    buffer.Clear();
                    cursorPos = 0;
                    break;

                case ConsoleKey.Backspace:
                    if (cursorPos > 0)
                    {
                        buffer.Remove(cursorPos - 1, 1);
                        cursorPos--;
                        RedrawLine(prompt, buffer.ToString(), cursorPos);
                    }
                    break;

                case ConsoleKey.Delete:
                    if (cursorPos < buffer.Length)
                    {
                        buffer.Remove(cursorPos, 1);
                        RedrawLine(prompt, buffer.ToString(), cursorPos);
                    }
                    break;

                case ConsoleKey.LeftArrow:
                    if (cursorPos > 0)
                    {
                        cursorPos--;
                        System.Console.CursorLeft--;
                    }
                    break;

                case ConsoleKey.RightArrow:
                    if (cursorPos < buffer.Length)
                    {
                        cursorPos++;
                        System.Console.CursorLeft++;
                    }
                    break;

                case ConsoleKey.Home:
                    System.Console.CursorLeft -= cursorPos;
                    cursorPos = 0;
                    break;

                case ConsoleKey.End:
                    System.Console.CursorLeft += buffer.Length - cursorPos;
                    cursorPos = buffer.Length;
                    break;

                case ConsoleKey.UpArrow:
                    if (_history.Count > 0 && _historyIndex > 0)
                    {
                        if (_historyIndex == _history.Count)
                            savedInput = buffer.ToString();
                        _historyIndex--;
                        buffer.Clear();
                        buffer.Append(_history[_historyIndex]);
                        cursorPos = buffer.Length;
                        RedrawLine(prompt, buffer.ToString(), cursorPos);
                    }
                    break;

                case ConsoleKey.DownArrow:
                    if (_historyIndex < _history.Count)
                    {
                        _historyIndex++;
                        buffer.Clear();
                        if (_historyIndex == _history.Count)
                            buffer.Append(savedInput);
                        else
                            buffer.Append(_history[_historyIndex]);
                        cursorPos = buffer.Length;
                        RedrawLine(prompt, buffer.ToString(), cursorPos);
                    }
                    break;

                case ConsoleKey.Tab:
                    var completion = GetCompletion(buffer.ToString(), cursorPos);
                    if (completion != null)
                    {
                        // Find where the word starts
                        var wordStart = cursorPos;
                        while (wordStart > 0 && (char.IsLetterOrDigit(buffer[wordStart - 1]) || buffer[wordStart - 1] == '.'))
                            wordStart--;
                        
                        // Replace the word with completion
                        var wordLen = cursorPos - wordStart;
                        buffer.Remove(wordStart, wordLen);
                        buffer.Insert(wordStart, completion);
                        cursorPos = wordStart + completion.Length;
                        RedrawLine(prompt, buffer.ToString(), cursorPos);
                    }
                    break;

                default:
                    if (key.KeyChar >= 32) // Printable character
                    {
                        buffer.Insert(cursorPos, key.KeyChar);
                        cursorPos++;
                        if (cursorPos == buffer.Length)
                        {
                            System.Console.Write(key.KeyChar);
                        }
                        else
                        {
                            RedrawLine(prompt, buffer.ToString(), cursorPos);
                        }
                    }
                    break;
            }
        }
    }

    private void ClearLine(string prompt, int bufferLen, int cursorPos)
    {
        // Move to start of input
        System.Console.CursorLeft = prompt.Length;
        // Clear to end
        System.Console.Write(new string(' ', bufferLen));
        // Move back to start
        System.Console.CursorLeft = prompt.Length;
    }

    private void RedrawLine(string prompt, string text, int cursorPos)
    {
        var left = System.Console.CursorLeft;
        var promptLen = prompt.Length;
        
        // Move to start of line (after prompt)
        System.Console.CursorLeft = promptLen;
        
        // Write text with padding to clear old content
        var padding = Math.Max(0, left - promptLen - text.Length + 10);
        System.Console.Write(text + new string(' ', padding));
        
        // Position cursor
        System.Console.CursorLeft = promptLen + cursorPos;
    }

    private string? GetCompletion(string input, int cursorPos)
    {
        // Get the word being typed
        var wordStart = cursorPos;
        while (wordStart > 0 && (char.IsLetterOrDigit(input[wordStart - 1]) || input[wordStart - 1] == '.'))
            wordStart--;
        
        var word = input.Substring(wordStart, cursorPos - wordStart);
        if (string.IsNullOrEmpty(word)) return null;

        // Check if it's a namespace.function pattern
        if (word.Contains('.'))
        {
            var parts = word.Split('.');
            var ns = parts[0];
            var prefix = parts.Length > 1 ? parts[1] : "";
            
            if (_completions.TryGetValue(ns, out var funcs))
            {
                var match = funcs.FirstOrDefault(f => f.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
                if (match != null)
                    return ns + "." + match;
            }
        }
        else
        {
            // Top-level completion
            var match = _topLevel.FirstOrDefault(t => t.StartsWith(word, StringComparison.OrdinalIgnoreCase) && t != word);
            if (match != null)
                return match;
            
            // Namespace completion
            var nsMatch = _completions.Keys.FirstOrDefault(k => k.StartsWith(word, StringComparison.OrdinalIgnoreCase) && k != word);
            if (nsMatch != null)
                return nsMatch + ".";
        }

        return null;
    }

    private void AddToHistory(string line)
    {
        // Don't add duplicates of last entry
        if (_history.Count > 0 && _history[^1] == line)
            return;
        
        _history.Add(line);
        
        // Trim history if too long
        while (_history.Count > _maxHistory)
            _history.RemoveAt(0);
        
        SaveHistory();
    }

    private void LoadHistory()
    {
        try
        {
            if (File.Exists(_historyFile))
            {
                var lines = File.ReadAllLines(_historyFile);
                _history.AddRange(lines.TakeLast(_maxHistory));
            }
        }
        catch { }
    }

    private void SaveHistory()
    {
        try
        {
            var dir = Path.GetDirectoryName(_historyFile);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);
            
            File.WriteAllLines(_historyFile, _history.TakeLast(_maxHistory));
        }
        catch { }
    }
}
