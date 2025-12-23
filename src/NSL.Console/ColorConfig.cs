using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace NSL.Console;

/// <summary>
/// Terminal color configuration for NSL output
/// Colors based on human psychology research:
/// - Red: Errors (danger, immediate attention)
/// - Green: Success (positive, completion)
/// - Yellow: Warnings (caution, attention needed)
/// - Cyan: Info (neutral information)
/// - Magenta: Keywords/special
/// - Blue: Strings/values
/// </summary>
public class ColorConfig
{
    // ANSI Color Codes
    public const string Reset = "\u001b[0m";
    public const string Bold = "\u001b[1m";
    public const string Dim = "\u001b[2m";
    public const string Underline = "\u001b[4m";
    
    // Foreground colors
    public const string Black = "\u001b[30m";
    public const string Red = "\u001b[31m";
    public const string Green = "\u001b[32m";
    public const string Yellow = "\u001b[33m";
    public const string Blue = "\u001b[34m";
    public const string Magenta = "\u001b[35m";
    public const string Cyan = "\u001b[36m";
    public const string White = "\u001b[37m";
    
    // Bright foreground colors
    public const string BrightBlack = "\u001b[90m";
    public const string BrightRed = "\u001b[91m";
    public const string BrightGreen = "\u001b[92m";
    public const string BrightYellow = "\u001b[93m";
    public const string BrightBlue = "\u001b[94m";
    public const string BrightMagenta = "\u001b[95m";
    public const string BrightCyan = "\u001b[96m";
    public const string BrightWhite = "\u001b[97m";
    
    // Background colors
    public const string BgRed = "\u001b[41m";
    public const string BgGreen = "\u001b[42m";
    public const string BgYellow = "\u001b[43m";
    public const string BgBlue = "\u001b[44m";
    
    // Configurable semantic colors (can be changed by user)
    public string Error { get; set; } = BrightRed;
    public string ErrorBg { get; set; } = "";
    public string Warning { get; set; } = BrightYellow;
    public string Success { get; set; } = BrightGreen;
    public string Info { get; set; } = BrightCyan;
    public string Hint { get; set; } = Dim + Cyan;
    
    // Syntax highlighting colors
    public string Keyword { get; set; } = BrightMagenta;      // fn, let, if, else, for, while, return
    public string String { get; set; } = BrightGreen;         // "strings"
    public string Number { get; set; } = BrightCyan;          // 123, 3.14
    public string Comment { get; set; } = BrightBlack;        // # comments
    public string Operator { get; set; } = BrightYellow;      // +, -, *, /, =, |>
    public string Function { get; set; } = BrightBlue;        // function names
    public string Variable { get; set; } = White;             // variables
    public string Type { get; set; } = Cyan;                  // type annotations
    public string Namespace { get; set; } = BrightCyan;       // file., sys., etc.
    public string Punctuation { get; set; } = White;          // (), {}, []
    public string Boolean { get; set; } = BrightMagenta;      // true, false
    public string Null { get; set; } = BrightBlack;           // null
    
    // UI colors
    public string Prompt { get; set; } = BrightCyan;
    public string LineNumber { get; set; } = BrightBlack;
    public string Result { get; set; } = BrightWhite;
    public string Banner { get; set; } = BrightMagenta;
    
    // Is color enabled?
    public bool Enabled { get; set; } = true;
    
    // Singleton instance
    private static ColorConfig? _instance;
    private static readonly object _lock = new();
    
    public static ColorConfig Instance
    {
        get
        {
            if (_instance == null)
            {
                lock (_lock)
                {
                    _instance ??= Load();
                }
            }
            return _instance;
        }
    }
    
    /// <summary>
    /// Get the config file path
    /// </summary>
    public static string ConfigPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".nsl",
        "colors.json"
    );
    
    /// <summary>
    /// Load color config from file or create default
    /// </summary>
    public static ColorConfig Load()
    {
        try
        {
            if (File.Exists(ConfigPath))
            {
                var json = File.ReadAllText(ConfigPath);
                var config = JsonSerializer.Deserialize<ColorConfigJson>(json);
                if (config != null)
                {
                    return FromJson(config);
                }
            }
        }
        catch
        {
            // Fall back to defaults
        }
        
        return new ColorConfig();
    }
    
    /// <summary>
    /// Save current config to file
    /// </summary>
    public void Save()
    {
        try
        {
            var dir = Path.GetDirectoryName(ConfigPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
            
            var json = ToJson();
            var options = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(ConfigPath, JsonSerializer.Serialize(json, options));
        }
        catch
        {
            // Ignore save errors
        }
    }
    
    /// <summary>
    /// Create default config file with comments
    /// </summary>
    public static void CreateDefaultConfig()
    {
        var config = new ColorConfig();
        config.Save();
        
        // Also create a README for the colors
        var readmePath = Path.Combine(Path.GetDirectoryName(ConfigPath)!, "colors-readme.txt");
        var readme = @"NSL Color Configuration
=======================

Edit colors.json to customize NSL terminal colors.

AVAILABLE COLORS:
-----------------
Use these values for any color setting:

  ""black""        - Dark black
  ""red""          - Dark red
  ""green""        - Dark green  
  ""yellow""       - Dark yellow
  ""blue""         - Dark blue
  ""magenta""      - Dark magenta
  ""cyan""         - Dark cyan
  ""white""        - Light gray
  
  ""bright_black""   - Gray (dim)
  ""bright_red""     - Bright red (errors)
  ""bright_green""   - Bright green (success)
  ""bright_yellow""  - Bright yellow (warnings)
  ""bright_blue""    - Bright blue
  ""bright_magenta"" - Bright magenta (keywords)
  ""bright_cyan""    - Bright cyan (info)
  ""bright_white""   - Bright white

MODIFIERS (prefix with +):
--------------------------
  ""+bold""      - Bold text
  ""+dim""       - Dimmed text
  ""+underline"" - Underlined text

Example: ""bright_red+bold"" = bold bright red

COLOR CATEGORIES:
-----------------
- error     : Error messages (default: bright_red)
- warning   : Warning messages (default: bright_yellow)
- success   : Success messages (default: bright_green)
- info      : Info messages (default: bright_cyan)
- hint      : Hints and suggestions (default: dim cyan)

SYNTAX HIGHLIGHTING:
--------------------
- keyword   : fn, let, if, else, for, while, return, etc.
- string    : ""string literals""
- number    : 123, 3.14
- comment   : # comments
- operator  : +, -, *, /, =, |>
- function  : function names when called
- variable  : variable names
- namespace : file., sys., json., etc.
- boolean   : true, false
- null      : null

To disable colors entirely, set:
  ""enabled"": false
";
        File.WriteAllText(readmePath, readme);
    }
    
    /// <summary>
    /// Parse color string to ANSI code
    /// </summary>
    public static string ParseColor(string colorName)
    {
        if (string.IsNullOrEmpty(colorName)) return "";
        
        var result = "";
        var parts = colorName.ToLower().Split('+');
        
        foreach (var part in parts)
        {
            result += part.Trim() switch
            {
                "black" => Black,
                "red" => Red,
                "green" => Green,
                "yellow" => Yellow,
                "blue" => Blue,
                "magenta" => Magenta,
                "cyan" => Cyan,
                "white" => White,
                "bright_black" or "gray" or "grey" => BrightBlack,
                "bright_red" => BrightRed,
                "bright_green" => BrightGreen,
                "bright_yellow" => BrightYellow,
                "bright_blue" => BrightBlue,
                "bright_magenta" => BrightMagenta,
                "bright_cyan" => BrightCyan,
                "bright_white" => BrightWhite,
                "bold" => Bold,
                "dim" => Dim,
                "underline" => Underline,
                _ => ""
            };
        }
        
        return result;
    }
    
    private ColorConfigJson ToJson()
    {
        return new ColorConfigJson
        {
            Enabled = Enabled,
            Error = ColorToName(Error),
            Warning = ColorToName(Warning),
            Success = ColorToName(Success),
            Info = ColorToName(Info),
            Hint = ColorToName(Hint),
            Keyword = ColorToName(Keyword),
            String = ColorToName(String),
            Number = ColorToName(Number),
            Comment = ColorToName(Comment),
            Operator = ColorToName(Operator),
            Function = ColorToName(Function),
            Variable = ColorToName(Variable),
            Namespace = ColorToName(Namespace),
            Boolean = ColorToName(Boolean),
            Null = ColorToName(Null),
            Prompt = ColorToName(Prompt),
            LineNumber = ColorToName(LineNumber)
        };
    }
    
    private static ColorConfig FromJson(ColorConfigJson json)
    {
        var config = new ColorConfig
        {
            Enabled = json.Enabled
        };
        
        if (!string.IsNullOrEmpty(json.Error)) config.Error = ParseColor(json.Error);
        if (!string.IsNullOrEmpty(json.Warning)) config.Warning = ParseColor(json.Warning);
        if (!string.IsNullOrEmpty(json.Success)) config.Success = ParseColor(json.Success);
        if (!string.IsNullOrEmpty(json.Info)) config.Info = ParseColor(json.Info);
        if (!string.IsNullOrEmpty(json.Hint)) config.Hint = ParseColor(json.Hint);
        if (!string.IsNullOrEmpty(json.Keyword)) config.Keyword = ParseColor(json.Keyword);
        if (!string.IsNullOrEmpty(json.String)) config.String = ParseColor(json.String);
        if (!string.IsNullOrEmpty(json.Number)) config.Number = ParseColor(json.Number);
        if (!string.IsNullOrEmpty(json.Comment)) config.Comment = ParseColor(json.Comment);
        if (!string.IsNullOrEmpty(json.Operator)) config.Operator = ParseColor(json.Operator);
        if (!string.IsNullOrEmpty(json.Function)) config.Function = ParseColor(json.Function);
        if (!string.IsNullOrEmpty(json.Variable)) config.Variable = ParseColor(json.Variable);
        if (!string.IsNullOrEmpty(json.Namespace)) config.Namespace = ParseColor(json.Namespace);
        if (!string.IsNullOrEmpty(json.Boolean)) config.Boolean = ParseColor(json.Boolean);
        if (!string.IsNullOrEmpty(json.Null)) config.Null = ParseColor(json.Null);
        if (!string.IsNullOrEmpty(json.Prompt)) config.Prompt = ParseColor(json.Prompt);
        if (!string.IsNullOrEmpty(json.LineNumber)) config.LineNumber = ParseColor(json.LineNumber);
        
        return config;
    }
    
    private static string ColorToName(string ansi)
    {
        // Map ANSI codes back to names
        if (ansi.Contains(BrightRed)) return "bright_red";
        if (ansi.Contains(BrightGreen)) return "bright_green";
        if (ansi.Contains(BrightYellow)) return "bright_yellow";
        if (ansi.Contains(BrightBlue)) return "bright_blue";
        if (ansi.Contains(BrightMagenta)) return "bright_magenta";
        if (ansi.Contains(BrightCyan)) return "bright_cyan";
        if (ansi.Contains(BrightWhite)) return "bright_white";
        if (ansi.Contains(BrightBlack)) return "bright_black";
        if (ansi.Contains(Red)) return "red";
        if (ansi.Contains(Green)) return "green";
        if (ansi.Contains(Yellow)) return "yellow";
        if (ansi.Contains(Blue)) return "blue";
        if (ansi.Contains(Magenta)) return "magenta";
        if (ansi.Contains(Cyan)) return "cyan";
        if (ansi.Contains(White)) return "white";
        if (ansi.Contains(Black)) return "black";
        return "white";
    }
    
    // JSON serialization class
    private class ColorConfigJson
    {
        public bool Enabled { get; set; } = true;
        public string Error { get; set; } = "bright_red";
        public string Warning { get; set; } = "bright_yellow";
        public string Success { get; set; } = "bright_green";
        public string Info { get; set; } = "bright_cyan";
        public string Hint { get; set; } = "dim+cyan";
        public string Keyword { get; set; } = "bright_magenta";
        public string String { get; set; } = "bright_green";
        public string Number { get; set; } = "bright_cyan";
        public string Comment { get; set; } = "bright_black";
        public string Operator { get; set; } = "bright_yellow";
        public string Function { get; set; } = "bright_blue";
        public string Variable { get; set; } = "white";
        public string Namespace { get; set; } = "bright_cyan";
        public string Boolean { get; set; } = "bright_magenta";
        public string Null { get; set; } = "bright_black";
        public string Prompt { get; set; } = "bright_cyan";
        public string LineNumber { get; set; } = "bright_black";
    }
}

/// <summary>
/// Helper class for colored console output
/// </summary>
public static class ColorOutput
{
    private static ColorConfig Config => ColorConfig.Instance;
    
    /// <summary>
    /// Check if colors should be used
    /// </summary>
    public static bool UseColors => Config.Enabled && !System.Console.IsOutputRedirected;
    
    /// <summary>
    /// Apply color to text
    /// </summary>
    public static string Colorize(string text, string color)
    {
        if (!UseColors || string.IsNullOrEmpty(color)) return text;
        return $"{color}{text}{ColorConfig.Reset}";
    }
    
    // Semantic output methods
    public static void Error(string message) => 
        System.Console.WriteLine(Colorize($"✗ {message}", Config.Error));
    
    public static void ErrorLine(string message) =>
        System.Console.WriteLine(Colorize(message, Config.Error));
    
    public static void Warning(string message) => 
        System.Console.WriteLine(Colorize($"⚠ {message}", Config.Warning));
    
    public static void Success(string message) => 
        System.Console.WriteLine(Colorize($"✓ {message}", Config.Success));
    
    public static void Info(string message) => 
        System.Console.WriteLine(Colorize($"ℹ {message}", Config.Info));
    
    public static void Hint(string message) => 
        System.Console.WriteLine(Colorize($"  → {message}", Config.Hint));
    
    public static void Result(object? value) =>
        System.Console.WriteLine(Colorize(value?.ToString() ?? "null", Config.Result));
    
    /// <summary>
    /// Print a line with syntax highlighting
    /// </summary>
    public static void PrintCode(string code, int? lineNumber = null)
    {
        if (lineNumber.HasValue)
        {
            System.Console.Write(Colorize($"{lineNumber,4} │ ", Config.LineNumber));
        }
        System.Console.WriteLine(HighlightSyntax(code));
    }
    
    /// <summary>
    /// Apply syntax highlighting to NSL code
    /// </summary>
    public static string HighlightSyntax(string code)
    {
        if (!UseColors) return code;
        
        var result = new System.Text.StringBuilder();
        var i = 0;
        
        while (i < code.Length)
        {
            // Comments
            if (code[i] == '#')
            {
                var end = code.IndexOf('\n', i);
                if (end == -1) end = code.Length;
                result.Append(Colorize(code[i..end], Config.Comment));
                i = end;
                continue;
            }
            
            // Strings
            if (code[i] == '"' || (code[i] == 'r' && i + 1 < code.Length && code[i + 1] == '"'))
            {
                var start = i;
                if (code[i] == 'r') i++; // Skip 'r' prefix
                i++; // Skip opening quote
                
                // Check for heredoc
                if (i + 1 < code.Length && code[i] == '"' && code[i + 1] == '"')
                {
                    i += 2;
                    var endHeredoc = code.IndexOf("\"\"\"", i);
                    if (endHeredoc == -1) endHeredoc = code.Length;
                    else endHeredoc += 3;
                    result.Append(Colorize(code[start..endHeredoc], Config.String));
                    i = endHeredoc;
                }
                else
                {
                    while (i < code.Length && code[i] != '"')
                    {
                        if (code[i] == '\\' && i + 1 < code.Length) i++;
                        i++;
                    }
                    if (i < code.Length) i++; // Skip closing quote
                    result.Append(Colorize(code[start..i], Config.String));
                }
                continue;
            }
            
            // Numbers
            if (char.IsDigit(code[i]) || (code[i] == '.' && i + 1 < code.Length && char.IsDigit(code[i + 1])))
            {
                var start = i;
                while (i < code.Length && (char.IsDigit(code[i]) || code[i] == '.' || code[i] == 'e' || code[i] == 'E'))
                    i++;
                result.Append(Colorize(code[start..i], Config.Number));
                continue;
            }
            
            // Identifiers and keywords
            if (char.IsLetter(code[i]) || code[i] == '_')
            {
                var start = i;
                while (i < code.Length && (char.IsLetterOrDigit(code[i]) || code[i] == '_'))
                    i++;
                var word = code[start..i];
                
                // Check for namespace access (word followed by .)
                var isNamespace = i < code.Length && code[i] == '.';
                
                var color = word switch
                {
                    "fn" or "let" or "mut" or "const" or "if" or "else" or "for" or "while" or 
                    "return" or "break" or "continue" or "match" or "case" or "import" or 
                    "export" or "pub" or "async" or "await" or "struct" or "enum" or "trait" or
                    "impl" or "in" or "and" or "or" or "not" => Config.Keyword,
                    "true" or "false" => Config.Boolean,
                    "null" or "none" => Config.Null,
                    _ when isNamespace => Config.Namespace,
                    _ => Config.Variable
                };
                
                result.Append(Colorize(word, color));
                continue;
            }
            
            // Operators
            if ("|>=<+-*/%!&^~@:".Contains(code[i]))
            {
                var start = i;
                // Handle multi-char operators
                if (i + 1 < code.Length)
                {
                    var twoChar = code.Substring(i, 2);
                    if (twoChar is "|>" or "=>" or ">=" or "<=" or "==" or "!=" or "&&" or "||" or "::" or ".." or "**" or "//")
                    {
                        result.Append(Colorize(twoChar, Config.Operator));
                        i += 2;
                        continue;
                    }
                }
                result.Append(Colorize(code[i].ToString(), Config.Operator));
                i++;
                continue;
            }
            
            // Punctuation
            if ("(){}[].,;".Contains(code[i]))
            {
                result.Append(Colorize(code[i].ToString(), Config.Punctuation));
                i++;
                continue;
            }
            
            // Default: just append
            result.Append(code[i]);
            i++;
        }
        
        return result.ToString();
    }
    
    /// <summary>
    /// Print error with code context
    /// </summary>
    public static void PrintErrorWithContext(string error, string? code, int? line = null)
    {
        Error(error);
        
        if (!string.IsNullOrEmpty(code) && line.HasValue)
        {
            var lines = code.Split('\n');
            var startLine = Math.Max(0, line.Value - 3);
            var endLine = Math.Min(lines.Length, line.Value + 2);
            
            System.Console.WriteLine();
            for (int i = startLine; i < endLine; i++)
            {
                var isErrorLine = i == line.Value - 1;
                var prefix = isErrorLine ? Colorize("→ ", Config.Error) : "  ";
                var lineNum = Colorize($"{i + 1,4} │ ", Config.LineNumber);
                var lineText = isErrorLine 
                    ? Colorize(lines[i], Config.Error) 
                    : HighlightSyntax(lines[i]);
                System.Console.WriteLine($"{prefix}{lineNum}{lineText}");
            }
            System.Console.WriteLine();
        }
    }
    
    /// <summary>
    /// Print a banner/header
    /// </summary>
    public static void Banner(string text)
    {
        System.Console.WriteLine(Colorize(text, Config.Banner));
    }
    
    /// <summary>
    /// Print the REPL prompt
    /// </summary>
    public static void Prompt(string text = "nsl> ")
    {
        System.Console.Write(Colorize(text, Config.Prompt));
    }
}
