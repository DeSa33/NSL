using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace NSL.Core.AutoFix;

/// <summary>
/// Represents a single fix to apply to source code
/// </summary>
public class SourceFix
{
    /// <summary>Public API</summary>
    public int Line { get; }
    /// <summary>Public API</summary>
    public int Column { get; }
    /// <summary>Public API</summary>
    public int Length { get; }
    /// <summary>Public API</summary>
    public string OriginalText { get; }
    /// <summary>Public API</summary>
    public string ReplacementText { get; }
    /// <summary>Public API</summary>
    public string Description { get; }
    /// <summary>Public API</summary>
    public FixCategory Category { get; }

    /// <summary>Public API</summary>
    public SourceFix(int line, int column, int length, string original, string replacement, string description, FixCategory category = FixCategory.Error)
    {
        Line = line;
        Column = column;
        Length = length;
        OriginalText = original;
        ReplacementText = replacement;
        Description = description;
        Category = category;
    }

    /// <summary>Public API</summary>
    public override string ToString() =>
        $"[{Category}] Line {Line}:{Column} - {Description}: '{OriginalText}' -> '{ReplacementText}'";
}

/// <summary>Public API</summary>
public enum FixCategory
{
    Error,      // Must fix to compile
    Warning,    // Should fix for correctness
    Style,      // Optional style improvement
    Suggestion  // Potential improvement
}

/// <summary>
/// Auto-fix system for NSL - detects and fixes common errors automatically
/// </summary>
public class NSLAutoFix
{
    private readonly string _source;
    private readonly string[] _lines;
    private readonly List<SourceFix> _fixes = new();
    private readonly HashSet<string> _knownVariables = new();
    private readonly HashSet<string> _knownFunctions = new();
    private readonly HashSet<string> _knownTypes = new();

    // Built-in functions and types
    private static readonly HashSet<string> BuiltinFunctions = new()
    {
        // Core functions
        "print", "println", "input", "len", "type", "range",

        // Math functions
        "sqrt", "abs", "sin", "cos", "tan", "exp", "log", "pow", "floor", "ceil", "round",
        "min", "max", "sum", "avg", "random",

        // Neural/ML functions
        "relu", "sigmoid", "tanh", "leaky_relu", "softplus", "gelu", "softmax", "normalize",
        "mean", "dot", "zeros", "ones",

        // Shell/System functions
        "shell", "exec", "run", "powershell", "set_env", "env_all",
        "sleep", "exit", "timestamp", "now",

        // File system functions
        "read_file", "write_file", "file_exists", "dir_exists", "file_info",
        "list_dir", "mkdir", "delete_file", "delete_dir", "copy_file", "move_file",
        "cwd", "cd", "read_binary", "write_binary",

        // Encoding functions
        "base64_encode", "base64_decode", "hex_encode", "hex_decode",
        "url_encode", "url_decode",

        // Hashing functions
        "md5", "sha256", "sha512", "hash_file",

        // Compression functions
        "gzip", "gunzip", "zip_create", "zip_extract", "zip_list",

        // JSON functions
        "json_parse", "json_stringify",

        // String functions
        "split", "join", "replace", "trim", "upper", "lower",
        "contains", "starts_with", "ends_with", "substring",
        "regex_match", "regex_replace",

        // Array functions
        "map", "filter", "reduce", "find", "sort", "reverse", "unique", "flatten", "slice",

        // HTTP functions
        "http_get", "http_post", "download",

        // Utility functions
        "uuid",

        // Consciousness functions
        "measure", "entangle",

        // Semantic file access (AI-optimized)
        "attention_read", "extract", "stream_file",

        // GPU namespace functions (accessed via gpu.xxx)
        "gpu"
    };

    // NSL Namespaces - these are valid identifiers that should NOT be "fixed"
    private static readonly HashSet<string> Namespaces = new()
    {
        // All NSL namespaces
        "file", "dir", "path", "sys", "env", "json", "yaml", "xml",
        "git", "proc", "net", "http", "crypto", "clip", "zip", "diff",
        "template", "date", "math", "regex", "list", "string", "text",
        
        // Common namespace function names (so they aren't "fixed" to typos)
        "parse", "stringify", "pretty", "valid", "read", "write", "append",
        "exists", "delete", "copy", "move", "create", "list", "files", "dirs",
        "tree", "walk", "join", "dirname", "basename", "ext", "absolute",
        "temp", "home", "user", "os", "arch", "get", "set", "all", "keys",
        "expand", "exec", "shell", "pipe", "sleep", "status", "branch",
        "branches", "log", "show", "remote", "isRepo", "kill", "info",
        "ping", "lookup", "localIp", "isOnline", "ports", "post", "download",
        "hash", "uuid", "random", "base64encode", "base64decode",
        "test", "match", "matches", "split", "render", "lines", "files",
        "patch", "extract", "now", "utc", "format", "sin", "cos", "sqrt",
        "abs", "pow", "floor", "ceil", "round", "upper", "lower", "trim",
        "contains", "repeat", "sum", "avg", "min", "max", "sort", "reverse",
        "unique", "flatten", "matmul", "tensor", "transpose", "benchmark",
        
        // Common variable names that shouldn't be flagged
        "msg", "err", "result", "data", "value", "item", "items", "count",
        "index", "key", "val", "tmp", "temp", "src", "dst", "dest",
        "old", "new", "before", "after", "left", "right", "top", "bottom",
        "start", "end", "first", "last", "next", "prev", "current",
        "name", "text", "content", "body", "title", "desc", "description",
        "config", "options", "settings", "params", "args", "argv",
        "response", "request", "req", "res", "ctx", "context",
        "callback", "handler", "listener", "event", "events",
        "width", "height", "size", "length", "offset", "limit",
        "weights", "bias", "loss", "epoch", "epochs", "batch", "lr",
        "gradient", "grads", "output", "outputs", "input", "inputs",
        "layer", "layers", "model", "network", "tensor", "tensors",
        "sample", "samples", "label", "labels", "target", "targets",
        "prediction", "predictions", "accuracy", "precision", "recall",
        "score", "scores", "feature", "features", "embedding", "embeddings"
    };

    // GPU namespace functions for typo detection
    private static readonly HashSet<string> GpuFunctions = new()
    {
        "init", "devices", "tensor", "zeros", "ones", "random",
        "matmul", "add", "sub", "mul", "div",
        "relu", "sigmoid", "tanh", "softmax",
        "exp", "log", "sqrt", "pow", "transpose",
        "to_cpu", "shape", "size", "dispose", "shutdown"
    };

    private static readonly HashSet<string> BuiltinTypes = new()
    {
        "number", "int", "bool", "string", "void", "any", "vec", "mat", "tensor", "prob"
    };

    private static readonly HashSet<string> Keywords = new()
    {
        "fn", "let", "mut", "if", "else", "while", "for", "in", "return", "break", "continue",
        "true", "false", "null", "import", "export", "from", "pub", "module", "as", "match", "type"
    };

    /// <summary>Public API</summary>
    public IReadOnlyList<SourceFix> Fixes => _fixes;

    /// <summary>Public API</summary>
    public NSLAutoFix(string source)
    {
        _source = source;
        _lines = source.Split('\n');
        _knownFunctions.UnionWith(BuiltinFunctions);
        _knownTypes.UnionWith(BuiltinTypes);
    }

    /// <summary>
    /// Analyze source and generate fixes
    /// </summary>
    public void Analyze()
    {
        // First pass: collect declared variables and functions
        CollectDeclarations();

        // Second pass: find and fix errors (ordered by priority)
        // Keyword conversions first (before structural fixes to avoid position conflicts)
        FixCommonKeywordTypos();
        FixWrongKeywords();
        FixOperatorTypos();
        
        // Track which lines have keyword fixes to avoid conflicting comma fixes
        var linesWithKeywordFixes = _fixes.Select(f => f.Line).ToHashSet();
        
        // Structural fixes (skip lines with keyword fixes to avoid corruption)
        FixMissingCommas(linesWithKeywordFixes);
        FixMissingColonsInParameters();
        FixWrongCommentSyntax();
        FixForLoopSyntax();
        FixControlFlowSyntax();
        FixUnclosedBrackets();
        FixUnclosedStrings();

        // Semantic errors
        FixCommonSyntaxErrors();
        FixTypoIdentifiers();
        FixReservedWordVariables();

        // Suggestions
        FixNumberLiterals();
        FixMissingTypeAnnotations();
        FixStyleIssues();
        
        // Remove duplicate fixes (same line, column, and replacement)
        DeduplicateFixes();
    }
    
    /// <summary>
    /// Remove duplicate fixes that target the same location with the same replacement
    /// </summary>
    private void DeduplicateFixes()
    {
        var seen = new HashSet<string>();
        var uniqueFixes = new List<SourceFix>();
        
        foreach (var fix in _fixes.OrderBy(f => f.Category)) // Keep higher priority (lower enum value) first
        {
            // Create a unique key for this fix
            var key = $"{fix.Line}:{fix.Column}:{fix.Length}:{fix.OriginalText}:{fix.ReplacementText}";
            
            if (!seen.Contains(key))
            {
                seen.Add(key);
                uniqueFixes.Add(fix);
            }
        }
        
        _fixes.Clear();
        _fixes.AddRange(uniqueFixes);
    }

    /// <summary>
    /// Analyze with error messages from compiler
    /// </summary>
    public void AnalyzeWithErrors(IEnumerable<string> errorMessages)
    {
        foreach (var error in errorMessages)
        {
            TryFixFromErrorMessage(error);
        }
    }

    private void CollectDeclarations()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];

            // Collect function declarations: fn name(
            var fnMatch = Regex.Match(line, @"\bfn\s+(\w+)\s*\(");
            if (fnMatch.Success)
            {
                _knownFunctions.Add(fnMatch.Groups[1].Value);
            }

            // Collect variable declarations: let name or let name:
            var letMatch = Regex.Match(line, @"\blet\s+(mut\s+)?(\w+)");
            if (letMatch.Success)
            {
                _knownVariables.Add(letMatch.Groups[2].Value);
            }

            // Collect function parameters: (name: type, name2: type)
            var paramMatches = Regex.Matches(line, @"\(([^)]+)\)");
            foreach (Match pm in paramMatches)
            {
                var paramStr = pm.Groups[1].Value;
                var params_ = Regex.Matches(paramStr, @"(\w+)\s*:");
                foreach (Match p in params_)
                {
                    _knownVariables.Add(p.Groups[1].Value);
                }
            }
        }
    }

    /// <summary>
    /// Fix integer literals that should be floats (e.g., 5 -> 5.0)
    /// NSL uses floating point by default
    /// </summary>
    private void FixNumberLiterals()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];

            // Skip lines that are entirely comments
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("//"))
                continue;

            // Remove comment portion from line for analysis
            var commentIdx = line.IndexOf("//");
            var codePart = commentIdx >= 0 ? line.Substring(0, commentIdx) : line;

            // Find standalone integers that aren't part of a float or array index
            // Match integers not followed by . or preceded by [
            var matches = Regex.Matches(codePart, @"(?<![.\w\[])\b(\d+)\b(?!\.\d)");

            foreach (Match match in matches)
            {
                var intValue = match.Groups[1].Value;
                var col = match.Groups[1].Index;

                // Skip if it's already part of a context where int is appropriate
                // (like array indices, but NSL doesn't have those yet)
                // Also skip if followed by closing paren with .0 style numbers around
                var before = col > 0 ? codePart.Substring(0, col) : "";

                // Check if this is in a type annotation context (e.g., vec[3])
                if (before.EndsWith("[") || before.EndsWith("vec[") || before.EndsWith("mat["))
                    continue;

                // Suggest fix
                _fixes.Add(new SourceFix(
                    i + 1,
                    col + 1,
                    intValue.Length,
                    intValue,
                    intValue + ".0",
                    $"Convert integer literal to float",
                    FixCategory.Suggestion
                ));
            }
        }
    }

    /// <summary>
    /// Fix typos in identifiers by finding closest match
    /// </summary>
    private void FixTypoIdentifiers()
    {
        var allKnown = _knownVariables.Union(_knownFunctions).Union(Keywords).Union(Namespaces).ToHashSet();

        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];

            // Skip lines that are entirely comments (# or //)
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("//") || trimmed.StartsWith("#"))
                continue;

            // Remove comment portion from line for analysis (both # and //)
            var codePart = line;
            var hashIdx = FindCommentStart(codePart, '#');
            if (hashIdx >= 0)
                codePart = codePart.Substring(0, hashIdx);
            var slashIdx = FindCommentStart(codePart, '/');
            if (slashIdx >= 0 && slashIdx + 1 < codePart.Length && codePart[slashIdx + 1] == '/')
                codePart = codePart.Substring(0, slashIdx);

            // Remove string literals to avoid false positives
            codePart = RemoveStringLiterals(codePart);

            // Find all identifiers in the code part only
            var matches = Regex.Matches(codePart, @"\b([a-zA-Z_]\w*)\b");

            foreach (Match match in matches)
            {
                var ident = match.Groups[1].Value;
                var col = match.Groups[1].Index;

                // Skip if it's a known identifier, keyword, or in a declaration
                if (allKnown.Contains(ident))
                    continue;
                
                // Skip if it's a namespace access (e.g., json.parse - don't fix "json")
                var afterIdent = col + ident.Length < codePart.Length 
                    ? codePart.Substring(col + ident.Length).TrimStart() 
                    : "";
                if (afterIdent.StartsWith("."))
                {
                    // This is a namespace - check if it's a known namespace with typo
                    var closestNs = FindClosestMatch(ident, Namespaces);
                    if (closestNs != null && closestNs != ident && LevenshteinDistance(ident, closestNs) == 1)
                    {
                        _fixes.Add(new SourceFix(
                            i + 1,
                            col + 1,
                            ident.Length,
                            ident,
                            closestNs,
                            $"Fix namespace typo (did you mean '{closestNs}'?)",
                            FixCategory.Error
                        ));
                    }
                    continue;
                }
                
                // Get context before identifier
                var before = col > 0 ? codePart.Substring(0, col).TrimEnd() : "";
                
                // Skip if it's a member access (e.g., obj.member - don't fix "member")
                if (before.EndsWith("."))
                    continue;

                // Skip if it's a type annotation (after :)
                if (before.EndsWith(":"))
                {
                    // Only fix type typos for identifiers that are at least 3 chars
                    // and have distance 1 from a known type (very conservative)
                    if (ident.Length >= 3)
                    {
                        var closestType = FindClosestMatch(ident, BuiltinTypes);
                        if (closestType != null && closestType != ident && 
                            LevenshteinDistance(ident, closestType) == 1 &&
                            Math.Abs(ident.Length - closestType.Length) <= 1)
                        {
                            _fixes.Add(new SourceFix(
                                i + 1,
                                col + 1,
                                ident.Length,
                                ident,
                                closestType,
                                $"Fix type name typo",
                                FixCategory.Warning
                            ));
                        }
                    }
                    continue;
                }

                // Skip if this is a function/variable declaration
                if (before.EndsWith("fn ") || before.EndsWith("let ") || before.EndsWith("mut "))
                    continue;

                // Find closest match - but be conservative!
                // Only suggest fixes for very close matches (distance 1) or exact prefix matches
                var closest = FindClosestMatch(ident, allKnown);
                if (closest != null && closest != ident)
                {
                    var distance = LevenshteinDistance(ident, closest);
                    
                    // Be very conservative - only fix if:
                    // 1. Distance is 1 (single typo) AND length is similar
                    // 2. Or it's a known keyword typo from our dictionary
                    bool shouldFix = distance == 1 && 
                                     Math.Abs(ident.Length - closest.Length) <= 1 &&
                                     ident.Length >= 3; // Don't fix very short identifiers
                    
                    // Extra check: don't fix if identifier looks intentional
                    // (e.g., starts with same 2+ chars and is reasonably long)
                    if (shouldFix && ident.Length >= 4 && closest.Length >= 4)
                    {
                        // If they share a common prefix of 3+, more likely intentional
                        int commonPrefix = 0;
                        for (int k = 0; k < Math.Min(ident.Length, closest.Length); k++)
                        {
                            if (char.ToLower(ident[k]) == char.ToLower(closest[k]))
                                commonPrefix++;
                            else break;
                        }
                        // If most of the word matches, it's probably a typo
                        shouldFix = commonPrefix >= ident.Length - 2;
                    }
                    
                    if (shouldFix)
                    {
                        _fixes.Add(new SourceFix(
                            i + 1,
                            col + 1,
                            ident.Length,
                            ident,
                            closest,
                            $"Fix typo (did you mean '{closest}'?)",
                            FixCategory.Warning // Changed from Error to Warning
                        ));
                    }
                }
            }
        }
    }

    /// <summary>
    /// Find the start of a comment, accounting for strings
    /// </summary>
    private int FindCommentStart(string line, char commentChar)
    {
        bool inString = false;
        for (int i = 0; i < line.Length; i++)
        {
            if (line[i] == '"' && (i == 0 || line[i - 1] != '\\'))
                inString = !inString;
            if (!inString && line[i] == commentChar)
                return i;
        }
        return -1;
    }

    /// <summary>
    /// Remove string literals from code to avoid false typo detection
    /// </summary>
    private string RemoveStringLiterals(string code)
    {
        var result = new System.Text.StringBuilder();
        bool inString = false;
        for (int i = 0; i < code.Length; i++)
        {
            if (code[i] == '"' && (i == 0 || code[i - 1] != '\\'))
            {
                inString = !inString;
                result.Append(code[i]);
            }
            else if (inString)
            {
                result.Append(' '); // Replace string content with spaces to preserve positions
            }
            else
            {
                result.Append(code[i]);
            }
        }
        return result.ToString();
    }

    /// <summary>
    /// Add missing type annotations where they can be inferred
    /// </summary>
    private void FixMissingTypeAnnotations()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];

            // Find let declarations without type: let x = value
            var match = Regex.Match(line, @"\blet\s+(mut\s+)?(\w+)\s*=\s*(.+)");
            if (match.Success)
            {
                var varName = match.Groups[2].Value;
                var value = match.Groups[3].Value.Trim();
                var afterLet = match.Groups[2].Index + match.Groups[2].Length;

                // Check if there's already a type annotation
                var afterVar = line.Substring(afterLet);
                if (afterVar.TrimStart().StartsWith(":"))
                    continue;

                // Try to infer type from value
                string? inferredType = InferType(value);
                if (inferredType != null)
                {
                    _fixes.Add(new SourceFix(
                        i + 1,
                        afterLet + 1,
                        0,
                        "",
                        $": {inferredType}",
                        $"Add inferred type annotation",
                        FixCategory.Suggestion
                    ));
                }
            }
        }
    }

    /// <summary>
    /// Fix common syntax errors
    /// </summary>
    private void FixCommonSyntaxErrors()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];

            // Fix == used as = in assignments (let x == 5 -> let x = 5)
            var doubleEqMatch = Regex.Match(line, @"\blet\s+(\w+)\s*==\s*");
            if (doubleEqMatch.Success)
            {
                var eqPos = line.IndexOf("==", doubleEqMatch.Index);
                _fixes.Add(new SourceFix(
                    i + 1,
                    eqPos + 1,
                    2,
                    "==",
                    "=",
                    "Use single = for assignment",
                    FixCategory.Error
                ));
            }

            // Fix = used as == in conditions (if x = 5 -> if x == 5)
            var condMatch = Regex.Match(line, @"\b(if|while)\s+.+[^=!<>]=[^=]");
            if (condMatch.Success)
            {
                // Find the = that's not part of == or !=
                var condPart = line.Substring(condMatch.Index);
                var singleEqMatch = Regex.Match(condPart, @"([^=!<>])=([^=])");
                if (singleEqMatch.Success)
                {
                    var eqPos = condMatch.Index + singleEqMatch.Index + 1;
                    _fixes.Add(new SourceFix(
                        i + 1,
                        eqPos + 1,
                        1,
                        "=",
                        "==",
                        "Use == for comparison",
                        FixCategory.Error
                    ));
                }
            }

            // Fix missing .0 on numbers in arithmetic (5 + 3 -> 5.0 + 3.0)
            // Already handled by FixNumberLiterals

            // Fix print("...") when println intended (common mistake)
            // This is subjective, skip for now
        }
    }

    /// <summary>
    /// Detect unclosed brackets: (, [, {
    /// </summary>
    private void FixUnclosedBrackets()
    {
        var bracketStack = new Stack<(char bracket, int line, int col)>();
        var bracketPairs = new Dictionary<char, char>
        {
            { '(', ')' },
            { '[', ']' },
            { '{', '}' }
        };
        var closingBrackets = new HashSet<char> { ')', ']', '}' };

        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            bool inString = false;
            bool inComment = false;

            for (int j = 0; j < line.Length; j++)
            {
                char c = line[j];

                // Skip comments
                if (!inString && j + 1 < line.Length && line[j] == '/' && line[j + 1] == '/')
                    break;
                if (!inString && c == '#')
                    break;

                // Track string state
                if (c == '"' && (j == 0 || line[j - 1] != '\\'))
                    inString = !inString;

                if (inString || inComment)
                    continue;

                // Track brackets
                if (bracketPairs.ContainsKey(c))
                {
                    bracketStack.Push((c, i + 1, j + 1));
                }
                else if (closingBrackets.Contains(c))
                {
                    if (bracketStack.Count > 0)
                    {
                        var (openBracket, openLine, openCol) = bracketStack.Peek();
                        if (bracketPairs[openBracket] == c)
                        {
                            bracketStack.Pop();
                        }
                        else
                        {
                            // Mismatched bracket
                            _fixes.Add(new SourceFix(
                                i + 1,
                                j + 1,
                                1,
                                c.ToString(),
                                bracketPairs[openBracket].ToString(),
                                $"Mismatched bracket - expected '{bracketPairs[openBracket]}' to close '{openBracket}' from line {openLine}",
                                FixCategory.Error
                            ));
                        }
                    }
                    else
                    {
                        // Extra closing bracket
                        _fixes.Add(new SourceFix(
                            i + 1,
                            j + 1,
                            1,
                            c.ToString(),
                            "",
                            $"Unexpected closing bracket '{c}' - no matching opening bracket",
                            FixCategory.Error
                        ));
                    }
                }
            }
        }

        // Report unclosed brackets
        while (bracketStack.Count > 0)
        {
            var (bracket, line, col) = bracketStack.Pop();
            var expected = bracketPairs[bracket];
            _fixes.Add(new SourceFix(
                line,
                col,
                1,
                bracket.ToString(),
                bracket.ToString(),
                $"Unclosed '{bracket}' - missing closing '{expected}'",
                FixCategory.Error
            ));
        }
    }

    /// <summary>
    /// Detect unclosed string literals
    /// </summary>
    private void FixUnclosedStrings()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            bool inString = false;
            int stringStart = -1;

            // Skip comment lines
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("//") || trimmed.StartsWith("#"))
                continue;

            for (int j = 0; j < line.Length; j++)
            {
                char c = line[j];

                // Skip single-line comments
                if (!inString && j + 1 < line.Length && line[j] == '/' && line[j + 1] == '/')
                    break;
                if (!inString && c == '#')
                    break;

                // Track string state
                if (c == '"' && (j == 0 || line[j - 1] != '\\'))
                {
                    if (!inString)
                    {
                        inString = true;
                        stringStart = j;
                    }
                    else
                    {
                        inString = false;
                        stringStart = -1;
                    }
                }
            }

            // If still in string at end of line, it's unclosed
            if (inString && stringStart >= 0)
            {
                _fixes.Add(new SourceFix(
                    i + 1,
                    line.Length + 1,
                    0,
                    "",
                    "\"",
                    $"Unclosed string literal starting at column {stringStart + 1}",
                    FixCategory.Error
                ));
            }
        }
    }

    /// <summary>
    /// Fix missing commas in arrays, function calls, and dict literals
    /// This is one of the most common syntax errors (22% according to research)
    /// </summary>
    private void FixMissingCommas(HashSet<int> linesToSkip)
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            // Skip lines that already have keyword fixes to avoid position conflicts
            if (linesToSkip.Contains(i + 1)) continue;
            var line = _lines[i];
            var codePart = GetCodePart(line);

            // Pattern 1: Array literals with missing commas [1 2 3] or ["a" "b"]
            FixMissingCommasInBrackets(i, codePart, '[', ']', "array literal");

            // Pattern 2: Function call arguments func(a b c) or func(1 2)
            FixMissingCommasInBrackets(i, codePart, '(', ')', "function arguments");

            // Pattern 3: Dict literals {a: 1 b: 2}
            FixMissingCommasInDictLiteral(i, codePart);

            // Pattern 4: Function parameters fn foo(a b) instead of fn foo(a, b)
            FixMissingCommasInFunctionParams(i, codePart);
        }
    }

    private void FixMissingCommasInBrackets(int lineIdx, string code, char open, char close, string context)
    {
        int depth = 0;
        int start = -1;

        for (int j = 0; j < code.Length; j++)
        {
            if (code[j] == open)
            {
                if (depth == 0) start = j;
                depth++;
            }
            else if (code[j] == close)
            {
                depth--;
                if (depth == 0 && start >= 0)
                {
                    // Extract content between brackets
                    var content = code.Substring(start + 1, j - start - 1);
                    FindMissingCommasInContent(lineIdx, start + 1, content, context);
                    start = -1;
                }
            }
        }
    }

    private void FindMissingCommasInContent(int lineIdx, int offset, string content, string context)
    {
        if (string.IsNullOrWhiteSpace(content)) return;
        
        // If content has commas but also has nested brackets, process the nested brackets
        if (content.Contains(',') && content.Contains('['))
        {
            // Process nested array brackets
            int depth = 0;
            int start = -1;
            for (int j = 0; j < content.Length; j++)
            {
                if (content[j] == '[')
                {
                    if (depth == 0) start = j;
                    depth++;
                }
                else if (content[j] == ']')
                {
                    depth--;
                    if (depth == 0 && start >= 0)
                    {
                        var innerContent = content.Substring(start + 1, j - start - 1);
                        FindMissingCommasInContent(lineIdx, offset + start + 1, innerContent, "array literal");
                        start = -1;
                    }
                }
            }
            return;
        }
        
        if (content.Contains(',')) return; // Already has commas - skip to avoid double-fixing
        
        // Safe comma insertion for simple numeric arrays: [1 2 3] -> [1, 2, 3]
        var trimmed = content.Trim();
        
        // Pattern 1: Pure numbers separated by whitespace
        if (Regex.IsMatch(trimmed, @"^-?\d+\.?\d*(\s+-?\d+\.?\d*)+$"))
        {
            // Split by whitespace and rebuild with commas
            var numbers = Regex.Split(trimmed, @"\s+");
            if (numbers.Length >= 2)
            {
                var fixedContent = string.Join(", ", numbers);
                _fixes.Add(new SourceFix(
                    lineIdx + 1,
                    offset + 1,
                    content.Length,
                    content,
                    fixedContent,
                    $"Add commas between {context} elements",
                    FixCategory.Warning
                ));
            }
            return;
        }
        
        // Pattern 2: Pure string literals separated by whitespace
        if (Regex.IsMatch(trimmed, @"^""[^""]*""(\s+""[^""]*"")+$"))
        {
            var strings = Regex.Matches(trimmed, @"""[^""]*""");
            if (strings.Count >= 2)
            {
                var fixedContent = string.Join(", ", strings.Cast<Match>().Select(m => m.Value));
                _fixes.Add(new SourceFix(
                    lineIdx + 1,
                    offset + 1,
                    content.Length,
                    content,
                    fixedContent,
                    $"Add commas between {context} elements",
                    FixCategory.Warning
                ));
            }
            return;
        }
        
        // Pattern 3: Nested arrays like [1 2] [3 4] or [[a]] [[b]]
        if (Regex.IsMatch(trimmed, @"^\[.*?\](\s+\[.*?\])+$"))
        {
            // Extract each [...] element
            var arrayElements = Regex.Matches(trimmed, @"\[[^\[\]]*\]|\[\[.*?\]\]");
            if (arrayElements.Count >= 2)
            {
                var fixedContent = string.Join(", ", arrayElements.Cast<Match>().Select(m => m.Value));
                _fixes.Add(new SourceFix(
                    lineIdx + 1,
                    offset + 1,
                    content.Length,
                    content,
                    fixedContent,
                    $"Add commas between {context} elements",
                    FixCategory.Warning
                ));
            }
            return;
        }
        
        // Pattern 4: Simple identifiers (true, false, null, variable names)
        if (Regex.IsMatch(trimmed, @"^[a-zA-Z_]\w*(\s+[a-zA-Z_]\w*)+$"))
        {
            var idents = Regex.Split(trimmed, @"\s+");
            if (idents.Length >= 2)
            {
                var fixedContent = string.Join(", ", idents);
                _fixes.Add(new SourceFix(
                    lineIdx + 1,
                    offset + 1,
                    content.Length,
                    content,
                    fixedContent,
                    $"Add commas between {context} elements",
                    FixCategory.Warning
                ));
            }
            return;
        }
        
        // For mixed content, just warn without auto-fixing
        if (Regex.IsMatch(trimmed, @"\S\s+\S"))
        {
            _fixes.Add(new SourceFix(
                lineIdx + 1,
                offset + 1,
                0,
                "",
                "",
                $"Array appears to have missing commas - please add manually",
                FixCategory.Suggestion
            ));
        }
    }
    
    /// <summary>
    /// Tokenize array/object content into value tokens
    /// </summary>
    private List<(int start, int end, bool isValue)> TokenizeArrayContent(string content)
    {
        var tokens = new List<(int start, int end, bool isValue)>();
        int i = 0;
        
        while (i < content.Length)
        {
            // Skip whitespace
            while (i < content.Length && char.IsWhiteSpace(content[i])) i++;
            if (i >= content.Length) break;
            
            int start = i;
            bool isValue = false;
            
            // String literal
            if (content[i] == '"')
            {
                i++;
                while (i < content.Length && content[i] != '"')
                {
                    if (content[i] == '\\' && i + 1 < content.Length) i++;
                    i++;
                }
                if (i < content.Length) i++; // closing quote
                isValue = true;
            }
            // Number
            else if (char.IsDigit(content[i]) || (content[i] == '-' && i + 1 < content.Length && char.IsDigit(content[i + 1])))
            {
                while (i < content.Length && (char.IsDigit(content[i]) || content[i] == '.' || content[i] == 'e' || content[i] == 'E' || content[i] == '-' || content[i] == '+'))
                    i++;
                isValue = true;
            }
            // Identifier or keyword (true, false, null)
            else if (char.IsLetter(content[i]) || content[i] == '_')
            {
                while (i < content.Length && (char.IsLetterOrDigit(content[i]) || content[i] == '_'))
                    i++;
                isValue = true;
            }
            // Punctuation (comma, colon, etc.)
            else if (content[i] == ',' || content[i] == ':')
            {
                i++;
                isValue = false;
            }
            // Nested structure - skip
            else if (content[i] == '[' || content[i] == '{' || content[i] == '(')
            {
                char open = content[i];
                char close = open == '[' ? ']' : (open == '{' ? '}' : ')');
                int depth = 1;
                i++;
                while (i < content.Length && depth > 0)
                {
                    if (content[i] == open) depth++;
                    else if (content[i] == close) depth--;
                    i++;
                }
                isValue = true;
            }
            else
            {
                i++; // Skip unknown char
            }
            
            if (i > start)
            {
                tokens.Add((start, i, isValue));
            }
        }
        
        return tokens;
    }

    private void FixMissingCommasInDictLiteral(int lineIdx, string code)
    {
        // Find dict literals: { key: value key2: value2 }
        var dictMatch = Regex.Match(code, @"\{([^{}]+)\}");
        if (!dictMatch.Success) return;

        var content = dictMatch.Groups[1].Value;
        
        // Skip if already has commas
        if (content.Contains(',')) return;

        // Parse key: value pairs and rebuild with commas
        // Pattern: key: value (where value can be string, number, identifier, or nested)
        var pairs = Regex.Matches(content, @"(\w+)\s*:\s*(""[^""]*""|-?\d+\.?\d*|\w+)");
        if (pairs.Count >= 2)
        {
            // Rebuild with commas
            var fixedPairs = string.Join(", ", pairs.Cast<Match>().Select(m => m.Value.Trim()));
            var fixedContent = " " + fixedPairs + " ";
            
            _fixes.Add(new SourceFix(
                lineIdx + 1,
                dictMatch.Index + 2,  // After the {
                content.Length,
                content,
                fixedContent,
                "Add commas between object entries",
                FixCategory.Warning
            ));
        }
        else if (pairs.Count == 0 && Regex.IsMatch(content, @"\w+\s*:\s*\S+\s+\w+\s*:"))
        {
            // Fallback: just warn if pattern detected but couldn't parse safely
            _fixes.Add(new SourceFix(
                lineIdx + 1,
                dictMatch.Index + 1,
                0,
                "",
                "",
                "Object literal appears to have missing commas - please add manually",
                FixCategory.Suggestion
            ));
        }
    }

    private void FixMissingCommasInFunctionParams(int lineIdx, string code)
    {
        // Match function definition: fn name(params)
        var fnMatch = Regex.Match(code, @"\bfn\s+\w+\s*\(([^)]+)\)");
        if (!fnMatch.Success) return;

        var paramList = fnMatch.Groups[1].Value;

        // Skip if already has commas
        if (paramList.Contains(',')) return;

        // Parse parameters: either "name" or "name: type" separated by whitespace
        var paramMatches = Regex.Matches(paramList.Trim(), @"\w+(?:\s*:\s*\w+)?");
        if (paramMatches.Count >= 2)
        {
            // Rebuild with commas
            var fixedParams = string.Join(", ", paramMatches.Cast<Match>().Select(m => m.Value.Trim()));
            
            _fixes.Add(new SourceFix(
                lineIdx + 1,
                fnMatch.Groups[1].Index + 1,
                paramList.Length,
                paramList,
                fixedParams,
                "Add commas between function parameters",
                FixCategory.Warning
            ));
        }
    }

    /// <summary>
    /// Fix missing colons in function parameters: fn foo(x int) -> fn foo(x: int)
    /// </summary>
    private void FixMissingColonsInParameters()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);

            // Match function definition parameters without colons
            var fnMatch = Regex.Match(codePart, @"\bfn\s+\w+\s*\(([^)]+)\)");
            if (!fnMatch.Success) continue;

            var paramsStr = fnMatch.Groups[1].Value;
            var paramsStart = fnMatch.Groups[1].Index;

            // Look for pattern: identifier space identifier (without colon)
            // e.g., "x int" or "name string"
            var matches = Regex.Matches(paramsStr, @"\b(\w+)\s+(" + string.Join("|", BuiltinTypes) + @")\b");
            foreach (Match match in matches)
            {
                // Check if there's already a colon
                var beforeType = paramsStr.Substring(0, match.Groups[2].Index);
                if (beforeType.TrimEnd().EndsWith(":")) continue;

                var insertPos = paramsStart + match.Groups[1].Index + match.Groups[1].Length;
                _fixes.Add(new SourceFix(
                    i + 1,
                    insertPos + 1,
                    0,
                    "",
                    ":",
                    $"Missing colon before type annotation",
                    FixCategory.Error
                ));
            }
        }
    }

    /// <summary>
    /// Fix wrong comment syntax: // should be # in NSL
    /// </summary>
    private void FixWrongCommentSyntax()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];

            // Find // that's not inside a string
            bool inString = false;
            for (int j = 0; j < line.Length - 1; j++)
            {
                if (line[j] == '"' && (j == 0 || line[j - 1] != '\\'))
                    inString = !inString;

                if (!inString && line[j] == '/' && line[j + 1] == '/')
                {
                    // Check if this is integer division (has operands on both sides)
                    var before = line.Substring(0, j).TrimEnd();
                    var after = line.Substring(j + 2).TrimStart();

                    // If there's an operand before and after, it's integer division
                    bool isIntDiv = Regex.IsMatch(before, @"[\w\)\]]\s*$") &&
                                    Regex.IsMatch(after, @"^[\w\(\[]");

                    if (!isIntDiv)
                    {
                        _fixes.Add(new SourceFix(
                            i + 1,
                            j + 1,
                            2,
                            "//",
                            "#",
                            "Use # for comments (// is integer division in NSL)",
                            FixCategory.Error
                        ));
                    }
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Fix common keyword typos
    /// </summary>
    private void FixCommonKeywordTypos()
    {
        var keywordTypos = new Dictionary<string, string>
        {
            // Return typos
            { "retrun", "return" },
            { "reutrn", "return" },
            { "retrn", "return" },
            { "retunr", "return" },
            { "returm", "return" },

            // Function typos
            { "fucntion", "fn" },
            { "funtion", "fn" },
            { "funciton", "fn" },
            { "func", "fn" },
            { "function", "fn" },
            { "def", "fn" },

            // Boolean typos
            { "ture", "true" },
            { "treu", "true" },
            { "flase", "false" },
            { "fasle", "false" },
            { "fales", "false" },

            // Print typos
            { "pritn", "print" },
            { "prnt", "print" },
            { "pirnt", "print" },
            { "prnit", "print" },

            // Let/mut typos
            { "lte", "let" },
            { "elt", "let" },
            { "mtu", "mut" },

            // Control flow typos
            { "whlie", "while" },
            { "whlile", "while" },
            { "eles", "else" },
            { "esle", "else" },
            { "elseif", "else if" },
            // Note: "elif" handled by ConvertPythonStyleBlocks()

            // Import typos
            { "improt", "import" },
            { "imoprt", "import" },
            { "ipmort", "import" },

            // Null typos
            { "nul", "null" },
            { "nulll", "null" },
            { "nil", "null" },
            { "none", "null" },
            { "None", "null" },
            { "NULL", "null" },

            // Common function typos
            { "langth", "length" },
            { "lenght", "length" },
        };

        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);
            codePart = RemoveStringLiterals(codePart);

            foreach (var (typo, correct) in keywordTypos)
            {
                var pattern = $@"\b{Regex.Escape(typo)}\b";
                var match = Regex.Match(codePart, pattern, RegexOptions.None);
                if (match.Success)
                {
                    _fixes.Add(new SourceFix(
                        i + 1,
                        match.Index + 1,
                        typo.Length,
                        typo,
                        correct,
                        $"Fix typo: '{typo}' -> '{correct}'",
                        FixCategory.Error
                    ));
                }
            }
        }
    }

    /// <summary>
    /// Fix operator typos
    /// </summary>
    private void FixOperatorTypos()
    {
        var operatorTypos = new Dictionary<string, string>
        {
            // Comparison operator typos
            { "=<", "<=" },
            { "=>", ">=" },
            { "=!", "!=" },
            { "!==", "!=" },
            { "===", "==" },
            { "<>", "!=" },

            // Logical operator alternatives (if NSL uses and/or)
            // Uncomment these if NSL uses words instead of symbols:
            // { "&&", "and" },
            // { "||", "or" },

            // Arrow typos
            { "->", "=>" }, // If NSL uses => for lambdas
            { "<-", "=" },  // Assignment confusion

            // Increment/decrement (not supported, suggest alternatives)
            { "++", "+ 1" },
            { "--", "- 1" },
        };

        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);
            codePart = RemoveStringLiterals(codePart);

            foreach (var (typo, correct) in operatorTypos)
            {
                var idx = codePart.IndexOf(typo);
                if (idx >= 0)
                {
                    // Skip if it's part of a valid operator sequence
                    // e.g., don't flag => in arrow functions if valid
                    if (typo == "=>" && IsValidArrowContext(codePart, idx))
                        continue;

                    _fixes.Add(new SourceFix(
                        i + 1,
                        idx + 1,
                        typo.Length,
                        typo,
                        correct,
                        $"Fix operator: '{typo}' -> '{correct}'",
                        FixCategory.Error
                    ));
                }
            }
        }
    }

    private bool IsValidArrowContext(string code, int arrowIdx)
    {
        // Check if => is used in a valid lambda/arrow function context
        var before = code.Substring(0, arrowIdx).TrimEnd();
        return before.EndsWith(")") || Regex.IsMatch(before, @"\w$");
    }

    /// <summary>
    /// Fix wrong keywords from other languages
    /// </summary>
    private void FixWrongKeywords()
    {
        var wrongKeywords = new Dictionary<string, string>
        {
            // Variable declarations
            { "var", "let" },
            { "const", "let" },
            { "val", "let" },

            // Loops
            { "foreach", "for" },
            { "loop", "while" },

            // Other language keywords
            { "class", "struct" },
            { "interface", "trait" },
            // Note: "void" removal disabled - can leave invalid syntax
            // { "void", "" },
            { "async", "" }, // Different syntax in NSL
            { "await", "" },

            // Python-isms
            { "def", "fn" },
            { "pass", "# pass" },
            { "self", "this" },

            // Ruby/other
            { "end", "}" },
            { "do", "{" },
            { "begin", "{" },
            { "then", "{" },
            
            // TypeScript/JavaScript
            { "function", "fn" },
            // Note: typeof/instanceof need special handling, not simple replacement
            { "undefined", "null" },
            { "export", "" }, // No exports in NSL scripts
            { "require", "use" },
            
            // Go
            { "func", "fn" },
            { "package", "" }, // No packages in NSL
            { "nil", "null" },
            { "defer", "" }, // No defer in NSL
            
            // Rust
            { "pub", "" }, // No visibility modifiers
            { "impl", "fn" }, // Implementation blocks
            { "match", "switch" },
            { "Some", "" },
            { "None", "null" },
            { "Ok", "" },
            { "Err", "" },
            { "unwrap", "" },
            { "println!", "print" },
            { "print!", "print" },
            { "vec!", "[]" },
            
            // C/C++
            { "printf", "print" },
            { "cout", "print" },
            { "cin", "input" },
            { "NULL", "null" },
            { "nullptr", "null" },
            { "auto", "let" },
            { "int", "number" },
            { "float", "number" },
            { "double", "number" },
            { "char", "string" },
            { "bool", "bool" },
            { "string", "string" },
        };

        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);
            codePart = RemoveStringLiterals(codePart);

            foreach (var (wrong, correct) in wrongKeywords)
            {
                // Match as whole word
                var pattern = $@"\b{Regex.Escape(wrong)}\b";
                var match = Regex.Match(codePart, pattern);
                if (match.Success)
                {
                    var desc = string.IsNullOrEmpty(correct)
                        ? $"'{wrong}' is not used in NSL"
                        : $"Use '{correct}' instead of '{wrong}'";

                    _fixes.Add(new SourceFix(
                        i + 1,
                        match.Index + 1,
                        wrong.Length,
                        wrong,
                        correct,
                        desc,
                        FixCategory.Error
                    ));
                }
            }
            
            // Special handling for lambda expressions: lambda x: expr or lambda x, y: expr
            // Convert to: fn lambda_N(x) { return expr } or fn lambda_N(x, y) { return expr }
            var lambdaMatch = Regex.Match(codePart, @"\blambda\s+([\w\s,]+)\s*:\s*(.+)$");
            if (lambdaMatch.Success)
            {
                var paramsRaw = lambdaMatch.Groups[1].Value.Trim();
                var body = lambdaMatch.Groups[2].Value.TrimEnd();
                
                // Parse parameters - could be "x" or "x, y" or "x,y,z"
                var paramList = paramsRaw.Split(',')
                    .Select(p => p.Trim())
                    .Where(p => !string.IsNullOrEmpty(p))
                    .ToList();
                var paramsFormatted = string.Join(", ", paramList);
                
                // Generate unique lambda name based on line number
                var lambdaName = $"lambda_{i + 1}";
                _fixes.Add(new SourceFix(
                    i + 1,
                    lambdaMatch.Index + 1,
                    lambdaMatch.Length,
                    lambdaMatch.Value,
                    $"fn {lambdaName}({paramsFormatted}) {{ return {body} }}",
                    $"Convert Python lambda to NSL function '{lambdaName}'",
                    FixCategory.Warning
                ));
            }
        }
    }

    /// <summary>
    /// Fix for loop syntax errors
    /// </summary>
    private void FixForLoopSyntax()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);

            // Pattern 1: Missing 'in' keyword: for x arr -> for x in arr
            var missingIn = Regex.Match(codePart, @"\bfor\s+(\w+)\s+(\w+|\[)");
            if (missingIn.Success)
            {
                var varName = missingIn.Groups[1].Value;
                var iterName = missingIn.Groups[2].Value;

                // Check it's not already "for x in"
                if (iterName != "in")
                {
                    var insertPos = missingIn.Groups[1].Index + missingIn.Groups[1].Length;
                    _fixes.Add(new SourceFix(
                        i + 1,
                        insertPos + 1,
                        0,
                        "",
                        " in",
                        "Missing 'in' keyword in for loop",
                        FixCategory.Error
                    ));
                }
            }

            // Pattern 2: C-style for loop: for (let i = 0; i < n; i++)
            var cStyleFor = Regex.Match(codePart, @"\bfor\s*\(\s*(let|var)");
            if (cStyleFor.Success)
            {
                _fixes.Add(new SourceFix(
                    i + 1,
                    cStyleFor.Index + 1,
                    cStyleFor.Length,
                    cStyleFor.Value,
                    "# C-style for loops not supported. Use: for i in range(0, n)",
                    "NSL uses 'for x in iterable' syntax",
                    FixCategory.Warning
                ));
            }

            // Pattern 3: Python-style for with colon: for x in arr:
            var pythonFor = Regex.Match(codePart, @"\bfor\s+\w+\s+in\s+\w+\s*:");
            if (pythonFor.Success)
            {
                var colonPos = codePart.IndexOf(':', pythonFor.Index);
                _fixes.Add(new SourceFix(
                    i + 1,
                    colonPos + 1,
                    1,
                    ":",
                    " {",
                    "Use { } for blocks instead of :",
                    FixCategory.Error
                ));
            }
        }
    }

    /// <summary>
    /// Convert Python-style indentation blocks to NSL brace blocks
    /// Handles if/elif/else/for/while with colons
    /// </summary>
    private void ConvertPythonStyleBlocks()
    {
        // Find all Python-style control statements
        var pythonBlocks = new List<(int line, int indent, string keyword, string condition)>();
        
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);
            
            // Skip if already has braces
            if (codePart.Contains("{")) continue;
            
            // Match Python-style: if/elif/else/for/while ... :
            var match = Regex.Match(codePart, @"^(\s*)(if|elif|else|for|while)\b(.*):\s*$");
            if (match.Success)
            {
                var indent = match.Groups[1].Value.Length;
                var keyword = match.Groups[2].Value;
                var condition = match.Groups[3].Value;
                pythonBlocks.Add((i, indent, keyword, condition));
            }
        }
        
        if (pythonBlocks.Count == 0) return;
        
        // Check for nested blocks (different indentation levels) - too complex, warn only
        var indentLevels = pythonBlocks.Select(b => b.indent).Distinct().ToList();
        if (indentLevels.Count > 1)
        {
            // Nested Python blocks - add warning and skip auto-fix
            _fixes.Add(new SourceFix(
                pythonBlocks[0].line + 1,
                1,
                0,
                "",
                "",
                "Nested Python-style blocks detected - please convert to NSL braces manually",
                FixCategory.Suggestion
            ));
            return;
        }
        
        // Process blocks and generate fixes
        for (int idx = 0; idx < pythonBlocks.Count; idx++)
        {
            var (lineNum, indent, keyword, condition) = pythonBlocks[idx];
            var line = _lines[lineNum];
            var indentStr = new string(' ', indent);
            
            string newLine;
            string desc;
            
            // Remove trailing colon from condition if present
            condition = condition.TrimEnd().TrimEnd(':').TrimEnd();
            
            if (keyword == "elif")
            {
                // elif condition: -> } else if condition {
                newLine = $"{indentStr}}} else if{condition} {{";
                desc = "Convert Python 'elif' to NSL '} else if ... {'";
            }
            else if (keyword == "else")
            {
                // else: -> } else {
                newLine = $"{indentStr}}} else {{";
                desc = "Convert Python 'else:' to NSL '} else {'";
            }
            else
            {
                // if/for/while condition: -> if/for/while condition {
                newLine = $"{indentStr}{keyword}{condition} {{";
                desc = $"Convert Python '{keyword}:' to NSL '{keyword} {{'";
            }
            
            _fixes.Add(new SourceFix(
                lineNum + 1,
                1,
                line.Length,
                line,
                newLine,
                desc,
                FixCategory.Warning
            ));
        }
        
        // Add closing braces at end of indented blocks
        AddClosingBracesForPythonBlocks(pythonBlocks);
    }
    
    /// <summary>
    /// Add closing braces where Python indentation blocks end
    /// </summary>
    private void AddClosingBracesForPythonBlocks(List<(int line, int indent, string keyword, string condition)> blocks)
    {
        if (blocks.Count == 0) return;
        
        // Track which blocks need closing braces and where
        var closingBraces = new HashSet<int>(); // Line indices that need closing braces
        
        // Find the last block in each if/elif/else chain (only those need closing braces)
        for (int idx = 0; idx < blocks.Count; idx++)
        {
            var (blockLine, blockIndent, keyword, _) = blocks[idx];
            
            // Check if next block is at same indentation and is elif/else (part of same chain)
            bool isLastInChain = true;
            if (idx + 1 < blocks.Count)
            {
                var (nextLine, nextIndent, nextKeyword, _) = blocks[idx + 1];
                if (nextIndent == blockIndent && (nextKeyword == "elif" || nextKeyword == "else"))
                {
                    isLastInChain = false;
                }
            }
            
            if (!isLastInChain) continue;
            
            // Find where this block/chain ends based on indentation
            int blockEnd = _lines.Length - 1;
            
            for (int i = blockLine + 1; i < _lines.Length; i++)
            {
                var line = _lines[i];
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                var lineIndent = line.Length - line.TrimStart().Length;
                var codePart = GetCodePart(line);
                
                // Block ends when we find a line at same or lower indentation
                // that's not part of our elif/else chain
                if (lineIndent <= blockIndent)
                {
                    var isChainContinuation = Regex.IsMatch(codePart, @"^\s*(elif|else)\b.*:\s*$");
                    if (!isChainContinuation)
                    {
                        blockEnd = i - 1;
                        break;
                    }
                }
            }
            
            // Find the last non-empty line in the block
            while (blockEnd > blockLine && string.IsNullOrWhiteSpace(_lines[blockEnd]))
            {
                blockEnd--;
            }
            
            if (blockEnd > blockLine)
            {
                closingBraces.Add(blockEnd);
            }
        }
        
        // Add fixes for closing braces
        foreach (var afterLine in closingBraces.OrderByDescending(x => x))
        {
            var currentLine = _lines[afterLine];
            var indent = currentLine.Length - currentLine.TrimStart().Length;
            // Go back to block indent level (one level less than content)
            indent = Math.Max(0, indent - 2);
            var indentStr = new string(' ', indent);
            
            // Append closing brace on a new line
            _fixes.Add(new SourceFix(
                afterLine + 1,
                currentLine.Length + 1,
                0,
                "",
                $"\n{indentStr}}}",
                "Add closing brace for Python-style block",
                FixCategory.Warning
            ));
        }
    }

    /// <summary>
    /// Fix control flow syntax errors including Python-style indentation blocks
    /// </summary>
    private void FixControlFlowSyntax()
    {
        // First pass: Convert Python-style blocks (if/elif/else/for/while with colons)
        ConvertPythonStyleBlocks();
        
        // Second pass: Fix remaining issues
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);

            // Skip lines that already have braces
            if (codePart.Contains("{"))
                continue;

            // Pattern 3: Triple equals (JavaScript) - safe to fix
            var tripleEq = codePart.IndexOf("===");
            if (tripleEq >= 0)
            {
                _fixes.Add(new SourceFix(
                    i + 1,
                    tripleEq + 1,
                    3,
                    "===",
                    "==",
                    "NSL uses == for equality (not ===)",
                    FixCategory.Error
                ));
            }

            // Pattern 4: !== (JavaScript) - safe to fix
            var notTripleEq = codePart.IndexOf("!==");
            if (notTripleEq >= 0)
            {
                _fixes.Add(new SourceFix(
                    i + 1,
                    notTripleEq + 1,
                    3,
                    "!==",
                    "!=",
                    "NSL uses != for inequality (not !==)",
                    FixCategory.Error
                ));
            }
        }
    }

    /// <summary>
    /// Detect reserved words used as variable names
    /// </summary>
    private void FixReservedWordVariables()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);

            // Match variable declarations
            var letMatch = Regex.Match(codePart, @"\b(let|mut)\s+(\w+)");
            if (letMatch.Success)
            {
                var varName = letMatch.Groups[2].Value;

                if (Keywords.Contains(varName))
                {
                    _fixes.Add(new SourceFix(
                        i + 1,
                        letMatch.Groups[2].Index + 1,
                        varName.Length,
                        varName,
                        varName + "_",
                        $"'{varName}' is a reserved keyword - consider renaming",
                        FixCategory.Error
                    ));
                }
                else if (BuiltinFunctions.Contains(varName))
                {
                    _fixes.Add(new SourceFix(
                        i + 1,
                        letMatch.Groups[2].Index + 1,
                        varName.Length,
                        varName,
                        varName + "_var",
                        $"'{varName}' shadows built-in function - consider renaming",
                        FixCategory.Warning
                    ));
                }
            }
        }
    }

    /// <summary>
    /// Fix style issues (optional improvements)
    /// </summary>
    private void FixStyleIssues()
    {
        for (int i = 0; i < _lines.Length; i++)
        {
            var line = _lines[i];
            var codePart = GetCodePart(line);

            // Pattern 1: No space around operators (style)
            // x=5 -> x = 5, x+y -> x + y
            var noSpaceAssign = Regex.Match(codePart, @"(\w)=([^=])");
            if (noSpaceAssign.Success)
            {
                var pos = noSpaceAssign.Index;
                // Check it's not <=, >=, ==, !=
                if (pos > 0 && !"<>!=".Contains(codePart[pos]))
                {
                    _fixes.Add(new SourceFix(
                        i + 1,
                        pos + 2,
                        1,
                        "=",
                        " = ",
                        "Add spaces around assignment operator",
                        FixCategory.Style
                    ));
                }
            }

            // Pattern 2: Trailing whitespace
            if (line.Length > 0 && line.EndsWith(" "))
            {
                var trimmedLength = line.TrimEnd().Length;
                _fixes.Add(new SourceFix(
                    i + 1,
                    trimmedLength + 1,
                    line.Length - trimmedLength,
                    line.Substring(trimmedLength),
                    "",
                    "Remove trailing whitespace",
                    FixCategory.Style
                ));
            }

            // Pattern 3: Multiple consecutive blank lines
            if (string.IsNullOrWhiteSpace(line) && i > 0 && string.IsNullOrWhiteSpace(_lines[i - 1]))
            {
                // Check if this is 3+ consecutive blank lines
                if (i > 1 && string.IsNullOrWhiteSpace(_lines[i - 2]))
                {
                    _fixes.Add(new SourceFix(
                        i + 1,
                        1,
                        line.Length,
                        line,
                        "",
                        "Remove excessive blank lines",
                        FixCategory.Style
                    ));
                }
            }

            // Pattern 4: Inconsistent indentation (tabs vs spaces)
            if (line.StartsWith("\t") && line.Contains("    "))
            {
                _fixes.Add(new SourceFix(
                    i + 1,
                    1,
                    0,
                    "",
                    "",
                    "Mixed tabs and spaces in indentation",
                    FixCategory.Style
                ));
            }

            // Pattern 5: Trailing comma in last element (sometimes causes issues)
            var trailingComma = Regex.Match(codePart, @",\s*[\]\}]\s*$");
            if (trailingComma.Success)
            {
                // This might be intentional, just flag as suggestion
                _fixes.Add(new SourceFix(
                    i + 1,
                    trailingComma.Index + 1,
                    1,
                    ",",
                    "",
                    "Remove trailing comma (optional)",
                    FixCategory.Suggestion
                ));
            }
        }
    }

    /// <summary>
    /// Get the code portion of a line (excluding comments)
    /// </summary>
    private string GetCodePart(string line)
    {
        var codePart = line;

        // Remove # comments (but not inside strings)
        var hashIdx = FindCommentStart(codePart, '#');
        if (hashIdx >= 0)
            codePart = codePart.Substring(0, hashIdx);

        return codePart;
    }

    /// <summary>
    /// Try to generate fix from compiler error message
    /// </summary>
    private void TryFixFromErrorMessage(string error)
    {
        // Pattern: "Undefined variable: xyz"
        var undefinedVar = Regex.Match(error, @"Undefined variable: (\w+)");
        if (undefinedVar.Success)
        {
            var varName = undefinedVar.Groups[1].Value;
            var closest = FindClosestMatch(varName, _knownVariables.Union(_knownFunctions));
            if (closest != null)
            {
                // Find where this variable is used
                for (int i = 0; i < _lines.Length; i++)
                {
                    var match = Regex.Match(_lines[i], $@"\b{Regex.Escape(varName)}\b");
                    if (match.Success)
                    {
                        _fixes.Add(new SourceFix(
                            i + 1,
                            match.Index + 1,
                            varName.Length,
                            varName,
                            closest,
                            $"Fix undefined variable (did you mean '{closest}'?)",
                            FixCategory.Error
                        ));
                    }
                }
            }
        }

        // Pattern: "Type mismatch" or "expected X, got Y"
        var typeMismatch = Regex.Match(error, @"expected (\w+), got (\w+)");
        if (typeMismatch.Success)
        {
            var expected = typeMismatch.Groups[1].Value;
            var got = typeMismatch.Groups[2].Value;

            // If expected number and got int-like, suggest .0 conversion
            if (expected == "number" && got == "int")
            {
                // This would require more context to fix properly
            }
        }

        // Pattern: "Left operand must be numeric, got any"
        // This usually means a recursive call issue - already fixed in type checker
    }

    /// <summary>
    /// Apply all fixes to source and return new source
    /// Handles overlapping fixes by only applying non-conflicting ones
    /// Supports multi-line replacements (fixes containing \n)
    /// </summary>
    public string ApplyFixes(FixCategory minCategory = FixCategory.Error)
    {
        if (_fixes.Count == 0)
            return _source;

        // Separate line-insertion fixes from regular fixes
        var regularFixes = new List<SourceFix>();
        var lineInsertionFixes = new List<SourceFix>();
        
        foreach (var fix in _fixes.Where(f => f.Category <= minCategory))
        {
            if (fix.ReplacementText.Contains('\n'))
                lineInsertionFixes.Add(fix);
            else
                regularFixes.Add(fix);
        }

        // Sort regular fixes by priority then position (reverse order)
        var candidateFixes = regularFixes
            .OrderBy(f => f.Category)
            .ThenByDescending(f => f.Line)
            .ThenByDescending(f => f.Column)
            .ToList();

        var lines = _source.Split('\n').ToList();
        
        // Track which regions have been modified per line to prevent overlapping fixes
        var modifiedRegions = new Dictionary<int, List<(int start, int end)>>();

        // Apply regular fixes first
        foreach (var fix in candidateFixes)
        {
            var lineIdx = fix.Line - 1;
            if (lineIdx < 0 || lineIdx >= lines.Count)
                continue;

            var line = lines[lineIdx];
            var col = fix.Column - 1;

            if (col < 0 || col > line.Length)
                continue;

            var fixStart = col;
            var fixEnd = col + fix.Length;

            // Check if this fix overlaps with any already-applied fix on this line
            if (modifiedRegions.TryGetValue(lineIdx, out var regions))
            {
                bool overlaps = false;
                foreach (var (start, end) in regions)
                {
                    if (fixStart < end && fixEnd > start)
                    {
                        overlaps = true;
                        break;
                    }
                }
                if (overlaps)
                    continue;
            }

            // Validate the fix - ensure original text matches what we expect
            if (fix.Length > 0 && col + fix.Length <= line.Length)
            {
                var actualText = line.Substring(col, fix.Length);
                if (!string.IsNullOrEmpty(fix.OriginalText) && actualText != fix.OriginalText)
                {
                    // Original text doesn't match - skip this fix to prevent corruption
                    continue;
                }
            }
            
            // Apply the fix
            var before = line.Substring(0, col);
            var after = col + fix.Length <= line.Length
                ? line.Substring(col + fix.Length)
                : "";

            var newLine = before + fix.ReplacementText + after;
            
            if (!modifiedRegions.ContainsKey(lineIdx))
                modifiedRegions[lineIdx] = new List<(int, int)>();
            
            var lengthDiff = fix.ReplacementText.Length - fix.Length;
            modifiedRegions[lineIdx].Add((fixStart, fixStart + fix.ReplacementText.Length));
            
            for (int i = 0; i < modifiedRegions[lineIdx].Count - 1; i++)
            {
                var (s, e) = modifiedRegions[lineIdx][i];
                if (s > fixStart)
                {
                    modifiedRegions[lineIdx][i] = (s + lengthDiff, e + lengthDiff);
                }
            }

            lines[lineIdx] = newLine;
        }

        // Apply line-insertion fixes (sorted by line descending to maintain indices)
        var sortedInsertions = lineInsertionFixes
            .OrderByDescending(f => f.Line)
            .ToList();
            
        foreach (var fix in sortedInsertions)
        {
            var lineIdx = fix.Line - 1;
            if (lineIdx < 0 || lineIdx >= lines.Count)
                continue;

            var line = lines[lineIdx];
            var col = fix.Column - 1;

            if (col < 0 || col > line.Length)
                continue;

            // Validate the fix before applying
            if (fix.Length > 0 && col + fix.Length <= line.Length)
            {
                var actualText = line.Substring(col, fix.Length);
                if (!string.IsNullOrEmpty(fix.OriginalText) && actualText != fix.OriginalText)
                {
                    continue; // Skip to prevent corruption
                }
            }
            
            // Apply the fix which may contain newlines
            var before = line.Substring(0, col);
            var after = col + fix.Length <= line.Length
                ? line.Substring(col + fix.Length)
                : "";

            var replacement = before + fix.ReplacementText + after;
            
            // Split replacement into multiple lines
            var newLines = replacement.Split('\n');
            
            // Replace the original line with the new lines
            lines.RemoveAt(lineIdx);
            for (int i = newLines.Length - 1; i >= 0; i--)
            {
                lines.Insert(lineIdx, newLines[i]);
            }
        }

        return string.Join('\n', lines);
    }

    /// <summary>
    /// Get a summary of fixes to be applied
    /// </summary>
    public string GetFixSummary()
    {
        if (_fixes.Count == 0)
            return "No fixes needed.";

        var summary = $"Found {_fixes.Count} potential fix(es):\n";
        foreach (var fix in _fixes.OrderBy(f => f.Line).ThenBy(f => f.Column))
        {
            summary += $"  {fix}\n";
        }
        return summary;
    }

    #region Helper Methods

    private string? InferType(string value)
    {
        value = value.Trim().TrimEnd(';', ')');

        // String literal
        if (value.StartsWith("\"") && value.EndsWith("\""))
            return "string";

        // Boolean
        if (value == "true" || value == "false")
            return "bool";

        // Number (float)
        if (Regex.IsMatch(value, @"^\d+\.\d*$") || Regex.IsMatch(value, @"^\d+\.?\d*e[+-]?\d+$", RegexOptions.IgnoreCase))
            return "number";

        // Integer (suggest as number since NSL uses floats)
        if (Regex.IsMatch(value, @"^\d+$"))
            return "number";

        // Function call - would need more context
        return null;
    }

    private string? FindClosestMatch(string input, IEnumerable<string> candidates)
    {
        string? best = null;
        int bestDistance = int.MaxValue;
        int bestLengthDiff = int.MaxValue;

        foreach (var candidate in candidates)
        {
            var distance = LevenshteinDistance(input.ToLower(), candidate.ToLower());
            var lengthDiff = Math.Abs(input.Length - candidate.Length);
            
            // Prefer matches with:
            // 1. Lower distance
            // 2. Similar length (tie-breaker)
            // 3. Only consider distance <= 2
            if (distance < bestDistance && distance <= 2)
            {
                bestDistance = distance;
                bestLengthDiff = lengthDiff;
                best = candidate;
            }
            else if (distance == bestDistance && lengthDiff < bestLengthDiff)
            {
                bestLengthDiff = lengthDiff;
                best = candidate;
            }
        }

        return best;
    }

    private static int LevenshteinDistance(string s1, string s2)
    {
        int[,] d = new int[s1.Length + 1, s2.Length + 1];

        for (int i = 0; i <= s1.Length; i++)
            d[i, 0] = i;
        for (int j = 0; j <= s2.Length; j++)
            d[0, j] = j;

        for (int i = 1; i <= s1.Length; i++)
        {
            for (int j = 1; j <= s2.Length; j++)
            {
                int cost = s1[i - 1] == s2[j - 1] ? 0 : 1;
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost
                );
            }
        }

        return d[s1.Length, s2.Length];
    }

    /// <summary>
    /// Get a detailed error message with code context and visual pointer
    /// </summary>
    public string GetErrorWithContext(SourceFix fix, int contextLines = 1)
    {
        var sb = new System.Text.StringBuilder();

        // Error header
        sb.AppendLine($"Error at line {fix.Line}, column {fix.Column}:");
        sb.AppendLine();

        // Show context lines before
        int startLine = Math.Max(0, fix.Line - 1 - contextLines);
        int endLine = Math.Min(_lines.Length - 1, fix.Line - 1 + contextLines);

        for (int i = startLine; i <= endLine; i++)
        {
            var lineNum = (i + 1).ToString().PadLeft(4);
            var marker = i == fix.Line - 1 ? " > " : "   ";
            sb.AppendLine($"{marker}{lineNum} | {_lines[i]}");

            // Add pointer on the error line
            if (i == fix.Line - 1)
            {
                var padding = new string(' ', 4 + 3 + 3 + fix.Column - 1); // lineNum + marker + " | " + column
                var pointer = fix.Length > 0 ? new string('^', fix.Length) : "^";
                sb.AppendLine($"{padding}{pointer}");
            }
        }

        // Add fix description
        sb.AppendLine();
        sb.AppendLine($"  {fix.Description}");

        if (!string.IsNullOrEmpty(fix.ReplacementText) && fix.OriginalText != fix.ReplacementText)
        {
            if (fix.Length > 0)
                sb.AppendLine($"  Suggested fix: {fix.OriginalText} -> {fix.ReplacementText}");
            else
                sb.AppendLine($"  Suggested fix: Add '{fix.ReplacementText}'");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Get all errors with context in a formatted string
    /// </summary>
    public string GetAllErrorsWithContext()
    {
        if (_fixes.Count == 0)
            return "No issues found - code looks good!";

        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Found {_fixes.Count} issue(s):");
        sb.AppendLine();

        foreach (var fix in _fixes.OrderBy(f => f.Line).ThenBy(f => f.Column))
        {
            sb.AppendLine(GetErrorWithContext(fix));
            sb.AppendLine();
        }

        sb.AppendLine("To auto-fix errors: nsl --fix <file>");
        return sb.ToString();
    }

    /// <summary>
    /// Static helper to format an error message with context from source
    /// </summary>
    public static string FormatErrorWithContext(string source, int line, int column, int length, string message, string? suggestion = null)
    {
        var lines = source.Split('\n');
        var sb = new System.Text.StringBuilder();

        sb.AppendLine($"Error at line {line}, column {column}:");
        sb.AppendLine();

        // Show the error line with context
        int startLine = Math.Max(0, line - 2);
        int endLine = Math.Min(lines.Length - 1, line);

        for (int i = startLine; i <= endLine; i++)
        {
            var lineNum = (i + 1).ToString().PadLeft(4);
            var marker = i == line - 1 ? " > " : "   ";
            sb.AppendLine($"{marker}{lineNum} | {lines[i]}");

            if (i == line - 1)
            {
                var padding = new string(' ', 4 + 3 + 3 + column - 1);
                var pointer = length > 0 ? new string('^', length) : "^";
                sb.AppendLine($"{padding}{pointer}");
            }
        }

        sb.AppendLine();
        sb.AppendLine($"  {message}");

        if (!string.IsNullOrEmpty(suggestion))
        {
            sb.AppendLine($"  Did you mean: {suggestion}?");
        }

        return sb.ToString();
    }

    #endregion
}


#region Multi-Language Support

    /// <summary>
    /// Detected programming language
    /// </summary>
    public enum DetectedLanguage
    {
        NSL,
        CSharp,
        Java,
        Python,
        JavaScript,
        TypeScript,
        Go,
        Rust,
        Ruby,
        PHP,
        Swift,
        Kotlin,
        Cpp,
        C,
        Scala,
        Haskell,
        Lua,
        Perl,
        R,
        SQL,
        HTML,
        CSS,
        Shell,
        PowerShell,
        Unknown
    }

    /// <summary>
    /// Multi-language auto-fix system - detects language and applies appropriate fixes
    /// </summary>
    public class MultiLanguageAutoFix
    {
        private readonly string _source;
        private readonly string[] _lines;
        private readonly string? _fileExtension;
        private readonly List<SourceFix> _fixes = new();
        private DetectedLanguage _detectedLanguage = DetectedLanguage.Unknown;

        /// <summary>Public API</summary>
        public IReadOnlyList<SourceFix> Fixes => _fixes;
        /// <summary>Public API</summary>
        public DetectedLanguage Language => _detectedLanguage;

        /// <summary>Public API</summary>
        public MultiLanguageAutoFix(string source, string? filePath = null)
        {
            _source = source;
            _lines = source.Split('\n');
            _fileExtension = filePath != null ? System.IO.Path.GetExtension(filePath).ToLower() : null;
            _detectedLanguage = DetectLanguage();
        }

        /// <summary>
        /// Detect the programming language from file extension and content
        /// </summary>
        private DetectedLanguage DetectLanguage()
        {
            // First try by file extension
            if (_fileExtension != null)
            {
                var lang = _fileExtension switch
                {
                    ".nsl" => DetectedLanguage.NSL,
                    ".cs" => DetectedLanguage.CSharp,
                    ".java" => DetectedLanguage.Java,
                    ".py" => DetectedLanguage.Python,
                    ".js" => DetectedLanguage.JavaScript,
                    ".jsx" => DetectedLanguage.JavaScript,
                    ".ts" => DetectedLanguage.TypeScript,
                    ".tsx" => DetectedLanguage.TypeScript,
                    ".go" => DetectedLanguage.Go,
                    ".rs" => DetectedLanguage.Rust,
                    ".rb" => DetectedLanguage.Ruby,
                    ".php" => DetectedLanguage.PHP,
                    ".swift" => DetectedLanguage.Swift,
                    ".kt" => DetectedLanguage.Kotlin,
                    ".kts" => DetectedLanguage.Kotlin,
                    ".cpp" => DetectedLanguage.Cpp,
                    ".cc" => DetectedLanguage.Cpp,
                    ".cxx" => DetectedLanguage.Cpp,
                    ".hpp" => DetectedLanguage.Cpp,
                    ".c" => DetectedLanguage.C,
                    ".h" => DetectedLanguage.C,
                    ".scala" => DetectedLanguage.Scala,
                    ".hs" => DetectedLanguage.Haskell,
                    ".lua" => DetectedLanguage.Lua,
                    ".pl" => DetectedLanguage.Perl,
                    ".pm" => DetectedLanguage.Perl,
                    ".r" => DetectedLanguage.R,
                    ".sql" => DetectedLanguage.SQL,
                    ".html" => DetectedLanguage.HTML,
                    ".htm" => DetectedLanguage.HTML,
                    ".css" => DetectedLanguage.CSS,
                    ".sh" => DetectedLanguage.Shell,
                    ".bash" => DetectedLanguage.Shell,
                    ".zsh" => DetectedLanguage.Shell,
                    ".ps1" => DetectedLanguage.PowerShell,
                    ".psm1" => DetectedLanguage.PowerShell,
                    _ => DetectedLanguage.Unknown
                };
                if (lang != DetectedLanguage.Unknown) return lang;
            }

            // Detect by content patterns
            var content = _source;
            
            // C# patterns
            if (Regex.IsMatch(content, @"\bnamespace\s+\w+|\busing\s+System|\bclass\s+\w+\s*:\s*\w+|\bpublic\s+(class|interface|struct|enum)"))
                return DetectedLanguage.CSharp;
            
            // Java patterns
            if (Regex.IsMatch(content, @"\bpackage\s+[\w.]+;|\bimport\s+java\.|\bpublic\s+class\s+\w+\s*(extends|implements)"))
                return DetectedLanguage.Java;
            
            // Python patterns
            if (Regex.IsMatch(content, @"\bdef\s+\w+\s*\(|\bimport\s+\w+|\bfrom\s+\w+\s+import|^\s*#.*coding", RegexOptions.Multiline))
                return DetectedLanguage.Python;
            
            // JavaScript/TypeScript patterns
            if (Regex.IsMatch(content, @"\bconst\s+\w+\s*=|\blet\s+\w+\s*=|\bfunction\s+\w+\s*\(|\b=>"))
            {
                if (Regex.IsMatch(content, @":\s*(string|number|boolean|any)\b|\binterface\s+\w+"))
                    return DetectedLanguage.TypeScript;
                return DetectedLanguage.JavaScript;
            }
            
            // Go patterns
            if (Regex.IsMatch(content, @"\bpackage\s+main|\bfunc\s+\w+\s*\(|\bimport\s+\("))
                return DetectedLanguage.Go;
            
            // Rust patterns
            if (Regex.IsMatch(content, @"\bfn\s+\w+\s*\(|\blet\s+mut\s+|\bimpl\s+\w+|\buse\s+std::"))
                return DetectedLanguage.Rust;
            
            // Ruby patterns
            if (Regex.IsMatch(content, @"\bdef\s+\w+|\bend\b|\brequire\s+['""]|\bclass\s+\w+\s*<"))
                return DetectedLanguage.Ruby;
            
            // PHP patterns
            if (Regex.IsMatch(content, @"<\?php|\$\w+\s*=|\bfunction\s+\w+\s*\("))
                return DetectedLanguage.PHP;
            
            // Swift patterns
            if (Regex.IsMatch(content, @"\bfunc\s+\w+\s*\(|\bvar\s+\w+\s*:|\blet\s+\w+\s*:|\bimport\s+Foundation"))
                return DetectedLanguage.Swift;
            
            // Kotlin patterns
            if (Regex.IsMatch(content, @"\bfun\s+\w+\s*\(|\bval\s+\w+\s*:|\bvar\s+\w+\s*:"))
                return DetectedLanguage.Kotlin;

            // NSL patterns (check last as fallback)
            if (Regex.IsMatch(content, @"\|>|~>|=>>"))
                return DetectedLanguage.NSL;

            return DetectedLanguage.Unknown;
        }

        /// <summary>
        /// Analyze and fix based on detected language
        /// </summary>
        public void Analyze()
        {
            switch (_detectedLanguage)
            {
                case DetectedLanguage.NSL:
                    var nslFixer = new NSLAutoFix(_source);
                    nslFixer.Analyze();
                    _fixes.AddRange(nslFixer.Fixes);
                    break;
                case DetectedLanguage.CSharp:
                    AnalyzeCSharp();
                    break;
                case DetectedLanguage.Java:
                    AnalyzeJava();
                    break;
                case DetectedLanguage.Python:
                    AnalyzePython();
                    break;
                case DetectedLanguage.JavaScript:
                case DetectedLanguage.TypeScript:
                    AnalyzeJavaScript();
                    break;
                case DetectedLanguage.Go:
                    AnalyzeGo();
                    break;
                case DetectedLanguage.Rust:
                    AnalyzeRust();
                    break;
                case DetectedLanguage.Ruby:
                    AnalyzeRuby();
                    break;
                case DetectedLanguage.PHP:
                    AnalyzePHP();
                    break;
                case DetectedLanguage.Swift:
                    AnalyzeSwift();
                    break;
                case DetectedLanguage.Kotlin:
                    AnalyzeKotlin();
                    break;
                case DetectedLanguage.Cpp:
                case DetectedLanguage.C:
                    AnalyzeCpp();
                    break;
                default:
                    // Try common fixes for all languages
                    AnalyzeCommon();
                    break;
            }
        }

        #region C# Analysis
        private void AnalyzeCSharp()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                
                // CS1591: Missing XML comment for publicly visible type
                if (Regex.IsMatch(line, @"^\s*public\s+(class|struct|interface|enum|void|static|async|string|int|bool|double|float|object)") &&
                    (i == 0 || !_lines[i-1].TrimStart().StartsWith("///")))
                {
                    var match = Regex.Match(line, @"public\s+(?:static\s+)?(?:async\s+)?(?:class|struct|interface|enum|void|string|int|bool|double|float|object|\w+)\s+(\w+)");
                    if (match.Success)
                    {
                        var name = match.Groups[1].Value;
                        var indent = line.Length - line.TrimStart().Length;
                        _fixes.Add(new SourceFix(lineNum, 1, 0, "", new string(' ', indent) + $"/// <summary>{name}</summary>\n", "Add XML documentation comment (CS1591)", FixCategory.Warning));
                    }
                }
                
                // CS8600/CS8602: Null reference warnings - suggest null checks
                if (Regex.IsMatch(line, @"\w+\s*=\s*\w+\.\w+") && !line.Contains("??") && !line.Contains("?."))
                {
                    var match = Regex.Match(line, @"(\w+)\.(\w+)");
                    if (match.Success)
                    {
                        var obj = match.Groups[1].Value;
                        var col = match.Index + 1;
                        _fixes.Add(new SourceFix(lineNum, col, match.Length, match.Value, $"{obj}?.{match.Groups[2].Value}", "Use null-conditional operator to avoid CS8602", FixCategory.Suggestion));
                    }
                }
                
                // Missing semicolon at end of statement
                var trimmed = line.TrimEnd();
                if (trimmed.Length > 0 && !trimmed.EndsWith(";") && !trimmed.EndsWith("{") && !trimmed.EndsWith("}") &&
                    !trimmed.EndsWith(",") && !trimmed.EndsWith("(") && !trimmed.EndsWith(")") &&
                    !trimmed.TrimStart().StartsWith("//") && !trimmed.TrimStart().StartsWith("/*") &&
                    !trimmed.TrimStart().StartsWith("*") && !trimmed.TrimStart().StartsWith("#") &&
                    !trimmed.TrimStart().StartsWith("[") && !trimmed.EndsWith("]") &&
                    Regex.IsMatch(trimmed, @"\b(return|var|int|string|bool|double|float|public|private|protected)\b"))
                {
                    if (!trimmed.EndsWith("=>") && !Regex.IsMatch(trimmed, @"^\s*(if|else|for|foreach|while|switch|case|try|catch|finally|using|lock)\b"))
                    {
                        _fixes.Add(new SourceFix(lineNum, trimmed.Length + 1, 0, "", ";", "Add missing semicolon", FixCategory.Error));
                    }
                }
                
                // Common typos in C#
                FixCSharpTypos(line, lineNum);
            }
        }
        
        private void FixCSharpTypos(string line, int lineNum)
        {
            var typos = new Dictionary<string, string>
            {
                { "pubic", "public" }, { "privat", "private" }, { "protcted", "protected" },
                { "statc", "static" }, { "viod", "void" }, { "retrun", "return" },
                { "stirng", "string" }, { "strng", "string" }, { "integr", "int" },
                { "boolen", "bool" }, { "boolean", "bool" }, { "flase", "false" },
                { "ture", "true" }, { "nul", "null" }, { "nulll", "null" },
                { "clss", "class" }, { "calss", "class" }, { "interfce", "interface" },
                { "namepsace", "namespace" }, { "namesapce", "namespace" },
                { "usign", "using" }, { "uing", "using" },
                { "cosnt", "const" }, { "conts", "const" },
                { "overide", "override" }, { "overrride", "override" },
                { "virtaul", "virtual" }, { "abstrat", "abstract" },
                { "asynch", "async" }, { "awiat", "await" }, { "awayt", "await" },
                { "Logg", "Log" }, { "Loger", "Logger" },
                { "Aborrt", "Abort" }, { "Cancle", "Cancel" },
                { "Looop", "Loop" }, { "Whlie", "While" },
                { "Console.Writeline", "Console.WriteLine" },
                { "Console.Writline", "Console.WriteLine" },
                { "Console.wirteLine", "Console.WriteLine" },
                { "Console.ReadLIne", "Console.ReadLine" },
                { "Abortt", "Abort" }, { "Cancell", "Cancel" },
                { "strign", "string" }, { "sting", "string" },
                { "Strng", "String" }, { "Integr", "Integer" },
                { "Systm", "System" }, { "Sytem", "System" },
                { "colection", "collection" }, { "colleciton", "collection" },
                { "dictionry", "dictionary" }, { "dictinary", "dictionary" },
                { "excepton", "exception" }, { "excption", "exception" }
            };
            
            foreach (var (typo, fix) in typos)
            {
                var idx = line.IndexOf(typo, StringComparison.Ordinal);
                if (idx >= 0)
                {
                    _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                }
            }
        }
        #endregion

        #region Java Analysis
        private void AnalyzeJava()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                
                // Missing semicolon
                var trimmed = line.TrimEnd();
                if (trimmed.Length > 0 && !trimmed.EndsWith(";") && !trimmed.EndsWith("{") && !trimmed.EndsWith("}") &&
                    !trimmed.TrimStart().StartsWith("//") && !trimmed.TrimStart().StartsWith("/*") &&
                    !trimmed.TrimStart().StartsWith("*") && !trimmed.TrimStart().StartsWith("@") &&
                    Regex.IsMatch(trimmed, @"\b(return|int|String|boolean|double|float|public|private|protected|import|package)\b") &&
                    !Regex.IsMatch(trimmed, @"^\s*(if|else|for|while|switch|case|try|catch|finally|class|interface)\b"))
                {
                    _fixes.Add(new SourceFix(lineNum, trimmed.Length + 1, 0, "", ";", "Add missing semicolon", FixCategory.Error));
                }
                
                // Java typos
                var typos = new Dictionary<string, string>
                {
                    { "pubic", "public" }, { "privat", "private" }, { "Stirng", "String" },
                    { "stirng", "string" }, { "Strng", "String" }, { "boolen", "boolean" },
                    { "retrun", "return" }, { "viod", "void" }, { "statc", "static" },
                    { "finaly", "finally" }, { "catach", "catch" }, { "trow", "throw" },
                    { "thorw", "throw" }, { "thorws", "throws" }, { "improt", "import" },
                    { "packge", "package" }, { "extens", "extends" }, { "implments", "implements" },
                    { "System.out.printl", "System.out.println" },
                    { "System.out.prtln", "System.out.println" },
                    { "ArrayLsit", "ArrayList" }, { "HashMpa", "HashMap" },
                    { "Interger", "Integer" }, { "Charcter", "Character" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region Python Analysis
        private void AnalyzePython()
        {
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                var trimmed = line.TrimEnd();
                
                // Missing colon after control statements
                if (Regex.IsMatch(trimmed, @"^\s*(if|elif|else|for|while|def|class|try|except|finally|with|async\s+def|async\s+for|async\s+with)\b") &&
                    !trimmed.EndsWith(":") && !trimmed.EndsWith("\\"))
                {
                    _fixes.Add(new SourceFix(lineNum, trimmed.Length + 1, 0, "", ":", "Add missing colon after statement", FixCategory.Error));
                }
                
                // IndentationError: Check for mixed tabs/spaces
                if (line.StartsWith("\t") && line.Contains(" ") && Regex.IsMatch(line, @"^[\t ]+"))
                {
                    var match = Regex.Match(line, @"^([\t ]+)");
                    if (match.Success && match.Value.Contains("\t") && match.Value.Contains(" "))
                    {
                        var spaces = match.Value.Replace("\t", "    ");
                        _fixes.Add(new SourceFix(lineNum, 1, match.Length, match.Value, spaces, "Replace mixed tabs/spaces with spaces (IndentationError)", FixCategory.Error));
                    }
                }
                
                // print statement without parentheses (Python 2 vs 3)
                var printMatch = Regex.Match(line, @"\bprint\s+([^(].*?)$");
                if (printMatch.Success && !printMatch.Groups[1].Value.StartsWith("("))
                {
                    var content = printMatch.Groups[1].Value.TrimEnd();
                    _fixes.Add(new SourceFix(lineNum, printMatch.Index + 1, printMatch.Length, printMatch.Value, $"print({content})", "Use print() function (Python 3 syntax)", FixCategory.Error));
                }
                
                // Common Python typos
                var typos = new Dictionary<string, string>
                {
                    { "pritn", "print" }, { "pirnt", "print" }, { "prnit", "print" },
                    { "retrun", "return" }, { "reutrn", "return" }, { "retrn", "return" },
                    { "def ", "def " }, { "dfe ", "def " }, { "deff ", "def " },
                    { "imoprt", "import" }, { "improt", "import" }, { "imoport", "import" },
                    { "foriegn", "foreign" }, { "fucntion", "function" },
                    { "flase", "False" }, { "fasle", "False" }, { "Flase", "False" },
                    { "ture", "True" }, { "Ture", "True" }, { "treu", "True" },
                    { "Non", "None" }, { "NONe", "None" }, { "NON", "None" },
                    { "esle", "else" }, { "elseif", "elif" }, { "else if", "elif" },
                    { "elfi", "elif" }, { "eilf", "elif" },
                    { "whlie", "while" }, { "whlile", "while" },
                    { "contniue", "continue" }, { "contine", "continue" },
                    { "berak", "break" }, { "braek", "break" },
                    { "sleef", "self" }, { "slef", "self" }, { "sefl", "self" },
                    { "__inti__", "__init__" }, { "__iinit__", "__init__" },
                    { "__strr__", "__str__" }, { "__repr_()", "__repr__()" },
                    { "langth", "length" }, { "lenght", "length" },
                    { "apend", "append" }, { "appned", "append" },
                    { "excpet", "except" }, { "exept", "except" },
                    { "finaly", "finally" }, { "fianlly", "finally" },
                    { "lamda", "lambda" }, { "labmda", "lambda" },
                    { "yeild", "yield" }, { "yiled", "yield" },
                    { "glboal", "global" }, { "gloabl", "global" },
                    { "assrt", "assert" }, { "asert", "assert" },
                    { "raies", "raise" }, { "rasie", "raise" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
                
                // == vs = in conditionals
                var assignInCondition = Regex.Match(line, @"\b(if|elif|while)\s+.*[^=!<>]=[^=].*:");
                if (assignInCondition.Success && !line.Contains(":="))
                {
                    _fixes.Add(new SourceFix(lineNum, assignInCondition.Index + 1, 0, "", "", "Possible assignment (=) instead of comparison (==) in condition", FixCategory.Warning));
                }
            }
        }
        #endregion

        #region JavaScript/TypeScript Analysis
        private void AnalyzeJavaScript()
        {
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                var trimmed = line.TrimEnd();
                
                // Missing semicolon (optional but common style)
                if (trimmed.Length > 0 && !trimmed.EndsWith(";") && !trimmed.EndsWith("{") && !trimmed.EndsWith("}") &&
                    !trimmed.EndsWith(",") && !trimmed.EndsWith("(") && !trimmed.TrimStart().StartsWith("//") &&
                    !trimmed.TrimStart().StartsWith("/*") && !trimmed.TrimStart().StartsWith("*") &&
                    Regex.IsMatch(trimmed, @"\b(const|let|var|return|import|export)\b") &&
                    !trimmed.EndsWith("=>") && !Regex.IsMatch(trimmed, @"^\s*(if|else|for|while|switch|case|try|catch|finally|function|class)\b"))
                {
                    _fixes.Add(new SourceFix(lineNum, trimmed.Length + 1, 0, "", ";", "Add missing semicolon", FixCategory.Style));
                }
                
                // == vs === (strict equality)
                var looseEquality = Regex.Match(line, @"[^=!]==[^=]");
                if (looseEquality.Success)
                {
                    _fixes.Add(new SourceFix(lineNum, looseEquality.Index + 2, 2, "==", "===", "Use strict equality (===) instead of loose equality (==)", FixCategory.Warning));
                }
                
                // != vs !== 
                var looseInequality = Regex.Match(line, @"!=[^=]");
                if (looseInequality.Success)
                {
                    _fixes.Add(new SourceFix(lineNum, looseInequality.Index + 1, 2, "!=", "!==", "Use strict inequality (!==) instead of loose inequality (!=)", FixCategory.Warning));
                }
                
                // var vs let/const
                var varUsage = Regex.Match(line, @"\bvar\s+(\w+)");
                if (varUsage.Success)
                {
                    _fixes.Add(new SourceFix(lineNum, varUsage.Index + 1, 3, "var", "let", "Use 'let' or 'const' instead of 'var'", FixCategory.Suggestion));
                }
                
                // Common JS typos
                var typos = new Dictionary<string, string>
                {
                    { "fucntion", "function" }, { "funcion", "function" }, { "funciton", "function" },
                    { "cosnt", "const" }, { "conts", "const" }, { "ocnst", "const" },
                    { "lte", "let" }, { "leet", "let" },
                    { "retrun", "return" }, { "reutrn", "return" }, { "retrn", "return" },
                    { "undefiend", "undefined" }, { "undifined", "undefined" },
                    { "NaN", "NaN" }, { "Nan", "NaN" },
                    { "treu", "true" }, { "ture", "true" }, { "flase", "false" },
                    { "nulll", "null" }, { "nul", "null" },
                    { "console.lgo", "console.log" }, { "console.lo", "console.log" },
                    { "consoel.log", "console.log" }, { "consloe.log", "console.log" },
                    { "docuemnt", "document" }, { "documnet", "document" },
                    { "widnow", "window" }, { "windwo", "window" },
                    { "addEventListner", "addEventListener" },
                    { "getElementByID", "getElementById" },
                    { "querySelecotr", "querySelector" },
                    { "innterHTML", "innerHTML" }, { "innerHtml", "innerHTML" },
                    { "settimeout", "setTimeout" }, { "setTimeOut", "setTimeout" },
                    { "setinterval", "setInterval" }, { "setInterVal", "setInterval" },
                    { "parserInt", "parseInt" }, { "parsefloat", "parseFloat" },
                    { "Json.parse", "JSON.parse" }, { "JSON.Parse", "JSON.parse" },
                    { "Json.stringify", "JSON.stringify" },
                    { "Ojbect", "Object" }, { "Arary", "Array" },
                    { "Stirng", "String" }, { "Nubmer", "Number" },
                    { "porperty", "property" }, { "propety", "property" },
                    { "lenght", "length" }, { "legnth", "length" },
                    { "spilce", "splice" }, { "splcie", "splice" },
                    { "concact", "concat" }, { "concta", "concat" },
                    { "inclues", "includes" }, { "inlcudes", "includes" },
                    { "fitler", "filter" }, { "fliter", "filter" },
                    { "redcue", "reduce" }, { "rduce", "reduce" },
                    { "forEahc", "forEach" }, { "foreahc", "forEach" },
                    { "aysnc", "async" }, { "asyc", "async" },
                    { "awiat", "await" }, { "awayt", "await" },
                    { "Pormsie", "Promise" }, { "Proimse", "Promise" },
                    { "improt", "import" }, { "exoprt", "export" },
                    { "exprot", "export" }, { "deafult", "default" },
                    { "requrie", "require" }, { "rquire", "require" },
                    { "moduel", "module" }, { "exprots", "exports" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region Go Analysis
        private void AnalyzeGo()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                
                // Unused variable (Go is strict about this)
                var varDecl = Regex.Match(line, @"(\w+)\s*:=\s*");
                if (varDecl.Success)
                {
                    var varName = varDecl.Groups[1].Value;
                    if (varName == "_") continue;
                    
                    // Check if variable is used later (simple check)
                    bool used = false;
                    for (int j = i + 1; j < _lines.Length && j < i + 50; j++)
                    {
                        if (Regex.IsMatch(_lines[j], $@"\b{Regex.Escape(varName)}\b"))
                        {
                            used = true;
                            break;
                        }
                    }
                    if (!used)
                    {
                        _fixes.Add(new SourceFix(lineNum, varDecl.Index + 1, varName.Length, varName, "_", $"Variable '{varName}' declared but not used (use _ to ignore)", FixCategory.Warning));
                    }
                }
                
                // Common Go typos
                var typos = new Dictionary<string, string>
                {
                    { "fucn", "func" }, { "fnuc", "func" }, { "funct", "func" },
                    { "packge", "package" }, { "pacakge", "package" },
                    { "improt", "import" }, { "imoport", "import" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "stirng", "string" }, { "strng", "string" },
                    { "interger", "int" }, { "integr", "int" },
                    { "flase", "false" }, { "ture", "true" },
                    { "nill", "nil" }, { "nul", "nil" },
                    { "fmt.Pritnln", "fmt.Println" }, { "fmt.Prinltn", "fmt.Println" },
                    { "fmt.Printl", "fmt.Println" }, { "fmt.Printf", "fmt.Printf" },
                    { "fmt.Spritnf", "fmt.Sprintf" }, { "fmt.Errof", "fmt.Errorf" },
                    { "strconv.Atoi", "strconv.Atoi" }, { "strconv.Itao", "strconv.Itoa" },
                    { "errer", "error" }, { "erorr", "error" },
                    { "sturct", "struct" }, { "strcut", "struct" },
                    { "interfce", "interface" }, { "inteface", "interface" },
                    { "chanell", "channel" }, { "chanel", "channel" },
                    { "goroutien", "goroutine" }, { "gorutine", "goroutine" },
                    { "defre", "defer" }, { "deferr", "defer" },
                    { "slcie", "slice" }, { "sclice", "slice" },
                    { "apend", "append" }, { "appned", "append" },
                    { "lenght", "length" }, { "lnegth", "length" },
                    { "contex", "context" }, { "contxt", "context" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region Rust Analysis
        private void AnalyzeRust()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                var trimmed = line.TrimEnd();
                
                // Missing semicolon
                if (trimmed.Length > 0 && !trimmed.EndsWith(";") && !trimmed.EndsWith("{") && !trimmed.EndsWith("}") &&
                    !trimmed.EndsWith(",") && !trimmed.TrimStart().StartsWith("//") &&
                    !trimmed.TrimStart().StartsWith("/*") && !trimmed.TrimStart().StartsWith("*") &&
                    !trimmed.TrimStart().StartsWith("#") &&
                    Regex.IsMatch(trimmed, @"\b(let|return|use|pub|mod)\b") &&
                    !Regex.IsMatch(trimmed, @"^\s*(fn|if|else|for|while|loop|match|impl|struct|enum|trait)\b"))
                {
                    _fixes.Add(new SourceFix(lineNum, trimmed.Length + 1, 0, "", ";", "Add missing semicolon", FixCategory.Error));
                }
                
                // Common Rust typos
                var typos = new Dictionary<string, string>
                {
                    { "fucn", "fn" }, { "fnn", "fn" },
                    { "lte", "let" }, { "leet", "let" },
                    { "mtu", "mut" }, { "mutt", "mut" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "stirng", "String" }, { "Strng", "String" },
                    { "flase", "false" }, { "ture", "true" },
                    { "Soem", "Some" }, { "Noone", "None" },
                    { "Reuslt", "Result" }, { "Resutl", "Result" },
                    { "Optoin", "Option" }, { "Optiom", "Option" },
                    { "Vecor", "Vector" }, { "Vect", "Vec" },
                    { "pritnln", "println" }, { "prinltn", "println" },
                    { "epritnln", "eprintln" }, { "eprnitln", "eprintln" },
                    { "foramt", "format" }, { "fromat", "format" },
                    { "impel", "impl" }, { "imple", "impl" },
                    { "strcut", "struct" }, { "sturct", "struct" },
                    { "enumm", "enum" }, { "enmu", "enum" },
                    { "trati", "trait" }, { "triait", "trait" },
                    { "modd", "mod" }, { "mdule", "mod" },
                    { "pubb", "pub" }, { "pubilc", "pub" },
                    { "crat", "crate" }, { "crete", "crate" },
                    { "matc", "match" }, { "mtach", "match" },
                    { "looop", "loop" }, { "lopop", "loop" },
                    { "whlie", "while" }, { "whlile", "while" },
                    { "unswrap", "unwrap" }, { "unwrpa", "unwrap" },
                    { "expec", "expect" }, { "expcet", "expect" },
                    { "cloen", "clone" }, { "colne", "clone" },
                    { "borrwo", "borrow" }, { "borwro", "borrow" },
                    { "lifetiem", "lifetime" }, { "liftime", "lifetime" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region Ruby Analysis
        private void AnalyzeRuby()
        {
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                
                // Missing 'end' keyword detection (basic)
                // Check for unmatched def/class/module/if/unless/while/until/case/begin
                
                // Common Ruby typos
                var typos = new Dictionary<string, string>
                {
                    { "dfe", "def" }, { "deef", "def" },
                    { "calss", "class" }, { "clss", "class" },
                    { "modle", "module" }, { "moduel", "module" },
                    { "reuqire", "require" }, { "reqiure", "require" },
                    { "incldue", "include" }, { "inlcude", "include" },
                    { "extnend", "extend" }, { "extedn", "extend" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "flase", "false" }, { "ture", "true" },
                    { "nill", "nil" }, { "niil", "nil" },
                    { "attr_accesor", "attr_accessor" }, { "attr_accesro", "attr_accessor" },
                    { "attr_reade", "attr_reader" }, { "attr_readr", "attr_reader" },
                    { "attr_wirter", "attr_writer" }, { "attr_writter", "attr_writer" },
                    { "initalize", "initialize" }, { "intialize", "initialize" },
                    { "pust", "puts" }, { "ptus", "puts" },
                    { "pirnt", "print" }, { "pritn", "print" },
                    { "endd", "end" }, { "ned", "end" },
                    { "esle", "else" }, { "elseif", "elsif" }, { "else if", "elsif" },
                    { "elsfi", "elsif" }, { "elsefi", "elsif" },
                    { "unlses", "unless" }, { "unles", "unless" },
                    { "utnil", "until" }, { "unitl", "until" },
                    { "whlie", "while" }, { "whlile", "while" },
                    { "yeild", "yield" }, { "yiled", "yield" },
                    { "lamda", "lambda" }, { "labmda", "lambda" },
                    { "proc", "proc" }, { "Proc", "Proc" },
                    { "blcok", "block" }, { "blokc", "block" },
                    { "hashe", "hash" }, { "hahs", "hash" },
                    { "arary", "array" }, { "arry", "array" },
                    { "symblo", "symbol" }, { "symobl", "symbol" },
                    { "stirng", "string" }, { "strng", "string" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region PHP Analysis
        private void AnalyzePHP()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                var trimmed = line.TrimEnd();
                
                // Missing semicolon
                if (trimmed.Length > 0 && !trimmed.EndsWith(";") && !trimmed.EndsWith("{") && !trimmed.EndsWith("}") &&
                    !trimmed.EndsWith(":") && !trimmed.EndsWith(",") &&
                    !trimmed.TrimStart().StartsWith("//") && !trimmed.TrimStart().StartsWith("/*") &&
                    !trimmed.TrimStart().StartsWith("*") && !trimmed.TrimStart().StartsWith("#") &&
                    !trimmed.TrimStart().StartsWith("<?") && !trimmed.TrimStart().StartsWith("?>") &&
                    Regex.IsMatch(trimmed, @"\$\w+\s*=|\breturn\b|\becho\b") &&
                    !Regex.IsMatch(trimmed, @"^\s*(if|else|elseif|for|foreach|while|switch|case|function|class|interface|trait)\b"))
                {
                    _fixes.Add(new SourceFix(lineNum, trimmed.Length + 1, 0, "", ";", "Add missing semicolon", FixCategory.Error));
                }
                
                // $ missing on variable
                var varWithout = Regex.Match(line, @"\b([a-z_][a-z0-9_]*)\s*=[^=]", RegexOptions.IgnoreCase);
                if (varWithout.Success && !line.Contains("$" + varWithout.Groups[1].Value) &&
                    !Regex.IsMatch(varWithout.Groups[1].Value, @"^(true|false|null|function|class|const|define)$", RegexOptions.IgnoreCase))
                {
                    // Could be missing $ - suggest
                    var varName = varWithout.Groups[1].Value;
                    if (!Regex.IsMatch(varName, @"^[A-Z_]+$")) // Not a constant
                    {
                        _fixes.Add(new SourceFix(lineNum, varWithout.Index + 1, varName.Length, varName, "$" + varName, "Add $ prefix to variable", FixCategory.Suggestion));
                    }
                }
                
                // Common PHP typos
                var typos = new Dictionary<string, string>
                {
                    { "fucntion", "function" }, { "funcion", "function" },
                    { "ecoh", "echo" }, { "ehco", "echo" },
                    { "pritn", "print" }, { "pirnt", "print" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "flase", "false" }, { "ture", "true" },
                    { "nul", "null" }, { "NULL", "null" },
                    { "pubic", "public" }, { "privat", "private" },
                    { "protcted", "protected" }, { "statc", "static" },
                    { "clss", "class" }, { "calss", "class" },
                    { "interfce", "interface" }, { "inteface", "interface" },
                    { "trati", "trait" }, { "triait", "trait" },
                    { "namesapce", "namespace" }, { "namepsace", "namespace" },
                    { "reqiure", "require" }, { "reuqire", "require" },
                    { "incldue", "include" }, { "inlcude", "include" },
                    { "reqiure_once", "require_once" }, { "include_onec", "include_once" },
                    { "arary", "array" }, { "arry", "array" },
                    { "forach", "foreach" }, { "forech", "foreach" },
                    { "whlie", "while" }, { "whlile", "while" },
                    { "esle", "else" }, { "elsefi", "elseif" },
                    { "swtich", "switch" }, { "swtch", "switch" },
                    { "caes", "case" }, { "csae", "case" },
                    { "defualt", "default" }, { "deafult", "default" },
                    { "breask", "break" }, { "braek", "break" },
                    { "contniue", "continue" }, { "contineu", "continue" },
                    { "thsi", "this" }, { "tihs", "this" },
                    { "slef", "self" }, { "sefl", "self" },
                    { "parnet", "parent" }, { "praent", "parent" },
                    { "extneds", "extends" }, { "extedns", "extends" },
                    { "implments", "implements" }, { "impelments", "implements" },
                    { "absract", "abstract" }, { "abstrac", "abstract" },
                    { "finaly", "final" }, { "fianl", "final" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region Swift Analysis
        private void AnalyzeSwift()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                
                // Common Swift typos
                var typos = new Dictionary<string, string>
                {
                    { "fucn", "func" }, { "funct", "func" },
                    { "lte", "let" }, { "leet", "let" },
                    { "varr", "var" }, { "vra", "var" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "flase", "false" }, { "ture", "true" },
                    { "nill", "nil" }, { "niil", "nil" },
                    { "stirng", "String" }, { "Strng", "String" },
                    { "Interger", "Int" }, { "integr", "Int" },
                    { "Dobule", "Double" }, { "Duoble", "Double" },
                    { "Bololean", "Bool" }, { "boolen", "Bool" },
                    { "calss", "class" }, { "clss", "class" },
                    { "strcut", "struct" }, { "sturct", "struct" },
                    { "enumm", "enum" }, { "enmu", "enum" },
                    { "protocl", "protocol" }, { "protcol", "protocol" },
                    { "extenison", "extension" }, { "extensoin", "extension" },
                    { "improt", "import" }, { "imoport", "import" },
                    { "gaurd", "guard" }, { "gurad", "guard" },
                    { "swithc", "switch" }, { "swtich", "switch" },
                    { "defualt", "default" }, { "deafult", "default" },
                    { "optoinal", "optional" }, { "optinal", "optional" },
                    { "unwrpa", "unwrap" }, { "unswrap", "unwrap" },
                    { "pirnt", "print" }, { "pritn", "print" },
                    { "overrride", "override" }, { "overide", "override" },
                    { "mutaing", "mutating" }, { "mutatign", "mutating" },
                    { "privat", "private" }, { "pubic", "public" },
                    { "internl", "internal" }, { "filepriavte", "fileprivate" },
                    { "inot", "init" }, { "iinit", "init" },
                    { "denit", "deinit" }, { "dienit", "deinit" },
                    { "selff", "self" }, { "slef", "self" },
                    { "Selff", "Self" }, { "Slef", "Self" },
                    { "supre", "super" }, { "supr", "super" },
                    { "thorw", "throw" }, { "trhow", "throw" },
                    { "thorws", "throws" }, { "trhows", "throws" },
                    { "rethrow", "rethrows" }, { "rethrow", "rethrows" },
                    { "tryu", "try" }, { "tyr", "try" },
                    { "ctach", "catch" }, { "caatch", "catch" },
                    { "asynv", "async" }, { "aysnc", "async" },
                    { "awiat", "await" }, { "awayt", "await" },
                    { "clsoure", "closure" }, { "closuer", "closure" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region Kotlin Analysis
        private void AnalyzeKotlin()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                
                // Common Kotlin typos
                var typos = new Dictionary<string, string>
                {
                    { "fucn", "fun" }, { "funn", "fun" },
                    { "vla", "val" }, { "vall", "val" },
                    { "varr", "var" }, { "vra", "var" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "flase", "false" }, { "ture", "true" },
                    { "nulll", "null" }, { "nul", "null" },
                    { "stirng", "String" }, { "Strng", "String" },
                    { "Interger", "Int" }, { "integr", "Int" },
                    { "calss", "class" }, { "clss", "class" },
                    { "ojbect", "object" }, { "obejct", "object" },
                    { "interfce", "interface" }, { "inteface", "interface" },
                    { "enumm", "enum" }, { "enmu", "enum" },
                    { "improt", "import" }, { "imoport", "import" },
                    { "packge", "package" }, { "pacakge", "package" },
                    { "pritnln", "println" }, { "prinltn", "println" },
                    { "overrride", "override" }, { "overide", "override" },
                    { "privat", "private" }, { "pubic", "public" },
                    { "internl", "internal" }, { "proected", "protected" },
                    { "opne", "open" }, { "abstarct", "abstract" },
                    { "finall", "final" }, { "selaed", "sealed" },
                    { "comapnion", "companion" }, { "compainon", "companion" },
                    { "susepnd", "suspend" }, { "suspned", "suspend" },
                    { "coroutien", "coroutine" }, { "corutine", "coroutine" },
                    { "lamdba", "lambda" }, { "lamda", "lambda" },
                    { "extnesion", "extension" }, { "extensoin", "extension" },
                    { "inlien", "inline" }, { "inlin", "inline" },
                    { "reifeid", "reified" }, { "reifed", "reified" },
                    { "latenit", "lateinit" }, { "lateinti", "lateinit" },
                    { "layz", "lazy" }, { "lazzy", "lazy" },
                    { "delgate", "delegate" }, { "delegat", "delegate" },
                    { "anntoation", "annotation" }, { "annoation", "annotation" },
                    { "dat", "data" }, { "dtaa", "data" },
                    { "whne", "when" }, { "wehn", "when" },
                    { "thorw", "throw" }, { "trhow", "throw" },
                    { "tryu", "try" }, { "tyr", "try" },
                    { "ctach", "catch" }, { "caatch", "catch" },
                    { "finaly", "finally" }, { "fianlly", "finally" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region C/C++ Analysis
        private void AnalyzeCpp()
        {
            AnalyzeCommon();
            
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                var trimmed = line.TrimEnd();
                
                // Missing semicolon
                if (trimmed.Length > 0 && !trimmed.EndsWith(";") && !trimmed.EndsWith("{") && !trimmed.EndsWith("}") &&
                    !trimmed.EndsWith(",") && !trimmed.EndsWith("\\") &&
                    !trimmed.TrimStart().StartsWith("//") && !trimmed.TrimStart().StartsWith("/*") &&
                    !trimmed.TrimStart().StartsWith("*") && !trimmed.TrimStart().StartsWith("#") &&
                    Regex.IsMatch(trimmed, @"\b(return|int|void|char|float|double|bool|auto|const)\b") &&
                    !Regex.IsMatch(trimmed, @"^\s*(if|else|for|while|switch|case|class|struct|namespace|template)\b"))
                {
                    _fixes.Add(new SourceFix(lineNum, trimmed.Length + 1, 0, "", ";", "Add missing semicolon", FixCategory.Error));
                }
                
                // Common C/C++ typos
                var typos = new Dictionary<string, string>
                {
                    { "viod", "void" }, { "vioid", "void" },
                    { "intt", "int" }, { "itn", "int" },
                    { "cahr", "char" }, { "charr", "char" },
                    { "flota", "float" }, { "flaot", "float" },
                    { "dobule", "double" }, { "duoble", "double" },
                    { "boool", "bool" }, { "boll", "bool" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "flase", "false" }, { "ture", "true" },
                    { "nulptr", "nullptr" }, { "NULKL", "NULL" },
                    { "pritnf", "printf" }, { "prinft", "printf" },
                    { "scanff", "scanf" }, { "scnaf", "scanf" },
                    { "stirng", "string" }, { "strng", "string" },
                    { "incldue", "include" }, { "inlcude", "include" },
                    { "defien", "define" }, { "dfeine", "define" },
                    { "strcut", "struct" }, { "sturct", "struct" },
                    { "calss", "class" }, { "clss", "class" },
                    { "pubic", "public" }, { "privat", "private" },
                    { "protcted", "protected" }, { "virtaul", "virtual" },
                    { "constt", "const" }, { "cosnt", "const" },
                    { "statc", "static" }, { "staitc", "static" },
                    { "extren", "extern" }, { "extenr", "extern" },
                    { "tyepdef", "typedef" }, { "typdef", "typedef" },
                    { "namespacee", "namespace" }, { "namepsace", "namespace" },
                    { "templat", "template" }, { "tempalte", "template" },
                    { "tyepname", "typename" }, { "typname", "typename" },
                    { "frined", "friend" }, { "freind", "friend" },
                    { "operaotr", "operator" }, { "oeprator", "operator" },
                    { "deleet", "delete" }, { "delet", "delete" },
                    { "maloc", "malloc" }, { "mallloc", "malloc" },
                    { "fre", "free" }, { "feee", "free" },
                    { "siezof", "sizeof" }, { "sizof", "sizeof" },
                    { "std::cotu", "std::cout" }, { "std::cou", "std::cout" },
                    { "std::cinn", "std::cin" }, { "std::ci", "std::cin" },
                    { "std::enld", "std::endl" }, { "std::end", "std::endl" },
                    { "std::vecto", "std::vector" }, { "std::vecor", "std::vector" },
                    { "std::strin", "std::string" }, { "std::strng", "std::string" },
                    { "std::ma", "std::map" }, { "std::mpa", "std::map" }
                };
                
                foreach (var (typo, fix) in typos)
                {
                    var idx = line.IndexOf(typo, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, typo, fix, $"Fix typo: {typo} -> {fix}", FixCategory.Error));
                    }
                }
            }
        }
        #endregion

        #region Common Analysis (All Languages)
        private void AnalyzeCommon()
        {
            for (int i = 0; i < _lines.Length; i++)
            {
                var line = _lines[i];
                var lineNum = i + 1;
                
                // Unclosed strings
                int singleQuotes = line.Count(c => c == '\'') - Regex.Matches(line, @"\\'").Count;
                int doubleQuotes = line.Count(c => c == '"') - Regex.Matches(line, @"\\""").Count;
                
                if (singleQuotes % 2 != 0)
                {
                    _fixes.Add(new SourceFix(lineNum, line.Length + 1, 0, "", "'", "Unclosed single quote string", FixCategory.Error));
                }
                if (doubleQuotes % 2 != 0)
                {
                    _fixes.Add(new SourceFix(lineNum, line.Length + 1, 0, "", "\"", "Unclosed double quote string", FixCategory.Error));
                }
                
                // Mismatched brackets
                int openParens = line.Count(c => c == '(');
                int closeParens = line.Count(c => c == ')');
                int openBrackets = line.Count(c => c == '[');
                int closeBrackets = line.Count(c => c == ']');
                int openBraces = line.Count(c => c == '{');
                int closeBraces = line.Count(c => c == '}');
                
                if (openParens > closeParens)
                {
                    _fixes.Add(new SourceFix(lineNum, line.Length + 1, 0, "", new string(')', openParens - closeParens), "Missing closing parenthesis", FixCategory.Error));
                }
                if (openBrackets > closeBrackets)
                {
                    _fixes.Add(new SourceFix(lineNum, line.Length + 1, 0, "", new string(']', openBrackets - closeBrackets), "Missing closing bracket", FixCategory.Error));
                }
                
                // Trailing whitespace (style)
                if (line.EndsWith(" ") || line.EndsWith("\t"))
                {
                    var trimmedLen = line.TrimEnd().Length;
                    _fixes.Add(new SourceFix(lineNum, trimmedLen + 1, line.Length - trimmedLen, line.Substring(trimmedLen), "", "Remove trailing whitespace", FixCategory.Style));
                }
                
                // TODO/FIXME/HACK detection
                if (Regex.IsMatch(line, @"\b(TODO|FIXME|HACK|XXX|BUG)\b", RegexOptions.IgnoreCase))
                {
                    var match = Regex.Match(line, @"\b(TODO|FIXME|HACK|XXX|BUG)\b", RegexOptions.IgnoreCase);
                    _fixes.Add(new SourceFix(lineNum, match.Index + 1, match.Length, match.Value, match.Value, $"Found {match.Value} comment - needs attention", FixCategory.Suggestion));
                }
                
                // Common universal typos
                var universalTypos = new Dictionary<string, string>
                {
                    { "teh", "the" }, { "taht", "that" }, { "wiht", "with" },
                    { "hte", "the" }, { "adn", "and" }, { "fo", "of" },
                    { "lenght", "length" }, { "widht", "width" }, { "heigth", "height" },
                    { "fucntion", "function" }, { "funciton", "function" },
                    { "retrun", "return" }, { "reutrn", "return" },
                    { "ture", "true" }, { "flase", "false" },
                    { "calback", "callback" }, { "callbakc", "callback" },
                    { "paramter", "parameter" }, { "parmeter", "parameter" },
                    { "arguemnt", "argument" }, { "arguent", "argument" },
                    { "varaible", "variable" }, { "varialbe", "variable" },
                    { "condtion", "condition" }, { "conditon", "condition" },
                    { "statment", "statement" }, { "statemnt", "statement" },
                    { "expresion", "expression" }, { "expressin", "expression" },
                    { "opertator", "operator" }, { "opertor", "operator" },
                    { "initalize", "initialize" }, { "intialize", "initialize" },
                    { "implment", "implement" }, { "impelment", "implement" },
                    { "excpetion", "exception" }, { "excepton", "exception" },
                    { "reponse", "response" }, { "respones", "response" },
                    { "reqeust", "request" }, { "requets", "request" },
                    { "messge", "message" }, { "mesage", "message" },
                    { "recieve", "receive" }, { "recive", "receive" },
                    { "seperate", "separate" }, { "seprate", "separate" },
                    { "occured", "occurred" }, { "occure", "occur" },
                    { "sucessful", "successful" }, { "succesful", "successful" },
                    { "neccessary", "necessary" }, { "neccesary", "necessary" },
                    { "defualt", "default" }, { "deafult", "default" },
                    { "availble", "available" }, { "avaiable", "available" },
                    { "accross", "across" }, { "acrross", "across" },
                    { "refernce", "reference" }, { "referece", "reference" },
                    { "defintion", "definition" }, { "defnition", "definition" },
                    { "descripion", "description" }, { "desciption", "description" }
                };
                
                foreach (var (typo, fix) in universalTypos)
                {
                    var idx = line.IndexOf(typo, StringComparison.OrdinalIgnoreCase);
                    if (idx >= 0)
                    {
                        var actual = line.Substring(idx, typo.Length);
                        var fixCase = char.IsUpper(actual[0]) ? char.ToUpper(fix[0]) + fix.Substring(1) : fix;
                        _fixes.Add(new SourceFix(lineNum, idx + 1, typo.Length, actual, fixCase, $"Fix typo: {actual} -> {fixCase}", FixCategory.Warning));
                    }
                }
            }
        }
        #endregion

        /// <summary>
        /// Apply all fixes to source and return new source
        /// </summary>
        public string ApplyFixes(FixCategory minCategory = FixCategory.Error)
        {
            if (_fixes.Count == 0)
                return _source;

            var fixesToApply = _fixes
                .Where(f => f.Category <= minCategory)
                .OrderByDescending(f => f.Line)
                .ThenByDescending(f => f.Column)
                .ToList();

            var lines = _source.Split('\n').ToList();

            foreach (var fix in fixesToApply)
            {
                var lineIdx = fix.Line - 1;
                if (lineIdx < 0 || lineIdx >= lines.Count)
                    continue;

                var line = lines[lineIdx];
                var col = fix.Column - 1;

                if (col < 0 || col > line.Length)
                    continue;

                var before = line.Substring(0, col);
                var after = col + fix.Length <= line.Length
                    ? line.Substring(col + fix.Length)
                    : "";

                lines[lineIdx] = before + fix.ReplacementText + after;
            }

            return string.Join('\n', lines);
        }

        /// <summary>
        /// Get all errors with context
        /// </summary>
        public string GetAllErrorsWithContext()
        {
            if (_fixes.Count == 0)
                return $"No issues found in {_detectedLanguage} code - looks good!";

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Detected language: {_detectedLanguage}");
            sb.AppendLine($"Found {_fixes.Count} issue(s):");
            sb.AppendLine();

            foreach (var fix in _fixes.OrderBy(f => f.Line).ThenBy(f => f.Column))
            {
                sb.AppendLine(GetErrorWithContext(fix));
                sb.AppendLine();
            }

            sb.AppendLine("Run with --fix to automatically apply fixes.");
            return sb.ToString();
        }

        private string GetErrorWithContext(SourceFix fix, int contextLines = 1)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"{fix.Category} at line {fix.Line}, column {fix.Column}:");
            sb.AppendLine();

            int startLine = Math.Max(0, fix.Line - 1 - contextLines);
            int endLine = Math.Min(_lines.Length - 1, fix.Line - 1 + contextLines);

            for (int i = startLine; i <= endLine; i++)
            {
                var lineNum = (i + 1).ToString().PadLeft(4);
                var marker = i == fix.Line - 1 ? " > " : "   ";
                sb.AppendLine($"{marker}{lineNum} | {_lines[i]}");

                if (i == fix.Line - 1)
                {
                    var padding = new string(' ', 4 + 3 + 3 + fix.Column - 1);
                    var pointer = fix.Length > 0 ? new string('^', fix.Length) : "^";
                    sb.AppendLine($"{padding}{pointer}");
                }
            }

            sb.AppendLine();
            sb.AppendLine($"  {fix.Description}");

            if (!string.IsNullOrEmpty(fix.ReplacementText) && fix.OriginalText != fix.ReplacementText)
            {
                if (fix.Length > 0)
                    sb.AppendLine($"  Suggested fix: {fix.OriginalText} -> {fix.ReplacementText}");
                else
                    sb.AppendLine($"  Suggested fix: Add '{fix.ReplacementText}'");
            }

            return sb.ToString();
        }
    }

#endregion