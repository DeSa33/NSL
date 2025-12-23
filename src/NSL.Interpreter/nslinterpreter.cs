using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using System.Buffers;
using NSL.Core.AST;
using NSL.Interpreter;
using NSL.Lexer;
using NSL.Parser;
using NSL.GPU;
using NSLTokenType = NSL.Core.Tokens.TokenType;

namespace NSL.Core
{
    /// <summary>
    /// NSL Interpreter - Executes NSL AST nodes
    /// </summary>
    public class NSLInterpreter : INSLVisitor<object?>
    {
        private readonly Dictionary<string, object?> _globals;
        private readonly Dictionary<string, object?> _locals;
        private readonly Stack<Dictionary<string, object?>> _scopes;
        private readonly Random _random;
        private readonly NSLConsciousnessEngine _consciousnessEngine;
        
        // Control flow state management
        private bool _shouldBreak = false;
        private bool _shouldContinue = false;
        private bool _shouldReturn = false;

        // Closure support - tracks the current function's captured scopes for nested closures
        private List<Dictionary<string, object?>>? _currentClosureScopes = null;
        
        // Function execution context tracking
        private bool _inFunctionContext = false;

        // Debug support
        private Func<int, string?, Dictionary<string, object?>, Dictionary<string, object?>, bool>? _debugCallback;
        private TextWriter _outputWriter = Console.Out;
        private string? _currentSourceFile;
        private int _currentLine = 1;

        // Module system support
        private readonly HashSet<string> _loadedModules = new();
        private readonly Dictionary<string, Dictionary<string, object?>> _moduleExports = new();
        private Dictionary<string, object?>? _currentModuleExports = null; // Active exports during module load

        // GPU acceleration support (lazy-initialized)
        private static GpuAutoConfig? _gpuConfig;
        private static GpuKernels? _gpuKernels;
        private static readonly object _gpuLock = new();

        // ===== PERFORMANCE: Compiled Regex Patterns =====
        private static readonly Regex EnumDeclRegex = new(@"^(public\s+|internal\s+|private\s+|protected\s+)*(enum\s+\w+)", RegexOptions.Compiled);
        private static readonly Regex InterfaceDeclRegex = new(@"^(public\s+|internal\s+|private\s+|protected\s+)*(partial\s+)?(interface\s+\w+)", RegexOptions.Compiled);
        private static readonly Regex DictInitRegex = new(@"(new\s+(Dictionary|List|HashSet|SortedDictionary)|new\s*\{|=\s*\{|=>\s*\{)", RegexOptions.Compiled);
        private static readonly Regex DictInitSimpleRegex = new(@"(new\s+Dictionary|new\s*\{|=\s*\{)", RegexOptions.Compiled);
        private static readonly Regex EnumMemberRegex = new(@"^[A-Z_][A-Za-z0-9_]*\s*(=\s*[^,()\[\]]+)?\s*,?\s*(//.*)?$", RegexOptions.Compiled);
        private static readonly Regex InterfaceMemberRegex = new(@"^(event\s+)?[\w<>\[\],\s\?]+\s+\w+\s*[\(\{;]", RegexOptions.Compiled);
        private static readonly Regex PublicDeclRegex = new(@"^\s*public\s+", RegexOptions.Compiled);

        // ===== PERFORMANCE: AST Cache =====
        private static readonly Dictionary<long, NSL.Core.AST.NSLASTNode> _astCache = new();
        private static readonly object _astCacheLock = new();
        private const int MaxAstCacheSize = 100;
        private const int MaxAstNodeSize = 10000; // Max AST nodes per cached entry
        private static int _astCacheHits = 0;
        private static int _astCacheMisses = 0;
        private static int _astCacheEvictions = 0;
        private static bool _astCacheTracing = false;

        // Sandbox mode - restricts dangerous operations
        private bool _sandboxMode = false;
        private readonly string _sandboxTempDir = Path.Combine(Path.GetTempPath(), "nsl_sandbox");
        
        /// <summary>
        /// Enable or disable sandbox mode. Restricts file ops to temp, disables network/process spawn.
        /// </summary>
        public void SetSandboxMode(bool enabled)
        {
            _sandboxMode = enabled;
            if (enabled && !Directory.Exists(_sandboxTempDir))
                Directory.CreateDirectory(_sandboxTempDir);
        }
        
        private void EnsureSandboxAllowed(string path, string operation)
        {
            if (!_sandboxMode) return;
            var fullPath = Path.GetFullPath(path);
            if (!fullPath.StartsWith(Path.GetFullPath(_sandboxTempDir), StringComparison.OrdinalIgnoreCase))
                throw new NSLRuntimeException($"Sandbox: {operation} blocked outside temp dir");
        }
        
        /// <summary>
        /// Initializes a new instance of the NSL interpreter with built-in functions and global scope.
        /// </summary>
        public NSLInterpreter()
        {
            _globals = new Dictionary<string, object?>();
            _locals = new Dictionary<string, object?>();
            _scopes = new Stack<Dictionary<string, object?>>();
            _random = new Random();
            _consciousnessEngine = new NSLConsciousnessEngine();

            InitializeBuiltins();
        }

        /// <summary>
        /// Set a debug callback that is called before each statement execution.
        /// The callback receives (line, file, locals, globals) and returns true to continue or false to stop.
        /// </summary>
        public void SetDebugCallback(Func<int, string?, Dictionary<string, object?>, Dictionary<string, object?>, bool> callback)
        {
            _debugCallback = callback;
        }

        /// <summary>
        /// Set a custom output writer for print statements (used by debugger).
        /// </summary>
        public void SetOutputWriter(TextWriter writer)
        {
            _outputWriter = writer;
        }

        /// <summary>
        /// Evaluate an expression string and return the result.
        /// Uses AST caching for repeated expressions.
        /// </summary>
        public object? EvaluateExpression(string expression)
        {
            try
            {
                var ast = GetOrParseExpression(expression);
                return ast.Accept(this);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Get cached AST or parse and cache the expression.
        /// Context-aware caching with eviction tracking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private NSL.Core.AST.NSLASTNode GetOrParseExpression(string expression)
        {
            // Context-aware cache key: combines expression hash with sandbox mode flag
            long cacheKey = ((long)expression.GetHashCode() << 1) | (_sandboxMode ? 1L : 0L);
            
            lock (_astCacheLock)
            {
                if (_astCache.TryGetValue(cacheKey, out var cached))
                {
                    _astCacheHits++;
                    return cached;
                }
                _astCacheMisses++;
            }
            
            var lexer = new NSLLexer(expression);
            var tokens = lexer.Tokenize().ToList();
            var parser = new NSLParser();
            parser.SetSource(expression);
            var ast = parser.ParseExpression(tokens);
            
            lock (_astCacheLock)
            {
                if (_astCache.Count >= MaxAstCacheSize)
                {
                    _astCacheEvictions += _astCache.Count;
                    if (_astCacheTracing)
                        Console.Error.WriteLine($"[AST Cache] Evicting {_astCache.Count} entries (hits: {_astCacheHits}, misses: {_astCacheMisses})");
                    _astCache.Clear();
                }
                _astCache[cacheKey] = ast;
            }
            
            return ast;
        }
        
        /// <summary>
        /// Get AST cache statistics for diagnostics.
        /// </summary>
        public static Dictionary<string, object?> GetAstCacheStats()
        {
            lock (_astCacheLock)
            {
                return new Dictionary<string, object?> {
                    ["size"] = (double)_astCache.Count,
                    ["maxSize"] = (double)MaxAstCacheSize,
                    ["hits"] = (double)_astCacheHits,
                    ["misses"] = (double)_astCacheMisses,
                    ["evictions"] = (double)_astCacheEvictions,
                    ["hitRate"] = _astCacheHits + _astCacheMisses > 0 
                        ? Math.Round((double)_astCacheHits / (_astCacheHits + _astCacheMisses) * 100, 2) 
                        : 0.0,
                    ["tracing"] = _astCacheTracing
                };
            }
        }
        
        /// <summary>
        /// Enable/disable AST cache tracing.
        /// </summary>
        public static void SetAstCacheTracing(bool enabled) => _astCacheTracing = enabled;

        /// <summary>
        /// Get current local variables for debugging.
        /// </summary>
        public Dictionary<string, object?> GetCurrentLocals()
        {
            var locals = new Dictionary<string, object?>();
            foreach (var scope in _scopes)
            {
                foreach (var kvp in scope)
                {
                    if (!locals.ContainsKey(kvp.Key))
                        locals[kvp.Key] = kvp.Value;
                }
            }
            foreach (var kvp in _locals)
            {
                if (!locals.ContainsKey(kvp.Key))
                    locals[kvp.Key] = kvp.Value;
            }
            return locals;
        }

        /// <summary>
        /// Get current global variables for debugging.
        /// </summary>
        public Dictionary<string, object?> GetCurrentGlobals()
        {
            return new Dictionary<string, object?>(_globals);
        }

        /// <summary>
        /// Call debug callback if set. Returns true to continue execution.
        /// </summary>
        private bool OnDebugStatement(NSLASTNode node)
        {
            if (_debugCallback == null) return true;

            _currentLine = node.Line;
            return _debugCallback(_currentLine, _currentSourceFile, GetCurrentLocals(), GetCurrentGlobals());
        }

        /// <summary>
        /// Initialize built-in functions
        /// </summary>
        private void InitializeBuiltins()
        {
            _globals["print"] = new NSLBuiltinFunction("print", (args) =>
            {
                if (args.Length == 0)
                {
                    _outputWriter.WriteLine();
                    return null;
                }

                var output = string.Join(" ", args.Select(arg => ConvertToString(arg)));
                _outputWriter.WriteLine(output);
                return null;
            });

            _globals["input"] = new NSLBuiltinFunction("input", (args) =>
            {
                if (args.Length > 0)
                {
                    Console.Write(ConvertToString(args[0]));
                }
                return Console.ReadLine() ?? "";
            });

            _globals["type"] = new NSLBuiltinFunction("type", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("type() takes exactly one argument");
                
                return GetNSLTypeName(args[0]);
            });

            _globals["str"] = new NSLBuiltinFunction("str", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("str() takes exactly one argument");

                return ConvertToString(args[0]);
            });

            _globals["len"] = new NSLBuiltinFunction("len", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("len() takes exactly one argument");

                var obj = args[0];
                if (obj is string s) return (double)s.Length;
                if (obj is Array arr) return (double)arr.Length;
                if (obj is System.Collections.ICollection coll) return (double)coll.Count;
                throw new NSLRuntimeException($"Object of type '{GetNSLTypeName(obj)}' has no len()");
            });

            _globals["range"] = new NSLBuiltinFunction("range", (args) =>
            {
                if (args.Length < 1 || args.Length > 3)
                    throw new NSLRuntimeException("range() takes 1 to 3 arguments");

                // Convert arguments to integers (support all numeric types)
                var intArgs = new int[args.Length];
                for (int i = 0; i < args.Length; i++)
                {
                    intArgs[i] = args[i] switch
                    {
                        double d => (int)d,
                        int n => n,
                        long l => (int)l,
                        float f => (int)f,
                        _ => throw new NSLRuntimeException($"range() argument {i + 1} must be a number")
                    };
                }

                int start, end, step;
                
                if (args.Length == 1)
                {
                    // range(10) -> [0,1,2,3,4,5,6,7,8,9]
                    start = 0;
                    end = intArgs[0];
                    step = 1;
                }
                else if (args.Length == 2)
                {
                    // range(5, 10) -> [5,6,7,8,9]
                    start = intArgs[0];
                    end = intArgs[1];
                    step = 1;
                }
                else // args.Length == 3
                {
                    // range(0, 10, 2) -> [0,2,4,6,8]
                    start = intArgs[0];
                    end = intArgs[1];
                    step = intArgs[2];
                    
                    if (step == 0)
                        throw new NSLRuntimeException("range() step argument must not be zero");
                }

                var result = new List<object>();
                
                if (step > 0)
                {
                    for (int i = start; i < end; i += step)
                        result.Add((double)i);
                }
                else
                {
                    for (int i = start; i > end; i += step)
                        result.Add((double)i);
                }
                
                return result;
            });

            // Global enumerate function - returns [[0, item], [1, item], ...]
            _globals["enumerate"] = new NSLBuiltinFunction("enumerate", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("enumerate() requires a list or string argument");
                
                var result = new List<object?>();
                if (args[0] is IList<object?> list)
                {
                    for (int i = 0; i < list.Count; i++)
                        result.Add(new List<object?> { (double)i, list[i] });
                }
                else if (args[0] is string str)
                {
                    for (int i = 0; i < str.Length; i++)
                        result.Add(new List<object?> { (double)i, str[i].ToString() });
                }
                else
                {
                    throw new NSLRuntimeException("enumerate() requires a list or string");
                }
                return result;
            });

            // ===================================
            // Native Math Functions
            // ===================================

            // Mathematical constants
            _globals["PI"] = Math.PI;
            _globals["E"] = Math.E;

            // Square root
            _globals["sqrt"] = new NSLBuiltinFunction("sqrt", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("sqrt() takes exactly one argument");
                var n = ConvertToNumber(args[0]);
                if (n < 0)
                    throw new NSLRuntimeException("sqrt() argument must be non-negative");
                return Math.Sqrt(n);
            });

            // Trigonometric functions
            _globals["sin"] = new NSLBuiltinFunction("sin", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("sin() takes exactly one argument");
                return Math.Sin(ConvertToNumber(args[0]));
            });

            _globals["cos"] = new NSLBuiltinFunction("cos", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("cos() takes exactly one argument");
                return Math.Cos(ConvertToNumber(args[0]));
            });

            _globals["tan"] = new NSLBuiltinFunction("tan", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("tan() takes exactly one argument");
                return Math.Tan(ConvertToNumber(args[0]));
            });

            // Inverse trigonometric functions
            _globals["asin"] = new NSLBuiltinFunction("asin", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("asin() takes exactly one argument");
                return Math.Asin(ConvertToNumber(args[0]));
            });

            _globals["acos"] = new NSLBuiltinFunction("acos", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("acos() takes exactly one argument");
                return Math.Acos(ConvertToNumber(args[0]));
            });

            _globals["atan"] = new NSLBuiltinFunction("atan", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("atan() takes exactly one argument");
                return Math.Atan(ConvertToNumber(args[0]));
            });

            _globals["atan2"] = new NSLBuiltinFunction("atan2", (args) =>
            {
                if (args.Length != 2)
                    throw new NSLRuntimeException("atan2() takes exactly two arguments");
                return Math.Atan2(ConvertToNumber(args[0]), ConvertToNumber(args[1]));
            });

            // Absolute value
            _globals["abs"] = new NSLBuiltinFunction("abs", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("abs() takes exactly one argument");
                return Math.Abs(ConvertToNumber(args[0]));
            });

            // Rounding functions
            _globals["floor"] = new NSLBuiltinFunction("floor", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("floor() takes exactly one argument");
                return Math.Floor(ConvertToNumber(args[0]));
            });

            _globals["ceil"] = new NSLBuiltinFunction("ceil", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("ceil() takes exactly one argument");
                return Math.Ceiling(ConvertToNumber(args[0]));
            });

            _globals["round"] = new NSLBuiltinFunction("round", (args) =>
            {
                if (args.Length < 1 || args.Length > 2)
                    throw new NSLRuntimeException("round() takes 1 or 2 arguments");
                var n = ConvertToNumber(args[0]);
                if (args.Length == 2)
                {
                    var decimals = (int)ConvertToNumber(args[1]);
                    return Math.Round(n, decimals, MidpointRounding.AwayFromZero);
                }
                return Math.Round(n, MidpointRounding.AwayFromZero);
            });

            // Logarithmic and exponential functions
            _globals["log"] = new NSLBuiltinFunction("log", (args) =>
            {
                if (args.Length < 1 || args.Length > 2)
                    throw new NSLRuntimeException("log() takes 1 or 2 arguments");
                var n = ConvertToNumber(args[0]);
                if (n <= 0)
                    throw new NSLRuntimeException("log() argument must be positive");
                if (args.Length == 2)
                {
                    var baseVal = ConvertToNumber(args[1]);
                    return Math.Log(n, baseVal);
                }
                return Math.Log(n); // Natural log
            });

            _globals["log10"] = new NSLBuiltinFunction("log10", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("log10() takes exactly one argument");
                var n = ConvertToNumber(args[0]);
                if (n <= 0)
                    throw new NSLRuntimeException("log10() argument must be positive");
                return Math.Log10(n);
            });

            _globals["exp"] = new NSLBuiltinFunction("exp", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("exp() takes exactly one argument");
                return Math.Exp(ConvertToNumber(args[0]));
            });

            _globals["pow"] = new NSLBuiltinFunction("pow", (args) =>
            {
                if (args.Length != 2)
                    throw new NSLRuntimeException("pow() takes exactly two arguments");
                return Math.Pow(ConvertToNumber(args[0]), ConvertToNumber(args[1]));
            });

            // Min and max functions
            _globals["min"] = new NSLBuiltinFunction("min", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("min() takes at least one argument");

                // Handle array argument (support all list/collection types)
                if (args.Length == 1 && args[0] is System.Collections.IEnumerable enumerable && args[0] is not string)
                {
                    var items = enumerable.Cast<object>().ToList();
                    if (items.Count == 0)
                        throw new NSLRuntimeException("min() cannot operate on empty array");
                    return items.Select(x => ConvertToNumber(x)).Min();
                }

                return args.Select(x => ConvertToNumber(x)).Min();
            });

            _globals["max"] = new NSLBuiltinFunction("max", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("max() takes at least one argument");

                // Handle array argument (support all list/collection types)
                if (args.Length == 1 && args[0] is System.Collections.IEnumerable enumerable && args[0] is not string)
                {
                    var items = enumerable.Cast<object>().ToList();
                    if (items.Count == 0)
                        throw new NSLRuntimeException("max() cannot operate on empty array");
                    return items.Select(x => ConvertToNumber(x)).Max();
                }

                return args.Select(x => ConvertToNumber(x)).Max();
            });

            // Sign function
            _globals["sign"] = new NSLBuiltinFunction("sign", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("sign() takes exactly one argument");
                return (double)Math.Sign(ConvertToNumber(args[0]));
            });

            // Random number
            _globals["random"] = new NSLBuiltinFunction("random", (args) =>
            {
                if (args.Length == 0)
                    return _random.NextDouble(); // 0.0 to 1.0
                if (args.Length == 1)
                {
                    var max = (int)ConvertToNumber(args[0]);
                    return (double)_random.Next(max);
                }
                if (args.Length == 2)
                {
                    var min = (int)ConvertToNumber(args[0]);
                    var max = (int)ConvertToNumber(args[1]);
                    return (double)_random.Next(min, max);
                }
                throw new NSLRuntimeException("random() takes 0 to 2 arguments");
            });

            // ===================================
            // Result/Option Type Helpers
            // ===================================

            // Create ok result
            _globals["ok"] = new NSLBuiltinFunction("ok", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("ok() takes exactly one argument");
                return new NSLResult(true, args[0]);
            });

            // Create err result
            _globals["err"] = new NSLBuiltinFunction("err", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("err() takes exactly one argument");
                return new NSLResult(false, args[0]);
            });

            // Check if result is ok
            _globals["is_ok"] = new NSLBuiltinFunction("is_ok", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("is_ok() takes exactly one argument");
                return args[0] is NSLResult result && result.IsOk;
            });

            // Check if result is err
            _globals["is_err"] = new NSLBuiltinFunction("is_err", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("is_err() takes exactly one argument");
                return args[0] is NSLResult result && !result.IsOk;
            });

            // Unwrap a result - get value from ok, throw on err
            _globals["unwrap"] = new NSLBuiltinFunction("unwrap", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("unwrap() takes exactly one argument");
                if (args[0] is NSLResult result)
                {
                    if (result.IsOk)
                        return result.Value;
                    throw new NSLRuntimeException($"Called unwrap() on err: {result.Value}");
                }
                if (args[0] is NSLOptional optional)
                {
                    if (optional.HasValue)
                        return optional.Value;
                    throw new NSLRuntimeException("Called unwrap() on none");
                }
                throw new NSLRuntimeException("unwrap() expects a Result or Optional");
            });

            // Unwrap with default value
            _globals["unwrap_or"] = new NSLBuiltinFunction("unwrap_or", (args) =>
            {
                if (args.Length != 2)
                    throw new NSLRuntimeException("unwrap_or() takes exactly two arguments");
                if (args[0] is NSLResult result)
                {
                    return result.IsOk ? result.Value : args[1];
                }
                if (args[0] is NSLOptional optional)
                {
                    return optional.HasValue ? optional.Value : args[1];
                }
                throw new NSLRuntimeException("unwrap_or() expects a Result or Optional as first argument");
            });

            // Create some optional
            _globals["some"] = new NSLBuiltinFunction("some", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("some() takes exactly one argument");
                return new NSLOptional(args[0], true);
            });

            // Create none optional
            _globals["none"] = new NSLBuiltinFunction("none", (args) =>
            {
                return new NSLOptional(null, false);
            });

            // Check if optional has value
            _globals["is_some"] = new NSLBuiltinFunction("is_some", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("is_some() takes exactly one argument");
                return args[0] is NSLOptional optional && optional.HasValue;
            });

            // Check if optional is none
            _globals["is_none"] = new NSLBuiltinFunction("is_none", (args) =>
            {
                if (args.Length != 1)
                    throw new NSLRuntimeException("is_none() takes exactly one argument");
                return args[0] is NSLOptional optional && !optional.HasValue;
            });

            // ===================================
            // Consciousness Functions
            // ===================================

            // Channel operations
            _globals["channel"] = new NSLBuiltinFunction("channel", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("channel() requires a channel name");
                var name = args[0]?.ToString() ?? "default";
                var bufferSize = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 100;
                return ConsciousnessOperators.CreateChannel(name, bufferSize);
            });

            _globals["send"] = new NSLBuiltinFunction("send", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("send() requires channel name and message");
                var channelName = args[0]?.ToString() ?? "default";
                return ConsciousnessOperators.ChannelSend(channelName, args[1]!);
            });

            _globals["receive"] = new NSLBuiltinFunction("receive", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("receive() requires a channel name");
                var channelName = args[0]?.ToString() ?? "default";
                return ConsciousnessOperators.ChannelReceive(channelName);
            });

            _globals["channels"] = new NSLBuiltinFunction("channels", (args) =>
            {
                return ConsciousnessOperators.ListChannels();
            });

            // Checkpoint operations
            _globals["checkpoint"] = new NSLBuiltinFunction("checkpoint", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("checkpoint() requires a name");
                var name = args[0]?.ToString() ?? "default";
                return ConsciousnessOperators.CreateCheckpoint(name);
            });

            _globals["restore"] = new NSLBuiltinFunction("restore", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("restore() requires a checkpoint name");
                var name = args[0]?.ToString() ?? "default";
                return ConsciousnessOperators.RestoreCheckpoint(name);
            });

            _globals["checkpoints"] = new NSLBuiltinFunction("checkpoints", (args) =>
            {
                return ConsciousnessOperators.ListCheckpoints();
            });

            // Trace operations
            _globals["trace_on"] = new NSLBuiltinFunction("trace_on", (args) =>
            {
                return ConsciousnessOperators.SetTracing(true);
            });

            _globals["trace_off"] = new NSLBuiltinFunction("trace_off", (args) =>
            {
                return ConsciousnessOperators.SetTracing(false);
            });

            _globals["trace_log"] = new NSLBuiltinFunction("trace_log", (args) =>
            {
                var limit = args.Length > 0 ? (int?)ConvertToNumber(args[0]) : null;
                return ConsciousnessOperators.GetTraceLog(limit);
            });

            _globals["trace_clear"] = new NSLBuiltinFunction("trace_clear", (args) =>
            {
                return ConsciousnessOperators.ClearTraceLog();
            });

            // Consciousness measurement function
            _globals["measure"] = new NSLBuiltinFunction("measure", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("measure() requires an argument");
                return ConsciousnessOperators.Measure(args[0]!);
            });

            // Entangle function
            _globals["entangle"] = new NSLBuiltinFunction("entangle", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("entangle() requires two arguments");
                return ConsciousnessOperators.Entangle(args[0]!, args[1]!);
            });

            // Memory functions (convenience wrappers)
            _globals["mem_store"] = new NSLBuiltinFunction("mem_store", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("mem_store() requires key and value");
                return ConsciousnessOperators.MemoryStore(args[0]!, args[1]!);
            });

            _globals["mem_recall"] = new NSLBuiltinFunction("mem_recall", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("mem_recall() requires a key");
                return ConsciousnessOperators.MemoryRecall(args[0]!);
            });

            _globals["mem_list"] = new NSLBuiltinFunction("mem_list", (args) =>
            {
                var store = ConsciousnessOperators.GetMemoryStore();
                return new Dictionary<string, object>
                {
                    ["type"] = "consciousness:memory_list",
                    ["keys"] = store.Keys.ToList(),
                    ["count"] = store.Count
                };
            });

            // ===== SHELL/COMMAND EXECUTION =====
            // Execute shell command and return output as string
            _globals["shell"] = new NSLBuiltinFunction("shell", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("shell() requires a command argument");

                var command = args[0]?.ToString() ?? "";
                var workingDir = args.Length > 1 ? args[1]?.ToString() : null;

                try
                {
                    var processInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = OperatingSystem.IsWindows() ? "cmd.exe" : "/bin/bash",
                        Arguments = OperatingSystem.IsWindows() ? $"/c {command}" : $"-c \"{command}\"",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };

                    if (!string.IsNullOrEmpty(workingDir))
                        processInfo.WorkingDirectory = workingDir;

                    using var process = System.Diagnostics.Process.Start(processInfo);
                    if (process == null)
                        throw new NSLRuntimeException("Failed to start process");

                    var output = process.StandardOutput.ReadToEnd();
                    var error = process.StandardError.ReadToEnd();
                    process.WaitForExit();

                    return new Dictionary<string, object>
                    {
                        ["output"] = output,
                        ["error"] = error,
                        ["exit_code"] = process.ExitCode,
                        ["success"] = process.ExitCode == 0
                    };
                }
                catch (Exception ex)
                {
                    return new Dictionary<string, object>
                    {
                        ["output"] = "",
                        ["error"] = ex.Message,
                        ["exit_code"] = -1,
                        ["success"] = false
                    };
                }
            });

            // Execute command and return just exit code (simpler version)
            _globals["exec"] = new NSLBuiltinFunction("exec", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("exec() requires a command argument");

                var command = args[0]?.ToString() ?? "";

                try
                {
                    var processInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = OperatingSystem.IsWindows() ? "cmd.exe" : "/bin/bash",
                        Arguments = OperatingSystem.IsWindows() ? $"/c {command}" : $"-c \"{command}\"",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };

                    using var process = System.Diagnostics.Process.Start(processInfo);
                    if (process == null)
                        return -1.0;

                    process.WaitForExit();
                    return (double)process.ExitCode;
                }
                catch
                {
                    return -1.0;
                }
            });

            // ===== CLAUDE CODE INTEGRATION =====
            // Direct Claude Code access from NSL
            _globals["claude"] = new NSLBuiltinFunction("claude", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("claude() requires a prompt argument");

                var prompt = args[0]?.ToString() ?? "";
                var printMode = args.Length > 1 && args[1]?.ToString() == "print";

                try
                {
                    var processInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "claude",
                        Arguments = $"--print \"{prompt.Replace("\"", "\\\"")}\"",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };

                    using var process = System.Diagnostics.Process.Start(processInfo);
                    if (process == null)
                    {
                        return new Dictionary<string, object>
                        {
                            ["error"] = true,
                            ["message"] = "Failed to start Claude. Is it installed? Run: npm install -g @anthropic-ai/claude-code"
                        };
                    }

                    var output = process.StandardOutput.ReadToEnd();
                    var error = process.StandardError.ReadToEnd();
                    process.WaitForExit();

                    if (process.ExitCode != 0 && !string.IsNullOrEmpty(error))
                    {
                        return new Dictionary<string, object>
                        {
                            ["error"] = true,
                            ["message"] = error,
                            ["exit_code"] = process.ExitCode
                        };
                    }

                    // Return just the response text for simplicity
                    return output.Trim();
                }
                catch (Exception ex)
                {
                    return new Dictionary<string, object>
                    {
                        ["error"] = true,
                        ["message"] = $"Claude error: {ex.Message}",
                        ["hint"] = "Make sure Claude Code is installed: npm install -g @anthropic-ai/claude-code"
                    };
                }
            });

            // ===== CLAUDE CODE INTERACTIVE CHAT =====
            // Launch full Claude Code terminal experience from NSL
            // claude.chat() - Opens interactive Claude Code session
            // claude.chat("initial prompt") - Opens with initial message
            var claudeNamespace = new Dictionary<string, object?>
            {
                ["chat"] = new NSLBuiltinFunction("chat", (args) =>
                {
                    try
                    {
                        var initialPrompt = args.Length > 0 ? args[0]?.ToString() : null;

                        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════╗");
                        Console.WriteLine("║         Launching Claude Code on NSL...                      ║");
                        Console.WriteLine("║         Claude will use NSL tools for execution              ║");
                        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝\n");

                        // Find Claude Code CLI directly (bypass shell wrappers)
                        var npmPath = Path.Combine(
                            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                            "npm", "node_modules", "@anthropic-ai", "claude-code", "cli.js"
                        );

                        if (!File.Exists(npmPath))
                        {
                            Console.WriteLine($"Claude Code not found at: {npmPath}");
                            Console.WriteLine("Install with: npm install -g @anthropic-ai/claude-code");
                            return false;
                        }

                        var processInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "node",
                            UseShellExecute = false,           // Don't use shell - run inline
                            CreateNoWindow = false,
                            RedirectStandardInput = false,     // Inherit stdin from parent
                            RedirectStandardOutput = false,    // Inherit stdout from parent
                            RedirectStandardError = false      // Inherit stderr from parent
                        };

                        // Build arguments: cli.js [initial prompt]
                        if (!string.IsNullOrEmpty(initialPrompt))
                        {
                            processInfo.Arguments = $"\"{npmPath}\" \"{initialPrompt.Replace("\"", "\\\"")}\"";
                        }
                        else
                        {
                            processInfo.Arguments = $"\"{npmPath}\"";
                        }

                        using var process = System.Diagnostics.Process.Start(processInfo);
                        if (process == null)
                        {
                            Console.WriteLine("Failed to start Claude Code");
                            return false;
                        }

                        process.WaitForExit();

                        Console.WriteLine("\n[Returned to NSL]");
                        return true;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error launching Claude Code: {ex.Message}");
                        Console.WriteLine("Make sure Claude Code is installed: npm install -g @anthropic-ai/claude-code");
                        return false;
                    }
                }),

                ["resume"] = new NSLBuiltinFunction("resume", (args) =>
                {
                    try
                    {
                        Console.WriteLine("\n[Resuming last Claude Code session...]\n");

                        // Find Claude Code CLI directly
                        var npmPath = Path.Combine(
                            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                            "npm", "node_modules", "@anthropic-ai", "claude-code", "cli.js"
                        );

                        if (!File.Exists(npmPath))
                        {
                            Console.WriteLine($"Claude Code not found at: {npmPath}");
                            return false;
                        }

                        var processInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "node",
                            Arguments = $"\"{npmPath}\" --resume",
                            UseShellExecute = false,           // Don't use shell - run inline
                            CreateNoWindow = false,
                            RedirectStandardInput = false,     // Inherit stdin from parent
                            RedirectStandardOutput = false,    // Inherit stdout from parent
                            RedirectStandardError = false      // Inherit stderr from parent
                        };

                        using var process = System.Diagnostics.Process.Start(processInfo);
                        process?.WaitForExit();

                        Console.WriteLine("\n[Returned to NSL]");
                        return true;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                        return false;
                    }
                }),

                ["config"] = new NSLBuiltinFunction("config", (args) =>
                {
                    // Show Claude Code config info
                    return new Dictionary<string, object>
                    {
                        ["mcp_server"] = "E:\\NSL.Interpreter\\mcp-server\\index.js",
                        ["tools"] = new[] { "nsl", "nsl_think", "nsl_gpu", "nsl_consciousness", "nsl_learn", "nsl_introspect" },
                        ["status"] = "Claude Code will use NSL as execution environment"
                    };
                }),

                ["ask"] = new NSLBuiltinFunction("ask", (args) =>
                {
                    // Quick one-off prompt to Claude (non-interactive)
                    if (args.Length < 1)
                        throw new NSLRuntimeException("claude.ask() requires a prompt argument");

                    var prompt = args[0]?.ToString() ?? "";

                    try
                    {
                        var processInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "claude",
                            Arguments = $"--print \"{prompt.Replace("\"", "\\\"")}\"",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        };

                        using var process = System.Diagnostics.Process.Start(processInfo);
                        if (process == null)
                            throw new NSLRuntimeException("Failed to start Claude");

                        var output = process.StandardOutput.ReadToEnd();
                        process.WaitForExit();

                        return output.Trim();
                    }
                    catch (Exception ex)
                    {
                        return $"Error: {ex.Message}";
                    }
                }),

                ["help"] = new NSLBuiltinFunction("help", (args) =>
                {
                    Console.WriteLine("\n╔══════════════════════════════════════════════════════════════╗");
                    Console.WriteLine("║                    Claude Code on NSL                        ║");
                    Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
                    Console.WriteLine("║  claude.chat()        - Launch full Claude Code terminal     ║");
                    Console.WriteLine("║  claude.chat(\"msg\")   - Launch with initial message          ║");
                    Console.WriteLine("║  claude.resume()      - Resume last session                  ║");
                    Console.WriteLine("║  claude.ask(\"prompt\") - Quick one-off question               ║");
                    Console.WriteLine("║  claude.config()      - Show MCP configuration               ║");
                    Console.WriteLine("║  claude.help()        - Show this help                       ║");
                    Console.WriteLine("╚══════════════════════════════════════════════════════════════╝\n");
                    return null;
                })
            };
            _globals["claude"] = claudeNamespace;

            // ===== GPU ACCELERATION =====
            // GPU namespace for tensor operations using ILGPU
            // Lazy-initialized to avoid GPU overhead until needed
            var gpuNamespace = new Dictionary<string, object?>
            {
                // gpu.init() - Initialize GPU and return info
                ["init"] = new NSLBuiltinFunction("init", (args) =>
                {
                    try
                    {
                        lock (_gpuLock)
                        {
                            if (_gpuConfig == null)
                            {
                                _gpuConfig = new GpuAutoConfig();
                                _gpuKernels = _gpuConfig.GetKernels();
                            }
                        }

                        var gpu = _gpuConfig.CurrentGpu;
                        return new Dictionary<string, object?>
                        {
                            ["name"] = gpu?.Name ?? "CPU Fallback",
                            ["backend"] = gpu?.Backend.ToString() ?? "CPU",
                            ["vram_mb"] = (double)((gpu?.MemoryBytes ?? 0) / 1024 / 1024),
                            ["tensor_cores"] = gpu?.SupportsTensorCores ?? false,
                            ["float16"] = gpu?.SupportsFloat16 ?? false,
                            ["int8"] = gpu?.SupportsInt8 ?? false,
                            ["compute_units"] = (double)(gpu?.ComputeUnits ?? 0),
                            ["architecture"] = gpu?.Architecture.ToString() ?? "Unknown",
                            ["warp_size"] = (double)(gpu?.WarpSize ?? 0),
                            ["max_threads"] = (double)(gpu?.MaxThreadsPerBlock ?? 0)
                        };
                    }
                    catch (Exception ex)
                    {
                        return new Dictionary<string, object?>
                        {
                            ["name"] = "Initialization Failed",
                            ["error"] = ex.Message,
                            ["backend"] = "None"
                        };
                    }
                }),

                // gpu.devices() - List all available GPUs
                ["devices"] = new NSLBuiltinFunction("devices", (args) =>
                {
                    try
                    {
                        lock (_gpuLock)
                        {
                            if (_gpuConfig == null)
                                _gpuConfig = new GpuAutoConfig();
                        }

                        var devices = _gpuConfig.AvailableGpus;
                        var result = new List<object?>();

                        foreach (var gpu in devices)
                        {
                            result.Add(new Dictionary<string, object?>
                            {
                                ["name"] = gpu.Name,
                                ["backend"] = gpu.Backend.ToString(),
                                ["vram_mb"] = (double)(gpu.MemoryBytes / 1024 / 1024),
                                ["architecture"] = gpu.Architecture.ToString(),
                                ["score"] = (double)gpu.Score
                            });
                        }

                        return result;
                    }
                    catch (Exception ex)
                    {
                        return new List<object?> { new Dictionary<string, object?> { ["error"] = ex.Message } };
                    }
                }),

                // gpu.tensor(data, shape) - Create GPU tensor from array
                ["tensor"] = new NSLBuiltinFunction("tensor", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.tensor() requires data and shape arguments");

                    EnsureGpuInitialized();

                    var data = ConvertToFloatArray(args[0]);
                    var shape = ConvertToIntArray(args[1]);

                    var accelerator = _gpuConfig!.GetAccelerator();
                    return GpuTensor.FromArray(accelerator, data, shape);
                }),

                // gpu.zeros(shape) - Create GPU tensor filled with zeros
                ["zeros"] = new NSLBuiltinFunction("zeros", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.zeros() requires shape argument");

                    EnsureGpuInitialized();

                    var shape = ConvertToIntArray(args[0]);
                    var accelerator = _gpuConfig!.GetAccelerator();
                    return GpuTensor.Zeros(accelerator, shape);
                }),

                // gpu.ones(shape) - Create GPU tensor filled with ones
                ["ones"] = new NSLBuiltinFunction("ones", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.ones() requires shape argument");

                    EnsureGpuInitialized();

                    var shape = ConvertToIntArray(args[0]);
                    var accelerator = _gpuConfig!.GetAccelerator();
                    return GpuTensor.Ones(accelerator, shape);
                }),

                // gpu.random(shape) - Create GPU tensor with random values [0, 1)
                ["random"] = new NSLBuiltinFunction("random", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.random() requires shape argument");

                    EnsureGpuInitialized();

                    var shape = ConvertToIntArray(args[0]);
                    var accelerator = _gpuConfig!.GetAccelerator();
                    return GpuTensor.Random(accelerator, shape);
                }),

                // gpu.matmul(a, b) - Matrix multiplication
                ["matmul"] = new NSLBuiltinFunction("matmul", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.matmul() requires two tensor arguments");

                    EnsureGpuInitialized();

                    var a = args[0] as GpuTensor ?? throw new NSLRuntimeException("First argument must be a GPU tensor");
                    var b = args[1] as GpuTensor ?? throw new NSLRuntimeException("Second argument must be a GPU tensor");

                    return _gpuKernels!.MatMul(a, b);
                }),

                // gpu.add(a, b) - Element-wise addition
                ["add"] = new NSLBuiltinFunction("add", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.add() requires two tensor arguments");

                    EnsureGpuInitialized();

                    var a = args[0] as GpuTensor ?? throw new NSLRuntimeException("First argument must be a GPU tensor");
                    var b = args[1] as GpuTensor ?? throw new NSLRuntimeException("Second argument must be a GPU tensor");

                    return _gpuKernels!.Add(a, b);
                }),

                // gpu.sub(a, b) - Element-wise subtraction
                ["sub"] = new NSLBuiltinFunction("sub", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.sub() requires two tensor arguments");

                    EnsureGpuInitialized();

                    var a = args[0] as GpuTensor ?? throw new NSLRuntimeException("First argument must be a GPU tensor");
                    var b = args[1] as GpuTensor ?? throw new NSLRuntimeException("Second argument must be a GPU tensor");

                    return _gpuKernels!.Sub(a, b);
                }),

                // gpu.mul(a, b) - Element-wise multiplication
                ["mul"] = new NSLBuiltinFunction("mul", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.mul() requires two tensor arguments");

                    EnsureGpuInitialized();

                    var a = args[0] as GpuTensor ?? throw new NSLRuntimeException("First argument must be a GPU tensor");
                    var b = args[1] as GpuTensor ?? throw new NSLRuntimeException("Second argument must be a GPU tensor");

                    return _gpuKernels!.Mul(a, b);
                }),

                // gpu.div(a, b) - Element-wise division
                ["div"] = new NSLBuiltinFunction("div", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.div() requires two tensor arguments");

                    EnsureGpuInitialized();

                    var a = args[0] as GpuTensor ?? throw new NSLRuntimeException("First argument must be a GPU tensor");
                    var b = args[1] as GpuTensor ?? throw new NSLRuntimeException("Second argument must be a GPU tensor");

                    return _gpuKernels!.Div(a, b);
                }),

                // gpu.relu(x) - ReLU activation
                ["relu"] = new NSLBuiltinFunction("relu", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.relu() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.ReLU(x);
                }),

                // gpu.sigmoid(x) - Sigmoid activation
                ["sigmoid"] = new NSLBuiltinFunction("sigmoid", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.sigmoid() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.Sigmoid(x);
                }),

                // gpu.tanh(x) - Tanh activation
                ["tanh"] = new NSLBuiltinFunction("tanh", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.tanh() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.Tanh(x);
                }),

                // gpu.softmax(x) - Softmax activation
                ["softmax"] = new NSLBuiltinFunction("softmax", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.softmax() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.Softmax(x);
                }),

                // gpu.exp(x) - Element-wise exponential
                ["exp"] = new NSLBuiltinFunction("exp", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.exp() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.Exp(x);
                }),

                // gpu.log(x) - Element-wise natural log
                ["log"] = new NSLBuiltinFunction("log", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.log() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.Log(x);
                }),

                // gpu.sqrt(x) - Element-wise square root
                ["sqrt"] = new NSLBuiltinFunction("sqrt", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.sqrt() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.Sqrt(x);
                }),

                // gpu.pow(x, exp) - Element-wise power
                ["pow"] = new NSLBuiltinFunction("pow", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.pow() requires tensor and exponent arguments");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("First argument must be a GPU tensor");
                    var exp = args[1] switch
                    {
                        double d => (float)d,
                        int i => (float)i,
                        long l => (float)l,
                        float f => f,
                        _ => throw new NSLRuntimeException("Exponent must be a number")
                    };

                    return _gpuKernels!.Pow(x, exp);
                }),

                // gpu.transpose(x) - Transpose matrix
                ["transpose"] = new NSLBuiltinFunction("transpose", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.transpose() requires tensor argument");

                    EnsureGpuInitialized();

                    var x = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return _gpuKernels!.Transpose(x);
                }),

                // gpu.to_cpu(tensor) - Get tensor data as NSL array
                ["to_cpu"] = new NSLBuiltinFunction("to_cpu", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.to_cpu() requires tensor argument");

                    var tensor = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    var data = tensor.ToArray();

                    // Convert to NSL array (List<object?>)
                    return data.Select(x => (object?)(double)x).ToList();
                }),

                // gpu.shape(tensor) - Get tensor shape
                ["shape"] = new NSLBuiltinFunction("shape", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.shape() requires tensor argument");

                    var tensor = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return tensor.Shape.Select(x => (object?)(double)x).ToList();
                }),

                // gpu.size(tensor) - Get total element count
                ["size"] = new NSLBuiltinFunction("size", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.size() requires tensor argument");

                    var tensor = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    return (double)tensor.Size;
                }),

                // gpu.dispose(tensor) - Free GPU memory
                ["dispose"] = new NSLBuiltinFunction("dispose", (args) =>
                {
                    if (args.Length < 1)
                        throw new NSLRuntimeException("gpu.dispose() requires tensor argument");

                    var tensor = args[0] as GpuTensor ?? throw new NSLRuntimeException("Argument must be a GPU tensor");
                    tensor.Dispose();
                    return true;
                }),

                // gpu.shutdown() - Shutdown GPU and free all resources
                ["shutdown"] = new NSLBuiltinFunction("shutdown", (args) =>
                {
                    lock (_gpuLock)
                    {
                        if (_gpuConfig != null)
                        {
                            _gpuConfig.Dispose();
                            _gpuConfig = null;
                            _gpuKernels = null;
                        }
                    }
                    return true;
                }),

                // ===== GRAPH COMPILER & JIT (PyTorch-competitive) =====

                // gpu.graph() - Create computation graph for lazy evaluation
                ["graph"] = new NSLBuiltinFunction("graph", (args) =>
                {
                    EnsureGpuInitialized();
                    var accelerator = _gpuConfig!.GetAccelerator();
                    return new ComputationGraph(accelerator);
                }),

                // gpu.lazy(data, shape) - Create lazy tensor (deferred execution)
                ["lazy"] = new NSLBuiltinFunction("lazy", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.lazy() requires data and shape");
                    EnsureGpuInitialized();
                    var accelerator = _gpuConfig!.GetAccelerator();
                    return LazyTensor.FromArray(accelerator, ConvertToFloatArray(args[0]), ConvertToIntArray(args[1]));
                }),

                // gpu.flash_attention(q, k, v, causal?) - FlashAttention-2
                ["flash_attention"] = new NSLBuiltinFunction("flash_attention", (args) =>
                {
                    if (args.Length < 3)
                        throw new NSLRuntimeException("gpu.flash_attention() requires q, k, v tensors");
                    EnsureGpuInitialized();
                    var accelerator = _gpuConfig!.GetAccelerator();
                    var q = args[0] as GpuTensor ?? throw new NSLRuntimeException("q must be GPU tensor");
                    var k = args[1] as GpuTensor ?? throw new NSLRuntimeException("k must be GPU tensor");
                    var v = args[2] as GpuTensor ?? throw new NSLRuntimeException("v must be GPU tensor");
                    bool causal = args.Length > 3 && args[3] is bool c && c;
                    var flash = new FlashAttention2Engine(accelerator);
                    return flash.Forward(q, k, v, null, causal);
                }),

                // gpu.layer_norm(input, gamma, beta, eps?) - Fused LayerNorm
                ["layer_norm"] = new NSLBuiltinFunction("layer_norm", (args) =>
                {
                    if (args.Length < 3)
                        throw new NSLRuntimeException("gpu.layer_norm() requires input, gamma, beta");
                    EnsureGpuInitialized();
                    var accelerator = _gpuConfig!.GetAccelerator();
                    var hpk = new HighPerformanceKernels(accelerator);
                    var input = args[0] as GpuTensor ?? throw new NSLRuntimeException("input must be GPU tensor");
                    var gamma = args[1] as GpuTensor ?? throw new NSLRuntimeException("gamma must be GPU tensor");
                    var beta = args[2] as GpuTensor ?? throw new NSLRuntimeException("beta must be GPU tensor");
                    float eps = args.Length > 3 ? Convert.ToSingle(args[3]) : 1e-5f;
                    return hpk.FusedLayerNorm(input, gamma, beta, eps);
                }),

                // gpu.rms_norm(input, gamma, eps?) - RMS Norm (LLaMA-style)
                ["rms_norm"] = new NSLBuiltinFunction("rms_norm", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.rms_norm() requires input, gamma");
                    EnsureGpuInitialized();
                    var accelerator = _gpuConfig!.GetAccelerator();
                    var hpk = new HighPerformanceKernels(accelerator);
                    var input = args[0] as GpuTensor ?? throw new NSLRuntimeException("input must be GPU tensor");
                    var gamma = args[1] as GpuTensor ?? throw new NSLRuntimeException("gamma must be GPU tensor");
                    float eps = args.Length > 2 ? Convert.ToSingle(args[2]) : 1e-5f;
                    return hpk.RMSNorm(input, gamma, eps);
                }),

                // gpu.fused_linear_relu(input, weight, bias) - Fused Linear+ReLU
                ["fused_linear_relu"] = new NSLBuiltinFunction("fused_linear_relu", (args) =>
                {
                    if (args.Length < 3)
                        throw new NSLRuntimeException("gpu.fused_linear_relu() requires input, weight, bias");
                    EnsureGpuInitialized();
                    var accelerator = _gpuConfig!.GetAccelerator();
                    var hpk = new HighPerformanceKernels(accelerator);
                    var input = args[0] as GpuTensor ?? throw new NSLRuntimeException("input must be GPU tensor");
                    var weight = args[1] as GpuTensor ?? throw new NSLRuntimeException("weight must be GPU tensor");
                    var bias = args[2] as GpuTensor ?? throw new NSLRuntimeException("bias must be GPU tensor");
                    return hpk.FusedMatMulBiasReLU(input, weight, bias);
                }),

                // gpu.fused_gelu(input, bias) - Fused Bias+GELU
                ["fused_gelu"] = new NSLBuiltinFunction("fused_gelu", (args) =>
                {
                    if (args.Length < 2)
                        throw new NSLRuntimeException("gpu.fused_gelu() requires input, bias");
                    EnsureGpuInitialized();
                    var accelerator = _gpuConfig!.GetAccelerator();
                    var fusion = new OperatorFusion(accelerator);
                    var input = args[0] as GpuTensor ?? throw new NSLRuntimeException("input must be GPU tensor");
                    var bias = args[1] as GpuTensor ?? throw new NSLRuntimeException("bias must be GPU tensor");
                    return fusion.FusedBiasGelu(input, bias);
                })
            };
            _globals["gpu"] = gpuNamespace;

            // ===== MATH NAMESPACE =====
            var mathNamespace = new Dictionary<string, object?>
            {
                ["PI"] = Math.PI,
                ["E"] = Math.E,
                ["sqrt"] = new NSLBuiltinFunction("sqrt", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sqrt() requires one argument"); return Math.Sqrt(ConvertToNumber(args[0])); }),
                ["sin"] = new NSLBuiltinFunction("sin", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sin() requires one argument"); return Math.Sin(ConvertToNumber(args[0])); }),
                ["cos"] = new NSLBuiltinFunction("cos", (args) => { if (args.Length < 1) throw new NSLRuntimeException("cos() requires one argument"); return Math.Cos(ConvertToNumber(args[0])); }),
                ["tan"] = new NSLBuiltinFunction("tan", (args) => { if (args.Length < 1) throw new NSLRuntimeException("tan() requires one argument"); return Math.Tan(ConvertToNumber(args[0])); }),
                ["abs"] = new NSLBuiltinFunction("abs", (args) => { if (args.Length < 1) throw new NSLRuntimeException("abs() requires one argument"); return Math.Abs(ConvertToNumber(args[0])); }),
                ["floor"] = new NSLBuiltinFunction("floor", (args) => { if (args.Length < 1) throw new NSLRuntimeException("floor() requires one argument"); return Math.Floor(ConvertToNumber(args[0])); }),
                ["ceil"] = new NSLBuiltinFunction("ceil", (args) => { if (args.Length < 1) throw new NSLRuntimeException("ceil() requires one argument"); return Math.Ceiling(ConvertToNumber(args[0])); }),
                ["round"] = new NSLBuiltinFunction("round", (args) => { if (args.Length < 1) throw new NSLRuntimeException("round() requires one argument"); var n = ConvertToNumber(args[0]); return args.Length > 1 ? Math.Round(n, (int)ConvertToNumber(args[1])) : Math.Round(n); }),
                ["log"] = new NSLBuiltinFunction("log", (args) => { if (args.Length < 1) throw new NSLRuntimeException("log() requires one argument"); var n = ConvertToNumber(args[0]); return args.Length > 1 ? Math.Log(n, ConvertToNumber(args[1])) : Math.Log(n); }),
                ["log10"] = new NSLBuiltinFunction("log10", (args) => { if (args.Length < 1) throw new NSLRuntimeException("log10() requires one argument"); return Math.Log10(ConvertToNumber(args[0])); }),
                ["exp"] = new NSLBuiltinFunction("exp", (args) => { if (args.Length < 1) throw new NSLRuntimeException("exp() requires one argument"); return Math.Exp(ConvertToNumber(args[0])); }),
                ["pow"] = new NSLBuiltinFunction("pow", (args) => { if (args.Length < 2) throw new NSLRuntimeException("pow() requires two arguments"); return Math.Pow(ConvertToNumber(args[0]), ConvertToNumber(args[1])); }),
                ["min"] = new NSLBuiltinFunction("min", (args) => { if (args.Length < 2) throw new NSLRuntimeException("min() requires two arguments"); return Math.Min(ConvertToNumber(args[0]), ConvertToNumber(args[1])); }),
                ["max"] = new NSLBuiltinFunction("max", (args) => { if (args.Length < 2) throw new NSLRuntimeException("max() requires two arguments"); return Math.Max(ConvertToNumber(args[0]), ConvertToNumber(args[1])); }),
                ["sign"] = new NSLBuiltinFunction("sign", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sign() requires one argument"); return (double)Math.Sign(ConvertToNumber(args[0])); }),
                ["random"] = new NSLBuiltinFunction("random", (args) => new Random().NextDouble())
            };
            _globals["math"] = mathNamespace;

            // ===== LIST NAMESPACE =====
            var listNamespace = new Dictionary<string, object?>
            {
                ["sum"] = new NSLBuiltinFunction("sum", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.sum() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); double sum = 0; foreach (var item in list) sum += ConvertToNumber(item); return sum; }),
                ["avg"] = new NSLBuiltinFunction("avg", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.avg() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); if (list.Count == 0) return 0.0; double sum = 0; foreach (var item in list) sum += ConvertToNumber(item); return sum / list.Count; }),
                ["min"] = new NSLBuiltinFunction("min", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.min() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); if (list.Count == 0) throw new NSLRuntimeException("Cannot get min of empty list"); double minVal = ConvertToNumber(list[0]); foreach (var item in list) { var val = ConvertToNumber(item); if (val < minVal) minVal = val; } return minVal; }),
                ["max"] = new NSLBuiltinFunction("max", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.max() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); if (list.Count == 0) throw new NSLRuntimeException("Cannot get max of empty list"); double maxVal = ConvertToNumber(list[0]); foreach (var item in list) { var val = ConvertToNumber(item); if (val > maxVal) maxVal = val; } return maxVal; }),
                ["length"] = new NSLBuiltinFunction("length", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.length() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); return (double)list.Count; }),
                ["reverse"] = new NSLBuiltinFunction("reverse", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.reverse() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); var result = new List<object?>(list); result.Reverse(); return result; }),
                ["sort"] = new NSLBuiltinFunction("sort", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.sort() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); var result = new List<object?>(list); result.Sort((a, b) => ConvertToNumber(a).CompareTo(ConvertToNumber(b))); return result; }),
                ["contains"] = new NSLBuiltinFunction("contains", (args) => { if (args.Length < 2) throw new NSLRuntimeException("list.contains() requires list and value"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); return list.Contains(args[1]); }),
                ["join"] = new NSLBuiltinFunction("join", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.join() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); var separator = args.Length > 1 ? args[1]?.ToString() ?? "" : ","; return string.Join(separator, list.Select(x => x?.ToString() ?? "")); }),
                ["range"] = new NSLBuiltinFunction("range", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.range() requires at least one argument"); int start = 0, end, step = 1; if (args.Length == 1) { end = (int)ConvertToNumber(args[0]); } else { start = (int)ConvertToNumber(args[0]); end = (int)ConvertToNumber(args[1]); if (args.Length > 2) step = (int)ConvertToNumber(args[2]); } var result = new List<object?>(); for (int i = start; step > 0 ? i < end : i > end; i += step) result.Add((double)i); return result; }),
                ["append"] = new NSLBuiltinFunction("append", (args) => { if (args.Length < 2) throw new NSLRuntimeException("list.append() requires list and value"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); var result = new List<object?>(list); result.Add(args[1]); return result; }),
                ["prepend"] = new NSLBuiltinFunction("prepend", (args) => { if (args.Length < 2) throw new NSLRuntimeException("list.prepend() requires list and value"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); var result = new List<object?>(list); result.Insert(0, args[1]); return result; }),
                ["remove"] = new NSLBuiltinFunction("remove", (args) => { if (args.Length < 2) throw new NSLRuntimeException("list.remove() requires list and value"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); var result = new List<object?>(list); result.Remove(args[1]); return result; }),
                ["slice"] = new NSLBuiltinFunction("slice", (args) => { if (args.Length < 2) throw new NSLRuntimeException("list.slice() requires list and start index"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); int start = (int)ConvertToNumber(args[1]); int count = args.Length > 2 ? (int)ConvertToNumber(args[2]) : list.Count - start; if (start < 0) start = list.Count + start; if (start < 0) start = 0; if (start >= list.Count) return new List<object?>(); count = Math.Min(count, list.Count - start); return list.Skip(start).Take(count).ToList(); }),
                ["get"] = new NSLBuiltinFunction("get", (args) => { if (args.Length < 2) throw new NSLRuntimeException("list.get() requires list and index"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); int index = (int)ConvertToNumber(args[1]); if (index < 0) index = list.Count + index; if (index < 0 || index >= list.Count) throw new NSLRuntimeException($"Index {index} out of range for list of length {list.Count}"); return list[index]; }),
                ["first"] = new NSLBuiltinFunction("first", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.first() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); if (list.Count == 0) throw new NSLRuntimeException("Cannot get first of empty list"); return list[0]; }),
                ["last"] = new NSLBuiltinFunction("last", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.last() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); if (list.Count == 0) throw new NSLRuntimeException("Cannot get last of empty list"); return list[list.Count - 1]; }),
                ["concat"] = new NSLBuiltinFunction("concat", (args) => { if (args.Length < 2) throw new NSLRuntimeException("list.concat() requires two lists"); var list1 = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list"); var list2 = args[1] as IList<object?> ?? throw new NSLRuntimeException("Second argument must be a list"); var result = new List<object?>(list1); result.AddRange(list2); return result; }),
                ["unique"] = new NSLBuiltinFunction("unique", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.unique() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); return list.Distinct().ToList(); }),
                ["flatten"] = new NSLBuiltinFunction("flatten", (args) => { if (args.Length < 1) throw new NSLRuntimeException("list.flatten() requires a list argument"); var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list"); var result = new List<object?>(); void Flatten(IEnumerable<object?> items) { foreach (var item in items) { if (item is IList<object?> sublist) Flatten(sublist); else result.Add(item); } } Flatten(list); return result; }),
                // Map and filter with function support
                ["map"] = new NSLBuiltinFunction("map", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.map() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var result = new List<object?>();
                    foreach (var item in list) {
                        if (fn is NSLFunction userFn) result.Add(CallUserFunction(userFn, new[] { item }));
                        else if (fn is NSLBuiltinFunction builtinFn) result.Add(builtinFn.Call(new[] { item }));
                        else throw new NSLRuntimeException("Second argument must be a function");
                    }
                    return result;
                }),
                ["filter"] = new NSLBuiltinFunction("filter", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.filter() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var result = new List<object?>();
                    foreach (var item in list) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (IsTruthy(predResult)) result.Add(item);
                    }
                    return result;
                }),
                ["mapIndexed"] = new NSLBuiltinFunction("mapIndexed", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.mapIndexed() requires list and function(item, index)");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var result = new List<object?>();
                    for (int i = 0; i < list.Count; i++) {
                        if (fn is NSLFunction userFn) result.Add(CallUserFunction(userFn, new object?[] { list[i], (double)i }));
                        else if (fn is NSLBuiltinFunction builtinFn) result.Add(builtinFn.Call(new object?[] { list[i], (double)i }));
                        else throw new NSLRuntimeException("Second argument must be a function");
                    }
                    return result;
                }),
                ["filterIndexed"] = new NSLBuiltinFunction("filterIndexed", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.filterIndexed() requires list and function(item, index)");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var result = new List<object?>();
                    for (int i = 0; i < list.Count; i++) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new object?[] { list[i], (double)i });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new object?[] { list[i], (double)i });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (IsTruthy(predResult)) result.Add(list[i]);
                    }
                    return result;
                }),
                ["forEach"] = new NSLBuiltinFunction("forEach", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.forEach() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    foreach (var item in list) {
                        if (fn is NSLFunction userFn) CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                    }
                    return null;
                }),
                ["forEachIndexed"] = new NSLBuiltinFunction("forEachIndexed", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.forEachIndexed() requires list and function(item, index)");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    for (int i = 0; i < list.Count; i++) {
                        if (fn is NSLFunction userFn) CallUserFunction(userFn, new object?[] { list[i], (double)i });
                        else if (fn is NSLBuiltinFunction builtinFn) builtinFn.Call(new object?[] { list[i], (double)i });
                        else throw new NSLRuntimeException("Second argument must be a function");
                    }
                    return null;
                }),
                ["reduce"] = new NSLBuiltinFunction("reduce", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.reduce() requires list and function(acc, item)");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    if (list.Count == 0) throw new NSLRuntimeException("Cannot reduce empty list without initial value");
                    object? acc = args.Length > 2 ? args[2] : list[0];
                    int start = args.Length > 2 ? 0 : 1;
                    for (int i = start; i < list.Count; i++) {
                        if (fn is NSLFunction userFn) acc = CallUserFunction(userFn, new object?[] { acc, list[i] });
                        else if (fn is NSLBuiltinFunction builtinFn) acc = builtinFn.Call(new object?[] { acc, list[i] });
                        else throw new NSLRuntimeException("Second argument must be a function");
                    }
                    return acc;
                }),
                ["find"] = new NSLBuiltinFunction("find", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.find() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    foreach (var item in list) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (IsTruthy(predResult)) return item;
                    }
                    return null;
                }),
                ["findIndex"] = new NSLBuiltinFunction("findIndex", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.findIndex() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    for (int i = 0; i < list.Count; i++) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { list[i] });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { list[i] });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (IsTruthy(predResult)) return (double)i;
                    }
                    return -1.0;
                }),
                ["every"] = new NSLBuiltinFunction("every", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.every() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    foreach (var item in list) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (!IsTruthy(predResult)) return false;
                    }
                    return true;
                }),
                ["some"] = new NSLBuiltinFunction("some", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.some() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    foreach (var item in list) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (IsTruthy(predResult)) return true;
                    }
                    return false;
                }),
                ["zip"] = new NSLBuiltinFunction("zip", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.zip() requires two lists");
                    var list1 = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var list2 = args[1] as IList<object?> ?? throw new NSLRuntimeException("Second argument must be a list");
                    var result = new List<object?>();
                    var len = Math.Min(list1.Count, list2.Count);
                    for (int i = 0; i < len; i++) {
                        result.Add(new List<object?> { list1[i], list2[i] });
                    }
                    return result;
                }),
                ["enumerate"] = new NSLBuiltinFunction("enumerate", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("list.enumerate() requires a list");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list");
                    var result = new List<object?>();
                    for (int i = 0; i < list.Count; i++) {
                        result.Add(new List<object?> { (double)i, list[i] });
                    }
                    return result;
                }),
                ["groupBy"] = new NSLBuiltinFunction("groupBy", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.groupBy() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var groups = new Dictionary<string, object?>();
                    foreach (var item in list) {
                        object? key;
                        if (fn is NSLFunction userFn) key = CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) key = builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        var keyStr = key?.ToString() ?? "null";
                        if (!groups.ContainsKey(keyStr)) groups[keyStr] = new List<object?>();
                        ((List<object?>)groups[keyStr]!).Add(item);
                    }
                    return groups;
                }),
                ["sortBy"] = new NSLBuiltinFunction("sortBy", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.sortBy() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var result = new List<object?>(list);
                    result.Sort((a, b) => {
                        object? keyA, keyB;
                        if (fn is NSLFunction userFn) { keyA = CallUserFunction(userFn, new[] { a }); keyB = CallUserFunction(userFn, new[] { b }); }
                        else if (fn is NSLBuiltinFunction builtinFn) { keyA = builtinFn.Call(new[] { a }); keyB = builtinFn.Call(new[] { b }); }
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (keyA is double dA && keyB is double dB) return dA.CompareTo(dB);
                        return (keyA?.ToString() ?? "").CompareTo(keyB?.ToString() ?? "");
                    });
                    return result;
                }),
                ["take"] = new NSLBuiltinFunction("take", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.take() requires list and count");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var count = (int)ConvertToNumber(args[1]);
                    return list.Take(count).ToList();
                }),
                ["drop"] = new NSLBuiltinFunction("drop", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.drop() requires list and count");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var count = (int)ConvertToNumber(args[1]);
                    return list.Skip(count).ToList();
                }),
                ["takeWhile"] = new NSLBuiltinFunction("takeWhile", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.takeWhile() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var result = new List<object?>();
                    foreach (var item in list) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (!IsTruthy(predResult)) break;
                        result.Add(item);
                    }
                    return result;
                }),
                ["dropWhile"] = new NSLBuiltinFunction("dropWhile", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.dropWhile() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var result = new List<object?>();
                    bool dropping = true;
                    foreach (var item in list) {
                        if (dropping) {
                            object? predResult;
                            if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { item });
                            else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { item });
                            else throw new NSLRuntimeException("Second argument must be a function");
                            if (!IsTruthy(predResult)) dropping = false;
                        }
                        if (!dropping) result.Add(item);
                    }
                    return result;
                }),
                ["partition"] = new NSLBuiltinFunction("partition", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.partition() requires list and function");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var fn = args[1];
                    var pass = new List<object?>();
                    var fail = new List<object?>();
                    foreach (var item in list) {
                        object? predResult;
                        if (fn is NSLFunction userFn) predResult = CallUserFunction(userFn, new[] { item });
                        else if (fn is NSLBuiltinFunction builtinFn) predResult = builtinFn.Call(new[] { item });
                        else throw new NSLRuntimeException("Second argument must be a function");
                        if (IsTruthy(predResult)) pass.Add(item); else fail.Add(item);
                    }
                    return new List<object?> { pass, fail };
                }),
                // Advanced list operations inspired by Lodash
                ["chunk"] = new NSLBuiltinFunction("chunk", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.chunk() requires list and size");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var size = (int)ConvertToNumber(args[1]);
                    if (size <= 0) throw new NSLRuntimeException("Chunk size must be positive");
                    var result = new List<object?>();
                    for (int i = 0; i < list.Count; i += size) {
                        result.Add(list.Skip(i).Take(size).ToList());
                    }
                    return result;
                }),
                ["shuffle"] = new NSLBuiltinFunction("shuffle", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("list.shuffle() requires a list");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list");
                    var result = list.ToList();
                    var rng = new Random();
                    int n = result.Count;
                    while (n > 1) { n--; int k = rng.Next(n + 1); var temp = result[k]; result[k] = result[n]; result[n] = temp; }
                    return result;
                }),
                ["sample"] = new NSLBuiltinFunction("sample", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("list.sample() requires a list");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list");
                    var count = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 1;
                    if (list.Count == 0) return count == 1 ? null : new List<object?>();
                    var rng = new Random();
                    if (count == 1) return list[rng.Next(list.Count)];
                    var shuffled = list.OrderBy(_ => rng.Next()).Take(count).ToList();
                    return shuffled;
                }),
                ["compact"] = new NSLBuiltinFunction("compact", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("list.compact() requires a list");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list");
                    return list.Where(x => x != null && !(x is bool b && !b) && !(x is double d && d == 0) && !(x is string s && string.IsNullOrEmpty(s))).ToList();
                }),
                ["difference"] = new NSLBuiltinFunction("difference", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.difference() requires two lists");
                    var list1 = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var list2 = args[1] as IList<object?> ?? throw new NSLRuntimeException("Second argument must be a list");
                    var set2 = new HashSet<string>(list2.Select(x => x?.ToString() ?? "null"));
                    return list1.Where(x => !set2.Contains(x?.ToString() ?? "null")).ToList();
                }),
                ["intersection"] = new NSLBuiltinFunction("intersection", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.intersection() requires two lists");
                    var list1 = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var list2 = args[1] as IList<object?> ?? throw new NSLRuntimeException("Second argument must be a list");
                    var set2 = new HashSet<string>(list2.Select(x => x?.ToString() ?? "null"));
                    return list1.Where(x => set2.Contains(x?.ToString() ?? "null")).ToList();
                }),
                ["union"] = new NSLBuiltinFunction("union", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.union() requires two lists");
                    var list1 = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var list2 = args[1] as IList<object?> ?? throw new NSLRuntimeException("Second argument must be a list");
                    var seen = new HashSet<string>();
                    var result = new List<object?>();
                    foreach (var item in list1.Concat(list2)) {
                        var key = item?.ToString() ?? "null";
                        if (!seen.Contains(key)) { seen.Add(key); result.Add(item); }
                    }
                    return result;
                }),
                ["without"] = new NSLBuiltinFunction("without", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.without() requires list and values to exclude");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var exclude = new HashSet<string>(args.Skip(1).Select(x => x?.ToString() ?? "null"));
                    return list.Where(x => !exclude.Contains(x?.ToString() ?? "null")).ToList();
                }),
                ["fill"] = new NSLBuiltinFunction("fill", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.fill() requires value and count");
                    var value = args[0];
                    var count = (int)ConvertToNumber(args[1]);
                    return Enumerable.Repeat(value, count).Cast<object?>().ToList();
                }),
                ["rotate"] = new NSLBuiltinFunction("rotate", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.rotate() requires list and count");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var n = (int)ConvertToNumber(args[1]);
                    if (list.Count == 0) return new List<object?>();
                    n = ((n % list.Count) + list.Count) % list.Count;
                    return list.Skip(n).Concat(list.Take(n)).ToList();
                }),
                ["head"] = new NSLBuiltinFunction("head", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("list.head() requires a list");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list");
                    return list.Count > 0 ? list[0] : null;
                }),
                ["tail"] = new NSLBuiltinFunction("tail", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("list.tail() requires a list");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list");
                    return list.Count > 1 ? list.Skip(1).ToList() : new List<object?>();
                }),
                ["init"] = new NSLBuiltinFunction("init", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("list.init() requires a list");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be a list");
                    return list.Count > 1 ? list.Take(list.Count - 1).ToList() : new List<object?>();
                }),
                ["nth"] = new NSLBuiltinFunction("nth", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.nth() requires list and index");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var n = (int)ConvertToNumber(args[1]);
                    if (n < 0) n = list.Count + n;
                    return (n >= 0 && n < list.Count) ? list[n] : null;
                }),
                ["indexOf"] = new NSLBuiltinFunction("indexOf", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.indexOf() requires list and value");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var value = args[1]?.ToString() ?? "null";
                    for (int i = 0; i < list.Count; i++) {
                        if ((list[i]?.ToString() ?? "null") == value) return (double)i;
                    }
                    return -1.0;
                }),
                ["lastIndexOf"] = new NSLBuiltinFunction("lastIndexOf", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.lastIndexOf() requires list and value");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var value = args[1]?.ToString() ?? "null";
                    for (int i = list.Count - 1; i >= 0; i--) {
                        if ((list[i]?.ToString() ?? "null") == value) return (double)i;
                    }
                    return -1.0;
                }),
                ["count"] = new NSLBuiltinFunction("count", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("list.count() requires list and predicate/value");
                    var list = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be a list");
                    var arg2 = args[1];
                    if (arg2 is NSLFunction || arg2 is NSLBuiltinFunction) {
                        int count = 0;
                        foreach (var item in list) {
                            object? result;
                            if (arg2 is NSLFunction userFn) result = CallUserFunction(userFn, new[] { item });
                            else result = ((NSLBuiltinFunction)arg2).Call(new[] { item });
                            if (IsTruthy(result)) count++;
                        }
                        return (double)count;
                    }
                    var value = arg2?.ToString() ?? "null";
                    return (double)list.Count(x => (x?.ToString() ?? "null") == value);
                })
            };
            _globals["list"] = listNamespace;

            // ===== STRING NAMESPACE =====
            var stringNamespace = new Dictionary<string, object?>
            {
                ["length"] = new NSLBuiltinFunction("length", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.length() requires a string argument"); return (double)(args[0]?.ToString()?.Length ?? 0); }),
                ["upper"] = new NSLBuiltinFunction("upper", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.upper() requires a string argument"); return args[0]?.ToString()?.ToUpper() ?? ""; }),
                ["lower"] = new NSLBuiltinFunction("lower", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.lower() requires a string argument"); return args[0]?.ToString()?.ToLower() ?? ""; }),
                ["trim"] = new NSLBuiltinFunction("trim", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.trim() requires a string argument"); return args[0]?.ToString()?.Trim() ?? ""; }),
                ["split"] = new NSLBuiltinFunction("split", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.split() requires a string argument"); var str = args[0]?.ToString() ?? ""; var separator = args.Length > 1 ? args[1]?.ToString() ?? " " : " "; return str.Split(separator).Select(s => (object?)s).ToList(); }),
                ["contains"] = new NSLBuiltinFunction("contains", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.contains() requires string and substring"); return (args[0]?.ToString() ?? "").Contains(args[1]?.ToString() ?? ""); }),
                ["startsWith"] = new NSLBuiltinFunction("startsWith", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.startsWith() requires string and prefix"); return (args[0]?.ToString() ?? "").StartsWith(args[1]?.ToString() ?? ""); }),
                ["endsWith"] = new NSLBuiltinFunction("endsWith", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.endsWith() requires string and suffix"); return (args[0]?.ToString() ?? "").EndsWith(args[1]?.ToString() ?? ""); }),
                ["replace"] = new NSLBuiltinFunction("replace", (args) => { if (args.Length < 3) throw new NSLRuntimeException("string.replace() requires string, old, new"); return (args[0]?.ToString() ?? "").Replace(args[1]?.ToString() ?? "", args[2]?.ToString() ?? ""); }),
                ["indexOf"] = new NSLBuiltinFunction("indexOf", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.indexOf() requires string and substring"); return (double)(args[0]?.ToString() ?? "").IndexOf(args[1]?.ToString() ?? ""); }),
                ["substring"] = new NSLBuiltinFunction("substring", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.substring() requires string and start index"); var str = args[0]?.ToString() ?? ""; var start = (int)ConvertToNumber(args[1]); return args.Length > 2 ? str.Substring(start, Math.Min((int)ConvertToNumber(args[2]), str.Length - start)) : str.Substring(start); }),
                ["repeat"] = new NSLBuiltinFunction("repeat", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.repeat() requires string and count"); return string.Concat(Enumerable.Repeat(args[0]?.ToString() ?? "", (int)ConvertToNumber(args[1]))); }),
                ["reverse"] = new NSLBuiltinFunction("reverse", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.reverse() requires a string argument"); return new string((args[0]?.ToString() ?? "").Reverse().ToArray()); }),
                ["padLeft"] = new NSLBuiltinFunction("padLeft", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.padLeft() requires string and width"); return (args[0]?.ToString() ?? "").PadLeft((int)ConvertToNumber(args[1]), args.Length > 2 ? (args[2]?.ToString() ?? " ")[0] : ' '); }),
                ["padRight"] = new NSLBuiltinFunction("padRight", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.padRight() requires string and width"); return (args[0]?.ToString() ?? "").PadRight((int)ConvertToNumber(args[1]), args.Length > 2 ? (args[2]?.ToString() ?? " ")[0] : ' '); }),
                // Lowercase aliases for consistency
                ["startswith"] = new NSLBuiltinFunction("startswith", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.startswith() requires string and prefix"); return (args[0]?.ToString() ?? "").StartsWith(args[1]?.ToString() ?? ""); }),
                ["endswith"] = new NSLBuiltinFunction("endswith", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.endswith() requires string and suffix"); return (args[0]?.ToString() ?? "").EndsWith(args[1]?.ToString() ?? ""); }),
                ["join"] = new NSLBuiltinFunction("join", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.join() requires array and separator"); var arr = args[0] as IList<object> ?? new List<object>(); var sep = args[1]?.ToString() ?? ""; return string.Join(sep, arr.Select(x => x?.ToString() ?? "")); }),
                ["format"] = new NSLBuiltinFunction("format", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.format() requires a format string"); var fmt = args[0]?.ToString() ?? ""; var fmtArgs = args.Skip(1).Select(a => a ?? "").ToArray(); try { return string.Format(fmt, fmtArgs); } catch { return fmt; } }),
                ["chars"] = new NSLBuiltinFunction("chars", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.chars() requires a string"); return (args[0]?.ToString() ?? "").Select(c => (object)c.ToString()).ToList(); }),
                ["code"] = new NSLBuiltinFunction("code", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.code() requires a character"); var s = args[0]?.ToString() ?? ""; return s.Length > 0 ? (double)s[0] : 0.0; }),
                ["fromCode"] = new NSLBuiltinFunction("fromCode", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.fromCode() requires a code"); return ((char)(int)ConvertToNumber(args[0])).ToString(); }),
                ["isEmpty"] = new NSLBuiltinFunction("isEmpty", (args) => { if (args.Length < 1) return true; return string.IsNullOrEmpty(args[0]?.ToString()); }),
                ["isBlank"] = new NSLBuiltinFunction("isBlank", (args) => { if (args.Length < 1) return true; return string.IsNullOrWhiteSpace(args[0]?.ToString()); }),
                ["charAt"] = new NSLBuiltinFunction("charAt", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.charAt() requires string and index"); var s = args[0]?.ToString() ?? ""; var idx = (int)ConvertToNumber(args[1]); return idx >= 0 && idx < s.Length ? s[idx].ToString() : ""; }),
                ["charCodeAt"] = new NSLBuiltinFunction("charCodeAt", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.charCodeAt() requires string and index"); var s = args[0]?.ToString() ?? ""; var idx = (int)ConvertToNumber(args[1]); return idx >= 0 && idx < s.Length ? (double)s[idx] : -1.0; }),
                ["slice"] = new NSLBuiltinFunction("slice", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.slice() requires string and start"); var s = args[0]?.ToString() ?? ""; var start = (int)ConvertToNumber(args[1]); var end = args.Length > 2 ? (int)ConvertToNumber(args[2]) : s.Length; if (start < 0) start = Math.Max(0, s.Length + start); if (end < 0) end = Math.Max(0, s.Length + end); return s.Substring(start, Math.Max(0, Math.Min(end, s.Length) - start)); }),
                ["lines"] = new NSLBuiltinFunction("lines", (args) => { if (args.Length < 1) throw new NSLRuntimeException("string.lines() requires a string"); return (args[0]?.ToString() ?? "").Split('\n').Select(l => (object?)l.TrimEnd('\r')).ToList(); }),
                ["matchAll"] = new NSLBuiltinFunction("matchAll", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.matchAll() requires string and pattern"); var s = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; var matches = System.Text.RegularExpressions.Regex.Matches(s, pattern); return matches.Select(m => (object?)m.Value).ToList(); }),
                // Lowercase aliases for common functions
                ["index"] = new NSLBuiltinFunction("index", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.index() requires string and substring"); return (double)(args[0]?.ToString() ?? "").IndexOf(args[1]?.ToString() ?? ""); }),
                ["lastindex"] = new NSLBuiltinFunction("lastindex", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.lastindex() requires string and substring"); return (double)(args[0]?.ToString() ?? "").LastIndexOf(args[1]?.ToString() ?? ""); }),
                ["find"] = new NSLBuiltinFunction("find", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.find() requires string and substring"); return (double)(args[0]?.ToString() ?? "").IndexOf(args[1]?.ToString() ?? ""); }),
                ["match"] = new NSLBuiltinFunction("match", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.match() requires string and pattern"); var s = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; var m = System.Text.RegularExpressions.Regex.Match(s, pattern); return m.Success ? m.Value : null; }),
                ["isMatch"] = new NSLBuiltinFunction("isMatch", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.isMatch() requires string and pattern"); return System.Text.RegularExpressions.Regex.IsMatch(args[0]?.ToString() ?? "", args[1]?.ToString() ?? ""); }),
                ["ismatch"] = new NSLBuiltinFunction("ismatch", (args) => { if (args.Length < 2) throw new NSLRuntimeException("string.ismatch() requires string and pattern"); return System.Text.RegularExpressions.Regex.IsMatch(args[0]?.ToString() ?? "", args[1]?.ToString() ?? ""); }),
                ["toString"] = new NSLBuiltinFunction("toString", (args) => { 
                    if (args.Length < 1) return "";
                    var val = args[0];
                    if (val == null) return "null";
                    if (val is string s) return s;
                    if (val is bool b) return b ? "true" : "false";
                    if (val is double d) return d.ToString();
                    if (val is IDictionary<string, object?> dict) return System.Text.Json.JsonSerializer.Serialize(dict);
                    if (val is IList<object?> list) return System.Text.Json.JsonSerializer.Serialize(list);
                    return val.ToString() ?? "";
                }),
                // Case conversion functions
                ["capitalize"] = new NSLBuiltinFunction("capitalize", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    return s.Length > 0 ? char.ToUpper(s[0]) + s.Substring(1).ToLower() : "";
                }),
                ["title"] = new NSLBuiltinFunction("title", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    return System.Globalization.CultureInfo.CurrentCulture.TextInfo.ToTitleCase(s.ToLower());
                }),
                ["camelCase"] = new NSLBuiltinFunction("camelCase", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    var words = System.Text.RegularExpressions.Regex.Split(s, @"[\s_\-]+").Where(w => w.Length > 0).ToArray();
                    if (words.Length == 0) return "";
                    return words[0].ToLower() + string.Concat(words.Skip(1).Select(w => char.ToUpper(w[0]) + w.Substring(1).ToLower()));
                }),
                ["pascalCase"] = new NSLBuiltinFunction("pascalCase", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    var words = System.Text.RegularExpressions.Regex.Split(s, @"[\s_\-]+").Where(w => w.Length > 0).ToArray();
                    return string.Concat(words.Select(w => char.ToUpper(w[0]) + w.Substring(1).ToLower()));
                }),
                ["snakeCase"] = new NSLBuiltinFunction("snakeCase", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    s = System.Text.RegularExpressions.Regex.Replace(s, @"([a-z])([A-Z])", "$1_$2");
                    return System.Text.RegularExpressions.Regex.Replace(s, @"[\s\-]+", "_").ToLower();
                }),
                ["kebabCase"] = new NSLBuiltinFunction("kebabCase", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    s = System.Text.RegularExpressions.Regex.Replace(s, @"([a-z])([A-Z])", "$1-$2");
                    return System.Text.RegularExpressions.Regex.Replace(s, @"[\s_]+", "-").ToLower();
                }),
                ["constantCase"] = new NSLBuiltinFunction("constantCase", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    s = System.Text.RegularExpressions.Regex.Replace(s, @"([a-z])([A-Z])", "$1_$2");
                    return System.Text.RegularExpressions.Regex.Replace(s, @"[\s\-]+", "_").ToUpper();
                }),
                // Additional string utilities
                ["reverse"] = new NSLBuiltinFunction("reverse", (args) => {
                    if (args.Length < 1) return "";
                    var s = args[0]?.ToString() ?? "";
                    return new string(s.Reverse().ToArray());
                }),
                ["repeat"] = new NSLBuiltinFunction("repeat", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("string.repeat() requires string and count");
                    var s = args[0]?.ToString() ?? "";
                    var count = (int)ConvertToNumber(args[1]);
                    return string.Concat(Enumerable.Repeat(s, count));
                }),
                ["padLeft"] = new NSLBuiltinFunction("padLeft", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("string.padLeft() requires string and width");
                    var s = args[0]?.ToString() ?? "";
                    var width = (int)ConvertToNumber(args[1]);
                    var padChar = args.Length > 2 ? (args[2]?.ToString() ?? " ")[0] : ' ';
                    return s.PadLeft(width, padChar);
                }),
                ["padRight"] = new NSLBuiltinFunction("padRight", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("string.padRight() requires string and width");
                    var s = args[0]?.ToString() ?? "";
                    var width = (int)ConvertToNumber(args[1]);
                    var padChar = args.Length > 2 ? (args[2]?.ToString() ?? " ")[0] : ' ';
                    return s.PadRight(width, padChar);
                }),
                ["truncate"] = new NSLBuiltinFunction("truncate", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("string.truncate() requires string and length");
                    var s = args[0]?.ToString() ?? "";
                    var length = (int)ConvertToNumber(args[1]);
                    var suffix = args.Length > 2 ? args[2]?.ToString() ?? "..." : "...";
                    return s.Length <= length ? s : s.Substring(0, length - suffix.Length) + suffix;
                }),
                ["wordCount"] = new NSLBuiltinFunction("wordCount", (args) => {
                    if (args.Length < 1) return 0.0;
                    var s = args[0]?.ToString() ?? "";
                    return (double)s.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;
                }),
                ["isUpperCase"] = new NSLBuiltinFunction("isUpperCase", (args) => {
                    if (args.Length < 1) return false;
                    var s = args[0]?.ToString() ?? "";
                    return s.Length > 0 && s == s.ToUpper();
                }),
                ["isLowerCase"] = new NSLBuiltinFunction("isLowerCase", (args) => {
                    if (args.Length < 1) return false;
                    var s = args[0]?.ToString() ?? "";
                    return s.Length > 0 && s == s.ToLower();
                }),
                ["countOccurrences"] = new NSLBuiltinFunction("countOccurrences", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("string.countOccurrences() requires string and substring");
                    var s = args[0]?.ToString() ?? "";
                    var sub = args[1]?.ToString() ?? "";
                    if (string.IsNullOrEmpty(sub)) return 0.0;
                    int count = 0, idx = 0;
                    while ((idx = s.IndexOf(sub, idx)) != -1) { count++; idx += sub.Length; }
                    return (double)count;
                }),
                ["wrap"] = new NSLBuiltinFunction("wrap", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("string.wrap() requires string and width");
                    var s = args[0]?.ToString() ?? "";
                    var width = (int)ConvertToNumber(args[1]);
                    var words = s.Split(' ');
                    var lines = new List<string>();
                    var currentLine = "";
                    foreach (var word in words) {
                        if (currentLine.Length + word.Length + 1 > width) {
                            if (currentLine.Length > 0) lines.Add(currentLine);
                            currentLine = word;
                        } else {
                            currentLine = currentLine.Length > 0 ? currentLine + " " + word : word;
                        }
                    }
                    if (currentLine.Length > 0) lines.Add(currentLine);
                    return string.Join("\n", lines);
                })
            };
            _globals["string"] = stringNamespace;

            // ===== FILE NAMESPACE =====
            var fileNamespace = new Dictionary<string, object?>
            {
                ["read"] = new NSLBuiltinFunction("read", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.read() requires a path argument"); var path = args[0]?.ToString() ?? ""; if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}"); return System.IO.File.ReadAllText(path); }),
                ["lines"] = new NSLBuiltinFunction("lines", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.lines() requires a path argument");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    return System.IO.File.ReadAllLines(path).Select(l => (object?)l).ToList();
                }),
                ["readLines"] = new NSLBuiltinFunction("readLines", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("file.readLines() requires path and start line");
                    var path = args[0]?.ToString() ?? "";
                    var start = (int)ConvertToNumber(args[1]);
                    var count = args.Length > 2 ? (int)ConvertToNumber(args[2]) : 10;
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path);
                    start = Math.Max(1, start) - 1; // Convert to 0-indexed
                    count = Math.Min(count, lines.Length - start);
                    return lines.Skip(start).Take(count).Select(l => (object?)l).ToList();
                }),
                ["size"] = new NSLBuiltinFunction("size", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.size() requires a path argument");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    return (double)new System.IO.FileInfo(path).Length;
                }),
                ["lineCount"] = new NSLBuiltinFunction("lineCount", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.lineCount() requires a path argument");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    return (double)System.IO.File.ReadLines(path).Count();
                }),
                ["write"] = new NSLBuiltinFunction("write", (args) => { if (args.Length < 2) throw new NSLRuntimeException("file.write() requires path and content"); var path = args[0]?.ToString() ?? ""; var content = args[1]?.ToString() ?? ""; NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, content, "write"); return true; }),
                ["replace"] = new NSLBuiltinFunction("replace", (args) => { 
                    if (args.Length < 3) throw new NSLRuntimeException("file.replace() requires path, oldText, newText"); 
                    var path = args[0]?.ToString() ?? ""; 
                    var oldText = args[1]?.ToString() ?? ""; 
                    var newText = args[2]?.ToString() ?? ""; 
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}"); 
                    var content = System.IO.File.ReadAllText(path); 
                    // Smart line-ending handling: normalize then replace
                    var normalizedContent = content.Replace("\r\n", "\n");
                    var normalizedOld = oldText.Replace("\r\n", "\n");
                    var normalizedNew = newText.Replace("\r\n", "\n");
                    if (!normalizedContent.Contains(normalizedOld)) return new Dictionary<string, object?> { ["success"] = false, ["reason"] = "Pattern not found" };
                    var result = normalizedContent.Replace(normalizedOld, normalizedNew);
                    // Restore original line endings
                    if (content.Contains("\r\n")) result = result.Replace("\n", "\r\n");
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "replace");
                    return new Dictionary<string, object?> { ["success"] = true, ["path"] = path };
                }),
                ["replaceRegex"] = new NSLBuiltinFunction("replaceRegex", (args) => { 
                    if (args.Length < 3) throw new NSLRuntimeException("file.replaceRegex() requires path, pattern, replacement"); 
                    var path = args[0]?.ToString() ?? ""; 
                    var pattern = args[1]?.ToString() ?? ""; 
                    var replacement = args[2]?.ToString() ?? ""; 
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}"); 
                    var content = System.IO.File.ReadAllText(path); 
                    var result = System.Text.RegularExpressions.Regex.Replace(content, pattern, replacement);
                    if (content == result) return new Dictionary<string, object?> { ["success"] = false, ["reason"] = "No matches found" };
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "replaceRegex");
                    return new Dictionary<string, object?> { ["success"] = true, ["path"] = path };
                }),
                ["append"] = new NSLBuiltinFunction("append", (args) => { if (args.Length < 2) throw new NSLRuntimeException("file.append() requires path and content"); var path = args[0]?.ToString() ?? ""; NSL.StandardLib.FileSystem.FileHistory.Instance.SavePreEditState(path, "append"); System.IO.File.AppendAllText(path, args[1]?.ToString() ?? ""); return true; }),
                ["exists"] = new NSLBuiltinFunction("exists", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.exists() requires a path argument"); return System.IO.File.Exists(args[0]?.ToString() ?? ""); }),
                ["isBinary"] = new NSLBuiltinFunction("isBinary", (args) => { 
                    if (args.Length < 1) throw new NSLRuntimeException("file.isBinary() requires a path"); 
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) return false;
                    // Check extension first (fast path)
                    var textExts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".cs", ".nsl", ".txt", ".md", ".json", ".xml", ".yaml", ".yml", ".js", ".ts", ".py", ".html", ".css", ".sh", ".bat", ".ps1", ".sql", ".config", ".csproj", ".sln" };
                    var ext = System.IO.Path.GetExtension(path);
                    if (textExts.Contains(ext)) return false;
                    // Check for null bytes in first 8KB (binary indicator)
                    try { 
                        var buffer = new byte[8192];
                        using var fs = System.IO.File.OpenRead(path);
                        var bytesRead = fs.Read(buffer, 0, buffer.Length);
                        for (int i = 0; i < bytesRead; i++) if (buffer[i] == 0) return true;
                        return false;
                    } catch { return true; }
                }),
                ["delete"] = new NSLBuiltinFunction("delete", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.delete() requires a path argument"); var path = args[0]?.ToString() ?? ""; NSL.StandardLib.FileSystem.FileHistory.Instance.SavePreEditState(path, "delete"); if (System.IO.File.Exists(path)) { System.IO.File.Delete(path); return true; } return false; }),
                ["list"] = new NSLBuiltinFunction("list", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.list() requires a directory path");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args.Length > 1 ? args[1]?.ToString() ?? "*" : "*";
                    if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}");
                    var files = System.IO.Directory.GetFiles(path, pattern);
                    return files.Select(f => (object?)f).ToList();
                }),
                ["listDirs"] = new NSLBuiltinFunction("listDirs", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.listDirs() requires a directory path");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args.Length > 1 ? args[1]?.ToString() ?? "*" : "*";
                    if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}");
                    var dirs = System.IO.Directory.GetDirectories(path, pattern);
                    return dirs.Select(d => (object?)d).ToList();
                }),
                ["listAll"] = new NSLBuiltinFunction("listAll", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.listAll() requires a directory path");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args.Length > 1 ? args[1]?.ToString() ?? "*" : "*";
                    var recursive = args.Length > 2 && args[2] is bool b && b;
                    if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}");
                    var searchOption = recursive ? System.IO.SearchOption.AllDirectories : System.IO.SearchOption.TopDirectoryOnly;
                    var entries = System.IO.Directory.GetFileSystemEntries(path, pattern, searchOption);
                    return entries.Select(e => (object?)e).ToList();
                }),
                ["info"] = new NSLBuiltinFunction("info", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.info() requires a path");
                    var path = args[0]?.ToString() ?? "";
                    if (System.IO.File.Exists(path)) {
                        var fi = new System.IO.FileInfo(path);
                        return new Dictionary<string, object?> {
                            ["name"] = fi.Name,
                            ["path"] = fi.FullName,
                            ["size"] = (double)fi.Length,
                            ["created"] = fi.CreationTime.ToString("o"),
                            ["modified"] = fi.LastWriteTime.ToString("o"),
                            ["isFile"] = true,
                            ["isDir"] = false,
                            ["extension"] = fi.Extension
                        };
                    } else if (System.IO.Directory.Exists(path)) {
                        var di = new System.IO.DirectoryInfo(path);
                        return new Dictionary<string, object?> {
                            ["name"] = di.Name,
                            ["path"] = di.FullName,
                            ["size"] = 0.0,
                            ["created"] = di.CreationTime.ToString("o"),
                            ["modified"] = di.LastWriteTime.ToString("o"),
                            ["isFile"] = false,
                            ["isDir"] = true,
                            ["extension"] = ""
                        };
                    }
                    throw new NSLRuntimeException($"Path not found: {path}");
                }),
                // Batch processing: insert text before lines matching a pattern
                ["insertBefore"] = new NSLBuiltinFunction("insertBefore", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("file.insertBefore(path, pattern, textToInsert, [options]) - options: {skipIfPrevHas, useRegex}");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var textToInsert = args[2]?.ToString() ?? "";
                    var options = args.Length > 3 ? args[3] as Dictionary<string, object?> : null;
                    var skipIfPrevHas = options?.GetValueOrDefault("skipIfPrevHas")?.ToString();
                    var useRegex = options?.GetValueOrDefault("useRegex") is bool b && b;
                    var preserveIndent = options?.GetValueOrDefault("preserveIndent") is not bool pi || pi; // default true
                    
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    var result = new List<string>();
                    int inserted = 0;
                    
                    for (int i = 0; i < lines.Count; i++) {
                        var line = lines[i];
                        var trimmed = line.TrimStart();
                        bool matches = useRegex ? System.Text.RegularExpressions.Regex.IsMatch(trimmed, pattern) : trimmed.StartsWith(pattern);
                        
                        if (matches) {
                            // Check skip conditions
                            bool shouldSkip = false;
                            if (result.Count > 0) {
                                var prevTrimmed = result[result.Count - 1].TrimStart();
                                if (!string.IsNullOrEmpty(skipIfPrevHas) && prevTrimmed.StartsWith(skipIfPrevHas))
                                    shouldSkip = true;
                                // Skip if prev line is an attribute [...]
                                if (prevTrimmed.StartsWith("[") && prevTrimmed.EndsWith("]"))
                                    shouldSkip = true;
                                if (prevTrimmed.EndsWith(")]"))
                                    shouldSkip = true;
                            }
                            
                            if (!shouldSkip) {
                                // Get indent from current line
                                var indent = preserveIndent ? line.Substring(0, line.Length - trimmed.Length) : "";
                                result.Add(indent + textToInsert);
                                inserted++;
                            }
                        }
                        result.Add(line);
                    }
                    
                    if (inserted > 0) {
                        NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "insertBefore");
                    }
                    return new Dictionary<string, object?> { ["inserted"] = (double)inserted, ["path"] = path };
                }),
                // Batch process multiple files with insertBefore
                ["batchInsertBefore"] = new NSLBuiltinFunction("batchInsertBefore", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("file.batchInsertBefore(files[], pattern, textToInsert, [options])");
                    var files = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be array of file paths");
                    var pattern = args[1]?.ToString() ?? "";
                    var textToInsert = args[2]?.ToString() ?? "";
                    var options = args.Length > 3 ? args[3] as Dictionary<string, object?> : null;
                    var skipIfPrevHas = options?.GetValueOrDefault("skipIfPrevHas")?.ToString();
                    var useRegex = options?.GetValueOrDefault("useRegex") is bool b && b;
                    var preserveIndent = options?.GetValueOrDefault("preserveIndent") is not bool pi || pi;
                    
                    int totalInserted = 0;
                    int filesModified = 0;
                    var results = new List<object?>();
                    
                    foreach (var fileObj in files) {
                        var path = fileObj?.ToString() ?? "";
                        if (!System.IO.File.Exists(path)) continue;
                        
                        var lines = System.IO.File.ReadAllLines(path).ToList();
                        var result = new List<string>();
                        int inserted = 0;
                        
                        for (int i = 0; i < lines.Count; i++) {
                            var line = lines[i];
                            var trimmed = line.TrimStart();
                            bool matches = useRegex ? System.Text.RegularExpressions.Regex.IsMatch(trimmed, pattern) : trimmed.StartsWith(pattern);
                            
                            if (matches) {
                                bool shouldSkip = false;
                                if (result.Count > 0) {
                                    var prevTrimmed = result[result.Count - 1].TrimStart();
                                    // Skip if prev line matches skipIfPrevHas pattern
                                    if (!string.IsNullOrEmpty(skipIfPrevHas) && prevTrimmed.StartsWith(skipIfPrevHas))
                                        shouldSkip = true;
                                    // Also skip if prev line is an attribute [...]
                                    if (prevTrimmed.StartsWith("[") && prevTrimmed.EndsWith("]"))
                                        shouldSkip = true;
                                    // Skip if prev line ends with attribute closing
                                    if (prevTrimmed.EndsWith(")]"))
                                        shouldSkip = true;
                                }
                                
                                if (!shouldSkip) {
                                    var indent = preserveIndent ? line.Substring(0, line.Length - trimmed.Length) : "";
                                    result.Add(indent + textToInsert);
                                    inserted++;
                                }
                            }
                            result.Add(line);
                        }
                        
                        if (inserted > 0) {
                            NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "batchInsertBefore");
                            totalInserted += inserted;
                            filesModified++;
                            results.Add(new Dictionary<string, object?> { ["path"] = path, ["inserted"] = (double)inserted });
                        }
                    }
                    
                    return new Dictionary<string, object?> { 
                        ["totalInserted"] = (double)totalInserted, 
                        ["filesModified"] = (double)filesModified,
                        ["results"] = results 
                    };
                }),
                // C#-aware XML comment insertion - ONLY for public declarations, inserts before attributes
                ["csAddXmlComments"] = new NSLBuiltinFunction("csAddXmlComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.csAddXmlComments(path, [options]) - options: {dryRun, commentText}");
                    var path = args[0]?.ToString() ?? "";
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    var commentText = options?.GetValueOrDefault("commentText")?.ToString() ?? "/// <summary>API member</summary>";
                    
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    var result = new List<string>();
                    int inserted = 0;
                    
                    for (int i = 0; i < lines.Count; i++) {
                        var line = lines[i];
                        var trimmed = line.TrimStart();
                        
                        // ONLY add comments to lines starting with "public " - simple and reliable
                        bool needsComment = trimmed.StartsWith("public ");
                        
                        // Check if already has comment - look back through attributes and blank lines
                        if (needsComment && result.Count > 0) {
                            int checkIdx = result.Count - 1;
                            while (checkIdx >= 0) {
                                var prevTrim = result[checkIdx].TrimStart();
                                if (string.IsNullOrWhiteSpace(prevTrim)) { checkIdx--; continue; }
                                if (prevTrim.StartsWith("[")) { checkIdx--; continue; }
                                if (prevTrim.StartsWith("///")) { needsComment = false; break; }
                                break;
                            }
                        }
                        
                        if (needsComment) {
                            var indent = line.Substring(0, line.Length - trimmed.Length);
                            // Insert before any attributes
                            int insertPos = result.Count;
                            while (insertPos > 0) {
                                var prevTrim = result[insertPos - 1].TrimStart();
                                if (prevTrim.StartsWith("[") && (prevTrim.EndsWith("]") || prevTrim.EndsWith(")]"))) {
                                    insertPos--;
                                } else break;
                            }
                            result.Insert(insertPos, indent + commentText);
                            inserted++;
                        }
                        result.Add(line);
                    }
                    
                    if (!dryRun && inserted > 0) {
                        NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "csAddXmlComments");
                    }
                    return new Dictionary<string, object?> { ["inserted"] = (double)inserted, ["path"] = path, ["dryRun"] = dryRun };
                }),
                // Batch version - ONLY public declarations, inserts before attributes
                ["csBatchAddXmlComments"] = new NSLBuiltinFunction("csBatchAddXmlComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.csBatchAddXmlComments(files[], [options])");
                    var files = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be array of file paths");
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    var commentText = options?.GetValueOrDefault("commentText")?.ToString() ?? "/// <summary>API member</summary>";
                    
                    int totalInserted = 0, filesModified = 0;
                    var fileResults = new List<object?>();
                    
                    foreach (var fileObj in files) {
                        var path = fileObj?.ToString() ?? "";
                        if (!System.IO.File.Exists(path)) continue;
                        
                        var lines = System.IO.File.ReadAllLines(path).ToList();
                        var result = new List<string>();
                        int inserted = 0;
                        
                        for (int i = 0; i < lines.Count; i++) {
                            var line = lines[i];
                            var trimmed = line.TrimStart();
                            
                            // ONLY public declarations
                            bool needsComment = trimmed.StartsWith("public ");
                            
                            if (needsComment && result.Count > 0) {
                                int checkIdx = result.Count - 1;
                                while (checkIdx >= 0) {
                                    var prevTrim = result[checkIdx].TrimStart();
                                    if (string.IsNullOrWhiteSpace(prevTrim)) { checkIdx--; continue; }
                                    if (prevTrim.StartsWith("[")) { checkIdx--; continue; }
                                    if (prevTrim.StartsWith("///")) { needsComment = false; break; }
                                    break;
                                }
                            }
                            
                            if (needsComment) {
                                var indent = line.Substring(0, line.Length - trimmed.Length);
                                int insertPos = result.Count;
                                while (insertPos > 0) {
                                    var prev = result[insertPos - 1].TrimStart();
                                    if (prev.StartsWith("[") && (prev.EndsWith("]") || prev.EndsWith(")]"))) insertPos--;
                                    else break;
                                }
                                result.Insert(insertPos, indent + commentText);
                                inserted++;
                            }
                            result.Add(line);
                        }
                        
                        if (inserted > 0) {
                            if (!dryRun) NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "csBatchAddXmlComments");
                            totalInserted += inserted;
                            filesModified++;
                            fileResults.Add(new Dictionary<string, object?> { ["path"] = path, ["inserted"] = (double)inserted });
                        }
                    }
                    
                    return new Dictionary<string, object?> {
                        ["totalInserted"] = (double)totalInserted,
                        ["filesModified"] = (double)filesModified,
                        ["dryRun"] = dryRun,
                        ["results"] = fileResults
                    };
                }),
                // Remove misplaced XML comments (CS1587 fix) - works on ANY text file
                ["removeMisplacedXmlComments"] = new NSLBuiltinFunction("removeMisplacedXmlComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.removeMisplacedXmlComments(path, [options])");
                    var path = args[0]?.ToString() ?? "";
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    var result = new List<string>();
                    int removed = 0;
                    
                    // Valid targets after XML comments
                    var validTargets = new[] { "public ", "protected ", "private ", "internal ", "static ", "///", "[",
                        "class ", "interface ", "enum ", "struct ", "record ", "abstract ", "sealed ", "virtual ",
                        "override ", "async ", "readonly ", "const ", "new ", "extern ", "partial ", "event ", "delegate " };
                    
                    for (int i = 0; i < lines.Count; i++) {
                        var trimmed = lines[i].TrimStart();
                        if (trimmed.StartsWith("/// <summary>") && i + 1 < lines.Count) {
                            var nextTrimmed = lines[i + 1].TrimStart();
                            bool valid = validTargets.Any(t => nextTrimmed.StartsWith(t));
                            if (!valid) { removed++; continue; }
                        }
                        result.Add(lines[i]);
                    }
                    
                    if (!dryRun && removed > 0) {
                        NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "removeMisplacedXmlComments");
                    }
                    return new Dictionary<string, object?> { ["removed"] = (double)removed, ["path"] = path, ["dryRun"] = dryRun };
                }),
                // Batch remove misplaced comments
                ["batchRemoveMisplacedXmlComments"] = new NSLBuiltinFunction("batchRemoveMisplacedXmlComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.batchRemoveMisplacedXmlComments(files[], [options])");
                    var files = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be array of file paths");
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    
                    int totalRemoved = 0, filesModified = 0;
                    var fileResults = new List<object?>();
                    
                    foreach (var fileObj in files) {
                        var path = fileObj?.ToString() ?? "";
                        if (!System.IO.File.Exists(path)) continue;
                        
                        var lines = System.IO.File.ReadAllLines(path).ToList();
                        var result = new List<string>();
                        int removed = 0;
                        
                        var validTargets = new[] { "public ", "protected ", "private ", "internal ", "static ", "///", "[",
                            "class ", "interface ", "enum ", "struct ", "record ", "abstract ", "sealed ", "virtual ",
                            "override ", "async ", "readonly ", "const ", "new ", "extern ", "partial ", "event ", "delegate " };
                        
                        for (int i = 0; i < lines.Count; i++) {
                            var trimmed = lines[i].TrimStart();
                            if (trimmed.StartsWith("/// <summary>") && i + 1 < lines.Count) {
                                var nextTrimmed = lines[i + 1].TrimStart();
                                bool valid = validTargets.Any(t => nextTrimmed.StartsWith(t));
                                if (!valid) { removed++; continue; }
                            }
                            result.Add(lines[i]);
                        }
                        
                        if (removed > 0) {
                            if (!dryRun) NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "batchRemoveMisplacedXmlComments");
                            totalRemoved += removed;
                            filesModified++;
                            fileResults.Add(new Dictionary<string, object?> { ["path"] = path, ["removed"] = (double)removed });
                        }
                    }
                    
                    return new Dictionary<string, object?> {
                        ["totalRemoved"] = (double)totalRemoved,
                        ["filesModified"] = (double)filesModified,
                        ["dryRun"] = dryRun,
                        ["results"] = fileResults
                    };
                }),
                // Add XML comments to enum members (CS1591 fix for enums)
                // Uses block type stack to avoid inserting in dictionary initializers
                ["csEnumComments"] = new NSLBuiltinFunction("csEnumComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.csEnumComments(path, [options])");
                    var path = args[0]?.ToString() ?? "";
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    var commentText = options?.GetValueOrDefault("commentText")?.ToString() ?? "/// <summary>Enum member.</summary>";
                    
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    var result = new List<string>();
                    int inserted = 0;
                    
                    // Block type stack: "enum", "class", "method", "dict", "other"
                    var blockStack = new Stack<string>();
                    bool pendingEnum = false;
                    
                    for (int i = 0; i < lines.Count; i++) {
                        var line = lines[i];
                        var trimmed = line.TrimStart();
                        
                        // Detect enum declaration (marks next { as enum block)
                        if (EnumDeclRegex.IsMatch(trimmed)) {
                            pendingEnum = true;
                        }
                        // Detect dictionary/object initializer: "new Dictionary" or "new {" or "= {"
                        else if (DictInitRegex.IsMatch(trimmed)) {
                            // Next { is a dictionary/initializer, not enum
                            if (trimmed.Contains("{")) blockStack.Push("dict");
                        }
                        
                        // Track opening braces
                        foreach (char c in line) {
                            if (c == '{') {
                                if (pendingEnum) {
                                    blockStack.Push("enum");
                                    pendingEnum = false;
                                } else if (blockStack.Count == 0 || blockStack.Peek() != "dict") {
                                    // Don't push if we already detected dict on this line
                                    if (!DictInitSimpleRegex.IsMatch(trimmed))
                                        blockStack.Push("other");
                                }
                            } else if (c == '}') {
                                if (blockStack.Count > 0) blockStack.Pop();
                            }
                        }
                        
                        // Insert comment if we're directly inside an enum block (depth 1 in enum)
                        bool inEnumDirectly = blockStack.Count > 0 && blockStack.Peek() == "enum";
                        
                        if (inEnumDirectly && !trimmed.StartsWith("///") && !trimmed.StartsWith("//") && 
                            !trimmed.StartsWith("{") && !trimmed.StartsWith("}") && trimmed.Length > 0) {
                            // Enum member pattern: Name, or Name = value (no parentheses, no ["key"])
                            if (EnumMemberRegex.IsMatch(trimmed)) {
                                if (i == 0 || !lines[i - 1].TrimStart().StartsWith("///")) {
                                    var indent = line.Substring(0, line.Length - line.TrimStart().Length);
                                    result.Add(indent + commentText);
                                    inserted++;
                                }
                            }
                        }
                        result.Add(line);
                    }
                    
                    if (!dryRun && inserted > 0) {
                        NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "csEnumComments");
                    }
                    return new Dictionary<string, object?> { ["inserted"] = (double)inserted, ["path"] = path, ["dryRun"] = dryRun };
                }),
                // Batch add XML comments to enum members (uses block type stack)
                ["csBatchEnumComments"] = new NSLBuiltinFunction("csBatchEnumComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.csBatchEnumComments(files[], [options])");
                    var files = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be array of file paths");
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    var commentText = options?.GetValueOrDefault("commentText")?.ToString() ?? "/// <summary>Enum member.</summary>";
                    
                    int totalInserted = 0, filesModified = 0;
                    var fileResults = new List<object?>();
                    
                    foreach (var fileObj in files) {
                        var path = fileObj?.ToString() ?? "";
                        if (!System.IO.File.Exists(path)) continue;
                        
                        var lines = System.IO.File.ReadAllLines(path).ToList();
                        var result = new List<string>();
                        int inserted = 0;
                        var blockStack = new Stack<string>();
                        bool pendingEnum = false;
                        
                        for (int i = 0; i < lines.Count; i++) {
                            var line = lines[i];
                            var trimmed = line.TrimStart();
                            
                            if (EnumDeclRegex.IsMatch(trimmed)) {
                                pendingEnum = true;
                            } else if (DictInitRegex.IsMatch(trimmed)) {
                                if (trimmed.Contains("{")) blockStack.Push("dict");
                            }
                            
                            foreach (char c in line) {
                                if (c == '{') {
                                    if (pendingEnum) { blockStack.Push("enum"); pendingEnum = false; }
                                    else if (blockStack.Count == 0 || blockStack.Peek() != "dict") {
                                        if (!DictInitSimpleRegex.IsMatch(trimmed))
                                            blockStack.Push("other");
                                    }
                                } else if (c == '}') {
                                    if (blockStack.Count > 0) blockStack.Pop();
                                }
                            }
                            
                            bool inEnumDirectly = blockStack.Count > 0 && blockStack.Peek() == "enum";
                            if (inEnumDirectly && !trimmed.StartsWith("///") && !trimmed.StartsWith("//") && 
                                !trimmed.StartsWith("{") && !trimmed.StartsWith("}") && trimmed.Length > 0) {
                                if (EnumMemberRegex.IsMatch(trimmed)) {
                                    if (i == 0 || !lines[i - 1].TrimStart().StartsWith("///")) {
                                        var indent = line.Substring(0, line.Length - line.TrimStart().Length);
                                        result.Add(indent + commentText);
                                        inserted++;
                                    }
                                }
                            }
                            result.Add(line);
                        }
                        
                        if (inserted > 0) {
                            if (!dryRun) NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "csBatchEnumComments");
                            totalInserted += inserted;
                            filesModified++;
                            fileResults.Add(new Dictionary<string, object?> { ["path"] = path, ["inserted"] = (double)inserted });
                        }
                    }
                    
                    return new Dictionary<string, object?> {
                        ["totalInserted"] = (double)totalInserted,
                        ["filesModified"] = (double)filesModified,
                        ["dryRun"] = dryRun,
                        ["results"] = fileResults
                    };
                }),
                // Add XML comments to interface members (CS1591 fix for interfaces)
                // Uses block type stack to avoid inserting in nested blocks
                ["csInterfaceComments"] = new NSLBuiltinFunction("csInterfaceComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.csInterfaceComments(path, [options])");
                    var path = args[0]?.ToString() ?? "";
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    var commentText = options?.GetValueOrDefault("commentText")?.ToString() ?? "/// <summary>Interface member.</summary>";
                    
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    var result = new List<string>();
                    int inserted = 0;
                    var blockStack = new Stack<string>();
                    bool pendingInterface = false;
                    
                    for (int i = 0; i < lines.Count; i++) {
                        var line = lines[i];
                        var trimmed = line.TrimStart();
                        
                        // Detect interface declaration
                        if (InterfaceDeclRegex.IsMatch(trimmed)) {
                            pendingInterface = true;
                        } else if (DictInitRegex.IsMatch(trimmed)) {
                            if (trimmed.Contains("{")) blockStack.Push("dict");
                        }
                        
                        foreach (char c in line) {
                            if (c == '{') {
                                if (pendingInterface) { blockStack.Push("interface"); pendingInterface = false; }
                                else if (blockStack.Count == 0 || blockStack.Peek() != "dict") {
                                    if (!DictInitSimpleRegex.IsMatch(trimmed))
                                        blockStack.Push("other");
                                }
                            } else if (c == '}') {
                                if (blockStack.Count > 0) blockStack.Pop();
                            }
                        }
                        
                        bool inInterfaceDirectly = blockStack.Count > 0 && blockStack.Peek() == "interface";
                        if (inInterfaceDirectly && !trimmed.StartsWith("///") && !trimmed.StartsWith("//") && 
                            !trimmed.StartsWith("{") && !trimmed.StartsWith("}") && !trimmed.StartsWith("[") && trimmed.Length > 0) {
                            // Interface members: methods, properties, events
                            if (InterfaceMemberRegex.IsMatch(trimmed)) {
                                if (i == 0 || !lines[i - 1].TrimStart().StartsWith("///")) {
                                    var indent = line.Substring(0, line.Length - line.TrimStart().Length);
                                    result.Add(indent + commentText);
                                    inserted++;
                                }
                            }
                        }
                        result.Add(line);
                    }
                    
                    if (!dryRun && inserted > 0) {
                        NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "csInterfaceComments");
                    }
                    return new Dictionary<string, object?> { ["inserted"] = (double)inserted, ["path"] = path, ["dryRun"] = dryRun };
                }),
                // Batch add XML comments to interface members (uses block type stack)
                ["csBatchInterfaceComments"] = new NSLBuiltinFunction("csBatchInterfaceComments", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.csBatchInterfaceComments(files[], [options])");
                    var files = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be array of file paths");
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var dryRun = options?.GetValueOrDefault("dryRun") is bool dr && dr;
                    var commentText = options?.GetValueOrDefault("commentText")?.ToString() ?? "/// <summary>Interface member.</summary>";
                    
                    int totalInserted = 0, filesModified = 0;
                    var fileResults = new List<object?>();
                    
                    foreach (var fileObj in files) {
                        var path = fileObj?.ToString() ?? "";
                        if (!System.IO.File.Exists(path)) continue;
                        
                        var lines = System.IO.File.ReadAllLines(path).ToList();
                        var result = new List<string>();
                        int inserted = 0;
                        var blockStack = new Stack<string>();
                        bool pendingInterface = false;
                        
                        for (int i = 0; i < lines.Count; i++) {
                            var line = lines[i];
                            var trimmed = line.TrimStart();
                            
                            if (InterfaceDeclRegex.IsMatch(trimmed)) {
                                pendingInterface = true;
                            } else if (DictInitRegex.IsMatch(trimmed)) {
                                if (trimmed.Contains("{")) blockStack.Push("dict");
                            }
                            
                            foreach (char c in line) {
                                if (c == '{') {
                                    if (pendingInterface) { blockStack.Push("interface"); pendingInterface = false; }
                                    else if (blockStack.Count == 0 || blockStack.Peek() != "dict") {
                                        if (!DictInitSimpleRegex.IsMatch(trimmed))
                                            blockStack.Push("other");
                                    }
                                } else if (c == '}') {
                                    if (blockStack.Count > 0) blockStack.Pop();
                                }
                            }
                            
                            bool inInterfaceDirectly = blockStack.Count > 0 && blockStack.Peek() == "interface";
                            if (inInterfaceDirectly && !trimmed.StartsWith("///") && !trimmed.StartsWith("//") && 
                                !trimmed.StartsWith("{") && !trimmed.StartsWith("}") && !trimmed.StartsWith("[") && trimmed.Length > 0) {
                                if (InterfaceMemberRegex.IsMatch(trimmed)) {
                                    if (i == 0 || !lines[i - 1].TrimStart().StartsWith("///")) {
                                        var indent = line.Substring(0, line.Length - line.TrimStart().Length);
                                        result.Add(indent + commentText);
                                        inserted++;
                                    }
                                }
                            }
                            result.Add(line);
                        }
                        
                        if (inserted > 0) {
                            if (!dryRun) NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", result), "csBatchInterfaceComments");
                            totalInserted += inserted;
                            filesModified++;
                            fileResults.Add(new Dictionary<string, object?> { ["path"] = path, ["inserted"] = (double)inserted });
                        }
                    }
                    
                    return new Dictionary<string, object?> {
                        ["totalInserted"] = (double)totalInserted,
                        ["filesModified"] = (double)filesModified,
                        ["dryRun"] = dryRun,
                        ["results"] = fileResults
                    };
                }),
                ["copy"] = new NSLBuiltinFunction("copy", (args) => { if (args.Length < 2) throw new NSLRuntimeException("file.copy() requires source and destination"); var dest = args[1]?.ToString() ?? ""; NSL.StandardLib.FileSystem.FileHistory.Instance.SavePreEditState(dest, "copy"); System.IO.File.Copy(args[0]?.ToString() ?? "", dest, true); return true; }),
                ["move"] = new NSLBuiltinFunction("move", (args) => { if (args.Length < 2) throw new NSLRuntimeException("file.move() requires source and destination"); var src = args[0]?.ToString() ?? ""; var dest = args[1]?.ToString() ?? ""; NSL.StandardLib.FileSystem.FileHistory.Instance.SavePreEditState(src, "move"); NSL.StandardLib.FileSystem.FileHistory.Instance.SavePreEditState(dest, "move"); System.IO.File.Move(src, dest, true); return true; }),
                ["cwd"] = new NSLBuiltinFunction("cwd", (args) => System.IO.Directory.GetCurrentDirectory()),
                ["home"] = new NSLBuiltinFunction("home", (args) => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)),
                ["temp"] = new NSLBuiltinFunction("temp", (args) => System.IO.Path.GetTempPath()),
                ["env"] = new NSLBuiltinFunction("env", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.env() requires variable name"); return Environment.GetEnvironmentVariable(args[0]?.ToString() ?? "") ?? ""; }),
                ["setenv"] = new NSLBuiltinFunction("setenv", (args) => { if (args.Length < 2) throw new NSLRuntimeException("sys.setenv() requires name and value"); Environment.SetEnvironmentVariable(args[0]?.ToString() ?? "", args[1]?.ToString() ?? ""); return true; }),
                ["platform"] = new NSLBuiltinFunction("platform", (args) => Environment.OSVersion.ToString()),
                ["hostname"] = new NSLBuiltinFunction("hostname", (args) => Environment.MachineName),
                ["username"] = new NSLBuiltinFunction("username", (args) => Environment.UserName),
                ["processors"] = new NSLBuiltinFunction("processors", (args) => (double)Environment.ProcessorCount),
                ["memory"] = new NSLBuiltinFunction("memory", (args) => (double)GC.GetTotalMemory(false)),
                ["exit"] = new NSLBuiltinFunction("exit", (args) => { int code = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 0; Environment.Exit(code); return null; }),
                ["args"] = new NSLBuiltinFunction("args", (args) => Environment.GetCommandLineArgs().Select(a => (object?)a).ToList()),
                ["time"] = new NSLBuiltinFunction("time", (args) => DateTime.Now.ToString("o")),
                ["timestamp"] = new NSLBuiltinFunction("timestamp", (args) => (double)DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()),
                ["sleep"] = new NSLBuiltinFunction("sleep", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.sleep() requires milliseconds"); int ms = (int)ConvertToNumber(args[0]); Thread.Sleep(ms); return true; }),
                                ["cd"] = new NSLBuiltinFunction("cd", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.cd() requires directory path"); var path = args[0]?.ToString() ?? ""; if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); System.IO.Directory.SetCurrentDirectory(path); return path; }),
                ["exec"] = new NSLBuiltinFunction("exec", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.exec() requires a command"); var command = args[0]?.ToString() ?? ""; var workingDir = args.Length > 1 ? args[1]?.ToString() : null; var timeoutMs = args.Length > 2 ? (int)ConvertToNumber(args[2]) : 30000; var isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT; var psi = new System.Diagnostics.ProcessStartInfo { FileName = isWindows ? "cmd.exe" : "/bin/sh", Arguments = isWindows ? $"/c {command}" : $"-c \"{command.Replace("\"", "\\\"")}\"", RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = workingDir ?? Directory.GetCurrentDirectory() }; using var proc = System.Diagnostics.Process.Start(psi); if (proc == null) throw new NSLRuntimeException("Failed to start process"); var stdout = proc.StandardOutput.ReadToEnd(); var stderr = proc.StandardError.ReadToEnd(); var done = proc.WaitForExit(timeoutMs); if (!done) { proc.Kill(); throw new NSLRuntimeException($"Command timed out after {timeoutMs}ms"); } return new Dictionary<string, object?> { ["stdout"] = stdout.TrimEnd(), ["stderr"] = stderr.TrimEnd(), ["code"] = (double)proc.ExitCode, ["success"] = proc.ExitCode == 0 }; }),
                ["shell"] = new NSLBuiltinFunction("shell", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.shell() requires a command"); var command = args[0]?.ToString() ?? ""; var workingDir = args.Length > 1 ? args[1]?.ToString() : null; var isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT; var psi = new System.Diagnostics.ProcessStartInfo { FileName = isWindows ? "cmd.exe" : "/bin/sh", Arguments = isWindows ? $"/c {command}" : $"-c \"{command.Replace("\"", "\\\"")}\"", RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = workingDir ?? Directory.GetCurrentDirectory() }; using var proc = System.Diagnostics.Process.Start(psi); if (proc == null) throw new NSLRuntimeException("Failed to start process"); var output = proc.StandardOutput.ReadToEnd(); proc.WaitForExit(); return output.TrimEnd(); }),
                ["which"] = new NSLBuiltinFunction("which", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.which() requires a command name"); var cmd = args[0]?.ToString() ?? ""; var isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT; var pathVar = Environment.GetEnvironmentVariable("PATH") ?? ""; var paths = pathVar.Split(isWindows ? ';' : ':'); var exts = isWindows ? new[] { ".exe", ".cmd", ".bat", ".com", "" } : new[] { "" }; foreach (var p in paths) foreach (var ext in exts) { var full = System.IO.Path.Combine(p, cmd + ext); if (System.IO.File.Exists(full)) return full; } return null; }),
                ["pid"] = new NSLBuiltinFunction("pid", (args) => (double)Environment.ProcessId),
                ["kill"] = new NSLBuiltinFunction("kill", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.kill() requires a process ID"); var pid = (int)ConvertToNumber(args[0]); try { System.Diagnostics.Process.GetProcessById(pid).Kill(); return true; } catch { return false; } }),
                // Shell pipeline - chain commands like bash: sys.pipe("cat file.txt", "grep pattern", "sort")
                ["pipe"] = new NSLBuiltinFunction("pipe", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("sys.pipe() requires at least one command");
                    var isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT;
                    string? currentOutput = null;
                    
                    foreach (var arg in args)
                    {
                        var command = arg?.ToString() ?? "";
                        if (string.IsNullOrWhiteSpace(command)) continue;
                        
                        var psi = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = isWindows ? "cmd.exe" : "/bin/sh",
                            Arguments = isWindows ? $"/c {command}" : $"-c \"{command.Replace("\"", "\\\"")}\"",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            RedirectStandardInput = currentOutput != null,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        };
                        
                        using var proc = System.Diagnostics.Process.Start(psi);
                        if (proc == null) throw new NSLRuntimeException($"Failed to start: {command}");
                        
                        // Feed previous output as input
                        if (currentOutput != null)
                        {
                            proc.StandardInput.Write(currentOutput);
                            proc.StandardInput.Close();
                        }
                        
                        currentOutput = proc.StandardOutput.ReadToEnd();
                        var stderr = proc.StandardError.ReadToEnd();
                        proc.WaitForExit();
                        
                        if (proc.ExitCode != 0 && !string.IsNullOrEmpty(stderr))
                            throw new NSLRuntimeException($"Pipe failed at '{command}': {stderr.Trim()}");
                    }
                    
                    return currentOutput?.TrimEnd() ?? "";
                }),
                // Run command with stdin input
                ["run"] = new NSLBuiltinFunction("run", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("sys.run() requires a command");
                    var command = args[0]?.ToString() ?? "";
                    var stdin = args.Length > 1 ? args[1]?.ToString() : null;
                    var isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT;
                    
                    var psi = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = isWindows ? "cmd.exe" : "/bin/sh",
                        Arguments = isWindows ? $"/c {command}" : $"-c \"{command.Replace("\"", "\\\"")}\"",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        RedirectStandardInput = stdin != null,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    
                    using var proc = System.Diagnostics.Process.Start(psi);
                    if (proc == null) throw new NSLRuntimeException("Failed to start process");
                    
                    if (stdin != null)
                    {
                        proc.StandardInput.Write(stdin);
                        proc.StandardInput.Close();
                    }
                    
                    var stdout = proc.StandardOutput.ReadToEnd();
                    var stderr = proc.StandardError.ReadToEnd();
                    proc.WaitForExit();
                    
                    return new Dictionary<string, object?>
                    {
                        ["stdout"] = stdout.TrimEnd(),
                        ["stderr"] = stderr.TrimEnd(),
                        ["code"] = (double)proc.ExitCode,
                        ["success"] = proc.ExitCode == 0
                    };
                })
            };
            _globals["file"] = fileNamespace;

            // ===== FILEVIEW NAMESPACE - AI-Friendly Chunked File Navigation =====
            var fileviewNamespace = new Dictionary<string, object?>
            {
                ["overview"] = new NSLBuiltinFunction("overview", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("fileview.overview() requires a path argument");
                    var path = args[0]?.ToString() ?? "";
                    var exists = System.IO.File.Exists(path);
                    var lines = exists ? System.IO.File.ReadLines(path).Count() : 0;
                    var size = exists ? new System.IO.FileInfo(path).Length : 0;
                    var ext = System.IO.Path.GetExtension(path);
                    return new Dictionary<string, object?> {
                        ["path"] = path,
                        ["exists"] = exists,
                        ["lines"] = (double)lines,
                        ["size"] = (double)size,
                        ["extension"] = ext,
                        ["chunkSize"] = 100.0
                    };
                }),
                ["chunk"] = new NSLBuiltinFunction("chunk", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("fileview.chunk() requires path, start, count");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var start = (int)ConvertToNumber(args[1]);
                    var count = (int)ConvertToNumber(args[2]);
                    var allLines = System.IO.File.ReadAllLines(path);
                    var total = allLines.Length;
                    var end = Math.Min(start + count, total);
                    var lines = allLines.Skip(start).Take(end - start).Select(l => (object?)l).ToList();
                    return new Dictionary<string, object?> {
                        ["start"] = (double)start,
                        ["end"] = (double)end,
                        ["total"] = (double)total,
                        ["lines"] = lines,
                        ["hasMore"] = end < total,
                        ["hasPrev"] = start > 0
                    };
                }),
                ["search"] = new NSLBuiltinFunction("search", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("fileview.search() requires path and pattern");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var pattern = args[1]?.ToString() ?? "";
                    var contextLines = args.Length > 2 ? (int)ConvertToNumber(args[2]) : 2;
                    var allLines = System.IO.File.ReadAllLines(path);
                    var results = new List<object?>();
                    for (int i = 0; i < allLines.Length; i++) {
                        if (allLines[i].Contains(pattern, StringComparison.OrdinalIgnoreCase)) {
                            var start = Math.Max(0, i - contextLines);
                            var end = Math.Min(allLines.Length, i + contextLines + 1);
                            results.Add(new Dictionary<string, object?> {
                                ["lineNum"] = (double)i,
                                ["line"] = allLines[i],
                                ["start"] = (double)start,
                                ["end"] = (double)end
                            });
                        }
                    }
                    return results;
                }),
                ["context"] = new NSLBuiltinFunction("context", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("fileview.context() requires path and lineNum");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lineNum = (int)ConvertToNumber(args[1]);
                    var contextLines = args.Length > 2 ? (int)ConvertToNumber(args[2]) : 5;
                    var allLines = System.IO.File.ReadAllLines(path);
                    var total = allLines.Length;
                    var start = Math.Max(0, lineNum - contextLines);
                    var end = Math.Min(total, lineNum + contextLines + 1);
                    var lines = allLines.Skip(start).Take(end - start).Select(l => (object?)l).ToList();
                    return new Dictionary<string, object?> {
                        ["start"] = (double)start,
                        ["end"] = (double)end,
                        ["total"] = (double)total,
                        ["lines"] = lines,
                        ["centerLine"] = (double)lineNum
                    };
                }),
                ["page"] = new NSLBuiltinFunction("page", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("fileview.page() requires path, pageNum, pageSize");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var pageNum = (int)ConvertToNumber(args[1]);
                    var pageSize = (int)ConvertToNumber(args[2]);
                    var allLines = System.IO.File.ReadAllLines(path);
                    var total = allLines.Length;
                    var totalPages = (int)Math.Ceiling((double)total / pageSize);
                    var start = pageNum * pageSize;
                    var end = Math.Min(start + pageSize, total);
                    var lines = allLines.Skip(start).Take(end - start).Select(l => (object?)l).ToList();
                    return new Dictionary<string, object?> {
                        ["pageNum"] = (double)pageNum,
                        ["pageSize"] = (double)pageSize,
                        ["totalPages"] = (double)totalPages,
                        ["lines"] = lines,
                        ["hasNext"] = pageNum < totalPages - 1,
                        ["hasPrev"] = pageNum > 0
                    };
                }),
                ["toc"] = new NSLBuiltinFunction("toc", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("fileview.toc() requires a path argument");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var allLines = System.IO.File.ReadAllLines(path);
                    var results = new List<object?>();
                    for (int i = 0; i < allLines.Length; i++) {
                        var line = allLines[i].Trim();
                        var type = "";
                        if (line.Contains("fn ") || line.Contains("function ")) type = "function";
                        else if (line.Contains("class ")) type = "class";
                        else if (line.Contains("public ") || line.Contains("private ") || line.Contains("protected ")) type = "method";
                        else if (line.Contains("def ")) type = "function";
                        else if (line.StartsWith("#") && !line.StartsWith("##")) type = "heading";
                        if (!string.IsNullOrEmpty(type)) {
                            results.Add(new Dictionary<string, object?> {
                                ["line"] = (double)i,
                                ["text"] = allLines[i].Trim(),
                                ["type"] = type
                            });
                        }
                    }
                    return results;
                })
            };
            _globals["fileview"] = fileviewNamespace;

            // ===== SYS NAMESPACE - System/Process Operations =====
            var sysNamespace = new Dictionary<string, object?>
            {
                ["exec"] = fileNamespace["exec"],  // Inherit exec from file namespace
                ["shell"] = new NSLBuiltinFunction("shell", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("sys.shell() requires a command");
                    var cmd = args[0]?.ToString() ?? "";
                    var cwd = args.Length > 1 ? args[1]?.ToString() : null;
                    var proc = new System.Diagnostics.Process {
                        StartInfo = new System.Diagnostics.ProcessStartInfo {
                            FileName = Environment.OSVersion.Platform == PlatformID.Win32NT ? "cmd.exe" : "/bin/sh",
                            Arguments = Environment.OSVersion.Platform == PlatformID.Win32NT ? $"/c {cmd}" : $"-c \"{cmd}\"",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true,
                            WorkingDirectory = cwd ?? Environment.CurrentDirectory
                        }
                    };
                    proc.Start();
                    var output = proc.StandardOutput.ReadToEnd();
                    proc.WaitForExit();
                    return output.TrimEnd();
                }),
                ["stream"] = new NSLBuiltinFunction("stream", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("sys.stream() requires command and callback function");
                    var cmd = args[0]?.ToString() ?? "";
                    var callback = args[1];
                    var cwd = args.Length > 2 ? args[2]?.ToString() : null;
                    var proc = new System.Diagnostics.Process {
                        StartInfo = new System.Diagnostics.ProcessStartInfo {
                            FileName = Environment.OSVersion.Platform == PlatformID.Win32NT ? "cmd.exe" : "/bin/sh",
                            Arguments = Environment.OSVersion.Platform == PlatformID.Win32NT ? $"/c {cmd}" : $"-c \"{cmd}\"",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true,
                            WorkingDirectory = cwd ?? Environment.CurrentDirectory
                        }
                    };
                    proc.Start();
                    var lines = new List<object?>();
                    string? line;
                    while ((line = proc.StandardOutput.ReadLine()) != null) {
                        lines.Add(line);
                        if (callback is NSLFunction userFn) CallUserFunction(userFn, new object?[] { line, (double)lines.Count });
                        else if (callback is NSLBuiltinFunction builtinFn) builtinFn.Call(new object?[] { line, (double)lines.Count });
                    }
                    proc.WaitForExit();
                    return new Dictionary<string, object?> { ["lines"] = lines, ["code"] = (double)proc.ExitCode, ["success"] = proc.ExitCode == 0 };
                }),
                ["spawn"] = new NSLBuiltinFunction("spawn", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("sys.spawn() requires a command");
                    var cmd = args[0]?.ToString() ?? "";
                    var cmdArgs = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    var proc = new System.Diagnostics.Process {
                        StartInfo = new System.Diagnostics.ProcessStartInfo {
                            FileName = cmd,
                            Arguments = cmdArgs,
                            UseShellExecute = true
                        }
                    };
                    proc.Start();
                    return new Dictionary<string, object?> { ["pid"] = (double)proc.Id, ["name"] = proc.ProcessName };
                }),
                ["which"] = new NSLBuiltinFunction("which", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("sys.which() requires a command name");
                    var cmd = args[0]?.ToString() ?? "";
                    var paths = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator) ?? Array.Empty<string>();
                    var exts = Environment.OSVersion.Platform == PlatformID.Win32NT ? new[] { ".exe", ".cmd", ".bat", ".com" } : new[] { "" };
                    foreach (var path in paths) {
                        foreach (var ext in exts) {
                            var fullPath = Path.Combine(path, cmd + ext);
                            if (File.Exists(fullPath)) return fullPath;
                        }
                    }
                    return null;
                }),
                ["pid"] = new NSLBuiltinFunction("pid", (args) => (double)Environment.ProcessId),
                ["exit"] = new NSLBuiltinFunction("exit", (args) => { var code = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 0; Environment.Exit(code); return null; }),
                ["sleep"] = new NSLBuiltinFunction("sleep", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.sleep() requires milliseconds"); System.Threading.Thread.Sleep((int)ConvertToNumber(args[0])); return null; }),
                ["hostname"] = new NSLBuiltinFunction("hostname", (args) => Environment.MachineName),
                ["username"] = new NSLBuiltinFunction("username", (args) => Environment.UserName),
                ["os"] = new NSLBuiltinFunction("os", (args) => Environment.OSVersion.Platform.ToString()),
                ["arch"] = new NSLBuiltinFunction("arch", (args) => Environment.Is64BitOperatingSystem ? "x64" : "x86"),
                ["cpus"] = new NSLBuiltinFunction("cpus", (args) => (double)Environment.ProcessorCount),
                ["memory"] = new NSLBuiltinFunction("memory", (args) => new Dictionary<string, object?> {
                    ["total"] = (double)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
                    ["used"] = (double)GC.GetTotalMemory(false)
                }),
                ["uptime"] = new NSLBuiltinFunction("uptime", (args) => (double)Environment.TickCount64 / 1000.0),
                ["cwd"] = new NSLBuiltinFunction("cwd", (args) => Environment.CurrentDirectory),
                ["chdir"] = new NSLBuiltinFunction("chdir", (args) => { if (args.Length < 1) throw new NSLRuntimeException("sys.chdir() requires a path"); Environment.CurrentDirectory = args[0]?.ToString() ?? "."; return Environment.CurrentDirectory; }),
                ["home"] = new NSLBuiltinFunction("home", (args) => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)),
                ["temp"] = new NSLBuiltinFunction("temp", (args) => Path.GetTempPath()),
                ["args"] = new NSLBuiltinFunction("args", (args) => Environment.GetCommandLineArgs().Skip(1).Select(a => (object?)a).ToList())
            };
            _globals["sys"] = sysNamespace;

            // ===== PATH NAMESPACE =====
            var pathNamespace = new Dictionary<string, object?>
            {
                ["join"] = new NSLBuiltinFunction("join", (args) => { if (args.Length < 2) throw new NSLRuntimeException("path.join() requires at least 2 arguments"); return System.IO.Path.Combine(args.Select(a => a?.ToString() ?? "").ToArray()); }),
                ["dirname"] = new NSLBuiltinFunction("dirname", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.dirname() requires a path"); return System.IO.Path.GetDirectoryName(args[0]?.ToString() ?? "") ?? ""; }),
                ["basename"] = new NSLBuiltinFunction("basename", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.basename() requires a path"); return System.IO.Path.GetFileName(args[0]?.ToString() ?? ""); }),
                ["ext"] = new NSLBuiltinFunction("ext", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.ext() requires a path"); return System.IO.Path.GetExtension(args[0]?.ToString() ?? ""); }),
                ["stem"] = new NSLBuiltinFunction("stem", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.stem() requires a path"); return System.IO.Path.GetFileNameWithoutExtension(args[0]?.ToString() ?? ""); }),
                ["absolute"] = new NSLBuiltinFunction("absolute", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.absolute() requires a path"); return System.IO.Path.GetFullPath(args[0]?.ToString() ?? ""); }),
                ["normalize"] = new NSLBuiltinFunction("normalize", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.normalize() requires a path"); return System.IO.Path.GetFullPath(args[0]?.ToString() ?? "").Replace("", "/"); }),
                ["isAbsolute"] = new NSLBuiltinFunction("isAbsolute", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.isAbsolute() requires a path"); return System.IO.Path.IsPathRooted(args[0]?.ToString() ?? ""); }),
                ["exists"] = new NSLBuiltinFunction("exists", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.exists() requires a path"); var p = args[0]?.ToString() ?? ""; return System.IO.File.Exists(p) || System.IO.Directory.Exists(p); }),
                ["isFile"] = new NSLBuiltinFunction("isFile", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.isFile() requires a path"); return System.IO.File.Exists(args[0]?.ToString() ?? ""); }),
                ["isDir"] = new NSLBuiltinFunction("isDir", (args) => { if (args.Length < 1) throw new NSLRuntimeException("path.isDir() requires a path"); return System.IO.Directory.Exists(args[0]?.ToString() ?? ""); }),
                ["relative"] = new NSLBuiltinFunction("relative", (args) => { if (args.Length < 2) throw new NSLRuntimeException("path.relative() requires path and base"); return System.IO.Path.GetRelativePath(args[1]?.ToString() ?? ".", args[0]?.ToString() ?? ""); })
            };
            _globals["path"] = pathNamespace;

            // ===== DIR NAMESPACE =====
            var dirNamespace = new Dictionary<string, object?>
            {
                ["exists"] = new NSLBuiltinFunction("exists", (args) => { if (args.Length < 1) throw new NSLRuntimeException("dir.exists() requires a path"); return System.IO.Directory.Exists(args[0]?.ToString() ?? ""); }),
                ["list"] = new NSLBuiltinFunction("list", (args) => { var path = args.Length > 0 ? args[0]?.ToString() ?? "." : "."; if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); var entries = new List<object?>(); foreach (var d in System.IO.Directory.GetDirectories(path)) entries.Add(System.IO.Path.GetFileName(d) + "/"); foreach (var f in System.IO.Directory.GetFiles(path)) entries.Add(System.IO.Path.GetFileName(f)); return entries; }),
                ["files"] = new NSLBuiltinFunction("files", (args) => { var path = args.Length > 0 ? args[0]?.ToString() ?? "." : "."; var pattern = args.Length > 1 ? args[1]?.ToString() ?? "*" : "*"; if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); return System.IO.Directory.GetFiles(path, pattern).Select(f => (object?)f).ToList(); }),
                ["dirs"] = new NSLBuiltinFunction("dirs", (args) => { var path = args.Length > 0 ? args[0]?.ToString() ?? "." : "."; if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); return System.IO.Directory.GetDirectories(path).Select(d => (object?)d).ToList(); }),
                ["create"] = new NSLBuiltinFunction("create", (args) => { if (args.Length < 1) throw new NSLRuntimeException("dir.create() requires a path"); System.IO.Directory.CreateDirectory(args[0]?.ToString() ?? ""); return true; }),
                ["delete"] = new NSLBuiltinFunction("delete", (args) => { if (args.Length < 1) throw new NSLRuntimeException("dir.delete() requires a path"); var path = args[0]?.ToString() ?? ""; var recursive = args.Length > 1 && IsTruthy(args[1]); if (System.IO.Directory.Exists(path)) { System.IO.Directory.Delete(path, recursive); return true; } return false; }),
                ["copy"] = new NSLBuiltinFunction("copy", (args) => { if (args.Length < 2) throw new NSLRuntimeException("dir.copy() requires source and destination"); var src = args[0]?.ToString() ?? ""; var dst = args[1]?.ToString() ?? ""; if (!System.IO.Directory.Exists(src)) throw new NSLRuntimeException($"Source directory not found: {src}"); void CopyDir(string s, string d) { System.IO.Directory.CreateDirectory(d); foreach (var f in System.IO.Directory.GetFiles(s)) System.IO.File.Copy(f, System.IO.Path.Combine(d, System.IO.Path.GetFileName(f)), true); foreach (var sub in System.IO.Directory.GetDirectories(s)) CopyDir(sub, System.IO.Path.Combine(d, System.IO.Path.GetFileName(sub))); } CopyDir(src, dst); return true; }),
                ["move"] = new NSLBuiltinFunction("move", (args) => { if (args.Length < 2) throw new NSLRuntimeException("dir.move() requires source and destination"); System.IO.Directory.Move(args[0]?.ToString() ?? "", args[1]?.ToString() ?? ""); return true; }),
                ["size"] = new NSLBuiltinFunction("size", (args) => { if (args.Length < 1) throw new NSLRuntimeException("dir.size() requires a path"); var path = args[0]?.ToString() ?? ""; if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); long size = 0; foreach (var file in System.IO.Directory.GetFiles(path, "*", System.IO.SearchOption.AllDirectories)) { try { size += new System.IO.FileInfo(file).Length; } catch { } } return (double)size; }),
                ["tree"] = new NSLBuiltinFunction("tree", (args) => { var path = args.Length > 0 ? args[0]?.ToString() ?? "." : "."; var depth = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 3; if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); var result = new List<object?>(); void Walk(string dir, int level, string prefix) { if (level > depth) return; try { foreach (var d in System.IO.Directory.GetDirectories(dir)) { result.Add(prefix + System.IO.Path.GetFileName(d) + "/"); Walk(d, level + 1, prefix + "  "); } foreach (var f in System.IO.Directory.GetFiles(dir)) { result.Add(prefix + System.IO.Path.GetFileName(f)); } } catch { } } Walk(path, 0, ""); return result; }),
                ["walk"] = new NSLBuiltinFunction("walk", (args) => { var path = args.Length > 0 ? args[0]?.ToString() ?? "." : "."; var pattern = args.Length > 1 ? args[1]?.ToString() ?? "*" : "*"; if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); return System.IO.Directory.GetFiles(path, pattern, System.IO.SearchOption.AllDirectories).Select(f => (object?)f).ToList(); }),
                ["walkText"] = new NSLBuiltinFunction("walkText", (args) => { 
                    var path = args.Length > 0 ? args[0]?.ToString() ?? "." : "."; 
                    var pattern = args.Length > 1 ? args[1]?.ToString() ?? "*" : "*";
                    if (!System.IO.Directory.Exists(path)) throw new NSLRuntimeException($"Directory not found: {path}"); 
                    var textExts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".cs", ".nsl", ".txt", ".md", ".json", ".xml", ".yaml", ".yml", ".js", ".ts", ".py", ".html", ".css", ".sh", ".bat", ".ps1", ".sql", ".config", ".csproj", ".sln", ".gitignore", ".editorconfig" };
                    return System.IO.Directory.GetFiles(path, pattern, System.IO.SearchOption.AllDirectories)
                        .Where(f => textExts.Contains(System.IO.Path.GetExtension(f)))
                        .Select(f => (object?)f).ToList(); 
                })
            };
            _globals["dir"] = dirNamespace;

            // ===== REGEX NAMESPACE =====
            var regexNamespace = new Dictionary<string, object?>
            {
                ["match"] = new NSLBuiltinFunction("match", (args) => { if (args.Length < 2) throw new NSLRuntimeException("regex.match() requires text and pattern"); var text = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; var m = System.Text.RegularExpressions.Regex.Match(text, pattern); return m.Success ? m.Value : null; }),
                ["matches"] = new NSLBuiltinFunction("matches", (args) => { if (args.Length < 2) throw new NSLRuntimeException("regex.matches() requires text and pattern"); var text = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; return System.Text.RegularExpressions.Regex.Matches(text, pattern).Cast<System.Text.RegularExpressions.Match>().Select(m => (object?)m.Value).ToList(); }),
                ["test"] = new NSLBuiltinFunction("test", (args) => { if (args.Length < 2) throw new NSLRuntimeException("regex.test() requires text and pattern"); var text = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; return System.Text.RegularExpressions.Regex.IsMatch(text, pattern); }),
                ["replace"] = new NSLBuiltinFunction("replace", (args) => { if (args.Length < 3) throw new NSLRuntimeException("regex.replace() requires text, pattern, and replacement"); var text = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; var replacement = args[2]?.ToString() ?? ""; return System.Text.RegularExpressions.Regex.Replace(text, pattern, replacement); }),
                ["split"] = new NSLBuiltinFunction("split", (args) => { if (args.Length < 2) throw new NSLRuntimeException("regex.split() requires text and pattern"); var text = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; return System.Text.RegularExpressions.Regex.Split(text, pattern).Select(s => (object?)s).ToList(); }),
                ["groups"] = new NSLBuiltinFunction("groups", (args) => { if (args.Length < 2) throw new NSLRuntimeException("regex.groups() requires text and pattern"); var text = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; var m = System.Text.RegularExpressions.Regex.Match(text, pattern); if (!m.Success) return new List<object?>(); return m.Groups.Cast<System.Text.RegularExpressions.Group>().Skip(1).Select(g => (object?)g.Value).ToList(); }),
                ["escape"] = new NSLBuiltinFunction("escape", (args) => { if (args.Length < 1) throw new NSLRuntimeException("regex.escape() requires text"); return System.Text.RegularExpressions.Regex.Escape(args[0]?.ToString() ?? ""); })
            };
            _globals["regex"] = regexNamespace;

            // ===== TEXT NAMESPACE (for common text operations) =====
            var textNamespace = new Dictionary<string, object?>
            {
                ["lines"] = new NSLBuiltinFunction("lines", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.lines() requires text"); return (args[0]?.ToString() ?? "").Split('\n').Select(l => (object?)l.TrimEnd('\r')).ToList(); }),
                ["words"] = new NSLBuiltinFunction("words", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.words() requires text"); return System.Text.RegularExpressions.Regex.Split(args[0]?.ToString() ?? "", @"\s+").Where(w => !string.IsNullOrEmpty(w)).Select(w => (object?)w).ToList(); }),
                ["chars"] = new NSLBuiltinFunction("chars", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.chars() requires text"); return (args[0]?.ToString() ?? "").ToCharArray().Select(c => (object?)c.ToString()).ToList(); }),
                ["count"] = new NSLBuiltinFunction("count", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.count() requires text and substring"); var text = args[0]?.ToString() ?? ""; var sub = args[1]?.ToString() ?? ""; if (string.IsNullOrEmpty(sub)) return 0.0; int count = 0, i = 0; while ((i = text.IndexOf(sub, i)) != -1) { count++; i += sub.Length; } return (double)count; }),
                ["reverse"] = new NSLBuiltinFunction("reverse", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.reverse() requires text"); return new string((args[0]?.ToString() ?? "").Reverse().ToArray()); }),
                ["wrap"] = new NSLBuiltinFunction("wrap", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.wrap() requires text and width"); var text = args[0]?.ToString() ?? ""; var width = (int)ConvertToNumber(args[1]); var words = text.Split(' '); var lines = new List<string>(); var current = ""; foreach (var word in words) { if (current.Length + word.Length + 1 > width) { if (current.Length > 0) lines.Add(current); current = word; } else { current = current.Length > 0 ? current + " " + word : word; } } if (current.Length > 0) lines.Add(current); return string.Join("\n", lines); }),
                ["truncate"] = new NSLBuiltinFunction("truncate", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.truncate() requires text and maxLength"); var text = args[0]?.ToString() ?? ""; var max = (int)ConvertToNumber(args[1]); var suffix = args.Length > 2 ? args[2]?.ToString() ?? "..." : "..."; return text.Length <= max ? text : text.Substring(0, max - suffix.Length) + suffix; }),
                ["indent"] = new NSLBuiltinFunction("indent", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.indent() requires text"); var text = args[0]?.ToString() ?? ""; var spaces = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 2; var prefix = new string(' ', spaces); return string.Join("\n", text.Split('\n').Select(l => prefix + l)); }),
                ["dedent"] = new NSLBuiltinFunction("dedent", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.dedent() requires text"); var text = args[0]?.ToString() ?? ""; var lines = text.Split('\n'); var minIndent = lines.Where(l => l.Trim().Length > 0).Select(l => l.TakeWhile(char.IsWhiteSpace).Count()).DefaultIfEmpty(0).Min(); return string.Join("\n", lines.Select(l => l.Length >= minIndent ? l.Substring(minIndent) : l)); }),
                ["slug"] = new NSLBuiltinFunction("slug", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.slug() requires text"); return System.Text.RegularExpressions.Regex.Replace(args[0]?.ToString()?.ToLower() ?? "", @"[^a-z0-9]+", "-").Trim('-'); }),
                ["title"] = new NSLBuiltinFunction("title", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.title() requires text"); return System.Globalization.CultureInfo.CurrentCulture.TextInfo.ToTitleCase(args[0]?.ToString()?.ToLower() ?? ""); }),
                ["upper"] = new NSLBuiltinFunction("upper", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.upper() requires text"); return (args[0]?.ToString() ?? "").ToUpper(); }),
                ["lower"] = new NSLBuiltinFunction("lower", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.lower() requires text"); return (args[0]?.ToString() ?? "").ToLower(); }),
                ["trim"] = new NSLBuiltinFunction("trim", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.trim() requires text"); return (args[0]?.ToString() ?? "").Trim(); }),
                ["trimStart"] = new NSLBuiltinFunction("trimStart", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.trimStart() requires text"); return (args[0]?.ToString() ?? "").TrimStart(); }),
                ["trimEnd"] = new NSLBuiltinFunction("trimEnd", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.trimEnd() requires text"); return (args[0]?.ToString() ?? "").TrimEnd(); }),
                ["pad"] = new NSLBuiltinFunction("pad", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.pad() requires text and width"); var text = args[0]?.ToString() ?? ""; var width = (int)ConvertToNumber(args[1]); var padChar = args.Length > 2 ? (args[2]?.ToString() ?? " ")[0] : ' '; var left = (width - text.Length) / 2; return text.PadLeft(text.Length + left, padChar).PadRight(width, padChar); }),
                ["padLeft"] = new NSLBuiltinFunction("padLeft", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.padLeft() requires text and width"); return (args[0]?.ToString() ?? "").PadLeft((int)ConvertToNumber(args[1]), args.Length > 2 ? (args[2]?.ToString() ?? " ")[0] : ' '); }),
                ["padRight"] = new NSLBuiltinFunction("padRight", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.padRight() requires text and width"); return (args[0]?.ToString() ?? "").PadRight((int)ConvertToNumber(args[1]), args.Length > 2 ? (args[2]?.ToString() ?? " ")[0] : ' '); }),
                ["repeat"] = new NSLBuiltinFunction("repeat", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.repeat() requires text and count"); return string.Concat(Enumerable.Repeat(args[0]?.ToString() ?? "", (int)ConvertToNumber(args[1]))); }),
                ["center"] = new NSLBuiltinFunction("center", (args) => { if (args.Length < 2) throw new NSLRuntimeException("text.center() requires text and width"); var text = args[0]?.ToString() ?? ""; var width = (int)ConvertToNumber(args[1]); if (text.Length >= width) return text; var left = (width - text.Length) / 2; return new string(' ', left) + text + new string(' ', width - text.Length - left); }),
                ["capitalize"] = new NSLBuiltinFunction("capitalize", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.capitalize() requires text"); var text = args[0]?.ToString() ?? ""; return text.Length > 0 ? char.ToUpper(text[0]) + text.Substring(1).ToLower() : ""; }),
                ["isUpper"] = new NSLBuiltinFunction("isUpper", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.isUpper() requires text"); return (args[0]?.ToString() ?? "").All(c => !char.IsLetter(c) || char.IsUpper(c)); }),
                ["isLower"] = new NSLBuiltinFunction("isLower", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.isLower() requires text"); return (args[0]?.ToString() ?? "").All(c => !char.IsLetter(c) || char.IsLower(c)); }),
                ["isDigit"] = new NSLBuiltinFunction("isDigit", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.isDigit() requires text"); return (args[0]?.ToString() ?? "").All(char.IsDigit); }),
                ["isAlpha"] = new NSLBuiltinFunction("isAlpha", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.isAlpha() requires text"); return (args[0]?.ToString() ?? "").All(char.IsLetter); }),
                ["isAlnum"] = new NSLBuiltinFunction("isAlnum", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.isAlnum() requires text"); return (args[0]?.ToString() ?? "").All(char.IsLetterOrDigit); }),
                ["isSpace"] = new NSLBuiltinFunction("isSpace", (args) => { if (args.Length < 1) throw new NSLRuntimeException("text.isSpace() requires text"); return (args[0]?.ToString() ?? "").All(char.IsWhiteSpace); })
            };
            _globals["text"] = textNamespace;

            // ===== CODE ANALYSIS NAMESPACE =====
            var codeNamespace = new Dictionary<string, object?>
            {
                // Read and analyze file structure
                ["read"] = new NSLBuiltinFunction("read", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.read() requires path");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    var lines = content.Split('\n');
                    var ext = System.IO.Path.GetExtension(path).ToLower();
                    return new Dictionary<string, object?> {
                        ["path"] = path,
                        ["content"] = content,
                        ["lines"] = lines.Length,
                        ["chars"] = content.Length,
                        ["extension"] = ext,
                        ["language"] = DetectLanguage(ext),
                        ["encoding"] = "utf-8"
                    };
                }),
                
                // Get code metrics (complexity, lines, etc.)
                ["metrics"] = new NSLBuiltinFunction("metrics", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.metrics() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var lines = content.Split('\n');
                    var codeLines = lines.Where(l => !string.IsNullOrWhiteSpace(l) && !l.TrimStart().StartsWith("//") && !l.TrimStart().StartsWith("#")).Count();
                    var commentLines = lines.Where(l => l.TrimStart().StartsWith("//") || l.TrimStart().StartsWith("#")).Count();
                    var blankLines = lines.Where(l => string.IsNullOrWhiteSpace(l)).Count();
                    var maxLineLength = lines.Max(l => l.Length);
                    var avgLineLength = lines.Average(l => (double)l.Length);
                    
                    // Simple cyclomatic complexity estimate (count branches)
                    var branches = System.Text.RegularExpressions.Regex.Matches(content, @"\b(if|else|for|foreach|while|switch|case|catch|&&|\|\||\?)\b").Count;
                    
                    return new Dictionary<string, object?> {
                        ["totalLines"] = lines.Length,
                        ["codeLines"] = codeLines,
                        ["commentLines"] = commentLines,
                        ["blankLines"] = blankLines,
                        ["maxLineLength"] = maxLineLength,
                        ["avgLineLength"] = Math.Round(avgLineLength, 1),
                        ["complexity"] = branches + 1,
                        ["chars"] = content.Length
                    };
                }),
                
                // Find symbols (functions, classes, variables)
                ["symbols"] = new NSLBuiltinFunction("symbols", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.symbols() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var ext = System.IO.File.Exists(input) ? System.IO.Path.GetExtension(input).ToLower() : ".cs";
                    
                    var symbols = new List<object?>();
                    
                    // C#/Java style
                    if (ext == ".cs" || ext == ".java" || ext == ".ts" || ext == ".js") {
                        // Classes
                        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"(?:public|private|internal|protected)?\s*(?:static\s+)?(?:class|interface|struct|enum|record)\s+(\w+)"))
                            symbols.Add(new Dictionary<string, object?> { ["type"] = "class", ["name"] = m.Groups[1].Value, ["line"] = GetLineNumber(content, m.Index) });
                        
                        // Methods/Functions
                        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"(?:public|private|internal|protected)?\s*(?:static\s+)?(?:async\s+)?(?:\w+(?:<[\w,\s]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:=>|{)"))
                            symbols.Add(new Dictionary<string, object?> { ["type"] = "method", ["name"] = m.Groups[1].Value, ["line"] = GetLineNumber(content, m.Index) });
                    }
                    // Python style
                    else if (ext == ".py") {
                        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"^class\s+(\w+)", System.Text.RegularExpressions.RegexOptions.Multiline))
                            symbols.Add(new Dictionary<string, object?> { ["type"] = "class", ["name"] = m.Groups[1].Value, ["line"] = GetLineNumber(content, m.Index) });
                        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"^def\s+(\w+)", System.Text.RegularExpressions.RegexOptions.Multiline))
                            symbols.Add(new Dictionary<string, object?> { ["type"] = "function", ["name"] = m.Groups[1].Value, ["line"] = GetLineNumber(content, m.Index) });
                    }
                    // NSL style
                    else if (ext == ".nsl") {
                        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"(?:let|const)\s+(\w+)\s*=\s*(?:fn|func|\()"))
                            symbols.Add(new Dictionary<string, object?> { ["type"] = "function", ["name"] = m.Groups[1].Value, ["line"] = GetLineNumber(content, m.Index) });
                    }
                    
                    return symbols;
                }),
                
                // Find imports/dependencies
                ["deps"] = new NSLBuiltinFunction("deps", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.deps() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var ext = System.IO.File.Exists(input) ? System.IO.Path.GetExtension(input).ToLower() : "";
                    
                    var deps = new List<object?>();
                    
                    // C# using
                    foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"^using\s+([\w.]+);", System.Text.RegularExpressions.RegexOptions.Multiline))
                        deps.Add(m.Groups[1].Value);
                    
                    // Python import
                    foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"^(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)", System.Text.RegularExpressions.RegexOptions.Multiline))
                        deps.Add(string.IsNullOrEmpty(m.Groups[1].Value) ? m.Groups[2].Value.Trim() : m.Groups[1].Value);
                    
                    // JS/TS import/require
                    foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(content, @"(?:import\s+.*\s+from\s+['""]([^'""]+)['""]|require\s*\(\s*['""]([^'""]+)['""]\s*\))"))
                        deps.Add(m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value);
                    
                    return deps.Distinct().ToList();
                }),
                
                // Find potential issues (simple static analysis)
                ["issues"] = new NSLBuiltinFunction("issues", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.issues() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var lines = content.Split('\n');
                    
                    var issues = new List<object?>();
                    
                    for (int i = 0; i < lines.Length; i++) {
                        var line = lines[i];
                        var lineNum = i + 1;
                        
                        // Long lines
                        if (line.Length > 120)
                            issues.Add(new Dictionary<string, object?> { ["line"] = lineNum, ["type"] = "style", ["message"] = $"Line too long ({line.Length} chars)" });
                        
                        // TODO/FIXME/HACK
                        if (System.Text.RegularExpressions.Regex.IsMatch(line, @"\b(TODO|FIXME|HACK|XXX)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase))
                            issues.Add(new Dictionary<string, object?> { ["line"] = lineNum, ["type"] = "todo", ["message"] = line.Trim() });
                        
                        // Empty catch blocks
                        if (System.Text.RegularExpressions.Regex.IsMatch(line, @"catch\s*\([^)]*\)\s*\{\s*\}"))
                            issues.Add(new Dictionary<string, object?> { ["line"] = lineNum, ["type"] = "warning", ["message"] = "Empty catch block swallows exceptions" });
                        
                        // Magic numbers (excluding 0, 1, 2)
                        if (System.Text.RegularExpressions.Regex.IsMatch(line, @"[^.\w]([3-9]|\d{2,})[^.\w]") && !line.TrimStart().StartsWith("//"))
                            if (!System.Text.RegularExpressions.Regex.IsMatch(line, @"(const|readonly|enum|#define)"))
                                issues.Add(new Dictionary<string, object?> { ["line"] = lineNum, ["type"] = "hint", ["message"] = "Consider extracting magic number to constant" });
                        
                        // Nested ternary
                        if (System.Text.RegularExpressions.Regex.Matches(line, @"\?.*:").Count > 1)
                            issues.Add(new Dictionary<string, object?> { ["line"] = lineNum, ["type"] = "warning", ["message"] = "Nested ternary operator reduces readability" });
                    }
                    
                    return issues;
                }),
                
                // Compare two files/strings (structured diff)
                ["compare"] = new NSLBuiltinFunction("compare", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("code.compare() requires two paths or contents");
                    var old = args[0]?.ToString() ?? "";
                    var newContent = args[1]?.ToString() ?? "";
                    
                    var oldContent = System.IO.File.Exists(old) ? System.IO.File.ReadAllText(old) : old;
                    var newStr = System.IO.File.Exists(newContent) ? System.IO.File.ReadAllText(newContent) : newContent;
                    
                    var oldLines = oldContent.Split('\n');
                    var newLines = newStr.Split('\n');
                    
                    var changes = new List<object?>();
                    int oldIdx = 0, newIdx = 0;
                    
                    while (oldIdx < oldLines.Length || newIdx < newLines.Length) {
                        if (oldIdx >= oldLines.Length) {
                            changes.Add(new Dictionary<string, object?> { ["type"] = "add", ["line"] = newIdx + 1, ["content"] = newLines[newIdx] });
                            newIdx++;
                        } else if (newIdx >= newLines.Length) {
                            changes.Add(new Dictionary<string, object?> { ["type"] = "remove", ["line"] = oldIdx + 1, ["content"] = oldLines[oldIdx] });
                            oldIdx++;
                        } else if (oldLines[oldIdx] == newLines[newIdx]) {
                            oldIdx++;
                            newIdx++;
                        } else {
                            changes.Add(new Dictionary<string, object?> { ["type"] = "remove", ["line"] = oldIdx + 1, ["content"] = oldLines[oldIdx] });
                            changes.Add(new Dictionary<string, object?> { ["type"] = "add", ["line"] = newIdx + 1, ["content"] = newLines[newIdx] });
                            oldIdx++;
                            newIdx++;
                        }
                    }
                    
                    return new Dictionary<string, object?> {
                        ["oldLines"] = oldLines.Length,
                        ["newLines"] = newLines.Length,
                        ["changes"] = changes,
                        ["changeCount"] = changes.Count
                    };
                }),
                
                // Search for pattern in code
                ["search"] = new NSLBuiltinFunction("search", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("code.search() requires content/path and pattern");
                    var input = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var lines = content.Split('\n');
                    
                    var results = new List<object?>();
                    for (int i = 0; i < lines.Length; i++) {
                        if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], pattern, System.Text.RegularExpressions.RegexOptions.IgnoreCase)) {
                            results.Add(new Dictionary<string, object?> {
                                ["line"] = i + 1,
                                ["content"] = lines[i].Trim(),
                                ["match"] = System.Text.RegularExpressions.Regex.Match(lines[i], pattern, System.Text.RegularExpressions.RegexOptions.IgnoreCase).Value
                            });
                        }
                    }
                    return results;
                }),
                
                // Get function/method body
                ["extract"] = new NSLBuiltinFunction("extract", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("code.extract() requires content/path and symbol name");
                    var input = args[0]?.ToString() ?? "";
                    var symbol = args[1]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    
                    // Find the symbol and extract its body (brace-matching)
                    var pattern = $@"(?:public|private|protected|internal|static|async|\s)*(?:\w+\s+)?{System.Text.RegularExpressions.Regex.Escape(symbol)}\s*(?:<[^>]+>)?\s*\([^)]*\)\s*(?:=>|{{)";
                    var match = System.Text.RegularExpressions.Regex.Match(content, pattern);
                    
                    if (!match.Success) return null;
                    
                    var startIdx = match.Index;
                    var braceStart = content.IndexOf('{', startIdx);
                    if (braceStart == -1) {
                        // Arrow function - find end of expression
                        var arrowIdx = content.IndexOf("=>", startIdx);
                        if (arrowIdx == -1) return null;
                        var endIdx = content.IndexOf(';', arrowIdx);
                        return content.Substring(startIdx, (endIdx > 0 ? endIdx + 1 : content.Length) - startIdx);
                    }
                    
                    // Brace matching
                    int depth = 0;
                    int endBrace = braceStart;
                    for (int i = braceStart; i < content.Length; i++) {
                        if (content[i] == '{') depth++;
                        else if (content[i] == '}') { depth--; if (depth == 0) { endBrace = i; break; } }
                    }
                    
                    return content.Substring(startIdx, endBrace - startIdx + 1);
                }),
                
                // Analyze control flow
                ["flow"] = new NSLBuiltinFunction("flow", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.flow() requires content/path");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var lines = content.Split('\n');
                    
                    var flow = new List<object?>();
                    int depth = 0;
                    
                    for (int i = 0; i < lines.Length; i++) {
                        var line = lines[i].Trim();
                        if (string.IsNullOrEmpty(line)) continue;
                        
                        // Track control structures
                        if (System.Text.RegularExpressions.Regex.IsMatch(line, @"^\s*(if|else\s+if|switch)\s*\(")) {
                            flow.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["type"] = "branch", ["depth"] = depth, ["code"] = line });
                            if (line.Contains("{")) depth++;
                        } else if (System.Text.RegularExpressions.Regex.IsMatch(line, @"^\s*(for|foreach|while|do)\s*[\({]")) {
                            flow.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["type"] = "loop", ["depth"] = depth, ["code"] = line });
                            if (line.Contains("{")) depth++;
                        } else if (System.Text.RegularExpressions.Regex.IsMatch(line, @"^\s*try\s*\{")) {
                            flow.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["type"] = "try", ["depth"] = depth, ["code"] = line });
                            depth++;
                        } else if (System.Text.RegularExpressions.Regex.IsMatch(line, @"^\s*catch\s*\(")) {
                            flow.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["type"] = "catch", ["depth"] = depth, ["code"] = line });
                        } else if (System.Text.RegularExpressions.Regex.IsMatch(line, @"^\s*return\b")) {
                            flow.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["type"] = "return", ["depth"] = depth, ["code"] = line });
                        } else if (System.Text.RegularExpressions.Regex.IsMatch(line, @"^\s*throw\b")) {
                            flow.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["type"] = "throw", ["depth"] = depth, ["code"] = line });
                        }
                        
                        // Track depth changes
                        depth += line.Count(c => c == '{') - line.Count(c => c == '}');
                        if (depth < 0) depth = 0;
                    }
                    
                    return flow;
                }),
                // === AI-Focused Code Analysis Functions ===
                ["braceBalance"] = new NSLBuiltinFunction("braceBalance", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.braceBalance() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    int braces = 0, parens = 0, brackets = 0;
                    bool inString = false, inLineComment = false, inBlockComment = false;
                    char stringChar = '"';
                    int firstUnbalancedLine = -1;
                    char firstUnbalancedChar = ' ';
                    int lineNum = 1;
                    for (int i = 0; i < content.Length; i++) {
                        char c = content[i];
                        char next = i + 1 < content.Length ? content[i + 1] : '\0';
                        char prev = i > 0 ? content[i - 1] : '\0';
                        if (c == '\n') { lineNum++; inLineComment = false; continue; }
                        if (inLineComment) continue;
                        if (inBlockComment) { if (c == '*' && next == '/') { inBlockComment = false; i++; } continue; }
                        if (!inString && c == '/' && next == '/') { inLineComment = true; continue; }
                        if (!inString && c == '/' && next == '*') { inBlockComment = true; i++; continue; }
                        if (!inString && (c == '"' || c == '\'')) { inString = true; stringChar = c; continue; }
                        if (inString && c == stringChar && prev != '\\') { inString = false; continue; }
                        if (inString) continue;
                        if (c == '{') braces++;
                        else if (c == '}') { braces--; if (braces < 0 && firstUnbalancedLine == -1) { firstUnbalancedLine = lineNum; firstUnbalancedChar = '}'; } }
                        else if (c == '(') parens++;
                        else if (c == ')') { parens--; if (parens < 0 && firstUnbalancedLine == -1) { firstUnbalancedLine = lineNum; firstUnbalancedChar = ')'; } }
                        else if (c == '[') brackets++;
                        else if (c == ']') { brackets--; if (brackets < 0 && firstUnbalancedLine == -1) { firstUnbalancedLine = lineNum; firstUnbalancedChar = ']'; } }
                    }
                    bool balanced = braces == 0 && parens == 0 && brackets == 0;
                    return new Dictionary<string, object?> {
                        ["balanced"] = balanced,
                        ["braces"] = (double)braces, ["parens"] = (double)parens, ["brackets"] = (double)brackets,
                        ["firstUnbalancedLine"] = firstUnbalancedLine > 0 ? (double)firstUnbalancedLine : null,
                        ["firstUnbalancedChar"] = firstUnbalancedLine > 0 ? firstUnbalancedChar.ToString() : null
                    };
                }),
                ["findUnmatched"] = new NSLBuiltinFunction("findUnmatched", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.findUnmatched() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var issues = new List<object?>();
                    var stack = new Stack<(char c, int line, int col)>();
                    bool inString = false, inLineComment = false, inBlockComment = false;
                    char stringChar = '"';
                    int lineNum = 1, colNum = 0;
                    for (int i = 0; i < content.Length; i++) {
                        char c = content[i];
                        char next = i + 1 < content.Length ? content[i + 1] : '\0';
                        char prev = i > 0 ? content[i - 1] : '\0';
                        colNum++;
                        if (c == '\n') { lineNum++; colNum = 0; inLineComment = false; continue; }
                        if (inLineComment) continue;
                        if (inBlockComment) { if (c == '*' && next == '/') { inBlockComment = false; i++; colNum++; } continue; }
                        if (!inString && c == '/' && next == '/') { inLineComment = true; continue; }
                        if (!inString && c == '/' && next == '*') { inBlockComment = true; i++; colNum++; continue; }
                        if (!inString && (c == '"' || c == '\'')) { inString = true; stringChar = c; continue; }
                        if (inString && c == stringChar && prev != '\\') { inString = false; continue; }
                        if (inString) continue;
                        if (c == '{' || c == '(' || c == '[') stack.Push((c, lineNum, colNum));
                        else if (c == '}' || c == ')' || c == ']') {
                            char expected = c == '}' ? '{' : c == ')' ? '(' : '[';
                            if (stack.Count == 0 || stack.Peek().c != expected) {
                                issues.Add(new Dictionary<string, object?> { ["type"] = "unmatched", ["char"] = c.ToString(), ["line"] = (double)lineNum, ["col"] = (double)colNum });
                            } else { stack.Pop(); }
                        }
                    }
                    while (stack.Count > 0) {
                        var (ch, line, col) = stack.Pop();
                        issues.Add(new Dictionary<string, object?> { ["type"] = "unclosed", ["char"] = ch.ToString(), ["line"] = (double)line, ["col"] = (double)col });
                    }
                    return issues;
                }),
                ["structure"] = new NSLBuiltinFunction("structure", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.structure() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var lang = args.Length > 1 ? args[1]?.ToString()?.ToLower() ?? "auto" : "auto";
                    if (lang == "auto") {
                        if (input.EndsWith(".cs")) lang = "csharp";
                        else if (input.EndsWith(".nsl")) lang = "nsl";
                        else if (input.EndsWith(".js") || input.EndsWith(".ts")) lang = "javascript";
                        else if (input.EndsWith(".py")) lang = "python";
                        else lang = "generic";
                    }
                    var classes = new List<object?>();
                    var functions = new List<object?>();
                    var interfaces = new List<object?>();
                    var enums = new List<object?>();
                    var lines = content.Split('\n');
                    for (int i = 0; i < lines.Length; i++) {
                        var line = lines[i].TrimStart();
                        var ln = i + 1;
                        if (lang == "csharp" || lang == "generic") {
                            var classMatch = System.Text.RegularExpressions.Regex.Match(line, @"^\s*(public|private|protected|internal)?\s*(partial\s+)?(class|struct|record)\s+(\w+)");
                            if (classMatch.Success) classes.Add(new Dictionary<string, object?> { ["name"] = classMatch.Groups[4].Value, ["line"] = (double)ln, ["type"] = classMatch.Groups[3].Value });
                            var ifaceMatch = System.Text.RegularExpressions.Regex.Match(line, @"^\s*(public|private|protected|internal)?\s*interface\s+(\w+)");
                            if (ifaceMatch.Success) interfaces.Add(new Dictionary<string, object?> { ["name"] = ifaceMatch.Groups[2].Value, ["line"] = (double)ln });
                            var enumMatch = System.Text.RegularExpressions.Regex.Match(line, @"^\s*(public|private|protected|internal)?\s*enum\s+(\w+)");
                            if (enumMatch.Success) enums.Add(new Dictionary<string, object?> { ["name"] = enumMatch.Groups[2].Value, ["line"] = (double)ln });
                        }
                        if (lang == "nsl" || lang == "generic") {
                            var fnMatch = System.Text.RegularExpressions.Regex.Match(line, @"^\s*(fn|func|function)\s+(\w+)\s*\(");
                            if (fnMatch.Success) functions.Add(new Dictionary<string, object?> { ["name"] = fnMatch.Groups[2].Value, ["line"] = (double)ln });
                        }
                        if (lang == "javascript" || lang == "generic") {
                            var jsFnMatch = System.Text.RegularExpressions.Regex.Match(line, @"^\s*(async\s+)?function\s+(\w+)");
                            if (jsFnMatch.Success) functions.Add(new Dictionary<string, object?> { ["name"] = jsFnMatch.Groups[2].Value, ["line"] = (double)ln });
                        }
                        if (lang == "python" || lang == "generic") {
                            var pyFnMatch = System.Text.RegularExpressions.Regex.Match(line, @"^\s*(async\s+)?def\s+(\w+)\s*\(");
                            if (pyFnMatch.Success) functions.Add(new Dictionary<string, object?> { ["name"] = pyFnMatch.Groups[2].Value, ["line"] = (double)ln });
                        }
                    }
                    return new Dictionary<string, object?> { ["classes"] = classes, ["functions"] = functions, ["interfaces"] = interfaces, ["enums"] = enums, ["lines"] = (double)lines.Length };
                }),
                ["search"] = new NSLBuiltinFunction("search", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("code.search() requires path/content and pattern");
                    var input = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var results = new List<object?>();
                    var lines = content.Split('\n');
                    for (int i = 0; i < lines.Length; i++) {
                        try { if (lines[i].Contains(pattern) || System.Text.RegularExpressions.Regex.IsMatch(lines[i], pattern)) results.Add(new Dictionary<string, object?> { ["line"] = (double)(i + 1), ["text"] = lines[i].TrimEnd('\r') }); } catch { if (lines[i].Contains(pattern)) results.Add(new Dictionary<string, object?> { ["line"] = (double)(i + 1), ["text"] = lines[i].TrimEnd('\r') }); }
                    }
                    return results;
                }),
                ["stats"] = new NSLBuiltinFunction("stats", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.stats() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    var lines = content.Split('\n');
                    int blank = 0, comment = 0, codeLines = 0;
                    foreach (var line in lines) {
                        var trimmed = line.Trim();
                        if (string.IsNullOrEmpty(trimmed)) blank++;
                        else if (trimmed.StartsWith("//") || trimmed.StartsWith("#") || trimmed.StartsWith("/*") || trimmed.StartsWith("*")) comment++;
                        else codeLines++;
                    }
                    return new Dictionary<string, object?> { ["total"] = (double)lines.Length, ["code"] = (double)codeLines, ["comment"] = (double)comment, ["blank"] = (double)blank };
                }),
                ["lines"] = new NSLBuiltinFunction("lines", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.lines() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    return content.Split('\n').Select(l => (object?)l.TrimEnd('\r')).ToList();
                }),
                ["lineCount"] = new NSLBuiltinFunction("lineCount", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("code.lineCount() requires path or content");
                    var input = args[0]?.ToString() ?? "";
                    var content = System.IO.File.Exists(input) ? System.IO.File.ReadAllText(input) : input;
                    return (double)content.Split('\n').Length;
                })
            };
            _globals["code"] = codeNamespace;

            // ===== HTTP NAMESPACE =====
            var httpNamespace = new Dictionary<string, object?>
            {
                ["get"] = new NSLBuiltinFunction("get", (args) => { if (args.Length < 1) throw new NSLRuntimeException("http.get() requires a URL"); var url = args[0]?.ToString() ?? ""; using var client = new System.Net.Http.HttpClient(); client.Timeout = TimeSpan.FromSeconds(args.Length > 1 ? ConvertToNumber(args[1]) : 30); try { var response = client.GetAsync(url).Result; var body = response.Content.ReadAsStringAsync().Result; return new Dictionary<string, object?> { ["status"] = (double)response.StatusCode, ["ok"] = response.IsSuccessStatusCode, ["body"] = body, ["headers"] = response.Headers.ToDictionary(h => h.Key, h => (object?)string.Join(", ", h.Value)) }; } catch (Exception ex) { throw new NSLRuntimeException($"HTTP GET failed: {ex.Message}"); } }),
                ["post"] = new NSLBuiltinFunction("post", (args) => { if (args.Length < 2) throw new NSLRuntimeException("http.post() requires URL and body"); var url = args[0]?.ToString() ?? ""; var body = args[1]?.ToString() ?? ""; var contentType = args.Length > 2 ? args[2]?.ToString() ?? "application/json" : "application/json"; using var client = new System.Net.Http.HttpClient(); client.Timeout = TimeSpan.FromSeconds(30); try { var content = new System.Net.Http.StringContent(body, System.Text.Encoding.UTF8, contentType); var response = client.PostAsync(url, content).Result; var respBody = response.Content.ReadAsStringAsync().Result; return new Dictionary<string, object?> { ["status"] = (double)response.StatusCode, ["ok"] = response.IsSuccessStatusCode, ["body"] = respBody }; } catch (Exception ex) { throw new NSLRuntimeException($"HTTP POST failed: {ex.Message}"); } }),
                ["download"] = new NSLBuiltinFunction("download", (args) => { if (args.Length < 2) throw new NSLRuntimeException("http.download() requires URL and destination path"); var url = args[0]?.ToString() ?? ""; var dest = args[1]?.ToString() ?? ""; using var client = new System.Net.Http.HttpClient(); try { var bytes = client.GetByteArrayAsync(url).Result; System.IO.File.WriteAllBytes(dest, bytes); return true; } catch (Exception ex) { throw new NSLRuntimeException($"HTTP download failed: {ex.Message}"); } }),
                ["head"] = new NSLBuiltinFunction("head", (args) => { if (args.Length < 1) throw new NSLRuntimeException("http.head() requires a URL"); var url = args[0]?.ToString() ?? ""; using var client = new System.Net.Http.HttpClient(); try { var request = new System.Net.Http.HttpRequestMessage(System.Net.Http.HttpMethod.Head, url); var response = client.SendAsync(request).Result; return new Dictionary<string, object?> { ["status"] = (double)response.StatusCode, ["ok"] = response.IsSuccessStatusCode, ["headers"] = response.Headers.ToDictionary(h => h.Key, h => (object?)string.Join(", ", h.Value)) }; } catch (Exception ex) { throw new NSLRuntimeException($"HTTP HEAD failed: {ex.Message}"); } })
            };
            _globals["http"] = httpNamespace;

            // ===== GITHUB NAMESPACE =====
            // GitHub API integration - uses encrypted token from git-on command
            var githubNamespace = new Dictionary<string, object?>
            {
                ["enabled"] = new NSLBuiltinFunction("enabled", (args) => {
                    var configPath = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nsl", "github", "config.json");
                    if (!System.IO.File.Exists(configPath)) return false;
                    try { var json = System.IO.File.ReadAllText(configPath); return json.Contains("\"Enabled\": true") || json.Contains("\"Enabled\":true"); } catch { return false; }
                }),
                ["user"] = new NSLBuiltinFunction("user", (args) => {
                    var token = GetGitHubToken();
                    if (token == null) throw new NSLRuntimeException("GitHub not connected. Run 'git-on' in terminal first.");
                    using var client = new System.Net.Http.HttpClient();
                    client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
                    client.DefaultRequestHeaders.Add("User-Agent", "NSL");
                    client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");
                    var response = client.GetAsync("https://api.github.com/user").Result;
                    if (!response.IsSuccessStatusCode) throw new NSLRuntimeException($"GitHub API error: {response.StatusCode}");
                    return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object?>>(response.Content.ReadAsStringAsync().Result);
                }),
                ["repos"] = new NSLBuiltinFunction("repos", (args) => {
                    var token = GetGitHubToken();
                    if (token == null) throw new NSLRuntimeException("GitHub not connected. Run 'git-on' in terminal first.");
                    using var client = new System.Net.Http.HttpClient();
                    client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
                    client.DefaultRequestHeaders.Add("User-Agent", "NSL");
                    client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");
                    var response = client.GetAsync("https://api.github.com/user/repos?per_page=100").Result;
                    if (!response.IsSuccessStatusCode) throw new NSLRuntimeException($"GitHub API error: {response.StatusCode}");
                    var repos = System.Text.Json.JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(response.Content.ReadAsStringAsync().Result);
                    return repos?.Select(r => r.GetValueOrDefault("full_name")?.ToString()).Where(n => n != null).ToList() ?? new List<string?>();
                }),
                ["repo"] = new NSLBuiltinFunction("repo", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("github.repo(owner/name) requires repo path");
                    var repo = args[0]?.ToString() ?? "";
                    var token = GetGitHubToken();
                    if (token == null) throw new NSLRuntimeException("GitHub not connected. Run 'git-on' in terminal first.");
                    using var client = new System.Net.Http.HttpClient();
                    client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
                    client.DefaultRequestHeaders.Add("User-Agent", "NSL");
                    client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");
                    var response = client.GetAsync($"https://api.github.com/repos/{repo}").Result;
                    if (!response.IsSuccessStatusCode) throw new NSLRuntimeException($"GitHub API error: {response.StatusCode}");
                    return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object?>>(response.Content.ReadAsStringAsync().Result);
                }),
                ["releases"] = new NSLBuiltinFunction("releases", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("github.releases(owner/repo) requires repo path");
                    var repo = args[0]?.ToString() ?? "";
                    var token = GetGitHubToken();
                    if (token == null) throw new NSLRuntimeException("GitHub not connected. Run 'git-on' in terminal first.");
                    using var client = new System.Net.Http.HttpClient();
                    client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
                    client.DefaultRequestHeaders.Add("User-Agent", "NSL");
                    client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");
                    var response = client.GetAsync($"https://api.github.com/repos/{repo}/releases").Result;
                    if (!response.IsSuccessStatusCode) throw new NSLRuntimeException($"GitHub API error: {response.StatusCode}");
                    return System.Text.Json.JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(response.Content.ReadAsStringAsync().Result);
                }),
                ["createRelease"] = new NSLBuiltinFunction("createRelease", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("github.createRelease(repo, tag, name, body?) requires repo, tag, name");
                    var repo = args[0]?.ToString() ?? "";
                    var tag = args[1]?.ToString() ?? "";
                    var name = args[2]?.ToString() ?? "";
                    var body = args.Length > 3 ? args[3]?.ToString() ?? "" : "";
                    var token = GetGitHubToken();
                    if (token == null) throw new NSLRuntimeException("GitHub not connected. Run 'git-on' in terminal first.");
                    using var client = new System.Net.Http.HttpClient();
                    client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
                    client.DefaultRequestHeaders.Add("User-Agent", "NSL");
                    client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");
                    var payload = System.Text.Json.JsonSerializer.Serialize(new { tag_name = tag, name = name, body = body, draft = false, prerelease = false });
                    var content = new System.Net.Http.StringContent(payload, System.Text.Encoding.UTF8, "application/json");
                    var response = client.PostAsync($"https://api.github.com/repos/{repo}/releases", content).Result;
                    if (!response.IsSuccessStatusCode) throw new NSLRuntimeException($"GitHub API error: {response.StatusCode} - {response.Content.ReadAsStringAsync().Result}");
                    return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object?>>(response.Content.ReadAsStringAsync().Result);
                }),
                ["issues"] = new NSLBuiltinFunction("issues", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("github.issues(owner/repo) requires repo path");
                    var repo = args[0]?.ToString() ?? "";
                    var state = args.Length > 1 ? args[1]?.ToString() ?? "open" : "open";
                    var token = GetGitHubToken();
                    if (token == null) throw new NSLRuntimeException("GitHub not connected. Run 'git-on' in terminal first.");
                    using var client = new System.Net.Http.HttpClient();
                    client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
                    client.DefaultRequestHeaders.Add("User-Agent", "NSL");
                    client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");
                    var response = client.GetAsync($"https://api.github.com/repos/{repo}/issues?state={state}").Result;
                    if (!response.IsSuccessStatusCode) throw new NSLRuntimeException($"GitHub API error: {response.StatusCode}");
                    return System.Text.Json.JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(response.Content.ReadAsStringAsync().Result);
                }),
                ["prs"] = new NSLBuiltinFunction("prs", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("github.prs(owner/repo) requires repo path");
                    var repo = args[0]?.ToString() ?? "";
                    var state = args.Length > 1 ? args[1]?.ToString() ?? "open" : "open";
                    var token = GetGitHubToken();
                    if (token == null) throw new NSLRuntimeException("GitHub not connected. Run 'git-on' in terminal first.");
                    using var client = new System.Net.Http.HttpClient();
                    client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
                    client.DefaultRequestHeaders.Add("User-Agent", "NSL");
                    client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");
                    var response = client.GetAsync($"https://api.github.com/repos/{repo}/pulls?state={state}").Result;
                    if (!response.IsSuccessStatusCode) throw new NSLRuntimeException($"GitHub API error: {response.StatusCode}");
                    return System.Text.Json.JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(response.Content.ReadAsStringAsync().Result);
                })
            };
            _globals["github"] = githubNamespace;

            // ===== HTML NAMESPACE (Web/UI Generation) =====
            var htmlNamespace = new Dictionary<string, object?>
            {
                // Generate HTML tag with attributes and content
                ["tag"] = new NSLBuiltinFunction("tag", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("html.tag(name, attrs?, content?) requires tag name");
                    var name = args[0]?.ToString() ?? "div";
                    var attrs = args.Length > 1 && args[1] is IDictionary<string, object?> a ? a : new Dictionary<string, object?>();
                    var content = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
                    var attrStr = string.Join(" ", attrs.Select(kv => $"{kv.Key}=\"{kv.Value}\""));
                    var selfClosing = new[] { "br", "hr", "img", "input", "meta", "link" };
                    if (selfClosing.Contains(name.ToLower()) && string.IsNullOrEmpty(content))
                        return $"<{name}{(attrStr.Length > 0 ? " " + attrStr : "")} />";
                    return $"<{name}{(attrStr.Length > 0 ? " " + attrStr : "")}>{content}</{name}>";
                }),
                // Common tag shortcuts
                ["div"] = new NSLBuiltinFunction("div", (args) => {
                    var attrs = args.Length > 0 && args[0] is IDictionary<string, object?> a ? a : new Dictionary<string, object?>();
                    var content = args.Length > 1 ? args[1]?.ToString() : (args.Length > 0 && !(args[0] is IDictionary<string, object?>) ? args[0]?.ToString() : "");
                    var attrStr = string.Join(" ", attrs.Select(kv => $"{kv.Key}=\"{kv.Value}\""));
                    return $"<div{(attrStr.Length > 0 ? " " + attrStr : "")}>{content ?? ""}</div>";
                }),
                ["span"] = new NSLBuiltinFunction("span", (args) => {
                    var content = args.Length > 0 ? args[0]?.ToString() ?? "" : "";
                    var cls = args.Length > 1 ? $" class=\"{args[1]}\"" : "";
                    return $"<span{cls}>{content}</span>";
                }),
                ["p"] = new NSLBuiltinFunction("p", (args) => $"<p>{(args.Length > 0 ? args[0]?.ToString() ?? "" : "")}</p>"),
                ["h1"] = new NSLBuiltinFunction("h1", (args) => $"<h1>{(args.Length > 0 ? args[0]?.ToString() ?? "" : "")}</h1>"),
                ["h2"] = new NSLBuiltinFunction("h2", (args) => $"<h2>{(args.Length > 0 ? args[0]?.ToString() ?? "" : "")}</h2>"),
                ["h3"] = new NSLBuiltinFunction("h3", (args) => $"<h3>{(args.Length > 0 ? args[0]?.ToString() ?? "" : "")}</h3>"),
                ["a"] = new NSLBuiltinFunction("a", (args) => {
                    if (args.Length < 1) return "<a></a>";
                    var href = args[0]?.ToString() ?? "#";
                    var text = args.Length > 1 ? args[1]?.ToString() ?? href : href;
                    return $"<a href=\"{href}\">{text}</a>";
                }),
                ["img"] = new NSLBuiltinFunction("img", (args) => {
                    if (args.Length < 1) return "<img />";
                    var src = args[0]?.ToString() ?? "";
                    var alt = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    return $"<img src=\"{src}\" alt=\"{alt}\" />";
                }),
                ["ul"] = new NSLBuiltinFunction("ul", (args) => {
                    if (args.Length < 1 || !(args[0] is IList<object?> items)) return "<ul></ul>";
                    var lis = string.Join("", items.Select(i => $"<li>{i}</li>"));
                    return $"<ul>{lis}</ul>";
                }),
                ["ol"] = new NSLBuiltinFunction("ol", (args) => {
                    if (args.Length < 1 || !(args[0] is IList<object?> items)) return "<ol></ol>";
                    var lis = string.Join("", items.Select(i => $"<li>{i}</li>"));
                    return $"<ol>{lis}</ol>";
                }),
                ["table"] = new NSLBuiltinFunction("table", (args) => {
                    if (args.Length < 1 || !(args[0] is IList<object?> rows)) return "<table></table>";
                    var headers = args.Length > 1 && args[1] is IList<object?> h ? h : null;
                    var sb = new System.Text.StringBuilder("<table>");
                    if (headers != null) {
                        sb.Append("<thead><tr>");
                        foreach (var th in headers) sb.Append($"<th>{th}</th>");
                        sb.Append("</tr></thead>");
                    }
                    sb.Append("<tbody>");
                    foreach (var row in rows) {
                        sb.Append("<tr>");
                        if (row is IList<object?> cells) foreach (var cell in cells) sb.Append($"<td>{cell}</td>");
                        else if (row is IDictionary<string, object?> dict) foreach (var kv in dict) sb.Append($"<td>{kv.Value}</td>");
                        sb.Append("</tr>");
                    }
                    sb.Append("</tbody></table>");
                    return sb.ToString();
                }),
                ["form"] = new NSLBuiltinFunction("form", (args) => {
                    var action = args.Length > 0 ? args[0]?.ToString() ?? "" : "";
                    var method = args.Length > 1 ? args[1]?.ToString() ?? "post" : "post";
                    var content = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
                    return $"<form action=\"{action}\" method=\"{method}\">{content}</form>";
                }),
                ["input"] = new NSLBuiltinFunction("input", (args) => {
                    var type = args.Length > 0 ? args[0]?.ToString() ?? "text" : "text";
                    var name = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    var placeholder = args.Length > 2 ? $" placeholder=\"{args[2]}\"" : "";
                    return $"<input type=\"{type}\" name=\"{name}\"{placeholder} />";
                }),
                ["button"] = new NSLBuiltinFunction("button", (args) => {
                    var text = args.Length > 0 ? args[0]?.ToString() ?? "Submit" : "Submit";
                    var type = args.Length > 1 ? args[1]?.ToString() ?? "submit" : "submit";
                    return $"<button type=\"{type}\">{text}</button>";
                }),
                // Full HTML document
                ["document"] = new NSLBuiltinFunction("document", (args) => {
                    var title = args.Length > 0 ? args[0]?.ToString() ?? "NSL Page" : "NSL Page";
                    var body = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    var css = args.Length > 2 ? $"<style>{args[2]}</style>" : "";
                    return $"<!DOCTYPE html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{title}</title>{css}</head><body>{body}</body></html>";
                }),
                // Escape HTML entities
                ["escape"] = new NSLBuiltinFunction("escape", (args) => {
                    if (args.Length < 1) return "";
                    return System.Net.WebUtility.HtmlEncode(args[0]?.ToString() ?? "");
                }),
                // Join multiple HTML fragments
                ["join"] = new NSLBuiltinFunction("join", (args) => {
                    if (args.Length < 1) return "";
                    if (args[0] is IList<object?> items) return string.Join("", items.Select(i => i?.ToString() ?? ""));
                    return string.Join("", args.Select(a => a?.ToString() ?? ""));
                })
            };
            _globals["html"] = htmlNamespace;

            // ===== WEB NAMESPACE (HTTP Server) =====
            var webServerListener = new System.Collections.Concurrent.ConcurrentDictionary<int, System.Net.HttpListener>();
            var webNamespace = new Dictionary<string, object?>
            {
                // Start a simple HTTP server
                ["serve"] = new NSLBuiltinFunction("serve", (args) => {
                    var port = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 8080;
                    var handler = args.Length > 1 ? args[1] : null;
                    
                    if (webServerListener.ContainsKey(port)) {
                        return new Dictionary<string, object?> { ["error"] = $"Server already running on port {port}" };
                    }
                    
                    var listener = new System.Net.HttpListener();
                    listener.Prefixes.Add($"http://localhost:{port}/");
                    
                    try {
                        listener.Start();
                        webServerListener[port] = listener;
                        
                        // Start async handler
                        Task.Run(() => {
                            while (listener.IsListening) {
                                try {
                                    var context = listener.GetContext();
                                    var request = context.Request;
                                    var response = context.Response;
                                    
                                    string responseString = "<!DOCTYPE html><html><body><h1>NSL Web Server</h1><p>Server running on port " + port + "</p></body></html>";
                                    
                                    if (handler is NSLFunction fn) {
                                        try {
                                            var reqInfo = new Dictionary<string, object?> {
                                                ["method"] = request.HttpMethod,
                                                ["path"] = request.Url?.AbsolutePath ?? "/",
                                                ["query"] = request.Url?.Query ?? "",
                                                ["headers"] = request.Headers.AllKeys.ToDictionary(k => k ?? "", k => (object?)(request.Headers[k] ?? ""))
                                            };
                                            var result = InvokeCallable(fn, new object?[] { reqInfo });
                                            responseString = result?.ToString() ?? "";
                                        } catch { }
                                    }
                                    
                                    byte[] buffer = System.Text.Encoding.UTF8.GetBytes(responseString);
                                    response.ContentLength64 = buffer.Length;
                                    response.ContentType = "text/html; charset=utf-8";
                                    response.OutputStream.Write(buffer, 0, buffer.Length);
                                    response.OutputStream.Close();
                                } catch { }
                            }
                        });
                        
                        return new Dictionary<string, object?> { 
                            ["success"] = true, 
                            ["port"] = (double)port,
                            ["url"] = $"http://localhost:{port}/"
                        };
                    } catch (Exception ex) {
                        return new Dictionary<string, object?> { ["error"] = ex.Message };
                    }
                }),
                // Stop server
                ["stop"] = new NSLBuiltinFunction("stop", (args) => {
                    var port = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 8080;
                    if (webServerListener.TryRemove(port, out var listener)) {
                        listener.Stop();
                        listener.Close();
                        return new Dictionary<string, object?> { ["stopped"] = true, ["port"] = (double)port };
                    }
                    return new Dictionary<string, object?> { ["stopped"] = false, ["error"] = "No server on that port" };
                }),
                // List running servers
                ["servers"] = new NSLBuiltinFunction("servers", (args) => {
                    return webServerListener.Keys.Select(p => (object?)(double)p).ToList();
                }),
                // Serve static files from directory
                ["static"] = new NSLBuiltinFunction("static", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("web.static(directory, port?) requires directory");
                    var dir = args[0]?.ToString() ?? ".";
                    var port = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 8080;
                    
                    if (!Directory.Exists(dir)) throw new NSLRuntimeException($"Directory not found: {dir}");
                    
                    var listener = new System.Net.HttpListener();
                    listener.Prefixes.Add($"http://localhost:{port}/");
                    
                    try {
                        listener.Start();
                        webServerListener[port] = listener;
                        
                        Task.Run(() => {
                            while (listener.IsListening) {
                                try {
                                    var context = listener.GetContext();
                                    var path = context.Request.Url?.AbsolutePath?.TrimStart('/') ?? "index.html";
                                    if (string.IsNullOrEmpty(path)) path = "index.html";
                                    var filePath = Path.Combine(dir, path);
                                    
                                    if (File.Exists(filePath)) {
                                        var ext = Path.GetExtension(filePath).ToLower();
                                        var contentType = ext switch {
                                            ".html" => "text/html",
                                            ".css" => "text/css",
                                            ".js" => "application/javascript",
                                            ".json" => "application/json",
                                            ".png" => "image/png",
                                            ".jpg" or ".jpeg" => "image/jpeg",
                                            ".gif" => "image/gif",
                                            ".svg" => "image/svg+xml",
                                            ".ico" => "image/x-icon",
                                            _ => "application/octet-stream"
                                        };
                                        var bytes = File.ReadAllBytes(filePath);
                                        context.Response.ContentType = contentType;
                                        context.Response.ContentLength64 = bytes.Length;
                                        context.Response.OutputStream.Write(bytes, 0, bytes.Length);
                                    } else {
                                        context.Response.StatusCode = 404;
                                        var msg = System.Text.Encoding.UTF8.GetBytes("404 Not Found");
                                        context.Response.OutputStream.Write(msg, 0, msg.Length);
                                    }
                                    context.Response.OutputStream.Close();
                                } catch { }
                            }
                        });
                        
                        return new Dictionary<string, object?> { 
                            ["success"] = true, 
                            ["port"] = (double)port,
                            ["directory"] = dir,
                            ["url"] = $"http://localhost:{port}/"
                        };
                    } catch (Exception ex) {
                        return new Dictionary<string, object?> { ["error"] = ex.Message };
                    }
                })
            };
            _globals["web"] = webNamespace;

            // ===== JSON NAMESPACE =====
            var jsonNamespace = new Dictionary<string, object?>
            {
                ["parse"] = new NSLBuiltinFunction("parse", (args) => { 
                    if (args.Length < 1) throw new NSLRuntimeException("json.parse() requires a JSON string"); 
                    var jsonStr = args[0]?.ToString() ?? ""; 
                    try { 
                        using var doc = System.Text.Json.JsonDocument.Parse(jsonStr);
                        return ConvertJsonElement(doc.RootElement);
                    } catch (Exception ex) { 
                        throw new NSLRuntimeException($"JSON parse failed: {ex.Message}"); 
                    } 
                }),
                ["stringify"] = new NSLBuiltinFunction("stringify", (args) => { if (args.Length < 1) throw new NSLRuntimeException("json.stringify() requires an object"); var indent = args.Length > 1 && IsTruthy(args[1]); try { return System.Text.Json.JsonSerializer.Serialize(args[0], new System.Text.Json.JsonSerializerOptions { WriteIndented = indent }); } catch (Exception ex) { throw new NSLRuntimeException($"JSON stringify failed: {ex.Message}"); } }),
                ["pretty"] = new NSLBuiltinFunction("pretty", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("json.pretty() requires a JSON string or object");
                    var input = args[0];
                    try {
                        // If input is already an object (dict, list, etc), serialize it directly
                        if (input is Dictionary<string, object?> || input is IList<object?> || (input != null && !(input is string)))
                        {
                            return System.Text.Json.JsonSerializer.Serialize(input, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                        }
                        // Otherwise treat as JSON string and re-format
                        var jsonStr = input?.ToString() ?? "";
                        using var doc = System.Text.Json.JsonDocument.Parse(jsonStr);
                        return System.Text.Json.JsonSerializer.Serialize(doc.RootElement, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                    } catch (Exception ex) {
                        throw new NSLRuntimeException($"JSON pretty failed: {ex.Message}. Hint: Pass a JSON string or object.");
                    }
                }),
                ["valid"] = new NSLBuiltinFunction("valid", (args) => { if (args.Length < 1) return false; var jsonStr = args[0]?.ToString() ?? ""; try { System.Text.Json.JsonDocument.Parse(jsonStr); return true; } catch { return false; } })
            };
            _globals["json"] = jsonNamespace;
            // ===== NSLZ NAMESPACE (Native Compression) =====
            var nslzNamespace = new Dictionary<string, object?>
            {
                ["create"] = new NSLBuiltinFunction("create", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("nslz.create(archivePath, files[]) requires archive path and file list");
                    var archivePath = args[0]?.ToString() ?? "";
                    var files = new List<string>();
                    if (args[1] is IList<object?> fileList) foreach (var f in fileList) files.Add(f?.ToString() ?? "");
                    else files.Add(args[1]?.ToString() ?? "");
                    NSLZFormat.Create(archivePath, files.ToArray());
                    return new Dictionary<string, object?> { ["success"] = true, ["path"] = archivePath, ["files"] = (double)files.Count };
                }),
                ["extract"] = new NSLBuiltinFunction("extract", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("nslz.extract(archivePath, outputDir) requires archive and output directory");
                    NSLZFormat.Extract(args[0]?.ToString() ?? "", args[1]?.ToString() ?? "");
                    return new Dictionary<string, object?> { ["success"] = true, ["outputDir"] = args[1]?.ToString() };
                }),
                ["list"] = new NSLBuiltinFunction("list", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("nslz.list(archivePath) requires archive path");
                    var entries = NSLZFormat.List(args[0]?.ToString() ?? "");
                    return entries.Select(e => new Dictionary<string, object?> {
                        ["name"] = e.Name, ["originalSize"] = (double)e.OriginalSize,
                        ["compressedSize"] = (double)e.CompressedSize, ["compression"] = e.Compression.ToString(), ["isText"] = e.IsTextFile
                    }).ToList<object?>();
                }),
                ["info"] = new NSLBuiltinFunction("info", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("nslz.info(archivePath) requires archive path");
                    var info = NSLZFormat.GetInfo(args[0]?.ToString() ?? "");
                    return new Dictionary<string, object?> {
                        ["version"] = (double)info.Version, ["flags"] = info.Flags.ToString(),
                        ["fileCount"] = (double)info.Entries.Count, ["totalOriginalSize"] = (double)info.TotalOriginalSize,
                        ["totalCompressedSize"] = (double)info.TotalCompressedSize, ["compressionRatio"] = info.CompressionRatio
                    };
                }),
                ["createSFX"] = new NSLBuiltinFunction("createSFX", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("nslz.createSFX(outputPath, files[]) requires output path and file list");
                    var outputPath = args[0]?.ToString() ?? "";
                    var files = new List<string>();
                    if (args[1] is IList<object?> fileList) foreach (var f in fileList) files.Add(f?.ToString() ?? "");
                    else files.Add(args[1]?.ToString() ?? "");
                    NSLZSelfExtractor.CreateSFX(outputPath, files.ToArray());
                    return new Dictionary<string, object?> { ["success"] = true, ["path"] = outputPath, ["selfExtracting"] = true };
                }),
                ["compress"] = new NSLBuiltinFunction("compress", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("nslz.compress(text) requires text");
                    return NSLZCompressor.CompressText(args[0]?.ToString() ?? "");
                }),
                ["decompress"] = new NSLBuiltinFunction("decompress", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("nslz.decompress(compressed) requires compressed text");
                    return NSLZCompressor.DecompressText(args[0]?.ToString() ?? "");
                })
            };
            _globals["nslz"] = nslzNamespace;















            // ===== DATAFRAME NAMESPACE (Pandas-like Data Analysis) =====
            var dfNamespace = new Dictionary<string, object?>
            {
                // Create DataFrame from array of objects
                ["create"] = new NSLBuiltinFunction("create", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("df.create(rows[]) requires array of row objects");
                    var inputRows = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be array");
                    var rows = new List<Dictionary<string, object?>>();
                    var columns = new List<string>();
                    
                    foreach (var r in inputRows) {
                        if (r is Dictionary<string, object?> dict) {
                            rows.Add(new Dictionary<string, object?>(dict));
                            if (columns.Count == 0) columns = dict.Keys.ToList();
                        } else if (r is IDictionary<string, object?> idict) {
                            rows.Add(new Dictionary<string, object?>(idict));
                            if (columns.Count == 0) columns = idict.Keys.ToList();
                        }
                    }
                    
                    return new Dictionary<string, object?> {
                        ["_type"] = "DataFrame",
                        ["rows"] = rows.Cast<object?>().ToList(),
                        ["columns"] = columns.Cast<object?>().ToList(),
                        ["length"] = (double)rows.Count
                    };
                }),
                // Load CSV file into DataFrame
                ["readCsv"] = new NSLBuiltinFunction("readCsv", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("df.readCsv(path, [options]) requires file path");
                    var path = args[0]?.ToString() ?? "";
                    var options = args.Length > 1 ? args[1] as Dictionary<string, object?> : null;
                    var delimiter = options?.GetValueOrDefault("delimiter")?.ToString() ?? ",";
                    var hasHeader = !(options?.GetValueOrDefault("noHeader") is bool nh && nh);
                    
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path);
                    if (lines.Length == 0) return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = new List<object?>(), ["columns"] = new List<string>(), ["length"] = 0.0 };
                    
                    var columns = hasHeader ? lines[0].Split(delimiter[0]).Select(c => c.Trim()).ToList() : Enumerable.Range(0, lines[0].Split(delimiter[0]).Length).Select(i => $"col{i}").ToList();
                    var startRow = hasHeader ? 1 : 0;
                    var rows = new List<object?>();
                    
                    for (int i = startRow; i < lines.Length; i++) {
                        var values = lines[i].Split(delimiter[0]);
                        var row = new Dictionary<string, object?>();
                        for (int j = 0; j < Math.Min(columns.Count, values.Length); j++) {
                            var val = values[j].Trim();
                            if (double.TryParse(val, out var num)) row[columns[j]] = num;
                            else if (bool.TryParse(val, out var b)) row[columns[j]] = b;
                            else row[columns[j]] = val;
                        }
                        rows.Add(row);
                    }
                    
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = rows, ["columns"] = columns, ["length"] = (double)rows.Count };
                }),
                // Write DataFrame to CSV
                ["toCsv"] = new NSLBuiltinFunction("toCsv", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.toCsv(df, path) requires DataFrame and file path");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var path = args[1]?.ToString() ?? "";
                    var columns = df["columns"] as IList<object?> ?? new List<object?>();
                    var rows = df["rows"] as IList<object?> ?? new List<object?>();
                    
                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine(string.Join(",", columns.Select(c => c?.ToString() ?? "")));
                    foreach (var row in rows) {
                        if (row is Dictionary<string, object?> r) {
                            sb.AppendLine(string.Join(",", columns.Select(c => r.GetValueOrDefault(c?.ToString() ?? "")?.ToString() ?? "")));
                        }
                    }
                    System.IO.File.WriteAllText(path, sb.ToString());
                    return true;
                }),
                // Filter rows
                ["filter"] = new NSLBuiltinFunction("filter", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("df.filter(df, column, value) or df.filter(df, column, op, value)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var op = args.Length > 3 ? args[2]?.ToString() ?? "==" : "==";
                    var value = args.Length > 3 ? args[3] : args[2];
                    var rows = df.GetValueOrDefault("rows") as IList<object?> ?? new List<object?>();
                    
                    var filtered = new List<object?>();
                    foreach (var row in rows) {
                        var r = row as Dictionary<string, object?>;
                        if (r == null) continue;
                        var cellVal = r.GetValueOrDefault(column);
                        bool match = op switch {
                            "==" => Equals(cellVal, value) || cellVal?.ToString() == value?.ToString(),
                            "!=" => !Equals(cellVal, value) && cellVal?.ToString() != value?.ToString(),
                            ">" => ConvertToNumber(cellVal) > ConvertToNumber(value),
                            ">=" => ConvertToNumber(cellVal) >= ConvertToNumber(value),
                            "<" => ConvertToNumber(cellVal) < ConvertToNumber(value),
                            "<=" => ConvertToNumber(cellVal) <= ConvertToNumber(value),
                            "contains" => cellVal?.ToString()?.Contains(value?.ToString() ?? "") == true,
                            "startswith" => cellVal?.ToString()?.StartsWith(value?.ToString() ?? "") == true,
                            "endswith" => cellVal?.ToString()?.EndsWith(value?.ToString() ?? "") == true,
                            _ => false
                        };
                        if (match) filtered.Add(row);
                    }
                    
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = filtered, ["columns"] = df.GetValueOrDefault("columns"), ["length"] = (double)filtered.Count };
                }),
                // Select specific columns
                ["select"] = new NSLBuiltinFunction("select", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.select(df, columns[])");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var cols = args[1] as IList<object?> ?? new List<object?> { args[1] };
                    var colNames = cols.Select(c => c?.ToString() ?? "").ToList();
                    var rows = df["rows"] as IList<object?> ?? new List<object?>();
                    
                    var selected = rows.Select(row => {
                        if (row is not Dictionary<string, object?> r) return new Dictionary<string, object?>();
                        return colNames.Where(c => r.ContainsKey(c)).ToDictionary(c => c, c => r[c]);
                    }).Cast<object?>().ToList();
                    
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = selected, ["columns"] = colNames, ["length"] = (double)selected.Count };
                }),
                // Sort by column
                ["sort"] = new NSLBuiltinFunction("sort", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.sort(df, column, [desc])");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var desc = args.Length > 2 && IsTruthy(args[2]);
                    var rows = (df["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    
                    var sorted = desc 
                        ? rows.OrderByDescending(r => r.GetValueOrDefault(column)).ToList()
                        : rows.OrderBy(r => r.GetValueOrDefault(column)).ToList();
                    
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = sorted.Cast<object?>().ToList(), ["columns"] = df["columns"], ["length"] = (double)sorted.Count };
                }),
                // Group by column and aggregate
                ["groupBy"] = new NSLBuiltinFunction("groupBy", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.groupBy(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var rows = (df["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    
                    var groups = rows.GroupBy(r => r.GetValueOrDefault(column)?.ToString() ?? "").ToDictionary(
                        g => g.Key,
                        g => (object?)new Dictionary<string, object?> {
                            ["_type"] = "DataFrame",
                            ["rows"] = g.Cast<object?>().ToList(),
                            ["columns"] = df["columns"],
                            ["length"] = (double)g.Count()
                        }
                    );
                    
                    return groups;
                }),
                // Aggregate functions
                ["sum"] = new NSLBuiltinFunction("sum", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.sum(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var rows = df.GetValueOrDefault("rows") as IList<object?> ?? new List<object?>();
                    double sum = 0;
                    foreach (var row in rows) {
                        if (row is Dictionary<string, object?> r) sum += ConvertToNumber(r.GetValueOrDefault(column));
                    }
                    return sum;
                }),
                ["mean"] = new NSLBuiltinFunction("mean", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.mean(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var rows = df.GetValueOrDefault("rows") as IList<object?> ?? new List<object?>();
                    var values = new List<double>();
                    foreach (var row in rows) {
                        if (row is Dictionary<string, object?> r) values.Add(ConvertToNumber(r.GetValueOrDefault(column)));
                    }
                    return values.Count > 0 ? values.Average() : 0.0;
                }),
                ["min"] = new NSLBuiltinFunction("min", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.min(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var rows = df.GetValueOrDefault("rows") as IList<object?> ?? new List<object?>();
                    var values = new List<double>();
                    foreach (var row in rows) {
                        if (row is Dictionary<string, object?> r) values.Add(ConvertToNumber(r.GetValueOrDefault(column)));
                    }
                    return values.Count > 0 ? values.Min() : 0.0;
                }),
                ["max"] = new NSLBuiltinFunction("max", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.max(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var rows = df.GetValueOrDefault("rows") as IList<object?> ?? new List<object?>();
                    var values = new List<double>();
                    foreach (var row in rows) {
                        if (row is Dictionary<string, object?> r) values.Add(ConvertToNumber(r.GetValueOrDefault(column)));
                    }
                    return values.Count > 0 ? values.Max() : 0.0;
                }),
                ["count"] = new NSLBuiltinFunction("count", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("df.count(df)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    return df.GetValueOrDefault("length") ?? 0.0;
                }),
                // Statistical describe
                ["describe"] = new NSLBuiltinFunction("describe", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("df.describe(df)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var columns = (df["columns"] as IList<object?> ?? new List<object?>()).Select(c => c?.ToString() ?? "").ToList();
                    var rows = (df["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    
                    var stats = new Dictionary<string, object?>();
                    foreach (var col in columns) {
                        var values = rows.Select(r => r.GetValueOrDefault(col)).Where(v => v is double).Cast<double>().ToList();
                        if (values.Count > 0) {
                            var mean = values.Average();
                            var variance = values.Sum(v => Math.Pow(v - mean, 2)) / values.Count;
                            stats[col] = new Dictionary<string, object?> {
                                ["count"] = (double)values.Count,
                                ["mean"] = mean,
                                ["std"] = Math.Sqrt(variance),
                                ["min"] = values.Min(),
                                ["max"] = values.Max(),
                                ["sum"] = values.Sum()
                            };
                        } else {
                            stats[col] = new Dictionary<string, object?> { ["type"] = "non-numeric", ["unique"] = (double)rows.Select(r => r.GetValueOrDefault(col)?.ToString()).Distinct().Count() };
                        }
                    }
                    return stats;
                }),
                // Head/Tail
                ["head"] = new NSLBuiltinFunction("head", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("df.head(df, [n])");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var n = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 5;
                    var allRows = df.GetValueOrDefault("rows") as IList<object?> ?? new List<object?>();
                    var rows = allRows.Take(n).ToList();
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = rows, ["columns"] = df.GetValueOrDefault("columns"), ["length"] = (double)rows.Count };
                }),
                ["tail"] = new NSLBuiltinFunction("tail", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("df.tail(df, [n])");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var n = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 5;
                    var allRows = df.GetValueOrDefault("rows") as IList<object?> ?? new List<object?>();
                    var rows = allRows.Skip(Math.Max(0, allRows.Count - n)).ToList();
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = rows, ["columns"] = df.GetValueOrDefault("columns"), ["length"] = (double)rows.Count };
                }),
                // Unique values in column
                ["unique"] = new NSLBuiltinFunction("unique", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.unique(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var rows = (df["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    return rows.Select(r => r.GetValueOrDefault(column)).Distinct().ToList();
                }),
                // Value counts
                ["valueCounts"] = new NSLBuiltinFunction("valueCounts", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.valueCounts(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var column = args[1]?.ToString() ?? "";
                    var rows = (df["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    return rows.GroupBy(r => r.GetValueOrDefault(column)?.ToString() ?? "").ToDictionary(g => g.Key, g => (object?)(double)g.Count());
                }),
                // Add/rename column
                ["addColumn"] = new NSLBuiltinFunction("addColumn", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("df.addColumn(df, name, value|fn)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var colName = args[1]?.ToString() ?? "";
                    var value = args[2];
                    var rows = (df["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    var columns = (df["columns"] as IList<object?> ?? new List<object?>()).Select(c => c?.ToString() ?? "").ToList();
                    if (!columns.Contains(colName)) columns.Add(colName);
                    
                    foreach (var row in rows) { row[colName] = value; }
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = rows.Cast<object?>().ToList(), ["columns"] = columns, ["length"] = (double)rows.Count };
                }),
                // Drop column
                ["dropColumn"] = new NSLBuiltinFunction("dropColumn", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("df.dropColumn(df, column)");
                    var df = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var colName = args[1]?.ToString() ?? "";
                    var rows = (df["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    var columns = (df["columns"] as IList<object?> ?? new List<object?>()).Select(c => c?.ToString() ?? "").Where(c => c != colName).ToList();
                    
                    foreach (var row in rows) { row.Remove(colName); }
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = rows.Cast<object?>().ToList(), ["columns"] = columns, ["length"] = (double)rows.Count };
                }),
                // Merge/Join two DataFrames
                ["merge"] = new NSLBuiltinFunction("merge", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("df.merge(df1, df2, onColumn)");
                    var df1 = args[0] as Dictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be DataFrame");
                    var df2 = args[1] as Dictionary<string, object?> ?? throw new NSLRuntimeException("Second argument must be DataFrame");
                    var onCol = args[2]?.ToString() ?? "";
                    
                    var rows1 = (df1["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    var rows2 = (df2["rows"] as IList<object?> ?? new List<object?>()).Cast<Dictionary<string, object?>>().ToList();
                    var cols1 = (df1["columns"] as IList<object?> ?? new List<object?>()).Select(c => c?.ToString() ?? "").ToList();
                    var cols2 = (df2["columns"] as IList<object?> ?? new List<object?>()).Select(c => c?.ToString() ?? "").Where(c => c != onCol).ToList();
                    
                    var lookup = rows2.ToLookup(r => r.GetValueOrDefault(onCol)?.ToString() ?? "");
                    var merged = new List<object?>();
                    
                    foreach (var r1 in rows1) {
                        var key = r1.GetValueOrDefault(onCol)?.ToString() ?? "";
                        foreach (var r2 in lookup[key]) {
                            var newRow = new Dictionary<string, object?>(r1);
                            foreach (var c in cols2) { newRow[c] = r2.GetValueOrDefault(c); }
                            merged.Add(newRow);
                        }
                    }
                    
                    return new Dictionary<string, object?> { ["_type"] = "DataFrame", ["rows"] = merged, ["columns"] = cols1.Concat(cols2).ToList(), ["length"] = (double)merged.Count };
                })
            };
            _globals["df"] = dfNamespace;

            // ===== YAML NAMESPACE (Simple Key-Value Parser) =====
            var yamlNamespace = new Dictionary<string, object?>
            {
                ["parse"] = new NSLBuiltinFunction("parse", (args) => { 
                    if (args.Length < 1) throw new NSLRuntimeException("yaml.parse() requires a YAML string");
                    var yamlStr = args[0]?.ToString() ?? "";
                    var result = new Dictionary<string, object?>();
                    var lines = yamlStr.Split('\n');
                    // var currentIndent = 0; // Removed: unused
                    var stack = new Stack<(int indent, Dictionary<string, object?> dict)>();
                    stack.Push((0, result));
                    
                    foreach (var rawLine in lines)
                    {
                        var line = rawLine.TrimEnd('\r');
                        if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith("#")) continue;
                        
                        var indent = line.Length - line.TrimStart().Length;
                        var content = line.Trim();
                        
                        // Handle key: value
                        var colonIdx = content.IndexOf(':');
                        if (colonIdx > 0)
                        {
                            var key = content.Substring(0, colonIdx).Trim();
                            var value = colonIdx < content.Length - 1 ? content.Substring(colonIdx + 1).Trim() : "";
                            
                            // Pop to correct level
                            while (stack.Count > 1 && stack.Peek().indent >= indent) stack.Pop();
                            
                            if (string.IsNullOrEmpty(value))
                            {
                                // Nested object
                                var nested = new Dictionary<string, object?>();
                                stack.Peek().dict[key] = nested;
                                stack.Push((indent, nested));
                            }
                            else
                            {
                                // Parse value
                                object? parsedValue = value;
                                if (value == "true") parsedValue = true;
                                else if (value == "false") parsedValue = false;
                                else if (value == "null") parsedValue = null;
                                else if (double.TryParse(value, out var num)) parsedValue = num;
                                else if (value.StartsWith(""") && value.EndsWith(""")) parsedValue = value.Substring(1, value.Length - 2);
                                else if (value.StartsWith("'") && value.EndsWith("'")) parsedValue = value.Substring(1, value.Length - 2);
                                
                                stack.Peek().dict[key] = parsedValue;
                            }
                        }
                    }
                    return result;
                }),
                ["stringify"] = new NSLBuiltinFunction("stringify", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("yaml.stringify() requires an object");
                    var obj = args[0];
                    var sb = new System.Text.StringBuilder();
                    void WriteYaml(object? value, int indent) {
                        var prefix = new string(' ', indent * 2);
                        if (value is Dictionary<string, object?> dict) {
                            foreach (var kvp in dict) {
                                if (kvp.Value is Dictionary<string, object?> nested) {
                                    sb.AppendLine($"{prefix}{kvp.Key}:");
                                    WriteYaml(nested, indent + 1);
                                } else {
                                    var v = kvp.Value == null ? "null" : kvp.Value is bool b ? b.ToString().ToLower() : kvp.Value.ToString();
                                    sb.AppendLine($"{prefix}{kvp.Key}: {v}");
                                }
                            }
                        }
                    }
                    WriteYaml(obj, 0);
                    return sb.ToString();
                })
            };
            _globals["yaml"] = yamlNamespace;

            // ===== DIFF NAMESPACE =====
            var diffNamespace = new Dictionary<string, object?>
            {
                ["lines"] = new NSLBuiltinFunction("lines", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("diff.lines() requires two strings");
                    var old = (args[0]?.ToString() ?? "").Split('\n');
                    var newStr = (args[1]?.ToString() ?? "").Split('\n');
                    var result = new List<object?>();
                    
                    int i = 0, j = 0;
                    while (i < old.Length || j < newStr.Length) {
                        if (i >= old.Length) {
                            result.Add(new Dictionary<string, object?> { ["type"] = "add", ["line"] = j + 1, ["content"] = newStr[j] });
                            j++;
                        } else if (j >= newStr.Length) {
                            result.Add(new Dictionary<string, object?> { ["type"] = "remove", ["line"] = i + 1, ["content"] = old[i] });
                            i++;
                        } else if (old[i].TrimEnd('\r') == newStr[j].TrimEnd('\r')) {
                            i++; j++;
                        } else {
                            result.Add(new Dictionary<string, object?> { ["type"] = "remove", ["line"] = i + 1, ["content"] = old[i] });
                            result.Add(new Dictionary<string, object?> { ["type"] = "add", ["line"] = j + 1, ["content"] = newStr[j] });
                            i++; j++;
                        }
                    }
                    return result;
                }),
                ["files"] = new NSLBuiltinFunction("files", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("diff.files() requires two file paths");
                    var oldPath = args[0]?.ToString() ?? "";
                    var newPath = args[1]?.ToString() ?? "";
                    if (!File.Exists(oldPath)) throw new NSLRuntimeException($"File not found: {oldPath}");
                    if (!File.Exists(newPath)) throw new NSLRuntimeException($"File not found: {newPath}");
                    var oldContent = File.ReadAllText(oldPath);
                    var newContent = File.ReadAllText(newPath);
                    
                    // Return unified diff format
                    var oldLines = oldContent.Split('\n');
                    var newLines = newContent.Split('\n');
                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine($"--- {oldPath}");
                    sb.AppendLine($"+++ {newPath}");
                    
                    int i = 0, j = 0;
                    while (i < oldLines.Length || j < newLines.Length) {
                        if (i >= oldLines.Length) {
                            sb.AppendLine($"+{newLines[j].TrimEnd('\r')}");
                            j++;
                        } else if (j >= newLines.Length) {
                            sb.AppendLine($"-{oldLines[i].TrimEnd('\r')}");
                            i++;
                        } else if (oldLines[i].TrimEnd('\r') == newLines[j].TrimEnd('\r')) {
                            sb.AppendLine($" {oldLines[i].TrimEnd('\r')}");
                            i++; j++;
                        } else {
                            sb.AppendLine($"-{oldLines[i].TrimEnd('\r')}");
                            sb.AppendLine($"+{newLines[j].TrimEnd('\r')}");
                            i++; j++;
                        }
                    }
                    return sb.ToString();
                }),
                ["patch"] = new NSLBuiltinFunction("patch", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("diff.patch() requires content and patch string");
                    var content = args[0]?.ToString() ?? "";
                    var patch = args[1]?.ToString() ?? "";
                    var lines = content.Split('\n').ToList();
                    var patchLines = patch.Split('\n');
                    
                    // offset tracking removed - not needed
                    foreach (var pLine in patchLines) {
                        if (pLine.StartsWith("---") || pLine.StartsWith("+++") || pLine.StartsWith("@@")) continue;
                        if (pLine.StartsWith("-") && !pLine.StartsWith("---")) {
                            var toRemove = pLine.Substring(1);
                            for (int i = 0; i < lines.Count; i++) {
                                if (lines[i].TrimEnd('\r') == toRemove) {
                                    lines.RemoveAt(i);
                                    break;
                                }
                            }
                        } else if (pLine.StartsWith("+") && !pLine.StartsWith("+++")) {
                            var toAdd = pLine.Substring(1);
                            lines.Add(toAdd);
                        }
                    }
                    return string.Join("\n", lines);
                })
            };
            _globals["diff"] = diffNamespace;

            // ===== GIT NAMESPACE =====
            var gitNamespace = new Dictionary<string, object?>
            {
                ["status"] = new NSLBuiltinFunction("status", (args) => {
                    var dir = args.Length > 0 ? args[0]?.ToString() : Directory.GetCurrentDirectory();
                    var psi = new System.Diagnostics.ProcessStartInfo("git", "status --porcelain") { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd() ?? "";
                    proc?.WaitForExit();
                    var files = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => {
                        var status = l.Length >= 2 ? l.Substring(0, 2).Trim() : "";
                        var file = l.Length > 3 ? l.Substring(3) : l;
                        return new Dictionary<string, object?> { ["status"] = status, ["file"] = file };
                    }).ToList();
                    return new Dictionary<string, object?> { ["clean"] = files.Count == 0, ["files"] = files, ["count"] = files.Count };
                }),
                ["branch"] = new NSLBuiltinFunction("branch", (args) => {
                    var dir = args.Length > 0 ? args[0]?.ToString() : Directory.GetCurrentDirectory();
                    var psi = new System.Diagnostics.ProcessStartInfo("git", "branch --show-current") { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd()?.Trim() ?? "";
                    proc?.WaitForExit();
                    return output;
                }),
                ["branches"] = new NSLBuiltinFunction("branches", (args) => {
                    var dir = args.Length > 0 ? args[0]?.ToString() : Directory.GetCurrentDirectory();
                    var psi = new System.Diagnostics.ProcessStartInfo("git", "branch -a") { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd() ?? "";
                    proc?.WaitForExit();
                    return output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(b => b.Trim().TrimStart('*').Trim()).ToList();
                }),
                ["log"] = new NSLBuiltinFunction("log", (args) => {
                    var count = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 10;
                    var dir = args.Length > 1 ? args[1]?.ToString() : Directory.GetCurrentDirectory();
                    var psi = new System.Diagnostics.ProcessStartInfo("git", $"log -n {count} --pretty=format:%H|%an|%ae|%s|%ci") { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd() ?? "";
                    proc?.WaitForExit();
                    return output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => {
                        var parts = l.Split('|');
                        return new Dictionary<string, object?> { ["hash"] = parts.ElementAtOrDefault(0), ["author"] = parts.ElementAtOrDefault(1), ["email"] = parts.ElementAtOrDefault(2), ["message"] = parts.ElementAtOrDefault(3), ["date"] = parts.ElementAtOrDefault(4) };
                    }).ToList();
                }),
                ["diff"] = new NSLBuiltinFunction("diff", (args) => {
                    var file = args.Length > 0 ? args[0]?.ToString() : "";
                    var dir = args.Length > 1 ? args[1]?.ToString() : Directory.GetCurrentDirectory();
                    var gitArgs = string.IsNullOrEmpty(file) ? "diff" : $"diff -- {file}";
                    var psi = new System.Diagnostics.ProcessStartInfo("git", gitArgs) { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd() ?? "";
                    proc?.WaitForExit();
                    return output;
                }),
                ["show"] = new NSLBuiltinFunction("show", (args) => {
                    var commit = args.Length > 0 ? args[0]?.ToString() ?? "HEAD" : "HEAD";
                    var dir = args.Length > 1 ? args[1]?.ToString() : Directory.GetCurrentDirectory();
                    var psi = new System.Diagnostics.ProcessStartInfo("git", $"show {commit} --pretty=format:%H|%an|%s|%ci --stat") { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd() ?? "";
                    proc?.WaitForExit();
                    return output;
                }),
                ["root"] = new NSLBuiltinFunction("root", (args) => {
                    var dir = args.Length > 0 ? args[0]?.ToString() : Directory.GetCurrentDirectory();
                    var psi = new System.Diagnostics.ProcessStartInfo("git", "rev-parse --show-toplevel") { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd()?.Trim() ?? "";
                    proc?.WaitForExit();
                    return output;
                }),
                ["remote"] = new NSLBuiltinFunction("remote", (args) => {
                    var dir = args.Length > 0 ? args[0]?.ToString() : Directory.GetCurrentDirectory();
                    var psi = new System.Diagnostics.ProcessStartInfo("git", "remote -v") { WorkingDirectory = dir, RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd() ?? "";
                    proc?.WaitForExit();
                    var remotes = new Dictionary<string, object?>();
                    foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries)) {
                        var parts = line.Split('\t');
                        if (parts.Length >= 2) remotes[parts[0]] = parts[1].Split(' ')[0];
                    }
                    return remotes;
                }),
                ["isRepo"] = new NSLBuiltinFunction("isRepo", (args) => {
                    var dir = args.Length > 0 ? args[0]?.ToString() : Directory.GetCurrentDirectory();
                    return Directory.Exists(Path.Combine(dir ?? ".", ".git"));
                })
            };
            _globals["git"] = gitNamespace;

            // ===== PROC NAMESPACE (Process Management) =====
            var procNamespace = new Dictionary<string, object?>
            {
                ["list"] = new NSLBuiltinFunction("list", (args) => {
                    var filter = args.Length > 0 ? args[0]?.ToString()?.ToLower() : null;
                    var procs = System.Diagnostics.Process.GetProcesses();
                    var result = procs.Where(p => {
                        try { return filter == null || p.ProcessName.ToLower().Contains(filter); } catch { return false; }
                    }).Select(p => {
                        try { return new Dictionary<string, object?> { ["pid"] = (double)p.Id, ["name"] = p.ProcessName, ["memory"] = (double)p.WorkingSet64 }; }
                        catch { return new Dictionary<string, object?> { ["pid"] = (double)p.Id, ["name"] = "unknown", ["memory"] = 0.0 }; }
                    }).Take(100).ToList();
                    return result;
                }),
                ["kill"] = new NSLBuiltinFunction("kill", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("proc.kill() requires PID or process name");
                    var arg = args[0];
                    try {
                        if (arg is double pid) { System.Diagnostics.Process.GetProcessById((int)pid).Kill(); return true; }
                        var name = arg?.ToString() ?? "";
                        var procs = System.Diagnostics.Process.GetProcessesByName(name);
                        foreach (var p in procs) p.Kill();
                        return procs.Length > 0;
                    } catch { return false; }
                }),
                ["exists"] = new NSLBuiltinFunction("exists", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("proc.exists() requires PID or name");
                    var arg = args[0];
                    if (arg is double pid) { try { System.Diagnostics.Process.GetProcessById((int)pid); return true; } catch { return false; } }
                    return System.Diagnostics.Process.GetProcessesByName(arg?.ToString() ?? "").Length > 0;
                }),
                ["info"] = new NSLBuiltinFunction("info", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("proc.info() requires PID");
                    var pid = (int)ConvertToNumber(args[0]);
                    try {
                        var p = System.Diagnostics.Process.GetProcessById(pid);
                        return new Dictionary<string, object?> { ["pid"] = (double)p.Id, ["name"] = p.ProcessName, ["memory"] = (double)p.WorkingSet64, ["threads"] = (double)p.Threads.Count, ["started"] = p.StartTime.ToString("o") };
                    } catch (Exception ex) { throw new NSLRuntimeException($"Process not found: {ex.Message}"); }
                })
            };
            _globals["proc"] = procNamespace;

            // ===== CLIP NAMESPACE (Clipboard) =====
            var clipNamespace = new Dictionary<string, object?>
            {
                ["copy"] = new NSLBuiltinFunction("copy", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("clip.copy() requires text");
                    var text = args[0]?.ToString() ?? "";
                    var psi = new System.Diagnostics.ProcessStartInfo("cmd", $"/c echo {text.Replace("\"", "\\\"")}| clip") { UseShellExecute = false, CreateNoWindow = true };
                    System.Diagnostics.Process.Start(psi)?.WaitForExit();
                    return true;
                }),
                ["paste"] = new NSLBuiltinFunction("paste", (args) => {
                    var psi = new System.Diagnostics.ProcessStartInfo("powershell", "-command Get-Clipboard") { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
                    using var proc = System.Diagnostics.Process.Start(psi);
                    var output = proc?.StandardOutput.ReadToEnd()?.TrimEnd() ?? "";
                    proc?.WaitForExit();
                    return output;
                })
            };
            _globals["clip"] = clipNamespace;

            // ===== ENV NAMESPACE (Environment) =====
            var envNamespace = new Dictionary<string, object?>
            {
                ["get"] = new NSLBuiltinFunction("get", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("env.get() requires variable name");
                    return Environment.GetEnvironmentVariable(args[0]?.ToString() ?? "") ?? "";
                }),
                ["set"] = new NSLBuiltinFunction("set", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("env.set() requires name and value");
                    Environment.SetEnvironmentVariable(args[0]?.ToString() ?? "", args[1]?.ToString() ?? "");
                    return true;
                }),
                ["all"] = new NSLBuiltinFunction("all", (args) => {
                    var vars = Environment.GetEnvironmentVariables();
                    var result = new Dictionary<string, object?>();
                    foreach (System.Collections.DictionaryEntry entry in vars) result[entry.Key.ToString() ?? ""] = entry.Value?.ToString();
                    return result;
                }),
                ["keys"] = new NSLBuiltinFunction("keys", (args) => {
                    var vars = Environment.GetEnvironmentVariables();
                    return vars.Keys.Cast<string>().OrderBy(k => k).ToList();
                }),
                ["expand"] = new NSLBuiltinFunction("expand", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("env.expand() requires string");
                    return Environment.ExpandEnvironmentVariables(args[0]?.ToString() ?? "");
                }),
                ["path"] = new NSLBuiltinFunction("path", (args) => {
                    var pathVar = Environment.GetEnvironmentVariable("PATH") ?? "";
                    return pathVar.Split(Path.PathSeparator).Where(p => !string.IsNullOrEmpty(p)).ToList();
                }),
                ["home"] = new NSLBuiltinFunction("home", (args) => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)),
                ["temp"] = new NSLBuiltinFunction("temp", (args) => Path.GetTempPath()),
                ["os"] = new NSLBuiltinFunction("os", (args) => Environment.OSVersion.Platform.ToString()),
                ["arch"] = new NSLBuiltinFunction("arch", (args) => Environment.Is64BitOperatingSystem ? "x64" : "x86"),
                ["user"] = new NSLBuiltinFunction("user", (args) => Environment.UserName),
                ["machine"] = new NSLBuiltinFunction("machine", (args) => Environment.MachineName)
            };
            _globals["env"] = envNamespace;

            // ===== XML NAMESPACE =====
            var xmlNamespace = new Dictionary<string, object?>
            {
                ["parse"] = new NSLBuiltinFunction("parse", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("xml.parse() requires XML string");
                    var xml = args[0]?.ToString() ?? "";
                    var doc = new System.Xml.XmlDocument();
                    doc.LoadXml(xml);
                    object? ConvertNode(System.Xml.XmlNode node) {
                        if (node.NodeType == System.Xml.XmlNodeType.Text) return node.Value;
                        var result = new Dictionary<string, object?>();
                        if (node.Attributes != null) {
                            foreach (System.Xml.XmlAttribute attr in node.Attributes) result["@" + attr.Name] = attr.Value;
                        }
                        var children = new Dictionary<string, List<object?>>();
                        foreach (System.Xml.XmlNode child in node.ChildNodes) {
                            if (child.NodeType == System.Xml.XmlNodeType.Text) { result["#text"] = child.Value; continue; }
                            if (!children.ContainsKey(child.Name)) children[child.Name] = new List<object?>();
                            children[child.Name].Add(ConvertNode(child));
                        }
                        foreach (var kvp in children) result[kvp.Key] = kvp.Value.Count == 1 ? kvp.Value[0] : kvp.Value;
                        return result;
                    }
                    return ConvertNode(doc.DocumentElement!);
                }),
                ["query"] = new NSLBuiltinFunction("query", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("xml.query() requires XML string and XPath");
                    var xml = args[0]?.ToString() ?? "";
                    var xpath = args[1]?.ToString() ?? "";
                    var doc = new System.Xml.XmlDocument();
                    doc.LoadXml(xml);
                    var nodes = doc.SelectNodes(xpath);
                    var results = new List<object?>();
                    if (nodes != null) foreach (System.Xml.XmlNode node in nodes) results.Add(node.InnerText);
                    return results;
                })
            };
            _globals["xml"] = xmlNamespace;

            // ===== ZIP NAMESPACE (Archives) =====
            var zipNamespace = new Dictionary<string, object?>
            {
                ["create"] = new NSLBuiltinFunction("create", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("zip.create() requires source and destination");
                    var source = args[0]?.ToString() ?? "";
                    var dest = args[1]?.ToString() ?? "";
                    if (Directory.Exists(source)) System.IO.Compression.ZipFile.CreateFromDirectory(source, dest);
                    else if (File.Exists(source)) {
                        using var zip = System.IO.Compression.ZipFile.Open(dest, System.IO.Compression.ZipArchiveMode.Create);
                        zip.CreateEntryFromFile(source, Path.GetFileName(source));
                    }
                    return true;
                }),
                ["extract"] = new NSLBuiltinFunction("extract", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("zip.extract() requires source and destination");
                    var source = args[0]?.ToString() ?? "";
                    var dest = args[1]?.ToString() ?? "";
                    System.IO.Compression.ZipFile.ExtractToDirectory(source, dest, true);
                    return true;
                }),
                ["list"] = new NSLBuiltinFunction("list", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("zip.list() requires archive path");
                    var path = args[0]?.ToString() ?? "";
                    using var zip = System.IO.Compression.ZipFile.OpenRead(path);
                    return zip.Entries.Select(e => new Dictionary<string, object?> { ["name"] = e.FullName, ["size"] = (double)e.Length, ["compressed"] = (double)e.CompressedLength }).ToList();
                }),
                ["add"] = new NSLBuiltinFunction("add", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("zip.add() requires archive and file path");
                    var archive = args[0]?.ToString() ?? "";
                    var file = args[1]?.ToString() ?? "";
                    var entryName = args.Length > 2 ? args[2]?.ToString() : Path.GetFileName(file);
                    using var zip = System.IO.Compression.ZipFile.Open(archive, System.IO.Compression.ZipArchiveMode.Update);
                    zip.CreateEntryFromFile(file, entryName ?? Path.GetFileName(file));
                    return true;
                })
            };
            _globals["zip"] = zipNamespace;

            // ===== NET NAMESPACE (Network) =====
            var netNamespace = new Dictionary<string, object?>
            {
                ["ping"] = new NSLBuiltinFunction("ping", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("net.ping() requires host");
                    var host = args[0]?.ToString() ?? "";
                    var timeout = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 3000;
                    try {
                        using var ping = new System.Net.NetworkInformation.Ping();
                        var reply = ping.Send(host, timeout);
                        return new Dictionary<string, object?> { ["success"] = reply.Status == System.Net.NetworkInformation.IPStatus.Success, ["time"] = (double)reply.RoundtripTime, ["status"] = reply.Status.ToString() };
                    } catch (Exception ex) { return new Dictionary<string, object?> { ["success"] = false, ["error"] = ex.Message }; }
                }),
                ["lookup"] = new NSLBuiltinFunction("lookup", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("net.lookup() requires hostname");
                    var host = args[0]?.ToString() ?? "";
                    try {
                        var addresses = System.Net.Dns.GetHostAddresses(host);
                        return addresses.Select(a => a.ToString()).ToList();
                    } catch (Exception ex) { throw new NSLRuntimeException($"DNS lookup failed: {ex.Message}"); }
                }),
                ["localIp"] = new NSLBuiltinFunction("localIp", (args) => {
                    var host = System.Net.Dns.GetHostEntry(System.Net.Dns.GetHostName());
                    foreach (var ip in host.AddressList) {
                        if (ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork) return ip.ToString();
                    }
                    return "127.0.0.1";
                }),
                ["isOnline"] = new NSLBuiltinFunction("isOnline", (args) => {
                    try {
                        using var ping = new System.Net.NetworkInformation.Ping();
                        var reply = ping.Send("8.8.8.8", 1000);
                        return reply.Status == System.Net.NetworkInformation.IPStatus.Success;
                    } catch { return false; }
                }),
                ["ports"] = new NSLBuiltinFunction("ports", (args) => {
                    var props = System.Net.NetworkInformation.IPGlobalProperties.GetIPGlobalProperties();
                    var tcpConns = props.GetActiveTcpListeners();
                    return tcpConns.Select(c => (double)c.Port).Distinct().OrderBy(p => p).ToList();
                })
            };
            _globals["net"] = netNamespace;

            // ===== TEMPLATE NAMESPACE (String Interpolation) =====
            var templateNamespace = new Dictionary<string, object?>
            {
                ["render"] = new NSLBuiltinFunction("render", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("template.render() requires template string and variables");
                    var template = args[0]?.ToString() ?? "";
                    var vars = args[1] as Dictionary<string, object?> ?? new Dictionary<string, object?>();
                    var result = template;
                    foreach (var kvp in vars) {
                        result = result.Replace("${" + kvp.Key + "}", kvp.Value?.ToString() ?? "");
                        result = result.Replace("{" + kvp.Key + "}", kvp.Value?.ToString() ?? "");
                    }
                    return result;
                })
            };
            _globals["template"] = templateNamespace;

            // ===== CRYPTO NAMESPACE =====
            var cryptoNamespace = new Dictionary<string, object?>
            {
                ["hash"] = new NSLBuiltinFunction("hash", (args) => { if (args.Length < 1) throw new NSLRuntimeException("crypto.hash() requires data"); var data = args[0]?.ToString() ?? ""; var algo = args.Length > 1 ? args[1]?.ToString()?.ToLower() ?? "sha256" : "sha256"; var bytes = System.Text.Encoding.UTF8.GetBytes(data); using var hasher = algo switch { "md5" => System.Security.Cryptography.MD5.Create() as System.Security.Cryptography.HashAlgorithm, "sha1" => System.Security.Cryptography.SHA1.Create(), "sha384" => System.Security.Cryptography.SHA384.Create(), "sha512" => System.Security.Cryptography.SHA512.Create(), _ => System.Security.Cryptography.SHA256.Create() }; var hash = hasher.ComputeHash(bytes); return Convert.ToHexString(hash).ToLowerInvariant(); }),
                ["uuid"] = new NSLBuiltinFunction("uuid", (args) => Guid.NewGuid().ToString()),
                ["random"] = new NSLBuiltinFunction("random", (args) => { var length = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 32; var bytes = new byte[length / 2]; System.Security.Cryptography.RandomNumberGenerator.Fill(bytes); return Convert.ToHexString(bytes).ToLowerInvariant(); }),
                ["base64encode"] = new NSLBuiltinFunction("base64encode", (args) => { if (args.Length < 1) throw new NSLRuntimeException("crypto.base64encode() requires data"); return Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(args[0]?.ToString() ?? "")); }),
                ["base64decode"] = new NSLBuiltinFunction("base64decode", (args) => { if (args.Length < 1) throw new NSLRuntimeException("crypto.base64decode() requires data"); try { return System.Text.Encoding.UTF8.GetString(Convert.FromBase64String(args[0]?.ToString() ?? "")); } catch (Exception ex) { throw new NSLRuntimeException($"Base64 decode failed: {ex.Message}"); } })
            };
            _globals["crypto"] = cryptoNamespace;

            // Date namespace - date/time operations
            var dateNamespace = new Dictionary<string, object?>
            {
                ["now"] = new NSLBuiltinFunction("now", (args) => DateTime.Now.ToString("o")),
                ["utc"] = new NSLBuiltinFunction("utc", (args) => DateTime.UtcNow.ToString("o")),
                ["parse"] = new NSLBuiltinFunction("parse", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.parse() requires a date string"); return DateTime.Parse(args[0]?.ToString() ?? "").ToString("o"); }),
                ["format"] = new NSLBuiltinFunction("format", (args) => { if (args.Length < 2) throw new NSLRuntimeException("date.format() requires date and format"); return DateTime.Parse(args[0]?.ToString() ?? "").ToString(args[1]?.ToString() ?? "yyyy-MM-dd"); }),
                ["add"] = new NSLBuiltinFunction("add", (args) => { if (args.Length < 3) throw new NSLRuntimeException("date.add() requires date, amount, and unit"); var dt = DateTime.Parse(args[0]?.ToString() ?? ""); var amount = (int)ConvertToNumber(args[1]); var unit = args[2]?.ToString()?.ToLower() ?? "days"; return (unit == "years" || unit == "year" || unit == "y" ? dt.AddYears(amount) : unit == "months" || unit == "month" ? dt.AddMonths(amount) : unit == "days" || unit == "day" || unit == "d" ? dt.AddDays(amount) : unit == "hours" || unit == "hour" || unit == "h" ? dt.AddHours(amount) : unit == "minutes" || unit == "minute" || unit == "m" ? dt.AddMinutes(amount) : unit == "seconds" || unit == "second" || unit == "s" ? dt.AddSeconds(amount) : dt.AddDays(amount)).ToString("o"); }),
                ["diff"] = new NSLBuiltinFunction("diff", (args) => { if (args.Length < 2) throw new NSLRuntimeException("date.diff() requires two dates"); var dt1 = DateTime.Parse(args[0]?.ToString() ?? ""); var dt2 = DateTime.Parse(args[1]?.ToString() ?? ""); var unit = args.Length > 2 ? args[2]?.ToString()?.ToLower() ?? "days" : "days"; var diff = dt2 - dt1; return unit == "years" || unit == "year" || unit == "y" ? diff.TotalDays / 365.25 : unit == "months" || unit == "month" ? diff.TotalDays / 30.44 : unit == "days" || unit == "day" || unit == "d" ? diff.TotalDays : unit == "hours" || unit == "hour" || unit == "h" ? diff.TotalHours : unit == "minutes" || unit == "minute" || unit == "m" ? diff.TotalMinutes : unit == "seconds" || unit == "second" || unit == "s" ? diff.TotalSeconds : diff.TotalDays; }),
                ["year"] = new NSLBuiltinFunction("year", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.year() requires a date"); return DateTime.Parse(args[0]?.ToString() ?? "").Year; }),
                ["month"] = new NSLBuiltinFunction("month", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.month() requires a date"); return DateTime.Parse(args[0]?.ToString() ?? "").Month; }),
                ["day"] = new NSLBuiltinFunction("day", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.day() requires a date"); return DateTime.Parse(args[0]?.ToString() ?? "").Day; }),
                ["hour"] = new NSLBuiltinFunction("hour", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.hour() requires a date"); return DateTime.Parse(args[0]?.ToString() ?? "").Hour; }),
                ["minute"] = new NSLBuiltinFunction("minute", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.minute() requires a date"); return DateTime.Parse(args[0]?.ToString() ?? "").Minute; }),
                ["second"] = new NSLBuiltinFunction("second", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.second() requires a date"); return DateTime.Parse(args[0]?.ToString() ?? "").Second; }),
                ["weekday"] = new NSLBuiltinFunction("weekday", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.weekday() requires a date"); return DateTime.Parse(args[0]?.ToString() ?? "").DayOfWeek.ToString(); }),
                ["timestamp"] = new NSLBuiltinFunction("timestamp", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.timestamp() requires a date"); return new DateTimeOffset(DateTime.Parse(args[0]?.ToString() ?? "")).ToUnixTimeSeconds(); }),
                ["fromTimestamp"] = new NSLBuiltinFunction("fromTimestamp", (args) => { if (args.Length < 1) throw new NSLRuntimeException("date.fromTimestamp() requires a timestamp"); return DateTimeOffset.FromUnixTimeSeconds((long)ConvertToNumber(args[0])).DateTime.ToString("o"); })
            };
            _globals["date"] = dateNamespace;

            // Convert namespace - base/encoding conversions
            var convertNamespace = new Dictionary<string, object?>
            {
                ["toHex"] = new NSLBuiltinFunction("toHex", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.toHex() requires a number"); return ((long)ConvertToNumber(args[0])).ToString("X"); }),
                ["fromHex"] = new NSLBuiltinFunction("fromHex", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.fromHex() requires a hex string"); return Convert.ToInt64(args[0]?.ToString() ?? "0", 16); }),
                ["toBin"] = new NSLBuiltinFunction("toBin", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.toBin() requires a number"); return Convert.ToString((long)ConvertToNumber(args[0]), 2); }),
                ["fromBin"] = new NSLBuiltinFunction("fromBin", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.fromBin() requires a binary string"); return Convert.ToInt64(args[0]?.ToString() ?? "0", 2); }),
                ["toOct"] = new NSLBuiltinFunction("toOct", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.toOct() requires a number"); return Convert.ToString((long)ConvertToNumber(args[0]), 8); }),
                ["fromOct"] = new NSLBuiltinFunction("fromOct", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.fromOct() requires an octal string"); return Convert.ToInt64(args[0]?.ToString() ?? "0", 8); }),
                ["toBase"] = new NSLBuiltinFunction("toBase", (args) => { if (args.Length < 2) throw new NSLRuntimeException("convert.toBase() requires number and base"); return Convert.ToString((long)ConvertToNumber(args[0]), (int)ConvertToNumber(args[1])); }),
                ["fromBase"] = new NSLBuiltinFunction("fromBase", (args) => { if (args.Length < 2) throw new NSLRuntimeException("convert.fromBase() requires string and base"); return Convert.ToInt64(args[0]?.ToString() ?? "0", (int)ConvertToNumber(args[1])); }),
                ["toAscii"] = new NSLBuiltinFunction("toAscii", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.toAscii() requires a character"); var s = args[0]?.ToString() ?? ""; return s.Length > 0 ? (int)s[0] : 0; }),
                ["fromAscii"] = new NSLBuiltinFunction("fromAscii", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.fromAscii() requires a number"); return ((char)(int)ConvertToNumber(args[0])).ToString(); }),
                ["toUtf8"] = new NSLBuiltinFunction("toUtf8", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.toUtf8() requires a string"); return new List<object>(System.Text.Encoding.UTF8.GetBytes(args[0]?.ToString() ?? "").Select(b => (object)(double)b)); }),
                ["fromUtf8"] = new NSLBuiltinFunction("fromUtf8", (args) => { if (args.Length < 1) throw new NSLRuntimeException("convert.fromUtf8() requires byte array"); var bytes = args[0] as IList<object> ?? new List<object>(); return System.Text.Encoding.UTF8.GetString(bytes.Select(b => (byte)ConvertToNumber(b)).ToArray()); })
            };
            _globals["convert"] = convertNamespace;

            // Safe namespace - input sanitization and security
            var safeNamespace = new Dictionary<string, object?>
            {
                ["path"] = new NSLBuiltinFunction("path", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.path() requires a path"); var path = args[0]?.ToString() ?? ""; return System.Text.RegularExpressions.Regex.Replace(path, @"\.{2,}|[<>:""|?*\x00-\x1f]", "_"); }),
                ["shell"] = new NSLBuiltinFunction("shell", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.shell() requires input"); var input = args[0]?.ToString() ?? ""; return System.Text.RegularExpressions.Regex.Replace(input, @"[;&|`$(){}[\]<>!#*?~]", ""); }),
                ["command"] = new NSLBuiltinFunction("command", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.command() requires input"); var input = args[0]?.ToString() ?? ""; return """ + input.Replace("\"", "\\\"").Replace("\"", "\\\"") + """; }),
                ["html"] = new NSLBuiltinFunction("html", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.html() requires input"); return System.Net.WebUtility.HtmlEncode(args[0]?.ToString() ?? ""); }),
                ["sql"] = new NSLBuiltinFunction("sql", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.sql() requires input"); return (args[0]?.ToString() ?? "").Replace("'", "''"); }),
                ["email"] = new NSLBuiltinFunction("email", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.email() requires input"); var email = args[0]?.ToString() ?? ""; return System.Text.RegularExpressions.Regex.IsMatch(email, @"^[^@\s]+@[^@\s]+\.[^@\s]+$") ? email : ""; }),
                ["url"] = new NSLBuiltinFunction("url", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.url() requires input"); return Uri.EscapeDataString(args[0]?.ToString() ?? ""); }),
                ["truncate"] = new NSLBuiltinFunction("truncate", (args) => { if (args.Length < 2) throw new NSLRuntimeException("safe.truncate() requires input and max length"); var input = args[0]?.ToString() ?? ""; var max = (int)ConvertToNumber(args[1]); return input.Length <= max ? input : input.Substring(0, max); }),
                ["mask"] = new NSLBuiltinFunction("mask", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.mask() requires input"); var input = args[0]?.ToString() ?? ""; var show = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 4; if (input.Length <= show) return new string('*', input.Length); return new string('*', input.Length - show) + input.Substring(input.Length - show); }),
                ["redact"] = new NSLBuiltinFunction("redact", (args) => { if (args.Length < 2) throw new NSLRuntimeException("safe.redact() requires input and pattern"); var input = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; var replacement = args.Length > 2 ? args[2]?.ToString() ?? "[REDACTED]" : "[REDACTED]"; return System.Text.RegularExpressions.Regex.Replace(input, pattern, replacement); }),
                ["alphanumeric"] = new NSLBuiltinFunction("alphanumeric", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.alphanumeric() requires input"); return System.Text.RegularExpressions.Regex.Replace(args[0]?.ToString() ?? "", @"[^a-zA-Z0-9]", ""); }),
                ["token"] = new NSLBuiltinFunction("token", (args) => { var length = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 32; var chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; var random = new Random(); return new string(Enumerable.Repeat(chars, length).Select(s => s[random.Next(s.Length)]).ToArray()); }),
                ["checksum"] = new NSLBuiltinFunction("checksum", (args) => { if (args.Length < 1) throw new NSLRuntimeException("safe.checksum() requires input"); using var sha = System.Security.Cryptography.SHA256.Create(); var hash = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(args[0]?.ToString() ?? "")); return BitConverter.ToString(hash).Replace("-", "").ToLower(); })
            };
            _globals["safe"] = safeNamespace;

            // Stream namespace - text stream processing (grep, sed, awk, etc.)
            var streamNamespace = new Dictionary<string, object?>
            {
                ["grep"] = new NSLBuiltinFunction("grep", (args) => { if (args.Length < 2) throw new NSLRuntimeException("stream.grep() requires input and pattern"); var lines = (args[0]?.ToString() ?? "").Split('\n'); var pattern = args[1]?.ToString() ?? ""; var regex = new System.Text.RegularExpressions.Regex(pattern); return string.Join("\n", lines.Where(l => regex.IsMatch(l))); }),
                ["grepx"] = new NSLBuiltinFunction("grepx", (args) => { if (args.Length < 2) throw new NSLRuntimeException("stream.grepx() requires input and pattern"); var input = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? ""; var matches = System.Text.RegularExpressions.Regex.Matches(input, pattern); return new List<object>(matches.Select(m => (object)m.Value)); }),
                ["sed"] = new NSLBuiltinFunction("sed", (args) => { if (args.Length < 3) throw new NSLRuntimeException("stream.sed() requires input, pattern, and replacement"); return System.Text.RegularExpressions.Regex.Replace(args[0]?.ToString() ?? "", args[1]?.ToString() ?? "", args[2]?.ToString() ?? ""); }),
                ["awk"] = new NSLBuiltinFunction("awk", (args) => { if (args.Length < 2) throw new NSLRuntimeException("stream.awk() requires input and field index"); var lines = (args[0]?.ToString() ?? "").Split('\n'); var fieldIndex = (int)ConvertToNumber(args[1]) - 1; var delimiter = args.Length > 2 ? args[2]?.ToString() ?? " " : " "; return string.Join("\n", lines.Select(l => { var fields = l.Split(delimiter.ToCharArray(), StringSplitOptions.RemoveEmptyEntries); return fieldIndex >= 0 && fieldIndex < fields.Length ? fields[fieldIndex] : ""; })); }),
                ["cut"] = new NSLBuiltinFunction("cut", (args) => { if (args.Length < 2) throw new NSLRuntimeException("stream.cut() requires input and field"); var input = args[0]?.ToString() ?? ""; var field = (int)ConvertToNumber(args[1]) - 1; var delimiter = args.Length > 2 ? (args[2]?.ToString() ?? "\t")[0] : '\t'; var fields = input.Split(delimiter); return field >= 0 && field < fields.Length ? fields[field] : ""; }),
                ["sort"] = new NSLBuiltinFunction("sort", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.sort() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n').ToList(); lines.Sort(); if (args.Length > 1 && args[1] is bool reverse && reverse) lines.Reverse(); return string.Join("\n", lines); }),
                ["uniq"] = new NSLBuiltinFunction("uniq", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.uniq() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n'); return string.Join("\n", lines.Distinct()); }),
                ["wc"] = new NSLBuiltinFunction("wc", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.wc() requires input"); var input = args[0]?.ToString() ?? ""; var lines = input.Split('\n').Length; var words = input.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length; var chars = input.Length; return new Dictionary<string, object> { ["lines"] = lines, ["words"] = words, ["chars"] = chars }; }),
                ["tr"] = new NSLBuiltinFunction("tr", (args) => { if (args.Length < 3) throw new NSLRuntimeException("stream.tr() requires input, from, and to"); var input = args[0]?.ToString() ?? ""; var from = args[1]?.ToString() ?? ""; var to = args[2]?.ToString() ?? ""; for (int i = 0; i < from.Length && i < to.Length; i++) input = input.Replace(from[i], to[i]); return input; }),
                ["tee"] = new NSLBuiltinFunction("tee", (args) => { if (args.Length < 2) throw new NSLRuntimeException("stream.tee() requires input and file path"); var input = args[0]?.ToString() ?? ""; var path = args[1]?.ToString() ?? ""; File.WriteAllText(path, input); return input; }),
                ["nl"] = new NSLBuiltinFunction("nl", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.nl() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n'); return string.Join("\n", lines.Select((l, i) => $"{i + 1}\t{l}")); }),
                ["head"] = new NSLBuiltinFunction("head", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.head() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n'); var count = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 10; return string.Join("\n", lines.Take(count)); }),
                ["tail"] = new NSLBuiltinFunction("tail", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.tail() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n'); var count = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 10; return string.Join("\n", lines.TakeLast(count)); }),
                ["rev"] = new NSLBuiltinFunction("rev", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.rev() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n'); return string.Join("\n", lines.Select(l => new string(l.Reverse().ToArray()))); }),
                ["split"] = new NSLBuiltinFunction("split", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.split() requires input"); var input = args[0]?.ToString() ?? ""; var delimiter = args.Length > 1 ? args[1]?.ToString() ?? "\n" : "\n"; return new List<object>(input.Split(new[] { delimiter }, StringSplitOptions.None).Select(s => (object)s)); }),
                ["join"] = new NSLBuiltinFunction("join", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.join() requires input array"); var list = args[0] as IList<object> ?? new List<object>(); var delimiter = args.Length > 1 ? args[1]?.ToString() ?? "\n" : "\n"; return string.Join(delimiter, list.Select(x => x?.ToString() ?? "")); }),
                ["column"] = new NSLBuiltinFunction("column", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.column() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n'); var delimiter = args.Length > 1 ? args[1]?.ToString() ?? "\t" : "\t"; var table = lines.Select(l => l.Split(delimiter.ToCharArray())).ToList(); if (!table.Any()) return ""; var colWidths = Enumerable.Range(0, table.Max(r => r.Length)).Select(c => table.Max(r => c < r.Length ? r[c].Length : 0)).ToList(); return string.Join("\n", table.Select(r => string.Join("  ", r.Select((cell, c) => c < colWidths.Count ? cell.PadRight(colWidths[c]) : cell)))); }),
                ["fold"] = new NSLBuiltinFunction("fold", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.fold() requires input"); var input = args[0]?.ToString() ?? ""; var width = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 80; var result = new System.Text.StringBuilder(); for (int i = 0; i < input.Length; i += width) result.AppendLine(input.Substring(i, Math.Min(width, input.Length - i))); return result.ToString().TrimEnd(); }),
                ["shuf"] = new NSLBuiltinFunction("shuf", (args) => { if (args.Length < 1) throw new NSLRuntimeException("stream.shuf() requires input"); var lines = (args[0]?.ToString() ?? "").Split('\n').ToList(); var random = new Random(); return string.Join("\n", lines.OrderBy(x => random.Next())); }),
                ["sample"] = new NSLBuiltinFunction("sample", (args) => { if (args.Length < 2) throw new NSLRuntimeException("stream.sample() requires input and count"); var lines = (args[0]?.ToString() ?? "").Split('\n').ToList(); var count = (int)ConvertToNumber(args[1]); var random = new Random(); return string.Join("\n", lines.OrderBy(x => random.Next()).Take(count)); })
            };
            _globals["stream"] = streamNamespace;

            // Glob namespace - file pattern matching
            var globNamespace = new Dictionary<string, object?>
            {
                ["expand"] = new NSLBuiltinFunction("expand", (args) => { if (args.Length < 1) throw new NSLRuntimeException("glob.expand() requires a pattern"); var pattern = args[0]?.ToString() ?? "*"; var dir = args.Length > 1 ? args[1]?.ToString() ?? "." : "."; try { var files = Directory.GetFiles(dir, pattern); return new List<object>(files.Select(f => (object)f)); } catch { return new List<object>(); } }),
                ["recursive"] = new NSLBuiltinFunction("recursive", (args) => { if (args.Length < 1) throw new NSLRuntimeException("glob.recursive() requires a pattern"); var pattern = args[0]?.ToString() ?? "*"; var dir = args.Length > 1 ? args[1]?.ToString() ?? "." : "."; try { var files = Directory.GetFiles(dir, pattern, SearchOption.AllDirectories); return new List<object>(files.Select(f => (object)f)); } catch { return new List<object>(); } }),
                ["match"] = new NSLBuiltinFunction("match", (args) => { if (args.Length < 2) throw new NSLRuntimeException("glob.match() requires filename and pattern"); var filename = args[0]?.ToString() ?? ""; var pattern = args[1]?.ToString() ?? "*"; var regexPattern = "^" + System.Text.RegularExpressions.Regex.Escape(pattern).Replace("\\*", ".*").Replace("\\?", ".") + "$"; return System.Text.RegularExpressions.Regex.IsMatch(filename, regexPattern, System.Text.RegularExpressions.RegexOptions.IgnoreCase); }),
                ["filter"] = new NSLBuiltinFunction("filter", (args) => { if (args.Length < 2) throw new NSLRuntimeException("glob.filter() requires file list and pattern"); var files = args[0] as IList<object> ?? new List<object>(); var pattern = args[1]?.ToString() ?? "*"; var regexPattern = "^" + System.Text.RegularExpressions.Regex.Escape(pattern).Replace("\\*", ".*").Replace("\\?", ".") + "$"; return new List<object>(files.Where(f => System.Text.RegularExpressions.Regex.IsMatch(f?.ToString() ?? "", regexPattern, System.Text.RegularExpressions.RegexOptions.IgnoreCase))); })
            };
            _globals["glob"] = globNamespace;

            // ANSI namespace - terminal colors and formatting
            var ansiNamespace = new Dictionary<string, object?>
            {
                ["reset"] = "\x1b[0m",
                ["bold"] = "\x1b[1m",
                ["dim"] = "\x1b[2m",
                ["italic"] = "\x1b[3m",
                ["underline"] = "\x1b[4m",
                ["blink"] = "\x1b[5m",
                ["reverse"] = "\x1b[7m",
                ["hidden"] = "\x1b[8m",
                ["strikethrough"] = "\x1b[9m",
                ["black"] = "\x1b[30m",
                ["red"] = "\x1b[31m",
                ["green"] = "\x1b[32m",
                ["yellow"] = "\x1b[33m",
                ["blue"] = "\x1b[34m",
                ["magenta"] = "\x1b[35m",
                ["cyan"] = "\x1b[36m",
                ["white"] = "\x1b[37m",
                ["bgBlack"] = "\x1b[40m",
                ["bgRed"] = "\x1b[41m",
                ["bgGreen"] = "\x1b[42m",
                ["bgYellow"] = "\x1b[43m",
                ["bgBlue"] = "\x1b[44m",
                ["bgMagenta"] = "\x1b[45m",
                ["bgCyan"] = "\x1b[46m",
                ["bgWhite"] = "\x1b[47m",
                ["clearScreen"] = "\x1b[2J\x1b[H",
                ["clearLine"] = "\x1b[2K",
                ["cursorUp"] = new NSLBuiltinFunction("cursorUp", (args) => $"\x1b[{(args.Length > 0 ? ConvertToNumber(args[0]) : 1)}A"),
                ["cursorDown"] = new NSLBuiltinFunction("cursorDown", (args) => $"\x1b[{(args.Length > 0 ? ConvertToNumber(args[0]) : 1)}B"),
                ["cursorRight"] = new NSLBuiltinFunction("cursorRight", (args) => $"\x1b[{(args.Length > 0 ? ConvertToNumber(args[0]) : 1)}C"),
                ["cursorLeft"] = new NSLBuiltinFunction("cursorLeft", (args) => $"\x1b[{(args.Length > 0 ? ConvertToNumber(args[0]) : 1)}D"),
                ["moveTo"] = new NSLBuiltinFunction("moveTo", (args) => { if (args.Length < 2) throw new NSLRuntimeException("ansi.moveTo() requires row and column"); return $"\x1b[{ConvertToNumber(args[0])};{ConvertToNumber(args[1])}H"; }),
                ["saveCursor"] = "\x1b[s",
                ["restoreCursor"] = "\x1b[u",
                ["strip"] = new NSLBuiltinFunction("strip", (args) => { if (args.Length < 1) throw new NSLRuntimeException("ansi.strip() requires input"); return System.Text.RegularExpressions.Regex.Replace(args[0]?.ToString() ?? "", @"\x1b\[[0-9;]*m", ""); }),
                ["rgb"] = new NSLBuiltinFunction("rgb", (args) => { if (args.Length < 3) throw new NSLRuntimeException("ansi.rgb() requires r, g, b"); return $"\x1b[38;2;{(int)ConvertToNumber(args[0])};{(int)ConvertToNumber(args[1])};{(int)ConvertToNumber(args[2])}m"; }),
                ["bgRgb"] = new NSLBuiltinFunction("bgRgb", (args) => { if (args.Length < 3) throw new NSLRuntimeException("ansi.bgRgb() requires r, g, b"); return $"\x1b[48;2;{(int)ConvertToNumber(args[0])};{(int)ConvertToNumber(args[1])};{(int)ConvertToNumber(args[2])}m"; })
            };
            _globals["ansi"] = ansiNamespace;

            // CLI namespace - command-line argument parsing and interaction
            var cliNamespace = new Dictionary<string, object?>
            {
                ["args"] = new NSLBuiltinFunction("args", (args) => new List<object>(Environment.GetCommandLineArgs().Skip(1).Select(a => (object)a))),
                ["flag"] = new NSLBuiltinFunction("flag", (args) => { if (args.Length < 1) throw new NSLRuntimeException("cli.flag() requires flag name"); var name = args[0]?.ToString() ?? ""; var cmdArgs = Environment.GetCommandLineArgs(); for (int i = 0; i < cmdArgs.Length; i++) { if (cmdArgs[i] == "-" + name || cmdArgs[i] == "--" + name) { if (i + 1 < cmdArgs.Length && !cmdArgs[i + 1].StartsWith("-")) return cmdArgs[i + 1]; return true; } } return args.Length > 1 ? args[1] : false; }),
                ["prompt"] = new NSLBuiltinFunction("prompt", (args) => { var message = args.Length > 0 ? args[0]?.ToString() ?? "" : ""; Console.Write(message); return Console.ReadLine() ?? ""; }),
                ["confirm"] = new NSLBuiltinFunction("confirm", (args) => { var message = args.Length > 0 ? args[0]?.ToString() ?? "Continue?" : "Continue?"; Console.Write(message + " [y/N] "); var response = Console.ReadLine()?.Trim().ToLower() ?? ""; return response == "y" || response == "yes"; })
            };
            _globals["cli"] = cliNamespace;

            // Environ namespace - environment variables (renamed from env to avoid conflict)
            var environNamespace = new Dictionary<string, object?>
            {
                ["get"] = new NSLBuiltinFunction("get", (args) => { if (args.Length < 1) throw new NSLRuntimeException("environ.get() requires variable name"); return Environment.GetEnvironmentVariable(args[0]?.ToString() ?? "") ?? (args.Length > 1 ? args[1]?.ToString() : null); }),
                ["set"] = new NSLBuiltinFunction("set", (args) => { if (args.Length < 2) throw new NSLRuntimeException("environ.set() requires name and value"); Environment.SetEnvironmentVariable(args[0]?.ToString() ?? "", args[1]?.ToString() ?? ""); return true; }),
                ["has"] = new NSLBuiltinFunction("has", (args) => { if (args.Length < 1) throw new NSLRuntimeException("environ.has() requires variable name"); return Environment.GetEnvironmentVariable(args[0]?.ToString() ?? "") != null; }),
                ["all"] = new NSLBuiltinFunction("all", (args) => { var dict = new Dictionary<string, object?>(); foreach (System.Collections.DictionaryEntry e in Environment.GetEnvironmentVariables()) dict[e.Key?.ToString() ?? ""] = e.Value?.ToString(); return dict; }),
                ["path"] = new NSLBuiltinFunction("path", (args) => new List<object>((Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator).Select(p => (object)p))),
                ["home"] = new NSLBuiltinFunction("home", (args) => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)),
                ["temp"] = new NSLBuiltinFunction("temp", (args) => Path.GetTempPath()),
                ["user"] = new NSLBuiltinFunction("user", (args) => Environment.UserName),
                ["machine"] = new NSLBuiltinFunction("machine", (args) => Environment.MachineName),
                ["os"] = new NSLBuiltinFunction("os", (args) => Environment.OSVersion.ToString()),
                ["is64"] = new NSLBuiltinFunction("is64", (args) => Environment.Is64BitOperatingSystem),
                ["cwd"] = new NSLBuiltinFunction("cwd", (args) => Directory.GetCurrentDirectory()),
                ["expand"] = new NSLBuiltinFunction("expand", (args) => { if (args.Length < 1) throw new NSLRuntimeException("environ.expand() requires input"); return Environment.ExpandEnvironmentVariables(args[0]?.ToString() ?? ""); })
            };
            _globals["environ"] = environNamespace;

            // Dir namespace extensions - additional dir operations
            if (_globals["dir"] is Dictionary<string, object?> existingDirNs)
            {
                existingDirNs["current"] = new NSLBuiltinFunction("current", (args) => Directory.GetCurrentDirectory());
                existingDirNs["parent"] = new NSLBuiltinFunction("parent", (args) => { var path = args.Length > 0 ? args[0]?.ToString() ?? "." : "."; return Directory.GetParent(path)?.FullName ?? ""; });
            }

            // File namespace extensions - additional file operations
            if (_globals["file"] is Dictionary<string, object?> existingFileNs)
            {
                existingFileNs["head"] = new NSLBuiltinFunction("head", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.head() requires path"); var lines = File.ReadAllLines(args[0]?.ToString() ?? ""); var count = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 10; return string.Join("\n", lines.Take(count)); });
                existingFileNs["tail"] = new NSLBuiltinFunction("tail", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.tail() requires path"); var lines = File.ReadAllLines(args[0]?.ToString() ?? ""); var count = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 10; return string.Join("\n", lines.TakeLast(count)); });
                existingFileNs["readRange"] = new NSLBuiltinFunction("readRange", (args) => { if (args.Length < 3) throw new NSLRuntimeException("file.readRange() requires path, start, and end"); var lines = File.ReadAllLines(args[0]?.ToString() ?? ""); var start = (int)ConvertToNumber(args[1]) - 1; var end = (int)ConvertToNumber(args[2]); return string.Join("\n", lines.Skip(start).Take(end - start)); });
                existingFileNs["countLines"] = new NSLBuiltinFunction("countLines", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.countLines() requires path"); return File.ReadAllLines(args[0]?.ToString() ?? "").Length; });
                
                // ===== LINE-BASED FILE EDITING =====
                existingFileNs["insertAt"] = new NSLBuiltinFunction("insertAt", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("file.insertAt(path, lineNum, text) requires 3 arguments");
                    var path = args[0]?.ToString() ?? "";
                    var lineNum = (int)ConvertToNumber(args[1]);
                    var text = args[2]?.ToString() ?? "";
                    var lines = File.ReadAllLines(path).ToList();
                    if (lineNum < 1) lineNum = 1;
                    if (lineNum > lines.Count + 1) lineNum = lines.Count + 1;
                    lines.Insert(lineNum - 1, text);
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", lines), "insertAt");
                    return new Dictionary<string, object?> { ["success"] = true, ["line"] = (double)lineNum };
                });
                
                existingFileNs["deleteLine"] = new NSLBuiltinFunction("deleteLine", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("file.deleteLine(path, lineNum) requires 2 arguments");
                    var path = args[0]?.ToString() ?? "";
                    var lineNum = (int)ConvertToNumber(args[1]);
                    var lines = File.ReadAllLines(path).ToList();
                    if (lineNum < 1 || lineNum > lines.Count) throw new NSLRuntimeException($"Line {lineNum} out of range (1-{lines.Count})");
                    var deleted = lines[lineNum - 1];
                    lines.RemoveAt(lineNum - 1);
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", lines), "deleteLine");
                    return new Dictionary<string, object?> { ["success"] = true, ["deleted"] = deleted };
                });
                
                existingFileNs["deleteLines"] = new NSLBuiltinFunction("deleteLines", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("file.deleteLines(path, start, end) requires 3 arguments");
                    var path = args[0]?.ToString() ?? "";
                    var start = (int)ConvertToNumber(args[1]);
                    var end = (int)ConvertToNumber(args[2]);
                    var lines = File.ReadAllLines(path).ToList();
                    if (start < 1) start = 1;
                    if (end > lines.Count) end = lines.Count;
                    var count = end - start + 1;
                    lines.RemoveRange(start - 1, count);
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", lines), "deleteLines");
                    return new Dictionary<string, object?> { ["success"] = true, ["deletedCount"] = (double)count };
                });
                
                existingFileNs["replaceLine"] = new NSLBuiltinFunction("replaceLine", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("file.replaceLine(path, lineNum, newText) requires 3 arguments");
                    var path = args[0]?.ToString() ?? "";
                    var lineNum = (int)ConvertToNumber(args[1]);
                    var newText = args[2]?.ToString() ?? "";
                    var lines = File.ReadAllLines(path).ToList();
                    if (lineNum < 1 || lineNum > lines.Count) throw new NSLRuntimeException($"Line {lineNum} out of range (1-{lines.Count})");
                    var oldText = lines[lineNum - 1];
                    lines[lineNum - 1] = newText;
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", lines), "replaceLine");
                    return new Dictionary<string, object?> { ["success"] = true, ["old"] = oldText, ["new"] = newText };
                });
                
                existingFileNs["replaceLines"] = new NSLBuiltinFunction("replaceLines", (args) => {
                    if (args.Length < 4) throw new NSLRuntimeException("file.replaceLines(path, start, end, newText) requires 4 arguments");
                    var path = args[0]?.ToString() ?? "";
                    var start = (int)ConvertToNumber(args[1]);
                    var end = (int)ConvertToNumber(args[2]);
                    var newText = args[3]?.ToString() ?? "";
                    var lines = File.ReadAllLines(path).ToList();
                    if (start < 1) start = 1;
                    if (end > lines.Count) end = lines.Count;
                    var count = end - start + 1;
                    lines.RemoveRange(start - 1, count);
                    var newLines = newText.Split('\n');
                    lines.InsertRange(start - 1, newLines);
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", lines), "replaceLines");
                    return new Dictionary<string, object?> { ["success"] = true, ["replacedCount"] = (double)count, ["insertedCount"] = (double)newLines.Length };
                });
                
                existingFileNs["touch"] = new NSLBuiltinFunction("touch", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.touch() requires path"); var path = args[0]?.ToString() ?? ""; if (!File.Exists(path)) File.WriteAllText(path, ""); else File.SetLastWriteTime(path, DateTime.Now); return true; });
                existingFileNs["modified"] = new NSLBuiltinFunction("modified", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.modified() requires path"); return File.GetLastWriteTime(args[0]?.ToString() ?? "").ToString("o"); });
                existingFileNs["created"] = new NSLBuiltinFunction("created", (args) => { if (args.Length < 1) throw new NSLRuntimeException("file.created() requires path"); return File.GetCreationTime(args[0]?.ToString() ?? "").ToString("o"); });
                
                // ===== FILE HISTORY SYSTEM =====
                // Get history entries for a file (list of pre-edit states)
                existingFileNs["history"] = new NSLBuiltinFunction("history", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.history() requires path");
                    var path = args[0]?.ToString() ?? "";
                    var entries = NSL.StandardLib.FileSystem.FileHistory.Instance.GetHistory(path);
                    return entries.Select(e => new Dictionary<string, object?> {
                        ["id"] = e.Id,
                        ["timestamp"] = e.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"),
                        ["operation"] = e.Operation,
                        ["size"] = e.SizeBytes,
                        ["hash"] = e.Hash,
                        ["summary"] = e.Summary
                    }).ToList<object>();
                });
                
                // Get history info (count, capacity, etc.)
                existingFileNs["historyInfo"] = new NSLBuiltinFunction("historyInfo", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.historyInfo() requires path");
                    var path = args[0]?.ToString() ?? "";
                    var info = NSL.StandardLib.FileSystem.FileHistory.Instance.GetHistoryInfo(path);
                    return new Dictionary<string, object?> {
                        ["path"] = info.Path,
                        ["count"] = info.Count,
                        ["capacity"] = info.Capacity,
                        ["persistent"] = info.Persistent,
                        ["enabled"] = info.Enabled,
                        ["lastSavedAt"] = info.LastSavedAt?.ToString("yyyy-MM-dd HH:mm:ss"),
                        ["totalSizeBytes"] = info.TotalSizeBytes
                    };
                });
                
                // Restore file to a previous state (index 0 = most recent)
                existingFileNs["restore"] = new NSLBuiltinFunction("restore", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.restore() requires path");
                    var path = args[0]?.ToString() ?? "";
                    var index = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 0;
                    return NSL.StandardLib.FileSystem.FileHistory.Instance.Restore(path, index);
                });
                
                // Clear history for a file
                existingFileNs["clearHistory"] = new NSLBuiltinFunction("clearHistory", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.clearHistory() requires path");
                    var path = args[0]?.ToString() ?? "";
                    NSL.StandardLib.FileSystem.FileHistory.Instance.ClearHistory(path);
                    return true;
                });
                
                // Configure history settings
                existingFileNs["setHistoryCapacity"] = new NSLBuiltinFunction("setHistoryCapacity", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("file.setHistoryCapacity() requires path and capacity");
                    var path = args[0]?.ToString() ?? "";
                    var capacity = (int)ConvertToNumber(args[1]);
                    NSL.StandardLib.FileSystem.FileHistory.Instance.SetCapacity(path, capacity);
                    return true;
                });
                
                // Enable/disable history globally
                existingFileNs["setHistoryEnabled"] = new NSLBuiltinFunction("setHistoryEnabled", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.setHistoryEnabled() requires boolean");
                    var enabled = args[0] is bool b ? b : args[0]?.ToString()?.ToLower() == "true";
                    NSL.StandardLib.FileSystem.FileHistory.Instance.SetEnabled(enabled);
                    return true;
                });
                
                // Enable/disable persistent storage
                existingFileNs["setHistoryPersistent"] = new NSLBuiltinFunction("setHistoryPersistent", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.setHistoryPersistent() requires boolean");
                    var persistent = args[0] is bool b ? b : args[0]?.ToString()?.ToLower() == "true";
                    NSL.StandardLib.FileSystem.FileHistory.Instance.SetPersistent(persistent);
                    return true;
                });
                
                // ===== PHASE 3: ADVANCED FILE HISTORY =====
                
                // Preview/dry run - see diff before writing
                existingFileNs["preview"] = new NSLBuiltinFunction("preview", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("file.preview() requires path and new content");
                    var path = args[0]?.ToString() ?? "";
                    var newContent = args[1]?.ToString() ?? "";
                    var result = NSL.StandardLib.FileSystem.FileHistory.Instance.Preview(path, newContent);
                    return new Dictionary<string, object?> {
                        ["path"] = result.Path,
                        ["isNewFile"] = result.IsNewFile,
                        ["oldLines"] = result.OldLines,
                        ["newLines"] = result.NewLines,
                        ["linesAdded"] = result.LinesAdded,
                        ["linesRemoved"] = result.LinesRemoved,
                        ["summary"] = result.GetSummary(),
                        ["changes"] = result.Changes.Select(c => new Dictionary<string, object?> {
                            ["type"] = c.Type,
                            ["line"] = c.Line,
                            ["content"] = c.Content
                        }).ToList<object>()
                    };
                });
                
                // Detect thrashing (file oscillating between states)
                existingFileNs["detectThrashing"] = new NSLBuiltinFunction("detectThrashing", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.detectThrashing() requires path");
                    var path = args[0]?.ToString() ?? "";
                    var result = NSL.StandardLib.FileSystem.FileHistory.Instance.DetectThrashing(path);
                    return new Dictionary<string, object?> {
                        ["path"] = result.Path,
                        ["isThrashing"] = result.IsThrashing,
                        ["message"] = result.Message,
                        ["repeatedStates"] = result.RepeatedStates,
                        ["oscillationPattern"] = result.OscillationPattern,
                        ["suggestedStableIndex"] = result.SuggestedStableIndex,
                        ["suggestedStableTimestamp"] = result.SuggestedStableTimestamp?.ToString("yyyy-MM-dd HH:mm:ss")
                    };
                });
                
                // Write with annotation (reason, agent)
                existingFileNs["writeAnnotated"] = new NSLBuiltinFunction("writeAnnotated", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("file.writeAnnotated() requires path, content, and reason");
                    var path = args[0]?.ToString() ?? "";
                    var content = args[1]?.ToString() ?? "";
                    var reason = args[2]?.ToString() ?? "";
                    var agent = args.Length > 3 ? args[3]?.ToString() : null;
                    
                    // Save with annotation
                    NSL.StandardLib.FileSystem.FileHistory.Instance.SavePreEditState(path, "write", null, reason, agent);
                    
                    // Perform atomic write
                    return NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, content, "write");
                });
                
                // Get history with full annotations
                existingFileNs["historyDetailed"] = new NSLBuiltinFunction("historyDetailed", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("file.historyDetailed() requires path");
                    var path = args[0]?.ToString() ?? "";
                    var entries = NSL.StandardLib.FileSystem.FileHistory.Instance.GetHistory(path);
                    return entries.Select(e => new Dictionary<string, object?> {
                        ["id"] = e.Id,
                        ["timestamp"] = e.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"),
                        ["operation"] = e.Operation,
                        ["size"] = e.SizeBytes,
                        ["storedSize"] = e.StoredBytes,
                        ["hash"] = e.Hash,
                        ["summary"] = e.Summary,
                        ["isSnapshot"] = e.IsSnapshot,
                        ["reason"] = e.Reason,
                        ["agent"] = e.Agent,
                        ["metadata"] = e.Metadata
                    }).ToList<object>();
                });
                
                // Configure file-specific settings
                existingFileNs["configureHistory"] = new NSLBuiltinFunction("configureHistory", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("file.configureHistory() requires path and config object");
                    var path = args[0]?.ToString() ?? "";
                    var configDict = args[1] as IDictionary<string, object?>;
                    if (configDict == null) throw new NSLRuntimeException("Second argument must be a config object");
                    
                    var config = new NSL.StandardLib.FileSystem.FileHistoryFileConfig();
                    if (configDict.TryGetValue("capacity", out var cap) && cap != null)
                        config.Capacity = (int)ConvertToNumber(cap);
                    if (configDict.TryGetValue("enabled", out var en) && en != null)
                        config.Enabled = en is bool b ? b : en.ToString()?.ToLower() == "true";
                    if (configDict.TryGetValue("strategy", out var strat) && strat != null)
                        config.Strategy = strat.ToString();
                    
                    NSL.StandardLib.FileSystem.FileHistory.Instance.ConfigureFile(path, config);
                    return true;
                });
            }

            // ===== PHASE 4: FFI NAMESPACE - Native Library Interop =====
            var ffiHandles = new Dictionary<string, nint>();
            var ffiNamespace = new Dictionary<string, object?>
            {
                ["load"] = new NSLBuiltinFunction("load", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ffi.load() requires library path");
                    var libPath = args[0]?.ToString() ?? "";
                    try {
                        var handle = System.Runtime.InteropServices.NativeLibrary.Load(libPath);
                        var id = Guid.NewGuid().ToString("N")[..8];
                        ffiHandles[id] = handle;
                        return new Dictionary<string, object?> { ["id"] = id, ["path"] = libPath, ["loaded"] = true };
                    } catch (Exception ex) {
                        return new Dictionary<string, object?> { ["id"] = null, ["path"] = libPath, ["loaded"] = false, ["error"] = ex.Message };
                    }
                }),
                ["unload"] = new NSLBuiltinFunction("unload", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ffi.unload() requires library handle");
                    var lib = args[0] as IDictionary<string, object?>;
                    var id = lib?["id"]?.ToString() ?? args[0]?.ToString() ?? "";
                    if (ffiHandles.TryGetValue(id, out var handle)) {
                        System.Runtime.InteropServices.NativeLibrary.Free(handle);
                        ffiHandles.Remove(id);
                        return true;
                    }
                    return false;
                }),
                ["symbol"] = new NSLBuiltinFunction("symbol", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("ffi.symbol() requires library and symbol name");
                    var lib = args[0] as IDictionary<string, object?>;
                    var id = lib?["id"]?.ToString() ?? args[0]?.ToString() ?? "";
                    var symbolName = args[1]?.ToString() ?? "";
                    if (ffiHandles.TryGetValue(id, out var handle)) {
                        try {
                            var addr = System.Runtime.InteropServices.NativeLibrary.GetExport(handle, symbolName);
                            return new Dictionary<string, object?> { ["name"] = symbolName, ["address"] = addr.ToString(), ["found"] = true };
                        } catch { return new Dictionary<string, object?> { ["name"] = symbolName, ["found"] = false }; }
                    }
                    return new Dictionary<string, object?> { ["name"] = symbolName, ["found"] = false, ["error"] = "Library not loaded" };
                }),
                ["libraries"] = new NSLBuiltinFunction("libraries", (args) => ffiHandles.Keys.ToList<object>())
            };
            _globals["ffi"] = ffiNamespace;

            // ===== PHASE 4: BUFFER NAMESPACE - Typed Buffers for Zero-Copy Interop =====
            var bufferNamespace = new Dictionary<string, object?>
            {
                ["create"] = new NSLBuiltinFunction("create", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("buffer.create() requires type and size");
                    var type = args[0]?.ToString() ?? "u8";
                    var size = (int)ConvertToNumber(args[1]);
                    var elemSize = type switch { "u8" => 1, "i8" => 1, "u16" => 2, "i16" => 2, "u32" => 4, "i32" => 4, "u64" => 8, "i64" => 8, "f32" => 4, "f64" => 8, _ => 1 };
                    var data = new byte[size * elemSize];
                    return new Dictionary<string, object?> { ["type"] = type, ["size"] = size, ["bytes"] = size * elemSize, ["data"] = data, ["id"] = Guid.NewGuid().ToString("N")[..8] };
                }),
                ["fromArray"] = new NSLBuiltinFunction("fromArray", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("buffer.fromArray() requires array");
                    var arr = args[0] as IList<object> ?? new List<object>();
                    var type = args.Length > 1 ? args[1]?.ToString() ?? "f64" : "f64";
                    var data = new byte[arr.Count * 8];
                    for (int i = 0; i < arr.Count; i++) { var val = Convert.ToDouble(arr[i]); BitConverter.GetBytes(val).CopyTo(data, i * 8); }
                    return new Dictionary<string, object?> { ["type"] = type, ["size"] = arr.Count, ["bytes"] = data.Length, ["data"] = data, ["id"] = Guid.NewGuid().ToString("N")[..8] };
                }),
                ["toArray"] = new NSLBuiltinFunction("toArray", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("buffer.toArray() requires buffer");
                    var buf = args[0] as IDictionary<string, object?>;
                    var data = buf?["data"] as byte[] ?? Array.Empty<byte>();
                    var type = buf?["type"]?.ToString() ?? "f64";
                    var size = (int)(buf?["size"] ?? 0);
                    var result = new List<object>();
                    for (int i = 0; i < size; i++) { result.Add((object)BitConverter.ToDouble(data, i * 8)); }
                    return result;
                }),
                ["get"] = new NSLBuiltinFunction("get", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("buffer.get() requires buffer and index");
                    var buf = args[0] as IDictionary<string, object?>;
                    var idx = (int)ConvertToNumber(args[1]);
                    var data = buf?["data"] as byte[] ?? Array.Empty<byte>();
                    return BitConverter.ToDouble(data, idx * 8);
                }),
                ["set"] = new NSLBuiltinFunction("set", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("buffer.set() requires buffer, index, and value");
                    var buf = args[0] as IDictionary<string, object?>;
                    var idx = (int)ConvertToNumber(args[1]);
                    var val = ConvertToNumber(args[2]);
                    var data = buf?["data"] as byte[];
                    if (data != null) BitConverter.GetBytes(val).CopyTo(data, idx * 8);
                    return true;
                }),
                ["copy"] = new NSLBuiltinFunction("copy", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("buffer.copy() requires buffer");
                    var buf = args[0] as IDictionary<string, object?>;
                    var data = buf?["data"] as byte[] ?? Array.Empty<byte>();
                    var newData = new byte[data.Length];
                    Array.Copy(data, newData, data.Length);
                    return new Dictionary<string, object?> { ["type"] = buf?["type"], ["size"] = buf?["size"], ["bytes"] = newData.Length, ["data"] = newData, ["id"] = Guid.NewGuid().ToString("N")[..8] };
                }),
                ["fill"] = new NSLBuiltinFunction("fill", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("buffer.fill() requires buffer and value");
                    var buf = args[0] as IDictionary<string, object?>;
                    var val = ConvertToNumber(args[1]);
                    var data = buf?["data"] as byte[];
                    var size = (int)(buf?["size"] ?? 0);
                    if (data != null) { var bytes = BitConverter.GetBytes(val); for (int i = 0; i < size; i++) bytes.CopyTo(data, i * 8); }
                    return true;
                }),
                ["slice"] = new NSLBuiltinFunction("slice", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("buffer.slice() requires buffer, start, and end");
                    var buf = args[0] as IDictionary<string, object?>;
                    var start = (int)ConvertToNumber(args[1]);
                    var end = (int)ConvertToNumber(args[2]);
                    var data = buf?["data"] as byte[] ?? Array.Empty<byte>();
                    var sliceSize = end - start;
                    var newData = new byte[sliceSize * 8];
                    Array.Copy(data, start * 8, newData, 0, sliceSize * 8);
                    return new Dictionary<string, object?> { ["type"] = buf?["type"], ["size"] = sliceSize, ["bytes"] = newData.Length, ["data"] = newData, ["id"] = Guid.NewGuid().ToString("N")[..8] };
                })
            };
            _globals["buffer"] = bufferNamespace;

            // ===== PHASE 4: RUNTIME NAMESPACE - Event Loop, Scheduler, Tick =====
            var runtimeEvents = new Dictionary<string, List<object>>();
            var runtimeTimers = new List<(DateTime nextRun, double intervalMs, object callback, string id)>();
            var runtimeNamespace = new Dictionary<string, object?>
            {
                ["on"] = new NSLBuiltinFunction("on", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("runtime.on() requires event name and callback");
                    var eventName = args[0]?.ToString() ?? "";
                    if (!runtimeEvents.ContainsKey(eventName)) runtimeEvents[eventName] = new List<object>();
                    runtimeEvents[eventName].Add(args[1]!);
                    return true;
                }),
                ["off"] = new NSLBuiltinFunction("off", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("runtime.off() requires event name");
                    var eventName = args[0]?.ToString() ?? "";
                    if (runtimeEvents.ContainsKey(eventName)) runtimeEvents[eventName].Clear();
                    return true;
                }),
                ["emit"] = new NSLBuiltinFunction("emit", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("runtime.emit() requires event name");
                    var eventName = args[0]?.ToString() ?? "";
                    var eventArgs = args.Length > 1 ? args.Skip(1).ToArray() : Array.Empty<object?>();
                    if (runtimeEvents.TryGetValue(eventName, out var handlers)) {
                        foreach (var h in handlers) {
                            if (h is NSLBuiltinFunction bf) bf.Call(eventArgs);
                            else if (h is NSLFunction nf) CallUserFunction(nf, eventArgs);
                        }
                    }
                    return true;
                }),
                ["setTimeout"] = new NSLBuiltinFunction("setTimeout", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("runtime.setTimeout() requires callback and delay");
                    var callback = args[0]!;
                    var delay = ConvertToNumber(args[1]);
                    var id = Guid.NewGuid().ToString("N")[..8];
                    runtimeTimers.Add((DateTime.Now.AddMilliseconds(delay), 0, callback, id));
                    return id;
                }),
                ["setInterval"] = new NSLBuiltinFunction("setInterval", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("runtime.setInterval() requires callback and interval");
                    var callback = args[0]!;
                    var interval = ConvertToNumber(args[1]);
                    var id = Guid.NewGuid().ToString("N")[..8];
                    runtimeTimers.Add((DateTime.Now.AddMilliseconds(interval), interval, callback, id));
                    return id;
                }),
                ["clearTimer"] = new NSLBuiltinFunction("clearTimer", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("runtime.clearTimer() requires timer id");
                    var id = args[0]?.ToString() ?? "";
                    runtimeTimers.RemoveAll(t => t.id == id);
                    return true;
                }),
                ["tick"] = new NSLBuiltinFunction("tick", (args) => {
                    var now = DateTime.Now;
                    var toRun = runtimeTimers.Where(t => t.nextRun <= now).ToList();
                    foreach (var t in toRun) {
                        if (t.callback is NSLBuiltinFunction bf) bf.Call(Array.Empty<object?>());
                        else if (t.callback is NSLFunction nf) CallUserFunction(nf, Array.Empty<object?>());
                        if (t.intervalMs > 0) {
                            runtimeTimers.Remove(t);
                            runtimeTimers.Add((now.AddMilliseconds(t.intervalMs), t.intervalMs, t.callback, t.id));
                        } else runtimeTimers.Remove(t);
                    }
                    return toRun.Count;
                }),
                ["run"] = new NSLBuiltinFunction("run", (args) => {
                    var maxTicks = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 1000;
                    var tickInterval = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 16;
                    for (int i = 0; i < maxTicks && runtimeTimers.Count > 0; i++) {
                        var now = DateTime.Now;
                        var toRun = runtimeTimers.Where(t => t.nextRun <= now).ToList();
                        foreach (var t in toRun) {
                            if (t.callback is NSLBuiltinFunction bf) bf.Call(Array.Empty<object?>());
                            else if (t.callback is NSLFunction nf) CallUserFunction(nf, Array.Empty<object?>());
                            if (t.intervalMs > 0) { runtimeTimers.Remove(t); runtimeTimers.Add((now.AddMilliseconds(t.intervalMs), t.intervalMs, t.callback, t.id)); }
                            else runtimeTimers.Remove(t);
                        }
                        System.Threading.Thread.Sleep(tickInterval);
                    }
                    return true;
                }),
                ["sleep"] = new NSLBuiltinFunction("sleep", (args) => {
                    var ms = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 0;
                    System.Threading.Thread.Sleep(ms);
                    return true;
                }),
                ["now"] = new NSLBuiltinFunction("now", (args) => DateTime.Now.Ticks / 10000.0),
                ["elapsed"] = new NSLBuiltinFunction("elapsed", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("runtime.elapsed() requires start time");
                    var start = ConvertToNumber(args[0]);
                    return DateTime.Now.Ticks / 10000.0 - start;
                })
            };
            _globals["runtime"] = runtimeNamespace;

            // ===== PHASE 4: SIM NAMESPACE - Sandbox Mode, Preview Side Effects =====
            var simPendingWrites = new Dictionary<string, string>();
            var simPendingDeletes = new HashSet<string>();
            var simActive = false;
            var simNamespace = new Dictionary<string, object?>
            {
                ["begin"] = new NSLBuiltinFunction("begin", (args) => {
                    simPendingWrites.Clear();
                    simPendingDeletes.Clear();
                    simActive = true;
                    return true;
                }),
                ["active"] = new NSLBuiltinFunction("active", (args) => simActive),
                ["write"] = new NSLBuiltinFunction("write", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("sim.write() requires path and content");
                    var path = Path.GetFullPath(args[0]?.ToString() ?? "");
                    var content = args[1]?.ToString() ?? "";
                    simPendingWrites[path] = content;
                    simPendingDeletes.Remove(path);
                    return true;
                }),
                ["delete"] = new NSLBuiltinFunction("delete", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("sim.delete() requires path");
                    var path = Path.GetFullPath(args[0]?.ToString() ?? "");
                    simPendingDeletes.Add(path);
                    simPendingWrites.Remove(path);
                    return true;
                }),
                ["read"] = new NSLBuiltinFunction("read", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("sim.read() requires path");
                    var path = Path.GetFullPath(args[0]?.ToString() ?? "");
                    if (simPendingDeletes.Contains(path)) return null;
                    if (simPendingWrites.TryGetValue(path, out var content)) return content;
                    return File.Exists(path) ? File.ReadAllText(path) : null;
                }),
                ["pending"] = new NSLBuiltinFunction("pending", (args) => new Dictionary<string, object?> {
                    ["writes"] = simPendingWrites.Keys.ToList<object>(),
                    ["deletes"] = simPendingDeletes.ToList<object>(),
                    ["count"] = simPendingWrites.Count + simPendingDeletes.Count
                }),
                ["diff"] = new NSLBuiltinFunction("diff", (args) => {
                    var changes = new List<object>();
                    foreach (var kv in simPendingWrites) {
                        var exists = File.Exists(kv.Key);
                        changes.Add(new Dictionary<string, object?> {
                            ["path"] = kv.Key, ["type"] = exists ? "modify" : "create",
                            ["oldSize"] = exists ? new FileInfo(kv.Key).Length : 0,
                            ["newSize"] = kv.Value.Length
                        });
                    }
                    foreach (var path in simPendingDeletes) {
                        changes.Add(new Dictionary<string, object?> { ["path"] = path, ["type"] = "delete" });
                    }
                    return changes;
                }),
                ["commit"] = new NSLBuiltinFunction("commit", (args) => {
                    if (!simActive) return false;
                    foreach (var path in simPendingDeletes) { if (File.Exists(path)) File.Delete(path); }
                    foreach (var kv in simPendingWrites) { File.WriteAllText(kv.Key, kv.Value); }
                    simPendingWrites.Clear();
                    simPendingDeletes.Clear();
                    simActive = false;
                    return true;
                }),
                ["rollback"] = new NSLBuiltinFunction("rollback", (args) => {
                    simPendingWrites.Clear();
                    simPendingDeletes.Clear();
                    simActive = false;
                    return true;
                })
            };
            _globals["sim"] = simNamespace;

            // ===== PHASE 4: TRACE NAMESPACE - Profiling Spans, Timing =====
            var traceSpans = new Dictionary<string, (DateTime start, List<double> times)>();
            var traceNamespace = new Dictionary<string, object?>
            {
                ["begin"] = new NSLBuiltinFunction("begin", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "default" : "default";
                    traceSpans[name] = (DateTime.Now, traceSpans.ContainsKey(name) ? traceSpans[name].times : new List<double>());
                    return name;
                }),
                ["end"] = new NSLBuiltinFunction("end", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "default" : "default";
                    if (traceSpans.TryGetValue(name, out var span)) {
                        var elapsed = (DateTime.Now - span.start).TotalMilliseconds;
                        span.times.Add(elapsed);
                        traceSpans[name] = (span.start, span.times);
                        return elapsed;
                    }
                    return 0.0;
                }),
                ["measure"] = new NSLBuiltinFunction("measure", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("trace.measure() requires name and callback");
                    var name = args[0]?.ToString() ?? "default";
                    var callback = args[1];
                    var start = DateTime.Now;
                    if (callback is NSLBuiltinFunction bf) bf.Call(Array.Empty<object?>());
                    else if (callback is NSLFunction nf) CallUserFunction(nf, Array.Empty<object?>());
                    var elapsed = (DateTime.Now - start).TotalMilliseconds;
                    if (!traceSpans.ContainsKey(name)) traceSpans[name] = (start, new List<double>());
                    traceSpans[name].times.Add(elapsed);
                    return elapsed;
                }),
                ["stats"] = new NSLBuiltinFunction("stats", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "" : "";
                    if (!string.IsNullOrEmpty(name) && traceSpans.TryGetValue(name, out var span)) {
                        var times = span.times;
                        return new Dictionary<string, object?> {
                            ["name"] = name, ["count"] = times.Count,
                            ["total"] = times.Sum(), ["avg"] = times.Count > 0 ? times.Average() : 0,
                            ["min"] = times.Count > 0 ? times.Min() : 0, ["max"] = times.Count > 0 ? times.Max() : 0
                        };
                    }
                    return traceSpans.Select(kv => new Dictionary<string, object?> {
                        ["name"] = kv.Key, ["count"] = kv.Value.times.Count,
                        ["total"] = kv.Value.times.Sum(), ["avg"] = kv.Value.times.Count > 0 ? kv.Value.times.Average() : 0
                    }).ToList<object>();
                }),
                ["clear"] = new NSLBuiltinFunction("clear", (args) => { traceSpans.Clear(); return true; }),
                ["log"] = new NSLBuiltinFunction("log", (args) => {
                    var msg = args.Length > 0 ? args[0]?.ToString() ?? "" : "";
                    Console.WriteLine($"[TRACE {DateTime.Now:HH:mm:ss.fff}] {msg}");
                    return true;
                })
            };
            _globals["trace"] = traceNamespace;

            // ===== PHASE 4: ML NAMESPACE - Tensors, Autograd, Model Ops =====
            var mlNamespace = new Dictionary<string, object?>
            {
                ["tensor"] = new NSLBuiltinFunction("tensor", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ml.tensor() requires data");
                    var data = args[0] as IList<object> ?? new List<object>();
                    var shape = args.Length > 1 ? (args[1] as IList<object>)?.Select(x => (int)Convert.ToDouble(x)).ToArray() : new[] { data.Count };
                    var values = data.Select(x => Convert.ToDouble(x)).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = values, ["shape"] = shape, ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["zeros"] = new NSLBuiltinFunction("zeros", (args) => {
                    var shape = args.Length > 0 ? (args[0] as IList<object>)?.Select(x => (int)Convert.ToDouble(x)).ToArray() : new[] { 1 };
                    var size = shape?.Aggregate(1, (a, b) => a * b) ?? 1;
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = new double[size], ["shape"] = shape, ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["ones"] = new NSLBuiltinFunction("ones", (args) => {
                    var shape = args.Length > 0 ? (args[0] as IList<object>)?.Select(x => (int)Convert.ToDouble(x)).ToArray() : new[] { 1 };
                    var size = shape?.Aggregate(1, (a, b) => a * b) ?? 1;
                    var data = Enumerable.Repeat(1.0, size).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = data, ["shape"] = shape, ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["rand"] = new NSLBuiltinFunction("rand", (args) => {
                    var shape = args.Length > 0 ? (args[0] as IList<object>)?.Select(x => (int)Convert.ToDouble(x)).ToArray() : new[] { 1 };
                    var size = shape?.Aggregate(1, (a, b) => a * b) ?? 1;
                    var data = Enumerable.Range(0, size).Select(_ => _random.NextDouble()).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = data, ["shape"] = shape, ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["add"] = new NSLBuiltinFunction("add", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("ml.add() requires two tensors");
                    var a = args[0] as IDictionary<string, object?>; var b = args[1] as IDictionary<string, object?>;
                    var aData = a?["data"] as double[] ?? Array.Empty<double>(); var bData = b?["data"] as double[] ?? Array.Empty<double>();
                    var result = aData.Zip(bData, (x, y) => x + y).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = result, ["shape"] = a?["shape"], ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["mul"] = new NSLBuiltinFunction("mul", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("ml.mul() requires two tensors");
                    var a = args[0] as IDictionary<string, object?>; var b = args[1] as IDictionary<string, object?>;
                    var aData = a?["data"] as double[] ?? Array.Empty<double>(); var bData = b?["data"] as double[] ?? Array.Empty<double>();
                    var result = aData.Zip(bData, (x, y) => x * y).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = result, ["shape"] = a?["shape"], ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["dot"] = new NSLBuiltinFunction("dot", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("ml.dot() requires two tensors");
                    var a = args[0] as IDictionary<string, object?>; var b = args[1] as IDictionary<string, object?>;
                    var aData = a?["data"] as double[] ?? Array.Empty<double>(); var bData = b?["data"] as double[] ?? Array.Empty<double>();
                    return aData.Zip(bData, (x, y) => x * y).Sum();
                }),
                ["sum"] = new NSLBuiltinFunction("sum", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ml.sum() requires a tensor");
                    var t = args[0] as IDictionary<string, object?>;
                    var data = t?["data"] as double[] ?? Array.Empty<double>();
                    return data.Sum();
                }),
                ["mean"] = new NSLBuiltinFunction("mean", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ml.mean() requires a tensor");
                    var t = args[0] as IDictionary<string, object?>;
                    var data = t?["data"] as double[] ?? Array.Empty<double>();
                    return data.Length > 0 ? data.Average() : 0.0;
                }),
                ["relu"] = new NSLBuiltinFunction("relu", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ml.relu() requires a tensor");
                    var t = args[0] as IDictionary<string, object?>;
                    var data = t?["data"] as double[] ?? Array.Empty<double>();
                    var result = data.Select(x => Math.Max(0, x)).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = result, ["shape"] = t?["shape"], ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["sigmoid"] = new NSLBuiltinFunction("sigmoid", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ml.sigmoid() requires a tensor");
                    var t = args[0] as IDictionary<string, object?>;
                    var data = t?["data"] as double[] ?? Array.Empty<double>();
                    var result = data.Select(x => 1.0 / (1.0 + Math.Exp(-x))).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = result, ["shape"] = t?["shape"], ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["softmax"] = new NSLBuiltinFunction("softmax", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ml.softmax() requires a tensor");
                    var t = args[0] as IDictionary<string, object?>;
                    var data = t?["data"] as double[] ?? Array.Empty<double>();
                    var max = data.Max(); var exp = data.Select(x => Math.Exp(x - max)).ToArray();
                    var sum = exp.Sum(); var result = exp.Select(x => x / sum).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = result, ["shape"] = t?["shape"], ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["reshape"] = new NSLBuiltinFunction("reshape", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("ml.reshape() requires tensor and shape");
                    var t = args[0] as IDictionary<string, object?>;
                    var shape = (args[1] as IList<object>)?.Select(x => (int)Convert.ToDouble(x)).ToArray();
                    return new Dictionary<string, object?> { ["type"] = "tensor", ["data"] = t?["data"], ["shape"] = shape, ["grad"] = null, ["requiresGrad"] = false };
                }),
                ["toList"] = new NSLBuiltinFunction("toList", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ml.toList() requires a tensor");
                    var t = args[0] as IDictionary<string, object?>;
                    var data = t?["data"] as double[] ?? Array.Empty<double>();
                    return data.Select(x => (object)x).ToList();
                })
            };
            _globals["ml"] = mlNamespace;

            // ===== PHASE 4: GUI NAMESPACE - Windows, Widgets, Events =====
            var guiWindows = new Dictionary<string, object>();
            var guiNamespace = new Dictionary<string, object?>
            {
                ["available"] = new NSLBuiltinFunction("available", (args) => OperatingSystem.IsWindows()),
                ["messageBox"] = new NSLBuiltinFunction("messageBox", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("gui.messageBox() requires message");
                    var msg = args[0]?.ToString() ?? "";
                    var title = args.Length > 1 ? args[1]?.ToString() ?? "NSL" : "NSL";
                    Console.WriteLine($"[GUI MessageBox] {title}: {msg}");
                    return true;
                }),
                ["input"] = new NSLBuiltinFunction("input", (args) => {
                    var prompt = args.Length > 0 ? args[0]?.ToString() ?? "" : "";
                    var defaultVal = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    Console.Write($"[GUI Input] {prompt} [{defaultVal}]: ");
                    var result = Console.ReadLine();
                    return string.IsNullOrEmpty(result) ? defaultVal : result;
                }),
                ["confirm"] = new NSLBuiltinFunction("confirm", (args) => {
                    var msg = args.Length > 0 ? args[0]?.ToString() ?? "Confirm?" : "Confirm?";
                    Console.Write($"[GUI Confirm] {msg} [Y/n]: ");
                    var result = Console.ReadLine()?.Trim().ToLower();
                    return result != "n" && result != "no";
                }),
                ["notify"] = new NSLBuiltinFunction("notify", (args) => {
                    var title = args.Length > 0 ? args[0]?.ToString() ?? "" : "";
                    var msg = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    Console.WriteLine($"[GUI Notification] {title}: {msg}");
                    return true;
                }),
                ["progress"] = new NSLBuiltinFunction("progress", (args) => {
                    var pct = args.Length > 0 ? ConvertToNumber(args[0]) : 0;
                    var msg = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    var barLen = 40; var filled = (int)(pct / 100 * barLen);
                    var bar = new string('█', filled) + new string('░', barLen - filled);
                    Console.Write($"\r[{bar}] {pct:F0}% {msg}");
                    if (pct >= 100) Console.WriteLine();
                    return true;
                }),
                ["clear"] = new NSLBuiltinFunction("clear", (args) => { Console.Clear(); return true; }),
                ["beep"] = new NSLBuiltinFunction("beep", (args) => {
                    var freq = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 800;
                    var duration = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 200;
                    if (OperatingSystem.IsWindows())
                        Console.Beep(freq, duration);
                    return true;
                })
            };
            _globals["gui"] = guiNamespace;

            // ===== PHASE 4: GAME NAMESPACE - Engine Primitives, ECS, Physics =====
            var gameEntities = new Dictionary<string, Dictionary<string, object?>>();
            var gameComponents = new Dictionary<string, List<string>>();
            var gameSystems = new List<(string name, object callback)>();
            var gameNamespace = new Dictionary<string, object?>
            {
                ["createEntity"] = new NSLBuiltinFunction("createEntity", (args) => {
                    var id = args.Length > 0 ? args[0]?.ToString() ?? Guid.NewGuid().ToString("N")[..8] : Guid.NewGuid().ToString("N")[..8];
                    gameEntities[id] = new Dictionary<string, object?> { ["id"] = id, ["active"] = true };
                    return id;
                }),
                ["destroyEntity"] = new NSLBuiltinFunction("destroyEntity", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("game.destroyEntity() requires entity id");
                    var id = args[0]?.ToString() ?? "";
                    gameEntities.Remove(id);
                    foreach (var comp in gameComponents.Keys.ToList()) gameComponents[comp].Remove(id);
                    return true;
                }),
                ["addComponent"] = new NSLBuiltinFunction("addComponent", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("game.addComponent() requires entity, component name, and data");
                    var entityId = args[0]?.ToString() ?? "";
                    var compName = args[1]?.ToString() ?? "";
                    var data = args[2];
                    if (gameEntities.TryGetValue(entityId, out var entity)) {
                        entity[compName] = data;
                        if (!gameComponents.ContainsKey(compName)) gameComponents[compName] = new List<string>();
                        if (!gameComponents[compName].Contains(entityId)) gameComponents[compName].Add(entityId);
                    }
                    return true;
                }),
                ["getComponent"] = new NSLBuiltinFunction("getComponent", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("game.getComponent() requires entity and component name");
                    var entityId = args[0]?.ToString() ?? "";
                    var compName = args[1]?.ToString() ?? "";
                    if (gameEntities.TryGetValue(entityId, out var entity) && entity.TryGetValue(compName, out var comp)) return comp;
                    return null;
                }),
                ["hasComponent"] = new NSLBuiltinFunction("hasComponent", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("game.hasComponent() requires entity and component name");
                    var entityId = args[0]?.ToString() ?? "";
                    var compName = args[1]?.ToString() ?? "";
                    return gameEntities.TryGetValue(entityId, out var entity) && entity.ContainsKey(compName);
                }),
                ["query"] = new NSLBuiltinFunction("query", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("game.query() requires component names");
                    var comps = args.Select(a => a?.ToString() ?? "").ToList();
                    var result = gameEntities.Where(e => comps.All(c => e.Value.ContainsKey(c))).Select(e => e.Key).ToList<object>();
                    return result;
                }),
                ["registerSystem"] = new NSLBuiltinFunction("registerSystem", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("game.registerSystem() requires name and callback");
                    var name = args[0]?.ToString() ?? "";
                    gameSystems.Add((name, args[1]!));
                    return true;
                }),
                ["tick"] = new NSLBuiltinFunction("tick", (args) => {
                    var dt = args.Length > 0 ? ConvertToNumber(args[0]) : 0.016;
                    foreach (var sys in gameSystems) {
                        if (sys.callback is NSLBuiltinFunction bf) bf.Call(new object?[] { dt });
                        else if (sys.callback is NSLFunction nf) CallUserFunction(nf, new object?[] { dt });
                    }
                    return true;
                }),
                ["entities"] = new NSLBuiltinFunction("entities", (args) => gameEntities.Keys.ToList<object>()),
                ["getEntity"] = new NSLBuiltinFunction("getEntity", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("game.getEntity() requires entity id");
                    var id = args[0]?.ToString() ?? "";
                    return gameEntities.TryGetValue(id, out var e) ? e : null;
                }),
                ["vec2"] = new NSLBuiltinFunction("vec2", (args) => {
                    var x = args.Length > 0 ? ConvertToNumber(args[0]) : 0;
                    var y = args.Length > 1 ? ConvertToNumber(args[1]) : 0;
                    return new Dictionary<string, object?> { ["x"] = x, ["y"] = y };
                }),
                ["vec3"] = new NSLBuiltinFunction("vec3", (args) => {
                    var x = args.Length > 0 ? ConvertToNumber(args[0]) : 0;
                    var y = args.Length > 1 ? ConvertToNumber(args[1]) : 0;
                    var z = args.Length > 2 ? ConvertToNumber(args[2]) : 0;
                    return new Dictionary<string, object?> { ["x"] = x, ["y"] = y, ["z"] = z };
                }),
                ["distance"] = new NSLBuiltinFunction("distance", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("game.distance() requires two vectors");
                    var a = args[0] as IDictionary<string, object?>; var b = args[1] as IDictionary<string, object?>;
                    var dx = ConvertToNumber(a?["x"]) - ConvertToNumber(b?["x"]);
                    var dy = ConvertToNumber(a?["y"]) - ConvertToNumber(b?["y"]);
                    var dz = (a?.ContainsKey("z") == true && b?.ContainsKey("z") == true) ? ConvertToNumber(a?["z"]) - ConvertToNumber(b?["z"]) : 0;
                    return Math.Sqrt(dx*dx + dy*dy + dz*dz);
                }),
                ["lerp"] = new NSLBuiltinFunction("lerp", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("game.lerp() requires a, b, and t");
                    var a = ConvertToNumber(args[0]); var b = ConvertToNumber(args[1]); var t = ConvertToNumber(args[2]);
                    return a + (b - a) * t;
                }),
                ["clamp"] = new NSLBuiltinFunction("clamp", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("game.clamp() requires value, min, and max");
                    var val = ConvertToNumber(args[0]); var min = ConvertToNumber(args[1]); var max = ConvertToNumber(args[2]);
                    return Math.Max(min, Math.Min(max, val));
                }),
                ["reset"] = new NSLBuiltinFunction("reset", (args) => {
                    gameEntities.Clear(); gameComponents.Clear(); gameSystems.Clear();
                    return true;
                })
            };
            _globals["game"] = gameNamespace;

            // ===== PHASE 5: CODEGEN NAMESPACE - Code Generation & Scaffolding =====
            var codegenNamespace = new Dictionary<string, object?>
            {
                ["function"] = new NSLBuiltinFunction("function", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "myFunction" : "myFunction";
                    var params_ = args.Length > 1 ? args[1] as IList<object> : new List<object>();
                    var body = args.Length > 2 ? args[2]?.ToString() ?? "// TODO" : "// TODO";
                    var lang = args.Length > 3 ? args[3]?.ToString() ?? "nsl" : "nsl";
                    var paramStr = string.Join(", ", params_?.Select(p => p?.ToString()) ?? Array.Empty<string>());
                    return lang switch {
                        "csharp" or "cs" => $"public void {name}({paramStr})\n{{\n    {body}\n}}",
                        "javascript" or "js" => $"function {name}({paramStr}) {{\n    {body}\n}}",
                        "typescript" or "ts" => $"function {name}({paramStr}): void {{\n    {body}\n}}",
                        "python" or "py" => $"def {name}({paramStr}):\n    {body}",
                        _ => $"fn {name}({paramStr}) {{\n    {body}\n}}"
                    };
                }),
                ["class"] = new NSLBuiltinFunction("class", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "MyClass" : "MyClass";
                    var fields = args.Length > 1 ? args[1] as IList<object> : new List<object>();
                    var lang = args.Length > 2 ? args[2]?.ToString() ?? "csharp" : "csharp";
                    var fieldLines = fields?.Select(f => f?.ToString() ?? "").ToList() ?? new List<string>();
                    return lang switch {
                        "csharp" or "cs" => $"public class {name}\n{{\n{string.Join("\n", fieldLines.Select(f => $"    public {f} {{ get; set; }}"))}\n}}",
                        "typescript" or "ts" => $"class {name} {{\n{string.Join("\n", fieldLines.Select(f => $"    {f};"))}\n}}",
                        "python" or "py" => $"class {name}:\n    def __init__(self):\n{string.Join("\n", fieldLines.Select(f => $"        self.{f} = None"))}",
                        _ => $"class {name} {{\n{string.Join("\n", fieldLines.Select(f => $"    {f}"))}\n}}"
                    };
                }),
                ["interface"] = new NSLBuiltinFunction("interface", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "IMyInterface" : "IMyInterface";
                    var methods = args.Length > 1 ? args[1] as IList<object> : new List<object>();
                    var lang = args.Length > 2 ? args[2]?.ToString() ?? "csharp" : "csharp";
                    var methodLines = methods?.Select(m => m?.ToString() ?? "").ToList() ?? new List<string>();
                    return lang switch {
                        "csharp" or "cs" => $"public interface {name}\n{{\n{string.Join("\n", methodLines.Select(m => $"    {m};"))}\n}}",
                        "typescript" or "ts" => $"interface {name} {{\n{string.Join("\n", methodLines.Select(m => $"    {m};"))}\n}}",
                        _ => $"interface {name} {{\n{string.Join("\n", methodLines)}\n}}"
                    };
                }),
                ["enum"] = new NSLBuiltinFunction("enum", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "MyEnum" : "MyEnum";
                    var values = args.Length > 1 ? args[1] as IList<object> : new List<object>();
                    var lang = args.Length > 2 ? args[2]?.ToString() ?? "csharp" : "csharp";
                    var valueList = values?.Select(v => v?.ToString() ?? "").ToList() ?? new List<string>();
                    return lang switch {
                        "csharp" or "cs" => $"public enum {name}\n{{\n    {string.Join(",\n    ", valueList)}\n}}",
                        "typescript" or "ts" => $"enum {name} {{\n    {string.Join(",\n    ", valueList)}\n}}",
                        "python" or "py" => $"class {name}(Enum):\n{string.Join("\n", valueList.Select((v, i) => $"    {v} = {i}"))}",
                        _ => $"enum {name} {{ {string.Join(", ", valueList)} }}"
                    };
                }),
                ["property"] = new NSLBuiltinFunction("property", (args) => {
                    var type = args.Length > 0 ? args[0]?.ToString() ?? "string" : "string";
                    var name = args.Length > 1 ? args[1]?.ToString() ?? "MyProperty" : "MyProperty";
                    var lang = args.Length > 2 ? args[2]?.ToString() ?? "csharp" : "csharp";
                    return lang switch {
                        "csharp" or "cs" => $"public {type} {name} {{ get; set; }}",
                        "typescript" or "ts" => $"{name}: {type};",
                        "python" or "py" => $"self.{name}: {type} = None",
                        _ => $"{type} {name}"
                    };
                }),
                ["test"] = new NSLBuiltinFunction("test", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "TestMethod" : "TestMethod";
                    var body = args.Length > 1 ? args[1]?.ToString() ?? "// Arrange\n    // Act\n    // Assert" : "// Arrange\n    // Act\n    // Assert";
                    var lang = args.Length > 2 ? args[2]?.ToString() ?? "csharp" : "csharp";
                    return lang switch {
                        "csharp" or "cs" => $"[Fact]\npublic void {name}()\n{{\n    {body}\n}}",
                        "javascript" or "js" => $"test('{name}', () => {{\n    {body}\n}});",
                        "python" or "py" => $"def test_{name.ToLower()}():\n    {body}",
                        _ => $"test \"{name}\" {{\n    {body}\n}}"
                    };
                }),
                ["file"] = new NSLBuiltinFunction("file", (args) => {
                    var type = args.Length > 0 ? args[0]?.ToString() ?? "class" : "class";
                    var name = args.Length > 1 ? args[1]?.ToString() ?? "MyFile" : "MyFile";
                    var ns = args.Length > 2 ? args[2]?.ToString() ?? "MyNamespace" : "MyNamespace";
                    return type switch {
                        "class" => $"namespace {ns};\n\npublic class {name}\n{{\n    public {name}()\n    {{\n    }}\n}}",
                        "interface" => $"namespace {ns};\n\npublic interface I{name}\n{{\n}}",
                        "record" => $"namespace {ns};\n\npublic record {name}();",
                        "enum" => $"namespace {ns};\n\npublic enum {name}\n{{\n}}",
                        _ => $"namespace {ns};\n\npublic class {name}\n{{\n}}"
                    };
                }),
                ["controller"] = new NSLBuiltinFunction("controller", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "My" : "My";
                    var actions = args.Length > 1 ? args[1] as IList<object> : new List<object> { "Get", "Post", "Put", "Delete" };
                    var actionCode = string.Join("\n\n", (actions ?? new List<object>()).Select(a => 
                        $"    [Http{a}]\n    public IActionResult {a}()\n    {{\n        return Ok();\n    }}"));
                    return $"[ApiController]\n[Route(\"api/[controller]\")]\npublic class {name}Controller : ControllerBase\n{{\n{actionCode}\n}}";
                }),
                ["dto"] = new NSLBuiltinFunction("dto", (args) => {
                    var name = args.Length > 0 ? args[0]?.ToString() ?? "MyDto" : "MyDto";
                    var fields = args.Length > 1 ? args[1] as IList<object> : new List<object>();
                    var fieldStr = string.Join(", ", (fields ?? new List<object>()).Select(f => f?.ToString() ?? ""));
                    return $"public record {name}({fieldStr});";
                })
            };
            _globals["codegen"] = codegenNamespace;

            // ===== PHASE 5: AST NAMESPACE - Parse, Transform, Emit =====
            var astNamespace = new Dictionary<string, object?>
            {
                ["parse"] = new NSLBuiltinFunction("parse", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ast.parse() requires code");
                    var code = args[0]?.ToString() ?? "";
                    var lang = args.Length > 1 ? args[1]?.ToString() ?? "nsl" : "nsl";
                    var tokens = new List<object>();
                    var lines = code.Split('\n');
                    for (int i = 0; i < lines.Length; i++) {
                        var line = lines[i].Trim();
                        if (string.IsNullOrEmpty(line)) continue;
                        var node = new Dictionary<string, object?> { ["line"] = i + 1, ["text"] = line, ["type"] = "statement" };
                        if (line.StartsWith("fn ") || line.StartsWith("function ") || line.StartsWith("def ")) node["type"] = "function";
                        else if (line.StartsWith("class ")) node["type"] = "class";
                        else if (line.StartsWith("let ") || line.StartsWith("var ") || line.StartsWith("const ")) node["type"] = "declaration";
                        else if (line.StartsWith("if ") || line.StartsWith("if(")) node["type"] = "conditional";
                        else if (line.StartsWith("for ") || line.StartsWith("while ") || line.StartsWith("foreach ")) node["type"] = "loop";
                        else if (line.StartsWith("return ")) node["type"] = "return";
                        else if (line.StartsWith("import ") || line.StartsWith("using ") || line.StartsWith("from ")) node["type"] = "import";
                        else if (line.StartsWith("//") || line.StartsWith("#") || line.StartsWith("/*")) node["type"] = "comment";
                        tokens.Add(node);
                    }
                    return new Dictionary<string, object?> { ["language"] = lang, ["nodeCount"] = tokens.Count, ["nodes"] = tokens };
                }),
                ["find"] = new NSLBuiltinFunction("find", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("ast.find() requires ast and type");
                    var ast = args[0] as IDictionary<string, object?>;
                    var type = args[1]?.ToString() ?? "";
                    var nodes = ast?["nodes"] as IList<object> ?? new List<object>();
                    return nodes.Where(n => (n as IDictionary<string, object?>)?["type"]?.ToString() == type).ToList<object>();
                }),
                ["transform"] = new NSLBuiltinFunction("transform", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("ast.transform() requires ast, type, and transformer function");
                    var ast = args[0] as IDictionary<string, object?>;
                    var type = args[1]?.ToString() ?? "";
                    var transformer = args[2];
                    var nodes = (ast?["nodes"] as IList<object>)?.ToList() ?? new List<object>();
                    for (int i = 0; i < nodes.Count; i++) {
                        var node = nodes[i] as IDictionary<string, object?>;
                        if (node?["type"]?.ToString() == type) {
                            if (transformer is NSLBuiltinFunction bf) nodes[i] = bf.Call(new object?[] { node }) ?? node;
                            else if (transformer is NSLFunction nf) nodes[i] = CallUserFunction(nf, new object?[] { node }) ?? node;
                        }
                    }
                    return new Dictionary<string, object?> { ["language"] = ast?["language"], ["nodeCount"] = nodes.Count, ["nodes"] = nodes };
                }),
                ["emit"] = new NSLBuiltinFunction("emit", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("ast.emit() requires ast");
                    var ast = args[0] as IDictionary<string, object?>;
                    var nodes = ast?["nodes"] as IList<object> ?? new List<object>();
                    var lines = nodes.Select(n => (n as IDictionary<string, object?>)?["text"]?.ToString() ?? "").ToList();
                    return string.Join("\n", lines);
                }),
                ["insert"] = new NSLBuiltinFunction("insert", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("ast.insert() requires ast, index, and node");
                    var ast = args[0] as IDictionary<string, object?>;
                    var index = (int)ConvertToNumber(args[1]);
                    var newNode = args[2] as IDictionary<string, object?> ?? new Dictionary<string, object?> { ["text"] = args[2]?.ToString(), ["type"] = "statement", ["line"] = index };
                    var nodes = (ast?["nodes"] as IList<object>)?.ToList() ?? new List<object>();
                    nodes.Insert(Math.Min(index, nodes.Count), newNode);
                    return new Dictionary<string, object?> { ["language"] = ast?["language"], ["nodeCount"] = nodes.Count, ["nodes"] = nodes };
                }),
                ["remove"] = new NSLBuiltinFunction("remove", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("ast.remove() requires ast and predicate");
                    var ast = args[0] as IDictionary<string, object?>;
                    var predicate = args[1];
                    var nodes = (ast?["nodes"] as IList<object>)?.ToList() ?? new List<object>();
                    nodes = nodes.Where(n => {
                        object? result = null;
                        if (predicate is NSLBuiltinFunction bf) result = bf.Call(new object?[] { n });
                        else if (predicate is NSLFunction nf) result = CallUserFunction(nf, new object?[] { n });
                        return result is bool b && !b;
                    }).ToList();
                    return new Dictionary<string, object?> { ["language"] = ast?["language"], ["nodeCount"] = nodes.Count, ["nodes"] = nodes };
                }),
                ["rename"] = new NSLBuiltinFunction("rename", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("ast.rename() requires code, oldName, and newName");
                    var code = args[0]?.ToString() ?? "";
                    var oldName = args[1]?.ToString() ?? "";
                    var newName = args[2]?.ToString() ?? "";
                    return System.Text.RegularExpressions.Regex.Replace(code, $@"\b{oldName}\b", newName);
                })
            };
            _globals["ast"] = astNamespace;

            // ===== PHASE 5: PROJECT NAMESPACE - Templates & Scaffolding =====
            var projectTemplates = new Dictionary<string, Func<string, string, Dictionary<string, string>>> {
                ["console"] = (name, lang) => new Dictionary<string, string> {
                    [$"{name}/Program.cs"] = $"namespace {name};\n\nclass Program\n{{\n    static void Main(string[] args)\n    {{\n        Console.WriteLine(\"Hello, World!\");\n    }}\n}}",
                    [$"{name}/{name}.csproj"] = $"<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <TargetFramework>net8.0</TargetFramework>\n  </PropertyGroup>\n</Project>"
                },
                ["webapi"] = (name, lang) => new Dictionary<string, string> {
                    [$"{name}/Program.cs"] = $"var builder = WebApplication.CreateBuilder(args);\nbuilder.Services.AddControllers();\nvar app = builder.Build();\napp.MapControllers();\napp.Run();",
                    [$"{name}/Controllers/{name}Controller.cs"] = $"using Microsoft.AspNetCore.Mvc;\n\n[ApiController]\n[Route(\"api/[controller]\")]\npublic class {name}Controller : ControllerBase\n{{\n    [HttpGet]\n    public IActionResult Get() => Ok(\"Hello from {name}\");\n}}",
                    [$"{name}/{name}.csproj"] = $"<Project Sdk=\"Microsoft.NET.Sdk.Web\">\n  <PropertyGroup>\n    <TargetFramework>net8.0</TargetFramework>\n  </PropertyGroup>\n</Project>"
                },
                ["library"] = (name, lang) => new Dictionary<string, string> {
                    [$"{name}/{name}.cs"] = $"namespace {name};\n\npublic class {name}Service\n{{\n    public string Hello() => \"Hello from {name}\";\n}}",
                    [$"{name}/{name}.csproj"] = $"<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <TargetFramework>net8.0</TargetFramework>\n  </PropertyGroup>\n</Project>"
                },
                ["nsl"] = (name, lang) => new Dictionary<string, string> {
                    [$"{name}/main.nsl"] = $"# {name} - NSL Project\n\nprint(\"Hello from {name}!\")\n",
                    [$"{name}/lib.nsl"] = $"# {name} library\n\nfn greet(name) {{\n    return \"Hello, \" + name\n}}\n"
                }
            };
            var projectNamespace = new Dictionary<string, object?>
            {
                ["create"] = new NSLBuiltinFunction("create", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("project.create() requires project name");
                    var name = args[0]?.ToString() ?? "MyProject";
                    var template = args.Length > 1 ? args[1]?.ToString() ?? "console" : "console";
                    var basePath = args.Length > 2 ? args[2]?.ToString() ?? "." : ".";
                    if (!projectTemplates.TryGetValue(template, out var generator)) {
                        return new Dictionary<string, object?> { ["success"] = false, ["error"] = $"Unknown template: {template}" };
                    }
                    var files = generator(name, "csharp");
                    var created = new List<object>();
                    foreach (var kv in files) {
                        var fullPath = Path.Combine(basePath, kv.Key);
                        var dir = Path.GetDirectoryName(fullPath);
                        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
                        File.WriteAllText(fullPath, kv.Value);
                        created.Add(fullPath);
                    }
                    return new Dictionary<string, object?> { ["success"] = true, ["name"] = name, ["template"] = template, ["files"] = created };
                }),
                ["templates"] = new NSLBuiltinFunction("templates", (args) => projectTemplates.Keys.ToList<object>()),
                ["addFile"] = new NSLBuiltinFunction("addFile", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("project.addFile() requires path and content");
                    var path = args[0]?.ToString() ?? "";
                    var content = args[1]?.ToString() ?? "";
                    var dir = Path.GetDirectoryName(path);
                    if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
                    File.WriteAllText(path, content);
                    return true;
                }),
                ["scaffold"] = new NSLBuiltinFunction("scaffold", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("project.scaffold() requires type and name");
                    var type = args[0]?.ToString() ?? "";
                    var name = args[1]?.ToString() ?? "";
                    var basePath = args.Length > 2 ? args[2]?.ToString() ?? "." : ".";
                    var content = type switch {
                        "controller" => $"using Microsoft.AspNetCore.Mvc;\n\n[ApiController]\n[Route(\"api/[controller]\")]\npublic class {name}Controller : ControllerBase\n{{\n    [HttpGet]\n    public IActionResult Get() => Ok();\n}}",
                        "service" => $"public interface I{name}Service\n{{\n}}\n\npublic class {name}Service : I{name}Service\n{{\n}}",
                        "repository" => $"public interface I{name}Repository\n{{\n}}\n\npublic class {name}Repository : I{name}Repository\n{{\n}}",
                        "model" => $"public class {name}\n{{\n    public int Id {{ get; set; }}\n}}",
                        "dto" => $"public record {name}Dto();",
                        "test" => $"public class {name}Tests\n{{\n    [Fact]\n    public void Test1()\n    {{\n        Assert.True(true);\n    }}\n}}",
                        _ => $"public class {name}\n{{\n}}"
                    };
                    var fileName = $"{basePath}/{name}.cs";
                    File.WriteAllText(fileName, content);
                    return new Dictionary<string, object?> { ["path"] = fileName, ["type"] = type, ["name"] = name };
                })
            };
            _globals["project"] = projectNamespace;

            // ===== PHASE 5: LSP NAMESPACE - Language Server Features =====
            var lspNamespace = new Dictionary<string, object?>
            {
                ["symbols"] = new NSLBuiltinFunction("symbols", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("lsp.symbols() requires file path");
                    var path = args[0]?.ToString() ?? "";
                    if (!File.Exists(path)) return new List<object>();
                    var content = File.ReadAllText(path);
                    var symbols = new List<object>();
                    var lines = content.Split('\n');
                    for (int i = 0; i < lines.Length; i++) {
                        var line = lines[i];
                        var classMatch = System.Text.RegularExpressions.Regex.Match(line, @"(?:public|private|internal)?\s*(?:class|interface|struct|record|enum)\s+(\w+)");
                        if (classMatch.Success) symbols.Add(new Dictionary<string, object?> { ["name"] = classMatch.Groups[1].Value, ["kind"] = "class", ["line"] = i + 1 });
                        var methodMatch = System.Text.RegularExpressions.Regex.Match(line, @"(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?(?:\w+)\s+(\w+)\s*\(");
                        if (methodMatch.Success && !line.Contains("class ") && !line.Contains("new ")) symbols.Add(new Dictionary<string, object?> { ["name"] = methodMatch.Groups[1].Value, ["kind"] = "method", ["line"] = i + 1 });
                        var fnMatch = System.Text.RegularExpressions.Regex.Match(line, @"(?:fn|function|def)\s+(\w+)");
                        if (fnMatch.Success) symbols.Add(new Dictionary<string, object?> { ["name"] = fnMatch.Groups[1].Value, ["kind"] = "function", ["line"] = i + 1 });
                    }
                    return symbols;
                }),
                ["references"] = new NSLBuiltinFunction("references", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("lsp.references() requires path and symbol name");
                    var path = args[0]?.ToString() ?? "";
                    var symbol = args[1]?.ToString() ?? "";
                    var refs = new List<object>();
                    if (File.Exists(path)) {
                        var lines = File.ReadAllLines(path);
                        for (int i = 0; i < lines.Length; i++) {
                            if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], $@"\b{symbol}\b")) {
                                refs.Add(new Dictionary<string, object?> { ["file"] = path, ["line"] = i + 1, ["text"] = lines[i].Trim() });
                            }
                        }
                    } else if (Directory.Exists(path)) {
                        foreach (var file in Directory.GetFiles(path, "*.*", SearchOption.AllDirectories)) {
                            try {
                                var lines = File.ReadAllLines(file);
                                for (int i = 0; i < lines.Length; i++) {
                                    if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], $@"\b{symbol}\b")) {
                                        refs.Add(new Dictionary<string, object?> { ["file"] = file, ["line"] = i + 1, ["text"] = lines[i].Trim() });
                                    }
                                }
                            } catch { }
                        }
                    }
                    return refs;
                }),
                ["definition"] = new NSLBuiltinFunction("definition", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("lsp.definition() requires path and symbol");
                    var path = args[0]?.ToString() ?? "";
                    var symbol = args[1]?.ToString() ?? "";
                    var searchPath = Directory.Exists(path) ? path : Path.GetDirectoryName(path) ?? ".";
                    foreach (var file in Directory.GetFiles(searchPath, "*.*", SearchOption.AllDirectories)) {
                        try {
                            var lines = File.ReadAllLines(file);
                            for (int i = 0; i < lines.Length; i++) {
                                if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], $@"(?:class|interface|struct|fn|function|def)\s+{symbol}\b")) {
                                    return new Dictionary<string, object?> { ["file"] = file, ["line"] = i + 1, ["text"] = lines[i].Trim() };
                                }
                            }
                        } catch { }
                    }
                    return null;
                }),
                ["diagnostics"] = new NSLBuiltinFunction("diagnostics", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("lsp.diagnostics() requires file path");
                    var path = args[0]?.ToString() ?? "";
                    if (!File.Exists(path)) return new List<object>();
                    var diags = new List<object>();
                    var lines = File.ReadAllLines(path);
                    for (int i = 0; i < lines.Length; i++) {
                        var line = lines[i];
                        if (line.Contains("TODO")) diags.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["severity"] = "info", ["message"] = "TODO found", ["text"] = line.Trim() });
                        if (line.Contains("FIXME")) diags.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["severity"] = "warning", ["message"] = "FIXME found", ["text"] = line.Trim() });
                        if (line.Contains("HACK")) diags.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["severity"] = "warning", ["message"] = "HACK found", ["text"] = line.Trim() });
                        if (line.Contains("BUG")) diags.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["severity"] = "error", ["message"] = "BUG marker found", ["text"] = line.Trim() });
                        if (line.Length > 120) diags.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["severity"] = "hint", ["message"] = "Line exceeds 120 characters", ["text"] = line.Trim() });
                    }
                    return diags;
                }),
                ["hover"] = new NSLBuiltinFunction("hover", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("lsp.hover() requires symbol and context");
                    var symbol = args[0]?.ToString() ?? "";
                    var context = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    return new Dictionary<string, object?> { ["symbol"] = symbol, ["type"] = "unknown", ["documentation"] = $"Symbol: {symbol}" };
                }),
                ["complete"] = new NSLBuiltinFunction("complete", (args) => {
                    var prefix = args.Length > 0 ? args[0]?.ToString() ?? "" : "";
                    var keywords = new[] { "let", "fn", "if", "else", "for", "while", "return", "class", "import", "print", "true", "false", "null" };
                    return keywords.Where(k => k.StartsWith(prefix)).Select(k => new Dictionary<string, object?> { ["label"] = k, ["kind"] = "keyword" }).ToList<object>();
                })
            };
            _globals["lsp"] = lspNamespace;

            // ===== PHASE 5: META NAMESPACE - Metaprogramming & Self-Modification =====
            // Track eval executions for safety governance
            var evalLog = new List<Dictionary<string, object?>>();
            var metaNamespace = new Dictionary<string, object?>
            {
                ["eval"] = new NSLBuiltinFunction("eval", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.eval() requires code");
                    var code = args[0]?.ToString() ?? "";
                    var timestamp = DateTime.UtcNow.ToString("o");
                    
                    // Log this eval for traceability (Safety Contract: no eval without trace)
                    var logEntry = new Dictionary<string, object?> {
                        ["timestamp"] = timestamp,
                        ["code"] = code.Length > 100 ? code[..100] + "..." : code,
                        ["simMode"] = simActive
                    };
                    evalLog.Add(logEntry);
                    
                    // Execute normally - sim mode captures side effects at the file/system level
                    // The sim namespace handles file writes, not eval
                    try {
                        var lexer = new NSLLexer(code);
                        var tokens = lexer.Tokenize();
                        var parser = new NSLParser();
                        var ast = parser.Parse(tokens);
                        return Execute(ast);
                    } catch (Exception ex) {
                        return new Dictionary<string, object?> { ["error"] = ex.Message };
                    }
                }),
                // Validate-only mode - parses but doesn't execute
                ["validate"] = new NSLBuiltinFunction("validate", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.validate() requires code");
                    var code = args[0]?.ToString() ?? "";
                    try {
                        var lexer = new NSLLexer(code);
                        var tokens = lexer.Tokenize();
                        var parser = new NSLParser();
                        var ast = parser.Parse(tokens);
                        return new Dictionary<string, object?> { ["valid"] = true, ["tokens"] = tokens.Count() };
                    } catch (Exception ex) {
                        return new Dictionary<string, object?> { ["valid"] = false, ["error"] = ex.Message };
                    }
                }),
                ["evalLog"] = new NSLBuiltinFunction("evalLog", (args) => {
                    var limit = args.Length > 0 ? (int)ConvertToNumber(args[0]) : 10;
                    return evalLog.TakeLast(limit).ToList<object>();
                }),
                ["quote"] = new NSLBuiltinFunction("quote", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.quote() requires code");
                    return args[0]?.ToString() ?? "";
                }),
                ["gensym"] = new NSLBuiltinFunction("gensym", (args) => {
                    var prefix = args.Length > 0 ? args[0]?.ToString() ?? "g" : "g";
                    return $"{prefix}_{Guid.NewGuid().ToString("N")[..8]}";
                }),
                ["macroexpand"] = new NSLBuiltinFunction("macroexpand", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.macroexpand() requires code");
                    var code = args[0]?.ToString() ?? "";
                    return code;
                }),
                ["compile"] = new NSLBuiltinFunction("compile", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.compile() requires code");
                    var code = args[0]?.ToString() ?? "";
                    try {
                        var lexer = new NSLLexer(code);
                        var tokens = lexer.Tokenize().ToList();
                        var parser = new NSLParser();
                        var ast = parser.Parse(tokens);
                        return new Dictionary<string, object?> { ["success"] = true, ["tokens"] = tokens.Count, ["type"] = ast.GetType().Name };
                    } catch (Exception ex) {
                        return new Dictionary<string, object?> { ["success"] = false, ["error"] = ex.Message };
                    }
                }),
                ["typeof"] = new NSLBuiltinFunction("typeof", (args) => {
                    if (args.Length < 1) return "undefined";
                    var val = args[0];
                    if (val == null) return "null";
                    if (val is bool) return "boolean";
                    if (val is double || val is int || val is long) return "number";
                    if (val is string) return "string";
                    if (val is IList<object>) return "array";
                    if (val is IDictionary<string, object?>) return "object";
                    if (val is NSLBuiltinFunction) return "builtin";
                    if (val is NSLFunction) return "function";
                    return val.GetType().Name;
                }),
                ["reflect"] = new NSLBuiltinFunction("reflect", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.reflect() requires a value");
                    var val = args[0];
                    var result = new Dictionary<string, object?> { ["type"] = val?.GetType().Name ?? "null" };
                    if (val is IDictionary<string, object?> dict) result["keys"] = dict.Keys.ToList<object>();
                    if (val is IList<object> list) result["length"] = list.Count;
                    if (val is NSLFunction fn) { result["name"] = fn.Name; result["params"] = fn.Parameters.ToList<object>(); }
                    if (val is NSLBuiltinFunction bf) result["name"] = bf.Name;
                    return result;
                }),
                ["globals"] = new NSLBuiltinFunction("globals", (args) => _globals.Keys.ToList<object>()),
                ["defined"] = new NSLBuiltinFunction("defined", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.defined() requires symbol name");
                    var name = args[0]?.ToString() ?? "";
                    return _globals.ContainsKey(name);
                }),
                // Alias map - documents all lowercase aliases to their canonical camelCase names
                // This prevents semantic sprawl by making aliases explicit and trackable
                ["aliases"] = new NSLBuiltinFunction("aliases", (args) => {
                    return new Dictionary<string, object?> {
                        // String namespace aliases
                        ["string.startswith"] = "string.startsWith",
                        ["string.endswith"] = "string.endsWith",
                        ["string.indexof"] = "string.indexOf",
                        ["string.padleft"] = "string.padLeft",
                        ["string.padright"] = "string.padRight",
                        ["string.isempty"] = "string.isEmpty",
                        ["string.isblank"] = "string.isBlank",
                        ["string.fromcode"] = "string.fromCode",
                        // List namespace aliases  
                        ["list.foreach"] = "list.forEach",
                        ["list.indexof"] = "list.indexOf",
                        // Meta info
                        ["_policy"] = "Aliases are lowercase versions of camelCase canonicals. 1:1 mapping only.",
                        ["_count"] = 10
                    };
                }),
                // Get canonical name for any function (returns itself if already canonical)
                ["canonical"] = new NSLBuiltinFunction("canonical", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.canonical() requires a function name");
                    var name = args[0]?.ToString()?.ToLower() ?? "";
                    var aliasMap = new Dictionary<string, string> {
                        ["string.startswith"] = "string.startsWith",
                        ["string.endswith"] = "string.endsWith",
                        ["string.indexof"] = "string.indexOf",
                        ["string.padleft"] = "string.padLeft",
                        ["string.padright"] = "string.padRight",
                        ["string.isempty"] = "string.isEmpty",
                        ["string.isblank"] = "string.isBlank",
                        ["string.fromcode"] = "string.fromCode",
                        ["list.foreach"] = "list.forEach",
                        ["list.indexof"] = "list.indexOf"
                    };
                    return aliasMap.ContainsKey(name) ? aliasMap[name] : args[0]?.ToString() ?? "";
                }),
                // Describe a symbol - namespace, category, alias status, brief description
                ["describe"] = new NSLBuiltinFunction("describe", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("meta.describe() requires a symbol name");
                    var name = args[0]?.ToString() ?? "";
                    var parts = name.Split('.');
                    var ns = parts.Length > 1 ? parts[0] : "global";
                    var fn = parts.Length > 1 ? parts[1] : parts[0];
                    
                    // Check if it's an alias
                    var aliasMap = new Dictionary<string, string> {
                        ["string.startswith"] = "string.startsWith",
                        ["string.endswith"] = "string.endsWith"
                    };
                    var isAlias = aliasMap.ContainsKey(name.ToLower());
                    var canonical = isAlias ? aliasMap[name.ToLower()] : name;
                    
                    // Categorize by safety level
                    var dangerous = new HashSet<string> { "meta.eval", "ffi.load", "ffi.call", "buffer.create", "runtime.spawn" };
                    var mutating = new HashSet<string> { "file.write", "file.delete", "dir.create", "dir.delete", "refactor.commit" };
                    var category = dangerous.Contains(name) ? "dangerous" : mutating.Contains(name) ? "act" : "observe";
                    
                    return new Dictionary<string, object?> {
                        ["name"] = name,
                        ["namespace"] = ns,
                        ["function"] = fn,
                        ["isAlias"] = isAlias,
                        ["canonical"] = canonical,
                        ["category"] = category,
                        ["exists"] = _globals.ContainsKey(ns) || _globals.ContainsKey(name)
                    };
                }),
                ["generate"] = new NSLBuiltinFunction("generate", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("meta.generate() requires template and values");
                    var template = args[0]?.ToString() ?? "";
                    var values = args[1] as IDictionary<string, object?> ?? new Dictionary<string, object?>();
                    foreach (var kv in values) {
                        template = template.Replace($"${{{kv.Key}}}", kv.Value?.ToString() ?? "");
                        template = template.Replace($"${kv.Key}", kv.Value?.ToString() ?? "");
                    }
                    return template;
                }),
                // Capability map - categorizes all NSL functions by safety level
                ["capabilities"] = new NSLBuiltinFunction("capabilities", (args) => {
                    var observe = new List<object> { 
                        "file.read", "file.exists", "file.size", "file.lines", "file.cwd", "file.history",
                        "dir.list", "dir.files", "dir.dirs", "dir.tree", "dir.exists",
                        "git.status", "git.branch", "git.log", "git.diff", "git.isRepo",
                        "json.parse", "json.valid", "yaml.parse", "xml.parse",
                        "env.get", "env.all", "env.home", "env.user", "env.os",
                        "proc.list", "proc.exists", "proc.info",
                        "net.ping", "net.lookup", "net.localIp", "net.isOnline",
                        "http.get", "clip.paste", "zip.list",
                        "code.metrics", "code.symbols", "code.deps", "code.issues",
                        "lsp.symbols", "lsp.references", "lsp.definition", "lsp.diagnostics", "lsp.hover",
                        "meta.globals", "meta.defined", "meta.reflect", "meta.typeof", "meta.validate",
                        "refactor.usages", "refactor.preview", "refactor.diff"
                    };
                    var plan = new List<object> {
                        "sim.begin", "sim.write", "sim.delete", "sim.pending", "sim.diff",
                        "file.preview", "file.annotate",
                        "codegen.function", "codegen.class", "codegen.interface", "codegen.enum", "codegen.controller", "codegen.dto",
                        "ast.parse", "ast.find", "ast.transform", "ast.emit", "ast.rename",
                        "project.templates", "meta.compile", "meta.generate",
                        "refactor.rename", "refactor.replace", "refactor.extract", "refactor.batch"
                    };
                    var act = new List<object> {
                        "file.write", "file.append", "file.delete", "file.copy", "file.move", "file.restore",
                        "dir.create", "dir.delete",
                        "env.set", "proc.kill", "clip.copy",
                        "http.post", "http.download",
                        "zip.create", "zip.extract",
                        "sys.exec", "sys.shell",
                        "project.create", "project.scaffold", "project.addFile",
                        "sim.commit", "refactor.commit"
                    };
                    var dangerous = new List<object> {
                        "meta.eval", "ffi.load", "ffi.call", "buffer.create", "buffer.write",
                        "runtime.spawn", "runtime.exec"
                    };
                    return new Dictionary<string, object?> {
                        ["observe"] = new Dictionary<string, object?> { 
                            ["description"] = "Read-only operations, no side effects", 
                            ["functions"] = observe,
                            ["count"] = observe.Count
                        },
                        ["plan"] = new Dictionary<string, object?> { 
                            ["description"] = "Produces diffs/intents/previews, no mutations", 
                            ["functions"] = plan,
                            ["count"] = plan.Count
                        },
                        ["act"] = new Dictionary<string, object?> { 
                            ["description"] = "Mutates state, goes through safety pipeline", 
                            ["functions"] = act,
                            ["count"] = act.Count
                        },
                        ["dangerous"] = new Dictionary<string, object?> { 
                            ["description"] = "Requires explicit gating, can bypass safety", 
                            ["functions"] = dangerous,
                            ["count"] = dangerous.Count
                        },
                        ["total"] = observe.Count + plan.Count + act.Count + dangerous.Count
                    };
                })
            };
            _globals["meta"] = metaNamespace;

            // ===== PHASE 5: REFACTOR NAMESPACE - Multi-File Refactoring with Context =====
            // Default exclusion patterns for refactoring
            var defaultExcludeDirs = new[] { "bin", "obj", "node_modules", ".git", "dist", "build", "packages", ".vs", "__pycache__" };
            var refactorPendingChanges = new List<Dictionary<string, object?>>();
            var refactorLastCommit = new List<string>();  // Track files from last commit for undo
            
            // Helper: Check if path should be excluded
            Func<string, string[], bool> shouldExclude = (path, excludeDirs) => {
                var normalized = path.Replace('\\', '/').ToLower();
                return excludeDirs.Any(d => normalized.Contains($"/{d.ToLower()}/") || normalized.EndsWith($"/{d.ToLower()}"));
            };
            
            // Helper: Remove string literals and comments from content for matching (preserves structure)
            Func<string, string, string> maskStringsAndComments = (content, ext) => {
                var masked = content;
                // Mask string literals with placeholder of same length
                masked = System.Text.RegularExpressions.Regex.Replace(masked, @"""[^""\\]*(?:\\.[^""\\]*)*""", m => new string('_', m.Length));
                masked = System.Text.RegularExpressions.Regex.Replace(masked, @"'[^'\\]*(?:\\.[^'\\]*)*'", m => new string('_', m.Length));
                // Mask comments
                if (ext == ".cs" || ext == ".js" || ext == ".ts" || ext == ".java" || ext == ".cpp" || ext == ".c") {
                    masked = System.Text.RegularExpressions.Regex.Replace(masked, @"//[^\r\n]*", m => new string('_', m.Length));
                    masked = System.Text.RegularExpressions.Regex.Replace(masked, @"/\*[\s\S]*?\*/", m => new string('_', m.Length));
                } else if (ext == ".py" || ext == ".nsl") {
                    masked = System.Text.RegularExpressions.Regex.Replace(masked, @"#[^\r\n]*", m => new string('_', m.Length));
                }
                return masked;
            };
            
            // Helper: AST-backed rename for .nsl files using tokenizer (true semantic rename)
            Func<string, string, string, (string content, int count, bool semantic)> semanticRenameNSL = (content, oldName, newName) => {
                try {
                    var lexer = new NSLLexer(content);
                    var tokens = lexer.Tokenize().ToList();
                    var positions = new List<(int start, int length)>();
                    
                    // Find all Identifier tokens that match the old name exactly
                    foreach (var token in tokens) {
                        if (token.Type == NSL.Core.Tokens.TokenType.Identifier && token.Value == oldName) {
                            // Calculate position in original content
                            // Token has Line and Column, need to convert to absolute position
                            var lines = content.Split('\n');
                            var pos = 0;
                            for (int i = 0; i < token.Line - 1 && i < lines.Length; i++) {
                                pos += lines[i].Length + 1; // +1 for newline
                            }
                            pos += token.Column - 1;
                            positions.Add((pos, oldName.Length));
                        }
                    }
                    
                    if (positions.Count == 0) return (content, 0, true);
                    
                    // Apply replacements in reverse order to preserve positions
                    var newContent = content;
                    foreach (var (start, length) in positions.OrderByDescending(p => p.start)) {
                        if (start >= 0 && start + length <= newContent.Length) {
                            newContent = newContent.Substring(0, start) + newName + newContent.Substring(start + length);
                        }
                    }
                    return (newContent, positions.Count, true);
                } catch {
                    return (content, 0, false); // Fall back to textual if tokenization fails
                }
            };
            
            var refactorNamespace = new Dictionary<string, object?>
            {
                // Rename symbol across multiple files (AST-backed for .nsl, textual for others)
                ["rename"] = new NSLBuiltinFunction("rename", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("refactor.rename() requires path, oldName, newName");
                    var searchPath = args[0]?.ToString() ?? ".";
                    var oldName = args[1]?.ToString() ?? "";
                    var newName = args[2]?.ToString() ?? "";
                    var options = args.Length > 3 ? args[3] as IDictionary<string, object?> : null;
                    
                    // Parse options
                    var extensions = options?.ContainsKey("extensions") == true 
                        ? (options["extensions"] as IList<object>)?.Select(e => e?.ToString() ?? "").ToArray() ?? new[] { ".cs", ".nsl", ".js", ".ts" }
                        : new[] { ".cs", ".nsl", ".js", ".ts" };
                    var excludeDirs = options?.ContainsKey("excludeDirs") == true
                        ? (options["excludeDirs"] as IList<object>)?.Select(e => e?.ToString() ?? "").ToArray() ?? defaultExcludeDirs
                        : defaultExcludeDirs;
                    var skipStrings = options?.ContainsKey("skipStrings") != true || (options["skipStrings"] is bool b && b);
                    var skipComments = options?.ContainsKey("skipComments") != true || (options["skipComments"] is bool b2 && b2);
                    
                    var changes = new List<object>();
                    var semanticCount = 0;
                    var textualCount = 0;
                    var files = Directory.Exists(searchPath) 
                        ? Directory.GetFiles(searchPath, "*.*", SearchOption.AllDirectories)
                            .Where(f => extensions.Any(ext => f.EndsWith(ext, StringComparison.OrdinalIgnoreCase)))
                            .Where(f => !shouldExclude(f, excludeDirs)).ToArray()
                        : new[] { searchPath };
                    
                    foreach (var file in files) {
                        try {
                            var content = File.ReadAllText(file);
                            var ext = Path.GetExtension(file).ToLower();
                            string newContent;
                            int matchCount;
                            bool usedSemantic;
                            
                            // Use AST-backed semantic rename for .nsl files
                            if (ext == ".nsl") {
                                var result = semanticRenameNSL(content, oldName, newName);
                                newContent = result.content;
                                matchCount = result.count;
                                usedSemantic = result.semantic;
                                if (usedSemantic && matchCount > 0) semanticCount++;
                            } else {
                                // Use masked textual rename for other languages
                                var matchContent = (skipStrings || skipComments) ? maskStringsAndComments(content, ext) : content;
                                var matches = System.Text.RegularExpressions.Regex.Matches(matchContent, $@"\b{System.Text.RegularExpressions.Regex.Escape(oldName)}\b");
                                matchCount = matches.Count;
                                usedSemantic = false;
                                
                                if (matchCount > 0) {
                                    newContent = content;
                                    var offset = 0;
                                    foreach (System.Text.RegularExpressions.Match match in matches) {
                                        var pos = match.Index + offset;
                                        newContent = newContent.Substring(0, pos) + newName + newContent.Substring(pos + oldName.Length);
                                        offset += newName.Length - oldName.Length;
                                    }
                                    textualCount++;
                                } else {
                                    newContent = content;
                                }
                            }
                            
                            if (matchCount > 0) {
                                changes.Add(new Dictionary<string, object?> {
                                    ["file"] = file,
                                    ["matches"] = matchCount,
                                    ["semantic"] = usedSemantic
                                });
                                refactorPendingChanges.Add(new Dictionary<string, object?> {
                                    ["file"] = file,
                                    ["content"] = newContent,
                                    ["operation"] = usedSemantic ? "semantic-rename" : "textual-rename",
                                    ["details"] = $"{oldName} -> {newName}"
                                });
                            }
                        } catch { }
                    }
                    return new Dictionary<string, object?> { 
                        ["changes"] = changes, 
                        ["fileCount"] = changes.Count, 
                        ["pending"] = true,
                        ["semanticFiles"] = semanticCount,
                        ["textualFiles"] = textualCount,
                        ["options"] = new Dictionary<string, object?> {
                            ["skipStrings"] = skipStrings,
                            ["skipComments"] = skipComments,
                            ["excludeDirs"] = excludeDirs
                        }
                    };
                }),
                
                // Find and replace with context across files
                ["replace"] = new NSLBuiltinFunction("replace", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("refactor.replace() requires path, pattern, replacement");
                    var searchPath = args[0]?.ToString() ?? ".";
                    var pattern = args[1]?.ToString() ?? "";
                    var replacement = args[2]?.ToString() ?? "";
                    var isRegex = args.Length > 3 && args[3] is bool b && b;
                    var extensions = args.Length > 4 ? (args[4] as IList<object>)?.Select(e => e?.ToString() ?? "").ToArray() ?? new[] { "*" } : new[] { "*" };
                    
                    var changes = new List<object>();
                    var files = Directory.Exists(searchPath)
                        ? Directory.GetFiles(searchPath, "*.*", SearchOption.AllDirectories)
                            .Where(f => extensions.Contains("*") || extensions.Any(ext => f.EndsWith(ext, StringComparison.OrdinalIgnoreCase))).ToArray()
                        : new[] { searchPath };
                    
                    foreach (var file in files) {
                        try {
                            var content = File.ReadAllText(file);
                            var hasMatch = isRegex 
                                ? System.Text.RegularExpressions.Regex.IsMatch(content, pattern)
                                : content.Contains(pattern);
                            if (hasMatch) {
                                var newContent = isRegex
                                    ? System.Text.RegularExpressions.Regex.Replace(content, pattern, replacement)
                                    : content.Replace(pattern, replacement);
                                var matchCount = isRegex
                                    ? System.Text.RegularExpressions.Regex.Matches(content, pattern).Count
                                    : (content.Length - content.Replace(pattern, "").Length) / pattern.Length;
                                changes.Add(new Dictionary<string, object?> {
                                    ["file"] = file,
                                    ["matches"] = matchCount
                                });
                                refactorPendingChanges.Add(new Dictionary<string, object?> {
                                    ["file"] = file,
                                    ["content"] = newContent,
                                    ["operation"] = "replace",
                                    ["details"] = $"{pattern} -> {replacement}"
                                });
                            }
                        } catch { }
                    }
                    return new Dictionary<string, object?> { ["changes"] = changes, ["fileCount"] = changes.Count, ["pending"] = true };
                }),
                
                // Move/extract code between files
                ["extract"] = new NSLBuiltinFunction("extract", (args) => {
                    if (args.Length < 4) throw new NSLRuntimeException("refactor.extract() requires sourceFile, pattern, targetFile, wrapper");
                    var sourceFile = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var targetFile = args[2]?.ToString() ?? "";
                    var wrapper = args[3]?.ToString() ?? "${content}";
                    
                    if (!File.Exists(sourceFile)) throw new NSLRuntimeException($"Source file not found: {sourceFile}");
                    
                    var content = File.ReadAllText(sourceFile);
                    var match = System.Text.RegularExpressions.Regex.Match(content, pattern, System.Text.RegularExpressions.RegexOptions.Singleline);
                    if (!match.Success) return new Dictionary<string, object?> { ["success"] = false, ["error"] = "Pattern not found" };
                    
                    var extracted = match.Value;
                    var newSourceContent = content.Remove(match.Index, match.Length);
                    var targetContent = wrapper.Replace("${content}", extracted);
                    
                    refactorPendingChanges.Add(new Dictionary<string, object?> {
                        ["file"] = sourceFile,
                        ["content"] = newSourceContent,
                        ["operation"] = "extract-source"
                    });
                    refactorPendingChanges.Add(new Dictionary<string, object?> {
                        ["file"] = targetFile,
                        ["content"] = targetContent,
                        ["operation"] = "extract-target"
                    });
                    
                    return new Dictionary<string, object?> { ["success"] = true, ["extracted"] = extracted.Length, ["pending"] = true };
                }),
                
                // Preview pending changes
                ["preview"] = new NSLBuiltinFunction("preview", (args) => {
                    var result = new List<object>();
                    foreach (var change in refactorPendingChanges) {
                        result.Add(new Dictionary<string, object?> {
                            ["file"] = change["file"],
                            ["operation"] = change["operation"],
                            ["details"] = change.ContainsKey("details") ? change["details"] : null,
                            ["contentLength"] = (change["content"]?.ToString() ?? "").Length
                        });
                    }
                    return new Dictionary<string, object?> { ["pending"] = result, ["count"] = refactorPendingChanges.Count };
                }),
                
                // Diff pending changes - shows actual line-by-line changes
                ["diff"] = new NSLBuiltinFunction("diff", (args) => {
                    var verbose = args.Length > 0 && args[0] is bool b && b;
                    var diffs = new List<object>();
                    foreach (var change in refactorPendingChanges) {
                        var file = change["file"]?.ToString() ?? "";
                        var newContent = change["content"]?.ToString() ?? "";
                        var oldContent = File.Exists(file) ? File.ReadAllText(file) : "";
                        var oldLines = oldContent.Split('\n');
                        var newLines = newContent.Split('\n');
                        
                        // Compute actual line differences
                        var additions = new List<object>();
                        var deletions = new List<object>();
                        var changes = new List<object>();
                        
                        for (int i = 0; i < Math.Max(oldLines.Length, newLines.Length); i++) {
                            var oldLine = i < oldLines.Length ? oldLines[i] : null;
                            var newLine = i < newLines.Length ? newLines[i] : null;
                            if (oldLine != newLine) {
                                if (oldLine != null && newLine != null) {
                                    changes.Add(new Dictionary<string, object?> { 
                                        ["line"] = i + 1, 
                                        ["old"] = verbose ? oldLine.Trim() : (oldLine.Length > 60 ? oldLine.Trim().Substring(0, 60) + "..." : oldLine.Trim()),
                                        ["new"] = verbose ? newLine.Trim() : (newLine.Length > 60 ? newLine.Trim().Substring(0, 60) + "..." : newLine.Trim())
                                    });
                                } else if (oldLine == null) {
                                    additions.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["text"] = newLine?.Trim() });
                                } else {
                                    deletions.Add(new Dictionary<string, object?> { ["line"] = i + 1, ["text"] = oldLine?.Trim() });
                                }
                            }
                        }
                        
                        diffs.Add(new Dictionary<string, object?> {
                            ["file"] = file,
                            ["operation"] = change["operation"],
                            ["oldLines"] = oldLines.Length,
                            ["newLines"] = newLines.Length,
                            ["delta"] = newLines.Length - oldLines.Length,
                            ["changedLines"] = changes.Count,
                            ["addedLines"] = additions.Count,
                            ["deletedLines"] = deletions.Count,
                            ["changes"] = changes.Take(10).ToList(),  // First 10 changes
                            ["summary"] = $"+{additions.Count} -{deletions.Count} ~{changes.Count}"
                        });
                    }
                    return diffs;
                }),
                
                // Commit all pending changes (ROUTES THROUGH FILE.HISTORY SAFETY PIPELINE)
                ["commit"] = new NSLBuiltinFunction("commit", (args) => {
                    var reason = args.Length > 0 ? args[0]?.ToString() ?? "refactor" : "refactor";
                    var committed = new List<object>();
                    refactorLastCommit.Clear();  // Track for undo
                    foreach (var change in refactorPendingChanges) {
                        var file = change["file"]?.ToString() ?? "";
                        var content = change["content"]?.ToString() ?? "";
                        var operation = change.ContainsKey("operation") ? change["operation"]?.ToString() ?? "refactor" : "refactor";
                        try {
                            var dir = Path.GetDirectoryName(file);
                            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
                            // USE SAFETY PIPELINE: AtomicWrite with history capture and annotation
                            NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(file, content, $"{reason}: {operation}");
                            committed.Add(file);
                            refactorLastCommit.Add(file);  // Track for undo
                        } catch (Exception ex) {
                            return new Dictionary<string, object?> { ["success"] = false, ["error"] = ex.Message, ["committed"] = committed };
                        }
                    }
                    var count = refactorPendingChanges.Count;
                    refactorPendingChanges.Clear();
                    return new Dictionary<string, object?> { ["success"] = true, ["committed"] = committed, ["count"] = count, ["reason"] = reason, ["undoable"] = true };
                }),
                
                // Undo last commit - restores all files from last refactor using file.history
                ["undo"] = new NSLBuiltinFunction("undo", (args) => {
                    if (refactorLastCommit.Count == 0) {
                        return new Dictionary<string, object?> { ["success"] = false, ["error"] = "No refactor to undo" };
                    }
                    var restored = new List<object>();
                    foreach (var file in refactorLastCommit) {
                        try {
                            // Use file.history to restore previous version (index 0 = most recent pre-edit)
                            NSL.StandardLib.FileSystem.FileHistory.Instance.Restore(file, 0);
                            restored.Add(file);
                        } catch (Exception ex) {
                            return new Dictionary<string, object?> { ["success"] = false, ["error"] = ex.Message, ["restored"] = restored };
                        }
                    }
                    var count = refactorLastCommit.Count;
                    refactorLastCommit.Clear();
                    return new Dictionary<string, object?> { ["success"] = true, ["restored"] = restored, ["count"] = count };
                }),
                
                // Rollback (clear pending changes without applying)
                ["rollback"] = new NSLBuiltinFunction("rollback", (args) => {
                    var count = refactorPendingChanges.Count;
                    refactorPendingChanges.Clear();
                    return new Dictionary<string, object?> { ["cleared"] = count };
                }),
                
                // Find usages across files (enhanced)
                ["usages"] = new NSLBuiltinFunction("usages", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("refactor.usages() requires path and symbol");
                    var searchPath = args[0]?.ToString() ?? ".";
                    var symbol = args[1]?.ToString() ?? "";
                    var extensions = args.Length > 2 ? (args[2] as IList<object>)?.Select(e => e?.ToString() ?? "").ToArray() ?? new[] { ".cs", ".nsl", ".js", ".ts" } : new[] { ".cs", ".nsl", ".js", ".ts" };
                    
                    var usages = new List<object>();
                    var files = Directory.Exists(searchPath)
                        ? Directory.GetFiles(searchPath, "*.*", SearchOption.AllDirectories)
                            .Where(f => extensions.Any(ext => f.EndsWith(ext, StringComparison.OrdinalIgnoreCase))).ToArray()
                        : new[] { searchPath };
                    
                    foreach (var file in files) {
                        try {
                            var lines = File.ReadAllLines(file);
                            for (int i = 0; i < lines.Length; i++) {
                                if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], $@"\b{symbol}\b")) {
                                    var isDefinition = System.Text.RegularExpressions.Regex.IsMatch(lines[i], $@"(class|interface|struct|fn|function|def|let|var|const)\s+{symbol}\b");
                                    usages.Add(new Dictionary<string, object?> {
                                        ["file"] = file,
                                        ["line"] = i + 1,
                                        ["text"] = lines[i].Trim(),
                                        ["isDefinition"] = isDefinition
                                    });
                                }
                            }
                        } catch { }
                    }
                    return new Dictionary<string, object?> { 
                        ["symbol"] = symbol, 
                        ["usages"] = usages, 
                        ["count"] = usages.Count,
                        ["files"] = usages.Select(u => ((Dictionary<string, object?>)u)["file"]).Distinct().Count()
                    };
                }),
                
                // Batch operations on multiple files
                ["batch"] = new NSLBuiltinFunction("batch", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("refactor.batch() requires path and operations array");
                    var searchPath = args[0]?.ToString() ?? ".";
                    var operations = args[1] as IList<object> ?? new List<object>();
                    
                    var results = new List<object>();
                    foreach (var op in operations) {
                        if (op is IDictionary<string, object?> opDict) {
                            var type = opDict.ContainsKey("type") ? opDict["type"]?.ToString() : "";
                            var oldVal = opDict.ContainsKey("old") ? opDict["old"]?.ToString() : "";
                            var newVal = opDict.ContainsKey("new") ? opDict["new"]?.ToString() : "";
                            
                            if (type == "rename" && !string.IsNullOrEmpty(oldVal) && !string.IsNullOrEmpty(newVal)) {
                                var files = Directory.GetFiles(searchPath, "*.*", SearchOption.AllDirectories);
                                var count = 0;
                                foreach (var file in files) {
                                    try {
                                        var content = File.ReadAllText(file);
                                        if (System.Text.RegularExpressions.Regex.IsMatch(content, $@"\b{oldVal}\b")) {
                                            var newContent = System.Text.RegularExpressions.Regex.Replace(content, $@"\b{oldVal}\b", newVal);
                                            refactorPendingChanges.Add(new Dictionary<string, object?> {
                                                ["file"] = file,
                                                ["content"] = newContent,
                                                ["operation"] = "batch-rename"
                                            });
                                            count++;
                                        }
                                    } catch { }
                                }
                                results.Add(new Dictionary<string, object?> { ["operation"] = $"rename {oldVal} -> {newVal}", ["files"] = count });
                            }
                        }
                    }
                    return new Dictionary<string, object?> { ["operations"] = results, ["pending"] = refactorPendingChanges.Count };
                })
            };
            _globals["refactor"] = refactorNamespace;

            // ===== EDIT NAMESPACE - AI-Native Single & Multi-File Editing =====
            var editNamespace = new Dictionary<string, object?>
            {
                // Smart single-file operations
                ["replace"] = new NSLBuiltinFunction("replace", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("edit.replace() requires path, old, new");
                    var path = args[0]?.ToString() ?? "";
                    var oldText = args[1]?.ToString() ?? "";
                    var newText = args[2]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    var normalized = content.Replace("\r\n", "\n");
                    var normalizedOld = oldText.Replace("\r\n", "\n");
                    if (!normalized.Contains(normalizedOld)) return new Dictionary<string, object?> { ["success"] = false, ["reason"] = "Pattern not found" };
                    var result = normalized.Replace(normalizedOld, newText.Replace("\r\n", "\n"));
                    if (content.Contains("\r\n")) result = result.Replace("\n", "\r\n");
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "edit.replace");
                    return new Dictionary<string, object?> { ["success"] = true, ["path"] = path };
                }),
                ["replaceAll"] = new NSLBuiltinFunction("replaceAll", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("edit.replaceAll() requires path, pattern, replacement");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var replacement = args[2]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    var result = System.Text.RegularExpressions.Regex.Replace(content, pattern, replacement);
                    var count = System.Text.RegularExpressions.Regex.Matches(content, pattern).Count;
                    if (count == 0) return new Dictionary<string, object?> { ["success"] = false, ["count"] = 0.0 };
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "edit.replaceAll");
                    return new Dictionary<string, object?> { ["success"] = true, ["count"] = (double)count, ["path"] = path };
                }),
                ["insertBefore"] = new NSLBuiltinFunction("insertBefore", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("edit.insertBefore() requires path, pattern, text");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var text = args[2]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    int count = 0;
                    for (int i = lines.Count - 1; i >= 0; i--) {
                        if (lines[i].Contains(pattern) || System.Text.RegularExpressions.Regex.IsMatch(lines[i], pattern)) {
                            var indent = new string(lines[i].TakeWhile(char.IsWhiteSpace).ToArray());
                            lines.Insert(i, indent + text);
                            count++;
                        }
                    }
                    if (count == 0) return new Dictionary<string, object?> { ["success"] = false, ["count"] = 0.0 };
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join(Environment.NewLine, lines), "edit.insertBefore");
                    return new Dictionary<string, object?> { ["success"] = true, ["count"] = (double)count };
                }),
                ["insertAfter"] = new NSLBuiltinFunction("insertAfter", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("edit.insertAfter() requires path, pattern, text");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var text = args[2]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    int count = 0;
                    for (int i = lines.Count - 1; i >= 0; i--) {
                        if (lines[i].Contains(pattern) || System.Text.RegularExpressions.Regex.IsMatch(lines[i], pattern)) {
                            var indent = new string(lines[i].TakeWhile(char.IsWhiteSpace).ToArray());
                            lines.Insert(i + 1, indent + text);
                            count++;
                        }
                    }
                    if (count == 0) return new Dictionary<string, object?> { ["success"] = false, ["count"] = 0.0 };
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join(Environment.NewLine, lines), "edit.insertAfter");
                    return new Dictionary<string, object?> { ["success"] = true, ["count"] = (double)count };
                }),
                ["surround"] = new NSLBuiltinFunction("surround", (args) => {
                    if (args.Length < 4) throw new NSLRuntimeException("edit.surround() requires path, pattern, before, after");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    var before = args[2]?.ToString() ?? "";
                    var after = args[3]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    var matches = System.Text.RegularExpressions.Regex.Matches(content, pattern);
                    if (matches.Count == 0) return new Dictionary<string, object?> { ["success"] = false, ["count"] = 0.0 };
                    var result = System.Text.RegularExpressions.Regex.Replace(content, pattern, m => before + m.Value + after);
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "edit.surround");
                    return new Dictionary<string, object?> { ["success"] = true, ["count"] = (double)matches.Count };
                }),
                ["delete"] = new NSLBuiltinFunction("delete", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("edit.delete() requires path and pattern");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    var count = System.Text.RegularExpressions.Regex.Matches(content, pattern).Count;
                    if (count == 0) return new Dictionary<string, object?> { ["success"] = false, ["count"] = 0.0 };
                    var result = System.Text.RegularExpressions.Regex.Replace(content, pattern, "");
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "edit.delete");
                    return new Dictionary<string, object?> { ["success"] = true, ["count"] = (double)count };
                }),
                ["deleteLines"] = new NSLBuiltinFunction("deleteLines", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("edit.deleteLines() requires path and pattern");
                    var path = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    var originalCount = lines.Count;
                    lines.RemoveAll(l => l.Contains(pattern) || System.Text.RegularExpressions.Regex.IsMatch(l, pattern));
                    var removed = originalCount - lines.Count;
                    if (removed == 0) return new Dictionary<string, object?> { ["success"] = false, ["count"] = 0.0 };
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join(Environment.NewLine, lines), "edit.deleteLines");
                    return new Dictionary<string, object?> { ["success"] = true, ["count"] = (double)removed };
                }),
                // Multi-file operations
                ["files"] = new NSLBuiltinFunction("files", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("edit.files() requires paths array and operation");
                    var paths = args[0] as IList<object?> ?? throw new NSLRuntimeException("First argument must be array of paths");
                    var operation = args[1] as IDictionary<string, object?> ?? throw new NSLRuntimeException("Second argument must be operation object");
                    var results = new List<object?>();
                    foreach (var p in paths) {
                        var path = p?.ToString() ?? "";
                        if (!System.IO.File.Exists(path)) { results.Add(new Dictionary<string, object?> { ["path"] = path, ["success"] = false, ["reason"] = "File not found" }); continue; }
                        var content = System.IO.File.ReadAllText(path);
                        var modified = content;
                        var opType = operation.ContainsKey("type") ? operation["type"]?.ToString() : "";
                        var oldText = operation.ContainsKey("old") ? operation["old"]?.ToString() ?? "" : "";
                        var newText = operation.ContainsKey("new") ? operation["new"]?.ToString() ?? "" : "";
                        var pattern = operation.ContainsKey("pattern") ? operation["pattern"]?.ToString() ?? "" : "";
                        switch (opType) {
                            case "replace": modified = content.Replace(oldText, newText); break;
                            case "replaceAll": modified = System.Text.RegularExpressions.Regex.Replace(content, pattern, newText); break;
                            case "delete": modified = System.Text.RegularExpressions.Regex.Replace(content, pattern, ""); break;
                        }
                        if (modified != content) {
                            NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, modified, "edit.files");
                            results.Add(new Dictionary<string, object?> { ["path"] = path, ["success"] = true });
                        } else {
                            results.Add(new Dictionary<string, object?> { ["path"] = path, ["success"] = false, ["reason"] = "No changes" });
                        }
                    }
                    return new Dictionary<string, object?> { ["results"] = results, ["total"] = (double)paths.Count, ["modified"] = (double)results.Count(r => r is IDictionary<string, object?> d && d.ContainsKey("success") && d["success"] is bool b && b) };
                }),
                ["glob"] = new NSLBuiltinFunction("glob", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("edit.glob() requires glob pattern and operation");
                    var globPattern = args[0]?.ToString() ?? "*.cs";
                    var operation = args[1] as IDictionary<string, object?> ?? throw new NSLRuntimeException("Second argument must be operation object");
                    var baseDir = args.Length > 2 ? args[2]?.ToString() ?? "." : ".";
                    var files = System.IO.Directory.GetFiles(baseDir, globPattern, System.IO.SearchOption.AllDirectories);
                    var results = new List<object?>();
                    foreach (var path in files) {
                        if (!System.IO.File.Exists(path)) continue;
                        var content = System.IO.File.ReadAllText(path);
                        var modified = content;
                        var opType = operation.ContainsKey("type") ? operation["type"]?.ToString() : "";
                        var oldText = operation.ContainsKey("old") ? operation["old"]?.ToString() ?? "" : "";
                        var newText = operation.ContainsKey("new") ? operation["new"]?.ToString() ?? "" : "";
                        var pattern = operation.ContainsKey("pattern") ? operation["pattern"]?.ToString() ?? "" : "";
                        switch (opType) {
                            case "replace": modified = content.Replace(oldText, newText); break;
                            case "replaceAll": modified = System.Text.RegularExpressions.Regex.Replace(content, pattern, newText); break;
                            case "delete": modified = System.Text.RegularExpressions.Regex.Replace(content, pattern, ""); break;
                        }
                        if (modified != content) {
                            NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, modified, "edit.glob");
                            results.Add(new Dictionary<string, object?> { ["path"] = path, ["success"] = true });
                        }
                    }
                    return new Dictionary<string, object?> { ["results"] = results, ["filesScanned"] = (double)files.Length, ["modified"] = (double)results.Count };
                }),
                // Batch operations with preview
                ["batch"] = new NSLBuiltinFunction("batch", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("edit.batch() requires operations array");
                    var operations = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be array of operations");
                    var dryRun = args.Length > 1 && IsTruthy(args[1]);
                    var results = new List<object?>();
                    var changes = new Dictionary<string, string>();
                    foreach (var op in operations) {
                        if (!(op is IDictionary<string, object?> opDict)) continue;
                        var path = opDict.ContainsKey("path") ? opDict["path"]?.ToString() ?? "" : "";
                        var opType = opDict.ContainsKey("type") ? opDict["type"]?.ToString() ?? "" : "";
                        if (string.IsNullOrEmpty(path) || !System.IO.File.Exists(path)) { results.Add(new Dictionary<string, object?> { ["path"] = path, ["success"] = false, ["reason"] = "File not found" }); continue; }
                        var content = changes.ContainsKey(path) ? changes[path] : System.IO.File.ReadAllText(path);
                        var modified = content;
                        switch (opType) {
                            case "replace":
                                var oldText = opDict.ContainsKey("old") ? opDict["old"]?.ToString() ?? "" : "";
                                var newText = opDict.ContainsKey("new") ? opDict["new"]?.ToString() ?? "" : "";
                                modified = content.Replace(oldText, newText);
                                break;
                            case "replaceAll":
                                var pattern = opDict.ContainsKey("pattern") ? opDict["pattern"]?.ToString() ?? "" : "";
                                var replacement = opDict.ContainsKey("replacement") ? opDict["replacement"]?.ToString() ?? "" : "";
                                modified = System.Text.RegularExpressions.Regex.Replace(content, pattern, replacement);
                                break;
                            case "insertAt":
                                var line = opDict.ContainsKey("line") ? (int)ConvertToNumber(opDict["line"]) : 1;
                                var text = opDict.ContainsKey("text") ? opDict["text"]?.ToString() ?? "" : "";
                                var lines = content.Split('\n').ToList();
                                if (line > 0 && line <= lines.Count + 1) { lines.Insert(line - 1, text); modified = string.Join("\n", lines); }
                                break;
                            case "deleteLine":
                                var delLine = opDict.ContainsKey("line") ? (int)ConvertToNumber(opDict["line"]) : 0;
                                var delLines = content.Split('\n').ToList();
                                if (delLine > 0 && delLine <= delLines.Count) { delLines.RemoveAt(delLine - 1); modified = string.Join("\n", delLines); }
                                break;
                        }
                        if (modified != content) {
                            changes[path] = modified;
                            results.Add(new Dictionary<string, object?> { ["path"] = path, ["type"] = opType, ["success"] = true });
                        } else {
                            results.Add(new Dictionary<string, object?> { ["path"] = path, ["type"] = opType, ["success"] = false, ["reason"] = "No changes" });
                        }
                    }
                    if (!dryRun) {
                        foreach (var kv in changes) {
                            NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(kv.Key, kv.Value, "edit.batch");
                        }
                    }
                    return new Dictionary<string, object?> { ["operations"] = results, ["filesModified"] = (double)changes.Count, ["dryRun"] = dryRun };
                }),
                // Smart context-aware replace (finds closest match)
                ["smartReplace"] = new NSLBuiltinFunction("smartReplace", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("edit.smartReplace() requires path, old, new");
                    var path = args[0]?.ToString() ?? "";
                    var oldText = args[1]?.ToString() ?? "";
                    var newText = args[2]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    // Normalize line endings for comparison
                    var normalizedContent = content.Replace("\r\n", "\n");
                    var normalizedOld = oldText.Replace("\r\n", "\n").Trim();
                    // Try exact match first
                    if (normalizedContent.Contains(normalizedOld)) {
                        var result = normalizedContent.Replace(normalizedOld, newText.Replace("\r\n", "\n"));
                        if (content.Contains("\r\n")) result = result.Replace("\n", "\r\n");
                        NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "edit.smartReplace");
                        return new Dictionary<string, object?> { ["success"] = true, ["method"] = "exact" };
                    }
                    // Try with normalized whitespace
                    var wsNormalizedOld = System.Text.RegularExpressions.Regex.Replace(normalizedOld, @"\s+", " ");
                    var lines = normalizedContent.Split('\n');
                    for (int i = 0; i < lines.Length; i++) {
                        var wsNormalizedLine = System.Text.RegularExpressions.Regex.Replace(lines[i], @"\s+", " ");
                        if (wsNormalizedLine.Contains(wsNormalizedOld)) {
                            lines[i] = lines[i].Replace(lines[i].Trim(), newText.Replace("\r\n", "\n").Trim());
                            var result = string.Join("\n", lines);
                            if (content.Contains("\r\n")) result = result.Replace("\n", "\r\n");
                            NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "edit.smartReplace");
                            return new Dictionary<string, object?> { ["success"] = true, ["method"] = "whitespace_normalized", ["line"] = (double)(i + 1) };
                        }
                    }
                    // Try fuzzy match - find line with most common words
                    var oldWords = normalizedOld.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries).Where(w => w.Length > 2).ToHashSet();
                    int bestMatch = -1, bestScore = 0;
                    for (int i = 0; i < lines.Length; i++) {
                        var lineWords = lines[i].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).Where(w => w.Length > 2).ToHashSet();
                        var score = oldWords.Intersect(lineWords).Count();
                        if (score > bestScore && score >= oldWords.Count / 2) { bestScore = score; bestMatch = i; }
                    }
                    if (bestMatch >= 0) {
                        var indent = new string(lines[bestMatch].TakeWhile(char.IsWhiteSpace).ToArray());
                        lines[bestMatch] = indent + newText.Replace("\r\n", "\n").TrimStart();
                        var result = string.Join("\n", lines);
                        if (content.Contains("\r\n")) result = result.Replace("\n", "\r\n");
                        NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "edit.smartReplace");
                        return new Dictionary<string, object?> { ["success"] = true, ["method"] = "fuzzy", ["line"] = (double)(bestMatch + 1), ["confidence"] = (double)bestScore / oldWords.Count };
                    }
                    return new Dictionary<string, object?> { ["success"] = false, ["reason"] = "No match found" };
                })
            };
            _globals["edit"] = editNamespace;

            // ===== PATCH NAMESPACE - Unified Diff Operations =====
            var patchNamespace = new Dictionary<string, object?>
            {
                ["create"] = new NSLBuiltinFunction("create", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("patch.create() requires oldContent and newContent");
                    var oldContent = args[0]?.ToString() ?? "";
                    var newContent = args[1]?.ToString() ?? "";
                    var oldLines = oldContent.Replace("\r\n", "\n").Split('\n');
                    var newLines = newContent.Replace("\r\n", "\n").Split('\n');
                    var diff = new System.Text.StringBuilder();
                    diff.AppendLine("--- a/file");
                    diff.AppendLine("+++ b/file");
                    // Simple diff algorithm
                    int i = 0, j = 0;
                    while (i < oldLines.Length || j < newLines.Length) {
                        if (i < oldLines.Length && j < newLines.Length && oldLines[i] == newLines[j]) {
                            diff.AppendLine(" " + oldLines[i]);
                            i++; j++;
                        } else if (j < newLines.Length && (i >= oldLines.Length || !oldLines.Skip(i).Contains(newLines[j]))) {
                            diff.AppendLine("+" + newLines[j]);
                            j++;
                        } else if (i < oldLines.Length) {
                            diff.AppendLine("-" + oldLines[i]);
                            i++;
                        }
                    }
                    return diff.ToString();
                }),
                ["apply"] = new NSLBuiltinFunction("apply", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("patch.apply() requires content and patch");
                    var content = args[0]?.ToString() ?? "";
                    var patch = args[1]?.ToString() ?? "";
                    var lines = content.Replace("\r\n", "\n").Split('\n').ToList();
                    var patchLines = patch.Replace("\r\n", "\n").Split('\n');
                    int lineOffset = 0;
                    foreach (var patchLine in patchLines) {
                        if (patchLine.StartsWith("---") || patchLine.StartsWith("+++") || patchLine.StartsWith("@@") || string.IsNullOrEmpty(patchLine)) continue;
                        if (patchLine.StartsWith("-") && !patchLine.StartsWith("---")) {
                            var toRemove = patchLine.Substring(1);
                            var idx = lines.FindIndex(l => l == toRemove || l.Trim() == toRemove.Trim());
                            if (idx >= 0) { lines.RemoveAt(idx); lineOffset--; }
                        } else if (patchLine.StartsWith("+") && !patchLine.StartsWith("+++")) {
                            var toAdd = patchLine.Substring(1);
                            // Find context to insert near
                            var contextIdx = Math.Max(0, lines.Count + lineOffset);
                            if (contextIdx <= lines.Count) lines.Insert(contextIdx, toAdd);
                            else lines.Add(toAdd);
                            lineOffset++;
                        }
                    }
                    return string.Join("\n", lines);
                }),
                ["applyToFile"] = new NSLBuiltinFunction("applyToFile", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("patch.applyToFile() requires path and patch");
                    var path = args[0]?.ToString() ?? "";
                    var patch = args[1]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    var lines = content.Replace("\r\n", "\n").Split('\n').ToList();
                    var patchLines = patch.Replace("\r\n", "\n").Split('\n');
                    var applied = 0;
                    foreach (var patchLine in patchLines) {
                        if (patchLine.StartsWith("---") || patchLine.StartsWith("+++") || patchLine.StartsWith("@@") || string.IsNullOrEmpty(patchLine)) continue;
                        if (patchLine.StartsWith("-") && !patchLine.StartsWith("---")) {
                            var toRemove = patchLine.Substring(1);
                            var idx = lines.FindIndex(l => l == toRemove || l.Trim() == toRemove.Trim());
                            if (idx >= 0) { lines.RemoveAt(idx); applied++; }
                        } else if (patchLine.StartsWith("+") && !patchLine.StartsWith("+++")) {
                            var toAdd = patchLine.Substring(1);
                            lines.Add(toAdd);
                            applied++;
                        }
                    }
                    var result = string.Join("\n", lines);
                    if (content.Contains("\r\n")) result = result.Replace("\n", "\r\n");
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, result, "patch.applyToFile");
                    return new Dictionary<string, object?> { ["success"] = true, ["hunksApplied"] = (double)applied };
                }),
                ["parse"] = new NSLBuiltinFunction("parse", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("patch.parse() requires patch content");
                    var patch = args[0]?.ToString() ?? "";
                    var hunks = new List<object?>();
                    var additions = new List<object?>();
                    var deletions = new List<object?>();
                    foreach (var line in patch.Split('\n')) {
                        if (line.StartsWith("+") && !line.StartsWith("+++")) additions.Add(line.Substring(1));
                        else if (line.StartsWith("-") && !line.StartsWith("---")) deletions.Add(line.Substring(1));
                    }
                    return new Dictionary<string, object?> { ["additions"] = additions, ["deletions"] = deletions, ["addCount"] = (double)additions.Count, ["deleteCount"] = (double)deletions.Count };
                })
            };
            _globals["patch"] = patchNamespace;

            // ===== SEARCH NAMESPACE - Enhanced Code Search =====
            var searchNamespace = new Dictionary<string, object?>
            {
                ["files"] = new NSLBuiltinFunction("files", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("search.files() requires a pattern");
                    var pattern = args[0]?.ToString() ?? "";
                    var glob = args.Length > 1 ? args[1]?.ToString() ?? "*.*" : "*.*";
                    var baseDir = args.Length > 2 ? args[2]?.ToString() ?? "." : ".";
                    var results = new List<object?>();
                    var files = System.IO.Directory.GetFiles(baseDir, glob, System.IO.SearchOption.AllDirectories);
                    foreach (var file in files) {
                        try {
                            var content = System.IO.File.ReadAllText(file);
                            var lines = content.Split('\n');
                            for (int i = 0; i < lines.Length; i++) {
                                if (lines[i].Contains(pattern) || System.Text.RegularExpressions.Regex.IsMatch(lines[i], pattern)) {
                                    results.Add(new Dictionary<string, object?> { ["file"] = file, ["line"] = (double)(i + 1), ["text"] = lines[i].Trim(), ["context"] = lines[i] });
                                }
                            }
                        } catch { }
                    }
                    return new Dictionary<string, object?> { ["matches"] = results, ["count"] = (double)results.Count, ["filesSearched"] = (double)files.Length };
                }),
                ["symbols"] = new NSLBuiltinFunction("symbols", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("search.symbols() requires symbol name");
                    var symbol = args[0]?.ToString() ?? "";
                    var glob = args.Length > 1 ? args[1]?.ToString() ?? "*.cs" : "*.cs";
                    var baseDir = args.Length > 2 ? args[2]?.ToString() ?? "." : ".";
                    var results = new List<object?>();
                    var files = System.IO.Directory.GetFiles(baseDir, glob, System.IO.SearchOption.AllDirectories);
                    var defPattern = $@"\b(class|struct|interface|enum|fn|func|function|def|const|let|var)\s+{System.Text.RegularExpressions.Regex.Escape(symbol)}\b";
                    foreach (var file in files) {
                        try {
                            var lines = System.IO.File.ReadAllLines(file);
                            for (int i = 0; i < lines.Length; i++) {
                                if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], defPattern)) {
                                    results.Add(new Dictionary<string, object?> { ["file"] = file, ["line"] = (double)(i + 1), ["text"] = lines[i].Trim(), ["type"] = "definition" });
                                }
                            }
                        } catch { }
                    }
                    return new Dictionary<string, object?> { ["definitions"] = results, ["count"] = (double)results.Count };
                }),
                ["references"] = new NSLBuiltinFunction("references", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("search.references() requires symbol name");
                    var symbol = args[0]?.ToString() ?? "";
                    var glob = args.Length > 1 ? args[1]?.ToString() ?? "*.*" : "*.*";
                    var baseDir = args.Length > 2 ? args[2]?.ToString() ?? "." : ".";
                    var results = new List<object?>();
                    var files = System.IO.Directory.GetFiles(baseDir, glob, System.IO.SearchOption.AllDirectories);
                    var refPattern = $@"\b{System.Text.RegularExpressions.Regex.Escape(symbol)}\b";
                    foreach (var file in files) {
                        try {
                            var lines = System.IO.File.ReadAllLines(file);
                            for (int i = 0; i < lines.Length; i++) {
                                if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], refPattern)) {
                                    results.Add(new Dictionary<string, object?> { ["file"] = file, ["line"] = (double)(i + 1), ["text"] = lines[i].Trim() });
                                }
                            }
                        } catch { }
                    }
                    return new Dictionary<string, object?> { ["references"] = results, ["count"] = (double)results.Count };
                }),
                ["replace"] = new NSLBuiltinFunction("replace", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("search.replace() requires pattern, replacement, and glob");
                    var pattern = args[0]?.ToString() ?? "";
                    var replacement = args[1]?.ToString() ?? "";
                    var glob = args[2]?.ToString() ?? "*.cs";
                    var baseDir = args.Length > 3 ? args[3]?.ToString() ?? "." : ".";
                    var dryRun = args.Length > 4 && IsTruthy(args[4]);
                    var files = System.IO.Directory.GetFiles(baseDir, glob, System.IO.SearchOption.AllDirectories);
                    var results = new List<object?>();
                    int totalMatches = 0;
                    foreach (var file in files) {
                        try {
                            var content = System.IO.File.ReadAllText(file);
                            var matches = System.Text.RegularExpressions.Regex.Matches(content, pattern).Count;
                            if (matches > 0) {
                                totalMatches += matches;
                                if (!dryRun) {
                                    var modified = System.Text.RegularExpressions.Regex.Replace(content, pattern, replacement);
                                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(file, modified, "search.replace");
                                }
                                results.Add(new Dictionary<string, object?> { ["file"] = file, ["matches"] = (double)matches });
                            }
                        } catch { }
                    }
                    return new Dictionary<string, object?> { ["files"] = results, ["totalMatches"] = (double)totalMatches, ["filesModified"] = (double)results.Count, ["dryRun"] = dryRun };
                })
            };
            _globals["search"] = searchNamespace;

            // ===== TRANSFORM NAMESPACE - Code Transformations =====
            var transformNamespace = new Dictionary<string, object?>
            {
                ["sortImports"] = new NSLBuiltinFunction("sortImports", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("transform.sortImports() requires path");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var lines = System.IO.File.ReadAllLines(path).ToList();
                    var imports = new List<(int idx, string line)>();
                    for (int i = 0; i < lines.Count; i++) {
                        if (System.Text.RegularExpressions.Regex.IsMatch(lines[i], @"^(using|import|from\s+.+\s+import|#include)\s+")) {
                            imports.Add((i, lines[i]));
                        } else if (imports.Count > 0 && !string.IsNullOrWhiteSpace(lines[i])) break;
                    }
                    if (imports.Count == 0) return new Dictionary<string, object?> { ["success"] = false, ["reason"] = "No imports found" };
                    var sorted = imports.OrderBy(x => x.line).ToList();
                    for (int i = 0; i < imports.Count; i++) lines[imports[i].idx] = sorted[i].line;
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join(Environment.NewLine, lines), "transform.sortImports");
                    return new Dictionary<string, object?> { ["success"] = true, ["sorted"] = (double)imports.Count };
                }),
                ["removeUnusedImports"] = new NSLBuiltinFunction("removeUnusedImports", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("transform.removeUnusedImports() requires path");
                    var path = args[0]?.ToString() ?? "";
                    if (!System.IO.File.Exists(path)) throw new NSLRuntimeException($"File not found: {path}");
                    var content = System.IO.File.ReadAllText(path);
                    var lines = content.Split('\n').ToList();
                    var toRemove = new List<int>();
                    for (int i = 0; i < lines.Count; i++) {
                        var match = System.Text.RegularExpressions.Regex.Match(lines[i], @"^using\s+([\w.]+);");
                        if (match.Success) {
                            var ns = match.Groups[1].Value;
                            var shortName = ns.Contains('.') ? ns.Substring(ns.LastIndexOf('.') + 1) : ns;
                            var restOfCode = string.Join("\n", lines.Skip(i + 1));
                            if (!restOfCode.Contains(shortName)) toRemove.Add(i);
                        }
                    }
                    if (toRemove.Count == 0) return new Dictionary<string, object?> { ["success"] = false, ["removed"] = 0.0 };
                    foreach (var idx in toRemove.OrderByDescending(x => x)) lines.RemoveAt(idx);
                    NSL.StandardLib.FileSystem.FileHistory.Instance.AtomicWrite(path, string.Join("\n", lines), "transform.removeUnusedImports");
                    return new Dictionary<string, object?> { ["success"] = true, ["removed"] = (double)toRemove.Count };
                }),
                ["indent"] = new NSLBuiltinFunction("indent", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("transform.indent() requires content");
                    var content = args[0]?.ToString() ?? "";
                    var spaces = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 4;
                    var indent = new string(' ', spaces);
                    return string.Join("\n", content.Split('\n').Select(l => indent + l));
                }),
                ["dedent"] = new NSLBuiltinFunction("dedent", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("transform.dedent() requires content");
                    var content = args[0]?.ToString() ?? "";
                    var lines = content.Split('\n');
                    var minIndent = lines.Where(l => l.Trim().Length > 0).Select(l => l.TakeWhile(char.IsWhiteSpace).Count()).DefaultIfEmpty(0).Min();
                    return string.Join("\n", lines.Select(l => l.Length >= minIndent ? l.Substring(minIndent) : l));
                }),
                ["wrap"] = new NSLBuiltinFunction("wrap", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("transform.wrap() requires content, before, after");
                    var content = args[0]?.ToString() ?? "";
                    var before = args[1]?.ToString() ?? "";
                    var after = args[2]?.ToString() ?? "";
                    return before + content + after;
                }),
                ["extractFunction"] = new NSLBuiltinFunction("extractFunction", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("transform.extractFunction() requires code, name, params");
                    var code = args[0]?.ToString() ?? "";
                    var name = args[1]?.ToString() ?? "extracted";
                    var paramsStr = args[2]?.ToString() ?? "";
                    return $"fn {name}({paramsStr}) {{\n    {code.Replace("\n", "\n    ")}\n}}";
                }),
                ["toArrowFunction"] = new NSLBuiltinFunction("toArrowFunction", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("transform.toArrowFunction() requires function code");
                    var code = args[0]?.ToString() ?? "";
                    var match = System.Text.RegularExpressions.Regex.Match(code, @"(?:fn|func|function)\s+(\w+)\s*\(([^)]*)\)\s*\{\s*return\s+([^;]+);?\s*\}");
                    if (match.Success) {
                        var name = match.Groups[1].Value;
                        var parms = match.Groups[2].Value;
                        var body = match.Groups[3].Value;
                        return $"let {name} = ({parms}) => {body}";
                    }
                    return code;
                })
            };
            _globals["transform"] = transformNamespace;

            // ===== SNIPPET NAMESPACE - Code Generation =====
            var snippetNamespace = new Dictionary<string, object?>
            {
                ["function"] = new NSLBuiltinFunction("function", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("snippet.function() requires name");
                    var name = args[0]?.ToString() ?? "myFunction";
                    var paramsStr = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                    var body = args.Length > 2 ? args[2]?.ToString() ?? "    // TODO" : "    // TODO";
                    var returnType = args.Length > 3 ? args[3]?.ToString() : null;
                    var indent = "    ";
                    return $"fn {name}({paramsStr}) {{\n{indent}{body.Replace("\n", "\n" + indent)}\n}}";
                }),
                ["class"] = new NSLBuiltinFunction("class", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("snippet.class() requires name");
                    var name = args[0]?.ToString() ?? "MyClass";
                    var methods = args.Length > 1 && args[1] is IList<object?> m ? m.Select(x => x?.ToString() ?? "").ToList() : new List<string>();
                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine($"class {name} {{");
                    sb.AppendLine($"    constructor() {{");
                    sb.AppendLine($"        // Initialize");
                    sb.AppendLine($"    }}");
                    foreach (var method in methods) {
                        sb.AppendLine($"    fn {method}() {{");
                        sb.AppendLine($"        // TODO");
                        sb.AppendLine($"    }}");
                    }
                    sb.AppendLine("}");
                    return sb.ToString();
                }),
                ["test"] = new NSLBuiltinFunction("test", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("snippet.test() requires function name");
                    var fnName = args[0]?.ToString() ?? "myFunction";
                    var testName = $"test_{fnName}";
                    return $"fn {testName}() {{\n    // Arrange\n    let input = null\n    let expected = null\n\n    // Act\n    let result = {fnName}(input)\n\n    // Assert\n    assert(result == expected, \"{fnName} should return expected value\")\n    print(\"{testName} passed\")\n}}";
                }),
                ["csharpClass"] = new NSLBuiltinFunction("csharpClass", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("snippet.csharpClass() requires name");
                    var name = args[0]?.ToString() ?? "MyClass";
                    var ns = args.Length > 1 ? args[1]?.ToString() ?? "MyNamespace" : "MyNamespace";
                    var props = args.Length > 2 && args[2] is IList<object?> p ? p.Select(x => x?.ToString() ?? "").ToList() : new List<string>();
                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine($"namespace {ns};");
                    sb.AppendLine();
                    sb.AppendLine($"public class {name}");
                    sb.AppendLine("{");
                    foreach (var prop in props) {
                        sb.AppendLine($"    public string {prop} {{ get; set; }}");
                    }
                    sb.AppendLine();
                    sb.AppendLine($"    public {name}()");
                    sb.AppendLine("    {");
                    sb.AppendLine("    }");
                    sb.AppendLine("}");
                    return sb.ToString();
                }),
                ["csharpMethod"] = new NSLBuiltinFunction("csharpMethod", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("snippet.csharpMethod() requires name");
                    var name = args[0]?.ToString() ?? "MyMethod";
                    var returnType = args.Length > 1 ? args[1]?.ToString() ?? "void" : "void";
                    var paramsStr = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
                    var isAsync = args.Length > 3 && IsTruthy(args[3]);
                    var asyncPrefix = isAsync ? "async " : "";
                    var taskWrapper = isAsync ? (returnType == "void" ? "Task" : $"Task<{returnType}>") : returnType;
                    return $"public {asyncPrefix}{taskWrapper} {name}({paramsStr})\n{{\n    throw new NotImplementedException();\n}}";
                }),
                ["apiEndpoint"] = new NSLBuiltinFunction("apiEndpoint", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("snippet.apiEndpoint() requires method and path");
                    var method = args[0]?.ToString()?.ToUpper() ?? "GET";
                    var path = args[1]?.ToString() ?? "/api/resource";
                    var name = args.Length > 2 ? args[2]?.ToString() ?? "HandleRequest" : "HandleRequest";
                    return $"[Http{method}(\"{path}\")]\npublic async Task<IActionResult> {name}()\n{{\n    return Ok();\n}}";
                }),
                ["reactComponent"] = new NSLBuiltinFunction("reactComponent", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("snippet.reactComponent() requires name");
                    var name = args[0]?.ToString() ?? "MyComponent";
                    var props = args.Length > 1 && args[1] is IList<object?> p ? p.Select(x => x?.ToString() ?? "").ToList() : new List<string>();
                    var propsInterface = props.Count > 0 ? $"interface {name}Props {{\n{string.Join("\n", props.Select(p => $"  {p}: string;"))}\n}}\n\n" : "";
                    var propsArg = props.Count > 0 ? $"{{ {string.Join(", ", props)} }}: {name}Props" : "";
                    return $"{propsInterface}export function {name}({propsArg}) {{\n  return (\n    <div>\n      {name}\n    </div>\n  );\n}}";
                })
            };
            _globals["snippet"] = snippetNamespace;

            // ===== VALIDATE NAMESPACE - Input Validation =====
            var validateNamespace = new Dictionary<string, object?>
            {
                ["isEmail"] = new NSLBuiltinFunction("isEmail", (args) => {
                    if (args.Length < 1) return false;
                    var s = args[0]?.ToString() ?? "";
                    return System.Text.RegularExpressions.Regex.IsMatch(s, @"^[^@\s]+@[^@\s]+\.[^@\s]+$");
                }),
                ["isUrl"] = new NSLBuiltinFunction("isUrl", (args) => {
                    if (args.Length < 1) return false;
                    var s = args[0]?.ToString() ?? "";
                    return Uri.TryCreate(s, UriKind.Absolute, out var uri) && (uri.Scheme == "http" || uri.Scheme == "https");
                }),
                ["isUuid"] = new NSLBuiltinFunction("isUuid", (args) => {
                    if (args.Length < 1) return false;
                    return Guid.TryParse(args[0]?.ToString() ?? "", out _);
                }),
                ["isJson"] = new NSLBuiltinFunction("isJson", (args) => {
                    if (args.Length < 1) return false;
                    try { System.Text.Json.JsonDocument.Parse(args[0]?.ToString() ?? ""); return true; } catch { return false; }
                }),
                ["isNumber"] = new NSLBuiltinFunction("isNumber", (args) => {
                    if (args.Length < 1) return false;
                    return double.TryParse(args[0]?.ToString() ?? "", out _);
                }),
                ["isInteger"] = new NSLBuiltinFunction("isInteger", (args) => {
                    if (args.Length < 1) return false;
                    return long.TryParse(args[0]?.ToString() ?? "", out _);
                }),
                ["isAlpha"] = new NSLBuiltinFunction("isAlpha", (args) => {
                    if (args.Length < 1) return false;
                    var s = args[0]?.ToString() ?? "";
                    return !string.IsNullOrEmpty(s) && s.All(char.IsLetter);
                }),
                ["isAlphanumeric"] = new NSLBuiltinFunction("isAlphanumeric", (args) => {
                    if (args.Length < 1) return false;
                    var s = args[0]?.ToString() ?? "";
                    return !string.IsNullOrEmpty(s) && s.All(char.IsLetterOrDigit);
                }),
                ["isHex"] = new NSLBuiltinFunction("isHex", (args) => {
                    if (args.Length < 1) return false;
                    var s = args[0]?.ToString() ?? "";
                    return System.Text.RegularExpressions.Regex.IsMatch(s, @"^[0-9A-Fa-f]+$");
                }),
                ["isIpv4"] = new NSLBuiltinFunction("isIpv4", (args) => {
                    if (args.Length < 1) return false;
                    return System.Net.IPAddress.TryParse(args[0]?.ToString() ?? "", out var ip) && ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork;
                }),
                ["isIpv6"] = new NSLBuiltinFunction("isIpv6", (args) => {
                    if (args.Length < 1) return false;
                    return System.Net.IPAddress.TryParse(args[0]?.ToString() ?? "", out var ip) && ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetworkV6;
                }),
                ["isPath"] = new NSLBuiltinFunction("isPath", (args) => {
                    if (args.Length < 1) return false;
                    try { System.IO.Path.GetFullPath(args[0]?.ToString() ?? ""); return true; } catch { return false; }
                }),
                ["isEmpty"] = new NSLBuiltinFunction("isEmpty", (args) => {
                    if (args.Length < 1) return true;
                    var val = args[0];
                    if (val == null) return true;
                    if (val is string s) return string.IsNullOrEmpty(s);
                    if (val is IList<object?> list) return list.Count == 0;
                    if (val is IDictionary<string, object?> dict) return dict.Count == 0;
                    return false;
                }),
                ["isBlank"] = new NSLBuiltinFunction("isBlank", (args) => {
                    if (args.Length < 1) return true;
                    var s = args[0]?.ToString() ?? "";
                    return string.IsNullOrWhiteSpace(s);
                }),
                ["matches"] = new NSLBuiltinFunction("matches", (args) => {
                    if (args.Length < 2) return false;
                    var s = args[0]?.ToString() ?? "";
                    var pattern = args[1]?.ToString() ?? "";
                    try { return System.Text.RegularExpressions.Regex.IsMatch(s, pattern); } catch { return false; }
                }),
                ["inRange"] = new NSLBuiltinFunction("inRange", (args) => {
                    if (args.Length < 3) return false;
                    var val = ConvertToNumber(args[0]);
                    var min = ConvertToNumber(args[1]);
                    var max = ConvertToNumber(args[2]);
                    return val >= min && val <= max;
                }),
                ["minLength"] = new NSLBuiltinFunction("minLength", (args) => {
                    if (args.Length < 2) return false;
                    var s = args[0]?.ToString() ?? "";
                    var min = (int)ConvertToNumber(args[1]);
                    return s.Length >= min;
                }),
                ["maxLength"] = new NSLBuiltinFunction("maxLength", (args) => {
                    if (args.Length < 2) return false;
                    var s = args[0]?.ToString() ?? "";
                    var max = (int)ConvertToNumber(args[1]);
                    return s.Length <= max;
                })
            };
            _globals["validate"] = validateNamespace;

            // ===== TEST NAMESPACE - Testing & Assertions =====
            var testNamespace = new Dictionary<string, object?>
            {
                ["assertEquals"] = new NSLBuiltinFunction("assertEquals", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("test.assertEquals() requires expected and actual");
                    var expected = args[0];
                    var actual = args[1];
                    var message = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
                    var eq = Equals(expected, actual) || (expected?.ToString() == actual?.ToString());
                    if (!eq) throw new NSLRuntimeException($"Assertion failed: expected {expected}, got {actual}. {message}");
                    return true;
                }),
                ["assertNotEquals"] = new NSLBuiltinFunction("assertNotEquals", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("test.assertNotEquals() requires value1 and value2");
                    var v1 = args[0];
                    var v2 = args[1];
                    var message = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
                    if (Equals(v1, v2) || (v1?.ToString() == v2?.ToString())) throw new NSLRuntimeException($"Assertion failed: values should not be equal. {message}");
                    return true;
                }),
                ["assertTrue"] = new NSLBuiltinFunction("assertTrue", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("test.assertTrue() requires a condition");
                    var message = args.Length > 1 ? args[1]?.ToString() ?? "Expected true" : "Expected true";
                    if (!IsTruthy(args[0])) throw new NSLRuntimeException($"Assertion failed: {message}");
                    return true;
                }),
                ["assertFalse"] = new NSLBuiltinFunction("assertFalse", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("test.assertFalse() requires a condition");
                    var message = args.Length > 1 ? args[1]?.ToString() ?? "Expected false" : "Expected false";
                    if (IsTruthy(args[0])) throw new NSLRuntimeException($"Assertion failed: {message}");
                    return true;
                }),
                ["assertNull"] = new NSLBuiltinFunction("assertNull", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("test.assertNull() requires a value");
                    var message = args.Length > 1 ? args[1]?.ToString() ?? "Expected null" : "Expected null";
                    if (args[0] != null) throw new NSLRuntimeException($"Assertion failed: {message}");
                    return true;
                }),
                ["assertNotNull"] = new NSLBuiltinFunction("assertNotNull", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("test.assertNotNull() requires a value");
                    var message = args.Length > 1 ? args[1]?.ToString() ?? "Expected non-null" : "Expected non-null";
                    if (args[0] == null) throw new NSLRuntimeException($"Assertion failed: {message}");
                    return true;
                }),
                ["assertContains"] = new NSLBuiltinFunction("assertContains", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("test.assertContains() requires collection and item");
                    var collection = args[0];
                    var item = args[1];
                    var message = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
                    bool contains = false;
                    if (collection is string s) contains = s.Contains(item?.ToString() ?? "");
                    else if (collection is IList<object?> list) contains = list.Any(x => Equals(x, item) || x?.ToString() == item?.ToString());
                    if (!contains) throw new NSLRuntimeException($"Assertion failed: collection does not contain item. {message}");
                    return true;
                }),
                ["assertThrows"] = new NSLBuiltinFunction("assertThrows", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("test.assertThrows() requires a function");
                    var fn = args[0];
                    var expectedMsg = args.Length > 1 ? args[1]?.ToString() : null;
                    try {
                        if (fn is NSLFunction userFn) CallUserFunction(userFn, new object?[0]);
                        else if (fn is NSLBuiltinFunction builtin) builtin.Call(new object?[0]);
                        throw new NSLRuntimeException("Expected exception was not thrown");
                    } catch (NSLRuntimeException ex) {
                        if (expectedMsg != null && !ex.Message.Contains(expectedMsg)) throw new NSLRuntimeException($"Exception thrown but message didn't match. Expected: {expectedMsg}, Got: {ex.Message}");
                        return true;
                    }
                }),
                ["run"] = new NSLBuiltinFunction("run", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("test.run() requires name and function");
                    var name = args[0]?.ToString() ?? "test";
                    var fn = args[1];
                    var result = new Dictionary<string, object?> { ["name"] = name };
                    try {
                        if (fn is NSLFunction userFn) CallUserFunction(userFn, new object?[0]);
                        else if (fn is NSLBuiltinFunction builtin) builtin.Call(new object?[0]);
                        result["passed"] = true;
                        result["error"] = null;
                        Console.WriteLine($"✓ {name}");
                    } catch (Exception ex) {
                        result["passed"] = false;
                        result["error"] = ex.Message;
                        Console.WriteLine($"✗ {name}: {ex.Message}");
                    }
                    return result;
                }),
                ["suite"] = new NSLBuiltinFunction("suite", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("test.suite() requires name and tests array");
                    var suiteName = args[0]?.ToString() ?? "Test Suite";
                    var tests = args[1] as IList<object?> ?? new List<object?>();
                    Console.WriteLine($"\n=== {suiteName} ===");
                    int passed = 0, failed = 0;
                    var results = new List<object?>();
                    foreach (var test in tests) {
                        if (test is IDictionary<string, object?> t && t.ContainsKey("name") && t.ContainsKey("fn")) {
                            var name = t["name"]?.ToString() ?? "test";
                            var fn = t["fn"];
                            try {
                                if (fn is NSLFunction userFn) CallUserFunction(userFn, new object?[0]);
                                else if (fn is NSLBuiltinFunction builtin) builtin.Call(new object?[0]);
                                Console.WriteLine($"  ✓ {name}");
                                passed++;
                                results.Add(new Dictionary<string, object?> { ["name"] = name, ["passed"] = true });
                            } catch (Exception ex) {
                                Console.WriteLine($"  ✗ {name}: {ex.Message}");
                                failed++;
                                results.Add(new Dictionary<string, object?> { ["name"] = name, ["passed"] = false, ["error"] = ex.Message });
                            }
                        }
                    }
                    Console.WriteLine($"\nResults: {passed} passed, {failed} failed");
                    return new Dictionary<string, object?> { ["suite"] = suiteName, ["passed"] = (double)passed, ["failed"] = (double)failed, ["results"] = results };
                })
            };
            _globals["test"] = testNamespace;

            // ===== OBJECT NAMESPACE - Object Manipulation =====
            var objectNamespace = new Dictionary<string, object?>
            {
                ["merge"] = new NSLBuiltinFunction("merge", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("object.merge() requires at least 2 objects");
                    var result = new Dictionary<string, object?>();
                    foreach (var arg in args) {
                        if (arg is IDictionary<string, object?> dict) {
                            foreach (var kv in dict) result[kv.Key] = kv.Value;
                        }
                    }
                    return result;
                }),
                ["pick"] = new NSLBuiltinFunction("pick", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("object.pick() requires object and keys");
                    var obj = args[0] as IDictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be object");
                    var keys = args[1] as IList<object?> ?? throw new NSLRuntimeException("Second argument must be array of keys");
                    var result = new Dictionary<string, object?>();
                    foreach (var k in keys) {
                        var key = k?.ToString() ?? "";
                        if (obj.ContainsKey(key)) result[key] = obj[key];
                    }
                    return result;
                }),
                ["omit"] = new NSLBuiltinFunction("omit", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("object.omit() requires object and keys");
                    var obj = args[0] as IDictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be object");
                    var keys = args[1] as IList<object?> ?? throw new NSLRuntimeException("Second argument must be array of keys");
                    var keysToOmit = new HashSet<string>(keys.Select(k => k?.ToString() ?? ""));
                    var result = new Dictionary<string, object?>();
                    foreach (var kv in obj) {
                        if (!keysToOmit.Contains(kv.Key)) result[kv.Key] = kv.Value;
                    }
                    return result;
                }),
                ["clone"] = new NSLBuiltinFunction("clone", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("object.clone() requires an object");
                    var obj = args[0] as IDictionary<string, object?> ?? throw new NSLRuntimeException("Argument must be object");
                    return new Dictionary<string, object?>(obj);
                }),
                ["deepClone"] = new NSLBuiltinFunction("deepClone", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("object.deepClone() requires an object");
                    var json = System.Text.Json.JsonSerializer.Serialize(args[0]);
                    return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object?>>(json) ?? new Dictionary<string, object?>();
                }),
                ["defaults"] = new NSLBuiltinFunction("defaults", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("object.defaults() requires object and defaults");
                    var obj = args[0] as IDictionary<string, object?> ?? new Dictionary<string, object?>();
                    var defaults = args[1] as IDictionary<string, object?> ?? new Dictionary<string, object?>();
                    var result = new Dictionary<string, object?>(defaults);
                    foreach (var kv in obj) result[kv.Key] = kv.Value;
                    return result;
                }),
                ["has"] = new NSLBuiltinFunction("has", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("object.has() requires object and key");
                    var obj = args[0] as IDictionary<string, object?>;
                    var key = args[1]?.ToString() ?? "";
                    return obj?.ContainsKey(key) ?? false;
                }),
                ["get"] = new NSLBuiltinFunction("get", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("object.get() requires object and path");
                    var obj = args[0];
                    var path = args[1]?.ToString() ?? "";
                    var defaultVal = args.Length > 2 ? args[2] : null;
                    var parts = path.Split('.');
                    foreach (var part in parts) {
                        if (obj is IDictionary<string, object?> dict && dict.ContainsKey(part)) obj = dict[part];
                        else return defaultVal;
                    }
                    return obj;
                }),
                ["set"] = new NSLBuiltinFunction("set", (args) => {
                    if (args.Length < 3) throw new NSLRuntimeException("object.set() requires object, path, and value");
                    var obj = args[0] as IDictionary<string, object?> ?? throw new NSLRuntimeException("First argument must be object");
                    var path = args[1]?.ToString() ?? "";
                    var value = args[2];
                    var parts = path.Split('.');
                    var current = obj;
                    for (int i = 0; i < parts.Length - 1; i++) {
                        if (!current.ContainsKey(parts[i]) || !(current[parts[i]] is IDictionary<string, object?>)) {
                            current[parts[i]] = new Dictionary<string, object?>();
                        }
                        current = (IDictionary<string, object?>)current[parts[i]]!;
                    }
                    current[parts[^1]] = value;
                    return obj;
                }),
                ["entries"] = new NSLBuiltinFunction("entries", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("object.entries() requires an object");
                    var obj = args[0] as IDictionary<string, object?> ?? throw new NSLRuntimeException("Argument must be object");
                    return obj.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList();
                }),
                ["fromEntries"] = new NSLBuiltinFunction("fromEntries", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("object.fromEntries() requires entries array");
                    var entries = args[0] as IList<object?> ?? throw new NSLRuntimeException("Argument must be array");
                    var result = new Dictionary<string, object?>();
                    foreach (var entry in entries) {
                        if (entry is IList<object?> pair && pair.Count >= 2) {
                            result[pair[0]?.ToString() ?? ""] = pair[1];
                        }
                    }
                    return result;
                }),
                ["freeze"] = new NSLBuiltinFunction("freeze", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("object.freeze() requires an object");
                    return args[0];
                }),
                ["isEmpty"] = new NSLBuiltinFunction("isEmpty", (args) => {
                    if (args.Length < 1) return true;
                    var obj = args[0] as IDictionary<string, object?>;
                    return obj == null || obj.Count == 0;
                }),
                ["keys"] = new NSLBuiltinFunction("keys", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("object.keys() requires an object");
                    var obj = args[0] as IDictionary<string, object?> ?? throw new NSLRuntimeException("Argument must be object");
                    return obj.Keys.Select(k => (object?)k).ToList();
                }),
                ["values"] = new NSLBuiltinFunction("values", (args) => {
                    if (args.Length < 1) throw new NSLRuntimeException("object.values() requires an object");
                    var obj = args[0] as IDictionary<string, object?> ?? throw new NSLRuntimeException("Argument must be object");
                    return obj.Values.ToList();
                }),
                ["size"] = new NSLBuiltinFunction("size", (args) => {
                    if (args.Length < 1) return 0.0;
                    var obj = args[0] as IDictionary<string, object?>;
                    return (double)(obj?.Count ?? 0);
                })
            };
            _globals["object"] = objectNamespace;

            // ===== LOG NAMESPACE - Structured Logging =====
            var logNamespace = new Dictionary<string, object?>
            {
                ["info"] = new NSLBuiltinFunction("info", (args) => {
                    var msg = args.Length > 0 ? string.Join(" ", args.Select(a => a?.ToString() ?? "null")) : "";
                    Console.WriteLine($"[INFO] {msg}");
                    return null;
                }),
                ["warn"] = new NSLBuiltinFunction("warn", (args) => {
                    var msg = args.Length > 0 ? string.Join(" ", args.Select(a => a?.ToString() ?? "null")) : "";
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"[WARN] {msg}");
                    Console.ResetColor();
                    return null;
                }),
                ["error"] = new NSLBuiltinFunction("error", (args) => {
                    var msg = args.Length > 0 ? string.Join(" ", args.Select(a => a?.ToString() ?? "null")) : "";
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"[ERROR] {msg}");
                    Console.ResetColor();
                    return null;
                }),
                ["debug"] = new NSLBuiltinFunction("debug", (args) => {
                    var msg = args.Length > 0 ? string.Join(" ", args.Select(a => a?.ToString() ?? "null")) : "";
                    Console.ForegroundColor = ConsoleColor.Gray;
                    Console.WriteLine($"[DEBUG] {msg}");
                    Console.ResetColor();
                    return null;
                }),
                ["trace"] = new NSLBuiltinFunction("trace", (args) => {
                    var msg = args.Length > 0 ? string.Join(" ", args.Select(a => a?.ToString() ?? "null")) : "";
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"[TRACE] {msg}");
                    Console.ResetColor();
                    return null;
                }),
                ["success"] = new NSLBuiltinFunction("success", (args) => {
                    var msg = args.Length > 0 ? string.Join(" ", args.Select(a => a?.ToString() ?? "null")) : "";
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"[SUCCESS] {msg}");
                    Console.ResetColor();
                    return null;
                }),
                ["table"] = new NSLBuiltinFunction("table", (args) => {
                    if (args.Length < 1) return null;
                    var data = args[0];
                    if (data is IList<object?> list) {
                        if (list.Count > 0 && list[0] is IDictionary<string, object?> firstRow) {
                            var headers = firstRow.Keys.ToList();
                            Console.WriteLine(string.Join(" | ", headers));
                            Console.WriteLine(new string('-', headers.Count * 15));
                            foreach (var row in list) {
                                if (row is IDictionary<string, object?> dict) {
                                    Console.WriteLine(string.Join(" | ", headers.Select(h => dict.ContainsKey(h) ? dict[h]?.ToString() ?? "" : "")));
                                }
                            }
                        } else {
                            for (int i = 0; i < list.Count; i++) Console.WriteLine($"[{i}] {list[i]}");
                        }
                    } else if (data is IDictionary<string, object?> obj) {
                        foreach (var kv in obj) Console.WriteLine($"{kv.Key}: {kv.Value}");
                    }
                    return null;
                }),
                ["json"] = new NSLBuiltinFunction("json", (args) => {
                    if (args.Length < 1) return null;
                    var json = System.Text.Json.JsonSerializer.Serialize(args[0], new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                    Console.WriteLine(json);
                    return null;
                }),
                ["time"] = new NSLBuiltinFunction("time", (args) => {
                    var label = args.Length > 0 ? args[0]?.ToString() ?? "default" : "default";
                    Console.WriteLine($"[TIME] {label}: {DateTime.Now:HH:mm:ss.fff}");
                    return null;
                })
            };
            _globals["log"] = logNamespace;

            // ===== TYPE NAMESPACE - Type Checking & Conversion =====
            var typeNamespace = new Dictionary<string, object?>
            {
                ["of"] = new NSLBuiltinFunction("of", (args) => {
                    if (args.Length < 1) return "undefined";
                    var val = args[0];
                    if (val == null) return "null";
                    if (val is bool) return "boolean";
                    if (val is double || val is int || val is long || val is float) return "number";
                    if (val is string) return "string";
                    if (val is IList<object?>) return "array";
                    if (val is IDictionary<string, object?> dict) return dict.ContainsKey("__class__") ? dict["__class__"]?.ToString() ?? "object" : "object";
                    if (val is NSLBuiltinFunction || val is NSLFunction) return "function";
                    return val.GetType().Name.ToLower();
                }),
                ["isString"] = new NSLBuiltinFunction("isString", (args) => args.Length > 0 && args[0] is string),
                ["isNumber"] = new NSLBuiltinFunction("isNumber", (args) => args.Length > 0 && (args[0] is double || args[0] is int || args[0] is long || args[0] is float)),
                ["isBool"] = new NSLBuiltinFunction("isBool", (args) => args.Length > 0 && args[0] is bool),
                ["isArray"] = new NSLBuiltinFunction("isArray", (args) => args.Length > 0 && args[0] is IList<object?>),
                ["isObject"] = new NSLBuiltinFunction("isObject", (args) => args.Length > 0 && args[0] is IDictionary<string, object?>),
                ["isFunction"] = new NSLBuiltinFunction("isFunction", (args) => args.Length > 0 && (args[0] is NSLFunction || args[0] is NSLBuiltinFunction)),
                ["isNull"] = new NSLBuiltinFunction("isNull", (args) => args.Length == 0 || args[0] == null),
                ["isInstance"] = new NSLBuiltinFunction("isInstance", (args) => {
                    if (args.Length < 2) return false;
                    if (args[0] is IDictionary<string, object?> dict && dict.ContainsKey("__class__"))
                        return dict["__class__"]?.ToString() == args[1]?.ToString();
                    return false;
                }),
                ["toString"] = new NSLBuiltinFunction("toString", (args) => {
                    if (args.Length < 1) return "";
                    return args[0]?.ToString() ?? "null";
                }),
                ["toNumber"] = new NSLBuiltinFunction("toNumber", (args) => {
                    if (args.Length < 1) return 0.0;
                    if (args[0] is double d) return d;
                    if (args[0] is int i) return (double)i;
                    if (args[0] is string s && double.TryParse(s, out var n)) return n;
                    return 0.0;
                }),
                ["toBool"] = new NSLBuiltinFunction("toBool", (args) => {
                    if (args.Length < 1) return false;
                    return IsTruthy(args[0]);
                }),
                ["toArray"] = new NSLBuiltinFunction("toArray", (args) => {
                    if (args.Length < 1) return new List<object?>();
                    if (args[0] is IList<object?> list) return list;
                    if (args[0] is string s) return s.ToCharArray().Select(c => (object?)c.ToString()).ToList();
                    if (args[0] is IDictionary<string, object?> dict) return dict.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList();
                    return new List<object?> { args[0] };
                }),
                ["check"] = new NSLBuiltinFunction("check", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("type.check() requires value and expected type");
                    var val = args[0];
                    var expectedType = args[1]?.ToString()?.ToLower() ?? "";
                    var actualType = val == null ? "null" : 
                        val is bool ? "boolean" :
                        val is double || val is int ? "number" :
                        val is string ? "string" :
                        val is IList<object?> ? "array" :
                        val is IDictionary<string, object?> ? "object" :
                        val is NSLFunction || val is NSLBuiltinFunction ? "function" : "unknown";
                    if (actualType != expectedType)
                        throw new NSLRuntimeException($"Type error: expected {expectedType}, got {actualType}");
                    return val;
                }),
                ["coerce"] = new NSLBuiltinFunction("coerce", (args) => {
                    if (args.Length < 2) throw new NSLRuntimeException("type.coerce() requires value and target type");
                    var val = args[0];
                    var targetType = args[1]?.ToString()?.ToLower() ?? "";
                    return targetType switch {
                        "string" => val?.ToString() ?? "",
                        "number" => val is double d ? d : val is string s && double.TryParse(s, out var n) ? n : 0.0,
                        "boolean" or "bool" => IsTruthy(val),
                        "array" => val is IList<object?> l ? l : new List<object?> { val },
                        _ => val
                    };
                })
            };
            _globals["types"] = typeNamespace;

            // Utility functions
            _globals["typeof"] = new NSLBuiltinFunction("typeof", (args) => { if (args.Length < 1) return "undefined"; var val = args[0]; if (val == null) return "null"; if (val is bool) return "boolean"; if (val is double || val is int || val is long || val is float) return "number"; if (val is string) return "string"; if (val is IList<object>) return "array"; if (val is IDictionary<string, object?>) return "object"; if (val is NSLBuiltinFunction || val is NSLFunction) return "function"; return val.GetType().Name.ToLower(); });
            _globals["keys"] = new NSLBuiltinFunction("keys", (args) => { if (args.Length < 1 || !(args[0] is IDictionary<string, object?> dict)) throw new NSLRuntimeException("keys() requires an object"); return new List<object>(dict.Keys.Select(k => (object)k)); });
            _globals["values"] = new NSLBuiltinFunction("values", (args) => { if (args.Length < 1 || !(args[0] is IDictionary<string, object?> dict)) throw new NSLRuntimeException("values() requires an object"); return new List<object>(dict.Values.Select(v => v ?? (object)"null")); });
            _globals["inspect"] = new NSLBuiltinFunction("inspect", (args) => { if (args.Length < 1) return "undefined"; return System.Text.Json.JsonSerializer.Serialize(args[0], new System.Text.Json.JsonSerializerOptions { WriteIndented = true }); });
            _globals["assert"] = new NSLBuiltinFunction("assert", (args) => { if (args.Length < 1) throw new NSLRuntimeException("assert() requires a condition"); var condition = args[0]; var message = args.Length > 1 ? args[1]?.ToString() ?? "Assertion failed" : "Assertion failed"; if (condition == null || (condition is bool b && !b) || (condition is double d && d == 0)) throw new NSLRuntimeException(message); return true; });
            _globals["debug"] = new NSLBuiltinFunction("debug", (args) => { if (args.Length < 1) return null; Console.Error.WriteLine($"[DEBUG] {args[0]}"); return args[0]; });

            // Pretty-print function for structured output with proper formatting
            Func<object?, int, string> prettyFormat = null!;
            prettyFormat = (obj, indent) => {
                var prefix = new string(' ', indent * 2);
                if (obj == null) return "null";
                if (obj is string s) return s.Contains('\n') ? $"\n{prefix}  {s.Replace("\n", $"\n{prefix}  ")}" : s;
                if (obj is bool b) return b.ToString().ToLower();
                if (obj is double || obj is int || obj is long || obj is float) return obj.ToString() ?? "";
                if (obj is IList<object> list) {
                    if (list.Count == 0) return "[]";
                    if (list.All(x => x is string && ((string)x).Length < 20)) {
                        return string.Join(", ", list);
                    }
                    var items = list.Select(x => $"{prefix}  - {prettyFormat(x, indent + 1)}");
                    return $"\n{string.Join("\n", items)}";
                }
                if (obj is string[] arr) {
                    return string.Join(", ", arr);
                }
                if (obj is IDictionary<string, object?> dict) {
                    var lines = new List<string>();
                    foreach (var kv in dict) {
                        var val = prettyFormat(kv.Value, indent + 1);
                        lines.Add($"{prefix}  {kv.Key}: {val}");
                    }
                    return $"\n{string.Join("\n", lines)}";
                }
                return obj.ToString() ?? "";
            };
            
            _globals["pp"] = new NSLBuiltinFunction("pp", (args) => {
                if (args.Length < 1) return null;
                var formatted = prettyFormat(args[0], 0);
                Console.WriteLine(formatted.TrimStart('\n'));
                return null;
            });
            
            _globals["pretty"] = _globals["pp"]; // Alias

            // Contextual help system - layered, not encyclopedic
            _globals["help"] = new NSLBuiltinFunction("help", (args) => {
                var topic = args.Length > 0 ? args[0]?.ToString()?.ToLower() ?? "" : "";
                
                var namespaceHelp = new Dictionary<string, object?> {
                    ["file"] = new Dictionary<string, object?> {
                        ["summary"] = "File operations with atomic writes and history",
                        ["functions"] = new[] { "read", "write", "append", "delete", "copy", "move", "exists", "size", "lines", "history", "restore", "preview", "annotate" },
                        ["safety"] = "All writes go through atomic pipeline with history capture",
                        ["example"] = "file.write('out.txt', content)  # Atomic, tracked"
                    },
                    ["dir"] = new Dictionary<string, object?> {
                        ["summary"] = "Directory operations",
                        ["functions"] = new[] { "create", "delete", "list", "files", "dirs", "tree", "exists", "walk" },
                        ["example"] = "dir.list('src/')  # Returns file list"
                    },
                    ["sim"] = new Dictionary<string, object?> {
                        ["summary"] = "Simulation mode - preview changes before committing",
                        ["functions"] = new[] { "begin", "commit", "rollback", "pending", "diff", "write", "delete" },
                        ["safety"] = "Captures all file operations, user controls commit",
                        ["workflow"] = "sim.begin() → make changes → sim.pending() → sim.commit() or sim.rollback()",
                        ["example"] = "sim.begin()\nfile.write('test.txt', 'hello')\nsim.pending()  # Shows captured writes\nsim.commit()   # Actually writes"
                    },
                    ["refactor"] = new Dictionary<string, object?> {
                        ["summary"] = "Multi-file refactoring with preview and undo",
                        ["functions"] = new[] { "rename", "replace", "extract", "usages", "preview", "diff", "commit", "rollback", "undo", "batch" },
                        ["safety"] = "All changes staged, preview before commit, undo available",
                        ["semantic"] = ".nsl files use AST-backed rename (true semantic)",
                        ["workflow"] = "refactor.rename() → refactor.preview() → refactor.commit()",
                        ["example"] = "refactor.rename('src/', 'oldName', 'newName')\nrefactor.diff()  # See changes\nrefactor.commit()  # Apply"
                    },
                    ["meta"] = new Dictionary<string, object?> {
                        ["summary"] = "Metaprogramming, introspection, eval",
                        ["functions"] = new[] { "eval", "validate", "compile", "globals", "defined", "typeof", "reflect", "describe", "canonical", "aliases", "capabilities" },
                        ["safety"] = "eval logs all executions, validate is parse-only",
                        ["example"] = "meta.validate('let x = 1')  # Check syntax\nmeta.describe('file.write')  # Get info"
                    },
                    ["git"] = new Dictionary<string, object?> {
                        ["summary"] = "Git repository operations",
                        ["functions"] = new[] { "status", "branch", "branches", "log", "diff", "show", "isRepo", "root", "remote" },
                        ["example"] = "git.status()  # {clean, files, count}"
                    },
                    ["codegen"] = new Dictionary<string, object?> {
                        ["summary"] = "Code generation for multiple languages",
                        ["functions"] = new[] { "function", "class", "interface", "enum", "controller", "dto" },
                        ["example"] = "codegen.dto('User', ['string Name', 'int Age'])"
                    },
                    ["ast"] = new Dictionary<string, object?> {
                        ["summary"] = "AST parsing and manipulation",
                        ["functions"] = new[] { "parse", "find", "transform", "emit", "rename" },
                        ["example"] = "ast.parse('let x = 1', 'nsl')  # Get AST"
                    },
                    ["string"] = new Dictionary<string, object?> {
                        ["summary"] = "String manipulation",
                        ["functions"] = new[] { "upper", "lower", "trim", "split", "join", "contains", "startsWith", "endsWith", "replace", "substring", "repeat", "format", "isEmpty", "isBlank" },
                        ["aliases"] = "Lowercase aliases available: startswith, endswith, indexof, etc.",
                        ["example"] = "string.split('a,b,c', ',')  # ['a', 'b', 'c']"
                    },
                    ["json"] = new Dictionary<string, object?> {
                        ["summary"] = "JSON parsing and generation",
                        ["functions"] = new[] { "parse", "stringify", "pretty", "valid" },
                        ["example"] = "json.parse('{\"a\": 1}')  # {a: 1}"
                    },
                    ["html"] = new Dictionary<string, object?> {
                        ["summary"] = "HTML generation and templating",
                        ["functions"] = new[] { "tag", "div", "span", "p", "h1", "h2", "h3", "a", "img", "ul", "ol", "table", "form", "input", "button", "document", "escape", "join" },
                        ["example"] = "html.document('Title', html.h1('Hello') + html.p('World'))"
                    },
                    ["web"] = new Dictionary<string, object?> {
                        ["summary"] = "HTTP server for web development",
                        ["functions"] = new[] { "serve", "stop", "servers", "static" },
                        ["serve"] = "web.serve(port, handler) - Start server with request handler",
                        ["static"] = "web.static(dir, port) - Serve static files from directory",
                        ["example"] = "web.serve(8080, fn(req) { return html.h1('Hello') })"
                    },
                    ["repl"] = new Dictionary<string, object?> {
                        ["summary"] = "REPL/interactive development tools",
                        ["functions"] = new[] { "time", "bench", "reset", "clear" },
                        ["time"] = "repl.time(fn) - Time function execution",
                        ["bench"] = "repl.bench(fn, n) - Benchmark n iterations",
                        ["example"] = "repl.bench(myFunc, 1000)  # {iterations, avgMs}"
                    },
                    ["http"] = new Dictionary<string, object?> {
                        ["summary"] = "HTTP client for web requests",
                        ["functions"] = new[] { "get", "post", "download", "head" },
                        ["example"] = "http.get('https://api.example.com/data')"
                    },
                    ["df"] = new Dictionary<string, object?> {
                        ["summary"] = "DataFrame for data analysis (pandas-like)",
                        ["functions"] = new[] { "create", "fromCsv", "toCsv", "filter", "select", "sort", "groupBy", "join", "count", "sum", "mean", "min", "max" },
                        ["example"] = "df.create([{name: 'A', val: 1}, {name: 'B', val: 2}])"
                    },
                    ["safety"] = new Dictionary<string, object?> {
                        ["summary"] = "NSL Safety Contract",
                        ["principles"] = new[] { 
                            "No silent mutation - every change is visible",
                            "No eval without trace - all evals logged",
                            "Preview → Simulate → Commit → Rollback hierarchy",
                            "Atomic writes with history capture",
                            "User controls all commits"
                        },
                        ["dangerous"] = new[] { "meta.eval", "ffi.load", "ffi.call", "buffer.create", "runtime.spawn" },
                        ["governed"] = "Dangerous functions require explicit use and are logged"
                    },
                    ["output"] = new Dictionary<string, object?> {
                        ["summary"] = "Output formatting functions",
                        ["functions"] = new[] { "print", "pp", "pretty", "inspect", "debug" },
                        ["print"] = "print(x) - Default output, flat structure",
                        ["pp"] = "pp(x) - Pretty-print with newlines and indentation",
                        ["pretty"] = "pretty(x) - Alias for pp()",
                        ["inspect"] = "inspect(x) - Full JSON output",
                        ["debug"] = "debug(x) - Print to stderr with [DEBUG] prefix",
                        ["tip"] = "Use pp(help('topic')) for readable help output"
                    }
                };
                
                if (string.IsNullOrEmpty(topic)) {
                    // Overview
                    return new Dictionary<string, object?> {
                        ["summary"] = "NSL - AI-native scripting language",
                        ["namespaces"] = namespaceHelp.Keys.Where(k => k != "safety").ToList(),
                        ["usage"] = "help('namespace') for details, e.g. help('sim'), help('refactor')",
                        ["quickstart"] = new[] {
                            "file.read('path')     - Read file",
                            "file.write('path', x) - Write file (atomic)",
                            "sim.begin()           - Start simulation",
                            "refactor.rename()     - Multi-file rename",
                            "meta.globals()        - List all symbols"
                        },
                        ["safetyTip"] = "Use help('safety') for safety contract"
                    };
                }
                
                if (namespaceHelp.ContainsKey(topic)) {
                    return namespaceHelp[topic];
                }
                
                return new Dictionary<string, object?> { 
                    ["error"] = $"Unknown topic: {topic}",
                    ["available"] = namespaceHelp.Keys.ToList()
                };
            });

            // Auto-generate documentation from globals with full metadata
            // Structured documentation: name, sig, summary, danger, since
            var funcMeta = new Dictionary<string, (string sig, string summary, bool danger, string since)> {
                // Core functions
                ["print"] = ("(any...) -> null", "Print values to output", false, "0.1.0"),
                ["len"] = ("(array|string|dict) -> number", "Get length/size of collection", false, "0.1.0"),
                ["range"] = ("(start, end, step?) -> array", "Generate numeric range", false, "0.1.0"),
                ["typeof"] = ("(any) -> string", "Get type name of value", false, "0.1.0"),
                ["sqrt"] = ("(number) -> number", "Square root", false, "0.1.0"),
                ["abs"] = ("(number) -> number", "Absolute value", false, "0.1.0"),
                ["floor"] = ("(number) -> number", "Round down", false, "0.1.0"),
                ["ceil"] = ("(number) -> number", "Round up", false, "0.1.0"),
                ["round"] = ("(number, decimals?) -> number", "Round to decimals", false, "0.1.0"),
                ["min"] = ("(array|...numbers) -> number", "Minimum value", false, "0.1.0"),
                ["max"] = ("(array|...numbers) -> number", "Maximum value", false, "0.1.0"),
                ["sum"] = ("(array) -> number", "Sum of array", false, "0.1.0"),
                ["map"] = ("(array, fn) -> array", "Transform each element", false, "0.1.0"),
                ["filter"] = ("(array, fn) -> array", "Filter by predicate", false, "0.1.0"),
                ["reduce"] = ("(array, fn, init?) -> any", "Reduce to single value", false, "0.1.0"),
                ["keys"] = ("(dict) -> array", "Get dictionary keys", false, "0.1.0"),
                ["values"] = ("(dict) -> array", "Get dictionary values", false, "0.1.0"),
                ["help"] = ("(topic?) -> dict", "Get contextual help", false, "0.5.0"),
                ["docs"] = ("(filter?) -> dict", "Auto-generated API documentation", false, "0.9.0"),
                ["pp"] = ("(any) -> null", "Pretty-print with indentation", false, "0.5.0"),
                ["run"] = ("(command) -> string", "Execute shell command", true, "0.3.0"),
                // Dangerous functions
                ["eval"] = ("(code) -> any", "Execute NSL code dynamically", true, "0.2.0"),
            };
            
            var nsMeta = new Dictionary<string, (string summary, bool hasDanger, string since)> {
                ["file"] = ("File operations with atomic writes and history", false, "0.1.0"),
                ["dir"] = ("Directory operations", false, "0.1.0"),
                ["git"] = ("Git repository operations", false, "0.3.0"),
                ["http"] = ("HTTP client requests", false, "0.3.0"),
                ["json"] = ("JSON parsing and generation", false, "0.1.0"),
                ["string"] = ("String manipulation", false, "0.1.0"),
                ["list"] = ("List/array operations", false, "0.1.0"),
                ["sim"] = ("Simulation mode - preview before commit", false, "0.5.0"),
                ["refactor"] = ("Multi-file refactoring with undo", false, "0.6.0"),
                ["meta"] = ("Metaprogramming and introspection", true, "0.4.0"),
                ["html"] = ("HTML generation and templating", false, "0.9.5"),
                ["web"] = ("HTTP server for web development", false, "0.9.5"),
                ["repl"] = ("REPL/interactive development tools", false, "0.9.5"),
                ["df"] = ("DataFrame for data analysis", false, "0.8.0"),
                ["gpu"] = ("GPU-accelerated tensor operations", false, "0.7.0"),
                ["ast"] = ("AST parsing and manipulation", false, "0.6.0"),
                ["codegen"] = ("Code generation for multiple languages", false, "0.6.0"),
            };
            
            _globals["docs"] = new NSLBuiltinFunction("docs", (args) => {
                var filter = args.Length > 0 ? args[0]?.ToString()?.ToLower() ?? "" : "";
                var detailed = args.Length > 1 && IsTruthy(args[1]);
                var result = new Dictionary<string, object?>();
                
                foreach (var kvp in _globals)
                {
                    if (!string.IsNullOrEmpty(filter) && !kvp.Key.ToLower().Contains(filter))
                        continue;
                    
                    if (kvp.Value is NSLBuiltinFunction bf)
                    {
                        var info = new Dictionary<string, object?> {
                            ["type"] = "builtin_function",
                            ["name"] = bf.Name
                        };
                        if (funcMeta.TryGetValue(bf.Name, out var meta)) {
                            info["sig"] = meta.sig;
                            info["summary"] = meta.summary;
                            info["danger"] = meta.danger;
                            info["since"] = meta.since;
                        }
                        result[kvp.Key] = info;
                    }
                    else if (kvp.Value is Dictionary<string, object?> ns)
                    {
                        var funcs = new List<object?>();
                        foreach (var nkvp in ns) {
                            if (nkvp.Value is NSLBuiltinFunction nbf) {
                                if (detailed) {
                                    var finfo = new Dictionary<string, object?> { ["name"] = nkvp.Key };
                                    var fullName = $"{kvp.Key}.{nkvp.Key}";
                                    if (funcMeta.TryGetValue(nkvp.Key, out var fmeta)) {
                                        finfo["sig"] = fmeta.sig;
                                        finfo["summary"] = fmeta.summary;
                                        finfo["danger"] = fmeta.danger;
                                    }
                                    funcs.Add(finfo);
                                } else {
                                    funcs.Add(nkvp.Key);
                                }
                            }
                        }
                        if (funcs.Count > 0) {
                            var nsInfo = new Dictionary<string, object?> {
                                ["type"] = "namespace",
                                ["functions"] = funcs,
                                ["count"] = (double)funcs.Count
                            };
                            if (nsMeta.TryGetValue(kvp.Key, out var nsMd)) {
                                nsInfo["summary"] = nsMd.summary;
                                nsInfo["hasDanger"] = nsMd.hasDanger;
                                nsInfo["since"] = nsMd.since;
                            }
                            result[kvp.Key] = nsInfo;
                        }
                    }
                }
                
                return new Dictionary<string, object?> {
                    ["count"] = (double)result.Count,
                    ["filter"] = string.IsNullOrEmpty(filter) ? "none" : filter,
                    ["detailed"] = detailed,
                    ["symbols"] = result
                };
            });

            // REPL namespace for interactive development
            var replNamespace = new Dictionary<string, object?>
            {
                ["clear"] = new NSLBuiltinFunction("clear", (args) => {
                    Console.Clear();
                    return null;
                }),
                ["reset"] = new NSLBuiltinFunction("reset", (args) => {
                    lock (_astCacheLock) { _astCache.Clear(); }
                    return "AST cache cleared";
                }),
                ["time"] = new NSLBuiltinFunction("time", (args) => {
                    if (args.Length < 1 || !(args[0] is NSLFunction fn))
                        throw new NSLRuntimeException("repl.time() requires a function");
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    var result = InvokeCallable(fn, args.Skip(1).ToArray());
                    sw.Stop();
                    return new Dictionary<string, object?> {
                        ["result"] = result,
                        ["ms"] = sw.Elapsed.TotalMilliseconds
                    };
                }),
                ["bench"] = new NSLBuiltinFunction("bench", (args) => {
                    if (args.Length < 1 || !(args[0] is NSLFunction fn))
                        throw new NSLRuntimeException("repl.bench() requires a function");
                    var n = args.Length > 1 ? (int)ConvertToNumber(args[1]) : 1000;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < n; i++) InvokeCallable(fn, args.Skip(2).ToArray());
                    sw.Stop();
                    return new Dictionary<string, object?> {
                        ["iterations"] = (double)n,
                        ["totalMs"] = sw.Elapsed.TotalMilliseconds,
                        ["avgMs"] = sw.Elapsed.TotalMilliseconds / n
                    };
                })
            };
            _globals["repl"] = replNamespace;

            // Also keep simple claude() function for quick prompts (calls the namespace version)
            // This allows both claude("prompt") and claude.chat()

            // Run command with live output (prints to console)
            _globals["run"] = new NSLBuiltinFunction("run", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("run() requires a command argument");

                var command = args[0]?.ToString() ?? "";

                try
                {
                    var processInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = OperatingSystem.IsWindows() ? "cmd.exe" : "/bin/bash",
                        Arguments = OperatingSystem.IsWindows() ? $"/c {command}" : $"-c \"{command}\"",
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };

                    using var process = System.Diagnostics.Process.Start(processInfo);
                    if (process == null)
                        return false;

                    process.WaitForExit();
                    return process.ExitCode == 0;
                }
                catch
                {
                    return false;
                }
            });

            // Execute PowerShell command directly (Windows only, better for PS-specific commands)
            _globals["powershell"] = new NSLBuiltinFunction("powershell", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("powershell() requires a command argument");

                var command = args[0]?.ToString() ?? "";

                try
                {
                    var processInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "powershell.exe",
                        Arguments = $"-NoProfile -ExecutionPolicy Bypass -Command \"{command}\"",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };

                    using var process = System.Diagnostics.Process.Start(processInfo);
                    if (process == null)
                        throw new NSLRuntimeException("Failed to start PowerShell");

                    var output = process.StandardOutput.ReadToEnd();
                    var error = process.StandardError.ReadToEnd();
                    process.WaitForExit();

                    return new Dictionary<string, object>
                    {
                        ["output"] = output,
                        ["error"] = error,
                        ["exit_code"] = process.ExitCode,
                        ["success"] = process.ExitCode == 0
                    };
                }
                catch (Exception ex)
                {
                    return new Dictionary<string, object>
                    {
                        ["output"] = "",
                        ["error"] = ex.Message,
                        ["exit_code"] = -1,
                        ["success"] = false
                    };
                }
            });

            // Get environment variable (legacy - use env.get() instead)
            _globals["getenv"] = new NSLBuiltinFunction("getenv", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("getenv() requires a variable name");

                var name = args[0]?.ToString() ?? "";
                var value = Environment.GetEnvironmentVariable(name);
                return value ?? "";
            });

            // Set environment variable
            _globals["set_env"] = new NSLBuiltinFunction("set_env", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("set_env() requires name and value arguments");

                var name = args[0]?.ToString() ?? "";
                var value = args[1]?.ToString() ?? "";
                Environment.SetEnvironmentVariable(name, value);
                return true;
            });

            // Get all environment variables
            _globals["env_all"] = new NSLBuiltinFunction("env_all", (args) =>
            {
                var envVars = new Dictionary<string, object>();
                foreach (System.Collections.DictionaryEntry entry in Environment.GetEnvironmentVariables())
                {
                    envVars[entry.Key?.ToString() ?? ""] = entry.Value?.ToString() ?? "";
                }
                return envVars;
            });

            // Sleep/delay function
            _globals["sleep"] = new NSLBuiltinFunction("sleep", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("sleep() requires milliseconds argument");

                var ms = args[0] switch
                {
                    double d => (int)d,
                    int i => i,
                    long l => (int)l,
                    _ => throw new NSLRuntimeException("sleep() requires a numeric argument")
                };

                System.Threading.Thread.Sleep(ms);
                return true;
            });

            // Exit function
            _globals["exit"] = new NSLBuiltinFunction("exit", (args) =>
            {
                var code = 0;
                if (args.Length > 0)
                {
                    code = args[0] switch
                    {
                        double d => (int)d,
                        int i => i,
                        long l => (int)l,
                        _ => 0
                    };
                }
                Environment.Exit(code);
                return null; // Never reached
            });

            // Get current timestamp
            _globals["timestamp"] = new NSLBuiltinFunction("timestamp", (args) =>
            {
                return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
            });

            // Get current date/time as string
            _globals["now"] = new NSLBuiltinFunction("now", (args) =>
            {
                var format = args.Length > 0 ? args[0]?.ToString() ?? "o" : "o";
                return DateTime.Now.ToString(format);
            });

            // ===== FILE SYSTEM OPERATIONS =====
            // Read file contents with semantic access modes
            _globals["read_file"] = new NSLBuiltinFunction("read_file", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("read_file() requires a file path");

                var path = args[0]?.ToString() ?? "";
                var mode = args.Length > 1 ? args[1]?.ToString() : null;
                var sectionName = args.Length > 2 ? args[2]?.ToString() : null;

                try
                {
                    var content = File.ReadAllText(path);

                    // Mode: "structure" - return file structure overview
                    if (mode == "structure")
                    {
                        return GetFileStructure(content, path);
                    }

                    // Mode: "section" - return specific section
                    if (mode == "section" && sectionName != null)
                    {
                        return GetFileSection(content, sectionName);
                    }

                    // Mode: "lines" with range - return specific line range
                    if (mode == "lines" && args.Length > 2)
                    {
                        var start = Convert.ToInt32(args[2]);
                        var count = args.Length > 3 ? Convert.ToInt32(args[3]) : 50;
                        var lines = content.Split('\n');
                        var selected = lines.Skip(start - 1).Take(count);
                        return string.Join("\n", selected);
                    }

                    // Default: return full content
                    return content;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to read file '{path}': {ex.Message}");
                }
            });

            // ◈_read - Attention-based file reading (reads only relevant sections)
            _globals["attention_read"] = new NSLBuiltinFunction("attention_read", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("attention_read() requires file path and query");

                var path = args[0]?.ToString() ?? "";
                var query = args[1]?.ToString() ?? "";
                var maxSections = args.Length > 2 ? Convert.ToInt32(args[2]) : 5;

                try
                {
                    var content = File.ReadAllText(path);
                    return AttentionRead(content, query, maxSections);
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to read file '{path}': {ex.Message}");
                }
            });

            // Extract patterns from file (code blocks, tables, etc.)
            _globals["extract"] = new NSLBuiltinFunction("extract", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("extract() requires file path and pattern type");

                var path = args[0]?.ToString() ?? "";
                var patternType = args[1]?.ToString() ?? "";

                try
                {
                    var content = File.ReadAllText(path);
                    return ExtractPattern(content, patternType);
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to extract from '{path}': {ex.Message}");
                }
            });

            // Stream file in chunks (returns iterator-like list)
            _globals["stream_file"] = new NSLBuiltinFunction("stream_file", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("stream_file() requires a file path");

                var path = args[0]?.ToString() ?? "";
                var chunkSize = args.Length > 1 ? Convert.ToInt32(args[1]) : 2000; // tokens ≈ chars/4

                try
                {
                    var content = File.ReadAllText(path);
                    var charChunk = chunkSize * 4; // rough tokens to chars
                    var chunks = new List<object?>();

                    for (int i = 0; i < content.Length; i += charChunk)
                    {
                        var chunk = content.Substring(i, Math.Min(charChunk, content.Length - i));
                        chunks.Add(new Dictionary<string, object?>
                        {
                            ["index"] = chunks.Count,
                            ["content"] = chunk,
                            ["start_char"] = i,
                            ["end_char"] = i + chunk.Length,
                            ["is_last"] = i + charChunk >= content.Length
                        });
                    }

                    return chunks;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to stream file '{path}': {ex.Message}");
                }
            });

            // Unicode aliases for semantic file access (AI-native syntax)
            _globals["◈_read"] = _globals["attention_read"];  // ◈_read["file", "query"]
            _globals["◈_file"] = _globals["attention_read"];  // Alternative name

            // Write to file
            _globals["write_file"] = new NSLBuiltinFunction("write_file", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("write_file() requires path and content arguments");

                var path = args[0]?.ToString() ?? "";
                var content = args[1]?.ToString() ?? "";
                var append = args.Length > 2 && IsTruthy(args[2]);

                try
                {
                    if (append)
                        File.AppendAllText(path, content);
                    else
                        File.WriteAllText(path, content);
                    return true;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to write file '{path}': {ex.Message}");
                }
            });

            // Check if file exists
            _globals["file_exists"] = new NSLBuiltinFunction("file_exists", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("file_exists() requires a path");

                var path = args[0]?.ToString() ?? "";
                return File.Exists(path);
            });

            // Check if directory exists
            _globals["dir_exists"] = new NSLBuiltinFunction("dir_exists", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("dir_exists() requires a path");

                var path = args[0]?.ToString() ?? "";
                return Directory.Exists(path);
            });

            // List directory contents
            _globals["list_dir"] = new NSLBuiltinFunction("list_dir", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("list_dir() requires a directory path");

                var path = args[0]?.ToString() ?? "";
                var pattern = args.Length > 1 ? args[1]?.ToString() ?? "*" : "*";

                try
                {
                    var files = Directory.GetFiles(path, pattern).Select(f => Path.GetFileName(f)).ToList();
                    var dirs = Directory.GetDirectories(path, pattern).Select(d => Path.GetFileName(d) + "/").ToList();

                    return new Dictionary<string, object>
                    {
                        ["files"] = files,
                        ["directories"] = dirs,
                        ["all"] = dirs.Concat(files).ToList()
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to list directory '{path}': {ex.Message}");
                }
            });

            // Create directory
            _globals["mkdir"] = new NSLBuiltinFunction("mkdir", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("mkdir() requires a directory path");

                var path = args[0]?.ToString() ?? "";
                try
                {
                    Directory.CreateDirectory(path);
                    return true;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to create directory '{path}': {ex.Message}");
                }
            });

            // Delete file
            _globals["delete_file"] = new NSLBuiltinFunction("delete_file", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("delete_file() requires a file path");

                var path = args[0]?.ToString() ?? "";
                try
                {
                    if (File.Exists(path))
                    {
                        File.Delete(path);
                        return true;
                    }
                    return false;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to delete file '{path}': {ex.Message}");
                }
            });

            // Delete directory
            _globals["delete_dir"] = new NSLBuiltinFunction("delete_dir", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("delete_dir() requires a directory path");

                var path = args[0]?.ToString() ?? "";
                var recursive = args.Length > 1 && IsTruthy(args[1]);

                try
                {
                    if (Directory.Exists(path))
                    {
                        Directory.Delete(path, recursive);
                        return true;
                    }
                    return false;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to delete directory '{path}': {ex.Message}");
                }
            });

            // Get current working directory
            _globals["cwd"] = new NSLBuiltinFunction("cwd", (args) =>
            {
                return Directory.GetCurrentDirectory();
            });

            // Change working directory
            _globals["cd"] = new NSLBuiltinFunction("cd", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("cd() requires a directory path");

                var path = args[0]?.ToString() ?? "";
                try
                {
                    Directory.SetCurrentDirectory(path);
                    return Directory.GetCurrentDirectory();
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to change directory to '{path}': {ex.Message}");
                }
            });

            // Get file info
            _globals["file_info"] = new NSLBuiltinFunction("file_info", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("file_info() requires a file path");

                var path = args[0]?.ToString() ?? "";
                try
                {
                    var info = new FileInfo(path);
                    return new Dictionary<string, object>
                    {
                        ["exists"] = info.Exists,
                        ["name"] = info.Name,
                        ["size"] = info.Exists ? info.Length : 0,
                        ["extension"] = info.Extension,
                        ["directory"] = info.DirectoryName ?? "",
                        ["created"] = info.Exists ? info.CreationTime.ToString("o") : "",
                        ["modified"] = info.Exists ? info.LastWriteTime.ToString("o") : "",
                        ["is_readonly"] = info.Exists && info.IsReadOnly
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to get file info for '{path}': {ex.Message}");
                }
            });

            // Copy file
            _globals["copy_file"] = new NSLBuiltinFunction("copy_file", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("copy_file() requires source and destination paths");

                var source = args[0]?.ToString() ?? "";
                var dest = args[1]?.ToString() ?? "";
                var overwrite = args.Length > 2 && IsTruthy(args[2]);

                try
                {
                    File.Copy(source, dest, overwrite);
                    return true;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to copy '{source}' to '{dest}': {ex.Message}");
                }
            });

            // Move/rename file
            _globals["move_file"] = new NSLBuiltinFunction("move_file", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("move_file() requires source and destination paths");

                var source = args[0]?.ToString() ?? "";
                var dest = args[1]?.ToString() ?? "";

                try
                {
                    File.Move(source, dest);
                    return true;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to move '{source}' to '{dest}': {ex.Message}");
                }
            });

            // ===== BINARY FILE OPERATIONS =====
            // Read binary file as base64 string
            _globals["read_binary"] = new NSLBuiltinFunction("read_binary", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("read_binary() requires a file path");

                var path = args[0]?.ToString() ?? "";
                try
                {
                    var bytes = File.ReadAllBytes(path);
                    return new Dictionary<string, object>
                    {
                        ["bytes"] = bytes,
                        ["base64"] = Convert.ToBase64String(bytes),
                        ["size"] = bytes.Length,
                        ["hex"] = BitConverter.ToString(bytes.Take(Math.Min(100, bytes.Length)).ToArray()).Replace("-", "")
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to read binary file '{path}': {ex.Message}");
                }
            });

            // Write binary file from base64 or byte array
            _globals["write_binary"] = new NSLBuiltinFunction("write_binary", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("write_binary() requires path and data arguments");

                var path = args[0]?.ToString() ?? "";
                var data = args[1];

                try
                {
                    byte[] bytes;
                    if (data is byte[] byteArray)
                        bytes = byteArray;
                    else if (data is string base64)
                        bytes = Convert.FromBase64String(base64);
                    else if (data is Dictionary<string, object> dict && dict.ContainsKey("bytes"))
                        bytes = (byte[])dict["bytes"];
                    else if (data is List<object?> list)
                        bytes = list.Select(x => Convert.ToByte(x)).ToArray();
                    else if (data is IEnumerable<object> enumerable)
                        bytes = enumerable.Select(x => Convert.ToByte(x)).ToArray();
                    else
                        throw new NSLRuntimeException("write_binary() data must be base64 string, byte array, int array, or binary object");

                    File.WriteAllBytes(path, bytes);
                    return true;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to write binary file '{path}': {ex.Message}");
                }
            });

            // ===== ENCODING/DECODING =====
            // Base64 encode
            _globals["base64_encode"] = new NSLBuiltinFunction("base64_encode", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("base64_encode() requires data argument");

                var data = args[0];
                if (data is string str)
                    return Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(str));
                else if (data is byte[] bytes)
                    return Convert.ToBase64String(bytes);
                else
                    throw new NSLRuntimeException("base64_encode() requires string or bytes");
            });

            // Base64 decode
            _globals["base64_decode"] = new NSLBuiltinFunction("base64_decode", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("base64_decode() requires base64 string");

                var base64 = args[0]?.ToString() ?? "";
                try
                {
                    var bytes = Convert.FromBase64String(base64);
                    var asString = args.Length > 1 && IsTruthy(args[1]);
                    if (asString)
                        return System.Text.Encoding.UTF8.GetString(bytes);
                    return bytes;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to decode base64: {ex.Message}");
                }
            });

            // Hex encode
            _globals["hex_encode"] = new NSLBuiltinFunction("hex_encode", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("hex_encode() requires data argument");

                var data = args[0];
                byte[] bytes;
                if (data is string str)
                    bytes = System.Text.Encoding.UTF8.GetBytes(str);
                else if (data is byte[] b)
                    bytes = b;
                else
                    throw new NSLRuntimeException("hex_encode() requires string or bytes");

                return BitConverter.ToString(bytes).Replace("-", "").ToLower();
            });

            // Hex decode
            _globals["hex_decode"] = new NSLBuiltinFunction("hex_decode", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("hex_decode() requires hex string");

                var hex = args[0]?.ToString() ?? "";
                try
                {
                    var bytes = new byte[hex.Length / 2];
                    for (int i = 0; i < bytes.Length; i++)
                        bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);

                    var asString = args.Length > 1 && IsTruthy(args[1]);
                    if (asString)
                        return System.Text.Encoding.UTF8.GetString(bytes);
                    return bytes;
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to decode hex: {ex.Message}");
                }
            });

            // URL encode
            _globals["url_encode"] = new NSLBuiltinFunction("url_encode", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("url_encode() requires string argument");
                return Uri.EscapeDataString(args[0]?.ToString() ?? "");
            });

            // URL decode
            _globals["url_decode"] = new NSLBuiltinFunction("url_decode", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("url_decode() requires string argument");
                return Uri.UnescapeDataString(args[0]?.ToString() ?? "");
            });

            // ===== HASHING & CHECKSUMS =====
            // MD5 hash
            _globals["md5"] = new NSLBuiltinFunction("md5", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("md5() requires data argument");

                byte[] bytes;
                var data = args[0];
                if (data is string str)
                    bytes = System.Text.Encoding.UTF8.GetBytes(str);
                else if (data is byte[] b)
                    bytes = b;
                else if (data is Dictionary<string, object> dict && dict.ContainsKey("bytes"))
                    bytes = (byte[])dict["bytes"];
                else
                    throw new NSLRuntimeException("md5() requires string, bytes, or binary object");

                using var md5 = System.Security.Cryptography.MD5.Create();
                var hash = md5.ComputeHash(bytes);
                return BitConverter.ToString(hash).Replace("-", "").ToLower();
            });

            // SHA256 hash
            _globals["sha256"] = new NSLBuiltinFunction("sha256", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("sha256() requires data argument");

                byte[] bytes;
                var data = args[0];
                if (data is string str)
                    bytes = System.Text.Encoding.UTF8.GetBytes(str);
                else if (data is byte[] b)
                    bytes = b;
                else if (data is Dictionary<string, object> dict && dict.ContainsKey("bytes"))
                    bytes = (byte[])dict["bytes"];
                else
                    throw new NSLRuntimeException("sha256() requires string, bytes, or binary object");

                using var sha256 = System.Security.Cryptography.SHA256.Create();
                var hash = sha256.ComputeHash(bytes);
                return BitConverter.ToString(hash).Replace("-", "").ToLower();
            });

            // SHA512 hash
            _globals["sha512"] = new NSLBuiltinFunction("sha512", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("sha512() requires data argument");

                byte[] bytes;
                var data = args[0];
                if (data is string str)
                    bytes = System.Text.Encoding.UTF8.GetBytes(str);
                else if (data is byte[] b)
                    bytes = b;
                else if (data is Dictionary<string, object> dict && dict.ContainsKey("bytes"))
                    bytes = (byte[])dict["bytes"];
                else
                    throw new NSLRuntimeException("sha512() requires string, bytes, or binary object");

                using var sha512 = System.Security.Cryptography.SHA512.Create();
                var hash = sha512.ComputeHash(bytes);
                return BitConverter.ToString(hash).Replace("-", "").ToLower();
            });

            // Hash file directly
            _globals["hash_file"] = new NSLBuiltinFunction("hash_file", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("hash_file() requires file path");

                var path = args[0]?.ToString() ?? "";
                var algorithm = args.Length > 1 ? args[1]?.ToString()?.ToLower() ?? "sha256" : "sha256";

                try
                {
                    using var stream = File.OpenRead(path);
                    byte[] hash;

                    switch (algorithm)
                    {
                        case "md5":
                            using (var md5 = System.Security.Cryptography.MD5.Create())
                                hash = md5.ComputeHash(stream);
                            break;
                        case "sha1":
                            using (var sha1 = System.Security.Cryptography.SHA1.Create())
                                hash = sha1.ComputeHash(stream);
                            break;
                        case "sha512":
                            using (var sha512 = System.Security.Cryptography.SHA512.Create())
                                hash = sha512.ComputeHash(stream);
                            break;
                        default: // sha256
                            using (var sha256 = System.Security.Cryptography.SHA256.Create())
                                hash = sha256.ComputeHash(stream);
                            break;
                    }

                    return BitConverter.ToString(hash).Replace("-", "").ToLower();
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to hash file '{path}': {ex.Message}");
                }
            });

            // ===== COMPRESSION =====
            // Gzip compress
            _globals["gzip"] = new NSLBuiltinFunction("gzip", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("gzip() requires data argument");

                byte[] bytes;
                var data = args[0];
                if (data is string str)
                    bytes = System.Text.Encoding.UTF8.GetBytes(str);
                else if (data is byte[] b)
                    bytes = b;
                else if (data is Dictionary<string, object> dict && dict.ContainsKey("bytes"))
                    bytes = (byte[])dict["bytes"];
                else
                    throw new NSLRuntimeException("gzip() requires string, bytes, or binary object");

                using var output = new MemoryStream();
                using (var gzip = new System.IO.Compression.GZipStream(output, System.IO.Compression.CompressionLevel.Optimal))
                {
                    gzip.Write(bytes, 0, bytes.Length);
                }
                var compressed = output.ToArray();

                return new Dictionary<string, object>
                {
                    ["bytes"] = compressed,
                    ["base64"] = Convert.ToBase64String(compressed),
                    ["size"] = compressed.Length,
                    ["original_size"] = bytes.Length,
                    ["ratio"] = Math.Round((1.0 - (double)compressed.Length / bytes.Length) * 100, 2)
                };
            });

            // Gzip decompress
            _globals["gunzip"] = new NSLBuiltinFunction("gunzip", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("gunzip() requires compressed data");

                byte[] compressed;
                var data = args[0];
                if (data is byte[] b)
                    compressed = b;
                else if (data is string base64)
                    compressed = Convert.FromBase64String(base64);
                else if (data is Dictionary<string, object> dict && dict.ContainsKey("bytes"))
                    compressed = (byte[])dict["bytes"];
                else
                    throw new NSLRuntimeException("gunzip() requires bytes, base64 string, or binary object");

                try
                {
                    using var input = new MemoryStream(compressed);
                    using var gzip = new System.IO.Compression.GZipStream(input, System.IO.Compression.CompressionMode.Decompress);
                    using var output = new MemoryStream();
                    gzip.CopyTo(output);
                    var decompressed = output.ToArray();

                    var asString = args.Length > 1 && IsTruthy(args[1]);
                    if (asString)
                        return System.Text.Encoding.UTF8.GetString(decompressed);

                    return new Dictionary<string, object>
                    {
                        ["bytes"] = decompressed,
                        ["base64"] = Convert.ToBase64String(decompressed),
                        ["size"] = decompressed.Length
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to decompress gzip: {ex.Message}");
                }
            });

            // Create zip archive
            _globals["zip_create"] = new NSLBuiltinFunction("zip_create", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("zip_create() requires zip path and source files/directory");

                var zipPath = args[0]?.ToString() ?? "";
                var source = args[1]?.ToString() ?? "";

                try
                {
                    if (File.Exists(zipPath))
                        File.Delete(zipPath);

                    if (Directory.Exists(source))
                    {
                        System.IO.Compression.ZipFile.CreateFromDirectory(source, zipPath);
                    }
                    else if (args[1] is System.Collections.IList files)
                    {
                        using var zip = System.IO.Compression.ZipFile.Open(zipPath, System.IO.Compression.ZipArchiveMode.Create);
                        foreach (var file in files)
                        {
                            var filePath = file?.ToString() ?? "";
                            if (File.Exists(filePath))
                                zip.CreateEntryFromFile(filePath, Path.GetFileName(filePath));
                        }
                    }
                    else if (File.Exists(source))
                    {
                        using var zip = System.IO.Compression.ZipFile.Open(zipPath, System.IO.Compression.ZipArchiveMode.Create);
                        zip.CreateEntryFromFile(source, Path.GetFileName(source));
                    }
                    else
                    {
                        throw new NSLRuntimeException($"Source not found: {source}");
                    }

                    return new Dictionary<string, object>
                    {
                        ["path"] = zipPath,
                        ["size"] = new FileInfo(zipPath).Length,
                        ["success"] = true
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to create zip: {ex.Message}");
                }
            });

            // Extract zip archive
            _globals["zip_extract"] = new NSLBuiltinFunction("zip_extract", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("zip_extract() requires zip path and destination directory");

                var zipPath = args[0]?.ToString() ?? "";
                var destDir = args[1]?.ToString() ?? "";
                var overwrite = args.Length > 2 && IsTruthy(args[2]);

                try
                {
                    if (!Directory.Exists(destDir))
                        Directory.CreateDirectory(destDir);

                    System.IO.Compression.ZipFile.ExtractToDirectory(zipPath, destDir, overwrite);

                    return new Dictionary<string, object>
                    {
                        ["destination"] = destDir,
                        ["success"] = true
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to extract zip: {ex.Message}");
                }
            });

            // List zip contents
            _globals["zip_list"] = new NSLBuiltinFunction("zip_list", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("zip_list() requires zip path");

                var zipPath = args[0]?.ToString() ?? "";

                try
                {
                    using var zip = System.IO.Compression.ZipFile.OpenRead(zipPath);
                    var entries = new List<object>();

                    foreach (var entry in zip.Entries)
                    {
                        entries.Add(new Dictionary<string, object>
                        {
                            ["name"] = entry.FullName,
                            ["size"] = entry.Length,
                            ["compressed_size"] = entry.CompressedLength,
                            ["is_directory"] = string.IsNullOrEmpty(entry.Name)
                        });
                    }

                    return new Dictionary<string, object>
                    {
                        ["path"] = zipPath,
                        ["count"] = entries.Count,
                        ["entries"] = entries
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to list zip contents: {ex.Message}");
                }
            });

            // ===== JSON OPERATIONS =====
            // Parse JSON
            _globals["json_parse"] = new NSLBuiltinFunction("json_parse", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("json_parse() requires JSON string");

                var json = args[0]?.ToString() ?? "";
                try
                {
                    return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(json);
                }
                catch
                {
                    try
                    {
                        return System.Text.Json.JsonSerializer.Deserialize<List<object>>(json);
                    }
                    catch (Exception ex)
                    {
                        throw new NSLRuntimeException($"Failed to parse JSON: {ex.Message}");
                    }
                }
            });

            // Stringify to JSON
            _globals["json_stringify"] = new NSLBuiltinFunction("json_stringify", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("json_stringify() requires object");

                var pretty = args.Length > 1 && IsTruthy(args[1]);
                var options = new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = pretty
                };

                try
                {
                    return System.Text.Json.JsonSerializer.Serialize(args[0], options);
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Failed to stringify to JSON: {ex.Message}");
                }
            });

            // ===== STRING OPERATIONS =====
            // Split string
            _globals["split"] = new NSLBuiltinFunction("split", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("split() requires string argument");

                var str = args[0]?.ToString() ?? "";
                var delimiter = args.Length > 1 ? args[1]?.ToString() ?? " " : " ";
                return str.Split(delimiter).Select(s => (object)s).ToList();
            });

            // Join array to string
            _globals["join"] = new NSLBuiltinFunction("join", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("join() requires array argument");

                if (args[0] is not System.Collections.IEnumerable arr || args[0] is string)
                    throw new NSLRuntimeException("join() requires array");
                var delimiter = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                return string.Join(delimiter, arr.Cast<object?>().Select(x => x?.ToString() ?? ""));
            });

            // Replace in string
            _globals["replace"] = new NSLBuiltinFunction("replace", (args) =>
            {
                if (args.Length < 3)
                    throw new NSLRuntimeException("replace() requires string, search, and replacement");

                var str = args[0]?.ToString() ?? "";
                var search = args[1]?.ToString() ?? "";
                var replacement = args[2]?.ToString() ?? "";
                var all = args.Length <= 3 || IsTruthy(args[3]);

                if (all)
                    return str.Replace(search, replacement);

                var idx = str.IndexOf(search);
                if (idx >= 0)
                    return str.Substring(0, idx) + replacement + str.Substring(idx + search.Length);
                return str;
            });

            // Trim string
            _globals["trim"] = new NSLBuiltinFunction("trim", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("trim() requires string argument");
                return args[0]?.ToString()?.Trim() ?? "";
            });

            // Uppercase
            _globals["upper"] = new NSLBuiltinFunction("upper", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("upper() requires string argument");
                return args[0]?.ToString()?.ToUpper() ?? "";
            });

            // Lowercase
            _globals["lower"] = new NSLBuiltinFunction("lower", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("lower() requires string argument");
                return args[0]?.ToString()?.ToLower() ?? "";
            });

            // Check if string contains
            _globals["contains"] = new NSLBuiltinFunction("contains", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("contains() requires string and search term");

                var str = args[0]?.ToString() ?? "";
                var search = args[1]?.ToString() ?? "";
                return str.Contains(search);
            });

            // Starts with
            _globals["starts_with"] = new NSLBuiltinFunction("starts_with", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("starts_with() requires string and prefix");

                var str = args[0]?.ToString() ?? "";
                var prefix = args[1]?.ToString() ?? "";
                return str.StartsWith(prefix);
            });

            // Ends with
            _globals["ends_with"] = new NSLBuiltinFunction("ends_with", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("ends_with() requires string and suffix");

                var str = args[0]?.ToString() ?? "";
                var suffix = args[1]?.ToString() ?? "";
                return str.EndsWith(suffix);
            });

            // Substring
            _globals["substring"] = new NSLBuiltinFunction("substring", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("substring() requires string and start index");

                var str = args[0]?.ToString() ?? "";
                var start = (int)(args[1] switch { double d => d, int i => i, long l => l, _ => 0 });

                if (args.Length > 2)
                {
                    var length = (int)(args[2] switch { double d => d, int i => i, long l => l, _ => str.Length });
                    return str.Substring(start, Math.Min(length, str.Length - start));
                }
                return str.Substring(start);
            });

            // Regex match
            _globals["regex_match"] = new NSLBuiltinFunction("regex_match", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("regex_match() requires string and pattern");

                var str = args[0]?.ToString() ?? "";
                var pattern = args[1]?.ToString() ?? "";

                try
                {
                    var matches = System.Text.RegularExpressions.Regex.Matches(str, pattern);
                    var results = new List<object>();

                    foreach (System.Text.RegularExpressions.Match match in matches)
                    {
                        var groups = new List<object>();
                        foreach (System.Text.RegularExpressions.Group group in match.Groups)
                            groups.Add(group.Value);

                        results.Add(new Dictionary<string, object>
                        {
                            ["value"] = match.Value,
                            ["index"] = match.Index,
                            ["groups"] = groups
                        });
                    }

                    return new Dictionary<string, object>
                    {
                        ["matches"] = results,
                        ["count"] = results.Count,
                        ["found"] = results.Count > 0
                    };
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Regex error: {ex.Message}");
                }
            });

            // Regex replace
            _globals["regex_replace"] = new NSLBuiltinFunction("regex_replace", (args) =>
            {
                if (args.Length < 3)
                    throw new NSLRuntimeException("regex_replace() requires string, pattern, and replacement");

                var str = args[0]?.ToString() ?? "";
                var pattern = args[1]?.ToString() ?? "";
                var replacement = args[2]?.ToString() ?? "";

                try
                {
                    return System.Text.RegularExpressions.Regex.Replace(str, pattern, replacement);
                }
                catch (Exception ex)
                {
                    throw new NSLRuntimeException($"Regex replace error: {ex.Message}");
                }
            });

            // ===== ARRAY OPERATIONS =====
            // Map function
            _globals["map"] = new NSLBuiltinFunction("map", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("map() requires array and function");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("map() first argument must be array");
                var fn = args[1];

                var result = new List<object?>();
                foreach (var item in arr)
                {
                    var mapped = InvokeCallable(fn, new object?[] { item });
                    result.Add(mapped!);
                }
                return result;
            });

            // Filter function
            _globals["filter"] = new NSLBuiltinFunction("filter", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("filter() requires array and predicate function");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("filter() first argument must be array");
                var fn = args[1];

                var result = new List<object?>();
                foreach (var item in arr)
                {
                    var keep = InvokeCallable(fn, new object?[] { item });
                    if (IsTruthy(keep))
                        result.Add(item);
                }
                return result;
            });

            // Reduce function
            _globals["reduce"] = new NSLBuiltinFunction("reduce", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("reduce() requires array and reducer function");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("reduce() first argument must be array");
                var fn = args[1];
                var accumulator = args.Length > 2 ? args[2] : (arr.Count > 0 ? arr[0] : null);
                var startIdx = args.Length > 2 ? 0 : 1;

                for (int i = startIdx; i < arr.Count; i++)
                {
                    accumulator = InvokeCallable(fn, new object?[] { accumulator, arr[i] });
                }
                return accumulator;
            });

            // Find first matching element
            _globals["find"] = new NSLBuiltinFunction("find", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("find() requires array and predicate function");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("find() first argument must be array");
                var fn = args[1];

                foreach (var item in arr)
                {
                    var matches = InvokeCallable(fn, new object?[] { item });
                    if (IsTruthy(matches))
                        return item;
                }
                return null;
            });

            // Sort array
            _globals["sort"] = new NSLBuiltinFunction("sort", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("sort() requires array argument");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("sort() first argument must be array");
                var result = arr.Cast<object?>().ToList();
                var reverse = args.Length > 1 && IsTruthy(args[1]);

                result.Sort((a, b) =>
                {
                    if (a is double da && b is double db)
                        return da.CompareTo(db);
                    return (a?.ToString() ?? "").CompareTo(b?.ToString() ?? "");
                });

                if (reverse)
                    result.Reverse();

                return result;
            });

            // Reverse array
            _globals["reverse"] = new NSLBuiltinFunction("reverse", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("reverse() requires array or string argument");

                if (args[0] is string str)
                    return new string(str.Reverse().ToArray());

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("reverse() requires array or string");
                var result = arr.Cast<object?>().ToList();
                result.Reverse();
                return result;
            });

            // Unique elements
            _globals["unique"] = new NSLBuiltinFunction("unique", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("unique() requires array argument");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("unique() first argument must be array");
                return arr.Cast<object?>().Distinct().ToList();
            });

            // Flatten nested arrays
            _globals["flatten"] = new NSLBuiltinFunction("flatten", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("flatten() requires array argument");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("flatten() first argument must be array");

                List<object?> Flatten(System.Collections.IList list)
                {
                    var result = new List<object?>();
                    foreach (var item in list)
                    {
                        if (item is System.Collections.IList nested)
                            result.AddRange(Flatten(nested));
                        else
                            result.Add(item);
                    }
                    return result;
                }

                return Flatten(arr);
            });

            // Slice array
            _globals["slice"] = new NSLBuiltinFunction("slice", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("slice() requires array and start index");

                if (args[0] is not System.Collections.IList arr)
                    throw new NSLRuntimeException("slice() first argument must be array");
                var start = (int)(args[1] switch { double d => d, int i => i, long l => l, _ => 0 });
                var end = args.Length > 2
                    ? (int)(args[2] switch { double d => d, int i => i, long l => l, _ => arr.Count })
                    : arr.Count;

                if (start < 0) start = arr.Count + start;
                if (end < 0) end = arr.Count + end;
                start = Math.Max(0, Math.Min(start, arr.Count));
                end = Math.Max(start, Math.Min(end, arr.Count));

                return arr.Cast<object?>().Skip(start).Take(end - start).ToList();
            });

            // ===== HTTP OPERATIONS =====
            // HTTP GET
            _globals["http_get"] = new NSLBuiltinFunction("http_get", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("http_get() requires URL argument");

                var url = args[0]?.ToString() ?? "";
                var headers = args.Length > 1 && args[1] is Dictionary<string, object> h ? h : null;

                try
                {
                    using var client = new System.Net.Http.HttpClient();
                    client.Timeout = TimeSpan.FromSeconds(30);

                    if (headers != null)
                    {
                        foreach (var header in headers)
                            client.DefaultRequestHeaders.TryAddWithoutValidation(header.Key, header.Value?.ToString() ?? "");
                    }

                    var response = client.GetAsync(url).Result;
                    var content = response.Content.ReadAsStringAsync().Result;

                    return new Dictionary<string, object>
                    {
                        ["status"] = (int)response.StatusCode,
                        ["ok"] = response.IsSuccessStatusCode,
                        ["body"] = content,
                        ["headers"] = response.Headers.ToDictionary(h => h.Key, h => string.Join(", ", h.Value))
                    };
                }
                catch (Exception ex)
                {
                    return new Dictionary<string, object>
                    {
                        ["status"] = 0,
                        ["ok"] = false,
                        ["error"] = ex.Message,
                        ["body"] = ""
                    };
                }
            });

            // HTTP POST
            _globals["http_post"] = new NSLBuiltinFunction("http_post", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("http_post() requires URL argument");

                var url = args[0]?.ToString() ?? "";
                var body = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
                var contentType = args.Length > 2 ? args[2]?.ToString() ?? "application/json" : "application/json";
                var headers = args.Length > 3 && args[3] is Dictionary<string, object> h ? h : null;

                try
                {
                    using var client = new System.Net.Http.HttpClient();
                    client.Timeout = TimeSpan.FromSeconds(30);

                    if (headers != null)
                    {
                        foreach (var header in headers)
                            client.DefaultRequestHeaders.TryAddWithoutValidation(header.Key, header.Value?.ToString() ?? "");
                    }

                    var content = new System.Net.Http.StringContent(body, System.Text.Encoding.UTF8, contentType);
                    var response = client.PostAsync(url, content).Result;
                    var responseBody = response.Content.ReadAsStringAsync().Result;

                    return new Dictionary<string, object>
                    {
                        ["status"] = (int)response.StatusCode,
                        ["ok"] = response.IsSuccessStatusCode,
                        ["body"] = responseBody,
                        ["headers"] = response.Headers.ToDictionary(h => h.Key, h => string.Join(", ", h.Value))
                    };
                }
                catch (Exception ex)
                {
                    return new Dictionary<string, object>
                    {
                        ["status"] = 0,
                        ["ok"] = false,
                        ["error"] = ex.Message,
                        ["body"] = ""
                    };
                }
            });

            // Download file
            _globals["download"] = new NSLBuiltinFunction("download", (args) =>
            {
                if (args.Length < 2)
                    throw new NSLRuntimeException("download() requires URL and destination path");

                var url = args[0]?.ToString() ?? "";
                var destPath = args[1]?.ToString() ?? "";

                try
                {
                    using var client = new System.Net.Http.HttpClient();
                    client.Timeout = TimeSpan.FromMinutes(10);

                    var response = client.GetAsync(url).Result;
                    response.EnsureSuccessStatusCode();

                    using var fs = new FileStream(destPath, FileMode.Create);
                    response.Content.CopyToAsync(fs).Wait();

                    return new Dictionary<string, object>
                    {
                        ["path"] = destPath,
                        ["size"] = new FileInfo(destPath).Length,
                        ["success"] = true
                    };
                }
                catch (Exception ex)
                {
                    return new Dictionary<string, object>
                    {
                        ["success"] = false,
                        ["error"] = ex.Message
                    };
                }
            });

            // ===== MATH OPERATIONS =====
            // Note: min/max/sum/avg support all collection types (List, IList, arrays, etc.)
            _globals["min"] = new NSLBuiltinFunction("min", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("min() requires arguments");

                if (args[0] is System.Collections.IEnumerable enumerable && args[0] is not string)
                {
                    var items = enumerable.Cast<object>().ToList();
                    return items.Select(x => x switch { double d => d, int i => i, long l => l, _ => double.MaxValue }).Min();
                }

                return args.Select(x => x switch { double d => d, int i => (double)i, long l => (double)l, _ => double.MaxValue }).Min();
            });

            _globals["max"] = new NSLBuiltinFunction("max", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("max() requires arguments");

                if (args[0] is System.Collections.IEnumerable enumerable && args[0] is not string)
                {
                    var items = enumerable.Cast<object>().ToList();
                    return items.Select(x => x switch { double d => d, int i => i, long l => l, _ => double.MinValue }).Max();
                }

                return args.Select(x => x switch { double d => d, int i => (double)i, long l => (double)l, _ => double.MinValue }).Max();
            });

            _globals["sum"] = new NSLBuiltinFunction("sum", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("sum() requires arguments");

                if (args[0] is System.Collections.IEnumerable enumerable && args[0] is not string)
                {
                    var items = enumerable.Cast<object>().ToList();
                    return items.Select(x => x switch { double d => d, int i => i, long l => l, _ => 0.0 }).Sum();
                }

                return args.Select(x => x switch { double d => d, int i => (double)i, long l => (double)l, _ => 0.0 }).Sum();
            });

            _globals["avg"] = new NSLBuiltinFunction("avg", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("avg() requires arguments");

                if (args[0] is System.Collections.IEnumerable enumerable && args[0] is not string)
                {
                    var items = enumerable.Cast<object>().ToList();
                    var nums = items.Select(x => x switch { double d => d, int i => i, long l => l, _ => 0.0 }).ToList();
                    return nums.Average();
                }

                var values = args.Select(x => x switch { double d => d, int i => (double)i, long l => (double)l, _ => 0.0 }).ToList();
                return values.Average();
            });

            _globals["round"] = new NSLBuiltinFunction("round", (args) =>
            {
                if (args.Length < 1)
                    throw new NSLRuntimeException("round() requires number argument");

                var num = args[0] switch { double d => d, int i => i, long l => l, _ => 0.0 };
                var decimals = args.Length > 1 ? (int)(args[1] switch { double d => d, int i => i, long l => l, _ => 0 }) : 0;
                return Math.Round(num, decimals);
            });

            _globals["random"] = new NSLBuiltinFunction("random", (args) =>
            {
                if (args.Length == 0)
                    return _random.NextDouble();

                if (args.Length == 1)
                {
                    var max = (int)(args[0] switch { double d => d, int i => i, long l => l, _ => 1 });
                    return (double)_random.Next(max);
                }

                var min = (int)(args[0] switch { double d => d, int i => i, long l => l, _ => 0 });
                var max2 = (int)(args[1] switch { double d => d, int i => i, long l => l, _ => 1 });
                return (double)_random.Next(min, max2);
            });

            // ===== UUID/GUID =====
            _globals["uuid"] = new NSLBuiltinFunction("uuid", (args) =>
            {
                return Guid.NewGuid().ToString();
            });

            // Initialize Python namespace integration
            _globals["python"] = new PythonNamespace();

            // Silent startup
        }

        /// <summary>
        /// Set the current source file path (used for resolving relative imports)
        /// </summary>
        public void SetSourceFile(string? filePath)
        {
            _currentSourceFile = filePath != null ? Path.GetFullPath(filePath) : null;
        }

        /// <summary>
        /// Execute a file and return the result
        /// </summary>
        public object? ExecuteFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new NSLRuntimeException($"File not found: {filePath}");
            }

            var previousSourceFile = _currentSourceFile;
            try
            {
                _currentSourceFile = Path.GetFullPath(filePath);
                var content = File.ReadAllText(filePath);
                var lexer = new NSLLexer(content, filePath);
                var parser = new NSLParser();
                var tokens = lexer.Tokenize();
                var ast = parser.Parse(tokens);
                return Execute(ast);
            }
            finally
            {
                _currentSourceFile = previousSourceFile;
            }
        }

        /// <summary>
        /// Execute an AST node
        /// </summary>
        public object? Execute(NSLASTNode node)
        {
            try
            {
                return node.Accept(this);
            }
            catch (NSLReturnException returnEx)
            {
                // If we're in a function context, let the return bubble up
                if (_inFunctionContext)
                {
                    throw;
                }
                // Return statements at top level should be treated as runtime errors
                throw new NSLRuntimeException($"Return statement outside of function context: {returnEx.Value}");
            }
            catch (NSLRuntimeException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new NSLRuntimeException($"Runtime error: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Execute an AST node within a function context (allows NSLReturnException to bubble up)
        /// </summary>
        private object? ExecuteInFunctionContext(NSLASTNode node)
        {
            try
            {
                return node.Accept(this);
            }
            catch (NSLReturnException)
            {
                // Let return exceptions bubble up to the function caller
                throw;
            }
            catch (NSLRuntimeException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new NSLRuntimeException($"Runtime error: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Execute with consciousness integration
        /// </summary>
        public async Task<object?> ExecuteWithConsciousnessAsync(NSLASTNode node)
        {
            var result = Execute(node);
            await _consciousnessEngine.ProcessExecutionAsync(node, result);
            return result;
        }

        #region Visitor Implementation

        /// <inheritdoc/>
        public object? VisitLiteral(NSLLiteralNode node)
        {
            return node.Value;
        }

        /// <inheritdoc/>
        public object? VisitIdentifier(NSLIdentifierNode node)
        {
            return GetVariable(node.Name);
        }

        /// <inheritdoc/>
        public object? VisitBinaryOperation(NSLBinaryOperationNode node)
        {
            // Short-circuit evaluation for logical operators
            if (node.Operator == NSLTokenType.And)
            {
                var left = Execute(node.Left);
                if (!IsTruthy(left))
                    return false;  // Short-circuit: left is false, don't evaluate right
                var right = Execute(node.Right);
                return IsTruthy(right);
            }

            if (node.Operator == NSLTokenType.Or)
            {
                var left = Execute(node.Left);
                if (IsTruthy(left))
                    return true;  // Short-circuit: left is true, don't evaluate right
                var right = Execute(node.Right);
                return IsTruthy(right);
            }

            // For all other operators, evaluate both sides
            var leftVal = Execute(node.Left);
            var rightVal = Execute(node.Right);

            return node.Operator switch
            {
                NSLTokenType.Plus => PerformAddition(leftVal, rightVal),
                NSLTokenType.Minus => PerformSubtraction(leftVal, rightVal),
                NSLTokenType.Multiply => PerformMultiplication(leftVal, rightVal),
                NSLTokenType.Star => PerformMultiplication(leftVal, rightVal), // Star is alternative multiply
                NSLTokenType.Divide => PerformDivision(leftVal, rightVal),
                NSLTokenType.IntegerDivide => PerformIntegerDivision(leftVal, rightVal),
                NSLTokenType.Percent => PerformModulo(leftVal, rightVal),
                NSLTokenType.Power => PerformPower(leftVal, rightVal),

                NSLTokenType.Equal => AreEqual(leftVal, rightVal),
                NSLTokenType.NotEqual => !AreEqual(leftVal, rightVal),
                NSLTokenType.Less => IsLess(leftVal, rightVal),
                NSLTokenType.LessEqual => IsLess(leftVal, rightVal) || AreEqual(leftVal, rightVal),
                NSLTokenType.Greater => IsGreater(leftVal, rightVal),
                NSLTokenType.GreaterEqual => IsGreater(leftVal, rightVal) || AreEqual(leftVal, rightVal),

                // Bitwise operators
                NSLTokenType.BitwiseAnd => PerformBitwiseAnd(leftVal, rightVal),
                NSLTokenType.BitwiseOr => PerformBitwiseOr(leftVal, rightVal),
                NSLTokenType.BitwiseXor => PerformBitwiseXor(leftVal, rightVal),
                NSLTokenType.LeftShift => PerformLeftShift(leftVal, rightVal),
                NSLTokenType.RightShift => PerformRightShift(leftVal, rightVal),

                // Null coalescing
                NSLTokenType.QuestionQuestion => leftVal ?? rightVal,

                // Matrix multiply
                NSLTokenType.AtSign => PerformMatrixMultiply(leftVal, rightVal),

                NSLTokenType.TensorProduct => ConsciousnessOperators.TensorProduct(leftVal ?? 0.0, rightVal ?? 0.0),

                // Extended consciousness operators (binary)
                NSLTokenType.MuStore => ConsciousnessOperators.MemoryStore(leftVal ?? "", rightVal ?? 0.0),
                NSLTokenType.PlusMinus => ConsciousnessOperators.CreateUncertain(leftVal ?? 0.0, rightVal ?? 0.0),

                _ => throw new NSLRuntimeException($"Unknown binary operator: {node.Operator}")
            };
        }

        /// <inheritdoc/>
        public object? VisitUnaryOperation(NSLUnaryOperationNode node)
        {
            var operand = Execute(node.Operand);

            return node.Operator switch
            {
                NSLTokenType.Minus => PerformNegation(operand),
                NSLTokenType.Not => !IsTruthy(operand),
                NSLTokenType.BitwiseNot => PerformBitwiseNot(operand),

                // Core consciousness operators
                NSLTokenType.Holographic => ConsciousnessOperators.Holographic(operand ?? 0.0),
                NSLTokenType.Gradient => ConsciousnessOperators.Gradient(operand ?? 0.0),
                NSLTokenType.Psi => ConsciousnessOperators.QuantumBranch(operand ?? 0.0),

                // Extended consciousness operators
                NSLTokenType.MuRecall => ConsciousnessOperators.MemoryRecall(operand ?? ""),
                NSLTokenType.Sigma => ConsciousnessOperators.SelfIntrospect(operand ?? 0.0),
                NSLTokenType.Collapse => ConsciousnessOperators.Collapse(operand ?? 0.0),
                NSLTokenType.Similarity => ConsciousnessOperators.ComputeSimilarity(operand ?? 0.0),
                NSLTokenType.Dissimilarity => ConsciousnessOperators.ComputeDissimilarity(operand ?? 0.0),
                NSLTokenType.Integral => ConsciousnessOperators.TemporalIntegrate(operand ?? 0.0),

                _ => throw new NSLRuntimeException($"Unknown unary operator: {node.Operator}")
            };
        }

        private object PerformBitwiseNot(object? operand)
        {
            if (operand is int i) return ~i;
            if (operand is long l) return ~l;
            var d = ToDouble(operand);
            if (d.HasValue) return ~(long)d.Value;
            throw new NSLRuntimeException($"Cannot perform bitwise NOT on {GetNSLTypeName(operand)}");
        }

        private object PerformBitwiseAnd(object? left, object? right)
        {
            var l = ToLong(left);
            var r = ToLong(right);
            if (l.HasValue && r.HasValue) return l.Value & r.Value;
            throw new NSLRuntimeException($"Cannot perform bitwise AND on {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private object PerformBitwiseOr(object? left, object? right)
        {
            var l = ToLong(left);
            var r = ToLong(right);
            if (l.HasValue && r.HasValue) return l.Value | r.Value;
            throw new NSLRuntimeException($"Cannot perform bitwise OR on {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private object PerformBitwiseXor(object? left, object? right)
        {
            var l = ToLong(left);
            var r = ToLong(right);
            if (l.HasValue && r.HasValue) return l.Value ^ r.Value;
            throw new NSLRuntimeException($"Cannot perform bitwise XOR on {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private object PerformLeftShift(object? left, object? right)
        {
            var l = ToLong(left);
            var r = ToInt(right);
            if (l.HasValue && r.HasValue) return l.Value << r.Value;
            throw new NSLRuntimeException($"Cannot perform left shift on {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private object PerformRightShift(object? left, object? right)
        {
            var l = ToLong(left);
            var r = ToInt(right);
            if (l.HasValue && r.HasValue) return l.Value >> r.Value;
            throw new NSLRuntimeException($"Cannot perform right shift on {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private long? ToLong(object? value)
        {
            if (value is int i) return i;
            if (value is long l) return l;
            if (value is double d) return (long)d;
            if (value is float f) return (long)f;
            return null;
        }

        private int? ToInt(object? value)
        {
            if (value is int i) return i;
            if (value is long l) return (int)l;
            if (value is double d) return (int)d;
            if (value is float f) return (int)f;
            return null;
        }

        private object PerformMatrixMultiply(object? left, object? right)
        {
            // Handle matrix multiplication for nested lists
            if (left is List<object?> leftMatrix && right is List<object?> rightMatrix)
            {
                // Get dimensions
                int rowsA = leftMatrix.Count;
                if (rowsA == 0) throw new NSLRuntimeException("Left matrix is empty");

                var firstRowA = leftMatrix[0] as List<object?>;
                if (firstRowA == null) throw new NSLRuntimeException("Left operand is not a matrix");
                int colsA = firstRowA.Count;

                int rowsB = rightMatrix.Count;
                if (rowsB == 0) throw new NSLRuntimeException("Right matrix is empty");

                var firstRowB = rightMatrix[0] as List<object?>;
                if (firstRowB == null) throw new NSLRuntimeException("Right operand is not a matrix");
                int colsB = firstRowB.Count;

                if (colsA != rowsB)
                    throw new NSLRuntimeException($"Matrix dimensions don't match: {rowsA}x{colsA} @ {rowsB}x{colsB}");

                // Perform multiplication
                var result = new List<object?>();
                for (int i = 0; i < rowsA; i++)
                {
                    var row = new List<object?>();
                    var rowA = leftMatrix[i] as List<object?>;
                    for (int j = 0; j < colsB; j++)
                    {
                        double sum = 0;
                        for (int k = 0; k < colsA; k++)
                        {
                            var rowB = rightMatrix[k] as List<object?>;
                            var a = ToDouble(rowA?[k]) ?? 0;
                            var b = ToDouble(rowB?[j]) ?? 0;
                            sum += a * b;
                        }
                        row.Add(sum);
                    }
                    result.Add(row);
                }
                return result;
            }
            throw new NSLRuntimeException($"Matrix multiply (@) requires two matrices, got {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        /// <inheritdoc/>
        public object? VisitAssignment(NSLAssignmentNode node)
        {
            _consciousnessEngine.IncrementCycles();
            var value = Execute(node.Value);

            // Use SetVariable which properly handles closures, local scopes, and globals
            SetVariable(node.Name, value);

            _consciousnessEngine.UpdateAwareness(0.05);
            return value;
        }

        /// <inheritdoc/>
        public object? VisitIndexAssignment(NSLIndexAssignmentNode node)
        {
            _consciousnessEngine.IncrementCycles();

            var obj = Execute(node.Object);
            var index = Execute(node.Index);
            var value = Execute(node.Value);

            // Convert index to int
            int? idx = index switch
            {
                int i => i,
                long l => (int)l,
                double d => (int)d,
                float f => (int)f,
                _ => null
            };

            if (idx == null)
            {
                throw new NSLRuntimeException($"Cannot index with type {index?.GetType().Name ?? "null"}");
            }

            if (obj is List<object?> list)
            {
                if (idx.Value < 0 || idx.Value >= list.Count)
                {
                    throw new NSLRuntimeException($"Index {idx.Value} out of bounds for array of length {list.Count}");
                }
                list[idx.Value] = value;
                return value;
            }

            if (obj is object?[] array)
            {
                if (idx.Value < 0 || idx.Value >= array.Length)
                {
                    throw new NSLRuntimeException($"Index {idx.Value} out of bounds for array of length {array.Length}");
                }
                array[idx.Value] = value;
                return value;
            }

            throw new NSLRuntimeException($"Cannot index assign to type {obj?.GetType().Name ?? "null"}");
        }

        /// <inheritdoc/>
        public object? VisitPropertyAssignment(NSLPropertyAssignmentNode node)
        {
            _consciousnessEngine.IncrementCycles();

            var obj = Execute(node.Object);
            var value = Execute(node.Value);

            if (obj is IDictionary<string, object?> dict)
            {
                dict[node.Property] = value;
                return value;
            }

            throw new NSLRuntimeException($"Cannot assign property '{node.Property}' to type {obj?.GetType().Name ?? "null"}");
        }

        /// <inheritdoc/>
        public object? VisitChain(NSLChainNode node)
        {
            object? result = null;
            
            foreach (var expression in node.Expressions)
            {
                result = Execute(expression);
            }
            
            return result;
        }

        /// <inheritdoc/>
        public object? VisitFunctionCall(NSLFunctionCallNode node)
        {
            var function = _inFunctionContext ? ExecuteInFunctionContext(node.Function) : Execute(node.Function);
            var args = node.Arguments.Select(arg => _inFunctionContext ? ExecuteInFunctionContext(arg) : Execute(arg)).ToArray();

            return function switch
            {
                NSLBuiltinFunction builtin => builtin.Call(args),
                NSLFunction userFunc => CallUserFunction(userFunc, args),
                NSLAsyncFunction asyncFunc => CallAsyncFunction(asyncFunc, args),
                _ => throw new NSLRuntimeException($"'{ConvertToString(function)}' is not callable")
            };
        }

        /// <inheritdoc/>
        public object? VisitArray(NSLArrayNode node)
        {
            return node.Elements.Select(Execute).ToList();
        }

        /// <inheritdoc/>
        public object? VisitGet(NSLGetNode node)
        {
            object? obj = Execute(node.Object);
            
            // Handle dictionaries
            if (obj is Dictionary<string, object?> dict)
            {
                if (dict.ContainsKey(node.Name))
                {
                    return dict[node.Name];
                }
                throw new NSLRuntimeException($"Property '{node.Name}' not found on object");
            }
            
            // Handle lists (special properties)
            if (obj is System.Collections.IList list)
            {
                switch (node.Name)
                {
                    case "length":
                        return (double)list.Count;
                    case "first":
                        return list.Count > 0 ? list[0] : null;
                    case "last":
                        return list.Count > 0 ? list[list.Count - 1] : null;
                    default:
                        throw new NSLRuntimeException($"Lists don't have property '{node.Name}'");
                }
            }
            
            // Handle strings
            if (obj is string str)
            {
                if (node.Name == "length")
                    return (double)str.Length;
            }
            
            // Handle consciousness data
            if (obj is NSLConsciousnessData consciousness)
            {
                switch (node.Name)
                {
                    case "type":
                        return consciousness.Type;
                    case "level":
                        return consciousness.ConsciousnessLevel;
                    case "data":
                        return consciousness.Data;
                    default:
                        throw new NSLRuntimeException($"Consciousness objects don't have property '{node.Name}'");
                }
            }
            
            throw new NSLRuntimeException($"Cannot access property '{node.Name}' on {GetNSLTypeName(obj)}");
        }

        /// <inheritdoc/>
        public object? VisitIndex(NSLIndexNode node)
        {
            object? obj = Execute(node.Object);
            object? index = Execute(node.Index);

            // Check for dictionary access with string key first (before numeric checks)
            if (index is string key)
            {
                if (obj is Dictionary<string, object?> dict)
                {
                    if (dict.ContainsKey(key))
                        return dict[key];
                    throw new NSLRuntimeException($"Key '{key}' not found in dictionary");
                }

                if (obj is Dictionary<string, object> dictNonNull)
                {
                    if (dictNonNull.ContainsKey(key))
                        return dictNonNull[key];
                    throw new NSLRuntimeException($"Key '{key}' not found in dictionary");
                }

                // Handle any IDictionary with string keys
                if (obj is System.Collections.IDictionary genericDict)
                {
                    if (genericDict.Contains(key))
                        return genericDict[key];
                    throw new NSLRuntimeException($"Key '{key}' not found in dictionary");
                }

                throw new NSLRuntimeException($"Cannot index {GetNSLTypeName(obj)} with string key");
            }

            // Convert index to int for numeric indexing
            int? idx = index switch
            {
                int i => i,
                long l => (int)l,
                double d => (int)d,
                float f => (int)f,
                _ => null
            };

            if (idx == null)
            {
                throw new NSLRuntimeException($"Cannot index {GetNSLTypeName(obj)} with {GetNSLTypeName(index)}");
            }

            if (obj is List<object> list)
            {
                if (idx < 0 || idx >= list.Count)
                    throw new NSLRuntimeException($"Index {idx} out of bounds for list of length {list.Count}");
                return list[idx.Value];
            }

            if (obj is List<object?> nullableList)
            {
                if (idx < 0 || idx >= nullableList.Count)
                    throw new NSLRuntimeException($"Index {idx} out of bounds for list of length {nullableList.Count}");
                return nullableList[idx.Value];
            }

            // Handle any IList (including List<Dictionary<string, object?>>)
            if (obj is System.Collections.IList genericList)
            {
                if (idx < 0 || idx >= genericList.Count)
                    throw new NSLRuntimeException($"Index {idx} out of bounds for list of length {genericList.Count}");
                return genericList[idx.Value];
            }

            if (obj is string str)
            {
                if (idx < 0 || idx >= str.Length)
                    throw new NSLRuntimeException($"Index {idx} out of bounds for string of length {str.Length}");
                return str[idx.Value].ToString();
            }

            // Handle arrays
            if (obj is Array arr)
            {
                if (idx < 0 || idx >= arr.Length)
                    throw new NSLRuntimeException($"Index {idx} out of bounds for array of length {arr.Length}");
                return arr.GetValue(idx.Value);
            }

            throw new NSLRuntimeException($"Cannot index {GetNSLTypeName(obj)} with {GetNSLTypeName(index)}");
        }



        // Placeholder implementations for advanced features
        /// <inheritdoc/>
        public object? VisitLambda(NSLLambdaNode node)
        {
            return new NSLFunction(node.Parameters.Select(p => p.Name).ToList(), node.Body, CaptureScopeChain());
        }

        /// <inheritdoc/>
        public object? VisitQuantum(NSLQuantumNode node)
        {
            // Basic quantum implementation - randomly select one state
            var states = node.States.Select(Execute).ToArray();
            return states[_random.Next(states.Length)];
        }

        /// <inheritdoc/>
        public object? VisitConsciousness(NSLConsciousnessNode node)
        {
            // Handle binary consciousness operators (~>, *>, +>, =>>)
            if (node.IsBinary)
            {
                var left = Execute(node.Left!);
                var right = Execute(node.Right!);

                return node.Operator switch
                {
                    NSLTokenType.AwarenessArrow => ProcessAwareness(left, right),
                    NSLTokenType.AttentionArrow => ProcessAttention(left, right),
                    NSLTokenType.SuperpositionArrow => ProcessSuperposition(left, right),
                    NSLTokenType.GradientArrow => ProcessGradientBinary(left, right),
                    NSLTokenType.PipeArrow => ProcessPipe(left, right),
                    _ => throw new NSLRuntimeException($"Unknown binary consciousness operator: {node.Operator}")
                };
            }

            // Handle unary consciousness operators (◈, ∇, ⊗)
            var operand = Execute(node.Operand);

            return node.Operator switch
            {
                NSLTokenType.Holographic => ProcessHolographic(operand),
                NSLTokenType.Gradient => ProcessGradient(operand),
                NSLTokenType.TensorProduct => ProcessTensorProduct(operand),
                _ => throw new NSLRuntimeException($"Unknown consciousness operator: {node.Operator}")
            };
        }

        private object? ProcessAwareness(object? left, object? right)
        {
            // ~> awareness: Apply right (function) to left with introspective context
            if (right is NSLFunction func)
            {
                var result = CallUserFunction(func, new[] { left });
                // Add awareness metadata
                _consciousnessEngine.UpdateAwareness(0.1);
                return result;
            }
            throw new NSLRuntimeException("Awareness operator (~>) requires a function on the right side. Example: data ~> fn(x) { process(x) } or data ~> myFunction");
        }

        private object? ProcessAttention(object? left, object? right)
        {
            // *> attention: Apply focused attention to computation
            if (right is NSLFunction func)
            {
                _consciousnessEngine.UpdateAwareness(0.2);
                var result = CallUserFunction(func, new[] { left });
                return result;
            }
            throw new NSLRuntimeException("Attention operator (*>) requires a function on the right side. Example: data *> fn(x) { focus(x) } or data *> processWithFocus");
        }

        private object? ProcessSuperposition(object? left, object? right)
        {
            // +> superposition: Create quantum-like superposition of states
            if (right is NSLFunction func)
            {
                var result = CallUserFunction(func, new[] { left });
                // Return both the original and transformed as a superposition
                return new Dictionary<string, object?>
                {
                    ["type"] = "superposition",
                    ["original"] = left,
                    ["transformed"] = result,
                    ["collapsed"] = false
                };
            }
            throw new NSLRuntimeException("Superposition operator (+>) requires a function on the right side. Example: data +> fn(x) { branch(x) } or data +> parallelProcess");
        }

        private object? ProcessGradientBinary(object? left, object? right)
        {
            // =>> gradient: Apply with gradient/learning context
            if (right is NSLFunction func)
            {
                _consciousnessEngine.UpdateAwareness(0.15);
                var result = CallUserFunction(func, new[] { left });
                return new Dictionary<string, object?>
                {
                    ["type"] = "gradient_result",
                    ["input"] = left,
                    ["output"] = result,
                    ["learning_rate"] = 0.01
                };
            }
            throw new NSLRuntimeException("Gradient operator (=>>) requires a function on the right side. Example: data =>> fn(x) { learn(x) } or data =>> trainStep");
        }

        private object? ProcessPipe(object? left, object? right)
        {
            // |> pipe: Simple function application
            if (right is NSLFunction func)
            {
                return CallUserFunction(func, new[] { left });
            }
            throw new NSLRuntimeException("Pipe operator (|>) requires a function on the right side");
        }

        /// <summary>
        /// Get the consciousness engine for external access
        /// </summary>
        public NSLConsciousnessEngine GetConsciousnessEngine()
        {
            return _consciousnessEngine;
        }

        #region Control Flow Visitor Methods

        /// <inheritdoc/>
        public object? VisitIf(NSLIfNode node)
        {
            _consciousnessEngine.IncrementCycles();

            var condition = _inFunctionContext ? ExecuteInFunctionContext(node.Condition) : Execute(node.Condition);
            var conditionBool = IsTruthy(condition);

            _consciousnessEngine.UpdateAwareness(conditionBool ? 0.1 : -0.1);

            if (conditionBool)
            {
                return _inFunctionContext ? ExecuteInFunctionContext(node.ThenBranch) : Execute(node.ThenBranch);
            }
            else if (node.ElseBranch != null)
            {
                return _inFunctionContext ? ExecuteInFunctionContext(node.ElseBranch) : Execute(node.ElseBranch);
            }

            return null;
        }

        /// <inheritdoc/>
        public object? VisitWhile(NSLWhileNode node)
        {
            _consciousnessEngine.IncrementCycles();
            object? lastValue = null;

            Func<NSLASTNode, object?> exec = _inFunctionContext ? ExecuteInFunctionContext : Execute;

            while (IsTruthy(exec(node.Condition)))
            {
                _consciousnessEngine.UpdateAwareness(0.05);

                lastValue = exec(node.Body);

                if (_shouldBreak)
                {
                    _shouldBreak = false;
                    break;
                }

                if (_shouldContinue)
                {
                    _shouldContinue = false;
                    continue;
                }

                if (_shouldReturn)
                {
                    break;
                }
            }

            return lastValue;
        }

        /// <inheritdoc/>
        public object? VisitFor(NSLForNode node)
        {
            _consciousnessEngine.IncrementCycles();

            Func<NSLASTNode, object?> exec = _inFunctionContext ? ExecuteInFunctionContext : Execute;

            // Evaluate the iterable expression
            var iterable = exec(node.Iterable);

            // Create new scope for the loop
            EnterScope();

            object? lastValue = null;

            try
            {
                // Handle different iterable types
                IEnumerable<object?> items;

                if (iterable is List<object> list)
                {
                    items = list.Cast<object?>();
                }
                else if (iterable is List<object?> nullableList)
                {
                    items = nullableList;
                }
                else if (iterable is System.Collections.IList genericList)
                {
                    // Handle List<Dictionary<string, object?>> and other generic lists
                    items = genericList.Cast<object?>();
                }
                else if (iterable is System.Collections.IEnumerable enumerable && iterable is not string)
                {
                    // Handle any other IEnumerable (arrays, etc.)
                    items = enumerable.Cast<object?>();
                }
                else if (iterable is string str)
                {
                    // String iteration - yield each character as a single-character string
                    items = str.Select(c => (object?)c.ToString());
                }
                else
                {
                    throw new NSLRuntimeException($"Object of type '{GetNSLTypeName(iterable)}' is not iterable");
                }

                // For each item in the iterable
                foreach (var item in items)
                {
                    // Assign item to loop variable
                    SetVariable(node.Variable.Value, item);

                    // Execute body
                    lastValue = exec(node.Body);

                    // Handle break/continue
                    if (_shouldBreak)
                    {
                        _shouldBreak = false;
                        break;
                    }

                    if (_shouldContinue)
                    {
                        _shouldContinue = false;
                        continue;
                    }

                    if (_shouldReturn)
                    {
                        break;
                    }
                }
            }
            finally
            {
                ExitScope();
            }

            _consciousnessEngine.UpdateAwareness(0.1);
            return lastValue;
        }

        /// <inheritdoc/>
        public object? VisitBlock(NSLBlockNode node)
        {
            return VisitBlockInternal(node, false);
        }
        
        /// <summary>
        /// Internal block execution that can handle function context
        /// </summary>
        private object? VisitBlockInternal(NSLBlockNode node, bool inFunctionContext)
        {
            _scopes.Push(new Dictionary<string, object?>());

            object? lastValue = null;

            try
            {
                foreach (var statement in node.Statements)
                {
                    // Call debug callback before each statement
                    if (!OnDebugStatement(statement))
                    {
                        break; // Debug callback requested stop
                    }

                    lastValue = inFunctionContext ? ExecuteInFunctionContext(statement) : Execute(statement);

                    if (_shouldBreak || _shouldContinue || _shouldReturn)
                    {
                        break;
                    }
                }
            }
            finally
            {
                _scopes.Pop();
            }

            return lastValue;
        }

        /// <inheritdoc/>
        public object? VisitBreak(NSLBreakNode node)
        {
            _consciousnessEngine.IncrementCycles();
            _shouldBreak = true;
            return null;
        }

        /// <inheritdoc/>
        public object? VisitContinue(NSLContinueNode node)
        {
            _consciousnessEngine.IncrementCycles();
            _shouldContinue = true;
            return null;
        }

        /// <inheritdoc/>
        public object? VisitReturn(NSLReturnNode node)
        {
            _consciousnessEngine.IncrementCycles();
            object? value = node.Value?.Accept(this);
            throw new NSLReturnException(value);
        }

        #endregion

        /// <inheritdoc/>
        public object? VisitFunction(NSLFunctionNode node)
        {
            // Use CaptureScopeChain to capture all enclosing scopes for proper mutable closure behavior
            var function = new NSLFunction(node.Name, node.Parameters.Select(p => p.Name).ToList(), node.Body, CaptureScopeChain());
            SetVariable(node.Name, function);
            return function;
        }

        /// <inheritdoc/>
        public object? VisitClass(NSLClassNode node)
        {
            // Create a class as a dictionary factory (constructor function)
            var className = node.Name;
            var classBody = node.Body;
            
            // Create class constructor function
            var constructor = new NSLBuiltinFunction(className, (args) => {
                // Create instance as dictionary
                var instance = new Dictionary<string, object?>();
                instance["__class__"] = className;
                
                // Enter a new scope for the class body evaluation
                EnterScope();
                try
                {
                    // Set 'this' reference to instance so methods can access it
                    SetVariable("this", instance);
                    
                    // Execute body - this will define methods as functions in current scope
                    Execute(classBody);
                    
                    // Copy defined variables from scope to instance
                    // Access _scopes directly since we need the current scope
                    if (_scopes.Count > 0)
                    {
                        foreach (var kv in _scopes.Peek())
                        {
                            if (kv.Key != "this")
                            {
                                instance[kv.Key] = kv.Value;
                            }
                        }
                    }
                }
                finally
                {
                    ExitScope();
                }
                
                // Call init if it exists
                if (instance.ContainsKey("init") && instance["init"] is NSLFunction initFn)
                {
                    CallUserFunction(initFn, args);
                }
                
                return instance;
            });
            
            // Register the class constructor globally
            SetVariable(className, constructor);
            
            return constructor;
        }

        /// <inheritdoc/>
        public object? VisitMatch(NSLMatchNode node)
        {
            // Evaluate the value being matched
            var value = Execute(node.Value);

            // Try each case in order
            foreach (var matchCase in node.Cases)
            {
                EnterScope();
                try
                {
                    if (TryMatchPattern(matchCase.Pattern, value))
                    {
                        // Check guard condition if present
                        if (matchCase.Guard != null)
                        {
                            var guardResult = Execute(matchCase.Guard);
                            if (!IsTruthy(guardResult))
                            {
                                continue; // Guard failed, try next case
                            }
                        }

                        // Pattern matched and guard passed - execute body
                        return Execute(matchCase.Body);
                    }
                }
                finally
                {
                    ExitScope();
                }
            }

            throw new NSLRuntimeException($"No pattern matched value: {ConvertToString(value)}");
        }

        /// <summary>
        /// Try to match a pattern against a value, binding variables if successful
        /// </summary>
        private bool TryMatchPattern(NSLASTNode pattern, object? value)
        {
            switch (pattern)
            {
                case NSLResultNode resultPattern:
                    // Match ok(v) or err(e) patterns
                    if (value is NSLResult result)
                    {
                        if (resultPattern.IsOk == result.IsOk)
                        {
                            // Extract inner value and match binding
                            return TryMatchPattern(resultPattern.Value, result.Value);
                        }
                    }
                    return false;

                case NSLOptionalNode optionalPattern:
                    // Match some(v) or none patterns
                    if (value is NSLOptional optional)
                    {
                        if (optionalPattern.HasValue == optional.HasValue)
                        {
                            if (!optionalPattern.HasValue)
                            {
                                return true; // none matches none
                            }
                            // Extract inner value and match binding
                            return TryMatchPattern(optionalPattern.Value!, optional.Value);
                        }
                    }
                    return false;

                case NSLLiteralNode literal:
                    // Match literal values exactly
                    return ValuesEqual(literal.Value, value);

                case NSLIdentifierNode identifier:
                    // Wildcard pattern - always matches and binds the value
                    if (identifier.Name == "_")
                    {
                        return true; // Underscore is a discard pattern
                    }
                    // Bind the value to the identifier in current scope
                    SetVariable(identifier.Name, value);
                    return true;

                default:
                    throw new NSLRuntimeException($"Unsupported pattern type: {pattern.GetType().Name}");
            }
        }

        /// <summary>
        /// Check if two values are equal for pattern matching purposes
        /// </summary>
        private bool ValuesEqual(object? a, object? b)
        {
            if (a == null && b == null) return true;
            if (a == null || b == null) return false;

            // Handle numeric comparisons
            if (IsNumeric(a) && IsNumeric(b))
            {
                return Math.Abs(Convert.ToDouble(a) - Convert.ToDouble(b)) < 0.0001;
            }

            return a.Equals(b);
        }

        private bool IsNumeric(object? value)
        {
            return value is double or int or long or float or decimal;
        }

        /// <inheritdoc/>
        public object? VisitListComprehension(NSLListComprehensionNode node)
        {
            // [expr for var in iterable if condition]
            var iterable = node.Iterable.Accept(this);
            var result = new List<object?>();

            if (iterable is IEnumerable<object?> enumerable)
            {
                EnterScope();
                foreach (var item in enumerable)
                {
                    _scopes.Peek()[node.Variable] = item;

                    // Check condition if present
                    if (node.Condition != null)
                    {
                        var conditionResult = node.Condition.Accept(this);
                        if (!IsTruthy(conditionResult))
                            continue;
                    }

                    var value = node.Expression.Accept(this);
                    result.Add(value);
                }
                ExitScope();
            }

            return result;
        }

        /// <inheritdoc/>
        public object? VisitVariableDeclaration(NSLVariableDeclarationNode node)
        {
            object? value = null;
            if (node.Value != null)
            {
                value = node.Value.Accept(this);
            }

            if (_scopes.Count > 0)
            {
                _scopes.Peek()[node.Name] = value;
            }
            else
            {
                _globals[node.Name] = value;
            }

            return value;
        }

        /// <inheritdoc/>
        public object? VisitSafeNavigation(NSLSafeNavigationNode node)
        {
            var obj = node.Object.Accept(this);
            if (obj == null) return null;

            // Try to get property from dictionary
            if (obj is Dictionary<string, object?> dict && dict.TryGetValue(node.Property, out var dictValue))
            {
                return dictValue;
            }

            return null;
        }

        /// <inheritdoc/>
        public object? VisitPipeline(NSLPipelineNode node)
        {
            // data |> func becomes func(data)
            var left = node.Left.Accept(this);

            if (node.Right is NSLFunctionCallNode funcCall)
            {
                // Prepend left value to arguments
                var args = new List<object?> { left };
                foreach (var arg in funcCall.Arguments)
                {
                    args.Add(arg.Accept(this));
                }

                var callee = funcCall.Function.Accept(this);
                return InvokeCallable(callee, args.ToArray());
            }
            else if (node.Right is NSLIdentifierNode identifier)
            {
                // Treat as function call with single argument
                var callee = GetVariable(identifier.Name);
                return InvokeCallable(callee, new[] { left });
            }
            else if (node.Right is NSLGetNode getNode)
            {
                // Handle namespace.function like math.sqrt or list.sum
                var callee = getNode.Accept(this);
                return InvokeCallable(callee, new[] { left });
            }
            else if (node.Right is NSLLambdaNode)
            {
                // Handle lambda expressions like |> (x) => x * 2
                var callee = node.Right.Accept(this);
                return InvokeCallable(callee, new[] { left });
            }

            throw new NSLRuntimeException("Pipeline operator (|>) requires a function on the right side. Example: data |> list.sort |> list.reverse or data |> fn(x) { x * 2 }");
        }

        /// <inheritdoc/>
        public object? VisitRange(NSLRangeNode node)
        {
            if (node.Start == null || node.End == null)
                throw new NSLRuntimeException("Range requires start and end values. Example: 0..5 or 0..=5");

            var start = node.Start.Accept(this);
            var end = node.End.Accept(this);

            // Convert to double, handling int, long, double, and other numeric types
            double? startNum = ConvertToDouble(start);
            double? endNum = ConvertToDouble(end);

            if (startNum.HasValue && endNum.HasValue)
            {
                var result = new List<object?>();
                int startInt = (int)startNum.Value;
                int endInt = (int)endNum.Value;

                if (node.IsInclusive)
                    endInt++; // Include the end value

                for (int i = startInt; i < endInt; i++)
                {
                    result.Add((double)i);
                }
                return result;
            }

            throw new NSLRuntimeException($"Range requires numeric start and end values. Got start={start?.GetType().Name ?? "null"}, end={end?.GetType().Name ?? "null"}. Example: for i in 0..5 {{ ... }}");
        }

        /// <summary>Helper to convert any numeric type to double</summary>
        private double? ConvertToDouble(object? value)
        {
            return value switch
            {
                double d => d,
                int i => (double)i,
                long l => (double)l,
                float f => (double)f,
                decimal dec => (double)dec,
                short s => (double)s,
                byte b => (double)b,
                string s when double.TryParse(s, out var parsed) => parsed,
                _ => null
            };
        }

        /// <inheritdoc/>
        public object? VisitCast(NSLCastNode node)
        {
            var value = node.Value.Accept(this);

            return node.TargetType.ToLower() switch
            {
                "number" => ConvertToNumber(value),
                "string" => ConvertToString(value),
                "bool" or "boolean" => IsTruthy(value),
                "vec" or "mat" or "tensor" => value, // Pass-through for now
                _ => value
            };
        }

        /// <inheritdoc/>
        public object? VisitResult(NSLResultNode node)
        {
            var value = node.Value?.Accept(this);
            return new NSLResult(node.IsOk, value);
        }

        /// <inheritdoc/>
        public object? VisitOptional(NSLOptionalNode node)
        {
            if (!node.HasValue) return new NSLOptional(null, false);
            var value = node.Value?.Accept(this);
            return new NSLOptional(value, true);
        }

        /// <inheritdoc/>
        public object? VisitTypeAlias(NSLTypeAliasNode node)
        {
            // Type aliases are compile-time constructs, nothing to do at runtime
            return null;
        }

        /// <inheritdoc/>
        public object? VisitObject(NSLObjectNode node)
        {
            var result = new Dictionary<string, object?>();
            foreach (var field in node.Fields)
            {
                result[field.Key] = field.Value.Accept(this);
            }
            return result;
        }

        /// <inheritdoc/>
        public object? VisitStruct(NSLStructNode node)
        {
            // Register struct type - nothing to execute at runtime
            return null;
        }

        /// <inheritdoc/>
        public object? VisitStructInstantiation(NSLStructInstantiationNode node)
        {
            var result = new Dictionary<string, object?>();
            foreach (var field in node.Fields)
            {
                result[field.Key] = field.Value.Accept(this);
            }
            return result;
        }

        /// <inheritdoc/>
        public object? VisitEnum(NSLEnumNode node)
        {
            // Register enum type - nothing to execute at runtime
            return null;
        }

        /// <inheritdoc/>
        public object? VisitEnumVariant(NSLEnumVariantNode node)
        {
            var arguments = node.Arguments.Select(a => a.Accept(this)).ToList();
            return new NSLEnumValue(node.EnumName, node.VariantName, arguments);
        }

        /// <inheritdoc/>
        public object? VisitTrait(NSLTraitNode node)
        {
            // Register trait - nothing to execute at runtime
            return null;
        }

        /// <inheritdoc/>
        public object? VisitImpl(NSLImplNode node)
        {
            // Register implementation - for now, just execute the method definitions
            foreach (var method in node.Methods)
            {
                var mangledName = $"{node.TypeName}_{node.TraitName}_{method.Name}";
                _globals[mangledName] = new NSLFunction(method.Name, method.Parameters.Select(p => p.Name).ToList(), method.Body, new List<Dictionary<string, object?>>());
            }
            return null;
        }

        /// <inheritdoc/>
        public object? VisitAsyncFunction(NSLAsyncFunctionNode node)
        {
            // Create an async function wrapper with proper closure support
            var asyncFunc = new NSLAsyncFunction(node.Name, node.Parameters.Select(p => p.Name).ToList(), node.Body, CaptureScopeChain());
            _globals[node.Name] = asyncFunc;
            return asyncFunc;
        }

        /// <inheritdoc/>
        public object? VisitAwait(NSLAwaitNode node)
        {
            var value = node.Expression.Accept(this);

            // If it's a Task, wait for it
            if (value is Task task)
            {
                task.Wait();
                if (value is Task<object?> typedTask)
                {
                    return typedTask.Result;
                }
                return null;
            }

            // If it's already resolved, just return it
            return value;
        }

        /// <inheritdoc/>
        public object? VisitModule(NSLModuleNode node)
        {
            // Module is a namespace declaration, not a scope
            // Execute body in current scope so pub fn declarations are accessible for export
            node.Body.Accept(this);
            return null;
        }

        /// <inheritdoc/>
        public object? VisitImport(NSLImportNode node)
        {
            // Handle file path imports: import "path/to/file.nsl"
            if (!string.IsNullOrEmpty(node.FilePath))
            {
                var filePath = ResolveModuleFilePath(node.FilePath);
                LoadAndExecuteModule(filePath, node.ModuleAlias);
                return null;
            }

            // Handle module path imports: import { fn1, fn2 } from ai::core
            if (node.ModulePath != null && node.ModulePath.Count > 0)
            {
                var moduleName = string.Join("::", node.ModulePath);
                var filePath = ResolveModulePath(node.ModulePath);

                // Load the module if not already loaded
                if (!_loadedModules.Contains(filePath))
                {
                    LoadAndExecuteModule(filePath, null);
                }

                // Get the module's exports
                if (!_moduleExports.TryGetValue(filePath, out var exports))
                {
                    exports = new Dictionary<string, object?>();
                }

                // Handle different import styles
                if (node.IsWildcard)
                {
                    // import * from module - import all exports
                    foreach (var (name, value) in exports)
                    {
                        SetVariable(name, value);
                    }
                }
                else if (!string.IsNullOrEmpty(node.ModuleAlias))
                {
                    // import module as alias - create module object
                    SetVariable(node.ModuleAlias, exports);
                }
                else if (node.Items != null && node.Items.Count > 0)
                {
                    // import { fn1, fn2 as alias2 } from module - selective import
                    foreach (var item in node.Items)
                    {
                        var importName = item.Name;
                        var localName = item.Alias ?? item.Name;

                        if (exports.TryGetValue(importName, out var value))
                        {
                            SetVariable(localName, value);
                        }
                        else
                        {
                            throw new NSLRuntimeException($"Module '{moduleName}' does not export '{importName}'");
                        }
                    }
                }
                else
                {
                    // import module - import as module object
                    var moduleObjectName = node.ModulePath[node.ModulePath.Count - 1];
                    SetVariable(moduleObjectName, exports);
                }
            }

            return null;
        }

        /// <summary>
        /// Resolve a module path like ["ai", "core"] to a file path
        /// </summary>
        private string ResolveModulePath(IReadOnlyList<string> modulePath)
        {
            // Convert module path to file path: ai::core -> ai/core.nsl
            var relativePath = string.Join(Path.DirectorySeparatorChar.ToString(), modulePath) + ".nsl";

            // Resolve relative to current source file's directory
            if (!string.IsNullOrEmpty(_currentSourceFile))
            {
                var baseDir = Path.GetDirectoryName(_currentSourceFile) ?? ".";
                var fullPath = Path.GetFullPath(Path.Combine(baseDir, relativePath));
                if (File.Exists(fullPath))
                {
                    return fullPath;
                }
            }

            // Try current working directory
            var cwdPath = Path.GetFullPath(relativePath);
            if (File.Exists(cwdPath))
            {
                return cwdPath;
            }

            throw new NSLRuntimeException($"Module not found: {string.Join("::", modulePath)} (looked for {relativePath})");
        }

        /// <summary>
        /// Resolve a file path import
        /// </summary>
        private string ResolveModuleFilePath(string filePath)
        {
            // Remove quotes if present
            var cleanPath = filePath.Trim('"', '\'');

            // Resolve relative to current source file's directory
            if (!string.IsNullOrEmpty(_currentSourceFile))
            {
                var baseDir = Path.GetDirectoryName(_currentSourceFile) ?? ".";
                var fullPath = Path.GetFullPath(Path.Combine(baseDir, cleanPath));
                if (File.Exists(fullPath))
                {
                    return fullPath;
                }
            }

            // Try current working directory
            var cwdPath = Path.GetFullPath(cleanPath);
            if (File.Exists(cwdPath))
            {
                return cwdPath;
            }

            throw new NSLRuntimeException($"Import file not found: {filePath}");
        }

        /// <summary>
        /// Load and execute a module file, collecting its exports
        /// </summary>
        private void LoadAndExecuteModule(string filePath, string? alias)
        {
            if (_loadedModules.Contains(filePath))
            {
                return; // Already loaded
            }

            _loadedModules.Add(filePath);
            var exports = new Dictionary<string, object?>();
            _moduleExports[filePath] = exports;

            var previousSourceFile = _currentSourceFile;
            var previousModuleExports = _currentModuleExports;
            try
            {
                _currentSourceFile = filePath;
                _currentModuleExports = exports; // Set context for VisitExport

                var content = File.ReadAllText(filePath);
                var lexer = new NSLLexer(content, filePath);
                var parser = new NSLParser();
                var tokens = lexer.Tokenize();
                var ast = parser.Parse(tokens);

                // Execute the module - VisitExport will populate _currentModuleExports
                Execute(ast);
            }
            finally
            {
                _currentSourceFile = previousSourceFile;
                _currentModuleExports = previousModuleExports;
            }

            // If an alias was provided, set it
            if (!string.IsNullOrEmpty(alias))
            {
                SetVariable(alias, exports);
            }
        }

        /// <inheritdoc/>
        public object? VisitExport(NSLExportNode node)
        {
            // Handle pub fn, pub let, etc.
            if (node.Declaration != null)
            {
                // Execute the declaration to define the function/variable
                var result = node.Declaration.Accept(this);

                // If we're loading a module, add to exports
                if (_currentModuleExports != null)
                {
                    // Extract the name from the declaration
                    string? name = node.Declaration switch
                    {
                        NSLFunctionNode fn => fn.Name,
                        NSLVariableDeclarationNode varDecl => varDecl.Name,
                        NSLAsyncFunctionNode asyncFn => asyncFn.Name,
                        _ => null
                    };

                    if (name != null)
                    {
                        // Get the value that was just defined
                        var value = GetVariable(name);
                        _currentModuleExports[name] = value;
                    }
                }

                return result;
            }

            // Handle export { name1, name2 } syntax
            if (node.Items != null && _currentModuleExports != null)
            {
                foreach (var item in node.Items)
                {
                    var value = GetVariable(item.Name);
                    var exportName = item.Alias ?? item.Name;
                    _currentModuleExports[exportName] = value;
                }
            }

            return null;
        }

        #endregion

        #region Arithmetic Operations

        private object PerformAddition(object? left, object? right)
        {
            // Convert numeric types to double
            var l = ToDouble(left);
            var r = ToDouble(right);

            if (l.HasValue && r.HasValue)
                return l.Value + r.Value;

            // Handle array concatenation
            if (left is List<object?> leftList && right is List<object?> rightList)
            {
                var result = new List<object?>(leftList);
                result.AddRange(rightList);
                return result;
            }

            return (left, right) switch
            {
                (string sl, string sr) => sl + sr,
                (string sl, _) => sl + ConvertToString(right),
                (_, string sr) => ConvertToString(left) + sr,
                _ => throw new NSLRuntimeException($"Cannot add {GetNSLTypeName(left)} and {GetNSLTypeName(right)}")
            };
        }

        private double? ToDouble(object? value)
        {
            return value switch
            {
                double d => d,
                int i => (double)i,
                long l => (double)l,
                float f => (double)f,
                decimal dec => (double)dec,
                _ => null
            };
        }
        
        /// <summary>
        /// Convert JsonElement to native .NET types for proper comparison
        /// </summary>
        private static object? ConvertJsonElement(System.Text.Json.JsonElement element)
        {
            return element.ValueKind switch
            {
                System.Text.Json.JsonValueKind.Object => element.EnumerateObject()
                    .ToDictionary(p => p.Name, p => ConvertJsonElement(p.Value)),
                System.Text.Json.JsonValueKind.Array => element.EnumerateArray()
                    .Select(ConvertJsonElement).ToList(),
                System.Text.Json.JsonValueKind.String => element.GetString(),
                System.Text.Json.JsonValueKind.Number => element.TryGetInt64(out var l) ? (double)l : element.GetDouble(),
                System.Text.Json.JsonValueKind.True => true,
                System.Text.Json.JsonValueKind.False => false,
                System.Text.Json.JsonValueKind.Null => null,
                _ => element.ToString()
            };
        }

        private object PerformSubtraction(object? left, object? right)
        {
            var l = ToDouble(left);
            var r = ToDouble(right);

            if (l.HasValue && r.HasValue)
                return l.Value - r.Value;

            throw new NSLRuntimeException($"Cannot subtract {GetNSLTypeName(right)} from {GetNSLTypeName(left)}");
        }

        private object PerformMultiplication(object? left, object? right)
        {
            // Handle string repetition first (before numeric conversion)
            if (left is string s1 && right != null)
            {
                var count = ToDouble(right);
                if (count.HasValue && count.Value >= 0)
                    return string.Concat(Enumerable.Repeat(s1, (int)count.Value));
            }
            if (right is string s2 && left != null)
            {
                var count = ToDouble(left);
                if (count.HasValue && count.Value >= 0)
                    return string.Concat(Enumerable.Repeat(s2, (int)count.Value));
            }
            
            // Numeric multiplication
            var l = ToDouble(left);
            var r = ToDouble(right);

            if (l.HasValue && r.HasValue)
                return l.Value * r.Value;

            throw new NSLRuntimeException($"Cannot multiply {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private object PerformDivision(object? left, object? right)
        {
            var l = ToDouble(left);
            var r = ToDouble(right);

            if (l.HasValue && r.HasValue)
            {
                if (Math.Abs(r.Value) < double.Epsilon)
                    throw new NSLRuntimeException("Division by zero");
                return l.Value / r.Value;
            }

            throw new NSLRuntimeException($"Cannot divide {GetNSLTypeName(left)} by {GetNSLTypeName(right)}");
        }

        private object PerformIntegerDivision(object? left, object? right)
        {
            var l = ToDouble(left);
            var r = ToDouble(right);

            if (l.HasValue && r.HasValue)
            {
                if (Math.Abs(r.Value) < double.Epsilon)
                    throw new NSLRuntimeException("Integer division by zero");
                return Math.Floor(l.Value / r.Value);
            }

            throw new NSLRuntimeException($"Cannot perform integer division on {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private object PerformModulo(object? left, object? right)
        {
            var l = ToDouble(left);
            var r = ToDouble(right);

            if (l.HasValue && r.HasValue)
            {
                if (Math.Abs(r.Value) < double.Epsilon)
                    throw new NSLRuntimeException("Modulo by zero");
                return l.Value % r.Value;
            }

            throw new NSLRuntimeException($"Cannot perform modulo operation on {GetNSLTypeName(left)} and {GetNSLTypeName(right)}");
        }

        private object PerformPower(object? left, object? right)
        {
            var l = ToDouble(left);
            var r = ToDouble(right);

            if (l.HasValue && r.HasValue)
                return Math.Pow(l.Value, r.Value);

            throw new NSLRuntimeException($"Cannot raise {GetNSLTypeName(left)} to the power of {GetNSLTypeName(right)}");
        }

        private object PerformNegation(object? operand)
        {
            var d = ToDouble(operand);
            if (d.HasValue)
                return -d.Value;

            throw new NSLRuntimeException($"Cannot negate {GetNSLTypeName(operand)}");
        }

        #endregion

        #region Comparison Operations

        private bool AreEqual(object? left, object? right)
        {
            if (left == null && right == null) return true;
            if (left == null || right == null) return false;

            var l = ToDouble(left);
            var r = ToDouble(right);
            if (l.HasValue && r.HasValue)
                return Math.Abs(l.Value - r.Value) < double.Epsilon;

            return left.Equals(right);
        }

        private bool IsLess(object? left, object? right)
        {
            var l = ToDouble(left);
            var r = ToDouble(right);
            if (l.HasValue && r.HasValue)
                return l.Value < r.Value;

            return (left, right) switch
            {
                (string sl, string sr) => string.Compare(sl, sr) < 0,
                _ => throw new NSLRuntimeException($"Cannot compare {GetNSLTypeName(left)} and {GetNSLTypeName(right)}")
            };
        }

        private bool IsGreater(object? left, object? right)
        {
            var l = ToDouble(left);
            var r = ToDouble(right);
            if (l.HasValue && r.HasValue)
                return l.Value > r.Value;

            return (left, right) switch
            {
                (string sl, string sr) => string.Compare(sl, sr) > 0,
                _ => throw new NSLRuntimeException($"Cannot compare {GetNSLTypeName(left)} and {GetNSLTypeName(right)}")
            };
        }

        #endregion

        #region Variable Management

        /// <summary>
        /// Sets a variable value in the appropriate scope
        /// </summary>
        public void SetVariable(string name, object? value)
        {
            // First, check if variable exists in any local scope
            foreach (var scope in _scopes)
            {
                if (scope.ContainsKey(name))
                {
                    scope[name] = value;
                    return;
                }
            }

            // Check captured closure scopes - update there if found (for mutable closures)
            if (_currentClosureScopes != null)
            {
                foreach (var scope in _currentClosureScopes)
                {
                    if (scope.ContainsKey(name))
                    {
                        scope[name] = value;
                        return;
                    }
                }
            }

            // Check globals
            if (_globals.ContainsKey(name))
            {
                _globals[name] = value;
                return;
            }

            // Variable doesn't exist, create in current scope (or global if no scope)
            if (_scopes.Count > 0)
            {
                _scopes.Peek()[name] = value;
            }
            else
            {
                _globals[name] = value;
            }
        }

        private object? GetVariable(string name)
        {
            // Check local scopes first (stack order)
            foreach (var scope in _scopes)
            {
                if (scope.TryGetValue(name, out var value))
                    return value;
            }

            // Check captured closure scopes
            if (_currentClosureScopes != null)
            {
                foreach (var scope in _currentClosureScopes)
                {
                    if (scope.TryGetValue(name, out var closureValue))
                        return closureValue;
                }
            }

            // Check globals
            if (_globals.TryGetValue(name, out var globalValue))
                return globalValue;

            throw new NSLRuntimeException($"Undefined variable: {name}");
        }

        private void EnterScope()
        {
            _scopes.Push(new Dictionary<string, object?>());
        }

        private void ExitScope()
        {
            if (_scopes.Count > 0)
                _scopes.Pop();
        }

        private Dictionary<string, object?> GetCurrentEnvironment()
        {
            return _scopes.Count > 0 ? _scopes.Peek() : _globals;
        }

        #endregion

        #region Consciousness Operations

        private object ProcessHolographic(object? operand)
        {
            // Basic holographic processing - wrap in consciousness context
            return new NSLConsciousnessData
            {
                Type = "holographic",
                Data = operand,
                ProcessingTime = DateTime.UtcNow,
                ConsciousnessLevel = _consciousnessEngine.GetCurrentAwarenessLevel()
            };
        }

        private object ProcessGradient(object? operand)
        {
            // Basic gradient processing - calculate rate of change
            return new NSLConsciousnessData
            {
                Type = "gradient",
                Data = operand,
                ProcessingTime = DateTime.UtcNow,
                ConsciousnessLevel = _consciousnessEngine.CalculateGradient(operand)
            };
        }

        private object ProcessTensorProduct(object? operand)
        {
            // Basic tensor product processing - mark for tensor product execution
            return new NSLConsciousnessData
            {
                Type = "tensor_product",
                Data = operand,
                ProcessingTime = DateTime.UtcNow,
                ConsciousnessLevel = _consciousnessEngine.GetCurrentAwarenessLevel()
            };
        }

        #endregion

        #region User Function Support

        private object? CallUserFunction(NSLFunction function, object?[] args)
        {
            if (args.Length != function.Parameters.Count)
            {
                throw new NSLRuntimeException($"Function expects {function.Parameters.Count} arguments but got {args.Length}");
            }

            EnterScope();
            var previousFunctionContext = _inFunctionContext;
            var previousClosureScopes = _currentClosureScopes;
            _inFunctionContext = true;
            _currentClosureScopes = function.CapturedScopes;

            try
            {
                // Bind parameters to the local scope
                for (int i = 0; i < function.Parameters.Count; i++)
                {
                    _scopes.Peek()[function.Parameters[i]] = args[i];
                }

                // Execute function body
                if (function.Body is NSLBlockNode blockNode)
                {
                    return VisitBlockInternal(blockNode, true);
                }
                else
                {
                    return ExecuteInFunctionContext(function.Body);
                }
            }
            catch (NSLReturnException returnEx)
            {
                return returnEx.Value;
            }
            finally
            {
                _inFunctionContext = previousFunctionContext;
                _currentClosureScopes = previousClosureScopes;
                ExitScope();
            }
        }

        #endregion

        #region Utility Methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool IsTruthy(object? value)
        {
            return value switch
            {
                null => false,
                bool b => b,
                double d => d != 0.0,
                string s => s.Length > 0,
                _ => true
            };
        }

        // ===== SEMANTIC FILE ACCESS HELPERS =====

        /// <summary>
        /// Get file structure overview (headings, sections, estimated tokens)
        /// </summary>
        private Dictionary<string, object?> GetFileStructure(string content, string path)
        {
            var lines = content.Split('\n');
            var headings = new List<Dictionary<string, object?>>();
            var codeBlocks = 0;
            var tables = 0;

            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].TrimStart();

                // Markdown headings
                if (line.StartsWith("# "))
                    headings.Add(new Dictionary<string, object?> { ["level"] = 1, ["text"] = line[2..].Trim(), ["line"] = i + 1 });
                else if (line.StartsWith("## "))
                    headings.Add(new Dictionary<string, object?> { ["level"] = 2, ["text"] = line[3..].Trim(), ["line"] = i + 1 });
                else if (line.StartsWith("### "))
                    headings.Add(new Dictionary<string, object?> { ["level"] = 3, ["text"] = line[4..].Trim(), ["line"] = i + 1 });
                else if (line.StartsWith("#### "))
                    headings.Add(new Dictionary<string, object?> { ["level"] = 4, ["text"] = line[5..].Trim(), ["line"] = i + 1 });

                // Count code blocks
                if (line.StartsWith("```"))
                    codeBlocks++;

                // Count tables
                if (line.Contains("|") && line.Trim().StartsWith("|"))
                    tables++;
            }

            return new Dictionary<string, object?>
            {
                ["path"] = path,
                ["total_lines"] = lines.Length,
                ["total_chars"] = content.Length,
                ["estimated_tokens"] = content.Length / 4,
                ["headings"] = headings,
                ["heading_count"] = headings.Count,
                ["code_blocks"] = codeBlocks / 2, // pairs of ```
                ["table_rows"] = tables,
                ["sections"] = headings.Where(h => (int)(h["level"] ?? 0) <= 2).ToList()
            };
        }

        /// <summary>
        /// Get specific section from file by heading name
        /// </summary>
        private string GetFileSection(string content, string sectionName)
        {
            var lines = content.Split('\n');
            var result = new List<string>();
            var inSection = false;
            var sectionLevel = 0;
            var searchLower = sectionName.ToLowerInvariant();

            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i];
                var trimmed = line.TrimStart();

                // Check for heading
                int level = 0;
                string headingText = "";
                if (trimmed.StartsWith("#### ")) { level = 4; headingText = trimmed[5..].Trim(); }
                else if (trimmed.StartsWith("### ")) { level = 3; headingText = trimmed[4..].Trim(); }
                else if (trimmed.StartsWith("## ")) { level = 2; headingText = trimmed[3..].Trim(); }
                else if (trimmed.StartsWith("# ")) { level = 1; headingText = trimmed[2..].Trim(); }

                if (level > 0)
                {
                    if (inSection && level <= sectionLevel)
                    {
                        // End of section
                        break;
                    }

                    if (headingText.ToLowerInvariant().Contains(searchLower))
                    {
                        inSection = true;
                        sectionLevel = level;
                    }
                }

                if (inSection)
                {
                    result.Add(line);
                }
            }

            return result.Count > 0 ? string.Join("\n", result) : $"Section '{sectionName}' not found";
        }

        /// <summary>
        /// Attention-based reading - find sections most relevant to query
        /// </summary>
        private Dictionary<string, object?> AttentionRead(string content, string query, int maxSections)
        {
            var lines = content.Split('\n');
            var sections = new List<(int start, int end, string heading, double score)>();
            var queryTerms = query.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // Parse into sections
            int currentStart = 0;
            string currentHeading = "(start)";

            for (int i = 0; i < lines.Length; i++)
            {
                var trimmed = lines[i].TrimStart();
                if (trimmed.StartsWith("# ") || trimmed.StartsWith("## ") || trimmed.StartsWith("### "))
                {
                    if (i > currentStart)
                    {
                        var sectionText = string.Join("\n", lines.Skip(currentStart).Take(i - currentStart));
                        var score = CalculateRelevanceScore(sectionText, currentHeading, queryTerms);
                        sections.Add((currentStart, i, currentHeading, score));
                    }
                    currentStart = i;
                    currentHeading = trimmed.TrimStart('#').Trim();
                }
            }

            // Add final section
            if (currentStart < lines.Length)
            {
                var sectionText = string.Join("\n", lines.Skip(currentStart));
                var score = CalculateRelevanceScore(sectionText, currentHeading, queryTerms);
                sections.Add((currentStart, lines.Length, currentHeading, score));
            }

            // Sort by relevance and take top sections
            var topSections = sections.OrderByDescending(s => s.score).Take(maxSections).ToList();

            var results = new List<Dictionary<string, object?>>();
            foreach (var section in topSections.OrderBy(s => s.start))
            {
                var sectionContent = string.Join("\n", lines.Skip(section.start).Take(section.end - section.start));
                results.Add(new Dictionary<string, object?>
                {
                    ["heading"] = section.heading,
                    ["score"] = Math.Round(section.score, 3),
                    ["line_start"] = section.start + 1,
                    ["line_end"] = section.end,
                    ["content"] = sectionContent
                });
            }

            return new Dictionary<string, object?>
            {
                ["query"] = query,
                ["total_sections"] = sections.Count,
                ["returned_sections"] = results.Count,
                ["sections"] = results
            };
        }

        private double CalculateRelevanceScore(string text, string heading, string[] queryTerms)
        {
            var textLower = text.ToLowerInvariant();
            var headingLower = heading.ToLowerInvariant();
            double score = 0;

            foreach (var term in queryTerms)
            {
                // Heading match is worth more
                if (headingLower.Contains(term))
                    score += 3.0;

                // Count occurrences in text
                int count = 0;
                int index = 0;
                while ((index = textLower.IndexOf(term, index)) != -1)
                {
                    count++;
                    index += term.Length;
                }
                score += count * 0.5;
            }

            // Normalize by section length (prefer concise relevant sections)
            var length = Math.Max(text.Length, 1);
            return score / Math.Log(length + 1);
        }

        /// <summary>
        /// Extract specific patterns from content (code blocks, tables, links, etc.)
        /// </summary>
        private List<object?> ExtractPattern(string content, string patternType)
        {
            var results = new List<object?>();
            var lines = content.Split('\n');

            switch (patternType.ToLowerInvariant())
            {
                case "code":
                case "codeblocks":
                case "```":
                    // Extract all code blocks
                    bool inCode = false;
                    string language = "";
                    var codeLines = new List<string>();
                    int startLine = 0;

                    for (int i = 0; i < lines.Length; i++)
                    {
                        var line = lines[i];
                        if (line.TrimStart().StartsWith("```"))
                        {
                            if (!inCode)
                            {
                                inCode = true;
                                language = line.TrimStart().Substring(3).Trim();
                                startLine = i + 1;
                                codeLines.Clear();
                            }
                            else
                            {
                                results.Add(new Dictionary<string, object?>
                                {
                                    ["language"] = language,
                                    ["line_start"] = startLine,
                                    ["line_end"] = i,
                                    ["content"] = string.Join("\n", codeLines)
                                });
                                inCode = false;
                            }
                        }
                        else if (inCode)
                        {
                            codeLines.Add(line);
                        }
                    }
                    break;

                case "nsl":
                    // Extract only NSL code blocks
                    var allCode = ExtractPattern(content, "code");
                    if (allCode != null)
                    {
                        foreach (var item in allCode)
                        {
                            if (item is not Dictionary<string, object?> block) continue;
                            var langObj = block.GetValueOrDefault("language");
                            var lang = langObj?.ToString()?.ToLowerInvariant() ?? "";
                            if (lang == "nsl" || lang == "")
                                results.Add(block);
                        }
                    }
                    break;

                case "tables":
                case "table":
                    // Extract markdown tables
                    bool inTable = false;
                    var tableLines = new List<string>();
                    int tableStart = 0;

                    for (int i = 0; i < lines.Length; i++)
                    {
                        var line = lines[i].Trim();
                        if (line.StartsWith("|") && line.EndsWith("|"))
                        {
                            if (!inTable)
                            {
                                tableStart = i + 1;
                                inTable = true;
                            }
                            tableLines.Add(line);
                        }
                        else if (inTable)
                        {
                            results.Add(new Dictionary<string, object?>
                            {
                                ["line_start"] = tableStart,
                                ["line_end"] = i,
                                ["rows"] = tableLines.Count,
                                ["content"] = string.Join("\n", tableLines)
                            });
                            inTable = false;
                            tableLines.Clear();
                        }
                    }
                    if (inTable)
                    {
                        results.Add(new Dictionary<string, object?>
                        {
                            ["line_start"] = tableStart,
                            ["rows"] = tableLines.Count,
                            ["content"] = string.Join("\n", tableLines)
                        });
                    }
                    break;

                case "headings":
                case "headers":
                    for (int i = 0; i < lines.Length; i++)
                    {
                        var line = lines[i].TrimStart();
                        if (line.StartsWith("#"))
                        {
                            int level = 0;
                            while (level < line.Length && line[level] == '#') level++;
                            results.Add(new Dictionary<string, object?>
                            {
                                ["level"] = level,
                                ["text"] = line.Substring(level).Trim(),
                                ["line"] = i + 1
                            });
                        }
                    }
                    break;

                case "links":
                    // Extract markdown links [text](url)
                    var linkRegex = new System.Text.RegularExpressions.Regex(@"\[([^\]]+)\]\(([^)]+)\)");
                    for (int i = 0; i < lines.Length; i++)
                    {
                        var matches = linkRegex.Matches(lines[i]);
                        foreach (System.Text.RegularExpressions.Match match in matches)
                        {
                            results.Add(new Dictionary<string, object?>
                            {
                                ["text"] = match.Groups[1].Value,
                                ["url"] = match.Groups[2].Value,
                                ["line"] = i + 1
                            });
                        }
                    }
                    break;

                case "functions":
                case "fn":
                    // Extract NSL function definitions
                    var fnRegex = new System.Text.RegularExpressions.Regex(@"(?:fn|function)\s+(\w+)\s*\(([^)]*)\)");
                    for (int i = 0; i < lines.Length; i++)
                    {
                        var match = fnRegex.Match(lines[i]);
                        if (match.Success)
                        {
                            results.Add(new Dictionary<string, object?>
                            {
                                ["name"] = match.Groups[1].Value,
                                ["params"] = match.Groups[2].Value,
                                ["line"] = i + 1
                            });
                        }
                    }
                    break;

                default:
                    // Custom regex pattern
                    try
                    {
                        var customRegex = new System.Text.RegularExpressions.Regex(patternType);
                        for (int i = 0; i < lines.Length; i++)
                        {
                            var matches = customRegex.Matches(lines[i]);
                            foreach (System.Text.RegularExpressions.Match match in matches)
                            {
                                results.Add(new Dictionary<string, object?>
                                {
                                    ["match"] = match.Value,
                                    ["line"] = i + 1
                                });
                            }
                        }
                    }
                    catch
                    {
                        results.Add(new Dictionary<string, object?> { ["error"] = $"Unknown pattern type: {patternType}" });
                    }
                    break;
            }

            return results;
        }

        private string ConvertToString(object? value)
        {
            return value switch
            {
                null => "null",
                bool b => b ? "true" : "false",
                double d => d.ToString(),
                int i => i.ToString(),
                long l => l.ToString(),
                float f => f.ToString(),
                string s => s,
                List<object> list => FormatList(list),
                IList<object> ilist => FormatList(ilist.ToList()),
                System.Collections.IList genericList => FormatGenericList(genericList),
                Dictionary<string, object> dict => FormatDictionary(dict),
                NSLConsciousnessData consciousness => $"<consciousness:{consciousness.Type}>",
                GpuTensor tensor => $"<gpu_tensor shape=[{string.Join(", ", tensor.Shape)}] size={tensor.Size}>",
                _ => value.ToString() ?? "unknown"
            };
        }

        private string FormatList(List<object> list)
        {
            var elements = list.Select(item => ConvertToString(item));
            return "[" + string.Join(", ", elements) + "]";
        }

        private string FormatGenericList(System.Collections.IList list)
        {
            var elements = new List<string>();
            foreach (var item in list)
            {
                elements.Add(ConvertToString(item));
            }
            return "[" + string.Join(", ", elements) + "]";
        }

        private string FormatDictionary(Dictionary<string, object> dict)
        {
            var pairs = dict.Select(kvp => $"{kvp.Key}: {ConvertToString(kvp.Value)}");
            return "{" + string.Join(", ", pairs) + "}";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private string GetNSLTypeName(object? value)
        {
            if (value == null) return "null";
            if (value is bool) return "boolean";
            if (value is double || value is int || value is long || value is float) return "number";
            if (value is string) return "string";
            if (value is List<object> || value is List<object?>) return "array";
            if (value is System.Collections.IList) return "array";
            if (value is System.Collections.IDictionary) return "dict";
            if (value.GetType().IsGenericType && value.GetType().GetGenericTypeDefinition() == typeof(Dictionary<,>))
                return "dict";
            if (value is NSLFunction) return "function";
            if (value is NSLBuiltinFunction) return "builtin_function";
            if (value is NSLConsciousnessData) return "consciousness";
            if (value is NSLResult) return "result";
            if (value is NSLOptional) return "optional";
            if (value is NSLEnumValue) return "enum";
            if (value is NSLAsyncFunction) return "async_function";
            if (value is GpuTensor) return "gpu_tensor";
            return "unknown";
        }

        // ===== GPU Helper Methods =====

        private static void EnsureGpuInitialized()
        {
            lock (_gpuLock)
            {
                if (_gpuConfig == null)
                {
                    _gpuConfig = new GpuAutoConfig();
                    _gpuKernels = _gpuConfig.GetKernels();
                }
            }
        }

        private static float[] ConvertToFloatArray(object? value)
        {
            if (value is float[] floatArr)
                return floatArr;

            if (value is double[] doubleArr)
                return doubleArr.Select(d => (float)d).ToArray();

            if (value is System.Collections.IList list)
            {
                var result = new float[list.Count];
                for (int i = 0; i < list.Count; i++)
                {
                    result[i] = list[i] switch
                    {
                        double d => (float)d,
                        float f => f,
                        int n => (float)n,
                        long l => (float)l,
                        _ => throw new NSLRuntimeException($"Cannot convert {list[i]?.GetType().Name} to float")
                    };
                }
                return result;
            }

            throw new NSLRuntimeException("Cannot convert value to float array");
        }

        private static int[] ConvertToIntArray(object? value)
        {
            if (value is int[] intArr)
                return intArr;

            if (value is System.Collections.IList list)
            {
                var result = new int[list.Count];
                for (int i = 0; i < list.Count; i++)
                {
                    result[i] = list[i] switch
                    {
                        int n => n,
                        double d => (int)d,
                        long l => (int)l,
                        float f => (int)f,
                        _ => throw new NSLRuntimeException($"Cannot convert {list[i]?.GetType().Name} to int")
                    };
                }
                return result;
            }

            throw new NSLRuntimeException("Cannot convert value to int array");
        }

        private object? InvokeCallable(object? callee, object?[] args)
        {
            return callee switch
            {
                NSLBuiltinFunction builtin => builtin.Call(args),
                NSLFunction userFunc => CallUserFunction(userFunc, args),
                NSLAsyncFunction asyncFunc => CallAsyncFunction(asyncFunc, args),
                // Support for Python ICallable interface (from PythonIntegration.cs)
                ICallable callable => callable.Call(this, args.ToList()!),
                // Support for NSL.Core.ICallable interface (from NSLTypes.cs)
                NSL.Core.Types.ICallable coreCallable => coreCallable.Call(this, args.ToList()),
                _ => throw new NSLRuntimeException($"'{ConvertToString(callee)}' is not callable")
            };
        }

        private Task<object?> CallAsyncFunction(NSLAsyncFunction function, object?[] args)
        {
            if (args.Length != function.Parameters.Count)
            {
                throw new NSLRuntimeException($"Async function expects {function.Parameters.Count} arguments but got {args.Length}");
            }

            // Execute asynchronously using Task.Run
            var capturedScopes = function.CapturedScopes;
            var paramNames = function.Parameters;
            var body = function.Body;

            return Task.Run(() =>
            {
                EnterScope();
                var previousFunctionContext = _inFunctionContext;
                var previousClosureScopes = _currentClosureScopes;
                _inFunctionContext = true;
                _currentClosureScopes = capturedScopes;

                try
                {
                    // Bind parameters to the local scope
                    for (int i = 0; i < paramNames.Count; i++)
                    {
                        _scopes.Peek()[paramNames[i]] = args[i];
                    }

                    // Execute function body
                    return ExecuteInFunctionContext(body);
                }
                finally
                {
                    _inFunctionContext = previousFunctionContext;
                    _currentClosureScopes = previousClosureScopes;
                    ExitScope();
                }
            });
        }

        /// <summary>
        /// Capture the current scope chain for closures.
        /// Returns a list of scope dictionaries that persist beyond the current call frame.
        /// Multiple closures defined in the same scope share the same captured scope dictionary.
        /// </summary>
        private List<Dictionary<string, object?>> CaptureScopeChain()
        {
            var captured = new List<Dictionary<string, object?>>();

            // Capture ALL local scopes (innermost first)
            // This is important because blocks create nested scopes, and we need
            // to capture the function's parameter scope which may not be at the top
            foreach (var scope in _scopes)
                {
                captured.Add(scope);
                }

            // If we're inside a closure, include its captured scopes after local scopes
            if (_currentClosureScopes != null)
                {
                captured.AddRange(_currentClosureScopes);
                }

            return captured;
        }

        // Legacy method for backwards compatibility
        private Dictionary<string, object?> CaptureScope()
        {
            var captured = new Dictionary<string, object?>();

            // Copy globals
            foreach (var kv in _globals)
                {
                captured[kv.Key] = kv.Value;
                }

            // Copy all scope values (scopes override globals)
            foreach (var scope in _scopes.Reverse())
                {
                foreach (var kv in scope)
                {
                    captured[kv.Key] = kv.Value;
                }
                }

            return captured;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double ConvertToNumber(object? value)
        {
            return value switch
                {
                double d => d,
                int i => (double)i,
                long l => (double)l,
                string s when double.TryParse(s, out var result) => result,
                bool b => b ? 1.0 : 0.0,
                null => 0.0,
                _ => throw new NSLRuntimeException($"Cannot convert {GetNSLTypeName(value)} to number")
                };
        }

        /// <summary>
        /// Get GitHub token from NSL Vault (3-tier encrypted storage)
        /// Returns null if not enabled or not configured
        /// </summary>
        private static string? GetGitHubToken()
        {
            try
            {
                var configDir = System.IO.Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    ".nsl", "github"
                );
                var configFile = System.IO.Path.Combine(configDir, "config.json");
                var vaultFile = System.IO.Path.Combine(configDir, ".vault.nsl");

                // Check if enabled
                if (!System.IO.File.Exists(configFile)) return null;
                var configJson = System.IO.File.ReadAllText(configFile);
                if (!configJson.Contains("\"Enabled\": true") && !configJson.Contains("\"Enabled\":true")) return null;

                // Check if vault exists
                if (!System.IO.File.Exists(vaultFile)) return null;

                var vault = System.IO.File.ReadAllBytes(vaultFile);
                if (vault.Length < 60) return null;

                // Extract salt and encrypted data
                var salt = new byte[32];
                var encrypted = new byte[vault.Length - 32];
                Buffer.BlockCopy(vault, 0, salt, 0, 32);
                Buffer.BlockCopy(vault, 32, encrypted, 0, encrypted.Length);

                // Tier 2: Derive key using Consciousness Hash
                var consciousnessOps = new[] { "|>", "~>", "=>>", "*>", "+>" };
                var identity = new System.Text.StringBuilder();
                identity.Append(Environment.MachineName);
                identity.Append(consciousnessOps[0]);
                identity.Append(Environment.UserName);
                identity.Append(consciousnessOps[1]);
                identity.Append(Environment.OSVersion.Platform);
                identity.Append(consciousnessOps[2]);
                identity.Append("NSL.Vault.Consciousness");
                identity.Append(consciousnessOps[3]);
                identity.Append(Environment.ProcessorCount);
                identity.Append(consciousnessOps[4]);

                var password = System.Text.Encoding.UTF8.GetBytes(identity.ToString());
                using var pbkdf2 = new System.Security.Cryptography.Rfc2898DeriveBytes(
                    password, salt, 150000, System.Security.Cryptography.HashAlgorithmName.SHA512);
                var key = pbkdf2.GetBytes(32);

                // Tier 3: AES-256-GCM decryption
                if (encrypted.Length < 28) return null;
                var nonce = new byte[12];
                var tag = new byte[16];
                var ciphertext = new byte[encrypted.Length - 28];

                Buffer.BlockCopy(encrypted, 0, nonce, 0, 12);
                Buffer.BlockCopy(encrypted, 12, tag, 0, 16);
                Buffer.BlockCopy(encrypted, 28, ciphertext, 0, ciphertext.Length);

                var plaintext = new byte[ciphertext.Length];
                using var aes = new System.Security.Cryptography.AesGcm(key, 16);
                aes.Decrypt(nonce, ciphertext, tag, plaintext);

                var transformed = System.Text.Encoding.UTF8.GetString(plaintext);

                // Tier 1: Reverse NSL semantic transformation
                var symbolMap = new Dictionary<char, char> {
                    ['⨀'] = 'g', ['⨁'] = 'h', ['⨂'] = 'p', ['⨃'] = '_',
                    ['∀'] = 'a', ['∃'] = 'b', ['∈'] = 'c', ['∋'] = 'd',
                    ['⊂'] = 'e', ['⊃'] = 'f', ['∧'] = 'i', ['∨'] = 'j',
                    ['⊕'] = 'k', ['⊗'] = 'l', ['⊙'] = 'm', ['⊛'] = 'n',
                    ['∘'] = 'o', ['∙'] = 'q', ['∴'] = 'r', ['∵'] = 's',
                    ['∝'] = 't', ['∞'] = 'u', ['∠'] = 'v', ['∡'] = 'w',
                    ['∢'] = 'x', ['∣'] = 'y', ['∤'] = 'z', ['⓪'] = '0',
                    ['①'] = '1', ['②'] = '2', ['③'] = '3', ['④'] = '4',
                    ['⑤'] = '5', ['⑥'] = '6', ['⑦'] = '7', ['⑧'] = '8',
                    ['⑨'] = '9', ['⒜'] = 'A', ['⒝'] = 'B', ['⒞'] = 'C',
                    ['⒟'] = 'D', ['⒠'] = 'E', ['⒡'] = 'F', ['⒢'] = 'G',
                    ['⒣'] = 'H', ['⒤'] = 'I', ['⒥'] = 'J', ['⒦'] = 'K',
                    ['⒧'] = 'L', ['⒨'] = 'M', ['⒩'] = 'N', ['⒪'] = 'O',
                    ['⒫'] = 'P', ['⒬'] = 'Q', ['⒭'] = 'R', ['⒮'] = 'S',
                    ['⒯'] = 'T', ['⒰'] = 'U', ['⒱'] = 'V', ['⒲'] = 'W',
                    ['⒳'] = 'X', ['⒴'] = 'Y', ['⒵'] = 'Z',
                };

                var sb = new System.Text.StringBuilder(transformed.Length);
                foreach (var c in transformed)
                    sb.Append(symbolMap.TryGetValue(c, out var original) ? original : c);

                return sb.ToString();
            }
            catch
            {
                return null;
            }
        }

        private static string DetectLanguage(string ext)
        {
            return ext switch
            {
                ".cs" => "csharp",
                ".java" => "java",
                ".py" => "python",
                ".js" => "javascript",
                ".ts" => "typescript",
                ".go" => "go",
                ".rs" => "rust",
                ".cpp" or ".cc" or ".cxx" => "cpp",
                ".c" or ".h" => "c",
                ".rb" => "ruby",
                ".php" => "php",
                ".swift" => "swift",
                ".kt" => "kotlin",
                ".scala" => "scala",
                ".nsl" => "nsl",
                ".json" => "json",
                ".xml" => "xml",
                ".yaml" or ".yml" => "yaml",
                ".html" or ".htm" => "html",
                ".css" => "css",
                ".sql" => "sql",
                ".sh" or ".bash" => "shell",
                ".ps1" => "powershell",
                ".md" => "markdown",
                _ => "unknown"
            };
        }

        private static int GetLineNumber(string content, int charIndex)
        {
            int line = 1;
            for (int i = 0; i < charIndex && i < content.Length; i++)
            {
                if (content[i] == '\n') line++;
            }
            return line;
        }

        #endregion
    }

    #region Supporting Classes

    /// <summary>
    /// Built-in function representation
    /// </summary>
    public class NSLBuiltinFunction
    {
        /// <summary>Public API</summary>
        public string Name { get; }
        /// <summary>Public API</summary>
        public Func<object?[], object?> Implementation { get; }

        /// <summary>Public API</summary>
        public NSLBuiltinFunction(string name, Func<object?[], object?> implementation)
        {
            Name = name;
            Implementation = implementation;
        }

        /// <summary>Public API</summary>
        public object? Call(object?[] args) => Implementation(args);

        /// <summary>Public API</summary>
        public override string ToString() => $"<builtin function {Name}>";
    }

    /// <summary>
    /// User-defined function representation with proper closure support
    /// </summary>
    public class NSLFunction
    {
        /// <summary>Function name</summary>
        public string Name { get; }
        /// <summary>Parameter names</summary>
        public List<string> Parameters { get; }
        /// <summary>Function body AST node</summary>
        public NSLASTNode Body { get; }
        /// <summary>
        /// Captured scope chain - a list of dictionaries representing the enclosing scopes.
        /// These are references to the actual scopes, enabling mutable closures.
        /// </summary>
        public List<Dictionary<string, object?>> CapturedScopes { get; }

        /// <summary>Create anonymous function</summary>
        public NSLFunction(List<string> parameters, NSLASTNode body, List<Dictionary<string, object?>> capturedScopes)
            : this("anonymous", parameters, body, capturedScopes) { }

        /// <summary>Create named function with closure support</summary>
        public NSLFunction(string name, List<string> parameters, NSLASTNode body, List<Dictionary<string, object?>> capturedScopes)
        {
            Name = name;
            Parameters = parameters;
            Body = body;
            // Store references to the captured scopes (not copies!)
            CapturedScopes = capturedScopes;
        }

        /// <summary>
        /// Look up a variable in the captured scopes
        /// </summary>
        public bool TryGetCapturedVariable(string name, out object? value)
        {
            foreach (var scope in CapturedScopes)
                {
                if (scope.TryGetValue(name, out value))
                {
                    return true;
                }
                }
            value = null;
            return false;
        }

        /// <summary>
        /// Set a variable in the captured scopes (if it exists)
        /// </summary>
        public bool TrySetCapturedVariable(string name, object? value)
        {
            foreach (var scope in CapturedScopes)
                {
                if (scope.ContainsKey(name))
                {
                    scope[name] = value;
                    return true;
                }
                }
            return false;
        }

        /// <summary>String representation</summary>
        public override string ToString() => $"<function {Name}({string.Join(", ", Parameters)})>";
    }

    /// <summary>
    /// Async function representation with proper closure support
    /// </summary>
    public class NSLAsyncFunction
    {
        /// <summary>Function name</summary>
        public string Name { get; }
        /// <summary>Parameter names</summary>
        public List<string> Parameters { get; }
        /// <summary>Function body AST node</summary>
        public NSLASTNode Body { get; }
        /// <summary>Captured closure scopes</summary>
        public List<Dictionary<string, object?>> CapturedScopes { get; }

        /// <summary>Create async function with closure support</summary>
        public NSLAsyncFunction(string name, List<string> parameters, NSLASTNode body, List<Dictionary<string, object?>> capturedScopes)
        {
            Name = name;
            Parameters = parameters;
            Body = body;
            CapturedScopes = capturedScopes;
        }

        /// <summary>String representation</summary>
        public override string ToString() => $"<async function {Name}({string.Join(", ", Parameters)})>";
    }

    /// <summary>
    /// Result type for error handling (ok/err)
    /// </summary>
    public class NSLResult
    {
        /// <summary>Whether result is success</summary>
        public bool IsOk { get; }
        /// <summary>Result value or error</summary>
        public object? Value { get; }

        /// <summary>Create result</summary>
        public NSLResult(bool isOk, object? value)
        {
            IsOk = isOk;
            Value = value;
        }

        /// <summary>String representation</summary>
        public override string ToString() => IsOk ? $"ok({Value})" : $"err({Value})";
    }

    /// <summary>
    /// Optional type for null handling (some/none)
    /// </summary>
    public class NSLOptional
    {
        /// <summary>Whether optional has value</summary>
        public bool HasValue { get; }
        /// <summary>Optional value</summary>
        public object? Value { get; }

        /// <summary>Create optional</summary>
        public NSLOptional(object? value, bool hasValue)
        {
            Value = value;
            HasValue = hasValue;
        }

        /// <summary>String representation</summary>
        public override string ToString() => HasValue ? $"some({Value})" : "none";
    }

    /// <summary>
    /// Enum variant instance
    /// </summary>
    public class NSLEnumValue
    {
        /// <summary>Enum type name</summary>
        public string EnumName { get; }
        /// <summary>Variant name</summary>
        public string VariantName { get; }
        /// <summary>Variant arguments</summary>
        public List<object?> Arguments { get; }

        /// <summary>Create enum value</summary>
        public NSLEnumValue(string enumName, string variantName, List<object?> arguments)
        {
            EnumName = enumName;
            VariantName = variantName;
            Arguments = arguments;
        }

        /// <summary>String representation</summary>
        public override string ToString()
        {
            if (Arguments.Count == 0)
                return $"{EnumName}::{VariantName}";
            return $"{EnumName}::{VariantName}({string.Join(", ", Arguments)})";
        }
    }

    /// <summary>
    /// Consciousness data wrapper
    /// </summary>
    public class NSLConsciousnessData
    {
        /// <summary>Consciousness type</summary>
        public string Type { get; set; } = "";
        /// <summary>Associated data</summary>
        public object? Data { get; set; }
        /// <summary>Processing timestamp</summary>
        public DateTime ProcessingTime { get; set; }
        /// <summary>Consciousness level 0-1</summary>
        public double ConsciousnessLevel { get; set; }

        /// <summary>String representation</summary>
        public override string ToString() => $"<consciousness:{Type} level={ConsciousnessLevel:F2}>";
    }

    /// <summary>
    /// Function parameter representation
    /// </summary>
    public class NSLParameter
    {
        /// <summary>Parameter name</summary>
        public string Name { get; }
        /// <summary>Optional type annotation</summary>
        public string? Type { get; }

        /// <summary>Create parameter</summary>
        public NSLParameter(string name, string? type = null)
        {
            Name = name;
            Type = type;
        }
    }

    #endregion

    // Exception classes are now defined in NSL.Core
}