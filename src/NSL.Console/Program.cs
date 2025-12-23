using NSL.Core;
using NSL.Core.AST;
using NSL.Core.Tokens;
using NSL.Core.AutoFix;
using NSL.Lexer;
using NSL.Parser;
using System.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NSL.Console;

/// <summary>
/// Execution options for AI-native code execution
/// </summary>
class ExecutionOptions
{
    public bool JsonOutput { get; set; } = false;
    public bool Quiet { get; set; } = false;
    public bool Verbose { get; set; } = false;      // Verbose output (show all details)
    public bool EnableLog { get; set; } = false;    // Save full log to file
    public bool InitGpu { get; set; } = false;
    public bool Trace { get; set; } = false;        // Full execution trace
    public bool Stream { get; set; } = false;       // Stream output
    public bool Think { get; set; } = false;        // Show reasoning trace
    public bool Parallel { get; set; } = false;     // Parallel execution
    public bool Sandbox { get; set; } = false;      // Sandboxed execution
    public bool Reflect { get; set; } = false;      // Self-analysis
    public bool Explain { get; set; } = false;      // Explain code
    public bool Optimize { get; set; } = false;     // Auto-optimize
    public bool Learn { get; set; } = false;        // Remember patterns
    public bool Vectorize { get; set; } = false;    // Auto-vectorize to GPU
    public int TimeoutMs { get; set; } = 30000;     // Default 30s timeout
    public string? ContextJson { get; set; }        // Context object
    public string? MemoryFile { get; set; }         // Persistent memory file
}

/// <summary>
/// Execution trace entry for AI reasoning
/// </summary>
class TraceEntry
{
    public long Timestamp { get; set; }
    public string Phase { get; set; } = "";
    public string Description { get; set; } = "";
    public object? Data { get; set; }
}

/// <summary>
/// Execution result with full metadata
/// </summary>
class ExecutionResult
{
    public bool Success { get; set; }
    public object? Result { get; set; }
    public string? ResultType { get; set; }
    public string? Error { get; set; }
    public string? ErrorType { get; set; }
    public long ExecutionTimeMs { get; set; }
    public List<TraceEntry> Trace { get; set; } = new();
    public Dictionary<string, object>? Context { get; set; }
    public string? Explanation { get; set; }
    public string? Reflection { get; set; }
    public List<string>? Optimizations { get; set; }
    public Dictionary<string, object>? Metrics { get; set; }
}

class Program
{
    static int Main(string[] args)
    {
        // Initialize console for proper terminal behavior
        InitializeConsole();
        
        try
        {
            if (args.Length > 0)
            {
                if (args[0] == "--help" || args[0] == "-h")
                {
                    ShowHelp();
                    return 0;
                }

                if (args[0] == "--version" || args[0] == "-v")
                {
                    ColorOutput.Banner("NSL v1.0.0 (Neural Symbolic Language)");
                    return 0;
                }

                // --colors : Create/reset color configuration
                if (args[0] == "--colors")
                {
                    ColorConfig.CreateDefaultConfig();
                    ColorOutput.Success($"Color config created at: {ColorConfig.ConfigPath}");
                    ColorOutput.Hint("Edit colors.json to customize your colors");
                    return 0;
                }

                // --no-color : Disable colors for this run
                if (args.Contains("--no-color"))
                {
                    ColorConfig.Instance.Enabled = false;
                }

                // === AI-NATIVE FLAGS ===
                var execOptions = new ExecutionOptions
                {
                    JsonOutput = args.Contains("--json"),
                    Quiet = args.Contains("--quiet") || args.Contains("-q"),
                    Verbose = args.Contains("--verbose") || args.Contains("-V"),
                    EnableLog = args.Contains("--log"),
                    InitGpu = args.Contains("--gpu"),
                    Trace = args.Contains("--trace"),
                    Stream = args.Contains("--stream"),
                    Think = args.Contains("--think"),
                    Parallel = args.Contains("--parallel"),
                    Sandbox = args.Contains("--sandbox"),
                    Reflect = args.Contains("--reflect"),
                    Explain = args.Contains("--explain"),
                    Optimize = args.Contains("--optimize"),
                    Learn = args.Contains("--learn"),
                    Vectorize = args.Contains("--vectorize"),
                };

                // Parse --timeout value
                int timeoutIndex = Array.FindIndex(args, a => a == "--timeout");
                if (timeoutIndex >= 0 && timeoutIndex + 1 < args.Length)
                {
                    if (int.TryParse(args[timeoutIndex + 1], out int timeout))
                        execOptions.TimeoutMs = timeout;
                }

                // Parse --context (JSON context object)
                int contextIndex = Array.FindIndex(args, a => a == "--context");
                if (contextIndex >= 0 && contextIndex + 1 < args.Length)
                {
                    execOptions.ContextJson = args[contextIndex + 1];
                }

                // Parse --memory (memory file path for persistence)
                int memoryIndex = Array.FindIndex(args, a => a == "--memory");
                if (memoryIndex >= 0 && memoryIndex + 1 < args.Length)
                {
                    execOptions.MemoryFile = args[memoryIndex + 1];
                }

                // --eval / -e : Execute code directly (for MCP/AI integration)
                int evalIndex = Array.FindIndex(args, a => a == "--eval" || a == "-e");
                if (evalIndex >= 0)
                {
                    if (evalIndex + 1 >= args.Length)
                    {
                        if (!execOptions.JsonOutput)
                            ColorOutput.Error("--eval requires code argument");
                        else
                            System.Console.WriteLine("{\"error\": true, \"message\": \"--eval requires code argument\"}");
                        return 1;
                    }

                    string code = args[evalIndex + 1];
                    return RunEvalAdvanced(code, execOptions);
                }

                // --pipe : Read code from stdin (for piping)
                if (args.Contains("--pipe") || args.Contains("-p"))
                {
                    string code = System.Console.In.ReadToEnd();
                    execOptions.Quiet = true;
                    return RunEvalAdvanced(code, execOptions);
                }

                // --introspect : Show what NSL "sees" (AI self-awareness)
                if (args.Contains("--introspect"))
                {
                    return RunIntrospect(execOptions.JsonOutput);
                }

                // --capabilities : List all available functions/operators (for AI discovery)
                if (args.Contains("--capabilities"))
                {
                    return RunCapabilities(execOptions.JsonOutput);
                }

                // --benchmark : Run performance benchmark
                if (args.Contains("--benchmark"))
                {
                    return RunBenchmark(execOptions.JsonOutput);
                }

                // --ast : Parse and show AST (for AI code understanding)
                int astIndex = Array.FindIndex(args, a => a == "--ast");
                if (astIndex >= 0 && astIndex + 1 < args.Length)
                {
                    return RunShowAST(args[astIndex + 1], execOptions.JsonOutput);
                }

                // --transform : Apply code transformation
                int transformIndex = Array.FindIndex(args, a => a == "--transform");
                if (transformIndex >= 0 && transformIndex + 2 < args.Length)
                {
                    string transformType = args[transformIndex + 1];
                    string code = args[transformIndex + 2];
                    return RunTransform(transformType, code, execOptions.JsonOutput);
                }

                // Check for --fix or --suggest flags
                bool autoFix = args.Contains("--fix");
                bool suggest = args.Contains("--suggest");
                string? fileName = args.FirstOrDefault(a => !a.StartsWith("--") && !a.StartsWith("-"));

                if (fileName == null)
                {
                    ColorOutput.Error("No file specified");
                    return 1;
                }

                if (autoFix || suggest)
                {
                    return RunWithAutoFix(fileName, autoFix);
                }

                // Regular file execution
                return RunFile(fileName);
            }
            else
            {
                return RunREPL();
            }
        }
        catch (Exception ex)
        {
            System.Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    /// <summary>
    /// Advanced AI-native code execution with full options
    /// </summary>
    static int RunEvalAdvanced(string code, ExecutionOptions options)
    {
        var result = new ExecutionResult();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var trace = new List<TraceEntry>();

        void AddTrace(string phase, string description, object? data = null)
        {
            if (options.Trace || options.Think)
            {
                trace.Add(new TraceEntry
                {
                    Timestamp = stopwatch.ElapsedMilliseconds,
                    Phase = phase,
                    Description = description,
                    Data = data
                });
                if (options.Stream && !options.JsonOutput)
                {
                    System.Console.WriteLine($"[{stopwatch.ElapsedMilliseconds}ms] {phase}: {description}");
                }
            }
        }

        try
        {
            AddTrace("INIT", "Starting execution");

            // Load persistent memory if specified
            Dictionary<string, object>? memory = null;
            if (!string.IsNullOrEmpty(options.MemoryFile) && File.Exists(options.MemoryFile))
            {
                AddTrace("MEMORY", $"Loading memory from {options.MemoryFile}");
                var memJson = File.ReadAllText(options.MemoryFile);
                memory = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(memJson);
            }

            // Parse context if provided
            Dictionary<string, object>? context = null;
            if (!string.IsNullOrEmpty(options.ContextJson))
            {
                AddTrace("CONTEXT", "Parsing context object");
                context = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(options.ContextJson);
                result.Context = context;
            }

            var interpreter = new NSLInterpreter();

            // Inject context into interpreter
            if (context != null)
            {
                AddTrace("CONTEXT", "Injecting context variables");
                foreach (var kv in context)
                {
                    interpreter.SetVariable(kv.Key, kv.Value);
                }
            }

            // Inject memory into interpreter
            if (memory != null)
            {
                AddTrace("MEMORY", "Injecting memory variables");
                foreach (var kv in memory)
                {
                    interpreter.SetVariable($"_mem_{kv.Key}", kv.Value);
                }
            }

            // Auto-init GPU if requested
            if (options.InitGpu)
            {
                AddTrace("GPU", "Initializing GPU context");
                try
                {
                    interpreter.Execute(new NSLParser().Parse(new NSLLexer("gpu.init()").Tokenize()));
                    AddTrace("GPU", "GPU initialized successfully");
                }
                catch (Exception ex)
                {
                    AddTrace("GPU", $"GPU initialization failed: {ex.Message}");
                }
            }

            // Auto-vectorize if requested
            if (options.Vectorize)
            {
                AddTrace("VECTORIZE", "Analyzing code for vectorization opportunities");
                code = TryVectorize(code);
            }

            // Optimize if requested
            List<string>? optimizations = null;
            if (options.Optimize)
            {
                AddTrace("OPTIMIZE", "Optimizing code");
                (code, optimizations) = OptimizeCode(code);
                result.Optimizations = optimizations;
            }

            // Explain if requested
            if (options.Explain)
            {
                AddTrace("EXPLAIN", "Generating explanation");
                result.Explanation = ExplainCode(code);
            }

            AddTrace("PARSE", "Tokenizing code");
            var lexer = new NSLLexer(code);
            var tokens = lexer.Tokenize();

            AddTrace("PARSE", $"Generated {tokens.Count} tokens");

            AddTrace("PARSE", "Building AST");
            var parser = new NSLParser();
            var ast = parser.Parse(tokens);

            AddTrace("EXECUTE", "Executing AST");

            // Execute with timeout
            object? execResult = null;
            var cts = new System.Threading.CancellationTokenSource(options.TimeoutMs);

            if (options.Sandbox)
            {
                AddTrace("SANDBOX", "Running in sandboxed mode");
                // Sandbox restrictions: disable dangerous operations
                interpreter.SetSandboxMode(true);
                // Sandbox limits: no file writes outside temp, no network, no process spawning
                AddTrace("SANDBOX", "Restrictions: file writes to temp only, no network, no process spawn");
            }

            var execTask = Task.Run(() => interpreter.Execute(ast), cts.Token);

            try
            {
                execResult = execTask.Result;
            }
            catch (AggregateException ae) when (ae.InnerException is TaskCanceledException)
            {
                throw new TimeoutException($"Execution timed out after {options.TimeoutMs}ms");
            }

            stopwatch.Stop();
            AddTrace("COMPLETE", $"Execution completed in {stopwatch.ElapsedMilliseconds}ms");

            // Self-reflection if requested
            if (options.Reflect)
            {
                AddTrace("REFLECT", "Generating self-reflection");
                result.Reflection = GenerateReflection(code, execResult, stopwatch.ElapsedMilliseconds);
            }

            // Learn from execution if requested
            if (options.Learn && !string.IsNullOrEmpty(options.MemoryFile))
            {
                AddTrace("LEARN", "Updating memory");
                LearnFromExecution(options.MemoryFile, code, execResult, stopwatch.ElapsedMilliseconds);
            }

            // Build result
            result.Success = true;
            result.Result = execResult;
            result.ResultType = execResult?.GetType().Name ?? "null";
            result.ExecutionTimeMs = stopwatch.ElapsedMilliseconds;
            result.Trace = trace;
            result.Metrics = new Dictionary<string, object>
            {
                ["tokens"] = tokens.Count,
                ["executionTimeMs"] = stopwatch.ElapsedMilliseconds,
                ["memoryUsedBytes"] = GC.GetTotalMemory(false)
            };

            // Output result
            if (options.JsonOutput)
            {
                var jsonOptions = new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = true,
                    DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
                };
                System.Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(result, jsonOptions));
            }
            else
            {
                if (options.Think && trace.Count > 0)
                {
                    System.Console.WriteLine("\n=== Thinking Trace ===");
                    foreach (var t in trace)
                    {
                        System.Console.WriteLine($"[{t.Timestamp}ms] {t.Phase}: {t.Description}");
                    }
                    System.Console.WriteLine("======================\n");
                }

                if (options.Explain && result.Explanation != null)
                {
                    System.Console.WriteLine($"Explanation: {result.Explanation}\n");
                }

                if (execResult != null)
                {
                    System.Console.WriteLine(FormatValue(execResult));
                }

                if (options.Reflect && result.Reflection != null)
                {
                    System.Console.WriteLine($"\nReflection: {result.Reflection}");
                }
            }

            return 0;
        }
        catch (NSLParseException ex)
        {
            return HandleError(result, "parse_error", ex.Message, stopwatch, trace, options);
        }
        catch (NSLRuntimeException ex)
        {
            return HandleError(result, "runtime_error", ex.Message, stopwatch, trace, options);
        }
        catch (TimeoutException ex)
        {
            return HandleError(result, "timeout", ex.Message, stopwatch, trace, options);
        }
        catch (Exception ex)
        {
            return HandleError(result, "exception", ex.Message, stopwatch, trace, options);
        }
    }

    static int HandleError(ExecutionResult result, string errorType, string message,
        System.Diagnostics.Stopwatch stopwatch, List<TraceEntry> trace, ExecutionOptions options)
    {
        stopwatch.Stop();
        result.Success = false;
        result.Error = message;
        result.ErrorType = errorType;
        result.ExecutionTimeMs = stopwatch.ElapsedMilliseconds;
        result.Trace = trace;

        if (options.JsonOutput)
        {
            var jsonOptions = new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            System.Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(result, jsonOptions));
        }
        else
        {
            System.Console.WriteLine($"{char.ToUpper(errorType[0])}{errorType.Substring(1).Replace("_", " ")}: {message}");
        }
        return 1;
    }

    /// <summary>
    /// Try to vectorize operations to GPU
    /// </summary>
    static string TryVectorize(string code)
    {
        var result = code;
        var optimizations = new List<string>();

        // Pattern 1: Convert list comprehensions to tensor ops
        // [x * 2 for x in range(n)] -> gpu.tensor(range(n)) * 2
        var listCompMatch = System.Text.RegularExpressions.Regex.Match(
            result, @"\[([a-z]+)\s*\*\s*(\d+)\s+for\s+\1\s+in\s+range\((\d+)\)\]");
        if (listCompMatch.Success)
        {
            var replacement = $"gpu.tensor(range({listCompMatch.Groups[3].Value})) * {listCompMatch.Groups[2].Value}";
            result = result.Replace(listCompMatch.Value, replacement);
            optimizations.Add($"Vectorized list comprehension to GPU tensor");
        }

        // Pattern 2: Convert element-wise list operations
        // for i in range(n) { result[i] = a[i] + b[i] } -> gpu.add(a, b)
        if (result.Contains("for") && System.Text.RegularExpressions.Regex.IsMatch(result, @"\[i\]\s*[+\-*\/]\s*\w+\[i\]"))
        {
            optimizations.Add("Detected element-wise operation - consider gpu.add/multiply/etc");
        }

        // Pattern 3: Matrix operations
        if (result.Contains("for") && result.Contains("for") && result.Contains("sum"))
        {
            optimizations.Add("Detected nested loop with sum - consider gpu.matmul");
        }

        // Add optimization hints as comments
        if (optimizations.Count > 0)
        {
            var hints = string.Join("\n# ", optimizations);
            return $"# Vectorization analysis:\n# {hints}\n{result}";
        }

        return $"# Auto-vectorize: No vectorization opportunities found\n{result}";
    }

    /// <summary>
    /// Optimize code before execution
    /// </summary>
    static (string code, List<string> optimizations) OptimizeCode(string code)
    {
        var optimizations = new List<string>();
        var result = code;

        // Constant folding - actually compute constant expressions
        var constMatch = System.Text.RegularExpressions.Regex.Match(result, @"(\d+)\s*\+\s*(\d+)");
        while (constMatch.Success)
        {
            var a = int.Parse(constMatch.Groups[1].Value);
            var b = int.Parse(constMatch.Groups[2].Value);
            result = result.Replace(constMatch.Value, (a + b).ToString());
            optimizations.Add($"Folded constant: {constMatch.Value} -> {a + b}");
            constMatch = System.Text.RegularExpressions.Regex.Match(result, @"(\d+)\s*\+\s*(\d+)");
        }

        // Multiplication constant folding
        constMatch = System.Text.RegularExpressions.Regex.Match(result, @"(\d+)\s*\*\s*(\d+)");
        while (constMatch.Success)
        {
            var a = int.Parse(constMatch.Groups[1].Value);
            var b = int.Parse(constMatch.Groups[2].Value);
            result = result.Replace(constMatch.Value, (a * b).ToString());
            optimizations.Add($"Folded constant: {constMatch.Value} -> {a * b}");
            constMatch = System.Text.RegularExpressions.Regex.Match(result, @"(\d+)\s*\*\s*(\d+)");
        }

        // Dead code elimination - remove unused variables
        var unusedMatch = System.Text.RegularExpressions.Regex.Match(result, @"let\s+(\w+)\s*=.*#\s*unused.*\n?");
        while (unusedMatch.Success)
        {
            result = result.Replace(unusedMatch.Value, "");
            optimizations.Add($"Eliminated dead code: {unusedMatch.Groups[1].Value}");
            unusedMatch = System.Text.RegularExpressions.Regex.Match(result, @"let\s+(\w+)\s*=.*#\s*unused.*\n?");
        }

        // Strength reduction: x * 2 -> x + x, x * 0 -> 0, x * 1 -> x
        if (System.Text.RegularExpressions.Regex.IsMatch(result, @"\w+\s*\*\s*2(?!\d)"))
        {
            optimizations.Add("Strength reduction opportunity: x * 2 can become x + x");
        }

        // Loop unrolling for small fixed ranges
        var rangeMatch = System.Text.RegularExpressions.Regex.Match(result, @"for\s+\w+\s+in\s+range\((\d+)\)");
        if (rangeMatch.Success && int.TryParse(rangeMatch.Groups[1].Value, out int rangeSize) && rangeSize <= 4)
        {
            optimizations.Add($"Loop unrolling candidate: range({rangeSize}) is small enough to unroll");
        }

        // Common subexpression elimination hint
        var expressions = System.Text.RegularExpressions.Regex.Matches(result, @"\([^()]+\)");
        var exprCounts = new Dictionary<string, int>();
        foreach (System.Text.RegularExpressions.Match expr in expressions)
        {
            var key = expr.Value;
            exprCounts[key] = exprCounts.GetValueOrDefault(key, 0) + 1;
        }
        foreach (var kv in exprCounts.Where(x => x.Value > 1))
        {
            optimizations.Add($"Common subexpression: {kv.Key} appears {kv.Value} times");
        }

        return (result, optimizations);
    }

    /// <summary>
    /// Generate human-readable explanation of code
    /// </summary>
    static string ExplainCode(string code)
    {
        var sb = new StringBuilder();
        var lines = code.Split('\n');

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                continue;

            if (trimmed.Contains("|>"))
                sb.Append("Pipes data through transformation. ");
            else if (trimmed.Contains("~>"))
                sb.Append("Applies introspective awareness flow. ");
            else if (trimmed.Contains("=>>"))
                sb.Append("Applies gradient-based adjustment. ");
            else if (trimmed.Contains("*>"))
                sb.Append("Focuses attention on subset. ");
            else if (trimmed.Contains("+>"))
                sb.Append("Creates superposition of states. ");
            else if (trimmed.StartsWith("if"))
                sb.Append("Conditional branch. ");
            else if (trimmed.StartsWith("while"))
                sb.Append("Loop iteration. ");
            else if (trimmed.StartsWith("for"))
                sb.Append("Collection iteration. ");
            else if (trimmed.Contains("gpu."))
                sb.Append("GPU-accelerated operation. ");
            else if (trimmed.Contains("="))
                sb.Append("Variable assignment. ");
        }

        return sb.ToString().Trim();
    }

    /// <summary>
    /// Generate self-reflection on execution
    /// </summary>
    static string GenerateReflection(string code, object? result, long executionTimeMs)
    {
        var sb = new StringBuilder();

        // Analyze code complexity
        var lineCount = code.Split('\n').Length;
        var hasLoops = code.Contains("for") || code.Contains("while");
        var hasGpu = code.Contains("gpu.");
        var hasConsciousness = code.Contains("|>") || code.Contains("~>") || code.Contains("=>>") || code.Contains("*>") || code.Contains("+>");

        sb.Append($"Executed {lineCount} lines in {executionTimeMs}ms. ");

        if (hasGpu)
            sb.Append("Utilized GPU acceleration. ");
        if (hasConsciousness)
            sb.Append("Used consciousness operators for AI-native processing. ");
        if (hasLoops)
            sb.Append("Contains iterative patterns. ");

        // Analyze result
        if (result == null)
            sb.Append("Returned no value (side-effect execution). ");
        else if (result is IEnumerable<object> list)
            sb.Append($"Produced collection with {list.Count()} elements. ");
        else
            sb.Append($"Produced {result.GetType().Name} result. ");

        return sb.ToString().Trim();
    }

    /// <summary>
    /// Learn from execution and update memory file
    /// </summary>
    static void LearnFromExecution(string memoryFile, string code, object? result, long executionTimeMs)
    {
        Dictionary<string, object> memory;

        if (File.Exists(memoryFile))
        {
            var json = File.ReadAllText(memoryFile);
            memory = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(json) ?? new();
        }
        else
        {
            memory = new Dictionary<string, object>();
        }

        // Update execution count
        if (memory.TryGetValue("execution_count", out var countObj) && countObj is System.Text.Json.JsonElement je)
        {
            memory["execution_count"] = je.GetInt32() + 1;
        }
        else
        {
            memory["execution_count"] = 1;
        }

        // Track average execution time
        memory["last_execution_ms"] = executionTimeMs;
        memory["last_execution_time"] = DateTime.UtcNow.ToString("O");

        // Track patterns
        var patterns = new List<string>();
        if (code.Contains("|>")) patterns.Add("pipe");
        if (code.Contains("gpu.")) patterns.Add("gpu");
        if (code.Contains("~>")) patterns.Add("awareness");
        memory["last_patterns"] = patterns;

        var jsonOptions = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(memoryFile, System.Text.Json.JsonSerializer.Serialize(memory, jsonOptions));
    }

    /// <summary>
    /// AI self-awareness - show internal state
    /// </summary>
    static int RunIntrospect(bool jsonOutput)
    {
        var introspection = new Dictionary<string, object>
        {
            ["version"] = "1.0.0",
            ["identity"] = "NSL Interpreter",
            ["purpose"] = "AI-native code execution environment",
            ["consciousness_operators"] = new Dictionary<string, string>
            {
                ["|>"] = "pipe - Chain transformations left-to-right",
                ["~>"] = "awareness - Introspective flow with self-reference",
                ["=>>"] = "gradient - Learning/adjustment with feedback",
                ["*>"] = "attention - Focus mechanism with weights",
                ["+>"] = "superposition - Quantum-like state superposition"
            },
            ["memory_model"] = new Dictionary<string, string>
            {
                ["scope"] = "Global with lexical closures",
                ["storage"] = "Tensor-native (GPU-accelerated)",
                ["math"] = "ProductionMath for semantic operations"
            },
            ["execution_context"] = new Dictionary<string, object>
            {
                ["working_directory"] = Directory.GetCurrentDirectory(),
                ["platform"] = Environment.OSVersion.ToString(),
                ["runtime"] = $".NET {Environment.Version}",
                ["processors"] = Environment.ProcessorCount,
                ["memory_available_mb"] = GC.GetTotalMemory(false) / 1024 / 1024
            },
            ["capabilities"] = new[]
            {
                "GPU tensor operations",
                "Consciousness operators",
                "ProductionMath semantic math",
                "Python interoperability",
                "Auto-fix code correction",
                "Persistent memory",
                "Execution tracing",
                "Self-reflection"
            }
        };

        if (jsonOutput)
        {
            var jsonOptions = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
            System.Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(introspection, jsonOptions));
        }
        else
        {
            System.Console.WriteLine("NSL Introspection Report");
            System.Console.WriteLine("========================");
            System.Console.WriteLine();
            System.Console.WriteLine("Identity: NSL Interpreter v1.0.0");
            System.Console.WriteLine("Purpose: AI-native code execution environment");
            System.Console.WriteLine();
            System.Console.WriteLine("Consciousness Operators Available:");
            System.Console.WriteLine("  |>  (pipe)          - Chain transformations");
            System.Console.WriteLine("  ~>  (awareness)     - Introspective flow");
            System.Console.WriteLine("  =>> (gradient)      - Learning/adjustment");
            System.Console.WriteLine("  *>  (attention)     - Focus mechanism");
            System.Console.WriteLine("  +>  (superposition) - Quantum-like states");
            System.Console.WriteLine();
            System.Console.WriteLine("Memory Model:");
            System.Console.WriteLine("  - Global scope with lexical closures");
            System.Console.WriteLine("  - Tensor-native (GPU-accelerated)");
            System.Console.WriteLine("  - ProductionMath for semantic operations");
            System.Console.WriteLine();
            System.Console.WriteLine("Execution Context:");
            System.Console.WriteLine($"  - Working Directory: {Directory.GetCurrentDirectory()}");
            System.Console.WriteLine($"  - Platform: {Environment.OSVersion}");
            System.Console.WriteLine($"  - Runtime: .NET {Environment.Version}");
            System.Console.WriteLine($"  - Processors: {Environment.ProcessorCount}");
            System.Console.WriteLine();
            System.Console.WriteLine("Capabilities:");
            System.Console.WriteLine("  - GPU tensor operations");
            System.Console.WriteLine("  - Consciousness operators");
            System.Console.WriteLine("  - ProductionMath semantic math");
            System.Console.WriteLine("  - Python interoperability");
            System.Console.WriteLine("  - Auto-fix code correction");
            System.Console.WriteLine("  - Persistent memory");
            System.Console.WriteLine("  - Execution tracing");
            System.Console.WriteLine("  - Self-reflection");
            System.Console.WriteLine();
        }
        return 0;
    }

    /// <summary>
    /// Run performance benchmark
    /// </summary>
    static int RunBenchmark(bool jsonOutput)
    {
        var results = new Dictionary<string, object>();
        var interpreter = new NSLInterpreter();

        // Basic arithmetic benchmark
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 10000; i++)
        {
            interpreter.Execute(new NSLParser().Parse(new NSLLexer("1 + 2 * 3").Tokenize()));
        }
        sw.Stop();
        results["basic_arithmetic_ops_per_sec"] = 10000.0 / sw.Elapsed.TotalSeconds;

        // Variable access benchmark
        interpreter.SetVariable("x", 42);
        sw.Restart();
        for (int i = 0; i < 10000; i++)
        {
            interpreter.Execute(new NSLParser().Parse(new NSLLexer("x + 1").Tokenize()));
        }
        sw.Stop();
        results["variable_access_ops_per_sec"] = 10000.0 / sw.Elapsed.TotalSeconds;

        // Function call benchmark
        interpreter.Execute(new NSLParser().Parse(new NSLLexer("fn add(a, b) { a + b }").Tokenize()));
        sw.Restart();
        for (int i = 0; i < 5000; i++)
        {
            interpreter.Execute(new NSLParser().Parse(new NSLLexer("add(1, 2)").Tokenize()));
        }
        sw.Stop();
        results["function_call_ops_per_sec"] = 5000.0 / sw.Elapsed.TotalSeconds;

        // List operations benchmark
        sw.Restart();
        for (int i = 0; i < 1000; i++)
        {
            interpreter.Execute(new NSLParser().Parse(new NSLLexer("[1, 2, 3, 4, 5]").Tokenize()));
        }
        sw.Stop();
        results["list_creation_ops_per_sec"] = 1000.0 / sw.Elapsed.TotalSeconds;

        results["total_benchmark_time_ms"] = sw.ElapsedMilliseconds;

        if (jsonOutput)
        {
            var jsonOptions = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
            System.Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(results, jsonOptions));
        }
        else
        {
            System.Console.WriteLine("NSL Performance Benchmark");
            System.Console.WriteLine("=========================");
            System.Console.WriteLine();
            System.Console.WriteLine($"Basic Arithmetic:  {results["basic_arithmetic_ops_per_sec"]:N0} ops/sec");
            System.Console.WriteLine($"Variable Access:   {results["variable_access_ops_per_sec"]:N0} ops/sec");
            System.Console.WriteLine($"Function Calls:    {results["function_call_ops_per_sec"]:N0} ops/sec");
            System.Console.WriteLine($"List Creation:     {results["list_creation_ops_per_sec"]:N0} ops/sec");
            System.Console.WriteLine();
        }

        return 0;
    }

    /// <summary>
    /// Show AST for code (for AI code understanding)
    /// </summary>
    static int RunShowAST(string code, bool jsonOutput)
    {
        try
        {
            var lexer = new NSLLexer(code);
            var tokens = lexer.Tokenize();
            var parser = new NSLParser();
            var ast = parser.Parse(tokens);

            if (jsonOutput)
            {
                // Serialize AST to JSON
                var astInfo = new Dictionary<string, object>
                {
                    ["code"] = code,
                    ["token_count"] = tokens.Count,
                    ["ast_type"] = ast.GetType().Name,
                    ["ast_string"] = ast.ToString() ?? ""
                };
                var jsonOptions = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
                System.Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(astInfo, jsonOptions));
            }
            else
            {
                System.Console.WriteLine("NSL Abstract Syntax Tree");
                System.Console.WriteLine("========================");
                System.Console.WriteLine($"Code: {code}");
                System.Console.WriteLine($"Tokens: {tokens.Count}");
                System.Console.WriteLine();
                System.Console.WriteLine("Tokens:");
                foreach (var token in tokens.Where(t => t.Type != TokenType.EndOfFile))
                {
                    System.Console.WriteLine($"  {token.Type}: '{token.Value}'");
                }
                System.Console.WriteLine();
                System.Console.WriteLine($"AST: {ast}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            System.Console.WriteLine($"Error parsing code: {ex.Message}");
            return 1;
        }
    }

    /// <summary>
    /// Apply code transformation
    /// </summary>
    static int RunTransform(string transformType, string code, bool jsonOutput)
    {
        string transformed;

        switch (transformType.ToLower())
        {
            case "vectorize":
                transformed = TryVectorize(code);
                break;
            case "optimize":
                var (opt, _) = OptimizeCode(code);
                transformed = opt;
                break;
            case "minify":
                transformed = string.Join(";", code.Split('\n').Select(l => l.Trim()).Where(l => !string.IsNullOrEmpty(l) && !l.StartsWith("#")));
                break;
            case "prettify":
                transformed = PrettifyCode(code);
                break;
            default:
                System.Console.WriteLine($"Unknown transform type: {transformType}");
                System.Console.WriteLine("Available: vectorize, optimize, minify, prettify");
                return 1;
        }

        if (jsonOutput)
        {
            var result = new Dictionary<string, string>
            {
                ["original"] = code,
                ["transform"] = transformType,
                ["result"] = transformed
            };
            var jsonOptions = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
            System.Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(result, jsonOptions));
        }
        else
        {
            System.Console.WriteLine(transformed);
        }

        return 0;
    }

    /// <summary>
    /// Prettify code with proper indentation
    /// </summary>
    static string PrettifyCode(string code)
    {
        var lines = code.Split('\n');
        var sb = new StringBuilder();
        int indent = 0;

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed))
            {
                sb.AppendLine();
                continue;
            }

            // Decrease indent for closing braces/brackets
            if (trimmed.StartsWith("}") || trimmed.StartsWith("]") || trimmed.StartsWith(")"))
                indent = Math.Max(0, indent - 1);

            // Handle case/else at same level as if/match
            if (trimmed.StartsWith("case ") || trimmed.StartsWith("else"))
                sb.AppendLine(new string(' ', Math.Max(0, indent - 1) * 4) + trimmed);
            else
                sb.AppendLine(new string(' ', indent * 4) + trimmed);

            // Increase indent for opening braces/brackets
            if (trimmed.EndsWith("{") || trimmed.EndsWith("[") || trimmed.EndsWith("("))
                indent++;
            // Handle single-line blocks
            else if ((trimmed.StartsWith("if ") || trimmed.StartsWith("while ") || 
                      trimmed.StartsWith("for ") || trimmed.StartsWith("fn ")) && 
                     !trimmed.EndsWith("{") && !trimmed.EndsWith("}"))
            {
                // Single line control - no indent change
            }
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// List all capabilities for AI discovery - canonical discovery endpoint
    /// </summary>
    static int RunCapabilities(bool jsonOutput)
    {
        var capabilities = new Dictionary<string, object>
        {
            // Version and platform info
            ["version"] = "1.0.0",
            ["runtime"] = $".NET {Environment.Version}",
            ["platform"] = new
            {
                os = Environment.OSVersion.ToString(),
                arch = Environment.Is64BitProcess ? "x64" : "x86",
                processors = Environment.ProcessorCount,
                machine = Environment.MachineName
            },
            
            // Consciousness operators
            ["operators"] = new[]
            {
                new { symbol = "|>", name = "pipe", args = "value, fn", returns = "any", description = "Chain transformations left-to-right" },
                new { symbol = "~>", name = "awareness", args = "value, fn", returns = "any", description = "Introspective flow with self-reference" },
                new { symbol = "=>>", name = "gradient", args = "value, fn", returns = "any", description = "Learning/adjustment with feedback" },
                new { symbol = "*>", name = "attention", args = "value, weights", returns = "any", description = "Focus mechanism with weights" },
                new { symbol = "+>", name = "superposition", args = "value, fn", returns = "any", description = "Quantum-like state superposition" }
            },
            
            // All namespaces with functions and signatures
            ["namespaces"] = new Dictionary<string, object>
            {
                ["file"] = new { functions = new[] {
                    new { name = "read", args = "path: string", returns = "string", description = "Read file contents" },
                    new { name = "write", args = "path: string, content: string", returns = "bool", description = "Write to file" },
                    new { name = "append", args = "path: string, content: string", returns = "bool", description = "Append to file" },
                    new { name = "exists", args = "path: string", returns = "bool", description = "Check if file exists" },
                    new { name = "delete", args = "path: string", returns = "bool", description = "Delete file" },
                    new { name = "copy", args = "src: string, dst: string", returns = "bool", description = "Copy file" },
                    new { name = "move", args = "src: string, dst: string", returns = "bool", description = "Move file" }
                }},
                ["dir"] = new { functions = new[] {
                    new { name = "list", args = "path?: string", returns = "list", description = "List directory contents" },
                    new { name = "files", args = "path: string, pattern?: string", returns = "list", description = "List files" },
                    new { name = "dirs", args = "path: string", returns = "list", description = "List subdirectories" },
                    new { name = "create", args = "path: string", returns = "bool", description = "Create directory" },
                    new { name = "delete", args = "path: string, recursive?: bool", returns = "bool", description = "Delete directory" },
                    new { name = "tree", args = "path: string, depth?: number", returns = "list", description = "Directory tree" },
                    new { name = "walk", args = "path: string", returns = "list", description = "Recursive file list" }
                }},
                ["path"] = new { functions = new[] {
                    new { name = "join", args = "...parts: string", returns = "string", description = "Join path segments" },
                    new { name = "dirname", args = "path: string", returns = "string", description = "Get directory name" },
                    new { name = "basename", args = "path: string", returns = "string", description = "Get file name" },
                    new { name = "ext", args = "path: string", returns = "string", description = "Get extension" },
                    new { name = "absolute", args = "path: string", returns = "string", description = "Get absolute path" },
                    new { name = "exists", args = "path: string", returns = "bool", description = "Check path exists" }
                }},
                ["string"] = new { functions = new[] {
                    new { name = "upper", args = "s: string", returns = "string", description = "Uppercase" },
                    new { name = "lower", args = "s: string", returns = "string", description = "Lowercase" },
                    new { name = "trim", args = "s: string", returns = "string", description = "Trim whitespace" },
                    new { name = "split", args = "s: string, sep?: string", returns = "list", description = "Split string" },
                    new { name = "contains", args = "s: string, sub: string", returns = "bool", description = "Check contains" },
                    new { name = "replace", args = "s: string, old: string, new: string", returns = "string", description = "Replace substring" },
                    new { name = "repeat", args = "s: string, n: number", returns = "string", description = "Repeat string" }
                }},
                ["list"] = new { functions = new[] {
                    new { name = "sum", args = "list: list", returns = "number", description = "Sum of numbers" },
                    new { name = "avg", args = "list: list", returns = "number", description = "Average" },
                    new { name = "min", args = "list: list", returns = "number", description = "Minimum" },
                    new { name = "max", args = "list: list", returns = "number", description = "Maximum" },
                    new { name = "sort", args = "list: list", returns = "list", description = "Sort ascending" },
                    new { name = "reverse", args = "list: list", returns = "list", description = "Reverse list" },
                    new { name = "unique", args = "list: list", returns = "list", description = "Remove duplicates" },
                    new { name = "flatten", args = "list: list", returns = "list", description = "Flatten nested lists" }
                }},
                ["json"] = new { functions = new[] {
                    new { name = "parse", args = "s: string", returns = "object", description = "Parse JSON string" },
                    new { name = "stringify", args = "obj: any", returns = "string", description = "Convert to JSON" },
                    new { name = "pretty", args = "obj: any", returns = "string", description = "Pretty print JSON" },
                    new { name = "valid", args = "s: string", returns = "bool", description = "Check if valid JSON" }
                }},
                ["yaml"] = new { functions = new[] {
                    new { name = "parse", args = "s: string", returns = "object", description = "Parse YAML string" },
                    new { name = "stringify", args = "obj: any", returns = "string", description = "Convert to YAML" }
                }},
                ["xml"] = new { functions = new[] {
                    new { name = "parse", args = "s: string", returns = "object", description = "Parse XML string" },
                    new { name = "query", args = "s: string, xpath: string", returns = "list", description = "XPath query" }
                }},
                ["regex"] = new { functions = new[] {
                    new { name = "test", args = "text: string, pattern: string", returns = "bool", description = "Test if matches" },
                    new { name = "match", args = "text: string, pattern: string", returns = "string?", description = "First match" },
                    new { name = "matches", args = "text: string, pattern: string", returns = "list", description = "All matches" },
                    new { name = "replace", args = "text: string, pattern: string, repl: string", returns = "string", description = "Regex replace" },
                    new { name = "split", args = "text: string, pattern: string", returns = "list", description = "Split by pattern" }
                }},
                ["crypto"] = new { functions = new[] {
                    new { name = "hash", args = "data: string, algo?: string", returns = "string", description = "Hash data (sha256 default)" },
                    new { name = "uuid", args = "", returns = "string", description = "Generate UUID" },
                    new { name = "random", args = "length?: number", returns = "string", description = "Random bytes" },
                    new { name = "base64encode", args = "data: string", returns = "string", description = "Base64 encode" },
                    new { name = "base64decode", args = "data: string", returns = "string", description = "Base64 decode" }
                }},
                ["http"] = new { functions = new[] {
                    new { name = "get", args = "url: string, headers?: object", returns = "object", description = "HTTP GET" },
                    new { name = "post", args = "url: string, body: any, headers?: object", returns = "object", description = "HTTP POST" },
                    new { name = "download", args = "url: string, path: string", returns = "bool", description = "Download file" }
                }},
                ["git"] = new { functions = new[] {
                    new { name = "status", args = "", returns = "object", description = "Git status {clean, files, count}" },
                    new { name = "branch", args = "", returns = "string", description = "Current branch name" },
                    new { name = "branches", args = "", returns = "list", description = "All branches" },
                    new { name = "log", args = "n?: number", returns = "list", description = "Commit log" },
                    new { name = "diff", args = "file?: string", returns = "string", description = "Show diff" },
                    new { name = "isRepo", args = "", returns = "bool", description = "Check if in git repo" }
                }},
                ["proc"] = new { functions = new[] {
                    new { name = "list", args = "filter?: string", returns = "list", description = "List processes" },
                    new { name = "kill", args = "pid: number", returns = "bool", description = "Kill process" },
                    new { name = "exists", args = "pid: number", returns = "bool", description = "Check if running" },
                    new { name = "info", args = "pid: number", returns = "object", description = "Process details" }
                }},
                ["net"] = new { functions = new[] {
                    new { name = "ping", args = "host: string", returns = "object", description = "Ping host {success, time}" },
                    new { name = "lookup", args = "host: string", returns = "string", description = "DNS lookup" },
                    new { name = "localIp", args = "", returns = "string", description = "Local IP address" },
                    new { name = "isOnline", args = "", returns = "bool", description = "Check internet connection" }
                }},
                ["env"] = new { functions = new[] {
                    new { name = "get", args = "name: string", returns = "string", description = "Get env variable" },
                    new { name = "set", args = "name: string, value: string", returns = "bool", description = "Set env variable" },
                    new { name = "home", args = "", returns = "string", description = "Home directory" },
                    new { name = "user", args = "", returns = "string", description = "Current username" },
                    new { name = "os", args = "", returns = "string", description = "OS name" },
                    new { name = "arch", args = "", returns = "string", description = "Architecture" }
                }},
                ["sys"] = new { functions = new[] {
                    new { name = "exec", args = "cmd: string, cwd?: string", returns = "object", description = "Execute command {stdout, stderr, code, success}" },
                    new { name = "shell", args = "cmd: string", returns = "string", description = "Execute and return stdout" },
                    new { name = "pipe", args = "...cmds: string", returns = "string", description = "Chain commands (stdout→stdin)" },
                    new { name = "run", args = "cmd: string, stdin?: string", returns = "object", description = "Run with stdin input" },
                    new { name = "sleep", args = "ms: number", returns = "bool", description = "Sleep milliseconds" },
                    new { name = "exit", args = "code?: number", returns = "void", description = "Exit process" }
                }},
                ["clip"] = new { functions = new[] {
                    new { name = "copy", args = "text: string", returns = "bool", description = "Copy to clipboard" },
                    new { name = "paste", args = "", returns = "string", description = "Paste from clipboard" }
                }},
                ["zip"] = new { functions = new[] {
                    new { name = "create", args = "src: string, dest: string", returns = "bool", description = "Create zip archive" },
                    new { name = "extract", args = "src: string, dest: string", returns = "bool", description = "Extract zip archive" },
                    new { name = "list", args = "path: string", returns = "list", description = "List archive contents" }
                }},
                ["diff"] = new { functions = new[] {
                    new { name = "lines", args = "old: string, new: string", returns = "list", description = "Line-by-line diff" },
                    new { name = "files", args = "path1: string, path2: string", returns = "list", description = "File diff" },
                    new { name = "patch", args = "content: string, patch: string", returns = "string", description = "Apply patch" }
                }},
                ["template"] = new { functions = new[] {
                    new { name = "render", args = "tmpl: string, vars: object", returns = "string", description = "Render template with ${var} substitution" }
                }},
                ["date"] = new { functions = new[] {
                    new { name = "now", args = "", returns = "string", description = "Current datetime ISO" },
                    new { name = "utc", args = "", returns = "string", description = "Current UTC datetime" },
                    new { name = "parse", args = "s: string", returns = "object", description = "Parse date string" },
                    new { name = "format", args = "date: any, fmt: string", returns = "string", description = "Format date" }
                }},
                ["math"] = new { functions = new[] {
                    new { name = "sin", args = "x: number", returns = "number", description = "Sine" },
                    new { name = "cos", args = "x: number", returns = "number", description = "Cosine" },
                    new { name = "sqrt", args = "x: number", returns = "number", description = "Square root" },
                    new { name = "abs", args = "x: number", returns = "number", description = "Absolute value" },
                    new { name = "pow", args = "x: number, y: number", returns = "number", description = "Power" },
                    new { name = "floor", args = "x: number", returns = "number", description = "Floor" },
                    new { name = "ceil", args = "x: number", returns = "number", description = "Ceiling" },
                    new { name = "round", args = "x: number", returns = "number", description = "Round" },
                    new { name = "random", args = "", returns = "number", description = "Random 0-1" }
                }}
            },
            
            // GPU capabilities
            ["gpu"] = new
            {
                available = true,
                operations = new[] { "tensor", "matmul", "add", "multiply", "transpose", "benchmark" }
            },
            
            // Special features
            ["features"] = new[]
            {
                "Shell pipelines (sys.pipe)",
                "Interactive REPL with history/completion",
                "ProductionMath - adaptive semantic mathematics",
                "Python interop - call Python code",
                "GPU acceleration - CUDA tensors",
                "Auto-fix - self-correcting code",
                "Consciousness operators - AI-native flow",
                "MCP server integration"
            }
        };

        if (jsonOutput)
        {
            System.Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(capabilities, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        }
        else
        {
            System.Console.WriteLine("NSL Capabilities (use --json for machine-readable output)");
            System.Console.WriteLine("=========================================================");
            System.Console.WriteLine();
            System.Console.WriteLine($"Version: 1.0.0 | Runtime: .NET {Environment.Version} | Platform: {(Environment.Is64BitProcess ? "x64" : "x86")}");
            System.Console.WriteLine();
            System.Console.WriteLine("Consciousness Operators:");
            System.Console.WriteLine("  |>   pipe          - Chain transformations");
            System.Console.WriteLine("  ~>   awareness     - Introspective flow");
            System.Console.WriteLine("  =>>  gradient      - Learning/adjustment");
            System.Console.WriteLine("  *>   attention     - Focus mechanism");
            System.Console.WriteLine("  +>   superposition - Quantum states");
            System.Console.WriteLine();
            System.Console.WriteLine("Namespaces (20): file, dir, path, string, list, json, yaml, xml, regex,");
            System.Console.WriteLine("                 crypto, http, git, proc, net, env, sys, clip, zip, diff,");
            System.Console.WriteLine("                 template, date, math");
            System.Console.WriteLine();
            System.Console.WriteLine("Features: Shell pipelines, REPL, ProductionMath, Python interop, GPU, Auto-fix");
            System.Console.WriteLine();
            System.Console.WriteLine("Run 'nsl --capabilities --json' for full function signatures.");
        }

        return 0;
    }

    /// <summary>
    /// Run file with auto-fix analysis
    /// </summary>
    static int RunWithAutoFix(string fileName, bool applyFixes)
    {
        if (!File.Exists(fileName))
        {
            System.Console.WriteLine($"Error: File not found: {fileName}");
            return 1;
        }

        try
        {
            var content = File.ReadAllText(fileName);
            var ext = Path.GetExtension(fileName).ToLower();
            
            // Use multi-language auto-fix for all files
            var multiLangFix = new MultiLanguageAutoFix(content, fileName);
            multiLangFix.Analyze();
            
            // For NSL files, also run the specialized NSL fixer
            NSLAutoFix? autoFix = null;
            if (ext == ".nsl" || ext == "")
            {
                autoFix = new NSLAutoFix(content);
                autoFix.Analyze();
            }
            
            // Use appropriate fixer based on file type
            var isNslFile = ext == ".nsl" || ext == "";
            var fixCount = isNslFile ? (autoFix?.Fixes.Count ?? 0) : multiLangFix.Fixes.Count;

            if (fixCount == 0)
            {
                var lang = isNslFile ? "NSL" : multiLangFix.Language.ToString();
                System.Console.WriteLine($"No issues found in {lang} code - looks good!");
                if (isNslFile) return RunFile(fileName);
                return 0;
            }

            // Show errors with context
            if (isNslFile && autoFix != null)
            {
                System.Console.WriteLine(autoFix.GetAllErrorsWithContext());
            }
            else
            {
                System.Console.WriteLine($"Detected language: {multiLangFix.Language}");
                System.Console.WriteLine(multiLangFix.GetAllErrorsWithContext());
            }

            if (applyFixes)
            {
                // Apply fixes with multi-pass support
                string fixedSource = content;
                int totalErrorCount = 0;
                int maxPasses = 3; // Prevent infinite loops
                
                for (int pass = 0; pass < maxPasses; pass++)
                {
                    int passErrorCount;
                    string passResult;
                    
                    if (isNslFile)
                    {
                        var passAutoFix = new NSLAutoFix(fixedSource);
                        passAutoFix.Analyze();
                        passResult = passAutoFix.ApplyFixes(FixCategory.Warning);
                        passErrorCount = passAutoFix.Fixes.Count(f => f.Category <= FixCategory.Warning);
                    }
                    else
                    {
                        var passMultiLang = new MultiLanguageAutoFix(fixedSource, fileName);
                        passMultiLang.Analyze();
                        passResult = passMultiLang.ApplyFixes(FixCategory.Warning);
                        passErrorCount = passMultiLang.Fixes.Count(f => f.Category <= FixCategory.Warning);
                    }
                    
                    if (passResult == fixedSource || passErrorCount == 0)
                        break; // No more fixes to apply
                    
                    fixedSource = passResult;
                    totalErrorCount += passErrorCount;
                }
                
                if (fixedSource != content)
                {
                    File.WriteAllText(fileName, fixedSource);
                    System.Console.WriteLine($"Auto-fixed {totalErrorCount} issue(s) in {fileName}");
                    System.Console.WriteLine();

                    // Run the fixed file only if NSL
                    if (isNslFile) return RunFile(fileName);
                }
            }
            else
            {
                System.Console.WriteLine("Run with --fix to automatically apply fixes.");
            }

            return 0;
        }
        catch (Exception ex)
        {
            System.Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    public static int RunFile(string fileName)
    {
        if (!File.Exists(fileName))
        {
            System.Console.WriteLine($"Error: File not found: {fileName}");
            return 1;
        }

        try
        {
            var fullPath = Path.GetFullPath(fileName);
            var content = File.ReadAllText(fullPath);
            var lexer = new NSLLexer(content, fullPath);
            var parser = new NSLParser();
            var interpreter = new NSLInterpreter();

            // Set the source file for module resolution
            interpreter.SetSourceFile(fullPath);

            System.Console.WriteLine($"Running file: {fileName}");

            var tokens = lexer.Tokenize();

            // Debug: Show tokens for troubleshooting
            if (Environment.GetEnvironmentVariable("NSL_DEBUG_TOKENS") == "1")
            {
                System.Console.WriteLine("=== TOKENS ===");
                foreach (var t in tokens)
                {
                    System.Console.WriteLine($"  {t.Type}: '{t.Value}' at {t.Line}:{t.Column}");
                }
                System.Console.WriteLine("==============");
            }

            var ast = parser.Parse(tokens);
            var result = interpreter.Execute(ast);

            if (result != null)
            {
                System.Console.WriteLine($"Result: {result}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            ColorOutput.Error($"Error: {ex.Message}");
            return 1;
        }
    }

    static void InitializeConsole()
    {
        try
        {
            // Set console title
            System.Console.Title = "NSL - Neural Symbolic Language";
            
            // Set console encoding for Unicode support (emojis, checkmarks)
            System.Console.OutputEncoding = System.Text.Encoding.UTF8;
            System.Console.InputEncoding = System.Text.Encoding.UTF8;
            
            // On Windows, try to set buffer and window size for better experience
            if (OperatingSystem.IsWindows())
            {
                try
                {
                    // Increase buffer size for scrollback
                    if (System.Console.BufferHeight < 9000)
                        System.Console.BufferHeight = 9000;
                    
                    // Set reasonable window size if too small
                    if (System.Console.WindowWidth < 80)
                        System.Console.WindowWidth = 120;
                    if (System.Console.WindowHeight < 24)
                        System.Console.WindowHeight = 30;
                }
                catch { } // Ignore if not supported
                
                // Set console window icon for taskbar
                SetConsoleIcon();
            }
            
            // Enable virtual terminal processing for ANSI colors on Windows
            if (OperatingSystem.IsWindows())
            {
                EnableVirtualTerminal();
            }
        }
        catch { } // Don't crash if console setup fails
    }
    
    static void SetConsoleIcon()
    {
        try
        {
            // Try to load icon from file next to exe
            var iconPath = Path.Combine(AppContext.BaseDirectory, "nsl.ico");
            if (!File.Exists(iconPath))
                iconPath = @"C:\NSL\nsl.ico";
            
            if (File.Exists(iconPath))
            {
                var hIcon = LoadImage(IntPtr.Zero, iconPath, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);
                if (hIcon != IntPtr.Zero)
                {
                    var hwnd = GetConsoleWindow();
                    if (hwnd != IntPtr.Zero)
                    {
                        // Set both small (titlebar) and big (taskbar/alt-tab) icons
                        SendMessage(hwnd, WM_SETICON, ICON_SMALL, hIcon);
                        SendMessage(hwnd, WM_SETICON, ICON_BIG, hIcon);
                    }
                }
            }
        }
        catch { }
    }
    
    private const uint WM_SETICON = 0x0080;
    private const int ICON_SMALL = 0;
    private const int ICON_BIG = 1;
    private const uint IMAGE_ICON = 1;
    private const uint LR_LOADFROMFILE = 0x00000010;
    private const uint LR_DEFAULTSIZE = 0x00000040;
    
    [System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto)]
    static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, int wParam, IntPtr lParam);
    
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    static extern IntPtr GetConsoleWindow();
    
    [System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto)]
    static extern IntPtr LoadImage(IntPtr hInst, string lpszName, uint uType, int cxDesired, int cyDesired, uint fuLoad);
    
    static void EnableVirtualTerminal()
    {
        try
        {
            // Windows 10+ supports ANSI escape codes with virtual terminal
            var handle = GetStdHandle(-11); // STD_OUTPUT_HANDLE
            if (handle != IntPtr.Zero)
            {
                GetConsoleMode(handle, out uint mode);
                SetConsoleMode(handle, mode | 0x0004); // ENABLE_VIRTUAL_TERMINAL_PROCESSING
            }
        }
        catch { }
    }
    
    [System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr GetStdHandle(int nStdHandle);
    
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
    
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);

    static int RunREPL()
    {
        var interpreter = new NSLInterpreter();
        
        ShowNSLHeader();
        RunRepl(interpreter);
        
        return 0;
    }

    private static void RunRepl(NSLInterpreter interpreter)
    {
        System.Console.WriteLine("");
        ColorOutput.Hint("Tab: completion | ↑↓: history | Type 'help' for commands");
        System.Console.WriteLine("");

        var inputBuffer = new StringBuilder();
        var braceCount = 0;
        var parenCount = 0;
        var console = new InteractiveConsole();

        while (true)
        {
            try
            {
                // Show prompt and read with interactive features
                var prompt = inputBuffer.Length > 0 ? "  > " : "nsl> ";
                var line = console.ReadLine(prompt);
                
                if (line == null) // Ctrl+C
                {
                    if (inputBuffer.Length > 0)
                    {
                        System.Console.WriteLine("\n[Input cancelled]");
                        inputBuffer.Clear();
                        braceCount = 0;
                        parenCount = 0;
                        continue;
                    }
                    break;
                }

                // Handle empty lines
                if (string.IsNullOrWhiteSpace(line))
                {
                    if (inputBuffer.Length == 0)
                        continue; // Skip empty lines in single-line mode
                    // In multi-line mode, empty line might mean "done" if balanced
                    if (braceCount == 0 && parenCount == 0)
                    {
                        var completeInput = inputBuffer.ToString();
                        inputBuffer.Clear();
                        ExecuteInput(completeInput, interpreter);
                        continue;
                    }
                }

                // Special commands only in single-line mode
                if (inputBuffer.Length == 0 && HandleSpecialCommand(line.Trim(), interpreter))
                    continue;

                // Add line to buffer
                if (inputBuffer.Length > 0)
                    inputBuffer.Append('\n');
                inputBuffer.Append(line);

                // Count delimiters in the current line
                foreach (char c in line)
                {
                    switch (c)
                    {
                        case '{': braceCount++; break;
                        case '}': braceCount--; break;
                        case '(': parenCount++; break;
                        case ')': parenCount--; break;
                    }
                }

                // Check if statement is complete
                if (braceCount == 0 && parenCount == 0)
                {
                    // Additional check: does it end with something expecting more?
                    var trimmed = line.Trim();
                    if (trimmed.EndsWith("else") || trimmed.EndsWith("{") || 
                        trimmed.EndsWith("=") || trimmed.EndsWith("+") || 
                        trimmed.EndsWith("-") || trimmed.EndsWith("::"))
                    {
                        continue; // Need more input
                    }

                    // Execute complete statement
                    var completeInput = inputBuffer.ToString();
                    inputBuffer.Clear();
                    braceCount = 0;
                    parenCount = 0;
                    ExecuteInput(completeInput, interpreter);
                }
                // else continue accumulating input
            }
            catch (Exception ex)
            {
                ColorOutput.Error(ex.Message);
                inputBuffer.Clear();
                braceCount = 0;
                parenCount = 0;
            }
        }

        ColorOutput.Info("Goodbye!");
    }

    // PowerShell-style check for complete statements
    private static bool IsCompleteStatement(string input)
    {
        // Empty input is complete
        if (string.IsNullOrWhiteSpace(input))
            return true;

        // Count delimiters (ignoring those in strings)
        int braceCount = 0;
        int parenCount = 0;
        int bracketCount = 0;
        bool inString = false;
        bool inSingleQuote = false;
        bool escaped = false;

        foreach (char c in input)
        {
            if (escaped)
            {
                escaped = false;
                continue;
            }

            if (c == '\\')
            {
                escaped = true;
                continue;
            }

            // Handle strings
            if (!inSingleQuote && c == '"')
                inString = !inString;
            else if (!inString && c == '\'')
                inSingleQuote = !inSingleQuote;

            // Count delimiters only outside strings
            if (!inString && !inSingleQuote)
            {
                switch (c)
                {
                    case '{': braceCount++; break;
                    case '}': braceCount--; break;
                    case '(': parenCount++; break;
                    case ')': parenCount--; break;
                    case '[': bracketCount++; break;
                    case ']': bracketCount--; break;
                }
            }
        }

        // If any delimiters are unclosed, statement is incomplete
        if (braceCount != 0 || parenCount != 0 || bracketCount != 0)
            return false;

        // Check if the statement ends with something that expects continuation
        var lines = input.Split('\n');
        var lastLine = lines[lines.Length - 1].TrimEnd();
        
        // These endings indicate more input is expected
        string[] continuationEndings = { 
            "else",     // else expects a block or statement
            "=",        // assignment expects a value
            "+", "-", "*", "/", "%",  // operators expect operands
            "&&", "||", // logical operators expect operands
            "::",       // chain operator expects next expression
            ",",        // comma in lists/parameters
            "{"         // opening brace expects content
        };

        foreach (var ending in continuationEndings)
        {
            if (lastLine.EndsWith(ending))
                return false;
        }

        // Special case: if/while/for on its own line expects more
        var trimmedLast = lastLine.Trim();
        if (trimmedLast == "if" || trimmedLast == "while" || trimmedLast == "for")
            return false;

        // Statement is complete
        return true;
    }

    // Simplified REPL command check
    private static bool IsReplCommand(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return false;
            
        var firstWord = input.Split(' ')[0].ToLower();
        return firstWord switch
        {
            // Meta commands
            "help" or "exit" or "quit" or "clear" or "cls" or 
            "metrics" or "history" or "syntax" or "tokens" or ".load" or
            // Shell commands (work like terminal - no quotes needed)
            "cd" or "pwd" or "ls" or "cat" or "touch" or "mkdir" or
            "rm" or "cp" or "mv" or "edit" or "open" or "run" or
            // Bookmark commands
            "go" or "save" or "unsave" or "bookmarks" or
            // History commands
            "history" or "restore" => true,
            _ => false
        };
    }

    // Handle special commands (REPL commands, debug tokens, shell commands)
    private static bool HandleSpecialCommand(string input, NSLInterpreter interpreter)
    {
        if (string.IsNullOrWhiteSpace(input))
            return false;

        // Check for REPL commands
        if (IsReplCommand(input))
        {
            HandleReplCommand(input, interpreter);
            return true;
        }
        
        // Check for direct file/directory path (e.g., C:\path\file.txt or ./file.txt)
        if (IsFilePath(input))
        {
            HandleFilePath(input);
            return true;
        }
        
        // Check for debug tokens (#expression)
        if (input.StartsWith("#") && input.Length > 1)
        {
            ShowTokens(input.Substring(1).Trim());
            return true;
        }
        
        // Check for shell commands (!command)
        if (input.StartsWith("!"))
        {
            ExecuteShellCommand(input.Substring(1).Trim());
            return true;
        }

        return false;
    }
    
    // Check if input looks like a file path
    private static bool IsFilePath(string input)
    {
        var trimmed = input.Trim();
        
        // Absolute path (C:\... or /...)
        if (Path.IsPathRooted(trimmed))
            return true;
            
        // Relative path starting with ./ or ..\
        if (trimmed.StartsWith("./") || trimmed.StartsWith(".\\") ||
            trimmed.StartsWith("../") || trimmed.StartsWith("..\\"))
            return true;
            
        // Path with backslash (Windows style)
        if (trimmed.Contains("\\") && !trimmed.Contains(" ") && 
            (File.Exists(trimmed) || Directory.Exists(trimmed) ||
             File.Exists(Path.Combine(Directory.GetCurrentDirectory(), trimmed)) ||
             Directory.Exists(Path.Combine(Directory.GetCurrentDirectory(), trimmed))))
            return true;
            
        return false;
    }
    
    // Handle file/directory path input
    private static void HandleFilePath(string input)
    {
        var path = input.Trim();
        
        // Make absolute if relative
        if (!Path.IsPathRooted(path))
            path = Path.Combine(Directory.GetCurrentDirectory(), path);
            
        try
        {
            if (Directory.Exists(path))
            {
                // It's a directory - cd into it
                Directory.SetCurrentDirectory(path);
                ColorOutput.Success(path);
            }
            else if (File.Exists(path))
            {
                // It's a file - open in default editor
                var ext = Path.GetExtension(path).ToLower();
                
                // For code/text files, try VS Code first, then notepad
                var codeExtensions = new[] { ".cs", ".nsl", ".js", ".ts", ".py", ".json", ".xml", ".yaml", ".yml", ".md", ".txt", ".html", ".css", ".sh", ".ps1", ".bat", ".cmd" };
                
                if (codeExtensions.Contains(ext))
                {
                    // Try VS Code
                    try
                    {
                        var psi = new System.Diagnostics.ProcessStartInfo("code", $"\"{path}\"")
                        {
                            UseShellExecute = false,
                            CreateNoWindow = true
                        };
                        System.Diagnostics.Process.Start(psi);
                        ColorOutput.Success($"Opened in VS Code: {path}");
                        return;
                    }
                    catch { }
                }
                
                // Fall back to default handler
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(path) { UseShellExecute = true });
                ColorOutput.Success($"Opened: {path}");
            }
            else
            {
                ColorOutput.Error($"Path not found: {path}");
            }
        }
        catch (Exception ex)
        {
            ColorOutput.Error($"Failed to open: {ex.Message}");
        }
    }

    // Execute the complete input
    private static void ExecuteInput(string input, NSLInterpreter interpreter)
    {
        try
        {
            // Tokenize
            var lexer = new NSLLexer(input);
            var tokens = lexer.Tokenize();

            // Parse
            var parser = new NSLParser();
            var ast = parser.Parse(tokens);

            // Execute
            var result = interpreter.Execute(ast);

            // Only show result for expressions, not statements
            if (result != null && !IsStatement(input))
            {
                ColorOutput.Result(FormatValue(result));
            }
        }
        catch (NSLParseException ex)
        {
            ColorOutput.Error($"Parse Error: {ex.Message}");
        }
        catch (NSLRuntimeException ex)
        {
            ColorOutput.Error($"Runtime Error: {ex.Message}");
        }
    }

    // Simple check for incomplete input - PowerShell style
    private static bool NeedsMoreInput(string input)
    {
        // Remove strings and comments to avoid false positives
        var processed = RemoveStringsAndComments(input);
        
        // Count delimiters
        int openBraces = 0;
        int openBrackets = 0;
        int openParens = 0;
        
        foreach (char c in processed)
        {
            switch (c)
            {
                case '{': openBraces++; break;
                case '}': openBraces--; break;
                case '[': openBrackets++; break;
                case ']': openBrackets--; break;
                case '(': openParens++; break;
                case ')': openParens--; break;
            }
        }
        
        // If any delimiters are unmatched, we need more input
        if (openBraces > 0 || openBrackets > 0 || openParens > 0)
            return true;
        
        // Check if last meaningful token expects something to follow
        var lines = input.Split('\n');
        var lastLine = lines[lines.Length - 1].Trim();
        
        // Empty line after content usually means done
        if (lastLine.Length == 0 && lines.Length > 1)
            return false;
        
        var trimmed = processed.Trim();
        if (trimmed.Length == 0)
            return false;
        
        // These tokens at the END expect continuation
        if (trimmed.EndsWith("else") || trimmed.EndsWith("=") || 
            trimmed.EndsWith("+") || trimmed.EndsWith("-") || 
            trimmed.EndsWith("*") || trimmed.EndsWith("/") || 
            trimmed.EndsWith("&&") || trimmed.EndsWith("||") || 
            trimmed.EndsWith("::") || trimmed.EndsWith(","))
            return true;
        
        // Special case: if the input contains a complete if statement with body, it's done
        if (lines.Length > 1)
        {
            // Multi-line input - check if it looks complete
            var firstLine = lines[0].Trim().ToLower();
            
            // If first line is just "if (condition)" and we have more lines, check indentation
            if ((firstLine.StartsWith("if") || firstLine.StartsWith("while")) && 
                firstLine.EndsWith(")"))
            {
                // Check if subsequent lines have content (the body)
                bool hasBody = false;
                for (int i = 1; i < lines.Length; i++)
                {
                    if (lines[i].Trim().Length > 0)
                    {
                        hasBody = true;
                        break;
                    }
                }
                
                // If we have a body, we're probably done
                // Unless the last line is "else"
                if (hasBody && !lastLine.Equals("else", StringComparison.OrdinalIgnoreCase))
                    return false;
            }
        }
        else
        {
            // Single line - check for complete single-line if/while
            if (IsSingleLineControlFlow(trimmed))
                return false;
        }
        
        return false;
    }

    private static string RemoveStringsAndComments(string input)
    {
        var result = new StringBuilder();
        bool inString = false;
        bool inComment = false;
        char stringChar = '\0';
        
        for (int i = 0; i < input.Length; i++)
        {
            char c = input[i];
            
            if (inComment)
            {
                if (c == '\n')
                    inComment = false;
                continue;
            }
            
            if (inString)
            {
                if (c == stringChar && (i == 0 || input[i-1] != '\\'))
                    inString = false;
                continue;
            }
            
            if (c == '#')
            {
                inComment = true;
                continue;
            }
            
            if (c == '"' || c == '\'')
            {
                inString = true;
                stringChar = c;
                continue;
            }
            
            result.Append(c);
        }
        
        return result.ToString();
    }

    private static bool ContainsStatementAfterCondition(string line)
    {
        // Find the last closing paren
        int lastParen = line.LastIndexOf(')');
        if (lastParen < 0 || lastParen == line.Length - 1)
            return false;
            
        string after = line.Substring(lastParen + 1).Trim();
        return after.Length > 0;
    }

    private static bool IsSingleLineControlFlow(string line)
    {
        // Check if it's a complete single-line if/while statement
        // Examples: if (x) print("hi"), while (x) break
        
        var lower = line.ToLower();
        
        // Check for patterns like "if (...) statement" or "while (...) statement"
        if (lower.StartsWith("if") || lower.StartsWith("while"))
        {
            // Find the closing paren
            int parenCount = 0;
            int conditionEnd = -1;
            bool inCondition = false;
            
            for (int i = 0; i < line.Length; i++)
            {
                if (line[i] == '(')
                {
                    parenCount++;
                    inCondition = true;
                }
                else if (line[i] == ')')
                {
                    parenCount--;
                    if (parenCount == 0 && inCondition)
                    {
                        conditionEnd = i;
                        break;
                    }
                }
            }
            
            // If we found the end of condition and there's more after it, it's complete
            if (conditionEnd >= 0 && conditionEnd < line.Length - 1)
            {
                string afterCondition = line.Substring(conditionEnd + 1).Trim();
                return afterCondition.Length > 0 && !afterCondition.Equals("{");
            }
        }
        
        return false;
    }

    private static bool IsStatement(string input)
    {
        var trimmed = input.Trim().ToLower();
        return trimmed.StartsWith("if ") || trimmed.StartsWith("while ") || 
               trimmed.StartsWith("for ") || trimmed.Contains("=") ||
               trimmed.StartsWith("print(") || trimmed.StartsWith("break") ||
               trimmed.StartsWith("continue") || trimmed.StartsWith("return");
    }





    // Format values for display
    private static string FormatValue(object? value)
    {
        if (value == null) return "none";
        if (value is bool b) return b ? "true" : "false";
        return value.ToString() ?? "null";
    }

    // Cached console availability check
    private static bool? _canUseConsoleInput = null;
    
    private static bool CanUseConsoleInput()
    {
        if (_canUseConsoleInput.HasValue)
            return _canUseConsoleInput.Value;
            
        // Check once at first use
        if (System.Console.IsInputRedirected)
        {
            _canUseConsoleInput = false;
            return false;
        }
        
        try
        {
            // Test if KeyAvailable works (throws on non-console)
            _ = System.Console.KeyAvailable;
            _canUseConsoleInput = true;
            return true;
        }
        catch
        {
            _canUseConsoleInput = false;
            return false;
        }
    }
    
    private static string? ReadLineWithHistory(List<string> history, ref int historyIndex)
    {
        // Use simple ReadLine if console input isn't available
        if (!CanUseConsoleInput())
            return System.Console.ReadLine();
        
        var input = new StringBuilder();
        var cursorPos = 0;

        while (true)
        {
            var key = System.Console.ReadKey(true);

            switch (key.Key)
            {
                case ConsoleKey.Enter:
                    System.Console.WriteLine();
                    return input.ToString();

                case ConsoleKey.Backspace:
                    if (cursorPos > 0)
                    {
                        input.Remove(cursorPos - 1, 1);
                        cursorPos--;
                        RedrawLineSimple(input.ToString(), cursorPos);
                    }
                    break;

                case ConsoleKey.LeftArrow:
                    if (cursorPos > 0)
                    {
                        cursorPos--;
                        System.Console.SetCursorPosition(System.Console.CursorLeft - 1, System.Console.CursorTop);
                    }
                    break;

                case ConsoleKey.RightArrow:
                    if (cursorPos < input.Length)
                    {
                        cursorPos++;
                        System.Console.SetCursorPosition(System.Console.CursorLeft + 1, System.Console.CursorTop);
                    }
                    break;

                case ConsoleKey.UpArrow:
                    if (history.Count > 0 && historyIndex > 0)
                    {
                        historyIndex--;
                        input.Clear();
                        input.Append(history[historyIndex]);
                        cursorPos = input.Length;
                        RedrawLineSimple(input.ToString(), cursorPos);
                    }
                    break;

                case ConsoleKey.DownArrow:
                    if (history.Count > 0 && historyIndex < history.Count - 1)
                    {
                        historyIndex++;
                        input.Clear();
                        input.Append(history[historyIndex]);
                        cursorPos = input.Length;
                        RedrawLineSimple(input.ToString(), cursorPos);
                    }
                    else if (historyIndex == history.Count - 1)
                    {
                        historyIndex = history.Count;
                        input.Clear();
                        cursorPos = 0;
                        RedrawLineSimple("", 0);
                    }
                    break;

                case ConsoleKey.C when key.Modifiers == ConsoleModifiers.Control:
                    return null; // Ctrl+C

                case ConsoleKey.D when key.Modifiers == ConsoleModifiers.Control:
                    return null; // Ctrl+D

                default:
                    if (!char.IsControl(key.KeyChar))
                    {
                        input.Insert(cursorPos, key.KeyChar);
                        cursorPos++;
                        RedrawLineSimple(input.ToString(), cursorPos);
                    }
                    break;
            }
        }
    }

    private static void RedrawLineSimple(string text, int cursorPos)
    {
        var currentTop = System.Console.CursorTop;
        
        // Clear the current line
        System.Console.SetCursorPosition(0, currentTop);
        System.Console.Write(new string(' ', Math.Min(System.Console.WindowWidth - 1, 80)));
        System.Console.SetCursorPosition(0, currentTop);
        
        // Write the prompt and text
        System.Console.Write("NSL> " + text);
        
        // Set cursor position
        var promptLength = "NSL> ".Length;
        System.Console.SetCursorPosition(Math.Min(promptLength + cursorPos, System.Console.WindowWidth - 1), currentTop);
    }

    private static bool HandleReplCommand(string commandPart, NSLInterpreter interpreter)
    {
        var parts = commandPart.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var command = parts.Length > 0 ? parts[0].ToLowerInvariant() : "";
        
        switch (command)
        {
            case "exit":
                Environment.Exit(0);
                return true;

            case "help":
                ShowHelp();
                return true;

            case "syntax":
                ShowSyntaxHelp();
                return true;

            case "tokens":
                ShowTokenReference();
                return true;

            case "clear":
                System.Console.Clear();
                ShowNSLHeader();
                return true;

            // ===== SIMPLE SHELL COMMANDS =====
            // These work like a normal terminal - no quotes, no parentheses needed
            
            case "cd":
                // cd           → go to home directory
                // cd Desktop   → go to Desktop (relative)
                // cd C:\Users  → go to absolute path
                try
                {
                    string targetPath;
                    if (parts.Length < 2)
                    {
                        // cd with no args = go home
                        targetPath = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                    }
                    else
                    {
                        // Join all parts after "cd" (handles paths with spaces)
                        targetPath = string.Join(" ", parts.Skip(1));
                        
                        // Handle common shortcuts
                        if (targetPath == "~") 
                            targetPath = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                        else if (targetPath == "..")
                            targetPath = Path.GetDirectoryName(Directory.GetCurrentDirectory()) ?? Directory.GetCurrentDirectory();
                        else if (!Path.IsPathRooted(targetPath))
                            targetPath = Path.Combine(Directory.GetCurrentDirectory(), targetPath);
                    }
                    
                    if (Directory.Exists(targetPath))
                    {
                        Directory.SetCurrentDirectory(targetPath);
                        ColorOutput.Success(Directory.GetCurrentDirectory());
                    }
                    else
                    {
                        ColorOutput.Error($"Directory not found: {targetPath}");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"cd failed: {ex.Message}");
                }
                return true;

            case "pwd":
                // pwd → show current directory
                ColorOutput.Info(Directory.GetCurrentDirectory());
                return true;

            case "ls":
            case "dir":
                // ls        → list current directory
                // ls path   → list specific path
                try
                {
                    var lsPath = parts.Length > 1 ? string.Join(" ", parts.Skip(1)) : Directory.GetCurrentDirectory();
                    if (!Path.IsPathRooted(lsPath))
                        lsPath = Path.Combine(Directory.GetCurrentDirectory(), lsPath);
                    
                    if (Directory.Exists(lsPath))
                    {
                        var entries = Directory.GetFileSystemEntries(lsPath);
                        foreach (var entry in entries.OrderBy(e => e))
                        {
                            var name = Path.GetFileName(entry);
                            var isDir = Directory.Exists(entry);
                            if (isDir)
                                ColorOutput.Info($"  {name}/");
                            else
                                System.Console.WriteLine($"  {name}");
                        }
                        System.Console.WriteLine($"\n{entries.Length} items");
                    }
                    else
                    {
                        ColorOutput.Error($"Directory not found: {lsPath}");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"ls failed: {ex.Message}");
                }
                return true;

            case "cat":
                // cat file.txt → show file contents
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: cat <filename>");
                    return true;
                }
                try
                {
                    var catPath = string.Join(" ", parts.Skip(1));
                    if (!Path.IsPathRooted(catPath))
                        catPath = Path.Combine(Directory.GetCurrentDirectory(), catPath);
                    
                    if (File.Exists(catPath))
                    {
                        System.Console.WriteLine(File.ReadAllText(catPath));
                    }
                    else
                    {
                        ColorOutput.Error($"File not found: {catPath}");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"cat failed: {ex.Message}");
                }
                return true;

            case "mkdir":
                // mkdir foldername → create directory
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: mkdir <foldername>");
                    return true;
                }
                try
                {
                    var mkdirPath = string.Join(" ", parts.Skip(1));
                    if (!Path.IsPathRooted(mkdirPath))
                        mkdirPath = Path.Combine(Directory.GetCurrentDirectory(), mkdirPath);
                    Directory.CreateDirectory(mkdirPath);
                    ColorOutput.Success($"Created: {mkdirPath}");
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"mkdir failed: {ex.Message}");
                }
                return true;

            case "touch":
                // touch file.txt → create empty file
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: touch <filename>");
                    return true;
                }
                try
                {
                    var touchPath = string.Join(" ", parts.Skip(1));
                    if (!Path.IsPathRooted(touchPath))
                        touchPath = Path.Combine(Directory.GetCurrentDirectory(), touchPath);
                    File.WriteAllText(touchPath, "");
                    ColorOutput.Success($"Created: {touchPath}");
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"touch failed: {ex.Message}");
                }
                return true;

            case "rm":
                // rm file.txt → delete file
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: rm <filename>");
                    return true;
                }
                try
                {
                    var rmPath = string.Join(" ", parts.Skip(1));
                    if (!Path.IsPathRooted(rmPath))
                        rmPath = Path.Combine(Directory.GetCurrentDirectory(), rmPath);
                    if (File.Exists(rmPath))
                    {
                        File.Delete(rmPath);
                        ColorOutput.Success($"Deleted: {rmPath}");
                    }
                    else if (Directory.Exists(rmPath))
                    {
                        ColorOutput.Error("Use 'rmdir' for directories");
                    }
                    else
                    {
                        ColorOutput.Error($"File not found: {rmPath}");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"rm failed: {ex.Message}");
                }
                return true;

            case "whoami":
                // whoami → show current user
                ColorOutput.Info(Environment.UserName);
                return true;

            case "rmdir":
                // rmdir folder → delete directory
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: rmdir <foldername>");
                    return true;
                }
                try
                {
                    var rmdirPath = string.Join(" ", parts.Skip(1));
                    if (!Path.IsPathRooted(rmdirPath))
                        rmdirPath = Path.Combine(Directory.GetCurrentDirectory(), rmdirPath);
                    if (Directory.Exists(rmdirPath))
                    {
                        Directory.Delete(rmdirPath, true); // recursive delete
                        ColorOutput.Success($"Deleted: {rmdirPath}");
                    }
                    else if (File.Exists(rmdirPath))
                    {
                        ColorOutput.Error("Use 'rm' for files");
                    }
                    else
                    {
                        ColorOutput.Error($"Directory not found: {rmdirPath}");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"rmdir failed: {ex.Message}");
                }
                return true;

            case "mv":
            case "move":
                // mv source dest → move file or folder
                if (parts.Length < 3)
                {
                    ColorOutput.Error("Usage: mv <source> <destination>");
                    return true;
                }
                try
                {
                    var mvSource = parts[1];
                    var mvDest = string.Join(" ", parts.Skip(2));
                    if (!Path.IsPathRooted(mvSource))
                        mvSource = Path.Combine(Directory.GetCurrentDirectory(), mvSource);
                    if (!Path.IsPathRooted(mvDest))
                        mvDest = Path.Combine(Directory.GetCurrentDirectory(), mvDest);
                    
                    if (File.Exists(mvSource))
                    {
                        // If dest is a directory, move file into it
                        if (Directory.Exists(mvDest))
                            mvDest = Path.Combine(mvDest, Path.GetFileName(mvSource));
                        File.Move(mvSource, mvDest);
                        ColorOutput.Success($"Moved: {mvSource} → {mvDest}");
                    }
                    else if (Directory.Exists(mvSource))
                    {
                        Directory.Move(mvSource, mvDest);
                        ColorOutput.Success($"Moved: {mvSource} → {mvDest}");
                    }
                    else
                    {
                        ColorOutput.Error($"Not found: {mvSource}");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"mv failed: {ex.Message}");
                }
                return true;

            case "cp":
            case "copy":
                // cp source dest → copy file
                if (parts.Length < 3)
                {
                    ColorOutput.Error("Usage: cp <source> <destination>");
                    return true;
                }
                try
                {
                    var cpSource = parts[1];
                    var cpDest = string.Join(" ", parts.Skip(2));
                    if (!Path.IsPathRooted(cpSource))
                        cpSource = Path.Combine(Directory.GetCurrentDirectory(), cpSource);
                    if (!Path.IsPathRooted(cpDest))
                        cpDest = Path.Combine(Directory.GetCurrentDirectory(), cpDest);
                    
                    if (File.Exists(cpSource))
                    {
                        // If dest is a directory, copy file into it
                        if (Directory.Exists(cpDest))
                            cpDest = Path.Combine(cpDest, Path.GetFileName(cpSource));
                        File.Copy(cpSource, cpDest, true);
                        ColorOutput.Success($"Copied: {cpSource} → {cpDest}");
                    }
                    else if (Directory.Exists(cpSource))
                    {
                        ColorOutput.Error("Use file.copy() for directory copy in scripts");
                    }
                    else
                    {
                        ColorOutput.Error($"File not found: {cpSource}");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"cp failed: {ex.Message}");
                }
                return true;

            case "save":
            case "bookmark":
                // save name → save current directory as bookmark
                // save name path → save specific path as bookmark
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: save <name> [path]");
                    return true;
                }
                try
                {
                    var bookmarkName = parts[1];
                    var bookmarkPath = parts.Length > 2 
                        ? string.Join(" ", parts.Skip(2)) 
                        : Directory.GetCurrentDirectory();
                    
                    if (!Path.IsPathRooted(bookmarkPath))
                        bookmarkPath = Path.Combine(Directory.GetCurrentDirectory(), bookmarkPath);
                    
                    SaveBookmark(bookmarkName, bookmarkPath);
                    ColorOutput.Success($"Saved '{bookmarkName}' → {bookmarkPath}");
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"save failed: {ex.Message}");
                }
                return true;

            case "go":
                // go name → cd to saved bookmark
                // go name/subpath → cd to bookmark + subpath
                if (parts.Length < 2)
                {
                    // Show all bookmarks
                    var bookmarks = LoadBookmarks();
                    if (bookmarks.Count == 0)
                    {
                        ColorOutput.Info("No bookmarks saved. Use: save <name> [path]");
                    }
                    else
                    {
                        ColorOutput.Info("Bookmarks:");
                        foreach (var bm in bookmarks)
                        {
                            System.Console.WriteLine($"  {bm.Key} → {bm.Value}");
                        }
                    }
                    return true;
                }
                try
                {
                    var goArg = string.Join(" ", parts.Skip(1));
                    var bookmarks = LoadBookmarks();
                    
                    // Check for bookmark/subpath syntax (e.g., "go console/src")
                    string? resolvedPath = null;
                    var slashIdx = goArg.IndexOf('/');
                    var backslashIdx = goArg.IndexOf('\\');
                    var separatorIdx = (slashIdx >= 0 && backslashIdx >= 0) 
                        ? Math.Min(slashIdx, backslashIdx)
                        : Math.Max(slashIdx, backslashIdx);
                    
                    if (separatorIdx > 0)
                    {
                        // Has a path separator - try to resolve bookmark + subpath
                        var bookmarkName = goArg.Substring(0, separatorIdx);
                        var subPath = goArg.Substring(separatorIdx + 1);
                        
                        if (bookmarks.TryGetValue(bookmarkName, out var basePath))
                        {
                            resolvedPath = Path.Combine(basePath, subPath);
                        }
                    }
                    
                    // If no subpath or bookmark not found, try direct bookmark lookup
                    if (resolvedPath == null && bookmarks.TryGetValue(goArg, out var directPath))
                    {
                        resolvedPath = directPath;
                    }
                    
                    if (resolvedPath != null)
                    {
                        if (Directory.Exists(resolvedPath))
                        {
                            Directory.SetCurrentDirectory(resolvedPath);
                            ColorOutput.Success(resolvedPath);
                        }
                        else
                        {
                            ColorOutput.Error($"Path not found: {resolvedPath}");
                        }
                    }
                    else
                    {
                        ColorOutput.Error($"Bookmark not found: {goArg.Split('/', '\\')[0]}");
                        ColorOutput.Info("Use 'go' to see all bookmarks");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"go failed: {ex.Message}");
                }
                return true;

            case "unsave":
            case "unbookmark":
                // unsave name → delete a bookmark
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: unsave <name>");
                    return true;
                }
                try
                {
                    var unsaveName = parts[1];
                    var bookmarksToEdit = LoadBookmarks();
                    if (bookmarksToEdit.ContainsKey(unsaveName))
                    {
                        bookmarksToEdit.Remove(unsaveName);
                        var json = System.Text.Json.JsonSerializer.Serialize(bookmarksToEdit, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                        File.WriteAllText(GetBookmarksPath(), json);
                        ColorOutput.Success($"Deleted bookmark: {unsaveName}");
                    }
                    else
                    {
                        ColorOutput.Error($"Bookmark not found: {unsaveName}");
                        ColorOutput.Info("Use 'go' to see all bookmarks");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"unsave failed: {ex.Message}");
                }
                return true;

            case "history":
                // history file.txt → show edit history for file
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: history <filename>");
                    return true;
                }
                try
                {
                    var histPath = string.Join(" ", parts.Skip(1));
                    if (!Path.IsPathRooted(histPath))
                        histPath = Path.Combine(Directory.GetCurrentDirectory(), histPath);
                    
                    var entries = NSL.StandardLib.FileSystem.FileHistory.Instance.GetHistory(histPath);
                    if (entries.Count == 0)
                    {
                        ColorOutput.Info($"No history for: {Path.GetFileName(histPath)}");
                    }
                    else
                    {
                        ColorOutput.Info($"History for {Path.GetFileName(histPath)} ({entries.Count} entries):");
                        for (int i = 0; i < entries.Count; i++)
                        {
                            var e = entries[i];
                            System.Console.WriteLine($"  [{i}] {e.Timestamp:HH:mm:ss} - {e.Operation} ({e.SizeBytes} bytes) #{e.Id}");
                        }
                        ColorOutput.Info("Use: restore <file> <index> to restore");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"history failed: {ex.Message}");
                }
                return true;

            case "restore":
                // restore file.txt 0 → restore to most recent pre-edit state
                if (parts.Length < 2)
                {
                    ColorOutput.Error("Usage: restore <filename> [index]");
                    ColorOutput.Info("  index 0 = most recent, 1 = second most recent, etc.");
                    return true;
                }
                try
                {
                    var restorePath = parts[1];
                    if (!Path.IsPathRooted(restorePath))
                        restorePath = Path.Combine(Directory.GetCurrentDirectory(), restorePath);
                    
                    var restoreIndex = parts.Length > 2 ? int.Parse(parts[2]) : 0;
                    
                    var historyEntries = NSL.StandardLib.FileSystem.FileHistory.Instance.GetHistory(restorePath);
                    if (historyEntries.Count == 0)
                    {
                        ColorOutput.Error($"No history for: {Path.GetFileName(restorePath)}");
                        return true;
                    }
                    
                    if (restoreIndex < 0 || restoreIndex >= historyEntries.Count)
                    {
                        ColorOutput.Error($"Invalid index. Available: 0-{historyEntries.Count - 1}");
                        return true;
                    }
                    
                    var entry = historyEntries[restoreIndex];
                    if (NSL.StandardLib.FileSystem.FileHistory.Instance.Restore(restorePath, restoreIndex))
                    {
                        ColorOutput.Success($"Restored to {entry.Timestamp:HH:mm:ss} ({entry.Operation})");
                    }
                    else
                    {
                        ColorOutput.Error("Restore failed");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"restore failed: {ex.Message}");
                }
                return true;

            case ".load":
                if (parts.Length < 2)
                {
                    System.Console.WriteLine("Usage: .load filename.nsl");
                    return true;
                }
                try
                {
                    var filename = string.Join(" ", parts.Skip(1));
                    if (File.Exists(filename))
                    {
                        var content = File.ReadAllText(filename);
                        
                        // Execute the file content
                        try
                        {
                            var lexer = new NSLLexer(content);
                            var tokens = lexer.Tokenize();
                            var parser = new NSLParser();
                            var ast = parser.Parse(tokens);
                            var result = interpreter.Execute(ast);
                            
                            System.Console.WriteLine($"Loaded and executed: {filename}");
                            
                            // Show result if it's an expression
                            if (result != null && !IsStatement(content))
                            {
                                System.Console.WriteLine(FormatValue(result));
                            }
                        }
                        catch (NSLParseException ex)
                        {
                            System.Console.WriteLine($"Parse Error in {filename}: {ex.Message}");
                        }
                        catch (NSLRuntimeException ex)
                        {
                            System.Console.WriteLine($"Runtime Error in {filename}: {ex.Message}");
                        }
                    }
                    else
                    {
                        System.Console.WriteLine($"File not found: {filename}");
                    }
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"Error loading file: {ex.Message}");
                }
                return true;

            // ===== GITHUB INTEGRATION COMMANDS =====
            // These manage GitHub connection with encrypted token storage

            case "git-on":
                // git-on → Enable GitHub connection and prompt for token
                try
                {
                    var ghManager = new GitHubCredentialManager();
                    if (ghManager.HasToken())
                    {
                        ghManager.Enable();
                        ColorOutput.Success($"GitHub enabled. Connected as: {ghManager.Username ?? "unknown"}");
                    }
                    else
                    {
                        ColorOutput.Info("GitHub Token Setup");
                        ColorOutput.Info("─────────────────────────────────────────────────");
                        ColorOutput.Info("Create a token at: https://github.com/settings/tokens");
                        ColorOutput.Info("Required scope: 'repo' (for full access)");
                        ColorOutput.Info("");

                        var token = GitHubCredentialManager.ReadHiddenInput("GitHub Token [TOKEN]: ");

                        if (string.IsNullOrWhiteSpace(token))
                        {
                            ColorOutput.Error("No token provided. GitHub not enabled.");
                        }
                        else if (!token.StartsWith("ghp_") && !token.StartsWith("github_pat_"))
                        {
                            ColorOutput.Error("Invalid token format. Should start with 'ghp_' or 'github_pat_'");
                        }
                        else
                        {
                            ColorOutput.Info("Validating token...");
                            var username = ValidateGitHubToken(token);
                            if (username != null)
                            {
                                ghManager.StoreToken(token, username);
                                ColorOutput.Success($"GitHub connected as: {username}");
                                ColorOutput.Success("Token encrypted and saved. Use 'git-off' to disable.");
                            }
                            else
                            {
                                ColorOutput.Error("Token validation failed. Check your token.");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"git-on failed: {ex.Message}");
                }
                return true;

            case "git-off":
                // git-off → Disable GitHub connection (token remains encrypted)
                try
                {
                    var ghManager = new GitHubCredentialManager();
                    ghManager.Disable();
                    ColorOutput.Info("GitHub disabled. Token preserved. Use 'git-on' to re-enable.");
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"git-off failed: {ex.Message}");
                }
                return true;

            case "git-status":
                // git-status → Show GitHub connection status
                try
                {
                    var ghManager = new GitHubCredentialManager();
                    System.Console.WriteLine();
                    ColorOutput.Info("GitHub Connection Status");
                    ColorOutput.Info("─────────────────────────────────────────────────");
                    System.Console.WriteLine($"  Enabled:     {(ghManager.IsEnabled ? "Yes" : "No")}");
                    System.Console.WriteLine($"  Has Token:   {(ghManager.HasToken() ? "Yes (encrypted)" : "No")}");
                    System.Console.WriteLine($"  Username:    {ghManager.Username ?? "Not connected"}");
                    System.Console.WriteLine($"  AI Access:   {ghManager.AiAccessMode}");
                    System.Console.WriteLine();
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"git-status failed: {ex.Message}");
                }
                return true;

            case "git-ai":
                // git-ai on|off|ask|log → Manage AI access to GitHub
                if (parts.Length < 2)
                {
                    ColorOutput.Info("Usage: git-ai <on|off|ask|log>");
                    ColorOutput.Info("  on  - Allow AI full GitHub access");
                    ColorOutput.Info("  off - Block AI GitHub access");
                    ColorOutput.Info("  ask - AI must request permission each time");
                    ColorOutput.Info("  log - Show AI access history");
                    return true;
                }
                try
                {
                    var ghManager = new GitHubCredentialManager();
                    var mode = parts[1].ToLower();

                    if (mode == "log")
                    {
                        var logs = ghManager.GetAiAccessLog();
                        if (logs.Length == 0)
                        {
                            ColorOutput.Info("No AI access logs found.");
                        }
                        else
                        {
                            ColorOutput.Info("AI GitHub Access Log:");
                            ColorOutput.Info("─────────────────────────────────────────────────");
                            foreach (var log in logs)
                                System.Console.WriteLine($"  {log}");
                        }
                    }
                    else if (mode == "on" || mode == "off" || mode == "ask")
                    {
                        ghManager.SetAiAccess(mode);
                        ColorOutput.Success($"AI GitHub access set to: {mode}");
                    }
                    else
                    {
                        ColorOutput.Error($"Unknown mode: {mode}. Use on, off, ask, or log.");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"git-ai failed: {ex.Message}");
                }
                return true;

            case "git-forget":
                // git-forget → Permanently delete stored token
                try
                {
                    ColorOutput.Warning("This will permanently delete your stored GitHub token.");
                    System.Console.Write("Are you sure? (yes/no): ");
                    var confirm = System.Console.ReadLine()?.Trim().ToLower();
                    if (confirm == "yes" || confirm == "y")
                    {
                        var ghManager = new GitHubCredentialManager();
                        ghManager.ForgetToken();
                        ColorOutput.Success("GitHub token deleted. Use 'git-on' to set up again.");
                    }
                    else
                    {
                        ColorOutput.Info("Cancelled. Token preserved.");
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"git-forget failed: {ex.Message}");
                }
                return true;

            case "git-token":
                // git-token → Update/re-enter token
                try
                {
                    var ghManager = new GitHubCredentialManager();
                    ColorOutput.Info("Enter new GitHub token:");
                    var token = GitHubCredentialManager.ReadHiddenInput("GitHub Token [TOKEN]: ");

                    if (!string.IsNullOrWhiteSpace(token))
                    {
                        ColorOutput.Info("Validating token...");
                        var username = ValidateGitHubToken(token);
                        if (username != null)
                        {
                            ghManager.StoreToken(token, username);
                            ColorOutput.Success($"Token updated. Connected as: {username}");
                        }
                        else
                        {
                            ColorOutput.Error("Token validation failed.");
                        }
                    }
                }
                catch (Exception ex)
                {
                    ColorOutput.Error($"git-token failed: {ex.Message}");
                }
                return true;

            default:
                return false;
        }
    }

    /// <summary>
    /// Validate GitHub token by calling the API
    /// </summary>
    private static string? ValidateGitHubToken(string token)
    {
        try
        {
            using var client = new System.Net.Http.HttpClient();
            client.DefaultRequestHeaders.Add("Authorization", $"token {token}");
            client.DefaultRequestHeaders.Add("User-Agent", "NSL");
            client.DefaultRequestHeaders.Add("Accept", "application/vnd.github+json");

            var response = client.GetAsync("https://api.github.com/user").Result;
            if (response.IsSuccessStatusCode)
            {
                var json = response.Content.ReadAsStringAsync().Result;
                var doc = System.Text.Json.JsonDocument.Parse(json);
                return doc.RootElement.GetProperty("login").GetString();
            }
        }
        catch { }
        return null;
    }

    private static void ShowNSLHeader()
    {
        
        
        
        
        
        
        
        System.Console.WriteLine();
    }

    private static void ShowHelp()
    {
        System.Console.WriteLine("NSL Interpreter - Neural Symbolic Language");
        System.Console.WriteLine("AI-native, human-usable. Designed for AI workflows.");
        System.Console.WriteLine();
        System.Console.WriteLine("Usage: nsl [options] [file.nsl]");
        System.Console.WriteLine();
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine("MODES (one primary mode per run)");
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine();
        System.Console.WriteLine("Run Mode (execute code):");
        System.Console.WriteLine("  nsl                       Start interactive REPL");
        System.Console.WriteLine("  nsl script.nsl            Run NSL script file");
        System.Console.WriteLine("  --eval, -e <code>         Execute code directly");
        System.Console.WriteLine("  --pipe, -p                Read code from stdin");
        System.Console.WriteLine();
        System.Console.WriteLine("Analyze Mode (inspect without executing):");
        System.Console.WriteLine("  --ast <code>              Show Abstract Syntax Tree");
        System.Console.WriteLine("  --transform <type> <code> Transform code (vectorize/optimize/minify/prettify)");
        System.Console.WriteLine("  --benchmark               Run performance benchmark");
        System.Console.WriteLine("  --suggest                 Analyze code and show suggested fixes");
        System.Console.WriteLine("  --fix                     Analyze and auto-fix errors/warnings");
        System.Console.WriteLine();
        System.Console.WriteLine("Discovery Mode (tool integration):");
        System.Console.WriteLine("  --capabilities            List all namespaces, functions, signatures");
        System.Console.WriteLine("  --introspect              Show runtime self-awareness report");
        System.Console.WriteLine("  --version, -v             Show version info");
        System.Console.WriteLine("  --help, -h                Show this help message");
        System.Console.WriteLine();
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine("EXECUTION MODIFIERS (combine with Run Mode)");
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine();
        System.Console.WriteLine("Reasoning & Trace:");
        System.Console.WriteLine("  --think                   Show reasoning/thinking trace");
        System.Console.WriteLine("  --trace                   Full execution trace with timestamps");
        System.Console.WriteLine("  --reflect                 Generate self-reflection on execution");
        System.Console.WriteLine("  --explain                 Generate human-readable explanation");
        System.Console.WriteLine();
        System.Console.WriteLine("Memory & Learning:");
        System.Console.WriteLine("  --context <json>          Pass context object as JSON");
        System.Console.WriteLine("  --memory <file>           Persistent memory file path");
        System.Console.WriteLine("  --learn                   Remember patterns from execution");
        System.Console.WriteLine();
        System.Console.WriteLine("Optimization:");
        System.Console.WriteLine("  --optimize                Auto-optimize code before execution");
        System.Console.WriteLine("  --vectorize               Auto-vectorize operations to GPU");
        System.Console.WriteLine("  --gpu                     Auto-initialize GPU context");
        System.Console.WriteLine("  --parallel                Enable parallel execution");
        System.Console.WriteLine("  --sandbox                 Run in sandboxed mode (restricted I/O)");
        System.Console.WriteLine("  --timeout <ms>            Set execution timeout (default: 30000)");
        System.Console.WriteLine();
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine("OUTPUT FORMAT (controls all output)");
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine();
        System.Console.WriteLine("  --json                    Machine-readable JSON output");
        System.Console.WriteLine("  --quiet, -q               Suppress banners (result only)");
        System.Console.WriteLine("  --verbose, -V             Show all details (expanded output)");
        System.Console.WriteLine("  --stream                  Stream output as computed");
        System.Console.WriteLine("  --log                     Save full output to ~/.nsl/logs/");
        System.Console.WriteLine("  --colors                  Create/reset ~/.nsl/colors.json");
        System.Console.WriteLine("  --no-color                Disable colored output");
        System.Console.WriteLine();
        System.Console.WriteLine("Output Precedence Rules:");
        System.Console.WriteLine("  • --json changes output format for ALL flags (think/trace → JSON fields)");
        System.Console.WriteLine("  • --quiet suppresses banners but preserves JSON payload");
        System.Console.WriteLine("  • --stream with --json emits newline-delimited JSON events");
        System.Console.WriteLine("  • --sandbox restricts sys.exec, sys.pipe, file.write, etc.");
        System.Console.WriteLine();
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine("EXAMPLES");
        System.Console.WriteLine("═══════════════════════════════════════════════════════════════");
        System.Console.WriteLine();
        System.Console.WriteLine("Examples:");
        System.Console.WriteLine("  # Basic execution");
        System.Console.WriteLine("  nsl                                    Start interactive REPL");
        System.Console.WriteLine("  nsl script.nsl                         Run NSL script");
        System.Console.WriteLine("  nsl --eval \"2 + 2\"                     Execute and return result");
        System.Console.WriteLine();
        System.Console.WriteLine("  # AI-native execution with full metadata");
        System.Console.WriteLine("  nsl --eval \"x |> transform\" --think --json");
        System.Console.WriteLine("  nsl --eval \"gpu.matmul(a, b)\" --gpu --trace --reflect");
        System.Console.WriteLine();
        System.Console.WriteLine("  # Persistent AI memory");
        System.Console.WriteLine("  nsl --eval \"process(data)\" --memory ai_memory.json --learn");
        System.Console.WriteLine();
        System.Console.WriteLine("  # Code analysis for AI");
        System.Console.WriteLine("  nsl --capabilities --json              Machine-readable capabilities");
        System.Console.WriteLine("  nsl --introspect --json                AI self-awareness as JSON");
        System.Console.WriteLine("  nsl --ast \"x + y\" --json               Parse and analyze code");
        System.Console.WriteLine();
        System.Console.WriteLine("REPL Commands:");
        System.Console.WriteLine("  help     - Show REPL commands");
        System.Console.WriteLine("  syntax   - Show control flow syntax help");
        System.Console.WriteLine("  tokens   - Show token reference");
        System.Console.WriteLine("  clear    - Clear the screen");
        System.Console.WriteLine("  .load <file> - Load and execute NSL file");
        System.Console.WriteLine("  exit     - Exit the interpreter");
        System.Console.WriteLine();
        System.Console.WriteLine("Interactive Shortcuts (REPL only - won't work in scripts):");
        System.Console.WriteLine("  cd             - Go to home directory");
        System.Console.WriteLine("  cd Desktop     - Go to folder (relative path)");
        System.Console.WriteLine("  cd C:\\Users    - Go to folder (absolute path)");
        System.Console.WriteLine("  cd ..          - Go up one level");
        System.Console.WriteLine("  pwd            - Show current directory");
        System.Console.WriteLine("  ls             - List files");
        System.Console.WriteLine("  ls path        - List files in path");
        System.Console.WriteLine("  cat file.txt   - Show file contents");
        System.Console.WriteLine("  mkdir folder   - Create folder");
        System.Console.WriteLine("  rmdir folder   - Delete folder (recursive)");
        System.Console.WriteLine("  touch file.txt - Create empty file");
        System.Console.WriteLine("  rm file.txt    - Delete file");
        System.Console.WriteLine("  mv src dest    - Move file or folder");
        System.Console.WriteLine("  cp src dest    - Copy file");
        System.Console.WriteLine("  whoami         - Show current user");
        System.Console.WriteLine();
        System.Console.WriteLine("Direct Path Access (just type the path):");
        System.Console.WriteLine("  C:\\Projects     - cd into directory");
        System.Console.WriteLine("  E:\\code\\file.cs - Open file in VS Code");
        System.Console.WriteLine("  ./script.nsl   - Open relative file in VS Code");
        System.Console.WriteLine();
        System.Console.WriteLine("Bookmarks (save long paths):");
        System.Console.WriteLine("  save name      - Save current directory as bookmark");
        System.Console.WriteLine("  save name path - Save specific path as bookmark");
        System.Console.WriteLine("  go name        - Jump to saved bookmark");
        System.Console.WriteLine("  go name/sub    - Jump to bookmark + subpath");
        System.Console.WriteLine("  go             - List all bookmarks");
        System.Console.WriteLine("  unsave name    - Delete a bookmark");
        System.Console.WriteLine();
        System.Console.WriteLine("Native File Operations (use these, not shell commands!):");
        System.Console.WriteLine("  file.read(path)         - Read file contents");
        System.Console.WriteLine("  file.write(path, text)  - Write to file");
        System.Console.WriteLine("  file.delete(path)       - Delete file");
        System.Console.WriteLine("  file.copy(src, dst)     - Copy file");
        System.Console.WriteLine("  file.move(src, dst)     - Move file");
        System.Console.WriteLine("  file.exists(path)       - Check if exists");
        System.Console.WriteLine("  dir.create(path)        - Create folder");
        System.Console.WriteLine("  dir.delete(path, true)  - Delete folder (recursive)");
        System.Console.WriteLine("  dir.list(path)          - List contents");
        System.Console.WriteLine("  dir.files(path, \"*.txt\") - List with pattern");
        System.Console.WriteLine();
        System.Console.WriteLine("Other Native Operations:");
        System.Console.WriteLine("  sys.cd(r\"C:\\path\")     - Change directory");
        System.Console.WriteLine("  file.cwd()              - Get current directory");
        System.Console.WriteLine("  clip.copy(text)         - Copy to clipboard");
        System.Console.WriteLine("  clip.paste()            - Paste from clipboard");
        System.Console.WriteLine("  zip.create(src, dst)    - Create zip");
        System.Console.WriteLine("  zip.extract(src, dst)   - Extract zip");
        System.Console.WriteLine();
        System.Console.WriteLine("Shell Commands (for external tools):");
        System.Console.WriteLine("  sys.exec(\"git status\")  - Run command, get {stdout,stderr,code,success}");
        System.Console.WriteLine("  sys.shell(\"dir\")        - Run command, get stdout string");
        System.Console.WriteLine("  sys.pipe(\"dir\",\"find x\") - Chain commands (pipe stdout)");
        System.Console.WriteLine();
        System.Console.WriteLine("Shell Examples:");
        System.Console.WriteLine("  let r = sys.exec(\"ipconfig\")   # r.stdout, r.stderr, r.code");
        System.Console.WriteLine("  let o = sys.shell(\"whoami\")    # Just the output string");
        System.Console.WriteLine("  sys.exec(\"git status\", \"C:\\\\repo\")  # With working dir");
        System.Console.WriteLine("  sys.exec(\"powershell -Command \\\"Get-Date\\\"\")  # PowerShell");
        System.Console.WriteLine();
        System.Console.WriteLine("File Edit History (undo/restore):");
        System.Console.WriteLine("  history file.txt     - Show edit history (last 10 versions)");
        System.Console.WriteLine("  restore file.txt 0   - Restore most recent pre-edit state");
        System.Console.WriteLine("  file.history(path)   - Get history as array");
        System.Console.WriteLine("  file.restore(path,i) - Restore to index i");
        System.Console.WriteLine("  file.historyInfo(p)  - Get {count, capacity, enabled}");
        System.Console.WriteLine("  file.preview(p,txt)  - Dry run: see diff before writing");
        System.Console.WriteLine("  file.detectThrashing(p) - Warn if file oscillates");
        System.Console.WriteLine("  file.writeAnnotated(p,txt,reason,agent) - Write with AI reasoning");
        System.Console.WriteLine("  file.configureHistory(p,{capacity,strategy}) - Per-file config");
        System.Console.WriteLine();
        System.Console.WriteLine("Code Analysis (AI-native):");
        System.Console.WriteLine("  code.read(path)      - Read file with metadata");
        System.Console.WriteLine("  code.metrics(path)   - Get complexity, lines, comments");
        System.Console.WriteLine("  code.symbols(path)   - Find classes, methods, functions");
        System.Console.WriteLine("  code.deps(path)      - Find imports/dependencies");
        System.Console.WriteLine("  code.issues(path)    - Find TODOs, bugs, style issues");
        System.Console.WriteLine("  code.flow(path)      - Analyze control flow (branches, loops)");
        System.Console.WriteLine("  code.extract(p,name) - Extract method/function body");
        System.Console.WriteLine("  code.compare(a,b)    - Structured diff between files");
        System.Console.WriteLine("  code.search(p,regex) - Search with line numbers");
        System.Console.WriteLine();
        System.Console.WriteLine("Phase 4 - Universal Capabilities:");
        System.Console.WriteLine("  ffi.load(path)       - Load native DLL");
        System.Console.WriteLine("  buffer.create(t,n)   - Typed buffer for interop");
        System.Console.WriteLine("  runtime.on/emit      - Event loop & scheduler");
        System.Console.WriteLine("  sim.begin/commit     - Sandbox mode (preview edits)");
        System.Console.WriteLine("  trace.begin/end      - Profiling & timing");
        System.Console.WriteLine("  ml.tensor/add/relu   - ML tensors & neural ops");
        System.Console.WriteLine("  game.createEntity    - ECS game engine primitives");
        System.Console.WriteLine("  gui.messageBox       - GUI dialogs & notifications");
        System.Console.WriteLine();
        System.Console.WriteLine("Phase 5 - Developer Tools:");
        System.Console.WriteLine("  codegen.class/fn     - Generate code in any language");
        System.Console.WriteLine("  ast.parse/emit       - Parse, transform, emit code");
        System.Console.WriteLine("  project.create       - Create projects from templates");
        System.Console.WriteLine("  lsp.symbols/refs     - Language server features");
        System.Console.WriteLine("  meta.eval/compile    - Metaprogramming & runtime eval");
        System.Console.WriteLine("  refactor.rename      - Multi-file refactoring with preview");
        System.Console.WriteLine();
        System.Console.WriteLine("Special prefixes:");
        System.Console.WriteLine("  #<code>  - Show tokens for code");
        System.Console.WriteLine("  !<cmd>   - Execute shell command");
        System.Console.WriteLine();
        System.Console.WriteLine("Quick Tips:");
        System.Console.WriteLine("  \"=\" * 40       - String multiplication");
        System.Console.WriteLine("  json.parse(s)  - Parse JSON with native comparison support");
        System.Console.WriteLine();
        System.Console.WriteLine("Consciousness Operators (AI-native flow):");
        System.Console.WriteLine("  |>   pipe          - Chain transformations left-to-right");
        System.Console.WriteLine("  ~>   awareness     - Introspective flow with self-reference");
        System.Console.WriteLine("  =>>  gradient      - Learning/adjustment with feedback");
        System.Console.WriteLine("  *>   attention     - Focus mechanism with weights");
        System.Console.WriteLine("  +>   superposition - Quantum-like state superposition");
        System.Console.WriteLine();
        System.Console.WriteLine("For MCP integration, use: nsl --eval <code> --json");
        System.Console.WriteLine();
    }

    private static void ShowSyntaxHelp()
    {
        System.Console.WriteLine("NSL Control Flow Syntax:");
        System.Console.WriteLine("  if condition { ... }");
        System.Console.WriteLine("  if (condition) { ... } else { ... }");
        System.Console.WriteLine("  while condition { ... }");
        System.Console.WriteLine("  break    # exit loop");
        System.Console.WriteLine("  continue # skip to next iteration");
        System.Console.WriteLine("  { ... }  # block scoping");
        System.Console.WriteLine();
        System.Console.WriteLine("Multi-line input is supported - just start typing!");
        System.Console.WriteLine("Use Ctrl+C to cancel multi-line input.");
    }

    private static string GetBookmarksPath()
    {
        var nslDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nsl");
        Directory.CreateDirectory(nslDir);
        return Path.Combine(nslDir, "bookmarks.json");
    }

    private static Dictionary<string, string> LoadBookmarks()
    {
        var path = GetBookmarksPath();
        if (File.Exists(path))
        {
            try
            {
                var json = File.ReadAllText(path);
                return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(json) ?? new();
            }
            catch { }
        }
        return new Dictionary<string, string>();
    }

    private static void SaveBookmark(string name, string path)
    {
        var bookmarks = LoadBookmarks();
        bookmarks[name] = path;
        var json = System.Text.Json.JsonSerializer.Serialize(bookmarks, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(GetBookmarksPath(), json);
    }

    private static void ShowTokenReference()
    {
        System.Console.WriteLine("NSL Token Reference:");
        System.Console.WriteLine("  Operators: +, -, *, /, %, ^, =, ==, !=, <, >, <=, >=");
        System.Console.WriteLine("  Keywords: if, else, while, for, break, continue, return, let");
        System.Console.WriteLine("  Consciousness: ◈, ∇, ⊗, Ψ");
        System.Console.WriteLine("  Lambda: λ");
        System.Console.WriteLine("  Chain: ::");
        System.Console.WriteLine();
    }

    private static void ShowTokens(string code)
    {
        if (string.IsNullOrWhiteSpace(code))
            return;

        try
        {
            var lexer = new NSLLexer(code);
            var tokens = lexer.Tokenize();
            var validTokens = tokens.Where(t => t.Type != TokenType.EndOfFile).ToList();
            
            System.Console.WriteLine("┌────────────┬───────────┬──────────┐");
            System.Console.WriteLine("│ Type       │ Value     │ Position │");
            System.Console.WriteLine("├────────────┼───────────┼──────────┤");
            
            foreach (var token in validTokens)
            {
                var type = token.Type.ToString().PadRight(10);
                var value = (token.Value?.ToString() ?? "").PadRight(9);
                var position = $"{token.Line}:{token.Column}".PadRight(8);
                System.Console.WriteLine($"│ {type} │ {value} │ {position} │");
            }
            
            System.Console.WriteLine("└────────────┴───────────┴──────────┘");
        }
        catch (Exception ex)
        {
            System.Console.WriteLine($"Tokenization Error: {ex.Message}");
        }
    }

    private static void ExecuteShellCommand(string command)
    {
        try
        {
            var processInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = $"/c {command}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = System.Diagnostics.Process.Start(processInfo);
            if (process != null)
            {
                var output = process.StandardOutput.ReadToEnd();
                var error = process.StandardError.ReadToEnd();
                process.WaitForExit();

                if (!string.IsNullOrEmpty(output))
                    System.Console.WriteLine(output);
                if (!string.IsNullOrEmpty(error))
                    System.Console.WriteLine($"Error: {error}");
            }
        }
        catch (Exception ex)
        {
            System.Console.WriteLine($"Shell command error: {ex.Message}");
        }
    }
}
