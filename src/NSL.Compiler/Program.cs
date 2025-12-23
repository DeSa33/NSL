using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NSL.Lexer;
using NSL.Parser;
using NSL.TypeSystem;
using NSL.CodeGen;
using NSL.Core.AutoFix;

namespace NSL.Compiler;

/// <summary>
/// NSL Compiler (nslc) - Compiles NSL source code to native executables
///
/// NSL is an AI-native programming language designed for AI to code with fewer errors.
/// Features: Type inference, immutable by default, pipeline operators, pattern matching,
/// safe navigation, result types, and built-in ML/AI primitives.
/// </summary>
class Program
{
    static int Main(string[] args)
    {
        var options = ParseArgs(args);

        if (options.ShowHelp)
        {
            PrintHelp();
            return 0;
        }

        if (options.ShowVersion)
        {
            PrintVersion();
            return 0;
        }

        if (string.IsNullOrEmpty(options.InputFile))
        {
            Console.Error.WriteLine("Error: No input file specified");
            Console.Error.WriteLine("Usage: nslc <source.nsl> [options]");
            return 1;
        }

        if (!File.Exists(options.InputFile))
        {
            Console.Error.WriteLine($"Error: File not found: {options.InputFile}");
            return 1;
        }

        try
        {
            return Compile(options);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Compilation failed: {ex.Message}");
            if (options.Verbose)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
            return 1;
        }
    }

    static int Compile(CompilerOptions options)
    {
        var sourceCode = File.ReadAllText(options.InputFile!);
        var moduleName = Path.GetFileNameWithoutExtension(options.InputFile);

        // Auto-fix mode: analyze and fix source before compiling
        if (options.AutoFix || options.SuggestFixes)
        {
            var autoFix = new NSLAutoFix(sourceCode);
            autoFix.Analyze();

            if (autoFix.Fixes.Count > 0)
            {
                Console.WriteLine(autoFix.GetFixSummary());

                if (options.AutoFix)
                {
                    // Apply fixes and rewrite the source file
                    var fixedSource = autoFix.ApplyFixes(FixCategory.Warning);
                    if (fixedSource != sourceCode)
                    {
                        File.WriteAllText(options.InputFile!, fixedSource);
                        sourceCode = fixedSource;
                        Console.WriteLine($"Auto-fixed {autoFix.Fixes.Count(f => f.Category <= FixCategory.Warning)} issue(s) in {options.InputFile}");
                    }
                }
                else
                {
                    // Just show suggestions, don't apply
                    Console.WriteLine("Run with --fix to automatically apply fixes.");
                    if (!options.Verbose)
                        return 0; // Exit if only suggesting
                }
            }
            else
            {
                Console.WriteLine("No fixes needed - code looks good!");
                if (options.SuggestFixes && !options.Verbose)
                    return 0;
            }
        }

        if (options.Verbose)
        {
            Console.WriteLine($"Compiling {options.InputFile}...");
        }

        // Stage 1: Lexical Analysis
        if (options.Verbose) Console.WriteLine("  [1/5] Lexing...");

        var lexer = new NSLLexer(sourceCode, options.InputFile);
        var tokens = lexer.Tokenize();

        if (options.EmitTokens)
        {
            Console.WriteLine("=== Tokens ===");
            foreach (var token in tokens)
            {
                Console.WriteLine($"  {token}");
            }
        }

        // Stage 2: Parsing
        if (options.Verbose) Console.WriteLine("  [2/5] Parsing...");

        var parser = new NSLParser();
        var ast = parser.Parse(tokens);

        if (options.EmitAST)
        {
            Console.WriteLine("=== AST ===");
            PrintAST(ast, 0);
        }

        // Stage 3: Type Checking
        if (options.Verbose) Console.WriteLine("  [3/5] Type checking...");

        var typeChecker = new TypeChecker();
        typeChecker.RegisterBuiltins();

        // Set up module resolver for imports
        var moduleParser = new NSLModuleParser();
        var moduleResolver = new ModuleResolver(moduleParser);
        moduleResolver.SetupDefaultPaths(options.InputFile);

        // Also add the stdlib path explicitly if it exists next to the source
        var sourceDir = Path.GetDirectoryName(options.InputFile);
        if (!string.IsNullOrEmpty(sourceDir))
        {
            var stdlibPath = Path.Combine(sourceDir, "stdlib");
            if (Directory.Exists(stdlibPath))
            {
                moduleResolver.AddSearchPath(stdlibPath);
            }
        }

        typeChecker.SetModuleResolver(moduleResolver);

        if (options.Verbose)
        {
            Console.WriteLine($"    Module search paths configured");
        }

        var resultType = typeChecker.Check(ast);

        if (options.Verbose && moduleResolver.LoadedModules.Count > 0)
        {
            Console.WriteLine($"    Loaded {moduleResolver.LoadedModules.Count} module(s)");
        }

        if (typeChecker.HasErrors)
        {
            Console.Error.WriteLine("Type errors:");
            foreach (var error in typeChecker.Errors)
            {
                Console.Error.WriteLine($"  {error}");
            }
            return 1;
        }

        if (options.TypeCheckOnly)
        {
            Console.WriteLine("Type check passed.");
            return 0;
        }

        // Stage 4: LLVM IR Generation
        if (options.Verbose) Console.WriteLine("  [4/5] Generating LLVM IR...");

        using var codeGen = new LLVMCodeGen(moduleName!, typeChecker);
        codeGen.SetModuleResolver(moduleResolver);
        codeGen.ProcessImportedModules();  // Generate code for imported modules first
        codeGen.Generate(ast);
        codeGen.GenerateMain();

        if (!codeGen.Verify(out var verifyError))
        {
            Console.Error.WriteLine($"LLVM verification failed: {verifyError}");
            return 1;
        }

        // Output IR if requested
        if (options.EmitLLVM)
        {
            var irPath = options.OutputFile != null
                ? Path.ChangeExtension(options.OutputFile, ".ll")
                : moduleName + ".ll";
            codeGen.WriteIR(irPath);
            Console.WriteLine($"LLVM IR written to: {irPath}");

            if (options.Verbose)
            {
                Console.WriteLine("=== LLVM IR ===");
                Console.WriteLine(codeGen.GetIR());
            }
        }

        // Stage 5: Native Code Generation
        if (!options.EmitLLVM || options.OutputFile != null)
        {
            if (options.Verbose) Console.WriteLine("  [5/5] Generating native code...");

            var outputPath = options.OutputFile ?? moduleName;
            if (!outputPath!.EndsWith(".exe") && Environment.OSVersion.Platform == PlatformID.Win32NT)
            {
                outputPath += ".exe";
            }

            // Write bitcode for native compilation
            var bcPath = Path.GetTempFileName() + ".bc";
            codeGen.WriteBitcode(bcPath);

            // Compile to native executable (tries multiple strategies)
            var compileResult = RunNativeCompilation(bcPath, outputPath, options.OptimizationLevel, options.Verbose);

            // Cleanup
            try { File.Delete(bcPath); } catch { }

            if (compileResult != 0)
            {
                Console.Error.WriteLine("Native code generation failed.");
                Console.Error.WriteLine("Tried: clang+MinGW, clang+gcc, llc+gcc, clang+MSVC");
                Console.Error.WriteLine("Install one of: winget install LLVM.LLVM");
                Console.Error.WriteLine("                winget install BrechtSanders.WinLibs.POSIX.UCRT");
                return compileResult;
            }

            Console.WriteLine($"Compiled successfully: {outputPath}");
        }

        return 0;
    }

    static int RunNativeCompilation(string inputPath, string outputPath, int optLevel, bool verbose)
    {
        // Strategy 1: Try clang with MinGW target (doesn't need MSVC)
        var clangPath = FindExecutable("clang", @"C:\Program Files\LLVM\bin\clang.exe");
        if (clangPath != null)
        {
            if (verbose) Console.WriteLine("    Trying clang with MinGW target...");
            var result = RunProcess(clangPath,
                $"--target=x86_64-w64-mingw32 {inputPath} -o {outputPath} -O{optLevel}");
            if (result == 0) return 0;

            // Strategy 2: Try clang to object, then gcc to link
            if (verbose) Console.WriteLine("    Trying clang compile + gcc link...");
            var objPath = Path.GetTempFileName() + ".o";
            result = RunProcess(clangPath, $"-c {inputPath} -o {objPath} -O{optLevel}");
            if (result == 0)
            {
                var gccPath = FindExecutable("gcc", null);
                if (gccPath != null)
                {
                    result = RunProcess(gccPath, $"{objPath} -o {outputPath} -lm");
                    try { File.Delete(objPath); } catch { }
                    if (result == 0) return 0;
                }
            }
        }

        // Strategy 3: Use gcc directly with the IR file
        var gcc = FindExecutable("gcc", null);
        if (gcc != null)
        {
            if (verbose) Console.WriteLine("    Trying gcc with LLVM IR...");
            // gcc can compile LLVM IR if it has the dragonegg plugin, but more reliably
            // we can use llc to convert to assembly first, or just try direct compilation

            // First, let's try if there's an llc available
            var llcPath = FindExecutable("llc", @"C:\Program Files\LLVM\bin\llc.exe");
            if (llcPath != null)
            {
                var asmPath = Path.GetTempFileName() + ".s";
                var result = RunProcess(llcPath, $"{inputPath} -o {asmPath} -O{optLevel}");
                if (result == 0)
                {
                    result = RunProcess(gcc, $"{asmPath} -o {outputPath} -lm");
                    try { File.Delete(asmPath); } catch { }
                    if (result == 0) return 0;
                }
            }
        }

        // Strategy 4: Last resort - clang with MSVC (needs VS Build Tools)
        if (clangPath != null)
        {
            if (verbose) Console.WriteLine("    Trying clang with MSVC target (requires VS Build Tools)...");
            var result = RunProcess(clangPath, $"{inputPath} -o {outputPath} -O{optLevel}");
            if (result == 0) return 0;
        }

        return 1;
    }

    static string? FindExecutable(string name, string? fallbackPath)
    {
        // Check if in PATH
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = Environment.OSVersion.Platform == PlatformID.Win32NT ? "where" : "which",
                    Arguments = name,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };
            process.Start();
            var output = process.StandardOutput.ReadLine();
            process.WaitForExit();
            if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
                return output;
        }
        catch { }

        // Try fallback path
        if (fallbackPath != null && File.Exists(fallbackPath))
            return fallbackPath;

        return null;
    }

    static int RunProcess(string fileName, string arguments)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = fileName,
                    Arguments = arguments,
                    UseShellExecute = false,
                    RedirectStandardError = true,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                }
            };
            process.Start();
            var stderr = process.StandardError.ReadToEnd();
            var stdout = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                // Only show errors for final attempt, not intermediate failures
                if (!string.IsNullOrEmpty(stderr))
                    Console.Error.WriteLine(stderr);
            }

            return process.ExitCode;
        }
        catch
        {
            return 1;
        }
    }

    static void PrintAST(NSL.Core.AST.NSLASTNode node, int indent)
    {
        var prefix = new string(' ', indent * 2);
        Console.WriteLine($"{prefix}{node}");

        // Recursively print children based on node type
        switch (node)
        {
            case NSL.Core.AST.NSLBlockNode block:
                foreach (var stmt in block.Statements)
                    PrintAST(stmt, indent + 1);
                break;
            case NSL.Core.AST.NSLFunctionNode func:
                PrintAST(func.Body, indent + 1);
                break;
            case NSL.Core.AST.NSLIfNode ifNode:
                Console.WriteLine($"{prefix}  Condition:");
                PrintAST(ifNode.Condition, indent + 2);
                Console.WriteLine($"{prefix}  Then:");
                PrintAST(ifNode.ThenBranch, indent + 2);
                if (ifNode.ElseBranch != null)
                {
                    Console.WriteLine($"{prefix}  Else:");
                    PrintAST(ifNode.ElseBranch, indent + 2);
                }
                break;
            case NSL.Core.AST.NSLBinaryOperationNode binary:
                PrintAST(binary.Left, indent + 1);
                PrintAST(binary.Right, indent + 1);
                break;
        }
    }

    static CompilerOptions ParseArgs(string[] args)
    {
        var options = new CompilerOptions();

        for (int i = 0; i < args.Length; i++)
        {
            var arg = args[i];

            switch (arg)
            {
                case "-h":
                case "--help":
                    options.ShowHelp = true;
                    break;
                case "-v":
                case "--version":
                    options.ShowVersion = true;
                    break;
                case "-V":
                case "--verbose":
                    options.Verbose = true;
                    break;
                case "-o":
                case "--output":
                    if (i + 1 < args.Length)
                        options.OutputFile = args[++i];
                    break;
                case "--emit-llvm":
                    options.EmitLLVM = true;
                    break;
                case "--emit-tokens":
                    options.EmitTokens = true;
                    break;
                case "--emit-ast":
                    options.EmitAST = true;
                    break;
                case "--type-check":
                    options.TypeCheckOnly = true;
                    break;
                case "-O0":
                    options.OptimizationLevel = 0;
                    break;
                case "-O1":
                    options.OptimizationLevel = 1;
                    break;
                case "-O2":
                    options.OptimizationLevel = 2;
                    break;
                case "-O3":
                    options.OptimizationLevel = 3;
                    break;
                case "--fix":
                    options.AutoFix = true;
                    break;
                case "--suggest":
                    options.SuggestFixes = true;
                    break;
                default:
                    if (!arg.StartsWith("-") && options.InputFile == null)
                        options.InputFile = arg;
                    break;
            }
        }

        return options;
    }

    static void PrintHelp()
    {
        Console.WriteLine(@"
NSL Compiler (nslc) - AI-Native Programming Language Compiler

USAGE:
    nslc <source.nsl> [options]

OPTIONS:
    -o, --output <file>    Output file path
    -h, --help             Show this help message
    -v, --version          Show version information
    -V, --verbose          Verbose output

    --emit-llvm            Output LLVM IR (.ll file)
    --emit-tokens          Print lexer tokens
    --emit-ast             Print abstract syntax tree
    --type-check           Only perform type checking

    -O0                    No optimization
    -O1                    Basic optimization
    -O2                    Standard optimization (default)
    -O3                    Aggressive optimization

    --fix                  Auto-fix errors and warnings in source
    --suggest              Show suggested fixes without applying

EXAMPLES:
    nslc hello.nsl                    # Compile to hello.exe
    nslc hello.nsl -o myapp           # Compile to myapp.exe
    nslc hello.nsl --emit-llvm        # Output LLVM IR
    nslc hello.nsl -V                 # Verbose compilation

AI-FRIENDLY FEATURES:
    - Immutable by default (use 'mut' for mutable)
    - Type inference with optional hints
    - Pipeline operator (|>) for data flow
    - Safe navigation (?.) and null coalescing (??)
    - Pattern matching with match/case
    - Result types (ok/err) for explicit error handling
    - Built-in vec, mat, tensor types for ML
    - Matrix multiply operator (@)

DOCUMENTATION:
    https://github.com/your-repo/nsl
");
    }

    static void PrintVersion()
    {
        Console.WriteLine(@"
NSL Compiler (nslc) version 0.1.0
AI-Native Programming Language

Built with:
  - .NET 8.0
  - LLVM 16.0 (via LLVMSharp)

Copyright (c) 2024 NSL Contributors
");
    }
}

class CompilerOptions
{
    public string? InputFile { get; set; }
    public string? OutputFile { get; set; }
    public bool ShowHelp { get; set; }
    public bool ShowVersion { get; set; }
    public bool Verbose { get; set; }
    public bool EmitLLVM { get; set; }
    public bool EmitTokens { get; set; }
    public bool EmitAST { get; set; }
    public bool TypeCheckOnly { get; set; }
    public int OptimizationLevel { get; set; } = 2;
    public bool AutoFix { get; set; }           // Automatically fix errors
    public bool SuggestFixes { get; set; }      // Show suggested fixes without applying
}

/// <summary>
/// Implementation of IModuleParser for the compiler
/// </summary>
class NSLModuleParser : IModuleParser
{
    public NSL.Core.AST.NSLASTNode Parse(string source, string filePath)
    {
        var lexer = new NSLLexer(source, filePath);
        var tokens = lexer.Tokenize();
        var parser = new NSLParser();
        return parser.Parse(tokens);
    }
}
