using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NSL.Core;
using NSL.Core.AST;
using NSLTokenType = NSL.Core.Tokens.TokenType;

namespace NSL.TypeSystem;

/// <summary>
/// Interface for parsing module source code - implemented by Compiler
/// </summary>
public interface IModuleParser
{
    NSLASTNode Parse(string source, string filePath);
}

/// <summary>
/// Resolves and loads NSL modules from the file system
/// </summary>
public class ModuleResolver
{
    private readonly List<string> _searchPaths = new();
    private readonly Dictionary<string, LoadedModule> _loadedModules = new();
    private readonly HashSet<string> _loadingModules = new(); // Detect circular imports
    private readonly IModuleParser _parser;

    public ModuleResolver(IModuleParser parser)
    {
        _parser = parser ?? throw new ArgumentNullException(nameof(parser));
    }

    /// <summary>
    /// Represents a loaded module with its exports
    /// </summary>
    public class LoadedModule
    {
        public string FilePath { get; init; } = "";
        public string ModuleName { get; init; } = "";
        public NSLASTNode AST { get; init; } = null!;
        public Dictionary<string, ExportedSymbol> Exports { get; } = new();
        public bool IsLoaded { get; set; }
    }

    /// <summary>
    /// Represents an exported symbol from a module
    /// </summary>
    public class ExportedSymbol
    {
        public string Name { get; init; } = "";
        public NSLType Type { get; init; } = NSLTypes.Any;
        public NSLASTNode? Declaration { get; init; }
        public SymbolKind Kind { get; init; }
    }

    public enum SymbolKind
    {
        Variable,
        Function,
        Type,
        Constant
    }

    /// <summary>
    /// Add a path to search for modules
    /// </summary>
    public void AddSearchPath(string path)
    {
        if (Directory.Exists(path) && !_searchPaths.Contains(path))
        {
            _searchPaths.Add(path);
        }
    }

    /// <summary>
    /// Set up default search paths including stdlib
    /// </summary>
    public void SetupDefaultPaths(string? sourceFilePath = null)
    {
        // Add current directory
        _searchPaths.Add(Directory.GetCurrentDirectory());

        // Add source file directory if provided
        if (!string.IsNullOrEmpty(sourceFilePath))
        {
            var sourceDir = Path.GetDirectoryName(sourceFilePath);
            if (!string.IsNullOrEmpty(sourceDir) && Directory.Exists(sourceDir))
            {
                _searchPaths.Add(sourceDir);
            }
        }

        // Find stdlib directory
        var stdlibPath = FindStdlibPath();
        if (!string.IsNullOrEmpty(stdlibPath))
        {
            _searchPaths.Add(stdlibPath);
        }
    }

    private string? FindStdlibPath()
    {
        // Try various locations for stdlib
        var candidates = new[]
        {
            // Relative to current directory
            Path.Combine(Directory.GetCurrentDirectory(), "stdlib"),
            // Relative to executable
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "stdlib"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "stdlib"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "stdlib"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "stdlib"),
            // Common install locations
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "NSL", "stdlib"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nsl", "stdlib"),
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate))
            {
                return Path.GetFullPath(candidate);
            }
        }

        return null;
    }

    /// <summary>
    /// Resolve a module path to a file path
    /// </summary>
    public string? ResolveModulePath(IReadOnlyList<string> modulePath)
    {
        if (modulePath == null || modulePath.Count == 0)
            return null;

        // Convert module path to file path
        // e.g., ["math"] -> "math/mod.nsl" or "math.nsl"
        // e.g., ["collections", "list"] -> "collections/list.nsl" or "collections/list/mod.nsl"

        var pathParts = modulePath.ToArray();

        foreach (var searchPath in _searchPaths)
        {
            // Try: path/mod.nsl (for module directories)
            var modFilePath = Path.Combine(searchPath, Path.Combine(pathParts), "mod.nsl");
            if (File.Exists(modFilePath))
                return modFilePath;

            // Try: path.nsl (for single file modules)
            var singleFilePath = Path.Combine(searchPath, Path.Combine(pathParts) + ".nsl");
            if (File.Exists(singleFilePath))
                return singleFilePath;

            // Try: path (last part).nsl in parent directory
            if (pathParts.Length > 1)
            {
                var parentPath = Path.Combine(searchPath, Path.Combine(pathParts.Take(pathParts.Length - 1).ToArray()));
                var childFile = Path.Combine(parentPath, pathParts.Last() + ".nsl");
                if (File.Exists(childFile))
                    return childFile;
            }
        }

        return null;
    }

    /// <summary>
    /// Load and parse a module file
    /// </summary>
    public LoadedModule? LoadModule(IReadOnlyList<string> modulePath)
    {
        var moduleKey = string.Join("::", modulePath);

        // Check if already loaded
        if (_loadedModules.TryGetValue(moduleKey, out var existing))
            return existing;

        // Check for circular imports
        if (_loadingModules.Contains(moduleKey))
        {
            throw new InvalidOperationException($"Circular import detected: {moduleKey}");
        }

        // Resolve file path
        var filePath = ResolveModulePath(modulePath);
        if (filePath == null)
            return null;

        _loadingModules.Add(moduleKey);

        try
        {
            // Read and parse the module using the injected parser
            var source = File.ReadAllText(filePath);
            var ast = _parser.Parse(source, filePath);

            var module = new LoadedModule
            {
                FilePath = filePath,
                ModuleName = moduleKey,
                AST = ast,
                IsLoaded = true
            };

            // Extract exports from the AST
            ExtractExports(module, ast);

            _loadedModules[moduleKey] = module;
            return module;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load module '{moduleKey}' from '{filePath}': {ex.Message}", ex);
        }
        finally
        {
            _loadingModules.Remove(moduleKey);
        }
    }

    /// <summary>
    /// Load a module from an explicit file path
    /// </summary>
    public LoadedModule? LoadModuleFromFile(string filePath)
    {
        if (!File.Exists(filePath))
            return null;

        var moduleKey = Path.GetFullPath(filePath);

        if (_loadedModules.TryGetValue(moduleKey, out var existing))
            return existing;

        try
        {
            var source = File.ReadAllText(filePath);
            var ast = _parser.Parse(source, filePath);

            var module = new LoadedModule
            {
                FilePath = filePath,
                ModuleName = Path.GetFileNameWithoutExtension(filePath),
                AST = ast,
                IsLoaded = true
            };

            ExtractExports(module, ast);

            _loadedModules[moduleKey] = module;
            return module;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load module from '{filePath}': {ex.Message}", ex);
        }
    }

    private void ExtractExports(LoadedModule module, NSLASTNode ast)
    {
        // Track which names are marked as pub within module blocks
        var pubSymbols = new HashSet<string>();
        // Track explicit export statements
        var explicitExports = new HashSet<string>();
        // Track all declared symbols with their types
        var declarations = new Dictionary<string, (NSLType Type, NSLASTNode Node, SymbolKind Kind)>();

        CollectDeclarations(ast, pubSymbols, explicitExports, declarations);

        // Export symbols that are either:
        // 1. Marked with pub (in a module block)
        // 2. Listed in an export statement
        // 3. At top level without module block (implicitly exported)

        foreach (var (name, (type, node, kind)) in declarations)
        {
            if (pubSymbols.Contains(name) || explicitExports.Contains(name) || explicitExports.Count == 0)
            {
                module.Exports[name] = new ExportedSymbol
                {
                    Name = name,
                    Type = type,
                    Declaration = node,
                    Kind = kind
                };
            }
        }
    }

    private void CollectDeclarations(
        NSLASTNode node,
        HashSet<string> pubSymbols,
        HashSet<string> explicitExports,
        Dictionary<string, (NSLType Type, NSLASTNode Node, SymbolKind Kind)> declarations,
        bool inModuleBlock = false)
    {
        switch (node)
        {
            case NSLBlockNode block:
                foreach (var stmt in block.Statements)
                {
                    CollectDeclarations(stmt, pubSymbols, explicitExports, declarations, inModuleBlock);
                }
                break;

            case NSLModuleNode moduleNode:
                // Enter module block - contents may have pub modifiers
                if (moduleNode.Body != null)
                {
                    CollectDeclarations(moduleNode.Body, pubSymbols, explicitExports, declarations, inModuleBlock: true);
                }
                break;

            case NSLExportNode exportNode:
                if (exportNode.Items != null)
                {
                    foreach (var item in exportNode.Items)
                    {
                        // Use the internal Name to match against declarations
                        // PublicName is the externally visible name
                        explicitExports.Add(item.Name);
                    }
                }
                if (exportNode.Declaration != null)
                {
                    // pub fn / pub let - mark as public and collect
                    var declName = GetDeclarationName(exportNode.Declaration);
                    if (declName != null)
                    {
                        pubSymbols.Add(declName);
                        CollectDeclarations(exportNode.Declaration, pubSymbols, explicitExports, declarations, inModuleBlock);
                    }
                }
                break;

            case NSLFunctionNode func:
                var funcType = InferFunctionType(func);
                declarations[func.Name] = (funcType, func, SymbolKind.Function);
                // Note: functions are marked as public via NSLExportNode wrapping, not via IsPublic property
                break;

            case NSLVariableDeclarationNode varDecl:
                var varType = InferVariableType(varDecl);
                var kind = varDecl.IsMutable ? SymbolKind.Variable : SymbolKind.Constant;
                declarations[varDecl.Name] = (varType, varDecl, kind);
                // Note: variables are marked as public via NSLExportNode wrapping
                break;

            case NSLTypeAliasNode typeAlias:
                declarations[typeAlias.Name] = (NSLTypes.Any, typeAlias, SymbolKind.Type);
                break;
        }
    }

    private string? GetDeclarationName(NSLASTNode node)
    {
        return node switch
        {
            NSLFunctionNode func => func.Name,
            NSLVariableDeclarationNode varDecl => varDecl.Name,
            NSLTypeAliasNode typeAlias => typeAlias.Name,
            _ => null
        };
    }

    private NSLType InferFunctionType(NSLFunctionNode func)
    {
        // Create a function type from the function definition
        var paramTypes = func.Parameters.Select(p =>
        {
            // If parameter has type annotation, use it
            // For now, default to Any
            return NSLTypes.Any;
        }).ToArray();

        // Return type - for now default to Any (full type checking happens later)
        var returnType = NSLTypes.Any;

        return new NSLFunctionType(paramTypes, returnType);
    }

    private NSLType InferVariableType(NSLVariableDeclarationNode varDecl)
    {
        // Try to infer type from initializer
        if (varDecl.Value != null)
        {
            return InferExpressionType(varDecl.Value);
        }
        return NSLTypes.Any;
    }

    private NSLType InferExpressionType(NSLASTNode expr)
    {
        return expr switch
        {
            NSLLiteralNode lit => lit.ValueType switch
            {
                NSLTokenType.Number => NSLTypes.Number,
                NSLTokenType.Integer => NSLTypes.Int,
                NSLTokenType.String => NSLTypes.String,
                NSLTokenType.Boolean => NSLTypes.Bool,
                _ => NSLTypes.Any
            },
            _ => NSLTypes.Any
        };
    }

    /// <summary>
    /// Get all loaded modules
    /// </summary>
    public IReadOnlyDictionary<string, LoadedModule> LoadedModules => _loadedModules;

    /// <summary>
    /// Check if a module is already loaded
    /// </summary>
    public bool IsModuleLoaded(IReadOnlyList<string> modulePath)
    {
        var moduleKey = string.Join("::", modulePath);
        return _loadedModules.ContainsKey(moduleKey);
    }
}
