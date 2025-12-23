using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;

namespace NSL.LanguageServer
{
    /// <summary>
    /// Global symbol table for cross-file symbol resolution
    /// </summary>
    public class SymbolTable
    {
        private readonly ConcurrentDictionary<DocumentUri, List<SymbolInfo>> _fileSymbols = new();
        private readonly ConcurrentDictionary<string, List<Location>> _references = new();

        /// <summary>
        /// Update symbols for a file
        /// </summary>
        public void UpdateFile(DocumentUri uri, List<SymbolInfo> symbols)
        {
            _fileSymbols[uri] = symbols;

            // Update references
            foreach (var symbol in symbols)
            {
                if (!_references.ContainsKey(symbol.Name))
                    _references[symbol.Name] = new List<Location>();

                // Add definition as a reference
                if (symbol.Location != null)
                {
                    var existing = _references[symbol.Name];
                    if (!existing.Any(l => l.Uri == symbol.Location.Uri &&
                                          l.Range.Start.Line == symbol.Location.Range.Start.Line))
                    {
                        existing.Add(symbol.Location);
                    }
                }
            }
        }

        /// <summary>
        /// Remove all symbols for a file
        /// </summary>
        public void RemoveFile(DocumentUri uri)
        {
            _fileSymbols.TryRemove(uri, out _);
        }

        /// <summary>
        /// Get all symbols across all files
        /// </summary>
        public IEnumerable<SymbolInfo> GetAllSymbols()
        {
            return _fileSymbols.Values.SelectMany(s => s);
        }

        /// <summary>
        /// Get symbols for a specific file
        /// </summary>
        public IEnumerable<SymbolInfo> GetFileSymbols(DocumentUri uri)
        {
            return _fileSymbols.TryGetValue(uri, out var symbols) ? symbols : Enumerable.Empty<SymbolInfo>();
        }

        /// <summary>
        /// Find a symbol by name
        /// </summary>
        public SymbolInfo? FindSymbol(string name)
        {
            return GetAllSymbols().FirstOrDefault(s => s.Name == name);
        }

        /// <summary>
        /// Find all symbols matching a query
        /// </summary>
        public IEnumerable<SymbolInfo> Search(string query)
        {
            if (string.IsNullOrWhiteSpace(query))
                return GetAllSymbols();

            return GetAllSymbols().Where(s =>
                s.Name.Contains(query, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Find all references to a symbol
        /// </summary>
        public IEnumerable<Location> FindReferences(string name)
        {
            return _references.TryGetValue(name, out var refs) ? refs : Enumerable.Empty<Location>();
        }

        /// <summary>
        /// Add a reference location
        /// </summary>
        public void AddReference(string name, Location location)
        {
            if (!_references.ContainsKey(name))
                _references[name] = new List<Location>();

            var existing = _references[name];
            if (!existing.Any(l => l.Uri == location.Uri &&
                                  l.Range.Start.Line == location.Range.Start.Line &&
                                  l.Range.Start.Character == location.Range.Start.Character))
            {
                existing.Add(location);
            }
        }

        /// <summary>
        /// Clear all symbols
        /// </summary>
        public void Clear()
        {
            _fileSymbols.Clear();
            _references.Clear();
        }
    }

    /// <summary>
    /// Information about a symbol (function, variable, type, etc.)
    /// </summary>
    public class SymbolInfo
    {
        /// <summary>Symbol name</summary>
        public string Name { get; init; } = "";

        /// <summary>Symbol kind</summary>
        public SymbolKind Kind { get; init; }

        /// <summary>Type annotation</summary>
        public string? Type { get; init; }

        /// <summary>Symbol location (definition)</summary>
        public Location? Location { get; init; }

        /// <summary>Documentation/description</summary>
        public string? Documentation { get; init; }

        /// <summary>Container name (class, module, etc.)</summary>
        public string? ContainerName { get; init; }

        /// <summary>Whether the symbol is exported/public</summary>
        public bool IsPublic { get; init; }

        /// <summary>Parameters (for functions)</summary>
        public List<ParameterInfo>? Parameters { get; init; }

        /// <summary>Return type (for functions)</summary>
        public string? ReturnType { get; init; }

        /// <summary>
        /// Returns a string representation of the symbol
        /// </summary>
        /// <returns>Symbol kind and name</returns>
        public override string ToString() => $"{Kind}: {Name}";
    }

    /// <summary>
    /// Information about a function parameter
    /// </summary>
    public class ParameterInfo
    {
        /// <summary>Parameter name</summary>
        public string Name { get; init; } = "";

        /// <summary>Parameter type</summary>
        public string? Type { get; init; }

        /// <summary>Default value</summary>
        public string? DefaultValue { get; init; }

        /// <summary>Parameter documentation</summary>
        public string? Documentation { get; init; }
    }
}
