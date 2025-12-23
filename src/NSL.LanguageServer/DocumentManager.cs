using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace NSL.LanguageServer
{
    /// <summary>
    /// Manages open documents and their state
    /// </summary>
    public class DocumentManager
    {
        private readonly ConcurrentDictionary<DocumentUri, DocumentState> _documents = new();

        /// <summary>
        /// Open or update a document
        /// </summary>
        public void UpdateDocument(DocumentUri uri, string content, int? version = null)
        {
            _documents.AddOrUpdate(
                uri,
                _ => new DocumentState(uri, content, version ?? 1),
                (_, existing) => new DocumentState(uri, content, version ?? existing.Version + 1)
            );
        }

        /// <summary>
        /// Close a document
        /// </summary>
        public void CloseDocument(DocumentUri uri)
        {
            _documents.TryRemove(uri, out _);
        }

        /// <summary>
        /// Get document content
        /// </summary>
        public DocumentState? GetDocument(DocumentUri uri)
        {
            return _documents.TryGetValue(uri, out var doc) ? doc : null;
        }

        /// <summary>
        /// Get all open documents
        /// </summary>
        public IEnumerable<DocumentState> GetAllDocuments()
        {
            return _documents.Values;
        }

        /// <summary>
        /// Check if document is open
        /// </summary>
        public bool IsOpen(DocumentUri uri)
        {
            return _documents.ContainsKey(uri);
        }

        /// <summary>
        /// Apply incremental changes to a document
        /// </summary>
        public void ApplyChanges(DocumentUri uri, IEnumerable<TextDocumentContentChangeEvent> changes, int? version)
        {
            if (!_documents.TryGetValue(uri, out var document))
                return;

            var content = document.Content;

            foreach (var change in changes)
            {
                if (change.Range == null)
                {
                    // Full document replacement
                    content = change.Text;
                }
                else
                {
                    // Incremental change
                    var lines = content.Split('\n');
                    var startLine = change.Range.Start.Line;
                    var startChar = change.Range.Start.Character;
                    var endLine = change.Range.End.Line;
                    var endChar = change.Range.End.Character;

                    // Calculate start and end positions
                    var startPos = GetOffset(lines, (int)startLine, (int)startChar);
                    var endPos = GetOffset(lines, (int)endLine, (int)endChar);

                    // Apply change
                    content = content[..startPos] + change.Text + content[endPos..];
                }
            }

            UpdateDocument(uri, content, version);
        }

        private static int GetOffset(string[] lines, int line, int character)
        {
            int offset = 0;
            for (int i = 0; i < line && i < lines.Length; i++)
            {
                offset += lines[i].Length + 1; // +1 for newline
            }
            if (line < lines.Length)
            {
                offset += Math.Min(character, lines[line].Length);
            }
            return offset;
        }
    }

    /// <summary>
    /// State of a single document
    /// </summary>
    public class DocumentState
    {
        /// <summary>Document URI</summary>
        public DocumentUri Uri { get; }

        /// <summary>Document content</summary>
        public string Content { get; }

        /// <summary>Document version</summary>
        public int Version { get; }

        /// <summary>Cached lines</summary>
        public string[] Lines { get; }

        /// <summary>Last parse result</summary>
        public ParseResult? ParseResult { get; set; }

        /// <summary>Cached diagnostics</summary>
        public List<Diagnostic> Diagnostics { get; set; } = new();

        /// <summary>
        /// Creates a new document state
        /// </summary>
        /// <param name="uri">Document URI</param>
        /// <param name="content">Document content</param>
        /// <param name="version">Document version</param>
        public DocumentState(DocumentUri uri, string content, int version)
        {
            Uri = uri;
            Content = content;
            Version = version;
            Lines = content.Split('\n');
        }

        /// <summary>
        /// Get position from offset
        /// </summary>
        public Position OffsetToPosition(int offset)
        {
            int line = 0;
            int col = 0;
            int current = 0;

            foreach (var lineText in Lines)
            {
                if (current + lineText.Length + 1 > offset)
                {
                    col = offset - current;
                    break;
                }
                current += lineText.Length + 1;
                line++;
            }

            return new Position(line, col);
        }

        /// <summary>
        /// Get offset from position
        /// </summary>
        public int PositionToOffset(Position position)
        {
            int offset = 0;
            for (int i = 0; i < position.Line && i < Lines.Length; i++)
            {
                offset += Lines[i].Length + 1;
            }
            if (position.Line < Lines.Length)
            {
                offset += (int)Math.Min(position.Character, Lines[(int)position.Line].Length);
            }
            return offset;
        }

        /// <summary>
        /// Get text at a range
        /// </summary>
        public string GetText(Range range)
        {
            var start = PositionToOffset(range.Start);
            var end = PositionToOffset(range.End);
            return Content.Substring(start, end - start);
        }

        /// <summary>
        /// Get word at position
        /// </summary>
        public (string word, Range range)? GetWordAtPosition(Position position)
        {
            if (position.Line >= Lines.Length)
                return null;

            var line = Lines[(int)position.Line];
            var col = (int)Math.Min(position.Character, line.Length);

            // Find word boundaries
            int start = col;
            int end = col;

            while (start > 0 && IsIdentifierChar(line[start - 1]))
                start--;

            while (end < line.Length && IsIdentifierChar(line[end]))
                end++;

            if (start == end)
                return null;

            var word = line[start..end];
            var range = new Range(
                new Position(position.Line, start),
                new Position(position.Line, end)
            );

            return (word, range);
        }

        private static bool IsIdentifierChar(char c)
        {
            return char.IsLetterOrDigit(c) || c == '_' || c == '@';
        }
    }

    /// <summary>
    /// Result of parsing a document
    /// </summary>
    public class ParseResult
    {
        /// <summary>Whether parsing succeeded</summary>
        public bool Success { get; init; }

        /// <summary>Syntax errors</summary>
        public List<SyntaxError> Errors { get; init; } = new();

        /// <summary>AST nodes</summary>
        public object? Ast { get; init; }

        /// <summary>All symbols defined in this file</summary>
        public List<SymbolInfo> Symbols { get; init; } = new();
    }

    /// <summary>
    /// Syntax error information
    /// </summary>
    public class SyntaxError
    {
        /// <summary>Error message</summary>
        public string Message { get; init; } = "";

        /// <summary>Error location</summary>
        public Range Range { get; init; } = new(new Position(0, 0), new Position(0, 0));

        /// <summary>Error severity</summary>
        public DiagnosticSeverity Severity { get; init; } = DiagnosticSeverity.Error;
    }
}
