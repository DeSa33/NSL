using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace NSL.LanguageServer
{
    /// <summary>
    /// Handles semantic tokens for enhanced syntax highlighting
    /// </summary>
    public class SemanticTokensHandler : SemanticTokensHandlerBase
    {
        private readonly DocumentManager _documentManager;

        // Token types for NSL
        private static readonly string[] TokenTypes = new[]
        {
            "namespace", "type", "class", "enum", "interface", "struct",
            "typeParameter", "parameter", "variable", "property", "enumMember",
            "event", "function", "method", "macro", "keyword", "modifier",
            "comment", "string", "number", "regexp", "operator"
        };

        // Token modifiers
        private static readonly string[] TokenModifiers = new[]
        {
            "declaration", "definition", "readonly", "static", "deprecated",
            "abstract", "async", "modification", "documentation", "defaultLibrary"
        };

        /// <summary>
        /// Creates a new semantic tokens handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        public SemanticTokensHandler(DocumentManager documentManager)
        {
            _documentManager = documentManager;
        }

        /// <inheritdoc/>
        protected override SemanticTokensRegistrationOptions CreateRegistrationOptions(
            SemanticTokensCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new SemanticTokensRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl"),
                Full = new SemanticTokensCapabilityRequestFull { Delta = false },
                Range = true,
                Legend = new SemanticTokensLegend
                {
                    TokenTypes = new Container<SemanticTokenType>(
                        TokenTypes.Select(t => new SemanticTokenType(t))),
                    TokenModifiers = new Container<SemanticTokenModifier>(
                        TokenModifiers.Select(m => new SemanticTokenModifier(m)))
                }
            };
        }

        /// <inheritdoc/>
        protected override Task<SemanticTokensDocument> GetSemanticTokensDocument(
            ITextDocumentIdentifierParams request,
            CancellationToken cancellationToken)
        {
            return Task.FromResult(new SemanticTokensDocument(CreateRegistrationOptions(null!, null!).Legend));
        }

        /// <inheritdoc/>
        protected override Task Tokenize(
            SemanticTokensBuilder builder,
            ITextDocumentIdentifierParams request,
            CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.CompletedTask;

            var tokens = TokenizeDocument(document);
            foreach (var token in tokens)
            {
                builder.Push(
                    token.Line,
                    token.StartChar,
                    token.Length,
                    token.TokenType,
                    token.Modifiers);
            }

            return Task.CompletedTask;
        }

        private IEnumerable<SemanticToken> TokenizeDocument(DocumentState document)
        {
            var tokens = new List<SemanticToken>();
            var content = document.Content;
            var lines = content.Split('\n');

            for (int lineNum = 0; lineNum < lines.Length; lineNum++)
            {
                var line = lines[lineNum];
                var col = 0;

                while (col < line.Length)
                {
                    // Skip whitespace
                    while (col < line.Length && char.IsWhiteSpace(line[col]))
                        col++;

                    if (col >= line.Length)
                        break;

                    // Comments
                    if (col + 1 < line.Length && line[col] == '/' && line[col + 1] == '/')
                    {
                        tokens.Add(new SemanticToken(lineNum, col, line.Length - col, GetTypeIndex("comment")));
                        break;
                    }

                    // Strings
                    if (line[col] == '"' || line[col] == '\'')
                    {
                        var quote = line[col];
                        var start = col;
                        col++;
                        while (col < line.Length && line[col] != quote)
                        {
                            if (line[col] == '\\' && col + 1 < line.Length)
                                col++;
                            col++;
                        }
                        col++;
                        tokens.Add(new SemanticToken(lineNum, start, col - start, GetTypeIndex("string")));
                        continue;
                    }

                    // Numbers
                    if (char.IsDigit(line[col]))
                    {
                        var start = col;
                        while (col < line.Length && (char.IsDigit(line[col]) || line[col] == '.' || line[col] == 'e' || line[col] == 'E'))
                            col++;
                        tokens.Add(new SemanticToken(lineNum, start, col - start, GetTypeIndex("number")));
                        continue;
                    }

                    // Identifiers and keywords
                    if (char.IsLetter(line[col]) || line[col] == '_' || line[col] == '@')
                    {
                        var start = col;
                        while (col < line.Length && (char.IsLetterOrDigit(line[col]) || line[col] == '_'))
                            col++;

                        var word = line[start..col];
                        var tokenType = GetTokenTypeForWord(word);
                        tokens.Add(new SemanticToken(lineNum, start, col - start, tokenType));
                        continue;
                    }

                    // Operators (including Unicode consciousness operators)
                    if (line[col] == '◈' || line[col] == '∇' || line[col] == '⊗' || line[col] == 'Ψ')
                    {
                        tokens.Add(new SemanticToken(lineNum, col, 1, GetTypeIndex("operator"), GetModifierMask("defaultLibrary")));
                        col++;
                        continue;
                    }

                    // Other operators
                    if ("+-*/%=<>!&|^~?.,:;()[]{}@#".Contains(line[col]))
                    {
                        var start = col;
                        // Multi-char operators
                        if (col + 1 < line.Length)
                        {
                            var twoChar = line.Substring(col, 2);
                            if (twoChar is "|>" or "?." or "??" or ".." or "::" or "=>" or "==" or "!=" or "<=" or ">=")
                            {
                                tokens.Add(new SemanticToken(lineNum, col, 2, GetTypeIndex("operator")));
                                col += 2;
                                continue;
                            }
                        }
                        tokens.Add(new SemanticToken(lineNum, col, 1, GetTypeIndex("operator")));
                        col++;
                        continue;
                    }

                    col++;
                }
            }

            return tokens;
        }

        private int GetTokenTypeForWord(string word)
        {
            if (NslAnalyzer.Keywords.Contains(word))
                return GetTypeIndex("keyword");
            if (NslAnalyzer.Types.Contains(word))
                return GetTypeIndex("type");
            if (NslAnalyzer.BuiltinFunctions.Contains(word))
                return GetTypeIndex("function");
            return GetTypeIndex("variable");
        }

        private static int GetTypeIndex(string typeName)
        {
            var index = System.Array.IndexOf(TokenTypes, typeName);
            return index >= 0 ? index : 0;
        }

        private static int GetModifierMask(params string[] modifiers)
        {
            int mask = 0;
            foreach (var mod in modifiers)
            {
                var index = System.Array.IndexOf(TokenModifiers, mod);
                if (index >= 0)
                    mask |= (1 << index);
            }
            return mask;
        }

        private class SemanticToken
        {
            /// <summary>Public API</summary>
            public int Line { get; }
            /// <summary>Public API</summary>
            public int StartChar { get; }
            /// <summary>Public API</summary>
            public int Length { get; }
            /// <summary>Public API</summary>
            public int TokenType { get; }
            /// <summary>Public API</summary>
            public int Modifiers { get; }

            /// <summary>Public API</summary>
            public SemanticToken(int line, int startChar, int length, int tokenType, int modifiers = 0)
            {
                Line = line;
                StartChar = startChar;
                Length = length;
                TokenType = tokenType;
                Modifiers = modifiers;
            }
        }
    }
}