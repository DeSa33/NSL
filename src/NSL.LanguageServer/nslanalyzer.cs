using System;
using System.Collections.Generic;
using System.Linq;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace NSL.LanguageServer
{
    /// <summary>
    /// Analyzes NSL source code for the language server
    /// </summary>
    public class NslAnalyzer
    {
        private readonly SymbolTable _symbolTable;

        /// <summary>
        /// NSL keywords
        /// </summary>
        public static readonly HashSet<string> Keywords = new()
        {
            "fn", "function", "let", "mut", "const", "var",
            "if", "else", "elif", "for", "while", "loop",
            "match", "case", "return", "break", "continue",
            "import", "from", "export", "pub", "as",
            "struct", "enum", "trait", "impl", "type",
            "async", "await", "yield",
            "true", "false", "nil", "null",
            "and", "or", "not", "in", "is",
            "try", "catch", "throw", "finally",
            "Ok", "Err", "Some", "None"
        };

        /// <summary>
        /// NSL built-in types
        /// </summary>
        public static readonly HashSet<string> Types = new()
        {
            "int", "float", "bool", "string", "char",
            "Vec", "Mat", "Tensor", "Prob",
            "Result", "Option", "List", "Map", "Set",
            "i8", "i16", "i32", "i64",
            "u8", "u16", "u32", "u64",
            "f32", "f64"
        };

        /// <summary>
        /// NSL built-in functions
        /// </summary>
        public static readonly HashSet<string> BuiltinFunctions = new()
        {
            // Core I/O
            "print", "println", "input", "type", "len", "range",

            // Math
            "abs", "sqrt", "exp", "log", "pow", "sin", "cos", "tan",
            "floor", "ceil", "round", "min", "max", "sum", "avg", "random",

            // Neural/ML
            "zeros", "ones", "tensor", "matmul", "softmax", "relu", "sigmoid",
            "tanh", "gelu", "leaky_relu", "softplus", "grad", "backward",
            "shape", "reshape", "transpose", "mean", "dot", "normalize",

            // Shell/System
            "shell", "exec", "run", "powershell", "env", "set_env", "env_all",
            "sleep", "exit", "timestamp", "now",

            // File System
            "read_file", "write_file", "file_exists", "dir_exists", "file_info",
            "list_dir", "mkdir", "delete_file", "delete_dir", "copy_file", "move_file",
            "cwd", "cd", "read_binary", "write_binary",

            // Encoding
            "base64_encode", "base64_decode", "hex_encode", "hex_decode",
            "url_encode", "url_decode",

            // Hashing
            "md5", "sha256", "sha512", "hash_file",

            // Compression
            "gzip", "gunzip", "zip_create", "zip_extract", "zip_list",

            // JSON
            "json_parse", "json_stringify",

            // String
            "split", "join", "replace", "trim", "upper", "lower",
            "contains", "starts_with", "ends_with", "substring",
            "regex_match", "regex_replace",

            // Array
            "map", "filter", "reduce", "find", "sort", "reverse",
            "unique", "flatten", "slice", "enumerate", "zip",

            // HTTP
            "http_get", "http_post", "download",

            // Utility
            "uuid",

            // Consciousness
            "measure", "entangle"
        };

        /// <summary>
        /// NSL operators (including consciousness operators)
        /// </summary>
        public static readonly Dictionary<string, string> Operators = new()
        {
            { "◈", "Holographic operator - creates distributed representations" },
            { "∇", "Gradient operator - computes gradients for learning" },
            { "⊗", "Tensor product - outer product for entanglement" },
            { "Ψ", "Quantum branching - creates superposition states" },
            { "|>", "Pipeline operator - functional composition" },
            { "?.", "Safe navigation - null-safe property access" },
            { "??", "Null coalescing - default value for null" },
            { "@", "Matrix multiplication" },
            { "..", "Range (exclusive)" },
            { "..=", "Range (inclusive)" },
            { "::", "Method chaining / namespace access" },
            { "=>", "Match arm / lambda arrow" }
        };

        /// <summary>
        /// Creates a new NSL analyzer
        /// </summary>
        /// <param name="symbolTable">Symbol table for cross-file resolution</param>
        public NslAnalyzer(SymbolTable symbolTable)
        {
            _symbolTable = symbolTable;
        }

        /// <summary>
        /// Analyze a document and return diagnostics
        /// </summary>
        public ParseResult Analyze(DocumentState document)
        {
            var errors = new List<SyntaxError>();
            var symbols = new List<SymbolInfo>();

            try
            {
                // Lexical analysis
                var tokens = Tokenize(document.Content);

                // Parse tokens to find symbols and errors
                var context = new ParseContext(tokens, document);

                // Find function definitions
                FindFunctions(context, symbols);

                // Find variable declarations
                FindVariables(context, symbols);

                // Find struct/enum definitions
                FindTypeDefinitions(context, symbols);

                // Find imports
                FindImports(context, symbols);

                // Check for syntax errors
                CheckSyntax(context, errors);

                // Type checking (simplified)
                CheckTypes(context, errors);

                // Update symbol table
                _symbolTable.UpdateFile(document.Uri, symbols);
            }
            catch (Exception ex)
            {
                errors.Add(new SyntaxError
                {
                    Message = $"Analysis error: {ex.Message}",
                    Range = new Range(new Position(0, 0), new Position(0, 0)),
                    Severity = DiagnosticSeverity.Error
                });
            }

            return new ParseResult
            {
                Success = errors.Count == 0,
                Errors = errors,
                Symbols = symbols
            };
        }

        /// <summary>
        /// Get completion items at position
        /// </summary>
        public IEnumerable<CompletionItem> GetCompletions(DocumentState document, Position position)
        {
            var completions = new List<CompletionItem>();

            var wordInfo = document.GetWordAtPosition(position);
            var prefix = wordInfo?.word ?? "";

            // Keywords
            foreach (var keyword in Keywords.Where(k => k.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
            {
                completions.Add(new CompletionItem
                {
                    Label = keyword,
                    Kind = CompletionItemKind.Keyword,
                    Detail = "keyword",
                    InsertText = keyword
                });
            }

            // Types
            foreach (var type in Types.Where(t => t.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
            {
                completions.Add(new CompletionItem
                {
                    Label = type,
                    Kind = CompletionItemKind.Class,
                    Detail = "type",
                    InsertText = type
                });
            }

            // Built-in functions
            foreach (var func in BuiltinFunctions.Where(f => f.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
            {
                completions.Add(new CompletionItem
                {
                    Label = func,
                    Kind = CompletionItemKind.Function,
                    Detail = "built-in function",
                    InsertText = func,
                    InsertTextFormat = InsertTextFormat.PlainText
                });
            }

            // Consciousness operators
            foreach (var (op, desc) in Operators)
            {
                completions.Add(new CompletionItem
                {
                    Label = op,
                    Kind = CompletionItemKind.Operator,
                    Detail = desc,
                    InsertText = op
                });
            }

            // Symbols from current file
            var parseResult = document.ParseResult;
            if (parseResult != null)
            {
                foreach (var symbol in parseResult.Symbols.Where(s => s.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
                {
                    completions.Add(new CompletionItem
                    {
                        Label = symbol.Name,
                        Kind = SymbolKindToCompletionKind(symbol.Kind),
                        Detail = symbol.Type ?? symbol.Kind.ToString(),
                        InsertText = symbol.Name
                    });
                }
            }

            // Symbols from other files
            foreach (var symbol in _symbolTable.GetAllSymbols()
                .Where(s => s.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
            {
                if (parseResult?.Symbols.Any(ps => ps.Name == symbol.Name) != true)
                {
                    completions.Add(new CompletionItem
                    {
                        Label = symbol.Name,
                        Kind = SymbolKindToCompletionKind(symbol.Kind),
                        Detail = symbol.Type ?? symbol.Kind.ToString(),
                        InsertText = symbol.Name
                    });
                }
            }

            // Snippets
            AddSnippets(completions, prefix);

            return completions;
        }

        /// <summary>
        /// Get hover information at position
        /// </summary>
        public Hover? GetHover(DocumentState document, Position position)
        {
            var wordInfo = document.GetWordAtPosition(position);
            if (wordInfo == null)
                return null;

            var (word, range) = wordInfo.Value;

            // Check keywords
            if (Keywords.Contains(word))
            {
                return new Hover
                {
                    Contents = new MarkedStringsOrMarkupContent(
                        new MarkupContent
                        {
                            Kind = MarkupKind.Markdown,
                            Value = GetKeywordDocumentation(word)
                        }
                    ),
                    Range = range
                };
            }

            // Check types
            if (Types.Contains(word))
            {
                return new Hover
                {
                    Contents = new MarkedStringsOrMarkupContent(
                        new MarkupContent
                        {
                            Kind = MarkupKind.Markdown,
                            Value = GetTypeDocumentation(word)
                        }
                    ),
                    Range = range
                };
            }

            // Check built-in functions
            if (BuiltinFunctions.Contains(word))
            {
                return new Hover
                {
                    Contents = new MarkedStringsOrMarkupContent(
                        new MarkupContent
                        {
                            Kind = MarkupKind.Markdown,
                            Value = GetBuiltinFunctionDocumentation(word)
                        }
                    ),
                    Range = range
                };
            }

            // Check operators
            if (Operators.TryGetValue(word, out var opDesc))
            {
                return new Hover
                {
                    Contents = new MarkedStringsOrMarkupContent(
                        new MarkupContent
                        {
                            Kind = MarkupKind.Markdown,
                            Value = $"**Operator `{word}`**\n\n{opDesc}"
                        }
                    ),
                    Range = range
                };
            }

            // Check symbols
            var symbol = _symbolTable.FindSymbol(word);
            if (symbol != null)
            {
                return new Hover
                {
                    Contents = new MarkedStringsOrMarkupContent(
                        new MarkupContent
                        {
                            Kind = MarkupKind.Markdown,
                            Value = FormatSymbolHover(symbol)
                        }
                    ),
                    Range = range
                };
            }

            return null;
        }

        /// <summary>
        /// Get definition location
        /// </summary>
        public Location? GetDefinition(DocumentState document, Position position)
        {
            var wordInfo = document.GetWordAtPosition(position);
            if (wordInfo == null)
                return null;

            var (word, _) = wordInfo.Value;

            var symbol = _symbolTable.FindSymbol(word);
            if (symbol == null)
                return null;

            return symbol.Location;
        }

        /// <summary>
        /// Get all references to a symbol
        /// </summary>
        public IEnumerable<Location> GetReferences(DocumentState document, Position position, bool includeDeclaration)
        {
            var wordInfo = document.GetWordAtPosition(position);
            if (wordInfo == null)
                yield break;

            var (word, _) = wordInfo.Value;

            foreach (var reference in _symbolTable.FindReferences(word))
            {
                yield return reference;
            }
        }

        #region Private Helpers

        private List<Token> Tokenize(string content)
        {
            var tokens = new List<Token>();
            var i = 0;
            var line = 0;
            var col = 0;

            while (i < content.Length)
            {
                var c = content[i];

                // Skip whitespace
                if (char.IsWhiteSpace(c))
                {
                    if (c == '\n')
                    {
                        line++;
                        col = 0;
                    }
                    else
                    {
                        col++;
                    }
                    i++;
                    continue;
                }

                // Comments
                if (c == '/' && i + 1 < content.Length)
                {
                    if (content[i + 1] == '/')
                    {
                        // Line comment
                        while (i < content.Length && content[i] != '\n')
                            i++;
                        continue;
                    }
                    if (content[i + 1] == '*')
                    {
                        // Block comment
                        i += 2;
                        while (i + 1 < content.Length && !(content[i] == '*' && content[i + 1] == '/'))
                        {
                            if (content[i] == '\n') { line++; col = 0; }
                            i++;
                        }
                        i += 2;
                        continue;
                    }
                }

                // String literals
                if (c == '"' || c == '\'')
                {
                    var quote = c;
                    var start = i;
                    var startLine = line;
                    var startCol = col;
                    i++;
                    col++;
                    while (i < content.Length && content[i] != quote)
                    {
                        if (content[i] == '\\' && i + 1 < content.Length)
                        {
                            i++;
                            col++;
                        }
                        if (content[i] == '\n') { line++; col = 0; }
                        else col++;
                        i++;
                    }
                    i++; col++;
                    tokens.Add(new Token(TokenKind.String, content[start..i], startLine, startCol));
                    continue;
                }

                // Numbers
                if (char.IsDigit(c))
                {
                    var start = i;
                    var startCol = col;
                    while (i < content.Length && (char.IsDigit(content[i]) || content[i] == '.' || content[i] == 'e' || content[i] == 'E'))
                    {
                        i++;
                        col++;
                    }
                    tokens.Add(new Token(TokenKind.Number, content[start..i], line, startCol));
                    continue;
                }

                // Identifiers and keywords
                if (char.IsLetter(c) || c == '_' || c == '@')
                {
                    var start = i;
                    var startCol = col;
                    while (i < content.Length && (char.IsLetterOrDigit(content[i]) || content[i] == '_'))
                    {
                        i++;
                        col++;
                    }
                    var text = content[start..i];
                    var kind = Keywords.Contains(text) ? TokenKind.Keyword :
                               Types.Contains(text) ? TokenKind.Type :
                               BuiltinFunctions.Contains(text) ? TokenKind.Function :
                               TokenKind.Identifier;
                    tokens.Add(new Token(kind, text, line, startCol));
                    continue;
                }

                // Unicode operators
                if (c == '◈' || c == '∇' || c == '⊗' || c == 'Ψ')
                {
                    tokens.Add(new Token(TokenKind.Operator, c.ToString(), line, col));
                    i++;
                    col++;
                    continue;
                }

                // Multi-char operators
                if (i + 1 < content.Length)
                {
                    var twoChar = content.Substring(i, 2);
                    if (twoChar is "|>" or "?." or "??" or ".." or "::" or "=>" or "==" or "!=" or "<=" or ">=" or "->" or "&&" or "||")
                    {
                        tokens.Add(new Token(TokenKind.Operator, twoChar, line, col));
                        i += 2;
                        col += 2;
                        continue;
                    }
                }

                // Single char tokens
                tokens.Add(new Token(TokenKind.Punctuation, c.ToString(), line, col));
                i++;
                col++;
            }

            return tokens;
        }

        private void FindFunctions(ParseContext ctx, List<SymbolInfo> symbols)
        {
            for (int i = 0; i < ctx.Tokens.Count - 1; i++)
            {
                var token = ctx.Tokens[i];
                if (token.Kind == TokenKind.Keyword && (token.Text == "fn" || token.Text == "function"))
                {
                    var nextToken = ctx.Tokens[i + 1];
                    if (nextToken.Kind == TokenKind.Identifier)
                    {
                        // Find parameters
                        var parameters = new List<string>();
                        var returnType = "unknown";

                        // Look for parameter list
                        int j = i + 2;
                        if (j < ctx.Tokens.Count && ctx.Tokens[j].Text == "(")
                        {
                            j++;
                            while (j < ctx.Tokens.Count && ctx.Tokens[j].Text != ")")
                            {
                                if (ctx.Tokens[j].Kind == TokenKind.Identifier)
                                {
                                    parameters.Add(ctx.Tokens[j].Text);
                                }
                                j++;
                            }
                        }

                        symbols.Add(new SymbolInfo
                        {
                            Name = nextToken.Text,
                            Kind = SymbolKind.Function,
                            Type = $"fn({string.Join(", ", parameters)}) -> {returnType}",
                            Location = new Location
                            {
                                Uri = ctx.Document.Uri,
                                Range = new Range(
                                    new Position(token.Line, token.Column),
                                    new Position(nextToken.Line, nextToken.Column + nextToken.Text.Length)
                                )
                            }
                        });
                    }
                }
            }
        }

        private void FindVariables(ParseContext ctx, List<SymbolInfo> symbols)
        {
            for (int i = 0; i < ctx.Tokens.Count - 1; i++)
            {
                var token = ctx.Tokens[i];
                if (token.Kind == TokenKind.Keyword && (token.Text == "let" || token.Text == "mut" || token.Text == "const"))
                {
                    var nextToken = ctx.Tokens[i + 1];
                    if (nextToken.Kind == TokenKind.Identifier)
                    {
                        symbols.Add(new SymbolInfo
                        {
                            Name = nextToken.Text,
                            Kind = token.Text == "const" ? SymbolKind.Constant : SymbolKind.Variable,
                            Type = "unknown",
                            Location = new Location
                            {
                                Uri = ctx.Document.Uri,
                                Range = new Range(
                                    new Position(nextToken.Line, nextToken.Column),
                                    new Position(nextToken.Line, nextToken.Column + nextToken.Text.Length)
                                )
                            }
                        });
                    }
                }
            }
        }

        private void FindTypeDefinitions(ParseContext ctx, List<SymbolInfo> symbols)
        {
            for (int i = 0; i < ctx.Tokens.Count - 1; i++)
            {
                var token = ctx.Tokens[i];
                if (token.Kind == TokenKind.Keyword && (token.Text == "struct" || token.Text == "enum" || token.Text == "trait" || token.Text == "type"))
                {
                    var nextToken = ctx.Tokens[i + 1];
                    if (nextToken.Kind == TokenKind.Identifier)
                    {
                        symbols.Add(new SymbolInfo
                        {
                            Name = nextToken.Text,
                            Kind = token.Text switch
                            {
                                "struct" => SymbolKind.Struct,
                                "enum" => SymbolKind.Enum,
                                "trait" => SymbolKind.Interface,
                                _ => SymbolKind.Class
                            },
                            Location = new Location
                            {
                                Uri = ctx.Document.Uri,
                                Range = new Range(
                                    new Position(token.Line, token.Column),
                                    new Position(nextToken.Line, nextToken.Column + nextToken.Text.Length)
                                )
                            }
                        });
                    }
                }
            }
        }

        private void FindImports(ParseContext ctx, List<SymbolInfo> symbols)
        {
            for (int i = 0; i < ctx.Tokens.Count - 1; i++)
            {
                var token = ctx.Tokens[i];
                if (token.Kind == TokenKind.Keyword && token.Text == "import")
                {
                    var nextToken = ctx.Tokens[i + 1];
                    symbols.Add(new SymbolInfo
                    {
                        Name = nextToken.Text,
                        Kind = SymbolKind.Module,
                        Location = new Location
                        {
                            Uri = ctx.Document.Uri,
                            Range = new Range(
                                new Position(nextToken.Line, nextToken.Column),
                                new Position(nextToken.Line, nextToken.Column + nextToken.Text.Length)
                            )
                        }
                    });
                }
            }
        }

        private void CheckSyntax(ParseContext ctx, List<SyntaxError> errors)
        {
            var braceStack = new Stack<Token>();

            foreach (var token in ctx.Tokens)
            {
                if (token.Text is "{" or "(" or "[")
                {
                    braceStack.Push(token);
                }
                else if (token.Text is "}" or ")" or "]")
                {
                    if (braceStack.Count == 0)
                    {
                        errors.Add(new SyntaxError
                        {
                            Message = $"Unexpected closing '{token.Text}'",
                            Range = new Range(
                                new Position(token.Line, token.Column),
                                new Position(token.Line, token.Column + 1)
                            )
                        });
                    }
                    else
                    {
                        var opening = braceStack.Pop();
                        var expected = token.Text switch
                        {
                            "}" => "{",
                            ")" => "(",
                            "]" => "[",
                            _ => ""
                        };
                        if (opening.Text != expected)
                        {
                            errors.Add(new SyntaxError
                            {
                                Message = $"Mismatched brackets: expected closing for '{opening.Text}' but found '{token.Text}'",
                                Range = new Range(
                                    new Position(token.Line, token.Column),
                                    new Position(token.Line, token.Column + 1)
                                )
                            });
                        }
                    }
                }
            }

            foreach (var unclosed in braceStack)
            {
                errors.Add(new SyntaxError
                {
                    Message = $"Unclosed '{unclosed.Text}'",
                    Range = new Range(
                        new Position(unclosed.Line, unclosed.Column),
                        new Position(unclosed.Line, unclosed.Column + 1)
                    )
                });
            }
        }

        private void CheckTypes(ParseContext ctx, List<SyntaxError> errors)
        {
            // Simplified type checking - just look for obvious issues
            // A full type checker would need the complete AST
        }

        private void AddSnippets(List<CompletionItem> completions, string prefix)
        {
            var snippets = new[]
            {
                ("fn", "fn ${1:name}(${2:params}) {\n\t$0\n}", "Function definition"),
                ("if", "if ${1:condition} {\n\t$0\n}", "If statement"),
                ("for", "for ${1:i} in ${2:range} {\n\t$0\n}", "For loop"),
                ("while", "while ${1:condition} {\n\t$0\n}", "While loop"),
                ("match", "match ${1:expr} {\n\tcase ${2:pattern} => $0\n}", "Match expression"),
                ("struct", "struct ${1:Name} {\n\t${2:field}: ${3:Type}\n}", "Struct definition"),
                ("impl", "impl ${1:Trait} for ${2:Type} {\n\t$0\n}", "Trait implementation"),
                ("test", "fn test_${1:name}() {\n\t$0\n}", "Test function")
            };

            foreach (var (label, snippet, detail) in snippets)
            {
                if (label.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    completions.Add(new CompletionItem
                    {
                        Label = label,
                        Kind = CompletionItemKind.Snippet,
                        Detail = detail,
                        InsertText = snippet,
                        InsertTextFormat = InsertTextFormat.Snippet
                    });
                }
            }
        }

        private static CompletionItemKind SymbolKindToCompletionKind(SymbolKind kind)
        {
            return kind switch
            {
                SymbolKind.Function => CompletionItemKind.Function,
                SymbolKind.Method => CompletionItemKind.Method,
                SymbolKind.Variable => CompletionItemKind.Variable,
                SymbolKind.Constant => CompletionItemKind.Constant,
                SymbolKind.Class => CompletionItemKind.Class,
                SymbolKind.Struct => CompletionItemKind.Struct,
                SymbolKind.Enum => CompletionItemKind.Enum,
                SymbolKind.Interface => CompletionItemKind.Interface,
                SymbolKind.Module => CompletionItemKind.Module,
                _ => CompletionItemKind.Text
            };
        }

        private static string GetKeywordDocumentation(string keyword)
        {
            return keyword switch
            {
                "fn" or "function" => "**fn** - Define a function\n\n```nsl\nfn name(param: Type) -> ReturnType {\n    // body\n}\n```",
                "let" => "**let** - Declare an immutable variable\n\n```nsl\nlet x = 42\n```",
                "mut" => "**mut** - Declare a mutable variable\n\n```nsl\nmut counter = 0\ncounter = counter + 1\n```",
                "const" => "**const** - Declare a compile-time constant\n\n```nsl\nconst PI = 3.14159\n```",
                "if" => "**if** - Conditional statement\n\n```nsl\nif condition {\n    // then\n} else {\n    // else\n}\n```",
                "match" => "**match** - Pattern matching expression\n\n```nsl\nmatch value {\n    case 1 => \"one\"\n    case 2 => \"two\"\n    _ => \"other\"\n}\n```",
                "for" => "**for** - For loop\n\n```nsl\nfor i in 0..10 {\n    print(i)\n}\n```",
                "struct" => "**struct** - Define a structure\n\n```nsl\nstruct Point {\n    x: float\n    y: float\n}\n```",
                "trait" => "**trait** - Define a trait (interface)\n\n```nsl\ntrait Drawable {\n    fn draw()\n}\n```",
                "import" => "**import** - Import a module\n\n```nsl\nimport math\nfrom \"./utils\" import helper\n```",
                _ => $"**{keyword}** - NSL keyword"
            };
        }

        private static string GetTypeDocumentation(string type)
        {
            return type switch
            {
                "int" => "**int** - 64-bit signed integer\n\nRange: -2^63 to 2^63-1",
                "float" => "**float** - 64-bit floating point number\n\nIEEE 754 double precision",
                "bool" => "**bool** - Boolean value\n\nCan be `true` or `false`",
                "string" => "**string** - UTF-8 encoded string\n\nImmutable sequence of characters",
                "Vec" => "**Vec** - Dynamic array/vector\n\n```nsl\nlet v: Vec<int> = [1, 2, 3]\n```",
                "Tensor" => "**Tensor** - Multi-dimensional array for neural networks\n\n```nsl\nlet t = tensor([[1, 2], [3, 4]])\n```",
                "Result" => "**Result<T>** - Success or error value\n\n```nsl\nfn parse(s: string) -> Result<int> {\n    Ok(42)\n}\n```",
                "Option" => "**Option<T>** - Optional value\n\n```nsl\nlet x: Option<int> = Some(42)\n```",
                "Prob" => "**Prob** - Probability value (0..1)\n\nUsed in probabilistic computations",
                "Mat" => "**Mat** - 2D matrix type\n\nFor linear algebra operations",
                _ => $"**{type}** - NSL type"
            };
        }

        private static string GetBuiltinFunctionDocumentation(string func)
        {
            return func switch
            {
                "print" => "**print(value)**\n\nPrint value to stdout without newline",
                "println" => "**println(value)**\n\nPrint value to stdout with newline",
                "len" => "**len(collection) -> int**\n\nReturn the length of a collection",
                "range" => "**range(start, end, step?) -> Iterator**\n\nGenerate a sequence of numbers",
                "map" => "**map(fn, collection) -> Collection**\n\nApply function to each element",
                "filter" => "**filter(predicate, collection) -> Collection**\n\nKeep elements matching predicate",
                "reduce" => "**reduce(fn, collection, initial?) -> T**\n\nReduce collection to single value",
                "tensor" => "**tensor(data) -> Tensor**\n\nCreate a tensor from nested arrays",
                "zeros" => "**zeros(shape) -> Tensor**\n\nCreate tensor filled with zeros",
                "ones" => "**ones(shape) -> Tensor**\n\nCreate tensor filled with ones",
                "matmul" => "**matmul(a, b) -> Tensor**\n\nMatrix multiplication: a @ b",
                "softmax" => "**softmax(tensor, axis?) -> Tensor**\n\nSoftmax activation function",
                "relu" => "**relu(tensor) -> Tensor**\n\nReLU activation: max(0, x)",
                "grad" => "**grad(tensor) -> Tensor**\n\nGet gradient of tensor (requires backward())",
                "backward" => "**backward(tensor)**\n\nCompute gradients via backpropagation",
                _ => $"**{func}()** - Built-in function"
            };
        }

        private static string FormatSymbolHover(SymbolInfo symbol)
        {
            var kindStr = symbol.Kind.ToString().ToLowerInvariant();
            var type = symbol.Type != null ? $": {symbol.Type}" : "";
            return $"**{kindStr}** `{symbol.Name}`{type}\n\nDefined in {symbol.Location?.Uri.Path}";
        }

        #endregion

        private class ParseContext
        {
            /// <summary>Public API</summary>
            public List<Token> Tokens { get; }
            /// <summary>Public API</summary>
            public DocumentState Document { get; }

            /// <summary>Public API</summary>
            public ParseContext(List<Token> tokens, DocumentState document)
            {
                Tokens = tokens;
                Document = document;
            }
        }
    }

    internal enum TokenKind
    {
        /// <summary>API member</summary>
        Keyword,
        /// <summary>API member</summary>
        Identifier,
        /// <summary>API member</summary>
        Type,
        /// <summary>API member</summary>
        Function,
        /// <summary>API member</summary>
        Number,
        /// <summary>API member</summary>
        String,
        /// <summary>API member</summary>
        Operator,
        Punctuation
    }

    internal class Token
    {
        /// <summary>Public API</summary>
        public TokenKind Kind { get; }
        /// <summary>Public API</summary>
        public string Text { get; }
        /// <summary>Public API</summary>
        public int Line { get; }
        /// <summary>Public API</summary>
        public int Column { get; }

        /// <summary>Public API</summary>
        public Token(TokenKind kind, string text, int line, int column)
        {
            Kind = kind;
            Text = text;
            Line = line;
            Column = column;
        }
    }
}