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
    /// Handles code action requests (quick fixes, refactorings)
    /// </summary>
    public class CodeActionHandler : ICodeActionHandler
    {
        private readonly DocumentManager _documentManager;

        /// <summary>
        /// Creates a new code action handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        public CodeActionHandler(DocumentManager documentManager)
        {
            _documentManager = documentManager;
        }

        /// <inheritdoc/>
        public Task<CommandOrCodeActionContainer?> Handle(CodeActionParams request, CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult<CommandOrCodeActionContainer?>(null);

            var actions = new List<CodeAction>();

            // Check for diagnostics at this location
            var diagnosticsAtRange = request.Context.Diagnostics
                .Where(d => RangeOverlaps(d.Range, request.Range))
                .ToList();

            foreach (var diagnostic in diagnosticsAtRange)
            {
                // Add quick fixes based on diagnostic
                var fixes = GetQuickFixes(document, diagnostic, request.TextDocument.Uri);
                actions.AddRange(fixes);
            }

            // Add refactoring actions
            var refactorings = GetRefactorings(document, request.Range, request.TextDocument.Uri);
            actions.AddRange(refactorings);

            return Task.FromResult<CommandOrCodeActionContainer?>(new CommandOrCodeActionContainer(
                actions.Select(a => new CommandOrCodeAction(a))));
        }

        /// <inheritdoc/>
        public CodeActionRegistrationOptions GetRegistrationOptions(
            CodeActionCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new CodeActionRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl"),
                CodeActionKinds = new Container<CodeActionKind>(
                    CodeActionKind.QuickFix,
                    CodeActionKind.Refactor,
                    CodeActionKind.RefactorExtract,
                    CodeActionKind.RefactorInline,
                    CodeActionKind.Source
                )
            };
        }

        private IEnumerable<CodeAction> GetQuickFixes(DocumentState document, Diagnostic diagnostic, DocumentUri uri)
        {
            var message = diagnostic.Message;
            var range = diagnostic.Range;

            // Missing closing brace
            if (message.Contains("Unclosed '{'"))
            {
                yield return new CodeAction
                {
                    Title = "Add closing brace",
                    Kind = CodeActionKind.QuickFix,
                    Diagnostics = new Container<Diagnostic>(diagnostic),
                    Edit = new WorkspaceEdit
                    {
                        Changes = new Dictionary<DocumentUri, IEnumerable<TextEdit>>
                        {
                            [uri] = new[]
                            {
                                new TextEdit
                                {
                                    Range = new Range(range.End, range.End),
                                    NewText = "\n}"
                                }
                            }
                        }
                    }
                };
            }

            // Unused variable (if we detect it)
            if (message.Contains("unused variable"))
            {
                var wordInfo = document.GetWordAtPosition(range.Start);
                if (wordInfo != null)
                {
                    yield return new CodeAction
                    {
                        Title = $"Prefix with underscore: _{wordInfo.Value.word}",
                        Kind = CodeActionKind.QuickFix,
                        Diagnostics = new Container<Diagnostic>(diagnostic),
                        Edit = new WorkspaceEdit
                        {
                            Changes = new Dictionary<DocumentUri, IEnumerable<TextEdit>>
                            {
                                [uri] = new[]
                                {
                                    new TextEdit
                                    {
                                        Range = wordInfo.Value.range,
                                        NewText = $"_{wordInfo.Value.word}"
                                    }
                                }
                            }
                        }
                    };
                }
            }
        }

        private IEnumerable<CodeAction> GetRefactorings(DocumentState document, Range range, DocumentUri uri)
        {
            var text = document.GetText(range);

            // Extract to function
            if (!string.IsNullOrWhiteSpace(text) && text.Contains('\n'))
            {
                yield return new CodeAction
                {
                    Title = "Extract to function",
                    Kind = CodeActionKind.RefactorExtract,
                    Edit = CreateExtractFunctionEdit(document, text, range, uri)
                };
            }

            // Extract to variable
            if (!string.IsNullOrWhiteSpace(text) && !text.Contains('\n') && text.Length > 10)
            {
                yield return new CodeAction
                {
                    Title = "Extract to variable",
                    Kind = CodeActionKind.RefactorExtract,
                    Edit = CreateExtractVariableEdit(document, text, range, uri)
                };
            }

            // Convert to arrow function
            var line = range.Start.Line < document.Lines.Length
                ? document.Lines[(int)range.Start.Line]
                : "";
            if (line.Contains("fn ") && line.Contains("{") && line.Contains("return"))
            {
                yield return new CodeAction
                {
                    Title = "Convert to arrow function",
                    Kind = CodeActionKind.Refactor
                    // Would need more complex edit
                };
            }

            // Add type annotation
            var wordInfo = document.GetWordAtPosition(range.Start);
            if (wordInfo != null && line.Contains("let ") && !line.Contains(':'))
            {
                yield return new CodeAction
                {
                    Title = "Add type annotation",
                    Kind = CodeActionKind.Refactor
                };
            }
        }

        private WorkspaceEdit CreateExtractFunctionEdit(DocumentState document, string text, Range range, DocumentUri uri)
        {
            var functionName = "extractedFunction";
            var indent = GetIndent(document, range.Start);

            var functionDef = $"\nfn {functionName}() {{\n{indent}    {text.Trim()}\n{indent}}}\n";
            var functionCall = $"{functionName}()";

            return new WorkspaceEdit
            {
                Changes = new Dictionary<DocumentUri, IEnumerable<TextEdit>>
                {
                    [uri] = new[]
                    {
                        // Insert function definition before current function
                        new TextEdit
                        {
                            Range = new Range(new Position(0, 0), new Position(0, 0)),
                            NewText = functionDef
                        },
                        // Replace selection with function call
                        new TextEdit
                        {
                            Range = range,
                            NewText = functionCall
                        }
                    }
                }
            };
        }

        private WorkspaceEdit CreateExtractVariableEdit(DocumentState document, string text, Range range, DocumentUri uri)
        {
            var varName = "extracted";
            var indent = GetIndent(document, range.Start);

            return new WorkspaceEdit
            {
                Changes = new Dictionary<DocumentUri, IEnumerable<TextEdit>>
                {
                    [uri] = new[]
                    {
                        // Insert variable declaration before current line
                        new TextEdit
                        {
                            Range = new Range(
                                new Position(range.Start.Line, 0),
                                new Position(range.Start.Line, 0)),
                            NewText = $"{indent}let {varName} = {text.Trim()}\n"
                        },
                        // Replace selection with variable name
                        new TextEdit
                        {
                            Range = range,
                            NewText = varName
                        }
                    }
                }
            };
        }

        private string GetIndent(DocumentState document, Position position)
        {
            if (position.Line >= document.Lines.Length)
                return "";

            var line = document.Lines[(int)position.Line];
            var indent = "";
            foreach (var c in line)
            {
                if (c == ' ' || c == '\t')
                    indent += c;
                else
                    break;
            }
            return indent;
        }

        private static bool RangeOverlaps(Range a, Range b)
        {
            return !(a.End.Line < b.Start.Line ||
                    (a.End.Line == b.Start.Line && a.End.Character < b.Start.Character) ||
                    a.Start.Line > b.End.Line ||
                    (a.Start.Line == b.End.Line && a.Start.Character > b.End.Character));
        }
    }
}
