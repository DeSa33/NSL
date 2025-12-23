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
    /// Handles rename symbol requests
    /// </summary>
    public class RenameHandler : IRenameHandler
    {
        private readonly DocumentManager _documentManager;
        private readonly NslAnalyzer _analyzer;
        private readonly SymbolTable _symbolTable;

        /// <summary>
        /// Creates a new rename handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        /// <param name="analyzer">NSL analyzer for symbol information</param>
        /// <param name="symbolTable">Symbol table for finding references</param>
        public RenameHandler(DocumentManager documentManager, NslAnalyzer analyzer, SymbolTable symbolTable)
        {
            _documentManager = documentManager;
            _analyzer = analyzer;
            _symbolTable = symbolTable;
        }

        /// <inheritdoc/>
        public Task<WorkspaceEdit?> Handle(RenameParams request, CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult<WorkspaceEdit?>(null);

            var wordInfo = document.GetWordAtPosition(request.Position);
            if (wordInfo == null)
                return Task.FromResult<WorkspaceEdit?>(null);

            var (oldName, _) = wordInfo.Value;
            var newName = request.NewName;

            // Validate new name
            if (string.IsNullOrWhiteSpace(newName) ||
                !char.IsLetter(newName[0]) && newName[0] != '_')
            {
                return Task.FromResult<WorkspaceEdit?>(null);
            }

            // Check if it's a keyword or built-in
            if (NslAnalyzer.Keywords.Contains(oldName) ||
                NslAnalyzer.BuiltinFunctions.Contains(oldName) ||
                NslAnalyzer.Types.Contains(oldName))
            {
                return Task.FromResult<WorkspaceEdit?>(null);
            }

            // Find all references
            var references = _symbolTable.FindReferences(oldName).ToList();

            // Also search in all open documents
            foreach (var doc in _documentManager.GetAllDocuments())
            {
                var content = doc.Content;
                var lines = doc.Lines;

                for (int lineNum = 0; lineNum < lines.Length; lineNum++)
                {
                    var line = lines[lineNum];
                    var col = 0;

                    while ((col = FindWord(line, oldName, col)) >= 0)
                    {
                        var location = new Location
                        {
                            Uri = doc.Uri,
                            Range = new Range(
                                new Position(lineNum, col),
                                new Position(lineNum, col + oldName.Length)
                            )
                        };

                        if (!references.Any(r =>
                            r.Uri == location.Uri &&
                            r.Range.Start.Line == location.Range.Start.Line &&
                            r.Range.Start.Character == location.Range.Start.Character))
                        {
                            references.Add(location);
                        }

                        col += oldName.Length;
                    }
                }
            }

            if (!references.Any())
                return Task.FromResult<WorkspaceEdit?>(null);

            // Group by document
            var changesByDocument = references
                .GroupBy(r => r.Uri)
                .ToDictionary(
                    g => g.Key,
                    g => g.Select(r => new TextEdit
                    {
                        Range = r.Range,
                        NewText = newName
                    }).ToList()
                );

            var documentChanges = changesByDocument.Select(kvp =>
                new WorkspaceEditDocumentChange(new TextDocumentEdit
                {
                    TextDocument = new OptionalVersionedTextDocumentIdentifier
                    {
                        Uri = kvp.Key,
                        Version = _documentManager.GetDocument(kvp.Key)?.Version
                    },
                    Edits = new TextEditContainer(kvp.Value)
                })
            ).ToList();

            return Task.FromResult<WorkspaceEdit?>(new WorkspaceEdit
            {
                DocumentChanges = new Container<WorkspaceEditDocumentChange>(documentChanges)
            });
        }

        /// <inheritdoc/>
        public RenameRegistrationOptions GetRegistrationOptions(
            RenameCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new RenameRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl"),
                PrepareProvider = true
            };
        }

        private static int FindWord(string line, string word, int startIndex)
        {
            while (startIndex < line.Length)
            {
                var index = line.IndexOf(word, startIndex);
                if (index < 0)
                    return -1;

                // Check word boundaries
                var beforeOk = index == 0 || !char.IsLetterOrDigit(line[index - 1]) && line[index - 1] != '_';
                var afterOk = index + word.Length >= line.Length ||
                    !char.IsLetterOrDigit(line[index + word.Length]) && line[index + word.Length] != '_';

                if (beforeOk && afterOk)
                    return index;

                startIndex = index + 1;
            }

            return -1;
        }
    }
}
