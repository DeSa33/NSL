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
    /// Handles document symbol requests (outline view)
    /// </summary>
    public class DocumentSymbolHandler : IDocumentSymbolHandler
    {
        private readonly DocumentManager _documentManager;

        /// <summary>
        /// Creates a new document symbol handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        public DocumentSymbolHandler(DocumentManager documentManager)
        {
            _documentManager = documentManager;
        }

        /// <inheritdoc/>
        public Task<SymbolInformationOrDocumentSymbolContainer?> Handle(
            DocumentSymbolParams request,
            CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document?.ParseResult == null)
                return Task.FromResult<SymbolInformationOrDocumentSymbolContainer?>(null);

            var symbols = document.ParseResult.Symbols.Select(s => new DocumentSymbol
            {
                Name = s.Name,
                Kind = s.Kind,
                Range = s.Location?.Range ?? new Range(new Position(0, 0), new Position(0, 0)),
                SelectionRange = s.Location?.Range ?? new Range(new Position(0, 0), new Position(0, 0)),
                Detail = s.Type
            });

            return Task.FromResult<SymbolInformationOrDocumentSymbolContainer?>(new SymbolInformationOrDocumentSymbolContainer(
                symbols.Select(s => new SymbolInformationOrDocumentSymbol(s))));
        }

        /// <inheritdoc/>
        public DocumentSymbolRegistrationOptions GetRegistrationOptions(
            DocumentSymbolCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new DocumentSymbolRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl"),
                Label = "NSL"
            };
        }
    }
}
