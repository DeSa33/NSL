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
    /// Handles hover requests (tooltip information)
    /// </summary>
    public class HoverHandler : IHoverHandler
    {
        private readonly DocumentManager _documentManager;
        private readonly NslAnalyzer _analyzer;

        /// <summary>
        /// Creates a new hover handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        /// <param name="analyzer">NSL analyzer for hover information</param>
        public HoverHandler(DocumentManager documentManager, NslAnalyzer analyzer)
        {
            _documentManager = documentManager;
            _analyzer = analyzer;
        }

        /// <inheritdoc/>
        public Task<Hover?> Handle(HoverParams request, CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult<Hover?>(null);

            return Task.FromResult(_analyzer.GetHover(document, request.Position));
        }

        /// <inheritdoc/>
        public HoverRegistrationOptions GetRegistrationOptions(
            HoverCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new HoverRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl")
            };
        }
    }
}
