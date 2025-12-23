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
    /// Handles go-to-definition requests
    /// </summary>
    public class DefinitionHandler : IDefinitionHandler
    {
        private readonly DocumentManager _documentManager;
        private readonly NslAnalyzer _analyzer;

        /// <summary>
        /// Creates a new definition handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        /// <param name="analyzer">NSL analyzer for definition lookup</param>
        public DefinitionHandler(DocumentManager documentManager, NslAnalyzer analyzer)
        {
            _documentManager = documentManager;
            _analyzer = analyzer;
        }

        /// <inheritdoc/>
        public Task<LocationOrLocationLinks?> Handle(DefinitionParams request, CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult<LocationOrLocationLinks?>(null);

            var location = _analyzer.GetDefinition(document, request.Position);
            if (location == null)
                return Task.FromResult<LocationOrLocationLinks?>(null);

            return Task.FromResult<LocationOrLocationLinks?>(new LocationOrLocationLinks(location));
        }

        /// <inheritdoc/>
        public DefinitionRegistrationOptions GetRegistrationOptions(
            DefinitionCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new DefinitionRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl")
            };
        }
    }
}
