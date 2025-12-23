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
    /// Handles find-all-references requests
    /// </summary>
    public class ReferencesHandler : IReferencesHandler
    {
        private readonly DocumentManager _documentManager;
        private readonly NslAnalyzer _analyzer;

        /// <summary>
        /// Creates a new references handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        /// <param name="analyzer">NSL analyzer for finding references</param>
        public ReferencesHandler(DocumentManager documentManager, NslAnalyzer analyzer)
        {
            _documentManager = documentManager;
            _analyzer = analyzer;
        }

        /// <inheritdoc/>
        public Task<LocationContainer?> Handle(ReferenceParams request, CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult<LocationContainer?>(null);

            var references = _analyzer.GetReferences(
                document,
                request.Position,
                request.Context.IncludeDeclaration);

            return Task.FromResult<LocationContainer?>(new LocationContainer(references.ToList()));
        }

        /// <inheritdoc/>
        public ReferenceRegistrationOptions GetRegistrationOptions(
            ReferenceCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new ReferenceRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl")
            };
        }
    }
}
