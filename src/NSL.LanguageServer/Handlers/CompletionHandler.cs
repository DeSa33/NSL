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
    /// Handles code completion requests
    /// </summary>
    public class CompletionHandler : ICompletionHandler
    {
        private readonly DocumentManager _documentManager;
        private readonly NslAnalyzer _analyzer;

        /// <summary>
        /// Creates a new completion handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        /// <param name="analyzer">NSL analyzer for completion items</param>
        public CompletionHandler(DocumentManager documentManager, NslAnalyzer analyzer)
        {
            _documentManager = documentManager;
            _analyzer = analyzer;
        }

        /// <inheritdoc/>
        public Task<CompletionList> Handle(CompletionParams request, CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult(new CompletionList());

            var completions = _analyzer.GetCompletions(document, request.Position);

            return Task.FromResult(new CompletionList(completions.ToList(), isIncomplete: false));
        }

        /// <inheritdoc/>
        public CompletionRegistrationOptions GetRegistrationOptions(
            CompletionCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new CompletionRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl"),
                TriggerCharacters = new Container<string>(".", ":", "@", "◈", "∇", "⊗", "Ψ"),
                ResolveProvider = true
            };
        }
    }
}
