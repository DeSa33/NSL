using System;
using System.Threading;
using System.Threading.Tasks;
using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Server.Capabilities;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace NSL.LanguageServer
{
    /// <summary>
    /// Handles document open/change/close events
    /// </summary>
    public class TextDocumentSyncHandler : TextDocumentSyncHandlerBase
    {
        private readonly DocumentManager _documentManager;
        private readonly NslAnalyzer _analyzer;
        private readonly ILanguageServerFacade _server;

        /// <summary>
        /// Creates a new text document sync handler
        /// </summary>
        /// <param name="documentManager">Document manager for storing document content</param>
        /// <param name="analyzer">NSL analyzer for diagnostics</param>
        /// <param name="server">Language server facade for publishing diagnostics</param>
        public TextDocumentSyncHandler(
            DocumentManager documentManager,
            NslAnalyzer analyzer,
            ILanguageServerFacade server)
        {
            _documentManager = documentManager;
            _analyzer = analyzer;
            _server = server;
        }

        /// <inheritdoc/>
        public override TextDocumentAttributes GetTextDocumentAttributes(DocumentUri uri)
        {
            return new TextDocumentAttributes(uri, "nsl");
        }

        /// <inheritdoc/>
        public override Task<Unit> Handle(DidOpenTextDocumentParams request, CancellationToken cancellationToken)
        {
            var uri = request.TextDocument.Uri;
            var content = request.TextDocument.Text;
            var version = (int?)request.TextDocument.Version;

            _documentManager.UpdateDocument(uri, content, version);

            // Analyze and publish diagnostics
            AnalyzeAndPublishDiagnostics(uri);

            return Unit.Task;
        }

        /// <inheritdoc/>
        public override Task<Unit> Handle(DidChangeTextDocumentParams request, CancellationToken cancellationToken)
        {
            var uri = request.TextDocument.Uri;
            var version = (int?)request.TextDocument.Version;

            _documentManager.ApplyChanges(uri, request.ContentChanges, version);

            // Re-analyze
            AnalyzeAndPublishDiagnostics(uri);

            return Unit.Task;
        }

        /// <inheritdoc/>
        public override Task<Unit> Handle(DidSaveTextDocumentParams request, CancellationToken cancellationToken)
        {
            // Re-analyze on save
            AnalyzeAndPublishDiagnostics(request.TextDocument.Uri);
            return Unit.Task;
        }

        /// <inheritdoc/>
        public override Task<Unit> Handle(DidCloseTextDocumentParams request, CancellationToken cancellationToken)
        {
            var uri = request.TextDocument.Uri;
            _documentManager.CloseDocument(uri);

            // Clear diagnostics
            _server.TextDocument.PublishDiagnostics(new PublishDiagnosticsParams
            {
                Uri = uri,
                Diagnostics = new Container<Diagnostic>()
            });

            return Unit.Task;
        }

        /// <inheritdoc/>
        protected override TextDocumentSyncRegistrationOptions CreateRegistrationOptions(
            TextSynchronizationCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new TextDocumentSyncRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl"),
                Change = TextDocumentSyncKind.Incremental,
                Save = new SaveOptions { IncludeText = true }
            };
        }

        private void AnalyzeAndPublishDiagnostics(DocumentUri uri)
        {
            var document = _documentManager.GetDocument(uri);
            if (document == null)
                return;

            var result = _analyzer.Analyze(document);
            document.ParseResult = result;
            document.Diagnostics = result.Errors.Select(e => new Diagnostic
            {
                Range = e.Range,
                Severity = e.Severity,
                Source = "nsl",
                Message = e.Message
            }).ToList();

            _server.TextDocument.PublishDiagnostics(new PublishDiagnosticsParams
            {
                Uri = uri,
                Version = document.Version,
                Diagnostics = new Container<Diagnostic>(document.Diagnostics)
            });
        }
    }
}
