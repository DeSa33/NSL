using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Workspace;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace NSL.LanguageServer
{
    /// <summary>
    /// Handles workspace symbol search
    /// </summary>
    public class WorkspaceSymbolHandler : IWorkspaceSymbolsHandler
    {
        private readonly SymbolTable _symbolTable;

        /// <summary>
        /// Creates a new workspace symbol handler
        /// </summary>
        /// <param name="symbolTable">Symbol table for searching symbols</param>
        public WorkspaceSymbolHandler(SymbolTable symbolTable)
        {
            _symbolTable = symbolTable;
        }

        /// <inheritdoc/>
        public Task<Container<WorkspaceSymbol>?> Handle(
            WorkspaceSymbolParams request,
            CancellationToken cancellationToken)
        {
            var symbols = _symbolTable.Search(request.Query)
                .Where(s => s.Location != null)
                .Select(s => new WorkspaceSymbol
                {
                    Name = s.Name,
                    Kind = s.Kind,
                    Location = new Location
                    {
                        Uri = s.Location!.Uri,
                        Range = s.Location.Range
                    },
                    ContainerName = s.ContainerName
                });

            return Task.FromResult<Container<WorkspaceSymbol>?>(new Container<WorkspaceSymbol>(symbols.ToList()));
        }

        /// <inheritdoc/>
        public WorkspaceSymbolRegistrationOptions GetRegistrationOptions(
            WorkspaceSymbolCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new WorkspaceSymbolRegistrationOptions();
        }
    }
}
