using System;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Server;

namespace NSL.LanguageServer
{
    /// <summary>
    /// NSL Language Server - Provides IDE support through LSP.
    ///
    /// Features:
    /// - Syntax highlighting
    /// - Code completion
    /// - Go to definition
    /// - Find references
    /// - Hover information
    /// - Diagnostics (errors/warnings)
    /// - Code formatting
    /// - Symbol search
    ///
    /// Usage:
    /// Run the language server and connect from your IDE (VSCode, Neovim, etc.)
    /// </summary>
    class Program
    {
        static async Task Main(string[] args)
        {
            var server = await OmniSharp.Extensions.LanguageServer.Server.LanguageServer.From(options =>
                options
                    .WithInput(Console.OpenStandardInput())
                    .WithOutput(Console.OpenStandardOutput())
                    .ConfigureLogging(logging =>
                    {
                        logging.AddLanguageProtocolLogging();
                        logging.SetMinimumLevel(LogLevel.Debug);
                    })
                    .WithServices(ConfigureServices)
                    .WithHandler<TextDocumentSyncHandler>()
                    .WithHandler<CompletionHandler>()
                    .WithHandler<HoverHandler>()
                    .WithHandler<DefinitionHandler>()
                    .WithHandler<ReferencesHandler>()
                    .WithHandler<DocumentSymbolHandler>()
                    .WithHandler<WorkspaceSymbolHandler>()
                    .WithHandler<SemanticTokensHandler>()
                    .WithHandler<FormattingHandler>()
                    .WithHandler<SignatureHelpHandler>()
                    .WithHandler<RenameHandler>()
                    .WithHandler<CodeActionHandler>()
                    .OnInitialize((server, request, token) =>
                    {
                        return Task.CompletedTask;
                    })
                    .OnInitialized(async (server, request, response, token) =>
                    {
                        // Log to stderr (standard for LSP servers)
                        await Console.Error.WriteLineAsync("NSL Language Server initialized!");
                    })
            );

            await server.WaitForExit;
        }

        private static void ConfigureServices(IServiceCollection services)
        {
            services.AddSingleton<DocumentManager>();
            services.AddSingleton<NslAnalyzer>();
            services.AddSingleton<SymbolTable>();
        }
    }
}
