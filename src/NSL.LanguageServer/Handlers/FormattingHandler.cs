using System.Collections.Generic;
using System.Text;
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
    /// Handles document formatting requests
    /// </summary>
    public class FormattingHandler : IDocumentFormattingHandler
    {
        private readonly DocumentManager _documentManager;

        /// <summary>
        /// Creates a new formatting handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        public FormattingHandler(DocumentManager documentManager)
        {
            _documentManager = documentManager;
        }

        /// <inheritdoc/>
        public Task<TextEditContainer?> Handle(
            DocumentFormattingParams request,
            CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult<TextEditContainer?>(null);

            var formatted = FormatDocument(document.Content, request.Options);
            if (formatted == document.Content)
                return Task.FromResult<TextEditContainer?>(new TextEditContainer());

            var fullRange = new Range(
                new Position(0, 0),
                new Position(document.Lines.Length, 0)
            );

            return Task.FromResult<TextEditContainer?>(new TextEditContainer(new TextEdit
            {
                Range = fullRange,
                NewText = formatted
            }));
        }

        /// <inheritdoc/>
        public DocumentFormattingRegistrationOptions GetRegistrationOptions(
            DocumentFormattingCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new DocumentFormattingRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl")
            };
        }

        private string FormatDocument(string content, FormattingOptions options)
        {
            var sb = new StringBuilder();
            var lines = content.Split('\n');
            var indentLevel = 0;
            var indentString = options.InsertSpaces
                ? new string(' ', (int)options.TabSize)
                : "\t";

            foreach (var rawLine in lines)
            {
                var line = rawLine.TrimEnd('\r').Trim();

                // Adjust indent before writing
                if (line.StartsWith('}') || line.StartsWith(']') || line.StartsWith(')'))
                {
                    indentLevel = System.Math.Max(0, indentLevel - 1);
                }

                // Skip empty lines but preserve them
                if (string.IsNullOrEmpty(line))
                {
                    sb.AppendLine();
                    continue;
                }

                // Write indented line
                for (int i = 0; i < indentLevel; i++)
                    sb.Append(indentString);
                sb.AppendLine(line);

                // Adjust indent after writing
                var openBraces = CountChars(line, '{') + CountChars(line, '[') + CountChars(line, '(');
                var closeBraces = CountChars(line, '}') + CountChars(line, ']') + CountChars(line, ')');
                indentLevel += openBraces - closeBraces;
                indentLevel = System.Math.Max(0, indentLevel);

                // Handle single-line blocks
                if (line.EndsWith('{') || line.EndsWith('['))
                {
                    // Already handled
                }
            }

            return sb.ToString();
        }

        private static int CountChars(string s, char c)
        {
            int count = 0;
            bool inString = false;
            char stringChar = '\0';

            for (int i = 0; i < s.Length; i++)
            {
                if (!inString && (s[i] == '"' || s[i] == '\''))
                {
                    inString = true;
                    stringChar = s[i];
                }
                else if (inString && s[i] == stringChar && (i == 0 || s[i - 1] != '\\'))
                {
                    inString = false;
                }
                else if (!inString && s[i] == c)
                {
                    count++;
                }
            }

            return count;
        }
    }
}
