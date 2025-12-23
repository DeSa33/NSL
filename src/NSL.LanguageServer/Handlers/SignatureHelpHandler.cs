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
    /// Handles signature help (parameter info) requests
    /// </summary>
    public class SignatureHelpHandler : ISignatureHelpHandler
    {
        private readonly DocumentManager _documentManager;
        private readonly SymbolTable _symbolTable;

        /// <summary>
        /// Built-in function signatures for signature help
        /// </summary>
        private static readonly Dictionary<string, (string signature, string[] parameters)> BuiltinSignatures = new()
        {
            // Core I/O
            ["print"] = ("print(...values: any)", new[] { "values - Values to print (space-separated)" }),
            ["input"] = ("input(prompt?: string) -> string", new[] { "prompt - Optional prompt to display" }),
            ["len"] = ("len(collection: Collection) -> int", new[] { "collection - The collection to measure" }),
            ["type"] = ("type(value: any) -> string", new[] { "value - Value to get type of" }),
            ["range"] = ("range(end: int) or range(start: int, end: int, step?: int) -> Iterator<int>",
                new[] { "start - Starting value (default: 0)", "end - Ending value (exclusive)", "step - Step size (default: 1)" }),

            // Shell/System
            ["shell"] = ("shell(command: string) -> {output, exit_code, success}", new[] { "command - Command to execute" }),
            ["exec"] = ("exec(command: string) -> int", new[] { "command - Command to execute, returns exit code" }),
            ["powershell"] = ("powershell(command: string) -> {output, success}", new[] { "command - PowerShell command to execute" }),
            ["env"] = ("env(name: string) -> string", new[] { "name - Environment variable name" }),
            ["set_env"] = ("set_env(name: string, value: string)", new[] { "name - Variable name", "value - Variable value" }),
            ["sleep"] = ("sleep(ms: int)", new[] { "ms - Milliseconds to sleep" }),
            ["timestamp"] = ("timestamp() -> int", System.Array.Empty<string>()),
            ["now"] = ("now(format?: string) -> string", new[] { "format - Date format string (default: ISO 8601)" }),

            // File System
            ["read_file"] = ("read_file(path: string) -> string", new[] { "path - File path to read" }),
            ["write_file"] = ("write_file(path: string, content: string) -> bool", new[] { "path - File path", "content - Content to write" }),
            ["file_exists"] = ("file_exists(path: string) -> bool", new[] { "path - File path to check" }),
            ["dir_exists"] = ("dir_exists(path: string) -> bool", new[] { "path - Directory path to check" }),
            ["list_dir"] = ("list_dir(path: string) -> {files, dirs}", new[] { "path - Directory path" }),
            ["mkdir"] = ("mkdir(path: string) -> bool", new[] { "path - Directory path to create" }),
            ["delete_file"] = ("delete_file(path: string) -> bool", new[] { "path - File path to delete" }),
            ["copy_file"] = ("copy_file(src: string, dst: string) -> bool", new[] { "src - Source path", "dst - Destination path" }),
            ["cwd"] = ("cwd() -> string", System.Array.Empty<string>()),
            ["read_binary"] = ("read_binary(path: string) -> {bytes, base64, size}", new[] { "path - Binary file path" }),
            ["write_binary"] = ("write_binary(path: string, data: bytes) -> bool", new[] { "path - File path", "data - Binary data" }),

            // Encoding
            ["base64_encode"] = ("base64_encode(text: string) -> string", new[] { "text - Text to encode" }),
            ["base64_decode"] = ("base64_decode(b64: string) -> bytes", new[] { "b64 - Base64 string to decode" }),
            ["hex_encode"] = ("hex_encode(text: string) -> string", new[] { "text - Text to encode" }),
            ["hex_decode"] = ("hex_decode(hex: string) -> bytes", new[] { "hex - Hex string to decode" }),
            ["url_encode"] = ("url_encode(text: string) -> string", new[] { "text - Text to URL encode" }),
            ["url_decode"] = ("url_decode(text: string) -> string", new[] { "text - Text to URL decode" }),

            // Hashing
            ["md5"] = ("md5(text: string) -> string", new[] { "text - Text to hash" }),
            ["sha256"] = ("sha256(text: string) -> string", new[] { "text - Text to hash" }),
            ["sha512"] = ("sha512(text: string) -> string", new[] { "text - Text to hash" }),
            ["hash_file"] = ("hash_file(path: string, algo: string) -> {hash, size}", new[] { "path - File path", "algo - Algorithm: md5, sha256, sha512" }),

            // Compression
            ["gzip"] = ("gzip(text: string) -> {bytes, base64, ratio}", new[] { "text - Text to compress" }),
            ["gunzip"] = ("gunzip(data: string) -> {bytes, base64}", new[] { "data - Base64 gzipped data" }),
            ["zip_create"] = ("zip_create(path: string, files: string[]) -> {path, count, size}", new[] { "path - Archive path", "files - Files to include" }),
            ["zip_extract"] = ("zip_extract(path: string, dest: string) -> {path, count}", new[] { "path - Archive path", "dest - Destination directory" }),

            // JSON
            ["json_parse"] = ("json_parse(json: string) -> object", new[] { "json - JSON string to parse" }),
            ["json_stringify"] = ("json_stringify(obj: object, pretty?: bool) -> string", new[] { "obj - Object to serialize", "pretty - Pretty print (default: false)" }),

            // String
            ["split"] = ("split(text: string, delimiter: string) -> string[]", new[] { "text - String to split", "delimiter - Separator" }),
            ["join"] = ("join(arr: string[], delimiter: string) -> string", new[] { "arr - Array to join", "delimiter - Separator" }),
            ["replace"] = ("replace(text: string, old: string, new: string) -> string", new[] { "text - Input string", "old - Substring to replace", "new - Replacement" }),
            ["trim"] = ("trim(text: string) -> string", new[] { "text - String to trim" }),
            ["upper"] = ("upper(text: string) -> string", new[] { "text - String to uppercase" }),
            ["lower"] = ("lower(text: string) -> string", new[] { "text - String to lowercase" }),
            ["contains"] = ("contains(text: string, substring: string) -> bool", new[] { "text - String to search", "substring - Substring to find" }),
            ["starts_with"] = ("starts_with(text: string, prefix: string) -> bool", new[] { "text - String to check", "prefix - Prefix to match" }),
            ["ends_with"] = ("ends_with(text: string, suffix: string) -> bool", new[] { "text - String to check", "suffix - Suffix to match" }),
            ["substring"] = ("substring(text: string, start: int, length?: int) -> string", new[] { "text - Source string", "start - Start index", "length - Number of chars" }),
            ["regex_match"] = ("regex_match(text: string, pattern: string) -> {found, matches}", new[] { "text - String to search", "pattern - Regex pattern" }),
            ["regex_replace"] = ("regex_replace(text: string, pattern: string, replacement: string) -> string", new[] { "text - Input string", "pattern - Regex pattern", "replacement - Replacement" }),

            // Array
            ["map"] = ("map(arr: T[], fn: (T) -> U) -> U[]", new[] { "arr - Input array", "fn - Transform function" }),
            ["filter"] = ("filter(arr: T[], fn: (T) -> bool) -> T[]", new[] { "arr - Input array", "fn - Predicate function" }),
            ["reduce"] = ("reduce(arr: T[], fn: (acc, T) -> T, initial: T) -> T", new[] { "arr - Input array", "fn - Reducer function", "initial - Initial value" }),
            ["find"] = ("find(arr: T[], fn: (T) -> bool) -> T?", new[] { "arr - Input array", "fn - Predicate function" }),
            ["sort"] = ("sort(arr: T[]) -> T[]", new[] { "arr - Array to sort" }),
            ["reverse"] = ("reverse(arr: T[]) -> T[]", new[] { "arr - Array to reverse" }),
            ["unique"] = ("unique(arr: T[]) -> T[]", new[] { "arr - Array with duplicates" }),
            ["flatten"] = ("flatten(arr: T[][]) -> T[]", new[] { "arr - Nested array" }),
            ["slice"] = ("slice(arr: T[], start: int, count?: int) -> T[]", new[] { "arr - Input array", "start - Start index", "count - Number of elements" }),

            // HTTP
            ["http_get"] = ("http_get(url: string, headers?: object) -> {status, body, headers}", new[] { "url - URL to fetch", "headers - Optional headers" }),
            ["http_post"] = ("http_post(url: string, body: object, headers?: object) -> {status, body, headers}", new[] { "url - URL to post to", "body - Request body", "headers - Optional headers" }),
            ["download"] = ("download(url: string, path: string) -> {path, size, success}", new[] { "url - URL to download", "path - Destination path" }),

            // Math
            ["min"] = ("min(...values: number) -> number", new[] { "values - Numbers to compare" }),
            ["max"] = ("max(...values: number) -> number", new[] { "values - Numbers to compare" }),
            ["sum"] = ("sum(...values: number) -> number", new[] { "values - Numbers to sum" }),
            ["avg"] = ("avg(...values: number) -> number", new[] { "values - Numbers to average" }),
            ["round"] = ("round(x: number, decimals?: int) -> number", new[] { "x - Number to round", "decimals - Decimal places (default: 0)" }),
            ["random"] = ("random(min?: number, max?: number) -> number", new[] { "min - Minimum value (default: 0)", "max - Maximum value (default: 1)" }),
            ["abs"] = ("abs(x: number) -> number", new[] { "x - Number" }),
            ["sqrt"] = ("sqrt(x: number) -> number", new[] { "x - Number" }),
            ["pow"] = ("pow(base: number, exp: number) -> number", new[] { "base - Base number", "exp - Exponent" }),

            // Utility
            ["uuid"] = ("uuid() -> string", System.Array.Empty<string>()),

            // Neural/Tensor
            ["tensor"] = ("tensor(data: Array, requires_grad?: bool) -> Tensor", new[] { "data - Nested array of numbers", "requires_grad - Enable gradient tracking" }),
            ["zeros"] = ("zeros(shape: int[]) -> Tensor", new[] { "shape - Shape of the tensor" }),
            ["ones"] = ("ones(shape: int[]) -> Tensor", new[] { "shape - Shape of the tensor" }),
            ["matmul"] = ("matmul(a: Tensor, b: Tensor) -> Tensor", new[] { "a - Left matrix", "b - Right matrix" }),
            ["softmax"] = ("softmax(tensor: Tensor, axis?: int) -> Tensor", new[] { "tensor - Input tensor", "axis - Axis (default: -1)" }),
            ["relu"] = ("relu(tensor: Tensor) -> Tensor", new[] { "tensor - Input tensor" }),
            ["sigmoid"] = ("sigmoid(tensor: Tensor) -> Tensor", new[] { "tensor - Input tensor" }),
            ["grad"] = ("grad(tensor: Tensor) -> Tensor", new[] { "tensor - Tensor with gradients" }),
            ["backward"] = ("backward(tensor: Tensor)", new[] { "tensor - Output tensor to backpropagate from" })
        };

        /// <summary>
        /// Creates a new signature help handler
        /// </summary>
        /// <param name="documentManager">Document manager for accessing document content</param>
        /// <param name="symbolTable">Symbol table for user-defined function signatures</param>
        public SignatureHelpHandler(DocumentManager documentManager, SymbolTable symbolTable)
        {
            _documentManager = documentManager;
            _symbolTable = symbolTable;
        }

        /// <inheritdoc/>
        public Task<SignatureHelp?> Handle(SignatureHelpParams request, CancellationToken cancellationToken)
        {
            var document = _documentManager.GetDocument(request.TextDocument.Uri);
            if (document == null)
                return Task.FromResult<SignatureHelp?>(null);

            // Find the function name before the cursor
            var (functionName, paramIndex) = FindFunctionContext(document, request.Position);
            if (functionName == null)
                return Task.FromResult<SignatureHelp?>(null);

            // Check built-ins first
            if (BuiltinSignatures.TryGetValue(functionName, out var builtin))
            {
                return Task.FromResult<SignatureHelp?>(new SignatureHelp
                {
                    Signatures = new Container<SignatureInformation>(new SignatureInformation
                    {
                        Label = builtin.signature,
                        Documentation = $"Built-in function `{functionName}`",
                        Parameters = new Container<ParameterInformation>(
                            builtin.parameters.Select(p =>
                            {
                                var parts = p.Split(" - ", 2);
                                return new ParameterInformation
                                {
                                    Label = parts[0],
                                    Documentation = parts.Length > 1 ? parts[1] : null
                                };
                            }))
                    }),
                    ActiveSignature = 0,
                    ActiveParameter = paramIndex
                });
            }

            // Check user-defined functions
            var symbol = _symbolTable.FindSymbol(functionName);
            if (symbol != null && symbol.Kind == SymbolKind.Function)
            {
                var parameters = symbol.Parameters ?? new List<ParameterInfo>();
                return Task.FromResult<SignatureHelp?>(new SignatureHelp
                {
                    Signatures = new Container<SignatureInformation>(new SignatureInformation
                    {
                        Label = symbol.Type ?? $"{functionName}()",
                        Documentation = symbol.Documentation,
                        Parameters = new Container<ParameterInformation>(
                            parameters.Select(p => new ParameterInformation
                            {
                                Label = p.Name,
                                Documentation = p.Documentation ?? p.Type
                            }))
                    }),
                    ActiveSignature = 0,
                    ActiveParameter = paramIndex
                });
            }

            return Task.FromResult<SignatureHelp?>(null);
        }

        /// <inheritdoc/>
        public SignatureHelpRegistrationOptions GetRegistrationOptions(
            SignatureHelpCapability capability,
            ClientCapabilities clientCapabilities)
        {
            return new SignatureHelpRegistrationOptions
            {
                DocumentSelector = TextDocumentSelector.ForLanguage("nsl"),
                TriggerCharacters = new Container<string>("(", ","),
                RetriggerCharacters = new Container<string>(",")
            };
        }

        private (string? functionName, int paramIndex) FindFunctionContext(DocumentState document, Position position)
        {
            var line = position.Line < document.Lines.Length
                ? document.Lines[(int)position.Line]
                : "";

            var col = (int)System.Math.Min(position.Character, line.Length);

            // Walk backwards to find opening paren
            int parenDepth = 0;
            int paramIndex = 0;
            int funcEnd = -1;

            for (int i = col - 1; i >= 0; i--)
            {
                var c = line[i];

                if (c == ')' || c == ']' || c == '}')
                    parenDepth++;
                else if (c == '(' && parenDepth == 0)
                {
                    funcEnd = i;
                    break;
                }
                else if (c == '(' || c == '[' || c == '{')
                    parenDepth--;
                else if (c == ',' && parenDepth == 0)
                    paramIndex++;
            }

            if (funcEnd < 0)
                return (null, 0);

            // Find function name
            int funcStart = funcEnd - 1;
            while (funcStart >= 0 && (char.IsLetterOrDigit(line[funcStart]) || line[funcStart] == '_'))
                funcStart--;
            funcStart++;

            if (funcStart >= funcEnd)
                return (null, 0);

            var functionName = line[funcStart..funcEnd];
            return (functionName, paramIndex);
        }
    }
}
