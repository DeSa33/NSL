using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NSL.Debugger
{
    /// <summary>
    /// NSL Debug Adapter implementing the Debug Adapter Protocol (DAP).
    ///
    /// DAP is a wire protocol for communication between a development tool
    /// and a debugger. This adapter enables debugging NSL programs in
    /// VSCode and other DAP-compatible editors.
    ///
    /// Supported features:
    /// - Breakpoints (line, conditional, logpoints)
    /// - Step over, step into, step out
    /// - Variable inspection
    /// - Stack trace viewing
    /// - Expression evaluation
    /// - Exception breakpoints
    /// </summary>
    public class NslDebugAdapter
    {
        private readonly Stream _input;
        private readonly Stream _output;
        private readonly DebugAdapterOptions _options;
        private readonly DebugSession _session;
        private readonly object _outputLock = new();
        private int _sequenceNumber = 1;
        private bool _running = true;

        public NslDebugAdapter(Stream input, Stream output, DebugAdapterOptions options)
        {
            _input = input;
            _output = output;
            _options = options;
            _session = new DebugSession(this);
        }

        /// <summary>
        /// Run the debug adapter, processing messages until terminated.
        /// </summary>
        public void Run()
        {
            try
            {
                var reader = new StreamReader(_input, Encoding.UTF8);

                while (_running)
                {
                    var message = ReadMessage(reader);
                    if (message == null)
                        break;

                    ProcessMessage(message);
                }
            }
            catch (Exception ex)
            {
                LogError($"Debug adapter error: {ex.Message}");
            }
        }

        private JObject? ReadMessage(StreamReader reader)
        {
            try
            {
                // Read headers
                var headers = new Dictionary<string, string>();
                string? line;

                while (!string.IsNullOrEmpty(line = reader.ReadLine()))
                {
                    var parts = line.Split(new[] { ": " }, 2, StringSplitOptions.None);
                    if (parts.Length == 2)
                    {
                        headers[parts[0]] = parts[1];
                    }
                }

                if (!headers.TryGetValue("Content-Length", out var lengthStr) ||
                    !int.TryParse(lengthStr, out var contentLength))
                {
                    return null;
                }

                // Read content
                var buffer = new char[contentLength];
                var read = 0;
                while (read < contentLength)
                {
                    var n = reader.Read(buffer, read, contentLength - read);
                    if (n == 0) break;
                    read += n;
                }

                var content = new string(buffer, 0, read);
                LogTrace($"<-- {content}");

                return JObject.Parse(content);
            }
            catch
            {
                return null;
            }
        }

        private void ProcessMessage(JObject message)
        {
            var type = message["type"]?.Value<string>();
            var seq = message["seq"]?.Value<int>() ?? 0;

            switch (type)
            {
                case "request":
                    HandleRequest(message, seq);
                    break;
                case "response":
                    // We don't expect responses from the client
                    break;
                case "event":
                    // We don't expect events from the client
                    break;
            }
        }

        private void HandleRequest(JObject message, int seq)
        {
            var command = message["command"]?.Value<string>() ?? "";
            var arguments = message["arguments"] as JObject;

            try
            {
                var (success, body, errorMessage) = command switch
                {
                    "initialize" => HandleInitialize(arguments),
                    "launch" => HandleLaunch(arguments),
                    "attach" => HandleAttach(arguments),
                    "disconnect" => HandleDisconnect(arguments),
                    "setBreakpoints" => HandleSetBreakpoints(arguments),
                    "setFunctionBreakpoints" => HandleSetFunctionBreakpoints(arguments),
                    "setExceptionBreakpoints" => HandleSetExceptionBreakpoints(arguments),
                    "configurationDone" => HandleConfigurationDone(arguments),
                    "continue" => HandleContinue(arguments),
                    "next" => HandleNext(arguments),
                    "stepIn" => HandleStepIn(arguments),
                    "stepOut" => HandleStepOut(arguments),
                    "pause" => HandlePause(arguments),
                    "threads" => HandleThreads(arguments),
                    "stackTrace" => HandleStackTrace(arguments),
                    "scopes" => HandleScopes(arguments),
                    "variables" => HandleVariables(arguments),
                    "evaluate" => HandleEvaluate(arguments),
                    "source" => HandleSource(arguments),
                    "loadedSources" => HandleLoadedSources(arguments),
                    "terminate" => HandleTerminate(arguments),
                    _ => (false, null, $"Unknown command: {command}")
                };

                SendResponse(seq, command, success, body, errorMessage);
            }
            catch (Exception ex)
            {
                SendResponse(seq, command, false, null, ex.Message);
            }
        }

        #region Request Handlers

        private (bool, object?, string?) HandleInitialize(JObject? args)
        {
            var capabilities = new Dictionary<string, object>
            {
                ["supportsConfigurationDoneRequest"] = true,
                ["supportsFunctionBreakpoints"] = true,
                ["supportsConditionalBreakpoints"] = true,
                ["supportsHitConditionalBreakpoints"] = true,
                ["supportsEvaluateForHovers"] = true,
                ["supportsStepBack"] = false,
                ["supportsSetVariable"] = true,
                ["supportsRestartFrame"] = false,
                ["supportsGotoTargetsRequest"] = false,
                ["supportsStepInTargetsRequest"] = false,
                ["supportsCompletionsRequest"] = true,
                ["supportsModulesRequest"] = false,
                ["supportsRestartRequest"] = true,
                ["supportsExceptionOptions"] = true,
                ["supportsValueFormattingOptions"] = true,
                ["supportsExceptionInfoRequest"] = true,
                ["supportTerminateDebuggee"] = true,
                ["supportsLogPoints"] = true,
                ["supportsLoadedSourcesRequest"] = true,
                ["supportsDataBreakpoints"] = false,
                ["supportsReadMemoryRequest"] = false,
                ["supportsDisassembleRequest"] = false,
                ["supportsCancelRequest"] = false,
                ["supportsBreakpointLocationsRequest"] = true,
                ["supportsClipboardContext"] = false,
                ["supportsSteppingGranularity"] = true,
                ["supportsInstructionBreakpoints"] = false,
                ["supportsExceptionFilterOptions"] = true,
                ["exceptionBreakpointFilters"] = new[]
                {
                    new { filter = "all", label = "All Exceptions", @default = false },
                    new { filter = "uncaught", label = "Uncaught Exceptions", @default = true }
                }
            };

            // Send initialized event after responding
            Task.Run(() =>
            {
                Thread.Sleep(100);
                SendEvent("initialized");
            });

            return (true, capabilities, null);
        }

        private (bool, object?, string?) HandleLaunch(JObject? args)
        {
            var program = args?["program"]?.Value<string>();
            var workingDir = args?["cwd"]?.Value<string>();
            var stopOnEntry = args?["stopOnEntry"]?.Value<bool>() ?? false;
            var programArgs = args?["args"]?.ToObject<string[]>() ?? Array.Empty<string>();

            if (string.IsNullOrEmpty(program))
            {
                return (false, null, "Program path not specified");
            }

            if (!File.Exists(program))
            {
                return (false, null, $"Program not found: {program}");
            }

            _session.Launch(program, workingDir, stopOnEntry, programArgs);
            return (true, null, null);
        }

        private (bool, object?, string?) HandleAttach(JObject? args)
        {
            // NSL doesn't support attaching to running processes yet
            return (false, null, "Attach is not supported. Use 'launch' instead.");
        }

        private (bool, object?, string?) HandleDisconnect(JObject? args)
        {
            var terminateDebuggee = args?["terminateDebuggee"]?.Value<bool>() ?? false;
            _session.Disconnect(terminateDebuggee);
            _running = false;
            return (true, null, null);
        }

        private (bool, object?, string?) HandleSetBreakpoints(JObject? args)
        {
            var source = args?["source"];
            var path = source?["path"]?.Value<string>();
            var breakpoints = args?["breakpoints"]?.ToObject<List<SourceBreakpointRequest>>()
                ?? new List<SourceBreakpointRequest>();

            if (string.IsNullOrEmpty(path))
            {
                return (false, null, "Source path not specified");
            }

            var result = _session.SetBreakpoints(path, breakpoints);
            return (true, new { breakpoints = result }, null);
        }

        private (bool, object?, string?) HandleSetFunctionBreakpoints(JObject? args)
        {
            var breakpoints = args?["breakpoints"]?.ToObject<List<FunctionBreakpointRequest>>()
                ?? new List<FunctionBreakpointRequest>();

            var result = _session.SetFunctionBreakpoints(breakpoints);
            return (true, new { breakpoints = result }, null);
        }

        private (bool, object?, string?) HandleSetExceptionBreakpoints(JObject? args)
        {
            var filters = args?["filters"]?.ToObject<string[]>() ?? Array.Empty<string>();
            _session.SetExceptionBreakpoints(filters);
            return (true, null, null);
        }

        private (bool, object?, string?) HandleConfigurationDone(JObject? args)
        {
            _session.ConfigurationDone();
            return (true, null, null);
        }

        private (bool, object?, string?) HandleContinue(JObject? args)
        {
            var threadId = args?["threadId"]?.Value<int>() ?? 1;
            _session.Continue(threadId);
            return (true, new { allThreadsContinued = true }, null);
        }

        private (bool, object?, string?) HandleNext(JObject? args)
        {
            var threadId = args?["threadId"]?.Value<int>() ?? 1;
            _session.StepOver(threadId);
            return (true, null, null);
        }

        private (bool, object?, string?) HandleStepIn(JObject? args)
        {
            var threadId = args?["threadId"]?.Value<int>() ?? 1;
            _session.StepIn(threadId);
            return (true, null, null);
        }

        private (bool, object?, string?) HandleStepOut(JObject? args)
        {
            var threadId = args?["threadId"]?.Value<int>() ?? 1;
            _session.StepOut(threadId);
            return (true, null, null);
        }

        private (bool, object?, string?) HandlePause(JObject? args)
        {
            var threadId = args?["threadId"]?.Value<int>() ?? 1;
            _session.Pause(threadId);
            return (true, null, null);
        }

        private (bool, object?, string?) HandleThreads(JObject? args)
        {
            var threads = _session.GetThreads();
            return (true, new { threads }, null);
        }

        private (bool, object?, string?) HandleStackTrace(JObject? args)
        {
            var threadId = args?["threadId"]?.Value<int>() ?? 1;
            var startFrame = args?["startFrame"]?.Value<int>() ?? 0;
            var levels = args?["levels"]?.Value<int>() ?? 20;

            var (frames, totalFrames) = _session.GetStackTrace(threadId, startFrame, levels);
            return (true, new { stackFrames = frames, totalFrames }, null);
        }

        private (bool, object?, string?) HandleScopes(JObject? args)
        {
            var frameId = args?["frameId"]?.Value<int>() ?? 0;
            var scopes = _session.GetScopes(frameId);
            return (true, new { scopes }, null);
        }

        private (bool, object?, string?) HandleVariables(JObject? args)
        {
            var variablesReference = args?["variablesReference"]?.Value<int>() ?? 0;
            var start = args?["start"]?.Value<int>();
            var count = args?["count"]?.Value<int>();

            var variables = _session.GetVariables(variablesReference, start, count);
            return (true, new { variables }, null);
        }

        private (bool, object?, string?) HandleEvaluate(JObject? args)
        {
            var expression = args?["expression"]?.Value<string>() ?? "";
            var frameId = args?["frameId"]?.Value<int>();
            var context = args?["context"]?.Value<string>() ?? "watch";

            var (result, type, variablesReference) = _session.Evaluate(expression, frameId, context);
            return (true, new { result, type, variablesReference }, null);
        }

        private (bool, object?, string?) HandleSource(JObject? args)
        {
            var sourceReference = args?["sourceReference"]?.Value<int>() ?? 0;
            var content = _session.GetSource(sourceReference);
            return (true, new { content }, null);
        }

        private (bool, object?, string?) HandleLoadedSources(JObject? args)
        {
            var sources = _session.GetLoadedSources();
            return (true, new { sources }, null);
        }

        private (bool, object?, string?) HandleTerminate(JObject? args)
        {
            _session.Terminate();
            return (true, null, null);
        }

        #endregion

        #region Event Sending

        public void SendEvent(string eventType, object? body = null)
        {
            var evt = new Dictionary<string, object>
            {
                ["seq"] = Interlocked.Increment(ref _sequenceNumber),
                ["type"] = "event",
                ["event"] = eventType
            };

            if (body != null)
            {
                evt["body"] = body;
            }

            SendMessage(evt);
        }

        public void SendStoppedEvent(string reason, int threadId, string? description = null, string? text = null)
        {
            var body = new Dictionary<string, object>
            {
                ["reason"] = reason,
                ["threadId"] = threadId,
                ["allThreadsStopped"] = true
            };

            if (description != null) body["description"] = description;
            if (text != null) body["text"] = text;

            SendEvent("stopped", body);
        }

        public void SendOutputEvent(string category, string output, string? source = null, int? line = null)
        {
            var body = new Dictionary<string, object>
            {
                ["category"] = category,
                ["output"] = output
            };

            if (source != null)
            {
                body["source"] = new { path = source };
                if (line.HasValue) body["line"] = line.Value;
            }

            SendEvent("output", body);
        }

        public void SendTerminatedEvent()
        {
            SendEvent("terminated");
        }

        public void SendExitedEvent(int exitCode)
        {
            SendEvent("exited", new { exitCode });
        }

        #endregion

        private void SendResponse(int requestSeq, string command, bool success, object? body, string? message)
        {
            var response = new Dictionary<string, object>
            {
                ["seq"] = Interlocked.Increment(ref _sequenceNumber),
                ["type"] = "response",
                ["request_seq"] = requestSeq,
                ["success"] = success,
                ["command"] = command
            };

            if (body != null)
            {
                response["body"] = body;
            }

            if (!success && message != null)
            {
                response["message"] = message;
            }

            SendMessage(response);
        }

        private void SendMessage(object message)
        {
            lock (_outputLock)
            {
                var json = JsonConvert.SerializeObject(message);
                var content = Encoding.UTF8.GetBytes(json);
                var header = Encoding.ASCII.GetBytes($"Content-Length: {content.Length}\r\n\r\n");

                LogTrace($"--> {json}");

                _output.Write(header, 0, header.Length);
                _output.Write(content, 0, content.Length);
                _output.Flush();
            }
        }

        private void LogTrace(string message)
        {
            if (_options.TraceLevel >= TraceLevel.Verbose)
            {
                if (_options.LogFile != null)
                {
                    File.AppendAllText(_options.LogFile, message + Environment.NewLine);
                }
            }
        }

        private void LogError(string message)
        {
            if (_options.LogFile != null)
            {
                File.AppendAllText(_options.LogFile, $"ERROR: {message}{Environment.NewLine}");
            }
        }
    }

    #region Request Types

    public class SourceBreakpointRequest
    {
        [JsonProperty("line")]
        public int Line { get; set; }

        [JsonProperty("column")]
        public int? Column { get; set; }

        [JsonProperty("condition")]
        public string? Condition { get; set; }

        [JsonProperty("hitCondition")]
        public string? HitCondition { get; set; }

        [JsonProperty("logMessage")]
        public string? LogMessage { get; set; }
    }

    public class FunctionBreakpointRequest
    {
        [JsonProperty("name")]
        public string Name { get; set; } = "";

        [JsonProperty("condition")]
        public string? Condition { get; set; }

        [JsonProperty("hitCondition")]
        public string? HitCondition { get; set; }
    }

    #endregion
}
