using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NSL.Core;
using NSL.Lexer;
using NSL.Parser;
using NSL.Core.AST;

namespace NSL.Debugger
{
    /// <summary>
    /// Manages an NSL debug session.
    ///
    /// This class handles:
    /// - Program execution with debugging
    /// - Breakpoint management
    /// - Stack frame tracking
    /// - Variable inspection
    /// - Expression evaluation
    /// </summary>
    public class DebugSession
    {
        private readonly NslDebugAdapter _adapter;
        private readonly Dictionary<string, List<NslBreakpoint>> _breakpoints = new();
        private readonly Dictionary<string, NslBreakpoint> _functionBreakpoints = new();
        private readonly List<NslStackFrame> _stackFrames = new();
        private readonly Dictionary<int, VariableScope> _variableScopes = new();
        private readonly List<SourceInfo> _loadedSources = new();

        private string? _programPath;
        private string? _workingDirectory;
        private bool _stopOnEntry;
        private string[]? _programArgs;

        private Thread? _executionThread;
        private ManualResetEventSlim _pauseEvent = new(true);
        private StepAction _stepAction = StepAction.None;
        private int _stepDepth = 0;
        private bool _terminated;
        private bool _breakOnAllExceptions;
        private bool _breakOnUncaughtExceptions = true;

        private int _currentLine = 1;
        private string? _currentFile;
        private int _nextVariableReference = 1000;
        private readonly object _lock = new();

        // NSL Interpreter integration
        private NSLInterpreter? _interpreter;
        private Dictionary<string, object?> _currentLocals = new();
        private Dictionary<string, object?> _currentGlobals = new();

        public DebugSession(NslDebugAdapter adapter)
        {
            _adapter = adapter;
        }

        /// <summary>
        /// Debug hook called before each statement execution
        /// </summary>
        private bool OnStatementExecuting(int line, string file, Dictionary<string, object?> locals, Dictionary<string, object?> globals)
        {
            if (_terminated) return false;

            _currentLine = line;
            _currentFile = file;
            _currentLocals = new Dictionary<string, object?>(locals);
            _currentGlobals = new Dictionary<string, object?>(globals);

            // Check for breakpoints
            if (ShouldBreak(file, line, ""))
            {
                PauseExecution("breakpoint");
            }

            // Check for step actions
            if (ShouldStopForStep())
            {
                PauseExecution("step");
            }

            return !_terminated;
        }

        #region Session Control

        public void Launch(string program, string? workingDir, bool stopOnEntry, string[] args)
        {
            _programPath = program;
            _workingDirectory = workingDir ?? Path.GetDirectoryName(program);
            _stopOnEntry = stopOnEntry;
            _programArgs = args;

            // Add source to loaded sources
            _loadedSources.Add(new SourceInfo
            {
                Name = Path.GetFileName(program),
                Path = program,
                SourceReference = 0
            });

            _adapter.SendEvent("process", new
            {
                name = Path.GetFileName(program),
                systemProcessId = Environment.ProcessId,
                isLocalProcess = true,
                startMethod = "launch"
            });
        }

        public void ConfigurationDone()
        {
            // Start execution after configuration is complete
            _executionThread = new Thread(ExecuteProgram)
            {
                Name = "NSL Execution Thread",
                IsBackground = true
            };
            _executionThread.Start();
        }

        public void Disconnect(bool terminate)
        {
            if (terminate)
            {
                Terminate();
            }
            _pauseEvent.Set();
        }

        public void Terminate()
        {
            _terminated = true;
            _pauseEvent.Set();
        }

        #endregion

        #region Execution Control

        public void Continue(int threadId)
        {
            _stepAction = StepAction.None;
            _pauseEvent.Set();
        }

        public void StepOver(int threadId)
        {
            _stepAction = StepAction.StepOver;
            _stepDepth = _stackFrames.Count;
            _pauseEvent.Set();
        }

        public void StepIn(int threadId)
        {
            _stepAction = StepAction.StepIn;
            _pauseEvent.Set();
        }

        public void StepOut(int threadId)
        {
            _stepAction = StepAction.StepOut;
            _stepDepth = _stackFrames.Count;
            _pauseEvent.Set();
        }

        public void Pause(int threadId)
        {
            _stepAction = StepAction.Pause;
            _pauseEvent.Reset();
        }

        private void ExecuteProgram()
        {
            try
            {
                // Notify that the thread has started
                _adapter.SendEvent("thread", new { reason = "started", threadId = 1 });

                if (_stopOnEntry)
                {
                    _currentLine = 1;
                    _currentFile = _programPath;
                    PauseExecution("entry");
                }

                // Execute the program
                if (_programPath != null)
                {
                    ExecuteNslProgram(_programPath);
                }

                // Program completed
                if (!_terminated)
                {
                    _adapter.SendExitedEvent(0);
                    _adapter.SendTerminatedEvent();
                }
            }
            catch (Exception ex)
            {
                _adapter.SendOutputEvent("stderr", $"Runtime error: {ex.Message}\n");

                if (_breakOnAllExceptions || _breakOnUncaughtExceptions)
                {
                    PauseExecution("exception", ex.Message);
                }

                _adapter.SendExitedEvent(1);
                _adapter.SendTerminatedEvent();
            }
        }

        private void ExecuteNslProgram(string path)
        {
            // Read and parse the program
            var source = File.ReadAllText(path);
            _currentFile = path;

            try
            {
                // Create lexer and parser
                var lexer = new NSLLexer(source);
                var tokens = lexer.Tokenize().ToList();
                var parser = new NSLParser();
                var ast = parser.Parse(tokens);

                // Create interpreter with debug hooks
                _interpreter = new NSLInterpreter();

                // Set up output capture
                _interpreter.SetOutputWriter(new DebugOutputWriter(_adapter, path));

                // Register debug callback for statement execution
                _interpreter.SetDebugCallback((line, file, locals, globals) =>
                {
                    return OnStatementExecuting(line, file ?? path, locals, globals);
                });

                // Execute with debugging
                _interpreter.Execute(ast);
            }
            catch (Exception ex) when (!_terminated)
            {
                _adapter.SendOutputEvent("stderr", $"Execution error: {ex.Message}\n", path, _currentLine);
                throw;
            }
        }

        /// <summary>
        /// Output writer that sends output to debug console
        /// </summary>
        private class DebugOutputWriter : TextWriter
        {
            private readonly NslDebugAdapter _adapter;
            private readonly string _source;

            public DebugOutputWriter(NslDebugAdapter adapter, string source)
            {
                _adapter = adapter;
                _source = source;
            }

            public override Encoding Encoding => Encoding.UTF8;

            public override void Write(string? value)
            {
                if (value != null)
                    _adapter.SendOutputEvent("stdout", value, _source);
            }

            public override void WriteLine(string? value)
            {
                _adapter.SendOutputEvent("stdout", (value ?? "") + "\n", _source);
            }
        }

        private bool ShouldBreak(string path, int line, string lineContent)
        {
            if (!_breakpoints.TryGetValue(path, out var bps))
                return false;

            var bp = bps.FirstOrDefault(b => b.Line == line && b.Verified);
            if (bp == null)
                return false;

            // Check hit count
            bp.HitCount++;
            if (bp.HitCondition != null)
            {
                if (!EvaluateHitCondition(bp.HitCondition, bp.HitCount))
                    return false;
            }

            // Check condition
            if (bp.Condition != null)
            {
                if (!EvaluateCondition(bp.Condition))
                    return false;
            }

            // Handle logpoint
            if (bp.LogMessage != null)
            {
                var message = InterpolateLogMessage(bp.LogMessage);
                _adapter.SendOutputEvent("console", message + "\n", path, line);
                return false; // Don't break for logpoints
            }

            return true;
        }

        private bool ShouldStopForStep()
        {
            switch (_stepAction)
            {
                case StepAction.StepIn:
                    return true;
                case StepAction.StepOver:
                    return _stackFrames.Count <= _stepDepth;
                case StepAction.StepOut:
                    return _stackFrames.Count < _stepDepth;
                case StepAction.Pause:
                    _stepAction = StepAction.None;
                    return true;
                default:
                    return false;
            }
        }

        private void PauseExecution(string reason, string? description = null)
        {
            _stepAction = StepAction.None;
            _pauseEvent.Reset();

            _adapter.SendStoppedEvent(reason, 1, description);

            // Wait until resumed
            _pauseEvent.Wait();
        }

        #endregion

        #region Breakpoints

        public List<BreakpointResponse> SetBreakpoints(string path, List<SourceBreakpointRequest> requests)
        {
            var results = new List<BreakpointResponse>();
            var newBreakpoints = new List<NslBreakpoint>();

            foreach (var req in requests)
            {
                var bp = new NslBreakpoint
                {
                    Id = _nextVariableReference++,
                    Line = req.Line,
                    Column = req.Column,
                    Condition = req.Condition,
                    HitCondition = req.HitCondition,
                    LogMessage = req.LogMessage,
                    Verified = File.Exists(path),
                    Source = path
                };

                newBreakpoints.Add(bp);

                results.Add(new BreakpointResponse
                {
                    Id = bp.Id,
                    Verified = bp.Verified,
                    Line = bp.Line,
                    Column = bp.Column,
                    Message = bp.Verified ? null : "Source not found"
                });
            }

            lock (_lock)
            {
                _breakpoints[path] = newBreakpoints;
            }

            return results;
        }

        public List<BreakpointResponse> SetFunctionBreakpoints(List<FunctionBreakpointRequest> requests)
        {
            var results = new List<BreakpointResponse>();

            lock (_lock)
            {
                _functionBreakpoints.Clear();

                foreach (var req in requests)
                {
                    var bp = new NslBreakpoint
                    {
                        Id = _nextVariableReference++,
                        FunctionName = req.Name,
                        Condition = req.Condition,
                        HitCondition = req.HitCondition,
                        Verified = true // Will be verified at runtime
                    };

                    _functionBreakpoints[req.Name] = bp;

                    results.Add(new BreakpointResponse
                    {
                        Id = bp.Id,
                        Verified = bp.Verified
                    });
                }
            }

            return results;
        }

        public void SetExceptionBreakpoints(string[] filters)
        {
            _breakOnAllExceptions = filters.Contains("all");
            _breakOnUncaughtExceptions = filters.Contains("uncaught");
        }

        #endregion

        #region Stack and Variables

        public List<ThreadInfo> GetThreads()
        {
            return new List<ThreadInfo>
            {
                new ThreadInfo { Id = 1, Name = "Main Thread" }
            };
        }

        public (List<StackFrameInfo>, int) GetStackTrace(int threadId, int startFrame, int levels)
        {
            var frames = new List<StackFrameInfo>();
            int total;

            lock (_lock)
            {
                total = _stackFrames.Count + 1; // +1 for current location

                // Current location
                if (startFrame == 0 && _currentFile != null)
                {
                    frames.Add(new StackFrameInfo
                    {
                        Id = 0,
                        Name = "<current>",
                        Source = new SourceInfo { Path = _currentFile, Name = Path.GetFileName(_currentFile) },
                        Line = _currentLine,
                        Column = 1
                    });
                }

                // Stack frames
                var offset = startFrame > 0 ? startFrame - 1 : 0;
                for (int i = offset; i < _stackFrames.Count && frames.Count < levels; i++)
                {
                    var sf = _stackFrames[_stackFrames.Count - 1 - i];
                    frames.Add(new StackFrameInfo
                    {
                        Id = i + 1,
                        Name = sf.FunctionName,
                        Source = new SourceInfo { Path = sf.File, Name = Path.GetFileName(sf.File) },
                        Line = sf.Line,
                        Column = 1
                    });
                }
            }

            return (frames, total);
        }

        public List<ScopeInfo> GetScopes(int frameId)
        {
            var scopes = new List<ScopeInfo>();

            // Local scope
            var localRef = RegisterVariableScope(frameId, ScopeType.Local);
            scopes.Add(new ScopeInfo
            {
                Name = "Local",
                VariablesReference = localRef,
                Expensive = false
            });

            // Global scope
            var globalRef = RegisterVariableScope(frameId, ScopeType.Global);
            scopes.Add(new ScopeInfo
            {
                Name = "Global",
                VariablesReference = globalRef,
                Expensive = false
            });

            return scopes;
        }

        public List<VariableInfo> GetVariables(int variablesReference, int? start, int? count)
        {
            var variables = new List<VariableInfo>();

            if (_variableScopes.TryGetValue(variablesReference, out var scope))
            {
                Dictionary<string, object?> varsToShow;

                if (scope.Type == ScopeType.Local)
                {
                    varsToShow = _currentLocals;
                }
                else if (scope.Type == ScopeType.Global)
                {
                    varsToShow = _currentGlobals;
                }
                else
                {
                    varsToShow = new Dictionary<string, object?>();
                }

                foreach (var kvp in varsToShow)
                {
                    var varRef = 0;
                    var valueStr = FormatValue(kvp.Value, out var type, out var hasChildren);

                    if (hasChildren)
                    {
                        varRef = RegisterObjectForExpansion(kvp.Value);
                    }

                    variables.Add(new VariableInfo
                    {
                        Name = kvp.Key,
                        Value = valueStr,
                        Type = type,
                        VariablesReference = varRef
                    });
                }
            }
            else
            {
                // Check if this is an expandable object reference
                if (_expandableObjects.TryGetValue(variablesReference, out var obj))
                {
                    variables.AddRange(GetObjectChildren(obj));
                }
            }

            return variables;
        }

        private readonly Dictionary<int, object?> _expandableObjects = new();

        private int RegisterObjectForExpansion(object? value)
        {
            var reference = _nextVariableReference++;
            _expandableObjects[reference] = value;
            return reference;
        }

        private string FormatValue(object? value, out string type, out bool hasChildren)
        {
            hasChildren = false;

            if (value == null)
            {
                type = "null";
                return "nil";
            }

            switch (value)
            {
                case double d:
                    type = "number";
                    return d.ToString();
                case int i:
                    type = "number";
                    return i.ToString();
                case long l:
                    type = "number";
                    return l.ToString();
                case string s:
                    type = "string";
                    return $"\"{s}\"";
                case bool b:
                    type = "boolean";
                    return b ? "true" : "false";
                case List<object?> list:
                    type = "array";
                    hasChildren = list.Count > 0;
                    return $"[{list.Count} items]";
                case Dictionary<string, object?> dict:
                    type = "object";
                    hasChildren = dict.Count > 0;
                    return $"{{{dict.Count} properties}}";
                case NSLFunction func:
                    type = "function";
                    return $"<function {func.Name}>";
                default:
                    type = value.GetType().Name;
                    hasChildren = false;
                    return value.ToString() ?? "?";
            }
        }

        private List<VariableInfo> GetObjectChildren(object? value)
        {
            var children = new List<VariableInfo>();

            if (value is List<object?> list)
            {
                for (int i = 0; i < list.Count; i++)
                {
                    var varRef = 0;
                    var valueStr = FormatValue(list[i], out var type, out var hasChildren);
                    if (hasChildren) varRef = RegisterObjectForExpansion(list[i]);

                    children.Add(new VariableInfo
                    {
                        Name = $"[{i}]",
                        Value = valueStr,
                        Type = type,
                        VariablesReference = varRef
                    });
                }
            }
            else if (value is Dictionary<string, object?> dict)
            {
                foreach (var kvp in dict)
                {
                    var varRef = 0;
                    var valueStr = FormatValue(kvp.Value, out var type, out var hasChildren);
                    if (hasChildren) varRef = RegisterObjectForExpansion(kvp.Value);

                    children.Add(new VariableInfo
                    {
                        Name = kvp.Key,
                        Value = valueStr,
                        Type = type,
                        VariablesReference = varRef
                    });
                }
            }

            return children;
        }

        public (string, string?, int) Evaluate(string expression, int? frameId, string context)
        {
            try
            {
                // Try to evaluate using the interpreter if available
                if (_interpreter != null)
                {
                    var result = _interpreter.EvaluateExpression(expression);
                    var valueStr = FormatValue(result, out var type, out var hasChildren);
                    var varRef = hasChildren ? RegisterObjectForExpansion(result) : 0;
                    return (valueStr, type, varRef);
                }

                // Fallback: check if it's a known variable
                if (_currentLocals.TryGetValue(expression, out var localValue))
                {
                    var valueStr = FormatValue(localValue, out var type, out var hasChildren);
                    var varRef = hasChildren ? RegisterObjectForExpansion(localValue) : 0;
                    return (valueStr, type, varRef);
                }

                if (_currentGlobals.TryGetValue(expression, out var globalValue))
                {
                    var valueStr = FormatValue(globalValue, out var type, out var hasChildren);
                    var varRef = hasChildren ? RegisterObjectForExpansion(globalValue) : 0;
                    return (valueStr, type, varRef);
                }

                // Simple literal evaluation
                if (int.TryParse(expression, out var num))
                {
                    return (num.ToString(), "number", 0);
                }
                if (double.TryParse(expression, out var dbl))
                {
                    return (dbl.ToString(), "number", 0);
                }
                if (expression.StartsWith("\"") && expression.EndsWith("\""))
                {
                    return (expression, "string", 0);
                }

                return ($"undefined: {expression}", "undefined", 0);
            }
            catch (Exception ex)
            {
                return ($"Error: {ex.Message}", "error", 0);
            }
        }

        public string GetSource(int sourceReference)
        {
            if (sourceReference == 0 && _programPath != null && File.Exists(_programPath))
            {
                return File.ReadAllText(_programPath);
            }
            return "";
        }

        public List<SourceInfo> GetLoadedSources()
        {
            return _loadedSources.ToList();
        }

        #endregion

        #region Helpers

        private void PushStackFrame(string functionName, string file, int line)
        {
            lock (_lock)
            {
                _stackFrames.Add(new NslStackFrame
                {
                    FunctionName = functionName,
                    File = file,
                    Line = line
                });
            }
        }

        private void PopStackFrame()
        {
            lock (_lock)
            {
                if (_stackFrames.Count > 0)
                {
                    _stackFrames.RemoveAt(_stackFrames.Count - 1);
                }
            }
        }

        private int RegisterVariableScope(int frameId, ScopeType type)
        {
            var reference = _nextVariableReference++;
            _variableScopes[reference] = new VariableScope { FrameId = frameId, Type = type };
            return reference;
        }

        private string ExtractFunctionName(string line)
        {
            var fnIndex = line.IndexOf("fn ");
            if (fnIndex >= 0)
            {
                var rest = line[(fnIndex + 3)..];
                var parenIndex = rest.IndexOf('(');
                if (parenIndex > 0)
                {
                    return rest[..parenIndex].Trim();
                }
            }
            return "<anonymous>";
        }

        private string ExtractPrintContent(string line)
        {
            var start = line.IndexOf('(');
            var end = line.LastIndexOf(')');
            if (start >= 0 && end > start)
            {
                var content = line[(start + 1)..end];
                // Remove quotes if it's a string literal
                if (content.StartsWith("\"") && content.EndsWith("\""))
                {
                    content = content[1..^1];
                }
                return content;
            }
            return line;
        }

        private bool EvaluateCondition(string condition)
        {
            // Placeholder - would use NSL.Interpreter in real implementation
            return true;
        }

        private bool EvaluateHitCondition(string hitCondition, int hitCount)
        {
            // Simple hit condition evaluation
            if (int.TryParse(hitCondition, out var target))
            {
                return hitCount >= target;
            }
            if (hitCondition.StartsWith(">=") && int.TryParse(hitCondition[2..], out var gte))
            {
                return hitCount >= gte;
            }
            if (hitCondition.StartsWith("==") && int.TryParse(hitCondition[2..], out var eq))
            {
                return hitCount == eq;
            }
            return true;
        }

        private string InterpolateLogMessage(string message)
        {
            // Replace {expression} with evaluated values
            // Placeholder implementation
            return message;
        }

        #endregion
    }

    #region Types

    public enum StepAction
    {
        None,
        StepOver,
        StepIn,
        StepOut,
        Pause
    }

    public enum ScopeType
    {
        Local,
        Global,
        Closure
    }

    public class NslBreakpoint
    {
        public int Id { get; set; }
        public int Line { get; set; }
        public int? Column { get; set; }
        public string? Condition { get; set; }
        public string? HitCondition { get; set; }
        public string? LogMessage { get; set; }
        public string? FunctionName { get; set; }
        public bool Verified { get; set; }
        public string? Source { get; set; }
        public int HitCount { get; set; }
    }

    public class NslStackFrame
    {
        public string FunctionName { get; set; } = "";
        public string File { get; set; } = "";
        public int Line { get; set; }
    }

    public class VariableScope
    {
        public int FrameId { get; set; }
        public ScopeType Type { get; set; }
    }

    public class ThreadInfo
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
    }

    public class StackFrameInfo
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public SourceInfo? Source { get; set; }
        public int Line { get; set; }
        public int Column { get; set; }
    }

    public class SourceInfo
    {
        public string Name { get; set; } = "";
        public string Path { get; set; } = "";
        public int SourceReference { get; set; }
    }

    public class ScopeInfo
    {
        public string Name { get; set; } = "";
        public int VariablesReference { get; set; }
        public bool Expensive { get; set; }
    }

    public class VariableInfo
    {
        public string Name { get; set; } = "";
        public string Value { get; set; } = "";
        public string? Type { get; set; }
        public int VariablesReference { get; set; }
    }

    public class BreakpointResponse
    {
        public int Id { get; set; }
        public bool Verified { get; set; }
        public int? Line { get; set; }
        public int? Column { get; set; }
        public string? Message { get; set; }
    }

    #endregion
}
