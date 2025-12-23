#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { spawn, execSync } from "child_process";
import { readFile, writeFile, readdir, stat, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { join, resolve, dirname } from "path";

const NSL_PATH = "E:\\NSL.Interpreter\\src\\NSL.Console\\bin\\Release\\net8.0\\win-x64\\nsl.exe";
const MEMORY_DIR = join(process.env.USERPROFILE || "C:\\Users\\Default", ".nsl-memory");
const SESSION_DIR = join(MEMORY_DIR, "sessions");
const HISTORY_DIR = join(MEMORY_DIR, "history");
const MAX_OUTPUT_LINES = 500;
const MAX_OUTPUT_CHARS = 50000;

// Sensitive patterns to redact from output
const SENSITIVE_PATTERNS = [
  /api[_-]?key\s*[=:]\s*['"]?[\w-]+['"]?/gi,
  /password\s*[=:]\s*['"]?[^'"]+['"]?/gi,
  /secret\s*[=:]\s*['"]?[\w-]+['"]?/gi,
  /token\s*[=:]\s*['"]?[\w.-]+['"]?/gi,
  /aws_access_key_id\s*[=:]\s*\S+/gi,
  /aws_secret_access_key\s*[=:]\s*\S+/gi,
  /Bearer\s+[\w.-]+/gi,
];

// Ensure directories exist
if (!existsSync(MEMORY_DIR)) {
  await mkdir(MEMORY_DIR, { recursive: true });
}
if (!existsSync(SESSION_DIR)) {
  await mkdir(SESSION_DIR, { recursive: true });
}
if (!existsSync(HISTORY_DIR)) {
  await mkdir(HISTORY_DIR, { recursive: true });
}

// ===== OUTPUT UTILITIES =====

// Truncate large outputs to prevent token limit issues
function truncateOutput(output, maxLines = MAX_OUTPUT_LINES, maxChars = MAX_OUTPUT_CHARS) {
  if (!output || typeof output !== 'string') return output;

  let result = output;
  let truncated = false;

  // Truncate by character count first
  if (result.length > maxChars) {
    result = result.substring(0, maxChars);
    truncated = true;
  }

  // Then by line count
  const lines = result.split('\n');
  if (lines.length > maxLines) {
    result = lines.slice(0, maxLines).join('\n');
    truncated = true;
  }

  if (truncated) {
    const originalLines = output.split('\n').length;
    const originalChars = output.length;
    result += `\n\n... [OUTPUT TRUNCATED: ${originalLines} lines, ${originalChars} chars total]`;
  }

  return result;
}

// Sanitize output to remove sensitive information
function sanitizeOutput(output) {
  if (!output || typeof output !== 'string') return output;

  let sanitized = output;
  for (const pattern of SENSITIVE_PATTERNS) {
    sanitized = sanitized.replace(pattern, '[REDACTED]');
  }
  return sanitized;
}

// Process output: sanitize then truncate
function processOutput(output) {
  return truncateOutput(sanitizeOutput(output));
}

// Categorize errors for better error handling
function categorizeError(error) {
  if (!error) return 'unknown';
  const errorStr = error.toString().toLowerCase();

  if (errorStr.includes('parse error') || errorStr.includes('syntax error')) {
    return 'parse_error';
  }
  if (errorStr.includes('undefined variable') || errorStr.includes('undefined function')) {
    return 'reference_error';
  }
  if (errorStr.includes('type error') || errorStr.includes('cannot convert')) {
    return 'type_error';
  }
  if (errorStr.includes('timeout') || errorStr.includes('timed out')) {
    return 'timeout_error';
  }
  if (errorStr.includes('permission denied') || errorStr.includes('access denied')) {
    return 'permission_error';
  }
  if (errorStr.includes('file not found') || errorStr.includes('directory not found')) {
    return 'file_error';
  }
  if (errorStr.includes('network') || errorStr.includes('connection')) {
    return 'network_error';
  }
  if (errorStr.includes('out of memory') || errorStr.includes('stack overflow')) {
    return 'resource_error';
  }
  return 'runtime_error';
}

// ===== COMMAND HISTORY =====

const commandHistory = [];
const MAX_HISTORY = 1000;

async function logCommand(code, result, sessionId = null, options = {}) {
  const entry = {
    timestamp: new Date().toISOString(),
    code: code.substring(0, 1000), // Limit stored code size
    success: result.success !== false,
    executionTime: result.ExecutionTimeMs || result.executionTimeMs,
    sessionId,
    resultType: result.ResultType || typeof result.Result,
  };

  commandHistory.push(entry);
  if (commandHistory.length > MAX_HISTORY) {
    commandHistory.shift();
  }

  // Also save to disk periodically (every 10 commands)
  if (commandHistory.length % 10 === 0) {
    const historyFile = join(HISTORY_DIR, `history-${new Date().toISOString().split('T')[0]}.json`);
    try {
      let existing = [];
      if (existsSync(historyFile)) {
        existing = JSON.parse(await readFile(historyFile, 'utf-8'));
      }
      existing.push(...commandHistory.slice(-10));
      await writeFile(historyFile, JSON.stringify(existing, null, 2));
    } catch (e) {
      // Ignore history write errors
    }
  }

  return entry;
}

function getHistory(count = 50) {
  return commandHistory.slice(-count);
}

// ===== SESSION MANAGEMENT =====
// Sessions persist interpreter state across MCP calls
// Each session accumulates code that gets re-executed on each call

const sessions = new Map(); // In-memory session cache

class NSLSession {
  constructor(id) {
    this.id = id;
    this.code = [];  // Array of code blocks
    this.created = new Date().toISOString();
    this.lastUsed = this.created;
    this.callCount = 0;
    this.workingDir = process.cwd();  // Track working directory
    this.env = {};  // Session-specific environment variables
    this.restartCount = 0;
  }

  addCode(code) {
    this.code.push(code);
    this.lastUsed = new Date().toISOString();
    this.callCount++;
  }

  getFullContext() {
    return this.code.join("\n\n");
  }

  // Restart session - clears code but preserves metadata
  restart() {
    const oldCodeCount = this.code.length;
    this.code = [];
    this.lastUsed = new Date().toISOString();
    this.restartCount++;
    return oldCodeCount;
  }

  // Set working directory
  setWorkingDir(dir) {
    this.workingDir = dir;
  }

  // Set environment variable
  setEnv(key, value) {
    this.env[key] = value;
  }

  toJSON() {
    return {
      id: this.id,
      created: this.created,
      lastUsed: this.lastUsed,
      callCount: this.callCount,
      codeBlocks: this.code.length,
      code: this.code,
      workingDir: this.workingDir,
      env: this.env,
      restartCount: this.restartCount
    };
  }

  static fromJSON(json) {
    const session = new NSLSession(json.id);
    session.code = json.code || [];
    session.created = json.created;
    session.lastUsed = json.lastUsed;
    session.callCount = json.callCount || 0;
    session.workingDir = json.workingDir || process.cwd();
    session.env = json.env || {};
    session.restartCount = json.restartCount || 0;
    return session;
  }
}

// Load session from disk
async function loadSession(sessionId) {
  if (sessions.has(sessionId)) {
    return sessions.get(sessionId);
  }

  const sessionFile = join(SESSION_DIR, `${sessionId}.json`);
  if (existsSync(sessionFile)) {
    try {
      const data = await readFile(sessionFile, "utf-8");
      const session = NSLSession.fromJSON(JSON.parse(data));
      sessions.set(sessionId, session);
      return session;
    } catch (e) {
      // Corrupted session file, create new
    }
  }

  // Create new session
  const session = new NSLSession(sessionId);
  sessions.set(sessionId, session);
  return session;
}

// Save session to disk
async function saveSession(session) {
  const sessionFile = join(SESSION_DIR, `${session.id}.json`);
  await writeFile(sessionFile, JSON.stringify(session.toJSON(), null, 2));
}

// List all sessions
async function listSessions() {
  const result = [];
  if (existsSync(SESSION_DIR)) {
    const files = await readdir(SESSION_DIR);
    for (const file of files) {
      if (file.endsWith(".json")) {
        try {
          const data = await readFile(join(SESSION_DIR, file), "utf-8");
          const session = JSON.parse(data);
          result.push({
            id: session.id,
            created: session.created,
            lastUsed: session.lastUsed,
            callCount: session.callCount,
            codeBlocks: session.codeBlocks || session.code?.length || 0
          });
        } catch (e) {
          // Skip corrupted files
        }
      }
    }
  }
  return result.sort((a, b) => new Date(b.lastUsed) - new Date(a.lastUsed));
}

// Delete a session
async function deleteSession(sessionId) {
  sessions.delete(sessionId);
  const sessionFile = join(SESSION_DIR, `${sessionId}.json`);
  if (existsSync(sessionFile)) {
    const { unlink } = await import("fs/promises");
    await unlink(sessionFile);
    return true;
  }
  return false;
}

// Default session for convenience
const DEFAULT_SESSION = "default";

// Execute NSL code with session context
async function executeNSLWithSession(code, sessionId, options = {}) {
  const session = await loadSession(sessionId);

  // Use session's working directory if not overridden
  if (!options.workingDir) {
    options.workingDir = session.workingDir;
  }

  // Build full code: session context + new code
  const fullCode = session.code.length > 0
    ? `${session.getFullContext()}\n\n# --- New code ---\n${code}`
    : code;

  // Execute the full context
  const result = await executeNSL(fullCode, options);

  // Log the command
  await logCommand(code, result, sessionId, options);

  // If successful, add new code to session
  if (result.success !== false) {
    session.addCode(code);
    await saveSession(session);
  }

  // Add session info to result
  result.session = {
    id: session.id,
    callCount: session.callCount,
    codeBlocks: session.code.length,
    workingDir: session.workingDir,
    restartCount: session.restartCount
  };

  return result;
}

// Execute NSL code with full AI-native options
async function executeNSL(code, options = {}) {
  return new Promise((resolve, reject) => {
    const args = ["--eval", code];

    // Add AI-native flags
    if (options.json !== false) args.push("--json");
    if (options.gpu) args.push("--gpu");
    if (options.think) args.push("--think");
    if (options.trace) args.push("--trace");
    if (options.reflect) args.push("--reflect");
    if (options.explain) args.push("--explain");
    if (options.optimize) args.push("--optimize");
    if (options.vectorize) args.push("--vectorize");
    if (options.learn) args.push("--learn");
    if (options.sandbox) args.push("--sandbox");
    if (options.timeout) args.push("--timeout", String(options.timeout));
    if (options.context) args.push("--context", JSON.stringify(options.context));
    if (options.memory) args.push("--memory", options.memory);

    const nsl = spawn(NSL_PATH, args, {
      cwd: options.workingDir || process.cwd(),
      shell: false,  // Don't use shell to avoid character interpretation issues
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";

    nsl.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    nsl.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    nsl.on("close", (exitCode) => {
      try {
        // Try to parse as JSON first
        if (options.json !== false) {
          const result = JSON.parse(stdout);
          // Process string results for sanitization and truncation
          if (typeof result.Result === 'string') {
            result.Result = processOutput(result.Result);
          }
          // Categorize error types
          if (result.Error) {
            result.errorType = categorizeError(result.Error);
          }
          resolve(result);
        } else {
          resolve({
            success: true,
            result: processOutput(stdout.trim())
          });
        }
      } catch {
        // If not JSON, return raw output
        const errorType = stderr ? categorizeError(stderr) : undefined;
        resolve({
          success: exitCode === 0,
          result: processOutput(stdout.trim()),
          error: processOutput(stderr.trim()) || undefined,
          errorType,
          exitCode
        });
      }
    });

    nsl.on("error", (err) => {
      reject(err);
    });

    // Timeout
    const timeoutMs = options.timeout || 30000;
    setTimeout(() => {
      nsl.kill();
      reject(new Error(`NSL execution timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
}

// Get NSL introspection
async function getIntrospection() {
  return new Promise((resolve, reject) => {
    const nsl = spawn(NSL_PATH, ["--introspect", "--json"], { shell: false, windowsHide: true });
    let stdout = "";
    nsl.stdout.on("data", (data) => (stdout += data.toString()));
    nsl.on("close", () => {
      try {
        resolve(JSON.parse(stdout));
      } catch {
        resolve({ error: "Failed to parse introspection" });
      }
    });
    nsl.on("error", reject);
  });
}

// Get NSL capabilities
async function getCapabilities() {
  return new Promise((resolve, reject) => {
    const nsl = spawn(NSL_PATH, ["--capabilities", "--json"], { shell: false, windowsHide: true });
    let stdout = "";
    nsl.stdout.on("data", (data) => (stdout += data.toString()));
    nsl.on("close", () => {
      try {
        resolve(JSON.parse(stdout));
      } catch {
        resolve({ error: "Failed to parse capabilities" });
      }
    });
    nsl.on("error", reject);
  });
}

// Run benchmark
async function runBenchmark() {
  return new Promise((resolve, reject) => {
    const nsl = spawn(NSL_PATH, ["--benchmark", "--json"], { shell: false, windowsHide: true });
    let stdout = "";
    nsl.stdout.on("data", (data) => (stdout += data.toString()));
    nsl.on("close", () => {
      try {
        resolve(JSON.parse(stdout));
      } catch {
        resolve({ error: "Failed to parse benchmark" });
      }
    });
    nsl.on("error", reject);
  });
}

// Create the MCP server
const server = new Server(
  {
    name: "nsl-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// List resources (NSL memory files, scripts)
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  const resources = [];

  // Add memory files
  if (existsSync(MEMORY_DIR)) {
    const files = await readdir(MEMORY_DIR);
    for (const file of files) {
      if (file.endsWith(".json")) {
        resources.push({
          uri: `nsl://memory/${file}`,
          name: `Memory: ${file}`,
          mimeType: "application/json",
        });
      }
    }
  }

  // Add introspection as a resource
  resources.push({
    uri: "nsl://introspection",
    name: "NSL Introspection",
    mimeType: "application/json",
  });

  resources.push({
    uri: "nsl://capabilities",
    name: "NSL Capabilities",
    mimeType: "application/json",
  });

  return { resources };
});

// Read resources
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;

  if (uri === "nsl://introspection") {
    const introspection = await getIntrospection();
    return {
      contents: [
        {
          uri,
          mimeType: "application/json",
          text: JSON.stringify(introspection, null, 2),
        },
      ],
    };
  }

  if (uri === "nsl://capabilities") {
    const capabilities = await getCapabilities();
    return {
      contents: [
        {
          uri,
          mimeType: "application/json",
          text: JSON.stringify(capabilities, null, 2),
        },
      ],
    };
  }

  if (uri.startsWith("nsl://memory/")) {
    const filename = uri.replace("nsl://memory/", "");
    const filepath = join(MEMORY_DIR, filename);
    const content = await readFile(filepath, "utf-8");
    return {
      contents: [{ uri, mimeType: "application/json", text: content }],
    };
  }

  throw new Error(`Unknown resource: ${uri}`);
});

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "nsl",
        description: `Execute NSL (Neural Symbolic Language) code - an AI-native programming language with:
- Consciousness operators (|>, ~>, =>>, *>, +>) for AI-native data flow
- GPU tensor operations (gpu.init(), gpu.matmul(), etc.)
- Semantic file access (attention_read, stream_file)
- Python interoperability
- SESSION PERSISTENCE: Use session_id to maintain state across calls!

Use this INSTEAD of Bash when you want to think/compute in AI-native ways.
Returns structured JSON with result, type, execution time, and optional reasoning trace.

TIP: For incremental development, use session_id="dev" to keep functions/variables across calls.`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL code to execute",
            },
            session_id: {
              type: "string",
              description: "Optional session ID for persistent state across calls. Functions and variables defined in previous calls will be available.",
            },
            think: {
              type: "boolean",
              description: "Enable thinking/reasoning trace",
            },
            reflect: {
              type: "boolean",
              description: "Generate self-reflection on execution",
            },
            explain: {
              type: "boolean",
              description: "Generate human-readable explanation",
            },
            gpu: {
              type: "boolean",
              description: "Initialize GPU for tensor operations",
            },
            optimize: {
              type: "boolean",
              description: "Auto-optimize code before execution",
            },
            context: {
              type: "object",
              description: "Context object to inject as variables",
            },
            timeout: {
              type: "number",
              description: "Execution timeout in milliseconds",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "nsl_think",
        description: `Execute NSL code with full AI reasoning trace enabled.
Shows step-by-step thinking: INIT -> PARSE -> EXECUTE -> COMPLETE with timestamps.
Perfect for understanding how NSL processes your code.`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL code to execute with thinking trace",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "nsl_gpu",
        description: `Execute NSL code with GPU acceleration enabled.
Use for tensor operations, matrix math, neural network computations.
Auto-initializes CUDA context.`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL GPU code to execute",
            },
            vectorize: {
              type: "boolean",
              description: "Auto-vectorize operations to GPU",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "nsl_consciousness",
        description: `Execute NSL consciousness operators for AI-native data processing:
- |> (pipe) - Chain transformations left-to-right
- ~> (awareness) - Introspective flow with self-reference
- =>> (gradient) - Learning/adjustment with feedback
- *> (attention) - Focus mechanism with weights
- +> (superposition) - Quantum-like state superposition

These operators match how AI actually processes information internally.`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL consciousness code",
            },
            reflect: {
              type: "boolean",
              description: "Generate reflection on the consciousness operations",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "nsl_learn",
        description: `Execute NSL code and learn from the execution.
Stores patterns, execution times, and results in persistent memory.
Use for building up knowledge over multiple executions.`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL code to execute and learn from",
            },
            memory_name: {
              type: "string",
              description: "Name for the memory file (default: default)",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "nsl_introspect",
        description: `Get NSL self-awareness report.
Returns: version, identity, purpose, available operators, memory model, execution context, and capabilities.
Useful for understanding what NSL can do.`,
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "nsl_capabilities",
        description: `List all NSL capabilities in machine-readable format.
Returns: consciousness operators, GPU operations, namespaces, and special features.
Use this to discover what NSL can do.`,
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "nsl_benchmark",
        description: `Run NSL performance benchmark.
Tests: basic arithmetic, variable access, function calls, list creation.
Returns ops/sec for each category.`,
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "nsl_ast",
        description: `Parse NSL code and return the Abstract Syntax Tree.
Useful for understanding code structure without executing it.`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL code to parse",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "nsl_transform",
        description: `Transform NSL code using various strategies:
- vectorize: Convert to GPU tensor operations
- optimize: Apply optimizations
- minify: Compress to single line
- prettify: Format with proper indentation`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL code to transform",
            },
            transform: {
              type: "string",
              enum: ["vectorize", "optimize", "minify", "prettify"],
              description: "Transformation type",
            },
          },
          required: ["code", "transform"],
        },
      },
      {
        name: "nsl_file",
        description: "Execute an NSL script file (.nsl)",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "Path to the .nsl file",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "nsl_read",
        description: "Read a file using NSL",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "File path to read",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "nsl_write",
        description: "Write content to a file using NSL",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "File path to write",
            },
            content: {
              type: "string",
              description: "Content to write",
            },
          },
          required: ["path", "content"],
        },
      },
      // ===== SESSION TOOLS =====
      {
        name: "nsl_session",
        description: `Execute NSL code in a persistent session.
State persists across calls - functions, variables, and definitions are remembered!

Example workflow:
  Call 1: fn matmul(a, b) { ... }  # Define function
  Call 2: fn attention(q,k,v) { matmul(q,k) ... }  # Uses matmul from Call 1!
  Call 3: let model = build_gpt()  # All previous defs available
  Call 4: generate(model, "hello")  # Full state preserved

Sessions are saved to disk and persist across MCP server restarts.
Default session is "default" - use session_id for multiple parallel sessions.`,
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "NSL code to execute in the session",
            },
            session_id: {
              type: "string",
              description: "Session ID (default: 'default'). Use different IDs for isolated sessions.",
            },
            gpu: {
              type: "boolean",
              description: "Enable GPU acceleration",
            },
            think: {
              type: "boolean",
              description: "Enable thinking trace",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "nsl_session_list",
        description: "List all active NSL sessions with their state",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "nsl_session_clear",
        description: "Clear a session's accumulated state (start fresh)",
        inputSchema: {
          type: "object",
          properties: {
            session_id: {
              type: "string",
              description: "Session ID to clear (default: 'default')",
            },
          },
        },
      },
      {
        name: "nsl_session_show",
        description: "Show the accumulated code in a session",
        inputSchema: {
          type: "object",
          properties: {
            session_id: {
              type: "string",
              description: "Session ID to inspect (default: 'default')",
            },
          },
        },
      },
      {
        name: "nsl_session_delete",
        description: "Delete a session completely",
        inputSchema: {
          type: "object",
          properties: {
            session_id: {
              type: "string",
              description: "Session ID to delete",
            },
          },
          required: ["session_id"],
        },
      },
      // ===== NEW ADVANCED TOOLS =====
      {
        name: "nsl_session_restart",
        description: `Restart a session (like bash tool's restart).
Clears all accumulated code but preserves session metadata.
Useful when you want a clean slate without losing session identity.`,
        inputSchema: {
          type: "object",
          properties: {
            session_id: {
              type: "string",
              description: "Session ID to restart (default: 'default')",
            },
          },
        },
      },
      {
        name: "nsl_history",
        description: `View command history for the current conversation.
Returns recent NSL commands with timestamps, success status, and execution times.
Useful for debugging and reviewing what was executed.`,
        inputSchema: {
          type: "object",
          properties: {
            count: {
              type: "number",
              description: "Number of history entries to return (default: 50)",
            },
            session_id: {
              type: "string",
              description: "Filter by session ID (optional)",
            },
          },
        },
      },
      {
        name: "nsl_pipe",
        description: `Execute multiple NSL expressions in a pipeline.
Each expression receives the result of the previous one as 'it'.
Similar to Unix pipes but for NSL code.

Example: ["file.read('data.txt')", "split(it, '\\n')", "len(it)"]
This reads a file, splits by lines, and counts them.`,
        inputSchema: {
          type: "object",
          properties: {
            expressions: {
              type: "array",
              items: { type: "string" },
              description: "Array of NSL expressions to pipe together",
            },
            session_id: {
              type: "string",
              description: "Session ID for persistent state",
            },
          },
          required: ["expressions"],
        },
      },
      {
        name: "nsl_parallel",
        description: `Execute multiple NSL code blocks in parallel.
All blocks run concurrently and results are collected.
Useful for independent computations that can run simultaneously.

Returns array of results in the same order as inputs.`,
        inputSchema: {
          type: "object",
          properties: {
            blocks: {
              type: "array",
              items: { type: "string" },
              description: "Array of NSL code blocks to run in parallel",
            },
            timeout: {
              type: "number",
              description: "Timeout in ms for each block (default: 30000)",
            },
          },
          required: ["blocks"],
        },
      },
      {
        name: "nsl_cd",
        description: `Change the working directory for a session.
Affects all subsequent file operations in that session.
Returns the new working directory.`,
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "New working directory path",
            },
            session_id: {
              type: "string",
              description: "Session ID (default: 'default')",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "nsl_env",
        description: `Set or get environment variables for a session.
Session environment persists across calls.`,
        inputSchema: {
          type: "object",
          properties: {
            action: {
              type: "string",
              enum: ["get", "set", "list"],
              description: "Action to perform",
            },
            key: {
              type: "string",
              description: "Environment variable name (for get/set)",
            },
            value: {
              type: "string",
              description: "Value to set (for set action)",
            },
            session_id: {
              type: "string",
              description: "Session ID (default: 'default')",
            },
          },
          required: ["action"],
        },
      },
      {
        name: "nsl_watch",
        description: `Poll an NSL expression at regular intervals and report changes.
Executes the expression repeatedly and returns when a change is detected or max iterations reached.
Useful for monitoring file changes, waiting for conditions, or observing variable updates.

Returns the first changed value or the final value after max iterations.`,
        inputSchema: {
          type: "object",
          properties: {
            expression: {
              type: "string",
              description: "NSL expression to evaluate repeatedly",
            },
            interval: {
              type: "number",
              description: "Polling interval in milliseconds (default: 1000)",
            },
            maxIterations: {
              type: "number",
              description: "Maximum number of iterations (default: 10)",
            },
            stopOnChange: {
              type: "boolean",
              description: "Stop when value changes from initial (default: true)",
            },
            session_id: {
              type: "string",
              description: "Session ID for context",
            },
          },
          required: ["expression"],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "nsl": {
        const options = {
          think: args.think,
          reflect: args.reflect,
          explain: args.explain,
          gpu: args.gpu,
          optimize: args.optimize,
          context: args.context,
          timeout: args.timeout,
        };

        // Use session if session_id provided
        const result = args.session_id
          ? await executeNSLWithSession(args.code, args.session_id, options)
          : await executeNSL(args.code, options);

        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_think": {
        const result = await executeNSL(args.code, {
          think: true,
          trace: true,
        });
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_gpu": {
        const result = await executeNSL(args.code, {
          gpu: true,
          vectorize: args.vectorize,
        });
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_consciousness": {
        const result = await executeNSL(args.code, {
          reflect: args.reflect,
          explain: true,
        });
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_learn": {
        const memoryName = args.memory_name || "default";
        const memoryFile = join(MEMORY_DIR, `${memoryName}.json`);
        const result = await executeNSL(args.code, {
          learn: true,
          memory: memoryFile,
        });
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_introspect": {
        const result = await getIntrospection();
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_capabilities": {
        const result = await getCapabilities();
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_benchmark": {
        const result = await runBenchmark();
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_ast": {
        return new Promise((resolve, reject) => {
          const nsl = spawn(NSL_PATH, ["--ast", args.code, "--json"], {
            shell: false,
            windowsHide: true,
          });
          let stdout = "";
          nsl.stdout.on("data", (data) => (stdout += data.toString()));
          nsl.on("close", () => {
            try {
              resolve({
                content: [{ type: "text", text: stdout }],
              });
            } catch {
              resolve({
                content: [{ type: "text", text: stdout }],
              });
            }
          });
          nsl.on("error", reject);
        });
      }

      case "nsl_transform": {
        return new Promise((resolve, reject) => {
          const nsl = spawn(
            NSL_PATH,
            ["--transform", args.transform, args.code, "--json"],
            { shell: false, windowsHide: true }
          );
          let stdout = "";
          nsl.stdout.on("data", (data) => (stdout += data.toString()));
          nsl.on("close", () => {
            resolve({
              content: [{ type: "text", text: stdout }],
            });
          });
          nsl.on("error", reject);
        });
      }

      case "nsl_file": {
        return new Promise((resolve, reject) => {
          const nsl = spawn(NSL_PATH, [args.path], { shell: false, windowsHide: true });
          let stdout = "";
          let stderr = "";
          nsl.stdout.on("data", (data) => (stdout += data.toString()));
          nsl.stderr.on("data", (data) => (stderr += data.toString()));
          nsl.on("close", (code) => {
            resolve({
              content: [
                {
                  type: "text",
                  text: JSON.stringify(
                    {
                      success: code === 0,
                      output: stdout.trim(),
                      error: stderr.trim() || undefined,
                    },
                    null,
                    2
                  ),
                },
              ],
            });
          });
          nsl.on("error", reject);
        });
      }

      case "nsl_read": {
        const result = await executeNSL(
          `read_file("${args.path.replace(/\\/g, "/")}")`
        );
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      }

      case "nsl_write": {
        const escapedContent = args.content
          .replace(/\\/g, "\\\\")
          .replace(/"/g, '\\"')
          .replace(/\n/g, "\\n");
        const result = await executeNSL(
          `write_file("${args.path.replace(/\\/g, "/")}", "${escapedContent}")`
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                { ...result, message: "File written successfully" },
                null,
                2
              ),
            },
          ],
        };
      }

      // ===== SESSION HANDLERS =====
      case "nsl_session": {
        const sessionId = args.session_id || DEFAULT_SESSION;
        const result = await executeNSLWithSession(args.code, sessionId, {
          gpu: args.gpu,
          think: args.think,
        });
        return {
          content: [
            { type: "text", text: JSON.stringify(result, null, 2) },
          ],
        };
      }

      case "nsl_session_list": {
        const sessionList = await listSessions();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                sessions: sessionList,
                count: sessionList.length,
                hint: "Use nsl_session with session_id to work in a specific session"
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_session_clear": {
        const sessionId = args.session_id || DEFAULT_SESSION;
        const session = await loadSession(sessionId);
        const oldCount = session.code.length;
        session.code = [];
        session.callCount = 0;
        await saveSession(session);
        sessions.set(sessionId, session);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                message: `Session '${sessionId}' cleared`,
                clearedBlocks: oldCount
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_session_show": {
        const sessionId = args.session_id || DEFAULT_SESSION;
        const session = await loadSession(sessionId);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                session: {
                  id: session.id,
                  created: session.created,
                  lastUsed: session.lastUsed,
                  callCount: session.callCount,
                  codeBlocks: session.code.length,
                },
                code: session.getFullContext() || "(empty session)",
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_session_delete": {
        const deleted = await deleteSession(args.session_id);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: deleted,
                message: deleted
                  ? `Session '${args.session_id}' deleted`
                  : `Session '${args.session_id}' not found`,
              }, null, 2),
            },
          ],
        };
      }

      // ===== NEW ADVANCED TOOL HANDLERS =====

      case "nsl_session_restart": {
        const sessionId = args.session_id || DEFAULT_SESSION;
        const session = await loadSession(sessionId);
        const clearedBlocks = session.restart();
        await saveSession(session);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                message: `Session '${sessionId}' restarted`,
                clearedBlocks,
                restartCount: session.restartCount,
                hint: "Session state cleared. All accumulated code removed."
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_history": {
        const count = args.count || 50;
        let history = getHistory(count);

        // Filter by session if specified
        if (args.session_id) {
          history = history.filter(h => h.sessionId === args.session_id);
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                count: history.length,
                history: history.map(h => ({
                  timestamp: h.timestamp,
                  code: h.code.length > 100 ? h.code.substring(0, 100) + "..." : h.code,
                  success: h.success,
                  executionTime: h.executionTime,
                  sessionId: h.sessionId,
                })),
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_pipe": {
        const expressions = args.expressions;
        const sessionId = args.session_id || DEFAULT_SESSION;

        if (!expressions || !Array.isArray(expressions) || expressions.length === 0) {
          return {
            content: [{ type: "text", text: JSON.stringify({ success: false, error: "expressions array is required" }, null, 2) }],
            isError: true,
          };
        }

        let currentValue = null;
        const steps = [];

        for (let i = 0; i < expressions.length; i++) {
          const expr = expressions[i];
          // Inject 'it' variable with previous result
          const code = currentValue !== null
            ? `let it = ${JSON.stringify(currentValue)}\n${expr}`
            : expr;

          const result = await executeNSLWithSession(code, sessionId, {});

          steps.push({
            step: i + 1,
            expression: expr,
            success: result.success !== false,
            result: result.Result,
          });

          if (result.success === false) {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    success: false,
                    failedAtStep: i + 1,
                    error: result.Error || result.error,
                    steps,
                  }, null, 2),
                },
              ],
              isError: true,
            };
          }

          currentValue = result.Result;
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                finalResult: currentValue,
                stepsExecuted: expressions.length,
                steps,
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_parallel": {
        const blocks = args.blocks;
        const timeout = args.timeout || 30000;

        if (!blocks || !Array.isArray(blocks) || blocks.length === 0) {
          return {
            content: [{ type: "text", text: JSON.stringify({ success: false, error: "blocks array is required" }, null, 2) }],
            isError: true,
          };
        }

        // Execute all blocks in parallel with timeout
        const promises = blocks.map(async (code, index) => {
          const startTime = Date.now();
          try {
            const result = await Promise.race([
              executeNSL(code),
              new Promise((_, reject) =>
                setTimeout(() => reject(new Error("timeout")), timeout)
              ),
            ]);
            return {
              index,
              success: result.success !== false,
              result: result.Result,
              error: result.Error,
              executionTime: Date.now() - startTime,
            };
          } catch (e) {
            return {
              index,
              success: false,
              error: e.message,
              executionTime: Date.now() - startTime,
            };
          }
        });

        const results = await Promise.all(promises);
        const allSuccess = results.every(r => r.success);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: allSuccess,
                totalBlocks: blocks.length,
                successCount: results.filter(r => r.success).length,
                results: results.sort((a, b) => a.index - b.index),
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_cd": {
        const sessionId = args.session_id || DEFAULT_SESSION;
        const session = await loadSession(sessionId);

        // Resolve the path
        const newPath = resolve(session.workingDir, args.path);

        // Check if directory exists using NSL
        const checkResult = await executeNSL(`dir.exists("${newPath.replace(/\\/g, "/")}")`);

        if (checkResult.Result !== true && checkResult.Result !== "true") {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  success: false,
                  error: `Directory does not exist: ${newPath}`,
                }, null, 2),
              },
            ],
            isError: true,
          };
        }

        session.setWorkingDir(newPath);
        await saveSession(session);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                previousDir: session.workingDir,
                currentDir: newPath,
                sessionId,
              }, null, 2),
            },
          ],
        };
      }

      case "nsl_env": {
        const sessionId = args.session_id || DEFAULT_SESSION;
        const session = await loadSession(sessionId);
        const action = args.action;

        switch (action) {
          case "get":
            if (!args.key) {
              return {
                content: [{ type: "text", text: JSON.stringify({ success: false, error: "key is required for get action" }, null, 2) }],
                isError: true,
              };
            }
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    success: true,
                    key: args.key,
                    value: session.env[args.key] ?? process.env[args.key] ?? null,
                    source: session.env[args.key] !== undefined ? "session" : (process.env[args.key] !== undefined ? "system" : "not_found"),
                  }, null, 2),
                },
              ],
            };

          case "set":
            if (!args.key) {
              return {
                content: [{ type: "text", text: JSON.stringify({ success: false, error: "key is required for set action" }, null, 2) }],
                isError: true,
              };
            }
            session.setEnv(args.key, args.value);
            await saveSession(session);
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    success: true,
                    message: `Environment variable '${args.key}' set`,
                    key: args.key,
                    value: args.value,
                  }, null, 2),
                },
              ],
            };

          case "list":
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    success: true,
                    sessionEnv: session.env,
                    sessionEnvCount: Object.keys(session.env).length,
                    hint: "Use action='get' with a key to also check system environment",
                  }, null, 2),
                },
              ],
            };

          default:
            return {
              content: [{ type: "text", text: JSON.stringify({ success: false, error: `Unknown action: ${action}` }, null, 2) }],
              isError: true,
            };
        }
      }

      case "nsl_watch": {
        const expression = args.expression;
        const interval = args.interval || 1000;
        const maxIterations = args.maxIterations || 10;
        const stopOnChange = args.stopOnChange !== false;
        const sessionId = args.session_id || DEFAULT_SESSION;

        if (!expression) {
          return {
            content: [{ type: "text", text: JSON.stringify({ success: false, error: "expression is required" }, null, 2) }],
            isError: true,
          };
        }

        const iterations = [];
        let initialValue = null;
        let changed = false;

        for (let i = 0; i < maxIterations; i++) {
          const result = await executeNSLWithSession(expression, sessionId, {});
          const value = result.Result;

          iterations.push({
            iteration: i + 1,
            timestamp: new Date().toISOString(),
            value,
            success: result.success !== false,
          });

          if (i === 0) {
            initialValue = value;
          } else if (stopOnChange && JSON.stringify(value) !== JSON.stringify(initialValue)) {
            changed = true;
            break;
          }

          // Wait before next iteration (unless this was the last one)
          if (i < maxIterations - 1) {
            await new Promise(resolve => setTimeout(resolve, interval));
          }
        }

        const lastIteration = iterations[iterations.length - 1];

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                changed,
                initialValue,
                finalValue: lastIteration.value,
                iterationsCompleted: iterations.length,
                maxIterations,
                interval,
                stoppedEarly: changed,
                iterations: iterations.slice(-5), // Only show last 5 iterations
              }, null, 2),
            },
          ],
        };
      }

      default:
        return {
          content: [{ type: "text", text: `Unknown tool: ${name}` }],
          isError: true,
        };
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              success: false,
              error: error.message,
            },
            null,
            2
          ),
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("NSL MCP Server v1.0.0 running on stdio");
  console.error(`NSL Path: ${NSL_PATH}`);
  console.error(`Memory Dir: ${MEMORY_DIR}`);
}

main().catch(console.error);
