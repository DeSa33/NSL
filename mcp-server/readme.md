# NSL MCP Server

**Model Context Protocol (MCP) integration for NSL**

The NSL MCP Server allows AI assistants to execute NSL code directly, enabling AI-native programming capabilities across different platforms.

---

## What is MCP?

MCP (Model Context Protocol) is a standard that allows AI assistants to interact with external tools and services. The NSL MCP Server exposes NSL's capabilities to AI systems, enabling:

- **Direct code execution** - AI can run NSL code and get results
- **Persistent sessions** - State persists across multiple calls
- **GPU acceleration** - Access tensor operations
- **Consciousness operators** - Use AI-native operators (`|>`, `~>`, `=>>`, `*>`, `+>`)
- **File operations** - Read, write, and manipulate files

---

## Quick Start

### 1. Install NSL MCP Server

```bash
# Clone NSL repository
git clone https://github.com/DeSa33/NSL.git

# Navigate to MCP server
cd NSL/mcp-server

# Install dependencies
npm install
```

### 2. Configure Your AI Tool

Choose your AI platform below for specific setup instructions.

---

## AI Platform Setup

| Platform | Config File | Status |
|----------|------------|--------|
| [Claude Code](#claude-code) | `claude_desktop_config.json` | ✅ Full Support |
| [Claude Desktop](#claude-desktop) | `claude_desktop_config.json` | ✅ Full Support |
| [Cursor](#cursor) | `.cursor/mcp.json` | ✅ Full Support |
| [Windsurf](#windsurf) | `.windsurf/mcp.json` | ✅ Full Support |
| [VS Code + Continue](#vs-code--continue) | `config.json` | ✅ Full Support |
| [ChatGPT](#chatgpt) | Custom GPT Actions | ⚠️ Limited |
| [GitHub Copilot](#github-copilot) | Extension API | 🔜 Coming Soon |

---

## Claude Code

Claude Code has native MCP support.

### Configuration

**Location:** `~/.claude/claude_desktop_config.json` (or settings in Claude Code)

```json
{
  "mcpServers": {
    "nsl-mcp": {
      "command": "node",
      "args": ["C:/NSL/mcp-server/index.js"],
      "env": {
        "NSL_PATH": "C:/NSL/nsl.exe"
      }
    }
  }
}
```

### Usage in Claude Code

Once configured, you can ask Claude to:

```
Run this NSL code:
let x = 10
let y = 20
print(x + y)
```

Or use NSL tools directly:
- `nsl_session` - Execute with persistent state
- `nsl_gpu` - GPU-accelerated operations
- `nsl_consciousness` - AI-native operators
- `nsl_pipe` - Chain operations

### Example Session

```
User: Use NSL to calculate fibonacci(10)

Claude: [Uses nsl_session tool]
fn fib(n) {
    if (n <= 1) { return n }
    return fib(n - 1) + fib(n - 2)
}
print(fib(10))

Result: 55
```

---

## Claude Desktop

Same configuration as Claude Code.

### Configuration

**Location:** 
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "nsl-mcp": {
      "command": "node",
      "args": ["C:/NSL/mcp-server/index.js"]
    }
  }
}
```

---

## Cursor

Cursor IDE supports MCP through its configuration.

### Configuration

**Location:** `.cursor/mcp.json` in your project root or global settings

```json
{
  "mcpServers": {
    "nsl": {
      "command": "node",
      "args": ["C:/NSL/mcp-server/index.js"],
      "env": {
        "NSL_PATH": "C:/NSL/nsl.exe"
      }
    }
  }
}
```

### Usage

In Cursor's AI chat, you can ask it to use NSL for computations:

```
@nsl Calculate the mean of [1, 2, 3, 4, 5]

let data = [1, 2, 3, 4, 5]
let total = 0
for x in data { total = total + x }
print(total / len(data))
```

---

## Windsurf

Windsurf (Codeium's AI IDE) supports MCP servers.

### Configuration

**Location:** `.windsurf/mcp.json` or global Windsurf settings

```json
{
  "mcpServers": {
    "nsl-mcp": {
      "command": "node",
      "args": ["C:/NSL/mcp-server/index.js"],
      "transport": "stdio"
    }
  }
}
```

### Cascade Integration

In Windsurf's Cascade AI:

```
Use NSL to process this data with consciousness operators:

[1, 2, 3, 4, 5] |> map(x => x * 2) |> filter(x => x > 5)
```

---

## VS Code + Continue

Continue extension for VS Code supports MCP.

### Configuration

**Location:** `~/.continue/config.json`

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "node",
          "args": ["C:/NSL/mcp-server/index.js"]
        }
      }
    ]
  }
}
```

---

## ChatGPT

ChatGPT doesn't natively support MCP, but you can create a Custom GPT with Actions.

### Option 1: API Wrapper

Deploy NSL MCP as an API endpoint:

```bash
# Start NSL API server
node C:/NSL/mcp-server/api-server.js --port 3000
```

### Option 2: Custom GPT Action

Create a Custom GPT with this OpenAPI spec:

```yaml
openapi: 3.0.0
info:
  title: NSL API
  version: 1.0.0
servers:
  - url: https://your-nsl-server.com
paths:
  /execute:
    post:
      summary: Execute NSL code
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                code:
                  type: string
      responses:
        '200':
          description: Execution result
```

### Limitations

- No persistent sessions (each call is independent)
- No direct GPU access
- Requires hosting the API server

---

## GitHub Copilot

GitHub Copilot MCP support is in development.

### Current Status

🔜 **Coming Soon** - GitHub is working on MCP integration for Copilot.

### Workaround

For now, you can use NSL with Copilot by:

1. Adding NSL code examples to your codebase
2. Copilot will learn from the patterns
3. Use comments to guide NSL generation:

```nsl
# Copilot: create a function that calculates factorial
fn factorial(n) {
    if (n <= 1) { return 1 }
    return n * factorial(n - 1)
}
```

---

## Available NSL Tools

When the MCP server is connected, these tools become available:

| Tool | Description |
|------|-------------|
| `nsl` | Execute NSL code |
| `nsl_session` | Execute with persistent state |
| `nsl_think` | Execute with reasoning trace |
| `nsl_gpu` | GPU-accelerated execution |
| `nsl_consciousness` | AI-native operators |
| `nsl_pipe` | Chain expressions |
| `nsl_parallel` | Parallel execution |
| `nsl_read` | Read files |
| `nsl_write` | Write files |
| `nsl_ast` | Parse code to AST |
| `nsl_transform` | Transform code |

---

## Troubleshooting

### Server won't start

```bash
# Check Node.js version (requires 18+)
node --version

# Check NSL path
nsl --version
```

### Connection failed

1. Verify the path in your config is correct
2. Check that `nsl.exe` exists at the specified path
3. Restart your AI tool after config changes

### Permission errors

```bash
# Windows: Run as Administrator
# Linux/macOS: Check file permissions
chmod +x /path/to/nsl
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/DeSa33/NSL/issues)
- **Discussions:** [GitHub Discussions](https://github.com/DeSa33/NSL/discussions)

---

**Copyright © 2025 DeSavior Emmanuel White. All Rights Reserved.**
