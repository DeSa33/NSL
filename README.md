# NSL - Neural Symbolic Language

**Write code the way AI thinks.**

NSL is a programming language designed for AI-native computing. It replaces bash/shell scripting with clean, predictable syntax.

---

## Quick Start

### Install (Windows)

1. [Download the latest release](https://github.com/DeSa33/NSL/releases)
2. Extract to `C:\NSL`
3. Add to PATH:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\NSL", "User")
   ```
4. Restart your terminal

### Install (Linux / macOS)

```bash
curl -L https://github.com/DeSa33/NSL/releases/latest/download/nsl-linux-x64.tar.gz | tar xz
sudo mv nsl /usr/local/bin/
```

### Verify

```bash
nsl --version
```

---

## Basic Usage

```bash
nsl                    # Start interactive mode
nsl script.nsl         # Run a script
nsl --eval "2 + 2"     # Run code directly
```

---

## What Can NSL Do?

### Run Commands
```nsl
sys.exec("git status")
sys.shell("npm install")
```

### Work with Files
```nsl
file.read("config.json")
file.write("output.txt", "Hello")
file.exists("myfile.txt")
```

### List Directories
```nsl
dir.list()
dir.list("src")
dir.files(".", "*.js")
```

### HTTP Requests
```nsl
http.get("https://api.example.com/data")
http.post(url, body, "application/json")
```

### Parse Data
```nsl
json.parse('{"name": "NSL"}')
json.stringify(data)
```

### String Operations
```nsl
string.split("a,b,c", ",")
string.upper("hello")
string.replace(text, "old", "new")
```

---

## Interactive Mode

Type `nsl` to enter interactive mode. Shortcuts available:

| Command | Action |
|---------|--------|
| `cd folder` | Change directory |
| `ls` | List files |
| `pwd` | Show current path |
| `cat file.txt` | View file contents |

Use ↑/↓ arrows for command history. Press Tab for auto-complete.

---

## Pipe Operator

Chain operations left to right:

```nsl
"hello" |> string.upper |> string.reverse
# Result: "OLLEH"

[3, 1, 2] |> list.sort |> list.reverse
# Result: [3, 2, 1]
```

---

## Quick Reference

### System
| Function | Description |
|----------|-------------|
| `sys.exec(cmd)` | Run command, get full result |
| `sys.shell(cmd)` | Run command, get stdout only |
| `sys.env("PATH")` | Get environment variable |
| `sys.cwd()` | Current directory |
| `sys.platform()` | OS information |

### Files
| Function | Description |
|----------|-------------|
| `file.read(path)` | Read file contents |
| `file.write(path, text)` | Write to file |
| `file.append(path, text)` | Append to file |
| `file.exists(path)` | Check if file exists |
| `file.delete(path)` | Delete file |
| `file.copy(src, dst)` | Copy file |

### Directories
| Function | Description |
|----------|-------------|
| `dir.list(path)` | List all entries |
| `dir.files(path, pattern)` | List files only |
| `dir.create(path)` | Create directory |
| `dir.exists(path)` | Check if exists |

### Data
| Function | Description |
|----------|-------------|
| `json.parse(str)` | Parse JSON string |
| `json.stringify(obj)` | Convert to JSON |
| `list.sort(arr)` | Sort array |
| `list.length(arr)` | Array length |
| `string.split(s, delim)` | Split string |

---

## Examples

### Read a config file
```nsl
let config = file.read("config.json") |> json.parse
print(config.name)
```

### Find all JavaScript files
```nsl
let files = file.find("./src", "*.js")
print(files)
```

### Check git status
```nsl
let result = sys.exec("git status")
if (result.success) {
    print(result.stdout)
}
```

### Download a file
```nsl
http.download("https://example.com/file.zip", "file.zip")
```

---

## Get Help

- **Issues**: [GitHub Issues](https://github.com/DeSa33/NSL/issues)
- **Email**: desaviorwhite@protonmail.com

---

## License

Copyright © 2025 DeSavior Emmanuel White. All Rights Reserved.

NSL is proprietary software. See [LICENSE](LICENSE) for terms.
