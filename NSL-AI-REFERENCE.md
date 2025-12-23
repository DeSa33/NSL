# NSL (Neural Symbolic Language) - AI Reference Guide

## Overview
NSL is an AI-native scripting language designed as a bash replacement. Use NSL for file operations, shell commands, HTTP requests, and data transformation.

## How to Execute NSL
```
nsl --eval "<code>" --json --quiet
```

## Namespaces & Functions

### sys - System Operations (BASH REPLACEMENT)
| Function | Description | Example |
|----------|-------------|---------|
| `sys.exec(cmd, [cwd], [timeout])` | Run shell command, returns {stdout, stderr, code, success} | `sys.exec("git status")` |
| `sys.shell(cmd, [cwd])` | Run command, returns stdout only | `sys.shell("ls -la")` |
| `sys.which(name)` | Find executable path | `sys.which("python")` |
| `sys.env(name)` | Get environment variable | `sys.env("PATH")` |
| `sys.setenv(name, value)` | Set environment variable | `sys.setenv("MY_VAR", "value")` |
| `sys.cwd()` | Get current directory | `sys.cwd()` |
| `sys.cd(path)` | Change directory | `sys.cd("/home")` |
| `sys.home()` | Get home directory | `sys.home()` |
| `sys.temp()` | Get temp directory | `sys.temp()` |
| `sys.pid()` | Get current process ID | `sys.pid()` |
| `sys.kill(pid)` | Kill process by ID | `sys.kill(1234)` |
| `sys.sleep(ms)` | Sleep milliseconds | `sys.sleep(1000)` |
| `sys.time()` | Get ISO timestamp | `sys.time()` |
| `sys.timestamp()` | Get Unix timestamp ms | `sys.timestamp()` |
| `sys.platform()` | Get OS info | `sys.platform()` |
| `sys.hostname()` | Get machine name | `sys.hostname()` |
| `sys.username()` | Get current user | `sys.username()` |
| `sys.args()` | Get command line args | `sys.args()` |
| `sys.exit(code)` | Exit with code | `sys.exit(0)` |

### file - File Operations
| Function | Description | Example |
|----------|-------------|---------|
| `file.read(path)` | Read file contents | `file.read("config.json")` |
| `file.write(path, content)` | Write to file | `file.write("out.txt", "hello")` |
| `file.append(path, content)` | Append to file | `file.append("log.txt", "entry")` |
| `file.exists(path)` | Check if file exists | `file.exists("file.txt")` |
| `file.delete(path)` | Delete file | `file.delete("temp.txt")` |
| `file.copy(src, dst)` | Copy file | `file.copy("a.txt", "b.txt")` |
| `file.move(src, dst)` | Move/rename file | `file.move("old.txt", "new.txt")` |
| `file.size(path)` | Get file size bytes | `file.size("data.bin")` |
| `file.list(dir, [pattern])` | List files in directory | `file.list(".", "*.txt")` |
| `file.readLines(path)` | Read as line array | `file.readLines("data.csv")` |
| `file.search(path, text)` | Search file for text | `file.search("log.txt", "error")` |
| `file.find(dir, pattern)` | Find files recursively | `file.find(".", "*.js")` |
| `file.grep(dir, text, [pattern])` | Grep directory | `file.grep("src", "TODO")` |
| `file.head(path, [n])` | First n lines | `file.head("log.txt", 10)` |
| `file.tail(path, [n])` | Last n lines | `file.tail("log.txt", 10)` |

### dir - Directory Operations
| Function | Description | Example |
|----------|-------------|---------|
| `dir.exists(path)` | Check if dir exists | `dir.exists("./src")` |
| `dir.create(path)` | Create directory | `dir.create("./new")` |
| `dir.delete(path, [recursive])` | Delete directory | `dir.delete("./tmp", true)` |
| `dir.list(path)` | List entries | `dir.list(".")` |
| `dir.files(path, [pattern])` | List files only | `dir.files(".", "*.txt")` |
| `dir.dirs(path)` | List subdirs only | `dir.dirs(".")` |
| `dir.copy(src, dst)` | Copy directory | `dir.copy("./a", "./b")` |
| `dir.move(src, dst)` | Move directory | `dir.move("./old", "./new")` |
| `dir.size(path)` | Get total size | `dir.size("./data")` |
| `dir.tree(path, [depth])` | Directory tree | `dir.tree(".", 2)` |
| `dir.walk(path)` | All files recursive | `dir.walk("./src")` |

### path - Path Utilities
| Function | Description | Example |
|----------|-------------|---------|
| `path.join(a, b, ...)` | Join paths | `path.join("a", "b", "c.txt")` |
| `path.dirname(path)` | Get directory | `path.dirname("/a/b/c.txt")` |
| `path.basename(path)` | Get filename | `path.basename("/a/b/c.txt")` |
| `path.ext(path)` | Get extension | `path.ext("file.txt")` |
| `path.stem(path)` | Name without ext | `path.stem("file.txt")` |
| `path.absolute(path)` | Get absolute path | `path.absolute("./file")` |
| `path.exists(path)` | Check exists | `path.exists("./file")` |
| `path.isFile(path)` | Is it a file | `path.isFile("./file")` |
| `path.isDir(path)` | Is it a directory | `path.isDir("./dir")` |

### string - String Operations  
| Function | Description | Example |
|----------|-------------|---------|
| `string.length(s)` | String length | `string.length("hello")` |
| `string.upper(s)` | Uppercase | `string.upper("hello")` |
| `string.lower(s)` | Lowercase | `string.lower("HELLO")` |
| `string.trim(s)` | Trim whitespace | `string.trim("  hi  ")` |
| `string.split(s, delim)` | Split string | `string.split("a,b,c", ",")` |
| `string.contains(s, sub)` | Contains check | `string.contains("hello", "ell")` |
| `string.replace(s, old, new)` | Replace text | `string.replace("hi", "i", "ello")` |
| `string.substring(s, start, [len])` | Get substring | `string.substring("hello", 1, 3)` |
| `string.startsWith(s, prefix)` | Starts with | `string.startsWith("hello", "he")` |
| `string.endsWith(s, suffix)` | Ends with | `string.endsWith("hello", "lo")` |
| `string.indexOf(s, sub)` | Find index | `string.indexOf("hello", "l")` |

### list - List/Array Operations
| Function | Description | Example |
|----------|-------------|---------|
| `list.length(arr)` | Array length | `list.length([1,2,3])` |
| `list.sum(arr)` | Sum numbers | `list.sum([1,2,3])` |
| `list.avg(arr)` | Average | `list.avg([1,2,3])` |
| `list.min(arr)` | Minimum | `list.min([1,2,3])` |
| `list.max(arr)` | Maximum | `list.max([1,2,3])` |
| `list.sort(arr)` | Sort array | `list.sort([3,1,2])` |
| `list.reverse(arr)` | Reverse array | `list.reverse([1,2,3])` |
| `list.unique(arr)` | Remove duplicates | `list.unique([1,1,2])` |
| `list.flatten(arr)` | Flatten nested | `list.flatten([[1],[2]])` |
| `list.contains(arr, val)` | Contains check | `list.contains([1,2], 2)` |
| `list.slice(arr, start, [count])` | Get slice | `list.slice([1,2,3], 1, 2)` |
| `list.join(arr, [sep])` | Join to string | `list.join(["a","b"], "-")` |
| `list.range(start, end, [step])` | Generate range | `list.range(0, 10, 2)` |
| `list.append(arr, val)` | Add element | `list.append([1,2], 3)` |
| `list.concat(arr1, arr2)` | Concatenate | `list.concat([1],[2])` |

### regex - Regular Expressions
| Function | Description | Example |
|----------|-------------|---------|
| `regex.match(text, pattern)` | First match | `regex.match("a1b2", "[0-9]+")` |
| `regex.matches(text, pattern)` | All matches | `regex.matches("a1b2", "[0-9]+")` |
| `regex.test(text, pattern)` | Test if matches | `regex.test("hello", "^h")` |
| `regex.replace(text, pat, repl)` | Replace matches | `regex.replace("a1b", "[0-9]", "X")` |
| `regex.split(text, pattern)` | Split by pattern | `regex.split("a1b2c", "[0-9]")` |
| `regex.groups(text, pattern)` | Capture groups | `regex.groups("ab", "(a)(b)")` |

### http - HTTP Client
| Function | Description | Example |
|----------|-------------|---------|
| `http.get(url, [timeout])` | GET request | `http.get("https://api.example.com")` |
| `http.post(url, body, [contentType])` | POST request | `http.post(url, "{}", "application/json")` |
| `http.download(url, destPath)` | Download file | `http.download(url, "file.zip")` |
| `http.head(url)` | HEAD request | `http.head(url)` |

### json - JSON Operations
| Function | Description | Example |
|----------|-------------|---------|
| `json.parse(str)` | Parse JSON string | `json.parse("{\"a\":1}")` |
| `json.stringify(obj, [indent])` | Convert to JSON | `json.stringify(data, true)` |
| `json.pretty(str)` | Pretty print JSON | `json.pretty(jsonStr)` |
| `json.valid(str)` | Validate JSON | `json.valid("{}")` |

### crypto - Cryptography
| Function | Description | Example |
|----------|-------------|---------|
| `crypto.hash(data, [algo])` | Hash string (md5/sha1/sha256/sha512) | `crypto.hash("hello", "sha256")` |
| `crypto.uuid()` | Generate UUID | `crypto.uuid()` |
| `crypto.random([length])` | Random hex string | `crypto.random(32)` |
| `crypto.base64encode(data)` | Base64 encode | `crypto.base64encode("hello")` |
| `crypto.base64decode(data)` | Base64 decode | `crypto.base64decode("aGVsbG8=")` |

### math - Math Operations
| Function | Description | Example |
|----------|-------------|---------|
| `math.sqrt(n)` | Square root | `math.sqrt(16)` |
| `math.pow(base, exp)` | Power | `math.pow(2, 8)` |
| `math.abs(n)` | Absolute value | `math.abs(-5)` |
| `math.floor(n)` | Floor | `math.floor(3.7)` |
| `math.ceil(n)` | Ceiling | `math.ceil(3.2)` |
| `math.round(n, [decimals])` | Round | `math.round(3.456, 2)` |
| `math.sin/cos/tan(n)` | Trig functions | `math.sin(0)` |
| `math.log(n, [base])` | Logarithm | `math.log(100, 10)` |
| `math.exp(n)` | e^n | `math.exp(1)` |
| `math.min(...)` | Minimum | `math.min(1, 2, 3)` |
| `math.max(...)` | Maximum | `math.max(1, 2, 3)` |
| `math.random([min], [max])` | Random number | `math.random(1, 100)` |

### text - Text Processing
| Function | Description | Example |
|----------|-------------|---------|
| `text.lines(s)` | Split to lines | `text.lines(content)` |
| `text.words(s)` | Split to words | `text.words("hello world")` |
| `text.count(s, sub)` | Count occurrences | `text.count("aaa", "a")` |
| `text.wrap(s, width)` | Word wrap | `text.wrap(text, 80)` |
| `text.truncate(s, max, [suffix])` | Truncate | `text.truncate(s, 100, "...")` |
| `text.indent(s, [spaces])` | Indent text | `text.indent(code, 4)` |
| `text.dedent(s)` | Remove indent | `text.dedent(code)` |
| `text.slug(s)` | Slugify | `text.slug("Hello World")` |
| `text.title(s)` | Title case | `text.title("hello world")` |

## Pipe Operator
Chain operations left-to-right:
```nsl
[1,2,3,4,5] |> list.sum |> math.sqrt
file.read("data.txt") |> text.lines |> list.length
sys.exec("git log").stdout |> text.lines
```

## Consciousness Operators
| Operator | Name | Description |
|----------|------|-------------|
| `|>` | pipe | Chain transformations |
| `~>` | awareness | Introspective flow |
| `=>>` | gradient | Learning flow |
| `*>` | attention | Focus mechanism |
| `+>` | superposition | Quantum-like states |

## Common Patterns

### Run shell command and process output:
```nsl
result = sys.exec("git status")
if (result.success) { result.stdout |> text.lines }
```

### Read JSON config:
```nsl
config = file.read("config.json") |> json.parse
config.setting
```

### Find and process files:
```nsl
files = file.find("./src", "*.js")
list.length(files)
```

### HTTP API call:
```nsl
response = http.get("https://api.example.com/data")
if (response.ok) { json.parse(response.body) }
```

### File transformation:
```nsl
content = file.read("input.txt")
processed = string.upper(content)
file.write("output.txt", processed)
```
