# NSL Feature Reference

**Complete Technical Reference for Neural Symbolic Language**

*Every feature verified through live execution*

---

## Table of Contents

1. [Language Overview](#language-overview)
2. [Data Types](#data-types)
3. [Variables](#variables)
4. [Operators](#operators)
5. [Control Flow](#control-flow)
6. [Functions](#functions)
7. [Namespaces](#namespaces)
   - [sys - System Operations](#sys---system-operations)
   - [file - File Operations](#file---file-operations)
   - [dir - Directory Operations](#dir---directory-operations)
   - [path - Path Utilities](#path---path-utilities)
   - [string - String Operations](#string---string-operations)
   - [list - Array Operations](#list---array-operations)
   - [math - Mathematics](#math---mathematics)
   - [json - JSON Handling](#json---json-handling)
   - [yaml - YAML Handling](#yaml---yaml-handling)
   - [regex - Regular Expressions](#regex---regular-expressions)
   - [crypto - Cryptography](#crypto---cryptography)
   - [http - HTTP Client](#http---http-client)
   - [net - Network Utilities](#net---network-utilities)
   - [env - Environment](#env---environment)
   - [date - Date/Time](#date---datetime)
   - [git - Git Integration](#git---git-integration)
   - [proc - Process Management](#proc---process-management)
   - [clip - Clipboard](#clip---clipboard)
   - [zip - Archive Operations](#zip---archive-operations)
   - [diff - Text Comparison](#diff---text-comparison)
   - [template - Template Engine](#template---template-engine)
8. [Consciousness Operators](#consciousness-operators)
9. [GPU Operations](#gpu-operations)
10. [Interactive Mode](#interactive-mode)
11. [Execution Modes](#execution-modes)
12. [Performance](#performance)

---

## Language Overview

NSL (Neural Symbolic Language) is an AI-native programming language designed for:

- **Shell Replacement** - Clean syntax instead of bash/PowerShell
- **AI Integration** - Structured output, predictable behavior
- **GPU Acceleration** - Native CUDA tensor operations
- **Consciousness Operators** - AI-native data flow patterns

### Runtime

- **.NET 8.0** - High-performance managed runtime
- **Cross-platform** - Windows, Linux, macOS
- **Self-contained** - No dependencies required

---

## Data Types

| Type | Example | Description |
|------|---------|-------------|
| `number` | `42`, `3.14`, `-7` | Integer or floating-point |
| `string` | `"hello"` | Text enclosed in quotes |
| `boolean` | `true`, `false` | Logical values |
| `array` | `[1, 2, 3]` | Ordered collection |
| `dict` | `{name: "NSL"}` | Key-value pairs |
| `null` | `null` | Absence of value |

### Type Checking

```nsl
type(42)           # "number"
type("hello")      # "string"
type(true)         # "boolean"
type([1, 2, 3])    # "array"
type({a: 1})       # "dict"
```

---

## Variables

### Immutable (let)

```nsl
let name = "NSL"
let count = 42
# name = "other"  # Error! Cannot reassign
```

### Mutable (mut)

```nsl
mut counter = 0
counter = counter + 1
counter = 10  # OK
```

### Constants (const)

```nsl
const PI = 3.14159
const APP_NAME = "MyApp"
# PI = 3.0  # Error! Constants cannot change
```

---

## Operators

### Arithmetic

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `+` | Addition | `3 + 2` | `5` |
| `-` | Subtraction | `3 - 2` | `1` |
| `*` | Multiplication | `3 * 2` | `6` |
| `/` | Division | `7 / 2` | `3.5` |
| `%` | Modulo | `7 % 2` | `1` |

### Comparison

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `==` | Equal | `3 == 3` | `true` |
| `!=` | Not equal | `3 != 2` | `true` |
| `<` | Less than | `2 < 3` | `true` |
| `<=` | Less or equal | `3 <= 3` | `true` |
| `>` | Greater than | `3 > 2` | `true` |
| `>=` | Greater or equal | `3 >= 3` | `true` |

### Logical

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `and` | Logical AND | `true and false` | `false` |
| `or` | Logical OR | `true or false` | `true` |
| `not` | Logical NOT | `not true` | `false` |

---

## Control Flow

### If/Else

```nsl
let score = 85

if score >= 90 {
    print("Grade: A")
} else if score >= 80 {
    print("Grade: B")
} else if score >= 70 {
    print("Grade: C")
} else {
    print("Grade: F")
}
```

### For Loop

```nsl
# Using range()
for i in range(0, 5) {
    print(i)  # 0, 1, 2, 3, 4
}

# With step
for i in range(0, 10, 2) {
    print(i)  # 0, 2, 4, 6, 8
}

# Iterating arrays
let fruits = ["apple", "banana", "cherry"]
for fruit in fruits {
    print(fruit)
}
```

### While Loop

```nsl
mut count = 0
while count < 5 {
    print(count)
    count = count + 1
}
```

### Nested Loops

```nsl
for i in range(0, 3) {
    for j in range(0, 2) {
        print("(" + i + "," + j + ")")
    }
}
```

---

## Functions

### Basic Function

```nsl
fn greet(name) {
    return "Hello, " + name + "!"
}

print(greet("World"))  # Hello, World!
```

### Multiple Parameters

```nsl
fn add(a, b) {
    return a + b
}

print(add(3, 4))  # 7
```

### Recursive Functions

```nsl
fn factorial(n) {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

print(factorial(5))  # 120

fn fibonacci(n) {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

print(fibonacci(10))  # 55
```

### Closures

```nsl
fn makeAdder(x) {
    fn adder(y) {
        return x + y
    }
    return adder
}

let add5 = makeAdder(5)
print(add5(3))   # 8
print(add5(10))  # 15
```

### Higher-Order Functions

```nsl
fn applyTwice(f, x) {
    return f(f(x))
}

fn double(n) {
    return n * 2
}

print(applyTwice(double, 3))  # 12
```

---

## Namespaces

### sys - System Operations

Execute shell commands and access system information.

| Function | Description | Example |
|----------|-------------|---------|
| `sys.exec(cmd, [cwd])` | Execute command, returns `{stdout, stderr, code, success}` | `sys.exec("git status")` |
| `sys.shell(cmd)` | Execute command, returns stdout only | `sys.shell("dir /b")` |
| `sys.cd(path)` | Change directory | `sys.cd("C:/Projects")` |

```nsl
let result = sys.exec("whoami")
print(result.stdout)   # "username"
print(result.success)  # true

let output = sys.shell("dir /b")
print(output)
```

---

### file - File Operations

Read, write, and manipulate files.

| Function | Description | Returns |
|----------|-------------|---------|
| `file.read(path)` | Read file contents | `string` |
| `file.write(path, content)` | Write to file | `bool` |
| `file.append(path, content)` | Append to file | `bool` |
| `file.exists(path)` | Check if file exists | `bool` |
| `file.delete(path)` | Delete file | `bool` |
| `file.copy(src, dst)` | Copy file | `bool` |
| `file.move(src, dst)` | Move/rename file | `bool` |
| `file.size(path)` | Get file size in bytes | `number` |

```nsl
# Write and read
file.write("test.txt", "Hello NSL!")
let content = file.read("test.txt")
print(content)  # Hello NSL!

# Append
file.append("test.txt", "\nLine 2")

# Check existence
if file.exists("config.json") {
    let config = file.read("config.json") |> json.parse
}
```

---

### dir - Directory Operations

Work with directories and file listings.

| Function | Description | Returns |
|----------|-------------|---------|
| `dir.list(path)` | List all entries | `array` |
| `dir.files(path, [pattern])` | List files only | `array` |
| `dir.dirs(path)` | List subdirectories | `array` |
| `dir.create(path)` | Create directory | `bool` |
| `dir.delete(path, [recursive])` | Delete directory | `bool` |
| `dir.tree(path, [depth])` | Directory tree | `array` |
| `dir.walk(path)` | All files recursive | `array` |

```nsl
# Create structure
dir.create("project/src")
dir.create("project/tests")

# List contents
print(dir.list("project"))       # [src/, tests/]
print(dir.files(".", "*.txt"))   # [file1.txt, file2.txt]
print(dir.dirs("."))             # [folder1, folder2]

# Recursive listing
print(dir.walk("src"))
print(dir.tree(".", 2))
```

---

### path - Path Utilities

Path manipulation without filesystem access.

| Function | Description | Example Result |
|----------|-------------|----------------|
| `path.dirname(p)` | Directory portion | `C:\Users\test` |
| `path.basename(p)` | Filename portion | `file.txt` |
| `path.ext(p)` | File extension | `.txt` |
| `path.join(...)` | Join path segments | `C:\Users\test\file.txt` |
| `path.absolute(p)` | Get absolute path | `C:\full\path` |
| `path.exists(p)` | Check if path exists | `true` |

```nsl
let p = "C:/Users/test/documents/file.txt"

print(path.dirname(p))   # C:\Users\test\documents
print(path.basename(p))  # file.txt
print(path.ext(p))       # .txt
print(path.join("C:/", "Users", "file.txt"))  # C:/Users\file.txt
```

---

### string - String Operations

Text manipulation functions.

| Function | Description | Example |
|----------|-------------|---------|
| `string.length(s)` | String length | `5` |
| `string.upper(s)` | Uppercase | `"HELLO"` |
| `string.lower(s)` | Lowercase | `"hello"` |
| `string.trim(s)` | Remove whitespace | `"hello"` |
| `string.split(s, delim)` | Split to array | `["a", "b", "c"]` |
| `string.contains(s, sub)` | Contains check | `true` |
| `string.replace(s, old, new)` | Replace text | `"hello NSL"` |
| `string.repeat(s, n)` | Repeat string | `"NaNaNa"` |
| `string.substring(s, start, [len])` | Get substring | `"ell"` |
| `string.startsWith(s, prefix)` | Starts with check | `true` |
| `string.endsWith(s, suffix)` | Ends with check | `true` |
| `string.indexOf(s, sub)` | Find position | `2` |

```nsl
let text = "  Hello World  "

print(string.trim(text))           # "Hello World"
print(string.upper("hello"))       # "HELLO"
print(string.split("a,b,c", ","))  # ["a", "b", "c"]
print(string.replace("hello world", "world", "NSL"))  # "hello NSL"
print(string.repeat("Na", 4))      # "NaNaNaNa"

# Chained operations
let result = "  HELLO  " |> string.trim |> string.lower
print(result)  # "hello"
```

---

### list - Array Operations

Array manipulation and statistics.

| Function | Description | Example Result |
|----------|-------------|----------------|
| `list.length(arr)` | Array length | `5` |
| `list.sum(arr)` | Sum of numbers | `15` |
| `list.avg(arr)` | Average | `3` |
| `list.min(arr)` | Minimum value | `1` |
| `list.max(arr)` | Maximum value | `9` |
| `list.sort(arr)` | Sort ascending | `[1, 2, 3]` |
| `list.reverse(arr)` | Reverse order | `[3, 2, 1]` |
| `list.unique(arr)` | Remove duplicates | `[1, 2, 3]` |
| `list.flatten(arr)` | Flatten nested | `[1, 2, 3, 4]` |
| `list.append(arr, val)` | Add element | `[1, 2, 3, 4]` |
| `list.contains(arr, val)` | Contains check | `true` |
| `list.join(arr, sep)` | Join to string | `"a-b-c"` |

```nsl
let nums = [5, 2, 8, 1, 9, 3]

print(list.sum(nums))      # 28
print(list.avg(nums))      # 4.666...
print(list.min(nums))      # 1
print(list.max(nums))      # 9
print(list.sort(nums))     # [1, 2, 3, 5, 8, 9]
print(list.reverse(nums))  # [3, 9, 1, 8, 2, 5]

let dups = [1, 2, 2, 3, 3, 3]
print(list.unique(dups))   # [1, 2, 3]

let nested = [[1, 2], [3, 4]]
print(list.flatten(nested))  # [1, 2, 3, 4]

# Chained operations
let result = [3, 1, 4, 1, 5] |> list.unique |> list.sort |> list.reverse
print(result)  # [5, 4, 3, 1]
```

---

### math - Mathematics

Mathematical functions.

| Function | Description | Example |
|----------|-------------|---------|
| `math.sqrt(n)` | Square root | `4` |
| `math.pow(x, y)` | Power | `1024` |
| `math.abs(n)` | Absolute value | `42` |
| `math.floor(n)` | Round down | `3` |
| `math.ceil(n)` | Round up | `4` |
| `math.round(n)` | Round | `4` |
| `math.sin(n)` | Sine | `0` |
| `math.cos(n)` | Cosine | `1` |
| `math.random()` | Random 0-1 | `0.564...` |

```nsl
print(math.sqrt(16))    # 4
print(math.pow(2, 10))  # 1024
print(math.abs(-42))    # 42
print(math.floor(3.7))  # 3
print(math.ceil(3.2))   # 4
print(math.round(3.5))  # 4
print(math.random())    # 0.xxx
```

---

### json - JSON Handling

Parse and generate JSON.

| Function | Description | Returns |
|----------|-------------|---------|
| `json.parse(str)` | Parse JSON string | `object` |
| `json.stringify(obj)` | Convert to JSON | `string` |
| `json.pretty(obj)` | Pretty print | `string` |
| `json.valid(str)` | Validate JSON | `bool` |

```nsl
# Parse JSON
let data = json.parse('{"name": "NSL", "version": "1.0"}')
print(data.name)     # NSL
print(data.version)  # 1.0

# Create JSON
let obj = {name: "Test", active: true}
print(json.stringify(obj))  # {"name":"Test","active":true}

# Validate
print(json.valid('{"ok": true}'))   # true
print(json.valid('not json'))       # false
```

---

### yaml - YAML Handling

Parse and generate YAML.

| Function | Description | Returns |
|----------|-------------|---------|
| `yaml.parse(str)` | Parse YAML string | `object` |
| `yaml.stringify(obj)` | Convert to YAML | `string` |

```nsl
let yaml_str = "name: NSL\nversion: 1.0.0"
let parsed = yaml.parse(yaml_str)
print(parsed.name)  # NSL

let obj = {app: "Test", port: 8080}
print(yaml.stringify(obj))
# app: Test
# port: 8080
```

---

### regex - Regular Expressions

Pattern matching and text manipulation.

| Function | Description | Returns |
|----------|-------------|---------|
| `regex.test(text, pattern)` | Test if matches | `bool` |
| `regex.match(text, pattern)` | First match | `string` |
| `regex.matches(text, pattern)` | All matches | `array` |
| `regex.replace(text, pattern, repl)` | Replace matches | `string` |
| `regex.split(text, pattern)` | Split by pattern | `array` |

```nsl
let text = "The price is $42.99 and $15.50"

print(regex.test(text, "\\d+"))                    # true
print(regex.match(text, "\\$\\d+\\.\\d+"))         # $42.99
print(regex.matches(text, "\\$\\d+\\.\\d+"))       # [$42.99, $15.50]
print(regex.replace(text, "\\$\\d+\\.\\d+", "[PRICE]"))
# The price is [PRICE] and [PRICE]
print(regex.split("a1b2c3", "\\d"))                # [a, b, c]
```

---

### crypto - Cryptography

Hashing, encoding, and random generation.

| Function | Description | Returns |
|----------|-------------|---------|
| `crypto.hash(data, [algo])` | Hash (sha256, md5, sha1, sha512) | `string` |
| `crypto.uuid()` | Generate UUID | `string` |
| `crypto.random([length])` | Random hex bytes | `string` |
| `crypto.base64encode(data)` | Base64 encode | `string` |
| `crypto.base64decode(data)` | Base64 decode | `string` |

```nsl
print(crypto.hash("hello", "sha256"))
# 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824

print(crypto.hash("hello", "md5"))     # 5d41402abc4b2a76b9719d911017c592
print(crypto.uuid())                   # 61e8639a-8a22-4aa2-a509-9740c1b56895
print(crypto.random(16))               # 0cdc3da418d3c213

let encoded = crypto.base64encode("Hello NSL!")
print(encoded)                         # SGVsbG8gTlNMIQ==
print(crypto.base64decode(encoded))    # Hello NSL!
```

---

### http - HTTP Client

Make HTTP requests.

| Function | Description | Returns |
|----------|-------------|---------|
| `http.get(url, [headers])` | GET request | `{status, ok, body, headers}` |
| `http.post(url, body, [headers])` | POST request | `{status, ok, body, headers}` |
| `http.download(url, path)` | Download file | `bool` |

```nsl
# GET request
let resp = http.get("https://api.example.com/data")
if resp.ok {
    let data = json.parse(resp.body)
    print(data)
}

# Download file
http.download("https://example.com/file.zip", "download.zip")
print("Downloaded: " + file.exists("download.zip"))
```

---

### net - Network Utilities

Network diagnostics and information.

| Function | Description | Returns |
|----------|-------------|---------|
| `net.ping(host)` | Ping host | `{success, time, status}` |
| `net.lookup(host)` | DNS lookup | `array` |
| `net.localIp()` | Local IP address | `string` |
| `net.isOnline()` | Check internet | `bool` |

```nsl
print(net.localIp())     # 192.168.1.100
print(net.isOnline())    # true

let ping = net.ping("google.com")
print(ping.success)      # true
print(ping.time)         # 25 (ms)

print(net.lookup("github.com"))  # [140.82.113.4]
```

---

### env - Environment

Environment variables and system info.

| Function | Description | Returns |
|----------|-------------|---------|
| `env.get(name)` | Get variable | `string` |
| `env.set(name, value)` | Set variable | `bool` |
| `env.home()` | Home directory | `string` |
| `env.user()` | Current username | `string` |
| `env.os()` | OS name | `string` |
| `env.arch()` | Architecture | `string` |

```nsl
print(env.user())  # "desav"
print(env.home())  # "C:\Users\desav"
print(env.os())    # "Windows"
print(env.arch())  # "X64"
```

---

### date - Date/Time

Date and time operations.

| Function | Description | Returns |
|----------|-------------|---------|
| `date.now()` | Current datetime ISO | `string` |
| `date.utc()` | Current UTC datetime | `string` |
| `date.parse(str)` | Parse date string | `object` |
| `date.format(date, fmt)` | Format date | `string` |

```nsl
print(date.now())   # 2025-12-23T00:12:34.827+00:00
print(date.utc())   # 2025-12-23T00:12:34.832Z

let christmas = date.parse("2025-12-25")
print(christmas)    # 2025-12-25T00:00:00.000
```

---

### git - Git Integration

Git repository operations.

| Function | Description | Returns |
|----------|-------------|---------|
| `git.isRepo()` | Check if in git repo | `bool` |
| `git.branch()` | Current branch name | `string` |
| `git.branches()` | All branches | `array` |
| `git.status()` | Repository status | `{clean, files, count}` |
| `git.log([n])` | Commit log | `array` |
| `git.diff([file])` | Show diff | `string` |

```nsl
if git.isRepo() {
    print("Branch: " + git.branch())

    let status = git.status()
    if not status.clean {
        print("Modified files: " + status.count)
    }

    let commits = git.log(5)
    for commit in commits {
        print(commit.message)
    }
}
```

---

### proc - Process Management

System process control.

| Function | Description | Returns |
|----------|-------------|---------|
| `proc.list([filter])` | List processes | `array` |
| `proc.kill(pid)` | Kill process | `bool` |
| `proc.exists(pid)` | Check if running | `bool` |
| `proc.info(pid)` | Process details | `object` |

```nsl
let procs = proc.list("chrome")
print("Chrome processes: " + procs)

# Kill by PID
# proc.kill(1234)
```

---

### clip - Clipboard

System clipboard access.

| Function | Description | Returns |
|----------|-------------|---------|
| `clip.copy(text)` | Copy to clipboard | `bool` |
| `clip.paste()` | Paste from clipboard | `string` |

```nsl
clip.copy("Hello from NSL!")
let content = clip.paste()
print(content)  # Hello from NSL!
```

---

### zip - Archive Operations

Create and extract zip archives.

| Function | Description | Returns |
|----------|-------------|---------|
| `zip.create(src, dest)` | Create zip | `bool` |
| `zip.extract(src, dest)` | Extract zip | `bool` |
| `zip.list(path)` | List contents | `array` |

```nsl
# Create archive
zip.create("my_folder", "archive.zip")

# List contents
let contents = zip.list("archive.zip")
print(contents)

# Extract
zip.extract("archive.zip", "extracted/")
```

---

### diff - Text Comparison

Compare text and files.

| Function | Description | Returns |
|----------|-------------|---------|
| `diff.lines(old, new)` | Line-by-line diff | `array` |
| `diff.files(path1, path2)` | File diff | `array` |
| `diff.patch(content, patch)` | Apply patch | `string` |

```nsl
let old = "line 1\nline 2\nline 3"
let new = "line 1\nline 2 modified\nline 3\nline 4"

let changes = diff.lines(old, new)
print(changes)
# [{type: remove, line: 2, content: line 2},
#  {type: add, line: 2, content: line 2 modified},
#  {type: add, line: 4, content: line 4}]
```

---

### template - Template Engine

String template rendering with variable substitution.

| Function | Description | Returns |
|----------|-------------|---------|
| `template.render(tmpl, vars)` | Render template | `string` |

```nsl
let tmpl = "Hello ${name}! You have ${count} messages."
let vars = {name: "DeSavior", count: 42}

print(template.render(tmpl, vars))
# Hello DeSavior! You have 42 messages.

# HTML template
let html = "<h1>${title}</h1><p>By ${author}</p>"
print(template.render(html, {title: "NSL Guide", author: "DeSavior"}))
# <h1>NSL Guide</h1><p>By DeSavior</p>
```

---

## Consciousness Operators

AI-native data flow operators that match how AI processes information.

| Operator | Name | Description |
|----------|------|-------------|
| `\|>` | **Pipe** | Chain transformations left-to-right |
| `~>` | **Awareness** | Introspective flow with self-reference |
| `=>>` | **Gradient** | Learning/adjustment with feedback |
| `*>` | **Attention** | Focus mechanism with weights |
| `+>` | **Superposition** | Quantum-like state superposition |

### Pipe Operator `|>`

The most commonly used operator. Chains functions left-to-right:

```nsl
# Without pipe (nested, hard to read)
print(list.reverse(list.sort(list.unique([3, 1, 4, 1, 5]))))

# With pipe (clear data flow)
[3, 1, 4, 1, 5] |> list.unique |> list.sort |> list.reverse |> print
# [5, 4, 3, 1]

# String processing
"  HELLO WORLD  " |> string.trim |> string.lower |> print
# hello world

# File processing
file.read("data.txt") |> string.split("\n") |> list.length |> print
```

---

## GPU Operations

NSL supports CUDA GPU acceleration for tensor operations.

### GPU Initialization

```nsl
let info = gpu.init()
print(info.name)          # NVIDIA GeForce RTX 3050 Laptop GPU
print(info.vram_mb)       # 4095
print(info.tensor_cores)  # true
print(info.backend)       # CUDA
print(info.architecture)  # Ampere
```

### GPU Capabilities

| Property | Description |
|----------|-------------|
| `name` | GPU model name |
| `backend` | CUDA backend |
| `vram_mb` | Video RAM in MB |
| `tensor_cores` | Tensor core support |
| `float16` | FP16 support |
| `int8` | INT8 support |
| `compute_units` | Number of compute units |
| `architecture` | GPU architecture |
| `warp_size` | Warp size |
| `max_threads` | Maximum threads |

---

## Interactive Mode

Launch interactive REPL with `nsl` command.

### Quick Commands

| Command | Description |
|---------|-------------|
| `cd folder` | Change directory |
| `ls` | List files |
| `pwd` | Show current path |
| `cat file` | View file contents |
| `mkdir name` | Create folder |
| `rm file` | Delete file |
| `whoami` | Current user |

### Bookmarks

```bash
save myproject              # Save current directory
save work C:\Projects\App   # Save specific path
go myproject                # Jump to bookmark
go                          # List all bookmarks
unsave myproject            # Delete bookmark
```

### Navigation

- **Up/Down arrows** - Command history
- **Tab** - Autocomplete
- **Escape** - Clear line

---

## Execution Modes

### Run Script

```bash
nsl script.nsl
```

### Evaluate Expression

```bash
nsl --eval "2 + 2"
nsl --eval "list.sum([1,2,3,4,5])"
```

### JSON Output

```bash
nsl --eval "sys.platform()" --json
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--eval "code"` | Execute code directly |
| `--json` | Output as JSON |
| `--quiet` | Minimal output |
| `--verbose` | Detailed output |
| `--version` | Show version |
| `--help` | Show help |

---

## Performance

### Benchmark Results

| Operation | Ops/Second |
|-----------|------------|
| Basic arithmetic | ~85,000 |
| Variable access | ~158,000 |
| Function calls | ~83,000 |
| List creation | ~55,000 |

### Optimization Tips

1. **Use pipes** - Cleaner and often faster
2. **Avoid deep recursion** - Use loops when possible
3. **Leverage GPU** - For tensor operations
4. **Use sessions** - For persistent state

---

## License

Copyright 2025 DeSavior Emmanuel White. All Rights Reserved.

NSL is proprietary software. See [LICENSE](LICENSE) for terms.

---

*This reference was generated through live execution testing of every feature.*
