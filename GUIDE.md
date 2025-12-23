# NSL Complete User Guide

**The complete guide to programming in NSL (Neural Symbolic Language)**

---

## Table of Contents

1. [What is NSL?](#what-is-nsl)
2. [Installation](#installation)
3. [Your First Program](#your-first-program)
4. [Running NSL](#running-nsl)
5. [Interactive Mode (REPL)](#interactive-mode-repl)
6. [Variables](#variables)
7. [Data Types](#data-types)
8. [Operators](#operators)
9. [Control Flow](#control-flow)
10. [Functions](#functions)
11. [Arrays and Lists](#arrays-and-lists)
12. [Strings](#strings)
13. [Working with Files](#working-with-files)
14. [Working with Directories](#working-with-directories)
15. [Running Shell Commands](#running-shell-commands)
16. [HTTP Requests](#http-requests)
17. [JSON Handling](#json-handling)
18. [The Pipe Operator](#the-pipe-operator)
19. [Pattern Matching](#pattern-matching)
20. [Error Handling](#error-handling)
21. [Structs](#structs)
22. [Modules and Imports](#modules-and-imports)
23. [Complete Function Reference](#complete-function-reference)
24. [Common Patterns](#common-patterns)
25. [Troubleshooting](#troubleshooting)

---

## What is NSL?

NSL (Neural Symbolic Language) is a programming language designed to be:

- **Simple** - Clean syntax, no boilerplate
- **Powerful** - Full-featured scripting capabilities
- **AI-friendly** - Structured output, predictable behavior

Think of it as a modern replacement for bash/PowerShell with cleaner syntax.

---

## Installation

### Windows

1. Download from [GitHub Releases](https://github.com/DeSa33/NSL/releases)
2. Extract to `C:\NSL`
3. Add to PATH:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\NSL", "User")
   ```
4. Restart your terminal
5. Verify: `nsl --version`

### Linux / macOS

```bash
curl -L https://github.com/DeSa33/NSL/releases/latest/download/nsl-linux-x64.tar.gz | tar xz
sudo mv nsl /usr/local/bin/
nsl --version
```

### Using Scoop (Windows)

```powershell
scoop bucket add nsl https://github.com/DeSa33/scoop-nsl
scoop install nsl
```

---

## Your First Program

Create a file called `hello.nsl`:

```nsl
# This is a comment
print("Hello, NSL!")

# Define a variable
let name = "World"
print("Hello, " + name + "!")

# Define a function
fn greet(person) {
    return "Welcome, " + person + "!"
}

print(greet("Developer"))
```

Run it:
```bash
nsl hello.nsl
```

Output:
```
Hello, NSL!
Hello, World!
Welcome, Developer!
```

---

## Running NSL

### Run a Script
```bash
nsl myscript.nsl
```

### Run Code Directly
```bash
nsl --eval "print(2 + 2)"
```

### Start Interactive Mode
```bash
nsl
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--eval "code"` | Execute code directly |
| `--json` | Output results as JSON |
| `--quiet` | Minimal output |
| `--verbose` | Show all details |
| `--version` | Show version |
| `--help` | Show help |

---

## Interactive Mode (REPL)

Type `nsl` with no arguments to enter interactive mode.

### Features

| Feature | Key | Description |
|---------|-----|-------------|
| History | ↑ / ↓ | Browse previous commands |
| Autocomplete | Tab | Complete function names |
| Clear line | Escape | Clear current input |
| Navigate | ← / → | Move within line |

### Quick Commands

These shortcuts work only in interactive mode (no quotes needed):

```bash
cd Documents      # Change directory
cd ..             # Go up one level
pwd               # Show current path
ls                # List files
cat file.txt      # Show file contents
mkdir newfolder   # Create folder
touch newfile.txt # Create empty file
rm oldfile.txt    # Delete file
whoami            # Show current user
```

### Bookmarks

Save frequently used paths:

```bash
save myproject              # Save current directory
save work C:\Projects\App   # Save specific path
go myproject                # Jump to bookmark
go                          # List all bookmarks
unsave myproject            # Delete bookmark
```

Bookmarks are saved to `~/.nsl/bookmarks.json`.

---

## Variables

### Immutable Variables (Default)

```nsl
let name = "NSL"
let count = 42
let pi = 3.14159
let active = true
```

Immutable means the value cannot be changed after assignment.

### Mutable Variables

Use `mut` when you need to change a value:

```nsl
mut counter = 0
counter = counter + 1
counter = 10  # This is allowed
```

### Constants

```nsl
const MAX_SIZE = 100
const APP_NAME = "MyApp"
```

Constants cannot be changed and are typically UPPERCASE.

---

## Data Types

| Type | Example | Description |
|------|---------|-------------|
| Integer | `42`, `-7`, `0` | Whole numbers |
| Float | `3.14`, `1.5e-10` | Decimal numbers |
| String | `"hello"` | Text |
| Boolean | `true`, `false` | True or false |
| Array | `[1, 2, 3]` | Ordered list |
| Null | `null` | No value |

### Checking Types

```nsl
let x = 42
print(type(x))  # "number"

let s = "hello"
print(type(s))  # "string"
```

---

## Operators

### Math Operators

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `+` | Add | `3 + 2` | `5` |
| `-` | Subtract | `3 - 2` | `1` |
| `*` | Multiply | `3 * 2` | `6` |
| `/` | Divide | `7 / 2` | `3.5` |
| `//` | Integer divide | `7 // 2` | `3` |
| `%` | Remainder | `7 % 2` | `1` |
| `**` | Power | `2 ** 3` | `8` |

### Comparison Operators

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `==` | Equal | `3 == 3` | `true` |
| `!=` | Not equal | `3 != 2` | `true` |
| `<` | Less than | `2 < 3` | `true` |
| `<=` | Less or equal | `3 <= 3` | `true` |
| `>` | Greater than | `3 > 2` | `true` |
| `>=` | Greater or equal | `3 >= 3` | `true` |

### Logical Operators

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `and` | Both true | `true and false` | `false` |
| `or` | Either true | `true or false` | `true` |
| `not` | Opposite | `not true` | `false` |

Alternative syntax: `&&`, `||`, `!`

### Special Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `\|>` | Pipe (chain operations) | `data \|> process \|> output` |
| `??` | Default if null | `value ?? "default"` |
| `?.` | Safe access | `obj?.property` |
| `..` | Range (exclusive) | `0..5` gives `[0,1,2,3,4]` |
| `..=` | Range (inclusive) | `0..=5` gives `[0,1,2,3,4,5]` |

---

## Control Flow

### If / Else

```nsl
let age = 18

if age >= 21 {
    print("Can drink")
} else if age >= 18 {
    print("Can vote")
} else {
    print("Too young")
}
```

Parentheses around conditions are optional.

### If as Expression

```nsl
let status = if age >= 18 { "adult" } else { "minor" }
```

### While Loop

```nsl
mut i = 0
while i < 5 {
    print(i)
    i = i + 1
}
```

### For Loop

```nsl
# Loop through a range
for i in 0..5 {
    print(i)  # 0, 1, 2, 3, 4
}

# Loop through an array
let fruits = ["apple", "banana", "cherry"]
for fruit in fruits {
    print(fruit)
}

# Using range function
for i in range(1, 10, 2) {
    print(i)  # 1, 3, 5, 7, 9
}
```

### Break and Continue

```nsl
for i in 0..10 {
    if i == 5 {
        break  # Exit the loop
    }
    if i % 2 == 0 {
        continue  # Skip to next iteration
    }
    print(i)  # 1, 3
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

### Function with Multiple Parameters

```nsl
fn add(a, b) {
    return a + b
}

print(add(3, 4))  # 7
```

### Function with Type Hints

```nsl
fn multiply(x: number, y: number) -> number {
    return x * y
}
```

### Default Parameters

```nsl
fn greet(name, greeting = "Hello") {
    return greeting + ", " + name + "!"
}

print(greet("World"))           # Hello, World!
print(greet("World", "Hi"))     # Hi, World!
```

### Arrow Functions

Short form for simple functions:

```nsl
let double = fn(x) => x * 2
print(double(5))  # 10
```

### Returning Multiple Values

```nsl
fn minmax(arr) {
    return [list.min(arr), list.max(arr)]
}

let result = minmax([3, 1, 4, 1, 5])
print(result)  # [1, 5]
```

---

## Arrays and Lists

### Creating Arrays

```nsl
let numbers = [1, 2, 3, 4, 5]
let mixed = [1, "two", true, null]
let empty = []
```

### Accessing Elements

```nsl
let fruits = ["apple", "banana", "cherry"]
print(fruits[0])   # apple
print(fruits[1])   # banana
print(fruits[-1])  # cherry (last element)
```

### Array Functions

| Function | Description | Example |
|----------|-------------|---------|
| `list.length(arr)` | Get length | `list.length([1,2,3])` → `3` |
| `list.append(arr, val)` | Add element | `list.append([1,2], 3)` → `[1,2,3]` |
| `list.concat(a, b)` | Join arrays | `list.concat([1],[2])` → `[1,2]` |
| `list.slice(arr, start, count)` | Get portion | `list.slice([1,2,3,4], 1, 2)` → `[2,3]` |
| `list.contains(arr, val)` | Check if contains | `list.contains([1,2], 2)` → `true` |
| `list.sort(arr)` | Sort | `list.sort([3,1,2])` → `[1,2,3]` |
| `list.reverse(arr)` | Reverse | `list.reverse([1,2,3])` → `[3,2,1]` |
| `list.unique(arr)` | Remove duplicates | `list.unique([1,1,2])` → `[1,2]` |
| `list.flatten(arr)` | Flatten nested | `list.flatten([[1],[2]])` → `[1,2]` |
| `list.sum(arr)` | Sum numbers | `list.sum([1,2,3])` → `6` |
| `list.avg(arr)` | Average | `list.avg([1,2,3])` → `2` |
| `list.min(arr)` | Minimum | `list.min([3,1,2])` → `1` |
| `list.max(arr)` | Maximum | `list.max([3,1,2])` → `3` |
| `list.join(arr, sep)` | Join to string | `list.join(["a","b"], "-")` → `"a-b"` |
| `list.range(start, end, step)` | Generate range | `list.range(0, 10, 2)` → `[0,2,4,6,8]` |

### Examples

```nsl
let nums = [5, 2, 8, 1, 9]

# Sort and reverse
let sorted = list.sort(nums)        # [1, 2, 5, 8, 9]
let reversed = list.reverse(nums)   # [9, 1, 8, 2, 5]

# Statistics
print(list.sum(nums))   # 25
print(list.avg(nums))   # 5
print(list.min(nums))   # 1
print(list.max(nums))   # 9

# Combine arrays
let more = [10, 11]
let combined = list.concat(nums, more)  # [5, 2, 8, 1, 9, 10, 11]
```

---

## Strings

### Creating Strings

```nsl
# Regular string
let greeting = "Hello, World!"

# With escape sequences
let multiline = "Line 1\nLine 2"
let quoted = "She said \"Hi\""

# Raw string (no escapes)
let path = r"C:\Users\name\Documents"
let regex = r"\d+\.\d+"

# Multi-line (heredoc)
let html = """
<!DOCTYPE html>
<html>
    <body>Hello</body>
</html>
"""
```

### String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `string.length(s)` | Get length | `string.length("hello")` → `5` |
| `string.upper(s)` | Uppercase | `string.upper("hello")` → `"HELLO"` |
| `string.lower(s)` | Lowercase | `string.lower("HELLO")` → `"hello"` |
| `string.trim(s)` | Remove whitespace | `string.trim("  hi  ")` → `"hi"` |
| `string.split(s, delim)` | Split to array | `string.split("a,b,c", ",")` → `["a","b","c"]` |
| `string.join(arr, sep)` | Join array | `string.join(["a","b"], "-")` → `"a-b"` |
| `string.replace(s, old, new)` | Replace text | `string.replace("hello", "l", "L")` → `"heLLo"` |
| `string.contains(s, sub)` | Check contains | `string.contains("hello", "ell")` → `true` |
| `string.startsWith(s, prefix)` | Check start | `string.startsWith("hello", "he")` → `true` |
| `string.endsWith(s, suffix)` | Check end | `string.endsWith("hello", "lo")` → `true` |
| `string.substring(s, start, len)` | Get portion | `string.substring("hello", 1, 3)` → `"ell"` |
| `string.indexOf(s, sub)` | Find position | `string.indexOf("hello", "l")` → `2` |

### String Multiplication

```nsl
let line = "=" * 40
print(line)  # ========================================
```

---

## Working with Files

### Reading Files

```nsl
# Read entire file as string
let content = file.read("document.txt")
print(content)

# Read file as array of lines
let lines = file.readLines("data.csv")
for line in lines {
    print(line)
}

# Read first 10 lines
let top = file.head("log.txt", 10)

# Read last 10 lines
let bottom = file.tail("log.txt", 10)
```

### Writing Files

```nsl
# Write to file (overwrites)
file.write("output.txt", "Hello, World!")

# Append to file
file.append("log.txt", "New entry\n")
```

### File Operations

```nsl
# Check if file exists
if file.exists("config.json") {
    print("Config found")
}

# Get file size (in bytes)
let size = file.size("video.mp4")
print(size)

# Copy a file
file.copy("original.txt", "backup.txt")

# Move/rename a file
file.move("old_name.txt", "new_name.txt")

# Delete a file
file.delete("temp.txt")
```

### Searching Files

```nsl
# Find files matching pattern (recursive)
let jsFiles = file.find("./src", "*.js")
print(jsFiles)

# Search for text in a file
let matches = file.search("log.txt", "error")
print(matches)

# Grep a directory
let results = file.grep("./src", "TODO")
print(results)
```

### File Functions Reference

| Function | Description |
|----------|-------------|
| `file.read(path)` | Read file contents |
| `file.write(path, content)` | Write to file |
| `file.append(path, content)` | Append to file |
| `file.exists(path)` | Check if exists |
| `file.delete(path)` | Delete file |
| `file.copy(src, dst)` | Copy file |
| `file.move(src, dst)` | Move/rename |
| `file.size(path)` | Get size in bytes |
| `file.readLines(path)` | Read as line array |
| `file.head(path, n)` | First n lines |
| `file.tail(path, n)` | Last n lines |
| `file.search(path, text)` | Search for text |
| `file.find(dir, pattern)` | Find files recursively |
| `file.grep(dir, text)` | Grep directory |

---

## Working with Directories

### Listing Contents

```nsl
# List all files and folders
let items = dir.list(".")
print(items)

# List only files (with pattern)
let txtFiles = dir.files(".", "*.txt")
print(txtFiles)

# List only subdirectories
let folders = dir.dirs(".")
print(folders)

# List all files recursively
let allFiles = dir.walk("./src")
print(allFiles)

# Show directory tree
let tree = dir.tree(".", 2)  # depth of 2
print(tree)
```

### Directory Operations

```nsl
# Check if directory exists
if dir.exists("./backup") {
    print("Backup folder exists")
}

# Create a directory
dir.create("./new_folder")

# Copy a directory
dir.copy("./src", "./src_backup")

# Move a directory
dir.move("./old_folder", "./new_folder")

# Delete a directory (recursive)
dir.delete("./temp", true)

# Get directory size
let size = dir.size("./project")
print(size)
```

### Directory Functions Reference

| Function | Description |
|----------|-------------|
| `dir.list(path)` | List all entries |
| `dir.files(path, pattern)` | List files only |
| `dir.dirs(path)` | List subdirectories |
| `dir.walk(path)` | All files recursive |
| `dir.tree(path, depth)` | Directory tree |
| `dir.exists(path)` | Check if exists |
| `dir.create(path)` | Create directory |
| `dir.copy(src, dst)` | Copy directory |
| `dir.move(src, dst)` | Move directory |
| `dir.delete(path, recursive)` | Delete directory |
| `dir.size(path)` | Get total size |

---

## Running Shell Commands

### Basic Execution

```nsl
# Run command and get full result
let result = sys.exec("git status")

if result.success {
    print(result.stdout)
} else {
    print("Error: " + result.stderr)
}
```

The result contains:
- `stdout` - Standard output
- `stderr` - Error output
- `code` - Exit code
- `success` - true if exit code is 0

### Simple Execution

```nsl
# Just get stdout
let output = sys.shell("date")
print(output)
```

### With Options

```nsl
# Run in specific directory
let result = sys.exec("npm install", "C:/Projects/MyApp")

# With timeout (milliseconds)
let result = sys.exec("long-command", null, 30000)
```

### Piping Commands

```nsl
# Chain commands like bash pipes
let count = sys.pipe("dir /b", "find /c /v \"\"")
print(count)
```

### System Functions Reference

| Function | Description |
|----------|-------------|
| `sys.exec(cmd, cwd, timeout)` | Run command, get full result |
| `sys.shell(cmd)` | Run command, get stdout only |
| `sys.pipe(cmd1, cmd2, ...)` | Chain commands |
| `sys.cwd()` | Get current directory |
| `sys.cd(path)` | Change directory |
| `sys.env(name)` | Get environment variable |
| `sys.setenv(name, value)` | Set environment variable |
| `sys.platform()` | Get OS info |
| `sys.hostname()` | Get machine name |
| `sys.username()` | Get current user |
| `sys.home()` | Get home directory |
| `sys.temp()` | Get temp directory |
| `sys.pid()` | Get process ID |
| `sys.kill(pid)` | Kill a process |
| `sys.sleep(ms)` | Sleep milliseconds |
| `sys.time()` | Get ISO timestamp |
| `sys.timestamp()` | Get Unix timestamp |
| `sys.exit(code)` | Exit with code |
| `sys.which(name)` | Find executable path |

---

## HTTP Requests

### GET Request

```nsl
let response = http.get("https://api.example.com/users")

if response.ok {
    let data = json.parse(response.body)
    print(data)
}
```

### POST Request

```nsl
let body = json.stringify({
    name: "John",
    email: "john@example.com"
})

let response = http.post(
    "https://api.example.com/users",
    body,
    "application/json"
)

print(response.status)
```

### Download File

```nsl
http.download(
    "https://example.com/file.zip",
    "downloaded.zip"
)
print("Download complete!")
```

### HTTP Functions Reference

| Function | Description |
|----------|-------------|
| `http.get(url, timeout)` | GET request |
| `http.post(url, body, contentType)` | POST request |
| `http.head(url)` | HEAD request |
| `http.download(url, path)` | Download file |

---

## JSON Handling

### Parsing JSON

```nsl
let jsonString = '{"name": "John", "age": 30}'
let data = json.parse(jsonString)

print(data.name)  # John
print(data.age)   # 30
```

### Creating JSON

```nsl
let user = {
    name: "John",
    age: 30,
    active: true
}

let jsonString = json.stringify(user)
print(jsonString)

# Pretty print
let pretty = json.stringify(user, true)
print(pretty)
```

### Reading JSON Files

```nsl
let config = file.read("config.json") |> json.parse
print(config.setting)
```

### Writing JSON Files

```nsl
let data = { users: ["Alice", "Bob"] }
file.write("data.json", json.stringify(data, true))
```

### JSON Functions Reference

| Function | Description |
|----------|-------------|
| `json.parse(str)` | Parse JSON string to object |
| `json.stringify(obj, indent)` | Convert object to JSON string |
| `json.pretty(str)` | Pretty print JSON string |
| `json.valid(str)` | Check if valid JSON |

---

## The Pipe Operator

The pipe operator `|>` chains operations left-to-right:

```nsl
# Without pipe (nested calls)
print(string.upper(string.trim("  hello  ")))

# With pipe (readable chain)
"  hello  " |> string.trim |> string.upper |> print
```

### More Examples

```nsl
# Process numbers
let result = [3, 1, 4, 1, 5]
    |> list.unique
    |> list.sort
    |> list.reverse
print(result)  # [5, 4, 3, 1]

# Process file
let lineCount = file.read("data.txt")
    |> text.lines
    |> list.length
print(lineCount)

# Process API response
let users = http.get("https://api.example.com/users").body
    |> json.parse
print(users)
```

---

## Pattern Matching

### Basic Match

```nsl
let day = "Monday"

match day {
    "Monday" => print("Start of week"),
    "Friday" => print("Almost weekend"),
    "Saturday" | "Sunday" => print("Weekend!"),
    _ => print("Regular day")
}
```

### Match with Values

```nsl
fn describe(value) {
    return match value {
        0 => "zero",
        1 => "one",
        n if n < 0 => "negative",
        n if n > 100 => "large",
        _ => "other"
    }
}
```

### Match Arrays

```nsl
let point = [3, 4]

match point {
    [0, 0] => print("Origin"),
    [x, 0] => print("On X axis"),
    [0, y] => print("On Y axis"),
    [x, y] => print("Point at " + x + ", " + y)
}
```

---

## Error Handling

### Result Type

Functions can return `Ok(value)` or `Err(error)`:

```nsl
fn divide(a, b) {
    if b == 0 {
        return Err("Cannot divide by zero")
    }
    return Ok(a / b)
}

let result = divide(10, 2)

match result {
    Ok(value) => print("Result: " + value),
    Err(msg) => print("Error: " + msg)
}
```

### Option Type

For values that might not exist:

```nsl
fn find(arr, target) {
    for i in 0..list.length(arr) {
        if arr[i] == target {
            return Some(i)
        }
    }
    return None
}

let index = find([10, 20, 30], 20)

match index {
    Some(i) => print("Found at index " + i),
    None => print("Not found")
}
```

### Null Coalescing

```nsl
let value = maybeNull ?? "default"
```

### Safe Navigation

```nsl
let name = user?.profile?.name ?? "Anonymous"
```

---

## Structs

### Defining a Struct

```nsl
struct Person {
    name: string,
    age: number
}

let person = Person {
    name: "Alice",
    age: 30
}

print(person.name)  # Alice
print(person.age)   # 30
```

### Struct Methods

```nsl
struct Rectangle {
    width: number,
    height: number
}

impl Rectangle {
    fn area(self) {
        return self.width * self.height
    }
    
    fn perimeter(self) {
        return 2 * (self.width + self.height)
    }
}

let rect = Rectangle { width: 10, height: 5 }
print(rect.area())       # 50
print(rect.perimeter())  # 30
```

---

## Modules and Imports

### Importing a Module

```nsl
import math
import utils from "./utils.nsl"
```

### Selective Import

```nsl
from math import sin, cos, PI
```

### Using Imported Functions

```nsl
import math

let x = math.sin(0)
let y = math.cos(0)
```

---

## Complete Function Reference

### System (sys)

| Function | Description |
|----------|-------------|
| `sys.exec(cmd)` | Run command |
| `sys.shell(cmd)` | Run command, get stdout |
| `sys.pipe(...)` | Chain commands |
| `sys.cwd()` | Current directory |
| `sys.cd(path)` | Change directory |
| `sys.env(name)` | Get env variable |
| `sys.setenv(name, val)` | Set env variable |
| `sys.platform()` | OS info |
| `sys.hostname()` | Machine name |
| `sys.username()` | Current user |
| `sys.home()` | Home directory |
| `sys.temp()` | Temp directory |
| `sys.time()` | ISO timestamp |
| `sys.timestamp()` | Unix timestamp |
| `sys.sleep(ms)` | Sleep |
| `sys.exit(code)` | Exit |

### Files (file)

| Function | Description |
|----------|-------------|
| `file.read(path)` | Read file |
| `file.write(path, content)` | Write file |
| `file.append(path, content)` | Append to file |
| `file.exists(path)` | Check exists |
| `file.delete(path)` | Delete file |
| `file.copy(src, dst)` | Copy file |
| `file.move(src, dst)` | Move file |
| `file.size(path)` | File size |
| `file.readLines(path)` | Read as lines |
| `file.head(path, n)` | First n lines |
| `file.tail(path, n)` | Last n lines |
| `file.find(dir, pattern)` | Find files |
| `file.grep(dir, text)` | Search files |
| `file.search(path, text)` | Search in file |

### Directories (dir)

| Function | Description |
|----------|-------------|
| `dir.list(path)` | List contents |
| `dir.files(path, pattern)` | List files |
| `dir.dirs(path)` | List subdirs |
| `dir.walk(path)` | All files recursive |
| `dir.tree(path, depth)` | Directory tree |
| `dir.exists(path)` | Check exists |
| `dir.create(path)` | Create dir |
| `dir.delete(path, recursive)` | Delete dir |
| `dir.copy(src, dst)` | Copy dir |
| `dir.move(src, dst)` | Move dir |
| `dir.size(path)` | Total size |

### Strings (string)

| Function | Description |
|----------|-------------|
| `string.length(s)` | Length |
| `string.upper(s)` | Uppercase |
| `string.lower(s)` | Lowercase |
| `string.trim(s)` | Trim whitespace |
| `string.split(s, delim)` | Split to array |
| `string.replace(s, old, new)` | Replace |
| `string.contains(s, sub)` | Contains check |
| `string.startsWith(s, prefix)` | Starts with |
| `string.endsWith(s, suffix)` | Ends with |
| `string.substring(s, start, len)` | Substring |
| `string.indexOf(s, sub)` | Find index |

### Lists (list)

| Function | Description |
|----------|-------------|
| `list.length(arr)` | Length |
| `list.append(arr, val)` | Add element |
| `list.concat(a, b)` | Join arrays |
| `list.slice(arr, start, count)` | Get portion |
| `list.contains(arr, val)` | Contains check |
| `list.sort(arr)` | Sort |
| `list.reverse(arr)` | Reverse |
| `list.unique(arr)` | Remove duplicates |
| `list.flatten(arr)` | Flatten nested |
| `list.sum(arr)` | Sum |
| `list.avg(arr)` | Average |
| `list.min(arr)` | Minimum |
| `list.max(arr)` | Maximum |
| `list.join(arr, sep)` | Join to string |
| `list.range(start, end, step)` | Generate range |

### JSON (json)

| Function | Description |
|----------|-------------|
| `json.parse(str)` | Parse JSON |
| `json.stringify(obj, indent)` | Create JSON |
| `json.pretty(str)` | Pretty print |
| `json.valid(str)` | Validate |

### HTTP (http)

| Function | Description |
|----------|-------------|
| `http.get(url, timeout)` | GET request |
| `http.post(url, body, type)` | POST request |
| `http.head(url)` | HEAD request |
| `http.download(url, path)` | Download file |

### Math (math)

| Function | Description |
|----------|-------------|
| `math.sqrt(n)` | Square root |
| `math.pow(base, exp)` | Power |
| `math.abs(n)` | Absolute value |
| `math.floor(n)` | Round down |
| `math.ceil(n)` | Round up |
| `math.round(n, decimals)` | Round |
| `math.sin(n)` | Sine |
| `math.cos(n)` | Cosine |
| `math.tan(n)` | Tangent |
| `math.log(n, base)` | Logarithm |
| `math.exp(n)` | e^n |
| `math.min(...)` | Minimum |
| `math.max(...)` | Maximum |
| `math.random(min, max)` | Random number |

### Crypto (crypto)

| Function | Description |
|----------|-------------|
| `crypto.hash(data, algo)` | Hash (md5/sha256) |
| `crypto.uuid()` | Generate UUID |
| `crypto.random(length)` | Random hex |
| `crypto.base64encode(data)` | Base64 encode |
| `crypto.base64decode(data)` | Base64 decode |

### Path (path)

| Function | Description |
|----------|-------------|
| `path.join(...)` | Join paths |
| `path.dirname(path)` | Get directory |
| `path.basename(path)` | Get filename |
| `path.ext(path)` | Get extension |
| `path.stem(path)` | Name without ext |
| `path.absolute(path)` | Absolute path |
| `path.exists(path)` | Check exists |
| `path.isFile(path)` | Is file? |
| `path.isDir(path)` | Is directory? |

### Regex (regex)

| Function | Description |
|----------|-------------|
| `regex.match(text, pattern)` | First match |
| `regex.matches(text, pattern)` | All matches |
| `regex.test(text, pattern)` | Test if matches |
| `regex.replace(text, pat, repl)` | Replace matches |
| `regex.split(text, pattern)` | Split by pattern |
| `regex.groups(text, pattern)` | Capture groups |

---

## Common Patterns

### Read and Process a Config File

```nsl
let config = file.read("config.json") |> json.parse

print("App: " + config.name)
print("Version: " + config.version)
```

### Find All Files of a Type

```nsl
let jsFiles = file.find("./src", "*.js")

for f in jsFiles {
    print(f)
}
```

### Process CSV Data

```nsl
let lines = file.readLines("data.csv")

for line in lines {
    let parts = string.split(line, ",")
    print(parts[0] + ": " + parts[1])
}
```

### Make an API Call

```nsl
let response = http.get("https://api.github.com/users/octocat")
let user = json.parse(response.body)

print("Name: " + user.name)
print("Followers: " + user.followers)
```

### Run Git Commands

```nsl
# Check status
let status = sys.exec("git status")
print(status.stdout)

# Add and commit
sys.exec("git add .")
sys.exec("git commit -m \"Update\"")
```

### Watch a Log File

```nsl
while true {
    let lastLines = file.tail("app.log", 5)
    print(lastLines)
    sys.sleep(1000)
}
```

### Create a Directory Structure

```nsl
let folders = ["src", "tests", "docs", "build"]

for folder in folders {
    if not dir.exists(folder) {
        dir.create(folder)
        print("Created: " + folder)
    }
}
```

---

## Troubleshooting

### Common Errors

**"// is not a comment"**
```nsl
# Wrong - // is integer division
// This is NOT a comment

# Correct - use #
# This IS a comment
```

**"Variable cannot be reassigned"**
```nsl
# Wrong - let is immutable
let x = 1
x = 2  # Error!

# Correct - use mut
mut x = 1
x = 2  # OK
```

**"File not found"**
```nsl
# Check path
print(sys.cwd())  # Where am I?

# Use absolute path
file.read(r"C:\full\path\to\file.txt")
```

**"JSON parse error"**
```nsl
# Check if valid JSON first
let str = file.read("data.json")
if json.valid(str) {
    let data = json.parse(str)
} else {
    print("Invalid JSON!")
}
```

### Getting Help

- **Issues:** https://github.com/DeSa33/NSL/issues
- **Email:** desaviorwhite@protonmail.com

---

## License

Copyright © 2025 DeSavior Emmanuel White. All Rights Reserved.

NSL is proprietary software. See [LICENSE](https://github.com/DeSa33/NSL/blob/main/LICENSE) for terms.
