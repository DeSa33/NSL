# NSL Quick Reference

## Global Functions

| Function | Description | Example |
|----------|-------------|---------|
| `print(...)` | Print to console | `print("Hello", x)` |
| `input(prompt)` | Read user input | `let name = input("Name: ")` |
| `type(x)` | Get type name | `type([1,2,3])` → "list" |
| `typeof(x)` | Get NSL type | `typeof(42)` → "number" |
| `len(x)` | Get length | `len("hello")` → 5 |
| `range(n)` | Create range | `range(5)` → [0,1,2,3,4] |
| `range(start, end)` | Range with start | `range(2, 5)` → [2,3,4] |
| `help()` | Show help | `help()` or `help("file")` |
| `keys(obj)` | Get keys of namespace | `keys(file)` |
| `inspect(obj)` | Detailed object info | `inspect(string)` |
| `assert(cond, msg)` | Assert condition | `assert(x > 0, "Must be positive")` |
| `debug(...)` | Debug print with types | `debug(x, y)` |
| `shell(cmd)` | Execute shell command | `shell("dir")` |
| `run(cmd)` | Run command (no output) | `run("npm install")` |
| `read_file(path)` | Read file contents | `read_file("config.json")` |
| `write_file(path, content)` | Write file | `write_file("out.txt", data)` |

## Namespaces

### file - File Operations
```nsl
file.read(path)           # Read file contents
file.write(path, content) # Write to file  
file.append(path, text)   # Append to file
file.exists(path)         # Check if exists
file.delete(path)         # Delete file
file.copy(src, dst)       # Copy file
file.move(src, dst)       # Move file
file.list(dir)            # List files in directory
file.list(dir, "*.txt")   # List with pattern
file.find(dir, "*.cs")    # Find files recursively
file.search(path, text)   # Search for text in file (returns line numbers)
file.grep(dir, text)      # Find files containing text
file.readLines(path)      # Read as list of lines
file.size(path)           # Get file size in bytes
```

### dir - Directory Operations
```nsl
dir.list()                # List current directory
dir.list(path)            # List directory contents
dir.files(path)           # List only files
dir.files(path, "*.js")   # List with pattern
dir.dirs(path)            # List only directories
dir.create(path)          # Create directory
dir.delete(path)          # Delete empty directory
dir.delete(path, true)    # Delete recursively
dir.copy(src, dst)        # Copy directory recursively
dir.move(src, dst)        # Move directory
dir.size(path)            # Total size of directory
dir.tree(path)            # Show directory tree
dir.tree(path, 2)         # Tree with max depth
dir.walk(path)            # List all files recursively
```

### path - Path Manipulation
```nsl
path.join("dir", "file")  # Join path parts
path.dirname(p)           # Get directory part
path.basename(p)          # Get filename part
path.ext(p)               # Get extension
path.stem(p)              # Get filename without extension
path.absolute(p)          # Get absolute path
path.normalize(p)         # Normalize with forward slashes
path.isAbsolute(p)        # Check if absolute
path.exists(p)            # Check if path exists
path.isFile(p)            # Check if file
path.isDir(p)             # Check if directory
path.relative(p, base)    # Get relative path
```

### sys - System Operations
```nsl
sys.cwd()                 # Current working directory
sys.cd(path)              # Change directory
sys.home()                # Home directory
sys.temp()                # Temp directory
sys.env("PATH")           # Get environment variable
sys.setenv("KEY", "val")  # Set environment variable
sys.platform()            # OS platform
sys.hostname()            # Machine name
sys.username()            # Current user
sys.processors()          # CPU count
sys.memory()              # Memory used (bytes)
sys.time()                # Current time (ISO format)
sys.timestamp()           # Unix timestamp (ms)
sys.sleep(1000)           # Sleep milliseconds
sys.exit(0)               # Exit with code
sys.args()                # Command line arguments
```

### string - String Operations
```nsl
string.length(s)          # String length
string.upper(s)           # To uppercase
string.lower(s)           # To lowercase
string.trim(s)            # Trim whitespace
string.split(s, ",")      # Split into list
string.join(list, ",")    # Join list to string
string.replace(s, a, b)   # Replace all occurrences
string.contains(s, sub)   # Check if contains
string.startsWith(s, p)   # Check prefix
string.endsWith(s, p)     # Check suffix
string.substring(s, 0, 5) # Get substring
string.indexOf(s, sub)    # Find position (-1 if not found)
string.repeat(s, 3)       # Repeat string
string.padLeft(s, 10)     # Pad left to width
string.padRight(s, 10)    # Pad right to width
```

### list - List Operations
```nsl
list.append(lst, val)     # Add to end (returns new list)
list.prepend(lst, val)    # Add to start
list.remove(lst, val)     # Remove first occurrence
list.slice(lst, 1, 3)     # Get subset [start, count]
list.get(lst, 0)          # Get item at index
list.first(lst)           # Get first item
list.last(lst)            # Get last item
list.length(lst)          # Get length
list.sum(lst)             # Sum of numbers
list.avg(lst)             # Average of numbers
list.min(lst)             # Minimum value
list.max(lst)             # Maximum value
list.sort(lst)            # Sort ascending
list.reverse(lst)         # Reverse list
list.contains(lst, val)   # Check if contains
list.join(lst, ",")       # Join to string
list.concat(lst1, lst2)   # Concatenate lists
list.unique(lst)          # Remove duplicates
list.flatten(lst)         # Flatten nested lists
list.range(5)             # Create [0,1,2,3,4]
list.range(1, 5)          # Create [1,2,3,4]
list.range(0, 10, 2)      # Create [0,2,4,6,8]
```

### regex - Regular Expressions
```nsl
regex.match(text, pattern)     # Get first match
regex.matches(text, pattern)   # Get all matches as list
regex.test(text, pattern)      # Test if matches
regex.replace(text, pat, repl) # Replace with regex
regex.split(text, pattern)     # Split by pattern
regex.groups(text, pattern)    # Get capture groups
regex.escape(text)             # Escape special chars
```

### text - Text Processing
```nsl
text.lines(s)             # Split into lines
text.words(s)             # Split into words
text.chars(s)             # Split into characters
text.count(s, sub)        # Count occurrences
text.reverse(s)           # Reverse text
text.wrap(s, 80)          # Word wrap to width
text.truncate(s, 50)      # Truncate with "..."
text.truncate(s, 50, "→") # Custom suffix
text.indent(s, 4)         # Indent all lines
text.dedent(s)            # Remove common indentation
text.slug(s)              # Convert to URL slug
text.title(s)             # Title Case
```

### math - Mathematical Functions
```nsl
math.sqrt(x)              # Square root
math.sin(x), math.cos(x), math.tan(x)  # Trig
math.asin(x), math.acos(x), math.atan(x)  # Inverse trig
math.abs(x)               # Absolute value
math.floor(x)             # Round down
math.ceil(x)              # Round up
math.round(x)             # Round to nearest
math.round(x, 2)          # Round to decimals
math.log(x)               # Natural log
math.log(x, base)         # Log with base
math.log10(x)             # Base 10 log
math.exp(x)               # e^x
math.pow(x, y)            # x^y
math.min(a, b)            # Minimum
math.max(a, b)            # Maximum
math.sign(x)              # Sign (-1, 0, 1)
math.random()             # Random 0-1
math.pi                   # π constant
math.e                    # e constant
```

### gpu - GPU Operations
```nsl
gpu.init()                # Initialize GPU
gpu.tensor(data)          # Create tensor
gpu.matmul(a, b)          # Matrix multiply
gpu.add(a, b)             # Element-wise add
gpu.multiply(a, b)        # Element-wise multiply
gpu.transpose(t)          # Transpose tensor
gpu.benchmark()           # Run benchmark
```

### json - JSON Operations
```nsl
json.parse(text)          # Parse JSON string
json.stringify(obj)       # Convert to JSON
json.pretty(obj)          # Pretty print JSON
```

### http - HTTP Operations
```nsl
http.get(url)             # GET request
http.post(url, data)      # POST request
http.put(url, data)       # PUT request
http.delete(url)          # DELETE request
```

## Consciousness Operators

| Operator | Name | Description |
|----------|------|-------------|
| `\|>` | pipe | Chain transformations left-to-right |
| `~>` | awareness | Introspective flow with self-reference |
| `=>>` | gradient | Learning/adjustment with feedback |
| `*>` | attention | Focus mechanism with weights |
| `+>` | superposition | Quantum-like state superposition |

## Control Flow

```nsl
# If/else
if condition {
    # code
} else {
    # code
}

# While loop
while condition {
    # code
}

# For loop
for item in list {
    print(item)
}

for i in range(10) {
    print(i)
}

# Break and continue
while true {
    if done { break }
    if skip { continue }
}
```

## Functions

```nsl
# Define function
fn add(a, b) {
    return a + b
}

# Call function
let result = add(2, 3)

# Lambda
let double = fn(x) { x * 2 }
```

## Variables

```nsl
let x = 42              # Number
let s = "hello"         # String
let b = true            # Boolean
let lst = [1, 2, 3]     # List
let none_val = none     # None/null
```
