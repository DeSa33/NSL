# How to Code in NSL (Neural Symbolic Language)

A comprehensive guide to programming in NSL - the Neural Symbolic Language designed for AI/ML applications.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Syntax](#basic-syntax)
3. [Variables and Data Types](#variables-and-data-types)
4. [Operators](#operators)
5. [Control Flow](#control-flow)
6. [Functions](#functions)
7. [Closures and Higher-Order Functions](#closures-and-higher-order-functions)
8. [Arrays and Collections](#arrays-and-collections)
9. [Structs](#structs)
10. [Enums (Algebraic Data Types)](#enums-algebraic-data-types)
11. [Pattern Matching](#pattern-matching)
12. [Error Handling (Result/Option)](#error-handling-resultoption)
13. [Traits and Interfaces](#traits-and-interfaces)
14. [Modules and Imports](#modules-and-imports)
15. [Async/Await](#asyncawait)
16. [Consciousness Operators](#consciousness-operators)
17. [AI-Centric Features](#ai-centric-features) ⭐ NEW
18. [Neural Network Operations](#neural-network-operations)
19. [Package Management (nslpm)](#package-management-nslpm)
20. [Standard Library](#standard-library)
21. [REPL and Debugging](#repl-and-debugging)
22. [Auto-Fix (Error Correction)](#auto-fix-error-correction)
23. [Best Practices](#best-practices)
24. [Phase 5: Developer Tools](#phase-5-developer-tools--new) ⭐ NEW

---

## Getting Started

### Running NSL

```bash
# Run a file
nsl script.nsl

# Start the REPL (interactive mode)
nsl

# Run with auto-fix suggestions
nsl --suggest script.nsl

# Auto-fix errors and warnings
nsl --fix script.nsl

# Output options
nsl script.nsl --verbose    # Show all details
nsl script.nsl --log        # Save output to ~/.nsl/logs/
nsl script.nsl --quiet      # Minimal output

# Color customization
nsl --colors                # Create ~/.nsl/colors.json config
nsl script.nsl --no-color   # Disable colors
```

### Interactive REPL Features

When you run `nsl` without arguments, you enter the interactive REPL with:

| Feature | Key | Description |
|---------|-----|-------------|
| **Command History** | ↑/↓ | Browse previous commands |
| **Tab Completion** | Tab | Complete namespaces and functions |
| **Line Editing** | ←/→, Home/End | Navigate within line |
| **Clear Input** | Escape | Clear current line |

Type `fi` then Tab → completes to `file.`
Type `file.re` then Tab → completes to `file.read`

History is saved to `~/.nsl/history` between sessions.

### Interactive Shortcuts (REPL Only)

These commands work in the interactive REPL - **no quotes, no parentheses needed**.

> ⚠️ **Note:** These are REPL conveniences only. For scripts, use the full syntax in the next section.

```bash
# Navigation
cd              # Go to home directory
cd Desktop      # Go to folder (relative path)
cd C:\Users     # Go to folder (absolute path)
cd ..           # Go up one level
pwd             # Show current directory

# Files & Folders
ls              # List files in current directory
ls Documents    # List files in specific folder
cat file.txt    # Show file contents
mkdir myfolder  # Create folder
touch note.txt  # Create empty file
rm file.txt     # Delete file

# System
whoami          # Show current user

# Bookmarks (save long paths)
save nsl                              # Save current dir as "nsl"
save console E:\NSL.Interpreter\src   # Save specific path
go nsl                                # Jump to bookmark
go nsl/src                            # Jump to bookmark + subpath
go                                    # List all bookmarks
unsave nsl                            # Delete a bookmark
```

Bookmarks persist in `~/.nsl/bookmarks.json`.

### Full Syntax (For Scripts)

For scripts and programmatic use, use the full function syntax:

```nsl
sys.cd(r"C:\Users\desav")   # Change directory
file.cwd()                   # Get current directory  
dir.list()                   # List as array
dir.list("Documents")        # List specific folder
```

### Quick Tips

```nsl
"=" * 40              # String multiplication works!
json.parse(str).key   # JSON values compare correctly
```

### Your First NSL Program

Create a file `hello.nsl`:

```nsl
# This is a comment (use # not //)
# // is the integer division operator in NSL

print("Hello, NSL!")
print("Welcome to the Neural Symbolic Language")

# Define a function
fn greet(name) {
    return "Hello, " + name + "!"
}

print(greet("World"))
```

Run it:
```bash
nsl hello.nsl
```

**Output:**
```
Hello, NSL!
Welcome to the Neural Symbolic Language
Hello, World!
```

---

## Basic Syntax

### Comments

```nsl
# Single-line comments use hash
# NOT double-slash! // is integer division

# Multi-line comments use /* */
/*
This is a
multi-line comment
*/
```

> **Important:** `//` is the **integer division operator**, not a comment. Use `#` for single-line comments.

### Statements

Statements are separated by newlines. Semicolons are optional:

```nsl
let x = 10
let y = 20
print(x + y)  # Output: 30
```

### Blocks

Blocks use curly braces `{}`:

```nsl
{
    let x = 10
    print(x)
}
```

---

## Variables and Data Types

### Variable Declaration

NSL variables are **immutable by default** (like Rust):

```nsl
# Immutable (cannot be changed)
let x = 42
let name = "NSL"
let pi = 3.14159

# Mutable (can be changed)
mut counter = 0
counter = counter + 1
counter = 10  # OK

# Constants
const MAX_SIZE = 100
```

### Data Types

| Type | Example | Description |
|------|---------|-------------|
| Integer | `42`, `-7` | Whole numbers |
| Number (Float) | `3.14`, `1e-5` | Floating-point numbers |
| String | `"hello"`, `r"raw"`, `"""heredoc"""` | Text |
| Boolean | `true`, `false` | True/false values |
| Array | `[1, 2, 3]` | Ordered collections |
| Null | `null` | Absence of value |
| Function | `fn(x) { x * 2 }` | First-class functions |

### String Literals

NSL supports three types of string literals:

```nsl
# Regular strings - escape sequences processed
let greeting = "Hello\nWorld"     # \n becomes newline
let quoted = "She said \"Hi\""    # \" becomes quote

# Raw strings (r prefix) - no escape processing
let regex = r"\d+\.\d+"           # Backslashes are literal
let winpath = r"C:\Users\name"    # No need to escape backslashes

# Heredoc strings (triple quotes) - multi-line, no escapes
let template = """
<!DOCTYPE html>
<html>
    <body>Hello</body>
</html>
"""
```

**When to use each:**
- **Regular `"..."`**: Normal text, simple strings
- **Raw `r"..."`**: Regex patterns, Windows paths, code with backslashes
- **Heredoc `"""..."""`**: Multi-line templates, embedded code, SQL queries

### Type Hints (Optional)

```nsl
let x: number = 42.0
let name: string = "NSL"
let items: vec = [1, 2, 3]
mut counter: int = 0

fn add(a: number, b: number) -> number {
    return a + b
}
```

Available type hints: `number`, `int`, `string`, `bool`, `vec`, `mat`, `tensor`, `prob`

---

## Operators

### Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `3 + 2` = `5` |
| `-` | Subtraction | `3 - 2` = `1` |
| `*` | Multiplication | `3 * 2` = `6` |
| `/` | Division | `7 / 2` = `3.5` |
| `//` | Integer Division | `7 // 2` = `3` |
| `%` | Modulo | `7 % 2` = `1` |
| `**` | Power | `2 ** 3` = `8` |

### Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `3 == 3` = `true` |
| `!=` | Not Equal | `3 != 2` = `true` |
| `<` | Less Than | `2 < 3` = `true` |
| `<=` | Less or Equal | `3 <= 3` = `true` |
| `>` | Greater Than | `3 > 2` = `true` |
| `>=` | Greater or Equal | `3 >= 3` = `true` |

### Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `and` | Logical AND | `true and false` = `false` |
| `or` | Logical OR | `true or false` = `true` |
| `not` | Logical NOT | `not true` = `false` |
| `&&` | Logical AND (alt) | `true && false` |
| `\|\|` | Logical OR (alt) | `true \|\| false` |
| `!` | Logical NOT (alt) | `!true` |

### Bitwise Operators

| Operator | Description |
|----------|-------------|
| `&` | Bitwise AND |
| `\|` | Bitwise OR |
| `^` | Bitwise XOR |
| `~` | Bitwise NOT |
| `<<` | Left Shift |
| `>>` | Right Shift |

### Special Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `\|>` | Pipeline | `data \|> normalize \|> predict` |
| `??` | Null Coalescing | `value ?? "default"` |
| `?.` | Safe Navigation | `obj?.property` |
| `..` | Range (exclusive) | `0..5` = [0,1,2,3,4] |
| `..=` | Range (inclusive) | `0..=5` = [0,1,2,3,4,5] |
| `=>` | Fat Arrow (lambda/match) | `x => x * 2` |
| `::` | Chain/Namespace | `Math::PI`, `Color::Red` |
| `@` | Matrix Multiply | `A @ B` |

---

## Control Flow

### If/Else

```nsl
let score = 85

if (score >= 90) {
    print("Grade: A")
} else if (score >= 80) {
    print("Grade: B")
} else if (score >= 70) {
    print("Grade: C")
} else {
    print("Grade: F")
}
```

Parentheses around conditions are optional:

```nsl
if score >= 90 {
    print("A")
}
```

### If Expression

If can be used as an expression:

```nsl
let grade = if score >= 90 { "A" } else { "B" }
```

### While Loop

```nsl
mut i = 0
while (i < 5) {
    print(i)
    i = i + 1
}
```

### For Loop

```nsl
# Range-based for loop
for i in range(0, 5) {
    print(i)  # 0, 1, 2, 3, 4
}

# Inclusive range
for i in 1..=5 {
    print(i)  # 1, 2, 3, 4, 5
}

# Exclusive range
for i in 0..5 {
    print(i)  # 0, 1, 2, 3, 4
}

# Iterate over array
let items = [10, 20, 30]
for item in items {
    print(item)
}
```

### Break and Continue

```nsl
for i in 0..10 {
    if (i == 5) {
        break  # Exit loop
    }
    if (i % 2 == 0) {
        continue  # Skip to next iteration
    }
    print(i)  # 1, 3
}
```

---

## Functions

### Basic Functions

```nsl
fn add(a, b) {
    return a + b
}

fn greet(name) {
    return "Hello, " + name + "!"
}

print(add(3, 4))       # 7
print(greet("World"))  # Hello, World!
```

### Functions with Type Hints

```nsl
fn square(x: number) -> number {
    return x * x
}

fn max(a: number, b: number) -> number {
    if (a > b) {
        return a
    }
    return b
}
```

### Arrow Functions (Expression Body)

```nsl
# Short form for single-expression functions
async fn square(n: number) => n * n
```

### Recursion

```nsl
fn factorial(n) {
    if (n <= 1) {
        return 1
    }
    return n * factorial(n - 1)
}

print(factorial(5))  # 120

fn fibonacci(n) {
    if (n <= 1) {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}
```

---

## Closures and Higher-Order Functions

### Closures

Functions can capture variables from their enclosing scope:

```nsl
fn makeMultiplier(factor) {
    fn multiplier(x) {
        return x * factor
    }
    return multiplier
}

let double = makeMultiplier(2)
let triple = makeMultiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

### Stateful Closures (Mutable Captures)

```nsl
fn makeCounter(start) {
    let count = start
    fn counter() {
        count = count + 1
        return count
    }
    return counter
}

let counter = makeCounter(0)
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

### Higher-Order Functions

Functions that take or return other functions:

```nsl
# Function as argument
fn applyTwice(f, x) {
    return f(f(x))
}

fn increment(n) {
    return n + 1
}

print(applyTwice(increment, 5))  # 7

# Function composition
fn compose(f, g) {
    fn composed(x) {
        return f(g(x))
    }
    return composed
}

fn addOne(x) { return x + 1 }
fn timesTwo(x) { return x * 2 }

let addThenDouble = compose(timesTwo, addOne)
print(addThenDouble(5))  # (5+1)*2 = 12
```

---

## Arrays and Collections

### Creating Arrays

```nsl
let numbers = [1, 2, 3, 4, 5]
let mixed = [1, "hello", true, 3.14]
let empty = []
```

### Accessing Elements

```nsl
let arr = [10, 20, 30, 40, 50]
print(arr[0])  # 10 (first element)
print(arr[2])  # 30 (third element)
print(arr[4])  # 50 (last element)
```

### Modifying Arrays

```nsl
mut arr = [1, 2, 3]
arr[0] = 100
print(arr)  # [100, 2, 3]
```

### Array Operations

```nsl
let arr = [1, 2, 3, 4, 5]

print(len(arr))   # 5 (length)
print(sum(arr))   # 15 (sum of elements)
print(mean(arr))  # 3.0 (average)
```

### List Comprehensions

```nsl
# Squares of 1-5
let squares = [x * x for x in range(1, 6)]
# Result: [1, 4, 9, 16, 25]

# Filter even numbers
let evens = [x for x in range(1, 11) if x % 2 == 0]
# Result: [2, 4, 6, 8, 10]

# Transform values
let doubled = [x * 2 for x in range(1, 6)]
# Result: [2, 4, 6, 8, 10]
```

---

## Structs

### Defining Structs

```nsl
struct Point {
    x: number,
    y: number
}

struct Person {
    name: string,
    age: int
}
```

### Creating Struct Instances

```nsl
let p = Point { x: 10.5, y: 20.3 }
let person = Person { name: "Claude", age: 2 }
```

### Accessing Fields

```nsl
print(p.x)        # 10.5
print(p.y)        # 20.3
print(person.name)  # Claude
print(person.age)   # 2

# Arithmetic with fields
let sum = p.x + p.y
print(sum)  # 30.8
```

---

## Enums (Algebraic Data Types)

### Simple Enums

```nsl
enum Color {
    Red,
    Green,
    Blue
}

let c = Color::Red
```

### Enums with Data (Sum Types)

```nsl
enum Shape {
    Circle(number),           # radius
    Rectangle(number, number), # width, height
    Point                      # no data
}

let circle = Shape::Circle(5.0)
let rect = Shape::Rectangle(3.0, 4.0)
let point = Shape::Point
```

---

## Pattern Matching

### Basic Match

```nsl
fn describe_number(n: int) {
    match n {
        case 0 => "zero"
        case 1 => "one"
        case 2 => "two"
        case x => "other"  # Wildcard/binding
    }
}

print(describe_number(0))  # zero
print(describe_number(5))  # other
```

### Match with Guards

```nsl
fn classify(n: int) {
    match n {
        case x when x < 0 => "negative"
        case 0 => "zero"
        case x when x > 0 => "positive"
    }
}
```

### Match with Result/Option

```nsl
fn process(result) {
    match result {
        case ok(value) => {
            print("Success:", value)
        }
        case err(error) => {
            print("Error:", error)
        }
    }
}

fn check_option(opt) {
    match opt {
        case some(v) => print("Got value:", v)
        case none => print("No value")
    }
}
```

---

## Error Handling (Result/Option)

NSL has **built-in** Result and Option types with globally available helper functions. No imports needed - these are always available like `print()` or `len()`.

### Result Type

For operations that can fail:

```nsl
# Create success/error results
let success = ok(42.0)
let failure = err("Something went wrong")

# Check results
if (is_ok(success)) {
    let value = unwrap(success)
    print("Value:", value)
}

if (is_err(failure)) {
    print("Got error!")
}

# Unwrap with default value
let safe_value = unwrap_or(failure, 0)  # Returns 0 if err
```

### Option Type

For values that may or may not exist:

```nsl
# Create optional values
let present = some(10.0)
let absent = none()

# Check and use
if (is_some(present)) {
    print("Has value:", unwrap(present))
}

if (is_none(absent)) {
    print("No value")
}

# Unwrap with default
let value = unwrap_or(present, 0)  # Use 0 if none
```

### Result/Option Built-in Functions

| Function | Description |
|----------|-------------|
| `ok(value)` | Create success result |
| `err(message)` | Create error result |
| `is_ok(result)` | Check if result is ok |
| `is_err(result)` | Check if result is err |
| `some(value)` | Create optional with value |
| `none()` | Create empty optional |
| `is_some(opt)` | Check if optional has value |
| `is_none(opt)` | Check if optional is empty |
| `unwrap(result_or_opt)` | Get value (throws on err/none) |
| `unwrap_or(result_or_opt, default)` | Get value or default |

---

## Traits and Interfaces

### Defining Traits

```nsl
trait Printable {
    fn describe(self) -> string;
    fn to_string(self) -> string;
}
```

### Implementing Traits

```nsl
struct Point {
    x: number,
    y: number
}

impl Printable for Point {
    fn describe(self) {
        return "A point in 2D space"
    }

    fn to_string(self) {
        return "Point"
    }
}

let p = Point { x: 3.0, y: 4.0 }
let desc = Point_Printable_describe(p)
print(desc)  # A point in 2D space
```

---

## Modules and Imports

### Import Statements

```nsl
# Import entire module
import math

# Import with namespace
import math::linear_algebra

# Import specific items
import { sin, cos } from math

# Import all from module
import * from math::constants
```

### Module Definition

```nsl
module geometry {
    pub fn area(width, height) {
        return width * height
    }

    pub fn perimeter(width, height) {
        return 2 * (width + height)
    }
}
```

### Public Declarations

```nsl
# Public function
pub fn calculate(x) {
    return x * 2
}

# Public constant
pub let PI = 3.14159

# Public type
pub type Point = { x: number, y: number }
```

### Export Statements

```nsl
export { foo, bar }
export * from "submodule"
```

---

## Async/Await

### Async Functions

```nsl
async fn fetch_data(url) {
    # Simulate async operation
    return "data from " + url
}

async fn compute_value(x: number) {
    return x * 2 + 1
}
```

### Await Expression

```nsl
# Wait for async result
let result = await fetch_data("https://api.example.com")
print(result)

# Chain async calls
async fn process(n: number) {
    let value = await compute_value(n)
    return value + 10
}

let processed = await process(10)  # 31
```

### Short Arrow Async

```nsl
async fn square(n: number) => n * n

let sq = await square(4)  # 16
```

### Public Async Functions

Use `pub async fn` to export async functions from modules:

```nsl
module api {
    pub async fn fetch_data(url) {
        # Async operation here
        return "data from " + url
    }
}
```

---

## Consciousness Operators

NSL includes unique operators for neural-symbolic AI operations:

### Holographic Operator (◈) - Attention/Focus

Creates distributed representations using attention mechanisms:

```nsl
let data = 42.5
let encoded = ◈[data]
print("Holographic encoding:", encoded)

# Also available as ASCII alias
# let encoded = holo(data)
```

### Gradient Operator (∇) - Differentiation

Computes gradients for backpropagation:

```nsl
let loss = 0.5
let grad = ∇[loss]
print("Gradient:", grad)

# ASCII alias: grad(value)
```

### Tensor Product (⊗) - Composition/Binding

Creates combined representations, outer products:

```nsl
let a = [1.0, 2.0, 3.0]
let b = [4.0, 5.0]
let outer = ⊗[a, b]
print("Outer product:", outer)

# ASCII alias: outer(a, b)
```

### Quantum Branching (Ψ) - Superposition

Creates superposition states for exploring multiple hypotheses:

```nsl
let state = 1.5
let superposition = Ψ[state]
print("Superposition:", superposition)

# ASCII alias: psi(value)
```

### Extended Operators

| Symbol | ASCII Alias | Description |
|--------|-------------|-------------|
| `μ` | `mem` | Memory operator |
| `σ` | `introspect` | Self/Introspection |
| `↓` | - | Collapse/Measurement |
| `≈` | - | Similarity |
| `∫` | - | Temporal Integration |
| `±` | - | Uncertainty range |

> **Note:** Many ASCII aliases were removed to avoid conflicts with common function/variable names.
> Use the Unicode operators directly for consciousness features.

### Combining Operators

```nsl
# Gradient of holographic projection
let input = [1.0, 2.0, 3.0]
let holo_input = ◈[input]
let grad_holo = ∇[holo_input]
```

---

## AI-Centric Features

NSL is designed **for AI to code in**. These features exist in no other language and match how AI naturally thinks.

### Why AI-Centric?

| Human Thinking | AI Thinking | NSL Support |
|----------------|-------------|-------------|
| Step-by-step procedures | Parallel pattern matching | Native tensor operations |
| Binary true/false | Probabilistic confidence | `Uncertain` types |
| "Do X then Y" | "Achieve goal G" | Intent-based programming |
| Silent execution | Explain reasoning | `Explainable` values |
| Fixed algorithms | Adaptive optimization | Self-tuning algorithms |

### Intent-Based Programming

Express **what** you want, not **how** to do it:

```nsl
# Instead of writing sorting algorithm:
fn bubble_sort(arr) {
    for i in 0..len(arr) {
        for j in 0..len(arr)-i-1 {
            if arr[j] > arr[j+1] { swap(arr, j, j+1) }
        }
    }
}

# Express your intent:
let sorted = Intent.Achieve("sort efficiently", data)
    .Constraints(c => {
        c.TimeComplexity("O(n log n)")
        c.Stable(true)
        c.MaxMemory("100MB")
    })
    .Preferences(p => {
        p.Parallel(0.8)       # Prefer parallel
        p.CacheFriendly(0.9)  # Prefer cache-friendly
    })
    .Execute()

# System automatically selects optimal algorithm
print(sorted.Resolution)  # "Selected: ParallelMergeSort"
```

### Uncertainty Types

AI predictions naturally have confidence levels - NSL supports this:

```nsl
# Values with confidence
let prediction = Uncertain.Value(0.87, confidence: 0.92)
let threshold = Uncertain.Value(0.80, confidence: 0.99)

# Check confidence before acting
if prediction.IsConfident(0.9) {
    take_action(prediction.Value)
} else {
    gather_more_data()
}

# Uncertainty propagates through computations
let result = Uncertain.Propagate(a, b, fn(x, y) { x + y })
print(result.Confidence)  # Combined confidence

# Full probability distributions
let dist = Distribution.Normal(mean: 0.85, std: 0.05)
print(dist.Sample())      # Random sample
print(dist.Percentile(95))  # 95th percentile
```

### Self-Explaining Code

Every computation can carry its own explanation:

```nsl
# Wrap computation with explanation
let answer = Explainable.Compute("calculate tax", fn() {
    return income * 0.25 + deductions * 0.1
})
    .WithReasoning("Applied standard 25% rate with deduction credits")
    .WithEvidence("Based on 2024 tax code section 401")
    .Execute()

# Query explanation later
print(answer.Value)        # 42500
print(answer.Explanation)  # Full reasoning chain

# Chain explanations through pipeline
let result = data
    |> Explainable.Step("preprocess", normalize)
    |> Explainable.Step("feature_extract", extract_features)
    |> Explainable.Step("predict", model_predict)

print(result.ExplainAll())  # Shows each step's reasoning
```

### Semantic Contracts

Define what code **means**, not just what it does:

```nsl
# Define semantic contract for temperature
let temperature = Contract.For(reading)
    .Semantics("Physical temperature in Celsius")
    .Invariant(t => t >= -273.15, "Cannot be below absolute zero")
    .Invariant(t => t < 1e9, "Must be physically reasonable")
    .Unit("°C")

# Automatic validation
let valid = temperature.Create(22.5)    # OK
let invalid = temperature.Create(-300)  # Throws: Violates absolute zero

# Semantic compatibility checking
let fahrenheit = Contract.Define("temperature_fahrenheit")
let compatible = temperature.IsCompatibleWith(fahrenheit)  # false
```

### Adaptive Algorithms

Let the system learn which implementation is fastest:

```nsl
# Define multiple strategies
let searcher = Adaptive.Algorithm("search")
    .Add("linear", LinearSearch,
         suitability: data => data.size < 100 ? 0.9 : 0.3)
    .Add("binary", BinarySearch,
         suitability: data => data.sorted ? 0.9 : 0.1)
    .Add("hash", HashLookup,
         suitability: data => data.unique_keys ? 0.95 : 0.5)
    .AutoSelect()

# System learns from each execution
for batch in dataset {
    let result = searcher.Execute(batch, target)
    # Performance tracked automatically
}

# View learned preferences
print(searcher.Statistics())
# Output: "linear for n<100, hash for unique keys, binary for sorted data"
```

### Natural Language Bridge

Convert between code and natural language:

```nsl
# Describe what code does
let description = NaturalLanguage.Describe(my_function)
# Output: "Filters items where price exceeds 100, then sorts by date"

# Generate code template from description
let template = NaturalLanguage.GenerateTemplate(
    "filter users who are active and have verified email"
)
# Output: users.filter(u => u.active && u.email_verified)
```

### Formal Verification Hints

Add hints for proving correctness:

```nsl
[Requires("input != null")]
[Requires("input.length > 0")]
[Ensures("result >= 0")]
[Ensures("result < input.length")]
[Pure]  # No side effects
fn find_min_index(input) {
    mut min_idx = 0
    for i in 1..len(input) {
        [LoopInvariant("min_idx >= 0 && min_idx < i")]
        if input[i] < input[min_idx] {
            min_idx = i
        }
    }
    return min_idx
}

[Terminates]  # Proven to halt
[ComplexityBound("O(n log n)")]
fn merge_sort(arr) {
    # Implementation...
}

[ThreadSafe(Mechanism: "ImmutableData")]
fn process_concurrent(data) {
    # Safe for parallel execution
}
```

### Quick Reference: AI Coding Patterns

```nsl
# Pattern 1: Uncertain computation
let result = Uncertain.Propagate(
    input1.WithConfidence(0.9),
    input2.WithConfidence(0.8),
    fn(a, b) { complex_operation(a, b) }
)

# Pattern 2: Intent with constraints
let optimized = Intent.Achieve("minimize cost", data)
    .Constraints(c => {
        c.MaxMemory("1GB")
        c.MaxLatency("50ms")
    })
    .Execute()

# Pattern 3: Explainable pipeline
let pipeline = data
    |> Explainable.Step("clean", preprocess)
    |> Explainable.Step("transform", feature_extract)
    |> Explainable.Step("predict", model_predict)

# Pattern 4: Adaptive selection with learning
let processor = Adaptive.Algorithm("process")
    .Add("fast", FastProcessor)
    .Add("accurate", AccurateProcessor)
    .Add("balanced", BalancedProcessor)
    .LearnFromFeedback(true)
    .AutoSelect()
```

---

## Neural Network Operations

### Activation Functions

```nsl
fn relu(x) {
    if (x > 0) {
        return x
    }
    return 0
}

fn sigmoid(x) {
    if (x > 5) { return 1.0 }
    if (x < -5) { return 0.0 }
    return 0.5 + x * 0.1  # Linear approximation
}

fn step(x) {
    if (x > 0) { return 1 }
    return 0
}
```

### Simple Perceptron

```nsl
fn perceptron(x1, x2, w1, w2, bias) {
    let weighted_sum = x1 * w1 + x2 * w2 + bias
    return step(weighted_sum)
}

# AND gate
print(perceptron(0, 0, 1, 1, -1.5))  # 0
print(perceptron(1, 1, 1, 1, -1.5))  # 1

# OR gate
print(perceptron(0, 0, 1, 1, -0.5))  # 0
print(perceptron(1, 0, 1, 1, -0.5))  # 1
```

### Loss Functions

```nsl
fn mse(predicted, actual) {
    let diff = predicted - actual
    return diff * diff
}

fn gradient_step(current, gradient, learning_rate) {
    return current - learning_rate * gradient
}
```


---

## ProductionMath - Adaptive Operations

NSL includes ProductionMath, a system that **learns which mathematical operation works best** for your specific data patterns.

### Why ProductionMath?

Traditional programming: You decide `a + b` or `a * b` based on your knowledge.

ProductionMath: The system tries multiple interpretations, learns which works best, then uses that automatically.

### Two Phases

**1. Learning Phase (slow, ~290 ops/sec)**
```nsl
# System explores 16 different interpretations
# Sum, Ratio, Geometric, Harmonic, Power, etc.
let result = engine.diamond(a, b)
engine.update_policy(result, reward)
```

**2. Inference Phase (fast, ~20,000 ops/sec)**
```nsl
# Uses what it learned - no exploration
let result = engine.direct_compute(a, b)
```

### Semantic Modes

| Mode | Meaning | Example Use |
|------|---------|-------------|
| Sum | Combine | Total quantities |
| Replicate | Scale | Areas, volumes |
| Ratio | Compare | Rates, proportions |
| Geometric | Balance | Growth rates |
| Harmonic | Average rates | Parallel systems |
| CoPresence | Limit | Bottleneck detection |

### Example: Learning Best Operation

```nsl
import gpu

# Create engine
let accel = gpu.GpuAccelerator()
let engine = gpu.ProductionMathEngine(accel)

# Training data with different patterns
let data_pairs = [
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ([10.0, 20.0], [2.0, 4.0]),
]

# Learning phase (run for a while)
for pair in data_pairs {
    let (a, b) = pair
    let result = engine.diamond(a, b)

    # Reward based on your criteria
    let reward = evaluate_result(result)
    engine.update_policy(result, reward)
}

# After learning, use fast mode
let fast_result = engine.direct_compute(new_a, new_b)
```

### When to Use ProductionMath

| Use ProductionMath When | Use Standard Math When |
|------------------------|----------------------|
| Optimal operation unknown | You know exactly what math you need |
| Data patterns vary | Consistent data patterns |
| Want adaptive behavior | Want predictable behavior |
| Can afford learning time | Need immediate results |

### Important Notes

- ProductionMath is **optional** - regular GPU ops run at full speed
- Learning mode is slow (~800% overhead) because it tries 16 interpretations
- After learning: nearly as fast as standard operations
- State can be saved and loaded for reuse


---

## AI-Native Command Line Interface

NSL is designed for AI, not humans. The CLI includes powerful flags for AI/MCP integration.

### Basic Execution

```bash
nsl                           # Start interactive REPL
nsl script.nsl                # Run NSL script
nsl --eval "2 + 2"           # Execute code directly
nsl --version                 # Show version
```

### AI-Native Execution Flags

```bash
# Execute with JSON output (machine-readable)
nsl --eval "tensor |> normalize" --json

# Execute with thinking/reasoning trace
nsl --eval "complex_computation()" --think --json

# Full execution trace with timestamps
nsl --eval "gpu.matmul(a, b)" --trace

# Auto-initialize GPU
nsl --eval "gpu.tensor([1,2,3])" --gpu

# Set execution timeout (milliseconds)
nsl --eval "long_computation()" --timeout 60000
```

### AI Introspection & Self-Awareness

```bash
# NSL self-awareness report
nsl --introspect --json

# List all capabilities (for AI discovery)
nsl --capabilities --json

# Generate explanation of code
nsl --eval "data |> transform" --explain

# Generate self-reflection on execution
nsl --eval "process()" --reflect
```

### Persistent AI Memory

```bash
# Pass context object as variables
nsl --eval "x + y" --context '{"x": 10, "y": 20}'

# Use persistent memory file
nsl --eval "learn_pattern(data)" --memory ai_memory.json

# Learn from execution (updates memory file)
nsl --eval "analyze(input)" --memory mem.json --learn
```

### Code Analysis & Transformation

```bash
# Show Abstract Syntax Tree
nsl --ast "x + y * z" --json

# Transform code
nsl --transform vectorize "for i in range(100) { process(i) }"
nsl --transform optimize "x + 0 + y * 1"
nsl --transform minify "fn foo() {\n  return 42\n}"
nsl --transform prettify "if(x){y=1}"

# Run performance benchmark
nsl --benchmark --json
```

### AI Code Optimization

```bash
# Auto-optimize code before execution
nsl --eval "compute()" --optimize

# Auto-vectorize to GPU
nsl --eval "for i in data { process(i) }" --vectorize

# Run in sandboxed mode
nsl --eval "untrusted_code()" --sandbox
```

### Example: Full AI Execution

```bash
# Complete AI-native execution with all metadata
nsl --eval "data |> normalize |> transform |> output" \
    --gpu \
    --think \
    --reflect \
    --explain \
    --memory session.json \
    --learn \
    --json
```

Output:
```json
{
  "Success": true,
  "Result": [...],
  "ResultType": "List",
  "ExecutionTimeMs": 45,
  "Explanation": "Pipes data through normalization, transformation, and output.",
  "Reflection": "Executed 1 lines in 45ms. Used consciousness operators.",
  "Trace": [
    {"Timestamp": 0, "Phase": "INIT", "Description": "Starting execution"},
    {"Timestamp": 1, "Phase": "GPU", "Description": "Initializing GPU context"},
    {"Timestamp": 15, "Phase": "PARSE", "Description": "Tokenizing code"},
    {"Timestamp": 18, "Phase": "EXECUTE", "Description": "Executing AST"},
    {"Timestamp": 45, "Phase": "COMPLETE", "Description": "Execution completed"}
  ],
  "Metrics": {
    "tokens": 12,
    "executionTimeMs": 45,
    "memoryUsedBytes": 135032
  }
}
```

---

## MCP Server Integration (Claude Code)

NSL includes an MCP (Model Context Protocol) server for integration with Claude Code and other AI assistants.

### Installation

```bash
cd E:\NSL.Interpreter\mcp-server
npm install
```

### Configuration

Add to your Claude Code settings (`~/.claude/settings.local.json`):

```json
{
  "mcpServers": {
    "nsl": {
      "command": "node",
      "args": ["E:\\NSL.Interpreter\\mcp-server\\index.js"],
      "env": {}
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `nsl` | Execute NSL code with full AI options |
| `nsl_think` | Execute with reasoning trace |
| `nsl_gpu` | Execute with GPU acceleration |
| `nsl_consciousness` | Execute consciousness operators |
| `nsl_learn` | Execute and learn from execution |
| `nsl_introspect` | Get NSL self-awareness report |
| `nsl_capabilities` | List all NSL capabilities |
| `nsl_benchmark` | Run performance benchmark |
| `nsl_ast` | Parse and return AST |
| `nsl_transform` | Transform code |
| `nsl_file` | Execute NSL script file |
| `nsl_read` | Read file using NSL |
| `nsl_write` | Write file using NSL |

### Using from Claude Code

Once configured, Claude Code can use NSL as an execution environment:

```
User: Calculate 2 + 2 using NSL
Claude: [Uses nsl tool with code="2 + 2"]
Result: {"Success": true, "Result": 4, ...}
```

NSL becomes an alternative to Bash for AI-native computation.


---

## Package Management (nslpm)

NSL has a full-featured package manager called `nslpm` (v1.2.0) - similar to npm/pip but designed for NSL.

```bash
nslpm version
# Output:
# nslpm 1.2.0
# NSL Package Manager - Like pip but for NSL
# Using local file-based registry (works offline)
```

### Initialize a Project

```bash
nslpm init my-package 1.0.0
```

**Output:**
```
✓ Initialized package: my-package@1.0.0
Created nsl-package.json
```

Creates `nsl-package.json`:
```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "An NSL package",
  "authors": [],
  "keywords": [],
  "main": "main.nsl",
  "license": "MIT",
  "dependencies": {},
  "devDependencies": {},
  "peerDependencies": {},
  "scripts": {},
  "files": ["**/*.nsl"],
  "private": false,
  "metadata": {},
  "exports": {}
}
```

### Complete Package Example

Here's a complete package structure:

```
my-package/
├── nsl-package.json
├── main.nsl
├── test.nsl
├── nsl_packages/      # Installed dependencies
└── .venv/             # Virtual environment (optional)
```

**nsl-package.json:**
```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "A sample NSL package",
  "authors": ["Your Name"],
  "keywords": ["math", "utils"],
  "main": "main.nsl",
  "license": "MIT",
  "repository": "https://github.com/user/my-package",
  "dependencies": {
    "other-package": "^1.0.0"
  },
  "devDependencies": {},
  "scripts": {
    "test": "nsl test.nsl",
    "build": "nsl build.nsl",
    "start": "nsl main.nsl"
  },
  "files": ["**/*.nsl"],
  "exports": {
    ".": "./main.nsl"
  }
}
```

**main.nsl:**
```nsl
# Main entry point

pub fn hello(name) {
    return "Hello, " + name + " from my-package!"
}

pub fn add(a, b) {
    return a + b
}

print("my-package loaded!")
```

### Validate Package

```bash
nslpm validate
# or
nslpm check
```

**Output:**
```
✓ Package manifest is valid!
```

### Install Packages

```bash
# Install a specific package
nslpm install lodash
nslpm i lodash@^1.0.0

# Install as dev dependency
nslpm install --save-dev test-framework
nslpm install -D test-framework

# Install all dependencies from manifest
nslpm install

# Install from requirements file (pip-style)
nslpm install -r requirements.nsl

# Install in editable/dev mode (creates symlink)
nslpm install -e .
nslpm install --editable ../my-other-package

# Dry-run mode - preview without installing
nslpm install --dry-run axios
nslpm install -n axios

# Upgrade to latest version
nslpm install -U lodash
nslpm install --upgrade lodash

# Quiet mode - suppress output
nslpm install -q lodash

# Force reinstall
nslpm install -f lodash
nslpm install --force-reinstall lodash
```

**Output:**
```
Installing dependencies...
  ↓ Installing lodash
  ✓ Installed lodash@1.2.0
✓ Installed 1 packages
```

**Dry-run output:**
```
Dry-run mode - would install:
  lodash@^1.0.0
  axios

No changes made (dry-run)
```

### List Installed Packages

```bash
nslpm list
# or
nslpm ls
```

**Output:**
```
Installed packages (2):

  lodash@1.2.0
  utils@0.5.0
```

Use `-v` for verbose output with descriptions:
```bash
nslpm list --verbose
```

### Dependency Tree

```bash
nslpm tree
```

**Output:**
```
my-package@1.0.0
├── lodash@1.2.0
└── utils@0.5.0
    └── helpers@0.1.0
```

### Search & Info

```bash
# Search for packages
nslpm search http client

# Show package details
nslpm info lodash
nslpm show lodash
```

### Update Packages

```bash
# Check for outdated packages
nslpm outdated

# Update all packages
nslpm update

# Update specific package
nslpm update lodash
```

### Remove Packages

```bash
nslpm uninstall lodash
nslpm remove lodash
nslpm rm lodash
```

### Virtual Environments

NSL supports Python-style virtual environments for isolated package installations:

```bash
# Create virtual environment
nslpm venv create .venv my-project-env
```

**Output:**
```
✓ Created virtual environment: my-project-env
  Path: /path/to/project/.venv

To activate:
  PowerShell: .\.venv\bin\activate.ps1
  CMD:        .\.venv\bin\activate.bat
```

```bash
# List virtual environments
nslpm venv list
```

**Output:**
```
Virtual environments:

  my-project-env
    /path/to/project/.venv
```

```bash
# Other venv commands
nslpm venv info         # Show active environment info
nslpm venv activate     # Show activation command
nslpm venv deactivate   # Show deactivation command
nslpm venv delete .venv # Delete environment
```

### Scripts

Define custom scripts in `nsl-package.json`:

```json
{
  "scripts": {
    "test": "nsl test.nsl",
    "build": "nsl build.nsl",
    "start": "nsl main.nsl",
    "dev": "nsl main.nsl --watch"
  }
}
```

```bash
# List available scripts
nslpm scripts
```

**Output:**
```
Available scripts:

  build
  dev
  start
  test

Run a script with: nslpm run <script-name>
```

```bash
# Run a script
nslpm run test
nslpm run start
```

### Pack & Publish

```bash
# Create package archive (.nslpkg)
nslpm pack
```

**Output:**
```
  • Packed my-package@1.0.0
✓ Created: /path/to/my-package-1.0.0.nslpkg
```

```bash
# Publish to local registry (no auth required!)
nslpm publish
```

**Output:**
```
Publishing to local registry...
  • Packed my-package@1.0.0
  ✓ Published my-package@1.0.0
✓ Package published to local registry!
```

> **Note:** NSL uses a local file-based registry stored in `~/.nsl/registry/`.
> No remote server required - works completely offline!

### Freeze - Export Installed Packages

```bash
# Output installed packages in requirements format (like pip freeze)
nslpm freeze

# Save to file
nslpm freeze > requirements.nsl
nslpm freeze requirements.nsl
```

**Output:**
```
http-client==1.0.0
json-parser==2.0.0
lodash==1.2.0
utils==0.5.0
```

### Requirements File Format

Create a `requirements.nsl` file (pip-style):

```
# Comments start with #
http-client==1.0.0
json-parser>=2.0.0
utils  # any version

# Version specifiers supported:
# ==  exact version
# >=  minimum version
# ~=  compatible version (converts to ~)
```

### Cache Management

```bash
# Show cache info
nslpm cache info
# or just
nslpm cache
```

**Output:**
```
Cache directory: C:\Users\you\AppData\Local\nsl\cache
Cached packages: 5
Total size: 2.3 MB
```

```bash
# List cached packages with sizes
nslpm cache list
nslpm cache ls
```

**Output:**
```
Cache directory: C:\Users\you\AppData\Local\nsl\cache

  lodash-1.2.0.nslpkg  (45.2 KB)
  utils-0.5.0.nslpkg  (12.8 KB)

Total: 2 packages, 58 KB
```

```bash
# Show cache directory path (useful for scripts)
nslpm cache dir
```

```bash
# Remove specific cached packages
nslpm cache rm lodash
nslpm cache remove utils
```

```bash
# Clear all cache
nslpm cache clean
nslpm cache clear
nslpm cache purge
```

### Configuration Management

```bash
# List all config values
nslpm config list
```

**Output:**
```
Configuration (~/.nsl/config.json):

  cache.dir = /custom/cache/path
  registry.local = true
```

```bash
# Get a specific config value
nslpm config get cache.dir

# Set a config value
nslpm config set cache.dir /custom/cache

# Remove a config value
nslpm config unset cache.dir

# Open config in editor
nslpm config edit

# Show config file path
nslpm config path
```

**Available Config Keys:**

| Key | Description |
|-----|-------------|
| `cache.dir` | Custom cache directory path |
| `registry.url` | Package registry URL |
| `registry.local` | Use local registry (true/false) |
| `install.timeout` | Download timeout in seconds |
| `install.retries` | Number of retry attempts |

### Hash - Compute File Hash

```bash
# Compute SHA256 hash (default)
nslpm hash package.nslpkg
```

**Output:**
```
--hash=sha256:a1b2c3d4e5f6...
```

```bash
# Compute SHA512 hash
nslpm hash package.nslpkg --sha512
nslpm hash package.nslpkg --algorithm sha512
```

### Debug - Show Debugging Info

```bash
nslpm debug
```

**Output:**
```
nslpm Debug Information

==================================================

nslpm version: 1.2.0
.NET version: 8.0.0
OS: Microsoft Windows 10.0.22631
Platform: 64-bit

Paths:
  Config:   C:\Users\you\.nsl\config.json
  Cache:    C:\Users\you\AppData\Local\nsl\cache
  Registry: C:\Users\you\.nsl\registry
  Packages: C:\Users\you\AppData\Local\nsl\packages

Current directory: E:\my-project
Has manifest: true
Project: my-project@1.0.0

Registry:
  Local registry exists: true
  Packages in registry: 12

Cache:
  Cached packages: 5
  Total size: 2.3 MB
```

### Command Reference

| Command | Aliases | Description |
|---------|---------|-------------|
| `init [name] [version]` | - | Initialize new package |
| `install [pkg...]` | `i`, `add` | Install packages |
| `uninstall <pkg...>` | `remove`, `rm` | Remove packages |
| `update [pkg...]` | `upgrade` | Update packages |
| `list` | `ls` | List installed packages |
| `freeze [file]` | - | Output packages in requirements format |
| `search <query>` | `s` | Search for packages |
| `info <pkg>` | `show` | Show package details |
| `outdated` | - | Check for outdated packages |
| `tree` | - | Show dependency tree |
| `validate` | `check` | Validate package manifest |
| `pack [output-dir]` | - | Create package archive |
| `publish` | - | Publish to registry |
| `run <script>` | - | Run a script |
| `exec <cmd>` | `x` | Execute command in context |
| `scripts` | - | List available scripts |
| `venv <cmd>` | - | Manage virtual environments |
| `cache <cmd>` | - | Manage package cache |
| `config <cmd>` | - | Manage configuration |
| `hash <file>` | - | Compute file hash |
| `debug` | - | Show debugging info |
| `version` | `--version` | Show version |

### Install Options

| Option | Description |
|--------|-------------|
| `-D`, `--save-dev` | Save as dev dependency |
| `-n`, `--dry-run` | Preview without installing |
| `-r`, `--requirements <file>` | Install from requirements file |
| `-e`, `--editable <path>` | Install in editable/dev mode |
| `-U`, `--upgrade` | Upgrade to latest version |
| `-q`, `--quiet` | Suppress output |
| `-f`, `--force-reinstall` | Force reinstall |

### Global Options

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Verbose output |
| `-h`, `--help` | Show help |

### Registry

NSL uses a **local file-based registry** that works completely offline:

| Path | Description |
|------|-------------|
| `~/.nsl/registry/` | Package storage directory |
| `~/.nsl/registry/index.json` | Package index |
| `~/.nsl/registry/packages/` | Package files (.nslpkg) |

No authentication required - share packages by copying files!

### Available Standard Packages

NSL includes these standard packages ready to install:

| Package | Version | Description |
|---------|---------|-------------|
| `nsl-collections` | 1.0.1 | Functional utilities: `map`, `filter`, `reduce`, `zip`, `flatten`, `unique`, `chunk`, `partition` |
| `nsl-math` | 1.2.0 | Extended math: `sqrt`, `sin`, `cos`, `exp`, `log`, `mean`, `std_dev`, `gcd`, `factorial`, `dot` |
| `nsl-string` | 1.0.1 | String utils: `capitalize`, `pad_start`, `truncate`, `split_words`, `format`, `is_numeric` |
| `nsl-json` | 1.1.0 | JSON utilities: `parse`, `stringify`, `get_path`, `merge`, `deep_merge`, `pick`, `omit` |
| `nsl-http` | 1.1.0 | HTTP client: `get`, `post`, `put`, `delete`, `get_json`, `build_url`, `auth_header` |
| `nsl-ml` | 1.1.0 | Machine learning: `relu`, `sigmoid`, `softmax`, `mse`, `adam_step`, `normalize`, `one_hot`, `accuracy` |
| `nsl-testing` | 1.0.0 | Testing framework: `describe`, `it`, `assert_eq`, `expect`, `assert_close`, `summary` |
| `nsl-datetime` | 1.2.0 | Date/time: `now`, `create`, `format`, `add_days`, `diff_days`, `is_weekend`, `relative_time` |
| `nsl-regex` | 1.0.0 | Pattern matching: `search`, `match_start`, `find_all`, `replace_with`, `validate_email`, `PATTERNS` |
| `nsl-random` | 1.0.0 | Random generation: `generator`, `shuffle`, `sample`, `choice`, `normal`, `uuid`, `random_string` |
| `nsl-csv` | 1.0.1 | CSV parsing: `parse`, `parse_with_headers`, `stringify`, `filter_rows`, `sort_by_column`, `validate` |

#### Example: Using nsl-collections

```bash
nslpm install nsl-collections
```

```nsl
import { map, filter, reduce, zip } from nsl-collections

let numbers = [1, 2, 3, 4, 5]

# Map - transform each element
let doubled = map(numbers, fn(x) { x * 2 })
# Result: [2, 4, 6, 8, 10]

# Filter - keep matching elements
let evens = filter(numbers, fn(x) { x % 2 == 0 })
# Result: [2, 4]

# Reduce - fold to single value
let total = reduce(numbers, fn(acc, x) { acc + x }, 0)
# Result: 15

# Zip - combine two arrays
let letters = ["a", "b", "c"]
let pairs = zip(numbers, letters)
# Result: [[1, "a"], [2, "b"], [3, "c"]]
```

#### Example: Using nsl-ml

```bash
nslpm install nsl-ml
```

```nsl
import { relu, sigmoid, softmax, mse, normalize, one_hot } from nsl-ml

# Activation functions
let activated = relu(-2.5)      # 0
let prob = sigmoid(0)           # 0.5
let probs = softmax([1, 2, 3])  # [0.09, 0.24, 0.67]

# Loss function
let loss = mse([0.9, 0.1], [1.0, 0.0])

# Data processing
let normalized = normalize([1, 2, 3, 4, 5])  # [0, 0.25, 0.5, 0.75, 1]
let encoded = one_hot(2, 5)                   # [0, 0, 1, 0, 0]
```

#### Example: Using nsl-testing

```bash
nslpm install nsl-testing
```

```nsl
import { describe, it, assert_eq, assert_close, summary } from nsl-testing

describe("Math operations", fn() {
    it("should add numbers correctly", fn() {
        assert_eq(2 + 2, 4)
    })

    it("should handle floating point", fn() {
        assert_close(0.1 + 0.2, 0.3, 0.0001)
    })
})

describe("Array operations", fn() {
    it("should have correct length", fn() {
        let arr = [1, 2, 3]
        assert_eq(len(arr), 3)
    })
})

# Print test summary
summary()
```

#### Example: Using nsl-regex

```bash
nslpm install nsl-regex
```

```nsl
import { search, find_all, replace_with, validate_email, PATTERNS } from nsl-regex

# Search returns Option<Match> - some() or none()
let result = search("Hello World", "World")
match result {
    case some(m) => print("Found:", m.matched, "at", m.start)
    case none => print("Not found")
}

# Find all matches
let numbers = find_all("a1b2c3", "\\d")
# Result: [{ matched: "1", ... }, { matched: "2", ... }, { matched: "3", ... }]

# Replace with function
let upper = replace_with("hello world", "\\w+", fn(m) { upper(m.matched) })
# Result: "HELLO WORLD"

# Validate email - returns Result<string>
match validate_email("user@example.com") {
    case ok(email) => print("Valid:", email)
    case err(msg) => print("Invalid:", msg)
}

# Use pre-built patterns
let has_email = test(text, PATTERNS.email)
```

#### Example: Using nsl-random

```bash
nslpm install nsl-random
```

```nsl
import { generator, shuffle, sample, choice, normal, uuid_simple } from nsl-random

# Create seeded generator for reproducibility
let gen = generator(42)

# Shuffle returns { value: shuffled_array, gen: updated_generator }
let result = shuffle(gen, [1, 2, 3, 4, 5])
print(result.value)  # Shuffled array

# Sample n items (without replacement)
let picked = sample(result.gen, ["a", "b", "c", "d"], 2)
print(picked.value)  # e.g., ["c", "a"]

# Normal distribution
let norm = normal(gen, 0, 1)  # mean=0, std=1
print(norm.value)  # Random value from normal distribution

# Generate UUID
let id = uuid_simple()
print(id)  # e.g., "a1b2c3d4-e5f6-4789-abcd-ef0123456789"
```

#### Example: Using nsl-csv

```bash
nslpm install nsl-csv
```

```nsl
import { parse, parse_with_headers, stringify, filter_rows, sort_by_field } from nsl-csv

# Parse CSV - returns Result<Array>
let csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA"
match parse_with_headers(csv_data) {
    case ok(rows) => {
        # rows = [{ name: "Alice", age: "30", city: "NYC" }, ...]

        # Filter rows
        let adults = filter_rows(rows, fn(r) { int(r.age) >= 21 })

        # Sort by field
        let sorted = sort_by_field(adults, "name")

        print(sorted)
    }
    case err(msg) => print("Parse error:", msg)
}

# Generate CSV from data
let data = [["name", "score"], ["Alice", "95"], ["Bob", "87"]]
let csv_string = stringify(data)
# Result: "name,score\nAlice,95\nBob,87"
```

---

## Built-in Namespaces ⭐ NEW

NSL includes powerful built-in namespaces that require no imports.

### Git Namespace

```nsl
# Check repository status
let status = git.status()
print("Clean:", status.clean)
print("Changed files:", status.count)

# Branch operations
print("Current branch:", git.branch())
let branches = git.branches()

# Commit history
let commits = git.log(5)  # Last 5 commits
for c in commits {
    print(c.hash, c.message)
}

# Other functions
git.diff()              # Show uncommitted changes
git.show("HEAD")        # Show commit details
git.root()              # Repository root path
git.remote()            # Remote URLs
git.isRepo()            # Check if in git repo
```

### Shell Pipelines ⭐ NEW

NSL can chain shell commands like bash pipes:

```nsl
# Chain commands - stdout flows to stdin of next command
# Like: dir /b | find /c /v ""
let count = sys.pipe("dir /b", "find /c /v \"\"")

# Like: cat file.txt | grep pattern | sort
let result = sys.pipe("type data.txt", "findstr error", "sort")

# Run command with stdin input
let data = "cherry\napple\nbanana"
let sorted = sys.run("sort", data)
print(sorted.stdout)  # apple, banana, cherry

# sys.run returns {stdout, stderr, code, success}
let result = sys.run("findstr pattern", fileContent)
if result.success {
    print(result.stdout)
}
```

### Process Namespace

```nsl
# List processes
let procs = proc.list("node")  # Filter by name
for p in procs {
    print(p.pid, p.name)
}

# Process control
proc.kill(12345)        # Kill by PID
proc.exists(12345)      # Check if running
proc.info(12345)        # Get details
```

### Network Namespace

```nsl
# Connectivity
print("Online:", net.isOnline())
print("Local IP:", net.localIp())

# Diagnostics
let ping = net.ping("google.com")
print("Response:", ping.time, "ms")

net.lookup("google.com")  # DNS lookup
net.ports()               # List open ports
```

### Environment Namespace

```nsl
# Variables
env.get("PATH")
env.set("MY_VAR", "value")
env.all()                # All variables
env.expand("%HOME%")     # Expand variables

# System info
env.home()               # Home directory
env.user()               # Username
env.os()                 # OS name
env.arch()               # x64/x86
```

### Clipboard Namespace

```nsl
clip.copy("Hello!")
let text = clip.paste()
```

### XML Namespace

```nsl
let xmlStr = "<root><item>Test</item></root>"
let doc = xml.parse(xmlStr)
let items = xml.query(xmlStr, "//item")
```

### ZIP Namespace

```nsl
zip.create("./folder", "archive.zip")
zip.extract("archive.zip", "./output")
let files = zip.list("archive.zip")
zip.add("archive.zip", "file.txt")
```

### YAML Namespace

```nsl
let config = yaml.parse(yamlString)
let output = yaml.stringify(config)
```

### Diff Namespace

```nsl
let changes = diff.lines(oldText, newText)
let fileDiff = diff.files("a.txt", "b.txt")
let patched = diff.patch(content, patchStr)
```

### Template Namespace

```nsl
let tmpl = "Hello ${name}!"
let result = template.render(tmpl, {name: "World"})
# "Hello World!"
```

---

## Standard Library

### Math Module

```nsl
import { PI, E, sin, cos, sqrt, abs } from math

# Constants
print(PI)     # 3.141592653589793
print(E)      # 2.718281828459045

# Functions available:
# sin, cos, tan, cot, sec, csc
# sinh, cosh, tanh
# sqrt, pow, abs, exp, log
# log2, log10
# floor, ceil, round, trunc
# min, max, clamp
# lerp (linear interpolation)
# radians, degrees
# square, cube
# factorial, gcd, lcm
# is_integer, is_even, is_odd
```

### Collections Module

```nsl
import { map, filter, reduce } from collections

# map: Apply function to each element
let doubled = map([1, 2, 3], fn(x) { x * 2 })

# filter: Keep elements matching predicate
let evens = filter([1, 2, 3, 4], fn(x) { x % 2 == 0 })

# reduce: Fold to single value
let sum = reduce([1, 2, 3, 4], fn(acc, x) { acc + x }, 0)

# Other functions:
# find, any, all
# first, last
# take, skip
# reverse, zip
# sum_all, average
# min_val, max_val
# count, flatten
```

### String Module

```nsl
import { concat, length, repeat, join } from string

let s = "hello"
print(length(s))      # 5
print(repeat("ab", 3))  # ababab
print(join(["a", "b", "c"], "-"))  # a-b-c

# Other functions:
# is_empty, contains
# starts_with, ends_with
# to_string, format
```

### I/O Module

```nsl
import { print, println } from io

print("Hello")     # No newline
println("World")   # With newline
print("x =", 42)   # Multiple args
```

### Semantic File Access (AI-Optimized) ⭐ NEW

NSL provides semantic file access functions designed for AI agents to efficiently read large files without exhausting token limits.

#### Structure-First Reading

```nsl
# Get file overview without reading full content
let overview = read_file("large_doc.md", "structure")
# Returns: {
#   path: "large_doc.md",
#   total_lines: 3863,
#   total_chars: 120588,
#   estimated_tokens: 30147,
#   headings: [{level: 1, text: "Title", line: 1}, ...],
#   code_blocks: 93,
#   table_rows: 45
# }

# Read specific section by heading name
let gpu_section = read_file("README.md", "section", "GPU Acceleration")
# Returns only that section's content

# Read specific line range
let lines = read_file("code.nsl", "lines", 100, 50)
# Returns lines 100-150
```

#### Attention-Based Reading (◈_read)

```nsl
# Find sections most relevant to your query
let relevant = attention_read("README.md", "consciousness operators GPU")
# Returns: {
#   query: "consciousness operators GPU",
#   total_sections: 503,
#   returned_sections: 5,
#   sections: [
#     {heading: "GPU Acceleration", score: 1.91, content: "..."},
#     {heading: "Consciousness Operators", score: 1.82, content: "..."},
#     ...
#   ]
# }

# Unicode alias
let relevant = ◈_read("README.md", "neural network training")
```

#### Pattern Extraction

```nsl
# Extract all code blocks
let code = extract("README.md", "code")
# Returns: [{language: "nsl", content: "...", line_start: 42}, ...]

# Extract only NSL code blocks
let nsl_code = extract("README.md", "nsl")

# Extract markdown tables
let tables = extract("doc.md", "tables")

# Extract all headings
let headings = extract("doc.md", "headings")
# Returns: [{level: 1, text: "Title", line: 1}, ...]

# Extract markdown links
let links = extract("doc.md", "links")
# Returns: [{text: "Click here", url: "https://...", line: 5}, ...]

# Extract function definitions
let functions = extract("code.nsl", "functions")
# Returns: [{name: "myFunc", params: "x, y", line: 10}, ...]

# Custom regex pattern
let matches = extract("log.txt", "ERROR.*")
```

#### Streaming Large Files

```nsl
# Read file in chunks (for processing large files)
let chunks = stream_file("huge_file.txt", 2000)  # ~2000 tokens per chunk

for chunk in chunks {
    print("Processing chunk " + chunk["index"])
    let content = chunk["content"]
    # Process each chunk...

    if chunk["is_last"] {
        print("Done!")
    }
}
```

#### Why This Matters for AI

| Old Way (Fails) | New Way (Works) |
|-----------------|-----------------|
| `read_file("big.md")` → 30000 tokens! | `read_file("big.md", "structure")` → ~100 tokens |
| Token limit exceeded | Get overview, drill down as needed |
| Can't read large files | `attention_read("big.md", "what I need")` |
| All or nothing | Semantic access to relevant parts |

### GPU Acceleration ⭐ NEW

NSL provides direct GPU acceleration through the `gpu` namespace, powered by ILGPU with automatic hardware detection.

#### Initializing the GPU

```nsl
# Initialize GPU and get info
let info = gpu.init()
print("GPU:", info["name"])
print("VRAM:", info["vram_mb"], "MB")
print("Backend:", info["backend"])
print("Architecture:", info["architecture"])
print("Tensor Cores:", info["tensor_cores"])

# List all available GPUs
let devices = gpu.devices()
for device in devices {
    print(device["name"], "-", device["vram_mb"], "MB")
}
```

#### Creating GPU Tensors

```nsl
# Create tensor from data
let a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
let b = gpu.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])

# Create special tensors
let zeros = gpu.zeros([3, 3])      # All zeros
let ones = gpu.ones([4, 4])        # All ones
let rand = gpu.random([2, 2])      # Random [0, 1)

# Get tensor info
print("Shape:", gpu.shape(a))      # [2, 2]
print("Size:", gpu.size(a))        # 4
```

#### Matrix Operations

```nsl
# Matrix multiplication (the core GPU operation)
let c = gpu.matmul(a, b)

# Element-wise operations
let sum = gpu.add(a, b)
let diff = gpu.sub(a, b)
let prod = gpu.mul(a, b)
let quot = gpu.div(a, b)

# Transpose
let t = gpu.transpose(a)
```

#### Activation Functions

```nsl
let x = gpu.random([100, 100])

# Neural network activations (all run on GPU)
let relu_out = gpu.relu(x)
let sigmoid_out = gpu.sigmoid(x)
let tanh_out = gpu.tanh(x)
let softmax_out = gpu.softmax(x)
```

#### Math Operations

```nsl
# Element-wise math (runs on GPU)
let exp_x = gpu.exp(x)
let log_x = gpu.log(x)
let sqrt_x = gpu.sqrt(x)
let pow_x = gpu.pow(x, 2.0)
```

#### Getting Results Back to CPU

```nsl
# Get tensor data as NSL array
let result = gpu.to_cpu(c)
print("Result:", result)

# Clean up GPU memory when done
gpu.dispose(a)
gpu.dispose(b)
gpu.dispose(c)

# Shutdown GPU completely (optional, frees all resources)
gpu.shutdown()
```

#### Complete Example: Matrix Multiplication

```nsl
# Initialize GPU
let info = gpu.init()
print("Using:", info["name"])

# Create two random matrices
let a = gpu.random([512, 512])
let b = gpu.random([512, 512])

# Matrix multiply on GPU (way faster than CPU for large matrices)
let c = gpu.matmul(a, b)

# Apply ReLU activation
let activated = gpu.relu(c)

# Get small sample of results
let sample = gpu.to_cpu(activated)
print("First few values:", slice(sample, 0, 5))

# Cleanup
gpu.dispose(a)
gpu.dispose(b)
gpu.dispose(c)
gpu.dispose(activated)
```

#### GPU API Reference

| Function | Description |
|----------|-------------|
| `gpu.init()` | Initialize GPU, returns info dict |
| `gpu.devices()` | List all available GPUs |
| `gpu.tensor(data, shape)` | Create tensor from array |
| `gpu.zeros(shape)` | Create tensor of zeros |
| `gpu.ones(shape)` | Create tensor of ones |
| `gpu.random(shape)` | Create random tensor [0,1) |
| `gpu.matmul(a, b)` | Matrix multiplication |
| `gpu.add(a, b)` | Element-wise addition |
| `gpu.sub(a, b)` | Element-wise subtraction |
| `gpu.mul(a, b)` | Element-wise multiplication |
| `gpu.div(a, b)` | Element-wise division |
| `gpu.relu(x)` | ReLU activation |
| `gpu.sigmoid(x)` | Sigmoid activation |
| `gpu.tanh(x)` | Tanh activation |
| `gpu.softmax(x)` | Softmax activation |
| `gpu.exp(x)` | Element-wise exp |
| `gpu.log(x)` | Element-wise log |
| `gpu.sqrt(x)` | Element-wise sqrt |
| `gpu.pow(x, exp)` | Element-wise power |
| `gpu.transpose(x)` | Transpose matrix |
| `gpu.to_cpu(tensor)` | Get data as NSL array |
| `gpu.shape(tensor)` | Get tensor shape |
| `gpu.size(tensor)` | Get element count |
| `gpu.dispose(tensor)` | Free GPU memory |
| `gpu.shutdown()` | Shutdown GPU |

### Graph Compilation & JIT (PyTorch-Competitive) ⭐ NEW

NSL now includes a graph compiler and JIT system for PyTorch-competitive performance:

#### Lazy Tensors & Graph Compilation

```nsl
# Create lazy tensors - operations are recorded, not executed
let x = gpu.lazy([1.0, 2.0, 3.0, 4.0], [2, 2])
let w = gpu.lazy([0.5, 0.5, 0.5, 0.5], [2, 2])

# Operations build a computation graph (not executed yet)
let y = x.matmul(w)
let z = y.relu()

# Execution happens here - optimized as single fused kernel
let result = gpu.lazy_eval(z)
```

#### FlashAttention-2 (Memory-Efficient Attention)

```nsl
# O(N) memory instead of O(N²) - enables 128K+ context
let q = gpu.tensor(query_data, [batch, seq_len, head_dim])
let k = gpu.tensor(key_data, [batch, seq_len, head_dim])
let v = gpu.tensor(value_data, [batch, seq_len, head_dim])

# 2-4x faster, 5-20x less memory
let output = gpu.flash_attention(q, k, v)

# With causal mask (for autoregressive models)
let causal_output = gpu.flash_attention(q, k, v, true)
```

#### Fused Operations

```nsl
# Single fused kernel instead of 3 separate ones
let out = gpu.fused_linear_relu(input, weights, bias)
let activated = gpu.fused_gelu(linear_out, bias)
let normalized = gpu.layer_norm(input, gamma, beta, 1e-5)
let rms_out = gpu.rms_norm(input, gamma, 1e-6)  # LLaMA-style
```

#### Graph Compiler API

| Function | Description |
|----------|-------------|
| `gpu.graph()` | Create computation graph |
| `gpu.lazy(data, shape)` | Create lazy tensor |
| `gpu.lazy_eval(tensor)` | Force evaluation |
| `gpu.flash_attention(q, k, v, causal?)` | FlashAttention-2 |
| `gpu.layer_norm(input, gamma, beta, eps?)` | Fused LayerNorm |
| `gpu.rms_norm(input, gamma, eps?)` | RMS Normalization |
| `gpu.fused_linear_relu(input, weight, bias)` | Fused Linear+ReLU |
| `gpu.fused_gelu(input, bias)` | Fused Bias+GELU |

### Python Integration

NSL includes a Python namespace for interoperability:

```nsl
# Execute Python code
let result = python.execute("2 + 2")
print(result)  # 4

# Generate numpy-style random arrays
let arr = python.execute("np.random.randn(10)")
print(len(arr))  # 10

# AI functions
let analysis = python.ai.analyze("Hello world")
print(analysis["complexity"])

let embedding = python.ai.embedding("Neural networks are cool")
print(len(embedding))  # 768-dimensional vector

# Neural simulation
let neural = python.ai.neural([1.0, 2.0], [0.5, 0.3], "relu")
print(neural["output"])

# Quantum simulation
let quantum = python.ai.quantum([1, 0], ["H", "X"])
print(quantum["final_state"])
```

#### Python AI Namespace

| Function | Description |
|----------|-------------|
| `python.execute(code)` | Execute Python code |
| `python.ai.complete(prompt, model)` | AI text completion |
| `python.ai.embedding(text)` | Generate 768-dim embedding |
| `python.ai.analyze(data)` | Analyze data structure |
| `python.ai.neural(input, weights, activation)` | Neural network simulation |
| `python.ai.quantum(state, operations)` | Quantum state simulation |
| `python.ai.transform(data, operator)` | Consciousness transforms |
| `python.ai.claude(prompt)` | Call Claude Code CLI |

### MCP Session Persistence ⭐ NEW

When using NSL via MCP (Claude Code integration), sessions preserve state across calls:

```javascript
// Without session: Each call is independent
nsl(code: "fn add(a, b) { a + b }")  // Define function
nsl(code: "add(1, 2)")               // Error: add undefined!

// With session: State persists
nsl(code: "fn add(a, b) { a + b }", session_id: "dev")
nsl(code: "add(1, 2)", session_id: "dev")  // Works! Returns 3
```

#### Session Workflow Example

```javascript
// Build a model incrementally
nsl(code: "fn matmul(a, b) { ... }", session_id: "gpt")
nsl(code: "fn attention(q, k, v) { matmul(q, k) ... }", session_id: "gpt")
nsl(code: "fn layer(x) { attention(x, x, x) }", session_id: "gpt")
nsl(code: "let model = build_gpt()", session_id: "gpt")
nsl(code: "generate(model, 'Hello')", session_id: "gpt")
```

#### Session Management Tools

| Tool | Description |
|------|-------------|
| `nsl_session` | Execute code in a persistent session |
| `nsl_session_list` | List all active sessions |
| `nsl_session_show` | Show accumulated code in a session |
| `nsl_session_clear` | Clear session state (start fresh) |
| `nsl_session_delete` | Delete a session completely |

Sessions are saved to `~/.nsl-memory/sessions/` and persist across MCP server restarts.

---

## REPL and Debugging

### Starting the REPL

```bash
nsl
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `help` | Show help |
| `syntax` | Show control flow syntax |
| `tokens` | Show token reference |
| `clear` | Clear screen |
| `.load filename.nsl` | Load and execute file |
| `exit` | Exit REPL |

### Special Prefixes

| Prefix | Description | Example |
|--------|-------------|---------|
| `#code` | Show tokens for code | `#let x = 5` |
| `!cmd` | Execute shell command | `!dir` |

### Debug Tokens

```bash
# Set environment variable to see tokens
set NSL_DEBUG_TOKENS=1
nsl script.nsl
```

### Multi-line Input

The REPL supports multi-line input automatically:

```
NSL.Interpreter> fn factorial(n) {
>>     if (n <= 1) {
>>         return 1
>>     }
>>     return n * factorial(n - 1)
>> }
NSL.Interpreter> factorial(5)
120
```

---

## Auto-Fix (Error Correction)

NSL includes a powerful auto-fix system that can detect and automatically correct common errors.

### Analyze Code for Issues

```bash
nsl --suggest script.nsl
```

**Example Output:**
```
Found 10 issue(s):

Error at line 4, column 23:

      3 | # Missing closing brace
 >    4 | fn broken_function(x) {
                                ^
      5 |     return x * 2

  Unclosed '{' - missing closing '}'


Error at line 15, column 6:

     14 | let x = 5
 >   15 | if x = 10 {
               ^
     16 |     print("equal")

  Use == for comparison
  Suggested fix: = -> ==


Run with --fix to automatically apply fixes.
```

### Auto-Fix Errors

```bash
nsl --fix script.nsl
```

**Output:**
```
Found 10 issue(s):
...
Auto-fixed 3 issue(s) in script.nsl
```

The `--fix` flag will:
1. Analyze the code for issues
2. Apply safe fixes automatically
3. Show which fixes were applied
4. Run the fixed code

### What Auto-Fix Can Detect and Fix

| Category | Issue | Auto-Fix |
|----------|-------|----------|
| **Syntax** | `=` instead of `==` in conditions | ✅ Fixes to `==` |
| **Syntax** | Unclosed braces `{` | ⚠️ Detects only |
| **Syntax** | Unclosed strings `"` | ⚠️ Detects only |
| **Types** | Integer literals (e.g., `5`) | ✅ Suggests `5.0` |
| **Types** | Missing type annotations | ✅ Suggests `: number` |
| **Typos** | Misspelled identifiers | ✅ Suggests correction |
| **Typos** | Misspelled keywords | ✅ Suggests correction |
| **Typos** | Misspelled built-in functions | ✅ Suggests correction |

### Fix Categories

| Category | Description | Applied with --fix |
|----------|-------------|-------------------|
| `Error` | Must fix to compile | ✅ Yes |
| `Warning` | Should fix for correctness | ✅ Yes |
| `Style` | Optional style improvement | ❌ No |
| `Suggestion` | Potential improvement | ❌ No |

### Example: Before and After

**Before (`broken.nsl`):**
```nsl
fn calculate(x) {
    let result = 5
    if x = 10 {
        return result
    }
    return 0
}
```

**After `nsl --fix broken.nsl`:**
```nsl
fn calculate(x) {
    let result = 5
    if x == 10 {
        return result
    }
    return 0
}
```

### Built-in Functions Recognized

The auto-fix system knows about these built-in functions and will suggest corrections for typos:

- **Core:** `print`, `println`, `input`, `len`, `type`, `range`
- **Math:** `sqrt`, `abs`, `sin`, `cos`, `tan`, `exp`, `log`, `pow`, `floor`, `ceil`, `round`, `min`, `max`, `sum`, `avg`, `random`
- **Neural/ML:** `relu`, `sigmoid`, `tanh`, `softmax`, `normalize`, `mean`, `dot`, `zeros`, `ones`
- **File I/O:** `read_file`, `write_file`, `file_exists`, `list_dir`, `mkdir`, `cwd`, `cd`, `attention_read`, `◈_read`, `extract`, `stream_file`
- **HTTP:** `http_get`, `http_post`, `download`
- **String:** `split`, `join`, `replace`, `trim`, `upper`, `lower`, `contains`
- **JSON:** `json_parse`, `json_stringify`
- **Encoding:** `base64_encode`, `base64_decode`, `hex_encode`, `hex_decode`
- **Hashing:** `md5`, `sha256`, `sha512`

---

## Best Practices

### 1. Use `#` for Comments

```nsl
# Correct - use hash for comments
let x = 10

# WRONG - // is integer division, not a comment!
# let y = 20 // this would error
```

### 2. Prefer Immutable Variables

```nsl
# Prefer let (immutable)
let pi = 3.14159

# Only use mut when mutation is needed
mut counter = 0
counter = counter + 1
```

### 3. Use Type Hints for Documentation

```nsl
fn calculate_area(width: number, height: number) -> number {
    return width * height
}
```

### 4. Pattern Match Instead of If Chains

```nsl
# Better
match grade {
    case "A" => "Excellent"
    case "B" => "Good"
    case "C" => "Average"
    case _ => "Needs improvement"
}

# Instead of
if (grade == "A") { ... }
else if (grade == "B") { ... }
```

### 5. Use Closures for State

```nsl
fn makeIdGenerator() {
    let id = 0
    fn next() {
        id = id + 1
        return id
    }
    return next
}

let nextId = makeIdGenerator()
print(nextId())  # 1
print(nextId())  # 2
```

### 6. Handle Errors with Result

```nsl
fn safe_divide(a, b) {
    if (b == 0) {
        return err("Division by zero")
    }
    return ok(a / b)
}

match safe_divide(10, 2) {
    case ok(v) => print("Result:", v)
    case err(e) => print("Error:", e)
}
```

### 7. Use Pipeline for Data Transformation

```nsl
# Clean data flow
let result = data
    |> normalize
    |> transform
    |> validate
    |> process
```

### 8. Organize Code with Modules

```nsl
# math_utils.nsl
module math_utils {
    pub fn clamp(x, min, max) {
        if (x < min) { return min }
        if (x > max) { return max }
        return x
    }
}

# main.nsl
import { clamp } from math_utils
```

---

## Quick Reference

### Keywords

```
let, mut, const, fn, return, if, else, while, for, in,
break, continue, match, case, when, struct, enum, trait,
impl, pub, import, from, export, module, async, await,
true, false, null, ok, err, some, none, and, or, not
```

### Built-in Functions

```
# I/O
print, println, input

# Type/Conversion
type, len, to_string, str

# Collections
range, sum, mean, zeros

# Math
sin, cos, tan, sqrt, pow, abs, exp, log, random

# String
trim, lower, upper, split, join, replace, contains

# Result/Option
ok, err, some, none, is_ok, is_err, is_some, is_none, unwrap, unwrap_or
```

### File Operations with History ⭐ NEW

```nsl
# All file mutations are automatically tracked
file.write("config.txt", "v1")
file.write("config.txt", "v2")

# View history
let h = file.history("config.txt")

# Restore previous version
file.restore("config.txt", 0)  # Most recent pre-edit

# REPL shortcuts
history config.txt     # View history
restore config.txt 0   # Restore
```

### Code Analysis ⭐ NEW

```nsl
# Analyze any code file
let m = code.metrics("file.cs")      # {complexity, codeLines, ...}
let s = code.symbols("file.cs")      # Find classes, methods
let d = code.deps("file.cs")         # Find imports
let i = code.issues("file.cs")       # Find potential bugs
let f = code.flow("file.cs")         # Control flow analysis
let b = code.extract("file.cs", "MyMethod")  # Extract method body
let c = code.compare("old.cs", "new.cs")     # Structured diff
```

### Consciousness Operators

```
◈ (holographic/attention)
∇ (gradient/learning)
⊗ (tensor product/composition)
Ψ (quantum branching/superposition)
μ (memory)
σ (self/introspection)
```

---

## Example: Complete Program

```nsl
# Neural Network Training Example

# Define activation
fn relu(x) {
    if (x > 0) { return x }
    return 0
}

# Simple neuron
fn neuron(inputs, weights, bias) {
    mut sum = bias
    for i in 0..len(inputs) {
        sum = sum + inputs[i] * weights[i]
    }
    return relu(sum)
}

# Training step
fn train_step(current_weights, gradients, lr) {
    let n = len(current_weights)
    mut new_weights = zeros(n)
    for i in 0..n {
        new_weights[i] = current_weights[i] - lr * gradients[i]
    }
    return new_weights
}

# Main
fn main() {
    print("=== NSL Neural Network Demo ===")

    let inputs = [1.0, 0.5, -0.5]
    let weights = [0.2, 0.8, -0.5]
    let bias = 0.1

    let output = neuron(inputs, weights, bias)
    print("Neuron output:", output)

    # Simulate training
    let gradients = [0.01, -0.02, 0.015]
    let learning_rate = 0.1

    let new_weights = train_step(weights, gradients, learning_rate)
    print("Updated weights:", new_weights)

    print("=== Complete ===")
}

main()
```

---

## Phase 5: Developer Tools ⭐ NEW

NSL now includes full development tooling for code generation, AST manipulation, and metaprogramming.

### Code Generation

Generate code in multiple languages:

```nsl
# Generate a C# class
let cls = codegen.class("User", ["string Name", "int Age"], "cs")
print(cls)
# Output:
# public class User
# {
#     public string Name { get; set; }
#     public int Age { get; set; }
# }

# Generate functions in different languages
let nslFn = codegen.function("greet", ["name"], "return hello", "nsl")
let jsFn = codegen.function("greet", ["name"], "return hello", "js")
let pyFn = codegen.function("greet", ["name"], "return hello", "python")

# Generate other constructs
let ctrl = codegen.controller("User", ["Get", "Post", "Delete"])
let dto = codegen.dto("UserDto", ["string Name", "int Age"])
let test = codegen.test("ShouldWork", "Assert.True(true)", "cs")
let prop = codegen.property("string", "Email", "cs")
let enumCode = codegen.enum("Status", ["Active", "Pending", "Done"], "cs")
```

### AST Manipulation

Parse, transform, and emit code:

```nsl
# Parse code to AST
let code = "let x = 1\nfn test() { return x }"
let parsed = ast.parse(code)
print(parsed.nodeCount)  # 2

# Find specific node types
let functions = ast.find(parsed, "function")
let declarations = ast.find(parsed, "declaration")

# Rename symbols across code
let renamed = ast.rename(code, "oldVar", "newVar")

# Emit AST back to code
let output = ast.emit(parsed)
```

### Project Scaffolding

Create projects from templates:

```nsl
# List available templates
let templates = project.templates()
# ["console", "webapi", "library", "nsl"]

# Create a new project
project.create("MyApp", "webapi", "C:/Projects")
# Creates:
#   MyApp/Program.cs
#   MyApp/Controllers/MyAppController.cs
#   MyApp/MyApp.csproj

# Scaffold individual components
project.scaffold("controller", "User", ".")
project.scaffold("service", "Auth", ".")
project.scaffold("repository", "Data", ".")
```

### LSP Features

Language server functionality for code intelligence:

```nsl
# Find symbols in a file
let symbols = lsp.symbols("MyClass.cs")
# [{name: "MyClass", kind: "class", line: 5}, ...]

# Find all references to a symbol
let refs = lsp.references("src/", "MyClass")
# [{file: "...", line: 10, text: "..."}, ...]

# Go to definition
let def = lsp.definition("src/", "MyClass")
# {file: "MyClass.cs", line: 5, text: "public class MyClass"}

# Get diagnostics (TODOs, FIXMEs, issues)
let diags = lsp.diagnostics("file.cs")
# [{line: 42, severity: "warning", message: "FIXME found"}]

# Autocomplete
let completions = lsp.complete("le")
# [{label: "let", kind: "keyword"}]
```

### Metaprogramming (Dual Mode)

NSL provides two modes for dynamic code:

```nsl
# Mode 1: meta.eval() - Executes code at runtime
let result = meta.eval("1 + 2 + 3")
print(result)  # 6

let dynamic = meta.eval("list.sum([1, 2, 3, 4, 5])")
print(dynamic)  # 15

# Mode 2: meta.validate() - Validates without executing
let check = meta.validate("let x = 42")
print(check)  # {valid: true, tokens: 5}

let bad = meta.validate("let = invalid")
print(bad)  # {valid: false, error: "Parse error..."}
```

### Other Meta Functions

```nsl
# Generate unique symbols
let sym = meta.gensym("var")  # "var_a1b2c3d4"

# Check if code compiles
let compiled = meta.compile("fn test() { return 1 }")
print(compiled.success)  # true

# List all global symbols
let globals = meta.globals()
print(len(globals))  # 181+

# Check if symbol is defined
let exists = meta.defined("print")  # true

# Template generation
let code = meta.generate("Hello $name!", {name: "World"})
print(code)  # "Hello World!"

# Reflection
let info = meta.reflect({a: 1, b: 2})
print(info.type)  # "Dictionary`2"
print(info.keys)  # ["a", "b"]
```

### Safe AI Development with Sim Mode

The key safety feature: **AI executes through sim, user controls commit**.

```nsl
# Step 1: AI enters sim mode
sim.begin()
print(sim.active())  # true

# Step 2: Code executes, but file writes are captured
meta.eval("file.write('output.txt', generatedCode)")
meta.eval("file.write('config.json', newConfig)")

# Step 3: User reviews what would happen
let pending = sim.pending()
print(pending)
# {writes: ["output.txt", "config.json"], deletes: [], count: 2}

let changes = sim.diff()
# Shows content of each pending write

# Step 4: User decides
sim.commit()    # → Real side effects happen
# OR
sim.rollback()  # → Nothing happened, safe to try again
```

### Eval Logging

All dynamic code execution is logged for safety:

```nsl
# Execute some evals
meta.eval("1 + 1")
meta.eval("let x = 42")

# View execution history
let history = meta.evalLog(10)
# Returns last 10 evals with:
#   - timestamp
#   - code (truncated if long)
#   - simMode (whether sim was active)
```

### Multi-File Refactoring ⭐ NEW

The `refactor` namespace enables complex multi-file operations with preview and rollback:

```nsl
# Rename a symbol across all files in a directory
# .nsl files: AST-backed semantic rename (true symbol rename)
# Other files: Textual with string/comment masking
refactor.rename("src/", "OldClassName", "NewClassName", {extensions: [".cs", ".nsl"]})
# Returns: {fileCount, semanticFiles, textualFiles, changes: [{file, matches, semantic}]}

# Find all usages of a symbol
let usages = refactor.usages("src/", "MyFunction", [".cs", ".nsl"])
print("Found", usages.count, "usages in", usages.files, "files")
for usage in usages.usages {
    if usage.isDefinition {
        print("Definition at", usage.file, "line", usage.line)
    } else {
        print("Usage at", usage.file, "line", usage.line)
    }
}

# Replace pattern across files (literal or regex)
refactor.replace("src/", "oldPattern", "newPattern", false)  # literal
refactor.replace("src/", r"old\w+", "new$0", true)           # regex

# Extract code to a new file
refactor.extract(
    "BigFile.cs",           # source file
    r"class Helper.*?\n}",  # pattern to extract
    "Helper.cs",            # target file
    "${content}"            # wrapper template
)

# Preview all pending changes before committing
let preview = refactor.preview()
for change in preview.pending {
    print(change.file, "-", change.operation, "-", change.details)
}

# See line count changes with actual diffs
let diffs = refactor.diff()
for d in diffs {
    print(d.file, ":", d.summary)  # e.g. "+3 -1 ~5"
    for change in d.changes {
        print("  Line", change.line, ":", change.old, "->", change.new)
    }
}

# Commit all changes (routes through file.history safety pipeline)
let result = refactor.commit("rename refactor")
if result.success {
    print("Committed", result.count, "files")
}

# Undo last commit (restores via file.history)
refactor.undo()  # Restores all files from last refactor

# Or rollback if something looks wrong
refactor.rollback()
```

### Batch Refactoring

Apply multiple refactoring operations at once:

```nsl
refactor.batch("src/", [
    {type: "rename", old: "oldVar1", new: "newVar1"},
    {type: "rename", old: "oldVar2", new: "newVar2"},
    {type: "rename", old: "oldFunc", new: "newFunc"}
])

# Preview before committing
refactor.preview()

# Apply all or rollback all
refactor.commit()
# or
refactor.rollback()
```

### Contextual Help System

NSL provides layered help - overview first, detail on demand:

```nsl
# Get overview
help()
# Returns: {summary, namespaces, quickstart, usage}

# Get namespace-specific help
help("sim")       # {summary, functions, workflow, safety, example}
help("refactor")  # {summary, functions, semantic, workflow, example}
help("file")      # {summary, functions, safety, example}
help("safety")    # {principles, dangerous, governed}

# Available topics: file, dir, sim, refactor, meta, git, codegen, ast, string, json, safety
```

### API Introspection

Understand and verify the API surface:

```nsl
# Check canonical name for any function
meta.canonical("string.startswith")  # → "string.startsWith"
meta.canonical("string.startsWith")  # → "string.startsWith" (already canonical)

# List all aliases with naming policy
meta.aliases()
# Returns: {
#   "string.startswith": "string.startsWith",
#   "_policy": "Aliases are lowercase versions of camelCase canonicals. 1:1 mapping only.",
#   "_count": 10
# }

# Describe any symbol
meta.describe("file.write")
# Returns: {name, namespace, function, isAlias, canonical, category, exists}

# Get capabilities by safety level
meta.capabilities()
# Returns: {
#   observe: {description, functions, count},  # Read-only
#   plan: {description, functions, count},     # Produces diffs/previews
#   act: {description, functions, count},      # Mutates (through safety pipeline)
#   dangerous: {description, functions, count} # Requires explicit gating
# }
```

### Pretty Print - Formatted Output

Use `pp()` or `pretty()` for human-readable output of structured data:

```nsl
# Default print shows flat structure
print(help("sim"))  # {summary: ..., functions: [...], ...}

# pp() formats with newlines and indentation
pp(help("sim"))
#   summary: Simulation mode - preview changes before committing
#   functions: begin, commit, rollback, pending, diff, write, delete
#   workflow: sim.begin() → make changes → sim.pending() → sim.commit()
#   example:
#     sim.begin()
#     file.write('test.txt', 'hello')
#     sim.pending()

# Works with any structured data
pp(refactor.diff())       # Formatted diff with line changes
pp(meta.capabilities())   # Formatted capability map
pp(git.status())          # Formatted git status

# pretty() is an alias
pretty(help("safety"))
```

### The Safety Hierarchy

```
Preview → Simulate → Commit → Rollback → Undo
   ↓          ↓          ↓         ↓        ↓
 See it   Test it    Do it    Discard   Restore
```

**Trust Invariants:**
1. No silent mutation - Every change is visible
2. No irreversible action without history - Everything can be undone
3. No eval without trace - Dynamic execution is observable
4. No codegen without inspection - Generated code is returned, not auto-committed
5. No observation that mutates - Read operations don't have side effects

See [PHILOSOPHY.md](PHILOSOPHY.md) for the complete safety contract.

---

Happy coding with NSL!
