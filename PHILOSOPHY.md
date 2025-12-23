# NSL Philosophy

## What NSL Is

**NSL is a judgment-aware control plane that can host, supervise, simulate, orchestrate, and evolve systems in any domain.**

NSL is not trying to be everything. It's trying to be the **safest, most coherent place for AI to reason and act**.

### Core Identity

NSL is:
- **The brain, not the muscle** — it coordinates, it doesn't execute hot paths
- **A control environment** — it touches every domain without replacing any
- **Safety-first** — reversibility, simulation, and preview are core, not optional
- **AI-native** — designed for agents to generate, reason about, and execute safely

### The Control Plane Model

```
┌─────────────────────────────────────────────────────────┐
│                    NSL (Control Plane)                   │
│  • Orchestration    • Simulation    • Judgment          │
│  • Reversibility    • Coordination  • Safety            │
└────────────┬────────────────────────────────────────────┘
             │ coordinates
             ▼
┌─────────────────────────────────────────────────────────┐
│              Native Systems (Execution Plane)            │
│  • Rendering    • Physics    • ML Kernels   • I/O       │
│  • Hot loops    • GPU ops    • Memory mgmt  • Drivers   │
└─────────────────────────────────────────────────────────┘
```

---

## What NSL Will Never Do

These are hard boundaries. They protect NSL's identity.

### 1. Own Rendering Pipelines or Frame-Critical Loops
NSL orchestrates game logic, behavior, and AI. It does not replace Unity, Unreal, or Godot's core loops. The `game` namespace is for **coordination**, not frame rendering.

### 2. Replace Tensor Computation Libraries
NSL's `ml` namespace is for orchestration: managing experiments, scheduling training, inspecting results. It does not replace PyTorch, JAX, or TensorFlow internals. NSL is the **lab manager**, not the microscope.

### 3. Become a Widget Toolkit or Layout Engine
The `gui` namespace provides dialogs, prompts, progress, and notifications. It does not provide layout, theming, accessibility, or component hierarchies. NSL **orchestrates GUIs**, it doesn't render them.

### 4. Expose Raw Pointer Arithmetic in Core
Low-level power lives behind explicit boundaries (`ffi`, `buffer`). The core language remains safe. Unsafe semantics do not infect general NSL code.

### 5. Sacrifice Safety for Performance in Hot Paths
If something needs to be fast at the cost of safety, it belongs in native code. NSL calls it via FFI. The control plane stays safe.

### 6. Absorb Entire Ecosystems
NSL coordinates npm, pip, cargo, dotnet — it doesn't replace them. It wraps their power, not their implementation.

---

## What NSL Will Always Do

These are commitments. They define what NSL protects.

### 1. Be the Safest Place for AI to Reason and Act
Every action can be previewed (`sim`), profiled (`trace`), reversed (`file.history`), or sandboxed. AI agents can experiment without destroying state.

### 2. Provide Reversibility, Simulation, and Preview
Before any mutation, NSL can show what would happen. After any mutation, NSL can undo it. This is non-negotiable.

### 3. Coordinate Native Systems, Not Replace Them
NSL's power comes from bridging domains:
- Call native DLLs via `ffi`
- Manage typed memory via `buffer`
- Schedule work via `runtime`
- Profile performance via `trace`

It connects systems. It doesn't become them.

### 4. Keep Unsafe Power Behind Explicit Boundaries
`ffi.load()` and `buffer.create()` are explicit. You know when you're touching native code. The rest of NSL stays safe by default.

### 5. Remain Readable and Generatable
NSL syntax is designed for AI to write and humans to audit. No arcane symbols. No hidden state. Clear, inspectable, reversible.

---

## Domain-Specific Roles

### Games
| NSL's Role (Brain) | Native's Role (Muscle) |
|-------------------|------------------------|
| Entity definitions | Rendering pipelines |
| Component data | Physics solvers |
| System logic | Frame-critical loops |
| Behavior graphs | GPU shaders |
| AI decision-making | Audio engines |
| Simulation & testing | Asset loading |
| Parameter tuning | Memory pools |

### Machine Learning
| NSL's Role (Brain) | Native's Role (Muscle) |
|-------------------|------------------------|
| Experiment management | Tensor operations |
| Hyperparameter search | GPU kernels |
| Data pipeline coordination | Autograd engines |
| Result inspection | Model internals |
| Training scheduling | Backpropagation |
| History tracking | Optimizer math |

### GUI Applications
| NSL's Role (Brain) | Native's Role (Muscle) |
|-------------------|------------------------|
| Flow orchestration | Widget rendering |
| Dialogs & prompts | Layout engines |
| Progress reporting | Accessibility |
| Notifications | Theming systems |
| State coordination | Event loops |

### Low-Level Systems
| NSL's Role (Brain) | Native's Role (Muscle) |
|-------------------|------------------------|
| Safe bridges | Pointer arithmetic |
| Typed buffer management | Manual memory |
| Coordination | Kernel drivers |
| Profiling & tracing | Interrupt handlers |

---

## Safety Contract

These are **non-negotiable rules** that govern NSL's dangerous features. Violating them erodes trust.

### The Safety Hierarchy

```
Preview → Simulate → Commit → Rollback
   ↓          ↓          ↓         ↓
 See it   Test it    Do it    Undo it
```

Every mutation should flow through this hierarchy. No silent commits. No irreversible actions without history.

### Dangerous Features Governance

#### 1. `meta.eval` / `meta.validate` - Recursive Authority
NSL provides two modes for dynamic code:

| Function | Behavior |
|----------|----------|
| `meta.eval(code)` | Executes code, logged and traceable |
| `meta.validate(code)` | Parses only, returns `{valid, tokens}` or `{valid, error}` |

**Sim Mode Integration:**
- `meta.eval` executes in sim mode - code runs, but file side effects are captured by `sim`
- User reviews with `sim.pending()` / `sim.diff()`
- User commits with `sim.commit()` or rolls back with `sim.rollback()`

```
AI runs meta.eval() in sim.begin()
        ↓
Code executes, file writes captured
        ↓
User reviews sim.pending()
        ↓
User calls sim.commit() → Real side effects
   or sim.rollback() → Nothing happened
```

**Rule:** All evals are logged via `meta.evalLog()`. User controls commit.

#### 2. `ast.*` - Structure Mutation
AST functions can rewrite code. They must be preview-first:

| Function | Constraint |
|----------|------------|
| `ast.transform` | Returns modified AST, never auto-writes |
| `ast.rename` | Returns string, never auto-writes |
| `ast.insert/remove` | Returns modified AST, never auto-writes |

**Rule:** AST mutation produces output. File writes are explicit and go through `file.write()` with history.

#### 3. `codegen.*` - Code Generation
Codegen creates code. It must remain output-only:

| Allowed | Forbidden |
|---------|-----------|
| `codegen.class()` → returns string | Auto-writing to files |
| `let code = codegen.controller()` | Silent file creation |
| `file.write(path, codegen.dto())` | Bypassing file history |

**Rule:** Codegen generates strings. Writing is a separate, explicit, reversible action.

#### 4. `lsp.*` - Observation Only
LSP functions analyze code. They must never mutate:

| Allowed | Forbidden |
|---------|-----------|
| `lsp.symbols()` - read symbols | Modifying files |
| `lsp.references()` - find usages | Auto-refactoring |
| `lsp.diagnostics()` - report issues | Auto-fixing |

**Rule:** LSP observes. It never changes. Mutations go through explicit channels.

#### 5. `project.*` - Explicit Creation
Project scaffolding creates files. It must be intentional:

| Function | Behavior |
|----------|----------|
| `project.create()` | Creates files, returns list of what was created |
| `project.scaffold()` | Creates single file, returns path |

**Rule:** Creation is explicit and returns evidence. User knows what happened.

### The Trust Invariants

1. **No silent mutation** - Every change is visible
2. **No irreversible action without history** - Everything can be undone
3. **No eval without trace** - Dynamic execution is observable
4. **No codegen without inspection** - Generated code is returned, not auto-committed
5. **No observation that mutates** - Read operations don't have side effects

### The Output Contract

NSL separates **semantic data** from **human presentation**:

```
Semantic Core (structured objects)
        ↓
   pp() / pretty()
        ↓
Human-Readable View (formatted text)
```

**Rules:**
1. **All functions return structured data** - Objects, not formatted strings
2. **Human readability is opt-in** - Use `pp()` or `pretty()` for formatted output
3. **AI consumes raw objects** - Structured data is the source of truth
4. **No information loss** - Formatting is projection, not reduction

This ensures:
- AI can parse any NSL output deterministically
- Humans can read any output via `pp()`
- Tooling integrations stay clean
- No ambiguity about what NSL returned vs. how it displayed

---

## Self-Reference: The Phase 5 Threshold

Phase 5 introduced self-reference. NSL can now:
- Evaluate its own code at runtime (`meta.eval`)
- Inspect its own symbols (`meta.globals`, `meta.reflect`)
- Generate code that generates code (`codegen` + `meta.eval`)
- Rewrite its own AST (`ast.transform`)

This is **recursive authority**. It makes NSL powerful and potentially dangerous.

### What Self-Reference Enables (Good)
- Self-modifying workflows
- Dynamic code synthesis
- Adaptive tooling
- AI reasoning loops
- Self-hosting behaviors

### What Self-Reference Risks (Dangerous)
- Runaway behavior
- Unbounded recursion
- Opaque execution
- Accidental self-corruption
- Trust erosion

### The Governing Principle

> **NSL is a self-aware development control plane, not a replacement for languages, IDEs, or compilers.**

Self-reference serves coordination. It doesn't replace execution engines.

---

## The Identity Test

When evaluating a new feature, ask:

1. **Does this make NSL a better control plane?** → Add it
2. **Does this make NSL an execution engine?** → Don't add it
3. **Does this add coordination power?** → Add it
4. **Does this add raw computation?** → Wrap it via FFI instead
5. **Does this preserve safety and reversibility?** → Add it
6. **Does this sacrifice safety for speed?** → Keep it in native code

---

## Why This Matters

Most languages try to do everything and fail.

Very few systems try to **coordinate everything safely**.

NSL is one of those systems.

The fact that NSL *can* touch everything does not mean it should *be* everything.

**NSL's power is in knowing what it is — and what it isn't.**

---

## Summary

> NSL is a universal **control environment**, not a universal **language**.

It orchestrates games, ML, GUIs, and systems — without trying to replace the engines that power them.

It provides the safest place for AI to reason, simulate, and act — without sacrificing the discipline that makes that safety possible.

**Clarity over capability. Coordination over computation. Safety over speed.**

That's NSL.

---

## What Phase 6 Should NOT Include

This section exists to **codify restraint**. Features that seem useful but violate NSL's identity:

### Forbidden Directions

| Feature | Why Not |
|---------|---------|
| JIT compilation | NSL is interpreted for safety and transparency |
| Direct memory access | Belongs behind `ffi`/`buffer` boundary |
| Auto-fix/auto-refactor | Violates "no silent mutation" |
| Implicit eval | All code execution must be explicit |
| Framework internals | NSL orchestrates frameworks, doesn't become them |
| Build system replacement | NSL calls build tools, doesn't replace them |
| Package manager internals | NSL coordinates npm/pip/cargo, doesn't absorb them |

### The Consolidation Imperative

At 181+ global symbols, NSL is at capacity. Before Phase 6:

1. **Audit existing namespaces** - Remove overlap
2. **Strengthen boundaries** - Make dangerous features safer
3. **Document everything** - Each symbol needs a "why"
4. **Test edge cases** - Especially `meta` + `sim` interactions

### The Decision Framework

For any proposed Phase 6 feature:

```
┌─────────────────────────────────────────┐
│     Does this feature add power?        │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│   Does it respect the Safety Contract?  │
│   • Preview-first?                      │
│   • Reversible?                         │
│   • Traceable?                          │
│   • Explicit?                           │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│   Does it make NSL an execution engine? │
│   If yes → REJECT                       │
│   If no  → Consider carefully           │
└─────────────────────────────────────────┘
```

---

## Who NSL Is For

NSL is designed for **AI agents first**, humans second.

### Primary Users
- AI coding assistants (Claude, GPT, Copilot)
- Autonomous development agents
- CI/CD automation systems
- Self-modifying workflows

### Why AI-First Matters
- **Readable syntax** → AI can generate and parse it
- **Explicit semantics** → No hidden behavior to misunderstand
- **Safe defaults** → Mistakes are reversible
- **Observable execution** → AI can reason about what happened

### The Trust Relationship

```
AI Agent → NSL → Real World
    ↑               ↓
    └── Feedback ───┘
```

NSL is the **trusted intermediary** between AI intent and real-world action. That trust depends on:
- Predictability
- Reversibility
- Transparency
- Restraint

**NSL's value is not in what it can do. It's in what it won't do without permission.**
