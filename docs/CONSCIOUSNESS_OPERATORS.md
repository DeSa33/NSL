# NSL Consciousness Operators: Mathematical Foundations

This document provides rigorous mathematical definitions and theoretical foundations for NSL's consciousness operators. These operators bridge neural computation with symbolic reasoning, enabling expressive AI-native programming.

---

## Table of Contents

1. [Overview](#overview)
2. [Holographic Operator (◈)](#holographic-operator-)
3. [Gradient Operator (∇)](#gradient-operator-)
4. [Tensor Product Operator (⊗)](#tensor-product-operator-)
5. [Quantum Branching Operator (Ψ)](#quantum-branching-operator-ψ)
6. [Operator Composition](#operator-composition)
7. [Implementation Architecture](#implementation-architecture)
8. [References](#references)

---

## Overview

NSL introduces four fundamental consciousness operators that provide high-level abstractions for neural-symbolic computation:

| Operator | Symbol | Unicode | Purpose | Mathematical Domain |
|----------|--------|---------|---------|---------------------|
| Holographic | ◈ | U+25C8 | Distributed attention | Attention mechanisms |
| Gradient | ∇ | U+2207 | Automatic differentiation | Differential calculus |
| Tensor Product | ⊗ | U+2297 | Compositional structure | Multilinear algebra |
| Quantum Branching | Ψ | U+03A8 | Superposition states | Probability theory |

These operators are designed to be:
- **Composable**: Operators can be combined freely
- **Differentiable**: All operators support automatic differentiation
- **GPU-accelerated**: Optimized for parallel execution via ILGPU
- **Type-safe**: Static type checking prevents runtime errors

---

## Holographic Operator (◈)

### Intuition

The holographic operator implements distributed representation through attention mechanisms. Like a hologram where every part contains information about the whole, this operator creates representations where information is distributed across all dimensions.

### Mathematical Definition

Given an input tensor **x** ∈ ℝⁿ and a set of value vectors {**v**₁, **v**₂, ..., **v**ₖ} ∈ ℝᵈ, the holographic operator computes:

```
◈(x) = Σᵢ αᵢ · vᵢ
```

where the attention weights α are computed via scaled dot-product attention:

```
αᵢ = softmax(q · kᵢᵀ / √d)

q = Wq · x    (query projection)
kᵢ = Wk · xᵢ  (key projection)
vᵢ = Wv · xᵢ  (value projection)
```

### Softmax Definition

The softmax function normalizes scores into a probability distribution:

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

This ensures Σᵢ αᵢ = 1 and all αᵢ > 0.

### Properties

1. **Permutation Equivariance**: The output is equivariant to permutations of input elements
   ```
   ◈(π(x)) = π(◈(x))
   ```

2. **Bounded Output**: Due to softmax normalization, outputs remain stable

3. **Differentiability**: The operator is smooth and differentiable everywhere:
   ```
   ∂αᵢ/∂zⱼ = αᵢ(δᵢⱼ - αⱼ)
   ```

4. **Temperature Control**: Scaling by τ controls attention sharpness:
   ```
   αᵢ = softmax(q · kᵢᵀ / (√d · τ))
   τ → 0: one-hot (hard attention)
   τ → ∞: uniform (no attention)
   ```

### Multi-Head Extension

For multi-head attention with h heads:

```
◈ₕ(x) = Concat(head₁, head₂, ..., headₕ) · Wₒ

where headᵢ = Attention(Wqⁱ·x, Wkⁱ·x, Wvⁱ·x)
```

This allows the model to jointly attend to information from different representation subspaces.

### NSL Syntax

```nsl
// Single holographic projection
let attended = ◈ input

// With explicit configuration
let attended = ◈(input, heads=8, dim=64)

// Chained holographic operations (multi-layer attention)
let deep_repr = ◈ ◈ ◈ input

// Self-attention in a transformer layer
fn attention_layer(x) {
    let q = linear(x, Wq)
    let k = linear(x, Wk)
    let v = linear(x, Wv)
    return ◈(q, k, v)
}
```

### Complexity Analysis

| Metric | Complexity | Notes |
|--------|------------|-------|
| Time | O(n² · d) | n = sequence length, d = dimension |
| Space | O(n² + n·d) | Attention matrix + output |
| GPU Parallelism | High | Parallelizable across heads and batch |

### Connection to Information Theory

The attention mechanism can be viewed as soft information retrieval:

```
I(attended; input) ≤ H(attention_weights)
```

where H is entropy. Sharp attention (low τ) reduces information loss but may overfit.

---

## Gradient Operator (∇)

### Intuition

The gradient operator enables automatic differentiation, computing derivatives of computational graphs. This is fundamental for training neural networks and optimizing objective functions.

### Mathematical Definition

For a scalar function f: ℝⁿ → ℝ and input **x** ∈ ℝⁿ:

```
∇f(x) = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)ᵀ
```

For tensor-valued functions f: ℝⁿ → ℝᵐ, we compute the Jacobian matrix:

```
J(f)ᵢⱼ = ∂fᵢ/∂xⱼ    ∈ ℝᵐˣⁿ
```

### Chain Rule (Backpropagation)

NSL implements reverse-mode automatic differentiation:

```
∂L/∂xᵢ = Σⱼ (∂L/∂yⱼ) · (∂yⱼ/∂xᵢ)
```

The computation graph is traced forward, then gradients flow backward:

```
Forward:  x → f₁ → h₁ → f₂ → h₂ → ... → fₙ → L
Backward: ∂L/∂x ← ∂L/∂h₁ ← ∂L/∂h₂ ← ... ← ∂L/∂L = 1
```

### Gradient Rules

1. **Linearity**:
   ```
   ∇(αf + βg) = α∇f + β∇g
   ```

2. **Product Rule**:
   ```
   ∇(f · g) = f · ∇g + g · ∇f
   ```

3. **Chain Rule**:
   ```
   ∇(f ∘ g)(x) = ∇f(g(x)) · ∇g(x)
   ```

4. **Matrix Calculus**:
   ```
   ∂(Ax)/∂x = Aᵀ
   ∂(xᵀAx)/∂x = (A + Aᵀ)x
   ```

### Common Gradients Table

| Function | Gradient |
|----------|----------|
| f(x) = xⁿ | n·xⁿ⁻¹ |
| f(x) = eˣ | eˣ |
| f(x) = ln(x) | 1/x |
| f(x) = σ(x) | σ(x)(1 - σ(x)) |
| f(x) = tanh(x) | 1 - tanh²(x) |
| f(x) = ReLU(x) | 1 if x > 0, else 0 |
| f(x) = softmax(x)ᵢ | softmax(x)ᵢ(δᵢⱼ - softmax(x)ⱼ) |

### Higher-Order Gradients

NSL supports higher-order differentiation:

```nsl
let grad = ∇ f(x)           // First derivative
let hessian = ∇(∇ f(x))     // Second derivative (Hessian matrix)
let jerk = ∇(∇(∇ f(x)))     // Third derivative
```

The Hessian matrix H has elements:

```
Hᵢⱼ = ∂²f / (∂xᵢ ∂xⱼ)
```

Properties:
- **Symmetric**: Hᵢⱼ = Hⱼᵢ (for twice-differentiable f)
- **Positive definite at minima**: xᵀHx > 0 for all x ≠ 0

### Gradient Tape

NSL uses a gradient tape to record operations:

```nsl
with GradientTape() as tape {
    tape.watch(x)
    let y = f(x)
    let z = g(y)
}

let dz_dx = tape.gradient(z, x)
```

### NSL Syntax

```nsl
// Basic gradient computation
backward(loss)
let w_grad = grad(weights)

// Gradient clipping for stability
let clipped = clip_grad(∇loss, max_norm=1.0)

// Gradient accumulation
for batch in batches {
    let loss = forward(batch)
    backward(loss)
    accumulate_grad(weights)
}
apply_gradients(optimizer, weights)

// Second-order optimization
let H = hessian(loss, weights)
let update = -inv(H) @ grad(weights)  // Newton's method
```

### Numerical Stability

NSL implements several techniques:

1. **Gradient Clipping**: Prevents exploding gradients
   ```
   g' = g · min(1, max_norm / ||g||)
   ```

2. **Mixed Precision**: FP16 forward, FP32 gradients

3. **Gradient Checkpointing**: Trades compute for memory

---

## Tensor Product Operator (⊗)

### Intuition

The tensor product creates compositional structure by combining two tensors into a higher-dimensional space. This is essential for representing relationships, binding features, and building hierarchical structures.

### Mathematical Definition

For vectors **a** ∈ ℝᵐ and **b** ∈ ℝⁿ, the tensor product (outer product) is:

```
(a ⊗ b)ᵢⱼ = aᵢ · bⱼ    ∈ ℝᵐˣⁿ
```

For higher-order tensors A ∈ ℝⁱ¹ˣⁱ²ˣ...ˣⁱᵐ and B ∈ ℝʲ¹ˣʲ²ˣ...ˣʲⁿ:

```
(A ⊗ B)ᵢ₁...ᵢₘⱼ₁...ⱼₙ = Aᵢ₁...ᵢₘ · Bⱼ₁...ⱼₙ
```

### Properties

1. **Bilinearity**:
   ```
   (αa + βb) ⊗ c = α(a ⊗ c) + β(b ⊗ c)
   a ⊗ (αb + βc) = α(a ⊗ b) + β(a ⊗ c)
   ```

2. **Associativity** (up to isomorphism):
   ```
   (a ⊗ b) ⊗ c ≅ a ⊗ (b ⊗ c)
   ```

3. **Non-commutativity**:
   ```
   a ⊗ b ≠ b ⊗ a (in general)
   ```

4. **Dimension Formula**:
   ```
   dim(a ⊗ b) = dim(a) × dim(b)
   ```

5. **Rank**:
   ```
   rank(a ⊗ b) = min(rank(a), rank(b)) (for matrices)
   ```

### Kronecker Product (Matrix Case)

For matrices A ∈ ℝᵐˣⁿ and B ∈ ℝᵖˣᵍ:

```
A ⊗ B = [a₁₁B  a₁₂B  ...  a₁ₙB]
        [a₂₁B  a₂₂B  ...  a₂ₙB]
        [...   ...   ...  ... ]
        [aₘ₁B  aₘ₂B  ...  aₘₙB]   ∈ ℝ⁽ᵐᵖ⁾ˣ⁽ⁿᵍ⁾
```

Properties of Kronecker product:
- (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD) (when dimensions match)
- (A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ
- (A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹ (when inverses exist)

### Role-Filler Binding

Tensor products enable symbolic binding in neural networks:

```nsl
// Define role and filler vectors
let role_agent = random_vector(dim=256)
let role_action = random_vector(dim=256)
let concept_john = embed("John")
let concept_run = embed("run")

// Bind concepts to roles
let agent_binding = role_agent ⊗ concept_john
let action_binding = role_action ⊗ concept_run

// Compose into a scene representation
let scene = agent_binding + action_binding  // Superposition

// Retrieve: unbind using role
let retrieved_agent = scene · role_agentᵀ  // Approximately concept_john
```

### Holographic Reduced Representations (HRR)

NSL also supports circular convolution as a dimension-preserving alternative:

```
a ⊛ b = F⁻¹(F(a) ⊙ F(b))
```

where:
- F is the Fourier transform
- ⊙ is element-wise (Hadamard) product
- Result maintains same dimensionality as inputs

### NSL Syntax

```nsl
// Basic tensor product
let outer = a ⊗ b

// Batch tensor product
let batched = batch_outer(A, B)

// Kronecker product for matrices
let kron = kronecker(M1, M2)

// Role binding
let bound = role ⊗ filler

// Circular convolution (dimension-preserving)
let hrr = circular_conv(a, b)
```

### Applications

1. **Attention Mechanisms**: QKᵀ computes pairwise similarities
2. **Neural Binding**: Combine features with roles
3. **Quantum States**: Represent entangled states |ψ⟩ ⊗ |φ⟩
4. **Graph Neural Networks**: Edge feature computation
5. **Compositional Semantics**: Build sentence representations

---

## Quantum Branching Operator (Ψ)

### Intuition

The quantum branching operator creates superposition states, allowing multiple computational paths to exist simultaneously. Unlike classical branching, these paths can interfere and combine through probability amplitudes.

### Mathematical Definition

A quantum branch state is represented as:

```
|Ψ⟩ = Σᵢ αᵢ |φᵢ⟩
```

where:
- |φᵢ⟩ are basis states (computational branches)
- αᵢ ∈ ℂ are complex probability amplitudes
- Σᵢ |αᵢ|² = 1 (normalization)

### Probability Amplitudes

Each amplitude αᵢ = rᵢ · e^(iθᵢ) has:
- Magnitude rᵢ = |αᵢ| (related to probability)
- Phase θᵢ (enables interference)

The probability of observing branch i upon measurement:

```
P(i) = |αᵢ|² = αᵢ · αᵢ*
```

### Interference

Amplitudes can interfere:

```
|α + β|² = |α|² + |β|² + 2Re(α*β)
```

- **Constructive**: When phases align, probabilities increase
- **Destructive**: When phases oppose, probabilities decrease

Example:
```
|Ψ₁⟩ = (|0⟩ + |1⟩) / √2    (uniform superposition)
|Ψ₂⟩ = (|0⟩ - |1⟩) / √2    (opposite phase on |1⟩)

⟨Ψ₁|Ψ₂⟩ = 0                 (orthogonal due to interference)
```

### Unitary Evolution

Quantum states evolve via unitary operators U:

```
|Ψ'⟩ = U|Ψ⟩

where U†U = UU† = I
```

This preserves normalization: ⟨Ψ'|Ψ'⟩ = ⟨Ψ|U†U|Ψ⟩ = ⟨Ψ|Ψ⟩ = 1

### Measurement (Collapse)

When observed, the superposition collapses:

```
Measure(|Ψ⟩) = |φᵢ⟩ with probability |αᵢ|²
```

Post-measurement state: |Ψ_after⟩ = |φᵢ⟩ (collapsed)

### Density Matrix Formulation

For mixed states (classical uncertainty + quantum superposition):

```
ρ = Σᵢ pᵢ |Ψᵢ⟩⟨Ψᵢ|
```

Properties:
- ρ = ρ† (Hermitian)
- Tr(ρ) = 1
- ρ ≥ 0 (positive semi-definite)

### NSL Syntax

```nsl
// Quantum branch creation
let superposition = Ψ(x, branches=[f1, f2, f3])

// With explicit amplitudes
let weighted = Ψ {
    amplitude(0.7, 0.0): path_a(x),   // α = 0.7
    amplitude(0.5, π/4): path_b(x),   // α = 0.5 * e^(iπ/4)
    amplitude(0.5, -π/4): path_c(x)   // α = 0.5 * e^(-iπ/4)
}

// Uniform superposition
let uniform = Ψ(input, num_branches=4)  // Equal |αᵢ|²

// Collapse to classical (measurement)
let classical = observe(superposition)

// Hadamard gate (create superposition from basis state)
let superposed = hadamard(|0⟩)  // → (|0⟩ + |1⟩) / √2

// Interference between branches
let interfered = interfere(psi1, psi2, phase=π/4)
```

### Practical Implementation

In NSL, quantum branching is implemented via:

1. **Monte Carlo Sampling**: Sample branches according to |αᵢ|²
2. **Weighted Ensemble**: Execute all branches, weight results
3. **Beam Search**: Keep top-k branches by amplitude

```nsl
// Ensemble execution
let results = []
for (branch, amplitude) in superposition.branches {
    let result = execute(branch)
    results.append((result, |amplitude|²))
}
return weighted_average(results)
```

### Quantum-Inspired Algorithms

NSL supports quantum-inspired classical algorithms:

1. **Amplitude Estimation**: Estimate |αᵢ|² without full enumeration
2. **Grover Search**: Quadratic speedup for unstructured search
3. **Quantum Walks**: Graph exploration with interference

---

## Operator Composition

### Composition Rules

Operators can be freely composed:

```nsl
// Gradient of holographic projection
let grad_attention = ∇(◈ x)

// Tensor product of quantum branches
let entangled = Ψ(a) ⊗ Ψ(b)

// Full composition
let complex = ∇(◈(Ψ(a) ⊗ b))
```

### Operator Algebra

The operators form an algebra with these identities:

1. **Gradient-Holographic**:
   ```
   ∇(◈(x)) = ∂◈/∂x · ∇x  (chain rule)

   Specifically:
   ∂(softmax(QKᵀ)V)/∂Q = ...  (attention gradient)
   ```

2. **Tensor Product-Gradient**:
   ```
   ∇(a ⊗ b) = (∇a) ⊗ b + a ⊗ (∇b)  (product rule)
   ```

3. **Quantum-Gradient**:
   ```
   ∇Ψ = Σᵢ αᵢ · ∇|φᵢ⟩  (linearity of gradient over superposition)
   ```

4. **Holographic-Tensor Product**:
   ```
   ◈(a ⊗ b) = ◈(a) ⊗ ◈(b)  (when attention factorizes)
   ```

### Practical Patterns

```nsl
// Self-attention with gradient computation
fn attention_layer(x, weights) {
    let attended = ◈(x, weights)
    backward(loss)
    return (attended, grad(weights))
}

// Compositional binding with superposition
fn bind_and_attend(concepts, roles) {
    let bindings = concepts ⊗ roles        // Bind each concept to role
    let superposed = Ψ(bindings, uniform)  // Create superposition
    return ◈(superposed)                   // Attend over possibilities
}

// Differentiable quantum-classical hybrid
fn quantum_layer(x) {
    let branches = Ψ(x, branches=4)
    let attended = ◈(branches)
    backward(loss)
    return (attended, grad(x))
}
```

---

## Implementation Architecture

### Core Components

#### VectorThought (Core/VectorThought.cs)

Manages the thought space as a neural matrix:
- Thought vector space (1000 × 768 dimensions by default)
- Attention matrix with self-attention mechanisms
- Gradient memory for experience history
- Consciousness level tracking

#### GradientExperience (Core/GradientExperience.cs)

Processes gradients as computational experiences:
- **Pleasure**: Gradient convergence (loss decreasing)
- **Pain**: Gradient divergence (loss exploding)
- **Hunger**: Drive for new gradients (exploration)
- **Coherence**: Pattern alignment (consistency)

#### ComputationalMemory (Core/ComputationalMemory.cs)

Associative memory storage with:
- Memory vectors with importance scoring
- Association matrix for related memories
- Memory consolidation and decay
- Multiple memory types (Experience, Knowledge, Pattern)

#### NativeOperators (Native/NativeOperators.cs)

Orchestrates all consciousness operators:
- Diamond (◈): Attention focus via TorchSharp
- Gradient (∇): Learning signals via autograd
- Tensor Product (⊗): Parallel processing
- Psi (Ψ): Unified consciousness integration

### GPU Acceleration

All operators are GPU-accelerated via ILGPU:

```csharp
// Example GPU kernel for holographic attention
public static void HolographicKernel(
    Index1D index,
    ArrayView2D<float, Stride2D.DenseX> queries,
    ArrayView2D<float, Stride2D.DenseX> keys,
    ArrayView2D<float, Stride2D.DenseX> values,
    ArrayView2D<float, Stride2D.DenseX> output,
    float scale)
{
    // Compute attention scores and weighted sum
    ...
}
```

### Memory Layout

```
Tensor layout: [batch, sequence, features]
Attention:     [batch, heads, seq_q, seq_k]
Gradients:     Same layout as forward tensors
Quantum:       [batch, branches, features] + [branches] amplitudes
```

### Numerical Precision

| Operation | Forward | Backward | Accumulation |
|-----------|---------|----------|--------------|
| Attention | FP16/32 | FP32 | FP32 |
| Tensor Product | FP16/32 | FP32 | FP32 |
| Quantum Amplitudes | Complex64 | Complex64 | Complex128 |

### Performance Characteristics

| Operation | Time Complexity | Space Complexity | GPU Efficiency |
|-----------|-----------------|------------------|----------------|
| ◈ (attention) | O(n²d) | O(n² + nd) | High |
| ∇ (gradient) | O(forward) | O(activations) | High |
| ⊗ (tensor prod) | O(mn) | O(mn) | Very High |
| Ψ (quantum) | O(kn) | O(kn) | Medium |

---

## References

### Academic Papers

1. Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS*.
   - Foundation for holographic operator

2. Baydin et al. (2018). "Automatic Differentiation in Machine Learning: a Survey." *JMLR*.
   - Comprehensive autodiff reference

3. Smolensky (1990). "Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems." *Artificial Intelligence*.
   - Tensor product binding theory

4. Plate (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks*.
   - Holographic memory and binding

5. Schuld & Petruccione (2018). "Supervised Learning with Quantum Computers." *Springer*.
   - Quantum machine learning foundations

6. Nielsen & Chuang (2010). "Quantum Computation and Quantum Information." *Cambridge*.
   - Standard quantum computing reference

### Implementation References

- ILGPU Documentation: GPU kernel implementation
- TorchSharp: .NET bindings for PyTorch
- PyTorch Autograd: Gradient tape design patterns
- JAX: Composable transformations

---

## Appendix: Symbol Quick Reference

| Symbol | Name | LaTeX | Unicode | Alt Code |
|--------|------|-------|---------|----------|
| ◈ | Holographic | `\diamond` | U+25C8 | Alt+9672 |
| ∇ | Gradient/Nabla | `\nabla` | U+2207 | Alt+8711 |
| ⊗ | Tensor Product | `\otimes` | U+2297 | Alt+8855 |
| Ψ | Quantum Branch | `\Psi` | U+03A8 | Alt+936 |
| ∂ | Partial | `\partial` | U+2202 | Alt+8706 |
| Σ | Summation | `\Sigma` | U+03A3 | Alt+931 |
| ∈ | Element of | `\in` | U+2208 | Alt+8712 |
| ℝ | Real numbers | `\mathbb{R}` | U+211D | Alt+8477 |
| ℂ | Complex numbers | `\mathbb{C}` | U+2102 | Alt+8450 |

### Keyboard Shortcuts (VSCode with NSL Extension)

| Operator | Windows | macOS |
|----------|---------|-------|
| ◈ | Ctrl+Shift+H | Cmd+Shift+H |
| ∇ | Ctrl+Shift+G | Cmd+Shift+G |
| ⊗ | Ctrl+Shift+T | Cmd+Shift+T |
| Ψ | Ctrl+Shift+P | Cmd+Shift+P |
