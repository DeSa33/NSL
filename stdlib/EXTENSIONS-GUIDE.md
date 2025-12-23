# NSL Extensions Library Guide

## Overview

The NSL Extensions library (`nsl-extensions.nsl`) provides additional functionality for:
- GPU tensor operations (sum, mean, std, normalize, etc.)
- Consciousness operator helpers with clear semantics
- Neural network building blocks
- Error handling utilities
- Common array operations

## Installation

Load the library in your NSL script:
```nsl
# In REPL
.load E:\NSL.Interpreter\stdlib\nsl-extensions.nsl

# Or copy the functions you need into your script
```

---

## GPU Helper Functions

These functions work with arrays (typically from `gpu.to_cpu()`).

### Statistical Operations

```nsl
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

# Sum all elements
gpu_sum(data)        # Returns: 15

# Mean (average)
gpu_mean(data)       # Returns: 3

# Min and Max
gpu_min(data)        # Returns: 1
gpu_max(data)        # Returns: 5

# Standard deviation
gpu_std(data)        # Returns: ~1.414

# Variance
gpu_variance(data)   # Returns: 2
```

### Normalization

```nsl
let data = [10.0, 20.0, 30.0, 40.0, 50.0]

# Normalize to [0, 1] range
gpu_normalize(data)
# Returns: [0, 0.25, 0.5, 0.75, 1.0]

# Standardize to mean=0, std=1 (z-score)
gpu_standardize(data)
# Returns: [-1.414, -0.707, 0, 0.707, 1.414]
```

### Index Operations

```nsl
let data = [3.0, 1.0, 4.0, 1.0, 5.0]

# Index of maximum value
gpu_argmax(data)     # Returns: 4 (index of 5.0)

# Index of minimum value
gpu_argmin(data)     # Returns: 1 (index of first 1.0)
```

### Vector Operations

```nsl
let a = [1.0, 2.0, 3.0]
let b = [4.0, 5.0, 6.0]

# Dot product
gpu_dot(a, b)        # Returns: 32 (1*4 + 2*5 + 3*6)

# L2 norm (Euclidean length)
gpu_l2_norm(a)       # Returns: ~3.742

# Cosine similarity
gpu_cosine_similarity(a, b)  # Returns: ~0.974
```

---

## Consciousness Operator Helpers

Clear implementations of NSL's consciousness operators.

### Pipe Chain (|>)

```nsl
fn double(x) { return x * 2 }
fn add_one(x) { return x + 1 }

# Chain functions: 5 -> 10 -> 11
pipe_chain(5, [double, add_one])  # Returns: 11
```

### Holographic Focus (Attention)

```nsl
let data = [10.0, 20.0, 30.0]
let weights = [0.5, 0.3, 0.2]  # Attention weights

# Apply weighted focus
match holographic_focus(data, weights) {
    case ok(result) => print(result)
    case err(msg) => print("Error:", msg)
}
# Returns: weighted values proportional to attention
```

### Gradient Descent

```nsl
# Update parameters using gradients
let params = [1.0, 2.0, 3.0]
let grads = [0.1, 0.2, 0.3]
let lr = 0.01

gradient_step(params, grads, lr)
# Returns: [0.999, 1.998, 2.997]

# Full gradient descent optimization
fn loss_fn(p) {
    # Example: minimize sum of squares
    mut total = 0.0
    for val in p {
        total = total + val * val
    }
    return total
}

gradient_descent([5.0, 5.0], loss_fn, 0.1, 100)
# Returns: values close to [0, 0]
```

### Superposition (Quantum-like States)

```nsl
let states = ["up", "down", "left"]
let amplitudes = [0.5, 0.3, 0.2]

# Create superposition
match superposition_create(states, amplitudes) {
    case ok(superpos) => {
        # Collapse to single state (probabilistic)
        let result = superposition_collapse(superpos)
        print("Collapsed to:", result)
    }
    case err(msg) => print(msg)
}
```

### Memory Operations

```nsl
# Create memory store
let mem = memory_create()

# Store values
let mem = memory_store(mem, "x", 42)
let mem = memory_store(mem, "name", "NSL")

# Recall values
match memory_recall(mem, "x") {
    case some(val) => print("Found:", val)
    case none => print("Not found")
}
```

---

## Neural Network Helpers

### Activation Functions

```nsl
# ReLU (Rectified Linear Unit)
nn_relu(-5)      # Returns: 0
nn_relu(5)       # Returns: 5

# Leaky ReLU
nn_leaky_relu(-5, 0.01)  # Returns: -0.05

# Sigmoid (0 to 1)
nn_sigmoid(0)    # Returns: 0.5
nn_sigmoid(5)    # Returns: ~0.993

# Tanh (-1 to 1)
nn_tanh_approx(0)    # Returns: 0
```

### Softmax

```nsl
let logits = [1.0, 2.0, 3.0]
nn_softmax(logits)
# Returns: [0.09, 0.24, 0.67] (sums to 1.0)
```

### Loss Functions

```nsl
let predicted = [0.9, 0.1, 0.0]
let actual = [1.0, 0.0, 0.0]

# Mean Squared Error
nn_mse(predicted, actual)
# Returns: 0.0067

# Cross Entropy
nn_cross_entropy(predicted, actual)
# Returns: 0.105
```

### Accuracy

```nsl
let pred = [0.1, 0.8, 0.1]  # Predicts class 1
let true_label = [0, 1, 0]  # True class is 1

nn_accuracy(pred, true_label)  # Returns: 1.0 (correct)
```

---

## Utility Functions

### Array Creation

```nsl
# N evenly spaced values
linspace(0, 10, 5)   # Returns: [0, 2.5, 5, 7.5, 10]

# Range with step
arange(0, 5, 1)      # Returns: [0, 1, 2, 3, 4]

# Zeros and ones
zeros_array(5)       # Returns: [0, 0, 0, 0, 0]
ones_array(3)        # Returns: [1, 1, 1]

# Random values [0, 1)
random_array(4)      # Returns: [0.23, 0.87, 0.12, 0.56]
```

### Array Operations

```nsl
let arr = [1, 2, 3, 4, 5]

# Map
fn double(x) { return x * 2 }
map_array(arr, double)  # Returns: [2, 4, 6, 8, 10]

# Filter
fn is_even(x) { return x % 2 == 0 }
filter_array(arr, is_even)  # Returns: [2, 4]

# Reduce
fn add(a, b) { return a + b }
reduce_array(arr, add, 0)  # Returns: 15

# Zip
let a = [1, 2, 3]
let b = ["a", "b", "c"]
zip_arrays(a, b)  # Returns: [[1,"a"], [2,"b"], [3,"c"]]

# Enumerate
enumerate_array(["x", "y", "z"])
# Returns: [[0,"x"], [1,"y"], [2,"z"]]

# Reverse
reverse_array([1, 2, 3])  # Returns: [3, 2, 1]

# Flatten
flatten_one([[1,2], [3,4]])  # Returns: [1, 2, 3, 4]

# Take/Drop
take_n([1,2,3,4,5], 3)  # Returns: [1, 2, 3]
drop_n([1,2,3,4,5], 2)  # Returns: [3, 4, 5]
```

### Predicates

```nsl
all_true([true, true, true])   # Returns: true
all_true([true, false, true])  # Returns: false

any_true([false, true, false]) # Returns: true
any_true([false, false])       # Returns: false

fn positive(x) { return x > 0 }
count_if([1, -2, 3, -4, 5], positive)  # Returns: 3
find_index([1, -2, 3], positive)       # Returns: 0
```

### Value Clipping

```nsl
clip(15, 0, 10)   # Returns: 10 (clamped to max)
clip(-5, 0, 10)   # Returns: 0 (clamped to min)
clip(5, 0, 10)    # Returns: 5 (within range)
```

---

## Error Handling

### Validation

```nsl
# Validate array length
match validate_array_length(data, 1, 100) {
    case ok(arr) => process(arr)
    case err(msg) => print("Invalid:", msg)
}

# Validate value range
match validate_range(x, 0, 1, "probability") {
    case ok(val) => use(val)
    case err(msg) => print(msg)
}

# Assert condition
match require(x > 0, "x must be positive") {
    case ok(_) => continue_processing()
    case err(msg) => handle_error(msg)
}
```

---

## Complete Example: Simple Neural Network

```nsl
# A simple 2-layer neural network for XOR

fn forward(inputs, w1, b1, w2, b2) {
    # Hidden layer
    let h1 = nn_relu(gpu_dot(inputs, [w1[0], w1[1]]) + b1[0])
    let h2 = nn_relu(gpu_dot(inputs, [w1[2], w1[3]]) + b1[1])

    # Output layer
    let out = nn_sigmoid(h1 * w2[0] + h2 * w2[1] + b2)
    return out
}

# XOR training data
let X = [[0,0], [0,1], [1,0], [1,1]]
let y = [0, 1, 1, 0]

# Initialize weights (would use random in practice)
let w1 = [0.5, 0.5, 0.5, 0.5]
let b1 = [0.0, 0.0]
let w2 = [1.0, 1.0]
let b2 = 0.0

# Forward pass
for i in 0..4 {
    let pred = forward(X[i], w1, b1, w2, b2)
    print("Input:", X[i], "Predicted:", pred, "Actual:", y[i])
}
```

---

## Version

NSL Extensions v1.0.0
