using System.Numerics;
using NSL.Core.Tokens;

namespace NSL.Core.Ast;

// ===== Consciousness Operations =====

/// <summary>
/// Consciousness operator types
/// </summary>
public enum ConsciousnessOperator
{
    Holographic,  // ◈
    Gradient,     // ∇
    Parallel      // ⊗
}

/// <summary>
/// Consciousness operation expression
/// </summary>
public record ConsciousnessOperation(
    ConsciousnessOperator Operator,
    Expression Operand,
    SourceLocation Location) : Expression(Location);

// ===== Quantum Operations =====

/// <summary>
/// Quantum superposition expression
/// </summary>
public record QuantumSuperposition(
    IReadOnlyList<Expression> States,
    SourceLocation Location) : Expression(Location);

// ===== Lambda Expressions =====

/// <summary>
/// Lambda expression
/// </summary>
public record LambdaExpression(
    IReadOnlyList<Parameter> Parameters,
    Expression Body,
    SourceLocation Location) : Expression(Location);

// ===== Chain Operations =====

/// <summary>
/// Chain operation expression (::)
/// </summary>
public record ChainOperation(
    IReadOnlyList<Expression> Operations,
    SourceLocation Location) : Expression(Location);

// ===== Pattern Matching =====

/// <summary>
/// Match expression
/// </summary>
public record MatchExpression(
    Expression Value,
    IReadOnlyList<MatchCase> Cases,
    Expression? DefaultCase,
    SourceLocation Location) : Expression(Location);

/// <summary>
/// Match case
/// </summary>
public record MatchCase(
    Pattern Pattern,
    Expression Result,
    SourceLocation Location);

/// <summary>
/// Base class for patterns
/// </summary>
public abstract record Pattern(SourceLocation Location);

/// <summary>
/// Literal pattern
/// </summary>
public record LiteralPattern(
    Expression Literal,
    SourceLocation Location) : Pattern(Location);

/// <summary>
/// Identifier pattern (binds a variable)
/// </summary>
public record IdentifierPattern(
    string Name,
    SourceLocation Location) : Pattern(Location);

/// <summary>
/// Wildcard pattern (_)
/// </summary>
public record WildcardPattern(SourceLocation Location) : Pattern(Location);

// ===== Advanced Consciousness Types =====

/// <summary>
/// Represents a holographic data structure
/// </summary>
public record HolographicData(
    object Value,
    Dictionary<string, object> Dimensions,
    double CoherenceLevel) : IConsciousnessData;

/// <summary>
/// Represents a consciousness gradient
/// </summary>
public record GradientData(
    double Value,
    double Direction,
    double Magnitude,
    DateTime Timestamp) : IConsciousnessData;

/// <summary>
/// Represents parallel processing state
/// </summary>
public record ParallelData(
    IReadOnlyList<object> States,
    Dictionary<int, double> Weights,
    bool IsSynchronized) : IConsciousnessData;

/// <summary>
/// Interface for consciousness data types
/// </summary>
public interface IConsciousnessData
{
    // Marker interface for consciousness-aware data structures
}

// ===== Quantum Types =====

/// <summary>
/// Represents a quantum state
/// </summary>
public record QuantumState(
    Dictionary<object, Complex> Amplitudes,
    bool IsCollapsed,
    object? CollapsedValue);

// ===== Special Operations =====

/// <summary>
/// Represents a consciousness fragment for memory operations
/// </summary>
public record MemoryFragment(
    string Id,
    object Content,
    double Salience,
    DateTime AccessTime,
    SourceLocation Location) : Expression(Location);

/// <summary>
/// Built-in function call for NSL-specific operations
/// </summary>
public record BuiltInFunction(
    string Name,
    IReadOnlyList<Expression> Arguments,
    SourceLocation Location) : Expression(Location);