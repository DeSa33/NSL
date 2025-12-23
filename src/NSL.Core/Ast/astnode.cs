using NSL.Core.Tokens;

namespace NSL.Core.Ast;

/// <summary>
/// Base class for all AST nodes
/// </summary>
public abstract record AstNode(SourceLocation Location);

/// <summary>
/// Base class for all expressions
/// </summary>
public abstract record Expression(SourceLocation Location) : AstNode(Location);

/// <summary>
/// Base class for all statements
/// </summary>
public abstract record Statement(SourceLocation Location) : AstNode(Location);

/// <summary>
/// Represents a program (root node)
/// </summary>
public record Program(
    IReadOnlyList<Statement> Statements,
    SourceLocation Location) : AstNode(Location);

// ===== Literal Expressions =====

/// <summary>
/// Numeric literal expression
/// </summary>
public record NumberLiteral(
    double Value,
    SourceLocation Location) : Expression(Location);

/// <summary>
/// String literal expression
/// </summary>
public record StringLiteral(
    string Value,
    SourceLocation Location) : Expression(Location);

/// <summary>
/// Boolean literal expression
/// </summary>
public record BooleanLiteral(
    bool Value,
    SourceLocation Location) : Expression(Location);

/// <summary>
/// Null literal expression
/// </summary>
public record NullLiteral(SourceLocation Location) : Expression(Location);

/// <summary>
/// Identifier expression
/// </summary>
public record Identifier(
    string Name,
    SourceLocation Location) : Expression(Location);

// ===== Binary Operations =====

/// <summary>
/// Binary operator types
/// </summary>
public enum BinaryOperator
{
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    
    // Comparison
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    
    // Logical
    And,
    Or
}

/// <summary>
/// Binary operation expression
/// </summary>
public record BinaryOperation(
    Expression Left,
    BinaryOperator Operator,
    Expression Right,
    SourceLocation Location) : Expression(Location);

// ===== Unary Operations =====

/// <summary>
/// Unary operator types
/// </summary>
public enum UnaryOperator
{
    Plus,
    Minus,
    Not
}

/// <summary>
/// Unary operation expression
/// </summary>
public record UnaryOperation(
    UnaryOperator Operator,
    Expression Operand,
    SourceLocation Location) : Expression(Location);

// ===== Assignment =====

/// <summary>
/// Assignment expression
/// </summary>
public record Assignment(
    string Name,
    Expression Value,
    SourceLocation Location) : Expression(Location);

// ===== Arrays =====

/// <summary>
/// Array literal expression
/// </summary>
public record ArrayLiteral(
    IReadOnlyList<Expression> Elements,
    SourceLocation Location) : Expression(Location);

/// <summary>
/// Array access expression
/// </summary>
public record ArrayAccess(
    Expression Array,
    Expression Index,
    SourceLocation Location) : Expression(Location);

// ===== Function Calls =====

/// <summary>
/// Function call expression
/// </summary>
public record FunctionCall(
    Expression Function,
    IReadOnlyList<Expression> Arguments,
    SourceLocation Location) : Expression(Location);

// ===== Statements =====

/// <summary>
/// Expression statement
/// </summary>
public record ExpressionStatement(
    Expression Expression,
    SourceLocation Location) : Statement(Location);

/// <summary>
/// Block statement
/// </summary>
public record BlockStatement(
    IReadOnlyList<Statement> Statements,
    SourceLocation Location) : Statement(Location);

/// <summary>
/// If statement
/// </summary>
public record IfStatement(
    Expression Condition,
    Statement ThenBranch,
    Statement? ElseBranch,
    SourceLocation Location) : Statement(Location);

/// <summary>
/// While statement
/// </summary>
public record WhileStatement(
    Expression Condition,
    Statement Body,
    SourceLocation Location) : Statement(Location);

/// <summary>
/// Return statement
/// </summary>
public record ReturnStatement(
    Expression? Value,
    SourceLocation Location) : Statement(Location);

/// <summary>
/// Function declaration
/// </summary>
public record FunctionDeclaration(
    string Name,
    IReadOnlyList<Parameter> Parameters,
    Statement Body,
    SourceLocation Location) : Statement(Location);

/// <summary>
/// Function parameter
/// </summary>
public record Parameter(string Name, SourceLocation Location);