namespace NSL.Core.Tokens;

/// <summary>
/// Represents a token produced by the lexer
/// </summary>
public record Token(
    TokenType Type,
    string Value,
    int Position,
    int Line,
    int Column,
    string? FileName = null)
{
    /// <summary>
    /// Creates an invalid token with error information
    /// </summary>
    public static Token Invalid(string errorMessage, int position, int line, int column, string? fileName = null)
        => new(TokenType.Invalid, errorMessage, position, line, column, fileName);
    
    /// <summary>
    /// Creates an end-of-file token
    /// </summary>
    public static Token EndOfFile(int position, int line, int column, string? fileName = null)
        => new(TokenType.EndOfFile, string.Empty, position, line, column, fileName);
    
    /// <summary>
    /// Gets the location information for error reporting
    /// </summary>
    public SourceLocation Location => new(Position, Line, Column, FileName);
    
    /// <summary>
    /// Returns a human-readable string representation
    /// </summary>
    public override string ToString() => Type switch
    {
        TokenType.String => $"{Type}(\"{Value}\")",
        TokenType.EndOfFile => "EOF",
        _ => string.IsNullOrEmpty(Value) ? Type.ToString() : $"{Type}({Value})"
    };
}

/// <summary>
/// Represents a location in source code
/// </summary>
public record SourceLocation(int Position, int Line, int Column, string? FileName = null)
{
    /// <summary>
    /// Returns a human-readable location string
    /// </summary>
    public override string ToString() => 
        FileName != null ? $"{FileName}:{Line}:{Column}" : $"{Line}:{Column}";
}