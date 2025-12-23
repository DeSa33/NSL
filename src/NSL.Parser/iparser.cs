using NSL.Core.Ast;
using NSL.Core.Tokens;

namespace NSL.Parser;

/// <summary>
/// Interface for the NSL parser
/// </summary>
public interface IParser
{
    /// <summary>
    /// Parses a list of tokens into an AST
    /// </summary>
    /// <param name="tokens">The tokens to parse</param>
    /// <returns>The root AST node (Program)</returns>
    Program Parse(IReadOnlyList<Token> tokens);
    
    /// <summary>
    /// Parses a single expression from tokens
    /// </summary>
    /// <param name="tokens">The tokens to parse</param>
    /// <returns>The parsed expression</returns>
    Expression ParseExpression(IReadOnlyList<Token> tokens);
}

/// <summary>
/// Parser exception for syntax errors
/// </summary>
public class ParseException : Exception
{
    /// <summary>API member</summary>
    public SourceLocation Location { get; }
    /// <summary>API member</summary>
    public string TokenValue { get; }
    
    /// <summary>API member</summary>
    public ParseException(string message, SourceLocation location, string tokenValue) 
        : base($"{message} at {location}")
    {
        Location = location;
        TokenValue = tokenValue;
    }
}