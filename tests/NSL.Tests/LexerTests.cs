using NSL.Lexer;
using NSL.Core.Tokens;
using FluentAssertions;

namespace NSL.Tests;

/// <summary>
/// Comprehensive tests for the NSL Lexer
/// </summary>
public class LexerTests
{
    #region Basic Tokens

    [Fact]
    public void Tokenize_EmptyInput_ReturnsEndOfFile()
    {
        var lexer = new NSLLexer("");
        var tokens = lexer.Tokenize();

        tokens.Should().HaveCount(1);
        tokens[0].Type.Should().Be(TokenType.EndOfFile);
    }

    [Fact]
    public void Tokenize_WhitespaceOnly_ReturnsEndOfFile()
    {
        var lexer = new NSLLexer("   \t\n  ");
        var tokens = lexer.Tokenize();

        tokens.Should().HaveCount(1);
        tokens[0].Type.Should().Be(TokenType.EndOfFile);
    }

    [Theory]
    [InlineData("42", TokenType.Integer)]
    [InlineData("3.14", TokenType.Number)]
    [InlineData("1e10", TokenType.Number)]
    [InlineData("2.5e-3", TokenType.Number)]
    public void Tokenize_Numbers_ReturnsCorrectType(string input, TokenType expectedType)
    {
        var lexer = new NSLLexer(input);
        var tokens = lexer.Tokenize();

        tokens.Should().HaveCount(2); // Number + EOF
        tokens[0].Type.Should().Be(expectedType);
        tokens[0].Value.Should().Be(input);
    }

    [Fact]
    public void Tokenize_String_ReturnsStringToken()
    {
        var lexer = new NSLLexer("\"hello world\"");
        var tokens = lexer.Tokenize();

        tokens.Should().HaveCount(2);
        tokens[0].Type.Should().Be(TokenType.String);
        tokens[0].Value.Should().Be("hello world");
    }

    [Fact]
    public void Tokenize_StringWithEscapes_HandlesEscapeSequences()
    {
        var lexer = new NSLLexer("\"hello\\nworld\\t!\"");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.String);
        tokens[0].Value.Should().Be("hello\nworld\t!");
    }

    [Fact]
    public void Tokenize_Identifier_ReturnsIdentifierToken()
    {
        var lexer = new NSLLexer("myVariable");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.Identifier);
        tokens[0].Value.Should().Be("myVariable");
    }

    #endregion

    #region Keywords

    [Theory]
    [InlineData("let", TokenType.Let)]
    [InlineData("mut", TokenType.Mut)]
    [InlineData("const", TokenType.Const)]
    [InlineData("if", TokenType.If)]
    [InlineData("else", TokenType.Else)]
    [InlineData("while", TokenType.While)]
    [InlineData("for", TokenType.For)]
    [InlineData("in", TokenType.In)]
    [InlineData("fn", TokenType.Function)]
    [InlineData("function", TokenType.Function)]
    [InlineData("return", TokenType.Return)]
    [InlineData("break", TokenType.Break)]
    [InlineData("continue", TokenType.Continue)]
    [InlineData("match", TokenType.Match)]
    [InlineData("case", TokenType.Case)]
    [InlineData("true", TokenType.True)]
    [InlineData("false", TokenType.False)]
    [InlineData("null", TokenType.Null)]
    [InlineData("and", TokenType.And)]
    [InlineData("or", TokenType.Or)]
    [InlineData("not", TokenType.Not)]
    [InlineData("async", TokenType.Async)]
    [InlineData("await", TokenType.Await)]
    [InlineData("import", TokenType.Import)]
    [InlineData("export", TokenType.Export)]
    [InlineData("trait", TokenType.Trait)]
    [InlineData("impl", TokenType.Impl)]
    public void Tokenize_Keywords_ReturnsCorrectTokenType(string keyword, TokenType expectedType)
    {
        var lexer = new NSLLexer(keyword);
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(expectedType);
    }

    #endregion

    #region Operators

    [Theory]
    [InlineData("+", TokenType.Plus)]
    [InlineData("-", TokenType.Minus)]
    [InlineData("*", TokenType.Multiply)]
    [InlineData("/", TokenType.Divide)]
    [InlineData("//", TokenType.IntegerDivide)]
    [InlineData("%", TokenType.Percent)]
    [InlineData("**", TokenType.Power)]
    [InlineData("=", TokenType.Assign)]
    [InlineData("==", TokenType.Equal)]
    [InlineData("!=", TokenType.NotEqual)]
    [InlineData("<", TokenType.Less)]
    [InlineData("<=", TokenType.LessEqual)]
    [InlineData(">", TokenType.Greater)]
    [InlineData(">=", TokenType.GreaterEqual)]
    [InlineData("&&", TokenType.LogicalAnd)]
    [InlineData("||", TokenType.LogicalOr)]
    [InlineData("!", TokenType.LogicalNot)]
    [InlineData("|>", TokenType.PipeArrow)]
    [InlineData("=>", TokenType.FatArrow)]
    [InlineData("?.", TokenType.QuestionDot)]
    [InlineData("??", TokenType.QuestionQuestion)]
    [InlineData("..", TokenType.DotDot)]
    [InlineData("..=", TokenType.DotDotEqual)]
    [InlineData("::", TokenType.Chain)]
    public void Tokenize_Operators_ReturnsCorrectTokenType(string op, TokenType expectedType)
    {
        var lexer = new NSLLexer(op);
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(expectedType);
    }

    #endregion

    #region Consciousness Operators

    [Theory]
    [InlineData("◈", TokenType.Holographic)]
    [InlineData("∇", TokenType.Gradient)]
    [InlineData("⊗", TokenType.TensorProduct)]
    [InlineData("Ψ", TokenType.Psi)]
    [InlineData("μ", TokenType.Mu)]
    [InlineData("σ", TokenType.Sigma)]
    [InlineData("↓", TokenType.Collapse)]
    [InlineData("≈", TokenType.Similarity)]
    [InlineData("≉", TokenType.Dissimilarity)]
    [InlineData("∫", TokenType.Integral)]
    [InlineData("±", TokenType.PlusMinus)]
    [InlineData("λ", TokenType.Lambda)]
    [InlineData("→", TokenType.Arrow)]
    [InlineData("←", TokenType.LeftArrow)]
    public void Tokenize_ConsciousnessOperators_ReturnsCorrectTokenType(string op, TokenType expectedType)
    {
        var lexer = new NSLLexer(op);
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(expectedType);
    }

    [Theory]
    // NOTE: Many ASCII aliases were removed from lexer to avoid conflicts with common function names.
    // Only short aliases "mem" and "introspect" are kept. Use Unicode operators directly: μ σ ↓ ≈ ∫ ±
    [InlineData("mem", TokenType.Mu)]
    [InlineData("introspect", TokenType.Sigma)]
    public void Tokenize_ConsciousnessAliases_ReturnsCorrectTokenType(string alias, TokenType expectedType)
    {
        var lexer = new NSLLexer(alias);
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(expectedType);
    }

    #endregion

    #region Delimiters

    [Theory]
    [InlineData("(", TokenType.LeftParen)]
    [InlineData(")", TokenType.RightParen)]
    [InlineData("[", TokenType.LeftBracket)]
    [InlineData("]", TokenType.RightBracket)]
    [InlineData("{", TokenType.LeftBrace)]
    [InlineData("}", TokenType.RightBrace)]
    [InlineData(",", TokenType.Comma)]
    [InlineData(";", TokenType.Semicolon)]
    [InlineData(":", TokenType.Colon)]
    [InlineData(".", TokenType.Dot)]
    [InlineData("@", TokenType.AtSign)]
    public void Tokenize_Delimiters_ReturnsCorrectTokenType(string delimiter, TokenType expectedType)
    {
        var lexer = new NSLLexer(delimiter);
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(expectedType);
    }

    #endregion

    #region Comments

    [Fact]
    public void Tokenize_HashComment_SkipsComment()
    {
        var lexer = new NSLLexer("# This is a comment\n42");
        var tokens = lexer.Tokenize();

        // Should have Integer(42) and EOF, comment is skipped
        tokens.Should().HaveCount(2);
        tokens[0].Type.Should().Be(TokenType.Integer);
        tokens[0].Value.Should().Be("42");
    }

    [Fact]
    public void Tokenize_MultiLineComment_SkipsComment()
    {
        var lexer = new NSLLexer("/* multi\nline\ncomment */ 42");
        var tokens = lexer.Tokenize();

        tokens.Should().HaveCount(2);
        tokens[0].Type.Should().Be(TokenType.Integer);
    }

    #endregion

    #region Complex Expressions

    [Fact]
    public void Tokenize_VariableDeclaration_ReturnsCorrectTokens()
    {
        var lexer = new NSLLexer("let x = 42");
        var tokens = lexer.Tokenize();

        tokens.Should().HaveCount(5); // let, x, =, 42, EOF
        tokens[0].Type.Should().Be(TokenType.Let);
        tokens[1].Type.Should().Be(TokenType.Identifier);
        tokens[1].Value.Should().Be("x");
        tokens[2].Type.Should().Be(TokenType.Assign);
        tokens[3].Type.Should().Be(TokenType.Integer);
        tokens[3].Value.Should().Be("42");
    }

    [Fact]
    public void Tokenize_FunctionDefinition_ReturnsCorrectTokens()
    {
        var lexer = new NSLLexer("fn add(a, b) { return a + b }");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.Function);
        tokens[1].Type.Should().Be(TokenType.Identifier);
        tokens[1].Value.Should().Be("add");
        tokens[2].Type.Should().Be(TokenType.LeftParen);
    }

    [Fact]
    public void Tokenize_ConsciousnessExpression_ReturnsCorrectTokens()
    {
        var lexer = new NSLLexer("let h = ◈[data]");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.Let);
        tokens[1].Type.Should().Be(TokenType.Identifier);
        tokens[2].Type.Should().Be(TokenType.Assign);
        tokens[3].Type.Should().Be(TokenType.Holographic);
        tokens[4].Type.Should().Be(TokenType.LeftBracket);
        tokens[5].Type.Should().Be(TokenType.Identifier);
        tokens[6].Type.Should().Be(TokenType.RightBracket);
    }

    [Fact]
    public void Tokenize_PipelineExpression_ReturnsCorrectTokens()
    {
        var lexer = new NSLLexer("data |> transform |> filter");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.Identifier);
        tokens[1].Type.Should().Be(TokenType.PipeArrow);
        tokens[2].Type.Should().Be(TokenType.Identifier);
        tokens[3].Type.Should().Be(TokenType.PipeArrow);
        tokens[4].Type.Should().Be(TokenType.Identifier);
    }

    [Fact]
    public void Tokenize_RangeExpression_ReturnsCorrectTokens()
    {
        var lexer = new NSLLexer("0..10");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.Integer);
        tokens[1].Type.Should().Be(TokenType.DotDot);
        tokens[2].Type.Should().Be(TokenType.Integer);
    }

    [Fact]
    public void Tokenize_InclusiveRangeExpression_ReturnsCorrectTokens()
    {
        var lexer = new NSLLexer("0..=10");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.Integer);
        tokens[1].Type.Should().Be(TokenType.DotDotEqual);
        tokens[2].Type.Should().Be(TokenType.Integer);
    }

    #endregion

    #region Line and Column Tracking

    [Fact]
    public void Tokenize_TracksLineNumbers()
    {
        var lexer = new NSLLexer("let x = 1\nlet y = 2");
        var tokens = lexer.Tokenize();

        // First line tokens
        tokens[0].Line.Should().Be(1); // let
        tokens[1].Line.Should().Be(1); // x

        // Second line tokens
        tokens[4].Line.Should().Be(2); // let
        tokens[5].Line.Should().Be(2); // y
    }

    [Fact]
    public void Tokenize_TracksColumnNumbers()
    {
        var lexer = new NSLLexer("let x = 42");
        var tokens = lexer.Tokenize();

        tokens[0].Column.Should().Be(1); // let starts at column 1
        tokens[1].Column.Should().Be(5); // x starts at column 5
        tokens[2].Column.Should().Be(7); // = starts at column 7
        tokens[3].Column.Should().Be(9); // 42 starts at column 9
    }

    #endregion

    #region Error Handling

    [Fact]
    public void Tokenize_UnterminatedString_ReturnsInvalidToken()
    {
        var lexer = new NSLLexer("\"unterminated");
        var tokens = lexer.Tokenize();

        tokens[0].Type.Should().Be(TokenType.Invalid);
    }

    #endregion
}
