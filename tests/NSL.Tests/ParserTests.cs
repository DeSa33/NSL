using NSL.Lexer;
using NSL.Parser;
using NSL.Core.AST;
using NSL.Core.Tokens;
using FluentAssertions;

namespace NSL.Tests;

/// <summary>
/// Tests for the NSL Parser
/// </summary>
public class ParserTests
{
    private NSLASTNode Parse(string code)
    {
        var lexer = new NSLLexer(code);
        var tokens = lexer.Tokenize();
        var parser = new NSLParser();
        return parser.Parse(tokens);
    }

    #region Literals

    [Fact]
    public void Parse_IntegerLiteral_ReturnsLiteralNode()
    {
        var ast = Parse("42");
        ast.Should().BeOfType<NSLLiteralNode>();
    }

    [Fact]
    public void Parse_FloatLiteral_ReturnsLiteralNode()
    {
        var ast = Parse("3.14");
        ast.Should().BeOfType<NSLLiteralNode>();
    }

    [Fact]
    public void Parse_StringLiteral_ReturnsLiteralNode()
    {
        var ast = Parse("\"hello\"");
        ast.Should().BeOfType<NSLLiteralNode>();
    }

    [Fact]
    public void Parse_BooleanTrue_ReturnsLiteralNode()
    {
        var ast = Parse("true");
        ast.Should().BeOfType<NSLLiteralNode>();
    }

    [Fact]
    public void Parse_BooleanFalse_ReturnsLiteralNode()
    {
        var ast = Parse("false");
        ast.Should().BeOfType<NSLLiteralNode>();
    }

    [Fact]
    public void Parse_Null_ReturnsLiteralNode()
    {
        var ast = Parse("null");
        ast.Should().BeOfType<NSLLiteralNode>();
    }

    #endregion

    #region Variable Declarations

    [Fact]
    public void Parse_ImmutableVariableDeclaration_ReturnsVariableNode()
    {
        var ast = Parse("let x = 42");
        ast.Should().BeOfType<NSLVariableDeclarationNode>();
        var decl = (NSLVariableDeclarationNode)ast;
        decl.Name.Should().Be("x");
        decl.IsMutable.Should().BeFalse();
    }

    [Fact]
    public void Parse_MutableVariableDeclaration_ReturnsVariableNode()
    {
        var ast = Parse("mut y = 10");
        ast.Should().BeOfType<NSLVariableDeclarationNode>();
        var decl = (NSLVariableDeclarationNode)ast;
        decl.Name.Should().Be("y");
        decl.IsMutable.Should().BeTrue();
    }

    #endregion

    #region Binary Expressions

    [Fact]
    public void Parse_Addition_ReturnsBinaryNode()
    {
        var ast = Parse("1 + 2");
        ast.Should().BeOfType<NSLBinaryOperationNode>();
    }

    [Fact]
    public void Parse_Multiplication_ReturnsBinaryNode()
    {
        var ast = Parse("3 * 4");
        ast.Should().BeOfType<NSLBinaryOperationNode>();
    }

    [Fact]
    public void Parse_Comparison_ReturnsBinaryNode()
    {
        var ast = Parse("5 > 3");
        ast.Should().BeOfType<NSLBinaryOperationNode>();
    }

    #endregion

    #region Unary Expressions

    [Fact]
    public void Parse_NegativeNumber_ReturnsUnaryNode()
    {
        var ast = Parse("-42");
        ast.Should().BeOfType<NSLUnaryOperationNode>();
    }

    [Fact]
    public void Parse_LogicalNot_ReturnsUnaryNode()
    {
        var ast = Parse("not true");
        ast.Should().BeOfType<NSLUnaryOperationNode>();
    }

    #endregion

    #region Functions

    [Fact]
    public void Parse_FunctionDefinition_ReturnsFunctionNode()
    {
        var ast = Parse("fn add(a, b) { return a + b }");
        ast.Should().BeOfType<NSLFunctionNode>();
        var func = (NSLFunctionNode)ast;
        func.Name.Should().Be("add");
    }

    [Fact]
    public void Parse_AsyncFunction_ReturnsAsyncFunctionNode()
    {
        var ast = Parse("async fn fetch() { return 1 }");
        ast.Should().BeOfType<NSLAsyncFunctionNode>();
    }

    #endregion

    #region Control Flow

    [Fact]
    public void Parse_IfStatement_ReturnsIfNode()
    {
        var ast = Parse("if (x > 0) { return 1 }");
        ast.Should().BeOfType<NSLIfNode>();
    }

    [Fact]
    public void Parse_WhileLoop_ReturnsWhileNode()
    {
        var ast = Parse("while (x > 0) { x = x - 1 }");
        ast.Should().BeOfType<NSLWhileNode>();
    }

    [Fact]
    public void Parse_ForLoop_ReturnsForNode()
    {
        var ast = Parse("for i in range(10) { print(i) }");
        ast.Should().BeOfType<NSLForNode>();
    }

    #endregion

    #region Arrays

    [Fact]
    public void Parse_ArrayLiteral_ReturnsArrayNode()
    {
        var ast = Parse("[1, 2, 3]");
        ast.Should().BeOfType<NSLArrayNode>();
    }

    [Fact]
    public void Parse_EmptyArray_ReturnsArrayNode()
    {
        var ast = Parse("[]");
        ast.Should().BeOfType<NSLArrayNode>();
    }

    [Fact]
    public void Parse_ArrayIndexing_ReturnsIndexNode()
    {
        var ast = Parse("arr[0]");
        ast.Should().BeOfType<NSLIndexNode>();
    }

    #endregion

    #region Function Calls

    [Fact]
    public void Parse_FunctionCall_ReturnsCallNode()
    {
        var ast = Parse("print(\"hello\")");
        ast.Should().BeOfType<NSLFunctionCallNode>();
    }

    #endregion

    #region Consciousness Operators

    [Fact]
    public void Parse_HolographicOperator_ParsesSuccessfully()
    {
        // Consciousness operators are parsed as unary operations
        var ast = Parse("◈[data]");
        ast.Should().NotBeNull();
    }

    [Fact]
    public void Parse_GradientOperator_ParsesSuccessfully()
    {
        var ast = Parse("∇[x]");
        ast.Should().NotBeNull();
    }

    [Fact]
    public void Parse_TensorProductOperator_ParsesSuccessfully()
    {
        // Tensor product with array syntax
        var ast = Parse("⊗[a, b]");
        ast.Should().NotBeNull();
    }

    [Fact]
    public void Parse_PsiOperator_ParsesSuccessfully()
    {
        var ast = Parse("Ψ[state]");
        ast.Should().NotBeNull();
    }

    #endregion

    #region Pipeline

    [Fact]
    public void Parse_PipelineExpression_ReturnsPipelineNode()
    {
        var ast = Parse("data |> transform");
        ast.Should().BeOfType<NSLPipelineNode>();
    }

    #endregion

    #region Pattern Matching

    [Fact]
    public void Parse_MatchExpression_ReturnsMatchNode()
    {
        var ast = Parse("match x { case 0 => \"zero\" case _ => \"other\" }");
        ast.Should().BeOfType<NSLMatchNode>();
    }

    #endregion

    #region Blocks

    [Fact]
    public void Parse_Block_ReturnsBlockNode()
    {
        var ast = Parse("{ let x = 1\n let y = 2 }");
        ast.Should().BeOfType<NSLBlockNode>();
    }

    #endregion

    #region Member Access

    [Fact]
    public void Parse_MemberAccess_ReturnsGetNode()
    {
        var ast = Parse("obj.property");
        ast.Should().BeOfType<NSLGetNode>();
    }

    [Fact]
    public void Parse_SafeNavigation_ReturnsSafeNavigationNode()
    {
        var ast = Parse("obj?.property");
        ast.Should().BeOfType<NSLSafeNavigationNode>();
    }

    #endregion
}
