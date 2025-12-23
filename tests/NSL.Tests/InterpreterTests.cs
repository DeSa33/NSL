using NSL.Lexer;
using NSL.Parser;
using NSL.Core;
using FluentAssertions;

namespace NSL.Tests;

/// <summary>
/// Tests for the NSL Interpreter
/// </summary>
public class InterpreterTests
{
    private readonly StringWriter _output;
    private readonly NSLInterpreter _interpreter;

    public InterpreterTests()
    {
        _output = new StringWriter();
        _interpreter = new NSLInterpreter();
        _interpreter.SetOutputWriter(_output);
    }

    private object? Execute(string code)
    {
        var lexer = new NSLLexer(code);
        var tokens = lexer.Tokenize();
        var parser = new NSLParser();
        var ast = parser.Parse(tokens);
        return ast.Accept(_interpreter);
    }

    private string GetOutput() => _output.ToString().Trim();

    #region Basic Expressions

    [Fact]
    public void Execute_IntegerLiteral_ReturnsValue()
    {
        var result = Execute("42");
        Convert.ToInt64(result).Should().Be(42);
    }

    [Fact]
    public void Execute_StringLiteral_ReturnsValue()
    {
        var result = Execute("\"hello\"");
        result.Should().Be("hello");
    }

    [Fact]
    public void Execute_BooleanTrue_ReturnsTrue()
    {
        var result = Execute("true");
        result.Should().Be(true);
    }

    [Fact]
    public void Execute_BooleanFalse_ReturnsFalse()
    {
        var result = Execute("false");
        result.Should().Be(false);
    }

    [Fact]
    public void Execute_Null_ReturnsNull()
    {
        var result = Execute("null");
        result.Should().BeNull();
    }

    #endregion

    #region Arithmetic Operations

    [Fact]
    public void Execute_Addition_ReturnsSum()
    {
        var result = Execute("2 + 3");
        Convert.ToDouble(result).Should().Be(5.0);
    }

    [Fact]
    public void Execute_Subtraction_ReturnsDifference()
    {
        var result = Execute("10 - 4");
        Convert.ToDouble(result).Should().Be(6.0);
    }

    [Fact]
    public void Execute_Multiplication_ReturnsProduct()
    {
        var result = Execute("3 * 4");
        Convert.ToDouble(result).Should().Be(12.0);
    }

    [Fact]
    public void Execute_Division_ReturnsQuotient()
    {
        var result = Execute("15 / 3");
        Convert.ToDouble(result).Should().Be(5.0);
    }

    [Fact]
    public void Execute_Power_ReturnsExponent()
    {
        var result = Execute("2 ** 3");
        Convert.ToDouble(result).Should().Be(8.0);
    }

    [Fact]
    public void Execute_ComplexExpression_RespectsOperatorPrecedence()
    {
        var result = Execute("2 + 3 * 4");
        Convert.ToDouble(result).Should().Be(14.0);
    }

    [Fact]
    public void Execute_ParenthesizedExpression_OverridesPrecedence()
    {
        var result = Execute("(2 + 3) * 4");
        Convert.ToDouble(result).Should().Be(20.0);
    }

    #endregion

    #region Comparison Operations

    [Fact]
    public void Execute_EqualityTrue_ReturnsTrue()
    {
        var result = Execute("5 == 5");
        result.Should().Be(true);
    }

    [Fact]
    public void Execute_EqualityFalse_ReturnsFalse()
    {
        var result = Execute("5 == 3");
        result.Should().Be(false);
    }

    [Fact]
    public void Execute_GreaterThan_ReturnsCorrectResult()
    {
        Execute("5 > 3").Should().Be(true);
        Execute("3 > 5").Should().Be(false);
    }

    [Fact]
    public void Execute_LessThan_ReturnsCorrectResult()
    {
        Execute("3 < 5").Should().Be(true);
        Execute("5 < 3").Should().Be(false);
    }

    #endregion

    #region Logical Operations

    [Fact]
    public void Execute_LogicalAnd_ReturnsCorrectResult()
    {
        Execute("true and true").Should().Be(true);
        Execute("true and false").Should().Be(false);
    }

    [Fact]
    public void Execute_LogicalOr_ReturnsCorrectResult()
    {
        Execute("false or true").Should().Be(true);
        Execute("false or false").Should().Be(false);
    }

    [Fact]
    public void Execute_LogicalNot_ReturnsCorrectResult()
    {
        Execute("not true").Should().Be(false);
        Execute("not false").Should().Be(true);
    }

    #endregion

    #region Variables

    [Fact]
    public void Execute_VariableDeclarationAndAccess_Works()
    {
        Execute("let x = 42");
        var result = Execute("x");
        Convert.ToDouble(result).Should().Be(42);
    }

    [Fact]
    public void Execute_MutableVariableAssignment_Works()
    {
        Execute("mut x = 1");
        Execute("x = 2");
        var result = Execute("x");
        Convert.ToDouble(result).Should().Be(2);
    }

    #endregion

    #region Arrays

    [Fact]
    public void Execute_ArrayLiteral_CreatesArray()
    {
        var result = Execute("[1, 2, 3]");
        result.Should().BeOfType<List<object?>>();
        ((List<object?>)result!).Should().HaveCount(3);
    }

    [Fact]
    public void Execute_ArrayIndexing_ReturnsCorrectElement()
    {
        Execute("let arr = [10, 20, 30]");
        var result = Execute("arr[1]");
        Convert.ToDouble(result).Should().Be(20);
    }

    #endregion

    #region Functions

    [Fact]
    public void Execute_FunctionDefinitionAndCall_Works()
    {
        Execute("fn add(a, b) { return a + b }");
        var result = Execute("add(2, 3)");
        Convert.ToDouble(result).Should().Be(5);
    }

    [Fact]
    public void Execute_RecursiveFunction_Works()
    {
        Execute(@"
            fn factorial(n) {
                if (n <= 1) { return 1 }
                return n * factorial(n - 1)
            }
        ");
        var result = Execute("factorial(5)");
        Convert.ToDouble(result).Should().Be(120);
    }

    #endregion

    #region Control Flow

    [Fact]
    public void Execute_IfTrue_ExecutesThenBranch()
    {
        Execute("mut x = 0");
        Execute("if (true) { x = 1 }");
        var result = Execute("x");
        Convert.ToDouble(result).Should().Be(1);
    }

    [Fact]
    public void Execute_IfFalse_ExecutesElseBranch()
    {
        Execute("mut x = 0");
        Execute("if (false) { x = 1 } else { x = 2 }");
        var result = Execute("x");
        Convert.ToDouble(result).Should().Be(2);
    }

    [Fact]
    public void Execute_WhileLoop_IteratesCorrectly()
    {
        Execute("mut sum = 0");
        Execute("mut i = 1");
        Execute("while (i <= 5) { sum = sum + i\n i = i + 1 }");
        var result = Execute("sum");
        Convert.ToDouble(result).Should().Be(15);
    }

    [Fact]
    public void Execute_ForLoop_IteratesOverRange()
    {
        Execute("mut sum = 0");
        Execute("for i in range(1, 6) { sum = sum + i }");
        var result = Execute("sum");
        Convert.ToDouble(result).Should().Be(15);
    }

    #endregion

    #region Built-in Functions

    [Fact]
    public void Execute_Print_OutputsToWriter()
    {
        Execute("print(\"hello\", \"world\")");
        GetOutput().Should().Be("hello world");
    }

    [Fact]
    public void Execute_Len_ReturnsArrayLength()
    {
        var result = Execute("len([1, 2, 3, 4, 5])");
        Convert.ToInt32(result).Should().Be(5);
    }

    // Note: type() function conflicts with 'type' keyword in the parser
    // This is a known limitation that could be addressed in a future version

    #endregion

    #region String Operations

    [Fact]
    public void Execute_StringConcatenation_Works()
    {
        var result = Execute("\"hello\" + \" \" + \"world\"");
        result.Should().Be("hello world");
    }

    [Fact]
    public void Execute_Upper_UppercasesString()
    {
        var result = Execute("upper(\"hello\")");
        result.Should().Be("HELLO");
    }

    [Fact]
    public void Execute_Lower_LowercasesString()
    {
        var result = Execute("lower(\"HELLO\")");
        result.Should().Be("hello");
    }

    [Fact]
    public void Execute_Trim_TrimsWhitespace()
    {
        var result = Execute("trim(\"  hello  \")");
        result.Should().Be("hello");
    }

    #endregion

    #region Math Functions

    [Fact]
    public void Execute_Min_ReturnsMinimum()
    {
        var result = Execute("min(5, 3, 8, 1, 9)");
        Convert.ToDouble(result).Should().Be(1);
    }

    [Fact]
    public void Execute_Max_ReturnsMaximum()
    {
        var result = Execute("max(5, 3, 8, 1, 9)");
        Convert.ToDouble(result).Should().Be(9);
    }

    [Fact]
    public void Execute_Abs_ReturnsAbsoluteValue()
    {
        Execute("abs(-5)").Should().Be(5.0);
        Execute("abs(5)").Should().Be(5.0);
    }

    [Fact]
    public void Execute_Sqrt_ReturnsSquareRoot()
    {
        var result = Execute("sqrt(16)");
        Convert.ToDouble(result).Should().Be(4);
    }

    #endregion

    #region Utility Functions

    [Fact]
    public void Execute_Uuid_ReturnsValidUuid()
    {
        var result = Execute("uuid()");
        result.Should().BeOfType<string>();
        Guid.TryParse((string)result!, out _).Should().BeTrue();
    }

    #endregion

    #region Expression Evaluation

    [Fact]
    public void EvaluateExpression_SimpleExpression_ReturnsResult()
    {
        Execute("let x = 10");
        var result = _interpreter.EvaluateExpression("x + 5");
        Convert.ToDouble(result).Should().Be(15);
    }

    #endregion
}
