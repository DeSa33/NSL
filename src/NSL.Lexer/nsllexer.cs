using System.Text;
using NSL.Core.Tokens;

namespace NSL.Lexer;

/// <summary>
/// Lexical analyzer for NSL language
/// </summary>
public class NSLLexer
{
    private readonly string _input;
    private readonly string? _fileName;
    private int _position;
    private int _line;
    private int _column;
    private int _tokenStart;
    private int _tokenLine;
    private int _tokenColumn;

    // Unicode consciousness operators
    private static readonly Dictionary<char, TokenType> UnicodeOperators = new()
    {
        // Core consciousness operators
        { '◈', TokenType.Holographic },  // U+25C8 - Attention/Focus
        { '∇', TokenType.Gradient },     // U+2207 - Gradient/Learning
        { '⊗', TokenType.TensorProduct },// U+2297 - Composition/Binding
        { 'Ψ', TokenType.Psi },          // U+03A8 - Quantum Branching

        // Extended consciousness operators
        { 'μ', TokenType.Mu },           // U+03BC - Memory
        { 'σ', TokenType.Sigma },        // U+03C3 - Self/Introspection
        { '↓', TokenType.Collapse },     // U+2193 - Collapse/Measurement
        { '≈', TokenType.Similarity },   // U+2248 - Similarity
        { '≉', TokenType.Dissimilarity },// U+2249 - Dissimilarity
        { '∫', TokenType.Integral },     // U+222B - Temporal Integration
        { '±', TokenType.PlusMinus },    // U+00B1 - Uncertainty

        // Arrows
        { 'λ', TokenType.Lambda },       // U+03BB
        { '→', TokenType.Arrow },        // U+2192
        { '←', TokenType.LeftArrow }     // U+2190
    };

    // Keywords
    private static readonly Dictionary<string, TokenType> Keywords = new()
    {
        // Core keywords
        { "let", TokenType.Let },
        { "mut", TokenType.Mut },           // Mutable (immutable by default)
        { "const", TokenType.Const },
        { "if", TokenType.If },
        { "then", TokenType.Then },
        { "else", TokenType.Else },
        { "while", TokenType.While },
        { "for", TokenType.For },
        { "in", TokenType.In },
        { "break", TokenType.Break },
        { "continue", TokenType.Continue },
        { "function", TokenType.Function },
        { "fn", TokenType.Function },       // Short form (AI prefers concise)
        { "func", TokenType.Function },     // Common alias
        { "class", TokenType.Class },
        { "return", TokenType.Return },
        { "match", TokenType.Match },
        { "case", TokenType.Case },         // Pattern matching
        { "when", TokenType.When },         // Pattern matching guard
        { "enum", TokenType.Enum },         // Algebraic data type
        { "and", TokenType.And },
        { "or", TokenType.Or },
        { "not", TokenType.Not },
        { "true", TokenType.True },
        { "false", TokenType.False },
        { "null", TokenType.Null },

        // AI-friendly type keywords
        { "type", TokenType.Type },         // Type aliases
        { "struct", TokenType.Struct },     // Struct definition
        { "vec", TokenType.Vec },           // Vector type
        { "mat", TokenType.Mat },           // Matrix type
        { "tensor", TokenType.Tensor },     // Tensor type
        { "prob", TokenType.Prob },         // Probability (0..1)
        // NOTE: "ok" and "err" removed as keywords - they're built-in functions now
        // This allows using them as variable names: let err = x - y
        { "some", TokenType.Some },         // Optional present
        { "none", TokenType.None },         // Optional absent
        { "as", TokenType.As },             // Type casting

        // Module system keywords
        { "import", TokenType.Import },     // Import declaration
        { "from", TokenType.From },         // Import source
        { "export", TokenType.Export },     // Export declaration
        { "pub", TokenType.Pub },           // Public visibility (Rust-style)
        { "module", TokenType.Module },     // Module declaration

        // Trait/Interface keywords
        { "trait", TokenType.Trait },       // Trait definition
        { "impl", TokenType.Impl },         // Implementation

        // Async/Await keywords
        { "async", TokenType.Async },       // Async function
        { "await", TokenType.Await },       // Await expression

        // Extended consciousness keywords (ASCII aliases)
        // NOTE: Many ASCII aliases removed to avoid conflicts with common function names.
        // Use the Unicode operators directly: μ σ ↓ ≈ ∫ ±
        { "mem", TokenType.Mu },            // μ short alias (memory operator)
        { "introspect", TokenType.Sigma }   // σ alias (self/introspection)
    };

    /// <summary>See implementation for details.</summary>
    public NSLLexer(string input, string? fileName = null)
    {
        _input = input ?? throw new ArgumentNullException(nameof(input));
        _fileName = fileName;
        _position = 0;
        _line = 1;
        _column = 1;
    }

    /// <summary>
    /// Tokenizes the entire input
    /// </summary>
    public List<Token> Tokenize()
    {
        var tokens = new List<Token>();
        Token token;
        
        do
        {
            token = NextToken();
            if (token.Type != TokenType.Whitespace && 
                token.Type != TokenType.SingleLineComment &&
                token.Type != TokenType.MultiLineComment)
            {
                tokens.Add(token);
            }
        } while (token.Type != TokenType.EndOfFile);
        
        return tokens;
    }

    /// <summary>
    /// Gets the next token from the input
    /// </summary>
    public Token NextToken()
    {
        SkipWhitespace();
        
        if (IsAtEnd())
            return CreateToken(TokenType.EndOfFile);
        
        MarkTokenStart();
        
        char c = Advance();
        
        // Check for Unicode operators first
        if (UnicodeOperators.TryGetValue(c, out var unicodeType))
            return CreateToken(unicodeType);
        
        // Handle other characters
        return c switch
        {
            // Single-character tokens
            '(' => CreateToken(TokenType.LeftParen),
            ')' => CreateToken(TokenType.RightParen),
            '[' => CreateToken(TokenType.LeftBracket),
            ']' => CreateToken(TokenType.RightBracket),
            '{' => CreateToken(TokenType.LeftBrace),
            '}' => CreateToken(TokenType.RightBrace),
            ',' => CreateToken(TokenType.Comma),
            ';' => CreateToken(TokenType.Semicolon),
            '@' => CreateToken(TokenType.AtSign),      // Matrix multiply / decorator
            '+' => Match('>') ? CreateToken(TokenType.SuperpositionArrow) : CreateToken(TokenType.Plus),  // +> or +
            '-' => CreateToken(TokenType.Minus),
            '/' => ScanSlash(),
            '%' => CreateToken(TokenType.Percent),
            '#' => ScanHashComment(),

            // AI-friendly: Dot operators (., .., ..=)
            '.' => ScanDot(),

            // AI-friendly: Question operators (?., ??)
            '?' => ScanQuestion(),

            // Multi-character operators
            '*' => Match('>') ? CreateToken(TokenType.AttentionArrow) :  // *> attention
                   Match('*') ? CreateToken(TokenType.Power) : CreateToken(TokenType.Multiply),
            '=' => ScanEquals(),  // =, ==, =>, =>>
            '!' => Match('=') ? CreateToken(TokenType.NotEqual) : CreateToken(TokenType.LogicalNot),
            '<' => Match('<') ? CreateToken(TokenType.LeftShift) :
                   Match('=') ? CreateToken(TokenType.LessEqual) : CreateToken(TokenType.Less),
            '>' => Match('>') ? CreateToken(TokenType.RightShift) :
                   Match('=') ? CreateToken(TokenType.GreaterEqual) : CreateToken(TokenType.Greater),
            '&' => Match('&') ? CreateToken(TokenType.LogicalAnd) : CreateToken(TokenType.BitwiseAnd),
            '|' => ScanPipe(),                         // |, ||, |>, bitwise or
            '^' => CreateToken(TokenType.BitwiseXor),
            '~' => Match('>') ? CreateToken(TokenType.AwarenessArrow) : CreateToken(TokenType.BitwiseNot),  // ~> or ~
            ':' => Match(':') ? CreateToken(TokenType.Chain) : CreateToken(TokenType.Colon),

            // Literals
            '"' => ScanStringOrHeredoc(),
            _ when IsDigit(c) => ScanNumber(),
            _ when IsAlpha(c) => ScanIdentifier(),

            // Invalid character
            _ => CreateInvalidToken($"Unexpected character '{c}'")
        };
    }

    private void SkipWhitespace()
    {
        while (!IsAtEnd())
        {
            switch (Peek())
            {
                case ' ':
                case '\r':
                case '\t':
                    Advance();
                    break;
                case '\n':
                    _line++;
                    _column = 0;
                    Advance();
                    break;
                default:
                    return;
            }
        }
    }

    private Token ScanSlash()
    {
        // Check for single-line comment (//)
        if (Match('/'))
        {
            // // is now a single-line comment (like most languages)
            while (Peek() != '\n' && !IsAtEnd())
                Advance();
            return NextToken(); // Skip comment, get next real token
        }
        else if (Match('*'))
        {
            // Multi-line comment /* */
            while (!IsAtEnd())
            {
                if (Peek() == '*' && PeekNext() == '/')
                {
                    Advance(); // *
                    Advance(); // /
                    break;
                }
                if (Peek() == '\n')
                {
                    _line++;
                    _column = 0;
                }
                Advance();
            }
            return CreateToken(TokenType.MultiLineComment);
        }

        return CreateToken(TokenType.Divide);
    }

    private Token ScanHashComment()
    {
        // Skip the # character (already consumed)
        // Read until end of line or end of file
        while (Peek() != '\n' && !IsAtEnd())
        {
            Advance();
        }

        return CreateToken(TokenType.SingleLineComment);
    }

    /// <summary>
    /// AI-friendly: Scan dot operators (., .., ..=)
    /// Range operators help AI avoid off-by-one errors
    /// </summary>
    private Token ScanDot()
    {
        if (Match('.'))
        {
            // Could be .. or ..=
            if (Match('='))
                return CreateToken(TokenType.DotDotEqual);  // ..= inclusive range
            return CreateToken(TokenType.DotDot);           // .. exclusive range
        }
        return CreateToken(TokenType.Dot);
    }

    /// <summary>
    /// AI-friendly: Scan question operators (?., ??)
    /// Safe navigation prevents null reference errors - a common AI mistake
    /// </summary>
    private Token ScanQuestion()
    {
        if (Match('.'))
            return CreateToken(TokenType.QuestionDot);      // ?. safe navigation
        if (Match('?'))
            return CreateToken(TokenType.QuestionQuestion); // ?? null coalescing
        return CreateInvalidToken("Expected '.' or '?' after '?'");
    }

    /// <summary>
    /// AI-friendly: Scan pipe operators (|, ||, |>)
    /// Pipeline operator |> matches how AI thinks about data flow
    /// </summary>
    private Token ScanPipe()
    {
        if (Match('>'))
            return CreateToken(TokenType.PipeArrow);        // |> pipeline
        if (Match('|'))
            return CreateToken(TokenType.LogicalOr);        // || logical or
        return CreateToken(TokenType.BitwiseOr);            // | bitwise or
    }

    /// <summary>
    /// Scan equals-based operators (=, ==, =>, =>>)
    /// =>> is the gradient/learning operator
    /// </summary>
    private Token ScanEquals()
    {
        if (Match('>'))
        {
            // Could be => or =>>
            if (Match('>'))
                return CreateToken(TokenType.GradientArrow);  // =>> gradient/learning
            return CreateToken(TokenType.FatArrow);           // => lambda/match
        }
        if (Match('='))
            return CreateToken(TokenType.Equal);              // == equality
        return CreateToken(TokenType.Assign);                 // = assignment
    }

    /// <summary>
    /// Scan string, heredoc ("""), or raw string (r"...")
    /// </summary>
    private Token ScanStringOrHeredoc()
    {
        // Check for heredoc (triple quotes)
        if (Peek() == '"' && PeekNext() == '"')
        {
            Advance(); // Second "
            Advance(); // Third "
            return ScanHeredoc();
        }
        
        return ScanString(processEscapes: true);
    }
    
    /// <summary>
    /// Scan raw string - no escape processing (called when 'r' prefix detected)
    /// Supports both r"..." and r'...' syntax for flexibility with embedded quotes
    /// </summary>
    private Token ScanRawString()
    {
        char quoteChar = Peek();
        // Consume opening quote
        Advance();
        return ScanStringWithQuote(quoteChar, processEscapes: false);
    }
    
    /// <summary>
    /// Scan string with specified quote character - allows single or double quotes
    /// </summary>
    private Token ScanStringWithQuote(char quoteChar, bool processEscapes)
    {
        var value = new StringBuilder();
        
        while (Peek() != quoteChar && !IsAtEnd())
        {
            if (Peek() == '\n')
            {
                _line++;
                _column = 0;
            }
            
            if (processEscapes && Peek() == '\\')
            {
                Advance(); // Skip backslash
                if (!IsAtEnd())
                {
                    char escaped = Advance();
                    char unescaped = escaped switch
                    {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '"' => '"',
                        '\'' => '\'',
                        '0' => '\0',
                        _ => escaped
                    };
                    value.Append(unescaped);
                }
            }
            else
            {
                value.Append(Advance());
            }
        }
        
        if (IsAtEnd())
            return CreateInvalidToken("Unterminated string");
        
        // Consume closing quote
        Advance();
        
        return CreateToken(TokenType.String, value.ToString());
    }
    
    /// <summary>
    /// Scan heredoc (triple-quoted string) - preserves newlines and doesn't process escapes
    /// </summary>
    private Token ScanHeredoc()
    {
        var value = new StringBuilder();
        
        while (!IsAtEnd())
        {
            // Check for closing """
            if (Peek() == '"' && PeekAt(1) == '"' && PeekAt(2) == '"')
            {
                Advance(); // First "
                Advance(); // Second "
                Advance(); // Third "
                return CreateToken(TokenType.String, value.ToString());
            }
            
            if (Peek() == '\n')
            {
                _line++;
                _column = 0;
            }
            
            value.Append(Advance());
        }
        
        return CreateInvalidToken("Unterminated heredoc string");
    }
    
    private Token ScanString(bool processEscapes)
    {
        var value = new StringBuilder();
        
        while (Peek() != '"' && !IsAtEnd())
        {
            if (Peek() == '\n')
            {
                _line++;
                _column = 0;
            }
            
            if (processEscapes && Peek() == '\\')
            {
                Advance(); // Skip backslash
                if (!IsAtEnd())
                {
                    char escaped = Advance();
                    char unescaped = escaped switch
                    {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '"' => '"',
                        '\'' => '\'',
                        '0' => '\0',
                        _ => escaped
                    };
                    value.Append(unescaped);
                }
            }
            else
            {
                value.Append(Advance());
            }
        }
        
        if (IsAtEnd())
            return CreateInvalidToken("Unterminated string");
        
        // Consume closing "
        Advance();
        
        return CreateToken(TokenType.String, value.ToString());
    }

    private Token ScanNumber()
    {
        bool isFloat = false;

        // Integer part
        while (IsDigit(Peek()))
            Advance();

        // Decimal part - makes this a float
        if (Peek() == '.' && IsDigit(PeekNext()))
        {
            isFloat = true;
            Advance(); // Consume .
            while (IsDigit(Peek()))
                Advance();
        }

        // Scientific notation - makes this a float
        if (Peek() == 'e' || Peek() == 'E')
        {
            isFloat = true;
            Advance();
            if (Peek() == '+' || Peek() == '-')
                Advance();

            if (!IsDigit(Peek()))
                return CreateInvalidToken("Invalid number format");

            while (IsDigit(Peek()))
                Advance();
        }

        return CreateToken(isFloat ? TokenType.Number : TokenType.Integer);
    }

    private Token ScanIdentifier()
    {
        // Check for raw string prefix: r"..." or r'...'
        if (_input[_tokenStart] == 'r' && (Peek() == '"' || Peek() == '\''))
        {
            return ScanRawString();
        }
        
        while (IsAlphaNumeric(Peek()))
            Advance();
        
        string text = GetTokenText();
        
        // Check if it's a keyword
        if (Keywords.TryGetValue(text, out var keywordType))
            return CreateToken(keywordType);
        
        return CreateToken(TokenType.Identifier);
    }

    // Helper methods
    private bool IsAtEnd() => _position >= _input.Length;
    
    private char Peek() => IsAtEnd() ? '\0' : _input[_position];
    
    private char PeekNext() => _position + 1 >= _input.Length ? '\0' : _input[_position + 1];
    
    private char PeekAt(int offset) => _position + offset >= _input.Length ? '\0' : _input[_position + offset];
    
    private char Advance()
    {
        _column++;
        return _input[_position++];
    }
    
    private bool Match(char expected)
    {
        if (IsAtEnd() || _input[_position] != expected)
            return false;
        
        Advance();
        return true;
    }
    
    private void MarkTokenStart()
    {
        _tokenStart = _position;
        _tokenLine = _line;
        _tokenColumn = _column;
    }
    
    private string GetTokenText() => _input[_tokenStart.._position];
    
    private Token CreateToken(TokenType type) => CreateToken(type, GetTokenText());
    
    private Token CreateToken(TokenType type, string value) =>
        new(type, value, _tokenStart, _tokenLine, _tokenColumn, _fileName);
    
    private Token CreateInvalidToken(string message) =>
        Token.Invalid(message, _tokenStart, _tokenLine, _tokenColumn, _fileName);
    
    private static bool IsDigit(char c) => c >= '0' && c <= '9';
    
    private static bool IsAlpha(char c) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    
    private static bool IsAlphaNumeric(char c) => IsAlpha(c) || IsDigit(c);
}