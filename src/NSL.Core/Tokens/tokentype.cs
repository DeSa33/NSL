namespace NSL.Core.Tokens;

/// <summary>
/// Defines all token types supported by the NSL language
/// </summary>
public enum TokenType
{
    // Special tokens
    EndOfFile,
    Invalid,
    
    // Literals
    Number,
    Integer,
    String,
    Identifier,
    True,
    False,
    Boolean,
    Null,
    
    // Keywords
    Let,               // Variable declaration (immutable by default)
    Mut,               // Mutable variable declaration
    Const,             // Constant declaration
    If,
    Then,
    Else,
    While,
    For,
    In,                // For loop 'in' keyword
    Break,             // Loop break statement
    Continue,          // Loop continue statement
    Function,
    Class,
    Return,
    Match,
    Case,              // Pattern matching case
    When,              // Pattern matching guard
    Enum,              // Algebraic data type
    And,
    Or,
    Not,

    // AI-friendly type keywords
    Type,              // Type declaration
    Struct,            // Struct definition
    Vec,               // Vector type
    Mat,               // Matrix type
    Tensor,            // Tensor type
    Prob,              // Probability type (0..1)
    Ok,                // Result success
    Err,               // Result error
    Some,              // Optional value present
    None,              // Optional value absent
    As,                // Type casting

    // Module system keywords
    Import,            // import keyword
    From,              // from keyword (import x from "module")
    Export,            // export keyword (explicit export)
    Pub,               // pub keyword (public visibility)
    Module,            // module keyword (module declaration)

    // Trait/Interface keywords
    Trait,             // trait definition keyword
    Impl,              // implementation keyword

    // Async/Await keywords
    Async,             // async function definition
    Await,             // await expression
    
    // Operators - Standard
    Plus,              // +
    Minus,             // -
    Multiply,          // *
    Star,              // * (alternative context)
    Divide,            // /
    Percent,           // % (modulo)
    Power,             // **
    Assign,            // =
    
    // Comparison operators
    Equal,             // ==
    NotEqual,          // !=
    Less,              // <
    LessEqual,         // <=
    Greater,           // >
    GreaterEqual,      // >=
    
    // Logical operators
    LogicalAnd,        // &&
    LogicalOr,         // ||
    LogicalNot,        // !

    // Bitwise operators
    BitwiseAnd,        // &
    BitwiseOr,         // | (single pipe)
    BitwiseXor,        // ^
    BitwiseNot,        // ~
    LeftShift,         // <<
    RightShift,        // >>
    IntegerDivide,     // // (integer division)

    // AI-friendly operators
    QuestionDot,       // ?. (safe navigation - avoid null errors)
    QuestionQuestion,  // ?? (null coalescing)
    PipeArrow,         // |> (pipeline - AI thinks in data flows)
    DotDot,            // .. (range operator)
    DotDotEqual,       // ..= (inclusive range)
    FatArrow,          // => (lambda/match arm)
    AtSign,            // @ (matrix multiply / decorator)

    // ASCII consciousness operators (AI-native flow)
    AwarenessArrow,    // ~> (introspective flow)
    AttentionArrow,    // *> (attention/focus mechanism)
    SuperpositionArrow,// +> (quantum-like superposition)
    GradientArrow,     // =>> (learning/gradient adjustment)
    
    // NSL-specific consciousness operators
    Holographic,       // ◈ (U+25C8) - Attention/Focus
    Gradient,          // ∇ (U+2207) - Gradient/Learning
    TensorProduct,     // ⊗ (U+2297) - Composition/Binding
    Psi,               // Ψ (U+03A8) - Quantum Branching/Superposition
    QuantumBranching,  // Ψ (U+03A8) - Alternative name for quantum operations

    // Extended consciousness operators
    Mu,                // μ (U+03BC) - Memory operator
    MuStore,           // μ→ - Memory store
    MuRecall,          // μ← - Memory recall
    Sigma,             // σ (U+03C3) - Self/Introspection
    Collapse,          // ↓ (U+2193) - Collapse/Measurement
    Similarity,        // ≈ (U+2248) - Similarity/Distance
    Dissimilarity,     // ≉ (U+2249) - Dissimilarity
    Integral,          // ∫ (U+222B) - Temporal Integration

    // Uncertainty operators
    PlusMinus,         // ± (U+00B1) - Uncertainty range

    Lambda,            // λ (U+03BB)
    Arrow,             // → (U+2192)
    LeftArrow,         // ← (U+2190)
    Chain,             // ::
    
    // Delimiters
    LeftParen,         // (
    RightParen,        // )
    LeftBracket,       // [
    RightBracket,      // ]
    LeftBrace,         // {
    RightBrace,        // }
    
    // Separators
    Comma,             // ,
    Semicolon,         // ;
    Colon,             // :
    Pipe,              // |
    Dot,               // .
    
    // Comments (for lexer processing)
    SingleLineComment,
    MultiLineComment,
    
    // Whitespace (for lexer processing)
    Whitespace,
    Newline
}