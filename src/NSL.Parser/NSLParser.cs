using System;
using System.Collections.Generic;
using System.Linq;
using NSL.Core;
using NSL.Core.AST;
using NSL.Core.Tokens;

// Using TokenType directly - TokenType has been consolidated into TokenType

namespace NSL.Parser
{
    /// <summary>
    /// NSL Parser compatible with existing NSLLexer
    /// Converts NSLToken objects into an Abstract Syntax Tree (AST)
    /// </summary>
    public class NSLParser
    {
        private List<NSLToken> _tokens = new();
        private int _current = 0;
        private string[]? _sourceLines = null;  // Store source for better error messages

        /// <summary>
        /// Set source code for better error messages
        /// </summary>
        public void SetSource(string source)
        {
            _sourceLines = source.Split('\n');
        }

        /// <summary>
        /// Parse Token list into an AST (converts from Token to NSLToken)
        /// </summary>
        public NSLASTNode Parse(IEnumerable<Token> tokens)
        {
            // Convert Token to NSLToken
            var nslTokens = tokens.Select(ConvertToNSLToken);
            return ParseNSLTokens(nslTokens);
        }

        /// <summary>
        /// Parse a single expression from Token list (for debugger expression evaluation)
        /// </summary>
        public NSLASTNode ParseExpression(IEnumerable<Token>? tokens = null)
        {
            if (tokens != null)
            {
                var nslTokens = tokens.Select(ConvertToNSLToken);
                _tokens = nslTokens.ToList();
                _current = 0;
            }

            if (_tokens.Count == 0 || IsAtEnd())
            {
                return new NSLLiteralNode(null!, TokenType.Null);
            }
            return ParseChainExpression();
        }

        /// <summary>
        /// Parse NSLToken list into an AST
        /// </summary>
        private NSLASTNode ParseNSLTokens(IEnumerable<NSLToken> tokens)
        {
            _tokens = tokens.ToList();
            _current = 0;

            try
            {
                if (_tokens.Count == 0 || (_tokens.Count > 0 && _tokens[0].Type == TokenType.EndOfFile))
                {
                    return new NSLLiteralNode(null!, TokenType.Null);
                }

                // Parse all statements in the file
                var statements = new List<NSLASTNode>();
                
                while (!IsAtEnd() && !Check(TokenType.EndOfFile))
                {
                    // Skip newlines between statements
                    ConsumeNewlines();
                    
                    if (!IsAtEnd() && !Check(TokenType.EndOfFile))
                    {
                        statements.Add(ParseStatement());
                    }
                }
                
                // If only one statement, return it directly
                if (statements.Count == 1)
                {
                    return statements[0];
                }
                
                // If multiple statements, wrap in a block
                if (statements.Count > 1)
                {
                    return new NSLBlockNode(statements);
                }
                
                // If no statements, return null literal
                return new NSLLiteralNode(null!, TokenType.Null);
            }
            catch (Exception ex)
            {
                throw new NSLParseException(FormatError(ex.Message, GetCurrentLine()), ex);
            }
        }

        private NSLASTNode ParseStatement()
        {
            try
            {
                // Module system statements (must be at top of file, but we'll parse them anywhere)
                // import math
                // import { sin, cos } from math
                // import * from math
                if (Match(TokenType.Import))
                {
                    return ParseImportStatement();
                }

                // export { foo, bar }
                // export * from "submodule"
                if (Match(TokenType.Export))
                {
                    return ParseExportStatement();
                }

                // module math::linear_algebra { ... }
                if (Match(TokenType.Module))
                {
                    return ParseModuleDeclaration();
                }

                // pub fn foo() { ... } - public function
                // pub let X = 10 - public constant
                if (Match(TokenType.Pub))
                {
                    return ParsePublicDeclaration();
                }

                // AI-friendly: Variable declarations with type hints
                // let x: number = 10 (immutable)
                // mut y: string = "hello" (mutable)
                if (Match(TokenType.Let))
                {
                    return ParseVariableDeclaration(isMutable: false);
                }
                if (Match(TokenType.Mut))
                {
                    return ParseVariableDeclaration(isMutable: true);
                }
                if (Match(TokenType.Const))
                {
                    return ParseVariableDeclaration(isMutable: false);
                }

                // AI-friendly: Type alias declarations
                // type Point = {x: number, y: number}
                // But NOT type() which is a built-in function call
                if (Check(TokenType.Type) && !CheckNext(TokenType.LeftParen))
                {
                    Advance(); // consume 'type'
                    return ParseTypeAlias();
                }

                // Struct definitions
                // struct Point { x: number, y: number }
                if (Match(TokenType.Struct))
                {
                    return ParseStructDefinition();
                }

                // Control flow statements
                if (Match(TokenType.If)) return ParseIfStatement();
                if (Match(TokenType.While)) return ParseWhileStatement();
                if (Match(TokenType.For)) return ParseForStatement();
                if (Match(TokenType.Function)) return ParseFunctionStatement();
                if (Match(TokenType.Return)) return ParseReturnStatement();
                if (Match(TokenType.Break)) return ParseBreakStatement();
                if (Match(TokenType.Continue)) return ParseContinueStatement();

                // AI-friendly: Pattern matching
                // match value { case 0 => "zero", case n => "other" }
                if (Match(TokenType.Match)) return ParseMatchStatement();

                // Algebraic data types (enums with data)
                // enum Color { Red, Green, Blue }
                // enum Shape { Circle(number), Rectangle(number, number) }
                if (Match(TokenType.Enum)) return ParseEnumDefinition();

                // Traits/Interfaces for polymorphism
                // trait Printable { fn print(self); }
                if (Match(TokenType.Trait)) return ParseTraitDefinition();

                // Trait implementations
                // impl Printable for Person { fn print(self) { ... } }
                if (Match(TokenType.Impl)) return ParseImplDefinition();

                // Async functions
                // async fn fetch_data() { ... }
                if (Match(TokenType.Async)) return ParseAsyncFunction();

                // Block statement
                if (Check(TokenType.LeftBrace)) return ParseBlock();

                // Expression statement (including assignments)
                return ParseExpressionStatement();
            }
            catch (Exception ex)
            {
                throw new NSLParseException($"Parse error at line {Peek().Line}: {ex.Message}");
            }
        }

        /// <summary>
        /// AI-friendly: Parse variable declaration with optional type hints
        /// Example: let x: number = 10, mut y = 20
        /// </summary>
        private NSLASTNode ParseVariableDeclaration(bool isMutable)
        {
            // Allow type keywords to be used as variable names
            // e.g., let tensor = ⊗[a, b] should work
            string name;
            if (Check(TokenType.Identifier))
            {
                name = Advance().Value;
            }
            else if (Check(TokenType.Vec) || Check(TokenType.Mat) ||
                     Check(TokenType.Tensor) || Check(TokenType.Prob))
            {
                // Allow type keywords as variable names
                name = Advance().Value;
            }
            else
            {
                throw new NSLParseException($"Expected variable name, got {Peek().Type}");
            }

            // Optional type hint: x: number
            string? typeHint = null;
            if (Match(TokenType.Colon))
            {
                typeHint = ParseTypeHint();
            }

            // Optional initializer
            NSLASTNode? value = null;
            if (Match(TokenType.Assign))
            {
                value = ParseExpression();
            }

            ConsumeStatementEnd();
            return new NSLVariableDeclarationNode(name, value, typeHint, isMutable);
        }

        /// <summary>
        /// Parse a type hint (number, string, vec, mat, etc.)
        /// </summary>
        private string ParseTypeHint()
        {
            // Handle built-in type keywords
            if (Match(TokenType.Vec)) return "vec";
            if (Match(TokenType.Mat)) return "mat";
            if (Match(TokenType.Tensor)) return "tensor";
            if (Match(TokenType.Prob)) return "prob";

            // Handle identifier-based types
            if (Check(TokenType.Identifier))
            {
                return Advance().Value;
            }

            throw new NSLParseException("Expected type name");
        }

        /// <summary>
        /// AI-friendly: Parse type alias declaration
        /// Example: type Point = {x: number, y: number}
        /// </summary>
        private NSLASTNode ParseTypeAlias()
        {
            var nameToken = Consume(TokenType.Identifier, "Expected type name");
            Consume(TokenType.Assign, "Expected '=' after type name");
            var definition = ParseExpression();
            ConsumeStatementEnd();
            return new NSLTypeAliasNode(nameToken.Value, definition);
        }

        /// <summary>
        /// Parse struct definition
        /// Example: struct Point { x: number, y: number }
        /// </summary>
        private NSLASTNode ParseStructDefinition()
        {
            var nameToken = Consume(TokenType.Identifier, "Expected struct name");
            Consume(TokenType.LeftBrace, "Expected '{' after struct name");
            ConsumeNewlines();

            var fields = new List<NSLStructField>();

            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                ConsumeNewlines();
                if (Check(TokenType.RightBrace)) break;

                // Parse field name
                var fieldName = Consume(TokenType.Identifier, "Expected field name");
                Consume(TokenType.Colon, "Expected ':' after field name");

                // Parse field type
                var fieldType = ParseTypeHint();

                fields.Add(new NSLStructField(fieldName.Value, fieldType));

                // Handle optional comma between fields
                if (!Check(TokenType.RightBrace))
                {
                    Match(TokenType.Comma);
                }
                ConsumeNewlines();
            }

            Consume(TokenType.RightBrace, "Expected '}' after struct fields");
            return new NSLStructNode(nameToken.Value, fields);
        }

        /// <summary>
        /// Parse enum definition (algebraic data type)
        /// Example: enum Color { Red, Green, Blue }
        /// Example: enum Shape { Circle(number), Rectangle(number, number) }
        /// </summary>
        private NSLASTNode ParseEnumDefinition()
        {
            var nameToken = Consume(TokenType.Identifier, "Expected enum name");
            Consume(TokenType.LeftBrace, "Expected '{' after enum name");
            ConsumeNewlines();

            var variants = new List<NSLEnumVariant>();

            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                ConsumeNewlines();
                if (Check(TokenType.RightBrace)) break;

                // Parse variant name
                var variantName = Consume(TokenType.Identifier, "Expected variant name");

                // Check for tuple variant: Circle(number)
                List<string>? fields = null;
                if (Match(TokenType.LeftParen))
                {
                    fields = new List<string>();

                    if (!Check(TokenType.RightParen))
                    {
                        do
                        {
                            // Parse field type
                            var fieldType = ParseTypeHint();
                            fields.Add(fieldType);
                        } while (Match(TokenType.Comma));
                    }

                    Consume(TokenType.RightParen, "Expected ')' after variant fields");
                }

                variants.Add(new NSLEnumVariant(variantName.Value, fields));

                // Handle optional comma between variants
                if (!Check(TokenType.RightBrace))
                {
                    Match(TokenType.Comma);
                }
                ConsumeNewlines();
            }

            Consume(TokenType.RightBrace, "Expected '}' after enum variants");
            return new NSLEnumNode(nameToken.Value, variants);
        }

        /// <summary>
        /// Parse trait definition
        /// Example: trait Printable { fn print(self); fn format(self) -> string; }
        /// </summary>
        private NSLASTNode ParseTraitDefinition()
        {
            var nameToken = Consume(TokenType.Identifier, "Expected trait name");
            Consume(TokenType.LeftBrace, "Expected '{' after trait name");
            ConsumeNewlines();

            var methods = new List<NSLTraitMethod>();

            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                ConsumeNewlines();
                if (Check(TokenType.RightBrace)) break;

                // Expect fn keyword for each method
                Consume(TokenType.Function, "Expected 'fn' in trait method");
                var methodName = Consume(TokenType.Identifier, "Expected method name");

                // Parse parameters
                Consume(TokenType.LeftParen, "Expected '(' after method name");
                var parameters = new List<NSLParameter>();

                if (!Check(TokenType.RightParen))
                {
                    do
                    {
                        var paramName = Consume(TokenType.Identifier, "Expected parameter name");
                        string? paramType = null;

                        if (Match(TokenType.Colon))
                        {
                            paramType = ParseTypeHint();
                        }

                        parameters.Add(new NSLParameter(paramName.Value, paramType));
                    } while (Match(TokenType.Comma));
                }

                Consume(TokenType.RightParen, "Expected ')' after parameters");

                // Optional return type
                string? returnType = null;
                if (Match(TokenType.Arrow))
                {
                    returnType = ParseTypeHint();
                }

                // Method signature ends with semicolon
                Match(TokenType.Semicolon);

                methods.Add(new NSLTraitMethod(methodName.Value, parameters, returnType));
                ConsumeNewlines();
            }

            Consume(TokenType.RightBrace, "Expected '}' after trait methods");
            return new NSLTraitNode(nameToken.Value, methods);
        }

        /// <summary>
        /// Parse trait implementation
        /// Example: impl Printable for Person { fn print(self) { println("Person"); } }
        /// </summary>
        private NSLASTNode ParseImplDefinition()
        {
            var traitName = Consume(TokenType.Identifier, "Expected trait name after 'impl'");
            Consume(TokenType.For, "Expected 'for' after trait name");
            var typeName = Consume(TokenType.Identifier, "Expected type name after 'for'");
            Consume(TokenType.LeftBrace, "Expected '{' after type name");
            ConsumeNewlines();

            var methods = new List<NSLFunctionNode>();

            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                ConsumeNewlines();
                if (Check(TokenType.RightBrace)) break;

                // Each method is a full function definition
                Consume(TokenType.Function, "Expected 'fn' in impl block");
                var func = ParseFunctionStatement() as NSLFunctionNode;
                if (func != null)
                {
                    methods.Add(func);
                }

                ConsumeNewlines();
            }

            Consume(TokenType.RightBrace, "Expected '}' after impl methods");
            return new NSLImplNode(traitName.Value, typeName.Value, methods);
        }

        /// <summary>
        /// Parse async function
        /// Example: async fn fetch_data() { ... }
        /// </summary>
        private NSLASTNode ParseAsyncFunction()
        {
            Consume(TokenType.Function, "Expected 'fn' after 'async'");
            var nameToken = Consume(TokenType.Identifier, "Expected function name");

            Consume(TokenType.LeftParen, "Expected '(' after function name");
            var parameters = new List<NSLParameter>();

            if (!Check(TokenType.RightParen))
            {
                do
                {
                    var paramName = Consume(TokenType.Identifier, "Expected parameter name");
                    string? paramType = null;

                    if (Match(TokenType.Colon))
                    {
                        paramType = ParseTypeHint();
                    }

                    parameters.Add(new NSLParameter(paramName.Value, paramType));
                } while (Match(TokenType.Comma));
            }

            Consume(TokenType.RightParen, "Expected ')' after parameters");

            // Optional return type
            string? returnType = null;
            if (Match(TokenType.Arrow))
            {
                returnType = ParseTypeHint();
            }

            // Parse body (block or expression)
            NSLASTNode body;
            if (Check(TokenType.LeftBrace))
            {
                body = ParseBlock();
            }
            else if (Match(TokenType.FatArrow))
            {
                body = ParseExpression();
            }
            else
            {
                throw new NSLParseException("Expected '{' or '=>' for async function body");
            }

            return new NSLAsyncFunctionNode(nameToken.Value, parameters, body, returnType);
        }

        #region Module System Parsing

        /// <summary>
        /// Parse import statement
        /// Supports multiple styles:
        /// - import math                           (import entire module)
        /// - import math::linear_algebra           (import submodule)
        /// - import { sin, cos } from math         (selective import)
        /// - import { sin as sine } from math      (aliased import)
        /// - import * from math                    (wildcard import)
        /// - import math as m                      (aliased module import)
        /// - import "path/to/file.nsl"             (file import)
        /// </summary>
        private NSLASTNode ParseImportStatement()
        {
            // Check for file path import: import "path/to/file.nsl"
            if (Check(TokenType.String))
            {
                var pathToken = Advance();
                var filePath = pathToken.Value;
                // Remove quotes
                if (filePath.StartsWith("\"") && filePath.EndsWith("\""))
                {
                    filePath = filePath[1..^1];
                }
                ConsumeStatementEnd();
                return new NSLImportNode(null, filePath: filePath);
            }

            // Check for selective import: import { sin, cos } from math
            if (Match(TokenType.LeftBrace))
            {
                var items = ParseImportItems();
                Consume(TokenType.RightBrace, "Expected '}' after import items");
                Consume(TokenType.From, "Expected 'from' after import items");
                var modulePath = ParseModulePath();
                ConsumeStatementEnd();
                return new NSLImportNode(modulePath, items: items);
            }

            // Check for wildcard import: import * from math
            if (Match(TokenType.Multiply))
            {
                Consume(TokenType.From, "Expected 'from' after '*'");
                var modulePath = ParseModulePath();
                ConsumeStatementEnd();
                return new NSLImportNode(modulePath, isWildcard: true);
            }

            // Regular module import: import math or import math as m
            var path = ParseModulePath();

            // Check for alias: import math as m
            string? alias = null;
            if (Match(TokenType.As))
            {
                alias = Consume(TokenType.Identifier, "Expected alias name after 'as'").Value;
            }

            ConsumeStatementEnd();
            return new NSLImportNode(path, moduleAlias: alias);
        }

        /// <summary>
        /// Parse import items list: { sin, cos as cosine, Matrix }
        /// </summary>
        private List<NSLImportItem> ParseImportItems()
        {
            var items = new List<NSLImportItem>();

            if (!Check(TokenType.RightBrace))
            {
                do
                {
                    var name = Consume(TokenType.Identifier, "Expected import item name").Value;
                    string? alias = null;

                    if (Match(TokenType.As))
                    {
                        alias = Consume(TokenType.Identifier, "Expected alias name after 'as'").Value;
                    }

                    items.Add(new NSLImportItem(name, alias));
                } while (Match(TokenType.Comma));
            }

            return items;
        }

        /// <summary>
        /// Parse module path: math or math::linear_algebra
        /// </summary>
        private List<string> ParseModulePath()
        {
            var path = new List<string>();

            path.Add(Consume(TokenType.Identifier, "Expected module name").Value);

            while (Match(TokenType.Chain))
            {
                path.Add(Consume(TokenType.Identifier, "Expected module name after '::'").Value);
            }

            return path;
        }

        /// <summary>
        /// Parse export statement
        /// Supports:
        /// - export { foo, bar }                   (export list)
        /// - export { foo as publicFoo }          (aliased export)
        /// - export * from "submodule"             (re-export)
        /// </summary>
        private NSLASTNode ParseExportStatement()
        {
            // Check for wildcard re-export: export * from "submodule"
            if (Match(TokenType.Multiply))
            {
                Consume(TokenType.From, "Expected 'from' after '*'");
                var modulePath = ParseModulePath();
                ConsumeStatementEnd();
                return new NSLExportNode(
                    isReExport: true,
                    reExportFrom: modulePath,
                    isWildcardReExport: true
                );
            }

            // Export list: export { foo, bar }
            if (Match(TokenType.LeftBrace))
            {
                var items = ParseExportItems();
                Consume(TokenType.RightBrace, "Expected '}' after export items");

                // Check for re-export: export { foo } from "submodule"
                if (Match(TokenType.From))
                {
                    var modulePath = ParseModulePath();
                    ConsumeStatementEnd();
                    return new NSLExportNode(
                        items: items,
                        isReExport: true,
                        reExportFrom: modulePath
                    );
                }

                ConsumeStatementEnd();
                return new NSLExportNode(items: items);
            }

            throw new NSLParseException("Expected '{' or '*' after 'export'");
        }

        /// <summary>
        /// Parse export items list: { foo, bar as publicBar }
        /// </summary>
        private List<NSLExportItem> ParseExportItems()
        {
            var items = new List<NSLExportItem>();

            if (!Check(TokenType.RightBrace))
            {
                do
                {
                    var name = Consume(TokenType.Identifier, "Expected export item name").Value;
                    string? alias = null;

                    if (Match(TokenType.As))
                    {
                        alias = Consume(TokenType.Identifier, "Expected alias name after 'as'").Value;
                    }

                    items.Add(new NSLExportItem(name, alias));
                } while (Match(TokenType.Comma));
            }

            return items;
        }

        /// <summary>
        /// Parse module declaration
        /// Example: module math::linear_algebra { ... }
        /// </summary>
        private NSLASTNode ParseModuleDeclaration()
        {
            var path = ParseModulePath();
            var body = ParseBlock();
            return new NSLModuleNode(path, body);
        }

        /// <summary>
        /// Parse public declaration (pub keyword)
        /// Supports:
        /// - pub fn foo() { ... }
        /// - pub let X = 10
        /// - pub type Alias = ...
        /// </summary>
        private NSLASTNode ParsePublicDeclaration()
        {
            NSLASTNode declaration;

            if (Match(TokenType.Async))
            {
                // pub async fn ...
                declaration = ParseAsyncFunction();
            }
            else if (Match(TokenType.Function))
            {
                declaration = ParseFunctionStatement();
            }
            else if (Match(TokenType.Let))
            {
                declaration = ParseVariableDeclaration(isMutable: false);
            }
            else if (Match(TokenType.Mut))
            {
                declaration = ParseVariableDeclaration(isMutable: true);
            }
            else if (Match(TokenType.Const))
            {
                declaration = ParseVariableDeclaration(isMutable: false);
            }
            else if (Match(TokenType.Type))
            {
                declaration = ParseTypeAlias();
            }
            else if (Match(TokenType.Struct))
            {
                declaration = ParseStructDefinition();
            }
            else if (Match(TokenType.Class))
            {
                // Parse class if we have it
                var nameToken = Consume(TokenType.Identifier, "Expected class name");
                var body = ParseBlock();
                declaration = new NSLClassNode(nameToken.Value, body);
            }
            else
            {
                throw new NSLParseException("Expected declaration after 'pub'");
            }

            return new NSLExportNode(declaration: declaration);
        }

        #endregion

        /// <summary>
        /// AI-friendly: Parse pattern matching statement
        /// Example: match value { case 0 => "zero", case n => "other" }
        /// </summary>
        private NSLASTNode ParseMatchStatement()
        {
            var value = ParseExpression();
            Consume(TokenType.LeftBrace, "Expected '{' after match expression");
            ConsumeNewlines();

            var cases = new List<NSLMatchCase>();
            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                ConsumeNewlines();
                if (Check(TokenType.RightBrace)) break;  // Handle trailing newlines

                Consume(TokenType.Case, "Expected 'case' in match statement");

                // Parse pattern (can be literal, identifier, function-like ok(v), err(e))
                var pattern = ParseMatchPattern();

                // Check for optional guard clause: when condition
                NSLASTNode? guard = null;
                if (Match(TokenType.When))
                {
                    guard = ParseExpression();
                }

                // Fat arrow separates pattern from body
                Consume(TokenType.FatArrow, "Expected '=>' after pattern");

                // Parse body (can be expression or block)
                NSLASTNode body;
                if (Check(TokenType.LeftBrace))
                {
                    body = ParseBlock();
                }
                else
                {
                    body = ParseExpression();
                }

                cases.Add(new NSLMatchCase(pattern, body, guard));
                ConsumeNewlines();
            }

            Consume(TokenType.RightBrace, "Expected '}' after match cases");
            return new NSLMatchNode(value, cases);
        }

        /// <summary>
        /// Parse a pattern in a match expression
        /// Handles: literals, identifiers, ok(v), err(e), some(v), none
        /// </summary>
        private NSLASTNode ParseMatchPattern()
        {
            // Handle result/optional patterns: ok(v), err(e), some(v)
            // Check for ok/err as identifiers (they're no longer keywords)
            if (Check(TokenType.Identifier) && Peek().Value == "ok")
            {
                Advance(); // consume 'ok'
                Consume(TokenType.LeftParen, "Expected '(' after 'ok'");
                var binding = ParseMatchPattern();
                Consume(TokenType.RightParen, "Expected ')' after ok pattern");
                return new NSLResultNode(isOk: true, binding);
            }

            if (Check(TokenType.Identifier) && Peek().Value == "err")
            {
                Advance(); // consume 'err'
                Consume(TokenType.LeftParen, "Expected '(' after 'err'");
                var binding = ParseMatchPattern();
                Consume(TokenType.RightParen, "Expected ')' after err pattern");
                return new NSLResultNode(isOk: false, binding);
            }

            if (Match(TokenType.Some))
            {
                Consume(TokenType.LeftParen, "Expected '(' after 'some'");
                var binding = ParseMatchPattern();
                Consume(TokenType.RightParen, "Expected ')' after some pattern");
                return new NSLOptionalNode(hasValue: true, binding);
            }

            if (Match(TokenType.None))
            {
                return new NSLOptionalNode(hasValue: false);
            }

            // Handle literals and identifiers
            if (Match(TokenType.Number))
            {
                var token = Previous();
                if (double.TryParse(token.Value, out var number))
                    return new NSLLiteralNode(number, TokenType.Number);
                throw new NSLParseException($"Invalid number: {token.Value}");
            }

            if (Match(TokenType.Integer))
            {
                var token = Previous();
                if (long.TryParse(token.Value, out var integer))
                    return new NSLLiteralNode(integer, TokenType.Integer);
                throw new NSLParseException($"Invalid integer: {token.Value}");
            }

            if (Match(TokenType.String))
            {
                var token = Previous();
                var stringValue = token.Value;
                if (stringValue.Length >= 2 && stringValue.StartsWith("\"") && stringValue.EndsWith("\""))
                {
                    stringValue = stringValue[1..^1];
                }
                return new NSLLiteralNode(stringValue, TokenType.String);
            }

            if (Match(TokenType.True))
            {
                return new NSLLiteralNode(true, TokenType.True);
            }

            if (Match(TokenType.False))
            {
                return new NSLLiteralNode(false, TokenType.False);
            }

            if (Match(TokenType.Null))
            {
                return new NSLLiteralNode(null!, TokenType.Null);
            }

            if (Match(TokenType.Identifier))
            {
                return new NSLIdentifierNode(Previous().Value);
            }

            throw new NSLParseException($"Expected pattern, got {Peek().Type}");
        }

        private NSLASTNode ParseExpressionStatement()
        {
            var expr = ParseExpression();
            ConsumeStatementEnd();
            return expr;
        }

        private NSLASTNode ParseExpression()
        {
            return ParseChainExpression();
        }

        private NSLASTNode ParseChainExpression()
        {
            var expressions = new List<NSLASTNode>();
            expressions.Add(ParseAssignment());

            while (Match(TokenType.Chain))
            {
                // Check if this is an enum variant: EnumName::VariantName or EnumName::VariantName(args)
                if (expressions.Count == 1 && expressions[0] is NSLIdentifierNode enumNameNode && Check(TokenType.Identifier))
                {
                    var variantToken = Advance();
                    var variantName = variantToken.Value;

                    // Check for variant arguments: Color::Circle(5.0)
                    var arguments = new List<NSLASTNode>();
                    if (Match(TokenType.LeftParen))
                    {
                        if (!Check(TokenType.RightParen))
                        {
                            do
                            {
                                arguments.Add(ParseExpression());
                            } while (Match(TokenType.Comma));
                        }
                        Consume(TokenType.RightParen, "Expected ')' after variant arguments");
                    }

                    return new NSLEnumVariantNode(enumNameNode.Name, variantName, arguments);
                }
                else
                {
                    expressions.Add(ParseAssignment());
                }
            }

            return expressions.Count == 1 ? expressions[0] : new NSLChainNode(expressions);
        }

        private NSLASTNode ParseAssignment()
        {
            var expr = ParsePipeline();

            if (Match(TokenType.Assign))
            {
                if (expr is NSLIdentifierNode identifier)
                {
                    var value = ParseAssignment();
                    return new NSLAssignmentNode(identifier.Name, value);
                }
                if (expr is NSLIndexNode indexNode)
                {
                    var value = ParseAssignment();
                    return new NSLIndexAssignmentNode(indexNode.Object, indexNode.Index, value);
                }
                // Support property assignment: obj.prop = value
                if (expr is NSLGetNode getNode)
                {
                    var value = ParseAssignment();
                    return new NSLPropertyAssignmentNode(getNode.Object, getNode.Name, value);
                }
                throw new NSLParseException("Invalid assignment target");
            }

            return expr;
        }

        /// <summary>
        /// AI-friendly: Parse pipeline and consciousness operators
        /// |>  pipe         - Chain transformations (data |> normalize |> encode)
        /// ~>  awareness    - Introspective flow
        /// *>  attention    - Focus mechanism
        /// +>  superposition- Quantum-like states
        /// =>> gradient     - Learning/adjustment
        /// </summary>
        private NSLASTNode ParsePipeline()
        {
            var expr = ParseNullCoalescing();

            while (Check(TokenType.PipeArrow) || Check(TokenType.AwarenessArrow) ||
                   Check(TokenType.AttentionArrow) || Check(TokenType.SuperpositionArrow) ||
                   Check(TokenType.GradientArrow))
            {
                var op = Advance();
                var right = ParseNullCoalescing();

                expr = op.Type switch
                {
                    TokenType.PipeArrow => new NSLPipelineNode(expr, right),
                    TokenType.AwarenessArrow => new NSLConsciousnessNode(expr, right, "awareness"),
                    TokenType.AttentionArrow => new NSLConsciousnessNode(expr, right, "attention"),
                    TokenType.SuperpositionArrow => new NSLConsciousnessNode(expr, right, "superposition"),
                    TokenType.GradientArrow => new NSLConsciousnessNode(expr, right, "gradient"),
                    _ => new NSLPipelineNode(expr, right)
                };
            }

            return expr;
        }

        /// <summary>
        /// AI-friendly: Parse null coalescing operator (??)
        /// Provides default values: data ?? "default"
        /// </summary>
        private NSLASTNode ParseNullCoalescing()
        {
            var expr = ParseLogicalOr();

            while (Match(TokenType.QuestionQuestion))
            {
                var op = Previous().Type;
                var right = ParseLogicalOr();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        private NSLASTNode ParseLogicalOr()
        {
            // Handle if expressions at this level
            if (Check(TokenType.If))
            {
                return ParseIfExpression();
            }

            var expr = ParseLogicalAnd();

            while (Match(TokenType.Or))
            {
                var op = Previous().Type;
                var right = ParseLogicalAnd();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        private NSLASTNode ParseLogicalAnd()
        {
            var expr = ParseBitwiseOr();

            while (Match(TokenType.And))
            {
                var op = Previous().Type;
                var right = ParseBitwiseOr();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        /// <summary>
        /// Parse bitwise OR (|) - lowest precedence bitwise operator
        /// </summary>
        private NSLASTNode ParseBitwiseOr()
        {
            var expr = ParseBitwiseXor();

            while (Match(TokenType.BitwiseOr))
            {
                var op = Previous().Type;
                var right = ParseBitwiseXor();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        /// <summary>
        /// Parse bitwise XOR (^)
        /// </summary>
        private NSLASTNode ParseBitwiseXor()
        {
            var expr = ParseBitwiseAnd();

            while (Match(TokenType.BitwiseXor))
            {
                var op = Previous().Type;
                var right = ParseBitwiseAnd();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        /// <summary>
        /// Parse bitwise AND (&) - highest precedence bitwise operator
        /// </summary>
        private NSLASTNode ParseBitwiseAnd()
        {
            var expr = ParseEquality();

            while (Match(TokenType.BitwiseAnd))
            {
                var op = Previous().Type;
                var right = ParseEquality();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        private NSLASTNode ParseEquality()
        {
            var expr = ParseComparison();

            while (Match(TokenType.Equal, TokenType.NotEqual))
            {
                var op = Previous().Type;
                var right = ParseComparison();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        private NSLASTNode ParseComparison()
        {
            var expr = ParseShift();

            while (Match(TokenType.Greater, TokenType.GreaterEqual, TokenType.Less, TokenType.LessEqual))
            {
                var op = Previous().Type;
                var right = ParseShift();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        /// <summary>
        /// Parse shift operators (<< and >>)
        /// </summary>
        private NSLASTNode ParseShift()
        {
            var expr = ParseRange();

            while (Match(TokenType.LeftShift, TokenType.RightShift))
            {
                var op = Previous().Type;
                var right = ParseRange();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        /// <summary>
        /// AI-friendly: Parse range operators (.. and ..=)
        /// Eliminates off-by-one errors: 0..10 (exclusive), 0..=10 (inclusive)
        /// </summary>
        private NSLASTNode ParseRange()
        {
            var expr = ParseTerm();

            if (Match(TokenType.DotDot))
            {
                // Exclusive range: 0..10
                var end = ParseTerm();
                return new NSLRangeNode(expr, end, isInclusive: false);
            }
            if (Match(TokenType.DotDotEqual))
            {
                // Inclusive range: 0..=10
                var end = ParseTerm();
                return new NSLRangeNode(expr, end, isInclusive: true);
            }

            return expr;
        }

        private NSLASTNode ParseTerm()
        {
            var expr = ParseFactor();

            while (Match(TokenType.Minus, TokenType.Plus))
            {
                var op = Previous().Type;
                var right = ParseFactor();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        private NSLASTNode ParseFactor()
        {
            var expr = ParseMatrixMultiply();

            while (Match(TokenType.Divide, TokenType.Multiply, TokenType.Power, TokenType.Percent, TokenType.IntegerDivide))
            {
                var op = Previous().Type;
                var right = ParseMatrixMultiply();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        /// <summary>
        /// AI-friendly: Parse matrix multiply operator (@)
        /// Built-in for ML: weights @ input
        /// </summary>
        private NSLASTNode ParseMatrixMultiply()
        {
            var expr = ParseTypeCast();

            while (Match(TokenType.AtSign))
            {
                var op = Previous().Type;
                var right = ParseTypeCast();
                expr = new NSLBinaryOperationNode(expr, op, right);
            }

            return expr;
        }

        /// <summary>
        /// AI-friendly: Parse type cast operator (as)
        /// Example: value as number, data as vec
        /// </summary>
        private NSLASTNode ParseTypeCast()
        {
            var expr = ParseUnary();

            while (Match(TokenType.As))
            {
                var targetType = ParseTypeHint();
                expr = new NSLCastNode(expr, targetType);
            }

            return expr;
        }

        private NSLASTNode ParseUnary()
        {
            // Handle await expression
            if (Match(TokenType.Await))
            {
                var expr = ParseUnary();
                return new NSLAwaitNode(expr);
            }

            if (Match(TokenType.Not, TokenType.Minus, TokenType.BitwiseNot))
            {
                var op = Previous().Type;
                var right = ParseUnary();
                return new NSLUnaryOperationNode(op, right);
            }

            return ParseConsciousness();
        }

        private NSLASTNode ParseConsciousness()
        {
            // Handle unary consciousness operators (◈, ∇, Ψ, σ, ↓, ≈, ∫)
            if (Match(TokenType.Holographic, TokenType.Gradient, TokenType.Psi,
                      TokenType.Sigma, TokenType.Collapse, TokenType.Similarity,
                      TokenType.Dissimilarity, TokenType.Integral))
            {
                var op = Previous().Type;
                Consume(TokenType.LeftBracket, "Expected '[' after consciousness operator");
                var operand = ParseExpression();
                Consume(TokenType.RightBracket, "Expected ']' after consciousness operand");
                return new NSLUnaryOperationNode(op, operand);
            }

            // Handle tensor product operator (⊗) - binary operator
            if (Match(TokenType.TensorProduct))
            {
                var op = Previous().Type;
                Consume(TokenType.LeftBracket, "Expected '[' after ⊗");
                var left = ParseExpression();
                Consume(TokenType.Comma, "Expected ',' in tensor product");
                var right = ParseExpression();
                Consume(TokenType.RightBracket, "Expected ']' after tensor product");
                return new NSLBinaryOperationNode(left, op, right);
            }

            // Handle memory operator (μ) - can be unary or binary
            // μ[key] - recall from memory
            // μ[key, value] - store to memory
            if (Match(TokenType.Mu))
            {
                var op = Previous().Type;
                Consume(TokenType.LeftBracket, "Expected '[' after μ");
                var key = ParseExpression();

                if (Match(TokenType.Comma))
                {
                    // Store operation: μ[key, value]
                    var value = ParseExpression();
                    Consume(TokenType.RightBracket, "Expected ']' after memory store");
                    return new NSLBinaryOperationNode(key, TokenType.MuStore, value);
                }
                else
                {
                    // Recall operation: μ[key]
                    Consume(TokenType.RightBracket, "Expected ']' after memory recall");
                    return new NSLUnaryOperationNode(TokenType.MuRecall, key);
                }
            }

            // Handle uncertainty operator (±)
            // ±[value, uncertainty] - creates uncertain value
            if (Match(TokenType.PlusMinus))
            {
                var op = Previous().Type;
                Consume(TokenType.LeftBracket, "Expected '[' after ±");
                var value = ParseExpression();
                Consume(TokenType.Comma, "Expected ',' in uncertainty expression");
                var uncertainty = ParseExpression();
                Consume(TokenType.RightBracket, "Expected ']' after uncertainty");
                return new NSLBinaryOperationNode(value, op, uncertainty);
            }

            return ParseCall();
        }

        private NSLASTNode ParseCall()
        {
            var expr = ParsePrimary();

            while (true)
            {
                if (Match(TokenType.LeftParen))
                {
                    expr = FinishCall(expr);
                }
                else if (Match(TokenType.Dot))
                {
                    var name = ConsumePropertyName("Expected property name after '.'");
                    expr = new NSLGetNode(expr, name);
                }
                // AI-friendly: Safe navigation operator (?.)
                // Prevents null reference errors: obj?.property
                else if (Match(TokenType.QuestionDot))
                {
                    var name = ConsumePropertyName("Expected property name after '?.'");
                    expr = new NSLSafeNavigationNode(expr, name);
                }
                // Index operation: expr[index]
                // Only treat [ as index operation if on the same line as the expression
                // Otherwise it's likely an array literal on a new line
                else if (Check(TokenType.LeftBracket) && Peek().Line == Previous().Line)
                {
                    Advance(); // consume [
                    var index = ParseExpression();
                    Consume(TokenType.RightBracket, "Expected ']' after index");
                    expr = new NSLIndexNode(expr, index);
                }
                // Struct instantiation: StructName { field1: value1, field2: value2 }
                // Only parse as struct if the next token after { is an identifier followed by :
                // This avoids ambiguity with match blocks: match value { case ... }
                else if (Check(TokenType.LeftBrace) && expr is NSLIdentifierNode structName && IsStructInstantiation())
                {
                    Advance(); // consume '{'
                    expr = ParseStructInstantiation(structName.Name);
                }
                else
                {
                    break;
                }
            }

            return expr;
        }

        /// <summary>
        /// Check if what follows is a struct instantiation (identifier: value, ...)
        /// rather than a match/case block or other brace-delimited construct
        /// </summary>
        private bool IsStructInstantiation()
        {
            if (!Check(TokenType.LeftBrace))
                return false;

            // Save position for lookahead
            int savedPosition = _current;
            Advance(); // consume '{'

            // Skip any newlines after '{'
            while (Check(TokenType.Newline))
                Advance();

            // If the next token is 'case', this is a match block, not a struct
            if (Check(TokenType.Case))
            {
                _current = savedPosition;
                return false;
            }

            // If the next token is '}', this is an empty struct
            if (Check(TokenType.RightBrace))
            {
                _current = savedPosition;
                return true;
            }

            // Check if it's an identifier followed by ':'
            if (Check(TokenType.Identifier))
            {
                Advance(); // consume identifier
                bool isStruct = Check(TokenType.Colon);
                _current = savedPosition;
                return isStruct;
            }

            // Not a struct instantiation
            _current = savedPosition;
            return false;
        }

        /// <summary>
        /// Parse struct instantiation: Point { x: 10, y: 20 }
        /// Called after consuming the struct name and '{'
        /// </summary>
        private NSLASTNode ParseStructInstantiation(string structName)
        {
            var fields = new List<NSLObjectField>();
            ConsumeNewlines();

            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                ConsumeNewlines();
                if (Check(TokenType.RightBrace)) break;

                // Parse field name
                var fieldName = Consume(TokenType.Identifier, "Expected field name in struct instantiation");
                Consume(TokenType.Colon, "Expected ':' after field name");

                // Parse field value
                var fieldValue = ParseExpression();

                fields.Add(new NSLObjectField(fieldName.Value, fieldValue));

                // Handle optional comma between fields
                if (!Check(TokenType.RightBrace))
                {
                    Match(TokenType.Comma);
                }
                ConsumeNewlines();
            }

            Consume(TokenType.RightBrace, "Expected '}' after struct fields");
            return new NSLStructInstantiationNode(structName, fields);
        }

        private NSLASTNode FinishCall(NSLASTNode callee)
        {
            var arguments = new List<NSLASTNode>();

            if (!Check(TokenType.RightParen))
            {
                do
                {
                    arguments.Add(ParseExpression());
                } while (Match(TokenType.Comma));
            }

            Consume(TokenType.RightParen, "Expected ')' after arguments");
            return new NSLFunctionCallNode(callee, arguments);
        }

        /// <summary>
        /// AI-friendly: Try to parse lambda parameters
        /// Returns a lambda node if successful, null if this is a grouped expression
        /// Called when we see '(' and need to determine if it's (a, b) => ... or (expr)
        /// </summary>
        private NSLASTNode? TryParseLambdaParams()
        {
            // Save current position for backtracking
            int savedPosition = _current;

            var parameters = new List<NSLParameter>();

            // Handle empty parameter list: () => ...
            if (Match(TokenType.RightParen))
            {
                if (Match(TokenType.FatArrow))
                {
                    var body = ParseExpression();
                    return new NSLLambdaNode(parameters, body);
                }
                // Not a lambda, backtrack
                _current = savedPosition;
                return null;
            }

            // Try to parse identifier list
            while (true)
            {
                if (!Check(TokenType.Identifier))
                {
                    // Not a parameter list, backtrack
                    _current = savedPosition;
                    return null;
                }

                var paramName = Advance().Value;

                // Optional type hint for parameter: (a: number, b: string) => ...
                string? paramType = null;
                if (Match(TokenType.Colon))
                {
                    if (Check(TokenType.Identifier) || Check(TokenType.Vec) ||
                        Check(TokenType.Mat) || Check(TokenType.Tensor) ||
                        Check(TokenType.Prob))
                    {
                        paramType = ParseTypeHint();
                    }
                    else
                    {
                        // Invalid type, backtrack
                        _current = savedPosition;
                        return null;
                    }
                }

                parameters.Add(new NSLParameter(paramName, paramType));

                if (Match(TokenType.Comma))
                {
                    // More parameters
                    continue;
                }

                if (Match(TokenType.RightParen))
                {
                    // End of parameter list, check for =>
                    if (Match(TokenType.FatArrow))
                    {
                        var body = ParseExpression();
                        return new NSLLambdaNode(parameters, body);
                    }
                    // Not a lambda (e.g., just (a, b)), backtrack
                    _current = savedPosition;
                    return null;
                }

                // Unexpected token, backtrack
                _current = savedPosition;
                return null;
            }
        }

        private NSLASTNode ParsePrimary()
        {
            // Handle boolean literals
            if (Match(TokenType.True))
                return new NSLLiteralNode(true, TokenType.True);

            if (Match(TokenType.False))
                return new NSLLiteralNode(false, TokenType.False);

            if (Match(TokenType.Null))
                return new NSLLiteralNode(null!, TokenType.Null);

            if (Match(TokenType.Number))
            {
                var token = Previous();
                if (double.TryParse(token.Value, out var number))
                    return new NSLLiteralNode(number, TokenType.Number);
                throw new NSLParseException($"Invalid number: {token.Value}");
            }

            if (Match(TokenType.Integer))
            {
                var token = Previous();
                if (long.TryParse(token.Value, out var integer))
                    return new NSLLiteralNode(integer, TokenType.Integer);
                throw new NSLParseException($"Invalid integer: {token.Value}");
            }

            if (Match(TokenType.String))
            {
                var token = Previous();
                // Remove quotes from string value
                var stringValue = token.Value;
                if (stringValue.Length >= 2 && stringValue.StartsWith("\"") && stringValue.EndsWith("\""))
                {
                    stringValue = stringValue[1..^1];
                }
                return new NSLLiteralNode(stringValue, TokenType.String);
            }

            // AI-friendly: Result types (ok/err) for explicit error handling
            // Example: ok(42), err("not found")
            // ok and err are now identifiers (not keywords) to allow using them as variable names
            if (Check(TokenType.Identifier) && Peek().Value == "ok" && CheckNext(TokenType.LeftParen))
            {
                Advance(); // consume 'ok'
                Consume(TokenType.LeftParen, "Expected '(' after 'ok'");
                var value = ParseExpression();
                Consume(TokenType.RightParen, "Expected ')' after ok value");
                return new NSLResultNode(isOk: true, value);
            }

            if (Check(TokenType.Identifier) && Peek().Value == "err" && CheckNext(TokenType.LeftParen))
            {
                Advance(); // consume 'err'
                Consume(TokenType.LeftParen, "Expected '(' after 'err'");
                var value = ParseExpression();
                Consume(TokenType.RightParen, "Expected ')' after err value");
                return new NSLResultNode(isOk: false, value);
            }

            // AI-friendly: Optional types (some/none) for explicit null handling
            // Example: some(value), none
            if (Match(TokenType.Some))
            {
                Consume(TokenType.LeftParen, "Expected '(' after 'some'");
                var value = ParseExpression();
                Consume(TokenType.RightParen, "Expected ')' after some value");
                return new NSLOptionalNode(hasValue: true, value);
            }

            if (Match(TokenType.None))
            {
                return new NSLOptionalNode(hasValue: false);
            }

            // Check for identifier that could be start of lambda: x => x * 2
            if (Match(TokenType.Identifier))
            {
                var identToken = Previous();

                // AI-friendly: Fat arrow lambda (single parameter)
                // Example: x => x * 2
                if (Check(TokenType.FatArrow))
                {
                    Advance(); // consume =>
                    var body = ParseExpression();
                    return new NSLLambdaNode(
                        new[] { new NSLParameter(identToken.Value) },
                        body
                    );
                }

                return new NSLIdentifierNode(identToken.Value);
            }

            // Handle if expressions in primary expressions
            if (Check(TokenType.If))
            {
                return ParseIfExpression();
            }

            // Handle inline anonymous functions: fn() { ... } or fn(a, b) { ... }
            if (Match(TokenType.Function))
            {
                return ParseAnonymousFunction();
            }

            // Handle type() as function call (type is keyword but also builtin function)
            if (Check(TokenType.Type) && CheckNext(TokenType.LeftParen))
            {
                Advance(); // consume 'type'
                return new NSLIdentifierNode("type");
            }

            if (Match(TokenType.LeftParen))
            {
                // Check if this is a lambda: (a, b) => a + b
                // or just a grouped expression: (x + y)
                if (Check(TokenType.Identifier) || Check(TokenType.RightParen))
                {
                    var mayBeLambda = TryParseLambdaParams();
                    if (mayBeLambda != null)
                    {
                        return mayBeLambda;
                    }
                }

                // Regular grouped expression
                var expr = ParseExpression();
                Consume(TokenType.RightParen, "Expected ')' after expression");
                return expr;
            }

            if (Match(TokenType.LeftBracket))
            {
                // Check for empty array first
                if (Check(TokenType.RightBracket))
                {
                    Advance(); // Consume ]
                    return new NSLArrayNode(new List<NSLASTNode>());
                }

                // Parse the first expression
                var firstExpr = ParseExpression();

                // Check if this is a list comprehension: [expr for var in iterable]
                if (Check(TokenType.For))
                {
                    Advance(); // Consume 'for'

                    // Parse variable name
                    var varToken = Consume(TokenType.Identifier, "Expected variable name in list comprehension");
                    var variable = varToken.Value;

                    // Consume 'in'
                    Consume(TokenType.In, "Expected 'in' in list comprehension");

                    // Parse iterable
                    var iterable = ParseExpression();

                    // Check for optional filter condition: if condition
                    NSLASTNode? condition = null;
                    if (Check(TokenType.If))
                    {
                        Advance(); // Consume 'if'
                        condition = ParseExpression();
                    }

                    Consume(TokenType.RightBracket, "Expected ']' after list comprehension");
                    return new NSLListComprehensionNode(firstExpr, variable, iterable, condition);
                }

                // Regular array literal
                var elements = new List<NSLASTNode> { firstExpr };

                while (Match(TokenType.Comma))
                {
                    elements.Add(ParseExpression());
                }

                Consume(TokenType.RightBracket, "Expected ']' after array elements");
                return new NSLArrayNode(elements);
            }

            // AI-friendly: Object literal / record type
            // Example: {x: number, y: number} or {name: "Claude", age: 1}
            if (Match(TokenType.LeftBrace))
            {
                return ParseObjectLiteral();
            }

            throw new NSLParseException($"Unexpected token: {Peek().Type}");
        }

        /// <summary>
        /// AI-friendly: Parse object literal / record type
        /// Example: {x: number, y: number} or {name: "Claude", age: 1}
        /// </summary>
        private NSLASTNode ParseObjectLiteral()
        {
            var fields = new List<NSLObjectField>();
            ConsumeNewlines();

            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                ConsumeNewlines();
                if (Check(TokenType.RightBrace)) break;

                // Parse key (identifier or string)
                NSLToken keyToken;
                if (Check(TokenType.String))
                {
                    keyToken = Advance();
                }
                else
                {
                    keyToken = Consume(TokenType.Identifier, "Expected field name in object literal");
                }

                // Parse colon separator
                Consume(TokenType.Colon, "Expected ':' after field name");

                // Parse value (can be type name or expression)
                NSLASTNode value;
                if (Check(TokenType.Identifier) || Check(TokenType.Vec) ||
                    Check(TokenType.Mat) || Check(TokenType.Tensor) ||
                    Check(TokenType.Prob))
                {
                    // Could be a type name (for type definitions) or an expression
                    value = ParseExpression();
                }
                else
                {
                    value = ParseExpression();
                }

                fields.Add(new NSLObjectField(keyToken.Value, value));

                // Handle optional comma between fields
                if (!Check(TokenType.RightBrace))
                {
                    Match(TokenType.Comma);
                }
                ConsumeNewlines();
            }

            Consume(TokenType.RightBrace, "Expected '}' after object literal");
            return new NSLObjectNode(fields);
        }

        private NSLASTNode ParseIfExpression()
        {
            Consume(TokenType.If, "Expected 'if' keyword");
            
            // Support optional parentheses around condition
            bool hasParens = Match(TokenType.LeftParen);
            var condition = ParseLogicalAnd(); // Use ParseLogicalAnd to avoid recursion
            if (hasParens)
            {
                Consume(TokenType.RightParen, "Expected ')' after condition");
            }
            
            // Parse then branch - single expression in braces
            Consume(TokenType.LeftBrace, "Expected '{' for if expression then branch");
            var thenBranch = ParseLogicalAnd(); // Parse single expression, avoid recursion
            Consume(TokenType.RightBrace, "Expected '}' after then branch");
            
            NSLASTNode? elseBranch = null;
            if (Match(TokenType.Else))
            {
                Consume(TokenType.LeftBrace, "Expected '{' for if expression else branch");
                elseBranch = ParseLogicalAnd(); // Parse single expression, avoid recursion
                Consume(TokenType.RightBrace, "Expected '}' after else branch");
            }
            
            return new NSLIfNode(condition, thenBranch, elseBranch);
        }

        #region Control Flow Parsing Methods

        private NSLASTNode ParseIfStatement()
        {
            // Support optional parentheses around condition
            bool hasParens = Match(TokenType.LeftParen);
            
            var condition = ParseExpression();
            
            if (hasParens)
            {
                Consume(TokenType.RightParen, "Expected ')' after condition");
            }

            // Support optional 'then' keyword (AI-friendly syntax)
            Match(TokenType.Then);

            ConsumeNewlines(); // Allow newline before { or statement

            NSLASTNode thenBranch;
            
            // Check if we have a block or a single statement
            if (Check(TokenType.LeftBrace))
            {
                // Block with braces
                thenBranch = ParseBlock();
            }
            else
            {
                // Single statement without braces
                var stmt = ParseStatement();
                // Wrap single statement in a block node for consistency
                thenBranch = new NSLBlockNode(new List<NSLASTNode> { stmt });
            }
            
            NSLASTNode? elseBranch = null;
            
            // Only check for else if we're not in the middle of a statement
            // This prevents "else" from being consumed when it's part of something else
            if (!IsInMiddleOfStatement())
            {
                ConsumeNewlines(); // Allow newlines between statement and else
                
                if (Match(TokenType.Else))
                {
                    ConsumeNewlines(); // Allow newline before { or statement
                    
                    if (Check(TokenType.If))
                    {
                        // else if - parse as another if statement
                        elseBranch = ParseStatement();
                    }
                    else if (Check(TokenType.LeftBrace))
                    {
                        // else with block
                        elseBranch = ParseBlock();
                    }
                    else
                    {
                        // else with single statement
                        var stmt = ParseStatement();
                        elseBranch = new NSLBlockNode(new List<NSLASTNode> { stmt });
                    }
                }
            }
            
            return new NSLIfNode(condition, thenBranch, elseBranch);
        }

        private NSLASTNode ParseWhileStatement()
        {
            // Support optional parentheses around condition
            bool hasParens = Match(TokenType.LeftParen);
            
            var condition = ParseExpression();
            
            if (hasParens)
            {
                Consume(TokenType.RightParen, "Expected ')' after condition");
            }
            
            ConsumeNewlines(); // Allow newline before { or statement
            
            NSLASTNode body;
            
            // Check if we have a block or a single statement
            if (Check(TokenType.LeftBrace))
            {
                // Block with braces
                body = ParseBlock();
            }
            else
            {
                // Single statement without braces
                var stmt = ParseStatement();
                // Wrap single statement in a block node for consistency
                body = new NSLBlockNode(new List<NSLASTNode> { stmt });
            }
            
            return new NSLWhileNode(condition, body);
        }

        private NSLASTNode ParseForStatement()
        {
            // Python-style for loop: for i in range(10) { ... }
            
            // Parse loop variable
            if (!Check(TokenType.Identifier))
                throw new NSLParseException("Expected identifier after 'for'");
            
            var variable = Advance();
            
            // Consume 'in' keyword
            Consume(TokenType.In, "Expected 'in' after for loop variable");
            
            // Parse iterable expression (should evaluate to a list)
            var iterable = ParseExpression();
            
            ConsumeNewlines();
            
            // Parse body
            NSLASTNode body;
            if (Check(TokenType.LeftBrace))
            {
                body = ParseBlock();
            }
            else
            {
                // Single statement body
                body = ParseStatement();
            }
            
            // Create ForStmt AST node
            return new NSLForNode(variable, iterable, body);
        }

        private NSLASTNode ParseFunctionStatement()
        {
            // Consume function name
            var nameToken = Consume(TokenType.Identifier, "Expected function name");
            string functionName = nameToken.Value;

            // Parse parameters with optional type hints
            // Example: fn add(a: number, b: number) { ... }
            Consume(TokenType.LeftParen, "Expected '(' after function name");
            var parameters = new List<NSLParameter>();

            if (!Check(TokenType.RightParen))
            {
                do
                {
                    var paramToken = Consume(TokenType.Identifier, "Expected parameter name");

                    // AI-friendly: Optional type hint for parameter
                    // Example: (input: vec, weights: mat)
                    string? paramType = null;
                    if (Match(TokenType.Colon))
                    {
                        paramType = ParseTypeHint();
                    }

                    parameters.Add(new NSLParameter(paramToken.Value, paramType));
                } while (Match(TokenType.Comma));
            }

            Consume(TokenType.RightParen, "Expected ')' after parameters");

            // Optional return type annotation: fn foo() -> number { ... }
            string? returnType = null;
            if (Match(TokenType.Arrow))
            {
                returnType = ParseTypeHint();
            }

            // Parse function body
            NSLASTNode body;
            if (Check(TokenType.LeftBrace))
            {
                body = ParseBlock();
            }
            else
            {
                throw new NSLParseException("Expected '{' before function body");
            }

            return new NSLFunctionNode(functionName, parameters, body, returnType);
        }

        /// <summary>
        /// Parse anonymous function in expression context: fn() { ... } or fn(a, b) { ... }
        /// </summary>
        private NSLASTNode ParseAnonymousFunction()
        {
            // Parse parameters
            Consume(TokenType.LeftParen, "Expected '(' after 'fn'");
            var parameters = new List<NSLParameter>();

            if (!Check(TokenType.RightParen))
            {
                do
                {
                    var paramToken = Consume(TokenType.Identifier, "Expected parameter name");
                    string? paramType = null;
                    if (Match(TokenType.Colon))
                    {
                        paramType = ParseTypeHint();
                    }
                    parameters.Add(new NSLParameter(paramToken.Value, paramType));
                } while (Match(TokenType.Comma));
            }

            Consume(TokenType.RightParen, "Expected ')' after parameters");

            // Optional return type annotation
            string? returnType = null;
            if (Match(TokenType.Arrow))
            {
                returnType = ParseTypeHint();
            }

            // Parse function body - can be block or expression
            NSLASTNode body;
            if (Check(TokenType.LeftBrace))
            {
                body = ParseBlock();
            }
            else
            {
                // Single expression body (like arrow functions)
                body = ParseExpression();
            }

            // Return as anonymous function (empty name)
            return new NSLFunctionNode("", parameters, body, returnType);
        }

        private NSLASTNode ParseReturnStatement()
        {
            NSLASTNode? value = null;
            
            if (!Check(TokenType.Newline) && !Check(TokenType.RightBrace) && !IsAtEnd())
            {
                value = ParseExpression();
            }
            
            ConsumeStatementEnd();
            return new NSLReturnNode(value);
        }

        private NSLASTNode ParseBreakStatement()
        {
            ConsumeStatementEnd();
            return new NSLBreakNode();
        }

        private NSLASTNode ParseContinueStatement()
        {
            ConsumeStatementEnd();
            return new NSLContinueNode();
        }

        private NSLBlockNode ParseBlock()
        {
            Consume(TokenType.LeftBrace, "Expected '{'");
            ConsumeNewlines();
            
            var statements = new List<NSLASTNode>();
            
            while (!Check(TokenType.RightBrace) && !IsAtEnd())
            {
                if (Match(TokenType.Newline))
                    continue;
                    
                var stmt = ParseStatement();
                if (stmt != null)
                    statements.Add(stmt);
            }
            
            Consume(TokenType.RightBrace, "Expected '}' after block");
            return new NSLBlockNode(statements);
        }

        private void ConsumeNewlines()
        {
            while (Match(TokenType.Newline)) { }
        }
        
        private bool IsInMiddleOfStatement()
        {
            // We're in the middle of a statement if the current token
            // doesn't typically start a new statement
            var currentType = Peek().Type;
            
            return currentType switch
            {
                TokenType.EndOfFile => false,
                TokenType.RightBrace => false,
                TokenType.Newline => false,
                TokenType.Semicolon => false,
                TokenType.Else => false,
                _ => true
            };
        }

        private void ConsumeStatementEnd()
        {
            // Allow implicit statement end at EOF or before }
            if (IsAtEnd() || Check(TokenType.RightBrace))
            {
                return;
            }
            
            // Also allow statement to end before control flow keywords
            if (Check(TokenType.Else) || Check(TokenType.If) || 
                Check(TokenType.While) || Check(TokenType.Break) || 
                Check(TokenType.Continue))
            {
                return;
            }
            
            // For everything else, consume newlines/semicolons if they exist
            // but don't require them
            while (Match(TokenType.Newline, TokenType.Semicolon))
            {
                // Consume all trailing newlines/semicolons
            }
        }

        #endregion

        // Helper methods
        private bool Match(params TokenType[] types)
        {
            foreach (var type in types)
            {
                if (Check(type))
                {
                    Advance();
                    return true;
                }
            }
            return false;
        }

        private bool Check(TokenType type)
        {
            if (IsAtEnd()) return false;
            return Peek().Type == type;
        }

        private bool CheckNext(TokenType type)
        {
            if (_current + 1 >= _tokens.Count) return false;
            return _tokens[_current + 1].Type == type;
        }

        private NSLToken Advance()
        {
            if (!IsAtEnd()) _current++;
            return Previous();
        }

        private bool IsAtEnd()
        {
            return _current >= _tokens.Count || Peek().Type == TokenType.EndOfFile;
        }

        private NSLToken Peek()
        {
            if (_current >= _tokens.Count)
                return new NSLToken(TokenType.EndOfFile, string.Empty, 0, 0, 0, 0);
            return _tokens[_current];
        }

        private NSLToken Previous()
        {
            if (_current <= 0)
                return new NSLToken(TokenType.EndOfFile, string.Empty, 0, 0, 0, 0);
            return _tokens[_current - 1];
        }

        private NSLToken Consume(TokenType type, string message)
        {
            if (Check(type)) return Advance();

            // Provide more context in error messages
            var actualType = IsAtEnd() ? "EOF" : Peek().Type.ToString();
            var line = IsAtEnd() ? Previous().Line : Peek().Line;

            throw new NSLParseException($"{message}. Got {actualType} at line {line}");
        }

        /// <summary>
        /// Consumes a property name, which can be an identifier or a keyword.
        /// This allows accessing properties like gpu.tensor where tensor is a keyword.
        /// </summary>
        private string ConsumePropertyName(string message)
        {
            var token = Peek();

            // Accept identifier
            if (Check(TokenType.Identifier))
            {
                Advance();
                return token.Value;
            }

            // Accept keywords as property names (e.g., gpu.tensor, obj.type, data.match)
            if (IsKeyword(token.Type))
            {
                Advance();
                return token.Value;
            }

            var actualType = IsAtEnd() ? "EOF" : token.Type.ToString();
            var line = IsAtEnd() ? Previous().Line : token.Line;
            throw new NSLParseException($"{message}. Got {actualType} at line {line}");
        }

        /// <summary>
        /// Checks if a token type is a keyword that can be used as a property name.
        /// </summary>
        private static bool IsKeyword(TokenType type)
        {
            return type switch
            {
                TokenType.Let or TokenType.Mut or TokenType.Const or TokenType.If or
                TokenType.Then or TokenType.Else or TokenType.While or TokenType.For or
                TokenType.In or TokenType.Break or TokenType.Continue or TokenType.Function or
                TokenType.Class or TokenType.Return or TokenType.Match or TokenType.Case or
                TokenType.When or TokenType.Enum or TokenType.And or TokenType.Or or
                TokenType.Not or TokenType.True or TokenType.False or TokenType.Null or
                TokenType.Type or TokenType.Struct or TokenType.Vec or TokenType.Mat or
                TokenType.Tensor or TokenType.Prob or TokenType.Some or TokenType.None or
                TokenType.As or TokenType.Import or TokenType.From or TokenType.Export or
                TokenType.Pub or TokenType.Module or TokenType.Trait or TokenType.Impl or
                TokenType.Async or TokenType.Await or TokenType.Mu or TokenType.Sigma => true,
                _ => false
            };
        }

        /// <summary>
        /// Converts Token to NSLToken
        /// </summary>
        private NSLToken ConvertToNSLToken(Token token)
        {
            // Map logical operators to their NSL equivalents
            // Note: True/False are kept as-is so the parser can recognize them
            var tokenType = token.Type switch
            {
                TokenType.LogicalAnd => TokenType.And,
                TokenType.LogicalOr => TokenType.Or,
                TokenType.LogicalNot => TokenType.Not,
                _ => token.Type
            };
            return new NSLToken(tokenType, token.Value, token.Line, token.Column, token.Position, token.Value.Length);
        }

        private int GetCurrentLine()
        {
            return _current < _tokens.Count ? _tokens[_current].Line : (_tokens.Count > 0 ? _tokens[^1].Line : 1);
        }

        /// <summary>
        /// Get current column number
        /// </summary>
        private int GetCurrentColumn()
        {
            return _current < _tokens.Count ? _tokens[_current].Column : 1;
        }

        /// <summary>
        /// Get source line content for error messages with caret underline
        /// </summary>
        private string GetSourceLineContext(int line, int column = 0)
        {
            if (_sourceLines == null || line < 1 || line > _sourceLines.Length)
                return "";
            
            var sourceLine = _sourceLines[line - 1].TrimEnd();
            var truncated = false;
            if (sourceLine.Length > 80) {
                sourceLine = sourceLine.Substring(0, 77) + "...";
                truncated = true;
            }
            
            var result = $"\n  | {sourceLine}";
            
            // Add caret underline pointing to error column
            if (column > 0 && column <= sourceLine.Length + 3) {
                var caretPos = Math.Min(column - 1, truncated ? 77 : sourceLine.Length);
                result += $"\n  | {new string(' ', caretPos)}^";
            }
            
            return result;
        }

        /// <summary>
        /// Format error with source context, line, column, and optional hints
        /// </summary>
        private string FormatError(string message, int line, int column = 0, string[]? expected = null)
        {
            var location = column > 0 ? $"line {line}, column {column}" : $"line {line}";
            var context = GetSourceLineContext(line, column);
            var hint = expected != null && expected.Length > 0 
                ? $"\n  Expected one of: {string.Join(", ", expected)}" 
                : "";
            return $"Parse error at {location}: {message}{context}{hint}";
        }

        /// <summary>
        /// Format error with current token position
        /// </summary>
        private string FormatErrorAtCurrent(string message, string[]? expected = null)
        {
            return FormatError(message, GetCurrentLine(), GetCurrentColumn(), expected);
        }
    }
}