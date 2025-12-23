using System;
using System.Collections.Generic;
using System.Linq;
using NSLTokenType = NSL.Core.Tokens.TokenType;

namespace NSL.Core.AST
{
    /// <summary>
    /// Base class for all NSL AST nodes
    /// </summary>
    public abstract class NSLASTNode
    {
        /// <summary>Gets or sets the source line number of this AST node.</summary>
        public int Line { get; set; }
        /// <summary>Gets or sets the source column number of this AST node.</summary>
        public int Column { get; set; }
        
        /// <summary>Accepts a visitor for AST traversal using the visitor pattern.</summary>
        /// <typeparam name="T">The return type of the visitor.</typeparam>
        /// <param name="visitor">The visitor to accept.</param>
        /// <returns>The result of visiting this node.</returns>
        public abstract T Accept<T>(INSLVisitor<T> visitor);
    }

    /// <summary>
    /// Visitor interface for AST traversal
    /// </summary>
    public interface INSLVisitor<T>
    {
        T VisitLiteral(NSLLiteralNode node);
        T VisitIdentifier(NSLIdentifierNode node);
        T VisitBinaryOperation(NSLBinaryOperationNode node);
        T VisitUnaryOperation(NSLUnaryOperationNode node);
        T VisitAssignment(NSLAssignmentNode node);
        T VisitIndexAssignment(NSLIndexAssignmentNode node);
        T VisitPropertyAssignment(NSLPropertyAssignmentNode node);
        T VisitChain(NSLChainNode node);
        T VisitLambda(NSLLambdaNode node);
        T VisitFunctionCall(NSLFunctionCallNode node);
        T VisitQuantum(NSLQuantumNode node);
        T VisitConsciousness(NSLConsciousnessNode node);
        T VisitArray(NSLArrayNode node);
        T VisitListComprehension(NSLListComprehensionNode node);
        T VisitGet(NSLGetNode node);
        T VisitIndex(NSLIndexNode node);
        T VisitBlock(NSLBlockNode node);
        T VisitIf(NSLIfNode node);
        T VisitWhile(NSLWhileNode node);
        T VisitFor(NSLForNode node);
        T VisitBreak(NSLBreakNode node);
        T VisitContinue(NSLContinueNode node);
        T VisitReturn(NSLReturnNode node);
        T VisitFunction(NSLFunctionNode node);
        T VisitClass(NSLClassNode node);
        T VisitMatch(NSLMatchNode node);

        // AI-Friendly feature visitors
        T VisitVariableDeclaration(NSLVariableDeclarationNode node);
        T VisitSafeNavigation(NSLSafeNavigationNode node);
        T VisitPipeline(NSLPipelineNode node);
        T VisitRange(NSLRangeNode node);
        T VisitCast(NSLCastNode node);
        T VisitResult(NSLResultNode node);
        T VisitOptional(NSLOptionalNode node);
        T VisitTypeAlias(NSLTypeAliasNode node);
        T VisitObject(NSLObjectNode node);
        T VisitStruct(NSLStructNode node);
        T VisitStructInstantiation(NSLStructInstantiationNode node);
        T VisitEnum(NSLEnumNode node);
        T VisitEnumVariant(NSLEnumVariantNode node);

        // Trait/Interface visitors
        T VisitTrait(NSLTraitNode node);
        T VisitImpl(NSLImplNode node);

        // Async/Await visitors
        T VisitAsyncFunction(NSLAsyncFunctionNode node);
        T VisitAwait(NSLAwaitNode node);

        // Module system visitors
        T VisitModule(NSLModuleNode node);
        T VisitImport(NSLImportNode node);
        T VisitExport(NSLExportNode node);
    }

    #region Expression Nodes

    /// <summary>
    /// Literal value node (numbers, strings, booleans)
    /// </summary>
    public class NSLLiteralNode : NSLASTNode
    {
        /// <summary>Gets the literal value.</summary>
        public object? Value { get; }
        /// <summary>Gets the token type of the literal value.</summary>
        public NSLTokenType ValueType { get; }

        /// <summary>Creates a new literal node with the specified value and type.</summary>
        public NSLLiteralNode(object? value, NSLTokenType valueType)
        {
            Value = value;
            ValueType = valueType;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitLiteral(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Literal({Value})";
        }
    }

    /// <summary>
    /// Identifier node (variable names, function names)
    /// </summary>
    public class NSLIdentifierNode : NSLASTNode
    {
        /// <summary>Gets the identifier name.</summary>
        public string Name { get; }

        /// <summary>Creates a new identifier node with the specified name.</summary>
        public NSLIdentifierNode(string name)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitIdentifier(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Identifier({Name})";
        }
    }

    /// <summary>
    /// Binary operation node (a + b, a :: b, etc.)
    /// </summary>
    public class NSLBinaryOperationNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Left { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLTokenType Operator { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Right { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLBinaryOperationNode(NSLASTNode left, NSLTokenType @operator, NSLASTNode right)
        {
            Left = left ?? throw new ArgumentNullException(nameof(left));
            Operator = @operator;
            Right = right ?? throw new ArgumentNullException(nameof(right));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitBinaryOperation(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"BinaryOp({Left} {Operator} {Right})";
        }
    }

    /// <summary>
    /// Unary operation node (-a, !a, etc.)
    /// </summary>
    public class NSLUnaryOperationNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLTokenType Operator { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Operand { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLUnaryOperationNode(NSLTokenType @operator, NSLASTNode operand)
        {
            Operator = @operator;
            Operand = operand ?? throw new ArgumentNullException(nameof(operand));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitUnaryOperation(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"UnaryOp({Operator} {Operand})";
        }
    }

    /// <summary>
    /// Assignment node (x = value)
    /// </summary>
    public class NSLAssignmentNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLAssignmentNode(string name, NSLASTNode value)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Value = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitAssignment(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Assignment({Name} = {Value})";
        }
    }

    /// <summary>
    /// Index assignment node (arr[i] = value)
    /// </summary>
    public class NSLIndexAssignmentNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Object { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Index { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLIndexAssignmentNode(NSLASTNode obj, NSLASTNode index, NSLASTNode value)
        {
            Object = obj ?? throw new ArgumentNullException(nameof(obj));
            Index = index ?? throw new ArgumentNullException(nameof(index));
            Value = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitIndexAssignment(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"IndexAssignment({Object}[{Index}] = {Value})";
        }
    }

    /// <summary>
    /// Property assignment node (obj.prop = value)
    /// </summary>
    public class NSLPropertyAssignmentNode : NSLASTNode
    {
        /// <summary>Gets the object being assigned to.</summary>
        public NSLASTNode Object { get; }
        /// <summary>Gets the property name.</summary>
        public string Property { get; }
        /// <summary>Gets the value being assigned.</summary>
        public NSLASTNode Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLPropertyAssignmentNode(NSLASTNode obj, string property, NSLASTNode value)
        {
            Object = obj ?? throw new ArgumentNullException(nameof(obj));
            Property = property ?? throw new ArgumentNullException(nameof(property));
            Value = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitPropertyAssignment(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"PropertyAssignment({Object}.{Property} = {Value})";
        }
    }

    /// <summary>
    /// Chain operation node (a :: b :: c)
    /// </summary>
    public class NSLChainNode : NSLASTNode
    {
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLASTNode> Expressions { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLChainNode(IEnumerable<NSLASTNode> expressions)
        {
            Expressions = expressions?.ToList() ?? throw new ArgumentNullException(nameof(expressions));
            
            if (Expressions.Count == 0)
                throw new ArgumentException("Chain must contain at least one expression", nameof(expressions));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitChain(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Chain({string.Join(" :: ", Expressions)})";
        }
    }

    /// <summary>
    /// Function call node (func(args))
    /// </summary>
    public class NSLFunctionCallNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Function { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLASTNode> Arguments { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLFunctionCallNode(NSLASTNode function, IEnumerable<NSLASTNode> arguments)
        {
            Function = function ?? throw new ArgumentNullException(nameof(function));
            Arguments = arguments?.ToList() ?? new List<NSLASTNode>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitFunctionCall(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"FunctionCall({Function}({string.Join(", ", Arguments)}))";
        }
    }

    /// <summary>
    /// Array/list literal node ([1, 2, 3])
    /// </summary>
    public class NSLArrayNode : NSLASTNode
    {
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLASTNode> Elements { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLArrayNode(IEnumerable<NSLASTNode> elements)
        {
            Elements = elements?.ToList() ?? new List<NSLASTNode>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitArray(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Array([{string.Join(", ", Elements)}])";
        }
    }

    /// <summary>
    /// List comprehension node: [expr for var in iterable] or [expr for var in iterable if condition]
    /// Example: [x * x for x in range(1, 10)]
    /// Example: [x for x in nums if x > 0]
    /// </summary>
    public class NSLListComprehensionNode : NSLASTNode
    {
        /// <summary>The expression to evaluate for each element</summary>
        public NSLASTNode Expression { get; }

        /// <summary>The loop variable name</summary>
        public string Variable { get; }

        /// <summary>The iterable to loop over</summary>
        public NSLASTNode Iterable { get; }

        /// <summary>Optional filter condition (if clause)</summary>
        public NSLASTNode? Condition { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLListComprehensionNode(NSLASTNode expression, string variable, NSLASTNode iterable, NSLASTNode? condition = null)
        {
            Expression = expression ?? throw new ArgumentNullException(nameof(expression));
            Variable = variable ?? throw new ArgumentNullException(nameof(variable));
            Iterable = iterable ?? throw new ArgumentNullException(nameof(iterable));
            Condition = condition;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitListComprehension(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            var condStr = Condition != null ? $" if {Condition}" : "";
            return $"ListComprehension([{Expression} for {Variable} in {Iterable}{condStr}])";
        }
    }

    #endregion

    #region Consciousness and Quantum Nodes

    /// <summary>
    /// Consciousness operation node - supports both unary (◈x) and binary (x ~> y) forms
    /// Unary: ◈, ∇, ⊗
    /// Binary: |> (pipe), ~> (awareness), *> (attention), +> (superposition), =>> (gradient)
    /// </summary>
    public class NSLConsciousnessNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLTokenType Operator { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Operand { get; }

        // Binary consciousness operators (x ~> y, x *> y, etc.)
        /// <summary>Public API</summary>
        public NSLASTNode? Left { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? Right { get; }
        /// <summary>See implementation for details.</summary>
        public string? OperatorName { get; }
        /// <summary>Gets the flag value.</summary>
        public bool IsBinary => Left != null && Right != null;

        // Unary constructor (original)
        /// <summary>Public API</summary>
        public NSLConsciousnessNode(NSLTokenType @operator, NSLASTNode operand)
        {
            Operator = @operator;
            Operand = operand ?? throw new ArgumentNullException(nameof(operand));
        }

        // Binary constructor for ~>, *>, +>, =>>
        /// <summary>Public API</summary>
        public NSLConsciousnessNode(NSLASTNode left, NSLASTNode right, string operatorName)
        {
            Left = left ?? throw new ArgumentNullException(nameof(left));
            Right = right ?? throw new ArgumentNullException(nameof(right));
            OperatorName = operatorName;
            Operand = right; // For compatibility
            Operator = operatorName switch
            {
                "awareness" => NSLTokenType.AwarenessArrow,
                "attention" => NSLTokenType.AttentionArrow,
                "superposition" => NSLTokenType.SuperpositionArrow,
                "gradient" => NSLTokenType.GradientArrow,
                _ => NSLTokenType.PipeArrow
            };
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitConsciousness(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (IsBinary)
            {
                var opSymbol = OperatorName switch
                {
                    "awareness" => "~>",
                    "attention" => "*>",
                    "superposition" => "+>",
                    "gradient" => "=>>",
                    _ => "~>"
                };
                return $"Consciousness({Left} {opSymbol} {Right})";
            }

            var unarySymbol = Operator switch
            {
                NSLTokenType.Holographic => "◈",
                NSLTokenType.Gradient => "∇",
                NSLTokenType.TensorProduct => "⊗",
                _ => Operator.ToString()
            };
            return $"Consciousness({unarySymbol}[{Operand}])";
        }
    }

    /// <summary>
    /// Quantum superposition node (Ψ[state1|state2|state3])
    /// </summary>
    public class NSLQuantumNode : NSLASTNode
    {
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLASTNode> States { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLQuantumNode(IEnumerable<NSLASTNode> states)
        {
            States = states?.ToList() ?? throw new ArgumentNullException(nameof(states));
            
            if (States.Count == 0)
                throw new ArgumentException("Quantum superposition must contain at least one state", nameof(states));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitQuantum(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Quantum(Ψ[{string.Join("|", States)}])";
        }
    }

    /// <summary>
    /// Lambda expression node (λ[x, y] → x + y)
    /// </summary>
    public class NSLLambdaNode : NSLASTNode
    {
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLParameter> Parameters { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Body { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLLambdaNode(IEnumerable<NSLParameter> parameters, NSLASTNode body)
        {
            Parameters = parameters?.ToList() ?? new List<NSLParameter>();
            Body = body ?? throw new ArgumentNullException(nameof(body));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitLambda(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Lambda(λ[{string.Join(", ", Parameters.Select(p => p.Name))}] → {Body})";
        }
    }

    /// <summary>
    /// Property access expression (dot notation)
    /// </summary>
    public class NSLGetNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Object { get; }
        /// <summary>Gets the name.</summary>
        public string Name { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLGetNode(NSLASTNode obj, string name)
        {
            Object = obj ?? throw new ArgumentNullException(nameof(obj));
            Name = name ?? throw new ArgumentNullException(nameof(name));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitGet(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Get({Object}.{Name})";
        }
    }

    /// <summary>
    /// Array index expression
    /// </summary>
    public class NSLIndexNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Object { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Index { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLIndexNode(NSLASTNode obj, NSLASTNode index)
        {
            Object = obj ?? throw new ArgumentNullException(nameof(obj));
            Index = index ?? throw new ArgumentNullException(nameof(index));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitIndex(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Index({Object}[{Index}])";
        }
    }

    #endregion

    #region Statement Nodes

    /// <summary>
    /// Block statement node ({ statements })
    /// </summary>
    public class NSLBlockNode : NSLASTNode
    {
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLASTNode> Statements { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLBlockNode(IEnumerable<NSLASTNode> statements)
        {
            Statements = statements?.ToList() ?? new List<NSLASTNode>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitBlock(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Block({Statements.Count} statements)";
        }
    }

    /// <summary>
    /// If statement node (if condition then_branch else_branch?)
    /// </summary>
    public class NSLIfNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Condition { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode ThenBranch { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? ElseBranch { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLIfNode(NSLASTNode condition, NSLASTNode thenBranch, NSLASTNode? elseBranch = null)
        {
            Condition = condition ?? throw new ArgumentNullException(nameof(condition));
            ThenBranch = thenBranch ?? throw new ArgumentNullException(nameof(thenBranch));
            ElseBranch = elseBranch;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitIf(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"If({Condition} then {ThenBranch}{(ElseBranch != null ? $" else {ElseBranch}" : "")})";
        }
    }

    /// <summary>
    /// While loop node (while condition body)
    /// </summary>
    public class NSLWhileNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Condition { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Body { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLWhileNode(NSLASTNode condition, NSLASTNode body)
        {
            Condition = condition ?? throw new ArgumentNullException(nameof(condition));
            Body = body ?? throw new ArgumentNullException(nameof(body));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitWhile(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"While({Condition} do {Body})";
        }
    }

    /// <summary>
    /// Python-style for loop node (for variable in iterable)
    /// </summary>
    public class NSLForNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLToken Variable { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Iterable { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Body { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLForNode(NSLToken variable, NSLASTNode iterable, NSLASTNode body)
        {
            Variable = variable ?? throw new ArgumentNullException(nameof(variable));
            Iterable = iterable ?? throw new ArgumentNullException(nameof(iterable));
            Body = body ?? throw new ArgumentNullException(nameof(body));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitFor(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"For({Variable.Value} in {Iterable}) {Body}";
        }
    }

    /// <summary>
    /// Break statement node
    /// </summary>
    public class NSLBreakNode : NSLASTNode
    {
        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitBreak(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return "Break";
        }
    }

    /// <summary>
    /// Continue statement node
    /// </summary>
    public class NSLContinueNode : NSLASTNode
    {
        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitContinue(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return "Continue";
        }
    }

    /// <summary>
    /// Return statement node (return value?)
    /// </summary>
    public class NSLReturnNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLReturnNode(NSLASTNode? value = null)
        {
            Value = value;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitReturn(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Return({Value?.ToString() ?? "void"})";
        }
    }



    /// <summary>
    /// Function definition node (func name(params) body)
    /// </summary>
    public class NSLFunctionNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLParameter> Parameters { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Body { get; }
        /// <summary>See implementation for details.</summary>
        public string? ReturnType { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLFunctionNode(string name, IEnumerable<NSLParameter> parameters, NSLASTNode body, string? returnType = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Parameters = parameters?.ToList() ?? new List<NSLParameter>();
            Body = body ?? throw new ArgumentNullException(nameof(body));
            ReturnType = returnType;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitFunction(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            var returnStr = ReturnType != null ? $" -> {ReturnType}" : "";
            return $"Function({Name}({string.Join(", ", Parameters.Select(p => p.Name))}){returnStr} {Body})";
        }
    }

    /// <summary>
    /// Class definition node (class name body)
    /// </summary>
    public class NSLClassNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Body { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLClassNode(string name, NSLASTNode body)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Body = body ?? throw new ArgumentNullException(nameof(body));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitClass(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Class({Name} {Body})";
        }
    }

    /// <summary>
    /// Match expression node (match value cases)
    /// </summary>
    public class NSLMatchNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Value { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLMatchCase> Cases { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLMatchNode(NSLASTNode value, IEnumerable<NSLMatchCase> cases)
        {
            Value = value ?? throw new ArgumentNullException(nameof(value));
            Cases = cases?.ToList() ?? throw new ArgumentNullException(nameof(cases));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitMatch(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Match({Value} with {Cases.Count} cases)";
        }
    }

    #endregion

    #region AI-Friendly AST Nodes

    /// <summary>
    /// Variable declaration with optional type hint and mutability
    /// AI Error Prevention: Explicit types catch mistakes early
    /// Example: let x: number = 10, mut y: string = "hello"
    /// </summary>
    public class NSLVariableDeclarationNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>See implementation for details.</summary>
        public string? TypeHint { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? Value { get; }
        /// <summary>Gets the flag value.</summary>
        public bool IsMutable { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLVariableDeclarationNode(string name, NSLASTNode? value, string? typeHint = null, bool isMutable = false)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Value = value;
            TypeHint = typeHint;
            IsMutable = isMutable;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitVariableDeclaration(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            var mutStr = IsMutable ? "mut " : "let ";
            var typeStr = TypeHint != null ? $": {TypeHint}" : "";
            return $"VarDecl({mutStr}{Name}{typeStr} = {Value})";
        }
    }

    /// <summary>
    /// Safe navigation operator (?.)
    /// AI Error Prevention: Avoids null reference errors - a common AI mistake
    /// Example: obj?.property, a?.b?.c
    /// </summary>
    public class NSLSafeNavigationNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Object { get; }
        /// <summary>Gets the property name.</summary>
        public string Property { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLSafeNavigationNode(NSLASTNode obj, string property)
        {
            Object = obj ?? throw new ArgumentNullException(nameof(obj));
            Property = property ?? throw new ArgumentNullException(nameof(property));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitSafeNavigation(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"SafeNav({Object}?.{Property})";
        }
    }

    /// <summary>
    /// Pipeline operator (|>)
    /// AI Natural Flow: Matches how AI thinks about data pipelines
    /// Example: data |> normalize |> encode |> predict
    /// </summary>
    public class NSLPipelineNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Left { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Right { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLPipelineNode(NSLASTNode left, NSLASTNode right)
        {
            Left = left ?? throw new ArgumentNullException(nameof(left));
            Right = right ?? throw new ArgumentNullException(nameof(right));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitPipeline(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Pipeline({Left} |> {Right})";
        }
    }

    /// <summary>
    /// Range expression (.., ..=)
    /// AI Error Prevention: Eliminates off-by-one errors
    /// Example: 0..10 (exclusive), 0..=10 (inclusive)
    /// </summary>
    public class NSLRangeNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? Start { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? End { get; }
        /// <summary>Gets the flag value.</summary>
        public bool IsInclusive { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLRangeNode(NSLASTNode? start, NSLASTNode? end, bool isInclusive)
        {
            Start = start;
            End = end;
            IsInclusive = isInclusive;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitRange(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            var op = IsInclusive ? "..=" : "..";
            return $"Range({Start}{op}{End})";
        }
    }

    /// <summary>
    /// Type cast expression (as)
    /// Example: value as number, data as vec
    /// </summary>
    public class NSLCastNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Value { get; }
        /// <summary>Gets the target type.</summary>
        public string TargetType { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLCastNode(NSLASTNode value, string targetType)
        {
            Value = value ?? throw new ArgumentNullException(nameof(value));
            TargetType = targetType ?? throw new ArgumentNullException(nameof(targetType));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitCast(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Cast({Value} as {TargetType})";
        }
    }

    /// <summary>
    /// Result type wrapper (ok/err)
    /// AI Error Prevention: Explicit error handling, no exceptions
    /// Example: ok(42), err("not found")
    /// </summary>
    public class NSLResultNode : NSLASTNode
    {
        /// <summary>Gets the flag value.</summary>
        public bool IsOk { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLResultNode(bool isOk, NSLASTNode value)
        {
            IsOk = isOk;
            Value = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitResult(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return IsOk ? $"Ok({Value})" : $"Err({Value})";
        }
    }

    /// <summary>
    /// Optional type wrapper (some/none)
    /// AI Error Prevention: Explicit handling of absent values
    /// Example: some(value), none
    /// </summary>
    public class NSLOptionalNode : NSLASTNode
    {
        /// <summary>Gets the boolean flag.</summary>
        public bool HasValue { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLOptionalNode(bool hasValue, NSLASTNode? value = null)
        {
            HasValue = hasValue;
            Value = value;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitOptional(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return HasValue ? $"Some({Value})" : "None";
        }
    }

    /// <summary>
    /// Type alias declaration
    /// Example: type Point = {x: number, y: number}
    /// </summary>
    public class NSLTypeAliasNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Definition { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLTypeAliasNode(string name, NSLASTNode definition)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Definition = definition ?? throw new ArgumentNullException(nameof(definition));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitTypeAlias(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"TypeAlias({Name} = {Definition})";
        }
    }

    /// <summary>
    /// Object literal / record type definition
    /// Example: {x: number, y: number} or {name: "Claude", age: 1}
    /// </summary>
    public class NSLObjectNode : NSLASTNode
    {
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLObjectField> Fields { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLObjectNode(IEnumerable<NSLObjectField> fields)
        {
            Fields = fields?.ToList() ?? new List<NSLObjectField>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitObject(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Object({{{string.Join(", ", Fields)}}})";
        }
    }

    /// <summary>
    /// Object field (key-value pair)
    /// </summary>
    public class NSLObjectField
    {
        /// <summary>Gets the key.</summary>
        public string Key { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLObjectField(string key, NSLASTNode value)
        {
            Key = key ?? throw new ArgumentNullException(nameof(key));
            Value = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{Key}: {Value}";
        }
    }

    /// <summary>
    /// Struct definition node
    /// Example: struct Point { x: number, y: number }
    /// </summary>
    public class NSLStructNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLStructField> Fields { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLStructNode(string name, IEnumerable<NSLStructField> fields)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Fields = fields?.ToList() ?? new List<NSLStructField>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitStruct(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Struct({Name} {{ {string.Join(", ", Fields)} }})";
        }
    }

    /// <summary>
    /// Struct field definition with name and type
    /// </summary>
    public class NSLStructField
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Gets the type name.</summary>
        public string TypeName { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLStructField(string name, string typeName)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            TypeName = typeName ?? throw new ArgumentNullException(nameof(typeName));
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{Name}: {TypeName}";
        }
    }

    /// <summary>
    /// Struct instantiation node
    /// Example: Point { x: 10.0, y: 20.0 }
    /// </summary>
    public class NSLStructInstantiationNode : NSLASTNode
    {
        /// <summary>Gets the string value.</summary>
        public string StructName { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLObjectField> Fields { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLStructInstantiationNode(string structName, IEnumerable<NSLObjectField> fields)
        {
            StructName = structName ?? throw new ArgumentNullException(nameof(structName));
            Fields = fields?.ToList() ?? new List<NSLObjectField>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitStructInstantiation(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"StructInst({StructName} {{ {string.Join(", ", Fields)} }})";
        }
    }

    /// <summary>
    /// Enum (algebraic data type) definition node
    /// Example: enum Color { Red, Green, Blue }
    /// Example: enum Shape { Circle(number), Rectangle(number, number) }
    /// </summary>
    public class NSLEnumNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLEnumVariant> Variants { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLEnumNode(string name, IEnumerable<NSLEnumVariant> variants)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Variants = variants?.ToList() ?? new List<NSLEnumVariant>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitEnum(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Enum({Name} {{ {string.Join(", ", Variants)} }})";
        }
    }

    /// <summary>
    /// Enum variant definition
    /// Example: Red (unit variant)
    /// Example: Circle(number) (tuple variant with one field)
    /// Example: Point(x: number, y: number) (struct variant)
    /// </summary>
    public class NSLEnumVariant
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Field types for tuple variants, null for unit variants</summary>
        public IReadOnlyList<string>? Fields { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLEnumVariant(string name, IEnumerable<string>? fields = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Fields = fields?.ToList();
        }

        /// <summary>Gets the flag value.</summary>
        public bool IsUnit => Fields == null || Fields.Count == 0;

        /// <inheritdoc/>
        public override string ToString()
        {
            if (IsUnit)
                return Name;
            return $"{Name}({string.Join(", ", Fields!)})";
        }
    }

    /// <summary>
    /// Enum variant instantiation node
    /// Example: Color::Red or Shape::Circle(5.0)
    /// </summary>
    public class NSLEnumVariantNode : NSLASTNode
    {
        /// <summary>Gets the string value.</summary>
        public string EnumName { get; }
        /// <summary>Gets the string value.</summary>
        public string VariantName { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLASTNode> Arguments { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLEnumVariantNode(string enumName, string variantName, IEnumerable<NSLASTNode>? arguments = null)
        {
            EnumName = enumName ?? throw new ArgumentNullException(nameof(enumName));
            VariantName = variantName ?? throw new ArgumentNullException(nameof(variantName));
            Arguments = arguments?.ToList() ?? new List<NSLASTNode>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitEnumVariant(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (Arguments.Count == 0)
                return $"{EnumName}::{VariantName}";
            return $"{EnumName}::{VariantName}({string.Join(", ", Arguments)})";
        }
    }

    #endregion

    #region Async/Await AST Nodes

    /// <summary>
    /// Async function node - an async function returns a Future/Promise
    /// Example: async fn fetch_data() { ... }
    /// </summary>
    public class NSLAsyncFunctionNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLParameter> Parameters { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Body { get; }
        /// <summary>See implementation for details.</summary>
        public string? ReturnType { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLAsyncFunctionNode(string name, IEnumerable<NSLParameter> parameters, NSLASTNode body, string? returnType = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Parameters = parameters?.ToList() ?? new List<NSLParameter>();
            Body = body ?? throw new ArgumentNullException(nameof(body));
            ReturnType = returnType;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitAsyncFunction(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            var paramsStr = string.Join(", ", Parameters);
            return $"AsyncFn {Name}({paramsStr}) {{ ... }}";
        }
    }

    /// <summary>
    /// Await expression node - waits for an async operation to complete
    /// Example: let result = await fetch_data()
    /// </summary>
    public class NSLAwaitNode : NSLASTNode
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Expression { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLAwaitNode(NSLASTNode expression)
        {
            Expression = expression ?? throw new ArgumentNullException(nameof(expression));
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitAwait(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Await({Expression})";
        }
    }

    #endregion

    #region Trait/Interface AST Nodes

    /// <summary>
    /// Trait method signature (not a full function, just the signature)
    /// </summary>
    public class NSLTraitMethod
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLParameter> Parameters { get; }
        /// <summary>See implementation for details.</summary>
        public string? ReturnType { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLTraitMethod(string name, IEnumerable<NSLParameter> parameters, string? returnType = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Parameters = parameters?.ToList() ?? new List<NSLParameter>();
            ReturnType = returnType;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            var paramsStr = string.Join(", ", Parameters);
            var returnStr = ReturnType != null ? $" -> {ReturnType}" : "";
            return $"fn {Name}({paramsStr}){returnStr}";
        }
    }

    /// <summary>
    /// Trait definition node
    /// Example: trait Printable { fn print(self); fn format(self) -> string }
    /// </summary>
    public class NSLTraitNode : NSLASTNode
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLTraitMethod> Methods { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLTraitNode(string name, IEnumerable<NSLTraitMethod> methods)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Methods = methods?.ToList() ?? new List<NSLTraitMethod>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitTrait(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Trait({Name} {{ {string.Join("; ", Methods)} }})";
        }
    }

    /// <summary>
    /// Trait implementation node
    /// Example: impl Printable for Person { fn print(self) { ... } }
    /// </summary>
    public class NSLImplNode : NSLASTNode
    {
        /// <summary>Gets the string value.</summary>
        public string TraitName { get; }
        /// <summary>Gets the type name.</summary>
        public string TypeName { get; }
        /// <summary>Gets the list of items.</summary>
        public IReadOnlyList<NSLFunctionNode> Methods { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLImplNode(string traitName, string typeName, IEnumerable<NSLFunctionNode> methods)
        {
            TraitName = traitName ?? throw new ArgumentNullException(nameof(traitName));
            TypeName = typeName ?? throw new ArgumentNullException(nameof(typeName));
            Methods = methods?.ToList() ?? new List<NSLFunctionNode>();
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitImpl(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Impl({TraitName} for {TypeName} {{ {Methods.Count} methods }})";
        }
    }

    #endregion

    #region Module System AST Nodes

    /// <summary>
    /// Module declaration node
    /// Declares the module name and its public interface
    /// Example: module math::linear_algebra
    /// </summary>
    public class NSLModuleNode : NSLASTNode
    {
        /// <summary>Full module path (e.g., ["math", "linear_algebra"])</summary>
        public IReadOnlyList<string> Path { get; }

        /// <summary>Module body containing all declarations</summary>
        public NSLBlockNode Body { get; }

        /// <summary>Module-level documentation</summary>
        public string? Documentation { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLModuleNode(IEnumerable<string> path, NSLBlockNode body, string? documentation = null)
        {
            Path = path?.ToList() ?? throw new ArgumentNullException(nameof(path));
            Body = body ?? throw new ArgumentNullException(nameof(body));
            Documentation = documentation;
        }

        /// <summary>Gets the full path.</summary>
        public string FullPath => string.Join("::", Path);

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitModule(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"Module({FullPath})";
        }
    }

    /// <summary>
    /// Import statement node
    /// Supports multiple import styles:
    /// - import math                           (import entire module)
    /// - import math::linear_algebra           (import submodule)
    /// - import { sin, cos } from math         (selective import)
    /// - import { sin as sine } from math      (aliased import)
    /// - import * from math                    (wildcard import)
    /// - import "path/to/file.nsl"             (file import)
    /// </summary>
    public class NSLImportNode : NSLASTNode
    {
        /// <summary>Module path to import from (e.g., ["math", "linear_algebra"])</summary>
        public IReadOnlyList<string> ModulePath { get; }

        /// <summary>Specific items to import (null = import whole module)</summary>
        public IReadOnlyList<NSLImportItem>? Items { get; }

        /// <summary>Whether this is a wildcard import (*)</summary>
        public bool IsWildcard { get; }

        /// <summary>Alias for the entire module (import math as m)</summary>
        public string? ModuleAlias { get; }

        /// <summary>If importing from a file path instead of module name</summary>
        public string? FilePath { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLImportNode(
            IEnumerable<string>? modulePath,
            IEnumerable<NSLImportItem>? items = null,
            bool isWildcard = false,
            string? moduleAlias = null,
            string? filePath = null)
        {
            ModulePath = modulePath?.ToList() ?? new List<string>();
            Items = items?.ToList();
            IsWildcard = isWildcard;
            ModuleAlias = moduleAlias;
            FilePath = filePath;
        }

        /// <summary>Gets the full module path.</summary>
        public string FullModulePath => string.Join("::", ModulePath);

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitImport(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (FilePath != null)
                return $"Import(\"{FilePath}\")";
            if (IsWildcard)
                return $"Import(* from {FullModulePath})";
            if (Items != null)
                return $"Import({{{string.Join(", ", Items)}}} from {FullModulePath})";
            if (ModuleAlias != null)
                return $"Import({FullModulePath} as {ModuleAlias})";
            return $"Import({FullModulePath})";
        }
    }

    /// <summary>
    /// Individual import item with optional alias
    /// Example: sin, cos as cosine, Matrix as Mat
    /// </summary>
    public class NSLImportItem
    {
        /// <summary>Original name in the source module</summary>
        public string Name { get; }

        /// <summary>Local alias (null = use original name)</summary>
        public string? Alias { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLImportItem(string name, string? alias = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Alias = alias;
        }

        /// <summary>The name to use locally (alias if present, otherwise original name)</summary>
        public string LocalName => Alias ?? Name;

        /// <inheritdoc/>
        public override string ToString()
        {
            return Alias != null ? $"{Name} as {Alias}" : Name;
        }
    }

    /// <summary>
    /// Export statement node
    /// Makes items available to other modules
    /// - pub fn foo()                     (public function)
    /// - pub let X = 10                   (public constant)
    /// - export { foo, bar }              (explicit export list)
    /// - export { foo as publicFoo }      (aliased export)
    /// - export * from "submodule"        (re-export)
    /// </summary>
    public class NSLExportNode : NSLASTNode
    {
        /// <summary>Items to export (null = attached to declaration with pub)</summary>
        public IReadOnlyList<NSLExportItem>? Items { get; }

        /// <summary>Declaration being exported (for pub fn, pub let, etc.)</summary>
        public NSLASTNode? Declaration { get; }

        /// <summary>Whether this is a re-export from another module</summary>
        public bool IsReExport { get; }

        /// <summary>Module to re-export from</summary>
        public IReadOnlyList<string>? ReExportFrom { get; }

        /// <summary>Whether to re-export everything (*)</summary>
        public bool IsWildcardReExport { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLExportNode(
            IEnumerable<NSLExportItem>? items = null,
            NSLASTNode? declaration = null,
            bool isReExport = false,
            IEnumerable<string>? reExportFrom = null,
            bool isWildcardReExport = false)
        {
            Items = items?.ToList();
            Declaration = declaration;
            IsReExport = isReExport;
            ReExportFrom = reExportFrom?.ToList();
            IsWildcardReExport = isWildcardReExport;
        }

        /// <inheritdoc/>
        public override T Accept<T>(INSLVisitor<T> visitor)
        {
            return visitor.VisitExport(this);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (Declaration != null)
                return $"Export(pub {Declaration})";
            if (IsWildcardReExport && ReExportFrom != null)
                return $"Export(* from {string.Join("::", ReExportFrom)})";
            if (Items != null)
                return $"Export({{{string.Join(", ", Items)}}})";
            return "Export()";
        }
    }

    /// <summary>
    /// Individual export item with optional alias
    /// </summary>
    public class NSLExportItem
    {
        /// <summary>Internal name</summary>
        public string Name { get; }

        /// <summary>Public name (null = use internal name)</summary>
        public string? Alias { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLExportItem(string name, string? alias = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Alias = alias;
        }

        /// <summary>The name exposed to importers</summary>
        public string PublicName => Alias ?? Name;

        /// <inheritdoc/>
        public override string ToString()
        {
            return Alias != null ? $"{Name} as {Alias}" : Name;
        }
    }

    #endregion

    #region Supporting Classes

    /// <summary>
    /// Function parameter representation
    /// </summary>
    public class NSLParameter
    {
        /// <summary>Gets the name.</summary>
        public string Name { get; }
        /// <summary>See implementation for details.</summary>
        public string? Type { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLParameter(string name, string? type = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Type = type;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return Type != null ? $"{Name}: {Type}" : Name;
        }
    }

    /// <summary>
    /// Match case representation
    /// </summary>
    public class NSLMatchCase
    {
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Pattern { get; }
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode? Guard { get; }  // Optional guard condition (when clause)
        /// <summary>Creates a new instance.</summary>
        public NSLASTNode Body { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLMatchCase(NSLASTNode pattern, NSLASTNode body, NSLASTNode? guard = null)
        {
            Pattern = pattern ?? throw new ArgumentNullException(nameof(pattern));
            Body = body ?? throw new ArgumentNullException(nameof(body));
            Guard = guard;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return Guard != null ? $"{Pattern} when {Guard} => {Body}" : $"{Pattern} => {Body}";
        }
    }

    #endregion
}