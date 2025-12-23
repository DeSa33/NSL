using System;
using System.Collections.Generic;
using System.Linq;
using NSL.Core;
using NSL.Core.AST;
using NSLTokenType = NSL.Core.Tokens.TokenType;

namespace NSL.TypeSystem;

/// <summary>
/// Type checker for NSL - performs semantic analysis and type inference
/// </summary>
public class TypeChecker
{
    private readonly Dictionary<string, NSLType> _variables = new();
    private readonly Dictionary<string, NSLFunctionType> _functions = new();
    private readonly Dictionary<string, NSLType> _typeAliases = new();
    private readonly Dictionary<string, NSLStructType> _structs = new();
    private readonly Dictionary<string, NSLEnumType> _enums = new();
    private readonly Dictionary<string, NSLTraitType> _traits = new();
    // Tracks which types implement which traits: (TraitName, TypeName) -> ImplementedMethods
    private readonly Dictionary<(string TraitName, string TypeName), HashSet<string>> _traitImplementations = new();
    private readonly List<TypeCheckError> _errors = new();
    private NSLType? _currentFunctionReturnType;
    private ModuleResolver? _moduleResolver;

    public IReadOnlyList<TypeCheckError> Errors => _errors;
    public bool HasErrors => _errors.Count > 0;

    /// <summary>
    /// Set the module resolver for loading imported modules
    /// </summary>
    public void SetModuleResolver(ModuleResolver resolver)
    {
        _moduleResolver = resolver;
    }

    /// <summary>
    /// Type check an AST node and return the resulting type
    /// </summary>
    public NSLType Check(NSLASTNode node)
    {
        return node switch
        {
            NSLLiteralNode literal => CheckLiteral(literal),
            NSLIdentifierNode ident => CheckIdentifier(ident),
            NSLBinaryOperationNode binary => CheckBinaryOperation(binary),
            NSLUnaryOperationNode unary => CheckUnaryOperation(unary),
            NSLVariableDeclarationNode varDecl => CheckVariableDeclaration(varDecl),
            NSLAssignmentNode assign => CheckAssignment(assign),
            NSLFunctionNode func => CheckFunction(func),
            NSLFunctionCallNode call => CheckFunctionCall(call),
            NSLReturnNode ret => CheckReturn(ret),
            NSLIfNode ifNode => CheckIf(ifNode),
            NSLWhileNode whileNode => CheckWhile(whileNode),
            NSLForNode forNode => CheckFor(forNode),
            NSLBlockNode block => CheckBlock(block),
            NSLArrayNode array => CheckArray(array),
            NSLListComprehensionNode listComp => CheckListComprehension(listComp),
            NSLIndexNode index => CheckIndex(index),
            NSLGetNode get => CheckGet(get),
            NSLPipelineNode pipeline => CheckPipeline(pipeline),
            NSLRangeNode range => CheckRange(range),
            NSLSafeNavigationNode safeNav => CheckSafeNavigation(safeNav),
            NSLMatchNode match => CheckMatch(match),
            NSLResultNode result => CheckResult(result),
            NSLOptionalNode optional => CheckOptional(optional),
            NSLCastNode cast => CheckCast(cast),
            NSLLambdaNode lambda => CheckLambda(lambda),
            NSLTypeAliasNode typeAlias => CheckTypeAlias(typeAlias),
            NSLObjectNode obj => CheckObject(obj),
            NSLStructNode structDef => CheckStructDefinition(structDef),
            NSLStructInstantiationNode structInst => CheckStructInstantiation(structInst),
            NSLEnumNode enumDef => CheckEnumDefinition(enumDef),
            NSLEnumVariantNode enumVariant => CheckEnumVariant(enumVariant),
            NSLTraitNode traitDef => CheckTraitDefinition(traitDef),
            NSLImplNode implDef => CheckImplDefinition(implDef),
            NSLAsyncFunctionNode asyncFunc => CheckAsyncFunction(asyncFunc),
            NSLAwaitNode awaitNode => CheckAwait(awaitNode),
            NSLBreakNode => NSLTypes.Void,
            NSLContinueNode => NSLTypes.Void,

            // Module system nodes
            NSLImportNode import => CheckImport(import),
            NSLExportNode export => CheckExport(export),
            NSLModuleNode module => CheckModule(module),

            _ => ReportError(node, $"Unknown node type: {node.GetType().Name}")
        };
    }

    private NSLType CheckLiteral(NSLLiteralNode node)
    {
        return node.ValueType switch
        {
            NSLTokenType.Number => NSLTypes.Number,
            NSLTokenType.Integer => NSLTypes.Int,
            NSLTokenType.String => NSLTypes.String,
            NSLTokenType.Boolean => NSLTypes.Bool,
            NSLTokenType.Null => NSLTypes.Null,
            _ => ReportError(node, $"Unknown literal type: {node.ValueType}")
        };
    }

    private NSLType CheckIdentifier(NSLIdentifierNode node)
    {
        if (_variables.TryGetValue(node.Name, out var type))
            return type;
        if (_functions.TryGetValue(node.Name, out var funcType))
            return funcType;
        return ReportError(node, $"Undefined variable: {node.Name}");
    }

    private NSLType CheckBinaryOperation(NSLBinaryOperationNode node)
    {
        var leftType = Check(node.Left);
        var rightType = Check(node.Right);

        return node.Operator switch
        {
            // Arithmetic operators
            NSLTokenType.Plus or NSLTokenType.Minus or
            NSLTokenType.Multiply or NSLTokenType.Divide or
            NSLTokenType.Power or NSLTokenType.Percent =>
                CheckArithmetic(node, leftType, rightType),

            // Matrix multiply
            NSLTokenType.AtSign => CheckMatrixMultiply(node, leftType, rightType),

            // Comparison operators
            NSLTokenType.Equal or NSLTokenType.NotEqual or
            NSLTokenType.Less or NSLTokenType.LessEqual or
            NSLTokenType.Greater or NSLTokenType.GreaterEqual =>
                CheckComparison(node, leftType, rightType),

            // Logical operators
            NSLTokenType.And or NSLTokenType.Or =>
                CheckLogical(node, leftType, rightType),

            // Bitwise operators - require integers, return integer
            NSLTokenType.BitwiseAnd or NSLTokenType.BitwiseOr or
            NSLTokenType.BitwiseXor or NSLTokenType.LeftShift or
            NSLTokenType.RightShift or NSLTokenType.IntegerDivide =>
                CheckBitwise(node, leftType, rightType),

            // Null coalescing
            NSLTokenType.QuestionQuestion =>
                CheckNullCoalescing(node, leftType, rightType),

            // Tensor product
            NSLTokenType.TensorProduct =>
                CheckTensorProduct(node, leftType, rightType),

            _ => ReportError(node, $"Unknown operator: {node.Operator}")
        };
    }

    private NSLType CheckArithmetic(NSLBinaryOperationNode node, NSLType left, NSLType right)
    {
        // String concatenation
        if (node.Operator == NSLTokenType.Plus &&
            (left is NSLStringType || right is NSLStringType))
            return NSLTypes.String;

        // Allow Any types to pass through (might be numeric at runtime)
        // This enables recursive functions where return type is initially unknown
        bool leftOk = left.IsNumeric || left is NSLAnyType;
        bool rightOk = right.IsNumeric || right is NSLAnyType;

        if (!leftOk)
            return ReportError(node, $"Left operand must be numeric, got {left}");
        if (!rightOk)
            return ReportError(node, $"Right operand must be numeric, got {right}");

        // Vector/Matrix operations return same type
        if (left is NSLVecType || right is NSLVecType)
            return new NSLVecType();
        if (left is NSLMatType || right is NSLMatType)
            return new NSLMatType();
        if (left is NSLTensorType || right is NSLTensorType)
            return new NSLTensorType();

        // If either side is Any (recursive call), result is Any
        // until actual return type is determined
        if (left is NSLAnyType || right is NSLAnyType)
            return NSLTypes.Any;

        // Integer arithmetic: both operands are integers -> result is integer
        // (except for division and power which may produce non-integers)
        if (left is NSLIntType && right is NSLIntType &&
            node.Operator != NSLTokenType.Divide && node.Operator != NSLTokenType.Power)
            return NSLTypes.Int;

        return NSLTypes.Number;
    }

    private NSLType CheckMatrixMultiply(NSLBinaryOperationNode node, NSLType left, NSLType right)
    {
        // Matrix @ Vector -> Vector
        if (left is NSLMatType && right is NSLVecType)
            return new NSLVecType();

        // Matrix @ Matrix -> Matrix
        if (left is NSLMatType && right is NSLMatType)
            return new NSLMatType();

        // Vector @ Vector -> Number (dot product)
        if (left is NSLVecType && right is NSLVecType)
            return NSLTypes.Number;

        // Tensor operations
        if (left is NSLTensorType || right is NSLTensorType)
            return new NSLTensorType();

        return ReportError(node, $"Matrix multiply requires mat/vec/tensor types, got {left} @ {right}");
    }

    private NSLType CheckComparison(NSLBinaryOperationNode node, NSLType left, NSLType right)
    {
        // Allow comparison between same types or numeric types
        // Also allow Any types to participate in comparisons (recursive calls)
        bool leftNumericOrAny = left.IsNumeric || left is NSLAnyType;
        bool rightNumericOrAny = right.IsNumeric || right is NSLAnyType;

        if (left.Equals(right) || (leftNumericOrAny && rightNumericOrAny))
            return NSLTypes.Bool;

        // Allow Any to be compared with anything
        if (left is NSLAnyType || right is NSLAnyType)
            return NSLTypes.Bool;

        return ReportError(node, $"Cannot compare {left} with {right}");
    }

    private NSLType CheckLogical(NSLBinaryOperationNode node, NSLType left, NSLType right)
    {
        if (left is not NSLBoolType)
            return ReportError(node, $"Left operand must be bool, got {left}");
        if (right is not NSLBoolType)
            return ReportError(node, $"Right operand must be bool, got {right}");
        return NSLTypes.Bool;
    }

    /// <summary>
    /// Check bitwise operations - these work on integers and return integers
    /// </summary>
    private NSLType CheckBitwise(NSLBinaryOperationNode node, NSLType left, NSLType right)
    {
        // Bitwise operations work on integers (int) or numeric types that can be converted
        bool leftOk = left is NSLIntType || left is NSLNumberType || left is NSLAnyType;
        bool rightOk = right is NSLIntType || right is NSLNumberType || right is NSLAnyType;

        if (!leftOk)
            return ReportError(node, $"Left operand must be numeric for bitwise operation, got {left}");
        if (!rightOk)
            return ReportError(node, $"Right operand must be numeric for bitwise operation, got {right}");

        // Bitwise operations always return int
        return NSLTypes.Int;
    }

    private NSLType CheckNullCoalescing(NSLBinaryOperationNode node, NSLType left, NSLType right)
    {
        // T? ?? T -> T
        if (left is NSLOptionalType optional)
        {
            if (!right.IsAssignableTo(optional.InnerType))
                return ReportError(node, $"Fallback type {right} not assignable to {optional.InnerType}");
            return optional.InnerType;
        }
        return right; // If left is not optional, return right type
    }

    private NSLType CheckTensorProduct(NSLBinaryOperationNode node, NSLType left, NSLType right)
    {
        // Tensor product produces higher-dimensional tensor
        return new NSLTensorType();
    }

    private NSLType CheckUnaryOperation(NSLUnaryOperationNode node)
    {
        var operandType = Check(node.Operand);

        return node.Operator switch
        {
            NSLTokenType.Minus when operandType.IsNumeric => operandType,
            NSLTokenType.Not when operandType is NSLBoolType => NSLTypes.Bool,
            // Bitwise NOT returns int
            NSLTokenType.BitwiseNot when operandType is NSLIntType or NSLNumberType => NSLTypes.Int,
            NSLTokenType.Holographic or NSLTokenType.Gradient or NSLTokenType.Psi =>
                new NSLTensorType(), // Consciousness operators return tensor
            _ => ReportError(node, $"Invalid unary operator {node.Operator} for type {operandType}")
        };
    }

    private NSLType CheckVariableDeclaration(NSLVariableDeclarationNode node)
    {
        NSLType type;

        if (node.Value != null)
        {
            type = Check(node.Value);

            // If type hint provided, verify compatibility
            if (node.TypeHint != null)
            {
                var hintType = ResolveType(node.TypeHint);
                if (!type.IsAssignableTo(hintType))
                {
                    return ReportError(node,
                        $"Type mismatch: cannot assign {type} to {node.Name}: {hintType}");
                }
                type = hintType; // Use declared type
            }
        }
        else if (node.TypeHint != null)
        {
            type = ResolveType(node.TypeHint);
        }
        else
        {
            return ReportError(node, $"Variable {node.Name} needs either type hint or initializer");
        }

        _variables[node.Name] = type;
        return type;
    }

    private NSLType CheckAssignment(NSLAssignmentNode node)
    {
        if (!_variables.TryGetValue(node.Name, out var varType))
            return ReportError(node, $"Undefined variable: {node.Name}");

        var valueType = Check(node.Value);
        if (!valueType.IsAssignableTo(varType))
            return ReportError(node, $"Cannot assign {valueType} to {node.Name}: {varType}");

        return varType;
    }

    private NSLType CheckFunction(NSLFunctionNode node)
    {
        var paramTypes = new List<NSLType>();

        // Enter function scope
        var savedVariables = new Dictionary<string, NSLType>(_variables);

        foreach (var param in node.Parameters)
        {
            var paramType = param.Type != null ? ResolveType(param.Type) : NSLTypes.Any;
            paramTypes.Add(paramType);
            _variables[param.Name] = paramType;
        }

        // IMPORTANT: Register function with preliminary type BEFORE checking body
        // This allows recursive calls to find the function
        // Use Any as initial return type, will be updated after checking body
        var preliminaryFuncType = new NSLFunctionType(paramTypes, NSLTypes.Any);
        _functions[node.Name] = preliminaryFuncType;

        // Infer return type from body
        _currentFunctionReturnType = null;
        var bodyType = Check(node.Body);
        var returnType = _currentFunctionReturnType ?? bodyType;

        // Restore scope
        _variables.Clear();
        foreach (var kv in savedVariables)
            _variables[kv.Key] = kv.Value;

        // Update function type with actual return type
        var funcType = new NSLFunctionType(paramTypes, returnType);
        _functions[node.Name] = funcType;
        return funcType;
    }

    private NSLType CheckFunctionCall(NSLFunctionCallNode node)
    {
        var funcType = Check(node.Function);

        if (funcType is not NSLFunctionType func)
            return ReportError(node, $"Cannot call non-function type: {funcType}");

        if (node.Arguments.Count != func.ParameterTypes.Count)
            return ReportError(node,
                $"Expected {func.ParameterTypes.Count} arguments, got {node.Arguments.Count}");

        for (int i = 0; i < node.Arguments.Count; i++)
        {
            var argType = Check(node.Arguments[i]);
            if (!argType.IsAssignableTo(func.ParameterTypes[i]))
                return ReportError(node,
                    $"Argument {i + 1}: expected {func.ParameterTypes[i]}, got {argType}");
        }

        return func.ReturnType;
    }

    private NSLType CheckReturn(NSLReturnNode node)
    {
        var returnType = node.Value != null ? Check(node.Value) : NSLTypes.Void;
        _currentFunctionReturnType = returnType;
        return returnType;
    }

    private NSLType CheckIf(NSLIfNode node)
    {
        var condType = Check(node.Condition);
        if (condType is not NSLBoolType)
            ReportError(node, $"If condition must be bool, got {condType}");

        var thenType = Check(node.ThenBranch);

        if (node.ElseBranch != null)
        {
            var elseType = Check(node.ElseBranch);

            // Special handling for Result types - unify them
            if (thenType is NSLResultType thenResult && elseType is NSLResultType elseResult)
            {
                var unifiedOk = thenResult.OkType.IsAssignableTo(elseResult.OkType) ? elseResult.OkType :
                                elseResult.OkType.IsAssignableTo(thenResult.OkType) ? thenResult.OkType :
                                NSLTypes.Any;
                var unifiedErr = thenResult.ErrType.IsAssignableTo(elseResult.ErrType) ? elseResult.ErrType :
                                 elseResult.ErrType.IsAssignableTo(thenResult.ErrType) ? thenResult.ErrType :
                                 NSLTypes.Any;
                return new NSLResultType(unifiedOk, unifiedErr);
            }

            // Special handling for Optional types - unify them
            if (thenType is NSLOptionalType thenOpt && elseType is NSLOptionalType elseOpt)
            {
                var unifiedInner = thenOpt.InnerType.IsAssignableTo(elseOpt.InnerType) ? elseOpt.InnerType :
                                   elseOpt.InnerType.IsAssignableTo(thenOpt.InnerType) ? thenOpt.InnerType :
                                   NSLTypes.Any;
                return new NSLOptionalType(unifiedInner);
            }

            // If-else expression returns common type
            if (thenType.IsAssignableTo(elseType))
                return elseType;
            if (elseType.IsAssignableTo(thenType))
                return thenType;
            return NSLTypes.Any; // Fallback to any if types don't match
        }

        return NSLTypes.Void;
    }

    private NSLType CheckWhile(NSLWhileNode node)
    {
        var condType = Check(node.Condition);
        if (condType is not NSLBoolType)
            ReportError(node, $"While condition must be bool, got {condType}");

        Check(node.Body);
        return NSLTypes.Void;
    }

    private NSLType CheckFor(NSLForNode node)
    {
        var iterableType = Check(node.Iterable);

        // Determine element type
        NSLType elementType;
        if (iterableType is NSLArrayType arrayType)
            elementType = arrayType.ElementType;
        else if (iterableType is NSLRangeType rangeType)
            elementType = rangeType.ElementType;
        else if (iterableType is NSLVecType)
            elementType = NSLTypes.Number;
        else
            elementType = NSLTypes.Any;

        // Add loop variable to scope
        var savedVariables = new Dictionary<string, NSLType>(_variables);
        _variables[node.Variable.Value] = elementType;

        Check(node.Body);

        // Restore scope
        _variables.Clear();
        foreach (var kv in savedVariables)
            _variables[kv.Key] = kv.Value;

        return NSLTypes.Void;
    }

    private NSLType CheckBlock(NSLBlockNode node)
    {
        NSLType lastType = NSLTypes.Void;
        foreach (var stmt in node.Statements)
        {
            lastType = Check(stmt);
        }
        return lastType;
    }

    private NSLType CheckArray(NSLArrayNode node)
    {
        if (node.Elements.Count == 0)
            return new NSLArrayType(NSLTypes.Any);

        var elementType = Check(node.Elements[0]);
        foreach (var elem in node.Elements.Skip(1))
        {
            var elemType = Check(elem);
            if (!elemType.IsAssignableTo(elementType))
            {
                // Widen to common type
                if (elementType.IsAssignableTo(elemType))
                    elementType = elemType;
                else
                    elementType = NSLTypes.Any;
            }
        }

        // Special case: array of numbers is vec
        if (elementType is NSLNumberType)
            return new NSLVecType(node.Elements.Count);

        return new NSLArrayType(elementType);
    }

    private NSLType CheckListComprehension(NSLListComprehensionNode node)
    {
        // Get the iterable's element type
        var iterableType = Check(node.Iterable);
        NSLType elementType = NSLTypes.Any;

        if (iterableType is NSLArrayType arrayType)
            elementType = arrayType.ElementType;
        else if (iterableType is NSLVecType)
            elementType = NSLTypes.Number;
        else if (iterableType is NSLRangeType)
            elementType = NSLTypes.Number;
        // String iteration would yield characters (strings of length 1)
        else if (iterableType is NSLStringType)
            elementType = NSLTypes.String;

        // Save scope and add loop variable
        var savedVariables = new Dictionary<string, NSLType>(_variables);
        _variables[node.Variable] = elementType;

        // Check condition if present (must be bool)
        if (node.Condition != null)
        {
            var condType = Check(node.Condition);
            if (condType is not NSLBoolType && condType is not NSLAnyType)
            {
                ReportError(node, $"List comprehension condition must be bool, got {condType}");
            }
        }

        // Check expression with variable in scope
        var exprType = Check(node.Expression);

        // Restore scope
        _variables.Clear();
        foreach (var kv in savedVariables)
            _variables[kv.Key] = kv.Value;

        // Result is an array of the expression type
        return new NSLArrayType(exprType);
    }

    private NSLType CheckIndex(NSLIndexNode node)
    {
        var objType = Check(node.Object);
        var indexType = Check(node.Index);

        if (objType is NSLArrayType arrayType)
        {
            if (indexType is not NSLIntType && indexType is not NSLNumberType &&
                indexType is not NSLRangeType)
                ReportError(node, $"Index must be int or range, got {indexType}");
            return indexType is NSLRangeType ? objType : arrayType.ElementType;
        }

        if (objType is NSLVecType)
        {
            if (indexType is NSLRangeType)
                return new NSLVecType();
            return NSLTypes.Number;
        }

        if (objType is NSLMatType)
        {
            return NSLTypes.Number; // Simplified - actual would depend on indices
        }

        if (objType is NSLStringType)
        {
            return NSLTypes.String;
        }

        return ReportError(node, $"Cannot index type: {objType}");
    }

    private NSLType CheckGet(NSLGetNode node)
    {
        var objType = Check(node.Object);

        if (objType is NSLObjectType recordType)
        {
            if (recordType.Fields.TryGetValue(node.Name, out var fieldType))
                return fieldType;
            return ReportError(node, $"Unknown field: {node.Name} on type {objType}");
        }

        // Allow any property access on any type (duck typing fallback)
        return NSLTypes.Any;
    }

    private NSLType CheckPipeline(NSLPipelineNode node)
    {
        var leftType = Check(node.Left);
        var rightType = Check(node.Right);

        // Pipeline: left |> right means right(left)
        if (rightType is NSLFunctionType func)
        {
            if (func.ParameterTypes.Count != 1)
                return ReportError(node, "Pipeline target must accept exactly 1 argument");
            if (!leftType.IsAssignableTo(func.ParameterTypes[0]))
                return ReportError(node,
                    $"Cannot pipe {leftType} to function expecting {func.ParameterTypes[0]}");
            return func.ReturnType;
        }

        // If right is identifier, assume it's a function that takes left type
        return NSLTypes.Any;
    }

    private NSLType CheckRange(NSLRangeNode node)
    {
        var startType = node.Start != null ? Check(node.Start) : NSLTypes.Int;
        var endType = node.End != null ? Check(node.End) : NSLTypes.Int;

        if (!startType.IsNumeric)
            ReportError(node, $"Range start must be numeric, got {startType}");
        if (!endType.IsNumeric)
            ReportError(node, $"Range end must be numeric, got {endType}");

        return new NSLRangeType(NSLTypes.Int, node.IsInclusive);
    }

    private NSLType CheckSafeNavigation(NSLSafeNavigationNode node)
    {
        var objType = Check(node.Object);

        // T?.prop returns T_prop? (optional)
        if (objType is NSLObjectType recordType)
        {
            if (recordType.Fields.TryGetValue(node.Property, out var fieldType))
                return new NSLOptionalType(fieldType);
        }

        return new NSLOptionalType(NSLTypes.Any);
    }

    private NSLType CheckMatch(NSLMatchNode node)
    {
        var valueType = Check(node.Value);
        NSLType? resultType = null;

        foreach (var caseNode in node.Cases)
        {
            // Save variables for scoping
            var savedVariables = new Dictionary<string, NSLType>(_variables);

            // Handle pattern binding
            if (caseNode.Pattern is NSLIdentifierNode identPattern && !IsLiteralKeyword(identPattern.Name))
            {
                // Wildcard pattern - bind the value to a variable
                _variables[identPattern.Name] = valueType;
            }
            else if (caseNode.Pattern is NSLResultNode resultPattern)
            {
                // ok(v) or err(e) pattern - bind the inner variable
                if (resultPattern.Value is NSLIdentifierNode innerIdent)
                {
                    // Get the inner type from the Result type if available
                    NSLType innerType = NSLTypes.Number;
                    if (valueType is NSLResultType rt)
                    {
                        innerType = resultPattern.IsOk ? rt.OkType : rt.ErrType;
                    }
                    _variables[innerIdent.Name] = innerType;
                }
            }
            else if (caseNode.Pattern is NSLOptionalNode optPattern)
            {
                // some(v) pattern - bind the inner variable
                if (optPattern.HasValue && optPattern.Value is NSLIdentifierNode innerIdent)
                {
                    // Get the inner type from the Optional type if available
                    NSLType innerType = NSLTypes.Number;
                    if (valueType is NSLOptionalType ot)
                    {
                        innerType = ot.InnerType;
                    }
                    _variables[innerIdent.Name] = innerType;
                }
                // none pattern has no binding
            }
            else
            {
                // Literal pattern or other - just check type compatibility
                Check(caseNode.Pattern);
            }

            // Check guard condition if present (must be bool)
            if (caseNode.Guard != null)
            {
                var guardType = Check(caseNode.Guard);
                if (guardType is not NSLBoolType && guardType is not NSLAnyType)
                {
                    ReportError(node, $"Match guard must be bool, got {guardType}");
                }
            }

            // Check body with pattern variable in scope
            var bodyType = Check(caseNode.Body);

            // Restore scope
            _variables.Clear();
            foreach (var kv in savedVariables)
                _variables[kv.Key] = kv.Value;

            if (resultType == null)
                resultType = bodyType;
            else if (!bodyType.IsAssignableTo(resultType))
            {
                if (resultType.IsAssignableTo(bodyType))
                    resultType = bodyType;
                else
                    resultType = NSLTypes.Any;
            }
        }

        return resultType ?? NSLTypes.Void;
    }

    /// <summary>
    /// Check if an identifier is a literal keyword (not a pattern variable)
    /// </summary>
    private bool IsLiteralKeyword(string name)
    {
        return name == "none" || name == "true" || name == "false" || name == "null";
    }

    private NSLType CheckResult(NSLResultNode node)
    {
        var valueType = Check(node.Value);
        if (node.IsOk)
            return new NSLResultType(valueType, NSLTypes.String);
        else
            return new NSLResultType(NSLTypes.Any, valueType);
    }

    private NSLType CheckOptional(NSLOptionalNode node)
    {
        if (node.HasValue && node.Value != null)
        {
            var valueType = Check(node.Value);
            return new NSLOptionalType(valueType);
        }
        return new NSLOptionalType(NSLTypes.Any);
    }

    private NSLType CheckCast(NSLCastNode node)
    {
        Check(node.Value); // Check for errors
        return ResolveType(node.TargetType);
    }

    private NSLType CheckLambda(NSLLambdaNode node)
    {
        var paramTypes = new List<NSLType>();

        // Enter lambda scope
        var savedVariables = new Dictionary<string, NSLType>(_variables);

        foreach (var param in node.Parameters)
        {
            var paramType = param.Type != null ? ResolveType(param.Type) : NSLTypes.Any;
            paramTypes.Add(paramType);
            _variables[param.Name] = paramType;
        }

        var bodyType = Check(node.Body);

        // Restore scope
        _variables.Clear();
        foreach (var kv in savedVariables)
            _variables[kv.Key] = kv.Value;

        return new NSLFunctionType(paramTypes, bodyType);
    }

    private NSLType CheckTypeAlias(NSLTypeAliasNode node)
    {
        var defType = Check(node.Definition);
        _typeAliases[node.Name] = defType;
        return NSLTypes.Void;
    }

    private NSLType CheckObject(NSLObjectNode node)
    {
        var fields = new Dictionary<string, NSLType>();
        foreach (var field in node.Fields)
        {
            fields[field.Key] = Check(field.Value);
        }
        return new NSLObjectType(fields);
    }

    private NSLType CheckStructDefinition(NSLStructNode node)
    {
        var fields = new Dictionary<string, NSLType>();
        foreach (var field in node.Fields)
        {
            var fieldType = ResolveType(field.TypeName);
            fields[field.Name] = fieldType;
        }

        var structType = new NSLStructType(node.Name, fields);
        _structs[node.Name] = structType;
        _typeAliases[node.Name] = structType;
        return structType;
    }

    private NSLType CheckStructInstantiation(NSLStructInstantiationNode node)
    {
        if (!_structs.TryGetValue(node.StructName, out var structType))
        {
            return ReportError(node, $"Unknown struct type: {node.StructName}");
        }

        // Check that all required fields are provided
        var providedFields = new HashSet<string>();
        foreach (var field in node.Fields)
        {
            providedFields.Add(field.Key);

            if (!structType.Fields.TryGetValue(field.Key, out var expectedType))
            {
                ReportError(node, $"Unknown field '{field.Key}' in struct {node.StructName}");
                continue;
            }

            var valueType = Check(field.Value);
            if (!valueType.IsAssignableTo(expectedType))
            {
                ReportError(node, $"Field '{field.Key}': expected {expectedType}, got {valueType}");
            }
        }

        // Check for missing fields
        foreach (var requiredField in structType.Fields.Keys)
        {
            if (!providedFields.Contains(requiredField))
            {
                ReportError(node, $"Missing field '{requiredField}' in struct {node.StructName}");
            }
        }

        return structType;
    }

    private NSLType CheckEnumDefinition(NSLEnumNode node)
    {
        var variants = new Dictionary<string, NSLEnumVariantType>();

        foreach (var variant in node.Variants)
        {
            List<NSLType>? fieldTypes = null;
            if (variant.Fields != null && variant.Fields.Count > 0)
            {
                fieldTypes = new List<NSLType>();
                foreach (var fieldType in variant.Fields)
                {
                    fieldTypes.Add(ResolveType(fieldType));
                }
            }

            variants[variant.Name] = new NSLEnumVariantType(variant.Name, fieldTypes);
        }

        var enumType = new NSLEnumType(node.Name, variants);
        _enums[node.Name] = enumType;
        _typeAliases[node.Name] = enumType;
        return enumType;
    }

    private NSLType CheckEnumVariant(NSLEnumVariantNode node)
    {
        if (!_enums.TryGetValue(node.EnumName, out var enumType))
        {
            return ReportError(node, $"Unknown enum type: {node.EnumName}");
        }

        var variant = enumType.GetVariant(node.VariantName);
        if (variant == null)
        {
            return ReportError(node, $"Unknown variant '{node.VariantName}' in enum {node.EnumName}");
        }

        // Check arguments match variant field types
        int expectedArgs = variant.FieldTypes?.Count ?? 0;
        if (node.Arguments.Count != expectedArgs)
        {
            return ReportError(node,
                $"Variant {node.EnumName}::{node.VariantName} expects {expectedArgs} arguments, got {node.Arguments.Count}");
        }

        if (variant.FieldTypes != null)
        {
            for (int i = 0; i < node.Arguments.Count; i++)
            {
                var argType = Check(node.Arguments[i]);
                if (!argType.IsAssignableTo(variant.FieldTypes[i]))
                {
                    return ReportError(node,
                        $"Argument {i + 1} of {node.EnumName}::{node.VariantName}: expected {variant.FieldTypes[i]}, got {argType}");
                }
            }
        }

        return enumType;
    }

    private NSLType CheckTraitDefinition(NSLTraitNode node)
    {
        var methods = new Dictionary<string, NSLTraitMethodType>();

        foreach (var method in node.Methods)
        {
            var paramTypes = new List<NSLType>();
            foreach (var param in method.Parameters)
            {
                var paramType = param.Type != null ? ResolveType(param.Type) : NSLTypes.Any;
                paramTypes.Add(paramType);
            }

            var returnType = method.ReturnType != null ? ResolveType(method.ReturnType) : NSLTypes.Void;
            methods[method.Name] = new NSLTraitMethodType(method.Name, paramTypes, returnType);
        }

        var traitType = new NSLTraitType(node.Name, methods);
        _traits[node.Name] = traitType;
        _typeAliases[node.Name] = traitType;
        return traitType;
    }

    private NSLType CheckImplDefinition(NSLImplNode node)
    {
        // Verify the trait exists
        if (!_traits.TryGetValue(node.TraitName, out var traitType))
        {
            return ReportError(node, $"Unknown trait: {node.TraitName}");
        }

        // Verify the type exists (struct or enum)
        var targetType = _structs.ContainsKey(node.TypeName) ? (NSLType)_structs[node.TypeName] :
                         _enums.ContainsKey(node.TypeName) ? _enums[node.TypeName] :
                         null;

        if (targetType == null)
        {
            return ReportError(node, $"Unknown type: {node.TypeName}");
        }

        // Track implemented methods
        var implKey = (node.TraitName, node.TypeName);
        if (!_traitImplementations.ContainsKey(implKey))
        {
            _traitImplementations[implKey] = new HashSet<string>();
        }

        // Check each implemented method
        foreach (var method in node.Methods)
        {
            // Verify the method is required by the trait
            if (!traitType.Methods.TryGetValue(method.Name, out var traitMethod))
            {
                ReportError(node, $"Method '{method.Name}' is not defined in trait {node.TraitName}");
                continue;
            }

            // Type check the method body
            Check(method);

            // Mark this method as implemented
            _traitImplementations[implKey].Add(method.Name);

            // Register the method as a function with mangled name: TypeName_TraitName_MethodName
            var mangledName = $"{node.TypeName}_{node.TraitName}_{method.Name}";
            var paramTypes = method.Parameters.Select(p => p.Type != null ? ResolveType(p.Type) : NSLTypes.Any).ToList();
            var returnType = method.ReturnType != null ? ResolveType(method.ReturnType) : NSLTypes.Any;
            _functions[mangledName] = new NSLFunctionType(paramTypes, returnType);
        }

        // Check that all required methods are implemented
        foreach (var requiredMethod in traitType.Methods.Keys)
        {
            if (!_traitImplementations[implKey].Contains(requiredMethod))
            {
                ReportError(node, $"Missing implementation of '{requiredMethod}' for trait {node.TraitName} on type {node.TypeName}");
            }
        }

        return NSLTypes.Void;
    }

    private NSLType CheckAsyncFunction(NSLAsyncFunctionNode node)
    {
        var paramTypes = new List<NSLType>();

        // Enter function scope
        var savedVariables = new Dictionary<string, NSLType>(_variables);

        foreach (var param in node.Parameters)
        {
            var paramType = param.Type != null ? ResolveType(param.Type) : NSLTypes.Any;
            paramTypes.Add(paramType);
            _variables[param.Name] = paramType;
        }

        // Register function before checking body (for recursion)
        var preliminaryFuncType = new NSLFunctionType(paramTypes, new NSLFutureType(NSLTypes.Any));
        _functions[node.Name] = preliminaryFuncType;

        // Infer return type from body
        _currentFunctionReturnType = null;
        var bodyType = Check(node.Body);
        var innerReturnType = _currentFunctionReturnType ?? bodyType;

        // Restore scope
        _variables.Clear();
        foreach (var kv in savedVariables)
            _variables[kv.Key] = kv.Value;

        // Async functions return Future<T>
        var returnType = new NSLFutureType(innerReturnType);
        var funcType = new NSLFunctionType(paramTypes, returnType);
        _functions[node.Name] = funcType;
        return funcType;
    }

    private NSLType CheckAwait(NSLAwaitNode node)
    {
        var exprType = Check(node.Expression);

        // await on Future<T> returns T
        if (exprType is NSLFutureType futureType)
        {
            return futureType.InnerType;
        }

        // await on non-Future is an error, but allow Any for flexibility
        if (exprType is NSLAnyType)
        {
            return NSLTypes.Any;
        }

        return ReportError(node, $"Cannot await non-Future type: {exprType}");
    }

    #region Module System Type Checking

    /// <summary>
    /// Track imported modules and their exports
    /// </summary>
    private readonly Dictionary<string, Dictionary<string, NSLType>> _importedModules = new();

    /// <summary>
    /// Check import statement - registers imported symbols
    /// </summary>
    private NSLType CheckImport(NSLImportNode node)
    {
        // Try to load the module
        ModuleResolver.LoadedModule? loadedModule = null;

        if (_moduleResolver != null)
        {
            try
            {
                if (!string.IsNullOrEmpty(node.FilePath))
                {
                    // Import from file path: import "./mymodule.nsl"
                    loadedModule = _moduleResolver.LoadModuleFromFile(node.FilePath);
                }
                else if (node.ModulePath != null && node.ModulePath.Count > 0)
                {
                    // Import from module path: import math, import { sin } from math
                    loadedModule = _moduleResolver.LoadModule(node.ModulePath);
                }
            }
            catch (Exception ex)
            {
                return ReportError(node, $"Failed to load module: {ex.Message}");
            }
        }

        if (node.IsWildcard)
        {
            // import * from module - all exports become available
            if (loadedModule != null)
            {
                foreach (var (name, symbol) in loadedModule.Exports)
                {
                    RegisterImportedSymbol(name, symbol);
                }
            }
            else
            {
                // Fallback if no module resolver
                ReportError(node, $"Cannot load module: {node.FullModulePath} (no module resolver configured)");
            }
        }
        else if (node.Items != null)
        {
            // import { x, y } from module - specific items
            foreach (var item in node.Items)
            {
                if (loadedModule != null && loadedModule.Exports.TryGetValue(item.Name, out var symbol))
                {
                    // Register with actual type from module
                    RegisterImportedSymbol(item.LocalName, symbol);
                }
                else if (loadedModule != null)
                {
                    ReportError(node, $"Module '{node.FullModulePath}' does not export '{item.Name}'");
                }
                else
                {
                    // Fallback to Any if module not loaded
                    _variables[item.LocalName] = NSLTypes.Any;
                }
            }
        }
        else if (node.ModuleAlias != null)
        {
            // import module as alias - namespace import
            if (loadedModule != null)
            {
                // Store module exports for qualified access
                var moduleExports = new Dictionary<string, NSLType>();
                foreach (var (name, symbol) in loadedModule.Exports)
                {
                    moduleExports[name] = symbol.Type;
                }
                _importedModules[node.ModuleAlias] = moduleExports;
            }
            _variables[node.ModuleAlias] = new NSLModuleType(node.FullModulePath);
        }
        else
        {
            // import module - make module available as namespace
            var moduleName = node.ModulePath?.LastOrDefault() ?? "unknown";
            if (loadedModule != null)
            {
                var moduleExports = new Dictionary<string, NSLType>();
                foreach (var (name, symbol) in loadedModule.Exports)
                {
                    moduleExports[name] = symbol.Type;
                }
                _importedModules[moduleName] = moduleExports;
            }
            _variables[moduleName] = new NSLModuleType(node.FullModulePath);
        }

        return NSLTypes.Void;
    }

    /// <summary>
    /// Register an imported symbol in the current scope
    /// </summary>
    private void RegisterImportedSymbol(string name, ModuleResolver.ExportedSymbol symbol)
    {
        switch (symbol.Kind)
        {
            case ModuleResolver.SymbolKind.Function:
                if (symbol.Type is NSLFunctionType funcType)
                {
                    _functions[name] = funcType;
                }
                else
                {
                    _variables[name] = symbol.Type;
                }
                break;

            case ModuleResolver.SymbolKind.Variable:
            case ModuleResolver.SymbolKind.Constant:
                _variables[name] = symbol.Type;
                break;

            case ModuleResolver.SymbolKind.Type:
                _typeAliases[name] = symbol.Type;
                break;
        }
    }

    /// <summary>
    /// Check export statement - marks symbols as public
    /// </summary>
    private NSLType CheckExport(NSLExportNode node)
    {
        // If this is a pub declaration, check the declaration first
        if (node.Declaration != null)
        {
            return Check(node.Declaration);
        }

        // If this is an export list, verify symbols exist
        if (node.Items != null)
        {
            foreach (var item in node.Items)
            {
                if (!_variables.ContainsKey(item.Name) && !_functions.ContainsKey(item.Name))
                {
                    ReportError(node, $"Cannot export undefined symbol: {item.Name}");
                }
            }
        }

        return NSLTypes.Void;
    }

    /// <summary>
    /// Check module declaration - creates a module scope
    /// </summary>
    private NSLType CheckModule(NSLModuleNode node)
    {
        // Save current scope
        var savedVariables = new Dictionary<string, NSLType>(_variables);
        var savedFunctions = new Dictionary<string, NSLFunctionType>(_functions);

        // Check module body
        Check(node.Body);

        // In a full implementation, would collect exports and store them
        // associated with the module path

        // Restore outer scope
        _variables.Clear();
        foreach (var kv in savedVariables)
            _variables[kv.Key] = kv.Value;
        _functions.Clear();
        foreach (var kv in savedFunctions)
            _functions[kv.Key] = kv.Value;

        return NSLTypes.Void;
    }

    #endregion

    private NSLType ResolveType(string typeName)
    {
        if (_typeAliases.TryGetValue(typeName, out var aliasType))
            return aliasType;

        try
        {
            return NSLTypes.FromName(typeName);
        }
        catch
        {
            _errors.Add(new TypeCheckError($"Unknown type: {typeName}", 0, 0));
            return NSLTypes.Any;
        }
    }

    private NSLType ReportError(NSLASTNode node, string message)
    {
        _errors.Add(new TypeCheckError(message, node.Line, node.Column));
        return NSLTypes.Any;
    }

    /// <summary>
    /// Register built-in functions
    /// </summary>
    public void RegisterBuiltins()
    {
        // I/O
        _functions["print"] = new NSLFunctionType(new[] { NSLTypes.Any }, NSLTypes.Void);
        _functions["println"] = new NSLFunctionType(new[] { NSLTypes.Any }, NSLTypes.Void);
        _functions["input"] = new NSLFunctionType(Array.Empty<NSLType>(), NSLTypes.String);

        // Math - basic functions
        _functions["sqrt"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["abs"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["sin"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["cos"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["tan"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["exp"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["log"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["log10"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["log2"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["pow"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);
        _functions["floor"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["ceil"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["round"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["trunc"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);

        // Math - inverse trigonometric
        _functions["asin"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["acos"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["atan"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["atan2"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);

        // Math - hyperbolic
        _functions["sinh"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["cosh"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);

        // Math - utility functions
        _functions["min"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);
        _functions["max"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);
        _functions["clamp"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);
        _functions["lerp"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);
        _functions["sign"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["copysign"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);

        // Vector/Matrix operations
        _functions["len"] = new NSLFunctionType(new[] { NSLTypes.Any }, NSLTypes.Int);
        _functions["sum"] = new NSLFunctionType(new[] { new NSLVecType() }, NSLTypes.Number);
        _functions["mean"] = new NSLFunctionType(new[] { new NSLVecType() }, NSLTypes.Number);
        _functions["dot"] = new NSLFunctionType(new[] { new NSLVecType(), new NSLVecType() }, NSLTypes.Number);
        _functions["zeros"] = new NSLFunctionType(new[] { NSLTypes.Int }, new NSLVecType());
        _functions["ones"] = new NSLFunctionType(new[] { NSLTypes.Int }, new NSLVecType());
        _functions["range"] = new NSLFunctionType(new[] { NSLTypes.Int, NSLTypes.Int }, new NSLArrayType(NSLTypes.Int));

        // AI/ML activations (work on both scalars and vectors)
        _functions["relu"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["sigmoid"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["tanh"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["leaky_relu"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["softplus"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["gelu"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Number);
        _functions["softmax"] = new NSLFunctionType(new[] { new NSLVecType() }, new NSLVecType());
        _functions["normalize"] = new NSLFunctionType(new[] { new NSLVecType() }, new NSLVecType());

        // String operations - basic
        _functions["str_len"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.Int);
        _functions["str_concat"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.String);
        _functions["str_slice"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.Int, NSLTypes.Int }, NSLTypes.String);
        _functions["substr"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.Int, NSLTypes.Int }, NSLTypes.String);
        _functions["slice"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.Int, NSLTypes.Int }, NSLTypes.String);
        _functions["strlen"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.Int);

        // String operations - search/check
        _functions["contains"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.Bool);
        _functions["starts_with"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.Bool);
        _functions["ends_with"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.Bool);
        _functions["index_of"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.Int);
        _functions["char_at"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.Int }, NSLTypes.Int);

        // String operations - transformation
        _functions["to_upper"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.String);
        _functions["to_lower"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.String);
        _functions["trim"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.String);
        _functions["replace"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String, NSLTypes.String }, NSLTypes.String);

        // String operations - split/join
        _functions["split"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, new NSLArrayType(NSLTypes.String));
        _functions["join"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.String), NSLTypes.String }, NSLTypes.String);

        // File I/O
        _functions["read_file"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.String);
        _functions["write_file"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.Bool);
        _functions["append_file"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.Bool);
        _functions["file_exists"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.Bool);
        _functions["delete_file"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.Bool);

        // Array utility functions
        _functions["array_sum"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Number) }, NSLTypes.Number);
        _functions["array_product"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Number) }, NSLTypes.Number);
        _functions["array_min"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Number) }, NSLTypes.Number);
        _functions["array_max"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Number) }, NSLTypes.Number);
        _functions["array_avg"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Number) }, NSLTypes.Number);
        _functions["array_reverse"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Any) }, new NSLArrayType(NSLTypes.Any));
        _functions["array_sort"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Number) }, new NSLArrayType(NSLTypes.Number));
        _functions["array_contains"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Any), NSLTypes.Any }, NSLTypes.Bool);
        _functions["array_push"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Any), NSLTypes.Any }, new NSLArrayType(NSLTypes.Any));
        _functions["array_pop"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Any) }, NSLTypes.Any);
        _functions["array_slice"] = new NSLFunctionType(new[] { new NSLArrayType(NSLTypes.Any), NSLTypes.Int, NSLTypes.Int }, new NSLArrayType(NSLTypes.Any));

        // Type conversion utilities
        _functions["to_int"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["to_number"] = new NSLFunctionType(new[] { NSLTypes.Int }, NSLTypes.Number);
        _functions["to_string"] = new NSLFunctionType(new[] { NSLTypes.Any }, NSLTypes.String);
        _functions["parse_int"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.Int);
        _functions["parse_number"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.Number);

        // Result type methods/functions
        // Use Any for Result types to allow any Result variant (ok(number) creates Result<number, string>)
        var resultType = new NSLResultType(NSLTypes.Any, NSLTypes.Any);
        _functions["is_ok"] = new NSLFunctionType(new[] { resultType }, NSLTypes.Bool);
        _functions["is_err"] = new NSLFunctionType(new[] { resultType }, NSLTypes.Bool);
        _functions["unwrap"] = new NSLFunctionType(new[] { resultType }, NSLTypes.Any);
        _functions["unwrap_or"] = new NSLFunctionType(new[] { resultType, NSLTypes.Any }, NSLTypes.Any);
        _functions["unwrap_err"] = new NSLFunctionType(new[] { resultType }, NSLTypes.Any);
        _functions["expect"] = new NSLFunctionType(new[] { resultType, NSLTypes.String }, NSLTypes.Any);
        // result_map and result_and_then take closures - using Any for the function type
        _functions["result_map"] = new NSLFunctionType(new[] { resultType, NSLTypes.Any }, resultType);
        _functions["result_and_then"] = new NSLFunctionType(new[] { resultType, NSLTypes.Any }, resultType);
        _functions["result_or_else"] = new NSLFunctionType(new[] { resultType, NSLTypes.Any }, resultType);
        _functions["result_ok_or"] = new NSLFunctionType(new[] { resultType, NSLTypes.Any }, NSLTypes.Any);

        // Optional type methods/functions
        // Use Any for Optional types to allow any Optional variant
        var optionalType = new NSLOptionalType(NSLTypes.Any);
        _functions["is_some"] = new NSLFunctionType(new[] { optionalType }, NSLTypes.Bool);
        _functions["is_none"] = new NSLFunctionType(new[] { optionalType }, NSLTypes.Bool);
        // unwrap and unwrap_or are overloaded for Optional as well (handled at codegen)
        _functions["option_map"] = new NSLFunctionType(new[] { optionalType, NSLTypes.Any }, optionalType);
        _functions["option_and_then"] = new NSLFunctionType(new[] { optionalType, NSLTypes.Any }, optionalType);
        _functions["option_or_else"] = new NSLFunctionType(new[] { optionalType, NSLTypes.Any }, optionalType);
        _functions["option_unwrap_or"] = new NSLFunctionType(new[] { optionalType, NSLTypes.Any }, NSLTypes.Any);
        _functions["option_expect"] = new NSLFunctionType(new[] { optionalType, NSLTypes.String }, NSLTypes.Any);

        // HTTP client functions
        _functions["http_get"] = new NSLFunctionType(new[] { NSLTypes.String }, new NSLResultType(NSLTypes.String, NSLTypes.String));
        _functions["http_post"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, new NSLResultType(NSLTypes.String, NSLTypes.String));
        _functions["http_put"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, new NSLResultType(NSLTypes.String, NSLTypes.String));
        _functions["http_delete"] = new NSLFunctionType(new[] { NSLTypes.String }, new NSLResultType(NSLTypes.String, NSLTypes.String));
        _functions["http_head"] = new NSLFunctionType(new[] { NSLTypes.String }, new NSLResultType(NSLTypes.String, NSLTypes.String));
        _functions["url_encode"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.String);
        _functions["url_decode"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.String);

        // JSON functions
        _functions["json_parse"] = new NSLFunctionType(new[] { NSLTypes.String }, new NSLResultType(NSLTypes.Any, NSLTypes.String));
        _functions["json_stringify"] = new NSLFunctionType(new[] { NSLTypes.Any }, NSLTypes.String);
        _functions["json_get"] = new NSLFunctionType(new[] { NSLTypes.Any, NSLTypes.String }, NSLTypes.Any);
        _functions["json_set"] = new NSLFunctionType(new[] { NSLTypes.Any, NSLTypes.String, NSLTypes.Any }, NSLTypes.Any);
        _functions["json_has"] = new NSLFunctionType(new[] { NSLTypes.Any, NSLTypes.String }, NSLTypes.Bool);
        _functions["json_keys"] = new NSLFunctionType(new[] { NSLTypes.Any }, new NSLArrayType(NSLTypes.String));
        _functions["json_values"] = new NSLFunctionType(new[] { NSLTypes.Any }, new NSLArrayType(NSLTypes.Any));

        // Regex functions
        _functions["regex_match"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, NSLTypes.Bool);
        _functions["regex_find"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, new NSLOptionalType(NSLTypes.String));
        _functions["regex_find_all"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, new NSLArrayType(NSLTypes.String));
        _functions["regex_replace"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String, NSLTypes.String }, NSLTypes.String);
        _functions["regex_split"] = new NSLFunctionType(new[] { NSLTypes.String, NSLTypes.String }, new NSLArrayType(NSLTypes.String));

        // Date/Time functions
        _functions["now"] = new NSLFunctionType(Array.Empty<NSLType>(), NSLTypes.Number);  // Unix timestamp
        _functions["now_ms"] = new NSLFunctionType(Array.Empty<NSLType>(), NSLTypes.Number);  // Unix timestamp in milliseconds
        _functions["date_parse"] = new NSLFunctionType(new[] { NSLTypes.String }, NSLTypes.Number);  // Parse date string to timestamp
        _functions["date_format"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.String }, NSLTypes.String);  // Format timestamp
        _functions["date_year"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["date_month"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["date_day"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["date_hour"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["date_minute"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["date_second"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["date_weekday"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Int);
        _functions["date_add_days"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Int }, NSLTypes.Number);
        _functions["date_add_hours"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Int }, NSLTypes.Number);
        _functions["date_diff"] = new NSLFunctionType(new[] { NSLTypes.Number, NSLTypes.Number }, NSLTypes.Number);  // Difference in seconds
        _functions["sleep"] = new NSLFunctionType(new[] { NSLTypes.Number }, NSLTypes.Void);  // Sleep for milliseconds
    }
}

/// <summary>
/// Type checking error
/// </summary>
public class TypeCheckError
{
    public string Message { get; }
    public int Line { get; }
    public int Column { get; }

    public TypeCheckError(string message, int line, int column)
    {
        Message = message;
        Line = line;
        Column = column;
    }

    public override string ToString() =>
        Line > 0 ? $"Error at {Line}:{Column}: {Message}" : $"Error: {Message}";
}
