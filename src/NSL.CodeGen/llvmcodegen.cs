using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using LLVMSharp.Interop;
using NSL.Core;
using NSL.Core.AST;
using NSL.TypeSystem;
using NSLTokenType = NSL.Core.Tokens.TokenType;

namespace NSL.CodeGen;

/// <summary>
/// LLVM IR Code Generator for NSL
/// Translates typed AST to LLVM IR for native compilation
/// </summary>
public class LLVMCodeGen : IDisposable
{
    private readonly LLVMContextRef _context;
    private readonly LLVMModuleRef _module;
    private readonly LLVMBuilderRef _builder;
    private readonly TypeChecker _typeChecker;

    private readonly Dictionary<string, LLVMValueRef> _namedValues = new();
    private readonly Dictionary<string, LLVMTypeRef> _namedValueTypes = new();  // Track pointee types for LLVM 16+
    private readonly Dictionary<string, LLVMValueRef> _globalValues = new();  // Global constants from imports
    private readonly Dictionary<string, LLVMTypeRef> _globalValueTypes = new();  // Types for global constants
    private readonly Dictionary<string, (LLVMTypeRef ElementType, uint Size)> _arrayInfo = new();  // Track array element type and size
    private readonly Dictionary<string, LLVMValueRef> _functions = new();
    private readonly Dictionary<string, LLVMTypeRef> _functionTypes = new();
    private readonly Dictionary<string, LLVMTypeRef> _structTypes = new();
    private readonly HashSet<string> _processedModules = new();

    // Closure support: track closure environment types and captured variables
    private readonly Dictionary<string, LLVMTypeRef> _closureEnvTypes = new();
    private readonly Dictionary<string, List<string>> _closureCapturedVars = new();
    private readonly Dictionary<string, LLVMTypeRef> _closureFuncTypes = new();  // Track function types for closure variables
    private readonly Dictionary<LLVMTypeRef, LLVMTypeRef> _closureStructToFuncType = new();  // Map closure struct type to its function type
    private LLVMValueRef? _currentEnvPtr;  // Environment pointer for current closure

    private LLVMValueRef _currentFunction;
    private LLVMBasicBlockRef _currentBlock;
    private ModuleResolver? _moduleResolver;
    
    // Loop context for break/continue support
    private readonly Stack<(LLVMBasicBlockRef ExitBlock, LLVMBasicBlockRef ContinueBlock)> _loopStack = new();

    public LLVMCodeGen(string moduleName, TypeChecker typeChecker)
    {
        // Use the global context to avoid context mismatch issues with static type refs
        _context = LLVMContextRef.Global;
        _module = LLVMModuleRef.CreateWithName(moduleName);
        _builder = _context.CreateBuilder();
        _typeChecker = typeChecker;

        DeclareBuiltins();
    }

    /// <summary>
    /// Set the module resolver for processing imports
    /// </summary>
    public void SetModuleResolver(ModuleResolver resolver)
    {
        _moduleResolver = resolver;
    }

    /// <summary>
    /// Process all imported modules and generate their code
    /// </summary>
    public void ProcessImportedModules()
    {
        if (_moduleResolver == null) return;

        foreach (var (_, loadedModule) in _moduleResolver.LoadedModules)
        {
            if (!_processedModules.Contains(loadedModule.ModuleName))
            {
                _processedModules.Add(loadedModule.ModuleName);
                ProcessImportedModule(loadedModule);
            }
        }

    }

    private void ProcessImportedModule(ModuleResolver.LoadedModule module)
    {
        // Generate code for all exported symbols from the module
        foreach (var (name, symbol) in module.Exports)
        {
            if (symbol.Declaration != null)
            {
                switch (symbol.Declaration)
                {
                    case NSLFunctionNode func:
                        // Generate the function if not already generated
                        if (!_functions.ContainsKey(func.Name))
                        {
                            GenerateFunction(func);
                        }
                        break;

                    case NSLVariableDeclarationNode varDecl:
                        // Generate a global constant for exported variables
                        if (!_namedValues.ContainsKey(varDecl.Name) && varDecl.Value != null)
                        {
                            GenerateGlobalConstant(varDecl.Name, varDecl.Value);
                        }
                        break;
                }
            }
        }
    }

    /// <summary>
    /// Generate a global constant from a variable declaration
    /// </summary>
    private void GenerateGlobalConstant(string name, NSLASTNode value)
    {
        // Evaluate the constant expression
        if (value is NSLLiteralNode literal)
        {
            LLVMValueRef globalValue;
            LLVMTypeRef globalType;

            switch (literal.ValueType)
            {
                case NSLTokenType.Number:
                    globalType = LLVMTypeRef.Double;
                    var doubleVal = double.Parse(literal.Value.ToString() ?? "0");
                    globalValue = _module.AddGlobal(globalType, name);
                    globalValue.Initializer = LLVMValueRef.CreateConstReal(globalType, doubleVal);
                    break;

                case NSLTokenType.Integer:
                    globalType = LLVMTypeRef.Int64;
                    var intVal = long.Parse(literal.Value.ToString() ?? "0");
                    globalValue = _module.AddGlobal(globalType, name);
                    globalValue.Initializer = LLVMValueRef.CreateConstInt(globalType, (ulong)intVal, true);
                    break;

                case NSLTokenType.String:
                    // String constants
                    var strVal = literal.Value?.ToString() ?? "";
                    globalValue = _builder.BuildGlobalStringPtr(strVal, name);
                    globalType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
                    break;

                default:
                    return; // Unknown type, skip
            }

            // Store in global dictionaries (preserved across function boundaries)
            _globalValues[name] = globalValue;
            _globalValueTypes[name] = globalType;
        }
    }

    /// <summary>
    /// Generate LLVM IR for an AST node
    /// </summary>
    public LLVMValueRef Generate(NSLASTNode node)
    {
        return node switch
        {
            NSLLiteralNode literal => GenerateLiteral(literal),
            NSLIdentifierNode ident => GenerateIdentifier(ident),
            NSLBinaryOperationNode binary => GenerateBinaryOp(binary),
            NSLUnaryOperationNode unary => GenerateUnaryOp(unary),
            NSLVariableDeclarationNode varDecl => GenerateVariableDecl(varDecl),
            NSLAssignmentNode assign => GenerateAssignment(assign),
            NSLFunctionNode func => GenerateFunction(func),
            NSLFunctionCallNode call => GenerateFunctionCall(call),
            NSLReturnNode ret => GenerateReturn(ret),
            NSLIfNode ifNode => GenerateIf(ifNode),
            NSLWhileNode whileNode => GenerateWhile(whileNode),
            NSLForNode forNode => GenerateFor(forNode),
            NSLBlockNode block => GenerateBlock(block),
            NSLArrayNode array => GenerateArray(array),
            NSLListComprehensionNode listComp => GenerateListComprehension(listComp),
            NSLIndexNode index => GenerateIndex(index),
            NSLPipelineNode pipeline => GeneratePipeline(pipeline),
            NSLRangeNode range => GenerateRange(range),
            NSLLambdaNode lambda => GenerateLambda(lambda),
            NSLBreakNode => GenerateBreak(),
            NSLContinueNode => GenerateContinue(),

            // Module system nodes - generate their contents
            NSLImportNode => GenerateImport(),
            NSLExportNode export => GenerateExport(export),
            NSLModuleNode module => GenerateModule(module),

            // Struct nodes
            NSLStructNode structDef => GenerateStructDefinition(structDef),
            NSLStructInstantiationNode structInst => GenerateStructInstantiation(structInst),
            NSLGetNode get => GenerateFieldAccess(get),

            // Pattern matching
            NSLMatchNode match => GenerateMatch(match),

            // Result/Option types
            NSLResultNode result => GenerateResult(result),
            NSLOptionalNode optional => GenerateOptional(optional),

            // Enum (algebraic data type) support
            NSLEnumNode enumDef => GenerateEnumDefinition(enumDef),
            NSLEnumVariantNode enumVariant => GenerateEnumVariant(enumVariant),

            // Trait/Interface support
            NSLTraitNode traitDef => GenerateTraitDefinition(traitDef),
            NSLImplNode implDef => GenerateImplDefinition(implDef),

            // Async/Await support
            NSLAsyncFunctionNode asyncFunc => GenerateAsyncFunction(asyncFunc),
            NSLAwaitNode awaitNode => GenerateAwait(awaitNode),

            _ => throw new NotImplementedException($"Code generation not implemented for {node.GetType().Name}")
        };
    }

    #region Literal Generation

    private LLVMValueRef GenerateLiteral(NSLLiteralNode node)
    {
        return node.ValueType switch
        {
            NSLTokenType.Number => LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, Convert.ToDouble(node.Value)),
            NSLTokenType.Integer => LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, Convert.ToUInt64(node.Value)),
            NSLTokenType.Boolean => LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, node.Value is true ? 1ul : 0ul),
            NSLTokenType.String => GenerateString(node.Value?.ToString() ?? ""),
            NSLTokenType.Null => LLVMValueRef.CreateConstNull(LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0)),
            _ => throw new InvalidOperationException($"Unknown literal type: {node.ValueType}")
        };
    }

    private LLVMValueRef GenerateString(string value)
    {
        return _builder.BuildGlobalStringPtr(value, "str");
    }

    #endregion

    #region Variable Generation

    private LLVMValueRef GenerateIdentifier(NSLIdentifierNode node)
    {
        // Check local variables first
        if (_namedValues.TryGetValue(node.Name, out var value))
        {
            // Load from alloca
            if (value.TypeOf.Kind == LLVMTypeKind.LLVMPointerTypeKind)
            {
                var pointeeType = GetPointeeType(node.Name, value);
                return _builder.BuildLoad2(pointeeType, value, node.Name);
            }
            return value;
        }

        // Check global constants (from imports)
        if (_globalValues.TryGetValue(node.Name, out var globalValue))
        {
            // Load from global variable
            if (globalValue.TypeOf.Kind == LLVMTypeKind.LLVMPointerTypeKind)
            {
                var pointeeType = _globalValueTypes.TryGetValue(node.Name, out var gType)
                    ? gType
                    : LLVMTypeRef.Double;
                return _builder.BuildLoad2(pointeeType, globalValue, node.Name);
            }
            return globalValue;
        }

        // Check functions
        if (_functions.TryGetValue(node.Name, out var func))
            return func;

        throw new InvalidOperationException($"Unknown variable: {node.Name}");
    }

    private LLVMValueRef GenerateVariableDecl(NSLVariableDeclarationNode node)
    {
        var varType = _typeChecker.Check(node);
        var llvmType = GetLLVMType(varType);
        LLVMValueRef? value = null;

        // Generate the value first to determine if it's a closure
        if (node.Value != null)
        {
            value = Generate(node.Value);

            // If the value is a closure struct, use its type instead of the declared type
            // This handles the case where TypeChecker says it's a function type but
            // the actual value is a closure struct
            if (value.Value.TypeOf.Kind == LLVMTypeKind.LLVMStructTypeKind &&
                _closureStructToFuncType.ContainsKey(value.Value.TypeOf))
            {
                llvmType = value.Value.TypeOf;
            }
        }

        // Create alloca in entry block
        var alloca = CreateEntryBlockAlloca(_currentFunction, node.Name, llvmType);

        if (value.HasValue)
        {
            var storeValue = EnsureType(value.Value, llvmType);
            _builder.BuildStore(storeValue, alloca);
        }

        _namedValues[node.Name] = alloca;
        _namedValueTypes[node.Name] = llvmType;  // Track pointee type

        // Track struct type for field access
        if (varType is NSLStructType structType)
        {
            _variableStructTypes[node.Name] = structType.StructName;
        }

        return alloca;
    }

    private LLVMValueRef GenerateAssignment(NSLAssignmentNode node)
    {
        if (!_namedValues.TryGetValue(node.Name, out var variable))
            throw new InvalidOperationException($"Unknown variable: {node.Name}");

        var value = Generate(node.Value);
        var targetType = GetPointeeType(node.Name, variable);
        value = EnsureType(value, targetType);

        _builder.BuildStore(value, variable);
        return value;
    }

    private LLVMValueRef CreateEntryBlockAlloca(LLVMValueRef function, string name, LLVMTypeRef type)
    {
        var entryBlock = function.EntryBasicBlock;
        var tmpBuilder = _context.CreateBuilder();

        if (entryBlock.FirstInstruction.Handle != IntPtr.Zero)
            tmpBuilder.PositionBefore(entryBlock.FirstInstruction);
        else
            tmpBuilder.PositionAtEnd(entryBlock);

        var alloca = tmpBuilder.BuildAlloca(type, name);
        tmpBuilder.Dispose();
        return alloca;
    }

    #endregion

    #region Binary Operations

    private LLVMValueRef GenerateBinaryOp(NSLBinaryOperationNode node)
    {
        var left = Generate(node.Left);
        var right = Generate(node.Right);

        // Check if both operands are strings (pointer types)
        bool isStringOp = left.TypeOf.Kind == LLVMTypeKind.LLVMPointerTypeKind &&
                          right.TypeOf.Kind == LLVMTypeKind.LLVMPointerTypeKind;

        // String concatenation with +
        if (node.Operator == NSLTokenType.Plus && isStringOp)
        {
            return _builder.BuildCall2(_functionTypes["str_concat"], _functions["str_concat"],
                new[] { left, right }, "concat");
        }

        // Check if both operands are integers
        bool isIntegerOp = IsInteger(left) && IsInteger(right);

        // For arithmetic, handle both int and float paths
        if (node.Operator is NSLTokenType.Plus or NSLTokenType.Minus or
            NSLTokenType.Multiply or NSLTokenType.Divide or NSLTokenType.Power or NSLTokenType.Percent)
        {
            if (isIntegerOp)
            {
                left = EnsureInt64(left);
                right = EnsureInt64(right);
            }
            else
            {
                left = EnsureDouble(left);
                right = EnsureDouble(right);
            }
        }

        // Bitwise operations always work on integers
        if (node.Operator is NSLTokenType.BitwiseAnd or NSLTokenType.BitwiseOr or
            NSLTokenType.BitwiseXor or NSLTokenType.LeftShift or NSLTokenType.RightShift or
            NSLTokenType.IntegerDivide)
        {
            left = EnsureInt64(left);
            right = EnsureInt64(right);
            isIntegerOp = true;
        }

        return node.Operator switch
        {
            // Arithmetic - integer or float
            NSLTokenType.Plus when isIntegerOp => _builder.BuildAdd(left, right, "addtmp"),
            NSLTokenType.Plus => _builder.BuildFAdd(left, right, "addtmp"),
            NSLTokenType.Minus when isIntegerOp => _builder.BuildSub(left, right, "subtmp"),
            NSLTokenType.Minus => _builder.BuildFSub(left, right, "subtmp"),
            NSLTokenType.Multiply when isIntegerOp => _builder.BuildMul(left, right, "multmp"),
            NSLTokenType.Multiply => _builder.BuildFMul(left, right, "multmp"),
            NSLTokenType.Divide when isIntegerOp => _builder.BuildSDiv(left, right, "divtmp"),
            NSLTokenType.Divide => _builder.BuildFDiv(left, right, "divtmp"),
            NSLTokenType.Percent when isIntegerOp => _builder.BuildSRem(left, right, "modtmp"),
            NSLTokenType.Percent => _builder.BuildFRem(left, right, "modtmp"),
            NSLTokenType.Power => GeneratePow(EnsureDouble(left), EnsureDouble(right)),

            // Integer division (always integer result)
            NSLTokenType.IntegerDivide => _builder.BuildSDiv(left, right, "idivtmp"),

            // Bitwise operations
            NSLTokenType.BitwiseAnd => _builder.BuildAnd(left, right, "bandtmp"),
            NSLTokenType.BitwiseOr => _builder.BuildOr(left, right, "bortmp"),
            NSLTokenType.BitwiseXor => _builder.BuildXor(left, right, "bxortmp"),
            NSLTokenType.LeftShift => _builder.BuildShl(left, right, "shltmp"),
            NSLTokenType.RightShift => _builder.BuildAShr(left, right, "ashrtmp"),

            // Comparison - handle both int and float
            NSLTokenType.Less when isIntegerOp => _builder.BuildICmp(LLVMIntPredicate.LLVMIntSLT, left, right, "cmptmp"),
            NSLTokenType.Less => _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOLT, EnsureDouble(left), EnsureDouble(right), "cmptmp"),
            NSLTokenType.LessEqual when isIntegerOp => _builder.BuildICmp(LLVMIntPredicate.LLVMIntSLE, left, right, "cmptmp"),
            NSLTokenType.LessEqual => _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOLE, EnsureDouble(left), EnsureDouble(right), "cmptmp"),
            NSLTokenType.Greater when isIntegerOp => _builder.BuildICmp(LLVMIntPredicate.LLVMIntSGT, left, right, "cmptmp"),
            NSLTokenType.Greater => _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOGT, EnsureDouble(left), EnsureDouble(right), "cmptmp"),
            NSLTokenType.GreaterEqual when isIntegerOp => _builder.BuildICmp(LLVMIntPredicate.LLVMIntSGE, left, right, "cmptmp"),
            NSLTokenType.GreaterEqual => _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOGE, EnsureDouble(left), EnsureDouble(right), "cmptmp"),
            NSLTokenType.Equal when isIntegerOp => _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, left, right, "cmptmp"),
            NSLTokenType.Equal => _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOEQ, EnsureDouble(left), EnsureDouble(right), "cmptmp"),
            NSLTokenType.NotEqual when isIntegerOp => _builder.BuildICmp(LLVMIntPredicate.LLVMIntNE, left, right, "cmptmp"),
            NSLTokenType.NotEqual => _builder.BuildFCmp(LLVMRealPredicate.LLVMRealONE, EnsureDouble(left), EnsureDouble(right), "cmptmp"),

            // Logical
            NSLTokenType.And => _builder.BuildAnd(EnsureBool(left), EnsureBool(right), "andtmp"),
            NSLTokenType.Or => _builder.BuildOr(EnsureBool(left), EnsureBool(right), "ortmp"),

            NSLTokenType.AtSign => GenerateMatMul(left, right),

            _ => throw new NotImplementedException($"Binary operator not implemented: {node.Operator}")
        };
    }

    /// <summary>
    /// Check if a value is an integer type
    /// </summary>
    private bool IsInteger(LLVMValueRef value)
    {
        return value.TypeOf.Kind == LLVMTypeKind.LLVMIntegerTypeKind && value.TypeOf.IntWidth > 1;
    }

    private LLVMValueRef GeneratePow(LLVMValueRef left, LLVMValueRef right)
    {
        // Call llvm.pow.f64
        var powFunc = GetOrDeclareIntrinsic("llvm.pow.f64", LLVMTypeRef.Double, new[] { LLVMTypeRef.Double, LLVMTypeRef.Double });
        return _builder.BuildCall2(
            LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double, LLVMTypeRef.Double }),
            powFunc,
            new[] { left, right },
            "powtmp"
        );
    }

    private LLVMValueRef GenerateMatMul(LLVMValueRef left, LLVMValueRef right)
    {
        // For now, treat as scalar multiply - real implementation would use vector intrinsics
        return _builder.BuildFMul(left, right, "matmultmp");
    }

    #endregion

    #region Unary Operations

    private LLVMValueRef GenerateUnaryOp(NSLUnaryOperationNode node)
    {
        var operand = Generate(node.Operand);

        return node.Operator switch
        {
            // Negation - integer or float
            NSLTokenType.Minus when IsInteger(operand) => _builder.BuildNeg(operand, "negtmp"),
            NSLTokenType.Minus => _builder.BuildFNeg(EnsureDouble(operand), "negtmp"),

            // Logical NOT
            NSLTokenType.Not => _builder.BuildNot(EnsureBool(operand), "nottmp"),

            // Bitwise NOT (complement)
            NSLTokenType.BitwiseNot => _builder.BuildNot(EnsureInt64(operand), "bnottmp"),

            _ => throw new NotImplementedException($"Unary operator not implemented: {node.Operator}")
        };
    }

    #endregion

    #region Function Generation

    private LLVMValueRef GenerateFunction(NSLFunctionNode node)
    {
        var funcType = _typeChecker.Check(node) as NSLFunctionType
            ?? throw new InvalidOperationException("Function did not type check to function type");

        var paramTypes = new LLVMTypeRef[node.Parameters.Count];
        for (int i = 0; i < node.Parameters.Count; i++)
        {
            paramTypes[i] = GetLLVMType(funcType.ParameterTypes[i]);
        }

        // Determine if this function returns a closure
        // We need to check if the body is or contains a lambda that captures variables
        var returnType = GetLLVMType(funcType.ReturnType);
        bool mayReturnClosure = funcType.ReturnType is NSLFunctionType;

        // For functions that may return closures, we need to generate the body first
        // to determine the actual return type, then create the function
        if (mayReturnClosure)
        {
            // Save state
            var savedFunction = _currentFunction;
            var savedValues = new Dictionary<string, LLVMValueRef>(_namedValues);
            var savedValueTypes = new Dictionary<string, LLVMTypeRef>(_namedValueTypes);

            // Temporarily set up parameter scope to generate the body correctly
            _namedValues.Clear();
            _namedValueTypes.Clear();

            // We need a placeholder function to generate lambdas
            var placeholderFuncType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, paramTypes);
            var placeholderFunc = _module.AddFunction($"__placeholder_{node.Name}", placeholderFuncType);
            var placeholderEntry = placeholderFunc.AppendBasicBlock("entry");
            _builder.PositionAtEnd(placeholderEntry);
            _currentFunction = placeholderFunc;

            // Set up parameters
            for (int i = 0; i < node.Parameters.Count; i++)
            {
                var param = placeholderFunc.GetParam((uint)i);
                var alloca = CreateEntryBlockAlloca(placeholderFunc, node.Parameters[i].Name, paramTypes[i]);
                _builder.BuildStore(param, alloca);
                _namedValues[node.Parameters[i].Name] = alloca;
                _namedValueTypes[node.Parameters[i].Name] = paramTypes[i];
            }

            // Generate body to determine actual return type
            var bodyValue = Generate(node.Body);

            // Check if the body returned a closure struct
            if (bodyValue.TypeOf.Kind == LLVMTypeKind.LLVMStructTypeKind &&
                _closureStructToFuncType.ContainsKey(bodyValue.TypeOf))
            {
                returnType = bodyValue.TypeOf;
            }

            // Delete the placeholder function
            placeholderFunc.DeleteFunction();

            // Restore state
            _currentFunction = savedFunction;
            _namedValues.Clear();
            _namedValueTypes.Clear();
            foreach (var kv in savedValues)
                _namedValues[kv.Key] = kv.Value;
            foreach (var kv in savedValueTypes)
                _namedValueTypes[kv.Key] = kv.Value;
        }

        var llvmFuncType = LLVMTypeRef.CreateFunction(returnType, paramTypes);

        // Rename user's "main" to "nsl_main" to avoid conflict with C-style main
        var llvmFuncName = node.Name == "main" ? "nsl_main" : node.Name;
        var function = _module.AddFunction(llvmFuncName, llvmFuncType);

        // IMPORTANT: Register function BEFORE generating body to enable recursive calls
        // Use original name for lookup so recursive calls still work
        _functions[node.Name] = function;
        _functionTypes[node.Name] = llvmFuncType;

        // Parameter naming is optional - skipping to avoid marshaling complexity
        // The parameters are tracked by position in our _namedValues dictionary

        var entry = function.AppendBasicBlock("entry");
        _builder.PositionAtEnd(entry);

        // Save current function
        var savedFunction2 = _currentFunction;
        var savedValues2 = new Dictionary<string, LLVMValueRef>(_namedValues);
        var savedValueTypes2 = new Dictionary<string, LLVMTypeRef>(_namedValueTypes);

        _currentFunction = function;
        _namedValues.Clear();
        _namedValueTypes.Clear();

        // Create allocas for parameters
        for (int i = 0; i < node.Parameters.Count; i++)
        {
            var param = function.GetParam((uint)i);
            var alloca = CreateEntryBlockAlloca(function, node.Parameters[i].Name, paramTypes[i]);
            _builder.BuildStore(param, alloca);
            _namedValues[node.Parameters[i].Name] = alloca;
            _namedValueTypes[node.Parameters[i].Name] = paramTypes[i];  // Track parameter types
        }

        // Generate body
        var bodyValue2 = Generate(node.Body);

        // Add return if not already present
        if (_builder.InsertBlock.Terminator.Handle == IntPtr.Zero)
        {
            if (funcType.ReturnType is NSLVoidType)
                _builder.BuildRetVoid();
            else
                _builder.BuildRet(EnsureType(bodyValue2, returnType));
        }

        // Restore
        _currentFunction = savedFunction2;
        _namedValues.Clear();
        _namedValueTypes.Clear();
        foreach (var kv in savedValues2)
            _namedValues[kv.Key] = kv.Value;
        foreach (var kv in savedValueTypes2)
            _namedValueTypes[kv.Key] = kv.Value;

        return function;
    }

    private LLVMValueRef GenerateFunctionCall(NSLFunctionCallNode node)
    {
        LLVMValueRef function;
        LLVMTypeRef funcType;
        string functionName = "";
        bool isClosure = false;
        LLVMValueRef? envPtr = null;

        if (node.Function is NSLIdentifierNode ident)
        {
            functionName = ident.Name;

            // Special handling for print - select overload based on argument type
            if ((functionName == "print" || functionName == "println") && node.Arguments.Count == 1)
            {
                var argValue = Generate(node.Arguments[0]);

                // Check if argument is a string (pointer type)
                if (argValue.TypeOf.Kind == LLVMTypeKind.LLVMPointerTypeKind)
                {
                    function = _functions["print_str"];
                    funcType = _functionTypes["print_str"];
                    _builder.BuildCall2(funcType, function, new[] { argValue }, "");
                    return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
                }
                // Check if argument is an integer (width > 1 to exclude booleans)
                else if (IsInteger(argValue))
                {
                    function = _functions["print_int"];
                    funcType = _functionTypes["print_int"];
                    var intArg = EnsureInt64(argValue);
                    _builder.BuildCall2(funcType, function, new[] { intArg }, "");
                    return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
                }
                else
                {
                    // Floating point - use print_double
                    function = _functions["print_double"];
                    funcType = _functionTypes["print_double"];
                    var doubleArg = EnsureDouble(argValue);
                    _builder.BuildCall2(funcType, function, new[] { doubleArg }, "");
                    return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
                }
            }

            // Check if this is a closure (local variable holding a closure struct)
            if (_namedValues.TryGetValue(functionName, out var closureAlloca) &&
                _namedValueTypes.TryGetValue(functionName, out var closureVarType))
            {
                // Load the closure struct
                var closureValue = _builder.BuildLoad2(closureVarType, closureAlloca, "closure_load");

                // Check if this is a struct type (closures are structs with func_ptr and env_ptr)
                if (closureVarType.Kind == LLVMTypeKind.LLVMStructTypeKind)
                {
                    isClosure = true;

                    // Extract function pointer (field 0)
                    function = _builder.BuildExtractValue(closureValue, 0, "closure_func");

                    // Extract environment pointer (field 1)
                    envPtr = _builder.BuildExtractValue(closureValue, 1, "closure_env");

                    // Look up the function type from our tracked closure types
                    if (_closureStructToFuncType.TryGetValue(closureVarType, out var trackedFuncType))
                    {
                        funcType = trackedFuncType;
                    }
                    else
                    {
                        throw new InvalidOperationException($"Unknown closure type for variable {functionName}");
                    }
                }
                else if (closureVarType.Kind == LLVMTypeKind.LLVMPointerTypeKind)
                {
                    // Plain function pointer (no captured variables) - look up from function types
                    function = closureValue;
                    if (!_functionTypes.TryGetValue(functionName, out funcType))
                    {
                        throw new InvalidOperationException($"Unknown function type for variable {functionName}");
                    }
                }
                else
                {
                    throw new InvalidOperationException($"Variable {functionName} is not callable");
                }
            }
            else if (_functions.TryGetValue(functionName, out function))
            {
                // Get the function type from our tracked types
                funcType = _functionTypes[functionName];
            }
            else
            {
                function = _module.GetNamedFunction(functionName);
                if (function.Handle == IntPtr.Zero)
                    throw new InvalidOperationException($"Unknown function: {functionName}");

                // For external functions, try to get from our dictionary or reconstruct
                if (!_functionTypes.TryGetValue(functionName, out funcType))
                    throw new InvalidOperationException($"Unknown function type for: {functionName}");
            }
        }
        else
        {
            // Handle indirect function calls (expression returning a function)
            var funcValue = Generate(node.Function);
            var funcValueType = funcValue.TypeOf;
            if (funcValueType.Kind == LLVMTypeKind.LLVMStructTypeKind)
            {
                // Closure struct
                isClosure = true;
                function = _builder.BuildExtractValue(funcValue, 0, "closure_func");
                envPtr = _builder.BuildExtractValue(funcValue, 1, "closure_env");

                // Look up the function type from our tracked closure types
                if (_closureStructToFuncType.TryGetValue(funcValueType, out var trackedFuncType))
                {
                    funcType = trackedFuncType;
                }
                else
                {
                    throw new InvalidOperationException("Unknown closure type for indirect call");
                }
            }
            else if (funcValueType.Kind == LLVMTypeKind.LLVMPointerTypeKind)
            {
                // Plain function pointer - this is typically a direct function reference
                function = funcValue;
                // For function pointers, we need the type from the TypeChecker
                var nslFuncType = _typeChecker.Check(node.Function) as NSLFunctionType
                    ?? throw new InvalidOperationException("Expression does not evaluate to a function");
                funcType = CreateFunctionLLVMType(nslFuncType);
            }
            else
            {
                throw new InvalidOperationException("Expression does not evaluate to a callable");
            }
        }

        // Build arguments list
        var argsList = new List<LLVMValueRef>();
        var paramTypes = funcType.ParamTypes;

        // If this is a closure, prepend the environment pointer
        if (isClosure && envPtr.HasValue)
        {
            // The env pointer in the closure struct is stored as i8*, but the function
            // expects a typed pointer to the environment struct. We need to bitcast it.
            var envPtrValue = envPtr.Value;
            if (paramTypes.Length > 0 && envPtrValue.TypeOf.Kind != paramTypes[0].Kind)
            {
                envPtrValue = _builder.BuildBitCast(envPtrValue, paramTypes[0], "env_cast");
            }
            argsList.Add(envPtrValue);
        }

        // Add regular arguments
        int paramOffset = isClosure ? 1 : 0;  // Skip env parameter when matching types

        for (int i = 0; i < node.Arguments.Count; i++)
        {
            var argValue = Generate(node.Arguments[i]);
            int paramIndex = i + paramOffset;
            if (paramIndex < paramTypes.Length)
            {
                argValue = EnsureType(argValue, paramTypes[paramIndex]);
            }
            argsList.Add(argValue);
        }

        var args = argsList.ToArray();

        if (funcType.ReturnType.Kind == LLVMTypeKind.LLVMVoidTypeKind)
        {
            _builder.BuildCall2(funcType, function, args, "");
            return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
        }

        return _builder.BuildCall2(funcType, function, args, "calltmp");
    }

    private LLVMValueRef GenerateReturn(NSLReturnNode node)
    {
        if (node.Value != null)
        {
            var value = Generate(node.Value);
            return _builder.BuildRet(value);
        }
        return _builder.BuildRetVoid();
    }

    private LLVMValueRef GenerateLambda(NSLLambdaNode node)
    {
        // Generate lambda as closure with captured variables
        var funcType = _typeChecker.Check(node) as NSLFunctionType
            ?? throw new InvalidOperationException("Lambda did not type check to function type");

        // Collect free variables (variables referenced but not defined as parameters)
        var paramNames = new HashSet<string>(node.Parameters.Select(p => p.Name));
        var freeVars = CollectFreeVariables(node.Body, paramNames);

        // Filter to only include variables that exist in the current scope
        var capturedVars = freeVars
            .Where(name => _namedValues.ContainsKey(name))
            .ToList();

        var lambdaName = $"__lambda_{Guid.NewGuid():N}";

        // Build parameter types - add environment pointer as first parameter if we have captures
        var paramTypes = new List<LLVMTypeRef>();
        LLVMTypeRef? envStructType = null;

        if (capturedVars.Count > 0)
        {
            // Create environment struct type
            var envFieldTypes = capturedVars
                .Select(name => _namedValueTypes.TryGetValue(name, out var t) ? t : LLVMTypeRef.Double)
                .ToArray();
            var envType = LLVMTypeRef.CreateStruct(envFieldTypes, false);
            envStructType = envType;
            _closureEnvTypes[lambdaName] = envType;
            _closureCapturedVars[lambdaName] = capturedVars;

            // Add environment pointer as first parameter
            paramTypes.Add(LLVMTypeRef.CreatePointer(envType, 0));
        }

        // Add regular parameters
        for (int i = 0; i < node.Parameters.Count; i++)
        {
            paramTypes.Add(GetLLVMType(funcType.ParameterTypes[i]));
        }

        var returnType = GetLLVMType(funcType.ReturnType);
        var llvmFuncType = LLVMTypeRef.CreateFunction(returnType, paramTypes.ToArray());

        var function = _module.AddFunction(lambdaName, llvmFuncType);
        _functions[lambdaName] = function;
        _functionTypes[lambdaName] = llvmFuncType;

        var entry = function.AppendBasicBlock("entry");

        var savedBuilder = _builder.InsertBlock;
        var savedFunction = _currentFunction;
        var savedValues = new Dictionary<string, LLVMValueRef>(_namedValues);
        var savedValueTypes = new Dictionary<string, LLVMTypeRef>(_namedValueTypes);
        var savedEnvPtr = _currentEnvPtr;

        _builder.PositionAtEnd(entry);
        _currentFunction = function;
        _namedValues.Clear();
        _namedValueTypes.Clear();

        int paramOffset = 0;

        // If we have captured variables, load them from the environment
        if (capturedVars.Count > 0 && envStructType.HasValue)
        {
            var envPtr = function.GetParam(0);
            _currentEnvPtr = envPtr;
            paramOffset = 1;

            // Load each captured variable from the environment struct
            for (int i = 0; i < capturedVars.Count; i++)
            {
                var varName = capturedVars[i];
                var varType = savedValueTypes.TryGetValue(varName, out var t) ? t : LLVMTypeRef.Double;

                // GEP into the environment struct
                var fieldPtr = _builder.BuildStructGEP2(envStructType.Value, envPtr, (uint)i, $"env_{varName}_ptr");
                var alloca = CreateEntryBlockAlloca(function, varName, varType);
                var loadedValue = _builder.BuildLoad2(varType, fieldPtr, $"env_{varName}");
                _builder.BuildStore(loadedValue, alloca);
                _namedValues[varName] = alloca;
                _namedValueTypes[varName] = varType;
            }
        }

        // Set up regular parameters
        for (int i = 0; i < node.Parameters.Count; i++)
        {
            var param = function.GetParam((uint)(i + paramOffset));
            var paramType = GetLLVMType(funcType.ParameterTypes[i]);
            var alloca = CreateEntryBlockAlloca(function, node.Parameters[i].Name, paramType);
            _builder.BuildStore(param, alloca);
            _namedValues[node.Parameters[i].Name] = alloca;
            _namedValueTypes[node.Parameters[i].Name] = paramType;
        }

        var bodyValue = Generate(node.Body);
        _builder.BuildRet(EnsureType(bodyValue, returnType));

        _builder.PositionAtEnd(savedBuilder);
        _currentFunction = savedFunction;
        _currentEnvPtr = savedEnvPtr;
        _namedValues.Clear();
        _namedValueTypes.Clear();
        foreach (var kv in savedValues)
            _namedValues[kv.Key] = kv.Value;
        foreach (var kv in savedValueTypes)
            _namedValueTypes[kv.Key] = kv.Value;

        // If we have captured variables, create and populate the environment struct
        if (capturedVars.Count > 0 && envStructType.HasValue)
        {
            // Allocate environment struct on the heap (closures may outlive their creating function)
            var envSize = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, (ulong)GetStructSize(envStructType.Value), false);
            var envMalloc = _builder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { envSize }, "env_malloc");
            var envAlloca = _builder.BuildBitCast(envMalloc, LLVMTypeRef.CreatePointer(envStructType.Value, 0), "closure_env");

            // Store captured values into the environment
            for (int i = 0; i < capturedVars.Count; i++)
            {
                var varName = capturedVars[i];
                var varType = savedValueTypes.TryGetValue(varName, out var t) ? t : LLVMTypeRef.Double;

                // Get the value from the saved scope
                if (savedValues.TryGetValue(varName, out var varAlloca))
                {
                    var value = _builder.BuildLoad2(varType, varAlloca, $"cap_{varName}");
                    var fieldPtr = _builder.BuildStructGEP2(envStructType.Value, envAlloca, (uint)i, $"env_field_{i}");
                    _builder.BuildStore(value, fieldPtr);
                }
            }

            // Create closure struct: { function_ptr, env_ptr }
            var closureType = GetClosureType(llvmFuncType);
            var closureAlloca = _builder.BuildAlloca(closureType, "closure");

            // Store function pointer
            var funcPtrField = _builder.BuildStructGEP2(closureType, closureAlloca, 0, "closure_func_ptr");
            _builder.BuildStore(function, funcPtrField);

            // Store environment pointer (bitcast to i8* for generic storage)
            var envPtrField = _builder.BuildStructGEP2(closureType, closureAlloca, 1, "closure_env_ptr");
            var envPtrAsI8 = _builder.BuildBitCast(envAlloca, LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0), "env_ptr_i8");
            _builder.BuildStore(envPtrAsI8, envPtrField);

            // Track the function type for this closure struct type
            _closureStructToFuncType[closureType] = llvmFuncType;

            // Return the closure struct value
            return _builder.BuildLoad2(closureType, closureAlloca, "closure_val");
        }

        // No captured variables - return just the function pointer
        return function;
    }

    /// <summary>
    /// Collect free variables from an AST node (variables referenced but not in the bound set)
    /// </summary>
    private HashSet<string> CollectFreeVariables(NSLASTNode node, HashSet<string> boundVars)
    {
        var freeVars = new HashSet<string>();
        CollectFreeVariablesRecursive(node, boundVars, freeVars);
        return freeVars;
    }

    private void CollectFreeVariablesRecursive(NSLASTNode node, HashSet<string> boundVars, HashSet<string> freeVars)
    {
        switch (node)
        {
            case NSLIdentifierNode ident:
                if (!boundVars.Contains(ident.Name) && !_functions.ContainsKey(ident.Name))
                {
                    freeVars.Add(ident.Name);
                }
                break;

            case NSLBinaryOperationNode binary:
                CollectFreeVariablesRecursive(binary.Left, boundVars, freeVars);
                CollectFreeVariablesRecursive(binary.Right, boundVars, freeVars);
                break;

            case NSLUnaryOperationNode unary:
                CollectFreeVariablesRecursive(unary.Operand, boundVars, freeVars);
                break;

            case NSLFunctionCallNode call:
                CollectFreeVariablesRecursive(call.Function, boundVars, freeVars);
                foreach (var arg in call.Arguments)
                    CollectFreeVariablesRecursive(arg, boundVars, freeVars);
                break;

            case NSLBlockNode block:
                var newBound = new HashSet<string>(boundVars);
                foreach (var stmt in block.Statements)
                {
                    if (stmt is NSLVariableDeclarationNode varDecl)
                    {
                        if (varDecl.Value != null)
                            CollectFreeVariablesRecursive(varDecl.Value, newBound, freeVars);
                        newBound.Add(varDecl.Name);
                    }
                    else
                    {
                        CollectFreeVariablesRecursive(stmt, newBound, freeVars);
                    }
                }
                break;

            case NSLVariableDeclarationNode varDecl:
                if (varDecl.Value != null)
                    CollectFreeVariablesRecursive(varDecl.Value, boundVars, freeVars);
                break;

            case NSLIfNode ifNode:
                CollectFreeVariablesRecursive(ifNode.Condition, boundVars, freeVars);
                CollectFreeVariablesRecursive(ifNode.ThenBranch, boundVars, freeVars);
                if (ifNode.ElseBranch != null)
                    CollectFreeVariablesRecursive(ifNode.ElseBranch, boundVars, freeVars);
                break;

            case NSLWhileNode whileNode:
                CollectFreeVariablesRecursive(whileNode.Condition, boundVars, freeVars);
                CollectFreeVariablesRecursive(whileNode.Body, boundVars, freeVars);
                break;

            case NSLForNode forNode:
                CollectFreeVariablesRecursive(forNode.Iterable, boundVars, freeVars);
                var forBound = new HashSet<string>(boundVars) { forNode.Variable.Value };
                CollectFreeVariablesRecursive(forNode.Body, forBound, freeVars);
                break;

            case NSLReturnNode ret:
                if (ret.Value != null)
                    CollectFreeVariablesRecursive(ret.Value, boundVars, freeVars);
                break;

            case NSLArrayNode array:
                foreach (var elem in array.Elements)
                    CollectFreeVariablesRecursive(elem, boundVars, freeVars);
                break;

            case NSLIndexNode index:
                CollectFreeVariablesRecursive(index.Object, boundVars, freeVars);
                CollectFreeVariablesRecursive(index.Index, boundVars, freeVars);
                break;

            case NSLLambdaNode lambda:
                // Nested lambda - add its parameters to bound vars
                var lambdaBound = new HashSet<string>(boundVars);
                foreach (var param in lambda.Parameters)
                    lambdaBound.Add(param.Name);
                CollectFreeVariablesRecursive(lambda.Body, lambdaBound, freeVars);
                break;

            case NSLMatchNode match:
                CollectFreeVariablesRecursive(match.Value, boundVars, freeVars);
                foreach (var caseNode in match.Cases)
                {
                    // Pattern variables are bound in the case body
                    var caseBound = new HashSet<string>(boundVars);
                    CollectPatternBoundVars(caseNode.Pattern, caseBound);
                    CollectFreeVariablesRecursive(caseNode.Body, caseBound, freeVars);
                }
                break;

            case NSLAssignmentNode assign:
                CollectFreeVariablesRecursive(assign.Value, boundVars, freeVars);
                if (!boundVars.Contains(assign.Name))
                    freeVars.Add(assign.Name);
                break;

            case NSLGetNode get:
                CollectFreeVariablesRecursive(get.Object, boundVars, freeVars);
                break;

            // Literal nodes don't reference variables
            case NSLLiteralNode:
            case NSLBreakNode:
            case NSLContinueNode:
                break;

            default:
                // For other node types, try to handle generically
                break;
        }
    }

    private void CollectPatternBoundVars(NSLASTNode pattern, HashSet<string> boundVars)
    {
        switch (pattern)
        {
            case NSLIdentifierNode ident when !IsPatternKeyword(ident.Name):
                boundVars.Add(ident.Name);
                break;
            case NSLResultNode result:
                if (result.Value is NSLIdentifierNode innerIdent)
                    boundVars.Add(innerIdent.Name);
                break;
            case NSLOptionalNode opt:
                if (opt.HasValue && opt.Value is NSLIdentifierNode optIdent)
                    boundVars.Add(optIdent.Name);
                break;
        }
    }

    private bool IsPatternKeyword(string name) =>
        name == "none" || name == "true" || name == "false" || name == "null";

    /// <summary>
    /// Get or create a closure type for a given function type
    /// Closure struct: { function_ptr, env_ptr (i8*) }
    /// </summary>
    private LLVMTypeRef GetClosureType(LLVMTypeRef funcType)
    {
        var funcPtrType = LLVMTypeRef.CreatePointer(funcType, 0);
        var envPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        return LLVMTypeRef.CreateStruct(new[] { funcPtrType, envPtrType }, false);
    }

    #endregion

    #region Control Flow

    private LLVMValueRef GenerateIf(NSLIfNode node)
    {
        var condition = Generate(node.Condition);
        condition = EnsureBool(condition);

        var thenBlock = _currentFunction.AppendBasicBlock("then");
        var elseBlock = _currentFunction.AppendBasicBlock("else");
        var mergeBlock = _currentFunction.AppendBasicBlock("ifcont");

        _builder.BuildCondBr(condition, thenBlock, elseBlock);

        // Then branch
        _builder.PositionAtEnd(thenBlock);
        var thenValue = Generate(node.ThenBranch);
        bool thenTerminated = IsBlockTerminated(_builder.InsertBlock);
        if (!thenTerminated)
        {
            _builder.BuildBr(mergeBlock);
        }
        thenBlock = _builder.InsertBlock; // Update for PHI

        // Else branch
        _builder.PositionAtEnd(elseBlock);
        LLVMValueRef elseValue;
        if (node.ElseBranch != null)
        {
            elseValue = Generate(node.ElseBranch);
        }
        else
        {
            elseValue = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0);
        }
        bool elseTerminated = IsBlockTerminated(_builder.InsertBlock);
        if (!elseTerminated)
        {
            _builder.BuildBr(mergeBlock);
        }
        elseBlock = _builder.InsertBlock;

        // Merge - only needed if at least one branch doesn't terminate
        if (!thenTerminated || !elseTerminated)
        {
            _builder.PositionAtEnd(mergeBlock);

            // Create PHI node only if both branches flow to merge and return values
            if (!thenTerminated && !elseTerminated)
            {
                var resultType = thenValue.TypeOf;
                if (resultType.Kind != LLVMTypeKind.LLVMVoidTypeKind)
                {
                    var phi = _builder.BuildPhi(resultType, "iftmp");
                    phi.AddIncoming(new[] { thenValue }, new[] { thenBlock }, 1);
                    phi.AddIncoming(new[] { EnsureType(elseValue, resultType) }, new[] { elseBlock }, 1);
                    return phi;
                }
            }
        }
        else
        {
            // Both branches terminate, merge block is unused
            // LLVM optimizer will remove dead blocks
            _builder.PositionAtEnd(mergeBlock);
            _builder.BuildUnreachable();
        }

        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    /// <summary>
    /// Check if a basic block is already terminated (has a return or branch)
    /// </summary>
    private bool IsBlockTerminated(LLVMBasicBlockRef block)
    {
        var terminator = block.Terminator;
        return terminator.Handle != IntPtr.Zero;
    }

    private LLVMValueRef GenerateWhile(NSLWhileNode node)
    {
        var condBlock = _currentFunction.AppendBasicBlock("while.cond");
        var bodyBlock = _currentFunction.AppendBasicBlock("while.body");
        var exitBlock = _currentFunction.AppendBasicBlock("while.exit");

        _builder.BuildBr(condBlock);

        // Condition
        _builder.PositionAtEnd(condBlock);
        var condition = Generate(node.Condition);
        condition = EnsureBool(condition);
        _builder.BuildCondBr(condition, bodyBlock, exitBlock);

        // Body
        _builder.PositionAtEnd(bodyBlock);
        Generate(node.Body);
        // Only branch back to condition if not already terminated (e.g., by return)
        if (!IsBlockTerminated(_builder.InsertBlock))
        {
            _builder.BuildBr(condBlock);
        }

        // Exit
        _builder.PositionAtEnd(exitBlock);
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    private LLVMValueRef GenerateFor(NSLForNode node)
    {
        // For now, implement a simple counted loop for ranges
        var loopVar = CreateEntryBlockAlloca(_currentFunction, node.Variable.Value, LLVMTypeRef.Double);
        _namedValues[node.Variable.Value] = loopVar;
        _namedValueTypes[node.Variable.Value] = LLVMTypeRef.Double;

        // Get range bounds from iterable
        LLVMValueRef startVal, endVal;
        bool isInclusive = false;

        if (node.Iterable is NSLRangeNode range)
        {
            startVal = range.Start != null ? Generate(range.Start) : LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0);
            endVal = range.End != null ? Generate(range.End) : LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 10);
            isInclusive = range.IsInclusive;
            startVal = EnsureDouble(startVal);
            endVal = EnsureDouble(endVal);
        }
        else
        {
            // Fallback: generate iterable and assume it's a length
            var iterable = Generate(node.Iterable);
            startVal = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0);
            endVal = EnsureDouble(iterable);
        }

        // Initialize loop variable to start
        _builder.BuildStore(startVal, loopVar);

        var condBlock = _currentFunction.AppendBasicBlock("for.cond");
        var bodyBlock = _currentFunction.AppendBasicBlock("for.body");
        var incBlock = _currentFunction.AppendBasicBlock("for.inc");
        var exitBlock = _currentFunction.AppendBasicBlock("for.exit");

        // Push loop context for break/continue
        _loopStack.Push((exitBlock, incBlock));

        _builder.BuildBr(condBlock);

        // Condition
        _builder.PositionAtEnd(condBlock);
        var currentVal = _builder.BuildLoad2(LLVMTypeRef.Double, loopVar, "i");
        LLVMValueRef cond;
        if (isInclusive)
        {
            cond = _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOLE, currentVal, endVal, "forcond");
        }
        else
        {
            cond = _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOLT, currentVal, endVal, "forcond");
        }
        _builder.BuildCondBr(cond, bodyBlock, exitBlock);

        // Body
        _builder.PositionAtEnd(bodyBlock);
        Generate(node.Body);
        // Only branch to increment if not already terminated
        if (!IsBlockTerminated(_builder.InsertBlock))
        {
            _builder.BuildBr(incBlock);
        }

        // Increment
        _builder.PositionAtEnd(incBlock);
        var loadedVal = _builder.BuildLoad2(LLVMTypeRef.Double, loopVar, "curval");
        var nextVal = _builder.BuildFAdd(loadedVal, LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1), "nextvar");
        _builder.BuildStore(nextVal, loopVar);
        _builder.BuildBr(condBlock);

        // Pop loop context
        _loopStack.Pop();

        // Exit
        _builder.PositionAtEnd(exitBlock);
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    private LLVMValueRef GenerateBreak()
    {
        if (_loopStack.Count == 0)
            throw new InvalidOperationException("'break' statement outside of loop");
        
        var (exitBlock, _) = _loopStack.Peek();
        _builder.BuildBr(exitBlock);
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    private LLVMValueRef GenerateContinue()
    {
        if (_loopStack.Count == 0)
            throw new InvalidOperationException("'continue' statement outside of loop");
        
        var (_, continueBlock) = _loopStack.Peek();
        _builder.BuildBr(continueBlock);
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    #endregion

    #region Collection Generation

    private LLVMValueRef GenerateBlock(NSLBlockNode node)
    {
        LLVMValueRef lastValue = LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
        foreach (var stmt in node.Statements)
        {
            lastValue = Generate(stmt);
        }
        return lastValue;
    }

    private LLVMValueRef GenerateArray(NSLArrayNode node)
    {
        // Allocate array on stack
        var elementType = LLVMTypeRef.Double;
        var arrayType = LLVMTypeRef.CreateArray(elementType, (uint)node.Elements.Count);
        var alloca = _builder.BuildAlloca(arrayType, "array");

        // Store elements
        for (int i = 0; i < node.Elements.Count; i++)
        {
            var value = Generate(node.Elements[i]);
            value = EnsureDouble(value);
            var indices = new[]
            {
                LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, 0),
                LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, (ulong)i)
            };
            var ptr = _builder.BuildGEP2(arrayType, alloca, indices, $"elem{i}");
            _builder.BuildStore(value, ptr);
        }

        return alloca;
    }

    /// <summary>
    /// Generate list comprehension: [expr for var in iterable] or [expr for var in iterable if cond]
    /// </summary>
    private LLVMValueRef GenerateListComprehension(NSLListComprehensionNode node)
    {
        var function = _currentFunction;
        var elementType = LLVMTypeRef.Double;

        // For simplicity, we'll handle range-based comprehensions specially
        // [x * x for x in range(1, 10)] style
        // Check this FIRST before generating the iterable
        if (node.Iterable is NSLFunctionCallNode funcCall &&
            funcCall.Function is NSLIdentifierNode funcIdent &&
            funcIdent.Name == "range")
        {
            return GenerateRangeComprehension(node, funcCall);
        }

        // For array-based comprehensions, generate the iterable
        var iterable = Generate(node.Iterable);

        // For array-based comprehensions, determine array size and iterate
        // This requires runtime knowledge of array size, which is complex
        // For now, return a placeholder - arrays would need length tracking
        return iterable;
    }

    /// <summary>
    /// Generate list comprehension over a range: [expr for x in range(start, end)]
    /// </summary>
    private LLVMValueRef GenerateRangeComprehension(NSLListComprehensionNode node, NSLFunctionCallNode rangeCall)
    {
        var function = _currentFunction;
        var elementType = LLVMTypeRef.Double;

        // Get range bounds
        var startValue = Generate(rangeCall.Arguments[0]);
        var endValue = rangeCall.Arguments.Count > 1 ? Generate(rangeCall.Arguments[1]) : startValue;

        if (rangeCall.Arguments.Count == 1)
        {
            // range(n) means 0..n
            startValue = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0);
        }

        startValue = EnsureDouble(startValue);
        endValue = EnsureDouble(endValue);

        // Calculate array size: (end - start)
        var sizeDouble = _builder.BuildFSub(endValue, startValue, "range_size_f");
        var sizeInt = _builder.BuildFPToSI(sizeDouble, LLVMTypeRef.Int64, "range_size");

        // Allocate result array on heap (size not known at compile time for filtered)
        // For unfiltered, we know the size
        if (node.Condition == null)
        {
            // Allocate fixed-size result array
            var allocSize = _builder.BuildMul(sizeInt, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 8), "alloc_size");
            var resultPtr = _builder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result_ptr");
            var resultArray = _builder.BuildBitCast(resultPtr, LLVMTypeRef.CreatePointer(elementType, 0), "result");

            // Create loop blocks
            var loopHeader = function.AppendBasicBlock("listcomp.header");
            var loopBody = function.AppendBasicBlock("listcomp.body");
            var loopEnd = function.AppendBasicBlock("listcomp.end");

            // Initialize loop counter
            var iAlloca = CreateEntryBlockAlloca(function, "listcomp.i", LLVMTypeRef.Double);
            _builder.BuildStore(startValue, iAlloca);

            // Also need an index for storing results
            var idxAlloca = CreateEntryBlockAlloca(function, "listcomp.idx", LLVMTypeRef.Int64);
            _builder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0), idxAlloca);

            _builder.BuildBr(loopHeader);

            // Loop header: check if i < end
            _builder.PositionAtEnd(loopHeader);
            var currentI = _builder.BuildLoad2(LLVMTypeRef.Double, iAlloca, "i");
            var cond = _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOLT, currentI, endValue, "listcomp.cond");
            _builder.BuildCondBr(cond, loopBody, loopEnd);

            // Loop body: evaluate expression and store
            _builder.PositionAtEnd(loopBody);

            // Bind loop variable
            var savedValue = _namedValues.TryGetValue(node.Variable, out var oldValue) ? oldValue : default;
            var savedType = _namedValueTypes.TryGetValue(node.Variable, out var oldType) ? oldType : default;
            _namedValues[node.Variable] = iAlloca;
            _namedValueTypes[node.Variable] = LLVMTypeRef.Double;

            // Generate expression
            var exprValue = Generate(node.Expression);
            exprValue = EnsureDouble(exprValue);

            // Store result at current index
            var currentIdx = _builder.BuildLoad2(LLVMTypeRef.Int64, idxAlloca, "idx");
            var elemPtr = _builder.BuildGEP2(elementType, resultArray, new[] { currentIdx }, "elem_ptr");
            _builder.BuildStore(exprValue, elemPtr);

            // Restore variable scope
            if (savedValue.Handle != IntPtr.Zero)
            {
                _namedValues[node.Variable] = savedValue;
                _namedValueTypes[node.Variable] = savedType;
            }
            else
            {
                _namedValues.Remove(node.Variable);
                _namedValueTypes.Remove(node.Variable);
            }

            // Increment loop counter and index
            var nextI = _builder.BuildFAdd(currentI, LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1.0), "nexti");
            _builder.BuildStore(nextI, iAlloca);
            var nextIdx = _builder.BuildAdd(currentIdx, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "nextidx");
            _builder.BuildStore(nextIdx, idxAlloca);

            _builder.BuildBr(loopHeader);

            // End block
            _builder.PositionAtEnd(loopEnd);

            return resultArray;
        }
        else
        {
            // With condition - more complex, need to count matches first or use dynamic approach
            // For now, use a simpler approach: allocate max size, then track actual count
            var maxSize = sizeInt;
            var allocSize = _builder.BuildMul(maxSize, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 8), "alloc_size");
            var resultPtr = _builder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result_ptr");
            var resultArray = _builder.BuildBitCast(resultPtr, LLVMTypeRef.CreatePointer(elementType, 0), "result");

            // Create loop blocks
            var loopHeader = function.AppendBasicBlock("listcomp.header");
            var loopBody = function.AppendBasicBlock("listcomp.body");
            var condTrue = function.AppendBasicBlock("listcomp.condtrue");
            var loopNext = function.AppendBasicBlock("listcomp.next");
            var loopEnd = function.AppendBasicBlock("listcomp.end");

            // Initialize loop counter and result index
            var iAlloca = CreateEntryBlockAlloca(function, "listcomp.i", LLVMTypeRef.Double);
            _builder.BuildStore(startValue, iAlloca);
            var idxAlloca = CreateEntryBlockAlloca(function, "listcomp.idx", LLVMTypeRef.Int64);
            _builder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0), idxAlloca);

            _builder.BuildBr(loopHeader);

            // Loop header
            _builder.PositionAtEnd(loopHeader);
            var currentI = _builder.BuildLoad2(LLVMTypeRef.Double, iAlloca, "i");
            var loopCond = _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOLT, currentI, endValue, "listcomp.cond");
            _builder.BuildCondBr(loopCond, loopBody, loopEnd);

            // Loop body: evaluate condition
            _builder.PositionAtEnd(loopBody);

            // Bind loop variable
            var savedValue = _namedValues.TryGetValue(node.Variable, out var oldValue) ? oldValue : default;
            var savedType = _namedValueTypes.TryGetValue(node.Variable, out var oldType) ? oldType : default;
            _namedValues[node.Variable] = iAlloca;
            _namedValueTypes[node.Variable] = LLVMTypeRef.Double;

            // Evaluate filter condition
            var filterCond = Generate(node.Condition);
            filterCond = EnsureBool(filterCond);
            _builder.BuildCondBr(filterCond, condTrue, loopNext);

            // Condition true: add to result
            _builder.PositionAtEnd(condTrue);
            var exprValue = Generate(node.Expression);
            exprValue = EnsureDouble(exprValue);

            var currentIdx = _builder.BuildLoad2(LLVMTypeRef.Int64, idxAlloca, "idx");
            var elemPtr = _builder.BuildGEP2(elementType, resultArray, new[] { currentIdx }, "elem_ptr");
            _builder.BuildStore(exprValue, elemPtr);

            var nextIdx = _builder.BuildAdd(currentIdx, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "nextidx");
            _builder.BuildStore(nextIdx, idxAlloca);
            _builder.BuildBr(loopNext);

            // Next iteration
            _builder.PositionAtEnd(loopNext);

            // Restore variable scope
            if (savedValue.Handle != IntPtr.Zero)
            {
                _namedValues[node.Variable] = savedValue;
                _namedValueTypes[node.Variable] = savedType;
            }
            else
            {
                _namedValues.Remove(node.Variable);
                _namedValueTypes.Remove(node.Variable);
            }

            var nextI = _builder.BuildFAdd(currentI, LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1.0), "nexti");
            _builder.BuildStore(nextI, iAlloca);
            _builder.BuildBr(loopHeader);

            // End block
            _builder.PositionAtEnd(loopEnd);

            return resultArray;
        }
    }

    private LLVMValueRef GenerateIndex(NSLIndexNode node)
    {
        var array = Generate(node.Object);
        var index = Generate(node.Index);

        // Convert index to i64 if needed
        index = EnsureInt64(index);

        // Get the array element type - for now assume double arrays
        var elementType = LLVMTypeRef.Double;

        // Check if this is an identifier that we have array info for
        if (node.Object is NSLIdentifierNode ident && _arrayInfo.TryGetValue(ident.Name, out var info))
        {
            elementType = info.ElementType;
        }

        // If array is a pointer to an array, we need to use GEP
        var indices = new[]
        {
            LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0),
            index
        };

        // Get pointer to element and load
        var arrayType = LLVMTypeRef.CreateArray(elementType, 0); // Size doesn't matter for GEP
        var ptr = _builder.BuildGEP2(arrayType, array, indices, "elemptr");
        return _builder.BuildLoad2(elementType, ptr, "elem");
    }

    private LLVMValueRef GeneratePipeline(NSLPipelineNode node)
    {
        // left |> right means right(left)
        var leftValue = Generate(node.Left);

        if (node.Right is NSLIdentifierNode funcName)
        {
            if (_functions.TryGetValue(funcName.Name, out var func) &&
                _functionTypes.TryGetValue(funcName.Name, out var funcType))
            {
                return _builder.BuildCall2(funcType, func, new[] { leftValue }, "pipetmp");
            }
        }

        throw new NotImplementedException("Complex pipeline targets not yet supported");
    }

    private LLVMValueRef GenerateRange(NSLRangeNode node)
    {
        // For now, return the start value - real implementation would create range object
        if (node.Start != null)
            return Generate(node.Start);
        return LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0);
    }

    #endregion

    #region Module System Generation

    /// <summary>
    /// Generate import statement - no-op at codegen level
    /// Imports are handled at the type checking stage
    /// </summary>
    private LLVMValueRef GenerateImport()
    {
        // Imports are resolved at compile time
        // No runtime code needed
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    /// <summary>
    /// Generate export statement - generates the declaration
    /// </summary>
    private LLVMValueRef GenerateExport(NSLExportNode node)
    {
        // If this is a pub declaration, generate the declaration
        if (node.Declaration != null)
        {
            return Generate(node.Declaration);
        }

        // Export lists are compile-time only
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    /// <summary>
    /// Generate module declaration - generates module body
    /// </summary>
    private LLVMValueRef GenerateModule(NSLModuleNode node)
    {
        // Generate all statements in the module body
        // In a full implementation, would prefix with module namespace
        return GenerateBlock(node.Body);
    }

    #endregion

    #region Struct Generation

    /// <summary>
    /// Map of struct names to their field information (name -> (index, type))
    /// </summary>
    private readonly Dictionary<string, Dictionary<string, (int Index, LLVMTypeRef Type)>> _structFieldInfo = new();

    /// <summary>
    /// Map of variable names to their struct type name (for field access)
    /// </summary>
    private readonly Dictionary<string, string> _variableStructTypes = new();

    /// <summary>
    /// Generate struct definition - creates LLVM struct type
    /// </summary>
    private LLVMValueRef GenerateStructDefinition(NSLStructNode node)
    {
        // Collect field types
        var fieldTypes = new List<LLVMTypeRef>();
        var fieldInfo = new Dictionary<string, (int Index, LLVMTypeRef Type)>();

        int index = 0;
        foreach (var field in node.Fields)
        {
            var fieldType = GetLLVMTypeFromName(field.TypeName);
            fieldTypes.Add(fieldType);
            fieldInfo[field.Name] = (index, fieldType);
            index++;
        }

        // Create the struct type
        var structType = _context.CreateNamedStruct(node.Name);
        structType.StructSetBody(fieldTypes.ToArray(), false);

        // Store struct information
        _structTypes[node.Name] = structType;
        _structFieldInfo[node.Name] = fieldInfo;

        // Return null - struct definitions don't produce values
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    /// <summary>
    /// Generate struct instantiation - allocate and initialize struct
    /// </summary>
    private LLVMValueRef GenerateStructInstantiation(NSLStructInstantiationNode node)
    {
        if (!_structTypes.TryGetValue(node.StructName, out var structType))
        {
            throw new InvalidOperationException($"Unknown struct type: {node.StructName}");
        }

        if (!_structFieldInfo.TryGetValue(node.StructName, out var fieldInfo))
        {
            throw new InvalidOperationException($"No field info for struct: {node.StructName}");
        }

        // Allocate the struct on the stack
        var structAlloca = _builder.BuildAlloca(structType, node.StructName + "_instance");

        // Initialize each field
        foreach (var field in node.Fields)
        {
            if (!fieldInfo.TryGetValue(field.Key, out var info))
            {
                throw new InvalidOperationException($"Unknown field: {field.Key} in struct {node.StructName}");
            }

            var fieldValue = Generate(field.Value);
            var fieldPtr = _builder.BuildStructGEP2(structType, structAlloca, (uint)info.Index, field.Key + "_ptr");
            _builder.BuildStore(fieldValue, fieldPtr);
        }

        // Return the struct by loading it from the alloca
        return _builder.BuildLoad2(structType, structAlloca, node.StructName + "_value");
    }

    /// <summary>
    /// Generate field access (obj.field)
    /// </summary>
    private LLVMValueRef GenerateFieldAccess(NSLGetNode node)
    {
        // If the object is a struct identifier, we need to handle it specially
        if (node.Object is NSLIdentifierNode ident)
        {
            // Check if this is a struct variable using our tracking dictionary
            if (_variableStructTypes.TryGetValue(ident.Name, out var structName) &&
                _namedValues.TryGetValue(ident.Name, out var allocaVal) &&
                _structFieldInfo.TryGetValue(structName, out var fieldInfo) &&
                fieldInfo.TryGetValue(node.Name, out var info))
            {
                var structType = _structTypes[structName];
                var fieldPtr = _builder.BuildStructGEP2(structType, allocaVal, (uint)info.Index, node.Name + "_ptr");
                return _builder.BuildLoad2(info.Type, fieldPtr, node.Name);
            }
        }

        // Get the object value for non-identifier cases
        var obj = Generate(node.Object);

        // For direct struct values (not allocas), use ExtractValue
        var objType = obj.TypeOf;
        if (objType.Kind == LLVMTypeKind.LLVMStructTypeKind)
        {
            // Find the struct info by type
            foreach (var (structName, structType) in _structTypes)
            {
                if (structType.Handle == objType.Handle || structType == objType)
                {
                    if (_structFieldInfo.TryGetValue(structName, out var fieldInfo) &&
                        fieldInfo.TryGetValue(node.Name, out var info))
                    {
                        // Extract value from struct
                        return _builder.BuildExtractValue(obj, (uint)info.Index, node.Name);
                    }
                }
            }
        }

        throw new InvalidOperationException($"Cannot access field '{node.Name}' - object is not a known struct type");
    }

    /// <summary>
    /// Get LLVM type from NSL type name string
    /// </summary>
    private LLVMTypeRef GetLLVMTypeFromName(string typeName)
    {
        return typeName.ToLower() switch
        {
            "number" => LLVMTypeRef.Double,
            "int" => LLVMTypeRef.Int64,
            "bool" => LLVMTypeRef.Int1,
            "string" => LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0),
            _ when _structTypes.TryGetValue(typeName, out var structType) => structType,
            _ => LLVMTypeRef.Double // Default to double for unknown types
        };
    }

    #endregion

    #region Enum (Algebraic Data Type) Generation

    // Enum type storage: maps enum name to its LLVM struct type
    private readonly Dictionary<string, LLVMTypeRef> _enumTypes = new();
    // Maps enum name to variant info: (variant name -> (tag value, field count))
    private readonly Dictionary<string, Dictionary<string, (int Tag, int FieldCount)>> _enumVariantInfo = new();
    // Maps enum name to the maximum size needed for variant data
    private readonly Dictionary<string, int> _enumMaxDataSize = new();

    /// <summary>
    /// Generate enum definition - creates tagged union LLVM type
    /// Enums are represented as: { i8 tag, [maxSize x double] data }
    /// </summary>
    private LLVMValueRef GenerateEnumDefinition(NSLEnumNode node)
    {
        var variantInfo = new Dictionary<string, (int Tag, int FieldCount)>();
        int maxFields = 0;

        // Process each variant to determine tag and field count
        int tagValue = 0;
        foreach (var variant in node.Variants)
        {
            int fieldCount = variant.Fields?.Count ?? 0;
            variantInfo[variant.Name] = (tagValue, fieldCount);
            maxFields = Math.Max(maxFields, fieldCount);
            tagValue++;
        }

        _enumVariantInfo[node.Name] = variantInfo;
        _enumMaxDataSize[node.Name] = maxFields;

        // Create enum struct type: { i8 tag, [maxFields x double] data }
        // Using doubles for all fields for simplicity (can be extended for type-specific storage)
        var dataArrayType = LLVMTypeRef.CreateArray(LLVMTypeRef.Double, (uint)Math.Max(1, maxFields));
        var enumType = _context.CreateNamedStruct($"enum.{node.Name}");
        enumType.StructSetBody(new[] { LLVMTypeRef.Int8, dataArrayType }, false);

        _enumTypes[node.Name] = enumType;

        // Return null - enum definitions don't produce values
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    /// <summary>
    /// Generate enum variant instantiation
    /// Example: Color::Red or Shape::Circle(5.0)
    /// </summary>
    private LLVMValueRef GenerateEnumVariant(NSLEnumVariantNode node)
    {
        if (!_enumTypes.TryGetValue(node.EnumName, out var enumType))
        {
            throw new InvalidOperationException($"Unknown enum type: {node.EnumName}");
        }

        if (!_enumVariantInfo.TryGetValue(node.EnumName, out var variantInfo) ||
            !variantInfo.TryGetValue(node.VariantName, out var info))
        {
            throw new InvalidOperationException($"Unknown variant: {node.EnumName}::{node.VariantName}");
        }

        int maxDataSize = _enumMaxDataSize[node.EnumName];

        // Allocate the enum on the stack
        var enumAlloca = _builder.BuildAlloca(enumType, $"{node.EnumName}_{node.VariantName}_inst");

        // Store the tag value
        var tagPtr = _builder.BuildStructGEP2(enumType, enumAlloca, 0, "tag_ptr");
        var tagVal = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, (ulong)info.Tag, false);
        _builder.BuildStore(tagVal, tagPtr);

        // Store variant arguments in the data array
        if (node.Arguments.Count > 0)
        {
            var dataArrayType = LLVMTypeRef.CreateArray(LLVMTypeRef.Double, (uint)Math.Max(1, maxDataSize));
            var dataPtr = _builder.BuildStructGEP2(enumType, enumAlloca, 1, "data_ptr");

            for (int i = 0; i < node.Arguments.Count; i++)
            {
                var argValue = Generate(node.Arguments[i]);

                // Convert to double if needed
                if (argValue.TypeOf != LLVMTypeRef.Double)
                {
                    if (argValue.TypeOf == LLVMTypeRef.Int64)
                        argValue = _builder.BuildSIToFP(argValue, LLVMTypeRef.Double, "to_double");
                    else if (argValue.TypeOf == LLVMTypeRef.Int1)
                        argValue = _builder.BuildUIToFP(argValue, LLVMTypeRef.Double, "to_double");
                }

                // Get pointer to the array element
                var indices = new LLVMValueRef[]
                {
                    LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0, false),
                    LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, (ulong)i, false)
                };
                var elemPtr = _builder.BuildInBoundsGEP2(dataArrayType, dataPtr, indices, $"data_{i}_ptr");
                _builder.BuildStore(argValue, elemPtr);
            }
        }

        // Return the enum by loading it from the alloca
        return _builder.BuildLoad2(enumType, enumAlloca, $"{node.EnumName}_{node.VariantName}_value");
    }

    #endregion

    #region Trait/Interface Generation

    // Trait storage: maps trait name to its method signatures
    private readonly Dictionary<string, Dictionary<string, LLVMTypeRef>> _traitMethods = new();
    // Implementation storage: maps (traitName, typeName) to implemented method names
    private readonly Dictionary<(string TraitName, string TypeName), HashSet<string>> _traitImplementations = new();

    /// <summary>
    /// Generate trait definition - records trait method signatures
    /// Traits are purely type-level, no runtime representation needed
    /// </summary>
    private LLVMValueRef GenerateTraitDefinition(NSLTraitNode node)
    {
        var methods = new Dictionary<string, LLVMTypeRef>();

        foreach (var method in node.Methods)
        {
            // Build the function type for this method signature
            var paramTypes = new List<LLVMTypeRef>();
            foreach (var param in method.Parameters)
            {
                var paramType = GetLLVMTypeFromName(param.Type ?? "any");
                paramTypes.Add(paramType);
            }

            var returnType = method.ReturnType != null ? GetLLVMTypeFromName(method.ReturnType) : LLVMTypeRef.Void;
            var funcType = LLVMTypeRef.CreateFunction(returnType, paramTypes.ToArray(), false);
            methods[method.Name] = funcType;
        }

        _traitMethods[node.Name] = methods;

        // Return null - trait definitions don't produce runtime values
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    /// <summary>
    /// Generate trait implementation - generates methods with mangled names
    /// </summary>
    private LLVMValueRef GenerateImplDefinition(NSLImplNode node)
    {
        var implKey = (node.TraitName, node.TypeName);
        if (!_traitImplementations.ContainsKey(implKey))
        {
            _traitImplementations[implKey] = new HashSet<string>();
        }

        foreach (var method in node.Methods)
        {
            // Generate the method with a mangled name: TypeName_TraitName_MethodName
            var mangledName = $"{node.TypeName}_{node.TraitName}_{method.Name}";

            // Create a copy of the method node with the mangled name
            var mangledMethod = new NSLFunctionNode(
                mangledName,
                method.Parameters,
                method.Body,
                method.ReturnType
            );

            // Generate the function
            GenerateFunction(mangledMethod);

            // Track that this method is implemented
            _traitImplementations[implKey].Add(method.Name);
        }

        // Return null - impl blocks don't produce values
        return LLVMValueRef.CreateConstNull(LLVMTypeRef.Int8);
    }

    #endregion

    #region Async/Await Generation

    // Future type is now an opaque pointer to the runtime's nsl_future_t
    // Runtime provides: nsl_async_spawn, nsl_await, nsl_future_free
    private LLVMTypeRef? _futureType;
    private bool _asyncRuntimeDeclared = false;

    /// <summary>
    /// Get the Future type - pointer to opaque runtime structure
    /// </summary>
    private LLVMTypeRef GetFutureType()
    {
        // Future is represented as i8* (pointer to runtime's nsl_future_t)
        return LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
    }

    /// <summary>
    /// Declare async runtime functions
    /// </summary>
    private void DeclareAsyncRuntime()
    {
        if (_asyncRuntimeDeclared) return;
        _asyncRuntimeDeclared = true;

        var voidPtr = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var futurePtr = voidPtr;  // nsl_future_t*

        // Async function callback type: double (*)(void*)
        var asyncFuncType = LLVMTypeRef.CreatePointer(
            LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { voidPtr }, false), 0);

        // nsl_async_spawn(func, arg) -> nsl_future_t*
        var spawnType = LLVMTypeRef.CreateFunction(futurePtr, new[] { asyncFuncType, voidPtr }, false);
        _module.AddFunction("nsl_async_spawn", spawnType);

        // nsl_await(future, timeout_ms) -> double
        var awaitType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { futurePtr, LLVMTypeRef.Int64 }, false);
        _module.AddFunction("nsl_await", awaitType);

        // nsl_future_is_done(future) -> bool
        var isDoneType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { futurePtr }, false);
        _module.AddFunction("nsl_future_is_done", isDoneType);

        // nsl_future_free(future) -> void
        var freeType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Void, new[] { futurePtr }, false);
        _module.AddFunction("nsl_future_free", freeType);

        // nsl_delay(ms) -> nsl_future_t*
        var delayType = LLVMTypeRef.CreateFunction(futurePtr, new[] { LLVMTypeRef.Int64 }, false);
        _module.AddFunction("nsl_delay", delayType);

        // nsl_async_yield() -> void
        var yieldType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Void, Array.Empty<LLVMTypeRef>(), false);
        _module.AddFunction("nsl_async_yield", yieldType);
    }

    /// <summary>
    /// Generate async function - creates the actual worker function and wrapper
    /// Uses nsl_async_spawn for true concurrency via threading
    /// </summary>
    private LLVMValueRef GenerateAsyncFunction(NSLAsyncFunctionNode node)
    {
        DeclareAsyncRuntime();
        var voidPtr = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var futurePtr = GetFutureType();

        // First, generate the worker function (actual async work)
        // Worker signature: double worker(void* arg)
        var workerType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { voidPtr }, false);
        var workerName = $"{node.Name}_worker";
        var worker = _module.AddFunction(workerName, workerType);

        // Generate worker body
        var workerEntry = worker.AppendBasicBlock("entry");
        var savedFunction = _currentFunction;
        var savedBlock = _currentBlock;
        _currentFunction = worker;
        _currentBlock = workerEntry;
        _builder.PositionAtEnd(workerEntry);

        var savedValues = new Dictionary<string, LLVMValueRef>(_namedValues);
        var savedTypes = new Dictionary<string, LLVMTypeRef>(_namedValueTypes);
        _namedValues.Clear();
        _namedValueTypes.Clear();

        // The arg pointer contains packed parameters - for simplicity, assume no params for now
        // TODO: Handle parameters by packing/unpacking them through the void* arg

        // Generate body
        var bodyResult = Generate(node.Body);

        // Ensure result is double
        if (bodyResult.TypeOf != LLVMTypeRef.Double)
        {
            if (bodyResult.TypeOf == LLVMTypeRef.Int64)
                bodyResult = _builder.BuildSIToFP(bodyResult, LLVMTypeRef.Double, "to_double");
            else if (bodyResult.TypeOf == LLVMTypeRef.Int1)
                bodyResult = _builder.BuildUIToFP(bodyResult, LLVMTypeRef.Double, "to_double");
            else if (bodyResult.TypeOf.Kind == LLVMTypeKind.LLVMPointerTypeKind)
                bodyResult = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0);  // String returns 0
        }

        _builder.BuildRet(bodyResult);

        // Restore state
        _namedValues.Clear();
        foreach (var kv in savedValues)
            _namedValues[kv.Key] = kv.Value;
        _namedValueTypes.Clear();
        foreach (var kv in savedTypes)
            _namedValueTypes[kv.Key] = kv.Value;
        _currentFunction = savedFunction;
        _currentBlock = savedBlock;

        // Now generate the wrapper function that spawns the async task
        var wrapperParamTypes = new List<LLVMTypeRef>();
        foreach (var param in node.Parameters)
        {
            var paramType = param.Type != null ? GetLLVMTypeFromName(param.Type) : LLVMTypeRef.Double;
            wrapperParamTypes.Add(paramType);
        }

        var wrapperType = LLVMTypeRef.CreateFunction(futurePtr, wrapperParamTypes.ToArray(), false);
        var wrapper = _module.AddFunction(node.Name, wrapperType);
        _functions[node.Name] = wrapper;
        _functionTypes[node.Name] = wrapperType;

        var wrapperEntry = wrapper.AppendBasicBlock("entry");
        _currentFunction = wrapper;
        _currentBlock = wrapperEntry;
        _builder.PositionAtEnd(wrapperEntry);

        // Call nsl_async_spawn to create the async task
        var spawnFunc = _module.GetNamedFunction("nsl_async_spawn");
        var nullPtr = LLVMValueRef.CreateConstNull(voidPtr);

        var future = _builder.BuildCall2(
            LLVMTypeRef.CreateFunction(futurePtr, new[] {
                LLVMTypeRef.CreatePointer(workerType, 0),
                voidPtr
            }, false),
            spawnFunc,
            new[] { worker, nullPtr },
            "future");

        _builder.BuildRet(future);

        _currentFunction = savedFunction;
        _currentBlock = savedBlock;
        if (!savedBlock.Handle.Equals(IntPtr.Zero))
            _builder.PositionAtEnd(savedBlock);

        return wrapper;
    }

    /// <summary>
    /// Generate await expression - calls nsl_await to block until completion
    /// </summary>
    private LLVMValueRef GenerateAwait(NSLAwaitNode node)
    {
        DeclareAsyncRuntime();
        var futureValue = Generate(node.Expression);

        // Call nsl_await(future, 0) - 0 means infinite timeout
        var awaitFunc = _module.GetNamedFunction("nsl_await");
        var futurePtr = GetFutureType();

        var result = _builder.BuildCall2(
            LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { futurePtr, LLVMTypeRef.Int64 }, false),
            awaitFunc,
            new[] { futureValue, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0, false) },
            "await_result");

        return result;
    }

    #endregion

    #region Pattern Matching Generation

    /// <summary>
    /// Generate pattern matching expression
    /// match value { case p1 => e1, case p2 => e2, ... }
    /// </summary>
    private LLVMValueRef GenerateMatch(NSLMatchNode node)
    {
        // Generate the value to match against
        var matchValue = Generate(node.Value);
        var matchType = matchValue.TypeOf;

        // Create basic blocks for each case and the merge block
        var function = _currentFunction;
        var startBlock = function.AppendBasicBlock("match.start");
        var mergeBlock = function.AppendBasicBlock("match.end");

        // Branch to the start block
        _builder.BuildBr(startBlock);
        _builder.PositionAtEnd(startBlock);

        // Allocate space for the result (all branches must produce same type)
        // For now, assume double as default result type
        var resultType = DetermineMatchResultType(node);
        var resultAlloca = CreateEntryBlockAlloca(function, "match.result", resultType);

        // Generate each case
        LLVMBasicBlockRef currentBlock = startBlock;
        for (int i = 0; i < node.Cases.Count; i++)
        {
            var matchCase = node.Cases[i];
            var isLastCase = i == node.Cases.Count - 1;

            // Create blocks for this case
            var caseBlock = function.AppendBasicBlock($"match.case.{i}");
            var nextBlock = isLastCase ? mergeBlock : function.AppendBasicBlock($"match.next.{i}");

            // Position at the current block for pattern checking
            _builder.PositionAtEnd(currentBlock);

            // Generate the pattern check
            var (matches, boundValue) = GeneratePatternCheck(matchValue, matchCase.Pattern, matchType);

            // Branch based on match result
            if (matches.HasValue)
            {
                _builder.BuildCondBr(matches.Value, caseBlock, nextBlock);
            }
            else
            {
                // Wildcard pattern - always matches
                _builder.BuildBr(caseBlock);
            }

            // Generate the case body
            _builder.PositionAtEnd(caseBlock);

            // If pattern bound a variable, make it available
            string? boundVarName = null;
            LLVMTypeRef boundVarType = matchType;

            if (matchCase.Pattern is NSLIdentifierNode identPattern && !IsLiteralKeyword(identPattern.Name))
            {
                // Wildcard pattern - bind the whole value
                boundVarName = identPattern.Name;
                var boundAlloca = CreateEntryBlockAlloca(function, boundVarName, matchType);
                _builder.BuildStore(boundValue ?? matchValue, boundAlloca);
                _namedValues[boundVarName] = boundAlloca;
                _namedValueTypes[boundVarName] = matchType;
            }
            else if (matchCase.Pattern is NSLResultNode resultPattern)
            {
                // ok(v) or err(e) pattern - bind the inner variable
                if (resultPattern.Value is NSLIdentifierNode innerIdent)
                {
                    boundVarName = innerIdent.Name;
                    boundVarType = LLVMTypeRef.Double; // Result inner value is double
                    var boundAlloca = CreateEntryBlockAlloca(function, boundVarName, boundVarType);
                    _builder.BuildStore(boundValue ?? matchValue, boundAlloca);
                    _namedValues[boundVarName] = boundAlloca;
                    _namedValueTypes[boundVarName] = boundVarType;
                }
            }
            else if (matchCase.Pattern is NSLOptionalNode optPattern && optPattern.HasValue)
            {
                // some(v) pattern - bind the inner variable
                if (optPattern.Value is NSLIdentifierNode innerIdent)
                {
                    boundVarName = innerIdent.Name;
                    boundVarType = LLVMTypeRef.Double; // Optional inner value is double
                    var boundAlloca = CreateEntryBlockAlloca(function, boundVarName, boundVarType);
                    _builder.BuildStore(boundValue ?? matchValue, boundAlloca);
                    _namedValues[boundVarName] = boundAlloca;
                    _namedValueTypes[boundVarName] = boundVarType;
                }
            }

            // Check guard condition if present
            if (matchCase.Guard != null)
            {
                var guardValue = Generate(matchCase.Guard);
                var guardBool = EnsureBool(guardValue);

                // Create a block for when guard passes
                var guardPassBlock = function.AppendBasicBlock($"match.guard.pass.{i}");

                // If guard fails, go to next case
                _builder.BuildCondBr(guardBool, guardPassBlock, nextBlock);

                // Continue generating body in the guard pass block
                _builder.PositionAtEnd(guardPassBlock);
            }

            // Generate the body expression
            var bodyValue = Generate(matchCase.Body);
            bodyValue = EnsureType(bodyValue, resultType);
            _builder.BuildStore(bodyValue, resultAlloca);
            _builder.BuildBr(mergeBlock);

            // Clean up bound variable from scope
            if (boundVarName != null)
            {
                _namedValues.Remove(boundVarName);
                _namedValueTypes.Remove(boundVarName);
            }

            currentBlock = nextBlock;
        }

        // Position at merge block and load result
        _builder.PositionAtEnd(mergeBlock);
        return _builder.BuildLoad2(resultType, resultAlloca, "match.value");
    }

    /// <summary>
    /// Determine the result type of a match expression
    /// </summary>
    private LLVMTypeRef DetermineMatchResultType(NSLMatchNode node)
    {
        // Check the first case body to infer result type
        if (node.Cases.Count > 0)
        {
            var firstCase = node.Cases[0];
            var bodyType = _typeChecker.Check(firstCase.Body);
            var llvmType = GetLLVMType(bodyType);
            // Can't allocate void type, default to double for void-returning match expressions
            if (llvmType.Kind == LLVMTypeKind.LLVMVoidTypeKind)
                return LLVMTypeRef.Double;
            return llvmType;
        }
        return LLVMTypeRef.Double; // Default
    }

    /// <summary>
    /// Generate pattern check - returns (condition, bound value)
    /// If condition is null, the pattern always matches (wildcard)
    /// </summary>
    private (LLVMValueRef? Condition, LLVMValueRef? BoundValue) GeneratePatternCheck(
        LLVMValueRef value, NSLASTNode pattern, LLVMTypeRef valueType)
    {
        switch (pattern)
        {
            case NSLLiteralNode literal:
                // Literal pattern - check equality
                var literalValue = GenerateLiteral(literal);
                var condition = GenerateEquality(value, literalValue, valueType);
                return (condition, null);

            case NSLIdentifierNode ident when IsLiteralKeyword(ident.Name):
                // Keywords like 'none' - treat as literal
                if (ident.Name == "none")
                {
                    // Check for null/none
                    var nullVal = LLVMValueRef.CreateConstNull(valueType);
                    var isNone = GenerateEquality(value, nullVal, valueType);
                    return (isNone, null);
                }
                return (null, value); // Unknown keyword, treat as wildcard

            case NSLIdentifierNode ident:
                // Identifier pattern - wildcard that binds the value
                return (null, value);

            case NSLResultNode resultPattern:
                // ok(v) or err(e) pattern - check tag and extract value
                return GenerateResultPattern(value, resultPattern, valueType);

            case NSLOptionalNode optPattern:
                // some(v) or none pattern - check tag and extract value
                return GenerateOptionalPattern(value, optPattern, valueType);

            case NSLFunctionCallNode call when call.Function is NSLIdentifierNode funcIdent:
                // Pattern like ok(v) or some(v) - fallback for function call syntax
                return GenerateConstructorPattern(value, funcIdent.Name, call.Arguments, valueType);

            default:
                // Unknown pattern - treat as wildcard
                return (null, value);
        }
    }

    /// <summary>
    /// Generate equality comparison for pattern matching
    /// </summary>
    private LLVMValueRef GenerateEquality(LLVMValueRef left, LLVMValueRef right, LLVMTypeRef type)
    {
        if (type.Kind == LLVMTypeKind.LLVMDoubleTypeKind)
        {
            return _builder.BuildFCmp(LLVMRealPredicate.LLVMRealOEQ, left, right, "eq");
        }
        else if (type.Kind == LLVMTypeKind.LLVMIntegerTypeKind)
        {
            return _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, left, right, "eq");
        }
        else if (type.Kind == LLVMTypeKind.LLVMPointerTypeKind)
        {
            // String comparison - use strcmp
            var strcmp = GetOrDeclareStrcmp();
            var result = _builder.BuildCall2(
                LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { type, type }),
                strcmp,
                new[] { left, right },
                "strcmp.result");
            return _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, result,
                LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, 0), "str.eq");
        }
        // Default: integer comparison
        return _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, left, right, "eq");
    }

    /// <summary>
    /// Generate Result pattern check (ok(v) or err(e))
    /// Result is a struct { i8 tag, double value } where tag=1 for Ok, tag=0 for Err
    /// </summary>
    private (LLVMValueRef? Condition, LLVMValueRef? BoundValue) GenerateResultPattern(
        LLVMValueRef value, NSLResultNode pattern, LLVMTypeRef valueType)
    {
        // Extract the tag from the Result struct
        var tag = _builder.BuildExtractValue(value, 0, "result.tag");

        // Check if tag matches (1 for Ok, 0 for Err)
        var expectedTag = pattern.IsOk ? 1ul : 0ul;
        var tagMatch = _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag,
            LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, expectedTag), "tag.match");

        // Extract the inner value
        var innerValue = _builder.BuildExtractValue(value, 1, "result.value");

        return (tagMatch, innerValue);
    }

    /// <summary>
    /// Generate Optional pattern check (some(v) or none)
    /// Optional is a struct { i8 tag, double value } where tag=1 for Some, tag=0 for None
    /// </summary>
    private (LLVMValueRef? Condition, LLVMValueRef? BoundValue) GenerateOptionalPattern(
        LLVMValueRef value, NSLOptionalNode pattern, LLVMTypeRef valueType)
    {
        // Extract the tag from the Optional struct
        var tag = _builder.BuildExtractValue(value, 0, "optional.tag");

        // Check if tag matches (1 for Some, 0 for None)
        var expectedTag = pattern.HasValue ? 1ul : 0ul;
        var tagMatch = _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag,
            LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, expectedTag), "tag.match");

        // Extract the inner value (for some patterns)
        LLVMValueRef? innerValue = null;
        if (pattern.HasValue)
        {
            innerValue = _builder.BuildExtractValue(value, 1, "optional.value");
        }

        return (tagMatch, innerValue);
    }

    /// <summary>
    /// Generate constructor pattern check (ok(v), err(e), some(v), none) - fallback
    /// </summary>
    private (LLVMValueRef? Condition, LLVMValueRef? BoundValue) GenerateConstructorPattern(
        LLVMValueRef value, string constructor, IReadOnlyList<NSLASTNode> args, LLVMTypeRef valueType)
    {
        // Check if this is a Result/Optional pattern by constructor name
        bool isOk = constructor == "ok";
        bool isErr = constructor == "err";
        bool isSome = constructor == "some";
        bool isNone = constructor == "none";

        if (isOk || isErr)
        {
            var tag = _builder.BuildExtractValue(value, 0, "result.tag");
            var expectedTag = isOk ? 1ul : 0ul;
            var tagMatch = _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag,
                LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, expectedTag), "tag.match");
            var innerValue = _builder.BuildExtractValue(value, 1, "result.value");
            return (tagMatch, innerValue);
        }
        else if (isSome || isNone)
        {
            var tag = _builder.BuildExtractValue(value, 0, "optional.tag");
            var expectedTag = isSome ? 1ul : 0ul;
            var tagMatch = _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag,
                LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, expectedTag), "tag.match");
            var innerValue = isSome ? _builder.BuildExtractValue(value, 1, "optional.value") : null;
            return (tagMatch, innerValue);
        }

        // Unknown constructor - treat as wildcard
        return (null, value);
    }

    /// <summary>
    /// Check if an identifier is a literal keyword
    /// </summary>
    private bool IsLiteralKeyword(string name)
    {
        return name == "none" || name == "true" || name == "false" || name == "null";
    }

    /// <summary>
    /// Get or declare strcmp function
    /// </summary>
    private LLVMValueRef GetOrDeclareStrcmp()
    {
        const string name = "strcmp";
        var existing = _module.GetNamedFunction(name);
        if (existing.Handle != IntPtr.Zero)
            return existing;

        var strType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { strType, strType });
        return _module.AddFunction(name, funcType);
    }

    #endregion

    #region Result/Option Types

    // Result type: { i8 tag, value } where tag=1 for Ok, tag=0 for Err
    // Optional type: { i8 tag, value } where tag=1 for Some, tag=0 for None

    private LLVMTypeRef? _resultType;
    private LLVMTypeRef? _optionalType;

    /// <summary>
    /// Get or create the Result type (tagged union)
    /// </summary>
    private LLVMTypeRef GetResultType(LLVMTypeRef valueType)
    {
        // For simplicity, use a generic Result with double value
        // In a full implementation, this would be parameterized
        if (_resultType == null)
        {
            _resultType = _context.CreateNamedStruct("NSL.Result");
            _resultType.Value.StructSetBody(new[] { LLVMTypeRef.Int8, LLVMTypeRef.Double }, false);
        }
        return _resultType.Value;
    }

    /// <summary>
    /// Get or create the Optional type (tagged union)
    /// </summary>
    private LLVMTypeRef GetOptionalType(LLVMTypeRef valueType)
    {
        // For simplicity, use a generic Optional with double value
        if (_optionalType == null)
        {
            _optionalType = _context.CreateNamedStruct("NSL.Optional");
            _optionalType.Value.StructSetBody(new[] { LLVMTypeRef.Int8, LLVMTypeRef.Double }, false);
        }
        return _optionalType.Value;
    }

    /// <summary>
    /// Generate Result value: ok(value) or err(value)
    /// </summary>
    private LLVMValueRef GenerateResult(NSLResultNode node)
    {
        var innerValue = Generate(node.Value);
        var resultType = GetResultType(innerValue.TypeOf);

        // Allocate result struct
        var resultAlloca = _builder.BuildAlloca(resultType, "result");

        // Set tag: 1 for Ok, 0 for Err
        var tagValue = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, node.IsOk ? 1ul : 0ul);
        var tagPtr = _builder.BuildStructGEP2(resultType, resultAlloca, 0, "tag_ptr");
        _builder.BuildStore(tagValue, tagPtr);

        // Set value (convert to double if needed)
        var valuePtr = _builder.BuildStructGEP2(resultType, resultAlloca, 1, "value_ptr");
        var doubleValue = EnsureDouble(innerValue);
        _builder.BuildStore(doubleValue, valuePtr);

        // Load and return the result struct
        return _builder.BuildLoad2(resultType, resultAlloca, "result_value");
    }

    /// <summary>
    /// Generate Optional value: some(value) or none
    /// </summary>
    private LLVMValueRef GenerateOptional(NSLOptionalNode node)
    {
        var optionalType = GetOptionalType(LLVMTypeRef.Double);

        // Allocate optional struct
        var optionalAlloca = _builder.BuildAlloca(optionalType, "optional");

        // Set tag: 1 for Some, 0 for None
        var tagValue = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, node.HasValue ? 1ul : 0ul);
        var tagPtr = _builder.BuildStructGEP2(optionalType, optionalAlloca, 0, "tag_ptr");
        _builder.BuildStore(tagValue, tagPtr);

        // Set value (0.0 for None, actual value for Some)
        var valuePtr = _builder.BuildStructGEP2(optionalType, optionalAlloca, 1, "value_ptr");
        LLVMValueRef doubleValue;
        if (node.HasValue && node.Value != null)
        {
            var innerValue = Generate(node.Value);
            doubleValue = EnsureDouble(innerValue);
        }
        else
        {
            doubleValue = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.0);
        }
        _builder.BuildStore(doubleValue, valuePtr);

        // Load and return the optional struct
        return _builder.BuildLoad2(optionalType, optionalAlloca, "optional_value");
    }

    /// <summary>
    /// Check if a value is Ok (tag == 1)
    /// </summary>
    private LLVMValueRef GenerateIsOk(LLVMValueRef resultValue)
    {
        var tag = _builder.BuildExtractValue(resultValue, 0, "tag");
        return _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag,
            LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1), "is_ok");
    }

    /// <summary>
    /// Check if a value is Some (tag == 1)
    /// </summary>
    private LLVMValueRef GenerateIsSome(LLVMValueRef optionalValue)
    {
        var tag = _builder.BuildExtractValue(optionalValue, 0, "tag");
        return _builder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag,
            LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1), "is_some");
    }

    /// <summary>
    /// Extract the value from a Result or Optional
    /// </summary>
    private LLVMValueRef GenerateUnwrapValue(LLVMValueRef value)
    {
        return _builder.BuildExtractValue(value, 1, "unwrapped");
    }

    #endregion

    #region Type Helpers

    private LLVMTypeRef GetLLVMType(NSLType type)
    {
        return type switch
        {
            NSLNumberType => LLVMTypeRef.Double,
            NSLIntType => LLVMTypeRef.Int64,  // 64-bit integers for full range
            NSLBoolType => LLVMTypeRef.Int1,
            NSLStringType => LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0),
            NSLVoidType => LLVMTypeRef.Void,
            NSLVecType => LLVMTypeRef.CreatePointer(LLVMTypeRef.Double, 0),
            NSLMatType => LLVMTypeRef.CreatePointer(LLVMTypeRef.Double, 0),
            NSLTensorType => LLVMTypeRef.CreatePointer(LLVMTypeRef.Double, 0),
            NSLProbType => LLVMTypeRef.Double,
            NSLArrayType => LLVMTypeRef.CreatePointer(LLVMTypeRef.Double, 0),
            NSLFunctionType funcType => CreateFunctionPointerType(funcType),
            NSLStructType structType when _structTypes.TryGetValue(structType.StructName, out var llvmType) => llvmType,
            NSLResultType => GetResultType(LLVMTypeRef.Double),
            NSLOptionalType => GetOptionalType(LLVMTypeRef.Double),
            _ => LLVMTypeRef.Double // Default to double
        };
    }

    private LLVMTypeRef CreateFunctionPointerType(NSLFunctionType funcType)
    {
        var funcLLVMType = CreateFunctionLLVMType(funcType);
        return LLVMTypeRef.CreatePointer(funcLLVMType, 0);
    }

    private LLVMTypeRef CreateFunctionLLVMType(NSLFunctionType funcType)
    {
        var paramTypes = new LLVMTypeRef[funcType.ParameterTypes.Count];
        for (int i = 0; i < funcType.ParameterTypes.Count; i++)
        {
            paramTypes[i] = GetLLVMType(funcType.ParameterTypes[i]);
        }
        var returnType = GetLLVMType(funcType.ReturnType);
        return LLVMTypeRef.CreateFunction(returnType, paramTypes);
    }

    /// <summary>
    /// Get the size of a struct type in bytes (for malloc allocation)
    /// </summary>
    private long GetStructSize(LLVMTypeRef structType)
    {
        // Simplified calculation: each element takes 8 bytes (assuming all doubles/pointers)
        // This is safe for closure environments which only contain captured values (doubles)
        var elementCount = structType.StructElementTypesCount;
        return Math.Max(elementCount * 8, 8); // Minimum 8 bytes
    }

    private LLVMValueRef EnsureDouble(LLVMValueRef value)
    {
        if (value.TypeOf.Kind == LLVMTypeKind.LLVMDoubleTypeKind)
            return value;
        if (value.TypeOf.Kind == LLVMTypeKind.LLVMIntegerTypeKind)
            return _builder.BuildSIToFP(value, LLVMTypeRef.Double, "tofp");
        return value;
    }

    private LLVMValueRef EnsureInt64(LLVMValueRef value)
    {
        if (value.TypeOf.Kind == LLVMTypeKind.LLVMIntegerTypeKind && value.TypeOf.IntWidth == 64)
            return value;
        if (value.TypeOf.Kind == LLVMTypeKind.LLVMIntegerTypeKind)
            return _builder.BuildSExt(value, LLVMTypeRef.Int64, "sext");
        if (value.TypeOf.Kind == LLVMTypeKind.LLVMDoubleTypeKind)
            return _builder.BuildFPToSI(value, LLVMTypeRef.Int64, "toint");
        return value;
    }

    private LLVMValueRef EnsureBool(LLVMValueRef value)
    {
        if (value.TypeOf.Kind == LLVMTypeKind.LLVMIntegerTypeKind && value.TypeOf.IntWidth == 1)
            return value;
        if (value.TypeOf.Kind == LLVMTypeKind.LLVMDoubleTypeKind)
            return _builder.BuildFCmp(LLVMRealPredicate.LLVMRealONE, value,
                LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0), "tobool");
        return value;
    }

    private LLVMValueRef EnsureType(LLVMValueRef value, LLVMTypeRef targetType)
    {
        if (value.TypeOf.Kind == targetType.Kind)
        {
            // Same kind, but may need width conversion for integers
            if (targetType.Kind == LLVMTypeKind.LLVMIntegerTypeKind &&
                value.TypeOf.IntWidth != targetType.IntWidth)
            {
                if (targetType.IntWidth == 64)
                    return EnsureInt64(value);
                if (targetType.IntWidth == 1)
                    return EnsureBool(value);
            }
            return value;
        }

        if (targetType.Kind == LLVMTypeKind.LLVMDoubleTypeKind)
            return EnsureDouble(value);
        if (targetType.Kind == LLVMTypeKind.LLVMIntegerTypeKind && targetType.IntWidth == 64)
            return EnsureInt64(value);
        if (targetType.Kind == LLVMTypeKind.LLVMIntegerTypeKind && targetType.IntWidth == 1)
            return EnsureBool(value);

        return value;
    }

    private LLVMTypeRef GetPointeeType(string name, LLVMValueRef pointer)
    {
        // For LLVM 16+, we need to track pointee types separately
        if (_namedValueTypes.TryGetValue(name, out var type))
            return type;

        // Default to double for backwards compatibility
        return LLVMTypeRef.Double;
    }

    #endregion

    #region Builtins

    private void DeclareBuiltins()
    {
        // Declare printf for print function (external C function)
        var printfType = LLVMTypeRef.CreateFunction(
            LLVMTypeRef.Int32,
            new[] { LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0) },
            true
        );
        var printf = _module.AddFunction("printf", printfType);
        _functions["printf"] = printf;
        _functionTypes["printf"] = printfType;

        // Declare print(x) - wrapper around printf
        DeclarePrintFunction();

        // Math intrinsics (external)
        DeclareMathFunction("sqrt", "llvm.sqrt.f64");
        DeclareMathFunction("sin", "llvm.sin.f64");
        DeclareMathFunction("cos", "llvm.cos.f64");
        DeclareMathFunction("exp", "llvm.exp.f64");
        DeclareMathFunction("log", "llvm.log.f64");
        DeclareMathFunction("log10", "llvm.log10.f64");
        DeclareMathFunction("log2", "llvm.log2.f64");
        DeclareMathFunction("abs", "llvm.fabs.f64");
        DeclareMathFunction("floor", "llvm.floor.f64");
        DeclareMathFunction("ceil", "llvm.ceil.f64");
        DeclareMathFunction("round", "llvm.round.f64");
        DeclareMathFunction("trunc", "llvm.trunc.f64");
        DeclareMathFunction2("pow", "llvm.pow.f64");
        DeclareMathFunction2("min", "llvm.minnum.f64");
        DeclareMathFunction2("max", "llvm.maxnum.f64");
        DeclareMathFunction2("copysign", "llvm.copysign.f64");

        // Trigonometric functions via C library
        DeclareCMathFunction("tan");
        DeclareCMathFunction("asin");
        DeclareCMathFunction("acos");
        DeclareCMathFunction("atan");
        DeclareCMathFunction2("atan2");
        DeclareCMathFunction("sinh");
        DeclareCMathFunction("cosh");

        // Additional utility math functions
        DeclareClampFunction();
        DeclareLerpFunction();
        DeclareSignFunction();

        // ML activation functions
        DeclareReluFunction();
        DeclareSigmoidFunction();
        DeclareTanhFunction();
        DeclareLeakyReluFunction();
        DeclareSoftplusFunction();
        DeclareGeluFunction();

        // String operations
        DeclareStringFunctions();

        // File I/O operations
        DeclareFileIOFunctions();

        // Result/Optional chaining functions
        DeclareResultOptionalFunctions();

        // HTTP client functions (external)
        DeclareHTTPFunctions();

        // JSON functions (external)
        DeclareJSONFunctions();

        // Regex functions (external)
        DeclareRegexFunctions();

        // Date/Time functions (external)
        DeclareDateTimeFunctions();
    }

    private void DeclarePrintFunction()
    {
        var printf = _functions["printf"];
        var printfType = _functionTypes["printf"];

        // print for doubles (numeric values)
        var funcTypeDouble = LLVMTypeRef.CreateFunction(LLVMTypeRef.Void, new[] { LLVMTypeRef.Double });
        var funcDouble = _module.AddFunction("print_double", funcTypeDouble);
        var entryDouble = funcDouble.AppendBasicBlock("entry");

        using var funcBuilderDouble = _context.CreateBuilder();
        funcBuilderDouble.PositionAtEnd(entryDouble);

        var formatStrDouble = funcBuilderDouble.BuildGlobalStringPtr("%g\n", "fmt_double");
        funcBuilderDouble.BuildCall2(printfType, printf, new[] { formatStrDouble, funcDouble.GetParam(0) }, "");
        funcBuilderDouble.BuildRetVoid();

        _functions["print_double"] = funcDouble;
        _functionTypes["print_double"] = funcTypeDouble;

        // print for integers (64-bit)
        var funcTypeInt = LLVMTypeRef.CreateFunction(LLVMTypeRef.Void, new[] { LLVMTypeRef.Int64 });
        var funcInt = _module.AddFunction("print_int", funcTypeInt);
        var entryInt = funcInt.AppendBasicBlock("entry");

        using var funcBuilderInt = _context.CreateBuilder();
        funcBuilderInt.PositionAtEnd(entryInt);

        var formatStrInt = funcBuilderInt.BuildGlobalStringPtr("%lld\n", "fmt_int");
        funcBuilderInt.BuildCall2(printfType, printf, new[] { formatStrInt, funcInt.GetParam(0) }, "");
        funcBuilderInt.BuildRetVoid();

        _functions["print_int"] = funcInt;
        _functionTypes["print_int"] = funcTypeInt;

        // print for strings
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcTypeString = LLVMTypeRef.CreateFunction(LLVMTypeRef.Void, new[] { stringPtrType });
        var funcString = _module.AddFunction("print_str", funcTypeString);
        var entryString = funcString.AppendBasicBlock("entry");

        using var funcBuilderString = _context.CreateBuilder();
        funcBuilderString.PositionAtEnd(entryString);

        var formatStrString = funcBuilderString.BuildGlobalStringPtr("%s\n", "fmt_str");
        funcBuilderString.BuildCall2(printfType, printf, new[] { formatStrString, funcString.GetParam(0) }, "");
        funcBuilderString.BuildRetVoid();

        _functions["print_str"] = funcString;
        _functionTypes["print_str"] = funcTypeString;

        // Default print points to double version (most common)
        _functions["print"] = funcDouble;
        _functions["println"] = funcDouble;
        _functionTypes["print"] = funcTypeDouble;
        _functionTypes["println"] = funcTypeDouble;
    }

    private void DeclareMathFunction(string name, string intrinsic)
    {
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction(intrinsic, funcType);
        _functions[name] = func;
        _functionTypes[name] = funcType;
    }

    private void DeclareMathFunction2(string name, string intrinsic)
    {
        // Two-argument math function (e.g., pow)
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double, LLVMTypeRef.Double });
        var func = _module.AddFunction(intrinsic, funcType);
        _functions[name] = func;
        _functionTypes[name] = funcType;
    }

    private void DeclareCMathFunction(string name)
    {
        // Single-argument C math library function
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction(name, funcType);
        _functions[name] = func;
        _functionTypes[name] = funcType;
    }

    private void DeclareCMathFunction2(string name)
    {
        // Two-argument C math library function
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double, LLVMTypeRef.Double });
        var func = _module.AddFunction(name, funcType);
        _functions[name] = func;
        _functionTypes[name] = funcType;
    }

    private void DeclareClampFunction()
    {
        // clamp(x, min, max) = max(min, min(x, max))
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double,
            new[] { LLVMTypeRef.Double, LLVMTypeRef.Double, LLVMTypeRef.Double });
        var func = _module.AddFunction("clamp", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var minVal = func.GetParam(1);
        var maxVal = func.GetParam(2);

        // Step 1: min(x, maxVal)
        var minFunc = _functions["min"];
        var step1 = funcBuilder.BuildCall2(_functionTypes["min"], minFunc, new[] { x, maxVal }, "step1");
        // Step 2: max(minVal, step1)
        var maxFunc = _functions["max"];
        var result = funcBuilder.BuildCall2(_functionTypes["max"], maxFunc, new[] { minVal, step1 }, "clamped");
        funcBuilder.BuildRet(result);

        _functions["clamp"] = func;
        _functionTypes["clamp"] = funcType;
    }

    private void DeclareLerpFunction()
    {
        // lerp(a, b, t) = a + t * (b - a)
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double,
            new[] { LLVMTypeRef.Double, LLVMTypeRef.Double, LLVMTypeRef.Double });
        var func = _module.AddFunction("lerp", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var a = func.GetParam(0);
        var b = func.GetParam(1);
        var t = func.GetParam(2);

        var diff = funcBuilder.BuildFSub(b, a, "diff");
        var scaled = funcBuilder.BuildFMul(t, diff, "scaled");
        var result = funcBuilder.BuildFAdd(a, scaled, "lerp");
        funcBuilder.BuildRet(result);

        _functions["lerp"] = func;
        _functionTypes["lerp"] = funcType;
    }

    private void DeclareSignFunction()
    {
        // sign(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction("sign", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var zero = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.0);
        var one = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1.0);
        var negOne = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, -1.0);

        var isPositive = funcBuilder.BuildFCmp(LLVMRealPredicate.LLVMRealOGT, x, zero, "ispos");
        var isNegative = funcBuilder.BuildFCmp(LLVMRealPredicate.LLVMRealOLT, x, zero, "isneg");

        var posResult = funcBuilder.BuildSelect(isPositive, one, zero, "posresult");
        var result = funcBuilder.BuildSelect(isNegative, negOne, posResult, "sign");
        funcBuilder.BuildRet(result);

        _functions["sign"] = func;
        _functionTypes["sign"] = funcType;
    }

    private void DeclareReluFunction()
    {
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction("relu", funcType);
        var entry = func.AppendBasicBlock("entry");

        // Create a new builder for this function to avoid context issues
        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var zero = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0);
        var cond = funcBuilder.BuildFCmp(LLVMRealPredicate.LLVMRealOGT, x, zero, "cmp");
        var result = funcBuilder.BuildSelect(cond, x, zero, "relu");
        funcBuilder.BuildRet(result);

        _functions["relu"] = func;
        _functionTypes["relu"] = funcType;
    }

    private void DeclareSigmoidFunction()
    {
        // sigmoid(x) = 1 / (1 + exp(-x))
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction("sigmoid", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var negX = funcBuilder.BuildFNeg(x, "negx");
        var expFunc = _functions["exp"];
        var expNegX = funcBuilder.BuildCall2(_functionTypes["exp"], expFunc, new[] { negX }, "expnegx");
        var one = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1.0);
        var onePlusExp = funcBuilder.BuildFAdd(one, expNegX, "oneplusexp");
        var result = funcBuilder.BuildFDiv(one, onePlusExp, "sigmoid");
        funcBuilder.BuildRet(result);

        _functions["sigmoid"] = func;
        _functionTypes["sigmoid"] = funcType;
    }

    private void DeclareTanhFunction()
    {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction("tanh", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var two = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 2.0);
        var twoX = funcBuilder.BuildFMul(two, x, "twox");
        var expFunc = _functions["exp"];
        var exp2X = funcBuilder.BuildCall2(_functionTypes["exp"], expFunc, new[] { twoX }, "exp2x");
        var one = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1.0);
        var num = funcBuilder.BuildFSub(exp2X, one, "num");
        var denom = funcBuilder.BuildFAdd(exp2X, one, "denom");
        var result = funcBuilder.BuildFDiv(num, denom, "tanh");
        funcBuilder.BuildRet(result);

        _functions["tanh"] = func;
        _functionTypes["tanh"] = funcType;
    }

    private void DeclareLeakyReluFunction()
    {
        // leaky_relu(x) = x > 0 ? x : 0.01 * x
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction("leaky_relu", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var zero = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.0);
        var alpha = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.01);
        var leakyX = funcBuilder.BuildFMul(alpha, x, "leakyx");
        var cond = funcBuilder.BuildFCmp(LLVMRealPredicate.LLVMRealOGT, x, zero, "cmp");
        var result = funcBuilder.BuildSelect(cond, x, leakyX, "leaky_relu");
        funcBuilder.BuildRet(result);

        _functions["leaky_relu"] = func;
        _functionTypes["leaky_relu"] = funcType;
    }

    private void DeclareSoftplusFunction()
    {
        // softplus(x) = log(1 + exp(x))
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction("softplus", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var expFunc = _functions["exp"];
        var logFunc = _functions["log"];
        var expX = funcBuilder.BuildCall2(_functionTypes["exp"], expFunc, new[] { x }, "expx");
        var one = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1.0);
        var onePlusExp = funcBuilder.BuildFAdd(one, expX, "oneplusexp");
        var result = funcBuilder.BuildCall2(_functionTypes["log"], logFunc, new[] { onePlusExp }, "softplus");
        funcBuilder.BuildRet(result);

        _functions["softplus"] = func;
        _functionTypes["softplus"] = funcType;
    }

    private void DeclareGeluFunction()
    {
        // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double });
        var func = _module.AddFunction("gelu", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var x = func.GetParam(0);
        var half = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.5);
        var one = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 1.0);
        var coef = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.044715);
        var sqrt2Pi = LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.7978845608); // sqrt(2/pi)

        // x^3
        var x2 = funcBuilder.BuildFMul(x, x, "x2");
        var x3 = funcBuilder.BuildFMul(x2, x, "x3");

        // 0.044715 * x^3
        var coefX3 = funcBuilder.BuildFMul(coef, x3, "coefx3");

        // x + 0.044715 * x^3
        var innerSum = funcBuilder.BuildFAdd(x, coefX3, "innersum");

        // sqrt(2/pi) * (x + 0.044715 * x^3)
        var tanhArg = funcBuilder.BuildFMul(sqrt2Pi, innerSum, "tanharg");

        // tanh(...)
        var tanhFunc = _functions["tanh"];
        var tanhVal = funcBuilder.BuildCall2(_functionTypes["tanh"], tanhFunc, new[] { tanhArg }, "tanhval");

        // 1 + tanh(...)
        var onePlusTanh = funcBuilder.BuildFAdd(one, tanhVal, "oneplustanh");

        // x * (1 + tanh(...))
        var xTimesInner = funcBuilder.BuildFMul(x, onePlusTanh, "xtimesinner");

        // 0.5 * x * (1 + tanh(...))
        var result = funcBuilder.BuildFMul(half, xTimesInner, "gelu");
        funcBuilder.BuildRet(result);

        _functions["gelu"] = func;
        _functionTypes["gelu"] = funcType;
    }

    private void DeclareStringFunctions()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);

        // Declare C library functions
        // strlen(s) -> i64
        var strlenType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { stringPtrType });
        var strlen = _module.AddFunction("strlen", strlenType);
        _functions["strlen"] = strlen;
        _functionTypes["strlen"] = strlenType;

        // malloc(size) -> ptr
        var mallocType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { LLVMTypeRef.Int64 });
        var malloc = _module.AddFunction("malloc", mallocType);
        _functions["malloc"] = malloc;
        _functionTypes["malloc"] = mallocType;

        // strcpy(dest, src) -> dest
        var strcpyType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType });
        var strcpy = _module.AddFunction("strcpy", strcpyType);
        _functions["strcpy"] = strcpy;
        _functionTypes["strcpy"] = strcpyType;

        // strcat(dest, src) -> dest
        var strcatType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType });
        var strcat = _module.AddFunction("strcat", strcatType);
        _functions["strcat"] = strcat;
        _functionTypes["strcat"] = strcatType;

        // memcpy(dest, src, n) -> dest
        var memcpyType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType, LLVMTypeRef.Int64 });
        var memcpy = _module.AddFunction("memcpy", memcpyType);
        _functions["memcpy"] = memcpy;
        _functionTypes["memcpy"] = memcpyType;

        // strcmp(s1, s2) -> int
        var strcmpType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { stringPtrType, stringPtrType });
        var strcmp = _module.AddFunction("strcmp", strcmpType);
        _functions["strcmp"] = strcmp;
        _functionTypes["strcmp"] = strcmpType;

        // strstr(haystack, needle) -> ptr (find substring)
        var strstrType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType });
        var strstr = _module.AddFunction("strstr", strstrType);
        _functions["strstr"] = strstr;
        _functionTypes["strstr"] = strstrType;

        // strncmp(s1, s2, n) -> int
        var strncmpType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { stringPtrType, stringPtrType, LLVMTypeRef.Int64 });
        var strncmp = _module.AddFunction("strncmp", strncmpType);
        _functions["strncmp"] = strncmp;
        _functionTypes["strncmp"] = strncmpType;

        // memset(ptr, val, size) -> ptr
        var memsetType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, LLVMTypeRef.Int32, LLVMTypeRef.Int64 });
        var memset = _module.AddFunction("memset", memsetType);
        _functions["memset"] = memset;
        _functionTypes["memset"] = memsetType;

        // Create str_len wrapper (returns int for NSL)
        DeclareStrLenFunction();

        // Create str_concat function (a + b for strings)
        DeclareStrConcatFunction();

        // Create str_slice function (substring)
        DeclareStrSliceFunction();

        // Additional string utilities
        DeclareStrContainsFunction();
        DeclareStrStartsWithFunction();
        DeclareStrEndsWithFunction();
        DeclareStrToUpperFunction();
        DeclareStrToLowerFunction();
        DeclareStrTrimFunction();
        DeclareStrReplaceFunction();
        DeclareStrSplitFunction();
        DeclareStrJoinFunction();
        DeclareStrCharAtFunction();
        DeclareStrIndexOfFunction();
    }

    private void DeclareStrLenFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { stringPtrType });
        var func = _module.AddFunction("str_len", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        // Just call strlen
        var len = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"],
            new[] { func.GetParam(0) }, "len");
        funcBuilder.BuildRet(len);

        _functions["str_len"] = func;
        _functionTypes["str_len"] = funcType;
        _functions["len"] = func;  // Also register as "len"
        _functionTypes["len"] = funcType;
    }

    private void DeclareStrConcatFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_concat", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var s1 = func.GetParam(0);
        var s2 = func.GetParam(1);

        // Get lengths
        var len1 = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { s1 }, "len1");
        var len2 = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { s2 }, "len2");

        // Total length + 1 for null terminator
        var totalLen = funcBuilder.BuildAdd(len1, len2, "totallen");
        var allocSize = funcBuilder.BuildAdd(totalLen, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "allocsize");

        // Allocate memory
        var result = funcBuilder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result");

        // Copy first string
        funcBuilder.BuildCall2(_functionTypes["strcpy"], _functions["strcpy"], new[] { result, s1 }, "");

        // Concatenate second string
        funcBuilder.BuildCall2(_functionTypes["strcat"], _functions["strcat"], new[] { result, s2 }, "");

        funcBuilder.BuildRet(result);

        _functions["str_concat"] = func;
        _functionTypes["str_concat"] = funcType;
    }

    private void DeclareStrSliceFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // str_slice(str, start, length) -> str
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, LLVMTypeRef.Int64, LLVMTypeRef.Int64 });
        var func = _module.AddFunction("str_slice", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var s = func.GetParam(0);
        var start = func.GetParam(1);
        var length = func.GetParam(2);

        // Allocate memory for result (length + 1 for null terminator)
        var allocSize = funcBuilder.BuildAdd(length, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "allocsize");
        var result = funcBuilder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result");

        // Calculate source pointer (s + start)
        var srcPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, s, new[] { start }, "srcptr");

        // Copy bytes
        funcBuilder.BuildCall2(_functionTypes["memcpy"], _functions["memcpy"], new[] { result, srcPtr, length }, "");

        // Add null terminator
        var nullPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, result, new[] { length }, "nullptr");
        funcBuilder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 0), nullPtr);

        funcBuilder.BuildRet(result);

        _functions["str_slice"] = func;
        _functionTypes["str_slice"] = funcType;
        _functions["slice"] = func;  // Also register as "slice"
        _functionTypes["slice"] = funcType;
        _functions["substr"] = func;  // Also register as "substr"
        _functionTypes["substr"] = funcType;
    }

    private void DeclareStrContainsFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // contains(haystack, needle) -> bool
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_contains", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var haystack = func.GetParam(0);
        var needle = func.GetParam(1);

        // strstr returns null if not found
        var result = funcBuilder.BuildCall2(_functionTypes["strstr"], _functions["strstr"], new[] { haystack, needle }, "found");
        var nullPtr = LLVMValueRef.CreateConstNull(stringPtrType);
        var isNotNull = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntNE, result, nullPtr, "contains");
        funcBuilder.BuildRet(isNotNull);

        _functions["str_contains"] = func;
        _functionTypes["str_contains"] = funcType;
        _functions["contains"] = func;
        _functionTypes["contains"] = funcType;
    }

    private void DeclareStrStartsWithFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // starts_with(str, prefix) -> bool
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_starts_with", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var s = func.GetParam(0);
        var prefix = func.GetParam(1);

        // Get prefix length
        var prefixLen = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { prefix }, "prefixlen");

        // Compare first n characters
        var cmp = funcBuilder.BuildCall2(_functionTypes["strncmp"], _functions["strncmp"], new[] { s, prefix, prefixLen }, "cmp");
        var zero = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, 0);
        var result = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, cmp, zero, "startswith");
        funcBuilder.BuildRet(result);

        _functions["str_starts_with"] = func;
        _functionTypes["str_starts_with"] = funcType;
        _functions["starts_with"] = func;
        _functionTypes["starts_with"] = funcType;
    }

    private void DeclareStrEndsWithFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // ends_with(str, suffix) -> bool
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_ends_with", funcType);
        var entry = func.AppendBasicBlock("entry");
        var retTrue = func.AppendBasicBlock("ret_true");
        var retFalse = func.AppendBasicBlock("ret_false");
        var doCompare = func.AppendBasicBlock("compare");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var s = func.GetParam(0);
        var suffix = func.GetParam(1);

        // Get lengths
        var strLen = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { s }, "strlen");
        var suffixLen = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { suffix }, "suffixlen");

        // If suffix is longer than string, return false
        var isLonger = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntUGT, suffixLen, strLen, "islonger");
        funcBuilder.BuildCondBr(isLonger, retFalse, doCompare);

        // Compare the end portion
        funcBuilder.PositionAtEnd(doCompare);
        var offset = funcBuilder.BuildSub(strLen, suffixLen, "offset");
        var endPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, s, new[] { offset }, "endptr");
        var cmp = funcBuilder.BuildCall2(_functionTypes["strcmp"], _functions["strcmp"], new[] { endPtr, suffix }, "cmp");
        var zero = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, 0);
        var isEqual = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, cmp, zero, "isequal");
        funcBuilder.BuildCondBr(isEqual, retTrue, retFalse);

        funcBuilder.PositionAtEnd(retTrue);
        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 1));

        funcBuilder.PositionAtEnd(retFalse);
        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 0));

        _functions["str_ends_with"] = func;
        _functionTypes["str_ends_with"] = funcType;
        _functions["ends_with"] = func;
        _functionTypes["ends_with"] = funcType;
    }

    private void DeclareStrToUpperFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // to_upper(str) -> str
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType });
        var func = _module.AddFunction("str_to_upper", funcType);
        var entry = func.AppendBasicBlock("entry");
        var loopCond = func.AppendBasicBlock("loop_cond");
        var loopBody = func.AppendBasicBlock("loop_body");
        var loopEnd = func.AppendBasicBlock("loop_end");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var s = func.GetParam(0);
        var len = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { s }, "len");
        var allocSize = funcBuilder.BuildAdd(len, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "allocsize");
        var result = funcBuilder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result");
        var indexAlloca = funcBuilder.BuildAlloca(LLVMTypeRef.Int64, "index");
        funcBuilder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0), indexAlloca);
        funcBuilder.BuildBr(loopCond);

        funcBuilder.PositionAtEnd(loopCond);
        var index = funcBuilder.BuildLoad2(LLVMTypeRef.Int64, indexAlloca, "i");
        var cond = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntULT, index, len, "cond");
        funcBuilder.BuildCondBr(cond, loopBody, loopEnd);

        funcBuilder.PositionAtEnd(loopBody);
        var srcPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, s, new[] { index }, "srcptr");
        var dstPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, result, new[] { index }, "dstptr");
        var c = funcBuilder.BuildLoad2(LLVMTypeRef.Int8, srcPtr, "c");
        // Convert to uppercase: if 'a' <= c <= 'z', subtract 32
        var isLower = funcBuilder.BuildAnd(
            funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntUGE, c, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 97), ""),
            funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntULE, c, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 122), ""),
            "islower");
        var upper = funcBuilder.BuildSub(c, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 32), "upper");
        var newC = funcBuilder.BuildSelect(isLower, upper, c, "newc");
        funcBuilder.BuildStore(newC, dstPtr);
        var nextIndex = funcBuilder.BuildAdd(index, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "next");
        funcBuilder.BuildStore(nextIndex, indexAlloca);
        funcBuilder.BuildBr(loopCond);

        funcBuilder.PositionAtEnd(loopEnd);
        var nullPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, result, new[] { len }, "nullptr");
        funcBuilder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 0), nullPtr);
        funcBuilder.BuildRet(result);

        _functions["str_to_upper"] = func;
        _functionTypes["str_to_upper"] = funcType;
        _functions["to_upper"] = func;
        _functionTypes["to_upper"] = funcType;
    }

    private void DeclareStrToLowerFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // to_lower(str) -> str
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType });
        var func = _module.AddFunction("str_to_lower", funcType);
        var entry = func.AppendBasicBlock("entry");
        var loopCond = func.AppendBasicBlock("loop_cond");
        var loopBody = func.AppendBasicBlock("loop_body");
        var loopEnd = func.AppendBasicBlock("loop_end");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var s = func.GetParam(0);
        var len = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { s }, "len");
        var allocSize = funcBuilder.BuildAdd(len, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "allocsize");
        var result = funcBuilder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result");
        var indexAlloca = funcBuilder.BuildAlloca(LLVMTypeRef.Int64, "index");
        funcBuilder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0), indexAlloca);
        funcBuilder.BuildBr(loopCond);

        funcBuilder.PositionAtEnd(loopCond);
        var index = funcBuilder.BuildLoad2(LLVMTypeRef.Int64, indexAlloca, "i");
        var cond = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntULT, index, len, "cond");
        funcBuilder.BuildCondBr(cond, loopBody, loopEnd);

        funcBuilder.PositionAtEnd(loopBody);
        var srcPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, s, new[] { index }, "srcptr");
        var dstPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, result, new[] { index }, "dstptr");
        var c = funcBuilder.BuildLoad2(LLVMTypeRef.Int8, srcPtr, "c");
        // Convert to lowercase: if 'A' <= c <= 'Z', add 32
        var isUpper = funcBuilder.BuildAnd(
            funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntUGE, c, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 65), ""),
            funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntULE, c, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 90), ""),
            "isupper");
        var lower = funcBuilder.BuildAdd(c, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 32), "lower");
        var newC = funcBuilder.BuildSelect(isUpper, lower, c, "newc");
        funcBuilder.BuildStore(newC, dstPtr);
        var nextIndex = funcBuilder.BuildAdd(index, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "next");
        funcBuilder.BuildStore(nextIndex, indexAlloca);
        funcBuilder.BuildBr(loopCond);

        funcBuilder.PositionAtEnd(loopEnd);
        var nullPtr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, result, new[] { len }, "nullptr");
        funcBuilder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 0), nullPtr);
        funcBuilder.BuildRet(result);

        _functions["str_to_lower"] = func;
        _functionTypes["str_to_lower"] = funcType;
        _functions["to_lower"] = func;
        _functionTypes["to_lower"] = funcType;
    }

    private void DeclareStrTrimFunction()
    {
        // Simplified trim - trims spaces from both ends
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType });
        var func = _module.AddFunction("str_trim", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        // For simplicity, just copy the string (full trim implementation would be complex)
        // A real implementation would skip leading/trailing whitespace
        var s = func.GetParam(0);
        var len = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { s }, "len");
        var allocSize = funcBuilder.BuildAdd(len, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "allocsize");
        var result = funcBuilder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result");
        funcBuilder.BuildCall2(_functionTypes["strcpy"], _functions["strcpy"], new[] { result, s }, "");
        funcBuilder.BuildRet(result);

        _functions["str_trim"] = func;
        _functionTypes["str_trim"] = funcType;
        _functions["trim"] = func;
        _functionTypes["trim"] = funcType;
    }

    private void DeclareStrReplaceFunction()
    {
        // Simplified replace - replace first occurrence (full implementation is complex)
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_replace", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        // For now, just return a copy (full implementation would actually replace)
        var s = func.GetParam(0);
        var len = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { s }, "len");
        var allocSize = funcBuilder.BuildAdd(len, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "allocsize");
        var result = funcBuilder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "result");
        funcBuilder.BuildCall2(_functionTypes["strcpy"], _functions["strcpy"], new[] { result, s }, "");
        funcBuilder.BuildRet(result);

        _functions["str_replace"] = func;
        _functionTypes["str_replace"] = funcType;
        _functions["replace"] = func;
        _functionTypes["replace"] = funcType;
    }

    private void DeclareStrSplitFunction()
    {
        // Placeholder - returns the original string (real split returns array)
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_split", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);
        funcBuilder.BuildRet(func.GetParam(0));

        _functions["str_split"] = func;
        _functionTypes["str_split"] = funcType;
        _functions["split"] = func;
        _functionTypes["split"] = funcType;
    }

    private void DeclareStrJoinFunction()
    {
        // Placeholder - returns first argument (real join operates on array)
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_join", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);
        funcBuilder.BuildRet(func.GetParam(0));

        _functions["str_join"] = func;
        _functionTypes["str_join"] = funcType;
        _functions["join"] = func;
        _functionTypes["join"] = funcType;
    }

    private void DeclareStrCharAtFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // char_at(str, index) -> int (character code)
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { stringPtrType, LLVMTypeRef.Int64 });
        var func = _module.AddFunction("str_char_at", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var s = func.GetParam(0);
        var index = func.GetParam(1);
        var ptr = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, s, new[] { index }, "ptr");
        var c = funcBuilder.BuildLoad2(LLVMTypeRef.Int8, ptr, "c");
        var result = funcBuilder.BuildZExt(c, LLVMTypeRef.Int64, "result");
        funcBuilder.BuildRet(result);

        _functions["str_char_at"] = func;
        _functionTypes["str_char_at"] = funcType;
        _functions["char_at"] = func;
        _functionTypes["char_at"] = funcType;
    }

    private void DeclareStrIndexOfFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        // index_of(haystack, needle) -> int (-1 if not found)
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("str_index_of", funcType);
        var entry = func.AppendBasicBlock("entry");
        var found = func.AppendBasicBlock("found");
        var notFound = func.AppendBasicBlock("not_found");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var haystack = func.GetParam(0);
        var needle = func.GetParam(1);
        var result = funcBuilder.BuildCall2(_functionTypes["strstr"], _functions["strstr"], new[] { haystack, needle }, "found");
        var nullPtr = LLVMValueRef.CreateConstNull(stringPtrType);
        var isFound = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntNE, result, nullPtr, "isfound");
        funcBuilder.BuildCondBr(isFound, found, notFound);

        funcBuilder.PositionAtEnd(found);
        var index = funcBuilder.BuildPtrDiff2(LLVMTypeRef.Int8, result, haystack, "index");
        funcBuilder.BuildRet(index);

        funcBuilder.PositionAtEnd(notFound);
        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, unchecked((ulong)-1), true));

        _functions["str_index_of"] = func;
        _functionTypes["str_index_of"] = funcType;
        _functions["index_of"] = func;
        _functionTypes["index_of"] = funcType;
    }

    #endregion

    #region File I/O Functions

    private void DeclareFileIOFunctions()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var filePtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);  // FILE* as opaque pointer

        // Declare C runtime file functions
        // FILE* fopen(const char* filename, const char* mode)
        var fopenType = LLVMTypeRef.CreateFunction(filePtrType, new[] { stringPtrType, stringPtrType });
        var fopen = _module.AddFunction("fopen", fopenType);
        _functions["fopen"] = fopen;
        _functionTypes["fopen"] = fopenType;

        // int fclose(FILE* stream)
        var fcloseType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { filePtrType });
        var fclose = _module.AddFunction("fclose", fcloseType);
        _functions["fclose"] = fclose;
        _functionTypes["fclose"] = fcloseType;

        // size_t fread(void* ptr, size_t size, size_t count, FILE* stream)
        var freadType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { stringPtrType, LLVMTypeRef.Int64, LLVMTypeRef.Int64, filePtrType });
        var fread = _module.AddFunction("fread", freadType);
        _functions["fread"] = fread;
        _functionTypes["fread"] = freadType;

        // size_t fwrite(const void* ptr, size_t size, size_t count, FILE* stream)
        var fwriteType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { stringPtrType, LLVMTypeRef.Int64, LLVMTypeRef.Int64, filePtrType });
        var fwrite = _module.AddFunction("fwrite", fwriteType);
        _functions["fwrite"] = fwrite;
        _functionTypes["fwrite"] = fwriteType;

        // int fseek(FILE* stream, long offset, int whence)
        var fseekType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { filePtrType, LLVMTypeRef.Int64, LLVMTypeRef.Int32 });
        var fseek = _module.AddFunction("fseek", fseekType);
        _functions["fseek"] = fseek;
        _functionTypes["fseek"] = fseekType;

        // long ftell(FILE* stream)
        var ftellType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { filePtrType });
        var ftell = _module.AddFunction("ftell", ftellType);
        _functions["ftell"] = ftell;
        _functionTypes["ftell"] = ftellType;

        // void rewind(FILE* stream)
        var rewindType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Void, new[] { filePtrType });
        var rewind = _module.AddFunction("rewind", rewindType);
        _functions["rewind"] = rewind;
        _functionTypes["rewind"] = rewindType;

        // int remove(const char* filename)
        var removeType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { stringPtrType });
        var remove = _module.AddFunction("remove", removeType);
        _functions["remove"] = remove;
        _functionTypes["remove"] = removeType;

        // int rename(const char* oldname, const char* newname)
        var renameType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, new[] { stringPtrType, stringPtrType });
        var rename = _module.AddFunction("rename", renameType);
        _functions["rename"] = rename;
        _functionTypes["rename"] = renameType;

        // Now create our high-level NSL file functions
        DeclareReadFileFunction();
        DeclareWriteFileFunction();
        DeclareAppendFileFunction();
        DeclareFileExistsFunction();
        DeclareDeleteFileFunction();
    }

    /// <summary>
    /// read_file(path: string) -> string
    /// Reads entire file contents into a string
    /// </summary>
    private void DeclareReadFileFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType });
        var func = _module.AddFunction("read_file", funcType);
        var entry = func.AppendBasicBlock("entry");
        var readBlock = func.AppendBasicBlock("read");
        var errorBlock = func.AppendBasicBlock("error");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var path = func.GetParam(0);

        // Open file for reading: fopen(path, "rb")
        var modeStr = funcBuilder.BuildGlobalStringPtr("rb", "readmode");
        var file = funcBuilder.BuildCall2(_functionTypes["fopen"], _functions["fopen"], new[] { path, modeStr }, "file");

        // Check if file opened successfully
        var nullPtr = LLVMValueRef.CreateConstNull(stringPtrType);
        var isNull = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, file, nullPtr, "isnull");
        funcBuilder.BuildCondBr(isNull, errorBlock, readBlock);

        // Read block - file opened successfully
        funcBuilder.PositionAtEnd(readBlock);

        // Seek to end to get file size
        var seekEnd = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, 2);  // SEEK_END
        funcBuilder.BuildCall2(_functionTypes["fseek"], _functions["fseek"], new[] { file, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 0), seekEnd }, "");

        // Get file size
        var fileSize = funcBuilder.BuildCall2(_functionTypes["ftell"], _functions["ftell"], new[] { file }, "filesize");

        // Rewind to beginning
        funcBuilder.BuildCall2(_functionTypes["rewind"], _functions["rewind"], new[] { file }, "");

        // Allocate buffer (size + 1 for null terminator)
        var allocSize = funcBuilder.BuildAdd(fileSize, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), "allocsize");
        var buffer = funcBuilder.BuildCall2(_functionTypes["malloc"], _functions["malloc"], new[] { allocSize }, "buffer");

        // Read file contents
        funcBuilder.BuildCall2(_functionTypes["fread"], _functions["fread"],
            new[] { buffer, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), fileSize, file }, "");

        // Add null terminator
        var nullPos = funcBuilder.BuildGEP2(LLVMTypeRef.Int8, buffer, new[] { fileSize }, "nullpos");
        funcBuilder.BuildStore(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 0), nullPos);

        // Close file
        funcBuilder.BuildCall2(_functionTypes["fclose"], _functions["fclose"], new[] { file }, "");

        funcBuilder.BuildRet(buffer);

        // Error block - return empty string
        funcBuilder.PositionAtEnd(errorBlock);
        var emptyStr = funcBuilder.BuildGlobalStringPtr("", "emptystr");
        funcBuilder.BuildRet(emptyStr);

        _functions["read_file"] = func;
        _functionTypes["read_file"] = funcType;
    }

    /// <summary>
    /// write_file(path: string, content: string) -> bool
    /// Writes string to file, returns true on success
    /// </summary>
    private void DeclareWriteFileFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("write_file", funcType);
        var entry = func.AppendBasicBlock("entry");
        var writeBlock = func.AppendBasicBlock("write");
        var errorBlock = func.AppendBasicBlock("error");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var path = func.GetParam(0);
        var content = func.GetParam(1);

        // Open file for writing: fopen(path, "wb")
        var modeStr = funcBuilder.BuildGlobalStringPtr("wb", "writemode");
        var file = funcBuilder.BuildCall2(_functionTypes["fopen"], _functions["fopen"], new[] { path, modeStr }, "file");

        // Check if file opened successfully
        var nullPtr = LLVMValueRef.CreateConstNull(stringPtrType);
        var isNull = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, file, nullPtr, "isnull");
        funcBuilder.BuildCondBr(isNull, errorBlock, writeBlock);

        // Write block
        funcBuilder.PositionAtEnd(writeBlock);

        // Get content length using strlen
        var contentLen = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { content }, "len");

        // Write content
        funcBuilder.BuildCall2(_functionTypes["fwrite"], _functions["fwrite"],
            new[] { content, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), contentLen, file }, "");

        // Close file
        funcBuilder.BuildCall2(_functionTypes["fclose"], _functions["fclose"], new[] { file }, "");

        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 1));  // true

        // Error block
        funcBuilder.PositionAtEnd(errorBlock);
        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 0));  // false

        _functions["write_file"] = func;
        _functionTypes["write_file"] = funcType;
    }

    /// <summary>
    /// append_file(path: string, content: string) -> bool
    /// Appends string to file, returns true on success
    /// </summary>
    private void DeclareAppendFileFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType, stringPtrType });
        var func = _module.AddFunction("append_file", funcType);
        var entry = func.AppendBasicBlock("entry");
        var writeBlock = func.AppendBasicBlock("write");
        var errorBlock = func.AppendBasicBlock("error");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var path = func.GetParam(0);
        var content = func.GetParam(1);

        // Open file for appending: fopen(path, "ab")
        var modeStr = funcBuilder.BuildGlobalStringPtr("ab", "appendmode");
        var file = funcBuilder.BuildCall2(_functionTypes["fopen"], _functions["fopen"], new[] { path, modeStr }, "file");

        // Check if file opened successfully
        var nullPtr = LLVMValueRef.CreateConstNull(stringPtrType);
        var isNull = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, file, nullPtr, "isnull");
        funcBuilder.BuildCondBr(isNull, errorBlock, writeBlock);

        // Write block
        funcBuilder.PositionAtEnd(writeBlock);

        // Get content length using strlen
        var contentLen = funcBuilder.BuildCall2(_functionTypes["strlen"], _functions["strlen"], new[] { content }, "len");

        // Write content
        funcBuilder.BuildCall2(_functionTypes["fwrite"], _functions["fwrite"],
            new[] { content, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int64, 1), contentLen, file }, "");

        // Close file
        funcBuilder.BuildCall2(_functionTypes["fclose"], _functions["fclose"], new[] { file }, "");

        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 1));  // true

        // Error block
        funcBuilder.PositionAtEnd(errorBlock);
        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 0));  // false

        _functions["append_file"] = func;
        _functionTypes["append_file"] = funcType;
    }

    /// <summary>
    /// file_exists(path: string) -> bool
    /// Checks if file exists by trying to open it
    /// </summary>
    private void DeclareFileExistsFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType });
        var func = _module.AddFunction("file_exists", funcType);
        var entry = func.AppendBasicBlock("entry");
        var existsBlock = func.AppendBasicBlock("exists");
        var notExistsBlock = func.AppendBasicBlock("notexists");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var path = func.GetParam(0);

        // Try to open file for reading: fopen(path, "rb")
        var modeStr = funcBuilder.BuildGlobalStringPtr("rb", "readmode");
        var file = funcBuilder.BuildCall2(_functionTypes["fopen"], _functions["fopen"], new[] { path, modeStr }, "file");

        // Check if file opened successfully
        var nullPtr = LLVMValueRef.CreateConstNull(stringPtrType);
        var isNull = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, file, nullPtr, "isnull");
        funcBuilder.BuildCondBr(isNull, notExistsBlock, existsBlock);

        // File exists - close it and return true
        funcBuilder.PositionAtEnd(existsBlock);
        funcBuilder.BuildCall2(_functionTypes["fclose"], _functions["fclose"], new[] { file }, "");
        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 1));  // true

        // File doesn't exist
        funcBuilder.PositionAtEnd(notExistsBlock);
        funcBuilder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int1, 0));  // false

        _functions["file_exists"] = func;
        _functionTypes["file_exists"] = funcType;
    }

    /// <summary>
    /// delete_file(path: string) -> bool
    /// Deletes a file, returns true on success
    /// </summary>
    private void DeclareDeleteFileFunction()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType });
        var func = _module.AddFunction("delete_file", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var path = func.GetParam(0);

        // Call remove(path) - returns 0 on success
        var result = funcBuilder.BuildCall2(_functionTypes["remove"], _functions["remove"], new[] { path }, "result");

        // Convert to bool (0 means success -> true)
        var success = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, result, LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, 0), "success");
        funcBuilder.BuildRet(success);

        _functions["delete_file"] = func;
        _functionTypes["delete_file"] = funcType;
    }

    private LLVMValueRef GetOrDeclareIntrinsic(string name, LLVMTypeRef returnType, LLVMTypeRef[] paramTypes)
    {
        var func = _module.GetNamedFunction(name);
        if (func.Handle != IntPtr.Zero)
            return func;

        var funcType = LLVMTypeRef.CreateFunction(returnType, paramTypes);
        return _module.AddFunction(name, funcType);
    }

    #endregion

    #region Result/Optional Chaining Functions

    /// <summary>
    /// Declares all Result and Optional chaining functions
    /// Result type: { i8 tag, double value } where tag=1 for Ok, tag=0 for Err
    /// Optional type: { i8 tag, double value } where tag=1 for Some, tag=0 for None
    /// </summary>
    private void DeclareResultOptionalFunctions()
    {
        DeclareIsOkFunction();
        DeclareIsErrFunction();
        DeclareUnwrapFunction();
        DeclareUnwrapOrFunction();
        DeclareUnwrapErrFunction();
        DeclareExpectFunction();
        DeclareIsSomeFunction();
        DeclareIsNoneFunction();
        DeclareOptionUnwrapOrFunction();
        DeclareOptionExpectFunction();
    }

    /// <summary>
    /// is_ok(result) -> bool
    /// Returns true if Result contains Ok value
    /// </summary>
    private void DeclareIsOkFunction()
    {
        var resultType = GetResultType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { resultType });
        var func = _module.AddFunction("is_ok", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var result = func.GetParam(0);

        // Extract tag (field 0)
        var tag = funcBuilder.BuildExtractValue(result, 0, "tag");

        // Tag == 1 means Ok
        var one = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1);
        var isOk = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, one, "is_ok");
        funcBuilder.BuildRet(isOk);

        _functions["is_ok"] = func;
        _functionTypes["is_ok"] = funcType;
    }

    /// <summary>
    /// is_err(result) -> bool
    /// Returns true if Result contains Err value
    /// </summary>
    private void DeclareIsErrFunction()
    {
        var resultType = GetResultType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { resultType });
        var func = _module.AddFunction("is_err", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var result = func.GetParam(0);

        // Extract tag (field 0)
        var tag = funcBuilder.BuildExtractValue(result, 0, "tag");

        // Tag == 0 means Err
        var zero = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 0);
        var isErr = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, zero, "is_err");
        funcBuilder.BuildRet(isErr);

        _functions["is_err"] = func;
        _functionTypes["is_err"] = funcType;
    }

    /// <summary>
    /// unwrap(result) -> value
    /// Returns the Ok value or prints error and returns 0
    /// </summary>
    private void DeclareUnwrapFunction()
    {
        var resultType = GetResultType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { resultType });
        var func = _module.AddFunction("unwrap", funcType);
        var entry = func.AppendBasicBlock("entry");
        var okBlock = func.AppendBasicBlock("ok");
        var errBlock = func.AppendBasicBlock("err");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var result = func.GetParam(0);

        // Extract tag
        var tag = funcBuilder.BuildExtractValue(result, 0, "tag");
        var one = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1);
        var isOk = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, one, "is_ok");
        funcBuilder.BuildCondBr(isOk, okBlock, errBlock);

        // Ok block - return the value
        funcBuilder.PositionAtEnd(okBlock);
        var value = funcBuilder.BuildExtractValue(result, 1, "value");
        funcBuilder.BuildRet(value);

        // Err block - print error message and return 0
        funcBuilder.PositionAtEnd(errBlock);
        var errMsg = funcBuilder.BuildGlobalStringPtr("Error: called unwrap on Err value\n", "unwrap_err_msg");
        funcBuilder.BuildCall2(_functionTypes["printf"], _functions["printf"], new[] { errMsg }, "");
        funcBuilder.BuildRet(LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.0));

        _functions["unwrap"] = func;
        _functionTypes["unwrap"] = funcType;
    }

    /// <summary>
    /// unwrap_or(result, default) -> value
    /// Returns the Ok value or the default
    /// </summary>
    private void DeclareUnwrapOrFunction()
    {
        var resultType = GetResultType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { resultType, LLVMTypeRef.Double });
        var func = _module.AddFunction("unwrap_or", funcType);
        var entry = func.AppendBasicBlock("entry");
        var okBlock = func.AppendBasicBlock("ok");
        var errBlock = func.AppendBasicBlock("err");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var result = func.GetParam(0);
        var defaultVal = func.GetParam(1);

        // Extract tag
        var tag = funcBuilder.BuildExtractValue(result, 0, "tag");
        var one = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1);
        var isOk = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, one, "is_ok");
        funcBuilder.BuildCondBr(isOk, okBlock, errBlock);

        // Ok block - return the value
        funcBuilder.PositionAtEnd(okBlock);
        var value = funcBuilder.BuildExtractValue(result, 1, "value");
        funcBuilder.BuildRet(value);

        // Err block - return default
        funcBuilder.PositionAtEnd(errBlock);
        funcBuilder.BuildRet(defaultVal);

        _functions["unwrap_or"] = func;
        _functionTypes["unwrap_or"] = funcType;
        // Also register as result_ok_or
        _functions["result_ok_or"] = func;
        _functionTypes["result_ok_or"] = funcType;
    }

    /// <summary>
    /// unwrap_err(result) -> value
    /// Returns the Err value or prints error and returns 0
    /// </summary>
    private void DeclareUnwrapErrFunction()
    {
        var resultType = GetResultType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { resultType });
        var func = _module.AddFunction("unwrap_err", funcType);
        var entry = func.AppendBasicBlock("entry");
        var okBlock = func.AppendBasicBlock("ok");
        var errBlock = func.AppendBasicBlock("err");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var result = func.GetParam(0);

        // Extract tag
        var tag = funcBuilder.BuildExtractValue(result, 0, "tag");
        var zero = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 0);
        var isErr = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, zero, "is_err");
        funcBuilder.BuildCondBr(isErr, errBlock, okBlock);

        // Err block - return the error value
        funcBuilder.PositionAtEnd(errBlock);
        var value = funcBuilder.BuildExtractValue(result, 1, "value");
        funcBuilder.BuildRet(value);

        // Ok block - print error and return 0
        funcBuilder.PositionAtEnd(okBlock);
        var errMsg = funcBuilder.BuildGlobalStringPtr("Error: called unwrap_err on Ok value\n", "unwrap_err_msg");
        funcBuilder.BuildCall2(_functionTypes["printf"], _functions["printf"], new[] { errMsg }, "");
        funcBuilder.BuildRet(LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.0));

        _functions["unwrap_err"] = func;
        _functionTypes["unwrap_err"] = funcType;
    }

    /// <summary>
    /// expect(result, message) -> value
    /// Returns the Ok value or prints custom message and returns 0
    /// </summary>
    private void DeclareExpectFunction()
    {
        var resultType = GetResultType(LLVMTypeRef.Double);
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { resultType, stringPtrType });
        var func = _module.AddFunction("expect", funcType);
        var entry = func.AppendBasicBlock("entry");
        var okBlock = func.AppendBasicBlock("ok");
        var errBlock = func.AppendBasicBlock("err");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var result = func.GetParam(0);
        var message = func.GetParam(1);

        // Extract tag
        var tag = funcBuilder.BuildExtractValue(result, 0, "tag");
        var one = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1);
        var isOk = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, one, "is_ok");
        funcBuilder.BuildCondBr(isOk, okBlock, errBlock);

        // Ok block - return the value
        funcBuilder.PositionAtEnd(okBlock);
        var value = funcBuilder.BuildExtractValue(result, 1, "value");
        funcBuilder.BuildRet(value);

        // Err block - print custom message and return 0
        funcBuilder.PositionAtEnd(errBlock);
        var formatStr = funcBuilder.BuildGlobalStringPtr("Error: %s\n", "expect_fmt");
        funcBuilder.BuildCall2(_functionTypes["printf"], _functions["printf"], new[] { formatStr, message }, "");
        funcBuilder.BuildRet(LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.0));

        _functions["expect"] = func;
        _functionTypes["expect"] = funcType;
    }

    /// <summary>
    /// is_some(optional) -> bool
    /// Returns true if Optional contains Some value
    /// </summary>
    private void DeclareIsSomeFunction()
    {
        var optionalType = GetOptionalType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { optionalType });
        var func = _module.AddFunction("is_some", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var optional = func.GetParam(0);

        // Extract tag (field 0)
        var tag = funcBuilder.BuildExtractValue(optional, 0, "tag");

        // Tag == 1 means Some
        var one = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1);
        var isSome = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, one, "is_some");
        funcBuilder.BuildRet(isSome);

        _functions["is_some"] = func;
        _functionTypes["is_some"] = funcType;
    }

    /// <summary>
    /// is_none(optional) -> bool
    /// Returns true if Optional is None
    /// </summary>
    private void DeclareIsNoneFunction()
    {
        var optionalType = GetOptionalType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { optionalType });
        var func = _module.AddFunction("is_none", funcType);
        var entry = func.AppendBasicBlock("entry");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var optional = func.GetParam(0);

        // Extract tag (field 0)
        var tag = funcBuilder.BuildExtractValue(optional, 0, "tag");

        // Tag == 0 means None
        var zero = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 0);
        var isNone = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, zero, "is_none");
        funcBuilder.BuildRet(isNone);

        _functions["is_none"] = func;
        _functionTypes["is_none"] = funcType;
    }

    /// <summary>
    /// option_unwrap_or(optional, default) -> value
    /// Returns the Some value or the default
    /// </summary>
    private void DeclareOptionUnwrapOrFunction()
    {
        var optionalType = GetOptionalType(LLVMTypeRef.Double);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { optionalType, LLVMTypeRef.Double });
        var func = _module.AddFunction("option_unwrap_or", funcType);
        var entry = func.AppendBasicBlock("entry");
        var someBlock = func.AppendBasicBlock("some");
        var noneBlock = func.AppendBasicBlock("none");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var optional = func.GetParam(0);
        var defaultVal = func.GetParam(1);

        // Extract tag
        var tag = funcBuilder.BuildExtractValue(optional, 0, "tag");
        var one = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1);
        var isSome = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, one, "is_some");
        funcBuilder.BuildCondBr(isSome, someBlock, noneBlock);

        // Some block - return the value
        funcBuilder.PositionAtEnd(someBlock);
        var value = funcBuilder.BuildExtractValue(optional, 1, "value");
        funcBuilder.BuildRet(value);

        // None block - return default
        funcBuilder.PositionAtEnd(noneBlock);
        funcBuilder.BuildRet(defaultVal);

        _functions["option_unwrap_or"] = func;
        _functionTypes["option_unwrap_or"] = funcType;
    }

    /// <summary>
    /// option_expect(optional, message) -> value
    /// Returns the Some value or prints custom message and returns 0
    /// </summary>
    private void DeclareOptionExpectFunction()
    {
        var optionalType = GetOptionalType(LLVMTypeRef.Double);
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var funcType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { optionalType, stringPtrType });
        var func = _module.AddFunction("option_expect", funcType);
        var entry = func.AppendBasicBlock("entry");
        var someBlock = func.AppendBasicBlock("some");
        var noneBlock = func.AppendBasicBlock("none");

        using var funcBuilder = _context.CreateBuilder();
        funcBuilder.PositionAtEnd(entry);

        var optional = func.GetParam(0);
        var message = func.GetParam(1);

        // Extract tag
        var tag = funcBuilder.BuildExtractValue(optional, 0, "tag");
        var one = LLVMValueRef.CreateConstInt(LLVMTypeRef.Int8, 1);
        var isSome = funcBuilder.BuildICmp(LLVMIntPredicate.LLVMIntEQ, tag, one, "is_some");
        funcBuilder.BuildCondBr(isSome, someBlock, noneBlock);

        // Some block - return the value
        funcBuilder.PositionAtEnd(someBlock);
        var value = funcBuilder.BuildExtractValue(optional, 1, "value");
        funcBuilder.BuildRet(value);

        // None block - print custom message and return 0
        funcBuilder.PositionAtEnd(noneBlock);
        var formatStr = funcBuilder.BuildGlobalStringPtr("Error: %s\n", "expect_fmt");
        funcBuilder.BuildCall2(_functionTypes["printf"], _functions["printf"], new[] { formatStr, message }, "");
        funcBuilder.BuildRet(LLVMValueRef.CreateConstReal(LLVMTypeRef.Double, 0.0));

        _functions["option_expect"] = func;
        _functionTypes["option_expect"] = funcType;
    }

    #endregion

    #region HTTP Client Functions

    /// <summary>
    /// Declares HTTP client functions
    /// These map to external C functions that must be linked
    /// </summary>
    private void DeclareHTTPFunctions()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var resultType = GetResultType(stringPtrType);

        // http_get(url: string) -> Result<string, string>
        var getType = LLVMTypeRef.CreateFunction(resultType, new[] { stringPtrType });
        var httpGet = _module.AddFunction("nsl_http_get", getType);
        _functions["http_get"] = httpGet;
        _functionTypes["http_get"] = getType;

        // http_post(url: string, body: string) -> Result<string, string>
        var postType = LLVMTypeRef.CreateFunction(resultType, new[] { stringPtrType, stringPtrType });
        var httpPost = _module.AddFunction("nsl_http_post", postType);
        _functions["http_post"] = httpPost;
        _functionTypes["http_post"] = postType;

        // http_put(url: string, body: string) -> Result<string, string>
        var putType = LLVMTypeRef.CreateFunction(resultType, new[] { stringPtrType, stringPtrType });
        var httpPut = _module.AddFunction("nsl_http_put", putType);
        _functions["http_put"] = httpPut;
        _functionTypes["http_put"] = putType;

        // http_delete(url: string) -> Result<string, string>
        var deleteType = LLVMTypeRef.CreateFunction(resultType, new[] { stringPtrType });
        var httpDelete = _module.AddFunction("nsl_http_delete", deleteType);
        _functions["http_delete"] = httpDelete;
        _functionTypes["http_delete"] = deleteType;

        // url_encode(str: string) -> string
        var encodeType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType });
        var urlEncode = _module.AddFunction("nsl_url_encode", encodeType);
        _functions["url_encode"] = urlEncode;
        _functionTypes["url_encode"] = encodeType;

        // url_decode(str: string) -> string
        var decodeType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType });
        var urlDecode = _module.AddFunction("nsl_url_decode", decodeType);
        _functions["url_decode"] = urlDecode;
        _functionTypes["url_decode"] = decodeType;
    }

    #endregion

    #region JSON Functions

    /// <summary>
    /// Declares JSON parsing and serialization functions
    /// These map to external C functions that must be linked
    /// </summary>
    private void DeclareJSONFunctions()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);
        var resultType = GetResultType(stringPtrType);

        // json_parse(json: string) -> Result<any, string>
        var parseType = LLVMTypeRef.CreateFunction(resultType, new[] { stringPtrType });
        var jsonParse = _module.AddFunction("nsl_json_parse", parseType);
        _functions["json_parse"] = jsonParse;
        _functionTypes["json_parse"] = parseType;

        // json_stringify(value: any) -> string
        var stringifyType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType });
        var jsonStringify = _module.AddFunction("nsl_json_stringify", stringifyType);
        _functions["json_stringify"] = jsonStringify;
        _functionTypes["json_stringify"] = stringifyType;

        // json_get(obj: any, key: string) -> any
        var getType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType });
        var jsonGet = _module.AddFunction("nsl_json_get", getType);
        _functions["json_get"] = jsonGet;
        _functionTypes["json_get"] = getType;

        // json_set(obj: any, key: string, value: any) -> any
        var setType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType, stringPtrType });
        var jsonSet = _module.AddFunction("nsl_json_set", setType);
        _functions["json_set"] = jsonSet;
        _functionTypes["json_set"] = setType;

        // json_has(obj: any, key: string) -> bool
        var hasType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType, stringPtrType });
        var jsonHas = _module.AddFunction("nsl_json_has", hasType);
        _functions["json_has"] = jsonHas;
        _functionTypes["json_has"] = hasType;
    }

    #endregion

    #region Regex Functions

    /// <summary>
    /// Declares regex functions for pattern matching
    /// These map to external C functions that must be linked
    /// </summary>
    private void DeclareRegexFunctions()
    {
        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);

        // regex_match(str: string, pattern: string) -> bool
        var matchType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int1, new[] { stringPtrType, stringPtrType });
        var regexMatch = _module.AddFunction("nsl_regex_match", matchType);
        _functions["regex_match"] = regexMatch;
        _functionTypes["regex_match"] = matchType;

        // regex_find(str: string, pattern: string) -> Optional<string>
        var optionalType = GetOptionalType(stringPtrType);
        var findType = LLVMTypeRef.CreateFunction(optionalType, new[] { stringPtrType, stringPtrType });
        var regexFind = _module.AddFunction("nsl_regex_find", findType);
        _functions["regex_find"] = regexFind;
        _functionTypes["regex_find"] = findType;

        // regex_replace(str: string, pattern: string, replacement: string) -> string
        var replaceType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { stringPtrType, stringPtrType, stringPtrType });
        var regexReplace = _module.AddFunction("nsl_regex_replace", replaceType);
        _functions["regex_replace"] = regexReplace;
        _functionTypes["regex_replace"] = replaceType;

        // regex_split(str: string, pattern: string) -> [string]
        var arrayType = LLVMTypeRef.CreatePointer(stringPtrType, 0);
        var splitType = LLVMTypeRef.CreateFunction(arrayType, new[] { stringPtrType, stringPtrType });
        var regexSplit = _module.AddFunction("nsl_regex_split", splitType);
        _functions["regex_split"] = regexSplit;
        _functionTypes["regex_split"] = splitType;
    }

    #endregion

    #region DateTime Functions

    /// <summary>
    /// Declares date/time functions
    /// </summary>
    private void DeclareDateTimeFunctions()
    {
        // now() -> number (unix timestamp in seconds)
        var nowType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, Array.Empty<LLVMTypeRef>());
        var now = _module.AddFunction("nsl_now", nowType);
        _functions["now"] = now;
        _functionTypes["now"] = nowType;

        // now_ms() -> number (unix timestamp in milliseconds)
        var nowMsType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, Array.Empty<LLVMTypeRef>());
        var nowMs = _module.AddFunction("nsl_now_ms", nowMsType);
        _functions["now_ms"] = nowMs;
        _functionTypes["now_ms"] = nowMsType;

        var stringPtrType = LLVMTypeRef.CreatePointer(LLVMTypeRef.Int8, 0);

        // date_parse(str: string) -> number (unix timestamp)
        var parseType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { stringPtrType });
        var dateParse = _module.AddFunction("nsl_date_parse", parseType);
        _functions["date_parse"] = dateParse;
        _functionTypes["date_parse"] = parseType;

        // date_format(timestamp: number, format: string) -> string
        var formatType = LLVMTypeRef.CreateFunction(stringPtrType, new[] { LLVMTypeRef.Double, stringPtrType });
        var dateFormat = _module.AddFunction("nsl_date_format", formatType);
        _functions["date_format"] = dateFormat;
        _functionTypes["date_format"] = formatType;

        // Date component extraction functions
        var extractType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int64, new[] { LLVMTypeRef.Double });

        var dateYear = _module.AddFunction("nsl_date_year", extractType);
        _functions["date_year"] = dateYear;
        _functionTypes["date_year"] = extractType;

        var dateMonth = _module.AddFunction("nsl_date_month", extractType);
        _functions["date_month"] = dateMonth;
        _functionTypes["date_month"] = extractType;

        var dateDay = _module.AddFunction("nsl_date_day", extractType);
        _functions["date_day"] = dateDay;
        _functionTypes["date_day"] = extractType;

        var dateHour = _module.AddFunction("nsl_date_hour", extractType);
        _functions["date_hour"] = dateHour;
        _functionTypes["date_hour"] = extractType;

        var dateMinute = _module.AddFunction("nsl_date_minute", extractType);
        _functions["date_minute"] = dateMinute;
        _functionTypes["date_minute"] = extractType;

        var dateSecond = _module.AddFunction("nsl_date_second", extractType);
        _functions["date_second"] = dateSecond;
        _functionTypes["date_second"] = extractType;

        var dateWeekday = _module.AddFunction("nsl_date_weekday", extractType);
        _functions["date_weekday"] = dateWeekday;
        _functionTypes["date_weekday"] = extractType;

        // Date arithmetic
        var addType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double, LLVMTypeRef.Int64 });

        var addDays = _module.AddFunction("nsl_date_add_days", addType);
        _functions["date_add_days"] = addDays;
        _functionTypes["date_add_days"] = addType;

        var addHours = _module.AddFunction("nsl_date_add_hours", addType);
        _functions["date_add_hours"] = addHours;
        _functionTypes["date_add_hours"] = addType;

        // date_diff(timestamp1, timestamp2) -> seconds
        var diffType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Double, new[] { LLVMTypeRef.Double, LLVMTypeRef.Double });
        var dateDiff = _module.AddFunction("nsl_date_diff", diffType);
        _functions["date_diff"] = dateDiff;
        _functionTypes["date_diff"] = diffType;

        // sleep(milliseconds: number) -> void
        var sleepType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Void, new[] { LLVMTypeRef.Double });
        var sleep = _module.AddFunction("nsl_sleep", sleepType);
        _functions["sleep"] = sleep;
        _functionTypes["sleep"] = sleepType;
    }

    #endregion

    #region Module Output

    /// <summary>
    /// Generate main function that calls the user's entry point
    /// </summary>
    public void GenerateMain(string entryPoint = "main")
    {
        if (_functions.ContainsKey(entryPoint) && _functionTypes.ContainsKey(entryPoint))
        {
            // Create C-style main that calls the NSL entry point
            var mainType = LLVMTypeRef.CreateFunction(LLVMTypeRef.Int32, Array.Empty<LLVMTypeRef>());
            var main = _module.AddFunction("main", mainType);
            var entry = main.AppendBasicBlock("entry");

            _builder.PositionAtEnd(entry);

            var nslMain = _functions[entryPoint];
            var nslMainType = _functionTypes[entryPoint];
            _builder.BuildCall2(nslMainType, nslMain, Array.Empty<LLVMValueRef>(), "");

            _builder.BuildRet(LLVMValueRef.CreateConstInt(LLVMTypeRef.Int32, 0));
        }
    }

    /// <summary>
    /// Get the generated LLVM IR as a string
    /// </summary>
    public string GetIR()
    {
        return _module.PrintToString();
    }

    /// <summary>
    /// Write LLVM IR to a file
    /// </summary>
    public void WriteIR(string path)
    {
        _module.PrintToFile(path);
    }

    /// <summary>
    /// Write LLVM bitcode to a file
    /// </summary>
    public void WriteBitcode(string path)
    {
        _module.WriteBitcodeToFile(path);
    }

    /// <summary>
    /// Verify the module for errors
    /// </summary>
    public bool Verify(out string error)
    {
        return _module.TryVerify(LLVMVerifierFailureAction.LLVMReturnStatusAction, out error);
    }

    public void Dispose()
    {
        _builder.Dispose();
        _module.Dispose();
        // Don't dispose global context
    }

    #endregion
}
