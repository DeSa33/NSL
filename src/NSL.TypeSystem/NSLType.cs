using System;
using System.Collections.Generic;
using System.Linq;

namespace NSL.TypeSystem;

/// <summary>
/// Base class for all NSL types
/// NSL has a rich type system designed for AI/ML workloads
/// </summary>
public abstract class NSLType
{
    public abstract string Name { get; }
    public abstract bool IsAssignableTo(NSLType target);
    public virtual bool IsNumeric => false;
    public virtual bool IsCallable => false;
    public virtual int SizeInBits => 64; // Default to 64-bit

    public override string ToString() => Name;

    public override bool Equals(object? obj) =>
        obj is NSLType other && Name == other.Name;

    public override int GetHashCode() => Name.GetHashCode();
}

#region Primitive Types

/// <summary>
/// 64-bit floating point number (default numeric type for AI)
/// </summary>
public class NSLNumberType : NSLType
{
    public static readonly NSLNumberType Instance = new();
    public override string Name => "number";
    public override bool IsNumeric => true;
    public override int SizeInBits => 64;

    public override bool IsAssignableTo(NSLType target) =>
        target is NSLNumberType || target is NSLAnyType;

    private NSLNumberType() { }
}

/// <summary>
/// 32-bit integer for indexing and counting
/// </summary>
public class NSLIntType : NSLType
{
    public static readonly NSLIntType Instance = new();
    public override string Name => "int";
    public override bool IsNumeric => true;
    public override int SizeInBits => 32;

    public override bool IsAssignableTo(NSLType target) =>
        target is NSLIntType || target is NSLNumberType || target is NSLAnyType;

    private NSLIntType() { }
}

/// <summary>
/// Boolean type
/// </summary>
public class NSLBoolType : NSLType
{
    public static readonly NSLBoolType Instance = new();
    public override string Name => "bool";
    public override int SizeInBits => 1;

    public override bool IsAssignableTo(NSLType target) =>
        target is NSLBoolType || target is NSLAnyType;

    private NSLBoolType() { }
}

/// <summary>
/// String type (UTF-8)
/// </summary>
public class NSLStringType : NSLType
{
    public static readonly NSLStringType Instance = new();
    public override string Name => "string";
    public override int SizeInBits => 64; // Pointer size

    public override bool IsAssignableTo(NSLType target) =>
        target is NSLStringType || target is NSLAnyType;

    private NSLStringType() { }
}

/// <summary>
/// Void type (for functions with no return value)
/// </summary>
public class NSLVoidType : NSLType
{
    public static readonly NSLVoidType Instance = new();
    public override string Name => "void";
    public override int SizeInBits => 0;

    public override bool IsAssignableTo(NSLType target) =>
        target is NSLVoidType;

    private NSLVoidType() { }
}

/// <summary>
/// Null type
/// </summary>
public class NSLNullType : NSLType
{
    public static readonly NSLNullType Instance = new();
    public override string Name => "null";
    public override int SizeInBits => 0;

    public override bool IsAssignableTo(NSLType target) =>
        target is NSLNullType || target is NSLOptionalType || target is NSLAnyType;

    private NSLNullType() { }
}

/// <summary>
/// Any type (top type - accepts anything)
/// </summary>
public class NSLAnyType : NSLType
{
    public static readonly NSLAnyType Instance = new();
    public override string Name => "any";

    public override bool IsAssignableTo(NSLType target) => true;

    private NSLAnyType() { }
}

#endregion

#region AI/ML Types

/// <summary>
/// Vector type - 1D array of numbers
/// Core type for AI/ML operations
/// </summary>
public class NSLVecType : NSLType
{
    public int? Length { get; }
    public override string Name => Length.HasValue ? $"vec[{Length}]" : "vec";
    public override bool IsNumeric => true;

    public NSLVecType(int? length = null)
    {
        Length = length;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLVecType otherVec)
        {
            // vec[n] is assignable to vec (unknown length)
            if (!otherVec.Length.HasValue) return true;
            // vec[n] is assignable to vec[n]
            return Length == otherVec.Length;
        }
        return false;
    }
}

/// <summary>
/// Matrix type - 2D array of numbers
/// Core type for neural network weights
/// </summary>
public class NSLMatType : NSLType
{
    public int? Rows { get; }
    public int? Cols { get; }
    public override string Name =>
        (Rows.HasValue && Cols.HasValue) ? $"mat[{Rows},{Cols}]" : "mat";
    public override bool IsNumeric => true;

    public NSLMatType(int? rows = null, int? cols = null)
    {
        Rows = rows;
        Cols = cols;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLMatType otherMat)
        {
            if (!otherMat.Rows.HasValue && !otherMat.Cols.HasValue) return true;
            return Rows == otherMat.Rows && Cols == otherMat.Cols;
        }
        return false;
    }
}

/// <summary>
/// Tensor type - N-dimensional array
/// For deep learning operations
/// </summary>
public class NSLTensorType : NSLType
{
    public int[]? Shape { get; }
    public override string Name =>
        Shape != null ? $"tensor[{string.Join(",", Shape)}]" : "tensor";
    public override bool IsNumeric => true;

    public NSLTensorType(int[]? shape = null)
    {
        Shape = shape;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLTensorType otherTensor)
        {
            if (otherTensor.Shape == null) return true;
            if (Shape == null) return false;
            return Shape.SequenceEqual(otherTensor.Shape);
        }
        return false;
    }
}

/// <summary>
/// Probability type - number constrained to [0, 1]
/// AI-friendly: prevents probability errors
/// </summary>
public class NSLProbType : NSLType
{
    public static readonly NSLProbType Instance = new();
    public override string Name => "prob";
    public override bool IsNumeric => true;
    public override int SizeInBits => 64;

    public override bool IsAssignableTo(NSLType target) =>
        target is NSLProbType || target is NSLNumberType || target is NSLAnyType;

    private NSLProbType() { }
}

#endregion

#region Composite Types

/// <summary>
/// Function type
/// </summary>
public class NSLFunctionType : NSLType
{
    public IReadOnlyList<NSLType> ParameterTypes { get; }
    public NSLType ReturnType { get; }
    public override string Name =>
        $"fn({string.Join(", ", ParameterTypes)}) -> {ReturnType}";
    public override bool IsCallable => true;

    public NSLFunctionType(IEnumerable<NSLType> parameterTypes, NSLType returnType)
    {
        ParameterTypes = parameterTypes.ToList();
        ReturnType = returnType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLFunctionType otherFunc)
        {
            if (ParameterTypes.Count != otherFunc.ParameterTypes.Count) return false;
            for (int i = 0; i < ParameterTypes.Count; i++)
            {
                // Contravariant parameters
                if (!otherFunc.ParameterTypes[i].IsAssignableTo(ParameterTypes[i]))
                    return false;
            }
            // Covariant return type
            return ReturnType.IsAssignableTo(otherFunc.ReturnType);
        }
        return false;
    }
}

/// <summary>
/// Array type - homogeneous collection
/// </summary>
public class NSLArrayType : NSLType
{
    public NSLType ElementType { get; }
    public override string Name => $"[{ElementType}]";

    public NSLArrayType(NSLType elementType)
    {
        ElementType = elementType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLArrayType otherArray)
        {
            return ElementType.IsAssignableTo(otherArray.ElementType);
        }
        return false;
    }
}

/// <summary>
/// Optional type - value may or may not be present
/// AI Error Prevention: explicit null handling
/// </summary>
public class NSLOptionalType : NSLType
{
    public NSLType InnerType { get; }
    public override string Name => $"{InnerType}?";

    public NSLOptionalType(NSLType innerType)
    {
        InnerType = innerType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLOptionalType otherOptional)
        {
            return InnerType.IsAssignableTo(otherOptional.InnerType);
        }
        // T? is not assignable to T (must unwrap)
        return false;
    }
}

/// <summary>
/// Result type - either success (Ok) or error (Err)
/// AI Error Prevention: explicit error handling
/// </summary>
public class NSLResultType : NSLType
{
    public NSLType OkType { get; }
    public NSLType ErrType { get; }
    public override string Name => $"Result<{OkType}, {ErrType}>";

    public NSLResultType(NSLType okType, NSLType errType)
    {
        OkType = okType;
        ErrType = errType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLResultType otherResult)
        {
            return OkType.IsAssignableTo(otherResult.OkType) &&
                   ErrType.IsAssignableTo(otherResult.ErrType);
        }
        return false;
    }
}

/// <summary>
/// Range type - represents a range of values
/// </summary>
public class NSLRangeType : NSLType
{
    public NSLType ElementType { get; }
    public bool IsInclusive { get; }
    public override string Name => IsInclusive ? $"range<={ElementType}>" : $"range<{ElementType}>";

    public NSLRangeType(NSLType elementType, bool isInclusive)
    {
        ElementType = elementType;
        IsInclusive = isInclusive;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLRangeType otherRange)
        {
            return ElementType.IsAssignableTo(otherRange.ElementType);
        }
        return false;
    }
}

/// <summary>
/// Object/Record type - named fields
/// </summary>
public class NSLObjectType : NSLType
{
    public IReadOnlyDictionary<string, NSLType> Fields { get; }
    public string? TypeName { get; }
    public override string Name =>
        TypeName ?? $"{{{string.Join(", ", Fields.Select(f => $"{f.Key}: {f.Value}"))}}}";

    public NSLObjectType(Dictionary<string, NSLType> fields, string? typeName = null)
    {
        Fields = fields;
        TypeName = typeName;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLObjectType otherObject)
        {
            // Structural typing: all fields of target must exist in this
            foreach (var field in otherObject.Fields)
            {
                if (!Fields.TryGetValue(field.Key, out var thisField))
                    return false;
                if (!thisField.IsAssignableTo(field.Value))
                    return false;
            }
            return true;
        }
        return false;
    }
}

/// <summary>
/// Struct type - named struct with typed fields
/// AI-friendly: Clear structured data definitions
/// </summary>
public class NSLStructType : NSLType
{
    public string StructName { get; }
    public IReadOnlyDictionary<string, NSLType> Fields { get; }
    public override string Name => StructName;

    public NSLStructType(string name, Dictionary<string, NSLType> fields)
    {
        StructName = name;
        Fields = fields;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLStructType otherStruct)
        {
            // Nominal typing: must be same struct name
            return StructName == otherStruct.StructName;
        }
        if (target is NSLObjectType otherObject)
        {
            // Struct can be used where object type with matching fields is expected
            foreach (var field in otherObject.Fields)
            {
                if (!Fields.TryGetValue(field.Key, out var thisField))
                    return false;
                if (!thisField.IsAssignableTo(field.Value))
                    return false;
            }
            return true;
        }
        return false;
    }
}

/// <summary>
/// Enum variant type - represents a single variant of an enum
/// </summary>
public class NSLEnumVariantType
{
    public string Name { get; }
    public IReadOnlyList<NSLType>? FieldTypes { get; }
    public bool IsUnit => FieldTypes == null || FieldTypes.Count == 0;

    public NSLEnumVariantType(string name, IEnumerable<NSLType>? fieldTypes = null)
    {
        Name = name;
        FieldTypes = fieldTypes?.ToList();
    }

    public override string ToString()
    {
        if (IsUnit)
            return Name;
        return $"{Name}({string.Join(", ", FieldTypes!)})";
    }
}

/// <summary>
/// Enum type - algebraic data type with variants
/// AI-friendly: Exhaustive pattern matching, no null pointer issues
/// Example: enum Result { Ok(T), Err(E) }
/// Example: enum Shape { Circle(number), Rectangle(number, number), Point }
/// </summary>
public class NSLEnumType : NSLType
{
    public string EnumName { get; }
    public IReadOnlyDictionary<string, NSLEnumVariantType> Variants { get; }
    public override string Name => EnumName;

    public NSLEnumType(string name, Dictionary<string, NSLEnumVariantType> variants)
    {
        EnumName = name;
        Variants = variants;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLEnumType otherEnum)
        {
            // Nominal typing: must be same enum name
            return EnumName == otherEnum.EnumName;
        }
        return false;
    }

    /// <summary>
    /// Get a specific variant by name
    /// </summary>
    public NSLEnumVariantType? GetVariant(string variantName)
    {
        return Variants.TryGetValue(variantName, out var variant) ? variant : null;
    }
}

/// <summary>
/// Trait method signature type
/// </summary>
public class NSLTraitMethodType
{
    public string Name { get; }
    public IReadOnlyList<NSLType> ParameterTypes { get; }
    public NSLType ReturnType { get; }

    public NSLTraitMethodType(string name, IEnumerable<NSLType> parameterTypes, NSLType returnType)
    {
        Name = name;
        ParameterTypes = parameterTypes.ToList();
        ReturnType = returnType;
    }

    public override string ToString()
    {
        return $"fn {Name}({string.Join(", ", ParameterTypes)}) -> {ReturnType}";
    }
}

/// <summary>
/// Trait type - defines a behavioral contract
/// AI-friendly: Enables polymorphism and generic programming
/// Example: trait Printable { fn print(self); }
/// </summary>
public class NSLTraitType : NSLType
{
    public string TraitName { get; }
    public IReadOnlyDictionary<string, NSLTraitMethodType> Methods { get; }
    public override string Name => $"trait {TraitName}";

    public NSLTraitType(string name, Dictionary<string, NSLTraitMethodType> methods)
    {
        TraitName = name;
        Methods = methods;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLTraitType otherTrait)
        {
            // Nominal typing: must be same trait name
            return TraitName == otherTrait.TraitName;
        }
        return false;
    }

    /// <summary>
    /// Check if a type implements this trait
    /// </summary>
    public bool IsImplementedBy(NSLType type, Dictionary<(string TraitName, string TypeName), HashSet<string>> implementations)
    {
        var key = (TraitName, type.Name);
        return implementations.TryGetValue(key, out var methods) &&
               Methods.Keys.All(m => methods.Contains(m));
    }
}

/// <summary>
/// Future/Promise type - represents an async operation that will complete later
/// AI-friendly: Enables non-blocking concurrent programming
/// Example: Future<number> represents an async operation that will return a number
/// </summary>
public class NSLFutureType : NSLType
{
    public NSLType InnerType { get; }
    public override string Name => $"Future<{InnerType}>";

    public NSLFutureType(NSLType innerType)
    {
        InnerType = innerType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLFutureType otherFuture)
        {
            return InnerType.IsAssignableTo(otherFuture.InnerType);
        }
        return false;
    }
}

#endregion

#region Ownership Types (Rust-inspired)

/// <summary>
/// Owned type - value is owned and will be dropped when out of scope
/// AI Safety: Prevents use-after-free and double-free bugs
/// </summary>
public class NSLOwnedType : NSLType
{
    public NSLType InnerType { get; }
    public override string Name => $"owned<{InnerType}>";

    public NSLOwnedType(NSLType innerType)
    {
        InnerType = innerType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLOwnedType otherOwned)
            return InnerType.IsAssignableTo(otherOwned.InnerType);
        // Can move to non-owned (consume the value)
        return InnerType.IsAssignableTo(target);
    }
}

/// <summary>
/// Borrowed reference type - read-only borrow
/// AI Safety: Prevents mutation during borrow
/// </summary>
public class NSLBorrowType : NSLType
{
    public NSLType InnerType { get; }
    public override string Name => $"&{InnerType}";

    public NSLBorrowType(NSLType innerType)
    {
        InnerType = innerType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLBorrowType otherBorrow)
            return InnerType.IsAssignableTo(otherBorrow.InnerType);
        return false; // Cannot assign borrow to owned
    }
}

/// <summary>
/// Mutable borrow type - exclusive mutable access
/// AI Safety: Prevents data races at compile time
/// </summary>
public class NSLMutBorrowType : NSLType
{
    public NSLType InnerType { get; }
    public override string Name => $"&mut {InnerType}";

    public NSLMutBorrowType(NSLType innerType)
    {
        InnerType = innerType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLMutBorrowType otherMut)
            return InnerType.IsAssignableTo(otherMut.InnerType);
        // Mutable borrow can be used as immutable
        if (target is NSLBorrowType otherBorrow)
            return InnerType.IsAssignableTo(otherBorrow.InnerType);
        return false;
    }
}

#endregion

#region GPU/Accelerator Types (Mojo-inspired)

/// <summary>
/// GPU Kernel type - function that runs on GPU
/// AI-native: First-class GPU compute
/// </summary>
public class NSLKernelType : NSLType
{
    public IReadOnlyList<NSLType> ParameterTypes { get; }
    public NSLType ReturnType { get; }
    public int BlockSize { get; }
    public override string Name =>
        $"@gpu fn({string.Join(", ", ParameterTypes)}) -> {ReturnType}";
    public override bool IsCallable => true;

    public NSLKernelType(IEnumerable<NSLType> parameterTypes, NSLType returnType, int blockSize = 256)
    {
        ParameterTypes = parameterTypes.ToList();
        ReturnType = returnType;
        BlockSize = blockSize;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLKernelType otherKernel)
        {
            if (ParameterTypes.Count != otherKernel.ParameterTypes.Count) return false;
            for (int i = 0; i < ParameterTypes.Count; i++)
            {
                if (!otherKernel.ParameterTypes[i].IsAssignableTo(ParameterTypes[i]))
                    return false;
            }
            return ReturnType.IsAssignableTo(otherKernel.ReturnType);
        }
        return false;
    }
}

/// <summary>
/// Device tensor - tensor that lives on GPU/accelerator
/// AI-native: Explicit device placement
/// </summary>
public class NSLDeviceTensorType : NSLType
{
    public int[]? Shape { get; }
    public string Device { get; } // "gpu", "tpu", "cpu"
    public override string Name =>
        Shape != null ? $"tensor[{string.Join(",", Shape)}]@{Device}" : $"tensor@{Device}";
    public override bool IsNumeric => true;

    public NSLDeviceTensorType(string device = "gpu", int[]? shape = null)
    {
        Device = device;
        Shape = shape;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLDeviceTensorType otherDev)
        {
            // Same device or target is cpu (can always copy to cpu)
            if (Device != otherDev.Device && otherDev.Device != "cpu") return false;
            if (otherDev.Shape == null) return true;
            if (Shape == null) return false;
            return Shape.SequenceEqual(otherDev.Shape);
        }
        // DeviceTensor can be used where Tensor is expected
        if (target is NSLTensorType otherTensor)
        {
            if (otherTensor.Shape == null) return true;
            if (Shape == null) return false;
            return Shape.SequenceEqual(otherTensor.Shape);
        }
        return false;
    }
}

#endregion

#region Autodiff Type (ML-native)

/// <summary>
/// Differentiable type - supports automatic differentiation
/// AI-native: Built-in backprop support
/// </summary>
public class NSLDiffType : NSLType
{
    public NSLType InnerType { get; }
    public override string Name => $"diff<{InnerType}>";
    public override bool IsNumeric => InnerType.IsNumeric;

    public NSLDiffType(NSLType innerType)
    {
        InnerType = innerType;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLDiffType otherDiff)
            return InnerType.IsAssignableTo(otherDiff.InnerType);
        // Diff types can be used where non-diff is expected
        return InnerType.IsAssignableTo(target);
    }
}

#endregion

#region Module Types

/// <summary>
/// Module type - represents an imported module as a namespace
/// AI-friendly: Enables clean organization of code
/// </summary>
public class NSLModuleType : NSLType
{
    public string ModulePath { get; }
    public override string Name => $"module<{ModulePath}>";

    public NSLModuleType(string modulePath)
    {
        ModulePath = modulePath;
    }

    public override bool IsAssignableTo(NSLType target)
    {
        if (target is NSLAnyType) return true;
        if (target is NSLModuleType otherModule)
            return ModulePath == otherModule.ModulePath;
        return false;
    }
}

#endregion

/// <summary>
/// Helper to create types from string names
/// </summary>
public static class NSLTypes
{
    public static NSLType FromName(string name)
    {
        return name.ToLower() switch
        {
            "number" => NSLNumberType.Instance,
            "int" => NSLIntType.Instance,
            "bool" => NSLBoolType.Instance,
            "string" => NSLStringType.Instance,
            "void" => NSLVoidType.Instance,
            "null" => NSLNullType.Instance,
            "any" => NSLAnyType.Instance,
            "vec" => new NSLVecType(),
            "mat" => new NSLMatType(),
            "tensor" => new NSLTensorType(),
            "prob" => NSLProbType.Instance,
            _ => throw new ArgumentException($"Unknown type: {name}")
        };
    }

    public static NSLType Number => NSLNumberType.Instance;
    public static NSLType Int => NSLIntType.Instance;
    public static NSLType Bool => NSLBoolType.Instance;
    public static NSLType String => NSLStringType.Instance;
    public static NSLType Void => NSLVoidType.Instance;
    public static NSLType Null => NSLNullType.Instance;
    public static NSLType Any => NSLAnyType.Instance;
    public static NSLType Prob => NSLProbType.Instance;
    public static NSLVecType Vec(int? length = null) => new(length);
    public static NSLMatType Mat(int? rows = null, int? cols = null) => new(rows, cols);
    public static NSLTensorType Tensor(int[]? shape = null) => new(shape);
    public static NSLArrayType Array(NSLType elementType) => new(elementType);
    public static NSLOptionalType Optional(NSLType innerType) => new(innerType);
    public static NSLResultType Result(NSLType okType, NSLType errType) => new(okType, errType);
    public static NSLFunctionType Function(IEnumerable<NSLType> paramTypes, NSLType returnType) =>
        new(paramTypes, returnType);
    public static NSLEnumType Enum(string name, Dictionary<string, NSLEnumVariantType> variants) =>
        new(name, variants);

    // Ownership types (Rust-inspired)
    public static NSLOwnedType Owned(NSLType innerType) => new(innerType);
    public static NSLBorrowType Borrow(NSLType innerType) => new(innerType);
    public static NSLMutBorrowType MutBorrow(NSLType innerType) => new(innerType);

    // GPU/Accelerator types (Mojo-inspired)
    public static NSLKernelType Kernel(IEnumerable<NSLType> paramTypes, NSLType returnType, int blockSize = 256) =>
        new(paramTypes, returnType, blockSize);
    public static NSLDeviceTensorType DeviceTensor(string device = "gpu", int[]? shape = null) =>
        new(device, shape);

    // ML types
    public static NSLDiffType Diff(NSLType innerType) => new(innerType);
}
