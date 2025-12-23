using System;

namespace NSL.Core
{
    /// <summary>
    /// Base exception for all NSL runtime errors
    /// </summary>
    public class NSLRuntimeException : Exception
    {
        /// <summary>Initializes a new NSL runtime exception with the specified message.</summary>
        /// <param name="message">The error message.</param>
        public NSLRuntimeException(string message) : base(message) { }
        /// <summary>Initializes a new NSL runtime exception with message and inner exception.</summary>
        /// <param name="message">The error message.</param>
        /// <param name="innerException">The inner exception that caused this error.</param>
        public NSLRuntimeException(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    /// Exception thrown during NSL parsing
    /// </summary>
    public class NSLParseException : Exception
    {
        /// <summary>Initializes a new NSL parse exception with the specified message.</summary>
        /// <param name="message">The parse error message.</param>
        public NSLParseException(string message) : base(message) { }
        /// <summary>Initializes a new NSL parse exception with message and inner exception.</summary>
        /// <param name="message">The parse error message.</param>
        /// <param name="innerException">The inner exception that caused this error.</param>
        public NSLParseException(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    /// Exception used for function returns (control flow, not an error)
    /// </summary>
    public class NSLReturnException : Exception
    {
        /// <summary>Gets the return value being propagated.</summary>
        public object? Value { get; }

        /// <summary>Initializes a return exception with the specified return value.</summary>
        /// <param name="value">The value being returned from the function.</param>
        public NSLReturnException(object? value)
        {
            Value = value;
        }
    }

    /// <summary>
    /// Exception thrown when a variable is not found
    /// </summary>
    public class NSLVariableNotFoundException : NSLRuntimeException
    {
        /// <summary>Gets the name of the variable that was not found.</summary>
        public string VariableName { get; }

        /// <summary>Initializes a new variable not found exception.</summary>
        /// <param name="variableName">The name of the undefined variable.</param>
        public NSLVariableNotFoundException(string variableName) 
            : base($"Undefined variable: {variableName}")
        {
            VariableName = variableName;
        }
    }

    /// <summary>
    /// Exception thrown for type mismatches
    /// </summary>
    public class NSLTypeException : NSLRuntimeException
    {
        /// <summary>Gets the expected type name.</summary>
        public string ExpectedType { get; }
        /// <summary>Gets the actual type name that was received.</summary>
        public string ActualType { get; }

        /// <summary>Initializes a new type exception for a type mismatch.</summary>
        /// <param name="expectedType">The expected type name.</param>
        /// <param name="actualType">The actual type name received.</param>
        public NSLTypeException(string expectedType, string actualType) 
            : base($"Expected {expectedType} but got {actualType}")
        {
            ExpectedType = expectedType;
            ActualType = actualType;
        }
    }

    /// <summary>
    /// Exception thrown for arithmetic errors
    /// </summary>
    public class NSLArithmeticException : NSLRuntimeException
    {
        /// <summary>Initializes a new arithmetic exception with the specified message.</summary>
        /// <param name="message">The arithmetic error message.</param>
        public NSLArithmeticException(string message) : base(message) { }
        /// <summary>Initializes a new arithmetic exception with message and inner exception.</summary>
        /// <param name="message">The arithmetic error message.</param>
        /// <param name="innerException">The inner exception that caused this error.</param>
        public NSLArithmeticException(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    /// Exception thrown for function call errors
    /// </summary>
    public class NSLFunctionException : NSLRuntimeException
    {
        /// <summary>Gets the name of the function that caused the error.</summary>
        public string FunctionName { get; }

        /// <summary>Initializes a new function exception with function name and message.</summary>
        /// <param name="functionName">The name of the function that caused the error.</param>
        /// <param name="message">The error message.</param>
        public NSLFunctionException(string functionName, string message) : base(message)
        {
            FunctionName = functionName;
        }
    }
}