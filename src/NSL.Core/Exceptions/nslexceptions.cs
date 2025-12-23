using System;

namespace NSL.Core.Exceptions
{
    /// <summary>
    /// Base NSL runtime exception
    /// </summary>
    public class NSLRuntimeException : Exception
    {
        /// <summary>Creates a new instance.</summary>
        public NSLRuntimeException(string message) : base(message) { }
        /// <summary>Creates a new instance.</summary>
        public NSLRuntimeException(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    /// NSL Python integration exception
    /// </summary>
    public class NSLPythonException : NSLRuntimeException
    {
        /// <summary>Creates a new instance.</summary>
        public NSLPythonException(string message) : base(message) { }
        /// <summary>Creates a new instance.</summary>
        public NSLPythonException(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    /// NSL return exception for function returns
    /// </summary>
    public class NSLReturnException : Exception
    {
        /// <summary>Gets the object value.</summary>
        public object? Value { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLReturnException(object? value)
        {
            Value = value;
        }
    }

    /// <summary>
    /// NSL break exception for loop control
    /// </summary>
    public class NSLBreakException : Exception
    {
    }

    /// <summary>
    /// NSL continue exception for loop control
    /// </summary>
    public class NSLContinueException : Exception
    {
    }
}
