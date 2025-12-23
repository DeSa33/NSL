using System;
using System.Collections.Generic;
using NSL.Core.Tokens;

namespace NSL.Core
{
    /// <summary>
    /// Represents a token in NSL source code
    /// Uses the unified TokenType enum from NSL.Core.Tokens
    /// </summary>
    public class NSLToken : IEquatable<NSLToken>
    {
        /// <summary>See implementation for details.</summary>
        public TokenType Type { get; }
        /// <summary>Gets the string value.</summary>
        public string Value { get; }
        /// <summary>Gets the integer value.</summary>
        public int Line { get; }
        /// <summary>Gets the integer value.</summary>
        public int Column { get; }
        /// <summary>Gets the integer value.</summary>
        public int Position { get; }
        /// <summary>Gets the integer value.</summary>
        public int Length { get; }

        /// <summary>Creates a new instance.</summary>
        public NSLToken(TokenType type, string value, int line, int column, int position, int length)
        {
            Type = type;
            Value = value ?? string.Empty;
            Line = line;
            Column = column;
            Position = position;
            Length = length;
        }

        /// <summary>Gets the boolean flag.</summary>
        public bool Equals(NSLToken? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;

            return Type == other.Type &&
                   Value == other.Value &&
                   Line == other.Line &&
                   Column == other.Column;
        }

        /// <inheritdoc/>
        public override bool Equals(object? obj)
        {
            return obj is NSLToken other && Equals(other);
        }

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            return HashCode.Combine(Type, Value, Line, Column);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{Type}('{Value}') at {Line}:{Column}";
        }

        /// <summary>Static helper method.</summary>
        public static bool operator ==(NSLToken? left, NSLToken? right)
        {
            if (left is null) return right is null;
            return left.Equals(right);
        }

        /// <summary>Static helper method.</summary>
        public static bool operator !=(NSLToken? left, NSLToken? right)
        {
            return !(left == right);
        }
    }

}
