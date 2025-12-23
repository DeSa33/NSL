using System;
using System.Collections.Generic;

namespace NSL.Core.Types
{
    /// <summary>
    /// Base NSL object type for dynamic properties
    /// </summary>
    public class NSLObject
    {
        private readonly Dictionary<string, object?> _properties = new();

        /// <summary>See implementation for details.</summary>
        public virtual void Set(string name, object? value)
        {
            _properties[name] = value;
        }

        /// <summary>See implementation for details.</summary>
        public virtual object? Get(string name)
        {
            return _properties.TryGetValue(name, out var value) ? value : null;
        }

        /// <summary>See implementation for details.</summary>
        public virtual bool Has(string name)
        {
            return _properties.ContainsKey(name);
        }

        /// <summary>See implementation for details.</summary>
        public virtual Dictionary<string, object?> GetProperties()
        {
            return new Dictionary<string, object?>(_properties);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return "NSLObject";
        }
    }

    /// <summary>
    /// NSL list type
    /// </summary>
    public class NSLList
    {
        private readonly List<object?> _elements = new();

        /// <summary>Performs the operation.</summary>
        public void Add(object? element)
        {
            _elements.Add(element);
        }

        /// <summary>Gets the object value.</summary>
        public object? Get(int index)
        {
            return index >= 0 && index < _elements.Count ? _elements[index] : null;
        }

        /// <summary>Performs the operation.</summary>
        public void Set(int index, object? value)
        {
            if (index >= 0 && index < _elements.Count)
            {
                _elements[index] = value;
            }
        }

        /// <summary>Gets the integer value.</summary>
        public int Count => _elements.Count;

        /// <summary>Gets the list.</summary>
        public List<object?> GetElements()
        {
            return new List<object?>(_elements);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"NSLList[{_elements.Count}]";
        }
    }

    /// <summary>
    /// Interface for callable NSL functions
    /// </summary>
    public interface ICallable
    {
        int Arity { get; }
        object? Call(object? interpreter, List<object?> arguments);
    }
}