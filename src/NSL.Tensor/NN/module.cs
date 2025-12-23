using System;
using System.Collections.Generic;
using System.Linq;

namespace NSL.Tensor.NN
{
    /// <summary>
    /// Base class for all neural network modules
    /// </summary>
    public abstract class Module
    {
        private bool _training = true;
        private readonly Dictionary<string, Tensor> _parameters = new();
        private readonly Dictionary<string, Tensor> _buffers = new();
        private readonly Dictionary<string, Module> _modules = new();

        /// <summary>
        /// Whether the module is in training mode
        /// </summary>
        public bool Training => _training;

        /// <summary>
        /// Forward pass - must be implemented by subclasses
        /// </summary>
        public abstract Tensor Forward(Tensor input);

        /// <summary>
        /// Call the module (same as Forward)
        /// </summary>
        public Tensor Call(Tensor input) => Forward(input);

        /// <summary>
        /// Register a parameter
        /// </summary>
        protected void RegisterParameter(string name, Tensor parameter)
        {
            parameter.RequiresGrad = true;
            _parameters[name] = parameter;
        }

        /// <summary>
        /// Register a buffer (non-trainable tensor)
        /// </summary>
        protected void RegisterBuffer(string name, Tensor buffer)
        {
            buffer.RequiresGrad = false;
            _buffers[name] = buffer;
        }

        /// <summary>
        /// Register a submodule
        /// </summary>
        protected void RegisterModule(string name, Module module)
        {
            _modules[name] = module;
        }

        /// <summary>
        /// Get all parameters (including submodules)
        /// </summary>
        public IEnumerable<Tensor> Parameters()
        {
            foreach (var param in _parameters.Values)
                yield return param;

            foreach (var module in _modules.Values)
                foreach (var param in module.Parameters())
                    yield return param;
        }

        /// <summary>
        /// Get named parameters
        /// </summary>
        public IEnumerable<(string name, Tensor param)> NamedParameters(string prefix = "")
        {
            foreach (var kv in _parameters)
                yield return (string.IsNullOrEmpty(prefix) ? kv.Key : $"{prefix}.{kv.Key}", kv.Value);

            foreach (var kv in _modules)
                foreach (var (name, param) in kv.Value.NamedParameters(string.IsNullOrEmpty(prefix) ? kv.Key : $"{prefix}.{kv.Key}"))
                    yield return (name, param);
        }

        /// <summary>
        /// Get all buffers
        /// </summary>
        public IEnumerable<Tensor> Buffers()
        {
            foreach (var buffer in _buffers.Values)
                yield return buffer;

            foreach (var module in _modules.Values)
                foreach (var buffer in module.Buffers())
                    yield return buffer;
        }

        /// <summary>
        /// Get all submodules
        /// </summary>
        public IEnumerable<Module> Modules()
        {
            yield return this;

            foreach (var module in _modules.Values)
                foreach (var m in module.Modules())
                    yield return m;
        }

        /// <summary>
        /// Set training mode
        /// </summary>
        public Module Train(bool mode = true)
        {
            _training = mode;
            foreach (var module in _modules.Values)
                module.Train(mode);
            return this;
        }

        /// <summary>
        /// Set evaluation mode
        /// </summary>
        public Module Eval() => Train(false);

        /// <summary>
        /// Zero all gradients
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var param in Parameters())
                param.ZeroGrad();
        }

        /// <summary>
        /// Move to device
        /// </summary>
        public Module To(Device device)
        {
            // For now, all tensors are on CPU
            return this;
        }

        /// <summary>
        /// Get number of trainable parameters
        /// </summary>
        public long NumParameters() => Parameters().Sum(p => p.NumElements);

        /// <summary>
        /// Get state dictionary (all parameters and buffers)
        /// </summary>
        public virtual Dictionary<string, Tensor> StateDict()
        {
            var state = new Dictionary<string, Tensor>();
            foreach (var (name, param) in NamedParameters())
                state[name] = param;
            foreach (var (name, buffer) in NamedBuffers())
                state[name] = buffer;
            return state;
        }

        /// <summary>
        /// Load state dictionary
        /// </summary>
        public virtual void LoadStateDict(Dictionary<string, Tensor> stateDict, bool strict = true)
        {
            var currentState = StateDict();
            foreach (var (name, tensor) in stateDict)
            {
                if (currentState.ContainsKey(name))
                {
                    // Copy data
                    currentState[name].CopyFrom(tensor);
                }
                else if (strict)
                {
                    throw new ArgumentException($"Unexpected key in state_dict: {name}");
                }
            }
        }

        /// <summary>
        /// Get named buffers
        /// </summary>
        public IEnumerable<(string name, Tensor buffer)> NamedBuffers(string prefix = "")
        {
            foreach (var kv in _buffers)
                yield return (string.IsNullOrEmpty(prefix) ? kv.Key : $"{prefix}.{kv.Key}", kv.Value);

            foreach (var kv in _modules)
                foreach (var (name, buffer) in kv.Value.NamedBuffers(string.IsNullOrEmpty(prefix) ? kv.Key : $"{prefix}.{kv.Key}"))
                    yield return (name, buffer);
        }

        /// <summary>
        /// Apply function to all modules
        /// </summary>
        public Module Apply(Action<Module> fn)
        {
            foreach (var module in Modules())
                fn(module);
            return this;
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"{GetType().Name}()";
        }
    }

    /// <summary>
    /// Sequential container - runs modules in order
    /// </summary>
    public class Sequential : Module
    {
        private readonly List<Module> _layers;

        /// <summary>Public API</summary>
        public Sequential(params Module[] modules)
        {
            _layers = modules.ToList();
            for (int i = 0; i < _layers.Count; i++)
                RegisterModule(i.ToString(), _layers[i]);
        }

        /// <summary>Public API</summary>
        public Sequential(IEnumerable<(string name, Module module)> namedModules)
        {
            _layers = new List<Module>();
            foreach (var (name, module) in namedModules)
            {
                _layers.Add(module);
                RegisterModule(name, module);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var x = input;
            foreach (var layer in _layers)
                x = layer.Forward(x);
            return x;
        }

        /// <summary>Public API</summary>
        public void Add(Module module)
        {
            RegisterModule(_layers.Count.ToString(), module);
            _layers.Add(module);
        }

        /// <summary>Public API</summary>
        public Module this[int index] => _layers[index];

        /// <summary>Public API</summary>
        public int Count => _layers.Count;

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var lines = new List<string> { "Sequential(" };
            for (int i = 0; i < _layers.Count; i++)
                lines.Add($"  ({i}): {_layers[i]}");
            lines.Add(")");
            return string.Join("\n", lines);
        }
    }

    /// <summary>
    /// Module list container
    /// </summary>
    public class ModuleList : Module
    {
        private readonly List<Module> _modules;

        /// <summary>Public API</summary>
        public ModuleList(params Module[] modules)
        {
            _modules = modules.ToList();
            for (int i = 0; i < _modules.Count; i++)
                RegisterModule(i.ToString(), _modules[i]);
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            throw new InvalidOperationException("ModuleList has no forward method. Access modules directly.");
        }

        /// <summary>Public API</summary>
        public void Add(Module module)
        {
            RegisterModule(_modules.Count.ToString(), module);
            _modules.Add(module);
        }

        /// <summary>Public API</summary>
        public Module this[int index] => _modules[index];

        /// <summary>Public API</summary>
        public int Count => _modules.Count;

        /// <summary>Public API</summary>
        public IEnumerator<Module> GetEnumerator() => _modules.GetEnumerator();
    }
}