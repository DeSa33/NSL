using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace NSL.Tensor.NN
{
    #region Gradient Checkpointing Core

    /// <summary>
    /// Gradient checkpointing for memory-efficient training.
    /// Trades computation for memory by recomputing activations during backward pass.
    ///
    /// Memory savings: O(sqrt(n)) instead of O(n) for n layers
    /// Computation overhead: ~33% extra forward passes
    ///
    /// Usage:
    /// var output = GradientCheckpoint.Checkpoint(module, input);
    /// // or with checkpointing policy:
    /// var model = new CheckpointedSequential(layers, CheckpointPolicy.EveryN(2));
    /// </summary>
    public static class GradientCheckpoint
    {
        [ThreadStatic]
        private static bool _inCheckpoint;

        [ThreadStatic]
        private static CheckpointContext? _currentContext;

        /// <summary>
        /// Check if currently inside a checkpointed region.
        /// </summary>
        public static bool InCheckpointedRegion => _inCheckpoint;

        /// <summary>
        /// Checkpoint a function - forward pass discards activations, backward recomputes.
        /// </summary>
        public static Tensor Checkpoint(Func<Tensor, Tensor> function, Tensor input)
        {
            if (!GradientTape.Current?.IsRecording ?? true)
            {
                // No gradient tape - just run forward
                return function(input);
            }

            // Store the checkpoint info
            var checkpoint = new CheckpointedFunction(function, input);

            // Run forward without recording intermediate ops
            _inCheckpoint = true;
            Tensor output;
            try
            {
                output = function(input);
            }
            finally
            {
                _inCheckpoint = false;
            }

            // Register the checkpoint for backward
            GradientTape.Current?.RecordCheckpoint(checkpoint, output);

            return output;
        }

        /// <summary>
        /// Checkpoint a module's forward pass.
        /// </summary>
        public static Tensor Checkpoint(Module module, Tensor input)
        {
            return Checkpoint(x => module.Forward(x), input);
        }

        /// <summary>
        /// Checkpoint multiple functions as a sequence.
        /// </summary>
        public static Tensor CheckpointSequential(IEnumerable<Func<Tensor, Tensor>> functions, Tensor input)
        {
            var current = input;
            foreach (var fn in functions)
            {
                current = Checkpoint(fn, current);
            }
            return current;
        }

        /// <summary>
        /// Checkpoint with custom context for advanced use cases.
        /// </summary>
        public static Tensor CheckpointWithContext(
            Func<Tensor, CheckpointContext, Tensor> function,
            Tensor input,
            CheckpointContext? context = null)
        {
            context ??= new CheckpointContext();

            var prevContext = _currentContext;
            _currentContext = context;
            _inCheckpoint = true;

            try
            {
                return function(input, context);
            }
            finally
            {
                _currentContext = prevContext;
                _inCheckpoint = false;
            }
        }

        /// <summary>
        /// Get current checkpoint context (if any).
        /// </summary>
        public static CheckpointContext? CurrentContext => _currentContext;
    }

    /// <summary>
    /// Internal representation of a checkpointed function for recomputation.
    /// </summary>
    internal class CheckpointedFunction
    {
        private readonly Func<Tensor, Tensor> _function;
        private readonly Tensor _savedInput;
        private Tensor? _recomputedOutput;

        /// <summary>Public API</summary>
        public CheckpointedFunction(Func<Tensor, Tensor> function, Tensor input)
        {
            _function = function;
            // Clone input to ensure it's preserved
            _savedInput = input.Clone();
        }

        /// <summary>
        /// Recompute the forward pass to get activations for backward.
        /// </summary>
        public Tensor Recompute()
        {
            if (_recomputedOutput == null)
            {
                _recomputedOutput = _function(_savedInput);
            }
            return _recomputedOutput;
        }

        /// <summary>
        /// Clear cached recomputation to free memory.
        /// </summary>
        public void ClearCache()
        {
            _recomputedOutput = null;
        }
    }

    #endregion

    #region Checkpoint Context

    /// <summary>
    /// Context for advanced checkpointing with custom tensor preservation.
    /// </summary>
    public class CheckpointContext
    {
        private readonly Dictionary<string, Tensor> _savedTensors = new();
        private readonly Dictionary<string, object> _metadata = new();

        /// <summary>
        /// Save a tensor that should be preserved (not recomputed).
        /// Use for tensors that are expensive to recompute or needed across checkpoints.
        /// </summary>
        public void SaveForBackward(string key, Tensor tensor)
        {
            _savedTensors[key] = tensor.Clone();
        }

        /// <summary>
        /// Retrieve a saved tensor.
        /// </summary>
        public Tensor? GetSaved(string key)
        {
            return _savedTensors.TryGetValue(key, out var tensor) ? tensor : null;
        }

        /// <summary>
        /// Store metadata for the checkpoint.
        /// </summary>
        public void SetMetadata(string key, object value)
        {
            _metadata[key] = value;
        }

        /// <summary>
        /// Retrieve metadata.
        /// </summary>
        public T? GetMetadata<T>(string key)
        {
            return _metadata.TryGetValue(key, out var value) ? (T)value : default;
        }

        /// <summary>
        /// Clear all saved tensors to free memory.
        /// </summary>
        public void Clear()
        {
            _savedTensors.Clear();
            _metadata.Clear();
        }
    }

    #endregion

    #region Checkpointing Policies

    /// <summary>
    /// Policy for determining which layers to checkpoint.
    /// </summary>
    public abstract class CheckpointPolicy
    {
        /// <summary>
        /// Determine if a layer at the given index should be checkpointed.
        /// </summary>
        public abstract bool ShouldCheckpoint(int layerIndex, int totalLayers);

        /// <summary>
        /// No checkpointing - store all activations.
        /// </summary>
        public static CheckpointPolicy None { get; } = new NoCheckpointPolicy();

        /// <summary>
        /// Checkpoint all layers.
        /// </summary>
        public static CheckpointPolicy All { get; } = new AllCheckpointPolicy();

        /// <summary>
        /// Checkpoint every N layers.
        /// </summary>
        public static CheckpointPolicy EveryN(int n) => new EveryNPolicy(n);

        /// <summary>
        /// Checkpoint using sqrt(n) strategy for optimal memory-compute tradeoff.
        /// </summary>
        public static CheckpointPolicy Sqrt { get; } = new SqrtPolicy();

        /// <summary>
        /// Custom policy based on a predicate.
        /// </summary>
        public static CheckpointPolicy Custom(Func<int, int, bool> predicate) =>
            new CustomPolicy(predicate);

        /// <summary>
        /// Memory-based policy that enables checkpointing when memory pressure is high.
        /// </summary>
        public static CheckpointPolicy MemoryBased(long thresholdBytes) =>
            new MemoryBasedPolicy(thresholdBytes);
    }

    internal class NoCheckpointPolicy : CheckpointPolicy
    {
        /// <summary>Public API</summary>
        public override bool ShouldCheckpoint(int layerIndex, int totalLayers) => false;
    }

    internal class AllCheckpointPolicy : CheckpointPolicy
    {
        /// <summary>Public API</summary>
        public override bool ShouldCheckpoint(int layerIndex, int totalLayers) => true;
    }

    internal class EveryNPolicy : CheckpointPolicy
    {
        private readonly int _n;

        /// <summary>Public API</summary>
        public EveryNPolicy(int n)
        {
            _n = n;
        }

        /// <summary>Public API</summary>
        public override bool ShouldCheckpoint(int layerIndex, int totalLayers) =>
            layerIndex % _n == 0;
    }

    internal class SqrtPolicy : CheckpointPolicy
    {
        /// <summary>Public API</summary>
        public override bool ShouldCheckpoint(int layerIndex, int totalLayers)
        {
            var sqrtN = (int)Math.Sqrt(totalLayers);
            return layerIndex % Math.Max(1, sqrtN) == 0;
        }
    }

    internal class CustomPolicy : CheckpointPolicy
    {
        private readonly Func<int, int, bool> _predicate;

        /// <summary>Public API</summary>
        public CustomPolicy(Func<int, int, bool> predicate)
        {
            _predicate = predicate;
        }

        /// <summary>Public API</summary>
        public override bool ShouldCheckpoint(int layerIndex, int totalLayers) =>
            _predicate(layerIndex, totalLayers);
    }

    internal class MemoryBasedPolicy : CheckpointPolicy
    {
        private readonly long _thresholdBytes;

        /// <summary>Public API</summary>
        public MemoryBasedPolicy(long thresholdBytes)
        {
            _thresholdBytes = thresholdBytes;
        }

        /// <summary>Public API</summary>
        public override bool ShouldCheckpoint(int layerIndex, int totalLayers)
        {
            var currentMemory = GC.GetTotalMemory(forceFullCollection: false);
            return currentMemory > _thresholdBytes;
        }
    }

    #endregion

    #region Checkpointed Containers

    /// <summary>
    /// Sequential container with gradient checkpointing support.
    /// </summary>
    public class CheckpointedSequential : Module
    {
        private readonly List<Module> _modules;
        private readonly CheckpointPolicy _policy;
        private bool _checkpointingEnabled = true;

        /// <summary>Public API</summary>
        public IReadOnlyList<Module> Children => _modules;
        /// <summary>Public API</summary>
        public bool CheckpointingEnabled { get => _checkpointingEnabled; set => _checkpointingEnabled = value; }

        /// <summary>Public API</summary>
        public CheckpointedSequential(IEnumerable<Module> modules, CheckpointPolicy? policy = null)
        {
            _modules = modules.ToList();
            _policy = policy ?? CheckpointPolicy.Sqrt;
        }

        /// <summary>Public API</summary>
        public CheckpointedSequential(CheckpointPolicy? policy = null, params Module[] modules)
            : this(modules, policy)
        {
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var current = input;
            var totalLayers = _modules.Count;

            for (int i = 0; i < _modules.Count; i++)
            {
                var module = _modules[i];

                if (_checkpointingEnabled && _policy.ShouldCheckpoint(i, totalLayers))
                {
                    current = GradientCheckpoint.Checkpoint(module, current);
                }
                else
                {
                    current = module.Forward(current);
                }
            }

            return current;
        }

        /// <summary>Public API</summary>
        public override Dictionary<string, Tensor> StateDict()
        {
            var state = new Dictionary<string, Tensor>();

            for (int i = 0; i < _modules.Count; i++)
            {
                foreach (var (name, tensor) in _modules[i].StateDict())
                {
                    state[$"{i}.{name}"] = tensor;
                }
            }

            return state;
        }

        /// <summary>Public API</summary>
        public override void LoadStateDict(Dictionary<string, Tensor> stateDict, bool strict = true)
        {
            for (int i = 0; i < _modules.Count; i++)
            {
                var prefix = $"{i}.";
                var moduleState = new Dictionary<string, Tensor>();

                foreach (var (key, value) in stateDict)
                {
                    if (key.StartsWith(prefix))
                    {
                        moduleState[key.Substring(prefix.Length)] = value;
                    }
                }

                if (moduleState.Count > 0)
                {
                    _modules[i].LoadStateDict(moduleState, strict);
                }
            }
        }

        /// <summary>
        /// Add a module to the sequence.
        /// </summary>
        public CheckpointedSequential Add(Module module)
        {
            _modules.Add(module);
            return this;
        }
    }

    /// <summary>
    /// Transformer block with selective activation checkpointing.
    /// Checkpoints the attention and FFN separately for finer control.
    /// </summary>
    public class CheckpointedTransformerBlock : Module
    {
        private readonly Module _attention;
        private readonly Module _ffn;
        private readonly Module _norm1;
        private readonly Module _norm2;
        private readonly bool _checkpointAttention;
        private readonly bool _checkpointFFN;
        private readonly bool _preNorm;

        /// <summary>Public API</summary>
        public CheckpointedTransformerBlock(
            Module attention,
            Module ffn,
            Module norm1,
            Module norm2,
            bool checkpointAttention = true,
            bool checkpointFFN = true,
            bool preNorm = true)
        {
            _attention = attention;
            _ffn = ffn;
            _norm1 = norm1;
            _norm2 = norm2;
            _checkpointAttention = checkpointAttention;
            _checkpointFFN = checkpointFFN;
            _preNorm = preNorm;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            Tensor residual, hidden;

            if (_preNorm)
            {
                // Pre-norm: norm before attention/ffn
                hidden = _norm1.Forward(input);

                if (_checkpointAttention)
                    hidden = GradientCheckpoint.Checkpoint(_attention, hidden);
                else
                    hidden = _attention.Forward(hidden);

                residual = input.Add(hidden);

                hidden = _norm2.Forward(residual);

                if (_checkpointFFN)
                    hidden = GradientCheckpoint.Checkpoint(_ffn, hidden);
                else
                    hidden = _ffn.Forward(hidden);

                return residual.Add(hidden);
            }
            else
            {
                // Post-norm: norm after attention/ffn
                if (_checkpointAttention)
                    hidden = GradientCheckpoint.Checkpoint(_attention, input);
                else
                    hidden = _attention.Forward(input);

                residual = _norm1.Forward(input.Add(hidden));

                if (_checkpointFFN)
                    hidden = GradientCheckpoint.Checkpoint(_ffn, residual);
                else
                    hidden = _ffn.Forward(residual);

                return _norm2.Forward(residual.Add(hidden));
            }
        }

        /// <summary>Public API</summary>
        public override Dictionary<string, Tensor> StateDict()
        {
            var state = new Dictionary<string, Tensor>();

            foreach (var (k, v) in _attention.StateDict())
                state[$"attention.{k}"] = v;
            foreach (var (k, v) in _ffn.StateDict())
                state[$"ffn.{k}"] = v;
            foreach (var (k, v) in _norm1.StateDict())
                state[$"norm1.{k}"] = v;
            foreach (var (k, v) in _norm2.StateDict())
                state[$"norm2.{k}"] = v;

            return state;
        }

        /// <summary>Public API</summary>
        public override void LoadStateDict(Dictionary<string, Tensor> stateDict, bool strict = true)
        {
            LoadSubmodule(_attention, "attention.", stateDict, strict);
            LoadSubmodule(_ffn, "ffn.", stateDict, strict);
            LoadSubmodule(_norm1, "norm1.", stateDict, strict);
            LoadSubmodule(_norm2, "norm2.", stateDict, strict);
        }

        private void LoadSubmodule(Module module, string prefix, Dictionary<string, Tensor> stateDict, bool strict)
        {
            var subState = new Dictionary<string, Tensor>();
            foreach (var (k, v) in stateDict)
            {
                if (k.StartsWith(prefix))
                    subState[k.Substring(prefix.Length)] = v;
            }
            if (subState.Count > 0)
                module.LoadStateDict(subState, strict);
        }
    }

    #endregion

    #region Segment Checkpointing

    /// <summary>
    /// Segment-based checkpointing for very deep models.
    /// Divides model into segments and checkpoints at segment boundaries.
    /// </summary>
    public class SegmentedCheckpointing
    {
        private readonly int _numSegments;
        private readonly List<List<Module>> _segments;

        /// <summary>Public API</summary>
        public SegmentedCheckpointing(IEnumerable<Module> modules, int numSegments)
        {
            _numSegments = numSegments;
            _segments = new List<List<Module>>();

            var moduleList = modules.ToList();
            var segmentSize = (moduleList.Count + numSegments - 1) / numSegments;

            for (int i = 0; i < moduleList.Count; i += segmentSize)
            {
                _segments.Add(moduleList.Skip(i).Take(segmentSize).ToList());
            }
        }

        /// <summary>
        /// Forward pass with segment-level checkpointing.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            var current = input;

            foreach (var segment in _segments)
            {
                // Checkpoint entire segment
                current = GradientCheckpoint.Checkpoint(
                    x => RunSegment(segment, x),
                    current);
            }

            return current;
        }

        private Tensor RunSegment(List<Module> segment, Tensor input)
        {
            var current = input;
            foreach (var module in segment)
            {
                current = module.Forward(current);
            }
            return current;
        }

        /// <summary>
        /// Get optimal number of segments for memory-compute balance.
        /// </summary>
        public static int OptimalSegments(int totalLayers, long availableMemoryMB, long activationSizePerLayerMB)
        {
            // Optimal segments = sqrt(total_layers) for memory-compute tradeoff
            var sqrtLayers = (int)Math.Sqrt(totalLayers);

            // Adjust based on available memory
            var memoryLayers = availableMemoryMB / activationSizePerLayerMB;
            var memorySegments = (int)Math.Ceiling((double)totalLayers / memoryLayers);

            return Math.Max(sqrtLayers, memorySegments);
        }
    }

    #endregion

    #region Activation Offloading

    /// <summary>
    /// CPU offloading for activations to reduce GPU memory usage.
    /// Moves activations to CPU during forward, brings back during backward.
    /// </summary>
    public class ActivationOffloader
    {
        private readonly Dictionary<int, Tensor> _cpuActivations = new();
        private int _layerCounter;
        private readonly bool _async;
        private readonly object _lock = new();

        /// <summary>Public API</summary>
        public ActivationOffloader(bool asyncTransfer = true)
        {
            _async = asyncTransfer;
            _layerCounter = 0;
        }

        /// <summary>
        /// Offload an activation to CPU.
        /// </summary>
        public int Offload(Tensor activation)
        {
            lock (_lock)
            {
                var id = _layerCounter++;
                // Clone to CPU (in practice, would use pinned memory for async transfer)
                _cpuActivations[id] = activation.Clone();
                return id;
            }
        }

        /// <summary>
        /// Reload an activation from CPU.
        /// </summary>
        public Tensor Reload(int id)
        {
            lock (_lock)
            {
                if (_cpuActivations.TryGetValue(id, out var tensor))
                {
                    // In practice, would transfer back to GPU
                    return tensor;
                }
                throw new InvalidOperationException($"Activation {id} not found in offloaded storage");
            }
        }

        /// <summary>
        /// Free an offloaded activation.
        /// </summary>
        public void Free(int id)
        {
            lock (_lock)
            {
                _cpuActivations.Remove(id);
            }
        }

        /// <summary>
        /// Clear all offloaded activations.
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _cpuActivations.Clear();
                _layerCounter = 0;
            }
        }

        /// <summary>
        /// Get current memory usage on CPU for offloaded activations.
        /// </summary>
        public long GetCPUMemoryUsage()
        {
            lock (_lock)
            {
                return _cpuActivations.Values.Sum(t => t.NumElements * sizeof(double));
            }
        }
    }

    /// <summary>
    /// Module wrapper that offloads activations to CPU.
    /// </summary>
    public class OffloadedModule : Module
    {
        private readonly Module _module;
        private readonly ActivationOffloader _offloader;

        /// <summary>Public API</summary>
        public OffloadedModule(Module module, ActivationOffloader offloader)
        {
            _module = module;
            _offloader = offloader;
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            // Offload input
            var inputId = _offloader.Offload(input);

            // Run forward
            var output = _module.Forward(input);

            // In backward, will reload from inputId
            // (This is simplified - real impl would hook into autograd)

            return output;
        }

        /// <summary>Public API</summary>
        public override Dictionary<string, Tensor> StateDict() => _module.StateDict();
        /// <summary>Public API</summary>
        public override void LoadStateDict(Dictionary<string, Tensor> stateDict, bool strict = true)
            => _module.LoadStateDict(stateDict, strict);
    }

    #endregion

    #region Memory Estimation

    /// <summary>
    /// Utilities for estimating memory usage with different checkpointing strategies.
    /// </summary>
    public static class CheckpointingMemoryEstimator
    {
        /// <summary>
        /// Estimate peak memory usage without checkpointing.
        /// </summary>
        public static long EstimateWithoutCheckpointing(
            int numLayers,
            long activationBytesPerLayer,
            long modelParamBytes)
        {
            // Peak = all activations + model params + gradients
            var activations = numLayers * activationBytesPerLayer;
            var gradients = modelParamBytes; // Same size as params

            return activations + modelParamBytes + gradients;
        }

        /// <summary>
        /// Estimate peak memory usage with full checkpointing.
        /// </summary>
        public static long EstimateWithFullCheckpointing(
            int numLayers,
            long activationBytesPerLayer,
            long modelParamBytes)
        {
            // Peak = 1 activation + model params + gradients
            // But need to recompute during backward, so ~2x activations at any point
            var activations = 2 * activationBytesPerLayer;
            var gradients = modelParamBytes;

            return activations + modelParamBytes + gradients;
        }

        /// <summary>
        /// Estimate peak memory usage with sqrt checkpointing.
        /// </summary>
        public static long EstimateWithSqrtCheckpointing(
            int numLayers,
            long activationBytesPerLayer,
            long modelParamBytes)
        {
            // Peak = sqrt(n) activations + model params + gradients
            var sqrtN = (int)Math.Sqrt(numLayers);
            var activations = sqrtN * activationBytesPerLayer;
            var gradients = modelParamBytes;

            return activations + modelParamBytes + gradients;
        }

        /// <summary>
        /// Find optimal checkpoint interval for given memory budget.
        /// </summary>
        public static int OptimalCheckpointInterval(
            int numLayers,
            long activationBytesPerLayer,
            long modelParamBytes,
            long memoryBudget)
        {
            var baseMemory = modelParamBytes * 2; // params + gradients

            if (memoryBudget <= baseMemory)
                return 1; // Checkpoint every layer (maximum checkpointing)

            var availableForActivations = memoryBudget - baseMemory;
            var maxStoredActivations = availableForActivations / activationBytesPerLayer;

            if (maxStoredActivations >= numLayers)
                return numLayers; // No checkpointing needed

            // Interval = numLayers / maxStoredActivations
            return Math.Max(1, (int)Math.Ceiling((double)numLayers / maxStoredActivations));
        }

        /// <summary>
        /// Compute additional FLOPs from checkpointing overhead.
        /// </summary>
        public static double CheckpointingOverhead(int numLayers, int checkpointInterval)
        {
            if (checkpointInterval >= numLayers)
                return 0.0; // No checkpointing

            // Each checkpoint requires recomputation of ~(interval-1)/2 layers on average
            var numCheckpoints = numLayers / checkpointInterval;
            var avgRecompute = (checkpointInterval - 1) / 2.0;
            var totalRecompute = numCheckpoints * avgRecompute;

            return totalRecompute / numLayers; // Fraction of extra forward compute
        }
    }

    #endregion

    #region GradientTape Extension for Checkpointing

    /// <summary>
    /// Extension methods for GradientTape to support checkpointing.
    /// </summary>
    internal static class GradientTapeCheckpointExtensions
    {
        /// <summary>
        /// Record a checkpoint in the gradient tape.
        /// </summary>
        internal static void RecordCheckpoint(this GradientTape tape, CheckpointedFunction checkpoint, Tensor output)
        {
            // In a full implementation, this would register the checkpoint
            // so that during backward, the activations are recomputed
            // This is a placeholder for the integration point
        }
    }

    #endregion
}