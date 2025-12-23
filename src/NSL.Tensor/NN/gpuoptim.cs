// GPU optimizer requires NSL.GPU project reference - uncomment when available:
// #define HAS_GPU

#if HAS_GPU
using System;
using System.Collections.Generic;
using System.Linq;
using NSL.GPU;

namespace NSL.Tensor.NN
{
    /// <summary>
    /// GPU-accelerated optimizer base class.
    /// Works with both CPU and GPU tensors, automatically using GPU acceleration when available.
    /// </summary>
    public abstract class GpuOptimizer
    {
        protected readonly List<Tensor> _parameters;
        protected readonly double _lr;
        protected int _step;
        protected readonly GpuAccelerator? _gpu;
        protected readonly bool _useGpu;

        /// <summary>Public API</summary>
        public double LearningRate => _lr;
        /// <summary>Public API</summary>
        public int Step => _step;
        /// <summary>Public API</summary>
        public bool UsingGpu => _useGpu && _gpu != null;

        protected GpuOptimizer(IEnumerable<Tensor> parameters, double lr, GpuAccelerator? gpu = null)
        {
            _parameters = parameters.ToList();
            _lr = lr;
            _step = 0;
            _gpu = gpu;
            _useGpu = gpu != null;
        }

        /// <summary>Public API</summary>
        public abstract void StepOptimizer();

        /// <summary>Public API</summary>
        public virtual void ZeroGrad()
        {
            foreach (var param in _parameters)
                param.ZeroGrad();
        }

        /// <summary>Public API</summary>
        public virtual Dictionary<string, object> StateDict()
        {
            return new Dictionary<string, object>
            {
                ["step"] = _step,
                ["lr"] = _lr,
                ["use_gpu"] = _useGpu
            };
        }

        /// <summary>Public API</summary>
        public virtual void LoadStateDict(Dictionary<string, object> state)
        {
            if (state.ContainsKey("step"))
                _step = Convert.ToInt32(state["step"]);
        }
    }

    /// <summary>
    /// GPU-accelerated Adam optimizer.
    /// Uses GPU kernels for parameter updates when available.
    /// </summary>
    public class GpuAdam : GpuOptimizer
    {
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;
        private readonly bool _amsgrad;
        private readonly List<Tensor> _m;  // First moment
        private readonly List<Tensor> _v;  // Second moment
        private readonly List<Tensor?> _vMax;  // Max second moment (AMSGrad)
        private readonly List<GpuTensor?> _gpuM;  // GPU first moment
        private readonly List<GpuTensor?> _gpuV;  // GPU second moment

        /// <summary>Public API</summary>
        public GpuAdam(IEnumerable<Tensor> parameters, double lr = 0.001, double beta1 = 0.9,
            double beta2 = 0.999, double eps = 1e-8, double weightDecay = 0, bool amsgrad = false,
            GpuAccelerator? gpu = null)
            : base(parameters, lr, gpu)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
            _amsgrad = amsgrad;

            // Initialize CPU moment estimates
            _m = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _v = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _vMax = _amsgrad
                ? _parameters.Select(p => (Tensor?)Tensor.Zeros(p.Shape)).ToList()
                : _parameters.Select(_ => (Tensor?)null).ToList();

            // Initialize GPU moment estimates if GPU available
            _gpuM = new List<GpuTensor?>();
            _gpuV = new List<GpuTensor?>();

            if (_useGpu && _gpu != null)
            {
                foreach (var p in _parameters)
                {
                    var intShape = p.Shape.Select(s => (int)s).ToArray();
                    _gpuM.Add(_gpu.Zeros(intShape));
                    _gpuV.Add(_gpu.Zeros(intShape));
                }
            }
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            if (_useGpu && _gpu != null)
            {
                StepGpu();
            }
            else
            {
                StepCpu();
            }
        }

        private void StepCpu()
        {
            double biasCorrection1 = 1 - Math.Pow(_beta1, _step);
            double biasCorrection2 = 1 - Math.Pow(_beta2, _step);

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay (decoupled from gradient)
                if (_weightDecay != 0)
                {
                    param.SubInPlace(param.Mul(_weightDecay * _lr));
                }

                // Update first moment: m = β₁m + (1-β₁)g
                _m[i] = _m[i].Mul(_beta1).Add(grad.Mul(1 - _beta1));

                // Update second moment: v = β₂v + (1-β₂)g²
                var gradSq = grad.Mul(grad);
                _v[i] = _v[i].Mul(_beta2).Add(gradSq.Mul(1 - _beta2));

                Tensor denom;
                if (_amsgrad)
                {
                    // AMSGrad: use max of past squared gradients
                    _vMax[i] = Tensor.Maximum(_vMax[i]!, _v[i]);
                    denom = _vMax[i]!.Sqrt().Div(Math.Sqrt(biasCorrection2)).Add(_eps);
                }
                else
                {
                    denom = _v[i].Sqrt().Div(Math.Sqrt(biasCorrection2)).Add(_eps);
                }

                // Bias-corrected first moment
                var mHat = _m[i].Div(biasCorrection1);

                // Update: θ = θ - lr * m_hat / (√v_hat + ε)
                param.SubInPlace(mHat.Div(denom).Mul(_lr));
            }
        }

        private void StepGpu()
        {
            if (_gpu == null) return;

            double biasCorrection1 = 1 - Math.Pow(_beta1, _step);
            double biasCorrection2 = 1 - Math.Pow(_beta2, _step);

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var intShape = param.Shape.Select(s => (int)s).ToArray();

                // Transfer gradient to GPU
                var gpuGrad = _gpu.ToGpu(ToFloatArray(param.Grad.Data), intShape);

                // Transfer parameter to GPU
                var gpuParam = _gpu.ToGpu(ToFloatArray(param.Data), intShape);

                // Weight decay
                if (_weightDecay != 0)
                {
                    var decay = _gpu.MulScalar(gpuParam, (float)(_weightDecay * _lr));
                    gpuParam = _gpu.Sub(gpuParam, decay);
                }

                // Update moments on GPU
                if (_gpuM[i] != null && _gpuV[i] != null)
                {
                    // m = β₁m + (1-β₁)g
                    var scaledM = _gpu.MulScalar(_gpuM[i]!, (float)_beta1);
                    var scaledG = _gpu.MulScalar(gpuGrad, (float)(1 - _beta1));
                    _gpuM[i] = _gpu.Add(scaledM, scaledG);

                    // v = β₂v + (1-β₂)g²
                    var gradSq = _gpu.Mul(gpuGrad, gpuGrad);
                    var scaledV = _gpu.MulScalar(_gpuV[i]!, (float)_beta2);
                    var scaledGSq = _gpu.MulScalar(gradSq, (float)(1 - _beta2));
                    _gpuV[i] = _gpu.Add(scaledV, scaledGSq);

                    // Bias correction
                    var mHat = _gpu.MulScalar(_gpuM[i]!, (float)(1.0 / biasCorrection1));
                    var vSqrt = _gpu.Sqrt(_gpuV[i]!);
                    var vHat = _gpu.MulScalar(vSqrt, (float)(1.0 / Math.Sqrt(biasCorrection2)));
                    var denom = _gpu.AddScalar(vHat, (float)_eps);

                    // Update
                    var update = _gpu.Div(mHat, denom);
                    update = _gpu.MulScalar(update, (float)_lr);
                    gpuParam = _gpu.Sub(gpuParam, update);
                }

                // Transfer back to CPU
                var (resultData, _) = _gpu.ToCpu(gpuParam);
                for (int j = 0; j < param.Data.Length; j++)
                {
                    param.Data[j] = resultData[j];
                }
            }
        }

        private static float[] ToFloatArray(double[] data)
        {
            var result = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                result[i] = (float)data[i];
            return result;
        }

        /// <summary>Public API</summary>
        public override string ToString() =>
            $"GpuAdam(lr={_lr}, beta1={_beta1}, beta2={_beta2}, gpu={UsingGpu})";
    }

    /// <summary>
    /// GPU-accelerated SGD optimizer with momentum.
    /// </summary>
    public class GpuSGD : GpuOptimizer
    {
        private readonly double _momentum;
        private readonly double _dampening;
        private readonly double _weightDecay;
        private readonly bool _nesterov;
        private readonly List<Tensor?> _velocities;
        private readonly List<GpuTensor?> _gpuVelocities;

        /// <summary>Public API</summary>
        public GpuSGD(IEnumerable<Tensor> parameters, double lr = 0.01, double momentum = 0,
            double dampening = 0, double weightDecay = 0, bool nesterov = false,
            GpuAccelerator? gpu = null)
            : base(parameters, lr, gpu)
        {
            if (nesterov && (momentum <= 0 || dampening != 0))
                throw new ArgumentException("Nesterov momentum requires momentum > 0 and dampening = 0");

            _momentum = momentum;
            _dampening = dampening;
            _weightDecay = weightDecay;
            _nesterov = nesterov;

            _velocities = _parameters.Select(_ => (Tensor?)null).ToList();
            _gpuVelocities = _parameters.Select(_ => (GpuTensor?)null).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            if (_useGpu && _gpu != null)
            {
                StepGpu();
            }
            else
            {
                StepCpu();
            }
        }

        private void StepCpu()
        {
            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Momentum
                if (_momentum != 0)
                {
                    if (_velocities[i] == null)
                    {
                        _velocities[i] = grad.Clone();
                    }
                    else
                    {
                        _velocities[i] = _velocities[i]!.Mul(_momentum).Add(grad.Mul(1 - _dampening));
                    }

                    if (_nesterov)
                        grad = grad.Add(_velocities[i]!.Mul(_momentum));
                    else
                        grad = _velocities[i]!;
                }

                param.SubInPlace(grad.Mul(_lr));
            }
        }

        private void StepGpu()
        {
            if (_gpu == null) return;

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var intShape = param.Shape.Select(s => (int)s).ToArray();
                var gpuGrad = _gpu.ToGpu(ToFloatArray(param.Grad.Data), intShape);
                var gpuParam = _gpu.ToGpu(ToFloatArray(param.Data), intShape);

                // Weight decay
                if (_weightDecay != 0)
                {
                    var wd = _gpu.MulScalar(gpuParam, (float)_weightDecay);
                    gpuGrad = _gpu.Add(gpuGrad, wd);
                }

                // Momentum
                if (_momentum != 0)
                {
                    if (_gpuVelocities[i] == null)
                    {
                        _gpuVelocities[i] = gpuGrad;
                    }
                    else
                    {
                        var scaledV = _gpu.MulScalar(_gpuVelocities[i]!, (float)_momentum);
                        var scaledG = _gpu.MulScalar(gpuGrad, (float)(1 - _dampening));
                        _gpuVelocities[i] = _gpu.Add(scaledV, scaledG);
                    }

                    if (_nesterov)
                    {
                        var momV = _gpu.MulScalar(_gpuVelocities[i]!, (float)_momentum);
                        gpuGrad = _gpu.Add(gpuGrad, momV);
                    }
                    else
                    {
                        gpuGrad = _gpuVelocities[i]!;
                    }
                }

                // Update
                var update = _gpu.MulScalar(gpuGrad, (float)_lr);
                gpuParam = _gpu.Sub(gpuParam, update);

                // Transfer back
                var (resultData, _) = _gpu.ToCpu(gpuParam);
                for (int j = 0; j < param.Data.Length; j++)
                {
                    param.Data[j] = resultData[j];
                }
            }
        }

        private static float[] ToFloatArray(double[] data)
        {
            var result = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                result[i] = (float)data[i];
            return result;
        }

        /// <summary>Public API</summary>
        public override string ToString() =>
            $"GpuSGD(lr={_lr}, momentum={_momentum}, gpu={UsingGpu})";
    }

    /// <summary>
    /// GPU-accelerated AdamW optimizer (decoupled weight decay).
    /// </summary>
    public class GpuAdamW : GpuAdam
    {
        /// <summary>Public API</summary>
        public GpuAdamW(IEnumerable<Tensor> parameters, double lr = 0.001, double beta1 = 0.9,
            double beta2 = 0.999, double eps = 1e-8, double weightDecay = 0.01,
            GpuAccelerator? gpu = null)
            : base(parameters, lr, beta1, beta2, eps, weightDecay, amsgrad: false, gpu)
        {
        }

        /// <summary>Public API</summary>
        public override string ToString() =>
            $"GpuAdamW(lr={LearningRate}, gpu={UsingGpu})";
    }

    /// <summary>
    /// Mixed-precision training utilities.
    /// Enables training with FP16/BF16 for memory and speed benefits.
    /// </summary>
    public class MixedPrecisionTrainer
    {
        private readonly GpuAccelerator _gpu;
        private readonly double _lossScale;
        private readonly bool _dynamicScaling;
        private double _currentScale;
        private int _overflowCount;
        private readonly int _scaleWindow;

        /// <summary>Public API</summary>
        public MixedPrecisionTrainer(GpuAccelerator gpu, double initialScale = 65536.0,
            bool dynamicScaling = true, int scaleWindow = 2000)
        {
            _gpu = gpu;
            _lossScale = initialScale;
            _currentScale = initialScale;
            _dynamicScaling = dynamicScaling;
            _scaleWindow = scaleWindow;
            _overflowCount = 0;
        }

        /// <summary>
        /// Scales the loss for mixed precision backward pass.
        /// </summary>
        public Tensor ScaleLoss(Tensor loss)
        {
            return loss.Mul(_currentScale);
        }

        /// <summary>
        /// Unscales gradients after backward pass.
        /// </summary>
        public void UnscaleGradients(IEnumerable<Tensor> parameters)
        {
            double invScale = 1.0 / _currentScale;

            foreach (var param in parameters)
            {
                if (param.Grad != null)
                {
                    for (int i = 0; i < param.Grad.Data.Length; i++)
                    {
                        param.Grad.Data[i] *= invScale;
                    }
                }
            }
        }

        /// <summary>
        /// Checks for gradient overflow and updates scale.
        /// Returns true if step should proceed, false if overflow detected.
        /// </summary>
        public bool CheckAndUpdateScale(IEnumerable<Tensor> parameters)
        {
            bool hasOverflow = false;

            foreach (var param in parameters)
            {
                if (param.Grad != null)
                {
                    for (int i = 0; i < param.Grad.Data.Length; i++)
                    {
                        if (double.IsInfinity(param.Grad.Data[i]) || double.IsNaN(param.Grad.Data[i]))
                        {
                            hasOverflow = true;
                            break;
                        }
                    }
                }
                if (hasOverflow) break;
            }

            if (_dynamicScaling)
            {
                if (hasOverflow)
                {
                    // Reduce scale on overflow
                    _currentScale /= 2.0;
                    _overflowCount = 0;
                }
                else
                {
                    _overflowCount++;
                    // Increase scale periodically if no overflow
                    if (_overflowCount >= _scaleWindow)
                    {
                        _currentScale *= 2.0;
                        _overflowCount = 0;
                    }
                }
            }

            return !hasOverflow;
        }

        /// <summary>Public API</summary>
        public double CurrentScale => _currentScale;
    }

    /// <summary>
    /// Distributed training utilities for data parallelism.
    /// </summary>
    public static class DistributedTraining
    {
        /// <summary>
        /// Averages gradients across multiple model replicas.
        /// For single-node multi-GPU setup.
        /// </summary>
        public static void AllReduceGradients(List<List<Tensor>> parameterLists)
        {
            if (parameterLists.Count == 0) return;

            int numReplicas = parameterLists.Count;
            int numParams = parameterLists[0].Count;

            for (int p = 0; p < numParams; p++)
            {
                // Sum gradients across replicas
                var sumGrad = Tensor.Zeros(parameterLists[0][p].Shape);

                foreach (var paramList in parameterLists)
                {
                    if (paramList[p].Grad != null)
                    {
                        for (int i = 0; i < sumGrad.Data.Length; i++)
                        {
                            sumGrad.Data[i] += paramList[p].Grad!.Data[i];
                        }
                    }
                }

                // Average
                for (int i = 0; i < sumGrad.Data.Length; i++)
                {
                    sumGrad.Data[i] /= numReplicas;
                }

                // Set averaged gradient to all replicas
                foreach (var paramList in parameterLists)
                {
                    paramList[p].ZeroGrad();
                    paramList[p].AccumulateGrad(sumGrad);
                }
            }
        }

        /// <summary>
        /// Synchronizes parameters across replicas.
        /// </summary>
        public static void SyncParameters(List<List<Tensor>> parameterLists)
        {
            if (parameterLists.Count <= 1) return;

            // Use first replica as source
            var source = parameterLists[0];

            for (int i = 1; i < parameterLists.Count; i++)
            {
                var target = parameterLists[i];
                for (int p = 0; p < source.Count; p++)
                {
                    Array.Copy(source[p].Data, target[p].Data, source[p].Data.Length);
                }
            }
        }

        /// <summary>
        /// Ring AllReduce implementation for efficient gradient synchronization.
        /// Reduces memory bandwidth by O(n) compared to naive AllReduce.
        /// </summary>
        public static void RingAllReduce(List<Tensor[]> gradients, int worldSize)
        {
            if (worldSize <= 1 || gradients.Count == 0) return;

            int numParams = gradients[0].Length;

            for (int p = 0; p < numParams; p++)
            {
                int numElements = gradients[0][p].Data.Length;
                int chunkSize = (numElements + worldSize - 1) / worldSize;

                // Scatter-reduce phase
                for (int step = 0; step < worldSize - 1; step++)
                {
                    for (int rank = 0; rank < worldSize; rank++)
                    {
                        int sendChunk = (rank - step + worldSize) % worldSize;
                        int recvChunk = (rank - step - 1 + worldSize) % worldSize;
                        int nextRank = (rank + 1) % worldSize;

                        int sendStart = sendChunk * chunkSize;
                        int sendEnd = Math.Min(sendStart + chunkSize, numElements);

                        int recvStart = recvChunk * chunkSize;
                        int recvEnd = Math.Min(recvStart + chunkSize, numElements);

                        // Reduce into next rank's buffer
                        for (int i = recvStart; i < recvEnd; i++)
                        {
                            gradients[nextRank][p].Data[i] += gradients[rank][p].Data[i];
                        }
                    }
                }

                // AllGather phase
                for (int step = 0; step < worldSize - 1; step++)
                {
                    for (int rank = 0; rank < worldSize; rank++)
                    {
                        int sendChunk = (rank - step + 1 + worldSize) % worldSize;
                        int nextRank = (rank + 1) % worldSize;

                        int start = sendChunk * chunkSize;
                        int end = Math.Min(start + chunkSize, numElements);

                        // Copy to next rank
                        for (int i = start; i < end; i++)
                        {
                            gradients[nextRank][p].Data[i] = gradients[rank][p].Data[i];
                        }
                    }
                }

                // Average
                double scale = 1.0 / worldSize;
                for (int rank = 0; rank < worldSize; rank++)
                {
                    for (int i = 0; i < numElements; i++)
                    {
                        gradients[rank][p].Data[i] *= scale;
                    }
                }
            }
        }
    }

    /// <summary>
    /// ZeRO (Zero Redundancy Optimizer) implementation for memory-efficient distributed training.
    /// Based on DeepSpeed ZeRO stages 1, 2, and 3.
    /// </summary>
    public class ZeROOptimizer
    {
        private readonly List<Tensor> _parameters;
        private readonly int _worldSize;
        private readonly int _rank;
        private readonly ZeROStage _stage;
        private readonly GpuOptimizer _baseOptimizer;

        // Partitioned states
        private readonly List<Tensor> _partitionedParams;      // Stage 3
        private readonly List<Tensor> _partitionedGrads;       // Stage 2+
        private readonly List<Tensor> _partitionedOptStates;   // Stage 1+

        /// <summary>Public API</summary>
        public enum ZeROStage
        {
            Stage1 = 1,
            Stage2 = 2,
            Stage3 = 3
        }

        /// <summary>Public API</summary>
        public ZeROOptimizer(
            IEnumerable<Tensor> parameters,
            GpuOptimizer baseOptimizer,
            int worldSize,
            int rank,
            ZeROStage stage = ZeROStage.Stage2)
        {
            _parameters = parameters.ToList();
            _baseOptimizer = baseOptimizer;
            _worldSize = worldSize;
            _rank = rank;
            _stage = stage;

            _partitionedParams = new List<Tensor>();
            _partitionedGrads = new List<Tensor>();
            _partitionedOptStates = new List<Tensor>();

            PartitionStates();
        }

        private void PartitionStates()
        {
            foreach (var param in _parameters)
            {
                int totalSize = param.Data.Length;
                int partitionSize = (totalSize + _worldSize - 1) / _worldSize;
                int start = _rank * partitionSize;
                int end = Math.Min(start + partitionSize, totalSize);
                int actualSize = Math.Max(0, end - start);

                if (actualSize > 0)
                {
                    // Create partitioned tensors
                    var partitionedParam = Tensor.Zeros(actualSize);
                    var partitionedGrad = Tensor.Zeros(actualSize);

                    // Copy data for this partition
                    if (_stage == ZeROStage.Stage3)
                    {
                        Array.Copy(param.Data, start, partitionedParam.Data, 0, actualSize);
                        _partitionedParams.Add(partitionedParam);
                    }

                    _partitionedGrads.Add(partitionedGrad);
                }
            }
        }

        /// <summary>
        /// Performs a single optimization step with ZeRO partitioning.
        /// </summary>
        public void Step()
        {
            // Stage 2+: Reduce-scatter gradients
            if (_stage >= ZeROStage.Stage2)
            {
                ReduceScatterGradients();
            }

            // Update partitioned parameters
            _baseOptimizer.StepOptimizer();

            // Stage 3: AllGather parameters
            if (_stage == ZeROStage.Stage3)
            {
                AllGatherParameters();
            }
        }

        private void ReduceScatterGradients()
        {
            for (int p = 0; p < _parameters.Count; p++)
            {
                var param = _parameters[p];
                if (param.Grad == null) continue;

                int totalSize = param.Grad.Data.Length;
                int partitionSize = (totalSize + _worldSize - 1) / _worldSize;
                int start = _rank * partitionSize;
                int end = Math.Min(start + partitionSize, totalSize);

                // Copy relevant partition to local gradient buffer
                if (end > start && p < _partitionedGrads.Count)
                {
                    Array.Copy(param.Grad.Data, start, _partitionedGrads[p].Data, 0, end - start);
                }
            }
        }

        private void AllGatherParameters()
        {
            for (int p = 0; p < _parameters.Count; p++)
            {
                var param = _parameters[p];
                int totalSize = param.Data.Length;
                int partitionSize = (totalSize + _worldSize - 1) / _worldSize;

                // In a real distributed setup, this would be an MPI AllGather
                // Here we simulate by copying from partitioned params
                if (p < _partitionedParams.Count)
                {
                    int start = _rank * partitionSize;
                    int end = Math.Min(start + partitionSize, totalSize);
                    if (end > start)
                    {
                        Array.Copy(_partitionedParams[p].Data, 0, param.Data, start, end - start);
                    }
                }
            }
        }

        /// <summary>Public API</summary>
        public void ZeroGrad()
        {
            foreach (var param in _parameters)
                param.ZeroGrad();

            foreach (var grad in _partitionedGrads)
                Array.Clear(grad.Data);
        }

        /// <summary>
        /// Estimates memory savings from ZeRO partitioning.
        /// </summary>
        public long EstimateMemorySavings()
        {
            long totalParams = _parameters.Sum(p => p.Data.Length * sizeof(double));

            return _stage switch
            {
                ZeROStage.Stage1 => totalParams * 4 / _worldSize,  // Optimizer states (K=4 for Adam)
                ZeROStage.Stage2 => totalParams * 6 / _worldSize,  // Gradients + optimizer states
                ZeROStage.Stage3 => totalParams * 8 / _worldSize,  // Parameters + gradients + optimizer states
                _ => 0
            };
        }

        /// <summary>Public API</summary>
        public override string ToString() =>
            $"ZeRO(stage={_stage}, world_size={_worldSize}, rank={_rank})";
    }

    /// <summary>
    /// CPU offloading for optimizer states (ZeRO-Offload style).
    /// Moves optimizer states to CPU to reduce GPU memory usage.
    /// </summary>
    public class OffloadOptimizer : GpuOptimizer
    {
        private readonly List<double[]> _cpuM;  // First moment on CPU
        private readonly List<double[]> _cpuV;  // Second moment on CPU
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;

        /// <summary>Public API</summary>
        public OffloadOptimizer(
            IEnumerable<Tensor> parameters,
            double lr = 0.001,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double eps = 1e-8,
            double weightDecay = 0.01,
            GpuAccelerator? gpu = null)
            : base(parameters, lr, gpu)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;

            // Allocate optimizer states on CPU
            _cpuM = _parameters.Select(p => new double[p.Data.Length]).ToList();
            _cpuV = _parameters.Select(p => new double[p.Data.Length]).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;
            double biasCorrection1 = 1 - Math.Pow(_beta1, _step);
            double biasCorrection2 = 1 - Math.Pow(_beta2, _step);

            // Parallel update on CPU
            System.Threading.Tasks.Parallel.For(0, _parameters.Count, i =>
            {
                var param = _parameters[i];
                if (param.Grad == null) return;

                var grad = param.Grad.Data;
                var m = _cpuM[i];
                var v = _cpuV[i];
                var data = param.Data;

                for (int j = 0; j < data.Length; j++)
                {
                    // Weight decay
                    if (_weightDecay != 0)
                    {
                        data[j] -= _weightDecay * _lr * data[j];
                    }

                    // Update moments on CPU
                    m[j] = _beta1 * m[j] + (1 - _beta1) * grad[j];
                    v[j] = _beta2 * v[j] + (1 - _beta2) * grad[j] * grad[j];

                    // Bias correction
                    double mHat = m[j] / biasCorrection1;
                    double vHat = v[j] / biasCorrection2;

                    // Update
                    data[j] -= _lr * mHat / (Math.Sqrt(vHat) + _eps);
                }
            });
        }

        /// <summary>
        /// Returns CPU memory used for optimizer states in bytes.
        /// </summary>
        public long CpuMemoryUsed()
        {
            return _cpuM.Sum(m => m.Length * sizeof(double)) +
                   _cpuV.Sum(v => v.Length * sizeof(double));
        }

        /// <summary>Public API</summary>
        public override string ToString() =>
            $"OffloadOptimizer(lr={_lr}, cpu_mem={CpuMemoryUsed() / 1024.0 / 1024.0:F2}MB)";
    }

    /// <summary>
    /// Gradient accumulation for effective larger batch sizes.
    /// </summary>
    public class GradientAccumulator
    {
        private readonly List<Tensor> _parameters;
        private readonly int _accumulationSteps;
        private int _currentStep;

        /// <summary>Public API</summary>
        public GradientAccumulator(IEnumerable<Tensor> parameters, int accumulationSteps)
        {
            _parameters = parameters.ToList();
            _accumulationSteps = accumulationSteps;
            _currentStep = 0;
        }

        /// <summary>
        /// Accumulates gradients. Returns true when ready for optimizer step.
        /// </summary>
        public bool Accumulate()
        {
            _currentStep++;

            if (_currentStep >= _accumulationSteps)
            {
                // Scale gradients by accumulation steps
                double scale = 1.0 / _accumulationSteps;
                foreach (var param in _parameters)
                {
                    if (param.Grad != null)
                    {
                        for (int i = 0; i < param.Grad.Data.Length; i++)
                        {
                            param.Grad.Data[i] *= scale;
                        }
                    }
                }

                _currentStep = 0;
                return true;
            }

            return false;
        }

        /// <summary>Public API</summary>
        public void Reset()
        {
            _currentStep = 0;
        }

        /// <summary>Public API</summary>
        public int AccumulationSteps => _accumulationSteps;
        /// <summary>Public API</summary>
        public int CurrentStep => _currentStep;
    }
}
#endif // HAS_GPU