using System;
using System.Diagnostics;
using System.Linq;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// Executes computation graph nodes on the GPU.
    ///
    /// Features:
    /// - Supports all standard and fused operations
    /// - Automatic kernel selection based on tensor sizes
    /// - Memory-efficient execution with tensor reuse
    /// - Profiling support for performance analysis
    /// </summary>
    public class GraphExecutor
    {
        private readonly Accelerator _accelerator;
        private readonly HighPerformanceKernels _hpKernels;
        private readonly OperatorFusion _fusion;

        // Standard kernels
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _addKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _subKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _mulKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _divKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _reluKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _geluKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _sigmoidKernel;
        private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _tanhKernel;
        private readonly Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _matmulKernel;

        private bool _enableProfiling = true;

        /// <summary>Public API</summary>
        public GraphExecutor(Accelerator accelerator)
        {
            _accelerator = accelerator;
            _hpKernels = new HighPerformanceKernels(accelerator);
            _fusion = new OperatorFusion(accelerator);

            // Compile standard kernels
            _addKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(AddKernelImpl);
            _subKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SubKernelImpl);
            _mulKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(MulKernelImpl);
            _divKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(DivKernelImpl);
            _reluKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ReLUKernelImpl);
            _geluKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(GELUKernelImpl);
            _sigmoidKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SigmoidKernelImpl);
            _tanhKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(TanhKernelImpl);
            _matmulKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(MatMulKernelImpl);
        }

        /// <summary>
        /// Execute a single node in the graph
        /// </summary>
        public void ExecuteNode(GraphNode node)
        {
            if (node.IsComputed || node.IsFused) return;

            // Ensure all inputs are computed
            foreach (var input in node.Inputs)
            {
                if (!input.IsComputed)
                {
                    ExecuteNode(input);
                }
            }

            var sw = _enableProfiling ? Stopwatch.StartNew() : null;

            // Execute based on operation type
            switch (node.Op)
            {
                case OpType.Add:
                    ExecuteAdd(node);
                    break;
                case OpType.Sub:
                    ExecuteSub(node);
                    break;
                case OpType.Mul:
                    ExecuteMul(node);
                    break;
                case OpType.Div:
                    ExecuteDiv(node);
                    break;
                case OpType.ReLU:
                    ExecuteReLU(node);
                    break;
                case OpType.GELU:
                    ExecuteGELU(node);
                    break;
                case OpType.Sigmoid:
                    ExecuteSigmoid(node);
                    break;
                case OpType.Tanh:
                    ExecuteTanh(node);
                    break;
                case OpType.Softmax:
                    ExecuteSoftmax(node);
                    break;
                case OpType.MatMul:
                    ExecuteMatMul(node);
                    break;
                case OpType.LayerNorm:
                    ExecuteLayerNorm(node);
                    break;
                case OpType.RMSNorm:
                    ExecuteRMSNorm(node);
                    break;
                case OpType.ScaledDotProductAttention:
                case OpType.FlashAttention:
                case OpType.FusedFlashAttention:
                    ExecuteFlashAttention(node);
                    break;
                case OpType.Reshape:
                    ExecuteReshape(node);
                    break;
                case OpType.Transpose:
                    ExecuteTranspose(node);
                    break;
                case OpType.Sum:
                    ExecuteSum(node);
                    break;
                case OpType.Mean:
                    ExecuteMean(node);
                    break;

                // Fused operations
                case OpType.FusedLinearReLU:
                case OpType.FusedMatMulBiasReLU:
                    ExecuteFusedLinearReLU(node);
                    break;
                case OpType.FusedLinearGELU:
                    ExecuteFusedLinearGELU(node);
                    break;
                case OpType.FusedLayerNormDropout:
                    ExecuteFusedLayerNormDropout(node);
                    break;
                case OpType.FusedResidualAdd:
                    ExecuteFusedResidualAdd(node);
                    break;
                case OpType.FusedQKVProjection:
                    ExecuteFusedQKVProjection(node);
                    break;

                default:
                    throw new NotSupportedException($"Operation {node.Op} not yet implemented in executor");
            }

            if (sw != null)
            {
                _accelerator.Synchronize();
                node.ExecutionTimeNs = sw.ElapsedTicks * 100; // Approximate ns
                node.ExecutionCount++;
            }

            node.IsComputed = true;
        }

        #region Standard Operations

        private void ExecuteAdd(GraphNode node)
        {
            var a = node.Inputs[0].OutputTensor!;
            var b = node.Inputs[1].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _addKernel(result.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteSub(GraphNode node)
        {
            var a = node.Inputs[0].OutputTensor!;
            var b = node.Inputs[1].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _subKernel(result.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteMul(GraphNode node)
        {
            var a = node.Inputs[0].OutputTensor!;
            var b = node.Inputs[1].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _mulKernel(result.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteDiv(GraphNode node)
        {
            var a = node.Inputs[0].OutputTensor!;
            var b = node.Inputs[1].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _divKernel(result.Size, a.Buffer.View, b.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteReLU(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _reluKernel(result.Size, input.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteGELU(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _geluKernel(result.Size, input.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteSigmoid(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _sigmoidKernel(result.Size, input.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteTanh(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var result = new GpuTensor(_accelerator, node.OutputShape);

            _tanhKernel(result.Size, input.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();

            node.OutputTensor = result;
        }

        private void ExecuteSoftmax(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            // Use high-performance softmax from _hpKernels
            var result = _hpKernels.Softmax(input);
            node.OutputTensor = result;
        }

        private void ExecuteMatMul(GraphNode node)
        {
            var a = node.Inputs[0].OutputTensor!;
            var b = node.Inputs[1].OutputTensor!;

            // Use high-performance tiled matmul for larger matrices
            var result = _hpKernels.TiledMatMul(a, b);
            node.OutputTensor = result;
        }

        private void ExecuteLayerNorm(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var gamma = node.Inputs[1].OutputTensor!;
            var beta = node.Inputs[2].OutputTensor!;
            var eps = node.GetAttribute<float>("eps", 1e-5f);

            var result = _hpKernels.FusedLayerNorm(input, gamma, beta, eps);
            node.OutputTensor = result;
        }

        private void ExecuteRMSNorm(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var gamma = node.Inputs[1].OutputTensor!;
            var eps = node.GetAttribute<float>("eps", 1e-5f);

            // RMSNorm: x * rsqrt(mean(x^2) + eps) * gamma
            var result = _hpKernels.RMSNorm(input, gamma, eps);
            node.OutputTensor = result;
        }

        private void ExecuteFlashAttention(GraphNode node)
        {
            var query = node.Inputs[0].OutputTensor!;
            var key = node.Inputs[1].OutputTensor!;
            var value = node.Inputs[2].OutputTensor!;

            // Use FlashAttention-2 for memory efficiency
            var result = _hpKernels.FlashAttention2(query, key, value);
            node.OutputTensor = result;
        }

        private void ExecuteReshape(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            // Reshape is a view operation - no copy needed
            node.OutputTensor = input.Reshape(node.OutputShape);
        }

        private void ExecuteTranspose(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var dim0 = node.GetAttribute<int>("dim0", -2);
            var dim1 = node.GetAttribute<int>("dim1", -1);

            // For 2D transpose, we can use a simple kernel
            // For higher dims, need more complex logic
            var result = _hpKernels.Transpose(input, dim0, dim1);
            node.OutputTensor = result;
        }

        private void ExecuteSum(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var axis = node.GetAttribute<int?>("axis", null);
            var keepDims = node.GetAttribute<bool>("keepDims", false);

            var result = _hpKernels.Sum(input, axis, keepDims);
            node.OutputTensor = result;
        }

        private void ExecuteMean(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var axis = node.GetAttribute<int?>("axis", null);
            var keepDims = node.GetAttribute<bool>("keepDims", false);

            var result = _hpKernels.Mean(input, axis, keepDims);
            node.OutputTensor = result;
        }

        #endregion

        #region Fused Operations

        private void ExecuteFusedLinearReLU(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var weight = node.Inputs[1].OutputTensor!;
            var bias = node.Inputs.Count > 2 ? node.Inputs[2].OutputTensor! : null;

            var result = bias != null
                ? _fusion.FusedLinearRelu(input, weight, bias)
                : _hpKernels.FusedMatMulBiasReLU(input, weight, GpuTensor.Zeros(_accelerator, weight.Shape[0]));

            node.OutputTensor = result;
        }

        private void ExecuteFusedLinearGELU(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var weight = node.Inputs[1].OutputTensor!;
            var bias = node.Inputs.Count > 2 ? node.Inputs[2].OutputTensor! : null;

            // MatMul + Bias + GELU fused
            var linear = _hpKernels.TiledMatMul(input, weight);
            if (bias != null)
            {
                var result = _fusion.FusedBiasGelu(linear, bias);
                node.OutputTensor = result;
                linear.Dispose();
            }
            else
            {
                // Just apply GELU without bias
                var result = new GpuTensor(_accelerator, node.OutputShape);
                _geluKernel(result.Size, linear.Buffer.View, result.Buffer.View);
                _accelerator.Synchronize();
                node.OutputTensor = result;
                linear.Dispose();
            }
        }

        private void ExecuteFusedLayerNormDropout(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var gamma = node.Inputs[1].OutputTensor!;
            var beta = node.Inputs[2].OutputTensor!;
            var dropoutProb = node.GetAttribute<float>("dropout", 0f);
            var eps = node.GetAttribute<float>("eps", 1e-5f);

            var result = _fusion.FusedLayerNormDropout(input, gamma, beta, dropoutProb, eps);
            node.OutputTensor = result;
        }

        private void ExecuteFusedResidualAdd(GraphNode node)
        {
            var input = node.Inputs[0].OutputTensor!;
            var residual = node.Inputs[1].OutputTensor!;

            var result = _fusion.FusedResidualAdd(input, residual);
            node.OutputTensor = result;
        }

        private void ExecuteFusedQKVProjection(GraphNode node)
        {
            // Fused QKV projection: compute Q, K, V in a single matmul
            var input = node.Inputs[0].OutputTensor!;
            var wq = node.Inputs[1].OutputTensor!;
            var wk = node.Inputs[2].OutputTensor!;
            var wv = node.Inputs[3].OutputTensor!;

            // Concatenate weights and do single matmul
            // This is more efficient than 3 separate matmuls
            var result = _hpKernels.FusedQKVProjection(input, wq, wk, wv);
            node.OutputTensor = result;
        }

        #endregion

        #region Kernel Implementations

        private static void AddKernelImpl(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
        {
            result[index] = a[index] + b[index];
        }

        private static void SubKernelImpl(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
        {
            result[index] = a[index] - b[index];
        }

        private static void MulKernelImpl(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
        {
            result[index] = a[index] * b[index];
        }

        private static void DivKernelImpl(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
        {
            result[index] = a[index] / b[index];
        }

        private static void ReLUKernelImpl(Index1D index, ArrayView<float> input, ArrayView<float> result)
        {
            result[index] = XMath.Max(0f, input[index]);
        }

        private static void GELUKernelImpl(Index1D index, ArrayView<float> input, ArrayView<float> result)
        {
            float x = input[index];
            // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            const float sqrt2OverPi = 0.7978845608f;
            const float coeff = 0.044715f;
            float x3 = x * x * x;
            float inner = sqrt2OverPi * (x + coeff * x3);
            result[index] = 0.5f * x * (1f + XMath.Tanh(inner));
        }

        private static void SigmoidKernelImpl(Index1D index, ArrayView<float> input, ArrayView<float> result)
        {
            result[index] = 1f / (1f + XMath.Exp(-input[index]));
        }

        private static void TanhKernelImpl(Index1D index, ArrayView<float> input, ArrayView<float> result)
        {
            result[index] = XMath.Tanh(input[index]);
        }

        private static void MatMulKernelImpl(Index2D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result,
            int m, int k, int n)
        {
            int row = index.X;
            int col = index.Y;

            if (row >= m || col >= n) return;

            float sum = 0f;
            for (int i = 0; i < k; i++)
            {
                sum += a[row * k + i] * b[i * n + col];
            }
            result[row * n + col] = sum;
        }

        #endregion
    }
}