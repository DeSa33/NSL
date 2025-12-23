using System;
using System.Linq;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// Extension methods for HighPerformanceKernels providing additional operations
    /// required by the graph compiler and JIT system.
    /// </summary>
    public partial class HighPerformanceKernels
    {
        // Lazy initialization of extended kernels
        private FlashAttention2Engine? _flashAttention2;
        private Action<Index2D, ArrayView<float>, ArrayView<float>, int, int>? _transposeKernel;
        private Action<Index1D, ArrayView<float>, ArrayView<float>, int, int>? _sumAxisKernel;
        private Action<Index1D, ArrayView<float>, ArrayView<float>, int, int>? _meanAxisKernel;
        private Action<Index1D, ArrayView<float>, ArrayView<float>, int>? _softmaxKernel;

        /// <summary>
        /// Initialize extended kernels on first use
        /// </summary>
        private void EnsureExtendedKernelsInitialized()
        {
            if (_transposeKernel != null) return;

            _transposeKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, int, int>(TransposeKernelImpl);

            _sumAxisKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(SumAxisKernelImpl);

            _meanAxisKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(MeanAxisKernelImpl);

            _softmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(SoftmaxKernelImpl);
        }

        /// <summary>
        /// FlashAttention-2 forward pass.
        /// Memory-efficient attention that avoids materializing the N×N attention matrix.
        /// </summary>
        public GpuTensor FlashAttention2(GpuTensor query, GpuTensor key, GpuTensor value,
            bool causal = false, float dropoutProb = 0f)
        {
            _flashAttention2 ??= new FlashAttention2Engine(_accelerator);
            return _flashAttention2.Forward(query, key, value, null, causal);
        }

        /// <summary>
        /// Multi-head FlashAttention-2.
        /// Input shape: [batch, num_heads, seq_len, head_dim]
        /// </summary>
        public GpuTensor MultiHeadFlashAttention(GpuTensor query, GpuTensor key, GpuTensor value,
            bool causal = false)
        {
            _flashAttention2 ??= new FlashAttention2Engine(_accelerator);
            return _flashAttention2.MultiHeadForward(query, key, value, causal);
        }

        /// <summary>
        /// Transpose a 2D tensor or swap two dimensions of a higher-dim tensor.
        /// </summary>
        public GpuTensor Transpose(GpuTensor input, int dim0 = -2, int dim1 = -1)
        {
            EnsureExtendedKernelsInitialized();

            var shape = input.Shape;
            int ndim = shape.Length;

            if (dim0 < 0) dim0 += ndim;
            if (dim1 < 0) dim1 += ndim;

            var newShape = (int[])shape.Clone();
            (newShape[dim0], newShape[dim1]) = (newShape[dim1], newShape[dim0]);

            var result = new GpuTensor(_accelerator, newShape);

            if (ndim == 2)
            {
                var gridDim = new Index2D(shape[0], shape[1]);
                _transposeKernel!(gridDim, input.Buffer.View, result.Buffer.View, shape[0], shape[1]);
            }
            else
            {
                var data = input.ToArray();
                var transposed = new float[data.Length];
                TransposeHigherDim(data, transposed, shape, newShape, dim0, dim1);
                result.Buffer.CopyFromCPU(transposed);
            }

            _accelerator.Synchronize();
            return result;
        }

        /// <summary>
        /// Sum reduction along an axis.
        /// </summary>
        public GpuTensor Sum(GpuTensor input, int? axis = null, bool keepDims = false)
        {
            EnsureExtendedKernelsInitialized();

            if (axis == null)
            {
                var sumResult = WarpReduceSum(input, -1);
                return keepDims ? sumResult.Reshape(Enumerable.Repeat(1, input.NDim).ToArray()) : sumResult;
            }

            int normalizedAxis = axis.Value < 0 ? input.NDim + axis.Value : axis.Value;
            var shape = input.Shape;
            var axisSize = shape[normalizedAxis];
            var innerSize = shape.Skip(normalizedAxis + 1).Aggregate(1, (a, b) => a * b);

            int[] resultShape;
            if (keepDims)
            {
                resultShape = (int[])shape.Clone();
                resultShape[normalizedAxis] = 1;
            }
            else
            {
                resultShape = shape.Where((_, i) => i != normalizedAxis).ToArray();
                if (resultShape.Length == 0) resultShape = new[] { 1 };
            }

            var result = new GpuTensor(_accelerator, resultShape);
            int outerInner = result.Size;

            _sumAxisKernel!(outerInner, input.Buffer.View, result.Buffer.View, axisSize, innerSize);
            _accelerator.Synchronize();

            return result;
        }

        /// <summary>
        /// Mean reduction along an axis.
        /// </summary>
        public GpuTensor Mean(GpuTensor input, int? axis = null, bool keepDims = false)
        {
            EnsureExtendedKernelsInitialized();

            if (axis == null)
            {
                var sum = Sum(input, null, keepDims);
                return ScaleInPlace(sum, 1.0f / input.Size);
            }

            int normalizedAxis = axis.Value < 0 ? input.NDim + axis.Value : axis.Value;
            var shape = input.Shape;
            var axisSize = shape[normalizedAxis];
            var innerSize = shape.Skip(normalizedAxis + 1).Aggregate(1, (a, b) => a * b);

            int[] resultShape;
            if (keepDims)
            {
                resultShape = (int[])shape.Clone();
                resultShape[normalizedAxis] = 1;
            }
            else
            {
                resultShape = shape.Where((_, i) => i != normalizedAxis).ToArray();
                if (resultShape.Length == 0) resultShape = new[] { 1 };
            }

            var result = new GpuTensor(_accelerator, resultShape);
            int outerInner = result.Size;

            _meanAxisKernel!(outerInner, input.Buffer.View, result.Buffer.View, axisSize, innerSize);
            _accelerator.Synchronize();

            return result;
        }

        /// <summary>
        /// Softmax activation (numerically stable).
        /// </summary>
        public GpuTensor Softmax(GpuTensor input, int axis = -1)
        {
            EnsureExtendedKernelsInitialized();

            var shape = input.Shape;
            int normalizedAxis = axis < 0 ? shape.Length + axis : axis;
            var axisSize = shape[normalizedAxis];
            var outerSize = input.Size / axisSize;

            var result = new GpuTensor(_accelerator, shape);

            _softmaxKernel!(outerSize, input.Buffer.View, result.Buffer.View, axisSize);
            _accelerator.Synchronize();

            return result;
        }

        /// <summary>
        /// Fused Q, K, V projection: compute all three projections in one kernel.
        /// </summary>
        public GpuTensor FusedQKVProjection(GpuTensor input, GpuTensor wQ, GpuTensor wK, GpuTensor wV)
        {
            int seqLen = input.Shape[^2];
            int headDim = wQ.Shape[^1];
            float scale = 1.0f / MathF.Sqrt(headDim);
            return FusedQKVAttention(input, wQ, wK, wV, scale);
        }

        private GpuTensor ScaleInPlace(GpuTensor tensor, float scale)
        {
            var data = tensor.ToArray();
            for (int i = 0; i < data.Length; i++) data[i] *= scale;
            tensor.Buffer.CopyFromCPU(data);
            return tensor;
        }

        private void TransposeHigherDim(float[] input, float[] output, int[] inputShape, int[] outputShape, int dim0, int dim1)
        {
            int ndim = inputShape.Length;
            var inputStrides = ComputeStrides(inputShape);
            var outputStrides = ComputeStrides(outputShape);
            var indices = new int[ndim];

            for (int i = 0; i < input.Length; i++)
            {
                int remaining = i;
                for (int d = 0; d < ndim; d++)
                {
                    indices[d] = remaining / inputStrides[d];
                    remaining %= inputStrides[d];
                }
                (indices[dim0], indices[dim1]) = (indices[dim1], indices[dim0]);
                int outIdx = 0;
                for (int d = 0; d < ndim; d++) outIdx += indices[d] * outputStrides[d];
                output[outIdx] = input[i];
            }
        }

        private int[] ComputeStrides(int[] shape)
        {
            var strides = new int[shape.Length];
            strides[^1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
                strides[i] = strides[i + 1] * shape[i + 1];
            return strides;
        }

        #region Extended Kernel Implementations

        private static void TransposeKernelImpl(Index2D index, ArrayView<float> input, ArrayView<float> output, int rows, int cols)
        {
            int row = index.X, col = index.Y;
            if (row < rows && col < cols)
                output[col * rows + row] = input[row * cols + col];
        }

        private static void SumAxisKernelImpl(Index1D index, ArrayView<float> input, ArrayView<float> output, int axisSize, int innerSize)
        {
            int outerIdx = index / innerSize;
            int innerIdx = index % innerSize;
            float sum = 0f;
            for (int i = 0; i < axisSize; i++)
                sum += input[outerIdx * axisSize * innerSize + i * innerSize + innerIdx];
            output[index] = sum;
        }

        private static void MeanAxisKernelImpl(Index1D index, ArrayView<float> input, ArrayView<float> output, int axisSize, int innerSize)
        {
            int outerIdx = index / innerSize;
            int innerIdx = index % innerSize;
            float sum = 0f;
            for (int i = 0; i < axisSize; i++)
                sum += input[outerIdx * axisSize * innerSize + i * innerSize + innerIdx];
            output[index] = sum / axisSize;
        }

        private static void SoftmaxKernelImpl(Index1D rowIndex, ArrayView<float> input, ArrayView<float> output, int axisSize)
        {
            int offset = rowIndex * axisSize;
            float maxVal = input[offset];
            for (int i = 1; i < axisSize; i++)
                maxVal = XMath.Max(maxVal, input[offset + i]);

            float sumExp = 0f;
            for (int i = 0; i < axisSize; i++)
            {
                float expVal = XMath.Exp(input[offset + i] - maxVal);
                output[offset + i] = expVal;
                sumExp += expVal;
            }

            float invSum = 1f / sumExp;
            for (int i = 0; i < axisSize; i++)
                output[offset + i] *= invSum;
        }

        #endregion
    }
}
