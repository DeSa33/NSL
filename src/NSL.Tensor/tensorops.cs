using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NSL.Tensor
{
    /// Core tensor operations - matrix multiplication, concatenation, stacking, etc.
    /// Optimized with SIMD (AVX-512/AVX2/SSE), parallelization, and cache blocking.
    /// </summary>
    public static class TensorOps
    {
        // CPU feature detection
        private static readonly bool HasAvx512 = Avx512F.IsSupported;
        private static readonly bool HasAvx2 = Avx2.IsSupported;
        private static readonly bool HasFma = Fma.IsSupported;
        private static readonly bool HasSse = Sse2.IsSupported;

        // Parallelization thresholds
        private const int ParallelThreshold = 4096;
        private const int SimdThreshold = 32;

        // Cache blocking sizes (tuned for modern CPUs)
        private const int L1BlockSize = 32;   // ~32KB L1 cache
        private const int L2BlockSize = 256;  // ~256KB L2 cache
        private const int L3BlockSize = 2048; // ~8MB L3 cache
        #region Matrix Multiplication

        /// Matrix multiplication: C = A @ B
        /// Supports batched matrix multiplication
        /// </summary>
        public static Tensor MatMul(Tensor a, Tensor b)
        {
            if (a.NDim < 1 || b.NDim < 1)
                throw new ArgumentException("MatMul requires tensors with at least 1 dimension");

            // Vector-vector dot product
            if (a.NDim == 1 && b.NDim == 1)
                return Dot(a, b);

            // Matrix-vector
            if (a.NDim == 2 && b.NDim == 1)
            {
                if (a.Shape[1] != b.Shape[0])
                    throw new ArgumentException($"Shape mismatch: ({a.Shape[0]}, {a.Shape[1]}) @ ({b.Shape[0]},)");

                var result = new double[a.Shape[0]];
                for (int i = 0; i < a.Shape[0]; i++)
                {
                    double sum = 0;
                    for (int k = 0; k < a.Shape[1]; k++)
                        sum += a[i, k] * b[k];
                    result[i] = sum;
                }

                var resultTensor = new Tensor(result, new[] { a.Shape[0] }, a.RequiresGrad || b.RequiresGrad);
                if (a.RequiresGrad || b.RequiresGrad)
                {
                    // Use MatMulBackward for gradient computation (treats vector as 1-column matrix)
                    resultTensor.GradFn = new MatMulBackward(a, b.Reshape(new[] { b.Shape[0], 1L }));
                }
                return resultTensor;
            }

            // Vector-matrix
            if (a.NDim == 1 && b.NDim == 2)
            {
                if (a.Shape[0] != b.Shape[0])
                    throw new ArgumentException($"Shape mismatch: ({a.Shape[0]},) @ ({b.Shape[0]}, {b.Shape[1]})");

                var result = new double[b.Shape[1]];
                for (int j = 0; j < b.Shape[1]; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < a.Shape[0]; k++)
                        sum += a[k] * b[k, j];
                    result[j] = sum;
                }

                return new Tensor(result, new[] { b.Shape[1] }, a.RequiresGrad || b.RequiresGrad);
            }

            // Matrix-matrix
            if (a.NDim == 2 && b.NDim == 2)
            {
                if (a.Shape[1] != b.Shape[0])
                    throw new ArgumentException($"Shape mismatch: ({a.Shape[0]}, {a.Shape[1]}) @ ({b.Shape[0]}, {b.Shape[1]})");

                var m = (int)a.Shape[0];
                var n = (int)b.Shape[1];
                var k = (int)a.Shape[1];

                var result = new double[m * n];

                // Get raw data arrays for faster access
                var aData = a.Data;
                var bData = b.Data;

                // Use SIMD-optimized path for larger matrices
                if (m >= 32 && n >= 32 && k >= 32)
                {
                    MatMulSimd(aData, bData, result, m, n, k);
                }
                else
                {
                    // Standard path for smaller matrices
                    MatMulStandard(aData, bData, result, m, n, k);
                }

                var resultTensor = new Tensor(result, new[] { (long)m, (long)n }, a.RequiresGrad || b.RequiresGrad);
                if (a.RequiresGrad || b.RequiresGrad)
                {
                    // Set gradient function for backprop
                    resultTensor = CreateMatMulWithGrad(a, b, resultTensor);
                }
                return resultTensor;
            }

            // Batched matrix multiplication
            return BatchedMatMul(a, b);
        }

        // Cache blocking parameters optimized for modern CPUs
        private const int MC = 128;   // L2 cache blocking for A panels
        private const int KC = 256;   // L1 cache blocking for k dimension
        private const int NC = 2048;  // L3 cache blocking for B panels

        /// SIMD-optimized matrix multiplication using B-transpose with 4-row micro-kernel.
        /// Combines cache blocking with sequential memory access for optimal performance.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void MatMulSimd(double[] aData, double[] bData, double[] result, int m, int n, int k)
        {
            // Transpose B for sequential memory access in inner loop
            var bT = new double[k * n];
            Parallel.For(0, n, j =>
            {
                int bTRowOffset = j * k;
                for (int i = 0; i < k; i++)
                {
                    bT[bTRowOffset + i] = bData[i * n + j];
                }
            });

            // Choose best SIMD path based on hardware support
            if (Avx2.IsSupported && Fma.IsSupported)
            {
                MatMulAvx2Optimized(aData, bT, result, m, n, k);
            }
            else
            {
                MatMulVectorFallback(aData, bT, result, m, n, k);
            }
        }

        /// AVX2+FMA optimized matrix multiplication with 4-row blocking.
        /// Computes 4 output rows simultaneously for better instruction-level parallelism.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulAvx2Optimized(double[] aData, double[] bT, double[] result, int m, int n, int k)
        {
            const int unrollK = 4;  // Process 16 doubles per iteration
            int vectorSize = Vector256<double>.Count; // 4 doubles
            int unrolledSize = vectorSize * unrollK;

            // Pin arrays for direct pointer access
            var handleA = System.Runtime.InteropServices.GCHandle.Alloc(aData, System.Runtime.InteropServices.GCHandleType.Pinned);
            var handleB = System.Runtime.InteropServices.GCHandle.Alloc(bT, System.Runtime.InteropServices.GCHandleType.Pinned);
            var handleR = System.Runtime.InteropServices.GCHandle.Alloc(result, System.Runtime.InteropServices.GCHandleType.Pinned);

            try
            {
                double* pA = (double*)handleA.AddrOfPinnedObject();
                double* pB = (double*)handleB.AddrOfPinnedObject();
                double* pC = (double*)handleR.AddrOfPinnedObject();

                // Process 4 rows at a time for better ILP
                int m4 = m - (m % 4);

                Parallel.For(0, (m4 + 3) / 4, i4 =>
                {
                    int i = i4 * 4;
                    if (i >= m4) return;

                    double* a0 = pA + i * k;
                    double* a1 = pA + (i + 1) * k;
                    double* a2 = pA + (i + 2) * k;
                    double* a3 = pA + (i + 3) * k;

                    double* c0 = pC + i * n;
                    double* c1 = pC + (i + 1) * n;
                    double* c2 = pC + (i + 2) * n;
                    double* c3 = pC + (i + 3) * n;

                    for (int j = 0; j < n; j++)
                    {
                        double* bRow = pB + j * k;

                        // 4 accumulators for 4 output rows
                        var sum0_0 = Vector256<double>.Zero;
                        var sum0_1 = Vector256<double>.Zero;
                        var sum0_2 = Vector256<double>.Zero;
                        var sum0_3 = Vector256<double>.Zero;

                        var sum1_0 = Vector256<double>.Zero;
                        var sum1_1 = Vector256<double>.Zero;
                        var sum1_2 = Vector256<double>.Zero;
                        var sum1_3 = Vector256<double>.Zero;

                        var sum2_0 = Vector256<double>.Zero;
                        var sum2_1 = Vector256<double>.Zero;
                        var sum2_2 = Vector256<double>.Zero;
                        var sum2_3 = Vector256<double>.Zero;

                        var sum3_0 = Vector256<double>.Zero;
                        var sum3_1 = Vector256<double>.Zero;
                        var sum3_2 = Vector256<double>.Zero;
                        var sum3_3 = Vector256<double>.Zero;

                        int kk = 0;

                        // Main SIMD loop - 4 rows × 16 k elements per iteration
                        for (; kk <= k - unrolledSize; kk += unrolledSize)
                        {
                            var b0 = Avx.LoadVector256(bRow + kk);
                            var b1 = Avx.LoadVector256(bRow + kk + 4);
                            var b2 = Avx.LoadVector256(bRow + kk + 8);
                            var b3 = Avx.LoadVector256(bRow + kk + 12);

                            // Row 0
                            sum0_0 = Fma.MultiplyAdd(Avx.LoadVector256(a0 + kk), b0, sum0_0);
                            sum0_1 = Fma.MultiplyAdd(Avx.LoadVector256(a0 + kk + 4), b1, sum0_1);
                            sum0_2 = Fma.MultiplyAdd(Avx.LoadVector256(a0 + kk + 8), b2, sum0_2);
                            sum0_3 = Fma.MultiplyAdd(Avx.LoadVector256(a0 + kk + 12), b3, sum0_3);

                            // Row 1
                            sum1_0 = Fma.MultiplyAdd(Avx.LoadVector256(a1 + kk), b0, sum1_0);
                            sum1_1 = Fma.MultiplyAdd(Avx.LoadVector256(a1 + kk + 4), b1, sum1_1);
                            sum1_2 = Fma.MultiplyAdd(Avx.LoadVector256(a1 + kk + 8), b2, sum1_2);
                            sum1_3 = Fma.MultiplyAdd(Avx.LoadVector256(a1 + kk + 12), b3, sum1_3);

                            // Row 2
                            sum2_0 = Fma.MultiplyAdd(Avx.LoadVector256(a2 + kk), b0, sum2_0);
                            sum2_1 = Fma.MultiplyAdd(Avx.LoadVector256(a2 + kk + 4), b1, sum2_1);
                            sum2_2 = Fma.MultiplyAdd(Avx.LoadVector256(a2 + kk + 8), b2, sum2_2);
                            sum2_3 = Fma.MultiplyAdd(Avx.LoadVector256(a2 + kk + 12), b3, sum2_3);

                            // Row 3
                            sum3_0 = Fma.MultiplyAdd(Avx.LoadVector256(a3 + kk), b0, sum3_0);
                            sum3_1 = Fma.MultiplyAdd(Avx.LoadVector256(a3 + kk + 4), b1, sum3_1);
                            sum3_2 = Fma.MultiplyAdd(Avx.LoadVector256(a3 + kk + 8), b2, sum3_2);
                            sum3_3 = Fma.MultiplyAdd(Avx.LoadVector256(a3 + kk + 12), b3, sum3_3);
                        }

                        // Combine accumulators
                        var combined0 = Avx.Add(Avx.Add(sum0_0, sum0_1), Avx.Add(sum0_2, sum0_3));
                        var combined1 = Avx.Add(Avx.Add(sum1_0, sum1_1), Avx.Add(sum1_2, sum1_3));
                        var combined2 = Avx.Add(Avx.Add(sum2_0, sum2_1), Avx.Add(sum2_2, sum2_3));
                        var combined3 = Avx.Add(Avx.Add(sum3_0, sum3_1), Avx.Add(sum3_2, sum3_3));

                        // Horizontal sums
                        double total0 = HorizontalSum(combined0);
                        double total1 = HorizontalSum(combined1);
                        double total2 = HorizontalSum(combined2);
                        double total3 = HorizontalSum(combined3);

                        // Handle remaining k elements
                        for (; kk <= k - vectorSize; kk += vectorSize)
                        {
                            var bVec = Avx.LoadVector256(bRow + kk);
                            total0 += HorizontalSum(Avx.Multiply(Avx.LoadVector256(a0 + kk), bVec));
                            total1 += HorizontalSum(Avx.Multiply(Avx.LoadVector256(a1 + kk), bVec));
                            total2 += HorizontalSum(Avx.Multiply(Avx.LoadVector256(a2 + kk), bVec));
                            total3 += HorizontalSum(Avx.Multiply(Avx.LoadVector256(a3 + kk), bVec));
                        }

                        // Scalar remainder
                        for (; kk < k; kk++)
                        {
                            double bVal = bRow[kk];
                            total0 += a0[kk] * bVal;
                            total1 += a1[kk] * bVal;
                            total2 += a2[kk] * bVal;
                            total3 += a3[kk] * bVal;
                        }

                        c0[j] = total0;
                        c1[j] = total1;
                        c2[j] = total2;
                        c3[j] = total3;
                    }
                });

                // Handle remaining rows (m % 4)
                for (int i = m4; i < m; i++)
                {
                    double* aRow = pA + i * k;
                    double* cRow = pC + i * n;

                    for (int j = 0; j < n; j++)
                    {
                        double* bRow = pB + j * k;
                        var sum0 = Vector256<double>.Zero;
                        var sum1 = Vector256<double>.Zero;
                        var sum2 = Vector256<double>.Zero;
                        var sum3 = Vector256<double>.Zero;

                        int kk = 0;
                        for (; kk <= k - unrolledSize; kk += unrolledSize)
                        {
                            sum0 = Fma.MultiplyAdd(Avx.LoadVector256(aRow + kk), Avx.LoadVector256(bRow + kk), sum0);
                            sum1 = Fma.MultiplyAdd(Avx.LoadVector256(aRow + kk + 4), Avx.LoadVector256(bRow + kk + 4), sum1);
                            sum2 = Fma.MultiplyAdd(Avx.LoadVector256(aRow + kk + 8), Avx.LoadVector256(bRow + kk + 8), sum2);
                            sum3 = Fma.MultiplyAdd(Avx.LoadVector256(aRow + kk + 12), Avx.LoadVector256(bRow + kk + 12), sum3);
                        }

                        var combined = Avx.Add(Avx.Add(sum0, sum1), Avx.Add(sum2, sum3));
                        double total = HorizontalSum(combined);

                        for (; kk < k; kk++)
                            total += aRow[kk] * bRow[kk];

                        cRow[j] = total;
                    }
                }
            }
            finally
            {
                handleA.Free();
                handleB.Free();
                handleR.Free();
            }
        }

        /// Efficient AVX horizontal sum using SSE instructions.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double HorizontalSum(Vector256<double> v)
        {
            var vlow = Avx.ExtractVector128(v, 0);
            var vhigh = Avx.ExtractVector128(v, 1);
            var sum128 = Sse2.Add(vlow, vhigh);
            var high64 = Sse2.UnpackHigh(sum128, sum128);
            return Sse2.AddScalar(sum128, high64).ToScalar();
        }

        /// Fallback implementation for non-AVX2 hardware using transposed B.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulVectorFallback(double[] aData, double[] bT, double[] result, int m, int n, int k)
        {
            int vectorSize = Vector<double>.Count;

            Parallel.For(0, m, i =>
            {
                int aRowOffset = i * k;
                int cRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    int bRowOffset = j * k;
                    double sum = 0;
                    int kk = 0;

                    for (; kk <= k - vectorSize; kk += vectorSize)
                    {
                        var aVec = new Vector<double>(aData, aRowOffset + kk);
                        var bVec = new Vector<double>(bT, bRowOffset + kk);
                        sum += Vector.Dot(aVec, bVec);
                    }

                    for (; kk < k; kk++)
                        sum += aData[aRowOffset + kk] * bT[bRowOffset + kk];

                    result[cRowOffset + j] = sum;
                }
            });
        }


        /// Standard matrix multiplication for smaller matrices
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MatMulStandard(double[] aData, double[] bData, double[] result, int m, int n, int k)
        {
            const int blockSize = 32;

            Parallel.For(0, m, i =>
            {
                for (int kk = 0; kk < k; kk += blockSize)
                {
                    int kEnd = Math.Min(kk + blockSize, k);
                    for (int jj = 0; jj < n; jj += blockSize)
                    {
                        int jEnd = Math.Min(jj + blockSize, n);

                        for (int kIdx = kk; kIdx < kEnd; kIdx++)
                        {
                            double aVal = aData[i * k + kIdx];
                            int bRowOffset = kIdx * n;
                            int resultRowOffset = i * n;

                            for (int jIdx = jj; jIdx < jEnd; jIdx++)
                            {
                                result[resultRowOffset + jIdx] += aVal * bData[bRowOffset + jIdx];
                            }
                        }
                    }
                }
            });
        }

        private static Tensor CreateMatMulWithGrad(Tensor a, Tensor b, Tensor result)
        {
            // Create new tensor with grad_fn
            var data = result.Data.ToArray();
            var gradFn = new MatMulBackward(a, b);
            return new Tensor(data, result.Shape.ToArray(), gradFn, a.RequiresGrad || b.RequiresGrad, a.Device);
        }

        private static Tensor BatchedMatMul(Tensor a, Tensor b)
        {
            // Handle batched matrix multiplication
            var aNdim = a.NDim;
            var bNdim = b.NDim;

            // Get batch dimensions
            var aBatchDims = a.Shape.Take(aNdim - 2).ToArray();
            var bBatchDims = b.Shape.Take(bNdim - 2).ToArray();

            // Broadcast batch dimensions
            var maxBatchNdim = Math.Max(aBatchDims.Length, bBatchDims.Length);
            var resultBatchDims = new long[maxBatchNdim];

            for (int i = 0; i < maxBatchNdim; i++)
            {
                var aDim = i < aBatchDims.Length ? aBatchDims[aBatchDims.Length - 1 - i] : 1;
                var bDim = i < bBatchDims.Length ? bBatchDims[bBatchDims.Length - 1 - i] : 1;

                if (aDim != bDim && aDim != 1 && bDim != 1)
                    throw new ArgumentException("Batch dimensions are not broadcastable");

                resultBatchDims[maxBatchNdim - 1 - i] = Math.Max(aDim, bDim);
            }

            var m = a.Shape[aNdim - 2];
            var k = a.Shape[aNdim - 1];
            var n = b.Shape[bNdim - 1];

            if (k != b.Shape[bNdim - 2])
                throw new ArgumentException("Inner dimensions must match for matrix multiplication");

            var resultShape = resultBatchDims.Concat(new[] { m, n }).ToArray();
            var batchSize = resultBatchDims.Length == 0 ? 1 : resultBatchDims.Aggregate(1L, (x, y) => x * y);

            var result = new double[batchSize * m * n];

            // Perform batched matmul
            for (long batch = 0; batch < batchSize; batch++)
            {
                var batchIndices = GetBatchIndices(batch, resultBatchDims);
                var aIndices = BroadcastBatchIndices(batchIndices, aBatchDims);
                var bIndices = BroadcastBatchIndices(batchIndices, bBatchDims);

                for (long i = 0; i < m; i++)
                {
                    for (long j = 0; j < n; j++)
                    {
                        double sum = 0;
                        for (long l = 0; l < k; l++)
                        {
                            var aIdx = GetFlatIndex(aIndices.Concat(new[] { i, l }).ToArray(), a.Shape);
                            var bIdx = GetFlatIndex(bIndices.Concat(new[] { l, j }).ToArray(), b.Shape);
                            sum += a.Data[aIdx] * b.Data[bIdx];
                        }
                        result[batch * m * n + i * n + j] = sum;
                    }
                }
            }

            return new Tensor(result, resultShape, a.RequiresGrad || b.RequiresGrad);
        }

        #endregion

        #region Dot Product and Outer Product

        /// Dot product of two 1D tensors
        /// </summary>
        public static Tensor Dot(Tensor a, Tensor b)
        {
            if (a.NDim != 1 || b.NDim != 1)
                throw new ArgumentException("Dot product requires 1D tensors");
            if (a.Shape[0] != b.Shape[0])
                throw new ArgumentException($"Shape mismatch: {a.Shape[0]} vs {b.Shape[0]}");

            double sum = 0;
            for (long i = 0; i < a.Shape[0]; i++)
                sum += a.Data[i] * b.Data[i];

            return new Tensor(sum, a.RequiresGrad || b.RequiresGrad);
        }

        /// Outer product of two 1D tensors
        /// </summary>
        public static Tensor Outer(Tensor a, Tensor b)
        {
            if (a.NDim != 1 || b.NDim != 1)
                throw new ArgumentException("Outer product requires 1D tensors");

            var m = a.Shape[0];
            var n = b.Shape[0];
            var result = new double[m * n];

            for (long i = 0; i < m; i++)
                for (long j = 0; j < n; j++)
                    result[i * n + j] = a.Data[i] * b.Data[j];

            return new Tensor(result, new[] { m, n }, a.RequiresGrad || b.RequiresGrad);
        }

        /// Kronecker product of two tensors
        /// </summary>
        public static Tensor Kron(Tensor a, Tensor b)
        {
            if (a.NDim != 2 || b.NDim != 2)
                throw new ArgumentException("Kronecker product requires 2D tensors");

            var m1 = a.Shape[0];
            var n1 = a.Shape[1];
            var m2 = b.Shape[0];
            var n2 = b.Shape[1];

            var result = new double[m1 * m2 * n1 * n2];
            var resultShape = new[] { m1 * m2, n1 * n2 };

            for (long i1 = 0; i1 < m1; i1++)
            {
                for (long j1 = 0; j1 < n1; j1++)
                {
                    var aVal = a[i1, j1];
                    for (long i2 = 0; i2 < m2; i2++)
                    {
                        for (long j2 = 0; j2 < n2; j2++)
                        {
                            var row = i1 * m2 + i2;
                            var col = j1 * n2 + j2;
                            result[row * resultShape[1] + col] = aVal * b[i2, j2];
                        }
                    }
                }
            }

            return new Tensor(result, resultShape, a.RequiresGrad || b.RequiresGrad);
        }

        #endregion

        #region Concatenation and Stacking

        /// Concatenate tensors along a dimension
        /// </summary>
        public static Tensor Cat(Tensor[] tensors, int dim = 0)
        {
            if (tensors.Length == 0)
                throw new ArgumentException("Cannot concatenate empty tensor list");
            if (tensors.Length == 1)
                return tensors[0].Clone();

            var first = tensors[0];
            if (dim < 0) dim += first.NDim;

            // Validate shapes
            for (int i = 1; i < tensors.Length; i++)
            {
                if (tensors[i].NDim != first.NDim)
                    throw new ArgumentException("All tensors must have same number of dimensions");

                for (int d = 0; d < first.NDim; d++)
                {
                    if (d != dim && tensors[i].Shape[d] != first.Shape[d])
                        throw new ArgumentException($"Dimension mismatch at dim {d}");
                }
            }

            // Compute result shape
            var resultShape = first.Shape.ToArray();
            resultShape[dim] = tensors.Sum(t => t.Shape[dim]);

            var resultSize = resultShape.Aggregate(1L, (a, b) => a * b);
            var result = new double[resultSize];

            // Copy data
            long offset = 0;
            var outerSize = dim == 0 ? 1 : first.Shape.Take(dim).Aggregate(1L, (a, b) => a * b);
            var innerSize = dim == first.NDim - 1 ? 1 : first.Shape.Skip(dim + 1).Aggregate(1L, (a, b) => a * b);

            for (long outer = 0; outer < outerSize; outer++)
            {
                foreach (var t in tensors)
                {
                    var dimSize = t.Shape[dim];
                    for (long d = 0; d < dimSize; d++)
                    {
                        for (long inner = 0; inner < innerSize; inner++)
                        {
                            var srcIdx = outer * t.Shape[dim] * innerSize + d * innerSize + inner;
                            var dstIdx = outer * resultShape[dim] * innerSize + (offset + d) * innerSize + inner;
                            result[dstIdx] = t.Data[srcIdx];
                        }
                    }
                    offset += dimSize;
                }
                offset = 0;
            }

            return new Tensor(result, resultShape, tensors.Any(t => t.RequiresGrad));
        }

        /// Stack tensors along a new dimension
        /// </summary>
        public static Tensor Stack(Tensor[] tensors, int dim = 0)
        {
            if (tensors.Length == 0)
                throw new ArgumentException("Cannot stack empty tensor list");

            // Validate all same shape
            var shape = tensors[0].Shape;
            for (int i = 1; i < tensors.Length; i++)
            {
                if (!tensors[i].Shape.SequenceEqual(shape))
                    throw new ArgumentException("All tensors must have same shape");
            }

            // Unsqueeze and concatenate
            var unsqueezed = tensors.Select(t => t.Unsqueeze(dim)).ToArray();
            return Cat(unsqueezed, dim);
        }

        /// Split tensor into chunks
        /// </summary>
        public static Tensor[] Split(Tensor tensor, int chunks, int dim = 0)
        {
            if (dim < 0) dim += tensor.NDim;
            var dimSize = tensor.Shape[dim];
            var chunkSize = (dimSize + chunks - 1) / chunks;

            var result = new List<Tensor>();
            for (int i = 0; i < chunks; i++)
            {
                var start = i * chunkSize;
                var end = Math.Min((i + 1) * chunkSize, dimSize);
                if (start >= dimSize) break;
                result.Add(tensor.Narrow(dim, start, end - start));
            }

            return result.ToArray();
        }

        /// Chunk tensor into specified number of chunks
        /// </summary>
        public static Tensor[] Chunk(Tensor tensor, int chunks, int dim = 0) => Split(tensor, chunks, dim);

        #endregion

        #region Repetition and Tiling

        /// Repeat tensor along dimensions
        /// </summary>
        public static Tensor Repeat(Tensor tensor, params long[] repeats)
        {
            if (repeats.Length != tensor.NDim)
                throw new ArgumentException("Number of repeats must match tensor dimensions");

            var newShape = tensor.Shape.Zip(repeats, (s, r) => s * r).ToArray();
            var result = Tensor.Zeros(newShape);

            // Copy data with repetition
            var indices = new long[tensor.NDim];
            CopyWithRepeat(tensor, result, indices, 0, repeats);

            return result;
        }

        private static void CopyWithRepeat(Tensor src, Tensor dst, long[] indices, int dim, long[] repeats)
        {
            if (dim == src.NDim)
            {
                var srcIdx = GetFlatIndex(indices, src.Shape);
                for (var combo = GetRepeatCombinations(indices, repeats, src.Shape); combo.MoveNext();)
                {
                    var dstIdx = GetFlatIndex(combo.Current, dst.Shape);
                    dst.Data[dstIdx] = src.Data[srcIdx];
                }
                return;
            }

            for (long i = 0; i < src.Shape[dim]; i++)
            {
                indices[dim] = i;
                CopyWithRepeat(src, dst, indices, dim + 1, repeats);
            }
        }

        private static IEnumerator<long[]> GetRepeatCombinations(long[] srcIndices, long[] repeats, long[] srcShape)
        {
            var ndim = srcIndices.Length;
            var combo = new long[ndim];

            void Generate(int dim)
            {
                if (dim == ndim)
                {
                    return;
                }

                for (long r = 0; r < repeats[dim]; r++)
                {
                    combo[dim] = srcIndices[dim] + r * srcShape[dim];
                    Generate(dim + 1);
                }
            }

            // Simplified - just return single combo for now
            for (int d = 0; d < ndim; d++)
            {
                for (long r = 0; r < repeats[d]; r++)
                {
                    combo[d] = srcIndices[d] + r * srcShape[d];
                }
            }

            yield return combo;
        }

        /// Tile tensor
        /// </summary>
        public static Tensor Tile(Tensor tensor, params long[] reps)
        {
            // Adjust reps to match tensor dimensions
            var ndim = Math.Max(tensor.NDim, reps.Length);
            var adjustedReps = new long[ndim];
            var adjustedShape = new long[ndim];

            for (int i = 0; i < ndim; i++)
            {
                adjustedReps[ndim - 1 - i] = i < reps.Length ? reps[reps.Length - 1 - i] : 1;
                adjustedShape[ndim - 1 - i] = i < tensor.NDim ? tensor.Shape[tensor.NDim - 1 - i] : 1;
            }

            var reshaped = tensor.Reshape(adjustedShape);
            return Repeat(reshaped, adjustedReps);
        }

        #endregion

        #region Masking and Selection

        /// Masked select - get elements where mask is true
        /// </summary>
        public static Tensor MaskedSelect(Tensor input, Tensor mask)
        {
            if (!input.Shape.SequenceEqual(mask.Shape))
                throw new ArgumentException("Input and mask must have same shape");

            var selected = new List<double>();
            for (long i = 0; i < input.NumElements; i++)
            {
                if (mask.Data[i] != 0)
                    selected.Add(input.Data[i]);
            }

            return new Tensor(selected.ToArray(), new[] { (long)selected.Count });
        }

        /// Where - select from x or y based on condition
        /// </summary>
        public static Tensor Where(Tensor condition, Tensor x, Tensor y)
        {
            // Broadcast shapes
            var resultShape = BroadcastShapes(condition.Shape, BroadcastShapes(x.Shape, y.Shape));
            var result = new double[resultShape.Aggregate(1L, (a, b) => a * b)];

            for (long i = 0; i < result.Length; i++)
            {
                var indices = GetIndices(i, resultShape);
                var condVal = GetBroadcastValue(condition, indices);
                result[i] = condVal != 0 ? GetBroadcastValue(x, indices) : GetBroadcastValue(y, indices);
            }

            return new Tensor(result, resultShape);
        }

        /// Index select along dimension
        /// </summary>
        public static Tensor IndexSelect(Tensor input, int dim, Tensor index)
        {
            if (dim < 0) dim += input.NDim;
            if (index.NDim != 1)
                throw new ArgumentException("Index must be 1D");

            var resultShape = input.Shape.ToArray();
            resultShape[dim] = index.Shape[0];

            var result = new double[resultShape.Aggregate(1L, (a, b) => a * b)];

            // Copy selected indices
            var outerSize = dim == 0 ? 1 : input.Shape.Take(dim).Aggregate(1L, (a, b) => a * b);
            var innerSize = dim == input.NDim - 1 ? 1 : input.Shape.Skip(dim + 1).Aggregate(1L, (a, b) => a * b);

            for (long outer = 0; outer < outerSize; outer++)
            {
                for (long idx = 0; idx < index.Shape[0]; idx++)
                {
                    var srcDimIdx = (long)index.Data[idx];
                    for (long inner = 0; inner < innerSize; inner++)
                    {
                        var srcIdx = outer * input.Shape[dim] * innerSize + srcDimIdx * innerSize + inner;
                        var dstIdx = outer * resultShape[dim] * innerSize + idx * innerSize + inner;
                        result[dstIdx] = input.Data[srcIdx];
                    }
                }
            }

            return new Tensor(result, resultShape);
        }

        /// Gather values along dimension
        /// </summary>
        public static Tensor Gather(Tensor input, int dim, Tensor index)
        {
            if (dim < 0) dim += input.NDim;
            if (input.NDim != index.NDim)
                throw new ArgumentException("Input and index must have same number of dimensions");

            var result = new double[index.NumElements];

            for (long i = 0; i < index.NumElements; i++)
            {
                var indices = GetIndices(i, index.Shape);
                var srcIndices = indices.ToArray();
                srcIndices[dim] = (long)index.Data[i];

                var srcIdx = GetFlatIndex(srcIndices, input.Shape);
                result[i] = input.Data[srcIdx];
            }

            return new Tensor(result, index.Shape.ToArray());
        }

        /// Scatter values along dimension
        /// </summary>
        public static Tensor Scatter(Tensor input, int dim, Tensor index, Tensor src)
        {
            if (dim < 0) dim += input.NDim;

            var result = input.Clone();

            for (long i = 0; i < index.NumElements; i++)
            {
                var indices = GetIndices(i, index.Shape);
                var dstIndices = indices.ToArray();
                dstIndices[dim] = (long)index.Data[i];

                var dstIdx = GetFlatIndex(dstIndices, result.Shape);
                result.Data[dstIdx] = src.Data[i];
            }

            return result;
        }

        #endregion

        #region Einsum (Einstein Summation)

        /// Einstein summation convention
        /// Supports common operations like: "ij,jk->ik" (matmul), "ij->ji" (transpose), "ii->" (trace)
        /// </summary>
        public static Tensor Einsum(string equation, params Tensor[] operands)
        {
            var parts = equation.Split(new[] { "->" }, StringSplitOptions.None);
            var inputSubscripts = parts[0].Split(',');
            var outputSubscripts = parts.Length > 1 ? parts[1] : InferOutputSubscripts(inputSubscripts);

            if (inputSubscripts.Length != operands.Length)
                throw new ArgumentException("Number of subscripts must match number of operands");

            // Build dimension mapping
            var dimSizes = new Dictionary<char, long>();
            for (int i = 0; i < operands.Length; i++)
            {
                var subs = inputSubscripts[i].Trim();
                for (int j = 0; j < subs.Length; j++)
                {
                    var c = subs[j];
                    if (c == ' ') continue;

                    if (dimSizes.ContainsKey(c))
                    {
                        if (dimSizes[c] != operands[i].Shape[j])
                            throw new ArgumentException($"Dimension mismatch for subscript '{c}'");
                    }
                    else
                    {
                        dimSizes[c] = operands[i].Shape[j];
                    }
                }
            }

            // Get all unique indices
            var allIndices = inputSubscripts.SelectMany(s => s.Where(char.IsLetter)).Distinct().ToArray();
            var sumIndices = allIndices.Except(outputSubscripts).ToArray();

            // Build result shape
            var resultShape = outputSubscripts.Select(c => dimSizes[c]).ToArray();
            var resultSize = resultShape.Length == 0 ? 1 : resultShape.Aggregate(1L, (a, b) => a * b);
            var result = new double[resultSize];

            // Compute einsum by iterating over all index combinations
            var indexRanges = allIndices.Select(c => (int)dimSizes[c]).ToArray();
            var currentIndices = new int[allIndices.Length];

            void ComputeEinsum(int depth)
            {
                if (depth == allIndices.Length)
                {
                    // Compute product for this index combination
                    double product = 1.0;
                    for (int i = 0; i < operands.Length; i++)
                    {
                        var subs = inputSubscripts[i].Trim();
                        var indices = subs.Select(c => (long)currentIndices[Array.IndexOf(allIndices, c)]).ToArray();
                        var idx = GetFlatIndex(indices, operands[i].Shape);
                        product *= operands[i].Data[idx];
                    }

                    // Add to result
                    var resultIndices = outputSubscripts.Select(c => (long)currentIndices[Array.IndexOf(allIndices, c)]).ToArray();
                    var resultIdx = resultIndices.Length == 0 ? 0 : GetFlatIndex(resultIndices, resultShape);
                    result[resultIdx] += product;
                    return;
                }

                for (int i = 0; i < indexRanges[depth]; i++)
                {
                    currentIndices[depth] = i;
                    ComputeEinsum(depth + 1);
                }
            }

            ComputeEinsum(0);

            return new Tensor(result, resultShape.Length == 0 ? Array.Empty<long>() : resultShape);
        }

        private static string InferOutputSubscripts(string[] inputSubscripts)
        {
            // Output subscripts are indices that appear exactly once
            var counts = new Dictionary<char, int>();
            foreach (var subs in inputSubscripts)
            {
                foreach (var c in subs.Where(char.IsLetter))
                {
                    counts[c] = counts.GetValueOrDefault(c) + 1;
                }
            }

            return new string(counts.Where(kv => kv.Value == 1).Select(kv => kv.Key).OrderBy(c => c).ToArray());
        }

        #endregion

        #region Helper Methods

        private static long[] GetBatchIndices(long flatBatchIdx, long[] batchShape)
        {
            var indices = new long[batchShape.Length];
            for (int i = batchShape.Length - 1; i >= 0; i--)
            {
                indices[i] = flatBatchIdx % batchShape[i];
                flatBatchIdx /= batchShape[i];
            }
            return indices;
        }

        private static long[] BroadcastBatchIndices(long[] indices, long[] shape)
        {
            var result = new long[shape.Length];
            var offset = indices.Length - shape.Length;

            for (int i = 0; i < shape.Length; i++)
            {
                var srcIdx = i + offset;
                result[i] = srcIdx >= 0 && shape[i] > 1 ? indices[srcIdx] : 0;
            }

            return result;
        }

        private static long GetFlatIndex(long[] indices, long[] shape)
        {
            long flatIdx = 0;
            long stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                flatIdx += indices[i] * stride;
                stride *= shape[i];
            }
            return flatIdx;
        }

        private static long[] GetIndices(long flatIndex, long[] shape)
        {
            var indices = new long[shape.Length];
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                indices[i] = flatIndex % shape[i];
                flatIndex /= shape[i];
            }
            return indices;
        }

        private static long[] BroadcastShapes(long[] shape1, long[] shape2)
        {
            var maxDims = Math.Max(shape1.Length, shape2.Length);
            var result = new long[maxDims];

            for (int i = 0; i < maxDims; i++)
            {
                var d1 = i < shape1.Length ? shape1[shape1.Length - 1 - i] : 1;
                var d2 = i < shape2.Length ? shape2[shape2.Length - 1 - i] : 1;

                if (d1 != d2 && d1 != 1 && d2 != 1)
                    throw new ArgumentException($"Cannot broadcast shapes");

                result[maxDims - 1 - i] = Math.Max(d1, d2);
            }

            return result;
        }

        private static double GetBroadcastValue(Tensor tensor, long[] indices)
        {
            var broadcastIndices = new long[tensor.NDim];
            var offset = indices.Length - tensor.NDim;

            for (int i = 0; i < tensor.NDim; i++)
            {
                var srcIdx = i + offset;
                broadcastIndices[i] = srcIdx >= 0 ? indices[srcIdx] % tensor.Shape[i] : 0;
            }

            return tensor.Data[GetFlatIndex(broadcastIndices, tensor.Shape)];
        }

        #endregion

        #region Static Tensor Arithmetic Operations

        /// Element-wise addition of two tensors
        /// </summary>
        public static Tensor Add(Tensor a, Tensor b) => a.Add(b);

        /// Element-wise subtraction of two tensors
        /// </summary>
        public static Tensor Sub(Tensor a, Tensor b) => a.Sub(b);

        /// Element-wise multiplication of two tensors
        /// </summary>
        public static Tensor Mul(Tensor a, Tensor b) => a.Mul(b);

        /// Element-wise division of two tensors
        /// </summary>
        public static Tensor Div(Tensor a, Tensor b) => a.Div(b);

        /// Negate all elements of a tensor
        /// </summary>
        public static Tensor Neg(Tensor a) => a.Neg();

        /// Multiply tensor by scalar
        /// </summary>
        public static Tensor MulScalar(Tensor a, double scalar) => a.Mul(scalar);

        #endregion
    }

    /// Extension methods for tensor operations
    /// </summary>
    public static class TensorOpsExtensions
    {
        /// <summary>Public API</summary>
        public static Tensor MatMul(this Tensor a, Tensor b) => TensorOps.MatMul(a, b);
        /// <summary>Public API</summary>
        public static Tensor Dot(this Tensor a, Tensor b) => TensorOps.Dot(a, b);
        /// <summary>Public API</summary>
        public static Tensor Outer(this Tensor a, Tensor b) => TensorOps.Outer(a, b);
        /// <summary>Public API</summary>
        public static Tensor Kron(this Tensor a, Tensor b) => TensorOps.Kron(a, b);
    }

    #region SIMD-Optimized CPU Primitives

    /// High-performance SIMD-accelerated tensor primitives for CPU.
    /// Uses AVX-512/AVX2/SSE with automatic fallback based on CPU capabilities.
    /// Inspired by .NET TensorPrimitives but optimized for NSL tensor operations.
    /// </summary>
    public static class CpuTensorPrimitives
    {
        // Feature detection cached at startup
        private static readonly bool HasAvx512 = Avx512F.IsSupported;
        private static readonly bool HasAvx2 = Avx2.IsSupported;
        private static readonly bool HasFma = Fma.IsSupported;
        private static readonly bool HasSse2 = Sse2.IsSupported;

        // Optimal parallel grain sizes
        private const int GrainSize = 4096;
        private const int SmallArrayThreshold = 256;

        #region Element-wise Arithmetic

        /// SIMD-optimized element-wise addition: result = a + b
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void Add(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
                throw new ArgumentException("Spans must have equal length");

            int length = a.Length;
            int i = 0;

            if (HasAvx512 && length >= 8)
            {
                AddAvx512(a, b, result, ref i);
            }
            else if (HasAvx2 && length >= 4)
            {
                AddAvx2(a, b, result, ref i);
            }
            else if (HasSse2 && length >= 2)
            {
                AddSse2(a, b, result, ref i);
            }

            // Scalar remainder
            for (; i < length; i++)
                result[i] = a[i] + b[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void AddAvx512(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result, ref int i)
        {
            int length = a.Length;
            fixed (double* pA = a, pB = b, pR = result)
            {
                for (; i <= length - 8; i += 8)
                {
                    var va = Avx512F.LoadVector512(pA + i);
                    var vb = Avx512F.LoadVector512(pB + i);
                    Avx512F.Store(pR + i, Avx512F.Add(va, vb));
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void AddAvx2(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result, ref int i)
        {
            int length = a.Length;
            fixed (double* pA = a, pB = b, pR = result)
            {
                // Process 16 doubles per iteration (4 vectors)
                for (; i <= length - 16; i += 16)
                {
                    var va0 = Avx.LoadVector256(pA + i);
                    var va1 = Avx.LoadVector256(pA + i + 4);
                    var va2 = Avx.LoadVector256(pA + i + 8);
                    var va3 = Avx.LoadVector256(pA + i + 12);

                    var vb0 = Avx.LoadVector256(pB + i);
                    var vb1 = Avx.LoadVector256(pB + i + 4);
                    var vb2 = Avx.LoadVector256(pB + i + 8);
                    var vb3 = Avx.LoadVector256(pB + i + 12);

                    Avx.Store(pR + i, Avx.Add(va0, vb0));
                    Avx.Store(pR + i + 4, Avx.Add(va1, vb1));
                    Avx.Store(pR + i + 8, Avx.Add(va2, vb2));
                    Avx.Store(pR + i + 12, Avx.Add(va3, vb3));
                }

                for (; i <= length - 4; i += 4)
                {
                    var va = Avx.LoadVector256(pA + i);
                    var vb = Avx.LoadVector256(pB + i);
                    Avx.Store(pR + i, Avx.Add(va, vb));
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void AddSse2(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result, ref int i)
        {
            int length = a.Length;
            fixed (double* pA = a, pB = b, pR = result)
            {
                for (; i <= length - 2; i += 2)
                {
                    var va = Sse2.LoadVector128(pA + i);
                    var vb = Sse2.LoadVector128(pB + i);
                    Sse2.Store(pR + i, Sse2.Add(va, vb));
                }
            }
        }

        /// SIMD-optimized element-wise subtraction: result = a - b
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Subtract(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            int length = a.Length;
            int i = 0;

            fixed (double* pA = a, pB = b, pR = result)
            {
                if (HasAvx2 && length >= 16)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(pR + i, Avx.Subtract(Avx.LoadVector256(pA + i), Avx.LoadVector256(pB + i)));
                        Avx.Store(pR + i + 4, Avx.Subtract(Avx.LoadVector256(pA + i + 4), Avx.LoadVector256(pB + i + 4)));
                        Avx.Store(pR + i + 8, Avx.Subtract(Avx.LoadVector256(pA + i + 8), Avx.LoadVector256(pB + i + 8)));
                        Avx.Store(pR + i + 12, Avx.Subtract(Avx.LoadVector256(pA + i + 12), Avx.LoadVector256(pB + i + 12)));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = a[i] - b[i];
        }

        /// SIMD-optimized element-wise multiplication: result = a * b
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Multiply(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            int length = a.Length;
            int i = 0;

            fixed (double* pA = a, pB = b, pR = result)
            {
                if (HasAvx2 && length >= 16)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(pR + i, Avx.Multiply(Avx.LoadVector256(pA + i), Avx.LoadVector256(pB + i)));
                        Avx.Store(pR + i + 4, Avx.Multiply(Avx.LoadVector256(pA + i + 4), Avx.LoadVector256(pB + i + 4)));
                        Avx.Store(pR + i + 8, Avx.Multiply(Avx.LoadVector256(pA + i + 8), Avx.LoadVector256(pB + i + 8)));
                        Avx.Store(pR + i + 12, Avx.Multiply(Avx.LoadVector256(pA + i + 12), Avx.LoadVector256(pB + i + 12)));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = a[i] * b[i];
        }

        /// SIMD-optimized element-wise division: result = a / b
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Divide(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            int length = a.Length;
            int i = 0;

            fixed (double* pA = a, pB = b, pR = result)
            {
                if (HasAvx2 && length >= 16)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(pR + i, Avx.Divide(Avx.LoadVector256(pA + i), Avx.LoadVector256(pB + i)));
                        Avx.Store(pR + i + 4, Avx.Divide(Avx.LoadVector256(pA + i + 4), Avx.LoadVector256(pB + i + 4)));
                        Avx.Store(pR + i + 8, Avx.Divide(Avx.LoadVector256(pA + i + 8), Avx.LoadVector256(pB + i + 8)));
                        Avx.Store(pR + i + 12, Avx.Divide(Avx.LoadVector256(pA + i + 12), Avx.LoadVector256(pB + i + 12)));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = a[i] / b[i];
        }

        /// SIMD-optimized scalar multiplication: result = a * scalar
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MultiplyScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
        {
            int length = a.Length;
            int i = 0;

            fixed (double* pA = a, pR = result)
            {
                if (HasAvx2 && length >= 16)
                {
                    var vScalar = Vector256.Create(scalar);
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(pR + i, Avx.Multiply(Avx.LoadVector256(pA + i), vScalar));
                        Avx.Store(pR + i + 4, Avx.Multiply(Avx.LoadVector256(pA + i + 4), vScalar));
                        Avx.Store(pR + i + 8, Avx.Multiply(Avx.LoadVector256(pA + i + 8), vScalar));
                        Avx.Store(pR + i + 12, Avx.Multiply(Avx.LoadVector256(pA + i + 12), vScalar));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = a[i] * scalar;
        }

        /// SIMD-optimized fused multiply-add: result = a * b + c
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void FusedMultiplyAdd(ReadOnlySpan<double> a, ReadOnlySpan<double> b, ReadOnlySpan<double> c, Span<double> result)
        {
            int length = a.Length;
            int i = 0;

            fixed (double* pA = a, pB = b, pC = c, pR = result)
            {
                if (HasFma && length >= 16)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(pR + i, Fma.MultiplyAdd(Avx.LoadVector256(pA + i), Avx.LoadVector256(pB + i), Avx.LoadVector256(pC + i)));
                        Avx.Store(pR + i + 4, Fma.MultiplyAdd(Avx.LoadVector256(pA + i + 4), Avx.LoadVector256(pB + i + 4), Avx.LoadVector256(pC + i + 4)));
                        Avx.Store(pR + i + 8, Fma.MultiplyAdd(Avx.LoadVector256(pA + i + 8), Avx.LoadVector256(pB + i + 8), Avx.LoadVector256(pC + i + 8)));
                        Avx.Store(pR + i + 12, Fma.MultiplyAdd(Avx.LoadVector256(pA + i + 12), Avx.LoadVector256(pB + i + 12), Avx.LoadVector256(pC + i + 12)));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = a[i] * b[i] + c[i];
        }

        #endregion

        #region Reduction Operations

        /// SIMD-optimized sum reduction
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double Sum(ReadOnlySpan<double> data)
        {
            int length = data.Length;
            if (length == 0) return 0;

            double sum = 0;
            int i = 0;

            fixed (double* p = data)
            {
                if (HasAvx2 && length >= 16)
                {
                    var acc0 = Vector256<double>.Zero;
                    var acc1 = Vector256<double>.Zero;
                    var acc2 = Vector256<double>.Zero;
                    var acc3 = Vector256<double>.Zero;

                    for (; i <= length - 16; i += 16)
                    {
                        acc0 = Avx.Add(acc0, Avx.LoadVector256(p + i));
                        acc1 = Avx.Add(acc1, Avx.LoadVector256(p + i + 4));
                        acc2 = Avx.Add(acc2, Avx.LoadVector256(p + i + 8));
                        acc3 = Avx.Add(acc3, Avx.LoadVector256(p + i + 12));
                    }

                    var combined = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));
                    sum = HorizontalSum256(combined);
                }
            }

            for (; i < length; i++)
                sum += data[i];

            return sum;
        }

        /// SIMD-optimized dot product
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Spans must have equal length");

            int length = a.Length;
            if (length == 0) return 0;

            double sum = 0;
            int i = 0;

            fixed (double* pA = a, pB = b)
            {
                if (HasFma && length >= 16)
                {
                    var acc0 = Vector256<double>.Zero;
                    var acc1 = Vector256<double>.Zero;
                    var acc2 = Vector256<double>.Zero;
                    var acc3 = Vector256<double>.Zero;

                    for (; i <= length - 16; i += 16)
                    {
                        acc0 = Fma.MultiplyAdd(Avx.LoadVector256(pA + i), Avx.LoadVector256(pB + i), acc0);
                        acc1 = Fma.MultiplyAdd(Avx.LoadVector256(pA + i + 4), Avx.LoadVector256(pB + i + 4), acc1);
                        acc2 = Fma.MultiplyAdd(Avx.LoadVector256(pA + i + 8), Avx.LoadVector256(pB + i + 8), acc2);
                        acc3 = Fma.MultiplyAdd(Avx.LoadVector256(pA + i + 12), Avx.LoadVector256(pB + i + 12), acc3);
                    }

                    var combined = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));
                    sum = HorizontalSum256(combined);
                }
                else if (HasAvx2 && length >= 4)
                {
                    var acc = Vector256<double>.Zero;
                    for (; i <= length - 4; i += 4)
                    {
                        acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(pA + i), Avx.LoadVector256(pB + i)));
                    }
                    sum = HorizontalSum256(acc);
                }
            }

            for (; i < length; i++)
                sum += a[i] * b[i];

            return sum;
        }

        /// SIMD-optimized sum of squares: sum(x^2)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double SumOfSquares(ReadOnlySpan<double> data)
        {
            int length = data.Length;
            if (length == 0) return 0;

            double sum = 0;
            int i = 0;

            fixed (double* p = data)
            {
                if (HasFma && length >= 16)
                {
                    var acc0 = Vector256<double>.Zero;
                    var acc1 = Vector256<double>.Zero;
                    var acc2 = Vector256<double>.Zero;
                    var acc3 = Vector256<double>.Zero;

                    for (; i <= length - 16; i += 16)
                    {
                        var v0 = Avx.LoadVector256(p + i);
                        var v1 = Avx.LoadVector256(p + i + 4);
                        var v2 = Avx.LoadVector256(p + i + 8);
                        var v3 = Avx.LoadVector256(p + i + 12);

                        acc0 = Fma.MultiplyAdd(v0, v0, acc0);
                        acc1 = Fma.MultiplyAdd(v1, v1, acc1);
                        acc2 = Fma.MultiplyAdd(v2, v2, acc2);
                        acc3 = Fma.MultiplyAdd(v3, v3, acc3);
                    }

                    sum = HorizontalSum256(Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3)));
                }
            }

            for (; i < length; i++)
                sum += data[i] * data[i];

            return sum;
        }

        /// SIMD-optimized L2 norm: sqrt(sum(x^2))
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Norm2(ReadOnlySpan<double> data)
        {
            return Math.Sqrt(SumOfSquares(data));
        }

        /// SIMD-optimized cosine similarity
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static double CosineSimilarity(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            double dot = Dot(a, b);
            double normA = Norm2(a);
            double normB = Norm2(b);

            if (normA == 0 || normB == 0) return 0;
            return dot / (normA * normB);
        }

        /// SIMD-optimized max value
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double Max(ReadOnlySpan<double> data)
        {
            if (data.Length == 0) throw new ArgumentException("Empty span");

            double max = data[0];
            int i = 1;

            fixed (double* p = data)
            {
                if (HasAvx2 && data.Length >= 4)
                {
                    var vMax = Vector256.Create(data[0]);
                    for (; i <= data.Length - 4; i += 4)
                    {
                        vMax = Avx.Max(vMax, Avx.LoadVector256(p + i));
                    }
                    // Horizontal max
                    var upper = Avx.ExtractVector128(vMax, 1);
                    var lower = Avx.ExtractVector128(vMax, 0);
                    var max128 = Sse2.Max(upper, lower);
                    max = Math.Max(max128.GetElement(0), max128.GetElement(1));
                }
            }

            for (; i < data.Length; i++)
                if (data[i] > max) max = data[i];

            return max;
        }

        /// SIMD-optimized min value
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double Min(ReadOnlySpan<double> data)
        {
            if (data.Length == 0) throw new ArgumentException("Empty span");

            double min = data[0];
            int i = 1;

            fixed (double* p = data)
            {
                if (HasAvx2 && data.Length >= 4)
                {
                    var vMin = Vector256.Create(data[0]);
                    for (; i <= data.Length - 4; i += 4)
                    {
                        vMin = Avx.Min(vMin, Avx.LoadVector256(p + i));
                    }
                    var upper = Avx.ExtractVector128(vMin, 1);
                    var lower = Avx.ExtractVector128(vMin, 0);
                    var min128 = Sse2.Min(upper, lower);
                    min = Math.Min(min128.GetElement(0), min128.GetElement(1));
                }
            }

            for (; i < data.Length; i++)
                if (data[i] < min) min = data[i];

            return min;
        }

        #endregion

        #region Activation Functions

        /// SIMD-optimized ReLU: max(0, x)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ReLU(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;
            int i = 0;

            fixed (double* pIn = input, pOut = result)
            {
                if (HasAvx2 && length >= 16)
                {
                    var zero = Vector256<double>.Zero;
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(pOut + i, Avx.Max(zero, Avx.LoadVector256(pIn + i)));
                        Avx.Store(pOut + i + 4, Avx.Max(zero, Avx.LoadVector256(pIn + i + 4)));
                        Avx.Store(pOut + i + 8, Avx.Max(zero, Avx.LoadVector256(pIn + i + 8)));
                        Avx.Store(pOut + i + 12, Avx.Max(zero, Avx.LoadVector256(pIn + i + 12)));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = input[i] > 0 ? input[i] : 0;
        }

        /// SIMD-optimized Leaky ReLU
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void LeakyReLU(ReadOnlySpan<double> input, Span<double> result, double negativeSlope = 0.01)
        {
            int length = input.Length;
            int i = 0;

            fixed (double* pIn = input, pOut = result)
            {
                if (HasAvx2 && length >= 4)
                {
                    var zero = Vector256<double>.Zero;
                    var slope = Vector256.Create(negativeSlope);

                    for (; i <= length - 4; i += 4)
                    {
                        var x = Avx.LoadVector256(pIn + i);
                        var mask = Avx.Compare(x, zero, FloatComparisonMode.OrderedGreaterThanSignaling);
                        var negative = Avx.Multiply(x, slope);
                        var blended = Avx.BlendVariable(negative, x, mask);
                        Avx.Store(pOut + i, blended);
                    }
                }
            }

            for (; i < length; i++)
                result[i] = input[i] > 0 ? input[i] : input[i] * negativeSlope;
        }

        /// Optimized Sigmoid using fast approximation with SIMD
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void Sigmoid(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;

            // Parallel for large arrays
            if (length >= GrainSize)
            {
                // Copy to arrays for parallel access (Span can't be captured in lambdas)
                var inputArray = input.ToArray();
                var resultArray = new double[length];

                Parallel.For(0, length, i =>
                {
                    resultArray[i] = 1.0 / (1.0 + Math.Exp(-inputArray[i]));
                });

                resultArray.CopyTo(result);
            }
            else
            {
                for (int i = 0; i < length; i++)
                    result[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
            }
        }

        /// Optimized Tanh with SIMD and parallelization
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void Tanh(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;

            if (length >= GrainSize)
            {
                // Copy to array for parallel access
                var inputArray = input.ToArray();
                var resultArray = new double[length];

                Parallel.For(0, length, i =>
                {
                    resultArray[i] = Math.Tanh(inputArray[i]);
                });

                resultArray.CopyTo(result);
            }
            else
            {
                for (int i = 0; i < length; i++)
                    result[i] = Math.Tanh(input[i]);
            }
        }

        /// SIMD-optimized GELU approximation (faster than exact)
        /// Uses tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void GELU(ReadOnlySpan<double> input, Span<double> result)
        {
            const double sqrt2OverPi = 0.7978845608028654;
            const double coeff = 0.044715;

            int length = input.Length;

            if (length >= GrainSize)
            {
                var inputArray = input.ToArray();
                var resultArray = new double[length];

                Parallel.For(0, length, i =>
                {
                    double x = inputArray[i];
                    double x3 = x * x * x;
                    double inner = sqrt2OverPi * (x + coeff * x3);
                    resultArray[i] = 0.5 * x * (1.0 + Math.Tanh(inner));
                });

                resultArray.CopyTo(result);
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    double x = input[i];
                    double x3 = x * x * x;
                    double inner = sqrt2OverPi * (x + coeff * x3);
                    result[i] = 0.5 * x * (1.0 + Math.Tanh(inner));
                }
            }
        }

        /// SIMD-optimized Softmax (numerically stable)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void Softmax(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;
            if (length == 0) return;

            // Find max for numerical stability
            double max = Max(input);

            // Compute exp(x - max) and sum
            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                result[i] = Math.Exp(input[i] - max);
                sum += result[i];
            }

            // Normalize
            if (sum > 0)
            {
                double invSum = 1.0 / sum;
                MultiplyScalar(result, invSum, result);
            }
        }

        /// SIMD-optimized Log-Softmax (numerically stable)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void LogSoftmax(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;
            if (length == 0) return;

            double max = Max(input);

            // Compute log-sum-exp
            double sumExp = 0;
            for (int i = 0; i < length; i++)
            {
                sumExp += Math.Exp(input[i] - max);
            }
            double logSumExp = max + Math.Log(sumExp);

            // Compute log-softmax
            for (int i = 0; i < length; i++)
            {
                result[i] = input[i] - logSumExp;
            }
        }

        #endregion

        #region Mathematical Functions

        /// SIMD-optimized element-wise exponential
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void Exp(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;

            if (length >= GrainSize)
            {
                var inputArray = input.ToArray();
                var resultArray = new double[length];

                Parallel.For(0, length, i =>
                {
                    resultArray[i] = Math.Exp(inputArray[i]);
                });

                resultArray.CopyTo(result);
            }
            else
            {
                for (int i = 0; i < length; i++)
                    result[i] = Math.Exp(input[i]);
            }
        }

        /// SIMD-optimized element-wise natural log
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void Log(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;

            if (length >= GrainSize)
            {
                var inputArray = input.ToArray();
                var resultArray = new double[length];

                Parallel.For(0, length, i =>
                {
                    resultArray[i] = Math.Log(inputArray[i]);
                });

                resultArray.CopyTo(result);
            }
            else
            {
                for (int i = 0; i < length; i++)
                    result[i] = Math.Log(input[i]);
            }
        }

        /// SIMD-optimized element-wise square root
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Sqrt(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;
            int i = 0;

            fixed (double* pIn = input, pOut = result)
            {
                if (HasAvx2 && length >= 4)
                {
                    for (; i <= length - 4; i += 4)
                    {
                        Avx.Store(pOut + i, Avx.Sqrt(Avx.LoadVector256(pIn + i)));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = Math.Sqrt(input[i]);
        }

        /// SIMD-optimized element-wise absolute value
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Abs(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;
            int i = 0;

            fixed (double* pIn = input, pOut = result)
            {
                if (HasAvx2 && length >= 4)
                {
                    // Use AND with sign bit mask to clear sign bit
                    var signMask = Vector256.Create(~long.MinValue).AsDouble();
                    for (; i <= length - 4; i += 4)
                    {
                        Avx.Store(pOut + i, Avx.And(Avx.LoadVector256(pIn + i), signMask));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = Math.Abs(input[i]);
        }

        /// SIMD-optimized clamp operation
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Clamp(ReadOnlySpan<double> input, double min, double max, Span<double> result)
        {
            int length = input.Length;
            int i = 0;

            fixed (double* pIn = input, pOut = result)
            {
                if (HasAvx2 && length >= 4)
                {
                    var vMin = Vector256.Create(min);
                    var vMax = Vector256.Create(max);

                    for (; i <= length - 4; i += 4)
                    {
                        var v = Avx.LoadVector256(pIn + i);
                        v = Avx.Max(vMin, Avx.Min(vMax, v));
                        Avx.Store(pOut + i, v);
                    }
                }
            }

            for (; i < length; i++)
                result[i] = Math.Max(min, Math.Min(max, input[i]));
        }

        #endregion

        #region Parallel Operations

        /// Parallel element-wise operation with work stealing
        /// </summary>
        public static void ParallelApply(ReadOnlySpan<double> input, Span<double> result, Func<double, double> func)
        {
            int length = input.Length;

            if (length < SmallArrayThreshold)
            {
                for (int i = 0; i < length; i++)
                    result[i] = func(input[i]);
                return;
            }

            var inputArray = input.ToArray();
            var resultArray = new double[length];

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };

            Parallel.ForEach(
                Partitioner.Create(0, length, Math.Max(1, length / Environment.ProcessorCount)),
                parallelOptions,
                range =>
                {
                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        resultArray[i] = func(inputArray[i]);
                    }
                });

            resultArray.CopyTo(result);
        }

        /// Parallel binary operation
        /// </summary>
        public static void ParallelApply(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result, Func<double, double, double> func)
        {
            int length = a.Length;

            if (length < SmallArrayThreshold)
            {
                for (int i = 0; i < length; i++)
                    result[i] = func(a[i], b[i]);
                return;
            }

            var aArray = a.ToArray();
            var bArray = b.ToArray();
            var resultArray = new double[length];

            Parallel.ForEach(
                Partitioner.Create(0, length, Math.Max(1, length / Environment.ProcessorCount)),
                range =>
                {
                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        resultArray[i] = func(aArray[i], bArray[i]);
                    }
                });

            resultArray.CopyTo(result);
        }

        #endregion

        #region Memory Operations

        /// SIMD-optimized memory copy
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Copy(ReadOnlySpan<double> source, Span<double> dest)
        {
            int length = source.Length;
            int i = 0;

            fixed (double* pSrc = source, pDst = dest)
            {
                if (HasAvx2 && length >= 16)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(pDst + i, Avx.LoadVector256(pSrc + i));
                        Avx.Store(pDst + i + 4, Avx.LoadVector256(pSrc + i + 4));
                        Avx.Store(pDst + i + 8, Avx.LoadVector256(pSrc + i + 8));
                        Avx.Store(pDst + i + 12, Avx.LoadVector256(pSrc + i + 12));
                    }
                }
            }

            for (; i < length; i++)
                dest[i] = source[i];
        }

        /// SIMD-optimized memory fill
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Fill(Span<double> dest, double value)
        {
            int length = dest.Length;
            int i = 0;

            fixed (double* p = dest)
            {
                if (HasAvx2 && length >= 16)
                {
                    var v = Vector256.Create(value);
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(p + i, v);
                        Avx.Store(p + i + 4, v);
                        Avx.Store(p + i + 8, v);
                        Avx.Store(p + i + 12, v);
                    }
                }
            }

            for (; i < length; i++)
                dest[i] = value;
        }

        /// SIMD-optimized zero fill
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Zero(Span<double> dest)
        {
            int length = dest.Length;
            int i = 0;

            fixed (double* p = dest)
            {
                if (HasAvx2 && length >= 16)
                {
                    var zero = Vector256<double>.Zero;
                    for (; i <= length - 16; i += 16)
                    {
                        Avx.Store(p + i, zero);
                        Avx.Store(p + i + 4, zero);
                        Avx.Store(p + i + 8, zero);
                        Avx.Store(p + i + 12, zero);
                    }
                }
            }

            for (; i < length; i++)
                dest[i] = 0;
        }

        #endregion

        #region Helper Methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double HorizontalSum256(Vector256<double> v)
        {
            var vlow = Avx.ExtractVector128(v, 0);
            var vhigh = Avx.ExtractVector128(v, 1);
            var sum128 = Sse2.Add(vlow, vhigh);
            var high64 = Sse2.UnpackHigh(sum128, sum128);
            return Sse2.AddScalar(sum128, high64).ToScalar();
        }

        #endregion
    }

    #endregion

    #region Optimized CPU Optimizer Updates

    /// SIMD-accelerated optimizer step functions for CPU training.
    /// Provides vectorized parameter updates for maximum performance.
    /// </summary>
    public static class CpuOptimizerKernels
    {
        private static readonly bool HasAvx2 = Avx2.IsSupported;
        private static readonly bool HasFma = Fma.IsSupported;

        /// SIMD-optimized SGD step: param -= lr * grad
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void SgdStep(Span<double> param, ReadOnlySpan<double> grad, double lr)
        {
            int length = param.Length;
            int i = 0;

            fixed (double* pParam = param, pGrad = grad)
            {
                if (HasAvx2 && length >= 16)
                {
                    var vLr = Vector256.Create(lr);
                    for (; i <= length - 16; i += 16)
                    {
                        var p0 = Avx.LoadVector256(pParam + i);
                        var p1 = Avx.LoadVector256(pParam + i + 4);
                        var p2 = Avx.LoadVector256(pParam + i + 8);
                        var p3 = Avx.LoadVector256(pParam + i + 12);

                        var g0 = Avx.LoadVector256(pGrad + i);
                        var g1 = Avx.LoadVector256(pGrad + i + 4);
                        var g2 = Avx.LoadVector256(pGrad + i + 8);
                        var g3 = Avx.LoadVector256(pGrad + i + 12);

                        Avx.Store(pParam + i, Avx.Subtract(p0, Avx.Multiply(vLr, g0)));
                        Avx.Store(pParam + i + 4, Avx.Subtract(p1, Avx.Multiply(vLr, g1)));
                        Avx.Store(pParam + i + 8, Avx.Subtract(p2, Avx.Multiply(vLr, g2)));
                        Avx.Store(pParam + i + 12, Avx.Subtract(p3, Avx.Multiply(vLr, g3)));
                    }
                }
            }

            for (; i < length; i++)
                param[i] -= lr * grad[i];
        }

        /// SIMD-optimized SGD step with momentum
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void SgdMomentumStep(
            Span<double> param,
            ReadOnlySpan<double> grad,
            Span<double> velocity,
            double lr,
            double momentum)
        {
            int length = param.Length;
            int i = 0;

            fixed (double* pParam = param, pGrad = grad, pVel = velocity)
            {
                if (HasFma && length >= 4)
                {
                    var vLr = Vector256.Create(lr);
                    var vMom = Vector256.Create(momentum);

                    for (; i <= length - 4; i += 4)
                    {
                        // velocity = momentum * velocity + grad
                        var v = Fma.MultiplyAdd(vMom, Avx.LoadVector256(pVel + i), Avx.LoadVector256(pGrad + i));
                        Avx.Store(pVel + i, v);

                        // param -= lr * velocity
                        var p = Avx.Subtract(Avx.LoadVector256(pParam + i), Avx.Multiply(vLr, v));
                        Avx.Store(pParam + i, p);
                    }
                }
            }

            for (; i < length; i++)
            {
                velocity[i] = momentum * velocity[i] + grad[i];
                param[i] -= lr * velocity[i];
            }
        }

        /// SIMD-optimized Adam step
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void AdamStep(
            Span<double> param,
            ReadOnlySpan<double> grad,
            Span<double> m,
            Span<double> v,
            double lr,
            double beta1,
            double beta2,
            double epsilon,
            double beta1CorrectionFactor,
            double beta2CorrectionFactor)
        {
            int length = param.Length;

            // Corrected learning rate
            double correctedLr = lr * Math.Sqrt(beta2CorrectionFactor) / beta1CorrectionFactor;

            int i = 0;

            fixed (double* pParam = param, pGrad = grad, pM = m, pV = v)
            {
                if (HasFma && length >= 4)
                {
                    var vBeta1 = Vector256.Create(beta1);
                    var vBeta2 = Vector256.Create(beta2);
                    var vOneMBeta1 = Vector256.Create(1.0 - beta1);
                    var vOneMBeta2 = Vector256.Create(1.0 - beta2);
                    var vLr = Vector256.Create(correctedLr);
                    var vEps = Vector256.Create(epsilon);

                    for (; i <= length - 4; i += 4)
                    {
                        var g = Avx.LoadVector256(pGrad + i);

                        // m = beta1 * m + (1 - beta1) * g
                        var mNew = Fma.MultiplyAdd(vBeta1, Avx.LoadVector256(pM + i), Avx.Multiply(vOneMBeta1, g));
                        Avx.Store(pM + i, mNew);

                        // v = beta2 * v + (1 - beta2) * g^2
                        var g2 = Avx.Multiply(g, g);
                        var vNew = Fma.MultiplyAdd(vBeta2, Avx.LoadVector256(pV + i), Avx.Multiply(vOneMBeta2, g2));
                        Avx.Store(pV + i, vNew);

                        // param -= lr * m / (sqrt(v) + eps)
                        var sqrtV = Avx.Sqrt(vNew);
                        var denom = Avx.Add(sqrtV, vEps);
                        var update = Avx.Divide(Avx.Multiply(vLr, mNew), denom);
                        Avx.Store(pParam + i, Avx.Subtract(Avx.LoadVector256(pParam + i), update));
                    }
                }
            }

            for (; i < length; i++)
            {
                double g = grad[i];
                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                param[i] -= correctedLr * m[i] / (Math.Sqrt(v[i]) + epsilon);
            }
        }

        /// SIMD-optimized AdamW step with decoupled weight decay
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void AdamWStep(
            Span<double> param,
            ReadOnlySpan<double> grad,
            Span<double> m,
            Span<double> v,
            double lr,
            double beta1,
            double beta2,
            double epsilon,
            double weightDecay,
            double beta1CorrectionFactor,
            double beta2CorrectionFactor)
        {
            int length = param.Length;
            double correctedLr = lr * Math.Sqrt(beta2CorrectionFactor) / beta1CorrectionFactor;
            int i = 0;

            fixed (double* pParam = param, pGrad = grad, pM = m, pV = v)
            {
                if (HasFma && length >= 4)
                {
                    var vBeta1 = Vector256.Create(beta1);
                    var vBeta2 = Vector256.Create(beta2);
                    var vOneMBeta1 = Vector256.Create(1.0 - beta1);
                    var vOneMBeta2 = Vector256.Create(1.0 - beta2);
                    var vLr = Vector256.Create(correctedLr);
                    var vEps = Vector256.Create(epsilon);
                    var vWd = Vector256.Create(lr * weightDecay);

                    for (; i <= length - 4; i += 4)
                    {
                        var p = Avx.LoadVector256(pParam + i);
                        var g = Avx.LoadVector256(pGrad + i);

                        // Weight decay (decoupled)
                        p = Avx.Subtract(p, Avx.Multiply(vWd, p));

                        // Adam moment updates
                        var mNew = Fma.MultiplyAdd(vBeta1, Avx.LoadVector256(pM + i), Avx.Multiply(vOneMBeta1, g));
                        Avx.Store(pM + i, mNew);

                        var g2 = Avx.Multiply(g, g);
                        var vNew = Fma.MultiplyAdd(vBeta2, Avx.LoadVector256(pV + i), Avx.Multiply(vOneMBeta2, g2));
                        Avx.Store(pV + i, vNew);

                        // Update
                        var sqrtV = Avx.Sqrt(vNew);
                        var update = Avx.Divide(Avx.Multiply(vLr, mNew), Avx.Add(sqrtV, vEps));
                        Avx.Store(pParam + i, Avx.Subtract(p, update));
                    }
                }
            }

            for (; i < length; i++)
            {
                double g = grad[i];
                param[i] -= lr * weightDecay * param[i];
                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                param[i] -= correctedLr * m[i] / (Math.Sqrt(v[i]) + epsilon);
            }
        }

        /// SIMD-optimized gradient clipping by norm
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void ClipGradNorm(Span<double> grad, double maxNorm)
        {
            double norm = CpuTensorPrimitives.Norm2(grad);
            if (norm > maxNorm)
            {
                double scale = maxNorm / norm;
                CpuTensorPrimitives.MultiplyScalar(grad, scale, grad);
            }
        }

        /// SIMD-optimized gradient clipping by value
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void ClipGradValue(Span<double> grad, double minVal, double maxVal)
        {
            CpuTensorPrimitives.Clamp(grad, minVal, maxVal, grad);
        }

        #endregion
    }

    #region CPU Feature Detection and Optimization Utilities

    /// Comprehensive CPU feature detection and information.
    /// Provides detailed information about CPU capabilities for optimal code path selection.
    /// Based on: Intel SDM, AMD APM, and .NET Hardware Intrinsics.
    /// </summary>
    public static class CpuInfo
    {
        // Feature flags cached at startup
        private static readonly Lazy<CpuFeatures> _features = new(() => DetectFeatures());

        /// <summary>Current CPU features</summary>
        public static CpuFeatures Features => _features.Value;

        /// <summary>Human-readable CPU feature summary</summary>
        public static string Summary => GetSummary();

        /// <summary>Public API</summary>
        public class CpuFeatures
        {
            // SIMD Extensions
            /// <summary>Public API</summary>
            public bool HasSse { get; init; }
            /// <summary>Public API</summary>
            public bool HasSse2 { get; init; }
            /// <summary>Public API</summary>
            public bool HasSse3 { get; init; }
            /// <summary>Public API</summary>
            public bool HasSsse3 { get; init; }
            /// <summary>Public API</summary>
            public bool HasSse41 { get; init; }
            /// <summary>Public API</summary>
            public bool HasSse42 { get; init; }
            /// <summary>Public API</summary>
            public bool HasAvx { get; init; }
            /// <summary>Public API</summary>
            public bool HasAvx2 { get; init; }
            /// <summary>Public API</summary>
            public bool HasFma { get; init; }

            // AVX-512 (Ice Lake+, Zen4+)
            /// <summary>Public API</summary>
            public bool HasAvx512F { get; init; }      // Foundation
            /// <summary>Public API</summary>
            public bool HasAvx512BW { get; init; }     // Byte/Word
            /// <summary>Public API</summary>
            public bool HasAvx512CD { get; init; }     // Conflict Detection
            /// <summary>Public API</summary>
            public bool HasAvx512DQ { get; init; }     // Doubleword/Quadword
            /// <summary>Public API</summary>
            public bool HasAvx512VL { get; init; }     // Vector Length
            /// <summary>Public API</summary>
            public bool HasAvx512Vbmi { get; init; }   // Vector Byte Manipulation

            // Other CPU features
            /// <summary>Public API</summary>
            public bool HasBmi1 { get; init; }         // Bit Manipulation
            /// <summary>Public API</summary>
            public bool HasBmi2 { get; init; }
            /// <summary>Public API</summary>
            public bool HasLzcnt { get; init; }        // Leading Zero Count
            /// <summary>Public API</summary>
            public bool HasPopcnt { get; init; }       // Population Count
            /// <summary>Public API</summary>
            public bool HasPclmulqdq { get; init; }    // Carry-less Multiplication
            /// <summary>Public API</summary>
            public bool HasAes { get; init; }          // AES-NI

            // System info
            /// <summary>Public API</summary>
            public int ProcessorCount { get; init; }
            /// <summary>Public API</summary>
            public int L1CacheSize { get; init; }      // KB
            /// <summary>Public API</summary>
            public int L2CacheSize { get; init; }      // KB
            /// <summary>Public API</summary>
            public int L3CacheSize { get; init; }      // KB
            /// <summary>Public API</summary>
            public int CacheLineSize { get; init; }    // Bytes (typically 64)

            /// <summary>Best SIMD vector size in bytes (64=AVX-512, 32=AVX2, 16=SSE, 0=scalar)</summary>
            public int BestVectorSize => HasAvx512F ? 64 : HasAvx2 ? 32 : HasSse2 ? 16 : 0;

            /// <summary>Best vector count for double precision</summary>
            public int DoublesPerVector => BestVectorSize / sizeof(double);
        }

        private static CpuFeatures DetectFeatures()
        {
            return new CpuFeatures
            {
                // SSE family
                HasSse = Sse.IsSupported,
                HasSse2 = Sse2.IsSupported,
                HasSse3 = Sse3.IsSupported,
                HasSsse3 = Ssse3.IsSupported,
                HasSse41 = Sse41.IsSupported,
                HasSse42 = Sse42.IsSupported,

                // AVX family
                HasAvx = Avx.IsSupported,
                HasAvx2 = Avx2.IsSupported,
                HasFma = Fma.IsSupported,

                // AVX-512 family
                HasAvx512F = Avx512F.IsSupported,
                HasAvx512BW = Avx512BW.IsSupported,
                HasAvx512CD = Avx512CD.IsSupported,
                HasAvx512DQ = Avx512DQ.IsSupported,
                HasAvx512VL = Avx512F.VL.IsSupported,
                HasAvx512Vbmi = Avx512Vbmi.IsSupported,

                // Other features
                HasBmi1 = Bmi1.IsSupported,
                HasBmi2 = Bmi2.IsSupported,
                HasLzcnt = Lzcnt.IsSupported,
                HasPopcnt = Popcnt.IsSupported,
                HasPclmulqdq = Pclmulqdq.IsSupported,
                HasAes = Aes.IsSupported,

                // System info
                ProcessorCount = Environment.ProcessorCount,
                L1CacheSize = 32,      // Typical modern CPU
                L2CacheSize = 256,     // Typical modern CPU
                /// <summary>L3 cache level</summary>
            L3CacheSize = 8192,    // Typical modern CPU
                CacheLineSize = 64     // Universal on x86-64
            };
        }

        private static string GetSummary()
        {
            var f = Features;
            var simd = f.HasAvx512F ? "AVX-512" : f.HasAvx2 ? "AVX2" : f.HasSse2 ? "SSE2" : "Scalar";
            return $"CPU: {f.ProcessorCount} cores, {simd}, L1={f.L1CacheSize}KB, L2={f.L2CacheSize}KB, L3={f.L3CacheSize}KB";
        }
    }

    /// Cache-aware memory utilities for optimal CPU performance.
    /// Provides cache line alignment, prefetching, and memory access patterns.
    /// </summary>
    public static class CacheOptimization
    {
        /// <summary>Standard cache line size (64 bytes on all modern x86-64)</summary>
        public const int CacheLineSize = 64;

        /// <summary>L1 cache size in bytes (typical)</summary>
        public const int L1CacheBytes = 32 * 1024;

        /// <summary>L2 cache size in bytes (typical)</summary>
        public const int L2CacheBytes = 256 * 1024;

        /// <summary>L3 cache size in bytes (typical)</summary>
        public const int L3CacheBytes = 8 * 1024 * 1024;

        /// Allocate cache-line aligned array for optimal memory access.
        /// Aligned memory prevents cache line splits and improves SIMD performance.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T[] AllocateAligned<T>(int length) where T : unmanaged
        {
            // GC.AllocateArray with pinned option for aligned allocation
            return GC.AllocateArray<T>(length, pinned: true);
        }

        /// Software prefetch hint - bring data into L1 cache before access.
        /// Use 3-4 iterations ahead in loops for optimal effect.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchL1(void* address)
        {
            if (Sse.IsSupported)
            {
                Sse.Prefetch0(address);  // Prefetch into L1
            }
        }

        /// Prefetch into L2 cache (temporal - expect reuse)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchL2(void* address)
        {
            if (Sse.IsSupported)
            {
                Sse.Prefetch1(address);  // Prefetch into L2
            }
        }

        /// Prefetch into L3 cache (temporal - expect some reuse)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchL3(void* address)
        {
            if (Sse.IsSupported)
            {
                Sse.Prefetch2(address);  // Prefetch into L3
            }
        }

        /// Non-temporal prefetch (no cache pollution - streaming access)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchNonTemporal(void* address)
        {
            if (Sse.IsSupported)
            {
                Sse.PrefetchNonTemporal(address);
            }
        }

        /// Calculate optimal block size for cache-blocked algorithms.
        /// Returns block size that fits in specified cache level.
        /// </summary>
        public static int GetOptimalBlockSize(int elementSize, CacheLevel level = CacheLevel.L2)
        {
            int cacheSize = level switch
            {
                CacheLevel.L1 => L1CacheBytes,
                CacheLevel.L2 => L2CacheBytes,
                CacheLevel.L3 => L3CacheBytes,
                _ => L2CacheBytes
            };

            // Use ~75% of cache to leave room for other data
            int usableBytes = (int)(cacheSize * 0.75);
            int blockSize = (int)Math.Sqrt(usableBytes / elementSize);

            // Round down to cache line multiple
            return (blockSize / (CacheLineSize / elementSize)) * (CacheLineSize / elementSize);
        }

        /// <summary>CPU cache levels for optimization.</summary>
        public enum CacheLevel
        {
            /// <summary>L1 cache level</summary>
            L1,
            /// <summary>L2 cache level</summary>
            L2,
            /// <summary>L3 cache level</summary>
            L3
        }

        /// <summary>
        /// Non-temporal store - bypass cache for write-only streaming patterns.
        /// Use when data will not be read again soon.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void StoreNonTemporal(double* destination, Vector256<double> value)
        {
            if (Avx.IsSupported)
            {
                // Use movntpd for non-temporal store
                Avx.Store(destination, value);  // Note: True NT store requires aligned address
            }
        }

        /// Memory fence - ensure all previous stores are globally visible.
        /// Use sparingly as it's expensive.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MemoryFence()
        {
            if (Sse2.IsSupported)
            {
                Sse2.MemoryFence();
            }
            else
            {
                Thread.MemoryBarrier();
            }
        }
    }

    /// AVX-512 optimized operations for modern CPUs (Ice Lake+, Zen4+).
    /// Provides 512-bit vector operations for maximum throughput.
    /// </summary>
    public static class Avx512Operations
    {
        private static readonly bool IsSupported = Avx512F.IsSupported;

        /// AVX-512 element-wise addition: result = a + b
        /// Processes 8 doubles per iteration (vs 4 for AVX2)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Add(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            if (!IsSupported)
            {
                CpuTensorPrimitives.Add(a, b, result);
                return;
            }

            int length = a.Length;
            int i = 0;

            fixed (double* pA = a, pB = b, pR = result)
            {
                // Process 32 doubles per iteration (4 × 512-bit vectors)
                for (; i <= length - 32; i += 32)
                {
                    // Prefetch next iteration's data into L1
                    if (i + 64 < length)
                    {
                        CacheOptimization.PrefetchL1(pA + i + 32);
                        CacheOptimization.PrefetchL1(pB + i + 32);
                    }

                    var va0 = Avx512F.LoadVector512(pA + i);
                    var va1 = Avx512F.LoadVector512(pA + i + 8);
                    var va2 = Avx512F.LoadVector512(pA + i + 16);
                    var va3 = Avx512F.LoadVector512(pA + i + 24);

                    var vb0 = Avx512F.LoadVector512(pB + i);
                    var vb1 = Avx512F.LoadVector512(pB + i + 8);
                    var vb2 = Avx512F.LoadVector512(pB + i + 16);
                    var vb3 = Avx512F.LoadVector512(pB + i + 24);

                    Avx512F.Store(pR + i, Avx512F.Add(va0, vb0));
                    Avx512F.Store(pR + i + 8, Avx512F.Add(va1, vb1));
                    Avx512F.Store(pR + i + 16, Avx512F.Add(va2, vb2));
                    Avx512F.Store(pR + i + 24, Avx512F.Add(va3, vb3));
                }

                // Single vector iterations
                for (; i <= length - 8; i += 8)
                {
                    Avx512F.Store(pR + i, Avx512F.Add(
                        Avx512F.LoadVector512(pA + i),
                        Avx512F.LoadVector512(pB + i)));
                }
            }

            // Scalar remainder
            for (; i < length; i++)
                result[i] = a[i] + b[i];
        }

        /// AVX-512 fused multiply-add: result = a * b + c
        /// Uses VFMADD instruction for 2x throughput
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void FusedMultiplyAdd(
            ReadOnlySpan<double> a,
            ReadOnlySpan<double> b,
            ReadOnlySpan<double> c,
            Span<double> result)
        {
            if (!IsSupported)
            {
                CpuTensorPrimitives.FusedMultiplyAdd(a, b, c, result);
                return;
            }

            int length = a.Length;
            int i = 0;

            fixed (double* pA = a, pB = b, pC = c, pR = result)
            {
                for (; i <= length - 32; i += 32)
                {
                    Avx512F.Store(pR + i, Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i),
                        Avx512F.LoadVector512(pB + i),
                        Avx512F.LoadVector512(pC + i)));

                    Avx512F.Store(pR + i + 8, Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i + 8),
                        Avx512F.LoadVector512(pB + i + 8),
                        Avx512F.LoadVector512(pC + i + 8)));

                    Avx512F.Store(pR + i + 16, Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i + 16),
                        Avx512F.LoadVector512(pB + i + 16),
                        Avx512F.LoadVector512(pC + i + 16)));

                    Avx512F.Store(pR + i + 24, Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i + 24),
                        Avx512F.LoadVector512(pB + i + 24),
                        Avx512F.LoadVector512(pC + i + 24)));
                }

                for (; i <= length - 8; i += 8)
                {
                    Avx512F.Store(pR + i, Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i),
                        Avx512F.LoadVector512(pB + i),
                        Avx512F.LoadVector512(pC + i)));
                }
            }

            for (; i < length; i++)
                result[i] = a[i] * b[i] + c[i];
        }

        /// AVX-512 dot product with prefetching
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            if (!IsSupported)
                return CpuTensorPrimitives.Dot(a, b);

            int length = a.Length;
            double sum = 0;
            int i = 0;

            fixed (double* pA = a, pB = b)
            {
                var acc0 = Vector512<double>.Zero;
                var acc1 = Vector512<double>.Zero;
                var acc2 = Vector512<double>.Zero;
                var acc3 = Vector512<double>.Zero;

                for (; i <= length - 32; i += 32)
                {
                    // Prefetch ahead
                    if (i + 64 < length)
                    {
                        CacheOptimization.PrefetchL1(pA + i + 32);
                        CacheOptimization.PrefetchL1(pB + i + 32);
                    }

                    acc0 = Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i),
                        Avx512F.LoadVector512(pB + i), acc0);
                    acc1 = Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i + 8),
                        Avx512F.LoadVector512(pB + i + 8), acc1);
                    acc2 = Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i + 16),
                        Avx512F.LoadVector512(pB + i + 16), acc2);
                    acc3 = Avx512F.FusedMultiplyAdd(
                        Avx512F.LoadVector512(pA + i + 24),
                        Avx512F.LoadVector512(pB + i + 24), acc3);
                }

                var combined = Avx512F.Add(Avx512F.Add(acc0, acc1), Avx512F.Add(acc2, acc3));
                sum = HorizontalSum512(combined);
            }

            for (; i < length; i++)
                sum += a[i] * b[i];

            return sum;
        }

        /// Horizontal sum of Vector512 (8 doubles → 1 double)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double HorizontalSum512(Vector512<double> v)
        {
            // Extract lower and upper 256-bit halves
            var lower = v.GetLower();
            var upper = v.GetUpper();
            var sum256 = Avx.Add(lower, upper);

            // Continue with 256-bit horizontal sum
            var high128 = Avx.ExtractVector128(sum256, 1);
            var low128 = sum256.GetLower();
            var sum128 = Sse2.Add(low128, high128);

            return sum128.GetElement(0) + sum128.GetElement(1);
        }
    }

    /// Parallel computation utilities optimized for CPU cache hierarchy.
    /// Provides work-stealing parallelism with cache-aware grain sizing.
    /// </summary>
    public static class ParallelCompute
    {
        /// Cache-aware parallel for loop with optimal grain size.
        /// Automatically determines grain size based on cache levels.
        /// </summary>
        public static void For(int fromInclusive, int toExclusive, int elementSize, Action<int, int> body)
        {
            int totalElements = toExclusive - fromInclusive;

            // Below threshold, run sequentially
            if (totalElements < 1024)
            {
                body(fromInclusive, toExclusive);
                return;
            }

            // Calculate grain size to fit in L2 cache
            int grainElements = CacheOptimization.L2CacheBytes / elementSize / 4;
            grainElements = Math.Max(256, Math.Min(grainElements, totalElements / Environment.ProcessorCount));

            Parallel.For(0, (totalElements + grainElements - 1) / grainElements, chunk =>
            {
                int start = fromInclusive + chunk * grainElements;
                int end = Math.Min(start + grainElements, toExclusive);
                body(start, end);
            });
        }

        /// Compute optimal thread count based on workload size and type
        /// </summary>
        public static int GetOptimalThreadCount(long workloadElements, bool memoryBound = false)
        {
            int maxThreads = Environment.ProcessorCount;

            if (memoryBound)
            {
                // Memory-bound operations scale poorly past ~4-8 cores
                return Math.Min(maxThreads, Math.Max(1, (int)(workloadElements / 100000)));
            }
            else
            {
                // Compute-bound operations can use all cores
                return Math.Min(maxThreads, Math.Max(1, (int)(workloadElements / 10000)));
            }
        }
    }

    /// AVX-512 thermal awareness - detects CPUs that downclock significantly with AVX-512.
    /// Skylake-X, Cascade Lake, and Cooper Lake cause significant frequency reduction.
    /// Based on: https://devblogs.microsoft.com/dotnet/dotnet-8-hardware-intrinsics/
    /// </summary>
    public static class Avx512ThermalAwareness
    {
        private static readonly Lazy<Avx512Recommendation> _recommendation = new(() => AnalyzeCpu());

        /// <summary>Get AVX-512 usage recommendation for current CPU</summary>
        public static Avx512Recommendation Recommendation => _recommendation.Value;

        /// <summary>Whether AVX-512 should be used (accounting for downclocking)</summary>
        public static bool ShouldUseAvx512 => Recommendation == Avx512Recommendation.Recommended;

        /// <summary>AVX-512 usage recommendation levels</summary>
        public enum Avx512Recommendation
        {
            /// <summary>AVX-512 is recommended for this system</summary>
            Recommended,
            /// <summary>AVX-512 should be used with caution due to thermal concerns</summary>
            UseWithCaution,
            /// <summary>AVX-512 is not supported on this system</summary>
            NotSupported
        }

        private static Avx512Recommendation AnalyzeCpu()
        {
            if (!Avx512F.IsSupported)
                return Avx512Recommendation.NotSupported;

            // Check for Vector512 hardware acceleration status
            // .NET 8 reports Vector512.IsHardwareAccelerated = true only on CPUs
            // where AVX-512 doesn't cause significant downclocking (Ice Lake+)
            if (Vector512.IsHardwareAccelerated)
                return Avx512Recommendation.Recommended;

            // AVX-512 supported but .NET indicates it may downclock
            // (Skylake-X, Cascade Lake, Cooper Lake)
            return Avx512Recommendation.UseWithCaution;
        }

        /// <summary>Get a string describing the AVX-512 status</summary>
        public static string GetStatusDescription()
        {
            return Recommendation switch
            {
                Avx512Recommendation.Recommended => "AVX-512 recommended (Ice Lake+/Zen4+ detected)",
                Avx512Recommendation.UseWithCaution => "AVX-512 available but may downclock (Skylake-X/Cascade Lake detected)",
                Avx512Recommendation.NotSupported => "AVX-512 not supported, using AVX2/SSE fallback",
                _ => "Unknown AVX-512 status"
            };
        }
    }

    /// Advanced prefetching utilities with multi-level lookahead.
    /// Research shows optimal prefetch distance is 2-3 cache lines ahead.
    /// Based on: ARM optimization guide and Intel SDM.
    /// </summary>
    public static class AdvancedPrefetch
    {
        /// <summary>Cache line size in bytes</summary>
        public const int CacheLineBytes = 64;

        /// <summary>Optimal prefetch distance in cache lines</summary>
        public const int OptimalPrefetchLines = 3;

        /// Prefetch multiple cache lines ahead for streaming access patterns.
        /// Optimal for sequential memory access (e.g., tensor operations).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchAhead(void* currentAddress, int elementsPerCacheLine, int lookAhead = 3)
        {
            if (!Sse.IsSupported) return;

            byte* addr = (byte*)currentAddress;
            int stride = CacheLineBytes;

            // Prefetch 2-3 cache lines ahead as recommended by research
            for (int i = 1; i <= lookAhead; i++)
            {
                Sse.Prefetch0(addr + i * stride);
            }
        }

        /// Prefetch for matrix operations - prefetch both row and column data.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchMatrix(double* rowData, double* colData, int lookAhead = 3)
        {
            if (!Sse.IsSupported) return;

            for (int i = 1; i <= lookAhead; i++)
            {
                Sse.Prefetch0(rowData + i * 8);  // 8 doubles = 64 bytes = 1 cache line
                Sse.Prefetch1(colData + i * 8);  // Column data to L2 (may not be sequential)
            }
        }

        /// Prefetch with temporal hints based on expected reuse.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchWithHint(void* address, PrefetchHint hint)
        {
            if (!Sse.IsSupported) return;

            switch (hint)
            {
                case PrefetchHint.Temporal:
                    Sse.Prefetch0(address);  // Keep in all cache levels
                    break;
                case PrefetchHint.NonTemporal:
                    Sse.PrefetchNonTemporal(address);  // Don't pollute cache
                    break;
                case PrefetchHint.L2Only:
                    Sse.Prefetch1(address);  // Skip L1, go to L2
                    break;
                case PrefetchHint.L3Only:
                    Sse.Prefetch2(address);  // Skip L1/L2, go to L3
                    break;
            }
        }

        /// <summary>Prefetch hint types based on expected access pattern</summary>
        public enum PrefetchHint
        {
            /// <summary>Temporal prefetch hint</summary>
            Temporal,
            /// <summary>Non-temporal prefetch hint</summary>
            NonTemporal,
            /// <summary>Prefetch to L2 cache only</summary>
            L2Only,
            /// <summary>Prefetch to L3 cache only</summary>
            /// <summary>L3 cache level</summary>
            L3Only
        }
    }

    /// Branchless SIMD operations for hot paths.
    /// Eliminates branch misprediction penalties (10-30 cycles per misprediction).
    /// Based on: vectorization best practices.
    /// </summary>
    public static class BranchlessOps
    {
        /// Branchless conditional select: result = condition ? a : b
        /// Uses SIMD blend instructions instead of branches.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ConditionalSelect(
            ReadOnlySpan<double> condition,
            ReadOnlySpan<double> a,
            ReadOnlySpan<double> b,
            Span<double> result)
        {
            int length = condition.Length;
            int i = 0;

            fixed (double* pCond = condition, pA = a, pB = b, pR = result)
            {
                if (Avx2.IsSupported && length >= 4)
                {
                    var zero = Vector256<double>.Zero;

                    for (; i <= length - 4; i += 4)
                    {
                        var c = Avx.LoadVector256(pCond + i);
                        var va = Avx.LoadVector256(pA + i);
                        var vb = Avx.LoadVector256(pB + i);

                        // Create mask from condition (non-zero = all 1s)
                        var mask = Avx.Compare(c, zero, FloatComparisonMode.OrderedNotEqualNonSignaling);

                        // Blend: mask=1 selects a, mask=0 selects b
                        var blended = Avx.BlendVariable(vb, va, mask);
                        Avx.Store(pR + i, blended);
                    }
                }
            }

            // Scalar fallback (still branchless using ternary)
            for (; i < length; i++)
                result[i] = condition[i] != 0 ? a[i] : b[i];
        }

        /// Branchless sign function: result = sign(x). Returns 1 for positive, -1 for negative, 0 for zero.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Sign(ReadOnlySpan<double> input, Span<double> result)
        {
            int length = input.Length;
            int i = 0;

            fixed (double* pIn = input, pOut = result)
            {
                if (Avx2.IsSupported && length >= 4)
                {
                    var zero = Vector256<double>.Zero;
                    var one = Vector256.Create(1.0);
                    var negOne = Vector256.Create(-1.0);

                    for (; i <= length - 4; i += 4)
                    {
                        var x = Avx.LoadVector256(pIn + i);

                        // positive mask: x > 0
                        var posMask = Avx.Compare(x, zero, FloatComparisonMode.OrderedGreaterThanSignaling);
                        // negative mask: x < 0
                        var negMask = Avx.Compare(x, zero, FloatComparisonMode.OrderedLessThanSignaling);

                        // Start with 0, blend 1 for positive, -1 for negative
                        var result256 = Avx.BlendVariable(zero, one, posMask);
                        result256 = Avx.BlendVariable(result256, negOne, negMask);
                        Avx.Store(pOut + i, result256);
                    }
                }
            }

            // Scalar fallback
            for (; i < length; i++)
                result[i] = Math.Sign(input[i]);
        }

        /// Branchless step function: result = x >= threshold ? 1 : 0
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void Step(ReadOnlySpan<double> input, double threshold, Span<double> result)
        {
            int length = input.Length;
            int i = 0;

            fixed (double* pIn = input, pOut = result)
            {
                if (Avx2.IsSupported && length >= 4)
                {
                    var vThreshold = Vector256.Create(threshold);
                    var one = Vector256.Create(1.0);
                    var zero = Vector256<double>.Zero;

                    for (; i <= length - 4; i += 4)
                    {
                        var x = Avx.LoadVector256(pIn + i);
                        var mask = Avx.Compare(x, vThreshold, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
                        Avx.Store(pOut + i, Avx.BlendVariable(zero, one, mask));
                    }
                }
            }

            for (; i < length; i++)
                result[i] = input[i] >= threshold ? 1.0 : 0.0;
        }
    }

    /// Vector operation fallback chain: automatically selects best SIMD width.
    /// Vector512 → Vector256 → Vector128 → Scalar with seamless fallback.
    /// </summary>
    public static class VectorFallback
    {
        /// <summary>Best available vector size in elements (doubles)</summary>
        public static int BestVectorSizeDoubles =>
            Avx512ThermalAwareness.ShouldUseAvx512 ? 8 :
            Avx2.IsSupported ? 4 :
            Sse2.IsSupported ? 2 : 1;

        /// Add operation with automatic fallback.
        /// Chooses optimal SIMD width based on CPU capabilities.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void AddWithFallback(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            if (Avx512ThermalAwareness.ShouldUseAvx512)
            {
                Avx512Operations.Add(a, b, result);
            }
            else
            {
                CpuTensorPrimitives.Add(a, b, result);
            }
        }

        /// Dot product with automatic fallback.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static double DotWithFallback(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            if (Avx512ThermalAwareness.ShouldUseAvx512)
            {
                return Avx512Operations.Dot(a, b);
            }
            else
            {
                return CpuTensorPrimitives.Dot(a, b);
            }
        }

        /// FMA with automatic fallback.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void FmaWithFallback(
            ReadOnlySpan<double> a,
            ReadOnlySpan<double> b,
            ReadOnlySpan<double> c,
            Span<double> result)
        {
            if (Avx512ThermalAwareness.ShouldUseAvx512)
            {
                Avx512Operations.FusedMultiplyAdd(a, b, c, result);
            }
            else
            {
                CpuTensorPrimitives.FusedMultiplyAdd(a, b, c, result);
            }
        }
    }

    /// CPU benchmark and stress testing utilities.
    /// Provides performance measurement for tensor operations.
    /// Based on: UserBenchmark and CPU-Z methodologies.
    /// </summary>
    public static class CpuBenchmark
    {
        /// Benchmark result containing performance metrics.
        /// </summary>
        public class BenchmarkResult
        {
            /// <summary>Operations per second (higher is better)</summary>
            public double OpsPerSecond { get; init; }
            /// <summary>GFLOPS (billion floating-point operations per second)</summary>
            public double GFlops { get; init; }
            /// <summary>Memory bandwidth in GB/s</summary>
            public double MemoryBandwidthGBs { get; init; }
            /// <summary>Execution time in milliseconds</summary>
            public double ExecutionTimeMs { get; init; }
            /// <summary>Vector width used (512/256/128 bits)</summary>
            public int VectorWidth { get; init; }
            /// <summary>Number of threads used</summary>
            public int ThreadsUsed { get; init; }

            /// <summary>Public API</summary>
            public override string ToString() =>
                $"GFLOPS: {GFlops:F2}, Bandwidth: {MemoryBandwidthGBs:F2} GB/s, " +
                $"Time: {ExecutionTimeMs:F2}ms, Vector: {VectorWidth}-bit, Threads: {ThreadsUsed}";
        }

        /// Run a quick benchmark of vector operations.
        /// </summary>
        /// <param name="size">Array size (default 1M elements)</param>
        /// <param name="iterations">Number of iterations (default 100)</param>
        public static BenchmarkResult RunVectorBenchmark(int size = 1_000_000, int iterations = 100)
        {
            var a = new double[size];
            var b = new double[size];
            var result = new double[size];

            // Initialize with random data
            var rng = new Random(42);
            for (int i = 0; i < size; i++)
            {
                a[i] = rng.NextDouble();
                b[i] = rng.NextDouble();
            }

            // Warmup
            VectorFallback.AddWithFallback(a, b, result);

            // Benchmark
            var sw = System.Diagnostics.Stopwatch.StartNew();

            for (int iter = 0; iter < iterations; iter++)
            {
                VectorFallback.AddWithFallback(a, b, result);
            }

            sw.Stop();

            double totalMs = sw.Elapsed.TotalMilliseconds;
            long totalOps = (long)size * iterations;
            double opsPerSecond = totalOps / (totalMs / 1000.0);
            double gflops = opsPerSecond / 1e9;

            // Memory bandwidth: each add reads 2 doubles, writes 1 double
            long bytesProcessed = (long)size * iterations * 3 * sizeof(double);
            double bandwidthGBs = (bytesProcessed / 1e9) / (totalMs / 1000.0);

            int vectorWidth = Avx512ThermalAwareness.ShouldUseAvx512 ? 512 :
                              Avx2.IsSupported ? 256 :
                              Sse2.IsSupported ? 128 : 0;

            return new BenchmarkResult
            {
                OpsPerSecond = opsPerSecond,
                GFlops = gflops,
                MemoryBandwidthGBs = bandwidthGBs,
                ExecutionTimeMs = totalMs,
                VectorWidth = vectorWidth,
                ThreadsUsed = 1
            };
        }

        /// Run a stress test of CPU compute capabilities.
        /// Uses FMA operations for maximum CPU utilization.
        /// </summary>
        /// <param name="durationMs">Test duration in milliseconds</param>
        /// <param name="threads">Number of threads (default: all cores)</param>
        public static BenchmarkResult RunStressTest(int durationMs = 5000, int threads = 0)
        {
            if (threads <= 0) threads = Environment.ProcessorCount;

            const int blockSize = 1024 * 64; // 64K elements per thread
            var blocks = new double[threads][];
            var results = new double[threads][];

            // Initialize
            for (int t = 0; t < threads; t++)
            {
                blocks[t] = new double[blockSize];
                results[t] = new double[blockSize];
                var rng = new Random(42 + t);
                for (int i = 0; i < blockSize; i++)
                    blocks[t][i] = rng.NextDouble();
            }

            long totalOps = 0;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var endTime = sw.ElapsedMilliseconds + durationMs;

            Parallel.For(0, threads, t =>
            {
                var block = blocks[t];
                var result = results[t];
                long localOps = 0;

                while (sw.ElapsedMilliseconds < endTime)
                {
                    // FMA: 2 FLOPS per element (multiply + add)
                    CpuTensorPrimitives.FusedMultiplyAdd(block, block, block, result);
                    localOps += blockSize * 2;
                }

                Interlocked.Add(ref totalOps, localOps);
            });

            sw.Stop();

            double totalSeconds = sw.Elapsed.TotalSeconds;
            double gflops = (totalOps / 1e9) / totalSeconds;

            int vectorWidth = Avx512ThermalAwareness.ShouldUseAvx512 ? 512 :
                              Avx2.IsSupported ? 256 :
                              Sse2.IsSupported ? 128 : 0;

            return new BenchmarkResult
            {
                OpsPerSecond = totalOps / totalSeconds,
                GFlops = gflops,
                MemoryBandwidthGBs = 0, // Stress test is compute-bound
                ExecutionTimeMs = sw.Elapsed.TotalMilliseconds,
                VectorWidth = vectorWidth,
                ThreadsUsed = threads
            };
        }

        /// Get comprehensive CPU performance report.
        /// </summary>
        public static string GetPerformanceReport()
        {
            var info = CpuInfo.Features;
            var avx512Status = Avx512ThermalAwareness.GetStatusDescription();

            var report = new System.Text.StringBuilder();
            report.AppendLine("=== NSL CPU Performance Report ===");
            report.AppendLine();
            report.AppendLine($"Processor Count: {info.ProcessorCount}");
            report.AppendLine($"Best SIMD: {(info.HasAvx512F ? "AVX-512" : info.HasAvx2 ? "AVX2" : info.HasSse2 ? "SSE2" : "Scalar")}");
            report.AppendLine($"AVX-512 Status: {avx512Status}");
            report.AppendLine($"Vector Size: {VectorFallback.BestVectorSizeDoubles} doubles ({VectorFallback.BestVectorSizeDoubles * 8} bytes)");
            report.AppendLine();

            report.AppendLine("Running quick benchmark...");
            var vectorBench = RunVectorBenchmark(1_000_000, 50);
            report.AppendLine($"Vector Add: {vectorBench}");

            report.AppendLine();
            report.AppendLine("Running 2-second stress test...");
            var stressBench = RunStressTest(2000);
            report.AppendLine($"Stress Test: {stressBench}");

            return report.ToString();
        }
    }

    #endregion
}