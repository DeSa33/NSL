using System;
using System.Collections.Generic;
using System.Linq;

namespace NSL.Tensor
{
    /// <summary>
    /// Linear algebra operations for NSL.Tensor
    /// Implements SVD, Eigenvalue decomposition, Cholesky, QR, LU, and more
    /// All algorithms are pure C# - no external dependencies
    /// </summary>
    public static class LinearAlgebra
    {
        private const double EPSILON = 1e-12;
        private const int MAX_ITERATIONS = 1000;

        #region SVD (Singular Value Decomposition)

        /// <summary>
        /// Compute Singular Value Decomposition: A = U * S * V^T
        /// </summary>
        public static (Tensor U, Tensor S, Tensor Vh) SVD(Tensor input, bool fullMatrices = true)
        {
            if (input.NDim != 2)
                throw new ArgumentException("SVD requires a 2D tensor");

            var m = (int)input.Shape[0];
            var n = (int)input.Shape[1];
            var k = Math.Min(m, n);

            // Convert to double array for computation
            var A = new double[m, n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = input[i, j];

            // Use Golub-Reinsch algorithm (bidiagonalization + QR iteration)
            var (U, S, V) = GolubReinschSVD(A, m, n, fullMatrices);

            // Convert back to tensors
            var uShape = fullMatrices ? new long[] { m, m } : new long[] { m, k };
            var vShape = fullMatrices ? new long[] { n, n } : new long[] { k, n };

            var uData = new double[uShape[0] * uShape[1]];
            var sData = new double[k];
            var vData = new double[vShape[0] * vShape[1]];

            for (int i = 0; i < uShape[0]; i++)
                for (int j = 0; j < uShape[1]; j++)
                    uData[i * uShape[1] + j] = U[i, j];

            for (int i = 0; i < k; i++)
                sData[i] = S[i];

            for (int i = 0; i < vShape[0]; i++)
                for (int j = 0; j < vShape[1]; j++)
                    vData[i * vShape[1] + j] = V[i, j];

            return (
                new Tensor(uData, uShape),
                new Tensor(sData, new[] { (long)k }),
                new Tensor(vData, vShape)
            );
        }

        /// <summary>
        /// Compute only singular values
        /// </summary>
        public static Tensor SVDVals(Tensor input)
        {
            var (_, S, _) = SVD(input, fullMatrices: false);
            return S;
        }

        private static (double[,] U, double[] S, double[,] V) GolubReinschSVD(double[,] A, int m, int n, bool full)
        {
            var k = Math.Min(m, n);
            var U = new double[m, full ? m : k];
            var V = new double[full ? n : k, n];
            var S = new double[k];

            // Initialize U as identity or copy of A
            var work = new double[m, n];
            Array.Copy(A, work, A.Length);

            // Bidiagonalization using Householder reflections
            var d = new double[k];
            var e = new double[k > 1 ? k - 1 : 0];
            Bidiagonalize(work, m, n, d, e, U, V);

            // QR iteration on bidiagonal matrix
            SVDIteration(d, e, U, V, m, n, k);

            // Sort singular values in descending order
            for (int i = 0; i < k - 1; i++)
            {
                int maxIdx = i;
                for (int j = i + 1; j < k; j++)
                    if (Math.Abs(d[j]) > Math.Abs(d[maxIdx]))
                        maxIdx = j;

                if (maxIdx != i)
                {
                    (d[i], d[maxIdx]) = (d[maxIdx], d[i]);
                    // Swap columns in U and V
                    for (int row = 0; row < m; row++)
                        (U[row, i], U[row, maxIdx]) = (U[row, maxIdx], U[row, i]);
                    for (int col = 0; col < n; col++)
                        (V[i, col], V[maxIdx, col]) = (V[maxIdx, col], V[i, col]);
                }
            }

            for (int i = 0; i < k; i++)
                S[i] = Math.Abs(d[i]);

            return (U, S, V);
        }

        private static void Bidiagonalize(double[,] A, int m, int n, double[] d, double[] e, double[,] U, double[,] V)
        {
            var k = Math.Min(m, n);

            // Initialize U and V as identity matrices
            for (int i = 0; i < U.GetLength(0); i++)
                for (int j = 0; j < U.GetLength(1); j++)
                    U[i, j] = (i == j) ? 1.0 : 0.0;

            for (int i = 0; i < V.GetLength(0); i++)
                for (int j = 0; j < V.GetLength(1); j++)
                    V[i, j] = (i == j) ? 1.0 : 0.0;

            for (int i = 0; i < k; i++)
            {
                // Left Householder
                double sigma = 0;
                for (int j = i; j < m; j++)
                    sigma += A[j, i] * A[j, i];
                sigma = Math.Sqrt(sigma);

                if (sigma > EPSILON)
                {
                    if (A[i, i] < 0) sigma = -sigma;
                    for (int j = i; j < m; j++)
                        A[j, i] /= sigma;
                    A[i, i] += 1;

                    for (int j = i + 1; j < n; j++)
                    {
                        double s = 0;
                        for (int l = i; l < m; l++)
                            s += A[l, i] * A[l, j];
                        s /= A[i, i];
                        for (int l = i; l < m; l++)
                            A[l, j] -= s * A[l, i];
                    }

                    // Update U
                    for (int j = 0; j < U.GetLength(1); j++)
                    {
                        double s = 0;
                        for (int l = i; l < m; l++)
                            s += A[l, i] * U[l, j];
                        s /= A[i, i];
                        for (int l = i; l < m; l++)
                            U[l, j] -= s * A[l, i];
                    }
                }

                d[i] = -sigma;

                // Right Householder
                if (i < k - 1)
                {
                    sigma = 0;
                    for (int j = i + 1; j < n; j++)
                        sigma += A[i, j] * A[i, j];
                    sigma = Math.Sqrt(sigma);

                    if (sigma > EPSILON)
                    {
                        if (A[i, i + 1] < 0) sigma = -sigma;
                        for (int j = i + 1; j < n; j++)
                            A[i, j] /= sigma;
                        A[i, i + 1] += 1;

                        for (int j = i + 1; j < m; j++)
                        {
                            double s = 0;
                            for (int l = i + 1; l < n; l++)
                                s += A[i, l] * A[j, l];
                            s /= A[i, i + 1];
                            for (int l = i + 1; l < n; l++)
                                A[j, l] -= s * A[i, l];
                        }

                        // Update V
                        for (int j = 0; j < n; j++)
                        {
                            double s = 0;
                            for (int l = i + 1; l < n; l++)
                                s += A[i, l] * V[l, j];
                            s /= A[i, i + 1];
                            for (int l = i + 1; l < n; l++)
                                V[l, j] -= s * A[i, l];
                        }
                    }

                    e[i] = -sigma;
                }
            }
        }

        private static void SVDIteration(double[] d, double[] e, double[,] U, double[,] V, int m, int n, int k)
        {
            // Simple QR iteration for bidiagonal SVD
            for (int iter = 0; iter < MAX_ITERATIONS; iter++)
            {
                bool converged = true;
                for (int i = 0; i < e.Length; i++)
                {
                    if (Math.Abs(e[i]) > EPSILON * (Math.Abs(d[i]) + Math.Abs(d[i + 1])))
                    {
                        converged = false;
                        break;
                    }
                }
                if (converged) break;

                // Givens rotations to zero out superdiagonal elements
                for (int i = 0; i < e.Length; i++)
                {
                    if (Math.Abs(e[i]) <= EPSILON) continue;

                    double f = d[i];
                    double g = e[i];
                    double h = d[i + 1];

                    double t = Math.Sqrt(f * f + g * g);
                    double cs = f / t;
                    double sn = g / t;

                    d[i] = t;
                    e[i] = cs * g + sn * h;
                    d[i + 1] = -sn * g + cs * h;

                    // Apply rotation to V
                    for (int j = 0; j < n; j++)
                    {
                        double temp = V[i, j];
                        V[i, j] = cs * temp + sn * V[i + 1, j];
                        V[i + 1, j] = -sn * temp + cs * V[i + 1, j];
                    }
                }
            }
        }

        #endregion

        #region Eigenvalue Decomposition

        /// <summary>
        /// Compute eigenvalues and eigenvectors: A = V * diag(eigenvalues) * V^-1
        /// </summary>
        public static (Tensor eigenvalues, Tensor eigenvectors) Eig(Tensor input)
        {
            if (input.NDim != 2 || input.Shape[0] != input.Shape[1])
                throw new ArgumentException("Eig requires a square 2D tensor");

            var n = (int)input.Shape[0];
            var A = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = input[i, j];

            // Use QR algorithm for eigenvalue decomposition
            var (eigenvalues, eigenvectors) = QRAlgorithm(A, n);

            var evalData = new double[n];
            var evecData = new double[n * n];

            for (int i = 0; i < n; i++)
            {
                evalData[i] = eigenvalues[i];
                for (int j = 0; j < n; j++)
                    evecData[i * n + j] = eigenvectors[i, j];
            }

            return (
                new Tensor(evalData, new[] { (long)n }),
                new Tensor(evecData, new[] { (long)n, (long)n })
            );
        }

        /// <summary>
        /// Compute eigenvalues only
        /// </summary>
        public static Tensor EigVals(Tensor input)
        {
            var (eigenvalues, _) = Eig(input);
            return eigenvalues;
        }

        /// <summary>
        /// Eigenvalue decomposition for symmetric/Hermitian matrices (faster)
        /// </summary>
        public static (Tensor eigenvalues, Tensor eigenvectors) Eigh(Tensor input)
        {
            if (input.NDim != 2 || input.Shape[0] != input.Shape[1])
                throw new ArgumentException("Eigh requires a square 2D tensor");

            // For symmetric matrices, use Jacobi method which is more stable
            var n = (int)input.Shape[0];
            var A = new double[n, n];
            var V = new double[n, n];

            // Symmetrize and initialize V as identity
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] = (input[i, j] + input[j, i]) / 2;
                    V[i, j] = (i == j) ? 1.0 : 0.0;
                }
            }

            // Jacobi iteration
            JacobiEigenvalue(A, V, n);

            var evalData = new double[n];
            var evecData = new double[n * n];

            for (int i = 0; i < n; i++)
            {
                evalData[i] = A[i, i];
                for (int j = 0; j < n; j++)
                    evecData[j * n + i] = V[j, i];
            }

            // Sort eigenvalues in ascending order
            var indices = Enumerable.Range(0, n).OrderBy(i => evalData[i]).ToArray();
            var sortedEval = new double[n];
            var sortedEvec = new double[n * n];

            for (int i = 0; i < n; i++)
            {
                sortedEval[i] = evalData[indices[i]];
                for (int j = 0; j < n; j++)
                    sortedEvec[j * n + i] = evecData[j * n + indices[i]];
            }

            return (
                new Tensor(sortedEval, new[] { (long)n }),
                new Tensor(sortedEvec, new[] { (long)n, (long)n })
            );
        }

        private static (double[] eigenvalues, double[,] eigenvectors) QRAlgorithm(double[,] A, int n)
        {
            var eigenvalues = new double[n];
            var eigenvectors = new double[n, n];

            // Initialize eigenvectors as identity
            for (int i = 0; i < n; i++)
                eigenvectors[i, i] = 1.0;

            var work = (double[,])A.Clone();

            for (int iter = 0; iter < MAX_ITERATIONS; iter++)
            {
                // Check convergence
                bool converged = true;
                for (int i = 0; i < n - 1; i++)
                {
                    if (Math.Abs(work[i + 1, i]) > EPSILON * (Math.Abs(work[i, i]) + Math.Abs(work[i + 1, i + 1])))
                    {
                        converged = false;
                        break;
                    }
                }
                if (converged) break;

                // QR decomposition with shift
                double shift = work[n - 1, n - 1];
                for (int i = 0; i < n; i++)
                    work[i, i] -= shift;

                var (Q, R) = QRDecomposition(work, n, n);

                // A = R * Q + shift * I
                work = MatrixMultiply(R, Q, n, n, n);
                for (int i = 0; i < n; i++)
                    work[i, i] += shift;

                // Update eigenvectors
                eigenvectors = MatrixMultiply(eigenvectors, Q, n, n, n);
            }

            for (int i = 0; i < n; i++)
                eigenvalues[i] = work[i, i];

            return (eigenvalues, eigenvectors);
        }

        private static void JacobiEigenvalue(double[,] A, double[,] V, int n)
        {
            for (int iter = 0; iter < MAX_ITERATIONS; iter++)
            {
                // Find largest off-diagonal element
                int p = 0, q = 1;
                double maxVal = Math.Abs(A[0, 1]);

                for (int i = 0; i < n - 1; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (Math.Abs(A[i, j]) > maxVal)
                        {
                            maxVal = Math.Abs(A[i, j]);
                            p = i;
                            q = j;
                        }
                    }
                }

                if (maxVal < EPSILON) break;

                // Compute rotation angle
                double theta = (A[q, q] - A[p, p]) / (2 * A[p, q]);
                double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                double c = 1 / Math.Sqrt(t * t + 1);
                double s = t * c;

                // Apply rotation
                for (int i = 0; i < n; i++)
                {
                    double api = A[p, i];
                    double aqi = A[q, i];
                    A[p, i] = c * api - s * aqi;
                    A[q, i] = s * api + c * aqi;

                    double vip = V[i, p];
                    double viq = V[i, q];
                    V[i, p] = c * vip - s * viq;
                    V[i, q] = s * vip + c * viq;
                }

                for (int i = 0; i < n; i++)
                {
                    double aip = A[i, p];
                    double aiq = A[i, q];
                    A[i, p] = c * aip - s * aiq;
                    A[i, q] = s * aip + c * aiq;
                }

                A[p, q] = 0;
                A[q, p] = 0;
            }
        }

        #endregion

        #region QR Decomposition

        /// <summary>
        /// Compute QR decomposition: A = Q * R
        /// </summary>
        public static (Tensor Q, Tensor R) QR(Tensor input)
        {
            if (input.NDim != 2)
                throw new ArgumentException("QR requires a 2D tensor");

            var m = (int)input.Shape[0];
            var n = (int)input.Shape[1];

            var A = new double[m, n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = input[i, j];

            var (Q, R) = QRDecomposition(A, m, n);

            var qData = new double[m * m];
            var rData = new double[m * n];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                    qData[i * m + j] = Q[i, j];
                for (int j = 0; j < n; j++)
                    rData[i * n + j] = R[i, j];
            }

            return (
                new Tensor(qData, new[] { (long)m, (long)m }),
                new Tensor(rData, new[] { (long)m, (long)n })
            );
        }

        private static (double[,] Q, double[,] R) QRDecomposition(double[,] A, int m, int n)
        {
            var Q = new double[m, m];
            var R = new double[m, n];

            // Initialize Q as identity
            for (int i = 0; i < m; i++)
                Q[i, i] = 1.0;

            // Copy A to R
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    R[i, j] = A[i, j];

            // Householder reflections
            for (int k = 0; k < Math.Min(m - 1, n); k++)
            {
                // Compute Householder vector
                double sigma = 0;
                for (int i = k; i < m; i++)
                    sigma += R[i, k] * R[i, k];
                sigma = Math.Sqrt(sigma);

                if (sigma < EPSILON) continue;

                if (R[k, k] < 0) sigma = -sigma;

                var v = new double[m];
                for (int i = k; i < m; i++)
                    v[i] = R[i, k];
                v[k] += sigma;

                double beta = 2.0 / DotProduct(v, v, k, m);

                // Apply to R
                for (int j = k; j < n; j++)
                {
                    double s = 0;
                    for (int i = k; i < m; i++)
                        s += v[i] * R[i, j];
                    for (int i = k; i < m; i++)
                        R[i, j] -= beta * s * v[i];
                }

                // Apply to Q
                for (int j = 0; j < m; j++)
                {
                    double s = 0;
                    for (int i = k; i < m; i++)
                        s += v[i] * Q[i, j];
                    for (int i = k; i < m; i++)
                        Q[i, j] -= beta * s * v[i];
                }
            }

            // Transpose Q (we computed Q^T)
            var QT = new double[m, m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    QT[i, j] = Q[j, i];

            return (QT, R);
        }

        #endregion

        #region Cholesky Decomposition

        /// <summary>
        /// Compute Cholesky decomposition: A = L * L^T (for positive definite matrices)
        /// </summary>
        public static Tensor Cholesky(Tensor input)
        {
            if (input.NDim != 2 || input.Shape[0] != input.Shape[1])
                throw new ArgumentException("Cholesky requires a square 2D tensor");

            var n = (int)input.Shape[0];
            var L = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < j; k++)
                        sum += L[i, k] * L[j, k];

                    if (i == j)
                    {
                        double val = input[i, i] - sum;
                        if (val <= 0)
                            throw new ArgumentException("Matrix is not positive definite");
                        L[i, j] = Math.Sqrt(val);
                    }
                    else
                    {
                        L[i, j] = (input[i, j] - sum) / L[j, j];
                    }
                }
            }

            var data = new double[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    data[i * n + j] = L[i, j];

            return new Tensor(data, new[] { (long)n, (long)n });
        }

        #endregion

        #region LU Decomposition

        /// <summary>
        /// Compute LU decomposition with partial pivoting: P * A = L * U
        /// </summary>
        public static (Tensor L, Tensor U, Tensor P) LU(Tensor input)
        {
            if (input.NDim != 2)
                throw new ArgumentException("LU requires a 2D tensor");

            var m = (int)input.Shape[0];
            var n = (int)input.Shape[1];

            var A = new double[m, n];
            var P = new int[m];

            for (int i = 0; i < m; i++)
            {
                P[i] = i;
                for (int j = 0; j < n; j++)
                    A[i, j] = input[i, j];
            }

            // LU decomposition with partial pivoting
            for (int k = 0; k < Math.Min(m, n); k++)
            {
                // Find pivot
                int maxIdx = k;
                double maxVal = Math.Abs(A[k, k]);
                for (int i = k + 1; i < m; i++)
                {
                    if (Math.Abs(A[i, k]) > maxVal)
                    {
                        maxVal = Math.Abs(A[i, k]);
                        maxIdx = i;
                    }
                }

                // Swap rows
                if (maxIdx != k)
                {
                    (P[k], P[maxIdx]) = (P[maxIdx], P[k]);
                    for (int j = 0; j < n; j++)
                        (A[k, j], A[maxIdx, j]) = (A[maxIdx, j], A[k, j]);
                }

                if (Math.Abs(A[k, k]) < EPSILON) continue;

                // Elimination
                for (int i = k + 1; i < m; i++)
                {
                    A[i, k] /= A[k, k];
                    for (int j = k + 1; j < n; j++)
                        A[i, j] -= A[i, k] * A[k, j];
                }
            }

            // Extract L and U
            var lData = new double[m * m];
            var uData = new double[m * n];
            var pData = new double[m * m];

            for (int i = 0; i < m; i++)
            {
                pData[i * m + P[i]] = 1.0;
                for (int j = 0; j < m; j++)
                {
                    if (i == j)
                        lData[i * m + j] = 1.0;
                    else if (i > j && j < n)
                        lData[i * m + j] = A[i, j];
                }
                for (int j = 0; j < n; j++)
                {
                    if (i <= j)
                        uData[i * n + j] = A[i, j];
                }
            }

            return (
                new Tensor(lData, new[] { (long)m, (long)m }),
                new Tensor(uData, new[] { (long)m, (long)n }),
                new Tensor(pData, new[] { (long)m, (long)m })
            );
        }

        #endregion

        #region Matrix Inverse

        /// <summary>
        /// Compute matrix inverse
        /// </summary>
        public static Tensor Inv(Tensor input)
        {
            if (input.NDim != 2 || input.Shape[0] != input.Shape[1])
                throw new ArgumentException("Inverse requires a square 2D tensor");

            var n = (int)input.Shape[0];

            // Use Gauss-Jordan elimination
            var A = new double[n, 2 * n];

            // Augmented matrix [A | I]
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    A[i, j] = input[i, j];
                A[i, n + i] = 1.0;
            }

            // Forward elimination with partial pivoting
            for (int k = 0; k < n; k++)
            {
                // Find pivot
                int maxIdx = k;
                for (int i = k + 1; i < n; i++)
                    if (Math.Abs(A[i, k]) > Math.Abs(A[maxIdx, k]))
                        maxIdx = i;

                if (Math.Abs(A[maxIdx, k]) < EPSILON)
                    throw new ArgumentException("Matrix is singular");

                // Swap rows
                if (maxIdx != k)
                    for (int j = 0; j < 2 * n; j++)
                        (A[k, j], A[maxIdx, j]) = (A[maxIdx, j], A[k, j]);

                // Scale pivot row
                double pivot = A[k, k];
                for (int j = 0; j < 2 * n; j++)
                    A[k, j] /= pivot;

                // Eliminate column
                for (int i = 0; i < n; i++)
                {
                    if (i == k) continue;
                    double factor = A[i, k];
                    for (int j = 0; j < 2 * n; j++)
                        A[i, j] -= factor * A[k, j];
                }
            }

            // Extract inverse
            var data = new double[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    data[i * n + j] = A[i, n + j];

            return new Tensor(data, new[] { (long)n, (long)n });
        }

        /// <summary>
        /// Compute Moore-Penrose pseudo-inverse
        /// </summary>
        public static Tensor PInv(Tensor input, double rcond = 1e-15)
        {
            var (U, S, Vh) = SVD(input, fullMatrices: false);

            // Compute reciprocal of singular values above threshold
            var threshold = rcond * S.Data.Max();
            var sInv = S.Apply(s => Math.Abs(s) > threshold ? 1.0 / s : 0.0);

            // A+ = V * S^-1 * U^T
            var sInvDiag = Tensor.Zeros(Vh.Shape[0], U.Shape[1]);
            for (int i = 0; i < Math.Min(sInv.NumElements, Math.Min(sInvDiag.Shape[0], sInvDiag.Shape[1])); i++)
                sInvDiag[i, i] = sInv.Data[i];

            return TensorOps.MatMul(TensorOps.MatMul(Vh.T(), sInvDiag), U.T());
        }

        #endregion

        #region Determinant and Trace

        /// <summary>
        /// Compute matrix determinant
        /// </summary>
        public static Tensor Det(Tensor input)
        {
            if (input.NDim != 2 || input.Shape[0] != input.Shape[1])
                throw new ArgumentException("Determinant requires a square 2D tensor");

            var (_, U, P) = LU(input);

            // Det = (-1)^(number of swaps) * product of diagonal of U
            double det = 1.0;
            var n = (int)input.Shape[0];

            // Count swaps from permutation matrix
            int swaps = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (P[i, j] == 1 && i != j)
                        swaps++;
            swaps /= 2;

            if (swaps % 2 == 1) det = -1.0;

            for (int i = 0; i < n; i++)
                det *= U[i, i];

            return new Tensor(det);
        }

        /// <summary>
        /// Compute matrix trace (sum of diagonal elements)
        /// </summary>
        public static Tensor Trace(Tensor input)
        {
            if (input.NDim != 2)
                throw new ArgumentException("Trace requires a 2D tensor");

            var n = Math.Min(input.Shape[0], input.Shape[1]);
            double trace = 0;
            for (long i = 0; i < n; i++)
                trace += input[i, i];

            return new Tensor(trace);
        }

        #endregion

        #region Linear Solve

        /// <summary>
        /// Solve linear system Ax = b
        /// </summary>
        public static Tensor Solve(Tensor A, Tensor b)
        {
            if (A.NDim != 2 || A.Shape[0] != A.Shape[1])
                throw new ArgumentException("Solve requires a square matrix A");

            var n = (int)A.Shape[0];

            // LU decomposition
            var (L, U, P) = LU(A);

            // Permute b
            var pb = TensorOps.MatMul(P, b.Dimensions == 1 ? b.Unsqueeze(1) : b);

            // Forward substitution: Ly = Pb
            var y = Tensor.Zeros(pb.Shape);
            for (int i = 0; i < n; i++)
            {
                var cols = pb.Dimensions == 2 ? (int)pb.Shape[1] : 1;
                for (int k = 0; k < cols; k++)
                {
                    double sum = pb.Dimensions == 2 ? pb[i, k] : pb[i, 0];
                    for (int j = 0; j < i; j++)
                        sum -= L[i, j] * (y.Dimensions == 2 ? y[j, k] : y[j, 0]);
                    if (y.Dimensions == 2)
                        y[i, k] = sum;
                    else
                        y[i, 0] = sum;
                }
            }

            // Back substitution: Ux = y
            var x = Tensor.Zeros(y.Shape);
            for (int i = n - 1; i >= 0; i--)
            {
                var cols = y.Dimensions == 2 ? (int)y.Shape[1] : 1;
                for (int k = 0; k < cols; k++)
                {
                    double sum = y.Dimensions == 2 ? y[i, k] : y[i, 0];
                    for (int j = i + 1; j < n; j++)
                        sum -= U[i, j] * (x.Dimensions == 2 ? x[j, k] : x[j, 0]);
                    var val = sum / U[i, i];
                    if (x.Dimensions == 2)
                        x[i, k] = val;
                    else
                        x[i, 0] = val;
                }
            }

            return b.Dimensions == 1 ? x.Squeeze(1) : x;
        }

        #endregion

        #region Matrix Norms

        /// <summary>
        /// Compute matrix norm
        /// </summary>
        public static Tensor MatrixNorm(Tensor input, string ord = "fro")
        {
            if (input.NDim != 2)
                throw new ArgumentException("Matrix norm requires a 2D tensor");

            return ord switch
            {
                "fro" => input.Square().Sum().Sqrt(), // Frobenius norm
                "nuc" => SVDVals(input).Sum(), // Nuclear norm
                "1" => ColumnNorms(input, 1).Max(), // Max column sum
                "inf" => RowNorms(input, 1).Max(), // Max row sum
                "2" => SVDVals(input).Max(), // Spectral norm (largest singular value)
                _ => throw new ArgumentException($"Unknown norm: {ord}")
            };
        }

        private static Tensor ColumnNorms(Tensor input, double p)
        {
            var norms = new double[input.Shape[1]];
            for (int j = 0; j < input.Shape[1]; j++)
            {
                double sum = 0;
                for (int i = 0; i < input.Shape[0]; i++)
                    sum += Math.Pow(Math.Abs(input[i, j]), p);
                norms[j] = Math.Pow(sum, 1.0 / p);
            }
            return new Tensor(norms, new[] { input.Shape[1] });
        }

        private static Tensor RowNorms(Tensor input, double p)
        {
            var norms = new double[input.Shape[0]];
            for (int i = 0; i < input.Shape[0]; i++)
            {
                double sum = 0;
                for (int j = 0; j < input.Shape[1]; j++)
                    sum += Math.Pow(Math.Abs(input[i, j]), p);
                norms[i] = Math.Pow(sum, 1.0 / p);
            }
            return new Tensor(norms, new[] { input.Shape[0] });
        }

        #endregion

        #region Matrix Rank and Condition Number

        /// <summary>
        /// Compute matrix rank
        /// </summary>
        public static int MatrixRank(Tensor input, double tol = -1)
        {
            var S = SVDVals(input);
            var sMax = S.Data.Max();

            if (tol < 0)
                tol = Math.Max(input.Shape[0], input.Shape[1]) * sMax * EPSILON;

            int rank = 0;
            for (int i = 0; i < S.NumElements; i++)
                if (S.Data[i] > tol) rank++;

            return rank;
        }

        /// <summary>
        /// Compute condition number
        /// </summary>
        public static Tensor Cond(Tensor input, string p = "2")
        {
            if (p == "2")
            {
                var S = SVDVals(input);
                return new Tensor(S.Data.Max() / S.Data.Where(x => x > EPSILON).Min());
            }
            else
            {
                return MatrixNorm(input, p).Mul(MatrixNorm(Inv(input), p));
            }
        }

        #endregion

        #region Correlation and Covariance

        /// <summary>
        /// Compute correlation coefficients matrix
        /// </summary>
        public static Tensor CorrCoef(Tensor input)
        {
            if (input.NDim != 2)
                throw new ArgumentException("CorrCoef requires a 2D tensor");

            var n = (int)input.Shape[0];
            var m = (int)input.Shape[1];

            // Compute means
            var means = input.Mean(1, keepDim: true);

            // Center the data
            var centered = input.Sub(means);

            // Compute covariance matrix
            var cov = TensorOps.MatMul(centered, centered.T()).Div(m - 1);

            // Compute standard deviations
            var stdsData = new double[n * n];
            for (int i = 0; i < n; i++)
            {
                stdsData[i * n + i] = Math.Sqrt(cov[i, i]);
            }
            var stds = new Tensor(stdsData, new[] { (long)n, (long)n });

            // Compute correlation
            var corr = new double[n * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var stdProd = stds[i, i] * stds[j, j];
                    corr[i * n + j] = stdProd > EPSILON ? cov[i, j] / stdProd : 0;
                }
            }

            return new Tensor(corr, new[] { (long)n, (long)n });
        }

        #endregion

        #region Helper Methods

        private static double[,] MatrixMultiply(double[,] A, double[,] B, int m, int n, int p)
        {
            var C = new double[m, p];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < p; j++)
                    for (int k = 0; k < n; k++)
                        C[i, j] += A[i, k] * B[k, j];
            return C;
        }

        private static double DotProduct(double[] a, double[] b, int start, int end)
        {
            double sum = 0;
            for (int i = start; i < end; i++)
                sum += a[i] * b[i];
            return sum;
        }

        #endregion
    }

    /// <summary>
    /// Extension methods for tensor linear algebra
    /// </summary>
    public static class TensorLinAlgExtensions
    {
        /// <summary>Public API</summary>
        public static (Tensor U, Tensor S, Tensor Vh) SVD(this Tensor t, bool fullMatrices = true) => LinearAlgebra.SVD(t, fullMatrices);
        /// <summary>Public API</summary>
        public static Tensor SVDVals(this Tensor t) => LinearAlgebra.SVDVals(t);
        /// <summary>Public API</summary>
        public static (Tensor eigenvalues, Tensor eigenvectors) Eig(this Tensor t) => LinearAlgebra.Eig(t);
        /// <summary>Public API</summary>
        public static (Tensor eigenvalues, Tensor eigenvectors) Eigh(this Tensor t) => LinearAlgebra.Eigh(t);
        /// <summary>Public API</summary>
        public static (Tensor Q, Tensor R) QR(this Tensor t) => LinearAlgebra.QR(t);
        /// <summary>Public API</summary>
        public static Tensor Cholesky(this Tensor t) => LinearAlgebra.Cholesky(t);
        /// <summary>Public API</summary>
        public static (Tensor L, Tensor U, Tensor P) LU(this Tensor t) => LinearAlgebra.LU(t);
        /// <summary>Public API</summary>
        public static Tensor Inv(this Tensor t) => LinearAlgebra.Inv(t);
        /// <summary>Public API</summary>
        public static Tensor PInv(this Tensor t, double rcond = 1e-15) => LinearAlgebra.PInv(t, rcond);
        /// <summary>Public API</summary>
        public static Tensor Det(this Tensor t) => LinearAlgebra.Det(t);
        /// <summary>Public API</summary>
        public static Tensor Trace(this Tensor t) => LinearAlgebra.Trace(t);
        /// <summary>Public API</summary>
        public static Tensor Solve(this Tensor A, Tensor b) => LinearAlgebra.Solve(A, b);
        /// <summary>Public API</summary>
        public static Tensor MatrixNorm(this Tensor t, string ord = "fro") => LinearAlgebra.MatrixNorm(t, ord);
        /// <summary>Public API</summary>
        public static int MatrixRank(this Tensor t, double tol = -1) => LinearAlgebra.MatrixRank(t, tol);
        /// <summary>Public API</summary>
        public static Tensor Cond(this Tensor t, string p = "2") => LinearAlgebra.Cond(t, p);
        /// <summary>Public API</summary>
        public static Tensor CorrCoef(this Tensor t) => LinearAlgebra.CorrCoef(t);
    }
}