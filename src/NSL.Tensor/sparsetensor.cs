using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace NSL.Tensor
{
    #region Sparse Tensor Formats

    /// <summary>
    /// Sparse tensor storage format.
    /// </summary>
    public enum SparseFormat
    {
        /// <summary>Coordinate format</summary>
        COO,
        /// <summary>Compressed sparse row format</summary>
        CSR,
        /// <summary>Compressed sparse column format</summary>
        CSC,
        /// <summary>Block sparse row format</summary>
        BSR,
        /// <summary>Diagonal format</summary>
        DIA
    }

    #endregion

    #region COO Format (Coordinate List)

    /// <summary>
    /// Sparse tensor in COO (Coordinate) format.
    /// Stores non-zero elements as (index..., value) tuples.
    /// Best for incremental construction and format conversion.
    /// </summary>
    public class SparseCOO
    {
        private readonly List<long[]> _indices;
        private readonly List<double> _values;
        private readonly long[] _shape;
        private readonly int _ndim;

        /// <summary>Public API</summary>
        public long[] Shape => _shape;
        /// <summary>Public API</summary>
        public int NDim => _ndim;
        /// <summary>Public API</summary>
        public int NNZ => _values.Count;
        /// <summary>Public API</summary>
        public double Sparsity => 1.0 - (double)NNZ / TotalElements;
        /// <summary>Public API</summary>
        public long TotalElements => _shape.Aggregate(1L, (a, b) => a * b);

        /// <summary>Public API</summary>
        public SparseCOO(long[] shape)
        {
            _shape = shape;
            _ndim = shape.Length;
            _indices = new List<long[]>();
            _values = new List<double>();
        }

        /// <summary>Public API</summary>
        public SparseCOO(long[][] indices, double[] values, long[] shape)
        {
            _shape = shape;
            _ndim = shape.Length;
            _indices = new List<long[]>();
            _values = new List<double>(values);

            // Convert from column-major to row-major index storage
            for (int i = 0; i < values.Length; i++)
            {
                var idx = new long[_ndim];
                for (int d = 0; d < _ndim; d++)
                {
                    idx[d] = indices[d][i];
                }
                _indices.Add(idx);
            }
        }

        /// <summary>
        /// Set a value at the given indices.
        /// </summary>
        public void Set(double value, params long[] indices)
        {
            if (indices.Length != _ndim)
                throw new ArgumentException($"Expected {_ndim} indices, got {indices.Length}");

            ValidateIndices(indices);

            // Check if index already exists
            for (int i = 0; i < _indices.Count; i++)
            {
                if (IndicesEqual(_indices[i], indices))
                {
                    if (Math.Abs(value) < 1e-15)
                    {
                        // Remove zero
                        _indices.RemoveAt(i);
                        _values.RemoveAt(i);
                    }
                    else
                    {
                        _values[i] = value;
                    }
                    return;
                }
            }

            // Add new entry if non-zero
            if (Math.Abs(value) >= 1e-15)
            {
                _indices.Add((long[])indices.Clone());
                _values.Add(value);
            }
        }

        /// <summary>
        /// Get value at the given indices.
        /// </summary>
        public double Get(params long[] indices)
        {
            if (indices.Length != _ndim)
                throw new ArgumentException($"Expected {_ndim} indices, got {indices.Length}");

            for (int i = 0; i < _indices.Count; i++)
            {
                if (IndicesEqual(_indices[i], indices))
                    return _values[i];
            }

            return 0.0;
        }

        /// <summary>
        /// Convert to dense tensor.
        /// </summary>
        public Tensor ToDense()
        {
            var data = new double[TotalElements];

            for (int i = 0; i < _values.Count; i++)
            {
                var flatIdx = FlattenIndex(_indices[i]);
                data[flatIdx] = _values[i];
            }

            return new Tensor(data, _shape);
        }

        /// <summary>
        /// Convert to CSR format (for 2D tensors).
        /// </summary>
        public SparseCSR ToCSR()
        {
            if (_ndim != 2)
                throw new InvalidOperationException("CSR format requires 2D tensor");

            return SparseCSR.FromCOO(this);
        }

        /// <summary>
        /// Create from dense tensor, keeping only non-zero values.
        /// </summary>
        public static SparseCOO FromDense(Tensor tensor, double threshold = 1e-15)
        {
            var sparse = new SparseCOO(tensor.Shape);

            for (long i = 0; i < tensor.NumElements; i++)
            {
                if (Math.Abs(tensor.Data[i]) >= threshold)
                {
                    var indices = UnflattenIndex(i, tensor.Shape);
                    sparse._indices.Add(indices);
                    sparse._values.Add(tensor.Data[i]);
                }
            }

            return sparse;
        }

        /// <summary>
        /// Get indices array (column-major format for compatibility).
        /// </summary>
        public long[][] GetIndices()
        {
            var result = new long[_ndim][];
            for (int d = 0; d < _ndim; d++)
            {
                result[d] = new long[_values.Count];
                for (int i = 0; i < _values.Count; i++)
                {
                    result[d][i] = _indices[i][d];
                }
            }
            return result;
        }

        /// <summary>
        /// Get values array.
        /// </summary>
        public double[] GetValues() => _values.ToArray();

        /// <summary>
        /// Coalesce duplicate indices by summing values.
        /// </summary>
        public SparseCOO Coalesce()
        {
            var indexMap = new Dictionary<string, int>();
            var newIndices = new List<long[]>();
            var newValues = new List<double>();

            for (int i = 0; i < _values.Count; i++)
            {
                var key = string.Join(",", _indices[i]);

                if (indexMap.TryGetValue(key, out var existingIdx))
                {
                    newValues[existingIdx] += _values[i];
                }
                else
                {
                    indexMap[key] = newIndices.Count;
                    newIndices.Add(_indices[i]);
                    newValues.Add(_values[i]);
                }
            }

            var result = new SparseCOO(_shape);
            result._indices.AddRange(newIndices);
            result._values.AddRange(newValues);
            return result;
        }

        private void ValidateIndices(long[] indices)
        {
            for (int d = 0; d < _ndim; d++)
            {
                if (indices[d] < 0 || indices[d] >= _shape[d])
                    throw new ArgumentOutOfRangeException($"Index {indices[d]} out of range for dimension {d} with size {_shape[d]}");
            }
        }

        private bool IndicesEqual(long[] a, long[] b)
        {
            for (int i = 0; i < _ndim; i++)
            {
                if (a[i] != b[i]) return false;
            }
            return true;
        }

        private long FlattenIndex(long[] indices)
        {
            long flat = 0;
            long stride = 1;

            for (int d = _ndim - 1; d >= 0; d--)
            {
                flat += indices[d] * stride;
                stride *= _shape[d];
            }

            return flat;
        }

        private static long[] UnflattenIndex(long flat, long[] shape)
        {
            var indices = new long[shape.Length];
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                indices[d] = flat % shape[d];
                flat /= shape[d];
            }
            return indices;
        }
    }

    #endregion

    #region CSR Format (Compressed Sparse Row)

    /// <summary>
    /// Sparse matrix in CSR (Compressed Sparse Row) format.
    /// Efficient for row slicing and matrix-vector products.
    /// </summary>
    public class SparseCSR
    {
        private readonly double[] _values;
        private readonly int[] _colIndices;
        private readonly int[] _rowPointers;
        private readonly long[] _shape;

        /// <summary>Public API</summary>
        public long[] Shape => _shape;
        /// <summary>Public API</summary>
        public long Rows => _shape[0];
        /// <summary>Public API</summary>
        public long Cols => _shape[1];
        /// <summary>Public API</summary>
        public int NNZ => _values.Length;
        /// <summary>Public API</summary>
        public double Sparsity => 1.0 - (double)NNZ / (Rows * Cols);

        /// <summary>Public API</summary>
        public double[] Values => _values;
        /// <summary>Public API</summary>
        public int[] ColIndices => _colIndices;
        /// <summary>Public API</summary>
        public int[] RowPointers => _rowPointers;

        /// <summary>Public API</summary>
        public SparseCSR(double[] values, int[] colIndices, int[] rowPointers, long rows, long cols)
        {
            _values = values;
            _colIndices = colIndices;
            _rowPointers = rowPointers;
            _shape = new[] { rows, cols };
        }

        /// <summary>
        /// Create CSR from COO format.
        /// </summary>
        public static SparseCSR FromCOO(SparseCOO coo)
        {
            if (coo.NDim != 2)
                throw new ArgumentException("CSR requires 2D tensor");

            var rows = coo.Shape[0];
            var cols = coo.Shape[1];

            // Sort by row then column
            var indices = coo.GetIndices();
            var values = coo.GetValues();

            var entries = new List<(int row, int col, double val)>();
            for (int i = 0; i < values.Length; i++)
            {
                entries.Add(((int)indices[0][i], (int)indices[1][i], values[i]));
            }
            entries.Sort((a, b) => a.row == b.row ? a.col.CompareTo(b.col) : a.row.CompareTo(b.row));

            // Build CSR arrays
            var csrValues = new double[entries.Count];
            var colIdx = new int[entries.Count];
            var rowPtr = new int[rows + 1];

            int currentRow = 0;
            rowPtr[0] = 0;

            for (int i = 0; i < entries.Count; i++)
            {
                while (currentRow < entries[i].row)
                {
                    currentRow++;
                    rowPtr[currentRow] = i;
                }

                csrValues[i] = entries[i].val;
                colIdx[i] = entries[i].col;
            }

            while (currentRow < rows)
            {
                currentRow++;
                rowPtr[currentRow] = entries.Count;
            }

            return new SparseCSR(csrValues, colIdx, rowPtr, rows, cols);
        }

        /// <summary>
        /// Create CSR from dense tensor.
        /// </summary>
        public static SparseCSR FromDense(Tensor tensor, double threshold = 1e-15)
        {
            if (tensor.Shape.Length != 2)
                throw new ArgumentException("CSR requires 2D tensor");

            var coo = SparseCOO.FromDense(tensor, threshold);
            return FromCOO(coo);
        }

        /// <summary>
        /// Get value at (row, col).
        /// </summary>
        public double Get(int row, int col)
        {
            var start = _rowPointers[row];
            var end = _rowPointers[row + 1];

            for (int i = start; i < end; i++)
            {
                if (_colIndices[i] == col)
                    return _values[i];
                if (_colIndices[i] > col)
                    break;
            }

            return 0.0;
        }

        /// <summary>
        /// Get a row as a dense array.
        /// </summary>
        public double[] GetRow(int row)
        {
            var result = new double[Cols];
            var start = _rowPointers[row];
            var end = _rowPointers[row + 1];

            for (int i = start; i < end; i++)
            {
                result[_colIndices[i]] = _values[i];
            }

            return result;
        }

        /// <summary>
        /// Convert to dense tensor.
        /// </summary>
        public Tensor ToDense()
        {
            var data = new double[Rows * Cols];

            for (int row = 0; row < Rows; row++)
            {
                var start = _rowPointers[row];
                var end = _rowPointers[row + 1];

                for (int i = start; i < end; i++)
                {
                    data[row * Cols + _colIndices[i]] = _values[i];
                }
            }

            return new Tensor(data, _shape);
        }

        /// <summary>
        /// Sparse matrix - dense vector multiplication.
        /// </summary>
        public Tensor MatVec(Tensor vector)
        {
            if (vector.Shape.Length != 1 || vector.Shape[0] != Cols)
                throw new ArgumentException($"Vector size {vector.Shape[0]} doesn't match matrix columns {Cols}");

            var result = new double[Rows];

            for (int row = 0; row < Rows; row++)
            {
                var start = _rowPointers[row];
                var end = _rowPointers[row + 1];
                double sum = 0;

                for (int i = start; i < end; i++)
                {
                    sum += _values[i] * vector.Data[_colIndices[i]];
                }

                result[row] = sum;
            }

            return new Tensor(result, new[] { Rows });
        }

        /// <summary>
        /// Sparse matrix - dense matrix multiplication.
        /// </summary>
        public Tensor MatMul(Tensor dense)
        {
            if (dense.Shape.Length != 2 || dense.Shape[0] != Cols)
                throw new ArgumentException($"Matrix shape mismatch: ({Rows}x{Cols}) @ ({dense.Shape[0]}x{dense.Shape[1]})");

            var k = dense.Shape[1];
            var result = new double[Rows * k];

            for (int row = 0; row < Rows; row++)
            {
                var start = _rowPointers[row];
                var end = _rowPointers[row + 1];

                for (int i = start; i < end; i++)
                {
                    var col = _colIndices[i];
                    var val = _values[i];

                    for (int j = 0; j < k; j++)
                    {
                        result[row * k + j] += val * dense.Data[col * k + j];
                    }
                }
            }

            return new Tensor(result, new[] { Rows, k });
        }

        /// <summary>
        /// Transpose the sparse matrix.
        /// </summary>
        public SparseCSR Transpose()
        {
            // CSR transpose becomes CSC, which we convert back to CSR
            var coo = new SparseCOO(new[] { Cols, Rows });

            for (int row = 0; row < Rows; row++)
            {
                var start = _rowPointers[row];
                var end = _rowPointers[row + 1];

                for (int i = start; i < end; i++)
                {
                    coo.Set(_values[i], _colIndices[i], row);
                }
            }

            return FromCOO(coo);
        }
    }

    #endregion

    #region Block Sparse Format

    /// <summary>
    /// Block sparse matrix - stores dense blocks in a sparse pattern.
    /// Efficient for structured sparsity like attention masks.
    /// </summary>
    public class BlockSparse
    {
        private readonly Dictionary<(int, int), double[,]> _blocks;
        private readonly int _blockRows;
        private readonly int _blockCols;
        private readonly long[] _shape;

        /// <summary>Public API</summary>
        public long[] Shape => _shape;
        /// <summary>Public API</summary>
        public int BlockRows => _blockRows;
        /// <summary>Public API</summary>
        public int BlockCols => _blockCols;
        /// <summary>Public API</summary>
        public int NumBlocks => _blocks.Count;
        /// <summary>Public API</summary>
        public int TotalBlocks => (int)((_shape[0] / _blockRows) * (_shape[1] / _blockCols));
        /// <summary>Public API</summary>
        public double BlockSparsity => 1.0 - (double)NumBlocks / TotalBlocks;

        /// <summary>Public API</summary>
        public BlockSparse(long rows, long cols, int blockRows, int blockCols)
        {
            _shape = new[] { rows, cols };
            _blockRows = blockRows;
            _blockCols = blockCols;
            _blocks = new Dictionary<(int, int), double[,]>();
        }

        /// <summary>
        /// Set a block at the given block indices.
        /// </summary>
        public void SetBlock(int blockRow, int blockCol, double[,] block)
        {
            if (block.GetLength(0) != _blockRows || block.GetLength(1) != _blockCols)
                throw new ArgumentException($"Block size must be {_blockRows}x{_blockCols}");

            _blocks[(blockRow, blockCol)] = block;
        }

        /// <summary>
        /// Get a block at the given block indices.
        /// </summary>
        public double[,]? GetBlock(int blockRow, int blockCol)
        {
            return _blocks.TryGetValue((blockRow, blockCol), out var block) ? block : null;
        }

        /// <summary>
        /// Set value at element indices.
        /// </summary>
        public void Set(double value, int row, int col)
        {
            var blockRow = row / _blockRows;
            var blockCol = col / _blockCols;
            var localRow = row % _blockRows;
            var localCol = col % _blockCols;

            if (!_blocks.TryGetValue((blockRow, blockCol), out var block))
            {
                block = new double[_blockRows, _blockCols];
                _blocks[(blockRow, blockCol)] = block;
            }

            block[localRow, localCol] = value;
        }

        /// <summary>
        /// Get value at element indices.
        /// </summary>
        public double Get(int row, int col)
        {
            var blockRow = row / _blockRows;
            var blockCol = col / _blockCols;

            if (!_blocks.TryGetValue((blockRow, blockCol), out var block))
                return 0.0;

            return block[row % _blockRows, col % _blockCols];
        }

        /// <summary>
        /// Convert to dense tensor.
        /// </summary>
        public Tensor ToDense()
        {
            var data = new double[_shape[0] * _shape[1]];

            foreach (var ((br, bc), block) in _blocks)
            {
                var rowStart = br * _blockRows;
                var colStart = bc * _blockCols;

                for (int i = 0; i < _blockRows; i++)
                {
                    for (int j = 0; j < _blockCols; j++)
                    {
                        data[(rowStart + i) * _shape[1] + (colStart + j)] = block[i, j];
                    }
                }
            }

            return new Tensor(data, _shape);
        }

        /// <summary>
        /// Create from dense tensor with given block size.
        /// Only keeps blocks with at least one non-zero element.
        /// </summary>
        public static BlockSparse FromDense(Tensor tensor, int blockRows, int blockCols, double threshold = 1e-15)
        {
            if (tensor.Shape.Length != 2)
                throw new ArgumentException("BlockSparse requires 2D tensor");

            var rows = tensor.Shape[0];
            var cols = tensor.Shape[1];
            var result = new BlockSparse(rows, cols, blockRows, blockCols);

            var numBlockRows = (int)((rows + blockRows - 1) / blockRows);
            var numBlockCols = (int)((cols + blockCols - 1) / blockCols);

            for (int br = 0; br < numBlockRows; br++)
            {
                for (int bc = 0; bc < numBlockCols; bc++)
                {
                    var block = new double[blockRows, blockCols];
                    bool hasNonZero = false;

                    for (int i = 0; i < blockRows; i++)
                    {
                        var row = br * blockRows + i;
                        if (row >= rows) break;

                        for (int j = 0; j < blockCols; j++)
                        {
                            var col = bc * blockCols + j;
                            if (col >= cols) break;

                            var val = tensor.Data[row * cols + col];
                            block[i, j] = val;
                            if (Math.Abs(val) >= threshold)
                                hasNonZero = true;
                        }
                    }

                    if (hasNonZero)
                    {
                        result._blocks[(br, bc)] = block;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Block sparse matrix multiplication with dense matrix.
        /// </summary>
        public Tensor MatMul(Tensor dense)
        {
            var cols = dense.Shape[1];
            var result = new double[_shape[0] * cols];

            foreach (var ((br, bc), block) in _blocks)
            {
                var rowStart = br * _blockRows;
                var colStart = bc * _blockCols;

                for (int i = 0; i < _blockRows; i++)
                {
                    var row = rowStart + i;
                    if (row >= _shape[0]) break;

                    for (int j = 0; j < _blockCols; j++)
                    {
                        var col = colStart + j;
                        if (col >= _shape[1]) break;

                        var val = block[i, j];
                        if (Math.Abs(val) < 1e-15) continue;

                        for (int k = 0; k < cols; k++)
                        {
                            result[row * cols + k] += val * dense.Data[col * cols + k];
                        }
                    }
                }
            }

            return new Tensor(result, new[] { _shape[0], cols });
        }

        /// <summary>
        /// Get block indices that are non-zero.
        /// </summary>
        public IEnumerable<(int row, int col)> GetBlockIndices()
        {
            return _blocks.Keys;
        }
    }

    #endregion

    #region Sparse Operations

    /// <summary>
    /// Operations for sparse tensors.
    /// </summary>
    public static class SparseOps
    {
        /// <summary>
        /// Sparse-dense matrix multiplication.
        /// </summary>
        public static Tensor SparseDenseMatMul(SparseCSR sparse, Tensor dense)
        {
            return sparse.MatMul(dense);
        }

        /// <summary>
        /// Dense-sparse matrix multiplication (transposes sparse).
        /// </summary>
        public static Tensor DenseSparseMatMul(Tensor dense, SparseCSR sparse)
        {
            // A @ B = (B^T @ A^T)^T
            var sparseT = sparse.Transpose();
            var denseT = dense.T();
            var result = sparseT.MatMul(denseT);
            return result.T();
        }

        /// <summary>
        /// Sparse-sparse matrix multiplication (result is dense).
        /// </summary>
        public static Tensor SparseSparseMul(SparseCSR a, SparseCSR b)
        {
            // For simplicity, convert b to dense
            // In production, would use sparse-sparse algorithms
            return a.MatMul(b.ToDense());
        }

        /// <summary>
        /// Element-wise sparse + dense.
        /// </summary>
        public static Tensor SparseAdd(SparseCSR sparse, Tensor dense)
        {
            var result = dense.Clone();

            for (int row = 0; row < sparse.Rows; row++)
            {
                var start = sparse.RowPointers[row];
                var end = sparse.RowPointers[row + 1];

                for (int i = start; i < end; i++)
                {
                    var col = sparse.ColIndices[i];
                    result.Data[row * sparse.Cols + col] += sparse.Values[i];
                }
            }

            return result;
        }

        /// <summary>
        /// Element-wise sparse * scalar.
        /// </summary>
        public static SparseCSR SparseScale(SparseCSR sparse, double scalar)
        {
            var newValues = sparse.Values.Select(v => v * scalar).ToArray();
            return new SparseCSR(newValues, sparse.ColIndices, sparse.RowPointers, sparse.Rows, sparse.Cols);
        }

        /// <summary>
        /// Sparse softmax along rows.
        /// </summary>
        public static SparseCSR SparseSoftmax(SparseCSR sparse)
        {
            var newValues = new double[sparse.NNZ];

            for (int row = 0; row < sparse.Rows; row++)
            {
                var start = sparse.RowPointers[row];
                var end = sparse.RowPointers[row + 1];

                if (start >= end) continue;

                // Find max for stability
                double max = double.MinValue;
                for (int i = start; i < end; i++)
                {
                    max = Math.Max(max, sparse.Values[i]);
                }

                // Compute exp and sum
                double sum = 0;
                for (int i = start; i < end; i++)
                {
                    newValues[i] = Math.Exp(sparse.Values[i] - max);
                    sum += newValues[i];
                }

                // Normalize
                for (int i = start; i < end; i++)
                {
                    newValues[i] /= sum;
                }
            }

            return new SparseCSR(newValues, sparse.ColIndices, sparse.RowPointers, sparse.Rows, sparse.Cols);
        }

        /// <summary>
        /// Create a sparse attention mask.
        /// </summary>
        public static SparseCSR CreateCausalMask(long seqLen)
        {
            var nnz = (int)(seqLen * (seqLen + 1) / 2);
            var values = new double[nnz];
            var colIdx = new int[nnz];
            var rowPtr = new int[seqLen + 1];

            int idx = 0;
            for (int i = 0; i < seqLen; i++)
            {
                rowPtr[i] = idx;
                for (int j = 0; j <= i; j++)
                {
                    values[idx] = 1.0;
                    colIdx[idx] = j;
                    idx++;
                }
            }
            rowPtr[seqLen] = nnz;

            return new SparseCSR(values, colIdx, rowPtr, seqLen, seqLen);
        }

        /// <summary>
        /// Create a sliding window attention mask.
        /// </summary>
        public static SparseCSR CreateSlidingWindowMask(long seqLen, int windowSize)
        {
            var values = new List<double>();
            var colIdx = new List<int>();
            var rowPtr = new List<int> { 0 };

            for (int i = 0; i < seqLen; i++)
            {
                var start = Math.Max(0, i - windowSize);
                var end = Math.Min((int)seqLen, i + windowSize + 1);

                for (int j = start; j < end; j++)
                {
                    values.Add(1.0);
                    colIdx.Add(j);
                }
                rowPtr.Add(values.Count);
            }

            return new SparseCSR(values.ToArray(), colIdx.ToArray(), rowPtr.ToArray(), seqLen, seqLen);
        }

        /// <summary>
        /// Create a block-diagonal sparse matrix.
        /// </summary>
        public static SparseCSR CreateBlockDiagonal(int numBlocks, int blockSize)
        {
            var totalSize = numBlocks * blockSize;
            var nnz = numBlocks * blockSize * blockSize;

            var values = new double[nnz];
            var colIdx = new int[nnz];
            var rowPtr = new int[totalSize + 1];

            int idx = 0;
            for (int block = 0; block < numBlocks; block++)
            {
                var blockStart = block * blockSize;

                for (int i = 0; i < blockSize; i++)
                {
                    rowPtr[blockStart + i] = idx;
                    for (int j = 0; j < blockSize; j++)
                    {
                        values[idx] = 1.0;
                        colIdx[idx] = blockStart + j;
                        idx++;
                    }
                }
            }
            rowPtr[totalSize] = nnz;

            return new SparseCSR(values, colIdx, rowPtr, totalSize, totalSize);
        }
    }

    #endregion

    #region Sparse Embedding

    /// <summary>
    /// Sparse embedding layer for large vocabulary models.
    /// Uses sparse updates for efficient gradient computation.
    /// </summary>
    public class SparseEmbedding
    {
        private readonly Tensor _weight;
        private readonly int _numEmbeddings;
        private readonly int _embeddingDim;
        private readonly HashSet<int> _accessedIndices;

        /// <summary>Public API</summary>
        public int NumEmbeddings => _numEmbeddings;
        /// <summary>Public API</summary>
        public int EmbeddingDim => _embeddingDim;
        /// <summary>Public API</summary>
        public Tensor Weight => _weight;

        /// <summary>Public API</summary>
        public SparseEmbedding(int numEmbeddings, int embeddingDim)
        {
            _numEmbeddings = numEmbeddings;
            _embeddingDim = embeddingDim;
            _accessedIndices = new HashSet<int>();

            // Initialize weights
            var data = new double[numEmbeddings * embeddingDim];
            var scale = 1.0 / Math.Sqrt(embeddingDim);
            var rng = new Random();

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (rng.NextDouble() * 2 - 1) * scale;
            }

            _weight = new Tensor(data, new[] { (long)numEmbeddings, embeddingDim });
        }

        /// <summary>
        /// Look up embeddings for given indices.
        /// </summary>
        public Tensor Forward(int[] indices)
        {
            var batchSize = indices.Length;
            var output = new double[batchSize * _embeddingDim];

            for (int i = 0; i < batchSize; i++)
            {
                var idx = indices[i];
                _accessedIndices.Add(idx);

                for (int j = 0; j < _embeddingDim; j++)
                {
                    output[i * _embeddingDim + j] = _weight.Data[idx * _embeddingDim + j];
                }
            }

            return new Tensor(output, new[] { (long)batchSize, _embeddingDim });
        }

        /// <summary>
        /// Get sparse gradient for accessed indices.
        /// </summary>
        public SparseCOO GetSparseGradient(Tensor gradOutput, int[] indices)
        {
            var sparse = new SparseCOO(new[] { (long)_numEmbeddings, _embeddingDim });

            for (int i = 0; i < indices.Length; i++)
            {
                var idx = indices[i];
                for (int j = 0; j < _embeddingDim; j++)
                {
                    var grad = gradOutput.Data[i * _embeddingDim + j];
                    var existing = sparse.Get(idx, j);
                    sparse.Set(existing + grad, idx, j);
                }
            }

            return sparse;
        }

        /// <summary>
        /// Clear accessed indices tracking.
        /// </summary>
        public void ResetAccessTracking()
        {
            _accessedIndices.Clear();
        }

        /// <summary>
        /// Get indices that were accessed in forward pass.
        /// </summary>
        public int[] GetAccessedIndices()
        {
            return _accessedIndices.ToArray();
        }
    }

    #endregion

    #region Sparse Attention

    /// <summary>
    /// Sparse attention implementation for efficient transformers.
    /// </summary>
    public static class SparseAttention
    {
        /// <summary>
        /// Compute sparse attention with a given attention mask.
        /// </summary>
        public static Tensor Forward(
            Tensor query,     // [batch, heads, seq, head_dim]
            Tensor key,       // [batch, heads, seq, head_dim]
            Tensor value,     // [batch, heads, seq, head_dim]
            SparseCSR mask,   // [seq, seq]
            double scale = 0)
        {
            // For simplicity, assume batch=1, heads=1
            // In production, would handle batched attention

            var seqLen = query.Shape[^2];
            var headDim = query.Shape[^1];

            if (scale <= 0)
                scale = 1.0 / Math.Sqrt(headDim);

            // Compute Q @ K^T only for non-zero positions in mask
            var attention = new double[seqLen * seqLen];

            // Apply sparse attention pattern
            for (int i = 0; i < seqLen; i++)
            {
                var rowStart = mask.RowPointers[i];
                var rowEnd = mask.RowPointers[i + 1];

                double maxScore = double.MinValue;

                // Compute scores only for valid positions
                for (int idx = rowStart; idx < rowEnd; idx++)
                {
                    var j = mask.ColIndices[idx];
                    double score = 0;

                    for (int d = 0; d < headDim; d++)
                    {
                        score += query.Data[i * headDim + d] * key.Data[j * headDim + d];
                    }

                    score *= scale;
                    attention[i * seqLen + j] = score;
                    maxScore = Math.Max(maxScore, score);
                }

                // Softmax over valid positions
                double sumExp = 0;
                for (int idx = rowStart; idx < rowEnd; idx++)
                {
                    var j = mask.ColIndices[idx];
                    attention[i * seqLen + j] = Math.Exp(attention[i * seqLen + j] - maxScore);
                    sumExp += attention[i * seqLen + j];
                }

                for (int idx = rowStart; idx < rowEnd; idx++)
                {
                    var j = mask.ColIndices[idx];
                    attention[i * seqLen + j] /= sumExp;
                }
            }

            // Compute attention @ V
            var output = new double[seqLen * headDim];

            for (int i = 0; i < seqLen; i++)
            {
                var rowStart = mask.RowPointers[i];
                var rowEnd = mask.RowPointers[i + 1];

                for (int idx = rowStart; idx < rowEnd; idx++)
                {
                    var j = mask.ColIndices[idx];
                    var attnWeight = attention[i * seqLen + j];

                    for (int d = 0; d < headDim; d++)
                    {
                        output[i * headDim + d] += attnWeight * value.Data[j * headDim + d];
                    }
                }
            }

            return new Tensor(output, new[] { seqLen, headDim });
        }

        /// <summary>
        /// Strided sparse attention for long sequences.
        /// </summary>
        public static Tensor StridedSparseAttention(
            Tensor query,
            Tensor key,
            Tensor value,
            int localWindow,
            int stride)
        {
            var seqLen = query.Shape[^2];

            // Create strided pattern
            var mask = CreateStridedMask(seqLen, localWindow, stride);

            return Forward(query, key, value, mask);
        }

        private static SparseCSR CreateStridedMask(long seqLen, int localWindow, int stride)
        {
            var values = new List<double>();
            var colIdx = new List<int>();
            var rowPtr = new List<int> { 0 };

            for (int i = 0; i < seqLen; i++)
            {
                // Local attention window
                var start = Math.Max(0, i - localWindow);
                var end = Math.Min((int)seqLen, i + 1);

                for (int j = start; j < end; j++)
                {
                    values.Add(1.0);
                    colIdx.Add(j);
                }

                // Strided attention
                for (int j = 0; j < i; j += stride)
                {
                    if (j < start || j >= end)
                    {
                        values.Add(1.0);
                        colIdx.Add(j);
                    }
                }

                rowPtr.Add(values.Count);
            }

            return new SparseCSR(values.ToArray(), colIdx.ToArray(), rowPtr.ToArray(), seqLen, seqLen);
        }
    }

    #endregion
}