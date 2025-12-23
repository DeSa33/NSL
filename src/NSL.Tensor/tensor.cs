using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

namespace NSL.Tensor
{
    /// Data type for tensor elements
    /// </summary>
    public enum DType
    {
        /// <summary>32-bit floating point data type</summary>
        Float32,
        /// <summary>64-bit floating point data type</summary>
        Float64,
        /// <summary>32-bit integer data type</summary>
        Int32,
        /// <summary>64-bit integer data type</summary>
        Int64,
        /// <summary>Boolean data type</summary>
        Bool,
        /// <summary>64-bit complex number data type</summary>
        Complex64,
        /// <summary>128-bit complex number data type</summary>
        Complex128
    }

    /// Device type for tensor storage
    /// </summary>
    public enum DeviceType
    {
        /// <summary>CPU device type</summary>
        CPU,
        /// <summary>CUDA GPU device type</summary>
        CUDA,
        /// <summary>Metal GPU device type</summary>
        Metal
    }

    /// Device specification for tensor operations
    /// </summary>
    public readonly struct Device
    {
        /// <summary>Public API</summary>
        public DeviceType Type { get; }
        /// <summary>Public API</summary>
        public int Index { get; }

        /// <summary>Public API</summary>
        public Device(DeviceType type, int index = 0)
        {
            Type = type;
            Index = index;
        }

        /// <summary>Public API</summary>
        public static Device CPU => new Device(DeviceType.CPU);
        /// <summary>Public API</summary>
        public static Device CUDA(int index = 0) => new Device(DeviceType.CUDA, index);

        /// <summary>Public API</summary>
        public override string ToString() => Type == DeviceType.CPU ? "cpu" : $"cuda:{Index}";
    }

    /// Core Tensor class - Multi-dimensional array with automatic differentiation support
    /// </summary>
    public class Tensor : IDisposable
    {
        #region Fields

        private static int _idCounter;
        private readonly int _id;
        private double[] _data;
        private readonly long[] _shape;
        private readonly long[] _strides;
        private readonly DType _dtype;
        private readonly Device _device;
        private bool _requiresGrad;
        private Tensor? _grad;
        private GradientFunction? _gradFn;
        private bool _isLeaf;
        private bool _disposed;

        #endregion

        #region Properties

        /// <summary>Raw data array</summary>
        public double[] Data => _data;

        /// <summary>Shape of the tensor</summary>
        public long[] Shape => _shape;

        /// <summary>Number of dimensions</summary>
        public int Dimensions => _shape.Length;

        /// <summary>Alias for Dimensions</summary>
        public int NDim => Dimensions;

        /// <summary>Total number of elements</summary>
        public long NumElements => _shape.Length == 0 ? 1 : _shape.Aggregate(1L, (a, b) => a * b);

        /// <summary>Data type</summary>
        public DType DType => _dtype;

        /// <summary>Device where tensor is stored</summary>
        public Device Device => _device;

        /// <summary>Whether gradients are tracked</summary>
        public bool RequiresGrad
        {
            get => _requiresGrad;
            set
            {
                _requiresGrad = value;
                if (!value) _grad = null;
            }
        }

        /// <summary>Gradient tensor</summary>
        public Tensor? Grad => _grad;

        /// <summary>Unique identifier for the tensor</summary>
        public int Id => _id;

        /// <summary>Gradient function for autograd</summary>
        public GradientFunction? GradFn
        {
            get => _gradFn;
            internal set => _gradFn = value;
        }

        /// <summary>Whether this is a leaf tensor (created by user, not by operation)</summary>
        public bool IsLeaf => _isLeaf;

        /// <summary>Strides for each dimension</summary>
        public long[] Strides => _strides;

        #endregion

        #region Constructors

        /// Create a tensor from data array with specified shape
        /// </summary>
        public Tensor(double[] data, long[] shape, bool requiresGrad = false, Device? device = null, DType dtype = DType.Float64)
        {
            _id = System.Threading.Interlocked.Increment(ref _idCounter);
            _data = data ?? throw new ArgumentNullException(nameof(data));
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _dtype = dtype;
            _device = device ?? Device.CPU;
            _requiresGrad = requiresGrad;
            _isLeaf = true;
            _strides = ComputeStrides(shape);

            var expectedElements = _shape.Length == 0 ? 1 : _shape.Aggregate(1L, (a, b) => a * b);
            if (_data.Length != expectedElements)
            {
                throw new ArgumentException($"Data length {_data.Length} doesn't match shape {string.Join("x", _shape)} = {expectedElements}");
            }
        }

        /// Create a scalar tensor
        /// </summary>
        public Tensor(double value, bool requiresGrad = false, Device? device = null)
            : this(new[] { value }, Array.Empty<long>(), requiresGrad, device)
        {
        }

        /// Internal constructor for operations
        /// </summary>
        internal Tensor(double[] data, long[] shape, GradientFunction? gradFn, bool requiresGrad, Device device, DType dtype = DType.Float64)
        {
            _id = System.Threading.Interlocked.Increment(ref _idCounter);
            _data = data;
            _shape = shape;
            _strides = ComputeStrides(shape);
            _gradFn = gradFn;
            _requiresGrad = requiresGrad;
            _isLeaf = gradFn == null;
            _device = device;
            _dtype = dtype;
        }

        #endregion

        #region Static Factory Methods

        /// <summary>Create tensor filled with zeros</summary>
        public static Tensor Zeros(params long[] shape) => Zeros(shape, false, Device.CPU);

        /// <summary>Create tensor filled with zeros</summary>
        public static Tensor Zeros(long[] shape, bool requiresGrad = false, Device? device = null, DType dtype = DType.Float64)
        {
            var size = shape.Length == 0 ? 1 : shape.Aggregate(1L, (a, b) => a * b);
            return new Tensor(new double[size], shape, requiresGrad, device ?? Device.CPU, dtype);
        }

        /// <summary>Create tensor filled with ones</summary>
        public static Tensor Ones(params long[] shape) => Ones(shape, false, Device.CPU);

        /// <summary>Create tensor filled with ones</summary>
        public static Tensor Ones(long[] shape, bool requiresGrad = false, Device? device = null, DType dtype = DType.Float64)
        {
            var size = shape.Length == 0 ? 1 : shape.Aggregate(1L, (a, b) => a * b);
            var data = new double[size];
            Array.Fill(data, 1.0);
            return new Tensor(data, shape, requiresGrad, device ?? Device.CPU, dtype);
        }

        /// <summary>Create tensor filled with specified value</summary>
        public static Tensor Full(long[] shape, double fillValue, bool requiresGrad = false, Device? device = null)
        {
            var size = shape.Length == 0 ? 1 : shape.Aggregate(1L, (a, b) => a * b);
            var data = new double[size];
            Array.Fill(data, fillValue);
            return new Tensor(data, shape, requiresGrad, device ?? Device.CPU);
        }

        /// <summary>Create tensor with random values from uniform distribution [0, 1)</summary>
        public static Tensor Rand(params long[] shape) => Rand(shape, false, Device.CPU);

        /// <summary>Create tensor with random values from uniform distribution [0, 1)</summary>
        public static Tensor Rand(long[] shape, bool requiresGrad = false, Device? device = null)
        {
            var size = shape.Length == 0 ? 1 : shape.Aggregate(1L, (a, b) => a * b);
            var data = new double[size];
            var random = Random.Shared;
            for (int i = 0; i < size; i++)
                data[i] = random.NextDouble();
            return new Tensor(data, shape, requiresGrad, device ?? Device.CPU);
        }

        /// <summary>Create tensor with random values from standard normal distribution</summary>
        public static Tensor Randn(params long[] shape) => Randn(shape, false, Device.CPU);

        /// <summary>Create tensor with random values from standard normal distribution</summary>
        public static Tensor Randn(long[] shape, bool requiresGrad = false, Device? device = null)
        {
            var size = shape.Length == 0 ? 1 : shape.Aggregate(1L, (a, b) => a * b);
            var data = new double[size];
            var random = Random.Shared;

            // Box-Muller transform for normal distribution
            for (int i = 0; i < size; i += 2)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = random.NextDouble();
                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;

                data[i] = radius * Math.Cos(theta);
                if (i + 1 < size)
                    data[i + 1] = radius * Math.Sin(theta);
            }

            return new Tensor(data, shape, requiresGrad, device ?? Device.CPU);
        }

        /// <summary>Create identity matrix</summary>
        public static Tensor Eye(long n, bool requiresGrad = false, Device? device = null)
        {
            var data = new double[n * n];
            for (long i = 0; i < n; i++)
                data[i * n + i] = 1.0;
            return new Tensor(data, new[] { n, n }, requiresGrad, device ?? Device.CPU);
        }

        /// <summary>Create tensor with evenly spaced values</summary>
        public static Tensor Arange(double start, double end, double step = 1.0, bool requiresGrad = false, Device? device = null)
        {
            var count = (long)Math.Ceiling((end - start) / step);
            if (count <= 0) return new Tensor(Array.Empty<double>(), new[] { 0L }, requiresGrad, device ?? Device.CPU);

            var data = new double[count];
            for (long i = 0; i < count; i++)
                data[i] = start + i * step;
            return new Tensor(data, new[] { count }, requiresGrad, device ?? Device.CPU);
        }

        /// <summary>Create tensor with evenly spaced values between start and end</summary>
        public static Tensor Linspace(double start, double end, long steps, bool requiresGrad = false, Device? device = null)
        {
            if (steps <= 0) return new Tensor(Array.Empty<double>(), new[] { 0L }, requiresGrad, device ?? Device.CPU);
            if (steps == 1) return new Tensor(new[] { start }, new[] { 1L }, requiresGrad, device ?? Device.CPU);

            var data = new double[steps];
            var step = (end - start) / (steps - 1);
            for (long i = 0; i < steps; i++)
                data[i] = start + i * step;
            return new Tensor(data, new[] { steps }, requiresGrad, device ?? Device.CPU);
        }

        /// <summary>Create tensor from nested arrays</summary>
        public static Tensor FromArray(Array array, bool requiresGrad = false, Device? device = null)
        {
            var shape = GetArrayShape(array);
            var data = FlattenArray(array);
            return new Tensor(data, shape, requiresGrad, device ?? Device.CPU);
        }

        /// <summary>Create 1D tensor from values</summary>
        public static Tensor FromValues(params double[] values)
        {
            return new Tensor(values.ToArray(), new[] { (long)values.Length });
        }

        /// <summary>Create tensor of ones with same shape as another tensor</summary>
        public static Tensor OnesLike(Tensor other, bool? requiresGrad = null, Device? device = null)
        {
            return Ones(other.Shape, requiresGrad ?? other.RequiresGrad, device ?? other.Device);
        }

        /// <summary>Create tensor of zeros with same shape as another tensor</summary>
        public static Tensor ZerosLike(Tensor other, bool? requiresGrad = null, Device? device = null)
        {
            return Zeros(other.Shape, requiresGrad ?? other.RequiresGrad, device ?? other.Device);
        }

        #endregion

        #region Indexing and Slicing

        /// <summary>Get or set element by indices</summary>
        public double this[params long[] indices]
        {
            get
            {
                var flatIndex = GetFlatIndex(indices);
                return _data[flatIndex];
            }
            set
            {
                var flatIndex = GetFlatIndex(indices);
                _data[flatIndex] = value;
            }
        }

        /// <summary>Get scalar value (for 0-d or 1-element tensors)</summary>
        public T Scalar<T>() where T : struct
        {
            if (NumElements != 1)
                throw new InvalidOperationException("Scalar() only works on single-element tensors");
            return (T)Convert.ChangeType(_data[0], typeof(T));
        }

        /// <summary>Get scalar value as double (compatible name for PyTorch-like API)</summary>
        public double ToScalar() => Scalar<double>();

        /// <summary>Slice tensor along dimension</summary>
        public Tensor Slice(int dim, long start, long end)
        {
            if (dim < 0) dim += Dimensions;
            if (dim < 0 || dim >= Dimensions)
                throw new ArgumentOutOfRangeException(nameof(dim));

            var dimSize = _shape[dim];
            if (start < 0) start += dimSize;
            if (end < 0) end += dimSize;

            start = Math.Max(0, Math.Min(start, dimSize));
            end = Math.Max(0, Math.Min(end, dimSize));

            if (end <= start)
                return Zeros(_shape.Select((s, i) => i == dim ? 0L : s).ToArray());

            var newShape = _shape.ToArray();
            newShape[dim] = end - start;

            var newSize = newShape.Aggregate(1L, (a, b) => a * b);
            var newData = new double[newSize];

            // Copy data
            CopySlice(_data, _shape, _strides, newData, newShape, dim, start, end);

            return new Tensor(newData, newShape, _requiresGrad, _device);
        }

        /// <summary>Set slice of tensor along dimension with values from another tensor</summary>
        public void SetSlice(int dim, long start, long end, Tensor value)
        {
            if (dim < 0) dim += Dimensions;
            if (dim < 0 || dim >= Dimensions)
                throw new ArgumentOutOfRangeException(nameof(dim));

            var dimSize = _shape[dim];
            if (start < 0) start += dimSize;
            if (end < 0) end += dimSize;

            start = Math.Max(0, Math.Min(start, dimSize));
            end = Math.Max(0, Math.Min(end, dimSize));

            if (end <= start) return;

            // Copy data from value into this tensor's slice
            var sliceSize = end - start;
            var outerSize = dim == 0 ? 1 : _shape.Take(dim).Aggregate(1L, (a, b) => a * b);
            var innerSize = dim == Dimensions - 1 ? 1 : _shape.Skip(dim + 1).Aggregate(1L, (a, b) => a * b);

            for (long outer = 0; outer < outerSize; outer++)
            {
                for (long i = 0; i < sliceSize; i++)
                {
                    for (long inner = 0; inner < innerSize; inner++)
                    {
                        var dstIdx = outer * _strides[dim > 0 ? dim - 1 : 0] * (dim > 0 ? _shape[dim] : 1) +
                                     (start + i) * innerSize + inner;
                        var srcIdx = outer * value._strides[dim > 0 ? dim - 1 : 0] * (dim > 0 ? sliceSize : 1) +
                                     i * innerSize + inner;
                        if (srcIdx < value._data.Length)
                            _data[dstIdx] = value._data[srcIdx];
                    }
                }
            }
        }

        /// <summary>Narrow tensor along dimension</summary>
        public Tensor Narrow(int dim, long start, long length)
        {
            return Slice(dim, start, start + length);
        }

        /// <summary>Slice tensor along a specific axis (alias for Slice)</summary>
        public Tensor SliceAxis(int axis, long start, long end)
        {
            return Slice(axis, start, end);
        }

        /// <summary>Select single index along dimension (reduces dimensionality)</summary>
        public Tensor Select(int dim, long index)
        {
            if (dim < 0) dim += Dimensions;
            if (dim < 0 || dim >= Dimensions)
                throw new ArgumentOutOfRangeException(nameof(dim));

            if (index < 0) index += _shape[dim];
            if (index < 0 || index >= _shape[dim])
                throw new ArgumentOutOfRangeException(nameof(index));

            var newShape = _shape.Where((_, i) => i != dim).ToArray();
            if (newShape.Length == 0) newShape = Array.Empty<long>();

            var newSize = newShape.Length == 0 ? 1 : newShape.Aggregate(1L, (a, b) => a * b);
            var newData = new double[newSize];

            // Copy selected data
            var outerSize = dim == 0 ? 1 : _shape.Take(dim).Aggregate(1L, (a, b) => a * b);
            var innerSize = dim == Dimensions - 1 ? 1 : _shape.Skip(dim + 1).Aggregate(1L, (a, b) => a * b);
            var dimStride = _strides[dim];

            for (long outer = 0; outer < outerSize; outer++)
            {
                for (long inner = 0; inner < innerSize; inner++)
                {
                    var srcIdx = outer * _strides[dim > 0 ? dim - 1 : 0] * (dim > 0 ? _shape[dim] : 1) +
                                 index * innerSize + inner;
                    if (dim == 0) srcIdx = index * innerSize + inner;
                    else srcIdx = outer * _shape[dim] * innerSize + index * innerSize + inner;

                    var dstIdx = outer * innerSize + inner;
                    if (srcIdx < _data.Length && dstIdx < newData.Length)
                        newData[dstIdx] = _data[srcIdx];
                }
            }

            return new Tensor(newData, newShape, _requiresGrad, _device);
        }

        #endregion

        #region Shape Operations

        /// <summary>Reshape tensor to new shape</summary>
        public Tensor Reshape(params long[] newShape)
        {
            // Handle -1 in shape (infer dimension)
            var inferIdx = Array.IndexOf(newShape, -1L);
            if (inferIdx >= 0)
            {
                var knownSize = newShape.Where(s => s != -1).Aggregate(1L, (a, b) => a * b);
                newShape[inferIdx] = NumElements / knownSize;
            }

            var newSize = newShape.Aggregate(1L, (a, b) => a * b);
            if (newSize != NumElements)
                throw new ArgumentException($"Cannot reshape tensor of size {NumElements} to shape [{string.Join(", ", newShape)}]");

            return new Tensor(_data.ToArray(), newShape, _requiresGrad, _device);
        }

        /// <summary>View tensor with new shape (shares data)</summary>
        public Tensor View(params long[] newShape)
        {
            // Handle -1 in shape
            var inferIdx = Array.IndexOf(newShape, -1L);
            if (inferIdx >= 0)
            {
                var knownSize = newShape.Where(s => s != -1).Aggregate(1L, (a, b) => a * b);
                newShape[inferIdx] = NumElements / knownSize;
            }

            var newSize = newShape.Aggregate(1L, (a, b) => a * b);
            if (newSize != NumElements)
                throw new ArgumentException($"Cannot view tensor of size {NumElements} as shape [{string.Join(", ", newShape)}]");

            // For now, copy data (true view would share storage)
            return new Tensor(_data, newShape, _requiresGrad, _device);
        }

        /// <summary>Flatten tensor to 1D</summary>
        public Tensor Flatten(int startDim = 0, int endDim = -1)
        {
            if (endDim < 0) endDim += Dimensions;
            if (startDim < 0) startDim += Dimensions;

            if (startDim == 0 && endDim == Dimensions - 1)
                return Reshape(NumElements);

            var newShape = new List<long>();
            for (int i = 0; i < startDim; i++)
                newShape.Add(_shape[i]);

            long flatSize = 1;
            for (int i = startDim; i <= endDim; i++)
                flatSize *= _shape[i];
            newShape.Add(flatSize);

            for (int i = endDim + 1; i < Dimensions; i++)
                newShape.Add(_shape[i]);

            return Reshape(newShape.ToArray());
        }

        /// <summary>Add dimension at specified position</summary>
        public Tensor Unsqueeze(int dim)
        {
            if (dim < 0) dim += Dimensions + 1;
            if (dim < 0 || dim > Dimensions)
                throw new ArgumentOutOfRangeException(nameof(dim));

            var newShape = new long[Dimensions + 1];
            for (int i = 0; i < dim; i++)
                newShape[i] = _shape[i];
            newShape[dim] = 1;
            for (int i = dim; i < Dimensions; i++)
                newShape[i + 1] = _shape[i];

            return Reshape(newShape);
        }

        /// <summary>Remove dimensions of size 1</summary>
        public Tensor Squeeze(int? dim = null)
        {
            if (dim.HasValue)
            {
                var d = dim.Value < 0 ? dim.Value + Dimensions : dim.Value;
                if (d < 0 || d >= Dimensions)
                    throw new ArgumentOutOfRangeException(nameof(dim));

                if (_shape[d] != 1)
                    return this.Clone();

                var newShape = _shape.Where((_, i) => i != d).ToArray();
                if (newShape.Length == 0) newShape = Array.Empty<long>();
                return Reshape(newShape);
            }
            else
            {
                var newShape = _shape.Where(s => s != 1).ToArray();
                if (newShape.Length == 0) newShape = Array.Empty<long>();
                return Reshape(newShape);
            }
        }

        /// <summary>Transpose tensor (swap two dimensions)</summary>
        public Tensor Transpose(int dim0, int dim1)
        {
            if (dim0 < 0) dim0 += Dimensions;
            if (dim1 < 0) dim1 += Dimensions;

            if (dim0 < 0 || dim0 >= Dimensions || dim1 < 0 || dim1 >= Dimensions)
                throw new ArgumentOutOfRangeException();

            if (dim0 == dim1) return Clone();

            var newShape = _shape.ToArray();
            (newShape[dim0], newShape[dim1]) = (newShape[dim1], newShape[dim0]);

            var newData = new double[_data.Length];
            var newStrides = ComputeStrides(newShape);

            // Compute transposed data
            for (long i = 0; i < NumElements; i++)
            {
                var oldIndices = GetIndices(i, _shape);
                (oldIndices[dim0], oldIndices[dim1]) = (oldIndices[dim1], oldIndices[dim0]);
                var newIndex = GetFlatIndexFromIndices(oldIndices, newStrides);
                newData[newIndex] = _data[i];
            }

            return new Tensor(newData, newShape, _requiresGrad, _device);
        }

        /// <summary>Transpose 2D tensor (matrix)</summary>
        public Tensor T()
        {
            if (Dimensions != 2)
                throw new InvalidOperationException("T() only works on 2D tensors");
            return Transpose(0, 1);
        }

        /// <summary>Permute dimensions</summary>
        public Tensor Permute(params int[] dims)
        {
            if (dims.Length != Dimensions)
                throw new ArgumentException("Number of dimensions must match");

            var newShape = dims.Select(d => _shape[d < 0 ? d + Dimensions : d]).ToArray();
            var newData = new double[_data.Length];
            var newStrides = ComputeStrides(newShape);

            for (long i = 0; i < NumElements; i++)
            {
                var oldIndices = GetIndices(i, _shape);
                var newIndices = dims.Select(d => oldIndices[d < 0 ? d + Dimensions : d]).ToArray();
                var newIndex = GetFlatIndexFromIndices(newIndices, newStrides);
                newData[newIndex] = _data[i];
            }

            return new Tensor(newData, newShape, _requiresGrad, _device);
        }

        /// <summary>Clone tensor</summary>
        public Tensor Clone()
        {
            return new Tensor(_data.ToArray(), _shape.ToArray(), _requiresGrad, _device);
        }

        /// <summary>Detach from computation graph</summary>
        public Tensor Detach()
        {
            return new Tensor(_data.ToArray(), _shape.ToArray(), false, _device);
        }

        /// <summary>Contiguous copy of tensor</summary>
        public Tensor Contiguous()
        {
            return Clone();
        }

        #endregion

        #region Element-wise Operations

        // Parallelization threshold for element-wise operations
        private const int ParallelThreshold = 4096;

        /// <summary>Apply function to each element (auto-parallelized for large tensors)</summary>
        public Tensor Apply(Func<double, double> func)
        {
            var result = new double[_data.Length];

            // Use parallel processing for large tensors
            if (_data.Length >= ParallelThreshold)
            {
                System.Threading.Tasks.Parallel.For(0, _data.Length, i =>
                {
                    result[i] = func(_data[i]);
                });
            }
            else
            {
                for (int i = 0; i < _data.Length; i++)
                    result[i] = func(_data[i]);
            }

            return new Tensor(result, _shape.ToArray(), _requiresGrad, _device);
        }

        /// <summary>Apply function with another tensor (element-wise, auto-parallelized)</summary>
        public Tensor Apply(Tensor other, Func<double, double, double> func)
        {
            var (broadcastedThis, broadcastedOther, resultShape) = BroadcastTensors(this, other);
            var result = new double[broadcastedThis._data.Length];
            var thisData = broadcastedThis._data;
            var otherData = broadcastedOther._data;

            // Use parallel processing for large tensors
            if (result.Length >= ParallelThreshold)
            {
                System.Threading.Tasks.Parallel.For(0, result.Length, i =>
                {
                    result[i] = func(thisData[i], otherData[i]);
                });
            }
            else
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] = func(thisData[i], otherData[i]);
            }

            return new Tensor(result, resultShape, _requiresGrad || other._requiresGrad, _device);
        }

        internal delegate void UnarySimdOp(ReadOnlySpan<double> src, Span<double> dst);

        internal delegate void BinarySimdOp(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> dst);

        internal Tensor ApplySimd(UnarySimdOp simdOp)
        {
            var result = new double[_data.Length];
            simdOp(_data.AsSpan(), result.AsSpan());
            return new Tensor(result, _shape.ToArray(), _requiresGrad, _device);
        }

        internal Tensor ApplySimd(Tensor other, BinarySimdOp simdOp)
        {
            var (broadcastedThis, broadcastedOther, resultShape) = BroadcastTensors(this, other);
            var result = new double[broadcastedThis._data.Length];
            simdOp(broadcastedThis._data.AsSpan(), broadcastedOther._data.AsSpan(), result.AsSpan());
            return new Tensor(result, resultShape, _requiresGrad || other._requiresGrad, _device);
        }

        // Arithmetic operations
        /// <summary>Public API</summary>
        public static Tensor operator +(Tensor a, Tensor b) => a.Add(b);
        /// <summary>Public API</summary>
        public static Tensor operator +(Tensor a, double b) => a.Add(b);
        /// <summary>Public API</summary>
        public static Tensor operator +(double a, Tensor b) => b.Add(a);

        /// <summary>Public API</summary>
        public static Tensor operator -(Tensor a, Tensor b) => a.Sub(b);
        /// <summary>Public API</summary>
        public static Tensor operator -(Tensor a, double b) => a.Sub(b);
        /// <summary>Public API</summary>
        public static Tensor operator -(double a, Tensor b) => Tensor.Full(b.Shape, a).Sub(b);
        /// <summary>Public API</summary>
        public static Tensor operator -(Tensor a) => a.Neg();

        /// <summary>Public API</summary>
        public static Tensor operator *(Tensor a, Tensor b) => a.Mul(b);
        /// <summary>Public API</summary>
        public static Tensor operator *(Tensor a, double b) => a.Mul(b);
        /// <summary>Public API</summary>
        public static Tensor operator *(double a, Tensor b) => b.Mul(a);

        /// <summary>Public API</summary>
        public static Tensor operator /(Tensor a, Tensor b) => a.Div(b);
        /// <summary>Public API</summary>
        public static Tensor operator /(Tensor a, double b) => a.Div(b);
        /// <summary>Public API</summary>
        public static Tensor operator /(double a, Tensor b) => Tensor.Full(b.Shape, a).Div(b);

        /// <summary>Element-wise addition</summary>
        public Tensor Add(Tensor other)
        {
            if (_requiresGrad || other._requiresGrad)
            {
                var result = Apply(other, (a, b) => a + b);
                result._gradFn = new AddBackward(this, other);
                result._isLeaf = false;
                return result;
            }
            return Apply(other, (a, b) => a + b);
        }

        /// <summary>Add scalar</summary>
        public Tensor Add(double scalar) => Apply(x => x + scalar);

        /// <summary>Element-wise subtraction</summary>
        public Tensor Sub(Tensor other)
        {
            if (_requiresGrad || other._requiresGrad)
            {
                var result = Apply(other, (a, b) => a - b);
                result._gradFn = new SubBackward(this, other);
                result._isLeaf = false;
                return result;
            }
            return Apply(other, (a, b) => a - b);
        }

        /// <summary>Subtract scalar</summary>
        public Tensor Sub(double scalar) => Apply(x => x - scalar);

        /// <summary>Element-wise multiplication</summary>
        public Tensor Mul(Tensor other)
        {
            if (_requiresGrad || other._requiresGrad)
            {
                var result = Apply(other, (a, b) => a * b);
                result._gradFn = new MulBackward(this, other);
                result._isLeaf = false;
                return result;
            }
            return Apply(other, (a, b) => a * b);
        }

        /// <summary>Multiply by scalar</summary>
        public Tensor Mul(double scalar)
        {
            if (_requiresGrad)
            {
                var result = Apply(x => x * scalar);
                result._gradFn = new MulScalarBackward(this, scalar);
                result._isLeaf = false;
                return result;
            }
            return Apply(x => x * scalar);
        }

        /// <summary>Element-wise division</summary>
        public Tensor Div(Tensor other)
        {
            if (_requiresGrad || other._requiresGrad)
            {
                var result = Apply(other, (a, b) => a / b);
                result._gradFn = new DivBackward(this, other);
                result._isLeaf = false;
                return result;
            }
            return Apply(other, (a, b) => a / b);
        }

        /// <summary>Divide by scalar</summary>
        public Tensor Div(double scalar) => Apply(x => x / scalar);

        /// <summary>Negation</summary>
        public Tensor Neg()
        {
            if (_requiresGrad)
            {
                var result = Apply(x => -x);
                result._gradFn = new NegBackward(this);
                result._isLeaf = false;
                return result;
            }
            return Apply(x => -x);
        }

        /// <summary>Element-wise power</summary>
        public Tensor Pow(double exponent) => Apply(x => Math.Pow(x, exponent));

        /// <summary>Element-wise power with tensor exponent</summary>
        public Tensor Pow(Tensor exponent) => Apply(exponent, Math.Pow);

        /// <summary>Square root</summary>
        public Tensor Sqrt() => Apply(Math.Sqrt);

        /// <summary>Square</summary>
        public Tensor Square() => Apply(x => x * x);

        /// <summary>Absolute value</summary>
        public Tensor Abs() => Apply(Math.Abs);

        /// <summary>Sign</summary>
        public Tensor Sign() => Apply(x => (double)Math.Sign(x));

        /// <summary>Exponential</summary>
        public Tensor Exp()
        {
            if (_requiresGrad)
            {
                var result = Apply(Math.Exp);
                result._gradFn = new ExpBackward(this, result);
                result._isLeaf = false;
                return result;
            }
            return Apply(Math.Exp);
        }

        /// <summary>Natural logarithm</summary>
        public Tensor Log()
        {
            if (_requiresGrad)
            {
                var result = Apply((double x) => Math.Log(x));
                result._gradFn = new LogBackward(this);
                result._isLeaf = false;
                return result;
            }
            return Apply((double x) => Math.Log(x));
        }

        /// <summary>Log base 10</summary>
        public Tensor Log10() => Apply(Math.Log10);

        /// <summary>Log base 2</summary>
        public Tensor Log2() => Apply(x => Math.Log(x) / Math.Log(2));

        // Trigonometric functions
        /// <summary>Public API</summary>
        public Tensor Sin() => Apply(Math.Sin);
        /// <summary>Public API</summary>
        public Tensor Cos() => Apply(Math.Cos);
        /// <summary>Public API</summary>
        public Tensor Tan() => Apply(Math.Tan);
        /// <summary>Public API</summary>
        public Tensor Asin() => Apply(Math.Asin);
        /// <summary>Public API</summary>
        public Tensor Acos() => Apply(Math.Acos);
        /// <summary>Public API</summary>
        public Tensor Atan() => Apply(Math.Atan);
        /// <summary>Public API</summary>
        public Tensor Sinh() => Apply(Math.Sinh);
        /// <summary>Public API</summary>
        public Tensor Cosh() => Apply(Math.Cosh);
        /// <summary>Public API</summary>
        public Tensor Tanh()
        {
            if (_requiresGrad)
            {
                var result = Apply(Math.Tanh);
                result._gradFn = new TanhBackward(this, result);
                result._isLeaf = false;
                return result;
            }
            return Apply(Math.Tanh);
        }

        // Rounding functions
        /// <summary>Public API</summary>
        public Tensor Floor() => Apply(Math.Floor);
        /// <summary>Public API</summary>
        public Tensor Ceil() => Apply(Math.Ceiling);
        /// <summary>Public API</summary>
        public Tensor Round() => Apply(Math.Round);
        /// <summary>Public API</summary>
        public Tensor Trunc() => Apply(Math.Truncate);

        // Clamping
        /// <summary>Public API</summary>
        public Tensor Clamp(double min, double max) => Apply(x => Math.Max(min, Math.Min(max, x)));
        /// <summary>Public API</summary>
        public Tensor ClampMin(double min) => Apply(x => Math.Max(min, x));
        /// <summary>Public API</summary>
        public Tensor ClampMax(double max) => Apply(x => Math.Min(max, x));

        #endregion

        #region Reduction Operations

        /// <summary>Sum all elements (SIMD-optimized)</summary>
        public Tensor Sum()
        {
            // Use SIMD-optimized sum
            var sum = CpuTensorPrimitives.Sum(_data.AsSpan());
            var result = new Tensor(sum, _requiresGrad, _device);
            if (_requiresGrad)
            {
                result._gradFn = new SumBackward(this);
                result._isLeaf = false;
            }
            return result;
        }

        /// <summary>Sum along dimension</summary>
        public Tensor Sum(int dim, bool keepDim = false)
        {
            if (dim < 0) dim += Dimensions;

            var newShape = _shape.ToArray();
            var resultShape = keepDim
                ? _shape.Select((s, i) => i == dim ? 1L : s).ToArray()
                : _shape.Where((_, i) => i != dim).ToArray();

            if (resultShape.Length == 0) resultShape = Array.Empty<long>();

            var resultSize = resultShape.Length == 0 ? 1 : resultShape.Aggregate(1L, (a, b) => a * b);
            var resultData = new double[resultSize];

            var dimSize = _shape[dim];
            var outerSize = dim == 0 ? 1 : _shape.Take(dim).Aggregate(1L, (a, b) => a * b);
            var innerSize = dim == Dimensions - 1 ? 1 : _shape.Skip(dim + 1).Aggregate(1L, (a, b) => a * b);

            for (long outer = 0; outer < outerSize; outer++)
            {
                for (long inner = 0; inner < innerSize; inner++)
                {
                    double sum = 0;
                    for (long d = 0; d < dimSize; d++)
                    {
                        var srcIdx = outer * dimSize * innerSize + d * innerSize + inner;
                        sum += _data[srcIdx];
                    }
                    var dstIdx = outer * innerSize + inner;
                    resultData[dstIdx] = sum;
                }
            }

            return new Tensor(resultData, resultShape, _requiresGrad, _device);
        }

        /// <summary>Mean of all elements</summary>
        public Tensor Mean() => Sum() / NumElements;

        /// <summary>Mean along dimension</summary>
        public Tensor Mean(int dim, bool keepDim = false)
        {
            if (dim < 0) dim += Dimensions;
            return Sum(dim, keepDim) / _shape[dim];
        }

        /// <summary>Variance of all elements</summary>
        public Tensor Var(bool unbiased = true)
        {
            var mean = Mean();
            var diff = this - mean.ToScalar();
            var sqDiff = diff.Square();
            var n = unbiased ? NumElements - 1 : NumElements;
            return sqDiff.Sum() / n;
        }

        /// <summary>Variance along dimension</summary>
        public Tensor Var(int dim, bool keepDim = false, bool unbiased = true)
        {
            if (dim < 0) dim += NDim;
            var mean = Mean(dim, keepDim: true);
            var diff = Sub(mean);
            var sqDiff = diff.Square();
            var n = unbiased ? _shape[dim] - 1 : _shape[dim];
            if (n <= 0) n = 1;
            return sqDiff.Sum(dim, keepDim).Div(n);
        }

        /// <summary>Standard deviation</summary>
        public Tensor Std(bool unbiased = true) => Var(unbiased).Sqrt();

        /// <summary>Standard deviation along dimension</summary>
        public Tensor Std(int dim, bool keepDim = false, bool unbiased = true) => Var(dim, keepDim, unbiased).Sqrt();

        /// <summary>Product of all elements</summary>
        public Tensor Prod() => new Tensor(_data.Aggregate(1.0, (a, b) => a * b), _requiresGrad, _device);

        /// <summary>Maximum element (SIMD-optimized)</summary>
        public Tensor Max() => new Tensor(CpuTensorPrimitives.Max(_data.AsSpan()), false, _device);

        /// <summary>Maximum along dimension</summary>
        public (Tensor values, Tensor indices) Max(int dim, bool keepDim = false)
        {
            return ReduceWithIndices(dim, keepDim, (a, b) => a > b);
        }

        /// <summary>Minimum element (SIMD-optimized)</summary>
        public Tensor Min() => new Tensor(CpuTensorPrimitives.Min(_data.AsSpan()), false, _device);

        /// <summary>Minimum along dimension</summary>
        public (Tensor values, Tensor indices) Min(int dim, bool keepDim = false)
        {
            return ReduceWithIndices(dim, keepDim, (a, b) => a < b);
        }

        /// <summary>Argument of maximum element</summary>
        public Tensor Argmax()
        {
            long maxIdx = 0;
            double maxVal = _data[0];
            for (long i = 1; i < _data.Length; i++)
            {
                if (_data[i] > maxVal)
                {
                    maxVal = _data[i];
                    maxIdx = i;
                }
            }
            return new Tensor((double)maxIdx, false, _device);
        }

        /// <summary>Argument of minimum element</summary>
        public Tensor Argmin()
        {
            long minIdx = 0;
            double minVal = _data[0];
            for (long i = 1; i < _data.Length; i++)
            {
                if (_data[i] < minVal)
                {
                    minVal = _data[i];
                    minIdx = i;
                }
            }
            return new Tensor((double)minIdx, false, _device);
        }

        /// <summary>L2 norm (SIMD-optimized for p=2)</summary>
        public Tensor Norm(double p = 2.0)
        {
            if (p == 2.0)
            {
                // Use SIMD-optimized L2 norm
                var norm = CpuTensorPrimitives.Norm2(_data.AsSpan());
                return new Tensor(norm, false, _device);
            }
            if (p == 1.0)
                return Abs().Sum();
            if (double.IsPositiveInfinity(p))
                return Abs().Max();

            return Abs().Pow(p).Sum().Pow(1.0 / p);
        }

        #endregion

        #region Comparison Operations

        /// <summary>Public API</summary>
        public Tensor Eq(Tensor other) => Apply(other, (a, b) => a == b ? 1.0 : 0.0);
        /// <summary>Public API</summary>
        public Tensor Eq(double value) => Apply(x => x == value ? 1.0 : 0.0);

        /// <summary>Public API</summary>
        public Tensor Ne(Tensor other) => Apply(other, (a, b) => a != b ? 1.0 : 0.0);
        /// <summary>Public API</summary>
        public Tensor Ne(double value) => Apply(x => x != value ? 1.0 : 0.0);

        /// <summary>Public API</summary>
        public Tensor Lt(Tensor other) => Apply(other, (a, b) => a < b ? 1.0 : 0.0);
        /// <summary>Public API</summary>
        public Tensor Lt(double value) => Apply(x => x < value ? 1.0 : 0.0);

        /// <summary>Public API</summary>
        public Tensor Le(Tensor other) => Apply(other, (a, b) => a <= b ? 1.0 : 0.0);
        /// <summary>Public API</summary>
        public Tensor Le(double value) => Apply(x => x <= value ? 1.0 : 0.0);

        /// <summary>Public API</summary>
        public Tensor Gt(Tensor other) => Apply(other, (a, b) => a > b ? 1.0 : 0.0);
        /// <summary>Public API</summary>
        public Tensor Gt(double value) => Apply(x => x > value ? 1.0 : 0.0);

        /// <summary>Public API</summary>
        public Tensor Ge(Tensor other) => Apply(other, (a, b) => a >= b ? 1.0 : 0.0);
        /// <summary>Public API</summary>
        public Tensor Ge(double value) => Apply(x => x >= value ? 1.0 : 0.0);

        /// <summary>Check if all elements are true (non-zero)</summary>
        public bool All() => _data.All(x => x != 0);

        /// <summary>Check if any element is true (non-zero)</summary>
        public bool Any() => _data.Any(x => x != 0);

        #endregion

        #region Autograd

        /// <summary>Set requires_grad flag</summary>
        public Tensor RequiresGrad_(bool requiresGrad = true)
        {
            _requiresGrad = requiresGrad;
            return this;
        }

        /// <summary>Get gradient</summary>
        public Tensor? GetGrad() => _grad;

        /// <summary>Zero gradients</summary>
        public void ZeroGrad()
        {
            if (_grad != null)
            {
                for (int i = 0; i < _grad._data.Length; i++)
                    _grad._data[i] = 0;
            }
        }

        /// <summary>Backward pass - compute gradients</summary>
        public void Backward(Tensor? gradient = null)
        {
            if (!_requiresGrad)
                throw new InvalidOperationException("Cannot call backward on tensor that doesn't require grad");

            gradient ??= Ones(_shape);

            // Accumulate gradient
            if (_grad == null)
                _grad = gradient.Clone();
            else
                _grad = _grad.Add(gradient);

            // Propagate through computation graph
            _gradFn?.Backward(gradient);
        }

        internal void AccumulateGrad(Tensor gradient)
        {
            if (!_requiresGrad) return;

            // Handle shape mismatch from broadcasting
            var grad = gradient;
            if (!_shape.SequenceEqual(gradient._shape))
            {
                grad = SumToShape(gradient, _shape);
            }

            if (_grad == null)
                _grad = grad.Clone();
            else
            {
                for (int i = 0; i < _grad._data.Length; i++)
                    _grad._data[i] += grad._data[i];
            }
        }

        #endregion

        #region Linear Algebra (Continued in LinearAlgebra.cs)

        /// <summary>Matrix multiplication</summary>
        public Tensor MatMul(Tensor other)
        {
            return TensorOps.MatMul(this, other);
        }

        /// <summary>Dot product</summary>
        public Tensor Dot(Tensor other)
        {
            return TensorOps.Dot(this, other);
        }

        /// <summary>Outer product</summary>
        public Tensor Outer(Tensor other)
        {
            return TensorOps.Outer(this, other);
        }

        #endregion

        #region Utility Methods

        private static long[] ComputeStrides(long[] shape)
        {
            if (shape.Length == 0) return Array.Empty<long>();

            var strides = new long[shape.Length];
            strides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
                strides[i] = strides[i + 1] * shape[i + 1];
            return strides;
        }

        private long GetFlatIndex(long[] indices)
        {
            if (indices.Length != Dimensions)
                throw new ArgumentException($"Expected {Dimensions} indices, got {indices.Length}");

            long flatIndex = 0;
            for (int i = 0; i < indices.Length; i++)
            {
                var idx = indices[i] < 0 ? indices[i] + _shape[i] : indices[i];
                if (idx < 0 || idx >= _shape[i])
                    throw new IndexOutOfRangeException($"Index {indices[i]} out of range for dimension {i} of size {_shape[i]}");
                flatIndex += idx * _strides[i];
            }
            return flatIndex;
        }

        private static long GetFlatIndexFromIndices(long[] indices, long[] strides)
        {
            long flatIndex = 0;
            for (int i = 0; i < indices.Length; i++)
                flatIndex += indices[i] * strides[i];
            return flatIndex;
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

        private static (Tensor, Tensor, long[]) BroadcastTensors(Tensor a, Tensor b)
        {
            var resultShape = BroadcastShapes(a._shape, b._shape);
            var broadcastedA = BroadcastTo(a, resultShape);
            var broadcastedB = BroadcastTo(b, resultShape);
            return (broadcastedA, broadcastedB, resultShape);
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
                    throw new ArgumentException($"Cannot broadcast shapes [{string.Join(", ", shape1)}] and [{string.Join(", ", shape2)}]");

                result[maxDims - 1 - i] = Math.Max(d1, d2);
            }

            return result;
        }

        /// <summary>Public API</summary>
        public static Tensor BroadcastTo(Tensor tensor, long[] targetShape)
        {
            if (tensor._shape.SequenceEqual(targetShape))
                return tensor;

            var targetSize = targetShape.Aggregate(1L, (a, b) => a * b);
            var result = new double[targetSize];
            var targetStrides = ComputeStrides(targetShape);

            // Compute broadcast strides
            var broadcastStrides = new long[targetShape.Length];
            var shapeOffset = targetShape.Length - tensor._shape.Length;

            for (int i = 0; i < targetShape.Length; i++)
            {
                var srcDim = i - shapeOffset;
                if (srcDim >= 0 && srcDim < tensor._shape.Length && tensor._shape[srcDim] > 1)
                    broadcastStrides[i] = tensor._strides[srcDim];
                else
                    broadcastStrides[i] = 0;
            }

            for (long i = 0; i < targetSize; i++)
            {
                var targetIndices = GetIndices(i, targetShape);
                long srcIndex = 0;
                for (int j = 0; j < targetShape.Length; j++)
                    srcIndex += targetIndices[j] * broadcastStrides[j];
                result[i] = tensor._data[srcIndex];
            }

            return new Tensor(result, targetShape, tensor._requiresGrad, tensor._device);
        }

        private static Tensor SumToShape(Tensor tensor, long[] targetShape)
        {
            var result = tensor;

            // Sum over extra leading dimensions
            while (result.Dimensions > targetShape.Length)
            {
                result = result.Sum(0);
            }

            // Sum over dimensions that were broadcast
            for (int i = 0; i < targetShape.Length; i++)
            {
                if (result._shape[i] != targetShape[i] && targetShape[i] == 1)
                {
                    result = result.Sum(i, keepDim: true);
                }
            }

            return result;
        }

        private static void CopySlice(double[] src, long[] srcShape, long[] srcStrides,
            double[] dst, long[] dstShape, int dim, long start, long end)
        {
            var dstStrides = ComputeStrides(dstShape);
            var dstSize = dstShape.Aggregate(1L, (a, b) => a * b);

            for (long i = 0; i < dstSize; i++)
            {
                var dstIndices = GetIndices(i, dstShape);
                var srcIndices = dstIndices.ToArray();
                srcIndices[dim] += start;

                long srcIdx = 0;
                for (int j = 0; j < srcIndices.Length; j++)
                    srcIdx += srcIndices[j] * srcStrides[j];

                dst[i] = src[srcIdx];
            }
        }

        private (Tensor values, Tensor indices) ReduceWithIndices(int dim, bool keepDim, Func<double, double, bool> comparator)
        {
            if (dim < 0) dim += Dimensions;

            var resultShape = keepDim
                ? _shape.Select((s, i) => i == dim ? 1L : s).ToArray()
                : _shape.Where((_, i) => i != dim).ToArray();

            if (resultShape.Length == 0) resultShape = Array.Empty<long>();

            var resultSize = resultShape.Length == 0 ? 1 : resultShape.Aggregate(1L, (a, b) => a * b);
            var values = new double[resultSize];
            var indices = new double[resultSize];

            var dimSize = _shape[dim];
            var outerSize = dim == 0 ? 1 : _shape.Take(dim).Aggregate(1L, (a, b) => a * b);
            var innerSize = dim == Dimensions - 1 ? 1 : _shape.Skip(dim + 1).Aggregate(1L, (a, b) => a * b);

            for (long outer = 0; outer < outerSize; outer++)
            {
                for (long inner = 0; inner < innerSize; inner++)
                {
                    long bestIdx = 0;
                    double bestVal = _data[outer * dimSize * innerSize + inner];

                    for (long d = 1; d < dimSize; d++)
                    {
                        var srcIdx = outer * dimSize * innerSize + d * innerSize + inner;
                        if (comparator(_data[srcIdx], bestVal))
                        {
                            bestVal = _data[srcIdx];
                            bestIdx = d;
                        }
                    }

                    var dstIdx = outer * innerSize + inner;
                    values[dstIdx] = bestVal;
                    indices[dstIdx] = bestIdx;
                }
            }

            return (new Tensor(values, resultShape, false, _device),
                    new Tensor(indices, resultShape, false, _device));
        }

        private static long[] GetArrayShape(Array array)
        {
            var shape = new List<long>();
            var current = array;
            while (current != null && current.Length > 0)
            {
                shape.Add(current.Length);
                var first = current.GetValue(0);
                current = first as Array;
            }
            return shape.ToArray();
        }

        private static double[] FlattenArray(Array array)
        {
            var result = new List<double>();
            FlattenArrayRecursive(array, result);
            return result.ToArray();
        }

        private static void FlattenArrayRecursive(Array array, List<double> result)
        {
            foreach (var item in array)
            {
                if (item is Array subArray)
                    FlattenArrayRecursive(subArray, result);
                else
                    result.Add(Convert.ToDouble(item));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("tensor(");

            if (Dimensions == 0)
            {
                sb.Append(_data[0]);
            }
            else if (Dimensions == 1 && NumElements <= 10)
            {
                sb.Append('[');
                sb.Append(string.Join(", ", _data.Select(x => x.ToString("G6"))));
                sb.Append(']');
            }
            else if (Dimensions == 2 && _shape[0] <= 6 && _shape[1] <= 6)
            {
                sb.Append('[');
                for (int i = 0; i < _shape[0]; i++)
                {
                    if (i > 0) sb.Append("\n        ");
                    sb.Append('[');
                    for (int j = 0; j < _shape[1]; j++)
                    {
                        if (j > 0) sb.Append(", ");
                        sb.Append(_data[i * _shape[1] + j].ToString("G6"));
                    }
                    sb.Append(']');
                    if (i < _shape[0] - 1) sb.Append(',');
                }
                sb.Append(']');
            }
            else
            {
                sb.Append($"<shape=[{string.Join(", ", _shape)}], {NumElements} elements>");
            }

            if (_requiresGrad)
                sb.Append(", requires_grad=True");
            if (_gradFn != null)
                sb.Append($", grad_fn=<{_gradFn.GetType().Name}>");

            sb.Append(')');
            return sb.ToString();
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _grad?.Dispose();
                _disposed = true;
            }
        }

        #endregion

        #region Missing Methods

        /// <summary>Convert to double array</summary>
        public double[] ToArray() => (double[])_data.Clone();

        /// <summary>ReLU activation</summary>
        public Tensor ReLU()
        {
            if (_requiresGrad)
            {
                var result = Apply(x => Math.Max(0, x));
                result._gradFn = new ReLUBackward(this);
                result._isLeaf = false;
                return result;
            }
            return Apply(x => Math.Max(0, x));
        }

        /// <summary>Leaky ReLU activation</summary>
        public Tensor LeakyReLU(double negativeSlope = 0.01)
        {
            if (_requiresGrad)
            {
                var result = Apply(x => x > 0 ? x : negativeSlope * x);
                result._gradFn = new LeakyReLUBackward(this, negativeSlope);
                result._isLeaf = false;
                return result;
            }
            return Apply(x => x > 0 ? x : negativeSlope * x);
        }

        /// <summary>Sigmoid activation</summary>
        public Tensor Sigmoid()
        {
            if (_requiresGrad)
            {
                var result = Apply(x => 1.0 / (1.0 + Math.Exp(-x)));
                result._gradFn = new SigmoidBackward(this, result);
                result._isLeaf = false;
                return result;
            }
            return Apply(x => 1.0 / (1.0 + Math.Exp(-x)));
        }

        /// <summary>Softmax along dimension</summary>
        public Tensor Softmax(int dim = -1)
        {
            if (dim < 0) dim += NDim;

            // Compute max for numerical stability
            var (maxVals, _) = Max(dim, keepDim: true);
            var shifted = Sub(maxVals);
            var expVals = shifted.Exp();
            var sumExp = expVals.Sum(dim, keepDim: true);
            var result = expVals.Div(sumExp);

            if (_requiresGrad)
            {
                result._gradFn = new SoftmaxBackward(this, result, dim);
                result._isLeaf = false;
            }
            return result;
        }

        /// <summary>Log softmax along dimension</summary>
        public Tensor LogSoftmax(int dim = -1)
        {
            if (dim < 0) dim += NDim;
            var (maxVals, _) = Max(dim, keepDim: true);
            var shifted = Sub(maxVals);
            var logSumExp = shifted.Exp().Sum(dim, keepDim: true).Log();
            return shifted.Sub(logSumExp);
        }

        /// <summary>Flip tensor along dimension</summary>
        public Tensor Flip(int dim)
        {
            if (dim < 0) dim += NDim;
            var result = Clone();
            var dimSize = (int)_shape[dim];
            var innerSize = dim == NDim - 1 ? 1 : _shape.Skip(dim + 1).Aggregate(1L, (a, b) => a * b);
            var outerSize = dim == 0 ? 1 : _shape.Take(dim).Aggregate(1L, (a, b) => a * b);

            for (long outer = 0; outer < outerSize; outer++)
            {
                for (long inner = 0; inner < innerSize; inner++)
                {
                    for (int d = 0; d < dimSize / 2; d++)
                    {
                        var idx1 = outer * dimSize * innerSize + d * innerSize + inner;
                        var idx2 = outer * dimSize * innerSize + (dimSize - 1 - d) * innerSize + inner;
                        var temp = result._data[idx1];
                        result._data[idx1] = result._data[idx2];
                        result._data[idx2] = temp;
                    }
                }
            }
            return result;
        }

        /// <summary>Subtract in place</summary>
        public void SubInPlace(Tensor other)
        {
            var otherBroadcast = BroadcastTo(other, _shape);
            for (int i = 0; i < _data.Length; i++)
                _data[i] -= otherBroadcast._data[i];
        }

        /// <summary>Add in place</summary>
        public void AddInPlace(Tensor other)
        {
            var otherBroadcast = BroadcastTo(other, _shape);
            for (int i = 0; i < _data.Length; i++)
                _data[i] += otherBroadcast._data[i];
        }

        /// <summary>Multiply in place</summary>
        public void MulInPlace(double scalar)
        {
            for (int i = 0; i < _data.Length; i++)
                _data[i] *= scalar;
        }

        /// <summary>Copy data from another tensor</summary>
        public void CopyFrom(Tensor src)
        {
            if (!_shape.SequenceEqual(src._shape))
                throw new ArgumentException("Shapes must match for CopyFrom");
            Array.Copy(src._data, _data, _data.Length);
        }

        /// <summary>Repeat tensor along dimensions</summary>
        public Tensor Repeat(long[] repeats)
        {
            if (repeats.Length != NDim)
                throw new ArgumentException("Repeats must have same length as tensor dimensions");

            var newShape = _shape.Zip(repeats, (s, r) => s * r).ToArray();
            var result = Zeros(newShape);

            // Simple implementation - can be optimized
            var resultStrides = ComputeStrides(newShape);
            for (long i = 0; i < result.NumElements; i++)
            {
                var resultIndices = GetIndices(i, newShape);
                var srcIndices = resultIndices.Zip(_shape, (ri, s) => ri % s).ToArray();
                long srcIdx = 0;
                for (int j = 0; j < _shape.Length; j++)
                    srcIdx += srcIndices[j] * _strides[j];
                result._data[i] = _data[srcIdx];
            }
            return result;
        }

        /// <summary>Apply function with index</summary>
        public Tensor Apply(Func<double, long, double> func)
        {
            var result = new double[_data.Length];
            for (long i = 0; i < _data.Length; i++)
                result[i] = func(_data[i], i);
            return new Tensor(result, _shape, false, _device);
        }

        /// <summary>Create uniform random tensor</summary>
        public static Tensor Uniform(long[] shape, double low = 0, double high = 1)
        {
            var random = new Random();
            var data = new double[shape.Aggregate(1L, (a, b) => a * b)];
            for (int i = 0; i < data.Length; i++)
                data[i] = low + random.NextDouble() * (high - low);
            return new Tensor(data, shape);
        }

        /// <summary>Element-wise maximum of two tensors</summary>
        public static Tensor Maximum(Tensor a, Tensor b)
        {
            var broadcastShape = BroadcastShapes(a._shape, b._shape);
            var aBroadcast = BroadcastTo(a, broadcastShape);
            var bBroadcast = BroadcastTo(b, broadcastShape);

            var result = new double[aBroadcast._data.Length];
            for (int i = 0; i < result.Length; i++)
                result[i] = Math.Max(aBroadcast._data[i], bBroadcast._data[i]);
            return new Tensor(result, broadcastShape);
        }

        /// <summary>Element-wise minimum of two tensors</summary>
        public static Tensor Minimum(Tensor a, Tensor b)
        {
            var broadcastShape = BroadcastShapes(a._shape, b._shape);
            var aBroadcast = BroadcastTo(a, broadcastShape);
            var bBroadcast = BroadcastTo(b, broadcastShape);

            var result = new double[aBroadcast._data.Length];
            for (int i = 0; i < result.Length; i++)
                result[i] = Math.Min(aBroadcast._data[i], bBroadcast._data[i]);
            return new Tensor(result, broadcastShape);
        }

        /// <summary>Create tensor from flat array with shape</summary>
        public static Tensor FromArray(double[] data, long[] shape, DType dtype = DType.Float64)
        {
            return new Tensor((double[])data.Clone(), shape, false, Device.CPU, dtype);
        }

        #endregion

        #region Diagnostics

        /// Check tensor health and return a report.
        /// Useful for debugging numerical issues during training.
        /// </summary>
        public TensorHealthReport CheckHealth()
        {
            return TensorHealthReport.Analyze(this);
        }

        /// Check if tensor contains any NaN or Inf values.
        /// </summary>
        public bool HasNanOrInf()
        {
            for (int i = 0; i < _data.Length; i++)
            {
                if (double.IsNaN(_data[i]) || double.IsInfinity(_data[i]))
                    return true;
            }
            return false;
        }

        /// Replace NaN and Inf values with specified replacements.
        /// </summary>
        public Tensor ReplaceNanInf(double nanReplacement = 0.0, double posInfReplacement = 1e10, double negInfReplacement = -1e10)
        {
            var result = new double[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                if (double.IsNaN(_data[i]))
                    result[i] = nanReplacement;
                else if (double.IsPositiveInfinity(_data[i]))
                    result[i] = posInfReplacement;
                else if (double.IsNegativeInfinity(_data[i]))
                    result[i] = negInfReplacement;
                else
                    result[i] = _data[i];
            }
            return new Tensor(result, _shape, false, _device, _dtype);
        }

        #endregion
    }

    /// Numerical health report for a tensor.
    /// Inspired by pmx_utils NumericalHealthReport.
    /// </summary>
    public class TensorHealthReport
    {
        /// <summary>Public API</summary>
        public long[] Shape { get; }
        /// <summary>Public API</summary>
        public long TotalElements { get; }
        /// <summary>Public API</summary>
        public int NanCount { get; }
        /// <summary>Public API</summary>
        public int PosInfCount { get; }
        /// <summary>Public API</summary>
        public int NegInfCount { get; }
        /// <summary>Public API</summary>
        public double Min { get; }
        /// <summary>Public API</summary>
        public double Max { get; }
        /// <summary>Public API</summary>
        public double Mean { get; }
        /// <summary>Public API</summary>
        public double Std { get; }
        /// <summary>Public API</summary>
        public double AbsMax { get; }
        /// <summary>Public API</summary>
        public bool IsHealthy { get; }
        /// <summary>Public API</summary>
        public string[] Issues { get; }

        private TensorHealthReport(
            long[] shape, long totalElements,
            int nanCount, int posInfCount, int negInfCount,
            double min, double max, double mean, double std, double absMax,
            bool isHealthy, string[] issues)
        {
            Shape = shape;
            TotalElements = totalElements;
            NanCount = nanCount;
            PosInfCount = posInfCount;
            NegInfCount = negInfCount;
            Min = min;
            Max = max;
            Mean = mean;
            Std = std;
            AbsMax = absMax;
            IsHealthy = isHealthy;
            Issues = issues;
        }

        /// Analyze a tensor and produce a health report.
        /// </summary>
        public static TensorHealthReport Analyze(Tensor tensor, double warningThreshold = 1e6)
        {
            var data = tensor.Data;
            var shape = tensor.Shape;
            long totalElements = data.Length;

            int nanCount = 0;
            int posInfCount = 0;
            int negInfCount = 0;
            double min = double.MaxValue;
            double max = double.MinValue;
            double sum = 0;
            double sumSq = 0;
            double absMax = 0;
            int validCount = 0;

            for (int i = 0; i < data.Length; i++)
            {
                double v = data[i];
                if (double.IsNaN(v))
                {
                    nanCount++;
                }
                else if (double.IsPositiveInfinity(v))
                {
                    posInfCount++;
                }
                else if (double.IsNegativeInfinity(v))
                {
                    negInfCount++;
                }
                else
                {
                    validCount++;
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sum += v;
                    sumSq += v * v;
                    if (Math.Abs(v) > absMax) absMax = Math.Abs(v);
                }
            }

            double mean = validCount > 0 ? sum / validCount : 0;
            double variance = validCount > 0 ? (sumSq / validCount) - (mean * mean) : 0;
            double std = Math.Sqrt(Math.Max(0, variance));

            if (validCount == 0)
            {
                min = 0;
                max = 0;
            }

            // Determine issues
            var issues = new List<string>();
            if (nanCount > 0)
                issues.Add($"Contains {nanCount} NaN values ({100.0 * nanCount / totalElements:F2}%)");
            if (posInfCount > 0)
                issues.Add($"Contains {posInfCount} +Inf values");
            if (negInfCount > 0)
                issues.Add($"Contains {negInfCount} -Inf values");
            if (absMax > warningThreshold)
                issues.Add($"Values exceed warning threshold: |max|={absMax:E3} > {warningThreshold:E0}");
            if (std == 0 && validCount > 1)
                issues.Add("Zero variance (all values identical)");
            if (validCount == 0)
                issues.Add("No valid (finite) values");

            bool isHealthy = nanCount == 0 && posInfCount == 0 && negInfCount == 0 && absMax <= warningThreshold;

            return new TensorHealthReport(
                shape, totalElements,
                nanCount, posInfCount, negInfCount,
                min, max, mean, std, absMax,
                isHealthy, issues.ToArray()
            );
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"TensorHealthReport:");
            sb.AppendLine($"  Shape: [{string.Join(", ", Shape)}]");
            sb.AppendLine($"  Elements: {TotalElements}");
            sb.AppendLine($"  Healthy: {IsHealthy}");
            sb.AppendLine($"  Min: {Min:G6}, Max: {Max:G6}, Mean: {Mean:G6}, Std: {Std:G6}");
            sb.AppendLine($"  AbsMax: {AbsMax:G6}");
            if (NanCount > 0 || PosInfCount > 0 || NegInfCount > 0)
                sb.AppendLine($"  NaN: {NanCount}, +Inf: {PosInfCount}, -Inf: {NegInfCount}");
            if (Issues.Length > 0)
            {
                sb.AppendLine("  Issues:");
                foreach (var issue in Issues)
                    sb.AppendLine($"    - {issue}");
            }
            return sb.ToString();
        }
    }
}