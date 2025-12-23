using System;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// A tensor stored in GPU memory for accelerated computation.
    ///
    /// GPU tensors can be 100-1000x faster than CPU tensors for large operations.
    /// Use GpuAccelerator.ToGpu() and GpuAccelerator.ToCpu() to transfer data.
    ///
    /// Example:
    /// <code>
    /// var gpu = new GpuAccelerator();
    /// var gpuTensor = gpu.ToGpu(data, shape);
    /// var result = gpu.MatMul(gpuTensor, weights);
    /// var (output, outShape) = gpu.ToCpu(result);
    /// </code>
    /// </summary>
    public class GpuTensor : IDisposable
    {
        private readonly Accelerator _accelerator;
        private MemoryBuffer1D<float, Stride1D.Dense> _buffer;
        private bool _disposed;

        /// <summary>
        /// Shape of the tensor
        /// </summary>
        public int[] Shape { get; }

        /// <summary>
        /// Total number of elements
        /// </summary>
        public int Size { get; }

        /// <summary>
        /// Number of dimensions
        /// </summary>
        public int NDim => Shape.Length;

        /// <summary>
        /// The underlying GPU memory buffer
        /// </summary>
        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer => _buffer;

        /// <summary>
        /// The accelerator this tensor is on
        /// </summary>
        internal Accelerator Accelerator => _accelerator;

        /// <summary>
        /// Creates a GPU tensor with the given shape
        /// </summary>
        internal GpuTensor(Accelerator accelerator, int[] shape)
        {
            _accelerator = accelerator;
            Shape = (int[])shape.Clone();
            Size = shape.Aggregate(1, (a, b) => a * b);
            _buffer = accelerator.Allocate1D<float>(Size);
        }

        /// <summary>
        /// Creates a GPU tensor from an existing buffer
        /// </summary>
        internal GpuTensor(Accelerator accelerator, int[] shape, MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            _accelerator = accelerator;
            Shape = (int[])shape.Clone();
            Size = shape.Aggregate(1, (a, b) => a * b);
            _buffer = buffer;
        }

        /// <summary>
        /// Create a GPU tensor from a float array
        /// </summary>
        public static GpuTensor FromArray(Accelerator accelerator, float[] data, params int[] shape)
        {
            var expectedSize = shape.Aggregate(1, (a, b) => a * b);
            if (data.Length != expectedSize)
                throw new ArgumentException($"Data length {data.Length} doesn't match shape {string.Join("x", shape)}");

            var gpuTensor = new GpuTensor(accelerator, shape);
            gpuTensor._buffer.CopyFromCPU(data);
            return gpuTensor;
        }

        /// <summary>
        /// Create a GPU tensor from double array (converts to float)
        /// </summary>
        public static GpuTensor FromDoubleArray(Accelerator accelerator, double[] data, params int[] shape)
        {
            var floatData = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                floatData[i] = (float)data[i];
            return FromArray(accelerator, floatData, shape);
        }

        /// <summary>
        /// Get data as a CPU float array
        /// </summary>
        public float[] ToArray()
        {
            ThrowIfDisposed();
            var data = new float[Size];
            _buffer.CopyToCPU(data);
            return data;
        }

        /// <summary>
        /// Create a GPU tensor filled with zeros
        /// </summary>
        public static GpuTensor Zeros(Accelerator accelerator, params int[] shape)
        {
            var tensor = new GpuTensor(accelerator, shape);
            tensor._buffer.MemSetToZero();
            return tensor;
        }

        /// <summary>
        /// Create a GPU tensor filled with ones
        /// </summary>
        public static GpuTensor Ones(Accelerator accelerator, params int[] shape)
        {
            var size = shape.Aggregate(1, (a, b) => a * b);
            var data = new float[size];
            Array.Fill(data, 1f);

            var tensor = new GpuTensor(accelerator, shape);
            tensor._buffer.CopyFromCPU(data);
            return tensor;
        }

        /// <summary>
        /// Create a GPU tensor with values from uniform distribution [0, 1)
        /// </summary>
        public static GpuTensor Random(Accelerator accelerator, params int[] shape)
        {
            var size = shape.Aggregate(1, (a, b) => a * b);
            var random = new Random();
            var data = new float[size];

            for (int i = 0; i < size; i++)
                data[i] = (float)random.NextDouble();

            var tensor = new GpuTensor(accelerator, shape);
            tensor._buffer.CopyFromCPU(data);
            return tensor;
        }

        /// <summary>
        /// Create a GPU tensor with values from normal distribution
        /// </summary>
        public static GpuTensor RandomNormal(Accelerator accelerator, float mean, float std, params int[] shape)
        {
            var size = shape.Aggregate(1, (a, b) => a * b);
            var random = new Random();
            var data = new float[size];

            // Box-Muller transform for normal distribution
            for (int i = 0; i < size; i += 2)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = random.NextDouble();
                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;

                data[i] = (float)(mean + std * radius * Math.Cos(theta));
                if (i + 1 < size)
                    data[i + 1] = (float)(mean + std * radius * Math.Sin(theta));
            }

            var tensor = new GpuTensor(accelerator, shape);
            tensor._buffer.CopyFromCPU(data);
            return tensor;
        }

        /// <summary>
        /// Create an identity matrix on GPU
        /// </summary>
        public static GpuTensor Eye(Accelerator accelerator, int n)
        {
            var data = new float[n * n];
            for (int i = 0; i < n; i++)
                data[i * n + i] = 1f;

            return FromArray(accelerator, data, n, n);
        }

        /// <summary>
        /// Create a tensor filled with a specific value
        /// </summary>
        public static GpuTensor Full(Accelerator accelerator, float value, params int[] shape)
        {
            var size = shape.Aggregate(1, (a, b) => a * b);
            var data = new float[size];
            Array.Fill(data, value);

            var tensor = new GpuTensor(accelerator, shape);
            tensor._buffer.CopyFromCPU(data);
            return tensor;
        }

        /// <summary>
        /// Create a range tensor [start, start+1, ..., end-1]
        /// </summary>
        public static GpuTensor Arange(Accelerator accelerator, int start, int end, int step = 1)
        {
            var size = (end - start + step - 1) / step;
            var data = new float[size];

            for (int i = 0, val = start; i < size; i++, val += step)
                data[i] = val;

            return FromArray(accelerator, data, size);
        }

        /// <summary>
        /// Reshape the tensor (returns a view with same data)
        /// </summary>
        public GpuTensor Reshape(params int[] newShape)
        {
            ThrowIfDisposed();

            // Handle -1 in shape (infer dimension)
            int inferDim = -1;
            int knownProduct = 1;

            for (int i = 0; i < newShape.Length; i++)
            {
                if (newShape[i] == -1)
                {
                    if (inferDim >= 0)
                        throw new ArgumentException("Can only have one -1 in shape");
                    inferDim = i;
                }
                else
                {
                    knownProduct *= newShape[i];
                }
            }

            if (inferDim >= 0)
            {
                newShape = (int[])newShape.Clone();
                newShape[inferDim] = Size / knownProduct;
            }

            var newSize = newShape.Aggregate(1, (a, b) => a * b);
            if (newSize != Size)
                throw new ArgumentException($"Cannot reshape tensor of size {Size} to shape [{string.Join(", ", newShape)}]");

            // Return new tensor sharing the same buffer
            return new GpuTensor(_accelerator, newShape, _buffer);
        }

        /// <summary>
        /// Flatten tensor to 1D
        /// </summary>
        public GpuTensor Flatten()
        {
            return Reshape(Size);
        }

        /// <summary>
        /// Add a dimension at the specified position
        /// </summary>
        public GpuTensor Unsqueeze(int dim)
        {
            if (dim < 0) dim = NDim + 1 + dim;
            if (dim < 0 || dim > NDim)
                throw new ArgumentException($"Invalid dimension {dim} for tensor with {NDim} dimensions");

            var newShape = new int[NDim + 1];
            for (int i = 0; i < dim; i++)
                newShape[i] = Shape[i];
            newShape[dim] = 1;
            for (int i = dim; i < NDim; i++)
                newShape[i + 1] = Shape[i];

            return Reshape(newShape);
        }

        /// <summary>
        /// Remove a dimension of size 1
        /// </summary>
        public GpuTensor Squeeze(int? dim = null)
        {
            int[] newShape;

            if (dim.HasValue)
            {
                var d = dim.Value < 0 ? NDim + dim.Value : dim.Value;
                if (Shape[d] != 1)
                    throw new ArgumentException($"Cannot squeeze dimension {d} with size {Shape[d]}");
                newShape = Shape.Where((_, i) => i != d).ToArray();
            }
            else
            {
                newShape = Shape.Where(s => s != 1).ToArray();
            }

            if (newShape.Length == 0)
                newShape = new[] { 1 };

            return Reshape(newShape);
        }

        /// <summary>
        /// Clone this tensor (deep copy)
        /// </summary>
        public GpuTensor Clone()
        {
            ThrowIfDisposed();

            var clone = new GpuTensor(_accelerator, Shape);
            _buffer.View.CopyTo(clone._buffer.View);
            _accelerator.Synchronize();
            return clone;
        }

        /// <summary>
        /// Copy data from another tensor
        /// </summary>
        public void CopyFrom(GpuTensor source)
        {
            ThrowIfDisposed();

            if (Size != source.Size)
                throw new ArgumentException("Source tensor has different size");

            source._buffer.View.CopyTo(_buffer.View);
            _accelerator.Synchronize();
        }

        /// <summary>
        /// Get string representation of tensor
        /// </summary>
        public override string ToString()
        {
            if (_disposed)
                return "GpuTensor (disposed)";

            return $"GpuTensor([{string.Join(", ", Shape)}], device={_accelerator.Name})";
        }

        /// <summary>
        /// Get detailed string with values (transfers to CPU)
        /// </summary>
        public string ToDetailedString(int maxElements = 100)
        {
            if (_disposed)
                return "GpuTensor (disposed)";

            var data = ToArray();
            var elements = data.Take(maxElements).Select(x => x.ToString("F4"));
            var suffix = Size > maxElements ? ", ..." : "";

            return $"GpuTensor([{string.Join(", ", Shape)}], device={_accelerator.Name})\n" +
                   $"[{string.Join(", ", elements)}{suffix}]";
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GpuTensor));
        }

        /// <summary>
        /// Dispose GPU memory
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _buffer.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer
        /// </summary>
        ~GpuTensor()
        {
            Dispose();
        }
    }
}
