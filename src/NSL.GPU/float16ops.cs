using System;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// FP16 (Half Precision) Operations for NSL.
    ///
    /// Benefits of FP16 on modern GPUs:
    /// - 2x memory savings compared to FP32
    /// - 2x bandwidth efficiency
    /// - Up to 8x faster on Tensor Cores (RTX 2000+)
    /// - Suitable for inference with minimal accuracy loss
    ///
    /// Architecture Support:
    /// - Pascal (SM 6.0+): FP16 storage, FP32 compute
    /// - Volta/Turing (SM 7.0+): Tensor Core FP16
    /// - Ampere (SM 8.0+): TF32 + enhanced FP16
    /// - Ada/Hopper (SM 8.9+): FP8 + FP16 improvements
    ///
    /// Note: ILGPU doesn't natively support Half type in kernels,
    /// so we use conversion utilities and packed operations.
    /// </summary>
    public class Float16Ops
    {
        private readonly Accelerator _accelerator;
        private readonly bool _hasTensorCores;

        // Kernels for FP16<->FP32 conversion
        private readonly Action<Index1D, ArrayView<float>, ArrayView<ushort>> _f32ToF16Kernel;
        private readonly Action<Index1D, ArrayView<ushort>, ArrayView<float>> _f16ToF32Kernel;

        /// <summary>Public API</summary>
        public Float16Ops(Accelerator accelerator, bool hasTensorCores = false)
        {
            _accelerator = accelerator;
            _hasTensorCores = hasTensorCores;

            _f32ToF16Kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<ushort>>(Float32ToFloat16Kernel);
            _f16ToF32Kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, ArrayView<float>>(Float16ToFloat32Kernel);
        }

        #region FP16 Tensor

        /// <summary>
        /// FP16 tensor stored as ushort (binary representation of half-precision float)
        /// </summary>
        public class Float16Tensor : IDisposable
        {
            /// <summary>Public API</summary>
            public MemoryBuffer1D<ushort, Stride1D.Dense> Buffer { get; }
            /// <summary>Public API</summary>
            public int[] Shape { get; }
            /// <summary>Public API</summary>
            public int Size { get; }
            private readonly Accelerator _accelerator;
            private bool _disposed;

            /// <summary>Public API</summary>
            public Float16Tensor(Accelerator accelerator, int[] shape)
            {
                _accelerator = accelerator;
                Shape = shape;
                Size = 1;
                foreach (var dim in shape) Size *= dim;
                Buffer = accelerator.Allocate1D<ushort>(Size);
            }

            /// <summary>Memory savings: FP16 uses half the memory of FP32</summary>
            public long MemoryBytes => Size * sizeof(ushort);

            /// <summary>Public API</summary>
            public void Dispose()
            {
                if (!_disposed)
                {
                    Buffer.Dispose();
                    _disposed = true;
                }
            }
        }

        #endregion

        #region Conversion Operations

        /// <summary>
        /// Convert FP32 tensor to FP16 for storage/bandwidth efficiency
        /// </summary>
        public Float16Tensor ToFloat16(GpuTensor tensor)
        {
            var result = new Float16Tensor(_accelerator, tensor.Shape);
            _f32ToF16Kernel(tensor.Size, tensor.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();
            return result;
        }

        /// <summary>
        /// Convert FP16 tensor back to FP32 for computation
        /// </summary>
        public GpuTensor ToFloat32(Float16Tensor tensor)
        {
            var result = new GpuTensor(_accelerator, tensor.Shape);
            _f16ToF32Kernel(tensor.Size, tensor.Buffer.View, result.Buffer.View);
            _accelerator.Synchronize();
            return result;
        }

        /// <summary>
        /// Create FP16 tensor directly from float array
        /// </summary>
        public Float16Tensor FromArray(float[] data, params int[] shape)
        {
            var hostData = new ushort[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                hostData[i] = FloatToHalf(data[i]);
            }

            var result = new Float16Tensor(_accelerator, shape);
            result.Buffer.CopyFromCPU(hostData);
            return result;
        }

        /// <summary>
        /// Extract FP16 tensor data as float array
        /// </summary>
        public float[] ToArray(Float16Tensor tensor)
        {
            var hostData = new ushort[tensor.Size];
            tensor.Buffer.CopyToCPU(hostData);

            var result = new float[tensor.Size];
            for (int i = 0; i < tensor.Size; i++)
            {
                result[i] = HalfToFloat(hostData[i]);
            }
            return result;
        }

        #endregion

        #region Mixed Precision Operations

        /// <summary>
        /// Mixed-precision matrix multiplication.
        /// Stores in FP16, computes in FP32 for accuracy.
        /// </summary>
        public GpuTensor MixedPrecisionMatMul(Float16Tensor a, Float16Tensor b, GpuKernels kernels)
        {
            // Convert to FP32 for computation
            using var aF32 = ToFloat32(a);
            using var bF32 = ToFloat32(b);

            // Compute in FP32
            return kernels.MatMul(aF32, bF32);
        }

        /// <summary>
        /// Perform operation in FP32 then convert result to FP16 for storage
        /// </summary>
        public Float16Tensor ComputeAndCompress(
            GpuTensor input,
            Func<GpuTensor, GpuTensor> operation)
        {
            using var result = operation(input);
            return ToFloat16(result);
        }

        #endregion

        #region GPU Kernels

        /// <summary>
        /// FP32 to FP16 conversion kernel
        /// Uses simplified conversion suitable for GPU execution
        /// </summary>
        private static void Float32ToFloat16Kernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<ushort> output)
        {
            float val = input[index];

            // Simplified conversion using scale factors
            // This approach avoids bit manipulation in GPU kernels
            ushort result;

            if (val == 0.0f)
            {
                result = 0;
            }
            else if (float.IsNaN(val))
            {
                result = 0x7E00; // NaN
            }
            else if (float.IsPositiveInfinity(val))
            {
                result = 0x7C00; // +Inf
            }
            else if (float.IsNegativeInfinity(val))
            {
                result = 0xFC00; // -Inf
            }
            else
            {
                // Clamp to FP16 range
                float absVal = XMath.Abs(val);
                if (absVal > 65504.0f) absVal = 65504.0f; // FP16 max
                if (absVal < 6.1035156e-5f) absVal = 0; // FP16 min denormal

                // Simple quantization (loses some precision but GPU compatible)
                int signBit = val < 0 ? 0x8000 : 0;

                // Approximate exponent and mantissa
                float log2Val = XMath.Log2(absVal);
                int exp = (int)XMath.Floor(log2Val) + 15;

                if (exp <= 0)
                {
                    result = (ushort)signBit;
                }
                else if (exp >= 31)
                {
                    result = (ushort)(signBit | 0x7C00);
                }
                else
                {
                    float mantissaF = absVal / XMath.Pow(2.0f, exp - 15) - 1.0f;
                    int mantissa = (int)(mantissaF * 1024.0f);
                    mantissa = XMath.Clamp(mantissa, 0, 1023);
                    result = (ushort)(signBit | (exp << 10) | mantissa);
                }
            }

            output[index] = result;
        }

        /// <summary>
        /// FP16 to FP32 conversion kernel
        /// </summary>
        private static void Float16ToFloat32Kernel(
            Index1D index,
            ArrayView<ushort> input,
            ArrayView<float> output)
        {
            ushort val = input[index];

            // Extract components
            int sign = (val >> 15) & 1;
            int exp = (val >> 10) & 0x1F;
            int mantissa = val & 0x3FF;

            float result;
            if (exp == 0)
            {
                if (mantissa == 0)
                {
                    result = sign == 1 ? -0.0f : 0.0f;
                }
                else
                {
                    // Denormalized
                    result = (mantissa / 1024.0f) * XMath.Pow(2.0f, -14);
                    if (sign == 1) result = -result;
                }
            }
            else if (exp == 31)
            {
                if (mantissa == 0)
                {
                    result = sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
                }
                else
                {
                    result = float.NaN;
                }
            }
            else
            {
                // Normalized
                result = (1.0f + mantissa / 1024.0f) * XMath.Pow(2.0f, exp - 15);
                if (sign == 1) result = -result;
            }

            output[index] = result;
        }

        #endregion

        #region CPU Helpers (for host-side operations)

        /// <summary>
        /// Convert float to half-precision (CPU side)
        /// </summary>
        public static ushort FloatToHalf(float value)
        {
            int bits = BitConverter.SingleToInt32Bits(value);
            int sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xFF) - 127 + 15;
            int mantissa = (bits >> 13) & 0x3FF;

            if (exp <= 0) return (ushort)sign;
            if (exp >= 31) return (ushort)(sign | 0x7C00);
            return (ushort)(sign | (exp << 10) | mantissa);
        }

        /// <summary>
        /// Convert half-precision to float (CPU side)
        /// </summary>
        public static float HalfToFloat(ushort value)
        {
            int sign = (value & 0x8000) << 16;
            int exp = (value >> 10) & 0x1F;
            int mantissa = value & 0x3FF;

            int bits;
            if (exp == 0)
            {
                bits = sign;
            }
            else if (exp == 31)
            {
                bits = sign | 0x7F800000 | (mantissa << 13);
            }
            else
            {
                exp = exp - 15 + 127;
                bits = sign | (exp << 23) | (mantissa << 13);
            }

            return BitConverter.Int32BitsToSingle(bits);
        }

        #endregion

        #region Memory Efficiency Utilities

        /// <summary>
        /// Calculate memory savings when using FP16 vs FP32
        /// </summary>
        public static (long fp32Bytes, long fp16Bytes, float savingsPercent) CalculateMemorySavings(int[] shape)
        {
            long elements = 1;
            foreach (var dim in shape) elements *= dim;

            long fp32 = elements * sizeof(float);
            long fp16 = elements * sizeof(ushort);
            float savings = (1.0f - (float)fp16 / fp32) * 100;

            return (fp32, fp16, savings);
        }

        #endregion
    }
}