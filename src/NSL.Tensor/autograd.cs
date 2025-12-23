using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;

namespace NSL.Tensor
{
    /// <summary>
    /// NSL Autograd System - Comprehensive Automatic Differentiation Engine.
    /// Provides PyTorch-style automatic gradient computation with computational graph tracking.
    ///
    /// Usage:
    /// using (var tape = new GradientTape())
    /// {
    ///     var y = x.MatMul(w).Add(b).ReLU();
    ///     tape.Backward(y, loss);
    /// }
    /// // x.Grad, w.Grad, b.Grad now contain gradients
    /// </summary>
    public sealed class GradientTape : IDisposable
    {
        [ThreadStatic]
        private static GradientTape? _current;

        private readonly List<TapeEntry> _tape;
        private readonly HashSet<int> _watchedTensors;
        private readonly bool _persistent;
        private bool _disposed;
        private int _nodeCounter;

        /// <summary>Public API</summary>
        public static GradientTape? Current => _current;
        /// <summary>Public API</summary>
        public bool IsPersistent => _persistent;
        /// <summary>Public API</summary>
        public int NumOperations => _tape.Count;
        /// <summary>Public API</summary>
        public bool IsRecording => !_disposed;

        /// <summary>
        /// Create a new gradient tape for recording operations.
        /// </summary>
        /// <param name="persistent">If true, tape can be used multiple times for backward.</param>
        public GradientTape(bool persistent = false)
        {
            _tape = new List<TapeEntry>(256);
            _watchedTensors = new HashSet<int>();
            _persistent = persistent;
            _current = this;
        }

        /// <summary>
        /// Watch a tensor for gradient computation.
        /// Tensors with RequiresGrad=true are automatically watched.
        /// </summary>
        public void Watch(Tensor tensor)
        {
            _watchedTensors.Add(tensor.Id);
        }

        /// <summary>
        /// Record an operation on the tape.
        /// </summary>
        internal void RecordOperation(GradientFunction gradFn, Tensor[] inputs, Tensor output)
        {
            if (_disposed) return;
            if (!Autograd.IsGradEnabled) return;

            bool anyRequiresGrad = false;
            foreach (var t in inputs)
            {
                if (t.RequiresGrad || _watchedTensors.Contains(t.Id))
                {
                    anyRequiresGrad = true;
                    break;
                }
            }
            if (!anyRequiresGrad) return;

            output.GradFn = gradFn;
            output.RequiresGrad = true;

            _tape.Add(new TapeEntry
            {
                GradFn = gradFn,
                Inputs = inputs,
                Output = output,
                NodeId = _nodeCounter++
            });
        }

        /// <summary>
        /// Compute gradients via reverse-mode autodiff.
        /// </summary>
        public void Backward(Tensor output, Tensor? gradOutput = null)
        {
            if (_disposed && !_persistent)
                throw new ObjectDisposedException("GradientTape has been disposed");

            gradOutput ??= Tensor.OnesLike(output);

            var gradients = new Dictionary<int, Tensor>();
            gradients[output.Id] = gradOutput;

            // Reverse topological order through tape
            for (int i = _tape.Count - 1; i >= 0; i--)
            {
                var entry = _tape[i];

                if (!gradients.TryGetValue(entry.Output.Id, out var outGrad))
                    continue;

                // Call the gradient function
                entry.GradFn.Backward(outGrad);

                // Accumulate gradients for inputs
                foreach (var input in entry.Inputs)
                {
                    if (input.Grad != null)
                    {
                        if (gradients.TryGetValue(input.Id, out var existing))
                        {
                            gradients[input.Id] = TensorOps.Add(existing, input.Grad);
                        }
                        else
                        {
                            gradients[input.Id] = input.Grad;
                        }
                    }
                }
            }

            if (!_persistent)
            {
                _tape.Clear();
            }
        }

        /// <summary>
        /// Compute gradients without storing them on tensors.
        /// </summary>
        public Dictionary<Tensor, Tensor> Gradient(Tensor output, Tensor[] inputs, Tensor? gradOutput = null)
        {
            gradOutput ??= Tensor.OnesLike(output);

            var result = new Dictionary<Tensor, Tensor>();
            var inputSet = new HashSet<int>(inputs.Select(t => t.Id));
            var gradients = new Dictionary<int, Tensor>();
            gradients[output.Id] = gradOutput;

            for (int i = _tape.Count - 1; i >= 0; i--)
            {
                var entry = _tape[i];

                if (!gradients.TryGetValue(entry.Output.Id, out var outGrad))
                    continue;

                entry.GradFn.Backward(outGrad);

                foreach (var input in entry.Inputs)
                {
                    if (input.Grad != null)
                    {
                        var grad = SumToShape(input.Grad, input.Shape);

                        if (gradients.TryGetValue(input.Id, out var existing))
                        {
                            gradients[input.Id] = TensorOps.Add(existing, grad);
                        }
                        else
                        {
                            gradients[input.Id] = grad;
                        }

                        if (inputSet.Contains(input.Id))
                        {
                            result[input] = gradients[input.Id];
                        }
                    }
                }
            }

            return result;
        }

        private static Tensor SumToShape(Tensor grad, long[] targetShape)
        {
            if (grad.Shape.SequenceEqual(targetShape))
                return grad;

            var result = grad;
            int gradNdim = grad.Shape.Length;
            int targetNdim = targetShape.Length;

            // Sum over leading dimensions
            while (result.Shape.Length > targetNdim)
            {
                result = result.Sum(0);
            }

            // Sum over broadcasted dimensions
            for (int i = 0; i < targetShape.Length; i++)
            {
                if (i < result.Shape.Length && targetShape[i] == 1 && result.Shape[i] > 1)
                {
                    result = result.Sum(i, keepDim: true);
                }
            }

            return result.Reshape(targetShape);
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _tape.Clear();
            if (_current == this)
                _current = null;
        }
    }

    internal class TapeEntry
    {
        /// <summary>Public API</summary>
        public GradientFunction GradFn { get; set; } = null!;
        /// <summary>Public API</summary>
        public Tensor[] Inputs { get; set; } = Array.Empty<Tensor>();
        /// <summary>Public API</summary>
        public Tensor Output { get; set; } = null!;
        /// <summary>Public API</summary>
        public int NodeId { get; set; }
    }

    /// <summary>
    /// Base class for gradient functions in the computation graph.
    /// </summary>
    public abstract class GradientFunction
    {
        /// <summary>Public API</summary>
        public abstract void Backward(Tensor gradient);
    }

    #region Arithmetic Backward Functions

    internal class AddBackward : GradientFunction
    {
        private readonly Tensor _a, _b;

        internal AddBackward(Tensor a, Tensor b)
        {
            _a = a;
            _b = b;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_a.RequiresGrad)
            {
                _a.AccumulateGrad(gradient);
                _a.GradFn?.Backward(gradient);
            }
            if (_b.RequiresGrad)
            {
                _b.AccumulateGrad(gradient);
                _b.GradFn?.Backward(gradient);
            }
        }
    }

    internal class SubBackward : GradientFunction
    {
        private readonly Tensor _a, _b;

        internal SubBackward(Tensor a, Tensor b)
        {
            _a = a;
            _b = b;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_a.RequiresGrad)
            {
                _a.AccumulateGrad(gradient);
                _a.GradFn?.Backward(gradient);
            }
            if (_b.RequiresGrad)
            {
                var negGrad = gradient.Neg();
                _b.AccumulateGrad(negGrad);
                _b.GradFn?.Backward(negGrad);
            }
        }
    }

    internal class MulBackward : GradientFunction
    {
        private readonly Tensor _a, _b;

        internal MulBackward(Tensor a, Tensor b)
        {
            _a = a;
            _b = b;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_a.RequiresGrad)
            {
                var gradA = TensorOps.Mul(gradient, _b.Detach());
                _a.AccumulateGrad(gradA);
                _a.GradFn?.Backward(gradA);
            }
            if (_b.RequiresGrad)
            {
                var gradB = TensorOps.Mul(gradient, _a.Detach());
                _b.AccumulateGrad(gradB);
                _b.GradFn?.Backward(gradB);
            }
        }
    }

    internal class MulScalarBackward : GradientFunction
    {
        private readonly Tensor _a;
        private readonly double _scalar;

        internal MulScalarBackward(Tensor a, double scalar)
        {
            _a = a;
            _scalar = scalar;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_a.RequiresGrad)
            {
                var gradA = TensorOps.MulScalar(gradient, _scalar);
                _a.AccumulateGrad(gradA);
                _a.GradFn?.Backward(gradA);
            }
        }
    }

    internal class DivBackward : GradientFunction
    {
        private readonly Tensor _a, _b;

        internal DivBackward(Tensor a, Tensor b)
        {
            _a = a;
            _b = b;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            var bDetach = _b.Detach();
            if (_a.RequiresGrad)
            {
                var gradA = TensorOps.Div(gradient, bDetach);
                _a.AccumulateGrad(gradA);
                _a.GradFn?.Backward(gradA);
            }
            if (_b.RequiresGrad)
            {
                var gradB = TensorOps.Div(TensorOps.Mul(TensorOps.Neg(gradient), _a.Detach()), TensorOps.Mul(bDetach, bDetach));
                _b.AccumulateGrad(gradB);
                _b.GradFn?.Backward(gradB);
            }
        }
    }

    internal class NegBackward : GradientFunction
    {
        private readonly Tensor _a;

        internal NegBackward(Tensor a)
        {
            _a = a;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_a.RequiresGrad)
            {
                var gradA = TensorOps.Neg(gradient);
                _a.AccumulateGrad(gradA);
                _a.GradFn?.Backward(gradA);
            }
        }
    }

    internal class PowBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly double _exponent;

        internal PowBackward(Tensor input, double exponent)
        {
            _input = input;
            _exponent = exponent;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = TensorOps.MulScalar(TensorOps.Mul(gradient, _input.Detach().Pow(_exponent - 1)), _exponent);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    #endregion

    #region Mathematical Backward Functions

    internal class ExpBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;

        internal ExpBackward(Tensor input, Tensor output)
        {
            _input = input;
            _output = output;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = TensorOps.Mul(gradient, _output.Detach());
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class LogBackward : GradientFunction
    {
        private readonly Tensor _input;

        internal LogBackward(Tensor input)
        {
            _input = input;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = TensorOps.Div(gradient, _input.Detach());
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class SqrtBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;

        internal SqrtBackward(Tensor input, Tensor output)
        {
            _input = input;
            _output = output;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = TensorOps.Div(gradient, TensorOps.MulScalar(_output.Detach(), 2.0));
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class TanhBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;

        internal TanhBackward(Tensor input, Tensor output)
        {
            _input = input;
            _output = output;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var outDet = _output.Detach();
                var grad = TensorOps.Mul(gradient, TensorOps.Sub(Tensor.Ones(outDet.Shape), TensorOps.Mul(outDet, outDet)));
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class SigmoidBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;

        internal SigmoidBackward(Tensor input, Tensor output)
        {
            _input = input;
            _output = output;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var s = _output.Detach();
                var grad = TensorOps.Mul(gradient, TensorOps.Mul(s, TensorOps.Sub(Tensor.Ones(s.Shape), s)));
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class ReLUBackward : GradientFunction
    {
        private readonly Tensor _input;

        internal ReLUBackward(Tensor input)
        {
            _input = input;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var mask = _input.Detach().Apply(x => x > 0 ? 1.0 : 0.0);
                var grad = TensorOps.Mul(gradient, mask);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class LeakyReLUBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly double _negativeSlope;

        internal LeakyReLUBackward(Tensor input, double negativeSlope)
        {
            _input = input;
            _negativeSlope = negativeSlope;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var slope = _negativeSlope;
                var mask = _input.Detach().Apply(x => x > 0 ? 1.0 : slope);
                var grad = TensorOps.Mul(gradient, mask);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class GELUBackward : GradientFunction
    {
        private readonly Tensor _input;

        private const double SQRT_2_OVER_PI = 0.7978845608028654;
        private const double COEFF = 0.044715;

        internal GELUBackward(Tensor input)
        {
            _input = input;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var x = _input.Detach();
                var grad = x.Apply(v =>
                {
                    double x3 = v * v * v;
                    double inner = SQRT_2_OVER_PI * (v + COEFF * x3);
                    double tanhInner = Math.Tanh(inner);
                    double sech2 = 1 - tanhInner * tanhInner;
                    double dInner = SQRT_2_OVER_PI * (1 + 3 * COEFF * v * v);
                    return 0.5 * (1 + tanhInner + v * sech2 * dInner);
                });
                grad = TensorOps.Mul(gradient, grad);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class SiLUBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;

        internal SiLUBackward(Tensor input, Tensor output)
        {
            _input = input;
            _output = output;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var x = _input.Detach();
                var sigmoid = x.Apply(v => 1.0 / (1.0 + Math.Exp(-v)));
                var grad = x.Apply((v, i) =>
                {
                    double s = sigmoid.Data[i];
                    return s * (1 + v * (1 - s));
                });
                grad = TensorOps.Mul(gradient, grad);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class SoftmaxBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;
        private readonly int _dim;

        internal SoftmaxBackward(Tensor input, Tensor output, int dim)
        {
            _input = input;
            _output = output;
            _dim = dim;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var s = _output.Detach();
                var gs = TensorOps.Mul(gradient, s);
                var sumGs = gs.Sum(_dim, keepDim: true);
                var grad = TensorOps.Mul(s, TensorOps.Sub(gradient, Tensor.BroadcastTo(sumGs, s.Shape)));
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class LogSoftmaxBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;
        private readonly int _dim;

        internal LogSoftmaxBackward(Tensor input, Tensor output, int dim)
        {
            _input = input;
            _output = output;
            _dim = dim;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var softmax = _output.Detach().Apply(Math.Exp);
                var sumGrad = gradient.Sum(_dim, keepDim: true);
                var grad = TensorOps.Sub(gradient, TensorOps.Mul(softmax, Tensor.BroadcastTo(sumGrad, softmax.Shape)));
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    #endregion

    #region Reduction Backward Functions

    internal class SumBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly int? _axis;
        private readonly bool _keepDim;

        internal SumBackward(Tensor input, int? axis = null, bool keepDim = false)
        {
            _input = input;
            _axis = axis;
            _keepDim = keepDim;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = gradient;
                if (_axis.HasValue && !_keepDim)
                {
                    var newShape = _input.Shape.ToArray();
                    newShape[_axis.Value] = 1;
                    grad = grad.Reshape(newShape);
                }
                grad = Tensor.BroadcastTo(grad, _input.Shape);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class MeanBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly int? _axis;
        private readonly bool _keepDim;

        internal MeanBackward(Tensor input, int? axis = null, bool keepDim = false)
        {
            _input = input;
            _axis = axis;
            _keepDim = keepDim;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                long n = _axis.HasValue ? _input.Shape[_axis.Value] : _input.Data.Length;
                var grad = TensorOps.MulScalar(gradient, 1.0 / n);

                if (_axis.HasValue && !_keepDim)
                {
                    var newShape = _input.Shape.ToArray();
                    newShape[_axis.Value] = 1;
                    grad = grad.Reshape(newShape);
                }
                grad = Tensor.BroadcastTo(grad, _input.Shape);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class MaxBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _output;
        private readonly int? _axis;

        internal MaxBackward(Tensor input, Tensor output, int? axis = null)
        {
            _input = input;
            _output = output;
            _axis = axis;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                // Gradient flows only to the max element
                var mask = _input.Detach().Apply((x, i) =>
                {
                    if (_axis.HasValue)
                    {
                        // Check if this element is the max along the axis
                        return x == _output.Data[i / _input.Shape[_axis.Value]] ? 1.0 : 0.0;
                    }
                    return x == _output.Data[0] ? 1.0 : 0.0;
                });
                var grad = TensorOps.Mul(Tensor.BroadcastTo(gradient, _input.Shape), mask);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    #endregion

    #region Matrix Operation Backward Functions

    internal class MatMulBackward : GradientFunction
    {
        private readonly Tensor _a, _b;

        internal MatMulBackward(Tensor a, Tensor b)
        {
            _a = a;
            _b = b;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_a.RequiresGrad)
            {
                var gradA = TensorOps.MatMul(gradient, _b.Detach().T());
                _a.AccumulateGrad(gradA);
                _a.GradFn?.Backward(gradA);
            }
            if (_b.RequiresGrad)
            {
                var gradB = TensorOps.MatMul(_a.Detach().T(), gradient);
                _b.AccumulateGrad(gradB);
                _b.GradFn?.Backward(gradB);
            }
        }
    }

    internal class ReshapeBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly long[] _originalShape;

        internal ReshapeBackward(Tensor input, long[] originalShape)
        {
            _input = input;
            _originalShape = originalShape;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = gradient.Reshape(_originalShape);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class TransposeBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly int _dim0, _dim1;

        internal TransposeBackward(Tensor input, int dim0 = 0, int dim1 = 1)
        {
            _input = input;
            _dim0 = dim0;
            _dim1 = dim1;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = gradient.Transpose(_dim0, _dim1);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class ConcatBackward : GradientFunction
    {
        private readonly Tensor[] _inputs;
        private readonly int _axis;

        internal ConcatBackward(Tensor[] inputs, int axis)
        {
            _inputs = inputs;
            _axis = axis;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            long offset = 0;
            foreach (var input in _inputs)
            {
                if (input.RequiresGrad)
                {
                    long size = input.Shape[_axis];
                    var grad = gradient.Slice(_axis, offset, offset + size);
                    input.AccumulateGrad(grad);
                    input.GradFn?.Backward(grad);
                }
                offset += input.Shape[_axis];
            }
        }
    }

    internal class SliceBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly int _axis;
        private readonly long _start;
        private readonly long _end;

        internal SliceBackward(Tensor input, int axis, long start, long end)
        {
            _input = input;
            _axis = axis;
            _start = start;
            _end = end;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = Tensor.Zeros(_input.Shape);
                grad.SetSlice(_axis, _start, _end, gradient);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    #endregion

    #region Layer Backward Functions

    internal class EmbeddingBackward : GradientFunction
    {
        private readonly Tensor _weight;
        private readonly Tensor _indices;
        private readonly int _numEmbeddings;
        private readonly int _embeddingDim;

        internal EmbeddingBackward(Tensor weight, Tensor indices, int numEmbeddings, int embeddingDim)
        {
            _weight = weight;
            _indices = indices;
            _numEmbeddings = numEmbeddings;
            _embeddingDim = embeddingDim;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_weight.RequiresGrad)
            {
                var gradWeight = Tensor.Zeros(_weight.Shape);
                var flatGrad = gradient.Reshape(new long[] { -1, _embeddingDim });

                for (int i = 0; i < _indices.Data.Length; i++)
                {
                    int idx = (int)_indices.Data[i];
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        gradWeight.Data[idx * _embeddingDim + j] += flatGrad.Data[i * _embeddingDim + j];
                    }
                }

                _weight.AccumulateGrad(gradWeight);
                _weight.GradFn?.Backward(gradWeight);
            }
        }
    }

    internal class LayerNormBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor? _weight;
        private readonly Tensor? _bias;
        private readonly Tensor _mean;
        private readonly Tensor _invStd;
        private readonly long[] _normalizedShape;

        internal LayerNormBackward(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invStd, long[] normalizedShape)
        {
            _input = input;
            _weight = weight;
            _bias = bias;
            _mean = mean;
            _invStd = invStd;
            _normalizedShape = normalizedShape;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            long n = _normalizedShape.Aggregate(1L, (a, b) => a * b);
            var xCentered = TensorOps.Sub(_input.Detach(), Tensor.BroadcastTo(_mean, _input.Shape));
            var xNorm = TensorOps.Mul(xCentered, Tensor.BroadcastTo(_invStd, _input.Shape));

            Tensor dxNorm;
            if (_weight != null)
            {
                dxNorm = TensorOps.Mul(gradient, Tensor.BroadcastTo(_weight.Reshape(new long[] { 1, _normalizedShape[0] }), gradient.Shape));

                if (_weight.RequiresGrad)
                {
                    var gradWeight = TensorOps.Mul(gradient, xNorm).Sum(0);
                    _weight.AccumulateGrad(gradWeight);
                }
            }
            else
            {
                dxNorm = gradient;
            }

            if (_bias != null && _bias.RequiresGrad)
            {
                var gradBias = gradient.Sum(0);
                _bias.AccumulateGrad(gradBias);
            }

            if (_input.RequiresGrad)
            {
                var invStdBroad = Tensor.BroadcastTo(_invStd, _input.Shape);
                var dxCentered = TensorOps.Mul(dxNorm, invStdBroad);
                var dMean = TensorOps.MulScalar(dxCentered.Sum(-1, keepDim: true), -1);
                var dVar = TensorOps.Mul(dxNorm, xCentered).Mul(TensorOps.MulScalar(TensorOps.Mul(invStdBroad, TensorOps.Mul(invStdBroad, invStdBroad)), -0.5)).Sum(-1, keepDim: true);

                var dx = TensorOps.Add(
                    TensorOps.Add(dxCentered, TensorOps.MulScalar(Tensor.BroadcastTo(dMean, _input.Shape), 1.0 / n)),
                    TensorOps.MulScalar(TensorOps.Mul(xCentered, Tensor.BroadcastTo(dVar, _input.Shape)), 2.0 / n)
                );

                _input.AccumulateGrad(dx);
                _input.GradFn?.Backward(dx);
            }
        }
    }

    internal class DropoutBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _mask;
        private readonly double _scale;

        internal DropoutBackward(Tensor input, Tensor mask, double scale)
        {
            _input = input;
            _mask = mask;
            _scale = scale;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = TensorOps.MulScalar(TensorOps.Mul(gradient, _mask), _scale);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class Conv2dBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _weight;
        private readonly Tensor? _bias;
        private readonly int _stride;
        private readonly int _padding;

        internal Conv2dBackward(Tensor input, Tensor weight, Tensor? bias, int stride, int padding)
        {
            _input = input;
            _weight = weight;
            _bias = bias;
            _stride = stride;
            _padding = padding;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            int batchSize = (int)_input.Shape[0];
            int inChannels = (int)_input.Shape[1];
            int inH = (int)_input.Shape[2];
            int inW = (int)_input.Shape[3];
            int outChannels = (int)_weight.Shape[0];
            int kH = (int)_weight.Shape[2];
            int kW = (int)_weight.Shape[3];
            int outH = (int)gradient.Shape[2];
            int outW = (int)gradient.Shape[3];

            if (_input.RequiresGrad)
            {
                var gradInput = Tensor.Zeros(_input.Shape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                double gradVal = gradient.Data[b * outChannels * outH * outW + oc * outH * outW + oh * outW + ow];
                                for (int ic = 0; ic < inChannels; ic++)
                                {
                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int ih = oh * _stride - _padding + kh;
                                            int iw = ow * _stride - _padding + kw;
                                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                            {
                                                double wVal = _weight.Data[oc * inChannels * kH * kW + ic * kH * kW + kh * kW + kw];
                                                gradInput.Data[b * inChannels * inH * inW + ic * inH * inW + ih * inW + iw] += gradVal * wVal;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _input.AccumulateGrad(gradInput);
                _input.GradFn?.Backward(gradInput);
            }

            if (_weight.RequiresGrad)
            {
                var gradWeight = Tensor.Zeros(_weight.Shape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kH; kh++)
                            {
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    double sum = 0;
                                    for (int oh = 0; oh < outH; oh++)
                                    {
                                        for (int ow = 0; ow < outW; ow++)
                                        {
                                            int ih = oh * _stride - _padding + kh;
                                            int iw = ow * _stride - _padding + kw;
                                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                            {
                                                sum += gradient.Data[b * outChannels * outH * outW + oc * outH * outW + oh * outW + ow]
                                                     * _input.Data[b * inChannels * inH * inW + ic * inH * inW + ih * inW + iw];
                                            }
                                        }
                                    }
                                    gradWeight.Data[oc * inChannels * kH * kW + ic * kH * kW + kh * kW + kw] += sum;
                                }
                            }
                        }
                    }
                }
                _weight.AccumulateGrad(gradWeight);
                _weight.GradFn?.Backward(gradWeight);
            }

            if (_bias != null && _bias.RequiresGrad)
            {
                var gradBias = Tensor.Zeros(_bias.Shape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                gradBias.Data[oc] += gradient.Data[b * outChannels * outH * outW + oc * outH * outW + oh * outW + ow];
                            }
                        }
                    }
                }
                _bias.AccumulateGrad(gradBias);
            }
        }
    }

    internal class MaxPool2dBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _indices;
        private readonly int _kernelSize;
        private readonly int _stride;

        internal MaxPool2dBackward(Tensor input, Tensor indices, int kernelSize, int stride)
        {
            _input = input;
            _indices = indices;
            _kernelSize = kernelSize;
            _stride = stride;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var gradInput = Tensor.Zeros(_input.Shape);
                for (int i = 0; i < gradient.Data.Length; i++)
                {
                    int maxIdx = (int)_indices.Data[i];
                    gradInput.Data[maxIdx] += gradient.Data[i];
                }
                _input.AccumulateGrad(gradInput);
                _input.GradFn?.Backward(gradInput);
            }
        }
    }

    #endregion

    #region Loss Function Backward Functions

    internal class MSELossBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _target;
        private readonly string _reduction;

        internal MSELossBackward(Tensor input, Tensor target, string reduction)
        {
            _input = input;
            _target = target;
            _reduction = reduction;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var diff = TensorOps.Sub(_input.Detach(), _target);
                Tensor grad;
                if (_reduction == "mean")
                    grad = TensorOps.MulScalar(diff, 2.0 / _input.Data.Length * gradient.Data[0]);
                else if (_reduction == "sum")
                    grad = TensorOps.MulScalar(diff, 2.0 * gradient.Data[0]);
                else
                    grad = TensorOps.MulScalar(TensorOps.Mul(diff, gradient), 2.0);

                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class CrossEntropyLossBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _target;
        private readonly Tensor _softmax;

        internal CrossEntropyLossBackward(Tensor input, Tensor target, Tensor softmax)
        {
            _input = input;
            _target = target;
            _softmax = softmax;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = TensorOps.MulScalar(TensorOps.Sub(_softmax.Detach(), _target), gradient.Data[0] / _input.Shape[0]);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class BCELossBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _target;

        internal BCELossBackward(Tensor input, Tensor target)
        {
            _input = input;
            _target = target;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var inp = _input.Detach();
                var grad = inp.Apply((x, i) =>
                {
                    double t = _target.Data[i];
                    double eps = 1e-7;
                    return (x - t) / (Math.Max(x * (1 - x), eps));
                });
                grad = TensorOps.MulScalar(grad, gradient.Data[0] / _input.Data.Length);
                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    internal class NLLLossBackward : GradientFunction
    {
        private readonly Tensor _input;
        private readonly Tensor _target;

        internal NLLLossBackward(Tensor input, Tensor target)
        {
            _input = input;
            _target = target;
        }

        /// <summary>Public API</summary>
        public override void Backward(Tensor gradient)
        {
            if (_input.RequiresGrad)
            {
                var grad = Tensor.Zeros(_input.Shape);
                int batchSize = (int)_input.Shape[0];
                int numClasses = (int)_input.Shape[1];

                for (int b = 0; b < batchSize; b++)
                {
                    int target = (int)_target.Data[b];
                    grad.Data[b * numClasses + target] = -gradient.Data[0] / batchSize;
                }

                _input.AccumulateGrad(grad);
                _input.GradFn?.Backward(grad);
            }
        }
    }

    #endregion

    /// <summary>
    /// Context manager for no_grad mode.
    /// </summary>
    public sealed class NoGradScope : IDisposable
    {
        /// <summary>Public API</summary>
        public NoGradScope()
        {
            Autograd.PushNoGrad();
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            Autograd.PopNoGrad();
        }
    }

    /// <summary>
    /// Context manager for enable_grad mode (overrides no_grad).
    /// </summary>
    public sealed class EnableGradScope : IDisposable
    {
        /// <summary>Public API</summary>
        public EnableGradScope()
        {
            Autograd.PushEnableGrad();
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            Autograd.PopEnableGrad();
        }
    }

    /// <summary>
    /// Static autograd utilities.
    /// </summary>
    public static class Autograd
    {
        [ThreadStatic]
        private static int _noGradDepth;

        /// <summary>Public API</summary>
        public static bool IsGradEnabled => _noGradDepth <= 0;

        internal static void PushNoGrad() => _noGradDepth++;
        internal static void PopNoGrad() => _noGradDepth = Math.Max(0, _noGradDepth - 1);
        internal static void PushEnableGrad() => _noGradDepth--;
        internal static void PopEnableGrad() => _noGradDepth++;

        /// <summary>
        /// Create a no_grad scope.
        /// </summary>
        public static NoGradScope NoGrad() => new NoGradScope();

        /// <summary>
        /// Create an enable_grad scope.
        /// </summary>
        public static EnableGradScope EnableGrad() => new EnableGradScope();

        /// <summary>
        /// Compute gradients of outputs with respect to inputs.
        /// </summary>
        public static Tensor[] Grad(Tensor[] outputs, Tensor[] inputs, Tensor[]? gradOutputs = null, bool retainGraph = false)
        {
            gradOutputs ??= outputs.Select(o => Tensor.OnesLike(o)).ToArray();

            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i].Backward(gradOutputs[i]);
            }

            return inputs.Select(inp => inp.Grad?.Clone() ?? Tensor.Zeros(inp.Shape)).ToArray();
        }

        /// <summary>
        /// Zero gradients for all tensors.
        /// </summary>
        public static void ZeroGrad(params Tensor[] tensors)
        {
            foreach (var t in tensors)
            {
                t.ZeroGrad();
            }
        }
    }
}