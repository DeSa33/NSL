using System;
using System.Collections.Generic;

namespace NSL.Tensor.NN
{
    /// <summary>
    /// Base class for recurrent layers
    /// </summary>
    public abstract class RNNBase : Module
    {
        protected readonly int _inputSize;
        protected readonly int _hiddenSize;
        protected readonly int _numLayers;
        protected readonly bool _batch_first;
        protected readonly double _dropout;
        protected readonly bool _bidirectional;
        protected readonly int _numDirections;

        protected RNNBase(int inputSize, int hiddenSize, int numLayers = 1, bool batchFirst = false,
            double dropout = 0, bool bidirectional = false)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;
            _numLayers = numLayers;
            _batch_first = batchFirst;
            _dropout = dropout;
            _bidirectional = bidirectional;
            _numDirections = bidirectional ? 2 : 1;
        }
    }

    /// <summary>
    /// Basic RNN layer
    /// </summary>
    public class RNN : RNNBase
    {
        private readonly string _nonlinearity;
        private readonly List<Tensor> _weightIh; // input-hidden weights
        private readonly List<Tensor> _weightHh; // hidden-hidden weights
        private readonly List<Tensor> _biasIh;
        private readonly List<Tensor> _biasHh;

        /// <summary>Public API</summary>
        public RNN(int inputSize, int hiddenSize, int numLayers = 1, string nonlinearity = "tanh",
            bool batchFirst = false, double dropout = 0, bool bidirectional = false, bool bias = true)
            : base(inputSize, hiddenSize, numLayers, batchFirst, dropout, bidirectional)
        {
            _nonlinearity = nonlinearity;
            _weightIh = new List<Tensor>();
            _weightHh = new List<Tensor>();
            _biasIh = new List<Tensor>();
            _biasHh = new List<Tensor>();

            for (int layer = 0; layer < numLayers; layer++)
            {
                for (int dir = 0; dir < _numDirections; dir++)
                {
                    int layerInputSize = layer == 0 ? inputSize : hiddenSize * _numDirections;

                    // Initialize weights with Xavier/Glorot initialization
                    double stdv = 1.0 / Math.Sqrt(hiddenSize);

                    var wIh = Tensor.Uniform(new long[] { hiddenSize, layerInputSize }, -stdv, stdv);
                    var wHh = Tensor.Uniform(new long[] { hiddenSize, hiddenSize }, -stdv, stdv);
                    RegisterParameter($"weight_ih_l{layer}" + (dir == 1 ? "_reverse" : ""), wIh);
                    RegisterParameter($"weight_hh_l{layer}" + (dir == 1 ? "_reverse" : ""), wHh);
                    _weightIh.Add(wIh);
                    _weightHh.Add(wHh);

                    if (bias)
                    {
                        var bIh = Tensor.Uniform(new long[] { hiddenSize }, -stdv, stdv);
                        var bHh = Tensor.Uniform(new long[] { hiddenSize }, -stdv, stdv);
                        RegisterParameter($"bias_ih_l{layer}" + (dir == 1 ? "_reverse" : ""), bIh);
                        RegisterParameter($"bias_hh_l{layer}" + (dir == 1 ? "_reverse" : ""), bHh);
                        _biasIh.Add(bIh);
                        _biasHh.Add(bHh);
                    }
                }
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var (output, _) = Forward(input, null);
            return output;
        }

        /// <summary>Public API</summary>
        public (Tensor output, Tensor hN) Forward(Tensor input, Tensor? h0)
        {
            // input: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
            if (_batch_first)
                input = input.Transpose(0, 1);

            var seqLen = (int)input.Shape[0];
            var batch = (int)input.Shape[1];

            // Initialize hidden state
            if (h0 == null)
                h0 = Tensor.Zeros(new long[] { _numLayers * _numDirections, batch, _hiddenSize });

            var currentInput = input;
            var allHidden = new List<Tensor>();

            for (int layer = 0; layer < _numLayers; layer++)
            {
                var outputs = new List<Tensor>();

                // Forward direction
                var h = h0.Slice(0, layer * _numDirections, layer * _numDirections + 1).Squeeze(0);
                int weightIdx = layer * _numDirections;

                for (int t = 0; t < seqLen; t++)
                {
                    var x = currentInput.Slice(0, t, t + 1).Squeeze(0);
                    h = ComputeCell(x, h, _weightIh[weightIdx], _weightHh[weightIdx],
                        _biasIh.Count > 0 ? _biasIh[weightIdx] : null,
                        _biasHh.Count > 0 ? _biasHh[weightIdx] : null);
                    outputs.Add(h.Unsqueeze(0));
                }

                var forwardOutput = TensorOps.Cat(outputs.ToArray(), 0);
                allHidden.Add(h);

                if (_bidirectional)
                {
                    // Backward direction
                    var outputsRev = new List<Tensor>();
                    var hRev = h0.Slice(0, layer * _numDirections + 1, layer * _numDirections + 2).Squeeze(0);
                    int weightIdxRev = layer * _numDirections + 1;

                    for (int t = seqLen - 1; t >= 0; t--)
                    {
                        var x = currentInput.Slice(0, t, t + 1).Squeeze(0);
                        hRev = ComputeCell(x, hRev, _weightIh[weightIdxRev], _weightHh[weightIdxRev],
                            _biasIh.Count > 0 ? _biasIh[weightIdxRev] : null,
                            _biasHh.Count > 0 ? _biasHh[weightIdxRev] : null);
                        outputsRev.Insert(0, hRev.Unsqueeze(0));
                    }

                    var backwardOutput = TensorOps.Cat(outputsRev.ToArray(), 0);
                    allHidden.Add(hRev);

                    // Concatenate forward and backward
                    currentInput = TensorOps.Cat(new[] { forwardOutput, backwardOutput }, 2);
                }
                else
                {
                    currentInput = forwardOutput;
                }

                // Apply dropout between layers (except last layer)
                if (_dropout > 0 && layer < _numLayers - 1 && Training)
                {
                    currentInput = F.Dropout(currentInput, _dropout, true);
                }
            }

            var hN = TensorOps.Stack(allHidden.ToArray(), 0);

            if (_batch_first)
                currentInput = currentInput.Transpose(0, 1);

            return (currentInput, hN);
        }

        private Tensor ComputeCell(Tensor x, Tensor h, Tensor wIh, Tensor wHh, Tensor? bIh, Tensor? bHh)
        {
            var gates = TensorOps.MatMul(x, wIh.T()).Add(TensorOps.MatMul(h, wHh.T()));
            if (bIh != null) gates = gates.Add(bIh);
            if (bHh != null) gates = gates.Add(bHh);

            return _nonlinearity == "relu" ? gates.ReLU() : gates.Tanh();
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"RNN({_inputSize}, {_hiddenSize}, num_layers={_numLayers}, nonlinearity={_nonlinearity}, bidirectional={_bidirectional})";
    }

    /// <summary>
    /// LSTM (Long Short-Term Memory) layer
    /// </summary>
    public class LSTM : RNNBase
    {
        private readonly List<Tensor> _weightIh; // [4*hidden, input]
        private readonly List<Tensor> _weightHh; // [4*hidden, hidden]
        private readonly List<Tensor> _biasIh;
        private readonly List<Tensor> _biasHh;

        /// <summary>Public API</summary>
        public LSTM(int inputSize, int hiddenSize, int numLayers = 1, bool batchFirst = false,
            double dropout = 0, bool bidirectional = false, bool bias = true)
            : base(inputSize, hiddenSize, numLayers, batchFirst, dropout, bidirectional)
        {
            _weightIh = new List<Tensor>();
            _weightHh = new List<Tensor>();
            _biasIh = new List<Tensor>();
            _biasHh = new List<Tensor>();

            for (int layer = 0; layer < numLayers; layer++)
            {
                for (int dir = 0; dir < _numDirections; dir++)
                {
                    int layerInputSize = layer == 0 ? inputSize : hiddenSize * _numDirections;

                    double stdv = 1.0 / Math.Sqrt(hiddenSize);

                    // LSTM has 4 gates: input, forget, cell, output
                    var wIh = Tensor.Uniform(new long[] { 4 * hiddenSize, layerInputSize }, -stdv, stdv);
                    var wHh = Tensor.Uniform(new long[] { 4 * hiddenSize, hiddenSize }, -stdv, stdv);
                    RegisterParameter($"weight_ih_l{layer}" + (dir == 1 ? "_reverse" : ""), wIh);
                    RegisterParameter($"weight_hh_l{layer}" + (dir == 1 ? "_reverse" : ""), wHh);
                    _weightIh.Add(wIh);
                    _weightHh.Add(wHh);

                    if (bias)
                    {
                        var bIh = Tensor.Uniform(new long[] { 4 * hiddenSize }, -stdv, stdv);
                        var bHh = Tensor.Uniform(new long[] { 4 * hiddenSize }, -stdv, stdv);
                        RegisterParameter($"bias_ih_l{layer}" + (dir == 1 ? "_reverse" : ""), bIh);
                        RegisterParameter($"bias_hh_l{layer}" + (dir == 1 ? "_reverse" : ""), bHh);
                        _biasIh.Add(bIh);
                        _biasHh.Add(bHh);
                    }
                }
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var (output, _, _) = Forward(input, null, null);
            return output;
        }

        /// <summary>Public API</summary>
        public (Tensor output, Tensor hN, Tensor cN) Forward(Tensor input, Tensor? h0, Tensor? c0)
        {
            if (_batch_first)
                input = input.Transpose(0, 1);

            var seqLen = (int)input.Shape[0];
            var batch = (int)input.Shape[1];

            // Initialize hidden and cell states
            if (h0 == null)
                h0 = Tensor.Zeros(new long[] { _numLayers * _numDirections, batch, _hiddenSize });
            if (c0 == null)
                c0 = Tensor.Zeros(new long[] { _numLayers * _numDirections, batch, _hiddenSize });

            var currentInput = input;
            var allH = new List<Tensor>();
            var allC = new List<Tensor>();

            for (int layer = 0; layer < _numLayers; layer++)
            {
                var outputs = new List<Tensor>();

                // Forward direction
                var h = h0.Slice(0, layer * _numDirections, layer * _numDirections + 1).Squeeze(0);
                var c = c0.Slice(0, layer * _numDirections, layer * _numDirections + 1).Squeeze(0);
                int weightIdx = layer * _numDirections;

                for (int t = 0; t < seqLen; t++)
                {
                    var x = currentInput.Slice(0, t, t + 1).Squeeze(0);
                    (h, c) = ComputeLSTMCell(x, h, c, _weightIh[weightIdx], _weightHh[weightIdx],
                        _biasIh.Count > 0 ? _biasIh[weightIdx] : null,
                        _biasHh.Count > 0 ? _biasHh[weightIdx] : null);
                    outputs.Add(h.Unsqueeze(0));
                }

                var forwardOutput = TensorOps.Cat(outputs.ToArray(), 0);
                allH.Add(h);
                allC.Add(c);

                if (_bidirectional)
                {
                    var outputsRev = new List<Tensor>();
                    var hRev = h0.Slice(0, layer * _numDirections + 1, layer * _numDirections + 2).Squeeze(0);
                    var cRev = c0.Slice(0, layer * _numDirections + 1, layer * _numDirections + 2).Squeeze(0);
                    int weightIdxRev = layer * _numDirections + 1;

                    for (int t = seqLen - 1; t >= 0; t--)
                    {
                        var x = currentInput.Slice(0, t, t + 1).Squeeze(0);
                        (hRev, cRev) = ComputeLSTMCell(x, hRev, cRev, _weightIh[weightIdxRev], _weightHh[weightIdxRev],
                            _biasIh.Count > 0 ? _biasIh[weightIdxRev] : null,
                            _biasHh.Count > 0 ? _biasHh[weightIdxRev] : null);
                        outputsRev.Insert(0, hRev.Unsqueeze(0));
                    }

                    var backwardOutput = TensorOps.Cat(outputsRev.ToArray(), 0);
                    allH.Add(hRev);
                    allC.Add(cRev);

                    currentInput = TensorOps.Cat(new[] { forwardOutput, backwardOutput }, 2);
                }
                else
                {
                    currentInput = forwardOutput;
                }

                if (_dropout > 0 && layer < _numLayers - 1 && Training)
                {
                    currentInput = F.Dropout(currentInput, _dropout, true);
                }
            }

            var hN = TensorOps.Stack(allH.ToArray(), 0);
            var cN = TensorOps.Stack(allC.ToArray(), 0);

            if (_batch_first)
                currentInput = currentInput.Transpose(0, 1);

            return (currentInput, hN, cN);
        }

        private (Tensor h, Tensor c) ComputeLSTMCell(Tensor x, Tensor h, Tensor c, Tensor wIh, Tensor wHh, Tensor? bIh, Tensor? bHh)
        {
            var gates = TensorOps.MatMul(x, wIh.T()).Add(TensorOps.MatMul(h, wHh.T()));
            if (bIh != null) gates = gates.Add(bIh);
            if (bHh != null) gates = gates.Add(bHh);

            // Split into 4 gates: input, forget, cell, output
            var chunks = TensorOps.Chunk(gates, 4, -1);
            var i = chunks[0].Sigmoid();  // input gate
            var f = chunks[1].Sigmoid();  // forget gate
            var g = chunks[2].Tanh();     // cell gate
            var o = chunks[3].Sigmoid();  // output gate

            var newC = f.Mul(c).Add(i.Mul(g));
            var newH = o.Mul(newC.Tanh());

            return (newH, newC);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"LSTM({_inputSize}, {_hiddenSize}, num_layers={_numLayers}, bidirectional={_bidirectional})";
    }

    /// <summary>
    /// GRU (Gated Recurrent Unit) layer
    /// </summary>
    public class GRU : RNNBase
    {
        private readonly List<Tensor> _weightIh; // [3*hidden, input]
        private readonly List<Tensor> _weightHh; // [3*hidden, hidden]
        private readonly List<Tensor> _biasIh;
        private readonly List<Tensor> _biasHh;

        /// <summary>Public API</summary>
        public GRU(int inputSize, int hiddenSize, int numLayers = 1, bool batchFirst = false,
            double dropout = 0, bool bidirectional = false, bool bias = true)
            : base(inputSize, hiddenSize, numLayers, batchFirst, dropout, bidirectional)
        {
            _weightIh = new List<Tensor>();
            _weightHh = new List<Tensor>();
            _biasIh = new List<Tensor>();
            _biasHh = new List<Tensor>();

            for (int layer = 0; layer < numLayers; layer++)
            {
                for (int dir = 0; dir < _numDirections; dir++)
                {
                    int layerInputSize = layer == 0 ? inputSize : hiddenSize * _numDirections;

                    double stdv = 1.0 / Math.Sqrt(hiddenSize);

                    // GRU has 3 gates: reset, update, new
                    var wIh = Tensor.Uniform(new long[] { 3 * hiddenSize, layerInputSize }, -stdv, stdv);
                    var wHh = Tensor.Uniform(new long[] { 3 * hiddenSize, hiddenSize }, -stdv, stdv);
                    RegisterParameter($"weight_ih_l{layer}" + (dir == 1 ? "_reverse" : ""), wIh);
                    RegisterParameter($"weight_hh_l{layer}" + (dir == 1 ? "_reverse" : ""), wHh);
                    _weightIh.Add(wIh);
                    _weightHh.Add(wHh);

                    if (bias)
                    {
                        var bIh = Tensor.Uniform(new long[] { 3 * hiddenSize }, -stdv, stdv);
                        var bHh = Tensor.Uniform(new long[] { 3 * hiddenSize }, -stdv, stdv);
                        RegisterParameter($"bias_ih_l{layer}" + (dir == 1 ? "_reverse" : ""), bIh);
                        RegisterParameter($"bias_hh_l{layer}" + (dir == 1 ? "_reverse" : ""), bHh);
                        _biasIh.Add(bIh);
                        _biasHh.Add(bHh);
                    }
                }
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            var (output, _) = Forward(input, null);
            return output;
        }

        /// <summary>Public API</summary>
        public (Tensor output, Tensor hN) Forward(Tensor input, Tensor? h0)
        {
            if (_batch_first)
                input = input.Transpose(0, 1);

            var seqLen = (int)input.Shape[0];
            var batch = (int)input.Shape[1];

            if (h0 == null)
                h0 = Tensor.Zeros(new long[] { _numLayers * _numDirections, batch, _hiddenSize });

            var currentInput = input;
            var allH = new List<Tensor>();

            for (int layer = 0; layer < _numLayers; layer++)
            {
                var outputs = new List<Tensor>();

                var h = h0.Slice(0, layer * _numDirections, layer * _numDirections + 1).Squeeze(0);
                int weightIdx = layer * _numDirections;

                for (int t = 0; t < seqLen; t++)
                {
                    var x = currentInput.Slice(0, t, t + 1).Squeeze(0);
                    h = ComputeGRUCell(x, h, _weightIh[weightIdx], _weightHh[weightIdx],
                        _biasIh.Count > 0 ? _biasIh[weightIdx] : null,
                        _biasHh.Count > 0 ? _biasHh[weightIdx] : null);
                    outputs.Add(h.Unsqueeze(0));
                }

                var forwardOutput = TensorOps.Cat(outputs.ToArray(), 0);
                allH.Add(h);

                if (_bidirectional)
                {
                    var outputsRev = new List<Tensor>();
                    var hRev = h0.Slice(0, layer * _numDirections + 1, layer * _numDirections + 2).Squeeze(0);
                    int weightIdxRev = layer * _numDirections + 1;

                    for (int t = seqLen - 1; t >= 0; t--)
                    {
                        var x = currentInput.Slice(0, t, t + 1).Squeeze(0);
                        hRev = ComputeGRUCell(x, hRev, _weightIh[weightIdxRev], _weightHh[weightIdxRev],
                            _biasIh.Count > 0 ? _biasIh[weightIdxRev] : null,
                            _biasHh.Count > 0 ? _biasHh[weightIdxRev] : null);
                        outputsRev.Insert(0, hRev.Unsqueeze(0));
                    }

                    var backwardOutput = TensorOps.Cat(outputsRev.ToArray(), 0);
                    allH.Add(hRev);

                    currentInput = TensorOps.Cat(new[] { forwardOutput, backwardOutput }, 2);
                }
                else
                {
                    currentInput = forwardOutput;
                }

                if (_dropout > 0 && layer < _numLayers - 1 && Training)
                {
                    currentInput = F.Dropout(currentInput, _dropout, true);
                }
            }

            var hN = TensorOps.Stack(allH.ToArray(), 0);

            if (_batch_first)
                currentInput = currentInput.Transpose(0, 1);

            return (currentInput, hN);
        }

        private Tensor ComputeGRUCell(Tensor x, Tensor h, Tensor wIh, Tensor wHh, Tensor? bIh, Tensor? bHh)
        {
            var gatesX = TensorOps.MatMul(x, wIh.T());
            var gatesH = TensorOps.MatMul(h, wHh.T());

            if (bIh != null) gatesX = gatesX.Add(bIh);
            if (bHh != null) gatesH = gatesH.Add(bHh);

            var chunksX = TensorOps.Chunk(gatesX, 3, -1);
            var chunksH = TensorOps.Chunk(gatesH, 3, -1);

            var r = chunksX[0].Add(chunksH[0]).Sigmoid();  // reset gate
            var z = chunksX[1].Add(chunksH[1]).Sigmoid();  // update gate
            var n = chunksX[2].Add(r.Mul(chunksH[2])).Tanh();  // new gate

            return Tensor.Ones(z.Shape).Sub(z).Mul(n).Add(z.Mul(h));
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"GRU({_inputSize}, {_hiddenSize}, num_layers={_numLayers}, bidirectional={_bidirectional})";
    }

    /// <summary>
    /// LSTM Cell - single step computation
    /// </summary>
    public class LSTMCell : Module
    {
        private readonly int _inputSize;
        private readonly int _hiddenSize;
        private Tensor _weightIh;
        private Tensor _weightHh;
        private Tensor? _biasIh;
        private Tensor? _biasHh;

        /// <summary>Public API</summary>
        public LSTMCell(int inputSize, int hiddenSize, bool bias = true)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;

            double stdv = 1.0 / Math.Sqrt(hiddenSize);
            _weightIh = Tensor.Uniform(new long[] { 4 * hiddenSize, inputSize }, -stdv, stdv);
            _weightHh = Tensor.Uniform(new long[] { 4 * hiddenSize, hiddenSize }, -stdv, stdv);
            RegisterParameter("weight_ih", _weightIh);
            RegisterParameter("weight_hh", _weightHh);

            if (bias)
            {
                _biasIh = Tensor.Uniform(new long[] { 4 * hiddenSize }, -stdv, stdv);
                _biasHh = Tensor.Uniform(new long[] { 4 * hiddenSize }, -stdv, stdv);
                RegisterParameter("bias_ih", _biasIh);
                RegisterParameter("bias_hh", _biasHh);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            throw new InvalidOperationException("Use Forward(input, (h, c)) instead");
        }

        /// <summary>Public API</summary>
        public (Tensor h, Tensor c) Forward(Tensor input, (Tensor h, Tensor c)? hc)
        {
            var batch = (int)input.Shape[0];
            var h = hc?.h ?? Tensor.Zeros(new long[] { batch, _hiddenSize });
            var c = hc?.c ?? Tensor.Zeros(new long[] { batch, _hiddenSize });

            var gates = TensorOps.MatMul(input, _weightIh.T()).Add(TensorOps.MatMul(h, _weightHh.T()));
            if (_biasIh != null) gates = gates.Add(_biasIh);
            if (_biasHh != null) gates = gates.Add(_biasHh);

            var chunks = TensorOps.Chunk(gates, 4, -1);
            var i = chunks[0].Sigmoid();
            var f = chunks[1].Sigmoid();
            var g = chunks[2].Tanh();
            var o = chunks[3].Sigmoid();

            var newC = f.Mul(c).Add(i.Mul(g));
            var newH = o.Mul(newC.Tanh());

            return (newH, newC);
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"LSTMCell({_inputSize}, {_hiddenSize})";
    }

    /// <summary>
    /// GRU Cell - single step computation
    /// </summary>
    public class GRUCell : Module
    {
        private readonly int _inputSize;
        private readonly int _hiddenSize;
        private Tensor _weightIh;
        private Tensor _weightHh;
        private Tensor? _biasIh;
        private Tensor? _biasHh;

        /// <summary>Public API</summary>
        public GRUCell(int inputSize, int hiddenSize, bool bias = true)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;

            double stdv = 1.0 / Math.Sqrt(hiddenSize);
            _weightIh = Tensor.Uniform(new long[] { 3 * hiddenSize, inputSize }, -stdv, stdv);
            _weightHh = Tensor.Uniform(new long[] { 3 * hiddenSize, hiddenSize }, -stdv, stdv);
            RegisterParameter("weight_ih", _weightIh);
            RegisterParameter("weight_hh", _weightHh);

            if (bias)
            {
                _biasIh = Tensor.Uniform(new long[] { 3 * hiddenSize }, -stdv, stdv);
                _biasHh = Tensor.Uniform(new long[] { 3 * hiddenSize }, -stdv, stdv);
                RegisterParameter("bias_ih", _biasIh);
                RegisterParameter("bias_hh", _biasHh);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return Forward(input, null);
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor input, Tensor? h)
        {
            var batch = (int)input.Shape[0];
            h ??= Tensor.Zeros(new long[] { batch, _hiddenSize });

            var gatesX = TensorOps.MatMul(input, _weightIh.T());
            var gatesH = TensorOps.MatMul(h, _weightHh.T());

            if (_biasIh != null) gatesX = gatesX.Add(_biasIh);
            if (_biasHh != null) gatesH = gatesH.Add(_biasHh);

            var chunksX = TensorOps.Chunk(gatesX, 3, -1);
            var chunksH = TensorOps.Chunk(gatesH, 3, -1);

            var r = chunksX[0].Add(chunksH[0]).Sigmoid();
            var z = chunksX[1].Add(chunksH[1]).Sigmoid();
            var n = chunksX[2].Add(r.Mul(chunksH[2])).Tanh();

            return Tensor.Ones(z.Shape).Sub(z).Mul(n).Add(z.Mul(h));
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"GRUCell({_inputSize}, {_hiddenSize})";
    }

    /// <summary>
    /// RNN Cell - single step computation
    /// </summary>
    public class RNNCell : Module
    {
        private readonly int _inputSize;
        private readonly int _hiddenSize;
        private readonly string _nonlinearity;
        private Tensor _weightIh;
        private Tensor _weightHh;
        private Tensor? _biasIh;
        private Tensor? _biasHh;

        /// <summary>Public API</summary>
        public RNNCell(int inputSize, int hiddenSize, string nonlinearity = "tanh", bool bias = true)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;
            _nonlinearity = nonlinearity;

            double stdv = 1.0 / Math.Sqrt(hiddenSize);
            _weightIh = Tensor.Uniform(new long[] { hiddenSize, inputSize }, -stdv, stdv);
            _weightHh = Tensor.Uniform(new long[] { hiddenSize, hiddenSize }, -stdv, stdv);
            RegisterParameter("weight_ih", _weightIh);
            RegisterParameter("weight_hh", _weightHh);

            if (bias)
            {
                _biasIh = Tensor.Uniform(new long[] { hiddenSize }, -stdv, stdv);
                _biasHh = Tensor.Uniform(new long[] { hiddenSize }, -stdv, stdv);
                RegisterParameter("bias_ih", _biasIh);
                RegisterParameter("bias_hh", _biasHh);
            }
        }

        /// <summary>Public API</summary>
        public override Tensor Forward(Tensor input)
        {
            return Forward(input, null);
        }

        /// <summary>Public API</summary>
        public Tensor Forward(Tensor input, Tensor? h)
        {
            var batch = (int)input.Shape[0];
            h ??= Tensor.Zeros(new long[] { batch, _hiddenSize });

            var gates = TensorOps.MatMul(input, _weightIh.T()).Add(TensorOps.MatMul(h, _weightHh.T()));
            if (_biasIh != null) gates = gates.Add(_biasIh);
            if (_biasHh != null) gates = gates.Add(_biasHh);

            return _nonlinearity == "relu" ? gates.ReLU() : gates.Tanh();
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"RNNCell({_inputSize}, {_hiddenSize}, nonlinearity={_nonlinearity})";
    }
}