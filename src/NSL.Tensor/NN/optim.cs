using System;
using System.Collections.Generic;
using System.Linq;

namespace NSL.Tensor.NN
{
    /// <summary>
    /// Base class for all optimizers
    /// </summary>
    public abstract class Optimizer
    {
        protected readonly List<Tensor> _parameters;
        protected readonly double _lr;
        /// <summary>Current optimization step</summary>
        protected int _step;

        /// <summary>Public API</summary>
        public double LearningRate => _lr;
        /// <summary>Public API</summary>
        public int Step => _step;

        /// <summary>Initializes optimizer with parameters and learning rate</summary>
        /// <param name="parameters">Parameters to optimize</param>
        /// <param name="lr">Learning rate</param>
        protected Optimizer(IEnumerable<Tensor> parameters, double lr)
        {
            _parameters = parameters.ToList();
            _lr = lr;
            _step = 0;
        }

        /// <summary>
        /// Perform a single optimization step
        /// </summary>
        public abstract void StepOptimizer();

        /// <summary>
        /// Zero all gradients
        /// </summary>
        public virtual void ZeroGrad()
        {
            foreach (var param in _parameters)
                param.ZeroGrad();
        }

        /// <summary>
        /// Get current state dictionary
        /// </summary>
        public virtual Dictionary<string, object> StateDict()
        {
            return new Dictionary<string, object>
            {
                ["step"] = _step,
                ["lr"] = _lr
            };
        }

        /// <summary>
        /// Load state from dictionary
        /// </summary>
        public virtual void LoadStateDict(Dictionary<string, object> state)
        {
            if (state.ContainsKey("step"))
                _step = Convert.ToInt32(state["step"]);
        }
    }

    /// <summary>
    /// Stochastic Gradient Descent optimizer
    /// </summary>
    public class SGD : Optimizer
    {
        private readonly double _momentum;
        private readonly double _dampening;
        private readonly double _weightDecay;
        private readonly bool _nesterov;
        private readonly List<Tensor?> _velocities;

        /// <summary>Public API</summary>
        public SGD(IEnumerable<Tensor> parameters, double lr = 0.01, double momentum = 0, double dampening = 0,
            double weightDecay = 0, bool nesterov = false)
            : base(parameters, lr)
        {
            if (nesterov && (momentum <= 0 || dampening != 0))
                throw new ArgumentException("Nesterov momentum requires a momentum and zero dampening");

            _momentum = momentum;
            _dampening = dampening;
            _weightDecay = weightDecay;
            _nesterov = nesterov;
            _velocities = _parameters.Select(_ => (Tensor?)null).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Momentum
                if (_momentum != 0)
                {
                    if (_velocities[i] == null)
                    {
                        _velocities[i] = grad.Clone();
                    }
                    else
                    {
                        _velocities[i] = _velocities[i]!.Mul(_momentum).Add(grad.Mul(1 - _dampening));
                    }

                    if (_nesterov)
                        grad = grad.Add(_velocities[i]!.Mul(_momentum));
                    else
                        grad = _velocities[i]!;
                }

                // Update parameters
                param.SubInPlace(grad.Mul(_lr));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"SGD(lr={_lr}, momentum={_momentum}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// Adam optimizer - Adaptive Moment Estimation
    /// </summary>
    public class Adam : Optimizer
    {
        protected readonly double _beta1;
        protected readonly double _beta2;
        protected readonly double _eps;
        protected readonly double _weightDecay;
        protected readonly bool _amsgrad;
        protected readonly List<Tensor> _m; // First moment
        protected readonly List<Tensor> _v; // Second moment
        protected readonly List<Tensor?> _vMax; // Max second moment (AMSGrad)

        /// <summary>Public API</summary>
        public Adam(IEnumerable<Tensor> parameters, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-8, double weightDecay = 0, bool amsgrad = false)
            : base(parameters, lr)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
            _amsgrad = amsgrad;
            _m = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _v = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _vMax = amsgrad ? _parameters.Select(p => (Tensor?)Tensor.Zeros(p.Shape)).ToList() : _parameters.Select(_ => (Tensor?)null).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            double biasCorrection1 = 1 - Math.Pow(_beta1, _step);
            double biasCorrection2 = 1 - Math.Pow(_beta2, _step);

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay (L2 regularization for standard Adam)
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Update biased first moment estimate
                _m[i] = _m[i].Mul(_beta1).Add(grad.Mul(1 - _beta1));

                // Update biased second raw moment estimate
                _v[i] = _v[i].Mul(_beta2).Add(grad.Square().Mul(1 - _beta2));

                Tensor denom;
                if (_amsgrad)
                {
                    // Maintains the maximum of all 2nd moment running avg.
                    _vMax[i] = Tensor.Maximum(_vMax[i]!, _v[i]);
                    denom = _vMax[i]!.Sqrt().Div(Math.Sqrt(biasCorrection2)).Add(_eps);
                }
                else
                {
                    denom = _v[i].Sqrt().Div(Math.Sqrt(biasCorrection2)).Add(_eps);
                }

                var stepSize = _lr / biasCorrection1;
                param.SubInPlace(_m[i].Div(denom).Mul(stepSize));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Adam(lr={_lr}, betas=({_beta1}, {_beta2}), eps={_eps}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// AdamW optimizer - Adam with decoupled weight decay
    /// </summary>
    public class AdamW : Optimizer
    {
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;
        private readonly bool _amsgrad;
        private readonly List<Tensor> _m;
        private readonly List<Tensor> _v;
        private readonly List<Tensor?> _vMax;

        /// <summary>Public API</summary>
        public AdamW(IEnumerable<Tensor> parameters, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-8, double weightDecay = 0.01, bool amsgrad = false)
            : base(parameters, lr)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
            _amsgrad = amsgrad;
            _m = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _v = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _vMax = amsgrad ? _parameters.Select(p => (Tensor?)Tensor.Zeros(p.Shape)).ToList() : _parameters.Select(_ => (Tensor?)null).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            double biasCorrection1 = 1 - Math.Pow(_beta1, _step);
            double biasCorrection2 = 1 - Math.Pow(_beta2, _step);

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Decoupled weight decay - applied directly to weights
                if (_weightDecay != 0)
                    param.SubInPlace(param.Mul(_lr * _weightDecay));

                // Update biased first moment estimate
                _m[i] = _m[i].Mul(_beta1).Add(grad.Mul(1 - _beta1));

                // Update biased second raw moment estimate
                _v[i] = _v[i].Mul(_beta2).Add(grad.Square().Mul(1 - _beta2));

                Tensor denom;
                if (_amsgrad)
                {
                    _vMax[i] = Tensor.Maximum(_vMax[i]!, _v[i]);
                    denom = _vMax[i]!.Sqrt().Div(Math.Sqrt(biasCorrection2)).Add(_eps);
                }
                else
                {
                    denom = _v[i].Sqrt().Div(Math.Sqrt(biasCorrection2)).Add(_eps);
                }

                var stepSize = _lr / biasCorrection1;
                param.SubInPlace(_m[i].Div(denom).Mul(stepSize));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"AdamW(lr={_lr}, betas=({_beta1}, {_beta2}), eps={_eps}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// RMSprop optimizer
    /// </summary>
    public class RMSprop : Optimizer
    {
        private readonly double _alpha;
        private readonly double _eps;
        private readonly double _weightDecay;
        private readonly double _momentum;
        private readonly bool _centered;
        private readonly List<Tensor> _squareAvg;
        private readonly List<Tensor?> _gradAvg;
        private readonly List<Tensor?> _momentumBuffer;

        /// <summary>Public API</summary>
        public RMSprop(IEnumerable<Tensor> parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8,
            double weightDecay = 0, double momentum = 0, bool centered = false)
            : base(parameters, lr)
        {
            _alpha = alpha;
            _eps = eps;
            _weightDecay = weightDecay;
            _momentum = momentum;
            _centered = centered;
            _squareAvg = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _gradAvg = centered ? _parameters.Select(p => (Tensor?)Tensor.Zeros(p.Shape)).ToList() : _parameters.Select(_ => (Tensor?)null).ToList();
            _momentumBuffer = momentum > 0 ? _parameters.Select(p => (Tensor?)Tensor.Zeros(p.Shape)).ToList() : _parameters.Select(_ => (Tensor?)null).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Update running average of squared gradients
                _squareAvg[i] = _squareAvg[i].Mul(_alpha).Add(grad.Square().Mul(1 - _alpha));

                Tensor avg;
                if (_centered)
                {
                    _gradAvg[i] = _gradAvg[i]!.Mul(_alpha).Add(grad.Mul(1 - _alpha));
                    avg = _squareAvg[i].Sub(_gradAvg[i]!.Square()).Sqrt().Add(_eps);
                }
                else
                {
                    avg = _squareAvg[i].Sqrt().Add(_eps);
                }

                if (_momentum > 0)
                {
                    _momentumBuffer[i] = _momentumBuffer[i]!.Mul(_momentum).Add(grad.Div(avg));
                    param.SubInPlace(_momentumBuffer[i]!.Mul(_lr));
                }
                else
                {
                    param.SubInPlace(grad.Div(avg).Mul(_lr));
                }
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"RMSprop(lr={_lr}, alpha={_alpha}, eps={_eps}, weight_decay={_weightDecay}, momentum={_momentum})";
    }

    /// <summary>
    /// Adagrad optimizer
    /// </summary>
    public class Adagrad : Optimizer
    {
        private readonly double _lrDecay;
        private readonly double _weightDecay;
        private readonly double _eps;
        private readonly List<Tensor> _sumSquares;

        /// <summary>Public API</summary>
        public Adagrad(IEnumerable<Tensor> parameters, double lr = 0.01, double lrDecay = 0,
            double weightDecay = 0, double eps = 1e-10)
            : base(parameters, lr)
        {
            _lrDecay = lrDecay;
            _weightDecay = weightDecay;
            _eps = eps;
            _sumSquares = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            double lr = _lr / (1 + (_step - 1) * _lrDecay);

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Accumulate squared gradients
                _sumSquares[i] = _sumSquares[i].Add(grad.Square());

                // Update parameters
                var std = _sumSquares[i].Sqrt().Add(_eps);
                param.SubInPlace(grad.Div(std).Mul(lr));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Adagrad(lr={_lr}, lr_decay={_lrDecay}, weight_decay={_weightDecay}, eps={_eps})";
    }

    /// <summary>
    /// Adadelta optimizer
    /// </summary>
    public class Adadelta : Optimizer
    {
        private readonly double _rho;
        private readonly double _eps;
        private readonly double _weightDecay;
        private readonly List<Tensor> _squareAvg;
        private readonly List<Tensor> _accDelta;

        /// <summary>Public API</summary>
        public Adadelta(IEnumerable<Tensor> parameters, double lr = 1.0, double rho = 0.9,
            double eps = 1e-6, double weightDecay = 0)
            : base(parameters, lr)
        {
            _rho = rho;
            _eps = eps;
            _weightDecay = weightDecay;
            _squareAvg = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _accDelta = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Update running average of squared gradients
                _squareAvg[i] = _squareAvg[i].Mul(_rho).Add(grad.Square().Mul(1 - _rho));

                // Compute update
                var std = _squareAvg[i].Add(_eps).Sqrt();
                var delta = _accDelta[i].Add(_eps).Sqrt().Div(std).Mul(grad);

                // Update accumulated delta
                _accDelta[i] = _accDelta[i].Mul(_rho).Add(delta.Square().Mul(1 - _rho));

                // Update parameters
                param.SubInPlace(delta.Mul(_lr));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Adadelta(lr={_lr}, rho={_rho}, eps={_eps}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// Adamax optimizer (variant of Adam based on infinity norm)
    /// </summary>
    public class Adamax : Optimizer
    {
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;
        private readonly List<Tensor> _m;
        private readonly List<Tensor> _u;

        /// <summary>Public API</summary>
        public Adamax(IEnumerable<Tensor> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-8, double weightDecay = 0)
            : base(parameters, lr)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
            _m = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _u = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            double biasCorrection1 = 1 - Math.Pow(_beta1, _step);

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Update biased first moment estimate
                _m[i] = _m[i].Mul(_beta1).Add(grad.Mul(1 - _beta1));

                // Update exponentially weighted infinity norm
                _u[i] = Tensor.Maximum(_u[i].Mul(_beta2), grad.Abs());

                // Update parameters
                var stepSize = _lr / biasCorrection1;
                param.SubInPlace(_m[i].Div(_u[i].Add(_eps)).Mul(stepSize));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Adamax(lr={_lr}, betas=({_beta1}, {_beta2}), eps={_eps}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// ASGD optimizer (Averaged Stochastic Gradient Descent)
    /// </summary>
    public class ASGD : Optimizer
    {
        private readonly double _lambd;
        private readonly double _alpha;
        private readonly double _t0;
        private readonly double _weightDecay;
        private readonly List<Tensor> _ax;
        private readonly List<double> _mu;
        private readonly List<double> _eta;

        /// <summary>Public API</summary>
        public ASGD(IEnumerable<Tensor> parameters, double lr = 0.01, double lambd = 0.0001,
            double alpha = 0.75, double t0 = 1e6, double weightDecay = 0)
            : base(parameters, lr)
        {
            _lambd = lambd;
            _alpha = alpha;
            _t0 = t0;
            _weightDecay = weightDecay;
            _ax = _parameters.Select(p => p.Clone()).ToList();
            _mu = _parameters.Select(_ => 1.0).ToList();
            _eta = _parameters.Select(_ => lr).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Update eta
                _eta[i] = _lr / Math.Pow(1 + _lambd * _lr * _step, _alpha);

                // Update parameters
                param.SubInPlace(grad.Mul(_eta[i]));

                // Update average
                if (_step > _t0)
                {
                    _mu[i] = 1.0 / Math.Max(1, _step - _t0);
                    _ax[i] = _ax[i].Add(param.Sub(_ax[i]).Mul(_mu[i]));
                }
            }
        }

        /// <summary>
        /// Get averaged parameters
        /// </summary>
        public IEnumerable<Tensor> AveragedParameters() => _ax;

        /// <summary>Public API</summary>
        public override string ToString() => $"ASGD(lr={_lr}, lambd={_lambd}, alpha={_alpha}, t0={_t0}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// RAdam optimizer (Rectified Adam)
    /// </summary>
    public class RAdam : Optimizer
    {
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;
        private readonly List<Tensor> _m;
        private readonly List<Tensor> _v;

        /// <summary>Public API</summary>
        public RAdam(IEnumerable<Tensor> parameters, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-8, double weightDecay = 0)
            : base(parameters, lr)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
            _m = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _v = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            double beta1T = Math.Pow(_beta1, _step);
            double beta2T = Math.Pow(_beta2, _step);

            // Maximum length of approximated SMA
            double rhoInf = 2.0 / (1 - _beta2) - 1;
            // Compute SMA
            double rhoT = rhoInf - 2 * _step * beta2T / (1 - beta2T);

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    grad = grad.Add(param.Mul(_weightDecay));

                // Update biased first moment estimate
                _m[i] = _m[i].Mul(_beta1).Add(grad.Mul(1 - _beta1));

                // Update biased second raw moment estimate
                _v[i] = _v[i].Mul(_beta2).Add(grad.Square().Mul(1 - _beta2));

                // Bias corrected first moment
                var mHat = _m[i].Div(1 - beta1T);

                if (rhoT > 5)
                {
                    // Variance is tractable
                    var rect = Math.Sqrt((rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT));
                    var vHat = _v[i].Div(1 - beta2T);
                    param.SubInPlace(mHat.Div(vHat.Sqrt().Add(_eps)).Mul(_lr * rect));
                }
                else
                {
                    // Variance is intractable, use unadapted update
                    param.SubInPlace(mHat.Mul(_lr));
                }
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"RAdam(lr={_lr}, betas=({_beta1}, {_beta2}), eps={_eps}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// NAdam optimizer (Nesterov-accelerated Adam)
    /// </summary>
    public class NAdam : Optimizer
    {
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;
        private readonly double _momentumDecay;
        private readonly List<Tensor> _m;
        private readonly List<Tensor> _v;
        private double _muProduct;

        /// <summary>Public API</summary>
        public NAdam(IEnumerable<Tensor> parameters, double lr = 0.002, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-8, double weightDecay = 0, double momentumDecay = 0.004)
            : base(parameters, lr)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
            _momentumDecay = momentumDecay;
            _m = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _v = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _muProduct = 1.0;
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            double muT = _beta1 * (1 - 0.5 * Math.Pow(0.96, _step * _momentumDecay));
            double muT1 = _beta1 * (1 - 0.5 * Math.Pow(0.96, (_step + 1) * _momentumDecay));
            _muProduct *= muT;
            double muProduct1 = _muProduct * muT1;

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;

                // Weight decay
                if (_weightDecay != 0)
                    param.SubInPlace(param.Mul(_lr * _weightDecay));

                // Update biased first moment estimate
                _m[i] = _m[i].Mul(_beta1).Add(grad.Mul(1 - _beta1));

                // Update biased second raw moment estimate
                _v[i] = _v[i].Mul(_beta2).Add(grad.Square().Mul(1 - _beta2));

                // Bias corrected estimates
                var mHat = _m[i].Div(1 - _muProduct).Mul(muT1).Add(grad.Mul(1 - muT).Div(1 - _muProduct));
                var vHat = _v[i].Div(1 - Math.Pow(_beta2, _step));

                // Update parameters
                param.SubInPlace(mHat.Div(vHat.Sqrt().Add(_eps)).Mul(_lr));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"NAdam(lr={_lr}, betas=({_beta1}, {_beta2}), eps={_eps}, weight_decay={_weightDecay})";
    }

    /// <summary>
    /// Rprop optimizer (Resilient backpropagation)
    /// </summary>
    public class Rprop : Optimizer
    {
        private readonly double _etaPlus;
        private readonly double _etaMinus;
        private readonly double _stepMin;
        private readonly double _stepMax;
        private readonly List<Tensor> _prevGrad;
        private readonly List<Tensor> _stepSizes;

        /// <summary>Public API</summary>
        public Rprop(IEnumerable<Tensor> parameters, double lr = 0.01, double etaPlus = 1.2, double etaMinus = 0.5,
            double stepMin = 1e-6, double stepMax = 50)
            : base(parameters, lr)
        {
            _etaPlus = etaPlus;
            _etaMinus = etaMinus;
            _stepMin = stepMin;
            _stepMax = stepMax;
            _prevGrad = _parameters.Select(p => Tensor.Zeros(p.Shape)).ToList();
            _stepSizes = _parameters.Select(p => Tensor.Full(p.Shape, lr)).ToList();
        }

        /// <summary>Public API</summary>
        public override void StepOptimizer()
        {
            _step++;

            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];
                if (param.Grad == null) continue;

                var grad = param.Grad;
                var gradData = grad.ToArray();
                var prevGradData = _prevGrad[i].ToArray();
                var stepData = _stepSizes[i].ToArray();
                var paramData = param.ToArray();

                for (int j = 0; j < gradData.Length; j++)
                {
                    double sign = gradData[j] * prevGradData[j];

                    if (sign > 0)
                    {
                        stepData[j] = Math.Min(stepData[j] * _etaPlus, _stepMax);
                        paramData[j] -= Math.Sign(gradData[j]) * stepData[j];
                        prevGradData[j] = gradData[j];
                    }
                    else if (sign < 0)
                    {
                        stepData[j] = Math.Max(stepData[j] * _etaMinus, _stepMin);
                        prevGradData[j] = 0;
                    }
                    else
                    {
                        paramData[j] -= Math.Sign(gradData[j]) * stepData[j];
                        prevGradData[j] = gradData[j];
                    }
                }

                _stepSizes[i] = Tensor.FromArray(stepData, _stepSizes[i].Shape);
                _prevGrad[i] = Tensor.FromArray(prevGradData, _prevGrad[i].Shape);
                param.CopyFrom(Tensor.FromArray(paramData, param.Shape));
            }
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"Rprop(lr={_lr}, etas=({_etaMinus}, {_etaPlus}), step_sizes=({_stepMin}, {_stepMax}))";
    }

    #region Learning Rate Schedulers

    /// <summary>
    /// Base class for learning rate schedulers
    /// </summary>
    public abstract class LRScheduler
    {
        protected readonly Optimizer _optimizer;
        protected int _lastEpoch;
        protected double _baseLr;

        protected LRScheduler(Optimizer optimizer, int lastEpoch = -1)
        {
            _optimizer = optimizer;
            _lastEpoch = lastEpoch;
            _baseLr = optimizer.LearningRate;
        }

        /// <summary>Public API</summary>
        public abstract double GetLr();

        /// <summary>Public API</summary>
        public virtual void Step(int? epoch = null)
        {
            _lastEpoch = epoch ?? _lastEpoch + 1;
        }
    }

    /// <summary>
    /// Step LR scheduler - decays LR by gamma every step_size epochs
    /// </summary>
    public class StepLR : LRScheduler
    {
        private readonly int _stepSize;
        private readonly double _gamma;

        /// <summary>Public API</summary>
        public StepLR(Optimizer optimizer, int stepSize, double gamma = 0.1, int lastEpoch = -1)
            : base(optimizer, lastEpoch)
        {
            _stepSize = stepSize;
            _gamma = gamma;
        }

        /// <summary>Public API</summary>
        public override double GetLr()
        {
            return _baseLr * Math.Pow(_gamma, _lastEpoch / _stepSize);
        }
    }

    /// <summary>
    /// Multi-step LR scheduler - decays LR by gamma at specified milestones
    /// </summary>
    public class MultiStepLR : LRScheduler
    {
        private readonly int[] _milestones;
        private readonly double _gamma;

        /// <summary>Public API</summary>
        public MultiStepLR(Optimizer optimizer, int[] milestones, double gamma = 0.1, int lastEpoch = -1)
            : base(optimizer, lastEpoch)
        {
            _milestones = milestones.OrderBy(x => x).ToArray();
            _gamma = gamma;
        }

        /// <summary>Public API</summary>
        public override double GetLr()
        {
            int count = _milestones.Count(m => m <= _lastEpoch);
            return _baseLr * Math.Pow(_gamma, count);
        }
    }

    /// <summary>
    /// Exponential LR scheduler
    /// </summary>
    public class ExponentialLR : LRScheduler
    {
        private readonly double _gamma;

        /// <summary>Public API</summary>
        public ExponentialLR(Optimizer optimizer, double gamma, int lastEpoch = -1)
            : base(optimizer, lastEpoch)
        {
            _gamma = gamma;
        }

        /// <summary>Public API</summary>
        public override double GetLr()
        {
            return _baseLr * Math.Pow(_gamma, _lastEpoch);
        }
    }

    /// <summary>
    /// Cosine Annealing LR scheduler
    /// </summary>
    public class CosineAnnealingLR : LRScheduler
    {
        private readonly int _tMax;
        private readonly double _etaMin;

        /// <summary>Public API</summary>
        public CosineAnnealingLR(Optimizer optimizer, int tMax, double etaMin = 0, int lastEpoch = -1)
            : base(optimizer, lastEpoch)
        {
            _tMax = tMax;
            _etaMin = etaMin;
        }

        /// <summary>Public API</summary>
        public override double GetLr()
        {
            return _etaMin + (_baseLr - _etaMin) * (1 + Math.Cos(Math.PI * _lastEpoch / _tMax)) / 2;
        }
    }

    /// <summary>
    /// Cosine Annealing with Warm Restarts
    /// </summary>
    public class CosineAnnealingWarmRestarts : LRScheduler
    {
        private readonly int _t0;
        private readonly int _tMult;
        private readonly double _etaMin;
        private int _tCur;
        private int _tI;

        /// <summary>Public API</summary>
        public CosineAnnealingWarmRestarts(Optimizer optimizer, int t0, int tMult = 1, double etaMin = 0, int lastEpoch = -1)
            : base(optimizer, lastEpoch)
        {
            _t0 = t0;
            _tMult = tMult;
            _etaMin = etaMin;
            _tCur = 0;
            _tI = t0;
        }

        /// <summary>Public API</summary>
        public override double GetLr()
        {
            return _etaMin + (_baseLr - _etaMin) * (1 + Math.Cos(Math.PI * _tCur / _tI)) / 2;
        }

        /// <summary>Public API</summary>
        public override void Step(int? epoch = null)
        {
            if (epoch == null)
            {
                _tCur++;
                if (_tCur >= _tI)
                {
                    _tCur = 0;
                    _tI = (int)(_tI * _tMult);
                }
            }
            else
            {
                _lastEpoch = epoch.Value;
                // Calculate which restart we're in
                int cumT = 0;
                _tI = _t0;
                while (cumT + _tI <= _lastEpoch)
                {
                    cumT += _tI;
                    _tI = (int)(_tI * _tMult);
                }
                _tCur = _lastEpoch - cumT;
            }
        }
    }

    /// <summary>
    /// Reduce LR on Plateau - reduces LR when a metric has stopped improving
    /// </summary>
    public class ReduceLROnPlateau
    {
        private readonly Optimizer _optimizer;
        private readonly string _mode;
        private readonly double _factor;
        private readonly int _patience;
        private readonly double _threshold;
        private readonly int _cooldown;
        private readonly double _minLr;
        private double _currentLr;
        private double _bestValue;
        private int _badEpochs;
        private int _cooldownCounter;

        /// <summary>Public API</summary>
        public ReduceLROnPlateau(Optimizer optimizer, string mode = "min", double factor = 0.1, int patience = 10,
            double threshold = 1e-4, int cooldown = 0, double minLr = 0)
        {
            _optimizer = optimizer;
            _mode = mode;
            _factor = factor;
            _patience = patience;
            _threshold = threshold;
            _cooldown = cooldown;
            _minLr = minLr;
            _currentLr = optimizer.LearningRate;
            _bestValue = mode == "min" ? double.MaxValue : double.MinValue;
            _badEpochs = 0;
            _cooldownCounter = 0;
        }

        /// <summary>Public API</summary>
        public void Step(double metric)
        {
            bool isImprovement = _mode == "min"
                ? metric < _bestValue - _threshold
                : metric > _bestValue + _threshold;

            if (isImprovement)
            {
                _bestValue = metric;
                _badEpochs = 0;
            }
            else
            {
                _badEpochs++;
            }

            if (_cooldownCounter > 0)
            {
                _cooldownCounter--;
                _badEpochs = 0;
            }

            if (_badEpochs > _patience)
            {
                _currentLr = Math.Max(_currentLr * _factor, _minLr);
                _cooldownCounter = _cooldown;
                _badEpochs = 0;
            }
        }

        /// <summary>Public API</summary>
        public double CurrentLr => _currentLr;
    }

    /// <summary>
    /// One Cycle LR scheduler
    /// </summary>
    public class OneCycleLR : LRScheduler
    {
        private readonly double _maxLr;
        private readonly int _totalSteps;
        private readonly double _pctStart;
        private readonly double _divFactor;
        private readonly double _finalDivFactor;
        private int _currentStep;

        /// <summary>Public API</summary>
        public OneCycleLR(Optimizer optimizer, double maxLr, int totalSteps, double pctStart = 0.3,
            double divFactor = 25, double finalDivFactor = 1e4, int lastEpoch = -1)
            : base(optimizer, lastEpoch)
        {
            _maxLr = maxLr;
            _totalSteps = totalSteps;
            _pctStart = pctStart;
            _divFactor = divFactor;
            _finalDivFactor = finalDivFactor;
            _currentStep = 0;
        }

        /// <summary>Public API</summary>
        public override double GetLr()
        {
            double initialLr = _maxLr / _divFactor;
            double minLr = _maxLr / _finalDivFactor;

            int warmupSteps = (int)(_totalSteps * _pctStart);

            if (_currentStep < warmupSteps)
            {
                // Warmup phase: linear increase
                double pct = _currentStep / (double)warmupSteps;
                return initialLr + (maxLr - initialLr) * pct;
            }
            else
            {
                // Annealing phase: cosine decay
                double pct = (_currentStep - warmupSteps) / (double)(_totalSteps - warmupSteps);
                return minLr + (_maxLr - minLr) * (1 + Math.Cos(Math.PI * pct)) / 2;
            }
        }

        private double maxLr => _maxLr;

        /// <summary>Public API</summary>
        public override void Step(int? epoch = null)
        {
            _currentStep++;
            base.Step(epoch);
        }
    }

    /// <summary>
    /// Linear warmup then linear decay scheduler
    /// </summary>
    public class LinearWarmupScheduler : LRScheduler
    {
        private readonly int _warmupSteps;
        private readonly int _totalSteps;
        private int _currentStep;

        /// <summary>Public API</summary>
        public LinearWarmupScheduler(Optimizer optimizer, int warmupSteps, int totalSteps, int lastEpoch = -1)
            : base(optimizer, lastEpoch)
        {
            _warmupSteps = warmupSteps;
            _totalSteps = totalSteps;
            _currentStep = 0;
        }

        /// <summary>Public API</summary>
        public override double GetLr()
        {
            if (_currentStep < _warmupSteps)
            {
                return _baseLr * _currentStep / _warmupSteps;
            }
            else
            {
                double progress = (_currentStep - _warmupSteps) / (double)(_totalSteps - _warmupSteps);
                return _baseLr * Math.Max(0, 1 - progress);
            }
        }

        /// <summary>Public API</summary>
        public override void Step(int? epoch = null)
        {
            _currentStep++;
            base.Step(epoch);
        }
    }

    #endregion
}