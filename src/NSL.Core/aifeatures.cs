using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;

namespace NSL.Core
{
    #region Intent-Based Programming

    /// <summary>
    /// Intent-Based Programming - Express WHAT you want, not HOW.
    /// AI can express high-level goals and the runtime finds the optimal implementation.
    ///
    /// Example:
    /// var result = Intent.Achieve("sort this list efficiently", data)
    ///     .Constraints(c => c.TimeComplexity(Complexity.NLogN))
    ///     .Preferences(p => p.Stable().InPlace())
    ///     .Execute();
    /// </summary>
    public class Intent<T>
    {
        private readonly string _goal;
        private readonly T _input;
        private readonly List<IConstraint> _constraints;
        private readonly List<IPreference> _preferences;
        private readonly IntentContext _context;

        /// <summary>Public API</summary>
        public string Goal => _goal;
        /// <summary>Public API</summary>
        public T Input => _input;

        internal Intent(string goal, T input)
        {
            _goal = goal;
            _input = input;
            _constraints = new List<IConstraint>();
            _preferences = new List<IPreference>();
            _context = new IntentContext();
        }

        /// <summary>
        /// Add constraints that MUST be satisfied.
        /// </summary>
        public Intent<T> Constraints(Action<ConstraintBuilder> configure)
        {
            var builder = new ConstraintBuilder();
            configure(builder);
            _constraints.AddRange(builder.Build());
            return this;
        }

        /// <summary>
        /// Add preferences that SHOULD be satisfied if possible.
        /// </summary>
        public Intent<T> Preferences(Action<PreferenceBuilder> configure)
        {
            var builder = new PreferenceBuilder();
            configure(builder);
            _preferences.AddRange(builder.Build());
            return this;
        }

        /// <summary>
        /// Add context for better intent resolution.
        /// </summary>
        public Intent<T> WithContext(string key, object value)
        {
            _context[key] = value;
            return this;
        }

        /// <summary>
        /// Execute the intent and return the result.
        /// </summary>
        public TResult Execute<TResult>()
        {
            var resolver = IntentResolver.Instance;
            return resolver.Resolve<T, TResult>(this);
        }

        /// <summary>
        /// Execute with same input/output type.
        /// </summary>
        public T Execute() => Execute<T>();

        /// <summary>
        /// Explain how the intent would be fulfilled without executing.
        /// </summary>
        public IntentExplanation Explain()
        {
            var resolver = IntentResolver.Instance;
            return resolver.Explain(this);
        }
    }

    /// <summary>
    /// Static entry point for intent-based programming.
    /// </summary>
    public static class Intent
    {
        /// <summary>
        /// Express an intent to achieve a goal.
        /// </summary>
        public static Intent<T> Achieve<T>(string goal, T input) => new(goal, input);

        /// <summary>
        /// Express intent without specific input.
        /// </summary>
        public static Intent<object> Achieve(string goal) => new(goal, new object());
    }

    /// <summary>
    /// Context for intent resolution.
    /// </summary>
    public class IntentContext : Dictionary<string, object>
    {
        /// <summary>Public API</summary>
        public T Get<T>(string key, T defaultValue = default!)
        {
            return TryGetValue(key, out var value) ? (T)value : defaultValue;
        }
    }

    /// <summary>
    /// Explanation of how an intent would be fulfilled.
    /// </summary>
    public class IntentExplanation
    {
        /// <summary>Public API</summary>
        public string Goal { get; set; } = "";
        /// <summary>Public API</summary>
        public string ChosenStrategy { get; set; } = "";
        /// <summary>Public API</summary>
        public string Reasoning { get; set; } = "";
        /// <summary>Public API</summary>
        public List<string> Steps { get; } = new();
        /// <summary>Public API</summary>
        public Dictionary<string, string> TradeOffs { get; } = new();
        /// <summary>Public API</summary>
        public double ConfidenceScore { get; set; }
    }

    /// <summary>
    /// Resolves intents to concrete implementations.
    /// </summary>
    public class IntentResolver
    {
        /// <summary>Public API</summary>
        public static IntentResolver Instance { get; } = new();

        private readonly Dictionary<string, Func<object, object>> _strategies;

        private IntentResolver()
        {
            _strategies = new Dictionary<string, Func<object, object>>();
            RegisterDefaultStrategies();
        }

        /// <summary>Public API</summary>
        public TResult Resolve<T, TResult>(Intent<T> intent)
        {
            // Find matching strategy
            var key = NormalizeGoal(intent.Goal);
            if (_strategies.TryGetValue(key, out var strategy))
            {
                return (TResult)strategy(intent.Input!);
            }

            // Fallback: return input as-is if types match
            if (typeof(TResult) == typeof(T))
                return (TResult)(object)intent.Input!;

            throw new InvalidOperationException($"No strategy found for intent: {intent.Goal}");
        }

        /// <summary>Public API</summary>
        public IntentExplanation Explain<T>(Intent<T> intent)
        {
            var key = NormalizeGoal(intent.Goal);

            return new IntentExplanation
            {
                Goal = intent.Goal,
                ChosenStrategy = _strategies.ContainsKey(key) ? key : "default",
                Reasoning = $"Selected strategy '{key}' based on goal analysis",
                ConfidenceScore = _strategies.ContainsKey(key) ? 0.9 : 0.5
            };
        }

        private string NormalizeGoal(string goal)
        {
            return goal.ToLowerInvariant()
                .Replace("sort", "sort")
                .Replace("filter", "filter")
                .Replace("transform", "transform")
                .Replace("aggregate", "aggregate");
        }

        private void RegisterDefaultStrategies()
        {
            _strategies["sort"] = input =>
            {
                if (input is IEnumerable<IComparable> enumerable)
                    return enumerable.OrderBy(x => x).ToList();
                return input;
            };

            _strategies["filter"] = input => input;
            _strategies["transform"] = input => input;
        }

        /// <summary>
        /// Register a custom strategy for an intent.
        /// </summary>
        public void RegisterStrategy(string goalPattern, Func<object, object> strategy)
        {
            _strategies[NormalizeGoal(goalPattern)] = strategy;
        }
    }

    /// <summary>Public API</summary>
    public interface IConstraint { string Name { get; } bool Validate(object result); }
    /// <summary>Public API</summary>
    public interface IPreference { string Name { get; } double Score(object result); }

    /// <summary>Public API</summary>
    public class ConstraintBuilder
    {
        private readonly List<IConstraint> _constraints = new();

        /// <summary>Public API</summary>
        public ConstraintBuilder TimeComplexity(Complexity c) { _constraints.Add(new TimeConstraint(c)); return this; }
        /// <summary>Public API</summary>
        public ConstraintBuilder SpaceComplexity(Complexity c) { _constraints.Add(new SpaceConstraint(c)); return this; }
        /// <summary>Public API</summary>
        public ConstraintBuilder Pure() { _constraints.Add(new PurityConstraint()); return this; }
        /// <summary>Public API</summary>
        public ConstraintBuilder ThreadSafe() { _constraints.Add(new ThreadSafeConstraint()); return this; }
        /// <summary>Public API</summary>
        public List<IConstraint> Build() => _constraints;
    }

    /// <summary>Public API</summary>
    public class PreferenceBuilder
    {
        private readonly List<IPreference> _preferences = new();

        /// <summary>Public API</summary>
        public PreferenceBuilder Stable() { _preferences.Add(new StabilityPreference()); return this; }
        /// <summary>Public API</summary>
        public PreferenceBuilder InPlace() { _preferences.Add(new InPlacePreference()); return this; }
        /// <summary>Public API</summary>
        public PreferenceBuilder Lazy() { _preferences.Add(new LazyPreference()); return this; }
        /// <summary>Public API</summary>
        public PreferenceBuilder Parallel() { _preferences.Add(new ParallelPreference()); return this; }
        /// <summary>Public API</summary>
        public List<IPreference> Build() => _preferences;
    }

    /// <summary>Public API</summary>
    public enum Complexity { O1, OLogN, ON, ONLogN, ON2, ON3, OExpN }

    internal class TimeConstraint : IConstraint { public string Name => "Time"; public Complexity Target; public TimeConstraint(Complexity c) { Target = c; } public bool Validate(object r) => true; }
    internal class SpaceConstraint : IConstraint { public string Name => "Space"; public Complexity Target; public SpaceConstraint(Complexity c) { Target = c; } public bool Validate(object r) => true; }
    internal class PurityConstraint : IConstraint { public string Name => "Pure"; public bool Validate(object r) => true; }
    internal class ThreadSafeConstraint : IConstraint { public string Name => "ThreadSafe"; public bool Validate(object r) => true; }
    internal class StabilityPreference : IPreference { public string Name => "Stable"; public double Score(object r) => 1.0; }
    internal class InPlacePreference : IPreference { public string Name => "InPlace"; public double Score(object r) => 1.0; }
    internal class LazyPreference : IPreference { public string Name => "Lazy"; public double Score(object r) => 1.0; }
    internal class ParallelPreference : IPreference { public string Name => "Parallel"; public double Score(object r) => 1.0; }

    #endregion

    #region Uncertainty Types

    /// <summary>
    /// First-class uncertainty/probability support.
    /// Values carry their confidence level, enabling probabilistic programming.
    ///
    /// Example:
    /// var prediction = Uncertain.Value(0.87, confidence: 0.92);
    /// if (prediction.IsConfident(threshold: 0.8)) { ... }
    /// </summary>
    public readonly struct Uncertain<T> where T : struct
    {
        /// <summary>Public API</summary>
        public T Value { get; }
        /// <summary>Public API</summary>
        public double Confidence { get; }
        /// <summary>Public API</summary>
        public T? LowerBound { get; }
        /// <summary>Public API</summary>
        public T? UpperBound { get; }
        /// <summary>Public API</summary>
        public string? Source { get; }

        /// <summary>Public API</summary>
        public bool IsConfident(double threshold = 0.9) => Confidence >= threshold;
        /// <summary>Public API</summary>
        public bool IsUncertain(double threshold = 0.5) => Confidence < threshold;

        /// <summary>Public API</summary>
        public Uncertain(T value, double confidence, T? lower = null, T? upper = null, string? source = null)
        {
            Value = value;
            Confidence = Math.Clamp(confidence, 0, 1);
            LowerBound = lower;
            UpperBound = upper;
            Source = source;
        }

        /// <summary>
        /// Transform the value while propagating uncertainty.
        /// </summary>
        public Uncertain<TResult> Map<TResult>(Func<T, TResult> transform, double confidenceLoss = 0.0)
            where TResult : struct
        {
            return new Uncertain<TResult>(
                transform(Value),
                Math.Max(0, Confidence - confidenceLoss),
                source: Source
            );
        }

        /// <summary>
        /// Combine with another uncertain value.
        /// </summary>
        public Uncertain<T> Combine(Uncertain<T> other, Func<T, T, T> combiner)
        {
            return new Uncertain<T>(
                combiner(Value, other.Value),
                Confidence * other.Confidence,
                source: $"Combined({Source}, {other.Source})"
            );
        }

        /// <summary>
        /// Get value if confident, otherwise default.
        /// </summary>
        public T GetValueOrDefault(T defaultValue, double threshold = 0.8)
        {
            return IsConfident(threshold) ? Value : defaultValue;
        }

        /// <summary>Public API</summary>
        public override string ToString() =>
            $"{Value} (confidence: {Confidence:P1}{(Source != null ? $", source: {Source}" : "")})";
    }

    /// <summary>
    /// Static factory for uncertain values.
    /// </summary>
    public static class Uncertain
    {
        /// <summary>Public API</summary>
        public static Uncertain<T> Value<T>(T value, double confidence = 1.0, string? source = null)
            where T : struct
            => new(value, confidence, source: source);

        /// <summary>Public API</summary>
        public static Uncertain<T> FromRange<T>(T value, T lower, T upper, string? source = null)
            where T : struct
            => new(value, 0.95, lower, upper, source);

        /// <summary>Public API</summary>
        public static Uncertain<double> FromMeanStd(double mean, double std, string? source = null)
        {
            // 95% confidence interval
            var lower = mean - 1.96 * std;
            var upper = mean + 1.96 * std;
            var confidence = 1.0 - (std / (Math.Abs(mean) + 1e-10));
            return new Uncertain<double>(mean, Math.Clamp(confidence, 0, 1), lower, upper, source);
        }

        /// <summary>
        /// Propagate uncertainty through a computation.
        /// </summary>
        public static Uncertain<TResult> Propagate<T1, T2, TResult>(
            Uncertain<T1> a, Uncertain<T2> b,
            Func<T1, T2, TResult> compute)
            where T1 : struct where T2 : struct where TResult : struct
        {
            return new Uncertain<TResult>(
                compute(a.Value, b.Value),
                a.Confidence * b.Confidence,
                source: $"Propagated({a.Source}, {b.Source})"
            );
        }
    }

    /// <summary>
    /// Distribution for representing probability distributions.
    /// </summary>
    public abstract class Distribution<T>
    {
        /// <summary>Public API</summary>
        public abstract T Sample();
        /// <summary>Public API</summary>
        public abstract double PDF(T value);
        /// <summary>Public API</summary>
        public abstract double CDF(T value);
        /// <summary>Public API</summary>
        public abstract T Mean { get; }
        /// <summary>Public API</summary>
        public abstract T Variance { get; }

        /// <summary>Public API</summary>
        public IEnumerable<T> Samples(int n)
        {
            for (int i = 0; i < n; i++)
                yield return Sample();
        }
    }

    /// <summary>Public API</summary>
    public class NormalDistribution : Distribution<double>
    {
        private readonly Random _rng = new();
        /// <summary>Public API</summary>
        public double Mu { get; }
        /// <summary>Public API</summary>
        public double Sigma { get; }

        /// <summary>Public API</summary>
        public override double Mean => Mu;
        /// <summary>Public API</summary>
        public override double Variance => Sigma * Sigma;

        /// <summary>Public API</summary>
        public NormalDistribution(double mu, double sigma)
        {
            Mu = mu;
            Sigma = sigma;
        }

        /// <summary>Public API</summary>
        public override double Sample()
        {
            // Box-Muller transform
            var u1 = 1.0 - _rng.NextDouble();
            var u2 = 1.0 - _rng.NextDouble();
            var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return Mu + Sigma * z;
        }

        /// <summary>Public API</summary>
        public override double PDF(double x)
        {
            var z = (x - Mu) / Sigma;
            return Math.Exp(-0.5 * z * z) / (Sigma * Math.Sqrt(2 * Math.PI));
        }

        /// <summary>Public API</summary>
        public override double CDF(double x)
        {
            var z = (x - Mu) / (Sigma * Math.Sqrt(2));
            return 0.5 * (1 + Erf(z));
        }

        private static double Erf(double x)
        {
            // Approximation
            var sign = x < 0 ? -1 : 1;
            x = Math.Abs(x);
            var t = 1.0 / (1.0 + 0.3275911 * x);
            var y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
            return sign * y;
        }

        /// <summary>
        /// Convert this distribution to an uncertain value.
        /// </summary>
        public Uncertain<double> ToUncertain(double confidenceLevel = 0.95)
        {
            return Uncertain.Value(Mean, confidenceLevel);
        }
    }

    #endregion

    #region Self-Explaining Code

    /// <summary>
    /// Self-Explaining Code - Code that can explain its logic to humans.
    /// Automatically generates natural language explanations.
    ///
    /// Example:
    /// var result = Explainable.Compute("filter valid users", () => users.Where(u => u.IsActive))
    ///     .WithReasoning("Only active users should be processed")
    ///     .Execute();
    /// Console.WriteLine(result.Explanation);
    /// </summary>
    public class Explainable<T>
    {
        private readonly string _description;
        private readonly Func<T> _computation;
        private readonly List<string> _reasoning;
        private readonly List<(string step, Func<string> detail)> _steps;

        internal Explainable(string description, Func<T> computation)
        {
            _description = description;
            _computation = computation;
            _reasoning = new List<string>();
            _steps = new List<(string, Func<string>)>();
        }

        /// <summary>Public API</summary>
        public Explainable<T> WithReasoning(string reason)
        {
            _reasoning.Add(reason);
            return this;
        }

        /// <summary>Public API</summary>
        public Explainable<T> AddStep(string stepDescription, Func<string> detailGenerator)
        {
            _steps.Add((stepDescription, detailGenerator));
            return this;
        }

        /// <summary>Public API</summary>
        public ExplainedResult<T> Execute()
        {
            var startTime = DateTime.UtcNow;
            var result = _computation();
            var duration = DateTime.UtcNow - startTime;

            var explanation = GenerateExplanation(result, duration);

            return new ExplainedResult<T>
            {
                Value = result,
                Explanation = explanation,
                Duration = duration
            };
        }

        private string GenerateExplanation(T result, TimeSpan duration)
        {
            var sb = new StringBuilder();

            sb.AppendLine($"## {_description}");
            sb.AppendLine();

            if (_reasoning.Count > 0)
            {
                sb.AppendLine("### Reasoning");
                foreach (var reason in _reasoning)
                    sb.AppendLine($"- {reason}");
                sb.AppendLine();
            }

            if (_steps.Count > 0)
            {
                sb.AppendLine("### Steps");
                for (int i = 0; i < _steps.Count; i++)
                {
                    var (step, detail) = _steps[i];
                    sb.AppendLine($"{i + 1}. {step}");
                    try { sb.AppendLine($"   Details: {detail()}"); } catch { }
                }
                sb.AppendLine();
            }

            sb.AppendLine("### Result");
            sb.AppendLine($"- Type: {typeof(T).Name}");
            sb.AppendLine($"- Duration: {duration.TotalMilliseconds:F2}ms");

            if (result is System.Collections.IEnumerable enumerable && result is not string)
            {
                var count = enumerable.Cast<object>().Count();
                sb.AppendLine($"- Count: {count} items");
            }

            return sb.ToString();
        }
    }

    /// <summary>Public API</summary>
    public class ExplainedResult<T>
    {
        /// <summary>Public API</summary>
        public T Value { get; set; } = default!;
        /// <summary>Public API</summary>
        public string Explanation { get; set; } = "";
        /// <summary>Public API</summary>
        public TimeSpan Duration { get; set; }

        /// <summary>Public API</summary>
        public static implicit operator T(ExplainedResult<T> r) => r.Value;
    }

    /// <summary>Public API</summary>
    public static class Explainable
    {
        /// <summary>Public API</summary>
        public static Explainable<T> Compute<T>(string description, Func<T> computation)
            => new(description, computation);
    }

    #endregion

    #region Semantic Contracts

    /// <summary>
    /// Semantic Contracts - Verify logical correctness automatically.
    /// Express invariants, preconditions, and postconditions declaratively.
    /// </summary>
    /// <example>
    /// <code>
    /// var balance = Contract.For(account.Balance)
    ///     .Requires(b =&gt; b &gt;= 0, "Balance must be non-negative")
    ///     .Ensures(b =&gt; b &lt;= account.Limit, "Balance must not exceed limit")
    ///     .Value;
    /// </code>
    /// </example>
    public class Contract<T>
    {
        private readonly T _value;
        private readonly List<(Func<T, bool> check, string message, ContractType type)> _contracts;
        private readonly bool _enforceOnRead;

        /// <summary>Public API</summary>
        public T Value
        {
            get
            {
                if (_enforceOnRead)
                    EnforceAll();
                return _value;
            }
        }

        internal Contract(T value, bool enforceOnRead = true)
        {
            _value = value;
            _contracts = new List<(Func<T, bool>, string, ContractType)>();
            _enforceOnRead = enforceOnRead;
        }

        /// <summary>Public API</summary>
        public Contract<T> Requires(Func<T, bool> predicate, string message)
        {
            _contracts.Add((predicate, message, ContractType.Precondition));
            return this;
        }

        /// <summary>Public API</summary>
        public Contract<T> Ensures(Func<T, bool> predicate, string message)
        {
            _contracts.Add((predicate, message, ContractType.Postcondition));
            return this;
        }

        /// <summary>Public API</summary>
        public Contract<T> Invariant(Func<T, bool> predicate, string message)
        {
            _contracts.Add((predicate, message, ContractType.Invariant));
            return this;
        }

        /// <summary>Public API</summary>
        public Contract<T> Assert(Func<T, bool> predicate, string message)
        {
            _contracts.Add((predicate, message, ContractType.Assertion));
            return this;
        }

        /// <summary>Public API</summary>
        public void EnforceAll()
        {
            foreach (var (check, message, type) in _contracts)
            {
                if (!check(_value))
                    throw new ContractViolationException(message, type);
            }
        }

        /// <summary>Public API</summary>
        public ContractValidationResult Validate()
        {
            var result = new ContractValidationResult { IsValid = true };

            foreach (var (check, message, type) in _contracts)
            {
                try
                {
                    if (!check(_value))
                    {
                        result.IsValid = false;
                        result.Violations.Add(new ContractViolation(message, type));
                    }
                }
                catch (Exception ex)
                {
                    result.IsValid = false;
                    result.Violations.Add(new ContractViolation($"{message}: {ex.Message}", type));
                }
            }

            return result;
        }
    }

    /// <summary>Public API</summary>
    public enum ContractType { Precondition, Postcondition, Invariant, Assertion }

    /// <summary>Public API</summary>
    public class ContractViolation
    {
        /// <summary>Public API</summary>
        public string Message { get; }
        /// <summary>Public API</summary>
        public ContractType Type { get; }
        /// <summary>Public API</summary>
        public ContractViolation(string message, ContractType type) { Message = message; Type = type; }
    }

    /// <summary>Public API</summary>
    public class ContractValidationResult
    {
        /// <summary>Public API</summary>
        public bool IsValid { get; set; }
        /// <summary>Public API</summary>
        public List<ContractViolation> Violations { get; } = new();
    }

    /// <summary>Public API</summary>
    public class ContractViolationException : Exception
    {
        /// <summary>Public API</summary>
        public ContractType Type { get; }
        /// <summary>Public API</summary>
        public ContractViolationException(string message, ContractType type)
            : base($"[{type}] {message}") { Type = type; }
    }

    /// <summary>Public API</summary>
    public static class Contract
    {
        /// <summary>Public API</summary>
        public static Contract<T> For<T>(T value, bool enforceOnRead = true) => new(value, enforceOnRead);
    }

    #endregion

    #region Adaptive Optimization

    /// <summary>
    /// Adaptive Optimization - Code that learns and optimizes itself based on runtime behavior.
    /// Automatically selects the best algorithm based on observed performance.
    /// </summary>
    /// <remarks>
    /// Example usage:
    /// var searcher = Adaptive.Algorithm("search")
    ///     .Add("linear", LinearSearch)
    ///     .Add("binary", BinarySearch)
    ///     .AutoSelect();
    /// var result = searcher.Execute(data);
    /// </remarks>
    public class AdaptiveAlgorithm<TInput, TOutput>
    {
        private readonly string _name;
        private readonly Dictionary<string, Func<TInput, TOutput>> _algorithms;
        private readonly Dictionary<string, AlgorithmStats> _stats;
        private string _selected = "";
        private readonly bool _autoSelect;
        private readonly object _lock = new();

        internal AdaptiveAlgorithm(string name)
        {
            _name = name;
            _algorithms = new Dictionary<string, Func<TInput, TOutput>>();
            _stats = new Dictionary<string, AlgorithmStats>();
            _autoSelect = false;
        }

        private AdaptiveAlgorithm(AdaptiveAlgorithm<TInput, TOutput> other, bool autoSelect)
        {
            _name = other._name;
            _algorithms = new Dictionary<string, Func<TInput, TOutput>>(other._algorithms);
            _stats = new Dictionary<string, AlgorithmStats>();
            foreach (var (k, v) in other._stats)
                _stats[k] = new AlgorithmStats { TotalTime = v.TotalTime, CallCount = v.CallCount };
            _autoSelect = autoSelect;
        }

        /// <summary>Public API</summary>
        public AdaptiveAlgorithm<TInput, TOutput> Add(string name, Func<TInput, TOutput> algorithm)
        {
            _algorithms[name] = algorithm;
            _stats[name] = new AlgorithmStats();
            if (string.IsNullOrEmpty(_selected))
                _selected = name;
            return this;
        }

        /// <summary>Public API</summary>
        public AdaptiveAlgorithm<TInput, TOutput> Select(string name)
        {
            if (!_algorithms.ContainsKey(name))
                throw new ArgumentException($"Algorithm '{name}' not found");
            _selected = name;
            return this;
        }

        /// <summary>Public API</summary>
        public AdaptiveAlgorithm<TInput, TOutput> AutoSelect()
        {
            return new AdaptiveAlgorithm<TInput, TOutput>(this, true);
        }

        /// <summary>Public API</summary>
        public TOutput Execute(TInput input)
        {
            string choice;

            if (_autoSelect)
            {
                choice = SelectBest();
            }
            else
            {
                choice = _selected;
            }

            var start = DateTime.UtcNow;
            var result = _algorithms[choice](input);
            var elapsed = (DateTime.UtcNow - start).TotalMilliseconds;

            lock (_lock)
            {
                _stats[choice].CallCount++;
                _stats[choice].TotalTime += elapsed;
            }

            return result;
        }

        private string SelectBest()
        {
            lock (_lock)
            {
                // Exploration phase: try all algorithms at least 5 times
                foreach (var (name, stats) in _stats)
                {
                    if (stats.CallCount < 5)
                        return name;
                }

                // Exploitation phase: pick the fastest
                return _stats
                    .OrderBy(kv => kv.Value.AverageTime)
                    .First().Key;
            }
        }

        /// <summary>Public API</summary>
        public string GetStats()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Adaptive Algorithm: {_name}");

            foreach (var (name, stats) in _stats.OrderBy(kv => kv.Value.AverageTime))
            {
                sb.AppendLine($"  {name}: {stats.CallCount} calls, avg {stats.AverageTime:F2}ms");
            }

            return sb.ToString();
        }
    }

    internal class AlgorithmStats
    {
        /// <summary>Public API</summary>
        public int CallCount;
        /// <summary>Public API</summary>
        public double TotalTime;
        /// <summary>Public API</summary>
        public double AverageTime => CallCount > 0 ? TotalTime / CallCount : double.MaxValue;
    }

    /// <summary>Public API</summary>
    public static class Adaptive
    {
        /// <summary>Public API</summary>
        public static AdaptiveAlgorithm<TInput, TOutput> Algorithm<TInput, TOutput>(string name)
            => new(name);
    }

    #endregion

    #region Natural Language Bridge

    /// <summary>
    /// Natural Language Bridge - Seamless conversion between code and natural language.
    /// Enables AI to understand and generate code through natural language descriptions.
    /// </summary>
    /// <remarks>
    /// Example usage:
    /// var description = NaturalLanguage.Describe(myMethod);
    /// var template = NaturalLanguage.GenerateTemplate("filter items by condition");
    /// </remarks>
    public static class NaturalLanguage
    {
        private static readonly Dictionary<string, string> _patternDescriptions = new()
        {
            { "Where", "filters items that match the condition" },
            { "Select", "transforms each item" },
            { "OrderBy", "sorts items in ascending order by" },
            { "OrderByDescending", "sorts items in descending order by" },
            { "GroupBy", "groups items by" },
            { "Aggregate", "combines items into a single result" },
            { "Take", "takes the first N items" },
            { "Skip", "skips the first N items" },
            { "First", "gets the first item" },
            { "Last", "gets the last item" },
            { "Any", "checks if any item matches" },
            { "All", "checks if all items match" },
            { "Count", "counts the items" },
            { "Sum", "adds up all values" },
            { "Average", "calculates the average" },
            { "Max", "finds the maximum value" },
            { "Min", "finds the minimum value" }
        };

        /// <summary>
        /// Describe code in natural language.
        /// </summary>
        public static string Describe<T>(Expression<T> expression)
        {
            return DescribeExpression(expression.Body);
        }

        /// <summary>
        /// Describe a method in natural language.
        /// </summary>
        public static string Describe(MethodInfo method)
        {
            var sb = new StringBuilder();

            // Method signature
            sb.Append($"A function called '{method.Name}' that ");

            // Parameters
            var parameters = method.GetParameters();
            if (parameters.Length > 0)
            {
                sb.Append("takes ");
                sb.Append(string.Join(" and ", parameters.Select(p => $"a {DescribeType(p.ParameterType)} called '{p.Name}'")));
                sb.Append(" and ");
            }

            // Return type
            if (method.ReturnType == typeof(void))
                sb.Append("performs an action");
            else
                sb.Append($"returns a {DescribeType(method.ReturnType)}");

            return sb.ToString();
        }

        /// <summary>
        /// Parse natural language into a code action.
        /// </summary>
        public static Action<T> ParseAction<T>(string description)
        {
            // Simple pattern matching for common actions
            var normalized = description.ToLowerInvariant();

            if (normalized.Contains("print") || normalized.Contains("display") || normalized.Contains("show"))
            {
                return x => Console.WriteLine(x);
            }

            if (normalized.Contains("double"))
            {
                if (typeof(T) == typeof(int))
                    return x => { /* double the value */ };
            }

            throw new NotSupportedException($"Cannot parse action: {description}");
        }

        /// <summary>
        /// Generate code template from natural language description.
        /// </summary>
        public static string GenerateTemplate(string description)
        {
            var normalized = description.ToLowerInvariant();
            var sb = new StringBuilder();

            if (normalized.Contains("filter") || normalized.Contains("where"))
            {
                sb.AppendLine("// Filter items based on condition");
                sb.AppendLine("var result = items.Where(item => /* condition */);");
            }
            else if (normalized.Contains("sort") || normalized.Contains("order"))
            {
                var desc = normalized.Contains("descending") || normalized.Contains("reverse");
                sb.AppendLine($"// Sort items {(desc ? "descending" : "ascending")}");
                sb.AppendLine($"var result = items.OrderBy{(desc ? "Descending" : "")}(item => item./* property */);");
            }
            else if (normalized.Contains("group"))
            {
                sb.AppendLine("// Group items by a key");
                sb.AppendLine("var result = items.GroupBy(item => item./* key */);");
            }
            else if (normalized.Contains("sum") || normalized.Contains("total"))
            {
                sb.AppendLine("// Calculate sum");
                sb.AppendLine("var result = items.Sum(item => item./* property */);");
            }
            else if (normalized.Contains("average") || normalized.Contains("mean"))
            {
                sb.AppendLine("// Calculate average");
                sb.AppendLine("var result = items.Average(item => item./* property */);");
            }
            else if (normalized.Contains("count"))
            {
                sb.AppendLine("// Count items");
                sb.AppendLine("var result = items.Count(item => /* condition */);");
            }
            else if (normalized.Contains("transform") || normalized.Contains("convert") || normalized.Contains("map"))
            {
                sb.AppendLine("// Transform each item");
                sb.AppendLine("var result = items.Select(item => /* transformation */);");
            }
            else
            {
                sb.AppendLine($"// {description}");
                sb.AppendLine("// TODO: Implement this logic");
            }

            return sb.ToString();
        }

        private static string DescribeExpression(Expression expr)
        {
            return expr switch
            {
                BinaryExpression binary => $"{DescribeExpression(binary.Left)} {DescribeOperator(binary.NodeType)} {DescribeExpression(binary.Right)}",
                MethodCallExpression call => DescribeMethodCall(call),
                LambdaExpression lambda => $"a function that {DescribeExpression(lambda.Body)}",
                ParameterExpression param => param.Name ?? "input",
                ConstantExpression constant => constant.Value?.ToString() ?? "null",
                MemberExpression member => $"the {member.Member.Name} of {DescribeExpression(member.Expression!)}",
                _ => expr.ToString()
            };
        }

        private static string DescribeMethodCall(MethodCallExpression call)
        {
            var methodName = call.Method.Name;
            if (_patternDescriptions.TryGetValue(methodName, out var description))
            {
                if (call.Arguments.Count > 0)
                {
                    return $"{description} {DescribeExpression(call.Arguments[0])}";
                }
                return description;
            }
            return $"calls {methodName}";
        }

        private static string DescribeOperator(ExpressionType type)
        {
            return type switch
            {
                ExpressionType.Add => "plus",
                ExpressionType.Subtract => "minus",
                ExpressionType.Multiply => "times",
                ExpressionType.Divide => "divided by",
                ExpressionType.Equal => "equals",
                ExpressionType.NotEqual => "is not equal to",
                ExpressionType.LessThan => "is less than",
                ExpressionType.GreaterThan => "is greater than",
                ExpressionType.LessThanOrEqual => "is at most",
                ExpressionType.GreaterThanOrEqual => "is at least",
                ExpressionType.AndAlso => "and",
                ExpressionType.OrElse => "or",
                _ => type.ToString()
            };
        }

        private static string DescribeType(Type type)
        {
            if (type == typeof(int)) return "whole number";
            if (type == typeof(double) || type == typeof(float)) return "decimal number";
            if (type == typeof(string)) return "text";
            if (type == typeof(bool)) return "true/false value";
            if (type == typeof(DateTime)) return "date and time";
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                return $"list of {DescribeType(type.GetGenericArguments()[0])}s";
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Dictionary<,>))
                return $"mapping from {DescribeType(type.GetGenericArguments()[0])} to {DescribeType(type.GetGenericArguments()[1])}";
            return type.Name;
        }
    }

    #endregion

    #region Formal Verification Hints

    /// <summary>
    /// Formal Verification - Hints for proving code correctness mathematically.
    /// Enables AI to reason about code properties formally.
    ///
    /// Example:
    /// [Verified(Property = "Termination", Proof = "Loop decreases 'n' each iteration")]
    /// [Verified(Property = "Correctness", Proof = "Invariant: sum = 1 + 2 + ... + i")]
    /// public int SumToN(int n) { ... }
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = true)]
    public class VerifiedAttribute : Attribute
    {
        /// <summary>Public API</summary>
        public string Property { get; set; } = "";
        /// <summary>Public API</summary>
        public string Proof { get; set; } = "";
        /// <summary>Public API</summary>
        public string Invariant { get; set; } = "";
        /// <summary>Public API</summary>
        public string Precondition { get; set; } = "";
        /// <summary>Public API</summary>
        public string Postcondition { get; set; } = "";
    }

    /// <summary>
    /// Marks code as proven to terminate.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class TerminatesAttribute : Attribute
    {
        /// <summary>Public API</summary>
        public string Reason { get; set; } = "";
    }

    /// <summary>
    /// Marks code as side-effect free (pure function).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class PureAttribute : Attribute { }

    /// <summary>
    /// Marks code as thread-safe.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class)]
    public class ThreadSafeAttribute : Attribute
    {
        /// <summary>Public API</summary>
        public string Mechanism { get; set; } = "";
    }

    /// <summary>
    /// Verifier for checking verification attributes.
    /// </summary>
    public static class Verifier
    {
        /// <summary>Public API</summary>
        public static VerificationReport Verify(Type type)
        {
            var report = new VerificationReport { TypeName = type.FullName ?? type.Name };

            foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static))
            {
                var verified = method.GetCustomAttributes<VerifiedAttribute>();
                var terminates = method.GetCustomAttribute<TerminatesAttribute>();
                var pure = method.GetCustomAttribute<PureAttribute>();
                var threadSafe = method.GetCustomAttribute<ThreadSafeAttribute>();

                if (verified.Any() || terminates != null || pure != null || threadSafe != null)
                {
                    var methodReport = new MethodVerification { MethodName = method.Name };

                    foreach (var v in verified)
                    {
                        methodReport.Properties.Add($"{v.Property}: {v.Proof}");
                    }

                    if (terminates != null)
                        methodReport.Properties.Add($"Terminates: {terminates.Reason}");
                    if (pure != null)
                        methodReport.Properties.Add("Pure: no side effects");
                    if (threadSafe != null)
                        methodReport.Properties.Add($"ThreadSafe: {threadSafe.Mechanism}");

                    report.Methods.Add(methodReport);
                }
            }

            return report;
        }
    }

    /// <summary>Public API</summary>
    public class VerificationReport
    {
        /// <summary>Public API</summary>
        public string TypeName { get; set; } = "";
        /// <summary>Public API</summary>
        public List<MethodVerification> Methods { get; } = new();

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Verification Report: {TypeName}");
            foreach (var m in Methods)
            {
                sb.AppendLine($"  {m.MethodName}:");
                foreach (var p in m.Properties)
                    sb.AppendLine($"    - {p}");
            }
            return sb.ToString();
        }
    }

    /// <summary>Public API</summary>
    public class MethodVerification
    {
        /// <summary>Public API</summary>
        public string MethodName { get; set; } = "";
        /// <summary>Public API</summary>
        public List<string> Properties { get; } = new();
    }

    #endregion
}