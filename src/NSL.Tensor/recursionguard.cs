using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;

namespace NSL.Tensor
{
    /// <summary>
    /// Exception thrown when recursion limit is exceeded.
    /// </summary>
    public class RecursionLimitException : Exception
    {
        /// <summary>Public API</summary>
        public string GuardName { get; }
        /// <summary>Public API</summary>
        public int CurrentDepth { get; }
        /// <summary>Public API</summary>
        public int MaxDepth { get; }

        /// <summary>Public API</summary>
        public RecursionLimitException(string guardName, int currentDepth, int maxDepth)
            : base($"Recursion limit exceeded in '{guardName}': depth {currentDepth} > max {maxDepth}")
        {
            GuardName = guardName;
            CurrentDepth = currentDepth;
            MaxDepth = maxDepth;
        }
    }

    /// <summary>
    /// Exception thrown when a cycle is detected.
    /// </summary>
    public class CycleDetectedException : Exception
    {
        /// <summary>Public API</summary>
        public string DetectorName { get; }
        /// <summary>Public API</summary>
        public object Key { get; }

        /// <summary>Public API</summary>
        public CycleDetectedException(string detectorName, object key)
            : base($"Cycle detected in '{detectorName}' with key: {key}")
        {
            DetectorName = detectorName;
            Key = key;
        }
    }

    /// <summary>
    /// Thread-local recursion depth tracking.
    /// Inspired by pmx_utils RecursionGuard.
    /// </summary>
    public class RecursionGuard : IDisposable
    {
        private static readonly ThreadLocal<Dictionary<string, int>> _depthTrackers =
            new(() => new Dictionary<string, int>());

        private readonly string _name;
        private readonly int _maxDepth;
        private readonly bool _raiseOnLimit;
        private bool _disposed;

        /// <summary>
        /// Current recursion depth for this guard.
        /// </summary>
        public int CurrentDepth
        {
            get
            {
                var tracker = _depthTrackers.Value!;
                return tracker.TryGetValue(_name, out var depth) ? depth : 0;
            }
        }

        /// <summary>
        /// Whether the limit was exceeded.
        /// </summary>
        public bool Exceeded { get; private set; }

        /// <summary>
        /// Create a recursion guard.
        /// </summary>
        /// <param name="name">Unique name for this guard</param>
        /// <param name="maxDepth">Maximum recursion depth</param>
        /// <param name="raiseOnLimit">Throw exception when limit exceeded</param>
        public RecursionGuard(string name, int maxDepth = 100, bool raiseOnLimit = true)
        {
            _name = name;
            _maxDepth = maxDepth;
            _raiseOnLimit = raiseOnLimit;
            _disposed = false;

            Enter();
        }

        private void Enter()
        {
            var tracker = _depthTrackers.Value!;
            tracker.TryGetValue(_name, out var depth);
            depth++;
            tracker[_name] = depth;

            if (depth > _maxDepth)
            {
                Exceeded = true;
                if (_raiseOnLimit)
                    throw new RecursionLimitException(_name, depth, _maxDepth);
            }
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            var tracker = _depthTrackers.Value!;
            if (tracker.TryGetValue(_name, out var depth))
            {
                depth--;
                if (depth <= 0)
                    tracker.Remove(_name);
                else
                    tracker[_name] = depth;
            }
        }

        /// <summary>
        /// Execute an action with recursion protection.
        /// </summary>
        public static T Execute<T>(string name, Func<T> action, int maxDepth = 100, T? fallback = default)
        {
            using var guard = new RecursionGuard(name, maxDepth, raiseOnLimit: false);
            if (guard.Exceeded)
                return fallback!;
            return action();
        }

        /// <summary>
        /// Execute an action with recursion protection.
        /// </summary>
        public static void Execute(string name, Action action, int maxDepth = 100)
        {
            using var guard = new RecursionGuard(name, maxDepth, raiseOnLimit: true);
            action();
        }
    }

    /// <summary>
    /// Cycle detection for recursive operations.
    /// Tracks visited keys to detect when the same key is visited twice in a call chain.
    /// Inspired by pmx_utils CycleDetector.
    /// </summary>
    public class CycleDetector
    {
        private readonly string _name;
        private readonly int _maxVisits;
        private readonly Dictionary<object, int> _visitCounts;

        /// <summary>
        /// Create a cycle detector.
        /// </summary>
        /// <param name="name">Name for error messages</param>
        /// <param name="maxVisits">Max times a key can be visited before cycle (default: 1)</param>
        public CycleDetector(string name, int maxVisits = 1)
        {
            _name = name;
            _maxVisits = maxVisits;
            _visitCounts = new Dictionary<object, int>();
        }

        /// <summary>
        /// Check if visiting this key would cause a cycle.
        /// </summary>
        public bool WouldCycle(object key)
        {
            return _visitCounts.TryGetValue(key, out var count) && count >= _maxVisits;
        }

        /// <summary>
        /// Check if currently visiting a key.
        /// </summary>
        public bool IsVisiting(object key)
        {
            return _visitCounts.ContainsKey(key);
        }

        /// <summary>
        /// Start tracking a key. Returns a disposable that stops tracking when disposed.
        /// </summary>
        /// <param name="key">Key to track</param>
        /// <param name="raiseOnCycle">Throw exception if cycle detected</param>
        public IDisposable Track(object key, bool raiseOnCycle = true)
        {
            _visitCounts.TryGetValue(key, out var count);
            count++;
            _visitCounts[key] = count;

            if (count > _maxVisits && raiseOnCycle)
                throw new CycleDetectedException(_name, key);

            return new TrackingScope(this, key);
        }

        /// <summary>
        /// Execute action while tracking a key.
        /// </summary>
        public T Track<T>(object key, Func<T> action, T? fallbackOnCycle = default)
        {
            if (WouldCycle(key))
                return fallbackOnCycle!;

            using var _ = Track(key, raiseOnCycle: false);
            return action();
        }

        /// <summary>
        /// Reset all tracking.
        /// </summary>
        public void Reset()
        {
            _visitCounts.Clear();
        }

        private void Untrack(object key)
        {
            if (_visitCounts.TryGetValue(key, out var count))
            {
                count--;
                if (count <= 0)
                    _visitCounts.Remove(key);
                else
                    _visitCounts[key] = count;
            }
        }

        private class TrackingScope : IDisposable
        {
            private readonly CycleDetector _detector;
            private readonly object _key;
            private bool _disposed;

            /// <summary>Public API</summary>
            public TrackingScope(CycleDetector detector, object key)
            {
                _detector = detector;
                _key = key;
            }

            /// <summary>Public API</summary>
            public void Dispose()
            {
                if (_disposed) return;
                _disposed = true;
                _detector.Untrack(_key);
            }
        }
    }

    /// <summary>
    /// Decorator-style recursion limiter for methods.
    /// </summary>
    public static class DepthLimiter
    {
        /// <summary>
        /// Wrap a function with depth limiting.
        /// </summary>
        public static Func<TArg, TResult> Limit<TArg, TResult>(
            Func<TArg, TResult> func,
            int maxDepth = 100,
            TResult? fallback = default,
            [CallerMemberName] string? name = null)
        {
            var guardName = $"DepthLimiter:{name}";

            return arg =>
            {
                using var guard = new RecursionGuard(guardName, maxDepth, raiseOnLimit: false);
                if (guard.Exceeded)
                    return fallback!;
                return func(arg);
            };
        }

        /// <summary>
        /// Wrap a function with depth limiting and cycle detection.
        /// </summary>
        public static Func<TArg, TResult> LimitWithCycleDetection<TArg, TResult>(
            Func<TArg, TResult> func,
            Func<TArg, object> keySelector,
            int maxDepth = 100,
            TResult? fallback = default,
            [CallerMemberName] string? name = null) where TArg : notnull
        {
            var guardName = $"DepthLimiter:{name}";
            var cycleDetector = new CycleDetector(guardName);

            return arg =>
            {
                using var guard = new RecursionGuard(guardName, maxDepth, raiseOnLimit: false);
                if (guard.Exceeded)
                    return fallback!;

                var key = keySelector(arg);
                if (cycleDetector.WouldCycle(key))
                    return fallback!;

                using var tracking = cycleDetector.Track(key, raiseOnCycle: false);
                return func(arg);
            };
        }
    }
}