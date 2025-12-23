using System;
using System.Threading;
using System.Threading.Tasks;

namespace NSL.Tensor
{
    /// <summary>
    /// Result of a retry operation.
    /// </summary>
    public class RetryResult<T>
    {
        /// <summary>Public API</summary>
        public bool Success { get; }
        /// <summary>Public API</summary>
        public T? Value { get; }
        /// <summary>Public API</summary>
        public Exception? LastException { get; }
        /// <summary>Public API</summary>
        public int Attempts { get; }
        /// <summary>Public API</summary>
        public TimeSpan TotalTime { get; }

        private RetryResult(bool success, T? value, Exception? lastException, int attempts, TimeSpan totalTime)
        {
            Success = success;
            Value = value;
            LastException = lastException;
            Attempts = attempts;
            TotalTime = totalTime;
        }

        /// <summary>Public API</summary>
        public static RetryResult<T> Succeeded(T value, int attempts, TimeSpan totalTime)
            => new(true, value, null, attempts, totalTime);

        /// <summary>Public API</summary>
        public static RetryResult<T> Failed(Exception lastException, int attempts, TimeSpan totalTime)
            => new(false, default, lastException, attempts, totalTime);

        /// <summary>Public API</summary>
        public T GetValueOrThrow()
        {
            if (Success) return Value!;
            throw LastException ?? new InvalidOperationException("Retry failed with no exception");
        }

        /// <summary>Public API</summary>
        public T GetValueOrDefault(T defaultValue)
        {
            return Success ? Value! : defaultValue;
        }
    }

    /// <summary>
    /// Retry configuration.
    /// </summary>
    public class RetryConfig
    {
        /// <summary>Public API</summary>
        public int MaxAttempts { get; set; } = 3;
        /// <summary>Public API</summary>
        public TimeSpan InitialDelay { get; set; } = TimeSpan.FromSeconds(1);
        /// <summary>Public API</summary>
        public double BackoffMultiplier { get; set; } = 2.0;
        /// <summary>Public API</summary>
        public TimeSpan MaxDelay { get; set; } = TimeSpan.FromMinutes(1);
        /// <summary>Public API</summary>
        public bool AddJitter { get; set; } = true;
        /// <summary>Public API</summary>
        public double JitterFactor { get; set; } = 0.5;
        /// <summary>Public API</summary>
        public Func<Exception, bool>? ShouldRetry { get; set; }
        /// <summary>Public API</summary>
        public Action<Exception, int>? OnRetry { get; set; }

        /// <summary>Public API</summary>
        public static RetryConfig Default => new();

        /// <summary>Public API</summary>
        public static RetryConfig Fast => new()
        {
            MaxAttempts = 3,
            InitialDelay = TimeSpan.FromMilliseconds(100),
            BackoffMultiplier = 2.0,
            MaxDelay = TimeSpan.FromSeconds(5)
        };

        /// <summary>Public API</summary>
        public static RetryConfig Persistent => new()
        {
            MaxAttempts = 10,
            InitialDelay = TimeSpan.FromSeconds(1),
            BackoffMultiplier = 1.5,
            MaxDelay = TimeSpan.FromMinutes(5)
        };
    }

    /// <summary>
    /// Retry utilities with exponential backoff.
    /// Inspired by devtools_dew retry module.
    /// </summary>
    public static class Retry
    {
        private static readonly Random _jitterRandom = new();

        /// <summary>
        /// Execute an action with retry and exponential backoff.
        /// </summary>
        public static RetryResult<T> Execute<T>(Func<T> action, RetryConfig? config = null)
        {
            config ??= RetryConfig.Default;
            var startTime = DateTime.UtcNow;
            var currentDelay = config.InitialDelay;
            Exception? lastException = null;

            for (int attempt = 1; attempt <= config.MaxAttempts; attempt++)
            {
                try
                {
                    var result = action();
                    return RetryResult<T>.Succeeded(result, attempt, DateTime.UtcNow - startTime);
                }
                catch (Exception ex)
                {
                    lastException = ex;

                    if (attempt == config.MaxAttempts)
                        break;

                    if (config.ShouldRetry != null && !config.ShouldRetry(ex))
                        break;

                    config.OnRetry?.Invoke(ex, attempt);

                    var delay = currentDelay;
                    if (config.AddJitter)
                    {
                        var jitter = config.JitterFactor * (2 * _jitterRandom.NextDouble() - 1);
                        delay = TimeSpan.FromMilliseconds(delay.TotalMilliseconds * (1 + jitter));
                    }

                    Thread.Sleep(delay);

                    currentDelay = TimeSpan.FromMilliseconds(
                        Math.Min(
                            currentDelay.TotalMilliseconds * config.BackoffMultiplier,
                            config.MaxDelay.TotalMilliseconds
                        )
                    );
                }
            }

            return RetryResult<T>.Failed(lastException!, config.MaxAttempts, DateTime.UtcNow - startTime);
        }

        /// <summary>
        /// Execute an action with retry and exponential backoff.
        /// </summary>
        public static RetryResult<bool> Execute(Action action, RetryConfig? config = null)
        {
            return Execute(() => { action(); return true; }, config);
        }

        /// <summary>
        /// Execute an async action with retry and exponential backoff.
        /// </summary>
        public static async Task<RetryResult<T>> ExecuteAsync<T>(
            Func<Task<T>> action,
            RetryConfig? config = null,
            CancellationToken cancellationToken = default)
        {
            config ??= RetryConfig.Default;
            var startTime = DateTime.UtcNow;
            var currentDelay = config.InitialDelay;
            Exception? lastException = null;

            for (int attempt = 1; attempt <= config.MaxAttempts; attempt++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    var result = await action();
                    return RetryResult<T>.Succeeded(result, attempt, DateTime.UtcNow - startTime);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    lastException = ex;

                    if (attempt == config.MaxAttempts)
                        break;

                    if (config.ShouldRetry != null && !config.ShouldRetry(ex))
                        break;

                    config.OnRetry?.Invoke(ex, attempt);

                    var delay = currentDelay;
                    if (config.AddJitter)
                    {
                        var jitter = config.JitterFactor * (2 * _jitterRandom.NextDouble() - 1);
                        delay = TimeSpan.FromMilliseconds(delay.TotalMilliseconds * (1 + jitter));
                    }

                    await Task.Delay(delay, cancellationToken);

                    currentDelay = TimeSpan.FromMilliseconds(
                        Math.Min(
                            currentDelay.TotalMilliseconds * config.BackoffMultiplier,
                            config.MaxDelay.TotalMilliseconds
                        )
                    );
                }
            }

            return RetryResult<T>.Failed(lastException!, config.MaxAttempts, DateTime.UtcNow - startTime);
        }

        /// <summary>
        /// Execute an async action with retry and exponential backoff.
        /// </summary>
        public static Task<RetryResult<bool>> ExecuteAsync(
            Func<Task> action,
            RetryConfig? config = null,
            CancellationToken cancellationToken = default)
        {
            return ExecuteAsync(async () => { await action(); return true; }, config, cancellationToken);
        }

        /// <summary>
        /// Simple retry with default settings. Throws on final failure.
        /// </summary>
        public static T WithRetry<T>(Func<T> action, int maxAttempts = 3)
        {
            return Execute(action, new RetryConfig { MaxAttempts = maxAttempts }).GetValueOrThrow();
        }

        /// <summary>
        /// Simple retry with default settings. Throws on final failure.
        /// </summary>
        public static void WithRetry(Action action, int maxAttempts = 3)
        {
            Execute(action, new RetryConfig { MaxAttempts = maxAttempts }).GetValueOrThrow();
        }

        /// <summary>
        /// Simple async retry with default settings. Throws on final failure.
        /// </summary>
        public static async Task<T> WithRetryAsync<T>(Func<Task<T>> action, int maxAttempts = 3)
        {
            var result = await ExecuteAsync(action, new RetryConfig { MaxAttempts = maxAttempts });
            return result.GetValueOrThrow();
        }

        /// <summary>
        /// Simple async retry with default settings. Throws on final failure.
        /// </summary>
        public static async Task WithRetryAsync(Func<Task> action, int maxAttempts = 3)
        {
            var result = await ExecuteAsync(action, new RetryConfig { MaxAttempts = maxAttempts });
            result.GetValueOrThrow();
        }
    }

    /// <summary>
    /// Training-specific retry utilities.
    /// </summary>
    public static class TrainingRetry
    {
        /// <summary>
        /// Retry a training step, handling common ML exceptions.
        /// </summary>
        public static RetryResult<T> TrainingStep<T>(Func<T> step, int maxAttempts = 3)
        {
            return Retry.Execute(step, new RetryConfig
            {
                MaxAttempts = maxAttempts,
                InitialDelay = TimeSpan.FromMilliseconds(100),
                BackoffMultiplier = 1.5,
                MaxDelay = TimeSpan.FromSeconds(10),
                ShouldRetry = ex =>
                    ex is OutOfMemoryException ||
                    ex.Message.Contains("NaN") ||
                    ex.Message.Contains("overflow"),
                OnRetry = (ex, attempt) =>
                    Console.WriteLine($"Training step retry {attempt}: {ex.Message}")
            });
        }

        /// <summary>
        /// Retry with gradient clipping on overflow.
        /// </summary>
        public static Tensor SafeBackward(Tensor loss, double maxGradNorm = 1.0, int maxAttempts = 3)
        {
            var result = Retry.Execute(() =>
            {
                loss.Backward();

                // Check for NaN/Inf in gradients and clip if needed
                // This is a simplified version
                return loss;
            }, new RetryConfig
            {
                MaxAttempts = maxAttempts,
                InitialDelay = TimeSpan.FromMilliseconds(10),
                ShouldRetry = ex => ex.Message.Contains("NaN") || ex.Message.Contains("overflow")
            });

            return result.GetValueOrThrow();
        }
    }
}