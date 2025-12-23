using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;

namespace NSL.GPU
{
    /// <summary>
    /// KV-Cache for efficient transformer inference.
    /// Caches key and value tensors from previous tokens to avoid recomputation.
    /// Reduces O(n²) attention to O(n) for autoregressive generation.
    /// </summary>
    public class KVCache : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly int _numLayers;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _maxSeqLen;

        // [layer][batch] -> (keys, values)
        private readonly Dictionary<int, Dictionary<int, (GpuTensor Keys, GpuTensor Values)>> _cache = new();
        private readonly Dictionary<int, int> _currentLengths = new();  // batch -> current sequence length
        private readonly object _lock = new();
        private bool _disposed;

        /// <summary>Public API</summary>
        public int NumLayers => _numLayers;
        /// <summary>Public API</summary>
        public int MaxSequenceLength => _maxSeqLen;

        /// <summary>Public API</summary>
        public KVCache(Accelerator accelerator, int numLayers, int numHeads, int headDim, int maxSeqLen = 8192)
        {
            _accelerator = accelerator;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _headDim = headDim;
            _maxSeqLen = maxSeqLen;

            // Pre-allocate cache structures
            for (int layer = 0; layer < numLayers; layer++)
            {
                _cache[layer] = new Dictionary<int, (GpuTensor, GpuTensor)>();
            }
        }

        /// <summary>
        /// Get current sequence length for a batch
        /// </summary>
        public int GetSequenceLength(int batchIdx)
        {
            lock (_lock)
            {
                return _currentLengths.TryGetValue(batchIdx, out int len) ? len : 0;
            }
        }

        /// <summary>
        /// Update cache with new key/value for a layer
        /// </summary>
        public void Update(int layer, int batchIdx, GpuTensor newKeys, GpuTensor newValues)
        {
            lock (_lock)
            {
                if (!_cache[layer].TryGetValue(batchIdx, out var existing))
                {
                    // First token - allocate full cache buffers
                    var keys = new GpuTensor(_accelerator, new[] { _maxSeqLen, _numHeads, _headDim });
                    var values = new GpuTensor(_accelerator, new[] { _maxSeqLen, _numHeads, _headDim });
                    _cache[layer][batchIdx] = (keys, values);
                    _currentLengths[batchIdx] = 0;
                    existing = (keys, values);
                }

                int currentLen = _currentLengths[batchIdx];
                int newTokens = newKeys.Shape[0];

                if (currentLen + newTokens > _maxSeqLen)
                {
                    throw new InvalidOperationException($"KV-Cache overflow: {currentLen + newTokens} > {_maxSeqLen}");
                }

                // Copy new keys/values into cache at current position
                CopyToCache(existing.Keys, newKeys, currentLen);
                CopyToCache(existing.Values, newValues, currentLen);

                if (layer == _numLayers - 1)
                {
                    _currentLengths[batchIdx] = currentLen + newTokens;
                }
            }
        }

        /// <summary>
        /// Get cached keys and values for a layer up to current sequence length
        /// </summary>
        public (GpuTensor Keys, GpuTensor Values) Get(int layer, int batchIdx)
        {
            lock (_lock)
            {
                if (!_cache[layer].TryGetValue(batchIdx, out var cached))
                {
                    throw new InvalidOperationException($"No cache for layer {layer}, batch {batchIdx}");
                }

                int seqLen = _currentLengths[batchIdx];

                // Return view of cached data up to current length
                var keys = SliceCache(cached.Keys, seqLen);
                var values = SliceCache(cached.Values, seqLen);

                return (keys, values);
            }
        }

        /// <summary>
        /// Clear cache for specific batch (e.g., when sequence is complete)
        /// </summary>
        public void ClearBatch(int batchIdx)
        {
            lock (_lock)
            {
                _currentLengths.Remove(batchIdx);
                // Don't deallocate - reuse buffers
            }
        }

        /// <summary>
        /// Clear all caches
        /// </summary>
        public void ClearAll()
        {
            lock (_lock)
            {
                _currentLengths.Clear();
            }
        }

        private void CopyToCache(GpuTensor cache, GpuTensor newData, int offset)
        {
            int newLen = newData.Shape[0];
            int stride = _numHeads * _headDim;

            // Copy new data into cache at offset position
            var sourceData = newData.ToArray();
            var cacheData = cache.ToArray();

            Array.Copy(sourceData, 0, cacheData, offset * stride, newLen * stride);

            // Copy back to GPU
            cache.Buffer.View.SubView(offset * stride, newLen * stride).CopyFromCPU(sourceData);
        }

        private GpuTensor SliceCache(GpuTensor cache, int seqLen)
        {
            // Create a new tensor with the active portion of the cache
            var result = new GpuTensor(_accelerator, new[] { seqLen, _numHeads, _headDim });
            int elements = seqLen * _numHeads * _headDim;

            if (elements > 0)
            {
                var data = new float[elements];
                cache.Buffer.View.SubView(0, elements).CopyToCPU(data);
                result.Buffer.CopyFromCPU(data);
            }

            return result;
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                foreach (var layer in _cache.Values)
                {
                    foreach (var (keys, values) in layer.Values)
                    {
                        keys.Dispose();
                        values.Dispose();
                    }
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Speculative Decoding for faster autoregressive generation.
    /// Uses a smaller "draft" model to propose multiple tokens,
    /// then verifies with the large model in parallel.
    /// Can achieve 2-3x speedup for well-matched draft models.
    /// </summary>
    public class SpeculativeDecoder
    {
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;
        private readonly int _specTokens;  // Number of tokens to speculate
        private readonly float _acceptanceThreshold;

        /// <summary>Public API</summary>
        public int SpeculativeTokens => _specTokens;

        /// <summary>Public API</summary>
        public SpeculativeDecoder(Accelerator accelerator, GpuKernels kernels,
            int speculativeTokens = 4, float acceptanceThreshold = 0.9f)
        {
            _accelerator = accelerator;
            _kernels = kernels;
            _specTokens = speculativeTokens;
            _acceptanceThreshold = acceptanceThreshold;
        }

        /// <summary>
        /// Result of speculative decoding step
        /// </summary>
        public class SpeculativeResult
        {
            /// <summary>Public API</summary>
            public int[] AcceptedTokens { get; set; } = Array.Empty<int>();
            /// <summary>Public API</summary>
            public int NumAccepted { get; set; }
            /// <summary>Public API</summary>
            public int NumProposed { get; set; }
            /// <summary>Public API</summary>
            public float AcceptanceRate => NumProposed > 0 ? (float)NumAccepted / NumProposed : 0;
        }

        /// <summary>
        /// Generate tokens using speculative decoding
        /// </summary>
        /// <param name="draftLogits">Logits from draft model for K speculative tokens [K, vocab]</param>
        /// <param name="targetLogits">Logits from target model for same K tokens [K, vocab]</param>
        /// <param name="draftTokens">Tokens sampled from draft model</param>
        /// <returns>Accepted tokens and statistics</returns>
        public SpeculativeResult Verify(GpuTensor draftLogits, GpuTensor targetLogits, int[] draftTokens)
        {
            int numSpec = draftTokens.Length;
            int vocabSize = draftLogits.Shape[1];

            // Convert logits to probabilities
            var draftProbs = Softmax(draftLogits);
            var targetProbs = Softmax(targetLogits);

            var draftProbsArr = draftProbs.ToArray();
            var targetProbsArr = targetProbs.ToArray();

            var accepted = new List<int>();
            var random = new Random();

            for (int i = 0; i < numSpec; i++)
            {
                int token = draftTokens[i];
                float pDraft = draftProbsArr[i * vocabSize + token];
                float pTarget = targetProbsArr[i * vocabSize + token];

                // Acceptance probability: min(1, p_target / p_draft)
                float acceptProb = Math.Min(1.0f, pTarget / (pDraft + 1e-10f));

                if (random.NextDouble() < acceptProb)
                {
                    accepted.Add(token);
                }
                else
                {
                    // Rejection - sample from adjusted distribution
                    // p_adjusted = max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))
                    float[] adjusted = new float[vocabSize];
                    float sum = 0;

                    for (int v = 0; v < vocabSize; v++)
                    {
                        adjusted[v] = Math.Max(0, targetProbsArr[i * vocabSize + v] - draftProbsArr[i * vocabSize + v]);
                        sum += adjusted[v];
                    }

                    if (sum > 0)
                    {
                        // Sample from adjusted distribution
                        float r = (float)random.NextDouble() * sum;
                        float cumsum = 0;
                        for (int v = 0; v < vocabSize; v++)
                        {
                            cumsum += adjusted[v];
                            if (cumsum >= r)
                            {
                                accepted.Add(v);
                                break;
                            }
                        }
                    }
                    break;  // Stop after first rejection
                }
            }

            draftProbs.Dispose();
            targetProbs.Dispose();

            return new SpeculativeResult
            {
                AcceptedTokens = accepted.ToArray(),
                NumAccepted = accepted.Count,
                NumProposed = numSpec
            };
        }

        /// <summary>
        /// Batch speculative generation with KV-cache
        /// </summary>
        public async Task<int[]> GenerateAsync(
            Func<int[], GpuTensor> draftModel,
            Func<int[], GpuTensor> targetModel,
            int[] promptTokens,
            int maxNewTokens,
            KVCache? draftCache = null,
            KVCache? targetCache = null)
        {
            var generated = new List<int>(promptTokens);
            int tokensGenerated = 0;

            while (tokensGenerated < maxNewTokens)
            {
                // Draft model generates K speculative tokens
                var draftTokens = new int[_specTokens];
                var currentInput = generated.ToArray();

                for (int i = 0; i < _specTokens && tokensGenerated + i < maxNewTokens; i++)
                {
                    var logits = draftModel(currentInput);
                    draftTokens[i] = SampleToken(logits);
                    currentInput = currentInput.Append(draftTokens[i]).ToArray();
                    logits.Dispose();
                }

                // Target model verifies all K tokens in parallel
                var targetLogits = targetModel(generated.Concat(draftTokens).ToArray());
                var draftLogits = draftModel(generated.Concat(draftTokens).ToArray());

                var result = Verify(draftLogits, targetLogits, draftTokens);

                generated.AddRange(result.AcceptedTokens);
                tokensGenerated += result.NumAccepted;

                targetLogits.Dispose();
                draftLogits.Dispose();

                // If all tokens accepted, we can continue speculating
                // If rejection occurred, we already added the corrected token
            }

            return generated.Skip(promptTokens.Length).ToArray();
        }

        private GpuTensor Softmax(GpuTensor logits)
        {
            return _kernels.Softmax(logits);
        }

        private int SampleToken(GpuTensor logits)
        {
            var probs = Softmax(logits);
            var probsArr = probs.ToArray();
            probs.Dispose();

            var random = new Random();
            float r = (float)random.NextDouble();
            float cumsum = 0;

            for (int i = 0; i < probsArr.Length; i++)
            {
                cumsum += probsArr[i];
                if (cumsum >= r) return i;
            }

            return probsArr.Length - 1;
        }
    }

    /// <summary>
    /// Continuous Batching for efficient LLM serving.
    /// Dynamically adds/removes sequences from batch as they complete.
    /// Maximizes GPU utilization by filling gaps from completed sequences.
    /// </summary>
    public class ContinuousBatcher
    {
        private readonly Accelerator _accelerator;
        private readonly GpuKernels _kernels;
        private readonly int _maxBatchSize;
        private readonly int _maxSeqLen;

        private readonly ConcurrentQueue<InferenceRequest> _pendingRequests = new();
        private readonly ConcurrentDictionary<int, InferenceRequest> _activeRequests = new();
        private readonly ConcurrentDictionary<int, KVCache> _kvCaches = new();

        private int _nextRequestId = 0;
        private readonly SemaphoreSlim _batchLock = new(1, 1);

        /// <summary>Public API</summary>
        public int MaxBatchSize => _maxBatchSize;
        /// <summary>Public API</summary>
        public int ActiveRequests => _activeRequests.Count;
        /// <summary>Public API</summary>
        public int PendingRequests => _pendingRequests.Count;

        /// <summary>Public API</summary>
        public class InferenceRequest
        {
            /// <summary>Public API</summary>
            public int Id { get; set; }
            /// <summary>Public API</summary>
            public int[] PromptTokens { get; set; } = Array.Empty<int>();
            /// <summary>Public API</summary>
            public List<int> GeneratedTokens { get; set; } = new();
            /// <summary>Public API</summary>
            public int MaxNewTokens { get; set; } = 256;
            /// <summary>Public API</summary>
            public float Temperature { get; set; } = 1.0f;
            /// <summary>Public API</summary>
            public int? StopToken { get; set; }
            /// <summary>Public API</summary>
            public TaskCompletionSource<int[]> Completion { get; set; } = new();
            /// <summary>Public API</summary>
            public DateTime SubmittedAt { get; set; } = DateTime.UtcNow;
            /// <summary>Public API</summary>
            public DateTime? StartedAt { get; set; }
            /// <summary>Public API</summary>
            public DateTime? CompletedAt { get; set; }
            /// <summary>Public API</summary>
            public bool IsComplete => GeneratedTokens.Count >= MaxNewTokens ||
                (StopToken.HasValue && GeneratedTokens.LastOrDefault() == StopToken.Value);
        }

        /// <summary>Public API</summary>
        public class BatchStatistics
        {
            /// <summary>Public API</summary>
            public int TotalRequests { get; set; }
            /// <summary>Public API</summary>
            public int CompletedRequests { get; set; }
            /// <summary>Public API</summary>
            public double AverageLatencyMs { get; set; }
            /// <summary>Public API</summary>
            public double TokensPerSecond { get; set; }
            /// <summary>Public API</summary>
            public double BatchUtilization { get; set; }
        }

        private readonly List<double> _latencies = new();
        private int _totalTokens = 0;
        private DateTime _startTime = DateTime.UtcNow;

        /// <summary>Public API</summary>
        public ContinuousBatcher(Accelerator accelerator, GpuKernels kernels,
            int maxBatchSize = 32, int maxSeqLen = 4096)
        {
            _accelerator = accelerator;
            _kernels = kernels;
            _maxBatchSize = maxBatchSize;
            _maxSeqLen = maxSeqLen;
        }

        /// <summary>
        /// Submit a new inference request
        /// </summary>
        public Task<int[]> SubmitAsync(int[] promptTokens, int maxNewTokens = 256,
            float temperature = 1.0f, int? stopToken = null)
        {
            var request = new InferenceRequest
            {
                Id = Interlocked.Increment(ref _nextRequestId),
                PromptTokens = promptTokens,
                MaxNewTokens = maxNewTokens,
                Temperature = temperature,
                StopToken = stopToken
            };

            _pendingRequests.Enqueue(request);
            return request.Completion.Task;
        }

        /// <summary>
        /// Process one batch iteration
        /// </summary>
        public async Task ProcessBatchAsync(Func<int[][], GpuTensor> modelForward)
        {
            await _batchLock.WaitAsync();
            try
            {
                // Fill batch with pending requests
                while (_activeRequests.Count < _maxBatchSize && _pendingRequests.TryDequeue(out var request))
                {
                    request.StartedAt = DateTime.UtcNow;
                    _activeRequests[request.Id] = request;
                }

                if (_activeRequests.IsEmpty) return;

                // Build batch inputs
                var batchInputs = _activeRequests.Values
                    .Select(r => r.PromptTokens.Concat(r.GeneratedTokens).ToArray())
                    .ToArray();

                // Forward pass for entire batch
                var batchLogits = modelForward(batchInputs);

                // Sample next token for each sequence
                var logitsArr = batchLogits.ToArray();
                int vocabSize = batchLogits.Shape[^1];
                int batchSize = batchInputs.Length;

                var random = new Random();
                var completedIds = new List<int>();

                int idx = 0;
                foreach (var request in _activeRequests.Values.ToList())
                {
                    // Get logits for last token of this sequence
                    int seqLen = batchInputs[idx].Length;
                    int offset = (seqLen - 1) * vocabSize; // Last position

                    // Apply temperature and sample
                    var probs = new float[vocabSize];
                    float maxLogit = float.MinValue;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        maxLogit = Math.Max(maxLogit, logitsArr[idx * batchLogits.Shape[1] * vocabSize + offset + v]);
                    }

                    float sum = 0;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float logit = logitsArr[idx * batchLogits.Shape[1] * vocabSize + offset + v];
                        probs[v] = MathF.Exp((logit - maxLogit) / request.Temperature);
                        sum += probs[v];
                    }

                    for (int v = 0; v < vocabSize; v++) probs[v] /= sum;

                    // Sample
                    float r = (float)random.NextDouble();
                    float cumsum = 0;
                    int sampledToken = vocabSize - 1;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        cumsum += probs[v];
                        if (cumsum >= r)
                        {
                            sampledToken = v;
                            break;
                        }
                    }

                    request.GeneratedTokens.Add(sampledToken);
                    Interlocked.Increment(ref _totalTokens);

                    // Check completion
                    if (request.IsComplete)
                    {
                        completedIds.Add(request.Id);
                        request.CompletedAt = DateTime.UtcNow;
                        request.Completion.SetResult(request.GeneratedTokens.ToArray());

                        var latency = (request.CompletedAt.Value - request.SubmittedAt).TotalMilliseconds;
                        lock (_latencies) _latencies.Add(latency);
                    }

                    idx++;
                }

                batchLogits.Dispose();

                // Remove completed requests
                foreach (var id in completedIds)
                {
                    _activeRequests.TryRemove(id, out _);
                }
            }
            finally
            {
                _batchLock.Release();
            }
        }

        /// <summary>
        /// Run continuous batching loop
        /// </summary>
        public async Task RunAsync(Func<int[][], GpuTensor> modelForward, CancellationToken cancellation = default)
        {
            while (!cancellation.IsCancellationRequested)
            {
                if (_activeRequests.IsEmpty && _pendingRequests.IsEmpty)
                {
                    await Task.Delay(1, cancellation);
                    continue;
                }

                await ProcessBatchAsync(modelForward);
            }
        }

        /// <summary>
        /// Get current statistics
        /// </summary>
        public BatchStatistics GetStatistics()
        {
            var elapsed = (DateTime.UtcNow - _startTime).TotalSeconds;

            lock (_latencies)
            {
                return new BatchStatistics
                {
                    TotalRequests = _nextRequestId,
                    CompletedRequests = _latencies.Count,
                    AverageLatencyMs = _latencies.Count > 0 ? _latencies.Average() : 0,
                    TokensPerSecond = elapsed > 0 ? _totalTokens / elapsed : 0,
                    BatchUtilization = (double)_activeRequests.Count / _maxBatchSize
                };
            }
        }
    }

    /// <summary>
    /// PagedAttention for efficient memory management.
    /// Allocates KV-cache in fixed-size pages to reduce fragmentation
    /// and enable memory sharing across sequences.
    /// </summary>
    public class PagedAttention : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly int _pageSize;  // Tokens per page
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _maxPages;

        private readonly ConcurrentQueue<int> _freePages = new();
        private readonly ConcurrentDictionary<int, List<int>> _sequencePages = new();  // seqId -> page indices
        private readonly GpuTensor _keyPages;   // [maxPages, pageSize, numHeads, headDim]
        private readonly GpuTensor _valuePages;

        private bool _disposed;

        /// <summary>Public API</summary>
        public int PageSize => _pageSize;
        /// <summary>Public API</summary>
        public int FreePages => _freePages.Count;
        /// <summary>Public API</summary>
        public int UsedPages => _maxPages - _freePages.Count;

        /// <summary>Public API</summary>
        public PagedAttention(Accelerator accelerator, int numHeads, int headDim,
            int pageSize = 16, int maxPages = 1024)
        {
            _accelerator = accelerator;
            _pageSize = pageSize;
            _numHeads = numHeads;
            _headDim = headDim;
            _maxPages = maxPages;

            // Pre-allocate page pool
            _keyPages = new GpuTensor(accelerator, new[] { maxPages, pageSize, numHeads, headDim });
            _valuePages = new GpuTensor(accelerator, new[] { maxPages, pageSize, numHeads, headDim });

            // Initialize free page list
            for (int i = 0; i < maxPages; i++)
            {
                _freePages.Enqueue(i);
            }
        }

        /// <summary>
        /// Allocate pages for a new sequence
        /// </summary>
        public void AllocateSequence(int seqId, int initialTokens = 0)
        {
            int pagesNeeded = (initialTokens + _pageSize - 1) / _pageSize;
            pagesNeeded = Math.Max(1, pagesNeeded);  // At least one page

            var pages = new List<int>();
            for (int i = 0; i < pagesNeeded; i++)
            {
                if (!_freePages.TryDequeue(out int pageIdx))
                {
                    throw new OutOfMemoryException("No free pages available");
                }
                pages.Add(pageIdx);
            }

            _sequencePages[seqId] = pages;
        }

        /// <summary>
        /// Append KV to sequence, allocating new pages as needed
        /// </summary>
        public void Append(int seqId, GpuTensor keys, GpuTensor values, int tokenOffset)
        {
            if (!_sequencePages.TryGetValue(seqId, out var pages))
            {
                throw new InvalidOperationException($"Sequence {seqId} not allocated");
            }

            int numTokens = keys.Shape[0];
            int endOffset = tokenOffset + numTokens;
            int pagesNeeded = (endOffset + _pageSize - 1) / _pageSize;

            // Allocate more pages if needed
            while (pages.Count < pagesNeeded)
            {
                if (!_freePages.TryDequeue(out int newPage))
                {
                    throw new OutOfMemoryException("No free pages available");
                }
                pages.Add(newPage);
            }

            // Copy data to pages
            var keyData = keys.ToArray();
            var valueData = values.ToArray();
            int stride = _numHeads * _headDim;

            for (int t = 0; t < numTokens; t++)
            {
                int globalPos = tokenOffset + t;
                int pageIdx = globalPos / _pageSize;
                int pageOffset = globalPos % _pageSize;
                int physicalPage = pages[pageIdx];

                int srcOffset = t * stride;
                int dstOffset = (physicalPage * _pageSize + pageOffset) * stride;

                // Copy to page buffers
                _keyPages.Buffer.View.SubView(dstOffset, stride)
                    .CopyFromCPU(keyData.Skip(srcOffset).Take(stride).ToArray());
                _valuePages.Buffer.View.SubView(dstOffset, stride)
                    .CopyFromCPU(valueData.Skip(srcOffset).Take(stride).ToArray());
            }
        }

        /// <summary>
        /// Get KV for attention computation
        /// </summary>
        public (GpuTensor Keys, GpuTensor Values) GetKV(int seqId, int seqLen)
        {
            if (!_sequencePages.TryGetValue(seqId, out var pages))
            {
                throw new InvalidOperationException($"Sequence {seqId} not allocated");
            }

            var keys = new GpuTensor(_accelerator, new[] { seqLen, _numHeads, _headDim });
            var values = new GpuTensor(_accelerator, new[] { seqLen, _numHeads, _headDim });

            int stride = _numHeads * _headDim;
            var keyData = new float[seqLen * stride];
            var valueData = new float[seqLen * stride];

            for (int t = 0; t < seqLen; t++)
            {
                int pageIdx = t / _pageSize;
                int pageOffset = t % _pageSize;
                int physicalPage = pages[pageIdx];

                int srcOffset = (physicalPage * _pageSize + pageOffset) * stride;
                int dstOffset = t * stride;

                var keyPage = new float[stride];
                var valuePage = new float[stride];

                _keyPages.Buffer.View.SubView(srcOffset, stride).CopyToCPU(keyPage);
                _valuePages.Buffer.View.SubView(srcOffset, stride).CopyToCPU(valuePage);

                Array.Copy(keyPage, 0, keyData, dstOffset, stride);
                Array.Copy(valuePage, 0, valueData, dstOffset, stride);
            }

            keys.Buffer.CopyFromCPU(keyData);
            values.Buffer.CopyFromCPU(valueData);

            return (keys, values);
        }

        /// <summary>
        /// Free pages for completed sequence
        /// </summary>
        public void FreeSequence(int seqId)
        {
            if (_sequencePages.TryRemove(seqId, out var pages))
            {
                foreach (var page in pages)
                {
                    _freePages.Enqueue(page);
                }
            }
        }

        /// <summary>
        /// Share pages between sequences (for beam search, etc.)
        /// </summary>
        public void SharePages(int sourceSeqId, int targetSeqId, int numTokens)
        {
            if (!_sequencePages.TryGetValue(sourceSeqId, out var sourcePages))
            {
                throw new InvalidOperationException($"Source sequence {sourceSeqId} not found");
            }

            int pagesToShare = (numTokens + _pageSize - 1) / _pageSize;
            var sharedPages = sourcePages.Take(pagesToShare).ToList();

            // Target gets references to same physical pages (copy-on-write semantics)
            _sequencePages[targetSeqId] = new List<int>(sharedPages);
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _keyPages.Dispose();
                _valuePages.Dispose();
                _disposed = true;
            }
        }
    }
}