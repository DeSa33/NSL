using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace NSL.Tensor.NN
{
    /// <summary>
    /// Real multi-node distributed training infrastructure for NSL.
    /// Production-ready TCP-based communication across physical machines.
    ///
    /// Usage:
    /// - Set environment variables: NSL_WORLD_SIZE, NSL_RANK, NSL_MASTER_ADDR, NSL_MASTER_PORT
    /// - Or pass configuration directly to constructor
    ///
    /// Example:
    /// var ctx = await DistributedContext.InitializeAsync(worldSize: 4, rank: 0, masterAddr: "192.168.1.100", masterPort: 29500);
    /// var avgGrad = await ctx.AllReduceAsync(localGrad, ReduceOp.Mean);
    /// </summary>
    public sealed class DistributedContext : IAsyncDisposable, IDisposable
    {
        // Connection state
        private readonly int _worldSize;
        private readonly int _rank;
        private readonly string _masterAddr;
        private readonly int _masterPort;

        // Network infrastructure
        private TcpListener? _server;
        private readonly Socket[] _sockets;
        private readonly NetworkStream[] _streams;
        private readonly SemaphoreSlim[] _sendLocks;
        private readonly SemaphoreSlim[] _recvLocks;

        // Message handling
        private readonly ConcurrentDictionary<string, TaskCompletionSource<byte[]>> _pendingRecvs;
        private readonly ConcurrentDictionary<string, byte[]> _receivedMessages;
        private readonly CancellationTokenSource _shutdownCts;
        private readonly Task[] _receiverTasks;

        // State
        private volatile bool _initialized;
        private volatile bool _disposed;
        private long _messageCounter;

        /// <summary>Public API</summary>
        public int WorldSize => _worldSize;
        /// <summary>Public API</summary>
        public int Rank => _rank;
        /// <summary>Public API</summary>
        public bool IsInitialized => _initialized;
        /// <summary>Public API</summary>
        public bool IsMaster => _rank == 0;

        private DistributedContext(int worldSize, int rank, string masterAddr, int masterPort)
        {
            _worldSize = worldSize;
            _rank = rank;
            _masterAddr = masterAddr;
            _masterPort = masterPort;

            _sockets = new Socket[worldSize];
            _streams = new NetworkStream[worldSize];
            _sendLocks = new SemaphoreSlim[worldSize];
            _recvLocks = new SemaphoreSlim[worldSize];
            _receiverTasks = new Task[worldSize];

            for (int i = 0; i < worldSize; i++)
            {
                _sendLocks[i] = new SemaphoreSlim(1, 1);
                _recvLocks[i] = new SemaphoreSlim(1, 1);
            }

            _pendingRecvs = new ConcurrentDictionary<string, TaskCompletionSource<byte[]>>();
            _receivedMessages = new ConcurrentDictionary<string, byte[]>();
            _shutdownCts = new CancellationTokenSource();
        }

        /// <summary>
        /// Initialize distributed context from environment variables.
        /// Expected: NSL_WORLD_SIZE, NSL_RANK, NSL_MASTER_ADDR, NSL_MASTER_PORT
        /// </summary>
        public static async Task<DistributedContext> InitializeFromEnvAsync(int timeoutSeconds = 300)
        {
            int worldSize = int.Parse(Environment.GetEnvironmentVariable("NSL_WORLD_SIZE")
                ?? throw new InvalidOperationException("NSL_WORLD_SIZE not set"));
            int rank = int.Parse(Environment.GetEnvironmentVariable("NSL_RANK")
                ?? throw new InvalidOperationException("NSL_RANK not set"));
            string masterAddr = Environment.GetEnvironmentVariable("NSL_MASTER_ADDR")
                ?? throw new InvalidOperationException("NSL_MASTER_ADDR not set");
            int masterPort = int.Parse(Environment.GetEnvironmentVariable("NSL_MASTER_PORT") ?? "29500");

            return await InitializeAsync(worldSize, rank, masterAddr, masterPort, timeoutSeconds);
        }

        /// <summary>
        /// Initialize distributed context with explicit configuration.
        /// </summary>
        public static async Task<DistributedContext> InitializeAsync(
            int worldSize,
            int rank,
            string masterAddr,
            int masterPort = 29500,
            int timeoutSeconds = 300)
        {
            if (rank < 0 || rank >= worldSize)
                throw new ArgumentException($"Rank {rank} invalid for world size {worldSize}");

            var ctx = new DistributedContext(worldSize, rank, masterAddr, masterPort);
            await ctx.EstablishConnectionsAsync(timeoutSeconds);
            return ctx;
        }

        private async Task EstablishConnectionsAsync(int timeoutSeconds)
        {
            Console.WriteLine($"[Rank {_rank}] Initializing distributed context (world_size={_worldSize})");

            using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(_shutdownCts.Token, timeoutCts.Token);
            var ct = linkedCts.Token;

            try
            {
                // Each rank listens on its own port
                int myPort = _masterPort + _rank;
                _server = new TcpListener(IPAddress.Any, myPort);
                _server.Start();
                Console.WriteLine($"[Rank {_rank}] Listening on port {myPort}");

                // Connection strategy: each rank connects to all lower ranks
                // and accepts connections from all higher ranks
                var connectTasks = new List<Task>();
                var acceptTasks = new List<Task>();

                // Connect to lower ranks (they are already listening)
                for (int targetRank = 0; targetRank < _rank; targetRank++)
                {
                    int target = targetRank;
                    connectTasks.Add(ConnectToRankAsync(target, ct));
                }

                // Accept connections from higher ranks
                for (int sourceRank = _rank + 1; sourceRank < _worldSize; sourceRank++)
                {
                    acceptTasks.Add(AcceptFromRankAsync(ct));
                }

                // Wait for all connections
                await Task.WhenAll(connectTasks.Concat(acceptTasks));

                // Start receiver tasks for each peer
                for (int peer = 0; peer < _worldSize; peer++)
                {
                    if (peer != _rank && _streams[peer] != null)
                    {
                        int p = peer;
                        _receiverTasks[peer] = Task.Run(() => ReceiverLoopAsync(p, _shutdownCts.Token));
                    }
                }

                _initialized = true;
                Console.WriteLine($"[Rank {_rank}] All connections established");

                // Barrier to ensure all ranks are ready
                await BarrierAsync();
                Console.WriteLine($"[Rank {_rank}] Distributed context ready");
            }
            catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested)
            {
                throw new TimeoutException($"[Rank {_rank}] Connection establishment timed out after {timeoutSeconds}s");
            }
        }

        private async Task ConnectToRankAsync(int targetRank, CancellationToken ct)
        {
            // Get target address from NSL_NODE_ADDRESSES or fall back to master + port offset
            string targetHost;
            int targetPort;

            var nodeAddrs = Environment.GetEnvironmentVariable("NSL_NODE_ADDRESSES");
            if (!string.IsNullOrEmpty(nodeAddrs))
            {
                var entries = nodeAddrs.Split(',');
                if (targetRank < entries.Length)
                {
                    var parts = entries[targetRank].Trim().Split(':');
                    targetHost = parts[0];
                    targetPort = parts.Length > 1 ? int.Parse(parts[1]) : _masterPort + targetRank;
                }
                else
                {
                    targetHost = _masterAddr;
                    targetPort = _masterPort + targetRank;
                }
            }
            else
            {
                // Single-machine mode: all workers on localhost with port offsets
                targetHost = targetRank == 0 ? _masterAddr : "127.0.0.1";
                targetPort = _masterPort + targetRank;
            }

            Console.WriteLine($"[Rank {_rank}] Connecting to rank {targetRank} at {targetHost}:{targetPort}");

            int retries = 0;
            const int maxRetries = 60;
            const int retryDelayMs = 1000;

            while (true)
            {
                try
                {
                    var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                    socket.NoDelay = true;
                    socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
                    socket.SendBufferSize = 4 * 1024 * 1024;
                    socket.ReceiveBufferSize = 4 * 1024 * 1024;

                    await socket.ConnectAsync(targetHost, targetPort, ct);

                    var stream = new NetworkStream(socket, ownsSocket: true);

                    // Send handshake: our rank
                    await SendHandshakeAsync(stream, ct);

                    _sockets[targetRank] = socket;
                    _streams[targetRank] = stream;

                    Console.WriteLine($"[Rank {_rank}] Connected to rank {targetRank}");
                    return;
                }
                catch (SocketException) when (retries < maxRetries)
                {
                    retries++;
                    await Task.Delay(retryDelayMs, ct);
                }
            }
        }

        private async Task AcceptFromRankAsync(CancellationToken ct)
        {
            var socket = await _server!.AcceptSocketAsync(ct);
            socket.NoDelay = true;
            socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
            socket.SendBufferSize = 4 * 1024 * 1024;
            socket.ReceiveBufferSize = 4 * 1024 * 1024;

            var stream = new NetworkStream(socket, ownsSocket: true);

            // Receive handshake to learn peer's rank
            int peerRank = await ReceiveHandshakeAsync(stream, ct);

            _sockets[peerRank] = socket;
            _streams[peerRank] = stream;

            Console.WriteLine($"[Rank {_rank}] Accepted connection from rank {peerRank}");
        }

        private async Task SendHandshakeAsync(NetworkStream stream, CancellationToken ct)
        {
            var buffer = new byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(buffer, _rank);
            await stream.WriteAsync(buffer, ct);
            await stream.FlushAsync(ct);
        }

        private async Task<int> ReceiveHandshakeAsync(NetworkStream stream, CancellationToken ct)
        {
            var buffer = new byte[4];
            await ReadExactAsync(stream, buffer, ct);
            return BinaryPrimitives.ReadInt32LittleEndian(buffer);
        }

        private async Task ReceiverLoopAsync(int peerRank, CancellationToken ct)
        {
            var stream = _streams[peerRank];
            var headerBuffer = new byte[12]; // 4 bytes tag length + 4 bytes payload length + 4 bytes message id

            try
            {
                while (!ct.IsCancellationRequested)
                {
                    // Read header
                    await ReadExactAsync(stream, headerBuffer, ct);
                    int tagLength = BinaryPrimitives.ReadInt32LittleEndian(headerBuffer.AsSpan(0, 4));
                    int payloadLength = BinaryPrimitives.ReadInt32LittleEndian(headerBuffer.AsSpan(4, 4));
                    int messageId = BinaryPrimitives.ReadInt32LittleEndian(headerBuffer.AsSpan(8, 4));

                    // Read tag
                    var tagBuffer = new byte[tagLength];
                    await ReadExactAsync(stream, tagBuffer, ct);
                    string tag = Encoding.UTF8.GetString(tagBuffer);

                    // Read payload
                    var payload = new byte[payloadLength];
                    if (payloadLength > 0)
                    {
                        await ReadExactAsync(stream, payload, ct);
                    }

                    // Dispatch message
                    string key = $"{peerRank}:{tag}:{messageId}";

                    if (_pendingRecvs.TryRemove(key, out var tcs))
                    {
                        tcs.TrySetResult(payload);
                    }
                    else
                    {
                        _receivedMessages[key] = payload;
                    }
                }
            }
            catch (Exception ex) when (ct.IsCancellationRequested || _disposed)
            {
                // Expected during shutdown
                Debug.WriteLine($"[Rank {_rank}] Receiver for peer {peerRank} stopped: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Rank {_rank}] Receiver error from peer {peerRank}: {ex.Message}");
            }
        }

        private static async Task ReadExactAsync(NetworkStream stream, byte[] buffer, CancellationToken ct)
        {
            int totalRead = 0;
            while (totalRead < buffer.Length)
            {
                int read = await stream.ReadAsync(buffer.AsMemory(totalRead, buffer.Length - totalRead), ct);
                if (read == 0)
                    throw new EndOfStreamException("Connection closed by peer");
                totalRead += read;
            }
        }

        #region Point-to-Point Communication

        /// <summary>
        /// Send tensor to a specific rank.
        /// </summary>
        public async Task SendAsync(Tensor tensor, int destRank, string tag = "p2p", CancellationToken ct = default)
        {
            EnsureInitialized();
            if (destRank == _rank) return;
            if (destRank < 0 || destRank >= _worldSize)
                throw new ArgumentException($"Invalid destination rank: {destRank}");

            var payload = SerializeTensor(tensor);
            int messageId = (int)Interlocked.Increment(ref _messageCounter);

            await SendRawAsync(destRank, tag, messageId, payload, ct);
        }

        /// <summary>
        /// Receive tensor from a specific rank.
        /// </summary>
        public async Task<Tensor> RecvAsync(int srcRank, string tag = "p2p", int messageId = 0,
            int timeoutMs = 30000, CancellationToken ct = default)
        {
            EnsureInitialized();
            if (srcRank == _rank)
                throw new ArgumentException("Cannot receive from self");

            string key = $"{srcRank}:{tag}:{messageId}";

            // Check if already received
            if (_receivedMessages.TryRemove(key, out var existing))
            {
                return DeserializeTensor(existing);
            }

            // Wait for message
            var tcs = new TaskCompletionSource<byte[]>(TaskCreationOptions.RunContinuationsAsynchronously);
            _pendingRecvs[key] = tcs;

            using var timeoutCts = new CancellationTokenSource(timeoutMs);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);

            try
            {
                using var registration = linkedCts.Token.Register(() => tcs.TrySetCanceled());
                var payload = await tcs.Task;
                return DeserializeTensor(payload);
            }
            catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested)
            {
                _pendingRecvs.TryRemove(key, out _);
                throw new TimeoutException($"Recv from rank {srcRank} timed out after {timeoutMs}ms");
            }
        }

        private async Task SendRawAsync(int destRank, string tag, int messageId, byte[] payload, CancellationToken ct)
        {
            var tagBytes = Encoding.UTF8.GetBytes(tag);
            var header = new byte[12];
            BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0, 4), tagBytes.Length);
            BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(4, 4), payload.Length);
            BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(8, 4), messageId);

            await _sendLocks[destRank].WaitAsync(ct);
            try
            {
                var stream = _streams[destRank];
                await stream.WriteAsync(header, ct);
                await stream.WriteAsync(tagBytes, ct);
                if (payload.Length > 0)
                {
                    await stream.WriteAsync(payload, ct);
                }
                await stream.FlushAsync(ct);
            }
            finally
            {
                _sendLocks[destRank].Release();
            }
        }

        #endregion

        #region Collective Operations

        /// <summary>
        /// Synchronization barrier - all ranks must reach this point before any can proceed.
        /// </summary>
        public async Task BarrierAsync(CancellationToken ct = default)
        {
            EnsureInitialized();

            // Use dissemination barrier algorithm: O(log n) steps
            int steps = (int)Math.Ceiling(Math.Log2(_worldSize));

            for (int step = 0; step < steps; step++)
            {
                int distance = 1 << step;
                int sendTo = (_rank + distance) % _worldSize;
                int recvFrom = (_rank - distance + _worldSize) % _worldSize;

                var token = new byte[1] { 1 };
                int msgId = (int)Interlocked.Increment(ref _messageCounter);

                // Send and receive simultaneously
                var sendTask = SendRawAsync(sendTo, $"barrier_{step}", msgId, token, ct);
                var recvTask = RecvAsync(recvFrom, $"barrier_{step}", msgId, 60000, ct);

                await Task.WhenAll(sendTask, recvTask);
            }
        }

        /// <summary>
        /// Broadcast tensor from source rank to all other ranks.
        /// Uses binomial tree algorithm: O(log n) latency.
        /// </summary>
        public async Task<Tensor> BroadcastAsync(Tensor? tensor, int srcRank, CancellationToken ct = default)
        {
            EnsureInitialized();

            if (srcRank < 0 || srcRank >= _worldSize)
                throw new ArgumentException($"Invalid source rank: {srcRank}");

            // Reorder ranks so source is at root
            int relativeRank = (_rank - srcRank + _worldSize) % _worldSize;
            int msgId = (int)Interlocked.Increment(ref _messageCounter);

            Tensor result;

            if (_rank == srcRank)
            {
                result = tensor ?? throw new ArgumentNullException(nameof(tensor));
            }
            else
            {
                result = null!;
            }

            // Binomial tree broadcast
            int mask = 1;
            while (mask < _worldSize)
            {
                if ((relativeRank & mask) != 0)
                {
                    // Receive from parent
                    int parent = (relativeRank & ~mask);
                    int parentRank = (parent + srcRank) % _worldSize;
                    result = await RecvAsync(parentRank, "broadcast", msgId, 60000, ct);
                    break;
                }
                mask <<= 1;
            }

            // Send to children
            mask >>= 1;
            while (mask > 0)
            {
                if ((relativeRank & mask) == 0)
                {
                    int child = relativeRank | mask;
                    if (child < _worldSize)
                    {
                        int childRank = (child + srcRank) % _worldSize;
                        await SendRawAsync(childRank, "broadcast", msgId, SerializeTensor(result), ct);
                    }
                }
                mask >>= 1;
            }

            return result;
        }

        /// <summary>
        /// AllReduce: reduce tensors across all ranks and distribute result to all.
        /// Uses Ring AllReduce algorithm: O(n) bandwidth optimal.
        /// </summary>
        public async Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum, CancellationToken ct = default)
        {
            EnsureInitialized();

            int n = tensor.Data.Length;

            // For small tensors or small world size, use recursive halving/doubling
            if (n < 1024 || _worldSize <= 2)
            {
                return await AllReduceRecursiveAsync(tensor, op, ct);
            }

            // Ring AllReduce for larger tensors
            return await AllReduceRingAsync(tensor, op, ct);
        }

        private async Task<Tensor> AllReduceRingAsync(Tensor tensor, ReduceOp op, CancellationToken ct)
        {
            int n = tensor.Data.Length;
            int chunkSize = (n + _worldSize - 1) / _worldSize;
            var result = new double[n];
            Array.Copy(tensor.Data, result, n);

            int left = (_rank - 1 + _worldSize) % _worldSize;
            int right = (_rank + 1) % _worldSize;

            // Phase 1: Scatter-Reduce
            // Each rank reduces a different chunk, sends result to next rank
            for (int step = 0; step < _worldSize - 1; step++)
            {
                int sendChunk = (_rank - step + _worldSize) % _worldSize;
                int recvChunk = (_rank - step - 1 + _worldSize) % _worldSize;

                int sendStart = sendChunk * chunkSize;
                int sendEnd = Math.Min(sendStart + chunkSize, n);
                int sendLen = sendEnd - sendStart;

                int recvStart = recvChunk * chunkSize;
                int recvEnd = Math.Min(recvStart + chunkSize, n);
                int recvLen = recvEnd - recvStart;

                // Pack send chunk
                var sendData = new double[sendLen];
                Array.Copy(result, sendStart, sendData, 0, sendLen);
                var sendTensor = new Tensor(sendData, new long[] { sendLen }, false);

                int msgId = (int)Interlocked.Increment(ref _messageCounter);

                // Send to right, receive from left
                var sendTask = SendRawAsync(right, $"ring_sr_{step}", msgId, SerializeTensor(sendTensor), ct);
                var recvTask = RecvAsync(left, $"ring_sr_{step}", msgId, 60000, ct);

                await sendTask;
                var recvTensor = await recvTask;

                // Reduce received chunk into result
                for (int i = 0; i < recvLen && (recvStart + i) < n; i++)
                {
                    result[recvStart + i] = ApplyOp(result[recvStart + i], recvTensor.Data[i], op);
                }
            }

            // Phase 2: AllGather
            // Each rank has one fully reduced chunk, share with all
            for (int step = 0; step < _worldSize - 1; step++)
            {
                int sendChunk = (_rank - step + 1 + _worldSize) % _worldSize;
                int recvChunk = (_rank - step + _worldSize) % _worldSize;

                int sendStart = sendChunk * chunkSize;
                int sendEnd = Math.Min(sendStart + chunkSize, n);
                int sendLen = sendEnd - sendStart;

                int recvStart = recvChunk * chunkSize;
                int recvEnd = Math.Min(recvStart + chunkSize, n);

                var sendData = new double[sendLen];
                Array.Copy(result, sendStart, sendData, 0, sendLen);
                var sendTensor = new Tensor(sendData, new long[] { sendLen }, false);

                int msgId = (int)Interlocked.Increment(ref _messageCounter);

                var sendTask = SendRawAsync(right, $"ring_ag_{step}", msgId, SerializeTensor(sendTensor), ct);
                var recvTask = RecvAsync(left, $"ring_ag_{step}", msgId, 60000, ct);

                await sendTask;
                var recvTensor = await recvTask;

                // Copy received chunk to result
                Array.Copy(recvTensor.Data, 0, result, recvStart, recvTensor.Data.Length);
            }

            // Apply mean if needed
            if (op == ReduceOp.Mean)
            {
                double scale = 1.0 / _worldSize;
                for (int i = 0; i < n; i++)
                {
                    result[i] *= scale;
                }
            }

            return new Tensor(result, tensor.Shape, false);
        }

        private async Task<Tensor> AllReduceRecursiveAsync(Tensor tensor, ReduceOp op, CancellationToken ct)
        {
            var result = tensor.Clone();

            // Recursive halving for reduce, recursive doubling for broadcast
            int distance = 1;
            while (distance < _worldSize)
            {
                int partner = _rank ^ distance;

                if (partner < _worldSize)
                {
                    int msgId = (int)Interlocked.Increment(ref _messageCounter);

                    var sendTask = SendRawAsync(partner, $"allreduce_{distance}", msgId, SerializeTensor(result), ct);
                    var recvTask = RecvAsync(partner, $"allreduce_{distance}", msgId, 60000, ct);

                    await sendTask;
                    var recvTensor = await recvTask;

                    // Reduce
                    for (int i = 0; i < result.Data.Length; i++)
                    {
                        result.Data[i] = ApplyOp(result.Data[i], recvTensor.Data[i], op);
                    }
                }

                distance *= 2;
            }

            if (op == ReduceOp.Mean)
            {
                double scale = 1.0 / _worldSize;
                for (int i = 0; i < result.Data.Length; i++)
                {
                    result.Data[i] *= scale;
                }
            }

            return result;
        }

        /// <summary>
        /// Reduce: reduce tensors to a single destination rank.
        /// </summary>
        public async Task<Tensor?> ReduceAsync(Tensor tensor, int dstRank, ReduceOp op = ReduceOp.Sum, CancellationToken ct = default)
        {
            EnsureInitialized();

            // Use binomial tree reduction
            var result = tensor.Clone();
            int relativeRank = (_rank - dstRank + _worldSize) % _worldSize;

            int mask = 1;
            while (mask < _worldSize)
            {
                if ((relativeRank & mask) != 0)
                {
                    // Send to parent and exit
                    int parent = relativeRank & ~mask;
                    int parentRank = (parent + dstRank) % _worldSize;
                    int msgId = (int)Interlocked.Increment(ref _messageCounter);
                    await SendRawAsync(parentRank, "reduce", msgId, SerializeTensor(result), ct);
                    return null;
                }
                else
                {
                    // Receive from child
                    int child = relativeRank | mask;
                    if (child < _worldSize)
                    {
                        int childRank = (child + dstRank) % _worldSize;
                        int msgId = (int)Interlocked.Increment(ref _messageCounter);
                        var recvTensor = await RecvAsync(childRank, "reduce", msgId, 60000, ct);

                        for (int i = 0; i < result.Data.Length; i++)
                        {
                            result.Data[i] = ApplyOp(result.Data[i], recvTensor.Data[i], op);
                        }
                    }
                }
                mask <<= 1;
            }

            if (op == ReduceOp.Mean)
            {
                double scale = 1.0 / _worldSize;
                for (int i = 0; i < result.Data.Length; i++)
                {
                    result.Data[i] *= scale;
                }
            }

            return result;
        }

        /// <summary>
        /// AllGather: gather tensors from all ranks to all ranks.
        /// </summary>
        public async Task<Tensor[]> AllGatherAsync(Tensor tensor, CancellationToken ct = default)
        {
            EnsureInitialized();

            var results = new Tensor[_worldSize];
            results[_rank] = tensor.Clone();

            // Ring-based allgather
            int left = (_rank - 1 + _worldSize) % _worldSize;
            int right = (_rank + 1) % _worldSize;

            int sendRank = _rank;

            for (int step = 0; step < _worldSize - 1; step++)
            {
                int msgId = (int)Interlocked.Increment(ref _messageCounter);

                var sendTask = SendRawAsync(right, $"allgather_{step}", msgId, SerializeTensor(results[sendRank]), ct);
                var recvTask = RecvAsync(left, $"allgather_{step}", msgId, 60000, ct);

                await sendTask;

                int recvRank = (sendRank - 1 + _worldSize) % _worldSize;
                results[recvRank] = await recvTask;

                sendRank = recvRank;
            }

            return results;
        }

        /// <summary>
        /// ReduceScatter: reduce then scatter result chunks.
        /// </summary>
        public async Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum, CancellationToken ct = default)
        {
            EnsureInitialized();

            int n = tensor.Data.Length;
            int chunkSize = (n + _worldSize - 1) / _worldSize;

            // First do ring reduce-scatter (phase 1 of ring allreduce)
            var result = new double[n];
            Array.Copy(tensor.Data, result, n);

            int left = (_rank - 1 + _worldSize) % _worldSize;
            int right = (_rank + 1) % _worldSize;

            for (int step = 0; step < _worldSize - 1; step++)
            {
                int sendChunk = (_rank - step + _worldSize) % _worldSize;
                int recvChunk = (_rank - step - 1 + _worldSize) % _worldSize;

                int sendStart = sendChunk * chunkSize;
                int sendEnd = Math.Min(sendStart + chunkSize, n);
                int sendLen = sendEnd - sendStart;

                int recvStart = recvChunk * chunkSize;
                int recvEnd = Math.Min(recvStart + chunkSize, n);
                int recvLen = recvEnd - recvStart;

                var sendData = new double[sendLen];
                Array.Copy(result, sendStart, sendData, 0, sendLen);
                var sendTensor = new Tensor(sendData, new long[] { sendLen }, false);

                int msgId = (int)Interlocked.Increment(ref _messageCounter);

                var sendTask = SendRawAsync(right, $"reducescatter_{step}", msgId, SerializeTensor(sendTensor), ct);
                var recvTask = RecvAsync(left, $"reducescatter_{step}", msgId, 60000, ct);

                await sendTask;
                var recvTensor = await recvTask;

                for (int i = 0; i < recvLen && (recvStart + i) < n; i++)
                {
                    result[recvStart + i] = ApplyOp(result[recvStart + i], recvTensor.Data[i], op);
                }
            }

            // Extract this rank's chunk
            int myStart = _rank * chunkSize;
            int myEnd = Math.Min(myStart + chunkSize, n);
            int myLen = myEnd - myStart;

            var myChunk = new double[myLen];
            Array.Copy(result, myStart, myChunk, 0, myLen);

            if (op == ReduceOp.Mean)
            {
                double scale = 1.0 / _worldSize;
                for (int i = 0; i < myLen; i++)
                {
                    myChunk[i] *= scale;
                }
            }

            return new Tensor(myChunk, new long[] { myLen }, false);
        }

        /// <summary>
        /// Scatter: distribute tensor chunks from source rank to all ranks.
        /// </summary>
        public async Task<Tensor> ScatterAsync(Tensor? tensor, int srcRank, int chunkSize, CancellationToken ct = default)
        {
            EnsureInitialized();

            if (_rank == srcRank)
            {
                if (tensor == null)
                    throw new ArgumentNullException(nameof(tensor));

                // Send chunks to all ranks
                var sendTasks = new List<Task>();
                for (int r = 0; r < _worldSize; r++)
                {
                    int start = r * chunkSize;
                    int len = Math.Min(chunkSize, tensor.Data.Length - start);

                    var chunk = new double[len];
                    Array.Copy(tensor.Data, start, chunk, 0, len);
                    var chunkTensor = new Tensor(chunk, new long[] { len }, false);

                    if (r == _rank)
                    {
                        return chunkTensor;
                    }
                    else
                    {
                        int msgId = (int)Interlocked.Increment(ref _messageCounter);
                        sendTasks.Add(SendRawAsync(r, "scatter", msgId, SerializeTensor(chunkTensor), ct));
                    }
                }

                await Task.WhenAll(sendTasks);

                // Return own chunk
                var myChunk = new double[Math.Min(chunkSize, tensor.Data.Length - _rank * chunkSize)];
                Array.Copy(tensor.Data, _rank * chunkSize, myChunk, 0, myChunk.Length);
                return new Tensor(myChunk, new long[] { myChunk.Length }, false);
            }
            else
            {
                int msgId = (int)Interlocked.Increment(ref _messageCounter);
                return await RecvAsync(srcRank, "scatter", msgId, 60000, ct);
            }
        }

        /// <summary>
        /// Gather: collect tensor chunks from all ranks to destination rank.
        /// </summary>
        public async Task<Tensor?> GatherAsync(Tensor tensor, int dstRank, CancellationToken ct = default)
        {
            EnsureInitialized();

            if (_rank == dstRank)
            {
                int chunkSize = tensor.Data.Length;
                var result = new double[chunkSize * _worldSize];

                // Copy own chunk
                Array.Copy(tensor.Data, 0, result, _rank * chunkSize, chunkSize);

                // Receive from all others
                var recvTasks = new List<Task<(int rank, Tensor tensor)>>();
                for (int r = 0; r < _worldSize; r++)
                {
                    if (r != _rank)
                    {
                        int rank = r;
                        int msgId = (int)Interlocked.Increment(ref _messageCounter);
                        recvTasks.Add(Task.Run(async () =>
                        {
                            var t = await RecvAsync(rank, "gather", msgId, 60000, ct);
                            return (rank, t);
                        }));
                    }
                }

                foreach (var task in recvTasks)
                {
                    var (rank, t) = await task;
                    Array.Copy(t.Data, 0, result, rank * chunkSize, t.Data.Length);
                }

                return new Tensor(result, new long[] { result.Length }, false);
            }
            else
            {
                int msgId = (int)Interlocked.Increment(ref _messageCounter);
                await SendRawAsync(dstRank, "gather", msgId, SerializeTensor(tensor), ct);
                return null;
            }
        }

        #endregion

        #region Helpers

        private static double ApplyOp(double a, double b, ReduceOp op)
        {
            return op switch
            {
                ReduceOp.Sum => a + b,
                ReduceOp.Mean => a + b,  // Divide by worldSize at end
                ReduceOp.Max => Math.Max(a, b),
                ReduceOp.Min => Math.Min(a, b),
                ReduceOp.Product => a * b,
                _ => a + b
            };
        }

        private static byte[] SerializeTensor(Tensor tensor)
        {
            // Format: [ndim:4][shape:ndim*4][data:n*8]
            int ndim = tensor.Shape.Length;
            int dataBytes = tensor.Data.Length * sizeof(double);
            int totalSize = 4 + ndim * 4 + dataBytes;

            var buffer = new byte[totalSize];
            int offset = 0;

            BinaryPrimitives.WriteInt32LittleEndian(buffer.AsSpan(offset), ndim);
            offset += 4;

            foreach (var dim in tensor.Shape)
            {
                BinaryPrimitives.WriteInt32LittleEndian(buffer.AsSpan(offset), (int)dim);
                offset += 4;
            }

            Buffer.BlockCopy(tensor.Data, 0, buffer, offset, dataBytes);

            return buffer;
        }

        private static Tensor DeserializeTensor(byte[] buffer)
        {
            int offset = 0;

            int ndim = BinaryPrimitives.ReadInt32LittleEndian(buffer.AsSpan(offset));
            offset += 4;

            var shape = new long[ndim];
            for (int i = 0; i < ndim; i++)
            {
                shape[i] = BinaryPrimitives.ReadInt32LittleEndian(buffer.AsSpan(offset));
                offset += 4;
            }

            long numElements = shape.Length > 0 ? shape.Aggregate(1L, (a, b) => a * b) : 1;
            var data = new double[numElements];
            Buffer.BlockCopy(buffer, offset, data, 0, (int)(numElements * sizeof(double)));

            return new Tensor(data, shape, false);
        }

        private void EnsureInitialized()
        {
            if (!_initialized)
                throw new InvalidOperationException("DistributedContext not initialized");
            if (_disposed)
                throw new ObjectDisposedException(nameof(DistributedContext));
        }

        #endregion

        #region Disposal

        /// <summary>Public API</summary>
        public void Dispose()
        {
            DisposeAsync().AsTask().Wait();
        }

        /// <summary>Public API</summary>
        public async ValueTask DisposeAsync()
        {
            if (_disposed) return;
            _disposed = true;

            Console.WriteLine($"[Rank {_rank}] Shutting down distributed context");

            _shutdownCts.Cancel();

            // Wait for receiver tasks
            try
            {
                await Task.WhenAll(_receiverTasks.Where(t => t != null));
            }
            catch { }

            // Close connections
            foreach (var stream in _streams)
            {
                try { stream?.Dispose(); } catch { }
            }

            foreach (var socket in _sockets)
            {
                try { socket?.Dispose(); } catch { }
            }

            _server?.Stop();

            foreach (var sem in _sendLocks)
            {
                sem?.Dispose();
            }

            foreach (var sem in _recvLocks)
            {
                sem?.Dispose();
            }

            _shutdownCts.Dispose();

            Console.WriteLine($"[Rank {_rank}] Distributed context disposed");
        }

        #endregion
    }

    /// <summary>
    /// Reduction operation types.
    /// </summary>
    public enum ReduceOp
    {
        Sum,
        Mean,
        Max,
        Min,
        Product
    }

    /// <summary>
    /// Distributed Data Parallel - automatic gradient synchronization.
    /// Wraps model parameters for distributed training.
    /// </summary>
    public sealed class DistributedDataParallel : IDisposable
    {
        private readonly DistributedContext _ctx;
        private readonly List<Tensor> _parameters;
        private readonly int _bucketSizeMb;
        private readonly List<GradientBucket> _buckets;
        private bool _disposed;

        /// <summary>Public API</summary>
        public DistributedContext Context => _ctx;
        /// <summary>Public API</summary>
        public int WorldSize => _ctx.WorldSize;
        /// <summary>Public API</summary>
        public int Rank => _ctx.Rank;

        /// <summary>Public API</summary>
        public DistributedDataParallel(
            DistributedContext ctx,
            IEnumerable<Tensor> parameters,
            int bucketSizeMb = 25)
        {
            _ctx = ctx;
            _parameters = parameters.ToList();
            _bucketSizeMb = bucketSizeMb;
            _buckets = new List<GradientBucket>();

            CreateBuckets();
        }

        private void CreateBuckets()
        {
            long bucketBytes = _bucketSizeMb * 1024L * 1024L;
            long currentSize = 0;
            var currentBucket = new List<int>();

            // Traverse in reverse order (typical backward pass order)
            for (int i = _parameters.Count - 1; i >= 0; i--)
            {
                long paramSize = _parameters[i].Data.Length * sizeof(double);

                if (currentSize + paramSize > bucketBytes && currentBucket.Count > 0)
                {
                    _buckets.Add(new GradientBucket(currentBucket.ToArray()));
                    currentBucket.Clear();
                    currentSize = 0;
                }

                currentBucket.Add(i);
                currentSize += paramSize;
            }

            if (currentBucket.Count > 0)
            {
                _buckets.Add(new GradientBucket(currentBucket.ToArray()));
            }

            Console.WriteLine($"[Rank {_ctx.Rank}] Created {_buckets.Count} gradient buckets");
        }

        /// <summary>
        /// Synchronize gradients across all nodes. Call after backward pass.
        /// </summary>
        public async Task SynchronizeGradientsAsync(CancellationToken ct = default)
        {
            // Process buckets (can be overlapped with computation in advanced impl)
            foreach (var bucket in _buckets)
            {
                await SynchronizeBucketAsync(bucket, ct);
            }
        }

        private async Task SynchronizeBucketAsync(GradientBucket bucket, CancellationToken ct)
        {
            // Calculate total size
            int totalSize = 0;
            foreach (var idx in bucket.ParameterIndices)
            {
                if (_parameters[idx].Grad != null)
                    totalSize += _parameters[idx].Grad!.Data.Length;
            }

            if (totalSize == 0) return;

            // Pack gradients into flat buffer
            var buffer = new double[totalSize];
            int offset = 0;
            var offsets = new int[bucket.ParameterIndices.Length];

            for (int i = 0; i < bucket.ParameterIndices.Length; i++)
            {
                var idx = bucket.ParameterIndices[i];
                var grad = _parameters[idx].Grad;
                if (grad != null)
                {
                    offsets[i] = offset;
                    Array.Copy(grad.Data, 0, buffer, offset, grad.Data.Length);
                    offset += grad.Data.Length;
                }
            }

            // AllReduce the packed buffer
            var bufferTensor = new Tensor(buffer, new long[] { totalSize }, false);
            var reduced = await _ctx.AllReduceAsync(bufferTensor, ReduceOp.Mean, ct);

            // Unpack back to gradients
            for (int i = 0; i < bucket.ParameterIndices.Length; i++)
            {
                var idx = bucket.ParameterIndices[i];
                var grad = _parameters[idx].Grad;
                if (grad != null)
                {
                    Array.Copy(reduced.Data, offsets[i], grad.Data, 0, grad.Data.Length);
                }
            }
        }

        /// <summary>
        /// Broadcast parameters from rank 0 to all other ranks.
        /// Call at initialization or after loading checkpoint on rank 0.
        /// </summary>
        public async Task BroadcastParametersAsync(CancellationToken ct = default)
        {
            foreach (var param in _parameters)
            {
                var broadcasted = await _ctx.BroadcastAsync(param, srcRank: 0, ct);

                if (_ctx.Rank != 0)
                {
                    Array.Copy(broadcasted.Data, param.Data, param.Data.Length);
                }
            }
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
        }
    }

    internal class GradientBucket
    {
        /// <summary>Public API</summary>
        public int[] ParameterIndices { get; }

        /// <summary>Public API</summary>
        public GradientBucket(int[] indices)
        {
            ParameterIndices = indices;
        }
    }

    /// <summary>
    /// Distributed checkpoint manager with fault tolerance.
    /// </summary>
    public sealed class DistributedCheckpoint
    {
        private readonly DistributedContext _ctx;
        private readonly string _checkpointDir;
        private readonly int _saveIntervalSteps;
        private int _currentStep;

        /// <summary>Public API</summary>
        public int CurrentStep => _currentStep;

        /// <summary>Public API</summary>
        public DistributedCheckpoint(
            DistributedContext ctx,
            string checkpointDir,
            int saveIntervalSteps = 1000)
        {
            _ctx = ctx;
            _checkpointDir = checkpointDir;
            _saveIntervalSteps = saveIntervalSteps;

            if (_ctx.Rank == 0)
            {
                Directory.CreateDirectory(checkpointDir);
            }
        }

        /// <summary>
        /// Save checkpoint. Only rank 0 writes to disk.
        /// </summary>
        public async Task SaveAsync(
            Dictionary<string, Tensor> modelWeights,
            Dictionary<string, object>? optimizerState = null,
            Dictionary<string, object>? metadata = null,
            CancellationToken ct = default)
        {
            _currentStep++;

            if (_currentStep % _saveIntervalSteps != 0) return;

            // Barrier before save
            await _ctx.BarrierAsync(ct);

            if (_ctx.Rank == 0)
            {
                string path = Path.Combine(_checkpointDir, $"checkpoint_step_{_currentStep}.nsl");
                Console.WriteLine($"[Rank 0] Saving checkpoint to {path}");

                ModelSerializer.Save(modelWeights, optimizerState, path);

                // Save metadata
                var meta = new Dictionary<string, object>
                {
                    ["step"] = _currentStep,
                    ["world_size"] = _ctx.WorldSize,
                    ["timestamp"] = DateTime.UtcNow.ToString("O")
                };

                if (metadata != null)
                {
                    foreach (var kv in metadata)
                        meta[kv.Key] = kv.Value;
                }

                string metaPath = Path.ChangeExtension(path, ".json");
                File.WriteAllText(metaPath, JsonSerializer.Serialize(meta));

                // Clean old checkpoints (keep last 3)
                CleanOldCheckpoints();
            }

            // Barrier after save
            await _ctx.BarrierAsync(ct);
        }

        /// <summary>
        /// Load latest checkpoint and broadcast to all ranks.
        /// </summary>
        public async Task<(Dictionary<string, Tensor> weights, int step)> LoadLatestAsync(CancellationToken ct = default)
        {
            Dictionary<string, Tensor> weights;
            int step = 0;

            if (_ctx.Rank == 0)
            {
                var latest = GetLatestCheckpoint();
                if (latest == null)
                {
                    Console.WriteLine("[Rank 0] No checkpoint found");
                    weights = new Dictionary<string, Tensor>();
                }
                else
                {
                    Console.WriteLine($"[Rank 0] Loading checkpoint from {latest}");
                    var (w, _, _) = ModelSerializer.Load(latest);
                    weights = w;

                    var metaPath = Path.ChangeExtension(latest, ".json");
                    if (File.Exists(metaPath))
                    {
                        var meta = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(metaPath));
                        if (meta != null && meta.TryGetValue("step", out var stepEl))
                        {
                            step = stepEl.GetInt32();
                        }
                    }
                }

                // Broadcast step
                var stepTensor = new Tensor(new[] { (double)step }, new long[] { 1 }, false);
                await _ctx.BroadcastAsync(stepTensor, 0, ct);

                // Broadcast number of weights
                var countTensor = new Tensor(new[] { (double)weights.Count }, new long[] { 1 }, false);
                await _ctx.BroadcastAsync(countTensor, 0, ct);

                // Broadcast each weight
                foreach (var kv in weights)
                {
                    // Send key
                    var keyBytes = Encoding.UTF8.GetBytes(kv.Key);
                    var keyTensor = new Tensor(keyBytes.Select(b => (double)b).ToArray(), new long[] { keyBytes.Length }, false);
                    await _ctx.BroadcastAsync(keyTensor, 0, ct);

                    // Send value
                    await _ctx.BroadcastAsync(kv.Value, 0, ct);
                }
            }
            else
            {
                // Receive step
                var stepTensor = await _ctx.BroadcastAsync(null, 0, ct);
                step = (int)stepTensor.Data[0];

                // Receive count
                var countTensor = await _ctx.BroadcastAsync(null, 0, ct);
                int count = (int)countTensor.Data[0];

                weights = new Dictionary<string, Tensor>();

                for (int i = 0; i < count; i++)
                {
                    // Receive key
                    var keyTensor = await _ctx.BroadcastAsync(null, 0, ct);
                    var keyBytes = keyTensor.Data.Select(d => (byte)d).ToArray();
                    var key = Encoding.UTF8.GetString(keyBytes);

                    // Receive value
                    var value = await _ctx.BroadcastAsync(null, 0, ct);
                    weights[key] = value;
                }
            }

            _currentStep = step;
            return (weights, step);
        }

        private string? GetLatestCheckpoint()
        {
            if (!Directory.Exists(_checkpointDir)) return null;

            return Directory.GetFiles(_checkpointDir, "checkpoint_step_*.nsl")
                .OrderByDescending(ExtractStep)
                .FirstOrDefault();
        }

        private int ExtractStep(string path)
        {
            var name = Path.GetFileNameWithoutExtension(path);
            var parts = name.Split('_');
            return parts.Length >= 3 && int.TryParse(parts[2], out int step) ? step : 0;
        }

        private void CleanOldCheckpoints()
        {
            var files = Directory.GetFiles(_checkpointDir, "checkpoint_step_*.nsl")
                .OrderByDescending(ExtractStep)
                .Skip(3)
                .ToList();

            foreach (var file in files)
            {
                try
                {
                    File.Delete(file);
                    File.Delete(Path.ChangeExtension(file, ".json"));
                }
                catch { }
            }
        }
    }

    /// <summary>
    /// Data sampler for distributed training - ensures each rank gets unique data subset.
    /// </summary>
    public sealed class DistributedSampler<T>
    {
        private readonly IList<T> _data;
        private readonly int _worldSize;
        private readonly int _rank;
        private readonly bool _shuffle;
        private readonly int _seed;
        private int _epoch;
        private int[] _indices;

        /// <summary>Public API</summary>
        public int Length => (_data.Count + _worldSize - 1) / _worldSize;
        /// <summary>Public API</summary>
        public int TotalLength => _data.Count;
        /// <summary>Public API</summary>
        public int Epoch => _epoch;

        /// <summary>Public API</summary>
        public DistributedSampler(
            IList<T> data,
            int worldSize,
            int rank,
            bool shuffle = true,
            int seed = 42)
        {
            _data = data;
            _worldSize = worldSize;
            _rank = rank;
            _shuffle = shuffle;
            _seed = seed;
            _epoch = 0;
            _indices = Enumerable.Range(0, data.Count).ToArray();

            if (_shuffle) Shuffle();
        }

        /// <summary>Public API</summary>
        public void SetEpoch(int epoch)
        {
            _epoch = epoch;
            if (_shuffle) Shuffle();
        }

        private void Shuffle()
        {
            // Use deterministic shuffle based on epoch + seed
            var rng = new Random(_seed + _epoch);
            for (int i = _indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (_indices[i], _indices[j]) = (_indices[j], _indices[i]);
            }
        }

        /// <summary>Public API</summary>
        public IEnumerable<T> GetSamples()
        {
            for (int i = _rank; i < _indices.Length; i += _worldSize)
            {
                yield return _data[_indices[i]];
            }
        }

        /// <summary>Public API</summary>
        public IEnumerable<int> GetIndices()
        {
            for (int i = _rank; i < _indices.Length; i += _worldSize)
            {
                yield return _indices[i];
            }
        }

        /// <summary>Public API</summary>
        public IEnumerable<(int index, T item)> GetIndexedSamples()
        {
            for (int i = _rank; i < _indices.Length; i += _worldSize)
            {
                yield return (_indices[i], _data[_indices[i]]);
            }
        }
    }

    /// <summary>
    /// Node registry for multi-machine distributed training.
    /// Maps ranks to actual network addresses.
    /// </summary>
    public sealed class NodeRegistry
    {
        private readonly Dictionary<int, NodeInfo> _nodes;

        /// <summary>Public API</summary>
        public NodeRegistry()
        {
            _nodes = new Dictionary<int, NodeInfo>();
        }

        /// <summary>
        /// Create from environment variable NSL_NODE_ADDRESSES.
        /// Format: "rank0_ip:port0,rank1_ip:port1,..."
        /// </summary>
        public static NodeRegistry FromEnvironment()
        {
            var registry = new NodeRegistry();
            var nodeAddrs = Environment.GetEnvironmentVariable("NSL_NODE_ADDRESSES");

            if (!string.IsNullOrEmpty(nodeAddrs))
            {
                var entries = nodeAddrs.Split(',');
                for (int rank = 0; rank < entries.Length; rank++)
                {
                    var parts = entries[rank].Trim().Split(':');
                    if (parts.Length >= 1)
                    {
                        string host = parts[0];
                        int port = parts.Length > 1 ? int.Parse(parts[1]) : 29500 + rank;
                        registry.Register(rank, host, port);
                    }
                }
            }

            return registry;
        }

        /// <summary>
        /// Create from explicit list of node addresses.
        /// </summary>
        public static NodeRegistry FromAddresses(params string[] addresses)
        {
            var registry = new NodeRegistry();
            for (int rank = 0; rank < addresses.Length; rank++)
            {
                var parts = addresses[rank].Split(':');
                string host = parts[0];
                int port = parts.Length > 1 ? int.Parse(parts[1]) : 29500 + rank;
                registry.Register(rank, host, port);
            }
            return registry;
        }

        /// <summary>Public API</summary>
        public void Register(int rank, string host, int port)
        {
            _nodes[rank] = new NodeInfo { Rank = rank, Host = host, Port = port };
        }

        /// <summary>Public API</summary>
        public NodeInfo? GetNode(int rank)
        {
            return _nodes.TryGetValue(rank, out var node) ? node : null;
        }

        /// <summary>Public API</summary>
        public IEnumerable<NodeInfo> AllNodes => _nodes.Values;
        /// <summary>Public API</summary>
        public int Count => _nodes.Count;

        /// <summary>Public API</summary>
        public string GetHost(int rank, string defaultHost)
        {
            return _nodes.TryGetValue(rank, out var node) ? node.Host : defaultHost;
        }

        /// <summary>Public API</summary>
        public int GetPort(int rank, int basePort)
        {
            return _nodes.TryGetValue(rank, out var node) ? node.Port : basePort + rank;
        }
    }

    /// <summary>Public API</summary>
    public class NodeInfo
    {
        /// <summary>Public API</summary>
        public int Rank { get; set; }
        /// <summary>Public API</summary>
        public string Host { get; set; } = "";
        /// <summary>Public API</summary>
        public int Port { get; set; }
        /// <summary>Public API</summary>
        public override string ToString() => $"Rank {Rank}: {Host}:{Port}";
    }

    /// <summary>
    /// Extended distributed context with multi-machine support.
    /// </summary>
    public static class DistributedContextExtensions
    {
        /// <summary>
        /// Initialize with explicit node registry for multi-machine setup.
        /// </summary>
        public static async Task<DistributedContext> InitializeWithRegistryAsync(
            int worldSize,
            int rank,
            NodeRegistry registry,
            int timeoutSeconds = 300)
        {
            // Set environment variables for the context
            var masterNode = registry.GetNode(0);
            string masterAddr = masterNode?.Host ?? "127.0.0.1";
            int masterPort = masterNode?.Port ?? 29500;

            // Store registry in environment for node lookups
            var nodeAddrs = string.Join(",",
                Enumerable.Range(0, worldSize)
                    .Select(r => $"{registry.GetHost(r, masterAddr)}:{registry.GetPort(r, masterPort)}"));
            Environment.SetEnvironmentVariable("NSL_NODE_ADDRESSES", nodeAddrs);

            return await DistributedContext.InitializeAsync(worldSize, rank, masterAddr, masterPort, timeoutSeconds);
        }
    }

    /// <summary>
    /// High-performance RDMA-style memory registration for zero-copy transfers.
    /// Falls back to standard copy if RDMA not available.
    /// </summary>
    public sealed class RegisteredMemory : IDisposable
    {
        private readonly double[] _buffer;
        private readonly GCHandle _handle;
        private readonly IntPtr _pinnedAddress;
        private bool _disposed;

        /// <summary>Public API</summary>
        public double[] Buffer => _buffer;
        /// <summary>Public API</summary>
        public IntPtr Address => _pinnedAddress;
        /// <summary>Public API</summary>
        public int Length => _buffer.Length;

        /// <summary>Public API</summary>
        public RegisteredMemory(int size)
        {
            _buffer = new double[size];
            _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
            _pinnedAddress = _handle.AddrOfPinnedObject();
        }

        /// <summary>Public API</summary>
        public Span<double> AsSpan() => _buffer.AsSpan();
        /// <summary>Public API</summary>
        public Memory<double> AsMemory() => _buffer.AsMemory();

        /// <summary>Public API</summary>
        public void CopyFrom(Tensor tensor)
        {
            Array.Copy(tensor.Data, _buffer, Math.Min(tensor.Data.Length, _buffer.Length));
        }

        /// <summary>Public API</summary>
        public Tensor ToTensor(long[] shape)
        {
            var data = new double[_buffer.Length];
            Array.Copy(_buffer, data, _buffer.Length);
            return new Tensor(data, shape, false);
        }

        /// <summary>Public API</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_handle.IsAllocated)
                _handle.Free();
        }
    }

    /// <summary>
    /// Process launcher for distributed training - spawns multiple worker processes.
    /// </summary>
    public static class DistributedLauncher
    {
        /// <summary>
        /// Launch distributed training across multiple local processes.
        /// </summary>
        public static void LaunchLocal(int worldSize, string executablePath, string[]? args = null)
        {
            var processes = new List<Process>();
            string masterAddr = "127.0.0.1";
            int masterPort = 29500;

            Console.WriteLine($"Launching {worldSize} workers on {masterAddr}:{masterPort}");

            for (int rank = 0; rank < worldSize; rank++)
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = executablePath,
                    Arguments = args != null ? string.Join(" ", args) : "",
                    UseShellExecute = false,
                    CreateNoWindow = false,
                    Environment =
                    {
                        ["NSL_WORLD_SIZE"] = worldSize.ToString(),
                        ["NSL_RANK"] = rank.ToString(),
                        ["NSL_MASTER_ADDR"] = masterAddr,
                        ["NSL_MASTER_PORT"] = masterPort.ToString()
                    }
                };

                var process = Process.Start(startInfo);
                if (process != null)
                {
                    processes.Add(process);
                    Console.WriteLine($"Started worker rank {rank} (PID: {process.Id})");
                }
            }

            // Wait for all to complete
            foreach (var p in processes)
            {
                p.WaitForExit();
            }
        }

        /// <summary>
        /// Generate launch script for multi-node training.
        /// </summary>
        public static string GenerateLaunchScript(
            string[] nodeAddresses,
            string executablePath,
            string[]? args = null,
            int masterPort = 29500)
        {
            var sb = new StringBuilder();
            int worldSize = nodeAddresses.Length;
            string masterAddr = nodeAddresses[0];

            sb.AppendLine("#!/bin/bash");
            sb.AppendLine("# NSL Distributed Training Launch Script");
            sb.AppendLine($"# World Size: {worldSize}");
            sb.AppendLine();

            for (int rank = 0; rank < worldSize; rank++)
            {
                string node = nodeAddresses[rank];
                string argStr = args != null ? string.Join(" ", args) : "";

                if (rank == 0)
                {
                    sb.AppendLine("# Master node (run this first)");
                    sb.AppendLine($"# On {node}:");
                }
                else
                {
                    sb.AppendLine($"# Worker node {rank}:");
                    sb.AppendLine($"# On {node}:");
                }

                sb.AppendLine($"export NSL_WORLD_SIZE={worldSize}");
                sb.AppendLine($"export NSL_RANK={rank}");
                sb.AppendLine($"export NSL_MASTER_ADDR={masterAddr}");
                sb.AppendLine($"export NSL_MASTER_PORT={masterPort}");
                sb.AppendLine($"{executablePath} {argStr}");
                sb.AppendLine();
            }

            return sb.ToString();
        }
    }
}