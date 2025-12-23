using System;
using System.Collections.Generic;
using System.Linq;

namespace NSL.Tensor
{
    /// <summary>
    /// Graph utilities for computational graph operations.
    /// Inspired by pmx_utils graph module.
    /// Useful for autograd optimization, dependency analysis, etc.
    /// </summary>
    public class Graph<T> where T : notnull
    {
        private readonly Dictionary<T, HashSet<T>> _adjacency;
        private readonly Dictionary<T, HashSet<T>> _reverseAdjacency;
        private readonly Dictionary<T, double> _nodeWeights;
        private readonly Dictionary<(T, T), double> _edgeWeights;
        private readonly bool _directed;

        /// <summary>
        /// Create a new graph.
        /// </summary>
        /// <param name="directed">Whether the graph is directed (default: true)</param>
        public Graph(bool directed = true)
        {
            _directed = directed;
            _adjacency = new Dictionary<T, HashSet<T>>();
            _reverseAdjacency = new Dictionary<T, HashSet<T>>();
            _nodeWeights = new Dictionary<T, double>();
            _edgeWeights = new Dictionary<(T, T), double>();
        }

        /// <summary>Add a node to the graph</summary>
        public void AddNode(T node, double weight = 1.0)
        {
            if (!_adjacency.ContainsKey(node))
            {
                _adjacency[node] = new HashSet<T>();
                _reverseAdjacency[node] = new HashSet<T>();
            }
            _nodeWeights[node] = weight;
        }

        /// <summary>Add an edge between nodes</summary>
        public void AddEdge(T from, T to, double weight = 1.0)
        {
            AddNode(from);
            AddNode(to);

            _adjacency[from].Add(to);
            _reverseAdjacency[to].Add(from);
            _edgeWeights[(from, to)] = weight;

            if (!_directed)
            {
                _adjacency[to].Add(from);
                _reverseAdjacency[from].Add(to);
                _edgeWeights[(to, from)] = weight;
            }
        }

        /// <summary>Remove an edge</summary>
        public bool RemoveEdge(T from, T to)
        {
            if (_adjacency.TryGetValue(from, out var neighbors))
            {
                if (neighbors.Remove(to))
                {
                    _reverseAdjacency[to].Remove(from);
                    _edgeWeights.Remove((from, to));

                    if (!_directed)
                    {
                        _adjacency[to].Remove(from);
                        _reverseAdjacency[from].Remove(to);
                        _edgeWeights.Remove((to, from));
                    }
                    return true;
                }
            }
            return false;
        }

        /// <summary>Remove a node and all its edges</summary>
        public bool RemoveNode(T node)
        {
            if (!_adjacency.ContainsKey(node)) return false;

            // Remove outgoing edges
            foreach (var neighbor in _adjacency[node].ToList())
            {
                _reverseAdjacency[neighbor].Remove(node);
                _edgeWeights.Remove((node, neighbor));
            }

            // Remove incoming edges
            foreach (var predecessor in _reverseAdjacency[node].ToList())
            {
                _adjacency[predecessor].Remove(node);
                _edgeWeights.Remove((predecessor, node));
            }

            _adjacency.Remove(node);
            _reverseAdjacency.Remove(node);
            _nodeWeights.Remove(node);

            return true;
        }

        /// <summary>Get all nodes</summary>
        public IEnumerable<T> Nodes => _adjacency.Keys;

        /// <summary>Get successors (outgoing neighbors) of a node</summary>
        public IEnumerable<T> Successors(T node)
        {
            return _adjacency.TryGetValue(node, out var neighbors) ? neighbors : Enumerable.Empty<T>();
        }

        /// <summary>Get predecessors (incoming neighbors) of a node</summary>
        public IEnumerable<T> Predecessors(T node)
        {
            return _reverseAdjacency.TryGetValue(node, out var neighbors) ? neighbors : Enumerable.Empty<T>();
        }

        /// <summary>Check if edge exists</summary>
        public bool HasEdge(T from, T to)
        {
            return _adjacency.TryGetValue(from, out var neighbors) && neighbors.Contains(to);
        }

        /// <summary>Get edge weight</summary>
        public double GetEdgeWeight(T from, T to, double defaultValue = 0.0)
        {
            return _edgeWeights.TryGetValue((from, to), out var weight) ? weight : defaultValue;
        }

        /// <summary>Number of nodes</summary>
        public int NodeCount => _adjacency.Count;

        /// <summary>Number of edges</summary>
        public int EdgeCount => _edgeWeights.Count / (_directed ? 1 : 2);

        /// <summary>In-degree of a node</summary>
        public int InDegree(T node)
        {
            return _reverseAdjacency.TryGetValue(node, out var pred) ? pred.Count : 0;
        }

        /// <summary>Out-degree of a node</summary>
        public int OutDegree(T node)
        {
            return _adjacency.TryGetValue(node, out var succ) ? succ.Count : 0;
        }

        #region Graph Algorithms

        /// <summary>
        /// Topological sort using Kahn's algorithm.
        /// Returns nodes in order such that for every edge (u,v), u comes before v.
        /// </summary>
        /// <exception cref="InvalidOperationException">If graph contains a cycle</exception>
        public List<T> TopologicalSort()
        {
            if (!_directed)
                throw new InvalidOperationException("Topological sort requires a directed graph");

            var inDegree = new Dictionary<T, int>();
            foreach (var node in _adjacency.Keys)
                inDegree[node] = InDegree(node);

            var queue = new Queue<T>();
            foreach (var node in inDegree.Keys.Where(n => inDegree[n] == 0))
                queue.Enqueue(node);

            var result = new List<T>();
            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                result.Add(node);

                foreach (var successor in Successors(node))
                {
                    inDegree[successor]--;
                    if (inDegree[successor] == 0)
                        queue.Enqueue(successor);
                }
            }

            if (result.Count != _adjacency.Count)
                throw new InvalidOperationException("Graph contains a cycle - topological sort not possible");

            return result;
        }

        /// <summary>
        /// Check if the graph is a DAG (Directed Acyclic Graph).
        /// </summary>
        public bool IsDAG()
        {
            if (!_directed) return false;

            try
            {
                TopologicalSort();
                return true;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
        }

        /// <summary>
        /// Detect if graph contains any cycle.
        /// Returns the cycle path if found, null otherwise.
        /// </summary>
        public List<T>? DetectCycle()
        {
            var visited = new HashSet<T>();
            var recStack = new HashSet<T>();
            var path = new List<T>();

            foreach (var node in _adjacency.Keys)
            {
                if (!visited.Contains(node))
                {
                    var cycle = DFSCycleDetect(node, visited, recStack, path);
                    if (cycle != null) return cycle;
                }
            }

            return null;
        }

        private List<T>? DFSCycleDetect(T node, HashSet<T> visited, HashSet<T> recStack, List<T> path)
        {
            visited.Add(node);
            recStack.Add(node);
            path.Add(node);

            foreach (var successor in Successors(node))
            {
                if (!visited.Contains(successor))
                {
                    var cycle = DFSCycleDetect(successor, visited, recStack, path);
                    if (cycle != null) return cycle;
                }
                else if (recStack.Contains(successor))
                {
                    // Found cycle - extract it
                    var cycleStart = path.IndexOf(successor);
                    return path.Skip(cycleStart).Append(successor).ToList();
                }
            }

            path.RemoveAt(path.Count - 1);
            recStack.Remove(node);
            return null;
        }

        /// <summary>
        /// Find strongly connected components using Tarjan's algorithm.
        /// Each SCC is a maximal set of nodes where every node is reachable from every other.
        /// </summary>
        public List<List<T>> StronglyConnectedComponents()
        {
            if (!_directed)
                return ConnectedComponents();

            var index = 0;
            var indices = new Dictionary<T, int>();
            var lowlinks = new Dictionary<T, int>();
            var onStack = new HashSet<T>();
            var stack = new Stack<T>();
            var sccs = new List<List<T>>();

            void StrongConnect(T node)
            {
                indices[node] = index;
                lowlinks[node] = index;
                index++;
                stack.Push(node);
                onStack.Add(node);

                foreach (var successor in Successors(node))
                {
                    if (!indices.ContainsKey(successor))
                    {
                        StrongConnect(successor);
                        lowlinks[node] = Math.Min(lowlinks[node], lowlinks[successor]);
                    }
                    else if (onStack.Contains(successor))
                    {
                        lowlinks[node] = Math.Min(lowlinks[node], indices[successor]);
                    }
                }

                if (lowlinks[node] == indices[node])
                {
                    var scc = new List<T>();
                    T w;
                    do
                    {
                        w = stack.Pop();
                        onStack.Remove(w);
                        scc.Add(w);
                    } while (!EqualityComparer<T>.Default.Equals(w, node));

                    sccs.Add(scc);
                }
            }

            foreach (var node in _adjacency.Keys)
            {
                if (!indices.ContainsKey(node))
                    StrongConnect(node);
            }

            return sccs;
        }

        /// <summary>
        /// Find connected components (for undirected graphs or weak connectivity).
        /// </summary>
        public List<List<T>> ConnectedComponents()
        {
            var visited = new HashSet<T>();
            var components = new List<List<T>>();

            foreach (var node in _adjacency.Keys)
            {
                if (!visited.Contains(node))
                {
                    var component = new List<T>();
                    var queue = new Queue<T>();
                    queue.Enqueue(node);

                    while (queue.Count > 0)
                    {
                        var current = queue.Dequeue();
                        if (visited.Contains(current)) continue;

                        visited.Add(current);
                        component.Add(current);

                        foreach (var neighbor in Successors(current))
                            if (!visited.Contains(neighbor))
                                queue.Enqueue(neighbor);

                        if (_directed)
                        {
                            foreach (var neighbor in Predecessors(current))
                                if (!visited.Contains(neighbor))
                                    queue.Enqueue(neighbor);
                        }
                    }

                    components.Add(component);
                }
            }

            return components;
        }

        /// <summary>
        /// Breadth-first search from a starting node.
        /// </summary>
        public IEnumerable<T> BFS(T start)
        {
            if (!_adjacency.ContainsKey(start))
                yield break;

            var visited = new HashSet<T>();
            var queue = new Queue<T>();
            queue.Enqueue(start);

            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                if (visited.Contains(node)) continue;

                visited.Add(node);
                yield return node;

                foreach (var neighbor in Successors(node))
                    if (!visited.Contains(neighbor))
                        queue.Enqueue(neighbor);
            }
        }

        /// <summary>
        /// Depth-first search from a starting node.
        /// </summary>
        public IEnumerable<T> DFS(T start)
        {
            if (!_adjacency.ContainsKey(start))
                yield break;

            var visited = new HashSet<T>();
            var stack = new Stack<T>();
            stack.Push(start);

            while (stack.Count > 0)
            {
                var node = stack.Pop();
                if (visited.Contains(node)) continue;

                visited.Add(node);
                yield return node;

                foreach (var neighbor in Successors(node).Reverse())
                    if (!visited.Contains(neighbor))
                        stack.Push(neighbor);
            }
        }

        /// <summary>
        /// Dijkstra's shortest path algorithm.
        /// Returns distances from start to all reachable nodes.
        /// </summary>
        public Dictionary<T, double> Dijkstra(T start)
        {
            var distances = new Dictionary<T, double>();
            var visited = new HashSet<T>();
            var pq = new SortedSet<(double dist, T node)>(
                Comparer<(double, T)>.Create((a, b) =>
                {
                    var cmp = a.Item1.CompareTo(b.Item1);
                    return cmp != 0 ? cmp : a.Item2!.GetHashCode().CompareTo(b.Item2!.GetHashCode());
                }));

            distances[start] = 0;
            pq.Add((0, start));

            while (pq.Count > 0)
            {
                var (dist, node) = pq.Min;
                pq.Remove(pq.Min);

                if (visited.Contains(node)) continue;
                visited.Add(node);

                foreach (var neighbor in Successors(node))
                {
                    if (visited.Contains(neighbor)) continue;

                    var newDist = dist + GetEdgeWeight(node, neighbor, 1.0);
                    if (!distances.ContainsKey(neighbor) || newDist < distances[neighbor])
                    {
                        distances[neighbor] = newDist;
                        pq.Add((newDist, neighbor));
                    }
                }
            }

            return distances;
        }

        /// <summary>
        /// Prune edges with weight below threshold.
        /// </summary>
        public void PruneWeakEdges(double threshold)
        {
            var toRemove = _edgeWeights
                .Where(kv => kv.Value < threshold)
                .Select(kv => kv.Key)
                .ToList();

            foreach (var (from, to) in toRemove)
                RemoveEdge(from, to);
        }

        /// <summary>
        /// Get reversed graph (all edges reversed).
        /// </summary>
        public Graph<T> Reverse()
        {
            var reversed = new Graph<T>(_directed);

            foreach (var node in _adjacency.Keys)
                reversed.AddNode(node, _nodeWeights.GetValueOrDefault(node, 1.0));

            foreach (var (from, to) in _edgeWeights.Keys)
                reversed.AddEdge(to, from, _edgeWeights[(from, to)]);

            return reversed;
        }

        #endregion
    }

    /// <summary>
    /// Graph operations for computational graphs.
    /// </summary>
    public static class GraphOps
    {
        /// <summary>
        /// Build a computational graph from a tensor by tracing gradients.
        /// </summary>
        public static Graph<Tensor> BuildComputationGraph(Tensor output)
        {
            var graph = new Graph<Tensor>();
            var visited = new HashSet<Tensor>();
            var queue = new Queue<Tensor>();

            queue.Enqueue(output);
            graph.AddNode(output);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (visited.Contains(current)) continue;
                visited.Add(current);

                // Follow gradient function to find inputs
                // This would need access to the gradient function's inputs
                // For now, just add the node
                graph.AddNode(current);
            }

            return graph;
        }
    }
}
