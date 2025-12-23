using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NSL.Tensor.NN;

namespace NSL.Tensor.Data
{
    /// <summary>
    /// Abstract base class for all datasets
    /// </summary>
    public abstract class Dataset : IEnumerable<object>
    {
        /// <summary>
        /// Get the number of samples in the dataset
        /// </summary>
        public abstract int Length { get; }

        /// <summary>
        /// Get a sample by index
        /// </summary>
        public abstract object GetItem(int index);

        /// <summary>Public API</summary>
        public object this[int index] => GetItem(index);

        /// <summary>Public API</summary>
        public IEnumerator<object> GetEnumerator()
        {
            for (int i = 0; i < Length; i++)
                yield return GetItem(i);
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    /// <summary>
    /// A dataset that wraps tensors
    /// </summary>
    public class TensorDataset : Dataset
    {
        private readonly Tensor[] _tensors;
        private readonly int _length;

        /// <summary>Public API</summary>
        public TensorDataset(params Tensor[] tensors)
        {
            if (tensors.Length == 0)
                throw new ArgumentException("At least one tensor is required");

            _length = (int)tensors[0].Shape[0];

            foreach (var tensor in tensors)
            {
                if (tensor.Shape[0] != _length)
                    throw new ArgumentException("All tensors must have the same first dimension");
            }

            _tensors = tensors;
        }

        /// <summary>Public API</summary>
        public override int Length => _length;

        /// <summary>Public API</summary>
        public override object GetItem(int index)
        {
            if (_tensors.Length == 1)
                return _tensors[0].Slice(0, index, index + 1).Squeeze(0);

            return _tensors.Select(t => t.Slice(0, index, index + 1).Squeeze(0)).ToArray();
        }
    }

    /// <summary>
    /// Concatenates multiple datasets
    /// </summary>
    public class ConcatDataset : Dataset
    {
        private readonly Dataset[] _datasets;
        private readonly int[] _cumulativeSizes;

        /// <summary>Public API</summary>
        public ConcatDataset(params Dataset[] datasets)
        {
            _datasets = datasets;
            _cumulativeSizes = new int[datasets.Length];

            int cumulative = 0;
            for (int i = 0; i < datasets.Length; i++)
            {
                cumulative += datasets[i].Length;
                _cumulativeSizes[i] = cumulative;
            }
        }

        /// <summary>Public API</summary>
        public override int Length => _cumulativeSizes[^1];

        /// <summary>Public API</summary>
        public override object GetItem(int index)
        {
            for (int i = 0; i < _datasets.Length; i++)
            {
                if (index < _cumulativeSizes[i])
                {
                    int prevCumulative = i == 0 ? 0 : _cumulativeSizes[i - 1];
                    return _datasets[i].GetItem(index - prevCumulative);
                }
            }
            throw new IndexOutOfRangeException();
        }
    }

    /// <summary>
    /// Subsets a dataset using indices
    /// </summary>
    public class Subset : Dataset
    {
        private readonly Dataset _dataset;
        private readonly int[] _indices;

        /// <summary>Public API</summary>
        public Subset(Dataset dataset, int[] indices)
        {
            _dataset = dataset;
            _indices = indices;
        }

        /// <summary>Public API</summary>
        public override int Length => _indices.Length;

        /// <summary>Public API</summary>
        public override object GetItem(int index) => _dataset.GetItem(_indices[index]);
    }

    /// <summary>
    /// Applies a transform to a dataset
    /// </summary>
    public class TransformDataset : Dataset
    {
        private readonly Dataset _dataset;
        private readonly Func<object, object> _transform;

        /// <summary>Public API</summary>
        public TransformDataset(Dataset dataset, Func<object, object> transform)
        {
            _dataset = dataset;
            _transform = transform;
        }

        /// <summary>Public API</summary>
        public override int Length => _dataset.Length;

        /// <summary>Public API</summary>
        public override object GetItem(int index) => _transform(_dataset.GetItem(index));
    }

    /// <summary>
    /// Sampler that yields indices for a dataset
    /// </summary>
    public abstract class Sampler : IEnumerable<int>
    {
        protected readonly Dataset _dataset;

        protected Sampler(Dataset dataset)
        {
            _dataset = dataset;
        }

        /// <summary>Public API</summary>
        public abstract int Length { get; }

        /// <summary>Public API</summary>
        public abstract IEnumerator<int> GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    /// <summary>
    /// Sequential sampler - returns indices in order
    /// </summary>
    public class SequentialSampler : Sampler
    {
        /// <summary>Public API</summary>
        public SequentialSampler(Dataset dataset) : base(dataset) { }

        /// <summary>Public API</summary>
        public override int Length => _dataset.Length;

        /// <summary>Public API</summary>
        public override IEnumerator<int> GetEnumerator()
        {
            for (int i = 0; i < _dataset.Length; i++)
                yield return i;
        }
    }

    /// <summary>
    /// Random sampler - returns indices in random order
    /// </summary>
    public class RandomSampler : Sampler
    {
        private readonly bool _replacement;
        private readonly int? _numSamples;
        private readonly Random _random;

        /// <summary>Public API</summary>
        public RandomSampler(Dataset dataset, bool replacement = false, int? numSamples = null, int? seed = null)
            : base(dataset)
        {
            _replacement = replacement;
            _numSamples = numSamples;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>Public API</summary>
        public override int Length => _numSamples ?? _dataset.Length;

        /// <summary>Public API</summary>
        public override IEnumerator<int> GetEnumerator()
        {
            if (_replacement)
            {
                for (int i = 0; i < Length; i++)
                    yield return _random.Next(_dataset.Length);
            }
            else
            {
                var indices = Enumerable.Range(0, _dataset.Length).OrderBy(_ => _random.Next()).ToList();
                foreach (var idx in indices.Take(Length))
                    yield return idx;
            }
        }
    }

    /// <summary>
    /// Weighted random sampler
    /// </summary>
    public class WeightedRandomSampler : Sampler
    {
        private readonly double[] _weights;
        private readonly int _numSamples;
        private readonly bool _replacement;
        private readonly Random _random;
        private readonly double[] _cumulativeWeights;

        /// <summary>Public API</summary>
        public WeightedRandomSampler(Dataset dataset, double[] weights, int numSamples, bool replacement = true, int? seed = null)
            : base(dataset)
        {
            if (weights.Length != dataset.Length)
                throw new ArgumentException("Weights length must match dataset length");

            _weights = weights;
            _numSamples = numSamples;
            _replacement = replacement;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();

            // Normalize and compute cumulative weights
            double sum = weights.Sum();
            _cumulativeWeights = new double[weights.Length];
            double cumulative = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                cumulative += weights[i] / sum;
                _cumulativeWeights[i] = cumulative;
            }
        }

        /// <summary>Public API</summary>
        public override int Length => _numSamples;

        /// <summary>Public API</summary>
        public override IEnumerator<int> GetEnumerator()
        {
            var selected = new HashSet<int>();

            for (int i = 0; i < _numSamples; i++)
            {
                int idx;
                do
                {
                    double r = _random.NextDouble();
                    idx = Array.BinarySearch(_cumulativeWeights, r);
                    if (idx < 0) idx = ~idx;
                    if (idx >= _cumulativeWeights.Length) idx = _cumulativeWeights.Length - 1;
                } while (!_replacement && selected.Contains(idx));

                if (!_replacement) selected.Add(idx);
                yield return idx;
            }
        }
    }

    /// <summary>
    /// Batch sampler - yields batches of indices
    /// </summary>
    public class BatchSampler : IEnumerable<int[]>
    {
        private readonly Sampler _sampler;
        private readonly int _batchSize;
        private readonly bool _dropLast;

        /// <summary>Public API</summary>
        public BatchSampler(Sampler sampler, int batchSize, bool dropLast = false)
        {
            _sampler = sampler;
            _batchSize = batchSize;
            _dropLast = dropLast;
        }

        /// <summary>Public API</summary>
        public int Length
        {
            get
            {
                if (_dropLast)
                    return _sampler.Length / _batchSize;
                else
                    return (_sampler.Length + _batchSize - 1) / _batchSize;
            }
        }

        /// <summary>Public API</summary>
        public IEnumerator<int[]> GetEnumerator()
        {
            var batch = new List<int>();
            foreach (var idx in _sampler)
            {
                batch.Add(idx);
                if (batch.Count == _batchSize)
                {
                    yield return batch.ToArray();
                    batch.Clear();
                }
            }
            if (batch.Count > 0 && !_dropLast)
            {
                yield return batch.ToArray();
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    /// <summary>
    /// Data loader - iterates over a dataset in batches
    /// </summary>
    public class DataLoader : IEnumerable<object>
    {
        private readonly Dataset _dataset;
        private readonly int _batchSize;
        private readonly bool _shuffle;
        private readonly bool _dropLast;
        private readonly int? _seed;
        private readonly Func<object[], object>? _collateFn;
        private readonly int _numWorkers;

        /// <summary>Public API</summary>
        public DataLoader(Dataset dataset, int batchSize = 1, bool shuffle = false, bool dropLast = false,
            int? seed = null, Func<object[], object>? collateFn = null, int numWorkers = 0)
        {
            _dataset = dataset;
            _batchSize = batchSize;
            _shuffle = shuffle;
            _dropLast = dropLast;
            _seed = seed;
            _collateFn = collateFn ?? DefaultCollate;
            _numWorkers = numWorkers;
        }

        /// <summary>Public API</summary>
        public int Length
        {
            get
            {
                if (_dropLast)
                    return _dataset.Length / _batchSize;
                else
                    return (_dataset.Length + _batchSize - 1) / _batchSize;
            }
        }

        /// <summary>Public API</summary>
        public IEnumerator<object> GetEnumerator()
        {
            Sampler sampler = _shuffle
                ? new RandomSampler(_dataset, seed: _seed)
                : new SequentialSampler(_dataset);

            var batchSampler = new BatchSampler(sampler, _batchSize, _dropLast);

            if (_numWorkers > 0)
            {
                // Parallel data loading
                foreach (var batchIndices in batchSampler)
                {
                    var samples = new object[batchIndices.Length];
                    Parallel.For(0, batchIndices.Length, i =>
                    {
                        samples[i] = _dataset.GetItem(batchIndices[i]);
                    });
                    yield return _collateFn!(samples);
                }
            }
            else
            {
                foreach (var batchIndices in batchSampler)
                {
                    var samples = batchIndices.Select(i => _dataset.GetItem(i)).ToArray();
                    yield return _collateFn!(samples);
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Default collate function - stacks tensors or returns arrays
        /// </summary>
        private static object DefaultCollate(object[] batch)
        {
            if (batch.Length == 0)
                return batch;

            var first = batch[0];

            if (first is Tensor)
            {
                // Stack tensors along new first dimension
                return TensorOps.Stack(batch.Cast<Tensor>().ToArray(), 0);
            }
            else if (first is Tensor[])
            {
                // Stack each tensor in the tuple
                var arrays = batch.Cast<Tensor[]>().ToArray();
                var numTensors = arrays[0].Length;
                var stacked = new Tensor[numTensors];

                for (int i = 0; i < numTensors; i++)
                {
                    stacked[i] = TensorOps.Stack(arrays.Select(a => a[i]).ToArray(), 0);
                }
                return stacked;
            }
            else if (first is ValueTuple<Tensor, Tensor>)
            {
                var tuples = batch.Cast<ValueTuple<Tensor, Tensor>>().ToArray();
                return (
                    TensorOps.Stack(tuples.Select(t => t.Item1).ToArray(), 0),
                    TensorOps.Stack(tuples.Select(t => t.Item2).ToArray(), 0)
                );
            }

            return batch;
        }
    }

    /// <summary>
    /// Utility to split a dataset into train/validation/test
    /// </summary>
    public static class DatasetUtils
    {
        /// <summary>
        /// Randomly split a dataset into multiple subsets
        /// </summary>
        public static Subset[] RandomSplit(Dataset dataset, int[] lengths, int? seed = null)
        {
            if (lengths.Sum() != dataset.Length)
                throw new ArgumentException("Sum of lengths must equal dataset length");

            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            var indices = Enumerable.Range(0, dataset.Length).OrderBy(_ => random.Next()).ToArray();

            var subsets = new Subset[lengths.Length];
            int offset = 0;

            for (int i = 0; i < lengths.Length; i++)
            {
                var subsetIndices = indices.Skip(offset).Take(lengths[i]).ToArray();
                subsets[i] = new Subset(dataset, subsetIndices);
                offset += lengths[i];
            }

            return subsets;
        }

        /// <summary>
        /// Split dataset by ratio (e.g., 0.8 for 80% train)
        /// </summary>
        public static (Subset train, Subset val) TrainValSplit(Dataset dataset, double trainRatio, int? seed = null)
        {
            int trainSize = (int)(dataset.Length * trainRatio);
            int valSize = dataset.Length - trainSize;
            var splits = RandomSplit(dataset, new[] { trainSize, valSize }, seed);
            return (splits[0], splits[1]);
        }

        /// <summary>
        /// K-Fold cross validation split
        /// </summary>
        public static IEnumerable<(Subset train, Subset val)> KFold(Dataset dataset, int k, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            var indices = Enumerable.Range(0, dataset.Length).OrderBy(_ => random.Next()).ToArray();
            int foldSize = dataset.Length / k;

            for (int fold = 0; fold < k; fold++)
            {
                int valStart = fold * foldSize;
                int valEnd = fold == k - 1 ? dataset.Length : valStart + foldSize;

                var valIndices = indices.Skip(valStart).Take(valEnd - valStart).ToArray();
                var trainIndices = indices.Take(valStart).Concat(indices.Skip(valEnd)).ToArray();

                yield return (new Subset(dataset, trainIndices), new Subset(dataset, valIndices));
            }
        }
    }

    #region In-Memory Datasets

    /// <summary>
    /// Simple labeled dataset for classification
    /// </summary>
    public class LabeledDataset : Dataset
    {
        private readonly Tensor _data;
        private readonly Tensor _labels;

        /// <summary>Public API</summary>
        public LabeledDataset(Tensor data, Tensor labels)
        {
            if (data.Shape[0] != labels.Shape[0])
                throw new ArgumentException("Data and labels must have same first dimension");

            _data = data;
            _labels = labels;
        }

        /// <summary>Public API</summary>
        public override int Length => (int)_data.Shape[0];

        /// <summary>Public API</summary>
        public override object GetItem(int index)
        {
            return (
                _data.Slice(0, index, index + 1).Squeeze(0),
                _labels.Slice(0, index, index + 1).Squeeze(0)
            );
        }
    }

    /// <summary>
    /// Dataset that generates data on-the-fly using a function
    /// </summary>
    public class FunctionalDataset : Dataset
    {
        private readonly int _length;
        private readonly Func<int, object> _generator;

        /// <summary>Public API</summary>
        public FunctionalDataset(int length, Func<int, object> generator)
        {
            _length = length;
            _generator = generator;
        }

        /// <summary>Public API</summary>
        public override int Length => _length;

        /// <summary>Public API</summary>
        public override object GetItem(int index) => _generator(index);
    }

    /// <summary>
    /// Caches dataset items in memory after first access
    /// </summary>
    public class CachedDataset : Dataset
    {
        private readonly Dataset _dataset;
        private readonly object?[] _cache;
        private readonly object _lock = new object();

        /// <summary>Public API</summary>
        public CachedDataset(Dataset dataset)
        {
            _dataset = dataset;
            _cache = new object?[dataset.Length];
        }

        /// <summary>Public API</summary>
        public override int Length => _dataset.Length;

        /// <summary>Public API</summary>
        public override object GetItem(int index)
        {
            if (_cache[index] == null)
            {
                lock (_lock)
                {
                    if (_cache[index] == null)
                        _cache[index] = _dataset.GetItem(index);
                }
            }
            return _cache[index]!;
        }

        /// <summary>Public API</summary>
        public void ClearCache()
        {
            Array.Clear(_cache, 0, _cache.Length);
        }
    }

    #endregion

    #region Transforms

    /// <summary>
    /// Common data transforms
    /// </summary>
    public static class Transforms
    {
        /// <summary>
        /// Normalize a tensor with mean and std
        /// </summary>
        public static Func<Tensor, Tensor> Normalize(double[] mean, double[] std)
        {
            return tensor =>
            {
                var result = tensor.Clone();
                var numChannels = mean.Length;

                if (tensor.NDim == 3) // [C, H, W]
                {
                    for (int c = 0; c < numChannels; c++)
                    {
                        var channelData = result.Slice(0, c, c + 1);
                        var normalized = channelData.Sub(mean[c]).Div(std[c]);
                        // Copy back
                        var nData = normalized.ToArray();
                        var rData = result.ToArray();
                        var channelSize = (int)result.Shape[1] * (int)result.Shape[2];
                        Array.Copy(nData, 0, rData, c * channelSize, channelSize);
                        result = Tensor.FromArray(rData, result.Shape);
                    }
                }
                return result;
            };
        }

        /// <summary>
        /// Convert tensor values to range [0, 1]
        /// </summary>
        public static Func<Tensor, Tensor> ToTensor()
        {
            return tensor => tensor.Div(255.0);
        }

        /// <summary>
        /// Random horizontal flip
        /// </summary>
        public static Func<Tensor, Tensor> RandomHorizontalFlip(double p = 0.5)
        {
            var random = new Random();
            return tensor =>
            {
                if (random.NextDouble() < p)
                {
                    // Flip along width dimension (last dimension for CHW format)
                    return tensor.Flip(-1);
                }
                return tensor;
            };
        }

        /// <summary>
        /// Random vertical flip
        /// </summary>
        public static Func<Tensor, Tensor> RandomVerticalFlip(double p = 0.5)
        {
            var random = new Random();
            return tensor =>
            {
                if (random.NextDouble() < p)
                {
                    // Flip along height dimension
                    return tensor.Flip(-2);
                }
                return tensor;
            };
        }

        /// <summary>
        /// Random crop
        /// </summary>
        public static Func<Tensor, Tensor> RandomCrop(int height, int width, int padding = 0)
        {
            var random = new Random();
            return tensor =>
            {
                // Assume CHW format
                var h = (int)tensor.Shape[^2];
                var w = (int)tensor.Shape[^1];

                if (padding > 0)
                {
                    // Pad the tensor
                    tensor = F.Pad(tensor.Unsqueeze(0), new[] { padding, padding, padding, padding }).Squeeze(0);
                    h += 2 * padding;
                    w += 2 * padding;
                }

                var top = random.Next(0, h - height + 1);
                var left = random.Next(0, w - width + 1);

                // Slice out the crop
                return tensor.Slice(-2, top, top + height).Slice(-1, left, left + width);
            };
        }

        /// <summary>
        /// Center crop
        /// </summary>
        public static Func<Tensor, Tensor> CenterCrop(int height, int width)
        {
            return tensor =>
            {
                var h = (int)tensor.Shape[^2];
                var w = (int)tensor.Shape[^1];

                var top = (h - height) / 2;
                var left = (w - width) / 2;

                return tensor.Slice(-2, top, top + height).Slice(-1, left, left + width);
            };
        }

        /// <summary>
        /// Resize tensor (nearest neighbor interpolation)
        /// </summary>
        public static Func<Tensor, Tensor> Resize(int height, int width)
        {
            return tensor =>
            {
                return F.Interpolate(tensor.Unsqueeze(0), new[] { height, width }, "bilinear").Squeeze(0);
            };
        }

        /// <summary>
        /// Compose multiple transforms
        /// </summary>
        public static Func<Tensor, Tensor> Compose(params Func<Tensor, Tensor>[] transforms)
        {
            return tensor =>
            {
                foreach (var transform in transforms)
                {
                    tensor = transform(tensor);
                }
                return tensor;
            };
        }

        /// <summary>
        /// Add Gaussian noise
        /// </summary>
        public static Func<Tensor, Tensor> GaussianNoise(double mean = 0, double std = 0.1)
        {
            return tensor =>
            {
                var noise = Tensor.Randn(tensor.Shape).Mul(std).Add(mean);
                return tensor.Add(noise);
            };
        }

        /// <summary>
        /// Random rotation (simple 90-degree rotations)
        /// </summary>
        public static Func<Tensor, Tensor> RandomRotation90()
        {
            var random = new Random();
            return tensor =>
            {
                int rotations = random.Next(4);
                for (int i = 0; i < rotations; i++)
                {
                    // Rotate 90 degrees: transpose H,W then flip W
                    tensor = tensor.Transpose(-2, -1).Flip(-1);
                }
                return tensor;
            };
        }

        /// <summary>
        /// Color jitter (brightness, contrast, saturation)
        /// </summary>
        public static Func<Tensor, Tensor> ColorJitter(double brightness = 0, double contrast = 0, double saturation = 0)
        {
            var random = new Random();
            return tensor =>
            {
                if (brightness > 0)
                {
                    double factor = 1 + (random.NextDouble() * 2 - 1) * brightness;
                    tensor = tensor.Mul(factor).Clamp(0, 1);
                }

                if (contrast > 0)
                {
                    double factor = 1 + (random.NextDouble() * 2 - 1) * contrast;
                    double mean = tensor.Mean().ToScalar();
                    tensor = tensor.Sub(mean).Mul(factor).Add(mean).Clamp(0, 1);
                }

                // Saturation would require converting to HSV which is complex

                return tensor;
            };
        }

        /// <summary>
        /// Random erasing (cutout)
        /// </summary>
        public static Func<Tensor, Tensor> RandomErasing(double p = 0.5, double scaleMin = 0.02, double scaleMax = 0.33, double ratioMin = 0.3, double ratioMax = 3.3)
        {
            var random = new Random();
            return tensor =>
            {
                if (random.NextDouble() > p)
                    return tensor;

                var h = (int)tensor.Shape[^2];
                var w = (int)tensor.Shape[^1];
                var area = h * w;

                for (int attempt = 0; attempt < 10; attempt++)
                {
                    var targetArea = area * (scaleMin + random.NextDouble() * (scaleMax - scaleMin));
                    var aspectRatio = Math.Exp(random.NextDouble() * Math.Log(ratioMax / ratioMin) + Math.Log(ratioMin));

                    var eH = (int)Math.Sqrt(targetArea / aspectRatio);
                    var eW = (int)Math.Sqrt(targetArea * aspectRatio);

                    if (eH < h && eW < w)
                    {
                        var top = random.Next(0, h - eH);
                        var left = random.Next(0, w - eW);

                        var result = tensor.Clone();
                        // Set the erased region to random values or 0
                        for (int c = 0; c < tensor.Shape[0]; c++)
                        {
                            for (int i = top; i < top + eH; i++)
                            {
                                for (int j = left; j < left + eW; j++)
                                {
                                    result[c, i, j] = random.NextDouble();
                                }
                            }
                        }
                        return result;
                    }
                }

                return tensor;
            };
        }
    }

    #endregion
}