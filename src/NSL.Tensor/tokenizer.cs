using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NSL.Tensor
{
    /// Interface for all tokenizers.
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>Encodes text into token IDs</summary>
        int[] Encode(string text);
        /// <summary>Decodes token IDs back to text</summary>
        string Decode(int[] tokens);
        /// <summary>Size of the vocabulary</summary>
        int VocabSize { get; }
        /// <summary>Padding token ID</summary>
        int PadTokenId { get; }
        /// <summary>Unknown token ID</summary>
        int UnkTokenId { get; }
        /// <summary>Beginning of sequence token ID</summary>
        int BosTokenId { get; }
        /// <summary>End of sequence token ID</summary>
        int EosTokenId { get; }
    }

    /// Byte-Pair Encoding (BPE) tokenizer implementation.
    /// Commonly used in GPT, RoBERTa, and other transformer models.
    /// </summary>
    public class BPETokenizer : ITokenizer
    {
        /// <summary>API member</summary>
        private readonly Dictionary<string, int> _vocab;
        /// <summary>API member</summary>
        private readonly Dictionary<int, string> _reverseVocab;
        /// <summary>API member</summary>
        private readonly Dictionary<(string, string), int> _merges;
        /// <summary>API member</summary>
        private readonly List<(string, string)> _mergeOrder;
        /// <summary>API member</summary>
        private readonly Dictionary<string, string> _cache;
        /// <summary>API member</summary>
        private readonly Regex _pattern;

        // Special tokens
        /// <summary>Public API</summary>
        public int PadTokenId { get; private set; } = 0;
        /// <summary>Public API</summary>
        public int UnkTokenId { get; private set; } = 1;
        /// <summary>Public API</summary>
        public int BosTokenId { get; private set; } = 2;
        /// <summary>Public API</summary>
        public int EosTokenId { get; private set; } = 3;

        /// <summary>Public API</summary>
        public int VocabSize => _vocab.Count;

        // Default GPT-2 style pattern
        private static readonly Regex DefaultPattern = new Regex(
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled);

        /// Creates a new BPE tokenizer with an empty vocabulary.
        /// </summary>
        public BPETokenizer()
        {
            _vocab = new Dictionary<string, int>();
            _reverseVocab = new Dictionary<int, string>();
            _merges = new Dictionary<(string, string), int>();
            _mergeOrder = new List<(string, string)>();
            _cache = new Dictionary<string, string>();
            _pattern = DefaultPattern;

            InitializeSpecialTokens();
        }

        /// Creates a BPE tokenizer from vocabulary and merges files.
        /// </summary>
        public BPETokenizer(string vocabPath, string mergesPath)
            : this()
        {
            LoadVocab(vocabPath);
            LoadMerges(mergesPath);
        }

        private void InitializeSpecialTokens()
        {
            AddToken("<pad>", 0);
            AddToken("<unk>", 1);
            AddToken("<s>", 2);    // BOS
            AddToken("</s>", 3);   // EOS

            PadTokenId = 0;
            UnkTokenId = 1;
            BosTokenId = 2;
            EosTokenId = 3;
        }

        private void AddToken(string token, int id)
        {
            if (!_vocab.ContainsKey(token))
            {
                _vocab[token] = id;
                _reverseVocab[id] = token;
            }
        }

        /// Trains the BPE tokenizer on a corpus of text.
        /// </summary>
        /// <param name="texts">Collection of text documents to train on</param>
        /// <param name="vocabSize">Target vocabulary size</param>
        /// <param name="minFrequency">Minimum frequency for a token to be included</param>
        public void Train(IEnumerable<string> texts, int vocabSize = 32000, int minFrequency = 2)
        {
            Console.WriteLine("Training BPE tokenizer...");

            // Step 1: Count character frequencies and initialize vocabulary with base characters
            var wordFreq = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                var matches = _pattern.Matches(text);
                foreach (Match match in matches)
                {
                    var word = match.Value;
                    if (!string.IsNullOrEmpty(word))
                    {
                        wordFreq[word] = wordFreq.GetValueOrDefault(word, 0) + 1;
                    }
                }
            }

            // Filter by minimum frequency
            wordFreq = wordFreq.Where(kv => kv.Value >= minFrequency)
                               .ToDictionary(kv => kv.Key, kv => kv.Value);

            // Initialize with character-level vocabulary
            var charSet = new HashSet<char>();
            foreach (var word in wordFreq.Keys)
            {
                foreach (var c in word)
                {
                    charSet.Add(c);
                }
            }

            // Add base characters to vocabulary (after special tokens)
            int nextId = 4; // After special tokens
            foreach (var c in charSet.OrderBy(c => c))
            {
                AddToken(c.ToString(), nextId++);
            }

            // Convert words to character sequences with end-of-word marker
            var wordTokens = new Dictionary<string, List<string>>();
            foreach (var word in wordFreq.Keys)
            {
                var chars = word.Select(c => c.ToString()).ToList();
                chars[^1] = chars[^1] + "</w>"; // Add end-of-word marker
                wordTokens[word] = chars;
            }

            Console.WriteLine($"Initial vocab size: {_vocab.Count}");

            // Step 2: Iteratively merge most frequent pairs
            while (_vocab.Count < vocabSize)
            {
                // Count pair frequencies
                var pairCounts = new Dictionary<(string, string), int>();

                foreach (var (word, freq) in wordFreq)
                {
                    var tokens = wordTokens[word];
                    for (int i = 0; i < tokens.Count - 1; i++)
                    {
                        var pair = (tokens[i], tokens[i + 1]);
                        pairCounts[pair] = pairCounts.GetValueOrDefault(pair, 0) + freq;
                    }
                }

                if (pairCounts.Count == 0)
                    break;

                // Find most frequent pair
                var bestPair = pairCounts.OrderByDescending(kv => kv.Value).First().Key;
                var newToken = bestPair.Item1 + bestPair.Item2;

                // Add to vocabulary
                AddToken(newToken, nextId++);
                _merges[bestPair] = _mergeOrder.Count;
                _mergeOrder.Add(bestPair);

                // Merge in all words
                foreach (var word in wordTokens.Keys.ToList())
                {
                    var tokens = wordTokens[word];
                    var newTokens = new List<string>();

                    int i = 0;
                    while (i < tokens.Count)
                    {
                        if (i < tokens.Count - 1 && tokens[i] == bestPair.Item1 && tokens[i + 1] == bestPair.Item2)
                        {
                            newTokens.Add(newToken);
                            i += 2;
                        }
                        else
                        {
                            newTokens.Add(tokens[i]);
                            i++;
                        }
                    }

                    wordTokens[word] = newTokens;
                }

                if (_vocab.Count % 1000 == 0)
                {
                    Console.WriteLine($"Vocab size: {_vocab.Count}");
                }
            }

            Console.WriteLine($"Final vocab size: {_vocab.Count}");
        }

        /// Encodes text into token IDs.
        /// </summary>
        public int[] Encode(string text)
        {
            var tokens = new List<int>();
            var matches = _pattern.Matches(text);

            foreach (Match match in matches)
            {
                var word = match.Value;
                if (string.IsNullOrEmpty(word))
                    continue;

                var bpeTokens = BPE(word);
                foreach (var token in bpeTokens)
                {
                    if (_vocab.TryGetValue(token, out int id))
                    {
                        tokens.Add(id);
                    }
                    else
                    {
                        tokens.Add(UnkTokenId);
                    }
                }
            }

            return tokens.ToArray();
        }

        /// Encodes text with special tokens (BOS and EOS).
        /// </summary>
        public int[] EncodeWithSpecialTokens(string text, bool addBos = true, bool addEos = true)
        {
            var tokens = new List<int>();

            if (addBos)
                tokens.Add(BosTokenId);

            tokens.AddRange(Encode(text));

            if (addEos)
                tokens.Add(EosTokenId);

            return tokens.ToArray();
        }

        /// Decodes token IDs back to text.
        /// </summary>
        public string Decode(int[] tokens)
        {
            var sb = new StringBuilder();

            foreach (var tokenId in tokens)
            {
                // Skip special tokens
                if (tokenId == PadTokenId || tokenId == BosTokenId || tokenId == EosTokenId)
                    continue;

                if (_reverseVocab.TryGetValue(tokenId, out string? token))
                {
                    // Remove end-of-word marker
                    token = token.Replace("</w>", "");
                    sb.Append(token);
                }
                else
                {
                    sb.Append("<unk>");
                }
            }

            return sb.ToString();
        }

        /// Apply BPE to a word.
        /// </summary>
        private List<string> BPE(string word)
        {
            // Check cache
            if (_cache.TryGetValue(word, out string? cached))
            {
                return cached.Split(' ').ToList();
            }

            // Initialize with characters
            var tokens = word.Select(c => c.ToString()).ToList();
            if (tokens.Count > 0)
            {
                tokens[^1] = tokens[^1] + "</w>";
            }

            while (tokens.Count > 1)
            {
                // Find the highest priority merge
                int bestIdx = -1;
                int bestPriority = int.MaxValue;

                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    if (_merges.TryGetValue(pair, out int priority) && priority < bestPriority)
                    {
                        bestPriority = priority;
                        bestIdx = i;
                    }
                }

                if (bestIdx == -1)
                    break;

                // Apply merge
                var newToken = tokens[bestIdx] + tokens[bestIdx + 1];
                tokens[bestIdx] = newToken;
                tokens.RemoveAt(bestIdx + 1);
            }

            // Cache result
            _cache[word] = string.Join(" ", tokens);

            return tokens;
        }

        /// Saves the tokenizer vocabulary and merges to files.
        /// </summary>
        public void Save(string directory)
        {
            Directory.CreateDirectory(directory);

            // Save vocabulary
            var vocabPath = Path.Combine(directory, "vocab.json");
            var vocabJson = JsonSerializer.Serialize(_vocab, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(vocabPath, vocabJson);

            // Save merges
            var mergesPath = Path.Combine(directory, "merges.txt");
            using var writer = new StreamWriter(mergesPath);
            writer.WriteLine("#version: 0.2");
            foreach (var merge in _mergeOrder)
            {
                writer.WriteLine($"{merge.Item1} {merge.Item2}");
            }

            // Save special tokens config
            var configPath = Path.Combine(directory, "tokenizer_config.json");
            var config = new Dictionary<string, object>
            {
                ["pad_token"] = "<pad>",
                ["unk_token"] = "<unk>",
                ["bos_token"] = "<s>",
                ["eos_token"] = "</s>",
                ["vocab_size"] = VocabSize
            };
            var configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(configPath, configJson);
        }

        /// Loads vocabulary from a JSON file.
        /// </summary>
        public void LoadVocab(string path)
        {
            var json = File.ReadAllText(path);
            var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(json);

            if (vocab != null)
            {
                foreach (var (token, id) in vocab)
                {
                    _vocab[token] = id;
                    _reverseVocab[id] = token;
                }
            }
        }

        /// Loads merges from a text file.
        /// </summary>
        public void LoadMerges(string path)
        {
            var lines = File.ReadAllLines(path);
            int priority = 0;

            foreach (var line in lines)
            {
                if (line.StartsWith("#"))
                    continue;

                var parts = line.Split(' ');
                if (parts.Length == 2)
                {
                    var merge = (parts[0], parts[1]);
                    _merges[merge] = priority;
                    _mergeOrder.Add(merge);
                    priority++;
                }
            }
        }

        /// Gets the token string for a token ID.
        /// </summary>
        public string? GetToken(int id)
        {
            return _reverseVocab.GetValueOrDefault(id);
        }

        /// Gets the token ID for a token string.
        /// </summary>
        public int GetTokenId(string token)
        {
            return _vocab.GetValueOrDefault(token, UnkTokenId);
        }
    }

    /// WordPiece tokenizer implementation (used in BERT).
    /// </summary>
    public class WordPieceTokenizer : ITokenizer
    {
        /// <summary>API member</summary>
        private readonly Dictionary<string, int> _vocab;
        /// <summary>API member</summary>
        private readonly Dictionary<int, string> _reverseVocab;
        /// <summary>API member</summary>
        private readonly int _maxInputCharsPerWord;
        /// <summary>API member</summary>
        private readonly string _unkToken;
        /// <summary>API member</summary>
        private readonly string _continuingSubwordPrefix;

        /// <summary>Public API</summary>
        public int PadTokenId { get; private set; } = 0;
        /// <summary>Public API</summary>
        public int UnkTokenId { get; private set; } = 100;
        /// <summary>Public API</summary>
        public int BosTokenId { get; private set; } = 101;  // [CLS]
        /// <summary>Public API</summary>
        public int EosTokenId { get; private set; } = 102;  // [SEP]

        /// <summary>Public API</summary>
        public int VocabSize => _vocab.Count;

        /// <summary>Public API</summary>
        public WordPieceTokenizer(
            string vocabPath,
            string unkToken = "[UNK]",
            int maxInputCharsPerWord = 100,
            string continuingSubwordPrefix = "##")
        {
            _vocab = new Dictionary<string, int>();
            _reverseVocab = new Dictionary<int, string>();
            _unkToken = unkToken;
            _maxInputCharsPerWord = maxInputCharsPerWord;
            _continuingSubwordPrefix = continuingSubwordPrefix;

            LoadVocab(vocabPath);
        }

        /// <summary>Public API</summary>
        public WordPieceTokenizer()
        {
            _vocab = new Dictionary<string, int>();
            _reverseVocab = new Dictionary<int, string>();
            _unkToken = "[UNK]";
            _maxInputCharsPerWord = 100;
            _continuingSubwordPrefix = "##";

            InitializeSpecialTokens();
        }

        private void InitializeSpecialTokens()
        {
            AddToken("[PAD]", 0);
            AddToken("[UNK]", 100);
            AddToken("[CLS]", 101);
            AddToken("[SEP]", 102);
            AddToken("[MASK]", 103);

            PadTokenId = 0;
            UnkTokenId = 100;
            BosTokenId = 101;
            EosTokenId = 102;
        }

        private void AddToken(string token, int id)
        {
            _vocab[token] = id;
            _reverseVocab[id] = token;
        }

        /// <summary>Public API</summary>
        public void LoadVocab(string path)
        {
            var lines = File.ReadAllLines(path);
            for (int i = 0; i < lines.Length; i++)
            {
                var token = lines[i].Trim();
                if (!string.IsNullOrEmpty(token))
                {
                    _vocab[token] = i;
                    _reverseVocab[i] = token;
                }
            }

            // Update special token IDs based on loaded vocab
            if (_vocab.TryGetValue("[PAD]", out int padId)) PadTokenId = padId;
            if (_vocab.TryGetValue("[UNK]", out int unkId)) UnkTokenId = unkId;
            if (_vocab.TryGetValue("[CLS]", out int clsId)) BosTokenId = clsId;
            if (_vocab.TryGetValue("[SEP]", out int sepId)) EosTokenId = sepId;
        }

        /// <summary>Public API</summary>
        public int[] Encode(string text)
        {
            var tokens = new List<int>();

            // Basic whitespace tokenization
            var words = text.ToLower().Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

            foreach (var word in words)
            {
                var subTokens = TokenizeWord(word);
                foreach (var subToken in subTokens)
                {
                    tokens.Add(_vocab.GetValueOrDefault(subToken, UnkTokenId));
                }
            }

            return tokens.ToArray();
        }

        private List<string> TokenizeWord(string word)
        {
            if (word.Length > _maxInputCharsPerWord)
            {
                return new List<string> { _unkToken };
            }

            var tokens = new List<string>();
            int start = 0;

            while (start < word.Length)
            {
                int end = word.Length;
                string? curSubstr = null;

                while (start < end)
                {
                    var substr = word.Substring(start, end - start);
                    if (start > 0)
                    {
                        substr = _continuingSubwordPrefix + substr;
                    }

                    if (_vocab.ContainsKey(substr))
                    {
                        curSubstr = substr;
                        break;
                    }
                    end--;
                }

                if (curSubstr == null)
                {
                    return new List<string> { _unkToken };
                }

                tokens.Add(curSubstr);
                start = end;
            }

            return tokens;
        }

        /// <summary>Public API</summary>
        public string Decode(int[] tokens)
        {
            var sb = new StringBuilder();

            foreach (var tokenId in tokens)
            {
                if (tokenId == PadTokenId || tokenId == BosTokenId || tokenId == EosTokenId)
                    continue;

                if (_reverseVocab.TryGetValue(tokenId, out string? token))
                {
                    if (token.StartsWith(_continuingSubwordPrefix))
                    {
                        sb.Append(token.Substring(_continuingSubwordPrefix.Length));
                    }
                    else
                    {
                        if (sb.Length > 0) sb.Append(' ');
                        sb.Append(token);
                    }
                }
            }

            return sb.ToString();
        }
    }

    /// Simple character-level tokenizer for basic use cases.
    /// </summary>
    public class CharTokenizer : ITokenizer
    {
        /// <summary>API member</summary>
        private readonly Dictionary<char, int> _charToId;
        /// <summary>API member</summary>
        private readonly Dictionary<int, char> _idToChar;

        /// <summary>Public API</summary>
        public int PadTokenId { get; } = 0;
        /// <summary>Public API</summary>
        public int UnkTokenId { get; } = 1;
        /// <summary>Public API</summary>
        public int BosTokenId { get; } = 2;
        /// <summary>Public API</summary>
        public int EosTokenId { get; } = 3;

        /// <summary>Public API</summary>
        public int VocabSize => _charToId.Count + 4; // +4 for special tokens

        /// <summary>Public API</summary>
        public CharTokenizer(string? charset = null)
        {
            _charToId = new Dictionary<char, int>();
            _idToChar = new Dictionary<int, char>();

            // Default charset: printable ASCII
            charset ??= " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n\t";

            int id = 4; // Start after special tokens
            foreach (var c in charset)
            {
                if (!_charToId.ContainsKey(c))
                {
                    _charToId[c] = id;
                    _idToChar[id] = c;
                    id++;
                }
            }
        }

        /// <summary>Public API</summary>
        public int[] Encode(string text)
        {
            var tokens = new List<int>();
            foreach (var c in text)
            {
                tokens.Add(_charToId.GetValueOrDefault(c, UnkTokenId));
            }
            return tokens.ToArray();
        }

        /// <summary>Public API</summary>
        public string Decode(int[] tokens)
        {
            var sb = new StringBuilder();
            foreach (var tokenId in tokens)
            {
                if (tokenId == PadTokenId || tokenId == BosTokenId || tokenId == EosTokenId)
                    continue;

                if (_idToChar.TryGetValue(tokenId, out char c))
                {
                    sb.Append(c);
                }
            }
            return sb.ToString();
        }
    }

    /// Utility class for tokenizer operations.
    /// </summary>
    public static class TokenizerUtils
    {
        /// Pads a batch of token sequences to the same length.
        /// </summary>
        public static int[][] PadSequences(int[][] sequences, int? maxLength = null, int padValue = 0, bool padLeft = false)
        {
            var maxLen = maxLength ?? sequences.Max(s => s.Length);
            var result = new int[sequences.Length][];

            for (int i = 0; i < sequences.Length; i++)
            {
                var seq = sequences[i];
                var padded = new int[maxLen];
                Array.Fill(padded, padValue);

                if (seq.Length > maxLen)
                {
                    // Truncate
                    Array.Copy(seq, 0, padded, 0, maxLen);
                }
                else if (padLeft)
                {
                    // Pad left
                    Array.Copy(seq, 0, padded, maxLen - seq.Length, seq.Length);
                }
                else
                {
                    // Pad right
                    Array.Copy(seq, 0, padded, 0, seq.Length);
                }

                result[i] = padded;
            }

            return result;
        }

        /// Creates an attention mask for padded sequences.
        /// </summary>
        public static int[][] CreateAttentionMask(int[][] sequences, int padValue = 0)
        {
            var masks = new int[sequences.Length][];

            for (int i = 0; i < sequences.Length; i++)
            {
                masks[i] = sequences[i].Select(t => t != padValue ? 1 : 0).ToArray();
            }

            return masks;
        }

        /// Truncates sequences to a maximum length.
        /// </summary>
        public static int[][] TruncateSequences(int[][] sequences, int maxLength, bool truncateLeft = false)
        {
            var result = new int[sequences.Length][];

            for (int i = 0; i < sequences.Length; i++)
            {
                var seq = sequences[i];
                if (seq.Length <= maxLength)
                {
                    result[i] = seq;
                }
                else if (truncateLeft)
                {
                    result[i] = seq.Skip(seq.Length - maxLength).ToArray();
                }
                else
                {
                    result[i] = seq.Take(maxLength).ToArray();
                }
            }

            return result;
        }
    }

    /// SentencePiece Unigram tokenizer implementation.
    /// Uses the Unigram language model for subword segmentation.
    /// Based on "Subword Regularization" by Kudo (2018).
    /// </summary>
    public class SentencePieceTokenizer : ITokenizer
    {
        /// <summary>API member</summary>
        private readonly Dictionary<string, int> _vocab;
        /// <summary>API member</summary>
        private readonly Dictionary<int, string> _reverseVocab;
        private readonly Dictionary<string, double> _scores;  // Log probabilities
        /// <summary>API member</summary>
        private readonly int _maxPieceLength;
        /// <summary>API member</summary>
        private readonly string _unkToken;

        /// <summary>Public API</summary>
        public int PadTokenId { get; private set; } = 0;
        /// <summary>Public API</summary>
        public int UnkTokenId { get; private set; } = 1;
        /// <summary>Public API</summary>
        public int BosTokenId { get; private set; } = 2;
        /// <summary>Public API</summary>
        public int EosTokenId { get; private set; } = 3;

        /// <summary>Public API</summary>
        public int VocabSize => _vocab.Count;

        // Unicode replacement character for unknown bytes
        /// <summary>API member</summary>
        private const char REPLACEMENT_CHAR = '\uFFFD';
        // SentencePiece uses ▁ (U+2581) as word boundary marker
        /// <summary>API member</summary>
        private const string WORD_BOUNDARY = "\u2581";

        /// <summary>Public API</summary>
        public SentencePieceTokenizer()
        {
            _vocab = new Dictionary<string, int>();
            _reverseVocab = new Dictionary<int, string>();
            _scores = new Dictionary<string, double>();
            _maxPieceLength = 16;
            _unkToken = "<unk>";

            InitializeSpecialTokens();
        }

        /// <summary>Public API</summary>
        public SentencePieceTokenizer(string modelPath) : this()
        {
            LoadModel(modelPath);
        }

        private void InitializeSpecialTokens()
        {
            AddToken("<pad>", 0, 0.0);
            AddToken("<unk>", 1, 0.0);
            AddToken("<s>", 2, 0.0);
            AddToken("</s>", 3, 0.0);
        }

        private void AddToken(string token, int id, double score)
        {
            _vocab[token] = id;
            _reverseVocab[id] = token;
            _scores[token] = score;
        }

        /// Trains the Unigram model on a corpus.
        /// </summary>
        public void Train(IEnumerable<string> texts, int vocabSize = 32000, int numIterations = 10)
        {
            Console.WriteLine("Training SentencePiece Unigram tokenizer...");

            // Step 1: Build initial seed vocabulary from character and substring frequencies
            var substringFreq = new Dictionary<string, long>();
            long totalChars = 0;

            foreach (var text in texts)
            {
                // Add word boundary markers
                var processed = WORD_BOUNDARY + text.Replace(" ", WORD_BOUNDARY);

                for (int i = 0; i < processed.Length; i++)
                {
                    for (int len = 1; len <= Math.Min(_maxPieceLength, processed.Length - i); len++)
                    {
                        var substr = processed.Substring(i, len);
                        substringFreq[substr] = substringFreq.GetValueOrDefault(substr, 0L) + 1;
                    }
                    totalChars++;
                }
            }

            // Initialize vocabulary with high-frequency substrings
            int seedVocabSize = vocabSize * 2;  // Start with larger vocab
            var sortedSubstrings = substringFreq
                .OrderByDescending(kv => kv.Value * kv.Key.Length)  // Prefer longer, frequent pieces
                .Take(seedVocabSize)
                .ToList();

            int nextId = 4;  // After special tokens
            foreach (var (substr, freq) in sortedSubstrings)
            {
                double score = Math.Log((double)freq / totalChars);
                AddToken(substr, nextId++, score);
            }

            Console.WriteLine($"Initial vocab size: {_vocab.Count}");

            // Step 2: EM-style training to prune vocabulary
            for (int iter = 0; iter < numIterations && _vocab.Count > vocabSize; iter++)
            {
                // E-step: Compute expected counts using Viterbi segmentation
                var expectedCounts = new Dictionary<string, double>();

                foreach (var text in texts.Take(10000))  // Sample for efficiency
                {
                    var processed = WORD_BOUNDARY + text.Replace(" ", WORD_BOUNDARY);
                    var segmentation = ViterbiSegment(processed);

                    foreach (var piece in segmentation)
                    {
                        expectedCounts[piece] = expectedCounts.GetValueOrDefault(piece, 0.0) + 1.0;
                    }
                }

                // M-step: Update scores and prune low-scoring pieces
                double totalCount = expectedCounts.Values.Sum();

                var piecesToRemove = new List<string>();
                foreach (var (piece, count) in expectedCounts)
                {
                    double newScore = Math.Log(count / totalCount);
                    if (_scores.ContainsKey(piece))
                    {
                        _scores[piece] = newScore;
                    }
                }

                // Prune vocabulary
                int targetRemoval = Math.Min((_vocab.Count - vocabSize) / (numIterations - iter), _vocab.Count / 10);

                var pruneCandidates = _vocab.Keys
                    .Where(k => !k.StartsWith("<"))  // Don't prune special tokens
                    .Select(k => (k, _scores.GetValueOrDefault(k, double.NegativeInfinity)))
                    .OrderBy(kv => kv.Item2)
                    .Take(targetRemoval)
                    .ToList();

                foreach (var (piece, _) in pruneCandidates)
                {
                    if (_vocab.TryGetValue(piece, out int id))
                    {
                        _vocab.Remove(piece);
                        _reverseVocab.Remove(id);
                        _scores.Remove(piece);
                    }
                }

                Console.WriteLine($"Iteration {iter + 1}: vocab size = {_vocab.Count}");
            }

            Console.WriteLine($"Final vocab size: {_vocab.Count}");
        }

        /// Viterbi algorithm for finding optimal segmentation.
        /// </summary>
        private List<string> ViterbiSegment(string text)
        {
            int n = text.Length;
            var bestScore = new double[n + 1];
            var bestPrev = new int[n + 1];

            Array.Fill(bestScore, double.NegativeInfinity);
            bestScore[0] = 0;

            // Forward pass
            for (int i = 0; i < n; i++)
            {
                if (double.IsNegativeInfinity(bestScore[i]))
                    continue;

                for (int len = 1; len <= Math.Min(_maxPieceLength, n - i); len++)
                {
                    var piece = text.Substring(i, len);
                    if (_scores.TryGetValue(piece, out double score))
                    {
                        double newScore = bestScore[i] + score;
                        if (newScore > bestScore[i + len])
                        {
                            bestScore[i + len] = newScore;
                            bestPrev[i + len] = i;
                        }
                    }
                }

                // Fallback: treat as unknown character
                if (i + 1 <= n && double.IsNegativeInfinity(bestScore[i + 1]))
                {
                    bestScore[i + 1] = bestScore[i] + _scores.GetValueOrDefault(_unkToken, -10.0);
                    bestPrev[i + 1] = i;
                }
            }

            // Backward pass to recover segmentation
            var result = new List<string>();
            int pos = n;

            while (pos > 0)
            {
                int prevPos = bestPrev[pos];
                var piece = text.Substring(prevPos, pos - prevPos);

                if (_vocab.ContainsKey(piece))
                {
                    result.Add(piece);
                }
                else
                {
                    result.Add(_unkToken);
                }

                pos = prevPos;
            }

            result.Reverse();
            return result;
        }

        /// <summary>Public API</summary>
        public int[] Encode(string text)
        {
            // Add word boundary markers
            var processed = WORD_BOUNDARY + text.Replace(" ", WORD_BOUNDARY);

            var pieces = ViterbiSegment(processed);
            var tokens = new List<int>();

            foreach (var piece in pieces)
            {
                tokens.Add(_vocab.GetValueOrDefault(piece, UnkTokenId));
            }

            return tokens.ToArray();
        }

        /// <summary>Public API</summary>
        public string Decode(int[] tokens)
        {
            var sb = new StringBuilder();

            foreach (var tokenId in tokens)
            {
                if (tokenId == PadTokenId || tokenId == BosTokenId || tokenId == EosTokenId)
                    continue;

                if (_reverseVocab.TryGetValue(tokenId, out string? piece))
                {
                    sb.Append(piece);
                }
            }

            // Remove word boundary markers and convert back to spaces
            return sb.ToString()
                .Replace(WORD_BOUNDARY, " ")
                .Trim();
        }

        /// Samples multiple segmentations for subword regularization.
        /// </summary>
        public List<int[]> SampleSegmentations(string text, int numSamples = 5, double temperature = 1.0)
        {
            var results = new List<int[]>();
            var processed = WORD_BOUNDARY + text.Replace(" ", WORD_BOUNDARY);
            var random = new Random();

            for (int s = 0; s < numSamples; s++)
            {
                var tokens = new List<int>();
                int pos = 0;

                while (pos < processed.Length)
                {
                    // Collect all possible pieces at this position
                    var candidates = new List<(string piece, double score)>();

                    for (int len = 1; len <= Math.Min(_maxPieceLength, processed.Length - pos); len++)
                    {
                        var piece = processed.Substring(pos, len);
                        if (_scores.TryGetValue(piece, out double score))
                        {
                            candidates.Add((piece, score));
                        }
                    }

                    if (candidates.Count == 0)
                    {
                        tokens.Add(UnkTokenId);
                        pos++;
                        continue;
                    }

                    // Sample from candidates using softmax with temperature
                    var probs = candidates.Select(c => Math.Exp(c.score / temperature)).ToArray();
                    double totalProb = probs.Sum();
                    double rand = random.NextDouble() * totalProb;
                    double cumulative = 0;

                    int selectedIdx = 0;
                    for (int i = 0; i < probs.Length; i++)
                    {
                        cumulative += probs[i];
                        if (cumulative >= rand)
                        {
                            selectedIdx = i;
                            break;
                        }
                    }

                    var selected = candidates[selectedIdx];
                    tokens.Add(_vocab.GetValueOrDefault(selected.piece, UnkTokenId));
                    pos += selected.piece.Length;
                }

                results.Add(tokens.ToArray());
            }

            return results;
        }

        /// Loads a SentencePiece model from file.
        /// </summary>
        public void LoadModel(string path)
        {
            // Simple format: each line is "piece\tscore"
            var lines = File.ReadAllLines(path);

            foreach (var line in lines)
            {
                var parts = line.Split('\t');
                if (parts.Length >= 2)
                {
                    var piece = parts[0];
                    if (double.TryParse(parts[1], out double score))
                    {
                        int id = _vocab.Count;
                        AddToken(piece, id, score);
                    }
                }
            }
        }

        /// Saves the model to file.
        /// </summary>
        public void SaveModel(string path)
        {
            using var writer = new StreamWriter(path);
            foreach (var (piece, id) in _vocab.OrderBy(kv => kv.Value))
            {
                var score = _scores.GetValueOrDefault(piece, 0.0);
                writer.WriteLine($"{piece}\t{score}");
            }
        }
    }

    /// Tiktoken-compatible tokenizer for OpenAI models.
    /// Uses byte-level BPE with a specific regex pattern.
    /// </summary>
    public class TiktokenTokenizer : ITokenizer
    {
        /// <summary>API member</summary>
        private readonly Dictionary<byte[], int> _encoder;
        /// <summary>API member</summary>
        private readonly Dictionary<int, byte[]> _decoder;
        /// <summary>API member</summary>
        private readonly Dictionary<(int, int), int> _bpeRanks;
        /// <summary>API member</summary>
        private readonly Regex _pattern;
        /// <summary>API member</summary>
        private readonly Dictionary<string, int> _specialTokens;

        /// <summary>Public API</summary>
        public int PadTokenId { get; } = -1;  // Tiktoken doesn't use padding
        /// <summary>Public API</summary>
        public int UnkTokenId { get; } = -1;  // Tiktoken encodes all bytes
        /// <summary>Public API</summary>
        public int BosTokenId { get; private set; }
        /// <summary>Public API</summary>
        public int EosTokenId { get; private set; }

        /// <summary>Public API</summary>
        public int VocabSize => _encoder.Count;

        // GPT-4 pattern
        private static readonly Regex Gpt4Pattern = new Regex(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled);

        /// <summary>Public API</summary>
        public TiktokenTokenizer()
        {
            _encoder = new Dictionary<byte[], int>(new ByteArrayComparer());
            _decoder = new Dictionary<int, byte[]>();
            _bpeRanks = new Dictionary<(int, int), int>();
            _pattern = Gpt4Pattern;
            _specialTokens = new Dictionary<string, int>();

            // Initialize with byte-level vocabulary (256 base tokens)
            for (int i = 0; i < 256; i++)
            {
                var bytes = new byte[] { (byte)i };
                _encoder[bytes] = i;
                _decoder[i] = bytes;
            }
        }

        /// <summary>Public API</summary>
        public int[] Encode(string text)
        {
            var tokens = new List<int>();
            var matches = _pattern.Matches(text);

            foreach (Match match in matches)
            {
                var piece = match.Value;
                var bytes = Encoding.UTF8.GetBytes(piece);

                // Check for special tokens first
                if (_specialTokens.TryGetValue(piece, out int specialId))
                {
                    tokens.Add(specialId);
                    continue;
                }

                // Apply BPE
                var bpeTokens = BytePairEncode(bytes);
                tokens.AddRange(bpeTokens);
            }

            return tokens.ToArray();
        }

        private List<int> BytePairEncode(byte[] bytes)
        {
            if (bytes.Length == 0)
                return new List<int>();

            if (bytes.Length == 1)
                return new List<int> { _encoder.GetValueOrDefault(bytes, 0) };

            // Convert bytes to initial token IDs
            var tokenIds = bytes.Select(b => _encoder.GetValueOrDefault(new byte[] { b }, 0)).ToList();

            while (tokenIds.Count >= 2)
            {
                // Find the pair with lowest rank
                int minRank = int.MaxValue;
                int minIdx = -1;

                for (int i = 0; i < tokenIds.Count - 1; i++)
                {
                    var pair = (tokenIds[i], tokenIds[i + 1]);
                    if (_bpeRanks.TryGetValue(pair, out int rank) && rank < minRank)
                    {
                        minRank = rank;
                        minIdx = i;
                    }
                }

                if (minIdx == -1)
                    break;

                // Merge the pair
                var mergedBytes = _decoder[tokenIds[minIdx]].Concat(_decoder[tokenIds[minIdx + 1]]).ToArray();
                int mergedId = _encoder.GetValueOrDefault(mergedBytes, tokenIds[minIdx]);

                tokenIds[minIdx] = mergedId;
                tokenIds.RemoveAt(minIdx + 1);
            }

            return tokenIds;
        }

        /// <summary>Public API</summary>
        public string Decode(int[] tokens)
        {
            var bytes = new List<byte>();

            foreach (var tokenId in tokens)
            {
                if (_decoder.TryGetValue(tokenId, out byte[]? tokenBytes))
                {
                    bytes.AddRange(tokenBytes);
                }
            }

            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        /// Loads a tiktoken model from base64-encoded merges.
        /// </summary>
        public void LoadMerges(string path)
        {
            var lines = File.ReadAllLines(path);
            int rank = 256;  // Start after base vocabulary

            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                try
                {
                    var bytes = Convert.FromBase64String(line.Trim());
                    _encoder[bytes] = rank;
                    _decoder[rank] = bytes;

                    // Extract merge ranks for pairs
                    if (bytes.Length >= 2)
                    {
                        // This is simplified - full implementation needs proper pair tracking
                    }

                    rank++;
                }
                catch
                {
                    // Skip invalid lines
                }
            }
        }

        /// Adds a special token to the vocabulary.
        /// </summary>
        public void AddSpecialToken(string token, int id)
        {
            _specialTokens[token] = id;

            if (token == "<|endoftext|>")
                EosTokenId = id;
            else if (token == "<|startoftext|>")
                BosTokenId = id;
        }

        private class ByteArrayComparer : IEqualityComparer<byte[]>
        {
            /// <summary>Public API</summary>
            public bool Equals(byte[]? x, byte[]? y)
            {
                if (x == null || y == null)
                    return x == y;
                return x.SequenceEqual(y);
            }

            /// <summary>Public API</summary>
            public int GetHashCode(byte[] obj)
            {
                int hash = 17;
                foreach (var b in obj)
                    hash = hash * 31 + b;
                return hash;
            }
        }
    }
}