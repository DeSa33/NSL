using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace NSL.Console;

/// <summary>
/// NSLZ - NSL Native Semantic Compression
/// 7-Tier system that compresses MEANING, not just bytes
/// </summary>
public class NSLZCompressor
{
    // ===== CASE MARKERS =====
    private const char CaseMarkerCapitalize = 'ˆ';  // U+02C6 - capitalize first letter
    private const char CaseMarkerAllCaps = 'ˇ';     // U+02C7 - ALL CAPS

    // ===== 7-TIER SEMANTIC DICTIONARY =====
    // Each tier maps words/phrases to single Unicode symbols

    /// <summary>Tier 1: Ultra-high frequency words (50-70% of text)</summary>
    private static readonly Dictionary<string, char> Tier1 = new(StringComparer.OrdinalIgnoreCase)
    {
        // Articles & Determiners
        ["the"] = '∀', ["a"] = '∁', ["an"] = '∂',
        // Pronouns
        ["i"] = '∃', ["you"] = '∄', ["he"] = '∅', ["she"] = '∆', ["it"] = '∇',
        ["we"] = '∈', ["they"] = '∉', ["me"] = '∊', ["him"] = '∋', ["her"] = '∌',
        ["my"] = '∍', ["your"] = '∎', ["his"] = '∏', ["its"] = '∐',
        ["our"] = '∑', ["their"] = '−', ["this"] = '∓', ["that"] = '∔',
        // Verbs (common)
        ["is"] = '∕', ["are"] = '∖', ["was"] = '∗', ["were"] = '∘',
        ["be"] = '∙', ["been"] = '√', ["being"] = '∛', ["am"] = '∜',
        ["have"] = '∝', ["has"] = '∞', ["had"] = '∟', ["do"] = '∠',
        ["does"] = '∡', ["did"] = '∢', ["will"] = '∣', ["would"] = '∤',
        ["can"] = '∥', ["could"] = '∦', ["should"] = '∧', ["may"] = '∨',
        ["might"] = '∩', ["must"] = '∪', ["shall"] = '∫',
        // Prepositions
        ["of"] = '∬', ["to"] = '∭', ["in"] = '∮', ["for"] = '∯',
        ["on"] = '∰', ["with"] = '∱', ["at"] = '∲', ["by"] = '∳',
        ["from"] = '∴', ["as"] = '∵', ["into"] = '∶', ["about"] = '∷',
        // Conjunctions
        ["and"] = '∸', ["or"] = '∹', ["but"] = '∺', ["if"] = '∻',
        ["then"] = '∼', ["so"] = '∽', ["because"] = '∾', ["when"] = '∿',
        // Common words
        ["not"] = '≀', ["no"] = '≁', ["yes"] = '≂', ["all"] = '≃',
        ["more"] = '≄', ["some"] = '≅', ["any"] = '≆', ["each"] = '≇',
        ["which"] = '≈', ["who"] = '≉', ["what"] = '≊', ["how"] = '≋',
        ["there"] = '≌', ["here"] = '≍', ["also"] = '≎', ["only"] = '≏',
        ["new"] = '≐', ["now"] = '≑', ["very"] = '≒', ["just"] = '≓',
        ["than"] = '≔', ["like"] = '≕', ["other"] = '≖', ["such"] = '≗',
        ["get"] = '≘', ["make"] = '≙', ["know"] = '≚', ["take"] = '≛',
        ["see"] = '≜', ["come"] = '≝', ["go"] = '≞', ["want"] = '≟',
        ["use"] = '≠', ["find"] = '≡', ["give"] = '≢', ["tell"] = '≣',
        ["work"] = '≤', ["call"] = '≥', ["try"] = '≦', ["need"] = '≧',
        ["feel"] = '≨', ["become"] = '≩', ["leave"] = '≪', ["put"] = '≫',
        ["mean"] = '≬', ["keep"] = '≭', ["let"] = '≮', ["begin"] = '≯',
        ["seem"] = '≰', ["help"] = '≱', ["show"] = '≲', ["hear"] = '≳',
        ["play"] = '≴', ["run"] = '≵', ["move"] = '≶', ["live"] = '≷',
        ["believe"] = '≸', ["hold"] = '≹', ["bring"] = '≺', ["happen"] = '≻',
        ["write"] = '≼', ["provide"] = '≽', ["sit"] = '≾', ["stand"] = '≿',
        ["lose"] = '⊀', ["pay"] = '⊁', ["meet"] = '⊂', ["include"] = '⊃',
        ["continue"] = '⊄', ["set"] = '⊅', ["learn"] = '⊆', ["change"] = '⊇',
        ["lead"] = '⊈', ["understand"] = '⊉', ["watch"] = '⊊', ["follow"] = '⊋',
        ["stop"] = '⊌', ["create"] = '⊍', ["speak"] = '⊎', ["read"] = '⊏',
        ["allow"] = '⊐', ["add"] = '⊑', ["spend"] = '⊒', ["grow"] = '⊓',
        ["open"] = '⊔', ["walk"] = '⊕', ["win"] = '⊖', ["offer"] = '⊗',
        ["remember"] = '⊘', ["love"] = '⊙', ["consider"] = '⊚', ["appear"] = '⊛',
        ["buy"] = '⊜', ["wait"] = '⊝', ["serve"] = '⊞', ["die"] = '⊟',
        ["send"] = '⊠', ["expect"] = '⊡', ["build"] = '⊢', ["stay"] = '⊣',
        ["fall"] = '⊤', ["cut"] = '⊥', ["reach"] = '⊦', ["kill"] = '⊧',
        ["remain"] = '⊨',
    };

    /// <summary>Tier 2: Morphology - word endings</summary>
    private static readonly Dictionary<string, char> Tier2 = new(StringComparer.OrdinalIgnoreCase)
    {
        ["ing"] = '⋀', ["ed"] = '⋁', ["tion"] = '⋂', ["ment"] = '⋃',
        ["ness"] = '⋄', ["able"] = '⋅', ["ible"] = '⋆', ["ful"] = '⋇',
        ["less"] = '⋈', ["ous"] = '⋉', ["ive"] = '⋊', ["ly"] = '⋋',
        ["er"] = '⋌', ["est"] = '⋍', ["ity"] = '⋎', ["ism"] = '⋏',
        ["ist"] = '⋐', ["ize"] = '⋑', ["ise"] = '⋒', ["ify"] = '⋓',
        ["ary"] = '⋔', ["ory"] = '⋕', ["al"] = '⋖', ["ic"] = '⋗',
        ["ical"] = '⋘', ["ous"] = '⋙', ["eous"] = '⋚', ["ious"] = '⋛',
    };

    /// <summary>Tier 6: Tech/AI vocabulary</summary>
    private static readonly Dictionary<string, char> Tier6 = new(StringComparer.OrdinalIgnoreCase)
    {
        // Programming
        ["function"] = '⌀', ["variable"] = '⌁', ["class"] = '⌂', ["method"] = '⌃',
        ["object"] = '⌄', ["array"] = '⌅', ["string"] = '⌆', ["integer"] = '⌇',
        ["boolean"] = '⌈', ["null"] = '⌉', ["undefined"] = '⌊', ["return"] = '⌋',
        ["import"] = '⌌', ["export"] = '⌍', ["async"] = '⌎', ["await"] = '⌏',
        ["interface"] = '⌐', ["type"] = '⌑', ["const"] = '⌒', ["static"] = '⌓',
        ["public"] = '⌔', ["private"] = '⌕', ["protected"] = '⌖', ["abstract"] = '⌗',
        // AI/ML
        ["model"] = '⌘', ["training"] = '⌙', ["dataset"] = '⌚', ["epoch"] = '⌛',
        ["batch"] = '⌜', ["layer"] = '⌝', ["weight"] = '⌞', ["bias"] = '⌟',
        ["gradient"] = '⌠', ["loss"] = '⌡', ["optimizer"] = '⌢', ["accuracy"] = '⌣',
        ["precision"] = '⌤', ["recall"] = '⌥', ["tensor"] = '⌦', ["vector"] = '⌧',
        ["matrix"] = '⌨', ["embedding"] = '〈', ["attention"] = '〉', ["transformer"] = '⌫',
        ["encoder"] = '⌬', ["decoder"] = '⌭', ["token"] = '⌮', ["vocabulary"] = '⌯',
        // Systems
        ["server"] = '⌰', ["client"] = '⌱', ["database"] = '⌲', ["cache"] = '⌳',
        ["memory"] = '⌴', ["process"] = '⌵', ["thread"] = '⌶', ["socket"] = '⌷',
        ["request"] = '⌸', ["response"] = '⌹', ["protocol"] = '⌺', ["network"] = '⌻',
        ["api"] = '⌼', ["endpoint"] = '⌽', ["authentication"] = '⌾', ["authorization"] = '⌿',
        // File types
        ["json"] = '⍀', ["xml"] = '⍁', ["html"] = '⍂', ["css"] = '⍃',
        ["javascript"] = '⍄', ["typescript"] = '⍅', ["python"] = '⍆', ["csharp"] = '⍇',
        ["binary"] = '⍈', ["executable"] = '⍉', ["library"] = '⍊', ["module"] = '⍋',
    };

    /// <summary>Tier 7: Multi-word phrases (highest compression)</summary>
    private static readonly Dictionary<string, char> Tier7 = new(StringComparer.OrdinalIgnoreCase)
    {
        // AI/ML Phrases
        ["artificial intelligence"] = '⨀', ["machine learning"] = '⨁',
        ["deep learning"] = '⨂', ["neural network"] = '⨃',
        ["natural language processing"] = '⨄', ["computer vision"] = '⨅',
        ["reinforcement learning"] = '⨆', ["supervised learning"] = '⨇',
        ["unsupervised learning"] = '⨈', ["transfer learning"] = '⨉',
        ["large language model"] = '⨊', ["generative ai"] = '⨋',
        ["prompt engineering"] = '⨌', ["fine tuning"] = '⨍',
        ["attention mechanism"] = '⨎', ["self attention"] = '⨏',
        ["gradient descent"] = '⨐', ["backpropagation"] = '⨑',
        ["activation function"] = '⨒', ["loss function"] = '⨓',
        ["convolutional neural network"] = '⨔', ["recurrent neural network"] = '⨕',
        // Programming Phrases
        ["source code"] = '⨖', ["open source"] = '⨗',
        ["version control"] = '⨘', ["pull request"] = '⨙',
        ["code review"] = '⨚', ["unit test"] = '⨛',
        ["integration test"] = '⨜', ["continuous integration"] = '⨝',
        ["continuous deployment"] = '⨞', ["build system"] = '⨟',
        ["design pattern"] = '⨠', ["best practice"] = '⨡',
        ["error handling"] = '⨢', ["exception handling"] = '⨣',
        ["memory management"] = '⨤', ["garbage collection"] = '⨥',
        ["stack overflow"] = '⨦', ["null pointer"] = '⨧',
        ["race condition"] = '⨨', ["dead lock"] = '⨩',
        // Common Phrases
        ["in order to"] = '⨪', ["as well as"] = '⨫',
        ["such as"] = '⨬', ["due to"] = '⨭',
        ["based on"] = '⨮', ["according to"] = '⨯',
        ["in addition"] = '⨰', ["for example"] = '⨱',
        ["on the other hand"] = '⨲', ["at the same time"] = '⨳',
        ["in terms of"] = '⨴', ["with respect to"] = '⨵',
        ["as a result"] = '⨶', ["in fact"] = '⨷',
        ["of course"] = '⨸', ["at least"] = '⨹',
        ["at most"] = '⨺', ["so far"] = '⨻',
        ["right now"] = '⨼', ["even though"] = '⨽',
        ["as long as"] = '⨾', ["in case"] = '⨿',
        // NSL-specific
        ["consciousness operator"] = '⩀', ["semantic compression"] = '⩁',
        ["neural symbolic"] = '⩂', ["cognitive state"] = '⩃',
    };

    // Reverse lookup tables (built once)
    private static readonly Dictionary<char, string> ReverseTier1;
    private static readonly Dictionary<char, string> ReverseTier2;
    private static readonly Dictionary<char, string> ReverseTier6;
    private static readonly Dictionary<char, string> ReverseTier7;

    static NSLZCompressor()
    {
        ReverseTier1 = Tier1.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        ReverseTier2 = Tier2.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        ReverseTier6 = Tier6.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        ReverseTier7 = Tier7.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
    }

    // ===== COMPRESSION ENGINE =====

    /// <summary>
    /// Compress text using 7-tier semantic compression
    /// </summary>
    public static string CompressText(string text)
    {
        if (string.IsNullOrEmpty(text)) return text;

        var result = new StringBuilder();

        // First pass: Tier 7 (multi-word phrases) - highest compression
        text = ApplyTier7(text);

        // Second pass: Word-by-word compression
        var words = TokenizePreservingWhitespace(text);

        foreach (var token in words)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                result.Append(token);
                continue;
            }

            // Check if already a symbol (from Tier 7)
            if (token.Length == 1 && IsNSLSymbol(token[0]))
            {
                result.Append(token);
                continue;
            }

            // Try Tier 6 (tech vocabulary)
            if (Tier6.TryGetValue(token, out var t6))
            {
                // Preserve case: add marker if word starts with uppercase
                if (token.Length > 0 && char.IsUpper(token[0]))
                {
                    if (token.All(char.IsUpper))
                        result.Append(CaseMarkerAllCaps);
                    else
                        result.Append(CaseMarkerCapitalize);
                }
                result.Append(t6);
                continue;
            }

            // Try Tier 1 (common words)
            if (Tier1.TryGetValue(token, out var t1))
            {
                // Preserve case: add marker if word starts with uppercase
                if (token.Length > 0 && char.IsUpper(token[0]))
                {
                    if (token.All(char.IsUpper))
                        result.Append(CaseMarkerAllCaps);
                    else
                        result.Append(CaseMarkerCapitalize);
                }
                result.Append(t1);
                continue;
            }

            // Keep original word
            result.Append(token);
        }

        return result.ToString();
    }

    /// <summary>
    /// Decompress text back to original
    /// </summary>
    public static string DecompressText(string compressed)
    {
        if (string.IsNullOrEmpty(compressed)) return compressed;

        var result = new StringBuilder();
        bool capitalizeNext = false;
        bool allCapsNext = false;

        foreach (var c in compressed)
        {
            // Handle case markers
            if (c == CaseMarkerCapitalize)
            {
                capitalizeNext = true;
                continue;
            }
            if (c == CaseMarkerAllCaps)
            {
                allCapsNext = true;
                continue;
            }

            string toAppend = null;
            if (ReverseTier7.TryGetValue(c, out var phrase))
            {
                toAppend = phrase;
            }
            else if (ReverseTier6.TryGetValue(c, out var tech))
            {
                toAppend = tech;
            }
            else if (ReverseTier1.TryGetValue(c, out var word))
            {
                toAppend = word;
            }
            else if (ReverseTier2.TryGetValue(c, out var morph))
            {
                toAppend = morph;
            }
            else
            {
                toAppend = c.ToString();
            }

            // Apply case transformation
            if (toAppend != null)
            {
                if (allCapsNext)
                {
                    toAppend = toAppend.ToUpperInvariant();
                    allCapsNext = false;
                }
                else if (capitalizeNext && toAppend.Length > 0)
                {
                    toAppend = char.ToUpperInvariant(toAppend[0]) + toAppend.Substring(1);
                    capitalizeNext = false;
                }
                result.Append(toAppend);
            }
        }

        return result.ToString();
    }

    private static string ApplyTier7(string text)
    {
        // Sort by length (longest first) to match longer phrases first
        var sortedPhrases = Tier7.OrderByDescending(kvp => kvp.Key.Length);

        foreach (var kvp in sortedPhrases)
        {
            text = ReplaceWholePhrase(text, kvp.Key, kvp.Value.ToString());
        }

        return text;
    }

    private static string ReplaceWholePhrase(string text, string phrase, string replacement)
    {
        var result = new StringBuilder();
        int i = 0;

        while (i < text.Length)
        {
            // Check if phrase matches at current position (case-insensitive)
            if (i + phrase.Length <= text.Length)
            {
                var substring = text.Substring(i, phrase.Length);
                if (substring.Equals(phrase, StringComparison.OrdinalIgnoreCase))
                {
                    // Check word boundaries
                    bool startOk = i == 0 || !char.IsLetterOrDigit(text[i - 1]);
                    bool endOk = i + phrase.Length == text.Length || !char.IsLetterOrDigit(text[i + phrase.Length]);

                    if (startOk && endOk)
                    {
                        result.Append(replacement);
                        i += phrase.Length;
                        continue;
                    }
                }
            }

            result.Append(text[i]);
            i++;
        }

        return result.ToString();
    }

    private static List<string> TokenizePreservingWhitespace(string text)
    {
        var tokens = new List<string>();
        var current = new StringBuilder();
        bool inWord = false;

        foreach (var c in text)
        {
            bool isWordChar = char.IsLetterOrDigit(c) || c == '_' || c == '-';

            if (isWordChar)
            {
                if (!inWord && current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
                inWord = true;
                current.Append(c);
            }
            else
            {
                if (inWord && current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
                inWord = false;
                current.Append(c);
            }
        }

        if (current.Length > 0)
            tokens.Add(current.ToString());

        return tokens;
    }

    private static bool IsNSLSymbol(char c)
    {
        return ReverseTier1.ContainsKey(c) ||
               ReverseTier2.ContainsKey(c) ||
               ReverseTier6.ContainsKey(c) ||
               ReverseTier7.ContainsKey(c);
    }

    // ===== BINARY COMPRESSION =====

    /// <summary>
    /// Compress binary data using Brotli (best ratio for .exe files)
    /// </summary>
    public static byte[] CompressBinary(byte[] data, CompressionLevel level = CompressionLevel.Optimal)
    {
        using var output = new MemoryStream();
        using (var brotli = new BrotliStream(output, level))
        {
            brotli.Write(data, 0, data.Length);
        }
        return output.ToArray();
    }

    /// <summary>
    /// Decompress binary data
    /// </summary>
    public static byte[] DecompressBinary(byte[] compressed)
    {
        using var input = new MemoryStream(compressed);
        using var brotli = new BrotliStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        brotli.CopyTo(output);
        return output.ToArray();
    }

    // ===== STATISTICS =====

    public static CompressionStats GetStats(string original, string compressed)
    {
        var originalBytes = Encoding.UTF8.GetByteCount(original);
        var compressedBytes = Encoding.UTF8.GetByteCount(compressed);

        return new CompressionStats
        {
            OriginalChars = original.Length,
            CompressedChars = compressed.Length,
            OriginalBytes = originalBytes,
            CompressedBytes = compressedBytes,
            CharReduction = 1.0 - (double)compressed.Length / original.Length,
            ByteReduction = 1.0 - (double)compressedBytes / originalBytes
        };
    }

    public class CompressionStats
    {
        public int OriginalChars { get; set; }
        public int CompressedChars { get; set; }
        public int OriginalBytes { get; set; }
        public int CompressedBytes { get; set; }
        public double CharReduction { get; set; }
        public double ByteReduction { get; set; }
    }
}
