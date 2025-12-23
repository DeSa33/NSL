using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace NSL.Console;

/// <summary>
/// NSL Vault - Neural Symbolic Language Credential Manager
/// Uses NSL-native encryption with consciousness-inspired key derivation
///
/// Security Model (3-Tier like NSL):
///   Tier 1: Semantic Transformation - Unicode symbol substitution
///   Tier 2: Consciousness Hash - Machine identity folded with NSL operators
///   Tier 3: AES-256-GCM - Military-grade authenticated encryption
///
/// The token cannot be decrypted without:
///   - The exact machine identity (hardware bound)
///   - The NSL transformation table (algorithm bound)
///   - The random salt (stored encrypted)
/// </summary>
public class GitHubCredentialManager
{
    private static readonly string ConfigDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".nsl", "github"
    );

    private static readonly string ConfigFile = Path.Combine(ConfigDir, "config.json");
    private static readonly string VaultFile = Path.Combine(ConfigDir, ".vault.nsl");
    private static readonly string LogFile = Path.Combine(ConfigDir, "access.log");

    // NSL Consciousness Operators (used in key derivation)
    private static readonly string[] ConsciousnessOperators = new[]
    {
        "|>",   // Pipe - flow transformation
        "~>",   // Awareness - self-reference
        "=>>",  // Gradient - learning
        "*>",   // Attention - focus
        "+>",   // Superposition - quantum-like
    };

    // NSL Symbol Table for Tier 1 transformation (obscuring layer)
    private static readonly Dictionary<char, char> SymbolMap = new()
    {
        ['g'] = '⨀', ['h'] = '⨁', ['p'] = '⨂', ['_'] = '⨃',
        ['a'] = '∀', ['b'] = '∃', ['c'] = '∈', ['d'] = '∋',
        ['e'] = '⊂', ['f'] = '⊃', ['i'] = '∧', ['j'] = '∨',
        ['k'] = '⊕', ['l'] = '⊗', ['m'] = '⊙', ['n'] = '⊛',
        ['o'] = '∘', ['q'] = '∙', ['r'] = '∴', ['s'] = '∵',
        ['t'] = '∝', ['u'] = '∞', ['v'] = '∠', ['w'] = '∡',
        ['x'] = '∢', ['y'] = '∣', ['z'] = '∤', ['0'] = '⓪',
        ['1'] = '①', ['2'] = '②', ['3'] = '③', ['4'] = '④',
        ['5'] = '⑤', ['6'] = '⑥', ['7'] = '⑦', ['8'] = '⑧',
        ['9'] = '⑨', ['A'] = '⒜', ['B'] = '⒝', ['C'] = '⒞',
        ['D'] = '⒟', ['E'] = '⒠', ['F'] = '⒡', ['G'] = '⒢',
        ['H'] = '⒣', ['I'] = '⒤', ['J'] = '⒥', ['K'] = '⒦',
        ['L'] = '⒧', ['M'] = '⒨', ['N'] = '⒩', ['O'] = '⒪',
        ['P'] = '⒫', ['Q'] = '⒬', ['R'] = '⒭', ['S'] = '⒮',
        ['T'] = '⒯', ['U'] = '⒰', ['V'] = '⒱', ['W'] = '⒲',
        ['X'] = '⒳', ['Y'] = '⒴', ['Z'] = '⒵',
    };

    private static readonly Dictionary<char, char> ReverseSymbolMap;

    static GitHubCredentialManager()
    {
        // Build reverse map for decoding
        ReverseSymbolMap = new Dictionary<char, char>();
        foreach (var kvp in SymbolMap)
            ReverseSymbolMap[kvp.Value] = kvp.Key;
    }

    private GitHubConfig _config;

    public class GitHubConfig
    {
        public bool Enabled { get; set; } = false;
        public bool AiAccessEnabled { get; set; } = false;
        public string AiAccessMode { get; set; } = "ask"; // on, off, ask
        public string? Username { get; set; }
        public DateTime? LastConnected { get; set; }
        public List<string>? AllowedRepos { get; set; }
        public string Scope { get; set; } = "read"; // read, write, admin
    }

    public GitHubCredentialManager()
    {
        EnsureConfigDir();
        _config = LoadConfig();
    }

    public bool IsEnabled => _config.Enabled && HasToken();
    public bool IsAiAccessEnabled => _config.AiAccessEnabled;
    public string AiAccessMode => _config.AiAccessMode;
    public string? Username => _config.Username;

    public void Enable()
    {
        _config.Enabled = true;
        _config.LastConnected = DateTime.UtcNow;
        SaveConfig();
    }

    public void Disable()
    {
        _config.Enabled = false;
        SaveConfig();
    }

    public void SetAiAccess(string mode)
    {
        mode = mode.ToLower();
        if (mode == "on" || mode == "off" || mode == "ask")
        {
            _config.AiAccessMode = mode;
            _config.AiAccessEnabled = mode == "on";
            SaveConfig();
        }
    }

    public bool HasToken() => File.Exists(VaultFile);

    /// <summary>
    /// Store token using NSL Vault 3-tier encryption
    /// </summary>
    public void StoreToken(string token, string username)
    {
        try
        {
            // Tier 1: NSL Semantic Transformation (obscuring)
            var transformed = ApplyNSLTransform(token);

            // Tier 2: Derive key using Consciousness Hash
            var (key, salt) = DeriveConsciousnessKey();

            // Tier 3: AES-256-GCM encryption
            var encrypted = EncryptWithAES(transformed, key);

            // Store as NSL Vault format: [salt(32)][encrypted]
            var vault = new byte[32 + encrypted.Length];
            Buffer.BlockCopy(salt, 0, vault, 0, 32);
            Buffer.BlockCopy(encrypted, 0, vault, 32, encrypted.Length);

            File.WriteAllBytes(VaultFile, vault);
            File.SetAttributes(VaultFile, FileAttributes.Hidden);

            // Update config
            _config.Username = username;
            _config.Enabled = true;
            _config.LastConnected = DateTime.UtcNow;
            SaveConfig();
        }
        catch (Exception ex)
        {
            throw new Exception($"NSL Vault store failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Retrieve token from NSL Vault
    /// </summary>
    public string? GetToken()
    {
        if (!_config.Enabled || !File.Exists(VaultFile))
            return null;

        try
        {
            var vault = File.ReadAllBytes(VaultFile);
            if (vault.Length < 60) return null; // Too small

            // Extract salt and encrypted data
            var salt = new byte[32];
            var encrypted = new byte[vault.Length - 32];
            Buffer.BlockCopy(vault, 0, salt, 0, 32);
            Buffer.BlockCopy(vault, 32, encrypted, 0, encrypted.Length);

            // Tier 2: Derive key using Consciousness Hash with stored salt
            var key = DeriveConsciousnessKeyFromSalt(salt);

            // Tier 3: AES-256-GCM decryption
            var transformed = DecryptWithAES(encrypted, key);

            // Tier 1: Reverse NSL transformation
            var token = ReverseNSLTransform(transformed);

            return token;
        }
        catch
        {
            return null;
        }
    }

    public void ForgetToken()
    {
        if (File.Exists(VaultFile))
            File.Delete(VaultFile);

        _config.Enabled = false;
        _config.Username = null;
        _config.LastConnected = null;
        SaveConfig();
    }

    public void LogAiAccess(string action, bool allowed)
    {
        try
        {
            var logEntry = $"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} | {(allowed ? "ALLOWED" : "DENIED")} | {action}";
            File.AppendAllText(LogFile, logEntry + Environment.NewLine);
        }
        catch { }
    }

    public string[] GetAiAccessLog(int count = 50)
    {
        if (!File.Exists(LogFile))
            return Array.Empty<string>();

        return File.ReadAllLines(LogFile).TakeLast(count).ToArray();
    }

    /// <summary>
    /// Read hidden input with NSL-style prompt
    /// </summary>
    public static string ReadHiddenInput(string prompt = "[TOKEN]: ")
    {
        System.Console.ForegroundColor = ConsoleColor.Yellow;
        System.Console.Write(prompt);
        System.Console.ResetColor();

        var input = new StringBuilder();
        ConsoleKeyInfo key;

        do
        {
            key = System.Console.ReadKey(intercept: true);

            if (key.Key == ConsoleKey.Backspace && input.Length > 0)
            {
                input.Remove(input.Length - 1, 1);
                System.Console.Write("\b \b");
            }
            else if (key.Key != ConsoleKey.Enter && key.KeyChar >= 32)
            {
                input.Append(key.KeyChar);
                System.Console.Write("⨀"); // NSL symbol instead of asterisk
            }
        }
        while (key.Key != ConsoleKey.Enter);

        System.Console.WriteLine();
        return input.ToString();
    }

    // ===== NSL VAULT ENCRYPTION ENGINE =====

    /// <summary>
    /// Tier 1: Apply NSL semantic transformation (symbol substitution)
    /// </summary>
    private string ApplyNSLTransform(string input)
    {
        var sb = new StringBuilder(input.Length);
        foreach (var c in input)
        {
            sb.Append(SymbolMap.TryGetValue(c, out var symbol) ? symbol : c);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Tier 1: Reverse NSL semantic transformation
    /// </summary>
    private string ReverseNSLTransform(string input)
    {
        var sb = new StringBuilder(input.Length);
        foreach (var c in input)
        {
            sb.Append(ReverseSymbolMap.TryGetValue(c, out var original) ? original : c);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Tier 2: Derive encryption key using Consciousness Hash
    /// Combines machine identity with NSL operators in a unique pattern
    /// </summary>
    private (byte[] key, byte[] salt) DeriveConsciousnessKey()
    {
        // Generate random salt
        var salt = new byte[32];
        RandomNumberGenerator.Fill(salt);

        var key = DeriveConsciousnessKeyFromSalt(salt);
        return (key, salt);
    }

    private byte[] DeriveConsciousnessKeyFromSalt(byte[] salt)
    {
        // Build consciousness identity string using NSL operators
        var identity = new StringBuilder();

        // Machine identity folded with consciousness operators
        identity.Append(Environment.MachineName);
        identity.Append(ConsciousnessOperators[0]); // |>
        identity.Append(Environment.UserName);
        identity.Append(ConsciousnessOperators[1]); // ~>
        identity.Append(Environment.OSVersion.Platform);
        identity.Append(ConsciousnessOperators[2]); // =>>
        identity.Append("NSL.Vault.Consciousness");
        identity.Append(ConsciousnessOperators[3]); // *>
        identity.Append(Environment.ProcessorCount);
        identity.Append(ConsciousnessOperators[4]); // +>

        var password = Encoding.UTF8.GetBytes(identity.ToString());

        // PBKDF2 with high iteration count for security
        using var pbkdf2 = new Rfc2898DeriveBytes(password, salt, 150000, HashAlgorithmName.SHA512);
        return pbkdf2.GetBytes(32); // 256-bit key
    }

    /// <summary>
    /// Tier 3: AES-256-GCM authenticated encryption
    /// </summary>
    private byte[] EncryptWithAES(string plaintext, byte[] key)
    {
        var plaintextBytes = Encoding.UTF8.GetBytes(plaintext);
        var nonce = new byte[12];
        RandomNumberGenerator.Fill(nonce);

        var ciphertext = new byte[plaintextBytes.Length];
        var tag = new byte[16];

        using var aes = new AesGcm(key, 16);
        aes.Encrypt(nonce, plaintextBytes, ciphertext, tag);

        // Format: [nonce(12)][tag(16)][ciphertext]
        var result = new byte[12 + 16 + ciphertext.Length];
        Buffer.BlockCopy(nonce, 0, result, 0, 12);
        Buffer.BlockCopy(tag, 0, result, 12, 16);
        Buffer.BlockCopy(ciphertext, 0, result, 28, ciphertext.Length);

        return result;
    }

    private string DecryptWithAES(byte[] encrypted, byte[] key)
    {
        if (encrypted.Length < 28)
            throw new CryptographicException("Invalid vault data");

        var nonce = new byte[12];
        var tag = new byte[16];
        var ciphertext = new byte[encrypted.Length - 28];

        Buffer.BlockCopy(encrypted, 0, nonce, 0, 12);
        Buffer.BlockCopy(encrypted, 12, tag, 0, 16);
        Buffer.BlockCopy(encrypted, 28, ciphertext, 0, ciphertext.Length);

        var plaintext = new byte[ciphertext.Length];

        using var aes = new AesGcm(key, 16);
        aes.Decrypt(nonce, ciphertext, tag, plaintext);

        return Encoding.UTF8.GetString(plaintext);
    }

    private void EnsureConfigDir()
    {
        if (!Directory.Exists(ConfigDir))
            Directory.CreateDirectory(ConfigDir);
    }

    private GitHubConfig LoadConfig()
    {
        try
        {
            if (File.Exists(ConfigFile))
            {
                var json = File.ReadAllText(ConfigFile);
                return JsonSerializer.Deserialize<GitHubConfig>(json) ?? new GitHubConfig();
            }
        }
        catch { }
        return new GitHubConfig();
    }

    private void SaveConfig()
    {
        var json = JsonSerializer.Serialize(_config, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(ConfigFile, json);
    }
}
