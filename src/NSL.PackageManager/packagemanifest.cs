using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NSL.PackageManager
{
    /// <summary>
    /// Represents an NSL package manifest (nsl-package.json)
    /// </summary>
    public class PackageManifest
    {
        /// <summary>
        /// Package name (required) - must be lowercase, alphanumeric with hyphens
        /// </summary>
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        /// <summary>
        /// Package version using semantic versioning (required)
        /// </summary>
        [JsonPropertyName("version")]
        public string Version { get; set; } = "1.0.0";

        /// <summary>
        /// Short description of the package
        /// </summary>
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        /// <summary>
        /// Package author(s)
        /// </summary>
        [JsonPropertyName("authors")]
        public List<string> Authors { get; set; } = new();

        /// <summary>
        /// Package license (SPDX identifier)
        /// </summary>
        [JsonPropertyName("license")]
        public string? License { get; set; }

        /// <summary>
        /// Package homepage URL
        /// </summary>
        [JsonPropertyName("homepage")]
        public string? Homepage { get; set; }

        /// <summary>
        /// Repository URL
        /// </summary>
        [JsonPropertyName("repository")]
        public string? Repository { get; set; }

        /// <summary>
        /// Keywords for package discovery
        /// </summary>
        [JsonPropertyName("keywords")]
        public List<string> Keywords { get; set; } = new();

        /// <summary>
        /// Main entry point file
        /// </summary>
        [JsonPropertyName("main")]
        public string Main { get; set; } = "main.nsl";

        /// <summary>
        /// Package dependencies with version constraints
        /// </summary>
        [JsonPropertyName("dependencies")]
        public Dictionary<string, string> Dependencies { get; set; } = new();

        /// <summary>
        /// Development-only dependencies
        /// </summary>
        [JsonPropertyName("devDependencies")]
        public Dictionary<string, string> DevDependencies { get; set; } = new();

        /// <summary>
        /// Peer dependencies (must be installed by consuming package)
        /// </summary>
        [JsonPropertyName("peerDependencies")]
        public Dictionary<string, string> PeerDependencies { get; set; } = new();

        /// <summary>
        /// Minimum NSL version required
        /// </summary>
        [JsonPropertyName("nslVersion")]
        public string? NslVersion { get; set; }

        /// <summary>
        /// Executable scripts/commands provided by this package
        /// </summary>
        [JsonPropertyName("scripts")]
        public Dictionary<string, string> Scripts { get; set; } = new();

        /// <summary>
        /// Files to include in the package (glob patterns)
        /// </summary>
        [JsonPropertyName("files")]
        public List<string> Files { get; set; } = new() { "**/*.nsl" };

        /// <summary>
        /// Whether this package is private (not publishable)
        /// </summary>
        [JsonPropertyName("private")]
        public bool Private { get; set; } = false;

        /// <summary>
        /// Custom metadata
        /// </summary>
        [JsonPropertyName("metadata")]
        public Dictionary<string, object> Metadata { get; set; } = new();

        /// <summary>
        /// Export definitions - what this package exposes
        /// </summary>
        [JsonPropertyName("exports")]
        public Dictionary<string, string> Exports { get; set; } = new();

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            PropertyNameCaseInsensitive = true
        };

        /// <summary>
        /// Load manifest from file
        /// </summary>
        public static PackageManifest Load(string path)
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<PackageManifest>(json, _jsonOptions)
                ?? throw new InvalidOperationException("Failed to parse package manifest");
        }

        /// <summary>
        /// Load manifest from directory (looks for nsl-package.json)
        /// </summary>
        public static PackageManifest LoadFromDirectory(string directory)
        {
            var manifestPath = Path.Combine(directory, "nsl-package.json");
            if (!File.Exists(manifestPath))
            {
                throw new FileNotFoundException($"No nsl-package.json found in {directory}");
            }
            return Load(manifestPath);
        }

        /// <summary>
        /// Save manifest to file
        /// </summary>
        public void Save(string path)
        {
            var json = JsonSerializer.Serialize(this, _jsonOptions);
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Save manifest to directory (creates nsl-package.json)
        /// </summary>
        public void SaveToDirectory(string directory)
        {
            Directory.CreateDirectory(directory);
            Save(Path.Combine(directory, "nsl-package.json"));
        }

        /// <summary>
        /// Validate the manifest
        /// </summary>
        public ValidationResult Validate()
        {
            var errors = new List<string>();
            var warnings = new List<string>();

            // Required fields
            if (string.IsNullOrWhiteSpace(Name))
                errors.Add("Package name is required");
            else if (!IsValidPackageName(Name))
                errors.Add("Package name must be lowercase, alphanumeric with hyphens only");

            if (string.IsNullOrWhiteSpace(Version))
                errors.Add("Package version is required");
            else if (!SemanticVersion.TryParse(Version, out _))
                errors.Add($"Invalid semantic version: {Version}");

            // Validate dependencies
            foreach (var (dep, constraint) in Dependencies)
            {
                if (!IsValidPackageName(dep))
                    errors.Add($"Invalid dependency name: {dep}");
                if (!VersionConstraint.TryParse(constraint, out _))
                    errors.Add($"Invalid version constraint for {dep}: {constraint}");
            }

            // Warnings
            if (string.IsNullOrWhiteSpace(Description))
                warnings.Add("Package description is recommended");
            if (!Authors.Any())
                warnings.Add("Package authors are recommended");
            if (string.IsNullOrWhiteSpace(License))
                warnings.Add("Package license is recommended");

            return new ValidationResult(errors.Count == 0, errors, warnings);
        }

        /// <summary>
        /// Get the full package identifier (name@version)
        /// </summary>
        public string GetPackageId() => $"{Name}@{Version}";

        private static bool IsValidPackageName(string name)
        {
            if (string.IsNullOrEmpty(name) || name.Length > 214)
                return false;

            // Must start with letter or @scope/
            if (!char.IsLetter(name[0]) && name[0] != '@')
                return false;

            // Only lowercase, digits, hyphens, and @ for scopes
            foreach (var c in name)
            {
                if (!char.IsLetterOrDigit(c) && c != '-' && c != '@' && c != '/' && c != '_')
                    return false;
            }

            return true;
        }

        /// <summary>Public API</summary>
        public override string ToString() => $"{Name}@{Version}";
    }

    /// <summary>
    /// Result of manifest validation
    /// </summary>
    public class ValidationResult
    {
        /// <summary>Public API</summary>
        public bool IsValid { get; }
        /// <summary>Public API</summary>
        public List<string> Errors { get; }
        /// <summary>Public API</summary>
        public List<string> Warnings { get; }

        /// <summary>Public API</summary>
        public ValidationResult(bool isValid, List<string> errors, List<string> warnings)
        {
            IsValid = isValid;
            Errors = errors;
            Warnings = warnings;
        }
    }

    /// <summary>
    /// Semantic version representation
    /// </summary>
    public class SemanticVersion : IComparable<SemanticVersion>
    {
        /// <summary>Public API</summary>
        public int Major { get; }
        /// <summary>Public API</summary>
        public int Minor { get; }
        /// <summary>Public API</summary>
        public int Patch { get; }
        /// <summary>Public API</summary>
        public string? PreRelease { get; }
        /// <summary>Public API</summary>
        public string? BuildMetadata { get; }

        /// <summary>Public API</summary>
        public SemanticVersion(int major, int minor, int patch, string? preRelease = null, string? buildMetadata = null)
        {
            Major = major;
            Minor = minor;
            Patch = patch;
            PreRelease = preRelease;
            BuildMetadata = buildMetadata;
        }

        /// <summary>Public API</summary>
        public static bool TryParse(string version, out SemanticVersion? result)
        {
            result = null;
            if (string.IsNullOrWhiteSpace(version))
                return false;

            // Remove leading 'v' if present
            if (version.StartsWith('v') || version.StartsWith('V'))
                version = version[1..];

            // Split build metadata
            string? buildMetadata = null;
            var plusIndex = version.IndexOf('+');
            if (plusIndex >= 0)
            {
                buildMetadata = version[(plusIndex + 1)..];
                version = version[..plusIndex];
            }

            // Split pre-release
            string? preRelease = null;
            var hyphenIndex = version.IndexOf('-');
            if (hyphenIndex >= 0)
            {
                preRelease = version[(hyphenIndex + 1)..];
                version = version[..hyphenIndex];
            }

            // Parse major.minor.patch
            var parts = version.Split('.');
            if (parts.Length < 1 || parts.Length > 3)
                return false;

            if (!int.TryParse(parts[0], out var major))
                return false;

            var minor = 0;
            if (parts.Length >= 2 && !int.TryParse(parts[1], out minor))
                return false;

            var patch = 0;
            if (parts.Length >= 3 && !int.TryParse(parts[2], out patch))
                return false;

            result = new SemanticVersion(major, minor, patch, preRelease, buildMetadata);
            return true;
        }

        /// <summary>Public API</summary>
        public static SemanticVersion Parse(string version)
        {
            if (!TryParse(version, out var result))
                throw new FormatException($"Invalid semantic version: {version}");
            return result!;
        }

        /// <summary>Public API</summary>
        public int CompareTo(SemanticVersion? other)
        {
            if (other == null) return 1;

            var majorCompare = Major.CompareTo(other.Major);
            if (majorCompare != 0) return majorCompare;

            var minorCompare = Minor.CompareTo(other.Minor);
            if (minorCompare != 0) return minorCompare;

            var patchCompare = Patch.CompareTo(other.Patch);
            if (patchCompare != 0) return patchCompare;

            // Pre-release versions have lower precedence
            if (PreRelease == null && other.PreRelease != null) return 1;
            if (PreRelease != null && other.PreRelease == null) return -1;
            if (PreRelease != null && other.PreRelease != null)
                return string.Compare(PreRelease, other.PreRelease, StringComparison.Ordinal);

            return 0;
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            var result = $"{Major}.{Minor}.{Patch}";
            if (!string.IsNullOrEmpty(PreRelease))
                result += $"-{PreRelease}";
            if (!string.IsNullOrEmpty(BuildMetadata))
                result += $"+{BuildMetadata}";
            return result;
        }

        /// <summary>Public API</summary>
        public static bool operator <(SemanticVersion? left, SemanticVersion? right)
        {
            if (left is null) return right is not null;
            return left.CompareTo(right) < 0;
        }

        /// <summary>Public API</summary>
        public static bool operator >(SemanticVersion? left, SemanticVersion? right)
        {
            if (left is null) return false;
            return left.CompareTo(right) > 0;
        }

        /// <summary>Public API</summary>
        public static bool operator <=(SemanticVersion? left, SemanticVersion? right)
        {
            if (left is null) return true;
            return left.CompareTo(right) <= 0;
        }

        /// <summary>Public API</summary>
        public static bool operator >=(SemanticVersion? left, SemanticVersion? right)
        {
            if (left is null) return right is null;
            return left.CompareTo(right) >= 0;
        }

        /// <summary>Public API</summary>
        public static bool operator ==(SemanticVersion? left, SemanticVersion? right)
        {
            if (left is null) return right is null;
            return left.CompareTo(right) == 0;
        }

        /// <summary>Public API</summary>
        public static bool operator !=(SemanticVersion? left, SemanticVersion? right)
        {
            if (left is null) return right is not null;
            return left.CompareTo(right) != 0;
        }

        /// <summary>Public API</summary>
        public override bool Equals(object? obj) => obj is SemanticVersion other && CompareTo(other) == 0;
        /// <summary>Public API</summary>
        public override int GetHashCode() => HashCode.Combine(Major, Minor, Patch, PreRelease);
    }

    /// <summary>
    /// Version constraint for dependencies
    /// </summary>
    public class VersionConstraint
    {
        /// <summary>Public API</summary>
        public enum ConstraintType
        {
            Exact,      // 1.2.3
            Caret,      // ^1.2.3 (compatible with 1.x.x)
            Tilde,      // ~1.2.3 (compatible with 1.2.x)
            /// <summary>API member</summary>
            GreaterThan,
            /// <summary>API member</summary>
            GreaterThanOrEqual,
            /// <summary>API member</summary>
            LessThan,
            /// <summary>API member</summary>
            LessThanOrEqual,
            Range,      // >=1.0.0 <2.0.0
            Any         // *
        }

        /// <summary>Public API</summary>
        public ConstraintType Type { get; }
        /// <summary>Public API</summary>
        public SemanticVersion? MinVersion { get; }
        /// <summary>Public API</summary>
        public SemanticVersion? MaxVersion { get; }
        /// <summary>Public API</summary>
        public bool MinInclusive { get; }
        /// <summary>Public API</summary>
        public bool MaxInclusive { get; }

        private VersionConstraint(ConstraintType type, SemanticVersion? minVersion = null,
            SemanticVersion? maxVersion = null, bool minInclusive = true, bool maxInclusive = false)
        {
            Type = type;
            MinVersion = minVersion;
            MaxVersion = maxVersion;
            MinInclusive = minInclusive;
            MaxInclusive = maxInclusive;
        }

        /// <summary>Public API</summary>
        public static bool TryParse(string constraint, out VersionConstraint? result)
        {
            result = null;
            if (string.IsNullOrWhiteSpace(constraint))
                return false;

            constraint = constraint.Trim();

            // Any version
            if (constraint == "*" || constraint == "latest")
            {
                result = new VersionConstraint(ConstraintType.Any);
                return true;
            }

            // Caret constraint ^1.2.3
            if (constraint.StartsWith('^'))
            {
                if (!SemanticVersion.TryParse(constraint[1..], out var version))
                    return false;
                result = new VersionConstraint(ConstraintType.Caret, version);
                return true;
            }

            // Tilde constraint ~1.2.3
            if (constraint.StartsWith('~'))
            {
                if (!SemanticVersion.TryParse(constraint[1..], out var version))
                    return false;
                result = new VersionConstraint(ConstraintType.Tilde, version);
                return true;
            }

            // Comparison operators
            if (constraint.StartsWith(">="))
            {
                if (!SemanticVersion.TryParse(constraint[2..].Trim(), out var version))
                    return false;
                result = new VersionConstraint(ConstraintType.GreaterThanOrEqual, version);
                return true;
            }

            if (constraint.StartsWith("<="))
            {
                if (!SemanticVersion.TryParse(constraint[2..].Trim(), out var version))
                    return false;
                result = new VersionConstraint(ConstraintType.LessThanOrEqual, null, version, maxInclusive: true);
                return true;
            }

            if (constraint.StartsWith('>'))
            {
                if (!SemanticVersion.TryParse(constraint[1..].Trim(), out var version))
                    return false;
                result = new VersionConstraint(ConstraintType.GreaterThan, version, minInclusive: false);
                return true;
            }

            if (constraint.StartsWith('<'))
            {
                if (!SemanticVersion.TryParse(constraint[1..].Trim(), out var version))
                    return false;
                result = new VersionConstraint(ConstraintType.LessThan, null, version);
                return true;
            }

            // Exact version
            if (SemanticVersion.TryParse(constraint, out var exactVersion))
            {
                result = new VersionConstraint(ConstraintType.Exact, exactVersion, exactVersion, true, true);
                return true;
            }

            return false;
        }

        /// <summary>Public API</summary>
        public static VersionConstraint Parse(string constraint)
        {
            if (!TryParse(constraint, out var result))
                throw new FormatException($"Invalid version constraint: {constraint}");
            return result!;
        }

        /// <summary>Public API</summary>
        public bool IsSatisfiedBy(SemanticVersion version)
        {
            return Type switch
            {
                /// <summary>API member</summary>
                ConstraintType.Any => true,
                /// <summary>API member</summary>
                ConstraintType.Exact => version == MinVersion,
                ConstraintType.Caret => IsSatisfiedByCaret(version),
                ConstraintType.Tilde => IsSatisfiedByTilde(version),
                ConstraintType.GreaterThan => MinVersion != null && version > MinVersion,
                ConstraintType.GreaterThanOrEqual => MinVersion != null && version >= MinVersion,
                ConstraintType.LessThan => MaxVersion != null && version < MaxVersion,
                ConstraintType.LessThanOrEqual => MaxVersion != null && version <= MaxVersion,
                ConstraintType.Range => IsSatisfiedByRange(version),
                _ => false
            };
        }

        private bool IsSatisfiedByCaret(SemanticVersion version)
        {
            if (MinVersion == null) return false;

            // ^1.2.3 allows 1.x.x where x >= 2.3
            if (MinVersion.Major == 0)
            {
                // ^0.x.y is more restrictive
                if (MinVersion.Minor == 0)
                    return version.Major == 0 && version.Minor == 0 && version.Patch >= MinVersion.Patch;
                return version.Major == 0 && version.Minor == MinVersion.Minor && version.Patch >= MinVersion.Patch;
            }

            return version.Major == MinVersion.Major && version >= MinVersion;
        }

        private bool IsSatisfiedByTilde(SemanticVersion version)
        {
            if (MinVersion == null) return false;

            // ~1.2.3 allows 1.2.x where x >= 3
            return version.Major == MinVersion.Major &&
                   version.Minor == MinVersion.Minor &&
                   version.Patch >= MinVersion.Patch;
        }

        private bool IsSatisfiedByRange(SemanticVersion version)
        {
            if (MinVersion != null)
            {
                var minSatisfied = MinInclusive ? version >= MinVersion : version > MinVersion;
                if (!minSatisfied) return false;
            }

            if (MaxVersion != null)
            {
                var maxSatisfied = MaxInclusive ? version <= MaxVersion : version < MaxVersion;
                if (!maxSatisfied) return false;
            }

            return true;
        }
    }
}