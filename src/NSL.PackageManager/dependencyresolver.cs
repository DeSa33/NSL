using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NSL.PackageManager
{
    /// <summary>
    /// Resolves package dependencies with conflict detection
    /// </summary>
    public class DependencyResolver
    {
        private readonly IPackageRegistry _registry;
        private readonly Dictionary<string, ResolvedPackage> _resolved = new();
        private readonly Dictionary<string, List<DependencyRequest>> _requests = new();

        /// <summary>Public API</summary>
        public DependencyResolver(IPackageRegistry registry)
        {
            _registry = registry;
        }

        /// <summary>
        /// Resolve all dependencies for the given root packages
        /// </summary>
        public async Task<DependencyResolution> ResolveAsync(Dictionary<string, string> rootDependencies)
        {
            _resolved.Clear();
            _requests.Clear();

            var errors = new List<string>();
            var warnings = new List<string>();

            // Process root dependencies
            foreach (var (name, constraint) in rootDependencies)
            {
                await ResolvePackageAsync(name, constraint, null, errors, warnings);
            }

            // Check for conflicts
            var conflicts = DetectConflicts();
            foreach (var conflict in conflicts)
            {
                errors.Add(conflict);
            }

            // Build install order (topological sort)
            var installOrder = new List<string>();
            try
            {
                installOrder = TopologicalSort();
            }
            catch (InvalidOperationException ex)
            {
                errors.Add(ex.Message);
            }

            return new DependencyResolution(
                success: errors.Count == 0,
                packages: _resolved.Values.ToList(),
                installOrder: installOrder,
                errors: errors,
                warnings: warnings
            );
        }

        private async Task ResolvePackageAsync(string name, string constraintStr, string? requiredBy,
            List<string> errors, List<string> warnings)
        {
            // Track who requested this package
            if (!_requests.ContainsKey(name))
                _requests[name] = new List<DependencyRequest>();

            _requests[name].Add(new DependencyRequest(constraintStr, requiredBy));

            // Skip if already resolved with compatible version
            if (_resolved.TryGetValue(name, out var existing))
            {
                if (!VersionConstraint.TryParse(constraintStr, out var newConstraint))
                {
                    errors.Add($"Invalid version constraint for {name}: {constraintStr}");
                    return;
                }

                var existingVersion = SemanticVersion.Parse(existing.Version);
                if (newConstraint!.IsSatisfiedBy(existingVersion))
                {
                    return; // Already have compatible version
                }

                // Need to find version that satisfies all constraints
                var allConstraints = _requests[name].Select(r => VersionConstraint.Parse(r.Constraint)).ToList();
                var versions = await _registry.GetVersionsAsync(name);
                var compatibleVersion = FindCompatibleVersion(versions, allConstraints);

                if (compatibleVersion == null)
                {
                    errors.Add($"No version of {name} satisfies all constraints: " +
                        string.Join(", ", _requests[name].Select(r => $"{r.Constraint} (from {r.RequiredBy ?? "root"})")));
                    return;
                }

                // Update to compatible version
                existing.Version = compatibleVersion.ToString();
                return;
            }

            // Resolve new package
            if (!VersionConstraint.TryParse(constraintStr, out var constraint))
            {
                errors.Add($"Invalid version constraint for {name}: {constraintStr}");
                return;
            }

            try
            {
                var versions = await _registry.GetVersionsAsync(name);
                if (!versions.Any())
                {
                    errors.Add($"Package not found: {name}");
                    return;
                }

                var bestVersion = versions.FirstOrDefault(v => constraint!.IsSatisfiedBy(v));
                if (bestVersion == null)
                {
                    errors.Add($"No version of {name} matches {constraintStr}. " +
                        $"Available: {string.Join(", ", versions.Take(5).Select(v => v.ToString()))}");
                    return;
                }

                // Get package info to resolve its dependencies
                var info = await _registry.GetInfoAsync(name);
                var resolved = new ResolvedPackage
                {
                    Name = name,
                    Version = bestVersion.ToString(),
                    Constraint = constraintStr,
                    Dependencies = info?.Dependencies ?? new Dictionary<string, string>(),
                    IsRoot = requiredBy == null
                };

                _resolved[name] = resolved;

                // Recursively resolve dependencies
                foreach (var (depName, depConstraint) in resolved.Dependencies)
                {
                    await ResolvePackageAsync(depName, depConstraint, name, errors, warnings);
                }
            }
            catch (Exception ex)
            {
                errors.Add($"Failed to resolve {name}: {ex.Message}");
            }
        }

        private SemanticVersion? FindCompatibleVersion(List<SemanticVersion> versions,
            List<VersionConstraint> constraints)
        {
            return versions.FirstOrDefault(v => constraints.All(c => c.IsSatisfiedBy(v)));
        }

        private List<string> DetectConflicts()
        {
            var conflicts = new List<string>();

            foreach (var (name, requests) in _requests)
            {
                if (requests.Count <= 1) continue;

                var constraints = requests.Select(r => VersionConstraint.Parse(r.Constraint)).ToList();
                var versions = _resolved.TryGetValue(name, out var pkg)
                    ? new List<SemanticVersion> { SemanticVersion.Parse(pkg.Version) }
                    : new List<SemanticVersion>();

                if (versions.Any())
                {
                    var resolvedVersion = versions[0];
                    var unsatisfied = requests
                        .Where(r => !VersionConstraint.Parse(r.Constraint).IsSatisfiedBy(resolvedVersion))
                        .ToList();

                    if (unsatisfied.Any())
                    {
                        conflicts.Add($"Conflict: {name}@{resolvedVersion} does not satisfy: " +
                            string.Join(", ", unsatisfied.Select(r => $"{r.Constraint} (from {r.RequiredBy ?? "root"})")));
                    }
                }
            }

            return conflicts;
        }

        private List<string> TopologicalSort()
        {
            var result = new List<string>();
            var visited = new HashSet<string>();
            var visiting = new HashSet<string>();

            void Visit(string name)
            {
                if (visited.Contains(name)) return;
                if (visiting.Contains(name))
                {
                    throw new InvalidOperationException($"Circular dependency detected: {name}");
                }

                visiting.Add(name);

                if (_resolved.TryGetValue(name, out var pkg))
                {
                    foreach (var dep in pkg.Dependencies.Keys)
                    {
                        Visit(dep);
                    }
                }

                visiting.Remove(name);
                visited.Add(name);
                result.Add(name);
            }

            foreach (var name in _resolved.Keys)
            {
                Visit(name);
            }

            return result;
        }
    }

    /// <summary>
    /// A dependency request from a package
    /// </summary>
    public class DependencyRequest
    {
        /// <summary>Public API</summary>
        public string Constraint { get; }
        /// <summary>Public API</summary>
        public string? RequiredBy { get; }

        /// <summary>Public API</summary>
        public DependencyRequest(string constraint, string? requiredBy)
        {
            Constraint = constraint;
            RequiredBy = requiredBy;
        }
    }

    /// <summary>
    /// A resolved package with its version and dependencies
    /// </summary>
    public class ResolvedPackage
    {
        /// <summary>Public API</summary>
        public string Name { get; set; } = "";
        /// <summary>Public API</summary>
        public string Version { get; set; } = "";
        /// <summary>Public API</summary>
        public string Constraint { get; set; } = "";
        /// <summary>Public API</summary>
        public Dictionary<string, string> Dependencies { get; set; } = new();
        /// <summary>Public API</summary>
        public bool IsRoot { get; set; }

        /// <summary>Public API</summary>
        public override string ToString() => $"{Name}@{Version}";
    }

    /// <summary>
    /// Result of dependency resolution
    /// </summary>
    public class DependencyResolution
    {
        /// <summary>Public API</summary>
        public bool Success { get; }
        /// <summary>Public API</summary>
        public List<ResolvedPackage> Packages { get; }
        /// <summary>Public API</summary>
        public List<string> InstallOrder { get; }
        /// <summary>Public API</summary>
        public List<string> Errors { get; }
        /// <summary>Public API</summary>
        public List<string> Warnings { get; }

        /// <summary>Public API</summary>
        public DependencyResolution(bool success, List<ResolvedPackage> packages,
            List<string> installOrder, List<string> errors, List<string> warnings)
        {
            Success = success;
            Packages = packages;
            InstallOrder = installOrder;
            Errors = errors;
            Warnings = warnings;
        }

        /// <summary>
        /// Get a package tree representation
        /// </summary>
        public string GetDependencyTree()
        {
            var sb = new System.Text.StringBuilder();
            var roots = Packages.Where(p => p.IsRoot).ToList();

            void PrintTree(ResolvedPackage pkg, string indent, bool isLast)
            {
                var prefix = isLast ? "└── " : "├── ";
                sb.AppendLine($"{indent}{prefix}{pkg.Name}@{pkg.Version}");

                var childIndent = indent + (isLast ? "    " : "│   ");
                var deps = pkg.Dependencies.Keys
                    .Where(d => Packages.Any(p => p.Name == d))
                    .ToList();

                for (int i = 0; i < deps.Count; i++)
                {
                    var dep = Packages.First(p => p.Name == deps[i]);
                    PrintTree(dep, childIndent, i == deps.Count - 1);
                }
            }

            for (int i = 0; i < roots.Count; i++)
            {
                PrintTree(roots[i], "", i == roots.Count - 1);
            }

            return sb.ToString();
        }
    }
}