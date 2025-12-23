using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NSL.StandardLib.Json
{
    /// <summary>
    /// JSON parsing and serialization for NSL.
    ///
    /// Usage in NSL:
    /// <code>
    /// import json
    ///
    /// // Parse JSON string
    /// let data = json.parse('{"name": "Alice", "age": 30}')
    /// println(data["name"])  // Alice
    ///
    /// // Create JSON from object
    /// let obj = { name: "Bob", scores: [95, 87, 92] }
    /// let str = json.stringify(obj, pretty=true)
    ///
    /// // Load from file
    /// let config = json.load("config.json")
    ///
    /// // Save to file
    /// json.save("output.json", data)
    /// </code>
    /// </summary>
    public static class JsonModule
    {
        private static readonly JsonSerializerOptions DefaultOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            WriteIndented = false,
            AllowTrailingCommas = true,
            ReadCommentHandling = JsonCommentHandling.Skip
        };

        private static readonly JsonSerializerOptions PrettyOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            WriteIndented = true,
            AllowTrailingCommas = true,
            ReadCommentHandling = JsonCommentHandling.Skip
        };

        #region Parsing

        /// <summary>
        /// Parse a JSON string into an object
        /// </summary>
        public static object? Parse(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                return null;

            var node = JsonNode.Parse(json);
            return ConvertJsonNode(node);
        }

        /// <summary>
        /// Parse JSON into a specific type
        /// </summary>
        public static T? Parse<T>(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                return default;

            return JsonSerializer.Deserialize<T>(json, DefaultOptions);
        }

        /// <summary>
        /// Try to parse JSON, returning success status
        /// </summary>
        public static bool TryParse(string json, out object? result)
        {
            try
            {
                result = Parse(json);
                return true;
            }
            catch
            {
                result = null;
                return false;
            }
        }

        /// <summary>
        /// Load JSON from a file
        /// </summary>
        public static object? Load(string filePath)
        {
            var json = File.ReadAllText(filePath);
            return Parse(json);
        }

        /// <summary>
        /// Load JSON from a file into a specific type
        /// </summary>
        public static T? Load<T>(string filePath)
        {
            var json = File.ReadAllText(filePath);
            return Parse<T>(json);
        }

        /// <summary>
        /// Asynchronously load JSON from a file
        /// </summary>
        public static async Task<object?> LoadAsync(string filePath)
        {
            var json = await File.ReadAllTextAsync(filePath);
            return Parse(json);
        }

        #endregion

        #region Serialization

        /// <summary>
        /// Convert an object to JSON string
        /// </summary>
        public static string Stringify(object? value, bool pretty = false)
        {
            if (value == null)
                return "null";

            var options = pretty ? PrettyOptions : DefaultOptions;
            return JsonSerializer.Serialize(value, options);
        }

        /// <summary>
        /// Save an object as JSON to a file
        /// </summary>
        public static void Save(string filePath, object? value, bool pretty = true)
        {
            var json = Stringify(value, pretty);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Asynchronously save an object as JSON to a file
        /// </summary>
        public static async Task SaveAsync(string filePath, object? value, bool pretty = true)
        {
            var json = Stringify(value, pretty);
            await File.WriteAllTextAsync(filePath, json);
        }

        #endregion

        #region Querying

        /// <summary>
        /// Get a value from a JSON object using a path (e.g., "user.address.city")
        /// </summary>
        public static object? Get(object? obj, string path)
        {
            if (obj == null || string.IsNullOrEmpty(path))
                return null;

            var parts = path.Split('.');
            var current = obj;

            foreach (var part in parts)
            {
                if (current == null)
                    return null;

                // Handle array index
                if (part.Contains('[') && part.Contains(']'))
                {
                    var bracketStart = part.IndexOf('[');
                    var key = part[..bracketStart];
                    var indexStr = part[(bracketStart + 1)..^1];

                    if (!int.TryParse(indexStr, out var index))
                        return null;

                    if (!string.IsNullOrEmpty(key))
                    {
                        current = GetProperty(current, key);
                        if (current == null)
                            return null;
                    }

                    current = GetAtIndex(current, index);
                }
                else
                {
                    current = GetProperty(current, part);
                }
            }

            return current;
        }

        /// <summary>
        /// Set a value in a JSON object using a path
        /// </summary>
        public static void Set(Dictionary<string, object?> obj, string path, object? value)
        {
            var parts = path.Split('.');
            var current = (object)obj;

            for (int i = 0; i < parts.Length - 1; i++)
            {
                var nextObj = GetProperty(current, parts[i]);
                if (nextObj == null)
                {
                    nextObj = new Dictionary<string, object?>();
                    if (current is Dictionary<string, object?> dict)
                        dict[parts[i]] = nextObj;
                }
                current = nextObj;
            }

            if (current is Dictionary<string, object?> targetDict)
            {
                targetDict[parts[^1]] = value;
            }
        }

        /// <summary>
        /// Check if a path exists in a JSON object
        /// </summary>
        public static bool Has(object? obj, string path)
        {
            return Get(obj, path) != null;
        }

        /// <summary>
        /// Get keys of a JSON object
        /// </summary>
        public static string[] Keys(object? obj)
        {
            if (obj is Dictionary<string, object?> dict)
                return dict.Keys.ToArray();

            if (obj is JsonObject jsonObj)
                return jsonObj.Select(p => p.Key).ToArray();

            return Array.Empty<string>();
        }

        /// <summary>
        /// Get values of a JSON object
        /// </summary>
        public static object?[] Values(object? obj)
        {
            if (obj is Dictionary<string, object?> dict)
                return dict.Values.ToArray();

            if (obj is JsonObject jsonObj)
                return jsonObj.Select(p => ConvertJsonNode(p.Value)).ToArray();

            return Array.Empty<object>();
        }

        #endregion

        #region Validation

        /// <summary>
        /// Check if a string is valid JSON
        /// </summary>
        public static bool IsValid(string json)
        {
            try
            {
                JsonDocument.Parse(json);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Validate JSON and return errors
        /// </summary>
        public static (bool valid, string? error) Validate(string json)
        {
            try
            {
                JsonDocument.Parse(json);
                return (true, null);
            }
            catch (JsonException ex)
            {
                return (false, ex.Message);
            }
        }

        #endregion

        #region Utilities

        /// <summary>
        /// Merge two JSON objects
        /// </summary>
        public static Dictionary<string, object?> Merge(Dictionary<string, object?> target, Dictionary<string, object?> source, bool deep = true)
        {
            var result = new Dictionary<string, object?>(target);

            foreach (var (key, value) in source)
            {
                if (deep && result.TryGetValue(key, out var existing) &&
                    existing is Dictionary<string, object?> existingDict &&
                    value is Dictionary<string, object?> valueDict)
                {
                    result[key] = Merge(existingDict, valueDict, true);
                }
                else
                {
                    result[key] = value;
                }
            }

            return result;
        }

        /// <summary>
        /// Deep clone a JSON object
        /// </summary>
        public static object? Clone(object? obj)
        {
            if (obj == null)
                return null;

            var json = Stringify(obj);
            return Parse(json);
        }

        /// <summary>
        /// Compare two JSON values for equality
        /// </summary>
        public static bool Equals(object? a, object? b)
        {
            var jsonA = Stringify(a);
            var jsonB = Stringify(b);
            return jsonA == jsonB;
        }

        #endregion

        #region Private Helpers

        private static object? ConvertJsonNode(JsonNode? node)
        {
            if (node == null)
                return null;

            return node switch
            {
                JsonValue value => ConvertJsonValue(value),
                JsonArray array => array.Select(ConvertJsonNode).ToList(),
                JsonObject obj => obj.ToDictionary(
                    kv => kv.Key,
                    kv => ConvertJsonNode(kv.Value)),
                _ => null
            };
        }

        private static object? ConvertJsonValue(JsonValue value)
        {
            var element = value.GetValue<JsonElement>();

            return element.ValueKind switch
            {
                JsonValueKind.String => element.GetString(),
                JsonValueKind.Number when element.TryGetInt64(out var l) => l,
                JsonValueKind.Number => element.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                _ => element.ToString()
            };
        }

        private static object? GetProperty(object? obj, string key)
        {
            if (obj is Dictionary<string, object?> dict)
                return dict.TryGetValue(key, out var val) ? val : null;

            if (obj is JsonObject jsonObj)
                return jsonObj.TryGetPropertyValue(key, out var node) ? ConvertJsonNode(node) : null;

            // Try reflection for other types
            var prop = obj?.GetType().GetProperty(key);
            return prop?.GetValue(obj);
        }

        private static object? GetAtIndex(object? obj, int index)
        {
            if (obj is IList<object?> list)
                return index >= 0 && index < list.Count ? list[index] : null;

            if (obj is JsonArray array)
                return index >= 0 && index < array.Count ? ConvertJsonNode(array[index]) : null;

            return null;
        }

        #endregion
    }
}
