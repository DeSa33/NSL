using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NSL.StandardLib.Http
{
    /// <summary>
    /// HTTP client for NSL - simple and intuitive API for web requests.
    ///
    /// Usage in NSL:
    /// <code>
    /// import http
    ///
    /// // Simple GET request
    /// let response = http.get("https://api.example.com/data")
    ///
    /// // POST with JSON body
    /// let result = http.post("https://api.example.com/submit", {
    ///     "name": "Alice",
    ///     "value": 42
    /// })
    ///
    /// // With headers
    /// let auth_response = http.get("https://api.example.com/private", {
    ///     headers: { "Authorization": "Bearer token123" }
    /// })
    /// </code>
    /// </summary>
    public class NslHttpClient : IDisposable
    {
        private readonly HttpClient _client;
        private readonly JsonSerializerOptions _jsonOptions;
        private bool _disposed;

        /// <summary>
        /// Default timeout in seconds
        /// </summary>
        public int TimeoutSeconds { get; set; } = 30;

        /// <summary>
        /// Default headers applied to all requests
        /// </summary>
        public Dictionary<string, string> DefaultHeaders { get; } = new();

        /// <summary>
        /// Base URL for relative requests
        /// </summary>
        public string? BaseUrl { get; set; }

        /// <summary>
        /// Create a new HTTP client
        /// </summary>
        public NslHttpClient()
        {
            _client = new HttpClient();
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = false
            };
        }

        /// <summary>
        /// Create an HTTP client with base URL
        /// </summary>
        public NslHttpClient(string baseUrl) : this()
        {
            BaseUrl = baseUrl;
        }

        #region HTTP Methods

        /// <summary>
        /// Perform a GET request
        /// </summary>
        public async Task<HttpResponse> GetAsync(string url, HttpRequestOptions? options = null)
        {
            return await SendAsync(HttpMethod.Get, url, null, options);
        }

        /// <summary>
        /// Perform a POST request
        /// </summary>
        public async Task<HttpResponse> PostAsync(string url, object? body = null, HttpRequestOptions? options = null)
        {
            return await SendAsync(HttpMethod.Post, url, body, options);
        }

        /// <summary>
        /// Perform a PUT request
        /// </summary>
        public async Task<HttpResponse> PutAsync(string url, object? body = null, HttpRequestOptions? options = null)
        {
            return await SendAsync(HttpMethod.Put, url, body, options);
        }

        /// <summary>
        /// Perform a PATCH request
        /// </summary>
        public async Task<HttpResponse> PatchAsync(string url, object? body = null, HttpRequestOptions? options = null)
        {
            return await SendAsync(HttpMethod.Patch, url, body, options);
        }

        /// <summary>
        /// Perform a DELETE request
        /// </summary>
        public async Task<HttpResponse> DeleteAsync(string url, HttpRequestOptions? options = null)
        {
            return await SendAsync(HttpMethod.Delete, url, null, options);
        }

        /// <summary>
        /// Perform a HEAD request
        /// </summary>
        public async Task<HttpResponse> HeadAsync(string url, HttpRequestOptions? options = null)
        {
            return await SendAsync(HttpMethod.Head, url, null, options);
        }

        /// <summary>
        /// Download a file
        /// </summary>
        public async Task<byte[]> DownloadAsync(string url, HttpRequestOptions? options = null)
        {
            var request = CreateRequest(HttpMethod.Get, url, null, options);

            var response = await _client.SendAsync(request);
            response.EnsureSuccessStatusCode();

            return await response.Content.ReadAsByteArrayAsync();
        }

        /// <summary>
        /// Download a file to disk
        /// </summary>
        public async Task DownloadToFileAsync(string url, string filePath, HttpRequestOptions? options = null)
        {
            var bytes = await DownloadAsync(url, options);
            await File.WriteAllBytesAsync(filePath, bytes);
        }

        #endregion

        #region Core Methods

        /// <summary>
        /// Send an HTTP request
        /// </summary>
        public async Task<HttpResponse> SendAsync(HttpMethod method, string url, object? body, HttpRequestOptions? options)
        {
            var request = CreateRequest(method, url, body, options);

            try
            {
                var response = await _client.SendAsync(request);
                return await CreateResponse(response);
            }
            catch (HttpRequestException ex)
            {
                return new HttpResponse
                {
                    Success = false,
                    StatusCode = 0,
                    Error = ex.Message
                };
            }
            catch (TaskCanceledException)
            {
                return new HttpResponse
                {
                    Success = false,
                    StatusCode = 0,
                    Error = "Request timed out"
                };
            }
        }

        private HttpRequestMessage CreateRequest(HttpMethod method, string url, object? body, HttpRequestOptions? options)
        {
            // Build URL
            var fullUrl = BaseUrl != null && !url.StartsWith("http")
                ? $"{BaseUrl.TrimEnd('/')}/{url.TrimStart('/')}"
                : url;

            // Add query parameters
            if (options?.Query != null && options.Query.Count > 0)
            {
                var queryString = string.Join("&",
                    options.Query.Select(kv => $"{Uri.EscapeDataString(kv.Key)}={Uri.EscapeDataString(kv.Value)}"));
                fullUrl += (fullUrl.Contains('?') ? "&" : "?") + queryString;
            }

            var request = new HttpRequestMessage(method, fullUrl);

            // Add default headers
            foreach (var (key, value) in DefaultHeaders)
            {
                request.Headers.TryAddWithoutValidation(key, value);
            }

            // Add request-specific headers
            if (options?.Headers != null)
            {
                foreach (var (key, value) in options.Headers)
                {
                    request.Headers.TryAddWithoutValidation(key, value);
                }
            }

            // Add body
            if (body != null)
            {
                var contentType = options?.ContentType ?? "application/json";

                if (body is string stringBody)
                {
                    request.Content = new StringContent(stringBody, Encoding.UTF8, contentType);
                }
                else if (body is byte[] byteBody)
                {
                    request.Content = new ByteArrayContent(byteBody);
                    request.Content.Headers.ContentType = new MediaTypeHeaderValue(contentType);
                }
                else if (body is Dictionary<string, string> formBody && contentType == "application/x-www-form-urlencoded")
                {
                    request.Content = new FormUrlEncodedContent(formBody);
                }
                else
                {
                    // Serialize as JSON
                    var json = JsonSerializer.Serialize(body, _jsonOptions);
                    request.Content = new StringContent(json, Encoding.UTF8, "application/json");
                }
            }

            // Set timeout
            var timeout = options?.TimeoutSeconds ?? TimeoutSeconds;
            _client.Timeout = TimeSpan.FromSeconds(timeout);

            return request;
        }

        private async Task<HttpResponse> CreateResponse(HttpResponseMessage response)
        {
            var body = await response.Content.ReadAsStringAsync();

            var headers = new Dictionary<string, string>();
            foreach (var header in response.Headers)
            {
                headers[header.Key] = string.Join(", ", header.Value);
            }
            foreach (var header in response.Content.Headers)
            {
                headers[header.Key] = string.Join(", ", header.Value);
            }

            return new HttpResponse
            {
                Success = response.IsSuccessStatusCode,
                StatusCode = (int)response.StatusCode,
                StatusText = response.ReasonPhrase ?? "",
                Headers = headers,
                Body = body,
                ContentType = response.Content.Headers.ContentType?.MediaType ?? ""
            };
        }

        #endregion

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _client.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// HTTP request options
    /// </summary>
    public class HttpRequestOptions
    {
        /// <summary>Request headers</summary>
        public Dictionary<string, string>? Headers { get; init; }

        /// <summary>Query parameters</summary>
        public Dictionary<string, string>? Query { get; init; }

        /// <summary>Request timeout in seconds</summary>
        public int? TimeoutSeconds { get; init; }

        /// <summary>Content type for the request body</summary>
        public string? ContentType { get; init; }
    }

    /// <summary>
    /// HTTP response
    /// </summary>
    public class HttpResponse
    {
        /// <summary>Whether the request succeeded (2xx status)</summary>
        public bool Success { get; init; }

        /// <summary>HTTP status code</summary>
        public int StatusCode { get; init; }

        /// <summary>Status text (e.g., "OK", "Not Found")</summary>
        public string StatusText { get; init; } = "";

        /// <summary>Response headers</summary>
        public Dictionary<string, string> Headers { get; init; } = new();

        /// <summary>Response body as string</summary>
        public string Body { get; init; } = "";

        /// <summary>Content type of the response</summary>
        public string ContentType { get; init; } = "";

        /// <summary>Error message if request failed</summary>
        public string? Error { get; init; }

        /// <summary>
        /// Parse the body as JSON
        /// </summary>
        public T? Json<T>()
        {
            if (string.IsNullOrEmpty(Body))
                return default;

            return JsonSerializer.Deserialize<T>(Body);
        }

        /// <summary>
        /// Parse the body as a dictionary
        /// </summary>
        public Dictionary<string, object>? JsonObject()
        {
            return Json<Dictionary<string, object>>();
        }

        /// <summary>
        /// Get the body as bytes
        /// </summary>
        public byte[] Bytes()
        {
            return Encoding.UTF8.GetBytes(Body);
        }

        /// <summary>Public API</summary>
        public override string ToString()
        {
            return $"HttpResponse {{ StatusCode: {StatusCode}, Success: {Success}, ContentLength: {Body.Length} }}";
        }
    }
}