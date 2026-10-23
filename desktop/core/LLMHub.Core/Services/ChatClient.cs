using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace LLMHub.Services;

public sealed record ChatTurn(string Role, string Content);

public readonly record struct ChatDelta(string? Content, string? Reasoning, double? TokensPerSecond = null);

/// <summary>Streams replies from llama-server's OpenAI-compatible /v1/chat/completions endpoint.</summary>
public static class ChatClient
{
    private static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    public static async IAsyncEnumerable<ChatDelta> StreamAsync(
        string baseUrl, string model, IEnumerable<ChatTurn> turns, [EnumeratorCancellation] CancellationToken ct)
    {
        var body = new
        {
            model,
            stream = true,
            messages = turns.Select(t => new { role = t.Role, content = t.Content }).ToArray(),
        };
        using var request = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions")
        {
            Content = JsonContent.Create(body),
        };
        using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"{(int)response.StatusCode}: {await response.Content.ReadAsStringAsync(ct)}");

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream);
        while (await reader.ReadLineAsync(ct) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal))
                continue;
            var data = line.AsSpan(6).Trim();
            if (data.SequenceEqual("[DONE]"))
                yield break;

            using var doc = JsonDocument.Parse(data.ToString());
            if (doc.RootElement.TryGetProperty("error", out var error))
                throw new HttpRequestException(error.ToString());
            if (doc.RootElement.TryGetProperty("timings", out var timings) && timings.TryGetProperty("predicted_per_second", out var tps) && tps.TryGetDouble(out var speed))
                yield return new ChatDelta(null, null, speed);
            if (!doc.RootElement.TryGetProperty("choices", out var choices) || choices.GetArrayLength() == 0)
                continue;
            if (!choices[0].TryGetProperty("delta", out var delta))
                continue;

            var content = delta.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String ? c.GetString() : null;
            var reasoning = delta.TryGetProperty("reasoning_content", out var r) && r.ValueKind == JsonValueKind.String ? r.GetString() : null;
            if (!string.IsNullOrEmpty(content) || !string.IsNullOrEmpty(reasoning))
                yield return new ChatDelta(content, reasoning);
        }
    }
}
