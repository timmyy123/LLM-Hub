using System.Collections.ObjectModel;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace LLMHub.Services;

public sealed record ApiRequestLog(string Time, string Client, string Path, string Model, string Status, bool Ok);

/// <summary>
/// OpenAI- and Anthropic-compatible HTTP API for the installed models.
/// GGUF models are proxied to llama-server (which speaks both APIs natively, tools included);
/// LiteRT-LM models are translated here.
/// </summary>
public sealed partial class LocalApiServer : ObservableObject
{
    private static readonly HttpClient Proxy = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly SemaphoreSlim _liteRtGate = new(1, 1);
    private readonly SemaphoreSlim _startGate = new(1, 1);
    private readonly bool _bridge;
    private WebApplication? _app;

    /// <param name="agentBridge">
    /// A private instance for the Agent: loopback only, a random port and key of its own, always available even when the
    /// user's Local API is off, plus the /mcp endpoint Claude Code asks for permission through.
    /// </param>
    public LocalApiServer(bool agentBridge = false)
    {
        _bridge = agentBridge;
        Error = "";
    }

    /// <summary>Port and key the server runs with (the settings' for the Local API, random ones for the bridge).</summary>
    public int Port { get; private set; }
    /// <remarks>The Local API reads the setting live, so "New key" applies without a restart.</remarks>
    public string Key => _bridge ? _bridgeKey : AppSettings.Current.ApiKey;
    private string _bridgeKey = "";

    public ObservableCollection<ApiRequestLog> Requests { get; } = [];

    [ObservableProperty] public partial bool IsRunning { get; set; }
    [ObservableProperty] public partial string Error { get; set; }

    public string BaseUrl => $"http://127.0.0.1:{(_bridge ? Port : AppSettings.Current.ApiPort)}";

    public static string NewKey() => "llmhub-" + Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(18));

    /// <summary>The model name clients send: the catalog id when unambiguous ("gemma-4-12b"), else the full id.</summary>
    public static string ApiName(InstalledModel m)
    {
        if (m.CatalogId is null)
            return m.Id;
        var sameCatalog = Ioc.Models.Installed.Count(x => x.CatalogId == m.CatalogId);
        return sameCatalog > 1 ? $"{m.CatalogId}:{m.Quant.ToLowerInvariant().Replace(' ', '-')}" : m.CatalogId;
    }

    public async Task RestartAsync()
    {
        await StopAsync();
        if (AppSettings.Current.ApiEnabled)
            await StartAsync();
    }

    /// <summary>Starts the Agent bridge if it is not running (see the constructor).</summary>
    public async Task EnsureBridgeAsync()
    {
        await _startGate.WaitAsync();
        try
        {
            if (_app is null)
                await StartAsync();
        }
        finally
        {
            _startGate.Release();
        }
        if (_app is null)
            throw new InvalidOperationException(Error);
    }

    private async Task StartAsync()
    {
        var settings = AppSettings.Current;
        if (_bridge)
        {
            Port = FreePort();
            _bridgeKey = NewKey();
        }
        else
        {
            if (string.IsNullOrEmpty(settings.ApiKey))
            {
                settings.ApiKey = NewKey();
                AppSettings.Save();
            }
            Port = settings.ApiPort;
        }
        try
        {
            var builder = WebApplication.CreateEmptyBuilder(new WebApplicationOptions { ContentRootPath = AppContext.BaseDirectory });
            builder.WebHost.UseKestrelCore().ConfigureKestrel(k =>
            {
                if (settings.ApiAllowLan && !_bridge)
                    k.ListenAnyIP(Port);
                else
                    k.ListenLocalhost(Port);
            });
            builder.Services.AddRouting();
            builder.Logging.ClearProviders();
            var app = builder.Build();
            app.UseRouting();
            app.Use(Authorize);
            app.MapGet("/v1/models", ListModels);
            app.MapPost("/v1/chat/completions", ctx => Handle(ctx, Api.OpenAi));
            app.MapPost("/v1/messages", ctx => Handle(ctx, Api.Anthropic));
            app.MapPost("/v1/messages/count_tokens", ctx => Handle(ctx, Api.AnthropicCount));
            app.MapGet("/", () => Results.Text("LLM Hub local API"));
            if (_bridge)
                app.MapPost("/mcp", ctx => Ioc.Agent.Permissions.HandleMcpAsync(ctx));
            await app.StartAsync();
            _app = app;
            UiThread.Run(() =>
            {
                IsRunning = true;
                Error = "";
            });
            Log.Info(_bridge ? $"Agent bridge on port {Port}" : $"Local API on port {Port} (LAN={settings.ApiAllowLan})");
        }
        catch (Exception e)
        {
            Log.Error("Local API failed to start", e);
            UiThread.Run(() =>
            {
                IsRunning = false;
                Error = Loc.F("ApiStartFailed", e.Message);
            });
        }
    }

    public async Task StopAsync()
    {
        if (_app is null)
            return;
        await _app.StopAsync();
        await _app.DisposeAsync();
        _app = null;
        UiThread.Run(() => IsRunning = false);
    }

    private enum Api { OpenAi, Anthropic, AnthropicCount }

    private async Task Authorize(HttpContext ctx, Func<Task> next)
    {
        var guarded = ctx.Request.Path.StartsWithSegments("/v1") || ctx.Request.Path.StartsWithSegments("/mcp");
        if (guarded && (_bridge || AppSettings.Current.ApiRequireKey))
        {
            var bearer = ctx.Request.Headers.Authorization.ToString();
            var key = bearer.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) ? bearer[7..].Trim() : ctx.Request.Headers["x-api-key"].ToString();
            if (!CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(key), Encoding.UTF8.GetBytes(Key)))
            {
                await WriteError(ctx, 401, "authentication_error", "Missing or wrong API key");
                return;
            }
        }
        await next();
    }

    private async Task ListModels(HttpContext ctx)
    {
        var data = new JsonArray(Ioc.Models.Installed.Select(m => (JsonNode)new JsonObject
        {
            ["id"] = ApiName(m),
            ["object"] = "model",
            ["type"] = "model",
            ["owned_by"] = "llm-hub",
            ["display_name"] = m.DisplayName,
            ["created_at"] = m.InstalledAt.ToString("O"),
        }).ToArray());
        await ctx.Response.WriteAsJsonAsync(new JsonObject { ["object"] = "list", ["data"] = data });
        Record(ctx, "", 200);
    }

    private async Task Handle(HttpContext ctx, Api api)
    {
        var body = await JsonNode.ParseAsync(ctx.Request.Body) as JsonObject ?? new JsonObject();
        var requested = body["model"]?.GetValue<string>() ?? "";
        var model = Ioc.Models.Installed.FirstOrDefault(m => ApiName(m) == requested || m.Id == requested || m.CatalogId == requested);
        if (model is null)
        {
            await WriteError(ctx, 404, "not_found_error", $"Model '{requested}' is not installed. GET /v1/models lists what is.");
            Record(ctx, requested, 404);
            return;
        }

        try
        {
            await Ioc.Engines.LoadAsync(model, ctx.RequestAborted, Math.Max(AppSettings.Current.ContextLength, AppSettings.Current.ApiContextLength));
            if (model.Format != "litertlm")
                await ProxyToLlama(ctx, body, model);
            else if (api == Api.AnthropicCount)
                await ctx.Response.WriteAsJsonAsync(new JsonObject { ["input_tokens"] = Turns(body, api).Sum(t => t.Content.Length) / 4 });
            else
                await GenerateLiteRt(ctx, body, model, api);
            Record(ctx, ApiName(model), ctx.Response.StatusCode, body["stream"]?.GetValue<bool>() == true);
        }
        catch (OperationCanceledException)
        {
            Record(ctx, ApiName(model), 499);
        }
        catch (Exception e)
        {
            Log.Error("Local API request failed", e);
            if (!ctx.Response.HasStarted)
                await WriteError(ctx, 500, "api_error", e.Message);
            Record(ctx, ApiName(model), 500);
        }
    }

    private static async Task ProxyToLlama(HttpContext ctx, JsonObject body, InstalledModel model)
    {
        body["model"] = model.Id;
        using var request = new HttpRequestMessage(HttpMethod.Post, Ioc.Engines.Llama.BaseUrl! + ctx.Request.Path)
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };
        using var response = await Proxy.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ctx.RequestAborted);
        ctx.Response.StatusCode = (int)response.StatusCode;
        ctx.Response.ContentType = response.Content.Headers.ContentType?.ToString() ?? "application/json";
        await using var stream = await response.Content.ReadAsStreamAsync(ctx.RequestAborted);
        var buffer = new byte[16 * 1024];
        int read;
        while ((read = await stream.ReadAsync(buffer, ctx.RequestAborted)) > 0)
        {
            await ctx.Response.Body.WriteAsync(buffer.AsMemory(0, read), ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        }
    }

    private async Task GenerateLiteRt(HttpContext ctx, JsonObject body, InstalledModel model, Api api)
    {
        var turns = Turns(body, api);
        var stream = body["stream"]?.GetValue<bool>() == true;
        var name = ApiName(model);
        var id = (api == Api.OpenAi ? "chatcmpl-" : "msg_") + Guid.NewGuid().ToString("N")[..24];
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var text = new StringBuilder();
        var pieces = 0;

        await _liteRtGate.WaitAsync(ctx.RequestAborted);
        try
        {
            if (stream)
            {
                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.Headers.CacheControl = "no-cache";
                if (api == Api.Anthropic)
                {
                    await Sse(ctx, "message_start", new JsonObject
                    {
                        ["type"] = "message_start",
                        ["message"] = new JsonObject
                        {
                            ["id"] = id, ["type"] = "message", ["role"] = "assistant", ["model"] = name,
                            ["content"] = new JsonArray(), ["stop_reason"] = null,
                            ["usage"] = new JsonObject { ["input_tokens"] = 0, ["output_tokens"] = 0 },
                        },
                    });
                    await Sse(ctx, "content_block_start", new JsonObject
                    {
                        ["type"] = "content_block_start", ["index"] = 0,
                        ["content_block"] = new JsonObject { ["type"] = "text", ["text"] = "" },
                    });
                }
            }

            await foreach (var delta in Ioc.Engines.StreamAsync(model, turns, ctx.RequestAborted))
            {
                if (delta.Content is not { Length: > 0 } piece)
                    continue;
                text.Append(piece);
                pieces++;
                if (!stream)
                    continue;
                if (api == Api.OpenAi)
                    await Sse(ctx, null, OpenAiChunk(id, created, name, new JsonObject { ["content"] = piece }, null));
                else
                    await Sse(ctx, "content_block_delta", new JsonObject
                    {
                        ["type"] = "content_block_delta", ["index"] = 0,
                        ["delta"] = new JsonObject { ["type"] = "text_delta", ["text"] = piece },
                    });
            }
        }
        finally
        {
            _liteRtGate.Release();
        }

        if (stream && api == Api.OpenAi)
        {
            await Sse(ctx, null, OpenAiChunk(id, created, name, new JsonObject(), "stop"));
            await ctx.Response.WriteAsync("data: [DONE]\n\n");
        }
        else if (stream)
        {
            await Sse(ctx, "content_block_stop", new JsonObject { ["type"] = "content_block_stop", ["index"] = 0 });
            await Sse(ctx, "message_delta", new JsonObject
            {
                ["type"] = "message_delta",
                ["delta"] = new JsonObject { ["stop_reason"] = "end_turn", ["stop_sequence"] = null },
                ["usage"] = new JsonObject { ["output_tokens"] = pieces },
            });
            await Sse(ctx, "message_stop", new JsonObject { ["type"] = "message_stop" });
        }
        else if (api == Api.OpenAi)
        {
            await ctx.Response.WriteAsJsonAsync(new JsonObject
            {
                ["id"] = id, ["object"] = "chat.completion", ["created"] = created, ["model"] = name,
                ["choices"] = new JsonArray(new JsonObject
                {
                    ["index"] = 0, ["finish_reason"] = "stop",
                    ["message"] = new JsonObject { ["role"] = "assistant", ["content"] = text.ToString() },
                }),
                ["usage"] = new JsonObject { ["prompt_tokens"] = 0, ["completion_tokens"] = pieces, ["total_tokens"] = pieces },
            });
        }
        else
        {
            await ctx.Response.WriteAsJsonAsync(new JsonObject
            {
                ["id"] = id, ["type"] = "message", ["role"] = "assistant", ["model"] = name,
                ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = text.ToString() }),
                ["stop_reason"] = "end_turn", ["stop_sequence"] = null,
                ["usage"] = new JsonObject { ["input_tokens"] = 0, ["output_tokens"] = pieces },
            });
        }
    }

    /// <summary>Flattens OpenAI or Anthropic messages (string or content-part arrays, plus Anthropic's top-level system) to text turns.</summary>
    private static List<ChatTurn> Turns(JsonObject body, Api api)
    {
        static string Flatten(JsonNode? content) => content switch
        {
            JsonValue v => v.GetValue<string>(),
            JsonArray parts => string.Concat(parts.Select(p => p?["text"]?.GetValue<string>() ?? "")),
            _ => "",
        };
        var turns = new List<ChatTurn>();
        if (api != Api.OpenAi && body["system"] is { } system && Flatten(system) is { Length: > 0 } sys)
            turns.Add(new ChatTurn("system", sys));
        foreach (var m in body["messages"] as JsonArray ?? [])
            turns.Add(new ChatTurn(m?["role"]?.GetValue<string>() ?? "user", Flatten(m?["content"])));
        return turns;
    }

    private static JsonObject OpenAiChunk(string id, long created, string model, JsonObject delta, string? finish) => new()
    {
        ["id"] = id, ["object"] = "chat.completion.chunk", ["created"] = created, ["model"] = model,
        ["choices"] = new JsonArray(new JsonObject { ["index"] = 0, ["delta"] = delta, ["finish_reason"] = finish }),
    };

    private static async Task Sse(HttpContext ctx, string? eventName, JsonObject data)
    {
        var text = (eventName is null ? "" : $"event: {eventName}\n") + $"data: {data.ToJsonString()}\n\n";
        await ctx.Response.WriteAsync(text, ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
    }

    private static async Task WriteError(HttpContext ctx, int status, string type, string message)
    {
        ctx.Response.StatusCode = status;
        // Both OpenAI ("error.message") and Anthropic ("type":"error") clients can read this shape.
        await ctx.Response.WriteAsJsonAsync(new JsonObject
        {
            ["type"] = "error",
            ["error"] = new JsonObject { ["type"] = type, ["message"] = message },
        });
    }

    private void Record(HttpContext ctx, string model, int status, bool streaming = false)
    {
        var agent = ctx.Request.Headers.UserAgent.ToString();
        var client = agent.Length == 0 ? "-" : agent.Split(' ')[0].Split('/')[0];
        var entry = new ApiRequestLog(DateTime.Now.ToString("HH:mm:ss"), client, ctx.Request.Path, model,
            streaming && status == 200 ? Loc.S("ApiStreaming") : status.ToString(), status is >= 200 and < 300);
        UiThread.Run(() =>
        {
            Requests.Insert(0, entry);
            while (Requests.Count > 50)
                Requests.RemoveAt(Requests.Count - 1);
        });
    }

    private static int FreePort()
    {
        var listener = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}
