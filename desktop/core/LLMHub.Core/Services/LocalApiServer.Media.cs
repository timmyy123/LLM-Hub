using System.Text;
using System.Text.Json.Nodes;
using LLMHub.Services.Studio;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace LLMHub.Services;

/// <summary>
/// The OpenAI endpoints beyond chat, as listed on the design's Local API page: /v1/responses (any chat model),
/// /v1/embeddings (EmbeddingGemma), /v1/audio/transcriptions (Whisper), /v1/audio/speech (Supertonic) and
/// /v1/images/generations (ComfyUI). Each one uses the same engine the app itself uses for that feature.
/// </summary>
/// <summary>One row of the Local API page's endpoint table: method, path and what serves it.</summary>
public sealed record ApiEndpoint(string Method, string Path, string ServedBy, bool Anthropic);

public sealed partial class LocalApiServer
{
    /// <summary>Every route this server answers, in the order the design lists them.</summary>
    public static IReadOnlyList<ApiEndpoint> Endpoints { get; } =
    [
        new("POST", "/v1/chat/completions", Loc.S("ApiChatFeatures"), false),
        new("POST", "/v1/responses", "", false),
        new("POST", "/v1/embeddings", "EmbeddingGemma", false),
        new("POST", "/v1/audio/transcriptions", "Whisper", false),
        new("POST", "/v1/audio/speech", "Supertonic", false),
        new("POST", "/v1/images/generations", ComfyService.IsSupported || !Studio.Npu.SdNpuService.IsSupported ? "ComfyUI" : "Stable Diffusion 1.5 · NPU", false),
        new("GET", "/v1/models", "", false),
        new("POST", "/v1/messages", Loc.S("ApiChatFeatures"), true),
        new("POST", "/v1/messages/count_tokens", "", true),
        new("GET", "/v1/models", "", true),
    ];

    /// <summary>"Never", "1 min", "10 min" for the Unload after idle picker; values in minutes.</summary>
    public static IReadOnlyList<(int Minutes, string Label)> UnloadChoices { get; } =
        [(10, Loc.F("ApiMinutes", "10")), (0, Loc.S("ApiNever")), (1, Loc.F("ApiMinutes", "1")), (30, Loc.F("ApiMinutes", "30"))];

    /// <summary>"llama.cpp", "LiteRT-LM" or "llama.cpp · Embedding" beside each model the API offers.</summary>
    public static string EngineLabel(InstalledModel m) =>
        m.Format == "litertlm" ? "LiteRT-LM" : m.IsEmbedding ? "llama.cpp · " + Loc.S("TagEmbedding") : "llama.cpp";

    private readonly SemaphoreSlim _mediaGate = new(1, 1);

    /// <summary>OpenAI voice names mapped onto Supertonic's four voices, so clients written for OpenAI keep working.</summary>
    private static readonly Dictionary<string, string> VoiceAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["alloy"] = "F1", ["nova"] = "F1", ["shimmer"] = "F2", ["coral"] = "F2", ["sage"] = "F2",
        ["echo"] = "M1", ["onyx"] = "M1", ["fable"] = "M2", ["ash"] = "M2", ["ballad"] = "M2", ["verse"] = "M1",
    };

    private void MapMediaRoutes(WebApplication app)
    {
        app.MapPost("/v1/responses", Responses);
        app.MapPost("/v1/embeddings", Embeddings);
        app.MapPost("/v1/audio/transcriptions", Transcriptions);
        app.MapPost("/v1/audio/speech", SpeechEndpoint);
        app.MapPost("/v1/images/generations", Images);
    }

    // ---------------------------------------------------------------- /v1/responses

    /// <summary>OpenAI's Responses API on any installed chat model (text in, text out; streaming as response.* events).</summary>
    private async Task Responses(HttpContext ctx)
    {
        var body = await JsonNode.ParseAsync(ctx.Request.Body) as JsonObject ?? new JsonObject();
        var requested = body["model"]?.GetValue<string>() ?? "";
        if (FindChatModel(requested) is not { } model)
        {
            await WriteError(ctx, 404, "not_found_error", $"Model '{requested}' is not installed. GET /v1/models lists what is.");
            Record(ctx, requested, 404);
            return;
        }
        var turns = new List<ChatTurn>();
        if (body["instructions"]?.GetValue<string>() is { Length: > 0 } instructions)
            turns.Add(new ChatTurn("system", instructions));
        switch (body["input"])
        {
            case JsonValue v:
                turns.Add(new ChatTurn("user", v.GetValue<string>()));
                break;
            case JsonArray items:
                foreach (var item in items)
                {
                    var role = item?["role"]?.GetValue<string>() ?? "user";
                    var content = item?["content"] switch
                    {
                        JsonValue c => c.GetValue<string>(),
                        JsonArray parts => string.Concat(parts.Select(p => p?["text"]?.GetValue<string>() ?? "")),
                        _ => "",
                    };
                    turns.Add(new ChatTurn(role == "developer" ? "system" : role, content));
                }
                break;
        }
        var stream = body["stream"]?.GetValue<bool>() == true;
        var name = ApiName(model);
        var id = "resp_" + Guid.NewGuid().ToString("N")[..24];
        var messageId = "msg_" + Guid.NewGuid().ToString("N")[..24];
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        JsonObject Envelope(string status, string text) => new()
        {
            ["id"] = id, ["object"] = "response", ["created_at"] = created, ["status"] = status, ["model"] = name,
            ["output"] = new JsonArray(new JsonObject
            {
                ["type"] = "message", ["id"] = messageId, ["status"] = status, ["role"] = "assistant",
                ["content"] = new JsonArray(new JsonObject { ["type"] = "output_text", ["text"] = text, ["annotations"] = new JsonArray() }),
            }),
            ["output_text"] = text,
        };

        try
        {
            await Ioc.Engines.LoadAsync(model, ctx.RequestAborted, Math.Max(AppSettings.Current.ContextLength, AppSettings.Current.ApiContextLength));
            MarkApiUse();
            var text = new StringBuilder();
            var splitter = new ThinkSplitter();
            if (stream)
            {
                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.Headers.CacheControl = "no-cache";
                await Sse(ctx, "response.created", new JsonObject { ["type"] = "response.created", ["response"] = Envelope("in_progress", "") });
            }
            await _liteRtGate.WaitAsync(ctx.RequestAborted);
            try
            {
                await foreach (var delta in Ioc.Engines.StreamAsync(model, turns, ctx.RequestAborted))
                {
                    if (delta.Content is not { Length: > 0 } raw)
                        continue;
                    var (piece, _) = splitter.Push(raw);
                    if (piece.Length == 0)
                        continue;
                    text.Append(piece);
                    if (stream)
                        await Sse(ctx, "response.output_text.delta", new JsonObject
                        {
                            ["type"] = "response.output_text.delta", ["item_id"] = messageId, ["output_index"] = 0, ["content_index"] = 0, ["delta"] = piece,
                        });
                }
            }
            finally
            {
                _liteRtGate.Release();
            }
            if (stream)
            {
                await Sse(ctx, "response.output_text.done", new JsonObject
                {
                    ["type"] = "response.output_text.done", ["item_id"] = messageId, ["output_index"] = 0, ["content_index"] = 0, ["text"] = text.ToString(),
                });
                await Sse(ctx, "response.completed", new JsonObject { ["type"] = "response.completed", ["response"] = Envelope("completed", text.ToString()) });
            }
            else
                await ctx.Response.WriteAsJsonAsync(Envelope("completed", text.ToString()));
            Record(ctx, name, 200, stream);
        }
        catch (OperationCanceledException)
        {
            Record(ctx, name, 499);
        }
        catch (Exception e)
        {
            Log.Error("Local API /v1/responses failed", e);
            if (!ctx.Response.HasStarted)
                await WriteError(ctx, 500, "api_error", e.Message);
            Record(ctx, name, 500);
        }
    }

    // ---------------------------------------------------------------- /v1/embeddings

    /// <summary>Embeddings from an installed embedding model (EmbeddingGemma), on its own small llama-server.</summary>
    private async Task Embeddings(HttpContext ctx)
    {
        var body = await JsonNode.ParseAsync(ctx.Request.Body) as JsonObject ?? new JsonObject();
        var requested = body["model"]?.GetValue<string>() ?? "";
        // EmbeddingGemma 2 first when it is installed alongside the original EmbeddingGemma.
        var embedders = Ioc.Models.Installed.Where(m => m.IsEmbedding && m.Format == "gguf").OrderByDescending(m => m.CatalogId == "embeddinggemma-2").ToList();
        var model = embedders.FirstOrDefault(m => ApiName(m) == requested || m.Id == requested || m.CatalogId == requested) ?? embedders.FirstOrDefault();
        if (model is null)
        {
            await WriteError(ctx, 404, "not_found_error", "No embedding model is installed. Download EmbeddingGemma 2 in Settings › Models.");
            Record(ctx, requested, 404);
            return;
        }
        try
        {
            await Ioc.Embeddings.LoadAsync(model, ctx.RequestAborted, Math.Clamp(model.ContextLength, 512, 8192));
            body["model"] = model.Id;
            using var request = new HttpRequestMessage(HttpMethod.Post, Ioc.Embeddings.BaseUrl + "/v1/embeddings")
            {
                Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
            };
            using var response = await Proxy.SendAsync(request, ctx.RequestAborted);
            var json = await JsonNode.ParseAsync(await response.Content.ReadAsStreamAsync(ctx.RequestAborted)) as JsonObject;
            if (json is not null)
                json["model"] = ApiName(model);
            ctx.Response.StatusCode = (int)response.StatusCode;
            await ctx.Response.WriteAsJsonAsync(json);
            Record(ctx, ApiName(model), (int)response.StatusCode);
        }
        catch (OperationCanceledException)
        {
            Record(ctx, ApiName(model), 499);
        }
        catch (Exception e)
        {
            Log.Error("Local API /v1/embeddings failed", e);
            if (!ctx.Response.HasStarted)
                await WriteError(ctx, 500, "api_error", e.Message);
            Record(ctx, ApiName(model), 500);
        }
    }

    // ---------------------------------------------------------------- /v1/audio/transcriptions

    /// <summary>Speech to text with Whisper (multipart upload as in OpenAI's API; json, text, srt-free formats).</summary>
    private async Task Transcriptions(HttpContext ctx)
    {
        if (!ctx.Request.HasFormContentType)
        {
            await WriteError(ctx, 400, "invalid_request_error", "Send the audio as multipart/form-data with a 'file' field.");
            Record(ctx, "whisper", 400);
            return;
        }
        var form = await ctx.Request.ReadFormAsync(ctx.RequestAborted);
        if (form.Files.GetFile("file") is not { Length: > 0 } file)
        {
            await WriteError(ctx, 400, "invalid_request_error", "The 'file' field is missing or empty.");
            Record(ctx, "whisper", 400);
            return;
        }
        var path = Path.Combine(Path.GetTempPath(), $"llmhub-api-{Guid.NewGuid():N}{Path.GetExtension(file.FileName)}");
        try
        {
            await using (var target = File.Create(path))
                await file.CopyToAsync(target, ctx.RequestAborted);
            var language = form["language"].ToString();
            await _mediaGate.WaitAsync(ctx.RequestAborted);
            string text;
            try
            {
                text = await Ioc.Speech.TranscribeAsync(path, language.Length > 0 ? language : null, ctx.RequestAborted);
            }
            finally
            {
                _mediaGate.Release();
            }
            if (form["response_format"].ToString() == "text")
                await ctx.Response.WriteAsync(text, ctx.RequestAborted);
            else
                await ctx.Response.WriteAsJsonAsync(new JsonObject { ["text"] = text });
            Record(ctx, "whisper", 200);
        }
        catch (OperationCanceledException)
        {
            Record(ctx, "whisper", 499);
        }
        catch (Exception e)
        {
            Log.Error("Local API /v1/audio/transcriptions failed", e);
            if (!ctx.Response.HasStarted)
                await WriteError(ctx, 500, "api_error", e.Message);
            Record(ctx, "whisper", 500);
        }
        finally
        {
            try { File.Delete(path); } catch (IOException) { }
        }
    }

    // ---------------------------------------------------------------- /v1/audio/speech

    /// <summary>Text to speech with Supertonic. Returns WAV (44.1 kHz) or raw 16-bit PCM.</summary>
    private async Task SpeechEndpoint(HttpContext ctx)
    {
        var body = await JsonNode.ParseAsync(ctx.Request.Body) as JsonObject ?? new JsonObject();
        var input = body["input"]?.GetValue<string>() ?? "";
        var format = body["response_format"]?.GetValue<string>() ?? "wav";
        if (input.Trim().Length == 0 || format is not ("wav" or "pcm"))
        {
            await WriteError(ctx, 400, "invalid_request_error", input.Trim().Length == 0 ? "'input' is empty." : "response_format must be wav or pcm.");
            Record(ctx, "supertonic", 400);
            return;
        }
        var requested = body["voice"]?.GetValue<string>() ?? AppSettings.Current.Voice;
        var voice = LLMHub.Services.Speech.SpeechService.Voices.Contains(requested) ? requested : VoiceAliases.GetValueOrDefault(requested, AppSettings.Current.Voice);
        try
        {
            await _mediaGate.WaitAsync(ctx.RequestAborted);
            string path;
            try
            {
                (path, _) = await Ioc.Speech.SynthesizeAsync(input, voice, null, ctx.RequestAborted);
            }
            finally
            {
                _mediaGate.Release();
            }
            var bytes = await File.ReadAllBytesAsync(path, ctx.RequestAborted);
            File.Delete(path);
            ctx.Response.ContentType = format == "pcm" ? "audio/pcm" : "audio/wav";
            // PCM: the samples without the 44-byte RIFF header.
            await ctx.Response.Body.WriteAsync(format == "pcm" && bytes.Length > 44 ? bytes.AsMemory(44) : bytes, ctx.RequestAborted);
            Record(ctx, "supertonic", 200);
        }
        catch (OperationCanceledException)
        {
            Record(ctx, "supertonic", 499);
        }
        catch (Exception e)
        {
            Log.Error("Local API /v1/audio/speech failed", e);
            if (!ctx.Response.HasStarted)
                await WriteError(ctx, 500, "api_error", e.Message);
            Record(ctx, "supertonic", 500);
        }
    }

    // ---------------------------------------------------------------- /v1/images/generations

    /// <summary>
    /// Text to image through ComfyUI (SDXL by default, "flux-schnell" when asked), or on Snapdragon PCs through Stable
    /// Diffusion 1.5 on the NPU (512 x 512; "model" picks the checkpoint by id or name), returned as base64 PNG.
    /// </summary>
    private async Task Images(HttpContext ctx)
    {
        var body = await JsonNode.ParseAsync(ctx.Request.Body) as JsonObject ?? new JsonObject();
        var prompt = body["prompt"]?.GetValue<string>() ?? "";
        var requested = body["model"]?.GetValue<string>() ?? "";
        if (!ComfyService.IsSupported && Studio.Npu.SdNpuService.IsSupported)
        {
            await ImagesOnNpu(ctx, body, prompt, requested);
            return;
        }
        var workflow = requested.Contains("flux", StringComparison.OrdinalIgnoreCase) ? ComfyWorkflow.FluxSchnell : ComfyWorkflow.Sdxl;
        if (!ComfyService.IsSupported)
        {
            await WriteError(ctx, 501, "not_supported", Loc.S("ComfyUnsupported"));
            Record(ctx, workflow.Id, 501);
            return;
        }
        if (prompt.Trim().Length == 0)
        {
            await WriteError(ctx, 400, "invalid_request_error", "'prompt' is empty.");
            Record(ctx, workflow.Id, 400);
            return;
        }
        var (width, height) = ParseSize(body["size"]?.GetValue<string>());
        var count = Math.Clamp(body["n"]?.GetValue<int>() ?? 1, 1, 4);
        try
        {
            await _mediaGate.WaitAsync(ctx.RequestAborted);
            List<string> files;
            try
            {
                await Ioc.Comfy.EnsureRunningAsync(ctx.RequestAborted);
                await Ioc.Comfy.EnsureModelsAsync(workflow, ctx.RequestAborted);
                var graph = workflow.Build(prompt.Trim(), width, height, count, workflow.DefaultSteps, Random.Shared.NextInt64(1, 1L << 48), null);
                files = await Ioc.Comfy.RunAsync(graph, (_, _) => { }, ctx.RequestAborted);
            }
            finally
            {
                _mediaGate.Release();
            }
            var data = new JsonArray();
            foreach (var file in files)
                data.Add(new JsonObject { ["b64_json"] = Convert.ToBase64String(await File.ReadAllBytesAsync(file, ctx.RequestAborted)), ["revised_prompt"] = prompt });
            await ctx.Response.WriteAsJsonAsync(new JsonObject { ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), ["data"] = data });
            Record(ctx, workflow.Id, 200);
        }
        catch (OperationCanceledException)
        {
            Record(ctx, workflow.Id, 499);
        }
        catch (Exception e)
        {
            Log.Error("Local API /v1/images/generations failed", e);
            if (!ctx.Response.HasStarted)
                await WriteError(ctx, 500, "api_error", e.Message);
            Record(ctx, workflow.Id, 500);
        }
    }

    /// <summary>/v1/images/generations on the Hexagon NPU: one of Studio's Stable Diffusion 1.5 models at 512 x 512.</summary>
    private async Task ImagesOnNpu(HttpContext ctx, JsonObject body, string prompt, string requested)
    {
        var model = Studio.Npu.SdNpuModel.All.FirstOrDefault(m => requested.Length > 0 &&
                        (m.Id.Equals(requested, StringComparison.OrdinalIgnoreCase) || m.Name.Contains(requested, StringComparison.OrdinalIgnoreCase)))
                    ?? Studio.Npu.SdNpuModel.All[0];
        var id = "npu-" + model.Id;
        if (prompt.Trim().Length == 0)
        {
            await WriteError(ctx, 400, "invalid_request_error", "'prompt' is empty.");
            Record(ctx, id, 400);
            return;
        }
        var count = Math.Clamp(body["n"]?.GetValue<int>() ?? 1, 1, 4);
        var steps = Math.Clamp(body["steps"]?.GetValue<int>() ?? 20, 1, 50);
        var seed = body["seed"]?.GetValue<long>() ?? Random.Shared.NextInt64(1, 1L << 31);
        var negative = body["negative_prompt"]?.GetValue<string>()
                       ?? "lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry, deformed";
        try
        {
            var files = await Ioc.SdNpu.GenerateAsync(model, prompt.Trim(), negative, steps, 7f, seed, count, null, 0.6f, (_, _) => { }, ctx.RequestAborted);
            var data = new JsonArray();
            foreach (var file in files)
                data.Add(new JsonObject { ["b64_json"] = Convert.ToBase64String(await File.ReadAllBytesAsync(file, ctx.RequestAborted)), ["revised_prompt"] = prompt });
            await ctx.Response.WriteAsJsonAsync(new JsonObject { ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), ["data"] = data });
            Record(ctx, id, 200);
        }
        catch (OperationCanceledException)
        {
            Record(ctx, id, 499);
        }
        catch (Exception e)
        {
            Log.Error("Local API /v1/images/generations (NPU) failed", e);
            if (!ctx.Response.HasStarted)
                await WriteError(ctx, 500, "api_error", e.Message);
            Record(ctx, id, 500);
        }
    }

    /// <summary>"1024x1024" style sizes, snapped to multiples of 64 within what the image models handle well.</summary>
    private static (int Width, int Height) ParseSize(string? size)
    {
        var parts = (size ?? "").Split('x', StringSplitOptions.TrimEntries);
        if (parts.Length == 2 && int.TryParse(parts[0], out var w) && int.TryParse(parts[1], out var h))
            return (Math.Clamp(w / 64 * 64, 512, 1536), Math.Clamp(h / 64 * 64, 512, 1536));
        return (1024, 1024);
    }

    private InstalledModel? FindChatModel(string requested) =>
        Ioc.Models.Installed.FirstOrDefault(m => !m.IsEmbedding && (ApiName(m) == requested || m.Id == requested || m.CatalogId == requested));

    // ---------------------------------------------------------------- unload after idle

    private DateTime _lastApiUse = DateTime.MinValue;
    private Timer? _idleTimer;

    /// <summary>Marks that a model was loaded or used for an API client, for "Unload after idle".</summary>
    private void MarkApiUse()
    {
        _lastApiUse = DateTime.UtcNow;
        _idleTimer ??= new Timer(_ => UnloadIfIdle(), null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
    }

    /// <summary>
    /// Frees the model's memory once API clients have been quiet for the chosen time, unless the app itself is using it
    /// (a chat reply, an Agent turn or a design being generated).
    /// </summary>
    private void UnloadIfIdle()
    {
        var minutes = AppSettings.Current.ApiUnloadAfterMinutes;
        if (minutes <= 0 || _lastApiUse == DateTime.MinValue || DateTime.UtcNow - _lastApiUse < TimeSpan.FromMinutes(minutes))
            return;
        UiThread.Post(() =>
        {
            if (Ioc.Chat.IsGenerating || Ioc.Agent.Sessions.Any(s => s.IsWorking) || Ioc.DesignVm.IsGenerating)
                return;
            _lastApiUse = DateTime.MinValue;
            Log.Info($"Local API idle for {minutes} min: unloading models");
            Ioc.Engines.Stop();
            Ioc.Embeddings.Stop();
        });
    }
}
