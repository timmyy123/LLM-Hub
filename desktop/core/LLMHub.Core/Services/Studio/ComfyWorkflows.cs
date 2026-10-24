using System.Text.Json.Nodes;

namespace LLMHub.Services.Studio;

/// <summary>A model file a workflow needs, and the ComfyUI models/ subfolder it goes in.</summary>
public sealed record ComfyFile(string Folder, string Name, string Url, long Size);

/// <summary>
/// The built-in Studio workflows, in ComfyUI's API format. No models from Chinese companies (the owner's rule): Flux.1
/// schnell (Black Forest Labs, Apache-2.0), SDXL (Stability AI), LTX-Video 2B 0.9.5 (Lightricks), 4x-ClearReality (Apache-2.0).
/// </summary>
public sealed record ComfyWorkflow(string Id, string Name, string Kind, bool NeedsImage, int DefaultSteps, IReadOnlyList<ComfyFile> Files)
{
    public override string ToString() => Name;

    /// <summary>Set for the Snapdragon NPU workflows (Stable Diffusion 1.5 through QNN) instead of ComfyUI.</summary>
    public LLMHub.Services.Studio.Npu.SdNpuModel? NpuModel { get; init; }
    /// <summary>The Snapdragon NPU video workflow (LTX-Video through stable-diffusion.cpp and QNN).</summary>
    public bool NpuVideo { get; init; }
    public bool IsNpu => NpuModel is not null || NpuVideo;

    public long DownloadSize => Files.Sum(f => f.Size);

    public static readonly ComfyWorkflow FluxSchnell = new("flux-schnell", "Text to image · Flux.1 schnell", "image", false, 4,
    [
        new("checkpoints", "flux1-schnell-fp8.safetensors", "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors", 17_236_328_572),
    ]);

    public static readonly ComfyWorkflow Sdxl = new("sdxl", "Text to image · SDXL", "image", false, 25,
    [
        new("checkpoints", "sd_xl_base_1.0.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", 6_938_078_334),
    ]);

    public static readonly ComfyWorkflow LtxVideo = new("ltxv-i2v", "Image to video · LTX-Video 2B", "video", true, 30,
    [
        new("checkpoints", "ltx-video-2b-v0.9.5.safetensors", "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors", 6_340_729_500),
        new("text_encoders", "t5xxl_fp8_e4m3fn.safetensors", "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors", 4_893_934_904),
    ]);

    public static readonly ComfyWorkflow Upscale = new("upscale", "Upscale 4x", "image", true, 0,
    [
        new("upscale_models", "4x-ClearRealityV1.safetensors", "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.safetensors", 4_492_232),
    ]);

    public static IReadOnlyList<ComfyWorkflow> All { get; } = [FluxSchnell, Sdxl, LtxVideo, Upscale];

    /// <summary>
    /// Windows on Arm: Stable Diffusion 1.5 checkpoints on the Hexagon NPU (512 x 512, optional reference for image to
    /// image), and LTX-Video 2B 0.9.8 distilled (Lightricks) when sd-cli.exe ships with the app (optional reference for
    /// image to video).
    /// </summary>
    public static IReadOnlyList<ComfyWorkflow> NpuWorkflows { get; } =
    [
        .. LLMHub.Services.Studio.Npu.SdNpuModel.All.Select(m => new ComfyWorkflow("npu-" + m.Id, $"{m.Name} · NPU", "image", false, 20, []) { NpuModel = m }),
        .. LLMHub.Services.Studio.Npu.LtxNpuService.IsSupported
            ? [new ComfyWorkflow("npu-ltxv", "LTX-Video 2B 0.9.8 · NPU", "video", false, LLMHub.Services.Studio.Npu.LtxNpuService.MaxSteps, []) { NpuVideo = true }]
            : Array.Empty<ComfyWorkflow>(),
    ];

    /// <summary>The API-format graph for one run. <paramref name="image"/> is a name returned by ComfyUI's /upload/image.</summary>
    public JsonObject Build(string prompt, int width, int height, int count, int steps, long seed, string? image)
    {
        static JsonArray Link(string node, int output) => new(node, output);
        static JsonObject Node(string type, JsonObject inputs) => new() { ["class_type"] = type, ["inputs"] = inputs };
        var prefix = "LLMHub/" + Id;

        return Id switch
        {
            "flux-schnell" => new JsonObject
            {
                ["1"] = Node("CheckpointLoaderSimple", new() { ["ckpt_name"] = Files[0].Name }),
                ["2"] = Node("CLIPTextEncode", new() { ["text"] = prompt, ["clip"] = Link("1", 1) }),
                ["3"] = Node("CLIPTextEncode", new() { ["text"] = "", ["clip"] = Link("1", 1) }),
                ["4"] = Node("EmptySD3LatentImage", new() { ["width"] = width, ["height"] = height, ["batch_size"] = count }),
                ["5"] = Node("KSampler", new()
                {
                    ["model"] = Link("1", 0), ["positive"] = Link("2", 0), ["negative"] = Link("3", 0), ["latent_image"] = Link("4", 0),
                    ["seed"] = seed, ["steps"] = steps, ["cfg"] = 1.0, ["sampler_name"] = "euler", ["scheduler"] = "simple", ["denoise"] = 1.0,
                }),
                ["6"] = Node("VAEDecode", new() { ["samples"] = Link("5", 0), ["vae"] = Link("1", 2) }),
                ["7"] = Node("SaveImage", new() { ["images"] = Link("6", 0), ["filename_prefix"] = prefix }),
            },
            "sdxl" => new JsonObject
            {
                ["1"] = Node("CheckpointLoaderSimple", new() { ["ckpt_name"] = Files[0].Name }),
                ["2"] = Node("CLIPTextEncode", new() { ["text"] = prompt, ["clip"] = Link("1", 1) }),
                ["3"] = Node("CLIPTextEncode", new() { ["text"] = "blurry, low quality, distorted, watermark, text", ["clip"] = Link("1", 1) }),
                ["4"] = Node("EmptyLatentImage", new() { ["width"] = width, ["height"] = height, ["batch_size"] = count }),
                ["5"] = Node("KSampler", new()
                {
                    ["model"] = Link("1", 0), ["positive"] = Link("2", 0), ["negative"] = Link("3", 0), ["latent_image"] = Link("4", 0),
                    ["seed"] = seed, ["steps"] = steps, ["cfg"] = 7.0, ["sampler_name"] = "euler", ["scheduler"] = "normal", ["denoise"] = 1.0,
                }),
                ["6"] = Node("VAEDecode", new() { ["samples"] = Link("5", 0), ["vae"] = Link("1", 2) }),
                ["7"] = Node("SaveImage", new() { ["images"] = Link("6", 0), ["filename_prefix"] = prefix }),
            },
            "ltxv-i2v" => new JsonObject
            {
                ["1"] = Node("CheckpointLoaderSimple", new() { ["ckpt_name"] = Files[0].Name }),
                ["2"] = Node("CLIPLoader", new() { ["clip_name"] = Files[1].Name, ["type"] = "ltxv" }),
                ["3"] = Node("CLIPTextEncode", new() { ["text"] = prompt, ["clip"] = Link("2", 0) }),
                ["4"] = Node("CLIPTextEncode", new() { ["text"] = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly", ["clip"] = Link("2", 0) }),
                ["5"] = Node("LoadImage", new() { ["image"] = image ?? "" }),
                ["6"] = Node("LTXVPreprocess", new() { ["image"] = Link("5", 0), ["img_compression"] = 40 }),
                ["7"] = Node("LTXVImgToVideo", new()
                {
                    ["positive"] = Link("3", 0), ["negative"] = Link("4", 0), ["vae"] = Link("1", 2), ["image"] = Link("6", 0),
                    ["width"] = width, ["height"] = height, ["length"] = 97, ["batch_size"] = 1, ["strength"] = 1.0,
                }),
                ["8"] = Node("LTXVConditioning", new() { ["positive"] = Link("7", 0), ["negative"] = Link("7", 1), ["frame_rate"] = 25.0 }),
                ["9"] = Node("LTXVScheduler", new()
                {
                    ["steps"] = steps, ["max_shift"] = 2.05, ["base_shift"] = 0.95, ["stretch"] = true, ["terminal"] = 0.1, ["latent"] = Link("7", 2),
                }),
                ["10"] = Node("KSamplerSelect", new() { ["sampler_name"] = "euler" }),
                ["11"] = Node("SamplerCustom", new()
                {
                    ["model"] = Link("1", 0), ["add_noise"] = true, ["noise_seed"] = seed, ["cfg"] = 3.0,
                    ["positive"] = Link("8", 0), ["negative"] = Link("8", 1), ["sampler"] = Link("10", 0), ["sigmas"] = Link("9", 0), ["latent_image"] = Link("7", 2),
                }),
                ["12"] = Node("VAEDecode", new() { ["samples"] = Link("11", 0), ["vae"] = Link("1", 2) }),
                ["13"] = Node("SaveAnimatedWEBP", new()
                {
                    ["images"] = Link("12", 0), ["filename_prefix"] = prefix, ["fps"] = 24.0, ["lossless"] = false, ["quality"] = 90, ["method"] = "default",
                }),
            },
            "upscale" => new JsonObject
            {
                ["1"] = Node("UpscaleModelLoader", new() { ["model_name"] = Files[0].Name }),
                ["2"] = Node("LoadImage", new() { ["image"] = image ?? "" }),
                ["3"] = Node("ImageUpscaleWithModel", new() { ["upscale_model"] = Link("1", 0), ["image"] = Link("2", 0) }),
                ["4"] = Node("SaveImage", new() { ["images"] = Link("3", 0), ["filename_prefix"] = prefix }),
            },
            _ => throw new ArgumentException(Id),
        };
    }
}
