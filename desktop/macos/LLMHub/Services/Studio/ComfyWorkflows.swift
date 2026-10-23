import Foundation

/// A model file a workflow needs, and the ComfyUI models/ subfolder it goes in.
struct ComfyFile: Hashable {
    let folder: String
    let name: String
    let url: String
    let size: Int64
}

/// The built-in Studio workflows in ComfyUI's API format, the same graphs as the Windows and Linux apps
/// (core/LLMHub.Core/Services/Studio/ComfyWorkflows.cs). No models from Chinese companies (the owner's rule): Flux.1
/// schnell (Black Forest Labs, Apache-2.0), SDXL (Stability AI), LTX-Video 2B 0.9.5 (Lightricks), 4x-ClearReality (Apache-2.0).
struct ComfyWorkflow: Identifiable, Hashable {
    let id: String
    let name: String
    /// image or video
    let kind: String
    let needsImage: Bool
    let defaultSteps: Int
    let files: [ComfyFile]

    var downloadSize: Int64 { files.reduce(0) { $0 + $1.size } }

    static let fluxSchnell = ComfyWorkflow(id: "flux-schnell", name: "Text to image · Flux.1 schnell", kind: "image", needsImage: false, defaultSteps: 4, files: [
        ComfyFile(folder: "checkpoints", name: "flux1-schnell-fp8.safetensors",
                  url: "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors", size: 17_236_328_572),
    ])

    static let sdxl = ComfyWorkflow(id: "sdxl", name: "Text to image · SDXL", kind: "image", needsImage: false, defaultSteps: 25, files: [
        ComfyFile(folder: "checkpoints", name: "sd_xl_base_1.0.safetensors",
                  url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", size: 6_938_078_334),
    ])

    static let ltxVideo = ComfyWorkflow(id: "ltxv-i2v", name: "Image to video · LTX-Video 2B", kind: "video", needsImage: true, defaultSteps: 30, files: [
        ComfyFile(folder: "checkpoints", name: "ltx-video-2b-v0.9.5.safetensors",
                  url: "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors", size: 6_340_729_500),
        ComfyFile(folder: "text_encoders", name: "t5xxl_fp8_e4m3fn.safetensors",
                  url: "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors", size: 4_893_934_904),
    ])

    static let upscale = ComfyWorkflow(id: "upscale", name: "Upscale 4x", kind: "image", needsImage: true, defaultSteps: 0, files: [
        ComfyFile(folder: "upscale_models", name: "4x-ClearRealityV1.safetensors",
                  url: "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.safetensors", size: 4_492_232),
    ])

    static let all = [fluxSchnell, sdxl, ltxVideo, upscale]

    /// The API-format graph for one run. `image` is a name returned by ComfyUI's /upload/image.
    func build(prompt: String, width: Int, height: Int, count: Int, steps: Int, seed: Int64, image: String?) -> [String: Any] {
        func link(_ node: String, _ output: Int) -> [Any] { [node, output] }
        func node(_ type: String, _ inputs: [String: Any]) -> [String: Any] { ["class_type": type, "inputs": inputs] }
        let prefix = "LLMHub/" + id

        switch id {
        case "flux-schnell":
            return [
                "1": node("CheckpointLoaderSimple", ["ckpt_name": files[0].name]),
                "2": node("CLIPTextEncode", ["text": prompt, "clip": link("1", 1)]),
                "3": node("CLIPTextEncode", ["text": "", "clip": link("1", 1)]),
                "4": node("EmptySD3LatentImage", ["width": width, "height": height, "batch_size": count]),
                "5": node("KSampler", [
                    "model": link("1", 0), "positive": link("2", 0), "negative": link("3", 0), "latent_image": link("4", 0),
                    "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                ]),
                "6": node("VAEDecode", ["samples": link("5", 0), "vae": link("1", 2)]),
                "7": node("SaveImage", ["images": link("6", 0), "filename_prefix": prefix]),
            ]
        case "sdxl":
            return [
                "1": node("CheckpointLoaderSimple", ["ckpt_name": files[0].name]),
                "2": node("CLIPTextEncode", ["text": prompt, "clip": link("1", 1)]),
                "3": node("CLIPTextEncode", ["text": "blurry, low quality, distorted, watermark, text", "clip": link("1", 1)]),
                "4": node("EmptyLatentImage", ["width": width, "height": height, "batch_size": count]),
                "5": node("KSampler", [
                    "model": link("1", 0), "positive": link("2", 0), "negative": link("3", 0), "latent_image": link("4", 0),
                    "seed": seed, "steps": steps, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                ]),
                "6": node("VAEDecode", ["samples": link("5", 0), "vae": link("1", 2)]),
                "7": node("SaveImage", ["images": link("6", 0), "filename_prefix": prefix]),
            ]
        case "ltxv-i2v":
            return [
                "1": node("CheckpointLoaderSimple", ["ckpt_name": files[0].name]),
                "2": node("CLIPLoader", ["clip_name": files[1].name, "type": "ltxv"]),
                "3": node("CLIPTextEncode", ["text": prompt, "clip": link("2", 0)]),
                "4": node("CLIPTextEncode", ["text": "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                                             "clip": link("2", 0)]),
                "5": node("LoadImage", ["image": image ?? ""]),
                "6": node("LTXVPreprocess", ["image": link("5", 0), "img_compression": 40]),
                "7": node("LTXVImgToVideo", [
                    "positive": link("3", 0), "negative": link("4", 0), "vae": link("1", 2), "image": link("6", 0),
                    "width": width, "height": height, "length": 97, "batch_size": 1, "strength": 1.0,
                ]),
                "8": node("LTXVConditioning", ["positive": link("7", 0), "negative": link("7", 1), "frame_rate": 25.0]),
                "9": node("LTXVScheduler", [
                    "steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": true, "terminal": 0.1, "latent": link("7", 2),
                ]),
                "10": node("KSamplerSelect", ["sampler_name": "euler"]),
                "11": node("SamplerCustom", [
                    "model": link("1", 0), "add_noise": true, "noise_seed": seed, "cfg": 3.0,
                    "positive": link("8", 0), "negative": link("8", 1), "sampler": link("10", 0), "sigmas": link("9", 0), "latent_image": link("7", 2),
                ]),
                "12": node("VAEDecode", ["samples": link("11", 0), "vae": link("1", 2)]),
                "13": node("SaveAnimatedWEBP", [
                    "images": link("12", 0), "filename_prefix": prefix, "fps": 24.0, "lossless": false, "quality": 90, "method": "default",
                ]),
            ]
        default: // upscale
            return [
                "1": node("UpscaleModelLoader", ["model_name": files[0].name]),
                "2": node("LoadImage", ["image": image ?? ""]),
                "3": node("ImageUpscaleWithModel", ["upscale_model": link("1", 0), "image": link("2", 0)]),
                "4": node("SaveImage", ["images": link("3", 0), "filename_prefix": prefix]),
            ]
        }
    }

    /// Pixel size for an aspect ratio: about one megapixel for images, LTX-Video's 768 x 512 class for video.
    static func size(_ workflow: ComfyWorkflow, aspect: String) -> (width: Int, height: Int) {
        if workflow.kind == "video" {
            switch aspect {
            case "9:16": return (512, 768)
            case "1:1": return (640, 640)
            default: return (768, 512)
            }
        }
        switch aspect {
        case "1:1": return (1024, 1024)
        case "16:9": return (1344, 768)
        case "9:16": return (768, 1344)
        default: return (1216, 832)
        }
    }
}
