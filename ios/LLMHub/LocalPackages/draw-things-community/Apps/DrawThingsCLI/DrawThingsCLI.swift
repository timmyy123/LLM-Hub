import ArgumentParser
import AudioConverter
import BinaryResources
import CLICloudAuth
import ConfigurationZoo
import DataModels
import Dflat
import Diffusion
import DiffusionCoreMLModelManager
import Dispatch
import Downloader
import Foundation
import GRPCServer
import ImageGenerator
import LocalImageGenerator
import ModelOp
import ModelZoo
import NNC
import PNG
import RemoteImageGenerator
import SQLiteDflat
import ScriptDataModels
import Tokenizer
import Trainer

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#endif

#if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
  import AVFoundation
  import AudioToolbox
  import CoreMedia
  import CoreVideo
#endif
#if canImport(CoreGraphics)
  import CoreGraphics
#endif
#if canImport(ImageIO)
  import ImageIO
#endif

private enum DrawThingsCLIError: LocalizedError {
  case invalidModelsDirectory(String)
  case invalidOutputPath(String)
  case invalidConfigurationJSON
  case invalidLoRAConfigurationJSON
  case missingModel
  case unsupportedModelInput(String)
  case missingModelFiles([String])
  case generationFailed
  case unsupportedTensorShape(String)
  case invalidAudioTensorShape(String)
  case invalidImageDimensions(Int, Int)
  case pngEncodeFailed(String)
  case videoEncodeFailed(String)
  case unsupportedVideoOutput(String)
  case invalidInputImagePath(String)
  case invalidInputImage(String)

  var errorDescription: String? {
    switch self {
    case .invalidModelsDirectory(let path):
      return "Models directory path is not valid: \(path)"
    case .invalidOutputPath(let path):
      return "Output path extension must be .png, .mov, or .mp4: \(path)"
    case .invalidConfigurationJSON:
      return "Failed to parse configuration override JSON"
    case .invalidLoRAConfigurationJSON:
      return "Failed to parse JSON as LoRATrainingConfiguration"
    case .missingModel:
      return "--model is required"
    case .unsupportedModelInput(let model):
      return "Unable to resolve model from input: \(model)"
    case .missingModelFiles(let files):
      return
        "Missing model files:\n\(files.map { "  - \($0)" }.joined(separator: "\n"))\nUse --download-missing or run `\(CLIIdentity.command("models ensure --model ..."))`."
    case .generationFailed:
      return "Generation failed (no tensors returned)"
    case .unsupportedTensorShape(let shape):
      return "Unsupported output tensor shape: \(shape)"
    case .invalidAudioTensorShape(let shape):
      return "Unsupported audio tensor shape: \(shape)"
    case .invalidImageDimensions(let width, let height):
      return "Image dimensions must be multiples of 64, got \(width)x\(height)"
    case .pngEncodeFailed(let outputPath):
      return "Failed to encode PNG at path: \(outputPath)"
    case .videoEncodeFailed(let outputPath):
      return "Failed to encode video at path: \(outputPath)"
    case .unsupportedVideoOutput(let outputPath):
      return
        "Video output is not supported on this platform for path: \(outputPath). Use .png output instead."
    case .invalidInputImagePath(let path):
      return "Input image path does not exist: \(path)"
    case .invalidInputImage(let path):
      return "Failed to decode input image: \(path)"
    }
  }
}

private struct ResolvedGenerationConfiguration {
  let configuration: GenerationConfiguration
  let recommendedNegativePrompt: String?
  let loraOverrideMapping: [String: LoRAZoo.Specification]
}

enum VideoExportFormat: String, ExpressibleByArgument {
  case prores4444
  case prores422hq
  case h264
  case hevc
}

enum TerminalImageProtocol: String, ExpressibleByArgument {
  case auto
  case iterm2
  case kitty
}

private enum TerminalImageRenderMode {
  case disabled
  case automatic
  case explicit
}

private enum CLIIdentity {
  static let commandName = "draw-things-cli"

  static let version: String = {
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      !version.isEmpty
    {
      return version
    }
    let fallbackInfoPlist = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
      .appendingPathComponent("Apps/DrawThings/SupportingFiles/Info.plist")
    if let info = NSDictionary(contentsOf: fallbackInfoPlist),
      let version = info["CFBundleShortVersionString"] as? String,
      !version.isEmpty
    {
      return version
    }
    return "dev"
  }()

  static func command(_ arguments: String) -> String {
    return "\(commandName) \(arguments)"
  }
}

private let defaultCloudAPIBaseURL = CLICloudDefaultAPIBaseURL

private enum DrawThingsCLICredentialsStore {
  private static let store = CLICloudCredentialsStore(applicationName: "DrawThingsCLI")

  static func load() -> CLICloudCredentials? {
    store.load()
  }

  static func save(_ credentials: CLICloudCredentials) throws {
    try store.save(credentials)
  }

  static func remove() throws {
    try store.remove()
  }

  static func description() -> String {
    store.credentialsPath
  }
}

private struct AuthCommandOutput: Encodable {
  enum Action: String, Encodable {
    case login
    case logout
  }

  let action: Action
  let provider: String?
  let credentialsPath: String
  let cloudAPIBaseURL: String?

  static func login(
    credentials: CLICloudCredentials,
    credentialsPath: String,
    apiBaseURL: URL
  ) -> AuthCommandOutput {
    AuthCommandOutput(
      action: .login,
      provider: credentials.provider,
      credentialsPath: credentialsPath,
      cloudAPIBaseURL: apiBaseURL.absoluteString
    )
  }

  static func logout(credentialsPath: String) -> AuthCommandOutput {
    AuthCommandOutput(
      action: .logout,
      provider: nil,
      credentialsPath: credentialsPath,
      cloudAPIBaseURL: nil
    )
  }
}

private enum CLIHelpText {
  static let root = """
    DESCRIPTION:
      Run local Draw Things inference and local LoRA training from the command line.

    COMMON COMMANDS:
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube on a table\""))
      \(CLIIdentity.command("generate --cloud-compute --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube on a table\""))
      \(CLIIdentity.command("models list --downloaded-only"))
      \(CLIIdentity.command("models ensure --model flux_2_dev_q8p.ckpt"))
      \(CLIIdentity.command("auth login"))
      \(CLIIdentity.command("train lora --model flux_2_klein_4b_q6p.ckpt --dataset ./dataset"))
      \(CLIIdentity.command("completion zsh"))

    HELP:
      Use `\(CLIIdentity.command("<command> --help"))` for command-specific help.

    ENVIRONMENT:
      DRAWTHINGS_MODELS_DIR
        Default models directory when --models-dir is not provided.
    """

  static let generate = """
    DESCRIPTION:
      Resolve a local inference model, load recommended settings, apply JSON overrides,
      then apply explicit command-line overrides before generation. By default generation
      runs locally in the standalone CLI. Local Code automatically uses cloud compute
      for supported images when its account has a saved Draw Things key. Use --local
      to force local generation, --remote for a Draw Things server, or --cloud-compute
      to explicitly request Draw Things cloud compute.

    MODEL REFERENCES:
      --model accepts a model file id, a human-readable model name, an hf://owner/repo
      reference, an owner/repo reference, or a Hugging Face model URL.

    CONFIGURATION PRECEDENCE:
      1. Recommended settings for the resolved model.
      2. Overrides from --config-json or --config-file.
      3. Explicit command-line flags such as --steps, --cfg, --width, --height,
         --frames, --seed, and --strength.
      4. --negative-prompt overrides any recommended negative prompt.

    AUDIO VIDEO CONTINUATION:
      --avc enables segmented audio video continuation. It currently supports only
      LongCat-Video-Avatar 1.5. The output duration follows --audio; use
      --segment-frames and --cond-frames to control continuation segments.

    EXAMPLES:
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube on a table\""))
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt-file prompt.txt"))
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"studio portrait\" --image input.png --strength 0.35"))
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"combine these references\" --image first.png --image second.png"))
      \(CLIIdentity.command("generate --cloud-compute --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube on a table\" --output cube.png"))
      \(CLIIdentity.command("generate --remote --remote-url 127.0.0.1 --no-remote-tls --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube\""))
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube on a table\" --terminal-image"))
      \(CLIIdentity.command("generate --model ltx_2.3_22b_distilled_q6p.ckpt --prompt \"ocean waves at sunset\" --frames 49 --output clip.mov"))
      \(CLIIdentity.command("generate --avc --model longcat_video_avatar_1.5_dmd_i8x.ckpt --image man.png --audio man.mp3 --output man.mp4"))
    """

  static let models = """
    DESCRIPTION:
      Inspect model mappings and ensure local model files exist before generation or training.
    """

  static let modelList = """
    DESCRIPTION:
      List official models in ModelZoo order, then append community models from the latest
      catalog, falling back to cached data if unavailable. --offline and --downloaded-only
      use cached catalog data without fetching updates.
    """

  static let modelEnsure = """
    DESCRIPTION:
      Resolve a model reference and download the model file, plus dependencies by default.

    EXAMPLE:
      \(CLIIdentity.command("models ensure --model flux_2_dev_q8p.ckpt"))
    """

  static let modelImport = """
    DESCRIPTION:
      Import a local checkpoint or safetensors artifact into Draw Things format, infer a
      custom model specification, and optionally download missing companion models.

    INPUTS:
      Local files only for now. Supported source formats depend on ModelImporter and
      commonly include .safetensors, .ckpt, .pth, .pt, .bin, and .zip.

    EXAMPLES:
      \(CLIIdentity.command("models import ./flux-2-klein-4b.safetensors"))
      \(CLIIdentity.command("models import ./model.safetensors --name \"My Model\" --trigger-word mytoken"))
      \(CLIIdentity.command("models import ./model.safetensors --dry-run"))
    """

  static let train = """
    DESCRIPTION:
      Training commands for local fine-tuning workflows.
    """

  static let completion = """
    DESCRIPTION:
      Generate a shell completion script for bash, zsh, or fish.

    EXAMPLES:
      \(CLIIdentity.command("completion zsh > ~/.zsh/completions/_draw-things-cli"))
      \(CLIIdentity.command("completion bash > /etc/bash_completion.d/draw-things-cli"))
      \(CLIIdentity.command("completion fish > ~/.config/fish/completions/draw-things-cli.fish"))
    """

  static let loraTrain = """
    DESCRIPTION:
      Train a LoRA adapter using a local dataset and a local base model.

    CONFIGURATION PRECEDENCE:
      1. LoRATrainingConfiguration.default.
      2. Partial overrides from --config-json.
      3. Explicit command-line flags such as --steps, --rank, --scale, --learning-rate,
         --width, --height, and execution flags.

    DATASET:
      --dataset should point to a directory of images. Matching .txt files are treated as captions.

    EXAMPLES:
      \(CLIIdentity.command("train lora --model flux_2_klein_4b_q6p.ckpt --dataset ./dataset --steps 200"))
      \(CLIIdentity.command("train lora --config-json '{\"base_model\":\"flux_2_klein_4b_q6p.ckpt\"}' --dataset ./dataset --dry-run"))
    """
}

private let modelsDirectoryHelp = ArgumentHelp(
  "Models directory.",
  discussion:
    "Resolution order: --models-dir, DRAWTHINGS_MODELS_DIR, then on macOS ~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models. On other platforms, fall back to <binary-dir>/Models (if it exists), then ~/Documents/Models."
)

private let modelReferenceHelp = ArgumentHelp(
  "Model file, model name, or Hugging Face repo/URL.",
  discussion:
    "Accepted forms: flux_2_klein_4b_q6p.ckpt, \"FLUX.2 [klein] 4B (6-bit)\", hf://owner/repo, owner/repo, or https://huggingface.co/owner/repo."
)

private let generateConfigJSONHelp = ArgumentHelp(
  "Inline JSON override in JSGenerationConfiguration format.",
  discussion:
    "This is merged onto the model's recommended settings. It is not treated as a complete configuration by itself. For custom local LoRAs that are not registered in custom_lora.json, include loras[].version (for example \"flux1\") so the CLI can construct per-invocation LoRA metadata."
)

private let generateConfigFileHelp = ArgumentHelp(
  "Path to a JSON override file in JSGenerationConfiguration format.",
  discussion:
    "The file is parsed as a partial override and merged onto the model's recommended settings. For custom local LoRAs that are not registered in custom_lora.json, include loras[].version (for example \"flux1\") so the CLI can construct per-invocation LoRA metadata."
)

private let promptFileHelp = ArgumentHelp(
  "Read prompt text from a file, or `-` for stdin.",
  discussion:
    "Use this for long or multiline prompts. Mutually exclusive with the inline prompt flag.")

private let negativePromptFileHelp = ArgumentHelp(
  "Read negative prompt text from a file, or `-` for stdin.",
  discussion:
    "Use this for long or multiline negative prompts. Mutually exclusive with the inline negative prompt flag."
)

private let generateImageHelp = ArgumentHelp(
  "Input image path. Repeat --image to add reference images for supported models.",
  discussion:
    "The first image is the primary img2img or reference input; subsequent images are additional references in the supplied order. Each image is resized with aspect-preserving scale and center crop to match the requested output size. --avc accepts exactly one image."
)

private let videoFormatHelp = ArgumentHelp(
  "Video export format for .mov/.mp4 outputs.",
  discussion:
    "Accepted values: prores4444, prores422hq, h264, hevc. ProRes formats require .mov output."
)

private let terminalImageHelp = ArgumentHelp(
  "Render the first generated PNG inline in supported terminals.",
  discussion:
    "Auto-detects iTerm2 or kitty-graphics terminals such as kitty and Ghostty. PNG output only."
)

private let terminalImageProtocolHelp = ArgumentHelp(
  "Terminal inline image protocol.",
  discussion:
    "Accepted values: auto, iterm2, kitty. `auto` prefers iTerm2, then kitty-graphics terminals such as Ghostty."
)

private let modelImportArtifactHelp = ArgumentHelp(
  "Local model artifact to import.",
  discussion:
    "Typically a .safetensors, .ckpt, .pth, .pt, .bin, or .zip file. Remote URLs are not supported by this command yet."
)

private let modelImportScaleHelp = ArgumentHelp(
  "Default finetune scale in 64px units.",
  discussion:
    "For example, 16 means a default 1024px base resolution. If omitted, the CLI mirrors the app's import defaults for the detected model family."
)

private let trainConfigJSONHelp = ArgumentHelp(
  "Inline JSON override in LoRATrainingConfiguration format.",
  discussion:
    "This is merged onto LoRATrainingConfiguration.default. It is not treated as a complete training configuration by itself."
)

private let trainDatasetHelp = ArgumentHelp(
  "Dataset directory containing images and optional .txt captions.",
  discussion:
    "Images are loaded from the directory and matching .txt files are treated as captions when present."
)

private enum NetworkCacheResolver {
  static func primaryURL(fileName: String) throws -> URL {
    guard
      let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
        .first
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    return cacheDirectory.appendingPathComponent("net", isDirectory: true).appendingPathComponent(
      fileName)
  }

  static func candidateURLs(fileName: String, modelsDirectory: URL) -> [URL] {
    var urls: [URL] = []
    if let primary = try? primaryURL(fileName: fileName) {
      urls.append(primary)
    }
    let modelsDirectoryPath = modelsDirectory.standardizedFileURL.pathComponents
    if modelsDirectoryPath.suffix(3) == ["Data", "Documents", "Models"] {
      let dataDirectory =
        modelsDirectory
        .deletingLastPathComponent()
        .deletingLastPathComponent()
      urls.append(
        dataDirectory
          .appendingPathComponent("Library", isDirectory: true)
          .appendingPathComponent("Caches", isDirectory: true)
          .appendingPathComponent("net", isDirectory: true)
          .appendingPathComponent(fileName))
    }
    return urls
  }

  static func persist(_ data: Data, fileName: String) throws {
    let fileURL = try primaryURL(fileName: fileName)
    try FileManager.default.createDirectory(
      at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: fileURL, options: .atomic)
  }
}

private func fetchShortTermToken(
  context: DrawThingsCLIContext, apiKey: String,
  baseURL: URL,
  emitStates: Bool
) throws -> String {
  let client = CLICloudAuthClient(baseURL: baseURL)
  return try client.fetchShortTermToken(
    apiKey: apiKey,
    emitStates: emitStates,
    stateHandler: { state in
      context.print("  [AuthState] \(describeCLICloudAuthState(state))")
    }
  ).value
}

struct ModelsDirectoryOptions: ParsableArguments {
  @Option(name: .long, help: modelsDirectoryHelp)
  var modelsDir: String?
}

struct GenerateModelResolutionOptions: ParsableArguments {
  @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

  @Option(name: .shortAndLong, help: modelReferenceHelp)
  var model: String?
}

struct GeneratePromptOptions: ParsableArguments {
  @Option(name: .shortAndLong, help: "Prompt text.")
  var prompt: String?

  @Option(name: .long, help: promptFileHelp)
  var promptFile: String?

  @Option(
    name: .long,
    help: ArgumentHelp(
      "Negative prompt text.",
      discussion: "If omitted, recommended settings may provide one."))
  var negativePrompt: String?

  @Option(name: .long, help: negativePromptFileHelp)
  var negativePromptFile: String?
}

struct GenerateSamplingOptions: ParsableArguments {
  @Option(name: .long, help: "Sampling steps.")
  var steps: Int?

  @Option(name: .long, help: "CFG guidance scale.")
  var cfg: Float?

  @Option(name: .long, help: "Output width in pixels (multiple of 64).")
  var width: Int?

  @Option(name: .long, help: "Output height in pixels (multiple of 64).")
  var height: Int?

  @Option(
    name: .long,
    help:
      "Number of frames for video-capable models. Without --avc, duration is the frame count divided by the model FPS."
  )
  var frames: Int?

  @Option(name: .long, help: "Denoising strength for img2img (0...1).")
  var strength: Float?

  @Option(name: .shortAndLong, help: "Random seed.")
  var seed: UInt32?
}

struct GenerateConfigurationOverrideOptions: ParsableArguments {
  @Option(name: .long, help: generateConfigJSONHelp)
  var configJSON: String?

  @Option(name: .long, help: generateConfigFileHelp)
  var configFile: String?
}

struct GenerateImageInputOptions: ParsableArguments {
  @Option(name: .long, help: generateImageHelp)
  var image: [String] = []

  @Option(
    name: .customLong("init-image"),
    help: ArgumentHelp("Alias of --image.", visibility: .hidden))
  var initImage: String?

  @Option(
    name: .customLong("input-image"),
    help: ArgumentHelp("Alias of --image.", visibility: .hidden))
  var inputImage: String?

  @Option(
    name: .long,
    help: "Audio file for LongCat driving audio or MiniMax H3 Ref2VA native audio references.")
  var audio: String?

  @Option(
    name: .customLong("audio-encoder-file"),
    help:
      "Audio encoder checkpoint filename in the models directory (defaults to the model's encoder)."
  )
  var audioEncoderFile: String?
}

struct GenerateOutputOptions: ParsableArguments {
  @Option(
    name: .shortAndLong,
    help: ArgumentHelp(
      "Output path (.png, .mov, or .mp4).",
      discussion:
        "Use .png for image output and .mov/.mp4 for video-capable models. If omitted, the CLI previews the result inline in supported interactive terminals and does not write a file."
    ))
  var output: String?

  @Option(name: .long, help: videoFormatHelp)
  var videoFormat: VideoExportFormat?

  @Flag(name: .long, help: terminalImageHelp)
  var terminalImage: Bool = false

  @Option(name: .long, help: terminalImageProtocolHelp)
  var terminalImageProtocol: TerminalImageProtocol = .auto
}

struct GenerateExecutionOptions: ParsableArguments {
  @Flag(name: .long, inversion: .prefixedNo, help: "Auto-download missing model files.")
  var downloadMissing: Bool = true

  @Flag(name: .long, help: "Disable live sampling preview during generation.")
  var disablePreview: Bool = false

  @Flag(
    name: .long,
    help:
      "Disable network access. Uses cached community catalogs and recommended settings only, and never downloads models."
  )
  var offline: Bool = false
}

struct GenerateBackendOptions: ParsableArguments {
  @Flag(name: .long, help: "Generate locally, overriding the embedding app's automatic backend.")
  var local: Bool = false

  @Flag(name: .long, help: "Generate on a remote Draw Things server instead of local models.")
  var remote: Bool = false

  @Option(name: .customLong("remote-url"), help: "Remote server host or URL.")
  var remoteURL: String?

  @Option(name: .customLong("remote-port"), help: "Remote server port.")
  var remotePort: Int = 7859

  @Flag(
    name: .customLong("remote-tls"),
    inversion: .prefixedNo,
    help: "Use TLS for remote generation.")
  var remoteTLS: Bool = true

  @Option(
    name: .customLong("remote-shared-secret"),
    help: "Shared secret for password-protected remote servers.")
  var remoteSharedSecret: String?

  @Flag(name: .customLong("cloud-compute"), help: "Generate on Draw Things cloud compute.")
  var cloudCompute: Bool = false

  var automaticCloudCompute = false

  @Option(
    name: .long,
    help:
      "Draw Things API key for cloud compute. Uses saved credentials or DRAWTHINGS_API_KEY if omitted."
  )
  var apiKey: String?

  @Option(name: .customLong("cloud-api-base-url"), help: "Cloud API base URL.")
  var cloudAPIBaseURL: String?

  var isRemoteOrCloud: Bool {
    remote || cloudCompute
  }

  func validate() throws {
    if local && (remote || cloudCompute) {
      throw ValidationError("--local cannot be combined with --remote or --cloud-compute.")
    }
    if remote && cloudCompute {
      throw ValidationError("--remote and --cloud-compute cannot be combined.")
    }
    if !remote
      && (remoteURL != nil || remotePort != 7859 || remoteTLS != true || remoteSharedSecret != nil)
    {
      throw ValidationError("Remote endpoint flags require --remote.")
    }
    if !cloudCompute && (apiKey != nil || cloudAPIBaseURL != nil) {
      throw ValidationError("Cloud compute flags require --cloud-compute.")
    }
  }
}

struct GenerateAVCOptions: ParsableArguments {
  @Flag(
    name: .customLong("avc"),
    help: ArgumentHelp(
      "Enable segmented audio video continuation.",
      discussion: "Currently supported only for LongCat-Video-Avatar 1.5."))
  var enabled: Bool = false

  @Option(
    name: .customLong("segment-frames"),
    help: "Frames generated per AVC segment. Requires --avc. (default: 93)")
  var segmentFrames: Int?

  @Option(
    name: .customLong("cond-frames"),
    help: "Overlap frames reused for AVC continuation. Requires --avc. (default: 13)")
  var condFrames: Int?

  @Flag(
    name: .customLong("zero-audio-features"),
    help: ArgumentHelp(
      "Use zero LongCat audio features for fast AVC pipeline validation.",
      visibility: .hidden)
  )
  var zeroAudioFeatures: Bool = false
}

struct LoRATrainModelAndDatasetOptions: ParsableArguments {
  @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

  @Option(name: .shortAndLong, help: modelReferenceHelp)
  var model: String?

  @Option(name: .long, help: trainDatasetHelp)
  var dataset: String?
}

struct LoRATrainConfigurationOverrideOptions: ParsableArguments {
  @Option(name: .long, help: trainConfigJSONHelp)
  var configJSON: String?

  @Option(
    name: .long,
    help: ArgumentHelp(
      "Memory saver mode.",
      discussion: "Accepted values: minimal, balanced, speed, turbo."))
  var memorySaver: String?

  @Option(
    name: .long,
    help: ArgumentHelp(
      "Weights memory mode.",
      discussion: "Accepted values: cached or justInTime."))
  var weightsMemory: String?
}

struct LoRATrainOutputOptions: ParsableArguments {
  @Option(name: .shortAndLong, help: "Output LoRA filename prefix (without extension).")
  var output: String?

  @Option(name: .long, help: "Display name for LoRA metadata.")
  var name: String?

  @Option(name: .long, help: "Resume from a saved checkpoint filename/path.")
  var resume: String?

  @Option(
    name: .customLong("train-checkpoint"),
    help: ArgumentHelp("Alias of --resume.", visibility: .hidden))
  var trainCheckpoint: String?
}

struct LoRATrainTrainingOptions: ParsableArguments {
  @Option(name: .long, help: "Training steps.")
  var steps: Int?

  @Option(name: .long, help: "LoRA rank.")
  var rank: Int?

  @Option(name: .long, help: "LoRA scale.")
  var scale: Float?

  @Option(
    name: .long,
    help:
      "UNet learning rate as a float or range. Examples: 1e-4, [5e-5,1e-4], 5e-5:1e-4.")
  var learningRate: String?

  @Option(name: .long, help: "Gradient accumulation steps.")
  var gradientAccumulation: Int?

  @Option(name: .long, help: "Warmup steps.")
  var warmupSteps: Int?

  @Option(name: .long, help: "Save checkpoint every N steps (0 means only final).")
  var saveEvery: Int?

  @Option(name: .long, help: "Random seed.")
  var seed: UInt32?
}

struct LoRATrainDataAndModelOptions: ParsableArguments {
  @Option(name: .long, help: "CLIP skip layers.")
  var clipSkip: Int?

  @Option(name: .long, help: "Noise offset.")
  var noiseOffset: Float?

  @Option(name: .long, help: "Caption dropout rate.")
  var captionDropout: Float?

  @Option(name: .long, help: "Output resolution in pixels (square).")
  var resolution: Int?

  @Option(name: .long, help: "Output width in pixels (multiple of 64).")
  var width: Int?

  @Option(name: .long, help: "Output height in pixels (multiple of 64).")
  var height: Int?

  @Flag(
    name: .long,
    inversion: .prefixedNo,
    help: "Use image aspect ratio bucket training.")
  var useAspectRatio: Bool?

  @Flag(
    name: .long,
    inversion: .prefixedNo,
    help: "Use orthonormal LoRA down initialization.")
  var orthonormalDown: Bool?

  @Flag(
    name: .long,
    inversion: .prefixedNo,
    help: "Co-train the text model.")
  var cotrainTextModel: Bool?
}

struct LoRATrainExecutionOptions: ParsableArguments {
  @Flag(name: .long, help: "Validate and print resolved training config without training.")
  var dryRun: Bool = false

  @Flag(name: .long, inversion: .prefixedNo, help: "Auto-download missing model files.")
  var downloadMissing: Bool = true

  @Flag(
    name: .long,
    help:
      "Disable network access. Uses cached community catalogs only, and never downloads models.")
  var offline: Bool = false
}

struct LoRATrainCommandOptions: ParsableArguments {
  @OptionGroup(title: "Model And Dataset") var modelAndDataset: LoRATrainModelAndDatasetOptions
  @OptionGroup(title: "Configuration Overrides")
  var configurationOverrides: LoRATrainConfigurationOverrideOptions
  @OptionGroup(title: "Output And Resume") var output: LoRATrainOutputOptions
  @OptionGroup(title: "Training") var training: LoRATrainTrainingOptions
  @OptionGroup(title: "Data And Model") var dataAndModel: LoRATrainDataAndModelOptions
  @OptionGroup(title: "Execution") var execution: LoRATrainExecutionOptions
}

private enum ModelsDirectoryResolver {
  private static let appContainerModelsDirectoryPath =
    "~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models"

  static func resolve(context: DrawThingsCLIContext, path: String?) throws -> URL {
    let path =
      path.flatMap { $0.isEmpty ? nil : $0 }
      ?? context.environment["DRAWTHINGS_MODELS_DIR"].flatMap { $0.isEmpty ? nil : $0 }
    let requested = path.map { URL(fileURLWithPath: context.path($0), isDirectory: true) }
    if let directory = try context.resolveModelsDirectoryAccess(requested) {
      return try normalizeAndEnsureDirectory(directory)
    }
    #if os(macOS)
      let appContainerModelsDirectory = URL(
        fileURLWithPath: (appContainerModelsDirectoryPath as NSString).expandingTildeInPath,
        isDirectory: true)
      return try normalizeAndEnsureDirectory(appContainerModelsDirectory)
    #else
      if let adjacent = executableAdjacentModelsDirectoryIfExists(context: context) {
        return adjacent
      }
      let fallback = try documentsModelsDirectory()
      return try normalizeAndEnsureDirectory(fallback)
    #endif
  }

  private static func executableAdjacentModelsDirectoryIfExists(context: DrawThingsCLIContext)
    -> URL?
  {
    guard let executable = context.executablePath else { return nil }
    let executablePath = URL(fileURLWithPath: executable).resolvingSymlinksInPath()
    let modelsPath = executablePath.deletingLastPathComponent().appendingPathComponent(
      "Models", isDirectory: true)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: modelsPath.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      return nil
    }
    return modelsPath
  }

  private static func documentsModelsDirectory() throws -> URL {
    guard
      let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    else {
      throw DrawThingsCLIError.invalidModelsDirectory("~/Documents/Models")
    }
    return documents.appendingPathComponent("Models", isDirectory: true)
  }

  private static func normalizeAndEnsureDirectory(_ url: URL) throws -> URL {
    let normalized = url.standardizedFileURL
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: normalized.path, isDirectory: &isDirectory)
    if exists && !isDirectory.boolValue {
      throw DrawThingsCLIError.invalidModelsDirectory(normalized.path)
    }
    if !exists {
      try FileManager.default.createDirectory(at: normalized, withIntermediateDirectories: true)
    }
    return normalized
  }
}

private enum ModelResolver {
  static func resolve(context: DrawThingsCLIContext, _ input: String, modelsDirectory: URL? = nil)
    -> ModelZoo.Specification?
  {
    if let specification = ModelZoo.resolveModelReference(input)?.specification {
      return specification
    }
    guard let modelsDirectory else { return nil }
    return CommunityModelResolver(modelsDirectory: modelsDirectory).resolve(
      input, allowNetwork: !context.offline)
  }

  static func suggestions(_ input: String, limit: Int = 5) -> [ModelZoo.Specification] {
    return ModelZoo.candidateSpecifications(forModelReference: input, limit: limit)
  }
}

private enum ConfigurationLoader {
  static func load(
    context: DrawThingsCLIContext, modelSpecification: ModelZoo.Specification, configJSON: String?,
    configFile: String?,
    modelsDirectory: URL
  ) throws -> ResolvedGenerationConfiguration {
    if configJSON != nil && configFile != nil {
      throw ValidationError("Use only one of --config-json or --config-file")
    }
    let overrideDictionary = try loadOverrideDictionary(
      context: context, configJSON: configJSON, configFile: configFile)
    let loraOverrideMapping = loraOverrideMapping(from: overrideDictionary)
    let (recommendedConfiguration, recommendedNegativePrompt) =
      RecommendedSettingsResolver.resolve(
        modelSpecification: modelSpecification, overrideDictionary: overrideDictionary,
        modelsDirectory: modelsDirectory, allowNetwork: !context.offline)
    let configuration = try mergeOverrides(
      overrideDictionary, onto: recommendedConfiguration, forcingModel: modelSpecification.file)
    return ResolvedGenerationConfiguration(
      configuration: configuration, recommendedNegativePrompt: recommendedNegativePrompt,
      loraOverrideMapping: loraOverrideMapping)
  }

  private static func loadOverrideDictionary(
    context: DrawThingsCLIContext, configJSON: String?, configFile: String?
  ) throws -> [String: Any]? {
    guard
      let rawString = try loadRawJSON(
        context: context, configJSON: configJSON, configFile: configFile)
    else {
      return nil
    }
    guard let data = rawString.data(using: .utf8),
      let dictionary = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    return dictionary
  }

  private static func mergeOverrides(
    _ overrideDictionary: [String: Any]?, onto baseConfiguration: GenerationConfiguration,
    forcingModel model: String
  ) throws -> GenerationConfiguration {
    guard
      let baseData = try? JSONEncoder().encode(
        JSGenerationConfiguration(configuration: baseConfiguration)),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: baseData) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    if let overrideDictionary {
      for (key, value) in overrideDictionary {
        mergedDictionary[key] = value
      }
    }
    mergedDictionary["model"] = model
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? JSONDecoder().decode(
        JSGenerationConfiguration.self, from: mergedData
      )
      .createGenerationConfiguration()
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    return configuration
  }

  private static func loadRawJSON(
    context: DrawThingsCLIContext, configJSON: String?, configFile: String?
  ) throws -> String? {
    if let configJSON {
      return configJSON
    }
    if let configFile {
      return try String(contentsOfFile: context.path(configFile), encoding: .utf8)
    }
    return nil
  }

  private static func loraOverrideMapping(
    from overrideDictionary: [String: Any]?
  ) -> [String: LoRAZoo.Specification] {
    guard let loras = overrideDictionary?["loras"] as? [[String: Any]] else { return [:] }
    let normalizedLoRAs: [[String: Any]] = loras.map { lora in
      var lora = lora
      lora["weight"] = nil
      if lora["name"] == nil || lora["name"] is NSNull, let file = lora["file"] as? String {
        lora["name"] = file
      }
      if lora["prefix"] == nil || lora["prefix"] is NSNull {
        lora["prefix"] = ""
      }
      return lora
    }
    guard let data = try? JSONSerialization.data(withJSONObject: normalizedLoRAs) else {
      return [:]
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let specifications = try? decoder.decode(
        [FailableDecodable<LoRAZoo.Specification>].self, from: data
      ).compactMap({ $0.value })
    else { return [:] }
    var mapping = [String: LoRAZoo.Specification]()
    for specification in specifications {
      mapping[specification.file] = specification
    }
    return mapping
  }
}

private enum RecommendedSettingsResolver {
  static func resolve(
    modelSpecification: ModelZoo.Specification, overrideDictionary: [String: Any]?,
    modelsDirectory: URL, allowNetwork: Bool
  ) -> (GenerationConfiguration, String?) {
    let defaultConfiguration = defaultConfiguration(for: modelSpecification)
    let loras = loras(from: overrideDictionary)
    guard
      let specification = findRecommendedSettings(
        model: modelSpecification.file, loras: loras, modelsDirectory: modelsDirectory,
        allowNetwork: allowNetwork)
    else {
      var builder = GenerationConfigurationBuilder(from: defaultConfiguration)
      builder.model = modelSpecification.file
      return (builder.build(), nil)
    }
    guard
      let defaultData = try? JSONEncoder().encode(
        JSGenerationConfiguration(configuration: defaultConfiguration)),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: defaultData) as? [String: Any]
    else {
      var builder = GenerationConfigurationBuilder(from: defaultConfiguration)
      builder.model = modelSpecification.file
      return (builder.build(), nil)
    }
    for (key, value) in specification.configuration {
      mergedDictionary[key] = value
    }
    mergedDictionary["model"] = modelSpecification.file
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? JSONDecoder().decode(
        JSGenerationConfiguration.self, from: mergedData
      )
      .createGenerationConfiguration()
    else {
      var builder = GenerationConfigurationBuilder(from: defaultConfiguration)
      builder.model = modelSpecification.file
      return (builder.build(), nil)
    }
    return (configuration, specification.negative?.trimmingCharacters(in: .whitespacesAndNewlines))
  }

  private static func defaultConfiguration(for modelSpecification: ModelZoo.Specification)
    -> GenerationConfiguration
  {
    let defaultScale = DeviceCapability.defaultScale(modelSpecification.defaultScale)
    var builder = GenerationConfigurationBuilder(from: GenerationConfiguration.default)
    builder.model = modelSpecification.file
    builder.startWidth = defaultScale
    builder.startHeight = defaultScale
    if modelSpecification.version == .qwenImage2_1 {
      builder.steps = 40
      builder.guidanceScale = 1
      builder.sampler = .dDIMTrailing
      builder.resolutionDependentShift = true
      builder.shift = 1
    } else if modelSpecification.version == .hiDreamO1 {
      builder.steps = 28
      builder.guidanceScale = 1
      builder.sampler = .dPMPP2MTrailing
      builder.seedMode = .legacy
      builder.resolutionDependentShift = true
      builder.shift = 1
    } else if modelSpecification.version == .longcatVideoAvatar1_5 {
      builder.steps = 8
      builder.guidanceScale = 1
      builder.sampler = .dDIMTrailing
      builder.numFrames = 93
      builder.shift = 7
    } else if modelSpecification.version == .minimaxH3 {
      builder.steps = 50
      builder.guidanceScale = 1
      builder.sampler = .dDIMTrailing
      builder.startWidth = 21
      builder.startHeight = 12
      builder.numFrames = 124
      builder.shift = 12
    }
    return builder.build()
  }

  private static func loras(from overrideDictionary: [String: Any]?) -> Set<String> {
    guard let loras = overrideDictionary?["loras"] as? [[String: Any]] else { return [] }
    return Set(loras.compactMap { $0["file"] as? String })
  }

  private static func findRecommendedSettings(
    model: String, loras: Set<String>, modelsDirectory: URL, allowNetwork: Bool,
    timeout: TimeInterval = 10
  ) -> ConfigurationZoo.Specification? {
    let configurations = refreshedCommunityConfigurations(
      timeout: timeout, modelsDirectory: modelsDirectory, allowNetwork: allowNetwork)
    guard !configurations.isEmpty else { return nil }
    let version = ModelZoo.versionForModel(model)
    let prefix = prefix(for: model)
    var bestMatch = matchWithLoRAs(
      configurations: configurations, loras,
      first: {
        ($0.configuration["model"] as? String) == model
      },
      second: {
        guard let configModel = $0.configuration["model"] as? String, !prefix.isEmpty else {
          return false
        }
        return Self.prefix(for: configModel) == prefix
      })
    if bestMatch == nil {
      bestMatch = matchWithLoRAs(configurations: configurations, loras) {
        guard let configModel = $0.configuration["model"] as? String, !prefix.isEmpty else {
          return false
        }
        let configPrefix = Self.prefix(for: configModel)
        return !configPrefix.isEmpty && prefix.hasPrefix("\(configPrefix)_")
      }
    }
    if bestMatch == nil {
      bestMatch = matchWithLoRAs(configurations: configurations, loras) {
        $0.version == version
      }
    }
    return bestMatch
  }

  private static func refreshedCommunityConfigurations(
    timeout: TimeInterval, modelsDirectory: URL, allowNetwork: Bool
  ) -> [ConfigurationZoo.Specification] {
    if allowNetwork,
      let fetched = try? fetchCommunityConfigurations(timeout: timeout), !fetched.isEmpty
    {
      return fetched
    }
    let cached = cachedCommunityConfigurations(modelsDirectory: modelsDirectory)
    if !cached.isEmpty {
      return cached
    }
    return ConfigurationZoo.community
  }

  private static func prefix(for file: String) -> String {
    let stem = (file as NSString).deletingPathExtension
    guard !stem.isEmpty else { return "" }
    var components = stem.components(separatedBy: "_")
    while let last = components.last, ["f16", "svd", "q5p", "q6p", "q8p", "i8x"].contains(last) {
      components.removeLast()
    }
    return components.joined(separator: "_")
  }

  private static func matchWithLoRAs(
    configurations: [ConfigurationZoo.Specification], _ loras: Set<String>,
    first: (ConfigurationZoo.Specification) -> Bool,
    second: ((ConfigurationZoo.Specification) -> Bool)? = nil
  ) -> ConfigurationZoo.Specification? {
    guard !loras.isEmpty else {
      guard let second = second else {
        return configurations.first(where: first)
      }
      return configurations.first(where: first) ?? configurations.first(where: second)
    }
    guard let second = second else {
      return configurations.first {
        let isFirst = first($0)
        guard isFirst else { return false }
        guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
        return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
      } ?? configurations.first(where: first)
    }
    return configurations.first {
      let isFirst = first($0)
      guard isFirst else { return false }
      guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
      return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
    } ?? configurations.first {
      let isSecond = second($0)
      guard isSecond else { return false }
      guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
      return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
    } ?? configurations.first(where: first) ?? configurations.first(where: second)
  }

  private static func fetchCommunityConfigurations(timeout: TimeInterval) throws
    -> [ConfigurationZoo.Specification]
  {
    guard let url = URL(string: "https://models.drawthings.ai/configs.json") else {
      return []
    }
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<Data, Error> = .failure(DrawThingsCLIError.invalidConfigurationJSON)
    URLSession.shared.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      if let error {
        result = .failure(error)
        return
      }
      guard let response = response as? HTTPURLResponse, 200...299 ~= response.statusCode else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      guard let data else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      result = .success(data)
    }.resume()
    guard semaphore.wait(timeout: .now() + timeout) != .timedOut else {
      return []
    }
    guard case .success(let data) = result else {
      return []
    }
    let specifications = parseCommunityConfigurations(data: data)
    guard !specifications.isEmpty else {
      return []
    }
    try? persistCommunityConfigurations(data)
    return specifications
  }

  private static func parseCommunityConfigurations(data: Data) -> [ConfigurationZoo.Specification] {
    guard let jsonSpecifications = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
    else {
      return []
    }
    return jsonSpecifications.compactMap { specification in
      guard let name = specification["name"] as? String,
        let configuration = specification["configuration"] as? [String: Any]
      else {
        return nil
      }
      return ConfigurationZoo.Specification(
        name: name,
        version: (specification["version"] as? String).flatMap { ModelVersion(rawValue: $0) },
        negative: specification["negative"] as? String,
        configuration: configuration)
    }
  }

  private static func cachedCommunityConfigurations(modelsDirectory: URL)
    -> [ConfigurationZoo.Specification]
  {
    for url in NetworkCacheResolver.candidateURLs(
      fileName: "configs.json", modelsDirectory: modelsDirectory)
    {
      guard let data = try? Data(contentsOf: url) else { continue }
      let configurations = parseCommunityConfigurations(data: data)
      if !configurations.isEmpty {
        return configurations
      }
    }
    return []
  }

  private static func persistCommunityConfigurations(_ data: Data) throws {
    try NetworkCacheResolver.persist(data, fileName: "configs.json")
  }
}

struct CommunityModelResolver {
  private let cachedSpecifications: () -> [ModelZoo.Specification]
  private let fetchSpecifications: () -> [ModelZoo.Specification]

  init(modelsDirectory: URL, timeout: TimeInterval = 10) {
    self.init(
      cachedSpecifications: {
        Self.cachedCommunitySpecifications(modelsDirectory: modelsDirectory)
      },
      fetchSpecifications: { (try? Self.fetchCommunitySpecifications(timeout: timeout)) ?? [] })
  }

  init(
    cachedSpecifications: @escaping () -> [ModelZoo.Specification],
    fetchSpecifications: @escaping () -> [ModelZoo.Specification]
  ) {
    self.cachedSpecifications = cachedSpecifications
    self.fetchSpecifications = fetchSpecifications
  }

  func resolve(_ input: String, allowNetwork: Bool = true) -> ModelZoo.Specification? {
    let local = cachedSpecifications().filter { $0.remoteApiModelConfig == nil }
    if let specification = Self.matchingSpecification(for: input, in: local) {
      Self.primeOverrideMapping(with: specification)
      return specification
    }
    guard allowNetwork else {
      return nil
    }
    // A downloaded checkpoint still needs its catalog metadata before it can be used.
    let fetched = fetchSpecifications()
    if let specification = Self.matchingSpecification(for: input, in: fetched) {
      Self.primeOverrideMapping(with: specification)
      return specification
    }
    return nil
  }

  func allSpecifications(allowNetwork: Bool = true) -> [ModelZoo.Specification] {
    let official = ModelZoo.availableSpecifications.filter { $0.remoteApiModelConfig == nil }
    // Listing should discover newly published models even when an older catalog is cached.
    let fetched = allowNetwork ? fetchSpecifications() : []
    let community = (fetched.isEmpty ? cachedSpecifications() : fetched).filter {
      $0.remoteApiModelConfig == nil
    }
    var seen = Set<String>()
    var combined = [ModelZoo.Specification]()
    for specification in official + community where !seen.contains(specification.file) {
      seen.insert(specification.file)
      combined.append(specification)
    }
    return combined
  }

  private static func cachedCommunitySpecifications(modelsDirectory: URL)
    -> [ModelZoo.Specification]
  {
    for url in NetworkCacheResolver.candidateURLs(
      fileName: "models.json", modelsDirectory: modelsDirectory)
    {
      guard let data = try? Data(contentsOf: url) else { continue }
      let specifications = parseCommunitySpecifications(data: data)
      if !specifications.isEmpty {
        return specifications
      }
    }
    return []
  }

  private static func fetchCommunitySpecifications(timeout: TimeInterval) throws
    -> [ModelZoo.Specification]
  {
    guard let url = URL(string: "https://models.drawthings.ai/models.json") else {
      return []
    }
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<Data, Error> = .failure(DrawThingsCLIError.invalidConfigurationJSON)
    URLSession.shared.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      if let error {
        result = .failure(error)
        return
      }
      guard let response = response as? HTTPURLResponse, 200...299 ~= response.statusCode else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      guard let data else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      result = .success(data)
    }.resume()
    guard semaphore.wait(timeout: .now() + timeout) != .timedOut else {
      return []
    }
    guard case .success(let data) = result else {
      return []
    }
    let specifications = parseCommunitySpecifications(data: data)
    guard !specifications.isEmpty else {
      return []
    }
    try? NetworkCacheResolver.persist(data, fileName: "models.json")
    return specifications
  }

  private static func parseCommunitySpecifications(data: Data) -> [ModelZoo.Specification] {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let specifications = try? decoder.decode(
        [FailableDecodable<ModelZoo.Specification>].self, from: data
      ).compactMap({ $0.value })
    else {
      return []
    }
    return specifications
  }

  private static func primeOverrideMapping(with specification: ModelZoo.Specification) {
    ModelZoo.overrideMapping[specification.file] = specification
    if let huggingFaceLink = specification.huggingFaceLink,
      let repo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
    {
      ModelZoo.huggingFaceRepoOverrideMapping[repo] = specification
    }
  }

  private static func matchingSpecification(
    for input: String, in specifications: [ModelZoo.Specification]
  ) -> ModelZoo.Specification? {
    if let exactFileMatch = specifications.first(where: { $0.file == input }) {
      return exactFileMatch
    }
    if let exactNameMatch = specifications.first(where: { $0.name == input }) {
      return exactNameMatch
    }
    guard let canonicalRepo = ModelZoo.normalizeHuggingFaceRepo(input) else {
      return nil
    }
    return specifications.first { specification in
      guard let huggingFaceLink = specification.huggingFaceLink,
        let specificationRepo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
      else {
        return false
      }
      return specificationRepo == canonicalRepo
    }
  }

}

private enum SHARefresh {
  static func refresh(context: DrawThingsCLIContext, timeout: TimeInterval = 15) throws {
    guard !context.offline else { return }
    let endpoints = [
      "https://models.drawthings.ai/models_sha256.json",
      "https://models.drawthings.ai/uncurated_models_sha256.json",
    ]
    for endpoint in endpoints {
      try context.checkCancellation()
      guard let url = URL(string: endpoint) else { continue }
      if let map = try? fetchSHA(context: context, url: url, timeout: timeout) {
        ModelZoo.mergeFileSHA256(map)
      }
      try context.checkCancellation()
    }
  }

  private static func fetchSHA(context: DrawThingsCLIContext, url: URL, timeout: TimeInterval)
    throws -> [String: String]
  {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<[String: String], Error> = .failure(
      DrawThingsCLIError.invalidConfigurationJSON)
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    let task = URLSession.shared.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      if let error {
        result = .failure(error)
        return
      }
      if let response = response as? HTTPURLResponse, !(200...299).contains(response.statusCode) {
        result = .failure(URLError(.badServerResponse))
        return
      }
      guard let data,
        let map = try? JSONDecoder().decode([String: String].self, from: data)
      else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      result = .success(map)
    }
    context.setCancellation { task.cancel() }
    defer {
      task.cancel()
      context.setCancellation(nil)
    }
    task.resume()
    while semaphore.wait(timeout: .now() + 0.1) == .timedOut {
      try context.checkCancellation()
    }
    try context.checkCancellation()
    return try result.get()
  }
}

enum ModelDownloader {
  private final class DownloadProgressPrinter {
    private let context: DrawThingsCLIContext
    private let file: String
    private let index: Int
    private let total: Int
    private let byteFormatter: ByteCountFormatter
    private var lastLineLength: Int = 0
    private var hasRendered = false

    init(context: DrawThingsCLIContext, file: String, index: Int, total: Int) {
      self.context = context
      self.file = file
      self.index = index
      self.total = total
      let byteFormatter = ByteCountFormatter()
      byteFormatter.countStyle = .file
      byteFormatter.allowedUnits = [.useBytes, .useKB, .useMB, .useGB, .useTB]
      self.byteFormatter = byteFormatter
    }

    func update(totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64, isComplete: Bool) {
      let line = renderLine(
        totalBytesWritten: totalBytesWritten,
        totalBytesExpectedToWrite: totalBytesExpectedToWrite,
        isComplete: isComplete)
      let padding = String(
        repeating: " ", count: max(0, lastLineLength - line.count))
      let output = "\r\(line)\(padding)\(isComplete ? "\n" : "")"
      context.write(output)
      hasRendered = true
      lastLineLength = isComplete ? 0 : line.count
    }

    func finishLineIfNeeded() {
      guard hasRendered, lastLineLength > 0 else { return }
      context.write("\n")
      lastLineLength = 0
    }

    private func renderLine(
      totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64, isComplete: Bool
    ) -> String {
      let progress =
        totalBytesExpectedToWrite > 0
        ? min(1, max(0, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
        : (isComplete ? 1 : 0)
      let percent = Int((progress * 100).rounded())
      let barWidth = 24
      let filledWidth = min(barWidth, Int((Double(barWidth) * progress).rounded(.down)))
      let bar: String
      if filledWidth >= barWidth {
        bar = String(repeating: "=", count: barWidth)
      } else {
        bar =
          String(repeating: "=", count: filledWidth)
          + ">"
          + String(repeating: " ", count: max(0, barWidth - filledWidth - 1))
      }
      let written = byteFormatter.string(fromByteCount: totalBytesWritten)
      let totalBytes =
        totalBytesExpectedToWrite > 0
        ? byteFormatter.string(fromByteCount: totalBytesExpectedToWrite)
        : "?"
      return
        "[\(index)/\(total)] \(file) [\(bar)] \(String(format: "%3d", percent))% \(written)/\(totalBytes)"
    }
  }

  static func ensureFiles(
    context: DrawThingsCLIContext, _ files: [String], modelsDirectory: URL, downloadMissing: Bool
  ) throws {
    let orderedUniqueFiles = orderedSet(files)
    let missing = orderedUniqueFiles.filter { !ModelZoo.isModelDownloaded($0) }
    guard !missing.isEmpty else { return }

    if !downloadMissing {
      throw DrawThingsCLIError.missingModelFiles(missing)
    }
    if context.offline {
      throw ValidationError(
        "Offline mode is enabled and model files are missing:\n\(missing.map { "  - \($0)" }.joined(separator: "\n"))"
      )
    }

    try context.checkCancellation()
    let specification = orderedUniqueFiles.compactMap { ModelZoo.specificationForModel($0) }.first
    let downloadID = context.beginModelDownload(
      name: specification?.name ?? missing[0],
      subtitle: specification.map { ModelZoo.humanReadableNameForVersion($0.version) }
        ?? "Draw Things",
      files: missing)
    defer { context.finishModelDownload(downloadID) }
    try context.checkCancellation()
    try SHARefresh.refresh(context: context)
    for (index, file) in missing.enumerated() {
      try downloadFile(
        context: context, file, downloadID: downloadID, index: index + 1, total: missing.count,
        modelsDirectory: modelsDirectory)
    }
  }

  private static func downloadFile(
    context: DrawThingsCLIContext, _ file: String, downloadID: UUID, index: Int, total: Int,
    modelsDirectory _: URL
  ) throws {
    try context.checkCancellation()
    let encodedName = file.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? file
    guard let remoteURL = URL(string: "https://static.libnnc.org/\(encodedName)") else {
      throw ValidationError("Invalid remote URL for file \(file)")
    }
    let localURL = URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file))
    try FileManager.default.createDirectory(
      at: localURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(file)
    let semaphore = DispatchSemaphore(value: 0)
    var outputError: Error?
    let progressPrinter = DownloadProgressPrinter(
      context: context, file: file, index: index, total: total)
    let downloader = ResumableDownloader(
      remoteUrl: remoteURL, localUrl: localURL, sha256: expectedSHA)
    // Cancellation suppresses future downloader events but may race an event
    // already being delivered. Drain it before the host closes these streams.
    let callbackLock = NSLock()
    var acceptingCallbacks = true
    defer {
      callbackLock.lock()
      acceptingCallbacks = false
      progressPrinter.finishLineIfNeeded()
      callbackLock.unlock()
    }
    downloader.resume { totalBytesWritten, totalBytesExpectedToWrite, isComplete, error in
      callbackLock.lock()
      defer { callbackLock.unlock() }
      guard acceptingCallbacks else { return }
      if let error {
        progressPrinter.finishLineIfNeeded()
        outputError = error
        semaphore.signal()
        return
      }
      progressPrinter.update(
        totalBytesWritten: totalBytesWritten,
        totalBytesExpectedToWrite: totalBytesExpectedToWrite,
        isComplete: isComplete)
      context.modelDownloadEvent?(
        .progress(
          id: downloadID, file: file, index: index - 1, totalBytesWritten: totalBytesWritten,
          totalBytesExpectedToWrite: totalBytesExpectedToWrite))
      if isComplete {
        semaphore.signal()
      }
    }
    context.setCancellation { downloader.cancel() }
    defer { context.setCancellation(nil) }
    while semaphore.wait(timeout: .now() + 0.1) == .timedOut {
      if context.isCancelled {
        downloader.cancel()
        try context.checkCancellation()
      }
    }
    try context.checkCancellation()
    if let outputError {
      throw outputError
    }
  }

  private static func orderedSet(_ files: [String]) -> [String] {
    var seen = Set<String>()
    var ordered = [String]()
    for file in files where !seen.contains(file) {
      seen.insert(file)
      ordered.append(file)
    }
    return ordered
  }
}

private func resolvedLocalFileURL(_ path: String) throws -> URL {
  let expanded = (path as NSString).expandingTildeInPath
  let url = URL(fileURLWithPath: expanded)
  guard FileManager.default.fileExists(atPath: url.path) else {
    throw ValidationError("File does not exist: \(path)")
  }
  return url.standardizedFileURL
}

private func resolvedOptionalLocalFileURL(_ path: String?) throws -> URL? {
  guard let path, !path.isEmpty else { return nil }
  return try resolvedLocalFileURL(path)
}

private func defaultImportedModelDisplayName(for artifactURL: URL) -> String {
  artifactURL.deletingPathExtension().lastPathComponent
}

private func defaultImportScale(for version: ModelVersion, artifactFileName: String) -> UInt16 {
  switch version {
  case .hunyuanVideo, .wan21_14b, .wan22_5b, .longcatVideoAvatar1_5, .minimaxH3:
    return 12
  case .wan21_1_3b:
    return 8
  case .sdxlBase, .sdxlRefiner, .ssd1b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .zImage,
    .ernieImage, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .auraflow, .flux1, .flux2,
    .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2, .ltx2_3, .ideogram4, .krea2:
    return 16
  case .pixart:
    return artifactFileName.contains("512") ? 8 : 16
  case .v1, .v2, .kandinsky21, .svdI2v, .seedvr2_3b, .seedvr2_7b:
    return 8
  }
}

private func validateCustomTextEncoderSupport(
  version: ModelVersion, textEncoderURL: URL?, textEncoder2URL: URL?
) throws {
  if let textEncoder2URL, version != .sdxlBase {
    throw ValidationError(
      "--text-encoder-2 is only supported for Stable Diffusion XL Base imports, got \(textEncoder2URL.lastPathComponent)."
    )
  }
  guard textEncoderURL != nil || textEncoder2URL != nil else { return }
  switch version {
  case .v1, .v2, .sdxlBase, .ssd1b, .sdxlRefiner, .ernieImage:
    return
  case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
    .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage,
    .qwenImage2_1, .wan22_5b, .zImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2, .ltx2_3,
    .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
    throw ValidationError(
      "Custom text encoder import is not supported for \(ModelZoo.humanReadableNameForVersion(version))."
    )
  }
}

private func projectedImportedOutputFiles(
  modelName: String, version: ModelVersion, includeAutoencoder: Bool, includeTextEncoder: Bool,
  includeTextEncoder2: Bool
) throws -> [String] {
  var files = ["\(modelName)_f16.ckpt"]
  if includeTextEncoder {
    switch version {
    case .v1:
      files.append("\(modelName)_clip_vit_l14_f16.ckpt")
    case .v2:
      files.append("\(modelName)_open_clip_vit_h14_f16.ckpt")
    case .sdxlBase, .ssd1b:
      files.append("\(modelName)_clip_vit_l14_f16.ckpt")
      files.append("\(modelName)_open_clip_vit_bigg14_f16.ckpt")
    case .sdxlRefiner:
      files.append("\(modelName)_open_clip_vit_bigg14_f16.ckpt")
    case .ernieImage:
      files.append("\(modelName)_ministral_3_3b_q8p.ckpt")
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage,
      .qwenImage2_1, .wan22_5b, .zImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2,
      .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
      break
    }
  }
  if includeTextEncoder2 && version == .sdxlBase {
    let secondEncoder = "\(modelName)_open_clip_vit_bigg14_f16.ckpt"
    if !files.contains(secondEncoder) {
      files.append(secondEncoder)
    }
  }
  if includeAutoencoder {
    files.append("\(modelName)_vae_f16.ckpt")
  }
  return files
}

private func existingImportedOutputs(for files: [String]) -> [String] {
  files.filter { FileManager.default.fileExists(atPath: ModelZoo.filePathForModelDownloaded($0)) }
}

private func importPrefix(triggerWord: String?) -> String {
  let trimmed = triggerWord?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
  return trimmed.isEmpty ? "" : "\(trimmed) "
}

private func importedDependencyFiles(
  specification: ModelZoo.Specification,
  additionalModels: [(name: String, subtitle: String, file: String)],
  importedFiles: [String]
) -> [String] {
  let imported = Set(importedFiles)
  let files =
    additionalModels.map(\.file)
    + ModelZoo.filesToDownload(specification).map(\.file).filter { !imported.contains($0) }
  var seen = Set<String>()
  return files.filter { seen.insert($0).inserted }
}

private func printImportedModelSummary(
  context: DrawThingsCLIContext, specification: ModelZoo.Specification, version: ModelVersion,
  modifier: SamplerModifier,
  importedFiles: [String], dependencyFiles: [String]
) {
  let rows = [
    ["MODEL", specification.file],
    ["NAME", specification.name],
    [
      "VERSION",
      "\(ModelZoo.humanReadableNameForVersion(version)) (\(String(describing: version)))",
    ],
    ["MODIFIER", String(describing: modifier)],
    ["TEXT_ENCODER", specification.textEncoder ?? "-"],
    ["AUTOENCODER", specification.autoencoder ?? "-"],
    [
      "DEFAULT_SCALE",
      "\(specification.defaultScale) (\(Int(specification.defaultScale) * 64)x\(Int(specification.defaultScale) * 64))",
    ],
    ["PREFIX", specification.prefix.isEmpty ? "-" : specification.prefix],
  ]
  printTable(context: context, headers: ["FIELD", "VALUE"], rows: rows, maxWidths: [18, 88])
  if !importedFiles.isEmpty {
    context.print("")
    printTable(
      context: context, headers: ["IMPORTED_FILE"],
      rows: importedFiles.map { [$0] },
      maxWidths: [88])
  }
  if !dependencyFiles.isEmpty {
    context.print("")
    printTable(
      context: context, headers: ["COMPANION_FILE"],
      rows: dependencyFiles.map { [$0] },
      maxWidths: [88])
  }
}

private func createTemporaryDirectory() throws -> String {
  let path = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
  return path.path
}

private func createLocalImageGenerator(queue: DispatchQueue) throws -> (String, LocalImageGenerator)
{
  let tempDir = try createTemporaryDirectory()
  let workspace = SQLiteWorkspace(
    filePath: "\(tempDir)/config.sqlite3", fileProtectionLevel: .noProtection)
  let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
    GenerationConfiguration.id == 0, limit: .limit(0))

  let tokenizerV1 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_json,
    merges: BinaryResources.merges_txt,
    textualInversions: [])
  let tokenizerV2 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_16e6_json,
    merges: BinaryResources.bpe_simple_vocab_16e6_txt,
    textualInversions: [])
  let tokenizerKandinsky = SentencePieceTokenizer(
    data: BinaryResources.xlmroberta_bpe_model, startToken: 0,
    endToken: 2, tokenShift: 1)
  let tokenizerXL = tokenizerV2
  let tokenizerT5 = SentencePieceTokenizer(
    data: BinaryResources.t5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerPileT5 = SentencePieceTokenizer(
    data: BinaryResources.pile_t5_spiece_model, startToken: nil,
    endToken: 2, tokenShift: 0)
  let tokenizerChatGLM3 = SentencePieceTokenizer(
    data: BinaryResources.chatglm3_spiece_model, startToken: nil,
    endToken: nil, tokenShift: 0)
  let tokenizerLlama3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_llama3_json, merges: BinaryResources.merges_llama3_txt,
    specialTokens: [
      "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
      "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
    ], unknownToken: "<|end_of_text|>", startToken: "<|begin_of_text|>",
    endToken: "<|end_of_text|>")
  let tokenizerUMT5 = SentencePieceTokenizer(
    data: BinaryResources.umt5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerQwen25 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen2_5_json, merges: BinaryResources.merges_qwen2_5_txt,
    specialTokens: [
      "</tool_call>": 151658, "<tool_call>": 151657, "<|box_end|>": 151649, "<|box_start|>": 151648,
      "<|endoftext|>": 151643, "<|file_sep|>": 151664, "<|fim_middle|>": 151660,
      "<|fim_pad|>": 151662, "<|fim_prefix|>": 151659, "<|fim_suffix|>": 151661,
      "<|im_end|>": 151645, "<|im_start|>": 151644, "<|image_pad|>": 151655,
      "<|object_ref_end|>": 151647, "<|object_ref_start|>": 151646, "<|quad_end|>": 151651,
      "<|quad_start|>": 151650, "<|repo_name|>": 151663, "<|video_pad|>": 151656,
      "<|vision_end|>": 151653, "<|vision_pad|>": 151654, "<|vision_start|>": 151652,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerQwen3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
    specialTokens: [
      "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
      "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647, "<|box_start|>": 151648,
      "<|box_end|>": 151649, "<|quad_start|>": 151650, "<|quad_end|>": 151651,
      "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|vision_pad|>": 151654,
      "<|image_pad|>": 151655, "<|video_pad|>": 151656, "<tool_call>": 151657,
      "</tool_call>": 151658, "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
      "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662, "<|repo_name|>": 151663,
      "<|file_sep|>": 151664, "<tool_response>": 151665, "</tool_response>": 151666,
      "<think>": 151667, "</think>": 151668, "<|boi_token|>": 151669,
      "<|bor_token|>": 151670, "<|eor_token|>": 151671, "<|bot_token|>": 151672,
      "<|tms_token|>": 151673,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerQwen3MinimaxH3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
    specialTokens: [
      "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
      "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647, "<|box_start|>": 151648,
      "<|box_end|>": 151649, "<|quad_start|>": 151650, "<|quad_end|>": 151651,
      "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|vision_pad|>": 151654,
      "<|image_pad|>": 151655, "<|video_pad|>": 151656, "<tool_call>": 151657,
      "</tool_call>": 151658, "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
      "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662, "<|repo_name|>": 151663,
      "<|file_sep|>": 151664, "<tool_response>": 151665, "</tool_response>": 151666,
      "<think>": 151667, "</think>": 151668,
      "<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
      "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
      "<|caption_start|>": 151674, "<|caption_end|>": 151675,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerMistral3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_mistral3_json, merges: BinaryResources.merges_mistral3_txt,
    specialTokens: [
      "<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AVAILABLE_TOOLS]": 5,
      "[/AVAILABLE_TOOLS]": 6, "[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8, "[TOOL_CALLS]": 9,
      "[IMG]": 10, "<pad>": 11, "[IMG_BREAK]": 12, "[IMG_END]": 13, "[PREFIX]": 14, "[MIDDLE]": 15,
      "[SUFFIX]": 16, "[SYSTEM_PROMPT]": 17, "[/SYSTEM_PROMPT]": 18, "[TOOL_CONTENT]": 19,
    ], unknownToken: "<unk>", startToken: "<s>", endToken: "</s>")
  let tokenizerGemma3 = SentencePieceTokenizer(
    data: BinaryResources.gemma3_spiece_model, startToken: 2, endToken: nil, tokenShift: 0)
  let generator = LocalImageGenerator(
    queue: queue, configurations: configurations, workspace: workspace, tokenizerV1: tokenizerV1,
    tokenizerV2: tokenizerV2, tokenizerXL: tokenizerXL, tokenizerKandinsky: tokenizerKandinsky,
    tokenizerT5: tokenizerT5, tokenizerPileT5: tokenizerPileT5,
    tokenizerChatGLM3: tokenizerChatGLM3, tokenizerLlama3: tokenizerLlama3,
    tokenizerUMT5: tokenizerUMT5, tokenizerQwen25: tokenizerQwen25,
    tokenizerQwen3: tokenizerQwen3, tokenizerQwen3MinimaxH3: tokenizerQwen3MinimaxH3,
    tokenizerMistral3: tokenizerMistral3,
    tokenizerGemma3: tokenizerGemma3
  )
  return (tempDir, generator)
}

private struct ImageTensorShape {
  let width: Int
  let height: Int
  let channels: Int
}

private func savePNGOutputs(_ tensors: [Tensor<FloatType>], outputURL: URL) throws -> [String] {
  try FileManager.default.createDirectory(
    at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
  let outputPaths = imageOutputPaths(baseOutputURL: outputURL, count: tensors.count)
  for (tensor, outputPath) in zip(tensors, outputPaths) {
    try writePNG(tensor: tensor, to: outputPath)
  }
  return outputPaths
}

private func imageOutputPaths(baseOutputURL: URL, count: Int) -> [String] {
  if count == 1 {
    return [baseOutputURL.path]
  }
  let basename = baseOutputURL.deletingPathExtension().path
  let ext = baseOutputURL.pathExtension
  let digits = max(4, String(count - 1).count)
  return (0..<count).map { index in
    "\(basename)-\(String(format: "%0\(digits)d", index)).\(ext)"
  }
}

private func imageTensorShape(_ tensor: Tensor<FloatType>) throws -> ImageTensorShape {
  let shape = tensor.shape
  switch shape.count {
  case 4:
    return ImageTensorShape(width: shape[2], height: shape[1], channels: shape[3])
  case 3:
    return ImageTensorShape(width: shape[1], height: shape[0], channels: shape[2])
  default:
    throw DrawThingsCLIError.unsupportedTensorShape("\(shape)")
  }
}

private func pixelByte(_ value: FloatType) -> UInt8 {
  UInt8(min(max(Int((value + 1) * 127.5), 0), 255))
}

private func writePNG(tensor: Tensor<FloatType>, to outputPath: String) throws {
  let shape = try imageTensorShape(tensor)
  guard shape.channels >= 3 else {
    throw DrawThingsCLIError.unsupportedTensorShape("\(tensor.shape)")
  }
  let pixelCount = shape.width * shape.height
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: pixelCount)
  tensor.withUnsafeBytes {
    guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
    for i in 0..<pixelCount {
      let base = i * shape.channels
      let rgb = base + (shape.channels == 4 ? 1 : 0)
      rgba[i].r = pixelByte(fp16[rgb])
      rgba[i].g = pixelByte(fp16[rgb + 1])
      rgba[i].b = pixelByte(fp16[rgb + 2])
      rgba[i].a = shape.channels == 4 ? UInt8(min(max(Int(fp16[base] * 255), 0), 255)) : 255
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba,
    size: (x: shape.width, y: shape.height),
    layout: PNG.Layout(
      format: shape.channels == 4
        ? .rgba8(palette: [], fill: nil) : .rgb8(palette: [], fill: nil, key: nil)))
  do {
    try PNGFile.write(image, path: outputPath)
  } catch {
    throw DrawThingsCLIError.pngEncodeFailed(outputPath)
  }
}

// SwiftPNG's path convenience API is not available on iOS. Keep the same
// codec on every platform by providing its small synchronous stream interface.
private struct PNGFile: PNG.Bytestream.Source, PNG.Bytestream.Destination {
  let file: UnsafeMutablePointer<FILE>

  func read(count: Int) -> [UInt8]? {
    var bytes = [UInt8](repeating: 0, count: count)
    guard fread(&bytes, 1, count, file) == count else { return nil }
    return bytes
  }

  func write(_ bytes: [UInt8]) -> Void? {
    guard fwrite(bytes, 1, bytes.count, file) == bytes.count else { return nil }
    return ()
  }

  static func read(path: String) throws -> PNG.Data.Rectangular {
    guard let file = fopen(path, "rb") else {
      throw DrawThingsCLIError.invalidInputImagePath(path)
    }
    defer { fclose(file) }
    var stream = PNGFile(file: file)
    return try PNG.Data.Rectangular.decompress(stream: &stream)
  }

  static func write(_ image: PNG.Data.Rectangular, path: String) throws {
    guard let file = fopen(path, "wb") else {
      throw DrawThingsCLIError.pngEncodeFailed(path)
    }
    defer { fclose(file) }
    var stream = PNGFile(file: file)
    try image.compress(stream: &stream, level: 4)
    guard fflush(file) == 0 else { throw DrawThingsCLIError.pngEncodeFailed(path) }
  }
}

private final class LocalGenerationRunner {
  private let context: DrawThingsCLIContext
  struct GenerationTimingSummary {
    let totalGenerationDuration: TimeInterval
    let samplingStepDurations: [TimeInterval]
  }

  struct GenerationRunResult {
    let outputPaths: [String]
    let timing: GenerationTimingSummary
  }

  struct GenerationTensorResult {
    let images: [Tensor<FloatType>]
    let audio: [Tensor<Float>]?
    let timing: GenerationTimingSummary
  }

  private enum OutputDestination {
    case png(URL)
    case video(URL, containerExtension: String)
  }

  private final class GenerationTimingTracker {
    private enum SamplingPhase {
      case firstPass
      case secondPass
    }

    private let startTime = Date()
    private var lastNonSamplingSignpostTime = Date()
    private var lastSamplingStepTime: Date?
    private var currentSamplingPhase: SamplingPhase?
    private(set) var samplingStepDurations: [TimeInterval] = []

    func record(signpost: ImageGeneratorSignpost, signposts: Set<ImageGeneratorSignpost>) {
      let now = Date()
      switch signpost {
      case .sampling(let step):
        guard let totalSteps = totalSamplingSteps(in: signposts, phase: .firstPass),
          step < totalSteps
        else {
          currentSamplingPhase = nil
          lastSamplingStepTime = nil
          lastNonSamplingSignpostTime = now
          return
        }
        recordSamplingStep(at: now, phase: .firstPass)
      case .secondPassSampling(let step):
        guard let totalSteps = totalSamplingSteps(in: signposts, phase: .secondPass),
          step < totalSteps
        else {
          currentSamplingPhase = nil
          lastSamplingStepTime = nil
          lastNonSamplingSignpostTime = now
          return
        }
        recordSamplingStep(at: now, phase: .secondPass)
      default:
        currentSamplingPhase = nil
        lastSamplingStepTime = nil
        lastNonSamplingSignpostTime = now
      }
    }

    func summary() -> GenerationTimingSummary {
      GenerationTimingSummary(
        totalGenerationDuration: Date().timeIntervalSince(startTime),
        samplingStepDurations: samplingStepDurations)
    }

    private func recordSamplingStep(at time: Date, phase: SamplingPhase) {
      let stepDuration: TimeInterval
      if currentSamplingPhase == phase, let lastSamplingStepTime {
        stepDuration = time.timeIntervalSince(lastSamplingStepTime)
      } else {
        currentSamplingPhase = phase
        stepDuration = time.timeIntervalSince(lastNonSamplingSignpostTime)
      }
      samplingStepDurations.append(stepDuration)
      lastSamplingStepTime = time
    }

    private func totalSamplingSteps(
      in signposts: Set<ImageGeneratorSignpost>, phase: SamplingPhase
    ) -> Int? {
      for signpost in signposts {
        switch (phase, signpost) {
        case (.firstPass, .sampling(let steps)):
          return steps
        case (.secondPass, .secondPassSampling(let steps)):
          return steps
        default:
          continue
        }
      }
      return nil
    }
  }

  private let queue = DispatchQueue(label: "com.drawthings.cli.generate", qos: .userInteractive)
  private let temporaryDirectory: String
  private let imageGenerator: LocalImageGenerator

  init(context: DrawThingsCLIContext) throws {
    try context.prepareForLocalModelExecution()
    self.context = context
    let (temporaryDirectory, imageGenerator) = try createLocalImageGenerator(queue: queue)
    self.temporaryDirectory = temporaryDirectory
    self.imageGenerator = imageGenerator
    DeviceCapability.cacheUri = URL(fileURLWithPath: temporaryDirectory)
  }

  deinit {
    try? FileManager.default.removeItem(atPath: temporaryDirectory)
  }

  func generate(
    prompt: String, negativePrompt: String, configuration: GenerationConfiguration,
    outputPath: String,
    inputImage: Tensor<FloatType>?, videoFormat: VideoExportFormat?,
    hints: [(ControlHintType, [(AnyTensor, Float)])] = [],
    fallbackAudio: Tensor<Float>? = nil,
    fileMapping: [String: String] = [:],
    livePreviewSession: TerminalImageRenderer.LivePreviewSession? = nil
  ) throws -> GenerationRunResult {
    let tensorResult = try generateTensors(
      prompt: prompt, negativePrompt: negativePrompt, configuration: configuration,
      inputImage: inputImage, hints: hints, fileMapping: fileMapping,
      livePreviewSession: livePreviewSession)
    let outputPaths = try saveOutputs(
      tensorResult.images, audio: tensorResult.audio?.first ?? fallbackAudio,
      outputPath: outputPath,
      configuration: configuration, videoFormat: videoFormat)
    return GenerationRunResult(outputPaths: outputPaths, timing: tensorResult.timing)
  }

  func generateTensors(
    prompt: String, negativePrompt: String, configuration: GenerationConfiguration,
    inputImage: Tensor<FloatType>?, hints: [(ControlHintType, [(AnyTensor, Float)])] = [],
    fileMapping: [String: String] = [:],
    livePreviewSession: TerminalImageRenderer.LivePreviewSession? = nil
  ) throws -> GenerationTensorResult {
    let context = self.context
    try context.checkCancellation()
    defer { context.setCancellation(nil) }
    let trace = ImageGeneratorTrace(fromBridge: true)
    let progressPrinter = ProgressBarPrinter(output: context.output)
    let estimation = GenerationEstimation.default
    let timingTracker = GenerationTimingTracker()
    progressPrinter.update(progress: 0, label: "Starting...", detail: nil)
    let generationResult: ([Tensor<FloatType>]?, [Tensor<Float>]?, Int) = queue.sync {
      () -> ([Tensor<FloatType>]?, [Tensor<Float>]?, Int) in
      // Dispatch may execute this block on a different worker from the CLI.
      return DynamicGraph.fork {
        let feedback:
          (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) ->
            Bool = { signpost, signposts, previewTensor in
              timingTracker.record(signpost: signpost, signposts: signposts)
              context.updateImageGeneration(
                signpost: signpost, signposts: signposts, preview: previewTensor)
              let (elapsed, estimatedTotal) = GenerationEstimator.estimateUpToDateDuration(
                from: estimation, signpost: signpost, signposts: signposts)
              if estimatedTotal > 0 {
                let progress = Float(elapsed / estimatedTotal)
                let progressText = cliProgressText(signpost: signpost, signposts: signposts)
                progressPrinter.update(
                  progress: progress, label: progressText.label, detail: progressText.detail
                )
              }
              if let previewTensor {
                livePreviewSession?.update(tensor: previewTensor)
              }
              return !context.isCancelled
            }
        let result = imageGenerator.generate(
          trace: trace, image: inputImage, scaleFactor: 1, mask: nil, hints: hints,
          text: prompt, negativeText: negativePrompt, configuration: configuration,
          fileMapping: fileMapping,
          keywords: [], cancellation: { context.setCancellation($0) },
          feedback: feedback)
        return result
      }
    }
    try context.checkCancellation()
    let (images, audio, _) = generationResult
    guard let images, !images.isEmpty else {
      throw DrawThingsCLIError.generationFailed
    }
    let timing = timingTracker.summary()
    progressPrinter.update(progress: 1, label: "Generated", detail: nil)
    return GenerationTensorResult(images: images, audio: audio, timing: timing)
  }

  func generateLongCatAvatarAVC(
    prompt: String, negativePrompt: String, configuration: GenerationConfiguration,
    inputImage: Tensor<FloatType>, audio: Tensor<Float>, conditionFrames: Int,
    fileMapping: [String: String], zeroAudioFeatures: Bool
  ) throws -> GenerationTensorResult {
    let context = self.context
    try context.checkCancellation()
    defer { context.setCancellation(nil) }
    let progressPrinter = ProgressBarPrinter(output: context.output)
    var timingTracker: GenerationTimingTracker?
    var timings = [GenerationTimingSummary]()
    let images = try queue.sync {
      try DynamicGraph.fork {
        try imageGenerator.generateLongCatAvatarAVC(
          trace: ImageGeneratorTrace(fromBridge: true), image: inputImage, audio: audio,
          text: prompt, negativeText: negativePrompt, configuration: configuration,
          conditionFrames: conditionFrames, fileMapping: fileMapping,
          zeroAudioFeatures: zeroAudioFeatures,
          cancellation: { context.setCancellation($0) },
          segmentStart: { index, count in
            if let timingTracker { timings.append(timingTracker.summary()) }
            timingTracker = GenerationTimingTracker()
            context.beginImageGenerationSegment()
            context.print("Generating segment \(index + 1)/\(count)...")
            progressPrinter.update(progress: 0, label: "Starting...", detail: nil)
            return !context.isCancelled
          },
          feedback: { signpost, signposts, previewTensor in
            context.updateImageGeneration(
              signpost: signpost, signposts: signposts, preview: previewTensor)
            timingTracker?.record(signpost: signpost, signposts: signposts)
            let (elapsed, estimatedTotal) = GenerationEstimator.estimateUpToDateDuration(
              from: GenerationEstimation.default, signpost: signpost, signposts: signposts)
            if estimatedTotal > 0 {
              let progressText = cliProgressText(signpost: signpost, signposts: signposts)
              progressPrinter.update(
                progress: Float(elapsed / estimatedTotal), label: progressText.label,
                detail: progressText.detail)
            }
            return !context.isCancelled
          })
      }
    }
    try context.checkCancellation()
    if let timingTracker { timings.append(timingTracker.summary()) }
    progressPrinter.update(progress: 1, label: "Generated", detail: nil)
    return GenerationTensorResult(
      images: images, audio: nil, timing: combinedTimingSummary(timings))
  }

  func saveOutputs(
    _ tensors: [Tensor<FloatType>], audio: Tensor<Float>?, outputPath: String,
    configuration: GenerationConfiguration, videoFormat: VideoExportFormat?
  ) throws -> [String] {
    let destination = try normalizedOutputDestination(outputPath)
    switch destination {
    case .png(let outputURL):
      if videoFormat != nil {
        throw ValidationError("--video-format can only be used with .mov or .mp4 output")
      }
      return try savePNGOutputs(tensors, outputURL: outputURL)
    case .video(let outputURL, let containerExtension):
      let framesPerSecond = ModelZoo.framesPerSecondForModel(configuration.model ?? "")
      let audioSampleRate = audio.map { _ in
        Double(ModelZoo.audioSampleRateForModel(configuration.model ?? ""))
      }
      let path = try writeVideo(
        tensors: tensors, to: outputURL, containerExtension: containerExtension,
        framesPerSecond: framesPerSecond, videoFormat: videoFormat ?? .h264, audio: audio,
        audioSampleRate: audioSampleRate)
      return [path]
    }
  }

  private func normalizedOutputDestination(_ outputPath: String) throws -> OutputDestination {
    var url = URL(fileURLWithPath: outputPath)
    if url.pathExtension.isEmpty {
      url = url.appendingPathExtension("png")
    }
    let ext = url.pathExtension.lowercased()
    let destination: OutputDestination
    switch ext {
    case "png":
      destination = .png(url)
    case "mov", "mp4":
      destination = .video(url, containerExtension: ext)
    default:
      throw DrawThingsCLIError.invalidOutputPath(url.path)
    }
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    return destination
  }

  private func writeVideo(
    tensors: [Tensor<FloatType>], to outputURL: URL, containerExtension: String,
    framesPerSecond: Double, videoFormat: VideoExportFormat, audio: Tensor<Float>?,
    audioSampleRate: Double?
  ) throws -> String {
    #if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
      guard let first = tensors.first else {
        throw DrawThingsCLIError.generationFailed
      }
      let firstShape = try imageTensorShape(first)
      guard firstShape.channels >= 3 else {
        throw DrawThingsCLIError.unsupportedTensorShape("\(first.shape)")
      }
      for frame in tensors.dropFirst() {
        let frameShape = try imageTensorShape(frame)
        guard frameShape.channels >= 3 else {
          throw DrawThingsCLIError.unsupportedTensorShape("\(frame.shape)")
        }
        guard frameShape.width == firstShape.width, frameShape.height == firstShape.height else {
          throw DrawThingsCLIError.unsupportedTensorShape(
            "Inconsistent frame dimensions: expected \(firstShape.width)x\(firstShape.height), got \(frameShape.width)x\(frameShape.height)"
          )
        }
      }
      if FileManager.default.fileExists(atPath: outputURL.path) {
        try? FileManager.default.removeItem(at: outputURL)
      }
      if containerExtension == "mp4",
        videoFormat == .prores4444 || videoFormat == .prores422hq
      {
        throw ValidationError("ProRes video formats require .mov output")
      }
      let fileType: AVFileType = containerExtension == "mp4" ? .mp4 : .mov
      let videoSettings = videoSettings(
        for: videoFormat, width: firstShape.width, height: firstShape.height)
      let writer = try AVAssetWriter(outputURL: outputURL, fileType: fileType)
      let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
      writerInput.expectsMediaDataInRealTime = false
      let audioWriterInput: AVAssetWriterInput?
      if let audio, let audioSampleRate {
        _ = try audioTensorShape(audio)
        let input = AVAssetWriterInput(
          mediaType: .audio,
          outputSettings: audioSettings(
            for: containerExtension, sampleRate: audioSampleRate))
        input.expectsMediaDataInRealTime = false
        audioWriterInput = input
      } else {
        audioWriterInput = nil
      }
      let pixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferWidthKey as String: firstShape.width,
        kCVPixelBufferHeightKey as String: firstShape.height,
      ]
      let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: writerInput, sourcePixelBufferAttributes: pixelBufferAttributes)

      guard writer.canAdd(writerInput) else {
        throw DrawThingsCLIError.videoEncodeFailed(outputURL.path)
      }
      if let audioWriterInput, !writer.canAdd(audioWriterInput) {
        throw DrawThingsCLIError.videoEncodeFailed(outputURL.path)
      }
      writer.add(writerInput)
      if let audioWriterInput {
        writer.add(audioWriterInput)
      }
      guard writer.startWriting() else {
        throw writer.error ?? DrawThingsCLIError.videoEncodeFailed(outputURL.path)
      }
      writer.startSession(atSourceTime: .zero)
      guard let pixelBufferPool = adaptor.pixelBufferPool else {
        throw DrawThingsCLIError.videoEncodeFailed(outputURL.path)
      }

      let frameDuration = frameDurationForVideo(frameRate: framesPerSecond)
      let audioChunkFrameCount = 1024
      let writerQueue = DispatchQueue(label: "draw-things-cli.video-export")
      let semaphore = DispatchSemaphore(value: 0)
      var finishError: Error?
      var completionCalled = false
      var videoInputFinished = false
      var audioInputFinished = (audioWriterInput == nil)
      var frameIndex = 0
      var sampleOffset = 0
      var presentationTime = CMTime.zero
      let totalSampleCount: Int
      if let audio {
        totalSampleCount = try audioTensorShape(audio)
      } else {
        totalSampleCount = 0
      }

      func cancelAndComplete(_ error: Error) {
        guard !completionCalled else { return }
        completionCalled = true
        finishError = error
        writer.cancelWriting()
        semaphore.signal()
      }

      func finishWritingIfReady() {
        guard !completionCalled, videoInputFinished, audioInputFinished else { return }
        completionCalled = true
        writer.finishWriting {
          if writer.status != .completed {
            finishError = writer.error ?? DrawThingsCLIError.videoEncodeFailed(outputURL.path)
          }
          semaphore.signal()
        }
      }

      writerInput.requestMediaDataWhenReady(on: writerQueue) {
        while writerInput.isReadyForMoreMediaData {
          guard !completionCalled else { return }
          if frameIndex < tensors.count {
            let tensor = tensors[frameIndex]
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBuffer)
            guard status == kCVReturnSuccess, let pixelBuffer else {
              cancelAndComplete(DrawThingsCLIError.videoEncodeFailed(outputURL.path))
              return
            }
            do {
              try self.populate(
                pixelBuffer: pixelBuffer, with: tensor, expected: firstShape,
                outputPath: outputURL.path)
            } catch {
              cancelAndComplete(error)
              return
            }
            guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
              cancelAndComplete(
                writer.error ?? DrawThingsCLIError.videoEncodeFailed(outputURL.path))
              return
            }
            presentationTime = CMTimeAdd(presentationTime, frameDuration)
            frameIndex += 1
          } else {
            if !videoInputFinished {
              writerInput.markAsFinished()
              videoInputFinished = true
            }
            finishWritingIfReady()
            break
          }
        }
      }

      if let audioWriterInput, let audio, let audioSampleRate {
        audioWriterInput.requestMediaDataWhenReady(on: writerQueue) {
          while audioWriterInput.isReadyForMoreMediaData {
            guard !completionCalled else { return }
            if sampleOffset < totalSampleCount {
              let sampleCount = min(audioChunkFrameCount, totalSampleCount - sampleOffset)
              guard
                let sampleBuffer = self.newAudioSampleBuffer(
                  from: audio, tensorSampleOffset: sampleOffset,
                  presentationSampleOffset: sampleOffset, sampleCount: sampleCount,
                  sampleRate: audioSampleRate)
              else {
                cancelAndComplete(DrawThingsCLIError.videoEncodeFailed(outputURL.path))
                return
              }
              guard audioWriterInput.append(sampleBuffer) else {
                cancelAndComplete(
                  writer.error ?? DrawThingsCLIError.videoEncodeFailed(outputURL.path))
                return
              }
              sampleOffset += sampleCount
            } else {
              if !audioInputFinished {
                audioWriterInput.markAsFinished()
                audioInputFinished = true
              }
              finishWritingIfReady()
              break
            }
          }
        }
      }

      semaphore.wait()
      if let finishError { throw finishError }
      return outputURL.path
    #else
      throw DrawThingsCLIError.unsupportedVideoOutput(outputURL.path)
    #endif
  }

  #if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
    private func audioTensorShape(_ tensor: Tensor<Float>) throws -> Int {
      let shape = tensor.shape
      guard shape.count == 2, shape[0] == 2, shape[1] > 0 else {
        throw DrawThingsCLIError.invalidAudioTensorShape("\(shape)")
      }
      return shape[1]
    }

    private func audioSettings(
      for containerExtension: String, sampleRate: Double
    ) -> [String: Any] {
      if containerExtension == "mp4" {
        // AAC caps the allowed bitrate per sample rate; 192kbps is invalid at 16kHz.
        let bitRate = min(192_000, Int(sampleRate) * 6)
        return [
          AVFormatIDKey: kAudioFormatMPEG4AAC,
          AVSampleRateKey: sampleRate,
          AVNumberOfChannelsKey: 2,
          AVEncoderBitRateKey: bitRate,
        ]
      }
      return [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: sampleRate,
        AVNumberOfChannelsKey: 2,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
      ]
    }

    private func videoSettings(
      for format: VideoExportFormat, width: Int, height: Int
    ) -> [String: Any] {
      switch format {
      case .prores4444:
        return [
          AVVideoCodecKey: AVVideoCodecType.proRes4444.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
        ]
      case .prores422hq:
        return [
          AVVideoCodecKey: AVVideoCodecType.proRes422HQ.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
        ]
      case .h264:
        return [
          AVVideoCodecKey: AVVideoCodecType.h264.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
          AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: max(9_500_000, Int((Double(width * height) * 5).rounded())),
            AVVideoProfileLevelKey: AVVideoProfileLevelH264High41,
            AVVideoMaxKeyFrameIntervalKey: 30,
            AVVideoAllowFrameReorderingKey: true,
          ],
        ]
      case .hevc:
        return [
          AVVideoCodecKey: AVVideoCodecType.hevc.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
          AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: max(7_500_000, Int((Double(width * height) * 4).rounded())),
            AVVideoMaxKeyFrameIntervalKey: 30,
            AVVideoAllowFrameReorderingKey: true,
          ],
        ]
      }
    }

    private func newAudioSampleBuffer(
      from audioTensor: Tensor<Float>, tensorSampleOffset: Int, presentationSampleOffset: Int,
      sampleCount: Int, sampleRate: Double
    ) -> CMSampleBuffer? {
      guard sampleCount > 0 else { return nil }

      let channelCount = 2
      let byteCount = sampleCount * channelCount * MemoryLayout<Float32>.size
      var interleaved = [Float32](repeating: 0, count: sampleCount * channelCount)
      for i in 0..<sampleCount {
        interleaved[i * 2] = max(-1, min(1, audioTensor[0, tensorSampleOffset + i]))
        interleaved[i * 2 + 1] = max(-1, min(1, audioTensor[1, tensorSampleOffset + i]))
      }

      var blockBuffer: CMBlockBuffer?
      let blockBufferStatus = CMBlockBufferCreateWithMemoryBlock(
        allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: byteCount,
        blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
        dataLength: byteCount, flags: 0, blockBufferOut: &blockBuffer)
      guard blockBufferStatus == kCMBlockBufferNoErr, let blockBuffer else {
        return nil
      }

      let replaceStatus = interleaved.withUnsafeBytes { rawBuffer in
        CMBlockBufferReplaceDataBytes(
          with: rawBuffer.baseAddress!, blockBuffer: blockBuffer, offsetIntoDestination: 0,
          dataLength: byteCount)
      }
      guard replaceStatus == kCMBlockBufferNoErr else { return nil }

      let bytesPerFrame = UInt32(channelCount * MemoryLayout<Float32>.size)
      var asbd = AudioStreamBasicDescription(
        mSampleRate: sampleRate,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: bytesPerFrame,
        mFramesPerPacket: 1,
        mBytesPerFrame: bytesPerFrame,
        mChannelsPerFrame: 2,
        mBitsPerChannel: 32,
        mReserved: 0)

      var audioFormatDescription: CMAudioFormatDescription?
      let formatStatus = CMAudioFormatDescriptionCreate(
        allocator: kCFAllocatorDefault, asbd: &asbd, layoutSize: 0, layout: nil,
        magicCookieSize: 0, magicCookie: nil, extensions: nil,
        formatDescriptionOut: &audioFormatDescription)
      guard formatStatus == noErr, let audioFormatDescription else {
        return nil
      }

      let presentationTime = CMTime(
        value: CMTimeValue(presentationSampleOffset),
        timescale: CMTimeScale(max(1, Int(sampleRate.rounded()))))
      var sampleBuffer: CMSampleBuffer?
      let sampleBufferStatus = CMAudioSampleBufferCreateWithPacketDescriptions(
        allocator: kCFAllocatorDefault, dataBuffer: blockBuffer, dataReady: true,
        makeDataReadyCallback: nil, refcon: nil, formatDescription: audioFormatDescription,
        sampleCount: sampleCount, presentationTimeStamp: presentationTime,
        packetDescriptions: nil, sampleBufferOut: &sampleBuffer)
      guard sampleBufferStatus == noErr else { return nil }
      return sampleBuffer
    }
  #endif

  #if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
    private func frameDurationForVideo(frameRate: Double) -> CMTime {
      guard frameRate > 0 else {
        return CMTime(value: 1, timescale: 30)
      }
      if abs(frameRate - frameRate.rounded()) < 1e-12 {
        return CMTime(value: 1, timescale: Int32(frameRate.rounded()))
      }
      let timescale: Int32 = 60_000
      let value = max(1, Int64((Double(timescale) / frameRate).rounded()))
      return CMTime(value: value, timescale: timescale)
    }

    private func populate(
      pixelBuffer: CVPixelBuffer, with tensor: Tensor<FloatType>, expected: ImageTensorShape,
      outputPath: String
    ) throws {
      let shape = try imageTensorShape(tensor)
      guard shape.width == expected.width, shape.height == expected.height, shape.channels >= 3
      else {
        throw DrawThingsCLIError.unsupportedTensorShape("\(tensor.shape)")
      }
      CVPixelBufferLockBaseAddress(pixelBuffer, [])
      defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
      guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw DrawThingsCLIError.videoEncodeFailed(outputPath)
      }
      let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
      let destination = baseAddress.assumingMemoryBound(to: UInt8.self)
      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for y in 0..<shape.height {
          let row = destination.advanced(by: y * bytesPerRow)
          for x in 0..<shape.width {
            let pixelIndex = y * shape.width + x
            let source = pixelIndex * shape.channels
            let destinationIndex = x * 4
            row[destinationIndex] = pixelByte(fp16[source + 2])  // B
            row[destinationIndex + 1] = pixelByte(fp16[source + 1])  // G
            row[destinationIndex + 2] = pixelByte(fp16[source])  // R
            row[destinationIndex + 3] = 255
          }
        }
      }
    }
  #endif
}

private final class RemoteGenerationRunner {
  private let context: DrawThingsCLIContext
  private struct RemoteTimingTracker {
    private let startTime = Date()
    private var lastSamplingStepTime: Date?
    private(set) var samplingStepDurations: [TimeInterval] = []

    mutating func record(signpost: ImageGeneratorSignpost) {
      guard case .sampling = signpost else { return }
      let now = Date()
      if let lastSamplingStepTime {
        samplingStepDurations.append(now.timeIntervalSince(lastSamplingStepTime))
      }
      lastSamplingStepTime = now
    }

    func summary() -> LocalGenerationRunner.GenerationTimingSummary {
      LocalGenerationRunner.GenerationTimingSummary(
        totalGenerationDuration: Date().timeIntervalSince(startTime),
        samplingStepDurations: samplingStepDurations
      )
    }
  }

  private let backendOptions: GenerateBackendOptions

  init(context: DrawThingsCLIContext, backendOptions: GenerateBackendOptions) {
    self.context = context
    self.backendOptions = backendOptions
  }

  func generate(
    prompt: String,
    negativePrompt: String,
    configuration: GenerationConfiguration,
    outputPath: String,
    inputImage: Tensor<FloatType>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    videoFormat: VideoExportFormat?
  ) throws -> LocalGenerationRunner.GenerationRunResult {
    if videoFormat != nil || isVideoOutputPath(outputPath) {
      throw ValidationError("Remote and cloud generation currently support .png output only.")
    }
    let context = self.context
    try context.checkCancellation()
    defer { context.setCancellation(nil) }
    let remote = try configuredRemoteGenerator()
    defer {
      try? remote.client.disconnect()
    }
    let progressPrinter = ProgressBarPrinter(output: context.output)
    var timingTracker = RemoteTimingTracker()
    progressPrinter.update(progress: 0, label: "Connecting...", detail: nil)

    let result = try remote.generator.generate(
      trace: ImageGeneratorTrace(fromBridge: true),
      image: inputImage,
      scaleFactor: 1,
      mask: nil as Tensor<UInt8>?,
      hints: hints,
      text: prompt,
      negativeText: negativePrompt,
      configuration: configuration,
      fileMapping: [:],
      keywords: [],
      cancellation: { context.setCancellation($0) },
      feedback: { signpost, signposts, previewTensor in
        context.updateImageGeneration(
          signpost: signpost, signposts: signposts, preview: previewTensor)
        timingTracker.record(signpost: signpost)
        self.updateProgress(progressPrinter, signpost: signpost, signposts: signposts)
        return !context.isCancelled
      })

    try context.checkCancellation()
    guard let tensors = result.0, !tensors.isEmpty else {
      throw DrawThingsCLIError.generationFailed
    }
    var outputURL = URL(fileURLWithPath: outputPath)
    if outputURL.pathExtension.isEmpty {
      outputURL = outputURL.appendingPathExtension("png")
    }
    guard outputURL.pathExtension.lowercased() == "png" else {
      throw ValidationError("Remote and cloud generation currently support .png output only.")
    }
    let outputPaths = try savePNGOutputs(tensors, outputURL: outputURL)
    progressPrinter.update(progress: 1, label: "Generated", detail: nil)
    return LocalGenerationRunner.GenerationRunResult(
      outputPaths: outputPaths,
      timing: timingTracker.summary()
    )
  }

  private func configuredRemoteGenerator() throws -> (
    generator: RemoteImageGenerator, client: ImageGenerationClientWrapper
  ) {
    let endpoint = try resolvedEndpoint()
    let client = ImageGenerationClientWrapper(
      deviceName: backendOptions.cloudCompute ? "" : CLIIdentity.commandName)
    try client.connect(
      host: endpoint.host,
      port: endpoint.port,
      TLS: endpoint.useTLS,
      hostnameVerification: endpoint.useTLS,
      sharedSecret: endpoint.sharedSecret
    )
    let serverIdentifier = try performHandshake(client: client)
    let generator = RemoteImageGenerator(
      name: CLIIdentity.commandName,
      deviceType: .laptop,
      client: client,
      serverIdentifier: serverIdentifier,
      authenticationHandler: try cloudAuthenticationHandlerIfNeeded(),
      requestExceedLimitHandler: nil
    )
    return (generator, client)
  }

  private func resolvedEndpoint() throws -> (
    host: String, port: Int, useTLS: Bool, sharedSecret: String?
  ) {
    if backendOptions.cloudCompute {
      return (host: "compute.drawthings.ai", port: 443, useTLS: true, sharedSecret: nil)
    }
    guard backendOptions.remote else {
      throw ValidationError("RemoteGenerationRunner requires --remote or --cloud-compute.")
    }
    let remote = try resolvedRemoteEndpoint()
    return (
      host: remote.host,
      port: remote.port,
      useTLS: remote.useTLS,
      sharedSecret: backendOptions.remoteSharedSecret
    )
  }

  private func resolvedRemoteEndpoint() throws -> (host: String, port: Int, useTLS: Bool) {
    guard let remoteURL = backendOptions.remoteURL, !remoteURL.isEmpty else {
      throw ValidationError("--remote-url is required with --remote.")
    }
    let trimmed = remoteURL.trimmingCharacters(in: .whitespacesAndNewlines)
    let candidate =
      trimmed.contains("://")
      ? trimmed : "\(backendOptions.remoteTLS ? "https" : "http")://\(trimmed)"
    guard
      let components = URLComponents(string: candidate),
      let host = components.host,
      !host.isEmpty
    else {
      throw ValidationError("Failed to parse --remote-url '\(remoteURL)'.")
    }
    let scheme = components.scheme?.lowercased()
    let useTLS =
      if scheme == "https" {
        true
      } else if scheme == "http" {
        false
      } else {
        backendOptions.remoteTLS
      }
    return (host: host, port: components.port ?? backendOptions.remotePort, useTLS: useTLS)
  }

  private func performHandshake(client: ImageGenerationClientWrapper) throws -> UInt64 {
    let semaphore = DispatchSemaphore(value: 0)
    var success = false
    var authenticated = false
    var serverIdentifier: UInt64 = 0
    client.echo { echoSuccess, echoAuthenticated, _, _, echoServerIdentifier in
      success = echoSuccess
      authenticated = echoAuthenticated
      serverIdentifier = echoServerIdentifier
      semaphore.signal()
    }
    guard semaphore.wait(timeout: .now() + 10) == .success else {
      throw ValidationError("Remote echo timed out.")
    }
    guard success else {
      throw ValidationError("Remote echo failed.")
    }
    guard authenticated else {
      throw ValidationError("Remote shared secret was rejected.")
    }
    return serverIdentifier
  }

  private func cloudAuthenticationHandlerIfNeeded() throws
    -> (
      (Bool, Data, GenerationConfiguration, Bool, Int, (@escaping () -> Void) -> Void) -> String?
    )?
  {
    let context = self.context
    guard backendOptions.cloudCompute else { return nil }
    let authentication = try context.cloudAuthentication(
      explicitAPIKey: backendOptions.apiKey, explicitBaseURL: backendOptions.cloudAPIBaseURL,
      storedCredentials: DrawThingsCLICredentialsStore.load(),
      prepareIfNeeded: !backendOptions.automaticCloudCompute)
    return { fromBridge, encodedBlob, configuration, hasImage, shuffleCount, cancellation in
      do {
        let shortTermToken = try fetchShortTermToken(
          context: context, apiKey: authentication.apiKey,
          baseURL: authentication.baseURL,
          emitStates: false
        )
        let estimatedComputeUnits = ComputeUnits.from(
          configuration,
          hasImage: hasImage,
          shuffleCount: shuffleCount
        ).map(Double.init)
        return try authenticateCloudRequest(
          shortTermToken: shortTermToken,
          encodedBlob: encodedBlob,
          fromBridge: fromBridge,
          estimatedComputeUnits: estimatedComputeUnits,
          baseURL: authentication.baseURL,
          cancellation: cancellation
        )
      } catch {
        context.handleCloudAuthenticationError(error, hostAPIKey: authentication.hostAPIKey)
        return nil
      }
    }
  }

  private func updateProgress(
    _ progressPrinter: ProgressBarPrinter,
    signpost: ImageGeneratorSignpost,
    signposts: Set<ImageGeneratorSignpost>
  ) {
    let progressText = cliProgressText(signpost: signpost, signposts: signposts)
    let progress: Float
    switch signpost {
    case .sampling(let step):
      let steps = max(1, cliSamplingSteps(signposts: signposts, isSecondPassSampling: false))
      progress = Float(step + 1) / Float(steps)
    case .secondPassSampling(let step):
      let steps = max(1, cliSamplingSteps(signposts: signposts, isSecondPassSampling: true))
      progress = Float(step + 1) / Float(steps)
    case .textEncoded, .imageEncoded, .controlsGenerated:
      progress = 0.1
    case .imageDecoded, .secondPassImageDecoded, .faceRestored, .imageUpscaled,
      .secondPassImageEncoded:
      progress = 0.9
    }
    progressPrinter.update(
      progress: min(0.95, max(0, progress)),
      label: progressText.label,
      detail: progressText.detail
    )
  }

  private func isVideoOutputPath(_ outputPath: String) -> Bool {
    let ext = URL(fileURLWithPath: outputPath).pathExtension.lowercased()
    return ext == "mov" || ext == "mp4"
  }
}

private func authenticateCloudRequest(
  shortTermToken: String,
  encodedBlob: Data,
  fromBridge: Bool,
  estimatedComputeUnits: Double?,
  baseURL: URL,
  cancellation: (@escaping () -> Void) -> Void
) throws -> String? {
  try CLICloudAuthClient(baseURL: baseURL).authenticateGenerationRequest(
    shortTermToken: shortTermToken,
    encodedBlob: encodedBlob,
    fromBridge: fromBridge,
    estimatedComputeUnits: estimatedComputeUnits,
    cancellation: cancellation
  )
}

private func formatDurationForCLI(_ duration: TimeInterval) -> String {
  if duration < 1 {
    return "\(Int((duration * 1000).rounded())) ms"
  }
  return String(format: "%.2f s", duration)
}

private func medianDuration(_ durations: [TimeInterval]) -> TimeInterval {
  let sorted = durations.sorted()
  guard !sorted.isEmpty else { return 0 }
  let midpoint = sorted.count / 2
  if sorted.count.isMultiple(of: 2) {
    return (sorted[midpoint - 1] + sorted[midpoint]) / 2
  }
  return sorted[midpoint]
}

private func printGenerationTimingSummary(
  context: DrawThingsCLIContext, _ summary: LocalGenerationRunner.GenerationTimingSummary
) {
  context.print("Generation timing:")
  context.print(
    "  Total generation time (including model loading): \(formatDurationForCLI(summary.totalGenerationDuration))"
  )
  guard !summary.samplingStepDurations.isEmpty else {
    return
  }
  let averageStepDuration =
    summary.samplingStepDurations.reduce(0, +) / Double(summary.samplingStepDurations.count)
  let medianStepDuration = medianDuration(summary.samplingStepDurations)
  context.print(
    "  Sampling step time (\(summary.samplingStepDurations.count) step(s)): avg \(formatDurationForCLI(averageStepDuration)), median \(formatDurationForCLI(medianStepDuration))"
  )
}

private func cliSamplingSteps(
  signposts: Set<ImageGeneratorSignpost>, isSecondPassSampling: Bool
) -> Int {
  for signpost in signposts {
    switch signpost {
    case .sampling(let steps):
      if !isSecondPassSampling {
        return steps
      }
    case .secondPassSampling(let steps):
      if isSecondPassSampling {
        return steps
      }
    case .faceRestored, .imageDecoded, .imageEncoded, .imageUpscaled, .secondPassImageDecoded,
      .secondPassImageEncoded, .textEncoded, .controlsGenerated:
      continue
    }
  }
  return 0
}

private func cliProgressText(
  signpost: ImageGeneratorSignpost, signposts: Set<ImageGeneratorSignpost>
) -> (label: String, detail: String?) {
  switch signpost {
  case .faceRestored, .secondPassImageDecoded, .imageUpscaled:
    return ("Finishing...", nil)
  case .imageEncoded, .secondPassImageEncoded, .textEncoded, .controlsGenerated:
    return ("Processing...", nil)
  case .sampling(let step):
    let steps = cliSamplingSteps(signposts: signposts, isSecondPassSampling: false)
    if step == steps {
      return ("Finishing...", nil)
    }
    return ("Sampling...", "\(step + 1) / \(steps)")
  case .secondPassSampling(let step):
    let steps = cliSamplingSteps(signposts: signposts, isSecondPassSampling: true)
    if step == steps {
      return ("Finishing...", nil)
    }
    return ("Sampling...", "\(step + 1) / \(steps)")
  case .imageDecoded:
    if signposts.contains(.secondPassImageDecoded) {
      return ("Processing...", nil)
    } else if signposts.contains(.imageUpscaled) {
      return ("Upscaling...", nil)
    }
    return ("Finishing...", nil)
  }
}

private enum TerminalImageRenderer {
  private enum ResolvedProtocol {
    case iterm2
    case kitty
  }

  private struct PixelSize {
    let width: Int
    let height: Int
  }

  private static let iTerm2PayloadLimit = 900_000
  private static let iTerm2MultipartChunkSize = 32_768
  private static let kittyChunkSize = 4_096
  private static var kittyLivePreviewImageID: Int { max(1, Int(getpid())) }
  private static let kittyLivePreviewPlacementID = 1
  final class LivePreviewSession {
    private let context: DrawThingsCLIContext
    private let resolvedProtocol: ResolvedProtocol
    private let modelVersion: ModelVersion
    private let previewColumns: Int
    private let previewRows: Int
    private var didReserveRegion = false
    private var didRenderFinalImage = false

    private init?(
      context: DrawThingsCLIContext, mode: TerminalImageRenderMode,
      protocolChoice: TerminalImageProtocol,
      outputPixelSize: PixelSize, modelVersion: ModelVersion
    ) {
      guard mode != .disabled, context.isStandardOutputTTY else { return nil }
      let environment = context.environment
      guard
        let resolvedProtocol = TerminalImageRenderer.resolvedProtocol(
          for: protocolChoice, environment: environment)
      else {
        return nil
      }
      self.context = context
      self.resolvedProtocol = resolvedProtocol
      self.modelVersion = modelVersion
      let previewSize = TerminalImageRenderer.fittedPreviewSize(
        for: outputPixelSize, environment: environment)
      switch resolvedProtocol {
      case .iterm2:
        self.previewColumns = previewSize.columns
        self.previewRows = TerminalImageRenderer.fittedPreviewRows(
          for: outputPixelSize, columns: previewSize.columns, environment: environment)
      case .kitty:
        self.previewColumns = previewSize.columns
        self.previewRows = previewSize.rows
      }
    }

    func update(tensor: Tensor<FloatType>) {
      let previewImages = ImageConverter.cgImages(
        fromLatent: tensor, canUseTAESD: true, version: modelVersion)
      guard let previewImage = previewImages.0.first else { return }
      switch resolvedProtocol {
      case .iterm2:
        guard previewRows > 0 else { return }
        ensureReservedRegion()
        guard let data = pngData(from: previewImage) else { return }
        clearPreviewRegion(context: context, rows: previewRows)
        renderITerm2(
          context: context, data: data, fileName: "preview.png", columns: previewColumns,
          rows: previewRows)
        moveCursor(context: context, up: previewRows)
        write(context: context, "\r", to: .standardOutput)
      case .kitty:
        guard previewRows > 0 else { return }
        ensureReservedRegion()
        guard let image = rgbaData(from: previewImage) else { return }
        renderKittyLivePreview(
          context: context, data: image.data, size: image.size, columns: previewColumns,
          rows: previewRows)
      }
    }

    func renderFinalImage(path: String) throws -> Bool {
      let fileURL = URL(fileURLWithPath: path)
      switch resolvedProtocol {
      case .iterm2:
        guard previewRows > 0 else { return false }
        ensureReservedRegion()
        let data = try Data(contentsOf: fileURL)
        clearPreviewRegion(context: context, rows: previewRows)
        let finalColumns = TerminalImageRenderer.previewColumns(
          environment: context.environment)
        renderITerm2(
          context: context, data: data, fileName: fileURL.lastPathComponent, columns: finalColumns)
        didRenderFinalImage = true
      case .kitty:
        guard previewRows > 0 else { return false }
        ensureReservedRegion()
        let image = try loadKittyImage(path: fileURL)
        deleteKittyImage(context: context, imageID: kittyLivePreviewImageID)
        renderKittyInline(context: context, data: image.data, size: image.size)
        didRenderFinalImage = true
      }
      return true
    }

    func finish() {
      switch resolvedProtocol {
      case .iterm2:
        guard didReserveRegion else { return }
        if !didRenderFinalImage {
          moveCursor(context: context, down: previewRows)
          write(context: context, "\r", to: .standardOutput)
        }
        didReserveRegion = false
        didRenderFinalImage = false
      case .kitty:
        if didReserveRegion && !didRenderFinalImage {
          moveCursor(context: context, down: previewRows)
          write(context: context, "\r", to: .standardOutput)
        }
        didReserveRegion = false
        didRenderFinalImage = false
      }
    }

    private func ensureReservedRegion() {
      if !didReserveRegion {
        reservePreviewRegion(context: context, rows: previewRows)
        moveCursor(context: context, up: previewRows)
        didReserveRegion = true
      }
    }

    fileprivate static func make(
      context: DrawThingsCLIContext, mode: TerminalImageRenderMode,
      protocolChoice: TerminalImageProtocol,
      configuration: GenerationConfiguration
    ) -> LivePreviewSession? {
      let outputPixelSize = PixelSize(
        width: Int(configuration.startWidth) * 64,
        height: Int(configuration.startHeight) * 64)
      let modelVersion = ModelZoo.versionForModel(configuration.model ?? "")
      return LivePreviewSession(
        context: context, mode: mode, protocolChoice: protocolChoice,
        outputPixelSize: outputPixelSize,
        modelVersion: modelVersion)
    }
  }

  static func validateRequestedOutput(
    context: DrawThingsCLIContext, outputPath: String, mode: TerminalImageRenderMode,
    protocolChoice: TerminalImageProtocol,
    requiresRenderableOutput: Bool
  ) throws {
    guard mode != .disabled else {
      if protocolChoice != .auto {
        throw ValidationError("--terminal-image-protocol requires --terminal-image")
      }
      return
    }
    var outputURL = URL(fileURLWithPath: outputPath)
    if outputURL.pathExtension.isEmpty {
      outputURL = outputURL.appendingPathExtension("png")
    }
    guard outputURL.pathExtension.lowercased() == "png" else {
      throw ValidationError("--terminal-image requires .png output")
    }
    if mode == .explicit && !context.isStandardOutputTTY {
      throw ValidationError("--terminal-image requires stdout to be a TTY")
    }
    if requiresRenderableOutput {
      guard context.isStandardOutputTTY else {
        throw ValidationError(
          "No --output provided and stdout is not a TTY. Pass --output to save a file."
        )
      }
      guard
        resolvedProtocol(for: protocolChoice, environment: context.environment)
          != nil
      else {
        throw ValidationError(
          "No --output provided and no supported terminal image protocol was detected. Pass --output to save a file, or run in iTerm2, kitty, or Ghostty."
        )
      }
    }
  }

  static func renderGeneratedOutputsIfRequested(
    context: DrawThingsCLIContext, _ outputPaths: [String], mode: TerminalImageRenderMode,
    protocolChoice: TerminalImageProtocol
  ) {
    guard mode != .disabled, let firstPath = outputPaths.first else { return }
    guard mode != .automatic || context.isStandardOutputTTY else { return }
    if outputPaths.count > 1 {
      write(
        context: context,
        "Terminal preview: rendering the first PNG only (\(outputPaths.count) files written).\n",
        to: .standardError)
    }
    do {
      try render(context: context, path: firstPath, protocolChoice: protocolChoice)
    } catch {
      let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
      write(context: context, "Terminal preview skipped: \(message)\n", to: .standardError)
    }
  }

  private static func render(
    context: DrawThingsCLIContext, path: String, protocolChoice: TerminalImageProtocol
  ) throws {
    let environment = context.environment
    guard
      let resolvedProtocol = resolvedProtocol(
        for: protocolChoice, environment: environment)
    else {
      throw ValidationError(
        "Could not detect a supported terminal image protocol. Use iTerm2 or a kitty-graphics terminal such as Ghostty, or set --terminal-image-protocol explicitly."
      )
    }
    let fileURL = URL(fileURLWithPath: path)
    let fileName = fileURL.lastPathComponent
    switch resolvedProtocol {
    case .iterm2:
      let data = try Data(contentsOf: fileURL)
      renderITerm2(
        context: context, data: data, fileName: fileName,
        columns: previewColumns(environment: environment))
    case .kitty:
      try renderKitty(context: context, path: fileURL)
    }
  }

  private static func resolvedProtocol(
    for choice: TerminalImageProtocol, environment: [String: String]
  ) -> ResolvedProtocol? {
    switch choice {
    case .iterm2:
      return .iterm2
    case .kitty:
      return .kitty
    case .auto:
      let termProgram = environment["TERM_PROGRAM"]?.lowercased()
      if environment["ITERM_SESSION_ID"] != nil || termProgram == "iterm.app" {
        return .iterm2
      }
      let term = environment["TERM"]?.lowercased() ?? ""
      if environment["KITTY_WINDOW_ID"] != nil || termProgram == "ghostty"
        || term.contains("kitty") || term.contains("ghostty")
      {
        return .kitty
      }
      return nil
    }
  }

  private static func previewColumns(environment: [String: String]) -> Int {
    let rawColumns = Int(environment["COLUMNS"] ?? "") ?? 80
    return max(1, rawColumns - 2)
  }

  private static func previewRows(environment: [String: String]) -> Int {
    let rawRows = Int(environment["LINES"] ?? "") ?? 24
    return max(1, rawRows - 2)
  }

  private static func fittedPreviewRows(
    for imageSize: PixelSize, columns: Int, environment: [String: String]
  ) -> Int {
    guard imageSize.width > 0, imageSize.height > 0 else {
      return min(previewRows(environment: environment), max(1, columns / 2))
    }
    let cellWidth = 1.0
    let cellHeight = 2.0
    let estimatedRows = Int(
      ceil(
        Double(imageSize.height) * Double(columns) * cellWidth
          / (Double(imageSize.width) * cellHeight)))
    return max(1, min(previewRows(environment: environment), estimatedRows))
  }

  private static func fittedPreviewSize(
    for imageSize: PixelSize, environment: [String: String]
  ) -> (columns: Int, rows: Int) {
    let maxColumns = previewColumns(environment: environment)
    let maxRows = previewRows(environment: environment)
    guard maxColumns > 0, maxRows > 0 else { return (1, 1) }
    guard imageSize.width > 0, imageSize.height > 0 else {
      return (max(1, min(maxColumns, maxRows * 2)), maxRows)
    }
    let cellWidth = 1.0
    let cellHeight = 2.0
    let rowsAtMaxColumns =
      Double(imageSize.height) * Double(maxColumns) * cellWidth
      / (Double(imageSize.width) * cellHeight)
    if Int(ceil(rowsAtMaxColumns)) <= maxRows {
      return (max(1, maxColumns), max(1, Int(floor(rowsAtMaxColumns))))
    }
    let columnsAtMaxRows =
      Double(imageSize.width) * Double(maxRows) * cellHeight
      / (Double(imageSize.height) * cellWidth)
    return (max(1, min(maxColumns, Int(floor(columnsAtMaxRows)))), max(1, maxRows))
  }

  private static func renderITerm2(
    context: DrawThingsCLIContext, data: Data, fileName: String, columns: Int, rows: Int? = nil
  ) {
    let encodedName = Data(fileName.utf8).base64EncodedString()
    var arguments =
      "name=\(encodedName);size=\(data.count);width=\(columns);preserveAspectRatio=1;inline=1"
    if let rows, rows > 0 {
      arguments += ";height=\(rows)"
    }
    let encoded = data.base64EncodedString()
    if context.environment["TMUX"] == nil,
      encoded.count + arguments.count < iTerm2PayloadLimit
    {
      write(
        context: context, "\u{1B}]1337;File=\(arguments):\(encoded)\u{07}\n", to: .standardOutput)
      return
    }
    write(context: context, "\u{1B}]1337;MultipartFile=\(arguments)\u{07}", to: .standardOutput)
    for chunk in base64Chunks(encoded, chunkSize: iTerm2MultipartChunkSize) {
      write(context: context, "\u{1B}]1337;FilePart=\(chunk)\u{07}", to: .standardOutput)
    }
    write(context: context, "\u{1B}]1337;FileEnd\u{07}\n", to: .standardOutput)
  }

  private static func renderKittyInline(context: DrawThingsCLIContext, data: Data, size: PixelSize)
  {
    let encoded = data.base64EncodedString()
    let chunks = base64Chunks(encoded, chunkSize: kittyChunkSize)
    for (index, chunk) in chunks.enumerated() {
      let isLast = index == chunks.count - 1
      if index == 0 {
        write(
          context: context,
          "\u{1B}_Ga=T,f=32,s=\(size.width),v=\(size.height),q=2,m=\(isLast ? 0 : 1);\(chunk)\u{1B}\\",
          to: .standardOutput)
      } else {
        write(context: context, "\u{1B}_Gm=\(isLast ? 0 : 1);\(chunk)\u{1B}\\", to: .standardOutput)
      }
    }
    write(context: context, "\r", to: .standardOutput)
  }

  private static func renderKittyLivePreview(
    context: DrawThingsCLIContext, data: Data, size: PixelSize, columns: Int? = nil,
    rows: Int? = nil, moveCursor: Bool = false
  ) {
    transmitKittyImage(context: context, data: data, size: size, imageID: kittyLivePreviewImageID)
    placeKittyImage(
      context: context, imageID: kittyLivePreviewImageID, placementID: kittyLivePreviewPlacementID,
      columns: columns, rows: rows, moveCursor: moveCursor)
  }

  private static func transmitKittyImage(
    context: DrawThingsCLIContext, data: Data, size: PixelSize, imageID: Int
  ) {
    let encoded = data.base64EncodedString()
    let chunks = base64Chunks(encoded, chunkSize: kittyChunkSize)
    for (index, chunk) in chunks.enumerated() {
      let isLast = index == chunks.count - 1
      if index == 0 {
        write(
          context: context,
          "\u{1B}_Ga=t,f=32,s=\(size.width),v=\(size.height),i=\(imageID),q=2,m=\(isLast ? 0 : 1);\(chunk)\u{1B}\\",
          to: .standardOutput)
      } else {
        write(context: context, "\u{1B}_Gm=\(isLast ? 0 : 1);\(chunk)\u{1B}\\", to: .standardOutput)
      }
    }
  }

  private static func placeKittyImage(
    context: DrawThingsCLIContext, imageID: Int, placementID: Int, columns: Int? = nil,
    rows: Int? = nil, moveCursor: Bool
  ) {
    var arguments = "a=p,i=\(imageID),p=\(placementID),C=\(moveCursor ? 0 : 1),q=2"
    if let columns, columns > 0 {
      arguments += ",c=\(columns)"
    }
    if let rows, rows > 0 {
      arguments += ",r=\(rows)"
    }
    write(
      context: context, "\u{1B}_G\(arguments)\u{1B}\\",
      to: .standardOutput)
    if !moveCursor {
      write(context: context, "\r", to: .standardOutput)
    }
  }

  private static func deleteKittyImage(context: DrawThingsCLIContext, imageID: Int) {
    write(context: context, "\u{1B}_Ga=d,i=\(imageID),q=2\u{1B}\\", to: .standardOutput)
  }

  private static func renderKitty(context: DrawThingsCLIContext, path: URL) throws {
    let image = try loadKittyImage(path: path)
    renderKittyInline(context: context, data: image.data, size: image.size)
  }

  private static func reservePreviewRegion(context: DrawThingsCLIContext, rows: Int) {
    guard rows > 0 else { return }
    write(context: context, String(repeating: "\n", count: rows), to: .standardOutput)
  }

  private static func clearPreviewRegion(context: DrawThingsCLIContext, rows: Int) {
    guard rows > 0 else { return }
    for row in 0..<rows {
      write(context: context, "\r\u{1B}[2K", to: .standardOutput)
      if row < rows - 1 {
        write(context: context, "\u{1B}[1B", to: .standardOutput)
      }
    }
    if rows > 1 {
      moveCursor(context: context, up: rows - 1)
    }
    write(context: context, "\r", to: .standardOutput)
  }

  private static func moveCursor(context: DrawThingsCLIContext, up rows: Int) {
    guard rows > 0 else { return }
    write(context: context, "\u{1B}[\(rows)A", to: .standardOutput)
  }

  private static func moveCursor(context: DrawThingsCLIContext, down rows: Int) {
    guard rows > 0 else { return }
    write(context: context, "\u{1B}[\(rows)B", to: .standardOutput)
  }

  private static func inspectImage(path: URL) -> PixelSize? {
    if let image = try? PNGFile.read(path: path.path) {
      return PixelSize(width: image.size.x, height: image.size.y)
    }
    #if canImport(ImageIO)
      guard
        let source = CGImageSourceCreateWithURL(path as CFURL, nil),
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
        let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
        let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue
      else {
        return nil
      }
      return PixelSize(width: width, height: height)
    #else
      return nil
    #endif
  }

  private static func loadKittyImage(path: URL) throws -> (data: Data, size: PixelSize) {
    if let image = try? PNGFile.read(path: path.path) {
      let rgba: [PNG.RGBA<UInt8>] = image.unpack(as: PNG.RGBA<UInt8>.self)
      var bytes = [UInt8]()
      bytes.reserveCapacity(rgba.count * 4)
      for pixel in rgba {
        bytes.append(pixel.r)
        bytes.append(pixel.g)
        bytes.append(pixel.b)
        bytes.append(pixel.a)
      }
      return (Data(bytes), PixelSize(width: image.size.x, height: image.size.y))
    }
    #if canImport(ImageIO) && canImport(CoreGraphics)
      guard
        let source = CGImageSourceCreateWithURL(path as CFURL, nil),
        let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
      else {
        throw DrawThingsCLIError.invalidInputImage(path.path)
      }
      let width = cgImage.width
      let height = cgImage.height
      guard width > 0, height > 0 else {
        throw DrawThingsCLIError.invalidInputImage(path.path)
      }
      var bytes = [UInt8](repeating: 0, count: width * height * 4)
      guard
        let context = CGContext(
          data: &bytes, width: width, height: height, bitsPerComponent: 8,
          bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
      else {
        throw DrawThingsCLIError.invalidInputImage(path.path)
      }
      context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
      return (Data(bytes), PixelSize(width: width, height: height))
    #else
      throw DrawThingsCLIError.invalidInputImage(path.path)
    #endif
  }

  private static func rgbaData(from cgImage: CGImage) -> (data: Data, size: PixelSize)? {
    let width = cgImage.width
    let height = cgImage.height
    guard width > 0, height > 0 else { return nil }
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    guard
      let context = CGContext(
        data: &bytes, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else {
      return nil
    }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    return (Data(bytes), PixelSize(width: width, height: height))
  }

  private static func pngData(from cgImage: CGImage) -> Data? {
    #if canImport(ImageIO)
      guard
        let mutableData = CFDataCreateMutable(nil, 0),
        let destination = CGImageDestinationCreateWithData(
          mutableData, "public.png" as CFString, 1, nil)
      else {
        return nil
      }
      CGImageDestinationAddImage(destination, cgImage, nil)
      guard CGImageDestinationFinalize(destination) else {
        return nil
      }
      return mutableData as Data
    #else
      return nil
    #endif
  }

  private static func base64Chunks(_ encoded: String, chunkSize: Int) -> [Substring] {
    guard !encoded.isEmpty else { return [] }
    var chunks: [Substring] = []
    var start = encoded.startIndex
    while start < encoded.endIndex {
      let end =
        encoded.index(start, offsetBy: chunkSize, limitedBy: encoded.endIndex)
        ?? encoded.endIndex
      chunks.append(encoded[start..<end])
      start = end
    }
    return chunks
  }

  private static func write(
    context: DrawThingsCLIContext, _ value: String, to handle: DrawThingsCLIContext.Output
  ) {
    context.write(value, to: handle)
  }
}

private func printTable(
  context: DrawThingsCLIContext, headers: [String], rows: [[String]], maxWidths: [Int]? = nil
) {
  guard !headers.isEmpty else { return }
  let columnCount = headers.count
  let normalizedRows: [[String]] = rows.map { row in
    (0..<columnCount).map { index in index < row.count ? row[index] : "" }
  }
  var widths = headers.map(\.count)
  for row in normalizedRows {
    for (index, value) in row.enumerated() {
      widths[index] = max(widths[index], value.count)
    }
  }
  if let maxWidths {
    for index in 0..<min(columnCount, maxWidths.count) {
      widths[index] = min(widths[index], maxWidths[index])
    }
  }

  func truncated(_ value: String, width: Int) -> String {
    guard value.count > width else { return value }
    guard width > 3 else { return String(value.prefix(width)) }
    return String(value.prefix(width - 3)) + "..."
  }

  func formattedRow(_ row: [String]) -> String {
    row.enumerated().map { index, rawValue in
      let value = truncated(rawValue, width: widths[index])
      if value.count < widths[index] {
        return value + String(repeating: " ", count: widths[index] - value.count)
      }
      return value
    }.joined(separator: "  ")
  }

  context.print(formattedRow(headers))
  context.print(widths.map { String(repeating: "-", count: $0) }.joined(separator: "  "))
  for row in normalizedRows {
    context.print(formattedRow(row))
  }
}

private func printModelList(
  context: DrawThingsCLIContext, limit: Int? = nil, downloadedOnly: Bool = false,
  modelsDirectory: URL, allowNetwork: Bool = true
) {
  let officialFiles = Set(
    ModelZoo.availableSpecifications
      .filter { $0.remoteApiModelConfig == nil }
      .map(\.file))
  let specs = CommunityModelResolver(modelsDirectory: modelsDirectory).allSpecifications(
    allowNetwork: allowNetwork && !downloadedOnly)
  let filtered = downloadedOnly ? specs.filter { ModelZoo.isModelDownloaded($0) } : specs
  let output = limit.map { Array(filtered.prefix($0)) } ?? filtered
  if output.isEmpty {
    context.print("No models found.")
    return
  }
  let rows = output.map { spec in
    let downloaded = ModelZoo.isModelDownloaded(spec) ? "yes" : "no"
    let source = officialFiles.contains(spec.file) ? "official" : "community"
    let hf = (spec.huggingFaceLink?.isEmpty == false) ? (spec.huggingFaceLink ?? "") : "-"
    return [spec.file, spec.name, source, downloaded, hf]
  }
  printTable(
    context: context, headers: ["MODEL", "NAME", "SOURCE", "DOWNLOADED", "HUGGING_FACE"],
    rows: rows,
    maxWidths: [42, 42, 10, 10, 52])
}

private func printModelResolutionHelp(
  context: DrawThingsCLIContext, limit: Int = 20, modelsDirectory: URL
) {
  printModelList(
    context: context, limit: limit, modelsDirectory: modelsDirectory, allowNetwork: !context.offline
  )
  context.print("Tip: run `\(CLIIdentity.command("models list"))` for the full list.")
}

private func unresolvedModelValidationError(_ input: String) -> ValidationError {
  let suggestions = ModelResolver.suggestions(input, limit: 5)
  guard !suggestions.isEmpty else {
    return ValidationError("Could not resolve --model '\(input)'.")
  }
  let lines = suggestions.map { "  - \($0.file) (\($0.name))" }.joined(separator: "\n")
  return ValidationError("Could not resolve --model '\(input)'.\nClosest matches:\n\(lines)")
}

private func requiredFiles(for configuration: GenerationConfiguration) -> [String] {
  ImageGeneratorUtils.filesToDownload(configuration, keywords: []).map(\.file)
}

private func validateLongCatTemporalFrameCount(_ frames: Int, flag: String) throws {
  guard frames % LongCatAudioConditioningEncoder.vaeScale == 1 else {
    throw ValidationError("\(flag) must be 4k + 1 for LongCat temporal VAE alignment.")
  }
}

private func validateMiniMaxH3TemporalFrameCount(_ frames: Int, flag: String) throws {
  guard frames == 1 || ((5...362).contains(frames) && frames % 17 == 5) else {
    throw ValidationError(
      "\(flag) must be 1, or 17k + 5 in the supported 5...362 range for MiniMax H3.")
  }
}

private func createConfiguration(
  context: DrawThingsCLIContext, modelSpecification: ModelZoo.Specification, steps: Int?,
  cfg: Float?, width: Int?, height: Int?,
  frames: Int?, seed: UInt32?, strength: Float?, configJSON: String?, configFile: String?,
  modelsDirectory: URL
) throws -> ResolvedGenerationConfiguration {
  let resolvedConfiguration = try ConfigurationLoader.load(
    context: context,
    modelSpecification: modelSpecification, configJSON: configJSON, configFile: configFile,
    modelsDirectory: modelsDirectory)
  var builder = GenerationConfigurationBuilder(from: resolvedConfiguration.configuration)
  builder.model = modelSpecification.file
  if let steps {
    guard steps >= 1 else { throw ValidationError("--steps must be >= 1") }
    builder.steps = UInt32(steps)
  }
  if let cfg {
    guard cfg >= 0 else { throw ValidationError("--cfg must be >= 0") }
    builder.guidanceScale = cfg
  }
  if let width {
    guard width % 64 == 0 else {
      throw DrawThingsCLIError.invalidImageDimensions(
        width, height ?? Int(builder.startHeight) * 64)
    }
    builder.startWidth = UInt16(width / 64)
  }
  if let height {
    guard height % 64 == 0 else {
      throw DrawThingsCLIError.invalidImageDimensions(width ?? Int(builder.startWidth) * 64, height)
    }
    builder.startHeight = UInt16(height / 64)
  }
  if let frames {
    guard frames >= 1 else { throw ValidationError("--frames must be >= 1") }
    let version = ModelZoo.versionForModel(modelSpecification.file)
    if version == .longcatVideoAvatar1_5 {
      try validateLongCatTemporalFrameCount(frames, flag: "--frames")
    } else if version == .minimaxH3 {
      try validateMiniMaxH3TemporalFrameCount(frames, flag: "--frames")
    }
    builder.numFrames = UInt32(frames)
  }
  if let seed {
    builder.seed = seed
  }
  if let strength {
    guard (0...1).contains(strength) else { throw ValidationError("--strength must be in [0, 1]") }
    builder.strength = strength
  }
  if ModelZoo.versionForModel(modelSpecification.file) == .longcatVideoAvatar1_5 {
    try validateLongCatTemporalFrameCount(Int(builder.numFrames), flag: "LongCat frame count")
  } else if ModelZoo.versionForModel(modelSpecification.file) == .minimaxH3 {
    try validateMiniMaxH3TemporalFrameCount(Int(builder.numFrames), flag: "MiniMax H3 frame count")
  }
  return ResolvedGenerationConfiguration(
    configuration: builder.build(),
    recommendedNegativePrompt: resolvedConfiguration.recommendedNegativePrompt,
    loraOverrideMapping: resolvedConfiguration.loraOverrideMapping)
}

private func runLoRATraining(context: DrawThingsCLIContext, _ options: LoRATrainCommandOptions)
  throws
{
  let configJSON = options.configurationOverrides.configJSON
  let (config, configJSONDictionary) = try LoRATrainConfigLoader.load(configJSON: configJSON)
  let resumeCheckpoint = try mergedAlias(
    primary: options.output.resume, alias: options.output.trainCheckpoint, primaryFlag: "--resume",
    aliasFlag: "--train-checkpoint"
  ).map(context.path)

  let modelsDirectory = try ModelsDirectoryResolver.resolve(
    context: context,
    path: options.modelAndDataset.modelsDirectoryOptions.modelsDir)
  ModelZoo.isExternalUrlsPreferred = true
  ModelZoo.externalUrls = [modelsDirectory]

  let modelInput = options.modelAndDataset.model ?? config.baseModel
  guard let modelInput else {
    printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
    throw ValidationError("--model is required.")
  }
  guard
    let modelSpecification = ModelResolver.resolve(
      context: context,
      modelInput, modelsDirectory: modelsDirectory)
  else {
    printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
    throw unresolvedModelValidationError(modelInput)
  }

  let datasetDirectory = options.modelAndDataset.dataset.map(context.path)
  guard let datasetDirectory, !datasetDirectory.isEmpty else {
    throw ValidationError("--dataset is required.")
  }

  let output = options.output.output ?? config.name ?? "lora_output"
  let name = options.output.name ?? config.name ?? output
  let trainingSteps = options.training.steps ?? Int(config.trainingSteps)
  guard trainingSteps > 0 else { throw ValidationError("--steps must be > 0") }
  let rank = options.training.rank ?? Int(config.networkDim)
  guard rank > 0 else { throw ValidationError("--rank must be > 0") }
  let loraScale = options.training.scale ?? config.networkScale
  guard loraScale > 0 else { throw ValidationError("--scale must be > 0") }
  let unetLearningRate: ClosedRange<Float> =
    if let learningRate = options.training.learningRate {
      try parseLearningRateRange(learningRate)
    } else {
      min(
        config.unetLearningRate, config.unetLearningRateLowerBound)...max(
          config.unetLearningRate, config.unetLearningRateLowerBound)
    }
  let textModelLearningRate = config.textModelLearningRate
  let gradientAccumulation =
    options.training.gradientAccumulation ?? Int(config.gradientAccumulationSteps)
  guard gradientAccumulation > 0 else {
    throw ValidationError("--gradient-accumulation must be > 0")
  }
  let warmupSteps = options.training.warmupSteps ?? Int(config.warmupSteps)
  let saveEvery = options.training.saveEvery ?? Int(config.saveEveryNSteps)
  let seed = options.training.seed ?? config.seed
  let clipSkip = options.dataAndModel.clipSkip ?? Int(config.clipSkip)
  let noiseOffset = options.dataAndModel.noiseOffset ?? config.noiseOffset
  let captionDropout = options.dataAndModel.captionDropout ?? config.captionDropoutRate
  let maxTextLength = Int(config.maxTextLength)
  let useAspectRatio = options.dataAndModel.useAspectRatio ?? config.useImageAspectRatio
  let guidanceEmbedLowerBound = min(config.guidanceEmbedLowerBound, config.guidanceEmbedUpperBound)
  let guidanceEmbedUpperBound = max(config.guidanceEmbedLowerBound, config.guidanceEmbedUpperBound)
  let guidanceEmbed = guidanceEmbedLowerBound...guidanceEmbedUpperBound
  let denoisingLowerBound = min(max(config.denoisingStart, 0), 1)
  let denoisingUpperBound = min(max(config.denoisingEnd, 0), 1)
  let denoisingTimesteps =
    Int(
      (min(denoisingLowerBound, denoisingUpperBound) * 999).rounded())...Int(
      (max(denoisingLowerBound, denoisingUpperBound) * 999).rounded())
  let orthonormalDown = options.dataAndModel.orthonormalDown ?? config.orthonormalLoraDown
  let cotrainTextModel = options.dataAndModel.cotrainTextModel ?? config.cotrainTextModel
  let cotrainCustomEmbedding = config.cotrainCustomEmbedding
  let customEmbeddingLength = Int(config.customEmbeddingLength)
  let customEmbeddingLearningRate = config.customEmbeddingLearningRate
  let stopEmbeddingTrainingAtStep = Int(config.stopEmbeddingTrainingAtStep)
  let memorySaverFromConfig = try parseMemorySaverValue(
    LoRATrainConfigLoader.memorySaverValue(from: configJSONDictionary))
  let weightsMemoryFromConfig = try parseWeightsMemoryValue(
    LoRATrainConfigLoader.weightsMemoryValue(from: configJSONDictionary))
  let memorySaver =
    try options.configurationOverrides.memorySaver.map { try parseMemorySaver($0) }
    ?? memorySaverFromConfig ?? .balanced
  let weightsMemory =
    try options.configurationOverrides.weightsMemory.map { try parseWeightsMemory($0) }
    ?? weightsMemoryFromConfig ?? .cached
  let shouldDryRun = options.execution.dryRun

  let widthPx =
    options.dataAndModel.width ?? options.dataAndModel.resolution
    ?? Int(config.startWidth) * 64
  let heightPx =
    options.dataAndModel.height ?? options.dataAndModel.resolution
    ?? Int(config.startHeight) * 64
  guard widthPx > 0 && widthPx % 64 == 0 else {
    throw ValidationError("--width must be a positive multiple of 64")
  }
  guard heightPx > 0 && heightPx % 64 == 0 else {
    throw ValidationError("--height must be a positive multiple of 64")
  }
  let imageScale = DeviceCapability.Scale(
    widthScale: UInt16(widthPx / 64), heightScale: UInt16(heightPx / 64))
  let additionalScales: [DeviceCapability.Scale] =
    useAspectRatio
    ? config.additionalScales.map {
      DeviceCapability.Scale(widthScale: $0, heightScale: $0)
    } : []
  let resolvedShift: Float =
    config.resolutionDependentShift
    ? Float(
      ModelZoo.shiftFor(
        (width: imageScale.widthScale, height: imageScale.heightScale),
        version: modelSpecification.version))
    : config.shift
  let powerEMA: ClosedRange<Float>? =
    config.powerEmaLowerBound > 0 && config.powerEmaUpperBound > 0
    ? min(
      config.powerEmaLowerBound, config.powerEmaUpperBound)...max(
        config.powerEmaLowerBound, config.powerEmaUpperBound) : nil
  let stepsBetweenRestarts = Int(config.stepsBetweenRestarts)

  let datasetInputs = try loadTrainingDataset(from: datasetDirectory)
  guard !datasetInputs.isEmpty else {
    throw ValidationError("No dataset images found in '\(datasetDirectory)'.")
  }

  context.print("Models directory: \(modelsDirectory.path)")
  context.print("Model: \(modelSpecification.file)")
  context.print("Dataset: \(datasetDirectory)")
  context.print("Name: \(name)")
  context.print("Training steps: \(trainingSteps)")
  context.print("Rank: \(rank), scale: \(loraScale)")
  context.print("Resolution: \(widthPx)x\(heightPx)")
  context.print("Dataset size: \(datasetInputs.count) sample(s)")
  if shouldDryRun {
    context.print("Dry run completed. Use the same command without --dry-run to start training.")
    return
  }

  let files = ModelZoo.filesToDownload(modelSpecification).map(\.file)
  try ModelDownloader.ensureFiles(
    context: context,
    files, modelsDirectory: modelsDirectory, downloadMissing: options.execution.downloadMissing)

  try context.prepareForLocalModelExecution()
  let tokenizers = createLoRATrainerTokenizers()
  let session = UUID().uuidString
  let useMFA = DeviceCapability.isMFAEnabled.load(ordering: .acquiring)
  let usesFlashAttention: UseFlashAttention =
    if useMFA == 0 {
      .none
    } else if useMFA == 1 {
      DeviceCapability.isMQASupported ? .quantized : .sdpa
    } else {
      .sdpa
    }
  let trainer = LoRATrainer(
    tensorBoard: nil, comment: "\(output)-",
    model: modelSpecification.file, scale: imageScale, additionalScales: additionalScales,
    useImageAspectRatio: useAspectRatio, cotrainTextModel: cotrainTextModel,
    cotrainCustomEmbedding: cotrainCustomEmbedding, clipSkip: clipSkip,
    maxTextLength: maxTextLength, session: session,
    resumeIfPossible: resumeCheckpoint?.isEmpty == false,
    usesFlashAttention: usesFlashAttention,
    imageInspector: inspectTrainingImage,
    imageLoader: { loadTrainingTensor(url: $0, imageWidth: $1, imageHeight: $2) })
  let tokenizerStack: [TextualInversionPoweredTokenizer & Tokenizer]
  switch trainer.version {
  case .v1:
    tokenizerStack = [tokenizers.tokenizerV1]
  case .v2:
    tokenizerStack = [tokenizers.tokenizerV2]
  case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b:
    switch trainer.textEncoderVersion {
    case .chatglm3_6b:
      tokenizerStack = [tokenizers.tokenizerChatGLM3]
    case nil:
      tokenizerStack = [tokenizers.tokenizerV2]
    }
  case .flux1:
    tokenizerStack = [tokenizers.tokenizerV2, tokenizers.tokenizerT5]
  case .flux2_9b, .flux2_4b, .krea2:
    tokenizerStack = [tokenizers.tokenizerQwen3]
  case .qwenImage:
    tokenizerStack = [tokenizers.tokenizerQwen25]
  case .zImage:
    tokenizerStack = [tokenizers.tokenizerQwen3]
  case .ernieImage:
    tokenizerStack = [tokenizers.tokenizerMistral3]
  case .cosmos2_5_2b:
    tokenizerStack = [tokenizers.tokenizerQwen3, tokenizers.tokenizerT5]
  default:
    throw ValidationError("Unsupported LoRA trainer model version: \(trainer.version)")
  }

  guard
    let (dataFrame, zeroCaption) = trainer.prepareDataset(
      inputs: datasetInputs, tokenizers: tokenizerStack,
      customEmbeddingLength: customEmbeddingLength,
      progressHandler: { state, index in
        switch state {
        case .imageEncoding:
          context.print("\rEncoding images: \(index)/\(datasetInputs.count)", terminator: "")
        case .conditionalEncoding:
          context.print("\rEncoding captions: \(index)/\(datasetInputs.count)", terminator: "")
        }
        fflush(context.output)
        return !context.isCancelled
      })
  else {
    throw ValidationError("Failed to prepare training dataset.")
  }
  context.print("")

  let trainableKeys: [String]
  let requestedTrainableLayers = Set(config.trainableLayers)
  let requestedLayerIndices = config.layerIndices.map { Int($0) }
  switch trainer.version {
  case .flux1:
    trainableKeys = flux1TrainableKeys(
      trainableLayers: requestedTrainableLayers, layerIndices: requestedLayerIndices)
  case .flux2_9b, .flux2_4b:
    trainableKeys = flux2TrainableKeys(
      version: trainer.version, trainableLayers: requestedTrainableLayers,
      layerIndices: requestedLayerIndices)
  case .qwenImage:
    trainableKeys = qwenImageTrainableKeys(
      trainableLayers: requestedTrainableLayers, layerIndices: requestedLayerIndices)
  case .krea2:
    trainableKeys = krea2TrainableKeys(
      trainableLayers: requestedTrainableLayers, layerIndices: requestedLayerIndices)
  case .zImage:
    trainableKeys = zImageTrainableKeys(
      trainableLayers: requestedTrainableLayers, layerIndices: requestedLayerIndices)
  case .ernieImage:
    trainableKeys = ernieImageTrainableKeys(
      trainableLayers: requestedTrainableLayers, layerIndices: requestedLayerIndices)
  case .cosmos2_5_2b:
    trainableKeys = cosmosTrainableKeys(
      trainableLayers: requestedTrainableLayers, layerIndices: requestedLayerIndices)
  default:
    trainableKeys = []
  }
  let startTime = Date()
  var lastStep = 0
  let triggerWord = config.triggerWord ?? ""

  let resumingLoRAFile: (String, Int)? = {
    guard let resumeCheckpoint, !resumeCheckpoint.isEmpty else { return nil }
    let fileName = URL(fileURLWithPath: resumeCheckpoint).lastPathComponent
    let nameWithoutSuffix = fileName.replacingOccurrences(of: "_lora_f32.ckpt", with: "")
    guard let stepString = nameWithoutSuffix.split(separator: "_").last,
      let step = Int(stepString)
    else {
      return nil
    }
    return (fileName, step)
  }()

  trainer.train(
    resumingLoRAFile: resumingLoRAFile,
    tokenizers: tokenizerStack, dataFrame: dataFrame, zeroCaption: zeroCaption,
    trainingSteps: trainingSteps, warmupSteps: warmupSteps,
    gradientAccumulationSteps: gradientAccumulation,
    rankOfLoRA: rank, scaleOfLoRA: loraScale,
    textModelLearningRate: textModelLearningRate, unetLearningRate: unetLearningRate,
    stepsBetweenRestarts: stepsBetweenRestarts, seed: seed, trainableKeys: trainableKeys,
    customEmbeddingLength: customEmbeddingLength,
    customEmbeddingLearningRate: customEmbeddingLearningRate,
    stopEmbeddingTrainingAtStep: stopEmbeddingTrainingAtStep,
    resolutionDependentShift: config.resolutionDependentShift, shift: resolvedShift,
    noiseOffset: noiseOffset, guidanceEmbed: guidanceEmbed,
    denoisingTimesteps: denoisingTimesteps,
    captionDropoutRate: captionDropout, orthonormalLoRADown: orthonormalDown,
    powerEMA: powerEMA,
    memorySaver: memorySaver, weightsMemory: weightsMemory
  ) { state, loss, checkpoint in
    switch state {
    case .compile:
      context.print("[LoRA] Compiling computation graph...")
    case .step(let step):
      lastStep = step
      let elapsed = Date().timeIntervalSince(startTime)
      let itPerSec = step > 0 ? Double(step) / elapsed : 0
      context.print(
        "[LoRA] Step \(step)/\(trainingSteps) | Loss: \(String(format: "%.4f", loss)) | \(String(format: "%.2f", itPerSec)) it/s"
      )
      let shouldSave =
        step == trainingSteps || (saveEvery > 0 && step > 0 && step % saveEvery == 0)
      if shouldSave {
        let filename = "\(output)_\(step)_lora_f32.ckpt"
        let outputPath = LoRAZoo.filePathForModelDownloaded(filename)
        checkpoint.makeLoRA(to: outputPath, scale: loraScale)
        let specification = LoRAZoo.Specification(
          name: "\(name) (\(step))", file: filename, prefix: triggerWord, version: trainer.version)
        if Thread.isMainThread {
          LoRAZoo.appendCustomSpecification(specification)
        } else {
          DispatchQueue.main.sync {
            LoRAZoo.appendCustomSpecification(specification)
          }
        }
        context.print("[LoRA] Saved checkpoint: \(filename)")
      }
    }
    fflush(context.output)
    return !context.isCancelled
  }

  try context.checkCancellation()
  let totalTime = Date().timeIntervalSince(startTime)
  context.print("Training complete in \(String(format: "%.1f", totalTime))s.")
  context.print("Final checkpoint: \(output)_\(lastStep)_lora_f32.ckpt")
}

private protocol DrawThingsCLICommand: ParsableCommand {
  mutating func run(context: DrawThingsCLIContext) throws
}

extension DrawThingsCLICommand {
  mutating func run() throws {
    try run(context: .process())
  }
}

public struct DrawThingsCLI: ParsableCommand {
  public init() {}
  public static let configuration = CommandConfiguration(
    commandName: CLIIdentity.commandName,
    abstract: "Local inference and training CLI for Draw Things models.",
    discussion: CLIHelpText.root,
    version: CLIIdentity.version,
    subcommands: [Generate.self, Auth.self, Models.self, Train.self, Completion.self]
  )

  private static let invocationCondition = NSCondition()
  private static var isRunning = false

  /// Executes the same command tree in a process or an embedded shell. Returning
  /// a status is essential: ArgumentParser's `exit` would terminate the host app.
  public static func run(
    arguments: [String], context: DrawThingsCLIContext = .process()
  ) -> Int32 {
    invocationCondition.lock()
    while isRunning && !context.isCancelled {
      invocationCondition.wait(until: Date().addingTimeInterval(0.1))
    }
    if context.isCancelled {
      invocationCondition.unlock()
      return 130
    }
    isRunning = true
    invocationCondition.unlock()
    defer {
      invocationCondition.lock()
      isRunning = false
      invocationCondition.broadcast()
      invocationCondition.unlock()
    }
    defer { context.finishInvocation() }
    context.offline = false
    do {
      var command = try parseAsRoot(arguments)
      if var command = command as? DrawThingsCLICommand {
        let externalUrls = ModelZoo.externalUrls
        let isExternalUrlsPreferred = ModelZoo.isExternalUrlsPreferred
        let overrides = ModelZoo.overrideMapping
        let huggingFaceOverrides = ModelZoo.huggingFaceRepoOverrideMapping
        let loraOverrides = LoRAZoo.overrideMapping
        let cacheUri = DeviceCapability.cacheUri
        let isPartialOffloadPreferred = DeviceCapability.isPartialOffloadPreferred.load(
          ordering: .acquiring)
        let isMFAEnabled = DeviceCapability.isMFAEnabled.load(ordering: .acquiring)
        let isMFAAppleNeuralEngineEnabled = DeviceCapability.isMFAAppleNeuralEngineEnabled.load(
          ordering: .acquiring)
        let isCoreMLSupported = CoreMLModelManager.isCoreMLSupported.load(ordering: .acquiring)
        let isLoRASupported = CoreMLModelManager.isLoRASupported.load(ordering: .acquiring)
        let flags = DynamicGraph.flags
        defer {
          ModelZoo.externalUrls = externalUrls
          ModelZoo.isExternalUrlsPreferred = isExternalUrlsPreferred
          ModelZoo.overrideMapping = overrides
          ModelZoo.huggingFaceRepoOverrideMapping = huggingFaceOverrides
          LoRAZoo.overrideMapping = loraOverrides
          DeviceCapability.cacheUri = cacheUri
          DeviceCapability.isPartialOffloadPreferred.store(
            isPartialOffloadPreferred, ordering: .releasing)
          DeviceCapability.isMFAEnabled.store(isMFAEnabled, ordering: .releasing)
          DeviceCapability.isMFAAppleNeuralEngineEnabled.store(
            isMFAAppleNeuralEngineEnabled, ordering: .releasing)
          CoreMLModelManager.isCoreMLSupported.store(isCoreMLSupported, ordering: .releasing)
          CoreMLModelManager.isLoRASupported.store(isLoRASupported, ordering: .releasing)
          DynamicGraph.flags = flags
        }
        // LoRA training can use the GPU on this calling thread, outside the generation queue.
        // Isolate it from the host's text generator.
        // The fork inside queue.sync reuses this state on the same thread or creates
        // separate state if Dispatch runs generation on another worker.
        try DynamicGraph.fork {
          try command.run(context: context)
        }
      } else {
        try command.run()
      }
      try context.checkCancellation()
      return 0
    } catch {
      if context.isCancelled || error is DrawThingsCLIInvocationError {
        var reportedError = error
        // A cancelled backend may throw its own transport error while unwinding.
        do { try context.checkCancellation() } catch { reportedError = error }
        context.write("\(reportedError.localizedDescription)\n", to: .standardError)
        return context.isCancelled ? 130 : 1
      }
      let status = exitCode(for: error).rawValue
      let message = fullMessage(for: error)
      if !message.isEmpty {
        context.write(message + "\n", to: status == 0 ? .standardOutput : .standardError)
      }
      return status
    }
  }
}

extension DrawThingsCLI {
  struct Generate: DrawThingsCLICommand {
    static let configuration = CommandConfiguration(
      abstract: "Run local inference and save generated output image(s) or video.",
      discussion: CLIHelpText.generate)

    @OptionGroup(title: "Model Resolution") var modelResolution: GenerateModelResolutionOptions
    @OptionGroup(title: "Prompts") var prompts: GeneratePromptOptions
    @OptionGroup(title: "Sampling") var sampling: GenerateSamplingOptions
    @OptionGroup(title: "Configuration Overrides")
    var configurationOverrides: GenerateConfigurationOverrideOptions
    @OptionGroup(title: "Image Input") var imageInput: GenerateImageInputOptions
    @OptionGroup(title: "Audio Video Continuation") var avc: GenerateAVCOptions
    @OptionGroup(title: "Output") var output: GenerateOutputOptions
    @OptionGroup(title: "Execution") var execution: GenerateExecutionOptions
    @OptionGroup(title: "Backend") var backend: GenerateBackendOptions

    mutating func run(context: DrawThingsCLIContext) throws {
      imageInput.image = imageInput.image.map(context.path)
      imageInput.initImage = imageInput.initImage.map(context.path)
      imageInput.inputImage = imageInput.inputImage.map(context.path)
      imageInput.audio = imageInput.audio.map(context.path)
      output.output = output.output.map(context.path)

      let imagePath = try mergedAlias(
        primary: try mergedAlias(
          primary: imageInput.image.first, alias: imageInput.initImage, primaryFlag: "--image",
          aliasFlag: "--init-image"),
        alias: imageInput.inputImage, primaryFlag: "--image", aliasFlag: "--input-image")
      let imagePaths = imagePath.map { [$0] + imageInput.image.dropFirst() } ?? []

      context.offline = execution.offline
      try backend.validate()
      if execution.offline && backend.isRemoteOrCloud {
        throw ValidationError("--offline cannot be combined with --remote or --cloud-compute.")
      }
      if avc.enabled {
        if backend.isRemoteOrCloud {
          throw ValidationError("--avc currently supports only local generation.")
        }
        try runLongCatAvatarAVC(context: context, imagePaths: imagePaths)
        return
      }
      if imageInput.audio != nil && backend.isRemoteOrCloud {
        throw ValidationError("--audio currently supports only local generation.")
      }
      if avc.segmentFrames != nil || avc.condFrames != nil || avc.zeroAudioFeatures {
        throw ValidationError(
          "--segment-frames, --cond-frames, and --zero-audio-features require --avc.")
      }
      let modelsDirectory = try ModelsDirectoryResolver.resolve(
        context: context,
        path: modelResolution.modelsDirectoryOptions.modelsDir)
      ModelZoo.isExternalUrlsPreferred = true
      ModelZoo.externalUrls = [modelsDirectory]
      let writesOutputFile = output.output != nil
      let outputPath =
        output.output
        ?? URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
          "draw-things-cli-preview-\(UUID().uuidString).png"
        ).path
      let terminalImageMode: TerminalImageRenderMode =
        if output.terminalImage {
          .explicit
        } else if output.output == nil {
          .automatic
        } else {
          .disabled
        }
      try validateVideoOutputOptions(outputPath: outputPath, videoFormat: output.videoFormat)
      try TerminalImageRenderer.validateRequestedOutput(
        context: context, outputPath: outputPath, mode: terminalImageMode,
        protocolChoice: output.terminalImageProtocol, requiresRenderableOutput: !writesOutputFile)

      guard let model = modelResolution.model else {
        printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
        throw ValidationError("--model is required.")
      }
      guard
        let modelSpecification = ModelResolver.resolve(
          context: context, model, modelsDirectory: modelsDirectory)
      else {
        printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
        throw unresolvedModelValidationError(model)
      }

      let resolvedConfiguration = try createConfiguration(
        context: context, modelSpecification: modelSpecification, steps: sampling.steps,
        cfg: sampling.cfg,
        width: sampling.width, height: sampling.height, frames: sampling.frames,
        seed: sampling.seed, strength: sampling.strength,
        configJSON: configurationOverrides.configJSON,
        configFile: configurationOverrides.configFile,
        modelsDirectory: modelsDirectory)
      let promptValues = try resolvedPrompts(context: context, prompts)
      let configuration = resolvedConfiguration.configuration
      if modelSpecification.version == .longcatVideoAvatar1_5 {
        guard configuration.guidanceScale == 1 else {
          throw ValidationError("LongCat-Video-Avatar currently requires --cfg 1.")
        }
      }
      let resolvedNegativePrompt =
        promptValues.negative ?? resolvedConfiguration.recommendedNegativePrompt ?? ""
      LoRAZoo.overrideMapping = resolvedConfiguration.loraOverrideMapping
      defer {
        LoRAZoo.overrideMapping = [:]
      }

      // Resolve the host default for this invocation before downloading local weights.
      // Explicit backends and local-only requests never consult the account.
      let outputExtension = URL(fileURLWithPath: outputPath).pathExtension.lowercased()
      if !backend.local && !backend.isRemoteOrCloud && !execution.offline,
        imageInput.audio == nil, output.videoFormat == nil,
        outputExtension.isEmpty || outputExtension == "png",
        !ImageGeneratorUtils.isVideoModel(modelSpecification.version),
        ModelZoo.isBuiltinModel(modelSpecification.file),
        configuration.refinerModel.map(ModelZoo.isBuiltinModel) ?? true,
        configuration.loras.allSatisfy({ $0.file.map(LoRAZoo.isBuiltinLoRA) ?? true }),
        configuration.controls.allSatisfy({ $0.file.map(ControlNetZoo.isBuiltinControl) ?? true }),
        configuration.upscaler.map(UpscalerZoo.isBuiltinUpscaler) ?? true
      {
        backend.cloudCompute = try context.cloudAPIKey(prepareIfNeeded: false) != nil
        backend.automaticCloudCompute = backend.cloudCompute
      }
      context.print(
        backend.cloudCompute
          ? "Backend: cloud-compute" : backend.remote ? "Backend: remote" : "Backend: local")
      let livePreviewEnabled =
        !backend.isRemoteOrCloud && !writesOutputFile && !execution.disablePreview

      var files = requiredFiles(for: configuration)
      var fileMapping = [String: String]()
      if imageInput.audio != nil {
        guard let defaultEncoder = ModelZoo.audioEncoderForModel(modelSpecification.file) else {
          throw ValidationError("--audio supports LongCat-Video-Avatar 1.5 and MiniMax H3 Ref2VA.")
        }
        if modelSpecification.version == .minimaxH3,
          ImageGeneratorUtils.modifierForModel(
            modelSpecification.file, LoRAs: configuration.loras.compactMap { $0.file }) != .ref2va
        {
          throw ValidationError("--audio requires a MiniMax H3 Ref2VA checkpoint.")
        }
        let encoderFile = imageInput.audioEncoderFile ?? defaultEncoder
        if !files.contains(encoderFile) { files.append(encoderFile) }
        fileMapping[defaultEncoder] = modelsDirectory.appendingPathComponent(encoderFile).path
      }

      if imageInput.audio != nil && imagePath == nil
        && modelSpecification.version == .longcatVideoAvatar1_5
      {
        throw ValidationError("--image is required with --audio for LongCat-Video-Avatar.")
      }

      if !backend.isRemoteOrCloud {
        try ModelDownloader.ensureFiles(
          context: context,
          files, modelsDirectory: modelsDirectory, downloadMissing: execution.downloadMissing)
      }
      let generationID = context.beginImageGeneration(
        name: modelSpecification.name, version: modelSpecification.version,
        prompt: promptValues.prompt,
        signposts: ImageGeneratorUtils.expectedSignposts(
          !imagePaths.isEmpty, mask: false, text: promptValues.prompt,
          negativeText: resolvedNegativePrompt, configuration: configuration,
          version: modelSpecification.version, memorizedBy: Set(files)))
      defer { context.finishImageGeneration(generationID) }
      try context.checkCancellation()
      let inputImageTensors = try imagePaths.map {
        try loadInputImageTensor(
          path: $0, imageWidth: Int(configuration.startWidth) * 64,
          imageHeight: Int(configuration.startHeight) * 64,
          preserveAlpha: modelSpecification.version == .qwenImage2_1)
      }
      var hints: [(ControlHintType, [(AnyTensor, Float)])] = []
      if inputImageTensors.count > 1 {
        hints.append((.shuffle, inputImageTensors.dropFirst().map { ($0 as AnyTensor, Float(1)) }))
      }

      if backend.isRemoteOrCloud {
        let runner = RemoteGenerationRunner(context: context, backendOptions: backend)
        let result = try runner.generate(
          prompt: promptValues.prompt,
          negativePrompt: resolvedNegativePrompt,
          configuration: configuration,
          outputPath: outputPath,
          inputImage: inputImageTensors.first,
          hints: hints,
          videoFormat: output.videoFormat
        )
        defer {
          if !writesOutputFile {
            for path in result.outputPaths {
              try? FileManager.default.removeItem(atPath: path)
            }
          }
        }
        if writesOutputFile {
          for path in result.outputPaths {
            context.print("Wrote: \(path)")
          }
        }
        TerminalImageRenderer.renderGeneratedOutputsIfRequested(
          context: context, result.outputPaths,
          mode: terminalImageMode,
          protocolChoice: output.terminalImageProtocol
        )
        printGenerationTimingSummary(context: context, result.timing)
        return
      }

      var audio: Tensor<Float>? = nil
      var fallbackAudio: Tensor<Float>? = nil
      if let audioPath = imageInput.audio {
        let fps = max(
          Int(ModelZoo.framesPerSecondForModel(configuration.model ?? "").rounded()), 1)
        let videoFrames = max(Int(configuration.numFrames), 1)
        let audioInput = try AudioInput(
          contentsOf: audioPath,
          sampleRate: ModelZoo.audioSampleRateForModel(configuration.model ?? ""))
        context.print("Loading audio: \(audioPath)")
        audio = audioInput.waveform
        if modelSpecification.version == .longcatVideoAvatar1_5 {
          // LongCat outputs silent video; H3 generates its own audio from the reference.
          fallbackAudio = audioInput.waveformTensor(
            videoFrames: videoFrames, framesPerSecond: fps)
        }
      }
      let runner = try LocalGenerationRunner(context: context)
      let livePreviewSession =
        livePreviewEnabled
        ? TerminalImageRenderer.LivePreviewSession.make(
          context: context, mode: terminalImageMode, protocolChoice: output.terminalImageProtocol,
          configuration: configuration)
        : nil
      defer {
        livePreviewSession?.finish()
      }
      if let audio {
        hints.append((.audio, [(audio as AnyTensor, Float(1))]))
      }
      context.print("Models directory: \(modelsDirectory.path)")
      let result = try runner.generate(
        prompt: promptValues.prompt, negativePrompt: resolvedNegativePrompt,
        configuration: configuration, outputPath: outputPath, inputImage: inputImageTensors.first,
        videoFormat: output.videoFormat, hints: hints, fallbackAudio: fallbackAudio,
        fileMapping: fileMapping,
        livePreviewSession: livePreviewSession)
      let renderedFinalImageInPlace =
        livePreviewSession.flatMap { session in
          result.outputPaths.first.flatMap { path in
            try? session.renderFinalImage(path: path)
          }
        } ?? false
      livePreviewSession?.finish()
      defer {
        if !writesOutputFile {
          for path in result.outputPaths {
            try? FileManager.default.removeItem(atPath: path)
          }
        }
      }
      if writesOutputFile {
        for path in result.outputPaths {
          context.print("Wrote: \(path)")
        }
      }
      if !renderedFinalImageInPlace {
        TerminalImageRenderer.renderGeneratedOutputsIfRequested(
          context: context, result.outputPaths, mode: terminalImageMode,
          protocolChoice: output.terminalImageProtocol)
      }
      printGenerationTimingSummary(context: context, result.timing)
    }
  }

  struct Auth: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Authentication helpers.",
      subcommands: [Login.self, Logout.self, Token.self, State.self, Status.self]
    )

    struct Status: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract:
          "Report the host's default generation backend without exposing credentials or signing in."
      )

      mutating func run(context: DrawThingsCLIContext) throws {
        // Standalone CLI defaults remain local. Only the embedding account host
        // can select cloud compute automatically; this command performs no network request.
        let hasCredential = try context.cloudAPIKey(prepareIfNeeded: false) != nil
        struct StatusOutput: Encodable {
          let defaultBackend: String
          let cloudCredentialAvailable: Bool
        }
        let data = try JSONEncoder().encode(
          StatusOutput(
            defaultBackend: hasCredential ? "cloud" : "local",
            cloudCredentialAvailable: hasCredential))
        context.print(String(decoding: data, as: UTF8.self))
      }
    }

    struct CloudAuthOptions: ParsableArguments {
      @Option(
        name: .long,
        help: "Draw Things API key. Uses saved credentials or DRAWTHINGS_API_KEY if omitted.")
      var apiKey: String?

      @Option(name: .customLong("cloud-api-base-url"), help: "Cloud API base URL.")
      var cloudAPIBaseURL: String?
    }

    struct Login: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract: "Sign in with Google in your browser and save the returned Draw Things API key."
      )

      @Option(name: .customLong("cloud-api-base-url"), help: "Cloud API base URL.")
      var cloudAPIBaseURL: String?

      @Flag(name: .long, help: "Emit login result as JSON.")
      var json: Bool = false

      mutating func run(context: DrawThingsCLIContext) throws {
        let baseURL: URL
        if let cloudAPIBaseURL {
          guard let parsedURL = URL(string: cloudAPIBaseURL) else {
            throw ValidationError("Invalid --cloud-api-base-url: \(cloudAPIBaseURL)")
          }
          baseURL = parsedURL
        } else {
          baseURL = defaultCloudAPIBaseURL
        }

        if !json {
          context.print("Starting Google sign-in in your browser...")
          context.print(
            "Credentials will be saved to: \(DrawThingsCLICredentialsStore.description())")
        }
        let credentials = try CLICloudGoogleOAuthDesktopFlow.signIn(
          apiBaseURL: baseURL,
          callbackApplicationName: CLIIdentity.commandName,
          listenerQueueLabel: "ai.drawthings.cli.google-oauth"
        )
        try DrawThingsCLICredentialsStore.save(credentials)
        let output = AuthCommandOutput.login(
          credentials: credentials,
          credentialsPath: DrawThingsCLICredentialsStore.description(),
          apiBaseURL: baseURL
        )
        if json {
          let encoder = JSONEncoder()
          encoder.outputFormatting = [.sortedKeys]
          let data = try encoder.encode(output)
          context.print(String(decoding: data, as: UTF8.self))
        } else {
          context.print("Google sign-in complete.")
          context.print("Saved API key to: \(DrawThingsCLICredentialsStore.description())")
        }
      }
    }

    struct Logout: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract: "Remove saved cloud credentials."
      )

      @Flag(name: .long, help: "Emit logout result as JSON.")
      var json: Bool = false

      mutating func run(context: DrawThingsCLIContext) throws {
        try DrawThingsCLICredentialsStore.remove()
        let output = AuthCommandOutput.logout(
          credentialsPath: DrawThingsCLICredentialsStore.description())
        if json {
          let encoder = JSONEncoder()
          encoder.outputFormatting = [.sortedKeys]
          let data = try encoder.encode(output)
          context.print(String(decoding: data, as: UTF8.self))
        } else {
          context.print(
            "Removed saved cloud credentials from: \(DrawThingsCLICredentialsStore.description())")
        }
      }
    }

    struct Token: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract: "Fetch a short-term cloud token to validate credentials."
      )

      @OptionGroup var auth: CloudAuthOptions

      mutating func run(context: DrawThingsCLIContext) throws {
        let authentication = try context.cloudAuthentication(
          explicitAPIKey: auth.apiKey, explicitBaseURL: auth.cloudAPIBaseURL,
          storedCredentials: DrawThingsCLICredentialsStore.load())
        _ = try fetchShortTermToken(
          context: context, apiKey: authentication.apiKey, baseURL: authentication.baseURL,
          emitStates: false)
        context.print("Short-term token fetched successfully.")
        context.print("Authentication test complete.")
      }
    }

    struct State: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract: "Validate auth-state progression."
      )

      @OptionGroup var auth: CloudAuthOptions

      mutating func run(context: DrawThingsCLIContext) throws {
        let authentication = try context.cloudAuthentication(
          explicitAPIKey: auth.apiKey, explicitBaseURL: auth.cloudAPIBaseURL,
          storedCredentials: DrawThingsCLICredentialsStore.load())
        context.print("Cloud authentication configured")
        context.print("  API base URL: \(authentication.baseURL)")
        context.print("  Testing authentication...")
        _ = try fetchShortTermToken(
          context: context, apiKey: authentication.apiKey, baseURL: authentication.baseURL,
          emitStates: true)
        context.print("Auth state validation complete.")
      }
    }
  }
}

extension DrawThingsCLI.Generate {
  // Keep this workflow model-specific because its audio conditioning, temporal alignment, and
  // continuation-frame semantics are defined by LongCat rather than a shared AVC contract.
  private func runLongCatAvatarAVC(context: DrawThingsCLIContext, imagePaths: [String]) throws {
    guard let imagePath = imagePaths.first else {
      throw ValidationError("--image is required with --avc.")
    }
    guard imagePaths.count == 1 else {
      throw ValidationError("--avc accepts exactly one --image.")
    }
    let segmentFrames = avc.segmentFrames ?? 93
    let condFrames = avc.condFrames ?? 13
    guard segmentFrames > condFrames, condFrames > 0 else {
      throw ValidationError("--segment-frames must be greater than --cond-frames")
    }
    try validateLongCatTemporalFrameCount(segmentFrames, flag: "--segment-frames")
    try validateLongCatTemporalFrameCount(condFrames, flag: "--cond-frames")
    if sampling.frames != nil {
      throw ValidationError("--frames cannot be used with --avc; use --segment-frames.")
    }
    guard let outputPath = output.output else {
      throw ValidationError("--output is required with --avc.")
    }
    var outputURL = URL(fileURLWithPath: outputPath)
    if outputURL.pathExtension.isEmpty {
      outputURL = outputURL.appendingPathExtension("mp4")
    }
    try validateVideoOutputOptions(outputPath: outputURL.path, videoFormat: output.videoFormat)
    switch outputURL.pathExtension.lowercased() {
    case "mp4", "mov":
      break
    default:
      throw DrawThingsCLIError.invalidOutputPath(outputURL.path)
    }

    let modelsDirectory = try ModelsDirectoryResolver.resolve(
      context: context,
      path: modelResolution.modelsDirectoryOptions.modelsDir)
    ModelZoo.isExternalUrlsPreferred = true
    ModelZoo.externalUrls = [modelsDirectory]
    guard let model = modelResolution.model else {
      printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
      throw ValidationError("--model is required.")
    }
    guard
      let modelSpecification = ModelResolver.resolve(
        context: context, model, modelsDirectory: modelsDirectory)
    else {
      printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
      throw unresolvedModelValidationError(model)
    }
    guard ModelZoo.versionForModel(modelSpecification.file) == .longcatVideoAvatar1_5 else {
      throw ValidationError("--avc currently supports only LongCat-Video-Avatar 1.5 models.")
    }

    let resolvedConfiguration = try createConfiguration(
      context: context, modelSpecification: modelSpecification, steps: sampling.steps,
      cfg: sampling.cfg,
      width: sampling.width, height: sampling.height, frames: segmentFrames,
      seed: sampling.seed, strength: sampling.strength,
      configJSON: configurationOverrides.configJSON,
      configFile: configurationOverrides.configFile,
      modelsDirectory: modelsDirectory)
    let promptValues = try resolvedPrompts(context: context, prompts)
    let configuration = resolvedConfiguration.configuration
    guard configuration.guidanceScale == 1 else {
      throw ValidationError("LongCat-Video-Avatar AVC currently requires --cfg 1.")
    }
    let resolvedNegativePrompt =
      promptValues.negative ?? resolvedConfiguration.recommendedNegativePrompt ?? ""
    LoRAZoo.overrideMapping = resolvedConfiguration.loraOverrideMapping
    defer {
      LoRAZoo.overrideMapping = [:]
    }

    var files = requiredFiles(for: configuration)
    var fileMapping = [String: String]()
    if let defaultEncoder = ModelZoo.audioEncoderForModel(modelSpecification.file) {
      let encoderFile = imageInput.audioEncoderFile ?? defaultEncoder
      if !avc.zeroAudioFeatures && !files.contains(encoderFile) { files.append(encoderFile) }
      fileMapping[defaultEncoder] = modelsDirectory.appendingPathComponent(encoderFile).path
    }
    try ModelDownloader.ensureFiles(
      context: context,
      files, modelsDirectory: modelsDirectory, downloadMissing: execution.downloadMissing)

    guard let audioPath = imageInput.audio else {
      throw ValidationError("--audio is required with --avc.")
    }
    let generationID = context.beginImageGeneration(
      name: modelSpecification.name, version: modelSpecification.version,
      prompt: promptValues.prompt,
      signposts: ImageGeneratorUtils.expectedSignposts(
        true, mask: false, text: promptValues.prompt, negativeText: resolvedNegativePrompt,
        configuration: configuration, version: modelSpecification.version, memorizedBy: Set(files)))
    defer { context.finishImageGeneration(generationID) }
    try context.checkCancellation()
    let referenceImage = try loadInputImageTensor(
      path: imagePath, imageWidth: Int(configuration.startWidth) * 64,
      imageHeight: Int(configuration.startHeight) * 64)
    let fps = max(Int(ModelZoo.framesPerSecondForModel(configuration.model ?? "").rounded()), 1)
    let audioInput = try AudioInput(
      contentsOf: audioPath,
      sampleRate: ModelZoo.audioSampleRateForModel(configuration.model ?? ""))
    let targetVideoFrames = audioInput.videoFrameCount(framesPerSecond: fps)
    if avc.zeroAudioFeatures {
      context.print("Using zero audio features for validation: \(audioPath)")
    } else {
      context.print("Loading audio: \(audioPath)")
    }
    let fallbackAudio = audioInput.waveformTensor(
      videoFrames: targetVideoFrames, framesPerSecond: fps)

    let runner = try LocalGenerationRunner(context: context)
    context.print("Models directory: \(modelsDirectory.path)")
    let result = try runner.generateLongCatAvatarAVC(
      prompt: promptValues.prompt, negativePrompt: resolvedNegativePrompt,
      configuration: configuration, inputImage: referenceImage, audio: audioInput.waveform,
      conditionFrames: condFrames, fileMapping: fileMapping,
      zeroAudioFeatures: avc.zeroAudioFeatures)
    let outputPaths = try runner.saveOutputs(
      result.images, audio: fallbackAudio, outputPath: outputURL.path, configuration: configuration,
      videoFormat: output.videoFormat)
    for path in outputPaths {
      context.print("Wrote: \(path)")
    }
    printGenerationTimingSummary(context: context, result.timing)
  }
}

extension DrawThingsCLI {
  struct Models: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Model utilities.",
      discussion: CLIHelpText.models,
      subcommands: [List.self, Ensure.self, Import.self]
    )

    struct List: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract: "List available local-inference model mappings.",
        discussion: CLIHelpText.modelList)

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Flag(name: .long, help: "Show downloaded models only.")
      var downloadedOnly: Bool = false

      @Flag(
        name: .long,
        help: "Disable network access and use cached community model catalogs only.")
      var offline: Bool = false

      mutating func run(context: DrawThingsCLIContext) throws {
        context.offline = offline
        let modelsDirectory = try ModelsDirectoryResolver.resolve(
          context: context,
          path: modelsDirectoryOptions.modelsDir)
        ModelZoo.isExternalUrlsPreferred = true
        ModelZoo.externalUrls = [modelsDirectory]
        context.print("Models directory: \(modelsDirectory.path)")
        printModelList(
          context: context, downloadedOnly: downloadedOnly, modelsDirectory: modelsDirectory,
          allowNetwork: !offline)
      }
    }

    struct Ensure: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract: "Ensure model files exist locally (download if missing).",
        discussion: CLIHelpText.modelEnsure)

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Option(name: .shortAndLong, help: modelReferenceHelp)
      var model: String?

      @Flag(name: .long, inversion: .prefixedNo, help: "Include model dependencies.")
      var includeDependencies: Bool = true

      @Flag(
        name: .long,
        help:
          "Disable network access. Uses cached community model catalogs only, and never downloads models."
      )
      var offline: Bool = false

      mutating func run(context: DrawThingsCLIContext) throws {
        context.offline = offline
        let modelsDirectory = try ModelsDirectoryResolver.resolve(
          context: context,
          path: modelsDirectoryOptions.modelsDir)
        ModelZoo.isExternalUrlsPreferred = true
        ModelZoo.externalUrls = [modelsDirectory]

        guard let model else {
          printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
          throw ValidationError("--model is required.")
        }
        guard
          let modelSpecification = ModelResolver.resolve(
            context: context, model, modelsDirectory: modelsDirectory)
        else {
          printModelResolutionHelp(context: context, modelsDirectory: modelsDirectory)
          throw unresolvedModelValidationError(model)
        }
        let files =
          includeDependencies
          ? ModelZoo.filesToDownload(modelSpecification).map(\.file)
          : [modelSpecification.file]
        if files.isEmpty {
          throw ValidationError(
            "Model '\(modelSpecification.file)' has no local downloadable files.")
        }
        try ModelDownloader.ensureFiles(
          context: context,
          files, modelsDirectory: modelsDirectory, downloadMissing: true)
        context.print("Model ready: \(modelSpecification.file)")
      }
    }

    struct Import: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        abstract: "Import a local checkpoint or safetensors artifact.",
        discussion: CLIHelpText.modelImport)

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Argument(help: modelImportArtifactHelp)
      var artifact: String

      @Option(name: .long, help: "Display name for the imported model.")
      var name: String?

      @Option(
        name: .long, help: "Optional trigger word / prefix stored in the model specification.")
      var triggerWord: String?

      @Option(
        name: .long, help: "Optional autoencoder artifact to import alongside the main model.")
      var autoencoder: String?

      @Option(
        name: .long,
        help:
          "Optional text encoder artifact to import when the detected model family supports it.")
      var textEncoder: String?

      @Option(
        name: .customLong("text-encoder-2"),
        help:
          "Optional second text encoder artifact for SDXL Base imports.")
      var textEncoder2: String?

      @Option(name: .long, help: modelImportScaleHelp)
      var scale: Int?

      @Flag(
        name: .long, help: "Inspect and infer the custom model specification without writing files."
      )
      var dryRun: Bool = false

      @Flag(
        name: .long, inversion: .prefixedNo, help: "Auto-download missing companion model files.")
      var downloadMissing: Bool = true

      @Flag(name: .long, help: "Replace existing imported files with the same internal model id.")
      var replace: Bool = false

      @Flag(
        name: .long,
        help:
          "Disable network access. Imports the local artifact only and skips any remote companion downloads."
      )
      var offline: Bool = false

      mutating func run(context: DrawThingsCLIContext) throws {
        artifact = context.path(artifact)
        autoencoder = autoencoder.map(context.path)
        textEncoder = textEncoder.map(context.path)
        textEncoder2 = textEncoder2.map(context.path)

        context.offline = offline
        let modelsDirectory = try ModelsDirectoryResolver.resolve(
          context: context,
          path: modelsDirectoryOptions.modelsDir)
        ModelZoo.isExternalUrlsPreferred = true
        ModelZoo.externalUrls = [modelsDirectory]

        let artifactURL = try resolvedLocalFileURL(artifact)
        let autoencoderURL = try resolvedOptionalLocalFileURL(autoencoder)
        let textEncoderURL = try resolvedOptionalLocalFileURL(textEncoder)
        let textEncoder2URL = try resolvedOptionalLocalFileURL(textEncoder2)

        let displayName =
          name?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
          ? name!.trimmingCharacters(in: .whitespacesAndNewlines)
          : defaultImportedModelDisplayName(for: artifactURL)
        let internalName = Importer.cleanup(
          filename: artifactURL.deletingPathExtension().lastPathComponent)
        guard !internalName.isEmpty else {
          throw ValidationError(
            "Unable to derive a valid internal model id from '\(artifactURL.lastPathComponent)'.")
        }

        let isTextEncoderCustomized = textEncoderURL != nil || textEncoder2URL != nil
        let importer = ModelImporter(
          filePath: artifactURL.path, modelName: internalName,
          isTextEncoderCustomized: isTextEncoderCustomized,
          autoencoderFilePath: autoencoderURL?.path, textEncoderFilePath: textEncoderURL?.path,
          textEncoder2FilePath: textEncoder2URL?.path)

        guard scale == nil || scale! > 0 else {
          throw ValidationError("--scale must be > 0")
        }
        let inspection = try importer.inspect()
        try validateCustomTextEncoderSupport(
          version: inspection.version, textEncoderURL: textEncoderURL,
          textEncoder2URL: textEncoder2URL)
        let expectedFiles = try projectedImportedOutputFiles(
          modelName: internalName, version: inspection.version,
          includeAutoencoder: autoencoderURL != nil,
          includeTextEncoder: textEncoderURL != nil, includeTextEncoder2: textEncoder2URL != nil)
        let existing = existingImportedOutputs(for: expectedFiles)
        if !replace && !existing.isEmpty {
          let lines = existing.map { "  - \($0)" }.joined(separator: "\n")
          throw ValidationError(
            "Refusing to overwrite existing imported files:\n\(lines)\nRe-run with --replace to overwrite them."
          )
        }
        let finetuneScale = UInt16(
          scale
            ?? Int(
              defaultImportScale(
                for: inspection.version, artifactFileName: artifactURL.lastPathComponent)))

        if dryRun {
          let (specification, additionalModels, _) = ModelImporter.inferModelSpecification(
            modelName: displayName,
            fileName: internalName,
            fileNames: expectedFiles,
            modelVersion: inspection.version,
            modifier: inspection.modifier,
            inspectionResult: inspection,
            prefix: importPrefix(triggerWord: triggerWord),
            objective: nil,
            conditioning: nil,
            noiseDiscretization: nil,
            upcastAttention: false,
            finetuneScale: finetuneScale
          )
          context.print("Dry run: no files written.")
          printImportedModelSummary(
            context: context, specification: specification, version: inspection.version,
            modifier: inspection.modifier, importedFiles: expectedFiles,
            dependencyFiles: importedDependencyFiles(
              specification: specification, additionalModels: additionalModels,
              importedFiles: expectedFiles))
          return
        }

        var lastPrintedPercent = -1
        try context.prepareForLocalModelExecution()
        let result = try importer.import { version in
          context.print(
            "Detected: \(ModelZoo.humanReadableNameForVersion(version)) (\(String(describing: version)))"
          )
        } progress: { progress in
          let percent = min(100, max(0, Int((progress * 100).rounded(.down))))
          if percent >= lastPrintedPercent + 5 || percent == 100 {
            context.print("Importing: \(percent)%")
            lastPrintedPercent = percent
          }
        }

        let fileNames = result.0.map { URL(fileURLWithPath: $0).lastPathComponent }
        let (specification, additionalModels, _) = ModelImporter.inferModelSpecification(
          modelName: displayName,
          fileName: internalName,
          fileNames: fileNames,
          modelVersion: result.1,
          modifier: result.2,
          inspectionResult: result.3,
          prefix: importPrefix(triggerWord: triggerWord),
          objective: nil,
          conditioning: nil,
          noiseDiscretization: nil,
          upcastAttention: false,
          finetuneScale: finetuneScale
        )
        let dependencyFiles = importedDependencyFiles(
          specification: specification, additionalModels: additionalModels,
          importedFiles: fileNames)

        var dependencyWarning: String?
        do {
          try ModelDownloader.ensureFiles(
            context: context,
            dependencyFiles, modelsDirectory: modelsDirectory, downloadMissing: downloadMissing)
        } catch {
          dependencyWarning = error.localizedDescription
        }

        ModelZoo.appendCustomSpecification(specification)

        printImportedModelSummary(
          context: context, specification: specification, version: result.1, modifier: result.2,
          importedFiles: fileNames, dependencyFiles: dependencyFiles)
        if let dependencyWarning {
          context.print("")
          context.print("Dependency download warning:")
          context.print(dependencyWarning)
        } else {
          context.print("")
          context.print("Model imported: \(specification.file)")
        }
      }
    }
  }

  struct Train: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Training utilities.",
      discussion: CLIHelpText.train,
      subcommands: [LoRA.self]
    )

    struct LoRA: DrawThingsCLICommand {
      static let configuration = CommandConfiguration(
        commandName: "lora",
        abstract: "Train a LoRA model locally.",
        discussion: CLIHelpText.loraTrain)

      @OptionGroup var options: LoRATrainCommandOptions

      mutating func run(context: DrawThingsCLIContext) throws {
        context.offline = options.execution.offline
        try runLoRATraining(context: context, options)
      }
    }
  }

  struct Completion: DrawThingsCLICommand {
    static let configuration = CommandConfiguration(
      abstract: "Generate shell completion scripts.",
      discussion: CLIHelpText.completion
    )

    enum Shell: String, ExpressibleByArgument {
      case bash
      case zsh
      case fish

      var completionShell: CompletionShell {
        switch self {
        case .bash:
          return .bash
        case .zsh:
          return .zsh
        case .fish:
          return .fish
        }
      }
    }

    @Argument(help: "Shell to generate completions for.")
    var shell: Shell

    @Option(name: .shortAndLong, help: "Write the completion script to a file instead of stdout.")
    var output: String?

    mutating func run(context: DrawThingsCLIContext) throws {
      let script = DrawThingsCLI.completionScript(for: shell.completionShell)
      if let output {
        try script.write(toFile: context.path(output), atomically: true, encoding: .utf8)
      } else {
        context.print(script, terminator: "")
      }
    }
  }
}

private func mergedAlias(
  primary: String?, alias: String?, primaryFlag: String, aliasFlag: String
) throws -> String? {
  if let primary, let alias, primary != alias {
    throw ValidationError("Use only one of \(primaryFlag) or \(aliasFlag)")
  }
  return primary ?? alias
}

private func loadTextOption(
  context: DrawThingsCLIContext, inline: String?, filePath: String?, inlineFlag: String,
  fileFlag: String
) throws -> String? {
  if inline != nil && filePath != nil {
    throw ValidationError("Use only one of \(inlineFlag) or \(fileFlag)")
  }
  if let inline {
    return inline
  }
  guard let filePath else {
    return nil
  }
  if filePath == "-" {
    let data = try context.readInput()
    guard let text = String(data: data, encoding: .utf8) else {
      throw ValidationError("Failed to read UTF-8 text from stdin for \(fileFlag)")
    }
    return text
  }
  return try String(contentsOfFile: context.path(filePath), encoding: .utf8)
}

private func resolvedPrompts(context: DrawThingsCLIContext, _ options: GeneratePromptOptions) throws
  -> (
    prompt: String, negative: String?
  )
{
  if options.promptFile == "-", options.negativePromptFile == "-" {
    throw ValidationError("Use stdin for only one of --prompt-file or --negative-prompt-file")
  }
  let prompt =
    try loadTextOption(
      context: context, inline: options.prompt, filePath: options.promptFile,
      inlineFlag: "--prompt",
      fileFlag: "--prompt-file") ?? ""
  let negative =
    try loadTextOption(
      context: context, inline: options.negativePrompt, filePath: options.negativePromptFile,
      inlineFlag: "--negative-prompt", fileFlag: "--negative-prompt-file")
  return (prompt, negative)
}

private func validateVideoOutputOptions(outputPath: String, videoFormat: VideoExportFormat?) throws
{
  guard let videoFormat else { return }
  var url = URL(fileURLWithPath: outputPath)
  if url.pathExtension.isEmpty {
    url = url.appendingPathExtension("png")
  }
  switch url.pathExtension.lowercased() {
  case "mov":
    return
  case "mp4":
    if videoFormat == .prores4444 || videoFormat == .prores422hq {
      throw ValidationError("ProRes video formats require .mov output")
    }
  case "png":
    throw ValidationError("--video-format can only be used with .mov or .mp4 output")
  default:
    return
  }
}

private func loadInputImageTensor(
  path: String, imageWidth: Int, imageHeight: Int, preserveAlpha: Bool = false
) throws
  -> Tensor<FloatType>
{
  let filePath = URL(fileURLWithPath: path).standardizedFileURL.path
  guard FileManager.default.fileExists(atPath: filePath) else {
    throw DrawThingsCLIError.invalidInputImagePath(filePath)
  }
  guard
    let (tensor, _, _, _) = loadTrainingTensor(
      url: URL(fileURLWithPath: filePath), imageWidth: imageWidth, imageHeight: imageHeight,
      preserveAlpha: preserveAlpha)
  else {
    throw DrawThingsCLIError.invalidInputImage(filePath)
  }
  return tensor
}

private func combinedTimingSummary(
  _ summaries: [LocalGenerationRunner.GenerationTimingSummary]
) -> LocalGenerationRunner.GenerationTimingSummary {
  LocalGenerationRunner.GenerationTimingSummary(
    totalGenerationDuration: summaries.reduce(0) { $0 + $1.totalGenerationDuration },
    samplingStepDurations: summaries.flatMap(\.samplingStepDurations))
}

private enum LoRATrainConfigLoader {
  static func load(configJSON: String?) throws -> (LoRATrainingConfiguration, [String: Any]) {
    let base = LoRATrainingConfiguration.default
    guard let configJSON, !configJSON.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      return (base, [:])
    }
    guard let configData = configJSON.data(using: .utf8),
      let configDictionary = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidLoRAConfigurationJSON
    }
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    guard let baseData = try? encoder.encode(base),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: baseData) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidLoRAConfigurationJSON
    }
    for (key, value) in configDictionary {
      mergedDictionary[key] = value
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? decoder.decode(LoRATrainingConfiguration.self, from: mergedData)
    else {
      throw DrawThingsCLIError.invalidLoRAConfigurationJSON
    }
    return (configuration, configDictionary)
  }

  static func memorySaverValue(from configDictionary: [String: Any]) -> Any? {
    configDictionary["memory_saver"] ?? configDictionary["memorySaver"]
  }

  static func weightsMemoryValue(from configDictionary: [String: Any]) -> Any? {
    configDictionary["weights_memory_management"] ?? configDictionary["weightsMemoryManagement"]
      ?? configDictionary["weights_memory"]
  }
}

private struct LoRATrainerTokenizers {
  var tokenizerV1: TextualInversionAttentionCLIPTokenizer
  var tokenizerV2: TextualInversionAttentionCLIPTokenizer
  var tokenizerT5: SentencePieceTokenizer
  var tokenizerChatGLM3: SentencePieceTokenizer
  var tokenizerQwen25: TiktokenTokenizer
  var tokenizerQwen3: TiktokenTokenizer
  var tokenizerMistral3: TiktokenTokenizer
}

private func createLoRATrainerTokenizers() -> LoRATrainerTokenizers {
  let tokenizerV1 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_json,
    merges: BinaryResources.merges_txt,
    textualInversions: [])
  let tokenizerV2 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_16e6_json,
    merges: BinaryResources.bpe_simple_vocab_16e6_txt,
    textualInversions: [])
  let tokenizerT5 = SentencePieceTokenizer(
    data: BinaryResources.t5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerChatGLM3 = SentencePieceTokenizer(
    data: BinaryResources.chatglm3_spiece_model, startToken: nil,
    endToken: nil, tokenShift: 0)
  let tokenizerQwen25 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen2_5_json, merges: BinaryResources.merges_qwen2_5_txt,
    specialTokens: [
      "</tool_call>": 151658, "<tool_call>": 151657, "<|box_end|>": 151649, "<|box_start|>": 151648,
      "<|quad_end|>": 151651, "<|quad_start|>": 151650, "<|vision_end|>": 151653,
      "<|vision_pad|>": 151655, "<|vision_start|>": 151652, "<|image_pad|>": 151655,
      "<|object_ref_end|>": 151647, "<|object_ref_start|>": 151646, "<|im_end|>": 151645,
      "<|im_start|>": 151644, "<|endoftext|>": 151643,
    ])
  let tokenizerQwen3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
    specialTokens: [
      "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
      "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647, "<|box_start|>": 151648,
      "<|box_end|>": 151649, "<|quad_start|>": 151650, "<|quad_end|>": 151651,
      "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|vision_pad|>": 151654,
      "<|image_pad|>": 151655, "<|video_pad|>": 151656, "<tool_call>": 151657,
      "</tool_call>": 151658, "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
      "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662, "<|repo_name|>": 151663,
      "<|file_sep|>": 151664, "<tool_response>": 151665, "</tool_response>": 151666,
      "<think>": 151667, "</think>": 151668, "<|boi_token|>": 151669,
      "<|bor_token|>": 151670, "<|eor_token|>": 151671, "<|bot_token|>": 151672,
      "<|tms_token|>": 151673,
    ])
  let tokenizerMistral3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_mistral3_json, merges: BinaryResources.merges_mistral3_txt,
    specialTokens: [
      "<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AVAILABLE_TOOLS]": 5,
      "[/AVAILABLE_TOOLS]": 6, "[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8, "[TOOL_CALLS]": 9,
      "[IMG]": 10, "<pad>": 11, "[IMG_BREAK]": 12, "[IMG_END]": 13, "[PREFIX]": 14,
      "[MIDDLE]": 15, "[SUFFIX]": 16, "[SYSTEM_PROMPT]": 17, "[/SYSTEM_PROMPT]": 18,
      "[TOOL_CONTENT]": 19,
    ], unknownToken: "<unk>", startToken: "<s>", endToken: "</s>")
  return LoRATrainerTokenizers(
    tokenizerV1: tokenizerV1, tokenizerV2: tokenizerV2, tokenizerT5: tokenizerT5,
    tokenizerChatGLM3: tokenizerChatGLM3, tokenizerQwen25: tokenizerQwen25,
    tokenizerQwen3: tokenizerQwen3, tokenizerMistral3: tokenizerMistral3)
}

private func inspectTrainingImage(url: URL) -> (width: Int, height: Int)? {
  if let image = try? PNGFile.read(path: url.path) {
    return (image.size.x, image.size.y)
  }
  #if canImport(ImageIO)
    guard
      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      return nil
    }
    return (cgImage.width, cgImage.height)
  #else
    return nil
  #endif
}

private func loadTrainingTensor(
  url: URL, imageWidth: Int, imageHeight: Int, preserveAlpha: Bool = false
) -> (
  Tensor<FloatType>, (width: Int, height: Int), (top: Int, left: Int), (width: Int, height: Int)
)? {
  if let image = try? PNGFile.read(path: url.path) {
    let rgba: [PNG.RGBA<UInt8>] = image.unpack(as: PNG.RGBA<UInt8>.self)
    let sourceWidth = image.size.x
    let sourceHeight = image.size.y

    var scaledWidth: Int
    var scaledHeight: Int
    if sourceWidth * imageHeight > sourceHeight * imageWidth {
      scaledHeight = imageHeight
      scaledWidth = Int(
        (Double(sourceWidth) * Double(imageHeight) / Double(sourceHeight)).rounded())
    } else {
      scaledWidth = imageWidth
      scaledHeight = Int(
        (Double(sourceHeight) * Double(imageWidth) / Double(sourceWidth)).rounded())
    }

    let offsetX = (scaledWidth - imageWidth) / 2
    let offsetY = (scaledHeight - imageHeight) / 2
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, preserveAlpha ? 4 : 3))
    for y in 0..<imageHeight {
      for x in 0..<imageWidth {
        let srcX = Int(Double(x + offsetX) * Double(sourceWidth) / Double(scaledWidth))
        let srcY = Int(Double(y + offsetY) * Double(sourceHeight) / Double(scaledHeight))
        let clampedX = min(max(srcX, 0), sourceWidth - 1)
        let clampedY = min(max(srcY, 0), sourceHeight - 1)
        let pixel = rgba[clampedY * sourceWidth + clampedX]
        let rgb = preserveAlpha ? 1 : 0
        if preserveAlpha { tensor[0, y, x, 0] = FloatType(Float(pixel.a) / 255) }
        tensor[0, y, x, rgb] = FloatType(Float(pixel.r) / 127.5 - 1)
        tensor[0, y, x, rgb + 1] = FloatType(Float(pixel.g) / 127.5 - 1)
        tensor[0, y, x, rgb + 2] = FloatType(Float(pixel.b) / 127.5 - 1)
      }
    }

    let top = Int(
      (Double(max(scaledWidth - imageWidth, 0))
        * (Double(sourceWidth) / Double(max(scaledWidth, 1)))
        / 2).rounded())
    let left = Int(
      (Double(max(scaledHeight - imageHeight, 0))
        * (Double(sourceHeight) / Double(max(scaledHeight, 1)))
        / 2).rounded())
    let originalWidth = Int(
      (Double(imageWidth) * Double(sourceWidth) / Double(max(scaledWidth, 1))).rounded())
    let originalHeight = Int(
      (Double(imageHeight) * Double(sourceHeight) / Double(max(scaledHeight, 1))).rounded())
    return (
      tensor, (width: originalWidth, height: originalHeight), (top: top, left: left),
      (width: imageWidth, height: imageHeight)
    )
  }

  #if canImport(ImageIO) && canImport(CoreGraphics)
    guard
      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      return nil
    }

    let cgImageWidth = cgImage.width
    let cgImageHeight = cgImage.height

    var scaledWidth: Int
    var scaledHeight: Int
    if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
      scaledHeight = imageHeight
      scaledWidth = Int(
        (Double(cgImageWidth) * Double(imageHeight) / Double(cgImageHeight)).rounded())
    } else {
      scaledWidth = imageWidth
      scaledHeight = Int(
        (Double(cgImageHeight) * Double(imageWidth) / Double(cgImageWidth)).rounded())
    }

    guard
      let bitmapContext = CGContext(
        data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
        bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
          | CGImageAlphaInfo.premultipliedLast.rawValue)
    else {
      return nil
    }
    bitmapContext.interpolationQuality = .high
    bitmapContext.draw(
      cgImage,
      in: CGRect(
        x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2, width: scaledWidth,
        height: scaledHeight))

    guard let data = bitmapContext.data else { return nil }
    let rgba = data.assumingMemoryBound(to: UInt8.self)
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, preserveAlpha ? 4 : 3))
    for y in 0..<imageHeight {
      for x in 0..<imageWidth {
        let i = (y * imageWidth + x) * 4
        let rgb = preserveAlpha ? 1 : 0
        let alpha = Float(rgba[i + 3]) / 255
        if preserveAlpha { tensor[0, y, x, 0] = FloatType(alpha) }
        let scale: Float = preserveAlpha && alpha > 0 ? 1 / alpha : 1
        for channel in 0..<3 {
          tensor[0, y, x, rgb + channel] = FloatType(
            min(255, Float(rgba[i + channel]) * scale) / 127.5 - 1)
        }
      }
    }

    let top = Int(
      (Double(max(scaledWidth - imageWidth, 0))
        * (Double(cgImageWidth) / Double(max(scaledWidth, 1)))
        / 2).rounded())
    let left = Int(
      (Double(max(scaledHeight - imageHeight, 0))
        * (Double(cgImageHeight) / Double(max(scaledHeight, 1)))
        / 2).rounded())
    let originalWidth = Int(
      (Double(imageWidth) * Double(cgImageWidth) / Double(max(scaledWidth, 1))).rounded())
    let originalHeight = Int(
      (Double(imageHeight) * Double(cgImageHeight) / Double(max(scaledHeight, 1))).rounded())
    return (
      tensor, (width: originalWidth, height: originalHeight), (top: top, left: left),
      (width: imageWidth, height: imageHeight)
    )
  #else
    return nil
  #endif
}

private func loadTrainingDataset(from directory: String) throws -> [LoRATrainer.Input] {
  let fileManager = FileManager.default
  let directoryURL = URL(fileURLWithPath: directory)
  let contents = try fileManager.contentsOfDirectory(
    at: directoryURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
  let imageExtensions = Set([
    "png", "jpg", "jpeg", "webp", "heic", "heif", "avif", "bmp", "tif", "tiff",
  ])

  var inputs: [LoRATrainer.Input] = []
  for url in contents.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
    let ext = url.pathExtension.lowercased()
    guard imageExtensions.contains(ext) else { continue }
    let baseName = url.deletingPathExtension().lastPathComponent

    let captionURL = directoryURL.appendingPathComponent("\(baseName).txt")
    var caption = ""
    if fileManager.fileExists(atPath: captionURL.path),
      let text = try? String(contentsOf: captionURL, encoding: .utf8)
    {
      caption = text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    inputs.append(
      LoRATrainer.Input(
        identifier: baseName,
        imageUrl: url,
        caption: caption)
    )
  }
  return inputs
}

private func flux1TrainableKeys(
  trainableLayers: Set<LoRATrainableLayer> = [], layerIndices: [Int] = []
) -> [String] {
  let trainableLayers =
    trainableLayers.isEmpty ? Set(LoRATrainableLayer.allCases) : trainableLayers
  var keys = [String]()
  if trainableLayers.contains(.latentsEmbedder) {
    keys.append("x_embedder-")
  }
  if trainableLayers.contains(.contextEmbedder) {
    keys.append("context_embedder-")
  }
  if trainableLayers.contains(.projectOut) {
    keys.append("linear-0-")
  }
  let layerKeys = [
    trainableLayers.contains(.qkv) ? ["x_q", "x_k", "x_v"] : [],
    trainableLayers.contains(.qkvContext) ? ["c_q", "c_k", "c_v"] : [],
    trainableLayers.contains(.out) ? ["x_o"] : [],
    trainableLayers.contains(.outContext) ? ["c_o"] : [],
    trainableLayers.contains(.feedForward) ? ["x_linear1", "x_out_proj"] : [],
    trainableLayers.contains(.feedForwardContext) ? ["c_linear1", "c_out_proj"] : [],
  ]
  .flatMap { $0 }
  let indices = layerIndices.isEmpty ? Array(0..<(19 + 38)) : layerIndices
  for layerIndex in indices {
    keys.append(contentsOf: layerKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func flux2TrainableKeys(
  version: ModelVersion, trainableLayers: Set<LoRATrainableLayer> = [], layerIndices: [Int] = []
) -> [String] {
  let trainableLayers =
    trainableLayers.isEmpty ? Set(LoRATrainableLayer.allCases) : trainableLayers
  let layers: (Int, Int)
  switch version {
  case .flux2_9b:
    layers = (8, 24)
  case .flux2_4b:
    layers = (5, 20)
  default:
    return []
  }
  var keys = [String]()
  if trainableLayers.contains(.latentsEmbedder) {
    keys.append("x_embedder-")
  }
  if trainableLayers.contains(.contextEmbedder) {
    keys.append("context_embedder-")
  }
  if trainableLayers.contains(.projectOut) {
    keys.append("linear-0-")
  }
  let doubleIndices =
    layerIndices.isEmpty ? Array(0..<layers.0) : layerIndices.filter { $0 >= 0 && $0 < layers.0 }
  let globalIndices =
    layerIndices.isEmpty
    ? Array(0..<(layers.0 + layers.1))
    : layerIndices.filter { $0 >= 0 && $0 < (layers.0 + layers.1) }
  let singleMLPIndices =
    layerIndices.isEmpty
    ? Array(0..<layers.1)
    : layerIndices.compactMap { layerIndex in
      guard layerIndex >= layers.0 && layerIndex < (layers.0 + layers.1) else { return nil }
      return layerIndex - layers.0
    }
  let doubleQKVKeys =
    trainableLayers.contains(.qkv) ? ["x_q", "x_k", "x_v"] : []
  let doubleContextQKVKeys =
    trainableLayers.contains(.qkvContext) ? ["c_q", "c_k", "c_v"] : []
  let doubleOutKeys =
    trainableLayers.contains(.out) ? ["x_o"] : []
  let doubleContextOutKeys =
    trainableLayers.contains(.outContext) ? ["c_o"] : []
  let doubleFFKeys =
    trainableLayers.contains(.feedForward) ? ["x_gate_proj", "x_up_proj", "x_down_proj"] : []
  let doubleContextFFKeys =
    trainableLayers.contains(.feedForwardContext)
    ? ["c_gate_proj", "c_up_proj", "c_down_proj"] : []
  let singleQKVKeys =
    trainableLayers.contains(.qkv) ? ["x_q", "x_k", "x_v"] : []
  let singleOutKeys =
    trainableLayers.contains(.out) ? ["x_o"] : []
  let singleFFKeys =
    trainableLayers.contains(.feedForward) ? ["x_w1", "x_w2", "x_w3"] : []
  for layerIndex in doubleIndices {
    keys.append(
      contentsOf: (doubleQKVKeys + doubleContextQKVKeys + doubleOutKeys + doubleContextOutKeys
        + doubleFFKeys + doubleContextFFKeys).map { "\($0)-\(layerIndex)-" })
  }
  for layerIndex in globalIndices where layerIndex >= layers.0 {
    keys.append(contentsOf: (singleQKVKeys + singleOutKeys).map { "\($0)-\(layerIndex)-" })
  }
  for layerIndex in singleMLPIndices {
    keys.append(contentsOf: singleFFKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func qwenImageTrainableKeys(
  trainableLayers: Set<LoRATrainableLayer> = [], layerIndices: [Int] = []
) -> [String] {
  let trainableLayers =
    trainableLayers.isEmpty ? Set(LoRATrainableLayer.allCases) : trainableLayers
  var keys = [String]()
  if trainableLayers.contains(.latentsEmbedder) {
    keys.append("x_embedder-0-")
  }
  if trainableLayers.contains(.contextEmbedder) {
    keys.append("context_embedder-0-")
  }
  if trainableLayers.contains(.projectOut) {
    keys.append("linear-0-")
  }
  let layerKeys = [
    trainableLayers.contains(.qkv) ? ["x_q", "x_k", "x_v"] : [],
    trainableLayers.contains(.qkvContext) ? ["c_q", "c_k", "c_v"] : [],
    trainableLayers.contains(.out) ? ["x_o"] : [],
    trainableLayers.contains(.outContext) ? ["c_o"] : [],
    trainableLayers.contains(.feedForward) ? ["x_linear1", "x_out_proj"] : [],
    trainableLayers.contains(.feedForwardContext) ? ["c_linear1", "c_out_proj"] : [],
  ]
  .flatMap { $0 }
  let indices = layerIndices.isEmpty ? Array(0..<60) : layerIndices
  for layerIndex in indices where layerIndex >= 0 && layerIndex < 60 {
    keys.append(contentsOf: layerKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func krea2TrainableKeys(
  trainableLayers: Set<LoRATrainableLayer> = [], layerIndices: [Int] = []
) -> [String] {
  let trainableLayers =
    trainableLayers.isEmpty ? Set(LoRATrainableLayer.allCases) : trainableLayers
  var keys = [String]()
  // Krea 2 i8x produces zero LoRA updates when img_in, to_q, or to_k are trainable.
  if trainableLayers.contains(.projectOut) {
    keys.append("final_linear-0-")
  }
  let layerKeys = [
    trainableLayers.contains(.qkv) ? ["to_v"] : [],
    trainableLayers.contains(.out) ? ["to_gate", "to_out"] : [],
    trainableLayers.contains(.feedForward) ? ["gate", "up", "down"] : [],
  ]
  .flatMap { $0 }
  let indices = layerIndices.isEmpty ? Array(0..<28) : layerIndices
  for layerIndex in indices where layerIndex >= 0 && layerIndex < 28 {
    keys.append(contentsOf: layerKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func zImageTrainableKeys(
  trainableLayers: Set<LoRATrainableLayer> = [], layerIndices: [Int] = []
) -> [String] {
  let trainableLayers =
    trainableLayers.isEmpty ? Set(LoRATrainableLayer.allCases) : trainableLayers
  var keys = [String]()
  if trainableLayers.contains(.latentsEmbedder) {
    keys.append("x_embedder-0-")
  }
  if trainableLayers.contains(.projectOut) {
    keys.append("linear_final-0-")
  }
  let noiseRefinerKeys = [
    trainableLayers.contains(.qkv)
      ? ["noise_refiner_q", "noise_refiner_k", "noise_refiner_v"] : [],
    trainableLayers.contains(.out) ? ["noise_refiner_o"] : [],
    trainableLayers.contains(.feedForward)
      ? ["noise_refiner_ffn_gate_proj", "noise_refiner_ffn_up_proj", "noise_refiner_ffn_down_proj"]
      : [],
  ]
  .flatMap { $0 }
  for layerIndex in 0..<2 {
    keys.append(contentsOf: noiseRefinerKeys.map { "\($0)-\(layerIndex)-" })
  }
  let layerKeys = [
    trainableLayers.contains(.qkv) ? ["q", "k", "v"] : [],
    trainableLayers.contains(.out) ? ["o"] : [],
    trainableLayers.contains(.feedForward)
      ? ["ffn_gate_proj", "ffn_up_proj", "ffn_down_proj"] : [],
  ]
  .flatMap { $0 }
  let indices = layerIndices.isEmpty ? Array(0..<30) : layerIndices
  for layerIndex in indices where layerIndex >= 0 && layerIndex < 30 {
    keys.append(contentsOf: layerKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func ernieImageTrainableKeys(
  trainableLayers: Set<LoRATrainableLayer> = [], layerIndices: [Int] = []
) -> [String] {
  let trainableLayers =
    trainableLayers.isEmpty ? Set(LoRATrainableLayer.allCases) : trainableLayers
  var keys = [String]()
  if trainableLayers.contains(.latentsEmbedder) {
    keys.append("x_embedder-0-")
  }
  if trainableLayers.contains(.projectOut) {
    keys.append("linear-0-")
  }
  let layerKeys = [
    trainableLayers.contains(.qkv) ? ["q", "k", "v"] : [],
    trainableLayers.contains(.out) ? ["o"] : [],
    trainableLayers.contains(.feedForward)
      ? ["ffn_gate_proj", "ffn_up_proj", "ffn_down_proj"] : [],
  ]
  .flatMap { $0 }
  let indices = layerIndices.isEmpty ? Array(0..<36) : layerIndices
  for layerIndex in indices where layerIndex >= 0 && layerIndex < 36 {
    keys.append(contentsOf: layerKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func cosmosTrainableKeys(
  trainableLayers: Set<LoRATrainableLayer> = [], layerIndices: [Int] = []
) -> [String] {
  let trainableLayers =
    trainableLayers.isEmpty ? Set(LoRATrainableLayer.allCases) : trainableLayers
  var keys = [String]()
  if trainableLayers.contains(.latentsEmbedder) {
    keys.append("x_embedder-0-")
  }
  if trainableLayers.contains(.projectOut) {
    keys.append("proj_out-0-")
  }
  let layerKeys = [
    trainableLayers.contains(.qkv) ? ["q_proj", "k_proj", "v_proj"] : [],
    trainableLayers.contains(.qkvContext) ? ["c_q_proj"] : [],
    trainableLayers.contains(.out) ? ["out_proj"] : [],
    trainableLayers.contains(.outContext) ? ["c_out_proj"] : [],
    trainableLayers.contains(.feedForward) ? ["fc1", "fc2"] : [],
  ]
  .flatMap { $0 }
  let indices = layerIndices.isEmpty ? Array(0..<28) : layerIndices
  for layerIndex in indices where layerIndex >= 0 && layerIndex < 28 {
    keys.append(contentsOf: layerKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func parseMemorySaver(_ value: String) throws -> LoRATrainer.MemorySaver {
  switch value.lowercased() {
  case "minimal":
    return .minimal
  case "balanced":
    return .balanced
  case "speed":
    return .speed
  case "turbo":
    return .turbo
  default:
    throw ValidationError(
      "Invalid memory saver mode '\(value)'. Use one of: minimal, balanced, speed, turbo")
  }
}

private func parseWeightsMemory(_ value: String) throws -> LoRATrainer.WeightsMemoryManagement {
  switch value.lowercased() {
  case "cached":
    return .cached
  case "justintime":
    return .justInTime
  default:
    throw ValidationError("Invalid weights memory mode '\(value)'. Use one of: cached, justInTime")
  }
}

private func parseLearningRateRange(_ value: String) throws -> ClosedRange<Float> {
  let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
  guard !trimmed.isEmpty else {
    throw ValidationError("Invalid --learning-rate value. Expected float or [low,high].")
  }
  let normalized: String
  if trimmed.hasPrefix("[") && trimmed.hasSuffix("]") && trimmed.count > 2 {
    normalized = String(trimmed.dropFirst().dropLast())
  } else {
    normalized = trimmed
  }

  func parsePair(separator: Character) -> ClosedRange<Float>? {
    let parts = normalized.split(separator: separator, omittingEmptySubsequences: true).map {
      $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    guard parts.count == 2, let first = Float(parts[0]), let second = Float(parts[1]) else {
      return nil
    }
    return min(first, second)...max(first, second)
  }

  let parsedRange =
    parsePair(separator: ",")
    ?? parsePair(separator: ":")
    ?? Float(normalized).map { $0...$0 }
  guard let parsedRange else {
    throw ValidationError(
      "Invalid --learning-rate value '\(value)'. Use float (e.g. 1e-4) or range (e.g. [5e-5,1e-4])."
    )
  }
  guard parsedRange.lowerBound >= 0 else {
    throw ValidationError("--learning-rate must be >= 0")
  }
  return parsedRange
}

private func parseMemorySaverValue(_ value: Any?) throws -> LoRATrainer.MemorySaver? {
  guard let value else { return nil }
  if let numeric = value as? NSNumber,
    let mode = LoRATrainer.MemorySaver(rawValue: numeric.intValue)
  {
    return mode
  }
  if let string = value as? String {
    return try parseMemorySaver(string)
  }
  throw ValidationError(
    "Invalid memory_saver value in --config-json. Use enum raw value or one of: minimal, balanced, speed, turbo."
  )
}

private func parseWeightsMemoryValue(_ value: Any?) throws -> LoRATrainer.WeightsMemoryManagement? {
  guard let value else { return nil }
  if let numeric = value as? NSNumber,
    let mode = LoRATrainer.WeightsMemoryManagement(rawValue: numeric.intValue)
  {
    return mode
  }
  if let string = value as? String {
    return try parseWeightsMemory(string)
  }
  throw ValidationError(
    "Invalid weights_memory_management value in --config-json. Use enum raw value or one of: cached, justInTime."
  )
}
