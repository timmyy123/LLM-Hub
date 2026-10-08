import ArgumentParser
import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC

@main
struct Converter: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The model file that is either the safetensors or the PyTorch checkpoint.")
  var file: String
  @Option(name: .shortAndLong, help: "The name of the model.")
  var name: String
  @Option(help: "The autoencoder file.")
  var autoencoderFile: String? = nil
  @Option(name: .shortAndLong, help: "The directory to write the output files to.")
  var outputDirectory: String
  @Flag(help: "Whether to convert the text encoder(s).")
  var textEncoders = false
  @Flag(
    help:
      "Treat the input file as a Whisper-large-v3 audio encoder safetensors and convert it alone.")
  var audioEncoder = false

  private struct Specification: Codable {
    var name: String
    var file: String
    var version: ModelVersion
    var modifier: SamplerModifier
    var textEncoder: String?
    var autoencoder: String?
    var clipEncoder: String?
    var t5Encoder: String?
    var guidanceEmbed: Bool?
  }

  mutating func run() throws {
    ModelZoo.externalUrls = [URL(fileURLWithPath: outputDirectory)]
    if audioEncoder {
      try convertAudioEncoder()
      return
    }
    let fileName = Importer.cleanup(filename: name)
    let importer = ModelImporter(
      filePath: file, modelName: fileName,
      isTextEncoderCustomized: textEncoders,
      autoencoderFilePath: autoencoderFile, textEncoderFilePath: nil, textEncoder2FilePath: nil)
    let (filePaths, modelVersion, modifier, inspectionResult) = try importer.import { _ in
    } progress: { _ in
    }
    let fileNames = filePaths.map { ($0 as NSString).lastPathComponent }
    var autoencoder = fileNames.first {
      $0.hasSuffix("_vae_f16.ckpt")
    }
    var clipEncoder: String? = nil
    let textEncoder: String?
    var t5Encoder: String? = nil
    switch modelVersion {
    case .v1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .v2:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
      }
    case .sdxlBase, .ssd1b:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sd3:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sd3_vae_f16.ckpt"
      }
    case .sd3Large:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sd3_vae_f16.ckpt"
      }
    case .pixart:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sdxlRefiner:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
        } ?? "open_clip_vit_bigg14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .svdI2v:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
        } ?? "open_clip_vit_h14_vision_model_f16.ckpt"
      if clipEncoder == nil {
        clipEncoder = "open_clip_vit_h14_visual_proj_f16.ckpt"
      }
    case .flux1:
      textEncoder = "t5_xxl_encoder_q6p.ckpt"
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_1_vae_f16.ckpt"
      }
    case .hunyuanVideo:
      textEncoder = "llava_llama_3_8b_v1.1_q8p.ckpt"
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "hunyuan_video_vae_f16.ckpt"
      }
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      if modifier == .inpainting {
        clipEncoder = "open_clip_xlm_roberta_large_vit_h14_f16.ckpt"
      }
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        if modelVersion == .wan22_5b {
          autoencoder = "wan_v2.2_video_vae_f16.ckpt"
        } else {
          autoencoder = "wan_v2.1_video_vae_f16.ckpt"
        }
      }
    case .longcatVideoAvatar1_5:
      textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      if autoencoder == nil {
        autoencoder = "wan_v2.1_video_vae_f16.ckpt"
      }
    case .hiDreamI1, .hiDreamO1, .minimaxH3:
      fatalError()
    case .qwenImage, .qwenImage2_1:
      fatalError()
    case .cosmos2_5_2b:
      textEncoder = "qwen_3_0.6b_f16.ckpt"
      clipEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "qwen_image_vae_f16.ckpt"
      }
    case .zImage:
      fatalError()
    case .ernieImage:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_ministral_3_3b_f16.ckpt")
        } ?? "ministral_3_3b_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_2_vae_f16.ckpt"
      }
    case .ideogram4:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_qwen_3_vl_8b_instruct_f16.ckpt")
            || $0.hasSuffix("_qwen_3_vl_8b_instruct_q8p.ckpt")
        } ?? "qwen_3_vl_8b_instruct_q8p.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_2_vae_f16.ckpt"
      }
    case .krea2:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_qwen_3_vl_4b_f16.ckpt") || $0.hasSuffix("_qwen_3_vl_4b_q8p.ckpt")
        } ?? "qwen_3_vl_4b_q8p.ckpt"
      clipEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "qwen_image_vae_f16.ckpt"
      }
    case .seedvr2_3b, .seedvr2_7b:
      textEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "seedvr2_vae_f16.ckpt"
      }
    case .flux2, .flux2_9b, .flux2_4b:
      fatalError()
    case .ltx2, .ltx2_3:
      fatalError()
    case .kandinsky21, .wurstchenStageC, .wurstchenStageB, .auraflow:
      fatalError()
    }
    var specification = Specification(
      name: name, file: "\(fileName)_f16.ckpt", version: modelVersion, modifier: modifier,
      textEncoder: textEncoder, autoencoder: autoencoder, clipEncoder: clipEncoder,
      t5Encoder: t5Encoder)
    if inspectionResult.hasGuidanceEmbed {
      specification.guidanceEmbed = true
    }
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    let jsonData = try jsonEncoder.encode(specification)
    print(String(decoding: jsonData, as: UTF8.self))
  }

  private enum AudioEncoderError: Swift.Error {
    case cannotOpenFile
    case missingTensor(String)
  }

  private func convertAudioEncoder() throws {
    guard let safeTensors = SafeTensors(url: URL(fileURLWithPath: file)) else {
      throw AudioEncoderError.cannotOpenFile
    }
    // transformers saves WhisperForConditionalGeneration keys with a "model." prefix; the
    // WhisperModel layout has none. Normalize to the bare "encoder." layout the mapper uses.
    var stateDict = [String: TensorDescriptor]()
    for (key, value) in safeTensors.states {
      if key.hasPrefix("model.") {
        stateDict[String(key.dropFirst("model.".count))] = value
      } else {
        stateDict[key] = value
      }
    }
    let (mapper, encoder) = WhisperEncoder(
      width: 1_280, layers: 32, heads: 20, melBins: 128, frames: 3_000, intermediateSize: 5_120,
      usesFlashAttention: false)
    let graph = DynamicGraph()
    try graph.withNoGrad {
      let mel = graph.variable(.CPU, .NCHW(1, 128, 1, 3_000), of: FloatType.self)
      encoder.compile(inputs: mel)
      let mapping = mapper(.diffusers)
      let fileName = Importer.cleanup(filename: name)
      let outputPath = URL(fileURLWithPath: outputDirectory)
        .appendingPathComponent("\(fileName)_f16.ckpt").path
      try graph.openStore(outputPath) { store in
        try store.withTransaction {
          for (key, value) in mapping.sorted(by: { $0.key < $1.key }) {
            guard let descriptor = stateDict[key] else {
              throw AudioEncoderError.missingTensor(key)
            }
            try safeTensors.with(descriptor) { tensor in
              var tensor = Tensor<FloatType>(from: tensor)
              let shape = tensor.shape
              if shape.count == 3 {
                // Conv1d weights map onto [O, I, 1, W] 2D convolutions.
                tensor = tensor.reshaped(format: .NCHW, shape: [shape[0], shape[1], 1, shape[2]])
              }
              value.write(
                graph: graph, to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                isDiagonalDown: false
              ) {
                return "__audio_encoder__[\($0)]"
              }
            }
          }
        }
      }
      print("Wrote audio encoder to \(outputPath)")
    }
  }
}
