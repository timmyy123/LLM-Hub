import BinaryResources
import CoreGraphics
@_exported import DataModels
import Dflat
import Diffusion
import Foundation
import ImageGenerator
import ImageIO
import LocalImageGenerator
import ModelZoo
import NNC
import SQLiteDflat
import Tokenizer

#if canImport(UIKit)
  import UIKit
#endif

public enum MediaGenerationKitError: Error, LocalizedError {
  case invalidModelsDirectory
  case localNotConfigured
  case generationFailed(String = "")
  case cancelled
  case remoteNotConfigured
  case unresolvedModelReference(query: String, suggestions: [String])
  case modelNotFoundOnRemote(String)
  case modelNotFoundInCatalog(String)
  case downloadFailed(String)
  case hashMismatch(String)
  case insufficientStorage
  case notConfigured
  case asyncOperationRequired(String)

  public var errorDescription: String? {
    switch self {
    case .invalidModelsDirectory: return "Invalid models directory"
    case .localNotConfigured:
      return "Local features are unavailable in remote-only mode"
    case .generationFailed(let msg):
      return msg.isEmpty ? "Generation failed" : "Generation failed: \(msg)"
    case .cancelled: return "Generation cancelled"
    case .remoteNotConfigured: return "Remote not configured"
    case .unresolvedModelReference(let query, let suggestions):
      guard !suggestions.isEmpty else {
        return "Could not resolve model reference: \(query)"
      }
      let lines = suggestions.map { "  - \($0)" }.joined(separator: "\n")
      return "Could not resolve model reference '\(query)'.\nClosest matches:\n\(lines)"
    case .modelNotFoundOnRemote(let m): return "Model not found on remote server: \(m)"
    case .modelNotFoundInCatalog(let m): return "Model not found in catalog: \(m)"
    case .downloadFailed(let msg): return "Download failed: \(msg)"
    case .hashMismatch(let f): return "Checksum mismatch for: \(f)"
    case .insufficientStorage: return "Insufficient storage"
    case .notConfigured: return "Cloud compute is not configured"
    case .asyncOperationRequired(let operation):
      return
        "Synchronous \(operation) would require a network fetch. Use the async overload instead."
    }
  }
}

internal struct MediaGenerationLocalResources: @unchecked Sendable {
  let queue: DispatchQueue
  let generator: LocalImageGenerator
  let tempDir: String
}

internal enum MediaGenerationExecutionUtilities {
  static func createLocalResources() -> MediaGenerationLocalResources {
    let queue = DispatchQueue(label: "com.drawthings.mediagenerationkit", qos: .userInteractive)
    let tempDir = createTemporaryDirectory()
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
        "</tool_call>": 151658, "<tool_call>": 151657, "<|box_end|>": 151649,
        "<|box_start|>": 151648,
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
        "[IMG]": 10, "<pad>": 11, "[IMG_BREAK]": 12, "[IMG_END]": 13, "[PREFIX]": 14,
        "[MIDDLE]": 15,
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

    return MediaGenerationLocalResources(queue: queue, generator: generator, tempDir: tempDir)
  }

  static func generate(
    on queue: DispatchQueue,
    generator: ImageGenerator,
    prompt: String,
    negativePrompt: String = "",
    configuration: GenerationConfiguration,
    image: Data? = nil,
    mask: Data? = nil,
    hints: [MediaGenerationExecutionHint] = [],
    cancellationBridge: MediaGenerationCancellationBridge?,
    feedback: @escaping (ImageGeneratorSignpost, Tensor<FloatType>?) -> Bool,
    completion: @escaping (Result<[Tensor<FloatType>], MediaGenerationKitError>) -> Void
  ) {
    queue.async {
      let targetWidth = Int(configuration.startWidth) * 64
      let targetHeight = Int(configuration.startHeight) * 64

      var inputTensor: Tensor<FloatType>? = nil
      var imageMask: Tensor<UInt8>? = nil
      if let imageData = image {
        let (tensor, autoMask) = imageDataToTensor(
          imageData, width: targetWidth, height: targetHeight)
        inputTensor = tensor
        imageMask = autoMask
      }

      let maskTensor: Tensor<UInt8>? =
        mask.flatMap {
          maskDataToTensor($0, width: targetWidth, height: targetHeight)
        } ?? imageMask
      let generatorHints: [(ControlHintType, [(AnyTensor, Float)])]
      do {
        generatorHints = try hintsToGeneratorHints(hints)
      } catch let error as MediaGenerationKitError {
        completion(.failure(error))
        return
      } catch {
        completion(.failure(.generationFailed(error.localizedDescription)))
        return
      }

      var wasCancelled = false
      let result: ([Tensor<FloatType>]?, [Tensor<Float>]?, Int)
      do {
        result = try generator.generate(
          trace: ImageGeneratorTrace(fromBridge: true),
          image: inputTensor,
          scaleFactor: 1,
          mask: maskTensor,
          hints: generatorHints,
          text: prompt,
          negativeText: negativePrompt,
          configuration: configuration,
          fileMapping: [:],
          keywords: [],
          cancellation: { cancel in
            cancellationBridge?.setCancellation(cancel)
          },
          feedback: { signpost, signposts, tensor in
            let shouldContinue = feedback(signpost, tensor)
            if !shouldContinue {
              wasCancelled = true
            }
            return shouldContinue
          }
        )
      } catch {
        completion(.failure(.generationFailed("generate threw: \(error)")))
        return
      }

      if wasCancelled {
        completion(.failure(.cancelled))
        return
      }

      guard let tensors = result.0, !tensors.isEmpty else {
        completion(.failure(.generationFailed()))
        return
      }
      completion(.success(tensors))
    }
  }

  private static func hintsToGeneratorHints(_ hints: [MediaGenerationExecutionHint])
    throws -> [(ControlHintType, [(AnyTensor, Float)])]
  {
    try hints.compactMap { hint in
      guard !hint.images.isEmpty else { return nil }
      let tensors: [(AnyTensor, Float)] = try hint.images.map { hintImage in
        guard hintImage.weight.isFinite, hintImage.weight >= 0 else {
          throw MediaGenerationKitError.generationFailed(
            "invalid request: hint image weight must be finite and >= 0")
        }
        guard let tensor = hintImageDataToTensor(hintImage.data) else {
          throw MediaGenerationKitError.generationFailed(
            "invalid request: failed to decode hint image for \(hint.type)")
        }
        return (tensor, hintImage.weight)
      }
      return (hint.type, tensors)
    }
  }

  private static var hasCleanedUpOldDirectories = false
  private static let cleanupLock = NSLock()

  private static func createTemporaryDirectory() -> String {
    let fileManager = FileManager.default
    let parentURL = (fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSTemporaryDirectory())).resolvingSymlinksInPath()
    
    cleanupLock.lock()
    if !hasCleanedUpOldDirectories {
      // Clean up old mgk- directories from previous sessions to prevent leaking disk space
      if let urls = try? fileManager.contentsOfDirectory(at: parentURL, includingPropertiesForKeys: nil) {
        for url in urls {
          let resolved = url.resolvingSymlinksInPath()
          if resolved.lastPathComponent.hasPrefix("mgk-") {
            try? fileManager.removeItem(at: resolved)
          }
        }
      }
      hasCleanedUpOldDirectories = true
    }
    cleanupLock.unlock()
    
    let tempDirURL = parentURL.appendingPathComponent("mgk-\(UUID().uuidString)")
    try! fileManager.createDirectory(
      at: tempDirURL, withIntermediateDirectories: true, attributes: nil)
    return tempDirURL.path
  }

  private static func imageDataToTensor(_ data: Data, width: Int, height: Int) -> (
    Tensor<FloatType>?, Tensor<UInt8>?
  ) {
    guard
      let cgImage = ImageConverter.cgImage(from: data),
      let bitmapContext = ImageConverter.resize(
        from: cgImage, imageWidth: width, imageHeight: height)
    else {
      return (nil, nil)
    }
    let (tensor, mask, _) = ImageConverter.tensor(from: bitmapContext)
    return (tensor, mask)
  }

  private static func maskDataToTensor(_ data: Data, width: Int, height: Int) -> Tensor<UInt8>? {
    guard
      let cgImage = ImageConverter.cgImage(from: data),
      let bitmapContext = ImageConverter.resize(
        from: cgImage, imageWidth: width, imageHeight: height)
    else {
      return nil
    }
    return ImageConverter.grayscaleTensor(from: bitmapContext)
  }

  private static func hintImageDataToTensor(_ data: Data) -> Tensor<FloatType>? {
    guard
      let cgImage = ImageConverter.cgImage(from: data),
      let bitmapContext = ImageConverter.bitmapContext(from: cgImage)
    else {
      return nil
    }
    return ImageConverter.tensor(from: bitmapContext).0
  }
}

extension MediaGenerationEnvironment.Storage {
  func generateLocally(
    prompt: String,
    negativePrompt: String = "",
    configuration: GenerationConfiguration,
    image: Data? = nil,
    mask: Data? = nil,
    hints: [MediaGenerationExecutionHint] = [],
    cancellationBridge: MediaGenerationCancellationBridge?,
    feedback: @escaping (ImageGeneratorSignpost, Tensor<FloatType>?) -> Bool,
    completion: @escaping (Result<[Tensor<FloatType>], MediaGenerationKitError>) -> Void
  ) throws {
    _ = try modelsDirectoryURL()
    let externalUrls = self.externalUrls
    ModelZoo.externalUrls = externalUrls
    let localResources = localResources()
    DeviceCapability.cacheUri = URL(fileURLWithPath: localResources.tempDir)
    MediaGenerationExecutionUtilities.generate(
      on: localResources.queue,
      generator: localResources.generator,
      prompt: prompt,
      negativePrompt: negativePrompt,
      configuration: configuration,
      image: image,
      mask: mask,
      hints: hints,
      cancellationBridge: cancellationBridge,
      feedback: feedback,
      completion: completion
    )
  }
}
