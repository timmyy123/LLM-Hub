import BinaryResources
import Foundation
import LLM
import NNC
import Tokenizer

public enum Ideogram4PromptJSONExpansionError: LocalizedError {
  case cancelled
  case invalidConfiguration
  case invalidDimensions(width: Int, height: Int)
  case missingModel(String)

  public var errorDescription: String? {
    switch self {
    case .cancelled:
      return "Ideogram 4 prompt expansion was cancelled."
    case .invalidConfiguration:
      return "Ideogram 4 prompt expansion token limits must be positive."
    case .invalidDimensions(let width, let height):
      return "Invalid target dimensions \(width)x\(height)."
    case .missingModel(let path):
      return "Qwen 3.5 checkpoint does not exist at \(path)."
    }
  }
}

public struct Ideogram4PromptJSONExpander {
  public enum Model {
    case qwen3_5_4B
    case qwen3_5_9B

    public var file: String {
      switch self {
      case .qwen3_5_4B:
        return "qwen_3.5_4b_i8x.ckpt"
      case .qwen3_5_9B:
        return "qwen_3.5_9b_i5x.ckpt"
      }
    }

    var configuration: Qwen3_5ModelConfiguration {
      switch self {
      case .qwen3_5_4B:
        return .qwen3_5_4B
      case .qwen3_5_9B:
        return .qwen3_5_9B
      }
    }

    var tieEmbedding: Bool {
      switch self {
      case .qwen3_5_4B:
        return true
      case .qwen3_5_9B:
        return false
      }
    }
  }

  public static let defaultMaxTokens = 8_192
  public static let defaultPrefillChunkSize = 2_048

  private static let eosTokenIDs = Set<Int32>([248_046, 248_044])
  private static let specialTokens: [String: Int32] = [
    "<|endoftext|>": 248_044,
    "<|im_start|>": 248_045,
    "<|im_end|>": 248_046,
    "<|object_ref_start|>": 248_047,
    "<|object_ref_end|>": 248_048,
    "<|box_start|>": 248_049,
    "<|box_end|>": 248_050,
    "<|quad_start|>": 248_051,
    "<|quad_end|>": 248_052,
    "<|vision_start|>": 248_053,
    "<|vision_end|>": 248_054,
    "<|vision_pad|>": 248_055,
    "<|image_pad|>": 248_056,
    "<|video_pad|>": 248_057,
    "<tool_call>": 248_058,
    "</tool_call>": 248_059,
    "<|fim_prefix|>": 248_060,
    "<|fim_middle|>": 248_061,
    "<|fim_suffix|>": 248_062,
    "<|fim_pad|>": 248_063,
    "<|repo_name|>": 248_064,
    "<|file_sep|>": 248_065,
    "<tool_response>": 248_066,
    "</tool_response>": 248_067,
    "<think>": 248_068,
    "</think>": 248_069,
  ]

  private static let tokenizer = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_5_json,
    merges: BinaryResources.merges_qwen3_5_txt,
    specialTokens: specialTokens, unknownToken: "<|endoftext|>",
    startToken: "<|endoftext|>", endToken: "<|im_end|>")

  public let model: Model
  public let modelFilePath: String
  public let maxTokens: Int
  public let prefillChunkSize: Int

  public init(
    model: Model, modelFilePath: String, maxTokens: Int = defaultMaxTokens,
    prefillChunkSize: Int = defaultPrefillChunkSize
  ) {
    self.model = model
    self.modelFilePath = modelFilePath
    self.maxTokens = maxTokens
    self.prefillChunkSize = prefillChunkSize
  }

  public func expand<FloatType: TensorNumeric & BinaryFloatingPoint>(
    _ dataType: FloatType.Type, prompt: String, width: Int, height: Int,
    shouldContinue: () -> Bool = { true },
    partialHandler: ((String) -> Void)? = nil
  ) throws -> String {
    guard FileManager.default.fileExists(atPath: modelFilePath) else {
      throw Ideogram4PromptJSONExpansionError.missingModel(modelFilePath)
    }
    guard width > 0 && height > 0 else {
      throw Ideogram4PromptJSONExpansionError.invalidDimensions(width: width, height: height)
    }
    guard maxTokens > 0, prefillChunkSize > 0 else {
      throw Ideogram4PromptJSONExpansionError.invalidConfiguration
    }
    guard shouldContinue() else {
      throw Ideogram4PromptJSONExpansionError.cancelled
    }
    let promptTokenIDs = Self.promptTokenIDs(prompt: prompt, width: width, height: height)
    let prefixTokenIDs = Self.cachePrefixTokenIDs(prefillChunkSize: prefillChunkSize)
    let contextCache = Qwen3_5PromptJSONPrefixCache<FloatType>(
      modelFilePath: modelFilePath, tokenIds: prefixTokenIDs)
    let graph = DynamicGraph()
    let textGeneration = Qwen3_5TextGeneration<FloatType>(
      filePath: modelFilePath, configuration: model.configuration,
      eosTokenIds: Self.eosTokenIDs, tieEmbedding: model.tieEmbedding)
    var completedResponse: String?
    let generated: [Int32]
    do {
      generated = try textGeneration.generate(
        graph: graph, promptTokenIds: promptTokenIDs, maxTokens: maxTokens,
        prefillChunkSize: prefillChunkSize, prefixCache: contextCache,
        shouldContinue: shouldContinue
      ) { tokenIDs in
        guard shouldContinue() else { return false }
        let response = Self.decode(tokenIDs)
        partialHandler?(response)
        guard Self.isCompleteJSONObject(response) else { return true }
        completedResponse = response
        return false
      }
    } catch Qwen3_5TextGenerationError.cancelled {
      throw Ideogram4PromptJSONExpansionError.cancelled
    }
    guard shouldContinue() else {
      throw Ideogram4PromptJSONExpansionError.cancelled
    }
    let response = completedResponse ?? Self.decode(generated)
    return Self.formatJSON(
      response, aspectRatio: Self.aspectRatio(width: width, height: height)) ?? response
  }

  static func aspectRatio(width: Int, height: Int) -> String {
    precondition(width > 0 && height > 0)
    var lhs = width
    var rhs = height
    while rhs != 0 {
      (lhs, rhs) = (rhs, lhs % rhs)
    }
    let divisor = max(lhs, 1)
    return "\(width / divisor):\(height / divisor)"
  }

  static func userPrompt(prompt: String, width: Int, height: Int) -> String {
    let aspectRatio = aspectRatio(width: width, height: height)
    return """
      TARGET IMAGE ASPECT RATIO: \(aspectRatio) (width:height).

      User idea: \(prompt)
      """
  }

  static func chatPrompt(prompt: String, width: Int, height: Int) -> String {
    let userPrompt = userPrompt(prompt: prompt, width: width, height: height)
    return systemTurn + "<|im_start|>user\n\(userPrompt)<|im_end|>\n"
      + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
  }

  static var systemTurn: String {
    return "<|im_start|>system\n\(Ideogram4MagicSystemPrompt)<|im_end|>\n"
  }

  static func promptTokenIDs(prompt: String, width: Int, height: Int) -> [Int32] {
    return tokenizer.tokenize(
      text: chatPrompt(prompt: prompt, width: width, height: height), addSpecialTokens: false
    ).0
  }

  static func cachePrefixTokenIDs(prefillChunkSize: Int) -> [Int32] {
    precondition(prefillChunkSize > 0)
    let tokenIDs = tokenizer.tokenize(text: systemTurn, addSpecialTokens: false).0
    let count = tokenIDs.count / prefillChunkSize * prefillChunkSize
    return Array(tokenIDs.prefix(count))
  }

  public static func isCompleteJSONObject(_ response: String) -> Bool {
    guard let data = response.data(using: .utf8),
      let object = try? JSONSerialization.jsonObject(with: data), object is [String: Any]
    else { return false }
    return true
  }

  private static func formatJSON(_ response: String, aspectRatio: String) -> String? {
    func encode(_ value: Any) -> String? {
      guard
        let data = try? JSONSerialization.data(
          withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes])
      else { return nil }
      return String(data: data, encoding: .utf8)
    }

    func parse(_ response: String) -> [String: Any]? {
      return try? JSONSerialization.jsonObject(
        with: Data(response.utf8), options: [.json5Allowed, .fragmentsAllowed]) as? [String: Any]
    }

    func completeStructure(_ response: String) -> String? {
      let bytes = Array(response.utf8)
      var closingDelimiters = [UInt8]()
      var quote: UInt8?
      var escaped = false
      var lineComment = false
      var blockComment = false
      var i = 0
      while i < bytes.count {
        let byte = bytes[i]
        if let activeQuote = quote {
          if escaped {
            escaped = false
          } else if byte == 0x5c {  // Backslash.
            escaped = true
          } else if byte == activeQuote {
            quote = nil
          }
        } else if lineComment {
          if byte == 0x0a || byte == 0x0d {
            lineComment = false
          }
        } else if blockComment {
          if byte == 0x2a, i + 1 < bytes.count, bytes[i + 1] == 0x2f {
            blockComment = false
            i += 1
          }
        } else {
          switch byte {
          case 0x22, 0x27:  // Double or single quote.
            quote = byte
          case 0x2f where i + 1 < bytes.count && bytes[i + 1] == 0x2f:
            lineComment = true
            i += 1
          case 0x2f where i + 1 < bytes.count && bytes[i + 1] == 0x2a:
            blockComment = true
            i += 1
          case 0x7b:  // Opening brace.
            closingDelimiters.append(0x7d)
          case 0x5b:  // Opening bracket.
            closingDelimiters.append(0x5d)
          case 0x7d, 0x5d:
            guard closingDelimiters.last == byte else { return nil }
            closingDelimiters.removeLast()
          default:
            break
          }
        }
        i += 1
      }
      guard quote == nil, !blockComment, !closingDelimiters.isEmpty else { return nil }
      let separator = lineComment ? "\n" : ""
      return response + separator + String(decoding: closingDelimiters.reversed(), as: UTF8.self)
    }

    func formatElement(_ value: Any) -> String? {
      guard let element = value as? [String: Any],
        let type = element["type"] as? String,
        let description = element["desc"] as? String,
        let encodedType = encode(type),
        let encodedDescription = encode(description)
      else { return nil }
      var fields = ["\"type\":\(encodedType)"]
      if let bbox = element["bbox"] {
        guard let coordinates = bbox as? [NSNumber], coordinates.count == 4,
          let encodedCoordinates = encode(coordinates)
        else { return nil }
        fields.append("\"bbox\":\(encodedCoordinates)")
      }
      switch type {
      case "obj":
        guard Set(element.keys).isSubset(of: ["type", "bbox", "desc"]) else { return nil }
      case "text":
        guard Set(element.keys).isSubset(of: ["type", "bbox", "text", "desc"]),
          let text = element["text"] as? String, let encodedText = encode(text)
        else { return nil }
        fields.append("\"text\":\(encodedText)")
      default:
        return nil
      }
      fields.append("\"desc\":\(encodedDescription)")
      return "{\(fields.joined(separator: ","))}"
    }

    guard let object = parse(response) ?? completeStructure(response).flatMap(parse),
      Set(object.keys) == [
        "aspect_ratio", "high_level_description", "compositional_deconstruction",
      ],
      object["aspect_ratio"] is String,
      let description = object["high_level_description"] as? String,
      let composition = object["compositional_deconstruction"] as? [String: Any],
      Set(composition.keys) == ["background", "elements"],
      let background = composition["background"] as? String,
      let elements = composition["elements"] as? [Any],
      let encodedAspectRatio = encode(aspectRatio),
      let encodedDescription = encode(description),
      let encodedBackground = encode(background)
    else { return nil }
    var formattedElements = [String]()
    formattedElements.reserveCapacity(elements.count)
    for element in elements {
      guard let formattedElement = formatElement(element) else { return nil }
      formattedElements.append(formattedElement)
    }
    return "{\"aspect_ratio\":\(encodedAspectRatio),"
      + "\"high_level_description\":\(encodedDescription),"
      + "\"compositional_deconstruction\":{"
      + "\"background\":\(encodedBackground),"
      + "\"elements\":[\(formattedElements.joined(separator: ","))]}}"
  }

  private static func decode(_ tokenIDs: [Int32]) -> String {
    let content: [Int32]
    if let endIndex = tokenIDs.firstIndex(where: { eosTokenIDs.contains($0) }) {
      content = Array(tokenIDs[..<endIndex])
    } else {
      content = tokenIDs
    }
    return tokenizer.decode(content).trimmingCharacters(in: .whitespacesAndNewlines)
  }
}
