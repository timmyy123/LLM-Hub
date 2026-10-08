import Foundation
import LLM

public struct LLMZoo: DownloadZoo {
  public struct Specification: Codable, Hashable {
    public let name: String
    public let file: String
    public let engram: String?
    public let version: LLMVersion
    public let deprecated: Bool?
    public let huggingFaceLink: String?

    public init(
      name: String, file: String, engram: String? = nil, version: LLMVersion,
      deprecated: Bool? = nil,
      huggingFaceLink: String? = nil
    ) {
      self.name = name
      self.file = file
      self.engram = engram
      self.version = version
      self.deprecated = deprecated
      self.huggingFaceLink = huggingFaceLink
    }
  }

  public static let builtinSpecifications: [Specification] = [
    Specification(
      name: "Qwen 3.8 27B (2-bit S)", file: "qwen_3.8_27b_i2x.ckpt",
      version: .qwen_3_5_27b, huggingFaceLink: "Qwen/Qwen3.8-27B"),
    Specification(
      name: "Qwen 3.8 27B (4-bit S)", file: "qwen_3.8_27b_i4x.ckpt",
      version: .qwen_3_5_27b, huggingFaceLink: "Qwen/Qwen3.8-27B"),
    Specification(
      name: "Qwen 3.8 27B (8-bit S)", file: "qwen_3.8_27b_i8x.ckpt",
      version: .qwen_3_5_27b),
    Specification(
      name: "DeepSeek 4.1 Flash 0910 (2-bit S)", file: "deepseek_4.1_flash_0910_i2x.ckpt",
      engram: "deepseek_4.1_flash_0910_i2x.engram",
      version: .deepseek_4_1_flash, huggingFaceLink: "deepseek-ai/DeepSeek-V4.1-Flash"),
    Specification(
      name: "DeepSeek 4 Flash 0731 (2-bit S)", file: "deepseek_4_flash_0731_i2x.ckpt",
      version: .deepseek_4_flash, huggingFaceLink: "deepseek-ai/DeepSeek-V4-Flash-0731"),
    Specification(
      name: "Qwen 3.6 27B (4-bit S)", file: "qwen_3.6_27b_i4x.ckpt",
      version: .qwen_3_5_27b, deprecated: true),
    Specification(
      name: "Qwen 3.6 27B (8-bit S)", file: "qwen_3.6_27b_i8x.ckpt",
      version: .qwen_3_5_27b, deprecated: true),
    Specification(
      name: "Qwen 3.5 9B (5-bit S)", file: "qwen_3.5_9b_i5x.ckpt",
      version: .qwen_3_5_9b, deprecated: true),
    Specification(
      name: "Qwen 3.5 4B (8-bit S)", file: "qwen_3.5_4b_i8x.ckpt",
      version: .qwen_3_5_4b, deprecated: true),
  ]

  private static let fileSHA256: [String: String] = [
    "qwen_3.6_27b_i2x.ckpt": "898deb3fef5d77a34d609031899460aa79d4ba663c07f6760c5767385f6c91d3",
    "qwen_3.6_27b_i4x.ckpt": "20a28fb30af7d1d5228c314fa76080ec0dae7c9519fd69e5ad54c44462eb7dbc",
    "qwen_3.6_27b_i8x.ckpt": "1cd4b97a358a11dd795326e549e210d88f0aff6d91ebee8adc40eda6f36f7f4d",
    "qwen_3.5_9b_i5x.ckpt": "559d41f5e6721b4edb5a661c42b6f328a243876c2637cad36cf9937fb0d453c1",
    "qwen_3.5_4b_i8x.ckpt": "9e07b480a4e093d1d304ee59a625fc3169a53dc3a597d17f69a2dc167958ee9c",
    "qwen_3.8_27b_i4x.ckpt": "49f442f9b784e4866d4ed209feec2213762fd5729d1b9980b64b83dd8dd69a0e",
    "qwen_3.8_27b_i8x.ckpt": "f208994678a8244c2a6a5c1178f2ca5b31a8543200013938a1462ee79413866e",
    "deepseek_4_flash_0731_i2x.ckpt":
      "9eaa72400970ab6b699885335ec6847929f2310e011785111afb75fb86d169c3",
    "deepseek_4.1_flash_0910_i2x.ckpt":
      "9f4fd99071f6d84e73c19ebe728eb0dc9b6bc1d54a92bcd605e0b61f92e49bf9",
    "deepseek_4.1_flash_0910_i2x.engram":
      "46a64d7a5d7dfbce0261b378627169ac62a6fa591ce14554dcac055215048c71",
  ]

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in builtinSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static var overrideMapping: [String: Specification] = [:]

  public static var fallbackMapping: [String: Specification] = [:]

  public static func specificationForModel(_ name: String) -> Specification? {
    if let override = overrideMapping[name] {
      return override
    }
    return specificationMapping[name] ?? fallbackMapping[name]
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func isModelDownloaded(_ name: String, memorizedBy: Set<String>) -> Bool {
    return ModelZoo.isModelDownloaded(name, memorizedBy: memorizedBy)
  }

  public static func isModelDownloaded(
    _ specification: Specification, memorizedBy: Set<String> = []
  ) -> Bool {
    guard isModelDownloaded(specification.file, memorizedBy: memorizedBy) else { return false }
    if let engram = specification.engram {
      return isModelDownloaded(engram, memorizedBy: memorizedBy)
    }
    return true
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return name }
    return specification.name
  }

  public static func isModelDeprecated(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.deprecated ?? false
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in builtinSpecifications {
      guard specification.file != file, LLMZoo.isModelDownloaded(specification.file)
      else { continue }
      files.insert(specification.file)
      if let engram = specification.engram {
        files.insert(engram)
      }
    }
    return files
  }
}
