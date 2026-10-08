import Foundation

public protocol DownloadZoo {
  static func isModelDownloaded(_ name: String, memorizedBy: Set<String>) -> Bool
  static func filePathForModelDownloaded(_ name: String) -> String
  static func fileSHA256ForModelDownloaded(_ name: String) -> String?
  static func availableFiles(excluding file: String?) -> Set<String>
}

extension DownloadZoo {
  public static func isModelDownloaded(_ name: String) -> Bool {
    return isModelDownloaded(name, memorizedBy: [])
  }
  public static func availableFiles(excluding of: DownloadZoo.Type = Self.self) -> Set<String> {
    var files = Set<String>()
    if of != ControlNetZoo.self {
      files.formUnion(ControlNetZoo.availableFiles(excluding: nil))
    }
    if of != EverythingZoo.self {
      files.formUnion(EverythingZoo.availableFiles(excluding: nil))
    }
    if of != LoRAZoo.self {
      files.formUnion(LoRAZoo.availableFiles(excluding: nil))
    }
    if of != LLMZoo.self {
      files.formUnion(LLMZoo.availableFiles(excluding: nil))
    }
    if of != ModelZoo.self {
      files.formUnion(ModelZoo.availableFiles(excluding: nil))
    }
    if of != TextGenerationZoo.self {
      files.formUnion(TextGenerationZoo.availableFiles(excluding: nil))
    }
    if of != TextualInversionZoo.self {
      files.formUnion(TextualInversionZoo.availableFiles(excluding: nil))
    }
    if of != UpscalerZoo.self {
      files.formUnion(UpscalerZoo.availableFiles(excluding: nil))
    }
    return files
  }
}
