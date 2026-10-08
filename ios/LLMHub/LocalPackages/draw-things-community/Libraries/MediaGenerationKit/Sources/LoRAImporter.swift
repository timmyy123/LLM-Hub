@preconcurrency import Diffusion
import Foundation
import ModelOp
import ModelZoo

/// Public wrapper over the existing internal LoRA import implementation.
///
/// This stays separate from `MediaGenerationPipeline` because importing LoRA
/// weights is a file conversion workflow, not a generation workflow.
public enum LoRAConvertError: Error, LocalizedError {
  case fileNotFound(String)
  case versionDetectionFailed
  case conversionFailed(Error)

  public var errorDescription: String? {
    switch self {
    case .fileNotFound(let path):
      return "LoRA source file not found: \(path)"
    case .versionDetectionFailed:
      return "Could not detect LoRA model version — pass version to override"
    case .conversionFailed(let error):
      return "LoRA conversion failed: \(error.localizedDescription)"
    }
  }
}

/// Converts local LoRA files into Draw Things format.
public struct LoRAImporter: Sendable {
  /// Source LoRA file to inspect or convert.
  public let file: URL

  /// `nil` means auto-detect. Calling `inspect()` resolves and stores the
  /// detected version for later `import(to:)` calls.
  public var version: ModelVersion?

  private static let importLock = NSLock()

  /// Creates an importer for a LoRA source file.
  public init(file: URL, version: ModelVersion? = nil) {
    self.file = file
    self.version = version
  }

  /// Optional preflight inspection. Call this only when you want to read
  /// importer metadata before conversion. `inspect()` is not required before
  /// `import(to:)`.
  ///
  /// The current wrapper reuses the full importer path because the underlying
  /// ModelOp implementation does not expose standalone metadata inspection yet.
  public mutating func inspect() throws {
    let inspectedVersion = try Self.performImport(
      sourceFile: file,
      destinationFile: nil,
      scaleFactor: 1.0,
      forceVersion: version,
      progressHandler: nil
    )
    version = inspectedVersion
  }

  /// Converts the source LoRA file into Draw Things format.
  ///
  /// On success, ``version`` is updated to the detected or forced model
  /// version used during conversion.
  public mutating func `import`(
    to file: URL,
    scaleFactor: Double = 1.0,
    progressHandler: ((Double) -> Void)? = nil
  ) throws {
    let importedVersion = try Self.performImport(
      sourceFile: self.file,
      destinationFile: file,
      scaleFactor: scaleFactor,
      forceVersion: version,
      progressHandler: progressHandler
    )
    version = importedVersion
  }

  private static func performImport(
    sourceFile: URL,
    destinationFile: URL?,
    scaleFactor: Double,
    forceVersion: ModelVersion?,
    progressHandler: ((Double) -> Void)?
  ) throws -> ModelVersion {
    guard FileManager.default.fileExists(atPath: sourceFile.path) else {
      throw LoRAConvertError.fileNotFound(sourceFile.path)
    }

    let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mediagenerationkit-lora-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)

    let outputFilename = destinationFile?.lastPathComponent
      ?? "__inspect_\(UUID().uuidString)_lora_f16.ckpt"
    let outputName =
      destinationFile?.deletingPathExtension().lastPathComponent
      ?? sourceFile.deletingPathExtension().lastPathComponent

    Self.importLock.lock()
    let originalExternalURLs = ModelZoo.externalUrls
    defer {
      ModelZoo.externalUrls = originalExternalURLs
      Self.importLock.unlock()
      try? FileManager.default.removeItem(at: tempDirectory)
    }

    ModelZoo.externalUrls = [tempDirectory]

    let modelVersion: ModelVersion
    do {
      (modelVersion, _, _, _) = try ModelOp.LoRAImporter.import(
        downloadedFile: sourceFile.path,
        name: outputName,
        filename: outputFilename,
        scaleFactor: scaleFactor,
        forceVersion: forceVersion,
        progress: { progress in
          progressHandler?(Double(progress))
        }
      )
    } catch ModelOp.LoRAImporter.Error.modelVersionFailed {
      throw LoRAConvertError.versionDetectionFailed
    } catch {
      throw LoRAConvertError.conversionFailed(error)
    }

    guard let destinationFile else {
      return modelVersion
    }

    let importedFile = tempDirectory.appendingPathComponent(outputFilename)
    try FileManager.default.createDirectory(
      at: destinationFile.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    if FileManager.default.fileExists(atPath: destinationFile.path) {
      try FileManager.default.removeItem(at: destinationFile)
    }
    try FileManager.default.moveItem(at: importedFile, to: destinationFile)
    return modelVersion
  }
}
