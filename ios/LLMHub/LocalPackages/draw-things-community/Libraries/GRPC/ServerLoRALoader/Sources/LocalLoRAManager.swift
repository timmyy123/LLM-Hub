import Crypto
import Foundation
import Logging

#if os(Linux)
  import FoundationNetworking
#endif

/// Utility functions for managing custom model directories with size limits
public final class LocalLoRAManager {

  public enum Error: Swift.Error {
    case noAttributes
    case contentHashMismatch
  }

  private let logger = Logger(label: "com.draw-things.local-lora-manager")
  private let r2Client: R2Client
  private let localDirectory: String
  public init(r2Client: R2Client, localDirectory: String) {
    self.r2Client = r2Client
    self.localDirectory = localDirectory
  }

  /// Download a model without performing cleanup
  /// - Parameters:
  ///   - modelName: Name of the model to download
  /// - Returns: A tuple with success status and the downloaded file size
  private func downloadRemoteLoRA(
    _ modelNames: [String], index: Int, results: [String: Bool], session: URLSession,
    objCResponder: R2Client.ObjCResponder,
    cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping ([String: Bool]) -> Void
  ) {
    guard index < modelNames.count else {
      completion(results)
      return
    }
    var results = results
    let modelName = modelNames[index]
    let dirURL = URL(fileURLWithPath: localDirectory)
    let logger = logger
    logger.info("Downloading LoRA \(modelName)")
    objCResponder.index = index
    let task = r2Client.downloadObject(
      key: modelName, session: session, objCResponder: objCResponder
    ) { result in
      switch result {
      case .success(let data):
        logger.info("Downloaded LoRA \(modelName)")
        do {
          // Get the file size from the downloaded temp file
          // only using the prefix hash as the file name, for example
          // "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d_d71b5bbc-0a6b-4b50-8c6f-3691b80bc2ee" --> "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d"
          let modelName = modelName.components(separatedBy: "_").first ?? modelName
          // Create destination URL
          let destinationUrl = dirURL.appendingPathComponent(modelName)
          // Create parent directories if needed
          try FileManager.default.createDirectory(
            at: destinationUrl.deletingLastPathComponent(),
            withIntermediateDirectories: true
          )

          let hash = SHA256.hash(data: data)
          let sha256 = hash.compactMap { String(format: "%02x", $0) }.joined()
          guard sha256 == modelName else {
            logger.info("Mismatch content hash \(modelName) \(sha256)")
            throw Error.contentHashMismatch
          }
          // Move downloaded file to destination
          try data.write(to: destinationUrl, options: .atomic)
          logger.info(
            "Successfully write model \(modelName) to custom directory \(self.localDirectory)")
          results[modelName] = true
        } catch {
          logger.info("Failed to save model \(modelName): \(error.localizedDescription)")
          results[modelName] = false
        }
        self.downloadRemoteLoRA(
          modelNames, index: index + 1, results: results, session: session,
          objCResponder: objCResponder, cancellation: cancellation, completion: completion)
      case .failure(let error):
        logger.info("Error downloading model \(modelName): \(error.localizedDescription)")
        // Try fallback: extract prefix and list objects with that prefix
        let prefix = modelName.components(separatedBy: "_").first ?? modelName
        logger.info("Attempting fallback for prefix: \(prefix)")

        self.r2Client.listObjects(prefix: prefix) { listResult in
          switch listResult {
          case .success(let keys):
            logger.info("Found \(keys.count) objects with prefix \(prefix): \(keys)")
            // Filter out the original failed key and try alternatives
            let alternatives = keys.filter { $0 != modelName && $0.hasPrefix(prefix) }
            if let alternativeKey = alternatives.first {
              logger.info("Trying alternative key: \(alternativeKey)")
              // Retry download with alternative key
              _ = self.r2Client.downloadObject(
                key: alternativeKey, session: session, objCResponder: objCResponder
              ) { alternativeResult in
                switch alternativeResult {
                case .success(let data):
                  logger.info("Successfully downloaded alternative LoRA \(alternativeKey)")
                  do {
                    // Use the prefix hash as the file name (same as original logic)
                    let dirURL = URL(fileURLWithPath: self.localDirectory)
                    let destinationUrl = dirURL.appendingPathComponent(prefix)
                    try FileManager.default.createDirectory(
                      at: destinationUrl.deletingLastPathComponent(),
                      withIntermediateDirectories: true
                    )

                    let hash = SHA256.hash(data: data)
                    let sha256 = hash.compactMap { String(format: "%02x", $0) }.joined()
                    guard sha256 == prefix else {
                      logger.info("Mismatch content hash \(prefix) \(sha256)")
                      throw Error.contentHashMismatch
                    }

                    try data.write(to: destinationUrl, options: .atomic)
                    logger.info(
                      "Successfully wrote alternative model \(prefix) to \(self.localDirectory)"
                    )
                    results[modelName] = true
                  } catch {
                    logger.info(
                      "Failed to save alternative model \(prefix): \(error.localizedDescription)"
                    )
                    results[modelName] = false
                  }
                  self.downloadRemoteLoRA(
                    modelNames, index: index + 1, results: results, session: session,
                    objCResponder: objCResponder, cancellation: cancellation,
                    completion: completion)
                case .failure(let alternativeError):
                  logger.info(
                    "Alternative download also failed for \(alternativeKey): \(alternativeError.localizedDescription)"
                  )
                  results[modelName] = false
                  self.downloadRemoteLoRA(
                    modelNames, index: index + 1, results: results, session: session,
                    objCResponder: objCResponder, cancellation: cancellation,
                    completion: completion)
                }
              }
            } else {
              logger.info("No alternative keys found for prefix \(prefix)")
              results[modelName] = false
              self.downloadRemoteLoRA(
                modelNames, index: index + 1, results: results, session: session,
                objCResponder: objCResponder, cancellation: cancellation, completion: completion
              )
            }
          case .failure(let listError):
            logger.info(
              "Failed to list objects with prefix \(prefix): \(listError.localizedDescription)")
            results[modelName] = false
            self.downloadRemoteLoRA(
              modelNames, index: index + 1, results: results, session: session,
              objCResponder: objCResponder, cancellation: cancellation, completion: completion)
          }
        }
        return  // Exit early to avoid calling downloadRemoteLoRA twice
      }
    }
    if let task = task {
      cancellation({
        task.cancel()
      })
    }
  }

  /// Download multiple models and perform cleanup only once after all downloads
  /// - Parameters:
  ///   - modelNames: Array of model names to download
  /// - Returns: Dictionary mapping model names to download success status
  public func downloadRemoteLoRAs(
    _ modelNames: [String],
    progress: @escaping (_ bytesReceived: Int64, _ bytesExpected: Int64, _ index: Int, _ total: Int)
      -> Void, cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping ([String: Bool]) -> Void
  ) {
    let total = modelNames.count
    let objCResponder = R2Client.ObjCResponder { bytesReceived, bytesExpected, index in
      progress(bytesReceived, bytesExpected, index, total)
    }
    downloadRemoteLoRA(
      modelNames, index: 0, results: [:],
      session: URLSession(configuration: .default, delegate: objCResponder, delegateQueue: nil),
      objCResponder: objCResponder, cancellation: cancellation, completion: completion)
  }
}
