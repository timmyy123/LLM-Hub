import DataModels
import Foundation
import Logging
import ModelZoo
import ServerConfigurationRewriter

public struct ServerLoRALoader: ServerConfigurationRewriter {

  private let localLoRAManager: LocalLoRAManager
  private let logger = Logger(label: "com.draw-things.image-generation-service")

  public init(localLoRAManager: LocalLoRAManager) {
    self.localLoRAManager = localLoRAManager
  }

  public func rewrite(
    configuration: GenerationConfiguration,
    progress: @escaping (_ bytesReceived: Int64, _ bytesExpected: Int64, _ index: Int, _ total: Int)
      -> Void,
    cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping (Result<ServerConfigurationRewriteResult, Error>) -> Void
  ) {

    let configLoras: [String] = configuration.loras.compactMap { $0.file }
    self.logger.info("Received LoRAs: \(configLoras)")
    let loRAsNeedToLoad = configLoras.filter { file in
      // Won't trigger download for non-SHA256 LoRA.
      guard isSHA256LoRA(loraName: file) else { return false }
      let sha256 = sha256HashName(loraName: file)
      if ModelZoo.isModelDownloaded(sha256) {
        return false
      }
      return true
    }

    var configurationBuilder = GenerationConfigurationBuilder(from: configuration)
    // Remove the suffix no matter if it needs download or not.
    configurationBuilder.loras = configuration.loras.compactMap { lora in
      guard let file = lora.file else {
        return nil
      }
      let sha256 = sha256HashName(loraName: file)
      return DataModels.LoRA(file: sha256, weight: lora.weight, mode: lora.mode)
    }
    let configuration = configurationBuilder.build()

    guard loRAsNeedToLoad.count > 0 else {
      self.logger.info("No loRAs need to load.")
      completion(.success(ServerConfigurationRewriteResult(configuration: configuration)))
      return
    }

    self.logger.info("LoRAs need to load: \(loRAsNeedToLoad)")
    localLoRAManager.downloadRemoteLoRAs(
      loRAsNeedToLoad, progress: progress, cancellation: cancellation
    ) { results in
      // Check if any model failed to download
      if let failedModel = results.first(where: { !$0.value })?.key {
        self.logger.info("Fail to download LoRA: \(failedModel)")
        completion(.failure(ServerConfigurationRewriteError.canNotLoadModel(failedModel)))
        return
      }

      self.logger.info("Downloaded LoRAs: \(results)")
      completion(.success(ServerConfigurationRewriteResult(configuration: configuration)))
    }
  }

  func sha256HashName(loraName: String) -> String {
    guard isSHA256LoRA(loraName: loraName) else {
      return loraName
    }
    if let sha256LoRA = loraName.split(separator: "_", maxSplits: 1).first {
      return String(sha256LoRA)
    }
    return loraName
  }

  func isSHA256LoRA(loraName: String) -> Bool {
    // Check if the string is empty
    guard !loraName.isEmpty else {
      return false
    }

    guard !loraName.hasSuffix(".ckpt") else {
      return false
    }

    // Split the name by the first dash or underscore
    let components: [String]
    if loraName.contains("_") {
      components = loraName.split(separator: "_", maxSplits: 1).map { String($0) }
    } else {
      // If there's no separator, check if the whole string is a valid hash
      components = [loraName]
    }

    // Extract the first component (potential hash)
    guard let potentialHash = components.first else {
      return false
    }

    // Check if the potential hash is 64 characters long (SHA256 length)
    guard potentialHash.count == 64, potentialHash.allSatisfy({ $0.isHexDigit }) else {
      return false
    }

    return true
  }

}
