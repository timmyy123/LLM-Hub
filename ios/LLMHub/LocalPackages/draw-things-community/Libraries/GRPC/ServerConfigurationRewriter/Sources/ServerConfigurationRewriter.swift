import DataModels
import Foundation

public enum ServerConfigurationRewriteError: Error {
  case canNotLoadModel(String)
}

public struct ServerConfigurationRewriteResult {
  public var configuration: GenerationConfiguration
  public var fileMapping: [String: String]

  public init(configuration: GenerationConfiguration, fileMapping: [String: String] = [:]) {
    self.configuration = configuration
    self.fileMapping = fileMapping
  }
}

public protocol ServerConfigurationRewriter {
  func rewrite(
    configuration: GenerationConfiguration,
    progress: @escaping (_ bytesReceived: Int64, _ bytesExpected: Int64, _ index: Int, _ total: Int)
      -> Void,
    cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping (Result<ServerConfigurationRewriteResult, Error>) -> Void)
}
