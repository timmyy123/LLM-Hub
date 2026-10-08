import CoreML

public final class ManagedMLModel {
  let modelUrl: URL
  let configuration: MLModelConfiguration
  private var loadedModel: MLModel? = nil
  public init(contentsOf modelUrl: URL, configuration: MLModelConfiguration) {
    self.modelUrl = modelUrl
    self.configuration = configuration
  }
  public func loadResources() throws {
    guard loadedModel == nil else { return }
    loadedModel = try MLModel(contentsOf: modelUrl, configuration: configuration)
  }
  public func unloadResources() {
    loadedModel = nil
  }
  public func perform<R>(_ body: (MLModel) throws -> R) throws -> R {
    return try autoreleasepool {
      try loadResources()
      return try body(loadedModel!)
    }
  }
}
