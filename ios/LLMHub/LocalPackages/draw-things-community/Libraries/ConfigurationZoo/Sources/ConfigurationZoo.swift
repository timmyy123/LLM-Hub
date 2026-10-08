import DataModels
import Dflat
import Diffusion
import Foundation
import ModelZoo
import ScriptDataModels

public struct ConfigurationZoo {
  public struct Specification {
    public var name: String
    public var version: ModelVersion?
    public var negative: String?
    public var configuration: [String: Any]
    public init(
      name: String, version: ModelVersion?, negative: String?, configuration: [String: Any]
    ) {
      self.name = name
      self.version = version
      self.negative = negative
      self.configuration = configuration
    }
  }

  // MARK: - Community Configurations (Network)

  /// Community configurations fetched from network and cached to disk
  public static var community: [Specification] = communityFromDisk()

  /// Request community configurations from network
  public static func requestNetworkPayload(completion: (([Specification]) -> Void)? = nil) {
    DispatchQueue.global(qos: .userInitiated).async {
      internalRequestNetworkPayload(completion: completion)
    }
  }

  private static func internalRequestNetworkPayload(completion: (([Specification]) -> Void)?) {
    let request = URLRequest(url: URL(string: "https://models.drawthings.ai/configs.json")!)
    let task = URLSession.shared.downloadTask(with: request) { url, response, error in
      guard let url = url, let response = response as? HTTPURLResponse else {
        if let error = error {
          print("request error \(error)")
        }
        completion?(communityFromDisk())
        return
      }
      guard 200...299 ~= response.statusCode else {
        completion?(communityFromDisk())
        return
      }
      guard let _ = try? Data(contentsOf: url) else {
        completion?(communityFromDisk())
        return
      }
      // If we can decode, then it is OK. Move this file to the cache directory.
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let cacheUrl = urls.first!
      let networkUrl = cacheUrl.appendingPathComponent("net")
      try? fileManager.createDirectory(at: networkUrl, withIntermediateDirectories: true)
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("configs.json"))
      try? fileManager.moveItem(at: url, to: networkUrl.appendingPathComponent("configs.json"))
      let json = communityFromDisk()
      DispatchQueue.main.async {
        community = json
      }
      completion?(json)
    }
    task.resume()
  }

  private static func communityFromDisk() -> [Specification] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let modelsUrl = cacheUrl.appendingPathComponent("net").appendingPathComponent(
      "configs.json")
    guard let jsonData = try? Data(contentsOf: modelsUrl) else {
      return []
    }

    guard
      let jsonSpecifications = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]]
    else {
      return []
    }
    var specifications = [Specification]()
    for specification in jsonSpecifications {
      guard let name = specification["name"] as? String,
        let configuration = specification["configuration"] as? [String: Any]
      else { continue }
      specifications.append(
        Specification(
          name: name,
          version: (specification["version"] as? String).flatMap { ModelVersion(rawValue: $0) },
          negative: specification["negative"] as? String, configuration: configuration))
    }
    return specifications
  }

  // MARK: - Custom Configurations (Local)

  public static var availableSpecifications: [Specification] = {
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) else {
      return []
    }

    guard
      let jsonSpecifications = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]]
    else {
      return []
    }
    var availableSpecifications = [Specification]()
    for specification in jsonSpecifications {
      guard let name = specification["name"] as? String,
        let configuration = specification["configuration"] as? [String: Any]
      else { continue }
      availableSpecifications.append(
        Specification(
          name: name,
          version: (specification["version"] as? String).flatMap { ModelVersion(rawValue: $0) },
          negative: specification["negative"] as? String, configuration: configuration))
    }
    return availableSpecifications
  }()

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.name] = specification
    }
    return mapping
  }()

  public static func read(from workspace: Workspace) {
    let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
      GenerationConfiguration.id != 0,
      orderBy: [
        GenerationConfiguration.name.descending
      ])
    guard !configurations.isEmpty else { return }
    var specifications = [Specification]()
    for configuration in configurations {
      guard let name = configuration.name else { continue }
      guard
        let jsonData = try? JSONEncoder().encode(
          JSGenerationConfiguration(configuration: configuration)),
        var configurationDictionary = try? JSONSerialization.jsonObject(
          with: jsonData) as? [String: Any]
      else { continue }
      configurationDictionary["id"] = nil
      specifications.append(
        Specification(
          name: name, version: nil, negative: nil, configuration: configurationDictionary))
    }
    var availableSpecifications = availableSpecifications
    for specification in specifications {
      guard !availableSpecifications.contains(where: { $0.name == specification.name }) else {
        continue
      }
      availableSpecifications.append(specification)
    }
    var specificationMapping = specificationMapping
    var customSpecifications = [[String: Any]]()
    for specification in availableSpecifications {
      specificationMapping[specification.name] = specification
      customSpecifications.append([
        "name": specification.name, "configuration": specification.configuration,
      ])
    }
    guard
      let jsonData = try? JSONSerialization.data(
        withJSONObject: customSpecifications, options: [.prettyPrinted, .sortedKeys])
    else { return }
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    do {
      try jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
      // If succeed. We can remove all configurations from the workspace.
      workspace.performChanges([GenerationConfiguration.self]) { transactionContext in
        let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
          GenerationConfiguration.id != 0)
        for configuration in configurations {
          guard
            let deletionRequest = GenerationConfigurationChangeRequest.deletionRequest(
              configuration)
          else { continue }
          transactionContext.try(submit: deletionRequest)
        }
      }
    } catch {
      // Do nothing.
    }
    self.availableSpecifications = availableSpecifications
    self.specificationMapping = specificationMapping
  }

  public static func specification(_ name: String) -> Specification? {
    return specificationMapping[name]
  }

  private static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func appendCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [[String: Any]]()
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      if let jsonSpecifications = try? JSONSerialization.jsonObject(with: jsonData)
        as? [[String: Any]]
      {
        customSpecifications.append(contentsOf: jsonSpecifications)
      }
    }
    if let firstIndex = customSpecifications.firstIndex(where: {
      $0["name"] as? String == specification.name
    }) {
      var dictionary = customSpecifications[firstIndex]
      dictionary["configuration"] = specification.configuration
      if let negative = specification.negative {
        dictionary["negative"] = negative
      }
      if let version = specification.version {
        dictionary["version"] = version.rawValue
      }
      customSpecifications[firstIndex] = dictionary
    } else {
      customSpecifications = customSpecifications.filter {
        guard let name = $0["name"] as? String else { return false }
        return name != specification.name
      }
      var dictionary: [String: Any] = [
        "name": specification.name, "configuration": specification.configuration,
      ]
      dictionary["negative"] = specification.negative
      dictionary["version"] = specification.version?.rawValue
      customSpecifications.append(dictionary)
    }
    guard
      let jsonData = try? JSONSerialization.data(
        withJSONObject: customSpecifications, options: [.prettyPrinted, .sortedKeys])
    else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    var availableSpecifications = availableSpecifications
    if let firstIndex = availableSpecifications.firstIndex(where: { $0.name == specification.name })
    {
      availableSpecifications[firstIndex] = specification
    } else {
      availableSpecifications = availableSpecifications.filter { $0.name != specification.name }
      availableSpecifications.append(specification)
    }
    self.availableSpecifications = availableSpecifications
    specificationMapping[specification.name] = specification
  }

  public static func remove(by nameToRemove: String) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [[String: Any]]()
    let jsonFile = filePathForModelDownloaded("custom_configs.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      if let jsonSpecifications = try? JSONSerialization.jsonObject(with: jsonData)
        as? [[String: Any]]
      {
        customSpecifications.append(contentsOf: jsonSpecifications)
      }
    }
    customSpecifications = customSpecifications.filter {
      guard let name = $0["name"] as? String else { return false }
      return name != nameToRemove
    }
    guard
      let jsonData = try? JSONSerialization.data(
        withJSONObject: customSpecifications, options: [.prettyPrinted, .sortedKeys])
    else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    var availableSpecifications = availableSpecifications
    availableSpecifications = availableSpecifications.filter { $0.name != nameToRemove }
    self.availableSpecifications = availableSpecifications
    specificationMapping[nameToRemove] = nil
  }
}
