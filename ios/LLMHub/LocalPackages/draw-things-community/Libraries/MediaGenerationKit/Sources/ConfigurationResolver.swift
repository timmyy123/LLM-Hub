import ConfigurationZoo
@preconcurrency import DataModels
@preconcurrency import Diffusion
import Foundation
import GRPCServer
import ModelZoo
import ScriptDataModels

internal enum ConfigurationZooSource {
  case builtin
  case bundled
  case remote
}

internal enum ConfigurationZooLoader {
  private static let remoteURL = URL(string: "https://models.drawthings.ai/configs.json")!
  private struct State {
    var bundledCache: [ConfigurationZoo.Specification]?
    var remoteCache: [ConfigurationZoo.Specification]?
    var remoteLoadTask: Task<[ConfigurationZoo.Specification], Never>?
  }

  private static var state = ProtectedValue(State())

  static func specificationsSync(from source: ConfigurationZooSource, offline: Bool)
    -> [ConfigurationZoo.Specification]
  {
    switch source {
    case .builtin:
      return ConfigurationZoo.community
    case .bundled:
      var bundledCache: [ConfigurationZoo.Specification]?
      state.modify { state in
        bundledCache = state.bundledCache
      }
      if let bundledCache { return bundledCache }
      let loaded = MediaGenerationResourceLoader.bundledData(resource: "configs").flatMap(parse) ?? []
      var resolved = loaded
      state.modify { state in
        if let cached = state.bundledCache {
          resolved = cached
        } else {
          state.bundledCache = loaded
        }
      }
      return resolved
    case .remote:
      guard !offline else { return [] }
      var remoteCache: [ConfigurationZoo.Specification]?
      state.modify { state in
        remoteCache = state.remoteCache
      }
      if let remoteCache { return remoteCache }
      let loaded = MediaGenerationResourceLoader.fetchRemoteData(url: remoteURL).flatMap(parse) ?? []
      var resolved = loaded
      state.modify { state in
        if let cached = state.remoteCache {
          resolved = cached
        } else {
          state.remoteCache = loaded
        }
      }
      return resolved
    }
  }

  static func specifications(from source: ConfigurationZooSource, offline: Bool) async
    -> [ConfigurationZoo.Specification]
  {
    switch source {
    case .builtin:
      return specificationsSync(from: .builtin, offline: offline)
    case .bundled:
      return specificationsSync(from: .bundled, offline: offline)
    case .remote:
      guard !offline else { return [] }
      let remoteState = cachedRemoteState()
      if let remoteCache = remoteState.cache {
        return remoteCache
      }
      if let remoteLoadTask = remoteState.task {
        return await remoteLoadTask.value
      }
      let loadTask = Task<[ConfigurationZoo.Specification], Never> {
        guard let data = await MediaGenerationResourceLoader.fetchRemoteData(url: remoteURL) else {
          return []
        }
        return parse(data)
      }
      updateRemoteState(cache: nil, task: loadTask)

      let loaded = await loadTask.value

      updateRemoteState(cache: loaded, task: nil)
      return loaded
    }
  }

  private static func cachedRemoteState() -> (
    cache: [ConfigurationZoo.Specification]?, task: Task<[ConfigurationZoo.Specification], Never>?
  ) {
    var cache: [ConfigurationZoo.Specification]?
    var task: Task<[ConfigurationZoo.Specification], Never>?
    state.modify { state in
      cache = state.remoteCache
      task = state.remoteLoadTask
    }
    return (cache, task)
  }

  private static func updateRemoteState(
    cache: [ConfigurationZoo.Specification]?,
    task: Task<[ConfigurationZoo.Specification], Never>?
  ) {
    state.modify { state in
      if let cache {
        state.remoteCache = cache
      }
      state.remoteLoadTask = task
    }
  }

  private static func parse(_ data: Data) -> [ConfigurationZoo.Specification] {
    guard let jsonSpecifications = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
    else {
      return []
    }
    return jsonSpecifications.compactMap { specification in
      guard let name = specification["name"] as? String,
        let configuration = specification["configuration"] as? [String: Any]
      else {
        return nil
      }
      return ConfigurationZoo.Specification(
        name: name,
        version: (specification["version"] as? String).flatMap { ModelVersion(rawValue: $0) },
        negative: specification["negative"] as? String,
        configuration: configuration
      )
    }
  }
}

internal enum ConfigurationResolver {
  static func configuration(
    for model: String,
    loras: Set<String>,
    offline: Bool
  ) -> ConfigurationZoo.Specification? {
    for source in [ConfigurationZooSource.builtin, .bundled, .remote] {
      let configurations = ConfigurationZooLoader.specificationsSync(from: source, offline: offline)
      guard !configurations.isEmpty else {
        continue
      }
      if let specification = matchingConfiguration(
        for: model,
        loras: loras,
        in: configurations
      ) {
        return specification
      }
    }
    return nil
  }

  static func configuration(
    for model: String,
    loras: Set<String>,
    offline: Bool
  ) async -> ConfigurationZoo.Specification? {
    for source in [ConfigurationZooSource.builtin, .bundled, .remote] {
      let configurations = await ConfigurationZooLoader.specifications(from: source, offline: offline)
      guard !configurations.isEmpty else {
        continue
      }
      if let specification = matchingConfiguration(
        for: model,
        loras: loras,
        in: configurations
      ) {
        return specification
      }
    }
    return nil
  }

  static func negativePrompt(
    for model: String,
    loras: Set<String>,
    offline: Bool
  ) -> String? {
    configuration(for: model, loras: loras, offline: offline)?.negative?
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }

  static func recommendedTemplate(for model: String, loras: Set<String>, offline: Bool)
    -> GenerationConfiguration
  {
    let defaultConfiguration = defaultConfiguration(for: model)
    guard let specification = configuration(for: model, loras: loras, offline: offline) else {
      return defaultConfiguration
    }
    guard
      let defaultData = try? JSONEncoder().encode(
        JSGenerationConfiguration(configuration: defaultConfiguration)
      ),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: defaultData) as? [String: Any]
    else {
      return defaultConfiguration
    }
    for (key, value) in specification.configuration {
      mergedDictionary[key] = value
    }
    mergedDictionary["model"] = model
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? JSONDecoder().decode(
        JSGenerationConfiguration.self,
        from: mergedData
      ).createGenerationConfiguration()
    else {
      return defaultConfiguration
    }
    return configuration
  }

  static func recommendedTemplate(for model: String, loras: Set<String>, offline: Bool) async
    -> GenerationConfiguration
  {
    let defaultConfiguration = defaultConfiguration(for: model)
    guard let specification = await configuration(for: model, loras: loras, offline: offline) else {
      return defaultConfiguration
    }
    guard
      let defaultData = try? JSONEncoder().encode(
        JSGenerationConfiguration(configuration: defaultConfiguration)
      ),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: defaultData) as? [String: Any]
    else {
      return defaultConfiguration
    }
    for (key, value) in specification.configuration {
      mergedDictionary[key] = value
    }
    mergedDictionary["model"] = model
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? JSONDecoder().decode(
        JSGenerationConfiguration.self,
        from: mergedData
      ).createGenerationConfiguration()
    else {
      return defaultConfiguration
    }
    return configuration
  }

  private static func defaultConfiguration(for model: String) -> GenerationConfiguration {
    let defaultScale = DeviceCapability.defaultScale(ModelZoo.defaultScaleForModel(model))
    var builder = GenerationConfigurationBuilder(from: GenerationConfiguration.default)
    builder.model = model
    builder.startWidth = defaultScale
    builder.startHeight = defaultScale
    return builder.build()
  }

  private static func modelPrefix(for model: String) -> String {
    let stem = (model as NSString).deletingPathExtension
    guard !stem.isEmpty else { return "" }
    var components = stem.components(separatedBy: "_")
    while let last = components.last, ["f16", "svd", "q5p", "q6p", "q8p", "i8x"].contains(last) {
      components.removeLast()
    }
    return components.joined(separator: "_")
  }

  private static func matchWithLoRAs(
    configurations: [ConfigurationZoo.Specification],
    _ loras: Set<String>,
    first: (ConfigurationZoo.Specification) -> Bool,
    second: ((ConfigurationZoo.Specification) -> Bool)? = nil
  ) -> ConfigurationZoo.Specification? {
    guard !loras.isEmpty else {
      guard let second else {
        return configurations.first(where: first)
      }
      return configurations.first(where: first) ?? configurations.first(where: second)
    }
    guard let second else {
      return configurations.first {
        let isFirst = first($0)
        guard isFirst else { return false }
        guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
        return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
      } ?? configurations.first(where: first)
    }
    return configurations.first {
      let isFirst = first($0)
      guard isFirst else { return false }
      guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
      return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
    } ?? configurations.first {
      let isSecond = second($0)
      guard isSecond else { return false }
      guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
      return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
    } ?? configurations.first(where: first) ?? configurations.first(where: second)
  }

  private static func matchingConfiguration(
    for model: String,
    loras: Set<String>,
    in configurations: [ConfigurationZoo.Specification]
  ) -> ConfigurationZoo.Specification? {
    let version = ModelZoo.versionForModel(model)
    let prefix = modelPrefix(for: model)

    var bestMatch = matchWithLoRAs(
      configurations: configurations,
      loras,
      first: { ($0.configuration["model"] as? String) == model },
      second: {
        guard let configModel = $0.configuration["model"] as? String, !prefix.isEmpty else {
          return false
        }
        return modelPrefix(for: configModel) == prefix
      }
    )
    if bestMatch == nil {
      bestMatch = matchWithLoRAs(configurations: configurations, loras) {
        guard let configModel = $0.configuration["model"] as? String, !prefix.isEmpty else {
          return false
        }
        let configPrefix = modelPrefix(for: configModel)
        return !configPrefix.isEmpty && prefix.hasPrefix("\(configPrefix)_")
      }
    }
    if bestMatch == nil {
      bestMatch = matchWithLoRAs(configurations: configurations, loras) {
        $0.version == version
      }
    }
    return bestMatch
  }
}
