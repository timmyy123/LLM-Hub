import Foundation
import ModelZoo

extension Notification.Name {
  public static let didLoadCommunityModels = NSNotification.Name(
    "com.draw-things.community-models-loaded")
  public static let didLoadFileSizeMetadata = NSNotification.Name(
    "com.draw-things.file-size-metadata-loaded")
}

extension ModelZoo {
  public struct FileSizeMetadata: Codable {
    public let file: String
    public let size: UInt
  }

  public static func requestNetworkPayload() {
    DispatchQueue.global(qos: .userInitiated).async {
      internalRequestNetworkPayload()
      internalUncuratedModelsRequestNetworkPayload()
      internalRequestFileSizeNetworkPayload()
      internalAPIsRequestNetworkPayload()
    }
  }

  private static func internalRequestFileSizeNetworkPayload() {
    let request = URLRequest(
      url: URL(string: "https://models.drawthings.ai/file_sizes_metadata.json")!)
    let task = URLSession.shared.downloadTask(with: request) { url, response, error in
      guard let url = url, let response = response as? HTTPURLResponse else {
        if let error = error {
          print("request error \(error)")
        }
        return
      }
      guard 200...299 ~= response.statusCode else { return }
      guard let jsonData = try? Data(contentsOf: url) else {
        return
      }
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      guard let _ = try? jsonDecoder.decode([FileSizeMetadata].self, from: jsonData) else {
        return
      }
      // If we can decode, then it is OK. Move this file to the cache directory.
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let cacheUrl = urls.first!
      let networkUrl = cacheUrl.appendingPathComponent("net")
      try? fileManager.createDirectory(at: networkUrl, withIntermediateDirectories: true)
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("file_size_metadata.json"))
      try? fileManager.moveItem(
        at: url, to: networkUrl.appendingPathComponent("file_size_metadata.json"))
      let json = fileSizeMetadataFromDisk()
      DispatchQueue.main.async {
        fileSizeMetadata = json
        NotificationCenter.default.post(Notification(name: .didLoadFileSizeMetadata))
      }
    }
    task.resume()
  }

  private static func internalRequestNetworkPayload() {
    let request = URLRequest(url: URL(string: "https://models.drawthings.ai/models.json")!)
    let task = URLSession.shared.downloadTask(with: request) { url, response, error in
      guard let url = url, let response = response as? HTTPURLResponse else {
        if let error = error {
          print("request error \(error)")
        }
        return
      }
      guard 200...299 ~= response.statusCode else { return }
      guard let jsonData = try? Data(contentsOf: url) else {
        return
      }
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      guard
        let _ = try? jsonDecoder.decode([FailableDecodable<Specification>].self, from: jsonData)
          .compactMap({ $0.value })
      else {
        return
      }
      // If we can decode, then it is OK. Move this file to the cache directory.
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let cacheUrl = urls.first!
      let networkUrl = cacheUrl.appendingPathComponent("net")
      try? fileManager.createDirectory(at: networkUrl, withIntermediateDirectories: true)
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("models.json"))
      try? fileManager.moveItem(at: url, to: networkUrl.appendingPathComponent("models.json"))
      let json = communityFromDisk()
      DispatchQueue.main.async {
        communityOnDisk = json
        refreshCommunity()
        NotificationCenter.default.post(Notification(name: .didLoadCommunityModels))
      }
    }

    task.resume()
    let sha256Request = URLRequest(
      url: URL(string: "https://models.drawthings.ai/models_sha256.json")!)
    let sha256Task = URLSession.shared.downloadTask(with: sha256Request) { url, response, error in
      guard let url = url, let response = response as? HTTPURLResponse else {
        if let error = error {
          print("request error \(error)")
        }
        return
      }
      guard 200...299 ~= response.statusCode else { return }
      guard let jsonData = try? Data(contentsOf: url) else {
        return
      }
      let jsonDecoder = JSONDecoder()
      guard let _ = try? jsonDecoder.decode([String: String].self, from: jsonData) else {
        return
      }
      // If we can decode, then it is OK. Move this file to the cache directory.
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let cacheUrl = urls.first!
      let networkUrl = cacheUrl.appendingPathComponent("net")
      try? fileManager.createDirectory(at: networkUrl, withIntermediateDirectories: true)
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("models_sha256.json"))
      try? fileManager.moveItem(
        at: url, to: networkUrl.appendingPathComponent("models_sha256.json"))
      mergeCommunityAndUncuratedFileSHA256()
    }
    sha256Task.resume()
  }

  private static func internalUncuratedModelsRequestNetworkPayload() {
    let request = URLRequest(
      url: URL(string: "https://models.drawthings.ai/uncurated_models.json")!)
    let task = URLSession.shared.downloadTask(with: request) { url, response, error in
      guard let url = url, let response = response as? HTTPURLResponse else {
        if let error = error {
          print("request error \(error)")
        }
        return
      }
      guard 200...299 ~= response.statusCode else { return }
      guard let jsonData = try? Data(contentsOf: url) else {
        return
      }
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      guard
        let _ = try? jsonDecoder.decode([FailableDecodable<Specification>].self, from: jsonData)
          .compactMap({ $0.value })
      else {
        return
      }
      // If we can decode, then it is OK. Move this file to the cache directory.
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let cacheUrl = urls.first!
      let networkUrl = cacheUrl.appendingPathComponent("net")
      try? fileManager.createDirectory(at: networkUrl, withIntermediateDirectories: true)
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("uncurated_models.json"))
      try? fileManager.moveItem(
        at: url, to: networkUrl.appendingPathComponent("uncurated_models.json"))
      let json = uncuratedCommunityFromDisk()
      DispatchQueue.main.async {
        uncuratedCommunityOnDisk = json
        refreshCommunity()
        NotificationCenter.default.post(Notification(name: .didLoadCommunityModels))
      }
    }
    task.resume()
    let sha256Request = URLRequest(
      url: URL(string: "https://models.drawthings.ai/uncurated_models_sha256.json")!)
    let sha256Task = URLSession.shared.downloadTask(with: sha256Request) { url, response, error in
      guard let url = url, let response = response as? HTTPURLResponse else {
        if let error = error {
          print("request error \(error)")
        }
        return
      }
      guard 200...299 ~= response.statusCode else { return }
      guard let jsonData = try? Data(contentsOf: url) else {
        return
      }
      let jsonDecoder = JSONDecoder()
      guard let _ = try? jsonDecoder.decode([String: String].self, from: jsonData) else {
        return
      }
      // If we can decode, then it is OK. Move this file to the cache directory.
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let cacheUrl = urls.first!
      let networkUrl = cacheUrl.appendingPathComponent("net")
      try? fileManager.createDirectory(at: networkUrl, withIntermediateDirectories: true)
      try? fileManager.removeItem(
        at: networkUrl.appendingPathComponent("uncurated_models_sha256.json"))
      try? fileManager.moveItem(
        at: url, to: networkUrl.appendingPathComponent("uncurated_models_sha256.json"))
      mergeCommunityAndUncuratedFileSHA256()
    }
    sha256Task.resume()
  }

  private static func internalAPIsRequestNetworkPayload() {
    let apisRequest = URLRequest(url: URL(string: "https://models.drawthings.ai/apis.json")!)
    let apisTask = URLSession.shared.downloadTask(with: apisRequest) { url, response, error in
      guard let url = url, let response = response as? HTTPURLResponse else {
        if let error = error {
          print("request error \(error)")
        }
        return
      }
      guard 200...299 ~= response.statusCode else { return }
      guard let jsonData = try? Data(contentsOf: url) else {
        return
      }
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      guard
        let _ = try? jsonDecoder.decode([FailableDecodable<Specification>].self, from: jsonData)
          .compactMap({ $0.value })
      else {
        return
      }
      // If we can decode, then it is OK. Move this file to the cache directory.
      let fileManager = FileManager.default
      let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
      let cacheUrl = urls.first!
      let networkUrl = cacheUrl.appendingPathComponent("net")
      try? fileManager.createDirectory(at: networkUrl, withIntermediateDirectories: true)
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("apis.json"))
      try? fileManager.moveItem(at: url, to: networkUrl.appendingPathComponent("apis.json"))
      let json = apiModelsFromDisk()
      DispatchQueue.main.async {
        APIs = json
      }
    }
    apisTask.resume()
  }

  private static func fileSizeMetadataFromDisk() -> [String: FileSizeMetadata] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let modelsUrl = cacheUrl.appendingPathComponent("net").appendingPathComponent(
      "file_size_metadata.json")
    guard let jsonData = try? Data(contentsOf: modelsUrl) else {
      return [:]
    }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard let jsonSpecifications = try? jsonDecoder.decode([FileSizeMetadata].self, from: jsonData)
    else {
      return [:]
    }
    return Dictionary(
      jsonSpecifications.map { ($0.file, $0) }, uniquingKeysWith: { (_, last) in last })
  }

  private static func communityFromDisk() -> [Specification] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let modelsUrl = cacheUrl.appendingPathComponent("net").appendingPathComponent("models.json")
    guard let jsonData = try? Data(contentsOf: modelsUrl) else {
      return []
    }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let jsonSpecifications = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value })
    else {
      return []
    }
    return jsonSpecifications
  }

  private static func uncuratedCommunityFromDisk() -> [Specification] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let modelsUrl = cacheUrl.appendingPathComponent("net").appendingPathComponent(
      "uncurated_models.json")
    guard let jsonData = try? Data(contentsOf: modelsUrl) else {
      return []
    }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let jsonSpecifications = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value })
    else {
      return []
    }
    return jsonSpecifications
  }

  private static var communityOnDisk: [Specification] = communityFromDisk()

  public static var uncuratedCommunityOnDisk: [Specification] = uncuratedCommunityFromDisk()

  private static func apiModelsFromDisk() -> [Specification] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let apisUrl = cacheUrl.appendingPathComponent("net").appendingPathComponent("apis.json")
    guard let jsonData = try? Data(contentsOf: apisUrl) else {
      return []
    }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let jsonSpecifications = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value })
    else {
      return []
    }

    return jsonSpecifications
  }

  public static var APIs: [Specification] = apiModelsFromDisk()

  public static var fileSizeMetadata: [String: FileSizeMetadata] = fileSizeMetadataFromDisk()

  public static var community: [Specification] = {
    let availableModels = Set(
      availableSpecifications.compactMap {
        return ($0.deprecated == true && isBuiltinModel($0.file)) || !isModelDownloaded($0)
          ? nil : $0.file
      })
    return communityOnDisk.filter { !availableModels.contains($0.file) }
  }()

  public static var uncuratedCommunity: [Specification] = {
    let availableModels = Set(
      availableSpecifications.compactMap {
        return ($0.deprecated == true && isBuiltinModel($0.file)) || !isModelDownloaded($0)
          ? nil : $0.file
      })
    return uncuratedCommunityOnDisk.filter { !availableModels.contains($0.file) }
  }()

  public static var communityAndUncurated: [Specification] {
    return community + uncuratedCommunity
  }

  public static func refreshCommunity() {
    dispatchPrecondition(condition: .onQueue(.main))
    let availableModels = Set(
      availableSpecifications.compactMap {
        ($0.deprecated == true && isBuiltinModel($0.file)) || !isModelDownloaded($0) ? nil : $0.file
      })
    community = communityOnDisk.filter { !availableModels.contains($0.file) }
  }

  public static func mergeCommunityAndUncuratedFileSHA256() {
    mergeCommunityFileSHA256()
    mergeCommunityUncuratedFileSHA256()
  }

  public static func mergeCommunityFileSHA256() {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let modelsSHA256Url = cacheUrl.appendingPathComponent("net").appendingPathComponent(
      "models_sha256.json")
    guard let jsonData = try? Data(contentsOf: modelsSHA256Url) else {
      return
    }
    let jsonDecoder = JSONDecoder()
    guard let sha256 = try? jsonDecoder.decode([String: String].self, from: jsonData) else {
      return
    }
    mergeFileSHA256(sha256)
  }

  public static func mergeCommunityUncuratedFileSHA256() {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let uncuratedModelsSHA256Url = cacheUrl.appendingPathComponent("net").appendingPathComponent(
      "uncurated_models_sha256.json")
    guard let jsonData = try? Data(contentsOf: uncuratedModelsSHA256Url) else {
      return
    }
    let jsonDecoder = JSONDecoder()
    guard let sha256 = try? jsonDecoder.decode([String: String].self, from: jsonData) else {
      return
    }
    mergeFileSHA256(sha256)
  }
}
