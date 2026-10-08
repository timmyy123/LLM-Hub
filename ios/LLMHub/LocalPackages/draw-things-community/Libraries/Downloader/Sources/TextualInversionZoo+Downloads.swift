import Foundation
import ModelZoo

extension Notification.Name {
  public static let didLoadCommunityEmbeddings = NSNotification.Name(
    "com.draw-things.community-embeddings-loaded")
}

extension TextualInversionZoo {
  public static func requestNetworkPayload() {
    DispatchQueue.global(qos: .userInitiated).async {
      internalRequestNetworkPayload()
    }
  }

  private static func internalRequestNetworkPayload() {
    let request = URLRequest(url: URL(string: "https://models.drawthings.ai/embeddings.json")!)
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
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("embeddings.json"))
      try? fileManager.moveItem(at: url, to: networkUrl.appendingPathComponent("embeddings.json"))
      let json = communityFromDisk()
      DispatchQueue.main.async {
        communityOnDisk = json
        refreshCommunity()
        NotificationCenter.default.post(Notification(name: .didLoadCommunityEmbeddings))
      }
    }
    task.resume()
    let sha256Request = URLRequest(
      url: URL(string: "https://models.drawthings.ai/embeddings_sha256.json")!)
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
      try? fileManager.removeItem(at: networkUrl.appendingPathComponent("embeddings_sha256.json"))
      try? fileManager.moveItem(
        at: url, to: networkUrl.appendingPathComponent("embeddings_sha256.json"))
      mergeFileSHA256()
    }
    sha256Task.resume()
  }

  private static func communityFromDisk() -> [Specification] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let modelsUrl = cacheUrl.appendingPathComponent("net").appendingPathComponent("embeddings.json")
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

  public static var community: [Specification] = {
    let availableModels = Set(
      availableSpecifications.compactMap {
        ($0.deprecated == true && isBuiltinTextualInversion($0.file)) || !isModelDownloaded($0.file)
          ? nil : $0.file
      })
    return communityOnDisk.filter { !availableModels.contains($0.file) }
  }()

  public static func refreshCommunity() {
    dispatchPrecondition(condition: .onQueue(.main))
    let availableModels = Set(
      availableSpecifications.compactMap {
        ($0.deprecated == true && isBuiltinTextualInversion($0.file)) || !isModelDownloaded($0.file)
          ? nil : $0.file
      })
    community = communityOnDisk.filter { !availableModels.contains($0.file) }
  }

  public static func mergeFileSHA256() {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
    let cacheUrl = urls.first!
    let modelsSHA256Url = cacheUrl.appendingPathComponent("net").appendingPathComponent(
      "embeddings_sha256.json")
    guard let jsonData = try? Data(contentsOf: modelsSHA256Url) else {
      return
    }
    let jsonDecoder = JSONDecoder()
    guard let sha256 = try? jsonDecoder.decode([String: String].self, from: jsonData) else {
      return
    }
    mergeFileSHA256(sha256)
  }
}
