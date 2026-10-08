import Foundation

internal enum MediaGenerationResourceLoader {
  static func bundledData(resource name: String, extension fileExtension: String = "json") -> Data? {
    if let resourceURL = Bundle.main.url(forResource: name, withExtension: fileExtension),
      let data = try? Data(contentsOf: resourceURL)
    {
      return data
    }

    let sourceURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("Resources", isDirectory: true)
      .appendingPathComponent("\(name).\(fileExtension)")
    return try? Data(contentsOf: sourceURL)
  }

  static func fetchRemoteData(url: URL, timeout: TimeInterval = 10) -> Data? {
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout

    let semaphore = DispatchSemaphore(value: 0)
    var responseData: Data?

    URLSession.shared.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      guard error == nil,
        let response = response as? HTTPURLResponse,
        200...299 ~= response.statusCode,
        let data
      else {
        return
      }
      responseData = data
    }.resume()

    guard semaphore.wait(timeout: .now() + timeout) != .timedOut else {
      return nil
    }
    return responseData
  }

  static func fetchRemoteData(url: URL, timeout: TimeInterval = 10) async -> Data? {
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout

    do {
      let (data, response) = try await URLSession.shared.data(for: request)
      guard let response = response as? HTTPURLResponse, 200...299 ~= response.statusCode else {
        return nil
      }
      return data
    } catch {
      return nil
    }
  }
}
