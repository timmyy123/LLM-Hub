import Foundation
import Utils

public struct ScriptZoo {

  public enum ScriptType: String, Codable, Equatable, Hashable {
    case user
    case sample
    case community
  }

  public struct Image: Codable, Equatable, Hashable {
    public var url: String
    public var tags: [String]?

    public init(
      url: String, tags: [String]
    ) {
      self.url = url
      self.tags = tags
    }
  }

  public struct Script: Codable, Equatable, Hashable {
    // display name/metadata name
    public var name: String
    // local script folder file name
    public var file: String
    // file path for downloaded script
    public var filePath: String?
    public var isSampleDuplicate: Bool?
    public var type: ScriptType?
    public var description: String?
    public var author: String?
    public var tags: [String]?
    public var images: [Image]?
    public var baseColor: String?
    public var favicon: String?
    public init(
      name: String, file: String, filePath: String?, isSampleDuplicate: Bool? = nil,
      type: ScriptType? = nil, description: String? = nil, author: String? = nil,
      tags: [String]? = nil, images: [Image]? = nil, baseColor: String? = nil,
      favicon: String? = nil
    ) {
      self.name = name
      self.file = file
      self.filePath = filePath
      self.isSampleDuplicate = isSampleDuplicate
      self.type = type
      self.description = description
      self.author = author
      self.tags = tags
      self.images = images
      self.baseColor = baseColor
      self.favicon = favicon
    }
  }

  public static let scriptsUrl: URL = {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return urls.first!.appendingPathComponent("Scripts")
  }()

  public static var availableScripts: [Script] {
    var userScriptNames = [String: Int]()
    var scripts =
      ((try? FileManager.default.contentsOfDirectory(
        at: scriptsUrl,
        includingPropertiesForKeys: [],
        options: .skipsHiddenFiles)) ?? []).filter({ $0.pathExtension == "js" })
      .enumerated().map {
        let filePath = $1.path
        let file = $1.lastPathComponent
        let script = Script(
          name: file, file: file, filePath: filePath, type: .user)
        userScriptNames[file] = $0
        return script
      }
    // Sort file list case-insensitive for user display.
    scripts = scripts.sorted {
      $0.name.localizedStandardCompare($1.name) == .orderedAscending
    }
    for path in Bundle.main.paths(forResourcesOfType: ".sample.js", inDirectory: nil) {
      if let filename = path.split(separator: "/").last {
        let name = filename.replacingOccurrences(of: ".sample", with: "")
        guard !userScriptNames.keys.contains(name) else {
          scripts[userScriptNames[name]!].isSampleDuplicate = true
          continue
        }
        scripts.append(
          Script(name: String(name), file: String(filename), filePath: path, type: .sample)
        )
      }
    }
    guard
      let jsonData = try? Data(contentsOf: scriptsUrl.appendingPathComponent("custom_scripts.json"))
    else {
      return scripts
    }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let scriptsMetadata = try? jsonDecoder.decode([Script].self, from: jsonData)
    else {
      return scripts
    }
    var scriptFileToMetadata: [String: Script] = [:]
    for metadata in scriptsMetadata {
      scriptFileToMetadata[metadata.file] = metadata
    }
    return scripts.map({
      guard var script = scriptFileToMetadata[$0.file] else { return $0 }
      if script.filePath == nil {
        script.filePath = scriptsUrl.appendingPathComponent(script.file).path
      }
      return script
    })
  }

  private static var jsHeaderDoc: String = {
    return
      (try? String(
        contentsOfFile: Bundle.main.path(forResource: "js-doc-header", ofType: ".js")!,
        encoding: .utf8)) ?? ""
  }()

  public static func removeScript(name: String) {
    try? FileManager.default.removeItem(at: scriptsUrl.appendingPathComponent(name))
  }

  public static func duplicateScript(_ script: Script) {
    var counter = 1
    var filename = script.name
    let insertionIndex = filename.index(filename.endIndex, offsetBy: -3)
    filename.insert(contentsOf: "-\(counter)", at: insertionIndex)
    while FileManager.default.fileExists(atPath: scriptsUrl.appendingPathComponent(filename).path) {
      counter += 1
      filename = script.name
      let insertionIndex = filename.index(filename.endIndex, offsetBy: -3)
      filename.insert(contentsOf: "-\(counter)", at: insertionIndex)
    }
    if let filePath = script.filePath {
      try? FileManager.default.copyItem(
        at: URL(fileURLWithPath: filePath), to: scriptsUrl.appendingPathComponent(filename))
    }
  }

  public static func save(_ content: String, to file: String, script: Script? = nil) {
    try? FileManager.default.createDirectory(at: scriptsUrl, withIntermediateDirectories: true)
    var presetContent = content
    if presetContent == "" {
      presetContent = jsHeaderDoc
    }
    do {
      try presetContent.write(
        to: scriptsUrl.appendingPathComponent(file), atomically: true, encoding: .utf8)
    } catch {
      return
    }
    // If there is no error, append to the disk if there is a script metadata.
    guard var script = script else { return }
    script.file = file
    script.filePath = nil
    let completionHandler: ([Script]) -> Void = {
      let jsonEncoder = JSONEncoder()
      jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
      jsonEncoder.outputFormatting = .prettyPrinted
      if let data = try? jsonEncoder.encode($0) {
        try? data.write(
          to: scriptsUrl.appendingPathComponent("custom_scripts.json"), options: [.atomic])
      }
    }
    guard
      let jsonData = try? Data(contentsOf: scriptsUrl.appendingPathComponent("custom_scripts.json"))
    else {
      completionHandler([script])
      return
    }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard var scripts = try? jsonDecoder.decode([Script].self, from: jsonData) else {
      completionHandler([script])
      return
    }
    var updated = false
    for (i, availableScript) in scripts.enumerated() {
      if availableScript.file == script.file {
        scripts[i] = script
        updated = true
        break
      }
    }
    if !updated {
      scripts.append(script)
    }
    completionHandler(scripts)
  }

  public static func contentOf(_ path: String) -> String? {
    guard let data = FileManager.default.contents(atPath: path) else { return nil }
    return String(data: data, encoding: .utf8)
  }
}

extension ScriptZoo.Script {
  public var isImageToImage: Bool {
    tags?.first(where: { $0 == "image-to-image" }) != nil
  }
}
