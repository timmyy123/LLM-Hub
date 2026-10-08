import CryptoKit
import Darwin
import Foundation
import LeanBridge
import ZIPFoundation

@objc public final class LeanRuntimeResources: NSObject {
  struct Manifest: Decodable {
    struct Module: Decodable {
      struct File: Decodable {
        let path: String
        let size: UInt64
        let sha256: String
      }
      let imports: [String]
      let files: [File]
    }
    let schemaVersion: Int
    let leanVersion: String
    let archiveSHA256: String
    let modules: [String: Module]
  }

  enum Error: LocalizedError {
    case invalidManifest
    case versionMismatch(String)
    case checksum(String)
    case invalidEntry(String)
    case metadata(String)
    case publish(String)

    var errorDescription: String? {
      switch self {
      case .invalidManifest: return "Invalid Lean resource manifest."
      case .versionMismatch(let version): return "Lean resource version mismatch: \(version)."
      case .checksum(let path): return "Lean resource checksum mismatch: \(path)."
      case .invalidEntry(let path): return "Invalid Lean archive entry: \(path)."
      case .metadata(let name): return "Cannot read Lean import metadata: \(name)."
      case .publish(let path): return "Cannot publish Lean cache file: \(path)."
      }
    }
  }

  // Own the cache lease through both extraction and checking. All instances in
  // this process, including future shell/tool clients, share this lock.
  private static let operationLock = NSLock()
  private let manifest: Manifest
  private let archive: Archive
  private let entries: [String: Entry]
  private let fileManager = FileManager.default
  @objc public let cacheRoot: URL
  @objc public var archiveModuleCount: Int { manifest.modules.count }
  @objc public private(set) var extractedModuleCount = 0
  @objc public private(set) var extractedByteCount: UInt64 = 0
  @objc public private(set) var requestedModuleCount = 0

  @objc public init(archiveURL: URL, manifestURL: URL, cacheURL: URL) throws {
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
    guard manifest.schemaVersion == 1,
      manifest.archiveSHA256.count == 64,
      manifest.archiveSHA256.allSatisfy({ "0123456789abcdef".contains($0) })
    else { throw Error.invalidManifest }
    guard manifest.leanVersion == String(cString: lean_bridge_version()) else {
      throw Error.versionMismatch(manifest.leanVersion)
    }
    guard try Self.checksum(archiveURL) == manifest.archiveSHA256 else {
      throw Error.checksum(archiveURL.lastPathComponent)
    }
    guard let archive = Archive(url: archiveURL, accessMode: .read) else {
      throw Error.invalidEntry(archiveURL.lastPathComponent)
    }
    var entries = [String: Entry]()
    for entry in archive {
      guard entry.type == .file, entries.updateValue(entry, forKey: entry.path) == nil else {
        throw Error.invalidEntry(entry.path)
      }
    }
    var fileCount = 0
    for (name, module) in manifest.modules {
      let stem = name.replacingOccurrences(of: ".", with: "/")
      guard !stem.isEmpty, !stem.contains("\\"), !stem.contains("\0"),
        stem.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty }),
        Set(module.files.map(\.path)) == Set(["\(stem).olean", "\(stem).ir.sig", "\(stem).ir"]),
        module.files.count == 3
      else { throw Error.invalidManifest }
      for file in module.files {
        guard let entry = entries[file.path], entry.uncompressedSize == file.size,
          file.sha256.count == 64,
          file.sha256.allSatisfy({ "0123456789abcdef".contains($0) })
        else { throw Error.invalidEntry(file.path) }
        fileCount += 1
      }
    }
    guard fileCount == entries.count else { throw Error.invalidManifest }
    self.manifest = manifest
    self.archive = archive
    self.entries = entries
    self.cacheRoot = cacheURL.appendingPathComponent(manifest.leanVersion, isDirectory: true)
      .appendingPathComponent(manifest.archiveSHA256, isDirectory: true)
    super.init()
  }

  @objc public func check(source: String, packageRoot: String?) -> String {
    Self.operationLock.lock()
    defer { Self.operationLock.unlock() }
    do {
      guard let buffer = lean_bridge_source_imports(source) else { throw Error.metadata("source") }
      let imports: [String]
      do {
        defer { lean_bridge_free(buffer) }
        imports = try JSONDecoder().decode([String].self, from: Data(String(cString: buffer).utf8))
      }
      let artifacts = try prepare(modules: imports, packageRoot: packageRoot)
      let json = String(decoding: try JSONEncoder().encode(artifacts), as: UTF8.self)
      guard let result = lean_bridge_check(source, cacheRoot.path, packageRoot ?? "", json) else {
        throw Error.metadata("checker result")
      }
      defer { lean_bridge_free(result) }
      return String(cString: result)
    } catch {
      return "status: error\nerror: \(error.localizedDescription)\n"
    }
  }

  @objc public func clearCache() throws {
    Self.operationLock.lock()
    defer { Self.operationLock.unlock() }
    if fileManager.fileExists(atPath: cacheRoot.path) {
      try fileManager.removeItem(at: cacheRoot)
    }
  }

  @objc public func runCommand(
    source: String, fileName: String, packageRoot: String?, options: UnsafeMutableRawPointer,
    audit: Bool, input: UnsafeMutablePointer<FILE>, outputFD: Int32, errorFD: Int32,
    cancelled: @escaping () -> Bool
  ) -> Int32 {
    while !Self.operationLock.lock(before: Date(timeIntervalSinceNow: 0.025)) {
      if cancelled() { return 130 }
    }
    defer { Self.operationLock.unlock() }
    do {
      if cancelled() { return 130 }
      guard let buffer = lean_bridge_source_imports(source) else { throw Error.metadata("source") }
      let imports: [String]
      do {
        defer { lean_bridge_free(buffer) }
        imports = try JSONDecoder().decode([String].self, from: Data(String(cString: buffer).utf8))
      }
      let artifacts = try prepare(modules: imports, packageRoot: packageRoot, cancelled: cancelled)
      let json = String(decoding: try JSONEncoder().encode(artifacts), as: UTF8.self)
      return withUnsafePointer(to: cancelled) { context in
        lean_bridge_run_command(
          source, fileName, cacheRoot.path, packageRoot ?? "", json, options,
          audit ? 1 : 0, input, outputFD, errorFD,
          { context in
            context!.assumingMemoryBound(to: (() -> Bool).self).pointee() ? 1 : 0
          }, context)
      }
    } catch {
      if cancelled() { return 130 }
      let text = (audit ? "status: error\n" : "") + "error: \(error.localizedDescription)\n"
      text.withCString { _ = Darwin.write(audit ? outputFD : errorFD, $0, text.utf8.count) }
      return 1
    }
  }

  // Called under the operation lock in production. Exposed internally for focused
  // archive/cache tests that do not need to elaborate a Lean proof.
  func prepare(modules: [String], packageRoot: String?, cancelled: () -> Bool = { false }) throws
    -> [String: [[String]]]
  {
    extractedModuleCount = 0
    extractedByteCount = 0
    requestedModuleCount = 0
    let packageRoots = (packageRoot ?? "").split(separator: ":").map {
      URL(fileURLWithPath: String($0), isDirectory: true)
    }
    var pending = modules
    var visited = Set<String>()
    var artifacts = [String: [[String]]]()
    while let name = pending.popLast() {
      if cancelled() { throw CancellationError() }
      guard visited.insert(name).inserted else { continue }
      let relative = name.replacingOccurrences(of: ".", with: "/") + ".olean"
      // Match Lean's package-before-base search order. Never replace a package
      // module with a bundled module of the same name.
      if let olean = packageRoots.map({ $0.appendingPathComponent(relative) })
        .first(where: { fileManager.fileExists(atPath: $0.path) })
      {
        guard let buffer = lean_bridge_module_imports(olean.path) else {
          throw Error.metadata(name)
        }
        defer { lean_bridge_free(buffer) }
        pending += try JSONDecoder().decode([String].self, from: Data(String(cString: buffer).utf8))
        let irSig = olean.deletingPathExtension().appendingPathExtension("ir.sig")
        let ir = olean.deletingPathExtension().appendingPathExtension("ir")
        let irFiles =
          fileManager.fileExists(atPath: irSig.path) && fileManager.fileExists(atPath: ir.path)
          ? [irSig.path, ir.path] : []
        artifacts[name] = [[olean.path], irFiles]
      } else if let module = manifest.modules[name] {
        pending += module.imports
        try materialize(module, name: name)
        let olean = cacheRoot.appendingPathComponent(relative)
        artifacts[name] = [
          [olean.path],
          [
            olean.deletingPathExtension().appendingPathExtension("ir.sig").path,
            olean.deletingPathExtension().appendingPathExtension("ir").path,
          ],
        ]
        requestedModuleCount += 1
      }
      // Leave unknown modules to Lean's normal import diagnostic.
    }
    return artifacts
  }

  private func materialize(_ module: Manifest.Module, name: String) throws {
    let marker = cacheRoot.appendingPathComponent(".complete", isDirectory: true)
      .appendingPathComponent(name.replacingOccurrences(of: ".", with: "/") + ".complete")
    if (try? String(contentsOf: marker, encoding: .utf8)) == manifest.archiveSHA256 {
      var intact = true
      for file in module.files {
        let url = cacheRoot.appendingPathComponent(file.path)
        let attributes = try? fileManager.attributesOfItem(atPath: url.path)
        if attributes?[.type] as? FileAttributeType != .typeRegular
          || (attributes?[.size] as? NSNumber)?.uint64Value != file.size
          || (try? Self.checksum(url)) != file.sha256
        {
          intact = false
          break
        }
      }
      if intact { return }
    }
    let staging = cacheRoot.appendingPathComponent(
      ".staging-\(UUID().uuidString)", isDirectory: true)
    try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: staging) }
    for file in module.files {
      guard let entry = entries[file.path] else { throw Error.invalidEntry(file.path) }
      let destination = staging.appendingPathComponent(file.path)
      let checksum = try archive.extract(entry, to: destination)
      guard checksum == entry.checksum, try Self.checksum(destination) == file.sha256 else {
        throw Error.checksum(file.path)
      }
    }
    // All three facets have been verified before publishing any of them. Each
    // rename is atomic; the completion marker is published last. A crash or purge
    // leaving a partial module is repaired on the next request.
    for file in module.files {
      let source = staging.appendingPathComponent(file.path)
      let destination = cacheRoot.appendingPathComponent(file.path)
      try fileManager.createDirectory(
        at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
      guard rename(source.path, destination.path) == 0 else { throw Error.publish(file.path) }
      extractedByteCount += file.size
    }
    try fileManager.createDirectory(
      at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(manifest.archiveSHA256.utf8).write(to: marker, options: .atomic)
    extractedModuleCount += 1
  }

  private static func checksum(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hash = SHA256()
    while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
      hash.update(data: data)
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
