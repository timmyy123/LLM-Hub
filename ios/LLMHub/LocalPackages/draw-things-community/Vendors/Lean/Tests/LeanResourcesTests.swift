import CryptoKit
import Foundation
import XCTest
import ZIPFoundation

@testable import LeanResources

final class LeanResourcesTests: XCTestCase {
  private var directory: URL!
  private var archiveURL: URL { directory.appendingPathComponent("runtime.zip") }
  private var manifestURL: URL { directory.appendingPathComponent("runtime.json") }
  private var cacheURL: URL { directory.appendingPathComponent("cache", isDirectory: true) }

  override func setUpWithError() throws {
    directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let archive = try XCTUnwrap(Archive(url: archiveURL, accessMode: .create))
    var modules = [String: Any]()
    for (name, imports) in [("A", ["B"]), ("A.Child", ["B"]), ("B", []), ("Unrelated", [])] {
      var files = [[String: Any]]()
      for suffix in ["olean", "ir.sig", "ir"] {
        let path = "\(name.replacingOccurrences(of: ".", with: "/")).\(suffix)"
        let data = Data("fixture: \(path)".utf8)
        try archive.addEntry(
          with: path, type: .file, uncompressedSize: Int64(data.count),
          compressionMethod: .deflate
        ) { position, size in
          data.subdata(in: Int(position)..<min(Int(position) + size, data.count))
        }
        files.append(["path": path, "size": data.count, "sha256": digest(data)])
      }
      modules[name] = ["imports": imports, "files": files]
    }
    try JSONSerialization.data(withJSONObject: [
      "schemaVersion": 1, "leanVersion": "4.34.0",
      "archiveSHA256": digest(Data(contentsOf: archiveURL)), "modules": modules,
    ]).write(to: manifestURL)
  }

  override func tearDownWithError() throws {
    try FileManager.default.removeItem(at: directory)
  }

  private func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private func resources() throws -> LeanRuntimeResources {
    try LeanRuntimeResources(archiveURL: archiveURL, manifestURL: manifestURL, cacheURL: cacheURL)
  }

  private func changeManifest(_ edit: (inout [String: Any]) -> Void) throws {
    var manifest = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
    edit(&manifest)
    try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL)
  }

  func testFocusedClosureAndWarmReuse() throws {
    let resources = try resources()
    XCTAssertFalse(FileManager.default.fileExists(atPath: resources.cacheRoot.path))
    let artifacts = try resources.prepare(modules: ["A"], packageRoot: nil)
    XCTAssertEqual(Set(artifacts.keys), ["A", "B"])
    XCTAssertEqual(resources.extractedModuleCount, 2)
    XCTAssertEqual(artifacts["A"]?.map(\.count), [1, 2])
    XCTAssertFalse(
      FileManager.default.fileExists(
        atPath: resources.cacheRoot.appendingPathComponent("Unrelated.olean").path))
    let file = resources.cacheRoot.appendingPathComponent("A.olean")
    let before =
      try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate] as? Date
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 0)
    XCTAssertEqual(resources.extractedByteCount, 0)
    XCTAssertEqual(
      before,
      try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate] as? Date)
  }

  func testLaterImportsExpandCache() throws {
    let resources = try resources()
    _ = try resources.prepare(modules: ["B"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 1)
    _ = try resources.prepare(modules: ["A", "A"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 1)
    XCTAssertEqual(resources.requestedModuleCount, 2)
  }

  func testReuseAcrossResourceInstances() throws {
    let first = try resources()
    _ = try first.prepare(modules: ["A"], packageRoot: nil)
    let second = try resources()
    _ = try second.prepare(modules: ["A"], packageRoot: nil)
    XCTAssertEqual(second.extractedModuleCount, 0)
  }

  func testNamespaceParentAndChildMarkersDoNotCollide() throws {
    let resources = try resources()
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    _ = try resources.prepare(modules: ["A.Child"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 1)
    _ = try resources.prepare(modules: ["A", "A.Child"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 0)
  }

  func testMissingFacetIsRepaired() throws {
    let resources = try resources()
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    let file = resources.cacheRoot.appendingPathComponent("A.ir")
    try FileManager.default.removeItem(at: file)
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 1)
    XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "fixture: A.ir")
  }

  func testSameSizeCorruptionIsRepaired() throws {
    let resources = try resources()
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    let file = resources.cacheRoot.appendingPathComponent("A.olean")
    let size = try Data(contentsOf: file).count
    try Data(repeating: 0, count: size).write(to: file)
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 1)
    XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "fixture: A.olean")
  }

  func testInterruptedModuleWithoutMarkerIsRepaired() throws {
    let resources = try resources()
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    try FileManager.default.removeItem(
      at: resources.cacheRoot.appendingPathComponent(".complete/A.complete"))
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 1)
  }

  func testPurgedCacheIsRebuilt() throws {
    let resources = try resources()
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    try resources.clearCache()
    _ = try resources.prepare(modules: ["A"], packageRoot: nil)
    XCTAssertEqual(resources.extractedModuleCount, 2)
  }

  func testWrongArchiveHashIsRejected() throws {
    try changeManifest { $0["archiveSHA256"] = String(repeating: "0", count: 64) }
    XCTAssertThrowsError(try resources())
    XCTAssertFalse(FileManager.default.fileExists(atPath: cacheURL.path))
  }

  func testWrongVersionIsRejected() throws {
    try changeManifest { $0["leanVersion"] = "4.34.0-rc2" }
    XCTAssertThrowsError(try resources())
  }

  func testWrongEntryHashDoesNotPublishModule() throws {
    try changeManifest { manifest in
      var modules = manifest["modules"] as! [String: [String: Any]]
      var files = modules["B"]!["files"] as! [[String: Any]]
      files[2]["sha256"] = String(repeating: "0", count: 64)
      modules["B"]!["files"] = files
      manifest["modules"] = modules
    }
    let resources = try resources()
    XCTAssertThrowsError(try resources.prepare(modules: ["B"], packageRoot: nil))
    XCTAssertFalse(
      FileManager.default.fileExists(
        atPath: resources.cacheRoot.appendingPathComponent("B.olean").path))
    XCTAssertFalse(
      FileManager.default.fileExists(
        atPath: resources.cacheRoot.appendingPathComponent(".complete/B.complete").path))
  }

  func testTraversalManifestIsRejected() throws {
    try changeManifest { manifest in
      var modules = manifest["modules"] as! [String: [String: Any]]
      modules["../escape"] = modules.removeValue(forKey: "B")
      manifest["modules"] = modules
    }
    XCTAssertThrowsError(try resources())
  }
}
