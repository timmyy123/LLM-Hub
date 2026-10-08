import LLM
import NNC
import XCTest

@testable import PromptJSONExpansion

final class Ideogram4PromptJSONExpansionTests: XCTestCase {
  func testModelConfiguration() {
    XCTAssertEqual(Ideogram4PromptJSONExpander.Model.qwen3_5_4B.file, "qwen_3.5_4b_i8x.ckpt")
    XCTAssertEqual(Ideogram4PromptJSONExpander.Model.qwen3_5_4B.configuration.layers, 32)
    XCTAssertEqual(Ideogram4PromptJSONExpander.Model.qwen3_5_4B.configuration.hiddenSize, 2_560)
    XCTAssertTrue(Ideogram4PromptJSONExpander.Model.qwen3_5_4B.tieEmbedding)
    XCTAssertEqual(Ideogram4PromptJSONExpander.Model.qwen3_5_9B.file, "qwen_3.5_9b_i5x.ckpt")
    XCTAssertEqual(Ideogram4PromptJSONExpander.Model.qwen3_5_9B.configuration.layers, 32)
    XCTAssertEqual(Ideogram4PromptJSONExpander.Model.qwen3_5_9B.configuration.hiddenSize, 4_096)
    XCTAssertFalse(Ideogram4PromptJSONExpander.Model.qwen3_5_9B.tieEmbedding)
  }

  func testAspectRatioReduction() {
    XCTAssertEqual(Ideogram4PromptJSONExpander.aspectRatio(width: 1_024, height: 1_024), "1:1")
    XCTAssertEqual(Ideogram4PromptJSONExpander.aspectRatio(width: 1_536, height: 1_024), "3:2")
  }

  func testOfficialUserPrompt() {
    XCTAssertEqual(
      Ideogram4PromptJSONExpander.userPrompt(
        prompt: "a café sign", width: 1_024, height: 1_280),
      "TARGET IMAGE ASPECT RATIO: 4:5 (width:height).\n\nUser idea: a café sign")
  }

  func testThinkingIsDisabled() {
    let prompt = Ideogram4PromptJSONExpander.chatPrompt(
      prompt: "a cube", width: 1_024, height: 1_024)
    XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\nYou convert a natural-language user idea"))
    XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
  }

  func testCachePrefixIsChunkAlignedAndPromptIndependent() {
    let chunkSize = 2_048
    let systemTokenIDs = Ideogram4PromptJSONExpander.cachePrefixTokenIDs(prefillChunkSize: 1)
    let prefixTokenIDs = Ideogram4PromptJSONExpander.cachePrefixTokenIDs(
      prefillChunkSize: chunkSize)
    XCTAssertFalse(prefixTokenIDs.isEmpty)
    XCTAssertEqual(prefixTokenIDs.count % chunkSize, 0)
    XCTAssertLessThan(systemTokenIDs.count - prefixTokenIDs.count, chunkSize)
    XCTAssertEqual(Array(systemTokenIDs.prefix(prefixTokenIDs.count)), prefixTokenIDs)
    XCTAssertTrue(
      Ideogram4PromptJSONExpander.promptTokenIDs(
        prompt: "a dog", width: 1_024, height: 1_024
      ).starts(with: prefixTokenIDs))
    XCTAssertTrue(
      Ideogram4PromptJSONExpander.promptTokenIDs(
        prompt: "a city at night", width: 1_536, height: 1_024
      ).starts(with: prefixTokenIDs))
  }

  func testCacheFingerprintMatchesLocalCodeEncoding() {
    let tokenIDs: [Int32] = [1, 0x0102_0304]
    let fingerprint = Qwen3_5PromptJSONPrefixCache<Float16>.fingerprint(
      modelFilePath: "/Models/model.ckpt", tokenIds: tokenIDs)
    XCTAssertEqual(
      fingerprint, "e06995781680170ad64473da431238ddc479dd890f2b8cf507dfd3f4eb9a8a92")
    XCTAssertEqual(
      fingerprint,
      Qwen3_5PromptJSONPrefixCache<Float16>.fingerprint(
        modelFilePath: "/another/location/model.ckpt", tokenIds: tokenIDs))
    XCTAssertNotEqual(
      fingerprint,
      Qwen3_5PromptJSONPrefixCache<Float16>.fingerprint(
        modelFilePath: "/Models/model.ckpt", tokenIds: tokenIDs + [2]))
  }

  func testCacheRoundTripAndIncompleteStoreFallback() throws {
    let cacheDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: cacheDirectory) }
    let tokenIDs: [Int32] = [10, 20]
    let cache = try XCTUnwrap(
      Qwen3_5PromptJSONPrefixCache<Float16>(
        modelFilePath: "/Models/model.ckpt", tokenIds: tokenIDs,
        cacheDirectoryURL: cacheDirectory))
    let configuration = Qwen3_5ModelConfiguration(
      vocabularySize: 16, hiddenSize: 4, intermediateSize: 8, layers: 2,
      fullAttentionInterval: 2, attentionHeads: 1, keyValueHeads: 1,
      attentionHeadDim: 2, rotaryDim: 2, mropeSection: (temporal: 1, height: 0, width: 0),
      ropeTheta: 10_000, linearNumKeyHeads: 1, linearNumValueHeads: 1,
      linearKeyHeadDim: 1, linearValueHeadDim: 1, linearConvKernel: 3)
    let graph = DynamicGraph()
    let sourceCaches: [DynamicGraph.AnyTensor] = [
      graph.variable(
        Tensor<Float16>([1, 2, 3, 4, 5, 6], .CPU, .NHWC(1, 2, 1, 3)).toGPU(0)),
      graph.variable(Tensor<Float>([7], .CPU, .NHWC(1, 1, 1, 1)).toGPU(0)),
      graph.variable(
        Tensor<Float16>([10, 11, 12, 13, 14, 15, 16, 17], .CPU, .NHWC(1, 4, 1, 2))
          .toGPU(0)),
      graph.variable(
        Tensor<Float16>([20, 21, 22, 23, 24, 25, 26, 27], .CPU, .NHWC(1, 4, 1, 2))
          .toGPU(0)),
    ]
    cache.save(graph: graph, configuration: configuration, caches: sourceCaches)

    var restoredCaches: [DynamicGraph.AnyTensor] = [
      graph.variable(
        Tensor<Float16>(Array(repeating: 0, count: 6), .CPU, .NHWC(1, 2, 1, 3)).toGPU(0)),
      graph.variable(Tensor<Float>([0], .CPU, .NHWC(1, 1, 1, 1)).toGPU(0)),
      graph.variable(
        Tensor<Float16>(Array(repeating: 0, count: 8), .CPU, .NHWC(1, 4, 1, 2)).toGPU(0)),
      graph.variable(
        Tensor<Float16>(Array(repeating: 0, count: 8), .CPU, .NHWC(1, 4, 1, 2)).toGPU(0)),
    ]
    XCTAssertTrue(
      cache.restore(graph: graph, configuration: configuration, caches: &restoredCaches))
    graph.joined()
    let restoredConv = restoredCaches[0].as(of: Float16.self).toCPU().reshaped(.C(6))
    let restoredRecurrent = restoredCaches[1].as(of: Float.self).toCPU().reshaped(.C(1))
    let restoredKey = restoredCaches[2].as(of: Float16.self).toCPU().reshaped(.C(8))
    let restoredValue = restoredCaches[3].as(of: Float16.self).toCPU().reshaped(.C(8))
    XCTAssertEqual((0..<6).map { restoredConv[$0] }, [1, 2, 3, 4, 5, 6])
    XCTAssertEqual(restoredRecurrent[0], 7)
    XCTAssertEqual((0..<8).map { restoredKey[$0] }, [10, 11, 12, 13, 0, 0, 0, 0])
    XCTAssertEqual((0..<8).map { restoredValue[$0] }, [20, 21, 22, 23, 0, 0, 0, 0])

    let storePath = cacheDirectory.appendingPathComponent("context_cache.sqlite3").path
    try FileManager.default.removeItem(atPath: storePath)
    _ = graph.openStore(storePath, flags: []) { _ in }
    XCTAssertFalse(
      cache.restore(graph: graph, configuration: configuration, caches: &restoredCaches))
  }

  func testUnavailableCacheDirectoryDisablesCache() throws {
    let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
      at: temporaryDirectory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let fileURL = temporaryDirectory.appendingPathComponent("not-a-directory")
    try Data().write(to: fileURL)
    XCTAssertNil(
      Qwen3_5PromptJSONPrefixCache<Float16>(
        modelFilePath: "/Models/model.ckpt", tokenIds: [1], cacheDirectoryURL: fileURL))
  }

  func testCompleteJSONObject() {
    XCTAssertTrue(
      Ideogram4PromptJSONExpander.isCompleteJSONObject(
        #"{"aspect_ratio":"1:1","high_level_description":"A café sign.","compositional_deconstruction":{"background":"A wall.","elements":[]}}"#
      ))
    XCTAssertFalse(
      Ideogram4PromptJSONExpander.isCompleteJSONObject(
        #"{"aspect_ratio":"1:1","high_level_description":"unfinished""#))
  }
}
