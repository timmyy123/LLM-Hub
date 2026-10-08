import ModelZoo
import XCTest

final class ModelZooTests: XCTestCase {
  func testLLMDownloadRequiresEngramWithoutChangingPerFileChecks() {
    let file = "\(UUID().uuidString).ckpt"
    let engram = "\(UUID().uuidString).engram"
    let specification = LLMZoo.Specification(
      name: "Engram model", file: file, engram: engram, version: .deepseek_4_1_flash)

    XCTAssertFalse(LLMZoo.isModelDownloaded(specification))
    XCTAssertFalse(LLMZoo.isModelDownloaded(specification, memorizedBy: [file]))
    XCTAssertFalse(LLMZoo.isModelDownloaded(specification, memorizedBy: [engram]))
    XCTAssertTrue(LLMZoo.isModelDownloaded(specification, memorizedBy: [file, engram]))
    XCTAssertTrue(LLMZoo.isModelDownloaded(file, memorizedBy: [file]))
  }

  func testLLMSpecificationDecodesWithoutEngram() throws {
    let json = """
      {"name":"Existing model","file":"existing.ckpt","version":"deepseek_4_flash"}
      """
    let specification = try JSONDecoder().decode(
      LLMZoo.Specification.self, from: Data(json.utf8))

    XCTAssertNil(specification.engram)
    XCTAssertTrue(LLMZoo.isModelDownloaded(specification, memorizedBy: [specification.file]))
  }

  func testMiniMaxH3FirstBlockCacheUsesIdentityCoefficients() {
    let name = "test-minimax-h3-community-cache.ckpt"
    let previous = ModelZoo.overrideMapping[name]
    defer { ModelZoo.overrideMapping[name] = previous }
    ModelZoo.overrideMapping[name] = ModelZoo.Specification(
      name: name, file: name, prefix: "", version: .minimaxH3)
    let coefficients = ModelZoo.teaCacheCoefficientsForModel(name)
    XCTAssertNotNil(coefficients)
    XCTAssertEqual(coefficients?.0, 0)
    XCTAssertEqual(coefficients?.1, 0)
    XCTAssertEqual(coefficients?.2, 0)
    XCTAssertEqual(coefficients?.3, 1)
    XCTAssertEqual(coefficients?.4, 0)
  }

  func testMiniMaxH3CheckpointSupportsFirstFrameConditioning() {
    XCTAssertEqual(ModelZoo.modifierForModel("minimax_h3_i8x.ckpt"), .fl2va)
  }

  func testMiniMaxH3DefaultLatentsScaling() {
    let name = "test-minimax-h3-default-latents.ckpt"
    let previous = ModelZoo.overrideMapping[name]
    defer { ModelZoo.overrideMapping[name] = previous }
    ModelZoo.overrideMapping[name] = ModelZoo.Specification(
      name: name, file: name, prefix: "", version: .minimaxH3)

    let scaling = ModelZoo.latentsScalingForModel(name)
    XCTAssertEqual(scaling.mean?.count, 24)
    XCTAssertEqual(scaling.std?.count, 24)
    XCTAssertEqual(scaling.audioMean?.count, 32)
    XCTAssertEqual(scaling.audioStd?.count, 32)
    XCTAssertEqual(scaling.mean?.first, 0.85809034)
    XCTAssertEqual(scaling.mean?.last, 0.7056032)
    XCTAssertEqual(scaling.std?.first, 1.2223774)
    XCTAssertEqual(scaling.std?.last, 2.6127844)
    XCTAssertEqual(scaling.audioMean?.first, -0.020211687)
    XCTAssertEqual(scaling.audioMean?.last, 0.39792585)
    XCTAssertEqual(scaling.audioStd?.first, 1.6895524)
    XCTAssertEqual(scaling.audioStd?.last, 1.5613768)
    XCTAssertEqual(scaling.scalingFactor, 1)
    XCTAssertNil(scaling.shiftFactor)

    let builtin = ModelZoo.latentsScalingForModel("minimax_h3_i8x.ckpt")
    XCTAssertEqual(scaling.mean, builtin.mean)
    XCTAssertEqual(scaling.std, builtin.std)
    XCTAssertEqual(scaling.audioMean, builtin.audioMean)
    XCTAssertEqual(scaling.audioStd, builtin.audioStd)
    XCTAssertEqual(scaling.scalingFactor, builtin.scalingFactor)
    XCTAssertEqual(scaling.shiftFactor, builtin.shiftFactor)
  }

  func testMiniMaxH3ExplicitLatentsScalingTakesPrecedence() {
    let name = "test-minimax-h3-explicit-latents.ckpt"
    let previous = ModelZoo.overrideMapping[name]
    defer { ModelZoo.overrideMapping[name] = previous }
    var specification = ModelZoo.Specification(
      name: name, file: name, prefix: "", version: .minimaxH3,
      latentsMean: Array(repeating: 0.5, count: 24),
      latentsStd: Array(repeating: 2, count: 24),
      audioLatentsMean: Array(repeating: -0.5, count: 32),
      audioLatentsStd: Array(repeating: 3, count: 32), latentsScalingFactor: 0.75)
    ModelZoo.overrideMapping[name] = specification

    let scaling = ModelZoo.latentsScalingForModel(name)
    XCTAssertEqual(scaling.mean, specification.latentsMean)
    XCTAssertEqual(scaling.std, specification.latentsStd)
    XCTAssertEqual(scaling.audioMean, specification.audioLatentsMean)
    XCTAssertEqual(scaling.audioStd, specification.audioLatentsStd)
    XCTAssertEqual(scaling.scalingFactor, 0.75)
    XCTAssertNil(scaling.shiftFactor)

    // Preserve the shared contract for specifications that override only scaling.
    specification.latentsMean = nil
    specification.latentsStd = nil
    ModelZoo.overrideMapping[name] = specification
    let scalingOnly = ModelZoo.latentsScalingForModel(name)
    XCTAssertNil(scalingOnly.mean)
    XCTAssertNil(scalingOnly.std)
    XCTAssertEqual(scalingOnly.scalingFactor, 0.75)
    XCTAssertEqual(scalingOnly.audioMean, specification.audioLatentsMean)
    XCTAssertEqual(scalingOnly.audioStd, specification.audioLatentsStd)
  }
}
