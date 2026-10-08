import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC
import XCTest

final class MiniMaxH3LoRATests: XCTestCase {
  private func writeAdapter(to url: URL, fused: Bool) throws {
    var header = [String: Any]()
    var data = Data()
    func append(_ key: String, shape: [Int], values: [Float]) {
      let start = data.count
      values.withUnsafeBytes { data.append(contentsOf: $0) }
      header[key] = ["dtype": "F32", "shape": shape, "data_offsets": [start, data.count]]
    }
    func down(_ part: Int) -> [Float] {
      (0..<(2 * 5_376)).map { Float(($0 / 5_376 + 1) * (part + 1)) / 8 }
    }
    func up(_ row: Int, _ column: Int, _ part: Int) -> Float {
      Float((row % 128 + 1) * (column + 1) * (part + 1)) / 8
    }
    for (refiner, layer) in [(false, 0), (false, 49), (true, 0), (true, 1)] {
      let prefix: String
      if fused {
        prefix =
          refiner
          ? "diffusion_model.token_refiner.blocks.\(layer)" : "diffusion_model.blocks.\(layer)"
        append(
          "\(prefix).attn.qkv_proj.lora_A.weight", shape: [6, 5_376],
          values: (0..<3).flatMap { down($0) })
        var values = [Float](repeating: 0, count: 21_504 * 6)
        for part in 0..<3 {
          for row in 0..<7_168 {
            for column in 0..<2 {
              values[(part * 7_168 + row) * 6 + part * 2 + column] = up(row, column, part)
            }
          }
        }
        append("\(prefix).attn.qkv_proj.lora_B.weight", shape: [21_504, 6], values: values)
        append("\(prefix).attn.qkv_proj.alpha", shape: [], values: [6 * 0.0625])
      } else {
        prefix = refiner ? "token_refiner.refiner_blocks.\(layer)" : "transformer_blocks.\(layer)"
        for (part, name) in ["q", "k", "v"].enumerated() {
          append(
            "\(prefix).attn.to_\(name).lora_A.default.weight", shape: [2, 5_376], values: down(part)
          )
          append(
            "\(prefix).attn.to_\(name).lora_B.default.weight", shape: [7_168, 2],
            values: (0..<(7_168 * 2)).map { up($0 / 2, $0 % 2, part) })
        }
      }
      let ff = fused ? "\(prefix).mlp.fc1" : "\(prefix).ff.net.0.proj"
      let suffix = fused ? "" : ".default"
      append("\(ff).lora_A\(suffix).weight", shape: [2, 5_376], values: down(0))
      append(
        "\(ff).lora_B\(suffix).weight", shape: [28_672, 2],
        values: (0..<(28_672 * 2)).map {
          let part = $0 / (14_336 * 2)
          return up(($0 / 2) % 14_336, $0 % 2, fused ? 1 - part : part)
        })
      if fused { append("\(ff).alpha", shape: [], values: [2 * 0.0625]) }
    }
    var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    while json.count % 8 != 0 { json.append(0x20) }
    var length = UInt64(json.count).littleEndian
    var file = withUnsafeBytes(of: &length) { Data($0) }
    file.append(json)
    file.append(data)
    try file.write(to: url)
  }

  func testFusedAndUnfusedImportPreserveUpdatesAndPartialRotary() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let previousUrls = ModelZoo.externalUrls
    let previousPreference = ModelZoo.isExternalUrlsPreferred
    ModelZoo.isExternalUrlsPreferred = true
    ModelZoo.externalUrls = [directory]
    defer {
      ModelZoo.externalUrls = previousUrls
      ModelZoo.isExternalUrlsPreferred = previousPreference
      try? FileManager.default.removeItem(at: directory)
    }
    let graph = DynamicGraph()
    var imported = [[String: Tensor<Float>]]()
    for fused in [false, true] {
      let source = directory.appendingPathComponent(
        fused ? "fused.safetensors" : "split.safetensors")
      try writeAdapter(to: source, fused: fused)
      let filename = fused ? "fused.ckpt" : "split.ckpt"
      let (version, embedding, _, loHa) = try LoRAImporter.import(
        downloadedFile: source.path, name: "H3 test", filename: filename,
        scaleFactor: fused ? 1 : 0.0625, forceVersion: nil, progress: { _ in })
      XCTAssertEqual(version, .minimaxH3)
      XCTAssertFalse(embedding)
      XCTAssertFalse(loHa)
      var tensors = [String: Tensor<Float>]()
      graph.openStore(directory.appendingPathComponent(filename).path, flags: .readOnly) { store in
        for key in store.keys {
          if key.hasSuffix("__up__") || key.hasSuffix("__down__") {
            tensors[key] = store.read(key).map { Tensor<Float>(from: $0) }
          }
        }
      }
      XCTAssertEqual(tensors.count, 40)
      imported.append(tensors)
    }
    XCTAssertEqual(Set(imported[0].keys), Set(imported[1].keys))
    for (name, splitUp) in imported[0] where name.hasSuffix("__up__") {
      let downName = String(name.dropLast(6)) + "__down__"
      let splitDown = try XCTUnwrap(imported[0][downName])
      let fusedUp = try XCTUnwrap(imported[1][name])
      let fusedDown = try XCTUnwrap(imported[1][downName])
      XCTAssertEqual(Array(splitUp.shape), Array(fusedUp.shape))
      XCTAssertEqual(Array(splitDown.shape), Array(fusedDown.shape))
      XCTAssertEqual(splitUp.shape[1], 2)
      for row in 0..<splitUp.shape[0] {
        let split = splitUp[row, 0] * splitDown[0, 0] + splitUp[row, 1] * splitDown[1, 0]
        let fused = fusedUp[row, 0] * fusedDown[0, 0] + fusedUp[row, 1] * fusedDown[1, 0]
        XCTAssertEqual(split, fused, accuracy: 1e-6, name)
        let isQuery = name.contains("-q-") || name.contains("-refiner_q-")
        let isKey = name.contains("-k-") || name.contains("-refiner_k-")
        let isValue = name.contains("-v-") || name.contains("-refiner_v-")
        let part =
          isKey
          ? 1 : (isValue || name.contains("-gate-") || name.contains("-refiner_gate-") ? 2 : 0)
        let coordinate = row % 128
        let sourceRow: Int
        if (isQuery || isKey) && !name.contains("refiner") && coordinate < 96 {
          sourceRow = coordinate / 2 + (coordinate % 2) * 48
        } else {
          sourceRow = coordinate
        }
        // FFN gate has twice the up values but shares the same A.
        let multiplier =
          isQuery || isKey || isValue ? Float((part + 1) * (part + 1)) : (part == 2 ? 2 : 1)
        let expected = Float(sourceRow + 1) * 5 / 64 * 0.0625 * multiplier
        XCTAssertEqual(split, expected, accuracy: 1e-6, "\(name), row \(row)")
      }
    }
  }

  func testMappingCoversAllTurboLayers() throws {
    let (fixed, main, _, _) = LoRAImporter.modelWeightsMapping(
      by: .minimaxH3, qkNorm: true, dualAttentionLayers: [],
      format: [.diffusers, .generativeModels])
    for layer in 0..<50 {
      let q = main["transformer_blocks.\(layer).attn.to_q.weight"]!
      let fused = main["diffusion_model.blocks.\(layer).attn.qkv_proj.weight"]!
      XCTAssertEqual(q.first, fused.first)
      XCTAssertEqual(q.interleavedDimension, 96)
      XCTAssertEqual(q.interleavedIndices, [0])
      let diffusers = main["transformer_blocks.\(layer).ff.net.0.proj.weight"]!
      let comfy = main["diffusion_model.blocks.\(layer).mlp.fc1.weight"]!
      XCTAssertEqual(Array(diffusers), Array(comfy.reversed()))
      for norm in ["norm1", "norm2"] {
        let diffusers = try XCTUnwrap(main["transformer_blocks.\(layer).\(norm).weight"])
        let comfy = try XCTUnwrap(main["diffusion_model.blocks.\(layer).\(norm).weight"])
        XCTAssertEqual(Array(diffusers), Array(comfy))
      }
      for parameter in ["weight", "bias"] {
        let diffusers = try XCTUnwrap(
          fixed["transformer_blocks.\(layer).adaln_proj.linear.\(parameter)"])
        let comfy = try XCTUnwrap(
          fixed["diffusion_model.blocks.\(layer).adaln_proj.linear.\(parameter)"])
        XCTAssertEqual(diffusers.count, 18)
        XCTAssertEqual(Array(diffusers), Array(comfy))
      }
    }
    for layer in 0..<2 {
      let q = fixed["token_refiner.refiner_blocks.\(layer).attn.to_q.weight"]!
      let fused = fixed["diffusion_model.token_refiner.blocks.\(layer).attn.qkv_proj.weight"]!
      XCTAssertEqual(q.first, fused.first)
      XCTAssertTrue(q.interleavedIndices.isEmpty)
      XCTAssertTrue(fused.interleavedIndices.isEmpty)
      for norm in ["norm1", "norm2"] {
        let diffusers = try XCTUnwrap(
          fixed["token_refiner.refiner_blocks.\(layer).\(norm).weight"])
        let comfy = try XCTUnwrap(
          fixed["diffusion_model.token_refiner.blocks.\(layer).\(norm).weight"])
        XCTAssertEqual(Array(diffusers), Array(comfy))
      }
    }
  }
}
