import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC
import XCTest

final class QwenImage2_1LoRATests: XCTestCase {
  func testImportSharedProjectionsAndPackedModulation() throws {
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
    // Include shared input / timestep projections as well as the last block, whose fixed
    // graph deliberately contains only K/V. Alpha differs from rank to check scaling.
    let projections: [(String, Int, Int)] = [
      ("img_in", 64, 4096), ("txt_in.in_layer", 4096, 4096),
      ("txt_in.out_layer", 4096, 4096),
      ("time_text_embed.timestep_embedder.linear_1", 256, 4096),
      ("time_text_embed.timestep_embedder.linear_2", 4096, 4096),
      ("modulation.1", 4096, 16384), ("norm_out.linear", 4096, 4096),
      ("proj_out", 4096, 64),
      ("transformer_blocks.31.attn.to_q", 4096, 4096),
      ("transformer_blocks.31.attn.to_k", 4096, 4096),
      ("transformer_blocks.31.img_mlp.gate_layer", 4096, 12288),
    ]
    var header = [String: Any]()
    var data = Data()
    func append(_ key: String, shape: [Int], values: [Float]) {
      let start = data.count
      values.withUnsafeBytes { data.append(contentsOf: $0) }
      header[key] = ["dtype": "F32", "shape": shape, "data_offsets": [start, data.count]]
    }
    for (key, input, output) in projections {
      append(
        "\(key).lora_A.default.weight", shape: [2, input],
        values: (0..<(2 * input)).map { Float($0 / input + 1) / 8 })
      append(
        "\(key).lora_B.default.weight", shape: [output, 2],
        values: (0..<(output * 2)).map { Float($0 / (4096 * 2) + 1) / 16 })
      append("\(key).alpha", shape: [], values: [1])
    }
    var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    while json.count % 8 != 0 { json.append(0x20) }
    var length = UInt64(json.count).littleEndian
    var file = withUnsafeBytes(of: &length) { Data($0) }
    file.append(json)
    file.append(data)
    let source = directory.appendingPathComponent("adapter.safetensors")
    try file.write(to: source)
    let (version, _, _, _) = try LoRAImporter.import(
      downloadedFile: source.path, name: "Qwen 2.1 test", filename: "adapter.ckpt",
      scaleFactor: 1, forceVersion: nil, progress: { _ in })
    XCTAssertEqual(version, .qwenImage2_1)
    let (fixed, main, _, _) = LoRAImporter.modelWeightsMapping(
      by: version, qkNorm: false, dualAttentionLayers: [], format: [.diffusers])
    for (key, _, _) in projections {
      if let fixedNames = fixed[key + ".weight"], let mainNames = main[key + ".weight"] {
        XCTAssertEqual(Array(fixedNames), Array(mainNames), key)
      }
    }
    let graph = DynamicGraph()
    graph.openStore(directory.appendingPathComponent("adapter.ckpt").path, flags: .readOnly) {
      store in
      XCTAssertEqual(store.keys.filter { $0.hasSuffix("__up__") }.count, projections.count + 3)
      for (key, input, output) in projections {
        let names = (main[key + ".weight"] ?? fixed[key + ".weight"])!
        for (part, name) in names.enumerated() {
          let down = Tensor<Float>(from: store.read("__dit__[\(name)]__down__")!)
          let up = Tensor<Float>(from: store.read("__dit__[\(name)]__up__")!)
          XCTAssertEqual(Array(down.shape), [2, input])
          XCTAssertEqual(Array(up.shape), [output / names.count, 2])
          let expected = Float(part + 1) * 3 / 256
          XCTAssertEqual(
            up[0, 0] * down[0, 0] + up[0, 1] * down[1, 0], expected,
            accuracy: 1e-6, key)
        }
      }
    }
  }

  func testPublishedAdapterMergedAndSeparateInference() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let checkpoint = environment["QWEN_IMAGE_2_1_CHECKPOINT"],
      let source = environment["QWEN_IMAGE_2_1_LORA"]
    else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_CHECKPOINT and QWEN_IMAGE_2_1_LORA")
    }
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
    let (version, _, _, _) = try LoRAImporter.import(
      downloadedFile: source, name: "Qwen 2.1 published adapter", filename: "adapter.ckpt",
      scaleFactor: 1, forceVersion: nil, progress: { _ in })
    XCTAssertEqual(version, .qwenImage2_1)
    let adapter = directory.appendingPathComponent("adapter.ckpt").path
    let graph = DynamicGraph()
    let (rank, filesRequireMerge) = LoRALoader.rank(graph, of: [adapter], modelFile: checkpoint)
    XCTAssertGreaterThan(rank, 0)
    let keys = LoRALoader.keys(graph, of: [adapter], modelFile: checkpoint)
    let sourceFile = try FileHandle(forReadingFrom: URL(fileURLWithPath: source))
    defer { try? sourceFile.close() }
    let headerLength = sourceFile.readData(ofLength: 8).withUnsafeBytes {
      Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
    }
    let header = try XCTUnwrap(
      JSONSerialization.jsonObject(with: sourceFile.readData(ofLength: headerLength))
        as? [String: Any])
    XCTAssertEqual(keys.count, header.keys.filter { $0.hasSuffix(".lora_A.default.weight") }.count)
    let lora = [
      LoRAConfiguration(
        file: adapter, weight: 0.75, version: version,
        isLoHa: false, modifier: .none, mode: .all)
    ]
    let configuration = LoRANetworkConfiguration(
      rank: rank, scale: 1, highPrecision: false, keys: keys)
    let mapping = Dictionary(uniqueKeysWithValues: (0..<32).map { ($0, $0) })
    graph.withNoGrad {
      var context = Tensor<Float16>(.CPU, .HWC(1, 4, 4096))
      for row in 0..<4 {
        for col in 0..<4096 { context[0, row, col] = Float16(sin(Float(row * 4096 + col) * 0.013)) }
      }
      var reference = Tensor<Float16>(.CPU, .WC(4, 64))
      var target = Tensor<Float16>(.CPU, .NHWC(1, 2, 2, 64))
      for row in 0..<4 {
        for col in 0..<64 {
          reference[row, col] = Float16(sin(Float(row * 64 + col) * 0.03))
          target[0, row / 2, row % 2, col] = Float16(cos(Float(row * 64 + col) * 0.07))
        }
      }
      var time = Tensor<Float16>(.CPU, .WC(2, 256))
      for (row, t) in [Float(500), Float(0)].enumerated() {
        let embedding = timeEmbedding(
          timestep: t, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
        for col in 0..<256 { time[row, col] = Float16(embedding[0, col]) }
      }
      // Identity RoPE is sufficient for comparing the two LoRA execution paths.
      var rotary = Tensor<Float>(.CPU, .NHWC(1, 8, 1, 128))
      for row in 0..<8 {
        for col in 0..<128 { rotary[0, row, 0, col] = col % 2 == 0 ? 1 : 0 }
      }
      let fixedInputs: [DynamicGraph.AnyTensor] = [
        graph.variable(context.toGPU(0)),
        graph.variable(time.toGPU(0)), graph.variable(rotary.toGPU(0)),
        graph.variable(reference.toGPU(0)),
      ]
      let targetInput = graph.variable(target.toGPU(0))
      let targetRotary = graph.variable(rotary[0..<1, 0..<4, 0..<1, 0..<128].copied().toGPU(0))
      var results = [[Tensor<Float>]]()
      // Base first proves the adapter actually changes both cached prefix and final output.
      for mode in 0..<3 {
        let separate = mode == 2
        let fixed: Model
        if separate {
          (_, fixed) = LoRAQwenImage2_1Fixed(
            Float16.self, batchSize: 1, textLength: 4,
            referenceLength: 4, timesteps: 1, channels: 4096, layers: 32,
            segments: [(4, false), (4, true)], usesFlashAttention: .scaleMerged,
            LoRAConfiguration: configuration)
        } else {
          (_, fixed) = QwenImage2_1Fixed(
            Float16.self, batchSize: 1, textLength: 4,
            referenceLength: 4, timesteps: 1, channels: 4096, layers: 32,
            segments: [(4, false), (4, true)], usesFlashAttention: .scaleMerged)
        }
        let denoiser: Model
        if separate {
          (_, denoiser) = LoRAQwenImage2_1(
            Float16.self, batchSize: 1, height: 2, width: 2,
            prefixLength: 8, channels: 4096, layers: 32, usesFlashAttention: .scaleMerged,
            LoRAConfiguration: configuration)
        } else {
          (_, denoiser) = QwenImage2_1(
            Float16.self, batchSize: 1, height: 2, width: 2,
            prefixLength: 8, channels: 4096, layers: 32, usesFlashAttention: .scaleMerged)
        }
        func load(_ model: Model) {
          graph.openStore(
            checkpoint, flags: .readOnly,
            externalStore: TensorData.externalStore(filePath: checkpoint)
          ) { store in
            if mode == 0 {
              try! store.read("dit", model: model, strict: true, codec: [.externalData])
            } else {
              LoRALoader.openStore(graph, lora: lora) { loader in
                try! store.read("dit", model: model, strict: true, codec: [.externalData]) {
                  name, dataType, format, shape in
                  if separate {
                    return loader.concatenateLoRA(
                      graph, LoRAMapping: mapping,
                      filesRequireMerge: filesRequireMerge, name: name, store: store,
                      dataType: dataType, format: format, shape: shape, of: Float16.self)
                  }
                  return loader.mergeLoRA(
                    graph, name: name, store: store, dataType: dataType,
                    shape: shape, of: Float16.self)
                }
              }
            }
          }
        }
        fixed.compile(inputs: fixedInputs)
        load(fixed)
        let conditions = fixed(inputs: fixedInputs[0], Array(fixedInputs.dropFirst()))
        let inputs: [DynamicGraph.AnyTensor] = [targetInput, targetRotary] + conditions
        denoiser.compile(inputs: inputs)
        load(denoiser)
        let output = denoiser(inputs: inputs[0], Array(inputs.dropFirst()))[0]
        results.append([
          Tensor<Float>(from: conditions[5].as(of: Float16.self).rawValue.toCPU()),
          Tensor<Float>(from: conditions[67].as(of: Float16.self).rawValue.toCPU()),
          Tensor<Float>(from: output.as(of: Float16.self).rawValue.toCPU()),
        ])
      }
      for component in 0..<3 {
        let base = results[0][component].reshaped(.C(results[0][component].shape.reduce(1, *)))
        let merged = results[1][component].reshaped(.C(base.shape[0]))
        let separate = results[2][component].reshaped(.C(base.shape[0]))
        var error: Float = 0
        var magnitude: Float = 0
        var change: Float = 0
        for i in 0..<base.shape[0] {
          XCTAssertTrue(merged[i].isFinite && separate[i].isFinite)
          error += (merged[i] - separate[i]) * (merged[i] - separate[i])
          magnitude += merged[i] * merged[i]
          change += abs(base[i] - merged[i])
        }
        let relative = sqrt(error / max(magnitude, 1e-12))
        print(
          "Qwen 2.1 LoRA component \(component): relative RMS \(relative), base delta \(change)")
        XCTAssertLessThan(relative, 0.02)
        XCTAssertGreaterThan(change, 0.001)
      }
    }
  }
}
