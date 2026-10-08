import Foundation
import NNC
import WeightsCache
import XCTest

@testable import Diffusion

final class TeaCacheTests: XCTestCase {
  func testModelBuilderBooleanRebuildPreservesWeights() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      var builds = [Bool]()
      let model = ModelBuilderOrModel.modelBuilder(
        ModelBuilder { parameter, _ in
          builds.append(parameter)
          let input = Input()
          let projected = Dense(count: 2, name: "projection")(input)
          return Model([input], [(parameter ? 3 : 2) * projected])
        })
      let input = graph.variable(Tensor<Float>([1, 2], .CPU, .WC(1, 2)))
      model.compile(false, inputs: [input])
      let identity = model.unwrapped as AnyObject
      let baseline = model(inputs: input)[0].as(of: Float.self).rawValue
      let expected = [baseline[0, 0], baseline[0, 1]]
      XCTAssertEqual(builds, [false])
      XCTAssertGreaterThan(abs(expected[0]) + abs(expected[1]), 0)
      for (parameter, buildCount) in [(true, 2), (true, 2), (false, 3), (false, 3)] {
        let result = model(parameter, inputs: input)[0].as(of: Float.self).rawValue
        XCTAssertTrue(identity === model.unwrapped as AnyObject)
        XCTAssertEqual(builds.count, buildCount)
        for i in 0..<2 {
          XCTAssertEqual(result[0, i], expected[i] * (parameter ? 1.5 : 1), accuracy: 1e-6)
        }
      }
      // Explicit compilation and shape-driven rebuilds must carry the same Boolean too.
      model.compile(true, inputs: [input])
      _ = model(true, inputs: input)
      XCTAssertEqual(builds, [false, true, false, true])
      let longer = graph.variable(Tensor<Float>([1, 2, 1, 2], .CPU, .WC(2, 2)))
      let result = model(true, inputs: longer)[0].as(of: Float.self).rawValue
      XCTAssertEqual(builds, [false, true, false, true, true])
      for row in 0..<2 {
        for i in 0..<2 {
          XCTAssertEqual(result[row, i], expected[i] * 1.5, accuracy: 1e-6)
        }
      }
    }
  }

  func testLoadModelsSkipsSharedWeights() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let graph = DynamicGraph()
    let input = Input()
    let model = Model([input], [input])
    graph.openStore(directory.appendingPathComponent("weights.ckpt").path) { store in
      for version: ModelVersion in [.flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1] {
        let cache = TeaCache<Float16>(
          modelVersion: version, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
          steps: 1...8, maxSkipSteps: 3, reducedModel: .model(model), inferModel: .model(model))
        cache.loadModels(from: store) { _, _ in
          XCTFail("\(version) should share its auxiliary weights from the main model")
        }
      }
      let cache = TeaCache<Float16>(
        modelVersion: .minimaxH3, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
        steps: 1...8, maxSkipSteps: 3, reducedModel: .model(model))
      cache.loadModels(from: store) { _, _ in
        XCTFail("No independent model to load")
      }
    }
  }

  func testMiniMaxH3FixedReferenceParameterCoverage() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      var namesByReferenceCount = [Set<String>]()
      for referenceCount in 0...2 {
        let model = MiniMaxH3Fixed(
          timesteps: 1, hiddenSize: 8, layers: 1, textLength: (0, 2),
          usesFlashAttention: .scale1, referenceImageCount: referenceCount
        ).1
        let inputs: [DynamicGraph.AnyTensor] =
          [
            graph.variable(.GPU(0), .HWC(1, 2, 5_120), of: Float16.self),
            graph.variable(
              .GPU(0), .HWC(1, referenceCount > 0 ? 3 : 2, 256), of: Float16.self),
          ]
          + (0..<referenceCount).map { _ in
            graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self)
          }
        model.compile(inputs: inputs)
        namesByReferenceCount.append(
          Set((0..<model.parameters.count).map { model.parameters(for: .index($0)).name }))
      }
      // A superset is safe, but the no-reference cache cannot supply the image projection.
      XCTAssertTrue(namesByReferenceCount[0].isSubset(of: namesByReferenceCount[1]))
      XCTAssertEqual(
        namesByReferenceCount[1].subtracting(namesByReferenceCount[0]),
        Set(["t-proj_in-0-0", "t-proj_in-0-1"]))
      // Additional images reuse that projection, so the cache key needs only presence, not count.
      XCTAssertEqual(namesByReferenceCount[1], namesByReferenceCount[2])
    }
  }

  func testMiniMaxH3SplitMergedAndSeparateLoRALoading() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let graph = DynamicGraph()
    graph.withNoGrad {
      func build(
        layers: Int, startLayer: Int = 0, outputResidual: Bool = false,
        configuration: LoRANetworkConfiguration? = nil
      ) -> Model {
        if let configuration {
          return LoRAMiniMaxH3(
            hiddenSize: 8, layers: layers, startLayer: startLayer, textLength: 3, audioLength: 4,
            videoFrames: 1, videoHeight: 2, videoWidth: 2, usesFlashAttention: .scale1,
            outputResidual: outputResidual, LoRAConfiguration: configuration
          ).1
        }
        return MiniMaxH3(
          hiddenSize: 8, layers: layers, startLayer: startLayer, textLength: 3, audioLength: 4,
          videoFrames: 1, videoHeight: 2, videoWidth: 2, usesFlashAttention: .scale1,
          outputResidual: outputResidual
        ).1
      }
      let inputs: [DynamicGraph.AnyTensor] =
        [
          graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self),
          graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self),
          graph.variable(.GPU(0), .HWC(1, 3, 8), of: Float.self),
          graph.variable(.GPU(0), .NHWC(1, 8, 1, 128), of: Float16.self),
        ]
        + (0..<(5 * 18 + 4)).map { _ in
          graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self)
        }
      for (i, input) in inputs.enumerated() {
        if i == 2 {
          input.as(of: Float.self).full(0.1)
        } else {
          input.as(of: Float16.self).full(0.1)
        }
      }
      let base = build(layers: 5)
      let original = base(inputs: inputs[0], Array(inputs.dropFirst()))
      let basePath = directory.appendingPathComponent("base.ckpt").path
      let loraPath = directory.appendingPathComponent("lora.ckpt").path
      graph.openStore(basePath) { $0.write("dit", model: base) }
      graph.openStore(basePath, flags: .readOnly) { store in
        graph.openStore(loraPath) { lora in
          for (index, key) in store.keys.sorted().enumerated() where key.hasSuffix("-0]") {
            guard let weight = store.read(like: key), weight.shape.count >= 2 else { continue }
            // Keep selected blocks/projections unadapted to test global-index key filtering too.
            if key.contains("-q-2-") || key.contains("-v-3-") || key.contains("-gate-4-") {
              continue
            }
            let inputSize = weight.shape.dropFirst().reduce(1, *)
            let outputSize = weight.shape[0]
            // Distinct updates make a wrong block's LoRA observable, not just its base weights.
            let value = Float16(0.005 + Double(index % 7) * 0.001)
            lora.write(
              key + "__down__",
              tensor: Tensor<Float16>(
                Array(repeating: value, count: 2 * inputSize), .CPU, .NC(2, inputSize)))
            lora.write(
              key + "__up__",
              tensor: Tensor<Float16>(
                Array(repeating: value, count: 2 * outputSize), .CPU, .NC(outputSize, 2)))
          }
        }
      }
      let keys = LoRALoader.keys(graph, of: [loraPath], modelFile: basePath)
      XCTAssertFalse(keys.isEmpty)
      for separate in [false, true] {
        let configuration =
          separate
          ? LoRANetworkConfiguration(rank: 2, scale: 1, highPrecision: false, keys: keys) : nil
        let baseline = build(layers: 5, configuration: configuration)
        let infer = build(layers: 1, outputResidual: true, configuration: configuration)
        let tail = build(layers: 4, startLayer: 1, configuration: configuration)
        let firstInputs = Array(inputs.prefix(4 + 18))
        let placeholder = graph.variable(.GPU(0), .HWC(1, 8, 8), of: Float.self)
        let tailInputs: [DynamicGraph.AnyTensor] =
          [placeholder, inputs[3]]
          + inputs.dropFirst(firstInputs.count)
        baseline.compile(inputs: inputs)
        infer.compile(inputs: firstInputs)
        tail.compile(inputs: tailInputs)
        let baselineNames = Set(
          (0..<baseline.parameters.count).map { baseline.parameters(for: .index($0)).name })
        let tailNames = Set(
          (0..<tail.parameters.count).map { tail.parameters(for: .index($0)).name })
        XCTAssertTrue(tailNames.isSubset(of: baselineNames))
        XCTAssertTrue(tailNames.contains("t-q-1-0"))
        XCTAssertFalse(tailNames.contains("t-q-0-0"))
        if separate {
          XCTAssertTrue(tailNames.contains("t-q_lora_down-1-0-0"))
        }
        graph.openStore(basePath, flags: .readOnly) { store in
          LoRALoader.openStore(
            graph,
            lora: [
              LoRAConfiguration(
                file: loraPath, weight: 1, version: .minimaxH3, isLoHa: false,
                modifier: .none, mode: .all)
            ]
          ) { loader in
            for model in [baseline, infer, tail] {
              do {
                try store.read("dit", model: model, strict: true) { name, dataType, format, shape in
                  if separate {
                    return loader.concatenateLoRA(
                      graph,
                      LoRAMapping: Dictionary(uniqueKeysWithValues: (0..<5).map { ($0, $0) }),
                      filesRequireMerge: [:], name: name, store: store, dataType: dataType,
                      format: format, shape: shape, of: Float16.self)
                  }
                  return loader.mergeLoRA(
                    graph, name: name, store: store, dataType: dataType, shape: shape,
                    of: Float16.self)
                }
              } catch { XCTFail("Strict split LoRA loading failed: \(error)") }
            }
          }
        }
        let expected = baseline(inputs: inputs[0], Array(inputs.dropFirst()))
        let first = infer(inputs: firstInputs[0], Array(firstInputs.dropFirst()))
        let actual = tail(inputs: first[0], Array(tailInputs.dropFirst()))
        for i in 0..<2 {
          let expected = DynamicGraph.Tensor<Float>(from: expected[i])
          let actual = DynamicGraph.Tensor<Float>(from: actual[i])
          let original = DynamicGraph.Tensor<Float>(from: original[i])
          let axes = Array(0..<expected.shape.count)
          let error = Functional.abs(expected - actual).reduced(.max, axis: axes)
            .rawValue.toCPU().reshaped(.C(1))[0]
          let change = Functional.abs(expected - original).reduced(.max, axis: axes)
            .rawValue.toCPU().reshaped(.C(1))[0]
          XCTAssertTrue(error.isFinite)
          XCTAssertLessThan(error, 0.002)
          XCTAssertGreaterThan(change, 0.00001)
        }
      }
    }
  }

  func testFirstBlockDecisionWaitsForAsyncStatistics() {
    let graph = DynamicGraph()
    let stream = StreamContext(.GPU(0))
    graph.withStream(stream) {
      graph.withNoGrad {
        defer { graph.joined() }
        let input = Input()
        let model = Model([input], [input])
        let cache = TeaCache<Float16>(
          modelVersion: .minimaxH3, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
          steps: 1...8, maxSkipSteps: 3, reducedModel: .model(model))
        let previous = graph.variable(.GPU(0), .HWC(1, 200, 8), of: Float.self)
        previous.full(1)
        XCTAssertFalse(
          cache.shouldUseCacheForTimeEmbedding(
            [previous], model: .model(model), step: 0, marker: 0, of: Float.self))
        let changed = graph.variable(.GPU(0), .HWC(1, 200, 8), of: Float.self)
        changed.full(1.2)
        XCTAssertFalse(
          cache.shouldUseCacheForTimeEmbedding(
            [changed], model: .model(model), step: 1, marker: 0, of: Float.self))
        XCTAssertTrue(
          cache.shouldUseCacheForTimeEmbedding(
            [changed], model: .model(model), step: 2, marker: 0, of: Float.self))
      }
    }
  }

  func testExistingTeaCacheAccumulationAndStaticModelSharing() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let x = Input()
      let shift = Input()
      let scale = Input()
      let projection = Dense(count: 8, name: "projection")
      let model = Model([x, shift, scale], [projection(x) .* scale + shift, x - x])
      let residual = Input()
      let reducedProjection = Dense(count: 8, name: "projection")
      let reduced = Model(
        [x, residual, shift, scale], [reducedProjection(x + residual) .* scale + shift])
      let cache = TeaCache<Float16>(
        modelVersion: .wan21_1_3b, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
        steps: 1...8, maxSkipSteps: 3, reducedModel: .model(reduced))
      let input = graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self)
      input.full(1)
      let inputs: [DynamicGraph.AnyTensor] = [input, input, input]
      let full = model(inputs: input, [input, input])
      cache.compile(model: .model(model), inputs: inputs)
      cache.cache(outputs: full, marker: 0)
      func hit(_ value: Float, step: Int) -> Bool {
        let t = graph.variable(.GPU(0), .C(1), of: Float16.self)
        t.full(value)
        return cache.shouldUseCacheForTimeEmbedding(
          [t], model: .model(model), step: step, marker: 0, of: Float16.self)
      }
      XCTAssertFalse(hit(1, step: 0))
      XCTAssertTrue(hit(1.03, step: 1))
      let cached = cache(model: .model(model), inputs: input, [input, input], marker: 0)!
      XCTAssertEqual(cached.count, 1)
      let expected = full[0].as(of: Float16.self).toCPU().rawValue
      let actual = cached[0].as(of: Float16.self).toCPU().rawValue
      for i in 0..<8 { XCTAssertEqual(actual[0, 0, i], expected[0, 0, i]) }
      XCTAssertTrue(hit(1.06, step: 2))
      XCTAssertFalse(hit(1.09, step: 3))  // Accumulated distance, not just this step's distance.
    }
  }

  func testCachedHeadInputOrderAndPrecision() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for (version, references): (ModelVersion, Int) in [
        (.hunyuanVideo, 0), (.wan21_1_3b, 0), (.wan21_14b, 0), (.hiDreamI1, 0),
        (.flux1, 0), (.flux1, 1), (.minimaxH3, 0),
      ] {
        func tensor(_ value: Float) -> DynamicGraph.Tensor<Float> {
          let tensor = graph.variable(.GPU(0), .C(1), of: Float.self)
          tensor.full(value)
          return tensor
        }
        let x = Input()
        let residual = Input()
        let shift = Input()
        let scale = Input()
        let model: Model
        var restInputs: [DynamicGraph.AnyTensor]
        if version == .minimaxH3 {
          let rotary = Input()
          let audioShift = Input()
          let audioScale = Input()
          model = Model(
            [x, rotary, shift, scale, audioShift, audioScale, residual],
            [
              (x + residual) .* scale + shift + rotary,
              (x + residual) .* audioScale + audioShift,
            ])
          restInputs = [tensor(0), tensor(1), tensor(0.5), tensor(2), tensor(0.25)]
        } else {
          model = Model([x, residual, shift, scale], [(x + residual) .* scale + shift])
          if version == .flux1 {
            let shiftIndex = (references > 0 ? 3 : 2) + 19 * 12 + 38 * 3
            restInputs = Array(repeating: tensor(-10), count: shiftIndex + 4)
            restInputs[shiftIndex] = tensor(1)
            restInputs[shiftIndex + 1] = tensor(0.5)
          } else {
            restInputs = [tensor(-10), tensor(1), tensor(0.5)]
          }
        }
        let cache = TeaCache<Float16>(
          modelVersion: version, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
          steps: 1...8, maxSkipSteps: 3, reducedModel: .model(model),
          referenceImageCount: references)
        let input = tensor(70_000)  // FP32 input must not be narrowed to the cache's Float16 type.
        if version == .minimaxH3 {
          model.compile(inputs: [input] + restInputs + [tensor(4)])
        } else {
          model.compile(inputs: [input, tensor(4), tensor(1), tensor(0.5)])
        }
        XCTAssertNil(cache(model: .model(model), inputs: input, restInputs, marker: 0))
        cache.cache(outputs: [tensor(4)], marker: 0)
        let result = cache(model: .model(model), inputs: input, restInputs, marker: 0)!
        XCTAssertEqual(result.count, version == .minimaxH3 ? 2 : 1)
        XCTAssertEqual(result[0].as(of: Float.self).rawValue.toCPU()[0], 35_003)
        if version == .minimaxH3 {
          XCTAssertEqual(result[1].as(of: Float.self).rawValue.toCPU()[0], 17_503)
        }
      }
    }
  }

  func testExistingInferModelInputsAndParameterSharing() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for (version, references, shiftIndex): (ModelVersion, Int, Int) in [
        (.hunyuanVideo, 0, 10), (.flux1, 0, 9), (.flux1, 1, 10),
      ] {
        func build() -> Model {
          let x = Input()
          let shift = Input()
          let scale = Input()
          return Model(
            [x, shift, scale], [Dense(count: 8, name: "projection")(x) .* scale + shift])
        }
        let model = build()
        let inferModel = build()
        let inputs = (0..<12).map { _ in
          let tensor = graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self)
          tensor.full(0.1)
          return tensor
        }
        let expected = model(inputs: inputs[0], [inputs[shiftIndex], inputs[shiftIndex + 1]])[0]
        inferModel.compile(inputs: [inputs[0], inputs[shiftIndex], inputs[shiftIndex + 1]])
        let residual = Input()
        let cache = TeaCache<Float16>(
          modelVersion: version, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
          steps: 1...8, maxSkipSteps: 3, reducedModel: .model(Model([residual], [residual])),
          inferModel: .model(inferModel), referenceImageCount: references)
        func hit(_ step: Int) -> Bool {
          cache.shouldUseCacheForTimeEmbedding(
            inputs, model: .model(model), step: step, marker: 0, of: Float16.self)
        }
        XCTAssertFalse(hit(0))
        let actual = inferModel(inputs: inputs[0], [inputs[shiftIndex], inputs[shiftIndex + 1]])[0]
        let error = Functional.abs(actual.as(of: Float16.self) - expected.as(of: Float16.self))
          .reduced(.max, axis: [0, 1, 2]).rawValue.toCPU().reshaped(.C(1))[0]
        XCTAssertEqual(error, 0)
        // Unselected conditioning must not change the signal or trigger a full step.
        for i in 1..<inputs.count where i != shiftIndex && i != shiftIndex + 1 {
          inputs[i].full(4)
        }
        XCTAssertTrue(hit(1))
        inputs[shiftIndex].full(10)
        XCTAssertFalse(hit(2))
      }
    }
  }

  func testFirstBlockDecisionUsesBothModalitiesAndLastFullStep() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let input = Input()
      let model = Model([input], [input])
      let cache = TeaCache<Float16>(
        modelVersion: .minimaxH3, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
        steps: 1...8, maxSkipSteps: 3, reducedModel: .model(model))
      func signals(_ audio: Float, _ video: Float, audioRows: Int = 2) -> [DynamicGraph.AnyTensor] {
        let a = graph.variable(.GPU(0), .HWC(1, audioRows, 8), of: Float.self)
        let v = graph.variable(.GPU(0), .HWC(1, 200, 8), of: Float.self)
        a.full(audio)
        v.full(video)
        return [a, v]
      }
      func hit(_ audio: Float, _ video: Float, step: Int, marker: Int = 0) -> Bool {
        cache.shouldUseCacheForTimeEmbedding(
          signals(audio, video), model: .model(model), step: step, marker: marker, of: Float.self)
      }
      XCTAssertFalse(hit(1, 1, step: 0))
      XCTAssertTrue(hit(1.04, 1, step: 1))
      // A 100x larger unchanged video stream must not hide the 8% audio change.
      // This also verifies comparison to the full step, not the previous 4% cache hit.
      XCTAssertFalse(hit(1.08, 1, step: 2))
      XCTAssertTrue(hit(1.08, 1.04, step: 3))
      XCTAssertFalse(hit(1.08, 1.08, step: 4))
      XCTAssertFalse(hit(1.08, 1.08, step: 5, marker: 1))
      XCTAssertTrue(hit(1.08, 1.08, step: 4))  // Repeated indices do not invalidate matching signals.
      XCTAssertTrue(hit(1.08, 1.08, step: 3))  // Nor do decreasing indices within the cache window.
      XCTAssertFalse(hit(1.08, 1.2, step: 3))  // The current signal still controls the decision.
      XCTAssertFalse(hit(.nan, 1, step: 5))
      XCTAssertFalse(hit(1, 1, step: 6))  // A non-finite anchor also forces refresh.
      XCTAssertTrue(hit(1, 1, step: 7))
      XCTAssertFalse(
        cache.shouldUseCacheForTimeEmbedding(
          signals(1, 1, audioRows: 4), model: .model(model), step: 8, marker: 0, of: Float.self))
      XCTAssertFalse(hit(1, 1, step: 9))  // Outside configured window.
      XCTAssertFalse(
        cache.shouldUseCacheForTimeEmbedding(
          Array(signals(1, 1).prefix(1)), model: .model(model), step: 10, marker: 0, of: Float.self)
      )
    }
  }

  func testMiniMaxH3SplitLoadingAndCachedHead() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let graph = DynamicGraph()
    try graph.withNoGrad {
      for useLoRA in [false, true] {
        for (references, audioReferences) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
          let visionLength = references
          let perLayer = 18 + (references > 0 ? 6 : 0) + audioReferences * 6
          let layers = 50
          func build(layers: Int, startLayer: Int = 0, outputResidual: Bool, inputResidual: Bool)
            -> ModelBuilderOrModel
          {
            .modelBuilder(
              ModelBuilder { _, inputs in
                let textLength =
                  startLayer > 0
                  ? inputs[0].shape[1] - references - audioReferences * 6 - 5 : inputs[2].shape[1]
                if useLoRA {
                  return LoRAMiniMaxH3(
                    hiddenSize: 8, layers: layers, startLayer: startLayer, textLength: textLength,
                    audioLength: 4,
                    videoFrames: 1, videoHeight: 2, videoWidth: 2, usesFlashAttention: .scale1,
                    referenceImageSizes: references > 0 ? [(2, 2)] : [],
                    referenceAudioLengths: audioReferences > 0 ? [6] : [],
                    visionLength: visionLength,
                    outputResidual: outputResidual, inputResidual: inputResidual,
                    LoRAConfiguration: LoRANetworkConfiguration(
                      rank: 2, scale: 0.1, highPrecision: false)
                  ).1
                }
                return MiniMaxH3(
                  hiddenSize: 8, layers: layers, startLayer: startLayer, textLength: textLength,
                  audioLength: 4,
                  videoFrames: 1, videoHeight: 2, videoWidth: 2, usesFlashAttention: .scale1,
                  referenceImageSizes: references > 0 ? [(2, 2)] : [],
                  referenceAudioLengths: audioReferences > 0 ? [6] : [], visionLength: visionLength,
                  outputResidual: outputResidual, inputResidual: inputResidual
                ).1
              })
          }
          let baseline = build(layers: layers, outputResidual: false, inputResidual: false)
          let model = build(
            layers: layers - 1, startLayer: 1, outputResidual: true, inputResidual: false)
          let reduced = build(
            layers: 0, startLayer: 1, outputResidual: false, inputResidual: true)
          let inferModel = build(layers: 1, outputResidual: true, inputResidual: false)
          let cache = TeaCache<Float16>(
            modelVersion: .minimaxH3, coefficients: (0, 0, 0, 1, 0), threshold: 0.06,
            steps: 1...10, maxSkipSteps: 2,
            reducedModel: reduced,
            inferModel: inferModel,
            referenceImageCount: references, referenceAudioCount: audioReferences)
          func inputs(textLength: Int) -> [DynamicGraph.AnyTensor] {
            let tensors: [DynamicGraph.AnyTensor] =
              [
                graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self),
                graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self),
                graph.variable(.GPU(0), .HWC(1, textLength, 8), of: Float.self),
                graph.variable(
                  .GPU(0), .NHWC(1, textLength + references + audioReferences * 6 + 5, 1, 128),
                  of: Float16.self),
              ]
              + (0..<references).map { _ in
                graph.variable(.GPU(0), .NHWC(1, 1, 1, 8), of: Float16.self)
              }
              + (0..<audioReferences).map { _ in
                graph.variable(.GPU(0), .HWC(1, 6, 8), of: Float16.self)
              }
              + (0..<(layers * perLayer + 4)).map { _ in
                graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self)
              }
            for (i, tensor) in tensors.enumerated() {
              if i == 2 {
                tensor.as(of: Float.self).full(0.01)
              } else {
                tensor.as(of: Float16.self).full(0.01)
              }
            }
            return tensors
          }
          let longest = inputs(textLength: 5)
          _ = baseline(inputs: longest[0], Array(longest.dropFirst()))
          cache.compile(model: model, inputs: longest)
          let path = directory.appendingPathComponent("\(useLoRA)-\(references).ckpt").path
          graph.openStore(path) { $0.write("dit", model: baseline.unwrapped) }
          graph.openStore(path, flags: .readOnly) { store in
            do {
              try store.read("dit", model: model.unwrapped, strict: true)
            } catch { XCTFail("Strict tail loading failed: \(error)") }
            var loadedModels = 0
            cache.loadModels(from: store) { part, store in
              loadedModels += 1
              do {
                try store.read("dit", model: part.unwrapped, strict: true)
              } catch { XCTFail("Strict independent-model loading failed: \(error)") }
            }
            XCTAssertEqual(loadedModels, 1)
          }
          for (marker, length) in [5, 3].enumerated() {
            let context =
              "LoRA=\(useLoRA), references=\(references), audioReferences=\(audioReferences), textLength=\(length)"
            let inputs = inputs(textLength: length)
            let audioStart = length - visionLength + references + audioReferences * 6
            let videoStart = length + references + audioReferences * 6 + 4
            func run(_ step: Int) -> [DynamicGraph.AnyTensor] {
              let firstInputs = Array(inputs.prefix(4 + references + audioReferences + perLayer))
              let firstOutput = cache.infer(inputs: firstInputs)
              let hiddenState = firstOutput[0]
              let firstBlock = firstOutput[1].as(of: Float.self)
              let channels = firstBlock.shape[2]
              let signals: [DynamicGraph.AnyTensor] = [
                firstBlock[0..<1, audioStart..<(audioStart + 4), 0..<channels].copied(),
                firstBlock[0..<1, videoStart..<(videoStart + 1), 0..<channels].copied(),
              ]
              let tailInputs = [inputs[3]] + inputs.dropFirst(firstInputs.count)
              if cache.shouldUseCacheForTimeEmbedding(
                signals, model: model, step: step, marker: marker, of: Float.self),
                let cached = cache(model: model, inputs: hiddenState, tailInputs, marker: marker)
              {
                return cached
              }
              let result = model(inputs: hiddenState, tailInputs)
              cache.cache(outputs: result, marker: marker)
              return result
            }
            let full = run(0)
            XCTAssertEqual(full.count, 3)
            let expectedFull = baseline(inputs: inputs[0], Array(inputs.dropFirst()))
            for i in 0..<2 {
              let expected = DynamicGraph.Tensor<Float>(from: expectedFull[i])
              let actual = DynamicGraph.Tensor<Float>(from: full[i])
              let error = Functional.abs(expected - actual).reduced(
                .max, axis: Array(0..<expected.shape.count)
              )
              .rawValue.toCPU().reshaped(.C(1))[0]
              XCTAssertTrue(error.isFinite)
              XCTAssertLessThan(error, 0.002, "Full/split output \(i): \(context)")
            }
            let cached = run(2)
            XCTAssertEqual(cached.count, 2)
            for i in 0..<2 {
              let expected = DynamicGraph.Tensor<Float>(from: full[i])
              let actual = DynamicGraph.Tensor<Float>(from: cached[i])
              let error = Functional.abs(expected - actual).reduced(
                .max, axis: Array(0..<expected.shape.count)
              ).toCPU().rawValue.reshaped(.C(1))[0]
              XCTAssertTrue(error.isFinite)
              XCTAssertLessThan(error, 0.002)
            }
            // A second evaluation at the same step must use today's modulation, not cached predictions.
            inputs[inputs.count - 4].as(of: Float16.self).full(0.04)
            inputs[inputs.count - 2].as(of: Float16.self).full(-0.02)
            let fresh = baseline(inputs: inputs[0], Array(inputs.dropFirst()))
            let changedHead = run(2)
            XCTAssertEqual(changedHead.count, 2)
            for i in 0..<2 {
              let expected = DynamicGraph.Tensor<Float>(from: fresh[i])
              let actual = DynamicGraph.Tensor<Float>(from: changedHead[i])
              let previous = DynamicGraph.Tensor<Float>(from: full[i])
              let change = Functional.abs(expected - previous).reduced(
                .max, axis: Array(0..<expected.shape.count)
              ).toCPU().rawValue.reshaped(.C(1))[0]
              XCTAssertGreaterThan(change, 0.0001)
              let error = Functional.abs(expected - actual).reduced(
                .max, axis: Array(0..<expected.shape.count)
              ).toCPU().rawValue.reshaped(.C(1))[0]
              XCTAssertTrue(error.isFinite)
              XCTAssertLessThan(error, 0.002, "Changed-head output \(i): \(context)")
            }
            let refreshed = run(2)
            XCTAssertEqual(refreshed.count, 3)  // Enforce maxSkipSteps = 2 even at the same index.
            // Even on a hit, consume this step's block-0 output, not the previous hidden state.
            let firstInputs = Array(inputs.prefix(4 + references + audioReferences + perLayer))
            let previousFirst = inferModel(
              inputs: firstInputs[0], Array(firstInputs.dropFirst()))[0].as(of: Float.self)
            inputs[0].as(of: Float16.self).full(0.0101)
            let currentFirst = inferModel(
              inputs: firstInputs[0], Array(firstInputs.dropFirst()))[0].as(of: Float.self)
            let change = Functional.abs(currentFirst - previousFirst).reduced(
              .max, axis: [0, 1, 2]
            )
            .rawValue.toCPU().reshaped(.C(1))[0]
            XCTAssertGreaterThan(change, 1e-8)
            let expectedHit = reduced(
              inputs: currentFirst, [inputs[3]] + Array(inputs.suffix(4)) + [refreshed[2]])
            let actualHit = run(1)  // A lower index still uses the current block-0 output.
            XCTAssertEqual(actualHit.count, 2)
            for i in 0..<2 {
              let expected = DynamicGraph.Tensor<Float>(from: expectedHit[i])
              let actual = DynamicGraph.Tensor<Float>(from: actualHit[i])
              let error = Functional.abs(expected - actual).reduced(
                .max, axis: Array(0..<expected.shape.count)
              ).rawValue.toCPU().reshaped(.C(1))[0]
              XCTAssertEqual(error, 0)
            }
          }
          if !useLoRA {
            // Exercise the app's distinct full/tail cache entries and strict reads after reattaching.
            let weights = WeightsCache(maxTotalCacheSize: 1 << 30, memorySubsystem: .UMA)
            let expected = baseline(inputs: longest[0], Array(longest.dropFirst()))
            let suffix = TeaCacheConfiguration(
              coefficients: (0, 0, 0, 1, 0), steps: 1...10, threshold: 0.06, maxSkipSteps: 2
            ).suffix(for: .minimaxH3)
            weights.attach(path + suffix, from: model.unwrapped.parameters)
            XCTAssertFalse(weights.detach(path, to: baseline.unwrapped.parameters))
            weights.attach(path, from: baseline.unwrapped.parameters)
            let full = build(layers: layers, outputResidual: false, inputResidual: false)
            let tail = build(
              layers: layers - 1, startLayer: 1, outputResidual: true, inputResidual: false)
            full.compile(inputs: longest)
            let firstInputs = Array(longest.prefix(4 + references + audioReferences + perLayer))
            let first = inferModel(inputs: firstInputs[0], Array(firstInputs.dropFirst()))
            let tailInputs = [first[0], longest[3]] + longest.dropFirst(firstInputs.count)
            tail.compile(inputs: tailInputs)
            XCTAssertTrue(weights.detach(path + suffix, to: tail.unwrapped.parameters))
            XCTAssertTrue(weights.detach(path, to: full.unwrapped.parameters))
            XCTAssertEqual(weights.count, 0)
            try graph.openStore(path, flags: .readOnly) { store in
              try store.read("dit", model: tail.unwrapped, strict: true) { _, _, _, _ in .fail }
              try store.read("dit", model: full.unwrapped, strict: true) { _, _, _, _ in .fail }
            }
            let reloadedFull = full(inputs: longest[0], Array(longest.dropFirst()))
            let reloadedTail = tail(inputs: tailInputs[0], Array(tailInputs.dropFirst()))
            for result in [reloadedFull, reloadedTail] {
              for i in 0..<2 {
                let original = DynamicGraph.Tensor<Float>(from: expected[i])
                let actual = DynamicGraph.Tensor<Float>(from: result[i])
                let error = Functional.abs(original - actual).reduced(
                  .max, axis: Array(0..<original.shape.count)
                ).rawValue.toCPU().reshaped(.C(1))[0]
                XCTAssertTrue(error.isFinite)
                XCTAssertLessThan(error, 0.002)
              }
            }
          }
        }
      }
    }
  }
}
