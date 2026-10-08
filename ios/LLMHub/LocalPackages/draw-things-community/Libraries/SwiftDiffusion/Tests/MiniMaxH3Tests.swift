import BinaryResources
import Foundation
import NNC
import Tokenizer
import XCTest

@testable import Diffusion

final class MiniMaxH3Tests: XCTestCase {
  #if os(macOS)
    func testSolAttentionDenseLayersAndProtectedTokens() throws {
      guard DeviceKind.GPUs.count > 0 else { throw XCTSkip("Metal GPU required") }
      let savedFlags = DynamicGraph.flags
      let savedWatermark = DynamicGraph.queueWatermark
      defer {
        DynamicGraph.flags = savedFlags
        DynamicGraph.queueWatermark = savedWatermark
      }
      DynamicGraph.queueWatermark = 1
      DynamicGraph.flags.insert(.disableMFAAppleNeuralEngine)
      let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: directory) }
      for disableNA in [false, true] {
        if disableNA {
          DynamicGraph.flags.insert(.disableMFANeuralAccelerators)
        } else {
          DynamicGraph.flags.remove(.disableMFANeuralAccelerators)
        }
        let graph = DynamicGraph()
        try graph.withNoGrad {
          // Include full protected VL blocks, image/audio references, a boundary-crossing
          // block and a tail. Only the final 512 generated-video tokens may be summarized.
          let protectedLength = 137 + 6 + 6 + 10
          let sequenceLength = protectedLength + 512
          let state = graph.variable(
            Tensor<Float>(
              (0..<(sequenceLength * 8)).map { sin(Float($0) * 0.37) * 0.4 },
              .CPU, .HWC(1, sequenceLength, 8)
            ).toGPU(0))
          let rotary = graph.variable(
            Tensor<Float16>(
              from: MiniMaxH3RotaryEmbedding(
                textLength: 137, audioLength: 10, videoFrames: 2, videoHeight: 32,
                videoWidth: 32, referenceImages: [(4, 6, 137)],
                referenceAudios: [(6, 137)], visionTokenRanges: [9..<137])
            ).toGPU(0))
          let modulation = graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self)
          modulation.full(0.1)
          // Check all five dense layers and the surrounding Sol layers using global indices.
          for layer in [0, 1, 2, 32, 33, 34, 35, 36, 49] {
            let usesDenseAttention = [0, 1, 33, 34, 35].contains(layer)
            var inputs: [DynamicGraph.AnyTensor]
            if layer == 0 {
              let video = graph.variable(.GPU(0), .NHWC(2, 32, 32, 24), of: Float16.self)
              let audio = graph.variable(.GPU(0), .HWC(1, 10, 32), of: Float16.self)
              let reference = graph.variable(.GPU(0), .HWC(1, 6, 8), of: Float16.self)
              video.full(0.1)
              audio.full(0.1)
              reference.full(0.1)
              inputs = [
                video, audio, state[0..<1, 0..<137, 0..<8].copied(), rotary, reference, reference,
              ]
            } else {
              inputs = [state, rotary]
            }
            inputs += Array(repeating: modulation, count: layer == 0 ? 30 : 30 + 4)
            func build(
              sol: (eligibleForApproximation: Range<Int>, tau: Float)?, lora: Bool
            ) -> (ModelWeightMapper, Model) {
              if lora {
                return LoRAMiniMaxH3(
                  hiddenSize: 8, layers: 1, startLayer: layer, textLength: 137, audioLength: 10,
                  videoFrames: 2, videoHeight: 32, videoWidth: 32,
                  usesFlashAttention: .quantized, usesSolAttention: sol,
                  referenceImageSizes: [(4, 6)], referenceAudioLengths: [6], visionLength: 128,
                  outputResidual: true,
                  LoRAConfiguration: LoRANetworkConfiguration(
                    rank: 0, scale: 1, highPrecision: false))
              }
              return MiniMaxH3(
                hiddenSize: 8, layers: 1, startLayer: layer, textLength: 137, audioLength: 10,
                videoFrames: 2, videoHeight: 32, videoWidth: 32,
                usesFlashAttention: .quantized, usesSolAttention: sol,
                referenceImageSizes: [(4, 6)], referenceAudioLengths: [6], visionLength: 128,
                outputResidual: true)
            }
            let (denseMapper, dense) = build(sol: nil, lora: false)
            let expected = dense(inputs: inputs[0], Array(inputs.dropFirst())).map {
              DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
            }
            let path = directory.appendingPathComponent("dense.ckpt").path
            graph.openStore(path) { $0.write("dit", model: dense) }
            var solOutputs = [Tensor<Float>]()
            for lora in [false, true] {
              let (mapper, model) = build(sol: (protectedLength..<sequenceLength, 0.5), lora: lora)
              model.compile(inputs: inputs)
              for format in [ModelWeightFormat.diffusers, .generativeModels] {
                let base = denseMapper(format)
                let candidate = mapper(format)
                XCTAssertEqual(Set(candidate.keys), Set(base.keys))
                for (key, names) in base {
                  XCTAssertEqual(candidate[key].map { Array($0) }, Array(names), key)
                }
              }
              try graph.openStore(path, flags: .readOnly) {
                try $0.read("dit", model: model, strict: true)
              }
              let actual = model(inputs: inputs[0], Array(inputs.dropFirst())).map {
                DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
              }
              let allExact = build(sol: (0..<0, 0.5), lora: lora).1
              allExact.compile(inputs: inputs)
              try graph.openStore(path, flags: .readOnly) {
                try $0.read("dit", model: allExact, strict: true)
              }
              let exactOutputs = allExact(inputs: inputs[0], Array(inputs.dropFirst())).map {
                DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
              }
              // Empty-range Sol retains the operator while disabling every summary route.
              // Its kernel can round differently from SDPA, so compare the FP32 residual.
              let exactResidual = exactOutputs.last!.reshaped(.C(sequenceLength * 8))
              let denseResidual = expected.last!.reshaped(.C(sequenceLength * 8))
              var exactError = 0.0
              var denseEnergy = 0.0
              for i in 0..<(sequenceLength * 8) {
                XCTAssertTrue(exactResidual[i].isFinite)
                exactError += pow(Double(exactResidual[i]) - Double(denseResidual[i]), 2)
                denseEnergy += pow(Double(denseResidual[i]), 2)
              }
              let exactRelativeError = sqrt(exactError / max(denseEnergy, 1e-30))
              print(
                "H3 all-exact Sol nonNA=\(disableNA) layer=\(layer) LoRA=\(lora) relativeL2=\(exactRelativeError)"
              )
              XCTAssertLessThan(exactRelativeError, 0.005)
              // Change only eligibility without changing input shapes or reloading weights.
              var tau: Float = 0.5
              let dynamic = ModelBuilderOrModel.modelBuilder(
                ModelBuilder { eligible, _ in
                  build(sol: (eligible ? protectedLength..<sequenceLength : 0..<0, tau), lora: lora)
                    .1
                })
              dynamic.compile(false, inputs: inputs)
              try graph.openStore(path, flags: .readOnly) {
                try $0.read("dit", model: dynamic.unwrapped, strict: true)
              }
              for sol in [false, true, true, false] {
                let outputs = dynamic(sol, inputs: inputs[0], Array(inputs.dropFirst())).map {
                  DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
                }
                let reference = sol ? actual : exactOutputs
                XCTAssertEqual(outputs.count, reference.count)
                for output in outputs.indices {
                  XCTAssertEqual(outputs[output].shape, reference[output].shape)
                  let count = outputs[output].shape.reduce(1, *)
                  let a = outputs[output].reshaped(.C(count))
                  let b = reference[output].reshaped(.C(count))
                  var differences = 0
                  var maxError: Float = 0
                  for i in 0..<count where a[i] != b[i] {
                    differences += 1
                    maxError = max(maxError, abs(a[i] - b[i]))
                  }
                  XCTAssertEqual(
                    differences, 0,
                    "nonNA=\(disableNA) layer=\(layer) LoRA=\(lora) eligible=\(sol) output=\(output) maxError=\(maxError)"
                  )
                }
              }
              if layer == 2 {
                // A non-default tau must reach the kernel and update through explicit compilation.
                tau = 1.3
                let alternate = build(sol: (protectedLength..<sequenceLength, tau), lora: lora).1
                alternate.compile(inputs: inputs)
                try graph.openStore(path, flags: .readOnly) {
                  try $0.read("dit", model: alternate, strict: true)
                }
                let reference = alternate(inputs: inputs[0], Array(inputs.dropFirst())).map {
                  DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
                }
                dynamic.compile(true, inputs: inputs)
                let outputs = dynamic(true, inputs: inputs[0], Array(inputs.dropFirst())).map {
                  DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
                }
                var tauDifferences = 0
                for output in outputs.indices {
                  let count = outputs[output].shape.reduce(1, *)
                  let a = outputs[output].reshaped(.C(count))
                  let b = reference[output].reshaped(.C(count))
                  let previous = actual[output].reshaped(.C(count))
                  var differences = 0
                  for i in 0..<count {
                    if a[i] != b[i] { differences += 1 }
                    if b[i] != previous[i] { tauDifferences += 1 }
                  }
                  XCTAssertEqual(differences, 0)
                }
                XCTAssertGreaterThan(tauDifferences, 0)
              }
              XCTAssertEqual(actual.count, layer == 0 ? 2 : 3)
              for output in actual.indices {
                XCTAssertEqual(actual[output].shape, expected[output].shape)
                let count = actual[output].shape.reduce(1, *)
                let a = actual[output].reshaped(.C(count))
                let e = expected[output].reshaped(.C(count))
                for i in 0..<count {
                  XCTAssertTrue(a[i].isFinite)
                  if usesDenseAttention { XCTAssertEqual(a[i], e[i]) }
                  if lora {
                    XCTAssertEqual(a[i], solOutputs[output].reshaped(.C(count))[i])
                  }
                }
              }
              // A single block cannot propagate approximated video-query results into
              // protected queries. Check its FP32 residual, before another block can mix them.
              let a = actual.last!.reshaped(.C(sequenceLength * 8))
              let e = expected.last!.reshaped(.C(sequenceLength * 8))
              var error = 0.0
              var energy = 0.0
              for i in 0..<(protectedLength * 8) {
                error += pow(Double(a[i]) - Double(e[i]), 2)
                energy += pow(Double(e[i]), 2)
              }
              let relativeError = sqrt(error / max(energy, 1e-30))
              print(
                "H3 Sol nonNA=\(disableNA) layer=\(layer) LoRA=\(lora) protected relativeL2=\(relativeError)"
              )
              XCTAssertLessThan(relativeError, 0.005)
              if !usesDenseAttention {
                var videoDifference: Float = 0
                for i in (protectedLength * 8)..<(sequenceLength * 8) {
                  videoDifference = max(videoDifference, abs(a[i] - e[i]))
                }
                // Ensure opting in really selected Sol, not another dense graph.
                XCTAssertGreaterThan(videoDifference, 0)
              }
              if !lora { solOutputs = actual }
            }
          }
        }
      }
    }
  #endif

  func testLoRAMergedAndSeparateExecution() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let graph = DynamicGraph()
    graph.withNoGrad {
      for (fixed, hasAudio) in [(false, false), (true, false), (false, true), (true, true)] {
        func build(_ configuration: LoRANetworkConfiguration?) -> (ModelWeightMapper, Model) {
          if fixed {
            if let configuration {
              return LoRAMiniMaxH3Fixed(
                timesteps: 1, hiddenSize: 8, layers: 2, textLength: (3, 5),
                usesFlashAttention: .scale1, referenceImageCount: 1,
                referenceAudioCount: hasAudio ? 1 : 0,
                LoRAConfiguration: configuration
              )
            }
            return MiniMaxH3Fixed(
              timesteps: 1, hiddenSize: 8, layers: 2, textLength: (3, 5),
              usesFlashAttention: .scale1, referenceImageCount: 1,
              referenceAudioCount: hasAudio ? 1 : 0
            )
          }
          if let configuration {
            return LoRAMiniMaxH3(
              hiddenSize: 8, layers: 2, textLength: 3, audioLength: 4, videoFrames: 1,
              videoHeight: 2, videoWidth: 2, usesFlashAttention: .scale1,
              referenceImageSizes: [(2, 2)], referenceAudioLengths: hasAudio ? [6] : [],
              LoRAConfiguration: configuration
            )
          }
          return MiniMaxH3(
            hiddenSize: 8, layers: 2, textLength: 3, audioLength: 4, videoFrames: 1,
            videoHeight: 2, videoWidth: 2, usesFlashAttention: .scale1,
            referenceImageSizes: [(2, 2)], referenceAudioLengths: hasAudio ? [6] : []
          )
        }
        let inputs: [DynamicGraph.AnyTensor]
        if fixed {
          inputs =
            [
              graph.variable(.GPU(0), .HWC(2, 5, 5_120), of: Float16.self),
              graph.variable(.GPU(0), .HWC(1, hasAudio ? 4 : 3, 256), of: Float16.self),
              graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self),
            ] + (hasAudio ? [graph.variable(.GPU(0), .HWC(1, 6, 32), of: Float16.self)] : [])
        } else {
          inputs =
            [
              graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self),
              graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self),
              graph.variable(.GPU(0), .HWC(1, 3, 8), of: Float.self),
              graph.variable(.GPU(0), .NHWC(1, hasAudio ? 15 : 9, 1, 128), of: Float16.self),
              graph.variable(.GPU(0), .NHWC(1, 1, 1, 8), of: Float16.self),
            ]
            + (hasAudio ? [graph.variable(.GPU(0), .HWC(1, 6, 8), of: Float16.self)] : [])
            + (0..<(hasAudio ? 64 : 52)).map { _ in
              graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self)
            }
        }
        for (index, input) in inputs.enumerated() {
          if !fixed && index == 2 {
            input.as(of: Float.self).full(0.1)
          } else {
            input.as(of: Float16.self).full(0.1)
          }
        }
        let (baseMapper, base) = build(nil)
        let baseline = base(inputs: inputs[0], Array(inputs.dropFirst())).map {
          DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
        }
        // Compare base parameter names without active LoRA branches in the graph.
        let (mapper, counterpart) = build(
          LoRANetworkConfiguration(rank: 0, scale: 1, highPrecision: false))
        counterpart.compile(inputs: inputs)
        for format in [ModelWeightFormat.diffusers, .generativeModels] {
          let expected = baseMapper(format)
          let actual = mapper(format)
          XCTAssertEqual(Set(actual.keys), Set(expected.keys))
          for (key, expected) in expected {
            guard let actual = actual[key] else {
              XCTFail("Missing mapping: \(key)")
              continue
            }
            XCTAssertEqual(Array(actual), Array(expected), key)
            XCTAssertTrue(actual.format == expected.format, key)
            XCTAssertEqual(actual.offsets, expected.offsets, key)
            XCTAssertEqual(actual.scale, expected.scale, key)
            XCTAssertEqual(actual.shift, expected.shift, key)
            XCTAssertEqual(actual.index, expected.index, key)
            XCTAssertEqual(actual.isBF16, expected.isBF16, key)
            XCTAssertEqual(actual.interleavedIndices, expected.interleavedIndices, key)
            XCTAssertEqual(actual.numberOfHeads, expected.numberOfHeads, key)
            XCTAssertEqual(actual.headDimension, expected.headDimension, key)
            XCTAssertEqual(actual.interleavedDimension, expected.interleavedDimension, key)
          }
        }
        let basePath = directory.appendingPathComponent(fixed ? "fixed.ckpt" : "main.ckpt").path
        let loraPath = directory.appendingPathComponent(
          fixed ? "fixed-lora.ckpt" : "main-lora.ckpt"
        ).path
        graph.openStore(basePath) { $0.write("dit", model: base) }
        graph.openStore(basePath, flags: .readOnly) { store in
          graph.openStore(loraPath) { lora in
            for key in store.keys where key.hasSuffix("-0]") {
              guard let weight = store.read(like: key), weight.shape.count >= 2 else { continue }
              let inputSize = weight.shape.dropFirst().reduce(1, *)
              let outputSize = weight.shape[0]
              let down = Tensor<Float16>(
                [Float16](repeating: 0.01, count: 2 * inputSize), .CPU, .NC(2, inputSize))
              let up = Tensor<Float16>(
                [Float16](repeating: 0.01, count: outputSize * 2), .CPU, .NC(outputSize, 2))
              lora.write(key + "__down__", tensor: down)
              lora.write(key + "__up__", tensor: up)
            }
          }
        }
        let keys = LoRALoader.keys(graph, of: [loraPath], modelFile: basePath)
        XCTAssertFalse(keys.isEmpty)
        var results = [[Tensor<Float>]]()
        for separate in [false, true] {
          let (_, model) = build(
            separate
              ? LoRANetworkConfiguration(
                rank: 2, scale: 1, highPrecision: false, keys: keys) : nil)
          model.compile(inputs: inputs)
          graph.openStore(basePath, flags: .readOnly) { store in
            LoRALoader.openStore(
              graph,
              lora: [
                LoRAConfiguration(
                  file: loraPath, weight: 1, version: .minimaxH3, isLoHa: false, modifier: .none,
                  mode: .all)
              ]
            ) { loader in
              store.read("dit", model: model) { name, dataType, format, shape in
                if separate {
                  return loader.concatenateLoRA(
                    graph, LoRAMapping: [0: 0, 1: 1], filesRequireMerge: [:],
                    name: name, store: store, dataType: dataType, format: format, shape: shape,
                    of: Float16.self)
                }
                return loader.mergeLoRA(
                  graph, name: name, store: store, dataType: dataType, shape: shape,
                  of: Float16.self)
              }
            }
          }
          results.append(
            model(inputs: inputs[0], Array(inputs.dropFirst())).map {
              DynamicGraph.Tensor<Float>(from: $0).toCPU().rawValue
            })
        }
        var error = 0.0
        var magnitude = 0.0
        var changed = 0.0
        for index in results[0].indices {
          let count = results[0][index].shape.reduce(1, *)
          let merged = results[0][index].reshaped(.C(count))
          let separate = results[1][index].reshaped(.C(count))
          let original = baseline[index].reshaped(.C(count))
          for i in 0..<count {
            XCTAssertTrue(merged[i].isFinite && separate[i].isFinite)
            error += pow(Double(merged[i] - separate[i]), 2)
            magnitude += pow(Double(merged[i]), 2)
            changed += pow(Double(merged[i] - original[i]), 2)
          }
        }
        XCTAssertGreaterThan(changed, 1e-8)
        XCTAssertLessThan(sqrt(error / max(magnitude, 1e-30)), 0.01)
      }
    }
  }

  func testAudioReferencesAdvanceClockAndKeepStereoRowsTogether() {
    let rotary = MiniMaxH3RotaryEmbedding(
      textLength: 10, audioLength: 4, videoFrames: 1, videoHeight: 4, videoWidth: 6,
      referenceImages: [(4, 6, 10)], referenceAudios: [(6, 11), (4, 14)], videoPosition: 16)
    XCTAssertEqual(rotary.shape, [1, 36, 1, 128])
    // Image rows follow text, then each reference's left and right channel.
    for (offset, length, start) in [(16, 3, Float(11)), (22, 2, Float(14))] {
      for channel in 0..<2 {
        for i in 0..<length {
          XCTAssertEqual(
            rotary[0, offset + channel * length + i, 0, 0], cos(start + Float(i)), accuracy: 1e-6)
        }
      }
      XCTAssertNotEqual(rotary[0, offset, 0, 64], rotary[0, offset + length, 0, 64])
    }
    // Generated audio/video both start after the references' temporal span.
    XCTAssertEqual(rotary[0, 26, 0, 0], cos(Float(16)), accuracy: 1e-6)
    XCTAssertEqual(rotary[0, 30, 0, 0], cos(Float(16)), accuracy: 1e-6)
  }

  func testAudioReferenceModulationStaysCleanAcrossTimesteps() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let fixed = MiniMaxH3Fixed(
        timesteps: 2, hiddenSize: 8, layers: 1, textLength: (0, 3),
        usesFlashAttention: .scale1, referenceAudioCount: 1
      ).1
      let text = graph.variable(.GPU(0), .HWC(1, 3, 5_120), of: Float16.self)
      text.full(0.1)
      var frequencies = Tensor<Float16>(.CPU, .HWC(2, 3, 256))
      for step in 0..<2 {
        for modality in 0..<3 {
          for feature in 0..<256 {
            frequencies[step, modality, feature] = modality == 2 ? 0.9 : Float16(step) * 0.5
          }
        }
      }
      let reference = graph.variable(.GPU(0), .HWC(1, 6, 32), of: Float16.self)
      reference.full(0.25)
      let outputs = fixed(inputs: text, graph.variable(frequencies.toGPU(0)), reference)
      XCTAssertEqual(outputs.count, 30)
      XCTAssertEqual(outputs[1].shape, [1, 6, 8])
      var targetChange: Float = 0
      for chunk in 0..<6 {
        let target = outputs[2 + chunk * 4 + 2].as(of: Float16.self).rawValue.toCPU()
        let anchor = outputs[2 + chunk * 4 + 3].as(of: Float16.self).rawValue.toCPU()
        for feature in 0..<8 {
          XCTAssertEqual(anchor[0, 0, feature], anchor[1, 0, feature])
          targetChange += abs(Float(target[0, 0, feature] - target[1, 0, feature]))
        }
      }
      XCTAssertGreaterThan(targetChange, 0)
    }
  }

  func testDenoiserUsesAudioWithoutReturningReferenceRows() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for imageCount in 0...1 {
        let model = MiniMaxH3(
          hiddenSize: 8, layers: 1, textLength: 3, audioLength: 4,
          videoFrames: 1, videoHeight: 2, videoWidth: 2, usesFlashAttention: .scale1,
          referenceImageSizes: imageCount == 0 ? [] : [(2, 2)], referenceAudioLengths: [6]
        ).1
        let video = graph.variable(.GPU(0), .NHWC(1, 2, 2, 24), of: Float16.self)
        let audio = graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self)
        let text = graph.variable(.GPU(0), .HWC(1, 3, 8), of: Float.self)
        let reference = graph.variable(.GPU(0), .HWC(1, 6, 8), of: Float16.self)
        let image = graph.variable(.GPU(0), .NHWC(1, 1, 1, 8), of: Float16.self)
        let modulation = graph.variable(.GPU(0), .HWC(1, 1, 8), of: Float16.self)
        video.full(0.1)
        audio.full(0.1)
        text.full(0.1)
        image.full(0.1)
        modulation.full(0.1)
        reference.full(0.25)
        let rotary = graph.variable(
          Tensor<Float16>(
            from: MiniMaxH3RotaryEmbedding(
              textLength: 3, audioLength: 4, videoFrames: 1, videoHeight: 2, videoWidth: 2,
              referenceImages: imageCount == 0 ? [] : [(2, 2, 3)],
              referenceAudios: [(6, Float(3 + imageCount))], videoPosition: Float(6 + imageCount))
          ).toGPU(0))
        let inputs: [DynamicGraph.AnyTensor] =
          [audio, text, rotary]
          + (imageCount == 0 ? [] : [image]) + [reference]
          + Array(repeating: modulation, count: 28 + imageCount * 6)
        let first = model(inputs: video, inputs).map { $0.as(of: Float16.self).rawValue.toCPU() }
        reference.full(-0.4)
        let second = model(inputs: video, inputs).map { $0.as(of: Float16.self).rawValue.toCPU() }
        XCTAssertEqual(first.count, 2)
        XCTAssertEqual(Array(first[0].shape), [1, 2, 2, 24])
        XCTAssertEqual(Array(first[1].shape), [1, 4, 32])
        for output in 0..<2 {
          let count = first[output].shape.reduce(1, *)
          let a = first[output].reshaped(.C(count))
          let b = second[output].reshaped(.C(count))
          var difference: Float = 0
          for i in 0..<count {
            XCTAssertTrue(a[i].isFinite && b[i].isFinite)
            difference += abs(Float(a[i]) - Float(b[i]))
          }
          XCTAssertGreaterThan(difference, 0.001)
        }
      }
    }
  }

  func testFirstPicturePrefixMatchesRotaryLayout() {
    let tokenizer = TiktokenTokenizer(
      vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
      specialTokens: [
        "<|endoftext|>": 151_643, "<|vision_start|>": 151_652,
        "<|vision_end|>": 151_653, "<|image_pad|>": 151_655,
      ])
    let tokens = tokenizer.tokenize(
      text: "<Picture 1>: <|vision_start|><|image_pad|><|vision_end|>Hello"
    ).0
    XCTAssertEqual(tokens.firstIndex(of: 151_652), 6)
    XCTAssertEqual(tokens.firstIndex(of: 151_655), 7)
  }

  func testGroupedRotaryPreservesEveryTokenPosition() {
    for frames in [1, 2, 7] {
      for (height, width) in [(4, 6), (6, 4)] {
        for textLength in [14, 20] {
          let original = MiniMaxH3RotaryEmbedding(
            textLength: textLength, audioLength: 4, videoFrames: frames,
            videoHeight: height, videoWidth: width,
            referenceImages: [(height, width, Float(textLength))])
          let grouped = MiniMaxH3RotaryEmbedding(
            textLength: textLength, audioLength: 4, videoFrames: frames,
            videoHeight: height, videoWidth: width,
            referenceImages: [(height, width, Float(textLength))], visionTokenRanges: [6..<12])
          let videoOffset = textLength + height / 2 * (width / 2) + 4
          let order =
            Array(0..<6) + Array(12..<textLength)
            + Array(textLength..<videoOffset) + Array(6..<12)
            + Array(videoOffset..<original.shape[1])
          XCTAssertEqual(order.count, grouped.shape[1])
          for (row, originalRow) in order.enumerated() {
            for channel in 0..<128 {
              XCTAssertEqual(grouped[0, row, 0, channel], original[0, originalRow, 0, channel])
            }
          }
        }
      }
    }
  }

  func testQwenVLImageRotaryUsesInterleavedAxes() {
    let plain = QwenVLRotaryEmbedding(sequenceLength: 12, of: Float.self)
    let image = QwenVLRotaryEmbedding(
      sequenceLength: 12, images: [(start: 3, height: 2, width: 3)], of: Float.self)
    for row in 0..<12 {
      for k in 0..<64 {
        let position: Int
        if row < 3 {
          position = row
        } else if row < 9 {
          if k % 3 == 1 && k < 60 {
            position = 3 + (row - 3) / 3
          } else if k % 3 == 2 && k < 60 {
            position = 3 + (row - 3) % 3
          } else {
            position = 3
          }
        } else {
          position = row - 3
        }
        XCTAssertEqual(image[0, row, 0, k * 2], plain[0, position, 0, k * 2])
        XCTAssertEqual(image[0, row, 0, k * 2 + 1], plain[0, position, 0, k * 2 + 1])
      }
    }
  }

  func testQwenDeepStackReusesTextWeightsAndInjectsAfterBlock() throws {
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "qwen-deepstack-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let tokens = graph.variable(Tensor<Int32>([1, 2, 3, 4], .CPU, .C(4)).toGPU(0))
      let rotary = graph.variable(
        QwenVLRotaryEmbedding(sequenceLength: 4, of: Float16.self).toGPU(0))
      let causalMask = graph.variable(.GPU(0), .NHWC(1, 1, 4, 4), of: Float16.self)
      causalMask.full(0)
      let plain = Qwen3(
        Float16.self, vocabularySize: 8, width: 16, tokenLength: 4, layers: 3, MLP: 32,
        heads: 8, outputHiddenStates: [0, 1, 2], noFinalNormalizedOutput: true,
        batchSize: 1, usesFlashAttention: true)
      let expected = plain(inputs: tokens, rotary, causalMask).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      graph.openStore(path) { $0.write("text_model", model: plain) }
      let injectedModel = Qwen3(
        Float16.self, vocabularySize: 8, width: 16, tokenLength: 4, layers: 3, MLP: 32,
        heads: 8, outputHiddenStates: [0, 1, 2], noFinalNormalizedOutput: true,
        batchSize: 1, usesFlashAttention: true, injectEmbeddings: true, deepStackLayers: 3)
      let mask = graph.variable(.GPU(0), .WC(4, 1), of: Float16.self)
      mask.full(1)
      let zero = graph.variable(.GPU(0), .WC(4, 16), of: Float16.self)
      zero.full(0)
      let deepStack = graph.variable(like: zero)
      deepStack.full(0)
      let inputs: [DynamicGraph.AnyTensor] = [
        rotary, causalMask, mask, zero, deepStack, zero, zero,
      ]
      injectedModel.compile(inputs: [tokens] + inputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("text_model", model: injectedModel, strict: true)
      }
      let unchanged = injectedModel(inputs: tokens, inputs).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      deepStack[2..<3, 0..<16].full(0.25)
      let actual = injectedModel(inputs: tokens, inputs).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      for layer in 0..<3 {
        for row in 0..<4 {
          for channel in 0..<16 {
            XCTAssertEqual(unchanged[layer][row, channel], expected[layer][row, channel])
            XCTAssertTrue(actual[layer][row, channel].isFinite)
            if layer == 0 {
              XCTAssertEqual(
                Float(actual[layer][row, channel]),
                Float(expected[layer][row, channel]) + (row == 2 ? 0.25 : 0), accuracy: 0.002)
            }
          }
        }
      }
    }
  }

  func testMultipleVisionRangesPreserveOriginalContextPositions() {
    let ranges = [6..<12, 18..<28]
    let plain = MiniMaxH3RotaryEmbedding(textLength: 32)
    let packed = MiniMaxH3RotaryEmbedding(textLength: 32, visionTokenRanges: ranges)
    let order =
      (0..<32).filter { row in !ranges.contains { $0.contains(row) } }
      + ranges.flatMap { Array($0) }
    for (row, original) in order.enumerated() {
      for channel in 0..<128 {
        XCTAssertEqual(packed[0, row, 0, channel], plain[0, original, 0, channel])
      }
    }
  }

  func testReferenceImagePositionsAndIndependentSpatialGrids() {
    let frames = 7
    let textLength = 20
    let lastPosition = Float(textLength) + Float(22 - 1) * 5 / 3
    let endpoints = MiniMaxH3RotaryEmbedding(
      textLength: textLength, audioLength: 4, videoFrames: frames,
      videoHeight: 4, videoWidth: 6,
      referenceImages: [(4, 6, Float(textLength)), (4, 6, lastPosition)])
    let plain = MiniMaxH3RotaryEmbedding(
      textLength: textLength, audioLength: 4, videoFrames: frames,
      videoHeight: 4, videoWidth: 6)
    for row in 0..<(4 + frames * 6) {
      for channel in 0..<128 {
        XCTAssertEqual(
          endpoints[0, textLength + 12 + row, 0, channel], plain[0, textLength + row, 0, channel])
      }
    }
    XCTAssertEqual(endpoints[0, textLength + 6, 0, 0], cos(lastPosition), accuracy: 1e-6)
    let references = MiniMaxH3RotaryEmbedding(
      textLength: textLength, audioLength: 4, videoFrames: frames,
      videoHeight: 4, videoWidth: 6,
      referenceImages: [(4, 6, 20), (8, 4, 21)], videoPosition: 22)
    let second = MiniMaxH3RotaryEmbedding(
      textLength: 0, videoFrames: 1, videoHeight: 8, videoWidth: 4, videoPosition: 21)
    for row in 0..<8 {
      for channel in 0..<128 {
        XCTAssertEqual(references[0, textLength + 6 + row, 0, channel], second[0, row, 0, channel])
      }
    }
    XCTAssertEqual(references[0, textLength + 14 + 4, 0, 0], cos(Float(22)), accuracy: 1e-6)
  }

  func testQwenVLMultipleImagesAdvanceByCompressedGridSize() {
    let rotary = QwenVLRotaryEmbedding(
      sequenceLength: 24, images: [(3, 2, 3), (12, 3, 2)], of: Float.self)
    let plain = QwenVLRotaryEmbedding(sequenceLength: 24, of: Float.self)
    for row in 0..<24 {
      for k in 0..<64 {
        let position: Int
        if (3..<9).contains(row) {
          position =
            3 + (k % 3 == 1 && k < 60 ? (row - 3) / 3 : (k % 3 == 2 && k < 60 ? (row - 3) % 3 : 0))
        } else if (12..<18).contains(row) {
          position =
            9
            + (k % 3 == 1 && k < 60 ? (row - 12) / 2 : (k % 3 == 2 && k < 60 ? (row - 12) % 2 : 0))
        } else {
          position = row - (row >= 9 ? 3 : 0) - (row >= 18 ? 3 : 0)
        }
        for component in 0..<2 {
          XCTAssertEqual(
            rotary[0, row, 0, 2 * k + component], plain[0, position, 0, 2 * k + component])
        }
      }
    }
  }

  func testFirstFrameRotaryPreservesTargetPositions() {
    for frames in [1, 2, 7] {
      for (height, width) in [(4, 6), (6, 4)] {
        let textLength = 3
        let audioLength = 4
        let frameLength = height / 2 * (width / 2)
        let plain = MiniMaxH3RotaryEmbedding(
          textLength: textLength, audioLength: audioLength, videoFrames: frames,
          videoHeight: height, videoWidth: width)
        let conditioned = MiniMaxH3RotaryEmbedding(
          textLength: textLength, audioLength: audioLength, videoFrames: frames,
          videoHeight: height, videoWidth: width,
          referenceImages: [(height, width, Float(textLength))])
        XCTAssertEqual(conditioned.shape[1], plain.shape[1] + frameLength)
        for row in 0..<plain.shape[1] {
          let targetRow = row < textLength ? row : row + frameLength
          for channel in 0..<128 {
            XCTAssertEqual(plain[0, row, 0, channel], conditioned[0, targetRow, 0, channel])
          }
        }
        for row in 0..<frameLength {
          for channel in 0..<128 {
            XCTAssertEqual(
              conditioned[0, textLength + row, 0, channel],
              plain[0, textLength + audioLength + row, 0, channel])
          }
        }
      }
    }
  }

  func testExtractConditionsKeepsKeyframeIndependentOfStepAndCFG() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let text = graph.variable(.CPU, .HWC(2, 3, 8), of: Float.self)
      let rotary = graph.variable(.CPU, .NHWC(2, 25, 1, 128), of: Float.self)
      let firstFrame = graph.variable(.CPU, .NHWC(1, 2, 3, 8), of: Float.self)
      firstFrame.full(0.25)
      let modulation = graph.variable(.CPU, .HWC(3, 1, 8), of: Float.self)
      for step in 0..<3 { modulation[step..<(step + 1), 0..<1, 0..<8].full(Float(step)) }
      for referenceCount in 0...3 {
        let prefix: [DynamicGraph.AnyTensor] =
          [text, rotary] + Array(repeating: firstFrame, count: referenceCount)
        let count = 50 * (referenceCount > 0 ? 24 : 18) + 4
        let conditions = prefix + Array(repeating: modulation, count: count)
        for step in 0..<3 {
          let extracted = UNetExtractConditions(
            of: Float.self, graph: graph, index: step, batchSize: 4,
            tokenLengthUncond: 2, tokenLengthCond: 3, conditions: conditions,
            referenceImageCount: referenceCount, referenceAudioCount: 0, version: .minimaxH3,
            modifier: .fl2va,
            isCfgEnabled: true)
          XCTAssertEqual(extracted.count, prefix.count + count)
          XCTAssertEqual(Array(extracted[0].shape), [2, 3, 8])
          if referenceCount > 0 {
            XCTAssertEqual(Array(extracted[2].shape), [1, 2, 3, 8])
            XCTAssertEqual(DynamicGraph.Tensor<Float>(extracted[2]).rawValue[0, 1, 2, 7], 0.25)
          }
          for index in [prefix.count, extracted.count - 1] {
            let value = DynamicGraph.Tensor<Float>(extracted[index]).rawValue
            XCTAssertEqual(Array(value.shape), [1, 8])
            XCTAssertEqual(value[0, 7], Float(step))
          }
        }
      }
    }
  }

  func testFixedKeyframeModulationReusesVideoWeights() throws {
    let hiddenSize = 128
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "h3-fixed-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let frequencies = graph.variable(.GPU(0), .HWC(2, 2, 256), of: Float.self)
      frequencies.randn()
      let text = graph.variable(.GPU(0), .HWC(1, 4, 5120), of: Float16.self)
      text.full(0.1)
      var conditionedFrequencies = graph.variable(.GPU(0), .HWC(2, 3, 256), of: Float.self)
      conditionedFrequencies[0..<2, 0..<2, 0..<256] = frequencies
      // At equal timesteps the new keyframe modulation must equal video modulation exactly.
      conditionedFrequencies[0..<2, 2..<3, 0..<256] = frequencies[0..<2, 0..<1, 0..<256]
      let firstFrame = graph.variable(.GPU(0), .NHWC(1, 4, 6, 24), of: Float16.self)
      let lastFrame = graph.variable(.GPU(0), .NHWC(1, 8, 4, 24), of: Float16.self)
      firstFrame.randn()
      lastFrame.randn()
      let conditioned = MiniMaxH3Fixed(
        timesteps: 2, hiddenSize: hiddenSize, layers: 2, textLength: (0, 4),
        usesFlashAttention: .scale1,
        referenceImageCount: 2
      ).1
      let outputs = conditioned(inputs: text, conditionedFrequencies, firstFrame, lastFrame)
      XCTAssertEqual(Array(outputs[1].shape), [1, 2, 3, hiddenSize])
      XCTAssertEqual(Array(outputs[2].shape), [1, 4, 2, hiddenSize])
      let actual = outputs.dropFirst(3).map { $0.as(of: Float.self).toCPU().rawValue }
      graph.openStore(path) { $0.write("dit", model: conditioned) }
      let plain = MiniMaxH3Fixed(
        timesteps: 2, hiddenSize: hiddenSize, layers: 2, textLength: (0, 4),
        usesFlashAttention: .scale1
      ).1
      plain.compile(inputs: text, frequencies)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: plain, strict: true)
      }.get()
      let expected = plain(inputs: text, frequencies).dropFirst().map {
        $0.as(of: Float.self).toCPU().rawValue
      }
      // The cached projection uses the original denoiser weight names, including its bias.
      for (index, reference) in [firstFrame, lastFrame].enumerated() {
        let image = Input()
        let projection = Model(
          [image],
          [
            Convolution(
              groups: 1, filters: hiddenSize, filterSize: [2, 2], hint: Hint(stride: [2, 2]),
              format: .OIHW, name: "proj_in")(image).to(.Float32)
          ])
        projection.compile(inputs: reference)
        try graph.openStore(path, flags: .readOnly) {
          try $0.read("dit", model: projection, strict: true)
        }.get()
        let projected = projection(inputs: reference)[0].as(of: Float.self).toCPU().rawValue
        let count = projected.shape.reduce(1, *)
        let cached = outputs[index + 1].as(of: Float16.self).toCPU().rawValue.reshaped(.C(count))
        let direct = projected.reshaped(.C(count))
        for i in 0..<count { XCTAssertEqual(Float(cached[i]), direct[i]) }
        let tile = reference[0..<1, 2..<4, 0..<4, 0..<24].copied()
        let tileInput = Input()
        let tileModel = Model(
          [tileInput],
          [
            Convolution(
              groups: 1, filters: hiddenSize, filterSize: [2, 2], hint: Hint(stride: [2, 2]),
              format: .OIHW, name: "proj_in")(tileInput).to(.Float32)
          ])
        tileModel.compile(inputs: tile)
        try graph.openStore(path, flags: .readOnly) {
          try $0.read("dit", model: tileModel, strict: true)
        }.get()
        let tileProjection = tileModel(inputs: tile)[0].as(of: Float.self).toCPU().rawValue
        for x in 0..<2 {
          for channel in 0..<hiddenSize {
            XCTAssertEqual(tileProjection[0, 0, x, channel], projected[0, 1, x, channel])
          }
        }
      }
      XCTAssertEqual(actual.count, 2 * 24 + 4)
      for layer in 0..<2 {
        for chunk in 0..<6 {
          for modality in 0..<4 {
            let value = actual[layer * 24 + chunk * 4 + modality]
            let reference = expected[layer * 18 + chunk * 3 + (modality == 3 ? 0 : modality)]
            for step in 0..<2 {
              for channel in 0..<hiddenSize {
                XCTAssertEqual(value[step, 0, channel], reference[step, 0, channel], accuracy: 1e-5)
              }
            }
          }
        }
      }
      for index in 0..<4 {
        for step in 0..<2 {
          for channel in 0..<hiddenSize {
            XCTAssertEqual(
              actual[48 + index][step, 0, channel], expected[36 + index][step, 0, channel],
              accuracy: 1e-5)
          }
        }
      }
    }
  }

  func testFixedRefinerSlicesCFGAndVisionWithoutCrossBranchAttention() throws {
    let flags = DynamicGraph.flags
    DynamicGraph.flags.insert(.disableMFAAppleNeuralEngine)
    defer { DynamicGraph.flags = flags }
    DynamicGraph.setSeed(42)
    let watermark = DynamicGraph.queueWatermark
    DynamicGraph.queueWatermark = 1
    defer { DynamicGraph.queueWatermark = watermark }
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "h3-refiner-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let hiddenSize = 128
      let frequencies = graph.variable(.GPU(0), .HWC(2, 2, 256), of: Float16.self)
      frequencies.full(0.1)
      for lengths in [(3, 5), (5, 3)] {
        for visionLength in [0, 2] {
          let paddedLength = max(lengths.0, lengths.1)
          var text = graph.variable(
            .GPU(0), .HWC(2, paddedLength + visionLength, 5120), of: Float16.self)
          text.randn()
          let fixed = MiniMaxH3Fixed(
            timesteps: 2, hiddenSize: hiddenSize, layers: 1,
            textLength: (lengths.0 + visionLength, lengths.1 + visionLength),
            usesFlashAttention: .scale1, visionLength: visionLength
          ).1
          let outputs = fixed(inputs: text, frequencies)
          let batched = outputs[0].as(of: Float.self).toCPU().rawValue
          XCTAssertEqual(Array(batched.shape), [2, paddedLength + visionLength, hiddenSize])
          XCTAssertEqual(outputs.count, 23)
          graph.openStore(path) { $0.write("dit", model: fixed) }
          for (batch, length) in [lengths.0, lengths.1].enumerated() {
            var branch = text[batch..<(batch + 1), 0..<length, 0..<5120].copied()
            if visionLength > 0 {
              branch = Functional.concat(
                axis: 1, branch,
                text[
                  batch..<(batch + 1), paddedLength..<(paddedLength + visionLength), 0..<5120
                ].copied())
            }
            let single = MiniMaxH3Fixed(
              timesteps: 2, hiddenSize: hiddenSize, layers: 1,
              textLength: (0, length + visionLength),
              usesFlashAttention: .scale1, visionLength: visionLength
            ).1
            single.compile(inputs: branch, frequencies)
            try graph.openStore(path, flags: .readOnly) {
              try $0.read("dit", model: single, strict: true)
            }.get()
            let expected = single(inputs: branch, frequencies)[0].as(of: Float.self).toCPU()
              .rawValue
            var error = 0.0
            var magnitude = 0.0
            for row in 0..<(length + visionLength) {
              let actualRow = row < length ? row : paddedLength + row - length
              for channel in 0..<hiddenSize {
                let a = Double(batched[batch, actualRow, channel])
                let b = Double(expected[0, row, channel])
                XCTAssertTrue(a.isFinite && b.isFinite)
                error += (a - b) * (a - b)
                magnitude += b * b
              }
            }
            // Packed and single-branch GEMMs have different row counts. Compare numerically;
            // padding and cross-branch independence below must still be bit-exact.
            XCTAssertLessThan(sqrt(error / max(magnitude, 1e-30)), 0.003)
            for row in length..<paddedLength {
              for channel in 0..<hiddenSize { XCTAssertEqual(batched[batch, row, channel], 0) }
            }
          }
          // Padding is not part of either refiner attention segment.
          let padding = graph.variable(
            .GPU(0), .HWC(1, paddedLength - min(lengths.0, lengths.1), 5120), of: Float16.self)
          padding.full(100)
          let shorter = lengths.0 < lengths.1 ? 0 : 1
          text[shorter..<(shorter + 1), min(lengths.0, lengths.1)..<paddedLength, 0..<5120] =
            padding
          let changedPadding = fixed(inputs: text, frequencies)[0].as(of: Float.self).toCPU()
            .rawValue
          for batch in 0..<2 {
            for row in 0..<(paddedLength + visionLength) {
              for channel in 0..<hiddenSize {
                XCTAssertEqual(changedPadding[batch, row, channel], batched[batch, row, channel])
              }
            }
          }
          // Changing one entire branch cannot affect the other branch (including vision rows).
          for changedBatch in 0..<2 {
            let changed = text.copied()
            changed[changedBatch..<(changedBatch + 1), 0..<(paddedLength + visionLength), 0..<5120]
              .full(1)
            let result = fixed(inputs: changed, frequencies)[0].as(of: Float.self).toCPU().rawValue
            let unchangedBatch = 1 - changedBatch
            for row in 0..<(paddedLength + visionLength) {
              for channel in 0..<hiddenSize {
                XCTAssertEqual(
                  result[unchangedBatch, row, channel], batched[unchangedBatch, row, channel])
              }
            }
          }
        }
      }
    }
  }

  func testDenoiserUsesKeyframeWithoutReturningConditionRows() throws {
    DynamicGraph.setSeed(42)
    let watermark = DynamicGraph.queueWatermark
    DynamicGraph.queueWatermark = 1
    defer { DynamicGraph.queueWatermark = watermark }
    let graph = DynamicGraph()
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      "h3-denoiser-\(UUID().uuidString).ckpt"
    ).path
    defer { try? FileManager.default.removeItem(atPath: path) }
    try graph.withNoGrad {
      let video = graph.variable(.GPU(0), .NHWC(2, 4, 6, 24), of: Float16.self)
      let audio = graph.variable(.GPU(0), .HWC(1, 4, 32), of: Float16.self)
      let text = graph.variable(.GPU(0), .HWC(1, 8, 8), of: Float.self)
      let firstFrame = graph.variable(.GPU(0), .NHWC(1, 2, 3, 8), of: Float16.self)
      video.full(0.1)
      audio.full(0.2)
      text.full(0.3)
      firstFrame.full(0.4)
      let rotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6,
            referenceImages: [(4, 6, 8)])
        ).toGPU(0))
      let modulation = graph.variable(.GPU(0), .WC(1, 8), of: Float16.self)
      modulation.full(0.1)
      let model = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1, referenceImageSizes: [(4, 6)]
      ).1
      let inputs: [DynamicGraph.AnyTensor] =
        [audio, text, rotary, firstFrame] + Array(repeating: modulation, count: 28)
      let plainRotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6)
        ).toGPU(0))
      let plain = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1
      ).1
      _ = plain(
        inputs: video, [audio, text, plainRotary] + Array(repeating: modulation, count: 22))
      graph.openStore(path) { $0.write("dit", model: plain) }
      model.compile(inputs: [video] + inputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: model, strict: true)
      }
      let first = model(inputs: video, inputs).map { $0.as(of: Float16.self).rawValue.toCPU() }
      let withVision = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1, referenceImageSizes: [(4, 6)],
        visionLength: 1
      ).1
      let groupedRotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6,
            referenceImages: [(4, 6, 8)], visionTokenRanges: [6..<7])
        ).toGPU(0))
      let visionInputs: [DynamicGraph.AnyTensor] =
        [
          audio,
          Functional.concat(
            axis: 1, text[0..<1, 0..<6, 0..<8].copied(),
            text[0..<1, 7..<8, 0..<8].copied(),
            text[0..<1, 6..<7, 0..<8].copied()), groupedRotary, firstFrame,
        ] + Array(repeating: modulation, count: 28)
      withVision.compile(inputs: [video] + visionInputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: withVision, strict: true)
      }
      let split = withVision(inputs: video, visionInputs).map {
        $0.as(of: Float16.self).rawValue.toCPU()
      }
      for output in 0..<2 {
        let count = first[output].shape.reduce(1, *)
        let a = first[output].reshaped(.C(count))
        let b = split[output].reshaped(.C(count))
        for i in 0..<count { XCTAssertEqual(Float(a[i]), Float(b[i]), accuracy: 0.002) }
      }
      firstFrame.full(-0.4)
      let second = model(inputs: video, inputs).map { $0.as(of: Float16.self).rawValue.toCPU() }
      XCTAssertEqual(Array(first[0].shape), [2, 4, 6, 24])
      XCTAssertEqual(Array(first[1].shape), [1, 4, 32])
      var difference: Float = 0
      for index in 0..<2 {
        let count = first[index].shape.reduce(1, *)
        let a = first[index].reshaped(.C(count))
        let b = second[index].reshaped(.C(count))
        for j in 0..<count {
          XCTAssertTrue(a[j].isFinite && b[j].isFinite)
          difference += abs(Float(a[j]) - Float(b[j]))
        }
      }
      XCTAssertGreaterThan(difference, 0.001)
      let reference = graph.variable(.GPU(0), .NHWC(1, 4, 2, 8), of: Float16.self)
      reference.full(0.2)
      let multipleRotary = graph.variable(
        Tensor<Float16>(
          from: MiniMaxH3RotaryEmbedding(
            textLength: 8, audioLength: 4, videoFrames: 2, videoHeight: 4, videoWidth: 6,
            referenceImages: [(4, 6, 8), (8, 4, 9)], videoPosition: 10)
        ).toGPU(0))
      let multiple = MiniMaxH3(
        hiddenSize: 8, layers: 1, textLength: 8, audioLength: 4, videoFrames: 2,
        videoHeight: 4, videoWidth: 6, usesFlashAttention: .scale1,
        referenceImageSizes: [(4, 6), (8, 4)]
      ).1
      let multipleInputs: [DynamicGraph.AnyTensor] =
        [audio, text, multipleRotary, firstFrame, reference]
        + Array(repeating: modulation, count: 28)
      multiple.compile(inputs: [video] + multipleInputs)
      try graph.openStore(path, flags: .readOnly) {
        try $0.read("dit", model: multiple, strict: true)
      }
      let outputs = multiple(inputs: video, multipleInputs)
      XCTAssertEqual(Array(outputs[0].shape), [2, 4, 6, 24])
      XCTAssertEqual(Array(outputs[1].shape), [1, 4, 32])
      for output in outputs {
        let value = output.as(of: Float16.self).rawValue.toCPU()
        let count = value.shape.reduce(1, *)
        let flat = value.reshaped(.C(count))
        for i in 0..<count { XCTAssertTrue(flat[i].isFinite) }
      }
    }
  }
}
