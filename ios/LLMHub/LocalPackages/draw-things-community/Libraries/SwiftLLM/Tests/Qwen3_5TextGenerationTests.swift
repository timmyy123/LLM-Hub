import Foundation
import LLM
import NNC
import XCTest

final class Qwen3_5TextGenerationTests: XCTestCase {
  private let configuration = Qwen3_5ModelConfiguration(
    vocabularySize: 32, hiddenSize: 128, intermediateSize: 256, layers: 1,
    fullAttentionInterval: 1, attentionHeads: 2, keyValueHeads: 1,
    attentionHeadDim: 64, rotaryDim: 64, mropeSection: (11, 11, 10),
    ropeTheta: 10_000, linearNumKeyHeads: 1, linearNumValueHeads: 1,
    linearKeyHeadDim: 64, linearValueHeadDim: 64, linearConvKernel: 4)
  private var path: String!

  override func setUpWithError() throws {
    path =
      FileManager.default.temporaryDirectory.appendingPathComponent(
        "qwen-generation-\(UUID().uuidString).ckpt"
      ).path
    // A tiny randomly initialized checkpoint exercises the real Metal execution
    // and compilation paths without downloading model weights.
    let graph = DynamicGraph()
    try graph.withNoGrad {
      let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
      let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Float16.self)
      let tokens = graph.variable(Tensor<Int32>([1], .CPU, .C(1)).toGPU(0))
      let rotary = graph.variable(
        Qwen3_5RotaryEmbedding(
          sequenceLength: 1, configuration: configuration, of: Float16.self
        ).toGPU(0))
      let k = graph.variable(.GPU(0), .NHWC(1, 1, 1, 64), of: Float16.self)
      let v = graph.variable(like: k)
      k.full(0)
      v.full(0)
      let model = Qwen3_5CausalLM(
        Float16.self, tokenLength: 1, configuration: configuration)
      _ = model(inputs: tokens, emptyEmbeddings, emptyTokens, rotary, k, v)[0].as(of: Float16.self)
        .toCPU()
      graph.openStore(path) { $0.write("text_model", model: model) }

      // Metal's random initializer supports FP16. Convert the saved drafter
      // weights on CPU to match the runtime's BF16 checkpoint contract.
      let mtp = Qwen3_5MTP(
        Float16.self, Float16.self, configuration: configuration, batchSize: 1,
        tokenLength: 1, cachedTokenLength: 0, lastNumberOfTokens: 1, tieEmbedding: false)
      let hidden = graph.variable(.GPU(0), .NC(1, 128), of: Float16.self)
      let mtpK = graph.variable(.GPU(0), .NHWC(1, 1, 1, 64), of: Float16.self)
      let mtpV = graph.variable(like: mtpK)
      hidden.full(0)
      mtpK.full(0)
      mtpV.full(0)
      _ = mtp(inputs: tokens, emptyEmbeddings, emptyTokens, hidden, rotary, mtpK, mtpV)[1].as(
        of: Float16.self
      ).toCPU()
      graph.openStore(path) { $0.write("text_model", model: mtp) }
      try graph.openStore(path) { store in
        let keys = store.keys.filter { $0.contains("mtp.") }
        XCTAssertFalse(keys.isEmpty)
        for key in keys {
          let tensor = try XCTUnwrap(store.read(key, kind: .CPU))
          try store.write(key, tensor: Tensor<BFloat16>(from: tensor), strict: true)
        }
      }
    }
  }

  override func tearDownWithError() throws {
    try FileManager.default.removeItem(atPath: path)
  }

  private var generator: Qwen3_5TextGeneration<Float16> {
    Qwen3_5TextGeneration(
      filePath: path, configuration: configuration, eosTokenIds: [], tieEmbedding: false)
  }

  func testSingleTokenPrefillAndDecode() throws {
    for prompt: [Int32] in [[1], [1, 2]] {
      var timing: Qwen3_5GenerationTiming?
      let tokens = try generator.generate(
        graph: DynamicGraph(), promptTokenIds: prompt, maxTokens: 4,
        partialHandler: { _ in true }, timingHandler: { timing = $0 })
      XCTAssertEqual(tokens.count, 4)
      XCTAssertTrue(tokens.allSatisfy { (0..<32).contains($0) })
      XCTAssertEqual(timing?.decodeLoopTokens, 3)
    }
  }

  func testGenerationStopsAtPartialHandlerBoundary() throws {
    let tokens = try generator.generate(
      graph: DynamicGraph(), promptTokenIds: [1], maxTokens: 8,
      partialHandler: { $0.count < 2 })
    XCTAssertEqual(tokens.count, 2)
  }

  func testMTPGenerationAndDraftAcceptance() throws {
    let result = try generator.generateWithMTPDrafting(
      graph: DynamicGraph(), promptTokenIds: [1], maxTokens: 4)
    XCTAssertEqual(result.generatedTokenIds.count, 4)
    XCTAssertTrue(result.generatedTokenIds.allSatisfy { (0..<32).contains($0) })
    let acceptance = try generator.measureMTPDraftAcceptance(
      graph: DynamicGraph(), promptTokenIds: [1], startCount: 2, maxDraftTokens: 2)
    XCTAssertEqual(acceptance.startCount, 2)
    XCTAssertEqual(acceptance.acceptedLengthCounts.reduce(0, +), 2)
    XCTAssertTrue(acceptance.acceptedPrefixRates.allSatisfy { (0...1).contains($0) })
  }
  func testThreeInputCompositionMatchesTokenLookup() throws {
    let graph = DynamicGraph()
    try graph.withNoGrad {
      let ids: [Int32] = [1, 2, 3, 4, 5]
      let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
      let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Float16.self)
      let rotary = graph.variable(
        Qwen3_5RotaryEmbedding(
          sequenceLength: ids.count, configuration: configuration, of: Float16.self
        ).toGPU(0))
      let embeddingWeights = try graph.openStore(path, flags: .readOnly) {
        store -> Tensor<Float16> in
        let key = try XCTUnwrap(store.keys.first { $0.contains("language_model.embed_tokens") })
        return Tensor<Float16>(from: try XCTUnwrap(store.read(key, kind: .CPU)))
      }.get()
      // Reuse each builder while moving the injected region through all boundary shapes.
      let model = ModelBuilder<Int> { length, _ in
        Qwen3_5CausalLM(
          Float16.self, tokenLength: length,
          configuration: self.configuration, lastNumberOfTokens: length)
      }
      let draft = ModelBuilder<Int> { length, _ in
        Qwen3_5MTP(
          BFloat16.self, Float16.self, configuration: self.configuration,
          batchSize: 1, tokenLength: length, cachedTokenLength: 0,
          lastNumberOfTokens: length, tieEmbedding: false)
      }
      let hidden = graph.variable(
        .GPU(0), .NC(ids.count, configuration.hiddenSize), of: BFloat16.self)
      hidden.full(0)
      var targetReference: Tensor<Float16>?
      var draftReference: Tensor<Float16>?
      for range in [0..<0, 1..<4, 0..<3, 2..<5, 0..<5, 0..<0] {
        func tokenInput(_ values: ArraySlice<Int32>) -> DynamicGraph.Tensor<Int32> {
          values.isEmpty
            ? emptyTokens
            : graph.variable(
              Tensor<Int32>(Array(values), kind: .CPU, format: .NHWC, shape: [values.count]).toGPU(
                0))
        }
        let pre = tokenInput(ids[..<range.lowerBound])
        let post = tokenInput(ids[range.upperBound...])
        let injected: DynamicGraph.Tensor<Float16>
        if range.isEmpty {
          injected = emptyEmbeddings
        } else {
          var rows = Tensor<Float16>(.CPU, .WC(range.count, configuration.hiddenSize))
          for (row, index) in range.enumerated() {
            let token = Int(ids[index])
            rows[row..<(row + 1), 0..<configuration.hiddenSize] =
              embeddingWeights[token..<(token + 1), 0..<configuration.hiddenSize]
          }
          injected = graph.variable(rows.toGPU(0))
        }
        let k = graph.variable(.GPU(0), .NHWC(1, ids.count, 1, 64), of: Float16.self)
        let v = graph.variable(like: k)
        k.full(0)
        v.full(0)
        let dk = graph.variable(.GPU(0), .NHWC(1, ids.count, 1, 64), of: BFloat16.self)
        let dv = graph.variable(like: dk)
        dk.full(0)
        dv.full(0)
        let targetInputs: [DynamicGraph.AnyTensor] = [pre, injected, post, rotary, k, v]
        let draftInputs: [DynamicGraph.AnyTensor] = [pre, injected, post, hidden, rotary, dk, dv]
        if targetReference == nil {
          model.compile(ids.count, inputs: targetInputs)
          draft.compile(ids.count, inputs: draftInputs)
          try graph.openStore(path, flags: .readOnly) { store in
            try store.read("text_model", model: model, strict: true)
            try store.read("text_model", model: draft, strict: true)
          }.get()
        }
        let actual = model(ids.count, inputs: targetInputs[0], Array(targetInputs.dropFirst()))[0]
          .as(of: Float16.self).toCPU().rawValue
        let actualDraft = draft(ids.count, inputs: draftInputs[0], Array(draftInputs.dropFirst()))[
          1
        ]
        .as(of: Float16.self).toCPU().rawValue
        if let expected = targetReference, let expectedDraft = draftReference {
          for row in 0..<ids.count {
            for column in 0..<configuration.vocabularySize {
              XCTAssertEqual(
                Float(actual[row, column]), Float(expected[row, column]), accuracy: 0.002)
              XCTAssertEqual(
                Float(actualDraft[row, column]), Float(expectedDraft[row, column]), accuracy: 0.02)
            }
          }
        } else {
          targetReference = actual
          draftReference = actualDraft
        }
      }
    }
  }
  func testMixedImageChunksPreserveLinearState() throws {
    var config = configuration
    config.layers = 2
    config.fullAttentionInterval = 2
    let graph = DynamicGraph()
    try graph.withNoGrad {
      let ids: [Int32] = Array(1...12)
      let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
      let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Float16.self)
      func caches() -> [DynamicGraph.AnyTensor] {
        let conv = graph.variable(.GPU(0), .NHWC(1, 3, 1, config.linearConvDim), of: Float16.self)
        let recurrent = graph.variable(.GPU(0), .NHWC(1, 1, 64, 64), of: Float.self)
        let k = graph.variable(.GPU(0), .NHWC(1, ids.count, 1, 64), of: Float16.self)
        let v = graph.variable(like: k)
        conv.full(0)
        recurrent.full(0)
        k.full(0)
        v.full(0)
        return [conv, recurrent, k, v]
      }
      let baseline = Qwen3_5CausalLM(
        Float16.self, tokenLength: ids.count,
        configuration: config, outputCacheStates: true)
      let fullTokens = graph.variable(
        Tensor<Int32>(ids, kind: .CPU, format: .NHWC, shape: [ids.count]).toGPU(0))
      let rotary = graph.variable(
        Qwen3_5RotaryEmbedding(
          sequenceLength: ids.count,
          configuration: config, of: Float16.self
        ).toGPU(0))
      let full = baseline(inputs: fullTokens, [emptyEmbeddings, emptyTokens, rotary] + caches())
      let reference = full[0].as(of: Float16.self).toCPU().rawValue
      let referenceState = full[2].as(of: Float.self).toCPU().rawValue
      graph.openStore(path) { $0.write("hybrid", model: baseline) }
      let weights = try graph.openStore(path, flags: .readOnly) { store -> Tensor<Float16> in
        let key = try XCTUnwrap(
          store.keys.first { $0.hasPrefix("__hybrid__") && $0.contains("embed_tokens") })
        return Tensor<Float16>(from: try XCTUnwrap(store.read(key, kind: .CPU)))
      }.get()
      let builder = ModelBuilder<(Int, Int)> { sizes, _ in
        Qwen3_5CausalLM(
          Float16.self, tokenLength: sizes.0, cachedTokenLength: sizes.1,
          configuration: config, outputCacheStates: true)
      }
      var cache = caches()
      var output: [DynamicGraph.AnyTensor] = []
      var start = 0
      // Start with a tiny mixed chunk and cross both image boundaries while reusing the builder.
      for end in [4, 7, 10, 12] {
        let image = (start < 7 ? 2..<6 : 8..<11).clamped(to: start..<end)
        func tokens(_ range: Range<Int>) -> DynamicGraph.Tensor<Int32> {
          range.isEmpty
            ? emptyTokens
            : graph.variable(
              Tensor<Int32>(
                Array(ids[range]),
                kind: .CPU, format: .NHWC, shape: [range.count]
              ).toGPU(0))
        }
        var rows = Tensor<Float16>(.CPU, .WC(image.count, config.hiddenSize))
        for (row, position) in image.enumerated() {
          let index = Int(ids[position])
          rows[row..<(row + 1), 0..<config.hiddenSize] =
            weights[index..<(index + 1), 0..<config.hiddenSize]
        }
        let imageEmbedding = graph.variable(rows.toGPU(0))
        let positions = graph.variable(
          Qwen3_5RotaryEmbedding(
            sequenceLength: end - start,
            cachedTokenLength: start, configuration: config, of: Float16.self
          ).toGPU(0))
        let slicedCache: [DynamicGraph.AnyTensor] =
          [cache[0], cache[1]]
          + cache[2...].map {
            $0.as(of: Float16.self).reshaped(
              .NHWC(1, end, 1, 64),
              offset: [0, 0, 0, 0], strides: [ids.count * 64, 64, 64, 1])
          }
        let inputs: [DynamicGraph.AnyTensor] =
          [
            tokens(start..<image.lowerBound), imageEmbedding,
            tokens(image.upperBound..<end), positions,
          ] + slicedCache
        if start == 0 {
          builder.compile((end - start, start), inputs: inputs)
          try graph.openStore(path, flags: .readOnly) {
            try $0.read("hybrid", model: builder, strict: true)
          }.get()
        }
        output = builder((end - start, start), inputs: inputs[0], Array(inputs.dropFirst()))
        cache[0] = output[1].as(of: Float16.self).copied()
        cache[1] = output[2].as(of: Float.self).copied()
        start = end
      }
      let actual = output[0].as(of: Float16.self).toCPU().rawValue
      let actualState = output[2].as(of: Float.self).toCPU().rawValue
      var logitError: Float = 0
      var stateError: Float = 0
      for column in 0..<config.vocabularySize {
        logitError = max(logitError, abs(Float(actual[0, column]) - Float(reference[0, column])))
      }
      for row in 0..<64 {
        for column in 0..<64 {
          stateError = max(
            stateError, abs(actualState[0, 0, row, column] - referenceState[0, 0, row, column]))
        }
      }
      XCTAssertLessThan(logitError, 0.03)
      XCTAssertLessThan(stateError, 0.03)
    }
  }
}
