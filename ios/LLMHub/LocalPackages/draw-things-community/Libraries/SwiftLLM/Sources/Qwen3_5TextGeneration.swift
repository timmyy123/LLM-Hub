import Foundation
import NNC

public enum Qwen3_5TextGenerationError: LocalizedError {
  case cancelled
  case missingStoreTensor(String)
  case invalidLogits

  public var errorDescription: String? {
    switch self {
    case .cancelled:
      return "Qwen3.5 generation was cancelled."
    case .missingStoreTensor(let name):
      return "Missing tensor '\(name)' in Qwen3.5 store."
    case .invalidLogits:
      return "Qwen3.5 produced NaN logits."
    }
  }
}

public struct Qwen3_5GenerationTiming: Sendable {
  public var loadAndCompileMilliseconds: Double
  public var prefillMilliseconds: Double
  public var decodeCompileMilliseconds: Double
  public var decodeLoopMilliseconds: Double
  public var decodeLoopTokens: Int

  public var decodeLoopTokensPerSecond: Double {
    decodeLoopMilliseconds > 0
      ? Double(decodeLoopTokens) / (decodeLoopMilliseconds / 1_000) : 0
  }
}

public protocol Qwen3_5TextGenerationPrefixCache {
  var tokenIds: [Int32] { get }

  func restore(
    graph: DynamicGraph, configuration: Qwen3_5ModelConfiguration,
    caches: inout [DynamicGraph.AnyTensor]
  ) -> Bool

  func save(
    graph: DynamicGraph, configuration: Qwen3_5ModelConfiguration,
    caches: [DynamicGraph.AnyTensor]
  )
}

public struct Qwen3_5MTPDraftAcceptanceResult: Sendable {
  public var promptTokenCount: Int
  public var startCount: Int
  public var maxDraftTokens: Int
  public var acceptedPrefixCounts: [Int]
  public var acceptedLengthCounts: [Int]

  public var averageAcceptedDraftTokens: Double {
    guard startCount > 0 else { return 0 }
    return Double(
      acceptedLengthCounts.enumerated().reduce(0) { $0 + $1.offset * $1.element }
    ) / Double(startCount)
  }

  public var acceptedPrefixRates: [Double] {
    guard startCount > 0 else {
      return Array(repeating: 0, count: acceptedPrefixCounts.count)
    }
    return acceptedPrefixCounts.map { Double($0) / Double(startCount) }
  }

}

public struct Qwen3_5MTPGenerationResult: Sendable {
  public var generatedTokenIds: [Int32]
  public var timing: Qwen3_5GenerationTiming
  public var acceptedDrafts: Int
  public var rejectedDrafts: Int
  public var d1AcceptedDrafts: Int
  public var d1RejectedDrafts: Int
  public var d2AcceptedDrafts: Int
  public var d2RejectedDrafts: Int
  public var replayRounds: Int

  public var d1AcceptanceRate: Double {
    let total = d1AcceptedDrafts + d1RejectedDrafts
    return total > 0 ? Double(d1AcceptedDrafts) / Double(total) : 0
  }

  public var d2AcceptanceRate: Double {
    let total = d2AcceptedDrafts + d2RejectedDrafts
    return total > 0 ? Double(d2AcceptedDrafts) / Double(total) : 0
  }
}

public struct Qwen3_5TextGeneration<FloatType: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  public let configuration: Qwen3_5ModelConfiguration
  public let eosTokenIds: Set<Int32>
  public let tieEmbedding: Bool

  public init(
    filePath: String,
    configuration: Qwen3_5ModelConfiguration,
    eosTokenIds: Set<Int32> = [248_046, 248_044], tieEmbedding: Bool
  ) {
    self.filePath = filePath
    self.configuration = configuration
    self.eosTokenIds = eosTokenIds
    self.tieEmbedding = tieEmbedding
  }

  public func generate(
    graph: DynamicGraph, promptTokenIds: [Int32], maxTokens: Int,
    prefillChunkSize: Int = 4_096,
    prefixCache: (any Qwen3_5TextGenerationPrefixCache)? = nil,
    shouldContinue: () -> Bool = { true },
    partialHandler: ([Int32]) -> Bool,
    timingHandler: ((Qwen3_5GenerationTiming) -> Void)? = nil
  ) throws -> [Int32] {
    precondition(!promptTokenIds.isEmpty)
    precondition(prefillChunkSize > 0)
    guard maxTokens > 0 else { return [] }
    let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
    let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: FloatType.self)
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad {
        guard shouldContinue() else { throw Qwen3_5TextGenerationError.cancelled }
        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let cacheCapacity = promptTokenIds.count + maxTokens + 1
        var caches = Self.makeCaches(
          graph: graph, capacity: cacheCapacity, configuration: configuration)
        let validPrefixCache: (any Qwen3_5TextGenerationPrefixCache)?
        if let prefixCache = prefixCache, !prefixCache.tokenIds.isEmpty,
          prefixCache.tokenIds.count < promptTokenIds.count,
          promptTokenIds.starts(with: prefixCache.tokenIds)
        {
          validPrefixCache = prefixCache
        } else {
          validPrefixCache = nil
        }
        let restoredPrefix =
          validPrefixCache?.restore(
            graph: graph, configuration: configuration, caches: &caches) == true
        let prefillTokenStart = restoredPrefix ? validPrefixCache?.tokenIds.count ?? 0 : 0

        let decoder:
          ModelBuilder<(cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int)> =
            ModelBuilder {
              (
                tokenLengths: (
                  cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int
                ), _
              ) in
              Qwen3_5CausalLM(
                FloatType.self, tokenLength: tokenLengths.tokenLength,
                cachedTokenLength: tokenLengths.cachedTokenLength, configuration: configuration,
                includeLogits: true, outputCacheStates: true, tieEmbedding: tieEmbedding,
                lastNumberOfTokens: tokenLengths.lastNumberOfTokens)
            }
        decoder.maxConcurrency = .limit(4)
        let promptTokens = graph.variable(
          Tensor<Int32>(promptTokenIds, .CPU, .C(promptTokenIds.count)).toGPU(0))
        let prefillRotaryGPU: DynamicGraph.Tensor<FloatType>?
        if hasFullAttentionLayer {
          let rotary = Qwen3_5RotaryEmbedding(
            sequenceLength: promptTokenIds.count, configuration: configuration, of: FloatType.self)
          prefillRotaryGPU = graph.variable(rotary.toGPU(0))
        } else {
          prefillRotaryGPU = nil
        }
        let maxDecodeCachedTokenLength = cacheCapacity - 1
        let decodeCompileAttentionInputs: [DynamicGraph.AnyTensor]
        if maxTokens > 1 && hasFullAttentionLayer {
          let decodeCompileRotary = Qwen3_5RotaryEmbedding(
            sequenceLength: 1, cachedTokenLength: maxDecodeCachedTokenLength,
            configuration: configuration, of: FloatType.self)
          decodeCompileAttentionInputs = [graph.variable(decodeCompileRotary.toGPU(0))]
        } else {
          decodeCompileAttentionInputs = []
        }
        var loadAndCompileMilliseconds = 0.0
        var prefillMilliseconds = 0.0
        do {
          let firstChunkLength = min(prefillChunkSize, promptTokenIds.count - prefillTokenStart)
          let firstPrefillTokens = promptTokens.reshaped(
            .C(firstChunkLength), offset: [prefillTokenStart], strides: [1])
          var firstPrefillAttentionInputs = [DynamicGraph.AnyTensor]()
          if let prefillRotaryGPU = prefillRotaryGPU {
            firstPrefillAttentionInputs.append(
              prefillRotaryGPU.reshaped(
                .NHWC(1, firstChunkLength, 1, configuration.attentionHeadDim),
                offset: [0, prefillTokenStart, 0, 0],
                strides: [
                  promptTokenIds.count * configuration.attentionHeadDim,
                  configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
                ]))
          }
          let firstPrefillCacheInputs = Self.cacheInputs(
            caches, currentTokenLength: prefillTokenStart + firstChunkLength,
            configuration: configuration)
          let inputs: [DynamicGraph.AnyTensor] =
            [firstPrefillTokens, emptyEmbeddings, emptyTokens] + firstPrefillAttentionInputs
            + firstPrefillCacheInputs
          let loadStart = Date.timeIntervalSinceReferenceDate
          guard shouldContinue() else { throw Qwen3_5TextGenerationError.cancelled }
          try graph.openStore(
            filePath, flags: .readOnly,
            externalStore: TensorData.externalStore(filePath: filePath)
          ) { store in
            decoder.compile(
              (
                cachedTokenLength: prefillTokenStart, tokenLength: firstChunkLength,
                lastNumberOfTokens: 1
              ),
              inputs: inputs)
            try store.read(
              "text_model", model: decoder, strict: true,
              codec: [.jit, .i8x, .ezm7, .externalData])
            // Establish execution before prefill, after loading quantized weights.
            // One-token prefill can reuse this graph during eager decode compilation.
            decoder.compile(
              (
                cachedTokenLength: prefillTokenStart, tokenLength: firstChunkLength,
                lastNumberOfTokens: 1
              ), inputs: inputs, isEager: true)
          }
          loadAndCompileMilliseconds = (Date.timeIntervalSinceReferenceDate - loadStart) * 1_000
          let prefillStart = Date.timeIntervalSinceReferenceDate
          var prefillTokenCPU: DynamicGraph.Tensor<Int32>?
          for start in stride(
            from: prefillTokenStart, to: promptTokenIds.count, by: prefillChunkSize)
          {
            guard shouldContinue() else { throw Qwen3_5TextGenerationError.cancelled }
            let length = min(prefillChunkSize, promptTokenIds.count - start)
            let isLast = start + length == promptTokenIds.count
            let chunkTokens = promptTokens.reshaped(.C(length), offset: [start], strides: [1])
            var chunkAttentionInputs = [DynamicGraph.AnyTensor]()
            if let prefillRotaryGPU = prefillRotaryGPU {
              chunkAttentionInputs.append(
                prefillRotaryGPU.reshaped(
                  .NHWC(1, length, 1, configuration.attentionHeadDim),
                  offset: [0, start, 0, 0],
                  strides: [
                    promptTokenIds.count * configuration.attentionHeadDim,
                    configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
                  ]
                ).copied())
            }
            let chunkCacheInputs = Self.cacheInputs(
              caches, currentTokenLength: start + length, configuration: configuration)
            var prefillOutputs = decoder(
              (
                cachedTokenLength: start, tokenLength: length,
                lastNumberOfTokens: isLast ? 1 : 0
              ), inputs: chunkTokens,
              [emptyEmbeddings, emptyTokens] + chunkAttentionInputs + chunkCacheInputs)
            Self.updateLinearCaches(
              &caches, from: prefillOutputs, outputOffset: 1, configuration: configuration)
            if !restoredPrefix, let validPrefixCache = validPrefixCache,
              start + length == validPrefixCache.tokenIds.count
            {
              validPrefixCache.save(
                graph: graph, configuration: configuration, caches: caches)
            }
            if isLast {
              let logits = prefillOutputs[0].as(of: FloatType.self)
              // On macOS 26.4.1, native FP16/BF16 MPSGraph argmax was unreliable on Qwen vocab-sized logits.
              prefillTokenCPU = Functional.argmax(
                DynamicGraph.Tensor<Float>(from: logits), axis: 1
              ).reshaped(.C(1)).toCPU()
            }
            prefillOutputs.removeAll(keepingCapacity: false)
          }
          graph.joined()
          prefillMilliseconds = (Date.timeIntervalSinceReferenceDate - prefillStart) * 1_000
          let prefillToken = Int32(prefillTokenCPU![0])
          var generated = [prefillToken]
          let shouldContinueAfterPrefill = partialHandler(generated)
          if eosTokenIds.contains(prefillToken) || !shouldContinueAfterPrefill {
            timingHandler?(
              Qwen3_5GenerationTiming(
                loadAndCompileMilliseconds: loadAndCompileMilliseconds,
                prefillMilliseconds: prefillMilliseconds,
                decodeCompileMilliseconds: 0,
                decodeLoopMilliseconds: 0,
                decodeLoopTokens: 0))
            return generated
          }
          let decodeCompileToken = graph.variable(
            Tensor<Int32>([prefillToken], .CPU, .C(1)).toGPU(0))
          var nextTokenGPU = graph.variable(Tensor<Int32>([prefillToken], .CPU, .C(1)).toGPU(0))
          var decodeCompileMilliseconds = 0.0
          var decodeLoopMilliseconds = 0.0
          var decodeLoopTokens = 0
          if maxTokens > 1 {
            let decodeCompileCacheInputs = Self.cacheInputs(
              caches, currentTokenLength: cacheCapacity, configuration: configuration)
            let decodeCompileInputs: [DynamicGraph.AnyTensor] =
              [decodeCompileToken, emptyEmbeddings, emptyTokens] + decodeCompileAttentionInputs
              + decodeCompileCacheInputs
            let decodeCompileStart = Date.timeIntervalSinceReferenceDate
            decoder.compile(
              (
                cachedTokenLength: maxDecodeCachedTokenLength, tokenLength: 1,
                lastNumberOfTokens: 1
              ),
              inputs: decodeCompileInputs, isEager: true)
            graph.joined()
            decodeCompileMilliseconds =
              (Date.timeIntervalSinceReferenceDate - decodeCompileStart) * 1_000
            let queueWatermark = DynamicGraph.queueWatermark
            DynamicGraph.queueWatermark = queueWatermark * 16
            defer { DynamicGraph.queueWatermark = queueWatermark }
            let decodeLoopStart = Date.timeIntervalSinceReferenceDate
            let cachedTokenLength = promptTokenIds.count + generated.count - 1
            let rotaryEmbeddingForMaxTokens: DynamicGraph.Tensor<FloatType>?
            if hasFullAttentionLayer {
              rotaryEmbeddingForMaxTokens = graph.variable(
                Qwen3_5RotaryEmbedding(
                  sequenceLength: maxTokens, cachedTokenLength: cachedTokenLength,
                  configuration: configuration, of: FloatType.self
                ).toGPU(0))
            } else {
              rotaryEmbeddingForMaxTokens = nil
            }
            var shouldAppendPendingToken = true
            for decodeIndex in 0..<(maxTokens - 1) {
              guard shouldContinue() else { throw Qwen3_5TextGenerationError.cancelled }
              let cachedTokenLength = promptTokenIds.count + decodeIndex
              var oneAttentionInputs = [DynamicGraph.AnyTensor]()
              if let rotaryEmbeddingForMaxTokens = rotaryEmbeddingForMaxTokens {
                let oneRotary = rotaryEmbeddingForMaxTokens.reshaped(
                  format: .NHWC, shape: [1, 1, 1, configuration.attentionHeadDim],
                  offset: [0, 0, decodeIndex, 0],
                  strides: [
                    configuration.attentionHeadDim, configuration.attentionHeadDim,
                    configuration.attentionHeadDim, 1,
                  ])
                oneAttentionInputs = [oneRotary.copied()]
              }
              let cacheInputs = Self.cacheInputs(
                caches, currentTokenLength: cachedTokenLength + 1, configuration: configuration)
              let oldDecodedTokenCPU = nextTokenGPU.toCPU()
              do {
                let decodeOutputs = decoder(
                  (
                    cachedTokenLength: cachedTokenLength, tokenLength: 1,
                    lastNumberOfTokens: 1
                  ), inputs: nextTokenGPU,
                  [emptyEmbeddings, emptyTokens] + oneAttentionInputs + cacheInputs)
                Self.updateLinearCaches(
                  &caches, from: decodeOutputs, outputOffset: 1, configuration: configuration)
                let logits = decodeOutputs[0].as(of: FloatType.self)
                nextTokenGPU = Functional.argmax(logits, axis: 1).reshaped(.C(1))
              }
              decodeLoopTokens += 1
              if decodeIndex > 0 {
                let decodedToken = oldDecodedTokenCPU[0]
                generated.append(decodedToken)
                if eosTokenIds.contains(decodedToken) {
                  shouldAppendPendingToken = false
                  break
                }
                if !partialHandler(generated) {
                  shouldAppendPendingToken = false
                  break
                }
              }
            }
            if shouldAppendPendingToken && generated.count < maxTokens {
              let finalTokenCPU = nextTokenGPU.toCPU()
              graph.joined()
              let decodedToken = finalTokenCPU[0]
              generated.append(decodedToken)
            } else {
              graph.joined()
            }
            decodeLoopMilliseconds =
              (Date.timeIntervalSinceReferenceDate - decodeLoopStart) * 1_000
          }
          timingHandler?(
            Qwen3_5GenerationTiming(
              loadAndCompileMilliseconds: loadAndCompileMilliseconds,
              prefillMilliseconds: prefillMilliseconds,
              decodeCompileMilliseconds: decodeCompileMilliseconds,
              decodeLoopMilliseconds: decodeLoopMilliseconds,
              decodeLoopTokens: decodeLoopTokens))
          _ = partialHandler(generated)
          return generated
        }
      }
    }
  }

  public func generateWithMTPDrafting(
    graph: DynamicGraph, promptTokenIds: [Int32], maxTokens: Int
  ) throws -> Qwen3_5MTPGenerationResult {
    precondition(!promptTokenIds.isEmpty)
    guard maxTokens > 0 else {
      return Qwen3_5MTPGenerationResult(
        generatedTokenIds: [],
        timing: Qwen3_5GenerationTiming(
          loadAndCompileMilliseconds: 0, prefillMilliseconds: 0,
          decodeCompileMilliseconds: 0, decodeLoopMilliseconds: 0, decodeLoopTokens: 0),
        acceptedDrafts: 0, rejectedDrafts: 0, d1AcceptedDrafts: 0, d1RejectedDrafts: 0,
        d2AcceptedDrafts: 0, d2RejectedDrafts: 0, replayRounds: 0)
    }
    let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
    let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: FloatType.self)
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad {
        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let cacheCapacity = promptTokenIds.count + maxTokens + 2
        let finalHiddenOutputIndex = 0
        let logitsOutputIndex = 1
        let cacheOutputOffset = 2
        var cacheBanks = [
          Self.makeCaches(graph: graph, capacity: cacheCapacity, configuration: configuration)
        ]
        for _ in 0..<2 {
          var bank = [DynamicGraph.AnyTensor]()
          var cursor = 0
          for layerIndex in 0..<configuration.layers {
            if configuration.isLinearAttentionLayer(layerIndex) {
              let convState = graph.variable(
                Tensor<FloatType>(
                  Array(
                    repeating: FloatType.zero,
                    count: (configuration.linearConvKernel - 1) * configuration.linearConvDim),
                  .CPU,
                  .NHWC(
                    1, configuration.linearConvKernel - 1, 1, configuration.linearConvDim)
                ).toGPU(0))
              let recurrentState = graph.variable(
                Tensor<Float>(
                  Array(
                    repeating: Float.zero,
                    count: configuration.linearNumValueHeads * configuration.linearKeyHeadDim
                      * configuration.linearValueHeadDim),
                  .CPU,
                  .NHWC(
                    1, configuration.linearNumValueHeads, configuration.linearValueHeadDim,
                    configuration.linearKeyHeadDim)
                ).toGPU(0))
              bank.append(contentsOf: [convState, recurrentState])
            } else {
              bank.append(cacheBanks[0][cursor])
              bank.append(cacheBanks[0][cursor + 1])
            }
            cursor += 2
          }
          cacheBanks.append(bank)
        }
        let decoder:
          ModelBuilder<
            (
              cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int,
              linearStateCheckpointCount: Int, includeLogits: Bool
            )
          > =
            ModelBuilder {
              (
                tokenLengths: (
                  cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int,
                  linearStateCheckpointCount: Int, includeLogits: Bool
                ), _
              ) in
              Qwen3_5CausalLM(
                FloatType.self, tokenLength: tokenLengths.tokenLength,
                cachedTokenLength: tokenLengths.cachedTokenLength, configuration: configuration,
                includeLogits: tokenLengths.includeLogits, outputCacheStates: true,
                outputFinalHiddenState: true, tieEmbedding: tieEmbedding,
                lastNumberOfTokens: tokenLengths.lastNumberOfTokens,
                linearStateCheckpointCount: tokenLengths.linearStateCheckpointCount)
            }
        decoder.maxConcurrency = .limit(4)
        let promptTokens = graph.variable(
          Tensor<Int32>(promptTokenIds, .CPU, .C(promptTokenIds.count)).toGPU(0))
        let prefillAttentionInputs: [DynamicGraph.AnyTensor]
        if hasFullAttentionLayer {
          prefillAttentionInputs = [
            graph.variable(
              Qwen3_5RotaryEmbedding(
                sequenceLength: promptTokenIds.count, configuration: configuration,
                of: FloatType.self
              ).toGPU(0))
          ]
        } else {
          prefillAttentionInputs = []
        }
        let prefillCacheInputs = Self.cacheInputs(
          cacheBanks[0], currentTokenLength: promptTokenIds.count, configuration: configuration)
        let loadStart = Date.timeIntervalSinceReferenceDate
        try graph.openStore(
          filePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          decoder.compile(
            (
              cachedTokenLength: 0, tokenLength: promptTokenIds.count,
              lastNumberOfTokens: 1, linearStateCheckpointCount: 0, includeLogits: true
            ),
            inputs: [promptTokens, emptyEmbeddings, emptyTokens] + prefillAttentionInputs
              + prefillCacheInputs)
          try store.read(
            "text_model", model: decoder, strict: true,
            codec: [.jit, .i8x, .ezm7, .externalData])
          decoder.compile(
            (
              cachedTokenLength: 0, tokenLength: promptTokenIds.count,
              lastNumberOfTokens: 1, linearStateCheckpointCount: 0, includeLogits: true
            ),
            inputs: [promptTokens, emptyEmbeddings, emptyTokens] + prefillAttentionInputs
              + prefillCacheInputs, isEager: true)
        }

        let decodeTokenLength = 2
        let mtpCacheCapacity = cacheCapacity
        let mtpCacheElementCount =
          mtpCacheCapacity * configuration.keyValueHeads * configuration.attentionHeadDim
        let mtpK = graph.variable(
          Tensor<BFloat16>(
            Array(repeating: BFloat16.zero, count: mtpCacheElementCount),
            .CPU,
            .NHWC(
              1, mtpCacheCapacity, configuration.keyValueHeads,
              configuration.attentionHeadDim)
          ).toGPU(0))
        let mtpV = graph.variable(
          Tensor<BFloat16>(
            Array(repeating: BFloat16.zero, count: mtpCacheElementCount),
            .CPU,
            .NHWC(
              1, mtpCacheCapacity, configuration.keyValueHeads,
              configuration.attentionHeadDim)
          ).toGPU(0))
        let mtpCacheRowStride = configuration.keyValueHeads * configuration.attentionHeadDim
        let mtpStep:
          ModelBuilder<(cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int)> =
            ModelBuilder { tokenLengths, _ in
              Qwen3_5MTP(
                BFloat16.self, FloatType.self, configuration: configuration, batchSize: 1,
                tokenLength: tokenLengths.tokenLength,
                cachedTokenLength: tokenLengths.cachedTokenLength,
                lastNumberOfTokens: tokenLengths.lastNumberOfTokens,
                tieEmbedding: tieEmbedding)
            }
        mtpStep.maxConcurrency = .limit(4)
        let mtpPrefillCompileHidden = graph.variable(
          Tensor<BFloat16>(
            Array(
              repeating: BFloat16.zero, count: promptTokenIds.count * configuration.hiddenSize),
            .CPU, .NC(promptTokenIds.count, configuration.hiddenSize)
          ).toGPU(0))
        let mtpPrefillRotary = graph.variable(
          Qwen3_5RotaryEmbedding(
            sequenceLength: promptTokenIds.count, configuration: configuration, of: FloatType.self
          ).toGPU(0))
        let mtpPrefillInputs: [DynamicGraph.AnyTensor] = [
          promptTokens, emptyEmbeddings, emptyTokens, mtpPrefillCompileHidden, mtpPrefillRotary,
          mtpK.reshaped(
            .NHWC(
              1, promptTokenIds.count, configuration.keyValueHeads,
              configuration.attentionHeadDim),
            offset: [0, 0, 0, 0],
            strides: [
              mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
              configuration.attentionHeadDim, 1,
            ]),
          mtpV.reshaped(
            .NHWC(
              1, promptTokenIds.count, configuration.keyValueHeads,
              configuration.attentionHeadDim),
            offset: [0, 0, 0, 0],
            strides: [
              mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
              configuration.attentionHeadDim, 1,
            ]),
        ]
        mtpStep.compile(
          (cachedTokenLength: 0, tokenLength: promptTokenIds.count, lastNumberOfTokens: 1),
          inputs: mtpPrefillInputs)
        graph.openStore(
          filePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          store.read(
            "text_model", model: mtpStep, codec: [.jit, .i8x, .ezm7, .externalData]
          ) { name, _, _, _ in
            if name.contains("lm_head") || name.contains("model.language_model.embed_tokens") {
              return .fail
            }
            return .continue(name)
          }
          mtpStep.parameters.share(from: decoder.parameters) { name, _ in
            .continue(name)
          }
        }
        mtpStep.compile(
          (cachedTokenLength: 0, tokenLength: promptTokenIds.count, lastNumberOfTokens: 1),
          inputs: mtpPrefillInputs, isEager: true)
        let loadAndCompileMilliseconds =
          (Date.timeIntervalSinceReferenceDate - loadStart) * 1_000

        let prefillStart = Date.timeIntervalSinceReferenceDate
        let prefillOutputs = decoder(
          (
            cachedTokenLength: 0, tokenLength: promptTokenIds.count, lastNumberOfTokens: 1,
            linearStateCheckpointCount: 0, includeLogits: true
          ),
          inputs: promptTokens,
          [emptyEmbeddings, emptyTokens] + prefillAttentionInputs + prefillCacheInputs)
        for bankIndex in cacheBanks.indices {
          Self.updateLinearCaches(
            &cacheBanks[bankIndex], from: prefillOutputs, outputOffset: cacheOutputOffset,
            configuration: configuration)
        }
        let prefillHidden = prefillOutputs[finalHiddenOutputIndex].as(of: FloatType.self)
        let prefillLogits = prefillOutputs[logitsOutputIndex].as(of: FloatType.self)
        var currentTokenGPU = Functional.argmax(
          DynamicGraph.Tensor<Float>(from: prefillLogits), axis: 1
        ).reshaped(.C(1))
        let firstTokenCPU = currentTokenGPU.toCPU()
        graph.joined()
        var generated = [Int32(firstTokenCPU[0])]
        guard maxTokens > 1 else {
          let prefillMilliseconds = (Date.timeIntervalSinceReferenceDate - prefillStart) * 1_000
          return Qwen3_5MTPGenerationResult(
            generatedTokenIds: generated,
            timing: Qwen3_5GenerationTiming(
              loadAndCompileMilliseconds: loadAndCompileMilliseconds,
              prefillMilliseconds: prefillMilliseconds,
              decodeCompileMilliseconds: 0, decodeLoopMilliseconds: 0, decodeLoopTokens: 0),
            acceptedDrafts: 0, rejectedDrafts: 0, d1AcceptedDrafts: 0, d1RejectedDrafts: 0,
            d2AcceptedDrafts: 0, d2RejectedDrafts: 0, replayRounds: 0)
        }
        let zeroMTPHidden = graph.variable(
          Tensor<BFloat16>(
            Array(repeating: BFloat16.zero, count: configuration.hiddenSize),
            .CPU, .NC(1, configuration.hiddenSize)
          ).toGPU(0))
        let mtpPrefillHidden: DynamicGraph.Tensor<BFloat16>
        if promptTokenIds.count > 1 {
          mtpPrefillHidden = Functional.concat(
            axis: 0,
            zeroMTPHidden,
            DynamicGraph.Tensor<BFloat16>(
              from: prefillHidden[
                0..<(promptTokenIds.count - 1), 0..<configuration.hiddenSize
              ]
            ).copied()
          )
        } else {
          mtpPrefillHidden = zeroMTPHidden
        }
        var mtpPrefillOutputs = mtpStep(
          (cachedTokenLength: 0, tokenLength: promptTokenIds.count, lastNumberOfTokens: 1),
          inputs: promptTokens,
          [
            emptyEmbeddings, emptyTokens,
            mtpPrefillHidden, mtpPrefillRotary,
            mtpK.reshaped(
              .NHWC(
                1, promptTokenIds.count, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
                configuration.attentionHeadDim, 1,
              ]),
            mtpV.reshaped(
              .NHWC(
                1, promptTokenIds.count, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
                configuration.attentionHeadDim, 1,
              ]),
          ])
        mtpPrefillOutputs.removeAll(keepingCapacity: false)
        let rotaryEmbedding: DynamicGraph.Tensor<FloatType>?
        if hasFullAttentionLayer {
          rotaryEmbedding = graph.variable(
            Qwen3_5RotaryEmbedding(
              sequenceLength: maxTokens + 2, cachedTokenLength: promptTokenIds.count,
              configuration: configuration, of: FloatType.self
            ).toGPU(0))
        } else {
          rotaryEmbedding = nil
        }
        let firstDraftRotary = rotaryEmbedding!.reshaped(
          .NHWC(1, 1, 1, configuration.attentionHeadDim),
          offset: [0, 0, 0, 0],
          strides: [
            (maxTokens + 2) * configuration.attentionHeadDim,
            configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
          ]
        ).copied()
        let firstDraftHidden = DynamicGraph.Tensor<BFloat16>(
          from: prefillHidden[
            (promptTokenIds.count - 1)..<promptTokenIds.count, 0..<configuration.hiddenSize
          ]
        ).copied()
        var firstDraftOutputs = mtpStep(
          (cachedTokenLength: promptTokenIds.count, tokenLength: 1, lastNumberOfTokens: 1),
          inputs: currentTokenGPU,
          [
            emptyEmbeddings, emptyTokens,
            firstDraftHidden, firstDraftRotary,
            mtpK.reshaped(
              .NHWC(
                1, promptTokenIds.count + 1, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
                configuration.attentionHeadDim, 1,
              ]),
            mtpV.reshaped(
              .NHWC(
                1, promptTokenIds.count + 1, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
                configuration.attentionHeadDim, 1,
              ]),
          ])
        let firstDraftLogits = firstDraftOutputs[1].as(of: FloatType.self)
        var currentDraftTokenGPU = Functional.argmax(firstDraftLogits, axis: 1).reshaped(.C(1))
          .copied()
        firstDraftOutputs.removeAll(keepingCapacity: false)
        let prefillMilliseconds = (Date.timeIntervalSinceReferenceDate - prefillStart) * 1_000

        let maxDecodeCachedTokenLength = cacheCapacity - decodeTokenLength
        var decodeCompileInputs: [DynamicGraph.AnyTensor] = [
          Functional.concat(axis: 0, currentTokenGPU, currentDraftTokenGPU), emptyEmbeddings,
          emptyTokens,
        ]
        if hasFullAttentionLayer {
          decodeCompileInputs.append(
            graph.variable(
              Qwen3_5RotaryEmbedding(
                sequenceLength: decodeTokenLength, cachedTokenLength: maxDecodeCachedTokenLength,
                configuration: configuration, of: FloatType.self
              ).toGPU(0)))
        }
        decodeCompileInputs += Self.cacheInputs(
          cacheBanks[0], currentTokenLength: cacheCapacity, configuration: configuration)
        let decodeCompileStart = Date.timeIntervalSinceReferenceDate
        decoder.compile(
          (
            cachedTokenLength: maxDecodeCachedTokenLength, tokenLength: decodeTokenLength,
            lastNumberOfTokens: decodeTokenLength,
            linearStateCheckpointCount: decodeTokenLength - 1, includeLogits: true
          ),
          inputs: decodeCompileInputs, isEager: true)
        let mtpCompileTokenBlock = Functional.concat(axis: 0, currentTokenGPU, currentDraftTokenGPU)
        var mtpCompileInputs: [DynamicGraph.AnyTensor] = [
          mtpCompileTokenBlock, emptyEmbeddings, emptyTokens,
          graph.variable(
            Tensor<BFloat16>(
              Array(
                repeating: BFloat16.zero,
                count: decodeTokenLength * configuration.hiddenSize),
              .CPU, .NC(decodeTokenLength, configuration.hiddenSize)
            ).toGPU(0)),
        ]
        mtpCompileInputs.append(
          graph.variable(
            Qwen3_5RotaryEmbedding(
              sequenceLength: decodeTokenLength, cachedTokenLength: maxDecodeCachedTokenLength,
              configuration: configuration, of: FloatType.self
            ).toGPU(0)))
        mtpCompileInputs += [
          mtpK.reshaped(
            .NHWC(
              1, cacheCapacity, configuration.keyValueHeads,
              configuration.attentionHeadDim),
            offset: [0, 0, 0, 0],
            strides: [
              mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
              configuration.attentionHeadDim, 1,
            ]),
          mtpV.reshaped(
            .NHWC(
              1, cacheCapacity, configuration.keyValueHeads,
              configuration.attentionHeadDim),
            offset: [0, 0, 0, 0],
            strides: [
              mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
              configuration.attentionHeadDim, 1,
            ]),
        ]
        mtpStep.compile(
          (
            cachedTokenLength: maxDecodeCachedTokenLength, tokenLength: decodeTokenLength,
            lastNumberOfTokens: 1
          ), inputs: mtpCompileInputs, isEager: true)
        graph.joined()
        let decodeCompileMilliseconds =
          (Date.timeIntervalSinceReferenceDate - decodeCompileStart) * 1_000
        let queueWatermark = DynamicGraph.queueWatermark
        DynamicGraph.queueWatermark = queueWatermark
        defer { DynamicGraph.queueWatermark = queueWatermark }

        var currentCachedTokenLength = promptTokenIds.count
        var currentCacheBankIndex = 0
        var acceptedDrafts = 0
        var rejectedDrafts = 0
        var d1AcceptedDrafts = 0
        var d1RejectedDrafts = 0
        var replayRounds = 0
        var shouldContinue = !eosTokenIds.contains(generated[0])
        let decodeLoopStart = Date.timeIntervalSinceReferenceDate

        while shouldContinue && generated.count < maxTokens {
          let roundCachedTokenLength = currentCachedTokenLength
          let firstTokenGPU = currentTokenGPU
          let firstDraftTokenGPU = currentDraftTokenGPU
          let inputCacheBankIndex = currentCacheBankIndex
          var outputCacheBankIndex = -1
          for bankIndex in cacheBanks.indices where bankIndex != currentCacheBankIndex {
            outputCacheBankIndex = bankIndex
            break
          }
          if outputCacheBankIndex < 0 {
            fatalError("No available Qwen3.5 MTP cache bank.")
          }

          var attentionInputs = [DynamicGraph.AnyTensor]()
          if let rotaryEmbedding = rotaryEmbedding {
            attentionInputs.append(
              rotaryEmbedding.reshaped(
                .NHWC(1, decodeTokenLength, 1, configuration.attentionHeadDim),
                offset: [0, roundCachedTokenLength - promptTokenIds.count, 0, 0],
                strides: [
                  (maxTokens + 2) * configuration.attentionHeadDim,
                  configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
                ]
              ).copied())
          }
          let cacheInputs = Self.cacheInputs(
            cacheBanks[inputCacheBankIndex],
            currentTokenLength: roundCachedTokenLength + decodeTokenLength,
            configuration: configuration)
          let tokenBlock = Functional.concat(axis: 0, firstTokenGPU, firstDraftTokenGPU)
          var outputCaches = cacheBanks[outputCacheBankIndex]
          var outputs = decoder(
            (
              cachedTokenLength: roundCachedTokenLength, tokenLength: decodeTokenLength,
              lastNumberOfTokens: decodeTokenLength,
              linearStateCheckpointCount: decodeTokenLength - 1, includeLogits: true
            ),
            inputs: tokenBlock, [emptyEmbeddings, emptyTokens] + attentionInputs + cacheInputs)
          let logits = outputs[logitsOutputIndex].as(of: FloatType.self)
          let tokenBlockGPU = Functional.argmax(logits, axis: 1).reshaped(.C(decodeTokenLength))
          let draftTokenCPU = firstDraftTokenGPU.toCPU()
          let tokenBlockCPU = tokenBlockGPU.toCPU()
          let firstDraftToken = Int32(draftTokenCPU[0])
          let firstVerifiedToken = Int32(tokenBlockCPU[0])

          let hidden = outputs[finalHiddenOutputIndex].as(of: FloatType.self)
          var acceptedDraftCount = 0
          if firstDraftToken == firstVerifiedToken {
            acceptedDrafts += 1
            d1AcceptedDrafts += 1
            acceptedDraftCount = 1
          } else {
            rejectedDrafts += 1
            d1RejectedDrafts += 1
            replayRounds += 1
          }
          let advanceCount = acceptedDraftCount + 1
          if advanceCount == decodeTokenLength {
            Self.updateLinearCaches(
              &outputCaches, from: outputs, outputOffset: cacheOutputOffset,
              configuration: configuration)
          } else {
            Self.updateLinearCaches(
              &outputCaches, from: outputs, outputOffset: cacheOutputOffset,
              stateCheckpointIndex: decodeTokenLength - advanceCount,
              tokenLength: decodeTokenLength, inputCaches: cacheBanks[inputCacheBankIndex],
              configuration: configuration)
          }
          cacheBanks[outputCacheBankIndex] = outputCaches

          for processIndex in 0..<advanceCount {
            let mtpCachedTokenLength = roundCachedTokenLength + 1 + processIndex
            let processToken = tokenBlockGPU.reshaped(
              .C(1), offset: [processIndex], strides: [1]
            ).copied()
            let processHidden = hidden[
              processIndex..<(processIndex + 1), 0..<configuration.hiddenSize
            ].copied()
            let processRotary = rotaryEmbedding!.reshaped(
              .NHWC(1, 1, 1, configuration.attentionHeadDim),
              offset: [
                0, roundCachedTokenLength - promptTokenIds.count + 1 + processIndex, 0, 0,
              ],
              strides: [
                (maxTokens + 2) * configuration.attentionHeadDim,
                configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
              ]
            ).copied()
            var processMTPOutputs = mtpStep(
              (
                cachedTokenLength: mtpCachedTokenLength, tokenLength: 1,
                lastNumberOfTokens: 1
              ),
              inputs: processToken,
              [
                emptyEmbeddings, emptyTokens,
                DynamicGraph.Tensor<BFloat16>(from: processHidden), processRotary,
                mtpK.reshaped(
                  .NHWC(
                    1, mtpCachedTokenLength + 1, configuration.keyValueHeads,
                    configuration.attentionHeadDim),
                  offset: [0, 0, 0, 0],
                  strides: [
                    mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
                    configuration.attentionHeadDim, 1,
                  ]),
                mtpV.reshaped(
                  .NHWC(
                    1, mtpCachedTokenLength + 1, configuration.keyValueHeads,
                    configuration.attentionHeadDim),
                  offset: [0, 0, 0, 0],
                  strides: [
                    mtpCacheCapacity * mtpCacheRowStride, mtpCacheRowStride,
                    configuration.attentionHeadDim, 1,
                  ]),
              ])
            let processedMTPLogits = processMTPOutputs[1].as(of: FloatType.self)
            currentDraftTokenGPU = Functional.argmax(
              DynamicGraph.Tensor<Float>(from: processedMTPLogits), axis: 1
            ).reshaped(.C(1)).copied()
            processMTPOutputs.removeAll(keepingCapacity: false)
          }

          currentTokenGPU = tokenBlockGPU.reshaped(
            .C(1), offset: [advanceCount - 1], strides: [1]
          ).copied()
          currentCachedTokenLength = roundCachedTokenLength + advanceCount
          currentCacheBankIndex = outputCacheBankIndex
          outputs.removeAll(keepingCapacity: false)

          var lastAppendedToken: Int32?
          for appendIndex in 0..<advanceCount where generated.count < maxTokens {
            let token = Int32(tokenBlockCPU[appendIndex])
            generated.append(token)
            lastAppendedToken = token
          }
          if let lastAppendedToken = lastAppendedToken {
            shouldContinue =
              generated.count < maxTokens && !eosTokenIds.contains(lastAppendedToken)
          } else {
            shouldContinue = generated.count < maxTokens
          }
        }
        graph.joined()
        let decodeLoopMilliseconds =
          (Date.timeIntervalSinceReferenceDate - decodeLoopStart) * 1_000
        let committedDecodeTokens = max(0, generated.count - 1)
        return Qwen3_5MTPGenerationResult(
          generatedTokenIds: generated,
          timing: Qwen3_5GenerationTiming(
            loadAndCompileMilliseconds: loadAndCompileMilliseconds,
            prefillMilliseconds: prefillMilliseconds,
            decodeCompileMilliseconds: decodeCompileMilliseconds,
            decodeLoopMilliseconds: decodeLoopMilliseconds,
            decodeLoopTokens: committedDecodeTokens),
          acceptedDrafts: acceptedDrafts, rejectedDrafts: rejectedDrafts,
          d1AcceptedDrafts: d1AcceptedDrafts, d1RejectedDrafts: d1RejectedDrafts,
          d2AcceptedDrafts: 0, d2RejectedDrafts: 0,
          replayRounds: replayRounds)
      }
    }
  }

  public func benchmarkTTFT(
    graph: DynamicGraph, promptTokenIds: [Int32], runs: Int, warmups: Int = 1
  ) throws -> [Double] {
    precondition(!promptTokenIds.isEmpty)
    guard runs > 0 else { return [] }
    let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
    let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: FloatType.self)
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad {
        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let tokenLength = promptTokenIds.count
        var caches = Self.makeCaches(
          graph: graph, capacity: tokenLength, configuration: configuration)

        let decoder:
          ModelBuilder<(cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int)> =
            ModelBuilder {
              (
                tokenLengths: (
                  cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int
                ), _
              ) in
              Qwen3_5CausalLM(
                FloatType.self, tokenLength: tokenLengths.tokenLength,
                cachedTokenLength: tokenLengths.cachedTokenLength, configuration: configuration,
                includeLogits: true, outputCacheStates: true, tieEmbedding: tieEmbedding,
                lastNumberOfTokens: tokenLengths.lastNumberOfTokens)
            }
        decoder.maxConcurrency = .limit(4)

        func promptInputs() -> (
          tokens: DynamicGraph.Tensor<Int32>, attentionInputs: [DynamicGraph.AnyTensor],
          cacheInputs: [DynamicGraph.AnyTensor]
        ) {
          let tokens = graph.variable(
            Tensor<Int32>(promptTokenIds, .CPU, .C(tokenLength)).toGPU(0))
          let attentionInputs: [DynamicGraph.AnyTensor]
          if hasFullAttentionLayer {
            let rotary = Qwen3_5RotaryEmbedding(
              sequenceLength: tokenLength, configuration: configuration, of: FloatType.self)
            attentionInputs = [graph.variable(rotary.toGPU(0))]
          } else {
            attentionInputs = []
          }
          let cacheInputs = Self.cacheInputs(
            caches, currentTokenLength: tokenLength, configuration: configuration)
          return (tokens, attentionInputs, cacheInputs)
        }

        try graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          let inputs = promptInputs()
          decoder.compile(
            (cachedTokenLength: 0, tokenLength: tokenLength, lastNumberOfTokens: 1),
            inputs: [inputs.tokens, emptyEmbeddings, emptyTokens] + inputs.attentionInputs
              + inputs.cacheInputs)
          try store.read(
            "text_model", model: decoder, strict: true, codec: [.jit, .i8x, .ezm7, .externalData])
        }

        func runPrefill() {
          Self.resetLinearCaches(&caches, configuration: configuration)
          graph.joined()
          let inputs = promptInputs()
          var outputs = decoder(
            (cachedTokenLength: 0, tokenLength: tokenLength, lastNumberOfTokens: 1),
            inputs: inputs.tokens,
            [emptyEmbeddings, emptyTokens] + inputs.attentionInputs + inputs.cacheInputs)
          Self.updateLinearCaches(
            &caches, from: outputs, outputOffset: 1, configuration: configuration)
          graph.joined()
          outputs.removeAll(keepingCapacity: false)
        }

        for _ in 0..<max(0, warmups) {
          runPrefill()
        }

        var durations = [Double]()
        durations.reserveCapacity(runs)
        for _ in 0..<runs {
          Self.resetLinearCaches(&caches, configuration: configuration)
          graph.joined()
          let inputs = promptInputs()
          let start = Date.timeIntervalSinceReferenceDate
          var outputs = decoder(
            (cachedTokenLength: 0, tokenLength: tokenLength, lastNumberOfTokens: 1),
            inputs: inputs.tokens,
            [emptyEmbeddings, emptyTokens] + inputs.attentionInputs + inputs.cacheInputs)
          Self.updateLinearCaches(
            &caches, from: outputs, outputOffset: 1, configuration: configuration)
          graph.joined()
          let end = Date.timeIntervalSinceReferenceDate
          durations.append((end - start) * 1_000)
          if outputs[0].isNaN {
            throw Qwen3_5TextGenerationError.invalidLogits
          }
          outputs.removeAll(keepingCapacity: false)
        }
        return durations
      }
    }
  }

  public func measureMTPDraftAcceptance(
    graph: DynamicGraph, promptTokenIds: [Int32], startCount: Int, maxDraftTokens: Int
  ) throws -> Qwen3_5MTPDraftAcceptanceResult {
    precondition(!promptTokenIds.isEmpty)
    let startCount = max(1, startCount)
    let maxDraftTokens = max(1, min(maxDraftTokens, 4))
    let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
    let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: FloatType.self)
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad { () throws -> Qwen3_5MTPDraftAcceptanceResult in
        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let finalHiddenOutputIndex = 0
        let logitsOutputIndex = 1
        let cacheOutputOffset = 2
        let cacheCapacity = promptTokenIds.count + startCount + maxDraftTokens + 1
        var caches = Self.makeCaches(
          graph: graph, capacity: cacheCapacity, configuration: configuration)
        let decoder: ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)> = ModelBuilder {
          (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
          return Qwen3_5CausalLM(
            FloatType.self, tokenLength: tokenLengths.tokenLength,
            cachedTokenLength: tokenLengths.cachedTokenLength, configuration: configuration,
            includeLogits: true, outputCacheStates: true,
            outputFinalHiddenState: true,
            tieEmbedding: tieEmbedding, lastNumberOfTokens: 1)
        }
        decoder.maxConcurrency = .limit(4)
        let promptPrefixCount = promptTokenIds.count - 1
        let promptLastToken = graph.variable(
          Tensor<Int32>([promptTokenIds[promptPrefixCount]], .CPU, .C(1)).toGPU(0))
        let prefillInputs: [DynamicGraph.AnyTensor]
        if promptPrefixCount > 0 {
          let promptPrefixTokens = graph.variable(
            Tensor<Int32>(
              Array(promptTokenIds[0..<promptPrefixCount]), .CPU, .C(promptPrefixCount)
            ).toGPU(0))
          let promptPrefixAttentionInputs: [DynamicGraph.AnyTensor]
          if hasFullAttentionLayer {
            let rotary = Qwen3_5RotaryEmbedding(
              sequenceLength: promptPrefixCount, configuration: configuration, of: FloatType.self)
            promptPrefixAttentionInputs = [graph.variable(rotary.toGPU(0))]
          } else {
            promptPrefixAttentionInputs = []
          }
          let promptPrefixCacheInputs = Self.cacheInputs(
            caches, currentTokenLength: promptPrefixCount, configuration: configuration)
          prefillInputs =
            [promptPrefixTokens, emptyEmbeddings, emptyTokens] + promptPrefixAttentionInputs
            + promptPrefixCacheInputs
        } else {
          let promptLastAttentionInputs: [DynamicGraph.AnyTensor]
          if hasFullAttentionLayer {
            let rotary = Qwen3_5RotaryEmbedding(
              sequenceLength: 1, cachedTokenLength: promptPrefixCount, configuration: configuration,
              of: FloatType.self)
            promptLastAttentionInputs = [graph.variable(rotary.toGPU(0))]
          } else {
            promptLastAttentionInputs = []
          }
          let promptLastCacheInputs = Self.cacheInputs(
            caches, currentTokenLength: 1, configuration: configuration)
          prefillInputs =
            [promptLastToken, emptyEmbeddings, emptyTokens] + promptLastAttentionInputs
            + promptLastCacheInputs
        }
        decoder.compile(
          (cachedTokenLength: 0, tokenLength: max(promptPrefixCount, 1)),
          inputs: prefillInputs)
        try graph.openStore(
          filePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          try store.read(
            "text_model", model: decoder, strict: true,
            codec: [.jit, .i8x, .ezm7, .externalData])
        }
        decoder.compile(
          (cachedTokenLength: 0, tokenLength: max(promptPrefixCount, 1)),
          inputs: prefillInputs, isEager: true)
        if promptPrefixCount > 0 {
          let promptPrefixTokens = graph.variable(
            Tensor<Int32>(
              Array(promptTokenIds[0..<promptPrefixCount]), .CPU, .C(promptPrefixCount)
            ).toGPU(0))
          let promptPrefixAttentionInputs: [DynamicGraph.AnyTensor]
          if hasFullAttentionLayer {
            let rotary = Qwen3_5RotaryEmbedding(
              sequenceLength: promptPrefixCount, configuration: configuration, of: FloatType.self)
            promptPrefixAttentionInputs = [graph.variable(rotary.toGPU(0))]
          } else {
            promptPrefixAttentionInputs = []
          }
          let promptPrefixCacheInputs = Self.cacheInputs(
            caches, currentTokenLength: promptPrefixCount, configuration: configuration)
          var promptPrefixOutputs = decoder(
            (cachedTokenLength: 0, tokenLength: promptPrefixCount), inputs: promptPrefixTokens,
            [emptyEmbeddings, emptyTokens] + promptPrefixAttentionInputs + promptPrefixCacheInputs)
          Self.updateLinearCaches(
            &caches, from: promptPrefixOutputs, outputOffset: cacheOutputOffset,
            configuration: configuration)
          promptPrefixOutputs.removeAll(keepingCapacity: false)
        }
        let promptLastAttentionInputs: [DynamicGraph.AnyTensor]
        if hasFullAttentionLayer {
          let rotary = Qwen3_5RotaryEmbedding(
            sequenceLength: 1, cachedTokenLength: promptPrefixCount, configuration: configuration,
            of: FloatType.self)
          promptLastAttentionInputs = [graph.variable(rotary.toGPU(0))]
        } else {
          promptLastAttentionInputs = []
        }
        let promptLastCacheInputs = Self.cacheInputs(
          caches, currentTokenLength: promptTokenIds.count, configuration: configuration)
        var prefillOutputs = decoder(
          (cachedTokenLength: promptPrefixCount, tokenLength: 1), inputs: promptLastToken,
          [emptyEmbeddings, emptyTokens] + promptLastAttentionInputs + promptLastCacheInputs)
        Self.updateLinearCaches(
          &caches, from: prefillOutputs, outputOffset: cacheOutputOffset,
          configuration: configuration)
        var currentHidden = prefillOutputs[finalHiddenOutputIndex].as(of: FloatType.self).reshaped(
          .NC(1, configuration.hiddenSize), offset: [0, 0],
          strides: [configuration.hiddenSize, 1]
        ).contiguous(streamContext: streamContext).copied()
        var currentTokenGPU = Functional.argmax(
          DynamicGraph.Tensor<Float>(
            from: prefillOutputs[logitsOutputIndex].as(of: FloatType.self)),
          axis: 1
        ).reshaped(.C(1)).copied()

        let maxDecodeCachedTokenLength = cacheCapacity - 1
        let decodeCompileInputs: [DynamicGraph.AnyTensor] =
          [currentTokenGPU, emptyEmbeddings, emptyTokens]
          + (hasFullAttentionLayer
            ? [
              graph.variable(
                Qwen3_5RotaryEmbedding(
                  sequenceLength: 1, cachedTokenLength: maxDecodeCachedTokenLength,
                  configuration: configuration, of: FloatType.self
                ).toGPU(0))
            ] : [])
          + Self.cacheInputs(
            caches, currentTokenLength: cacheCapacity, configuration: configuration)
        decoder.compile(
          (cachedTokenLength: maxDecodeCachedTokenLength, tokenLength: 1),
          inputs: decodeCompileInputs, isEager: true)
        graph.joined()
        prefillOutputs.removeAll(keepingCapacity: false)

        let traceTokenCount = startCount + maxDraftTokens
        let traceRotaryGPU: DynamicGraph.Tensor<FloatType>?
        if hasFullAttentionLayer {
          traceRotaryGPU = graph.variable(
            Qwen3_5RotaryEmbedding(
              sequenceLength: traceTokenCount, cachedTokenLength: promptTokenIds.count,
              configuration: configuration, of: FloatType.self
            ).toGPU(0))
        } else {
          traceRotaryGPU = nil
        }
        var traceTokensGPU = [DynamicGraph.Tensor<Int32>]()
        var traceTokenCPUs = [DynamicGraph.Tensor<Int32>]()
        var traceHiddens = [DynamicGraph.Tensor<BFloat16>]()
        var currentCachedTokenLength = promptTokenIds.count
        traceTokensGPU.reserveCapacity(traceTokenCount)
        traceTokenCPUs.reserveCapacity(traceTokenCount)
        traceHiddens.reserveCapacity(traceTokenCount)
        for tokenIndex in 0..<traceTokenCount {
          let tokenGPU = currentTokenGPU.copied()
          traceTokensGPU.append(tokenGPU)
          traceTokenCPUs.append(tokenGPU.toCPU())
          traceHiddens.append(DynamicGraph.Tensor<BFloat16>(from: currentHidden).copied())
          guard tokenIndex + 1 < traceTokenCount else { continue }
          var targetAttentionInputs = [DynamicGraph.AnyTensor]()
          if let traceRotaryGPU = traceRotaryGPU {
            let rotaryOffset = currentCachedTokenLength - promptTokenIds.count
            targetAttentionInputs.append(
              traceRotaryGPU.reshaped(
                .NHWC(1, 1, 1, configuration.attentionHeadDim),
                offset: [0, rotaryOffset, 0, 0],
                strides: [
                  traceTokenCount * configuration.attentionHeadDim,
                  configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
                ]
              ).copied())
          }
          let targetCacheInputs = Self.cacheInputs(
            caches, currentTokenLength: currentCachedTokenLength + 1,
            configuration: configuration)
          var targetOutputs = decoder(
            (cachedTokenLength: currentCachedTokenLength, tokenLength: 1),
            inputs: tokenGPU,
            [emptyEmbeddings, emptyTokens] + targetAttentionInputs + targetCacheInputs)
          Self.updateLinearCaches(
            &caches, from: targetOutputs, outputOffset: cacheOutputOffset,
            configuration: configuration)
          currentHidden = targetOutputs[finalHiddenOutputIndex].as(of: FloatType.self).reshaped(
            .NC(1, configuration.hiddenSize), offset: [0, 0],
            strides: [configuration.hiddenSize, 1]
          ).contiguous(streamContext: streamContext).copied()
          currentTokenGPU = Functional.argmax(
            DynamicGraph.Tensor<Float>(
              from: targetOutputs[logitsOutputIndex].as(of: FloatType.self)), axis: 1
          ).reshaped(.C(1)).copied()
          currentCachedTokenLength += 1
          targetOutputs.removeAll(keepingCapacity: false)
        }
        graph.joined()
        let traceTokens = traceTokenCPUs.map { Int32($0[0]) }

        let mtpTokenLength = traceTokenCount + maxDraftTokens
        let mtpCacheElementCount =
          mtpTokenLength * configuration.keyValueHeads * configuration.attentionHeadDim
        let mtpK = graph.variable(
          Tensor<BFloat16>(
            Array(repeating: BFloat16.zero, count: mtpCacheElementCount),
            .CPU,
            .NHWC(1, mtpTokenLength, configuration.keyValueHeads, configuration.attentionHeadDim)
          ).toGPU(0))
        let mtpV = graph.variable(
          Tensor<BFloat16>(
            Array(repeating: BFloat16.zero, count: mtpCacheElementCount),
            .CPU,
            .NHWC(1, mtpTokenLength, configuration.keyValueHeads, configuration.attentionHeadDim)
          ).toGPU(0))
        let mtpRowStride = configuration.keyValueHeads * configuration.attentionHeadDim
        func mtpCacheInputs(currentTokenLength: Int) -> [DynamicGraph.AnyTensor] {
          [
            mtpK.reshaped(
              .NHWC(
                1, currentTokenLength, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                mtpTokenLength * mtpRowStride, mtpRowStride, configuration.attentionHeadDim, 1,
              ]),
            mtpV.reshaped(
              .NHWC(
                1, currentTokenLength, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                mtpTokenLength * mtpRowStride, mtpRowStride, configuration.attentionHeadDim, 1,
              ]),
          ]
        }
        let mtpStep: ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)> = ModelBuilder {
          tokenLengths, _ in
          return Qwen3_5MTP(
            BFloat16.self, FloatType.self, configuration: configuration, batchSize: 1,
            tokenLength: tokenLengths.tokenLength,
            cachedTokenLength: tokenLengths.cachedTokenLength, lastNumberOfTokens: 1,
            tieEmbedding: tieEmbedding)
        }
        mtpStep.maxConcurrency = .limit(4)
        let mtpRotaryLength = traceTokenCount
        let mtpStepRotary = graph.variable(
          Qwen3_5RotaryEmbedding(
            sequenceLength: mtpRotaryLength, cachedTokenLength: promptTokenIds.count,
            configuration: configuration, of: FloatType.self
          ).toGPU(0))
        let compileRotary = mtpStepRotary.reshaped(
          .NHWC(1, 1, 1, configuration.attentionHeadDim),
          offset: [0, 0, 0, 0],
          strides: [
            mtpRotaryLength * configuration.attentionHeadDim,
            configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
          ])
        mtpStep.compile(
          (cachedTokenLength: 0, tokenLength: 1),
          inputs: [traceTokensGPU[0], emptyEmbeddings, emptyTokens, traceHiddens[0], compileRotary]
            + mtpCacheInputs(currentTokenLength: 1))
        mtpStep.compile(
          (cachedTokenLength: mtpTokenLength - 1, tokenLength: 1),
          inputs: [traceTokensGPU[0], emptyEmbeddings, emptyTokens, traceHiddens[0], compileRotary]
            + mtpCacheInputs(currentTokenLength: mtpTokenLength))
        try graph.openStore(
          filePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          try store.read(
            "text_model", model: mtpStep, strict: true,
            codec: [.jit, .i8x, .ezm7, .externalData])
        }

        var acceptedPrefixCounts = Array(repeating: 0, count: maxDraftTokens)
        var acceptedLengthCounts = Array(repeating: 0, count: maxDraftTokens + 1)
        for windowIndex in 0..<startCount {
          var mtpStepToken = traceTokensGPU[windowIndex]
          var mtpStepHidden = traceHiddens[windowIndex]
          var acceptedLength = 0
          var stillPrefix = true
          for depthIndex in 0..<maxDraftTokens {
            let rotaryOffset = windowIndex + depthIndex
            let mtpStepRotarySlice = mtpStepRotary.reshaped(
              .NHWC(1, 1, 1, configuration.attentionHeadDim),
              offset: [0, rotaryOffset, 0, 0],
              strides: [
                mtpRotaryLength * configuration.attentionHeadDim,
                configuration.attentionHeadDim, configuration.attentionHeadDim, 1,
              ]
            ).copied()
            let mtpCachedTokenLength = windowIndex + depthIndex
            var mtpOutputs = mtpStep(
              (cachedTokenLength: mtpCachedTokenLength, tokenLength: 1),
              inputs: mtpStepToken,
              [emptyEmbeddings, emptyTokens, mtpStepHidden, mtpStepRotarySlice]
                + mtpCacheInputs(currentTokenLength: mtpCachedTokenLength + 1))
            mtpStepHidden = DynamicGraph.Tensor<BFloat16>(from: mtpOutputs[0]).copied()
            let draftLogits = mtpOutputs[1].as(of: FloatType.self)
            mtpStepToken = Functional.argmax(
              DynamicGraph.Tensor<Float>(from: draftLogits), axis: 1
            ).reshaped(.C(1)).copied()
            let draftCPU = mtpStepToken.toCPU()
            graph.joined()
            let draftToken = Int32(draftCPU[0])
            let targetToken = traceTokens[windowIndex + depthIndex + 1]
            if stillPrefix && draftToken == targetToken {
              acceptedPrefixCounts[depthIndex] += 1
              acceptedLength = depthIndex + 1
            } else {
              stillPrefix = false
            }
            mtpOutputs.removeAll(keepingCapacity: false)
          }
          acceptedLengthCounts[acceptedLength] += 1
        }

        return Qwen3_5MTPDraftAcceptanceResult(
          promptTokenCount: promptTokenIds.count, startCount: startCount,
          maxDraftTokens: maxDraftTokens, acceptedPrefixCounts: acceptedPrefixCounts,
          acceptedLengthCounts: acceptedLengthCounts)
      }
    }
  }

  public func generateMultimodal(
    graph: DynamicGraph, promptTokenIds: [Int32], tokenTypeIds: [Int32],
    imagePatches: Tensor<Float>, imageGridThw: [(t: Int, h: Int, w: Int)],
    visionConfiguration: Qwen3_5VisionConfiguration = .qwen3_5_4B, maxTokens: Int,
    partialHandler: ([Int32]) -> Bool
  ) throws -> [Int32] {
    precondition(!promptTokenIds.isEmpty)
    precondition(promptTokenIds.count == tokenTypeIds.count)
    guard maxTokens > 0 else { return [] }
    let streamContext = StreamContext(.GPU(0))
    return try graph.withStream(streamContext) {
      try graph.withNoGrad {
        let imageTokenPositions = promptTokenIds.enumerated().compactMap {
          $0.element == 248_056 ? Int32($0.offset) : nil
        }
        let expectedImageTokenCount = imageGridThw.reduce(0) {
          $0 + $1.t * $1.h * $1.w
            / (visionConfiguration.spatialMergeSize * visionConfiguration.spatialMergeSize)
        }
        precondition(imageTokenPositions.count == expectedImageTokenCount)

        let hasFullAttentionLayer = (0..<configuration.layers).contains {
          !configuration.isLinearAttentionLayer($0)
        }
        let cacheCapacity = promptTokenIds.count + maxTokens + 1
        var caches = Self.makeCaches(
          graph: graph, capacity: cacheCapacity, configuration: configuration)

        let rotary = Qwen3_5VisionRotaryEmbedding(
          gridThw: imageGridThw, configuration: visionConfiguration, of: FloatType.self)
        let sequenceOffsets = Qwen3_5VisionSequenceOffsets(gridThw: imageGridThw).offsets
        let rotaryGPU = graph.variable(rotary.toGPU(0))
        let sequenceOffsetsGPU = graph.variable(sequenceOffsets.toGPU(0))
        var positionEmbedding: Tensor<Float>?
        try graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          guard
            let weight = store.read(
              "model.visual.pos_embed.weight", codec: [.jit, .i8x, .ezm7, .externalData])
          else {
            throw Qwen3_5TextGenerationError.missingStoreTensor(
              "model.visual.pos_embed.weight")
          }
          positionEmbedding = Qwen3_5VisionPositionEmbedding(
            weight: weight, gridThw: imageGridThw, configuration: visionConfiguration,
            of: Float.self)
        }
        let patchesGPU = graph.variable(imagePatches.toGPU(0))
        let positionGPU = graph.variable(positionEmbedding!.toGPU(0))
        let visionModel = Qwen3_5VisionTransformer(
          Float.self, gridThw: imageGridThw, configuration: visionConfiguration)
        visionModel.maxConcurrency = .limit(4)
        visionModel.compile(inputs: patchesGPU, positionGPU, rotaryGPU, sequenceOffsetsGPU)
        try graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          try store.read(
            "vision_model", model: visionModel, strict: true,
            codec: [.jit, .i8x, .ezm7, .externalData])
        }
        let imageEmbeds = visionModel(
          inputs: patchesGPU, positionGPU, rotaryGPU, sequenceOffsetsGPU
        )[0].as(of: Float.self)
        let imageEmbedsTyped = DynamicGraph.Tensor<FloatType>(from: imageEmbeds)

        let emptyTokens = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: Int32.self)
        let emptyEmbeddings = graph.variable(.GPU(0), format: .NHWC, shape: [0], of: FloatType.self)
        let decoder = ModelBuilder {
          (lengths: (cachedTokenLength: Int, tokenLength: Int, lastNumberOfTokens: Int), _) in
          Qwen3_5CausalLM(
            FloatType.self, tokenLength: lengths.tokenLength,
            cachedTokenLength: lengths.cachedTokenLength, configuration: configuration,
            includeLogits: true, outputCacheStates: true, tieEmbedding: tieEmbedding,
            lastNumberOfTokens: lengths.lastNumberOfTokens)
        }
        decoder.maxConcurrency = .limit(4)
        let positionIDs = Qwen3_5MakeMultimodalPositionIDs(
          tokenTypeIDs: tokenTypeIds, imageGridThw: imageGridThw, configuration: visionConfiguration
        )
        let decodePositionOffset = positionIDs.ropeDelta
        let allRotary = Qwen3_5RotaryEmbedding(
          positionIDs: positionIDs, configuration: configuration, of: FloatType.self)
        var imageRuns = [Range<Int>]()
        for position in tokenTypeIds.indices where tokenTypeIds[position] == 1 {
          if let last = imageRuns.last, last.upperBound == position {
            imageRuns[imageRuns.count - 1] = last.lowerBound..<(position + 1)
          } else {
            imageRuns.append(position..<(position + 1))
          }
        }
        var start = 0
        var imageRow = 0
        var prefillOutputs = [DynamicGraph.AnyTensor]()
        while start < promptTokenIds.count {
          let run = imageRuns.first { $0.upperBound > start }
          let end =
            run.flatMap { current in
              imageRuns.first { $0.lowerBound > current.lowerBound }?.lowerBound
            }
            ?? promptTokenIds.count
          let preEnd = run?.lowerBound ?? end
          let postStart = run?.upperBound ?? end
          let pre =
            preEnd == start
            ? emptyTokens
            : graph.variable(
              Tensor<Int32>(
                Array(promptTokenIds[start..<preEnd]), kind: .CPU, format: .NHWC,
                shape: [preEnd - start]
              ).toGPU(0))
          let post =
            postStart == end
            ? emptyTokens
            : graph.variable(
              Tensor<Int32>(
                Array(promptTokenIds[postStart..<end]), kind: .CPU, format: .NHWC,
                shape: [end - postStart]
              ).toGPU(0))
          let injected: DynamicGraph.Tensor<FloatType>
          if let run {
            injected = imageEmbedsTyped.reshaped(
              .WC(run.count, configuration.hiddenSize),
              offset: [imageRow, 0], strides: [configuration.hiddenSize, 1]
            ).copied()
            imageRow += run.count
          } else {
            injected = emptyEmbeddings
          }
          var inputs: [DynamicGraph.AnyTensor] = [pre, injected, post]
          if hasFullAttentionLayer {
            inputs.append(
              graph.variable(
                allRotary[0..<1, start..<end, 0..<1, 0..<configuration.attentionHeadDim].toGPU(0)))
          }
          inputs += Self.cacheInputs(caches, currentTokenLength: end, configuration: configuration)
          let lengths = (
            cachedTokenLength: start, tokenLength: end - start,
            lastNumberOfTokens: end == promptTokenIds.count ? 1 : 0
          )
          if start == 0 {
            decoder.compile(lengths, inputs: inputs)
            try graph.openStore(
              filePath, flags: .readOnly,
              externalStore: TensorData.externalStore(filePath: filePath)
            ) { store in
              try store.read(
                "text_model", model: decoder, strict: true,
                codec: [.jit, .i8x, .ezm7, .externalData])
            }
          }
          prefillOutputs = decoder(lengths, inputs: inputs[0], Array(inputs.dropFirst()))
          Self.updateLinearCaches(
            &caches, from: prefillOutputs, outputOffset: 1, configuration: configuration)
          start = end
        }
        let token = Functional.argmax(prefillOutputs[0].as(of: FloatType.self), axis: 1).toCPU()
        graph.joined()
        let nextTokenFromPrefill = Int32(token[0, 0])
        prefillOutputs.removeAll(keepingCapacity: false)

        let nextToken = nextTokenFromPrefill
        var generated = [nextToken]
        guard partialHandler(generated), !eosTokenIds.contains(nextToken) else {
          return generated
        }
        guard maxTokens > 1 else {
          return generated
        }

        let maxDecodeCachedTokenLength = cacheCapacity - 1
        let decodeCompileToken = graph.variable(Tensor<Int32>([nextToken], .CPU, .C(1)).toGPU(0))
        var decodeCompileAttentionInputs = [DynamicGraph.AnyTensor]()
        if hasFullAttentionLayer {
          let decodeCompileRotary = Qwen3_5RotaryEmbedding(
            sequenceLength: 1,
            cachedTokenLength: maxDecodeCachedTokenLength + decodePositionOffset,
            configuration: configuration, of: FloatType.self)
          decodeCompileAttentionInputs = [graph.variable(decodeCompileRotary.toGPU(0))]
        }
        let decodeCompileCacheInputs = Self.cacheInputs(
          caches, currentTokenLength: cacheCapacity, configuration: configuration)
        decoder.compile(
          (
            cachedTokenLength: maxDecodeCachedTokenLength, tokenLength: 1,
            lastNumberOfTokens: 1
          ),
          inputs: [decodeCompileToken, emptyEmbeddings, emptyTokens]
            + decodeCompileAttentionInputs + decodeCompileCacheInputs,
          isEager: true)
        graph.joined()
        let queueWatermark = DynamicGraph.queueWatermark
        DynamicGraph.queueWatermark = queueWatermark * 16  // Expanding queue watermark to support longer generation.
        defer {
          DynamicGraph.queueWatermark = queueWatermark
        }
        var nextTokenGPU = graph.variable(Tensor<Int32>([nextToken], .CPU, .C(1)).toGPU(0))
        let cachedTokenLength = promptTokenIds.count + generated.count - 1
        let rotaryEmbeddingForMaxTokens = graph.variable(
          Qwen3_5RotaryEmbedding(
            sequenceLength: maxTokens, cachedTokenLength: cachedTokenLength + decodePositionOffset,
            configuration: configuration, of: FloatType.self
          ).toGPU(0))
        for i in 1..<maxTokens {
          let cachedTokenLength = promptTokenIds.count + generated.count - 1
          var oneAttentionInputs = [DynamicGraph.AnyTensor]()
          if hasFullAttentionLayer {
            let oneRotary = rotaryEmbeddingForMaxTokens.reshaped(
              format: .NHWC, shape: [1, 1, 1, 256], offset: [0, 0, i - 1, 0],
              strides: [256, 256, 256, 1])
            oneAttentionInputs = [oneRotary.copied()]
          }
          let cacheInputs = Self.cacheInputs(
            caches, currentTokenLength: cachedTokenLength + 1, configuration: configuration)
          let oldDecodedTokenCPU = nextTokenGPU.toCPU()
          let decodeOutputs = decoder(
            (cachedTokenLength: cachedTokenLength, tokenLength: 1, lastNumberOfTokens: 1),
            inputs: nextTokenGPU,
            [emptyEmbeddings, emptyTokens] + oneAttentionInputs + cacheInputs)
          Self.updateLinearCaches(
            &caches, from: decodeOutputs, outputOffset: 1, configuration: configuration)
          let logits = decodeOutputs[0].as(of: FloatType.self)
          nextTokenGPU = Functional.argmax(logits, axis: 1).reshaped(.C(1))
          if i > 1 {
            let decodedToken = oldDecodedTokenCPU[0]
            generated.append(decodedToken)
            if eosTokenIds.contains(decodedToken) || !partialHandler(generated) {
              break
            }
          }
        }
        graph.joined()
        return generated
      }
    }
  }

  private static func makeCaches(
    graph: DynamicGraph, capacity: Int, configuration: Qwen3_5ModelConfiguration
  ) -> [DynamicGraph.AnyTensor] {
    var caches = [DynamicGraph.AnyTensor]()
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        let convState = graph.variable(
          Tensor<FloatType>(
            Array(
              repeating: FloatType.zero,
              count: (configuration.linearConvKernel - 1) * configuration.linearConvDim),
            .CPU,
            .NHWC(1, configuration.linearConvKernel - 1, 1, configuration.linearConvDim)
          ).toGPU(0))
        let recurrentState = graph.variable(
          Tensor<Float>(
            Array(
              repeating: Float.zero,
              count: configuration.linearNumValueHeads * configuration.linearKeyHeadDim
                * configuration.linearValueHeadDim),
            .CPU,
            .NHWC(
              1, configuration.linearNumValueHeads, configuration.linearValueHeadDim,
              configuration.linearKeyHeadDim)
          ).toGPU(0))
        caches.append(contentsOf: [convState, recurrentState])
      } else {
        let k = graph.variable(
          Tensor<FloatType>(
            Array(
              repeating: FloatType.zero,
              count: capacity * configuration.keyValueHeads * configuration.attentionHeadDim),
            .CPU,
            .NHWC(1, capacity, configuration.keyValueHeads, configuration.attentionHeadDim)
          ).toGPU(0))
        let v = graph.variable(
          Tensor<FloatType>(
            Array(
              repeating: FloatType.zero,
              count: capacity * configuration.keyValueHeads * configuration.attentionHeadDim),
            .CPU,
            .NHWC(1, capacity, configuration.keyValueHeads, configuration.attentionHeadDim)
          ).toGPU(0))
        caches.append(contentsOf: [k, v])
      }
    }
    return caches
  }

  private static func cacheInputs(
    _ caches: [DynamicGraph.AnyTensor], currentTokenLength: Int,
    configuration: Qwen3_5ModelConfiguration
  ) -> [DynamicGraph.AnyTensor] {
    var result = [DynamicGraph.AnyTensor]()
    var cursor = 0
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        result.append(caches[cursor])
        result.append(caches[cursor + 1])
      } else {
        result.append(
          caches[cursor].as(of: FloatType.self).reshaped(
            .NHWC(
              1, currentTokenLength, configuration.keyValueHeads, configuration.attentionHeadDim)))
        result.append(
          caches[cursor + 1].as(of: FloatType.self).reshaped(
            .NHWC(
              1, currentTokenLength, configuration.keyValueHeads, configuration.attentionHeadDim)))
      }
      cursor += 2
    }
    return result
  }

  private static func updateLinearCaches(
    _ caches: inout [DynamicGraph.AnyTensor], from outputs: [DynamicGraph.AnyTensor],
    outputOffset: Int, stateCheckpointIndex: Int = 0,
    tokenLength: Int = 1, inputCaches: [DynamicGraph.AnyTensor]? = nil,
    configuration: Qwen3_5ModelConfiguration
  ) {
    var cacheCursor = 0
    var outputCursor = outputOffset
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        let convStateHistory = outputs[outputCursor].as(of: FloatType.self)
        let recurrentStateHistory = outputs[outputCursor + 1].as(of: Float.self)
        let convStateLength = configuration.linearConvKernel - 1
        let convState: DynamicGraph.Tensor<FloatType>
        if stateCheckpointIndex == 0 {
          convState = convStateHistory.reshaped(
            .NHWC(1, convStateLength, 1, configuration.linearConvDim),
            offset: [0, 0, 0, 0],
            strides: [
              convStateHistory.shape[1] * configuration.linearConvDim,
              configuration.linearConvDim, configuration.linearConvDim, 1,
            ]
          ).copied()
        } else {
          precondition(tokenLength > stateCheckpointIndex)
          guard let inputCaches = inputCaches else {
            fatalError("Input linear caches are required to reconstruct intermediate conv state.")
          }
          let consumedTokenCount = tokenLength - stateCheckpointIndex
          precondition(consumedTokenCount > 0 && consumedTokenCount <= convStateLength)
          let oldTailCount = convStateLength - consumedTokenCount
          let newHeadOffset = max(0, convStateLength - tokenLength)
          let inputConvState = inputCaches[cacheCursor].as(of: FloatType.self)
          let newHead = convStateHistory.reshaped(
            .NHWC(1, consumedTokenCount, 1, configuration.linearConvDim),
            offset: [0, newHeadOffset, 0, 0],
            strides: [
              convStateHistory.shape[1] * configuration.linearConvDim,
              configuration.linearConvDim, configuration.linearConvDim, 1,
            ])
          if oldTailCount > 0 {
            let oldTail = inputConvState.reshaped(
              .NHWC(1, oldTailCount, 1, configuration.linearConvDim),
              offset: [0, consumedTokenCount, 0, 0],
              strides: [
                inputConvState.shape[1] * configuration.linearConvDim,
                configuration.linearConvDim, configuration.linearConvDim, 1,
              ])
            convState = Functional.concat(axis: 1, oldTail, newHead).copied()
          } else {
            convState = newHead.copied()
          }
        }
        let recurrentState = recurrentStateHistory.reshaped(
          .NHWC(
            1, configuration.linearNumValueHeads, configuration.linearValueHeadDim,
            configuration.linearKeyHeadDim),
          offset: [0, stateCheckpointIndex * configuration.linearNumValueHeads, 0, 0],
          strides: [
            recurrentStateHistory.shape[1] * configuration.linearValueHeadDim
              * configuration.linearKeyHeadDim,
            configuration.linearValueHeadDim * configuration.linearKeyHeadDim,
            configuration.linearKeyHeadDim, 1,
          ]
        ).copied()
        var convStateCache = caches[cacheCursor].as(of: FloatType.self)
        var recurrentStateCache = caches[cacheCursor + 1].as(of: Float.self)
        convStateCache[
          0..<1, 0..<convStateLength, 0..<1, 0..<configuration.linearConvDim
        ] = convState
        recurrentStateCache[
          0..<1, 0..<configuration.linearNumValueHeads, 0..<configuration.linearValueHeadDim,
          0..<configuration.linearKeyHeadDim
        ] = recurrentState
        outputCursor += 2
      }
      cacheCursor += 2
    }
  }

  private static func resetLinearCaches(
    _ caches: inout [DynamicGraph.AnyTensor], configuration: Qwen3_5ModelConfiguration
  ) {
    var cursor = 0
    for layerIndex in 0..<configuration.layers {
      if configuration.isLinearAttentionLayer(layerIndex) {
        caches[cursor].as(of: FloatType.self).full(0)
        caches[cursor + 1].as(of: Float.self).full(0)
      }
      cursor += 2
    }
  }

}
