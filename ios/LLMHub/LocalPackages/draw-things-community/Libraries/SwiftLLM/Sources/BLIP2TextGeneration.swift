import Foundation
import NNC

public struct BLIP2TextGeneration<T: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  let tokenMapping: [Int32: String]
  let usesFlashAttention: Bool
  public init(filePath: String, vocabulary: String, usesFlashAttention: Bool) {
    self.filePath = filePath
    let vocabJSONData = try! Data(contentsOf: URL(fileURLWithPath: vocabulary))
    let decoder = JSONDecoder()
    let vocabulary = try! decoder.decode([String: Int32].self, from: vocabJSONData)
    var tokenMapping = [Int32: String]()
    for (key, value) in vocabulary {
      tokenMapping[value] = key
    }
    self.tokenMapping = tokenMapping
    self.usesFlashAttention = usesFlashAttention
  }
}

extension BLIP2TextGeneration {
  public func generate(
    _ embedding: DynamicGraph.Tensor<T>,
    opt existingOpt: ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)>? = nil,
    lmHead existingLmHead: Model? = nil,
    partialHandler: (String) -> Bool
  ) -> (String, ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)>, Model) {
    let graph = embedding.graph
    return graph.withNoGrad {
      let opt =
        existingOpt
        ?? ModelBuilder { (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
          return OPTDecoder(
            T.self, vocabularySize: 50272, maxLength: 2050, width: 2560,
            tokenLength: tokenLengths.tokenLength,
            cachedTokenLength: tokenLengths.cachedTokenLength, layers: 32, MLP: 10240, heads: 32,
            batchSize: 1, usesFlashAttention: usesFlashAttention, injectKeysAndValues: true)
        }
      var kvs = (0..<64).map { _ in graph.variable(.GPU(0), format: .NHWC, shape: [], of: T.self) }
      let lmHead = existingLmHead ?? Dense(count: 50272, noBias: true)
      if existingOpt == nil || existingLmHead == nil {
        let tokensTensor = graph.variable(.GPU(0), format: .NHWC, shape: [4], of: Int32.self)
        let positionsTensor = graph.variable(.GPU(0), format: .NHWC, shape: [36], of: Int32.self)
        let causalAttentionMask = graph.variable(.GPU(0), .NHWC(1, 1, 96, 96), of: T.self)
        opt.maxConcurrency = .limit(4)
        opt.compile(
          (cachedTokenLength: 0, tokenLength: 36),
          inputs: [embedding, tokensTensor, positionsTensor, causalAttentionMask] + kvs)
        let mockOut = graph.variable(.GPU(0), .WC(1, 2560), of: T.self)
        lmHead.compile(inputs: mockOut)
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("opt", model: opt, codec: [.jit, .q6p, .ezm7, .externalData])
          $0.read("lm_head", model: lmHead, codec: [.jit, .q6p, .ezm7, .externalData])
        }
      }
      // Initial prompt.
      var tokenLength = 4
      var cpuTokens = graph.variable(.CPU, format: .NHWC, shape: [4], of: Int32.self)
      var cpuPositions = graph.variable(
        .CPU, format: .NHWC, shape: [32 + tokenLength], of: Int32.self)
      for i in 0..<(32 + tokenLength) {
        cpuPositions[i] = Int32(2 + i)
      }
      cpuTokens[0] = 2  // <bos>
      cpuTokens[1] = 102  // "a"
      cpuTokens[2] = 1345  // "photo"
      cpuTokens[3] = 9  // "of" -> "a photo of"
      var textTokens = [Int]()
      var cachedTokenLength = 0
      var expectedOutputLength = 32 + tokenLength
      var embedding = embedding
      var staticKVCacheSize = 0
      var text = ""
      while true {
        let tokensTensor = cpuTokens.toGPU(0)
        let positionsTensor = cpuPositions.toGPU(0)
        if expectedOutputLength == 1 && cachedTokenLength > staticKVCacheSize {
          staticKVCacheSize = max((staticKVCacheSize * 3) / 2, 64)
          let kvs = (0..<64).map { _ in
            let val = graph.variable(.GPU(0), .WC(staticKVCacheSize, 2560), of: T.self)
            val.full(0)
            return val
          }
          let causalAttentionMask = graph.variable(
            .GPU(0), .NHWC(1, 1, expectedOutputLength, staticKVCacheSize + 1), of: T.self)
          causalAttentionMask.full(0)
          // isEager: true will push the compilation as far as possible up until the real execution. This helps because when isEager: false, things like tensor allocations are pushed to the time of first execution.
          opt.compile(
            (cachedTokenLength: staticKVCacheSize, tokenLength: expectedOutputLength),
            inputs: [embedding, tokensTensor, positionsTensor, causalAttentionMask] + kvs,
            isEager: true
          )
        }
        let cpuCausalAttentionMask = graph.variable(
          .CPU, .NHWC(1, 1, expectedOutputLength, 32 + tokenLength), of: T.self)
        cpuCausalAttentionMask.full(0)
        for i in 0..<(expectedOutputLength - 1) {
          for j in (i + 1)..<(32 + tokenLength) {
            cpuCausalAttentionMask[0, 0, i, j] = -T.greatestFiniteMagnitude
          }
        }
        let causalAttentionMask = cpuCausalAttentionMask.toGPU(0)
        let tuple = opt(
          (cachedTokenLength: cachedTokenLength, tokenLength: expectedOutputLength),
          inputs: embedding, [tokensTensor, positionsTensor, causalAttentionMask] + kvs
        ).map {
          $0.as(
            of: T.self)
        }
        var out = tuple[0]
        kvs = Array(tuple[1..<65])
        let lastOut = out[(expectedOutputLength - 1)..<expectedOutputLength, 0..<2560].copied()
        out = lmHead(inputs: lastOut)[0].as(of: T.self).toCPU()
        var maxIdx: Int = -1
        var maxVal: T = -T.greatestFiniteMagnitude
        for i in 0..<50272 {
          if out[0, i] > maxVal {
            maxVal = out[0, i]
            maxIdx = i
          }
        }
        guard maxIdx >= 0 && maxIdx != 50118 && tokenLength < 2050 - 32 else {
          // Ready to return
          return (text.trimmingCharacters(in: .whitespacesAndNewlines), opt, lmHead)
        }
        if cachedTokenLength == 0 {
          embedding = graph.variable(.GPU(0), format: .NHWC, shape: [], of: T.self)
        }
        cachedTokenLength = 32 + tokenLength
        expectedOutputLength = 1
        cpuTokens = graph.variable(
          .CPU, format: .NHWC, shape: [1], of: Int32.self)
        cpuTokens[0] = Int32(maxIdx)
        textTokens.append(maxIdx)
        if let token = tokenMapping[Int32(maxIdx)] {
          if token.hasPrefix("Ġ") {
            text += " " + token.dropFirst()
          } else {
            text += token
          }
          if !partialHandler(text.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return (text.trimmingCharacters(in: .whitespacesAndNewlines), opt, lmHead)
          }
        }
        cpuPositions = graph.variable(
          .CPU, format: .NHWC, shape: [1], of: Int32.self)
        tokenLength += 1
        cpuPositions[0] = Int32(tokenLength - 1 + 2)
      }
    }
  }
}
