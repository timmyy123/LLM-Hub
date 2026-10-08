import Foundation
import NNC
import Tokenizer

public struct MoondreamTextGeneration<T: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  let tokenizer: TiktokenTokenizer
  let usesFlashAttention: Bool
  let question: String
  let version: MoondreamEncode<T>.Version
  public init(
    filePath: String, vocabulary: Data, merges: Data, usesFlashAttention: Bool,
    question: String, version: MoondreamEncode<T>.Version
  ) {
    self.filePath = filePath
    self.usesFlashAttention = usesFlashAttention
    self.question = question
    self.version = version
    tokenizer = TiktokenTokenizer(vocabulary: vocabulary, merges: merges)
  }
}

extension MoondreamTextGeneration {
  private static func removePartialEOS(_ text: String, eos: String) -> String {
    for i in (1..<(eos.count + 1)).reversed() {
      if text.hasSuffix(eos.prefix(i)) {
        return String(text.dropLast(i))
      }
    }
    return text
  }
  public func generate(
    _ embedding: DynamicGraph.Tensor<T>,
    phi existingPhi: ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)>? = nil,
    textEmbedding existingTextEmbedding: DynamicGraph.Tensor<T>? = nil,
    partialHandler: (String) -> Bool
  ) -> (String, ModelBuilder<(cachedTokenLength: Int, tokenLength: Int)>, DynamicGraph.Tensor<T>?) {
    let graph = embedding.graph
    return graph.withNoGrad {
      let phi =
        existingPhi
        ?? ModelBuilder { (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
          return PhiDecoder(
            T.self, vocabularySize: 51_200, width: 2048,
            tokenLength: tokenLengths.tokenLength,
            cachedTokenLength: tokenLengths.cachedTokenLength, layers: 24, MLP: 2048 * 4,
            rotaryDim: 32, heads: 32, batchSize: 1, usesFlashAttention: usesFlashAttention)
        }
      let eos: String
      let before: [Int32]
      let after: [Int32]
      switch version {
      case .moondream1, .moondream2_240306:
        eos = "<END>"
        before = tokenizer.tokenize(text: "<image>", addSpecialTokens: true).0
        after = tokenizer.tokenize(text: "</image>\n\nQuestion: \(question)\n\nAnswer:").0
      case .moondream2_240520:
        eos = "<|endoftext|>"
        before = tokenizer.tokenize(text: "", addSpecialTokens: true).0
        after = tokenizer.tokenize(text: "\n\nQuestion: \(question)\n\nAnswer:").0
      }
      guard
        let textEmbedding = existingTextEmbedding
          ?? {
            var textEmbedding: DynamicGraph.Tensor<T>? = nil
            graph.openStore(
              filePath, flags: .readOnly,
              externalStore: TensorData.externalStore(filePath: filePath)
            ) {
              guard
                let textEmbTensor = $0.read("text_emb", codec: [.q6p, .q8p, .ezm7, .externalData])
              else {
                return
              }
              textEmbedding = graph.variable(
                Tensor<T>(from: textEmbTensor).toGPU(0).reshaped(
                  .WC(textEmbTensor.shape[0], textEmbTensor.shape[1])))
            }
            return textEmbedding
          }()
      else {
        return ("", phi, nil)
      }
      let beforeTensor = Tensor<Int32>(before, .CPU, .C(before.count))
      let beforeEmb = Functional.indexSelect(
        input: textEmbedding, index: graph.variable(beforeTensor.toGPU(0)))
      let afterTensor = Tensor<Int32>(after, .CPU, .C(after.count))
      let afterEmb = Functional.indexSelect(
        input: textEmbedding, index: graph.variable(afterTensor.toGPU(0)))
      var inputEmb = graph.variable(
        .GPU(0), .WC(beforeEmb.shape[0] + embedding.shape[0] + afterEmb.shape[0], 2048), of: T.self)
      inputEmb[0..<beforeEmb.shape[0], 0..<2048] = beforeEmb
      inputEmb[beforeEmb.shape[0]..<(beforeEmb.shape[0] + embedding.shape[0]), 0..<2048] = embedding
      inputEmb[(beforeEmb.shape[0] + embedding.shape[0])..<inputEmb.shape[0], 0..<2048] = afterEmb
      let seqLen = inputEmb.shape[0]
      let kvs = (0..<48).map { _ in
        graph.variable(.GPU(0), .NHWC(1, seqLen + 256, 32, 64), of: T.self)
      }
      var costhetaTensor = graph.variable(.CPU, .NHWC(1, seqLen, 1, 16), of: Float.self)
      var sinthetaTensor = graph.variable(.CPU, .NHWC(1, seqLen, 1, 16), of: Float.self)
      for i in 0..<seqLen {
        for k in 0..<16 {
          let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 32)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          costhetaTensor[0, i, 0, k] = Float(costheta)
          sinthetaTensor[0, i, 0, k] = Float(sintheta)
        }
      }
      var causalAttentionMask = graph.variable(.CPU, .NHWC(1, 1, seqLen, seqLen), of: T.self)
      causalAttentionMask.full(0)
      for i in 0..<(seqLen - 1) {
        for j in (i + 1)..<seqLen {
          causalAttentionMask[0, 0, i, j] = -T.greatestFiniteMagnitude
        }
      }
      var causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
      var costhetaTensorGPU = DynamicGraph.Tensor<T>(from: costhetaTensor).toGPU(0)
      var sinthetaTensorGPU = DynamicGraph.Tensor<T>(from: sinthetaTensor).toGPU(0)
      var currentKvs = kvs.map { $0.reshaped(.NHWC(1, seqLen, 32, 64)) }
      if existingPhi == nil {
        phi.maxConcurrency = .limit(4)
        phi.compile(
          (cachedTokenLength: 0, tokenLength: seqLen),
          inputs: [inputEmb, costhetaTensorGPU, sinthetaTensorGPU, causalAttentionMaskGPU]
            + currentKvs)

        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("phi", model: phi, codec: [.jit, .q6p, .ezm7, .externalData])
        }
      }
      var out = phi(
        (cachedTokenLength: 0, tokenLength: seqLen), inputs: inputEmb,
        [costhetaTensorGPU, sinthetaTensorGPU, causalAttentionMaskGPU] + currentKvs)[0].as(
          of: T.self)
      let top1 = Functional.argmax(out, axis: 1)
      var topK = top1.rawValue.toCPU()[0, 0]
      var ids = [topK]
      causalAttentionMask = graph.variable(.CPU, .NHWC(1, 1, 1, 1025), of: T.self)
      causalAttentionMask.full(0)
      causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
      let maxTokens = 255
      phi.compile(
        (cachedTokenLength: seqLen + maxTokens, tokenLength: 1),
        inputs: [
          inputEmb.reshaped(.WC(1, 2048)), costhetaTensorGPU.reshaped(.NHWC(1, 1, 1, 16)),
          sinthetaTensorGPU.reshaped(.NHWC(1, 1, 1, 16)), causalAttentionMaskGPU,
        ] + kvs, isEager: true)
      var output = tokenizer.decode(ids).trimmingLeadingWhitespace()
      if !partialHandler(Self.removePartialEOS(output, eos: eos)) {
        output = Self.removePartialEOS(output, eos: eos)
        return (output, phi, embedding)
      }
      graph.joined()
      let queueWatermark = DynamicGraph.queueWatermark
      DynamicGraph.queueWatermark = queueWatermark * 8  // Expanding queue watermark to support longer generation.
      defer {
        DynamicGraph.queueWatermark = queueWatermark
      }
      for _ in 0..<maxTokens {
        let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [1], of: Int32.self)
        tokensTensor[0] = topK
        let inputEmb = Functional.indexSelect(input: textEmbedding, index: tokensTensor.toGPU(0))
        costhetaTensor = graph.variable(.CPU, .NHWC(1, 1, 1, 16), of: Float.self)
        sinthetaTensor = graph.variable(.CPU, .NHWC(1, 1, 1, 16), of: Float.self)
        let cachedTokenLength = currentKvs[0].shape[1]
        for k in 0..<16 {
          let theta = Double(cachedTokenLength) * 1.0 / pow(10_000, Double(k) * 2 / 32)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          costhetaTensor[0, 0, 0, k] = Float(costheta)
          sinthetaTensor[0, 0, 0, k] = Float(sintheta)
        }
        causalAttentionMask = graph.variable(
          .CPU, .NHWC(1, 1, 1, cachedTokenLength + 1), of: T.self)
        causalAttentionMask.full(0)
        causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
        costhetaTensorGPU = DynamicGraph.Tensor<T>(from: costhetaTensor).toGPU(0)
        sinthetaTensorGPU = DynamicGraph.Tensor<T>(from: sinthetaTensor).toGPU(0)
        currentKvs = kvs.map {
          $0.reshaped(.NHWC(1, cachedTokenLength + 1, 32, 64))
        }
        out = phi(
          (cachedTokenLength: cachedTokenLength, tokenLength: 1), inputs: inputEmb,
          [costhetaTensorGPU, sinthetaTensorGPU, causalAttentionMaskGPU] + currentKvs
        )[0].as(of: T.self)
        let top1 = Functional.argmax(out, axis: 1)
        topK = top1.rawValue.toCPU()[0, 0]
        ids.append(topK)
        output += tokenizer.decode([topK])
        if output.hasSuffix(eos) {
          output = String(output.dropLast(eos.count))
          break
        }
        if !partialHandler(Self.removePartialEOS(output, eos: eos)) {
          output = Self.removePartialEOS(output, eos: eos)
          break
        }
      }
      return (output, phi, textEmbedding)
    }
  }
}

extension String {
  fileprivate func trimmingLeadingWhitespace() -> String {
    var newString = self
    while let first = newString.first, first.isWhitespace {
      newString = String(newString.dropFirst())
    }
    return newString
  }
}
