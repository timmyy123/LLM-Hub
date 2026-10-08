import Foundation
import NNC

public func Mistral3RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, endAligned: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  for i in 0..<sequenceLength {
    for k in 0..<64 {
      let theta = Double(endAligned + i) * 1.0 / pow(1_000_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i, 0, k * 2] = FloatType(costheta)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
    }
  }
  return rotary
}

public func Ministral3YaRNRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, positionOffset: Int = 0, headDim: Int = 128,
  ropeTheta: Double = 1_000_000.0, factor: Double = 16.0,
  originalMaxPositionEmbeddings: Double = 16_384.0, betaFast: Double = 32.0,
  betaSlow: Double = 1.0, mscale: Double = 1.0, mscaleAllDim: Double = 1.0,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let lowCorrection =
    Double(headDim) * log(originalMaxPositionEmbeddings / (betaFast * 2.0 * Double.pi))
    / (2.0 * log(ropeTheta))
  let highCorrection =
    Double(headDim) * log(originalMaxPositionEmbeddings / (betaSlow * 2.0 * Double.pi))
    / (2.0 * log(ropeTheta))
  let low = max(
    Int(floor(lowCorrection)), 0)
  let high = min(
    Int(ceil(highCorrection)), headDim - 1)
  let attentionFactorNumerator =
    factor <= 1.0 ? 1.0 : 0.1 * mscale * log(factor) + 1.0
  let attentionFactorDenominator =
    factor <= 1.0 ? 1.0 : 0.1 * mscaleAllDim * log(factor) + 1.0
  let attentionFactor = attentionFactorNumerator / attentionFactorDenominator
  let halfDim = headDim / 2
  var invFreq = [Double](repeating: 0, count: halfDim)
  for k in 0..<halfDim {
    let posFreq = pow(ropeTheta, Double(k * 2) / Double(headDim))
    let invFreqExtrapolation = 1.0 / posFreq
    let invFreqInterpolation = 1.0 / (factor * posFreq)
    let rampFactor: Double
    if low == high {
      rampFactor = k >= high ? 1.0 : 0.0
    } else {
      rampFactor = Swift.min(
        Swift.max((Double(k) - Double(low)) / Double(high - low), 0.0), 1.0)
    }
    let invFreqExtrapolationFactor = 1.0 - rampFactor
    invFreq[k] =
      invFreqInterpolation * (1.0 - invFreqExtrapolationFactor)
      + invFreqExtrapolation * invFreqExtrapolationFactor
  }

  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, headDim))
  for i in 0..<sequenceLength {
    let position = Double(positionOffset + i)
    for k in 0..<halfDim {
      let theta = position * invFreq[k]
      rotary[0, i, 0, k * 2] = FloatType(cos(theta) * attentionFactor)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sin(theta) * attentionFactor)
    }
  }
  return rotary
}

private func SelfAttention(
  prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  var values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out: Model.IO
  if usesFlashAttention {
    out = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true)(
        queries, keys, values, causalAttentionMask
      ).reshaped([b * t, h * k])
  } else {
    values = values.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot().squareRoot()) * queries).transposed(1, 2)
    keys = ((1.0 / Float(k).squareRoot().squareRoot()) * keys).transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<hk {
      let query = queries.reshaped(
        [b, h / hk, t, k], offset: [0, i * (h / hk), 0, 0], strides: [h * t * k, t * k, k, 1]
      ).contiguous()
      let key = keys.reshaped(
        [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1]
      ).contiguous()
      let value = values.reshaped(
        [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1]
      ).contiguous()
      var dot = Matmul(transposeB: (2, 3))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot + causalAttentionMask
      dot = dot.reshaped([b * (h / hk) * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h / hk, t, t])
      let out = dot * value
      outs.append(out)
    }
    out = Concat(axis: 1)(outs).reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  }
  let unifyheads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyheads(out)
  return Model([x, rot, causalAttentionMask], [out])
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out]))
}

private func TransformerBlock<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type,
  prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int,
  usesFlashAttention: Bool
)
  -> Model
{
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x).to(T.dataType)
  let attention = SelfAttention(
    prefix: prefix, width: width, k: k, h: h, hk: hk, b: b, t: t,
    usesFlashAttention: usesFlashAttention)
  out = attention(out, rot, causalAttentionMask).to(of: x) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out).to(T.dataType)
  let (_, _, _, ffn) = FeedForward(hiddenSize: width, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out).to(of: residual)
  return Model([x, rot, causalAttentionMask], [out])
}

private func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, embeddingSize: Int
) -> Model {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  return Model([tokens], [embedding])
}

public func Mistral3<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: [Int], noFinalNormalizedOutput: Bool,
  batchSize: Int, usesAdditionalTokens: Bool, usesFlashAttention: Bool
) -> Model {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let additionalTokens: Input? = usesAdditionalTokens ? Input() : nil
  let embedding = TextEmbedding(
    BFloat16.self, batchSize: batchSize, vocabularySize: vocabularySize,
    embeddingSize: width)
  var out = embedding(tokens).to(.Float32)
  let additionalEmbeds = additionalTokens.map { embedding($0).to(T.dataType) }
  var hiddenStates = [Model.IO]()
  for i in 0..<layers {
    let layer = TransformerBlock(
      dataType,
      prefix: "layers.\(i)", width: width, k: 128, h: heads, hk: 8, b: batchSize,
      t: tokenLength, MLP: MLP, usesFlashAttention: usesFlashAttention)
    out = layer(out, rot, causalAttentionMask)
    if outputHiddenStates.contains(i) {
      hiddenStates.append(out.to(T.dataType))
    }
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  if !noFinalNormalizedOutput {
    hiddenStates.append(norm(out).to(T.dataType))
  }
  return Model(
    [tokens, rot, causalAttentionMask] + (additionalTokens.map { [$0] } ?? []),
    hiddenStates + (additionalEmbeds.map { [$0] } ?? []))
}
