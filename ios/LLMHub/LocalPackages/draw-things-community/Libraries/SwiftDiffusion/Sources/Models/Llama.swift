import Foundation
import NNC

func LlamaRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  for i in 0..<sequenceLength {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(500_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i, 0, k * 2] = FloatType(costheta)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
    }
  }
  return rotary
}

func Llama3RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  let invFreqLlama = (0..<64).map { k in
    let lowFreqWavelen = Double(8_192) / 1.0
    let highFreqWavelen = Double(8_192) / 4.0
    let invFreq = 1.0 / pow(500_000, Double(k) * 2 / 128)
    let wavelen = 2.0 * .pi / invFreq
    var invFreqLlama: Double
    if wavelen > lowFreqWavelen {
      invFreqLlama = invFreq / 8.0
    } else {
      invFreqLlama = invFreq
    }
    let smoothFactor = (Double(8_192) / wavelen - 1.0) / (4.0 - 1.0)
    let smoothInvFreq = (1 - smoothFactor) * invFreqLlama / 8.0 + smoothFactor * invFreqLlama
    if wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen {
      invFreqLlama = smoothInvFreq
    }
    return invFreqLlama
  }
  for i in 0..<sequenceLength {
    for k in 0..<64 {
      let theta = Double(i) * invFreqLlama[k]
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i, 0, k * 2] = FloatType(costheta)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
    }
  }
  return rotary
}

private func SelfAttention(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, usesFlashAttention: Bool
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
    queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    keys = keys.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<hk {
      let query = queries.reshaped(
        [b, h / hk, t, k], offset: [0, i * (h / hk), 0, 0], strides: [h * t * k, t * k, k, 1])
      let key = keys.reshaped(
        [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1])
      let value = values.reshaped(
        [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1])
      var dot = Matmul(transposeB: (2, 3))(query, key) + causalAttentionMask
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([b * (h / hk) * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h / hk, t, t])
      let out = dot * value
      outs.append(out)
    }
    out = Concat(axis: 1)(outs).reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  }
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
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
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func TransformerBlock(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int, usesFlashAttention: Bool
)
  -> Model
{
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let attention = SelfAttention(
    prefix: prefix, k: k, h: h, hk: hk, b: b, t: t, usesFlashAttention: usesFlashAttention)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (_, _, _, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
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

func Llama3<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: (Int, Int, Int),
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Set<Int>, batchSize: Int,
  usesFlashAttention: Bool
) -> Model {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let embedding = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, embeddingSize: width)
  var out = embedding(tokens)
  let textEmbedding: Input?
  if tokenLength.2 > max(tokenLength.0, tokenLength.1) {
    let additionalEmbedding = Input()
    if batchSize == 1 {  // Only conditional side.
      out = Functional.concat(axis: 0, out, additionalEmbedding)
    } else {
      precondition(batchSize == 2)
      out = Functional.concat(
        axis: 0,
        out.reshaped([tokenLength.0, width]).padded(
          .zero, begin: [0, 0], end: [tokenLength.2 - tokenLength.0, 0]),
        out.reshaped(
          [tokenLength.1, width], offset: [max(tokenLength.0, tokenLength.1), 0],
          strides: [width, 1]
        ).contiguous(), additionalEmbedding)
    }
    textEmbedding = additionalEmbedding
  } else {
    textEmbedding = nil
  }
  var hiddenStates = [Model.IO]()
  for i in 0..<layers {
    if outputHiddenStates.contains(i) {
      hiddenStates.append(out)
    }
    let layer = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize,
      t: tokenLength.2, MLP: MLP, usesFlashAttention: usesFlashAttention)
    out = layer(out, rot, causalAttentionMask)
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  return Model(
    [tokens, rot, causalAttentionMask] + (textEmbedding.flatMap { [$0] } ?? []),
    hiddenStates + [out])
}
