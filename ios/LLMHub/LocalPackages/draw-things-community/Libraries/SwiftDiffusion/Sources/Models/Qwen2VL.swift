import Foundation
import NNC

public func Qwen2VLRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, token: DynamicGraph.Tensor<Int32>,
  referenceSizes: [(height: Int, width: Int)],
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  guard !referenceSizes.isEmpty else {
    for i in 0..<sequenceLength {
      for k in 0..<64 {
        let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotary[0, i, 0, k * 2] = FloatType(costheta)
        rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
      }
    }
    return rotary
  }
  // First, find vision pad.
  var visionPad = [Int]()
  var first = true
  for i in 0..<sequenceLength {
    if token[i + sequenceLength] == 151655 {
      if first {
        visionPad.append(i)
        first = false
      }
    } else {
      first = true
    }
  }
  guard !visionPad.isEmpty else {
    for i in 0..<sequenceLength {
      for k in 0..<64 {
        let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotary[0, i, 0, k * 2] = FloatType(costheta)
        rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
      }
    }
    return rotary
  }
  // Go up to vision pad with normal 1d rotary encoding.
  for i in 0..<visionPad[0] {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i, 0, k * 2] = FloatType(costheta)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
    }
  }
  var i = visionPad[0]
  var startIndex = visionPad[0]
  for (n, referenceSize) in referenceSizes.enumerated() {
    for y in 0..<referenceSize.height {
      for x in 0..<referenceSize.width {
        for k in 0..<16 {
          let theta = Double(startIndex) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotary[0, i, 0, k * 2] = FloatType(costheta)
          rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
        }
        for k in 16..<40 {
          let theta = Double(y + startIndex) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotary[0, i, 0, k * 2] = FloatType(costheta)
          rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
        }
        for k in 40..<64 {
          let theta = Double(x + startIndex) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotary[0, i, 0, k * 2] = FloatType(costheta)
          rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
        }
        i += 1
      }
    }
    startIndex += max(referenceSize.height, referenceSize.width)
    let nextLength = n + 1 < visionPad.count ? visionPad[n + 1] : sequenceLength
    while i < nextLength {
      for k in 0..<64 {
        let theta = Double(startIndex) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotary[0, i, 0, k * 2] = FloatType(costheta)
        rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
      }
      startIndex += 1
      i += 1
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
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  var values = tovalues(x).reshaped([b, t, hk, k])
  queries = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1.0 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  var out: Model.IO
  if usesFlashAttention {
    out = ScaledDotProductAttention(
      scale: 1, isCausal: true, hasAttentionMask: true)(
        queries, keys, values, causalAttentionMask
      ).reshaped([b * t, h * k])
  } else {
    values = values.transposed(1, 2)
    queries = queries.transposed(1, 2)
    keys = keys.transposed(1, 2)
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
) -> Model {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let attention = SelfAttention(
    prefix: prefix, k: k, h: h, hk: hk, b: b, t: t, usesFlashAttention: usesFlashAttention)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (_, _, _, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  return Model([x, rot, causalAttentionMask], [out])
}

private func TextEmbedding<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type, injectEmbeddings: Bool, batchSize: Int, vocabularySize: Int, maxLength: Int,
  embeddingSize: Int
) -> Model {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  if injectEmbeddings {
    let tokenMask = Input()
    let injectedEmbeddings = Input()
    // Adding additional reshape to make sure the order between token embed and position embed never switch.
    let finalEmbedding = embedding .* tokenMask + injectedEmbeddings
    return Model([tokens, tokenMask, injectedEmbeddings], [finalEmbedding])
  } else {
    return Model([tokens], [embedding])
  }
}

public func Qwen2VL<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type, injectEmbeddings: Bool, vocabularySize: Int, maxLength: Int, width: Int,
  tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Int?, batchSize: Int,
  usesFlashAttention: Bool
) -> Model {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let embedding = TextEmbedding(
    T.self, injectEmbeddings: injectEmbeddings, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  let tokenMask: Input?
  let injectedEmbeddings: Input?
  var out: Model.IO
  if injectEmbeddings {
    let mask = Input()
    let embeddings = Input()
    out = embedding(tokens, mask, embeddings)
    injectedEmbeddings = embeddings
    tokenMask = mask
  } else {
    out = embedding(tokens)
    injectedEmbeddings = nil
    tokenMask = nil
  }
  var hiddenStates: Model.IO? = nil
  for i in 0..<layers {
    let layer = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: 4, b: batchSize,
      t: tokenLength, MLP: MLP, usesFlashAttention: usesFlashAttention)
    out = layer(out, rot, causalAttentionMask)
    if let outputHiddenStates = outputHiddenStates, outputHiddenStates == i {
      hiddenStates = out
    }
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  if injectEmbeddings, let tokenMask = tokenMask, let injectedEmbeddings = injectedEmbeddings {
    return Model(
      [tokens, rot, causalAttentionMask, tokenMask, injectedEmbeddings],
      (hiddenStates.map { [$0] } ?? []) + [out])
  } else {
    return Model([tokens, rot, causalAttentionMask], (hiddenStates.map { [$0] } ?? []) + [out])
  }
}

private func Qwen2VLViTSelfAttention(
  k: Int, h: Int, b: Int, t: Int, segments: [(Int, Int)], isFullAttention: Bool,
  usesFlashAttention: Bool
) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * h, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * h, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, h, k])
  var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
  var values = tovalues(x).reshaped([b, t, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out: Model.IO
  if isFullAttention {
    if usesFlashAttention {
      out = ScaledDotProductAttention(scale: 1)(queries, keys, values).reshaped([b * t, h * k])
    } else {
      queries = queries.transposed(1, 2)
      keys = keys.transposed(1, 2)
      values = values.transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t, t])
      out = dot * values
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
    }
  } else {
    var offset = 0
    var outs = [Model.IO]()
    for segment in segments {
      var query = queries.reshaped(
        [b * segment.0, segment.1, h, k], offset: [0, offset, 0, 0],
        strides: [segment.1 * h * k, h * k, k, 1]
      ).contiguous()
      var key = keys.reshaped(
        [b * segment.0, segment.1, h, k], offset: [0, offset, 0, 0],
        strides: [segment.1 * h * k, h * k, k, 1]
      ).contiguous()
      var value = values.reshaped(
        [b * segment.0, segment.1, h, k], offset: [0, offset, 0, 0],
        strides: [segment.1 * h * k, h * k, k, 1]
      ).contiguous()
      offset += b * segment.0 * segment.1
      if usesFlashAttention {
        let out = ScaledDotProductAttention(scale: 1)(query, key, value).reshaped([
          b * segment.0 * segment.1, h * k,
        ])
        outs.append(out)
      } else {
        query = query.transposed(1, 2)
        key = key.transposed(1, 2)
        value = value.transposed(1, 2)
        var dot = Matmul(transposeB: (2, 3))(query, key)
        dot = dot.reshaped([b * segment.0 * h * segment.1, segment.1])
        dot = dot.softmax()
        dot = dot.reshaped([b * segment.0, h, segment.1, segment.1])
        var out = dot * value
        out = out.reshaped([b * segment.0, h, segment.1, k]).transposed(1, 2).reshaped([
          b * segment.0 * segment.1, h * k,
        ])
        outs.append(out)
      }
    }
    out = Concat(axis: 0)(outs)
  }
  let unifyheads = Dense(count: k * h, name: "out_proj")
  out = unifyheads(out)
  return (toqueries, tokeys, tovalues, unifyheads, Model([x, rot], [out]))
}

private func Qwen2VLViTFeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func Qwen2VLViTResidualAttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, segments: [(Int, Int)], MLP: Int,
  isFullAttention: Bool, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [2], name: "norm1")
  let (_, _, _, _, attention) = Qwen2VLViTSelfAttention(
    k: k, h: h, b: b, t: t, segments: segments, isFullAttention: isFullAttention,
    usesFlashAttention: usesFlashAttention)
  var out = x.reshaped([b * t, h * k]) + attention(norm1(x), rot)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm2")
  let (_, _, _, ffn) = Qwen2VLViTFeedForward(
    hiddenSize: k * h, intermediateSize: MLP, name: "mlp")
  out = out + ffn(norm2(out))
  return Model([x, rot], [out])
}

func Qwen2VLVisionTransformer(
  gridX: Int, gridY: Int, width: Int, layers: Int, fullAttentionLayers: Set<Int>, heads: Int,
  MLP: Int, batchSize: Int, usesFlashAttention: Bool
) -> Model {
  assert(gridX % 2 == 0)
  assert(gridY % 2 == 0)
  var segments = [(Int, Int)]()  // This is determined by the gridX / gridY. If both are divisible by 8, then we only have one segment, otherwise we need 3~4 segments.
  segments.append(((gridX / 8) * (gridY / 8), 64))
  if gridX % 8 == 0, gridY % 8 == 0 {
    // Don't need anything else.
  } else if gridX % 8 == 0 {
    // Adding parts of gridY that cannot be divided by 8.
    segments.append((gridX / 8, 8 * (gridY % 8)))
  } else if gridY % 8 == 0 {
    // Adding parts of gridX that cannot be divided by 8.
    segments.append((gridY / 8, 8 * (gridX % 8)))
  } else if gridX % 8 == gridY % 8 {
    // The piece on the right and bottom side.
    segments.append(((gridX / 8) + (gridY / 8), 8 * (gridY % 8)))
    // The last piece.
    segments.append((1, (gridY % 8) * (gridX % 8)))
  } else {
    // The piece on the right side.
    segments.append((gridY / 8, 8 * (gridX % 8)))
    // The piece on the bottom side.
    segments.append((gridX / 8, 8 * (gridY % 8)))
    // The last piece.
    segments.append((1, (gridY % 8) * (gridX % 8)))
  }
  let x = Input()
  let rot = Input()
  let conv1 = Dense(count: width, noBias: true, name: "conv_in")
  var out = conv1(x).reshaped([batchSize, gridX * gridY, width])
  for i in 0..<layers {
    let isFullAttention = fullAttentionLayers.contains(i)
    let block = Qwen2VLViTResidualAttentionBlock(
      prefix: "blocks.\(i)", k: width / heads, h: heads, b: batchSize,
      t: gridX * gridY, segments: segments, MLP: MLP, isFullAttention: isFullAttention,
      usesFlashAttention: usesFlashAttention)
    out = block(out.reshaped([batchSize, gridX * gridY, width]), rot)
  }
  let normOut = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_out")
  out = normOut(out).reshaped([gridX * gridY / 4, 4 * width])
  let mlp0 = Dense(count: 5120, name: "merger_mlp_0")
  let mlp1 = Dense(count: 3584, name: "merger_mlp_1")
  out = mlp1(mlp0(out).GELU())
  return Model([x, rot], [out])
}

func Qwen2VLViTRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  gridX: Int, gridY: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, gridX * gridY, 1, 80))
  var i = 0
  for y0 in 0..<(gridY / 8) {
    for x0 in 0..<(gridX / 8) {
      for y1 in 0..<4 {
        for x1 in 0..<4 {
          for y2 in 0..<2 {
            for x2 in 0..<2 {
              for k in 0..<20 {
                let theta = Double(y0 * 8 + y1 * 2 + y2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
                let sintheta = sin(theta)
                let costheta = cos(theta)
                rotary[0, i, 0, k * 2] = FloatType(costheta)
                rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
              }
              for k in 0..<20 {
                let theta = Double(x0 * 8 + x1 * 2 + x2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
                let sintheta = sin(theta)
                let costheta = cos(theta)
                rotary[0, i, 0, 40 + k * 2] = FloatType(costheta)
                rotary[0, i, 0, 40 + k * 2 + 1] = FloatType(sintheta)
              }
              i += 1
            }
          }
        }
      }
    }
  }
  if gridX % 8 > 0 {
    let x0 = gridX / 8
    for y0 in 0..<(gridY / 8) {
      for y1 in 0..<4 {
        for x1 in 0..<(gridX % 8) / 2 {
          for y2 in 0..<2 {
            for x2 in 0..<2 {
              for k in 0..<20 {
                let theta = Double(y0 * 8 + y1 * 2 + y2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
                let sintheta = sin(theta)
                let costheta = cos(theta)
                rotary[0, i, 0, k * 2] = FloatType(costheta)
                rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
              }
              for k in 0..<20 {
                let theta = Double(x0 * 8 + x1 * 2 + x2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
                let sintheta = sin(theta)
                let costheta = cos(theta)
                rotary[0, i, 0, 40 + k * 2] = FloatType(costheta)
                rotary[0, i, 0, 40 + k * 2 + 1] = FloatType(sintheta)
              }
              i += 1
            }
          }
        }
      }
    }
  }
  if gridY % 8 > 0 {
    let y0 = gridY / 8
    for x0 in 0..<(gridX / 8) {
      for y1 in 0..<(gridY % 8) / 2 {
        for x1 in 0..<4 {
          for y2 in 0..<2 {
            for x2 in 0..<2 {
              for k in 0..<20 {
                let theta = Double(y0 * 8 + y1 * 2 + y2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
                let sintheta = sin(theta)
                let costheta = cos(theta)
                rotary[0, i, 0, k * 2] = FloatType(costheta)
                rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
              }
              for k in 0..<20 {
                let theta = Double(x0 * 8 + x1 * 2 + x2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
                let sintheta = sin(theta)
                let costheta = cos(theta)
                rotary[0, i, 0, 40 + k * 2] = FloatType(costheta)
                rotary[0, i, 0, 40 + k * 2 + 1] = FloatType(sintheta)
              }
              i += 1
            }
          }
        }
      }
    }
  }
  if gridY % 8 > 0, gridX % 8 > 0 {
    let y0 = gridY / 8
    let x0 = gridX / 8
    for y1 in 0..<(gridY % 8) / 2 {
      for x1 in 0..<(gridX % 8) / 2 {
        for y2 in 0..<2 {
          for x2 in 0..<2 {
            for k in 0..<20 {
              let theta = Double(y0 * 8 + y1 * 2 + y2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
              let sintheta = sin(theta)
              let costheta = cos(theta)
              rotary[0, i, 0, k * 2] = FloatType(costheta)
              rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
            }
            for k in 0..<20 {
              let theta = Double(x0 * 8 + x1 * 2 + x2) * 1.0 / pow(10_000, Double(k) * 2 / 40)
              let sintheta = sin(theta)
              let costheta = cos(theta)
              rotary[0, i, 0, 40 + k * 2] = FloatType(costheta)
              rotary[0, i, 0, 40 + k * 2 + 1] = FloatType(sintheta)
            }
            i += 1
          }
        }
      }
    }
  }
  return rotary
}
