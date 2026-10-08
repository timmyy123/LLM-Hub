import Foundation
import NNC

public func Eva02RotaryPositionEmbedding(height: Int, width: Int, tokenLength: Int, channels: Int)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, height * width + tokenLength, 1, channels))
  let dim0 = channels / 2
  let dim1 = dim0
  let scaleFactor: Double = 16 / 24
  assert(channels % 16 == 0)
  for i in 0..<tokenLength {
    for k in 0..<(dim0 / 2) {
      let theta = 0 * scaleFactor / pow(10_000, Double(k) * 2 / Double(dim0))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<(dim1 / 2) {
      let theta = 0 * scaleFactor / pow(10_000, Double(k) * 2 / Double(dim1))
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + (dim0 / 2)) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x + tokenLength
      for k in 0..<(dim0 / 2) {
        let theta = Double(y) * scaleFactor / pow(10_000, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = Double(x) * scaleFactor / pow(10_000, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
    }
  }
  return rotTensor
}

private func EvaSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, name: "q")
  let tovalues = Dense(count: k * h, name: "v")
  var keys = tokeys(x).reshaped([b, t, h, k])
  var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
  queries = Functional.cmul(left: queries, right: rot).transposed(1, 2)
  keys = Functional.cmul(left: keys, right: rot).transposed(1, 2)
  let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let innerAttnLn = LayerNorm(epsilon: 1e-6, axis: [1], name: "inner_attn_ln")
  let unifyheads = Dense(count: k * h, name: "proj")
  out = unifyheads(innerAttnLn(out))
  return Model([x, rot], [out])
}

private func EvaResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int)
  -> Model
{
  let x = Input()
  let rot = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2], name: "ln1")
  let attention = EvaSelfAttention(prefix: prefix, k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x), rot)
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1], name: "ln2")
  let w1 = Dense(count: MLP, name: "w1")
  let w2 = Dense(count: MLP, name: "w2")
  let ffnLn = LayerNorm(epsilon: 1e-6, axis: [1], name: "ffn_ln")
  let w3 = Dense(count: k * h, name: "w3")
  let residual = out
  out = ln2(out)
  out = residual + w3(ffnLn(w1(out).swish() .* w2(out)))
  return Model([x, rot], [out])
}

public func Eva02VisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type,
  grid: Int, outputChannels: Int, width: Int, MLP: Int, layers: Int, heads: Int, batchSize: Int
) -> Model {
  let x = Input()
  let rot = Input()
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]),
    format: .OIHW, name: "patch_embed")
  var out = conv1(x).reshaped([batchSize, grid * grid, width])
  let classEmbedding = Parameter<T>(.GPU(0), .HWC(1, 1, width), name: "cls_embed")
  let positionalEmbedding = Parameter<T>(
    .GPU(0), .HWC(1, grid * grid + 1, width), name: "pos_embed")
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  var outs = [Model.IO]()
  for i in 0..<layers {
    if [4, 8, 12, 16, 20].contains(i) {
      outs.append(out)
    }
    let block = EvaResidualAttentionBlock(
      prefix: "blocks.\(i)",
      k: width / heads, h: heads, b: batchSize, t: grid * grid + 1, MLP: MLP)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]), rot)
  }
  let norm = LayerNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  let head = Dense(count: outputChannels, name: "head")
  out = head(out)
  return Model([x, rot], [out] + outs)
}
