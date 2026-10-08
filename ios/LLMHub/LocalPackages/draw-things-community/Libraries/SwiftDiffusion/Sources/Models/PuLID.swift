import NNC

private func PuLIDCrossAttentionFixed(
  prefix: String, name: String, k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_norm1")
  let tokeys = Dense(count: k * h, noBias: true, name: "\(name)_k")
  let tovalues = Dense(count: k * h, noBias: true, name: "\(name)_v")
  let out = norm1(x).to(.Float16)
  var keys = ((1 / Float(k).squareRoot().squareRoot()) * tokeys(out)).reshaped([b, t, h, k])
  var values = tovalues(out).reshaped([b, t, h, k])
  if !usesFlashAttention {
    keys = keys.transposed(1, 2)
    values = values.transposed(1, 2)
  }
  return Model([x], [keys, values])
}

func PuLIDFixed(
  batchSize: Int, queries: Int, double: [Int], single: [Int],
  usesFlashAttention: FlashAttentionLevel
)
  -> Model
{
  let x = Input()
  var outs = [Model.IO]()
  var ca: Int = 0
  for i in double {
    let block = PuLIDCrossAttentionFixed(
      prefix: "\(ca)", name: "double_\(i)", k: 2048 / 16, h: 16, b: batchSize, t: queries,
      usesFlashAttention: usesFlashAttention != .none)
    let out = block(x)
    if let last = outs.last {
      out.add(dependencies: [last])
    }
    outs.append(out)
    ca += 1
  }
  for i in single {
    let block = PuLIDCrossAttentionFixed(
      prefix: "\(ca)", name: "single_\(i)", k: 2048 / 16, h: 16, b: batchSize, t: queries,
      usesFlashAttention: usesFlashAttention != .none)
    let out = block(x)
    if let last = outs.last {
      out.add(dependencies: [last])
    }
    outs.append(out)
    ca += 1
  }
  return Model([x], outs)
}

func PuLIDCrossAttentionKeysAndValues(
  prefix: String, name: String, outputDim: Int, k: Int, h: Int, b: Int, t: (Int, Int),
  usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let keys = Input()
  let values = Input()
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_norm2")
  let toqueries = Dense(count: k * h, noBias: true, name: "\(name)_q")
  var queries = toqueries(norm2(x).to(.Float16)).reshaped([b, t.1, h, k])
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    queries = ((1.0 / Float(k).squareRoot().squareRoot()) * queries).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * t.1, t.0])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t.1, t.0])
    out = dot * values
    out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b, t.1, h * k])
  case .scale1:
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1)
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, t.1, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot().squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, t.1, k * h])
  }
  let unifyheads = Dense(count: outputDim, noBias: true, name: "\(name)_to_out")
  out = unifyheads(out)
  return Model([x, keys, values], [out])
}
