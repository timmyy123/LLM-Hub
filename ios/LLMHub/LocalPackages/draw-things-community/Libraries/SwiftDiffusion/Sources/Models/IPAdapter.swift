import DiffusionMappings
import NNC

private func PerceiverAttention(
  prefix: String, k: Int, h: Int, queryDim: Int, b: Int, t: (Int, Int)
) -> Model {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outX = norm1(x)
  let c = Input()
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outC = norm2(c).to(.Float16)
  let outXC = Functional.concat(axis: 1, outX, outC)
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(outC)).reshaped([b, t.1, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t.1, t.0 + t.1])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t.1, t.0 + t.1])
  var out = dot * values
  out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: queryDim, noBias: true)
  out = unifyheads(out)
  return Model([x, c], [out])
}

func ResamplerLayer(prefix: String, k: Int, h: Int, queryDim: Int, b: Int, t: (Int, Int)) -> Model {
  let x = Input()
  let c = Input()
  let attention = PerceiverAttention(
    prefix: prefix + ".0", k: k, h: h, queryDim: queryDim, b: b, t: t)
  var out = c + attention(x, c).reshaped([b, t.1, queryDim]).to(of: c)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
  let fc1 = Dense(count: queryDim * 4, noBias: true)
  let gelu = GELU()
  let fc2 = Dense(count: queryDim, noBias: true)
  out = out + fc2(gelu(fc1(layerNorm(out).to(.Float16)))).to(of: out)
  return Model([x, c], [out])
}

func MLPProjModel(width: Int, outputDim: Int) -> Model {
  let x = Input()
  let linear1 = Dense(count: width)
  var out = linear1(x).GELU()
  let linear2 = Dense(count: outputDim)
  out = linear2(out)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm(out)
  return Model([x], [out])
}

func Resampler<T: TensorNumeric>(
  _ dataType: T.Type,
  inputDim: Int, queryDim: Int, outputDim: Int, headDim: Int, heads: Int, grid: Int, queries: Int,
  layers: Int, batchSize: Int
) -> Model {
  let x = Input()
  let latents = Parameter<T>(.GPU(0), .HWC(1, queries, inputDim))
  let projIn = Dense(count: inputDim)
  let projX = projIn(x)
  let firstLayer = ResamplerLayer(
    prefix: "layers.0", k: headDim, h: heads, queryDim: queryDim, b: batchSize,
    t: (grid * grid + 1, queries))
  var out = firstLayer(projX, latents)
  for i in 1..<layers {
    let layer = ResamplerLayer(
      prefix: "layers.\(i)", k: headDim, h: heads, queryDim: queryDim, b: batchSize,
      t: (grid * grid + 1, queries)
    )
    out = layer(projX, out)
  }
  let projOut = Dense(count: outputDim)
  out = projOut(out)
  let normOut = LayerNorm(epsilon: 1e-5, axis: [2])
  out = normOut(out)
  return Model([x], [out])
}

private func CrossAttentionFixed(
  k: Int, h: Int, b: Int, hw: Int, t: Int, usesFlashAttention: FlashAttentionLevel
)
  -> (Model, Model, Model)
{
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  var keys = tokeys(c).reshaped([b, t, h, k])
  var values = tovalues(c).reshaped([b, t, h, k])
  if usesFlashAttention == .none {
    keys = keys.transposed(1, 2)
    values = values.transposed(1, 2)
  }
  return (tokeys, tovalues, Model([c], [keys, values]))
}

private func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let (tokeys, tovalues, attn2) = CrossAttentionFixed(
    k: k, h: h, b: b, hw: hw, t: t, usesFlashAttention: usesFlashAttention)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn2.to_k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).attn2.to_v.weight"] = [tovalues.weight.name]
    return mapping
  }
  return (mapper, attn2)
}

private func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let c = Input()
  var outs = [Model.IO]()
  let hw = height * width
  var mappers = [ModelWeightMapper]()
  for i in 0..<depth {
    let (mapper, block) = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize, usesFlashAttention: usesFlashAttention)
    outs.append(block(c))
    mappers.append(mapper)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([c], outs))
}

private func BlockLayerFixed(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: Int, intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (mapper, transformer) = SpatialTransformerFixed(
    prefix: "\(prefix).attentions.\(layerStart)",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingLength,
    intermediateSize: channels * 4, usesFlashAttention: usesFlashAttention)
  return (mapper, transformer)
}

private func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: Int,
  attentionBlock: Int, usesFlashAttention: FlashAttentionLevel, c: Model.IO
) -> (ModelWeightMapper, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (mapper, transformer) = SpatialTransformerFixed(
    prefix: "mid_block.attentions.0", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingLength, intermediateSize: channels * 4,
    usesFlashAttention: usesFlashAttention)
  let out = transformer(c)
  return (mapper, out)
}

private func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: Int, attentionRes: [Int: Int],
  usesFlashAttention: FlashAttentionLevel,
  c: Model.IO
) -> (ModelWeightMapper, [Model.IO]) {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for j in 0..<numRepeat {
      if attentionBlock > 0 {
        let (inputMapper, inputLayer) = BlockLayerFixed(
          prefix: "down_blocks.\(i)",
          layerStart: j, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention)
        previousChannel = channel
        outs.append(inputLayer(c))
        mappers.append(inputMapper)
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, outs)
}

private func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: Int, attentionRes: [Int: Int],
  usesFlashAttention: FlashAttentionLevel,
  c: Model.IO
) -> (ModelWeightMapper, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  var mappers = [ModelWeightMapper]()
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for j in 0..<(numRepeat + 1) {
      if attentionBlock > 0 {
        let (outputMapper, outputLayer) = BlockLayerFixed(
          prefix: "up_blocks.\(channels.count - 1 - i)",
          layerStart: j, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention)
        outs.append(outputLayer(c))
        mappers.append(outputMapper)
      }
      layerStart += 1
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, outs)
}

func UNetXLIPFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  embeddingLength: Int, attentionRes: KeyValuePairs<Int, Int>,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let (inputMapper, inputBlocks) = InputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: attentionRes,
    usesFlashAttention: usesFlashAttention, c: c)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (middleMapper, middleBlock) = MiddleBlockFixed(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingLength: embeddingLength, attentionBlock: middleBlockAttentionBlock,
    usesFlashAttention: usesFlashAttention, c: c)
  out.append(middleBlock)
  let (outputMapper, outputBlocks) = OutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: attentionRes,
    usesFlashAttention: usesFlashAttention, c: c)
  out.append(contentsOf: outputBlocks)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    mapping.merge(middleMapper(format)) { v, _ in v }
    mapping.merge(outputMapper(format)) { v, _ in v }
    return mapping
  }
  return (mapper, Model([c], out))
}

func FaceResampler(
  width: Int, IDEmbedDim: Int, outputDim: Int, heads: Int, grid: Int,
  queries: Int, layers: Int, batchSize: Int
) -> Model {
  let x = Input()
  let IDEmbeds = Input()
  let proj0 = Dense(count: IDEmbedDim * 2)
  let proj2 = Dense(count: width * queries)
  let norm = LayerNorm(epsilon: 1e-5, axis: [2])
  let latents = norm(proj2(proj0(IDEmbeds).GELU()).reshaped([batchSize, queries, width]))
  let projIn = Dense(count: width)
  let projX = projIn(x)
  let firstLayer = ResamplerLayer(
    prefix: "perceiver_resampler.layers.0", k: outputDim / heads, h: heads,
    queryDim: outputDim, b: batchSize, t: (grid * grid + 1, queries))
  var out = firstLayer(projX, latents)
  for i in 1..<layers {
    let layer = ResamplerLayer(
      prefix: "perceiver_resampler.layers.\(i)", k: outputDim / heads, h: heads,
      queryDim: outputDim, b: batchSize, t: (grid * grid + 1, queries)
    )
    out = layer(projX, out)
  }
  let projOut = Dense(count: outputDim)
  out = projOut(out)
  let normOut = LayerNorm(epsilon: 1e-5, axis: [2])
  out = latents + normOut(out)
  return Model([IDEmbeds, x], [out])
}

private func PuLIDFormerMapping(prefix: String, channels: Int, outputChannels: Int) -> Model {
  let x = Input()
  let layer0 = Dense(count: channels, name: "\(prefix).0")
  var out = layer0(x)
  let layer1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(prefix).1")
  out = layer1(out)
  let layer2 = LeakyReLU(negativeSlope: 0.01)
  out = layer2(out)
  let layer3 = Dense(count: channels, name: "\(prefix).3")
  out = layer3(out)
  let layer4 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(prefix).4")
  out = layer4(out)
  let layer5 = LeakyReLU(negativeSlope: 0.01)
  out = layer5(out)
  let layer6 = Dense(count: outputChannels, name: "\(prefix).6")
  out = layer6(out)
  return Model([x], [out])
}

func PuLIDFormer<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type,
  width: Int, outputDim: Int, heads: Int, grid: Int, idQueries: Int, queries: Int, layers: Int,
  depth: Int
) -> Model {
  let x = Input()
  let y = (0..<layers).map { _ in Input() }
  let latents = Parameter<T>(.GPU(0), .HWC(1, queries, width), name: "latents")
  let idEmbeddingMapping = PuLIDFormerMapping(
    prefix: "id_embedding_mapping", channels: 1024, outputChannels: 1024 * idQueries)
  var out = idEmbeddingMapping(x).reshaped([1, idQueries, 1024])
  let idFeature = out
  out = Functional.concat(axis: 1, latents, out).to(.Float32)
  for i in 0..<layers {
    let mapping = PuLIDFormerMapping(prefix: "mapping_\(i)", channels: 1024, outputChannels: 1024)
    let vitFeature = mapping(y[i]).reshaped([1, grid * grid + 1, width])
    let ctxFeature = Functional.concat(axis: 1, idFeature, vitFeature)
    for j in 0..<depth {
      let layer = ResamplerLayer(
        prefix: "layers.\(i * depth + j)", k: width / heads, h: heads,
        queryDim: width, b: 1, t: (grid * grid + 1 + idQueries, queries + idQueries)
      )
      out = layer(ctxFeature, out)
    }
  }
  let projOut = Dense(count: outputDim, noBias: true, name: "proj_out")
  out = projOut(
    out.reshaped([1, queries, width], strides: [(queries + idQueries) * width, width, 1])
      .contiguous())
  return Model([x] + y, [out])
}
