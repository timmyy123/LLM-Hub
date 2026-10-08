import DiffusionMappings
import Foundation
import NNC

let Ideogram4ConditionCount = 3 + 34 * 4 + 1

func Ideogram4HasUnconditionalDiT(graph: DynamicGraph, filePath: String) -> Bool {
  var hasUnconditionalDiT = false
  graph.openStore(
    filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
  ) { store in
    hasUnconditionalDiT =
      store.read(like: "__unconditional_dit__[t-input_proj-0-0]") != nil
  }
  return hasUnconditionalDiT
}

public func Ideogram4TimeEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  timestep: Float, dim: Int = 4_608, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  precondition(dim % 2 == 0)
  let half = dim / 2
  var embedding = Tensor<FloatType>(.CPU, .WC(1, dim))
  for i in 0..<half {
    let value = 10_000 * timestep * exp(-log(Float(10_000)) * Float(i) / Float(half - 1))
    embedding[0, i] = FloatType(sin(value))
    embedding[0, i + half] = FloatType(cos(value))
  }
  return embedding
}

public func Ideogram4RotaryPositionEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  textLength: Int, gridHeight: Int, gridWidth: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let headDim = 256
  let half = headDim / 2
  let theta: Double = 5_000_000
  let imagePositionOffset = 65_536
  let imageLength = gridHeight * gridWidth
  var rotary = Tensor<FloatType>(
    .CPU, .NHWC(1, imageLength + textLength, 1, headDim))
  for y in 0..<gridHeight {
    for x in 0..<gridWidth {
      let i = y * gridWidth + x
      for k in 0..<half {
        let position: Int
        if k < 60 && k % 3 == 1 {
          position = imagePositionOffset + y
        } else if k < 60 && k % 3 == 2 {
          position = imagePositionOffset + x
        } else {
          position = imagePositionOffset
        }
        let angle = Double(position) / pow(theta, Double(k * 2) / Double(headDim))
        rotary[0, i, 0, k * 2] = FloatType(cos(angle))
        rotary[0, i, 0, k * 2 + 1] = FloatType(sin(angle))
      }
    }
  }
  for i in 0..<textLength {
    let offset = imageLength + i
    for k in 0..<half {
      let angle = Double(i) / pow(theta, Double(k * 2) / Double(headDim))
      rotary[0, offset, 0, k * 2] = FloatType(cos(angle))
      rotary[0, offset, 0, k * 2 + 1] = FloatType(sin(angle))
    }
  }
  return rotary
}

public func Ideogram4IndicatorIDs(textLength: Int, imageLength: Int) -> Tensor<Int32> {
  var indicator = Tensor<Int32>(.CPU, .C(textLength + imageLength))
  for i in 0..<textLength {
    indicator[i] = 0
  }
  for i in textLength..<(textLength + imageLength) {
    indicator[i] = 1
  }
  return indicator
}

private func Ideogram4Attention(
  prefix: String, batchSize: Int, tokenLength: Int, imageLength: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  precondition(!contextBlockPreOnly || imageLength <= tokenLength)
  let queryLength = contextBlockPreOnly ? imageLength : tokenLength
  let x = Input()
  let rot = Input()
  let q = Dense(count: 4_608, noBias: true, name: "q")
  let k = Dense(count: 4_608, noBias: true, name: "k")
  let v = Dense(count: 4_608, noBias: true, name: "v")
  let queryIn: Model.IO =
    contextBlockPreOnly
    ? x.reshaped([batchSize, queryLength, 4_608], strides: [tokenLength * 4_608, 4_608, 1])
      .contiguous() : x
  var queries = q(queryIn).reshaped([batchSize, queryLength, 18, 256])
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_q")
  queries = normQ(queries)
  var keys = k(x).reshaped([batchSize, tokenLength, 18, 256])
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_k")
  keys = normK(keys)
  let values = v(x).reshaped([batchSize, tokenLength, 18, 256])
  keys = Functional.cmul(left: keys, right: rot)
  let queryRot: Model.IO =
    contextBlockPreOnly
    ? rot.reshaped(
      [1, queryLength, 1, 256],
      strides: [
        tokenLength * 256, 256, 256, 1,
      ]) : rot
  queries = Functional.cmul(left: queries, right: queryRot)
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.permuted(0, 2, 1, 3)
    let transposedQueries = ((1.0 / Float(256).squareRoot()) * queries).permuted(0, 2, 1, 3)
    let transposedValues = values.permuted(0, 2, 1, 3)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.reshaped([batchSize * 18 * queryLength, tokenLength])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, 18, queryLength, tokenLength])
    out = dot * transposedValues
    out = out.reshaped([batchSize, 18, queryLength, 256]).transposed(1, 2).reshaped([
      batchSize, queryLength, 4_608,
    ])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(
      (1.0 / Float(256).squareRoot()) * queries, keys, values
    ).reshaped([batchSize, queryLength, 4_608])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(256).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([
      batchSize, queryLength, 4_608,
    ])
  }
  let o = Dense(count: 4_608, noBias: true, name: "o")
  out = o(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      // The packed official QKV tensor stores interleaved Q/K slices and a plain V slice.
      mapping["\(prefix).attention.qkv.weight"] = ModelWeightElement(
        [q.weight.name, k.weight.name, v.weight.name], offsets: [0, 4_608, 9_216],
        interleavedIndices: [0, 1], numberOfHeads: 18, headDimension: 256)
      mapping["\(prefix).attention.o.weight"] = [o.weight.name]
    case .diffusers:
      mapping["\(prefix).attention.to_q.weight"] = ModelWeightElement(
        [q.weight.name], interleavedIndices: [0], numberOfHeads: 18, headDimension: 256)
      mapping["\(prefix).attention.to_k.weight"] = ModelWeightElement(
        [k.weight.name], interleavedIndices: [0], numberOfHeads: 18, headDimension: 256)
      mapping["\(prefix).attention.to_v.weight"] = [v.weight.name]
      mapping["\(prefix).attention.to_out.0.weight"] = [o.weight.name]
    }
    mapping["\(prefix).attention.norm_q.weight"] = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: 256)
    mapping["\(prefix).attention.norm_k.weight"] = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: 256)
    return mapping
  }
  return (mapper, Model([x, rot], [out]))
}

private func Ideogram4MLP(prefix: String) -> (ModelWeightMapper, Model) {
  let x = Input()
  let w1 = Dense(count: 12_288, noBias: true, name: "w1")
  let w3 = Dense(count: 12_288, noBias: true, name: "w3")
  let w2 = Dense(count: 4_608, noBias: true, name: "w2")
  let out = w2(Functional.swishMul(value: w3(x), gate: w1(x)))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).feed_forward.w1.weight"] = [w1.weight.name]
    mapping["\(prefix).feed_forward.w2.weight"] = [w2.weight.name]
    mapping["\(prefix).feed_forward.w3.weight"] = [w3.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func Ideogram4Block(
  prefix: String, batchSize: Int, tokenLength: Int, imageLength: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let scaleMLP = Input()
  let gateMLP = Input()

  let attentionNorm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "attention_norm1")
  let attnIn = attentionNorm1(x).to(.Float16) .* scaleMSA
  let (attentionMapper, attention) = Ideogram4Attention(
    prefix: prefix, batchSize: batchSize, tokenLength: tokenLength, imageLength: imageLength,
    contextBlockPreOnly: contextBlockPreOnly, usesFlashAttention: usesFlashAttention)
  let attentionNorm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "attention_norm2")
  let xIn: Model.IO =
    contextBlockPreOnly
    ? x.reshaped([batchSize, imageLength, 4_608], strides: [tokenLength * 4_608, 4_608, 1])
      .contiguous() : x
  var out = xIn + (attentionNorm2(attention(attnIn, rot)) .* gateMSA).to(of: xIn)
  let ffnNorm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "ffn_norm1")
  let (mlpMapper, mlp) = Ideogram4MLP(prefix: prefix)
  let ffnNorm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "ffn_norm2")
  out = out + (ffnNorm2(mlp(ffnNorm1(out).to(.Float16) .* scaleMLP)) .* gateMLP).to(of: out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(attentionMapper(format)) { v, _ in v }
    mapping.merge(mlpMapper(format)) { v, _ in v }
    mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
    mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
    mapping["\(prefix).ffn_norm1.weight"] = [ffnNorm1.weight.name]
    mapping["\(prefix).ffn_norm2.weight"] = [ffnNorm2.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot, scaleMSA, gateMSA, scaleMLP, gateMLP], [out]))
}

private func Ideogram4BlockFixed(prefix: String) -> (ModelWeightMapper, Model) {
  let adaln = Input()
  let adaLNs = (0..<4).map { Dense(count: 4_608, name: "ada_ln_\($0)") }
  let scaleMSA = 1 + adaLNs[0](adaln)
  let gateMSA = adaLNs[1](adaln).tanh()
  let scaleMLP = 1 + adaLNs[2](adaln)
  let gateMLP = adaLNs[3](adaln).tanh()
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaln_modulation.weight"] = ModelWeightElement(
      adaLNs.map { $0.weight.name }, offsets: [0, 4_608, 9_216, 13_824])
    mapping["\(prefix).adaln_modulation.bias"] = ModelWeightElement(
      adaLNs.map { $0.bias.name }, offsets: [0, 4_608, 9_216, 13_824])
    return mapping
  }
  return (mapper, Model([adaln], [scaleMSA, gateMSA, scaleMLP, gateMLP]))
}

public func Ideogram4Fixed(timesteps: Int, hasTextConditioning: Bool = true) -> (
  ModelWeightMapper, Model
) {
  precondition(timesteps > 0)
  let imageIndicatorIDs = Input()
  let tEmbed = Input()

  let indicatorEmbedding = Embedding(
    FloatType.self, vocabularySize: 2, embeddingSize: 4_608, name: "indicator_embedding")
  let llmNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "llm_cond_norm")
  let llmProj = Dense(count: 4_608, name: "llm_cond_proj")
  let imageIndicator = indicatorEmbedding(imageIndicatorIDs)

  let inputs: [Model.IO]
  var outs: [Model.IO]
  if hasTextConditioning {
    let textFeatures = Input()
    let textIndicatorIDs = Input()
    let textIndicator = indicatorEmbedding(textIndicatorIDs).reshaped([1, -1, 4_608])
    let textOut = llmProj(llmNorm(textFeatures)) + textIndicator
    inputs = [textFeatures, textIndicatorIDs, imageIndicatorIDs, tEmbed]
    outs = [textOut, imageIndicator]
  } else {
    inputs = [imageIndicatorIDs, tEmbed]
    outs = [imageIndicator]
  }

  let tMlpIn = Dense(count: 4_608, name: "t_embedding_mlp_in")
  let tMlpOut = Dense(count: 4_608, name: "t_embedding_mlp_out")
  let adalnProj = Dense(count: 512, name: "adaln_proj")
  let adaln = adalnProj(tMlpOut(tMlpIn(tEmbed).swish())).swish()

  var blockMappers = [ModelWeightMapper]()
  for i in 0..<34 {
    let (mapper, block) = Ideogram4BlockFixed(prefix: "layers.\(i)")
    outs.append(block(adaln))
    blockMappers.append(mapper)
  }
  let finalAdaln = Dense(count: 4_608, name: "final_ada_ln")
  outs.append(1 + finalAdaln(adaln.swish()))

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    mapping["embed_image_indicator.weight"] = [indicatorEmbedding.weight.name]
    if hasTextConditioning {
      mapping["llm_cond_norm.weight"] = [llmNorm.weight.name]
      mapping["llm_cond_proj.weight"] = [llmProj.weight.name]
      mapping["llm_cond_proj.bias"] = [llmProj.bias.name]
    }
    mapping["t_embedding.mlp_in.weight"] = [tMlpIn.weight.name]
    mapping["t_embedding.mlp_in.bias"] = [tMlpIn.bias.name]
    mapping["t_embedding.mlp_out.weight"] = [tMlpOut.weight.name]
    mapping["t_embedding.mlp_out.bias"] = [tMlpOut.bias.name]
    mapping["adaln_proj.weight"] = [adalnProj.weight.name]
    mapping["adaln_proj.bias"] = [adalnProj.bias.name]
    mapping["final_layer.adaln_modulation.weight"] = [finalAdaln.weight.name]
    mapping["final_layer.adaln_modulation.bias"] = [finalAdaln.bias.name]
    return mapping
  }
  return (mapper, Model(inputs, outs))
}

public func Ideogram4(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  precondition(height % 2 == 0 && width % 2 == 0 && textLength >= 0)
  let h = height / 2
  let w = width / 2
  let imageLength = h * w
  let tokenLength = textLength + imageLength
  let x = Input()
  let imageIndicator = Input()
  let rot = Input()
  let xImage = x.reshaped([batchSize, h, 2, w, 2, 32]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous().reshaped([batchSize, imageLength, 128])
  let inputProj = Dense(count: 4_608, name: "input_proj")
  let imageOut = inputProj(xImage) + imageIndicator.reshaped([1, imageLength, 4_608])
  let textOut: Model.IO? = textLength > 0 ? Input() : nil
  var out =
    textOut.map { Functional.concat(axis: 1, imageOut, $0).to(.Float32) }
    ?? imageOut.to(.Float32)
  let rotResized = rot.reshaped(.NHWC(1, tokenLength, 1, 256))

  var blockMappers = [ModelWeightMapper]()
  var adalns = [Input]()
  let layers = 34
  for i in 0..<layers {
    let (mapper, block) = Ideogram4Block(
      prefix: "layers.\(i)", batchSize: batchSize, tokenLength: tokenLength,
      imageLength: imageLength,
      contextBlockPreOnly: i == layers - 1, usesFlashAttention: usesFlashAttention)
    let blockAdalns = (0..<4).map { _ in Input() }
    out = block([out, rotResized] + blockAdalns)
    adalns.append(contentsOf: blockAdalns)
    blockMappers.append(mapper)
  }

  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let finalScale = Input()
  let finalLinear = Dense(count: 128, name: "final_linear")
  out = -finalLinear(norm(out).to(.Float16) .* finalScale)
  out = out.reshaped([batchSize, h, w, 2, 2, 32]).permuted(0, 1, 3, 2, 4, 5).contiguous()
    .reshaped([
      batchSize, height, width, 32,
    ])

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    mapping["input_proj.weight"] = [inputProj.weight.name]
    mapping["input_proj.bias"] = [inputProj.bias.name]
    mapping["final_layer.linear.weight"] = [finalLinear.weight.name]
    mapping["final_layer.linear.bias"] = [finalLinear.bias.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x] + (textOut.map { [$0] } ?? []) + [imageIndicator, rot] + adalns + [finalScale], [out])
  )
}

private func LoRAIdeogram4Attention(
  prefix: String, batchSize: Int, tokenLength: Int, imageLength: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  precondition(!contextBlockPreOnly || imageLength <= tokenLength)
  let queryLength = contextBlockPreOnly ? imageLength : tokenLength
  let x = Input()
  let rot = Input()
  let q = LoRADense(
    count: 4_608, configuration: configuration, noBias: true, index: layerIndex, name: "q")
  let k = LoRADense(
    count: 4_608, configuration: configuration, noBias: true, index: layerIndex, name: "k")
  let v = LoRADense(
    count: 4_608, configuration: configuration, noBias: true, index: layerIndex, name: "v")
  let queryIn: Model.IO =
    contextBlockPreOnly
    ? x.reshaped([batchSize, queryLength, 4_608], strides: [tokenLength * 4_608, 4_608, 1])
      .contiguous() : x
  var queries = q(queryIn).reshaped([batchSize, queryLength, 18, 256])
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_q")
  queries = normQ(queries)
  var keys = k(x).reshaped([batchSize, tokenLength, 18, 256])
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_k")
  keys = normK(keys)
  let values = v(x).reshaped([batchSize, tokenLength, 18, 256])
  keys = Functional.cmul(left: keys, right: rot)
  let queryRot: Model.IO =
    contextBlockPreOnly
    ? rot.reshaped(
      [1, queryLength, 1, 256],
      strides: [
        tokenLength * 256, 256, 256, 1,
      ]) : rot
  queries = Functional.cmul(left: queries, right: queryRot)
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.permuted(0, 2, 1, 3)
    let transposedQueries = ((1.0 / Float(256).squareRoot()) * queries).permuted(0, 2, 1, 3)
    let transposedValues = values.permuted(0, 2, 1, 3)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.reshaped([batchSize * 18 * queryLength, tokenLength])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, 18, queryLength, tokenLength])
    out = dot * transposedValues
    out = out.reshaped([batchSize, 18, queryLength, 256]).transposed(1, 2).reshaped([
      batchSize, queryLength, 4_608,
    ])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(
      (1.0 / Float(256).squareRoot()) * queries, keys, values
    ).reshaped([batchSize, queryLength, 4_608])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(256).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([
      batchSize, queryLength, 4_608,
    ])
  }
  let o = LoRADense(
    count: 4_608, configuration: configuration, noBias: true, index: layerIndex, name: "o")
  out = o(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix).attention.qkv.weight"] = ModelWeightElement(
        [q.weight.name, k.weight.name, v.weight.name], offsets: [0, 4_608, 9_216],
        interleavedIndices: [0, 1], numberOfHeads: 18, headDimension: 256)
      mapping["\(prefix).attention.o.weight"] = [o.weight.name]
    case .diffusers:
      mapping["\(prefix).attention.to_q.weight"] = ModelWeightElement(
        [q.weight.name], interleavedIndices: [0], numberOfHeads: 18, headDimension: 256)
      mapping["\(prefix).attention.to_k.weight"] = ModelWeightElement(
        [k.weight.name], interleavedIndices: [0], numberOfHeads: 18, headDimension: 256)
      mapping["\(prefix).attention.to_v.weight"] = [v.weight.name]
      mapping["\(prefix).attention.to_out.0.weight"] = [o.weight.name]
    }
    mapping["\(prefix).attention.norm_q.weight"] = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: 256)
    mapping["\(prefix).attention.norm_k.weight"] = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: 256)
    return mapping
  }
  return (mapper, Model([x, rot], [out]))
}

private func LoRAIdeogram4MLP(
  prefix: String, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let w1 = LoRADense(
    count: 12_288, configuration: configuration, noBias: true, index: layerIndex, name: "w1")
  let w3 = LoRADense(
    count: 12_288, configuration: configuration, noBias: true, index: layerIndex, name: "w3")
  let w2 = LoRADense(
    count: 4_608, configuration: configuration, noBias: true, index: layerIndex, name: "w2")
  let out = w2(Functional.swishMul(value: w3(x), gate: w1(x)))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).feed_forward.w1.weight"] = [w1.weight.name]
    mapping["\(prefix).feed_forward.w2.weight"] = [w2.weight.name]
    mapping["\(prefix).feed_forward.w3.weight"] = [w3.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func LoRAIdeogram4Block(
  prefix: String, batchSize: Int, tokenLength: Int, imageLength: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let scaleMLP = Input()
  let gateMLP = Input()

  let attentionNorm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "attention_norm1")
  let attnIn = attentionNorm1(x).to(.Float16) .* scaleMSA
  let (attentionMapper, attention) = LoRAIdeogram4Attention(
    prefix: prefix, batchSize: batchSize, tokenLength: tokenLength, imageLength: imageLength,
    contextBlockPreOnly: contextBlockPreOnly, usesFlashAttention: usesFlashAttention,
    layerIndex: layerIndex, configuration: configuration)
  let attentionNorm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "attention_norm2")
  let xIn: Model.IO =
    contextBlockPreOnly
    ? x.reshaped([batchSize, imageLength, 4_608], strides: [tokenLength * 4_608, 4_608, 1])
      .contiguous() : x
  var out = xIn + (attentionNorm2(attention(attnIn, rot)) .* gateMSA).to(of: xIn)
  let ffnNorm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "ffn_norm1")
  let (mlpMapper, mlp) = LoRAIdeogram4MLP(
    prefix: prefix, layerIndex: layerIndex, configuration: configuration)
  let ffnNorm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "ffn_norm2")
  out = out + (ffnNorm2(mlp(ffnNorm1(out).to(.Float16) .* scaleMLP)) .* gateMLP).to(of: out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(attentionMapper(format)) { v, _ in v }
    mapping.merge(mlpMapper(format)) { v, _ in v }
    mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
    mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
    mapping["\(prefix).ffn_norm1.weight"] = [ffnNorm1.weight.name]
    mapping["\(prefix).ffn_norm2.weight"] = [ffnNorm2.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot, scaleMSA, gateMSA, scaleMLP, gateMLP], [out]))
}

private func LoRAIdeogram4BlockFixed(
  prefix: String, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let adaln = Input()
  let adaLNs = (0..<4).map {
    LoRADense(count: 4_608, configuration: configuration, index: layerIndex, name: "ada_ln_\($0)")
  }
  let scaleMSA = 1 + adaLNs[0](adaln)
  let gateMSA = adaLNs[1](adaln).tanh()
  let scaleMLP = 1 + adaLNs[2](adaln)
  let gateMLP = adaLNs[3](adaln).tanh()
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaln_modulation.weight"] = ModelWeightElement(
      adaLNs.map { $0.weight.name }, offsets: [0, 4_608, 9_216, 13_824])
    mapping["\(prefix).adaln_modulation.bias"] = ModelWeightElement(
      adaLNs.map { $0.bias.name }, offsets: [0, 4_608, 9_216, 13_824])
    return mapping
  }
  return (mapper, Model([adaln], [scaleMSA, gateMSA, scaleMLP, gateMLP]))
}

public func LoRAIdeogram4Fixed(
  timesteps: Int, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  precondition(timesteps > 0)
  let textFeatures = Input()
  let textIndicatorIDs = Input()
  let imageIndicatorIDs = Input()
  let tEmbed = Input()

  let indicatorEmbedding = Embedding(
    FloatType.self, vocabularySize: 2, embeddingSize: 4_608, name: "indicator_embedding")
  let llmNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "llm_cond_norm")
  let llmProj = LoRADense(
    count: 4_608, configuration: LoRAConfiguration, index: 0, name: "llm_cond_proj")
  let textIndicator = indicatorEmbedding(textIndicatorIDs).reshaped([1, -1, 4_608])
  let textOut = llmProj(llmNorm(textFeatures)) + textIndicator
  let imageIndicator = indicatorEmbedding(imageIndicatorIDs)

  let tMlpIn = LoRADense(
    count: 4_608, configuration: LoRAConfiguration, index: 0, name: "t_embedding_mlp_in")
  let tMlpOut = LoRADense(
    count: 4_608, configuration: LoRAConfiguration, index: 0, name: "t_embedding_mlp_out")
  let adalnProj = LoRADense(
    count: 512, configuration: LoRAConfiguration, index: 0, name: "adaln_proj")
  let adaln = adalnProj(tMlpOut(tMlpIn(tEmbed).swish())).swish()

  var blockMappers = [ModelWeightMapper]()
  var outs: [Model.IO] = [textOut, imageIndicator]
  for i in 0..<34 {
    let (mapper, block) = LoRAIdeogram4BlockFixed(
      prefix: "layers.\(i)", layerIndex: i, configuration: LoRAConfiguration)
    outs.append(block(adaln))
    blockMappers.append(mapper)
  }
  let finalAdaln = LoRADense(
    count: 4_608, configuration: LoRAConfiguration, index: 0, name: "final_ada_ln")
  outs.append(1 + finalAdaln(adaln.swish()))

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    mapping["embed_image_indicator.weight"] = [indicatorEmbedding.weight.name]
    mapping["llm_cond_norm.weight"] = [llmNorm.weight.name]
    mapping["llm_cond_proj.weight"] = [llmProj.weight.name]
    mapping["llm_cond_proj.bias"] = [llmProj.bias.name]
    mapping["t_embedding.mlp_in.weight"] = [tMlpIn.weight.name]
    mapping["t_embedding.mlp_in.bias"] = [tMlpIn.bias.name]
    mapping["t_embedding.mlp_out.weight"] = [tMlpOut.weight.name]
    mapping["t_embedding.mlp_out.bias"] = [tMlpOut.bias.name]
    mapping["adaln_proj.weight"] = [adalnProj.weight.name]
    mapping["adaln_proj.bias"] = [adalnProj.bias.name]
    mapping["final_layer.adaln_modulation.weight"] = [finalAdaln.weight.name]
    mapping["final_layer.adaln_modulation.bias"] = [finalAdaln.bias.name]
    return mapping
  }
  return (
    mapper,
    Model([textFeatures, textIndicatorIDs, imageIndicatorIDs, tEmbed], outs, trainable: false)
  )
}

public func LoRAIdeogram4(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  precondition(height % 2 == 0 && width % 2 == 0)
  let h = height / 2
  let w = width / 2
  let imageLength = h * w
  let tokenLength = textLength + imageLength
  let x = Input()
  let textOut = Input()
  let imageIndicator = Input()
  let rot = Input()
  let xImage = x.reshaped([batchSize, h, 2, w, 2, 32]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous().reshaped([batchSize, imageLength, 128])
  let inputProj = LoRADense(
    count: 4_608, configuration: LoRAConfiguration, index: 0, name: "input_proj")
  let imageOut = inputProj(xImage) + imageIndicator.reshaped([1, imageLength, 4_608])
  var out = Functional.concat(axis: 1, imageOut, textOut).to(.Float32)
  let rotResized = rot.reshaped(.NHWC(1, tokenLength, 1, 256))

  var blockMappers = [ModelWeightMapper]()
  var adalns = [Input]()
  let layers = 34
  for i in 0..<layers {
    let (mapper, block) = LoRAIdeogram4Block(
      prefix: "layers.\(i)", batchSize: batchSize, tokenLength: tokenLength,
      imageLength: imageLength,
      contextBlockPreOnly: i == layers - 1, usesFlashAttention: usesFlashAttention,
      layerIndex: i, configuration: LoRAConfiguration)
    let blockAdalns = (0..<4).map { _ in Input() }
    out = block([out, rotResized] + blockAdalns)
    adalns.append(contentsOf: blockAdalns)
    blockMappers.append(mapper)
  }

  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let finalScale = Input()
  let finalLinear = LoRADense(
    count: 128, configuration: LoRAConfiguration, index: 0, name: "final_linear")
  out = -finalLinear(norm(out).to(.Float16) .* finalScale)
  out = out.reshaped([batchSize, h, w, 2, 2, 32]).permuted(0, 1, 3, 2, 4, 5).contiguous()
    .reshaped([
      batchSize, height, width, 32,
    ])

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    mapping["input_proj.weight"] = [inputProj.weight.name]
    mapping["input_proj.bias"] = [inputProj.bias.name]
    mapping["final_layer.linear.weight"] = [finalLinear.weight.name]
    mapping["final_layer.linear.bias"] = [finalLinear.bias.name]
    return mapping
  }
  return (
    mapper,
    Model([x, textOut, imageIndicator, rot] + adalns + [finalScale], [out], trainable: false)
  )
}
