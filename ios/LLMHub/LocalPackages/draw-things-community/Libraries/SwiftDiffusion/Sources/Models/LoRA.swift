import DiffusionMappings
import NNC

public struct LoRANetworkConfiguration {
  public var rank: Int
  public var scale: Float
  public var highPrecision: Bool
  public var testing: Bool
  public var gradientCheckpointingFeedForward: Bool
  public var gradientCheckpointingTransformerLayer: Bool
  public var keys: [String]?
  public var orthonormalDown: Bool
  public init(
    rank: Int, scale: Float, highPrecision: Bool, testing: Bool = true,
    gradientCheckpointingFeedForward: Bool = false,
    gradientCheckpointingTransformerLayer: Bool = false, keys: [String]? = nil,
    orthonormalDown: Bool = false
  ) {
    self.rank = rank
    self.scale = scale
    self.highPrecision = highPrecision
    self.testing = testing
    self.gradientCheckpointingFeedForward = gradientCheckpointingFeedForward
    self.gradientCheckpointingTransformerLayer = gradientCheckpointingTransformerLayer
    self.keys = keys
    self.orthonormalDown = orthonormalDown
  }
}

public func LoRAConvolution(
  groups: Int, filters: Int, filterSize: [Int], configuration: LoRANetworkConfiguration,
  noBias: Bool = false, hint: Hint = Hint(), format: Convolution.Format? = nil, index: Int? = nil,
  name: String = ""
) -> Model {
  let conv2d = Convolution(
    groups: groups, filters: filters, filterSize: filterSize, noBias: noBias, hint: hint,
    format: format, name: name)
  guard configuration.rank > 0 else {
    return conv2d
  }
  if let keys = configuration.keys {
    let key: String
    if let index = index {
      key = "\(name)-\(index)-"
    } else {
      key = "\(name)-"
    }
    // If cannot find existing key that matches the prefix, return vanilla convolution.
    if !keys.contains(where: { $0.hasPrefix(key) }) {
      return conv2d
    }
  }
  let x = Input()
  let downKey = index.map { "\(name)_lora_down-\($0)" } ?? "\(name)_lora_down"
  let upKey = index.map { "\(name)_lora_up-\($0)" } ?? "\(name)_lora_up"
  let conv2dDown = Convolution(
    groups: groups, filters: configuration.rank, filterSize: filterSize, noBias: true, hint: hint,
    format: format, trainable: configuration.orthonormalDown ? false : true,
    name: name.isEmpty ? "lora_down" : downKey)
  let conv2dUp = Convolution(
    groups: groups, filters: filters, filterSize: [1, 1], noBias: true, hint: Hint(stride: [1, 1]),
    format: format, trainable: true, name: name.isEmpty ? "lora_up" : upKey)
  var out = conv2d(x)
  let downX: Model.IO
  if configuration.highPrecision {
    downX = conv2dDown(x.to(.Float32))
  } else {
    downX = conv2dDown(x)
  }
  if configuration.scale != 1 {
    out =
      out
      + (configuration.highPrecision
        ? conv2dUp(configuration.scale * downX).to(of: x)
        : conv2dUp(configuration.scale * downX))
  } else {
    out =
      out
      + (configuration.highPrecision
        ? conv2dUp(downX).to(of: x) : conv2dUp(downX))
  }
  return Model([x], [out])
}

public func LoRADense(
  count: Int, configuration: LoRANetworkConfiguration, noBias: Bool = false,
  flags: Functional.GEMMFlag = [], prefix: String? = nil, index: Int? = nil,
  startIndex: Int = 0, name: String = ""
) -> Model {
  let dense = Dense(count: count, noBias: noBias, flags: flags, name: name)
  dense.startIndex = startIndex
  guard configuration.rank > 0 else {
    return dense
  }
  if let keys = configuration.keys {
    let key: String
    if let prefix = prefix {
      if let index = index {
        key = "\(prefix)-\(index)-\(name)-"
      } else {
        key = "\(prefix)-"
      }
    } else {
      if let index = index {
        key = "\(name)-\(index)-"
      } else {
        key = "\(name)-"
      }
    }
    // If cannot find existing key that matches the prefix, return vanilla dense layer.
    if !keys.contains(where: { $0.hasPrefix(key) }) {
      return dense
    }
  }
  let x = Input()
  let downKey = index.map { "\(name)_lora_down-\($0)" } ?? "\(name)_lora_down"
  let upKey = index.map { "\(name)_lora_up-\($0)" } ?? "\(name)_lora_up"
  let denseDown = Dense(
    count: configuration.rank, noBias: true,
    trainable: configuration.orthonormalDown ? false : true,
    name: name.isEmpty ? "lora_down" : downKey)
  let denseUp = Dense(
    count: count, noBias: true, trainable: true, name: name.isEmpty ? "lora_up" : upKey)
  var out = dense(x)
  let downX: Model.IO
  if configuration.highPrecision {
    downX = denseDown(x.to(.Float32))
  } else {
    downX = denseDown(x)
  }
  if configuration.scale != 1 {
    out =
      out
      + (configuration.highPrecision
        ? denseUp(configuration.scale * downX).to(of: x)
        : denseUp(configuration.scale * downX))
  } else {
    out =
      out
      + (configuration.highPrecision
        ? denseUp(downX).to(of: x) : denseUp(downX))
  }
  return Model([x], [out])
}

public func LoRASegmentedDense(
  segments: Int,
  count: Int, configuration: LoRANetworkConfiguration, noBias: Bool = false,
  flags: Functional.GEMMFlag = [], index: Int? = nil, name: String = ""
) -> Model {
  let dense = SegmentedDense(
    segments: segments, count: count, noBias: noBias, flags: flags, name: name)
  guard configuration.rank > 0 else {
    return dense
  }
  if let keys = configuration.keys {
    let key: String
    if let index = index {
      key = "\(name)-\(index)-"
    } else {
      key = "\(name)-"
    }
    // If cannot find existing key that matches the prefix, return vanilla dense layer.
    if !keys.contains(where: { $0.hasPrefix(key) }) {
      return dense
    }
  }
  let x = Input()
  let expertIds = Input()
  let expertCounts = Input()
  let downKey = index.map { "\(name)_lora_down-\($0)" } ?? "\(name)_lora_down"
  let upKey = index.map { "\(name)_lora_up-\($0)" } ?? "\(name)_lora_up"
  let denseDown = SegmentedDense(
    segments: segments,
    count: configuration.rank, noBias: true,
    trainable: configuration.orthonormalDown ? false : true,
    name: name.isEmpty ? "lora_down" : downKey)
  let denseUp = SegmentedDense(
    segments: segments,
    count: count, noBias: true, trainable: true, name: name.isEmpty ? "lora_up" : upKey)
  var out = dense(x, expertIds, expertCounts)
  let downX: Model.IO
  if configuration.highPrecision {
    downX = denseDown(x.to(.Float32), expertIds, expertCounts)
  } else {
    downX = denseDown(x, expertIds, expertCounts)
  }
  if configuration.scale != 1 {
    out =
      out
      + (configuration.highPrecision
        ? denseUp(configuration.scale * downX, expertIds, expertCounts).to(of: x)
        : denseUp(configuration.scale * downX, expertIds, expertCounts))
  } else {
    out =
      out
      + (configuration.highPrecision
        ? denseUp(downX, expertIds, expertCounts).to(of: x)
        : denseUp(downX, expertIds, expertCounts))
  }
  return Model([x, expertIds, expertCounts], [out])
}

/// Text Model

func LoRACLIPAttention(
  k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let causalAttentionMask = Input()
  let tokeys = LoRADense(count: k * h, configuration: LoRAConfiguration)
  let toqueries = LoRADense(count: k * h, configuration: LoRAConfiguration)
  let tovalues = LoRADense(count: k * h, configuration: LoRAConfiguration)
  if usesFlashAttention {
    let queries = toqueries(x).reshaped([b, t, h, k]).identity().identity()
    let keys = tokeys(x).reshaped([b, t, h, k]).identity()
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true)
    var out = scaledDotProductAttention(queries, keys, values, causalAttentionMask).reshaped([
      b * t, h * k,
    ])
    let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
    out = unifyheads(out)
    return Model([x, causalAttentionMask], [out])
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
    dot = dot.reshaped([b * h * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    var out = dot * values
    out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
    let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
    out = unifyheads(out)
    return Model([x, causalAttentionMask], [out])
  }
}

func LoRACLIPMLP(
  hiddenSize: Int, intermediateSize: Int, LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let fc1 = LoRADense(count: intermediateSize, configuration: LoRAConfiguration)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = LoRADense(count: hiddenSize, configuration: LoRAConfiguration)
  out = fc2(out)
  return Model([x], [out])
}

func LoRACLIPEncoderLayer(
  k: Int, h: Int, b: Int, t: Int, intermediateSize: Int, usesFlashAttention: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let causalAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let attention = LoRACLIPAttention(
    k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  out = attention(out, causalAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let mlp = LoRACLIPMLP(
    hiddenSize: k * h, intermediateSize: intermediateSize, LoRAConfiguration: LoRAConfiguration)
  out = mlp(out) + residual
  return Model([x, causalAttentionMask], [out])
}

public func LoRACLIPTextModel<T: TensorNumeric>(
  _ dataType: T.Type, injectEmbeddings: Bool,
  vocabularySize: Int, maxLength: Int, maxTokenLength: Int, embeddingSize: Int, numLayers: Int,
  numHeads: Int, batchSize: Int, intermediateSize: Int, usesFlashAttention: Bool,
  LoRAConfiguration: LoRANetworkConfiguration, noFinalLayerNorm: Bool = false
) -> Model {
  let tokens = Input()
  let positions = Input()
  let causalAttentionMask = Input()
  let (_, _, embedding) = CLIPTextEmbedding(
    T.self, injectEmbeddings: injectEmbeddings, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, maxTokenLength: maxTokenLength,
    embeddingSize: embeddingSize)
  let embedMask: Input?
  let injectedEmbeddings: Input?
  var out: Model.IO
  if injectEmbeddings {
    let mask = Input()
    let embeddings = Input()
    out = embedding(tokens, positions, mask, embeddings)
    embedMask = mask
    injectedEmbeddings = embeddings
  } else {
    out = embedding(tokens, positions)
    embedMask = nil
    injectedEmbeddings = nil
  }
  let k = embeddingSize / numHeads
  for _ in 0..<numLayers {
    let encoderLayer =
      LoRACLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxTokenLength, intermediateSize: intermediateSize,
        usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
    out = encoderLayer(out, causalAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  if !noFinalLayerNorm {
    out = finalLayerNorm(out)
  }
  let model: Model
  if injectEmbeddings, let embedMask = embedMask, let injectedEmbeddings = injectedEmbeddings {
    model = Model(
      [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings], [out],
      trainable: false)
  } else {
    model = Model([tokens, positions, causalAttentionMask], [out], trainable: false)
  }
  return model
}

func LoRAOpenCLIPMLP(
  hiddenSize: Int, intermediateSize: Int, LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let fc1 = LoRADense(count: intermediateSize, configuration: LoRAConfiguration)
  var out = fc1(x)
  out = GELU()(out)
  let fc2 = LoRADense(count: hiddenSize, configuration: LoRAConfiguration)
  out = fc2(out)
  return Model([x], [out])
}

func LoRAOpenCLIPEncoderLayer(
  k: Int, h: Int, b: Int, t: Int, intermediateSize: Int, usesFlashAttention: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let causalAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let attention = LoRACLIPAttention(
    k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  out = attention(out, causalAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let mlp = LoRAOpenCLIPMLP(
    hiddenSize: k * h, intermediateSize: intermediateSize, LoRAConfiguration: LoRAConfiguration)
  out = mlp(out) + residual
  return Model([x, causalAttentionMask], [out])
}

public func LoRAOpenCLIPTextModel<T: TensorNumeric>(
  _ dataType: T.Type, injectEmbeddings: Bool,
  vocabularySize: Int, maxLength: Int, maxTokenLength: Int, embeddingSize: Int, numLayers: Int,
  numHeads: Int, batchSize: Int, intermediateSize: Int, usesFlashAttention: Bool,
  LoRAConfiguration: LoRANetworkConfiguration, outputPenultimate: Bool = false
) -> Model {
  let tokens = Input()
  let positions = Input()
  let causalAttentionMask = Input()
  let (_, _, embedding) = CLIPTextEmbedding(
    T.self, injectEmbeddings: injectEmbeddings, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, maxTokenLength: maxTokenLength,
    embeddingSize: embeddingSize)
  let embedMask: Input?
  let injectedEmbeddings: Input?
  var out: Model.IO
  if injectEmbeddings {
    let mask = Input()
    let embeddings = Input()
    out = embedding(tokens, positions, mask, embeddings)
    embedMask = mask
    injectedEmbeddings = embeddings
  } else {
    out = embedding(tokens, positions)
    embedMask = nil
    injectedEmbeddings = nil
  }
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 && outputPenultimate {
      penultimate = out
    }
    let encoderLayer =
      LoRAOpenCLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxTokenLength, intermediateSize: intermediateSize,
        usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
    out = encoderLayer(out, causalAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  let model: Model
  if injectEmbeddings, let embedMask = embedMask, let injectedEmbeddings = injectedEmbeddings {
    if let penultimate = penultimate {
      model = Model(
        [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings],
        [penultimate, out], trainable: false)
    } else {
      model = Model(
        [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings], [out],
        trainable: false)
    }
  } else {
    if let penultimate = penultimate {
      model = Model(
        [tokens, positions, causalAttentionMask], [penultimate, out], trainable: false)
    } else {
      model = Model([tokens, positions, causalAttentionMask], [out], trainable: false)
    }
  }
  return model
}

/// UNet

func LoRATimeEmbed(modelChannels: Int, LoRAConfiguration: LoRANetworkConfiguration) -> (
  Model, Model, Model
) {
  let x = Input()
  let fc0 = LoRADense(count: modelChannels * 4, configuration: LoRAConfiguration)
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: modelChannels * 4, configuration: LoRAConfiguration)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func LoRAResBlock(
  b: Int, outChannels: Int, skipConnection: Bool, LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, Model, Model, Model, Model, Model?, Model) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  var out = inLayerNorm(x)
  out = out.swish()
  let inLayerConv2d = LoRAConvolution(
    groups: 1, filters: outChannels, filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = inLayerConv2d(out)
  let embLayer = LoRADense(count: outChannels, configuration: LoRAConfiguration)
  var embOut = emb.swish()
  embOut = embLayer(embOut).reshaped([b, 1, 1, outChannels])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outLayerNorm(out)
  out = out.swish()
  // Dropout if needed in the future (for training).
  let outLayerConv2d = LoRAConvolution(
    groups: 1, filters: outChannels, filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let skipModel: Model?
  if skipConnection {
    let skip = LoRAConvolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], configuration: LoRAConfiguration,
      hint: Hint(stride: [1, 1]), format: .OIHW)
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = nil
  }
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel,
    Model([x, emb], [out])
  )
}

func LoRASelfAttention(
  k: Int, h: Int, b: Int, hw: Int, upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
)
  -> (Model, Model, Model, Model, Model)
{
  let x = Input()
  let tokeys = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  let toqueries = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  let tovalues = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  if usesFlashAttention != .none {
    var queries: Model.IO
    if usesFlashAttention == .scale1 {
      queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k]).identity()
    } else {
      queries = toqueries(x).reshaped([b, hw, h, k]).identity().identity()
    }
    let keys = tokeys(x).reshaped([b, hw, h, k]).identity()
    let values = tovalues(x).reshaped([b, hw, h, k])
    let scaledDotProductAttention: ScaledDotProductAttention
    if usesFlashAttention == .scale1 {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1, flags: upcastAttention ? [] : [.Float16])
    } else {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: upcastAttention
          ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
    }
    var out = scaledDotProductAttention(queries, keys, values)
    let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
    out = unifyheads(out.reshaped([b, hw, h * k]))
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  } else if !LoRAConfiguration.testing {  // For training, we don't break it down by b * h.
    var keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2)
    var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    if upcastAttention {
      keys = keys.to(.Float32)
      queries = queries.to(.Float32)
    }
    let values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * hw, hw])
    dot = dot.softmax()
    if upcastAttention {
      dot = dot.to(of: values)
    }
    dot = dot.reshaped([b, h, hw, hw])
    var out = dot * values
    out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
    let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  } else {
    let keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      var key = keys.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      var query = queries.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      if upcastAttention {
        key = key.to(.Float32)
        query = query.to(.Float32)
      }
      let value = values.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      var dot = Matmul(transposeB: (1, 2))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([hw, hw])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: value)
      }
      dot = dot.reshaped([1, hw, hw])
      outs.append(dot * value)
    }
    var out = Concat(axis: 0)(outs)
    out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
    let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  }
}

func LoRACrossAttention(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let c = Input()
  let tokeys = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  let toqueries = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  let tovalues = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  if usesFlashAttention != .none {
    if b == 1 || t.0 == t.1 {
      let t = t.0
      let queries = toqueries(x).reshaped([b, hw, h, k]).identity().identity()
      let keys = tokeys(c).reshaped([b, t, h, k]).identity()
      let values = tovalues(c).reshaped([b, t, h, k])
      if injectIPAdapterLengths.count > 0 {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention
            ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
        var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, h * k])
        var ipKVs = [Input]()
        for _ in injectIPAdapterLengths {
          let ipKeys = Input()
          let ipValues = Input()
          let scaledDotProductAttention = ScaledDotProductAttention(
            scale: 1.0 / Float(k).squareRoot(),
            flags: upcastAttention
              ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
          out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
          ipKVs.append(contentsOf: [ipKeys, ipValues])
        }
        let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
        out = unifyheads(out)
        return Model([x, c] + ipKVs, [out])
      } else {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention
            ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
        var out = scaledDotProductAttention(queries, keys, values)
        let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
        out = unifyheads(out.reshaped([b, hw, h * k]))
        return Model([x, c], [out])
      }
    } else {
      var queries = toqueries(x).reshaped([b, hw, h, k]).identity()
      var keys = tokeys(c).identity()
      var values = tovalues(c)
      let valueType = values
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
        values = values.to(.Float32)
      }
      let b0 = b / 2
      let out0: Model.IO
      if b0 == 1 || t.0 >= t.1 {
        let keys0 = keys.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries0 = queries.reshaped([b0, hw, h, k])
        let values0 = values.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out0 = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention
            ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))(
            queries0, keys0, values0
          ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys0 = keys.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries0 = queries.reshaped(
            [1, hw, h, k], offset: [i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values0 = values.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : [.Float16])(
                queries0, keys0, values0
              ).reshaped([1, hw, h * k]))
        }
        out0 = Concat(axis: 0)(outs)
      }
      let out1: Model.IO
      if b0 == 1 || t.1 >= t.0 {
        let keys1 = keys.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries1 = queries.reshaped(
          [b0, hw, h, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
        let values1 = values.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out1 = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : [.Float16])(
            queries1, keys1, values1
          ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys1 = keys.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries1 = queries.reshaped(
            [1, hw, h, k], offset: [b0 + i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values1 = values.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : [.Float16])(
                queries1, keys1, values1
              ).reshaped([1, hw, h * k]))
        }
        out1 = Concat(axis: 0)(outs)
      }
      var out = Functional.concat(axis: 0, out0, out1)
      var ipKVs = [Input]()
      for _ in injectIPAdapterLengths {
        let ipKeys = Input()
        let ipValues = Input()
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : [.Float16])
        out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
        ipKVs.append(contentsOf: [ipKeys, ipValues])
      }
      if upcastAttention {
        out = out.to(of: valueType)
      }
      let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
      out = unifyheads(out)
      return Model([x, c], [out])
    }
  } else {
    var out: Model.IO
    var keys = tokeys(c)
    var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    if b == 1 || t.0 == t.1 {
      let t = t.0
      keys = keys.reshaped([b, t, h, k]).transposed(1, 2)
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
      }
      let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * hw, t])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: values)
      }
      dot = dot.reshaped([b, h, hw, t])
      var out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
      out = unifyheads(out)
      return Model([x, c], [out])
    } else {
      let values = tovalues(c)
      let b0 = b / 2
      var keys0 = keys.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries0 = queries.reshaped([b0, h, hw, k])
      if upcastAttention {
        keys0 = keys0.to(.Float32)
        queries0 = queries0.to(.Float32)
      }
      let values0 = values.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot0 = Matmul(transposeB: (2, 3))(queries0, keys0)
      dot0 = dot0.reshaped([b0 * h * hw, t.0])
      dot0 = dot0.softmax()
      if upcastAttention {
        dot0 = dot0.to(of: values)
      }
      dot0 = dot0.reshaped([b0, h, hw, t.0])
      var out0 = dot0 * values0
      out0 = out0.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      var keys1 = keys.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries1 = queries.reshaped(
        [b0, h, hw, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, hw * k, k, 1])
      if upcastAttention {
        keys1 = keys1.to(.Float32)
        queries1 = queries1.to(.Float32)
      }
      let values1 = values.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot1 = Matmul(transposeB: (2, 3))(queries1, keys1)
      dot1.add(dependencies: [out0])
      dot1 = dot1.reshaped([b0 * h * hw, t.1])
      dot1 = dot1.softmax()
      if upcastAttention {
        dot1 = dot1.to(of: values)
      }
      dot1 = dot1.reshaped([b0, h, hw, t.1])
      var out1 = dot1 * values1
      out1 = out1.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      out = Functional.concat(axis: 0, out0, out1)
    }
    var ipKVs = [Input]()
    for injectIPAdapterLength in injectIPAdapterLengths {
      let ipKeys = Input()
      let ipValues = Input()
      var dot = Matmul(transposeB: (2, 3))(queries, ipKeys)
      dot = dot.reshaped([b * h * hw, injectIPAdapterLength])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: queries)
      }
      dot = dot.reshaped([b, h, hw, injectIPAdapterLength])
      out = out + (dot * ipValues).transposed(1, 2).reshaped([b, hw, h * k])
      ipKVs.append(contentsOf: [ipKeys, ipValues])
    }
    let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
    out = unifyheads(out)
    return Model([x, c], [out])
  }
}

func LoRAFeedForward(
  hiddenSize: Int, intermediateSize: Int, LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, Model, Model, Model) {
  let x = Input()
  let fc10 = LoRADense(count: intermediateSize, configuration: LoRAConfiguration)
  let fc11 = LoRADense(count: intermediateSize, configuration: LoRAConfiguration)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = LoRADense(count: hiddenSize, configuration: LoRAConfiguration)
  out = fc2(out)
  return (fc10, fc11, fc2, Model([x], [out]))
}

func LoRABasicTransformerBlock(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let c = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (_, _, _, _, attn1) = LoRASelfAttention(
    k: k, h: h, b: b, hw: hw, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let attn2 = LoRACrossAttention(
    k: k, h: h, b: b, hw: hw, t: t, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  out = attn2([out, c] + ipKVs) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (_, _, _, ff) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, LoRAConfiguration: LoRAConfiguration)
  if LoRAConfiguration.gradientCheckpointingFeedForward {
    ff.gradientCheckpointing = true
  }
  out = ff(out) + residual
  return Model([x, c] + ipKVs, [out])
}

func LoRASpatialTransformer(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let c = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm(x)
  let projIn = LoRAConvolution(
    groups: 1, filters: k * h, filterSize: [1, 1], configuration: LoRAConfiguration, format: .OIHW)
  let hw = height * width
  out = projIn(out).reshaped([b, hw, k * h])
  let block = LoRABasicTransformerBlock(
    k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  out = block([out, c] + ipKVs).reshaped([b, height, width, k * h])
  let projOut = LoRAConvolution(
    groups: 1, filters: ch, filterSize: [1, 1], configuration: LoRAConfiguration, format: .OIHW)
  out = projOut(out) + x
  return Model([x, c] + ipKVs, [out])
}

func LoRABlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Bool, channels: Int, numHeads: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let emb = Input()
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (_, _, _, _, _, _, resBlock) =
    LoRAResBlock(
      b: batchSize, outChannels: channels, skipConnection: skipConnection,
      LoRAConfiguration: LoRAConfiguration)
  var out = resBlock(x, emb)
  if attentionBlock {
    let c = Input()
    let transformer = LoRASpatialTransformer(
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      t: embeddingLength, intermediateSize: channels * 4,
      injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
      usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
    let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
    out = transformer([out, c] + ipKVs)
    return Model([x, emb, c] + ipKVs, [out])
  } else {
    return Model([x, emb], [out])
  }
}

func LoRAMiddleBlock(
  channels: Int, numHeads: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration,
  x: Model.IO, emb: Model.IO, c: Model.IO
) -> (Model.IO, [Input]) {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (_, _, _, _, _, _, resBlock1) =
    LoRAResBlock(
      b: batchSize, outChannels: channels, skipConnection: false,
      LoRAConfiguration: LoRAConfiguration)
  var out = resBlock1(x, emb)
  let transformer = LoRASpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingLength,
    intermediateSize: channels * 4, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  out = transformer([out, c] + ipKVs)
  let (_, _, _, _, _, _, resBlock2) =
    LoRAResBlock(
      b: batchSize, outChannels: channels, skipConnection: false,
      LoRAConfiguration: LoRAConfiguration)
  out = resBlock2(out, emb)
  return (out, ipKVs)
}

private func LoRAInputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingLength: (Int, Int), attentionRes: Set<Int>, injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration,
  x: Model.IO, emb: Model.IO, c: Model.IO, adapters: [Model.IO]
) -> ([Model.IO], Model.IO, [Input]) {
  let conv2d = LoRAConvolution(
    groups: 1, filters: 320, filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<numRepeat {
      let inputLayer = LoRABlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        LoRAConfiguration: LoRAConfiguration)
      previousChannel = channel
      if attentionBlock {
        let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
        out = inputLayer([out, emb, c] + ipKVs)
        kvs.append(contentsOf: ipKVs)
      } else {
        out = inputLayer(out, emb)
      }
      if j == numRepeat - 1 && adapters.count == channels.count {
        out = out + adapters[i]
      }
      passLayers.append(out)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = LoRAConvolution(
        groups: 1, filters: channel, filterSize: [3, 3], configuration: LoRAConfiguration,
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append(out)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  return (passLayers, out, kvs)
}

func LoRAOutputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingLength: (Int, Int), attentionRes: Set<Int>, injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration,
  x: Model.IO, emb: Model.IO,
  c: Model.IO, inputs: [Model.IO]
) -> (Model.IO, [Input]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 3)(out, inputs[inputIdx])
      inputIdx -= 1
      let outputLayer = LoRABlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        LoRAConfiguration: LoRAConfiguration)
      if attentionBlock {
        let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
        out = outputLayer([out, emb, c] + ipKVs)
        kvs.append(contentsOf: ipKVs)
      } else {
        out = outputLayer(out, emb)
      }
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = LoRAConvolution(
          groups: 1, filters: channel, filterSize: [3, 3], configuration: LoRAConfiguration,
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW
        )
        out = conv2d(out)
      }
      layerStart += 1
    }
  }
  return (out, kvs)
}

public func LoRAUNet(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  usesFlashAttention: FlashAttentionLevel, injectControls: Bool, injectT2IAdapters: Bool,
  injectIPAdapterLengths: [Int], LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  var injectedControls = [Model.IO]()
  if injectControls {
    injectedControls = (0..<13).map { _ in Input() }
  }
  var injectedT2IAdapters = [Model.IO]()
  if injectT2IAdapters {
    injectedT2IAdapters = (0..<4).map { _ in Input() }
  }
  let (_, _, timeEmbed) = LoRATimeEmbed(modelChannels: 320, LoRAConfiguration: LoRAConfiguration)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  var (inputs, inputBlocks, inputKVs) = LoRAInputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, x: x, emb: emb, c: c, adapters: injectedT2IAdapters)
  var out = inputBlocks
  let (middleBlock, middleKVs) = LoRAMiddleBlock(
    channels: 1280, numHeads: 8, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration, x: out, emb: emb,
    c: c)
  out = middleBlock
  if injectControls {
    out = out + injectedControls[12]
    precondition(inputs.count + 1 == injectedControls.count)
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + injectedControls[i]
    }
  }
  let (outputBlocks, outputKVs) = LoRAOutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: attentionRes, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: false, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, x: out, emb: emb, c: c, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outNorm(out)
  out = out.swish()
  let outConv2d = LoRAConvolution(
    groups: 1, filters: 4, filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = outConv2d(out)
  var modelInputs: [Model.IO] = [x, t_emb, c]
  modelInputs.append(contentsOf: inputKVs)
  modelInputs.append(contentsOf: middleKVs)
  modelInputs.append(contentsOf: outputKVs)
  modelInputs.append(contentsOf: injectedControls)
  modelInputs.append(contentsOf: injectedT2IAdapters)
  return Model(modelInputs, [out], trainable: false)
}

/// UNet v2

func LoRAMiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration,
  x: Model.IO, emb: Model.IO, c: Model.IO
) -> Model.IO {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (_, _, _, _, _, _, resBlock1) =
    LoRAResBlock(
      b: batchSize, outChannels: channels, skipConnection: false,
      LoRAConfiguration: LoRAConfiguration)
  var out = resBlock1(x, emb)
  let transformer = LoRASpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingLength,
    intermediateSize: channels * 4, injectIPAdapterLengths: [], upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
  out = transformer(out, c)
  let (_, _, _, _, _, _, resBlock2) =
    LoRAResBlock(
      b: batchSize, outChannels: channels, skipConnection: false,
      LoRAConfiguration: LoRAConfiguration)
  out = resBlock2(out, emb)
  return out
}

private func LoRAInputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration, x: Model.IO,
  emb: Model.IO,
  c: Model.IO
) -> ([Model.IO], Model.IO) {
  let conv2d = LoRAConvolution(
    groups: 1, filters: 320, filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for _ in 0..<numRepeat {
      let inputLayer = LoRABlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: channel / numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: [],
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        LoRAConfiguration: LoRAConfiguration)
      previousChannel = channel
      if attentionBlock {
        out = inputLayer(out, emb, c)
      } else {
        out = inputLayer(out, emb)
      }
      passLayers.append(out)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = LoRAConvolution(
        groups: 1, filters: channel, filterSize: [3, 3], configuration: LoRAConfiguration,
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append(out)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  return (passLayers, out)
}

func LoRAOutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration, x: Model.IO,
  emb: Model.IO,
  c: Model.IO, inputs: [Model.IO]
) -> Model.IO {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 3)(out, inputs[inputIdx])
      inputIdx -= 1
      let outputLayer = LoRABlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: channel / numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: [],
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        LoRAConfiguration: LoRAConfiguration)
      if attentionBlock {
        out = outputLayer(out, emb, c)
      } else {
        out = outputLayer(out, emb)
      }
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = LoRAConvolution(
          groups: 1, filters: channel, filterSize: [3, 3], configuration: LoRAConfiguration,
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW
        )
        out = conv2d(out)
      }
      layerStart += 1
    }
  }
  return out
}

public func LoRAUNetv2(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel, injectControls: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  var injectedControls = [Model.IO]()
  if injectControls {
    injectedControls = (0..<13).map { _ in Input() }
  }
  let (_, _, timeEmbed) = LoRATimeEmbed(modelChannels: 320, LoRAConfiguration: LoRAConfiguration)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  var (inputs, inputBlocks) = LoRAInputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, x: x, emb: emb, c: c)
  var out = inputBlocks
  let middleBlock = LoRAMiddleBlock(
    channels: 1280, numHeadChannels: 64, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration, x: out, emb: emb,
    c: c)
  out = middleBlock
  if injectControls {
    out = out + injectedControls[12]
    precondition(inputs.count + 1 == injectedControls.count)
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + injectedControls[i]
    }
  }
  let outputBlocks = LoRAOutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, x: out, emb: emb, c: c, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outNorm(out)
  out = out.swish()
  let outConv2d = LoRAConvolution(
    groups: 1, filters: 4, filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = outConv2d(out)
  var modelInputs: [Model.IO] = [x, t_emb, c]
  modelInputs.append(contentsOf: injectedControls)
  return Model(modelInputs, [out], trainable: false)
}

/// UNetXL

func LoRALabelEmbed(modelChannels: Int, LoRAConfiguration: LoRANetworkConfiguration) -> (
  Model, Model, Model
) {
  let x = Input()
  let fc0 = LoRADense(count: modelChannels * 4, configuration: LoRAConfiguration)
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: modelChannels * 4, configuration: LoRAConfiguration)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func LoRACrossAttentionKeysAndValues(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), upcastAttention: Bool,
  injectIPAdapterLengths: [Int],
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, Model, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  if usesFlashAttention != .none {
    if b == 1 || t.0 == t.1 {
      let queries = toqueries(x).reshaped([b, hw, h, k])
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: upcastAttention
          ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
      var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, h * k])
      var ipKVs = [Input]()
      for _ in injectIPAdapterLengths {
        let ipKeys = Input()
        let ipValues = Input()
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention
            ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
        out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
        ipKVs.append(contentsOf: [ipKeys, ipValues])
      }
      let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
      out = unifyheads(out)
      return (toqueries, unifyheads, Model([x, keys, values] + ipKVs, [out]))
    } else {
      let queries = toqueries(x).reshaped([b, hw, h, k])
      let b0 = b / 2
      let out0: Model.IO
      if b0 == 1 || t.0 >= t.1 {
        let keys0 = keys.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries0 = queries.reshaped([b0, hw, h, k])
        let values0 = values.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out0 = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
          queries0, keys0, values0
        ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys0 = keys.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries0 = queries.reshaped(
            [1, hw, h, k], offset: [i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values0 = values.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : [.Float16])(
                queries0, keys0, values0
              ).reshaped([1, hw, h * k]))
        }
        out0 = Concat(axis: 0)(outs)
      }
      let out1: Model.IO
      if b0 == 1 || t.1 >= t.0 {
        let keys1 = keys.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries1 = queries.reshaped(
          [b0, hw, h, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
        let values1 = values.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out1 = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : [.Float16])(
            queries1, keys1, values1
          ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys1 = keys.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries1 = queries.reshaped(
            [1, hw, h, k], offset: [b0 + i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values1 = values.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
              queries1, keys1, values1
            ).reshaped([1, hw, h * k]))
        }
        out1 = Concat(axis: 0)(outs)
      }
      var out = Functional.concat(axis: 0, out0, out1)
      var ipKVs = [Input]()
      for _ in injectIPAdapterLengths {
        let ipKeys = Input()
        let ipValues = Input()
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : [.Float16])
        out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
        ipKVs.append(contentsOf: [ipKeys, ipValues])
      }
      let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
      out = unifyheads(out)
      return (toqueries, unifyheads, Model([x, keys, values] + ipKVs, [out]))
    }
  } else {
    var out: Model.IO
    var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    if b == 1 || t.0 == t.1 {
      let t = t.0
      var keys: Model.IO = keys
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
      }
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * hw, t])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: values)
      }
      dot = dot.reshaped([b, h, hw, t])
      var out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      var ipKVs = [Input]()
      for injectIPAdapterLength in injectIPAdapterLengths {
        let ipKeys = Input()
        let ipValues = Input()
        var dot = Matmul(transposeB: (2, 3))(queries, ipKeys)
        dot = dot.reshaped([b * h * hw, injectIPAdapterLength])
        dot = dot.softmax()
        if upcastAttention {
          dot = dot.to(of: values)
        }
        dot = dot.reshaped([b, h, hw, injectIPAdapterLength])
        out = out + (dot * ipValues).reshaped([b, hw, h * k])
        ipKVs.append(contentsOf: [ipKeys, ipValues])
      }
      let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
      out = unifyheads(out)
      return (toqueries, unifyheads, Model([x, keys, values] + ipKVs, [out]))
    } else {
      let b0 = b / 2
      var keys0 = keys.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries0 = queries.reshaped([b0, h, hw, k])
      if upcastAttention {
        keys0 = keys0.to(.Float32)
        queries0 = queries0.to(.Float32)
      }
      let values0 = values.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot0 = Matmul(transposeB: (2, 3))(queries0, keys0)
      dot0 = dot0.reshaped([b0 * h * hw, t.0])
      dot0 = dot0.softmax()
      if upcastAttention {
        dot0 = dot0.to(of: values)
      }
      dot0 = dot0.reshaped([b0, h, hw, t.0])
      var out0 = dot0 * values0
      out0 = out0.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      var keys1 = keys.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries1 = queries.reshaped(
        [b0, h, hw, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, hw * k, k, 1])
      if upcastAttention {
        keys1 = keys1.to(.Float32)
        queries1 = queries1.to(.Float32)
      }
      let values1 = values.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot1 = Matmul(transposeB: (2, 3))(queries1, keys1)
      dot1.add(dependencies: [out0])
      dot1 = dot1.reshaped([b0 * h * hw, t.1])
      dot1 = dot1.softmax()
      if upcastAttention {
        dot1 = dot1.to(of: values)
      }
      dot1 = dot1.reshaped([b0, h, hw, t.1])
      var out1 = dot1 * values1
      out1 = out1.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      out = Functional.concat(axis: 0, out0, out1)
    }
    var ipKVs = [Input]()
    for injectIPAdapterLength in injectIPAdapterLengths {
      let ipKeys = Input()
      let ipValues = Input()
      var dot = Matmul(transposeB: (2, 3))(queries, ipKeys)
      dot = dot.reshaped([b * h * hw, injectIPAdapterLength])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: values)
      }
      dot = dot.reshaped([b, h, hw, injectIPAdapterLength])
      out = out + (dot * ipValues).reshaped([b, hw, h * k])
      ipKVs.append(contentsOf: [ipKeys, ipValues])
    }
    let unifyheads = LoRADense(count: k * h, configuration: LoRAConfiguration)
    out = unifyheads(out)
    return (toqueries, unifyheads, Model([x, keys, values] + ipKVs, [out]))
  }
}

private func LoRABasicTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = LoRASelfAttention(
    k: k, h: h, b: b, hw: hw, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (toqueries2, unifyheads2, attn2) = LoRACrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t, upcastAttention: false,
    injectIPAdapterLengths: injectIPAdapterLengths,
    usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  out = attn2([out, keys, values] + ipKVs) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, LoRAConfiguration: LoRAConfiguration)
  if LoRAConfiguration.gradientCheckpointingFeedForward {
    ff.gradientCheckpointing = true
  }
  out = ff(out) + residual
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let formatPrefix: String
    switch format {
    case .generativeModels:
      formatPrefix = prefix.0
    case .diffusers:
      formatPrefix = prefix.1
    }
    mapping["\(formatPrefix).attn1.to_k.down"] = [tokeys1.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).attn1.to_k.up"] = [tokeys1.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).attn1.to_k.weight"] = [tokeys1.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).attn1.to_q.down"] = [toqueries1.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).attn1.to_q.up"] = [toqueries1.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).attn1.to_q.weight"] = [toqueries1.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).attn1.to_v.down"] = [tovalues1.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).attn1.to_v.up"] = [tovalues1.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).attn1.to_v.weight"] = [tovalues1.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).attn1.to_out.0.down"] = [unifyheads1.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).attn1.to_out.0.up"] = [unifyheads1.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).attn1.to_out.0.weight"] = [unifyheads1.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).attn1.to_out.0.bias"] = [unifyheads1.bias.name]
    mapping["\(formatPrefix).ff.net.0.proj.down"] = [
      fc10.parameters(for: .index(0)).name, fc11.parameters(for: .index(0)).name,
    ]
    mapping["\(formatPrefix).ff.net.0.proj.up"] = [
      fc10.parameters(for: .index(1)).name, fc11.parameters(for: .index(1)).name,
    ]
    mapping["\(formatPrefix).ff.net.0.proj.weight"] = [
      fc10.parameters(for: .index(2)).name, fc11.parameters(for: .index(2)).name,
    ]
    mapping["\(formatPrefix).ff.net.0.proj.bias"] = [fc10.bias.name, fc11.bias.name]
    mapping["\(formatPrefix).ff.net.2.down"] = [fc2.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).ff.net.2.up"] = [fc2.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).ff.net.2.weight"] = [fc2.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).ff.net.2.bias"] = [fc2.bias.name]
    mapping["\(formatPrefix).norm2.weight"] = [layerNorm2.weight.name]
    mapping["\(formatPrefix).norm2.bias"] = [layerNorm2.bias.name]
    mapping["\(formatPrefix).attn2.to_q.down"] = [toqueries2.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).attn2.to_q.up"] = [toqueries2.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).attn2.to_q.weight"] = [toqueries2.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).attn2.to_out.0.down"] = [unifyheads2.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).attn2.to_out.0.up"] = [unifyheads2.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).attn2.to_out.0.weight"] = [unifyheads2.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
    mapping["\(formatPrefix).norm1.weight"] = [layerNorm1.weight.name]
    mapping["\(formatPrefix).norm1.bias"] = [layerNorm1.bias.name]
    mapping["\(formatPrefix).norm3.weight"] = [layerNorm3.weight.name]
    mapping["\(formatPrefix).norm3.bias"] = [layerNorm3.bias.name]
    return mapping
  }
  return (mapper, Model([x, keys, values] + ipKVs, [out]))
}

private func LoRASpatialTransformer(
  prefix: (String, String),
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: (Int, Int),
  intermediateSize: Int, injectIPAdapterLengths: [Int], upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm(x)
  let projIn = LoRAConvolution(
    groups: 1, filters: k * h, filterSize: [1, 1], configuration: LoRAConfiguration, format: .OIHW)
  let hw = height * width
  out = projIn(out).reshaped([b, hw, k * h])
  var mappers = [ModelWeightMapper]()
  for i in 0..<depth {
    let keys = Input()
    kvs.append(keys)
    let values = Input()
    kvs.append(values)
    let (mapper, block) = LoRABasicTransformerBlock(
      prefix: ("\(prefix.0).transformer_blocks.\(i)", "\(prefix.1).transformer_blocks.\(i)"), k: k,
      h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize, injectIPAdapterLengths: injectIPAdapterLengths,
      upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
      LoRAConfiguration: LoRAConfiguration)
    let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
    kvs.append(contentsOf: ipKVs)
    out = block([out, keys, values] + ipKVs)
    mappers.append(mapper)
  }
  out = out.reshaped([b, height, width, k * h])
  let projOut = LoRAConvolution(
    groups: 1, filters: ch, filterSize: [1, 1], configuration: LoRAConfiguration, format: .OIHW)
  out = projOut(out) + x
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    let formatPrefix: String
    switch format {
    case .generativeModels:
      formatPrefix = prefix.0
    case .diffusers:
      formatPrefix = prefix.1
    }
    mapping["\(formatPrefix).norm.weight"] = [norm.weight.name]
    mapping["\(formatPrefix).norm.bias"] = [norm.bias.name]
    mapping["\(formatPrefix).proj_in.down"] = [projIn.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).proj_in.up"] = [projIn.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).proj_in.weight"] = [projIn.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).proj_in.bias"] = [projIn.bias.name]
    mapping["\(formatPrefix).proj_out.down"] = [projOut.parameters(for: .index(0)).name]
    mapping["\(formatPrefix).proj_out.up"] = [projOut.parameters(for: .index(1)).name]
    mapping["\(formatPrefix).proj_out.weight"] = [projOut.parameters(for: .index(2)).name]
    mapping["\(formatPrefix).proj_out.bias"] = [projOut.bias.name]
    return mapping
  }
  return (mapper, Model([x] + kvs, [out]))
}

func LoRABlockLayer(
  prefix: (String, String),
  repeatStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    LoRAResBlock(
      b: batchSize, outChannels: channels, skipConnection: skipConnection,
      LoRAConfiguration: LoRAConfiguration)
  var out = resBlock(x, emb)
  var transformerMapper: ModelWeightMapper? = nil
  if attentionBlock > 0 {
    let c = (0..<(attentionBlock * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
    let transformer: Model
    (transformerMapper, transformer) = LoRASpatialTransformer(
      prefix: ("\(prefix.0).1", "\(prefix.1).attentions.\(repeatStart)"),
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingLength,
      intermediateSize: channels * 4, injectIPAdapterLengths: injectIPAdapterLengths,
      upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
      LoRAConfiguration: LoRAConfiguration)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    if let transformerMapper = transformerMapper {
      mapping.merge(transformerMapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).0.in_layers.0.weight"] = [inLayerNorm.weight.name]
      mapping["\(prefix.0).0.in_layers.0.bias"] = [inLayerNorm.bias.name]
      mapping["\(prefix.0).0.in_layers.2.down"] = [inLayerConv2d.parameters(for: .index(0)).name]
      mapping["\(prefix.0).0.in_layers.2.up"] = [inLayerConv2d.parameters(for: .index(1)).name]
      mapping["\(prefix.0).0.in_layers.2.weight"] = [inLayerConv2d.parameters(for: .index(2)).name]
      mapping["\(prefix.0).0.in_layers.2.bias"] = [inLayerConv2d.bias.name]
      mapping["\(prefix.0).0.emb_layers.1.down"] = [embLayer.parameters(for: .index(0)).name]
      mapping["\(prefix.0).0.emb_layers.1.up"] = [embLayer.parameters(for: .index(1)).name]
      mapping["\(prefix.0).0.emb_layers.1.weight"] = [embLayer.parameters(for: .index(2)).name]
      mapping["\(prefix.0).0.emb_layers.1.bias"] = [embLayer.bias.name]
      mapping["\(prefix.0).0.out_layers.0.weight"] = [outLayerNorm.weight.name]
      mapping["\(prefix.0).0.out_layers.0.bias"] = [outLayerNorm.bias.name]
      mapping["\(prefix.0).0.out_layers.3.down"] = [outLayerConv2d.parameters(for: .index(0)).name]
      mapping["\(prefix.0).0.out_layers.3.up"] = [outLayerConv2d.parameters(for: .index(1)).name]
      mapping["\(prefix.0).0.out_layers.3.weight"] = [
        outLayerConv2d.parameters(for: .index(2)).name
      ]
      mapping["\(prefix.0).0.out_layers.3.bias"] = [outLayerConv2d.bias.name]
      if let skipModel = skipModel {
        mapping["\(prefix.0).0.skip_connection.down"] = [skipModel.parameters(for: .index(0)).name]
        mapping["\(prefix.0).0.skip_connection.up"] = [skipModel.parameters(for: .index(1)).name]
        mapping["\(prefix.0).0.skip_connection.weight"] = [
          skipModel.parameters(for: .index(2)).name
        ]
        mapping["\(prefix.0).0.skip_connection.bias"] = [skipModel.bias.name]
      }
    case .diffusers:
      mapping["\(prefix.1).resnets.\(repeatStart).norm1.weight"] = [inLayerNorm.weight.name]
      mapping["\(prefix.1).resnets.\(repeatStart).norm1.bias"] = [inLayerNorm.bias.name]
      mapping["\(prefix.1).resnets.\(repeatStart).conv1.down"] = [
        inLayerConv2d.parameters(for: .index(0)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).conv1.up"] = [
        inLayerConv2d.parameters(for: .index(1)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).conv1.weight"] = [
        inLayerConv2d.parameters(for: .index(2)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).conv1.bias"] = [inLayerConv2d.bias.name]
      mapping["\(prefix.1).resnets.\(repeatStart).time_emb_proj.down"] = [
        embLayer.parameters(for: .index(0)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).time_emb_proj.up"] = [
        embLayer.parameters(for: .index(1)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).time_emb_proj.weight"] = [
        embLayer.parameters(for: .index(2)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).time_emb_proj.bias"] = [embLayer.bias.name]
      mapping["\(prefix.1).resnets.\(repeatStart).norm2.weight"] = [outLayerNorm.weight.name]
      mapping["\(prefix.1).resnets.\(repeatStart).norm2.bias"] = [outLayerNorm.bias.name]
      mapping["\(prefix.1).resnets.\(repeatStart).conv2.down"] = [
        outLayerConv2d.parameters(for: .index(0)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).conv2.up"] = [
        outLayerConv2d.parameters(for: .index(1)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).conv2.weight"] = [
        outLayerConv2d.parameters(for: .index(2)).name
      ]
      mapping["\(prefix.1).resnets.\(repeatStart).conv2.bias"] = [outLayerConv2d.bias.name]
      if let skipModel = skipModel {
        mapping["\(prefix.1).resnets.\(repeatStart).conv_shortcut.down"] = [
          skipModel.parameters(for: .index(0)).name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).conv_shortcut.up"] = [
          skipModel.parameters(for: .index(1)).name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).conv_shortcut.weight"] = [
          skipModel.parameters(for: .index(2)).name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).conv_shortcut.bias"] = [skipModel.bias.name]
      }
    }
    return mapping
  }
  return (mapper, Model([x, emb] + kvs, [out]))
}

func LoRAMiddleBlock(
  prefix: String,
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), attentionBlock: Int,
  retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks: Bool,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration, x: Model.IO, emb: Model.IO
) -> (ModelWeightMapper, Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    LoRAResBlock(
      b: batchSize, outChannels: channels, skipConnection: false,
      LoRAConfiguration: LoRAConfiguration)
  var out = resBlock1(x, emb)
  let kvs = (0..<(attentionBlock * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
  let transformerMapper: ModelWeightMapper?
  let inLayerNorm2: Model?
  let inLayerConv2d2: Model?
  let embLayer2: Model?
  let outLayerNorm2: Model?
  let outLayerConv2d2: Model?
  if attentionBlock > 0 {
    let (mapper, transformer) = LoRASpatialTransformer(
      prefix: ("\(prefix).middle_block.1", "mid_block.attentions.0"), ch: channels, k: k,
      h: numHeads, b: batchSize, height: height,
      width: width, depth: attentionBlock, t: embeddingLength, intermediateSize: channels * 4,
      injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
      usesFlashAttention: usesFlashAttention,
      LoRAConfiguration: LoRAConfiguration)
    transformerMapper = mapper
    out = transformer([out] + kvs)
    let resBlock2: Model
    (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
      LoRAResBlock(
        b: batchSize, outChannels: channels, skipConnection: false,
        LoRAConfiguration: LoRAConfiguration)
    out = resBlock2(out, emb)
  } else {
    if retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks {
      let (mapper, transformer) = LoRASpatialTransformer(
        prefix: ("\(prefix).middle_block.1", "mid_block.attentions.0"), ch: channels, k: k,
        h: numHeads, b: batchSize, height: height, width: width, depth: 0,  // Zero depth so we still apply norm, proj_in, proj_out.
        t: embeddingLength, intermediateSize: channels * 4,
        injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
        usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
      transformerMapper = mapper
      out = transformer(out)
      let resBlock2: Model
      (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
        LoRAResBlock(
          b: batchSize, outChannels: channels, skipConnection: false,
          LoRAConfiguration: LoRAConfiguration)
      out = resBlock2(out, emb)
    } else {
      transformerMapper = nil
      inLayerNorm2 = nil
      inLayerConv2d2 = nil
      embLayer2 = nil
      outLayerNorm2 = nil
      outLayerConv2d2 = nil
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    if let transformerMapper = transformerMapper {
      mapping.merge(transformerMapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["\(prefix).middle_block.0.in_layers.0.weight"] = [inLayerNorm1.weight.name]
      mapping["\(prefix).middle_block.0.in_layers.0.bias"] = [inLayerNorm1.bias.name]
      mapping["\(prefix).middle_block.0.in_layers.2.down"] = [
        inLayerConv2d1.parameters(for: .index(0)).name
      ]
      mapping["\(prefix).middle_block.0.in_layers.2.up"] = [
        inLayerConv2d1.parameters(for: .index(1)).name
      ]
      mapping["\(prefix).middle_block.0.in_layers.2.weight"] = [
        inLayerConv2d1.parameters(for: .index(2)).name
      ]
      mapping["\(prefix).middle_block.0.in_layers.2.bias"] = [inLayerConv2d1.bias.name]
      mapping["\(prefix).middle_block.0.emb_layers.1.down"] = [
        embLayer1.parameters(for: .index(0)).name
      ]
      mapping["\(prefix).middle_block.0.emb_layers.1.up"] = [
        embLayer1.parameters(for: .index(1)).name
      ]
      mapping["\(prefix).middle_block.0.emb_layers.1.weight"] = [
        embLayer1.parameters(for: .index(2)).name
      ]
      mapping["\(prefix).middle_block.0.emb_layers.1.bias"] = [embLayer1.bias.name]
      mapping["\(prefix).middle_block.0.out_layers.0.weight"] = [outLayerNorm1.weight.name]
      mapping["\(prefix).middle_block.0.out_layers.0.bias"] = [outLayerNorm1.bias.name]
      mapping["\(prefix).middle_block.0.out_layers.3.down"] = [
        outLayerConv2d1.parameters(for: .index(0)).name
      ]
      mapping["\(prefix).middle_block.0.out_layers.3.up"] = [
        outLayerConv2d1.parameters(for: .index(1)).name
      ]
      mapping["\(prefix).middle_block.0.out_layers.3.weight"] = [
        outLayerConv2d1.parameters(for: .index(2)).name
      ]
      mapping["\(prefix).middle_block.0.out_layers.3.bias"] = [outLayerConv2d1.bias.name]
      if let inLayerNorm2 = inLayerNorm2, let inLayerConv2d2 = inLayerConv2d2,
        let embLayer2 = embLayer2, let outLayerNorm2 = outLayerNorm2,
        let outLayerConv2d2 = outLayerConv2d2
      {
        mapping["\(prefix).middle_block.2.in_layers.0.weight"] = [inLayerNorm2.weight.name]
        mapping["\(prefix).middle_block.2.in_layers.0.bias"] = [inLayerNorm2.bias.name]
        mapping["\(prefix).middle_block.2.in_layers.2.down"] = [
          inLayerConv2d2.parameters(for: .index(0)).name
        ]
        mapping["\(prefix).middle_block.2.in_layers.2.up"] = [
          inLayerConv2d2.parameters(for: .index(1)).name
        ]
        mapping["\(prefix).middle_block.2.in_layers.2.weight"] = [
          inLayerConv2d2.parameters(for: .index(2)).name
        ]
        mapping["\(prefix).middle_block.2.in_layers.2.bias"] = [inLayerConv2d2.bias.name]
        mapping["\(prefix).middle_block.2.emb_layers.1.down"] = [
          embLayer2.parameters(for: .index(0)).name
        ]
        mapping["\(prefix).middle_block.2.emb_layers.1.up"] = [
          embLayer2.parameters(for: .index(1)).name
        ]
        mapping["\(prefix).middle_block.2.emb_layers.1.weight"] = [
          embLayer2.parameters(for: .index(2)).name
        ]
        mapping["\(prefix).middle_block.2.emb_layers.1.bias"] = [embLayer2.bias.name]
        mapping["\(prefix).middle_block.2.out_layers.0.weight"] = [outLayerNorm2.weight.name]
        mapping["\(prefix).middle_block.2.out_layers.0.bias"] = [outLayerNorm2.bias.name]
        mapping["\(prefix).middle_block.2.out_layers.3.down"] = [
          outLayerConv2d2.parameters(for: .index(0)).name
        ]
        mapping["\(prefix).middle_block.2.out_layers.3.up"] = [
          outLayerConv2d2.parameters(for: .index(1)).name
        ]
        mapping["\(prefix).middle_block.2.out_layers.3.weight"] = [
          outLayerConv2d2.parameters(for: .index(2)).name
        ]
        mapping["\(prefix).middle_block.2.out_layers.3.bias"] = [outLayerConv2d2.bias.name]
      }
    case .diffusers:
      mapping["mid_block.resnets.0.norm1.weight"] = [inLayerNorm1.weight.name]
      mapping["mid_block.resnets.0.norm1.bias"] = [inLayerNorm1.bias.name]
      mapping["mid_block.resnets.0.conv1.down"] = [inLayerConv2d1.parameters(for: .index(0)).name]
      mapping["mid_block.resnets.0.conv1.up"] = [inLayerConv2d1.parameters(for: .index(1)).name]
      mapping["mid_block.resnets.0.conv1.weight"] = [inLayerConv2d1.parameters(for: .index(2)).name]
      mapping["mid_block.resnets.0.conv1.bias"] = [inLayerConv2d1.bias.name]
      mapping["mid_block.resnets.0.time_emb_proj.down"] = [
        embLayer1.parameters(for: .index(0)).name
      ]
      mapping["mid_block.resnets.0.time_emb_proj.up"] = [embLayer1.parameters(for: .index(1)).name]
      mapping["mid_block.resnets.0.time_emb_proj.weight"] = [
        embLayer1.parameters(for: .index(2)).name
      ]
      mapping["mid_block.resnets.0.time_emb_proj.bias"] = [embLayer1.bias.name]
      mapping["mid_block.resnets.0.norm2.weight"] = [outLayerNorm1.weight.name]
      mapping["mid_block.resnets.0.norm2.bias"] = [outLayerNorm1.bias.name]
      mapping["mid_block.resnets.0.conv2.down"] = [outLayerConv2d1.parameters(for: .index(0)).name]
      mapping["mid_block.resnets.0.conv2.up"] = [outLayerConv2d1.parameters(for: .index(1)).name]
      mapping["mid_block.resnets.0.conv2.weight"] = [
        outLayerConv2d1.parameters(for: .index(2)).name
      ]
      mapping["mid_block.resnets.0.conv2.bias"] = [outLayerConv2d1.bias.name]
      if let inLayerNorm2 = inLayerNorm2, let inLayerConv2d2 = inLayerConv2d2,
        let embLayer2 = embLayer2, let outLayerNorm2 = outLayerNorm2,
        let outLayerConv2d2 = outLayerConv2d2
      {
        mapping["mid_block.resnets.1.norm1.weight"] = [inLayerNorm2.weight.name]
        mapping["mid_block.resnets.1.norm1.bias"] = [inLayerNorm2.bias.name]
        mapping["mid_block.resnets.1.conv1.down"] = [inLayerConv2d2.parameters(for: .index(0)).name]
        mapping["mid_block.resnets.1.conv1.up"] = [inLayerConv2d2.parameters(for: .index(1)).name]
        mapping["mid_block.resnets.1.conv1.weight"] = [
          inLayerConv2d2.parameters(for: .index(2)).name
        ]
        mapping["mid_block.resnets.1.conv1.bias"] = [inLayerConv2d2.bias.name]
        mapping["mid_block.resnets.1.time_emb_proj.down"] = [
          embLayer2.parameters(for: .index(0)).name
        ]
        mapping["mid_block.resnets.1.time_emb_proj.up"] = [
          embLayer2.parameters(for: .index(1)).name
        ]
        mapping["mid_block.resnets.1.time_emb_proj.weight"] = [
          embLayer2.parameters(for: .index(2)).name
        ]
        mapping["mid_block.resnets.1.time_emb_proj.bias"] = [embLayer2.bias.name]
        mapping["mid_block.resnets.1.norm2.weight"] = [outLayerNorm2.weight.name]
        mapping["mid_block.resnets.1.norm2.bias"] = [outLayerNorm2.bias.name]
        mapping["mid_block.resnets.1.conv2.down"] = [
          outLayerConv2d2.parameters(for: .index(0)).name
        ]
        mapping["mid_block.resnets.1.conv2.up"] = [outLayerConv2d2.parameters(for: .index(1)).name]
        mapping["mid_block.resnets.1.conv2.weight"] = [
          outLayerConv2d2.parameters(for: .index(2)).name
        ]
        mapping["mid_block.resnets.1.conv2.bias"] = [outLayerConv2d2.bias.name]
      }
    }
    return mapping
  }
  return (mapper, out, kvs)
}

func LoRAInputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  injectIPAdapterLengths: [Int], upcastAttention: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration, x: Model.IO,
  emb: Model.IO
) -> (ModelWeightMapper, [Model.IO], Model.IO, [Input]) {
  let conv2d = LoRAConvolution(
    groups: 1, filters: channels[0], filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    let upcastAttention = Set(upcastAttention[ds, default: []])
    for j in 0..<numRepeat {
      let (mapper, inputLayer) = LoRABlockLayer(
        prefix: ("model.diffusion_model.input_blocks.\(layerStart)", "down_blocks.\(i)"),
        repeatStart: j, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention.contains(j), usesFlashAttention: usesFlashAttention,
        LoRAConfiguration: LoRAConfiguration)
      previousChannel = channel
      let c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input()
      }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append(out)
      mappers.append(mapper)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = LoRAConvolution(
        groups: 1, filters: channel, filterSize: [3, 3], configuration: LoRAConfiguration,
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let mapper: ModelWeightMapper = { format in
        switch format {
        case .generativeModels:
          return [
            "model.diffusion_model.input_blocks.\(downLayer).0.op.down": [
              downsample.parameters(for: .index(0)).name
            ],
            "model.diffusion_model.input_blocks.\(downLayer).0.op.up": [
              downsample.parameters(for: .index(1)).name
            ],
            "model.diffusion_model.input_blocks.\(downLayer).0.op.weight": [
              downsample.parameters(for: .index(2)).name
            ],
            "model.diffusion_model.input_blocks.\(downLayer).0.op.bias": [downsample.bias.name],
          ]
        case .diffusers:
          return [
            "down_blocks.\(i).downsamplers.0.conv.down": [
              downsample.parameters(for: .index(0)).name
            ],
            "down_blocks.\(i).downsamplers.0.conv.up": [downsample.parameters(for: .index(1)).name],
            "down_blocks.\(i).downsamplers.0.conv.weight": [
              downsample.parameters(for: .index(2)).name
            ],
            "down_blocks.\(i).downsamplers.0.conv.bias": [downsample.bias.name],
          ]
        }
      }
      mappers.append(mapper)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["model.diffusion_model.input_blocks.0.0.down"] = [
        conv2d.parameters(for: .index(0)).name
      ]
      mapping["model.diffusion_model.input_blocks.0.0.up"] = [
        conv2d.parameters(for: .index(1)).name
      ]
      mapping["model.diffusion_model.input_blocks.0.0.weight"] = [
        conv2d.parameters(for: .index(2)).name
      ]
      mapping["model.diffusion_model.input_blocks.0.0.bias"] = [conv2d.bias.name]
    case .diffusers:
      mapping["conv_in.down"] = [conv2d.parameters(for: .index(0)).name]
      mapping["conv_in.up"] = [conv2d.parameters(for: .index(1)).name]
      mapping["conv_in.weight"] = [conv2d.parameters(for: .index(2)).name]
      mapping["conv_in.bias"] = [conv2d.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, passLayers, out, kvs)
}

func LoRAOutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  injectIPAdapterLengths: [Int], upcastAttention: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration, x: Model.IO,
  emb: Model.IO,
  inputs: [Model.IO]
) -> (ModelWeightMapper, Model.IO, [Input]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    let upcastAttention = Set(upcastAttention[ds, default: []])
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 3)(out, inputs[inputIdx])
      inputIdx -= 1
      let (mapper, outputLayer) = LoRABlockLayer(
        prefix: (
          "model.diffusion_model.output_blocks.\(layerStart)", "up_blocks.\(channels.count - 1 - i)"
        ),
        repeatStart: j, skipConnection: true,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention.contains(j), usesFlashAttention: usesFlashAttention,
        LoRAConfiguration: LoRAConfiguration)
      let c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input()
      }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      mappers.append(mapper)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = LoRAConvolution(
          groups: 1, filters: channel, filterSize: [3, 3], configuration: LoRAConfiguration,
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW
        )
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock[j] > 0 ? 2 : 1
        let reverseUpLayer = channels.count - 1 - i
        let mapper: ModelWeightMapper = { format in
          switch format {
          case .generativeModels:
            return [
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.down": [
                conv2d.parameters(for: .index(0)).name
              ],
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.up": [
                conv2d.parameters(for: .index(1)).name
              ],
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight": [
                conv2d.parameters(for: .index(2)).name
              ],
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias": [
                conv2d.bias.name
              ],
            ]
          case .diffusers:
            return [
              "up_blocks.\(reverseUpLayer).upsamplers.0.conv.down": [
                conv2d.parameters(for: .index(0)).name
              ],
              "up_blocks.\(reverseUpLayer).upsamplers.0.conv.up": [
                conv2d.parameters(for: .index(1)).name
              ],
              "up_blocks.\(reverseUpLayer).upsamplers.0.conv.weight": [
                conv2d.parameters(for: .index(2)).name
              ],
              "up_blocks.\(reverseUpLayer).upsamplers.0.conv.bias": [conv2d.bias.name],
            ]
          }
        }
        mappers.append(mapper)
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
  return (mapper, out, kvs)
}

public func LoRAUNetXL(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  outputAttentionRes: KeyValuePairs<Int, [Int]>, embeddingLength: (Int, Int),
  injectIPAdapterLengths: [Int],
  upcastAttention: (KeyValuePairs<Int, [Int]>, Bool, KeyValuePairs<Int, [Int]>),
  usesFlashAttention: FlashAttentionLevel, injectControls: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  var injectedControls = [Model.IO]()
  if injectControls {
    injectedControls = (0..<(channels.count * 3 + 1)).map { _ in Input() }
  }
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let inputUpcastAttention = [Int: [Int]](
    uniqueKeysWithValues: upcastAttention.0.map { ($0.key, $0.value) })
  let outputUpcastAttention = [Int: [Int]](
    uniqueKeysWithValues: upcastAttention.2.map { ($0.key, $0.value) })
  let (timeFc0, timeFc2, timeEmbed) = LoRATimeEmbed(
    modelChannels: channels[0], LoRAConfiguration: LoRAConfiguration)
  let (labelFc0, labelFc2, labelEmbed) = LoRALabelEmbed(
    modelChannels: channels[0], LoRAConfiguration: LoRAConfiguration)
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  var (inputMapper, inputs, inputBlocks, inputKVs) = LoRAInputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: inputUpcastAttention, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, x: x, emb: emb)
  var out = inputBlocks
  let (middleMapper, middleBlock, middleKVs) = LoRAMiddleBlock(
    prefix: "model.diffusion_model",
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
    retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks: false,  // This is particular to SSD-1B. If SSD-1B starts to have ControlNet, we might want to expose this.
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention.1,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration, x: out, emb: emb)
  out = middleBlock
  if injectControls {
    out = out + injectedControls[injectedControls.count - 1]
    precondition(inputs.count + 1 == injectedControls.count)
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + injectedControls[i]
    }
  }
  let (outputMapper, outputBlocks, outputKVs) = LoRAOutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: outputAttentionRes, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: outputUpcastAttention, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, x: out, emb: emb,
    inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = LoRAConvolution(
    groups: 1, filters: 4, filterSize: [3, 3], configuration: LoRAConfiguration,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = outConv2d(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    mapping.merge(middleMapper(format)) { v, _ in v }
    mapping.merge(outputMapper(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["model.diffusion_model.time_embed.0.down"] = [timeFc0.parameters(for: .index(0)).name]
      mapping["model.diffusion_model.time_embed.0.up"] = [timeFc0.parameters(for: .index(1)).name]
      mapping["model.diffusion_model.time_embed.0.weight"] = [
        timeFc0.parameters(for: .index(2)).name
      ]
      mapping["model.diffusion_model.time_embed.0.bias"] = [timeFc0.bias.name]
      mapping["model.diffusion_model.time_embed.2.down"] = [timeFc2.parameters(for: .index(0)).name]
      mapping["model.diffusion_model.time_embed.2.up"] = [timeFc2.parameters(for: .index(1)).name]
      mapping["model.diffusion_model.time_embed.2.weight"] = [
        timeFc2.parameters(for: .index(2)).name
      ]
      mapping["model.diffusion_model.time_embed.2.bias"] = [timeFc2.bias.name]
      mapping["model.diffusion_model.label_emb.0.0.down"] = [
        labelFc0.parameters(for: .index(0)).name
      ]
      mapping["model.diffusion_model.label_emb.0.0.up"] = [labelFc0.parameters(for: .index(1)).name]
      mapping["model.diffusion_model.label_emb.0.0.weight"] = [
        labelFc0.parameters(for: .index(2)).name
      ]
      mapping["model.diffusion_model.label_emb.0.0.bias"] = [labelFc0.bias.name]
      mapping["model.diffusion_model.label_emb.0.2.down"] = [
        labelFc2.parameters(for: .index(0)).name
      ]
      mapping["model.diffusion_model.label_emb.0.2.up"] = [labelFc2.parameters(for: .index(1)).name]
      mapping["model.diffusion_model.label_emb.0.2.weight"] = [
        labelFc2.parameters(for: .index(2)).name
      ]
      mapping["model.diffusion_model.label_emb.0.2.bias"] = [labelFc2.bias.name]
      mapping["model.diffusion_model.out.0.weight"] = [outNorm.weight.name]
      mapping["model.diffusion_model.out.0.bias"] = [outNorm.bias.name]
      mapping["model.diffusion_model.out.2.down"] = [outConv2d.parameters(for: .index(0)).name]
      mapping["model.diffusion_model.out.2.up"] = [outConv2d.parameters(for: .index(1)).name]
      mapping["model.diffusion_model.out.2.weight"] = [outConv2d.parameters(for: .index(2)).name]
      mapping["model.diffusion_model.out.2.bias"] = [outConv2d.bias.name]
    case .diffusers:
      mapping["time_embedding.linear_1.down"] = [timeFc0.parameters(for: .index(0)).name]
      mapping["time_embedding.linear_1.up"] = [timeFc0.parameters(for: .index(1)).name]
      mapping["time_embedding.linear_1.weight"] = [timeFc0.parameters(for: .index(2)).name]
      mapping["time_embedding.linear_1.bias"] = [timeFc0.bias.name]
      mapping["time_embedding.linear_2.down"] = [timeFc2.parameters(for: .index(0)).name]
      mapping["time_embedding.linear_2.up"] = [timeFc2.parameters(for: .index(1)).name]
      mapping["time_embedding.linear_2.weight"] = [timeFc2.parameters(for: .index(2)).name]
      mapping["time_embedding.linear_2.bias"] = [timeFc2.bias.name]
      mapping["add_embedding.linear_1.down"] = [labelFc0.parameters(for: .index(0)).name]
      mapping["add_embedding.linear_1.up"] = [labelFc0.parameters(for: .index(1)).name]
      mapping["add_embedding.linear_1.weight"] = [labelFc0.parameters(for: .index(2)).name]
      mapping["add_embedding.linear_1.bias"] = [labelFc0.bias.name]
      mapping["add_embedding.linear_2.down"] = [labelFc2.parameters(for: .index(0)).name]
      mapping["add_embedding.linear_2.up"] = [labelFc2.parameters(for: .index(1)).name]
      mapping["add_embedding.linear_2.weight"] = [labelFc2.parameters(for: .index(2)).name]
      mapping["add_embedding.linear_2.bias"] = [labelFc2.bias.name]
      mapping["conv_norm_out.weight"] = [outNorm.weight.name]
      mapping["conv_norm_out.bias"] = [outNorm.bias.name]
      mapping["conv_out.down"] = [outConv2d.parameters(for: .index(0)).name]
      mapping["conv_out.up"] = [outConv2d.parameters(for: .index(1)).name]
      mapping["conv_out.weight"] = [outConv2d.parameters(for: .index(2)).name]
      mapping["conv_out.bias"] = [outConv2d.bias.name]
    }
    return mapping
  }
  return (
    Model(
      [x, t_emb, y] + inputKVs + middleKVs + outputKVs + injectedControls, [out], trainable: false),
    mapper
  )
}

func LoRACrossAttentionFixed(
  k: Int, h: Int, b: Int, t: (Int, Int), usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
)
  -> (Model, Model, Model)
{
  let c = Input()
  let tokeys = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  let tovalues = LoRADense(count: k * h, configuration: LoRAConfiguration, noBias: true)
  // We shouldn't transpose if we are going to do that within the UNet.
  if t.0 == t.1 {
    var keys = tokeys(c).reshaped([b, t.0, h, k])
    var values = tovalues(c).reshaped([b, t.0, h, k])
    if usesFlashAttention == .none {
      keys = keys.transposed(1, 2)
      values = values.transposed(1, 2)
    }
    return (tokeys, tovalues, Model([c], [keys, values]))
  } else {
    let keys = tokeys(c)
    let values = tovalues(c)
    return (tokeys, tovalues, Model([c], [keys, values]))
  }
}

func LoRABasicTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int, b: Int, t: (Int, Int), intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let (tokeys2, tovalues2, attn2) = LoRACrossAttentionFixed(
    k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).attn2.to_k.down"] = [
        tokeys2.parameters(for: .index(0)).name
      ]
      mapping["\(prefix.0).attn2.to_k.up"] = [
        tokeys2.parameters(for: .index(1)).name
      ]
      mapping["\(prefix.0).attn2.to_k.weight"] = [
        tokeys2.parameters(for: .index(2)).name
      ]
      mapping["\(prefix.0).attn2.to_v.down"] = [
        tovalues2.parameters(for: .index(0)).name
      ]
      mapping["\(prefix.0).attn2.to_v.up"] = [
        tovalues2.parameters(for: .index(1)).name
      ]
      mapping["\(prefix.0).attn2.to_v.weight"] = [
        tovalues2.parameters(for: .index(2)).name
      ]
    case .diffusers:
      mapping["\(prefix.1).attn2.to_k.down"] = [tokeys2.parameters(for: .index(0)).name]
      mapping["\(prefix.1).attn2.to_k.up"] = [tokeys2.parameters(for: .index(1)).name]
      mapping["\(prefix.1).attn2.to_k.weight"] = [tokeys2.parameters(for: .index(2)).name]
      mapping["\(prefix.1).attn2.to_v.down"] = [tovalues2.parameters(for: .index(0)).name]
      mapping["\(prefix.1).attn2.to_v.up"] = [tovalues2.parameters(for: .index(1)).name]
      mapping["\(prefix.1).attn2.to_v.weight"] = [tovalues2.parameters(for: .index(2)).name]
    }
    return mapping
  }
  return (mapper, attn2)
}

func LoRASpatialTransformerFixed(
  prefix: (String, String),
  ch: Int, k: Int, h: Int, b: Int, depth: Int, t: (Int, Int),
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let c = Input()
  var outs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  for i in 0..<depth {
    let (mapper, block) = LoRABasicTransformerBlockFixed(
      prefix: ("\(prefix.0).transformer_blocks.\(i)", "\(prefix.1).transformer_blocks.\(i)"), k: k,
      h: h, b: b, t: t,
      intermediateSize: intermediateSize, usesFlashAttention: usesFlashAttention,
      LoRAConfiguration: LoRAConfiguration)
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

func LoRABlockLayerFixed(
  prefix: (String, String),
  repeatStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: (Int, Int), intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (transformerMapper, transformer) = LoRASpatialTransformerFixed(
    prefix: ("\(prefix.0).1", "\(prefix.1).attentions.\(repeatStart)"),
    ch: channels, k: k, h: numHeads, b: batchSize,
    depth: attentionBlock, t: embeddingLength,
    intermediateSize: channels * 4, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration)
  return (transformerMapper, transformer)
}

func LoRAMiddleBlockFixed(
  prefix: String,
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), attentionBlock: Int, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration, c: Model.IO
) -> (ModelWeightMapper, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (transformerMapper, transformer) = LoRASpatialTransformerFixed(
    prefix: ("\(prefix).middle_block.1", "mid_block.attentions.0"), ch: channels, k: k, h: numHeads,
    b: batchSize, depth: attentionBlock, t: embeddingLength, intermediateSize: channels * 4,
    usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration)
  let out = transformer(c)
  return (transformerMapper, out)
}

func LoRAInputBlocksFixed(
  prefix: String,
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration,
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
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    for j in 0..<numRepeat {
      if attentionBlock[j] > 0 {
        let (mapper, inputLayer) = LoRABlockLayerFixed(
          prefix: ("\(prefix).input_blocks.\(layerStart)", "down_blocks.\(i)"),
          repeatStart: j, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention,
          LoRAConfiguration: LoRAConfiguration)
        previousChannel = channel
        outs.append(inputLayer(c))
        mappers.append(mapper)
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

func LoRAOutputBlocksFixed(
  prefix: String,
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration,
  c: Model.IO
) -> (ModelWeightMapper, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var outs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    for j in 0..<(numRepeat + 1) {
      if attentionBlock[j] > 0 {
        let (mapper, outputLayer) = LoRABlockLayerFixed(
          prefix: ("\(prefix).output_blocks.\(layerStart)", "up_blocks.\(channels.count - 1 - i)"),
          repeatStart: j, skipConnection: true,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention,
          LoRAConfiguration: LoRAConfiguration)
        outs.append(outputLayer(c))
        mappers.append(mapper)
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

public func LoRAUNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  embeddingLength: (Int, Int), inputAttentionRes: KeyValuePairs<Int, [Int]>,
  middleAttentionBlocks: Int, outputAttentionRes: KeyValuePairs<Int, [Int]>,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let c = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let (_, inputBlocks) = LoRAInputBlocksFixed(
    prefix: "model.diffusion_model",
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, c: c)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  if middleAttentionBlocks > 0 {
    let (_, middleBlock) = LoRAMiddleBlockFixed(
      prefix: "model.diffusion_model",
      channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
      height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
      embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
      usesFlashAttention: usesFlashAttention, LoRAConfiguration: LoRAConfiguration, c: c)
    out.append(middleBlock)
  }
  let (_, outputBlocks) = LoRAOutputBlocksFixed(
    prefix: "model.diffusion_model",
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: outputAttentionRes, usesFlashAttention: usesFlashAttention,
    LoRAConfiguration: LoRAConfiguration, c: c)
  out.append(contentsOf: outputBlocks)
  return Model([c], out, trainable: false)
}
