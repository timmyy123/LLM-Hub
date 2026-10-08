import NNC

/// Text Model

func CLIPTextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, injectEmbeddings: Bool, batchSize: Int, vocabularySize: Int, maxLength: Int,
  maxTokenLength: Int, embeddingSize: Int
) -> (
  Model, Model, Model
) {
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(T.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let model: Model
  if injectEmbeddings {
    let tokenMask = Input()
    let injectedEmbeddings = Input()
    // Adding additional reshape to make sure the order between token embed and position embed never switch.
    let embedding =
      tokenEmbed(tokens) .* tokenMask
      + positionEmbed(positions).identity()
      + injectedEmbeddings
    model = Model(
      [tokens, positions, tokenMask, injectedEmbeddings], [embedding])
  } else {
    let embedding = tokenEmbed(tokens) + positionEmbed(positions)
    model = Model([tokens, positions], [embedding])
  }
  return (tokenEmbed, positionEmbed, model)
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  if usesFlashAttention {
    let queries = toqueries(x).reshaped([b, t, h, k]).identity().identity()
    let keys = tokeys(x).reshaped([b, t, h, k]).identity()
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true,
      multiHeadOutputProjectionFused: true)
    let out = scaledDotProductAttention(queries, keys, values, causalAttentionMask).reshaped([
      b * t, h * k,
    ])
    return (
      tokeys, toqueries, tovalues, scaledDotProductAttention,
      Model([x, causalAttentionMask], [out])
    )
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
    let unifyheads = Dense(count: k * h)
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x, causalAttentionMask], [out]))
  }
}

func QuickGELU() -> Model {
  let x = Input()
  let y = x .* Sigmoid()(1.702 * x)
  return Model([x], [y])
}

func CLIPMLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func CLIPEncoderLayer(
  k: Int, h: Int, b: Int, t: Int, intermediateSize: Int, usesFlashAttention: Bool
) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model
) {
  let x = Input()
  let causalAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let (tokeys, toqueries, tovalues, unifyheads, attention) = CLIPAttention(
    k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention)
  out = attention(out, causalAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let (fc1, fc2, mlp) = CLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return (
    layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2,
    Model([x, causalAttentionMask], [out])
  )
}

public func CLIPTextModel<T: TensorNumeric>(
  _ dataType: T.Type, injectEmbeddings: Bool,
  vocabularySize: Int, maxLength: Int, maxTokenLength: Int, embeddingSize: Int, numLayers: Int,
  numHeads: Int, batchSize: Int, intermediateSize: Int, usesFlashAttention: Bool,
  outputPenultimate: Bool = false, noFinalLayerNorm: Bool = false, trainable: Bool? = nil
) -> (Model, PythonReader) {
  let tokens = Input()
  let positions = Input()
  let causalAttentionMask = Input()
  let (tokenEmbed, positionEmbed, embedding) = CLIPTextEmbedding(
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
  var layerNorm1s = [Model]()
  var tokeyss = [Model]()
  var toqueriess = [Model]()
  var tovaluess = [Model]()
  var unifyheadss = [Model]()
  var layerNorm2s = [Model]()
  var fc1s = [Model]()
  var fc2s = [Model]()
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 && outputPenultimate {
      penultimate = out
    }
    let (layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2, encoderLayer) =
      CLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxTokenLength, intermediateSize: intermediateSize,
        usesFlashAttention: usesFlashAttention)
    layerNorm1s.append(layerNorm1)
    tokeyss.append(tokeys)
    toqueriess.append(toqueries)
    tovaluess.append(tovalues)
    unifyheadss.append(unifyheads)
    layerNorm2s.append(layerNorm2)
    fc1s.append(fc1)
    fc2s.append(fc2)
    out = encoderLayer(out, causalAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  if !noFinalLayerNorm {
    out = finalLayerNorm(out)
  }
  let reader: PythonReader = { stateDict, archive in
    guard
      let vocab =
        stateDict["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let pos =
        stateDict["cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokenEmbed.parameters.copy(from: vocab, zip: archive, of: T.self)
    try positionEmbed.parameters.copy(from: pos, zip: archive, of: T.self)

    for i in 0..<numLayers {
      guard
        let layer_norm_1_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm1.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let layer_norm_1_bias =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm1.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm1s[i].parameters(for: .weight).copy(
        from: layer_norm_1_weight, zip: archive, of: T.self)
      try layerNorm1s[i].parameters(for: .bias).copy(
        from: layer_norm_1_bias, zip: archive, of: T.self)

      guard
        let k_proj_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.k_proj.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let k_proj_bias =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.k_proj.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tokeyss[i].parameters(for: .weight).copy(
        from: k_proj_weight, zip: archive, of: T.self)
      try tokeyss[i].parameters(for: .bias).copy(
        from: k_proj_bias, zip: archive, of: T.self)

      guard
        let v_proj_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.v_proj.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let v_proj_bias =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.v_proj.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tovaluess[i].parameters(for: .weight).copy(
        from: v_proj_weight, zip: archive, of: T.self)
      try tovaluess[i].parameters(for: .bias).copy(
        from: v_proj_bias, zip: archive, of: T.self)

      guard
        let q_proj_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.q_proj.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let q_proj_bias =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.q_proj.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try toqueriess[i].parameters(for: .weight).copy(
        from: q_proj_weight, zip: archive, of: T.self)
      try toqueriess[i].parameters(for: .bias).copy(
        from: q_proj_bias, zip: archive, of: T.self)

      guard
        let out_proj_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.out_proj.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let out_proj_bias =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.out_proj.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheadss[i].parameters(for: .weight).copy(
        from: out_proj_weight, zip: archive, of: T.self)
      try unifyheadss[i].parameters(for: .bias).copy(
        from: out_proj_bias, zip: archive, of: T.self)

      guard
        let layer_norm_2_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm2.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let layer_norm_2_bias =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm2.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm2s[i].parameters(for: .weight).copy(
        from: layer_norm_2_weight, zip: archive, of: T.self)
      try layerNorm2s[i].parameters(for: .bias).copy(
        from: layer_norm_2_bias, zip: archive, of: T.self)

      guard
        let fc1_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc1.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let fc1_bias =
          stateDict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc1.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try fc1s[i].parameters(for: .weight).copy(
        from: fc1_weight, zip: archive, of: T.self)
      try fc1s[i].parameters(for: .bias).copy(from: fc1_bias, zip: archive, of: T.self)

      guard
        let fc2_weight =
          stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc2.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let fc2_bias =
          stateDict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc2.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try fc2s[i].parameters(for: .weight).copy(
        from: fc2_weight, zip: archive, of: T.self)
      try fc2s[i].parameters(for: .bias).copy(from: fc2_bias, zip: archive, of: T.self)
    }

    guard
      let final_layer_norm_weight =
        stateDict[
          "cond_stage_model.transformer.text_model.final_layer_norm.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let final_layer_norm_bias =
        stateDict["cond_stage_model.transformer.text_model.final_layer_norm.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try finalLayerNorm.parameters(for: .weight).copy(
      from: final_layer_norm_weight, zip: archive, of: T.self)
    try finalLayerNorm.parameters(for: .bias).copy(
      from: final_layer_norm_bias, zip: archive, of: T.self)
  }
  let model: Model
  if injectEmbeddings, let embedMask = embedMask, let injectedEmbeddings = injectedEmbeddings {
    model = Model(
      [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings],
      (penultimate.map { [$0] } ?? []) + [out],
      trainable: trainable)
  } else {
    model = Model(
      [tokens, positions, causalAttentionMask], (penultimate.map { [$0] } ?? []) + [out],
      trainable: trainable)
  }
  return (model, reader)
}
