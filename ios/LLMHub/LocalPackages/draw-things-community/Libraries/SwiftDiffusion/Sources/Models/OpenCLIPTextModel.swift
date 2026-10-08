import NNC

/// OpenCLIP model

func OpenCLIPMLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = fc1(x)
  out = GELU()(out)
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func OpenCLIPEncoderLayer(
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
  let (fc1, fc2, mlp) = OpenCLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return (
    layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2,
    Model([x, causalAttentionMask], [out])
  )
}

public func OpenCLIPTextModel<T: TensorNumeric>(
  _ dataType: T.Type, injectEmbeddings: Bool,
  vocabularySize: Int, maxLength: Int, maxTokenLength: Int, embeddingSize: Int, numLayers: Int,
  numHeads: Int, batchSize: Int, intermediateSize: Int, usesFlashAttention: Bool,
  outputPenultimate: Bool = false, outputHiddenState: Bool = false, trainable: Bool? = nil
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
      OpenCLIPEncoderLayer(
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
  let hiddenState: Model.IO? = outputHiddenState ? out : nil
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  let reader: PythonReader = { stateDict, archive in
    guard
      let vocab = stateDict["cond_stage_model.model.token_embedding.weight"]
        ?? stateDict["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let pos = stateDict["cond_stage_model.model.positional_embedding"]
        ?? stateDict["cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokenEmbed.parameters.copy(from: vocab, zip: archive, of: T.self)
    try positionEmbed.parameters.copy(from: pos, zip: archive, of: T.self)

    for i in 0..<numLayers {
      guard
        let layer_norm_1_weight =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).ln_1.weight"]
          ?? stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm1.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let layer_norm_1_bias =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).ln_1.bias"]
          ?? stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm1.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm1s[i].parameters(for: .weight).copy(
        from: layer_norm_1_weight, zip: archive, of: T.self)
      try layerNorm1s[i].parameters(for: .bias).copy(
        from: layer_norm_1_bias, zip: archive, of: T.self)

      if let in_proj_weight = try stateDict[
        "cond_stage_model.model.transformer.resblocks.\(i).attn.in_proj_weight"]?.inflate(
          from: archive, of: T.self),
        let in_proj_bias = try stateDict[
          "cond_stage_model.model.transformer.resblocks.\(i).attn.in_proj_bias"]?.inflate(
            from: archive, of: T.self)
      {
        toqueriess[i].parameters(for: .weight).copy(
          from: in_proj_weight[0..<(embeddingSize), 0..<in_proj_weight.shape[1]])
        toqueriess[i].parameters(for: .bias).copy(from: in_proj_bias[0..<(embeddingSize)])
        tokeyss[i].parameters(for: .weight).copy(
          from: in_proj_weight[(embeddingSize)..<(2 * embeddingSize), 0..<in_proj_weight.shape[1]])
        tokeyss[i].parameters(for: .bias).copy(
          from: in_proj_bias[(embeddingSize)..<(2 * embeddingSize)])
        tovaluess[i].parameters(for: .weight).copy(
          from: in_proj_weight[
            (2 * embeddingSize)..<in_proj_weight.shape[0], 0..<in_proj_weight.shape[1]])
        tovaluess[i].parameters(for: .bias).copy(
          from: in_proj_bias[(2 * embeddingSize)..<in_proj_bias.shape[0]])
      } else {
        guard
          let q_proj_weight = stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.q_proj.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let q_proj_bias = stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.q_proj.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try toqueriess[i].parameters(for: .weight).copy(
          from: q_proj_weight, zip: archive, of: T.self)
        try toqueriess[i].parameters(for: .bias).copy(from: q_proj_bias, zip: archive, of: T.self)
        guard
          let k_proj_weight = stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.k_proj.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let k_proj_bias = stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.k_proj.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try tokeyss[i].parameters(for: .weight).copy(from: k_proj_weight, zip: archive, of: T.self)
        try tokeyss[i].parameters(for: .bias).copy(from: k_proj_bias, zip: archive, of: T.self)
        guard
          let v_proj_weight = stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.v_proj.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let v_proj_bias = stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.v_proj.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try tovaluess[i].parameters(for: .weight).copy(
          from: v_proj_weight, zip: archive, of: T.self)
        try tovaluess[i].parameters(for: .bias).copy(from: v_proj_bias, zip: archive, of: T.self)
      }

      guard
        let out_proj_weight =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).attn.out_proj.weight"]
          ?? stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.out_proj.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let out_proj_bias =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).attn.out_proj.bias"]
          ?? stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.out_proj.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheadss[i].parameters(for: .weight).copy(
        from: out_proj_weight, zip: archive, of: T.self)
      try unifyheadss[i].parameters(for: .bias).copy(from: out_proj_bias, zip: archive, of: T.self)

      guard
        let layer_norm_2_weight =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).ln_2.weight"]
          ?? stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm2.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let layer_norm_2_bias =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).ln_2.bias"]
          ?? stateDict[
            "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm2.bias"]
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
            "cond_stage_model.model.transformer.resblocks.\(i).mlp.c_fc.weight"]
          ?? stateDict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc1.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let fc1_bias = stateDict["cond_stage_model.model.transformer.resblocks.\(i).mlp.c_fc.bias"]
          ?? stateDict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc1.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try fc1s[i].parameters(for: .weight).copy(from: fc1_weight, zip: archive, of: T.self)
      try fc1s[i].parameters(for: .bias).copy(from: fc1_bias, zip: archive, of: T.self)

      guard
        let fc2_weight =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).mlp.c_proj.weight"]
          ?? stateDict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc2.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let fc2_bias =
          stateDict[
            "cond_stage_model.model.transformer.resblocks.\(i).mlp.c_proj.bias"]
          ?? stateDict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc2.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try fc2s[i].parameters(for: .weight).copy(from: fc2_weight, zip: archive, of: T.self)
      try fc2s[i].parameters(for: .bias).copy(from: fc2_bias, zip: archive, of: T.self)
    }

    guard
      let final_layer_norm_weight = stateDict["cond_stage_model.model.ln_final.weight"]
        ?? stateDict["cond_stage_model.transformer.text_model.final_layer_norm.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let final_layer_norm_bias = stateDict["cond_stage_model.model.ln_final.bias"]
        ?? stateDict["cond_stage_model.transformer.text_model.final_layer_norm.bias"]
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
    if let penultimate = penultimate {
      if let hiddenState = hiddenState {
        model = Model(
          [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings],
          [penultimate, hiddenState, out], trainable: trainable)
      } else {
        model = Model(
          [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings],
          [penultimate, out], trainable: trainable)
      }
    } else if let hiddenState = hiddenState {
      model = Model(
        [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings],
        [hiddenState, out],
        trainable: trainable)
    } else {
      model = Model(
        [tokens, positions, causalAttentionMask, embedMask, injectedEmbeddings], [out],
        trainable: trainable)
    }
  } else {
    if let penultimate = penultimate {
      if let hiddenState = hiddenState {
        model = Model(
          [tokens, positions, causalAttentionMask], [penultimate, hiddenState, out],
          trainable: trainable)
      } else {
        model = Model(
          [tokens, positions, causalAttentionMask], [penultimate, out], trainable: trainable)
      }
    } else if let hiddenState = hiddenState {
      model = Model(
        [tokens, positions, causalAttentionMask], [hiddenState, out], trainable: trainable)
    } else {
      model = Model([tokens, positions, causalAttentionMask], [out], trainable: trainable)
    }
  }
  return (model, reader)
}
