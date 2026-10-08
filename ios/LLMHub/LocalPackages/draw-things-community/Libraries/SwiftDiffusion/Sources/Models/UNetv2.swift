import NNC

/// UNet for Stable Diffusion v2

func MiddleBlock(
  prefix: String,
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int),
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  x: Model.IO, emb: Model.IO, c: Model.IO, injectedAttentionKV: Bool
) -> (Model.IO, PythonReader) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false, flags: .Float16)
  var out = resBlock1(x, emb)
  let (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
  ) = SpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingLength,
    intermediateSize: channels * 4, injectIPAdapterLengths: [], upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, flags: .Float16, injectedAttentionKV: false,
    outputSpatialAttnInput: false)
  out = transformer(out, c)
  let (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false, flags: .Float16)
  out = resBlock2(out, emb)
  let reader: PythonReader = { stateDict, archive in
    let intermediateSize = channels * 4
    guard
      let in_layers_0_0_weight =
        stateDict["\(prefix).middle_block.0.in_layers.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_0_bias =
        stateDict["\(prefix).middle_block.0.in_layers.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerNorm1.parameters(for: .weight).copy(
      from: in_layers_0_0_weight, zip: archive, of: FloatType.self)
    try inLayerNorm1.parameters(for: .bias).copy(
      from: in_layers_0_0_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_0_2_weight =
        stateDict["\(prefix).middle_block.0.in_layers.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_2_bias =
        stateDict["\(prefix).middle_block.0.in_layers.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerConv2d1.parameters(for: .weight).copy(
      from: in_layers_0_2_weight, zip: archive, of: FloatType.self)
    try inLayerConv2d1.parameters(for: .bias).copy(
      from: in_layers_0_2_bias, zip: archive, of: FloatType.self)
    guard
      let emb_layers_0_1_weight =
        stateDict[
          "\(prefix).middle_block.0.emb_layers.1.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let emb_layers_0_1_bias =
        stateDict["\(prefix).middle_block.0.emb_layers.1.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try embLayer1.parameters(for: .weight).copy(
      from: emb_layers_0_1_weight, zip: archive, of: FloatType.self)
    try embLayer1.parameters(for: .bias).copy(
      from: emb_layers_0_1_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_0_weight =
        stateDict[
          "\(prefix).middle_block.0.out_layers.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_0_bias =
        stateDict[
          "\(prefix).middle_block.0.out_layers.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerNorm1.parameters(for: .weight).copy(
      from: out_layers_0_0_weight, zip: archive, of: FloatType.self)
    try outLayerNorm1.parameters(for: .bias).copy(
      from: out_layers_0_0_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_3_weight =
        stateDict[
          "\(prefix).middle_block.0.out_layers.3.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_3_bias =
        stateDict["\(prefix).middle_block.0.out_layers.3.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerConv2d1.parameters(for: .weight).copy(
      from: out_layers_0_3_weight, zip: archive, of: FloatType.self)
    try outLayerConv2d1.parameters(for: .bias).copy(
      from: out_layers_0_3_bias, zip: archive, of: FloatType.self)
    guard
      let norm_weight =
        stateDict["\(prefix).middle_block.1.norm.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm_bias =
        stateDict["\(prefix).middle_block.1.norm.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try norm.parameters(for: .weight).copy(
      from: norm_weight, zip: archive, of: FloatType.self)
    try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: FloatType.self)
    guard
      let proj_in_weight =
        stateDict["\(prefix).middle_block.1.proj_in.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let proj_in_bias =
        stateDict["\(prefix).middle_block.1.proj_in.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try projIn.parameters(for: .weight).copy(
      from: proj_in_weight, zip: archive, of: FloatType.self)
    try projIn.parameters(for: .bias).copy(
      from: proj_in_bias, zip: archive, of: FloatType.self)
    guard
      let attn1_to_k_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_k.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys1.parameters(for: .weight).copy(
      from: attn1_to_k_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_q_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_q.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries1.parameters(for: .weight).copy(
      from: attn1_to_q_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_v_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_v.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues1.parameters(for: .weight).copy(
      from: attn1_to_v_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_out_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let attn1_to_out_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_out.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try unifyheads1.parameters(for: .weight).copy(
      from: attn1_to_out_weight, zip: archive, of: FloatType.self)
    try unifyheads1.parameters(for: .bias).copy(
      from: attn1_to_out_bias, zip: archive, of: FloatType.self)
    guard
      let ff_net_0_proj_weight = try stateDict[
        "\(prefix).middle_block.1.transformer_blocks.0.ff.net.0.proj.weight"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_0_proj_bias = try stateDict[
        "\(prefix).middle_block.1.transformer_blocks.0.ff.net.0.proj.bias"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    fc10.parameters(for: .weight).copy(
      from: ff_net_0_proj_weight[0..<intermediateSize, 0..<ff_net_0_proj_weight.shape[1]])
    fc10.parameters(for: .bias).copy(
      from: ff_net_0_proj_bias[0..<intermediateSize])
    fc11.parameters(for: .weight).copy(
      from: ff_net_0_proj_weight[
        intermediateSize..<ff_net_0_proj_weight.shape[0], 0..<ff_net_0_proj_weight.shape[1]])
    fc11.parameters(for: .bias).copy(
      from: ff_net_0_proj_bias[intermediateSize..<ff_net_0_proj_bias.shape[0]])
    guard
      let ff_net_2_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.ff.net.2.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_2_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.ff.net.2.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tfc2.parameters(for: .weight).copy(
      from: ff_net_2_weight, zip: archive, of: FloatType.self)
    try tfc2.parameters(for: .bias).copy(
      from: ff_net_2_bias, zip: archive, of: FloatType.self)
    guard
      let attn2_to_k_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_k.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys2.parameters(for: .weight).copy(
      from: attn2_to_k_weight, zip: archive, of: FloatType.self)
    guard
      let attn2_to_q_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_q.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries2.parameters(for: .weight).copy(
      from: attn2_to_q_weight, zip: archive, of: FloatType.self)
    guard
      let attn2_to_v_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_v.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues2.parameters(for: .weight).copy(
      from: attn2_to_v_weight, zip: archive, of: FloatType.self)
    guard
      let attn2_to_out_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_out.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let attn2_to_out_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_out.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try unifyheads2.parameters(for: .weight).copy(
      from: attn2_to_out_weight, zip: archive, of: FloatType.self)
    try unifyheads2.parameters(for: .bias).copy(
      from: attn2_to_out_bias, zip: archive, of: FloatType.self)
    guard
      let norm1_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm1.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm1_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm1.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm1.parameters(for: .weight).copy(
      from: norm1_weight, zip: archive, of: FloatType.self)
    try layerNorm1.parameters(for: .bias).copy(
      from: norm1_bias, zip: archive, of: FloatType.self)
    guard
      let norm2_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm2.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm2_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm2.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm2.parameters(for: .weight).copy(
      from: norm2_weight, zip: archive, of: FloatType.self)
    try layerNorm2.parameters(for: .bias).copy(
      from: norm2_bias, zip: archive, of: FloatType.self)
    guard
      let norm3_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm3.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm3_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm3.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm3.parameters(for: .weight).copy(
      from: norm3_weight, zip: archive, of: FloatType.self)
    try layerNorm3.parameters(for: .bias).copy(
      from: norm3_bias, zip: archive, of: FloatType.self)
    guard
      let proj_out_weight =
        stateDict[
          "\(prefix).middle_block.1.proj_out.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let proj_out_bias =
        stateDict["\(prefix).middle_block.1.proj_out.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try projOut.parameters(for: .weight).copy(
      from: proj_out_weight, zip: archive, of: FloatType.self)
    try projOut.parameters(for: .bias).copy(
      from: proj_out_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_2_0_weight =
        stateDict["\(prefix).middle_block.2.in_layers.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_2_0_bias =
        stateDict["\(prefix).middle_block.2.in_layers.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerNorm2.parameters(for: .weight).copy(
      from: in_layers_2_0_weight, zip: archive, of: FloatType.self)
    try inLayerNorm2.parameters(for: .bias).copy(
      from: in_layers_2_0_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_2_2_weight =
        stateDict["\(prefix).middle_block.2.in_layers.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_2_2_bias =
        stateDict["\(prefix).middle_block.2.in_layers.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerConv2d2.parameters(for: .weight).copy(
      from: in_layers_2_2_weight, zip: archive, of: FloatType.self)
    try inLayerConv2d2.parameters(for: .bias).copy(
      from: in_layers_2_2_bias, zip: archive, of: FloatType.self)
    guard
      let emb_layers_2_1_weight =
        stateDict[
          "\(prefix).middle_block.2.emb_layers.1.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let emb_layers_2_1_bias =
        stateDict["\(prefix).middle_block.2.emb_layers.1.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try embLayer2.parameters(for: .weight).copy(
      from: emb_layers_2_1_weight, zip: archive, of: FloatType.self)
    try embLayer2.parameters(for: .bias).copy(
      from: emb_layers_2_1_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_2_0_weight =
        stateDict[
          "\(prefix).middle_block.2.out_layers.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_2_0_bias =
        stateDict[
          "\(prefix).middle_block.2.out_layers.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerNorm2.parameters(for: .weight).copy(
      from: out_layers_2_0_weight, zip: archive, of: FloatType.self)
    try outLayerNorm2.parameters(for: .bias).copy(
      from: out_layers_2_0_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_2_3_weight =
        stateDict[
          "\(prefix).middle_block.2.out_layers.3.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_2_3_bias =
        stateDict["\(prefix).middle_block.2.out_layers.3.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerConv2d2.parameters(for: .weight).copy(
      from: out_layers_2_3_weight, zip: archive, of: FloatType.self)
    try outLayerConv2d2.parameters(for: .bias).copy(
      from: out_layers_2_3_bias, zip: archive, of: FloatType.self)
  }
  return (out, reader)
}

private func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, x: Model.IO, emb: Model.IO, c: Model.IO
) -> ([Model.IO], Model.IO, PythonReader) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [PythonReader]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for _ in 0..<numRepeat {
      let (inputLayer, reader) = BlockLayer(
        prefix: "model.diffusion_model.input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: channel / numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: [],
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: .Float16,
        injectedAttentionKV: false, outputSpatialAttnInput: false)
      previousChannel = channel
      if attentionBlock {
        out = inputLayer(out, emb, c)
      } else {
        out = inputLayer(out, emb)
      }
      passLayers.append(out)
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: PythonReader = { stateDict, archive in
        guard
          let op_weight =
            stateDict["model.diffusion_model.input_blocks.\(downLayer).0.op.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let op_bias =
            stateDict["model.diffusion_model.input_blocks.\(downLayer).0.op.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try downsample.parameters(for: .weight).copy(
          from: op_weight, zip: archive, of: FloatType.self)
        try downsample.parameters(for: .bias).copy(
          from: op_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: PythonReader = { stateDict, archive in
    guard
      let input_blocks_0_0_weight =
        stateDict["model.diffusion_model.input_blocks.0.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let input_blocks_0_0_bias =
        stateDict["model.diffusion_model.input_blocks.0.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d.parameters(for: .weight).copy(
      from: input_blocks_0_0_weight, zip: archive, of: FloatType.self)
    try conv2d.parameters(for: .bias).copy(
      from: input_blocks_0_0_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  return (passLayers, out, reader)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, x: Model.IO, emb: Model.IO, c: Model.IO,
  inputs: [Model.IO]
) -> (Model.IO, PythonReader) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [PythonReader]()
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
      let (outputLayer, reader) = BlockLayer(
        prefix: "model.diffusion_model.output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: channel / numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: [],
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: .Float16,
        injectedAttentionKV: false, outputSpatialAttnInput: false)
      if attentionBlock {
        out = outputLayer(out, emb, c)
      } else {
        out = outputLayer(out, emb)
      }
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW
        )
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock ? 2 : 1
        let reader: PythonReader = { stateDict, archive in
          guard
            let op_weight =
              stateDict[
                "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight"
              ]
          else {
            throw UnpickleError.tensorNotFound
          }
          guard
            let op_bias =
              stateDict[
                "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias"
              ]
          else {
            throw UnpickleError.tensorNotFound
          }
          try conv2d.parameters(for: .weight).copy(
            from: op_weight, zip: archive, of: FloatType.self)
          try conv2d.parameters(for: .bias).copy(
            from: op_bias, zip: archive, of: FloatType.self)
        }
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: PythonReader = { stateDict, archive in
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  return (out, reader)
}

public func UNetv2(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel, injectControls: Bool,
  trainable: Bool? = nil
) -> (
  Model, PythonReader
) {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  var injectedControls = [Model.IO]()
  if injectControls {
    injectedControls = (0..<13).map { _ in Input() }
  }
  let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  var (inputs, inputBlocks, inputReader) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, x: x,
    emb: emb, c: c)
  var out = inputBlocks
  let (middleBlock, middleReader) = MiddleBlock(
    prefix: "model.diffusion_model",
    channels: 1280, numHeadChannels: 64, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, x: out, emb: emb, c: c, injectedAttentionKV: false)
  out = middleBlock
  if injectControls {
    out = out + injectedControls[12]
    precondition(inputs.count + 1 == injectedControls.count)
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + injectedControls[i]
    }
  }
  let (outputBlocks, outputReader) = OutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, x: out,
    emb: emb, c: c, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outNorm(out)
  out = out.swish()
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = outConv2d(out)
  let reader: PythonReader = { stateDict, archive in
    guard
      let time_embed_0_weight =
        stateDict["model.diffusion_model.time_embed.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_0_bias =
        stateDict["model.diffusion_model.time_embed.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_weight =
        stateDict["model.diffusion_model.time_embed.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_bias =
        stateDict["model.diffusion_model.time_embed.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try fc0.parameters(for: .weight).copy(
      from: time_embed_0_weight, zip: archive, of: FloatType.self)
    try fc0.parameters(for: .bias).copy(
      from: time_embed_0_bias, zip: archive, of: FloatType.self)
    try fc2.parameters(for: .weight).copy(
      from: time_embed_2_weight, zip: archive, of: FloatType.self)
    try fc2.parameters(for: .bias).copy(
      from: time_embed_2_bias, zip: archive, of: FloatType.self)
    try inputReader(stateDict, archive)
    try middleReader(stateDict, archive)
    try outputReader(stateDict, archive)
    guard let out_0_weight = stateDict["model.diffusion_model.out.0.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let out_0_bias = stateDict["model.diffusion_model.out.0.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try outNorm.parameters(for: .weight).copy(
      from: out_0_weight, zip: archive, of: FloatType.self)
    try outNorm.parameters(for: .bias).copy(
      from: out_0_bias, zip: archive, of: FloatType.self)
    guard let out_2_weight = stateDict["model.diffusion_model.out.2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let out_2_bias = stateDict["model.diffusion_model.out.2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try outConv2d.parameters(for: .weight).copy(
      from: out_2_weight, zip: archive, of: FloatType.self)
    try outConv2d.parameters(for: .bias).copy(
      from: out_2_bias, zip: archive, of: FloatType.self)
  }
  var modelInputs: [Model.IO] = [x, t_emb, c]
  modelInputs.append(contentsOf: injectedControls)
  return (Model(modelInputs, [out], trainable: trainable), reader)
}
