import DiffusionMappings
import NNC

func InputHintBlocks(modelChannel: Int, hint: Model.IO) -> (
  Model.IO, PythonReader, ModelWeightMapper
) {
  let conv2d0 = Convolution(
    groups: 1, filters: 16, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d0(hint)
  out = Swish()(out)
  let conv2d1 = Convolution(
    groups: 1, filters: 16, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2d1(out)
  out = Swish()(out)
  let conv2d2 = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2d2(out)
  out = Swish()(out)
  let conv2d3 = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2d3(out)
  out = Swish()(out)
  let conv2d4 = Convolution(
    groups: 1, filters: 96, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2d4(out)
  out = Swish()(out)
  let conv2d5 = Convolution(
    groups: 1, filters: 96, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2d5(out)
  out = Swish()(out)
  let conv2d6 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2d6(out)
  out = Swish()(out)
  let conv2d7 = Convolution(
    groups: 1, filters: modelChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2d7(out)
  let reader: PythonReader = { stateDict, archive in
    guard let input_hint_block_0_weight = stateDict["model.control_model.input_hint_block.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_0_bias = stateDict["model.control_model.input_hint_block.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d0.weight.copy(from: input_hint_block_0_weight, zip: archive, of: FloatType.self)
    try conv2d0.bias.copy(from: input_hint_block_0_bias, zip: archive, of: FloatType.self)
    guard let input_hint_block_2_weight = stateDict["model.control_model.input_hint_block.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_2_bias = stateDict["model.control_model.input_hint_block.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d1.weight.copy(from: input_hint_block_2_weight, zip: archive, of: FloatType.self)
    try conv2d1.bias.copy(from: input_hint_block_2_bias, zip: archive, of: FloatType.self)
    guard let input_hint_block_4_weight = stateDict["model.control_model.input_hint_block.4.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_4_bias = stateDict["model.control_model.input_hint_block.4.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d2.weight.copy(from: input_hint_block_4_weight, zip: archive, of: FloatType.self)
    try conv2d2.bias.copy(from: input_hint_block_4_bias, zip: archive, of: FloatType.self)
    guard let input_hint_block_6_weight = stateDict["model.control_model.input_hint_block.6.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_6_bias = stateDict["model.control_model.input_hint_block.6.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d3.weight.copy(from: input_hint_block_6_weight, zip: archive, of: FloatType.self)
    try conv2d3.bias.copy(from: input_hint_block_6_bias, zip: archive, of: FloatType.self)
    guard let input_hint_block_8_weight = stateDict["model.control_model.input_hint_block.8.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_8_bias = stateDict["model.control_model.input_hint_block.8.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d4.weight.copy(from: input_hint_block_8_weight, zip: archive, of: FloatType.self)
    try conv2d4.bias.copy(from: input_hint_block_8_bias, zip: archive, of: FloatType.self)
    guard
      let input_hint_block_10_weight = stateDict["model.control_model.input_hint_block.10.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_10_bias = stateDict["model.control_model.input_hint_block.10.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d5.weight.copy(from: input_hint_block_10_weight, zip: archive, of: FloatType.self)
    try conv2d5.bias.copy(from: input_hint_block_10_bias, zip: archive, of: FloatType.self)
    guard
      let input_hint_block_12_weight = stateDict["model.control_model.input_hint_block.12.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_12_bias = stateDict["model.control_model.input_hint_block.12.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d6.weight.copy(from: input_hint_block_12_weight, zip: archive, of: FloatType.self)
    try conv2d6.bias.copy(from: input_hint_block_12_bias, zip: archive, of: FloatType.self)
    guard
      let input_hint_block_14_weight = stateDict["model.control_model.input_hint_block.14.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_hint_block_14_bias = stateDict["model.control_model.input_hint_block.14.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d7.weight.copy(from: input_hint_block_14_weight, zip: archive, of: FloatType.self)
    try conv2d7.bias.copy(from: input_hint_block_14_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["controlnet_cond_embedding.conv_in.weight"] = [conv2d0.weight.name]
      mapping["controlnet_cond_embedding.conv_in.bias"] = [conv2d0.bias.name]
      mapping["controlnet_cond_embedding.blocks.0.weight"] = [conv2d1.weight.name]
      mapping["controlnet_cond_embedding.blocks.0.bias"] = [conv2d1.bias.name]
      mapping["controlnet_cond_embedding.blocks.1.weight"] = [conv2d2.weight.name]
      mapping["controlnet_cond_embedding.blocks.1.bias"] = [conv2d2.bias.name]
      mapping["controlnet_cond_embedding.blocks.2.weight"] = [conv2d3.weight.name]
      mapping["controlnet_cond_embedding.blocks.2.bias"] = [conv2d3.bias.name]
      mapping["controlnet_cond_embedding.blocks.3.weight"] = [conv2d4.weight.name]
      mapping["controlnet_cond_embedding.blocks.3.bias"] = [conv2d4.bias.name]
      mapping["controlnet_cond_embedding.blocks.4.weight"] = [conv2d5.weight.name]
      mapping["controlnet_cond_embedding.blocks.4.bias"] = [conv2d5.bias.name]
      mapping["controlnet_cond_embedding.blocks.5.weight"] = [conv2d6.weight.name]
      mapping["controlnet_cond_embedding.blocks.5.bias"] = [conv2d6.bias.name]
      mapping["controlnet_cond_embedding.conv_out.weight"] = [conv2d7.weight.name]
      mapping["controlnet_cond_embedding.conv_out.bias"] = [conv2d7.bias.name]
    case .generativeModels:
      mapping["model.control_model.input_hint_block.0.weight"] = [conv2d0.weight.name]
      mapping["model.control_model.input_hint_block.0.bias"] = [conv2d0.bias.name]
      mapping["model.control_model.input_hint_block.2.weight"] = [conv2d1.weight.name]
      mapping["model.control_model.input_hint_block.2.bias"] = [conv2d1.bias.name]
      mapping["model.control_model.input_hint_block.4.weight"] = [conv2d2.weight.name]
      mapping["model.control_model.input_hint_block.4.bias"] = [conv2d2.bias.name]
      mapping["model.control_model.input_hint_block.6.weight"] = [conv2d3.weight.name]
      mapping["model.control_model.input_hint_block.6.bias"] = [conv2d3.bias.name]
      mapping["model.control_model.input_hint_block.8.weight"] = [conv2d4.weight.name]
      mapping["model.control_model.input_hint_block.8.bias"] = [conv2d4.bias.name]
      mapping["model.control_model.input_hint_block.10.weight"] = [conv2d5.weight.name]
      mapping["model.control_model.input_hint_block.10.bias"] = [conv2d5.bias.name]
      mapping["model.control_model.input_hint_block.12.weight"] = [conv2d6.weight.name]
      mapping["model.control_model.input_hint_block.12.bias"] = [conv2d6.bias.name]
      mapping["model.control_model.input_hint_block.14.weight"] = [conv2d7.weight.name]
      mapping["model.control_model.input_hint_block.14.bias"] = [conv2d7.bias.name]
    }
    return mapping
  }
  return (out, reader, mapper)
}

private func InputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, x: Model.IO, hint: Model.IO, emb: Model.IO, c: Model.IO
) -> ([(Model.IO, Int)], Model.IO, PythonReader) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x) + hint
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [PythonReader]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [(out, 320)]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for _ in 0..<numRepeat {
      let (inputLayer, reader) = BlockLayer(
        prefix: "model.control_model.input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
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
      passLayers.append((out, channel))
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append((out, channel))
      let downLayer = layerStart
      let reader: PythonReader = { stateDict, archive in
        guard
          let op_weight =
            stateDict["model.control_model.input_blocks.\(downLayer).0.op.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let op_bias =
            stateDict["model.control_model.input_blocks.\(downLayer).0.op.bias"]
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
        stateDict["model.control_model.input_blocks.0.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let input_blocks_0_0_bias =
        stateDict["model.control_model.input_blocks.0.0.bias"]
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

private func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, x: Model.IO, hint: Model.IO, emb: Model.IO, c: Model.IO
) -> ([(Model.IO, Int)], Model.IO, PythonReader) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x) + hint
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [PythonReader]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [(out, 320)]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for _ in 0..<numRepeat {
      let (inputLayer, reader) = BlockLayer(
        prefix: "model.control_model.input_blocks",
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
      passLayers.append((out, channel))
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append((out, channel))
      let downLayer = layerStart
      let reader: PythonReader = { stateDict, archive in
        guard
          let op_weight =
            stateDict["model.control_model.input_blocks.\(downLayer).0.op.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let op_bias =
            stateDict["model.control_model.input_blocks.\(downLayer).0.op.bias"]
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
        stateDict["model.control_model.input_blocks.0.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let input_blocks_0_0_bias =
        stateDict["model.control_model.input_blocks.0.0.bias"]
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

public func HintNet(channels: Int) -> (Model, PythonReader, ModelWeightMapper) {
  let hint = Input()
  let (out, reader, mapper) = InputHintBlocks(modelChannel: channels, hint: hint)
  return (Model([hint], [out]), reader, mapper)
}

public func ControlNet(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  usesFlashAttention: FlashAttentionLevel
)
  -> (Model, PythonReader)
{
  let x = Input()
  let hint = Input()
  let t_emb = Input()
  let c = Input()
  let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  let (inputs, inputBlocks, inputReader) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: attentionRes, upcastAttention: false, usesFlashAttention: usesFlashAttention,
    x: x, hint: hint, emb: emb, c: c)
  var out = inputBlocks
  let (middleBlock, _, middleReader, _) = MiddleBlock(
    prefix: "model.control_model",
    channels: 1280, numHeads: 8, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength, injectIPAdapterLengths: [],
    upcastAttention: false, usesFlashAttention: usesFlashAttention, x: out, emb: emb, c: c,
    injectedAttentionKVs: [], outputSpatialAttnInput: false)
  out = middleBlock
  var zeroConvs = [Model]()
  var outputs = [Model.IO]()
  for i in 0..<inputs.count {
    let channel = inputs[i].1
    let zeroConv = Convolution(
      groups: 1, filters: channel, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW)
    outputs.append(zeroConv(inputs[i].0))
    zeroConvs.append(zeroConv)
  }
  let middleBlockOut = Convolution(
    groups: 1, filters: 1280, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  outputs.append(middleBlockOut(out))
  let reader: PythonReader = { stateDict, archive in
    guard
      let time_embed_0_weight =
        stateDict["model.control_model.time_embed.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_0_bias =
        stateDict["model.control_model.time_embed.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_weight =
        stateDict["model.control_model.time_embed.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_bias =
        stateDict["model.control_model.time_embed.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try fc0.weight.copy(
      from: time_embed_0_weight, zip: archive, of: FloatType.self)
    try fc0.bias.copy(
      from: time_embed_0_bias, zip: archive, of: FloatType.self)
    try fc2.weight.copy(
      from: time_embed_2_weight, zip: archive, of: FloatType.self)
    try fc2.bias.copy(
      from: time_embed_2_bias, zip: archive, of: FloatType.self)
    try inputReader(stateDict, archive)
    try middleReader(stateDict, archive)
    for (i, zeroConv) in zeroConvs.enumerated() {
      guard
        let zero_convs_weight =
          stateDict["model.control_model.zero_convs.\(i).0.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let zero_convs_bias =
          stateDict["model.control_model.zero_convs.\(i).0.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try zeroConv.weight.copy(from: zero_convs_weight, zip: archive, of: FloatType.self)
      try zeroConv.bias.copy(from: zero_convs_bias, zip: archive, of: FloatType.self)
    }
    guard
      let middle_block_out_0_weight =
        stateDict["model.control_model.middle_block_out.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let middle_block_out_0_bias =
        stateDict["model.control_model.middle_block_out.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try middleBlockOut.weight.copy(
      from: middle_block_out_0_weight, zip: archive, of: FloatType.self)
    try middleBlockOut.bias.copy(from: middle_block_out_0_bias, zip: archive, of: FloatType.self)
  }
  return (Model([x, hint, t_emb, c], outputs), reader)
}

public func ControlNetv2(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel
)
  -> (Model, PythonReader)
{
  let x = Input()
  let hint = Input()
  let t_emb = Input()
  let c = Input()
  let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  let (inputs, inputBlocks, inputReader) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: attentionRes, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, x: x, hint: hint, emb: emb, c: c)
  var out = inputBlocks
  let (middleBlock, middleReader) = MiddleBlock(
    prefix: "model.control_model",
    channels: 1280, numHeadChannels: 64, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, x: out, emb: emb, c: c, injectedAttentionKV: false)
  out = middleBlock
  var zeroConvs = [Model]()
  var outputs = [Model.IO]()
  for i in 0..<inputs.count {
    let channel = inputs[i].1
    let zeroConv = Convolution(
      groups: 1, filters: channel, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW)
    outputs.append(zeroConv(inputs[i].0))
    zeroConvs.append(zeroConv)
  }
  let middleBlockOut = Convolution(
    groups: 1, filters: 1280, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  outputs.append(middleBlockOut(out))
  let reader: PythonReader = { stateDict, archive in
    guard
      let time_embed_0_weight =
        stateDict["model.control_model.time_embed.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_0_bias =
        stateDict["model.control_model.time_embed.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_weight =
        stateDict["model.control_model.time_embed.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_bias =
        stateDict["model.control_model.time_embed.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try fc0.weight.copy(
      from: time_embed_0_weight, zip: archive, of: FloatType.self)
    try fc0.bias.copy(
      from: time_embed_0_bias, zip: archive, of: FloatType.self)
    try fc2.weight.copy(
      from: time_embed_2_weight, zip: archive, of: FloatType.self)
    try fc2.bias.copy(
      from: time_embed_2_bias, zip: archive, of: FloatType.self)
    try inputReader(stateDict, archive)
    try middleReader(stateDict, archive)
    for (i, zeroConv) in zeroConvs.enumerated() {
      guard
        let zero_convs_weight =
          stateDict["model.control_model.zero_convs.\(i).0.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let zero_convs_bias =
          stateDict["model.control_model.zero_convs.\(i).0.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try zeroConv.weight.copy(from: zero_convs_weight, zip: archive, of: FloatType.self)
      try zeroConv.bias.copy(from: zero_convs_bias, zip: archive, of: FloatType.self)
    }
    guard
      let middle_block_out_0_weight =
        stateDict["model.control_model.middle_block_out.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let middle_block_out_0_bias =
        stateDict["model.control_model.middle_block_out.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try middleBlockOut.weight.copy(
      from: middle_block_out_0_weight, zip: archive, of: FloatType.self)
    try middleBlockOut.bias.copy(from: middle_block_out_0_bias, zip: archive, of: FloatType.self)
  }
  return (Model([x, hint, t_emb, c], outputs), reader)
}
