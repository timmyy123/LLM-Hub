// MiniMax H3 audio autoencoder graph definitions.
import NNC

private func H3ConditioningAudioSnake(channels: Int, name: String) -> Model {
  let x = Input()
  let alpha = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_alpha")
  let reciprocal = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false,
    name: "\(name)_reciprocal")
  let out = x + reciprocal .* (alpha .* x).sin().pow(2)
  return Model([x], [out])
}

private func H3ConditioningAudioResidualUnit(
  channels: Int, dilation: Int, name: String
) -> Model {
  let x = Input()
  let snake1 = H3ConditioningAudioSnake(channels: channels, name: "\(name)_snake1")
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [1, 7], dilation: [1, dilation],
    hint: Hint(
      stride: [1, 1],
      border: Hint.Border(
        begin: [0, 3 * dilation], end: [0, 3 * dilation])), name: "\(name)_conv1")
  let snake2 = H3ConditioningAudioSnake(channels: channels, name: "\(name)_snake2")
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), name: "\(name)_conv2")
  let out = x + conv2(snake2(conv1(snake1(x))))
  return Model([x], [out])
}

private func H3ConditioningAudioEncoderBlock(
  channels: Int, outputChannels: Int, stride: Int, name: String
) -> Model {
  let x = Input()
  var out: Model.IO = x
  for (index, dilation) in [1, 3, 9].enumerated() {
    let unit = H3ConditioningAudioResidualUnit(
      channels: channels, dilation: dilation, name: "\(name)_res\(index)")
    out = unit(out)
  }
  let snake = H3ConditioningAudioSnake(channels: channels, name: "\(name)_snake")
  let downsample = Convolution(
    groups: 1, filters: outputChannels, filterSize: [1, 2 * stride],
    hint: Hint(
      stride: [1, stride],
      border: Hint.Border(
        begin: [0, (stride + 1) / 2], end: [0, (stride + 1) / 2])),
    name: "\(name)_downsample")
  out = downsample(snake(out))
  return Model([x], [out])
}

private func H3ConditioningAudioProjection(length: Int) -> Model {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  let qkv = Dense(count: 6_144, name: "qkv")
  let projectedQKV = qkv(norm1(x))
  let q = projectedQKV.reshaped(
    [1, length, 2_048], offset: [0, 0, 0],
    strides: [length * 6_144, 6_144, 1]
  ).contiguous().reshaped(.NHWC(1, length, 8, 256))
  let k = projectedQKV.reshaped(
    [1, length, 2_048], offset: [0, 0, 2_048],
    strides: [length * 6_144, 6_144, 1]
  ).contiguous().reshaped(.NHWC(1, length, 8, 256))
  let v = projectedQKV.reshaped(
    [1, length, 2_048], offset: [0, 0, 4_096],
    strides: [length * 6_144, 6_144, 1]
  ).contiguous().reshaped(.NHWC(1, length, 8, 256))
  var attended = ScaledDotProductAttention(
    scale: 1 / Float(256).squareRoot(), isCausal: true)(
      q.to(.BFloat16), k.to(.BFloat16), v.to(.BFloat16)
    ).to(.Float32)
  attended = attended.reduced(.mean, axis: [2]).reshaped([1, length, 32, 8])
    .reduced(.mean, axis: [3]).reshaped([1, length, 32])
  let attentionOutput = Dense(count: 32, name: "attention_output")
  attended = attentionOutput(attended)
  let norm3 = LayerNorm(epsilon: 1e-5, axis: [2], name: "norm3")
  let inputProjection = Dense(count: 32, name: "input_projection")
  var out = inputProjection(norm3(x)) + attended
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  let mlpNorm = LayerNorm(epsilon: 1e-5, axis: [2], name: "mlp_norm")
  let mlpGate = Dense(count: 64, name: "mlp_gate")
  let mlpUp = Dense(count: 64, name: "mlp_up")
  let mlpDown = Dense(count: 32, name: "mlp_down")
  let mlpInput = mlpNorm(norm2(out))
  out = out + mlpDown(mlpGate(mlpInput).GELU(approximate: .tanh) .* mlpUp(mlpInput))
  return Model([x], [out])
}

/// Encodes one 32 kHz channel from NCHW [1, 1, 1, inputLength] into
/// [1, inputLength / 800, 64], concatenating the mean_proj and logs_proj outputs.
/// The caller pads the waveform to a multiple of 800 samples and encodes stereo channels separately.
public func MiniMaxH3AudioEncoder(inputLength: Int) -> Model {
  precondition(inputLength > 0 && inputLength % 800 == 0)
  let x = Input()
  let input = Convolution(
    groups: 1, filters: 64, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: "encoder_input")
  var out = input(x)
  var width = inputLength
  var channels = 64
  for (level, stride) in [2, 4, 4, 5, 5].enumerated() {
    let block = H3ConditioningAudioEncoderBlock(
      channels: channels, outputChannels: channels * 2, stride: stride,
      name: "encoder_block_\(level)")
    out = block(out)
    channels *= 2
    width /= stride
  }
  let finalSnake = H3ConditioningAudioSnake(channels: channels, name: "encoder_final_snake")
  let finalConv = Convolution(
    groups: 1, filters: 2_048, filterSize: [1, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 1], end: [0, 1])),
    name: "encoder_final_conv")
  out = finalConv(finalSnake(out)).permuted(0, 3, 1, 2).copied()
    .reshaped([1, width, 2_048])
  let projection = H3ConditioningAudioProjection(length: width)
  out = projection(out)
  let mean = Dense(count: 32, name: "mean_proj")
  let logs = Dense(count: 32, name: "logs_proj")
  let output = Functional.concat(axis: 2, mean(out), logs(out))
  return Model([x], [output])
}

func H3AudioSnakeBeta(channels: Int, name: String) -> Model {
  let x = Input()
  let alpha = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_alpha")
  let beta = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_beta")
  let out = x + beta .* (x .* alpha).sin().pow(2)
  return Model([x], [out])
}

func H3AudioActivation(channels: Int, width: Int, name: String) -> Model {
  let x = Input()
  let ratio = 2
  let kernelSize = 12
  let upWidth = width * ratio
  let pad = kernelSize / ratio - 1
  let inputWidth = width + 2 * pad
  let rawWidth = (inputWidth - 1) * ratio + kernelSize
  let padLeft = pad * ratio + (kernelSize - ratio) / 2
  let upsample = ConvolutionTranspose(
    groups: 1, filters: 1, filterSize: [1, kernelSize], noBias: true,
    hint: Hint(stride: [1, ratio]), name: "\(name)_upsample")
  let snake = H3AudioSnakeBeta(channels: channels, name: "\(name)_snake")
  let downsample = Convolution(
    groups: 1, filters: 1, filterSize: [1, kernelSize], noBias: true,
    hint: Hint(stride: [1, ratio]), name: "\(name)_downsample")
  var out = x.reshaped([channels, 1, 1, width])
  out = out.padded(.replicate, begin: [0, 0, 0, pad], end: [0, 0, 0, pad])
  out = Float(ratio) * upsample(out)
  out = out.reshaped(
    [channels, 1, 1, upWidth], offset: [0, 0, 0, padLeft],
    strides: [rawWidth, rawWidth, rawWidth, 1]
  ).contiguous()
  out = snake(out.reshaped([1, channels, 1, upWidth]))
  out = out.reshaped([channels, 1, 1, upWidth])
  out = downsample(
    out.padded(.replicate, begin: [0, 0, 0, 5], end: [0, 0, 0, 6]))
  out = out.reshaped([1, channels, 1, width])
  return Model([x], [out])
}

func H3AudioAMPBlock(
  channels: Int, width: Int, kernelSize: Int, name: String
) -> Model {
  let x = Input()
  var out: Model.IO = x
  for (index, dilation) in [1, 3, 5].enumerated() {
    let residual = out
    let act1 = H3AudioActivation(
      channels: channels, width: width, name: "\(name)_a\(index)_0")
    let conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize], dilation: [1, dilation],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) * dilation / 2],
          end: [0, (kernelSize - 1) * dilation / 2])), name: "\(name)_c\(index)_0")
    let act2 = H3AudioActivation(
      channels: channels, width: width, name: "\(name)_a\(index)_1")
    let conv2 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) / 2], end: [0, (kernelSize - 1) / 2])),
      name: "\(name)_c\(index)_1")
    out = residual + conv2(act2(conv1(act1(out))))
  }
  return Model([x], [out])
}

public func MiniMaxH3AudioDecoder(latentWidth: Int) -> Model {
  let x = Input()
  let input = Convolution(
    groups: 1, filters: 2_048, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), name: "dec_in_proj")
  let pre = Convolution(
    groups: 1, filters: 1_024, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: "audio_conv_pre")
  var out = pre(input(x.permuted(0, 2, 1).contiguous().reshaped(.NCHW(1, 32, 1, latentWidth))))
  let rates = [5, 5, 2, 2, 2, 2, 2]
  let kernels = [9, 9, 4, 4, 4, 4, 4]
  let resKernels = [3, 7, 11]
  var width = latentWidth
  for layer in 0..<rates.count {
    let channels = 1_024 / (1 << (layer + 1))
    let padding = (kernels[layer] - rates[layer]) / 2
    let up = ConvolutionTranspose(
      groups: 1, filters: channels, filterSize: [1, kernels[layer]],
      hint: Hint(
        stride: [1, rates[layer]],
        border: Hint.Border(begin: [0, padding], end: [0, padding])),
      name: "audio_up_\(layer)")
    out = up(out)
    width *= rates[layer]
    var branches = [Model.IO]()
    for branch in 0..<resKernels.count {
      let blockIndex = layer * resKernels.count + branch
      let block = H3AudioAMPBlock(
        channels: channels, width: width,
        kernelSize: resKernels[branch], name: "audio_amp_\(blockIndex)")
      branches.append(block(out))
    }
    out = (1.0 / Float(branches.count)) * branches.dropFirst().reduce(branches[0]) { $0 + $1 }
  }
  let postActivation = H3AudioActivation(channels: 8, width: width, name: "audio_post")
  out = postActivation(out)
  let post = Convolution(
    groups: 1, filters: 1, filterSize: [1, 7], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: "audio_conv_post")
  out = post(out).clamped(-1...1)
  return Model([x], [out])
}
