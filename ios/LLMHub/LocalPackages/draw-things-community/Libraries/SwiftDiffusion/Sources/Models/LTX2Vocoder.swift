import DiffusionMappings
import Foundation
import NNC

enum LTX2VocoderActivation: Equatable {
  case leakyReLU
  case activation1d
}

enum LTX2VocoderFinalActivation: Equatable {
  case clamp
  case tanh
}

private func SnakeBeta(prefix: String, channels: Int, name: String) -> (ModelWeightMapper, Model) {
  let x = Input()
  let alpha = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_alpha")
  let beta = Parameter<Float>(
    .GPU(0), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_beta")
  let out = x + beta .* (x .* alpha).sin().pow(2)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).alpha"] = [alpha.weight.name]
    mapping["\(prefix).beta"] = [beta.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func Activation1d(prefix: String, channels: Int, width: Int, name: String)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let upWidth = width * 2
  let upPad = 12 / 2 - 1
  let upInputWidth = width + 2 * upPad
  let upRawWidth = (upInputWidth - 1) * 2 + 12
  let upPadLeft = upPad * 2 + (12 - 2) / 2
  let upsample = ConvolutionTranspose(
    groups: 1, filters: 1, filterSize: [1, 12], noBias: true, hint: Hint(stride: [1, 2]),
    format: .OIHW, name: "\(name)_upsample")
  let (snakeMapper, snake) = SnakeBeta(
    prefix: "\(prefix).act", channels: channels, name: "\(name)_snake")
  let downsample = Convolution(
    groups: 1, filters: 1, filterSize: [1, 12], noBias: true, hint: Hint(stride: [1, 2]),
    format: .OIHW, name: "\(name)_downsample")
  var out = x.reshaped([channels, 1, 1, width])
  out = out.padded(.replicate, begin: [0, 0, 0, upPad], end: [0, 0, 0, upPad])
  out = Float(2) * upsample(out)
  out = out.reshaped(
    [channels, 1, 1, upWidth], offset: [0, 0, 0, upPadLeft],
    strides: [upRawWidth, upRawWidth, upRawWidth, 1]
  ).contiguous()
  out = out.reshaped([1, channels, 1, upWidth])
  out = snake(out)
  out = out.reshaped([channels, 1, 1, upWidth])
  out = downsample(out.padded(.replicate, begin: [0, 0, 0, 5], end: [0, 0, 0, 6]))
  out = out.reshaped([1, channels, 1, width])
  let mapper: ModelWeightMapper = { format in
    var mapping = snakeMapper(format)
    mapping["\(prefix).upsample.filter"] = [upsample.weight.name]
    mapping["\(prefix).downsample.lowpass.filter"] = [downsample.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func ResBlock1(
  prefix: String, channels: Int, width: Int, kernelSize: Int, dilations: [Int],
  activation: LTX2VocoderActivation, name: String
) -> (ModelWeightMapper, Model) {
  let x = Input()
  var out: Model.IO = x
  var mappers = [ModelWeightMapper]()
  for (i, dilation) in dilations.enumerated() {
    let residual = out
    switch activation {
    case .leakyReLU:
      out = out.leakyReLU(negativeSlope: 0.1)
    case .activation1d:
      let (mapper, activation1d) = Activation1d(
        prefix: "\(prefix).acts1.\(i)", channels: channels, width: width,
        name: name.isEmpty ? "amp_act1_\(i)" : "\(name)_amp_act1_\(i)")
      out = activation1d(out)
      mappers.append(mapper)
    }
    let conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize], dilation: [1, dilation],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) * dilation / 2], end: [0, (kernelSize - 1) * dilation / 2])),
      format: .OIHW,
      name: activation == .activation1d
        ? (name.isEmpty ? "amp_resnet_conv1_\(i)" : "\(name)_amp_resnet_conv1_\(i)")
        : "resnet_conv1")
    out = conv1(out)
    switch activation {
    case .leakyReLU:
      out = out.leakyReLU(negativeSlope: 0.1)
    case .activation1d:
      let (mapper, activation1d) = Activation1d(
        prefix: "\(prefix).acts2.\(i)", channels: channels, width: width,
        name: name.isEmpty ? "amp_act2_\(i)" : "\(name)_amp_act2_\(i)")
      out = activation1d(out)
      mappers.append(mapper)
    }
    let conv2 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(begin: [0, (kernelSize - 1) / 2], end: [0, (kernelSize - 1) / 2])),
      format: .OIHW,
      name: activation == .activation1d
        ? (name.isEmpty ? "amp_resnet_conv2_\(i)" : "\(name)_amp_resnet_conv2_\(i)")
        : "resnet_conv2")
    out = conv2(out) + residual
    let idx = i
    mappers.append { _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).convs1.\(idx).weight"] = [conv1.weight.name]
      mapping["\(prefix).convs1.\(idx).bias"] = [conv1.bias.name]
      mapping["\(prefix).convs2.\(idx).weight"] = [conv2.weight.name]
      mapping["\(prefix).convs2.\(idx).bias"] = [conv2.bias.name]
      return mapping
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func LTX2Vocoder(
  width: Int, initialChannels: Int,
  layers: [(channels: Int, kernelSize: Int, stride: Int, padding: Int)], resblockKernelSizes: [Int],
  resblockDilations: [[Int]], activation: LTX2VocoderActivation, finalConvBias: Bool,
  finalActivation: LTX2VocoderFinalActivation, applyFinalActivation: Bool, prefix: String,
  name: String
)
  -> (
    ModelWeightMapper, Model
  )
{
  precondition(!resblockKernelSizes.isEmpty)
  precondition(resblockKernelSizes.count == resblockDilations.count)
  let x = Input()
  let convPre = Convolution(
    groups: 1, filters: initialChannels, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])), format: .OIHW,
    name: name.isEmpty ? "conv_pre" : "\(name)_conv_pre")
  var out = convPre(x)
  var currentWidth = width
  var mappers = [ModelWeightMapper]()
  for (i, layer) in layers.enumerated() {
    if activation == .leakyReLU {
      out = out.leakyReLU(negativeSlope: 0.1)
    }
    let up = ConvolutionTranspose(
      groups: 1, filters: layer.channels, filterSize: [1, layer.kernelSize],
      hint: Hint(
        stride: [1, layer.stride],
        border: Hint.Border(begin: [0, layer.padding], end: [0, layer.padding])), format: .OIHW,
      name: name.isEmpty ? "up" : "\(name)_up")
    out = up(out)
    currentWidth *= layer.stride
    let upIdx = i
    mappers.append { _ in
      var mapping = ModelWeightMapping()
      mapping["ups.\(upIdx).weight"] = [up.weight.name]
      mapping["ups.\(upIdx).bias"] = [up.bias.name]
      return mapping
    }
    let resBlocks = resblockKernelSizes.enumerated().map {
      ResBlock1(
        prefix: "resblocks.\(i * resblockKernelSizes.count + $0.offset)", channels: layer.channels,
        width: currentWidth, kernelSize: $0.element, dilations: resblockDilations[$0.offset],
        activation: activation, name: name)
    }
    for resBlock in resBlocks {
      mappers.append(resBlock.0)
    }
    out =
      (1.0 / Float(resBlocks.count))
      * resBlocks.dropFirst().reduce(resBlocks[0].1(out)) { $0 + $1.1(out) }
  }
  if activation == .leakyReLU {
    out = out.leakyReLU(negativeSlope: 0.01)
  } else {
    let (mapper, actPost) = Activation1d(
      prefix: "act_post", channels: layers.last?.channels ?? initialChannels, width: currentWidth,
      name: name.isEmpty ? "act_post" : "\(name)_act_post")
    out = actPost(out)
    mappers.append(mapper)
  }
  let convPost = Convolution(
    groups: 1, filters: 2, filterSize: [1, 7], noBias: !finalConvBias,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])), format: .OIHW,
    name: name.isEmpty ? "conv_post" : "\(name)_conv_post")
  out = convPost(out)
  if applyFinalActivation {
    switch finalActivation {
    case .clamp:
      out = out.clamped(-1...1)
    case .tanh:
      out = out.tanh()
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_pre.weight"] = [convPre.weight.name]
    mapping["conv_pre.bias"] = [convPre.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_post.weight"] = [convPost.weight.name]
    if finalConvBias {
      mapping["conv_post.bias"] = [convPost.bias.name]
    }
    if prefix.isEmpty {
      return mapping
    }
    var prefixedMapping = ModelWeightMapping()
    for (key, value) in mapping {
      prefixedMapping["\(prefix).\(key)"] = value
    }
    return prefixedMapping
  }
  return (mapper, Model([x], [out]))
}

func LTX2Vocoder(
  width: Int, initialChannels: Int,
  layers: [(channels: Int, kernelSize: Int, stride: Int, padding: Int)], resblockKernelSizes: [Int],
  resblockDilations: [[Int]], activation: LTX2VocoderActivation, finalConvBias: Bool,
  finalActivation: LTX2VocoderFinalActivation, name: String = ""
)
  -> (
    ModelWeightMapper, Model
  )
{
  LTX2Vocoder(
    width: width, initialChannels: initialChannels, layers: layers,
    resblockKernelSizes: resblockKernelSizes, resblockDilations: resblockDilations,
    activation: activation, finalConvBias: finalConvBias, finalActivation: finalActivation,
    applyFinalActivation: true, prefix: "", name: name)
}

private func LTX23HannUpSample1dFilterWeight(ratio: Int, kernelSize: Int) -> Tensor<Float> {
  let rolloff: Float = 0.99
  let lowpassFilterWidth: Float = 6
  let width = Int((lowpassFilterWidth / rolloff).rounded(.up))
  precondition(2 * width * ratio + 1 == kernelSize)
  var kernel = Tensor<Float>(.CPU, .NCHW(1, 1, 1, kernelSize))
  kernel.withUnsafeMutableBytes {
    guard let fp32 = $0.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
    for i in 0..<kernelSize {
      let t = (Float(i) / Float(ratio) - Float(width)) * rolloff
      let clamped = min(max(t, -lowpassFilterWidth), lowpassFilterWidth)
      let window = Float(cos(Double(clamped * Float.pi / lowpassFilterWidth / 2)))
      let sinc =
        abs(t) < 1e-8 ? Float(1) : Float(sin(Double(Float.pi * t)) / (Double(Float.pi * t)))
      fp32[i] = sinc * window * window * rolloff / Float(ratio)
    }
  }
  return kernel
}

func LTX2VocoderWithBWE(width: Int) -> (ModelWeightMapper, Model, () -> Void) {
  let hopLength = 80
  let nFFT = 512
  let nMelChannels = 64
  let bweResampleRatio = 3
  let bweResampleKernelSize = 43
  let bweResamplePad = 7
  let bweResamplePadLeft = 42
  let x = Input()
  let (coreMapper, coreVocoder) = LTX2Vocoder(
    width: width, initialChannels: 1536,
    layers: [
      (channels: 768, kernelSize: 11, stride: 5, padding: 3),
      (channels: 384, kernelSize: 4, stride: 2, padding: 1),
      (channels: 192, kernelSize: 4, stride: 2, padding: 1),
      (channels: 96, kernelSize: 4, stride: 2, padding: 1),
      (channels: 48, kernelSize: 4, stride: 2, padding: 1),
      (channels: 24, kernelSize: 4, stride: 2, padding: 1),
    ], resblockKernelSizes: [3, 7, 11], resblockDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    activation: .activation1d,
    finalConvBias: false, finalActivation: .clamp, applyFinalActivation: true, prefix: "vocoder",
    name: "")
  let coreOut = coreVocoder(x)
  let coreWidth = width * 5 * 2 * 2 * 2 * 2 * 2
  let remainder = coreWidth % hopLength
  let corePad = remainder == 0 ? 0 : (hopLength - remainder)
  let paddedCoreWidth = coreWidth + corePad
  let bweInputWidth = paddedCoreWidth / hopLength
  var paddedCoreOut = coreOut
  if corePad > 0 {
    paddedCoreOut = coreOut.padded(.zero, begin: [0, 0, 0, 0], end: [0, 0, 0, corePad])
  }
  let nFreqs = nFFT / 2 + 1
  let leftPad = max(0, nFFT - hopLength)
  let stft = Convolution(
    groups: 1, filters: nFreqs * 2, filterSize: [1, nFFT], noBias: true,
    hint: Hint(stride: [1, hopLength]), format: .OIHW, name: "mel_stft_forward")
  var stftInput = paddedCoreOut.reshaped([2, 1, 1, paddedCoreWidth])
  stftInput = stftInput.padded(.zero, begin: [0, 0, 0, leftPad], end: [0, 0, 0, 0])
  let stftOut = stft(stftInput)
  let stftParts = stftOut.chunked(2, axis: 1)
  let magnitude = ((stftParts[0] .* stftParts[0]) + (stftParts[1] .* stftParts[1])).squareRoot()
  let melProjection = Convolution(
    groups: 1, filters: nMelChannels, filterSize: [1, 1], noBias: true,
    hint: Hint(stride: [1, 1]), format: .OIHW, name: "mel_projection")
  var mel = melProjection(magnitude).clamped(1e-5...).log()
  mel = mel.reshaped(
    [1, 2, nMelChannels, bweInputWidth], offset: [0, 0, 0, 0],
    strides: [
      2 * nMelChannels * bweInputWidth, nMelChannels * bweInputWidth, bweInputWidth, 1,
    ]
  ).contiguous()
  let bweInput = mel.reshaped(
    [1, 2 * nMelChannels, 1, bweInputWidth], offset: [0, 0, 0, 0],
    strides: [2 * nMelChannels * bweInputWidth, bweInputWidth, bweInputWidth, 1]
  ).contiguous()
  let (bweMapper, bweGenerator) = LTX2Vocoder(
    width: bweInputWidth, initialChannels: 512,
    layers: [
      (channels: 256, kernelSize: 12, stride: 6, padding: 3),
      (channels: 128, kernelSize: 11, stride: 5, padding: 3),
      (channels: 64, kernelSize: 4, stride: 2, padding: 1),
      (channels: 32, kernelSize: 4, stride: 2, padding: 1),
      (channels: 16, kernelSize: 4, stride: 2, padding: 1),
    ], resblockKernelSizes: [3, 7, 11], resblockDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    activation: .activation1d,
    finalConvBias: false, finalActivation: .clamp, applyFinalActivation: false,
    prefix: "bwe_generator", name: "bwe")
  let residual = bweGenerator(bweInput)
  let resampler = ConvolutionTranspose(
    groups: 1, filters: 1, filterSize: [1, bweResampleKernelSize], noBias: true,
    hint: Hint(stride: [1, bweResampleRatio]), format: .OIHW, name: "bwe_resampler")
  let weightLoader = {
    resampler.weight.copy(
      from: LTX23HannUpSample1dFilterWeight(
        ratio: bweResampleRatio, kernelSize: bweResampleKernelSize)
    )
  }
  let resampleRawWidth =
    (paddedCoreWidth + 2 * bweResamplePad - 1) * bweResampleRatio + bweResampleKernelSize
  var skip = paddedCoreOut.reshaped([2, 1, 1, paddedCoreWidth]).padded(
    .replicate, begin: [0, 0, 0, bweResamplePad], end: [0, 0, 0, bweResamplePad])
  skip = Float(bweResampleRatio) * resampler(skip)
  skip = skip.reshaped(
    [2, 1, 1, paddedCoreWidth * bweResampleRatio], offset: [0, 0, 0, bweResamplePadLeft],
    strides: [resampleRawWidth, resampleRawWidth, resampleRawWidth, 1]
  ).contiguous()
  skip = skip.reshaped([1, 2, 1, paddedCoreWidth * bweResampleRatio])
  let outputLength = coreWidth * bweResampleRatio
  var out = (residual + skip).clamped(-1...1)
  if outputLength != paddedCoreWidth * bweResampleRatio {
    out = out.reshaped(
      [1, 2, 1, outputLength], offset: [0, 0, 0, 0],
      strides: [
        2 * paddedCoreWidth * bweResampleRatio, paddedCoreWidth * bweResampleRatio,
        paddedCoreWidth * bweResampleRatio, 1,
      ]
    ).contiguous()
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = coreMapper(format)
    mapping.merge(bweMapper(format)) { v, _ in v }
    mapping["mel_stft.stft_fn.forward_basis"] = [stft.weight.name]
    mapping["mel_stft.mel_basis"] = [melProjection.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]), weightLoader)
}
