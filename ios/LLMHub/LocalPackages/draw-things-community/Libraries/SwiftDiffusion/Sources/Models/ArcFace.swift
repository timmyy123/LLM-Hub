import NNC

private func PReLU<FloatType: TensorNumeric & BinaryFloatingPoint>(
  count: Int, name: String, of: FloatType.Type = FloatType.self
) -> Model {
  let x = Input()
  let weight = Parameter<FloatType>(.GPU(0), .HWC(1, 1, count), name: "\(name).slope")
  let out = x.ReLU() - (-x).ReLU() .* weight
  return Model([x], [out])
}

private func BatchNorm<FloatType: TensorNumeric & BinaryFloatingPoint>(
  count: Int, name: String, of: FloatType.Type = FloatType.self
) -> Model {
  let x = Input()
  let weight = Parameter<FloatType>(.GPU(0), .HWC(1, 1, count), name: "\(name).weight")
  let bias = Parameter<FloatType>(.GPU(0), .HWC(1, 1, count), name: "\(name).bias")
  let out = x .* weight + bias
  return Model([x], [out])
}

private func ResnetBlock<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: (Int, Int, String), inChannels: Int, outChannels: Int, downsample: Bool,
  of: FloatType.Type = FloatType.self
) -> Model {
  let x = Input()
  let bn1 = BatchNorm(count: inChannels, name: prefix.2, of: FloatType.self)
  var out = bn1(x)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv1")
  out = conv1(out)
  let prelu = PReLU(count: outChannels, name: "\(prefix.1)", of: FloatType.self)
  out = prelu(out)
  let conv2: Model
  if downsample {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
      name: "conv2")
    let convSkip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW,
      name: "skip")
    out = conv2(out) + convSkip(x)
  } else {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
      name: "conv2")
    out = conv2(out) + x
  }
  return Model([x], [out])
}

private func ResnetLayer<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: (Int, Int, String), inChannels: Int, outChannels: Int, layers: Int,
  of: FloatType.Type = FloatType.self
) -> Model {
  let x = Input()
  // First block is to downsample.
  let firstBlock = ResnetBlock(
    prefix: (prefix.0, prefix.1, "\(prefix.2).0.bn1"), inChannels: inChannels,
    outChannels: outChannels, downsample: true, of: FloatType.self)
  var out = firstBlock(x)
  if layers > 1 {
    for i in 0..<(layers - 1) {
      let block = ResnetBlock(
        prefix: (prefix.0 + i * 6 + 9, prefix.1 + i + 1, "\(prefix.2).\(i + 1).bn1"),
        inChannels: outChannels, outChannels: outChannels, downsample: false, of: FloatType.self)
      out = block(out)
    }
  }
  return Model([x], [out])
}

public func ArcFace<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, of: FloatType.Type = FloatType.self
) -> Model {
  let x = Input()
  let conv = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv(x)
  let prelu = PReLU(count: 64, name: "1643", of: FloatType.self)
  out = prelu(out)
  let layer1 = ResnetLayer(
    prefix: (1338, 1644, "layer1"), inChannels: 64, outChannels: 64, layers: 3, of: FloatType.self)
  out = layer1(out)
  let layer2 = ResnetLayer(
    prefix: (1359, 1647, "layer2"), inChannels: 64, outChannels: 128, layers: 13, of: FloatType.self
  )
  out = layer2(out)
  let layer3 = ResnetLayer(
    prefix: (1440, 1660, "layer3"), inChannels: 128, outChannels: 256, layers: 30,
    of: FloatType.self)
  out = layer3(out)
  let layer4 = ResnetLayer(
    prefix: (1623, 1690, "layer4"), inChannels: 256, outChannels: 512, layers: 3, of: FloatType.self
  )
  out = layer4(out)
  let fc = Dense(count: 512)
  out = fc(out.permuted(0, 3, 1, 2).reshaped([batchSize, 512 * 7 * 7]))
  return Model([x], [out])
}
