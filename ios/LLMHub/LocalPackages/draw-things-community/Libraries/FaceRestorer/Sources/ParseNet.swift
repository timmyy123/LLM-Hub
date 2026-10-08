import NNC

private func ResidualBlock(
  prefix: String, outChannels: Int, scaleUp: Bool, scaleDown: Bool, shortcut: Bool
) -> Model {
  let x = Input()
  let z: Model.IO
  if scaleUp {
    z = Upsample(.nearest, widthScale: 2, heightScale: 2)(x)
  } else {
    z = x
  }
  let y: Model.IO
  if shortcut {
    let conv: Model
    if scaleDown {
      conv = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    } else {
      conv = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    }
    y = conv(z)
  } else {
    y = z
  }
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv1(z).leakyReLU(negativeSlope: 0.2)
  let conv2: Model
  if scaleDown {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  } else {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  }
  out = y + conv2(out)
  return Model([x], [out])
}

public func ParseNet() -> Model {
  let x = Input()
  let conv = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv(x)
  let outChannels = [64, 128, 256, 256, 256]
  for i in 1..<5 {
    let encoder = ResidualBlock(
      prefix: "encoder.\(i)", outChannels: outChannels[i], scaleUp: false, scaleDown: true,
      shortcut: true)
    out = encoder(out)
  }
  let feat = out
  for i in 0..<10 {
    let body = ResidualBlock(
      prefix: "body.\(i)", outChannels: 256, scaleUp: false, scaleDown: false, shortcut: false)
    out = body(out)
  }
  out = feat + out
  for i in 0..<4 {
    let decoder = ResidualBlock(
      prefix: "decoder.\(i)", outChannels: outChannels[outChannels.count - 2 - i], scaleUp: true,
      scaleDown: false, shortcut: true)
    out = decoder(out)
  }
  let outMaskConv = Convolution(
    groups: 1, filters: 19, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outMaskConv(out)
  return Model([x], [out])
}
