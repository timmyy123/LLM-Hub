import NNC

private func ResnetBlock(prefix: Int, channels: Int, downsample: Bool) -> Model {
  let x = Input()
  let conv1: Model
  if downsample {
    conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  } else {
    conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  }
  var out = conv1(x).ReLU()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  if downsample {
    let convSkip = Convolution(
      groups: 1, filters: channels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
    out =
      conv2(out)
      + convSkip(
        AveragePool(
          filterSize: [2, 2],
          hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])))(x))
  } else {
    out = conv2(out) + x
  }
  return Model([x], [out])
}

private func BoxHead(prefix: (Int, String), multiplier: Float) -> Model {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 80, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv1(x).ReLU()
  let conv2 = Convolution(
    groups: 1, filters: 80, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out).ReLU()
  let conv3 = Convolution(
    groups: 1, filters: 80, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv3(out).ReLU()
  let cls = Convolution(
    groups: 1, filters: 2, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var outs = [Model.IO]()
  outs.append(cls(out).sigmoid())
  let reg = Convolution(
    groups: 1, filters: 8, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  outs.append(reg(out) * multiplier)
  let kps = Convolution(
    groups: 1, filters: 20, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  outs.append(kps(out))
  return Model([x], outs)
}

func RetinaFace(batchSize: Int) -> Model {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 28, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv1(x).ReLU()
  let conv2 = Convolution(
    groups: 1, filters: 28, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out).ReLU()
  let conv3 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv3(out).ReLU()
  out = MaxPool(
    filterSize: [2, 2],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])))(out)
  let resnetBlock_1_1 = ResnetBlock(prefix: 559, channels: 56, downsample: false)
  out = resnetBlock_1_1(out).ReLU()
  let resnetBlock_1_2 = ResnetBlock(prefix: 567, channels: 56, downsample: false)
  out = resnetBlock_1_2(out).ReLU()
  let resnetBlock_1_3 = ResnetBlock(prefix: 575, channels: 56, downsample: false)
  out = resnetBlock_1_3(out).ReLU()
  let resnetBlock_2_1 = ResnetBlock(prefix: 583, channels: 88, downsample: true)
  out = resnetBlock_2_1(out).ReLU()
  let resnetBlock_2_2 = ResnetBlock(prefix: 595, channels: 88, downsample: false)
  out = resnetBlock_2_2(out).ReLU()
  let resnetBlock_2_3 = ResnetBlock(prefix: 603, channels: 88, downsample: false)
  out = resnetBlock_2_3(out).ReLU()
  let resnetBlock_2_4 = ResnetBlock(prefix: 611, channels: 88, downsample: false)
  out = resnetBlock_2_4(out).ReLU()
  var layer2Out = out
  let resnetBlock_3_1 = ResnetBlock(prefix: 619, channels: 88, downsample: true)
  out = resnetBlock_3_1(out).ReLU()
  let resnetBlock_3_2 = ResnetBlock(prefix: 631, channels: 88, downsample: false)
  out = resnetBlock_3_2(out).ReLU()
  var layer3Out = out
  let resnetBlock_4_1 = ResnetBlock(prefix: 639, channels: 224, downsample: true)
  out = resnetBlock_4_1(out).ReLU()
  let resnetBlock_4_2 = ResnetBlock(prefix: 651, channels: 224, downsample: false)
  out = resnetBlock_4_2(out).ReLU()
  let resnetBlock_4_3 = ResnetBlock(prefix: 659, channels: 224, downsample: false)
  out = resnetBlock_4_3(out).ReLU()
  let conv4 = Convolution(
    groups: 1, filters: 56, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
  layer2Out = conv4(layer2Out)
  let conv5 = Convolution(
    groups: 1, filters: 56, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
  layer3Out = conv5(layer3Out)
  let conv6 = Convolution(
    groups: 1, filters: 56, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
  out = conv6(out)
  layer3Out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out) + layer3Out
  layer2Out = Upsample(.nearest, widthScale: 2, heightScale: 2)(layer3Out) + layer2Out

  let conv7 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  layer2Out = conv7(layer2Out)

  let conv8 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let conv9 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  layer3Out = conv8(layer3Out) + conv9(layer2Out)

  let conv10 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let conv11 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv10(out) + conv11(layer3Out)

  let conv12 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  layer3Out = conv12(layer3Out)

  let conv13 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv13(out)

  let boxHead8 = BoxHead(prefix: (667, "(8, 8)"), multiplier: 0.8463594317436218)
  layer2Out = boxHead8(layer2Out)

  let boxHead16 = BoxHead(prefix: (679, "(16, 16)"), multiplier: 0.8996264338493347)
  layer3Out = boxHead16(layer3Out)

  let boxHead32 = BoxHead(prefix: (691, "(32, 32)"), multiplier: 1.0812087059020996)
  out = boxHead32(out)

  return Model([x], [out, layer3Out, layer2Out])
}
