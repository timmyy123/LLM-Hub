import NNC

func vggConvLayer(outputChannels: Int, convLayers: Int) -> Model {
  let x = Input()
  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)

  var layerConv2dArray = [Convolution]()

  for _ in 0..<convLayers {
    let layerConv2d = Convolution(
      groups: 1, filters: outputChannels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
    out = layerConv2d(out)
    out = ReLU()(out)
    layerConv2dArray.append(layerConv2d)
  }

  return Model([x], [out])
}

public func HEDModel(inputWidth: Int, inputHeight: Int) -> Model {
  let x = Input()

  let vggOneInLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var vggOneEmbeds = vggOneInLayerConv2d(x)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let vggOneOutLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  vggOneEmbeds = vggOneOutLayerConv2d(vggOneEmbeds)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let vggTwo = vggConvLayer(outputChannels: 128, convLayers: 2)
  let vggTwoEmbeds = vggTwo(vggOneEmbeds)

  let vggThr = vggConvLayer(outputChannels: 256, convLayers: 3)
  let vggThrEmbeds = vggThr(vggTwoEmbeds)

  let vggFour = vggConvLayer(outputChannels: 512, convLayers: 3)
  let vggFouEmbeds = vggFour(vggThrEmbeds)

  let vggFiv = vggConvLayer(outputChannels: 512, convLayers: 3)
  let vggFiveEmbeds = vggFiv(vggFouEmbeds)

  let netScoreOne = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)

  let netScoreTwo = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)

  let netScoreThr = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)

  let netScoreFou = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)

  let netScoreFiv = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)

  let scaledNetScoreOneEmbeds = Upsample(.bilinear, widthScale: 1, heightScale: 1)(
    netScoreOne(vggOneEmbeds))
  let scaledNetScoreTwoEmbeds = Upsample(.bilinear, widthScale: 2, heightScale: 2)(
    netScoreTwo(vggTwoEmbeds))
  let scaledNetScoreThrEmbeds = Upsample(.bilinear, widthScale: 4, heightScale: 4)(
    netScoreThr(vggThrEmbeds))
  let scaledNetScoreFouEmbeds = Upsample(.bilinear, widthScale: 8, heightScale: 8)(
    netScoreFou(vggFouEmbeds))
  let scaledNetScoreFivEmbeds = Upsample(.bilinear, widthScale: 16, heightScale: 16)(
    netScoreFiv(vggFiveEmbeds))
  let mergedVggEmbedings = Functional.concat(
    axis: 3, scaledNetScoreOneEmbeds, scaledNetScoreTwoEmbeds, scaledNetScoreThrEmbeds,
    scaledNetScoreFouEmbeds, scaledNetScoreFivEmbeds)

  let netcombineLayerConv2d = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
  var out = netcombineLayerConv2d(mergedVggEmbedings)
  out = Sigmoid()(out)

  return Model([x], [out])
}
