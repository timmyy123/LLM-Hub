import Foundation
import NNC

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

public struct SCHPMaskGenerator {
  public enum Category {
    case dresses
    case neck
    case upperBody
    case lowerBody
  }

  public let atrModelPath: String
  public let lipModelPath: String
  public init(atrModelPath: String, lipModelPath: String) {
    self.atrModelPath = atrModelPath
    self.lipModelPath = lipModelPath
  }
}

extension SCHPMaskGenerator {
  public func mask<FloatType: TensorNumeric & BinaryFloatingPoint>(
    imageTensor: DynamicGraph.Tensor<FloatType>,
    categories: [Category],
    leftArmsImageTensor: Tensor<UInt8>?,
    rightArmsImageTensor: Tensor<UInt8>?,
    extraArea: Bool
  ) -> DynamicGraph.Tensor<FloatType> {
    guard imageTensor.format == .NHWC else { return imageTensor }
    let graph = imageTensor.graph
    let originalWidth = imageTensor.shape[2]
    let originalHeight = imageTensor.shape[1]

    let (inputTensor, margin) = SCHPTensorPreprocess(
      inputTensor: imageTensor, targetHeight: 512, targetWidth: 512, originalHeight: originalHeight,
      originalWidth: originalWidth, graph: graph)

    let atrResult = {
      let schpAtr = SCHP(numClasses: 18, originSize: 512)
      schpAtr.maxConcurrency = .limit(4)
      schpAtr.compile(inputs: inputTensor)
      graph.openStore(atrModelPath, flags: .readOnly) {
        $0.read("schp", model: schpAtr)
      }
      return schpAtr(inputs: inputTensor)[1].as(of: FloatType.self).toCPU()
    }()
    let postAtrResult = SCHPTensorPostprocess(
      inputTensor: atrResult, targetHeight: 512, targetWidth: 512, originalHeight: originalHeight,
      originalWidth: originalWidth,
      graph: graph, margin: margin, numberOfclass: 18)

    let (lipInputTensor, lipMargin) = SCHPTensorPreprocess(
      inputTensor: imageTensor, targetHeight: 473, targetWidth: 473,
      originalHeight: originalHeight, originalWidth: originalWidth, graph: graph)
    let lipResult = {
      let schpLip = SCHP(numClasses: 20, originSize: 473)
      schpLip.maxConcurrency = .limit(4)
      schpLip.compile(inputs: lipInputTensor)
      graph.openStore(lipModelPath, flags: .readOnly) {
        $0.read("schp", model: schpLip)
      }
      return schpLip(inputs: lipInputTensor)[1].as(of: FloatType.self).toCPU()
    }()
    let postLipResult = SCHPTensorPostprocess(
      inputTensor: lipResult, targetHeight: 473, targetWidth: 473, originalHeight: originalHeight,
      originalWidth: originalWidth,
      graph: graph, margin: lipMargin, numberOfclass: 20)

    let merged = mergeSCHPTensors(atrResult: postAtrResult, lipResult: postLipResult)

    let labelMap: [String: Int] = [
      "background": 0,
      "hat": 1,
      "hair": 2,
      "sunglasses": 3,
      "upper_clothes": 4,
      "skirt": 5,
      "pants": 6,
      "dress": 7,
      "belt": 8,
      "left_shoe": 9,
      "right_shoe": 10,
      "head": 11,
      "left_leg": 12,
      "right_leg": 13,
      "left_arm": 14,
      "right_arm": 15,
      "bag": 16,
      "scarf": 17,
    ]

    let parseHead = parserMaskLabelsToCheck(
      inputTensor: merged,
      labelsToCheck: [
        labelMap["hat"]!,
        labelMap["sunglasses"]!,
        labelMap["head"]!,
      ])

    var parserMaskFixed = parserMaskLabelsToCheck(
      inputTensor: merged,
      labelsToCheck: [
        labelMap["left_shoe"]!,
        labelMap["right_shoe"]!,
        labelMap["hat"]!,
        labelMap["sunglasses"]!,
        labelMap["bag"]!,
      ])

    let armsLeft = parserMaskLabelsToCheck(
      inputTensor: merged, labelsToCheck: [labelMap["left_arm"]!])
    let armsRight = parserMaskLabelsToCheck(
      inputTensor: merged, labelsToCheck: [labelMap["right_arm"]!])

    var parseMask = merged
    var parseMaskCategoryChecked = false
    if categories.contains(.dresses) {
      let dressesParseMask = parserMaskLabelsToCheck(
        inputTensor: merged,
        labelsToCheck: [
          labelMap["upper_clothes"]!,
          labelMap["skirt"]!,
          labelMap["pants"]!,
          labelMap["dress"]!,
        ])
      parseMask = dressesParseMask
      parseMaskCategoryChecked = true

    }

    if categories.contains(.upperBody) {
      let upperBodyParseMask = parserMaskLabelsToCheck(
        inputTensor: merged,
        labelsToCheck: [
          labelMap["upper_clothes"]!,
          labelMap["dress"]!,
          labelMap["left_arm"]!,
          labelMap["right_arm"]!,
        ])
      if parseMaskCategoryChecked {
        parseMask = schpMaskOr(parseMask, upperBodyParseMask)
      } else {
        parseMask = upperBodyParseMask
      }
      parseMaskCategoryChecked = true
    }

    if categories.contains(.lowerBody) {
      let lowerBodyParseMask = parserMaskLabelsToCheck(
        inputTensor: merged,
        labelsToCheck: [
          labelMap["skirt"]!,
          labelMap["pants"]!,
          labelMap["left_leg"]!,
          labelMap["right_leg"]!,
        ])
      if parseMaskCategoryChecked {
        parseMask = schpMaskOr(parseMask, lowerBodyParseMask)
      } else {
        parseMask = lowerBodyParseMask
      }
      parseMaskCategoryChecked = true
    }

    var imArmsLeft = armsLeft
    var imArmsRight = armsRight

    if let leftArmsImageTensor = leftArmsImageTensor {
      imArmsLeft = graph.variable(leftArmsImageTensor).reshaped(
        .NHWC(1, originalHeight, originalWidth, 1))
      let handsLeft = schpMaskAnd(schpMaskNot(imArmsLeft), armsLeft)
      parserMaskFixed = schpMaskOr(parserMaskFixed, handsLeft)

    }

    if let rightArmsImageTensor = rightArmsImageTensor {
      imArmsRight = graph.variable(rightArmsImageTensor).reshaped(
        .NHWC(1, originalHeight, originalWidth, 1))
      let handsRight = schpMaskAnd(schpMaskNot(imArmsRight), armsRight)
      parserMaskFixed = schpMaskOr(parserMaskFixed, handsRight)
    }

    parserMaskFixed = schpMaskOr(parserMaskFixed, parseHead)
    parseMask = schpMaskAnd(parseMask, schpMaskNot(parserMaskFixed))
    if extraArea {
      parseMask = schpMaskDilate(parseMask, iterations: 5)
    }

    if categories.contains(.neck) {
      var neckMask = parserMaskLabelsToCheck(inputTensor: merged, labelsToCheck: [18])
      if extraArea {
        neckMask = schpMaskDilate(neckMask, iterations: 1, size: 3)
      }
      neckMask = schpMaskAnd(neckMask, schpMaskNot(parseHead))
      parseMask = schpMaskOr(parseMask, neckMask)
    }

    if extraArea {
      var armMask = schpMaskOr(imArmsLeft, imArmsRight)
      armMask = schpMaskDilate(armMask, iterations: 4)
      parseMask = schpMaskOr(parseMask, armMask)
    }

    let parseMaskTotal = schpMaskAnd(parseMask, schpMaskNot(parserMaskFixed))
    let imageWidth = parseMaskTotal.shape[2]
    let imageHeight = parseMaskTotal.shape[1]
    var parseMaskTotalF16 = Tensor<FloatType>(
      .CPU, .NHWC(1, imageHeight, imageWidth, 1))

    parseMaskTotalF16.withUnsafeMutableBytes {
      guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      parseMaskTotal.rawValue.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for y in 0..<imageHeight {
          for x in 0..<imageWidth {
            f16[y * imageWidth + x] = u8[y * imageWidth + x] != 0 ? 0 : 255
          }
        }
      }
    }

    return graph.variable(parseMaskTotalF16)
  }

}

private func bottleneck(
  outCh: Int, stride: Int = 1, dilation: Int = 1, downsample: Bool = false, multi_grid: Int = 1,
  layerPrefix: String, block: String
) -> Model {
  let x = Input()
  var residual: Model.IO = x

  let conv1 = Convolution(
    groups: 1, filters: outCh, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  var out = ReLU()(conv1(x))
  let border = Hint.Border(begin: [dilation, dilation], end: [dilation, dilation])
  let conv2 = Convolution(
    groups: 1, filters: outCh, filterSize: [3, 3], dilation: [dilation, dilation],
    hint: Hint(stride: [stride, stride], border: border), format: .OIHW)
  out = ReLU()(conv2(out))

  let conv3 = Convolution(
    groups: 1, filters: outCh * 4, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  out = conv3(out)

  if downsample {
    let downsampleConv = Convolution(
      groups: 1, filters: outCh * 4, filterSize: [1, 1], hint: Hint(stride: [stride, stride]),
      format: .OIHW)
    residual = downsampleConv(x)
  }

  out = out + residual
  out = ReLU()(out)

  return Model([x], [out])
}

private func makeLayer(outCh: Int, stride: Int = 1, dilation: Int = 1, prefix: String, layers: Int)
  -> Model
{
  let x = Input()

  let layer = bottleneck(
    outCh: outCh, stride: stride, dilation: dilation, downsample: true, layerPrefix: prefix,
    block: "0")
  var out = layer(x)
  for i in 1..<layers {
    let layer = bottleneck(
      outCh: outCh, stride: 1, dilation: dilation, downsample: false, layerPrefix: prefix,
      block: "\(i)")
    out = layer(out)
  }

  return Model([x], [out])
}

private func contextEncoding(outCh: Int, targetSize: Int, sizes: [Int]) -> Model {
  let x = Input()
  var priors = [Model.IO]()
  for i in 0..<sizes.count {
    let stage = makeStage(outCh: outCh, targetSize: targetSize, size: sizes[i], block: "\(i)")
    var prior = stage(x)
    let scale = Float(targetSize) / Float(sizes[i])
    prior = Upsample(.bilinear, widthScale: scale, heightScale: scale, alignCorners: true)(prior)
    priors.append(prior)
  }
  priors.append(x)
  var out = Functional.concat(axis: 1, priors[0], priors[1], priors[2], priors[3], priors[4])  // 30, 15, 10, 5, 1

  let conv = Convolution(
    groups: 1, filters: outCh, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv(out).leakyReLU(negativeSlope: 0.01)

  return Model([x], [out])
}

private func makeStage(outCh: Int, targetSize: Int, size: Int, block: String) -> Model {
  let x = Input()
  let kernelSize = Int(ceil(Float(targetSize) / Float(size)))
  let stride = Int(floor(Float(targetSize) / Float(size)))

  var out = AveragePool(
    filterSize: [kernelSize, kernelSize], hint: Hint(stride: [stride, stride]))(x)

  let conv1 = Convolution(
    groups: 1, filters: outCh, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  out = conv1(out).leakyReLU(negativeSlope: 0.01)

  return Model([x], [out])
}

func decoder(numClasses: Int, scale3: Float) -> Model {
  let xt = Input()
  let xl = Input()

  let conv1 = Convolution(
    groups: 1, filters: 256, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  var xtOut = conv1(xt).leakyReLU(negativeSlope: 0.01)
  xtOut = Upsample(.bilinear, widthScale: scale3, heightScale: scale3, alignCorners: true)(xtOut)

  let conv2 = Convolution(
    groups: 1, filters: 48, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  let xlOut = conv2(xl).leakyReLU(negativeSlope: 0.01)

  var x = Functional.concat(axis: 1, xtOut, xlOut)

  let conv30 = Convolution(
    groups: 1, filters: 256, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  x = conv30(x).leakyReLU(negativeSlope: 0.01)

  let conv31 = Convolution(
    groups: 1, filters: 256, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  x = conv31(x).leakyReLU(negativeSlope: 0.01)

  let conv4 = Convolution(
    groups: 1, filters: numClasses, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  let seg = conv4(x)

  return Model([xt, xl], [seg, x])
}

func edge(scale2: Float, scale3: Float) -> Model {
  let x1 = Input()
  let x2 = Input()
  let x3 = Input()

  let conv1 = Convolution(
    groups: 1, filters: 256, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)

  let conv4 = Convolution(
    groups: 1, filters: 2, filterSize: [3, 3], dilation: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)

  let edge1Fea = conv1(x1).leakyReLU(negativeSlope: 0.01)
  let edge1 = conv4(edge1Fea)

  let conv2 = Convolution(
    groups: 1, filters: 256, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  var edge2Fea = conv2(x2).leakyReLU(negativeSlope: 0.01)
  var edge2 = conv4(edge2Fea)

  let conv3 = Convolution(
    groups: 1, filters: 256, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  var edge3Fea = conv3(x3).leakyReLU(negativeSlope: 0.01)
  var edge3 = conv4(edge3Fea)

  edge2Fea = Upsample(.bilinear, widthScale: scale2, heightScale: scale2, alignCorners: true)(
    edge2Fea)
  edge2 = Upsample(.bilinear, widthScale: scale2, heightScale: scale2, alignCorners: true)(edge2)

  edge3Fea = Upsample(.bilinear, widthScale: scale3, heightScale: scale3, alignCorners: true)(
    edge3Fea)
  edge3 = Upsample(.bilinear, widthScale: scale3, heightScale: scale3, alignCorners: true)(edge3)

  var edge = Functional.concat(axis: 1, edge1, edge2, edge3)  // 30, 15, 10, 5, 1
  let edgeFea = Functional.concat(axis: 1, edge1Fea, edge2Fea, edge3Fea)  // 30, 15, 10, 5, 1

  let conv5 = Convolution(
    groups: 1, filters: 2, filterSize: [1, 1], dilation: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  edge = conv5(edge)

  return Model([x1, x2, x3], [edge, edgeFea])
}

public func SCHP(numClasses: Int, originSize: Float) -> Model {
  let x = Input()
  let scale2width = ceil(originSize / 8)
  let scale3width = ceil(originSize / 16)
  let sacleOrigin = ceil(originSize / 4)

  let conv1 = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = ReLU()(conv1(x))

  let conv2 = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = ReLU()(conv2(out))

  let conv3 = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = ReLU()(conv3(out))
  out = MaxPool(
    filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))(out)

  let layer1 = makeLayer(outCh: 64, prefix: "layer1", layers: 3)
  let x2 = layer1(out)
  let layer2 = makeLayer(outCh: 128, stride: 2, prefix: "layer2", layers: 4)
  let x3 = layer2(x2)
  let layer3 = makeLayer(outCh: 256, stride: 2, prefix: "layer3", layers: 23)
  let x4 = layer3(x3)
  let layer4 = makeLayer(outCh: 512, stride: 1, dilation: 2, prefix: "layer4", layers: 3)
  let x5 = layer4(x4)
  let contextEncodingLayer = contextEncoding(
    outCh: 512, targetSize: Int(scale3width), sizes: [1, 2, 3, 6])
  out = contextEncodingLayer(x5)

  let scale2 = Float(sacleOrigin / scale2width)
  let scale3 = Float(sacleOrigin / scale3width)
  let edgeLayer = edge(scale2: scale2, scale3: scale3)
  let edgeOutput = edgeLayer(x2, x3, x4)
  let edgeResult = edgeOutput[0]
  let edgeFea = edgeOutput[1]

  let decoderLayer = decoder(numClasses: numClasses, scale3: scale3)
  let decoderOutput = decoderLayer(out, x2)
  let parsingResult = decoderOutput[0]
  let parsingFea = decoderOutput[1]

  out = Functional.concat(axis: 1, parsingFea, edgeFea)

  let fusionConv0 = Convolution(
    groups: 1, filters: 256, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  var fusionResult = fusionConv0(out).leakyReLU(negativeSlope: 0.01)

  let fusionConv1 = Convolution(
    groups: 1, filters: numClasses, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  fusionResult = fusionConv1(fusionResult)

  return Model([x], [parsingResult, fusionResult, edgeResult])
}

// Helpers for SCHP pre/post processing

private func SCHPTensorPreprocess<FloatType: TensorNumeric & BinaryFloatingPoint>(
  inputTensor: DynamicGraph.Tensor<FloatType>, targetHeight: Int, targetWidth: Int,
  originalHeight: Int,
  originalWidth: Int,
  graph: DynamicGraph
)
  -> (DynamicGraph.Tensor<FloatType>, Int)
{
  let w = originalWidth
  let h = originalHeight
  var scaleRatio = Float(targetWidth) / Float(w)
  if w < h {
    scaleRatio = Float(targetHeight) / Float(h)
  }
  let upsample = Upsample(.bilinear, widthScale: scaleRatio, heightScale: scaleRatio)(
    inputTensor.toGPU(0))
  var output = graph.variable(
    Tensor<FloatType>(.GPU(0), .NHWC(1, Int(targetHeight), Int(targetWidth), 3)))
  output.full(0)

  var margin = 0
  if w > h {
    margin = Int((Float(targetHeight) - Float(h) * scaleRatio) / 2)
    output[0..<1, margin..<(targetHeight - margin), 0..<targetWidth, 0..<3] = upsample
  } else {
    margin = Int((Float(targetWidth) - Float(w) * scaleRatio) / 2)
    output[0..<1, 0..<targetHeight, margin..<(targetWidth - margin), 0..<3] = upsample
  }

  let mean = graph.variable(
    Tensor<FloatType>(
      [FloatType(0.406), FloatType(0.456), FloatType(0.485)], .GPU(0), .NHWC(1, 1, 1, 3)))

  let invStd = graph.variable(
    Tensor<FloatType>(
      [FloatType(1 / 0.225), FloatType(1 / 0.224), FloatType(1 / 0.229)], .GPU(0), .NHWC(1, 1, 1, 3)
    ))
  output = (output / 255 - mean) .* invStd
  output = output.permuted(0, 3, 1, 2).contiguous().reshaped(.NCHW(1, 3, targetHeight, targetWidth))
  return (output, margin)
}

private func SCHPTensorPostprocess<FloatType: TensorNumeric & BinaryFloatingPoint>(
  inputTensor: DynamicGraph.Tensor<FloatType>, targetHeight: Int, targetWidth: Int,
  originalHeight: Int,
  originalWidth: Int,
  graph: DynamicGraph, margin: Int, numberOfclass: Int
)
  -> DynamicGraph.Tensor<Int32>
{
  let h = inputTensor.shape[2]
  let w = inputTensor.shape[3]
  let inputTensor = inputTensor.permuted(0, 2, 3, 1).contiguous().reshaped(
    .NHWC(1, h, w, numberOfclass))

  let output = Upsample(
    .bilinear, widthScale: Float(targetWidth) / Float(w),
    heightScale: Float(targetHeight) / Float(h))(inputTensor.toGPU(0))

  var postOutput: DynamicGraph.Tensor<FloatType>
  if originalWidth > originalHeight {
    postOutput = output[0..<1, margin..<(targetHeight - margin), 0..<targetWidth, 0..<numberOfclass]
  } else {
    postOutput = output[0..<1, 0..<targetHeight, margin..<(targetWidth - margin), 0..<numberOfclass]
  }
  postOutput = Upsample(
    .bilinear, widthScale: Float(originalWidth) / Float(postOutput.shape[2]),
    heightScale: Float(originalHeight) / Float(postOutput.shape[1]))(postOutput.contiguous())

  return Argmax(axis: 3)(inputs: postOutput)[0].as(of: Int32.self).toCPU()
}

private func mergeSCHPTensors(
  atrResult: DynamicGraph.Tensor<Int32>, lipResult: DynamicGraph.Tensor<Int32>
)
  -> DynamicGraph.Tensor<UInt8>
{
  precondition(lipResult.shape[1] == atrResult.shape[1])
  precondition(lipResult.shape[2] == atrResult.shape[2])
  let imageWidth = atrResult.shape[2]
  let imageHeight = atrResult.shape[1]
  let graph = lipResult.graph
  var newTensor = Tensor<UInt8>(
    .CPU, .NHWC(1, imageHeight, imageWidth, 1))

  newTensor.withUnsafeMutableBytes {
    guard let i8to = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
    atrResult.rawValue.withUnsafeBytes {
      guard let i8a = $0.baseAddress?.assumingMemoryBound(to: Int32.self) else { return }
      lipResult.rawValue.withUnsafeBytes {
        guard let i8b = $0.baseAddress?.assumingMemoryBound(to: Int32.self) else {
          return
        }
        for y in 0..<imageHeight {
          for x in 0..<imageWidth {
            if i8a[y * imageWidth + x] == 11 && i8b[y * imageWidth + x] != 13 {
              i8to[y * imageWidth + x] = 18
            } else {
              i8to[y * imageWidth + x] = UInt8(i8a[y * imageWidth + x])
            }
          }
        }
      }
    }
  }
  return graph.variable(newTensor)
}

private func parserMaskLabelsToCheck(inputTensor: DynamicGraph.Tensor<UInt8>, labelsToCheck: [Int])
  -> DynamicGraph.Tensor<UInt8>
{
  let imageWidth = inputTensor.shape[2]
  let imageHeight = inputTensor.shape[1]
  let graph = inputTensor.graph
  var newTensor = Tensor<UInt8>(
    .CPU, .NHWC(1, imageHeight, imageWidth, 1))
  let labels: [Int] = labelsToCheck.compactMap { $0 }

  newTensor.withUnsafeMutableBytes {
    guard let u8to = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
    inputTensor.rawValue.withUnsafeBytes {
      guard let u8from = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          if labels.contains(Int(u8from[y * imageWidth + x])) {
            u8to[y * imageWidth + x] = 255
          } else {
            u8to[y * imageWidth + x] = 0
          }
        }
      }
    }
  }
  return graph.variable(newTensor)
}

private func schpMaskAnd(_ a: DynamicGraph.Tensor<UInt8>, _ b: DynamicGraph.Tensor<UInt8>)
  -> DynamicGraph.Tensor<UInt8>
{
  guard a.shape == b.shape else { return a }
  let imageWidth = a.shape[2]
  let imageHeight = a.shape[1]
  let graph = a.graph
  var newTensor = Tensor<UInt8>(
    .CPU, .NHWC(1, imageHeight, imageWidth, 1))

  newTensor.withUnsafeMutableBytes {
    guard let u8to = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
    a.rawValue.withUnsafeBytes {
      guard let u8a = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      b.rawValue.withUnsafeBytes {
        guard let u8b = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
          return
        }
        for y in 0..<imageHeight {
          for x in 0..<imageWidth {
            if u8a[y * imageWidth + x] != 0, u8b[y * imageWidth + x] != 0 {
              u8to[y * imageWidth + x] = 255
            } else {
              u8to[y * imageWidth + x] = 0
            }
          }
        }
      }
    }
  }
  return graph.variable(newTensor)
}

private func schpMaskOr(_ a: DynamicGraph.Tensor<UInt8>, _ b: DynamicGraph.Tensor<UInt8>)
  -> DynamicGraph.Tensor<UInt8>
{
  guard a.shape == b.shape else { return a }
  let imageWidth = a.shape[2]
  let imageHeight = a.shape[1]
  let graph = a.graph
  var newTensor = Tensor<UInt8>(
    .CPU, .NHWC(1, imageHeight, imageWidth, 1))

  newTensor.withUnsafeMutableBytes {
    guard let u8to = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
    a.rawValue.withUnsafeBytes {
      guard let u8a = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      b.rawValue.withUnsafeBytes {
        guard let u8b = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
          return
        }
        for y in 0..<imageHeight {
          for x in 0..<imageWidth {
            if u8a[y * imageWidth + x] != 0 {
              u8to[y * imageWidth + x] = 255
            } else if u8b[y * imageWidth + x] != 0 {
              u8to[y * imageWidth + x] = 255
            } else {
              u8to[y * imageWidth + x] = 0
            }
          }
        }
      }
    }
  }
  return graph.variable(newTensor)
}

private func schpMaskNot(_ a: DynamicGraph.Tensor<UInt8>) -> DynamicGraph.Tensor<UInt8> {
  let imageWidth = a.shape[2]
  let imageHeight = a.shape[1]
  let graph = a.graph
  var newTensor = Tensor<UInt8>(
    .CPU, .NHWC(1, imageHeight, imageWidth, 1))

  newTensor.withUnsafeMutableBytes {
    guard let u8to = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
    a.rawValue.withUnsafeBytes {
      guard let u8a = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          u8to[y * imageWidth + x] = u8a[y * imageWidth + x] != 0 ? 0 : 255
        }
      }
    }
  }
  return graph.variable(newTensor)
}

private func schpMaskDilate(_ a: DynamicGraph.Tensor<UInt8>, iterations: Int, size: Int = 5)
  -> DynamicGraph.Tensor<UInt8>
{

  let imageWidth = a.shape[2]
  let imageHeight = a.shape[1]
  let graph = a.graph
  var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
    Int32(imageHeight), Int32(imageWidth), Int32(CCV_8U | CCV_C1), nil, 0)
  let u8 = b!.pointee.data.u8!

  a.rawValue.withUnsafeBytes {
    guard let u8a = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
    for y in 0..<imageHeight {
      for x in 0..<imageWidth {
        u8[y * imageWidth + x] = u8a[y * imageWidth + x]
      }
    }
  }

  for _ in 0..<iterations {
    ccv_dilate(b, &b, 0, Int32(size))
  }

  let newTensor = Tensor<UInt8>(
    .CPU, format: .NHWC, shape: [1, imageHeight, imageWidth, 1],
    unsafeMutablePointer: b!.pointee.data.u8, bindLifetimeOf: b!
  ).copied()

  ccv_matrix_free(b)
  return graph.variable(newTensor)
}
