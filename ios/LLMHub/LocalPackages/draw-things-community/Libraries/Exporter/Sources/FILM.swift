import Diffusion
import Foundation
import NNC

func naiveDenseImageWarp<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ image: DynamicGraph.Tensor<FloatType>, _ flow: DynamicGraph.Tensor<FloatType>
)
  -> DynamicGraph.Tensor<FloatType>
{
  let N = image.shape[0]
  let H = min(image.shape[1], flow.shape[1])
  let W = min(image.shape[2], flow.shape[2])
  let C = image.shape[3]
  let graph = image.graph
  let image = image.rawValue.toCPU()
  let flow = flow.rawValue.toCPU()
  var warpedImageTensor = Tensor<FloatType>(.CPU, .NHWC(N, H, W, C))
  warpedImageTensor.withUnsafeMutableBytes {
    var warped = $0.baseAddress!.assumingMemoryBound(to: FloatType.self)
    flow.withUnsafeBytes {
      var flowp = $0.baseAddress!.assumingMemoryBound(to: FloatType.self)
      image.withUnsafeBytes {
        var imagep = $0.baseAddress!.assumingMemoryBound(to: FloatType.self)
        for _ in 0..<N {
          for j in 0..<H {
            let fj = FloatType(j)
            for i in 0..<W {
              let fi = FloatType(i)
              let y = fj + flowp[1]
              let x = fi + flowp[0]
              let y0 = max(min(Int(floor(y)), H - 1), 0)
              let x0 = max(min(Int(floor(x)), W - 1), 0)
              let x1 = max(min(Int(x0 + 1), W - 1), 0)
              let y1 = max(min(Int(y0 + 1), H - 1), 0)
              let alpha1 = FloatType(max(min(Float(x) - Float(x0), 1), 0))
              let _alpha1 = 1 - alpha1
              let alpha0 = FloatType(max(min(Float(y) - Float(y0), 1), 0))
              let _alpha0 = 1 - alpha0
              let imageT = imagep + y0 * W * C + x0 * C
              let y10 = y1 - y0
              let x10 = x1 - x0
              let a = _alpha1 * _alpha0
              let b = alpha1 * _alpha0
              let c = _alpha1 * alpha0
              let d = alpha1 * alpha0
              for k in 0..<C {
                let v00 = imageT[k]
                let v01 = imageT[k + x10 * C]
                let v10 = imageT[k + y10 * W * C]
                let v11 = imageT[k + y10 * W * C + x10 * C]
                warped[k] = a * v00 + b * v01 + c * v10 + d * v11
              }
              flowp = flowp + 2
              warped = warped + C
            }
          }
          imagep = imagep + H * W * C
        }
      }
    }
  }

  return graph.variable(warpedImageTensor.toGPU(0))
}

func buildImagePyramid<FloatType: TensorNumeric & BinaryFloatingPoint>(
  pyramidLevels: Int, inputTensor: DynamicGraph.Tensor<FloatType>
)
  -> [DynamicGraph.Tensor<FloatType>]
{
  var out = [DynamicGraph.Tensor<FloatType>]()
  var image = inputTensor
  for i in 0..<pyramidLevels {
    out.append(image)
    if i < pyramidLevels - 1 {
      image = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(image)
    }
  }
  return out
}

func subTreeExtractor(k: Int = 64, n: Int = 4) -> Model {
  let x = Input()
  let pool = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var resultArray = [Model.IO]()
  var layerConv2dArray = [Convolution]()
  var out = x as Model.IO
  for i in 0..<n {
    let layerConv2d0 = Convolution(
      groups: 1, filters: (k << i), filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
    out = layerConv2d0(out).leakyReLU(negativeSlope: 0.2)
    layerConv2dArray.append(layerConv2d0)

    let layerConv2d1 = Convolution(
      groups: 1, filters: (k << i), filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
    out = layerConv2d1(out).leakyReLU(negativeSlope: 0.2)
    resultArray.append(out)
    if i < n - 1 {
      out = pool(out)
    }
    layerConv2dArray.append(layerConv2d1)
  }

  return Model([x], resultArray)
}

func buildFeaturePyramid<FloatType: TensorNumeric & BinaryFloatingPoint>(
  pyramidLevels: Int, subLevels: Int, subPyramids: [[DynamicGraph.Tensor<FloatType>]]
) -> [DynamicGraph.Tensor<FloatType>] {
  var featurePyramids = [DynamicGraph.Tensor<FloatType>]()
  for i in 0..<pyramidLevels {
    var featurePyramid = subPyramids[i][0]
    for j in 1..<subLevels {
      if j <= i {
        featurePyramid = Functional.concat(axis: 3, featurePyramid, subPyramids[i - j][j])
      }
    }
    featurePyramids.append(featurePyramid)
  }
  return featurePyramids
}

func flowPyramidSynthesis<FloatType: TensorNumeric & BinaryFloatingPoint>(
  residualPyramid: [DynamicGraph.Tensor<FloatType>]
) -> [DynamicGraph.Tensor<
  FloatType
>] {
  var flow = residualPyramid[6]
  var flowPyramid = [flow]

  for index in 0..<6 {
    let i = 5 - index
    let residualFlow = residualPyramid[i]
    flow = 2 * flow
    flow = Upsample(.bilinear, widthScale: 2, heightScale: 2)(inputs: flow)[0].as(
      of: FloatType.self)
    flow = flow + residualFlow
    flowPyramid.append(flow)
  }

  return Array(flowPyramid.reversed())
}

func Fusion() -> Model {
  var pyramid = [Model.IO]()
  var convs = [Convolution]()
  let input = Input()
  pyramid.append(input)
  var out: Model.IO = input
  for i in 0..<4 {
    let m = 3
    let k = 64
    var numFilters = k << m
    if i < m {
      numFilters = k << i
    }

    let layerConv2d0 = Convolution(
      groups: 1, filters: numFilters, filterSize: [2, 2],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [1, 1])), format: .OIHW)
    convs.append(layerConv2d0)

    let layerConv2d1 = Convolution(
      groups: 1, filters: numFilters, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
    convs.append(layerConv2d1)

    let layerConv2d2 = Convolution(
      groups: 1, filters: numFilters, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
    convs.append(layerConv2d2)
  }

  for index in 0..<4 {
    let i = 3 - index
    out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
    out = convs[i * 3](out)
    let curInput = Input()
    pyramid.append(curInput)
    out = Functional.concat(axis: 3, curInput, out)
    out = convs[i * 3 + 1](out).leakyReLU(negativeSlope: 0.2)
    out = convs[i * 3 + 2](out).leakyReLU(negativeSlope: 0.2)
  }

  let outputConv = Convolution(
    groups: 1, filters: 3, filterSize: [1, 1], format: .OIHW)
  out = outputConv(out)
  convs.append(outputConv)

  return Model(pyramid, [out])
}

func flowEstimator(numConvs: Int, numFilters: Int) -> Model {
  let x = Input()
  var out = x + 0
  var layerConv2dArray = [Convolution]()
  for _ in 0..<numConvs {
    let layerConv2d = Convolution(
      groups: 1, filters: numFilters, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
    out = layerConv2d(out).leakyReLU(negativeSlope: 0.2)
    layerConv2dArray.append(layerConv2d)
  }

  let layerConv2d3 = Convolution(
    groups: 1, filters: numFilters / 2, filterSize: [1, 1], format: .OIHW)
  out = layerConv2d3(out).leakyReLU(negativeSlope: 0.2)
  layerConv2dArray.append(layerConv2d3)

  let layerConv2d4 = Convolution(
    groups: 1, filters: 2, filterSize: [1, 1], format: .OIHW)
  out = layerConv2d4(out)
  layerConv2dArray.append(layerConv2d4)

  return Model([x], [out])
}

func pyramidWarp<FloatType: TensorNumeric & BinaryFloatingPoint>(
  imagePyramid: [DynamicGraph.Tensor<FloatType>], featurePyramid: [DynamicGraph.Tensor<FloatType>],
  flowPyramid: [DynamicGraph.Tensor<FloatType>]
) -> [DynamicGraph.Tensor<FloatType>] {
  let featurePyramid = zip(imagePyramid, featurePyramid).map { Functional.concat(axis: 3, $0, $1) }
  return zip(featurePyramid, flowPyramid).map {
    naiveDenseImageWarp($0, $1)
  }
}

func pyramidFlowEstimator<FloatType: TensorNumeric & BinaryFloatingPoint>(
  featurePyramid: [DynamicGraph.Tensor<FloatType>],
  featurePyramid2: [DynamicGraph.Tensor<FloatType>],
  predictors: [Model]
) -> [DynamicGraph.Tensor<FloatType>] {

  let sharedPredictor = predictors[3]
  let concat = Functional.concat(axis: 3, featurePyramid[6], featurePyramid2[6])
  let output = sharedPredictor(inputs: concat)
  var v = output[0].as(of: FloatType.self)
  var residuals = [v]

  for index in 0..<6 {
    let i = 5 - index
    v = 2 * v
    v = Upsample(.bilinear, widthScale: 2, heightScale: 2)(inputs: v)[0].as(of: FloatType.self)
    let warped = naiveDenseImageWarp(featurePyramid2[i], v)
    var predictor = sharedPredictor
    if i <= 2 {
      predictor = predictors[i]
    }

    let concat = Functional.concat(axis: 3, featurePyramid[i], warped)
    let output = predictor(inputs: concat)
    let vResidual = output[0].as(of: FloatType.self)

    residuals.append(vResidual)
    v = vResidual + v
  }

  return Array(residuals.reversed())
}

func pyramidFlowEstimatorModel<FloatType: TensorNumeric & BinaryFloatingPoint>(
  featurePyramid: [DynamicGraph.Tensor<FloatType>]
) -> [Model] {
  let flowConvs = [3, 3, 3, 3]
  let flowFilters = [32, 64, 128, 256]
  var predictors = [Model]()
  for i in 0..<3 {
    let predictor = flowEstimator(numConvs: flowConvs[i], numFilters: flowFilters[i])
    let concat = Functional.concat(axis: 3, featurePyramid[i], featurePyramid[i])
    predictor.compile(inputs: concat)
    predictors.append(predictor)
  }

  let sharedPredictor = flowEstimator(numConvs: 3, numFilters: 256)
  let concat = Functional.concat(axis: 3, featurePyramid[6], featurePyramid[6])
  sharedPredictor.compile(inputs: concat)
  predictors.append(sharedPredictor)

  return predictors
}

func generateSubPyramidExtractorModel() -> Model {
  var imagePyramids = [Model.IO]()
  var outputs = [Model.IO]()
  var lastSubPyramid: Model.IO? = nil
  for i in 0..<7 {
    let n = min(7 - i, 4)
    let imagePyramid = Input()
    imagePyramids.append(imagePyramid)
    let extractor = subTreeExtractor(n: n)
    let subPyramid = extractor(imagePyramid)
    if let lastSubPyramid = lastSubPyramid {
      subPyramid.add(dependencies: [lastSubPyramid])
    }
    lastSubPyramid = subPyramid
    outputs.append(subPyramid)
  }

  return Model(imagePyramids, outputs)
}

func subPyramidsExtractorModelParse<FloatType: TensorNumeric & BinaryFloatingPoint>(
  subPyramids: [DynamicGraph.Tensor<FloatType>]
) -> [[DynamicGraph
  .Tensor<FloatType>]]
{
  var result = [[DynamicGraph.Tensor<FloatType>]]()
  var index = 0
  for i in 0..<7 {
    let n = min(7 - i, 4)
    var currentSubPyramids = [DynamicGraph.Tensor<FloatType>]()
    for _ in 0..<n {
      currentSubPyramids.append(subPyramids[index])
      index += 1
    }
    result.append(currentSubPyramids)
  }
  return result
}

func buildFeaturePyramid<FloatType: TensorNumeric & BinaryFloatingPoint>(
  presetSubPyramidExtractorModel: Model?, filePath: String,
  imagePyramids: [DynamicGraph.Tensor<FloatType>]
) -> ([DynamicGraph.Tensor<FloatType>], Model) {
  let graph = imagePyramids[0].graph
  let subPyramidExtractorModel: Model
  if let presetSubPyramidExtractorModel = presetSubPyramidExtractorModel {
    subPyramidExtractorModel = presetSubPyramidExtractorModel
  } else {
    subPyramidExtractorModel = generateSubPyramidExtractorModel()
    subPyramidExtractorModel.maxConcurrency = .limit(4)
    subPyramidExtractorModel.compile(inputs: imagePyramids)
    graph.openStore(
      filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
    ) {
      $0.read("subPyramidExtractor", model: subPyramidExtractorModel)
    }
  }

  let subPyramids = subPyramidExtractorModel(
    inputs: imagePyramids[0], Array(imagePyramids[1...])
  ).map { $0.as(of: FloatType.self) }
  let subPyramids1 = subPyramidsExtractorModelParse(subPyramids: subPyramids)
  return (
    buildFeaturePyramid(
      pyramidLevels: 7, subLevels: 4, subPyramids: subPyramids1), subPyramidExtractorModel
  )
}

func flowPyramidSynthesis<FloatType: TensorNumeric & BinaryFloatingPoint>(
  featurePyramid: [DynamicGraph.Tensor<FloatType>],
  featurePyramid2: [DynamicGraph.Tensor<FloatType>],
  predictors: [Model]
) -> [DynamicGraph.Tensor<FloatType>] {
  let residualFlowPyramid = pyramidFlowEstimator(
    featurePyramid: featurePyramid, featurePyramid2: featurePyramid2,
    predictors: predictors)

  return flowPyramidSynthesis(
    residualPyramid: residualFlowPyramid)[0..<5].map { $0 * 0.5 }
}

func generateFILMIntermediateImage<FloatType: TensorNumeric & BinaryFloatingPoint>(
  imageTensor: Tensor<FloatType>, imageTensor2: Tensor<FloatType>,
  modelPath: String,
  presetSubPyramidExtractorModel: Model?, presetPredictors: [Model]?, presetFusion: Model?,
  graph: DynamicGraph
) -> (Tensor<FloatType>, Model, [Model], Model) {

  return graph.withNoGrad {

    var inputTensor = graph.variable(imageTensor).toGPU(0)
    var inputTensor2 = graph.variable(imageTensor2).toGPU(0)
    let shape = inputTensor.shape
    inputTensor = (inputTensor + 1) * 0.5
    inputTensor2 = (inputTensor2 + 1) * 0.5

    let resizeHeight = Int(Double(shape[1] / 64).rounded() * 64)
    let resizeWidth = Int(Double(shape[2] / 64).rounded() * 64)

    inputTensor = Upsample(
      .bilinear, widthScale: Float(resizeWidth) / Float(shape[2]),
      heightScale: Float(resizeHeight) / Float(shape[1]))(inputTensor)
    inputTensor2 = Upsample(
      .bilinear, widthScale: Float(resizeWidth) / Float(shape[2]),
      heightScale: Float(resizeHeight) / Float(shape[1]))(inputTensor2)

    let imagePyramids = buildImagePyramid(pyramidLevels: 7, inputTensor: inputTensor)
    let imagePyramids2 = buildImagePyramid(pyramidLevels: 7, inputTensor: inputTensor2)

    let (featurePyramid, subPyramidExtractorModel) = buildFeaturePyramid(
      presetSubPyramidExtractorModel: presetSubPyramidExtractorModel, filePath: modelPath,
      imagePyramids: imagePyramids)
    let (featurePyramid2, _) = buildFeaturePyramid(
      presetSubPyramidExtractorModel: subPyramidExtractorModel, filePath: modelPath,
      imagePyramids: imagePyramids2)

    let predictors: [Model]
    if let presetPredictors = presetPredictors {
      predictors = presetPredictors
    } else {
      predictors = pyramidFlowEstimatorModel(featurePyramid: featurePyramid)
      graph.openStore(
        modelPath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: modelPath)
      ) {
        for i in 0..<4 {
          $0.read("pyramid_flow_estimator_\(i)", model: predictors[i])
        }
      }
    }

    let forwardFlow = flowPyramidSynthesis(
      featurePyramid: featurePyramid, featurePyramid2: featurePyramid2,
      predictors: predictors)

    let backwardFlow = flowPyramidSynthesis(
      featurePyramid: featurePyramid2, featurePyramid2: featurePyramid,
      predictors: predictors)

    let forwardWarpedPyramid = pyramidWarp(
      imagePyramid: imagePyramids, featurePyramid: featurePyramid, flowPyramid: backwardFlow)
    let backwardWarpedPyramid = pyramidWarp(
      imagePyramid: imagePyramids2, featurePyramid: featurePyramid2, flowPyramid: forwardFlow)

    let alignedPyramid = Array(
      (zip(zip(forwardWarpedPyramid, backwardWarpedPyramid), zip(backwardFlow, forwardFlow)).map {
        Functional.concat(axis: 3, $0.0, $0.1, $1.0, $1.1)
      }).reversed())

    let fusion: Model
    if let presetFusion = presetFusion {
      fusion = presetFusion
    } else {
      fusion = Fusion()
      fusion.maxConcurrency = .limit(4)
      fusion.compile(inputs: alignedPyramid)
      graph.openStore(
        modelPath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: modelPath)
      ) {
        $0.read("fusion", model: fusion)
      }
    }

    let output = fusion(
      inputs: alignedPyramid[0], Array(alignedPyramid[1...]))
    var result = output[0].as(of: FloatType.self)
    result = Upsample(
      .bilinear, widthScale: Float(shape[2]) / Float(resizeWidth),
      heightScale: Float(shape[1]) / Float(resizeHeight))(result)

    result = result * 2 - 1
    result = result.toCPU()
    return (
      result.rawValue, subPyramidExtractorModel, predictors, fusion
    )
  }
}
