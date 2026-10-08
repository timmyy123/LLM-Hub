import Diffusion
import NNC

extension ControlModel {

  public static func hed(
    _ inputTensor: DynamicGraph.Tensor<FloatType>, modelFilePath: String
  )
    -> DynamicGraph.Tensor<FloatType>
  {
    assert(inputTensor.format == .NHWC)
    let graph = inputTensor.graph
    let height = inputTensor.shape[1]
    let width = inputTensor.shape[2]

    var rgbInputTensor = (inputTensor.copied() + 1.0) * 127.5
    let rgbAdjust = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 1, 3)))
    rgbAdjust[0, 0, 0, 0] = 104.00698793
    rgbAdjust[0, 0, 0, 1] = 116.66876762
    rgbAdjust[0, 0, 0, 2] = 122.67891434
    rgbInputTensor = rgbInputTensor - rgbAdjust.toGPU(0)
    let hedModelVgg = HEDModel(inputWidth: width, inputHeight: height)
    hedModelVgg.maxConcurrency = .limit(4)
    hedModelVgg.compile(inputs: rgbInputTensor)
    graph.openStore(
      modelFilePath, flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: modelFilePath)
    ) {
      $0.read("hed_vgg", model: hedModelVgg)
    }

    let result =
      hedModelVgg(inputs: rgbInputTensor)[0].as(of: FloatType.self).clamped(0...1)
    return result.copied()
  }
}
