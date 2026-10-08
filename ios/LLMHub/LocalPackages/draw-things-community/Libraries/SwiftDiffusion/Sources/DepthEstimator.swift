import NNC

public struct DepthEstimator<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePaths: (String, String)
  public let usesFlashAttention: Bool
  public init(filePaths: (String, String), usesFlashAttention: Bool) {
    self.filePaths = filePaths
    self.usesFlashAttention = usesFlashAttention
  }
}

extension DepthEstimator {
  public func estimate(_ image: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    let graph = image.graph
    return graph.withNoGrad {
      let mean = graph.variable(
        Tensor<FloatType>(
          from: Tensor<Float>(
            [
              Float(2 * 0.485 - 1), Float(2 * 0.456 - 1), Float(2 * 0.406 - 1),
            ], .GPU(0), .NHWC(1, 1, 1, 3))))
      let invStd = graph.variable(
        Tensor<FloatType>(
          from: Tensor<Float>(
            [
              Float(0.5 / 0.229), Float(0.5 / 0.224), Float(0.5 / 0.225),
            ], .GPU(0), .NHWC(1, 1, 1, 3))))
      var input = image.toGPU(0)
      let inputHeight = input.shape[1]
      let inputWidth = input.shape[2]
      let resizeWidth: Int
      let resizeHeight: Int
      if inputWidth > inputHeight {
        if inputHeight < 518 {
          resizeHeight = 518
        } else if inputHeight > 1036 {
          resizeHeight = 1036
        } else {
          resizeHeight = Int((Double(inputHeight) / 14).rounded() * 14)
        }
        resizeWidth = Int(
          (Double(inputWidth) * Double(resizeHeight / 14) / Double(inputHeight)).rounded() * 14)
      } else {
        if inputWidth < 518 {
          resizeWidth = 518
        } else if inputWidth > 1036 {
          resizeWidth = 1036
        } else {
          resizeWidth = Int((Double(inputWidth) / 14).rounded() * 14)
        }
        resizeHeight = Int(
          (Double(inputHeight) * Double(resizeWidth / 14) / Double(inputWidth)).rounded() * 14)
      }
      precondition(input.shape[3] == 3)
      if inputHeight != resizeHeight || inputWidth != resizeWidth {
        input =
          (Upsample(
            .bilinear, widthScale: Float(resizeWidth) / Float(inputWidth),
            heightScale: Float(resizeHeight) / Float(inputHeight))(input) - mean) .* invStd
      } else {
        input = (input - mean) .* invStd
      }
      let gridX = resizeWidth / 14
      let gridY = resizeHeight / 14
      let vit = DinoVisionTransformer(
        FloatType.self, gridX: gridX, gridY: gridY, width: 1024, layers: 24, heads: 16,
        batchSize: 1, intermediateLayers: [4, 11, 17, 23], usesFlashAttention: usesFlashAttention)
      vit.maxConcurrency = .limit(4)
      vit.compile(inputs: input)
      graph.openStore(
        filePaths.0, flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths.0)
      ) {
        $0.read("vit", model: vit)
      }
      let outs = vit(inputs: input).map { $0.as(of: FloatType.self) }
      let x0 = outs[0][1..<(gridX * gridY + 1), 0..<1024].reshaped(.NHWC(1, gridY, gridX, 1024))
        .copied()
      let x1 = outs[1][1..<(gridX * gridY + 1), 0..<1024].reshaped(.NHWC(1, gridY, gridX, 1024))
        .copied()
      let x2 = outs[2][1..<(gridX * gridY + 1), 0..<1024].reshaped(.NHWC(1, gridY, gridX, 1024))
        .copied()
      let x3 = outs[3][1..<(gridX * gridY + 1), 0..<1024].reshaped(.NHWC(1, gridY, gridX, 1024))
        .copied()
      let depthHead = DepthHead(gridX: gridX, gridY: gridY, paddingFinalConvLayer: true)
      depthHead.maxConcurrency = .limit(4)
      depthHead.compile(inputs: x0, x1, x2, x3)
      graph.openStore(
        filePaths.1, flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths.1)
      ) {
        $0.read("depth_head", model: depthHead)
      }
      let out = depthHead(inputs: x0, x1, x2, x3)[0].as(of: FloatType.self)
      return out[0..<1, 0..<resizeHeight, 0..<resizeWidth, 0..<1].copied()
    }
  }
}
