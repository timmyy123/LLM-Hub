import LLM
import NNC

public struct ImageEncoder<FloatType: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  let version: ImageEncoderVersion
  public init(filePath: String, version: ImageEncoderVersion) {
    self.filePath = filePath
    self.version = version
  }
}

extension ImageEncoder {
  public func encode(_ x: [DynamicGraph.Tensor<FloatType>]) -> [[DynamicGraph.Tensor<FloatType>]] {
    precondition(x.count > 0)
    let graph = x[0].graph
    return graph.withNoGrad {
      let vit: Model
      let otherInputs: [DynamicGraph.Tensor<FloatType>]
      let modelKey: String
      switch version {
      case .clipL14_336:
        vit = CLIPVisionTransformer(
          FloatType.self, grid: 24, width: 1024, layers: 23, heads: 16,
          batchSize: 1, noFinalLayerNorm: true)
        otherInputs = []
        modelKey = "vision_model"
      case .openClipH14:
        vit = VisionTransformer(
          FloatType.self, grid: 16, width: 1280, layers: 31, heads: 16, batchSize: 1,
          noFinalLayerNorm: true)
        otherInputs = []
        modelKey = "vision_model"
      case .siglipL27_384:
        vit = SigLIPVisionTransformer(
          FloatType.self, gridX: 27, gridY: 27, filterSize: 14, width: 1152, layers: 27, heads: 16,
          MLP: 4304, batchSize: 1, usesFlashAttention: false, approximate: .tanh)
        otherInputs = []
        modelKey = "vit"
      case .siglip2L27_512:
        vit = SigLIPVisionTransformer(
          FloatType.self, gridX: 32, gridY: 32, filterSize: 16, width: 1152, layers: 27, heads: 16,
          MLP: 4304, batchSize: 1, usesFlashAttention: false, approximate: .tanh)
        otherInputs = []
        modelKey = "vit"
      case .eva02L14_336:
        vit = Eva02VisionTransformer(
          FloatType.self, grid: 24, outputChannels: 768, width: 1024, MLP: 2730, layers: 24,
          heads: 16, batchSize: 1)
        let rotTensor = Eva02RotaryPositionEmbedding(
          height: 24, width: 24, tokenLength: 1, channels: 64)
        otherInputs = [graph.variable(Tensor<FloatType>(from: rotTensor).toGPU(0))]
        modelKey = "vision_model"
      }
      vit.maxConcurrency = .limit(4)
      vit.compile(inputs: [x[0]] + otherInputs)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read(modelKey, model: vit, codec: [.ezm7, .externalData, .q6p, .q8p])
      }
      return x.map {
        let output = vit(inputs: $0, otherInputs).map { $0.as(of: FloatType.self) }
        switch version {
        case .clipL14_336:
          return [output[0].reshaped(.HWC(1, 577, 1024))]
        case .siglipL27_384:
          return [output[0].reshaped(.HWC(1, 729, 1152))]
        case .siglip2L27_512:
          return [output[0].reshaped(.HWC(1, 1024, 1152))]
        case .eva02L14_336:
          return output.map {
            let shape = $0.shape
            return $0.reshaped(.HWC(1, 577, shape[shape.count - 1]))
          }
        case .openClipH14:
          return [output[0].reshaped(.HWC(1, 257, 1280))]
        }
      }
    }
  }
}
