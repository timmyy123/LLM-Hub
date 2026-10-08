import NNC
import Tokenizer

public struct CLIPModelWithProjection<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePaths: [String]
  public let tokenizer: Tokenizer
  public init(filePaths: [String], tokenizer: Tokenizer) {
    self.filePaths = filePaths
    self.tokenizer = tokenizer
  }
}

extension CLIPModelWithProjection {
  public func encode(_ image: DynamicGraph.Tensor<FloatType>, texts: [String]) -> [Float] {
    guard !texts.isEmpty else { return [] }
    let graph = image.graph
    let textEmbeds = graph.withNoGrad {
      let maxLength = 77
      let (_, tokens, _, _, _) = tokenizer.tokenize(text: texts[0], truncation: true, maxLength: 77)
      let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [maxLength], of: Int32.self)
      let positionTensor = graph.variable(
        .CPU, format: .NHWC, shape: [maxLength], of: Int32.self)
      // end token
      var endTokenPosition = maxLength - 1
      for i in 0..<maxLength {
        tokensTensor[i] = tokens[i]
        if endTokenPosition == maxLength - 1 && tokens[i] == 49407 {
          endTokenPosition = i
        }
        positionTensor[i] = Int32(i)
      }
      var causalAttentionMask = Tensor<FloatType>(
        Array(repeating: 0, count: maxLength * maxLength), .CPU, .NHWC(1, 1, maxLength, maxLength)
      )
      for i in 0..<(maxLength - 1) {
        for j in (i + 1)..<maxLength {
          causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      let tokensTensorGPU = tokensTensor.toGPU(0)
      let positionTensorGPU = positionTensor.toGPU(0)
      let causalAttentionMaskGPU = graph.variable(causalAttentionMask.toGPU())
      let textModel = CLIPTextModel(
        FloatType.self, injectEmbeddings: false,
        vocabularySize: 49408, maxLength: maxLength, maxTokenLength: maxLength, embeddingSize: 768,
        numLayers: 12, numHeads: 12, batchSize: 1, intermediateSize: 3072,
        usesFlashAttention: true
      ).0
      textModel.maxConcurrency = .limit(4)
      textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
      graph.openStore(
        filePaths[0], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[0])
      ) {
        $0.read("text_model", model: textModel, codec: [.q6p, .q8p, .jit, .externalData])
      }
      let textOut = textModel(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[
        0
      ].as(
        of: FloatType.self
      ).reshaped(.HWC(1, maxLength, 768))
      var textEmbeds = [textOut[0..<1, endTokenPosition..<(endTokenPosition + 1), 0..<768]]
      for i in 1..<texts.count {
        let (_, tokens, _, _, _) = tokenizer.tokenize(
          text: texts[i], truncation: true, maxLength: 77)
        var endTokenPosition = maxLength - 1
        for i in 0..<maxLength {
          tokensTensor[i] = tokens[i]
          if endTokenPosition == maxLength - 1 && tokens[i] == 49407 {
            endTokenPosition = i
          }
        }
        let tokensTensorGPU = tokensTensor.toGPU(0)
        let textOut = textModel(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[
          0
        ].as(
          of: FloatType.self
        ).reshaped(.HWC(1, maxLength, 768))
        textEmbeds.append(textOut[0..<1, endTokenPosition..<(endTokenPosition + 1), 0..<768])
      }
      return textEmbeds
    }
    return graph.withNoGrad {
      let inputHeight = image.shape[1]
      let inputWidth = image.shape[2]
      precondition(image.shape[3] == 3)
      let mean = graph.variable(
        Tensor<FloatType>(
          [
            FloatType(2 * 0.48145466 - 1), FloatType(2 * 0.4578275 - 1),
            FloatType(2 * 0.40821073 - 1),
          ], .GPU(0), .NHWC(1, 1, 1, 3)))
      let invStd = graph.variable(
        Tensor<FloatType>(
          [
            FloatType(0.5 / 0.26862954), FloatType(0.5 / 0.26130258), FloatType(0.5 / 0.27577711),
          ],
          .GPU(0), .NHWC(1, 1, 1, 3)))
      var imageTensorsGPU: DynamicGraph.Tensor<FloatType> = image.toGPU(0)
      if inputHeight != 224 || inputWidth != 224 {
        imageTensorsGPU =
          (Upsample(
            .bilinear, widthScale: Float(224) / Float(inputWidth),
            heightScale: Float(224) / Float(inputHeight))(imageTensorsGPU) - mean) .* invStd
      } else {
        imageTensorsGPU = (imageTensorsGPU - mean) .* invStd
      }
      let vit = CLIPVisionTransformer(
        FloatType.self, grid: 16, width: 1024, layers: 24, heads: 16, batchSize: 1)
      vit.maxConcurrency = .limit(4)
      vit.compile(inputs: imageTensorsGPU)
      let visualProj = graph.variable(.GPU(0), .NC(1024, 768), of: FloatType.self)
      let textProj = graph.variable(.GPU(0), .NC(768, 768), of: FloatType.self)
      graph.openStore(
        filePaths[filePaths.count - 1], flags: .readOnly,
        externalStore: TensorData.externalStore(filePath: filePaths[filePaths.count - 1])
      ) {
        $0.read("vision_model", model: vit)
        $0.read("visual_proj", variable: visualProj)
        $0.read("text_proj", variable: textProj)
      }
      var imageOut = vit(inputs: imageTensorsGPU)[0].as(of: FloatType.self)
      imageOut = imageOut * visualProj
      imageOut = imageOut .* (1 / imageOut.reduced(.norm2, axis: [1]))

      return textEmbeds.map { textEmbed in
        var textOut = textEmbed * textProj
        textOut = textOut .* (1 / textOut.reduced(.norm2, axis: [2]))
        let result = Functional.matmul(left: textOut, right: imageOut, rightTranspose: (0, 1))
        return Float(result.toCPU()[0, 0, 0])
      }
    }
  }
}
