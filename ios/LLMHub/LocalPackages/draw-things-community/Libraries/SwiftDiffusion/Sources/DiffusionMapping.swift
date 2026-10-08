import Foundation
import NNC

public struct DiffusionMapping<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePath: String
  public let usesFlashAttention: Bool
  public let steps: Int
  public let negativePromptForImagePrior: Bool
  public let CLIPWeight: Float
  public let externalOnDemand: Bool
  public init(
    filePath: String, usesFlashAttention: Bool, steps: Int, negativePromptForImagePrior: Bool,
    CLIPWeight: Float, externalOnDemand: Bool
  ) {
    self.filePath = filePath
    self.usesFlashAttention = usesFlashAttention
    self.steps = steps
    self.negativePromptForImagePrior = negativePromptForImagePrior
    self.CLIPWeight = CLIPWeight
    self.externalOnDemand = externalOnDemand
  }
}

extension DiffusionMapping {
  public func sample(
    textEncoding: DynamicGraph.Tensor<FloatType>, textEmbedding: DynamicGraph.Tensor<FloatType>,
    tokens: DynamicGraph.Tensor<Int32>
  ) -> DynamicGraph.Tensor<FloatType> {
    let graph = textEncoding.graph
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    guard CLIPWeight > 0 else {
      var imageEmb = graph.variable(.GPU(0), .WC(2, 768), of: FloatType.self)
      let zeroImgEmbGPU = graph.variable(.GPU(0), .WC(1, 768), of: FloatType.self)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("zero_img_emb", variable: zeroImgEmbGPU, codec: [.q6p, .q8p, .ezm7, .externalData])
      }
      imageEmb[0..<1, 0..<768] = zeroImgEmbGPU
      imageEmb[1..<2, 0..<768] = zeroImgEmbGPU
      return imageEmb
    }
    var betas = [Double]()
    for i in 0..<1_000 {
      let t1 = Double(i) / Double(1_000)
      let t2 = Double(i + 1) / Double(1_000)
      let cos1 = cos((t1 + 0.008) / 1.008 * Double.pi / 2)
      let cos2 = cos((t2 + 0.008) / 1.008 * Double.pi / 2)
      let beta = min(1 - (cos2 * cos2) / (cos1 * cos1), 0.999)
      betas.append(beta)
    }
    var cumprod: Double = 1
    let alphasCumprod = betas.map {
      cumprod *= 1 - $0
      return cumprod
    }
    var newBetas = [Double]()
    var lastAlphasCumprod: Double = 1.0
    let timestamps: [Int] = (0..<steps).map {
      (999 * $0 + (steps - 1) / 2) / (steps - 1)
    }
    for i in timestamps {
      newBetas.append(1 - alphasCumprod[i] / lastAlphasCumprod)
      lastAlphasCumprod = alphasCumprod[i]
    }
    cumprod = 1
    let newAlphasCumprod = newBetas.map {
      cumprod *= 1 - $0
      return cumprod
    }
    var posteriorVariance = [Double]()
    var posteriorLogVarianceClipped = [Double]()
    var posteriorMeanCoef1 = [Double]()
    var posteriorMeanCoef2 = [Double]()
    for i in 0..<newAlphasCumprod.count {
      let alphasCumProdPrev = i > 0 ? newAlphasCumprod[i - 1] : 1
      posteriorVariance.append(newBetas[i] * (1 - alphasCumProdPrev) / (1 - newAlphasCumprod[i]))
      if i == 0 {
        posteriorLogVarianceClipped.append(
          log(newBetas[i + 1] * (1 - newAlphasCumprod[i]) / (1 - newAlphasCumprod[i + 1])))
      } else {
        posteriorLogVarianceClipped.append(
          log(newBetas[i] * (1 - newAlphasCumprod[i - 1]) / (1 - newAlphasCumprod[i])))
      }
      posteriorMeanCoef1.append(
        newBetas[i] * alphasCumProdPrev.squareRoot() / (1 - newAlphasCumprod[i]))
      posteriorMeanCoef2.append(
        (1 - alphasCumProdPrev) * (1 - newBetas[i]).squareRoot() / (1 - newAlphasCumprod[i]))
    }
    var finalImageEmb: DynamicGraph.Tensor<FloatType>? = nil
    graph.openStore(
      filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
    ) {
      let textProjectionGPU = graph.variable(.GPU(0), .WC(768, 768), of: FloatType.self)
      $0.read(
        "text_projection", variable: textProjectionGPU, codec: [.q6p, .q8p, .ezm7, .externalData])
      let textProjectedEmbedding = textEmbedding * textProjectionGPU
      let textEncProj = Dense(count: 2048)
      let textEmbProj = Dense(count: 2048)
      let clipImgProj = Dense(count: 2048)
      let timeEmbed = timestepEmbedding(prefix: "model.time_embed", channels: 2048)
      var unconditionalTokenLength = 77
      var tokenLength = 77
      for i in 0..<77 {
        if tokens[i] == 49407 && unconditionalTokenLength == 77 {
          unconditionalTokenLength = i + 1
        }
        if tokens[i + 77] == 49407 && tokenLength == 77 {
          tokenLength = i + 1
        }
      }
      textEncProj.compile(inputs: textEncoding)
      textEmbProj.compile(inputs: textProjectedEmbedding)
      let noiseGPU = graph.variable(.GPU(0), .WC(1, 768), of: FloatType.self)
      clipImgProj.compile(inputs: noiseGPU)
      let timesteps = graph.variable(
        Tensor<FloatType>(
          from: timeEmbedding(timestep: 999, batchSize: 1, embeddingSize: 2048, maxPeriod: 10_000)
            .toGPU(0)))
      timeEmbed.compile(inputs: timesteps)
      $0.read("time_embed", model: timeEmbed, codec: [.q6p, .q8p, .ezm7, .externalData])
      $0.read("clip_img_proj", model: clipImgProj, codec: [.q6p, .q8p, .ezm7, .externalData])
      $0.read("text_enc_proj", model: textEncProj, codec: [.q6p, .q8p, .ezm7, .externalData])
      $0.read("text_emb_proj", model: textEmbProj, codec: [.q6p, .q8p, .ezm7, .externalData])
      let textEncOut = textEncProj(inputs: textEncoding)[0].as(of: FloatType.self)
      let textEmbOut = textEmbProj(inputs: textProjectedEmbedding)[0].as(of: FloatType.self)
      var dmInputTensorGPU = graph.variable(.GPU(0), .WC(2 * 81, 2048), of: FloatType.self)
      dmInputTensorGPU[0..<77, 0..<2048] = textEncOut[1..<2, 0..<77, 0..<2048].reshaped(
        .WC(77, 2048))
      dmInputTensorGPU[77..<78, 0..<2048] = textEmbOut[1..<2, 0..<2048]
      if negativePromptForImagePrior {
        dmInputTensorGPU[81..<(81 + 77), 0..<2048] = textEncOut[0..<1, 0..<77, 0..<2048].reshaped(
          .WC(77, 2048))
        dmInputTensorGPU[(81 + 77)..<(81 + 78), 0..<2048] = textEmbOut[0..<1, 0..<2048]
      } else {
        dmInputTensorGPU[81..<(81 + 77), 0..<2048] = textEncOut[2..<3, 0..<77, 0..<2048].reshaped(
          .WC(77, 2048))
        dmInputTensorGPU[(81 + 77)..<(81 + 78), 0..<2048] = textEmbOut[2..<3, 0..<2048]
      }
      let prdEmb = graph.variable(.GPU(0), .WC(1, 2048), of: FloatType.self)
      $0.read("prd_emb", variable: prdEmb, codec: [.q6p, .q8p, .ezm7, .externalData])
      dmInputTensorGPU[80..<81, 0..<2048] = prdEmb
      dmInputTensorGPU[(81 + 80)..<(81 + 81), 0..<2048] = prdEmb
      let positionalEmbedding = graph.variable(.GPU(0), .WC(81, 2048), of: FloatType.self)
      let clipStd = graph.variable(.GPU(0), .WC(1, 768), of: FloatType.self)
      let clipMean = graph.variable(.GPU(0), .WC(1, 768), of: FloatType.self)
      $0.read(
        "positional_embedding", variable: positionalEmbedding,
        codec: [.q6p, .q8p, .ezm7, .externalData])
      $0.read("clip_std", variable: clipStd, codec: [.q6p, .q8p, .ezm7, .externalData])
      $0.read("clip_mean", variable: clipMean, codec: [.q6p, .q8p, .ezm7, .externalData])
      var positionalEmbeddingGPU = graph.variable(.GPU(0), .WC(2 * 81, 2048), of: FloatType.self)
      positionalEmbeddingGPU[0..<81, 0..<2048] = positionalEmbedding
      positionalEmbeddingGPU[81..<(81 * 2), 0..<2048] = positionalEmbedding
      let diffusionMapping = DiffusionMappingModel(
        numberOfLayers: 20, k: 64, h: 32, b: 2, t: 81, outChannels: 768,
        usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
      let dmCasualAttensionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(2, 1, 81, 81)))
      for i in 0..<81 {
        for j in 0..<81 {
          dmCasualAttensionMask[0, 0, i, j] = 0
          dmCasualAttensionMask[1, 0, i, j] = 0
        }
      }
      for i in 0..<80 {
        for j in (i + 1)..<81 {
          dmCasualAttensionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
          dmCasualAttensionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      for i in 0..<81 {
        for j in tokenLength..<77 {
          dmCasualAttensionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
        for j in (negativePromptForImagePrior ? unconditionalTokenLength : 2)..<77 {
          dmCasualAttensionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      var dmCasualAttentionMaskGPU = dmCasualAttensionMask.toGPU(0)
      diffusionMapping.compile(inputs: dmInputTensorGPU, dmCasualAttentionMaskGPU)
      let zeroImgEmbGPU = graph.variable(.GPU(0), .WC(1, 768), of: FloatType.self)
      $0.read(
        "diffusion_mapping", model: diffusionMapping,
        codec: [.jit, .q6p, .q8p, .ezm7, externalData])
      $0.read("zero_img_emb", variable: zeroImgEmbGPU, codec: [.q6p, .q8p, .ezm7, .externalData])
      noiseGPU.randn()
      var x = noiseGPU
      for (i, timestep) in timestamps.enumerated().reversed() {
        let timesteps = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: Float(timestep), batchSize: 1, embeddingSize: 2048, maxPeriod: 10_000
            ).toGPU(0)))
        let tEmb = timeEmbed(inputs: timesteps)[0].as(of: FloatType.self)
        let xProj = clipImgProj(inputs: x)[0].as(of: FloatType.self)
        dmInputTensorGPU[78..<79, 0..<2048] = tEmb
        dmInputTensorGPU[(81 + 78)..<(81 + 79), 0..<2048] = tEmb
        dmInputTensorGPU[79..<80, 0..<2048] = xProj
        dmInputTensorGPU[(81 + 79)..<(81 + 80), 0..<2048] = xProj
        let input = dmInputTensorGPU + positionalEmbeddingGPU
        let result = diffusionMapping(inputs: input, dmCasualAttentionMaskGPU)[0].as(
          of: FloatType.self)
        let condEps = result[0..<1, 0..<1, 0..<768].reshaped(.WC(1, 768))
        let uncondEps = result[1..<2, 0..<1, 0..<768].reshaped(.WC(1, 768))
        let eps = (uncondEps + 4 * (condEps - uncondEps)).clamped(-10...10)
        let posteriorMean = Functional.add(
          left: eps, right: x, leftScalar: Float(posteriorMeanCoef1[i]),
          rightScalar: Float(posteriorMeanCoef2[i]))
        if i > 0 {
          noiseGPU.randn()
          x = Functional.add(
            left: posteriorMean, right: noiseGPU,
            rightScalar: Float(exp(0.5 * posteriorLogVarianceClipped[i])))
        } else {
          x = posteriorMean
        }
      }
      let posImageEmbGPU = x .* clipStd + clipMean
      var imageEmb = graph.variable(.GPU(0), .WC(2, 768), of: FloatType.self)
      imageEmb[1..<2, 0..<768] = Functional.add(
        left: zeroImgEmbGPU, right: posImageEmbGPU, leftScalar: 1 - CLIPWeight,
        rightScalar: CLIPWeight)
      guard !negativePromptForImagePrior && unconditionalTokenLength > 2 else {
        // This is it.
        imageEmb[0..<1, 0..<768] = zeroImgEmbGPU
        finalImageEmb = imageEmb.reshaped(.HWC(2, 1, 768))
        return
      }
      for i in 0..<81 {
        for j in 0..<(i + 1) {
          dmCasualAttensionMask[0, 0, i, j] = 0
        }
        for j in unconditionalTokenLength..<77 {
          dmCasualAttensionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
        }
      }
      dmCasualAttentionMaskGPU = dmCasualAttensionMask.toGPU(0)
      dmInputTensorGPU[0..<77, 0..<2048] = textEncOut[0..<1, 0..<77, 0..<2048].reshaped(
        .WC(77, 2048))
      dmInputTensorGPU[77..<78, 0..<2048] = textEmbOut[0..<1, 0..<2048]
      noiseGPU.randn()
      x = noiseGPU
      for (i, timestep) in timestamps.enumerated().reversed() {
        let timesteps = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: Float(timestep), batchSize: 1, embeddingSize: 2048, maxPeriod: 10_000
            ).toGPU(0)))
        let tEmb = timeEmbed(inputs: timesteps)[0].as(of: FloatType.self)
        let xProj = clipImgProj(inputs: x)[0].as(of: FloatType.self)
        dmInputTensorGPU[78..<79, 0..<2048] = tEmb
        dmInputTensorGPU[(81 + 78)..<(81 + 79), 0..<2048] = tEmb
        dmInputTensorGPU[79..<80, 0..<2048] = xProj
        dmInputTensorGPU[(81 + 79)..<(81 + 80), 0..<2048] = xProj
        let input = dmInputTensorGPU + positionalEmbeddingGPU
        let result = diffusionMapping(inputs: input, dmCasualAttentionMaskGPU)[0].as(
          of: FloatType.self)
        let condEps = result[0..<1, 0..<1, 0..<768].reshaped(.WC(1, 768))
        let uncondEps = result[1..<2, 0..<1, 0..<768].reshaped(.WC(1, 768))
        let eps = (uncondEps + 4 * (condEps - uncondEps)).clamped(-10...10)
        let posteriorMean = Functional.add(
          left: eps, right: x, leftScalar: Float(posteriorMeanCoef1[i]),
          rightScalar: Float(posteriorMeanCoef2[i]))
        if i > 0 {
          noiseGPU.randn()
          x = Functional.add(
            left: posteriorMean, right: noiseGPU,
            rightScalar: Float(exp(0.5 * posteriorLogVarianceClipped[i])))
        } else {
          x = posteriorMean
        }
      }
      let negImageEmbGPU = x .* clipStd + clipMean
      imageEmb[0..<1, 0..<768] = Functional.add(
        left: zeroImgEmbGPU, right: negImageEmbGPU, leftScalar: 1 - CLIPWeight,
        rightScalar: CLIPWeight)
      finalImageEmb = imageEmb.reshaped(.HWC(2, 1, 768))
    }
    return finalImageEmb!
  }
}
