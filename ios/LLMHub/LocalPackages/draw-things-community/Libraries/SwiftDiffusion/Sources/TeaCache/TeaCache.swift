import NNC

public struct TeaCacheConfiguration {
  public var coefficients: (Float, Float, Float, Float, Float)
  public var steps: ClosedRange<Int>
  public var threshold: Float
  public var maxSkipSteps: Int
  public init(
    coefficients: (Float, Float, Float, Float, Float), steps: ClosedRange<Int>, threshold: Float,
    maxSkipSteps: Int
  ) {
    self.coefficients = coefficients
    self.steps = steps
    self.threshold = threshold
    self.maxSkipSteps = maxSkipSteps
  }

  func suffix(for version: ModelVersion) -> String {
    switch version {
    case .minimaxH3:
      // The retained model contains only blocks 1–49 and the output heads.
      return threshold > 0 ? ":[teacache]" : ""
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
      .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3,
      .seedvr2_3b, .seedvr2_7b, .longcatVideoAvatar1_5:
      return ""
    }
  }
}

final class TeaCache<FloatType: TensorNumeric & BinaryFloatingPoint> {
  private let modelVersion: ModelVersion
  private let coefficients: (Float, Float, Float, Float, Float)
  private let threshold: Float
  private let steps: ClosedRange<Int>
  private let maxSkipSteps: Int
  private let reducedModel: ModelBuilderOrModel
  private let inferModel: ModelBuilderOrModel?
  private let referenceImageCount: Int
  private let referenceAudioCount: Int
  private var lastTs: [Int: [DynamicGraph.AnyTensor]]
  private var accumulatedRelL1Distances: [Int: Float]
  private var lastResiduals: [Int: DynamicGraph.AnyTensor]
  private var skipSteps: [Int: Int]
  private var reducedModelParameterShared: Bool
  private var inferModelParameterShared: Bool

  public init(
    modelVersion: ModelVersion, coefficients: (Float, Float, Float, Float, Float), threshold: Float,
    steps: ClosedRange<Int>, maxSkipSteps: Int, reducedModel: ModelBuilderOrModel,
    inferModel: ModelBuilderOrModel? = nil,
    referenceImageCount: Int = 0, referenceAudioCount: Int = 0
  ) {
    self.modelVersion = modelVersion
    self.coefficients = coefficients
    self.threshold = threshold
    self.steps = steps
    self.maxSkipSteps = max(1, maxSkipSteps)
    self.reducedModel = reducedModel
    self.inferModel = inferModel
    self.referenceImageCount = referenceImageCount
    self.referenceAudioCount = referenceAudioCount
    lastTs = [Int: [DynamicGraph.AnyTensor]]()
    accumulatedRelL1Distances = [Int: Float]()
    lastResiduals = [Int: DynamicGraph.AnyTensor]()
    reducedModelParameterShared = false
    inferModelParameterShared = false
    skipSteps = [Int: Int]()
  }

  func loadModels(
    from store: DynamicGraph.Store, _ body: (ModelBuilderOrModel, DynamicGraph.Store) -> Void
  ) {
    switch modelVersion {
    case .minimaxH3:
      // The first block is independent of the retained tail; the reduced head shares its weights.
      if let inferModel {
        body(inferModel, store)
      }
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
      .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3,
      .seedvr2_3b, .seedvr2_7b, .longcatVideoAvatar1_5:
      break
    }
  }

  public func shouldUseCacheForTimeEmbedding<T: TensorNumeric & BinaryFloatingPoint>(
    _ t: [DynamicGraph.AnyTensor], model: ModelBuilderOrModel, step: Int, marker: Int,
    of: T.Type = T.self
  )
    -> Bool
  {
    var t = t
    switch modelVersion {
    case .minimaxH3:
      // H3 receives the already-computed first-block signals, not inputs to the infer model.
      guard let lastT = lastTs[marker], steps.contains(step),
        skipSteps[marker, default: 0] < maxSkipSteps,
        lastT.count == t.count, zip(t, lastT).allSatisfy({ $0.shape == $1.shape })
      else {
        lastTs[marker] = t
        skipSteps[marker] = 0
        return false
      }
      // Compare each modality against the last *full* step, not the last cache hit. Taking the
      // maximum prevents the much larger video stream from hiding a changing audio stream.
      var distance: Float = 0
      for (tensor, lastT) in zip(t, lastT) {
        let current = tensor.as(of: T.self)
        let previous = lastT.as(of: T.self)
        let axes = Array(0..<current.shape.count)
        // Use synchronous raw-tensor readback, as below. Graph copies run on the sampler's stream.
        let numerator = Float(
          Functional.abs(current - previous).reduced(.mean, axis: axes).rawValue.toCPU()
            .reshaped(.C(1))[0])
        let denominator = Float(
          Functional.abs(previous).reduced(.mean, axis: axes).rawValue.toCPU().reshaped(.C(1))[0])
        guard numerator.isFinite && denominator.isFinite else {
          lastTs[marker] = t
          skipSteps[marker] = 0
          return false
        }
        distance = max(distance, numerator / max(denominator, 1e-6))
      }
      if distance < threshold {
        return true
      }
      lastTs[marker] = t
      skipSteps[marker] = 0
      return false
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .wan21_14b, .wan21_1_3b, .hiDreamI1,
      .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b,
      .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b,
      .longcatVideoAvatar1_5:
      guard inferModel == nil else { fatalError() }
    case .hunyuanVideo:
      if inferModel != nil {
        t = [t[0]] + Array(t[(4 + 6)..<(6 + 6)])  // context chunks is before x chunks. We need x chunks.
      }
    case .flux1:
      if inferModel != nil {
        if referenceImageCount > 0 {
          t = [t[0]] + Array(t[(4 + 6)..<(6 + 6)])  // context chunks is before x chunks. We need x chunks.
        } else {
          t = [t[0]] + Array(t[(3 + 6)..<(5 + 6)])  // context chunks is before x chunks. We need x chunks.
        }
      }
    }
    if let inferModel = inferModel {
      if !inferModelParameterShared {
        inferModel.unwrapped.parameters.share(from: model.unwrapped.parameters) { name, _ in
          return .continue(name)
        }
        inferModelParameterShared = true
      }
      t = [inferModel(inputs: t[0], Array(t.dropFirst()))[0]]
    }
    guard let lastT = lastTs[marker], steps.contains(step),
      skipSteps[marker, default: 0] < maxSkipSteps
    else {
      lastTs[marker] = t
      accumulatedRelL1Distances[marker] = 0
      skipSteps[marker] = 0
      return false
    }
    var totalR1: Float = 0
    var totalR2: Float = 0
    for (t, lastT) in zip(t, lastT) {
      let tf32 = t.as(of: T.self)
      let lastTf32 = lastT.as(of: T.self)
      let shape = tf32.shape
      let r1 = Functional.abs(tf32 - lastTf32).reduced(.mean, axis: Array(0..<shape.count)).rawValue
        .toCPU()
      let r2 = Functional.abs(lastTf32).reduced(.mean, axis: Array(0..<shape.count)).rawValue
        .toCPU()
      totalR1 += Float(r1.reshaped(.C(1))[0])
      totalR2 += Float(r2.reshaped(.C(1))[0])
    }
    let r = totalR1 / totalR2
    let dist =
      coefficients.0 * r * r * r * r + coefficients.1 * r * r * r + coefficients.2 * r * r
      + coefficients.3 * r + coefficients.4
    var accumulatedRelL1Distance = accumulatedRelL1Distances[marker] ?? 0
    accumulatedRelL1Distance += dist
    var shouldUseCache = true
    if accumulatedRelL1Distance >= threshold {
      accumulatedRelL1Distance = 0
      shouldUseCache = false
      skipSteps[marker] = 0  // Reset skip steps in this case.
    }
    accumulatedRelL1Distances[marker] = accumulatedRelL1Distance
    lastTs[marker] = t
    return shouldUseCache
  }

  public func compile(model: ModelBuilderOrModel, inputs: [DynamicGraph.AnyTensor]) {
    switch modelVersion {
    case .minimaxH3:
      let firstInputs = Array(
        inputs.prefix(
          4 + referenceImageCount + referenceAudioCount + 18 + (referenceImageCount > 0 ? 6 : 0)
            + (referenceAudioCount > 0 ? 6 : 0)))
      inferModel?.compile(inputs: firstInputs)
      let text = inputs[2].as(of: Float.self)
      let placeholder = text.graph.variable(
        .GPU(0), .HWC(1, inputs[3].shape[1], text.shape[2]), of: Float.self)
      model.compile(inputs: [placeholder, inputs[3]] + inputs.dropFirst(firstInputs.count))
      reducedModel.compile(
        inputs: [placeholder, inputs[3]] + Array(inputs.suffix(4)) + [placeholder])
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .qwenImage, .qwenImage2_1, .wan22_5b,
      .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2, .ltx2_3,
      .seedvr2_3b, .seedvr2_7b, .hiDreamO1, .ideogram4, .krea2, .longcatVideoAvatar1_5:
      fatalError()
    case .hunyuanVideo:
      if let inferModel = inferModel {
        inferModel.compile(inputs: [inputs[0]] + Array(inputs[(4 + 6)..<(6 + 6)]))
      }
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], inputs[inputs.count - 2], inputs[inputs.count - 1],
      ])
    case .flux1:
      if let inferModel = inferModel {
        if referenceImageCount > 0 {
          inferModel.compile(inputs: [inputs[0]] + Array(inputs[(4 + 6)..<(6 + 6)]))
        } else {
          inferModel.compile(inputs: [inputs[0]] + Array(inputs[(3 + 6)..<(5 + 6)]))
        }
      }
      let shift: DynamicGraph.AnyTensor = inputs[
        (referenceImageCount > 0 ? 4 : 3) + 19 * 12 + 38 * 3]
      let scale: DynamicGraph.AnyTensor = inputs[
        (referenceImageCount > 0 ? 4 : 3) + 19 * 12 + 38 * 3 + 1]
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], shift, scale,
      ])
    case .hiDreamI1:
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], inputs[inputs.count - 2], inputs[inputs.count - 1],
      ])
    case .wan21_1_3b, .wan21_14b:
      reducedModel.compile(inputs: [
        inputs[0], inputs[0], inputs[inputs.count - 2], inputs[inputs.count - 1],
      ])
    }
  }

  public func cache(outputs: [DynamicGraph.AnyTensor], marker: Int) {
    lastResiduals[marker] = outputs[outputs.count - 1]
  }

  func cancel() {
    inferModel?.cancel()
    reducedModel.cancel()
  }

  func infer(inputs: [DynamicGraph.AnyTensor]) -> [DynamicGraph.AnyTensor] {
    guard let inferModel else { fatalError() }
    return inferModel(inputs: inputs[0], Array(inputs.dropFirst()))
  }

  public func callAsFunction(
    model: ModelBuilderOrModel, inputs firstInput: DynamicGraph.AnyTensor,
    _ restInputs: [DynamicGraph.AnyTensor], marker: Int
  ) -> [DynamicGraph.AnyTensor]? {
    guard let lastResidual = lastResiduals[marker] else {
      return nil
    }
    if !reducedModelParameterShared {
      reducedModel.unwrapped.parameters.share(from: model.unwrapped.parameters) { name, _ in
        return .continue(name)
      }
      reducedModelParameterShared = true
    }
    let inputs: [DynamicGraph.AnyTensor]
    switch modelVersion {
    case .minimaxH3:
      inputs = [restInputs[0]] + Array(restInputs.suffix(4)) + [lastResidual]
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
      .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2,
      .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3, .seedvr2_3b,
      .seedvr2_7b, .longcatVideoAvatar1_5:
      inputs = [lastResidual, restInputs[restInputs.count - 2], restInputs[restInputs.count - 1]]
    case .flux1:
      let shift: DynamicGraph.AnyTensor = restInputs[
        (referenceImageCount > 0 ? 3 : 2) + 19 * 12 + 38 * 3]
      let scale: DynamicGraph.AnyTensor = restInputs[
        (referenceImageCount > 0 ? 3 : 2) + 19 * 12 + 38 * 3 + 1]
      inputs = [lastResidual, shift, scale]
    }
    // Running the reduced model, this is skipped.
    skipSteps[marker, default: 0] += 1
    return reducedModel(inputs: firstInput, inputs)
  }
}
