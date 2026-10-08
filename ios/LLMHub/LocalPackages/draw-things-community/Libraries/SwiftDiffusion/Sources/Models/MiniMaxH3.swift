// MiniMax H3 joint video/audio denoiser graph definitions.
import DiffusionMappings
import Foundation
import NNC

public enum MiniMaxH3Configuration {
  public static let videoChannels = 24
  public static let audioChannels = 32
  public static let spatialCompression = 16
  public static let framesPerSecond = 24
  public static let audioSampleRate = 32_000
}

public func MiniMaxH3AudioHeight(videoLatentFrames: Int, latentWidth: Int) -> Int {
  precondition(latentWidth > 0)
  precondition(
    videoLatentFrames == 1 || videoLatentFrames == 2
      || (videoLatentFrames >= 7 && (videoLatentFrames - 2) % 5 == 0))
  let frames =
    videoLatentFrames == 1 ? 1 : (videoLatentFrames - 2) / 5 * 17 + 5
  let rows =
    2
    * Int(
      (Double(frames) / Double(MiniMaxH3Configuration.framesPerSecond) * 40).rounded())
  let rowSize =
    videoLatentFrames * latentWidth * MiniMaxH3Configuration.videoChannels
  let audioSize = rows * MiniMaxH3Configuration.audioChannels
  var height = (audioSize + rowSize - 1) / rowSize
  while height * rowSize % MiniMaxH3Configuration.audioChannels != 0 {
    height += 1
  }
  return height
}

func MiniMaxH3AudioSigma(forVideoSigma sigma: Float, audioShiftRatio: Float) -> Float {
  // The shared unshifted schedule cancels out; only audio shift / video shift is needed.
  if sigma == 0 || sigma == 1 { return sigma }
  return audioShiftRatio * sigma / (1 + (audioShiftRatio - 1) * sigma)
}

public func MiniMaxH3RotaryEmbedding(
  textLength: Int, audioLength: Int = 0, videoFrames: Int = 0, videoHeight: Int = 0,
  videoWidth: Int = 0, referenceImages: [(height: Int, width: Int, position: Float)] = [],
  referenceAudios: [(length: Int, position: Float)] = [],
  videoPosition: Float? = nil, visionTokenRanges: [Range<Int>] = []
) -> Tensor<Float> {
  precondition(audioLength % 2 == 0)
  precondition(videoHeight % 2 == 0)
  precondition(videoWidth % 2 == 0)
  let audioLatents = audioLength / 2
  let height = videoHeight / 2
  let width = videoWidth / 2
  let rowsPerFrame = height * width
  let videoLength = videoFrames * rowsPerFrame
  let conditionLength = referenceImages.reduce(0) { $0 + $1.height / 2 * ($1.width / 2) }
  let referenceAudioLength = referenceAudios.reduce(0) { $0 + $1.length }
  let visionLength = visionTokenRanges.reduce(0) { $0 + $1.count }
  let mediaPosition = videoPosition ?? Float(textLength)
  let sequenceLength =
    textLength + conditionLength + referenceAudioLength + audioLength + videoLength
  let squareRootArea = max(1, sqrt(Double(videoHeight * videoWidth)))
  let heightRatio = Double(videoHeight) / squareRootArea
  let widthRatio = Double(videoWidth) / squareRootArea
  let heightLeft = (1 - heightRatio) / 2
  let widthLeft = (1 - widthRatio) / 2
  // These match numpy.linspace(start, stop, count, endpoint=False), not torch.linspace.
  let heightGrid = (0..<height).map {
    Float((heightLeft + heightRatio * Double($0) / Double(height)) * 32)
  }
  let widthGrid = (0..<width).map {
    Float((widthLeft + widthRatio * Double($0) / Double(width)) * 32)
  }
  let frameRescale = 5.0 / 3.0
  let framesPerLatent = [1.0, 4.0, 4.0, 4.0, 4.0]
  var temporalGrid = [Float](repeating: 0, count: videoFrames)
  var temporalPosition = Double(mediaPosition)
  for frame in 0..<videoFrames {
    temporalGrid[frame] = Float(temporalPosition)
    temporalPosition += frameRescale * framesPerLatent[frame % framesPerLatent.count]
  }
  let inverseFrequencies = (0..<16).map {
    pow(10_000, -Double($0) * 2 / 32)
  }
  var rotary = Tensor<Float>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  rotary.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) -> Void in
    guard let fp32 = bytes.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
    func write(row: Int, position: (Float, Float, Float)) {
      let rowStart = row * 128
      for axis in 0..<3 {
        let value: Float = axis == 0 ? position.0 : (axis == 1 ? position.1 : position.2)
        for frequency in 0..<16 {
          let angle = Double(value) * inverseFrequencies[frequency]
          let offset = rowStart + (axis * 16 + frequency) * 2
          fp32[offset] = Float(cos(angle))
          fp32[offset + 1] = Float(sin(angle))
        }
      }
      for index in 96..<128 {
        fp32[rowStart + index] = index % 2 == 0 ? 1 : 0
      }
    }
    var textRow = 0
    var visionRow = textLength - visionLength + conditionLength + referenceAudioLength + audioLength
    for token in 0..<textLength {
      if visionTokenRanges.contains(where: { $0.contains(token) }) {
        write(row: visionRow, position: (Float(token), 0, 0))
        visionRow += 1
      } else {
        write(row: textRow, position: (Float(token), 0, 0))
        textRow += 1
      }
    }
    var referenceRow = textLength - visionLength
    for image in referenceImages {
      precondition(
        image.height > 0 && image.width > 0 && image.height % 2 == 0 && image.width % 2 == 0)
      let area = sqrt(Double(image.height * image.width))
      let hRatio = Double(image.height) / area
      let wRatio = Double(image.width) / area
      let height = image.height / 2
      let width = image.width / 2
      let hLeft: Double = (1 - hRatio) / 2
      let wLeft: Double = (1 - wRatio) / 2
      for y in 0..<height {
        let yPosition = Float((hLeft + hRatio * Double(y) / Double(height)) * 32)
        for x in 0..<width {
          let xPosition = Float((wLeft + wRatio * Double(x) / Double(width)) * 32)
          write(row: referenceRow, position: (image.position, yPosition, xPosition))
          referenceRow += 1
        }
      }
    }
    for audio in referenceAudios {
      precondition(audio.length > 0 && audio.length % 2 == 0)
      for channel in 0..<2 {
        let spatialPosition = channel == 0 ? widthGrid.first! : widthGrid.last!
        for index in 0..<(audio.length / 2) {
          write(row: referenceRow, position: (audio.position + Float(index), 0, spatialPosition))
          referenceRow += 1
        }
      }
    }
    let audioOffset = textLength - visionLength + conditionLength + referenceAudioLength
    for channel in 0..<(audioLatents > 0 ? 2 : 0) {
      let spatialPosition = channel == 0 ? widthGrid.first! : widthGrid.last!
      for index in 0..<audioLatents {
        write(
          row: audioOffset + channel * audioLatents + index,
          position: (mediaPosition + Float(index), 0, spatialPosition))
      }
    }
    let videoOffset = audioOffset + audioLength + visionLength
    for frame in 0..<videoFrames {
      for y in 0..<height {
        for x in 0..<width {
          write(
            row: videoOffset + frame * rowsPerFrame + y * width + x,
            position: (temporalGrid[frame], heightGrid[y], widthGrid[x]))
        }
      }
    }
  }
  return rotary
}

private func H3Attention(
  prefix: (String, String), hiddenSize: Int, sequenceLength: Int, isRoPEEnabled: Bool,
  scaleFactor: Int,
  usesFlashAttention: FlashAttentionLevel, segments: [Int] = [], startIndex: Int = 0,
  usesSolAttention: (eligibleForApproximation: Range<Int>, tau: Float)? = nil,
  name: String = ""
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = isRoPEEnabled ? Input() : nil
  let toQ = Dense(
    count: 7_168, noBias: true, name: name.isEmpty ? "q" : "\(name)_q")
  let toK = Dense(
    count: 7_168, noBias: true, name: name.isEmpty ? "k" : "\(name)_k")
  let toV = Dense(
    count: 7_168, noBias: true, name: name.isEmpty ? "v" : "\(name)_v")
  let normQ = RMSNorm(
    epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  let normK = RMSNorm(
    epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  toQ.startIndex = startIndex
  toK.startIndex = startIndex
  toV.startIndex = startIndex
  normQ.startIndex = startIndex
  normK.startIndex = startIndex
  let qProjected = toQ(x)
  let kProjected = toK(x)
  let vProjected = scaleFactor > 1 ? toV((1 / Float(scaleFactor)) * x) : toV(x)
  var q = normQ(
    qProjected.reshaped(.NHWC(1, sequenceLength, 56, 128)))
  var k = normK(
    kProjected.reshaped(.NHWC(1, sequenceLength, 56, 128)))
  let v = vProjected.reshaped(.NHWC(1, sequenceLength, 56, 128))
  if let rot {
    q = Functional.cmul(left: q, right: rot)
    k = Functional.cmul(left: k, right: rot)
  }
  q = (1 / Float(128).squareRoot()) * q
  let attention = ScaledDotProductAttention(
    scale: 1,
    flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
  let attentionOutput: Model.IO
  if let usesSolAttention {
    precondition(segments.isEmpty)
    attentionOutput = SolAttention(
      scale: 1, tau: usesSolAttention.tau,
      approximationRange: usesSolAttention.eligibleForApproximation, blockSize: 64,
      queryBlockSize: 64, localBlockRadius: 1)(q, k, v)
  } else if segments.isEmpty {
    attentionOutput = attention(q, k, v)
  } else {
    precondition(segments.reduce(0, +) == sequenceLength)
    var offset = 0
    let outputs = segments.map { length in
      let inputs = [q, k, v].map {
        $0.reshaped(
          [1, length, 56, 128], offset: [0, offset, 0, 0],
          strides: [sequenceLength * 7_168, 7_168, 128, 1]
        ).contiguous()
      }
      offset += length
      return attention(inputs)
    }
    attentionOutput = Concat(axis: 1)(outputs)
  }
  let attended = attentionOutput.reshaped([1, sequenceLength, 7_168])
  let out = Dense(
    count: hiddenSize, noBias: true, name: name.isEmpty ? "o" : "\(name)_o")
  out.startIndex = startIndex
  let projected: Model.IO
  if scaleFactor > 1 {
    projected = Float(scaleFactor) * out(attended).to(.Float32)
  } else {
    projected = out(attended).to(.Float32)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["\(prefix.1).to_q.weight"] = ModelWeightElement(
        [toQ.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 56, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.1).to_k.weight"] = ModelWeightElement(
        [toK.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 56, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.1).to_v.weight"] = [toV.weight.name]
      mapping["\(prefix.1).to_out.0.weight"] = [out.weight.name]
      mapping["\(prefix.1).norm_q.weight"] = ModelWeightElement(
        [normQ.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.1).norm_k.weight"] = ModelWeightElement(
        [normK.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
    case .generativeModels:
      mapping["\(prefix.0).qkv_proj.weight"] = ModelWeightElement(
        [toQ.weight.name, toK.weight.name, toV.weight.name],
        interleavedIndices: isRoPEEnabled ? [0, 1] : [],
        numberOfHeads: 56, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.0).out_proj.weight"] = [out.weight.name]
      mapping["\(prefix.0).q_norm.weight"] = ModelWeightElement(
        [normQ.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.0).k_norm.weight"] = ModelWeightElement(
        [normK.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
    }
    return mapping
  }
  if let rot {
    return (mapper, Model([x, rot], [projected]))
  }
  return (mapper, Model([x], [projected]))
}

private func H3SwiGLU(
  prefix: (String, String), hiddenSize: Int, scaleFactor: Int? = nil, startIndex: Int = 0,
  name: String = ""
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let up = Dense(
    count: 14_336, noBias: true, name: name.isEmpty ? "up" : "\(name)_up")
  let gate = Dense(
    count: 14_336, noBias: true, name: name.isEmpty ? "gate" : "\(name)_gate")
  let down = Dense(
    count: hiddenSize, noBias: true, name: name.isEmpty ? "down" : "\(name)_down")
  up.startIndex = startIndex
  gate.startIndex = startIndex
  down.startIndex = startIndex
  let upOutput = scaleFactor.map { up((1 / Float($0)) * x) } ?? up(x)
  let out = down(Functional.swishMul(value: upOutput, gate: gate(x))).to(.Float32)
  let mapper: ModelWeightMapper = { format in
    switch format {
    case .diffusers:
      return [
        "\(prefix.1).net.0.proj.weight": [up.weight.name, gate.weight.name],
        "\(prefix.1).net.2.weight": [down.weight.name],
      ]
    case .generativeModels:
      return [
        "\(prefix.0).fc1.weight": [gate.weight.name, up.weight.name],
        "\(prefix.0).fc2.weight": [down.weight.name],
      ]
    }
  }
  return (mapper, Model([x], [out]))
}

private func H3TransformerBlock(
  prefix: (String, String), hiddenSize: Int, textLength: Int, audioLength: Int, videoLength: Int,
  conditionLength: Int, referenceAudioLength: Int,
  usesFlashAttention: FlashAttentionLevel, scaleFactor: Int?, startIndex: Int,
  usesSolAttention: (eligibleForApproximation: Range<Int>, tau: Float)?
) -> (ModelWeightMapper, Model) {
  let sequenceLength =
    textLength + conditionLength + referenceAudioLength + audioLength + videoLength
  let x = Input()
  let rot = Input()
  let modulationCount = 3 + (conditionLength > 0 ? 1 : 0) + (referenceAudioLength > 0 ? 1 : 0)
  let modulations = (0..<(6 * modulationCount)).map { _ in Input() }
  // Modulation order: video, text, audio, then optional image and audio references.
  var spans: [(range: Range<Int>, modality: Int)] = [(0..<textLength, 1)]
  if conditionLength > 0 {
    spans.append((textLength..<(textLength + conditionLength), 3))
  }
  let audioStart = textLength + conditionLength + referenceAudioLength
  if referenceAudioLength > 0 {
    spans.append(((textLength + conditionLength)..<audioStart, conditionLength > 0 ? 4 : 3))
  }
  spans += [
    (audioStart..<(audioStart + audioLength), 2),
    ((audioStart + audioLength)..<sequenceLength, 0),
  ]
  spans.removeAll { $0.range.isEmpty }
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  norm1.startIndex = startIndex
  let (attentionMapper, attention) = H3Attention(
    prefix: ("\(prefix.0).attn", "\(prefix.1).attn"), hiddenSize: hiddenSize,
    sequenceLength: sequenceLength, isRoPEEnabled: true, scaleFactor: 8,
    usesFlashAttention: usesFlashAttention, startIndex: startIndex,
    usesSolAttention: usesSolAttention)
  let normed1 = norm1(x).to(.Float16)
  let attentionInputs = spans.map { span in
    let modality = span.modality
    let shift = modulations[modality]
    let scale = 1 + modulations[modulationCount + modality]
    return
      (normed1.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let attentionInput = Concat(axis: 1)(attentionInputs)
  let attentionOutput = attention(attentionInput, rot)
  let gatedAttention = spans.map { span in
    return modulations[2 * modulationCount + span.modality].to(.Float32)
      .* attentionOutput.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  var out =
    x
    + Concat(axis: 1)(gatedAttention)
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  norm2.startIndex = startIndex
  let (feedForwardMapper, feedForward) = H3SwiGLU(
    prefix: ("\(prefix.0).mlp", "\(prefix.1).ff"), hiddenSize: hiddenSize, scaleFactor: scaleFactor,
    startIndex: startIndex)
  let normed2 = norm2(out).to(.Float16)
  let feedForwardInputs = spans.map { span in
    let modality = span.modality
    let shift = modulations[3 * modulationCount + modality]
    let scale = 1 + modulations[4 * modulationCount + modality]
    return
      (normed2.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let feedForwardInput = Concat(axis: 1)(feedForwardInputs)
  let rawFeedForwardOutput = feedForward(feedForwardInput)
  let gatedFeedForward = spans.map { span in
    return modulations[5 * modulationCount + span.modality].to(.Float32)
      .* rawFeedForwardOutput.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  let feedForwardOutput = Concat(axis: 1)(gatedFeedForward)
  if let scaleFactor {
    out = out + Float(scaleFactor) * feedForwardOutput
  } else {
    out = out + feedForwardOutput
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = attentionMapper(format)
    mapping.merge(feedForwardMapper(format)) { v, _ in v }
    switch format {
    case .diffusers:
      mapping["\(prefix.1).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.1).norm2.weight"] = [norm2.weight.name]
    case .generativeModels:
      mapping["\(prefix.0).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.0).norm2.weight"] = [norm2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + modulations, [out]))
}

private func H3TokenRefinerBlock(
  prefix: (String, String), hiddenSize: Int, sequenceLength: Int,
  usesFlashAttention: FlashAttentionLevel, segments: [Int]
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm1")
  let (attentionMapper, attention) = H3Attention(
    prefix: ("\(prefix.0).attn", "\(prefix.1).attn"), hiddenSize: hiddenSize,
    sequenceLength: sequenceLength, isRoPEEnabled: false, scaleFactor: 1,
    usesFlashAttention: usesFlashAttention, segments: segments, name: "refiner")
  let attentionInput = norm1(x).to(.Float16)
  let attentionOutput = attention(attentionInput)
  var out = x + attentionOutput
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm2")
  let (feedForwardMapper, feedForward) = H3SwiGLU(
    prefix: ("\(prefix.0).mlp", "\(prefix.1).ff"), hiddenSize: hiddenSize, name: "refiner")
  out = out + feedForward(norm2(out).to(.Float16))
  let mapper: ModelWeightMapper = { format in
    var mapping = attentionMapper(format)
    mapping.merge(feedForwardMapper(format)) { v, _ in v }
    switch format {
    case .diffusers:
      mapping["\(prefix.1).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.1).norm2.weight"] = [norm2.weight.name]
    case .generativeModels:
      mapping["\(prefix.0).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.0).norm2.weight"] = [norm2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func H3TokenRefiner(
  prefix: String, hiddenSize: Int, sequenceLength: Int, usesFlashAttention: FlashAttentionLevel,
  segments: [Int]
) -> (ModelWeightMapper, Model) {
  let x = Input()
  var out: Model.IO = x
  var mappers = [ModelWeightMapper]()
  for i in 0..<2 {
    let (mapper, block) = H3TokenRefinerBlock(
      prefix: ("\(prefix).blocks.\(i)", "\(prefix).refiner_blocks.\(i)"), hiddenSize: hiddenSize,
      sequenceLength: sequenceLength,
      usesFlashAttention: usesFlashAttention, segments: segments)
    out = block(out)
    mappers.append(mapper)
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_final_norm")
  out = norm(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["\(prefix).final_norm.weight"] = [norm.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func H3TimestepEmbedding(prefix: String, hiddenSize: Int) -> (ModelWeightMapper, Model) {
  let frequencies = Input()
  let linear1 = Dense(count: hiddenSize, name: "time_embedder_linear_1")
  let linear2 = Dense(count: 2_688, name: "time_embedder_linear_2")
  let out = linear2(linear1(frequencies).swish())
  let mapper: ModelWeightMapper = { format in
    switch format {
    case .diffusers:
      return [
        "\(prefix).linear_1.weight": [linear1.weight.name],
        "\(prefix).linear_1.bias": [linear1.bias.name],
        "\(prefix).linear_2.weight": [linear2.weight.name],
        "\(prefix).linear_2.bias": [linear2.bias.name],
      ]
    case .generativeModels:
      return [
        "\(prefix).proj_in.weight": [linear1.weight.name],
        "\(prefix).proj_in.bias": [linear1.bias.name],
        "\(prefix).proj_out.weight": [linear2.weight.name],
        "\(prefix).proj_out.bias": [linear2.bias.name],
      ]
    }
  }
  return (mapper, Model([frequencies], [out]))
}

public func MiniMaxH3Fixed(
  timesteps: Int, hiddenSize: Int, layers: Int, textLength: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, referenceImageCount: Int = 0,
  referenceAudioCount: Int = 0,
  visionLength: Int = 0
) -> (ModelWeightMapper, Model) {
  precondition(timesteps > 0 && hiddenSize > 0 && layers > 0)
  precondition(textLength.0 >= 0 && textLength.1 > 0)
  let text = Input()
  let referenceImages = (0..<referenceImageCount).map { _ in Input() }
  let referenceAudios = (0..<referenceAudioCount).map { _ in Input() }
  let paddedLength = max(textLength.0, textLength.1)
  let contextInput: Model.IO
  if textLength.0 > 0 {
    // Remove CFG padding before refinement; vision rows follow the padded text in each branch.
    let branches = [textLength.0, textLength.1].enumerated().map { batch, length in
      var branch = text.reshaped(
        [1, length - visionLength, 5_120], offset: [batch, 0, 0],
        strides: [paddedLength * 5_120, 5_120, 1]
      ).contiguous()
      if visionLength > 0 {
        branch = Functional.concat(
          axis: 1, branch,
          text.reshaped(
            [1, visionLength, 5_120], offset: [batch, paddedLength - visionLength, 0],
            strides: [paddedLength * 5_120, 5_120, 1]
          ).contiguous())
      }
      return branch
    }
    contextInput = Concat(axis: 1)(branches)
  } else {
    contextInput = text
  }
  let textInput = Dense(count: hiddenSize, name: "context_embedder")
  let (refinerMapper, refiner) = H3TokenRefiner(
    prefix: "token_refiner", hiddenSize: hiddenSize, sequenceLength: textLength.0 + textLength.1,
    usesFlashAttention: usesFlashAttention,
    segments: textLength.0 > 0 ? [textLength.0, textLength.1] : [])
  var mappers = [refinerMapper]
  // UNetFixedEncoder scales the text bias by 1/4 to match the pre-scaled input.
  let textProjected = 4 * textInput(0.25 * contextInput).to(.Float32)
  var context = refiner(textProjected)
  if textLength.0 > 0 {
    let refined = context
    var offset = 0
    let branches = [textLength.0, textLength.1].map { length in
      var branch = refined.reshaped(
        [1, length - visionLength, hiddenSize], offset: [0, offset, 0],
        strides: [(textLength.0 + textLength.1) * hiddenSize, hiddenSize, 1]
      ).contiguous()
      if length < paddedLength {
        branch = branch.padded(.zero, begin: [0, 0, 0], end: [0, paddedLength - length, 0])
      }
      if visionLength > 0 {
        branch = Functional.concat(
          axis: 1, branch,
          refined.reshaped(
            [1, visionLength, hiddenSize], offset: [0, offset + length - visionLength, 0],
            strides: [(textLength.0 + textLength.1) * hiddenSize, hiddenSize, 1]
          ).contiguous())
      }
      offset += length
      return branch
    }
    context = Concat(axis: 0)(branches)
  }
  let timestepFrequencies = Input()
  let (timeMapper, timeEmbedding) = H3TimestepEmbedding(
    prefix: "time_embedder", hiddenSize: hiddenSize)
  mappers.append(timeMapper)
  let activatedTimestep = timeEmbedding(timestepFrequencies).swish()
  let timestepCount = 2 + (referenceImages.isEmpty ? 0 : 1) + (referenceAudios.isEmpty ? 0 : 1)
  var outputs = [context]
  if !referenceImages.isEmpty {
    let xEmbedder = Convolution(
      groups: 1, filters: hiddenSize, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "proj_in")
    outputs += referenceImages.map { xEmbedder($0) }
    mappers.append { format in
      switch format {
      case .diffusers:
        return ["proj_in.weight": [xEmbedder.weight.name], "proj_in.bias": [xEmbedder.bias.name]]
      case .generativeModels:
        return [
          "video_patch_proj.weight": [xEmbedder.weight.name],
          "video_patch_proj.bias": [xEmbedder.bias.name],
        ]
      }
    }
  }
  if !referenceAudios.isEmpty {
    let audioInput = Dense(count: hiddenSize, name: "audio_proj_in")
    outputs += referenceAudios.map { audioInput($0) }
    mappers.append { format in
      switch format {
      case .diffusers:
        return [
          "audio_proj_in.weight": [audioInput.weight.name],
          "audio_proj_in.bias": [audioInput.bias.name],
        ]
      case .generativeModels:
        return [
          "audio_patch_proj.weight": [audioInput.weight.name],
          "audio_patch_proj.bias": [audioInput.bias.name],
        ]
      }
    }
  }
  for i in 0..<layers {
    var projections = [Model]()
    for chunk in 0..<6 {
      var conditions = [Model.IO]()
      for modality in 0..<3 {
        let projection = Dense(count: hiddenSize, name: "adaln_\(chunk)_\(modality)")
        projections.append(projection)
        let projected = projection(activatedTimestep)
        let timestep = modality == 2 ? 1 : 0
        outputs.append(
          projected.reshaped(
            [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
            strides: [timestepCount * hiddenSize, hiddenSize, 1]
          ).contiguous())
        if !referenceImages.isEmpty && modality == 0 {
          conditions.append(
            projected.reshaped(
              [timesteps, 1, hiddenSize], offset: [0, 2, 0],
              strides: [timestepCount * hiddenSize, hiddenSize, 1]
            ).contiguous())
        }
        if !referenceAudios.isEmpty && modality == 2 {
          conditions.append(
            projected.reshaped(
              [timesteps, 1, hiddenSize], offset: [0, timestepCount - 1, 0],
              strides: [timestepCount * hiddenSize, hiddenSize, 1]
            ).contiguous())
        }
      }
      outputs += conditions
    }
    mappers.append { format in
      // Checkpoints store [modality, modulation, hidden], not construction order.
      let ordered = (0..<3).flatMap { modality in
        (0..<6).map { projections[$0 * 3 + modality] }
      }
      switch format {
      case .diffusers:
        return [
          "transformer_blocks.\(i).adaln_proj.linear.weight": ModelWeightElement(
            ordered.map { $0.weight.name }),
          "transformer_blocks.\(i).adaln_proj.linear.bias": ModelWeightElement(
            ordered.map { $0.bias.name }),
        ]
      case .generativeModels:
        return [
          "blocks.\(i).adaln_proj.linear.weight": ModelWeightElement(
            ordered.map { $0.weight.name }),
          "blocks.\(i).adaln_proj.linear.bias": ModelWeightElement(ordered.map { $0.bias.name }),
        ]
      }
    }
  }
  let outputShift = Dense(count: hiddenSize, name: "norm_out_shift")
  let outputScale = Dense(count: hiddenSize, name: "norm_out_scale")
  let outputShifts = outputShift(activatedTimestep)
  let outputScales = outputScale(activatedTimestep)
  for timestep in 0..<2 {
    outputs.append(
      outputShifts.reshaped(
        [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
        strides: [timestepCount * hiddenSize, hiddenSize, 1]
      ).contiguous())
    outputs.append(
      outputScales.reshaped(
        [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
        strides: [timestepCount * hiddenSize, hiddenSize, 1]
      ).contiguous())
  }
  precondition(
    outputs.count == layers
      * (18 + (referenceImages.isEmpty ? 0 : 6) + (referenceAudios.isEmpty ? 0 : 6)) + 5
      + referenceImages.count + referenceAudios.count)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .diffusers:
      mapping["context_embedder.weight"] = [textInput.weight.name]
      mapping["context_embedder.bias"] = [textInput.bias.name]
      mapping["norm_out.linear.weight"] = [outputShift.weight.name, outputScale.weight.name]
      mapping["norm_out.linear.bias"] = [outputShift.bias.name, outputScale.bias.name]
    case .generativeModels:
      mapping["condition_proj.weight"] = [textInput.weight.name]
      mapping["condition_proj.bias"] = [textInput.bias.name]
      mapping["final_layer.adaln_proj.linear.weight"] = [
        outputShift.weight.name, outputScale.weight.name,
      ]
      mapping["final_layer.adaln_proj.linear.bias"] = [
        outputShift.bias.name, outputScale.bias.name,
      ]
    }
    return mapping
  }
  return (mapper, Model([text, timestepFrequencies] + referenceImages + referenceAudios, outputs))
}

/// Sol is opt-in. Global transformer layers
/// 0, 1, 33, 34 and 35 remain dense; only generated-video interactions may be approximated.
/// A nil configuration uses SDPA; an empty eligible range uses all-exact Sol attention.
public func MiniMaxH3(
  hiddenSize: Int, layers: Int, startLayer: Int = 0, textLength: Int, audioLength: Int,
  videoFrames: Int,
  videoHeight: Int, videoWidth: Int, usesFlashAttention: FlashAttentionLevel,
  usesSolAttention: (eligibleForApproximation: Range<Int>, tau: Float)? = nil,
  referenceImageSizes: [(height: Int, width: Int)] = [], referenceAudioLengths: [Int] = [],
  visionLength: Int = 0,
  outputResidual: Bool = false, inputResidual: Bool = false
) -> (ModelWeightMapper, Model) {
  precondition(hiddenSize > 0 && layers >= 0 && startLayer >= 0)
  precondition(startLayer > 0 || layers > 0)
  precondition(!inputResidual || (startLayer > 0 && layers == 0))
  precondition(videoHeight % 2 == 0 && videoWidth % 2 == 0)
  let video = Input()
  let audio = Input()
  let contextRows = Input()
  let rot = Input()
  let referenceImages = referenceImageSizes.map { _ in Input() }
  let referenceAudios = referenceAudioLengths.map { _ in Input() }
  let referenceAudioLength = referenceAudioLengths.reduce(0, +)
  let perLayerConditions =
    18 + (referenceImages.isEmpty ? 0 : 6) + (referenceAudios.isEmpty ? 0 : 6)
  let fixedConditions = (0..<(layers * perLayerConditions)).map { _ in Input() }
  let videoLength = videoFrames * videoHeight / 2 * videoWidth / 2
  let conditionLength = referenceImageSizes.reduce(0) { $0 + $1.height / 2 * ($1.width / 2) }
  let sequenceLength =
    textLength + conditionLength + referenceAudioLength + audioLength + videoLength

  if let usesSolAttention, !usesSolAttention.eligibleForApproximation.isEmpty {
    precondition(
      usesSolAttention.eligibleForApproximation.lowerBound >= sequenceLength - videoLength)
    precondition(usesSolAttention.eligibleForApproximation.upperBound <= sequenceLength)
  }

  var out: Model.IO
  var inputs: [Model.IO]
  let outputType: Model.IO
  var mappers = [ModelWeightMapper]()
  if startLayer == 0 {
    let xEmbedder = Convolution(
      groups: 1, filters: hiddenSize, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "proj_in")
    let audioInput = Dense(count: hiddenSize, name: "audio_proj_in")
    let audioRows = audioInput(audio).to(.Float32)
    let videoRows = xEmbedder(video).reshaped(.HWC(1, videoLength, hiddenSize)).to(.Float32)
    let textRows: Model.IO
    if visionLength > 0 {
      textRows = contextRows.reshaped(
        [1, textLength - visionLength, hiddenSize],
        strides: [textLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
    } else {
      textRows = contextRows
    }
    // Each modulation group stays contiguous throughout all transformer blocks.
    var rows = [textRows]
    for (image, size) in zip(referenceImages, referenceImageSizes) {
      rows.append(
        image.reshaped(.HWC(1, size.height / 2 * (size.width / 2), hiddenSize)).to(.Float32))
    }
    rows += referenceAudios.map { $0.to(.Float32) }
    rows.append(audioRows)
    if visionLength > 0 {
      rows.append(
        contextRows.reshaped(
          [1, visionLength, hiddenSize], offset: [0, textLength - visionLength, 0],
          strides: [textLength * hiddenSize, hiddenSize, 1]
        ).contiguous())
    }
    rows.append(videoRows)
    out = Concat(axis: 1)(rows)
    inputs = [video, audio, contextRows, rot] + referenceImages + referenceAudios
    outputType = video
    mappers.append { format in
      switch format {
      case .diffusers:
        return [
          "proj_in.weight": [xEmbedder.weight.name],
          "proj_in.bias": [xEmbedder.bias.name],
          "audio_proj_in.weight": [audioInput.weight.name],
          "audio_proj_in.bias": [audioInput.bias.name],
        ]
      case .generativeModels:
        return [
          "video_patch_proj.weight": [xEmbedder.weight.name],
          "video_patch_proj.bias": [xEmbedder.bias.name],
          "audio_patch_proj.weight": [audioInput.weight.name],
          "audio_patch_proj.bias": [audioInput.bias.name],
        ]
      }
    }
  } else {
    let hiddenState = Input()
    out = hiddenState
    inputs = [hiddenState, rot]
    outputType = rot
  }
  let embedded = out
  for layer in startLayer..<(startLayer + layers) {
    // Keep FFN outliers in FP16 range; the residual branch restores the factor in FP32.
    let scaleFactor: Int
    switch layer {
    case 0, 4, 11, 31, 32:
      scaleFactor = 4
    case 1, 7, 27, 33, 34, 35, 37, 38, 46:
      scaleFactor = 8
    case 18, 30, 40, 41, 42:
      scaleFactor = 16
    case 47:
      scaleFactor = 32
    case 13, 36, 43, 44, 48, 49:
      scaleFactor = 64
    case 39:
      scaleFactor = 512
    case 45:
      scaleFactor = 1_024
    default:
      scaleFactor = 2
    }
    let (mapper, block) = H3TransformerBlock(
      prefix: ("blocks.\(layer)", "transformer_blocks.\(layer)"), hiddenSize: hiddenSize,
      textLength: textLength - visionLength, audioLength: audioLength,
      videoLength: visionLength + videoLength, conditionLength: conditionLength,
      referenceAudioLength: referenceAudioLength,
      usesFlashAttention: usesFlashAttention,
      scaleFactor: scaleFactor, startIndex: startLayer,
      usesSolAttention: [0, 1, 33, 34, 35].contains(layer) ? nil : usesSolAttention)
    mappers.append(mapper)
    let conditionOffset = (layer - startLayer) * perLayerConditions
    out =
      block(
        [out, rot]
          + Array(
            fixedConditions[
              conditionOffset..<(conditionOffset + perLayerConditions)
            ]))[0]
  }

  inputs += fixedConditions
  if startLayer == 0 && outputResidual {
    precondition(layers == 1)
    let mapper: ModelWeightMapper = { format in
      var mapping = ModelWeightMapping()
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      return mapping
    }
    // Run the input projections and block 0 once; the tail consumes this exact hidden state.
    return (mapper, Model(inputs, [out, out - embedded]))
  }
  let videoShift = Input()
  let videoScale = Input()
  let audioShift = Input()
  let audioScale = Input()
  inputs += [videoShift, videoScale, audioShift, audioScale]
  var residualOutputs = [Model.IO]()
  if inputResidual {
    let residual = Input()
    inputs.append(residual)
    out = out + residual
  }
  if outputResidual {
    residualOutputs = [out - embedded]
  }
  let outputNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm_out")
  let normalized = outputNorm(out)
  let audioOutRows =
    normalized.reshaped(
      [1, audioLength, hiddenSize],
      offset: [0, textLength - visionLength + conditionLength + referenceAudioLength, 0],
      strides: [sequenceLength * hiddenSize, hiddenSize, 1]
    ).contiguous() .* (1 + audioScale).to(.Float32) + audioShift.to(.Float32)
  let videoOutRows =
    normalized.reshaped(
      [1, videoLength, hiddenSize],
      offset: [0, textLength + conditionLength + referenceAudioLength + audioLength, 0],
      strides: [sequenceLength * hiddenSize, hiddenSize, 1]
    ).contiguous() .* (1 + videoScale).to(.Float32) + videoShift.to(.Float32)
  let videoOutput = Dense(count: 96, name: "proj_out")
  let audioOutput = Dense(count: 32, name: "audio_proj_out")
  let projectedVideo = -videoOutput(videoOutRows).reshaped([
    videoFrames, videoHeight / 2, videoWidth / 2, 24, 2, 2,
  ]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    videoFrames, videoHeight, videoWidth, 24,
  ])
  let projectedAudio = -audioOutput(audioOutRows)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .diffusers:
      mapping["proj_out.weight"] = [videoOutput.weight.name]
      mapping["proj_out.bias"] = [videoOutput.bias.name]
      mapping["audio_proj_out.weight"] = [audioOutput.weight.name]
      mapping["audio_proj_out.bias"] = [audioOutput.bias.name]
      mapping["norm_out.norm.weight"] = [outputNorm.weight.name]
    case .generativeModels:
      mapping["final_layer.video_out.weight"] = [videoOutput.weight.name]
      mapping["final_layer.video_out.bias"] = [videoOutput.bias.name]
      mapping["final_layer.audio_out.weight"] = [audioOutput.weight.name]
      mapping["final_layer.audio_out.bias"] = [audioOutput.bias.name]
      mapping["final_layer.norm.weight"] = [outputNorm.weight.name]
    }
    return mapping
  }
  return (
    mapper,
    Model(
      inputs,
      [projectedVideo.to(of: outputType), projectedAudio.to(of: startLayer == 0 ? audio : rot)]
        + residualOutputs)
  )
}

private func LoRAH3Attention(
  prefix: (String, String), hiddenSize: Int, layerIndex: Int,
  configuration: LoRANetworkConfiguration, sequenceLength: Int,
  isRoPEEnabled: Bool, scaleFactor: Int,
  usesFlashAttention: FlashAttentionLevel, segments: [Int] = [], startIndex: Int = 0,
  usesSolAttention: (eligibleForApproximation: Range<Int>, tau: Float)? = nil,
  name: String = ""
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = isRoPEEnabled ? Input() : nil
  let toQ = LoRADense(
    count: 7_168, configuration: configuration, noBias: true, index: layerIndex,
    startIndex: startIndex, name: name.isEmpty ? "q" : "\(name)_q")
  let toK = LoRADense(
    count: 7_168, configuration: configuration, noBias: true, index: layerIndex,
    startIndex: startIndex, name: name.isEmpty ? "k" : "\(name)_k")
  let toV = LoRADense(
    count: 7_168, configuration: configuration, noBias: true, index: layerIndex,
    startIndex: startIndex, name: name.isEmpty ? "v" : "\(name)_v")
  let normQ = RMSNorm(
    epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  let normK = RMSNorm(
    epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  normQ.startIndex = startIndex
  normK.startIndex = startIndex
  let qProjected = toQ(x)
  let kProjected = toK(x)
  let vProjected = scaleFactor > 1 ? toV((1 / Float(scaleFactor)) * x) : toV(x)
  var q = normQ(
    qProjected.reshaped(.NHWC(1, sequenceLength, 56, 128)))
  var k = normK(
    kProjected.reshaped(.NHWC(1, sequenceLength, 56, 128)))
  let v = vProjected.reshaped(.NHWC(1, sequenceLength, 56, 128))
  if let rot {
    q = Functional.cmul(left: q, right: rot)
    k = Functional.cmul(left: k, right: rot)
  }
  q = (1 / Float(128).squareRoot()) * q
  let attention = ScaledDotProductAttention(
    scale: 1,
    flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
  let attentionOutput: Model.IO
  if let usesSolAttention {
    precondition(segments.isEmpty)
    attentionOutput = SolAttention(
      scale: 1, tau: usesSolAttention.tau,
      approximationRange: usesSolAttention.eligibleForApproximation, blockSize: 64,
      queryBlockSize: 64, localBlockRadius: 1)(q, k, v)
  } else if segments.isEmpty {
    attentionOutput = attention(q, k, v)
  } else {
    precondition(segments.reduce(0, +) == sequenceLength)
    var offset = 0
    let outputs = segments.map { length in
      let inputs = [q, k, v].map {
        $0.reshaped(
          [1, length, 56, 128], offset: [0, offset, 0, 0],
          strides: [sequenceLength * 7_168, 7_168, 128, 1]
        ).contiguous()
      }
      offset += length
      return attention(inputs)
    }
    attentionOutput = Concat(axis: 1)(outputs)
  }
  let attended = attentionOutput.reshaped([1, sequenceLength, 7_168])
  let out = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    startIndex: startIndex, name: name.isEmpty ? "o" : "\(name)_o")
  let projected: Model.IO
  if scaleFactor > 1 {
    projected = Float(scaleFactor) * out(attended).to(.Float32)
  } else {
    projected = out(attended).to(.Float32)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["\(prefix.1).to_q.weight"] = ModelWeightElement(
        [toQ.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 56, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.1).to_k.weight"] = ModelWeightElement(
        [toK.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 56, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.1).to_v.weight"] = [toV.weight.name]
      mapping["\(prefix.1).to_out.0.weight"] = [out.weight.name]
      mapping["\(prefix.1).norm_q.weight"] = ModelWeightElement(
        [normQ.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.1).norm_k.weight"] = ModelWeightElement(
        [normK.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
    case .generativeModels:
      mapping["\(prefix.0).qkv_proj.weight"] = ModelWeightElement(
        [toQ.weight.name, toK.weight.name, toV.weight.name],
        interleavedIndices: isRoPEEnabled ? [0, 1] : [],
        numberOfHeads: 56, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.0).out_proj.weight"] = [out.weight.name]
      mapping["\(prefix.0).q_norm.weight"] = ModelWeightElement(
        [normQ.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
      mapping["\(prefix.0).k_norm.weight"] = ModelWeightElement(
        [normK.weight.name], interleavedIndices: isRoPEEnabled ? [0] : [],
        numberOfHeads: 1, headDimension: 128, interleavedDimension: 96)
    }
    return mapping
  }
  if let rot {
    return (mapper, Model([x, rot], [projected]))
  }
  return (mapper, Model([x], [projected]))
}

private func LoRAH3SwiGLU(
  prefix: (String, String), hiddenSize: Int, layerIndex: Int,
  configuration: LoRANetworkConfiguration,
  scaleFactor: Int? = nil, startIndex: Int = 0, name: String = ""
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let up = LoRADense(
    count: 14_336, configuration: configuration, noBias: true, index: layerIndex,
    startIndex: startIndex, name: name.isEmpty ? "up" : "\(name)_up")
  let gate = LoRADense(
    count: 14_336, configuration: configuration, noBias: true, index: layerIndex,
    startIndex: startIndex, name: name.isEmpty ? "gate" : "\(name)_gate")
  let down = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    startIndex: startIndex, name: name.isEmpty ? "down" : "\(name)_down")
  let upOutput = scaleFactor.map { up((1 / Float($0)) * x) } ?? up(x)
  let out = down(Functional.swishMul(value: upOutput, gate: gate(x))).to(.Float32)
  let mapper: ModelWeightMapper = { format in
    switch format {
    case .diffusers:
      return [
        "\(prefix.1).net.0.proj.weight": [up.weight.name, gate.weight.name],
        "\(prefix.1).net.2.weight": [down.weight.name],
      ]
    case .generativeModels:
      return [
        "\(prefix.0).fc1.weight": [gate.weight.name, up.weight.name],
        "\(prefix.0).fc2.weight": [down.weight.name],
      ]
    }
  }
  return (mapper, Model([x], [out]))
}

private func LoRAH3TransformerBlock(
  prefix: (String, String), hiddenSize: Int, layerIndex: Int,
  configuration: LoRANetworkConfiguration, textLength: Int,
  audioLength: Int, videoLength: Int,
  conditionLength: Int, referenceAudioLength: Int,
  usesFlashAttention: FlashAttentionLevel, scaleFactor: Int?, startIndex: Int,
  usesSolAttention: (eligibleForApproximation: Range<Int>, tau: Float)?
) -> (ModelWeightMapper, Model) {
  let sequenceLength =
    textLength + conditionLength + referenceAudioLength + audioLength + videoLength
  let x = Input()
  let rot = Input()
  let modulationCount = 3 + (conditionLength > 0 ? 1 : 0) + (referenceAudioLength > 0 ? 1 : 0)
  let modulations = (0..<(6 * modulationCount)).map { _ in Input() }
  // Modulation order: video, text, audio, then optional image and audio references.
  var spans: [(range: Range<Int>, modality: Int)] = [(0..<textLength, 1)]
  if conditionLength > 0 {
    spans.append((textLength..<(textLength + conditionLength), 3))
  }
  let audioStart = textLength + conditionLength + referenceAudioLength
  if referenceAudioLength > 0 {
    spans.append(((textLength + conditionLength)..<audioStart, conditionLength > 0 ? 4 : 3))
  }
  spans += [
    (audioStart..<(audioStart + audioLength), 2),
    ((audioStart + audioLength)..<sequenceLength, 0),
  ]
  spans.removeAll { $0.range.isEmpty }
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  norm1.startIndex = startIndex
  let (attentionMapper, attention) = LoRAH3Attention(
    prefix: ("\(prefix.0).attn", "\(prefix.1).attn"), hiddenSize: hiddenSize,
    layerIndex: layerIndex, configuration: configuration,
    sequenceLength: sequenceLength, isRoPEEnabled: true, scaleFactor: 8,
    usesFlashAttention: usesFlashAttention, startIndex: startIndex,
    usesSolAttention: usesSolAttention)
  let normed1 = norm1(x).to(.Float16)
  let attentionInputs = spans.map { span in
    let modality = span.modality
    let shift = modulations[modality]
    let scale = 1 + modulations[modulationCount + modality]
    return
      (normed1.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let attentionInput = Concat(axis: 1)(attentionInputs)
  let attentionOutput = attention(attentionInput, rot)
  let gatedAttention = spans.map { span in
    return modulations[2 * modulationCount + span.modality].to(.Float32)
      .* attentionOutput.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  var out =
    x
    + Concat(axis: 1)(gatedAttention)
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  norm2.startIndex = startIndex
  let (feedForwardMapper, feedForward) = LoRAH3SwiGLU(
    prefix: ("\(prefix.0).mlp", "\(prefix.1).ff"), hiddenSize: hiddenSize, layerIndex: layerIndex,
    configuration: configuration,
    scaleFactor: scaleFactor, startIndex: startIndex)
  let normed2 = norm2(out).to(.Float16)
  let feedForwardInputs = spans.map { span in
    let modality = span.modality
    let shift = modulations[3 * modulationCount + modality]
    let scale = 1 + modulations[4 * modulationCount + modality]
    return
      (normed2.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
      .* scale + shift)
  }
  let feedForwardInput = Concat(axis: 1)(feedForwardInputs)
  let rawFeedForwardOutput = feedForward(feedForwardInput)
  let gatedFeedForward = spans.map { span in
    return modulations[5 * modulationCount + span.modality].to(.Float32)
      .* rawFeedForwardOutput.reshaped(
        [1, span.range.count, hiddenSize], offset: [0, span.range.lowerBound, 0],
        strides: [sequenceLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
  }
  let feedForwardOutput = Concat(axis: 1)(gatedFeedForward)
  if let scaleFactor {
    out = out + Float(scaleFactor) * feedForwardOutput
  } else {
    out = out + feedForwardOutput
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = attentionMapper(format)
    mapping.merge(feedForwardMapper(format)) { v, _ in v }
    switch format {
    case .diffusers:
      mapping["\(prefix.1).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.1).norm2.weight"] = [norm2.weight.name]
    case .generativeModels:
      mapping["\(prefix.0).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.0).norm2.weight"] = [norm2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + modulations, [out]))
}

private func LoRAH3TokenRefinerBlock(
  prefix: (String, String), hiddenSize: Int, layerIndex: Int,
  configuration: LoRANetworkConfiguration, sequenceLength: Int,
  usesFlashAttention: FlashAttentionLevel, segments: [Int]
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm1")
  let (attentionMapper, attention) = LoRAH3Attention(
    prefix: ("\(prefix.0).attn", "\(prefix.1).attn"), hiddenSize: hiddenSize,
    layerIndex: layerIndex, configuration: configuration,
    sequenceLength: sequenceLength, isRoPEEnabled: false, scaleFactor: 1,
    usesFlashAttention: usesFlashAttention, segments: segments, name: "refiner")
  let attentionInput = norm1(x).to(.Float16)
  let attentionOutput = attention(attentionInput)
  var out = x + attentionOutput
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_norm2")
  let (feedForwardMapper, feedForward) = LoRAH3SwiGLU(
    prefix: ("\(prefix.0).mlp", "\(prefix.1).ff"), hiddenSize: hiddenSize, layerIndex: layerIndex,
    configuration: configuration, name: "refiner")
  out = out + feedForward(norm2(out).to(.Float16))
  let mapper: ModelWeightMapper = { format in
    var mapping = attentionMapper(format)
    mapping.merge(feedForwardMapper(format)) { v, _ in v }
    switch format {
    case .diffusers:
      mapping["\(prefix.1).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.1).norm2.weight"] = [norm2.weight.name]
    case .generativeModels:
      mapping["\(prefix.0).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.0).norm2.weight"] = [norm2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func LoRAH3TokenRefiner(
  prefix: String, hiddenSize: Int, configuration: LoRANetworkConfiguration, sequenceLength: Int,
  usesFlashAttention: FlashAttentionLevel, segments: [Int]
) -> (ModelWeightMapper, Model) {
  let x = Input()
  var out: Model.IO = x
  var mappers = [ModelWeightMapper]()
  for layerIndex in 0..<2 {
    let (mapper, block) = LoRAH3TokenRefinerBlock(
      prefix: ("\(prefix).blocks.\(layerIndex)", "\(prefix).refiner_blocks.\(layerIndex)"),
      hiddenSize: hiddenSize, layerIndex: layerIndex, configuration: configuration,
      sequenceLength: sequenceLength,
      usesFlashAttention: usesFlashAttention, segments: segments)
    out = block(out)
    mappers.append(mapper)
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [2], name: "refiner_final_norm")
  out = norm(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["\(prefix).final_norm.weight"] = [norm.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func LoRAH3TimestepEmbedding(
  prefix: String, hiddenSize: Int, configuration: LoRANetworkConfiguration
) -> (
  ModelWeightMapper, Model
) {
  let layerIndex = 0
  let frequencies = Input()
  let linear1 = LoRADense(
    count: hiddenSize, configuration: configuration, index: layerIndex,
    name: "time_embedder_linear_1")
  let linear2 = LoRADense(
    count: 2_688, configuration: configuration, index: layerIndex, name: "time_embedder_linear_2")
  let out = linear2(linear1(frequencies).swish())
  let mapper: ModelWeightMapper = { format in
    switch format {
    case .diffusers:
      return [
        "\(prefix).linear_1.weight": [linear1.weight.name],
        "\(prefix).linear_1.bias": [linear1.bias.name],
        "\(prefix).linear_2.weight": [linear2.weight.name],
        "\(prefix).linear_2.bias": [linear2.bias.name],
      ]
    case .generativeModels:
      return [
        "\(prefix).proj_in.weight": [linear1.weight.name],
        "\(prefix).proj_in.bias": [linear1.bias.name],
        "\(prefix).proj_out.weight": [linear2.weight.name],
        "\(prefix).proj_out.bias": [linear2.bias.name],
      ]
    }
  }
  return (mapper, Model([frequencies], [out]))
}

public func LoRAMiniMaxH3Fixed(
  timesteps: Int, hiddenSize: Int, layers: Int, textLength: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, referenceImageCount: Int = 0,
  referenceAudioCount: Int = 0,
  visionLength: Int = 0, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let configuration = LoRAConfiguration
  let layerIndex = 0
  precondition(timesteps > 0 && hiddenSize > 0 && layers > 0)
  precondition(textLength.0 >= 0 && textLength.1 > 0)
  let text = Input()
  let referenceImages = (0..<referenceImageCount).map { _ in Input() }
  let referenceAudios = (0..<referenceAudioCount).map { _ in Input() }
  let paddedLength = max(textLength.0, textLength.1)
  let contextInput: Model.IO
  if textLength.0 > 0 {
    // Remove CFG padding before refinement; vision rows follow the padded text in each branch.
    let branches = [textLength.0, textLength.1].enumerated().map { batch, length in
      var branch = text.reshaped(
        [1, length - visionLength, 5_120], offset: [batch, 0, 0],
        strides: [paddedLength * 5_120, 5_120, 1]
      ).contiguous()
      if visionLength > 0 {
        branch = Functional.concat(
          axis: 1, branch,
          text.reshaped(
            [1, visionLength, 5_120], offset: [batch, paddedLength - visionLength, 0],
            strides: [paddedLength * 5_120, 5_120, 1]
          ).contiguous())
      }
      return branch
    }
    contextInput = Concat(axis: 1)(branches)
  } else {
    contextInput = text
  }
  let textInput = LoRADense(
    count: hiddenSize, configuration: configuration, index: layerIndex, name: "context_embedder")
  let (refinerMapper, refiner) = LoRAH3TokenRefiner(
    prefix: "token_refiner", hiddenSize: hiddenSize, configuration: configuration,
    sequenceLength: textLength.0 + textLength.1,
    usesFlashAttention: usesFlashAttention,
    segments: textLength.0 > 0 ? [textLength.0, textLength.1] : [])
  var mappers = [refinerMapper]
  // UNetFixedEncoder scales the text bias by 1/4 to match the pre-scaled input.
  let textProjected = 4 * textInput(0.25 * contextInput).to(.Float32)
  var context = refiner(textProjected)
  if textLength.0 > 0 {
    let refined = context
    var offset = 0
    let branches = [textLength.0, textLength.1].map { length in
      var branch = refined.reshaped(
        [1, length - visionLength, hiddenSize], offset: [0, offset, 0],
        strides: [(textLength.0 + textLength.1) * hiddenSize, hiddenSize, 1]
      ).contiguous()
      if length < paddedLength {
        branch = branch.padded(.zero, begin: [0, 0, 0], end: [0, paddedLength - length, 0])
      }
      if visionLength > 0 {
        branch = Functional.concat(
          axis: 1, branch,
          refined.reshaped(
            [1, visionLength, hiddenSize], offset: [0, offset + length - visionLength, 0],
            strides: [(textLength.0 + textLength.1) * hiddenSize, hiddenSize, 1]
          ).contiguous())
      }
      offset += length
      return branch
    }
    context = Concat(axis: 0)(branches)
  }
  let timestepFrequencies = Input()
  let (timeMapper, timeEmbedding) = LoRAH3TimestepEmbedding(
    prefix: "time_embedder", hiddenSize: hiddenSize, configuration: configuration)
  mappers.append(timeMapper)
  let activatedTimestep = timeEmbedding(timestepFrequencies).swish()
  let timestepCount = 2 + (referenceImages.isEmpty ? 0 : 1) + (referenceAudios.isEmpty ? 0 : 1)
  var outputs = [context]
  if !referenceImages.isEmpty {
    let xEmbedder = LoRAConvolution(
      groups: 1, filters: hiddenSize, filterSize: [2, 2], configuration: configuration,
      hint: Hint(stride: [2, 2]), format: .OIHW, index: layerIndex, name: "proj_in")
    outputs += referenceImages.map { xEmbedder($0) }
    mappers.append { format in
      switch format {
      case .diffusers:
        return ["proj_in.weight": [xEmbedder.weight.name], "proj_in.bias": [xEmbedder.bias.name]]
      case .generativeModels:
        return [
          "video_patch_proj.weight": [xEmbedder.weight.name],
          "video_patch_proj.bias": [xEmbedder.bias.name],
        ]
      }
    }
  }
  if !referenceAudios.isEmpty {
    let audioInput = LoRADense(
      count: hiddenSize, configuration: configuration, index: layerIndex, name: "audio_proj_in")
    outputs += referenceAudios.map { audioInput($0) }
    mappers.append { format in
      switch format {
      case .diffusers:
        return [
          "audio_proj_in.weight": [audioInput.weight.name],
          "audio_proj_in.bias": [audioInput.bias.name],
        ]
      case .generativeModels:
        return [
          "audio_patch_proj.weight": [audioInput.weight.name],
          "audio_patch_proj.bias": [audioInput.bias.name],
        ]
      }
    }
  }
  for layerIndex in 0..<layers {
    var projections = [Model]()
    for chunk in 0..<6 {
      var conditions = [Model.IO]()
      for modality in 0..<3 {
        let projection = LoRADense(
          count: hiddenSize, configuration: configuration, index: layerIndex,
          name: "adaln_\(chunk)_\(modality)")
        projections.append(projection)
        let projected = projection(activatedTimestep)
        let timestep = modality == 2 ? 1 : 0
        outputs.append(
          projected.reshaped(
            [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
            strides: [timestepCount * hiddenSize, hiddenSize, 1]
          ).contiguous())
        if !referenceImages.isEmpty && modality == 0 {
          conditions.append(
            projected.reshaped(
              [timesteps, 1, hiddenSize], offset: [0, 2, 0],
              strides: [timestepCount * hiddenSize, hiddenSize, 1]
            ).contiguous())
        }
        if !referenceAudios.isEmpty && modality == 2 {
          conditions.append(
            projected.reshaped(
              [timesteps, 1, hiddenSize], offset: [0, timestepCount - 1, 0],
              strides: [timestepCount * hiddenSize, hiddenSize, 1]
            ).contiguous())
        }
      }
      outputs += conditions
    }
    mappers.append { format in
      // Checkpoints store [modality, modulation, hidden], not construction order.
      let ordered = (0..<3).flatMap { modality in
        (0..<6).map { projections[$0 * 3 + modality] }
      }
      switch format {
      case .diffusers:
        return [
          "transformer_blocks.\(layerIndex).adaln_proj.linear.weight": ModelWeightElement(
            ordered.map { $0.weight.name }),
          "transformer_blocks.\(layerIndex).adaln_proj.linear.bias": ModelWeightElement(
            ordered.map { $0.bias.name }),
        ]
      case .generativeModels:
        return [
          "blocks.\(layerIndex).adaln_proj.linear.weight": ModelWeightElement(
            ordered.map { $0.weight.name }),
          "blocks.\(layerIndex).adaln_proj.linear.bias": ModelWeightElement(
            ordered.map { $0.bias.name }),
        ]
      }
    }
  }
  let outputShift = LoRADense(
    count: hiddenSize, configuration: configuration, index: layerIndex, name: "norm_out_shift")
  let outputScale = LoRADense(
    count: hiddenSize, configuration: configuration, index: layerIndex, name: "norm_out_scale")
  let outputShifts = outputShift(activatedTimestep)
  let outputScales = outputScale(activatedTimestep)
  for timestep in 0..<2 {
    outputs.append(
      outputShifts.reshaped(
        [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
        strides: [timestepCount * hiddenSize, hiddenSize, 1]
      ).contiguous())
    outputs.append(
      outputScales.reshaped(
        [timesteps, 1, hiddenSize], offset: [0, timestep, 0],
        strides: [timestepCount * hiddenSize, hiddenSize, 1]
      ).contiguous())
  }
  precondition(
    outputs.count == layers
      * (18 + (referenceImages.isEmpty ? 0 : 6) + (referenceAudios.isEmpty ? 0 : 6)) + 5
      + referenceImages.count + referenceAudios.count)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .diffusers:
      mapping["context_embedder.weight"] = [textInput.weight.name]
      mapping["context_embedder.bias"] = [textInput.bias.name]
      mapping["norm_out.linear.weight"] = [outputShift.weight.name, outputScale.weight.name]
      mapping["norm_out.linear.bias"] = [outputShift.bias.name, outputScale.bias.name]
    case .generativeModels:
      mapping["condition_proj.weight"] = [textInput.weight.name]
      mapping["condition_proj.bias"] = [textInput.bias.name]
      mapping["final_layer.adaln_proj.linear.weight"] = [
        outputShift.weight.name, outputScale.weight.name,
      ]
      mapping["final_layer.adaln_proj.linear.bias"] = [
        outputShift.bias.name, outputScale.bias.name,
      ]
    }
    return mapping
  }
  return (mapper, Model([text, timestepFrequencies] + referenceImages + referenceAudios, outputs))
}

/// Uses the same opt-in Sol policy and dense-layer selection as `MiniMaxH3`.
public func LoRAMiniMaxH3(
  hiddenSize: Int, layers: Int, startLayer: Int = 0, textLength: Int, audioLength: Int,
  videoFrames: Int,
  videoHeight: Int, videoWidth: Int, usesFlashAttention: FlashAttentionLevel,
  usesSolAttention: (eligibleForApproximation: Range<Int>, tau: Float)? = nil,
  referenceImageSizes: [(height: Int, width: Int)] = [], referenceAudioLengths: [Int] = [],
  visionLength: Int = 0,
  outputResidual: Bool = false, inputResidual: Bool = false,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let configuration = LoRAConfiguration
  let layerIndex = 0
  precondition(hiddenSize > 0 && layers >= 0 && startLayer >= 0)
  precondition(startLayer > 0 || layers > 0)
  precondition(!inputResidual || (startLayer > 0 && layers == 0))
  precondition(videoHeight % 2 == 0 && videoWidth % 2 == 0)
  let video = Input()
  let audio = Input()
  let contextRows = Input()
  let rot = Input()
  let referenceImages = referenceImageSizes.map { _ in Input() }
  let referenceAudios = referenceAudioLengths.map { _ in Input() }
  let referenceAudioLength = referenceAudioLengths.reduce(0, +)
  let perLayerConditions =
    18 + (referenceImages.isEmpty ? 0 : 6) + (referenceAudios.isEmpty ? 0 : 6)
  let fixedConditions = (0..<(layers * perLayerConditions)).map { _ in Input() }
  let videoLength = videoFrames * videoHeight / 2 * videoWidth / 2
  let conditionLength = referenceImageSizes.reduce(0) { $0 + $1.height / 2 * ($1.width / 2) }
  let sequenceLength =
    textLength + conditionLength + referenceAudioLength + audioLength + videoLength

  if let usesSolAttention, !usesSolAttention.eligibleForApproximation.isEmpty {
    precondition(
      usesSolAttention.eligibleForApproximation.lowerBound >= sequenceLength - videoLength)
    precondition(usesSolAttention.eligibleForApproximation.upperBound <= sequenceLength)
  }

  var out: Model.IO
  var inputs: [Model.IO]
  let outputType: Model.IO
  var mappers = [ModelWeightMapper]()
  if startLayer == 0 {
    let xEmbedder = LoRAConvolution(
      groups: 1, filters: hiddenSize, filterSize: [2, 2], configuration: configuration,
      hint: Hint(stride: [2, 2]), format: .OIHW, index: layerIndex, name: "proj_in")
    let audioInput = LoRADense(
      count: hiddenSize, configuration: configuration, index: layerIndex, name: "audio_proj_in")
    let audioRows = audioInput(audio).to(.Float32)
    let videoRows = xEmbedder(video).reshaped(.HWC(1, videoLength, hiddenSize)).to(.Float32)
    let textRows: Model.IO
    if visionLength > 0 {
      textRows = contextRows.reshaped(
        [1, textLength - visionLength, hiddenSize],
        strides: [textLength * hiddenSize, hiddenSize, 1]
      ).contiguous()
    } else {
      textRows = contextRows
    }
    // Each modulation group stays contiguous throughout all transformer blocks.
    var rows = [textRows]
    for (image, size) in zip(referenceImages, referenceImageSizes) {
      rows.append(
        image.reshaped(.HWC(1, size.height / 2 * (size.width / 2), hiddenSize)).to(.Float32))
    }
    rows += referenceAudios.map { $0.to(.Float32) }
    rows.append(audioRows)
    if visionLength > 0 {
      rows.append(
        contextRows.reshaped(
          [1, visionLength, hiddenSize], offset: [0, textLength - visionLength, 0],
          strides: [textLength * hiddenSize, hiddenSize, 1]
        ).contiguous())
    }
    rows.append(videoRows)
    out = Concat(axis: 1)(rows)
    inputs = [video, audio, contextRows, rot] + referenceImages + referenceAudios
    outputType = video
    mappers.append { format in
      switch format {
      case .diffusers:
        return [
          "proj_in.weight": [xEmbedder.weight.name],
          "proj_in.bias": [xEmbedder.bias.name],
          "audio_proj_in.weight": [audioInput.weight.name],
          "audio_proj_in.bias": [audioInput.bias.name],
        ]
      case .generativeModels:
        return [
          "video_patch_proj.weight": [xEmbedder.weight.name],
          "video_patch_proj.bias": [xEmbedder.bias.name],
          "audio_patch_proj.weight": [audioInput.weight.name],
          "audio_patch_proj.bias": [audioInput.bias.name],
        ]
      }
    }
  } else {
    let hiddenState = Input()
    out = hiddenState
    inputs = [hiddenState, rot]
    outputType = rot
  }
  let embedded = out
  for layerIndex in startLayer..<(startLayer + layers) {
    // Keep FFN outliers in FP16 range; the residual branch restores the factor in FP32.
    let scaleFactor: Int
    switch layerIndex {
    case 0, 4, 11, 31, 32:
      scaleFactor = 4
    case 1, 7, 27, 33, 34, 35, 37, 38, 46:
      scaleFactor = 8
    case 18, 30, 40, 41, 42:
      scaleFactor = 16
    case 47:
      scaleFactor = 32
    case 13, 36, 43, 44, 48, 49:
      scaleFactor = 64
    case 39:
      scaleFactor = 512
    case 45:
      scaleFactor = 1_024
    default:
      scaleFactor = 2
    }
    let (mapper, block) = LoRAH3TransformerBlock(
      prefix: ("blocks.\(layerIndex)", "transformer_blocks.\(layerIndex)"), hiddenSize: hiddenSize,
      layerIndex: layerIndex, configuration: configuration,
      textLength: textLength - visionLength, audioLength: audioLength,
      videoLength: visionLength + videoLength, conditionLength: conditionLength,
      referenceAudioLength: referenceAudioLength,
      usesFlashAttention: usesFlashAttention,
      scaleFactor: scaleFactor, startIndex: startLayer,
      usesSolAttention: [0, 1, 33, 34, 35].contains(layerIndex) ? nil : usesSolAttention)
    mappers.append(mapper)
    let conditionOffset = (layerIndex - startLayer) * perLayerConditions
    out =
      block(
        [out, rot]
          + Array(
            fixedConditions[
              conditionOffset..<(conditionOffset + perLayerConditions)
            ]))[0]
  }

  inputs += fixedConditions
  if startLayer == 0 && outputResidual {
    precondition(layers == 1)
    let mapper: ModelWeightMapper = { format in
      var mapping = ModelWeightMapping()
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      return mapping
    }
    // Run the input projections and block 0 once; the tail consumes this exact hidden state.
    return (mapper, Model(inputs, [out, out - embedded]))
  }
  let videoShift = Input()
  let videoScale = Input()
  let audioShift = Input()
  let audioScale = Input()
  inputs += [videoShift, videoScale, audioShift, audioScale]
  var residualOutputs = [Model.IO]()
  if inputResidual {
    let residual = Input()
    inputs.append(residual)
    out = out + residual
  }
  if outputResidual {
    residualOutputs = [out - embedded]
  }
  let outputNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "norm_out")
  let normalized = outputNorm(out)
  let audioOutRows =
    normalized.reshaped(
      [1, audioLength, hiddenSize],
      offset: [0, textLength - visionLength + conditionLength + referenceAudioLength, 0],
      strides: [sequenceLength * hiddenSize, hiddenSize, 1]
    ).contiguous() .* (1 + audioScale).to(.Float32) + audioShift.to(.Float32)
  let videoOutRows =
    normalized.reshaped(
      [1, videoLength, hiddenSize],
      offset: [0, textLength + conditionLength + referenceAudioLength + audioLength, 0],
      strides: [sequenceLength * hiddenSize, hiddenSize, 1]
    ).contiguous() .* (1 + videoScale).to(.Float32) + videoShift.to(.Float32)
  let videoOutput = LoRADense(
    count: 96, configuration: configuration, index: layerIndex, name: "proj_out")
  let audioOutput = LoRADense(
    count: 32, configuration: configuration, index: layerIndex, name: "audio_proj_out")
  let projectedVideo = -videoOutput(videoOutRows).reshaped([
    videoFrames, videoHeight / 2, videoWidth / 2, 24, 2, 2,
  ]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    videoFrames, videoHeight, videoWidth, 24,
  ])
  let projectedAudio = -audioOutput(audioOutRows)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .diffusers:
      mapping["proj_out.weight"] = [videoOutput.weight.name]
      mapping["proj_out.bias"] = [videoOutput.bias.name]
      mapping["audio_proj_out.weight"] = [audioOutput.weight.name]
      mapping["audio_proj_out.bias"] = [audioOutput.bias.name]
      mapping["norm_out.norm.weight"] = [outputNorm.weight.name]
    case .generativeModels:
      mapping["final_layer.video_out.weight"] = [videoOutput.weight.name]
      mapping["final_layer.video_out.bias"] = [videoOutput.bias.name]
      mapping["final_layer.audio_out.weight"] = [audioOutput.weight.name]
      mapping["final_layer.audio_out.bias"] = [audioOutput.bias.name]
      mapping["final_layer.norm.weight"] = [outputNorm.weight.name]
    }
    return mapping
  }
  return (
    mapper,
    Model(
      inputs,
      [projectedVideo.to(of: outputType), projectedAudio.to(of: startLayer == 0 ? audio : rot)]
        + residualOutputs)
  )
}
