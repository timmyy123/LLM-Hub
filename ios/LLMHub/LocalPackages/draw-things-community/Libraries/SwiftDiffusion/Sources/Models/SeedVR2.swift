import Foundation
import NNC

public enum SeedVR2MLPKind {
  case swiglu
  case gelu
}

public enum SeedVR2RotaryKind {
  case mmrope3d
  case pixelVideoOnly
}

public struct SeedVR2DiTConfiguration {
  public let hiddenSize: Int
  public let heads: Int
  public let headDim: Int
  public let layers: Int
  public let finalLayerContextAdaLN: Bool
  public let outputNormAda: Bool
  public let rotaryKind: SeedVR2RotaryKind
  public let rotaryDim: Int
  public let mlpHiddenSize: Int
  public let mlpKind: SeedVR2MLPKind

  public init(
    hiddenSize: Int, heads: Int, headDim: Int, layers: Int, finalLayerContextAdaLN: Bool,
    outputNormAda: Bool, rotaryKind: SeedVR2RotaryKind, rotaryDim: Int, mlpHiddenSize: Int,
    mlpKind: SeedVR2MLPKind
  ) {
    self.hiddenSize = hiddenSize
    self.heads = heads
    self.headDim = headDim
    self.layers = layers
    self.finalLayerContextAdaLN = finalLayerContextAdaLN
    self.outputNormAda = outputNormAda
    self.rotaryKind = rotaryKind
    self.rotaryDim = rotaryDim
    self.mlpHiddenSize = mlpHiddenSize
    self.mlpKind = mlpKind
  }

  public static let _3B = SeedVR2DiTConfiguration(
    hiddenSize: 2_560, heads: 20, headDim: 128, layers: 32, finalLayerContextAdaLN: false,
    outputNormAda: true, rotaryKind: .mmrope3d, rotaryDim: 126, mlpHiddenSize: 6_912,
    mlpKind: .swiglu)

  public static let _7B = SeedVR2DiTConfiguration(
    hiddenSize: 3_072, heads: 24, headDim: 128, layers: 36, finalLayerContextAdaLN: true,
    outputNormAda: false, rotaryKind: .pixelVideoOnly, rotaryDim: 60, mlpHiddenSize: 12_288,
    mlpKind: .gelu)

  var embeddingSize: Int { hiddenSize * 6 }
}

public func SeedVR2TimeEmbedding(
  timestep: Float, batchSize: Int, embeddingSize: Int = 256, maxPeriod: Int = 10_000
)
  -> Tensor<Float>
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timestep
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    for j in 0..<batchSize {
      embedding[j, i] = sinFreq
      embedding[j, i + half] = cosFreq
    }
  }
  return embedding
}

private func SeedVR2DiTTimeEmbedder(hiddenSize: Int) -> Model {
  let x = Input()
  let projIn = Dense(count: hiddenSize, name: "time_proj_in")
  let projHid = Dense(count: hiddenSize, name: "time_proj_hid")
  let projOut = Dense(count: hiddenSize * 6, name: "time_proj_out")
  var out = projIn(x).swish()
  out = projHid(out).swish()
  out = projOut(out)
  return Model([x], [out])
}

private func SeedVR2DiTBlockModulations(
  blockIndex: Int, branch: String, layer: String, emb: Model.IO, hiddenSize: Int,
  includeGate: Bool = true
) -> [Model.IO] {
  let layerOffset = layer == "attn" ? 0 : 3
  let shift = Parameter<Float>(
    .GPU(0), .NC(1, hiddenSize), trainable: false,
    name: "block\(blockIndex)_\(branch)_\(layer)_shift")
  let scale = Parameter<Float>(
    .GPU(0), .NC(1, hiddenSize), trainable: false,
    name: "block\(blockIndex)_\(branch)_\(layer)_scale")
  let gate = Parameter<Float>(
    .GPU(0), .NC(1, hiddenSize), trainable: false,
    name: "block\(blockIndex)_\(branch)_\(layer)_gate")
  var out = [
    emb.reshaped([1, hiddenSize], offset: [layerOffset, 0], strides: [1, 6]) + shift,
    emb.reshaped([1, hiddenSize], offset: [layerOffset + 1, 0], strides: [1, 6]) + scale,
  ]
  if includeGate {
    out.append(
      emb.reshaped([1, hiddenSize], offset: [layerOffset + 2, 0], strides: [1, 6]) + gate)
  }
  return out
}

private func SeedVR2DiTOutputModulations(emb: Model.IO, hiddenSize: Int) -> [Model.IO] {
  let shift = Parameter<Float>(
    .GPU(0), .NC(1, hiddenSize), trainable: false, name: "vid_out_ada_shift")
  let scale = Parameter<Float>(
    .GPU(0), .NC(1, hiddenSize), trainable: false, name: "vid_out_ada_scale")
  return [
    emb.reshaped([1, hiddenSize], offset: [0, 0], strides: [1, 6]) + shift,
    emb.reshaped([1, hiddenSize], offset: [1, 0], strides: [1, 6]) + scale,
  ]
}

private struct SeedVR2Window3D {
  let tStart: Int
  let tLength: Int
  let hStart: Int
  let hLength: Int
  let wStart: Int
  let wLength: Int

  var tokenLength: Int { tLength * hLength * wLength }
}

private struct SeedVR2WindowLayoutValue {
  let windows: [SeedVR2Window3D]
  let rasterToWindowIndex: [Int32]
  let windowToRasterIndex: [Int32]
  let attentionIndex: [Int32]
  let attentionToVideoIndex: [Int32]
  let attentionToTextIndex: [Int32]
  let sequenceOffsets: [Int32]
  let maxSequenceLength: Int

  var windowCount: Int { windows.count }
  var attentionLength: Int { attentionIndex.count }
}

public struct SeedVR2WindowAttentionValue {
  public let rasterToWindowIndex: Tensor<Int32>
  public let windowToShiftedIndex: Tensor<Int32>
  public let shiftedToWindowIndex: Tensor<Int32>
  public let windowToRasterIndex: Tensor<Int32>
  public let shiftedToRasterIndex: Tensor<Int32>
  public let regularAttentionIndex: Tensor<Int32>
  public let shiftedAttentionIndex: Tensor<Int32>
  public let regularAttentionToVideoIndex: Tensor<Int32>
  public let shiftedAttentionToVideoIndex: Tensor<Int32>
  public let regularAttentionToTextIndex: Tensor<Int32>
  public let shiftedAttentionToTextIndex: Tensor<Int32>
  public let regularSequenceOffsets: Tensor<Int32>
  public let shiftedSequenceOffsets: Tensor<Int32>
}

private func SeedVR2WindowRanges(size: Int, windowSize: Int, shifted: Bool) -> [(
  start: Int, length: Int
)] {
  if shifted {
    if windowSize >= size {
      return [(0, size)]
    }
    let shift = 0.5
    let windowCount = Int(ceil((Double(size) - shift) / Double(windowSize))) + 1
    var ranges = [(start: Int, length: Int)]()
    ranges.reserveCapacity(windowCount)
    for windowIndex in 0..<windowCount {
      let start = max(Int((Double(windowIndex) - shift) * Double(windowSize)), 0)
      let end = min(Int((Double(windowIndex) - shift + 1) * Double(windowSize)), size)
      if end > start {
        ranges.append((start, end - start))
      }
    }
    return ranges
  }
  let windowCount = (size + windowSize - 1) / windowSize
  var ranges = [(start: Int, length: Int)]()
  ranges.reserveCapacity(windowCount)
  for windowIndex in 0..<windowCount {
    let start = windowIndex * windowSize
    let end = min((windowIndex + 1) * windowSize, size)
    if end > start {
      ranges.append((start, end - start))
    }
  }
  return ranges
}

private func SeedVR2WindowLayout(
  frames: Int, patchHeight: Int, patchWidth: Int, textLength: Int, shifted: Bool
) -> SeedVR2WindowLayoutValue {
  let scale = sqrt(Double(45 * 80) / Double(patchHeight * patchWidth))
  let resizedHeight = max(1, Int((Double(patchHeight) * scale).rounded(.toNearestOrEven)))
  let resizedWidth = max(1, Int((Double(patchWidth) * scale).rounded(.toNearestOrEven)))
  let windowFrames = max(1, (min(frames, 30) + 3) / 4)
  let windowHeight = max(1, (resizedHeight + 2) / 3)
  let windowWidth = max(1, (resizedWidth + 2) / 3)
  let timeRanges = SeedVR2WindowRanges(size: frames, windowSize: windowFrames, shifted: shifted)
  let heightRanges = SeedVR2WindowRanges(
    size: patchHeight, windowSize: windowHeight, shifted: shifted)
  let widthRanges = SeedVR2WindowRanges(size: patchWidth, windowSize: windowWidth, shifted: shifted)
  var windows = [SeedVR2Window3D]()
  windows.reserveCapacity(timeRanges.count * heightRanges.count * widthRanges.count)
  for widthRange in widthRanges {
    for heightRange in heightRanges {
      for timeRange in timeRanges {
        windows.append(
          SeedVR2Window3D(
            tStart: timeRange.start, tLength: timeRange.length,
            hStart: heightRange.start, hLength: heightRange.length,
            wStart: widthRange.start, wLength: widthRange.length))
      }
    }
  }

  let videoLength = frames * patchHeight * patchWidth
  var windowToRaster = [Int]()
  windowToRaster.reserveCapacity(videoLength)
  for window in windows {
    for t in window.tStart..<(window.tStart + window.tLength) {
      for h in window.hStart..<(window.hStart + window.hLength) {
        for w in window.wStart..<(window.wStart + window.wLength) {
          windowToRaster.append((t * patchHeight + h) * patchWidth + w)
        }
      }
    }
  }
  precondition(windowToRaster.count == videoLength)
  var rasterToWindow = Array(repeating: 0, count: videoLength)
  for (windowIndex, rasterIndex) in windowToRaster.enumerated() {
    rasterToWindow[rasterIndex] = windowIndex
  }

  var attentionIndex = [Int32]()
  var attentionToVideoIndex = [Int32]()
  var attentionToTextIndex = Array(repeating: Int32(0), count: windows.count * textLength)
  var sequenceOffsets = [Int32]()
  sequenceOffsets.reserveCapacity(windows.count + 1)
  sequenceOffsets.append(0)
  var videoCursor = 0
  var attentionCursor: Int32 = 0
  var maxSequenceLength = 0
  for (windowIndex, window) in windows.enumerated() {
    maxSequenceLength = max(maxSequenceLength, window.tokenLength + textLength)
    for _ in 0..<window.tokenLength {
      attentionIndex.append(Int32(videoCursor))
      attentionToVideoIndex.append(attentionCursor)
      videoCursor += 1
      attentionCursor += 1
    }
    for textIndex in 0..<textLength {
      attentionIndex.append(Int32(videoLength + textIndex))
      attentionToTextIndex[textIndex * windows.count + windowIndex] = attentionCursor
      attentionCursor += 1
    }
    sequenceOffsets.append(attentionCursor)
  }
  return SeedVR2WindowLayoutValue(
    windows: windows,
    rasterToWindowIndex: windowToRaster.map { Int32($0) },
    windowToRasterIndex: rasterToWindow.map { Int32($0) },
    attentionIndex: attentionIndex,
    attentionToVideoIndex: attentionToVideoIndex,
    attentionToTextIndex: attentionToTextIndex,
    sequenceOffsets: sequenceOffsets,
    maxSequenceLength: maxSequenceLength)
}

private func SeedVR2TransitionIndex(
  from source: SeedVR2WindowLayoutValue, to target: SeedVR2WindowLayoutValue, videoLength: Int
) -> [Int32] {
  var sourceRasterToPacked = Array(repeating: 0, count: videoLength)
  for (rasterIndex, packedIndex) in source.windowToRasterIndex.enumerated() {
    sourceRasterToPacked[rasterIndex] = Int(packedIndex)
  }
  return target.rasterToWindowIndex.map { rasterIndex in
    Int32(sourceRasterToPacked[Int(rasterIndex)])
  }
}

public func SeedVR2WindowAttentionIndexer(
  frames: Int, latentHeight: Int, latentWidth: Int, textLength: Int
) -> SeedVR2WindowAttentionValue {
  let patchHeight = latentHeight / 2
  let patchWidth = latentWidth / 2
  let videoLength = frames * patchHeight * patchWidth
  let regularLayout = SeedVR2WindowLayout(
    frames: frames, patchHeight: patchHeight, patchWidth: patchWidth, textLength: textLength,
    shifted: false)
  let shiftedLayout = SeedVR2WindowLayout(
    frames: frames, patchHeight: patchHeight, patchWidth: patchWidth, textLength: textLength,
    shifted: true)
  let windowToShiftedIndex = SeedVR2TransitionIndex(
    from: regularLayout, to: shiftedLayout, videoLength: videoLength)
  let shiftedToWindowIndex = SeedVR2TransitionIndex(
    from: shiftedLayout, to: regularLayout, videoLength: videoLength)
  return SeedVR2WindowAttentionValue(
    rasterToWindowIndex: Tensor<Int32>(
      regularLayout.rasterToWindowIndex, kind: .CPU, format: .NHWC,
      shape: [regularLayout.rasterToWindowIndex.count]),
    windowToShiftedIndex: Tensor<Int32>(
      windowToShiftedIndex, kind: .CPU, format: .NHWC, shape: [windowToShiftedIndex.count]),
    shiftedToWindowIndex: Tensor<Int32>(
      shiftedToWindowIndex, kind: .CPU, format: .NHWC, shape: [shiftedToWindowIndex.count]),
    windowToRasterIndex: Tensor<Int32>(
      regularLayout.windowToRasterIndex, kind: .CPU, format: .NHWC,
      shape: [regularLayout.windowToRasterIndex.count]),
    shiftedToRasterIndex: Tensor<Int32>(
      shiftedLayout.windowToRasterIndex, kind: .CPU, format: .NHWC,
      shape: [shiftedLayout.windowToRasterIndex.count]),
    regularAttentionIndex: Tensor<Int32>(
      regularLayout.attentionIndex, kind: .CPU, format: .NHWC,
      shape: [regularLayout.attentionIndex.count]),
    shiftedAttentionIndex: Tensor<Int32>(
      shiftedLayout.attentionIndex, kind: .CPU, format: .NHWC,
      shape: [shiftedLayout.attentionIndex.count]),
    regularAttentionToVideoIndex: Tensor<Int32>(
      regularLayout.attentionToVideoIndex, kind: .CPU, format: .NHWC,
      shape: [regularLayout.attentionToVideoIndex.count]),
    shiftedAttentionToVideoIndex: Tensor<Int32>(
      shiftedLayout.attentionToVideoIndex, kind: .CPU, format: .NHWC,
      shape: [shiftedLayout.attentionToVideoIndex.count]),
    regularAttentionToTextIndex: Tensor<Int32>(
      regularLayout.attentionToTextIndex, kind: .CPU, format: .NHWC,
      shape: [regularLayout.attentionToTextIndex.count]),
    shiftedAttentionToTextIndex: Tensor<Int32>(
      shiftedLayout.attentionToTextIndex, kind: .CPU, format: .NHWC,
      shape: [shiftedLayout.attentionToTextIndex.count]),
    regularSequenceOffsets: Tensor<Int32>(
      regularLayout.sequenceOffsets, kind: .CPU, format: .NHWC,
      shape: [regularLayout.sequenceOffsets.count]),
    shiftedSequenceOffsets: Tensor<Int32>(
      shiftedLayout.sequenceOffsets, kind: .CPU, format: .NHWC,
      shape: [shiftedLayout.sequenceOffsets.count]))
}

public func SeedVR2RotaryPositionEmbedding(
  configuration: SeedVR2DiTConfiguration, frames: Int, latentHeight: Int, latentWidth: Int,
  textLength: Int, shifted: Bool = false
) -> Tensor<Float> {
  precondition(configuration.rotaryDim <= configuration.headDim)
  precondition(configuration.rotaryDim % 6 == 0)
  let patchHeight = latentHeight / 2
  let patchWidth = latentWidth / 2
  let layout = SeedVR2WindowLayout(
    frames: frames, patchHeight: patchHeight, patchWidth: patchWidth, textLength: textLength,
    shifted: shifted)
  var rotary = Tensor<Float>(.CPU, .NHWC(1, layout.attentionLength, 1, configuration.headDim))
  for token in 0..<layout.attentionLength {
    for k in 0..<(configuration.headDim / 2) {
      rotary[0, token, 0, k * 2] = 1
      rotary[0, token, 0, k * 2 + 1] = 0
    }
  }
  let axisDim = configuration.rotaryDim / 3
  let axisPairs = axisDim / 2
  var token = 0
  switch configuration.rotaryKind {
  case .mmrope3d:
    for window in layout.windows {
      for t in 0..<window.tLength {
        for h in 0..<window.hLength {
          for w in 0..<window.wLength {
            for k in 0..<axisPairs {
              let frequencyScale = pow(10_000, Double(k) * 2 / Double(axisDim))
              let temporalTheta = Double(textLength + t) / frequencyScale
              rotary[0, token, 0, k * 2] = Float(cos(temporalTheta))
              rotary[0, token, 0, k * 2 + 1] = Float(sin(temporalTheta))
              let heightTheta = Double(h) / frequencyScale
              rotary[0, token, 0, axisDim + k * 2] = Float(cos(heightTheta))
              rotary[0, token, 0, axisDim + k * 2 + 1] = Float(sin(heightTheta))
              let widthTheta = Double(w) / frequencyScale
              rotary[0, token, 0, axisDim * 2 + k * 2] = Float(cos(widthTheta))
              rotary[0, token, 0, axisDim * 2 + k * 2 + 1] = Float(sin(widthTheta))
            }
            token += 1
          }
        }
      }
      for textIndex in 0..<textLength {
        for k in 0..<axisPairs {
          let theta = Double(textIndex) / pow(10_000, Double(k) * 2 / Double(axisDim))
          let cosTheta = Float(cos(theta))
          let sinTheta = Float(sin(theta))
          rotary[0, token, 0, k * 2] = cosTheta
          rotary[0, token, 0, k * 2 + 1] = sinTheta
          rotary[0, token, 0, axisDim + k * 2] = cosTheta
          rotary[0, token, 0, axisDim + k * 2 + 1] = sinTheta
          rotary[0, token, 0, axisDim * 2 + k * 2] = cosTheta
          rotary[0, token, 0, axisDim * 2 + k * 2 + 1] = sinTheta
        }
        token += 1
      }
    }
  case .pixelVideoOnly:
    for window in layout.windows {
      for t in 0..<window.tLength {
        let temporalPosition =
          window.tLength <= 1 ? -1 : -1 + 2 * Double(t) / Double(window.tLength - 1)
        for h in 0..<window.hLength {
          let heightPosition =
            window.hLength <= 1 ? -1 : -1 + 2 * Double(h) / Double(window.hLength - 1)
          for w in 0..<window.wLength {
            let widthPosition =
              window.wLength <= 1 ? -1 : -1 + 2 * Double(w) / Double(window.wLength - 1)
            for k in 0..<axisPairs {
              let multiplier = axisPairs == 1 ? 1 : 1 + 127 * Double(k) / Double(axisPairs - 1)
              let temporalTheta = temporalPosition * multiplier * Double.pi
              rotary[0, token, 0, k * 2] = Float(cos(temporalTheta))
              rotary[0, token, 0, k * 2 + 1] = Float(sin(temporalTheta))
              let heightTheta = heightPosition * multiplier * Double.pi
              rotary[0, token, 0, axisDim + k * 2] = Float(cos(heightTheta))
              rotary[0, token, 0, axisDim + k * 2 + 1] = Float(sin(heightTheta))
              let widthTheta = widthPosition * multiplier * Double.pi
              rotary[0, token, 0, axisDim * 2 + k * 2] = Float(cos(widthTheta))
              rotary[0, token, 0, axisDim * 2 + k * 2 + 1] = Float(sin(widthTheta))
            }
            token += 1
          }
        }
      }
      token += textLength
    }
  }
  precondition(token == layout.attentionLength)
  return rotary
}

private func SeedVR2DiTBlock(
  configuration: SeedVR2DiTConfiguration, layerIndex: Int, frames: Int, height: Int, width: Int,
  txtLen: Int, attentionLength: Int, windowCount: Int, maxSequenceLength: Int,
  contextBlockPreOnly: Bool = false, contextAdaLN: Bool = true,
  usesFlashAttention: FlashAttentionLevel
) -> Model {
  let vid = Input()
  let txt = Input()
  let emb = Input()
  let rotaryEmbedding = Input()
  let sequenceOffsets = Input()
  let attentionIndex = Input()
  let attentionToVideoIndex = Input()
  let attentionToTextIndex = Input()
  let hiddenSize = configuration.hiddenSize
  let heads = configuration.heads
  let headDim = configuration.headDim
  let vidLen = frames * height * width

  let vidAttnMod = SeedVR2DiTBlockModulations(
    blockIndex: layerIndex, branch: "vid", layer: "attn", emb: emb, hiddenSize: hiddenSize)
  let vidMlpMod = SeedVR2DiTBlockModulations(
    blockIndex: layerIndex, branch: "vid", layer: "mlp", emb: emb, hiddenSize: hiddenSize)

  let vidAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_attn_norm")
  let txtAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "txt_attn_norm")
  let vidAttnIn = vidAttnNorm(vid) .* vidAttnMod[1] + vidAttnMod[0]
  let txtAttnIn: Model.IO
  let txtAttnGate: Model.IO?
  let txtMlpMod: [Model.IO]?
  if contextAdaLN {
    let txtAttnMod = SeedVR2DiTBlockModulations(
      blockIndex: layerIndex, branch: "txt", layer: "attn", emb: emb, hiddenSize: hiddenSize,
      includeGate: !contextBlockPreOnly)
    txtAttnIn = txtAttnNorm(txt) .* txtAttnMod[1] + txtAttnMod[0]
    txtAttnGate = contextBlockPreOnly ? nil : txtAttnMod[2]
    if !contextBlockPreOnly {
      txtMlpMod = SeedVR2DiTBlockModulations(
        blockIndex: layerIndex, branch: "txt", layer: "mlp", emb: emb, hiddenSize: hiddenSize)
    } else {
      txtMlpMod = nil
    }
  } else {
    txtAttnIn = txtAttnNorm(txt)
    txtAttnGate = nil
    txtMlpMod = nil
  }

  let vidQProj = Dense(count: hiddenSize, noBias: true, name: "vid_q")
  let vidKProj = Dense(count: hiddenSize, noBias: true, name: "vid_k")
  let vidVProj = Dense(count: hiddenSize, noBias: true, name: "vid_v")
  let txtQProj = Dense(count: hiddenSize, noBias: true, name: "txt_q")
  let txtKProj = Dense(count: hiddenSize, noBias: true, name: "txt_k")
  let txtVProj = Dense(count: hiddenSize, noBias: true, name: "txt_v")
  let vidNormQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "vid_norm_q")
  let vidNormK = RMSNorm(epsilon: 1e-5, axis: [2], name: "vid_norm_k")
  let txtNormQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "txt_norm_q")
  let txtNormK = RMSNorm(epsilon: 1e-5, axis: [2], name: "txt_norm_k")

  let vidAttnProjIn = vidAttnIn.to(.Float16)
  let txtAttnProjIn = txtAttnIn.to(.Float16)
  let vidQ = vidNormQ(vidQProj(vidAttnProjIn).reshaped([vidLen, heads, headDim]))
  let vidK = vidNormK(vidKProj(vidAttnProjIn).reshaped([vidLen, heads, headDim]))
  let vidV = vidVProj(vidAttnProjIn).reshaped([vidLen, heads, headDim])
  let txtK = txtNormK(txtKProj(txtAttnProjIn).reshaped([txtLen, heads, headDim]))
  let txtV = txtVProj(txtAttnProjIn).reshaped([txtLen, heads, headDim])
  let txtQ =
    !contextBlockPreOnly
    ? txtNormQ(txtQProj(txtAttnProjIn).reshaped([txtLen, heads, headDim])) : txtK

  let sourceLength = vidLen + txtLen
  let qSource = Functional.concat(axis: 0, vidQ, txtQ).reshaped([sourceLength, heads * headDim])
  let kSource = Functional.concat(axis: 0, vidK, txtK).reshaped([sourceLength, heads * headDim])
  let vSource = Functional.concat(axis: 0, vidV, txtV).reshaped([sourceLength, heads * headDim])
  let q = Functional.cmul(
    left: Functional.indexSelect(input: qSource, index: attentionIndex).reshaped([
      1, attentionLength, heads, headDim,
    ]),
    right: rotaryEmbedding)
  let k = Functional.cmul(
    left: Functional.indexSelect(input: kSource, index: attentionIndex).reshaped([
      1, attentionLength, heads, headDim,
    ]),
    right: rotaryEmbedding)
  let v = Functional.indexSelect(input: vSource, index: attentionIndex).reshaped([
    1, attentionLength, heads, headDim,
  ])
  let attn3D: Model.IO
  switch usesFlashAttention {
  case .none, .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1, isVariableLength: true,
      maxSequenceLength: (query: maxSequenceLength, keyValue: maxSequenceLength),
      flags: [.Float16])
    attn3D = scaledDotProductAttention([
      (1.0 / Float(headDim).squareRoot()) * q, k, v, sequenceOffsets, sequenceOffsets,
    ]).reshaped([
      attentionLength, heads * headDim,
    ])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(headDim).squareRoot(), isVariableLength: true,
      maxSequenceLength: (query: maxSequenceLength, keyValue: maxSequenceLength),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    attn3D = scaledDotProductAttention([q, k, v, sequenceOffsets, sequenceOffsets]).reshaped([
      attentionLength, heads * headDim,
    ])
  }
  let vidAttn3D = Functional.indexSelect(input: attn3D, index: attentionToVideoIndex).reshaped([
    vidLen, heads, headDim,
  ])
  let txtAttn3D: Model.IO?
  if contextBlockPreOnly {
    txtAttn3D = nil
  } else {
    txtAttn3D = Functional.indexSelect(input: attn3D, index: attentionToTextIndex)
      .reshaped([txtLen, windowCount, heads, headDim]).reduced(.mean, axis: [1])
  }
  let vidAttnFlat = vidAttn3D.contiguous().reshaped([vidLen, heads * headDim])
  let vidAttnOut = Dense(count: hiddenSize, name: "vid_attn_out")
  let vidAttnProjected = vidAttnOut(vidAttnFlat).to(.Float32)
  let txtAfterAttn: Model.IO?
  if let txtAttn3D = txtAttn3D {
    let txtAttnFlat = txtAttn3D.contiguous().reshaped([txtLen, heads * headDim])
    let txtAttnOut = Dense(count: hiddenSize, name: "txt_attn_out")
    let txtAttnProjected = txtAttnOut(txtAttnFlat).to(.Float32)
    txtAfterAttn = txt + txtAttnProjected .* txtAttnGate!
  } else {
    txtAfterAttn = nil
  }
  let vidAfterAttn = vid + vidAttnProjected .* vidAttnMod[2]

  let vidMlpNorm = RMSNorm(epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_mlp_norm")
  let txtMlpNorm = RMSNorm(epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "txt_mlp_norm")
  let vidMlpIn = vidMlpNorm(vidAfterAttn) .* vidMlpMod[1] + vidMlpMod[0]
  let txtMlpIn: Model.IO?
  if let txtAfterAttn = txtAfterAttn, let txtMlpMod = txtMlpMod {
    txtMlpIn = txtMlpNorm(txtAfterAttn) .* txtMlpMod[1] + txtMlpMod[0]
  } else {
    txtMlpIn = nil
  }

  let vidFinal: Model.IO
  let txtFinal: Model.IO?
  switch configuration.mlpKind {
  case .swiglu:
    let vidMlpGate = Dense(count: configuration.mlpHiddenSize, noBias: true, name: "vid_mlp_gate")
    let vidMlpInProj = Dense(count: configuration.mlpHiddenSize, noBias: true, name: "vid_mlp_in")
    let vidMlpOutProj = Dense(count: hiddenSize, noBias: true, name: "vid_mlp_out")
    let vidMlpProjIn = vidMlpIn.to(.Float16)
    let vidGate = vidMlpGate(vidMlpProjIn)
    let vidInner = vidMlpInProj(vidMlpProjIn)
    let vidMlpOut = vidMlpOutProj((vidGate .* vidGate.sigmoid()) .* vidInner).to(.Float32)
    vidFinal = (vidAfterAttn + vidMlpOut .* vidMlpMod[2]).to(of: vidAfterAttn)

    if let txtMlpIn = txtMlpIn, let txtMlpMod = txtMlpMod {
      let txtMlpGate = Dense(count: configuration.mlpHiddenSize, noBias: true, name: "txt_mlp_gate")
      let txtMlpInProj = Dense(count: configuration.mlpHiddenSize, noBias: true, name: "txt_mlp_in")
      let txtMlpOutProj = Dense(count: hiddenSize, noBias: true, name: "txt_mlp_out")
      let txtMlpProjIn = txtMlpIn.to(.Float16)
      let txtGate = txtMlpGate(txtMlpProjIn)
      let txtInner = txtMlpInProj(txtMlpProjIn)
      let txtMlpOut = txtMlpOutProj((txtGate .* txtGate.sigmoid()) .* txtInner).to(.Float32)
      txtFinal = (txtAfterAttn! + txtMlpOut .* txtMlpMod[2]).to(of: txtAfterAttn!)
    } else {
      txtFinal = nil
    }
  case .gelu:
    let vidMlpInProj = Dense(count: configuration.mlpHiddenSize, name: "vid_mlp_in")
    let vidMlpOutProj = Dense(count: hiddenSize, name: "vid_mlp_out")
    let vidMlpOut = vidMlpOutProj(vidMlpInProj(vidMlpIn.to(.Float16)).GELU(approximate: .tanh))
      .to(.Float32)
    vidFinal = (vidAfterAttn + vidMlpOut .* vidMlpMod[2]).to(of: vidAfterAttn)

    if let txtMlpIn = txtMlpIn, let txtMlpMod = txtMlpMod {
      let txtMlpInProj = Dense(count: configuration.mlpHiddenSize, name: "txt_mlp_in")
      let txtMlpOutProj = Dense(count: hiddenSize, name: "txt_mlp_out")
      let txtMlpOut = txtMlpOutProj(txtMlpInProj(txtMlpIn.to(.Float16)).GELU(approximate: .tanh))
        .to(.Float32)
      txtFinal = (txtAfterAttn! + txtMlpOut .* txtMlpMod[2]).to(of: txtAfterAttn!)
    } else {
      txtFinal = nil
    }
  }

  var blockInputs = [
    vid, txt, emb, rotaryEmbedding, sequenceOffsets, attentionIndex, attentionToVideoIndex,
  ]
  if !contextBlockPreOnly {
    blockInputs.append(attentionToTextIndex)
  }
  if let txtFinal = txtFinal {
    return Model(blockInputs, [vidFinal, txtFinal])
  }
  return Model(blockInputs, [vidFinal])
}

public func SeedVR2DiT(
  configuration: SeedVR2DiTConfiguration, frames: Int, latentHeight: Int, latentWidth: Int,
  textLength: Int, usesFlashAttention: FlashAttentionLevel
) -> Model {
  let vid = Input()
  let timestep = Input()
  let txt = Input()
  let rotaryEmbedding = Input()
  let shiftedRotaryEmbedding = Input()
  let rasterToWindowIndex = Input()
  let windowToShiftedIndex = Input()
  let shiftedToWindowIndex = Input()
  let windowToRasterIndex = Input()
  let shiftedToRasterIndex = Input()
  let regularAttentionIndex = Input()
  let shiftedAttentionIndex = Input()
  let regularAttentionToVideoIndex = Input()
  let shiftedAttentionToVideoIndex = Input()
  let regularAttentionToTextIndex = Input()
  let shiftedAttentionToTextIndex = Input()
  let regularSequenceOffsets = Input()
  let shiftedSequenceOffsets = Input()
  let patchHeight = latentHeight / 2
  let patchWidth = latentWidth / 2
  let regularLayout = SeedVR2WindowLayout(
    frames: frames, patchHeight: patchHeight, patchWidth: patchWidth, textLength: textLength,
    shifted: false)
  let shiftedLayout = SeedVR2WindowLayout(
    frames: frames, patchHeight: patchHeight, patchWidth: patchWidth, textLength: textLength,
    shifted: true)

  let txtIn = Dense(count: configuration.hiddenSize, name: "c_embedder")
  let patchIn = Dense(count: configuration.hiddenSize, name: "x_embedder")
  let embIn = SeedVR2DiTTimeEmbedder(hiddenSize: configuration.hiddenSize)
  var txtOut = txtIn(txt).to(.Float32)
  var vidOut = vid.reshaped([frames, latentHeight, latentWidth, 33], format: .NHWC)
  vidOut = vidOut.reshaped(
    [frames, patchHeight, 2, patchWidth, 2, 33], format: .NHWC
  ).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    frames * patchHeight * patchWidth, 132,
  ])
  vidOut = patchIn(vidOut).to(.Float32)
  vidOut = Functional.indexSelect(input: vidOut, index: rasterToWindowIndex)
  let emb = embIn(timestep).to(.Float32)
  var isShiftedLayout = false
  let rotResized = rotaryEmbedding.reshaped([
    1, regularLayout.attentionLength, 1, configuration.headDim,
  ])
  let shiftedRotResized = shiftedRotaryEmbedding.reshaped([
    1, shiftedLayout.attentionLength, 1, configuration.headDim,
  ])

  for layerIndex in 0..<configuration.layers {
    let isFinalLayer = layerIndex == configuration.layers - 1
    let contextBlockPreOnly = isFinalLayer
    let contextAdaLN = !contextBlockPreOnly || configuration.finalLayerContextAdaLN
    let useShiftedAttention = layerIndex % 2 == 1 || !contextAdaLN
    if useShiftedAttention && !isShiftedLayout {
      vidOut = Functional.indexSelect(input: vidOut, index: windowToShiftedIndex)
      isShiftedLayout = true
    } else if !useShiftedAttention && isShiftedLayout {
      vidOut = Functional.indexSelect(input: vidOut, index: shiftedToWindowIndex)
      isShiftedLayout = false
    }
    let layout = useShiftedAttention ? shiftedLayout : regularLayout
    let block = SeedVR2DiTBlock(
      configuration: configuration, layerIndex: layerIndex, frames: frames, height: patchHeight,
      width: patchWidth, txtLen: textLength, attentionLength: layout.attentionLength,
      windowCount: layout.windowCount, maxSequenceLength: layout.maxSequenceLength,
      contextBlockPreOnly: contextBlockPreOnly, contextAdaLN: contextAdaLN,
      usesFlashAttention: usesFlashAttention)
    var blockInputs = [
      vidOut, txtOut, emb,
      useShiftedAttention ? shiftedRotResized : rotResized,
      useShiftedAttention ? shiftedSequenceOffsets : regularSequenceOffsets,
      useShiftedAttention ? shiftedAttentionIndex : regularAttentionIndex,
      useShiftedAttention ? shiftedAttentionToVideoIndex : regularAttentionToVideoIndex,
    ]
    if !contextBlockPreOnly {
      blockInputs.append(
        useShiftedAttention ? shiftedAttentionToTextIndex : regularAttentionToTextIndex)
    }
    let blockOut = block(blockInputs)
    if !contextBlockPreOnly {
      vidOut = blockOut[0]
      txtOut = blockOut[1]
    } else {
      vidOut = blockOut
    }
  }
  vidOut = Functional.indexSelect(
    input: vidOut, index: isShiftedLayout ? shiftedToRasterIndex : windowToRasterIndex)

  var out: Model.IO
  if configuration.outputNormAda {
    let mod = SeedVR2DiTOutputModulations(emb: emb, hiddenSize: configuration.hiddenSize)
    let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm_out")
    let outProj = Dense(count: 64, name: "linear")
    let afterAda = (norm(vidOut) .* mod[1] + mod[0]).to(.Float16)
    out = outProj(afterAda).reshaped(
      [frames, patchHeight, patchWidth, 2, 2, 16], format: .NHWC)
    out = out.permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
      frames, latentHeight, latentWidth, 16,
    ])
  } else {
    let outProj = Dense(count: 64, name: "linear")
    out = outProj(vidOut.to(.Float16)).reshaped(
      [frames, patchHeight, patchWidth, 2, 2, 16], format: .NHWC)
    out = out.permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
      frames, latentHeight, latentWidth, 16,
    ])
  }
  out.add(dependencies: [isShiftedLayout ? windowToRasterIndex : shiftedToRasterIndex])
  return Model(
    [
      vid, timestep, txt, rotaryEmbedding, shiftedRotaryEmbedding, rasterToWindowIndex,
      windowToShiftedIndex, shiftedToWindowIndex, windowToRasterIndex, shiftedToRasterIndex,
      regularAttentionIndex, shiftedAttentionIndex, regularAttentionToVideoIndex,
      shiftedAttentionToVideoIndex, regularAttentionToTextIndex, shiftedAttentionToTextIndex,
      regularSequenceOffsets, shiftedSequenceOffsets,
    ],
    [out])
}
