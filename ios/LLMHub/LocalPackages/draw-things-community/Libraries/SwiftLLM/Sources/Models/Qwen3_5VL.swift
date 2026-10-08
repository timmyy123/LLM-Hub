import Foundation
import NNC

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

public struct Qwen3_5VisionConfiguration: Sendable {
  public var hiddenSize: Int
  public var intermediateSize: Int
  public var outputHiddenSize: Int
  public var layers: Int
  public var heads: Int
  public var patchSize: Int
  public var temporalPatchSize: Int
  public var spatialMergeSize: Int
  public var positionEmbeddings: Int
  public var layerNormEpsilon: Float

  public init(
    hiddenSize: Int = 1_024, intermediateSize: Int = 4_096, outputHiddenSize: Int = 2_560,
    layers: Int = 24, heads: Int = 16, patchSize: Int = 16, temporalPatchSize: Int = 2,
    spatialMergeSize: Int = 2, positionEmbeddings: Int = 2_304,
    layerNormEpsilon: Float = 1e-6
  ) {
    self.hiddenSize = hiddenSize
    self.intermediateSize = intermediateSize
    self.outputHiddenSize = outputHiddenSize
    self.layers = layers
    self.heads = heads
    self.patchSize = patchSize
    self.temporalPatchSize = temporalPatchSize
    self.spatialMergeSize = spatialMergeSize
    self.positionEmbeddings = positionEmbeddings
    self.layerNormEpsilon = layerNormEpsilon
  }

  public var headDim: Int { hiddenSize / heads }
  public var patchVectorSize: Int { 3 * temporalPatchSize * patchSize * patchSize }
  public var mergedHiddenSize: Int { hiddenSize * spatialMergeSize * spatialMergeSize }
}

extension Qwen3_5VisionConfiguration {
  public static let qwen3_5_4B = Qwen3_5VisionConfiguration()

  public static let qwen3_5_27B = Qwen3_5VisionConfiguration(
    hiddenSize: 1_152, intermediateSize: 4_304, outputHiddenSize: 5_120,
    layers: 27, heads: 16, patchSize: 16, temporalPatchSize: 2,
    spatialMergeSize: 2, positionEmbeddings: 2_304, layerNormEpsilon: 1e-6)

  public static let qwen3_5_9B = Qwen3_5VisionConfiguration(
    hiddenSize: 1_152, intermediateSize: 4_304, outputHiddenSize: 4_096,
    layers: 27, heads: 16, patchSize: 16, temporalPatchSize: 2,
    spatialMergeSize: 2, positionEmbeddings: 2_304, layerNormEpsilon: 1e-6)
}

public struct Qwen3_5MultimodalPositionIDs: Sendable {
  public var temporal: [Int]
  public var height: [Int]
  public var width: [Int]
  public var ropeDelta: Int

  public init(temporal: [Int], height: [Int], width: [Int], ropeDelta: Int) {
    self.temporal = temporal
    self.height = height
    self.width = width
    self.ropeDelta = ropeDelta
  }
}

public func Qwen3_5VisionTokenCount(
  gridThw: (t: Int, h: Int, w: Int),
  configuration: Qwen3_5VisionConfiguration = .qwen3_5_4B
) -> Int {
  return gridThw.t * gridThw.h * gridThw.w
    / (configuration.spatialMergeSize * configuration.spatialMergeSize)
}

public func Qwen3_5VisionSize(
  height: Int, width: Int, configuration: Qwen3_5VisionConfiguration = .qwen3_5_4B,
  size: (height: Int, width: Int)? = nil
) -> (height: Int, width: Int) {
  let factor = configuration.patchSize * configuration.spatialMergeSize
  let temporal =
    ((1 + configuration.temporalPatchSize - 1) / configuration.temporalPatchSize)
    * configuration.temporalPatchSize
  var resizedHeight = max(factor, Int((Double(height) / Double(factor)).rounded()) * factor)
  var resizedWidth = max(factor, Int((Double(width) / Double(factor)).rounded()) * factor)
  if temporal * resizedHeight * resizedWidth > 512 * 512 {
    let beta = sqrt(Double(width * height) / Double(512 * 512))
    resizedHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
    resizedWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
  } else if temporal * resizedHeight * resizedWidth < 256 * 256 {
    let beta = sqrt(Double(256 * 256) / Double(width * height))
    resizedHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
    resizedWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
  }
  if let size {
    precondition(
      size.height > 0 && size.width > 0 && size.height % factor == 0 && size.width % factor == 0)
    resizedHeight = size.height
    resizedWidth = size.width
  }
  return (resizedHeight, resizedWidth)
}

public func Qwen3_5VisionPreprocess<FloatType: TensorNumeric>(
  _ input: Tensor<FloatType>, configuration: Qwen3_5VisionConfiguration = .qwen3_5_4B,
  size: (height: Int, width: Int)? = nil
) -> (patches: Tensor<Float>, grid: (t: Int, h: Int, w: Int)) {
  let shape = input.shape
  precondition(shape.count == 4)
  precondition(shape[0] == 1)
  precondition(shape[3] == 3)
  let targetSize = Qwen3_5VisionSize(
    height: shape[1], width: shape[2], configuration: configuration, size: size)
  let resizedHeight = targetSize.height
  let resizedWidth = targetSize.width
  let resizedTensor: Tensor<Float>
  if resizedWidth == shape[2] && resizedHeight == shape[1] {
    resizedTensor = Tensor<Float>(from: input)
  } else {
    let f32 = Tensor<Float>(from: input.reshaped(.HWC(shape[1], shape[2], shape[3])))
    var output: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
      Int32(resizedHeight), Int32(resizedWidth), Int32(CCV_C3 | CCV_32F), nil, 0)
    ccv_resample(
      UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
      &output,
      0, Double(resizedHeight) / Double(shape[1]), Double(resizedWidth) / Double(shape[2]),
      Int32(CCV_INTER_AREA | CCV_INTER_CUBIC))
    resizedTensor = Tensor<Float>(
      .CPU, format: .NHWC, shape: [1, resizedHeight, resizedWidth, 3],
      unsafeMutablePointer: output!.pointee.data.f32, bindLifetimeOf: output!
    ).copied()
    ccv_matrix_free(output)
  }
  let patch = configuration.patchSize
  let temporalPatch = configuration.temporalPatchSize
  let merge = configuration.spatialMergeSize
  let grid = (t: 1, h: resizedHeight / patch, w: resizedWidth / patch)
  var patches = Tensor<Float>(.CPU, .WC(grid.t * grid.h * grid.w, configuration.patchVectorSize))
  patches.withUnsafeMutableBytes { patchBuffer in
    guard let patchPtr = patchBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
      return
    }
    resizedTensor.withUnsafeBytes { resizedBuffer in
      guard let resizedPtr = resizedBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
        return
      }
      var row = 0
      for _ in 0..<grid.t {
        for blockY in 0..<(grid.h / merge) {
          for blockX in 0..<(grid.w / merge) {
            for intraY in 0..<merge {
              for intraX in 0..<merge {
                let destination = patchPtr.advanced(by: row * configuration.patchVectorSize)
                let baseY = (blockY * merge + intraY) * patch
                let baseX = (blockX * merge + intraX) * patch
                var column = 0
                for channel in 0..<3 {
                  for _ in 0..<temporalPatch {
                    for py in 0..<patch {
                      let sourceBase = ((baseY + py) * resizedWidth + baseX) * 3 + channel
                      for px in 0..<patch {
                        destination[column] = resizedPtr[sourceBase + px * 3]
                        column += 1
                      }
                    }
                  }
                }
                row += 1
              }
            }
          }
        }
      }
    }
  }
  return (patches, grid)
}

/// Expands already-tokenized image/video placeholders. Qwen3.5 video prompts should first be split into per-frame timestamp groups by the caller.
public func Qwen3_5ExpandedMultimodalTokenIDs(
  _ tokenIDs: [Int32], imageGridThw: [(t: Int, h: Int, w: Int)] = [],
  videoGridThw: [(t: Int, h: Int, w: Int)] = [],
  imageTokenID: Int32 = 248_056, videoTokenID: Int32 = 248_057,
  configuration: Qwen3_5VisionConfiguration = .qwen3_5_4B
) -> (tokenIDs: [Int32], tokenTypeIDs: [Int32]) {
  var imageIndex = 0
  var videoIndex = 0
  var expandedTokenIDs = [Int32]()
  var tokenTypeIDs = [Int32]()
  for tokenID in tokenIDs {
    if tokenID == imageTokenID, imageIndex < imageGridThw.count {
      let count = Qwen3_5VisionTokenCount(
        gridThw: imageGridThw[imageIndex], configuration: configuration)
      expandedTokenIDs.append(contentsOf: Array(repeating: tokenID, count: count))
      tokenTypeIDs.append(contentsOf: Array(repeating: 1, count: count))
      imageIndex += 1
    } else if tokenID == videoTokenID, videoIndex < videoGridThw.count {
      let count = Qwen3_5VisionTokenCount(
        gridThw: videoGridThw[videoIndex], configuration: configuration)
      expandedTokenIDs.append(contentsOf: Array(repeating: tokenID, count: count))
      tokenTypeIDs.append(contentsOf: Array(repeating: 2, count: count))
      videoIndex += 1
    } else {
      expandedTokenIDs.append(tokenID)
      tokenTypeIDs.append(0)
    }
  }
  return (expandedTokenIDs, tokenTypeIDs)
}

public func Qwen3_5MakeMultimodalPositionIDs(
  tokenTypeIDs: [Int32], imageGridThw: [(t: Int, h: Int, w: Int)] = [],
  videoGridThw: [(t: Int, h: Int, w: Int)] = [],
  configuration: Qwen3_5VisionConfiguration = .qwen3_5_4B
) -> Qwen3_5MultimodalPositionIDs {
  // Qwen3.5 inserts timestamp text around videos, so M-RoPE consumes one frame grid per video token group.
  var videoFrameGrids = [(t: Int, h: Int, w: Int)]()
  for grid in videoGridThw {
    for _ in 0..<grid.t {
      videoFrameGrids.append((1, grid.h, grid.w))
    }
  }
  var imageIndex = 0
  var videoIndex = 0
  var currentPosition = 0
  var temporal = [Int]()
  var height = [Int]()
  var width = [Int]()
  var offset = 0
  while offset < tokenTypeIDs.count {
    let tokenType = tokenTypeIDs[offset]
    var end = offset + 1
    while end < tokenTypeIDs.count, tokenTypeIDs[end] == tokenType {
      end += 1
    }
    if tokenType == 0 {
      for i in 0..<(end - offset) {
        temporal.append(currentPosition + i)
        height.append(currentPosition + i)
        width.append(currentPosition + i)
      }
      currentPosition += end - offset
    } else {
      let grid: (t: Int, h: Int, w: Int)
      if tokenType == 1, imageIndex < imageGridThw.count {
        grid = imageGridThw[imageIndex]
        imageIndex += 1
      } else if tokenType == 2, videoIndex < videoFrameGrids.count {
        grid = videoFrameGrids[videoIndex]
        videoIndex += 1
      } else {
        grid = (1, 0, 0)
      }
      let merge = configuration.spatialMergeSize
      let llmT = grid.t
      let llmH = grid.h / merge
      let llmW = grid.w / merge
      for t in 0..<llmT {
        for y in 0..<llmH {
          for x in 0..<llmW {
            temporal.append(currentPosition + t)
            height.append(currentPosition + y)
            width.append(currentPosition + x)
          }
        }
      }
      currentPosition += max(grid.h, grid.w) / merge
    }
    offset = end
  }
  let maxPosition =
    zip(temporal, zip(height, width)).map { max($0.0, max($0.1.0, $0.1.1)) }.max()
    ?? -1
  return Qwen3_5MultimodalPositionIDs(
    temporal: temporal, height: height, width: width,
    ropeDelta: maxPosition + 1 - tokenTypeIDs.count)
}

public func Qwen3_5RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  positionIDs: Qwen3_5MultimodalPositionIDs,
  configuration: Qwen3_5ModelConfiguration,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let sequenceLength = positionIDs.temporal.count
  let halfRotaryDim = configuration.rotaryDim / 2
  let halfHeadDim = configuration.attentionHeadDim / 2
  let mropeSection = configuration.mropeSection
  var rotary = Tensor<FloatType>(
    Array(repeating: FloatType.zero, count: sequenceLength * configuration.attentionHeadDim), .CPU,
    .NHWC(1, sequenceLength, 1, configuration.attentionHeadDim))
  for i in 0..<sequenceLength {
    for k in 0..<halfRotaryDim {
      let sectionIndex = k % 3
      let position: Int
      if sectionIndex == 1 && k < mropeSection.height * 3 {
        position = positionIDs.height[i]
      } else if sectionIndex == 2 && k < mropeSection.width * 3 {
        position = positionIDs.width[i]
      } else {
        position = positionIDs.temporal[i]
      }
      let theta =
        Double(position)
        / pow(configuration.ropeTheta, Double(k) * 2 / Double(configuration.rotaryDim))
      rotary[0, i, 0, k * 2] = FloatType(cos(theta))
      rotary[0, i, 0, k * 2 + 1] = FloatType(sin(theta))
    }
    for k in halfRotaryDim..<halfHeadDim {
      rotary[0, i, 0, k * 2] = 1
    }
  }
  return rotary
}

public func Qwen3_5VisionRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  gridThw: [(t: Int, h: Int, w: Int)],
  configuration: Qwen3_5VisionConfiguration = .qwen3_5_4B,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let tokenLength = gridThw.reduce(0) { $0 + $1.t * $1.h * $1.w }
  let headDim = configuration.headDim
  let halfVisionDim = headDim / 2
  let quarterVisionDim = halfVisionDim / 2
  let merge = configuration.spatialMergeSize
  var rotary = Tensor<FloatType>(
    Array(repeating: FloatType.zero, count: tokenLength * headDim), .CPU,
    .NHWC(1, tokenLength, 1, headDim))
  var tokenIndex = 0
  for grid in gridThw {
    for _ in 0..<grid.t {
      // Match transformers' spatial-merge ordering: block row, block column, intra row, intra column.
      for blockY in 0..<(grid.h / merge) {
        for blockX in 0..<(grid.w / merge) {
          for intraY in 0..<merge {
            for intraX in 0..<merge {
              let y = blockY * merge + intraY
              let x = blockX * merge + intraX
              for k in 0..<halfVisionDim {
                let position = k < quarterVisionDim ? y : x
                let freqIndex = k < quarterVisionDim ? k : k - quarterVisionDim
                let theta =
                  Double(position) / pow(10_000.0, Double(freqIndex * 2) / Double(halfVisionDim))
                rotary[0, tokenIndex, 0, k * 2] = FloatType(cos(theta))
                rotary[0, tokenIndex, 0, k * 2 + 1] = FloatType(sin(theta))
              }
              tokenIndex += 1
            }
          }
        }
      }
    }
  }
  return rotary
}

public func Qwen3_5VisionPositionEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  weight positionEmbeddingWeight: AnyTensor, gridThw: [(t: Int, h: Int, w: Int)],
  configuration: Qwen3_5VisionConfiguration = .qwen3_5_4B,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let weight = Tensor<Float>(from: positionEmbeddingWeight)
  let tokenLength = gridThw.reduce(0) { $0 + $1.t * $1.h * $1.w }
  let side = Int(Double(configuration.positionEmbeddings).squareRoot())
  let merge = configuration.spatialMergeSize
  var embedding = Tensor<FloatType>(.CPU, .WC(tokenLength, configuration.hiddenSize))
  var tokenIndex = 0
  for grid in gridThw {
    for _ in 0..<grid.t {
      for blockY in 0..<(grid.h / merge) {
        for blockX in 0..<(grid.w / merge) {
          for intraY in 0..<merge {
            for intraX in 0..<merge {
              let y = blockY * merge + intraY
              let x = blockX * merge + intraX
              let yPosition = grid.h == 1 ? 0 : Double(y) * Double(side - 1) / Double(grid.h - 1)
              let xPosition = grid.w == 1 ? 0 : Double(x) * Double(side - 1) / Double(grid.w - 1)
              let yFloor = Int(yPosition)
              let xFloor = Int(xPosition)
              let yCeil = min(yFloor + 1, side - 1)
              let xCeil = min(xFloor + 1, side - 1)
              let dy = Float(yPosition - Double(yFloor))
              let dx = Float(xPosition - Double(xFloor))
              let indices = [
                yFloor * side + xFloor, yFloor * side + xCeil, yCeil * side + xFloor,
                yCeil * side + xCeil,
              ]
              let weights = [
                (1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx,
              ]
              for channel in 0..<configuration.hiddenSize {
                var value: Float = 0
                for i in 0..<4 {
                  value += weight[indices[i], channel] * weights[i]
                }
                embedding[tokenIndex, channel] = FloatType(value)
              }
              tokenIndex += 1
            }
          }
        }
      }
    }
  }
  return embedding
}

public func Qwen3_5VisionSequenceOffsets(gridThw: [(t: Int, h: Int, w: Int)]) -> (
  offsets: Tensor<Int32>, maxSequenceLength: Int
) {
  var offsets = [Int32]()
  offsets.reserveCapacity(gridThw.reduce(1) { $0 + $1.t })
  offsets.append(0)
  var cursor: Int32 = 0
  var maxSequenceLength = 0
  for grid in gridThw {
    let length = grid.h * grid.w
    maxSequenceLength = max(maxSequenceLength, length)
    for _ in 0..<grid.t {
      cursor += Int32(length)
      offsets.append(cursor)
    }
  }
  return (
    Tensor<Int32>(offsets, kind: .CPU, format: .NHWC, shape: [offsets.count]),
    maxSequenceLength
  )
}

private func Qwen3_5VisionAttention(
  prefix: String, configuration: Qwen3_5VisionConfiguration, tokenLength: Int,
  maxSequenceLength: Int, attentionDataType: DataType,
  x: Model.IO, rotary: Model.IO, sequenceOffsets: Model.IO
) -> Model.IO {
  let heads = configuration.heads
  let headDim = configuration.headDim
  let hiddenSize = configuration.hiddenSize
  let toqueries = Dense(count: hiddenSize, flags: [.Float16], name: "\(prefix).attn.q_proj")
  let tokeys = Dense(count: hiddenSize, flags: [], name: "\(prefix).attn.k_proj")
  let tovalues = Dense(count: hiddenSize, flags: [], name: "\(prefix).attn.v_proj")
  var queries = toqueries(x).reshaped([1, tokenLength, heads, headDim])
  var keys = tokeys(x).reshaped([1, tokenLength, heads, headDim])
  let values = tovalues(x).reshaped([1, tokenLength, heads, headDim]).to(attentionDataType)
  queries = Functional.cmul(left: queries, right: rotary).to(attentionDataType)
  keys = Functional.cmul(left: keys, right: rotary).to(attentionDataType)
  let attentionOut = ScaledDotProductAttention(
    scale: 1.0 / Float(headDim).squareRoot(), isVariableLength: true,
    maxSequenceLength: (query: maxSequenceLength, keyValue: maxSequenceLength),
    name: "\(prefix).attn.sdpa"
  )([queries, keys, values, sequenceOffsets, sequenceOffsets]).reshaped([
    tokenLength, hiddenSize,
  ])
  let outProj = Dense(count: hiddenSize, flags: [], name: "\(prefix).attn.proj")
  let out = outProj(attentionOut.to(of: x))
  return out
}

private func Qwen3_5VisionBlock(
  prefix: String, configuration: Qwen3_5VisionConfiguration, tokenLength: Int,
  maxSequenceLength: Int, attentionDataType: DataType,
  x: Model.IO, rotary: Model.IO, sequenceOffsets: Model.IO
) -> Model.IO {
  let norm1 = LayerNorm(
    epsilon: configuration.layerNormEpsilon, axis: [1], name: "\(prefix).norm1")
  let attention = Qwen3_5VisionAttention(
    prefix: prefix, configuration: configuration, tokenLength: tokenLength,
    maxSequenceLength: maxSequenceLength, attentionDataType: attentionDataType,
    x: norm1(x), rotary: rotary,
    sequenceOffsets: sequenceOffsets)
  var out = x + attention
  let norm2 = LayerNorm(
    epsilon: configuration.layerNormEpsilon, axis: [1], name: "\(prefix).norm2")
  let mlpInput = norm2(out)
  let fc1 = Dense(
    count: configuration.intermediateSize, flags: [.Float16], name: "\(prefix).mlp.linear_fc1")
  let fc2 = Dense(count: configuration.hiddenSize, flags: [], name: "\(prefix).mlp.linear_fc2")
  out = out + fc2(fc1(mlpInput).GELU(approximate: .tanh))
  return out
}

private func Qwen3_5VisionMerger<T: TensorNumeric>(
  _ dataType: T.Type, tokenLength: Int, configuration: Qwen3_5VisionConfiguration,
  x: Model.IO
) -> (normOut: Model.IO, out: Model.IO) {
  let mergerNorm = LayerNorm(
    epsilon: configuration.layerNormEpsilon, axis: [1], name: "model.visual.merger.norm")
  let mergedTokenLength =
    tokenLength / (configuration.spatialMergeSize * configuration.spatialMergeSize)
  let normOut = mergerNorm(x).reshaped([
    mergedTokenLength, configuration.mergedHiddenSize,
  ])
  let mergerFc1 = Dense(
    count: configuration.mergedHiddenSize, flags: [.Float16], name: "model.visual.merger.linear_fc1"
  )
  let mergerFc2 = Dense(
    count: configuration.outputHiddenSize, flags: [], name: "model.visual.merger.linear_fc2")
  let out = mergerFc2(mergerFc1(normOut).GELU())
  return (normOut, out)
}

public func Qwen3_5VisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type, gridThw: [(t: Int, h: Int, w: Int)],
  configuration: Qwen3_5VisionConfiguration, deepStackLayers: [Int] = [],
  attentionDataType: DataType = .BFloat16
) -> Model {
  let tokenLength = gridThw.reduce(0) { $0 + $1.t * $1.h * $1.w }
  let maxSequenceLength = Qwen3_5VisionSequenceOffsets(gridThw: gridThw).maxSequenceLength
  let patches = Input()
  let positionEmbedding = Input()
  let rotary = Input()
  let sequenceOffsets = Input()
  let patchEmbed = Dense(
    count: configuration.hiddenSize, name: "model.visual.patch_embed.proj")
  var out = patchEmbed(patches)
  out = out + positionEmbedding.to(T.dataType)
  var deepStack = [Model.IO]()
  for i in 0..<configuration.layers {
    let block = Qwen3_5VisionBlock(
      prefix: "model.visual.blocks.\(i)", configuration: configuration, tokenLength: tokenLength,
      maxSequenceLength: maxSequenceLength, attentionDataType: attentionDataType,
      x: out, rotary: rotary,
      sequenceOffsets: sequenceOffsets)
    out = block
    if let index = deepStackLayers.firstIndex(of: i) {
      let prefix = "model.visual.deepstack_merger_list.\(index)"
      // Qwen3-VL DeepStack normalizes after the spatial merge, unlike the final merger.
      let norm = LayerNorm(
        epsilon: configuration.layerNormEpsilon, axis: [1], name: "\(prefix).norm")
      let merged = norm(
        out.reshaped([
          tokenLength / (configuration.spatialMergeSize * configuration.spatialMergeSize),
          configuration.mergedHiddenSize,
        ]))
      let fc1 = Dense(
        count: configuration.mergedHiddenSize, flags: [.Float16], name: "\(prefix).linear_fc1")
      let fc2 = Dense(count: configuration.outputHiddenSize, name: "\(prefix).linear_fc2")
      deepStack.append(fc2(fc1(merged).GELU()))
    }
  }
  let merger = Qwen3_5VisionMerger(
    T.self, tokenLength: tokenLength, configuration: configuration, x: out)
  out = merger.out
  return Model([patches, positionEmbedding, rotary, sequenceOffsets], [out] + deepStack)
}
