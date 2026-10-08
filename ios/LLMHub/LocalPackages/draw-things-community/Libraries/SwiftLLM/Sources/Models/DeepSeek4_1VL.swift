import Foundation
import NNC

public struct DeepSeek4_1VisionConfiguration: Sendable {
  public var hiddenSize = 1024
  public var intermediateSize = 2816
  public var outputHiddenSize = 5120
  public var layers = 32
  public var heads = 16
  public var patchSize = 14
  public var spatialMergeSize = 3
  public var minimumPixels = 295_936
  public var maximumImageTokens = 1024
  public var normEpsilon: Float = 1e-6
  public var ropeTheta: Double = 10_000
  public var headDim: Int { hiddenSize / heads }
  public init() {}
  public static let deepSeekV4_1Flash = DeepSeek4_1VisionConfiguration()
}

/// Reference image_processor.py grid planning. The token budget includes image markers.
public func DeepSeek4_1VisionGrid(
  width: Int, height: Int,
  configuration: DeepSeek4_1VisionConfiguration = .deepSeekV4_1Flash
) -> (height: Int, width: Int) {
  precondition(width > 0 && height > 0)
  let p = configuration.patchSize
  let r = configuration.spatialMergeSize
  let budget = configuration.maximumImageTokens
  var w = width
  var h = height
  if Double(w) * Double(h) < Double(configuration.minimumPixels) {
    let scale = sqrt(Double(configuration.minimumPixels) / (Double(w) * Double(h)))
    w = max(1, Int(Double(w) * scale))
    h = max(1, Int(Double(h) * scale))
  }
  var pw = (w + p - 1) / p
  var ph = (h + p - 1) / p
  if ((ph + r - 1) / r) * ((pw + r - 1) / r + 1) + 2 > budget {
    let ratio = Double(h) / Double(w)
    let maxWidth = sqrt(Double(budget - 2) / ratio + 0.25) - 0.5
    let maxHeight = maxWidth * ratio
    if maxWidth < 1 {
      ph = (budget - 2) / 2 * r
      pw = r
    } else if maxHeight < 1 {
      ph = r
      pw = (budget - 3) * r
    } else {
      let scale = min(
        floor(maxWidth) * Double(p * r) / Double(w),
        floor(maxHeight) * Double(p * r) / Double(h))
      pw = max(1, Int(floor(Double(w) * scale / Double(p))))
      ph = max(1, Int(floor(Double(h) * scale / Double(p))))
    }
  }
  precondition(((ph + r - 1) / r) * ((pw + r - 1) / r + 1) + 2 <= budget)
  return (ph, pw)
}

public func DeepSeek4_1VisionTokenCount(
  grid: (height: Int, width: Int),
  configuration: DeepSeek4_1VisionConfiguration = .deepSeekV4_1Flash
) -> Int {
  let r = configuration.spatialMergeSize
  return ((grid.height + r - 1) / r) * ((grid.width + r - 1) / r + 1) + 2
}

public func DeepSeek4_1VisionRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  grid: (height: Int, width: Int),
  configuration: DeepSeek4_1VisionConfiguration = .deepSeekV4_1Flash,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  let d = configuration.headDim
  precondition(d.isMultiple(of: 4))
  var result = Tensor<FloatType>(.CPU, .NHWC(1, grid.height * grid.width, 1, d))
  for y in 0..<grid.height {
    for x in 0..<grid.width {
      for i in 0..<(d / 2) {
        let position = i < d / 4 ? y : x
        let angle =
          Double(position)
          / pow(
            configuration.ropeTheta,
            Double(2 * (i % (d / 4))) / Double(d / 2))
        result[0, y * grid.width + x, 0, 2 * i] = FloatType(cos(angle))
        result[0, y * grid.width + x, 0, 2 * i + 1] = FloatType(sin(angle))
      }
    }
  }
  return result
}

private func DeepSeek4_1VisionAttention(
  _ x: Model.IO, rotary: Model.IO, tokenLength: Int, prefix: String,
  configuration: DeepSeek4_1VisionConfiguration
) -> Model.IO {
  let h = configuration.hiddenSize
  let d = configuration.headDim
  let heads = configuration.heads
  let query = Dense(count: h, name: "\(prefix).wq")(x)
    .reshaped([1, tokenLength, heads, d])
  let key = Dense(count: h, name: "\(prefix).wk")(x)
    .reshaped([1, tokenLength, heads, d])
  let value = Dense(count: h, name: "\(prefix).wv")(x)
    .reshaped([1, tokenLength, heads, d])
  let attention = ScaledDotProductAttention(scale: 1 / Float(d).squareRoot())(
    Functional.cmul(left: query, right: rotary), Functional.cmul(left: key, right: rotary), value
  )
  .reshaped([tokenLength, h])
  return Dense(count: h, name: "\(prefix).wo")(attention)
}

/// Patches use [patch row, patch column, RGB channel, pixel row, pixel column] order.
/// Produces row-major projected image features; the caller inserts learned image markers.
public func DeepSeek4_1VisionTransformer(
  grid: (height: Int, width: Int),
  configuration: DeepSeek4_1VisionConfiguration = .deepSeekV4_1Flash
) -> Model {
  let patches = Input()
  let rotary = Input()
  let t = grid.height * grid.width
  let h = configuration.hiddenSize
  var x = Dense(count: h, name: "vision.patch_embed.proj")(patches).to(.Float32)
  for i in 0..<configuration.layers {
    let prefix = "vision.blocks.\(i)"
    let normalized = RMSNorm(
      epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).norm1")(x)
    x =
      x
      + DeepSeek4_1VisionAttention(
        normalized.to(.Float16), rotary: rotary,
        tokenLength: t, prefix: "\(prefix).attn", configuration: configuration
      ).to(.Float32)
    let mlp = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).norm2")(x)
      .to(.Float16)
    let hidden = SwiGLU(count: configuration.intermediateSize, name: "\(prefix).mlp")(mlp)
    x = x + Dense(count: h, noBias: true, name: "\(prefix).mlp.w2")(hidden).to(.Float32)
  }
  x = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "vision.norm")(x).to(.Float16)
  let r = configuration.spatialMergeSize
  let mh = (grid.height + r - 1) / r
  let mw = (grid.width + r - 1) / r
  x = x.reshaped([grid.height, grid.width, h])
    .padded(.zero, begin: [0, 0, 0], end: [mh * r - grid.height, mw * r - grid.width, 0])
    .reshaped([mh, r, mw, r, h]).permuted(0, 2, 4, 1, 3).contiguous()
    .reshaped([mh * mw, h * r * r])
  x = Dense(count: configuration.outputHiddenSize, name: "aligner.w1")(x).GELU()
  x = Dense(count: configuration.outputHiddenSize, name: "aligner.w2")(x)
  return Model([patches, rotary], [x])
}
