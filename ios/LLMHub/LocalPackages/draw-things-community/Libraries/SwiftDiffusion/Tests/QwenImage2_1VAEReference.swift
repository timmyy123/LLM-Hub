// Frozen NCHW implementation for NHWC checkpoint parity.
import Foundation
import NNC

private enum QwenImage2_1VAEConfiguration {
  static let encoderChannels = [96, 192, 384, 768, 768]
  static let decoderChannels = [1_152, 1_152, 576, 288, 144]
  static let latentChannels = 64
  static let pixelChannels = 4
}

struct QwenImage2_1VAEBinding {
  let model: Model
  let key: String
  var norm = false
}

// Single-image specialization of the released VAE. All convolutions are 2D;
// temporal resampling convolutions are inactive for the first/single frame.
func QwenImage2_1VAEReference(encoder: Bool, height: Int, width: Int, device: Int = 0)
  -> (Model, [QwenImage2_1VAEBinding])
{
  let input = Input()
  let pixels = input.to(.Float32).permuted(0, 3, 1, 2).contiguous().reshaped(
    .NCHW(1, encoder ? 4 : 64, height, width))
  var bindings = [QwenImage2_1VAEBinding]()
  var h = height
  var w = width
  func conv(_ key: String, _ channels: Int, kernel: Int = 3, down: Bool = false) -> Model {
    let border = kernel / 2
    let model = Convolution(
      groups: 1, filters: channels, filterSize: [kernel, kernel],
      hint: Hint(
        stride: [down ? 2 : 1, down ? 2 : 1],
        border: Hint.Border(
          begin: [down ? 0 : border, down ? 0 : border],
          end: [down ? 0 : border, down ? 0 : border])),
      name: key.replacingOccurrences(of: ".", with: "_"))
    bindings.append(.init(model: model, key: key))
    return model
  }
  func channelNorm(_ x: Model.IO, _ key: String, channels: Int) -> Model.IO {
    let gamma = Parameter<Float>(
      .GPU(device), .NCHW(1, channels, 1, 1), name: key.replacingOccurrences(of: ".", with: "_"))
    bindings.append(.init(model: gamma, key: key, norm: true))
    let sum = (x .* x).reduced(.sum, axis: [1])
    let inverse = sum.clamped(1e-24...).pow(-0.5)
    return (x .* inverse) * Float(channels).squareRoot() .* gamma()
  }
  func residual(_ x: Model.IO, _ key: String, from: Int, to: Int) -> Model.IO {
    let shortcut = from == to ? x : conv(key + ".conv_shortcut", to, kernel: 1)(x)
    let a = conv(key + ".conv1", to)(channelNorm(x, key + ".norm1", channels: from).swish())
    let b = conv(key + ".conv2", to)(channelNorm(a, key + ".norm2", channels: to).swish())
    return b + shortcut
  }
  func attention(_ x: Model.IO, _ key: String, channels: Int) -> Model.IO {
    let normalized = channelNorm(x, key + ".norm", channels: channels)
    let qkv = conv(key + ".to_qkv", 3 * channels, kernel: 1)(normalized).reshaped([
      3 * channels, h * w,
    ])
    let chunks = (0..<3).map { i in
      qkv.reshaped([channels, h * w], offset: [i * channels, 0], strides: [h * w, 1]).contiguous()
    }
    let scores =
      Matmul(transposeA: (0, 1))(chunks[0], chunks[1]) * (1 / Float(channels).squareRoot())
    let probabilities = scores.softmax()
    let attended = Matmul(transposeB: (0, 1))(chunks[2], probabilities).reshaped([
      1, channels, h, w,
    ])
    return x + conv(key + ".proj", channels, kernel: 1)(attended)
  }
  func middle(_ x: Model.IO, _ key: String, channels: Int) -> Model.IO {
    let a = residual(x, key + ".resnets.0", from: channels, to: channels)
    let b = attention(a, key + ".attentions.0", channels: channels)
    return residual(b, key + ".resnets.1", from: channels, to: channels)
  }
  func downShortcut(_ x: Model.IO, from: Int, to: Int, temporal: Int, spatial: Int) -> Model.IO {
    let nh = h / spatial
    let nw = w / spatial
    var folded = x.reshaped([from, nh, spatial, nw, spatial]).permuted(0, 2, 4, 1, 3).contiguous()
      .reshaped([from, 1, spatial * spatial, nh * nw])
    if temporal == 2 { folded = Functional.concat(axis: 1, folded * 0, folded) }
    let group = from * temporal * spatial * spatial / to
    return folded.reshaped([1, to, group, nh, nw]).reduced(.mean, axis: [2]).reshaped([
      1, to, nh, nw,
    ])
  }
  func upShortcut(_ x: Model.IO, from: Int, to: Int, temporal: Int) -> Model.IO {
    let repeats = to * temporal * 4 / from
    let row = x.reshaped([from, 1, h, w])
    var duplicated = row
    if repeats > 1 {
      for _ in 1..<repeats { duplicated = Functional.concat(axis: 1, duplicated, row) }
    }
    // First-image semantics: keep only the last of the duplicated temporal frames.
    let selected = duplicated.reshaped([to, temporal, 4, h * w])
      .reshaped(
        [to, 1, 4, h * w], offset: [0, temporal - 1, 0, 0],
        strides: [temporal * 4 * h * w, 4 * h * w, h * w, 1]
      ).contiguous()
    return selected.reshaped([to, 2, 2, h, w]).permuted(0, 3, 1, 4, 2).contiguous().reshaped([
      1, to, h * 2, w * 2,
    ])
  }
  var x: Model.IO
  if encoder {
    x = conv("encoder.conv_in", 96)(pixels)
    var channels = 96
    for (stage, next) in QwenImage2_1VAEConfiguration.encoderChannels.enumerated() {
      let key = "encoder.down_blocks.\(stage)"
      let down = stage < 4
      let shortcut = downShortcut(
        x, from: channels, to: next, temporal: stage > 0 && stage < 4 ? 2 : 1, spatial: down ? 2 : 1
      )
      for block in 0..<2 {
        x = residual(x, key + ".resnets.\(block)", from: block == 0 ? channels : next, to: next)
      }
      if down {
        x = conv(key + ".downsampler.resample.1", next, down: true)(
          x.padded(.zero, begin: [0, 0, 0, 0], end: [0, 0, 1, 1]))
      }
      x = x + shortcut
      channels = next
      if down {
        h /= 2
        w /= 2
      }
    }
    x = middle(x, "encoder.mid_block", channels: channels)
    x = conv("encoder.conv_out", 128)(
      channelNorm(x, "encoder.norm_out", channels: channels).swish())
    x = conv("quant_conv", 128, kernel: 1)(x)
  } else {
    x = conv("post_quant_conv", 64, kernel: 1)(pixels)
    x = conv("decoder.conv_in", 1_152)(x)
    x = middle(x, "decoder.mid_block", channels: 1_152)
    var channels = 1_152
    for (stage, next) in QwenImage2_1VAEConfiguration.decoderChannels.enumerated() {
      let key = "decoder.up_blocks.\(stage)"
      let up = stage < 4
      let shortcut = up ? upShortcut(x, from: channels, to: next, temporal: stage < 3 ? 2 : 1) : nil
      for block in 0..<3 {
        x = residual(x, key + ".resnets.\(block)", from: block == 0 ? channels : next, to: next)
      }
      if up {
        x = conv(key + ".upsampler.resample.1", next)(
          Upsample(.nearest, widthScale: 2, heightScale: 2)(x))
        x = x + shortcut!
        h *= 2
        w *= 2
      }
      channels = next
    }
    x = conv("decoder.conv_out", 4)(channelNorm(x, "decoder.norm_out", channels: channels).swish())
  }
  return (
    Model(
      [input],
      [
        x.permuted(0, 2, 3, 1).contiguous().reshaped(.NHWC(1, h, w, encoder ? 128 : 4)).to(
          of: input)
      ]), bindings
  )
}
