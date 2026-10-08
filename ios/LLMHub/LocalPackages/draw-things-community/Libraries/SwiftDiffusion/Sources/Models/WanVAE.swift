import DiffusionMappings
import NNC

private struct NHWCResnetBlockCausal3D {
  private let norm1: Model
  private let conv1: Model
  private let norm2: Model
  private let conv2: Model
  private let ninShortcut: Model?
  init(outChannels: Int, shortcut: Bool, startDepth: Int) {
    norm1 = RMSNorm(epsilon: 1e-6, axis: [3], name: "resnet_norm1")
    if startDepth > 1 {
      conv1 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "resnet_conv1")
    } else {
      conv1 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
        format: .OIHW, name: "resnet_conv1")
    }
    norm2 = RMSNorm(epsilon: 1e-6, axis: [3], name: "resnet_norm2")
    if startDepth > 1 {
      conv2 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "resnet_conv2")
    } else {
      conv2 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
        format: .OIHW, name: "resnet_conv2")
    }
    if shortcut {
      ninShortcut = Convolution(
        groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
        format: .OIHW, name: "resnet_shortcut")
    } else {
      ninShortcut = nil
    }
  }
  private var conv1Inputs: Model.IO? = nil
  private var conv2Inputs: Model.IO? = nil
  mutating func callAsFunction(
    input x: Model.IO, prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool,
    depth: Int, height: Int, width: Int, inputsOnly: Bool
  ) -> (
    ModelWeightMapper, Model.IO
  ) {
    var out = norm1(x.reshaped([depth, height, width, inChannels]))
    var pre = out.swish()
    if let conv1Inputs = conv1Inputs {
      out = conv1(
        Functional.concat(axis: 0, conv1Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]
        ).reshaped([
          1, depth + 2, height + 2, width + 2, inChannels,
        ]))
    } else {
      if depth > 1 {
        out = conv1(
          pre.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
            1, depth + 2, height + 2, width + 2, inChannels,
          ]))
      } else {
        out = conv1(pre)
      }
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [2, height, width, inChannels], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * inChannels, width * inChannels, inChannels, 1]
      ).copied()
      conv1Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
    } else {
      conv1Inputs = nil
    }
    out = norm2(out.reshaped([depth, height, width, outChannels]))
    pre = out.swish()
    if let conv2Inputs = conv2Inputs {
      out = conv2(
        Functional.concat(axis: 0, conv2Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]
        ).reshaped([
          1, depth + 2, height + 2, width + 2, outChannels,
        ]))
    } else {
      if depth > 1 {
        out = conv2(
          pre.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
            1, depth + 2, height + 2, width + 2, outChannels,
          ]))
      } else {
        out = conv2(pre)
      }
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [2, height, width, outChannels], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * outChannels, width * outChannels, outChannels, 1]
      ).copied()
      conv2Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
    } else {
      conv2Inputs = nil
    }
    out = out.reshaped([depth, height, width, outChannels])
    if let ninShortcut = ninShortcut {
      out = ninShortcut(x) + out
    } else {
      out = x + out
    }
    let mapper: ModelWeightMapper = { [norm1, conv1, norm2, conv2, ninShortcut] _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).residual.0.gamma"] = [norm1.weight.name]
      mapping["\(prefix).residual.2.weight"] = [conv1.weight.name]
      mapping["\(prefix).residual.2.bias"] = [conv1.bias.name]
      mapping["\(prefix).residual.3.gamma"] = [norm2.weight.name]
      mapping["\(prefix).residual.6.weight"] = [conv2.weight.name]
      mapping["\(prefix).residual.6.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["\(prefix).shortcut.weight"] = [ninShortcut.weight.name]
        mapping["\(prefix).shortcut.bias"] = [ninShortcut.bias.name]
      }
      return mapping
    }
    return (mapper, out)
  }
}

private struct NHWCAttnBlockCausal3D {
  private let norm: Model
  private let toqueries: Model
  private let tokeys: Model
  private let tovalues: Model
  private let projOut: Model
  init(inChannels: Int) {
    norm = RMSNorm(epsilon: 1e-6, axis: [3], name: "attn_norm")
    tokeys = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "to_k")
    toqueries = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "to_q")
    tovalues = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "to_v")
    projOut = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "proj_out")
  }
  func callAsFunction(
    prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
  ) -> (
    ModelWeightMapper, Model
  ) {
    let x = Input()
    var out = norm(x.reshaped([depth, height, width, inChannels]))
    let hw = width * height
    let k = tokeys(out).reshaped([depth, hw, inChannels])
    let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
      depth, hw, inChannels,
    ])
    var dot =
      Matmul(transposeB: (1, 2))(q, k)
    dot = dot.reshaped([depth * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([depth, hw, hw])
    let v = tovalues(out).reshaped([depth, hw, inChannels])
    out = dot * v
    out = x + projOut(out.reshaped([depth, height, width, inChannels]))
    let mapper: ModelWeightMapper = { [norm, toqueries, tokeys, tovalues, projOut] _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).norm.gamma"] = [norm.weight.name]
      mapping["\(prefix).to_qkv.weight"] = [
        toqueries.weight.name, tokeys.weight.name, tovalues.weight.name,
      ]
      mapping["\(prefix).to_qkv.bias"] = [
        toqueries.bias.name, tokeys.bias.name, tovalues.bias.name,
      ]
      mapping["\(prefix).proj.weight"] = [projOut.weight.name]
      mapping["\(prefix).proj.bias"] = [projOut.bias.name]
      return mapping
    }
    return (mapper, Model([x], [out]))
  }
}

// Wan 2.2 VAE is a 16x VAE with some subtle differences with Wan 2.1 VAE (used by Wan 2.2 A14B, Wan 2.1 1.3B, Wan 2.1 14B, Qwen Image).
// 1. The channel is not halved at the resample convolution time (Wan 2.2);
// 2. The input is 48-channel, output is 12-channel and pixel shuffle back to 2x;
// 3. There is a separate upsample shortcut such that the higher resolution res blocks should only learn high-frequency information.
private func NHWCWanDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool, wan22: Bool, outputChannels: Int,
  highPrecisionFinalNorm: Bool
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let inputChannels = wan22 ? 48 : 16
  let postQuantConv = Convolution(
    groups: 1, filters: inputChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "post_quant_conv")
  let postQuantX = postQuantConv(x)
  let convIn: Model
  let convOut: Model
  if startDepth > 1 {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_in")
    convOut = Convolution(
      groups: 1,
      filters: wan22 ? 12 : (paddingFinalConvLayer ? (outputChannels + 3) / 4 * 4 : outputChannels),
      filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_out")
  } else {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_in")
    convOut = Convolution(
      groups: 1,
      filters: wan22 ? 12 : (paddingFinalConvLayer ? (outputChannels + 3) / 4 * 4 : outputChannels),
      filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_out")
  }
  let normOut = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_out")
  var last: Model.IO? = nil
  var midBlock1Builder = NHWCResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  let midAttn1Builder = NHWCAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NHWCResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  var upBlockBuilders = [NHWCResnetBlockCausal3D]()
  for (i, channel) in channels.enumerated().reversed() {
    for _ in 0..<numRepeat + 1 {
      upBlockBuilders.append(
        NHWCResnetBlockCausal3D(
          outChannels: channel, shortcut: previousChannel != channel, startDepth: startDepth))
      previousChannel = channel
    }
    if i > 0 && !wan22 {
      previousChannel = channel / 2
    }
  }
  let timeConvs = (0..<(channels.count - 2)).map { i in
    return Convolution(
      groups: 1, filters: channels[channels.count - i - 1] * 2, filterSize: [3, 1, 1],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "time_conv")
  }
  let upsampleConv2d = (0..<(channels.count - 1)).map { i in
    let channels = wan22 ? channels[channels.count - i - 1] : channels[channels.count - i - 1] / 2
    return Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "upsample")
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var convOutInputs: Model.IO? = nil
  var outs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[channels.count - 1]
    var out: Model.IO
    if d == 0 {
      out = postQuantX.reshaped(
        [min(startDepth - d, 3), startHeight, startWidth, inputChannels], offset: [d, 0, 0, 0],
        strides: [
          startHeight * startWidth * inputChannels, startWidth * inputChannels, inputChannels, 1,
        ]
      ).contiguous()
      if startDepth > 1 {
        out = convIn(
          out.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
            1, min(startDepth - d, 3) + 2, startHeight + 2, startWidth + 2, inputChannels,
          ])
        ).reshaped([min(startDepth - d, 3), startHeight, startWidth, previousChannel])
      } else {
        out = convIn(out)
      }
    } else {
      out = postQuantX.reshaped(
        [min(startDepth - (d - 1), 4), startHeight, startWidth, inputChannels],
        offset: [d - 1, 0, 0, 0],
        strides: [
          startHeight * startWidth * inputChannels, startWidth * inputChannels, inputChannels, 1,
        ]
      ).contiguous().padded(.zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
      if let last = last {
        out.add(dependencies: [last])
      }
      out = convIn(
        out.reshaped([
          1, min(startDepth - (d - 1), 4), startHeight + 2, startWidth + 2, inputChannels,
        ])
      ).reshaped([min(startDepth - (d - 1), 4) - 2, startHeight, startWidth, previousChannel])
    }
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var width = startWidth
    var height = startHeight
    var depth = d > 0 ? min(startDepth - 1 - d, 2) : min(startDepth, 3)
    let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "decoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper1)
    out = midBlock1Out
    let (midAttnMapper1, midAttn1) = midAttn1Builder(
      prefix: "decoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    mappers.append(midAttnMapper1)
    out = midAttn1(out)
    let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "decoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper2)
    out = midBlock2Out
    var j = 0
    var k = 0
    var upShortcut: Model.IO? = nil
    for (i, channel) in channels.enumerated().reversed() {
      if i > 0 && wan22 {
        var shortcut = out
        if i > 1 {
          // Need to do temporal upscaling.
          shortcut = shortcut.reshaped([depth, height, width, previousChannel])
          if previousChannel != channel {
            shortcut = Functional.concat(
              axis: 3, shortcut, shortcut, shortcut, shortcut, flags: [.disableOpt])
            shortcut = shortcut.reshaped([depth, height, width, 2, 2, channel, 2]).permuted(
              0, 6, 1, 3, 2, 4, 5
            ).copied().reshaped([depth * 2, height * 2, width * 2, channel]).copied()
          } else {
            shortcut = Functional.concat(
              axis: 3, shortcut, shortcut, shortcut, shortcut, shortcut, shortcut, shortcut,
              shortcut, flags: [.disableOpt])
            shortcut = shortcut.reshaped([depth, height, width, 2, 2, 2, channel]).permuted(
              0, 3, 1, 4, 2, 5, 6
            ).copied().reshaped([depth * 2, height * 2, width * 2, channel]).copied()
          }
          if d == 0 {
            shortcut = shortcut.reshaped(
              [(depth - 1) * 2 + 1, height * 2, width * 2, channel], offset: [1, 0, 0, 0],
              strides: [height * 2 * width * 2 * channel, width * 2 * channel, channel, 1]
            ).contiguous().reshaped([(depth - 1) * 2 + 1, height * 2, width * 2, channel])
          }
        } else {
          shortcut = shortcut.reshaped([depth, height, width, previousChannel])
          if previousChannel != channel {
            shortcut = Functional.concat(axis: 3, shortcut, shortcut, flags: [.disableOpt])
            shortcut = shortcut.reshaped([depth, height, width, 2, channel, 2]).permuted(
              0, 1, 5, 2, 3, 4
            ).copied().reshaped([depth, height * 2, width * 2, channel])
          } else {
            shortcut = Functional.concat(
              axis: 3, shortcut, shortcut, shortcut, shortcut, flags: [.disableOpt])
            shortcut = shortcut.reshaped([depth, height, width, 2, 2, channel]).permuted(
              0, 1, 3, 2, 4, 5
            ).copied().reshaped([depth, height * 2, width * 2, channel])
          }
        }
        upShortcut = shortcut
      } else {
        upShortcut = nil
      }
      for _ in 0..<numRepeat + 1 {
        let (mapper, blockOut) = upBlockBuilders[j](
          input: out,
          prefix: "decoder.upsamples.\(k)",
          inChannels: previousChannel, outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        mappers.append(mapper)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i > 0 {
        if i > 1 && startDepth > 1 {  // Need to bump up on the depth axis.
          if d == 0 {  // Special case for first frame.
            let first = out.reshaped(
              [1, height, width, channel],
              strides: [height * width * channel, width * channel, channel, 1]
            ).contiguous()
            let more = out.reshaped(
              [(depth - 1), height, width, channel], offset: [1, 0, 0, 0],
              strides: [height * width * channel, width * channel, channel, 1]
            ).contiguous()
            let timeConv = timeConvs[channels.count - i - 1]
            var expanded = timeConv(
              more.padded(.zero, begin: [2, 0, 0, 0], end: [0, 0, 0, 0]).reshaped([
                1, depth + 1, height, width, channel,
              ]))
            if !inputsOnly {
              let inputs = out.reshaped(
                [2, height, width, channel], offset: [depth - 2, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              var mapping = ModelWeightMapping()
              mapping["decoder.upsamples.\(upLayer).time_conv.weight"] = [timeConv.weight.name]
              mapping["decoder.upsamples.\(upLayer).time_conv.bias"] = [timeConv.bias.name]
              return mapping
            }
            mappers.append(mapper)
            expanded = expanded.reshaped([depth - 1, height, width, 2, channel]).permuted(
              0, 3, 1, 2, 4
            )
            .contiguous().reshaped([2 * (depth - 1), height, width, channel])
            out = Functional.concat(axis: 0, first, expanded)
            depth = 1 + (depth - 1) * 2
            out = out.reshaped([depth, height, width, channel])
          } else if let timeInput = timeInputs[channels.count - i - 1] {
            let more = out.reshaped([depth, height, width, channel])
            let timeConv = timeConvs[channels.count - i - 1]
            let expanded = timeConv(
              Functional.concat(axis: 0, timeInput, more, flags: [.disableOpt]).reshaped([
                1, depth + 2, height, width, channel,
              ]))
            if !inputsOnly {
              let inputs = out.reshaped(
                [2, height, width, channel], offset: [depth - 2, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              var mapping = ModelWeightMapping()
              mapping["decoder.upsamples.\(upLayer).time_conv.weight"] = [timeConv.weight.name]
              mapping["decoder.upsamples.\(upLayer).time_conv.bias"] = [timeConv.bias.name]
              return mapping
            }
            mappers.append(mapper)
            out = expanded.reshaped([depth, height, width, 2, channel]).permuted(0, 3, 1, 2, 4)
              .contiguous().reshaped([2 * depth, height, width, channel])
            depth = depth * 2
          }
        }
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
          out.reshaped([depth, height, width, channel])
        ).reshaped([depth, height * 2, width * 2, channel])
        width *= 2
        height *= 2
        let conv2d = upsampleConv2d[channels.count - i - 1]
        out = conv2d(out)
        if let upShortcut = upShortcut {
          out = upShortcut + out
        }
        if !wan22 {
          previousChannel = channel / 2
        }
        let upLayer = k
        let mapper: ModelWeightMapper = { _ in
          var mapping = ModelWeightMapping()
          mapping["decoder.upsamples.\(upLayer).resample.1.weight"] = [conv2d.weight.name]
          mapping["decoder.upsamples.\(upLayer).resample.1.bias"] = [conv2d.bias.name]
          return mapping
        }
        mappers.append(mapper)
        k += 1
      }
    }
    let beforeNorm = out
    if highPrecisionFinalNorm {
      out = out.to(.Float32)
    }
    out = normOut(out.reshaped([depth, height, width, channels[0]]))
    if highPrecisionFinalNorm {
      out = out.to(of: beforeNorm)
    }
    let pre = out.swish()
    if let convOutInputs = convOutInputs {
      out = convOut(
        Functional.concat(axis: 0, convOutInputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]
        ).reshaped([
          1, depth + 2, height + 2, width + 2, channels[0],
        ])
      )
    } else {
      if startDepth > 1 {
        out = convOut(
          pre.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
            1, depth + 2, height + 2, width + 2, channels[0],
          ])
        )
      } else {
        out = convOut(pre)
      }
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [2, height, width, channels[0]], offset: [depth - 2, 0, 0, 0],
        strides: [height * width * channels[0], width * channels[0], channels[0], 1]
      ).copied()
      convOutInputs = inputs
      out.add(dependencies: [inputs])
    }
    last = out
    outs.append(
      out.reshaped([
        depth, height, width,
        wan22 ? 12 : (paddingFinalConvLayer ? (outputChannels + 3) / 4 * 4 : outputChannels),
      ]))
  }
  var out: Model.IO
  if outs.count > 1 {
    out = Concat(axis: 0)(outs)
  } else {
    out = outs[0]
  }
  if wan22 {
    out = out.reshaped([(startDepth - 1) * 4 + 1, startHeight * 8, startWidth * 8, 3, 2, 2])
      .permuted(0, 1, 4, 2, 5, 3).copied().reshaped([
        (startDepth - 1) * 4 + 1, startHeight * 16, startWidth * 16, 3,
      ])
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv2.weight"] = [postQuantConv.weight.name]
    mapping["conv2.bias"] = [postQuantConv.bias.name]
    mapping["decoder.conv1.weight"] = [convIn.weight.name]
    mapping["decoder.conv1.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["decoder.head.0.gamma"] = [normOut.weight.name]
    mapping["decoder.head.2.weight"] = [convOut.weight.name]
    mapping["decoder.head.2.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NHWCWanEncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int, wan22: Bool,
  inputChannels: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn: Model
  if startDepth > 1 {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_in")
  } else {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_in")
  }
  var mappers = [ModelWeightMapper]()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  var downBlockBuilders = [NHWCResnetBlockCausal3D]()
  var timeConvs = [Convolution]()
  var downsampleConv2d = [Convolution]()
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      downBlockBuilders.append(
        NHWCResnetBlockCausal3D(
          outChannels: channel, shortcut: previousChannel != channel, startDepth: startDepth))
      previousChannel = channel
    }
    if i < channels.count - 1 {
      downsampleConv2d.append(
        Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(
            stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])),
          format: .OIHW, name: "downsample"))
      if i > 0 && startDepth > 1 {
        timeConvs.append(
          Convolution(
            groups: 1, filters: channel, filterSize: [3, 1, 1],
            hint: Hint(
              stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
            format: .OIHW, name: "time_conv"))
      }
    }
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var midBlock1Builder = NHWCResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  let midAttn1Builder = NHWCAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NHWCResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  let normOut = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_out")
  let endHeight = height
  let endWidth = width
  let endDepth = depth
  var outs = [Model.IO]()
  let input: Model.IO
  if wan22 {
    input = x.reshaped([endDepth, endHeight, 2, endWidth, 2, 3]).permuted(0, 1, 3, 5, 2, 4)
      .contiguous().reshaped(.NHWC(endDepth, endHeight, endWidth, 3 * 2 * 2))
  } else {
    input = x
  }
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[0]
    height = endHeight
    width = endWidth
    var out: Model.IO
    if d == 0 {
      if startDepth > 1 {
        depth = min(endDepth, 9)
        out = input.reshaped(
          [depth, height, width, inputChannels],
          strides: [height * width * inputChannels, width * inputChannels, inputChannels, 1]
        ).contiguous().padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0])
        out = convIn(out.reshaped([1, depth + 2, height + 2, width + 2, inputChannels])).reshaped([
          depth, height, width, previousChannel,
        ])
      } else {
        out = convIn(input)
      }
    } else {
      depth = min(endDepth - (d * 4 + 1), 8)
      out = input.reshaped(
        [depth + 2, height, width, inputChannels], offset: [d * 4 - 1, 0, 0, 0],
        strides: [height * width * inputChannels, width * inputChannels, inputChannels, 1]
      ).contiguous().padded(.zero, begin: [0, 1, 1, 0], end: [0, 1, 1, 0])
      if let last = outs.last {
        out.add(dependencies: [last])
      }
      out = convIn(out.reshaped([1, depth + 2, height + 2, width + 2, inputChannels])).reshaped([
        depth, height, width, previousChannel,
      ])
    }
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var j = 0
    var k = 0
    for (i, channel) in channels.enumerated() {
      var downShortcut = out
      if i < channels.count - 1 && wan22 {
        if i > 0 && depth > 1 {
          let pad = (2 - depth % 2) % 2
          downShortcut = downShortcut.padded(.zero, begin: [pad, 0, 0, 0], end: [0, 0, 0, 0])
          let paddedDepth = depth + pad
          downShortcut = downShortcut.reshaped([
            paddedDepth / 2, 2, height / 2, 2, width / 2, 2, previousChannel,
          ]).permuted(0, 2, 4, 6, 1, 3, 5).copied()
          downShortcut = downShortcut.reshaped([
            paddedDepth / 2, height / 2, width / 2, channel, 8 * previousChannel / channel,
          ]).reduced(.mean, axis: [4]).reshaped([paddedDepth / 2, height / 2, width / 2, channel])
        } else {
          downShortcut = downShortcut.reshaped([
            depth, height / 2, 2, width / 2, 2, previousChannel,
          ])
          .permuted(0, 1, 3, 5, 2, 4).copied()
          downShortcut = downShortcut.reshaped([
            depth, height / 2, width / 2, channel, 4 * previousChannel / channel,
          ]).reduced(.mean, axis: [4]).reshaped([depth, height / 2, width / 2, channel])
        }
      }
      for _ in 0..<numRepeat {
        let (mapper, blockOut) = downBlockBuilders[j](
          input: out,
          prefix: "encoder.downsamples.\(k)", inChannels: previousChannel,
          outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        mappers.append(mapper)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i < channels.count - 1 {
        height /= 2
        width /= 2
        let conv2d = downsampleConv2d[i]
        out = conv2d(out.padded(.zero, begin: [0, 0, 0, 0], end: [0, 1, 1, 0]))
        let downLayer = k
        let mapper: ModelWeightMapper = { _ in
          var mapping = ModelWeightMapping()
          mapping[
            "encoder.downsamples.\(downLayer).resample.1.weight"
          ] = [conv2d.weight.name]
          mapping["encoder.downsamples.\(downLayer).resample.1.bias"] = [conv2d.bias.name]
          return mapping
        }
        mappers.append(mapper)
        if i > 0 && startDepth > 1 {
          if d == 0 {
            let first = out.reshaped(
              [1, height, width, channel],
              strides: [height * width * channel, width * channel, channel, 1]
            ).contiguous()
            let timeConv = timeConvs[i - 1]
            let shrunk = timeConv(out.reshaped([1, depth, height, width, channel]))
              .reshaped([(depth - 1) / 2, height, width, channel])
            if !inputsOnly {
              let input = out.reshaped(
                [1, height, width, channel], offset: [depth - 1, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              var mapping = ModelWeightMapping()
              mapping["encoder.downsamples.\(upLayer).time_conv.weight"] = [timeConv.weight.name]
              mapping["encoder.downsamples.\(upLayer).time_conv.bias"] = [timeConv.bias.name]
              return mapping
            }
            mappers.append(mapper)
            depth = (depth - 1) / 2 + 1
            out = Functional.concat(axis: 0, first, shrunk)
          } else if let timeInput = timeInputs[i - 1] {
            let shrunk = timeConvs[i - 1](
              Functional.concat(axis: 0, timeInput, out, flags: [.disableOpt]).reshaped([
                1, depth + 1, height, width, channel,
              ]))
            if !inputsOnly {
              let input = out.reshaped(
                [1, height, width, channel], offset: [depth - 1, 0, 0, 0],
                strides: [height * width * channel, width * channel, channel, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            depth = depth / 2
            out = shrunk.reshaped([depth, height, width, channel])
          }
        }
        k += 1
      }
      if wan22 {
        out = downShortcut + out
      }
    }
    let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "encoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper1)
    out = midBlock1Out
    let (midAttnMapper1, midAttn1) = midAttn1Builder(
      prefix: "encoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    mappers.append(midAttnMapper1)
    out = midAttn1(out)
    let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "encoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper2)
    out = midBlock2Out
    out = normOut(out.reshaped([depth, height, width, previousChannel]))
    out = out.swish()
    outs.append(out)
  }
  height = startHeight
  width = startWidth
  depth = startDepth
  var out: Model.IO
  if outs.count > 1 {
    out = Concat(axis: 0)(outs)
  } else {
    out = outs[0]
  }
  let convOut: Model
  if startDepth > 1 {
    convOut = Convolution(
      groups: 1, filters: wan22 ? 96 : 32, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_out")
    out = convOut(
      out.padded(.zero, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
        1, depth + 2, height + 2, width + 2, previousChannel,
      ])
    ).reshaped([depth, height, width, wan22 ? 96 : 32])
  } else {
    convOut = Convolution(
      groups: 1, filters: wan22 ? 96 : 32, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_out")
    out = convOut(out)
  }
  let quantConv = Convolution(
    groups: 1, filters: wan22 ? 96 : 32, filterSize: [1, 1], format: .OIHW, name: "quant_conv")
  out = quantConv(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["encoder.conv1.weight"] = [convIn.weight.name]
    mapping["encoder.conv1.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["encoder.head.0.gamma"] = [normOut.weight.name]
    mapping["encoder.head.2.weight"] = [convOut.weight.name]
    mapping["encoder.head.2.bias"] = [convOut.bias.name]
    mapping["conv1.weight"] = [quantConv.weight.name]
    mapping["conv1.bias"] = [quantConv.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private struct NCHWResnetBlockCausal3D {
  private let norm1: Model
  private let conv1: Model
  private let norm2: Model
  private let conv2: Model
  private let ninShortcut: Model?
  init(outChannels: Int, shortcut: Bool, startDepth: Int) {
    norm1 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm1")
    if startDepth > 1 {
      conv1 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "resnet_conv1")
    } else {
      conv1 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
        format: .OIHW, name: "resnet_conv1")
    }
    norm2 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm2")
    if startDepth > 1 {
      conv2 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "resnet_conv2")
    } else {
      conv2 = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
        format: .OIHW, name: "resnet_conv2")
    }
    if shortcut {
      ninShortcut = Convolution(
        groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
        format: .OIHW, name: "resnet_shortcut")
    } else {
      ninShortcut = nil
    }
  }
  private var conv1Inputs: Model.IO? = nil
  private var conv2Inputs: Model.IO? = nil
  mutating func callAsFunction(
    input x: Model.IO, prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool,
    depth: Int, height: Int, width: Int, inputsOnly: Bool
  ) -> (
    ModelWeightMapper, Model.IO
  ) {
    var out = norm1(x.reshaped([inChannels, depth, height, width]))
    var pre = out.swish()
    if let conv1Inputs = conv1Inputs {
      out = conv1(
        Functional.concat(axis: 1, conv1Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]
        ).reshaped([
          1, inChannels, depth + 2, height + 2, width + 2,
        ]))
    } else {
      if depth > 1 {
        out = conv1(
          pre.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
            1, inChannels, depth + 2, height + 2, width + 2,
          ]))
      } else {
        out = conv1(pre.reshaped([1, inChannels, height, width]))
      }
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [inChannels, 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).copied()
      conv1Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
    } else {
      conv1Inputs = nil
    }
    out = norm2(out.reshaped([outChannels, depth, height, width]))
    pre = out.swish()
    if let conv2Inputs = conv2Inputs {
      out = conv2(
        Functional.concat(axis: 1, conv2Inputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]
        ).reshaped([
          1, outChannels, depth + 2, height + 2, width + 2,
        ]))
    } else {
      if depth > 1 {
        out = conv2(
          pre.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
            1, outChannels, depth + 2, height + 2, width + 2,
          ]))
      } else {
        out = conv2(pre.reshaped([1, outChannels, height, width]))
      }
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [outChannels, 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).copied()
      conv2Inputs = inputs
      out.add(dependencies: [inputs])  // This makes sure the copy is done before the convolution, freeing the activations holds there.
    } else {
      conv2Inputs = nil
    }
    out = out.reshaped([outChannels, depth, height, width])
    if let ninShortcut = ninShortcut {
      out =
        ninShortcut(x.reshaped([1, inChannels, depth, height, width])).reshaped([
          outChannels, depth, height, width,
        ]) + out
    } else {
      out = x + out
    }
    let mapper: ModelWeightMapper = { [norm1, conv1, norm2, conv2, ninShortcut] _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).residual.0.gamma"] = [norm1.weight.name]
      mapping["\(prefix).residual.2.weight"] = [conv1.weight.name]
      mapping["\(prefix).residual.2.bias"] = [conv1.bias.name]
      mapping["\(prefix).residual.3.gamma"] = [norm2.weight.name]
      mapping["\(prefix).residual.6.weight"] = [conv2.weight.name]
      mapping["\(prefix).residual.6.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["\(prefix).shortcut.weight"] = [ninShortcut.weight.name]
        mapping["\(prefix).shortcut.bias"] = [ninShortcut.bias.name]
      }
      return mapping
    }
    return (mapper, out)
  }
}

private struct NCHWAttnBlockCausal3D {
  private let norm: Model
  private let toqueries: Model
  private let tokeys: Model
  private let tovalues: Model
  private let projOut: Model
  init(inChannels: Int) {
    norm = RMSNorm(epsilon: 1e-6, axis: [0], name: "attn_norm")
    tokeys = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "to_k")
    toqueries = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "to_q")
    tovalues = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "to_v")
    projOut = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "proj_out")
  }
  func callAsFunction(
    prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
  ) -> (
    ModelWeightMapper, Model
  ) {
    let x = Input()
    var out = norm(x.reshaped([inChannels, depth, height, width])).reshaped([
      1, inChannels, depth, height, width,
    ])
    let hw = width * height
    let k = tokeys(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
    let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
      inChannels, depth, hw,
    ]).transposed(0, 1)
    var dot =
      Matmul(transposeA: (1, 2))(q, k)
    dot = dot.reshaped([depth * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([depth, hw, hw])
    let v = tovalues(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
    out = Matmul(transposeB: (1, 2))(v, dot)
    out =
      x
      + projOut(out.transposed(0, 1).reshaped([1, inChannels, depth, height, width])).reshaped([
        inChannels, depth, height, width,
      ])
    let mapper: ModelWeightMapper = { [norm, toqueries, tokeys, tovalues, projOut] _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).norm.gamma"] = [norm.weight.name]
      mapping["\(prefix).to_qkv.weight"] = [
        toqueries.weight.name, tokeys.weight.name, tovalues.weight.name,
      ]
      mapping["\(prefix).to_qkv.bias"] = [
        toqueries.bias.name, tokeys.bias.name, tovalues.bias.name,
      ]
      mapping["\(prefix).proj.weight"] = [projOut.weight.name]
      mapping["\(prefix).proj.bias"] = [projOut.bias.name]
      return mapping
    }
    return (mapper, Model([x], [out]))
  }
}

func NCHWWanDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool, wan22: Bool, outputChannels: Int,
  highPrecisionFinalNorm: Bool
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let inputChannels = wan22 ? 48 : 16
  let postQuantConv = Convolution(
    groups: 1, filters: inputChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    format: .OIHW, name: "post_quant_conv")
  let postQuantX = postQuantConv(
    x.permuted(3, 0, 1, 2).contiguous().reshaped(
      [1, inputChannels, startDepth, startHeight, startWidth], format: .NCHW)
  ).reshaped([inputChannels, startDepth, startHeight, startWidth])
  let convIn: Model
  let convOut: Model
  if startDepth > 1 {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_in")
    convOut = Convolution(
      groups: 1,
      filters: wan22 ? 12 : (paddingFinalConvLayer ? (outputChannels + 3) / 4 * 4 : outputChannels),
      filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_out")
  } else {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_in")
    convOut = Convolution(
      groups: 1,
      filters: wan22 ? 12 : (paddingFinalConvLayer ? (outputChannels + 3) / 4 * 4 : outputChannels),
      filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_out")
  }
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  var last: Model.IO? = nil
  var midBlock1Builder = NCHWResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  let midAttn1Builder = NCHWAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NCHWResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  var upBlockBuilders = [NCHWResnetBlockCausal3D]()
  for (i, channel) in channels.enumerated().reversed() {
    for _ in 0..<numRepeat + 1 {
      upBlockBuilders.append(
        NCHWResnetBlockCausal3D(
          outChannels: channel, shortcut: previousChannel != channel, startDepth: startDepth))
      previousChannel = channel
    }
    if i > 0 && !wan22 {
      previousChannel = channel / 2
    }
  }
  let timeConvs = (0..<(channels.count - 2)).map { i in
    return Convolution(
      groups: 1, filters: channels[channels.count - i - 1] * 2, filterSize: [3, 1, 1],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "time_conv")
  }
  let upsampleConv2d = (0..<(channels.count - 1)).map { i in
    let channels = wan22 ? channels[channels.count - i - 1] : channels[channels.count - i - 1] / 2
    return Convolution(
      groups: 1, filters: channels, filterSize: [1, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "upsample")
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var convOutInputs: Model.IO? = nil
  var outs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[channels.count - 1]
    var out: Model.IO
    if d == 0 {
      out = postQuantX.reshaped(
        [inputChannels, min(startDepth - d, 3), startHeight, startWidth], offset: [0, d, 0, 0],
        strides: [startDepth * startHeight * startWidth, startHeight * startWidth, startWidth, 1]
      ).contiguous()
      if startDepth > 1 {
        out = convIn(
          out.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
            1, inputChannels, min(startDepth - d, 3) + 2, startHeight + 2, startWidth + 2,
          ])
        ).reshaped([previousChannel, min(startDepth - d, 3), startHeight, startWidth])
      } else {
        out = convIn(out.reshaped([1, inputChannels, startHeight, startWidth])).reshaped([
          previousChannel, 1, startHeight, startWidth,
        ])
      }
    } else {
      out = postQuantX.reshaped(
        [inputChannels, min(startDepth - (d - 1), 4), startHeight, startWidth],
        offset: [0, d - 1, 0, 0],
        strides: [startDepth * startHeight * startWidth, startHeight * startWidth, startWidth, 1]
      ).contiguous().padded(.zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1])
      if let last = last {
        out.add(dependencies: [last])
      }
      out = convIn(
        out.reshaped([
          1, inputChannels, min(startDepth - (d - 1), 4), startHeight + 2, startWidth + 2,
        ])
      ).reshaped([previousChannel, min(startDepth - (d - 1), 4) - 2, startHeight, startWidth])
    }
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var width = startWidth
    var height = startHeight
    var depth = d > 0 ? min(startDepth - 1 - d, 2) : min(startDepth, 3)
    let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "decoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper1)
    out = midBlock1Out
    let (midAttnMapper1, midAttn1) = midAttn1Builder(
      prefix: "decoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    mappers.append(midAttnMapper1)
    out = midAttn1(out)
    let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "decoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper2)
    out = midBlock2Out
    var j = 0
    var k = 0
    var upShortcut: Model.IO? = nil
    for (i, channel) in channels.enumerated().reversed() {
      if i > 0 && wan22 {
        var shortcut = out
        if i > 1 {
          // Need to do temporal upscaling.
          shortcut = shortcut.reshaped([previousChannel, depth, height, width])
          if previousChannel != channel {
            shortcut = Functional.concat(
              axis: 1, shortcut, shortcut, shortcut, shortcut, flags: [.disableOpt])
          } else {
            shortcut = Functional.concat(
              axis: 1, shortcut, shortcut, shortcut, shortcut, shortcut, shortcut, shortcut,
              shortcut, flags: [.disableOpt])
          }
          shortcut = shortcut.reshaped([channel, 2, 2, 2, depth, height, width]).permuted(
            0, 4, 1, 5, 2, 6, 3
          ).copied().reshaped([channel, depth * 2, height * 2, width * 2]).copied()
          if d == 0 {
            shortcut = shortcut.reshaped(
              [channel, (depth - 1) * 2 + 1, height * 2, width * 2], offset: [0, 1, 0, 0],
              strides: [depth * 2 * height * 2 * width * 2, height * 2 * width * 2, width * 2, 1]
            ).contiguous().reshaped([channel, (depth - 1) * 2 + 1, height * 2, width * 2])
          }
        } else {
          shortcut = shortcut.reshaped([previousChannel, depth, height, width])
          if previousChannel != channel {
            shortcut = Functional.concat(axis: 1, shortcut, shortcut, flags: [.disableOpt])
          } else {
            shortcut = Functional.concat(
              axis: 1, shortcut, shortcut, shortcut, shortcut, flags: [.disableOpt])
          }
          shortcut = shortcut.reshaped([channel, 2, 2, depth, height, width]).permuted(
            0, 3, 4, 1, 5, 2
          ).copied().reshaped([channel, depth, height * 2, width * 2])
        }
        upShortcut = shortcut
      } else {
        upShortcut = nil
      }
      for _ in 0..<numRepeat + 1 {
        let (mapper, blockOut) = upBlockBuilders[j](
          input: out,
          prefix: "decoder.upsamples.\(k)",
          inChannels: previousChannel, outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        mappers.append(mapper)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i > 0 {
        if i > 1 && startDepth > 1 {  // Need to bump up on the depth axis.
          if d == 0 {  // Special case for first frame.
            let first = out.reshaped(
              [channel, 1, height, width],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            let more = out.reshaped(
              [channel, (depth - 1), height, width], offset: [0, 1, 0, 0],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            let timeConv = timeConvs[channels.count - i - 1]
            var expanded = timeConv(
              more.padded(.zero, begin: [0, 2, 0, 0], end: [0, 0, 0, 0]).reshaped([
                1, channel, depth + 1, height, width,
              ]))
            if !inputsOnly {
              let inputs = out.reshaped(
                [channel, 2, height, width], offset: [0, depth - 2, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              var mapping = ModelWeightMapping()
              mapping["decoder.upsamples.\(upLayer).time_conv.weight"] = [timeConv.weight.name]
              mapping["decoder.upsamples.\(upLayer).time_conv.bias"] = [timeConv.bias.name]
              return mapping
            }
            mappers.append(mapper)
            expanded = expanded.reshaped([2, channel, depth - 1, height, width]).permuted(
              1, 2, 0, 3, 4
            )
            .contiguous().reshaped([channel, 2 * (depth - 1), height, width])
            out = Functional.concat(axis: 1, first, expanded)
            depth = 1 + (depth - 1) * 2
            out = out.reshaped([channel, depth, height, width])
          } else if let timeInput = timeInputs[channels.count - i - 1] {
            let more = out.reshaped([channel, depth, height, width])
            let timeConv = timeConvs[channels.count - i - 1]
            let expanded = timeConv(
              Functional.concat(axis: 1, timeInput, more, flags: [.disableOpt]).reshaped([
                1, channel, depth + 2, height, width,
              ]))
            if !inputsOnly {
              let inputs = out.reshaped(
                [channel, 2, height, width], offset: [0, depth - 2, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[channels.count - i - 1] = inputs
              expanded.add(dependencies: [inputs])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              var mapping = ModelWeightMapping()
              mapping["decoder.upsamples.\(upLayer).time_conv.weight"] = [timeConv.weight.name]
              mapping["decoder.upsamples.\(upLayer).time_conv.bias"] = [timeConv.bias.name]
              return mapping
            }
            mappers.append(mapper)
            out = expanded.reshaped([2, channel, depth, height, width]).permuted(1, 2, 0, 3, 4)
              .contiguous().reshaped([channel, 2 * depth, height, width])
            depth = depth * 2
          }
        }
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
          out.reshaped([channel, depth, height, width])
        ).reshaped([1, channel, depth, height * 2, width * 2])
        width *= 2
        height *= 2
        if !wan22 {
          previousChannel = channel / 2
        }
        let conv2d = upsampleConv2d[channels.count - i - 1]
        out = conv2d(out).reshaped([
          previousChannel, depth, height, width,
        ])
        if let upShortcut = upShortcut {
          out = upShortcut + out
        }
        let upLayer = k
        let mapper: ModelWeightMapper = { _ in
          var mapping = ModelWeightMapping()
          mapping["decoder.upsamples.\(upLayer).resample.1.weight"] = [conv2d.weight.name]
          mapping["decoder.upsamples.\(upLayer).resample.1.bias"] = [conv2d.bias.name]
          return mapping
        }
        mappers.append(mapper)
        k += 1
      }
    }
    let beforeNorm = out
    if highPrecisionFinalNorm {
      out = out.to(.Float32)
    }
    out = normOut(out.reshaped([channels[0], depth, height, width]))
    if highPrecisionFinalNorm {
      out = out.to(of: beforeNorm)
    }
    let pre = out.swish()
    if let convOutInputs = convOutInputs {
      out = convOut(
        Functional.concat(axis: 1, convOutInputs, pre, flags: [.disableOpt]).padded(
          .zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]
        ).reshaped([
          1, channels[0], depth + 2, height + 2, width + 2,
        ])
      )
    } else {
      if startDepth > 1 {
        out = convOut(
          pre.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
            1, channels[0], depth + 2, height + 2, width + 2,
          ])
        )
      } else {
        out = convOut(pre.reshaped([1, channels[0], height, width]))
      }
    }
    if !inputsOnly {
      let inputs = pre.reshaped(
        [channels[0], 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).copied()
      convOutInputs = inputs
      out.add(dependencies: [inputs])
    }
    last = out
    outs.append(
      out.reshaped([
        wan22 ? 12 : (paddingFinalConvLayer ? (outputChannels + 3) / 4 * 4 : outputChannels), depth,
        height, width,
      ]))
  }
  var out: Model.IO
  if outs.count > 1 {
    let concat = Concat(axis: 1)
    concat.flags = [.disableOpt]
    out = concat(outs)
  } else {
    out = outs[0]
  }
  if wan22 {
    out = out.reshaped([3, 2, 2, (startDepth - 1) * 4 + 1, startHeight * 8, startWidth * 8])
      .permuted(3, 4, 1, 5, 2, 0).copied().reshaped([
        (startDepth - 1) * 4 + 1, startHeight * 16, startWidth * 16, 3,
      ])
  } else {
    out = out.permuted(1, 2, 3, 0).contiguous().reshaped(
      .NHWC(
        (startDepth - 1) * 4 + 1, startHeight * 8, startWidth * 8,
        paddingFinalConvLayer ? (outputChannels + 3) / 4 * 4 : outputChannels))
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv2.weight"] = [postQuantConv.weight.name]
    mapping["conv2.bias"] = [postQuantConv.bias.name]
    mapping["decoder.conv1.weight"] = [convIn.weight.name]
    mapping["decoder.conv1.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["decoder.head.0.gamma"] = [normOut.weight.name]
    mapping["decoder.head.2.weight"] = [convOut.weight.name]
    mapping["decoder.head.2.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NCHWWanEncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int, wan22: Bool,
  inputChannels: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn: Model
  if startDepth > 1 {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_in")
  } else {
    convIn = Convolution(
      groups: 1, filters: previousChannel, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_in")
  }
  var mappers = [ModelWeightMapper]()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  var downBlockBuilders = [NCHWResnetBlockCausal3D]()
  var timeConvs = [Convolution]()
  var downsampleConv2d = [Convolution]()
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      downBlockBuilders.append(
        NCHWResnetBlockCausal3D(
          outChannels: channel, shortcut: previousChannel != channel, startDepth: startDepth))
      previousChannel = channel
    }
    if i < channels.count - 1 {
      downsampleConv2d.append(
        Convolution(
          groups: 1, filters: channel, filterSize: [1, 3, 3],
          hint: Hint(
            stride: [1, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          format: .OIHW, name: "downsample"))
      if i > 0 && startDepth > 1 {
        timeConvs.append(
          Convolution(
            groups: 1, filters: channel, filterSize: [3, 1, 1],
            hint: Hint(
              stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
            format: .OIHW, name: "time_conv"))
      }
    }
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var midBlock1Builder = NCHWResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  let midAttn1Builder = NCHWAttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = NCHWResnetBlockCausal3D(
    outChannels: previousChannel, shortcut: false, startDepth: startDepth)
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  let endHeight = height
  let endWidth = width
  let endDepth = depth
  var outs = [Model.IO]()
  let input: Model.IO
  if wan22 {
    input = x.reshaped([endDepth, endHeight, 2, endWidth, 2, 3]).permuted(5, 2, 4, 0, 1, 3)
      .contiguous().reshaped(.NCHW(3 * 2 * 2, endDepth, endHeight, endWidth))
  } else {
    input = x.permuted(3, 0, 1, 2).contiguous().reshaped(
      .NCHW(inputChannels, endDepth, endHeight, endWidth))
  }
  for d in stride(from: 0, to: max(startDepth - 1, 1), by: 2) {
    previousChannel = channels[0]
    height = endHeight
    width = endWidth
    var out: Model.IO
    if d == 0 {
      if startDepth > 1 {
        depth = min(endDepth, 9)
        out = input.reshaped(
          [inputChannels, depth, height, width],
          strides: [endDepth * height * width, height * width, width, 1]
        ).contiguous().padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1])
        out = convIn(out.reshaped([1, inputChannels, depth + 2, height + 2, width + 2])).reshaped([
          previousChannel, depth, height, width,
        ])
      } else {
        out = convIn(input.reshaped([1, inputChannels, height, width])).reshaped([
          previousChannel, depth, height, width,
        ])
      }
    } else {
      depth = min(endDepth - (d * 4 + 1), 8)
      out = input.reshaped(
        [inputChannels, depth + 2, height, width], offset: [0, d * 4 - 1, 0, 0],
        strides: [endDepth * height * width, height * width, width, 1]
      ).contiguous().padded(.zero, begin: [0, 0, 1, 1], end: [0, 0, 1, 1])
      if let last = outs.last {
        out.add(dependencies: [last])
      }
      out = convIn(out.reshaped([1, inputChannels, depth + 2, height + 2, width + 2])).reshaped([
        previousChannel, depth, height, width,
      ])
    }
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var j = 0
    var k = 0
    for (i, channel) in channels.enumerated() {
      var downShortcut = out
      if i < channels.count - 1 && wan22 {
        if i > 0 && depth > 1 {
          let pad = (2 - depth % 2) % 2
          downShortcut = downShortcut.padded(.zero, begin: [0, pad, 0, 0], end: [0, 0, 0, 0])
          let paddedDepth = depth + pad
          downShortcut = downShortcut.reshaped([
            previousChannel, paddedDepth / 2, 2, height / 2, 2, width / 2, 2,
          ]).permuted(0, 2, 4, 6, 1, 3, 5).copied()
          downShortcut = downShortcut.reshaped([
            channel, 8 * previousChannel / channel, paddedDepth / 2, height / 2, width / 2,
          ]).reduced(.mean, axis: [1]).reshaped([channel, paddedDepth / 2, height / 2, width / 2])
        } else {
          downShortcut = downShortcut.reshaped([
            previousChannel, depth, height / 2, 2, width / 2, 2,
          ])
          .permuted(0, 3, 5, 1, 2, 4).copied()
          downShortcut = downShortcut.reshaped([
            channel, 4 * previousChannel / channel, depth, height / 2, width / 2,
          ]).reduced(.mean, axis: [1]).reshaped([channel, depth, height / 2, width / 2])
        }
      }
      for _ in 0..<numRepeat {
        let (mapper, blockOut) = downBlockBuilders[j](
          input: out,
          prefix: "encoder.downsamples.\(k)", inChannels: previousChannel,
          outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        mappers.append(mapper)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i < channels.count - 1 {
        let conv2d = downsampleConv2d[i]
        out = conv2d(
          out.padded(.zero, begin: [0, 0, 0, 0], end: [0, 0, 1, 1]).reshaped([
            1, previousChannel, depth, height + 1, width + 1,
          ]))
        height /= 2
        width /= 2
        out = out.reshaped([channel, depth, height, width])
        let downLayer = k
        let mapper: ModelWeightMapper = { _ in
          var mapping = ModelWeightMapping()
          mapping[
            "encoder.downsamples.\(downLayer).resample.1.weight"
          ] = [conv2d.weight.name]
          mapping["encoder.downsamples.\(downLayer).resample.1.bias"] = [conv2d.bias.name]
          return mapping
        }
        mappers.append(mapper)
        if i > 0 && startDepth > 1 {
          if d == 0 {
            let first = out.reshaped(
              [channel, 1, height, width],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            let timeConv = timeConvs[i - 1]
            let shrunk = timeConv(out.reshaped([1, channel, depth, height, width]))
              .reshaped([channel, (depth - 1) / 2, height, width])
            if !inputsOnly {
              let input = out.reshaped(
                [channel, 1, height, width], offset: [0, depth - 1, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            let upLayer = k
            let mapper: ModelWeightMapper = { _ in
              var mapping = ModelWeightMapping()
              mapping["encoder.downsamples.\(upLayer).time_conv.weight"] = [timeConv.weight.name]
              mapping["encoder.downsamples.\(upLayer).time_conv.bias"] = [timeConv.bias.name]
              return mapping
            }
            mappers.append(mapper)
            depth = (depth - 1) / 2 + 1
            out = Functional.concat(axis: 1, first, shrunk)
          } else if let timeInput = timeInputs[i - 1] {
            let shrunk = timeConvs[i - 1](
              Functional.concat(axis: 1, timeInput, out, flags: [.disableOpt]).reshaped([
                1, channel, depth + 1, height, width,
              ]))
            if !inputsOnly {
              let input = out.reshaped(
                [channel, 1, height, width], offset: [0, depth - 1, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).copied()
              timeInputs[i - 1] = input
              shrunk.add(dependencies: [input])
            }
            depth = depth / 2
            out = shrunk.reshaped([channel, depth, height, width])
          }
        }
        k += 1
      }
      if wan22 {
        out = downShortcut + out
      }
    }
    let (midBlockMapper1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "encoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper1)
    out = midBlock1Out
    let (midAttnMapper1, midAttn1) = midAttn1Builder(
      prefix: "encoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    mappers.append(midAttnMapper1)
    out = midAttn1(out)
    let (midBlockMapper2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "encoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel,
      shortcut: false, depth: depth, height: height, width: width, inputsOnly: inputsOnly)
    mappers.append(midBlockMapper2)
    out = midBlock2Out
    out = normOut(out.reshaped([previousChannel, depth, height, width]))
    out = out.swish()
    outs.append(out)
  }
  height = startHeight
  width = startWidth
  depth = startDepth
  var out: Model.IO
  if outs.count > 1 {
    let concat = Concat(axis: 1)
    concat.flags = [.disableOpt]
    out = concat(outs)
  } else {
    out = outs[0]
  }
  let convOut: Model
  if startDepth > 1 {
    convOut = Convolution(
      groups: 1, filters: wan22 ? 96 : 32, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      format: .OIHW, name: "conv_out")
    out = convOut(
      out.padded(.zero, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
        1, previousChannel, depth + 2, height + 2, width + 2,
      ])
    ).reshaped([1, wan22 ? 96 : 32, depth, height, width])
  } else {
    convOut = Convolution(
      groups: 1, filters: wan22 ? 96 : 32, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_out")
    out = convOut(out.reshaped([1, previousChannel, height, width])).reshaped([
      1, wan22 ? 96 : 32, depth, height, width,
    ])
  }
  let quantConv = Convolution(
    groups: 1, filters: wan22 ? 96 : 32, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    format: .OIHW,
    name: "quant_conv")
  out = quantConv(out).permuted(0, 2, 3, 4, 1).contiguous().reshaped(
    .NHWC(depth, height, width, wan22 ? 96 : 32))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["encoder.conv1.weight"] = [convIn.weight.name]
    mapping["encoder.conv1.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["encoder.head.0.gamma"] = [normOut.weight.name]
    mapping["encoder.head.2.weight"] = [convOut.weight.name]
    mapping["encoder.head.2.bias"] = [convOut.bias.name]
    mapping["conv1.weight"] = [quantConv.weight.name]
    mapping["conv1.bias"] = [quantConv.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func WanDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool, wan22: Bool, outputChannels: Int,
  highPrecisionFinalNorm: Bool, format: TensorFormat
) -> (ModelWeightMapper, Model) {
  switch format {
  case .NHWC:
    return NHWCWanDecoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, paddingFinalConvLayer: paddingFinalConvLayer, wan22: wan22,
      outputChannels: outputChannels, highPrecisionFinalNorm: highPrecisionFinalNorm)
  case .NCHW:
    return NCHWWanDecoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, paddingFinalConvLayer: paddingFinalConvLayer, wan22: wan22,
      outputChannels: outputChannels, highPrecisionFinalNorm: highPrecisionFinalNorm)
  case .CHWN:
    fatalError()
  }
}

func WanEncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int,
  wan22: Bool, inputChannels: Int, format: TensorFormat
) -> (ModelWeightMapper, Model) {
  switch format {
  case .NHWC:
    return NHWCWanEncoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, wan22: wan22, inputChannels: inputChannels)
  case .NCHW:
    return NCHWWanEncoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, wan22: wan22, inputChannels: inputChannels)
  case .CHWN:
    fatalError()
  }
}
