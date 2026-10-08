import DiffusionMappings
import NNC

public enum LTX2VideoVAEPaddingMode {
  case zero
  case reflect
}

private func NHWCResnetBlockCausal3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int, isCausal: Bool,
  paddingMode: LTX2VideoVAEPaddingMode
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let spatialPadding = !isCausal && paddingMode == .zero ? 1 : 0
  let norm1 = RMSNorm(epsilon: 1e-8, axis: [3], elementwiseAffine: false, name: "resnet_norm1")
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding],
        end: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding])), format: .OIHW,
    name: "resnet_conv1")
  out = out.padded(
    .replicate, begin: [isCausal ? 2 : 1, 0, 0, 0], end: [isCausal ? 0 : 1, 0, 0, 0])
  if !isCausal && paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
      1, depth + 2, height + 2, width + 2, channels,
    ])
  } else {
    out = out.reshaped([
      1, depth + 2, height, width, channels,
    ])
  }
  out = conv1(out)
  let norm2 = RMSNorm(epsilon: 1e-8, axis: [3], elementwiseAffine: false, name: "resnet_norm2")
  out = norm2(out.reshaped([depth, height, width, channels]))
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding],
        end: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding])), format: .OIHW,
    name: "resnet_conv2")
  out = out.padded(
    .replicate, begin: [isCausal ? 2 : 1, 0, 0, 0], end: [isCausal ? 0 : 1, 0, 0, 0])
  if !isCausal && paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
      1, depth + 2, height + 2, width + 2, channels,
    ])
  } else {
    out = out.reshaped([
      1, depth + 2, height, width, channels,
    ])
  }
  out = conv2(out).reshaped([depth, height, width, channels])
  out = x + out
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).conv1.conv.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.conv.bias"] = [conv1.bias.name]
    mapping["\(prefix).conv2.conv.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.conv.bias"] = [conv2.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func NHWCLTX2VideoDecoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int, startDepth: Int, paddingMode: LTX2VideoVAEPaddingMode, residual: Bool, paddingFinalConvLayer: Bool
)
  -> (ModelWeightMapper, Model)
{
  precondition(!layers.isEmpty)
  let x = Input()
  var previousChannel = layers[0].channels
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0],
        end: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0])),
    format: .OIHW,
    name: "conv_in")
  var out = x.padded(.replicate, begin: [1, 0, 0, 0], end: [1, 0, 0, 0])
  if paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
      1, startDepth + 2, startHeight + 2, startWidth + 2, 128,
    ])
  } else {
    out = out.reshaped([
      1, startDepth + 2, startHeight, startWidth, 128,
    ])
  }
  out = convIn(out).reshaped([startDepth, startHeight, startWidth, previousChannel])
  var mappers = [ModelWeightMapper]()
  var j = 0
  var depth = startDepth
  var height = startHeight
  var width = startWidth
  for layer in layers {
    let channels = layer.channels
    if layer.stride.0 > 1 || layer.stride.1 > 1 || layer.stride.2 > 1 {
      // Convolution & reshape.
      let conv = Convolution(
        groups: 1, filters: channels * layer.stride.0 * layer.stride.1 * layer.stride.2,
        filterSize: [3, 3, 3],
        hint: Hint(
          stride: [1, 1, 1],
          border: Hint.Border(
            begin: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0],
            end: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0])),
        format: .OIHW,
        name: "depth_to_space_upsample")
      if layer.stride.0 == 1 {
        var residualOut: Model.IO? = nil
        if residual {
          precondition(previousChannel % (layer.stride.1 * layer.stride.2) == 0)
          precondition(channels * layer.stride.1 * layer.stride.2 % previousChannel == 0)
          var residualOutLocal = out.reshaped([
            depth, height, width, previousChannel / (layer.stride.1 * layer.stride.2),
            layer.stride.1, layer.stride.2,
          ]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
            depth, height * layer.stride.1, width * layer.stride.2,
            previousChannel / (layer.stride.1 * layer.stride.2),
          ])
          if channels * layer.stride.1 * layer.stride.2 / previousChannel > 1 {
            let concat = Concat(axis: 3)
            concat.flags = [.disableOpt]
            residualOutLocal = concat(
              Array(
                repeating: residualOutLocal,
                count: channels * layer.stride.1 * layer.stride.2 / previousChannel))
          }
          residualOut = residualOutLocal
        }
        out = out.padded(.replicate, begin: [1, 0, 0, 0], end: [1, 0, 0, 0])
        if paddingMode == .reflect {
          out = out.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
            1, depth + 2, height + 2, width + 2, previousChannel,
          ])
        } else {
          out = out.reshaped([
            1, depth + 2, height, width, previousChannel,
          ])
        }
        out = conv(out)
        out = out.reshaped([
          depth, height, width, channels, layer.stride.1, layer.stride.2,
        ]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
          depth, height * layer.stride.1, width * layer.stride.2, channels,
        ])
        if let residualOut = residualOut {
          out = residualOut + out
        }
        height = height * layer.stride.1
        width = width * layer.stride.2
      } else if layer.stride.1 == 1 && layer.stride.2 == 1 {
        precondition(layer.stride.0 == 2)
        var residualOut: Model.IO? = nil
        if residual {
          precondition(previousChannel % layer.stride.0 == 0)
          precondition(channels * layer.stride.0 % previousChannel == 0)
          var residualOutLocal = out.reshaped([
            depth, height, width, previousChannel / layer.stride.0, layer.stride.0,
          ]).permuted(0, 4, 1, 2, 3).contiguous().reshaped(
            [depth * layer.stride.0 - 1, height, width, previousChannel / layer.stride.0],
            offset: [1, 0, 0, 0],
            strides: [
              height * width * previousChannel / layer.stride.0,
              width * previousChannel / layer.stride.0, previousChannel / layer.stride.0, 1,
            ]
          )
          if channels * layer.stride.0 / previousChannel > 1 {
            residualOutLocal = residualOutLocal.contiguous()
            let concat = Concat(axis: 3)
            concat.flags = [.disableOpt]
            residualOutLocal = concat(
              Array(repeating: residualOutLocal, count: channels * layer.stride.0 / previousChannel))
          } else {
            residualOutLocal = residualOutLocal.copied()
          }
          residualOut = residualOutLocal
        }
        out = out.padded(.replicate, begin: [1, 0, 0, 0], end: [1, 0, 0, 0])
        if paddingMode == .reflect {
          out = out.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
            1, depth + 2, height + 2, width + 2, previousChannel,
          ])
        } else {
          out = out.reshaped([
            1, depth + 2, height, width, previousChannel,
          ])
        }
        out = conv(out)
        out = out.reshaped([
          depth, height, width, channels, layer.stride.0,
        ]).permuted(0, 4, 1, 2, 3).contiguous().reshaped(
          [depth * layer.stride.0 - 1, height, width, channels], offset: [1, 0, 0, 0],
          strides: [height * width * channels, width * channels, channels, 1]
        ).contiguous()
        if let residualOut = residualOut {
          out = residualOut + out
        }
        depth = depth * layer.stride.0 - 1
      } else {
        precondition(layer.stride.0 == 2)
        var residualOut: Model.IO? = nil
        if residual {
          precondition(previousChannel % (layer.stride.0 * layer.stride.1 * layer.stride.2) == 0)
          precondition(
            channels * layer.stride.0 * layer.stride.1 * layer.stride.2 % previousChannel == 0)
          var residualOutLocal = out.reshaped([
            depth, height, width,
            previousChannel / (layer.stride.0 * layer.stride.1 * layer.stride.2), layer.stride.0,
            layer.stride.1, layer.stride.2,
          ]).permuted(0, 4, 1, 5, 2, 6, 3).contiguous().reshaped(
            [
              depth * layer.stride.0 - 1, height * layer.stride.1, width * layer.stride.2,
              previousChannel / (layer.stride.0 * layer.stride.1 * layer.stride.2),
            ], offset: [1, 0, 0, 0],
            strides: [
              height * width * previousChannel / layer.stride.0,
              width * previousChannel / (layer.stride.0 * layer.stride.1),
              previousChannel / (layer.stride.0 * layer.stride.1 * layer.stride.2), 1,
            ]
          )
          if channels * layer.stride.0 * layer.stride.1 * layer.stride.2 / previousChannel > 1 {
            residualOutLocal = residualOutLocal.contiguous()
            let concat = Concat(axis: 3)
            concat.flags = [.disableOpt]
            residualOutLocal = concat(
              Array(
                repeating: residualOutLocal,
                count: channels * layer.stride.0 * layer.stride.1 * layer.stride.2 / previousChannel
              ))
          } else {
            residualOutLocal = residualOutLocal.copied()
          }
          residualOut = residualOutLocal
        }
        out = out.padded(.replicate, begin: [1, 0, 0, 0], end: [1, 0, 0, 0])
        if paddingMode == .reflect {
          out = out.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
            1, depth + 2, height + 2, width + 2, previousChannel,
          ])
        } else {
          out = out.reshaped([
            1, depth + 2, height, width, previousChannel,
          ])
        }
        out = conv(out)
        out = out.reshaped([
          depth, height, width, channels, layer.stride.0, layer.stride.1, layer.stride.2,
        ]).permuted(0, 4, 1, 5, 2, 6, 3).contiguous().reshaped(
          [depth * layer.stride.0 - 1, height * layer.stride.1, width * layer.stride.2, channels],
          offset: [1, 0, 0, 0],
          strides: [
            height * layer.stride.1 * width * layer.stride.2 * channels,
            width * layer.stride.2 * channels, channels, 1,
          ]
        ).contiguous()
        if let residualOut = residualOut {
          out = residualOut + out
        }
        depth = depth * layer.stride.0 - 1
        height = height * layer.stride.1
        width = width * layer.stride.2
      }
      previousChannel = channels
      let upBlocks = j
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["up_blocks.\(upBlocks).conv.conv.weight"] = [conv.weight.name]
        mapping["up_blocks.\(upBlocks).conv.conv.bias"] = [conv.bias.name]
        return mapping
      }
      j += 1
    } else {
      precondition(channels == previousChannel)
    }
    for i in 0..<layer.numRepeat {
      let (mapper, block) = NHWCResnetBlockCausal3D(
        prefix: "up_blocks.\(j).res_blocks.\(i)", channels: channels, depth: depth,
        height: height,
        width: width, isCausal: false, paddingMode: paddingMode)
      out = block(out)
      mappers.append(mapper)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [3], elementwiseAffine: false, name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 64 : 48, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0],
        end: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0])),
    format: .OIHW,
    name: "conv_out")
  out = out.padded(.replicate, begin: [1, 0, 0, 0], end: [1, 0, 0, 0])
  if paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
      1, depth + 2, height + 2, width + 2, previousChannel,
    ])
  } else {
    out = out.reshaped([
      1, depth + 2, height, width, previousChannel,
    ])
  }
  out = convOut(out)
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  out = out.reshaped([depth, height, width, paddingFinalConvLayer ? 4 : 3, 4, 4]).permuted(0, 1, 5, 2, 4, 3).contiguous()
    .reshaped([depth, height * 4, width * 4, paddingFinalConvLayer ? 4 : 3])

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func NHWCLTX2VideoEncoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int, startDepth: Int, paddingMode: LTX2VideoVAEPaddingMode
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for layer in layers {
    depth = (depth - 1) * layer.stride.0 + 1
    height *= layer.stride.1
    width *= layer.stride.2
  }
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  var out = x.reshaped([depth, height, 4, width, 4, 3]).permuted(0, 1, 3, 5, 4, 2).contiguous()
    .reshaped([depth, height, width, 48])
  var previousChannel = layers[0].channels
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    format: .OIHW, name: "conv_in")
  out = convIn(
    out.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0]).reshaped([
      1, depth + 2, height, width, 48,
    ])
  ).reshaped([depth, height, width, previousChannel])
  var j = 0
  var mappers = [ModelWeightMapper]()
  for layer in layers {
    let channels = layer.channels
    if layer.stride.0 > 1 || layer.stride.1 > 1 || layer.stride.2 > 1 {
      // Convolution & reshape.
      let conv = Convolution(
        groups: 1, filters: channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
        filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
        format: .OIHW, name: "space_to_depth_downsample")
      if layer.stride.0 == 1 {
        var residual = out.reshaped([
          depth, height / layer.stride.1, layer.stride.1, width / layer.stride.2,
          layer.stride.2, previousChannel,
        ]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped([
          depth * (height / layer.stride.1) * (width / layer.stride.2), channels,
          previousChannel * layer.stride.1 * layer.stride.2 / channels,
        ])
        residual = residual.reduced(.mean, axis: [2]).reshaped([
          depth, height / layer.stride.1, width / layer.stride.2, channels,
        ])
        out = conv(
          out.padded(.replicate, begin: [2, 0, 0, 0]).reshaped([
            1, depth + 2, height, width, previousChannel,
          ]))
        out = out.reshaped([
          depth, height / layer.stride.1, layer.stride.1, width / layer.stride.2, layer.stride.2,
          channels / (layer.stride.1 * layer.stride.2),
        ]).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped([
          depth, height / layer.stride.1, width / layer.stride.2, channels,
        ])
        out = residual + out
        height = height / layer.stride.1
        width = width / layer.stride.2
      } else if layer.stride.1 == 1 && layer.stride.2 == 1 {
        var residual = out.padded(.replicate, begin: [1, 0, 0, 0]).reshaped([
          (depth + 1) / layer.stride.0, layer.stride.0, height, width, previousChannel,
        ]).permuted(0, 2, 3, 4, 1).contiguous()
        if previousChannel * layer.stride.0 / channels > 1 {
          residual = residual.reshaped([
            (depth + 1) / layer.stride.0 * height * width, channels,
            previousChannel * layer.stride.0 / channels,
          ])
          residual = residual.reduced(.mean, axis: [2]).reshaped([
            (depth + 1) / layer.stride.0, height, width, channels,
          ])
        } else {
          residual = residual.reshaped([(depth + 1) / layer.stride.0, height, width, channels])
        }
        out = conv(
          out.padded(.replicate, begin: [3, 0, 0, 0]).reshaped([
            1, depth + 3, height, width, previousChannel,
          ]))
        out = out.reshaped([
          (depth + 1) / layer.stride.0, layer.stride.0, height, width, channels / layer.stride.0,
        ]).permuted(0, 2, 3, 4, 1).contiguous().reshaped([
          (depth + 1) / layer.stride.0, height, width, channels,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
      } else {
        var residual = out.padded(.replicate, begin: [1, 0, 0, 0]).reshaped([
          (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1,
          layer.stride.1, width / layer.stride.2, layer.stride.2, previousChannel,
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          ((depth + 1) / layer.stride.0) * (height / layer.stride.1) * (width / layer.stride.2),
          channels, previousChannel * layer.stride.0 * layer.stride.1 * layer.stride.2 / channels,
        ])
        residual = residual.reduced(.mean, axis: [2]).reshaped([
          (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2, channels,
        ])
        out = conv(
          out.padded(.replicate, begin: [3, 0, 0, 0]).reshaped([
            1, depth + 3, height, width, previousChannel,
          ]))
        out = out.reshaped([
          (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1, layer.stride.1,
          width / layer.stride.2, layer.stride.2,
          channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2, channels,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
        height = height / layer.stride.1
        width = width / layer.stride.2
      }
      previousChannel = channels
      let downBlocks = j
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["down_blocks.\(downBlocks).conv.conv.weight"] = [conv.weight.name]
        mapping["down_blocks.\(downBlocks).conv.conv.bias"] = [conv.bias.name]
        return mapping
      }
      j += 1
    }
    for i in 0..<layer.numRepeat {
      let (mapper, block) = NHWCResnetBlockCausal3D(
        prefix: "down_blocks.\(j).res_blocks.\(i)", channels: channels, depth: depth,
        height: height, width: width, isCausal: true, paddingMode: paddingMode)
      out = block(out)
      mappers.append(mapper)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [3], elementwiseAffine: false, name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: 128,  // Ignore the last channel which is only useful when training 129
    filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    format: .OIHW, name: "conv_out")
  out = convOut(
    out.padded(.replicate, begin: [2, 0, 0, 0]).reshaped([
      1, depth + 2, height, width, previousChannel,
    ])
  ).reshaped([depth, height, width, 128])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NCHWResnetBlockCausal3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int, isCausal: Bool,
  paddingMode: LTX2VideoVAEPaddingMode
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let spatialPadding = !isCausal && paddingMode == .zero ? 1 : 0
  let norm1 = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "resnet_norm1")
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding],
        end: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding])), format: .OIHW,
    name: "resnet_conv1")
  out = out.padded(
    .replicate, begin: [0, isCausal ? 2 : 1, 0, 0], end: [0, isCausal ? 0 : 1, 0, 0])
  if !isCausal && paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]).reshaped([
      1, channels, depth + 2, height + 2, width + 2,
    ])
  } else {
    out = out.reshaped([1, channels, depth + 2, height, width])
  }
  out = conv1(out)
  let norm2 = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "resnet_norm2")
  out = norm2(out.reshaped([channels, depth, height, width]))
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding],
        end: [0, isCausal ? 1 : spatialPadding, isCausal ? 1 : spatialPadding])), format: .OIHW,
    name: "resnet_conv2")
  out = out.padded(
    .replicate, begin: [0, isCausal ? 2 : 1, 0, 0], end: [0, isCausal ? 0 : 1, 0, 0])
  if !isCausal && paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]).reshaped([
      1, channels, depth + 2, height + 2, width + 2,
    ])
  } else {
    out = out.reshaped([1, channels, depth + 2, height, width])
  }
  out = conv2(out).reshaped([channels, depth, height, width])
  out = x + out
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).conv1.conv.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.conv.bias"] = [conv1.bias.name]
    mapping["\(prefix).conv2.conv.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.conv.bias"] = [conv2.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func NCHWLTX2VideoDecoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int, startDepth: Int, paddingMode: LTX2VideoVAEPaddingMode, residual: Bool, paddingFinalConvLayer: Bool
)
  -> (ModelWeightMapper, Model)
{
  precondition(!layers.isEmpty)
  let x = Input()
  var previousChannel = layers[0].channels
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0],
        end: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0])),
    format: .OIHW, name: "conv_in")
  var out = x.permuted(3, 0, 1, 2).contiguous().reshaped(
    [128, startDepth, startHeight, startWidth], format: .NCHW)
  out = out.padded(.replicate, begin: [0, 1, 0, 0], end: [0, 1, 0, 0])
  if paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]).reshaped([
      1, 128, startDepth + 2, startHeight + 2, startWidth + 2,
    ])
  } else {
    out = out.reshaped([1, 128, startDepth + 2, startHeight, startWidth])
  }
  out = convIn(out).reshaped([previousChannel, startDepth, startHeight, startWidth])
  var mappers = [ModelWeightMapper]()
  var j = 0
  var depth = startDepth
  var height = startHeight
  var width = startWidth
  for layer in layers {
    let channels = layer.channels
    if layer.stride.0 > 1 || layer.stride.1 > 1 || layer.stride.2 > 1 {
      // Convolution & reshape.
      let conv = Convolution(
        groups: 1, filters: channels * layer.stride.0 * layer.stride.1 * layer.stride.2,
        filterSize: [3, 3, 3],
        hint: Hint(
          stride: [1, 1, 1],
          border: Hint.Border(
            begin: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0],
            end: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0])),
        format: .OIHW, name: "depth_to_space_upsample")
      if layer.stride.0 == 1 {
        var residualOut: Model.IO? = nil
        if residual {
          precondition(previousChannel % (layer.stride.1 * layer.stride.2) == 0)
          precondition(channels * layer.stride.1 * layer.stride.2 % previousChannel == 0)
          var residualOutLocal = out.reshaped([
            previousChannel / (layer.stride.1 * layer.stride.2), layer.stride.1, layer.stride.2,
            depth, height, width,
          ]).permuted(0, 3, 4, 1, 5, 2).contiguous().reshaped([
            previousChannel / (layer.stride.1 * layer.stride.2), depth,
            height * layer.stride.1, width * layer.stride.2,
          ])
          if channels * layer.stride.1 * layer.stride.2 / previousChannel > 1 {
            let concat = Concat(axis: 0)
            concat.flags = [.disableOpt]
            residualOutLocal = concat(
              Array(
                repeating: residualOutLocal,
                count: channels * layer.stride.1 * layer.stride.2 / previousChannel))
          }
          residualOut = residualOutLocal
        }
        out = out.padded(.replicate, begin: [0, 1, 0, 0], end: [0, 1, 0, 0])
        if paddingMode == .reflect {
          out = out.padded(.reflect, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]).reshaped([
            1, previousChannel, depth + 2, height + 2, width + 2,
          ])
        } else {
          out = out.reshaped([1, previousChannel, depth + 2, height, width])
        }
        out = conv(out)
        out = out.reshaped([
          channels, layer.stride.1, layer.stride.2, depth, height, width,
        ]).permuted(0, 3, 4, 1, 5, 2).contiguous().reshaped([
          channels, depth, height * layer.stride.1, width * layer.stride.2,
        ])
        if let residualOut = residualOut {
          out = residualOut + out
        }
        height = height * layer.stride.1
        width = width * layer.stride.2
      } else if layer.stride.1 == 1 && layer.stride.2 == 1 {
        precondition(layer.stride.0 == 2)
        var residualOut: Model.IO? = nil
        if residual {
          precondition(previousChannel % layer.stride.0 == 0)
          precondition(channels * layer.stride.0 % previousChannel == 0)
          var residualOutLocal = out.reshaped([
            previousChannel / layer.stride.0, layer.stride.0, depth, height, width,
          ]).permuted(0, 2, 1, 3, 4).contiguous().reshaped(
            [previousChannel / layer.stride.0, depth * layer.stride.0 - 1, height, width],
            offset: [0, 1, 0, 0],
            strides: [depth * layer.stride.0 * height * width, height * width, width, 1]
          )
          if channels * layer.stride.0 / previousChannel > 1 {
            residualOutLocal = residualOutLocal.contiguous()
            let concat = Concat(axis: 0)
            concat.flags = [.disableOpt]
            residualOutLocal = concat(
              Array(repeating: residualOutLocal, count: channels * layer.stride.0 / previousChannel)
            )
          } else {
            residualOutLocal = residualOutLocal.copied()
          }
          residualOut = residualOutLocal
        }
        out = out.padded(.replicate, begin: [0, 1, 0, 0], end: [0, 1, 0, 0])
        if paddingMode == .reflect {
          out = out.padded(.reflect, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]).reshaped([
            1, previousChannel, depth + 2, height + 2, width + 2,
          ])
        } else {
          out = out.reshaped([1, previousChannel, depth + 2, height, width])
        }
        out = conv(out)
        out = out.reshaped([
          channels, layer.stride.0, depth, height, width,
        ]).permuted(0, 2, 1, 3, 4).contiguous().reshaped(
          [channels, depth * layer.stride.0 - 1, height, width], offset: [0, 1, 0, 0],
          strides: [depth * layer.stride.0 * height * width, height * width, width, 1]
        ).contiguous()
        if let residualOut = residualOut {
          out = residualOut + out
        }
        depth = depth * layer.stride.0 - 1
      } else {
        precondition(layer.stride.0 == 2)
        var residualOut: Model.IO? = nil
        if residual {
          precondition(previousChannel % (layer.stride.0 * layer.stride.1 * layer.stride.2) == 0)
          precondition(
            channels * layer.stride.0 * layer.stride.1 * layer.stride.2 % previousChannel == 0)
          var residualOutLocal = out.reshaped([
            previousChannel / (layer.stride.0 * layer.stride.1 * layer.stride.2), layer.stride.0,
            layer.stride.1, layer.stride.2, depth, height, width,
          ]).permuted(0, 4, 1, 5, 2, 6, 3).contiguous().reshaped(
            [
              previousChannel / (layer.stride.0 * layer.stride.1 * layer.stride.2),
              depth * layer.stride.0 - 1, height * layer.stride.1, width * layer.stride.2,
            ], offset: [0, 1, 0, 0],
            strides: [
              depth * layer.stride.0 * height * layer.stride.1 * width * layer.stride.2,
              height * layer.stride.1 * width * layer.stride.2, width * layer.stride.2, 1,
            ]
          )
          if channels * layer.stride.0 * layer.stride.1 * layer.stride.2 / previousChannel > 1 {
            residualOutLocal = residualOutLocal.contiguous()
            let concat = Concat(axis: 0)
            concat.flags = [.disableOpt]
            residualOutLocal = concat(
              Array(
                repeating: residualOutLocal,
                count: channels * layer.stride.0 * layer.stride.1 * layer.stride.2 / previousChannel
              ))
          } else {
            residualOutLocal = residualOutLocal.copied()
          }
          residualOut = residualOutLocal
        }
        out = out.padded(.replicate, begin: [0, 1, 0, 0], end: [0, 1, 0, 0])
        if paddingMode == .reflect {
          out = out.padded(.reflect, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]).reshaped([
            1, previousChannel, depth + 2, height + 2, width + 2,
          ])
        } else {
          out = out.reshaped([1, previousChannel, depth + 2, height, width])
        }
        out = conv(out)
        out = out.reshaped([
          channels, layer.stride.0, layer.stride.1, layer.stride.2, depth, height, width,
        ]).permuted(0, 4, 1, 5, 2, 6, 3).contiguous().reshaped(
          [channels, depth * layer.stride.0 - 1, height * layer.stride.1, width * layer.stride.2],
          offset: [0, 1, 0, 0],
          strides: [
            depth * layer.stride.0 * height * layer.stride.1 * width * layer.stride.2,
            height * layer.stride.1 * width * layer.stride.2, width * layer.stride.2, 1,
          ]
        ).contiguous()
        if let residualOut = residualOut {
          out = residualOut + out
        }
        depth = depth * layer.stride.0 - 1
        height = height * layer.stride.1
        width = width * layer.stride.2
      }
      previousChannel = channels
      let upBlocks = j
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["up_blocks.\(upBlocks).conv.conv.weight"] = [conv.weight.name]
        mapping["up_blocks.\(upBlocks).conv.conv.bias"] = [conv.bias.name]
        return mapping
      }
      j += 1
    } else {
      precondition(channels == previousChannel)
    }
    for i in 0..<layer.numRepeat {
      let (mapper, block) = NCHWResnetBlockCausal3D(
        prefix: "up_blocks.\(j).res_blocks.\(i)", channels: channels, depth: depth,
        height: height,
        width: width, isCausal: false, paddingMode: paddingMode)
      out = block(out)
      mappers.append(mapper)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).swish()
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 64 : 48, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(
        begin: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0],
        end: [0, paddingMode == .zero ? 1 : 0, paddingMode == .zero ? 1 : 0])),
    format: .OIHW,
    name: "conv_out")
  out = out.padded(.replicate, begin: [0, 1, 0, 0], end: [0, 1, 0, 0])
  if paddingMode == .reflect {
    out = out.padded(.reflect, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]).reshaped([
      1, previousChannel, depth + 2, height + 2, width + 2,
    ])
  } else {
    out = out.reshaped([1, previousChannel, depth + 2, height, width])
  }
  out = convOut(out)
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  out = out.reshaped([paddingFinalConvLayer ? 4 : 3, 4, 4, depth, height, width]).permuted(3, 4, 2, 5, 1, 0).contiguous()
    .reshaped(.NHWC(depth, height * 4, width * 4, paddingFinalConvLayer ? 4 : 3))

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func NCHWLTX2VideoEncoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int, startDepth: Int, paddingMode: LTX2VideoVAEPaddingMode
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for layer in layers {
    depth = (depth - 1) * layer.stride.0 + 1
    height *= layer.stride.1
    width *= layer.stride.2
  }
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  var out = x.reshaped([depth, height, 4, width, 4, 3]).permuted(5, 4, 2, 0, 1, 3).contiguous()
    .reshaped([48, depth, height, width], format: .NCHW)
  var previousChannel = layers[0].channels
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    format: .OIHW,
    name: "conv_in")
  out = convIn(
    out.padded(.replicate, begin: [0, 2, 0, 0], end: [0, 0, 0, 0]).reshaped([
      1, 48, 2 + depth, height, width,
    ])
  ).reshaped([previousChannel, depth, height, width])
  var j = 0
  var mappers = [ModelWeightMapper]()
  for layer in layers {
    let channels = layer.channels
    if layer.stride.0 > 1 || layer.stride.1 > 1 || layer.stride.2 > 1 {
      // Convolution & reshape.
      let conv = Convolution(
        groups: 1, filters: channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
        filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
        format: .OIHW,
        name: "space_to_depth_downsample")
      if layer.stride.0 == 1 {
        var residual = out.reshaped([
          previousChannel, depth, height / layer.stride.1, layer.stride.1, width / layer.stride.2,
          layer.stride.2,
        ]).permuted(0, 3, 5, 1, 2, 4).contiguous().reshaped([
          channels, previousChannel * layer.stride.1 * layer.stride.2 / channels, depth,
          height / layer.stride.1, width / layer.stride.2,
        ])
        residual = residual.reduced(.mean, axis: [1]).reshaped([
          channels, depth, height / layer.stride.1, width / layer.stride.2,
        ])
        out = conv(
          out.padded(.replicate, begin: [0, 2, 0, 0]).reshaped([
            1, previousChannel, depth + 2, height, width,
          ]))
        out = out.reshaped([
          channels / (layer.stride.1 * layer.stride.2), depth, height / layer.stride.1,
          layer.stride.1, width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 3, 5, 1, 2, 4).contiguous().reshaped([
          channels, depth, height / layer.stride.1, width / layer.stride.2,
        ])
        out = residual + out
        height = height / layer.stride.1
        width = width / layer.stride.2
      } else if layer.stride.1 == 1 && layer.stride.2 == 1 {
        var residual = out.padded(.replicate, begin: [0, 1, 0, 0]).reshaped([
          previousChannel, (depth + 1) / layer.stride.0, layer.stride.0, height, width,
        ]).permuted(0, 2, 1, 3, 4).contiguous()
        if previousChannel * layer.stride.0 / channels > 1 {
          residual = residual.reshaped([
            channels, previousChannel * layer.stride.0 / channels, (depth + 1) / layer.stride.0,
            height, width,
          ])
          residual = residual.reduced(.mean, axis: [1]).reshaped([
            channels, (depth + 1) / layer.stride.0, height, width,
          ])
        } else {
          residual = residual.reshaped([channels, (depth + 1) / layer.stride.0, height, width])
        }
        out = conv(
          out.padded(.replicate, begin: [0, 3, 0, 0]).reshaped([
            1, previousChannel, depth + 3, height, width,
          ]))
        out = out.reshaped([
          channels / layer.stride.0, (depth + 1) / layer.stride.0, layer.stride.0, height, width,
        ]).permuted(0, 2, 1, 3, 4).contiguous().reshaped([
          channels, (depth + 1) / layer.stride.0, height, width,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
      } else {
        var residual = out.padded(.replicate, begin: [0, 1, 0, 0]).reshaped([
          previousChannel, (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1,
          layer.stride.1, width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          channels,
          previousChannel * layer.stride.0 * layer.stride.1 * layer.stride.2 / channels,
          (depth + 1) / layer.stride.0, height / layer.stride.1, width / layer.stride.2,
        ])
        residual = residual.reduced(.mean, axis: [1]).reshaped([
          channels, (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2,
        ])
        out = conv(
          out.padded(.replicate, begin: [0, 3, 0, 0]).reshaped([
            1, previousChannel, depth + 3, height, width,
          ]))
        out = out.reshaped([
          channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
          (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1, layer.stride.1,
          width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          channels, (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
        height = height / layer.stride.1
        width = width / layer.stride.2
      }
      previousChannel = channels
      let downBlocks = j
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["down_blocks.\(downBlocks).conv.conv.weight"] = [conv.weight.name]
        mapping["down_blocks.\(downBlocks).conv.conv.bias"] = [conv.bias.name]
        return mapping
      }
      j += 1
    }
    for i in 0..<layer.numRepeat {
      let (mapper, block) = NCHWResnetBlockCausal3D(
        prefix: "down_blocks.\(j).res_blocks.\(i)", channels: channels, depth: depth,
        height: height, width: width, isCausal: true, paddingMode: paddingMode)
      out = block(out)
      mappers.append(mapper)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).swish()
  let convOut = Convolution(
    groups: 1, filters: 128,  // Ignore the last channel which is only useful when training 129
    filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    format: .OIHW,
    name: "conv_out")
  out = convOut(
    out.padded(.replicate, begin: [0, 2, 0, 0]).reshaped([
      1, previousChannel, depth + 2, height, width,
    ])
  ).reshaped([
    128, depth, height, width,
  ]).permuted(1, 2, 3, 0).copied()
    .reshaped(.NHWC(depth, height, width, 128))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

public func LTX2VideoDecoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int, startDepth: Int, paddingMode: LTX2VideoVAEPaddingMode, residual: Bool, paddingFinalConvLayer: Bool, format: TensorFormat
) -> (ModelWeightMapper, Model) {
  switch format {
  case .NHWC:
    return NHWCLTX2VideoDecoderCausal3D(
      layers: layers, startWidth: startWidth, startHeight: startHeight, startDepth: startDepth,
      paddingMode: paddingMode, residual: residual, paddingFinalConvLayer: paddingFinalConvLayer)
  case .NCHW:
    return NCHWLTX2VideoDecoderCausal3D(
      layers: layers, startWidth: startWidth, startHeight: startHeight, startDepth: startDepth,
      paddingMode: paddingMode, residual: residual, paddingFinalConvLayer: paddingFinalConvLayer)
  case .CHWN:
    fatalError()
  }
}

public func LTX2VideoEncoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int, startDepth: Int, paddingMode: LTX2VideoVAEPaddingMode, format: TensorFormat
) -> (ModelWeightMapper, Model) {
  switch format {
  case .NHWC:
    return NHWCLTX2VideoEncoderCausal3D(
      layers: layers, startWidth: startWidth, startHeight: startHeight, startDepth: startDepth,
      paddingMode: paddingMode)
  case .NCHW:
    return NCHWLTX2VideoEncoderCausal3D(
      layers: layers, startWidth: startWidth, startHeight: startHeight, startDepth: startDepth,
      paddingMode: paddingMode)
  case .CHWN:
    fatalError()
  }
}
