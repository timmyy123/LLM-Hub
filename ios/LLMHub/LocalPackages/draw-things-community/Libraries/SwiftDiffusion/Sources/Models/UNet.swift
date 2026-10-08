import DiffusionMappings
import Foundation
import NNC

/// UNet

public func timeEmbedding(timestep: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int)
  -> Tensor<
    Float
  >
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timestep
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

public func guidanceScaleEmbedding(guidanceScale: Float, embeddingSize: Int) -> Tensor<Float> {
  // This is only slightly different from timeEmbedding by:
  // 1. sin before cos.
  // 2. w is scaled by 1000.0
  // 3. half v.s. half - 1 when scale down.
  precondition(embeddingSize % 2 == 0)
  let guidanceScale = guidanceScale * 1000
  var embedding = Tensor<Float>(.CPU, .WC(1, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(10_000)) * Float(i) / Float(half - 1)) * guidanceScale
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    embedding[0, i] = sinFreq
    embedding[0, i + half] = cosFreq
  }
  return embedding
}

func TimeEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool, flags: Functional.GEMMFlag) -> (
  Model, Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  var out = inLayerNorm(x)
  out = out.swish()
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = inLayerConv2d(out)
  let embLayer = Dense(count: outChannels, flags: flags)
  var embOut = emb.swish()
  embOut = embLayer(embOut).reshaped([b, 1, 1, outChannels])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outLayerNorm(out)
  out = out.swish()
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let skipModel: Model?
  if skipConnection {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW)
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = nil
  }
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel,
    Model([x, emb], [out])
  )
}

func SelfAttention(
  k: Int, h: Int, b: Int, hw: Int, upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  flags: Functional.GEMMFlag, injectedAttentionKV: Bool
)
  -> (
    Model, Model, Model, Model, Model
  )
{
  let x = Input()

  let tokeys = Dense(count: k * h, noBias: true, flags: flags)
  let toqueries = Dense(count: k * h, noBias: true, flags: flags)
  let tovalues = Dense(count: k * h, noBias: true, flags: flags)
  if usesFlashAttention != .none {
    if injectedAttentionKV {
      var queries: Model.IO
      let xq = x.reshaped([b, hw, h * k], strides: [2 * hw * h * k, h * k, 1]).contiguous()
      if usesFlashAttention == .scale1 {
        queries = ((1.0 / Float(k).squareRoot()) * toqueries(xq)).reshaped([b, hw, h, k])
          .identity()
      } else {
        queries = toqueries(xq).reshaped([b, hw, h, k]).identity().identity()
      }
      let keys = tokeys(x).reshaped([b, 2 * hw, h, k]).identity()
      let values = tovalues(x).reshaped([b, 2 * hw, h, k])
      let scaledDotProductAttention: ScaledDotProductAttention
      if usesFlashAttention == .scale1 {
        scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1,
          flags: upcastAttention ? [] : [.Float16],
          multiHeadOutputProjectionFused: true)
      } else {
        scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention ? [] : [.Float16],
          multiHeadOutputProjectionFused: true)
      }
      let out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
      return (tokeys, toqueries, tovalues, scaledDotProductAttention, Model([x], [out]))
    }
    var queries: Model.IO
    if usesFlashAttention == .scale1 {
      queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k]).identity()
    } else {
      queries = toqueries(x).reshaped([b, hw, h, k]).identity().identity()
    }
    let keys = tokeys(x).reshaped([b, hw, h, k]).identity()
    let values = tovalues(x).reshaped([b, hw, h, k])
    let scaledDotProductAttention: ScaledDotProductAttention
    if usesFlashAttention == .scale1 {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1,
        flags: upcastAttention ? [] : [.Float16],
        multiHeadOutputProjectionFused: true)
    } else {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: upcastAttention ? [] : [.Float16],
        multiHeadOutputProjectionFused: true)
    }
    let out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
    return (tokeys, toqueries, tovalues, scaledDotProductAttention, Model([x], [out]))
  } else {

    if injectedAttentionKV {

      var out: Model.IO
      let xq = x.reshaped([b, hw, h * k], strides: [2 * hw * h * k, h * k, 1]).contiguous()
      let queries = ((1.0 / Float(k).squareRoot()) * toqueries(xq)).reshaped([b, hw, h, k])
        .transposed(1, 2).identity()
      // b, h, hw, k
      let keys = tokeys(x).reshaped([b, 2 * hw, h, k]).transposed(1, 2).identity()  // b, h, 2*hw,  k
      let values = tovalues(x).reshaped([b, 2 * hw, h, k]).transposed(1, 2)  // b, h, 2*hw,  k

      if b * h <= 256 {
        var outs = [Model.IO]()
        for i in 0..<(b * h) {
          var key = keys.reshaped([1, 2 * hw, k], offset: [i, 0, 0], strides: [2 * hw * k, k, 1])  // 1, 2*hw, k
          var query = queries.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])  // 1, hw, k
          if upcastAttention {
            key = key.to(.Float32)
            query = query.to(.Float32)
          }
          let value = values.reshaped(
            [1, 2 * hw, k], offset: [i, 0, 0], strides: [2 * hw * k, k, 1])  // 1, 2*hw, k
          var dot = Matmul(transposeB: (1, 2))(query, key)  // 1, hw, k @ 1, k, 2*hw -> 1, hw, 2*hw
          if let last = outs.last {
            dot.add(dependencies: [last])
          }
          dot = dot.reshaped([hw, 2 * hw])
          dot = dot.softmax()
          if upcastAttention {
            dot = dot.to(of: value)
          }
          dot = dot.reshaped([1, hw, 2 * hw])
          outs.append(dot * value)  // 1, hw, 2*hw @ 1, 2*hw, k -> 1, hw, k -> [1, hw, k]
        }
        out = Concat(axis: 0)(outs)  // -> b*h, hw, k
        out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      } else {
        var keys = keys
        var queries = queries
        if upcastAttention {
          keys = keys.to(.Float32)
          queries = queries.to(.Float32)
        }

        var dot = Matmul(transposeB: (2, 3))(queries, keys)  // b, h, hw, k @ b, h, k, 2*hw --> b, h, hw, 2*hw
        dot = dot.reshaped([b * h * hw, 2 * hw])  // b, h, hw, 2*hw
        dot = dot.softmax()
        if upcastAttention {
          dot = dot.to(of: values)
        }
        dot = dot.reshaped([b, h, hw, 2 * hw])
        out = dot * values  // b, h, hw, 2*hw @ b, h, 2*hw, k --> b, h, hw, k
        out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      }
      let unifyheads = Dense(count: k * h, flags: flags)
      out = unifyheads(out)
      return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
    }

    let keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k]).transposed(
      1, 2)
    let values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2)
    var out: Model.IO
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        var key = keys.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
        var query = queries.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
        if upcastAttention {
          key = key.to(.Float32)
          query = query.to(.Float32)
        }
        let value = values.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([hw, hw])
        dot = dot.softmax()
        if upcastAttention {
          dot = dot.to(of: value)
        }
        dot = dot.reshaped([1, hw, hw])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
    } else {
      var keys = keys
      var queries = queries
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
      }
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * hw, hw])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: values)
      }
      dot = dot.reshaped([b, h, hw, hw])
      out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
    }
    let unifyheads = Dense(count: k * h, flags: flags)
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  }
}

func CrossAttention(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel, flags: Functional.GEMMFlag
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true, flags: flags)
  let toqueries = Dense(count: k * h, noBias: true, flags: flags)
  let tovalues = Dense(count: k * h, noBias: true, flags: flags)
  if usesFlashAttention != .none {
    if t.0 == t.1 {
      let t = t.0
      var queries = toqueries(x).reshaped([b, hw, h, k]).identity().identity()
      var keys = tokeys(c).reshaped([b, t, h, k]).identity()
      var values = tovalues(c).reshaped([b, t, h, k])
      let valueType = values
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
        values = values.to(.Float32)
      }
      if injectIPAdapterLengths.count > 0 {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, h * k])
        var ipKVs = [Input]()
        for _ in injectIPAdapterLengths {
          let ipKeys = Input()
          let ipValues = Input()
          let scaledDotProductAttention = ScaledDotProductAttention(
            scale: 1.0 / Float(k).squareRoot(),
            flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
          out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
          ipKVs.append(contentsOf: [ipKeys, ipValues])
        }
        let unifyheads = Dense(count: k * h, flags: flags)
        out = unifyheads(out)
        return (tokeys, toqueries, tovalues, unifyheads, Model([x, c] + ipKVs, [out]))
      } else {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16],
          multiHeadOutputProjectionFused: true)
        var out = scaledDotProductAttention(queries, keys, values)
        if upcastAttention {
          out = out.to(of: valueType)
        }
        return (tokeys, toqueries, tovalues, scaledDotProductAttention, Model([x, c], [out]))
      }
    } else {
      var queries = toqueries(x).reshaped([b, hw, h, k]).identity()
      var keys = tokeys(c).identity()
      var values = tovalues(c)
      let valueType = values
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
        values = values.to(.Float32)
      }
      let b0 = b / 2
      let out0: Model.IO
      if b0 == 1 || t.0 >= t.1 {
        let keys0 = keys.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries0 = queries.reshaped([b0, hw, h, k])
        let values0 = values.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out0 = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
        )(queries0, keys0, values0).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys0 = keys.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries0 = queries.reshaped(
            [1, hw, h, k], offset: [i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values0 = values.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: 1.0 / Float(k).squareRoot(),
              flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
            )(queries0, keys0, values0).reshaped([1, hw, h * k]))
        }
        out0 = Concat(axis: 0)(outs)
      }
      let out1: Model.IO
      if b0 == 1 || t.1 >= t.0 {
        let keys1 = keys.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries1 = queries.reshaped(
          [b0, hw, h, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
        let values1 = values.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out1 = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
        )(queries1, keys1, values1).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys1 = keys.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries1 = queries.reshaped(
            [1, hw, h, k], offset: [b0 + i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values1 = values.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: 1.0 / Float(k).squareRoot(),
              flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
            )(queries1, keys1, values1).reshaped([1, hw, h * k]))
        }
        out1 = Concat(axis: 0)(outs)
      }
      var out = Functional.concat(axis: 0, out0, out1)
      var ipKVs = [Input]()
      for _ in injectIPAdapterLengths {
        let ipKeys = Input()
        let ipValues = Input()
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
        ipKVs.append(contentsOf: [ipKeys, ipValues])
      }
      if upcastAttention {
        out = out.to(of: valueType)
      }
      let unifyheads = Dense(count: k * h, flags: flags)
      out = unifyheads(out)
      return (tokeys, toqueries, tovalues, unifyheads, Model([x, c] + ipKVs, [out]))
    }
  } else {
    var out: Model.IO
    var keys = tokeys(c)
    var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    if t.0 == t.1 {
      let t = t.0
      keys = keys.reshaped([b, t, h, k]).transposed(1, 2)
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
      }
      let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * hw, t])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: values)
      }
      dot = dot.reshaped([b, h, hw, t])
      out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
    } else {
      let values = tovalues(c)
      let b0 = b / 2
      var keys0 = keys.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries0 = queries.reshaped([b0, h, hw, k])
      if upcastAttention {
        keys0 = keys0.to(.Float32)
        queries0 = queries0.to(.Float32)
      }
      let values0 = values.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot0 = Matmul(transposeB: (2, 3))(queries0, keys0)
      dot0 = dot0.reshaped([b0 * h * hw, t.0])
      dot0 = dot0.softmax()
      if upcastAttention {
        dot0 = dot0.to(of: values)
      }
      dot0 = dot0.reshaped([b0, h, hw, t.0])
      var out0 = dot0 * values0
      out0 = out0.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      var keys1 = keys.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries1 = queries.reshaped(
        [b0, h, hw, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, hw * k, k, 1])
      if upcastAttention {
        keys1 = keys1.to(.Float32)
        queries1 = queries1.to(.Float32)
      }
      let values1 = values.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot1 = Matmul(transposeB: (2, 3))(queries1, keys1)
      dot1.add(dependencies: [out0])
      dot1 = dot1.reshaped([b0 * h * hw, t.1])
      dot1 = dot1.softmax()
      if upcastAttention {
        dot1 = dot1.to(of: values)
      }
      dot1 = dot1.reshaped([b0, h, hw, t.1])
      var out1 = dot1 * values1
      out1 = out1.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      out = Functional.concat(axis: 0, out0, out1)
    }
    var ipKVs = [Input]()
    for injectIPAdapterLength in injectIPAdapterLengths {
      let ipKeys = Input()
      let ipValues = Input()
      var dot = Matmul(transposeB: (2, 3))(queries, ipKeys)
      dot = dot.reshaped([b * h * hw, injectIPAdapterLength])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: queries)
      }
      dot = dot.reshaped([b, h, hw, injectIPAdapterLength])
      out = out + (dot * ipValues).transposed(1, 2).reshaped([b, hw, h * k])
      ipKVs.append(contentsOf: [ipKeys, ipValues])
    }
    let unifyheads = Dense(count: k * h, flags: flags)
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x, c] + ipKVs, [out]))
  }
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, flags: Functional.GEMMFlag) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let fc10 = Dense(count: intermediateSize, flags: flags)
  let fc11 = Dense(count: intermediateSize, flags: flags)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = Dense(count: hiddenSize, flags: .Float16)
  out = fc2(out)
  return (fc10, fc11, fc2, Model([x], [out]))
}

func BasicTransformerBlock(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  flags: Functional.GEMMFlag, injectedAttentionKV: Bool
) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model
) {
  let x = Input()
  let c = Input()
  var attnKVs = [Input]()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var layerNorm1Input: Model.IO = x
  if injectedAttentionKV {
    let attnKV = Input()
    attnKVs.append(attnKV)
    let injectedKVCondition = attnKV.reshaped([b, hw, h * k])
    layerNorm1Input = Concat(axis: 1)(layerNorm1Input, injectedKVCondition)
  }
  var out = layerNorm1(layerNorm1Input)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(
    k: k, h: h, b: b, hw: hw, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, flags: flags, injectedAttentionKV: injectedAttentionKV)
  out = attn1([out]) + x

  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (tokeys2, toqueries2, tovalues2, unifyheads2, attn2) = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: flags)
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  out = attn2([out, c] + ipKVs) + residual
  residual = out

  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)

  let (fc10, fc11, fc2, ff) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, flags: flags)
  out = ff(out) + residual
  return (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2,
    Model([x, c] + ipKVs + attnKVs, [out])
  )
}

func SpatialTransformer(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  flags: Functional.GEMMFlag, injectedAttentionKV: Bool, outputSpatialAttnInput: Bool
) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model, Model, Model, Model
) {
  let x = Input()
  let c = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1], format: .OIHW)
  let hw = height * width
  out = projIn(out).reshaped([b, hw, k * h])
  let spatialAttnInput = out

  let (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, block
  ) = BasicTransformerBlock(
    k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, flags: flags, injectedAttentionKV: injectedAttentionKV)
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  var attnKVs = [Input]()
  if injectedAttentionKV {
    let attnKV = Input()
    attnKVs.append(attnKV)
  }
  out = block([out, c] + ipKVs + attnKVs).reshaped([b, height, width, k * h])

  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1], format: .OIHW)
  out = projOut(out) + x

  return (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, projOut,
    Model([x, c] + ipKVs + attnKVs, [out] + (outputSpatialAttnInput ? [spatialAttnInput] : []))
  )
}

func SpatialTransformerOnlyForSpatialAttnInput(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  flags: Functional.GEMMFlag, injectedAttentionKV: Bool, outputSpatialAttnInput: Bool
) -> Model {
  let x = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1], format: .OIHW)
  let hw = height * width
  out = projIn(out).reshaped([b, hw, k * h])
  return Model([x], [out])
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Bool, channels: Int, numHeads: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  flags: Functional.GEMMFlag, injectedAttentionKV: Bool, outputSpatialAttnInput: Bool,
  onlyOutputSpatialAttnInput: Bool = false
) -> (Model, PythonReader) {
  let x = Input()
  let emb = Input()
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection, flags: flags)
  var out = resBlock(x, emb)
  let resBlockReader: PythonReader = { stateDict, archive in
    guard
      let in_layers_0_weight =
        stateDict[
          "\(prefix).\(layerStart).0.in_layers.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_bias =
        stateDict[
          "\(prefix).\(layerStart).0.in_layers.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerNorm.parameters(for: .weight).copy(
      from: in_layers_0_weight, zip: archive, of: FloatType.self)
    try inLayerNorm.parameters(for: .bias).copy(
      from: in_layers_0_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_2_weight =
        stateDict[
          "\(prefix).\(layerStart).0.in_layers.2.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_2_bias =
        stateDict[
          "\(prefix).\(layerStart).0.in_layers.2.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerConv2d.parameters(for: .weight).copy(
      from: in_layers_2_weight, zip: archive, of: FloatType.self)
    try inLayerConv2d.parameters(for: .bias).copy(
      from: in_layers_2_bias, zip: archive, of: FloatType.self)
    guard
      let emb_layers_1_weight =
        stateDict[
          "\(prefix).\(layerStart).0.emb_layers.1.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let emb_layers_1_bias =
        stateDict[
          "\(prefix).\(layerStart).0.emb_layers.1.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try embLayer.parameters(for: .weight).copy(
      from: emb_layers_1_weight, zip: archive, of: FloatType.self)
    try embLayer.parameters(for: .bias).copy(
      from: emb_layers_1_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_weight =
        stateDict[
          "\(prefix).\(layerStart).0.out_layers.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_bias =
        stateDict[
          "\(prefix).\(layerStart).0.out_layers.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerNorm.parameters(for: .weight).copy(
      from: out_layers_0_weight, zip: archive, of: FloatType.self)
    try outLayerNorm.parameters(for: .bias).copy(
      from: out_layers_0_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_3_weight =
        stateDict[
          "\(prefix).\(layerStart).0.out_layers.3.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_3_bias =
        stateDict[
          "\(prefix).\(layerStart).0.out_layers.3.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerConv2d.parameters(for: .weight).copy(
      from: out_layers_3_weight, zip: archive, of: FloatType.self)
    try outLayerConv2d.parameters(for: .bias).copy(
      from: out_layers_3_bias, zip: archive, of: FloatType.self)
    if let skipModel = skipModel {
      guard
        let skip_connection_weight =
          stateDict[
            "\(prefix).\(layerStart).0.skip_connection.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let skip_connection_bias =
          stateDict[
            "\(prefix).\(layerStart).0.skip_connection.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try skipModel.parameters(for: .weight).copy(
        from: skip_connection_weight, zip: archive, of: FloatType.self)
      try skipModel.parameters(for: .bias).copy(
        from: skip_connection_bias, zip: archive, of: FloatType.self)
    }
  }
  if attentionBlock {
    if onlyOutputSpatialAttnInput && outputSpatialAttnInput {
      let transformer = SpatialTransformerOnlyForSpatialAttnInput(
        ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
        t: embeddingLength, intermediateSize: channels * 4,
        injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
        usesFlashAttention: usesFlashAttention, flags: flags,
        injectedAttentionKV: injectedAttentionKV, outputSpatialAttnInput: outputSpatialAttnInput)
      out = transformer(out)
      return (Model([x, emb], [out]), resBlockReader)
    }
    let c = Input()
    let (
      norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
      toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
    ) = SpatialTransformer(
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      t: embeddingLength, intermediateSize: channels * 4,
      injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
      usesFlashAttention: usesFlashAttention, flags: flags,
      injectedAttentionKV: injectedAttentionKV, outputSpatialAttnInput: outputSpatialAttnInput)
    let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
    var attnKVs = [Input]()
    if injectedAttentionKV {
      let attnKV = Input()
      attnKVs.append(attnKV)
    }
    out = transformer([out, c] + ipKVs + attnKVs)

    let reader: PythonReader = { stateDict, archive in
      try resBlockReader(stateDict, archive)
      guard
        let norm_weight =
          stateDict["\(prefix).\(layerStart).1.norm.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let norm_bias =
          stateDict["\(prefix).\(layerStart).1.norm.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try norm.parameters(for: .weight).copy(
        from: norm_weight, zip: archive, of: FloatType.self)
      try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: FloatType.self)
      guard
        let proj_in_weight =
          stateDict[
            "\(prefix).\(layerStart).1.proj_in.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let proj_in_bias =
          stateDict["\(prefix).\(layerStart).1.proj_in.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try projIn.parameters(for: .weight).copy(
        from: proj_in_weight, zip: archive, of: FloatType.self)
      try projIn.parameters(for: .bias).copy(
        from: proj_in_bias, zip: archive, of: FloatType.self)
      guard
        let attn1_to_k_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_k.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tokeys1.parameters(for: .weight).copy(
        from: attn1_to_k_weight, zip: archive, of: FloatType.self)
      guard
        let attn1_to_q_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_q.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try toqueries1.parameters(for: .weight).copy(
        from: attn1_to_q_weight, zip: archive, of: FloatType.self)
      guard
        let attn1_to_v_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_v.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tovalues1.parameters(for: .weight).copy(
        from: attn1_to_v_weight, zip: archive, of: FloatType.self)
      guard
        let attn1_to_out_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_out.0.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let attn1_to_out_bias =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_out.0.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheads1.parameters(for: .weight).copy(
        from: attn1_to_out_weight, zip: archive, of: FloatType.self)
      try unifyheads1.parameters(for: .bias).copy(
        from: attn1_to_out_bias, zip: archive, of: FloatType.self)
      guard
        let ff_net_0_proj_weight = try stateDict[
          "\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.0.proj.weight"
        ]?.inflate(from: archive, of: FloatType.self)
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let ff_net_0_proj_bias = try stateDict[
          "\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.0.proj.bias"
        ]?.inflate(from: archive, of: FloatType.self)
      else {
        throw UnpickleError.tensorNotFound
      }
      fc10.parameters(for: .weight).copy(
        from: ff_net_0_proj_weight[0..<intermediateSize, 0..<ff_net_0_proj_weight.shape[1]])
      fc10.parameters(for: .bias).copy(
        from: ff_net_0_proj_bias[0..<intermediateSize])
      fc11.parameters(for: .weight).copy(
        from: ff_net_0_proj_weight[
          intermediateSize..<ff_net_0_proj_weight.shape[0], 0..<ff_net_0_proj_weight.shape[1]])
      fc11.parameters(for: .bias).copy(
        from: ff_net_0_proj_bias[intermediateSize..<ff_net_0_proj_bias.shape[0]])
      guard
        let ff_net_2_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.2.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let ff_net_2_bias =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.2.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tfc2.parameters(for: .weight).copy(
        from: ff_net_2_weight, zip: archive, of: FloatType.self)
      try tfc2.parameters(for: .bias).copy(
        from: ff_net_2_bias, zip: archive, of: FloatType.self)
      guard
        let attn2_to_k_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_k.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tokeys2.parameters(for: .weight).copy(
        from: attn2_to_k_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_q_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_q.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try toqueries2.parameters(for: .weight).copy(
        from: attn2_to_q_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_v_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_v.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tovalues2.parameters(for: .weight).copy(
        from: attn2_to_v_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_out_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_out.0.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let attn2_to_out_bias =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_out.0.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheads2.parameters(for: .weight).copy(
        from: attn2_to_out_weight, zip: archive, of: FloatType.self)
      try unifyheads2.parameters(for: .bias).copy(
        from: attn2_to_out_bias, zip: archive, of: FloatType.self)
      guard
        let norm1_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.norm1.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let norm1_bias =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.norm1.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm1.parameters(for: .weight).copy(
        from: norm1_weight, zip: archive, of: FloatType.self)
      try layerNorm1.parameters(for: .bias).copy(
        from: norm1_bias, zip: archive, of: FloatType.self)
      guard
        let norm2_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.norm2.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let norm2_bias =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.norm2.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm2.parameters(for: .weight).copy(
        from: norm2_weight, zip: archive, of: FloatType.self)
      try layerNorm2.parameters(for: .bias).copy(
        from: norm2_bias, zip: archive, of: FloatType.self)
      guard
        let norm3_weight =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.norm3.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let norm3_bias =
          stateDict[
            "\(prefix).\(layerStart).1.transformer_blocks.0.norm3.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm3.parameters(for: .weight).copy(
        from: norm3_weight, zip: archive, of: FloatType.self)
      try layerNorm3.parameters(for: .bias).copy(
        from: norm3_bias, zip: archive, of: FloatType.self)
      guard
        let proj_out_weight =
          stateDict[
            "\(prefix).\(layerStart).1.proj_out.weight"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let proj_out_bias =
          stateDict[
            "\(prefix).\(layerStart).1.proj_out.bias"
          ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try projOut.parameters(for: .weight).copy(
        from: proj_out_weight, zip: archive, of: FloatType.self)
      try projOut.parameters(for: .bias).copy(
        from: proj_out_bias, zip: archive, of: FloatType.self)
    }
    return (
      Model([x, emb, c] + ipKVs + attnKVs, [out]), reader
    )
  } else {
    return (Model([x, emb], [out]), resBlockReader)
  }
}

func MiddleBlock(
  prefix: String,
  channels: Int, numHeads: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  x: Model.IO, emb: Model.IO, c: Model.IO, injectedAttentionKVs: [Model.IO],
  outputSpatialAttnInput: Bool
) -> (Model.IO, [Input], PythonReader, [Model.IO]) {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false, flags: .Float16)
  var attnKVs = [Model.IO]()

  var out = resBlock1(x, emb)
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  let (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
  ) = SpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingLength,
    intermediateSize: channels * 4, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: .Float16,
    injectedAttentionKV: !injectedAttentionKVs.isEmpty,
    outputSpatialAttnInput: outputSpatialAttnInput)
  if !injectedAttentionKVs.isEmpty {
    let attnKV = injectedAttentionKVs[6]
    attnKVs.append(attnKV)
  }
  out = transformer([out, c] + ipKVs + attnKVs)
  let spatialOutputs: [Model.IO]
  if outputSpatialAttnInput {
    spatialOutputs = [out[1]]
    out = out[0]
  } else {
    spatialOutputs = []
  }

  let (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false, flags: .Float16)
  out = resBlock2(out, emb)
  let reader: PythonReader = { stateDict, archive in
    let intermediateSize = channels * 4
    guard
      let in_layers_0_0_weight =
        stateDict["\(prefix).middle_block.0.in_layers.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_0_bias =
        stateDict["\(prefix).middle_block.0.in_layers.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerNorm1.parameters(for: .weight).copy(
      from: in_layers_0_0_weight, zip: archive, of: FloatType.self)
    try inLayerNorm1.parameters(for: .bias).copy(
      from: in_layers_0_0_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_0_2_weight =
        stateDict["\(prefix).middle_block.0.in_layers.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_2_bias =
        stateDict["\(prefix).middle_block.0.in_layers.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerConv2d1.parameters(for: .weight).copy(
      from: in_layers_0_2_weight, zip: archive, of: FloatType.self)
    try inLayerConv2d1.parameters(for: .bias).copy(
      from: in_layers_0_2_bias, zip: archive, of: FloatType.self)
    guard
      let emb_layers_0_1_weight =
        stateDict[
          "\(prefix).middle_block.0.emb_layers.1.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let emb_layers_0_1_bias =
        stateDict["\(prefix).middle_block.0.emb_layers.1.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try embLayer1.parameters(for: .weight).copy(
      from: emb_layers_0_1_weight, zip: archive, of: FloatType.self)
    try embLayer1.parameters(for: .bias).copy(
      from: emb_layers_0_1_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_0_weight =
        stateDict[
          "\(prefix).middle_block.0.out_layers.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_0_bias =
        stateDict[
          "\(prefix).middle_block.0.out_layers.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerNorm1.parameters(for: .weight).copy(
      from: out_layers_0_0_weight, zip: archive, of: FloatType.self)
    try outLayerNorm1.parameters(for: .bias).copy(
      from: out_layers_0_0_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_3_weight =
        stateDict[
          "\(prefix).middle_block.0.out_layers.3.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_3_bias =
        stateDict["\(prefix).middle_block.0.out_layers.3.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerConv2d1.parameters(for: .weight).copy(
      from: out_layers_0_3_weight, zip: archive, of: FloatType.self)
    try outLayerConv2d1.parameters(for: .bias).copy(
      from: out_layers_0_3_bias, zip: archive, of: FloatType.self)
    guard
      let norm_weight =
        stateDict["\(prefix).middle_block.1.norm.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm_bias =
        stateDict["\(prefix).middle_block.1.norm.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try norm.parameters(for: .weight).copy(
      from: norm_weight, zip: archive, of: FloatType.self)
    try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: FloatType.self)
    guard
      let proj_in_weight =
        stateDict["\(prefix).middle_block.1.proj_in.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let proj_in_bias =
        stateDict["\(prefix).middle_block.1.proj_in.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try projIn.parameters(for: .weight).copy(
      from: proj_in_weight, zip: archive, of: FloatType.self)
    try projIn.parameters(for: .bias).copy(
      from: proj_in_bias, zip: archive, of: FloatType.self)
    guard
      let attn1_to_k_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_k.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys1.parameters(for: .weight).copy(
      from: attn1_to_k_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_q_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_q.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries1.parameters(for: .weight).copy(
      from: attn1_to_q_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_v_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_v.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues1.parameters(for: .weight).copy(
      from: attn1_to_v_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_out_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let attn1_to_out_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn1.to_out.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try unifyheads1.parameters(for: .weight).copy(
      from: attn1_to_out_weight, zip: archive, of: FloatType.self)
    try unifyheads1.parameters(for: .bias).copy(
      from: attn1_to_out_bias, zip: archive, of: FloatType.self)
    guard
      let ff_net_0_proj_weight = try stateDict[
        "\(prefix).middle_block.1.transformer_blocks.0.ff.net.0.proj.weight"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_0_proj_bias = try stateDict[
        "\(prefix).middle_block.1.transformer_blocks.0.ff.net.0.proj.bias"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    fc10.parameters(for: .weight).copy(
      from: ff_net_0_proj_weight[0..<intermediateSize, 0..<ff_net_0_proj_weight.shape[1]])
    fc10.parameters(for: .bias).copy(
      from: ff_net_0_proj_bias[0..<intermediateSize])
    fc11.parameters(for: .weight).copy(
      from: ff_net_0_proj_weight[
        intermediateSize..<ff_net_0_proj_weight.shape[0], 0..<ff_net_0_proj_weight.shape[1]])
    fc11.parameters(for: .bias).copy(
      from: ff_net_0_proj_bias[intermediateSize..<ff_net_0_proj_bias.shape[0]])
    guard
      let ff_net_2_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.ff.net.2.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_2_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.ff.net.2.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tfc2.parameters(for: .weight).copy(
      from: ff_net_2_weight, zip: archive, of: FloatType.self)
    try tfc2.parameters(for: .bias).copy(
      from: ff_net_2_bias, zip: archive, of: FloatType.self)
    guard
      let attn2_to_k_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_k.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys2.parameters(for: .weight).copy(
      from: attn2_to_k_weight, zip: archive, of: FloatType.self)
    guard
      let attn2_to_q_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_q.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries2.parameters(for: .weight).copy(
      from: attn2_to_q_weight, zip: archive, of: FloatType.self)
    guard
      let attn2_to_v_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_v.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues2.parameters(for: .weight).copy(
      from: attn2_to_v_weight, zip: archive, of: FloatType.self)
    guard
      let attn2_to_out_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_out.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let attn2_to_out_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.attn2.to_out.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try unifyheads2.parameters(for: .weight).copy(
      from: attn2_to_out_weight, zip: archive, of: FloatType.self)
    try unifyheads2.parameters(for: .bias).copy(
      from: attn2_to_out_bias, zip: archive, of: FloatType.self)
    guard
      let norm1_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm1.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm1_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm1.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm1.parameters(for: .weight).copy(
      from: norm1_weight, zip: archive, of: FloatType.self)
    try layerNorm1.parameters(for: .bias).copy(
      from: norm1_bias, zip: archive, of: FloatType.self)
    guard
      let norm2_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm2.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm2_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm2.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm2.parameters(for: .weight).copy(
      from: norm2_weight, zip: archive, of: FloatType.self)
    try layerNorm2.parameters(for: .bias).copy(
      from: norm2_bias, zip: archive, of: FloatType.self)
    guard
      let norm3_weight =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm3.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm3_bias =
        stateDict[
          "\(prefix).middle_block.1.transformer_blocks.0.norm3.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm3.parameters(for: .weight).copy(
      from: norm3_weight, zip: archive, of: FloatType.self)
    try layerNorm3.parameters(for: .bias).copy(
      from: norm3_bias, zip: archive, of: FloatType.self)
    guard
      let proj_out_weight =
        stateDict[
          "\(prefix).middle_block.1.proj_out.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let proj_out_bias =
        stateDict["\(prefix).middle_block.1.proj_out.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try projOut.parameters(for: .weight).copy(
      from: proj_out_weight, zip: archive, of: FloatType.self)
    try projOut.parameters(for: .bias).copy(
      from: proj_out_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_2_0_weight =
        stateDict["\(prefix).middle_block.2.in_layers.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_2_0_bias =
        stateDict["\(prefix).middle_block.2.in_layers.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerNorm2.parameters(for: .weight).copy(
      from: in_layers_2_0_weight, zip: archive, of: FloatType.self)
    try inLayerNorm2.parameters(for: .bias).copy(
      from: in_layers_2_0_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_2_2_weight =
        stateDict["\(prefix).middle_block.2.in_layers.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_2_2_bias =
        stateDict["\(prefix).middle_block.2.in_layers.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerConv2d2.parameters(for: .weight).copy(
      from: in_layers_2_2_weight, zip: archive, of: FloatType.self)
    try inLayerConv2d2.parameters(for: .bias).copy(
      from: in_layers_2_2_bias, zip: archive, of: FloatType.self)
    guard
      let emb_layers_2_1_weight =
        stateDict[
          "\(prefix).middle_block.2.emb_layers.1.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let emb_layers_2_1_bias =
        stateDict["\(prefix).middle_block.2.emb_layers.1.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try embLayer2.parameters(for: .weight).copy(
      from: emb_layers_2_1_weight, zip: archive, of: FloatType.self)
    try embLayer2.parameters(for: .bias).copy(
      from: emb_layers_2_1_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_2_0_weight =
        stateDict[
          "\(prefix).middle_block.2.out_layers.0.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_2_0_bias =
        stateDict[
          "\(prefix).middle_block.2.out_layers.0.bias"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerNorm2.parameters(for: .weight).copy(
      from: out_layers_2_0_weight, zip: archive, of: FloatType.self)
    try outLayerNorm2.parameters(for: .bias).copy(
      from: out_layers_2_0_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_2_3_weight =
        stateDict[
          "\(prefix).middle_block.2.out_layers.3.weight"
        ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_2_3_bias =
        stateDict["\(prefix).middle_block.2.out_layers.3.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerConv2d2.parameters(for: .weight).copy(
      from: out_layers_2_3_weight, zip: archive, of: FloatType.self)
    try outLayerConv2d2.parameters(for: .bias).copy(
      from: out_layers_2_3_bias, zip: archive, of: FloatType.self)
  }
  return (out, ipKVs, reader, spatialOutputs)
}

private func InputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingLength: (Int, Int), attentionRes: Set<Int>, injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel, x: Model.IO, emb: Model.IO,
  c: Model.IO, adapters: [Model.IO], injectedAttentionKVs: [Model.IO], outputSpatialAttnInput: Bool
) -> ([Model.IO], Model.IO, [Input], PythonReader, [Model.IO]) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [PythonReader]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  var spatialAttnInputs = [Model.IO]()
  var attnKVs = [Model.IO]()

  var attnKVIdx = 0
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<numRepeat {
      let (inputLayer, reader) = BlockLayer(
        prefix: "model.diffusion_model.input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: .Float16,
        injectedAttentionKV: !injectedAttentionKVs.isEmpty,
        outputSpatialAttnInput: outputSpatialAttnInput)
      previousChannel = channel
      if attentionBlock {
        let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
        var attnKV = [Model.IO]()
        if attnKVIdx < injectedAttentionKVs.count {
          let injectedAttentionKV = injectedAttentionKVs[attnKVIdx]
          attnKVIdx += 1
          attnKVs.append(injectedAttentionKV)
          attnKV.append(injectedAttentionKV)
        }
        out = inputLayer([out, emb, c] + ipKVs + attnKV)
        if outputSpatialAttnInput {
          spatialAttnInputs.append(out[1])
          out = out[0]
        }
        kvs.append(contentsOf: ipKVs)
      } else {
        out = inputLayer(out, emb)
      }
      if j == numRepeat - 1 && adapters.count == channels.count {
        out = out + adapters[i]
      }
      passLayers.append(out)
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: PythonReader = { stateDict, archive in
        guard
          let op_weight =
            stateDict["model.diffusion_model.input_blocks.\(downLayer).0.op.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let op_bias =
            stateDict["model.diffusion_model.input_blocks.\(downLayer).0.op.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try downsample.parameters(for: .weight).copy(
          from: op_weight, zip: archive, of: FloatType.self)
        try downsample.parameters(for: .bias).copy(
          from: op_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: PythonReader = { stateDict, archive in
    guard
      let input_blocks_0_0_weight =
        stateDict["model.diffusion_model.input_blocks.0.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let input_blocks_0_0_bias =
        stateDict["model.diffusion_model.input_blocks.0.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d.parameters(for: .weight).copy(
      from: input_blocks_0_0_weight, zip: archive, of: FloatType.self)
    try conv2d.parameters(for: .bias).copy(
      from: input_blocks_0_0_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  return (passLayers, out, kvs, reader, spatialAttnInputs)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingLength: (Int, Int), attentionRes: Set<Int>, injectIPAdapterLengths: [Int],
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel, x: Model.IO, emb: Model.IO,
  c: Model.IO, inputs: [Model.IO], injectedAttentionKVs: [Model.IO], outputSpatialAttnInput: Bool
) -> (Model.IO, [Input], PythonReader, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [PythonReader]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var spatialAttnInputs = [Model.IO]()
  var inputIdx = inputs.count - 1
  var attnKVIdx = 7
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 3)(out, inputs[inputIdx])
      inputIdx -= 1
      let onlyOutputSpatialAttnInput = (outputSpatialAttnInput && j == numRepeat && i == 0)
      let (outputLayer, reader) = BlockLayer(
        prefix: "model.diffusion_model.output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: .Float16,
        injectedAttentionKV: !injectedAttentionKVs.isEmpty,
        outputSpatialAttnInput: outputSpatialAttnInput,
        onlyOutputSpatialAttnInput: onlyOutputSpatialAttnInput)
      if attentionBlock {
        if onlyOutputSpatialAttnInput {
          out = outputLayer([out, emb])
          spatialAttnInputs.append(out)
        } else {
          var attnKV: Model.IO?
          if !injectedAttentionKVs.isEmpty && attnKVIdx < injectedAttentionKVs.count {
            attnKV = injectedAttentionKVs[attnKVIdx]
            attnKVIdx += 1
          }
          let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
          if let attnKV = attnKV {
            out = outputLayer([out, emb, c] + ipKVs + [attnKV])
            if outputSpatialAttnInput {
              spatialAttnInputs.append(out[1])
              out = out[0]
            }
          } else {
            out = outputLayer([out, emb, c] + ipKVs)
            if outputSpatialAttnInput {
              spatialAttnInputs.append(out[1])
              out = out[0]
            }
          }
          kvs.append(contentsOf: ipKVs)
        }
      } else {
        out = outputLayer(out, emb)
      }
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW
        )
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock ? 2 : 1
        let reader: PythonReader = { stateDict, archive in
          guard
            let op_weight =
              stateDict[
                "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight"
              ]
          else {
            throw UnpickleError.tensorNotFound
          }
          guard
            let op_bias =
              stateDict[
                "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias"
              ]
          else {
            throw UnpickleError.tensorNotFound
          }
          try conv2d.parameters(for: .weight).copy(
            from: op_weight, zip: archive, of: FloatType.self)
          try conv2d.parameters(for: .bias).copy(
            from: op_bias, zip: archive, of: FloatType.self)
        }
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: PythonReader = { stateDict, archive in
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  return (out, kvs, reader, spatialAttnInputs)
}

public func UNet(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  usesFlashAttention: FlashAttentionLevel, injectControls: Bool, injectT2IAdapters: Bool,
  injectIPAdapterLengths: [Int], injectAttentionKV: Bool,
  trainable: Bool? = nil
) -> (
  Model, PythonReader
) {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  var injectedControls = [Model.IO]()
  if injectControls {
    injectedControls = (0..<13).map { _ in Input() }
  }
  var injectedT2IAdapters = [Model.IO]()
  if injectT2IAdapters {
    injectedT2IAdapters = (0..<4).map { _ in Input() }
  }
  var injectedAttentionKVs = [Model.IO]()
  if injectAttentionKV {
    injectedAttentionKVs = (0..<16).map { _ in Input() }
  }
  let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  var (inputs, inputBlocks, inputKVs, inputReader, _) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, x: x, emb: emb, c: c, adapters: injectedT2IAdapters,
    injectedAttentionKVs: injectedAttentionKVs, outputSpatialAttnInput: false)

  var out = inputBlocks
  let (middleBlock, middleKVs, middleReader, _) = MiddleBlock(
    prefix: "model.diffusion_model",
    channels: 1280, numHeads: 8, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, x: out, emb: emb, c: c,
    injectedAttentionKVs: injectedAttentionKVs, outputSpatialAttnInput: false)

  out = middleBlock
  if injectControls {
    out = out + injectedControls[12]
    precondition(inputs.count + 1 == injectedControls.count)
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + injectedControls[i]
    }
  }
  let (outputBlocks, outputKVs, outputReader, _) = OutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, x: out, emb: emb, c: c, inputs: inputs,
    injectedAttentionKVs: injectedAttentionKVs, outputSpatialAttnInput: false)

  out = outputBlocks
  let outNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outNorm(out)
  out = out.swish()
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = outConv2d(out)
  let reader: PythonReader = { stateDict, archive in
    guard
      let time_embed_0_weight =
        stateDict["model.diffusion_model.time_embed.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_0_bias =
        stateDict["model.diffusion_model.time_embed.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_weight =
        stateDict["model.diffusion_model.time_embed.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_bias =
        stateDict["model.diffusion_model.time_embed.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try fc0.parameters(for: .weight).copy(
      from: time_embed_0_weight, zip: archive, of: FloatType.self)
    try fc0.parameters(for: .bias).copy(
      from: time_embed_0_bias, zip: archive, of: FloatType.self)
    try fc2.parameters(for: .weight).copy(
      from: time_embed_2_weight, zip: archive, of: FloatType.self)
    try fc2.parameters(for: .bias).copy(
      from: time_embed_2_bias, zip: archive, of: FloatType.self)
    try inputReader(stateDict, archive)
    try middleReader(stateDict, archive)
    try outputReader(stateDict, archive)
    guard let out_0_weight = stateDict["model.diffusion_model.out.0.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let out_0_bias = stateDict["model.diffusion_model.out.0.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try outNorm.parameters(for: .weight).copy(
      from: out_0_weight, zip: archive, of: FloatType.self)
    try outNorm.parameters(for: .bias).copy(
      from: out_0_bias, zip: archive, of: FloatType.self)
    guard let out_2_weight = stateDict["model.diffusion_model.out.2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let out_2_bias = stateDict["model.diffusion_model.out.2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try outConv2d.parameters(for: .weight).copy(
      from: out_2_weight, zip: archive, of: FloatType.self)
    try outConv2d.parameters(for: .bias).copy(
      from: out_2_bias, zip: archive, of: FloatType.self)
  }
  var modelInputs: [Model.IO] = [x, t_emb, c]
  modelInputs.append(contentsOf: inputKVs)
  modelInputs.append(contentsOf: middleKVs)
  modelInputs.append(contentsOf: outputKVs)
  modelInputs.append(contentsOf: injectedControls)
  modelInputs.append(contentsOf: injectedT2IAdapters)
  modelInputs.append(contentsOf: injectedAttentionKVs)
  return (Model(modelInputs, [out], trainable: trainable), reader)
}

public func GarmentUNet(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  usesFlashAttention: FlashAttentionLevel, injectControls: Bool, injectT2IAdapters: Bool,
  injectIPAdapterLengths: [Int], injectAttentionKV: Bool,
  trainable: Bool? = nil
) -> (
  Model, PythonReader
) {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  var spatialAttnInputs = [Model.IO]()
  var injectedControls = [Model.IO]()
  if injectControls {
    injectedControls = (0..<13).map { _ in Input() }
  }
  var injectedT2IAdapters = [Model.IO]()
  if injectT2IAdapters {
    injectedT2IAdapters = (0..<4).map { _ in Input() }
  }
  var injectedAttentionKVs = [Model.IO]()
  if injectAttentionKV {
    injectedAttentionKVs = (0..<16).map { _ in Input() }
  }
  let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  var (inputs, inputBlocks, inputKVs, inputReader, inSpatialAttnInputs) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, x: x, emb: emb, c: c, adapters: injectedT2IAdapters,
    injectedAttentionKVs: injectedAttentionKVs, outputSpatialAttnInput: true)
  spatialAttnInputs.append(contentsOf: inSpatialAttnInputs)

  var out = inputBlocks
  let (middleBlock, middleKVs, middleReader, middleSpatialAttnInput) = MiddleBlock(
    prefix: "model.diffusion_model",
    channels: 1280, numHeads: 8, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, x: out, emb: emb, c: c,
    injectedAttentionKVs: injectedAttentionKVs, outputSpatialAttnInput: true)
  spatialAttnInputs.append(contentsOf: middleSpatialAttnInput)

  out = middleBlock
  if injectControls {
    out = out + injectedControls[12]
    precondition(inputs.count + 1 == injectedControls.count)
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + injectedControls[i]
    }
  }
  let (_, outputKVs, outputReader, outputSpatialAttnInputs) = OutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, x: out, emb: emb, c: c, inputs: inputs,
    injectedAttentionKVs: injectedAttentionKVs, outputSpatialAttnInput: true)

  spatialAttnInputs.append(contentsOf: outputSpatialAttnInputs)

  let reader: PythonReader = { stateDict, archive in
    guard
      let time_embed_0_weight =
        stateDict["model.diffusion_model.time_embed.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_0_bias =
        stateDict["model.diffusion_model.time_embed.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_weight =
        stateDict["model.diffusion_model.time_embed.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_embed_2_bias =
        stateDict["model.diffusion_model.time_embed.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try fc0.parameters(for: .weight).copy(
      from: time_embed_0_weight, zip: archive, of: FloatType.self)
    try fc0.parameters(for: .bias).copy(
      from: time_embed_0_bias, zip: archive, of: FloatType.self)
    try fc2.parameters(for: .weight).copy(
      from: time_embed_2_weight, zip: archive, of: FloatType.self)
    try fc2.parameters(for: .bias).copy(
      from: time_embed_2_bias, zip: archive, of: FloatType.self)
    try inputReader(stateDict, archive)
    try middleReader(stateDict, archive)
    try outputReader(stateDict, archive)
  }
  var modelInputs: [Model.IO] = [x, t_emb, c]
  modelInputs.append(contentsOf: inputKVs)
  modelInputs.append(contentsOf: middleKVs)
  modelInputs.append(contentsOf: outputKVs)
  modelInputs.append(contentsOf: injectedControls)
  modelInputs.append(contentsOf: injectedT2IAdapters)
  modelInputs.append(contentsOf: injectedAttentionKVs)
  return (Model(modelInputs, spatialAttnInputs, trainable: trainable), reader)
}

private func BlockLayerFixed(
  prefix: (String, String),
  repeatStart: Int, skipConnection: Bool, attentionBlock: Bool, channels: Int, numHeads: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: (Int, Int), intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (_, mapper, transformer) = SpatialTransformerFixed(
    prefix: ("\(prefix.0).1", "\(prefix.1).attentions.\(repeatStart)"),
    ch: channels, k: k, h: numHeads, b: batchSize,
    depth: 1, t: embeddingLength,
    intermediateSize: channels * 4, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: false)
  return (mapper, transformer)
}

private func MiddleBlockFixed(
  prefix: String,
  channels: Int, numHeads: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int),
  upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  c: Model.IO
) -> (ModelWeightMapper, Model.IO) {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (_, mapper, transformer) = SpatialTransformerFixed(
    prefix: ("middle_block.1", "mid_block.attentions.0"),
    ch: channels, k: k, h: numHeads, b: batchSize, depth: 1,
    t: embeddingLength, intermediateSize: channels * 4, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: false)
  let out = transformer(c)
  return (mapper, out)
}

private func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, c: Model.IO
) -> (ModelWeightMapper, [Model.IO]) {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var mappers = [ModelWeightMapper]()
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<numRepeat {
      if attentionBlock {
        let (inputMapper, inputLayer) = BlockLayerFixed(
          prefix: ("input_blocks.\(layerStart)", "down_blocks.\(i)"),
          repeatStart: j, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeads: numHeads,
          batchSize: batchSize,
          height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention)
        previousChannel = channel
        outs.append(inputLayer(c))
        mappers.append(inputMapper)
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, outs)
}

private func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingLength: (Int, Int), attentionRes: Set<Int>, upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, c: Model.IO
) -> (ModelWeightMapper, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var mappers = [ModelWeightMapper]()
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<(numRepeat + 1) {
      if attentionBlock {
        let (outputMapper, outputLayer) = BlockLayerFixed(
          prefix: ("output_blocks.\(layerStart)", "up_blocks.\(channels.count - 1 - i)"),
          repeatStart: j, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeads: numHeads,
          batchSize: batchSize,
          height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention)
        outs.append(outputLayer(c))
        mappers.append(outputMapper)
      }
      layerStart += 1
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, outs)
}

public func UNetIPFixed(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let attentionRes = Set([4, 2, 1])
  var (inputMapper, outs) = InputBlocksFixed(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    upcastAttention: false, usesFlashAttention: usesFlashAttention, c: c)
  let (middleMapper, middleBlock) = MiddleBlockFixed(
    prefix: "model.diffusion_model",
    channels: 1280, numHeads: 8, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingLength: embeddingLength, upcastAttention: false,
    usesFlashAttention: usesFlashAttention, c: c)
  outs.append(middleBlock)
  let (outputMapper, outputBlocks) = OutputBlocksFixed(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingLength: embeddingLength, attentionRes: attentionRes,
    upcastAttention: false, usesFlashAttention: usesFlashAttention, c: c)
  outs.append(contentsOf: outputBlocks)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    mapping.merge(middleMapper(format)) { v, _ in v }
    mapping.merge(outputMapper(format)) { v, _ in v }
    return mapping
  }
  return (mapper, Model([c], outs))
}
