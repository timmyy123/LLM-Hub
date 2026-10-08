import DiffusionMappings
import Foundation
import NNC

// Whisper-large-v3 encoder used as the audio conditioner for LongCat-Video-Avatar-1.5.
// The LongCat pipeline consumes all 33 encoder hidden states (the conv+position embedding and
// every transformer layer input, plus the final layer-normed output) grouped into 5 features:
// mean(h[0:8]), mean(h[8:16]), mean(h[16:24]), mean(h[24:32]) and h[32]. To avoid shipping 33
// full hidden states across the model boundary, the grouping happens inside the graph and the
// model emits the 5 grouped features directly, each [1, frames, 1280].

/// Log-mel spectrogram matching HF WhisperFeatureExtractor for whisper-large-v3:
/// 16kHz mono input, n_fft 400, hop 160, 128 slaney-scale/slaney-norm mel filters, fmax 8kHz.
/// The input chunk is zero-padded to 30s (480,000 samples) and produces [1, 128, 3000].
public func WhisperMelSpectrogram(samples: [Float]) -> Tensor<Float> {
  let sampleRate = 16_000
  let nFFT = 400
  let hopLength = 160
  let melBins = 128
  let chunkSamples = 30 * sampleRate
  let frames = chunkSamples / hopLength  // 3,001 frames computed, last one dropped -> 3,000.
  let freqBins = nFFT / 2 + 1
  var padded = [Float](repeating: 0, count: chunkSamples)
  for i in 0..<min(samples.count, chunkSamples) {
    padded[i] = samples[i]
  }
  // Hann window (periodic).
  var window = [Float](repeating: 0, count: nFFT)
  for i in 0..<nFFT {
    window[i] = Float(0.5 * (1 - cos(2 * Double.pi * Double(i) / Double(nFFT))))
  }
  // DFT basis tables.
  var cosTable = [Float](repeating: 0, count: freqBins * nFFT)
  var sinTable = [Float](repeating: 0, count: freqBins * nFFT)
  for k in 0..<freqBins {
    for i in 0..<nFFT {
      let theta = -2 * Double.pi * Double(k) * Double(i) / Double(nFFT)
      cosTable[k * nFFT + i] = Float(cos(theta))
      sinTable[k * nFFT + i] = Float(sin(theta))
    }
  }
  // Slaney-scale mel filter bank with slaney normalization (librosa-compatible).
  func hzToMel(_ f: Double) -> Double {
    if f < 1_000 {
      return 3 * f / 200
    }
    return 15 + 27 * log(f / 1_000) / log(6.4)
  }
  func melToHz(_ m: Double) -> Double {
    if m < 15 {
      return 200 * m / 3
    }
    return 1_000 * exp(log(6.4) * (m - 15) / 27)
  }
  let melPoints = (0..<(melBins + 2)).map {
    melToHz(hzToMel(8_000) * Double($0) / Double(melBins + 1))
  }
  var melFilters = [Float](repeating: 0, count: melBins * freqBins)
  for m in 0..<melBins {
    let fLow = melPoints[m]
    let fCenter = melPoints[m + 1]
    let fHigh = melPoints[m + 2]
    let norm = 2 / (fHigh - fLow)
    for k in 0..<freqBins {
      let freq = Double(k) * Double(sampleRate) / Double(nFFT)
      let lower = (freq - fLow) / max(fCenter - fLow, 1e-10)
      let upper = (fHigh - freq) / max(fHigh - fCenter, 1e-10)
      let weight = max(0, min(lower, upper))
      melFilters[m * freqBins + k] = Float(weight * norm)
    }
  }
  // Reflect-padded framing (torch.stft center=True) + power spectrum + mel projection.
  let outFrames = frames  // 3,000 after dropping the trailing frame.
  var logSpec = [Float](repeating: 0, count: melBins * outFrames)
  var frame = [Float](repeating: 0, count: nFFT)
  var power = [Float](repeating: 0, count: freqBins)
  let pad = nFFT / 2
  var maxVal = -Float.greatestFiniteMagnitude
  for t in 0..<outFrames {
    let start = t * hopLength - pad
    for i in 0..<nFFT {
      var idx = start + i
      if idx < 0 {
        idx = -idx
      } else if idx >= chunkSamples {
        idx = 2 * (chunkSamples - 1) - idx
      }
      frame[i] = padded[idx] * window[i]
    }
    for k in 0..<freqBins {
      var re: Float = 0
      var im: Float = 0
      for i in 0..<nFFT {
        re += frame[i] * cosTable[k * nFFT + i]
        im += frame[i] * sinTable[k * nFFT + i]
      }
      power[k] = re * re + im * im
    }
    for m in 0..<melBins {
      var acc: Float = 0
      for k in 0..<freqBins {
        acc += melFilters[m * freqBins + k] * power[k]
      }
      let value = log10(max(acc, 1e-10))
      logSpec[m * outFrames + t] = value
      maxVal = max(maxVal, value)
    }
  }
  let floorVal = maxVal - 8
  var tensor = Tensor<Float>(.CPU, .HWC(1, melBins, outFrames))
  for m in 0..<melBins {
    for t in 0..<outFrames {
      tensor[0, m, t] = (max(logSpec[m * outFrames + t], floorVal) + 4) / 4
    }
  }
  return tensor
}

private func WhisperEncoderLayer(
  prefix: String, k: Int, h: Int, b: Int, t: Int, intermediateSize: Int,
  usesFlashAttention: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "attn_norm")
  var out = norm1(x)
  let toKeys = Dense(count: k * h, noBias: true, name: "k_proj")
  let toQueries = Dense(count: k * h, name: "q_proj")
  let toValues = Dense(count: k * h, name: "v_proj")
  let keys = toKeys(out).reshaped([b, t, h, k])
  let queries = toQueries(out).reshaped([b, t, h, k])
  let values = toValues(out).reshaped([b, t, h, k])
  var attnOut: Model.IO
  if usesFlashAttention {
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1 / Float(k).squareRoot(), flags: [.Float16])
    attnOut = scaledDotProductAttention(queries, keys, values).reshaped([b, t, h * k])
  } else {
    let keysT = keys.transposed(1, 2)
    let queriesT = ((1 / Float(k).squareRoot()) * queries).transposed(1, 2)
    let valuesT = values.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queriesT, keysT)
    dot = dot.reshaped([b * h, t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    attnOut = (dot * valuesT).reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
  }
  let unifyheads = Dense(count: k * h, name: "out_proj")
  out = x + unifyheads(attnOut)
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "ffn_norm")
  let fc1 = Dense(count: intermediateSize, name: "fc1")
  let fc2 = Dense(count: k * h, name: "fc2")
  out = out + fc2(fc1(norm2(out)).GELU())
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).self_attn_layer_norm.weight"] = [norm1.weight.name]
    mapping["\(prefix).self_attn_layer_norm.bias"] = [norm1.bias.name]
    mapping["\(prefix).self_attn.q_proj.weight"] = [toQueries.weight.name]
    mapping["\(prefix).self_attn.q_proj.bias"] = [toQueries.bias.name]
    mapping["\(prefix).self_attn.k_proj.weight"] = [toKeys.weight.name]
    mapping["\(prefix).self_attn.v_proj.weight"] = [toValues.weight.name]
    mapping["\(prefix).self_attn.v_proj.bias"] = [toValues.bias.name]
    mapping["\(prefix).self_attn.out_proj.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).self_attn.out_proj.bias"] = [unifyheads.bias.name]
    mapping["\(prefix).final_layer_norm.weight"] = [norm2.weight.name]
    mapping["\(prefix).final_layer_norm.bias"] = [norm2.bias.name]
    mapping["\(prefix).fc1.weight"] = [fc1.weight.name]
    mapping["\(prefix).fc1.bias"] = [fc1.bias.name]
    mapping["\(prefix).fc2.weight"] = [fc2.weight.name]
    mapping["\(prefix).fc2.bias"] = [fc2.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

/// Whisper encoder over one 30s mel chunk [1, melBins, 1, frames] (NCHW-style 4D for the 1D
/// convolutions), emitting the 5 grouped hidden-state features, each [1, frames / 2, width].
public func WhisperEncoder(
  width: Int, layers: Int, heads: Int, melBins: Int, frames: Int, intermediateSize: Int,
  usesFlashAttention: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let t = frames / 2
  let k = width / heads
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [1, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 1], end: [0, 1])), format: .OIHW,
    name: "conv1")
  var out = conv1(x).GELU()
  let conv2 = Convolution(
    groups: 1, filters: width, filterSize: [1, 3],
    hint: Hint(stride: [1, 2], border: Hint.Border(begin: [0, 1], end: [0, 1])), format: .OIHW,
    name: "conv2")
  out = conv2(out).GELU()
  // [1, width, 1, t] (NCHW from the convolution stem) -> [1, t, width] NHWC for attention.
  out = out.reshaped([1, width, t], format: .NHWC).transposed(1, 2).contiguous()
  let positionEmbedding = Parameter<FloatType>(
    .GPU(0), .HWC(1, t, width), trainable: false, name: "pos_embed")
  out = out + positionEmbedding
  var mappers = [ModelWeightMapper]()
  var groupSums = [Model.IO]()
  var groupSum: Model.IO? = nil
  let groupSize = 8
  for i in 0..<layers {
    if i % groupSize == 0 {
      groupSum = out
    } else {
      groupSum = groupSum.map { $0 + out } ?? out
    }
    if i % groupSize == groupSize - 1, let sum = groupSum {
      groupSums.append((1 / Float(groupSize)) * sum)
    }
    let (mapper, layer) = WhisperEncoderLayer(
      prefix: "encoder.layers.\(i)", k: k, h: heads, b: 1, t: t,
      intermediateSize: intermediateSize, usesFlashAttention: usesFlashAttention)
    out = layer(out)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-5, axis: [2], name: "final_norm")
  out = normFinal(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["encoder.conv1.weight"] = [conv1.weight.name]
    mapping["encoder.conv1.bias"] = [conv1.bias.name]
    mapping["encoder.conv2.weight"] = [conv2.weight.name]
    mapping["encoder.conv2.bias"] = [conv2.bias.name]
    mapping["encoder.embed_positions.weight"] = [positionEmbedding.weight.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["encoder.layer_norm.weight"] = [normFinal.weight.name]
    mapping["encoder.layer_norm.bias"] = [normFinal.bias.name]
    return mapping
  }
  return (mapper, Model([x], groupSums + [out]))
}
