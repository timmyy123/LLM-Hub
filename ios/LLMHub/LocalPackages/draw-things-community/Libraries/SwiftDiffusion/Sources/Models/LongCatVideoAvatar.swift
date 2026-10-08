import DiffusionMappings
import Foundation
import NNC

// LongCat-Video-Avatar-1.5 (Meituan). 13.6B single-stream DiT sharing Wan 2.1's UMT5-XXL text
// encoder, Wan 2.1 VAE and the same 3D RoPE layout. Differences from Wan: SwiGLU FFN, per-frame
// timestep adaLN (Linear(512 -> 6C) per block), an audio cross-attention branch driven by
// Whisper-large-v3 features, and I2V conditioning by replacing the first latent frame (cond frames
// pinned to t=0) with asymmetric self-attention where cond tokens only attend to cond tokens while
// noise tokens attend to everything.

func LongCatVideoAvatarRotaryPositionEmbedding(
  height: Int, width: Int, time: Int, channels: Int, heads: Int = 1, refIndex: Int? = nil,
  numRefLatents: Int = 0
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, time * height * width, heads, channels))
  let dim1 = (channels / 6) * 2
  let dim2 = dim1
  let dim0 = channels - dim1 - dim2
  assert(channels % 16 == 0)
  for t in 0..<time {
    let temporalPosition: Int =
      if let refIndex, numRefLatents > 0 {
        t < numRefLatents ? refIndex : t - numRefLatents
      } else {
        t
      }
    for y in 0..<height {
      for x in 0..<width {
        let i = t * height * width + y * width + x
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta =
              Double(temporalPosition) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, k * 2] = Float(costheta)
            rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim1 / 2) {
            let theta = Double(y) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim2 / 2) {
            let theta = Double(x) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
          }
        }
      }
    }
  }
  return rotTensor
}

private func FeedForwardSwiGLU(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w1")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w3")
  var out = w1(x).swish() .* w3(x)
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out]))
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func TimeEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LongCatAvatarBlock(
  prefix: String, k: Int, h: Int, time: Int, condFrames: Int, hw: Int, textLength: Int,
  audioTokens: Int, intermediateSize: Int, refImgIndex: Int, maskFrameRange: Int,
  usesFlashAttention: FlashAttentionLevel, kvCache: Bool = false
) -> (ModelWeightMapper, Model) {
  let timeCurrent = kvCache ? time - condFrames : time
  let n = timeCurrent * hw
  let timeNoise = kvCache ? timeCurrent : time - condFrames
  let nCond = condFrames * hw
  let nNoise = kvCache ? n : n - nCond
  let c = k * h
  // x flows as [time, hw, c] Float32; per-frame modulation broadcasts on axis 1.
  let x = Input()
  let rot = Input()
  let tSilu = Input()
  let tSiluNoise = Input()
  let cK = Input()
  let cV = Input()
  let aK = Input()
  let aV = Input()
  let cachedKeys = kvCache ? Input() : nil
  let cachedValues = kvCache ? Input() : nil
  let adaLNs = (0..<6).map { Dense(count: c, name: "ada_ln_\($0)") }
  let chunks = adaLNs.map { $0(tSilu).to(.Float32) }
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xMod = (norm1(x) .* (chunks[1] + 1) + chunks[0]).to(.BFloat16).reshaped([1, n, c])
  let xToQueries = Dense(count: c, name: "x_q")
  let xToKeys = Dense(count: c, name: "x_k")
  let xToValues = Dense(count: c, name: "x_v")
  var xQ = xToQueries(xMod).reshaped([1, n, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  var xK = xToKeys(xMod).reshaped([1, n, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  let values = xToValues(xMod).reshaped([1, n, h, k])
  let rotBF16 = rot.to(.BFloat16)
  let xRotBF16: Model.IO
  if kvCache && condFrames > 0 {
    xRotBF16 = rotBF16.reshaped(
      [1, n, 1, k], offset: [0, nCond, 0, 0], strides: [time * hw * k, k, k, 1])
  } else {
    xRotBF16 = rotBF16
  }
  var queries = Functional.cmul(left: xQ, right: xRotBF16)
  var keys = Functional.cmul(left: xK, right: xRotBF16)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    queries = (1 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1 / Float(k).squareRoot().squareRoot()) * keys
  }
  var out: Model.IO
  if usesFlashAttention != .none {
    let scale = usesFlashAttention == .scale1 ? 1 : 1 / Float(k).squareRoot()
    if let cachedKeys = cachedKeys, let cachedValues = cachedValues {
      let rotCond = rotBF16.reshaped(
        [1, nCond, 1, k], offset: [0, 0, 0, 0],
        strides: [time * hw * k, k, k, 1])
      let qNoise = queries
      let kNoise = keys
      var kCached = Functional.cmul(left: cachedKeys.to(.BFloat16), right: rotCond)
      if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
        kCached = (1 / Float(k).squareRoot().squareRoot()) * kCached
      }
      let vCached = cachedValues.to(.BFloat16)
      let kFull = Functional.concat(axis: 1, kCached, kNoise, flags: [.disableOpt])
      let vFull = Functional.concat(axis: 1, vCached, values, flags: [.disableOpt])
      let nRef = hw
      let startNoise = refImgIndex - maskFrameRange - condFrames + 1
      let endNoise = refImgIndex + maskFrameRange - condFrames + 2
      let noiseFrames = timeNoise
      if maskFrameRange > 0 && startNoise >= 0 && endNoise > startNoise
        && endNoise <= noiseFrames
      {
        let startPos = startNoise * hw
        let endPos = endNoise * hw
        var noisePieces = [Model.IO]()
        if startPos > 0 {
          let qNoiseFront = qNoise.reshaped(
            [1, startPos, h, k], offset: [0, 0, 0, 0],
            strides: [nNoise * h * k, h * k, k, 1])
          let frontAttention = ScaledDotProductAttention(
            scale: scale,
            flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
          noisePieces.append(frontAttention(qNoiseFront, kFull, vFull))
        }
        let qNoiseMasked = qNoise.reshaped(
          [1, endPos - startPos, h, k], offset: [0, startPos, 0, 0],
          strides: [nNoise * h * k, h * k, k, 1])
        let kNonRef = kFull.reshaped(
          [1, nCond + nNoise - nRef, h, k], offset: [0, nRef, 0, 0],
          strides: [(nCond + nNoise) * h * k, h * k, k, 1])
        let vNonRef = vFull.reshaped(
          [1, nCond + nNoise - nRef, h, k], offset: [0, nRef, 0, 0],
          strides: [(nCond + nNoise) * h * k, h * k, k, 1])
        let maskedAttention = ScaledDotProductAttention(
          scale: scale,
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        noisePieces.append(maskedAttention(qNoiseMasked, kNonRef, vNonRef))
        if endPos < nNoise {
          let qNoiseBack = qNoise.reshaped(
            [1, nNoise - endPos, h, k], offset: [0, endPos, 0, 0],
            strides: [nNoise * h * k, h * k, k, 1])
          let backAttention = ScaledDotProductAttention(
            scale: scale,
            flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
          noisePieces.append(backAttention(qNoiseBack, kFull, vFull))
        }
        out = noisePieces.count == 1 ? noisePieces[0] : Concat(axis: 1)(noisePieces)
        out = out.reshaped([1, nNoise, c])
      } else {
        let noiseAttention = ScaledDotProductAttention(
          scale: scale,
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        out = noiseAttention(qNoise, kFull, vFull).reshaped([1, nNoise, c])
      }
    } else if condFrames > 1 {
      // AVC: one leading reference latent attends only to itself; continuation cond latents
      // attend only to continuation cond; noisy latents attend full context except the official
      // masked range where reference K/V are removed to avoid repeated motion.
      let nRef = hw
      let nContinuationCond = nCond - nRef
      let qRef = queries.reshaped(
        [1, nRef, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let kRef = keys.reshaped(
        [1, nRef, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let vRef = values.reshaped(
        [1, nRef, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let refAttention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      let outRef = refAttention(qRef, kRef, vRef)
      let qCond = queries.reshaped(
        [1, nContinuationCond, h, k], offset: [0, nRef, 0, 0],
        strides: [n * h * k, h * k, k, 1])
      let kCond = keys.reshaped(
        [1, nContinuationCond, h, k], offset: [0, nRef, 0, 0],
        strides: [n * h * k, h * k, k, 1])
      let vCond = values.reshaped(
        [1, nContinuationCond, h, k], offset: [0, nRef, 0, 0],
        strides: [n * h * k, h * k, k, 1])
      let condAttention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      let outCond = condAttention(qCond, kCond, vCond)
      let qNoise = queries.reshaped(
        [1, nNoise, h, k], offset: [0, nCond, 0, 0], strides: [n * h * k, h * k, k, 1])
      let startNoise = refImgIndex - maskFrameRange - condFrames + 1
      let endNoise = refImgIndex + maskFrameRange - condFrames + 2
      let noiseFrames = time - condFrames
      let outNoise: Model.IO
      if maskFrameRange > 0 && startNoise >= 0 && endNoise > startNoise
        && endNoise <= noiseFrames
      {
        let startPos = startNoise * hw
        let endPos = endNoise * hw
        var noisePieces = [Model.IO]()
        if startPos > 0 {
          let qNoiseFront = qNoise.reshaped(
            [1, startPos, h, k], offset: [0, 0, 0, 0],
            strides: [nNoise * h * k, h * k, k, 1])
          let frontAttention = ScaledDotProductAttention(
            scale: scale,
            flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
          noisePieces.append(frontAttention(qNoiseFront, keys, values))
        }
        let qNoiseMasked = qNoise.reshaped(
          [1, endPos - startPos, h, k], offset: [0, startPos, 0, 0],
          strides: [nNoise * h * k, h * k, k, 1])
        let kNonRef = keys.reshaped(
          [1, n - nRef, h, k], offset: [0, nRef, 0, 0],
          strides: [n * h * k, h * k, k, 1])
        let vNonRef = values.reshaped(
          [1, n - nRef, h, k], offset: [0, nRef, 0, 0],
          strides: [n * h * k, h * k, k, 1])
        let maskedAttention = ScaledDotProductAttention(
          scale: scale,
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        noisePieces.append(maskedAttention(qNoiseMasked, kNonRef, vNonRef))
        if endPos < nNoise {
          let qNoiseBack = qNoise.reshaped(
            [1, nNoise - endPos, h, k], offset: [0, endPos, 0, 0],
            strides: [nNoise * h * k, h * k, k, 1])
          let backAttention = ScaledDotProductAttention(
            scale: scale,
            flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
          noisePieces.append(backAttention(qNoiseBack, keys, values))
        }
        outNoise = noisePieces.count == 1 ? noisePieces[0] : Concat(axis: 1)(noisePieces)
      } else {
        let noiseAttention = ScaledDotProductAttention(
          scale: scale,
          flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        outNoise = noiseAttention(qNoise, keys, values)
      }
      outNoise.add(dependencies: [outRef, outCond])
      out = Concat(axis: 1)([outRef, outCond, outNoise]).reshaped([1, n, c])
    } else if condFrames > 0 {
      // Cond tokens attend only to cond tokens; noise tokens attend to the full sequence.
      let qCond = queries.reshaped(
        [1, nCond, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let kCond = keys.reshaped(
        [1, nCond, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let vCond = values.reshaped(
        [1, nCond, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let condAttention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      let outCond = condAttention(qCond, kCond, vCond)
      let qNoise = queries.reshaped(
        [1, nNoise, h, k], offset: [0, nCond, 0, 0], strides: [n * h * k, h * k, k, 1])
      let noiseAttention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      let outNoise = noiseAttention(qNoise, keys, values)
      outNoise.add(dependencies: [outCond])
      out = Concat(axis: 1)(outCond, outNoise).reshaped([1, n, c])
    } else {
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      out = scaledDotProductAttention(queries, keys, values).reshaped([1, n, c])
    }
  } else {
    keys = keys.transposed(1, 2)
    queries = queries.transposed(1, 2)
    let valuesT = values.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<h {
      let query = queries.reshaped([1, n, k], offset: [i, 0, 0], strides: [n * k, k, 1])
      let key = keys.reshaped([1, n, k], offset: [i, 0, 0], strides: [n * k, k, 1])
      let value = valuesT.reshaped([1, n, k], offset: [i, 0, 0], strides: [n * k, k, 1])
      if let cachedKeys = cachedKeys, let cachedValues = cachedValues {
        let rotCond = rotBF16.reshaped(
          [1, nCond, 1, k], offset: [0, 0, 0, 0],
          strides: [time * hw * k, k, k, 1])
        let qNoise = query
        let kNoise = key
        let kCached =
          ((1 / Float(k).squareRoot().squareRoot())
          * Functional.cmul(left: cachedKeys.to(.BFloat16), right: rotCond)).transposed(1, 2)
          .reshaped([1, nCond, k], offset: [i, 0, 0], strides: [nCond * k, k, 1])
        let vCached = cachedValues.to(.BFloat16).transposed(1, 2).reshaped(
          [1, nCond, k], offset: [i, 0, 0], strides: [nCond * k, k, 1])
        let kFull = Functional.concat(axis: 1, kCached, kNoise, flags: [.disableOpt])
        let vFull = Functional.concat(axis: 1, vCached, value, flags: [.disableOpt])
        let nRef = hw
        let startNoise = refImgIndex - maskFrameRange - condFrames + 1
        let endNoise = refImgIndex + maskFrameRange - condFrames + 2
        let noiseFrames = timeNoise
        let outNoise: Model.IO
        if maskFrameRange > 0 && startNoise >= 0 && endNoise > startNoise
          && endNoise <= noiseFrames
        {
          let startPos = startNoise * hw
          let endPos = endNoise * hw
          var noisePieces = [Model.IO]()
          if startPos > 0 {
            let qNoiseFront = qNoise.reshaped(
              [1, startPos, k], offset: [0, 0, 0], strides: [nNoise * k, k, 1])
            var dotFront = Matmul(transposeB: (1, 2))(qNoiseFront, kFull)
            if let last = outs.last {
              dotFront.add(dependencies: [last])
            }
            dotFront = dotFront.reshaped([startPos, nCond + nNoise]).softmax().reshaped([
              1, startPos, nCond + nNoise,
            ])
            noisePieces.append(dotFront * vFull)
          }
          let qNoiseMasked = qNoise.reshaped(
            [1, endPos - startPos, k], offset: [0, startPos, 0],
            strides: [nNoise * k, k, 1])
          let kNonRef = kFull.reshaped(
            [1, nCond + nNoise - nRef, k], offset: [0, nRef, 0],
            strides: [(nCond + nNoise) * k, k, 1])
          let vNonRef = vFull.reshaped(
            [1, nCond + nNoise - nRef, k], offset: [0, nRef, 0],
            strides: [(nCond + nNoise) * k, k, 1])
          var dotMasked = Matmul(transposeB: (1, 2))(qNoiseMasked, kNonRef)
          dotMasked.add(
            dependencies: noisePieces.last.map { [$0] } ?? (outs.last.map { [$0] } ?? []))
          dotMasked = dotMasked.reshaped([endPos - startPos, nCond + nNoise - nRef]).softmax()
            .reshaped([1, endPos - startPos, nCond + nNoise - nRef])
          noisePieces.append(dotMasked * vNonRef)
          if endPos < nNoise {
            let qNoiseBack = qNoise.reshaped(
              [1, nNoise - endPos, k], offset: [0, endPos, 0],
              strides: [nNoise * k, k, 1])
            var dotBack = Matmul(transposeB: (1, 2))(qNoiseBack, kFull)
            dotBack.add(dependencies: [noisePieces.last!])
            dotBack = dotBack.reshaped([nNoise - endPos, nCond + nNoise]).softmax()
              .reshaped([1, nNoise - endPos, nCond + nNoise])
            noisePieces.append(dotBack * vFull)
          }
          outNoise = noisePieces.count == 1 ? noisePieces[0] : Concat(axis: 1)(noisePieces)
        } else {
          var dotNoise = Matmul(transposeB: (1, 2))(qNoise, kFull)
          if let last = outs.last {
            dotNoise.add(dependencies: [last])
          }
          dotNoise = dotNoise.reshaped([nNoise, nCond + nNoise]).softmax().reshaped([
            1, nNoise, nCond + nNoise,
          ])
          outNoise = dotNoise * vFull
        }
        outs.append(outNoise)
      } else if condFrames > 1 {
        let nRef = hw
        let nContinuationCond = nCond - nRef
        let qRef = query.reshaped([1, nRef, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        let kRef = key.reshaped([1, nRef, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        let vRef = value.reshaped([1, nRef, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        var dotRef = Matmul(transposeB: (1, 2))(qRef, kRef)
        if let last = outs.last {
          dotRef.add(dependencies: [last])
        }
        dotRef = dotRef.reshaped([nRef, nRef]).softmax().reshaped([1, nRef, nRef])
        let outRef = dotRef * vRef
        let qCond = query.reshaped(
          [1, nContinuationCond, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
        let kCond = key.reshaped(
          [1, nContinuationCond, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
        let vCond = value.reshaped(
          [1, nContinuationCond, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
        var dotCond = Matmul(transposeB: (1, 2))(qCond, kCond)
        dotCond.add(dependencies: [outRef])
        dotCond = dotCond.reshaped([nContinuationCond, nContinuationCond]).softmax()
          .reshaped([1, nContinuationCond, nContinuationCond])
        let outCond = dotCond * vCond
        let qNoise = query.reshaped(
          [1, nNoise, k], offset: [0, nCond, 0], strides: [n * k, k, 1])
        let startNoise = refImgIndex - maskFrameRange - condFrames + 1
        let endNoise = refImgIndex + maskFrameRange - condFrames + 2
        let noiseFrames = time - condFrames
        let outNoise: Model.IO
        if maskFrameRange > 0 && startNoise >= 0 && endNoise > startNoise
          && endNoise <= noiseFrames
        {
          let startPos = startNoise * hw
          let endPos = endNoise * hw
          var noisePieces = [Model.IO]()
          if startPos > 0 {
            let qNoiseFront = qNoise.reshaped(
              [1, startPos, k], offset: [0, 0, 0], strides: [nNoise * k, k, 1])
            var dotFront = Matmul(transposeB: (1, 2))(qNoiseFront, key)
            dotFront.add(dependencies: [outCond])
            dotFront = dotFront.reshaped([startPos, n]).softmax().reshaped([1, startPos, n])
            noisePieces.append(dotFront * value)
          }
          let qNoiseMasked = qNoise.reshaped(
            [1, endPos - startPos, k], offset: [0, startPos, 0],
            strides: [nNoise * k, k, 1])
          let kNonRef = key.reshaped(
            [1, n - nRef, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
          let vNonRef = value.reshaped(
            [1, n - nRef, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
          var dotMasked = Matmul(transposeB: (1, 2))(qNoiseMasked, kNonRef)
          dotMasked.add(dependencies: noisePieces.last.map { [$0] } ?? [outCond])
          dotMasked = dotMasked.reshaped([endPos - startPos, n - nRef]).softmax()
            .reshaped([1, endPos - startPos, n - nRef])
          noisePieces.append(dotMasked * vNonRef)
          if endPos < nNoise {
            let qNoiseBack = qNoise.reshaped(
              [1, nNoise - endPos, k], offset: [0, endPos, 0],
              strides: [nNoise * k, k, 1])
            var dotBack = Matmul(transposeB: (1, 2))(qNoiseBack, key)
            dotBack.add(dependencies: [noisePieces.last!])
            dotBack = dotBack.reshaped([nNoise - endPos, n]).softmax()
              .reshaped([1, nNoise - endPos, n])
            noisePieces.append(dotBack * value)
          }
          outNoise = noisePieces.count == 1 ? noisePieces[0] : Concat(axis: 1)(noisePieces)
        } else {
          var dotNoise = Matmul(transposeB: (1, 2))(qNoise, key)
          dotNoise.add(dependencies: [outCond])
          dotNoise = dotNoise.reshaped([nNoise, n]).softmax().reshaped([1, nNoise, n])
          outNoise = dotNoise * value
        }
        outs.append(Concat(axis: 1)([outRef, outCond, outNoise]))
      } else if condFrames > 0 {
        let qCond = query.reshaped([1, nCond, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        let kCond = key.reshaped([1, nCond, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        let vCond = value.reshaped([1, nCond, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        var dotCond = Matmul(transposeB: (1, 2))(qCond, kCond)
        if let last = outs.last {
          dotCond.add(dependencies: [last])
        }
        dotCond = dotCond.reshaped([nCond, nCond]).softmax().reshaped([1, nCond, nCond])
        let outCond = dotCond * vCond
        let qNoise = query.reshaped(
          [1, nNoise, k], offset: [0, nCond, 0], strides: [n * k, k, 1])
        var dotNoise = Matmul(transposeB: (1, 2))(qNoise, key)
        dotNoise.add(dependencies: [outCond])
        dotNoise = dotNoise.reshaped([nNoise, n]).softmax().reshaped([1, nNoise, n])
        let outNoise = dotNoise * value
        outs.append(Concat(axis: 1)(outCond, outNoise))
      } else {
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([n, n]).softmax().reshaped([1, n, n])
        outs.append(dot * value)
      }
    }
    out = Concat(axis: 0)(outs).reshaped([1, h, n, k]).transposed(1, 2).reshaped([1, n, h * k])
  }
  let xUnifyheads = Dense(count: c, name: "x_o")
  out = xUnifyheads(out)
  let xRes = x + (out.to(of: x).reshaped([timeCurrent, hw, c]) .* chunks[2])
  // Text and audio cross-attention only update the noise frames; cond frames pass through.
  let xNoise: Model.IO
  if kvCache {
    xNoise = xRes
  } else if condFrames > 0 {
    xNoise = xRes.reshaped(
      [timeNoise, hw, c], offset: [condFrames, 0, 0], strides: [hw * c, c, 1])
  } else {
    xNoise = xRes
  }
  let norm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "x_norm_3")
  let xNoiseNorm3 = norm3(xNoise).to(.BFloat16).reshaped([1, nNoise, c])
  let xToContextQueries = Dense(count: c, name: "c_q")
  var cQ = xToContextQueries(xNoiseNorm3).reshaped([1, nNoise, h, k])
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  cQ = contextNormQ(cQ)
  let cK32 = cK.to(.BFloat16)
  let cV32 = cV.to(.BFloat16)
  var crossOut: Model.IO
  if usesFlashAttention != .none {
    let crossAttention = ScaledDotProductAttention(
      scale: 1 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    crossOut = crossAttention(cQ, cK32, cV32).reshaped([1, nNoise, c])
  } else {
    let cKT = cK32.transposed(1, 2)
    let cQT = (1 / Float(k).squareRoot() * cQ).transposed(1, 2)
    let cVT = cV32.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(cQT, cKT)
    dot = dot.reshaped([h, nNoise, textLength])
    dot = dot.softmax()
    dot = dot.reshaped([1, h, nNoise, textLength])
    crossOut = (dot * cVT).reshaped([1, h, nNoise, k]).transposed(1, 2).reshaped([1, nNoise, h * k])
  }
  let contextUnifyheads = Dense(count: c, name: "c_o")
  crossOut = contextUnifyheads(crossOut)
  var xNoiseOut = xNoise + crossOut.to(of: xNoise).reshaped([timeNoise, hw, c])
  // Audio cross-attention: each latent frame attends to its own audio context tokens, then the
  // result is layer-normed, per-frame modulated and gated before joining the residual stream.
  let audioAdaLNs = (0..<3).map { Dense(count: c, name: "a_ada_ln_\($0)") }
  let audioChunks = audioAdaLNs.map { $0(tSiluNoise).to(.Float32) }
  let audioPreNorm = LayerNorm(epsilon: 1e-6, axis: [2], name: "a_norm_x")
  let xNoiseAudioNorm = audioPreNorm(xNoiseOut).to(.BFloat16)
  let audioToQueries = Dense(count: c, name: "a_q")
  var aQ = audioToQueries(xNoiseAudioNorm).reshaped([timeNoise, hw, h, k])
  let audioNormQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "a_norm_q")
  aQ = audioNormQ(aQ)
  let aK32 = aK.to(.BFloat16)
  let aV32 = aV.to(.BFloat16)
  var audioOut: Model.IO
  if usesFlashAttention != .none {
    let audioAttention = ScaledDotProductAttention(
      scale: 1 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    audioOut = audioAttention(aQ, aK32, aV32).reshaped([timeNoise, hw, c])
  } else {
    let aKT = aK32.transposed(1, 2)
    let aQT = (1 / Float(k).squareRoot() * aQ).transposed(1, 2)
    let aVT = aV32.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(aQT, aKT)
    dot = dot.reshaped([timeNoise * h, hw, audioTokens])
    dot = dot.softmax()
    dot = dot.reshaped([timeNoise, h, hw, audioTokens])
    audioOut = (dot * aVT).reshaped([timeNoise, h, hw, k]).transposed(1, 2).reshaped([
      timeNoise, hw, h * k,
    ])
  }
  let audioUnifyheads = Dense(count: c, name: "a_o")
  audioOut = audioUnifyheads(audioOut)
  let audioModNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  audioOut =
    audioModNorm(audioOut.to(of: xNoiseOut)) .* (audioChunks[1] + 1) + audioChunks[0]
  xNoiseOut = xNoiseOut + audioOut .* audioChunks[2]
  var xAll: Model.IO
  if !kvCache && condFrames > 0 {
    let xCond = xRes.reshaped(
      [condFrames, hw, c], offset: [0, 0, 0], strides: [hw * c, c, 1])
    xAll = Concat(axis: 0)(xCond, xNoiseOut)
  } else {
    xAll = xNoiseOut
  }
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xMod2 = (norm2(xAll) .* (chunks[4] + 1) + chunks[3]).to(.BFloat16)
  let (w1, w2, w3, ffn) = FeedForwardSwiGLU(
    hiddenSize: c, intermediateSize: intermediateSize, name: "x")
  let ffnOut = ffn(xMod2.reshaped([1, n, c]))
  xAll = xAll + (ffnOut.to(of: xAll).reshaped([timeCurrent, hw, c]) .* chunks[5])
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaLN_modulation.1.weight"] = ModelWeightElement(
      (0..<6).map { adaLNs[$0].weight.name })
    mapping["\(prefix).adaLN_modulation.1.bias"] = ModelWeightElement(
      (0..<6).map { adaLNs[$0].bias.name })
    mapping["\(prefix).attn.qkv.weight"] = ModelWeightElement(
      [xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name], isBF16: true)
    mapping["\(prefix).attn.qkv.bias"] = ModelWeightElement(
      [xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name], isBF16: true)
    mapping["\(prefix).attn.q_norm.weight"] = ModelWeightElement([normQ.weight.name], isBF16: true)
    mapping["\(prefix).attn.k_norm.weight"] = ModelWeightElement([normK.weight.name], isBF16: true)
    mapping["\(prefix).attn.proj.weight"] = ModelWeightElement(
      [xUnifyheads.weight.name], isBF16: true)
    mapping["\(prefix).attn.proj.bias"] = ModelWeightElement(
      [xUnifyheads.bias.name], isBF16: true)
    mapping["\(prefix).pre_crs_attn_norm.weight"] = [norm3.weight.name]
    mapping["\(prefix).pre_crs_attn_norm.bias"] = [norm3.bias.name]
    mapping["\(prefix).cross_attn.q_linear.weight"] = ModelWeightElement(
      [xToContextQueries.weight.name], isBF16: true)
    mapping["\(prefix).cross_attn.q_linear.bias"] = ModelWeightElement(
      [xToContextQueries.bias.name], isBF16: true)
    mapping["\(prefix).cross_attn.q_norm.weight"] = ModelWeightElement(
      [contextNormQ.weight.name], isBF16: true)
    mapping["\(prefix).cross_attn.proj.weight"] = ModelWeightElement(
      [contextUnifyheads.weight.name], isBF16: true)
    mapping["\(prefix).cross_attn.proj.bias"] = ModelWeightElement(
      [contextUnifyheads.bias.name], isBF16: true)
    mapping["\(prefix).audio_adaLN_modulation.1.weight"] = ModelWeightElement(
      (0..<3).map { audioAdaLNs[$0].weight.name })
    mapping["\(prefix).audio_adaLN_modulation.1.bias"] = ModelWeightElement(
      (0..<3).map { audioAdaLNs[$0].bias.name })
    mapping["\(prefix).pre_video_crs_attn_norm.weight"] = [audioPreNorm.weight.name]
    mapping["\(prefix).pre_video_crs_attn_norm.bias"] = [audioPreNorm.bias.name]
    mapping["\(prefix).audio_cross_attn.q_linear.weight"] = ModelWeightElement(
      [audioToQueries.weight.name], isBF16: true)
    mapping["\(prefix).audio_cross_attn.q_linear.bias"] = ModelWeightElement(
      [audioToQueries.bias.name], isBF16: true)
    mapping["\(prefix).audio_cross_attn.q_norm.weight"] = ModelWeightElement(
      [audioNormQ.weight.name], isBF16: true)
    mapping["\(prefix).audio_cross_attn.proj.weight"] = ModelWeightElement(
      [audioUnifyheads.weight.name], isBF16: true)
    mapping["\(prefix).audio_cross_attn.proj.bias"] = ModelWeightElement(
      [audioUnifyheads.bias.name], isBF16: true)
    mapping["\(prefix).ffn.w1.weight"] = ModelWeightElement([w1.weight.name], isBF16: true)
    mapping["\(prefix).ffn.w2.weight"] = ModelWeightElement([w2.weight.name], isBF16: true)
    mapping["\(prefix).ffn.w3.weight"] = ModelWeightElement([w3.weight.name], isBF16: true)
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rot, tSilu, tSiluNoise, cK, cV, aK, aV]
        + (cachedKeys.map { [$0] } ?? []) + (cachedValues.map { [$0] } ?? []), [xAll])
  )
}

/// LongCat-Video-Avatar per-step model. `time` is the number of latent frames, `height` / `width`
/// are latent dimensions (before the 2x2 patchify), `condFrames` is the number of leading latent
/// frames that carry clean conditioning content (1 for ai2v, 0 for at2v).
public func LongCatVideoAvatar(
  time: Int, height: Int, width: Int, channels: Int, layers: Int, intermediateSize: Int,
  textLength: Int, audioTokens: Int, condFrames: Int, refImgIndex: Int = 10,
  // The official demo defaults this to 3 to reduce repeated motion, but the Swift split-attention
  // path currently introduces visible AVC boundary artifacts when the masked range is active.
  maskFrameRange: Int = 0,
  usesFlashAttention: FlashAttentionLevel, kvCache: Bool = false
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let tEmb = Input()
  let h = height / 2
  let w = width / 2
  let hw = h * w
  let k = 128
  let heads = channels / 128
  let timeCurrent = kvCache ? time - condFrames : time
  let timeNoise = kvCache ? timeCurrent : time - condFrames
  let imgIn = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = imgIn(x).reshaped([time, hw, channels]).to(.Float32)
  if kvCache {
    out = out.reshaped(
      [timeCurrent, hw, channels], offset: [condFrames, 0, 0],
      strides: [hw * channels, channels, 1]
    ).contiguous()
  }
  let tSiluAll = tEmb.reshaped([time, 1, 512]).swish()
  let tSilu: Model.IO
  if kvCache {
    tSilu = tSiluAll.reshaped(
      [timeCurrent, 1, 512], offset: [condFrames, 0, 0], strides: [512, 512, 1]
    ).contiguous()
  } else {
    tSilu = tSiluAll
  }
  let tSiluNoise: Model.IO
  if kvCache {
    tSiluNoise = tSilu
  } else if condFrames > 0 {
    tSiluNoise = tSilu.reshaped(
      [timeNoise, 1, 512], offset: [condFrames, 0, 0], strides: [512, 512, 1])
  } else {
    tSiluNoise = tSilu
  }
  var mappers = [ModelWeightMapper]()
  var contextIn = [Input]()
  let cachedAttentionKVs = kvCache ? (0..<layers).map { _ in (Input(), Input()) } : []
  for i in 0..<layers {
    let (mapper, block) = LongCatAvatarBlock(
      prefix: "blocks.\(i)", k: k, h: heads, time: time, condFrames: condFrames, hw: hw,
      textLength: textLength, audioTokens: audioTokens, intermediateSize: intermediateSize,
      refImgIndex: refImgIndex, maskFrameRange: maskFrameRange,
      usesFlashAttention: usesFlashAttention, kvCache: kvCache)
    let cK = Input()
    let cV = Input()
    let aK = Input()
    let aV = Input()
    contextIn.append(contentsOf: [cK, cV, aK, aV])
    var blockInputs: [Model.IO] = [out, rot, tSilu, tSiluNoise, cK, cV, aK, aV]
    if kvCache {
      let cachedAttentionKV = cachedAttentionKVs[i]
      contextIn.append(contentsOf: [cachedAttentionKV.0, cachedAttentionKV.1])
      blockInputs.append(cachedAttentionKV.0)
      blockInputs.append(cachedAttentionKV.1)
    }
    out = block(blockInputs)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let finalShift = Dense(count: channels, name: "final_ada_ln_0")
  let finalScale = Dense(count: channels, name: "final_ada_ln_1")
  let shift = finalShift(tSilu).to(.Float32)
  let scale = finalScale(tSilu).to(.Float32)
  out = (normFinal(out) .* (scale + 1) + shift)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  // LongCat predicts the negated flow velocity (its pipeline negates the DiT output before the
  // scheduler step); negate here so downstream samplers see the standard flow-matching convention.
  // The runtime contract expects FloatType output; the interior stays Float32 for range.
  out = ((-1) * projOut(out)).to(.Float16).reshaped([timeCurrent, h, w, 2, 2, 16]).permuted(
    0, 1, 3, 2, 4, 5
  )
  .contiguous()
  .reshaped([timeCurrent, h * 2, w * 2, 16])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["x_embedder.proj.weight"] = [imgIn.weight.name]
    mapping["x_embedder.proj.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.adaLN_modulation.1.weight"] = ModelWeightElement([
      finalShift.weight.name, finalScale.weight.name,
    ])
    mapping["final_layer.adaLN_modulation.1.bias"] = ModelWeightElement([
      finalShift.bias.name, finalScale.bias.name,
    ])
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot, tEmb] + contextIn, [out]))
}

private func LongCatAvatarBlockCleanKVFixed(
  prefix: String, k: Int, h: Int, condFrames: Int, hw: Int, intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let n = condFrames * hw
  let c = k * h
  let x = Input()
  let rot = Input()
  let tSilu = Input()
  let adaLNs = (0..<6).map { Dense(count: c, name: "ada_ln_\($0)") }
  let chunks = adaLNs.map { $0(tSilu).to(.Float32) }
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xMod = (norm1(x) .* (chunks[1] + 1) + chunks[0]).to(.BFloat16).reshaped([1, n, c])
  let xToQueries = Dense(count: c, name: "x_q")
  let xToKeys = Dense(count: c, name: "x_k")
  let xToValues = Dense(count: c, name: "x_v")
  var xQ = xToQueries(xMod).reshaped([1, n, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  var xK = xToKeys(xMod).reshaped([1, n, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  let values = xToValues(xMod).reshaped([1, n, h, k])
  let rotBF16 = rot.to(.BFloat16)
  var queries = Functional.cmul(left: xQ, right: rotBF16)
  var keys = Functional.cmul(left: xK, right: rotBF16)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    queries = (1 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1 / Float(k).squareRoot().squareRoot()) * keys
  }
  let attnOut: Model.IO
  if usesFlashAttention != .none {
    let scale = usesFlashAttention == .scale1 ? 1 : 1 / Float(k).squareRoot()
    if condFrames > 1 {
      let nRef = hw
      let nContinuationCond = n - nRef
      let qRef = queries.reshaped(
        [1, nRef, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let kRef = keys.reshaped(
        [1, nRef, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let vRef = values.reshaped(
        [1, nRef, h, k], offset: [0, 0, 0, 0], strides: [n * h * k, h * k, k, 1])
      let refAttention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      let outRef = refAttention(qRef, kRef, vRef)
      let qCond = queries.reshaped(
        [1, nContinuationCond, h, k], offset: [0, nRef, 0, 0],
        strides: [n * h * k, h * k, k, 1])
      let kCond = keys.reshaped(
        [1, nContinuationCond, h, k], offset: [0, nRef, 0, 0],
        strides: [n * h * k, h * k, k, 1])
      let vCond = values.reshaped(
        [1, nContinuationCond, h, k], offset: [0, nRef, 0, 0],
        strides: [n * h * k, h * k, k, 1])
      let condAttention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      let outCond = condAttention(qCond, kCond, vCond)
      outCond.add(dependencies: [outRef])
      attnOut = Concat(axis: 1)([outRef, outCond]).reshaped([1, n, c])
    } else {
      let attention = ScaledDotProductAttention(
        scale: scale,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      attnOut = attention(queries, keys, values).reshaped([1, n, c])
    }
  } else {
    keys = keys.transposed(1, 2)
    queries = queries.transposed(1, 2)
    let valuesT = values.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<h {
      let query = queries.reshaped([1, n, k], offset: [i, 0, 0], strides: [n * k, k, 1])
      let key = keys.reshaped([1, n, k], offset: [i, 0, 0], strides: [n * k, k, 1])
      let value = valuesT.reshaped([1, n, k], offset: [i, 0, 0], strides: [n * k, k, 1])
      if condFrames > 1 {
        let nRef = hw
        let nContinuationCond = n - nRef
        let qRef = query.reshaped([1, nRef, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        let kRef = key.reshaped([1, nRef, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        let vRef = value.reshaped([1, nRef, k], offset: [0, 0, 0], strides: [n * k, k, 1])
        var dotRef = Matmul(transposeB: (1, 2))(qRef, kRef)
        if let last = outs.last {
          dotRef.add(dependencies: [last])
        }
        dotRef = dotRef.reshaped([nRef, nRef]).softmax().reshaped([1, nRef, nRef])
        let outRef = dotRef * vRef
        let qCond = query.reshaped(
          [1, nContinuationCond, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
        let kCond = key.reshaped(
          [1, nContinuationCond, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
        let vCond = value.reshaped(
          [1, nContinuationCond, k], offset: [0, nRef, 0], strides: [n * k, k, 1])
        var dotCond = Matmul(transposeB: (1, 2))(qCond, kCond)
        dotCond.add(dependencies: [outRef])
        dotCond = dotCond.reshaped([nContinuationCond, nContinuationCond]).softmax()
          .reshaped([1, nContinuationCond, nContinuationCond])
        outs.append(Concat(axis: 1)(outRef, dotCond * vCond))
      } else {
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([n, n]).softmax().reshaped([1, n, n])
        outs.append(dot * value)
      }
    }
    attnOut = Concat(axis: 0)(outs).reshaped([1, h, n, k]).transposed(1, 2).reshaped([1, n, c])
  }
  let xUnifyheads = Dense(count: c, name: "x_o")
  var out = x + (xUnifyheads(attnOut).to(of: x).reshaped([condFrames, hw, c]) .* chunks[2])
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xMod2 = (norm2(out) .* (chunks[4] + 1) + chunks[3]).to(.BFloat16)
  let (w1, w2, w3, ffn) = FeedForwardSwiGLU(
    hiddenSize: c, intermediateSize: intermediateSize, name: "x")
  let ffnOut = ffn(xMod2.reshaped([1, n, c]))
  out = out + (ffnOut.to(of: out).reshaped([condFrames, hw, c]) .* chunks[5])
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaLN_modulation.1.weight"] = ModelWeightElement(
      (0..<6).map { adaLNs[$0].weight.name })
    mapping["\(prefix).adaLN_modulation.1.bias"] = ModelWeightElement(
      (0..<6).map { adaLNs[$0].bias.name })
    mapping["\(prefix).attn.qkv.weight"] = ModelWeightElement(
      [xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name], isBF16: true)
    mapping["\(prefix).attn.qkv.bias"] = ModelWeightElement(
      [xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name], isBF16: true)
    mapping["\(prefix).attn.q_norm.weight"] = ModelWeightElement([normQ.weight.name], isBF16: true)
    mapping["\(prefix).attn.k_norm.weight"] = ModelWeightElement([normK.weight.name], isBF16: true)
    mapping["\(prefix).attn.proj.weight"] = ModelWeightElement(
      [xUnifyheads.weight.name], isBF16: true)
    mapping["\(prefix).attn.proj.bias"] = ModelWeightElement(
      [xUnifyheads.bias.name], isBF16: true)
    mapping["\(prefix).ffn.w1.weight"] = ModelWeightElement([w1.weight.name], isBF16: true)
    mapping["\(prefix).ffn.w2.weight"] = ModelWeightElement([w2.weight.name], isBF16: true)
    mapping["\(prefix).ffn.w3.weight"] = ModelWeightElement([w3.weight.name], isBF16: true)
    return mapping
  }
  return (mapper, Model([x, rot, tSilu], [out, xK, values]))
}

private func LongCatAvatarBlockFixed(
  prefix: String, k: Int, h: Int, textLength: Int, audioFrames: Int, audioTokens: Int
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let audioContext = Input()
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var cK = contextToKeys(context).reshaped([1, textLength, h, k])
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  cK = contextNormK(cK)
  let cV = contextToValues(context).reshaped([1, textLength, h, k])
  let audioToKeys = Dense(count: k * h, name: "a_k")
  let audioToValues = Dense(count: k * h, name: "a_v")
  var aK = audioToKeys(audioContext).reshaped([audioFrames, audioTokens, h, k])
  let audioNormK = RMSNorm(epsilon: 1e-6, axis: [3], name: "a_norm_k")
  aK = audioNormK(aK)
  let aV = audioToValues(audioContext).reshaped([audioFrames, audioTokens, h, k])
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).cross_attn.kv_linear.weight"] = ModelWeightElement([
      contextToKeys.weight.name, contextToValues.weight.name,
    ])
    mapping["\(prefix).cross_attn.kv_linear.bias"] = ModelWeightElement([
      contextToKeys.bias.name, contextToValues.bias.name,
    ])
    mapping["\(prefix).cross_attn.k_norm.weight"] = [contextNormK.weight.name]
    mapping["\(prefix).audio_cross_attn.kv_linear.weight"] = ModelWeightElement([
      audioToKeys.weight.name, audioToValues.weight.name,
    ])
    mapping["\(prefix).audio_cross_attn.kv_linear.bias"] = ModelWeightElement([
      audioToKeys.bias.name, audioToValues.bias.name,
    ])
    mapping["\(prefix).audio_cross_attn.k_norm.weight"] = [audioNormK.weight.name]
    return mapping
  }
  return (mapper, Model([context, audioContext], [cK, cV, aK, aV]))
}

/// Per-generation precompute for LongCat-Video-Avatar: caption embedding + per-block text K/V,
/// audio projection (Whisper feature windows -> 32 context tokens per latent frame) + per-block
/// audio K/V, and the per-frame timestep embeddings for every sampling step.
///
/// Inputs:
/// - txt: [1, textLength, 4096] UMT5 embeddings, zero-padded to textLength.
/// - t: [timesteps * time, 256] sinusoidal timestep embeddings (cond frames use t = 0).
/// - audioFirst: [1, 1, audioWindow * audioBlocks * audioChannels] flattened Whisper window
///   of the first video frame.
/// - audioLatter: [1, time - 1, (audioWindow + vaeScale - 1) * audioBlocks * audioChannels]
///   flattened windows for the remaining latent frames.
public func LongCatVideoAvatarFixed(
  timesteps: Int, time: Int, condFrames: Int, height: Int = 0, width: Int = 0,
  channels: Int, layers: Int, textLength: Int, audioWindow: Int, audioBlocks: Int,
  audioChannels: Int, audioTokens: Int, audioIntermediateDim: Int, audioOutputDim: Int,
  kvCache: Bool = false, usesFlashAttention: FlashAttentionLevel = .scale1
) -> (ModelWeightMapper, Model) {
  let txt = Input()
  let t = Input()
  let audioFirst = Input()
  let audioLatter = Input()
  let cleanCondLatents = kvCache ? Input() : nil
  let cleanCondRot = kvCache ? Input() : nil
  let k = 128
  let h = channels / 128
  let timeNoise = time - condFrames
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: channels, name: "c")
  let context = contextEmbedder(txt)
  let (timeInMlp0, timeInMlp2, timeIn) = TimeEmbedder(channels: 512, name: "t")
  let vector = timeIn(t).reshaped([timesteps, time, 512])
  // Audio projection (AudioProjModel): window features -> 32 x 768 context tokens per frame.
  let audioProj1 = Dense(count: audioIntermediateDim, name: "audio_proj_1")
  let audioProj1VF = Dense(count: audioIntermediateDim, name: "audio_proj_1_vf")
  let audioFirstOut = audioProj1(audioFirst).ReLU()
  let audioLatterOut = audioProj1VF(audioLatter).ReLU()
  var audio: Model.IO = Concat(axis: 1)(audioFirstOut, audioLatterOut)
  let audioProj2 = Dense(count: audioIntermediateDim, name: "audio_proj_2")
  audio = audioProj2(audio).ReLU()
  let audioProj3 = Dense(count: audioTokens * audioOutputDim, name: "audio_proj_3")
  let audioFrames = condFrames > 1 ? time - 1 : time
  audio = audioProj3(audio).reshaped([audioFrames, audioTokens, audioOutputDim])
  let audioNorm = LayerNorm(epsilon: 1e-5, axis: [2], name: "audio_proj_norm")
  audio = audioNorm(audio)
  if condFrames > 1 {
    let refAudio = audio.reshaped(
      [1, audioTokens, audioOutputDim], offset: [0, 0, 0],
      strides: [audioTokens * audioOutputDim, audioOutputDim, 1])
    audio = Concat(axis: 0)(refAudio, audio)
  }
  // Cond-frame audio tokens are never attended to; only emit K/V for noise frames.
  if condFrames > 0 {
    audio = audio.reshaped(
      [timeNoise, audioTokens, audioOutputDim], offset: [condFrames, 0, 0],
      strides: [audioTokens * audioOutputDim, audioOutputDim, 1]
    ).contiguous()
  }
  var outs = [Model.IO]()
  outs.append(vector)
  var mappers = [ModelWeightMapper]()
  let xEmbedder: Model?
  var cleanOut: Model.IO?
  let cleanTSilu: Model.IO?
  if kvCache, let cleanCondLatents = cleanCondLatents {
    precondition(condFrames > 0 && height > 0 && width > 0)
    let embedder = Convolution(
      groups: 1, filters: channels, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
    xEmbedder = embedder
    cleanOut = embedder(cleanCondLatents).reshaped([
      condFrames, (height / 2) * (width / 2), channels,
    ]).to(.Float32)
    cleanTSilu = vector.reshaped(
      [condFrames, 1, 512], offset: [0, 0, 0], strides: [512, 512, 1]
    ).swish()
  } else {
    xEmbedder = nil
    cleanOut = nil
    cleanTSilu = nil
  }
  for i in 0..<layers {
    let (mapper, block) = LongCatAvatarBlockFixed(
      prefix: "blocks.\(i)", k: k, h: h, textLength: textLength, audioFrames: timeNoise,
      audioTokens: audioTokens)
    let blockOut = block(context, audio)
    outs.append(blockOut[0])
    outs.append(blockOut[1])
    outs.append(blockOut[2])
    outs.append(blockOut[3])
    mappers.append(mapper)
    if kvCache, let cleanCondRot = cleanCondRot, let cleanTSilu = cleanTSilu {
      let (kvMapper, kvBlock) = LongCatAvatarBlockCleanKVFixed(
        prefix: "blocks.\(i)", k: k, h: h, condFrames: condFrames,
        hw: (height / 2) * (width / 2), intermediateSize: 11_008,
        usesFlashAttention: usesFlashAttention)
      let kvOut = kvBlock([cleanOut!, cleanCondRot, cleanTSilu])
      cleanOut = kvOut[0]
      outs.append(kvOut[1])
      outs.append(kvOut[2])
      mappers.append(kvMapper)
    }
  }
  // Keep the clean-cache subgraph connected for ccv; the caller drops this helper output.
  if kvCache, let cleanOut = cleanOut {
    outs.append(cleanOut)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    if let xEmbedder = xEmbedder {
      mapping["x_embedder.proj.weight"] = [xEmbedder.weight.name]
      mapping["x_embedder.proj.bias"] = [xEmbedder.bias.name]
    }
    mapping["y_embedder.y_proj.0.weight"] = [cLinear1.weight.name]
    mapping["y_embedder.y_proj.0.bias"] = [cLinear1.bias.name]
    mapping["y_embedder.y_proj.2.weight"] = [cLinear2.weight.name]
    mapping["y_embedder.y_proj.2.bias"] = [cLinear2.bias.name]
    mapping["t_embedder.mlp.0.weight"] = [timeInMlp0.weight.name]
    mapping["t_embedder.mlp.0.bias"] = [timeInMlp0.bias.name]
    mapping["t_embedder.mlp.2.weight"] = [timeInMlp2.weight.name]
    mapping["t_embedder.mlp.2.bias"] = [timeInMlp2.bias.name]
    mapping["audio_proj.proj1.weight"] = [audioProj1.weight.name]
    mapping["audio_proj.proj1.bias"] = [audioProj1.bias.name]
    mapping["audio_proj.proj1_vf.weight"] = [audioProj1VF.weight.name]
    mapping["audio_proj.proj1_vf.bias"] = [audioProj1VF.bias.name]
    mapping["audio_proj.proj2.weight"] = [audioProj2.weight.name]
    mapping["audio_proj.proj2.bias"] = [audioProj2.bias.name]
    mapping["audio_proj.proj3.weight"] = [audioProj3.weight.name]
    mapping["audio_proj.proj3.bias"] = [audioProj3.bias.name]
    mapping["audio_proj.norm.weight"] = [audioNorm.weight.name]
    mapping["audio_proj.norm.bias"] = [audioNorm.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [txt, t, audioFirst, audioLatter] + (cleanCondLatents.map { [$0] } ?? [])
        + (cleanCondRot.map { [$0] } ?? []), outs)
  )
}

public func LongCatVideoAvatarFixedOutputShapes(
  timesteps: Int, time: Int, condFrames: Int, channels: Int, layers: Int, textLength: Int,
  audioTokens: Int, height: Int = 0, width: Int = 0, kvCache: Bool = false
) -> [TensorShape] {
  let k = 128
  let h = channels / 128
  let timeNoise = time - condFrames
  var outs = [TensorShape([timesteps, time, 512])]
  for _ in 0..<layers {
    outs.append(contentsOf: [
      TensorShape([1, textLength, h, k]), TensorShape([1, textLength, h, k]),
      TensorShape([timeNoise, audioTokens, h, k]), TensorShape([timeNoise, audioTokens, h, k]),
    ])
    if kvCache {
      outs.append(contentsOf: [
        TensorShape([1, condFrames * (height / 2) * (width / 2), h, k]),
        TensorShape([1, condFrames * (height / 2) * (width / 2), h, k]),
      ])
    }
  }
  return outs
}
