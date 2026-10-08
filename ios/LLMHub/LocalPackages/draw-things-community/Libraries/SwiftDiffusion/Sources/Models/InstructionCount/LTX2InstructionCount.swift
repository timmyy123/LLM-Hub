import Foundation

private func LTX2SelfAttentionInstructionCount(
  batchSize: Int, sequenceLength: Int, hiddenSize: Int, heads: Int, useGatedAttention: Bool
) -> Int {
  let headDimension = hiddenSize / heads
  let rows = batchSize * sequenceLength
  var total = 0
  total += 4 * DenseInstructionCount(rows: rows, input: hiddenSize, output: hiddenSize)  // q,k,v,o
  if useGatedAttention {
    total += DenseInstructionCount(rows: rows, input: hiddenSize, output: heads)  // gate
  }
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension,
    sequenceDimensionA: sequenceLength,
    sequenceDimensionB: sequenceLength)
  return total
}

private func LTX2CrossAttentionInstructionCount(
  batchSize: Int, queryLength: Int, keyLength: Int, queryInput: Int, keyValueInput: Int,
  attnHidden: Int, outputHidden: Int, heads: Int, KV: Bool, useGatedAttention: Bool
) -> Int {
  let headDimension = attnHidden / heads
  var total = 0
  total += DenseInstructionCount(
    rows: batchSize * queryLength, input: queryInput, output: attnHidden)  // q
  if !KV {
    total += DenseInstructionCount(
      rows: batchSize * keyLength, input: keyValueInput, output: attnHidden)  // k
    total += DenseInstructionCount(
      rows: batchSize * keyLength, input: keyValueInput, output: attnHidden)  // v
  }
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension,
    sequenceDimensionA: queryLength,
    sequenceDimensionB: keyLength)
  total += DenseInstructionCount(
    rows: batchSize * queryLength, input: attnHidden, output: outputHidden)  // o
  if useGatedAttention {
    total += DenseInstructionCount(rows: batchSize * queryLength, input: queryInput, output: heads)  // gate
  }
  return total
}

private func LTX2FeedForwardInstructionCount(
  batchSize: Int, sequenceLength: Int, hiddenSize: Int, intermediateSize: Int
) -> Int {
  DenseInstructionCount(
    rows: batchSize * sequenceLength, input: hiddenSize, output: intermediateSize)
    + DenseInstructionCount(
      rows: batchSize * sequenceLength, input: intermediateSize, output: hiddenSize)
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Extra input-channel params are needed because `LTX2(...)` builder does not expose them.
public func LTX2InstructionCount(
  time: Int, h: Int, w: Int, textLength: Int, audioFrames: Int, channels: (Int, Int), layers: Int,
  tokenModulation: Bool, KV: Bool, useGatedAttention: Bool, textCrossAttentionAdaLN: Bool,
  videoInputChannels: Int = 128, audioInputChannels: Int = 128
) -> Int {
  precondition(channels.0 % 32 == 0 && channels.1 % 32 == 0)
  let batchSize = 1
  let videoLength = time * h * w
  let audioLength = audioFrames
  let heads = 32
  var total = 0

  // Patchify/embed inputs.
  total += ConvolutionInstructionCount(
    batchSize: batchSize * time, outHeight: h, outWidth: w, outChannels: channels.0,
    kernelHeight: 1,
    kernelWidth: 1, inputChannels: videoInputChannels)
  total += DenseInstructionCount(
    rows: batchSize * audioLength, input: audioInputChannels, output: channels.1)

  for _ in 0..<layers {
    // Video self-attn + text cross-attn.
    total += LTX2SelfAttentionInstructionCount(
      batchSize: batchSize, sequenceLength: videoLength, hiddenSize: channels.0, heads: heads,
      useGatedAttention: useGatedAttention)
    total += LTX2CrossAttentionInstructionCount(
      batchSize: batchSize, queryLength: videoLength, keyLength: textLength, queryInput: channels.0,
      keyValueInput: channels.0, attnHidden: channels.0, outputHidden: channels.0, heads: heads,
      KV: KV, useGatedAttention: useGatedAttention)

    // Audio self-attn + text cross-attn.
    total += LTX2SelfAttentionInstructionCount(
      batchSize: batchSize, sequenceLength: audioLength, hiddenSize: channels.1, heads: heads,
      useGatedAttention: useGatedAttention)
    total += LTX2CrossAttentionInstructionCount(
      batchSize: batchSize, queryLength: audioLength, keyLength: textLength, queryInput: channels.1,
      keyValueInput: channels.1, attnHidden: channels.1, outputHidden: channels.1, heads: heads,
      KV: KV, useGatedAttention: useGatedAttention)

    // Cross-modal attentions.
    total += LTX2CrossAttentionInstructionCount(
      batchSize: batchSize, queryLength: videoLength, keyLength: audioLength,
      queryInput: channels.0,
      keyValueInput: channels.1, attnHidden: channels.1, outputHidden: channels.0, heads: heads,
      KV: false, useGatedAttention: useGatedAttention)
    total += LTX2CrossAttentionInstructionCount(
      batchSize: batchSize, queryLength: audioLength, keyLength: videoLength,
      queryInput: channels.1,
      keyValueInput: channels.0, attnHidden: channels.1, outputHidden: channels.1, heads: heads,
      KV: false, useGatedAttention: useGatedAttention)

    // FFNs.
    total += LTX2FeedForwardInstructionCount(
      batchSize: batchSize, sequenceLength: videoLength, hiddenSize: channels.0,
      intermediateSize: channels.0 * 4)
    total += LTX2FeedForwardInstructionCount(
      batchSize: batchSize, sequenceLength: audioLength, hiddenSize: channels.1,
      intermediateSize: channels.1 * 4)
  }

  // Final projections.
  total += DenseInstructionCount(rows: batchSize * videoLength, input: channels.0, output: 128)
  total += DenseInstructionCount(rows: batchSize * audioLength, input: channels.1, output: 128)

  _ = tokenModulation  // Modifies elementwise/reshape flow only.
  _ = textCrossAttentionAdaLN
  return total
}

private func LTX2AdaLNSingleInstructionCount(
  timesteps: Int, channels: Int, count: Int
) -> Int {
  var total = 0
  total += DenseInstructionCount(rows: timesteps, input: 256, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  total += count * DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  return total
}

// Counts only Dense (GEMM) and SDPA-equivalent attention instructions for `LTX2Fixed(...)`.
// No convolutions are present in the fixed builder.
public func LTX2FixedInstructionCount(
  time: Int, textLength: Int, audioFrames: Int, timesteps: Int, channels: (Int, Int), layers: Int,
  contextProjection: Bool, textCrossAttentionAdaLN: Bool, KV: Bool,
  textInputChannels: (Int, Int)? = nil
) -> Int {
  precondition(channels.0 % 32 == 0 && channels.1 % 32 == 0)
  let textInput = textInputChannels ?? channels
  var total = 0

  // caption projections (video + audio).
  if contextProjection {
    total += DenseInstructionCount(rows: textLength, input: textInput.0, output: channels.0)
    total += DenseInstructionCount(rows: textLength, input: channels.0, output: channels.0)
    total += DenseInstructionCount(rows: textLength, input: textInput.1, output: channels.1)
    total += DenseInstructionCount(rows: textLength, input: channels.1, output: channels.1)
  }

  total += LTX2AdaLNSingleInstructionCount(
    timesteps: timesteps, channels: channels.0, count: textCrossAttentionAdaLN ? 9 : 6)
  total += LTX2AdaLNSingleInstructionCount(
    timesteps: timesteps, channels: channels.1, count: textCrossAttentionAdaLN ? 9 : 6)
  if textCrossAttentionAdaLN {
    total += LTX2AdaLNSingleInstructionCount(timesteps: timesteps, channels: channels.0, count: 2)
    total += LTX2AdaLNSingleInstructionCount(timesteps: timesteps, channels: channels.1, count: 2)
  }
  total += LTX2AdaLNSingleInstructionCount(timesteps: timesteps, channels: channels.0, count: 4)
  total += LTX2AdaLNSingleInstructionCount(timesteps: timesteps, channels: channels.1, count: 4)
  total += LTX2AdaLNSingleInstructionCount(timesteps: timesteps, channels: channels.0, count: 1)
  total += LTX2AdaLNSingleInstructionCount(timesteps: timesteps, channels: channels.1, count: 1)

  // Per-layer fixed K/V precomputation for text cross-attn (video and audio streams).
  if KV {
    for _ in 0..<layers {
      total += 2 * DenseInstructionCount(rows: textLength, input: channels.0, output: channels.0)  // cv k,v
      total += 2 * DenseInstructionCount(rows: textLength, input: channels.1, output: channels.1)  // ca k,v
    }
  }

  _ = time
  _ = audioFrames
  return total
}
