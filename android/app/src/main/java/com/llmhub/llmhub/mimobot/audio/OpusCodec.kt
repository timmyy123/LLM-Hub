package com.llmhub.llmhub.mimobot.audio

/**
 * Opus encode/decode for 16 kHz mono, 20 ms frames.
 *
 * We want a JVM Opus binding here. Two reasonable choices:
 *   - concentus (pure-Java, easy to ship, ~3x slower than native — fine at 24 kbps).
 *       implementation("org.concentus:Concentus:0.3.1") or similar artifact.
 *   - an NDK build of libopus exposed via JNI (lower CPU, more work).
 *
 * We'll start with Concentus and switch later if phones struggle. Nothing above
 * this file depends on the choice.
 *
 * TODO(opus):
 *   1. Add the Concentus dependency to android/app/build.gradle.kts:
 *        implementation("org.concentus:Concentus:0.3.1")
 *   2. Replace the bodies below with OpusEncoder / OpusDecoder calls.
 *   3. Reuse Encoder/Decoder instances — they're stateful (FEC, predictor state).
 */
class OpusEncoder(
    val sampleRateHz: Int = 16_000,
    val channels: Int = 1,
    val bitrateBps: Int = 24_000,
    val frameSamples: Int = 320,
) {
    fun encode(pcm: ShortArray): ByteArray {
        require(pcm.size == frameSamples) { "expected $frameSamples samples, got ${pcm.size}" }
        TODO("Concentus OpusEncoder.encode(pcm, 0, frameSamples, outBuf, 0, outBuf.size)")
    }

    fun close() { /* no-op until we wire the encoder */ }
}

class OpusDecoder(
    val sampleRateHz: Int = 16_000,
    val channels: Int = 1,
    val frameSamples: Int = 320,
) {
    fun decode(opusFrame: ByteArray): ShortArray {
        TODO("Concentus OpusDecoder.decode(opusFrame, 0, opusFrame.size, out, 0, frameSamples, false)")
    }

    /** Produce PCM for a lost frame using Opus PLC. */
    fun decodePlc(): ShortArray {
        TODO("Concentus OpusDecoder.decode(null, ..., decodeFec=false)")
    }

    fun close() { /* no-op */ }
}
