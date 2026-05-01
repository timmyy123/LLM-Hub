package com.llmhub.llmhub.mimobot.speech.kokoro

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Loads a Kokoro voice file. The onnx-community export distributes each voice
 * as `voices/<id>.bin` — a flat float32 tensor of shape (N, 1, 256) where N is
 * usually 511. The first axis is indexed by token-sequence length: for a
 * sequence of length L tokens, the inference uses `voice[L]` (clamped to the
 * last row) as the 256-d style embedding.
 *
 * File layout: little-endian float32, no header.
 */
class VoicePack private constructor(
    val rows: Int,
    val styleDim: Int,
    private val data: FloatArray,
) {

    /** Get the style vector for a token sequence of [numTokens] tokens. */
    fun styleFor(numTokens: Int): FloatArray {
        val row = numTokens.coerceIn(0, rows - 1)
        val out = FloatArray(styleDim)
        System.arraycopy(data, row * styleDim, out, 0, styleDim)
        return out
    }

    companion object {
        const val DEFAULT_STYLE_DIM = 256

        fun load(file: File): VoicePack {
            val bytes = file.readBytes()
            val floats = bytes.size / 4
            require(floats > 0 && floats % DEFAULT_STYLE_DIM == 0) {
                "voice file size ${bytes.size} not a multiple of $DEFAULT_STYLE_DIM floats"
            }
            val rows = floats / DEFAULT_STYLE_DIM
            val arr = FloatArray(floats)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer().get(arr)
            return VoicePack(rows, DEFAULT_STYLE_DIM, arr)
        }
    }
}
