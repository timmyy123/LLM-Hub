package com.llmhub.llmhub.data

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.InputStream

/**
 * What an imported GGUF holds, read from its header: a chat model llama.cpp runs, a vision projector, an LTX-Video
 * transformer (stable-diffusion.cpp), or something else, such as MusicGen or a T5 encoder, that LLM Hub cannot run.
 */
sealed class GgufKind {
    object ChatModel : GgufKind()
    object VisionProjector : GgufKind()
    object LtxVideo : GgufKind()
    data class Unsupported(val architecture: String) : GgufKind()
}

object GgufInspector {
    /** Reads the header and tensor names only; the weights are never read. Null when the stream is not a GGUF. */
    fun inspect(input: InputStream): GgufKind? = runCatching {
        val data = DataInputStream(BufferedInputStream(input, 1 shl 16))
        fun u32(): Long = Integer.reverseBytes(data.readInt()).toLong() and 0xffffffffL
        fun u64(): Long = java.lang.Long.reverseBytes(data.readLong())
        fun skip(bytes: Long) {
            var left = bytes
            while (left > 0) {
                val skipped = data.skip(left)
                if (skipped <= 0) { data.readByte(); left-- } else left -= skipped
            }
        }
        fun string(): String {
            val length = u64()
            require(length in 0..(1L shl 20)) { "bad string length $length" }
            return ByteArray(length.toInt()).also(data::readFully).toString(Charsets.UTF_8)
        }
        fun size(type: Long): Long = when (type) {
            0L, 1L, 7L -> 1
            2L, 3L -> 2
            4L, 5L, 6L -> 4
            10L, 11L, 12L -> 8
            else -> error("unknown GGUF value type $type")
        }
        fun skipValue(type: Long) {
            when (type) {
                8L -> skip(u64())
                9L -> {
                    val itemType = u32()
                    val count = u64()
                    if (itemType == 8L || itemType == 9L) for (i in 0 until count) skipValue(itemType) else skip(count * size(itemType))
                }
                else -> skip(size(type))
            }
        }

        if (u32() != 0x46554747L) return null // "GGUF"
        u32() // version
        val tensorCount = u64()
        val keyCount = u64()
        var architecture = ""
        var hasTokenizer = false
        for (i in 0 until keyCount) {
            val key = string()
            val type = u32()
            when {
                key == "general.architecture" && type == 8L -> architecture = string()
                else -> {
                    if (key == "tokenizer.ggml.model") hasTokenizer = true
                    skipValue(type)
                }
            }
        }
        if (architecture == "clip") return GgufKind.VisionProjector
        val names = HashSet<String>()
        for (i in 0 until minOf(tensorCount, 4096)) {
            names += string().substringBefore('.')
            val dimensions = u32()
            skip(dimensions * 8 + 4 + 8)
        }
        when {
            // Diffusers LTX-Video transformer names (calcuis' files say "pig", city96's say "ltxv").
            "scale_shift_table" in names && "patchify_proj" in names -> GgufKind.LtxVideo
            // Every llama.cpp chat model carries its tokenizer; diffusion, audio and encoder-only files do not.
            hasTokenizer && architecture != "t5encoder" -> GgufKind.ChatModel
            else -> GgufKind.Unsupported(architecture.ifBlank { "unknown" })
        }
    }.getOrNull()
}
