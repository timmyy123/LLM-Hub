package com.llmhub.llmhub.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GgufInspectorTest {
    /** A GGUF header with string metadata, an int array, and tensor infos (no tensor data). */
    private fun gguf(metadata: Map<String, String>, tensors: List<String>, tokenTypes: Int = 0): ByteArray {
        val out = ByteArrayOutputStream()
        fun u32(v: Long) = out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v.toInt()).array())
        fun u64(v: Long) = out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array())
        fun str(s: String) { val b = s.toByteArray(); u64(b.size.toLong()); out.write(b) }
        out.write("GGUF".toByteArray()); u32(3); u64(tensors.size.toLong())
        u64(metadata.size.toLong() + if (tokenTypes > 0) 1 else 0)
        metadata.forEach { (k, v) -> str(k); u32(8); str(v) }
        if (tokenTypes > 0) { str("tokenizer.ggml.token_type"); u32(9); u32(5); u64(tokenTypes.toLong()); repeat(tokenTypes) { u32(1) } }
        tensors.forEach { str(it); u32(2); u64(4); u64(4); u32(0); u64(0) }
        return out.toByteArray()
    }

    private fun inspect(bytes: ByteArray) = GgufInspector.inspect(bytes.inputStream())

    @Test fun chatModelHasTokenizer() = assertEquals(GgufKind.ChatModel,
        inspect(gguf(mapOf("general.architecture" to "llama", "tokenizer.ggml.model" to "gpt2"), listOf("blk.0.attn_q.weight"), tokenTypes = 1000)))

    @Test fun ltxVideoTransformer() = assertEquals(GgufKind.LtxVideo,
        inspect(gguf(mapOf("general.architecture" to "pig"), listOf("adaln_single.linear.weight", "patchify_proj.weight", "scale_shift_table"))))

    @Test fun projector() = assertEquals(GgufKind.VisionProjector,
        inspect(gguf(mapOf("general.architecture" to "clip"), listOf("v.patch_embd.weight"))))

    @Test fun musicGenIsUnsupported() = assertEquals(GgufKind.Unsupported("musicgen"),
        inspect(gguf(mapOf("general.architecture" to "musicgen"), listOf("decoder.layers.0.self_attn.q_proj.weight"))))

    @Test fun t5EncoderIsUnsupported() = assertEquals(GgufKind.Unsupported("t5encoder"),
        inspect(gguf(mapOf("general.architecture" to "t5encoder", "tokenizer.ggml.model" to "t5"), listOf("enc.blk.0.attn_q.weight"))))

    @Test fun notGguf() = assertNull(inspect("PK\u0003\u0004 not a gguf".toByteArray()))
}
