package com.llmhub.llmhub.data

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GgufNpuExpertTypesTest {
    private class Gguf {
        private val out = ByteArrayOutputStream()
        fun u32(v: Long) = apply { out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v.toInt()).array()) }
        fun u64(v: Long) = apply { out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array()) }
        fun str(s: String) = apply { val b = s.toByteArray(); u64(b.size.toLong()); out.write(b) }
        fun raw(s: String) = apply { out.write(s.toByteArray()) }
        fun bytes() = out.toByteArray()
    }

    /** A header with an architecture string, a string array (like a tokenizer) and the given tensors. */
    private fun gguf(vararg tensors: Pair<String, Int>): File {
        val g = Gguf().apply {
            raw("GGUF")
            u32(3); u64(tensors.size.toLong()); u64(3)
            str("general.architecture"); u32(8); str("gemma4")
            str("gemma4.expert_count"); u32(4); u32(128)
            str("tokenizer.ggml.tokens"); u32(9); u32(8); u64(3); str("a"); str("bb"); str("ccc")
            tensors.forEach { (name, type) ->
                str(name); u32(3); u64(2816); u64(704); u64(128); u32(type.toLong()); u64(0)
            }
        }
        return File.createTempFile("npu", ".gguf").apply { writeBytes(g.bytes()); deleteOnExit() }
    }

    @Test
    fun gemmaUnslothIq2XsExpertsCannotStreamOnNpu() {
        val file = gguf("blk.0.ffn_gate_up_exps.weight" to 17, "blk.0.ffn_down_exps.weight" to 20, "blk.0.attn_q.weight" to 12)
        assertTrue(GgufLayerLimits.npuCannotStreamExperts(file))
    }

    @Test
    fun q4_0ExpertsStreamOnNpu() {
        val file = gguf("blk.0.ffn_gate_up_exps.weight" to 2, "blk.0.ffn_down_exps.weight" to 2, "blk.0.ffn_down_exps.scale" to 0)
        assertFalse(GgufLayerLimits.npuCannotStreamExperts(file))
    }

    @Test
    fun unsupportedDenseTypeDoesNotHideNpu() {
        // Dense matrices the NPU cannot take are converted at load; only the experts decide.
        val file = gguf("blk.0.ffn_gate_exps.weight" to 2, "blk.0.attn_q.weight" to 23)
        assertFalse(GgufLayerLimits.npuCannotStreamExperts(file))
    }

    @Test
    fun denseModelDoesNotHideNpu() {
        assertFalse(GgufLayerLimits.npuCannotStreamExperts(gguf("blk.0.attn_q.weight" to 17)))
    }
}
