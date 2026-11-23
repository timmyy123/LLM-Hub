package com.llmhub.llmhub.utils

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

class Float32WavStreamTest {

    /** The whole-track resampler AudioConversionUtils used before 4.6.9; streaming must match it. */
    private fun resampleWhole(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (input.isEmpty() || fromRate == toRate) return input
        val outLen = ((input.size.toDouble() * toRate) / fromRate).toInt().coerceAtLeast(1)
        val out = FloatArray(outLen)
        val ratio = fromRate.toDouble() / toRate
        for (i in out.indices) {
            val srcPos = i * ratio
            val i0 = srcPos.toInt().coerceIn(0, input.lastIndex)
            val i1 = (i0 + 1).coerceAtMost(input.lastIndex)
            val frac = srcPos - i0
            out[i] = ((1.0 - frac) * input[i0] + frac * input[i1]).toFloat()
        }
        return out
    }

    private fun resampleStreaming(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        val out = ArrayList<Float>()
        val r = StreamingLinearResampler(fromRate, toRate) { out.add(it) }
        input.forEach(r::push)
        r.finish()
        return out.toFloatArray()
    }

    @Test
    fun streamingResamplerMatchesWholeTrackResampler() {
        val rnd = Random(7)
        for (from in listOf(8_000, 11_025, 16_000, 22_050, 44_100, 48_000, 96_000)) {
            for (len in listOf(1, 2, 3, 159, 4_410, 48_001)) {
                val input = FloatArray(len) { rnd.nextFloat() * 2f - 1f }
                val expected = resampleWhole(input, from, 16_000)
                val actual = resampleStreaming(input, from, 16_000)
                assertEquals("length from=$from len=$len", expected.size, actual.size)
                for (i in expected.indices) {
                    assertEquals("from=$from len=$len i=$i", expected[i], actual[i], 1e-5f)
                }
            }
        }
    }

    @Test
    fun builderWritesValidMono16kFloatWav() {
        val samples = FloatArray(16_000) { i -> kotlin.math.sin(i / 10.0).toFloat() }
        val b = Float32WavBuilder(16_000, samples.size.toLong())
        samples.forEach(b::add)
        val wav = b.build()

        assertTrue(Float32WavBuilder.isMono16kFloat32Header(wav.copyOf(44), 16_000))
        val bb = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        val dataBytes = bb.getInt(40)
        assertEquals(wav.size - 44, dataBytes)
        assertEquals(36 + dataBytes, bb.getInt(4))
        for (i in samples.indices) assertEquals(samples[i], bb.getFloat(44 + i * 4), 0f)
        // Spare room from the duration estimate is at most 0.25 s of silence.
        val tail = dataBytes / 4 - samples.size
        assertTrue("tail=$tail", tail in 0..4_000)
        for (i in samples.size until dataBytes / 4) assertEquals(0f, bb.getFloat(44 + i * 4), 0f)
    }

    @Test
    fun builderGrowsWhenDurationIsUnknownOrShort() {
        for (estimate in listOf(-1L, 100L)) {
            val b = Float32WavBuilder(16_000, estimate)
            repeat(700_000) { b.add(0.5f) }
            val wav = b.build()
            val bb = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
            assertEquals(700_000 * 4, bb.getInt(40))
            assertEquals(44 + 700_000 * 4, wav.size)
            assertEquals(0.5f, bb.getFloat(44 + 699_999 * 4), 0f)
        }
    }

    @Test
    fun builderTrimsWhenDurationEstimateIsFarTooLong() {
        val b = Float32WavBuilder(16_000, 16_000L * 60)
        repeat(1_000) { b.add(0.1f) }
        assertEquals(44 + 1_000 * 4, b.build().size)
    }

    @Test
    fun headerCheckRejectsOtherFormats() {
        val pcm16 = ByteArray(44).also {
            val bb = ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN)
            bb.put("RIFF".toByteArray()).putInt(36).put("WAVE".toByteArray()).put("fmt ".toByteArray())
            bb.putInt(16).putShort(1).putShort(1).putInt(16_000).putInt(32_000).putShort(2).putShort(16)
        }
        assertFalse(Float32WavBuilder.isMono16kFloat32Header(pcm16, 16_000))
        assertFalse(Float32WavBuilder.isMono16kFloat32Header(ByteArray(10), 16_000))
        assertFalse(Float32WavBuilder.isMono16kFloat32Header("ID3".toByteArray() + ByteArray(41), 16_000))
    }
}
