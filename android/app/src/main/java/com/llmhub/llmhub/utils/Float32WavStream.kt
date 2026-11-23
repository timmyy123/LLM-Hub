package com.llmhub.llmhub.utils

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Linear resampler fed one mono sample at a time, so a long recording never has to sit in memory
 * at its source rate. Output matches resampling the whole track at once with linear interpolation:
 * floor(n * to / from) samples, positions past the last input sample hold the last sample.
 */
internal class StreamingLinearResampler(fromRate: Int, toRate: Int, private val sink: (Float) -> Unit) {
    private val from = fromRate.coerceAtLeast(1)
    private val to = toRate.coerceAtLeast(1)
    private val ratio = from.toDouble() / to
    // The last few input samples: an output position never lags the newest input by more than `ratio`.
    private val history = FloatArray(kotlin.math.ceil(ratio).toInt() + 3)
    private var count = 0L
    private var emitted = 0L

    fun push(sample: Float) {
        history[(count % history.size).toInt()] = sample
        count++
        // Emit only outputs the final length is sure to include (floor(n * to / from) grows with n)
        // whose two neighbouring input samples have both arrived.
        val confirmed = outputLength(count)
        while (emitted < confirmed) {
            val pos = emitted * ratio
            if (pos > count - 1) break
            emitAt(pos)
        }
    }

    fun finish() {
        if (count == 0L) return
        val target = outputLength(count).coerceAtLeast(1)
        while (emitted < target) emitAt(emitted * ratio)
    }

    private fun outputLength(n: Long): Long = ((n.toDouble() * to) / from).toLong()

    private fun emitAt(pos: Double) {
        val i0 = pos.toLong().coerceIn(0L, count - 1)
        val i1 = (i0 + 1).coerceAtMost(count - 1)
        val frac = pos - i0
        val a = history[(i0 % history.size).toInt()]
        val b = history[(i1 % history.size).toInt()]
        sink(((1.0 - frac) * a + frac * b).toFloat())
        emitted++
    }
}

/**
 * Builds a mono IEEE float32 WAV in a single byte array, header included, so the samples are never
 * held twice (float array, then bytes, then header + bytes). Sized from the track's duration when
 * known. A few hundred ms of spare room is left as trailing silence (zero bytes are 0.0f) rather
 * than copying the whole array again to trim it.
 */
internal class Float32WavBuilder(private val sampleRate: Int, estimatedSamples: Long) {
    private var data = ByteArray(HEADER + capacityFor(estimatedSamples) * 4)
    private var samples = 0

    fun add(value: Float) {
        val offset = HEADER + samples * 4
        if (offset + 4 > data.size) grow()
        val bits = java.lang.Float.floatToRawIntBits(value.coerceIn(-1f, 1f))
        data[offset] = bits.toByte()
        data[offset + 1] = (bits ushr 8).toByte()
        data[offset + 2] = (bits ushr 16).toByte()
        data[offset + 3] = (bits ushr 24).toByte()
        samples++
    }

    val sampleCount: Int get() = samples

    fun build(): ByteArray {
        val capacity = (data.size - HEADER) / 4
        val out = if (capacity - samples > MAX_SILENT_TAIL) data.copyOf(HEADER + samples * 4) else data
        val dataBytes = out.size - HEADER
        ByteBuffer.wrap(out, 0, HEADER).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray())
            putInt(36 + dataBytes)
            put("WAVE".toByteArray())
            put("fmt ".toByteArray())
            putInt(16)
            putShort(3) // IEEE float
            putShort(1)
            putInt(sampleRate)
            putInt(sampleRate * 4)
            putShort(4)
            putShort(32)
            put("data".toByteArray())
            putInt(dataBytes)
        }
        return out
    }

    private fun grow() {
        val capacity = (data.size - HEADER) / 4
        val extra = (capacity / 4).coerceAtLeast(sampleRate * 10)
        val newCapacity = (capacity.toLong() + extra).coerceAtMost(MAX_SAMPLES.toLong()).toInt()
        if (newCapacity <= capacity) throw OutOfMemoryError("Audio too long for a single WAV buffer")
        data = data.copyOf(HEADER + newCapacity * 4)
    }

    private fun capacityFor(estimated: Long): Int {
        val wanted = if (estimated > 0) estimated + sampleRate / 4 else sampleRate * 30L
        return wanted.coerceIn(1L, MAX_SAMPLES.toLong()).toInt()
    }

    companion object {
        const val HEADER = 44
        private const val MAX_SILENT_TAIL = 16_000
        private const val MAX_SAMPLES = (Int.MAX_VALUE - HEADER - 8) / 4

        /** True when the first 44 bytes describe the canonical mono 16 kHz float32 WAV the models take. */
        fun isMono16kFloat32Header(header: ByteArray, targetRate: Int): Boolean {
            if (header.size < HEADER) return false
            if (String(header, 0, 4, Charsets.US_ASCII) != "RIFF") return false
            if (String(header, 8, 4, Charsets.US_ASCII) != "WAVE") return false
            val bb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            val audioFormat = bb.getShort(20).toInt() and 0xFFFF
            val channels = bb.getShort(22).toInt() and 0xFFFF
            val sampleRate = bb.getInt(24)
            val bitsPerSample = bb.getShort(34).toInt() and 0xFFFF
            return audioFormat == 3 && channels == 1 && sampleRate == targetRate && bitsPerSample == 32
        }
    }
}
