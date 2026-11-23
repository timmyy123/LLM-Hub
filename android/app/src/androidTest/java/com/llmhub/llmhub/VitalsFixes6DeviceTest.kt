package com.llmhub.llmhub

import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.utils.AudioConversionUtils
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

/**
 * 4.6.8 vitals: OutOfMemoryError in AudioConversionUtils.convertUriToFloat32Wav (Oppo Find X9, ROG Phone 7)
 * on long audio files. Runs the real conversion (MediaExtractor + MediaCodec) on files the old code
 * could not hold in a 512 MB heap.
 */
class VitalsFixes6DeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val tag = "VitalsFixes6"

    /** Streams a 16-bit PCM WAV of a 440 Hz tone (amplitude 0.5) to disk without holding it in memory. */
    private fun writeToneWav(file: File, seconds: Int, rate: Int, channels: Int) {
        val frames = seconds.toLong() * rate
        val dataBytes = frames * channels * 2
        BufferedOutputStream(FileOutputStream(file), 1 shl 20).use { out ->
            val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            h.put("RIFF".toByteArray()).putInt((36 + dataBytes).toInt()).put("WAVE".toByteArray())
            h.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(channels.toShort())
            h.putInt(rate).putInt(rate * channels * 2).putShort((channels * 2).toShort()).putShort(16)
            h.put("data".toByteArray()).putInt(dataBytes.toInt())
            out.write(h.array())
            val chunk = ByteBuffer.allocate(rate * channels * 2).order(ByteOrder.LITTLE_ENDIAN)
            var frame = 0L
            while (frame < frames) {
                chunk.clear()
                val n = minOf(rate.toLong(), frames - frame).toInt()
                for (i in 0 until n) {
                    val v = (sin(2 * PI * 440 * (frame + i) / rate) * 0.5 * 32767).toInt().toShort()
                    repeat(channels) { chunk.putShort(v) }
                }
                out.write(chunk.array(), 0, n * channels * 2)
                frame += n
            }
        }
    }

    private fun floatsOf(wav: ByteArray): ByteBuffer = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)

    /** Zero crossings over one second starting at [startSample] of a 16 kHz float WAV. */
    private fun crossingsPerSecond(bb: ByteBuffer, startSample: Int): Int {
        var crossings = 0
        var prev = bb.getFloat(44 + startSample * 4)
        for (i in 1 until 16_000) {
            val v = bb.getFloat(44 + (startSample + i) * 4)
            if ((prev < 0f) != (v < 0f)) crossings++
            prev = v
        }
        return crossings
    }

    @Test
    fun hourLong48kStereoWavConvertsWithoutOom() {
        val file = File(context.cacheDir, "vitals6_hour.wav")
        try {
            val seconds = 3600
            writeToneWav(file, seconds, 48_000, 2)
            Log.i(tag, "Wrote ${file.length() / 1_000_000} MB, heap limit ${Runtime.getRuntime().maxMemory() / 1_000_000} MB")

            // What the 4.6.8 fast path did first: read the whole file into memory.
            val oldPathOom = try { file.readBytes(); false } catch (_: OutOfMemoryError) { true }
            Log.i(tag, "4.6.8 whole-file read OOM reproduced: $oldPathOom")
            System.gc()

            val t0 = SystemClock.elapsedRealtime()
            val wav = runBlocking { AudioConversionUtils.convertUriToFloat32Wav(context, Uri.fromFile(file)) }
            val ms = SystemClock.elapsedRealtime() - t0
            assertNotNull("conversion returned null (OOM or decode failure)", wav)
            wav!!
            val bb = floatsOf(wav)
            val samples = bb.getInt(40) / 4
            Log.i(tag, "Converted in ${ms / 1000} s: ${wav.size / 1_000_000} MB, $samples samples")
            assertEquals(wav.size - 44, bb.getInt(40))
            assertTrue("samples=$samples", samples in seconds * 16_000..seconds * 16_000 + 4_000)
            // 440 Hz tone: 880 zero crossings a second, amplitude 0.5, at the start, middle and end.
            for (start in listOf(16_000, seconds * 8_000, (seconds - 2) * 16_000)) {
                val c = crossingsPerSecond(bb, start)
                assertTrue("crossings=$c at $start", abs(c - 880) <= 2)
            }
            var peak = 0f
            for (i in 0 until 16_000) peak = maxOf(peak, abs(bb.getFloat(44 + (seconds * 8_000 + i) * 4)))
            assertEquals(0.5f, peak, 0.01f)
        } finally {
            file.delete()
        }
    }

    @Test
    fun shortStereoFileMatchesExpectedLengthAndPitch() {
        val file = File(context.cacheDir, "vitals6_short.wav")
        try {
            writeToneWav(file, 3, 44_100, 2)
            val wav = runBlocking { AudioConversionUtils.convertUriToFloat32Wav(context, Uri.fromFile(file)) }!!
            val bb = floatsOf(wav)
            val samples = bb.getInt(40) / 4
            assertTrue("samples=$samples", samples in 48_000..52_000)
            assertEquals(3, bb.getShort(20).toInt()) // IEEE float
            assertEquals(1, bb.getShort(22).toInt())
            assertEquals(16_000, bb.getInt(24))
            assertTrue(abs(crossingsPerSecond(bb, 8_000) - 880) <= 2)
        } finally {
            file.delete()
        }
    }

    @Test
    fun mono16kFloatWavPassesThroughUnchanged() {
        val file = File(context.cacheDir, "vitals6_float.wav")
        try {
            val n = 16_000 * 600 // 10 minutes
            val bytes = ByteArray(44 + n * 4)
            val bb = floatsOf(bytes)
            bb.put("RIFF".toByteArray()).putInt(36 + n * 4).put("WAVE".toByteArray())
            bb.put("fmt ".toByteArray()).putInt(16).putShort(3).putShort(1)
            bb.putInt(16_000).putInt(64_000).putShort(4).putShort(32)
            bb.put("data".toByteArray()).putInt(n * 4)
            for (i in 0 until n) bb.putFloat((sin(2 * PI * 440 * i / 16_000) * 0.25).toFloat())
            file.writeBytes(bytes)
            val wav = runBlocking { AudioConversionUtils.convertUriToFloat32Wav(context, Uri.fromFile(file)) }!!
            assertArrayEquals(bytes, wav)
        } finally {
            file.delete()
        }
    }
}
