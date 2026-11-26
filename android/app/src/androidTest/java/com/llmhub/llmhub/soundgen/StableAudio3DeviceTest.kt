package com.llmhub.llmhub.soundgen

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.ModelDownloader
import com.llmhub.llmhub.data.MusicGeneratorBackend
import com.llmhub.llmhub.data.hasCompleteDownloadedBundle
import io.ktor.client.HttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Opt-in physical-device check for the Stable Audio 3 models. Downloads the model through the app's own
 * ModelDownloader into the folder the app uses (skipped when already complete), generates each requested
 * length, and keeps the model afterwards, so it is non-destructive. Arguments:
 *   -e sa3Model "Stable Audio 3 Small Music"   (default)
 *   -e sa3Seconds 10,30                        (default 10,30)
 *   -e sa3Prompt "..."                         (default a lo-fi hip hop prompt)
 * Results (timings, peak memory, RMS/peak of each WAV) go to files/sa3-test-results and logcat.
 */
class StableAudio3DeviceTest {
    @Test fun generatesAudio() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelName = args.getString("sa3Model") ?: "Stable Audio 3 Small Music"
        val lengths = (args.getString("sa3Seconds") ?: "10,30").split(',').map { it.trim().toDouble() }
        val prompt = args.getString("sa3Prompt") ?: "Warm lo-fi hip hop beat with soft piano chords and vinyl crackle"
        val model = ModelData.musicGenerationModels.first { it.name == modelName }
        val results = File(context.filesDir, "sa3-test-results").apply { mkdirs() }
        val report = JSONObject().put("device", Build.MODEL).put("soc", Build.SOC_MODEL).put("model", modelName)
        fun save() = File(results, "results.json").writeText(report.toString(2))

        if (!model.hasCompleteDownloadedBundle(context)) {
            val started = SystemClock.elapsedRealtime()
            var lastLogged = -1L
            HttpClient(Android).use { client ->
                ModelDownloader(client, context).downloadModel(model).collect { status ->
                    val percent = if (status.totalBytes > 0) status.downloadedBytes * 100 / status.totalBytes else -1
                    if (percent / 5 != lastLogged / 5) {
                        lastLogged = percent
                        Log.i(TAG, "download $percent% (${status.downloadedBytes / 1_000_000} MB)")
                    }
                }
            }
            report.put("downloadMs", SystemClock.elapsedRealtime() - started)
        }
        assertTrue("$modelName download incomplete", model.hasCompleteDownloadedBundle(context))
        save()

        val failures = mutableListOf<String>()
        for ((index, seconds) in lengths.withIndex()) {
            val entry = JSONObject().put("seconds", seconds).put("prompt", prompt)
            report.put("run$index", entry)
            val started = SystemClock.elapsedRealtime()
            val sampler = launch(Dispatchers.IO) {
                var peak = 0L
                while (isActive) {
                    val (rss, swap) = memoryKb()
                    if (rss + swap > peak) {
                        peak = rss + swap
                        entry.put("peakRssMb", rss / 1024).put("peakSwapMb", swap / 1024)
                        save()
                    }
                    delay(1000)
                }
            }
            val wav = try {
                MusicGeneratorBackend.generateMusic(context, modelName, prompt, seconds) { }
            } finally {
                sampler.cancel()
            }
            entry.put("generateMs", SystemClock.elapsedRealtime() - started)
            if (wav == null) { failures += "$seconds s: generation failed"; save(); continue }
            val (rms, peak, frames) = stats(wav)
            entry.put("rms", rms).put("peak", peak).put("zeroCrossingsPerSec", zeroCrossingsPerSecond(wav)).put("frames", frames).put("wavBytes", wav.length())
            wav.copyTo(File(results, "sa3_${seconds.toInt()}s.wav"), overwrite = true)
            wav.delete()
            save()
            Log.i(TAG, "$modelName ${seconds}s: $entry")
            val expected = Math.round(seconds * 44_100)
            if (frames != expected) failures += "$seconds s: $frames frames, expected $expected"
            if (rms.isNaN() || rms < 1e-3) failures += "$seconds s: silent or NaN output (rms $rms)"
        }
        MusicGeneratorBackend.unloadModel()
        save()
        assertTrue(failures.joinToString("; "), failures.isEmpty())
    }

    private fun memoryKb(): Pair<Long, Long> {
        val status = File("/proc/self/status").readLines()
        fun field(name: String) = status.firstOrNull { it.startsWith("$name:") }?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull() ?: 0L
        return field("VmRSS") to field("VmSwap")
    }

    /** RMS and peak of a PCM16 stereo WAV, and its frame count, streamed so long clips fit in memory. */
    private fun stats(wav: File): Triple<Double, Double, Long> {
        var sum = 0.0
        var peak = 0.0
        var count = 0L
        RandomAccessFile(wav, "r").use { file ->
            file.seek(44)
            val bytes = ByteArray(1 shl 20)
            while (true) {
                val n = file.read(bytes)
                if (n <= 0) break
                val shorts = ByteBuffer.wrap(bytes, 0, n - n % 2).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                while (shorts.hasRemaining()) {
                    val v = shorts.get() / 32768.0
                    sum += v * v; peak = maxOf(peak, abs(v)); count++
                }
            }
        }
        return Triple(if (count == 0L) 0.0 else sqrt(sum / count), peak, count / 2)
    }

    /** Left-channel zero crossings per second: a cheap brightness measure, to check the prompt steers the sound. */
    private fun zeroCrossingsPerSecond(wav: File): Double {
        var crossings = 0L
        var frames = 0L
        var previous = 0
        RandomAccessFile(wav, "r").use { file ->
            file.seek(44)
            val bytes = ByteArray(1 shl 20)
            while (true) {
                val n = file.read(bytes)
                if (n <= 0) break
                val shorts = ByteBuffer.wrap(bytes, 0, n - n % 4).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                while (shorts.remaining() >= 2) {
                    val left = shorts.get().toInt(); shorts.get()
                    if ((left >= 0) != (previous >= 0)) crossings++
                    previous = left; frames++
                }
            }
        }
        return if (frames == 0L) 0.0 else crossings * 44_100.0 / frames
    }

    private companion object { const val TAG = "StableAudio3DeviceTest" }
}
