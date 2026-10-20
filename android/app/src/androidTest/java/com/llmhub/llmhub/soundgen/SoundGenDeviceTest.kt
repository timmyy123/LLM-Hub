package com.llmhub.llmhub.soundgen

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.MusicGeneratorBackend
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Opt-in physical-device check that every SoundGen model loads and generates audio on the app's LiteRT runtime.
 * Stage all SoundGen files in files/<suite>-test-fixtures (their names are unique), or pass `-e downloadFixtures 1`
 * to fetch them on the phone (e.g. BrowserStack); each model is copied to the folder the downloader uses, generated,
 * and removed. Results go to files/<suite>-test-results and logcat.
 */
class SoundGenDeviceTest {
    @Test fun everyModelGeneratesAudio() = check("soundgen", ModelData.musicGenerationModels.map { it.name })

    /** Quick and HD (suite soundgen_app), which run on phones; HD Long needs more memory than they allow. */
    @Test fun quickAndHdGenerateAudio() = check("soundgen_app", listOf("SoundGen Quick (LiteRT)", "SoundGen HD (LiteRT)"))

    /** HD Long alone (suite soundgen_long), with the process's peak resident memory. */
    @Test fun hdLongGeneratesAudio() = check("soundgen_long", listOf("SoundGen HD Long (AudioGen)"))

    private fun check(suite: String, names: List<String>) = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = File(context.filesDir, "$suite-test-fixtures").apply { mkdirs() }
        val download = InstrumentationRegistry.getArguments().getString("downloadFixtures") == "1"
        val results = File(context.filesDir, "$suite-test-results").apply { mkdirs() }
        val report = JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL).put("android", Build.VERSION.RELEASE)
        fun save() = File(results, "results.json").writeText(report.toString(2))
        val failures = mutableListOf<String>()

        for (model in ModelData.musicGenerationModels.filter { it.name in names }) {
            val dir = File(File(context.filesDir, "models"), model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")).apply { mkdirs() }
            val entry = JSONObject()
            report.put(model.name, entry)
            try {
                for (url in listOf(model.url) + model.additionalFiles) {
                    val name = url.substringAfterLast('/').substringBefore('?')
                    val source = File(fixtures, name)
                    if (download) OnDeviceFixtures.download(source, url.substringBefore('?'), SHA256.getValue(name))
                    assertTrue("Missing fixture $name", source.isFile)
                    source.copyTo(File(dir, name), overwrite = true)
                }
                var started = SystemClock.elapsedRealtime()
                val loaded = MusicGeneratorBackend.loadModel(context, model.name)
                entry.put("loaded", loaded).put("loadMs", SystemClock.elapsedRealtime() - started)
                save()
                if (!loaded) { failures += "${model.name}: load failed"; continue }

                val seconds = minOf(5.0, MusicGeneratorBackend.durationRange(model.name).endInclusive.toDouble())
                started = SystemClock.elapsedRealtime()
                // Peak resident and swapped memory, saved as it grows in case the process is killed.
                val sampler = launch(Dispatchers.IO) {
                    var peak = 0L
                    while (isActive) {
                        val (rss, swap) = memoryKb()
                        if (rss + swap > peak) {
                            peak = rss + swap
                            entry.put("peakRssMb", rss / 1024).put("peakSwapMb", swap / 1024)
                                .put("peakAtMs", SystemClock.elapsedRealtime() - started)
                            save()
                        }
                        delay(2000)
                    }
                }
                val wav = try {
                    MusicGeneratorBackend.generateMusic(context, model.name, "rain on a tin roof with distant thunder", seconds) { }
                } finally {
                    sampler.cancel()
                }
                entry.put("generateMs", SystemClock.elapsedRealtime() - started).put("seconds", seconds)
                if (wav == null) { failures += "${model.name}: generation failed"; save(); continue }
                val rms = pcm16Rms(wav.readBytes())
                wav.copyTo(File(results, dir.name + ".wav"), overwrite = true)
                entry.put("wavBytes", wav.length()).put("rms", rms)
                save()
                Log.i("SoundGenDeviceTest", "${model.name}: $entry")
                if (rms < 1e-3) failures += "${model.name}: silent output (rms $rms)"
                if (rms.isNaN()) failures += "${model.name}: NaN output"
            } finally {
                MusicGeneratorBackend.unloadModel()
                dir.deleteRecursively()
            }
        }
        save()
        assertTrue(failures.joinToString("; "), failures.isEmpty())
    }

    private companion object {
        /** SHA-256 of every SoundGen file at the commits ModelData pins, as in tools/imagegen_devicefarm/prepare_fixtures.py. */
        val SHA256 = mapOf(
            "dit_model.tflite" to "119fe0020f2fafd88366a2869fdbf2f1aaa3933f8daf64d2116dcc115348ee83",
            "autoencoder_model_undilated_1g.tflite" to "15c9c8236a48c64075124cad0737e5d8cb35c154f78cb40756db5d882076ec47",
            "conditioners_float32.tflite" to "c1139b09ea66a944966dfce4722075401f9eb723c1f59bfd5bdd797690bcce91",
            "spiece.model" to "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
            "dit_L256_int8.tflite" to "252da2091c8ab83ba8405c923a0f0e9fbd87402ae15d372c5b7075430fe0293c",
            "ae_dec_L256_int8.tflite" to "8d35f984242999ff6d7ce3975b3c5632e1c3d35c0a222e7869918202ee3f815c",
            "t5gemma_enc_int8.tflite" to "a418899d084e7141d508e4fb865fbbf177e5a689b1a3c57bcc90a9505c39566e",
            "tokenizer.model" to "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
            "dit_L2048_int8.tflite" to "cbab297f07650140b604ad5ba1e1a12d2f9ed320114b5831ad5aa383092b4cf7",
            "ae_dec_L2048_int8.tflite" to "a8db29d280c2ff0e6576ce1db0f7850a982492c51e8f71e739baa90c83db75c4",
        )
    }

    /** This process's VmRSS and VmSwap in KB. */
    private fun memoryKb(): Pair<Long, Long> {
        val status = File("/proc/self/status").readLines()
        fun field(name: String) = status.firstOrNull { it.startsWith("$name:") }?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull() ?: 0L
        return field("VmRSS") to field("VmSwap")
    }

    /** RMS of a PCM16 WAV's samples, in [0, 1]. */
    private fun pcm16Rms(bytes: ByteArray): Double {
        val samples = ByteBuffer.wrap(bytes, 44, bytes.size - 44).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        var sum = 0.0
        val count = samples.remaining()
        while (samples.hasRemaining()) { val v = samples.get() / 32768.0; sum += v * v }
        return if (count == 0) 0.0 else sqrt(sum / count)
    }
}
