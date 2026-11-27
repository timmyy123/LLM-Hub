package com.llmhub.llmhub.inference

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.LiteRtTtsModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The non-Chinese int8 `.litertlm` speech recognizers in litert-community's ASR collection (Whisper Tiny,
 * Moonshine Tiny, Parakeet CTC, Parakeet TDT v3) on the MediaPipe AudioTranscriber task, next to the
 * FP32 WhisperKit Tiny the catalog ships (GitHub issue 128).
 * Each test downloads what it needs into the app's models folder; results go to logcat as `RESULT {...}`.
 */
class WhisperLiteRtDeviceTest {
    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val models = File(context.filesDir, "models").apply { mkdirs() }

    @Test fun liteRtInt8Cpu() = liteRt(WHISPER, "whisper_tiny_30s_i8.litertlm", LITERT_SHA256, preferGpu = false)

    @Test fun liteRtInt8Gpu() = liteRt(WHISPER, "whisper_tiny_30s_i8.litertlm", LITERT_SHA256, preferGpu = true)

    @Test fun liteRtInt8StatefulCpu() = liteRt(WHISPER, "whisper_tiny_30s_i8_stateful.litertlm", LITERT_STATEFUL_SHA256, preferGpu = false)

    @Test fun liteRtInt8StatefulGpu() = liteRt(WHISPER, "whisper_tiny_30s_i8_stateful.litertlm", LITERT_STATEFUL_SHA256, preferGpu = true)

    @Test fun moonshineTinyCpu() = liteRt(MOONSHINE, "moonshine_tiny_5s_i8.litertlm", MOONSHINE_SHA256, preferGpu = false)

    @Test fun moonshineTinyGpu() = liteRt(MOONSHINE, "moonshine_tiny_5s_i8.litertlm", MOONSHINE_SHA256, preferGpu = true)

    @Test fun parakeetCtcCpu() = liteRt(PARAKEET_CTC, "parakeet_ctc_0.6b_5s_i8.litertlm", PARAKEET_CTC_SHA256, preferGpu = false)

    @Test fun parakeetCtcGpu() = liteRt(PARAKEET_CTC, "parakeet_ctc_0.6b_5s_i8.litertlm", PARAKEET_CTC_SHA256, preferGpu = true)

    /** The Parakeet TDT v3 file the catalog already ships, for reference. */
    @Test fun parakeetTdtV3Cpu() = liteRt(PARAKEET_TDT, "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm", PARAKEET_TDT_SHA256, preferGpu = false)

    @Test fun parakeetTdtV3Gpu() = liteRt(PARAKEET_TDT, "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm", PARAKEET_TDT_SHA256, preferGpu = true)

    @Test fun whisperKitNpu() = whisperKit(WhisperBackend.NPU)

    @Test fun whisperKitGpu() = whisperKit(WhisperBackend.GPU)

    @Test fun whisperKitCpu() = whisperKit(WhisperBackend.CPU)

    private fun liteRt(repo: String, file: String, sha256: String, preferGpu: Boolean) = runBlocking {
        val model = File(models, file)
        OnDeviceFixtures.download(model, "https://huggingface.co/$repo/resolve/main/$file", sha256)
        measure("litert $file ${if (preferGpu) "gpu" else "cpu"}") {
            val engine = ParakeetAsrEngine.create(context, model, preferGpu)
            checkNotNull(engine) { "AudioTranscriber did not load $file" }
            Recognizer({ engine.transcribe(it).orEmpty() }, engine::close)
        }
    }

    private fun whisperKit(backend: WhisperBackend) = runBlocking {
        val catalog = ModelData.models.first { it.name == "Whisper Tiny (Multilingual)" }
        val dir = File(models, "Whisper_Tiny_Multilingual").apply { mkdirs() }
        for (url in listOf(catalog.url) + catalog.additionalFiles) {
            val name = url.substringAfterLast('/')
            OnDeviceFixtures.download(File(dir, name), url, WHISPERKIT_SHA256.getValue(name))
        }
        measure("whisperkit tiny ${backend.name.lowercase()}") {
            val service = SpeechRecognitionService(context)
            check(service.loadModel(catalog, backend)) { "WhisperKit did not load on $backend" }
            Recognizer({ service.transcribe(it).orEmpty() }, service::release)
        }
    }

    /**
     * Every segment AudioTranscriber emits for 39 s of speech, logged as it comes, to see what
     * happens past Whisper's first 30 s window. `-e file`, `-e repo`, `-e sha256` pick another model.
     */
    @Test fun rawSegmentsForLongAudio() = runBlocking<Unit> {
        val args = InstrumentationRegistry.getArguments()
        val file = args.getString("file") ?: "whisper_tiny_30s_i8.litertlm"
        val model = File(models, file)
        OnDeviceFixtures.download(model, "https://huggingface.co/${args.getString("repo") ?: WHISPER}/resolve/main/$file",
            args.getString("sha256") ?: LITERT_SHA256)
        val passage = SpeechModelsDeviceTest().speech(LiteRtTtsModel.KOKORO_NAME, SpeechModelsDeviceTest.PASSAGE)
        val pcm = passage + passage + passage
        val shorts = java.nio.ByteBuffer.wrap(pcm).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val samples = FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
        val chunk = (args.getString("chunkMs")?.toInt() ?: 100) * 16
        val transcriber = com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriber.createFromOptions(context,
            com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriber.AudioTranscriberOptions.builder()
                .setBaseOptions(com.google.mediapipe.tasks.core.BaseOptions.builder().setModelAssetPath(model.absolutePath)
                    .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU).build())
                .setSampleRateHz(16000).setEnableInterimResults(true)
                .setModelArchitecture(com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriber.ModelArchitecture.AUTO)
                .setCacheDir(File(context.cacheDir, "parakeet_asr").apply { mkdirs() }.absolutePath).build())
        fun log(at: String, result: com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriberResult?) {
            for (s in result?.segments().orEmpty()) Log.i(TAG, "SEGMENT at=$at final=${s.isFinal} \"${s.transcript()}\"")
        }
        try {
            var offset = 0
            while (offset < samples.size) {
                val end = minOf(offset + chunk, samples.size)
                log("%.1fs".format(end / 16000.0), transcriber.transcribe(samples.copyOfRange(offset, end), false))
                offset = end
            }
            log("finish", transcriber.finishStream())
        } finally {
            transcriber.close()
        }
    }

    /**
     * One of litert-community's other Whisper graphs (base, medium, large-v3, large-v3-turbo, ACFT) as a
     * `.litertlm` bundle: the header, metadata and tokenizer come from the test assets (built with
     * litert-lm-builder), the tablet downloads the `.tflite` itself and writes it after them, which gives
     * the same bytes as running the builder on the real file. `-e variant <id>` from
     * assets/whisper_litertlm/index.json, `-e gpu true` for the GPU. The bundle is deleted afterwards.
     */
    @Test fun bundle() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        val id = requireNotNull(args.getString("variant")) { "pass -e variant <id>" }
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val index = org.json.JSONArray(assets.open("whisper_litertlm/index.json").bufferedReader().readText())
        val entry = (0 until index.length()).map { index.getJSONObject(it) }.first { it.getString("id") == id }
        val bundle = File(models, "test_$id.litertlm")
        try {
            val started = SystemClock.elapsedRealtime()
            val digest = java.security.MessageDigest.getInstance("SHA-256")
            bundle.outputStream().buffered(1 shl 20).use { out ->
                assets.open("whisper_litertlm/$id.prefix").use { it.copyTo(out) }
                val connection = java.net.URL(entry.getString("url")).openConnection() as java.net.HttpURLConnection
                connection.connectTimeout = 30_000
                connection.readTimeout = 120_000
                connection.inputStream.use { input ->
                    val buffer = ByteArray(1 shl 20)
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        digest.update(buffer, 0, n)
                        out.write(buffer, 0, n)
                    }
                }
            }
            val sha = digest.digest().joinToString("") { "%02x".format(it) }
            check(sha == entry.getString("sha256")) { "$id: downloaded .tflite has sha256 $sha" }
            check(bundle.length() == entry.getLong("prefixSize") + entry.getLong("tfliteSize")) { "$id: bundle size ${bundle.length()}" }
            // The ACFT graphs list the decode subgraph's inputs in another order than the decode signature;
            // LiteRT then pairs buffers with the wrong tensors. Rewrite the subgraph's order (int32 tensor indices).
            val patch = entry.optJSONArray("patch") ?: org.json.JSONArray()
            if (patch.length() > 0) java.io.RandomAccessFile(bundle, "rw").use { file ->
                for (i in 0 until patch.length()) {
                    val (offset, tensor) = patch.getJSONArray(i).let { it.getLong(0) to it.getInt(1) }
                    file.seek(entry.getLong("prefixSize") + offset)
                    file.write(java.nio.ByteBuffer.allocate(4).order(java.nio.ByteOrder.LITTLE_ENDIAN).putInt(tensor).array())
                }
            }
            Log.i(TAG, "BUNDLE $id ${bundle.length()} bytes in ${SystemClock.elapsedRealtime() - started} ms")
            val gpu = args.getString("gpu") == "true"
            measure("bundle $id ${if (gpu) "gpu" else "cpu"}") {
                val engine = ParakeetAsrEngine.create(context, bundle, preferGpu = gpu)
                checkNotNull(engine) { "AudioTranscriber did not load $id" }
                Recognizer({ engine.transcribe(it).orEmpty() }, engine::close)
            }
        } finally {
            bundle.delete()
        }
    }

    private class Recognizer(val transcribe: suspend (ByteArray) -> String, val close: () -> Unit)

    private suspend fun measure(label: String, load: suspend () -> Recognizer) {
        val speech = SpeechModelsDeviceTest()
        val clips = linkedMapOf(
            "sentence" to (SpeechModelsDeviceTest.SENTENCE to speech.speech(LiteRtTtsModel.KOKORO_NAME, SpeechModelsDeviceTest.SENTENCE)),
            "passage" to (SpeechModelsDeviceTest.PASSAGE to speech.speech(LiteRtTtsModel.KOKORO_NAME, SpeechModelsDeviceTest.PASSAGE)),
        )
        // About 40 seconds with no repeated sentences: Whisper's decoder trims repeated text.
        // `-e skipLong true` leaves out the 32-second clip (the big Whisper graphs take many minutes on it).
        if (InstrumentationRegistry.getArguments().getString("skipLong") != "true") {
            clips["long"] = SpeechModelsDeviceTest.LONG to speech.speech(LiteRtTtsModel.KOKORO_NAME, SpeechModelsDeviceTest.LONG)
        }

        System.gc()
        val pssBefore = Debug.getPss()
        val report = JSONObject().put("engine", label).put("soc", Build.SOC_MODEL)
        var started = SystemClock.elapsedRealtime()
        val recognizer = load()
        report.put("loadMs", SystemClock.elapsedRealtime() - started).put("pssAfterLoadMb", (Debug.getPss() - pssBefore) / 1024)
        var worst = 1.0
        try {
            val runs = JSONArray()
            // The first sentence is a warm-up run; it is reported as "sentence (first)".
            for ((name, clip) in listOf("sentence (first)" to clips.getValue("sentence")) + clips.toList()) {
                val (text, audio) = clip
                started = SystemClock.elapsedRealtime()
                val transcript = recognizer.transcribe(audio)
                val ms = SystemClock.elapsedRealtime() - started
                val recall = recall(text, transcript)
                if (name != "sentence (first)") worst = minOf(worst, recall)
                runs.put(JSONObject().put("clip", name).put("audioSeconds", audio.size / 32000.0).put("ms", ms)
                    .put("recall", recall).put("transcript", transcript))
            }
            report.put("runs", runs).put("peakPssMb", (Debug.getPss() - pssBefore) / 1024)
        } finally {
            recognizer.close()
            Log.i(TAG, "RESULT $report")
        }
        assertTrue("$label heard only ${(worst * 100).toInt()}% of a clip: $report", worst >= 0.85)
    }

    private fun words(text: String) = text.lowercase().split(Regex("[^\\p{L}\\p{N}']+")).filter { it.isNotEmpty() }

    private fun recall(expected: String, transcript: String): Double {
        val heard = words(transcript).groupingBy { it }.eachCount().toMutableMap()
        val wanted = words(expected)
        val found = wanted.count { word -> (heard[word] ?: 0) > 0 && heard.merge(word, -1, Int::plus) != null }
        return found.toDouble() / wanted.size
    }

    private companion object {
        const val TAG = "WhisperLiteRtDeviceTest"
        const val WHISPER = "litert-community/whisper-tiny"
        const val MOONSHINE = "litert-community/moonshine-tiny"
        const val PARAKEET_CTC = "litert-community/parakeet-ctc-0.6b"
        const val PARAKEET_TDT = "litert-community/parakeet-tdt-0.6b-v3"
        const val MOONSHINE_SHA256 = "9876ebf427f9b46cf986d0e5f2cef9647249c54453082722b59f70411d15c4d1"
        const val PARAKEET_CTC_SHA256 = "408d95264bd605ea89afcd0c6d359fe16fb695abf61e077cc9968f7d8790d3e6"
        const val PARAKEET_TDT_SHA256 = "66c5a74cf6c2c1d2b990901f6cece5bb53e3214dbceafa0989900a6f0eeb3504"
        const val LITERT_SHA256 = "92f194600d98e1d94b1ff6f636d109c0f4ac66f3f6a1803f7a665e93daf78012"
        const val LITERT_STATEFUL_SHA256 = "acb6585d69c9c5bc7d392608d1da0a946cd18e0bd59677da95e78b40bc8d7a26"
        val WHISPERKIT_SHA256 = mapOf(
            "AudioEncoder.tflite" to "e8b3333d3625da8935ed61d0475cf5e10a6d096008cec128f0262c751a03406a",
            "MelSpectrogram.tflite" to "71c0c30975ce22c25e9ad17da917420506e99e101c78510acb213586936d26a1",
            "TextDecoder.tflite" to "86846936c0224e7a567ca5df54edba83fda58ba6e177a35ec63b7838fa255cfb",
            "tokenizer.json" to "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
            "config.json" to "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98",
        )
    }
}
