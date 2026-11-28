package com.llmhub.llmhub.inference

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.LiteRtTtsModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
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
 * whisper.cpp Whisper models the catalog ships (GitHub issue 128; whisper.cpp replaced WhisperKit).
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

    /** whisper.cpp, which replaced WhisperKit. `-e threads N` overrides the thread count (default 4). */
    @Test fun whisperCppTiny() = whisperCpp("Whisper Tiny (Multilingual)")

    @Test fun whisperCppTinyEnglish() = whisperCpp("Whisper Tiny (English-only)")

    @Test fun whisperCppBase() = whisperCpp("Whisper Base (Multilingual)")

    @Test fun whisperCppBaseEnglish() = whisperCpp("Whisper Base (English-only)")

    @Test fun whisperCppSmallEnglish() = whisperCpp("Whisper Small (English-only)")

    /**
     * Any ggml file from ggerganov/whisper.cpp on any accelerator, outside the catalog: `-e file ggml-medium-q5_0.bin`
     * (default ggml-tiny-q8_0.bin), `-e accelerator CPU|VULKAN|OPENCL|HEXAGON`, `-e threads N`, `-e flash false`,
     * `-e language en`, `-e skipLong true`. The file is deleted afterwards unless the catalog lists it.
     */
    @Test fun whisperCppFile() {
        val args = InstrumentationRegistry.getArguments()
        args.getString("threads")?.toIntOrNull()?.let { WhisperCppService.threads = it }
        WhisperCppService.flashAttn = args.getString("flash") != "false"
        whisperCppOn(args.getString("file") ?: "ggml-tiny-q8_0.bin",
            WhisperAccelerator.valueOf(args.getString("accelerator") ?: "CPU"), args.getString("language") ?: "auto")
    }

    // Named runs for Device Farm, which cannot pass instrumentation arguments. Vulkan is for Mali and other
    // non-Adreno GPUs: ggml's Vulkan shaders do not compile on Adreno.
    @Test fun whisperCppTinyCpu() = whisperCppOn("ggml-tiny-q8_0.bin", WhisperAccelerator.CPU)

    @Test fun whisperCppBaseCpu() = whisperCppOn("ggml-base-q8_0.bin", WhisperAccelerator.CPU)

    @Test fun whisperCppTinyVulkan() = whisperCppOn("ggml-tiny-q8_0.bin", WhisperAccelerator.VULKAN)

    @Test fun whisperCppBaseVulkan() = whisperCppOn("ggml-base-q8_0.bin", WhisperAccelerator.VULKAN)

    private fun whisperCppOn(name: String, accelerator: WhisperAccelerator, language: String = "auto") = runBlocking {
        val file = File(models, name)
        OnDeviceFixtures.download(file, "${com.llmhub.llmhub.data.AsrModels.WHISPER_CPP_BASE_URL}/$name", WHISPER_CPP_SHA256.getValue(name))
        try {
            measure("whisper.cpp $name ${accelerator.name.lowercase()} ${WhisperCppService.threads}t fa=${WhisperCppService.flashAttn}") {
                val service = WhisperCppService(context)
                check(service.load(file, accelerator)) { "whisper.cpp did not load $name on $accelerator" }
                Recognizer({ service.transcribe(it, language).orEmpty() }, service::release)
            }
        } finally {
            if (ModelData.models.none { it.localFileName() == name }) file.delete()
        }
    }

    private fun liteRt(repo: String, file: String, sha256: String, preferGpu: Boolean) = runBlocking {
        val model = File(models, file)
        OnDeviceFixtures.download(model, "https://huggingface.co/$repo/resolve/main/$file", sha256)
        measure("litert $file ${if (preferGpu) "gpu" else "cpu"}") {
            val engine = ParakeetAsrEngine.create(context, model, preferGpu)
            checkNotNull(engine) { "AudioTranscriber did not load $file" }
            Recognizer({ engine.transcribe(it).orEmpty() }, engine::close)
        }
    }

    private fun whisperCpp(name: String) = runBlocking {
        val catalog = ModelData.models.first { it.name == name }
        OnDeviceFixtures.download(File(models, catalog.localFileName()), catalog.url, WHISPER_CPP_SHA256.getValue(catalog.url.substringAfterLast('/')))
        InstrumentationRegistry.getArguments().getString("threads")?.toIntOrNull()?.let { WhisperCppService.threads = it }
        measure("whisper.cpp ${name.lowercase()} ${WhisperCppService.threads} threads") {
            val service = SpeechRecognitionService(context)
            check(service.loadModel(catalog, WhisperBackend.CPU)) { "whisper.cpp did not load $name" }
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
        val WHISPER_CPP_SHA256 = mapOf(
            "ggml-tiny-q8_0.bin" to "c2085835d3f50733e2ff6e4b41ae8a2b8d8110461e18821b09a15c40c42d1cca",
            "ggml-tiny.en-q8_0.bin" to "5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94",
            "ggml-base-q8_0.bin" to "c577b9a86e7e048a0b7eada054f4dd79a56bbfa911fbdacf900ac5b567cbb7d9",
            "ggml-base.en-q8_0.bin" to "a4d4a0768075e13cfd7e19df3ae2dbc4a68d37d36a7dad45e8410c9a34f8c87e",
            "ggml-small-q8_0.bin" to "49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f",
            "ggml-small.en-q8_0.bin" to "67a179f608ea6114bd3fdb9060e762b588a3fb3bd00c4387971be4d177958067",
            "ggml-medium-q5_0.bin" to "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f",
            "ggml-medium.en-q5_0.bin" to "76733e26ad8fe1c7a5bf7531a9d41917b2adc0f20f2e4f5531688a8c6cd88eb0",
            "ggml-medium-q8_0.bin" to "42a1ffcbe4167d224232443396968db4d02d4e8e87e213d3ee2e03095dea6502",
            "ggml-large-v3-turbo-q5_0.bin" to "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
            "ggml-large-v3-turbo-q8_0.bin" to "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
            "ggml-large-v3-q5_0.bin" to "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1",
        )
    }
}
