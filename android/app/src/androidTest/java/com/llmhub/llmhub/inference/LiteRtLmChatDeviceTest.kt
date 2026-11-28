package com.llmhub.llmhub.inference

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * A `.litertlm` (or ONNX) chat model from the catalog, run the way the app runs it: the catalog file downloaded where the app
 * keeps it, loaded through [UnifiedInferenceService] (which picks LiteRT-LM for every `.litertlm`), and asked through
 * the chat session API with the chat screen's "user: ...\nassistant:" prompt.
 *
 * `-e model "Gemma-4 12B"` picks the model (required). `-e hfToken hf_...` downloads gated files (the app's catalog
 * points at public mirrors; the token is only for testing a gated upstream file before it is mirrored). `-e backend CPU`
 * runs on the CPU (default GPU when the model supports it). Results go to logcat, tag LiteRtLmChat; LiteRT-LM's own
 * prefill/decode speeds are logged by the engine (tag LiteRtLmInference). Each scenario is its own test.
 */
class LiteRtLmChatDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val args = InstrumentationRegistry.getArguments()

    /** Device Farm passes no arguments: these pick this phone's own NPU build. */
    private var modelOverride: String? = null

    @Test fun gemma3NpuText() { modelOverride = "Gemma-3 1B NPU"; text() }

    @Test fun gemma4E2bNpuText() { modelOverride = "Gemma-4 E2B NPU"; text() }

    @Test fun gemma3nE2bText() { modelOverride = "Gemma-3n E2B"; text() }

    @Test fun gemma3nE2bVision() { modelOverride = "Gemma-3n E2B"; vision() }

    @Test fun gemma3nE2bAudio() { modelOverride = "Gemma-3n E2B"; audio() }

    @Test fun gemma3nE4bText() { modelOverride = "Gemma-3n E4B"; text() }

    @Test fun gemma3nE4bVision() { modelOverride = "Gemma-3n E4B"; vision() }

    @Test fun gemma3nE4bAudio() { modelOverride = "Gemma-3n E4B"; audio() }

    @Test fun phi4Text() { modelOverride = "Phi-4 Mini (INT8, 4k)"; text() }

    @Test fun aionText() { modelOverride = "Aion 1.0 Instruct 2.3B (LiteRT-LM)"; text() }

    @Test fun gemma3Int4Text() { modelOverride = "Gemma-3 1B (INT4, 2k)"; text() }

    @Test fun text() = withModel { service, model ->
        val first = chat(service, model, "capital", "user: What is the capital of France? Answer in one word.\nassistant:")
        assertTrue("Unexpected answer: $first", first.contains("Paris", ignoreCase = true))
        val second = chat(service, model, "followUp", "user: What is the capital of France? Answer in one word.\n\nassistant: $first\n\n" +
            "user: And what river runs through it? Answer in one word.\nassistant:")
        assertTrue("Unexpected follow-up: $second", second.contains("Seine", ignoreCase = true))
        val story = chat(service, model, "story", "user: Write a short story of about 150 words about a lighthouse keeper.\nassistant:")
        assertTrue("Story too short: $story", story.split(Regex("\\s+")).size >= 60)
    }

    /** The owner's own three-message chat (Tab S9+, 2026-11-28) in the chat screen's transcript format; replies logged. */
    @Test fun ownerChat() = withModel { service, model ->
        var transcript = ""
        for ((i, message) in listOf("gay", "what", "what we talked").withIndex()) {
            val prompt = transcript + "user: $message\nassistant:"
            val reply = chat(service, model, "ownerChat$i", prompt)
            transcript = prompt + " " + reply.trim() + "\n\n"
        }
    }

    /** Two turns in the chat screen's transcript format: the second needs a fact only the first turn gave. */
    @Test fun memory() = withModel { service, model ->
        val intro = "user: My name is Priya and my favourite fruit is mango. Just reply OK.\nassistant:"
        val first = chat(service, model, "memoryIntro", intro)
        val recall = chat(service, model, "memoryRecall", "${intro} ${first.trim()}\n\nuser: What is my name and what is my favourite fruit?\nassistant:")
        assertTrue("Forgot the first turn: $recall", recall.contains("Priya", ignoreCase = true) && recall.contains("mango", ignoreCase = true))
    }

    /**
     * The per-model MTP switch: defaults (on for Gemma, off otherwise), shown only for files with an MTP drafter, and a
     * flip reloads the engine with MTP off, then on. Uses `-e model` (a Gemma .litertlm with a drafter).
     */
    @Test fun mtpSwitch() = runBlocking {
        val prefs = com.llmhub.llmhub.data.ModelPreferences(context)
        for (m in ModelData.models.filter { it.modelFormat == "litertlm" && it.category != "embedding" && it.category != "tts" && it.category != "asr" }) {
            val downloaded = File(File(context.filesDir, "models"), m.localFileName()).length() == m.sizeBytes
            Log.i(TAG, "RESULT " + JSONObject().put("test", "mtpDefaults").put("model", m.name)
                .put("default", ModelData.defaultMtpEnabled(m)).put("downloaded", downloaded)
                .put("switchShown", downloaded && LiteRtLmInferenceService.supportsMtpSwitch(context, m)))
        }
        val name = args.getString("model") ?: error("Pass -e model <catalog name>")
        val model = ModelData.models.first { it.name == name }
        download(model)
        val service = UnifiedInferenceService(context)
        val original = prefs.getModelConfig(model.name)
        try {
            service.setAgentToolsEnabled(false)
            service.setGenerationParameters(null, 1, 1.0f, 0.0f, null, false, null)
            for (mtp in listOf(false, true)) {
                prefs.setModelConfig(model.name, (original ?: com.llmhub.llmhub.data.defaultConfigForModel(model)).copy(mtpEnabled = mtp))
                assertTrue("${model.name} must load", service.loadModel(model, LlmInference.Backend.GPU, true, true, null))
                chat(service, model, "story_mtp_$mtp", "user: Write a short story of about 150 words about a lighthouse keeper.\nassistant:")
            }
        } finally {
            if (original != null) prefs.setModelConfig(model.name, original)
            service.unloadModel()
        }
    }

    /** A red disc on white: the model must see it. */
    @Test fun vision() = withModel { service, model ->
        assumeTrue("${model.name} has no vision", model.supportsVision)
        val image = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888).also { bitmap ->
            Canvas(bitmap).apply {
                drawColor(Color.WHITE)
                drawCircle(256f, 256f, 180f, Paint().apply { color = Color.RED; isAntiAlias = true })
            }
        }
        val reply = chat(service, model, "vision", "user: What shape is in this picture and what colour is it? Answer in a few words.\nassistant:",
            images = listOf(image))
        assertTrue("Did not see the red circle: $reply", reply.contains("red", ignoreCase = true) &&
            (reply.contains("circle", ignoreCase = true) || reply.contains("disc", ignoreCase = true) || reply.contains("round", ignoreCase = true)))
    }

    /** Kokoro reads the test passage; the model must transcribe it. */
    @Test fun audio() = withModel { service, model ->
        assumeTrue("${model.name} has no audio", model.supportsAudio)
        val pcm = SpeechModelsDeviceTest().speech(com.llmhub.llmhub.data.LiteRtTtsModel.KOKORO_NAME, SpeechModelsDeviceTest.PASSAGE)
        val reply = chat(service, model, "audio", "user: Transcribe this audio exactly.\nassistant:", audio = GraniteSpeechEngine.wav(pcm))
        val expected = SpeechModelsDeviceTest.PASSAGE.lowercase().split(Regex("[^a-z']+")).filter { it.isNotEmpty() }
        val heard = reply.lowercase().split(Regex("[^a-z']+")).toSet()
        val recall = expected.count { it in heard }.toDouble() / expected.size
        Log.i(TAG, "RESULT " + JSONObject().put("model", model.name).put("test", "audioRecall").put("recall", recall))
        assertTrue("Only ${(recall * 100).toInt()}% of the passage came back: $reply", recall >= 0.8)
    }

    private fun withModel(block: suspend (UnifiedInferenceService, LLMModel) -> Unit) = runBlocking {
        val name = modelOverride ?: args.getString("model") ?: error("Pass -e model <catalog name>")
        // -e mtp 0 loads the model with MTP (speculative decoding) off.
        // BrowserStack only passes [a-zA-Z0-9 ._-@,/] in arguments, so names also match with punctuation ignored.
        fun key(text: String) = text.lowercase().replace(Regex("[^a-z0-9]"), "")
        // A name prefix such as "Gemma-3 1B NPU" picks this phone's own NPU build (each phone lists only its chip's).
        val model = (ModelData.models.firstOrNull { it.name == name || key(it.name) == key(name) }
            ?: ModelData.models.firstOrNull { (it.modelFormat == "litertlm" || it.modelFormat == "onnx") && key(it.name).startsWith(key(name)) }
            ?: error("No catalog model matches $name on ${Build.SOC_MODEL}")).let { if (args.getString("mtp") == "0") it.copy(supportsMtp = false) else it }
        check(model.modelFormat in setOf("litertlm", "task", "onnx")) { "$name is not a .litertlm, .task or ONNX chat model" }
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${Build.SOC_MODEL} model=$name")
        download(model)
        val service = UnifiedInferenceService(context)
        try {
            // The chat screen leaves agent tools off unless the user turns them on: -e agentTools 1 tests with them.
            service.setAgentToolsEnabled(args.getString("agentTools") == "1")
            // Greedy by default; -e topK 64 -e topP 0.95 -e temperature 1.0 uses the chat screen's Gemma defaults.
            service.setGenerationParameters(null, args.getString("topK")?.toInt() ?: 1, args.getString("topP")?.toFloat() ?: 1.0f,
                args.getString("temperature")?.toFloat() ?: 0.0f, null, args.getString("thinking") == "1", null)
            val backend = if (args.getString("backend") == "CPU" || !model.supportsGpu) LlmInference.Backend.CPU else LlmInference.Backend.GPU
            val started = SystemClock.elapsedRealtime()
            assertTrue("${model.name} must load", service.loadModel(model, backend, false, false, null))
            Log.i(TAG, "RESULT " + JSONObject().put("model", name).put("test", "load").put("backend", backend.name)
                .put("loadMs", SystemClock.elapsedRealtime() - started))
            block(service, model)
        } finally {
            service.unloadModel()
        }
    }

    private suspend fun chat(service: UnifiedInferenceService, model: LLMModel, label: String, prompt: String,
                             images: List<Bitmap> = emptyList(), audio: ByteArray? = null): String {
        val out = StringBuilder()
        val started = SystemClock.elapsedRealtime()
        try {
            service.generateResponseStreamWithSession(prompt, model, "litertlm-chat-test-$label", images, audio, false, emptyList())
                .collect { out.append(it) }
        } catch (e: Exception) {
            Log.e(TAG, "PARTIAL $label after ${SystemClock.elapsedRealtime() - started} ms (${out.length} chars): " +
                out.take(1200).toString().replace("\n", "\\n") + " … " + out.takeLast(600).toString().replace("\n", "\\n"))
            throw e
        }
        Log.i(TAG, "RESULT " + JSONObject().put("model", model.name).put("soc", Build.SOC_MODEL).put("test", label)
            .put("replyMs", SystemClock.elapsedRealtime() - started).put("reply", out.toString().take(1500)))
        return out.toString()
    }

    /** The catalog file where the app keeps it, resumed after drops, checked by size; gated files need -e hfToken. */
    private fun download(model: LLMModel) {
        if (model.modelFormat == "onnx") {
            // Multi-file models (ONNX graphs, tokenizer, configs) come through the app's own downloader.
            val started = SystemClock.elapsedRealtime()
            runBlocking {
                io.ktor.client.HttpClient(io.ktor.client.engine.android.Android).use { client ->
                    com.llmhub.llmhub.data.ModelDownloader(client, context).downloadModel(model).collect { }
                }
            }
            Log.i(TAG, "downloaded ${model.name} in ${SystemClock.elapsedRealtime() - started} ms")
            return
        }
        val target = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        if (target.length() == model.sizeBytes) return
        val token = args.getString("hfToken")
        val started = SystemClock.elapsedRealtime()
        repeat(6) { attempt ->
            try {
                val partial = File(target.path + ".part")
                val connection = URL(model.url).openConnection() as HttpURLConnection
                token?.let { connection.setRequestProperty("Authorization", "Bearer $it") }
                if (partial.length() > 0) connection.setRequestProperty("Range", "bytes=${partial.length()}-")
                connection.instanceFollowRedirects = true
                val code = connection.responseCode
                check(code in 200..299) { "HTTP $code for ${model.name}" }
                connection.inputStream.use { input -> java.io.FileOutputStream(partial, code == 206).use { input.copyTo(it, 1 shl 20) } }
                check(partial.length() == model.sizeBytes) { "${model.name}: ${partial.length()} of ${model.sizeBytes} bytes" }
                partial.renameTo(target)
                Log.i(TAG, "downloaded ${model.name} in ${SystemClock.elapsedRealtime() - started} ms")
                return
            } catch (e: java.io.IOException) {
                Log.w(TAG, "Download of ${model.name} failed (try ${attempt + 1}): $e")
                Thread.sleep(10_000)
            }
        }
        error("Could not download ${model.name}")
    }

    private companion object {
        const val TAG = "LiteRtLmChat"
    }
}
