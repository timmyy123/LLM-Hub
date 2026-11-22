package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check of a local LiteRT-LM conversion of Microsoft Aion 1.0 Instruct 2.3B
 * (litert-torch 0.10.0 export_hf, not published). Push the file first, then pass its path:
 *
 *   adb push aion.litertlm /data/local/tmp/aion.litertlm
 *   -e aionModel /data/local/tmp/aion.litertlm
 *
 * Runs the LiteRT-LM Engine and Conversation API the app uses (LiteRtLmInferenceService), so the chat template
 * and tokenizer come from the .litertlm, with benchmarking on: every reply logs LiteRT-LM's own prefill and
 * decode token counts and tokens per second. Each scenario is its own test (a native crash ends the
 * instrumentation process). Results go to logcat (tag AionLiteRtLm).
 */
@OptIn(ExperimentalApi::class)
class AionLiteRtLmDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val arguments = InstrumentationRegistry.getArguments()
    private lateinit var model: File

    @Before fun setUp() {
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${Build.SOC_MODEL}")
        model = File(arguments.getString("aionModel") ?: "/data/local/tmp/aion.litertlm")
    }

    /**
     * The app's own path: the catalog entry's pinned Hugging Face file (SHA-256 checked) where the downloader puts it,
     * loaded and asked through [LiteRtLmInferenceService] with the chat screen's "user: ...\nassistant:" prompt,
     * including a follow-up turn that needs the history. Needs network; deletes the download afterwards.
     */
    @Test fun catalogEntryChatsThroughTheApp() = runBlocking {
        val entry = ModelData.models.first { it.name == "Aion 1.0 Instruct 2.3B (LiteRT-LM)" }
        val file = File(context.filesDir, "models/${entry.localFileName()}")
        file.parentFile?.mkdirs()
        OnDeviceFixtures.download(file, entry.url, CATALOG_SHA256)
        val service = LiteRtLmInferenceService(context)
        try {
            service.setGenerationParameters(null, 1, 1.0f, 0.0f, null, false, null)
            assertTrue("Catalog entry must load on GPU", service.loadModel(entry, LlmInference.Backend.GPU, null))
            suspend fun chat(prompt: String): String {
                val out = StringBuilder()
                service.generateResponseStreamWithSession(prompt, entry, "aion-catalog-test", emptyList(), null, false, emptyList())
                    .collect { out.append(it) }
                Log.i(TAG, "APP_CHAT prompt=${prompt.replace("\n", "\\n")} reply=${out.toString().take(400)}")
                return out.toString()
            }
            val first = chat("user: What is the capital of France? Answer in one word.\nassistant:")
            assertTrue("Unexpected answer: $first", first.contains("Paris", ignoreCase = true))
            val second = chat("user: What is the capital of France? Answer in one word.\n\nassistant: $first\n\n" +
                "user: And what river runs through it? Answer in one word.\nassistant:")
            assertTrue("Unexpected follow-up: $second", second.contains("Seine", ignoreCase = true))
        } finally {
            service.unloadModel()
            file.delete()
        }
    }

    private fun requireModel() {
        assertTrue("Push the model to ${model.path} first", model.isFile)
        Log.i(TAG, "model=${model.name} ${model.length()} bytes")
    }

    @Test fun capitalCpu() = askCapital(cpu = true)

    @Test fun capitalGpu() = askCapital(cpu = false)

    /** A fact stated before ~2,800 tokens of filler: only Aion's 7 full-attention layers can still see it. */
    @Test fun longPromptRecallCpu() = recallPastSlidingWindow(cpu = true)

    @Test fun longPromptRecallGpu() = recallPastSlidingWindow(cpu = false)

    /** One of Aion's other languages, to see the tokenizer and template handle non-English text. */
    @Test fun germanCpu() = withEngine(cpu = true) { engine ->
        val reply = ask(engine, "CPU", "german", "Wie heißt die Hauptstadt von Deutschland? Antworte mit einem Wort.")
        assertTrue("Unexpected answer: $reply", reply.contains("Berlin", ignoreCase = true))
    }

    private fun askCapital(cpu: Boolean) = withEngine(cpu) { engine ->
        val backend = if (cpu) "CPU" else "GPU"
        val reply = ask(engine, backend, "capital", "What is the capital of France? Answer in one word.")
        assertTrue("Unexpected answer on $backend: $reply", reply.contains("Paris", ignoreCase = true))
        // A longer free-form reply, to read the output quality and measure decode speed.
        ask(engine, backend, "explain", "Explain in three sentences why the sky is blue.")
    }

    private fun recallPastSlidingWindow(cpu: Boolean) = withEngine(cpu) { engine ->
        val filler = (1..200).joinToString(" ") { "Log entry $it: the routine inspection of section $it found nothing unusual." }
        val prompt = "Remember this: the access code for the archive is BLUE-HERON-4417.\n\n$filler\n\n" +
            "What is the access code for the archive? Reply with the code only."
        val reply = ask(engine, if (cpu) "CPU" else "GPU", "recall", prompt)
        assertTrue("Did not recall the code: $reply", reply.contains("4417"))
    }

    private fun withEngine(cpu: Boolean, block: suspend (Engine) -> Unit) = runBlocking {
        requireModel()
        ExperimentalFlags.enableBenchmark = true
        val started = SystemClock.elapsedRealtime()
        val engine = Engine(EngineConfig(
            modelPath = model.absolutePath,
            backend = if (cpu) Backend.CPU() else Backend.GPU(),
            maxNumTokens = 4096,
            cacheDir = context.cacheDir.path,
        ))
        engine.initialize()
        Log.i(TAG, "loaded on ${if (cpu) "CPU" else "GPU"} in ${SystemClock.elapsedRealtime() - started} ms")
        try { block(engine) } finally { engine.close() }
    }

    /** Greedy reply through a fresh Conversation (as the app does), logging LiteRT-LM's benchmark numbers. */
    private suspend fun ask(engine: Engine, backend: String, label: String, prompt: String): String {
        val out = StringBuilder()
        engine.createConversation(ConversationConfig(samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0))).use { conv ->
            conv.sendMessageAsync(prompt).collect { msg ->
                out.append(msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text })
            }
            val b = conv.getBenchmarkInfo()
            val reply = out.toString()
            Log.i(TAG, "RESULT " + JSONObject().put("test", label).put("backend", backend)
                .put("prefillTokens", b.lastPrefillTokenCount).put("prefillTokPerSec", "%.1f".format(b.lastPrefillTokensPerSecond))
                .put("decodeTokens", b.lastDecodeTokenCount).put("decodeTokPerSec", "%.1f".format(b.lastDecodeTokensPerSecond))
                .put("timeToFirstTokenSec", "%.2f".format(b.timeToFirstTokenInSecond))
                .put("reply", reply.take(1500)))
            return reply
        }
    }

    private companion object {
        const val TAG = "AionLiteRtLm"
        // Hugging Face LFS oid of the catalog file at the pinned commit.
        const val CATALOG_SHA256 = "ac4a1d75a948123c7365a50cb75fb27d28da5aad670d22e8ff3f91714749bc1f"
    }
}
