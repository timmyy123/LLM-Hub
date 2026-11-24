package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check that the chat screen's common models load and answer, through the same
 * [UnifiedInferenceService] routing the chat uses (MediaPipe .task, LiteRT-LM Gemma 4, llama.cpp GGUF).
 * Written for a Play review from a Galaxy S26 Ultra ("the model won't load even for plain text chat").
 * Each phone downloads its own pinned files (SHA-256 checked) into the app's models folder. Each scenario is its
 * own test, because a native crash ends the instrumentation process. Results go to logcat (tag ChatModelLoad).
 */
class ChatModelLoadDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun gemma3_1bGpu() = check(GEMMA3, LlmInference.Backend.GPU)
    @Test fun gemma3_1bCpu() = check(GEMMA3, LlmInference.Backend.CPU)
    @Test fun gemma4E2bGpu() = check(GEMMA4, LlmInference.Backend.GPU)
    @Test fun gemma4E2bCpu() = check(GEMMA4, LlmInference.Backend.CPU)
    @Test fun lfmCpu() = check(LFM, LlmInference.Backend.CPU)
    @Test fun lfmGpu() = check(LFM, LlmInference.Backend.GPU)
    @Test fun lfmNpu() = check(LFM, LlmInference.Backend.GPU, "HTP0")

    private fun check(fixture: Fixture, backend: LlmInference.Backend, deviceId: String? = null) = runBlocking {
        val label = "${fixture.name} ${deviceId ?: backend.name}"
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${Build.SOC_MODEL} sdk=${Build.VERSION.SDK_INT} case=$label")
        val model: LLMModel = ModelData.models.first { it.name == fixture.name }
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        val downloadStart = SystemClock.elapsedRealtime()
        OnDeviceFixtures.download(file, fixture.url, fixture.sha256)
        Log.i(TAG, "RESULT $label download ok ${file.length()} bytes in ${(SystemClock.elapsedRealtime() - downloadStart) / 1000}s")
        val service = UnifiedInferenceService(context)
        try {
            val loadStart = SystemClock.elapsedRealtime()
            val loaded = try {
                service.loadModel(model, backend, deviceId)
            } catch (e: Throwable) {
                Log.e(TAG, "RESULT $label LOAD FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
                throw e
            }
            Log.i(TAG, "RESULT $label load=$loaded backend=${service.getCurrentlyLoadedBackend()} in ${SystemClock.elapsedRealtime() - loadStart} ms")
            assertTrue("$label did not load", loaded)
            val reply = service.generateResponse("What is the capital of France? Answer in one word.", model)
            Log.i(TAG, "RESULT $label reply=${reply.take(200).replace("\n", " ")}")
            assertTrue("$label unexpected reply: $reply", reply.contains("Paris", ignoreCase = true))
        } finally {
            service.unloadModel()
        }
    }

    private data class Fixture(val name: String, val url: String, val sha256: String)

    private companion object {
        const val TAG = "ChatModelLoad"
        val GEMMA3 = Fixture(
            "Gemma-3 1B (INT4, 2k)",
            "https://huggingface.co/timmyboy26/Gemma3-1B-IT/resolve/cd5aed3aa79603a6e3fabd7fc1e5ce19de10be98/Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task",
            "ddfaf1210d8b4d1b812b5fadb6652999e852c8be6dd9abe353b9213a25262c10",
        )
        val GEMMA4 = Fixture(
            "Gemma-4 E2B",
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/a4a831c060880f3733135ad22f10e0e9f758f45d/gemma-4-E2B-it.litertlm",
            "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c",
        )
        val LFM = Fixture(
            "LFM-2.5 1.2B Instruct (Q4_0)",
            "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_0.gguf",
            "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f",
        )
    }
}
