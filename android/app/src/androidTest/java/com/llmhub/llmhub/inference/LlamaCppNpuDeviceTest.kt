package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Physical-device check of GGUF generation on the Snapdragon Hexagon NPU across model sizes: under, near and over the
 * ~3.1 GB address window of the app's one NPU session, with a long prompt so the prompt is decoded in several batches.
 * Larger models run with the GPU layers setting lowered to what fits.
 * Models are fetched on the device from the catalog URLs and checked against their SHA-256. Each scenario is its own
 * test, because a native crash ends the instrumentation process. Results go to logcat (tag LlamaCppNpu).
 */
class LlamaCppNpuDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelsDir = File(context.filesDir, "models").apply { mkdirs() }

    @Before fun setUp() {
        DeviceInfo.init(context)
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT} soc=${DeviceInfo.getDeviceSoc()} " +
            "hexagon=${DeviceInfo.isLlamaCppHexagonSupported()}")
        assumeTrue("llama.cpp Hexagon needs a supported Snapdragon", DeviceInfo.isLlamaCppHexagonSupported())
    }

    /** 0.8 GB: well inside the NPU session's window. */
    @Test fun small() = runBlocking { run("Llama-3.2 1B (Q4_0)", LLAMA_1B_SHA256, gpuLayers = null) }

    /** 3.4 GB with the GPU layers setting lowered to what fits the window: those layers on the NPU, the rest on the CPU. */
    @Test fun nearWindowLayers() = runBlocking { run("Llama-3.2 3B (Q8_0)", LLAMA_3B_Q8_SHA256, gpuLayers = 13) }

    /** 5.1 GB with the GPU layers setting lowered to what fits. */
    @Test fun largeLayers() = runBlocking { run("Granite 4.2 8B (Q4_0)", GRANITE_8B_SHA256, gpuLayers = 14) }

    private suspend fun run(name: String, sha256: String, gpuLayers: Int?, contextWindow: Int = 4096) {
        val model = ModelData.models.first { it.name == name }
        fetch(model, sha256)
        val service = LlamaCppInferenceService(context)
        service.setGenerationParameters(
            maxTokens = 256, topK = 40, topP = 0.95f, temperature = 0.1f,
            nGpuLayers = gpuLayers, enableThinking = false, contextWindow = contextWindow,
        )
        val loadStarted = SystemClock.elapsedRealtime()
        assertTrue("$name must load on the NPU", service.loadModel(model, LlmInference.Backend.GPU, "HTP0"))
        val loadMs = SystemClock.elapsedRealtime() - loadStarted
        assertEquals("$name fell back from the NPU", "HTP0", service.getCurrentlyLoadedDeviceId())
        try {
            val started = SystemClock.elapsedRealtime()
            // About 1,500 tokens, so the prompt is decoded in several 512-token batches like a chat with history.
            val prompt = buildString {
                repeat(40) { append("Note $it: the quick brown fox jumps over the lazy dog near the river bank. ") }
                append("\n\nIgnore the notes above. What is the capital of France? Answer in one word.")
            }
            val reply = service.generateResponse(prompt, model)
            val result = JSONObject().put("model", name).put("gpuLayers", gpuLayers ?: "all").put("loadMs", loadMs)
                .put("generateMs", SystemClock.elapsedRealtime() - started).put("tokPerSec", service.getLastDecodeSpeedTokPerSec())
                .put("deviceId", service.getCurrentlyLoadedDeviceId()).put("reply", reply.take(600))
            Log.i(TAG, "RESULT $result")
            assertTrue("Unexpected answer from $name on the NPU: $reply", reply.contains("Paris", ignoreCase = true))
        } finally {
            service.unloadModel()
        }
    }

    private fun fetch(model: LLMModel, sha256: String) =
        OnDeviceFixtures.download(File(modelsDir, model.localFileName()), model.url, sha256)

    private companion object {
        const val TAG = "LlamaCppNpu"
        // Hugging Face LFS oids (SHA-256) of the catalog files.
        const val LLAMA_1B_SHA256 = "fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8"
        const val LLAMA_3B_Q8_SHA256 = "b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921"
        const val GRANITE_8B_SHA256 = "82af51c80f8c2913430aa020f97bd3d4708c5fc49c5bfbff09ac012f3c2132d2"
    }
}
