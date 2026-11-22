package com.llmhub.llmhub.inference

import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelRequirements
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check of the Aion architecture patch (microsoft/Aion-1.0-Instruct-2.3B, `aion` in the
 * vendored llama.cpp and the rebuilt Snapdragon libllama.so) with a locally converted Q4_0 GGUF. The model is not in
 * the catalog yet, so the GGUF must already be in the app's models folder:
 *
 *   adb shell "cat /data/local/tmp/Aion-1.0-Instruct-2.3B-Q4_0.gguf | run-as <package> sh -c 'cat > files/models/Aion-1.0-Instruct-2.3B-Q4_0.gguf'"
 *
 * Each scenario loads through [LlamaCppInferenceService] as the app does, asserts the backend it really ran on and
 * logs a `RESULT` line with the decode speed. Run them one at a time with `-e class ...LlamaCppAionDeviceTest#<name>`.
 */
class LlamaCppAionDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val model = LLMModel(
        name = "Aion 1.0 Instruct 2.3B (Q4_0)",
        description = "Microsoft Aion 1.0 Instruct 2.3B, Q4_0 (local test build).",
        url = "https://example.invalid/Aion-1.0-Instruct-2.3B-Q4_0.gguf",
        category = "text",
        sizeBytes = 1401770144L,
        source = "Microsoft",
        supportsVision = false,
        supportsGpu = true,
        requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
        contextWindowSize = 32768,
        modelFormat = "gguf",
    )

    @Before fun setUp() {
        DeviceInfo.init(context)
        val file = File(File(context.filesDir, "models"), model.localFileName())
        assumeTrue("Copy the Aion GGUF to ${file.absolutePath} first", file.isFile)
    }

    @Test fun textCpu() = runBlocking { askCapital(LlmInference.Backend.CPU, null) }

    @Test fun textGpu() = runBlocking { askCapital(LlmInference.Backend.GPU, null) }

    @Test fun textNpu() = runBlocking {
        assumeTrue("llama.cpp Hexagon needs a supported Snapdragon", DeviceInfo.isLlamaCppHexagonSupported())
        askCapital(LlmInference.Backend.GPU, "HTP0")
    }

    private suspend fun askCapital(backend: LlmInference.Backend, deviceId: String?) {
        val service = LlamaCppInferenceService(context)
        service.setGenerationParameters(
            maxTokens = 512, topK = 40, topP = 0.95f, temperature = 0.1f,
            nGpuLayers = null, enableThinking = false, contextWindow = 8192,
        )
        try {
            var started = SystemClock.elapsedRealtime()
            assertTrue("${model.name} must load on $backend/$deviceId", service.loadModel(model, backend, deviceId))
            val loadMs = SystemClock.elapsedRealtime() - started
            assertEquals("Fell back from $backend/$deviceId", backend, service.getCurrentlyLoadedBackend())
            assertEquals(deviceId, service.getCurrentlyLoadedDeviceId())
            started = SystemClock.elapsedRealtime()
            val reply = service.generateResponse("What is the capital of France? Answer in one word.", model)
            val result = JSONObject().put("model", model.name).put("backend", backend.name).put("deviceId", deviceId)
                .put("loadMs", loadMs).put("generateMs", SystemClock.elapsedRealtime() - started)
                .put("tokPerSec", service.getLastDecodeSpeedTokPerSec()).put("reply", reply.take(600))
            assertTrue("Unexpected answer on $backend/$deviceId: $reply", reply.contains("Paris", ignoreCase = true))
            // A longer reply for a decode speed that is not dominated by the first token.
            started = SystemClock.elapsedRealtime()
            val paragraph = service.generateResponse("Write one paragraph of about 150 words about the history of the Eiffel Tower.", model)
            result.put("paragraphMs", SystemClock.elapsedRealtime() - started)
                .put("paragraphTokPerSec", service.getLastDecodeSpeedTokPerSec()).put("paragraph", paragraph.take(1200))
            Log.i(TAG, "RESULT $result")
            assertTrue("Unexpected paragraph on $backend/$deviceId: $paragraph", paragraph.contains("Eiffel", ignoreCase = true))
        } finally {
            service.unloadModel()
        }
    }

    private companion object { const val TAG = "LlamaCppAion" }
}
