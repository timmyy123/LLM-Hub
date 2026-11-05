package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Play vitals 4.6.3/4.6.4 on new Snapdragon chips, each scenario run with every layer on the accelerator and no CPU
 * fallback allowed:
 * - 8 Gen 3 NPU (Xiaomi 14 Ultra): a model larger than one Hexagon session's address window aborted on the first
 *   token (`fastrpc_mmap failed`).
 * - 8s Gen 4 GPU (Poco F7): OpenCL refused `kernel_mul_mat_f32_f32`, which runs the F32 expert router of MoE models.
 * Models download on the phone from the catalog URLs, pinned by SHA-256. One test per scenario, because a native abort
 * ends the instrumentation process. Results and the ggml error lines go to logcat (tag LlamaCppChipCrash).
 */
class LlamaCppChipCrashDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelsDir = File(context.filesDir, "models").apply { mkdirs() }

    @Before fun setUp() {
        DeviceInfo.init(context)
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT} soc=${DeviceInfo.getDeviceSoc()} " +
            "snapdragon=${DeviceInfo.isSnapdragon()} hexagon=${DeviceInfo.isLlamaCppHexagonSupported()}")
    }

    /** 5.1 GB, every layer on the NPU: larger than one session's window. */
    @Test fun npuLargeModelAllLayers() = runBlocking {
        selected("npu")
        assumeTrue("llama.cpp Hexagon needs a supported Snapdragon", DeviceInfo.isLlamaCppHexagonSupported())
        run("Granite 4.2 8B (Q4_0)", GRANITE_8B_URL, GRANITE_8B_SHA256, deviceId = "HTP0")
    }

    /** 3.4 GB, every layer on the NPU: needs two sessions. Uses the file already in files/models when its SHA-256 matches. */
    @Test fun npuTwoSessions() = runBlocking {
        selected("npu2")
        assumeTrue("llama.cpp Hexagon needs a supported Snapdragon", DeviceInfo.isLlamaCppHexagonSupported())
        run("Llama-3.2 3B (Q8_0)", LLAMA_3B_Q8_URL, LLAMA_3B_Q8_SHA256, deviceId = "HTP0")
    }

    /** 4.8 GB MoE with F32 router weights on the OpenCL GPU. */
    @Test fun gpuMoeLfm8b() = runBlocking {
        selected("gpu")
        assumeTrue("OpenCL GGUF runtime is for Snapdragon", DeviceInfo.isSnapdragon())
        run("LFM2.5-8B-A1B (Q4_0)", LFM_8B_URL, LFM_8B_SHA256, deviceId = null)
    }

    /** 2.6 GB hybrid MoE with F32 router weights on the OpenCL GPU. */
    @Test fun gpuMoeGraniteTiny() = runBlocking {
        selected("gpu")
        assumeTrue("OpenCL GGUF runtime is for Snapdragon", DeviceInfo.isSnapdragon())
        run("Granite 4.0 H-Tiny (Q2_K)", GRANITE_TINY_URL, GRANITE_TINY_SHA256, deviceId = null)
    }

    /** `-e chipScenario npu|gpu` runs one group only, so each phone downloads only the models it needs. */
    private fun selected(group: String) {
        val wanted = InstrumentationRegistry.getArguments().getString("chipScenario")
        assumeTrue("chipScenario=$wanted skips $group", wanted == null || wanted == group)
    }

    private suspend fun run(name: String, url: String, sha256: String, deviceId: String?) {
        val model = ModelData.models.first { it.name == name }
        val file = File(modelsDir, model.localFileName())
        val downloadStarted = SystemClock.elapsedRealtime()
        OnDeviceFixtures.download(file, url, sha256)
        Log.i(TAG, "FETCHED $name in ${SystemClock.elapsedRealtime() - downloadStarted} ms")
        val service = LlamaCppInferenceService(context)
        service.setGenerationParameters(
            maxTokens = 128, topK = 40, topP = 0.95f, temperature = 0.1f,
            nGpuLayers = null, enableThinking = false, contextWindow = 4096,
        )
        val loadStarted = SystemClock.elapsedRealtime()
        assertTrue("$name must load on the accelerator", service.loadModel(model, LlmInference.Backend.GPU, deviceId))
        val loadMs = SystemClock.elapsedRealtime() - loadStarted
        assertEquals("$name fell back to the CPU", LlmInference.Backend.GPU, service.getCurrentlyLoadedBackend())
        assertEquals("$name changed accelerator", deviceId, service.getCurrentlyLoadedDeviceId())
        try {
            val started = SystemClock.elapsedRealtime()
            // About 1,500 tokens, so the prompt is decoded in several batches like a chat with history.
            val prompt = buildString {
                repeat(40) { append("Note $it: the quick brown fox jumps over the lazy dog near the river bank. ") }
                append("\n\nIgnore the notes above. What is the capital of France? Answer in one word.")
            }
            val reply = service.generateResponse(prompt, model)
            val result = JSONObject().put("model", name).put("device", Build.MODEL).put("soc", Build.SOC_MODEL)
                .put("accelerator", deviceId ?: "GPU").put("loadMs", loadMs)
                .put("generateMs", SystemClock.elapsedRealtime() - started)
                .put("tokPerSec", service.getLastDecodeSpeedTokPerSec()).put("reply", reply.take(300))
            Log.i(TAG, "RESULT $result")
            assertTrue("Unexpected answer from $name: $reply", reply.contains("Paris", ignoreCase = true))
        } finally {
            service.unloadModel()
        }
    }

    private companion object {
        const val TAG = "LlamaCppChipCrash"
        const val GRANITE_8B_URL =
            "https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF/resolve/93f3f6a8938ee922b784cf4e5b4203cd3428df8f/granite-4.2-8b-Q4_0.gguf"
        const val GRANITE_8B_SHA256 = "82af51c80f8c2913430aa020f97bd3d4708c5fc49c5bfbff09ac012f3c2132d2"
        const val LLAMA_3B_Q8_URL =
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf"
        const val LLAMA_3B_Q8_SHA256 = "b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921"
        const val LFM_8B_URL =
            "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0/LFM2.5-8B-A1B-Q4_0.gguf"
        const val LFM_8B_SHA256 = "48ed1465d761311b2fd57b7fb46cf969a20b3a8281945b04e10f52bd1609e715"
        const val GRANITE_TINY_URL =
            "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/08d5a8a9741dd5c1a95d2d39e25253226aa1464e/granite-4.0-h-tiny-Q2_K.gguf"
        const val GRANITE_TINY_SHA256 = "c88fe8f38461636734ec6facab9a4a2977d122b6d85b925d2596271feebacefe"
    }
}
