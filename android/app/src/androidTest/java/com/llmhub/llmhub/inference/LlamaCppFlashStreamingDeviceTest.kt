package com.llmhub.llmhub.inference

import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.GgufLayerLimits
import com.llmhub.llmhub.data.ModelData
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Opt-in physical-device check of MoE flash streaming (llama_stream_jni.cpp) with LFM2.5-24B-A2B Q4_0 (13.6 GB).
 *
 * It uses the copy the app itself downloaded, read in place from its models folder, and skips when that download
 * is not there: it never downloads or copies the model. It loads through [LlamaCppInferenceService] as the app does,
 * so the model's saved flash streaming setting (on unless the user turned it off) decides the runtime, and logs a
 * `RESULT` line with the speeds.
 */
class LlamaCppFlashStreamingDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val model = ModelData.models.first { it.name == "LFM2.5-24B-A2B (Q4_0)" }

    @Test fun lfm25StreamsOnCpu() = runBlocking {
        DeviceInfo.init(context)
        assumeTrue("Download ${model.name} in the app first", GgufLayerLimits.isMoe(context, model))
        val service = LlamaCppInferenceService(context)
        service.setGenerationParameters(
            maxTokens = 256, topK = 40, topP = 0.95f, temperature = 0.1f,
            nGpuLayers = null, enableThinking = false, contextWindow = 4096,
        )
        try {
            var started = SystemClock.elapsedRealtime()
            assertTrue("${model.name} must load", service.loadModel(model, LlmInference.Backend.CPU, null))
            val loadMs = SystemClock.elapsedRealtime() - started
            assertTrue("${model.name} must load with flash streaming", service.isFlashStreamingLoaded())
            assertEquals(LlmInference.Backend.CPU, service.getCurrentlyLoadedBackend())
            started = SystemClock.elapsedRealtime()
            val reply = service.generateResponse("What is the capital of France? Answer in one word.", model)
            val result = JSONObject().put("model", model.name).put("loadMs", loadMs)
                .put("generateMs", SystemClock.elapsedRealtime() - started)
                .put("tokPerSec", service.getLastDecodeSpeedTokPerSec()).put("reply", reply.take(600))
            Log.i(TAG, "RESULT $result")
            assertTrue("Unexpected answer: $reply", reply.contains("Paris", ignoreCase = true))
            started = SystemClock.elapsedRealtime()
            val paragraph = service.generateResponse("Write three sentences about the history of the Eiffel Tower.", model)
            result.put("paragraphMs", SystemClock.elapsedRealtime() - started)
                .put("paragraphTokPerSec", service.getLastDecodeSpeedTokPerSec()).put("paragraph", paragraph.take(1200))
            Log.i(TAG, "RESULT $result")
            assertTrue("Unexpected paragraph: $paragraph", paragraph.contains("Eiffel", ignoreCase = true))
        } finally {
            service.unloadModel()
        }
    }

    private companion object {
        const val TAG = "LlamaCppFlashStream"
    }
}
