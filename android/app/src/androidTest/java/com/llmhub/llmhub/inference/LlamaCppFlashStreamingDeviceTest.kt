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

    /** The prompt on the Hexagon NPU (NPU selected), against CPU streaming, on a ~1,000-token prompt. */
    @Test fun lfm25NpuPrefillAgainstCpu() = runBlocking {
        DeviceInfo.init(context)
        assumeTrue("Download ${model.name} in the app first", GgufLayerLimits.isMoe(context, model))
        assumeTrue("llama.cpp Hexagon needs a supported Snapdragon", DeviceInfo.isLlamaCppHexagonSupported())
        val facts = listOf(
            "The Eiffel Tower was built for the 1889 World's Fair in Paris, which marked the centenary of the French Revolution.",
            "Gustave Eiffel's company designed and built it, with the engineers Maurice Koechlin and Emile Nouguier drawing the first plans.",
            "Construction began in January 1887 and the tower opened on 31 March 1889 after about two years, two months and five days of work.",
            "It is made of about 18,000 wrought iron parts joined by roughly 2.5 million rivets.",
            "At 300 metres it was the tallest man-made structure in the world until the Chrysler Building was finished in 1930.",
            "Many artists signed a protest against it in 1887, calling it useless and monstrous.",
            "The city's permit was for twenty years, and the tower was saved partly because it proved useful for radio transmission.",
            "It is repainted about every seven years, and each repainting uses around sixty tonnes of paint.",
        )
        val document = (1..5).joinToString("\n\n") { round -> facts.joinToString(" ") { "Part $round. $it" } }
        val prompt = "$document\n\nUsing only the text above, in which year did the Eiffel Tower open? Answer in one short sentence."
        // -e hex_vmem 1536M: ggml-hexagon's budget for its NPU buffer mappings (GGML_HEXAGON_VMEM), read when the
        // backend loads, so it is set before the first load in this process.
        val hexVmem = InstrumentationRegistry.getArguments().getString("hex_vmem")
        if (hexVmem != null) android.system.Os.setenv("GGML_HEXAGON_VMEM", hexVmem, true)
        val result = JSONObject().put("model", model.name).put("hexVmem", hexVmem ?: "default")
        val runs = listOf(Triple("npu", LlmInference.Backend.GPU, "HTP0"), Triple("cpu", LlmInference.Backend.CPU, null))
        val npuOnly = InstrumentationRegistry.getArguments().getString("npu_only") == "true"
        for ((label, backend, device) in if (npuOnly) runs.take(1) else runs) {
            val service = LlamaCppInferenceService(context)
            service.setGenerationParameters(
                // LFM2.5 thinks before it answers; 320 tokens leave room for both.
                maxTokens = 320, topK = 40, topP = 0.95f, temperature = 0.1f,
                nGpuLayers = null, enableThinking = false, contextWindow = 4096,
            )
            try {
                var started = SystemClock.elapsedRealtime()
                assertTrue("${model.name} must load ($label)", service.loadModel(model, backend, device))
                val entry = JSONObject().put("loadMs", SystemClock.elapsedRealtime() - started)
                    .put("npuStreaming", service.isFlashStreamingNpuLoaded()).put("streaming", service.isFlashStreamingLoaded())
                started = SystemClock.elapsedRealtime()
                var firstMs = -1L
                val reply = StringBuilder()
                service.generateResponseStream(prompt, model).collect {
                    if (firstMs < 0 && it.isNotEmpty()) firstMs = SystemClock.elapsedRealtime() - started
                    reply.append(it)
                }
                entry.put("firstTokenMs", firstMs).put("totalMs", SystemClock.elapsedRealtime() - started)
                    .put("decodeTokPerSec", service.getLastDecodeSpeedTokPerSec()).put("reply", reply.toString().takeLast(160))
                result.put(label, entry)
                Log.i(TAG, "RESULT $result")
                if (label == "npu") assertTrue("NPU flash streaming did not load", service.isFlashStreamingNpuLoaded())
                assertTrue("Unexpected answer ($label): $reply", reply.contains("1889"))
            } finally {
                service.unloadModel()
            }
        }
    }

    private companion object {
        const val TAG = "LlamaCppFlashStream"
    }
}
