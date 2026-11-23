package com.llmhub.llmhub.inference

import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.GgufLayerLimits
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.flow.first
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
    // -e model_name NAME: another MoE GGUF from the catalog (default LFM2.5-24B-A2B Q4_0).
    private val model = (InstrumentationRegistry.getArguments().getString("model_name") ?: "LFM2.5-24B-A2B (Q4_0)")
        .let { name -> ModelData.models.first { it.name == name } }

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

    /**
     * Each backend selection against CPU streaming, on a ~1,200-token prompt: the prompt on the accelerator, decode on
     * the CPU. `-e modes` picks them (comma-separated, default all that this device has): npu, opencl, vulkan, cpu.
     * Prompt and decode speeds per turn are in the runtime's "Turn:" log line; this records first-token time, decode
     * speed and the runtime that loaded.
     */
    @Test fun lfm25Accelerators() = runBlocking {
        DeviceInfo.init(context)
        // -e model_sha256 HEX: a phone without the app's download (a device farm) downloads it into the app's models
        // folder itself, from the catalog URL, checked against this hash, where the app keeps its own downloads.
        InstrumentationRegistry.getArguments().getString("model_sha256")?.let { sha ->
            val target = java.io.File(java.io.File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
            if (!target.isFile) com.llmhub.llmhub.imagegen.OnDeviceFixtures.download(target, model.url, sha)
        }
        assumeTrue("Download ${model.name} in the app first", GgufLayerLimits.isMoe(context, model))
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
        val args = InstrumentationRegistry.getArguments()
        // -e hex_vmem 1536M: ggml-hexagon's budget for its NPU buffer mappings (GGML_HEXAGON_VMEM), read when the
        // backend loads, so it is set before the first load in this process.
        args.getString("hex_vmem")?.let { android.system.Os.setenv("GGML_HEXAGON_VMEM", it, true) }
        // -e hex_env "VERBOSE=1;OPBATCH=64": other ggml-hexagon switches (GGML_HEXAGON_<NAME>) for diagnosing the NPU.
        args.getString("hex_env")?.split(';')?.filter { '=' in it }?.forEach {
            android.system.Os.setenv("GGML_HEXAGON_" + it.substringBefore('=').trim(), it.substringAfter('=').trim(), true)
        }
        data class Mode(val label: String, val backend: LlmInference.Backend, val device: String?, val vulkan: Boolean,
                        val gpuLayers: Int?, val runtime: String)
        val snapdragon = DeviceInfo.isLlamaCppHexagonSupported()
        val all = buildList {
            if (snapdragon) {
                add(Mode("npu", LlmInference.Backend.GPU, "HTP0", false, null, "NPU flash streaming"))
                // Adreno OpenCL cannot read a streamed prompt: the GPU selection streams on the CPU.
                add(Mode("opencl", LlmInference.Backend.GPU, null, false, null, "CPU flash streaming"))
            }
            // The app never offers Vulkan where it is blocked (every Snapdragon, drivers older than 1.2).
            if (!DeviceInfo.isLlamaCppVulkanBlocked()) {
                add(Mode("vulkan", LlmInference.Backend.GPU, null, true, null, "Vulkan flash streaming"))
            }
            add(Mode("cpu", LlmInference.Backend.CPU, null, false, null, "CPU flash streaming"))
        }
        // -e thinking true: the model thinks first (its thinking switch); -e max_tokens N: room for it (default 320).
        val thinking = args.getString("thinking") == "true"
        val maxTokens = args.getString("max_tokens")?.toIntOrNull() ?: 320
        val picked = args.getString("modes")?.split(',')?.map { it.trim() }
        val modes = if (picked == null) all else all.filter { it.label in picked }
        val prefs = com.llmhub.llmhub.data.ThemePreferences(context)
        val vulkanBefore = prefs.ggufUseVulkan.first()
        val result = JSONObject().put("model", model.name)
        val failures = mutableListOf<String>()
        try {
            for (mode in modes) {
                prefs.setGgufUseVulkan(mode.vulkan)
                val service = LlamaCppInferenceService(context)
                service.setGenerationParameters(
                    // LFM2.5 thinks before it answers; 320 tokens leave room for both.
                    maxTokens = maxTokens, topK = 40, topP = 0.95f, temperature = 0.1f,
                    nGpuLayers = mode.gpuLayers, enableThinking = thinking, contextWindow = 4096,
                )
                val entry = JSONObject()
                result.put(mode.label, entry)
                try {
                    var started = SystemClock.elapsedRealtime()
                    val loaded = service.loadModel(model, mode.backend, mode.device)
                    entry.put("loadMs", SystemClock.elapsedRealtime() - started).put("runtime", service.activeRuntimeName())
                    if (!loaded || service.activeRuntimeName() != mode.runtime) {
                        failures += "${mode.label}: loaded ${service.activeRuntimeName()}"
                        continue
                    }
                    started = SystemClock.elapsedRealtime()
                    var firstMs = -1L
                    val reply = StringBuilder()
                    service.generateResponseStream(prompt, model).collect {
                        if (firstMs < 0 && it.isNotEmpty()) firstMs = SystemClock.elapsedRealtime() - started
                        reply.append(it)
                    }
                    entry.put("firstTokenMs", firstMs).put("totalMs", SystemClock.elapsedRealtime() - started)
                        .put("decodeTokPerSec", service.getLastDecodeSpeedTokPerSec()).put("reply", reply.toString().takeLast(160))
                        .put("replyStart", reply.toString().take(200))
                    if (!reply.contains("1889")) failures += "${mode.label}: wrong answer"
                } catch (t: Throwable) {
                    entry.put("error", "${t.javaClass.simpleName}: ${t.message}")
                    failures += "${mode.label}: ${t.message}"
                } finally {
                    service.unloadModel()
                    Log.i(TAG, "RESULT $result")
                }
            }
        } finally {
            prefs.setGgufUseVulkan(vulkanBefore)
        }
        assertTrue("Failed: $failures", failures.isEmpty())
    }

    private companion object {
        const val TAG = "LlamaCppFlashStream"
    }
}
