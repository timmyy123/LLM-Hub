package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check that a GGUF model runs on the GPU through [LlamaCppInferenceService] with the app's
 * backend rules: OpenCL on Snapdragon SoCs where llama.cpp Vulkan crashes (e.g. Galaxy S25), Vulkan elsewhere.
 * Pass `-e downloadFixtures 1` to fetch LFM-2.5 1.2B Q4_0 on the phone, and `-e flashAttention -1|0|1` to override
 * llama.cpp's flash attention choice; results go to logcat.
 */
class LlamaCppGpuDeviceTest {
    @Test fun ggufGeneratesOnGpu() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = ModelData.models.first { it.name == "LFM-2.5 1.2B Instruct (Q4_0)" }
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        if (InstrumentationRegistry.getArguments().getString("downloadFixtures") == "1") {
            OnDeviceFixtures.download(
                file,
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_0.gguf",
                "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
            )
        }
        assertTrue("Missing ${file.name}", file.isFile)
        val report = JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL).put("android", Build.VERSION.RELEASE)
            .put("vulkanBlocked", DeviceInfo.isLlamaCppVulkanBlocked())
            .put("usesVulkan", ThemePreferences(context).ggufUseVulkan.first())

        val service = LlamaCppInferenceService(context)
        // -e flashAttention -1|0|1 overrides llama.cpp's flash attention choice.
        InstrumentationRegistry.getArguments().getString("flashAttention")?.toIntOrNull()?.let { service.flashAttention = it }
        report.put("flashAttention", service.flashAttention)
        try {
            var started = SystemClock.elapsedRealtime()
            val loaded = service.loadModel(model, LlmInference.Backend.GPU, null)
            report.put("loaded", loaded).put("loadMs", SystemClock.elapsedRealtime() - started)
            assertTrue("The GGUF model must load", loaded)
            started = SystemClock.elapsedRealtime()
            val reply = service.generateResponse("What is the capital of France? Answer in one word.", model)
            report.put("generateMs", SystemClock.elapsedRealtime() - started).put("reply", reply.take(200))
            Log.i(TAG, "RESULT $report")
            assertTrue("The model must answer", reply.isNotBlank())
            assertTrue("Unexpected answer: $reply", reply.contains("Paris", ignoreCase = true))
        } finally {
            service.unloadModel()
            file.delete()
        }
    }

    private companion object { const val TAG = "LlamaCppGpuDeviceTest" }
}
