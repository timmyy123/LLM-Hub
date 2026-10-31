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
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check of llama.cpp b12020 with LiquidAI LFM3 (lfm3 architecture, lfm3vl projector):
 * LFM3 1.7B Q4_0 (QAD) and LFM3 VL 2B Q4_0 (QAD) with its projector are fetched on the device from the
 * pinned catalog URLs and checked against their SHA-256 (the Hugging Face LFS oid). Every scenario loads
 * through [LlamaCppInferenceService] as the app does and asserts the backend it really ran on (a failed
 * accelerator load silently retries on CPU). Each scenario is its own test, because a native crash ends the
 * instrumentation process; run them one at a time with `-e class ...LlamaCppLfm3DeviceTest#<name>`.
 * `-e lfm3Projector Q8_0` uses the Q8_0 projector instead of BF16. Results go to logcat (tag LlamaCppLfm3).
 */
class LlamaCppLfm3DeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val arguments = InstrumentationRegistry.getArguments()
    private val modelsDir = File(context.filesDir, "models").apply { mkdirs() }
    private val textModel = ModelData.models.first { it.name == "LFM3 1.7B (Q4_0)" }
    private val visionModel = ModelData.models.first { it.name == "LFM3 VL 2B (Q4_0)" }

    @Before fun setUp() {
        DeviceInfo.init(context)
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${DeviceInfo.getDeviceSoc()} snapdragon=${DeviceInfo.isSnapdragon()} " +
            "hexagon=${DeviceInfo.isLlamaCppHexagonSupported()}")
    }

    /** CPU runtime (libllmhub_llama_cpu.so, source build): reasons in a think block, then answers. */
    @Test fun textCpu() = runBlocking { askCapital(Accelerator.CPU, thinking = true) }

    /** Thinking off: the prompt ends in an empty think block, so the reply is the answer alone. */
    @Test fun textCpuWithoutThinking() = runBlocking { askCapital(Accelerator.CPU, thinking = false) }

    /** Snapdragon Adreno GPU through the release runtime and the patched libggml-opencl.so. */
    @Test fun textGpuOpenCl() = runBlocking { askCapital(Accelerator.OPENCL, thinking = true) }

    /** Snapdragon Hexagon NPU (HTP0) through the release runtime and its skels. */
    @Test fun textNpu() = runBlocking { askCapital(Accelerator.NPU, thinking = true) }

    @Test fun visionCpu() = runBlocking { describeImage(Accelerator.CPU) }

    @Test fun visionGpuOpenCl() = runBlocking { describeImage(Accelerator.OPENCL) }

    @Test fun visionNpu() = runBlocking { describeImage(Accelerator.NPU) }

    private enum class Accelerator(val backend: LlmInference.Backend, val deviceId: String?) {
        CPU(LlmInference.Backend.CPU, null),
        OPENCL(LlmInference.Backend.GPU, null),
        NPU(LlmInference.Backend.GPU, "HTP0"),
    }

    private suspend fun askCapital(accelerator: Accelerator, thinking: Boolean) {
        requireAccelerator(accelerator)
        fetch(textModel, TEXT_SHA256)
        val service = loadOn(textModel, accelerator, thinking)
        try {
            val started = SystemClock.elapsedRealtime()
            val reply = service.generateResponse("What is the capital of France? Answer in one word.", textModel)
            report(service, textModel, accelerator, reply, SystemClock.elapsedRealtime() - started)
            val answer = answerOf(reply, thinking)
            assertTrue("Unexpected answer on $accelerator: $reply", answer.contains("Paris", ignoreCase = true))
        } finally {
            service.unloadModel()
        }
    }

    private suspend fun describeImage(accelerator: Accelerator) {
        requireAccelerator(accelerator)
        fetch(visionModel, VISION_SHA256)
        val useQ8Projector = arguments.getString("lfm3Projector").equals("Q8_0", ignoreCase = true)
        val projector = ModelData.models.first { it.name == "LFM3 VL 2B (Vision Projector, ${if (useQ8Projector) "Q8_0" else "BF16"})" }
        val otherProjector = ModelData.models.first { it.name == "LFM3 VL 2B (Vision Projector, ${if (useQ8Projector) "BF16" else "Q8_0"})" }
        // The service prefers BF16 when both are present; keep only the projector under test.
        File(modelsDir, otherProjector.localFileName()).delete()
        fetch(projector, if (useQ8Projector) PROJECTOR_Q8_SHA256 else PROJECTOR_BF16_SHA256)

        val service = loadOn(visionModel, accelerator, thinking = true)
        val image = redCircle()
        try {
            assertFalse("The LFM3 VL projector must load on $accelerator", service.isVisionCurrentlyDisabled())
            val started = SystemClock.elapsedRealtime()
            val output = StringBuilder()
            service.generateResponseStreamWithSession(
                prompt = "What color is the circle in this image? Answer in one word.",
                model = visionModel,
                chatId = "lfm3-vision-test",
                images = listOf(image),
                audioData = null,
                webSearchEnabled = false,
                imagePaths = emptyList(),
            ).collect { output.append(it) }
            val reply = output.toString()
            report(service, visionModel, accelerator, reply, SystemClock.elapsedRealtime() - started, projector.localFileName())
            val answer = answerOf(reply, thinking = true)
            assertTrue("Unexpected image answer on $accelerator: $reply", answer.contains("red", ignoreCase = true))
        } finally {
            image.recycle()
            service.unloadModel()
        }
    }

    private fun requireAccelerator(accelerator: Accelerator) {
        when (accelerator) {
            Accelerator.CPU -> Unit
            Accelerator.OPENCL -> assumeTrue("Snapdragon OpenCL runtime needs a Snapdragon SoC", DeviceInfo.isSnapdragon())
            Accelerator.NPU -> assumeTrue("llama.cpp Hexagon needs a supported Snapdragon", DeviceInfo.isLlamaCppHexagonSupported())
        }
    }

    private fun fetch(model: LLMModel, sha256: String) =
        OnDeviceFixtures.download(File(modelsDir, model.localFileName()), model.url, sha256)

    private suspend fun loadOn(model: LLMModel, accelerator: Accelerator, thinking: Boolean): LlamaCppInferenceService {
        if (accelerator == Accelerator.OPENCL) {
            // Snapdragon never uses the Vulkan runtime for GGUF, so GPU means OpenCL here.
            assertFalse("GPU must select OpenCL on Snapdragon", ThemePreferences(context).ggufUseVulkan.first())
        }
        val service = LlamaCppInferenceService(context)
        // A chat-sized context (LFM3 1.7B advertises 256K) and the model card's sampling.
        service.setGenerationParameters(
            maxTokens = 2048, topK = 50, topP = 1.0f, temperature = 0.1f,
            nGpuLayers = null, enableThinking = thinking, contextWindow = 8192,
        )
        val started = SystemClock.elapsedRealtime()
        assertTrue("${model.name} must load on $accelerator", service.loadModel(model, accelerator.backend, accelerator.deviceId))
        Log.i(TAG, "${model.name} loaded in ${SystemClock.elapsedRealtime() - started} ms")
        assertEquals("Fell back from $accelerator", accelerator.backend, service.getCurrentlyLoadedBackend())
        if (accelerator.deviceId == null) assertNull(service.getCurrentlyLoadedDeviceId())
        else assertEquals(accelerator.deviceId, service.getCurrentlyLoadedDeviceId())
        return service
    }

    /** The answer after the reasoning block; fails when reasoning never finished within the token budget. */
    private fun answerOf(reply: String, thinking: Boolean): String {
        if (!thinking) {
            assertFalse("Thinking was off but the reply has a think block: $reply", reply.contains(THINK) || reply.contains(END_THINK))
            return reply
        }
        assertTrue("Reasoning did not finish: ${reply.takeLast(400)}", reply.contains(END_THINK))
        return reply.substringAfterLast(END_THINK)
    }

    private fun report(
        service: LlamaCppInferenceService, model: LLMModel, accelerator: Accelerator, reply: String, generateMs: Long,
        projector: String? = null,
    ) {
        val result = JSONObject().put("model", model.name).put("accelerator", accelerator.name)
            .put("backend", service.getCurrentlyLoadedBackend()?.name).put("deviceId", service.getCurrentlyLoadedDeviceId())
            .put("projector", projector).put("context", service.getLoadedContextSize())
            .put("generateMs", generateMs).put("tokPerSec", service.getLastDecodeSpeedTokPerSec())
            .put("reply", reply.replace(THINK, "<think>").replace(END_THINK, "</think>").take(1500))
        Log.i(TAG, "RESULT $result")
    }

    /** 256x256 (LFM3-VL's minimum pixel budget): a red circle on white. */
    private fun redCircle(): Bitmap {
        val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).apply {
            drawColor(Color.WHITE)
            drawCircle(128f, 128f, 96f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.RED })
        }
        return bitmap
    }

    private companion object {
        const val TAG = "LlamaCppLfm3"
        // LlamaCppInferenceService's thinking stream sentinels.
        const val THINK = "​​THINK​​"
        const val END_THINK = "​​ENDTHINK​​"
        // Hugging Face LFS oids (SHA-256) at the pinned commits.
        const val TEXT_SHA256 = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"
        const val VISION_SHA256 = "621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6"
        const val PROJECTOR_BF16_SHA256 = "8e75a88afc0e6a3bcd637becbe720447ea0794934b7b417a1f42d5cfe0d9c288"
        const val PROJECTOR_Q8_SHA256 = "c5c5b58d544a18da919172d2d0368090ea1a4a351b6feb7f8a326a264a5918ab"
    }
}
