package com.llmhub.llmhub.inference

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.GgufLayerLimits
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.isInkling2
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * A vision MoE GGUF with flash streaming, as the app stores it: thinking on shows a thinking block then "Paris",
 * thinking off shows the answer alone, and a red circle is named red. `-e model_name NAME` picks the catalog model
 * (default Inkling-2-Mini UD-Q4_K_XL) and `-e model_sha256 HEX` downloads it when missing. Its projector is the catalog
 * entry `-e projector_name NAME` (default Inkling's Q8_0 projector; `-e projector_sha256 HEX` downloads another). `-e modes npu,cpu` (default both on a supported Snapdragon); the NPU is
 * expected where the app would use it. Prompt and decode tok/s per turn are in the runtime's "Turn:" log lines.
 */
class LlamaCppMoeVisionDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val args = InstrumentationRegistry.getArguments()
    private val model = ModelData.models.first { it.name == (args.getString("model_name") ?: "Inkling-2-Mini (UD-Q4_K_XL, GGUF)") }

    @Test fun thinkingAndImage() = runBlocking {
        DeviceInfo.init(context)
        val models = File(context.filesDir, "models").apply { mkdirs() }
        args.getString("model_sha256")?.let { sha ->
            val target = File(models, model.localFileName())
            if (!target.isFile) OnDeviceFixtures.download(target, model.url, sha)
        }
        val projectorName = args.getString("projector_name") ?: INKLING_PROJECTOR
        val projector = ModelData.models.first { it.name == projectorName }
        val projectorFile = File(models, projector.localFileName())
        if (!projectorFile.isFile) {
            val sha = args.getString("projector_sha256") ?: INKLING_PROJECTOR_SHA256.takeIf { projectorName == INKLING_PROJECTOR }
            OnDeviceFixtures.download(projectorFile, projector.url, sha ?: error("projector_sha256"))
        }
        assertTrue("${model.name} must be a MoE GGUF on disk", GgufLayerLimits.isMoe(context, model))
        val npuTakesExperts = !GgufLayerLimits.npuCannotStreamExperts(context, model)

        data class Mode(val label: String, val backend: LlmInference.Backend, val device: String?, val runtime: String)
        val all = buildList {
            // The NPU reads the prompt when it can multiply by the experts. Inkling 2 has no NPU path (its NPU prompt
            // graph outgrows a 12 GB phone, and cut down the NPU computes it wrong): an NPU choice, such as one saved
            // before the option was hidden, streams on the CPU, like experts the NPU cannot take.
            if (DeviceInfo.isLlamaCppHexagonSupported()) {
                val onNpu = npuTakesExperts && !model.isInkling2()
                add(Mode("npu", LlmInference.Backend.GPU, "HTP0", if (onNpu) "NPU flash streaming" else "CPU flash streaming"))
            }
            add(Mode("cpu", LlmInference.Backend.CPU, null, "CPU flash streaming"))
        }
        val picked = args.getString("modes")?.split(',')?.map { it.trim() }
        val modes = if (picked == null) all else all.filter { it.label in picked }
        val result = JSONObject().put("model", model.name).put("npuTakesExperts", npuTakesExperts)
        val failures = mutableListOf<String>()
        for (mode in modes) {
            val entry = JSONObject().also { result.put(mode.label, it) }
            val service = LlamaCppInferenceService(context)
            fun params(thinking: Boolean) = service.setGenerationParameters(
                maxTokens = 1024, topK = 40, topP = 0.95f, temperature = 0.1f,
                nGpuLayers = null, enableThinking = thinking, contextWindow = 4096,
            )
            try {
                params(thinking = true)
                val started = SystemClock.elapsedRealtime()
                val loaded = service.loadModel(model, mode.backend, mode.device)
                entry.put("loadMs", SystemClock.elapsedRealtime() - started).put("runtime", service.activeRuntimeName())
                    .put("vision", !service.isVisionCurrentlyDisabled())
                if (!loaded || service.activeRuntimeName() != mode.runtime) {
                    failures += "${mode.label}: loaded ${service.activeRuntimeName()}"
                    continue
                }
                for (thinking in listOf(true, false)) {
                    params(thinking)
                    val reply = StringBuilder()
                    service.generateResponseStream("What is the capital of France? Answer in one word.", model)
                        .collect { reply.append(it) }
                    val text = reply.toString()
                    val key = if (thinking) "thinkingOn" else "thinkingOff"
                    entry.put(key, show(text)).put("${key}TokPerSec", service.getLastDecodeSpeedTokPerSec())
                    val answer = text.substringAfterLast(END_THINK)
                    if (!answer.contains("Paris", ignoreCase = true)) failures += "${mode.label} $key: ${show(text).takeLast(200)}"
                    if (thinking != text.contains(THINK)) failures += "${mode.label} $key: thinking block ${text.contains(THINK)}"
                    if (text.contains("<|")) failures += "${mode.label} $key shows block tokens"
                }
                params(thinking = false)
                assertFalse("The vision projector must load on ${mode.label}", service.isVisionCurrentlyDisabled())
                val image = redCircle()
                val reply = StringBuilder()
                try {
                    service.generateResponseStreamWithSession(
                        prompt = "What color is the circle in this image? Answer in one word.",
                        model = model, chatId = "moe-vision-image-test", images = listOf(image),
                        audioData = null, webSearchEnabled = false, imagePaths = emptyList(),
                    ).collect { reply.append(it) }
                } finally {
                    image.recycle()
                }
                entry.put("image", show(reply.toString()))
                if (!reply.contains("red", ignoreCase = true)) failures += "${mode.label} image: ${show(reply.toString()).take(200)}"
            } catch (t: Throwable) {
                entry.put("error", "${t.javaClass.simpleName}: ${t.message}")
                failures += "${mode.label}: ${t.message}"
            } finally {
                service.unloadModel()
                Log.i(TAG, "RESULT $result")
            }
        }
        assertTrue("Failed: $failures", failures.isEmpty())
    }

    private fun show(text: String) = text.replace(THINK, "<think>").replace(END_THINK, "</think>").take(1500)

    /** 448x448: a red circle on white. */
    private fun redCircle(): Bitmap {
        val bitmap = Bitmap.createBitmap(448, 448, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).apply {
            drawColor(Color.WHITE)
            drawCircle(224f, 224f, 150f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.RED })
        }
        return bitmap
    }

    private companion object {
        const val TAG = "LlamaCppMoeVision"
        const val THINK = "​​THINK​​"
        const val END_THINK = "​​ENDTHINK​​"
        const val INKLING_PROJECTOR = "Inkling-2-Mini (Vision Projector, Q8_0)"
        const val INKLING_PROJECTOR_SHA256 = "233287272ec5ac39b4168677aba54a4c024f202410f5629fac1830e1059f8abb"
    }
}
