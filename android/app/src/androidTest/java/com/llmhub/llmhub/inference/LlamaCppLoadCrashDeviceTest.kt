package com.llmhub.llmhub.inference

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device repro for the Play vitals SIGSEGV in llama_vocab::n_tokens() during
 * load_arch_tensors (libllmhub_llama_cpu.so, every architecture). Each scenario is its own test,
 * because a native crash ends the instrumentation process; run them one at a time with
 * `-e class ...LlamaCppLoadCrashDeviceTest#<name>`. The model is fetched on the device (pinned).
 */
class LlamaCppLoadCrashDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val model = ModelData.models.first { it.name == "LFM-2.5 1.2B Instruct (Q4_0)" }

    @Before fun fixture() {
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        if (!file.isFile) {
            OnDeviceFixtures.download(
                file,
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_0.gguf",
                "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
            )
        }
    }

    /** Plain CPU load, repeated: the path every Snapdragon-less phone takes. */
    @Test fun cpuLoad() = runBlocking {
        val service = LlamaCppInferenceService(context)
        repeat(3) { round ->
            assertTrue("CPU load $round", service.loadModel(model, LlmInference.Backend.CPU, null))
            Log.i(TAG, "cpuLoad round $round ok")
        }
        val reply = service.generateResponse("What is the capital of France? Answer in one word.", model)
        Log.i(TAG, "cpu reply: ${reply.take(200)}")
        assertTrue("Unexpected CPU answer: $reply", reply.contains("Paris", ignoreCase = true))
        service.unloadModel()
    }

    /** Hexagon NPU (HTP0) on Snapdragon: loads, answers, unloads. */
    @Test fun npuGenerates() = runBlocking {
        val service = LlamaCppInferenceService(context)
        assertTrue("NPU load", service.loadModel(model, LlmInference.Backend.GPU, "HTP0"))
        val reply = service.generateResponse("What is the capital of France? Answer in one word.", model)
        Log.i(TAG, "npu reply: ${reply.take(200)}")
        assertTrue("Unexpected NPU answer: $reply", reply.contains("Paris", ignoreCase = true))
        service.unloadModel()
    }

    /** Snapdragon GPU (OpenCL) first, then CPU in the same process: the accelerator fallback path. */
    @Test fun gpuThenCpu() = runBlocking {
        val service = LlamaCppInferenceService(context)
        Log.i(TAG, "gpu load: ${service.loadModel(model, LlmInference.Backend.GPU, null)}")
        assertTrue("CPU load after GPU", service.loadModel(model, LlmInference.Backend.CPU, null))
        Log.i(TAG, "gpuThenCpu ok")
        service.unloadModel()
    }

    /** Two features loading at once: each service has its own dispatcher, the JNI bridge one global model. */
    @Test fun concurrentCpuLoads() = runBlocking {
        val services = List(2) { LlamaCppInferenceService(context) }
        repeat(3) { round ->
            val results = services.map { service -> async { service.loadModel(model, LlmInference.Backend.CPU, null) } }.awaitAll()
            Log.i(TAG, "concurrentCpuLoads round $round: $results")
        }
        services.forEach { it.unloadModel() }
    }

    private companion object { const val TAG = "LlamaCppLoadCrash" }
}
