package com.llmhub.llmhub.inference

import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import com.llmhub.llmhub.imagegen.SdCppImageRuntime
import com.llmhub.llmhub.videogen.VideoGenEngine
import com.llmhub.llmhub.videogen.VideoGenFormats
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Play vitals SIGSEGV in vk::Device::getBufferAddress during load_tensors (Mali-G57, 4.5.7-4.5.8): ggml
 * used GPUs whose driver is Vulkan 1.1. For Vulkan 1.1 devices (AWS Device Farm Mali-G57 phones); the
 * model is downloaded on the device. On Vulkan 1.2+ drivers the same tests check Vulkan still works.
 */
class VulkanDriverDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val model = ModelData.models.first { it.name == "LFM-2.5 1.2B Instruct (Q4_0)" }
    private val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())

    @Before fun fixture() {
        DeviceInfo.init(context)
        val vulkan = context.packageManager.systemAvailableFeatures
            .firstOrNull { it.name == PackageManager.FEATURE_VULKAN_HARDWARE_VERSION }?.version ?: 0
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE} " +
            "vulkan=${vulkan shr 22}.${(vulkan shr 12) and 0x3ff} hasVulkan12=${DeviceInfo.hasVulkan12} " +
            "imageGpu=${SdCppImageRuntime.usesGpu(true)} videoBackends=${VideoGenEngine.supportedBackends(ModelData.videoGenerationModels.first { it.modelFormat == VideoGenFormats.GGUF })}")
        if (!file.isFile) {
            OnDeviceFixtures.download(
                file,
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_0.gguf",
                "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
            )
        }
    }

    /** The raw Vulkan load: an error code (no usable GPU) on Vulkan 1.1 drivers instead of the SIGSEGV. */
    @Test fun nativeVulkanLoadDoesNotCrash() {
        val runtime = LlamaCppVulkanNative
        assertEquals(0, runtime.nativeInit(context.applicationInfo.nativeLibraryDir, null))
        val result = runtime.nativeLoadModel(file.absolutePath, null, 512, 4, "Vulkan0", 999, -1)
        Log.i(TAG, "RESULT native Vulkan0 load returned $result (hasVulkan12=${DeviceInfo.hasVulkan12})")
        if (!DeviceInfo.hasVulkan12) assertNotEquals("A Vulkan 1.1 GPU must be rejected", 0, result)
        else assertEquals("Vulkan 1.2+ GPU must still load", 0, result)
        runtime.nativeUnload()
    }

    /** What the app does when the user picks GPU: loads (CPU on Vulkan 1.1, Vulkan otherwise) and answers. */
    @Test fun gpuLoadFromTheAppAnswers(): Unit = runBlocking {
        val service = LlamaCppInferenceService(context)
        // A chat-sized context: the model's full 128K KV cache gets low-RAM phones killed by lmkd.
        service.setGenerationParameters(null, null, null, null, null, null, 4096)
        assertTrue("GPU load must succeed or fall back", service.loadModel(model, LlmInference.Backend.GPU, null))
        val answer = service.generateResponse("What is the capital of France? Answer in one word.", model)
        Log.i(TAG, "RESULT app GPU load ran on ${service.getCurrentlyLoadedBackend()}, answered: ${answer.take(60)}")
        assertTrue("Unexpected answer: $answer", answer.contains("Paris", ignoreCase = true))
        service.unloadModel()
    }

    private companion object { const val TAG = "VulkanDriverDeviceTest" }
}
