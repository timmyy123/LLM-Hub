package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Google's Gemma 4 E2B build for this phone's NPU (Tensor G5 or G6), the way the app runs it: the catalog entry is
 * downloaded by the app's downloader, loaded by [LiteRtLmInferenceService] (NPU text model) and asked with the chat
 * screen's prompt format, including a follow-up that needs the history. LiteRT-LM's own prefill and decode speeds are
 * logged by the service (tag LiteRtLmInferenceService); the test logs load and reply times (tag Gemma4Npu).
 * Skipped on chips without a build.
 */
class Gemma4NpuDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun catalogEntryChatsOnTheNpu() = runBlocking {
        val tag = DeviceInfo.getEmbeddingGemma2NpuTag()
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${Build.SOC_MODEL} tag=$tag")
        val entry = ModelData.allGemma4E2bNpuModels().firstOrNull { tag != null && it.url.contains("_$tag.litertlm") }
        assumeTrue("No Gemma 4 E2B NPU build for ${Build.SOC_MODEL}", entry != null)
        entry!!
        check(ModelData.models.any { it.name == entry.name }) { "${entry.name} is not in this device's catalog" }

        val file = File(context.filesDir, "models/${entry.localFileName()}")
        if (file.length() != entry.sizeBytes) {
            val started = SystemClock.elapsedRealtime()
            io.ktor.client.HttpClient(io.ktor.client.engine.android.Android).use { client ->
                com.llmhub.llmhub.data.ModelDownloader(client, context).downloadModel(entry).collect { }
            }
            Log.i(TAG, "downloaded ${entry.name} in ${SystemClock.elapsedRealtime() - started} ms")
        }
        assertTrue("${entry.name} download incomplete: ${file.length()} of ${entry.sizeBytes}", file.length() == entry.sizeBytes)

        val service = LiteRtLmInferenceService(context)
        try {
            service.setGenerationParameters(null, 1, 1.0f, 0.0f, null, false, null)
            var started = SystemClock.elapsedRealtime()
            assertTrue("${entry.name} must load on the NPU", service.loadModel(entry, LlmInference.Backend.CPU, null))
            val loadMs = SystemClock.elapsedRealtime() - started
            suspend fun chat(label: String, prompt: String): String {
                val out = StringBuilder()
                started = SystemClock.elapsedRealtime()
                service.generateResponseStreamWithSession(prompt, entry, "gemma4-npu-test", emptyList(), null, false, emptyList())
                    .collect { out.append(it) }
                Log.i(TAG, "RESULT " + JSONObject().put("model", entry.name).put("soc", Build.SOC_MODEL).put("test", label)
                    .put("loadMs", loadMs).put("replyMs", SystemClock.elapsedRealtime() - started).put("reply", out.toString().take(1500)))
                return out.toString()
            }
            val first = chat("capital", "user: What is the capital of France? Answer in one word.\nassistant:")
            assertTrue("Unexpected answer: $first", first.contains("Paris", ignoreCase = true))
            val second = chat("followUp", "user: What is the capital of France? Answer in one word.\n\nassistant: $first\n\n" +
                "user: And what river runs through it? Answer in one word.\nassistant:")
            assertTrue("Unexpected follow-up: $second", second.contains("Seine", ignoreCase = true))
            val story = chat("story", "user: Write a short story of about 150 words about a lighthouse keeper.\nassistant:")
            assertTrue("Story too short: $story", story.split(Regex("\\s+")).size >= 60)
        } finally {
            service.unloadModel()
        }
    }

    private companion object {
        const val TAG = "Gemma4Npu"
    }
}
