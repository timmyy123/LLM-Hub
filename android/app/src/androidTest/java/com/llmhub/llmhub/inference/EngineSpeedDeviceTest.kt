package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Test
import java.io.File

/**
 * The same catalog `.litertlm` file through the old MediaPipe LLM Inference engine and through LiteRT-LM, on the same
 * backend, with the same greedy settings and prompts, one engine per test (the file must already be downloaded where
 * the app keeps it, downloaded first when missing). Logs load time, time to the first text, total time and words per
 * second (tag EngineSpeed); LiteRT-LM also logs its own token speeds (tag LiteRtLmInference).
 *
 * `-e model "Aion 1.0 Instruct 2.3B (LiteRT-LM)"`, `-e backend CPU` (default GPU), `-e rounds 2`.
 */
class EngineSpeedDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val args = InstrumentationRegistry.getArguments()

    @Test fun mediaPipe() = run("MediaPipe") { MediaPipeInferenceService(context) }

    @Test fun liteRtLm() = run("LiteRT-LM") { LiteRtLmInferenceService(context) }

    private fun run(engine: String, create: () -> InferenceService) = runBlocking {
        val name = args.getString("model") ?: error("Pass -e model <catalog name>")
        fun key(text: String) = text.lowercase().replace(Regex("[^a-z0-9]"), "")
        val model = ModelData.models.first { it.name == name || key(it.name) == key(name) }
        download(model)
        val backend = if (args.getString("backend") == "CPU") LlmInference.Backend.CPU else LlmInference.Backend.GPU
        val service = create()
        try {
            service.setGenerationParameters(1024, 1, 1.0f, 0.0f, null, false, null)
            var started = SystemClock.elapsedRealtime()
            check(service.loadModel(model, backend, true, true, null)) { "${model.name} did not load on $engine" }
            val loadMs = SystemClock.elapsedRealtime() - started
            log(engine, model.name, "load", JSONObject().put("loadMs", loadMs).put("backend", backend.name))
            val prompts = listOf(
                "short" to "user: What is the capital of France? Answer in one word.\nassistant:",
                "story" to "user: Write a short story of about 150 words about a lighthouse keeper.\nassistant:",
                "explain" to "user: Explain in about 120 words how a refrigerator keeps food cold.\nassistant:",
            )
            repeat(args.getString("rounds")?.toIntOrNull() ?: 1) { round ->
                for ((label, prompt) in prompts) {
                    val out = StringBuilder()
                    var firstMs = -1L
                    started = SystemClock.elapsedRealtime()
                    service.generateResponseStreamWithSession(prompt, model, "engine-speed-$engine-$label-$round", emptyList(), null, false, emptyList())
                        .collect { chunk ->
                            if (firstMs < 0 && chunk.isNotBlank()) firstMs = SystemClock.elapsedRealtime() - started
                            out.append(chunk)
                        }
                    val ms = SystemClock.elapsedRealtime() - started
                    val words = out.split(Regex("\\s+")).count { it.isNotBlank() }
                    val afterFirst = (ms - firstMs).coerceAtLeast(1)
                    val tokens = mediaPipeTokens(service, out.toString())
                    log(engine, model.name, label, JSONObject().put("round", round).put("firstTextMs", firstMs).put("totalMs", ms)
                        .put("words", words).put("tokens", tokens ?: JSONObject.NULL)
                        .put("tokPerSecAfterFirst", tokens?.let { "%.2f".format(it * 1000.0 / afterFirst) } ?: JSONObject.NULL)
                        .put("reply", out.take(160).toString()))
                }
            }
        } finally {
            service.unloadModel()
        }
    }

    /** The catalog file where the app keeps it (public mirrors only), resumed after drops, checked by size. */
    private fun download(model: com.llmhub.llmhub.data.LLMModel) {
        val target = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        if (target.length() == model.sizeBytes) return
        repeat(6) { attempt ->
            try {
                val partial = File(target.path + ".part")
                val connection = java.net.URL(model.url).openConnection() as java.net.HttpURLConnection
                if (partial.length() > 0) connection.setRequestProperty("Range", "bytes=${partial.length()}-")
                connection.instanceFollowRedirects = true
                val code = connection.responseCode
                check(code in 200..299) { "HTTP $code for ${model.name}" }
                connection.inputStream.use { input -> java.io.FileOutputStream(partial, code == 206).use { input.copyTo(it, 1 shl 20) } }
                check(partial.length() == model.sizeBytes) { "${model.name}: ${partial.length()} of ${model.sizeBytes} bytes" }
                partial.renameTo(target)
                return
            } catch (e: java.io.IOException) {
                Log.w(TAG, "Download of ${model.name} failed (try ${attempt + 1}): $e")
                Thread.sleep(10_000)
            }
        }
        error("Could not download ${model.name}")
    }

    /** MediaPipe's own tokenizer count of a reply (LiteRT-LM logs its token speeds itself). */
    private fun mediaPipeTokens(service: InferenceService, text: String): Int? = (service as? MediaPipeInferenceService)?.let {
        val field = MediaPipeInferenceService::class.java.getDeclaredField("modelInstance").apply { isAccessible = true }
        (field.get(it) as? LlmModelInstance)?.engine?.sizeInTokens(text)
    }

    private fun log(engine: String, model: String, label: String, data: JSONObject) {
        Log.i(TAG, "RESULT " + data.put("engine", engine).put("model", model).put("test", label).put("soc", Build.SOC_MODEL))
    }

    private companion object {
        const val TAG = "EngineSpeed"
    }
}
