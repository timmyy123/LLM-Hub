package com.llmhub.llmhub.vibecode

import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.agent.TermuxStreamingCommand
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.data.ModelPreferences
import com.llmhub.llmhub.data.loadModelWithSavedConfig
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.inference.UnifiedInferenceService
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collect
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.util.UUID

/** Opt-in hardware test: selected local model, actual Codex tools, isolated shared-storage project. */
class CodexLocalModelInstrumentedTest {
    @Test fun selectedLocalModelFixesTypoThroughCodex() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("Enable explicitly with -e codex_real_model true", args.getString("codex_real_model") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val prefs = context.getSharedPreferences("vibe_coder_prefs", 0)
        val modelName = checkNotNull(prefs.getString("selected_model_name", null))
        val model = ModelAvailabilityProvider.loadAvailableModels(context).first { it.name == modelName }
        val inference = UnifiedInferenceService(context)
        val project = "/storage/emulated/0/Download/llmhub-real-model-${UUID.randomUUID()}"
        val fileName = args.getString("codex_file_name") ?: "index.html"
        require(Regex("[A-Za-z0-9_.-]+\\.html").matches(fileName))
        val activity = StringBuffer()
        try {
            java.io.File(context.cacheDir, "codex-real-model-raw.txt").writeText("")
            val source = args.getString("codex_source_file")
            val fixture = "<!DOCTYPE html>\n<html><body><p id='output'></p><script>\n" +
                "const display = document.getElementById('output');\nconst randomAffirmation = 'Hello';\n" +
                "display.textContent = randomAffation;\n</script></body></html>\n"
            val prepare = if (source != null) "cp ${CodexConfig.shellQuote(source)} ${CodexConfig.shellQuote("$project/$fileName")}" else
                "printf '%s' ${CodexConfig.shellQuote(fixture)} > ${CodexConfig.shellQuote("$project/$fileName")}"
            TermuxStreamingCommand.run(context,
                "mkdir -p ${CodexConfig.shellQuote(project)} && $prepare", 10000) {}
            val backend = prefs.getString("selected_backend_$modelName", prefs.getString("selected_backend", "CPU"))
            val device = prefs.getString("selected_npu_device_id_$modelName", prefs.getString("selected_npu_device_id", null))
            val layers = prefs.getInt("n_gpu_layers_$modelName", 0)
            val thinking = prefs.getBoolean("enable_thinking_$modelName", prefs.getBoolean("enable_thinking", true))
            assertTrue(loadModelWithSavedConfig(model, ModelPreferences(context), inference,
                backendOverride = LlmInference.Backend.valueOf(backend ?: "CPU"), deviceIdOverride = device,
                contextWindowOverride = 8192, maxTokensOverride = 4096, nGpuLayersOverride = layers,
                enableThinkingOverride = thinking))
            inference.setGenerationParameters(maxTokens = 4096, contextWindow = 8192, temperature = 0.2f, topK = 40,
                topP = 0.95f, nGpuLayers = layers, enableThinking = thinking)
            withTimeout(300_000) {
                CodexAgent(context).run(
                    args.getString("codex_prompt") ?: "Read index.html. Fix the JavaScript typo randomAffation to randomAffirmation. Make the actual file edit, verify the typo is gone, then finish.",
                    project, UUID.randomUUID().toString(), null, null, fileName, "", 8192,
                    infer = { prompt, emit ->
                        val raw = StringBuilder()
                        inference.generateResponseStream(prompt, model).collect { raw.append(it); emit(it) }
                        java.io.File(context.cacheDir, "codex-real-model-raw.txt").appendText("\nPROMPT\n$prompt\nRESPONSE\n$raw\n")
                        raw.toString()
                    }, onThread = {}, onMessage = { activity.append("\n").append(it) })
            }
            val file = StringBuffer()
            TermuxStreamingCommand.run(context,"cat ${CodexConfig.shellQuote("$project/$fileName")}",10000){file.append(it)}
            val expected = args.getString("codex_expected")
            if (expected != null) {
                assertTrue(activity.toString(), file.contains(expected))
                if (args.getString("codex_tailwind_import") == "true") {
                    assertTrue(activity.toString(), Regex("<(?:script|link)\\b[^>]*(?:src|href)=[\"'][^\"']*tailwind[^\"']*[\"']", RegexOption.IGNORE_CASE)
                        .containsMatchIn(file.toString()))
                }
                assertFalse(activity.toString(), file.toString() == fixture)
            } else {
                assertTrue(activity.toString(), file.contains("display.textContent = randomAffirmation;"))
                assertFalse(activity.toString(), file.contains("randomAffation"))
            }
            java.io.File(context.cacheDir, "codex-real-model-result.html").writeText(file.toString())
        } finally {
            java.io.File(context.cacheDir, "codex-real-model-activity.txt").writeText(activity.toString())
            withContext(NonCancellable) {
                inference.unloadModel()
                runCatching { TermuxStreamingCommand.run(context,"rm -rf -- ${CodexConfig.shellQuote(project)}",10000) {} }
            }
        }
    }
}
