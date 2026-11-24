package com.llmhub.llmhub.inference

import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.Gson
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test

/**
 * Opt-in physical-device check of the imported-model prompt template. Needs an imported GGUF named
 * "LFM3-1.7B-Test" (Models > Import > Hugging Face, LiquidAI/LFM3-1.7B-GGUF, LFM3-1.7B-QAD-Q4_0.gguf).
 * Run one at a time with `-e class ...LlamaCppPromptTemplateDeviceTest#<name>`. Results go to logcat (tag PromptTemplate).
 */
class LlamaCppPromptTemplateDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Before fun setUp() {
        DeviceInfo.init(context)
    }

    /** No template: the imported model is formatted exactly like the catalog LFM3 1.7B and answers as before. */
    @Test fun withoutTemplate() = runBlocking {
        val imported = importedModel()
        assertNull("The import must start without a template", imported.promptTemplate)
        val catalog = ModelData.models.first { it.name == "LFM3 1.7B (Q4_0)" }
        val service = load(imported)
        try {
            val importedPrompt = format(service, CHAT, imported)
            val catalogPrompt = format(service, CHAT, catalog)
            Log.i(TAG, "PROMPT no-template ${JSONObject.quote(importedPrompt)}")
            assertEquals("No template must format like the catalog model", catalogPrompt, importedPrompt)
            ask(service, imported, "no-template")
        } finally {
            service.unloadModel()
        }
    }

    /** A typed LFM3 ChatML template drives the prompt, and the model still answers and keeps the conversation. */
    @Test fun withTemplate() = runBlocking {
        val templated = importedModel().copy(promptTemplate = LFM3_TEMPLATE)
        val service = load(templated)
        try {
            val prompt = format(service, CHAT, templated)
            Log.i(TAG, "PROMPT template ${JSONObject.quote(prompt)}")
            assertEquals(
                "<|im_start|>user\nYou are a helpful assistant.\n\nWhat is the capital of France? Answer in one word.<|im_end|>\n" +
                    "<|im_start|>assistant\nParis<|im_end|>\n" +
                    "<|im_start|>user\nAnd of Italy? Answer in one word.<|im_end|>\n" +
                    "<|im_start|>assistant\n",
                prompt,
            )
            ask(service, templated, "template")
            val started = SystemClock.elapsedRealtime()
            val reply = service.generateResponse(CHAT, templated)
            report("template-multiturn", templated, reply, SystemClock.elapsedRealtime() - started)
            assertTrue("Follow-up should name Rome: $reply", answerOf(reply).contains("Rome", ignoreCase = true))
        } finally {
            service.unloadModel()
        }
    }

    /**
     * The same import renamed to "MyModel-1.7B": no name rule recognises it, as for any unknown model.
     * Runs with thinking on, without and with a template, and splits each reply with the chat screen's parser.
     */
    @Test fun unknownModelThinking() = runBlocking {
        val unknown = renameImport("LFM3-1.7B-Test", UNKNOWN_NAME)
        assertNull("The import must start without a template", unknown.promptTemplate)
        val service = load(unknown)
        try {
            for (model in listOf(unknown, unknown.copy(promptTemplate = LFM3_THINK_TEMPLATE))) {
                val label = if (model.promptTemplate == null) "unknown-no-template" else "unknown-template"
                Log.i(TAG, "PROMPT $label ${JSONObject.quote(format(service, QUESTION, model))}")
                val started = SystemClock.elapsedRealtime()
                val reply = service.generateResponse(QUESTION, model)
                report(label, model, reply, SystemClock.elapsedRealtime() - started)
                val (thinking, answer) = com.llmhub.llmhub.components.parseThinkingAndAnswer(reply)
                Log.i(TAG, "PARSED $label ${JSONObject().put("thinking", thinking.take(600)).put("answer", answer.take(300))}")
            }
        } finally {
            service.unloadModel()
        }
    }

    /** Renames the stored import (the app reads the new name on its next start). */
    private fun renameImport(from: String, to: String): LLMModel {
        val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        val models = Gson().fromJson(prefs.getString("imported_models", "[]"), Array<LLMModel>::class.java).toList()
        val renamed = models.map { if (it.name == from) it.copy(name = to) else it }
        prefs.edit().putString("imported_models", Gson().toJson(renamed)).commit()
        val model = renamed.firstOrNull { it.name == to }
        assertNotNull("No import named $from or $to", model)
        return model!!
    }

    private suspend fun ask(service: LlamaCppInferenceService, model: LLMModel, label: String) {
        val started = SystemClock.elapsedRealtime()
        val reply = service.generateResponse("What is the capital of France? Answer in one word.", model)
        report(label, model, reply, SystemClock.elapsedRealtime() - started)
        assertTrue("Unexpected answer ($label): $reply", answerOf(reply).contains("Paris", ignoreCase = true))
    }

    private fun importedModel(): LLMModel {
        val json = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE).getString("imported_models", null)
        assumeTrue("Import LFM3-1.7B-Test first", json != null)
        val model = Gson().fromJson(json, Array<LLMModel>::class.java).firstOrNull { it.name == "LFM3-1.7B-Test" }
        assertNotNull("Import LFM3-1.7B-Test first", model)
        return model!!
    }

    private suspend fun load(model: LLMModel): LlamaCppInferenceService {
        assumeTrue("OpenCL GGUF runtime needs a Snapdragon SoC", DeviceInfo.isSnapdragon())
        val service = LlamaCppInferenceService(context)
        service.setGenerationParameters(
            maxTokens = 2048, topK = 50, topP = 1.0f, temperature = 0.1f,
            nGpuLayers = null, enableThinking = true, contextWindow = 8192,
        )
        assertTrue("${model.name} must load on the GPU", service.loadModel(model, LlmInference.Backend.GPU, null))
        return service
    }

    /** The exact text llama.cpp receives (the service's private formatter, thinking on as in [load]). */
    private fun format(service: LlamaCppInferenceService, prompt: String, model: LLMModel): String {
        val method = LlamaCppInferenceService::class.java.getDeclaredMethod(
            "formatPromptLikeGeniex", String::class.java, LLMModel::class.java, Boolean::class.javaPrimitiveType,
        ).apply { isAccessible = true }
        return method.invoke(service, prompt, model, true) as String
    }

    private fun answerOf(reply: String): String = reply.substringAfterLast(END_THINK)

    private fun report(label: String, model: LLMModel, reply: String, generateMs: Long) {
        val result = JSONObject().put("case", label).put("model", model.name).put("template", model.promptTemplate != null)
            .put("generateMs", generateMs)
            .put("reply", reply.replace(THINK, "<think>").replace(END_THINK, "</think>").take(1500))
        Log.i(TAG, "RESULT $result")
    }

    private companion object {
        const val TAG = "PromptTemplate"
        const val THINK = "​​THINK​​"
        const val END_THINK = "​​ENDTHINK​​"
        const val UNKNOWN_NAME = "MyModel-1.7B"
        const val QUESTION = "What is the capital of France? Answer in one word."
        /** LFM3's generation prompt opens the think block, as the model's own chat template does. */
        const val LFM3_THINK_TEMPLATE = "<|im_start|>user\n{{USER}}<|im_end|>\n<|im_start|>assistant\n<think>{{ASSISTANT}}<|im_end|>\n"
        const val LFM3_TEMPLATE ="<|im_start|>user\n{{USER}}<|im_end|>\n<|im_start|>assistant\n{{ASSISTANT}}<|im_end|>\n"
        const val CHAT = "system: You are a helpful assistant.\n\n" +
            "user: What is the capital of France? Answer in one word.\n\n" +
            "assistant: Paris\n\n" +
            "user: And of Italy? Answer in one word.\n\n" +
            "assistant:"
    }
}
