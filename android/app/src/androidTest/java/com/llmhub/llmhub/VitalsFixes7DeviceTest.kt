package com.llmhub.llmhub

import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.SavedStateHandle
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.LiteRtLmInferenceService
import com.llmhub.llmhub.inference.LlamaCppInferenceService
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.llmhub.llmhub.inference.OnnxInferenceService
import com.llmhub.llmhub.viewmodels.ChatViewModel
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Leaving an LLM feature or switching chats while a reply is generating must stop the reply first
 * and then unload/reset, without holding the main thread (4.6.7/4.6.8 ANRs: ChatViewModel.initializeChat
 * runBlocking behind a llama.cpp reply, MediaPipe session.close() on the main thread). Every engine,
 * with the models already in the app's models folder (ONNX downloads its own).
 */
class VitalsFixes7DeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val tag = "VitalsFixes7"
    private val modelsDir = File(context.filesDir, "models")

    private fun model(name: String): LLMModel = ModelData.models.first { it.name == name }

    private fun requireFile(model: LLMModel): File =
        File(modelsDir, model.localFileName()).also { assertTrue("${it.name} is not on the device", it.isFile) }

    /** Longest gap between 16 ms main-thread heartbeats while [block] runs. */
    private class Heartbeat {
        private val handler = Handler(Looper.getMainLooper())
        private val last = AtomicLong(SystemClock.elapsedRealtime())
        val longestGap = AtomicLong(0)
        @Volatile private var running = true
        private val beat = object : Runnable {
            override fun run() {
                val now = SystemClock.elapsedRealtime()
                longestGap.set(maxOf(longestGap.get(), now - last.get())); last.set(now)
                if (running) handler.postDelayed(this, 16)
            }
        }
        fun start() { last.set(SystemClock.elapsedRealtime()); handler.post(beat) }
        fun stop() { running = false }
    }

    private class Reply(val tokens: AtomicInteger, val ended: CompletableDeferred<Long>, val firstTokenAt: CompletableDeferred<Long>)

    private fun kotlinx.coroutines.CoroutineScope.startReply(service: InferenceService, model: LLMModel, chatId: String, prompt: String): Reply {
        val tokens = AtomicInteger()
        val ended = CompletableDeferred<Long>()
        val first = CompletableDeferred<Long>()
        launch(Dispatchers.IO) {
            try {
                service.generateResponseStreamWithSession(prompt, model, chatId, webSearchEnabled = false).collect {
                    if (tokens.incrementAndGet() == 1) first.complete(SystemClock.elapsedRealtime())
                }
            } catch (e: Throwable) {
                Log.i(tag, "reply ended with ${e.javaClass.simpleName}: ${e.message}")
            } finally {
                ended.complete(SystemClock.elapsedRealtime())
            }
        }
        return reply(tokens, ended, first)
    }

    private fun reply(t: AtomicInteger, e: CompletableDeferred<Long>, f: CompletableDeferred<Long>) = Reply(t, e, f)

    private val story = "Write a very long, detailed story about a lighthouse keeper and the storms he records. At least 3000 words."

    /** About 2,500 words; [seed] first so llama.cpp cannot reuse an earlier prompt's cache. */
    private fun longPrompt(seed: Long): String = buildString {
        append("Reference number $seed. Summarize the following harbor log in five bullet points.\n\n")
        var i = 0
        while (length < 15_000) {
            append("Entry ${i++}: the north pier lamp was trimmed at dusk, the ferry from Saltmere arrived ")
            append("forty minutes late because of fog, two crates of rope and one barrel of oil were unloaded, ")
            append("and the tide gauge read three point two meters at the evening check. ")
        }
    }

    /**
     * Runs [action] on the main thread like the app does (navigation's LaunchedEffect, initializeChat),
     * and checks the reply stopped first: it ends by the time the action returns and sends nothing after.
     */
    private suspend fun stopAndCheck(label: String, reply: Reply, maxMainGapMs: Long, action: suspend () -> Unit): Long {
        val tokensAtAction = reply.tokens.get()
        val hb = Heartbeat()
        hb.start()
        val t0 = SystemClock.elapsedRealtime()
        withContext(Dispatchers.Main) { action() }
        val actionMs = SystemClock.elapsedRealtime() - t0
        hb.stop()
        val endedAt = withTimeoutOrNull(15_000) { reply.ended.await() }
        val tokensAtReturn = reply.tokens.get()
        delay(1500)
        val tokensLater = reply.tokens.get()
        Log.i(
            tag,
            "RESULT $label: action ${actionMs} ms, longest main-thread gap ${hb.longestGap.get()} ms, " +
                "reply ended ${endedAt?.let { "${it - t0} ms after the action started" } ?: "NEVER (15 s)"}, " +
                "tokens at action $tokensAtAction, at return $tokensAtReturn, 1.5 s later $tokensLater"
        )
        assertTrue("$label: reply never ended after the action", endedAt != null)
        assertEquals("$label: reply kept producing tokens after the action", tokensAtReturn, tokensLater)
        assertTrue("$label: main thread blocked ${hb.longestGap.get()} ms", hb.longestGap.get() < maxMainGapMs)
        return actionMs
    }

    // ---------------- llama.cpp ----------------

    @Test fun llamaUnloadDuringPromptEvaluationStopsFirst(): Unit = runBlocking {
        val model = model("LFM3 1.7B (Q4_0)"); requireFile(model)
        val service = LlamaCppInferenceService(context)
        assertTrue(service.loadModel(model, LlmInference.Backend.CPU, null))
        try {
            // Baseline: how long this prompt's evaluation takes before the first token.
            val base = startReply(service, model, "vitals7-base", longPrompt(SystemClock.uptimeMillis()))
            val start = SystemClock.elapsedRealtime()
            val firstAt = withTimeout(180_000) { base.firstTokenAt.await() }
            val prefillMs = firstAt - start
            service.resetChatSession("vitals7-base")
            withTimeout(30_000) { base.ended.await() }
            Log.i(tag, "RESULT llama.cpp prompt evaluation alone takes $prefillMs ms")
            assertTrue("prompt too short to test ($prefillMs ms)", prefillMs > 2_500)

            val reply = startReply(service, model, "vitals7-unload", longPrompt(SystemClock.uptimeMillis() + 1))
            delay(800) // inside prompt evaluation
            assertEquals(0, reply.tokens.get())
            val unloadMs = stopAndCheck("llama.cpp unload during prompt evaluation", reply, 300) { service.unloadModel() }
            assertTrue("unload waited ${unloadMs} ms of a ${prefillMs} ms prompt", unloadMs < prefillMs / 2)
            assertEquals("the stopped prompt still produced tokens", 0, reply.tokens.get())
        } finally {
            service.unloadModel()
        }
    }

    @Test fun llamaChatSwitchDuringPromptEvaluationStopsFirst(): Unit = runBlocking {
        val model = model("LFM3 1.7B (Q4_0)"); requireFile(model)
        val service = LlamaCppInferenceService(context)
        assertTrue(service.loadModel(model, LlmInference.Backend.CPU, null))
        val repository = (context.applicationContext as LlmHubApplication).chatRepository
        val chatA = repository.createNewChat("vitals7 switch A", model.name)
        val chatB = repository.createNewChat("vitals7 switch B", model.name)
        try {
            val reply = startReply(service, model, chatA, longPrompt(SystemClock.uptimeMillis()))
            delay(800)
            assertEquals(0, reply.tokens.get())
            // The saved current chat id makes the view model treat chatA as the chat being left.
            lateinit var viewModel: ChatViewModel
            instrumentation.runOnMainSync {
                viewModel = ChatViewModel(service, repository, context, SavedStateHandle(mapOf("current_chat_id" to chatA)))
            }
            // initializeChat's runBlocking reset runs on the main thread, as when the user picks another chat.
            stopAndCheck("llama.cpp chat switch during prompt evaluation", reply, 1500) {
                viewModel.initializeChat(chatB, context)
            }
        } finally {
            repository.deleteChat(chatA); repository.deleteChat(chatB)
            service.unloadModel()
        }
    }

    @Test fun llamaUnloadWhileGeneratingStopsFirst(): Unit = runBlocking {
        val model = model("LFM3 1.7B (Q4_0)"); requireFile(model)
        val service = LlamaCppInferenceService(context)
        assertTrue(service.loadModel(model, LlmInference.Backend.CPU, null))
        try {
            val reply = startReply(service, model, "vitals7-decode", story)
            withTimeout(120_000) { while (reply.tokens.get() < 30) delay(20) }
            stopAndCheck("llama.cpp unload while generating", reply, 300) { service.unloadModel() }
        } finally {
            service.unloadModel()
        }
    }

    /** Control: the same unload with nothing generating, to separate freeing the model from stopping a reply. */
    @Test fun llamaUnloadIdleMainThreadGap(): Unit = runBlocking {
        val model = model("LFM3 1.7B (Q4_0)"); requireFile(model)
        val service = LlamaCppInferenceService(context)
        repeat(3) { round ->
            assertTrue(service.loadModel(model, LlmInference.Backend.CPU, null))
            service.generateResponse("Say hi.", model)
            delay(500)
            val hb = Heartbeat(); hb.start()
            val t0 = SystemClock.elapsedRealtime()
            withContext(Dispatchers.Main) { service.unloadModel() }
            val ms = SystemClock.elapsedRealtime() - t0
            hb.stop()
            Log.i(tag, "RESULT llama.cpp idle unload round $round: action $ms ms, longest main-thread gap ${hb.longestGap.get()} ms")
        }
    }

    @Test fun llamaFlashStreamingUnloadWhileGeneratingStopsFirst(): Unit = runBlocking {
        val model = model("LFM2.5-24B-A2B (Q4_0)"); requireFile(model)
        val service = LlamaCppInferenceService(context)
        assertTrue(service.loadModel(model, LlmInference.Backend.CPU, null))
        try {
            val reply = startReply(service, model, "vitals7-flash", story)
            withTimeout(300_000) { while (reply.tokens.get() < 10) delay(20) }
            stopAndCheck("llama.cpp flash streaming unload while generating", reply, 300) { service.unloadModel() }
        } finally {
            service.unloadModel()
        }
    }

    // ---------------- MediaPipe ----------------

    @Test fun mediaPipeUnloadWhileGeneratingStopsFirstOffMainThread(): Unit = runBlocking {
        val model = model("Gemma-3 1B (INT4, 2k)"); requireFile(model)
        val service = MediaPipeInferenceService(context)
        assertTrue(service.loadModel(model, LlmInference.Backend.GPU, null))
        try {
            val reply = startReply(service, model, "vitals7-mp", story)
            withTimeout(120_000) { while (reply.tokens.get() < 30) delay(20) }
            stopAndCheck("MediaPipe unload while generating", reply, 300) { service.unloadModel() }
        } finally {
            service.unloadModel()
        }
    }

    // ---------------- ONNX ----------------

    @Test fun onnxUnloadWhileGeneratingStopsFirst(): Unit = runBlocking {
        val model = model("LFM-2.5 1.2B Instruct (ONNX Q4)")
        val dir = File(modelsDir, model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")).apply { mkdirs() }
        (listOf(model.url) + model.additionalFiles).forEach { download(it, File(dir, it.substringAfterLast('/').substringBefore('?'))) }
        val service = OnnxInferenceService(context)
        assertTrue(service.loadModel(model, LlmInference.Backend.CPU, null))
        try {
            val reply = startReply(service, model, "vitals7-onnx", story)
            withTimeout(180_000) { while (reply.tokens.get() < 20) delay(20) }
            stopAndCheck("ONNX unload while generating", reply, 300) { service.unloadModel() }
        } finally {
            service.unloadModel()
        }
    }

    // ---------------- LiteRT-LM ----------------

    @Test fun liteRtLmUnloadWhileGeneratingStopsFirst(): Unit = runBlocking {
        val model = model("Gemma-4 E2B"); requireFile(model)
        val service = LiteRtLmInferenceService(context)
        assertTrue(service.loadModel(model, LlmInference.Backend.GPU, null))
        try {
            val reply = startReply(service, model, "vitals7-litert", story)
            withTimeout(180_000) { while (reply.tokens.get() < 30) delay(20) }
            stopAndCheck("LiteRT-LM unload while generating", reply, 1000) { service.unloadModel() }
        } finally {
            service.unloadModel()
        }
    }

    private fun download(url: String, target: File) {
        if (target.isFile && target.length() > 0) return
        val partial = File(target.path + ".part")
        var attempt = 0
        while (true) {
            try {
                val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                    instanceFollowRedirects = true; connectTimeout = 30_000; readTimeout = 120_000
                    if (partial.length() > 0) setRequestProperty("Range", "bytes=${partial.length()}-")
                }
                val append = conn.responseCode == 206
                check(conn.responseCode in 200..299) { "HTTP ${conn.responseCode} for $url" }
                conn.inputStream.use { input -> java.io.FileOutputStream(partial, append).use { input.copyTo(it, 1 shl 20) } }
                partial.renameTo(target)
                return
            } catch (e: java.io.IOException) {
                if (++attempt >= 6) throw e
                Thread.sleep(5_000)
            }
        }
    }
}
