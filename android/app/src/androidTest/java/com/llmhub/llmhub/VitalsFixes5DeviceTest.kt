package com.llmhub.llmhub

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.ClickableText
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.LocalTextStyle
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.agent.AgentMessage
import com.llmhub.llmhub.agent.AgentViewModel
import com.llmhub.llmhub.components.SelectableMarkdownText
import com.llmhub.llmhub.components.annotateLinksAndPhones
import androidx.compose.foundation.text.selection.SelectionContainer
import com.llmhub.llmhub.components.parseMarkdownToAnnotatedString
import com.llmhub.llmhub.components.splitMarkdownIntoChunks
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.service.SDBackendService
import com.llmhub.llmhub.viewmodels.ModelDownloadViewModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicLong

/**
 * Opt-in physical-device checks for the 4.6.5 Play vitals fixes. Run one at a time with
 * `-e class com.llmhub.llmhub.VitalsFixes5DeviceTest#<name>`.
 */
class VitalsFixes5DeviceTest {
    @get:Rule val compose = createComposeRule()

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    /** ANR in ModelDownloadScreen: creating the ViewModel must not scan model files on the main thread. */
    @Test fun modelListLoadsOffMainThread() {
        // The first construction also loads the catalog classes; the second shows the per-open cost.
        val firstStore = ViewModelStore()
        var firstMs = 0L
        instrumentation.runOnMainSync {
            val started = SystemClock.elapsedRealtime()
            ViewModelProvider(
                firstStore, ViewModelProvider.AndroidViewModelFactory.getInstance(context.applicationContext as Application)
            )[ModelDownloadViewModel::class.java]
            firstMs = SystemClock.elapsedRealtime() - started
        }
        Thread.sleep(3_000)
        instrumentation.runOnMainSync { firstStore.clear() }
        val store = ViewModelStore()
        var viewModel: ModelDownloadViewModel? = null
        var constructMs = 0L
        instrumentation.runOnMainSync {
            val started = SystemClock.elapsedRealtime()
            viewModel = ViewModelProvider(
                store, ViewModelProvider.AndroidViewModelFactory.getInstance(context.applicationContext as Application)
            )[ModelDownloadViewModel::class.java]
            constructMs = SystemClock.elapsedRealtime() - started
        }
        val deadline = SystemClock.elapsedRealtime() + 30_000
        while (viewModel!!.models.value.isEmpty()) {
            check(SystemClock.elapsedRealtime() < deadline) { "Model list never loaded" }
            Thread.sleep(50)
        }
        val models = viewModel!!.models.value
        Log.i(TAG, "RESULT model list: first construction $firstMs ms, second $constructMs ms on the main thread, ${models.size} models, " +
            "${models.count { it.isDownloaded }} downloaded")
        instrumentation.runOnMainSync { store.clear() }
        assertTrue("ViewModel construction took $constructMs ms on the main thread", constructMs < 100)
        assertTrue("Catalog models missing", models.size >= ModelData.models.size)
    }

    /** Chunks stacked in a Column must be exactly as tall as the same text laid out as one Text. */
    @Test fun chunkedMarkdownLooksLikeOneText() {
        val text = longReply(paragraphs = 40)
        val chunks = splitMarkdownIntoChunks(text)
        assertTrue("Expected several chunks, got ${chunks.size}", chunks.size > 3)
        assertEquals("Chunks must rebuild the text", text, chunks.joinToString("\n"))

        var wholeHeight = 0
        var chunkedHeight = 0
        compose.setContent {
            val style = LocalTextStyle.current.copy(fontSize = 16.sp, lineHeight = 16.sp * 1.4)
            Column(Modifier.verticalScroll(rememberScrollState())) {
                ClickableText(
                    parseMarkdownToAnnotatedString(text, Color.Black, Color.Blue), style = style,
                    modifier = Modifier.fillMaxWidth().onSizeChanged { wholeHeight = it.height }, onClick = {}
                )
                Column(Modifier.fillMaxWidth().onSizeChanged { chunkedHeight = it.height }) {
                    chunks.forEach {
                        ClickableText(
                            parseMarkdownToAnnotatedString(it, Color.Black, Color.Blue, normalize = false),
                            style = style, modifier = Modifier.fillMaxWidth(), onClick = {}
                        )
                    }
                }
            }
        }
        compose.waitForIdle()
        Log.i(TAG, "RESULT chunks: ${chunks.size} chunks, one Text $wholeHeight px, chunked $chunkedHeight px")
        assertEquals("Chunked layout height differs from one Text", wholeHeight, chunkedHeight)
    }

    /**
     * ANRs in text layout: a long reply streamed into the chat bubble at the throttled rate must keep the
     * main thread responsive. A main-thread heartbeat records the longest gap.
     */
    @Test fun streamingLongReplyKeepsMainThreadResponsive() {
        val gap = streamAndMeasure("chunked") { text ->
            SelectableMarkdownText(markdown = text, color = Color.Black, fontSize = 16.sp, modifier = Modifier.fillMaxWidth())
        }
        assertTrue("Main thread blocked for $gap ms while streaming", gap < 500)
    }

    /** Baseline for the test above: the 4.6.5 renderer, one Text for the whole reply. */
    @Test fun streamingLongReplyBaselineOneText() {
        streamAndMeasure("one Text (4.6.5)") { text ->
            val annotated = androidx.compose.runtime.remember(text) {
                annotateLinksAndPhones(parseMarkdownToAnnotatedString(text, Color.Black, Color.Blue), Color.Blue)
            }
            SelectionContainer {
                ClickableText(
                    annotated, modifier = Modifier.fillMaxWidth(),
                    style = LocalTextStyle.current.copy(fontSize = 16.sp, lineHeight = 16.sp * 1.4), onClick = {}
                )
            }
        }
    }

    private fun streamAndMeasure(label: String, content: @androidx.compose.runtime.Composable (String) -> Unit): Long {
        val full = longReply(paragraphs = 60)
        var shown by mutableStateOf("")
        compose.setContent {
            Column(Modifier.verticalScroll(rememberScrollState())) { content(shown) }
        }
        val handler = Handler(Looper.getMainLooper())
        val last = AtomicLong(SystemClock.elapsedRealtime())
        val longestGap = AtomicLong(0)
        val slowFrames = java.util.concurrent.CopyOnWriteArrayList<String>()
        var length = 0
        var running = true
        val beat = object : Runnable {
            override fun run() {
                val now = SystemClock.elapsedRealtime()
                val gap = now - last.get()
                if (gap > 100) slowFrames += "${gap}ms@$length"
                longestGap.set(maxOf(longestGap.get(), gap))
                last.set(now)
                if (running) handler.postDelayed(this, 16)
            }
        }
        handler.post(beat)
        // About 40 characters per 100 ms update, like a fast model behind the 100 ms throttle.
        val started = SystemClock.elapsedRealtime()
        while (length < full.length) {
            length = minOf(full.length, length + 40)
            val next = full.substring(0, length)
            instrumentation.runOnMainSync { shown = next }
            compose.waitForIdle() // the test clock only draws frames when the test syncs
            Thread.sleep(100)
        }
        compose.waitForIdle()
        running = false
        Thread.sleep(100)
        Log.i(TAG, "RESULT streaming $label: ${full.length} chars in ${SystemClock.elapsedRealtime() - started} ms, " +
            "longest main-thread gap ${longestGap.get()} ms, gaps over 100 ms (${slowFrames.size}): ${slowFrames.take(40)}")
        return longestGap.get()
    }

    /**
     * SDBackendService: stopping must not call startForeground (ForegroundServiceStartNotAllowedException
     * from the background), whether or not the service is running.
     */
    @Test fun sdBackendStopDoesNotStartForeground() {
        SDBackendService.stop(context) // not running
        Thread.sleep(2_000)
        assertFalse("Stop started a foreground service", sdBackendRunning())

        SDBackendService.startUpscaler(context)
        val deadline = SystemClock.elapsedRealtime() + 15_000
        while (!sdBackendRunning()) {
            check(SystemClock.elapsedRealtime() < deadline) { "SDBackendService did not start" }
            Thread.sleep(200)
        }
        SDBackendService.stop(context)
        val stopDeadline = SystemClock.elapsedRealtime() + 15_000
        while (sdBackendRunning() && SystemClock.elapsedRealtime() < stopDeadline) Thread.sleep(200)
        Log.i(TAG, "RESULT sd backend: stop while idle and while running, running now = ${sdBackendRunning()}")
        assertFalse("SDBackendService still running after stop", sdBackendRunning())
    }

    /**
     * Uncaught LiteRtLmJniException in the agent: a prompt longer than the context window makes LiteRT-LM
     * report an error through the web search stream. The agent must show the context-full message, not crash.
     */
    @Test fun agentInferenceErrorDoesNotCrash() = agentContextOverflowShowsError(webSearch = true)

    /** Without web search the same error used to remove the empty reply, so nothing was shown. */
    @Test fun agentContextOverflowWithoutWebSearchShowsError() = agentContextOverflowShowsError(webSearch = false)

    private fun agentContextOverflowShowsError(webSearch: Boolean) {
        val model = ModelData.models.first { it.name == "Gemma-4 E2B" }
        download(model.url, File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName()))
        context.getSharedPreferences("agent_prefs", Context.MODE_PRIVATE).edit()
            .putString("selected_model_name", model.name)
            .putInt("selected_max_tokens", 1024)
            .commit()

        val store = ViewModelStore()
        var viewModel: AgentViewModel? = null
        instrumentation.runOnMainSync {
            viewModel = ViewModelProvider(
                store, ViewModelProvider.AndroidViewModelFactory.getInstance(context.applicationContext as Application)
            )[AgentViewModel::class.java]
            if (webSearch != viewModel!!.isWebSearchEnabled.value) viewModel!!.toggleWebSearch()
            viewModel!!.sendMessage("Summarize this. " + "The quick brown fox jumps over the lazy dog. ".repeat(1500))
        }
        val deadline = SystemClock.elapsedRealtime() + 300_000
        Thread.sleep(1_000)
        while (viewModel!!.isGenerating.value) {
            check(SystemClock.elapsedRealtime() < deadline) { "Agent never finished" }
            Thread.sleep(500)
        }
        val replies = viewModel!!.messages.value.filterIsInstance<AgentMessage.Text>()
            .filter { it.sender == AgentMessage.Sender.AGENT }.map { it.text }
        Log.i(TAG, "RESULT agent (web search $webSearch): process alive, agent replies = ${replies.map { it.take(160) }}")
        instrumentation.runOnMainSync { store.clear() }
        context.getSharedPreferences("agent_prefs", Context.MODE_PRIVATE).edit().remove("selected_max_tokens").commit()
        val expected = context.getString(R.string.context_window_full_error)
        assertTrue("No context-full message in $replies", replies.any { it.contains(expected) })
    }

    private fun longReply(paragraphs: Int): String = (1..paragraphs).joinToString("\n\n") { i ->
        when (i % 4) {
            0 -> "### Section $i\n- **Point** one with a link https://example.com/$i\n- Point *two* with `code`"
            else -> "Paragraph $i. " + "Streaming text should stay smooth on slower phones while the model writes. ".repeat(4)
        }
    }

    @Suppress("DEPRECATION")
    private fun sdBackendRunning(): Boolean =
        (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getRunningServices(100)
            .any { it.service.className == SDBackendService::class.java.name && it.foreground }

    // Resumable download with retries; skips files that are already complete.
    private fun download(url: String, target: File) {
        var attempt = 0
        while (true) {
            try {
                var have = if (target.exists()) target.length() else 0L
                val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                    instanceFollowRedirects = true
                    connectTimeout = 30_000
                    readTimeout = 120_000
                    if (have > 0) setRequestProperty("Range", "bytes=$have-")
                }
                when (conn.responseCode) {
                    416 -> return
                    200 -> have = 0L
                }
                conn.inputStream.use { input -> java.io.FileOutputStream(target, have > 0).use { input.copyTo(it, 1 shl 20) } }
                Log.i(TAG, "downloaded ${target.name}: ${target.length()} bytes")
                return
            } catch (e: java.io.IOException) {
                if (++attempt >= 6) throw e
                Thread.sleep(5_000)
            }
        }
    }

    private companion object { const val TAG = "VitalsFixes5DeviceTest" }
}
