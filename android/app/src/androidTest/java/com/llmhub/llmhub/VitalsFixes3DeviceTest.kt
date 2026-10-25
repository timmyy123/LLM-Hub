package com.llmhub.llmhub

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RenderNode
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.HfDefaultToken
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.BoundedBitmapDecoder
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import com.llmhub.llmhub.inference.LlamaCppInferenceService
import com.llmhub.llmhub.mediasearch.VideoMomentViewModel
import com.llmhub.llmhub.ui.components.TtsService
import com.llmhub.llmhub.viewmodels.ChatViewModel
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicLong

/**
 * Opt-in physical-device checks for the 4.5.8 Play vitals fixes and the Hugging Face token fallback.
 * Run one at a time with `-e class com.llmhub.llmhub.VitalsFixes3DeviceTest#<name>`.
 */
class VitalsFixes3DeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    /** An expired primary token (HTTP 401) is replaced by HF_TOKEN_FALLBACK, which can read gated Gemma files. */
    @Test fun hfTokenFallbackTakesOver() {
        val field = HfDefaultToken::class.java.getDeclaredField("tokens").apply { isAccessible = true }
        @Suppress("UNCHECKED_CAST") val tokens = field.get(null) as MutableList<String>
        val bundled = tokens.toList()
        assertEquals("Both bundled tokens must be configured", 2, bundled.size)
        val rejected = "hf_" + "x".repeat(34)
        try {
            tokens[0] = rejected
            HfDefaultToken.verify()
            assertEquals("Fallback token must take over", BuildConfig.HF_TOKEN_FALLBACK, HfDefaultToken.current)
            assertTrue("A 401 on the rejected token reports the switch", HfDefaultToken.replaceRejected(rejected))
            val gated = ModelData.models.first { it.name == "Gemma-3n E2B" }.url
            val code = (URL(gated).openConnection() as HttpURLConnection).run {
                requestMethod = "HEAD"; instanceFollowRedirects = false
                setRequestProperty("Authorization", "Bearer ${HfDefaultToken.current}")
                responseCode.also { disconnect() }
            }
            Log.i(TAG, "RESULT fallback token selected, gated Gemma HEAD: HTTP $code")
            assertTrue("Fallback token cannot read the gated file: HTTP $code", code in 200..399)
        } finally {
            tokens.clear(); tokens.addAll(bundled)
            HfDefaultToken.verify()
        }
        assertEquals("Primary token is used again when it is valid", BuildConfig.HF_TOKEN, HfDefaultToken.current)
    }

    /** RecordingCanvas "trying to draw too large bitmap": a 50 MP photo must be decoded within the limit. */
    @Test fun largePhotoDrawsAfterBoundedDecode() {
        val photo = File(context.cacheDir, "vitals_50mp.jpg")
        Bitmap.createBitmap(8160, 6120, Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.rgb(40, 120, 200))
            photo.outputStream().use { compress(Bitmap.CompressFormat.JPEG, 80, it) }
            recycle()
        }
        val full = BitmapFactory.decodeFile(photo.path)
        val unbounded = runCatching { draw(full) }.exceptionOrNull()
        full.recycle()
        Log.i(TAG, "full-size 8160x6120 draw: ${unbounded?.message ?: "no error"}")
        assertTrue("Repro: the full-size photo should exceed the Canvas limit", unbounded is RuntimeException)

        val bounded = BoundedBitmapDecoder.decodeUri(context, Uri.fromFile(photo))!!
        draw(bounded)
        Log.i(TAG, "RESULT bounded decode ${bounded.width}x${bounded.height} (${bounded.byteCount / 1_000_000} MB) drew fine")
        assertTrue(maxOf(bounded.width, bounded.height) <= BoundedBitmapDecoder.PICKED_MAX_SIDE)
        assertEquals(Bitmap.Config.ARGB_8888, bounded.config)
        photo.delete()
    }

    private fun draw(bitmap: Bitmap) {
        val node = RenderNode("vitals").apply { setPosition(0, 0, 1000, 1000) }
        val canvas = node.beginRecording()
        try { canvas.drawBitmap(bitmap, null, android.graphics.Rect(0, 0, 1000, 1000), null) } finally { node.endRecording() }
    }

    /** ANR in ChatViewModel.initializeChat: switching chats while a reply is generating must not block the main thread. */
    @Test fun chatSwitchDuringReplyKeepsMainThreadFree(): Unit = runBlocking {
        val model = ModelData.models.first { it.name == "LFM-2.5 1.2B Instruct (Q4_0)" }
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        OnDeviceFixtures.download(
            file,
            "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_0.gguf",
            "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
        )
        val service = LlamaCppInferenceService(context)
        assertTrue("LFM must load", service.loadModel(model, LlmInference.Backend.CPU, null))
        val repository = (context.applicationContext as LlmHubApplication).chatRepository
        val chatA = repository.createNewChat("vitals switch A", model.name)
        val chatB = repository.createNewChat("vitals switch B", model.name)

        val firstToken = CompletableDeferred<Unit>()
        val reply = launch(Dispatchers.IO) {
            service.generateResponseStreamWithSession(
                "Write a very long, detailed story about a lighthouse keeper. At least 2000 words.",
                model, chatA, webSearchEnabled = false
            ).collect { firstToken.complete(Unit) }
        }
        withTimeout(120_000) { firstToken.await() }

        lateinit var viewModel: ChatViewModel
        var switchMs = 0L
        instrumentation.runOnMainSync {
            viewModel = ChatViewModel(service, repository, context, SavedStateHandle(mapOf("current_chat_id" to chatA)))
            val started = SystemClock.elapsedRealtime()
            viewModel.initializeChat(chatB, context) // waited for the whole reply before the fix
            switchMs = SystemClock.elapsedRealtime() - started
        }
        Log.i(TAG, "RESULT initializeChat during a reply returned in $switchMs ms")
        withTimeoutOrNull(30_000) { reply.cancelAndJoin() }
        Thread.sleep(3000)
        repository.deleteChat(chatA); repository.deleteChat(chatB)
        service.unloadModel()
        assertTrue("initializeChat blocked the main thread for $switchMs ms", switchMs < 1000)
    }

    /** OrtSession.run abort: TTS shutdown and settings changes while Kokoro is mid-sentence must not crash. */
    @Test fun kokoroCloseWhileSpeakingDoesNotCrash(): Unit = runBlocking {
        val model = ModelData.models.first { it.name == "Kokoro-82M (ONNX Quantized)" }
        val dir = File(File(context.filesDir, "models"), model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")).apply { mkdirs() }
        download(model.url, File(dir, model.localFileName()))
        model.additionalFiles.forEach { download(it, File(dir, it.substringAfterLast('/').substringBefore('?'))) }
        download(model.url.substringBefore("/onnx/") + "/voices/af_sky.bin", File(dir, "af_sky.bin"))

        val preferences = ThemePreferences(context)
        val previousModel = preferences.selectedTtsModel.first()
        val previousDevice = preferences.selectedTtsDevice.first()
        preferences.setSelectedTtsModel(model.name)
        preferences.setSelectedTtsDevice("cpu")
        preferences.setSelectedTtsVoice("af_sky")
        val text = "The old lighthouse stood at the edge of the cliff, its light sweeping across the dark and restless sea, " +
            "while the keeper wrote long letters to a city he had not seen in many years, describing every storm in careful detail."
        try {
            repeat(4) { round ->
                lateinit var tts: TtsService
                instrumentation.runOnMainSync { tts = TtsService(context) }
                val sentencesBefore = ttsLog("sentence=").size
                instrumentation.runOnMainSync { tts.speak(text) }
                withTimeout(60_000) { tts.isSpeaking.first { it } }
                // The synthesis worker logs "sentence=" right before tokenizing and calling session.run.
                val deadline = SystemClock.elapsedRealtime() + 60_000
                while (ttsLog("sentence=").size <= sentencesBefore) {
                    check(SystemClock.elapsedRealtime() < deadline) { "Kokoro never started a sentence" }
                    Thread.sleep(20)
                }
                Thread.sleep(100L + round * 350L) // inside session.run for the first sentence
                if (round % 2 == 0) {
                    instrumentation.runOnMainSync { tts.shutdown() } // aborted in OrtSession.run before the fix
                } else {
                    preferences.setSelectedTtsDevice("gpu") // settings change closes the session too
                    Thread.sleep(2000)
                    preferences.setSelectedTtsDevice("cpu")
                    Thread.sleep(1000)
                    instrumentation.runOnMainSync { tts.shutdown() }
                }
                Thread.sleep(4000)
                Log.i(TAG, "kokoro round $round: process alive after closing mid-sentence")
            }
            ttsLog("Generated ").forEach { Log.i(TAG, "kokoro synthesis: ${it.substringAfter("TtsService: ")}") }
            Log.i(TAG, "RESULT kokoro: 4 closes while speaking, no crash")
        } finally {
            preferences.setSelectedTtsModel(previousModel)
            preferences.setSelectedTtsDevice(previousDevice)
        }
    }

    /** ANR in Video Moments: saved videos behind a slow content provider must not block the main thread. */
    @Test fun videoMomentsSlowProviderKeepsMainThreadFree() {
        val prefs = context.getSharedPreferences("video_moment_prefs", android.content.Context.MODE_PRIVATE)
        val saved = prefs.getString("picked_video_uris", null)
        prefs.edit().putString("picked_video_uris", (1..2).joinToString("\n") { "content://${SlowVideoProvider.AUTHORITY}/clip$it.mp4" }).commit()

        val handler = Handler(Looper.getMainLooper())
        val last = AtomicLong(SystemClock.elapsedRealtime())
        val gap = AtomicLong(0)
        var beating = true
        val beat = object : Runnable {
            override fun run() {
                val now = SystemClock.elapsedRealtime()
                gap.set(maxOf(gap.get(), now - last.get())); last.set(now)
                if (beating) handler.postDelayed(this, 16)
            }
        }
        handler.post(beat)
        val store = ViewModelStore()
        try {
            instrumentation.runOnMainSync {
                ViewModelProvider(store, ViewModelProvider.AndroidViewModelFactory.getInstance(context.applicationContext as Application))[VideoMomentViewModel::class.java]
            }
            Thread.sleep(15_000) // 2 videos x (query + open) x 3 s in the provider
        } finally {
            beating = false
            instrumentation.runOnMainSync { store.clear() }
            prefs.edit().putString("picked_video_uris", saved).commit()
        }
        Log.i(TAG, "RESULT video moments: longest main-thread gap ${gap.get()} ms with a 3 s provider")
        assertTrue("Main thread blocked ${gap.get()} ms loading saved videos", gap.get() < 1000)
    }

    /** This process's TtsService log lines containing [marker] (an app may read its own log). */
    private fun ttsLog(marker: String): List<String> =
        ProcessBuilder("logcat", "-d", "-s", "TtsService:D", "--pid", android.os.Process.myPid().toString())
            .redirectErrorStream(true).start().inputStream.bufferedReader().readLines().filter { marker in it }

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

    private companion object { const val TAG = "VitalsFixes3DeviceTest" }
}
