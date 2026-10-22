package com.llmhub.llmhub

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.inference.LiteRtLmInferenceService
import com.llmhub.llmhub.videogen.VideoGenEngine
import com.llmhub.llmhub.videogen.VideoGenProgress
import com.llmhub.llmhub.viewmodels.VideoGeneratorViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicLong

/**
 * Opt-in physical-device checks for the Play vitals fixes in 4.5.6. Models are downloaded on the
 * device from their pinned catalog links (resumable). Run one test at a time with
 * `-e class com.llmhub.llmhub.VitalsFixesDeviceTest#<name>`.
 */
class VitalsFixesDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    /**
     * ANR in LiteRtLmJni.nativeDeleteEngine: unloadModel() called from the main thread must not
     * block it. A main-thread heartbeat records the longest gap while the engine is freed.
     */
    @Test fun liteRtUnloadKeepsMainThreadResponsive() = runBlocking {
        val model = ModelData.models.first { it.name == "Gemma-4 E2B" }
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        download(model.url, file)
        val service = LiteRtLmInferenceService(context)
        assertTrue("Gemma-4 E2B must load", service.loadModel(model, LlmInference.Backend.CPU, null))

        val handler = Handler(Looper.getMainLooper())
        val last = AtomicLong(SystemClock.elapsedRealtime())
        val longestGap = AtomicLong(0)
        var running = true
        val beat = object : Runnable {
            override fun run() {
                val now = SystemClock.elapsedRealtime()
                longestGap.set(maxOf(longestGap.get(), now - last.get()))
                last.set(now)
                if (running) handler.postDelayed(this, 16)
            }
        }
        handler.post(beat)
        val started = SystemClock.elapsedRealtime()
        withContext(Dispatchers.Main) { service.unloadModel() }
        val unloadMs = SystemClock.elapsedRealtime() - started
        running = false
        Thread.sleep(100)
        Log.i(TAG, "RESULT liteRt unload ${unloadMs} ms, longest main-thread gap ${longestGap.get()} ms")
        assertTrue("Main thread blocked for ${longestGap.get()} ms during unload", longestGap.get() < 1000)
    }

    /**
     * Crash in SdCppNative.nativeFree (GGML_ASSERT(!graph_active_)): clearing the Video Generator
     * ViewModel mid-generation (leaving the screen) must wait for the native call, then unload.
     */
    @Test fun videoLeaveMidGenerationDoesNotCrash() {
        val model = ModelData.videoGenerationModels.first { "v73-int" in it.name }
        val dir = VideoGenEngine.modelDir(context, model).apply { mkdirs() }
        for (url in listOf(model.url) + model.additionalFiles) {
            download(url, File(dir, url.substringAfterLast('/').substringBefore('?')))
        }
        val store = ViewModelStore()
        lateinit var viewModel: VideoGeneratorViewModel
        instrumentation.runOnMainSync {
            viewModel = ViewModelProvider(
                store,
                ViewModelProvider.AndroidViewModelFactory.getInstance(context.applicationContext as Application)
            )[VideoGeneratorViewModel::class.java]
            viewModel.selectModel(model)
            viewModel.prompt = "A red vintage sports car drives along a winding coastal road at sunset, cinematic."
            viewModel.generate()
        }
        // Leave once denoising has started, while the native generation call is running.
        val deadline = SystemClock.elapsedRealtime() + 10 * 60_000
        while (viewModel.progress.value !is VideoGenProgress.Denoising) {
            check(SystemClock.elapsedRealtime() < deadline) { "Generation never reached denoising: ${viewModel.progress.value}" }
            Thread.sleep(200)
        }
        Log.i(TAG, "leaving Video Generator at ${viewModel.progress.value}")
        instrumentation.runOnMainSync { store.clear() }
        // Before the fix the process aborted right here. Give the generation time to finish and unload.
        Thread.sleep(90_000)
        Log.i(TAG, "RESULT video: process alive 90 s after leaving mid-generation")
    }

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

    private companion object { const val TAG = "VitalsFixesDeviceTest" }
}
