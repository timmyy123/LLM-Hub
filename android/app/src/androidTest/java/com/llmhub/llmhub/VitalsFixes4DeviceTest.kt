package com.llmhub.llmhub

import android.app.ActivityManager
import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.inference.LiteRtLmInferenceService
import com.llmhub.llmhub.service.SDBackendService
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device checks for the 4.5.9 Play vitals fixes. Run one at a time with
 * `-e class com.llmhub.llmhub.VitalsFixes4DeviceTest#<name>`. The Gemma-4 E2B file must be in the
 * app's models folder (pushed by the runner or downloaded by the app).
 */
class VitalsFixes4DeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    /**
     * SIGSEGV in liblitertlm_jni on a LiteRT-LM worker thread: stopping a reply closed the Conversation
     * while the worker still generated into it, and unloading closed the engine mid-reply.
     */
    @Test fun liteRtStopAndUnloadMidReplyDoNotCrash(): Unit = runBlocking {
        val model = ModelData.models.first { it.name == "Gemma-4 E2B" }
        val file = File(File(context.filesDir, "models"), model.localFileName())
        assertTrue("Push ${file.name} into files/models first", file.isFile && file.length() == model.sizeBytes)
        val service = LiteRtLmInferenceService(context)
        assertTrue("Gemma-4 E2B must load", service.loadModel(model, LlmInference.Backend.CPU, null))
        val prompt = "Write a very long story about a lighthouse keeper, at least 1500 words."

        repeat(6) { round ->
            val firstChunk = CompletableDeferred<Unit>()
            val reply = launch(Dispatchers.IO) {
                service.generateResponseStreamWithSession(prompt, model, "vitals-stop-$round", webSearchEnabled = false)
                    .collect { firstChunk.complete(Unit) }
            }
            withTimeout(120_000) { firstChunk.await() }
            Thread.sleep(200L + round * 300L)
            val started = SystemClock.elapsedRealtime()
            withTimeout(30_000) { reply.cancelAndJoin() } // Stop button / leaving the chat
            Log.i(TAG, "stop round $round: reply stopped in ${SystemClock.elapsedRealtime() - started} ms")
        }

        repeat(2) { round ->
            val firstChunk = CompletableDeferred<Unit>()
            val reply = launch(Dispatchers.IO) {
                runCatching {
                    service.generateResponseStreamWithSession(prompt, model, "vitals-unload-$round", webSearchEnabled = false)
                        .collect { firstChunk.complete(Unit) }
                }
            }
            withTimeout(120_000) { firstChunk.await() }
            Thread.sleep(500)
            val started = SystemClock.elapsedRealtime()
            service.unloadModel() // closed the engine under the worker thread before the fix
            Log.i(TAG, "unload round $round: engine closed mid-reply in ${SystemClock.elapsedRealtime() - started} ms")
            withTimeout(30_000) { reply.join() }
            assertTrue("Gemma-4 E2B must load again", service.loadModel(model, LlmInference.Backend.CPU, null))
        }

        val answer = service.generateResponse("What is the capital of France? Answer in one word.", model)
        Log.i(TAG, "RESULT litert: 6 stops and 2 unloads mid-reply, then answered: ${answer.take(80)}")
        assertTrue("Unexpected answer after the stops: $answer", answer.contains("Paris", ignoreCase = true))
        service.unloadModel()
    }

    /**
     * ForegroundServiceDidNotStopInTimeException: SDBackendService must stop itself in onTimeout.
     * Run after `adb shell device_config put activity_manager data_sync_fgs_timeout_duration 30000`.
     */
    @Test fun sdBackendStopsWhenDataSyncTimesOut() {
        SDBackendService.startUpscaler(context)
        val deadline = SystemClock.elapsedRealtime() + 15_000
        while (!sdBackendRunning()) {
            check(SystemClock.elapsedRealtime() < deadline) { "SDBackendService did not start" }
            Thread.sleep(200)
        }
        Log.i(TAG, "SDBackendService running, waiting for the dataSync timeout")
        val stopDeadline = SystemClock.elapsedRealtime() + 90_000
        while (sdBackendRunning() && SystemClock.elapsedRealtime() < stopDeadline) Thread.sleep(500)
        val stopped = !sdBackendRunning()
        Thread.sleep(10_000) // the system kills the app here if the service ignored the timeout
        Log.i(TAG, "RESULT sd backend: stopped after the timeout = $stopped, process alive")
        if (!stopped) SDBackendService.stop(context)
        assertFalse("SDBackendService kept running past its dataSync timeout", !stopped)
    }

    @Suppress("DEPRECATION")
    private fun sdBackendRunning(): Boolean =
        (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getRunningServices(100)
            .any { it.service.className == SDBackendService::class.java.name && it.foreground }

    private companion object { const val TAG = "VitalsFixes4DeviceTest" }
}
