package com.llmhub.llmhub.inference

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check for the 4.7.0 (187) vitals crash: SIGSEGV in LiteRtDestroyTensorBuffer under
 * Conversation.close (Motorola edge 50 neo, Dimensity 7300). Unloading the engine while a reply is running
 * (leaving the chat, switching models, Stop then leave) could close the engine while the reply's Conversation was
 * still being freed. Each round loads Gemma-4 E2B through [UnifiedInferenceService], starts a reply and unloads
 * part way through it. A native crash ends the instrumentation process; results go to logcat (tag LiteRtLmCloseRace).
 */
class LiteRtLmCloseRaceDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun unloadWhileGeneratingGpu() = race(LlmInference.Backend.GPU, cancelFirst = false)
    @Test fun stopThenUnloadGpu() = race(LlmInference.Backend.GPU, cancelFirst = true)
    @Test fun unloadWhileGeneratingCpu() = race(LlmInference.Backend.CPU, cancelFirst = false)
    @Test fun stopThenUnloadCpu() = race(LlmInference.Backend.CPU, cancelFirst = true)

    private fun race(backend: LlmInference.Backend, cancelFirst: Boolean): Unit = runBlocking {
        val label = "${backend.name} ${if (cancelFirst) "stop-then-unload" else "unload-while-generating"}"
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${Build.SOC_MODEL} sdk=${Build.VERSION.SDK_INT} case=$label")
        val model: LLMModel = ModelData.models.first { it.name == "Gemma-4 E2B" }
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        OnDeviceFixtures.download(file, GEMMA4_URL, GEMMA4_SHA256)
        val service = UnifiedInferenceService(context)
        val scope = CoroutineScope(Dispatchers.Default)
        for ((round, delayMs) in DELAYS_MS.withIndex()) {
            val loadStart = SystemClock.elapsedRealtime()
            assertTrue("$label round $round did not load", service.loadModel(model, backend, null))
            Log.i(TAG, "ROUND $label $round loaded in ${SystemClock.elapsedRealtime() - loadStart} ms, unload after $delayMs ms")
            var chunks = 0
            val reply = scope.launch {
                runCatching {
                    service.generateResponseStreamWithSession(
                        "Write a 400 word story about a lighthouse keeper.", model, "race-$round",
                        emptyList(), null, false, emptyList(),
                    ).collect { chunks++ }
                }
            }
            delay(delayMs)
            if (cancelFirst) reply.cancel() // the chat's Stop, then leaving the screen without waiting
            service.unloadModel()
            reply.join()
            Log.i(TAG, "ROUND $label $round ok: $chunks chunks before unload")
        }
        Log.i(TAG, "RESULT $label passed ${DELAYS_MS.size} rounds")
    }

    private companion object {
        const val TAG = "LiteRtLmCloseRace"
        val DELAYS_MS = longArrayOf(150, 400, 700, 1000, 1500, 2000, 3000, 250, 600, 1200)
        const val GEMMA4_URL = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/a4a831c060880f3733135ad22f10e0e9f758f45d/gemma-4-E2B-it.litertlm"
        const val GEMMA4_SHA256 = "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
    }
}
