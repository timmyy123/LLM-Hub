package com.llmhub.llmhub.soundgen

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.MagentaRealtime
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.ModelDownloader
import com.llmhub.llmhub.data.MusicGeneratorBackend
import com.llmhub.llmhub.data.hasCompleteDownloadedBundle
import com.llmhub.llmhub.service.MagentaLiveService
import io.ktor.client.HttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

/**
 * Opt-in physical-device check for Magenta RealTime 2 on a supported Snapdragon (8 Elite Gen 5 / Gen 6). Downloads
 * the model through the app's ModelDownloader (kept afterwards), renders a clip, then runs live generation through
 * MagentaLiveService for `-e liveSeconds N` (default 60) while the host may lock the screen. Every 5 s it logs how
 * much audio has played under the MagentaRealtimeDeviceTest tag; live must keep up with real time.
 */
class MagentaRealtimeDeviceTest {
    @Test fun rendersAndPlaysLive() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val liveSeconds = args.getString("liveSeconds")?.toIntOrNull() ?: 60
        val model = ModelData.musicGenerationModels.firstOrNull { MagentaRealtime.isModel(it.name) }
        assertTrue("Magenta RealTime 2 is not offered on ${Build.SOC_MODEL}", model != null)
        model!!

        if (!model.hasCompleteDownloadedBundle(context)) {
            val started = SystemClock.elapsedRealtime()
            HttpClient(Android).use { client -> ModelDownloader(client, context).downloadModel(model).collect { } }
            Log.i(TAG, "download ${SystemClock.elapsedRealtime() - started} ms")
        }
        assertTrue("download incomplete", model.hasCompleteDownloadedBundle(context))

        // Render first, then play (10 s).
        var started = SystemClock.elapsedRealtime()
        assertTrue("load failed", MusicGeneratorBackend.loadModel(context, model.name))
        Log.i(TAG, "load ${SystemClock.elapsedRealtime() - started} ms")
        started = SystemClock.elapsedRealtime()
        val wav = MusicGeneratorBackend.generateMusic(context, model.name, "lo-fi hip hop with warm piano", 10.0) { }
        assertTrue("render failed", wav != null)
        val pcm = MagentaRealtime.readPcm(wav!!)
        val samples = java.nio.ByteBuffer.wrap(pcm).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        var sum = 0.0
        while (samples.hasRemaining()) { val v = samples.get() / 32768.0; sum += v * v }
        val rms = sqrt(sum / (pcm.size / 2))
        Log.i(TAG, "render 10 s in ${SystemClock.elapsedRealtime() - started} ms, frames ${pcm.size / 4}, rms $rms")
        wav.delete()
        assertEquals(10 * MagentaRealtime.SAMPLE_RATE, pcm.size / 4)
        assertTrue("silent render", rms > 1e-3)
        MusicGeneratorBackend.unloadModel()

        // Live: unlimited, stopped by the test. Started from the foreground, as the app does from its screen
        // (Android does not let a background process start a foreground service).
        context.startActivity(
            android.content.Intent(context, com.llmhub.llmhub.MainActivity::class.java)
                .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
        )
        SystemClock.sleep(3000)
        started = SystemClock.elapsedRealtime()
        MagentaLiveService.start(context, "cinematic strings with soft percussion", maxSeconds = null)
        var lastLog = 0L
        while (SystemClock.elapsedRealtime() - started < liveSeconds * 1000L) {
            val state = MagentaLiveService.state.value
            if (state.error != null) break
            val elapsed = SystemClock.elapsedRealtime() - started
            if (elapsed - lastLog >= 5000) {
                lastLog = elapsed
                Log.i(TAG, "live elapsed=${elapsed / 1000}s played=${state.playedSeconds}s starting=${state.starting} paused=${state.paused}")
            }
            if (elapsed in 30_000..30_999) MagentaLiveService.updatePrompt("energetic synthwave with driving drums")
            SystemClock.sleep(200)
        }
        val final = MagentaLiveService.state.value
        Log.i(TAG, "live end played=${final.playedSeconds}s error=${final.error}")
        MagentaLiveService.stop(context)
        SystemClock.sleep(2000)
        assertNull("live error", final.error)
        // After start-up, playback must keep pace with the wall clock.
        assertTrue("live fell behind: played ${final.playedSeconds}s of $liveSeconds", final.playedSeconds >= liveSeconds - 20)
    }

    private companion object { const val TAG = "MagentaRealtimeDeviceTest" }
}
