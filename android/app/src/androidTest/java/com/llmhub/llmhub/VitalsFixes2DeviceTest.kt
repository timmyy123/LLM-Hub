package com.llmhub.llmhub

import android.content.Intent
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.llmhub.llmhub.data.DownloadStatus
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.service.ModelDownloadService
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device checks for the 4.5.7 Play vitals fixes. Run one at a time with
 * `-e class com.llmhub.llmhub.VitalsFixes2DeviceTest#<name>`.
 */
class VitalsFixes2DeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    /** SQLiteConstraintException (FOREIGN KEY) when a reply was saved into a chat deleted meanwhile. */
    @Test fun messageForDeletedChatIsDropped() = runBlocking {
        val repository = (context.applicationContext as LlmHubApplication).chatRepository
        val chatId = repository.createNewChat("vitals test", "test-model")
        repository.addMessage(chatId, "kept", isFromUser = true)
        repository.deleteChat(chatId)
        repository.addMessage(chatId, "reply after delete", isFromUser = false) // threw before the fix
        val left = repository.getMessagesForChatSync(chatId)
        Log.i(TAG, "RESULT deleted-chat insert dropped, messages left: ${left.size}")
        assertTrue("No message may be stored for a deleted chat", left.isEmpty())
    }

    /** Download progress bursts go through the notification throttle (ANR in NotificationManager.notify). */
    @Test fun progressBurstIsThrottled() {
        val started = System.nanoTime()
        repeat(5_000) { i ->
            ModelDownloadService.reportProgress("vitals-test", DownloadStatus(downloadedBytes = i * 1000L, totalBytes = 5_000_000L, downloadSpeedBytesPerSec = 1_000_000L))
        }
        val ms = (System.nanoTime() - started) / 1_000_000
        Log.i(TAG, "RESULT 5000 progress reports took $ms ms")
        assertTrue("Progress reporting is too slow: $ms ms", ms < 2_000)
    }

    /** Compose 1.12 smoke test: Home, AI Chat with typing in the message field, Settings; screenshots. */
    @Test fun composeUiSmoke() {
        val device = UiDevice.getInstance(instrumentation)
        val out = File(context.getExternalFilesDir(null), "vitals_ui").apply { mkdirs() }
        val launch = context.packageManager.getLaunchIntentForPackage(context.packageName)!!
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(launch)
        assertNotNull("Home did not appear", device.wait(Until.findObject(By.text(context.getString(R.string.chat_now))), 20_000))
        Thread.sleep(1500)
        device.takeScreenshot(File(out, "1_home.png"))

        device.findObject(By.text(context.getString(R.string.chat_now))).click()
        val input = device.wait(Until.findObject(By.clazz("android.widget.EditText")), 20_000)
        assertNotNull("Chat input did not appear", input)
        input.click()
        Thread.sleep(800)
        input.text = "Hello from the Compose 1.12 smoke test"
        Thread.sleep(1500)
        device.takeScreenshot(File(out, "2_chat_typing.png"))
        device.pressBack() // keyboard
        Thread.sleep(500)
        device.pressBack() // leave chat: the text field detaches while the IME may still query it
        Thread.sleep(1500)

        val settings = device.wait(Until.findObject(By.desc("Settings")), 5_000)
            ?: device.findObject(By.descContains("etting"))
        settings?.click()
        Thread.sleep(2000)
        device.takeScreenshot(File(out, "3_settings.png"))
        device.pressBack()
        Thread.sleep(1000)
        Log.i(TAG, "RESULT ui smoke passed, settings button found: ${settings != null}")
    }

    /**
     * MediaPipe 1.1.1 (tasks-core/vision/text) with tasks-genai 0.10.35: Gemma-3n E2B answers a text
     * prompt and describes an image (BitmapImageBuilder is from tasks-core), and resetChatSession
     * from the main thread keeps it responsive (ANR in LlmTaskRunner.nativeCreateSession).
     */
    @Test fun mediaPipeGemma3nTextAndImage() = runBlocking {
        val model = com.llmhub.llmhub.data.ModelData.models.first { it.name == "Gemma-3n E2B" }
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        download(model.url, file)
        val service = com.llmhub.llmhub.inference.MediaPipeInferenceService(context)
        assertTrue("Gemma-3n must load", service.loadModel(model, com.google.mediapipe.tasks.genai.llminference.LlmInference.Backend.GPU, null))

        val text = service.generateResponse("What is the capital of France? Answer in one word.", model)
        Log.i(TAG, "RESULT gemma3n text: ${text.take(120)}")
        assertTrue("Unexpected text answer: $text", text.contains("Paris", ignoreCase = true))

        val red = android.graphics.Bitmap.createBitmap(256, 256, android.graphics.Bitmap.Config.ARGB_8888).apply { eraseColor(android.graphics.Color.RED) }
        val imageAnswer = StringBuilder()
        service.generateResponseStreamWithSession(
            "What color is this image? Answer in one word.", model, "vitals-image", images = listOf(red), webSearchEnabled = false
        ).collect { imageAnswer.append(it) }
        Log.i(TAG, "RESULT gemma3n image: ${imageAnswer.take(120)}")
        assertTrue("Unexpected image answer: $imageAnswer", imageAnswer.contains("red", ignoreCase = true))

        val handler = android.os.Handler(android.os.Looper.getMainLooper())
        val last = java.util.concurrent.atomic.AtomicLong(android.os.SystemClock.elapsedRealtime())
        val gap = java.util.concurrent.atomic.AtomicLong(0)
        var beating = true
        val beat = object : Runnable {
            override fun run() {
                val now = android.os.SystemClock.elapsedRealtime()
                gap.set(maxOf(gap.get(), now - last.get())); last.set(now)
                if (beating) handler.postDelayed(this, 16)
            }
        }
        handler.post(beat)
        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) { service.resetChatSession("vitals-image") }
        beating = false
        Log.i(TAG, "RESULT gemma3n resetChatSession longest main-thread gap ${gap.get()} ms")
        assertTrue("Main thread blocked ${gap.get()} ms in resetChatSession", gap.get() < 1000)
        service.unloadModel()
    }

    /**
     * Audio input through MediaPipe 1.1.1: a spoken question (Android text-to-speech, converted with
     * the app's own AudioConversionUtils to 16 kHz Float32 WAV) answered by Gemma-3n E2B.
     */
    @Test fun mediaPipeGemma3nAudio() = runBlocking {
        val model = com.llmhub.llmhub.data.ModelData.models.first { it.name == "Gemma-3n E2B" }
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
        download(model.url, file)

        val wav = File(context.cacheDir, "vitals_question.wav")
        val spoken = kotlinx.coroutines.CompletableDeferred<Boolean>()
        lateinit var tts: android.speech.tts.TextToSpeech
        instrumentation.runOnMainSync {
            tts = android.speech.tts.TextToSpeech(context) { status ->
                if (status != android.speech.tts.TextToSpeech.SUCCESS) { spoken.complete(false); return@TextToSpeech }
                tts.language = java.util.Locale.US
                tts.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                    override fun onStart(id: String?) {}
                    override fun onDone(id: String?) { spoken.complete(true) }
                    @Deprecated("Deprecated in Java") override fun onError(id: String?) { spoken.complete(false) }
                })
                tts.synthesizeToFile("What is the capital of France?", null, wav, "q")
            }
        }
        assertTrue("Text-to-speech could not synthesize the question", kotlinx.coroutines.withTimeout(60_000) { spoken.await() })
        tts.shutdown()
        val audio = com.llmhub.llmhub.utils.AudioConversionUtils.convertUriToFloat32Wav(context, android.net.Uri.fromFile(wav))
        assertNotNull("Audio conversion failed", audio)
        Log.i(TAG, "spoken question: ${wav.length()} bytes wav -> ${audio!!.size} bytes Float32 WAV")

        val service = com.llmhub.llmhub.inference.MediaPipeInferenceService(context)
        assertTrue("Gemma-3n must load", service.loadModel(model, com.google.mediapipe.tasks.genai.llminference.LlmInference.Backend.GPU, null))
        val answer = StringBuilder()
        service.generateResponseStreamWithSession(
            "Answer the question asked in the audio in one word.", model, "vitals-audio", audioData = audio, webSearchEnabled = false
        ).collect { answer.append(it) }
        Log.i(TAG, "RESULT gemma3n audio: ${answer.take(160)}")
        assertTrue("Unexpected audio answer: $answer", answer.contains("Paris", ignoreCase = true))
        service.unloadModel()
    }

    private fun download(url: String, target: File) {
        var attempt = 0
        while (true) {
            try {
                var have = if (target.exists()) target.length() else 0L
                val conn = (java.net.URL(url).openConnection() as java.net.HttpURLConnection).apply {
                    instanceFollowRedirects = true; connectTimeout = 30_000; readTimeout = 120_000
                    if (have > 0) setRequestProperty("Range", "bytes=$have-")
                    // Gated repos (google/gemma-3n-*): the same token the app's downloader sends.
                    if (url.contains("huggingface.co") && BuildConfig.HF_TOKEN.isNotBlank()) {
                        setRequestProperty("Authorization", "Bearer ${BuildConfig.HF_TOKEN}")
                    }
                }
                when (conn.responseCode) { 416 -> return; 200 -> have = 0L }
                conn.inputStream.use { input -> java.io.FileOutputStream(target, have > 0).use { input.copyTo(it, 1 shl 20) } }
                return
            } catch (e: java.io.IOException) {
                if (++attempt >= 6) throw e
                Thread.sleep(5_000)
            }
        }
    }

    private companion object { const val TAG = "VitalsFixes2DeviceTest" }
}
