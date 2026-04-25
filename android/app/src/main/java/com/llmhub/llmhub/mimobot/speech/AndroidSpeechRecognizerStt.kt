package com.llmhub.llmhub.mimobot.speech

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

/**
 * [SpeechToText] backed by Android's built-in [SpeechRecognizer].
 *
 * On Android 12 (API 31) and newer we prefer `createOnDeviceSpeechRecognizer`
 * so transcription stays on-device. On older versions we fall back to the
 * network-backed Google recognizer (requires Google app / services).
 *
 * Must be constructed and used on the main thread (SpeechRecognizer is
 * main-thread-bound per its contract).
 */
class AndroidSpeechRecognizerStt(
    private val context: Context,
    private val preferOnDevice: Boolean = true,
) : SpeechToText {

    private var recognizer: SpeechRecognizer? = null

    private fun ensureRecognizer(): SpeechRecognizer {
        recognizer?.let { return it }
        val r = if (preferOnDevice && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
            } catch (t: Throwable) {
                Log.w(TAG, "on-device recognizer unavailable, falling back: ${t.message}")
                SpeechRecognizer.createSpeechRecognizer(context)
            }
        } else {
            SpeechRecognizer.createSpeechRecognizer(context)
        }
        recognizer = r
        return r
    }

    override suspend fun recognizeTurn(languageHint: String): String = withContext(Dispatchers.Main) {
        val r = ensureRecognizer()
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageHint)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, preferOnDevice)
            }
        }

        suspendCancellableCoroutine<String> { cont ->
            val listener = object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val text = list?.firstOrNull().orEmpty()
                    if (cont.isActive) cont.resume(text)
                }
                override fun onError(error: Int) {
                    Log.w(TAG, "recognizer error=$error")
                    if (cont.isActive) cont.resume("")
                }
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            }

            cont.invokeOnCancellation {
                try { r.cancel() } catch (_: Throwable) {}
            }

            r.setRecognitionListener(listener)
            try {
                r.startListening(intent)
            } catch (t: Throwable) {
                Log.e(TAG, "startListening failed: ${t.message}", t)
                if (cont.isActive) cont.resume("")
            }
        }
    }

    override fun cancel() {
        try { recognizer?.cancel() } catch (_: Throwable) {}
    }

    override fun close() {
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
    }

    companion object { private const val TAG = "MimoAndroidStt" }
}
