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
    private var recognizerIsOnDevice: Boolean? = null

    private fun ensureRecognizer(useOnDevice: Boolean): SpeechRecognizer {
        recognizer?.let { existing ->
            if (recognizerIsOnDevice == useOnDevice) return existing
        }
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
        recognizerIsOnDevice = null

        val r = if (useOnDevice && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
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
        recognizerIsOnDevice = useOnDevice && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
        return r
    }

    override suspend fun recognizeTurn(languageHint: String): String = withContext(Dispatchers.Main) {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            Log.w(TAG, "speech recognizer not available on this device")
            return@withContext ""
        }

        val first = recognizeOnce(languageHint, preferOnDevice)
        if (first.text.isNotBlank()) return@withContext first.text

        if (preferOnDevice && first.error != null && !first.beganSpeech) {
            Log.w(TAG, "on-device recognition failed, retrying with network")
            return@withContext recognizeOnce(languageHint, false).text
        }

        return@withContext first.text
    }

    private suspend fun recognizeOnce(languageHint: String, useOnDevice: Boolean): SttAttempt =
        suspendCancellableCoroutine { cont ->
            val r = ensureRecognizer(useOnDevice)
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageHint)
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, useOnDevice)
                }
            }

            var beganSpeech = false
            val listener = object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val text = list?.firstOrNull().orEmpty()
                    if (cont.isActive) cont.resume(SttAttempt(text, null, beganSpeech))
                }
                override fun onError(error: Int) {
                    Log.w(TAG, "recognizer error=$error (${errorName(error)})")
                    if (cont.isActive) cont.resume(SttAttempt("", error, beganSpeech))
                }
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() { beganSpeech = true }
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
                if (cont.isActive) cont.resume(SttAttempt("", SpeechRecognizer.ERROR_CLIENT, beganSpeech))
            }
        }
    }

    override fun cancel() {
        try { recognizer?.cancel() } catch (_: Throwable) {}
    }

    override fun close() {
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
        recognizerIsOnDevice = null
    }

    private data class SttAttempt(
        val text: String,
        val error: Int?,
        val beganSpeech: Boolean,
    )

    private fun errorName(error: Int): String = when (error) {
        SpeechRecognizer.ERROR_AUDIO -> "audio"
        SpeechRecognizer.ERROR_CLIENT -> "client"
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "permissions"
        SpeechRecognizer.ERROR_NETWORK -> "network"
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "network_timeout"
        SpeechRecognizer.ERROR_NO_MATCH -> "no_match"
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "busy"
        SpeechRecognizer.ERROR_SERVER -> "server"
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "speech_timeout"
        else -> "unknown"
    }

    companion object { private const val TAG = "MimoAndroidStt" }
}
