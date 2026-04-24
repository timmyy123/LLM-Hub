package com.llmhub.llmhub.mimobot.speech

import android.content.Context
import android.speech.tts.TextToSpeech
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * System TTS via Android's built-in [TextToSpeech]. Free, on-device, ships on
 * every phone.
 *
 * The API doesn't expose raw PCM cleanly — we use
 * `UtteranceProgressListener.onAudioAvailable` (API 24+) which streams PCM
 * while the voice renders. Most voices emit at 22 050 Hz; we resample to
 * 16 kHz mono and chunk into 320-sample frames before emitting.
 *
 * TODO(system-tts):
 *   1. Initialize TextToSpeech in [init], await the onInit callback.
 *   2. In [speakToPcm], set an UtteranceProgressListener, call
 *      `tts.synthesizeToFile(...)` OR use the callback-based API, collect the
 *      PCM as it arrives.
 *   3. Resample 22 050 → 16 000 with a simple polyphase filter (or via
 *      AudioTrack intermediate if you don't mind drift).
 *   4. Emit 320-sample Int16 frames.
 */
class SystemTts(private val context: Context) : Tts {
    private var tts: TextToSpeech? = null

    override suspend fun init() {
        TODO("new TextToSpeech(context) { status -> continuation.resume(Unit) }")
    }

    override fun speakToPcm(text: String, language: String): Flow<ShortArray> = flow {
        TODO("UtteranceProgressListener.onAudioAvailable → resample → 320-sample frames")
    }

    override fun stop() { tts?.stop() }
    override fun close() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }
}

/** Kept as a shallow alias so older references don't break. Prefer [SystemTts]. */
@Deprecated("Use SystemTts", ReplaceWith("SystemTts(context)"))
typealias TtsEngine = SystemTts
