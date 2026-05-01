package com.llmhub.llmhub.mimobot.speech

/**
 * Abstraction over a speech-to-text engine. v0 ships two implementations:
 *   - [AndroidSpeechRecognizerStt] — system SpeechRecognizer, owns its own mic,
 *     works today, offline on Android 12+ when the on-device recognizer is
 *     installed.
 *   - [WhisperStt]                 — whisper.cpp, offline everywhere. Not yet
 *     implemented; will consume PCM from a separate [com.llmhub.llmhub.mimobot.audio.AudioSource].
 *
 * The pipeline only needs "take a turn, give me text" — it does not care which
 * implementation owns the mic.
 */
interface SpeechToText {
    /**
     * Start listening and suspend until the user stops speaking (VAD tail) or
     * the backend returns a final result. Returns the transcript, or empty
     * string if nothing was recognized.
     */
    suspend fun recognizeTurn(languageHint: String = "en-US"): String

    /** Cancel an in-flight recognition. */
    fun cancel()

    fun close()
}
