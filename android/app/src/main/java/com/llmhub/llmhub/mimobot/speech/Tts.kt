package com.llmhub.llmhub.mimobot.speech

import kotlinx.coroutines.flow.Flow

/**
 * Abstraction over any text-to-speech backend. Implementations emit 16 kHz
 * mono 16-bit PCM in 320-sample frames so the caller can feed them straight
 * into [OpusEncoder].
 *
 * v0 ships two implementations:
 *   - [SystemTts]  — Android's built-in TextToSpeech (free, decent, ships on every phone)
 *   - [KokoroTts]  — Kokoro-82M ONNX (higher quality, larger download)
 *
 * Later additions (Piper, Moshi / Mini-Omni full speech-to-speech) implement
 * this interface too. The pipeline doesn't care which one is plugged in.
 */
interface Tts {
    suspend fun init()
    fun speakToPcm(text: String, language: String = "en-US"): Flow<ShortArray>
    fun stop()
    fun close()
}
