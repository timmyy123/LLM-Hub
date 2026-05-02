package com.llmhub.llmhub.mimobot.speech

/**
 * whisper.cpp-backed [SpeechToText].
 *
 * NOT IMPLEMENTED YET — the Android ships against a system recognizer for v0
 * (see [AndroidSpeechRecognizerStt]). When wired in, this implementation will:
 *   1. Own its own [com.llmhub.llmhub.mimobot.audio.MicSource] OR consume
 *      frames from the BLE transport.
 *   2. Buffer PCM until the VAD tail fires.
 *   3. Feed the buffer into whisper.cpp via JNI.
 *
 * TODO(whisper):
 *   1. Bundle libwhisper.so (build from https://github.com/ggerganov/whisper.cpp
 *      examples/whisper.android/lib, arm64-v8a + armeabi-v7a).
 *   2. Ship whisper-tiny.en-q5_1.bin (~75 MB) via the existing ModelDownloader.
 *   3. On [recognizeTurn], start mic + VAD; on end-of-utterance, call
 *      `WhisperContext.fullTranscribe(pcm)` and return the concatenated segments.
 */
class WhisperStt(
    @Suppress("unused") private val modelPath: String,
) : SpeechToText {

    override suspend fun recognizeTurn(languageHint: String): String {
        TODO("whisper.cpp JNI — load model, mic+VAD, whisper_full, join segments")
    }

    override fun cancel() { /* TODO */ }
    override fun close() { /* TODO */ }
}
