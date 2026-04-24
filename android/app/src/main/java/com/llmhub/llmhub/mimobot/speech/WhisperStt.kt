package com.llmhub.llmhub.mimobot.speech

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Speech-to-text entry point. Implementation is whisper.cpp via JNI.
 *
 * We use whisper.cpp (not MediaPipe audio) because:
 *   - It runs on any Android 8+ device, no Qualcomm/MediaTek dependency.
 *   - Tiny.en is ~75 MB and decodes at ~5-10× realtime on a midrange phone.
 *   - The ggml binary format is the same on iOS, so both platforms share the
 *     model catalog and downloader.
 *
 * TODO(stt):
 *   1. Bundle the whisper.cpp Android library. Two options:
 *        a) Build libwhisper.so from https://github.com/ggerganov/whisper.cpp
 *           with the Android example (whisper.cpp/examples/whisper.android/lib).
 *        b) Use a packaged artifact like io.github.giga-gpt:whisper-android.
 *      (a) is more work but gives us control over build flags (GGML_USE_OPENBLAS,
 *      WHISPER_CUBLAS off, etc).
 *   2. Load the GGUF/GGML model via WhisperContext.createContextFromFile(path).
 *   3. For streaming: feed PCM into whisper_full_with_state in ~5s chunks.
 *
 * Model: whisper-tiny.en-q5_1.bin (~75 MB) from
 *   https://huggingface.co/ggerganov/whisper.cpp/tree/main
 */
class WhisperStt(private val modelPath: String) {

    suspend fun load() { TODO("WhisperContext.createContextFromFile(modelPath)") }

    /**
     * Transcribe a completed PCM buffer (16 kHz mono, 16-bit). For v0 PTT this
     * is called once the user releases the button and the VAD tail has fired.
     * Streaming (partial transcripts) can come later.
     */
    suspend fun transcribe(pcm: ShortArray, languageHint: String? = "en"): String {
        TODO("whisper_full(ctx, params, pcm, pcm.size); collect segments into a String")
    }

    /** Future: streaming partials. Emits growing transcripts as audio arrives. */
    fun stream(pcmChunks: Flow<ShortArray>, languageHint: String? = "en"): Flow<String> = flow {
        TODO("chunked whisper_full_with_state calls, emit running transcript per chunk")
    }

    fun close() { /* TODO: whisper_free(ctx) */ }
}
