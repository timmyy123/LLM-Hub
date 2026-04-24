package com.llmhub.llmhub.mimobot.speech

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Kokoro-82M neural TTS via ONNX Runtime.
 *
 * Why Kokoro:
 *   - 82M parameters — small enough to load on a phone (~160 MB fp16 ONNX).
 *   - Runs on ONNX Runtime, which is ALREADY a dependency of this project
 *     (com.microsoft.onnxruntime:onnxruntime-android:1.24.1 — see
 *     android/app/build.gradle.kts line 267). Zero new native deps.
 *   - Higher quality than the system TTS, multilingual, voice-cloneable.
 *   - Runs at realtime on a midrange phone with NNAPI GPU EP.
 *
 * Model assets to ship (download on first use, same as our LLMs):
 *   - kokoro-v0_19.onnx          (ONNX graph)
 *   - voices.bin                 (speaker embeddings, ~5 MB)
 *   - phoneme vocab JSON         (~50 KB)
 * From https://huggingface.co/hexgrad/Kokoro-82M (or the ONNX mirror).
 *
 * Pipeline at runtime:
 *   text  ──► G2P (English) ──► phoneme IDs
 *                                     │
 *                                     ▼
 *            voice embedding ─────► ONNX Kokoro decoder ──► 24 kHz waveform
 *                                                                │
 *                                                                ▼
 *                                                  resample 24 kHz → 16 kHz
 *                                                                │
 *                                                                ▼
 *                                                  chunk → 320-sample frames
 *
 * TODO(kokoro):
 *   1. Add a model entry to ModelData (modelFormat="onnx", URL pointing at
 *      the Kokoro release). Reuse ModelDownloader — same path as LLMs.
 *   2. For English G2P, use misaki or a small espeak-ng fallback. For other
 *      languages, phoneme packs ship separately.
 *   3. Build OrtSession with NNAPI exec provider:
 *        val opts = OrtSession.SessionOptions().apply {
 *            addNnapi()          // GPU / NPU on supported devices
 *            setIntraOpNumThreads(4)
 *        }
 *   4. Feed [input_ids (int64), speaker_embed (float), speed (float)].
 *   5. Output is 24 kHz float PCM; resample + quantize to int16, chunk to
 *      320-sample frames, emit.
 *
 * When implementation lands, swap-in is a one-liner in VoicePipeline:
 *      val tts: Tts = KokoroTts(context, modelPath = ...)  // instead of SystemTts
 */
class KokoroTts(
    private val context: Context,
    private val modelPath: String,
    private val voicesPath: String,
    private val voiceId: String = "af_bella",
) : Tts {

    override suspend fun init() {
        TODO("OrtEnvironment + OrtSession from modelPath with NNAPI EP; load voices.bin")
    }

    override fun speakToPcm(text: String, language: String): Flow<ShortArray> = flow {
        TODO("G2P → run ONNX → resample 24k→16k → emit 320-sample frames")
    }

    override fun stop() { /* TODO: cancel the in-flight ONNX run */ }
    override fun close() { /* TODO: ortSession.close(); ortEnv.close() */ }
}
