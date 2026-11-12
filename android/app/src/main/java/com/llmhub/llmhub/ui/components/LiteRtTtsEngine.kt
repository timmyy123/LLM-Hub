package com.llmhub.llmhub.ui.components

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.audio.texttospeech.SynthesisOptions
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeech
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeechOptions
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors

/**
 * Kokoro and PocketTTS in Google's `.litertlm` format, on the MediaPipe TextToSpeech task.
 *
 * Creation, synthesis and close all run on one thread: the GPU delegate binds its context to the
 * thread that created it.
 */
class LiteRtTtsEngine private constructor(
    private val tts: TextToSpeech,
    private val executor: java.util.concurrent.ExecutorService,
    val usesGpu: Boolean
) : Closeable {
    private val dispatcher = executor.asCoroutineDispatcher()
    @Volatile private var closed = false

    /** Streams [text] as float PCM chunks at [SAMPLE_RATE]. Returns once the utterance is complete. */
    suspend fun synthesize(text: String, voice: String, locale: String?, onChunk: (FloatArray) -> Unit) {
        check(!closed) { "TTS engine is closed" }
        try {
            withContext(dispatcher) {
                tts.synthesizeStream(text, SynthesisOptions(voice, locale)).collect { chunk ->
                    if (chunk.samples.isNotEmpty()) onChunk(chunk.samples)
                }
            }
        } catch (e: CancellationException) {
            // The native loop keeps running until told to stop.
            runCatching { tts.cancel() }
            throw e
        }
    }

    /** Aborts the utterance being synthesized, from any thread. */
    fun cancel() {
        if (!closed) runCatching { tts.cancel() }
    }

    override fun close() {
        if (closed) return
        closed = true
        runCatching { tts.cancel() }
        executor.execute {
            runCatching { tts.close() }.onFailure { Log.w(TAG, "Failed to close TextToSpeech", it) }
        }
        executor.shutdown()
    }

    companion object {
        private const val TAG = "LiteRtTtsEngine"
        const val SAMPLE_RATE = 24000
        /** Kokoro's graph has a static input shape; longer clauses cost extra passes. */
        const val KOKORO_CLAUSE_CHARS = 110

        private val closingQuotesAfterPunctuation = Regex("""(?<=[.!?。！？…])["“”„‟«»‹›「」『』‘’'‚‛)\]}）】］]+""")
        private val doubleQuotes = Regex("""["“”„‟«»‹›「」『』]+""")
        private val nonContractionSingleQuotes = Regex("""[‘‚‛]+|(?<![\p{L}\p{N}])['’]+|['’]+(?![\p{L}\p{N}])""")

        /** Quotation marks make the native text front end mispronounce or drop words; apostrophes stay. */
        fun sanitize(text: String): String = text
            .replace(closingQuotesAfterPunctuation, "")
            .replace(doubleQuotes, "")
            .replace(nonContractionSingleQuotes, "")
            .trim()

        /**
         * Loads [modelFile], on GPU when [preferGpu] and it survives a short warm-up utterance,
         * otherwise on CPU. Blocking native work runs on the engine's own thread.
         */
        suspend fun create(
            context: Context,
            modelFile: File,
            preferGpu: Boolean,
            warmUpVoice: String,
            warmUpLocale: String?,
            speakingRate: Float
        ): LiteRtTtsEngine {
            val executor = Executors.newSingleThreadExecutor { Thread(it, "litert-tts").apply { isDaemon = true } }
            val dispatcher = executor.asCoroutineDispatcher()
            val cacheDir = File(context.cacheDir, "litert_tts").apply { mkdirs() }
            val caller = currentCoroutineContext()[Job]
            try {
                // Native loading cannot be interrupted, and a cancelled withContext would drop the
                // engine it returns, so finish the load and close it afterwards if nobody wants it.
                val engine = withContext(NonCancellable + dispatcher) {
                    var last: Throwable? = null
                    for (delegate in if (preferGpu) listOf(Delegate.GPU, Delegate.CPU) else listOf(Delegate.CPU)) {
                        if (caller?.isActive == false) throw CancellationException("TTS load cancelled")
                        var candidate: TextToSpeech? = null
                        try {
                            candidate = TextToSpeech.create(
                                TextToSpeechOptions(
                                    baseOptions = BaseOptions.builder()
                                        .setModelAssetPath(modelFile.absolutePath)
                                        .setDelegate(delegate)
                                        .build(),
                                    defaultVoiceId = null,
                                    defaultLocale = "en-US",
                                    speakingRate = speakingRate.coerceIn(0.5f, 2f),
                                    cacheDirectory = cacheDir.absolutePath
                                )
                            )
                            // GPU load can succeed and inference still fail, so prove it before committing.
                            candidate.synthesizeStream("Hi.", SynthesisOptions(warmUpVoice, warmUpLocale)).collect { }
                            Log.i(TAG, "Loaded ${modelFile.name} on $delegate")
                            return@withContext LiteRtTtsEngine(candidate, executor, delegate == Delegate.GPU)
                        } catch (e: CancellationException) {
                            runCatching { candidate?.cancel(); candidate?.close() }
                            throw e
                        } catch (t: Throwable) {
                            Log.w(TAG, "TextToSpeech failed on $delegate", t)
                            runCatching { candidate?.close() }
                            last = t
                        }
                    }
                    throw IllegalStateException("Unable to load ${modelFile.name}", last)
                }
                if (caller?.isActive == false) {
                    engine.close()
                    throw CancellationException("TTS load cancelled")
                }
                return engine
            } catch (t: Throwable) {
                executor.shutdown()
                throw t
            }
        }
    }
}
