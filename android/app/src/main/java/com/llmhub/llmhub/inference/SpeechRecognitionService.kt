package com.llmhub.llmhub.inference

import android.content.Context
import android.os.ParcelFileDescriptor
import android.util.Log
import com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriber
import com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriberResult
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.audio.AudioSource
import com.google.mlkit.genai.speechrecognition.SpeechRecognition
import com.google.mlkit.genai.speechrecognition.SpeechRecognizer
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerOptions
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerResponse
import com.google.mlkit.genai.speechrecognition.speechRecognizerOptions
import com.google.mlkit.genai.speechrecognition.speechRecognizerRequest
import com.llmhub.llmhub.data.AsrModels
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.Executors

/**
 * One entry point for every speech recognition model in the catalog, so features load and call
 * Whisper, Parakeet and ML Kit the same way: [loadModel], then [transcribe] 16 kHz mono PCM16.
 */
class SpeechRecognitionService(private val context: Context) {
    private val whisper = WhisperKitService(context)
    private var parakeet: ParakeetAsrEngine? = null
    private var parakeetNpu: ParakeetNpuEngine? = null
    private var mlKit: MlKitAsrEngine? = null

    val isLoaded: Boolean get() = whisper.isLoaded || parakeet != null || parakeetNpu != null || mlKit != null

    suspend fun loadModel(model: LLMModel, backend: WhisperBackend = WhisperBackend.NPU): Boolean {
        release()
        return when {
            AsrModels.isMlKit(model) -> {
                val engine = MlKitAsrEngine(AsrModels.isMlKitAdvanced(model))
                if (engine.prepare()) { mlKit = engine; true } else { engine.close(); false }
            }
            AsrModels.isParakeetNpu(model) -> {
                val engine = ParakeetNpuEngine.create(context, File(File(context.filesDir, "models"), model.localFileName()))
                parakeetNpu = engine
                engine != null
            }
            AsrModels.isParakeet(model) -> {
                val file = File(File(context.filesDir, "models"), model.localFileName())
                val engine = ParakeetAsrEngine.create(context, file, preferGpu = backend != WhisperBackend.CPU)
                parakeet = engine
                engine != null
            }
            else -> {
                val dirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                // English-only Whisper checkpoints need the language forced; multilingual ones auto-detect.
                whisper.transcribeLanguage = if (model.name.contains("english", ignoreCase = true)) "en" else ""
                whisper.loadModel(File(context.filesDir, "models/$dirName").absolutePath, backend)
            }
        }
    }

    /**
     * Returns the transcript, "" when the audio held no recognizable speech, or null when no model
     * is ready. Callers treat null as a failure and "" as "nothing was said" (Vibe Voice asks the
     * user to repeat), so a recording of room noise must not come back as null.
     */
    suspend fun transcribe(pcm16Bytes: ByteArray): String? {
        parakeet?.let { return it.transcribe(pcm16Bytes).orEmpty() }
        parakeetNpu?.let { return it.transcribe(pcm16Bytes).orEmpty() }
        mlKit?.let { return it.transcribe(pcm16Bytes).orEmpty() }
        return whisper.transcribe(pcm16Bytes, whisper.transcribeLanguage)
    }

    fun release() {
        whisper.release()
        parakeet?.close()
        parakeet = null
        parakeetNpu?.close()
        parakeetNpu = null
        mlKit?.close()
        mlKit = null
    }
}

/** Parakeet TDT in Google's `.litertlm` format, on the MediaPipe AudioTranscriber task. */
class ParakeetAsrEngine private constructor(
    private val transcriber: AudioTranscriber,
    private val executor: java.util.concurrent.ExecutorService
) {
    private val dispatcher = executor.asCoroutineDispatcher()

    suspend fun transcribe(pcm16Bytes: ByteArray): String? = withContext(dispatcher) {
        val shorts = ByteBuffer.wrap(pcm16Bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val samples = FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
        val finals = StringBuilder()
        var interim = ""
        fun take(result: AudioTranscriberResult?) {
            for (segment in result?.segments().orEmpty()) {
                val text = segment.transcript().trim()
                if (text.isEmpty()) continue
                if (segment.isFinal) {
                    if (finals.isNotEmpty()) finals.append(' ')
                    finals.append(text)
                    interim = ""
                } else interim = text
            }
        }
        try {
            var offset = 0
            while (offset < samples.size) {
                val end = minOf(offset + CHUNK_SAMPLES, samples.size)
                take(transcriber.transcribe(samples.copyOfRange(offset, end), false))
                offset = end
            }
            take(transcriber.finishStream())
        } finally {
            // Resets the stream state so the next recording does not continue this one.
            runCatching { transcriber.prewarm() }
        }
        listOf(finals.toString(), interim).filter { it.isNotEmpty() }.joinToString(" ").ifEmpty { null }
    }

    fun close() {
        executor.execute { runCatching { transcriber.close() } }
        executor.shutdown()
    }

    companion object {
        private const val TAG = "ParakeetAsrEngine"
        private const val SAMPLE_RATE = 16000
        /** 100 ms per call, the size the task is tuned for. */
        private const val CHUNK_SAMPLES = 1600

        /** Loads on GPU when [preferGpu] and it works, otherwise CPU. Null when neither loads. */
        suspend fun create(context: Context, modelFile: File, preferGpu: Boolean): ParakeetAsrEngine? {
            if (!modelFile.isFile) return null
            val executor = Executors.newSingleThreadExecutor { Thread(it, "parakeet-asr").apply { isDaemon = true } }
            val cacheDir = File(context.cacheDir, "parakeet_asr").apply { mkdirs() }
            // Uncancellable: a cancelled withContext would drop the native transcriber it returns.
            val engine = withContext(kotlinx.coroutines.NonCancellable + executor.asCoroutineDispatcher()) {
                for (delegate in if (preferGpu) listOf(Delegate.GPU, Delegate.CPU) else listOf(Delegate.CPU)) {
                    var candidate: AudioTranscriber? = null
                    try {
                        candidate = AudioTranscriber.createFromOptions(
                            context,
                            AudioTranscriber.AudioTranscriberOptions.builder()
                                .setBaseOptions(
                                    BaseOptions.builder().setModelAssetPath(modelFile.absolutePath).setDelegate(delegate).build()
                                )
                                .setSampleRateHz(SAMPLE_RATE)
                                .setEnableInterimResults(true)
                                .setModelArchitecture(AudioTranscriber.ModelArchitecture.AUTO)
                                .setCacheDir(cacheDir.absolutePath)
                                .build()
                        )
                        // Run silence through once: proves the delegate works and pays the first-call cost here.
                        candidate.prewarm()
                        candidate.transcribe(FloatArray(CHUNK_SAMPLES), false)
                        candidate.finishStream()
                        candidate.prewarm()
                        Log.i(TAG, "Loaded ${modelFile.name} on $delegate")
                        return@withContext ParakeetAsrEngine(candidate, executor)
                    } catch (t: Throwable) {
                        Log.w(TAG, "AudioTranscriber failed on $delegate", t)
                        runCatching { candidate?.close() }
                    }
                }
                null
            }
            if (engine == null) executor.shutdown()
            return engine
        }
    }
}

/**
 * ML Kit GenAI speech recognition. The weights belong to the system (AICore), so there is no file:
 * "downloaded" means the feature reports AVAILABLE for the recognition locale.
 */
class MlKitAsrEngine(advanced: Boolean, preferred: Locale? = null) {
    private val mode =
        if (advanced) SpeechRecognizerOptions.Mode.MODE_ADVANCED else SpeechRecognizerOptions.Mode.MODE_BASIC
    // The system's languages, not the app's: the app locale drops the region ("en"), which ML Kit
    // rejects. Not every language has a pack; English is the one every supported device offers.
    private val locales = (
        listOfNotNull(preferred) +
            android.content.res.Resources.getSystem().configuration.locales.let { list -> List(list.size()) { list[it] } } +
            Locale.getDefault() + Locale.US
        ).distinctBy { it.toLanguageTag() }
    private var recognizer: SpeechRecognizer? = null

    private fun client(locale: Locale): SpeechRecognizer =
        SpeechRecognition.getClient(speechRecognizerOptions { this.locale = locale; preferredMode = mode })

    /**
     * Keeps the recognizer of the first language that is installed, or failing that the first one
     * the system can install. Returns its status.
     */
    suspend fun status(): Int {
        recognizer?.let { return runCatching { it.checkStatus() }.getOrDefault(FeatureStatus.UNAVAILABLE) }
        var fallback: Pair<SpeechRecognizer, Int>? = null
        for (locale in locales) {
            val candidate = try { client(locale) } catch (t: Throwable) {
                Log.w(TAG, "ML Kit speech recognition client unavailable", t)
                break
            }
            val status = runCatching { candidate.checkStatus() }
                .onFailure { Log.w(TAG, "checkStatus failed for ${locale.toLanguageTag()}", it) }
                .getOrDefault(FeatureStatus.UNAVAILABLE)
            Log.d(TAG, "${locale.toLanguageTag()} ${if (mode == SpeechRecognizerOptions.Mode.MODE_ADVANCED) "advanced" else "basic"}: feature status $status")
            if (status == FeatureStatus.AVAILABLE) {
                fallback?.first?.let { runCatching { it.close() } }
                recognizer = candidate
                return status
            }
            if (status != FeatureStatus.UNAVAILABLE && fallback == null) fallback = candidate to status
            else runCatching { candidate.close() }
        }
        recognizer = fallback?.first
        return fallback?.second ?: FeatureStatus.UNAVAILABLE
    }

    /** Ready to transcribe. False when the model still has to be downloaded or the device has none. */
    suspend fun prepare(): Boolean {
        if (status() != FeatureStatus.AVAILABLE) return false
        runCatching { recognizer?.warmup() }.onFailure { Log.w(TAG, "warmup failed", it) }
        return true
    }

    /** Starts the system download. Null when this device cannot run the recognizer at all. */
    suspend fun download(): Flow<DownloadStatus>? =
        if (status() == FeatureStatus.UNAVAILABLE) null else recognizer?.download()

    suspend fun transcribe(pcm16Bytes: ByteArray): String? {
        var speed = feedSpeed
        while (true) {
            try {
                return transcribe(pcm16Bytes, speed)
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                // A slower device falls behind the feed; try again more slowly, down to real time.
                if (speed <= 1f || e.message?.contains("AUDIO_BUFFER_OVERFLOW") != true) throw e
                speed = maxOf(1f, speed / 2)
                Log.w(TAG, "Recognizer fell behind, feeding again at ${speed}x")
            }
        }
    }

    private suspend fun transcribe(pcm16Bytes: ByteArray, speed: Float): String? = coroutineScope {
        val recognizer = recognizer ?: return@coroutineScope null
        // The recognizer reads raw 16 kHz mono PCM16 from the pipe as if it were a microphone.
        val pipe = ParcelFileDescriptor.createPipe()
        val request = speechRecognizerRequest { audioSource = AudioSource.fromPfd(pipe[0]) }
        val finals = StringBuilder()
        var partial = ""
        val completed = CompletableDeferred<Unit>()
        val writer = launch(Dispatchers.IO) {
            runCatching {
                ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { output ->
                    // The recognizer expects a live microphone and overflows its buffer when a
                    // whole recording arrives at once, so feed it at a steady multiple of real time.
                    var offset = 0
                    while (offset < pcm16Bytes.size) {
                        val end = minOf(offset + FEED_CHUNK_BYTES, pcm16Bytes.size)
                        output.write(pcm16Bytes, offset, end - offset)
                        offset = end
                        kotlinx.coroutines.delay((FEED_CHUNK_MS / speed).toLong())
                    }
                }
            }
            // End of audio does not always end the session on its own; ask for the final results.
            if (withTimeoutOrNull(1_500) { completed.await() } == null) runCatching { recognizer.stopRecognition() }
        }
        try {
            val audioMs = pcm16Bytes.size / 32L
            withTimeout(audioMs * 3 + 30_000) {
                recognizer.startRecognition(request).takeWhile { response ->
                    when (response) {
                        is SpeechRecognizerResponse.PartialTextResponse -> { partial = response.text.trim(); true }
                        is SpeechRecognizerResponse.FinalTextResponse -> {
                            // One per pause in the speech, not one per recording.
                            val text = response.text.trim()
                            if (text.isNotEmpty()) {
                                if (finals.isNotEmpty()) finals.append(' ')
                                finals.append(text)
                            }
                            partial = ""
                            true
                        }
                        is SpeechRecognizerResponse.CompletedResponse -> false
                        is SpeechRecognizerResponse.ErrorResponse -> throw response.e
                        else -> true
                    }
                }.collect()
            }
        } finally {
            completed.complete(Unit)
            writer.cancel()
            // Unblocks a writer still waiting on a reader that has gone away.
            runCatching { pipe[0].close() }
        }
        listOf(finals.toString(), partial).filter { it.isNotEmpty() }.joinToString(" ").ifEmpty { null }
    }

    fun close() {
        runCatching { recognizer?.close() }
        recognizer = null
    }

    companion object {
        private const val TAG = "MlKitAsrEngine"
        private const val FEED_CHUNK_MS = 100
        private const val FEED_CHUNK_BYTES = 16000 * 2 * FEED_CHUNK_MS / 1000
        /** How many times faster than real time recorded audio is fed to the recognizer. */
        @Volatile internal var feedSpeed = 4f

        /** Status of the recognizer for [model] on this device, without keeping a client open. */
        suspend fun status(model: LLMModel): Int {
            val engine = MlKitAsrEngine(AsrModels.isMlKitAdvanced(model))
            return try { engine.status() } finally { engine.close() }
        }
    }
}
