package com.llmhub.llmhub.ui.components

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.isActive
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.LinkedBlockingQueue

/**
 * Text-to-Speech service for reading AI responses aloud.
 * Supports streaming TTS with sentence buffering for smooth playback.
 * Integrates system default TTS with custom ONNX-based Kokoro models.
 */
class TtsService(private val context: Context, private val isTranslationFeature: Boolean = false) {

    private var tts: TextToSpeech? = null
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private var isInitialized = false
    private var isCustomTts = false
    private var currentModelDir: File? = null

    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()

    private val _currentText = MutableStateFlow("")
    val currentText: StateFlow<String> = _currentText.asStateFlow()

// Buffer for streaming text
    private val textBuffer = StringBuilder()
    private var utteranceId = 0
    // Track whether we're currently inside a think block during streaming
    private var insideThinkBlock = false
    private val thinkBlockBuffer = StringBuilder()
    // Track number of in-flight utterances so we can reliably set speaking=false for System TTS
    private val inFlightUtterances = AtomicInteger(0)

    // Track number of active custom TTS synthesis/playback tasks
    private val activeJobsCount = AtomicInteger(0)

    private val themePreferences = ThemePreferences(context)
    private var currentSpeechRate = 1.0f

    // Calls buffered before initialization completed
    private val pendingSpeakQueue = mutableListOf<String>()

    // AudioTrack-based playback: synthesis writes ShortArrays to this queue,
    // a dedicated playback coroutine drains it — zero file I/O, no MediaPlayer prepare latency.
    // Sentinel null signals end-of-stream so playback coroutine can exit cleanly.
    private val pcmQueue = LinkedBlockingQueue<ShortArray?>()
    private var audioTrack: AudioTrack? = null
    private var playbackJob: Job? = null

    // Single-worker synthesis queue: sentences are processed strictly in order.
    // UNLIMITED capacity so senders never block; one coroutine drains serially.
    private val sentenceChannel = Channel<String>(Channel.UNLIMITED)
    private var synthesisWorkerJob: Job? = null

    // Cached voice styles keyed by absolute file path — avoids re-reading .bin on every speakAppend
    private val voiceStyleCache = HashMap<String, FloatArray>()

    private val synthesisScope = CoroutineScope(Dispatchers.IO + Job())

    private val TOKEN_MAP = mapOf(
        '$' to 0, ';' to 1, ':' to 2, ',' to 3, '.' to 4, '!' to 5, '?' to 6,
        '¡' to 7, '¿' to 8, '—' to 9, '…' to 10, '"' to 11, '«' to 12, '»' to 13,
        '“' to 14, '”' to 15, ' ' to 16,
        'A' to 17, 'B' to 18, 'C' to 19, 'D' to 20, 'E' to 21, 'F' to 22, 'G' to 23,
        'H' to 24, 'I' to 25, 'J' to 26, 'K' to 27, 'L' to 28, 'M' to 29, 'N' to 30,
        'O' to 31, 'P' to 32, 'Q' to 33, 'R' to 34, 'S' to 35, 'T' to 36, 'U' to 37,
        'V' to 38, 'W' to 39, 'X' to 40, 'Y' to 41, 'Z' to 42,
        'a' to 43, 'b' to 44, 'c' to 45, 'd' to 46, 'e' to 47, 'f' to 48, 'g' to 49,
        'h' to 50, 'i' to 51, 'j' to 52, 'k' to 53, 'l' to 54, 'm' to 55, 'n' to 56,
        'o' to 57, 'p' to 58, 'q' to 59, 'r' to 60, 's' to 61, 't' to 62, 'u' to 63,
        'v' to 64, 'w' to 65, 'x' to 66, 'y' to 67, 'z' to 68,
        'ɑ' to 69, 'ɐ' to 70, 'ɒ' to 71, 'æ' to 72, 'ɓ' to 73, 'ʙ' to 74, 'β' to 75,
        'ɔ' to 76, 'ɕ' to 77, 'ç' to 78, 'ɗ' to 79, 'ɖ' to 80, 'ð' to 81, 'ʤ' to 82,
        'ə' to 83, 'ɘ' to 84, 'ɚ' to 85, 'ɛ' to 86, 'ɜ' to 87, 'ɝ' to 88, 'ɞ' to 89,
        'ɟ' to 90, 'ʄ' to 91, 'ɡ' to 92, 'ɠ' to 93, 'ɢ' to 94, 'ʛ' to 95, 'ɦ' to 96,
        'ɧ' to 97, 'ħ' to 98, 'ɥ' to 99, 'ʜ' to 100, 'ɨ' to 101, 'ɪ' to 102, 'ʝ' to 103,
        'ɭ' to 104, 'ɬ' to 105, 'ɫ' to 106, 'ɮ' to 107, 'ʟ' to 108, 'ɱ' to 109, 'ɯ' to 110,
        'ɰ' to 111, 'ŋ' to 112, 'ɳ' to 113, 'ɲ' to 114, 'ɴ' to 115, 'ø' to 116, 'ɵ' to 117,
        'ɸ' to 118, 'θ' to 119, 'œ' to 120, 'ɶ' to 121, 'ʘ' to 122, 'ɹ' to 123, 'ɺ' to 124,
        'ɾ' to 125, 'ɻ' to 126, 'ʀ' to 127, 'ʁ' to 128, 'ɽ' to 129, 'ʂ' to 130, 'ʃ' to 131,
        'ʈ' to 132, 'ʧ' to 133, 'ʉ' to 134, 'ʊ' to 135, 'ʋ' to 136, 'ⱱ' to 137, 'ʌ' to 138,
        'ɣ' to 139, 'ɤ' to 140, 'ʍ' to 141, 'χ' to 142, 'ʎ' to 143, 'ʏ' to 144, 'ʑ' to 145,
        'ʐ' to 146, 'ʒ' to 147, 'ʔ' to 148, 'ʡ' to 149, 'ʕ' to 150, 'ʢ' to 151, 'ǀ' to 152,
        'ǁ' to 153, 'ǂ' to 154, 'ǃ' to 155, 'ˈ' to 156, 'ˌ' to 157, 'ː' to 158, 'ˑ' to 159,
        'ʼ' to 160, 'ʴ' to 161, 'ʰ' to 162, 'ʱ' to 163, 'ʲ' to 164, 'ʷ' to 165, 'ˠ' to 166,
        'ˤ' to 167, '˞' to 168, '↓' to 169, '↑' to 170, '→' to 171, '↗' to 172, '↘' to 173,
        '̩' to 175, '\'' to 176, 'ᵻ' to 177
    )

    companion object {
        private const val TAG = "TtsService"

        // Sentence delimiters for buffering
        private val SENTENCE_DELIMITERS = setOf('.', '!', '?', '。', '！', '？')
    }

    init {
        clearTempTtsFiles()
        initializeTts()
    }

    private fun initializeTts() {
        Log.d(TAG, "initializeTts() started")
        CoroutineScope(Dispatchers.Main).launch {
            val selectedModel = if (isTranslationFeature) null else themePreferences.selectedTtsModel.first()
            Log.d(TAG, "initializeTts: selectedModel = $selectedModel")
            if (selectedModel != null) {
                val model = ModelData.models.find { it.name == selectedModel }
                val modelDirName = model?.name?.replace(" ", "_")?.replace(Regex("[^a-zA-Z0-9_.-]"), "")
                val localName = model?.localFileName()
                val modelDir = if (modelDirName != null) File(File(context.filesDir, "models"), modelDirName) else null
                val modelFile = if (modelDir != null && localName != null) File(modelDir, localName) else null

                Log.d(TAG, "initializeTts: modelFile = ${modelFile?.absolutePath}, exists = ${modelFile?.exists()}, length = ${modelFile?.length()}")
                if (modelFile != null && modelFile.exists() && modelFile.length() > 0) {
                    try {
                        try {
                            System.loadLibrary("onnxruntime")
                        } catch (t: Throwable) {
                            Log.w(TAG, "ONNX Runtime library load failed: ${t.message}. Might already be loaded.")
                        }

                        withContext(Dispatchers.IO) {
                            val env = OrtEnvironment.getEnvironment()
                            val cpuCount = Runtime.getRuntime().availableProcessors()
                            val opts = OrtSession.SessionOptions().apply {
                                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                                setIntraOpNumThreads(cpuCount)
                                setInterOpNumThreads(cpuCount)
                            }
                            ortEnvironment = env
                            val session = env.createSession(modelFile.absolutePath, opts)
                            ortSession = session
                            Log.d(TAG, "Model inputs: ${session.inputNames.toList()}")
                            Log.d(TAG, "Model outputs: ${session.outputNames.toList()}")
                            for (name in session.inputNames) {
                                val info = session.inputInfo[name]
                                Log.d(TAG, "  input '$name': ${info?.info}")
                            }
                            for (name in session.outputNames) {
                                val info = session.outputInfo[name]
                                Log.d(TAG, "  output '$name': ${info?.info}")
                            }
                        }

                        isCustomTts = true
                        isInitialized = true
                        currentModelDir = modelDir
                        Log.d(TAG, "Custom Kokoro ONNX TTS initialized successfully with model: $selectedModel")
                        // Preload CMU dict in background so first sentence has no dict-load stall
                        startAudioTrackPlayback()
                        startSynthesisWorker()
                        flushPendingQueue()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error building custom ONNX TTS wrapper: ${e.message}", e)
                        initializeSystemTts()
                    }
                } else {
                    Log.w(TAG, "Custom TTS model file not found or empty, falling back to system default")
                    initializeSystemTts()
                }
            } else {
                Log.d(TAG, "selectedModel is null, initializing system default TTS")
                initializeSystemTts()
            }
        }
    }

    private fun initializeSystemTts() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                isInitialized = true
                flushPendingQueue()
                val locale = Locale.getDefault()
                val result = tts?.setLanguage(locale)

                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "Language not supported: $locale, falling back to English")
                    tts?.setLanguage(Locale.ENGLISH)
                }

                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        inFlightUtterances.incrementAndGet()
                        _isSpeaking.value = true
                    }

                    override fun onDone(utteranceId: String?) {
                        val remaining = inFlightUtterances.decrementAndGet().coerceAtLeast(0)
                        if (remaining == 0) {
                            _isSpeaking.value = false
                        }
                    }

                    override fun onError(utteranceId: String?) {
                        Log.e(TAG, "TTS error for utterance: $utteranceId")
                        val remaining = inFlightUtterances.decrementAndGet().coerceAtLeast(0)
                        if (remaining == 0) {
                            _isSpeaking.value = false
                        }
                    }
                })

                Log.d(TAG, "System TTS initialized successfully")
            } else {
                Log.e(TAG, "System TTS initialization failed")
            }
        }
    }

    fun speak(text: String) {
        Log.d(TAG, "speak() called with text: '$text'")
        if (!isInitialized) {
            Log.w(TAG, "TTS not initialized")
            return
        }

        if (text.isBlank()) {
            Log.w(TAG, "Cannot speak empty text")
            return
        }

        stop()
        textBuffer.clear()
        _currentText.value = text

        if (isCustomTts) {
            val cleanText = cleanTextForTts(text)
            val sentences = splitIntoSentences(cleanText)
            Log.d(TAG, "speak() custom TTS: split into ${sentences.size} sentences")

            activeJobsCount.addAndGet(sentences.size)
            _isSpeaking.value = activeJobsCount.get() > 0
            enqueueSentences(sentences)
        } else {
            val cleanText = cleanTextForTts(text)
            val chunks = splitIntoChunks(cleanText)
            Log.d(TAG, "speak() system TTS: split into ${chunks.size} chunks")

            chunks.forEachIndexed { index, chunk ->
                val id = "utterance_${utteranceId++}"
                val queueMode = if (index == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
                tts?.speak(chunk, queueMode, null, id)
            }
        }
    }

    fun speakAppend(text: String) {
        Log.d(TAG, "speakAppend() called with text: '$text'")
        if (!isInitialized) {
            Log.w(TAG, "TTS not initialized, buffering for later")
            synchronized(pendingSpeakQueue) { pendingSpeakQueue.add(text) }
            return
        }

        if (text.isBlank()) {
            Log.w(TAG, "Cannot speak empty text")
            return
        }

        _currentText.value = text

        if (isCustomTts) {
            val cleanText = cleanTextForTts(text)
            val sentences = splitIntoSentences(cleanText)
            Log.d(TAG, "speakAppend() custom TTS: split into ${sentences.size} sentences")

            activeJobsCount.addAndGet(sentences.size)
            _isSpeaking.value = activeJobsCount.get() > 0
            enqueueSentences(sentences)
        } else {
            val cleanText = cleanTextForTts(text)
            val chunks = splitIntoChunks(cleanText)
            Log.d(TAG, "speakAppend() system TTS: split into ${chunks.size} chunks")

            chunks.forEach { chunk ->
                val id = "utterance_${utteranceId++}"
                tts?.speak(chunk, TextToSpeech.QUEUE_ADD, null, id)
            }
        }
    }

    private fun startAudioTrackPlayback() {
        val sampleRate = 24000
        val minBuf = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, sampleRate * 2 * 4) // at least 4 seconds (samples * bytes/sample * seconds)
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        track.play()
        audioTrack = track

        playbackJob = synthesisScope.launch(Dispatchers.IO) {
            while (isActive) {
                // poll with timeout so coroutine cancellation is checked every 50ms
                // (take() blocks forever and ignores cancellation, causing first-sentence skip)
                val pcm = pcmQueue.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS)
                    ?: continue
                if (!isActive) break // re-check after unblocking
                val bb = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                for (s in pcm) bb.putShort(s)
                val bytes = bb.array()
                var offset = 0
                while (offset < bytes.size && isActive) {
                    val written = track.write(bytes, offset, bytes.size - offset)
                    if (written <= 0) break
                    offset += written
                }
                val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
                if (rem == 0) _isSpeaking.value = false
            }
        }
    }

    // Enqueue sentences for the serial synthesis worker. All callers funnel
    // sentences through sentenceChannel; the single worker processes them in
    // strict FIFO order so audio is never out-of-sequence.
    private fun enqueueSentences(sentences: List<String>) {
        for (sentence in sentences) {
            sentenceChannel.trySend(sentence)
        }
    }

    private fun startSynthesisWorker() {
        synthesisWorkerJob?.cancel()
        synthesisWorkerJob = synthesisScope.launch {
            val voice = themePreferences.selectedTtsVoice.first()
            val modelDir = currentModelDir ?: return@launch

            val voiceFile = resolveVoiceFile(modelDir, voice) ?: run {
                Log.e(TAG, "No voice style file found in synthesis worker")
                return@launch
            }

            val voiceStyles: FloatArray = voiceStyleCache.getOrPut(voiceFile.absolutePath) {
                val bytes = voiceFile.readBytes()
                val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                FloatArray(buf.remaining()).also { buf.get(it) }
            }
            val maxStyleRows = voiceStyles.size / 256

            loadCmuDictIfNeeded()

            for (sentence in sentenceChannel) {
                if (!isActive) {
                    val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
                    if (rem == 0) _isSpeaking.value = false
                    continue
                }
                if (sentence.isBlank()) {
                    val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
                    if (rem == 0) _isSpeaking.value = false
                    continue
                }

                val phonemes = englishToPhonemes(sentence)
                val tokens = mutableListOf<Int>()
                tokens.add(0)
                for (ch in phonemes) {
                    TOKEN_MAP[ch]?.let { tokens.add(it) }
                }
                tokens.add(0)

                val seqLen = tokens.size.coerceIn(0, maxStyleRows - 1)
                val styleVector = FloatArray(256).also {
                    System.arraycopy(voiceStyles, seqLen * 256, it, 0, 256)
                }
                Log.d(TAG, "sentence='$sentence' tokens=${tokens.size} seqLen=$seqLen")

                synthesizeAndQueue(tokens, styleVector)
            }
        }
    }

    private suspend fun synthesizeAndQueue(tokens: List<Int>, styleVector: FloatArray) {
        try {
            if (!coroutineContext.isActive) {
                val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
                if (rem == 0) _isSpeaking.value = false
                return
            }
            val env = ortEnvironment ?: run {
                val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
                if (rem == 0) _isSpeaking.value = false
                return
            }
            val session = ortSession ?: run {
                val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
                if (rem == 0) _isSpeaking.value = false
                return
            }

            val inputIdsBuf = LongBuffer.wrap(tokens.map { it.toLong() }.toLongArray())
            val inputIdsTensor = OnnxTensor.createTensor(env, inputIdsBuf, longArrayOf(1, tokens.size.toLong()))
            val styleBuf = FloatBuffer.wrap(styleVector)
            val styleTensor = OnnxTensor.createTensor(env, styleBuf, longArrayOf(1, 256))
            val speedBuf = FloatBuffer.wrap(floatArrayOf(currentSpeechRate))
            val speedTensor = OnnxTensor.createTensor(env, speedBuf, longArrayOf(1))

            val audioData = try {
                session.run(mapOf("input_ids" to inputIdsTensor, "style" to styleTensor, "speed" to speedTensor)).use { result ->
                    val iterator = result.iterator()
                    if (!iterator.hasNext()) return@use null
                    val outputTensor = iterator.next().value as OnnxTensor
                    when (val tensorType = outputTensor.info.type) {
                        ai.onnxruntime.OnnxJavaType.FLOAT -> {
                            val fb = outputTensor.floatBuffer
                            FloatArray(fb.remaining()).also { fb.get(it) }
                        }
                        ai.onnxruntime.OnnxJavaType.FLOAT16 -> {
                            val bb = outputTensor.byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
                            val count = bb.remaining() / 2
                            FloatArray(count) { fp16ToFloat(bb.getShort()) }
                        }
                        else -> { Log.e(TAG, "Unsupported output type: $tensorType"); null }
                    }
                }
            } finally {
                inputIdsTensor.close(); styleTensor.close(); speedTensor.close()
            }

            if (audioData != null && audioData.isNotEmpty() && coroutineContext.isActive) {
                val peak = audioData.maxOfOrNull { kotlin.math.abs(it) } ?: 0f
                val scale = if (peak > 0.001f) 0.9f / peak else 1.0f
                val pcmData = ShortArray(audioData.size) { i ->
                    ((audioData[i] * scale).coerceIn(-1f, 1f) * 32767f).toInt().toShort()
                }
                Log.d(TAG, "Generated ${audioData.size} samples (${audioData.size / 24000.0}s)")
                pcmQueue.put(pcmData)
                _isSpeaking.value = true
            } else {
                Log.e(TAG, "Audio synthesis failed or was cancelled")
                val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
                if (rem == 0) _isSpeaking.value = false
            }
        } catch (ce: kotlinx.coroutines.CancellationException) {
            val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
            if (rem == 0) _isSpeaking.value = false
            throw ce
        } catch (e: Exception) {
            Log.e(TAG, "Error in synthesizeAndQueue: ${e.message}", e)
            val rem = activeJobsCount.decrementAndGet().coerceAtLeast(0)
            if (rem == 0) _isSpeaking.value = false
        }
    }

    private fun stopAudioTrack() {
        synthesisWorkerJob?.cancel()
        synthesisWorkerJob = null
        // Drain pending sentences so the channel is empty for next use
        while (!sentenceChannel.isEmpty) sentenceChannel.tryReceive()
        playbackJob?.cancel()
        playbackJob = null
        pcmQueue.clear()
        try { audioTrack?.pause() } catch (_: Exception) {}
        try { audioTrack?.flush() } catch (_: Exception) {}
        try { audioTrack?.stop() } catch (_: Exception) {}
        try { audioTrack?.release() } catch (_: Exception) {}
        audioTrack = null
    }

    fun addStreamingText(partialText: String) {
        if (!isInitialized) {
            Log.d(TAG, "addStreamingText ignored: TTS not initialized yet. text: $partialText")
            return
        }

        val filtered = filterThinkBlocks(partialText)
        if (filtered.isEmpty()) return

        textBuffer.append(filtered)

        val bufferedText = textBuffer.toString()
        val lastChar = bufferedText.lastOrNull()

        if (lastChar != null && lastChar in SENTENCE_DELIMITERS) {
            val sentences = extractCompleteSentences(bufferedText)

            sentences.forEach { sentence ->
                if (sentence.isNotBlank()) {
                    speakAppend(sentence)
                }
            }

            val remaining = bufferedText.substringAfterLast(lastChar, "")
            textBuffer.clear()
            textBuffer.append(remaining)
        }
    }

    private fun filterThinkBlocks(text: String): String {
        thinkBlockBuffer.append(text)
        val combined = thinkBlockBuffer.toString()
        thinkBlockBuffer.clear()

        val openTags = listOf("​​THINK​​", "<think>")
        val closeTags = listOf("​​ENDTHINK​​", "</think>")

        val result = StringBuilder()
        var pos = 0
        while (pos < combined.length) {
            if (!insideThinkBlock) {
                var nearestOpen = -1
                var nearestOpenLen = 0
                var nearestCloseLen = 0
                for (i in openTags.indices) {
                    val idx = combined.indexOf(openTags[i], pos, ignoreCase = true)
                    if (idx != -1 && (nearestOpen == -1 || idx < nearestOpen)) {
                        nearestOpen = idx
                        nearestOpenLen = openTags[i].length
                        nearestCloseLen = closeTags[i].length
                    }
                }
                if (nearestOpen == -1) {
                    result.append(combined.substring(pos))
                    break
                }
                result.append(combined.substring(pos, nearestOpen))
                insideThinkBlock = true
                pos = nearestOpen + nearestOpenLen
            } else {
                var nearestClose = -1
                var nearestCloseLen = 0
                for (closeTag in closeTags) {
                    val idx = combined.indexOf(closeTag, pos, ignoreCase = true)
                    if (idx != -1 && (nearestClose == -1 || idx < nearestClose)) {
                        nearestClose = idx
                        nearestCloseLen = closeTag.length
                    }
                }
                if (nearestClose == -1) {
                    thinkBlockBuffer.append(combined.substring(pos))
                    break
                }
                insideThinkBlock = false
                pos = nearestClose + nearestCloseLen
            }
        }
        return result.toString()
    }

    private fun flushPendingQueue() {
        val pending = synchronized(pendingSpeakQueue) {
            val copy = pendingSpeakQueue.toList()
            pendingSpeakQueue.clear()
            copy
        }
        if (pending.isNotEmpty()) {
            Log.d(TAG, "Flushing ${pending.size} buffered speakAppend calls")
            pending.forEach { speakAppend(it) }
        }
    }

    fun flushStreamingBuffer() {
        if (!isInitialized) {
            Log.d(TAG, "flushStreamingBuffer ignored: TTS not initialized yet")
            return
        }

        val remaining = textBuffer.toString().trim()
        Log.d(TAG, "flushStreamingBuffer: remaining = '$remaining'")
        if (remaining.isNotBlank()) {
            speakAppend(remaining)
        }
        textBuffer.clear()
    }

    fun stop() {
        synchronized(pendingSpeakQueue) { pendingSpeakQueue.clear() }
        if (isCustomTts) {
            synthesisScope.coroutineContext[Job]?.cancelChildren()
            stopAudioTrack()
            // Restart fresh AudioTrack + serial synthesis worker for next utterance
            startAudioTrackPlayback()
            startSynthesisWorker()
            activeJobsCount.set(0)
            insideThinkBlock = false
            thinkBlockBuffer.clear()
            _isSpeaking.value = false
            _currentText.value = ""
        } else {
            tts?.stop()
            textBuffer.clear()
            inFlightUtterances.set(0)
            _isSpeaking.value = false
            _currentText.value = ""
        }
    }

    fun pause() {
        if (isCustomTts) {
            try { audioTrack?.pause() } catch (_: Exception) {}
            _isSpeaking.value = false
        } else {
            tts?.stop()
            _isSpeaking.value = false
        }
    }

    fun isSpeaking(): Boolean {
        return if (isCustomTts) {
            _isSpeaking.value
        } else {
            tts?.isSpeaking == true
        }
    }

    fun setSpeechRate(rate: Float) {
        currentSpeechRate = rate.coerceIn(0.5f, 2.0f)
        tts?.setSpeechRate(currentSpeechRate)
    }

    fun setPitch(pitch: Float) {
        tts?.setPitch(pitch.coerceIn(0.5f, 2.0f))
    }

    fun setLanguage(locale: Locale) {
        tts?.setLanguage(locale)
    }

    private fun cleanTextForTts(text: String): String {
        return text
            .replace(Regex("​​THINK​​[\\s\\S]*?​​ENDTHINK​​"), "")
            .replace(Regex("<think>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE), "")
            .replace("​", "")
            .replace(Regex("\\*\\*(.+?)\\*\\*"), "$1")
            .replace(Regex("_(.+?)_"), "$1")
            .replace(Regex("\\*(.+?)\\*"), "$1")
            .replace(Regex("^#+\\s+"), "")
            .replace(Regex("```[\\s\\S]*?```"), "")
            .replace(Regex("`(.+?)`"), "$1")
            .replace(Regex("\\[([^\\]]+)\\]\\([^)]+\\)"), "$1")
            .replace(Regex("^[•\\-*]\\s+", RegexOption.MULTILINE), "")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun extractCompleteSentences(text: String): List<String> {
        val sentences = mutableListOf<String>()
        var currentSentence = StringBuilder()

        for (char in text) {
            currentSentence.append(char)
            if (char in SENTENCE_DELIMITERS) {
                sentences.add(currentSentence.toString())
                currentSentence = StringBuilder()
            }
        }

        return sentences
    }

    private fun splitIntoChunks(text: String, maxLength: Int = 4000): List<String> {
        if (text.length <= maxLength) {
            return listOf(text)
        }

        val chunks = mutableListOf<String>()
        var currentChunk = StringBuilder()
        val sentences = text.split(Regex("(?<=[.!?])\\s+"))

        for (sentence in sentences) {
            if (currentChunk.length + sentence.length > maxLength) {
                if (currentChunk.isNotEmpty()) {
                    chunks.add(currentChunk.toString())
                    currentChunk = StringBuilder()
                }

                if (sentence.length > maxLength) {
                    val words = sentence.split(" ")
                    for (word in words) {
                        if (currentChunk.length + word.length + 1 > maxLength) {
                            chunks.add(currentChunk.toString())
                            currentChunk = StringBuilder()
                        }
                        if (currentChunk.isNotEmpty()) currentChunk.append(" ")
                        currentChunk.append(word)
                    }
                } else {
                    currentChunk.append(sentence)
                }
            } else {
                if (currentChunk.isNotEmpty()) currentChunk.append(" ")
                currentChunk.append(sentence)
            }
        }

        if (currentChunk.isNotEmpty()) {
            chunks.add(currentChunk.toString())
        }

        return chunks
    }

    private fun splitIntoSentences(text: String): List<String> {
        if (text.isBlank()) return emptyList()
        val maxChunkLength = 120
        val rawChunks = text.split(Regex("(?<=[.!?。！？\\n])\\s*"))
        val result = mutableListOf<String>()
        for (chunk in rawChunks) {
            val trimmed = chunk.trim()
            if (trimmed.isEmpty()) continue
            if (trimmed.length > maxChunkLength) {
                // Only hard-split on word boundary, never on comma/semicolon
                var start = 0
                while (start < trimmed.length) {
                    val end = (start + maxChunkLength).coerceAtMost(trimmed.length)
                    // Back up to last space to avoid splitting mid-word
                    val breakAt = if (end < trimmed.length) {
                        val spaceIdx = trimmed.lastIndexOf(' ', end)
                        if (spaceIdx > start) spaceIdx else end
                    } else end
                    val sub = trimmed.substring(start, breakAt).trim()
                    if (sub.isNotEmpty()) result.add(sub)
                    start = breakAt
                }
            } else {
                result.add(trimmed)
            }
        }
        return result
    }

    private fun resolveVoiceFile(modelDir: File, requestedVoice: String): File? {
        // Collect all Kokoro model dirs to search — voice .bin files are shared across variants
        val modelsRoot = modelDir.parentFile ?: modelDir
        val searchDirs = modelsRoot.listFiles()
            ?.filter { it.isDirectory && it.name.startsWith("Kokoro") }
            ?.sortedByDescending { it.name == modelDir.name } // prefer current model dir first
            ?: listOf(modelDir)

        for (dir in searchDirs) {
            val requested = File(dir, "$requestedVoice.bin")
            if (requested.exists() && requested.length() > 0) return requested
        }

        val preferredFallbacks = listOf("af_sky", "af")
        for (voice in preferredFallbacks) {
            for (dir in searchDirs) {
                val file = File(dir, "$voice.bin")
                if (file.exists() && file.length() > 0) {
                    Log.w(TAG, "Requested voice $requestedVoice missing; falling back to ${file.absolutePath}")
                    return file
                }
            }
        }

        val firstInstalled = searchDirs.flatMap { dir ->
            dir.listFiles()?.filter { it.isFile && it.name.endsWith(".bin") && it.length() > 0 } ?: emptyList()
        }.minByOrNull { it.name }
        if (firstInstalled != null) {
            Log.w(TAG, "Requested voice $requestedVoice missing; falling back to ${firstInstalled.absolutePath}")
        }
        return firstInstalled
    }

    private fun clearTempTtsFiles() {
        try {
            val files = context.cacheDir.listFiles { _, name -> name.startsWith("tts_chunk_") || name.startsWith("tts_stream_") }
            files?.forEach { file ->
                try { file.delete() } catch (_: Exception) {}
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning temp files: ${e.message}")
        }
    }

    // CMU dict: WORD -> IPA string, loaded lazily from cmudict_ipa.dict in any Kokoro model dir
    private var cmuDict: Map<String, String>? = null

    private fun loadCmuDictIfNeeded() {
        if (cmuDict != null) return
        val modelsRoot = currentModelDir?.parentFile ?: return
        val dictFile = modelsRoot.listFiles()
            ?.filter { it.isDirectory && it.name.startsWith("Kokoro") }
            ?.flatMap { dir -> listOf(File(dir, "cmudict_ipa.dict")) }
            ?.firstOrNull { it.exists() && it.length() > 0 }
            ?: return
        try {
            val map = HashMap<String, String>(130000)
            dictFile.bufferedReader().forEachLine { line ->
                if (line.startsWith(";;;") || line.isBlank()) return@forEachLine
                val tab = line.indexOf('\t')
                if (tab > 0) {
                    val word = line.substring(0, tab).trim().uppercase()
                    val ipa = line.substring(tab + 1).split(",").first().trim()
                    map[word] = ipa
                }
            }
            cmuDict = map
            Log.d(TAG, "Loaded CMU dict: ${map.size} entries from ${dictFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load CMU dict: ${e.message}")
        }
    }

    private fun englishToPhonemes(text: String): String {
        loadCmuDictIfNeeded()
        val dict = cmuDict

        val validChars = TOKEN_MAP.keys.toSet()
        val result = StringBuilder()

        // Split into word/punctuation tokens
        val tokens = Regex("[a-zA-Z']+|[.,!?;:\"\\s]").findAll(text)
        for (match in tokens) {
            val token = match.value
            when {
                token.isBlank() -> result.append(' ')
                token.matches(Regex("[.,!?;:\"]")) -> result.append(token)
                token.matches(Regex("[a-zA-Z']+")) -> {
                    val ipa = if (dict != null) {
                        dict[token.uppercase().trimEnd('\'')]
                    } else null

                    val phonemes = if (ipa != null) {
                        ipa
                    } else {
                        try {
                            com.github.medavox.ipa_transcribers.Language.ENGLISH.transcriber.transcribe(token)
                        } catch (e: Exception) {
                            token
                        }
                    }
                    // Apply en-us adjustments and filter to valid tokens
                    for (char in phonemes) {
                        if (char in validChars) result.append(char)
                    }
                }
            }
        }
        return result.toString().replace(Regex(" +"), " ").trim()
    }

    private fun fp16ToFloat(half: Short): Float {
        val h = half.toInt() and 0xFFFF
        val sign = (h ushr 15) shl 31
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x3FF
        val bits = when {
            exp == 0 && mant == 0 -> sign
            exp == 0x1F && mant != 0 -> sign or 0x7FC00000
            exp == 0x1F -> sign or 0x7F800000
            exp == 0 -> {
                var e = -14; var m = mant shl 1
                while (m and 0x400 == 0) { e--; m = m shl 1 }
                sign or ((e + 127) shl 23) or ((m and 0x3FF) shl 13)
            }
            else -> sign or ((exp + 112) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }

    private fun writeWavFile(pcmData: ShortArray, sampleRate: Int, outFile: File) {
        val totalAudioLen = pcmData.size * 2
        val totalDataLen = totalAudioLen + 36
        val longSampleRate = sampleRate.toLong()
        val channels = 1
        val byteRate = sampleRate * channels * 2

        val header = ByteArray(44)
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (longSampleRate and 0xff).toByte()
        header[25] = ((longSampleRate shr 8) and 0xff).toByte()
        header[26] = ((longSampleRate shr 16) and 0xff).toByte()
        header[27] = ((longSampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = 2
        header[33] = 0
        header[34] = 16
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

        outFile.outputStream().use { fos ->
            fos.write(header)
            val byteBuf = ByteBuffer.allocate(pcmData.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in pcmData) {
                byteBuf.putShort(sample)
            }
            fos.write(byteBuf.array())
        }
    }

    fun shutdown() {
        synchronized(pendingSpeakQueue) { pendingSpeakQueue.clear() }
        synthesisScope.coroutineContext[Job]?.cancel()
        stopAudioTrack()
        tts?.shutdown()
        tts = null
        try { ortSession?.close() } catch (_: Exception) {}
        try { ortEnvironment?.close() } catch (_: Exception) {}
        ortSession = null
        ortEnvironment = null
        isInitialized = false
        cmuDict = null
        voiceStyleCache.clear()
        clearTempTtsFiles()
    }
}
