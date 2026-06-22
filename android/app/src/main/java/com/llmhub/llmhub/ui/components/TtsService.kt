package com.llmhub.llmhub.ui.components

import android.content.Context
import android.media.MediaPlayer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.nexa.sdk.TtsWrapper
import com.nexa.sdk.bean.TtsCreateInput
import com.nexa.sdk.bean.TtsSynthesizeInput
import com.nexa.sdk.bean.TtsConfig
import com.nexa.sdk.bean.ModelConfig
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.coroutines.coroutineContext
import java.io.File
import java.util.*
import java.util.concurrent.atomic.AtomicInteger

/**
 * Text-to-Speech service for reading AI responses aloud.
 * Supports streaming TTS with sentence buffering for smooth playback.
 * Integrates system default TTS with custom GGUF models.
 */
class TtsService(private val context: Context, private val isTranslationFeature: Boolean = false) {
    
    private var tts: TextToSpeech? = null
    private var ttsWrapper: TtsWrapper? = null
    
    private var isInitialized = false
    private var isCustomTts = false
    
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
    
    private val themePreferences = ThemePreferences(context)
    private var currentSpeechRate = 1.0f
    
    // Custom TTS Audio playback state
    private val playQueue = Collections.synchronizedList(mutableListOf<File>())
    private var mediaPlayer: MediaPlayer? = null
    
    // Mutex and scopes for concurrency management
    private val ttsMutex = Mutex()
    private val synthesisScope = CoroutineScope(Dispatchers.IO + Job())
    
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
        CoroutineScope(Dispatchers.Main).launch {
            val selectedModel = if (isTranslationFeature) null else themePreferences.selectedTtsModel.first()
            if (selectedModel != null) {
                if (selectedModel.contains("eSpeak", ignoreCase = true) && !selectedModel.contains("No eSpeak", ignoreCase = true)) {
                    Log.e(TAG, "eSpeak models are not supported on Android (no native espeak-ng). Falling back to system default.")
                    themePreferences.setSelectedTtsModel(null)
                    initializeSystemTts()
                    return@launch
                }
                val model = ModelData.models.find { it.name == selectedModel }
                val localName = model?.localFileName()
                val modelFile = if (localName != null) File(File(context.filesDir, "models"), localName) else null
                
                if (modelFile != null && modelFile.exists() && modelFile.length() > 0) {
                    try {
                        val deviceId = themePreferences.selectedTtsDevice.first()
                        val nGpuLayers = if (deviceId.lowercase() == "gpu") 999 else 0
                        val config = ModelConfig(nCtx = 4096, nGpuLayers = nGpuLayers)
                        val createInput = TtsCreateInput(
                            model_name = selectedModel,
                            model_path = modelFile.absolutePath,
                            config = config,
                            plugin_id = "tts_cpp",
                            device_id = deviceId
                        )
                        val buildResult = TtsWrapper.builder()
                            .ttsCreateInput(createInput)
                            .build()
                            
                        if (buildResult.isSuccess) {
                            ttsWrapper = buildResult.getOrNull()
                            isCustomTts = true
                            isInitialized = true
                            Log.d(TAG, "Custom Kokoro GGUF TTS initialized successfully with model: $selectedModel")
                        } else {
                            Log.e(TAG, "Failed to initialize custom TTS wrapper, falling back to system TTS", buildResult.exceptionOrNull())
                            initializeSystemTts()
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error building custom TTS wrapper: ${e.message}", e)
                        initializeSystemTts()
                    }
                } else {
                    Log.w(TAG, "Custom TTS model file not found or empty, falling back to system default")
                    initializeSystemTts()
                }
            } else {
                initializeSystemTts()
            }
        }
    }
    
    private fun initializeSystemTts() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                isInitialized = true
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
    
    /**
     * Speak the given text immediately.
     * Stops any currently speaking text.
     * @param text The text to speak
     */
    fun speak(text: String) {
        if (!isInitialized) {
            Log.w(TAG, "TTS not initialized")
            return
        }
        
        if (text.isBlank()) {
            Log.w(TAG, "Cannot speak empty text")
            return
        }
        
        stop() // Stop any current speech and cancel active synthesis tasks
        textBuffer.clear()
        _currentText.value = text
        
        if (isCustomTts) {
            val cleanText = cleanTextForTts(text)
            val sentences = splitIntoSentences(cleanText)
            
            synthesisScope.launch {
                val voice = themePreferences.selectedTtsVoice.first()
                for (sentence in sentences) {
                    if (!isActive) break
                    synthesizeAndQueue(sentence, voice)
                }
            }
        } else {
            val cleanText = cleanTextForTts(text)
            val chunks = splitIntoChunks(cleanText)
            
            chunks.forEachIndexed { index, chunk ->
                val id = "utterance_${utteranceId++}"
                val queueMode = if (index == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
                tts?.speak(chunk, queueMode, null, id)
            }
        }
    }

    /**
     * Speak the given text by appending to the existing TTS queue without flushing it.
     * Used when enabling streaming so previously queued streaming utterances are not lost.
     */
    fun speakAppend(text: String) {
        if (!isInitialized) {
            Log.w(TAG, "TTS not initialized")
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

            synthesisScope.launch {
                val voice = themePreferences.selectedTtsVoice.first()
                for (sentence in sentences) {
                    if (!isActive) break
                    synthesizeAndQueue(sentence, voice)
                }
            }
        } else {
            val cleanText = cleanTextForTts(text)
            val chunks = splitIntoChunks(cleanText)

            chunks.forEach { chunk ->
                val id = "utterance_${utteranceId++}"
                tts?.speak(chunk, TextToSpeech.QUEUE_ADD, null, id)
            }
        }
    }
    
    private suspend fun synthesizeAndQueue(sentence: String, voice: String) {
        if (sentence.isBlank()) return
        try {
            val tempFile = File.createTempFile("tts_chunk_", ".wav", context.cacheDir)
            val config = TtsConfig(
                voice = voice,
                speed = currentSpeechRate,
                sampleRate = 22050
            )
            val input = TtsSynthesizeInput(
                textUtf8 = sentence,
                config = config,
                outputPath = tempFile.absolutePath
            )
            
            // Protect native call from concurrency
            val result = ttsMutex.withLock {
                if (!coroutineContext.isActive) {
                    return@withLock null
                }
                ttsWrapper?.synthesize(input)
            }
            
            if (result != null && result.isSuccess) {
                if (coroutineContext.isActive) {
                    synchronized(playQueue) {
                        val wasEmpty = playQueue.isEmpty() && mediaPlayer == null
                        playQueue.add(tempFile)
                        if (wasEmpty) {
                            CoroutineScope(Dispatchers.Main).launch {
                                playNextQueueItem()
                            }
                        }
                    }
                } else {
                    try { tempFile.delete() } catch (_: Exception) {}
                }
            } else {
                Log.e(TAG, "Synthesis failed for sentence: $sentence")
                try { tempFile.delete() } catch (_: Exception) {}
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in synthesizeAndQueue: ${e.message}", e)
        }
    }
    
    private fun playNextQueueItem() {
        synchronized(playQueue) {
            if (playQueue.isEmpty()) {
                mediaPlayer?.release()
                mediaPlayer = null
                _isSpeaking.value = false
                return
            }
            val nextFile = playQueue.removeAt(0)
            try {
                mediaPlayer?.release()
                mediaPlayer = MediaPlayer().apply {
                    setDataSource(nextFile.absolutePath)
                    setOnPreparedListener { start() }
                    setOnCompletionListener {
                        try { nextFile.delete() } catch (_: Exception) {}
                        playNextQueueItem()
                    }
                    setOnErrorListener { _, _, _ ->
                        try { nextFile.delete() } catch (_: Exception) {}
                        playNextQueueItem()
                        true
                    }
                    prepareAsync()
                }
                _isSpeaking.value = true
            } catch (e: Exception) {
                Log.e(TAG, "Error playing audio file: ${e.message}")
                try { nextFile.delete() } catch (_: Exception) {}
                playNextQueueItem()
            }
        }
    }
    
    /**
     * Add streaming text to buffer. Speaks complete sentences as they are formed.
     * This is for read-as-generating functionality.
     */
    fun addStreamingText(partialText: String) {
        if (!isInitialized) return

        // Filter out think block content before buffering
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

        // Two possible open/close pairs used by different models
        val openTags = listOf("​​THINK​​", "<think>")
        val closeTags = listOf("​​ENDTHINK​​", "</think>")

        val result = StringBuilder()
        var pos = 0
        while (pos < combined.length) {
            if (!insideThinkBlock) {
                // Find the earliest open tag
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
                // Find the earliest close tag
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
                    // Think block not closed yet — hold remainder for next call
                    thinkBlockBuffer.append(combined.substring(pos))
                    break
                }
                insideThinkBlock = false
                pos = nearestClose + nearestCloseLen
            }
        }
        return result.toString()
    }
    
    /**
     * Flush any remaining text in the buffer (called when streaming completes).
     */
    fun flushStreamingBuffer() {
        if (!isInitialized) return
        
        val remaining = textBuffer.toString().trim()
        if (remaining.isNotBlank()) {
            speakAppend(remaining)
        }
        textBuffer.clear()
    }
    
    /**
     * Stop current speech and clear queue.
     */
    fun stop() {
        if (isCustomTts) {
            synthesisScope.coroutineContext[Job]?.cancelChildren()
            synchronized(playQueue) {
                try {
                    mediaPlayer?.stop()
                } catch (_: Exception) {}
                mediaPlayer?.release()
                mediaPlayer = null

                for (file in playQueue) {
                    try { file.delete() } catch (_: Exception) {}
                }
                playQueue.clear()
            }
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
    
    /**
     * Pause speech.
     */
    fun pause() {
        if (isCustomTts) {
            synchronized(playQueue) {
                try {
                    mediaPlayer?.pause()
                } catch (_: Exception) {}
            }
            _isSpeaking.value = false
        } else {
            tts?.stop()
            _isSpeaking.value = false
        }
    }
    
    /**
     * Check if TTS is currently speaking.
     */
    fun isSpeaking(): Boolean {
        return if (isCustomTts) {
            _isSpeaking.value
        } else {
            tts?.isSpeaking == true
        }
    }
    
    /**
     * Set the speech rate (0.5 to 2.0, default is 1.0).
     */
    fun setSpeechRate(rate: Float) {
        currentSpeechRate = rate.coerceIn(0.5f, 2.0f)
        tts?.setSpeechRate(currentSpeechRate)
    }
    
    /**
     * Set the pitch (0.5 to 2.0, default is 1.0).
     */
    fun setPitch(pitch: Float) {
        tts?.setPitch(pitch.coerceIn(0.5f, 2.0f))
    }
    
    /**
     * Set the language for TTS.
     */
    fun setLanguage(locale: Locale) {
        tts?.setLanguage(locale)
    }
    
    /**
     * Clean text for TTS by removing markdown formatting and special characters.
     */
    private fun cleanTextForTts(text: String): String {
        return text
            // Strip think blocks (reasoning tokens not meant to be spoken)
            .replace(Regex("​​THINK​​[\\s\\S]*?​​ENDTHINK​​"), "")
            .replace(Regex("<think>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE), "")
            // Strip any leftover zero-width chars
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
    
    /**
     * Extract complete sentences from buffered text.
     */
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
    
    /**
     * Split text into chunks suitable for TTS (max 4000 characters per chunk).
     */
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
        val rawChunks = text.split(Regex("(?<=[.!?。！？\\n])\\s*"))
        val result = mutableListOf<String>()
        for (chunk in rawChunks) {
            val trimmed = chunk.trim()
            if (trimmed.isEmpty()) continue
            if (trimmed.length > 250) {
                // Split by minor punctuation
                val subChunks = trimmed.split(Regex("(?<=[,，;；:])\\s*"))
                for (subChunk in subChunks) {
                    val subTrimmed = subChunk.trim()
                    if (subTrimmed.isEmpty()) continue
                    if (subTrimmed.length > 250) {
                        var start = 0
                        while (start < subTrimmed.length) {
                            val end = (start + 250).coerceAtMost(subTrimmed.length)
                            val subSub = subTrimmed.substring(start, end).trim()
                            if (subSub.isNotEmpty()) {
                                result.add(subSub)
                            }
                            start = end
                        }
                    } else {
                        result.add(subTrimmed)
                    }
                }
            } else {
                result.add(trimmed)
            }
        }
        // Merge fragments until each chunk is large enough that its audio duration
        // exceeds the synthesis time of the next chunk. With RTF > 1 on typical mobile
        // hardware, each synthesis call takes ~10s regardless of chunk size due to fixed
        // model overhead. At ~0.075s of audio per char, 200 chars ≈ 15s audio, which
        // consistently exceeds observed synthesis latency and eliminates queue drain gaps.
        val minMergeLength = 200
        val merged = mutableListOf<String>()
        val current = StringBuilder()
        for (sentence in result) {
            if (current.isNotEmpty()) current.append(" ")
            current.append(sentence)
            if (current.length >= minMergeLength) {
                merged.add(current.toString())
                current.clear()
            }
        }
        if (current.isNotEmpty()) merged.add(current.toString())
        return merged
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
    
    /**
     * Shutdown TTS engine and release resources.
     */
    fun shutdown() {
        stop()
        synthesisScope.coroutineContext[Job]?.cancel()
        tts?.shutdown()
        tts = null
        CoroutineScope(Dispatchers.IO).launch {
            ttsMutex.withLock {
                try {
                    ttsWrapper?.close()
                    ttsWrapper?.destroy()
                } catch (_: Exception) {}
                ttsWrapper = null
            }
        }
        isInitialized = false
        clearTempTtsFiles()
    }
}
