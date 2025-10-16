package com.llmhub.llmhub.ui.components

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.*
import java.util.concurrent.atomic.AtomicInteger

/**
 * Text-to-Speech service for reading AI responses aloud.
 * Supports streaming TTS with sentence buffering for smooth playback.
 */
class TtsService(private val context: Context) {
    
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    
    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()
    
    private val _currentText = MutableStateFlow("")
    val currentText: StateFlow<String> = _currentText.asStateFlow()
    
    // Buffer for streaming text
    private val textBuffer = StringBuilder()
    private var utteranceId = 0
    // Track number of in-flight utterances so we can reliably set speaking=false
    private val inFlightUtterances = AtomicInteger(0)
    
    companion object {
        private const val TAG = "TtsService"
        
        // Sentence delimiters for buffering
        private val SENTENCE_DELIMITERS = setOf('.', '!', '?', '。', '！', '？')
        
        /**
         * Detect language from text using character-based heuristics.
         * Prioritizes the 14 supported app locales.
         */
        fun detectLanguage(text: String): Locale {
            if (text.isBlank()) return Locale.getDefault()
            
            // Sample first 200 chars for detection
            val sample = text.take(200)
            
            // Count character types
            var arabicChars = 0
            var cyrillicChars = 0
            var greekChars = 0
            var hangulChars = 0
            var hiraganaKatakanaChars = 0
            var latinChars = 0
            
            for (char in sample) {
                when (char.code) {
                    in 0x0600..0x06FF, in 0x0750..0x077F, in 0xFB50..0xFDFF, in 0xFE70..0xFEFF -> arabicChars++
                    in 0x0400..0x04FF, in 0x0500..0x052F -> cyrillicChars++
                    in 0x0370..0x03FF, in 0x1F00..0x1FFF -> greekChars++
                    in 0xAC00..0xD7AF -> hangulChars++
                    in 0x3040..0x309F, in 0x30A0..0x30FF -> hiraganaKatakanaChars++
                    in 0x0041..0x005A, in 0x0061..0x007A, in 0x00C0..0x00FF, in 0x0100..0x017F -> latinChars++
                }
            }
            
            val totalChars = sample.length
            
            // Detect based on character percentages (threshold: 20%)
            return when {
                arabicChars > totalChars * 0.2 -> Locale("ar")
                cyrillicChars > totalChars * 0.2 -> Locale("ru")
                greekChars > totalChars * 0.2 -> Locale("el")
                hangulChars > totalChars * 0.2 -> Locale("ko")
                hiraganaKatakanaChars > totalChars * 0.2 -> Locale("ja")
                latinChars > totalChars * 0.3 -> {
                    // For Latin scripts, check common words for specific languages
                    val lower = sample.lowercase()
                    when {
                        // German
                        lower.containsAny("der", "die", "das", "und", "ich", "ist", "nicht", "Sie", "mit", "für") -> Locale("de")
                        // Spanish
                        lower.containsAny("el", "la", "de", "que", "y", "en", "un", "ser", "se", "no", "está", "por", "para") -> Locale("es")
                        // French
                        lower.containsAny("le", "de", "un", "être", "et", "à", "il", "avoir", "ne", "je", "son", "que", "se", "qui", "ce", "dans", "en", "du", "elle", "au", "pour", "pas", "sur", "vous") -> Locale("fr")
                        // Italian
                        lower.containsAny("il", "di", "che", "è", "e", "la", "per", "un", "non", "in", "sono", "mi", "ho", "con", "si") -> Locale("it")
                        // Portuguese
                        lower.containsAny("o", "de", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele") -> Locale("pt")
                        // Polish
                        lower.containsAny("i", "w", "z", "na", "do", "nie", "się", "to", "jest", "że", "o", "co", "jak", "po", "z", "dla", "czy", "ale", "od", "za", "jego", "jej", "ich", "być", "który", "tym", "te", "są", "lub", "tylko", "przez", "może", "tego", "tak", "już", "pan", "bardzo", "może", "można", "nawet", "został", "został") -> Locale("pl")
                        // Turkish
                        lower.containsAny("bir", "ve", "bu", "da", "için", "ne", "ile", "mi", "daha", "var", "olan", "gibi", "çok", "ben", "sen", "biz", "ama", "şu", "olarak", "diye", "kadar", "değil", "yok", "mı", "ki") -> Locale("tr")
                        // Indonesian
                        lower.containsAny("yang", "dan", "di", "untuk", "dengan", "ini", "pada", "adalah", "dari", "ke", "tidak", "akan", "ada", "oleh", "saya", "sebagai", "atau", "sudah", "bisa", "itu", "seperti", "juga", "karena", "dalam", "dapat", "hal", "tersebut", "mereka", "kami", "telah", "tentang", "jika", "hanya", "sebuah", "ia", "lebih", "harus", "banyak", "anda", "bila", "kita", "semua", "bahwa") -> Locale("id")
                        // Default to English for Latin script
                        else -> Locale("en")
                    }
                }
                // Fallback to device locale
                else -> Locale.getDefault()
            }
        }
        
        private fun String.containsAny(vararg words: String): Boolean {
            val text = " $this "
            return words.any { word -> text.contains(" $word ", ignoreCase = true) }
        }
    }
    
    init {
        initializeTts()
    }
    
    private fun initializeTts() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                isInitialized = true
                // Set default language to device locale
                val locale = Locale.getDefault()
                val result = tts?.setLanguage(locale)
                
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "Language not supported: $locale, falling back to English")
                    tts?.setLanguage(Locale.ENGLISH)
                }
                
                // Set up progress listener
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        // Each utterance triggers onStart; mark speaking and increment counter
                        inFlightUtterances.incrementAndGet()
                        _isSpeaking.value = true
                    }
                    
                    override fun onDone(utteranceId: String?) {
                        // Decrement in-flight count and only mark not speaking when queue drains
                        val remaining = inFlightUtterances.decrementAndGet().coerceAtLeast(0)
                        if (remaining == 0) {
                            _isSpeaking.value = false
                        }
                    }
                    
                    override fun onError(utteranceId: String?) {
                        Log.e(TAG, "TTS error for utterance: $utteranceId")
                        // Treat errors as completion for the purpose of UI state
                        val remaining = inFlightUtterances.decrementAndGet().coerceAtLeast(0)
                        if (remaining == 0) {
                            _isSpeaking.value = false
                        }
                    }
                })
                
                Log.d(TAG, "TTS initialized successfully")
            } else {
                Log.e(TAG, "TTS initialization failed")
            }
        }
    }
    
    /**
     * Speak the given text immediately.
     * Stops any currently speaking text.
     * @param text The text to speak
     * @param autoDetectLanguage If true, automatically detect and set language before speaking
     */
    fun speak(text: String, autoDetectLanguage: Boolean = false) {
        if (!isInitialized || tts == null) {
            Log.w(TAG, "TTS not initialized")
            return
        }
        
        if (text.isBlank()) {
            Log.w(TAG, "Cannot speak empty text")
            return
        }
        
        // Auto-detect language if requested
        if (autoDetectLanguage) {
            val detectedLocale = detectLanguage(text)
            Log.d(TAG, "Auto-detected language: ${detectedLocale.language} for text: ${text.take(50)}...")
            setLanguage(detectedLocale)
        }
        
    stop() // Stop any current speech
        textBuffer.clear()
        _currentText.value = text
        
        // Clean text for TTS (remove markdown formatting)
        val cleanText = cleanTextForTts(text)
        
        // Split into chunks if text is too long
        val chunks = splitIntoChunks(cleanText)
        
        chunks.forEachIndexed { index, chunk ->
            val id = "utterance_${utteranceId++}"
            val queueMode = if (index == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            tts?.speak(chunk, queueMode, null, id)
        }
    }
    
    /**
     * Add streaming text to buffer. Speaks complete sentences as they are formed.
     * This is for read-as-generating functionality.
     */
    fun addStreamingText(partialText: String, autoDetectLanguage: Boolean = false) {
        if (!isInitialized || tts == null) return
        
        val wasEmpty = textBuffer.isEmpty()
        textBuffer.append(partialText)
        
        // Auto-detect language from first chunk (once per stream)
        if (wasEmpty && autoDetectLanguage && textBuffer.length >= 50) {
            val detectedLocale = detectLanguage(textBuffer.toString())
            Log.d(TAG, "Auto-detected language for streaming: ${detectedLocale.language}")
            setLanguage(detectedLocale)
        }
        
        // Check if we have a complete sentence
        val bufferedText = textBuffer.toString()
        val lastChar = bufferedText.lastOrNull()
        
        if (lastChar != null && lastChar in SENTENCE_DELIMITERS) {
            // Extract complete sentences
            val sentences = extractCompleteSentences(bufferedText)
            
            sentences.forEach { sentence ->
                if (sentence.isNotBlank()) {
                    val cleanText = cleanTextForTts(sentence)
                    val id = "stream_${utteranceId++}"
                    tts?.speak(cleanText, TextToSpeech.QUEUE_ADD, null, id)
                }
            }
            
            // Keep only incomplete text in buffer
            val remaining = bufferedText.substringAfterLast(lastChar, "")
            textBuffer.clear()
            textBuffer.append(remaining)
        }
    }
    
    /**
     * Flush any remaining text in the buffer (called when streaming completes).
     */
    fun flushStreamingBuffer() {
        if (!isInitialized || tts == null) return
        
        val remaining = textBuffer.toString().trim()
        if (remaining.isNotBlank()) {
            val cleanText = cleanTextForTts(remaining)
            val id = "stream_flush_${utteranceId++}"
            tts?.speak(cleanText, TextToSpeech.QUEUE_ADD, null, id)
        }
        textBuffer.clear()
    }
    
    /**
     * Stop current speech and clear queue.
     */
    fun stop() {
        tts?.stop()
        textBuffer.clear()
        inFlightUtterances.set(0)
        _isSpeaking.value = false
        _currentText.value = ""
    }
    
    /**
     * Pause speech (note: Not all TTS engines support pause/resume).
     */
    fun pause() {
        // Android TTS doesn't have native pause, so we stop
        tts?.stop()
        _isSpeaking.value = false
    }
    
    /**
     * Check if TTS is currently speaking.
     */
    fun isSpeaking(): Boolean {
        return tts?.isSpeaking == true
    }
    
    /**
     * Set the speech rate (0.5 to 2.0, default is 1.0).
     */
    fun setSpeechRate(rate: Float) {
        tts?.setSpeechRate(rate.coerceIn(0.5f, 2.0f))
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
        val result = tts?.setLanguage(locale)
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            Log.w(TAG, "Language not supported: $locale")
        }
    }
    
    /**
     * Clean text for TTS by removing markdown formatting and special characters.
     */
    private fun cleanTextForTts(text: String): String {
        return text
            // Remove markdown bold
            .replace(Regex("\\*\\*(.+?)\\*\\*"), "$1")
            // Remove markdown italic
            .replace(Regex("_(.+?)_"), "$1")
            .replace(Regex("\\*(.+?)\\*"), "$1")
            // Remove markdown headers
            .replace(Regex("^#+\\s+"), "")
            // Remove code blocks
            .replace(Regex("```[\\s\\S]*?```"), "")
            .replace(Regex("`(.+?)`"), "$1")
            // Remove links
            .replace(Regex("\\[([^\\]]+)\\]\\([^)]+\\)"), "$1")
            // Remove bullet points
            .replace(Regex("^[•\\-*]\\s+", RegexOption.MULTILINE), "")
            // Normalize whitespace
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
                
                // If a single sentence is too long, split it by words
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
    
    /**
     * Shutdown TTS engine and release resources.
     */
    fun shutdown() {
        stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
