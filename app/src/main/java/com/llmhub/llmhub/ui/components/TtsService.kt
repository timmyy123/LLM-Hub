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
        
        /*
        /**
         * Detect language from text using character-based heuristics.
         * Prioritizes the 14 supported app locales.
         * 
         * NOTE: Currently not used - TTS uses app language setting instead.
         * Kept here for potential future use.
         */
        // fun detectLanguage(text: String): Locale {
        //     if (text.isBlank()) return Locale.getDefault()
            
        //     // Sample first 200 chars for detection
        //     val sample = text.take(200)
            
        //     // Count character types
        //     var arabicChars = 0
        //     var cyrillicChars = 0
        //     var greekChars = 0
        //     var hangulChars = 0
        //     var hiraganaKatakanaChars = 0
        //     var latinChars = 0
            
        //     for (char in sample) {
        //         when (char.code) {
        //             in 0x0600..0x06FF, in 0x0750..0x077F, in 0xFB50..0xFDFF, in 0xFE70..0xFEFF -> arabicChars++
        //             in 0x0400..0x04FF, in 0x0500..0x052F -> cyrillicChars++
        //             in 0x0370..0x03FF, in 0x1F00..0x1FFF -> greekChars++
        //             in 0xAC00..0xD7AF -> hangulChars++
        //             in 0x3040..0x309F, in 0x30A0..0x30FF -> hiraganaKatakanaChars++
        //             in 0x0041..0x005A, in 0x0061..0x007A, in 0x00C0..0x00FF, in 0x0100..0x017F -> latinChars++
        //         }
        //     }
            
        //     val totalChars = sample.length
            
        //     // Detect based on character percentages (threshold: 20%)
        //     return when {
        //         arabicChars > totalChars * 0.2 -> Locale("ar")
        //         cyrillicChars > totalChars * 0.2 -> Locale("ru")
        //         greekChars > totalChars * 0.2 -> Locale("el")
        //         hangulChars > totalChars * 0.2 -> Locale("ko")
        //         hiraganaKatakanaChars > totalChars * 0.2 -> Locale("ja")
        //         latinChars > totalChars * 0.3 -> {
        //             // For Latin scripts, check common words for specific languages
        //             val lower = sample.lowercase()
                    
        //             // Count matches for each language (longer, more specific words get priority)
        //             var deScore = 0
        //             var esScore = 0
        //             var frScore = 0
        //             var itScore = 0
        //             var ptScore = 0
        //             var plScore = 0
        //             var trScore = 0
        //             var idScore = 0
                    
        //             // German - distinctive words
        //             if (lower.containsAny("und", "nicht", "ist", "der", "die", "das", "mit", "für", "auch", "aber", "oder", "wird", "wurden", "worden")) deScore += 3
        //             if (lower.containsAny("ich", "Sie", "sie", "wir", "ihm", "ihn", "vom", "zum", "zur")) deScore += 2
                    
        //             // Spanish - distinctive words
        //             if (lower.containsAny("que", "está", "son", "están", "pero", "porque", "como", "muy", "donde", "cuando", "sobre", "entre")) esScore += 3
        //             if (lower.containsAny("el", "la", "los", "las", "del", "por", "para", "con", "sin", "ser", "estar")) esScore += 2
                    
        //             // French - distinctive words
        //             if (lower.containsAny("est", "sont", "être", "avec", "dans", "pour", "qui", "que", "mais", "où", "aussi", "très", "plus", "tous", "toutes", "leurs", "votre", "notre")) frScore += 3
        //             if (lower.containsAny("le", "la", "les", "un", "une", "des", "du", "au", "aux", "ce", "cette", "ces", "vous", "nous", "elle", "ils", "elles")) frScore += 2
                    
        //             // Italian - distinctive words  
        //             if (lower.containsAny("che", "sono", "è", "non", "essere", "anche", "più", "tutti", "quale", "dove", "quando", "quindi", "perché", "però", "ancora")) itScore += 3
        //             if (lower.containsAny("il", "lo", "la", "gli", "le", "del", "della", "dei", "delle", "con", "per", "questa", "questo", "questi")) itScore += 2
                    
        //             // Portuguese - distinctive words
        //             if (lower.containsAny("que", "não", "são", "está", "estão", "também", "muito", "mais", "com", "ser", "onde", "quando", "porque", "mas", "ainda", "sobre")) ptScore += 3
        //             if (lower.containsAny("o", "os", "as", "um", "uma", "do", "da", "dos", "das", "para", "pelo", "pela", "pelos", "pelas", "você", "ele", "ela", "eles", "elas")) ptScore += 2
                    
        //             // Polish - distinctive words
        //             if (lower.containsAny("jest", "nie", "się", "był", "była", "było", "były", "można", "który", "która", "które", "bardzo", "zawsze", "teraz", "tylko", "jeszcze", "przez")) plScore += 3
        //             if (lower.containsAny("i", "w", "z", "na", "do", "po", "dla", "od", "za", "jego", "jej", "ich", "tym", "tego")) plScore += 2
                    
        //             // Turkish - distinctive words
        //             if (lower.containsAny("bir", "için", "değil", "olan", "olarak", "gibi", "çok", "daha", "ama", "şu", "bu", "ile", "kadar", "diye")) trScore += 3
        //             if (lower.containsAny("ve", "da", "de", "mi", "mı", "var", "yok", "ben", "sen", "biz", "onlar")) trScore += 2
                    
        //             // Indonesian - distinctive words
        //             if (lower.containsAny("yang", "tidak", "adalah", "untuk", "dengan", "dari", "pada", "akan", "juga", "karena", "seperti", "atau", "sudah", "telah", "sebagai", "tersebut")) idScore += 3
        //             if (lower.containsAny("dan", "di", "ke", "ini", "itu", "ada", "oleh", "dalam", "dapat", "bisa", "saya", "kami", "mereka", "anda", "kita")) idScore += 2
                    
        //             // Log detection scores for debugging
        //             Log.d(TAG, "Language scores - DE:$deScore ES:$esScore FR:$frScore IT:$itScore PT:$ptScore PL:$plScore TR:$trScore ID:$idScore")
                    
        //             // Return language with highest score
        //             val maxScore = maxOf(deScore, esScore, frScore, itScore, ptScore, plScore, trScore, idScore)
        //             when {
        //                 maxScore == 0 -> Locale("en") // No matches, default to English
        //                 deScore == maxScore -> Locale("de")
        //                 esScore == maxScore -> Locale("es")
        //                 frScore == maxScore -> Locale("fr")
        //                 itScore == maxScore -> Locale("it")
        //                 ptScore == maxScore -> Locale("pt")
        //                 plScore == maxScore -> Locale("pl")
        //                 trScore == maxScore -> Locale("tr")
        //                 idScore == maxScore -> Locale("id")
        //                 else -> Locale("en")
        //             }
        //         }
        //         // Fallback to device locale
        //         else -> Locale.getDefault()
        //     }
        // }
        
        private fun String.containsAny(vararg words: String): Boolean {
            val text = " $this "
            return words.any { word -> text.contains(" $word ", ignoreCase = true) }
        }
        */
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
     */
    fun speak(text: String) {
        if (!isInitialized || tts == null) {
            Log.w(TAG, "TTS not initialized")
            return
        }
        
        if (text.isBlank()) {
            Log.w(TAG, "Cannot speak empty text")
            return
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
    fun addStreamingText(partialText: String) {
        if (!isInitialized || tts == null) return
        
        textBuffer.append(partialText)
        
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
     * Falls back to device default language if requested locale is not supported.
     */
    fun setLanguage(locale: Locale) {
        val result = tts?.setLanguage(locale)
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            Log.w(TAG, "Language not supported: $locale, trying device default")
            // Fall back to device default locale
            val defaultLocale = Locale.getDefault()
            val fallbackResult = tts?.setLanguage(defaultLocale)
            if (fallbackResult == TextToSpeech.LANG_MISSING_DATA || fallbackResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w(TAG, "Device default language also not supported: $defaultLocale, using English")
                // Last resort: use English
                tts?.setLanguage(Locale.ENGLISH)
            } else {
                Log.d(TAG, "Using device default language: ${defaultLocale.language}")
            }
        } else {
            Log.d(TAG, "TTS language set to: ${locale.language}")
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
