package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.UUID
import java.util.concurrent.TimeUnit

class ScamDetectorViewModel(application: Application) : AndroidViewModel(application) {
    
    private val inferenceService = MediaPipeInferenceService(application)
    private val prefs = application.getSharedPreferences("scam_detector_prefs", android.content.Context.MODE_PRIVATE)
    
    companion object {
        private const val TAG = "ScamDetectorViewModel"
    }
    
    // OkHttpClient for URL fetching
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()
    
    // UI States
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()
    
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    private val _selectedBackend = MutableStateFlow<LlmInference.Backend?>(null)
    val selectedBackend: StateFlow<LlmInference.Backend?> = _selectedBackend.asStateFlow()
    
    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()
    
    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()
    
    private val _visionEnabled = MutableStateFlow(false)
    val visionEnabled: StateFlow<Boolean> = _visionEnabled.asStateFlow()
    
    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()
    
    private val _isFetchingUrl = MutableStateFlow(false)
    val isFetchingUrl: StateFlow<Boolean> = _isFetchingUrl.asStateFlow()
    
    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()
    
    private val _inputImageUri = MutableStateFlow<Uri?>(null)
    val inputImageUri: StateFlow<Uri?> = _inputImageUri.asStateFlow()
    
    private val _outputText = MutableStateFlow("")
    val outputText: StateFlow<String> = _outputText.asStateFlow()
    
    private val _loadError = MutableStateFlow<String?>(null)
    val loadError: StateFlow<String?> = _loadError.asStateFlow()
    
    private var analyzingJob: Job? = null
    
    init {
        loadAvailableModels()
        loadSavedSettings()
    }
    
    private fun loadSavedSettings() {
        // Restore backend (store enum name, fallback to GPU)
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
        
        // Restore vision setting
        _visionEnabled.value = prefs.getBoolean("vision_enabled", false)
    }
    
    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value?.name)
            putBoolean("vision_enabled", _visionEnabled.value)
            apply()
        }
    }
    
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val available = ModelAvailabilityProvider.loadAvailableModels(context)
                .filter { it.category != "embedding" }
            _availableModels.value = available
            
            // Restore saved model or use first as default
            val savedModelName = prefs.getString("selected_model_name", null)
            if (savedModelName != null) {
                val savedModel = available.find { it.name == savedModelName }
                if (savedModel != null) {
                    _selectedModel.value = savedModel
                }
            }
            
            if (available.isNotEmpty() && _selectedModel.value == null) {
                _selectedModel.value = available.first()
            }
        }
    }
    
    fun selectModel(model: LLMModel) {
        // Unload current model before switching
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedModel.value = model
        _isModelLoaded.value = false
        
        // Reset vision if model doesn't support it
        if (!model.supportsVision) {
            _visionEnabled.value = false
        }
        
        // Only auto-select backend if none is selected yet
        if (_selectedBackend.value == null) {
            _selectedBackend.value = if (model.supportsGpu) {
                LlmInference.Backend.GPU
            } else {
                LlmInference.Backend.CPU
            }
        }
        
        saveSettings()
    }
    
    fun selectBackend(backend: LlmInference.Backend) {
        // Unload current model before switching backend
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        _isModelLoaded.value = false
        saveSettings()
    }
    
    fun toggleVision(enabled: Boolean) {
        // Unload current model before changing vision setting
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _visionEnabled.value = enabled
        _isModelLoaded.value = false
        // Clear image if vision is disabled
        if (!enabled) {
            _inputImageUri.value = null
        }
        saveSettings()
    }
    
    fun setInputText(text: String) {
        _inputText.value = text
    }
    
    fun setInputImageUri(uri: Uri?) {
        _inputImageUri.value = uri
    }
    
    fun clearError() {
        _loadError.value = null
    }
    
    fun loadModel() {
        val model = _selectedModel.value ?: return
        val backend = _selectedBackend.value ?: return
        
        // Prevent concurrent loads
        if (_isLoadingModel.value || _isModelLoaded.value) {
            Log.d(TAG, "Model already loading or loaded, ignoring duplicate request")
            return
        }
        
        viewModelScope.launch {
            _isLoadingModel.value = true
            _loadError.value = null
            
            try {
                // Unload any existing model first
                inferenceService.unloadModel()
                
                // Load the selected model with vision setting
                val disableVision = !_visionEnabled.value
                val success = inferenceService.loadModel(
                    model = model,
                    preferredBackend = backend,
                    disableVision = disableVision,
                    disableAudio = true
                )
                
                _isModelLoaded.value = success
                if (success) {
                    Log.d(TAG, "Model loaded successfully: ${model.name}")
                } else {
                    _loadError.value = "Failed to load model"
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model", e)
                _loadError.value = "Failed to load model: ${e.message}"
                _isModelLoaded.value = false
            } finally {
                _isLoadingModel.value = false
            }
        }
    }
    
    fun unloadModel() {
        viewModelScope.launch {
            try {
                inferenceService.unloadModel()
                _isModelLoaded.value = false
                Log.d(TAG, "Model unloaded")
            } catch (e: Exception) {
                Log.e(TAG, "Error unloading model", e)
            }
        }
    }
    
    fun cancelAnalysis() {
        analyzingJob?.cancel()
        analyzingJob = null
        _isAnalyzing.value = false
        _isFetchingUrl.value = false
    }
    
    fun analyze() {
        val model = _selectedModel.value ?: return
        val inputText = _inputText.value
        val inputImageUri = _inputImageUri.value
        
        // Check if there's any input
        if (inputText.isBlank() && inputImageUri == null) {
            return
        }
        
        analyzingJob?.cancel()
        analyzingJob = viewModelScope.launch {
            try {
                _isAnalyzing.value = true
                _outputText.value = ""
                
                var contentToAnalyze = inputText
                
                // Check if input contains a URL
                val urlPattern = Regex("""https?://[^\s]+""")
                val urlMatch = urlPattern.find(inputText)
                
                if (urlMatch != null) {
                    val url = urlMatch.value
                    Log.d(TAG, "Detected URL in input: $url")
                    
                    _isFetchingUrl.value = true
                    val fetchedContent = fetchUrlContent(url)
                    _isFetchingUrl.value = false
                    
                    if (fetchedContent.isNotEmpty()) {
                        contentToAnalyze = """
                            URL: $url
                            
                            Content from URL:
                            ${fetchedContent.take(3000)}
                            
                            ${if (inputText != url) "Additional context: ${inputText.replace(url, "").trim()}" else ""}
                        """.trimIndent()
                    }
                }
                
                // Collect images if provided and vision is enabled (convert URI to Bitmap)
                val images = if (inputImageUri != null && _visionEnabled.value) {
                    loadBitmapFromUri(inputImageUri)?.let { listOf(it) } ?: emptyList()
                } else {
                    emptyList()
                }
                
                // Build the analysis prompt (indicate if image is present)
                val hasImage = images.isNotEmpty()
                val prompt = buildAnalysisPrompt(contentToAnalyze, hasImage)
                
                // Use unique chatId for each analysis session
                val chatId = "scam-detector-${UUID.randomUUID()}"
                
                // Generate analysis using inference service
                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = prompt,
                    model = model,
                    chatId = chatId,
                    images = images,
                    audioData = null,
                    webSearchEnabled = false
                )
                
                responseFlow.collect { token ->
                    _outputText.value += token
                }
                
            } catch (e: CancellationException) {
                Log.d(TAG, "Analysis cancelled")
                // Don't show error for user cancellation
            } catch (e: Exception) {
                Log.e(TAG, "Analysis failed", e)
                val errorMsg = e.message ?: "Unknown error"
                
                // Filter out technical error messages
                if (!errorMsg.contains("cancelled", ignoreCase = true) &&
                    !errorMsg.contains("Previous invocation", ignoreCase = true) &&
                    !errorMsg.contains("StandaloneCoroutine", ignoreCase = true)) {
                    _loadError.value = "Analysis failed: $errorMsg"
                }
            } finally {
                _isAnalyzing.value = false
                _isFetchingUrl.value = false
                analyzingJob = null
            }
        }
    }
    
    private suspend fun fetchUrlContent(url: String): String {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url(url)
                    .addHeader("User-Agent", "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0")
                    .build()
                
                val response = httpClient.newCall(request).execute()
                val html = response.body?.string() ?: return@withContext ""
                
                if (!response.isSuccessful) {
                    return@withContext ""
                }
                
                // Extract text from HTML
                extractTextFromHtml(html)
                
            } catch (e: Exception) {
                Log.w(TAG, "Failed to fetch URL content: ${e.message}")
                ""
            }
        }
    }
    
    private fun extractTextFromHtml(html: String): String {
        try {
            // Remove script and style tags
            var cleaned = html.replace(Regex("<script[^>]*>.*?</script>", RegexOption.DOT_MATCHES_ALL), "")
            cleaned = cleaned.replace(Regex("<style[^>]*>.*?</style>", RegexOption.DOT_MATCHES_ALL), "")
            
            // Remove HTML tags
            cleaned = cleaned.replace(Regex("<[^>]*>"), " ")
            
            // Clean up entities and whitespace
            cleaned = cleaned
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&nbsp;", " ")
                .replace(Regex("\\s+"), " ")
                .trim()
            
            // Extract meaningful sentences
            val sentences = cleaned.split(Regex("[.!?]+")).filter { sentence ->
                val s = sentence.trim()
                s.length > 20 && 
                s.length < 500 &&
                !s.contains("click", ignoreCase = true) &&
                !s.contains("menu", ignoreCase = true) &&
                !s.contains("navigation", ignoreCase = true) &&
                s.split(" ").size > 4
            }
            
            return sentences.take(10).joinToString(". ").take(3000)
            
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract text from HTML: ${e.message}")
            return ""
        }
    }
    
    private fun buildAnalysisPrompt(content: String, hasImage: Boolean): String {
        return if (hasImage && content.isNotBlank()) {
            // Both text and image present
            """
You are a scam detection expert. Analyze BOTH the provided image AND the text content below for potential scams, fraud, phishing attempts, or suspicious activity.

**Text content to analyze:**
$content

**Instructions:**
- Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
- Cross-reference the text content with what's shown in the image
- Look for inconsistencies between the image and text
- Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer

IMPORTANT: Respond in the same language as the input content.  Match the language of the content in the image.

Please provide a comprehensive analysis covering:
1. **Risk Level**: Low, Medium, High, or Critical
2. **Red Flags in Image**: List any suspicious visual elements (fake logos, poor quality graphics, misleading layouts, etc.)
3. **Red Flags in Text**: List any suspicious text elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
4. **Consistency Check**: Do the image and text align? Are there contradictions?
5. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
6. **Verdict**: Is this likely a scam? Explain your reasoning based on BOTH the image and text.
7. **Recommendations**: What should the user do?

Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """.trimIndent()
        } else if (hasImage) {
            // Only image present
            """
You are a scam detection expert. Analyze the provided image for potential scams, fraud, phishing attempts, or suspicious activity.

**Instructions:**
- Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
- Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer
- Look for common scam tactics in the visual content

Please provide a comprehensive analysis covering:
1. **Risk Level**: Low, Medium, High, or Critical
2. **Visual Red Flags**: List any suspicious elements in the image (fake logos, poor quality graphics, misleading layouts, urgency messages, too-good-to-be-true offers, etc.)
3. **Legitimacy Indicators**: Any visual signs suggesting it might be legitimate
4. **Verdict**: Is this likely a scam? Explain your reasoning based on the image.
5. **Recommendations**: What should the user do?

Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """.trimIndent()
        } else {
            // Only text present
            """
You are a scam detection expert. Analyze the following content for potential scams, fraud, phishing attempts, or suspicious activity.

Content to analyze:
$content

Please provide a comprehensive analysis covering:
1. **Risk Level**: Low, Medium, High, or Critical
2. **Red Flags**: List any suspicious elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
3. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
4. **Verdict**: Is this likely a scam? Explain your reasoning.
5. **Recommendations**: What should the user do?

Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """.trimIndent()
        }
    }
    
    private suspend fun loadBitmapFromUri(uri: Uri): Bitmap? {
        val app = getApplication<Application>()
        return withContext(Dispatchers.IO) {
            try {
                app.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
            } catch (_: Exception) {
                null
            }
        }
    }
}
