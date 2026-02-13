package com.llmhub.llmhub.viewmodels

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.CreatorEntity
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.repository.ChatRepository
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.TimeoutCancellationException

class CreatorViewModel(
    private val repository: ChatRepository,
    private val inferenceService: InferenceService,
    private val context: Context
) : ViewModel() {

    private val prefs = context.getSharedPreferences("creator_prefs", Context.MODE_PRIVATE)

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _generatedCreator = MutableStateFlow<CreatorEntity?>(null)
    val generatedCreator: StateFlow<CreatorEntity?> = _generatedCreator.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    // Model management states
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()

    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()

    private val _selectedBackend = MutableStateFlow<LlmInference.Backend?>(null)
    val selectedBackend: StateFlow<LlmInference.Backend?> = _selectedBackend.asStateFlow()

    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    init {
        loadAvailableModels()
        loadSavedSettings()
        checkIfModelIsAlreadyLoaded()
    }

    private fun checkIfModelIsAlreadyLoaded() {
        val currentModel = inferenceService.getCurrentlyLoadedModel()
        if (currentModel != null) {
            _selectedModel.value = currentModel
            _isModelLoaded.value = true
        }
    }

    private fun loadSavedSettings() {
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }

        val savedModelName = prefs.getString("selected_model_name", null)
        if (savedModelName != null && _selectedModel.value == null) { // Only load if not already set by loaded check
            viewModelScope.launch {
                // Wait for available models to populate
                kotlinx.coroutines.delay(100) 
                val model = _availableModels.value.find { it.name == savedModelName }
                if (model != null) {
                    _selectedModel.value = model
                    if (!model.supportsGpu && _selectedBackend.value == LlmInference.Backend.GPU) {
                        _selectedBackend.value = LlmInference.Backend.CPU
                    }
                }
            }
        }
    }

    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value?.name)
            apply()
        }
    }

    private fun loadAvailableModels() {
        viewModelScope.launch {
            val available = ModelAvailabilityProvider.loadAvailableModels(context)
                .filter { it.category != "embedding" && !it.name.contains("Projector", ignoreCase = true) }
            _availableModels.value = available
            
            // If no model selected yet and not loaded, pick first
            if (_selectedModel.value == null && !_isModelLoaded.value) {
                available.firstOrNull()?.let {
                    _selectedModel.value = it
                    _selectedBackend.value = if (it.supportsGpu) {
                        _selectedBackend.value ?: LlmInference.Backend.GPU
                    } else {
                        LlmInference.Backend.CPU
                    }
                }
            }
        }
    }

    fun selectModel(model: LLMModel) {
        if (_isModelLoaded.value && _selectedModel.value != model) {
            unloadModel()
        }
        
        _selectedModel.value = model
        _isModelLoaded.value = false // Require reload if model changed
        
        _selectedBackend.value = if (model.supportsGpu) {
            _selectedBackend.value ?: LlmInference.Backend.GPU
        } else {
            LlmInference.Backend.CPU
        }
        
        saveSettings()
    }

    fun selectBackend(backend: LlmInference.Backend) {
        if (_isModelLoaded.value && _selectedBackend.value != backend) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        _isModelLoaded.value = false
        saveSettings()
    }

    fun loadModel() {
        val model = _selectedModel.value ?: return
        val backend = _selectedBackend.value ?: return
        
        if (_isLoading.value || _isModelLoaded.value) {
            return
        }
        
        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null
            
            try {
                // Unload current if any
                inferenceService.unloadModel()
                
                // Load model 
                // Using standard settings, enabling generic vision/audio if supported by model, though not strictly used for generation here yet
                val success = inferenceService.loadModel(
                    model = model,
                    preferredBackend = backend,
                    disableVision = !model.supportsVision,
                    disableAudio = !model.supportsAudio
                )
                
                if (success) {
                    _isModelLoaded.value = true
                } else {
                    _error.value = "Failed to load model"
                }
            } catch (e: Exception) {
                _error.value = e.message ?: "Unknown error loading model"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun unloadModel() {
        viewModelScope.launch {
            try {
                inferenceService.unloadModel()
                _isModelLoaded.value = false
            } catch (e: Exception) {
                _error.value = e.message ?: "Failed to unload model"
            }
        }
    }

    fun generateCreator(userPrompt: String) {
        viewModelScope.launch {
            _isGenerating.value = true
            _error.value = null
            _generatedCreator.value = null

            try {
                // Ensure model is loaded
                if (!_isModelLoaded.value) {
                     _error.value = "Please load a model first."
                    _isGenerating.value = false
                    return@launch
                }
                
                // Double check service state just in case
                val model = inferenceService.getCurrentlyLoadedModel()
                if (model == null) {
                     // Try to reload implicitly if we think we are loaded but service isn't
                     // Or just fail. Let's fail to be safe and update state.
                     _isModelLoaded.value = false
                    _error.value = "Model not loaded in service. Please load again."
                    _isGenerating.value = false
                    return@launch
                }

                val metaPrompt = """
                    You are an expert AI persona creator. Your goal is to create a detailed system prompt for a new AI agent based on the user's description.
                    
                    User Description: "$userPrompt"
                    
                    Structure your response EXACTLY in this format (PCTF):
                    
                    NAME: [A creative name for the agent]
                    ICON: [A single emoji representing the agent]
                    DESCRIPTION: [A short 1-sentence description]
                    SYSTEM_PROMPT:
                    [System prompt the agent must follow. Include concise, firm instructions on Personality, Context, Task, and Format that would meet the User Description goals. Use markdown for clarity (bolding, lists, etc).]
                    
                    Do not add any other text or conversational filler. Just the format above.
                """.trimIndent()

                // Add 90-second timeout
                withTimeout(90_000L) {
                    val response = inferenceService.generateResponse(metaPrompt, model)
                    
                    val parsedCreator = parseResponse(response, userPrompt)
                    if (parsedCreator != null) {
                        _generatedCreator.value = parsedCreator
                    } else {
                        _error.value = "Failed to parse generation result. Try again."
                    }
                }

            } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
                Log.e("CreatorViewModel", "Generation timed out", e)
                _error.value = "Generation timed out (90s limit). Please try a simpler prompt or faster model."
            } catch (e: Exception) {
                Log.e("CreatorViewModel", "Generation failed", e)
                _error.value = "Error: ${e.message}"
            } finally {
                _isGenerating.value = false
            }
        }
    }

    private fun parseResponse(response: String, originalPrompt: String): CreatorEntity? {
        try {
            val nameRegex = Regex("NAME:\\s*(.+)")
            val iconRegex = Regex("ICON:\\s*(.+)")
            val descriptionRegex = Regex("DESCRIPTION:\\s*(.+)")
            val promptRegex = Regex("SYSTEM_PROMPT:\\s*([\\s\\S]*)")

            val name = nameRegex.find(response)?.groupValues?.get(1)?.trim() ?: "My Creator"
            val icon = iconRegex.find(response)?.groupValues?.get(1)?.trim()?.take(2) ?: "🤖" // Fallback to robot if parse fails or text is too long
            val description = descriptionRegex.find(response)?.groupValues?.get(1)?.trim() ?: originalPrompt
            val systemPrompt = promptRegex.find(response)?.groupValues?.get(1)?.trim() ?: response

            return CreatorEntity(
                name = name,
                icon = icon,
                description = description,
                pctfPrompt = "CRITICAL INSTRUCTIONS:\n\n" + systemPrompt
            )
        } catch (e: Exception) {
            Log.e("CreatorViewModel", "Parsing failed", e)
            return null
        }
    }

    fun saveCreator(creator: CreatorEntity, onSaved: () -> Unit) {
        viewModelScope.launch {
            repository.insertCreator(creator)
            onSaved()
        }
    }
    
    fun deleteCreator(creator: CreatorEntity) {
        viewModelScope.launch {
            try {
                repository.deleteCreator(creator)
            } catch (e: Exception) {
                _error.value = "Failed to delete creator: ${e.message}"
            }
        }
    }
    
    fun clearError() {
        _error.value = null
    }
}
