package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class CodeLanguage {
    HTML, PYTHON, JAVASCRIPT, UNKNOWN
}

/**
 * VibeCoderViewModel handles code generation using LLM inference.
 * Users provide a prompt, and the model generates HTML/Python/JavaScript code.
 */
class VibeCoderViewModel(application: Application) : AndroidViewModel(application) {
    
    private val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
    private val prefs = application.getSharedPreferences("vibe_coder_prefs", Context.MODE_PRIVATE)
    
    private var processingJob: Job? = null
    
    // Available models
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()
    
    // Model selection & backend
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    private val _selectedBackend = MutableStateFlow<LlmInference.Backend?>(null)
    val selectedBackend: StateFlow<LlmInference.Backend?> = _selectedBackend.asStateFlow()
    
    // Loading states
    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()
    
    private val _isPlanning = MutableStateFlow(false)
    val isPlanning: StateFlow<Boolean> = _isPlanning.asStateFlow()
    
    private var currentSpec: String = ""
    
    // Generated code & metadata
    private val _generatedCode = MutableStateFlow("")
    val generatedCode: StateFlow<String> = _generatedCode.asStateFlow()
    
    private val _codeLanguage = MutableStateFlow(CodeLanguage.UNKNOWN)
    val codeLanguage: StateFlow<CodeLanguage> = _codeLanguage.asStateFlow()
    
    private val _promptInput = MutableStateFlow("")
    val promptInput: StateFlow<String> = _promptInput.asStateFlow()
    
    // Error handling
    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()
    
    init {
        loadAvailableModels()
        loadSavedSettings()
    }
    
    /**
     * Load previously saved settings (model, backend)
     */
    private fun loadSavedSettings() {
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
        
        val savedModelName = prefs.getString("selected_model_name", null)
        if (savedModelName != null) {
            viewModelScope.launch {
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
    
    /**
     * Save current model and backend preferences
     */
    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value?.name)
            apply()
        }
    }
    
    /**
     * Load all available models from device
     */
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val available = ModelAvailabilityProvider.loadAvailableModels(context)
                .filter { it.category != "embedding" && !it.name.contains("Projector", ignoreCase = true) }
            _availableModels.value = available
            if (_selectedModel.value == null) {
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
    
    /**
     * Select a different model for code generation
     */
    fun selectModel(model: LLMModel) {
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedModel.value = model
        _isModelLoaded.value = false
        
        _selectedBackend.value = if (model.supportsGpu) {
            _selectedBackend.value ?: LlmInference.Backend.GPU
        } else {
            LlmInference.Backend.CPU
        }
        
        saveSettings()
    }
    
    /**
     * Select inference backend (GPU, CPU, etc.)
     */
    fun selectBackend(backend: LlmInference.Backend) {
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        _isModelLoaded.value = false
        saveSettings()
    }
    
    /**
     * Load the selected model into memory
     */
    fun loadModel() {
        val model = _selectedModel.value ?: return
        val backend = _selectedBackend.value ?: return
        
        if (_isLoading.value || _isModelLoaded.value) {
            return
        }
        
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            
            try {
                inferenceService.unloadModel()
                
                // Load model with text-only mode (vibe coder generates code as text)
                val success = inferenceService.loadModel(
                    model = model,
                    preferredBackend = backend,
                    disableVision = true,
                    disableAudio = true
                )
                
                if (success) {
                    _isModelLoaded.value = true
                } else {
                    _errorMessage.value = "Failed to load model"
                }
            } catch (e: Exception) {
                _errorMessage.value = e.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    /**
     * Unload the current model from memory
     */
    fun unloadModel() {
        viewModelScope.launch {
            try {
                inferenceService.unloadModel()
                _isModelLoaded.value = false
                _generatedCode.value = ""
            } catch (e: Exception) {
                _errorMessage.value = e.message ?: "Failed to unload model"
            }
        }
    }
    
    /**
     * Update the prompt input text
     */
    fun updatePromptInput(text: String) {
        _promptInput.value = text
    }
    
    /**
     * Generate code based on the user's prompt
     */
    fun generateCode(prompt: String) {
        if (prompt.isBlank()) return
        val model = _selectedModel.value ?: return
        
        if (!_isModelLoaded.value) {
            _errorMessage.value = "Please load a model first"
            return
        }
        
        processingJob?.cancel()
        
        processingJob = viewModelScope.launch {
            _isProcessing.value = true
            _generatedCode.value = ""
            _errorMessage.value = null
            
            // Determine if request is creative/game or utility/precise
            val isCreative = prompt.contains("game", ignoreCase = true) || 
                           prompt.contains("story", ignoreCase = true) ||
                           prompt.contains("art", ignoreCase = true) ||
                           prompt.contains("creative", ignoreCase = true)
            
            // Set optimized parameters based on intent:
            // - Utility/Math/Code (default): 0.2 temperature for high precision
            // - Games/Creative: 0.6 temperature for balanced creativity
            val temperature = if (isCreative) 0.6f else 0.2f
            
            inferenceService.setGenerationParameters(
                maxTokens = 8192,
                topK = 40,
                topP = 0.95f,
                temperature = temperature
            )
            
            try {
                // Step 1: Architect (Meta-Prompting) vs Direct Modification
                // If we have existing code and the prompt implies a revision, we SKIP the architect
                // and go straight to the coder with a "Modification Prompt".
                // If it's a new project, we use the Architect to plan it first.
                
                val currentCode = _generatedCode.value
                val isRevision = currentCode.isNotBlank() && !prompt.equals("new", ignoreCase = true)
                
                var builtSpec = ""
                
                // Only run Architect if this is a NEW project
                if (!isRevision) {
                    _isPlanning.value = true
                    try {
                        // Timeout for planning phase (90 seconds)
                        kotlinx.coroutines.withTimeout(90_000L) {
                            val specPrompt = buildSpecPrompt(prompt, "")
                            val specChatId = "vibe-spec-${UUID.randomUUID()}"
                            
                            val specResponseFlow = inferenceService.generateResponseStreamWithSession(
                                prompt = specPrompt,
                                model = model,
                                chatId = specChatId,
                                images = emptyList(),
                                audioData = null,
                                webSearchEnabled = false
                            )
                            
                            specResponseFlow.collect { token ->
                                builtSpec += token
                            }
                        }
                        
                        // DEBUG: Log the Architect's generated requirements
                        Log.d("VibeCoderVM", "Architect Requirements:\n$builtSpec")
                        
                        // CRITICAL: Explicitly reset the session between Architect and Coder phases
                        try {
                            inferenceService.resetChatSession("vibe-spec-handoff")
                            kotlinx.coroutines.delay(200)
                        } catch (e: Exception) {
                            Log.w("VibeCoderVM", "Session reset between phases failed: ${e.message}")
                        }
                        
                    } catch (e: Exception) {
                         Log.w("VibeCoderVM", "Planning phase failed or timed out: ${e.message}. Falling back to direct generation.")
                         try {
                            inferenceService.resetChatSession("vibe-spec-cleanup")
                         } catch (resetEx: Exception) {
                            Log.e("VibeCoderVM", "Failed to reset session after planning failure", resetEx)
                         }
                    }
                    _isPlanning.value = false
                }
                
                currentSpec = builtSpec
                
                // Step 2: Coder (Implementation or Modification)
                val implementationPrompt = if (isRevision) {
                    // Direct Modification Flow
                    Log.d("VibeCoderVM", "Direct Modification Mode")
                    buildModificationPrompt(prompt, currentCode)
                } else if (builtSpec.isNotBlank()) {
                    // Standard Flow (Architect -> Coder)
                    buildImplementationPrompt(builtSpec)
                } else {
                    // Fallback Flow (Direct Prompt)
                    buildPrompt(prompt)
                }
                
                val codeChatId = "vibe-coder-${UUID.randomUUID()}"
                
                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = implementationPrompt,
                    model = model,
                    chatId = codeChatId,
                    images = emptyList(),
                    audioData = null,
                    webSearchEnabled = false
                )
                
                var responseText = ""
                try {
                    responseFlow.collect { token ->
                        responseText += token
                        _generatedCode.value = responseText

                        // Early stopping: Check if we have two distinct sets of triple backticks
                        val firstTick = responseText.indexOf("```")
                        if (firstTick != -1) {
                            val lastTick = responseText.lastIndexOf("```")
                            if (lastTick > firstTick) {
                                // Found a closing code block, stop generating
                                throw kotlinx.coroutines.CancellationException("Code block complete")
                            }
                        }
                    }
                } catch (e: kotlinx.coroutines.CancellationException) {
                    if (e.message == "Code block complete") {
                        Log.d("VibeCoderVM", "Early stop: Code block complete")
                        // Fall through to extraction
                    } else {
                        throw e
                    }
                }
                
                // Detect code language and extract code from response
                detectAndExtractCode(responseText)
                
            } catch (e: kotlinx.coroutines.CancellationException) {
                Log.d("VibeCoderVM", "Generation cancelled")
            } catch (e: Exception) {
                val message = e.message ?: ""
                val shouldShowError = !message.contains("cancelled", ignoreCase = true) &&
                                    !message.contains("Previous invocation still processing", ignoreCase = true) &&
                                    !message.contains("StandaloneCoroutine", ignoreCase = true)
                
                if (shouldShowError) {
                    _errorMessage.value = message.ifBlank { "Generation failed" }
                    Log.e("VibeCoderVM", "Generation error: $message", e)
                } else {
                    Log.d("VibeCoderVM", "Suppressed error: $message")
                }
            } finally {
                // Reset parameters to defaults (null)
                inferenceService.setGenerationParameters(null, null, null, null)
                _isProcessing.value = false
                _isPlanning.value = false
                processingJob = null
            }
        }
    }
    
    /**
     * Detect code language and extract clean code from the response.
     * Supports HTML, Python, JavaScript wrapped in markdown code blocks or XML tags.
     * Handles edge cases where code block markers aren't perfectly formatted.
     */
    private fun detectAndExtractCode(response: String) {
        // Try to extract from markdown code blocks with language hints (```html, ```python, etc.)
        // Relaxed regex to allow immediate content after language tag (no newline required)
        val htmlMatch = Regex("```(?:html|htm)\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE).find(response)
        if (htmlMatch != null) {
            _generatedCode.value = htmlMatch.groupValues[1].trim()
            _codeLanguage.value = CodeLanguage.HTML
            return
        }
        
        val pythonMatch = Regex("```(?:python|py)\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE).find(response)
        if (pythonMatch != null) {
            _generatedCode.value = pythonMatch.groupValues[1].trim()
            _codeLanguage.value = CodeLanguage.PYTHON
            return
        }
        
        val jsMatch = Regex("```(?:javascript|js)\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE).find(response)
        if (jsMatch != null) {
            _generatedCode.value = jsMatch.groupValues[1].trim()
            _codeLanguage.value = CodeLanguage.JAVASCRIPT
            return
        }
        
        // Fallback: Extract any content between ``` markers (handles malformed responses)
        val genericMatch = Regex("```\\s*([\\s\\S]*?)```").find(response)
        if (genericMatch != null) {
            val extracted = genericMatch.groupValues[1].trim()
            _generatedCode.value = extracted
            // Detect language from content
            when {
                extracted.contains("<!DOCTYPE", ignoreCase = true) || extracted.contains("<html", ignoreCase = true) -> {
                    _codeLanguage.value = CodeLanguage.HTML
                }
                extracted.contains("def ") || extracted.contains("import ") -> {
                    _codeLanguage.value = CodeLanguage.PYTHON
                }
                extracted.contains("function ") || extracted.contains("const ") -> {
                    _codeLanguage.value = CodeLanguage.JAVASCRIPT
                }
                else -> {
                    _codeLanguage.value = CodeLanguage.UNKNOWN
                }
            }
            return
        }
        
        // Try to extract from XML-like tags (fallback)
        val xmlHtmlMatch = Regex("<code[^>]*>([\\s\\S]*?)</code>", RegexOption.IGNORE_CASE).find(response)
        if (xmlHtmlMatch != null) {
            val extracted = xmlHtmlMatch.groupValues[1].trim()
            _generatedCode.value = extracted
            _codeLanguage.value = CodeLanguage.HTML
            return
        }
        
        // Default detection based on content pattern
        when {
            response.contains("<!DOCTYPE html", ignoreCase = true) ||
            response.contains("<html", ignoreCase = true) -> {
                _codeLanguage.value = CodeLanguage.HTML
            }
            response.contains("def ", ignoreCase = true) ||
            response.contains("import ", ignoreCase = true) ||
            response.contains("python", ignoreCase = true) -> {
                _codeLanguage.value = CodeLanguage.PYTHON
            }
            response.contains("function ", ignoreCase = true) ||
            response.contains("const ", ignoreCase = true) ||
            response.contains("var ", ignoreCase = true) ||
            response.contains("javascript", ignoreCase = true) -> {
                _codeLanguage.value = CodeLanguage.JAVASCRIPT
            }
            else -> {
                _codeLanguage.value = CodeLanguage.UNKNOWN
            }
        }
    }
    
    /**
     * Cancel ongoing code generation
     */
    fun cancelGeneration() {
        processingJob?.cancel()
        processingJob = null
        _isProcessing.value = false
    }
    
    /**
     * Clear generated code
     */
    fun clearCode() {
        _generatedCode.value = ""
        _codeLanguage.value = CodeLanguage.UNKNOWN
        currentSpec = ""
    }
    
    /**
     * Clear error message
     */
    fun clearError() {
        _errorMessage.value = null
    }
    
    /**
     * Build the Architect Spec Prompt (Step 1)
     * Simplified to a "Technical Assistant" role that generates a concise Requirements List.
     */
    private fun buildSpecPrompt(userRequest: String, currentCode: String): String {
        val isRevision = currentCode.isNotBlank()
        return """
            You are a helpful Technical Assistant.
            Your goal is to expand the user's request into a clear, concise list of functional requirements.
            
            CONTEXT:
            ${if (isRevision) "The user wants to MODIFY this existing code:\n$currentCode" else "This is a NEW project request."}
            
            USER REQUEST: "$userRequest"
            
            TASK:
            1. Identify the core features needed.
            2. List specific UI elements required (buttons, inputs, displays).
            3. Define the basic logic flow (e.g., "User clicks -> Update Score").
            4. Keep it brief and actionable.
            
            OUTPUT FORMAT:
            - Feature: [Description]
            - UI: [Element]
            - Logic: [Rule]
            
            Output ONLY the list. Do not write code or introductions.
        """.trimIndent()
    }

    /**
     * Build the Developer Implementation Prompt (Step 2)
     */
    /**
     * Build the Developer Implementation Prompt (Step 2 - New Project)
     */
    private fun buildImplementationPrompt(requirements: String): String {
        return """
            You are an expert developer who is adept at generating production-ready stand-alone apps and games in either HTML or Python. 
            Your task is to generate clean, functional code based on the Requirements provided below. The code will run in an offline interpreter.
            
            REQUIREMENTS:
            $requirements
            
            Think about how to meet these requirements for the best stand-alone functional code to delight the user.

            CONSTRAINTS:
            Generate code that is:
            - Syntactically correct and ready to run
            - Well-commented where appropriate
            - Self-contained (no external dependencies)
            
            CRITICAL ANTI-PATTERNS (DO NOT DO THIS):
            - NO BLOCKING LOOPS: Never use 'while' or 'for' loops to manage turns or wait for user input (e.g., `while(guesses < 7)`). This freezes the browser.
            - NO ALERTS: Do not use `alert()` or `prompt()`. Use HTML elements for output and input.
            - NO EXTERNAL RESOURCES: No external images, CSS, or JS files.
            - TYPE SAFETY: Never compare `input.value` directly to a number. ALWAYS use `parseInt()` or `Number()` first.
            - UI INTEGRITY: Do not overwrite elements that contain labels (e.g., `<div id="score">Score: 0</div>` -> `document.getElementById("score").textContent = 5`). This destroys the label. Use a child `<span>` for the value or include the label in the update.
            
            REQUIREMENTS FOR APPS/GAMES (HTML/JS):
            - Create a complete, standalone Single Page Application (SPA).
            - Game Loop: State must persist between events. Each button click = one update.
            - ALWAYS include a "Reset" or "New" button to restart the application state.
            - Games should maintain a functional game state (Score, Win/Loss messages, turn history, etc.) in the UI. Turn history would be a list of previous moves/actions so the user can track progress, and summarize the results when the game is won or lost.
            - Ensure all interactive elements (buttons, inputs) are clearly visible and accessible.
            - FUNCTIONAL UI: Ensure ALL UI elements (including SVGs, Canvas) are functional and wired to the script. Do NOT add decorative elements that do nothing.
            - EVENT DRIVEN: Do NOT use blocking loops (while/for) to wait for user input. Use event listeners and state variables to handle user interactions asynchronously.
            
            REQUIREMENTS FOR UTILITY APPS (Calculators, Converters, Tools):
            - Use clear, labeled forms with appropriate input types (number, text, etc.).
            - Validate inputs before processing (show user-friendly error messages).
            - clearly display results in a distinct output area.
            - Ensure high precision for calculations.
            
            REQUIREMENTS FOR PYTHON:
            - Create a functional script (no external dependencies).
            - Since this runs in a text simulation check, use print() statements to simulate output/state.
            - For object simulations (e.g., "Park Sim"), create classes and a main execution block that demonstrates the logic.
            
            IMPORTANT:
            - If generating HTML/JavaScript, wrap it in a markdown code block: ```html
            YOUR HTML CODE HERE
            ```
            - If generating Python, wrap it in a markdown code block: ```python
            YOUR PYTHON CODE HERE
            ```
            - Respond ONLY with the production-ready stand-alone code in a markdown code block. DO NOT include explanations, warnings, or additional text before or after the code block.
    
        """.trimIndent()
    }

    /**
     * Build the Developer Modification Prompt (Step 2 - Revision)
     * Direct Code Modification skipping the Architect.
     */
    private fun buildModificationPrompt(userRequest: String, currentCode: String): String {
        return """
            You are an expert developer. The user wants to MODIFY the existing code below.
            
            EXISTING CODE:
            ```
            $currentCode
            ```
            
            USER REQUEST: "$userRequest"
            
            TASK:
            1. Analyze the existing code and the user's request.
            2. Rewrite the FULL code to incorporate the changes.
            3. Ensure the rest of the application remains functional.
            
            CRITICAL ANTI-PATTERNS (DO NOT DO THIS):
            - NO BLOCKING LOOPS: Never use 'while' or 'for' loops to manage turns or wait for user input (e.g., `while(guesses < 7)`). This freezes the browser.
            - NO ALERTS: Do not use `alert()` or `prompt()`. Use HTML elements for output and input.
            - NO EXTERNAL RESOURCES: No external images, CSS, or JS files.
            - TYPE SAFETY: Never compare `input.value` directly to a number. ALWAYS use `parseInt()` or `Number()` first.
            - UI INTEGRITY: Do not overwrite elements that contain labels. Use a child `<span>` for the value.
            
            IMPORTANT:
            - Wrap code in ```html or ```python blocks.
            - Return the FULL modified code, not just a diff.
            - No explanations.
        """.trimIndent()
    }

    /**
     * Legacy Prompt Builder (Fallback for v0.4 behavior)
     * Used when Planning Phase fails or times out.
     */
    private fun buildPrompt(userPrompt: String): String {
        return """
            You are an expert developer who is adept at generating production-ready stand-alone apps and games in either HTML or Python. 
            Your task is to generate clean, functional code based on the current user request. The code will run in an offline interpreter.

            User request: $userPrompt

            Think about how to meet the user's request for the best stand-alone functional code to delight the user, considering the constraints and requirements that follow.

            CONSTRAINTS:
            Generate code that is:
            - Syntactically correct and ready to run
            - Well-commented where appropriate
            - Self-contained (no external dependencies)
            
            CONSTRAINT: NO EXTERNAL RESOURCES
            - Do NOT use external images (<img> src must be data URI or SVG directly in code).
            - Do NOT use external scripts (CDNs) or CSS files.
            - Use standard HTML5/CSS3/ES6+ features.
            - For graphics, use inline SVG, Canvas API, or CSS shapes.
            - Provide a professional, polished look.
            
            REQUIREMENTS FOR APPS/GAMES (HTML/JS):
            - Create a complete, standalone Single Page Application (SPA).
            - ALWAYS include a "Reset" or "New" button to restart the application state.
            - Games should maintain a functional game state (Score, Win/Loss messages, turn history, etc.) in the UI. Turn history would be a list of previous moves/actions so the user can track progress, and summarize the results when the game is won or lost.
            - Ensure all interactive elements (buttons, inputs) are clearly visible and accessible.
            - FUNCTIONAL UI: Ensure ALL UI elements (including SVGs, Canvas) are functional and wired to the script. Do NOT add decorative elements that do nothing.
            - EVENT DRIVEN: Do NOT use blocking loops (while/for) to wait for user input. Use event listeners and state variables to handle user interactions asynchronously.
            
            REQUIREMENTS FOR UTILITY APPS (Calculators, Converters, Tools):
            - Use clear, labeled forms with appropriate input types (number, text, etc.).
            - Validate inputs before processing (show user-friendly error messages).
            - clearly display results in a distinct output area.
            - Ensure high precision for calculations.
            
            REQUIREMENTS FOR PYTHON:
            - Create a functional script (no external dependencies).
            - Since this runs in a text simulation check, use print() statements to simulate output/state.
            - For object simulations (e.g., "Park Sim"), create classes and a main execution block that demonstrates the logic.
            
            IMPORTANT:
            - If generating HTML/JavaScript, wrap it in a markdown code block: ```html
            YOUR HTML CODE HERE
            ```
            - If generating Python, wrap it in a markdown code block: ```python
            YOUR PYTHON CODE HERE
            ```
            - Respond ONLY with the production-ready stand-alone code in a markdown code block. DO NOT include explanations, warnings, or additional text before or after the code block.
    
        """.trimIndent()
    }
    
    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            inferenceService.onCleared()
        }
    }
}
