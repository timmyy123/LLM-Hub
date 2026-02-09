package com.llmhub.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.DeviceInfo
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

// Correct Nexa SDK Imports 
import com.nexa.sdk.NexaSdk
import com.nexa.sdk.LlmWrapper
import com.nexa.sdk.VlmWrapper
import com.nexa.sdk.bean.LlmCreateInput
import com.nexa.sdk.bean.VlmCreateInput
import com.nexa.sdk.bean.VlmChatMessage
import com.nexa.sdk.bean.VlmContent
import com.nexa.sdk.bean.ModelConfig
import com.nexa.sdk.bean.GenerationConfig
import com.nexa.sdk.bean.LlmStreamResult
import com.nexa.sdk.bean.ChatMessage
import com.nexa.sdk.bean.LlmApplyChatTemplateOutput

@Singleton
class NexaInferenceService @Inject constructor(
    private val context: Context
) : InferenceService {

    private val TAG = "NexaInferenceService"
    private var llmWrapper: LlmWrapper? = null
    private var vlmWrapper: VlmWrapper? = null
    private var isVlmLoaded: Boolean = false
    
    private var currentModel: LLMModel? = null
    private var currentPreferredBackend: LlmInference.Backend? = null
    private var currentVisionDisabled: Boolean = false
    
    private var overrideMaxTokens: Int? = null
    private var overrideTopK: Int? = null
    private var overrideTopP: Float? = null
    private var overrideTemperature: Float? = null
    
    init {
        try {
            NexaSdk.getInstance().init(context)
            // Clean up any stale VLM cache files from previous sessions
            cleanupStaleCacheFiles()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init NexaSdk", e)
        }
    }

    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?): Boolean {
        // Default to vision enabled for the two-arg load; clear any previous override
        currentVisionDisabled = false
        return loadModelInternal(model, preferredBackend, false)
    }

    override suspend fun loadModel(
        model: LLMModel,
        preferredBackend: LlmInference.Backend?,
        disableVision: Boolean,
        disableAudio: Boolean
    ): Boolean {
         // Respect the caller's disableVision flag so we can load as text-only if requested
         currentVisionDisabled = disableVision
         return loadModelInternal(model, preferredBackend, disableVision)
    }
    
    private suspend fun loadModelInternal(model: LLMModel, preferredBackend: LlmInference.Backend?, disableVision: Boolean = false): Boolean {
        if (currentModel?.name == model.name && (llmWrapper != null || vlmWrapper != null)) {
            return true
        }

        unloadModel()
        
        val modelDir = getModelDirectory(model)
        val modelFile = findGGUFModelFile(modelDir, model)
        
        if (!modelFile.exists()) {
            Log.e(TAG, "Model file not found: ${modelFile.absolutePath}")
            return false
        }

        val backendsToTry = mutableListOf<String>()
        if (preferredBackend == LlmInference.Backend.GPU || preferredBackend == null) {
            backendsToTry.add("GPUOpenCL")
        }
        backendsToTry.add("CPU")

        for (backendId in backendsToTry) {
            try {
                Log.d(TAG, "Attempting load with $backendId...")
                
                // Cap context size to prevent OOM on mobile devices.
                // GGUF models allocate KV cache proportional to nCtx;
                // 130K+ tokens will exhaust RAM on any phone.
                // VLM models' memory cost grows with nCtx. Cap to 8192 for vision-enabled GGUF to keep allocations reasonable
                // When vision is disabled, honor the user/model selected context window instead of forcing a cap.
                val MAX_SAFE_CTX = if (model.supportsVision && !disableVision) 8192 else Int.MAX_VALUE
                val rawCtx = overrideMaxTokens ?: model.contextWindowSize
                val nCtx = if (disableVision) rawCtx else rawCtx.coerceAtMost(MAX_SAFE_CTX)
                if (!disableVision && rawCtx != nCtx) {
                    Log.w(TAG, "Capped nCtx from $rawCtx to $nCtx to prevent OOM")
                }
                val deviceToUse = if (backendId == "CPU") null else "GPUOpenCL"
                val gpuLayers = if (backendId == "CPU") 0 else 999

                val modelConfig = ModelConfig(nCtx = nCtx, nGpuLayers = gpuLayers).apply {
                    val cls = this::class.java
                    val fieldsToSet = mutableMapOf<String, Any>(
                        "nCtx" to nCtx,
                        "n_ctx" to nCtx,
                        "maxTokens" to nCtx,
                        "max_tokens" to nCtx
                    )

                    // Enable Thinking mode if the model name suggests it
                    if (model.name.contains("Thinking", ignoreCase = true) || 
                        model.name.contains("Reasoning", ignoreCase = true)) {
                        fieldsToSet["enable_thinking"] = true
                        fieldsToSet["enableThinking"] = true
                    }
                    
                    for ((fieldName, value) in fieldsToSet) {
                        try {
                            val field = cls.getDeclaredField(fieldName)
                            field.isAccessible = true
                            field.set(this, value)
                            Log.d(TAG, "✓ Set ModelConfig.$fieldName = $value")
                        } catch (e: Exception) {}
                    }
                }

                // Find mmproj path for VLM models (only when vision is enabled)
                val mmprojPath = if (model.supportsVision && !disableVision) {
                    findMmprojFile(modelDir, modelFile)?.absolutePath
                } else null

                // Use VlmWrapper for vision-capable models, LlmWrapper for text-only
                if (model.supportsVision && !disableVision && mmprojPath != null) {
                    Log.i(TAG, "Loading as VLM with mmproj: $mmprojPath")
                    val vlmCreateInput = VlmCreateInput(
                        model_name = "",
                        model_path = modelFile.absolutePath,
                        mmproj_path = mmprojPath,
                        config = modelConfig,
                        plugin_id = "cpu_gpu",
                        device_id = deviceToUse
                    )

                    val buildResult = withContext(Dispatchers.IO) {
                        VlmWrapper.builder()
                            .vlmCreateInput(vlmCreateInput)
                            .build()
                    }

                    if (buildResult.isSuccess) {
                        vlmWrapper = buildResult.getOrNull()
                        isVlmLoaded = true
                        currentModel = model
                        currentPreferredBackend = if (backendId == "CPU") LlmInference.Backend.CPU else LlmInference.Backend.GPU
                        currentVisionDisabled = disableVision
                        Log.i(TAG, "✓ Successfully loaded VLM with $backendId backend")
                        return true
                    } else {
                        val err = buildResult.exceptionOrNull()
                        Log.w(TAG, "VLM Failed $backendId: ${err?.message}")
                    }
                } else {
                    val createInput = LlmCreateInput(
                        model_name = "",
                        model_path = modelFile.absolutePath,
                        tokenizer_path = null,
                        config = modelConfig,
                        plugin_id = "cpu_gpu",
                        device_id = deviceToUse
                    )

                    // Build on IO thread to avoid blocking the main thread
                    // (KV cache allocation can take seconds)
                    val buildResult = withContext(Dispatchers.IO) {
                        LlmWrapper.builder()
                            .llmCreateInput(createInput)
                            .build()
                    }

                    if (buildResult.isSuccess) {
                        llmWrapper = buildResult.getOrNull()
                        isVlmLoaded = false
                        currentModel = model
                        currentPreferredBackend = if (backendId == "CPU") LlmInference.Backend.CPU else LlmInference.Backend.GPU
                        currentVisionDisabled = disableVision
                        Log.i(TAG, "✓ Successfully loaded LLM with $backendId backend")
                        return true
                    } else {
                        val err = buildResult.exceptionOrNull()
                        Log.w(TAG, "LLM Failed $backendId: ${err?.message}")
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Exception during $backendId load attempt", e)
            }
        }
        
        return false
    }

    private fun getModelDirectory(model: LLMModel): File {
        val modelsDir = File(context.filesDir, "models")
        val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        val modelDir = File(modelsDir, modelDirName)
        return if (modelDir.exists()) modelDir else modelsDir
    }
    
    private fun findGGUFModelFile(modelDir: File, model: LLMModel): File {
        val localName = model.url.substringAfterLast("/").substringBefore("?")
        var modelFile = File(modelDir, localName)
        if (modelFile.exists()) return modelFile
        
        val modelsDir = File(context.filesDir, "models")
        modelFile = File(modelsDir, localName)
        if (modelFile.exists()) return modelFile
        
        // Find GGUF files but exclude mmproj files
        val files = modelDir.listFiles { _, name -> 
            name.endsWith(".gguf") && !name.contains("mmproj", ignoreCase = true)
        }
        if (files?.isNotEmpty() == true) return files.first()
        
        return File(modelDir, localName)
    }

    /**
     * Find the mmproj file for vision models, preferring variant-matched files.
     */
    private fun findMmprojFile(modelDir: File, modelFile: File): File? {
        val allMmproj = modelDir.listFiles { _, name ->
            name.contains("mmproj", ignoreCase = true) && name.endsWith(".gguf")
        } ?: return null

        if (allMmproj.isEmpty()) return null

        // Try to match the model variant (e.g. Instruct vs Reasoning)
        val modelBaseName = modelFile.nameWithoutExtension
            .replace(Regex("[-_]Q[0-9].*"), "") // Strip quant suffix
        
        val matched = allMmproj.firstOrNull { candidate ->
            val candidateBase = candidate.nameWithoutExtension
                .replace(Regex("[-_]BF16.*|[-_]mmproj.*", RegexOption.IGNORE_CASE), "")
            candidateBase.equals(modelBaseName, ignoreCase = true)
        }

        return matched ?: allMmproj.firstOrNull()
    }
    
    /**
     * Clean up stale VLM cache files from previous sessions to prevent accumulation
     */
    private fun cleanupStaleCacheFiles() {
        try {
            val cacheDir = context.cacheDir
            val vlmFiles = cacheDir.listFiles { file ->
                file.name.startsWith("nexa_vlm_") && file.name.endsWith(".jpg")
            } ?: return
            
            if (vlmFiles.isNotEmpty()) {
                Log.d(TAG, "Cleaning up ${vlmFiles.size} stale VLM cache files")
                vlmFiles.forEach { it.delete() }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to clean up VLM cache files: ${e.message}")
        }
    }

    override suspend fun unloadModel() {
         try {
             if (isVlmLoaded) {
                 vlmWrapper?.stopStream()
                 vlmWrapper?.destroy()
             } else {
                 llmWrapper?.destroy()
             }
         } catch (e: Exception) {
             Log.w(TAG, "Error closing Nexa model: ${e.message}")
         } finally {
             llmWrapper = null
             vlmWrapper = null
             isVlmLoaded = false
             currentModel = null
             currentPreferredBackend = null
         }
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
         val sb = StringBuilder()
         generateResponseStream(prompt, model).collect { sb.append(it) }
         return sb.toString()
    }

    override suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String> {
        return generateResponseStreamInternal(prompt, model, emptyList())
    }

    override suspend fun generateResponseStreamWithSession(
        prompt: String,
        model: LLMModel,
        chatId: String,
        images: List<Bitmap>,
        audioData: ByteArray?,
        webSearchEnabled: Boolean,
        imagePaths: List<String>
    ): Flow<String> {
        val imagePaths = if (images.isNotEmpty()) {
            images.mapIndexed { index, bitmap ->
                // Downscale large images to speed up the VLM vision encoder.
                val maxDim = 300
                val scaled = if (bitmap.width > maxDim || bitmap.height > maxDim) {
                    val scale = maxDim.toFloat() / maxOf(bitmap.width, bitmap.height)
                    val w = (bitmap.width * scale).toInt()
                    val h = (bitmap.height * scale).toInt()
                    Log.d(TAG, "Downscaling image from ${bitmap.width}x${bitmap.height} to ${w}x${h}")
                    Bitmap.createScaledBitmap(bitmap, w, h, true)
                } else bitmap
                
                val file = File(context.cacheDir, "nexa_vlm_${System.currentTimeMillis()}_$index.jpg")
                file.outputStream().use { 
                    // Lower JPEG quality to reduce I/O overhead but keep reasonable fidelity
                    scaled.compress(Bitmap.CompressFormat.JPEG, 70, it)
                }
                // Recycle the scaled copy if we created one
                if (scaled !== bitmap) scaled.recycle()
                file.absolutePath
            }
        } else {
            emptyList()
        }
        
        return generateResponseStreamInternal(prompt, model, imagePaths)
    }

    private suspend fun generateResponseStreamInternal(
        prompt: String, 
        model: LLMModel, 
        imagePaths: List<String> = emptyList()
    ): Flow<String> = callbackFlow {
        if (llmWrapper == null && vlmWrapper == null) {
            close(IllegalStateException("Model not loaded"))
            return@callbackFlow
        }
        
        val baseMaxTokens = overrideMaxTokens ?: model.contextWindowSize
        val maxTokensVal = if (isVlmLoaded && !currentVisionDisabled) baseMaxTokens.coerceAtMost(8192) else baseMaxTokens
        val temperatureVal = overrideTemperature ?: 0.7f
        val topKVal = overrideTopK ?: 40
        val topPVal = overrideTopP ?: 0.9f
        
        val isThinkingModel = model.name.contains("Thinking", ignoreCase = true) || 
                              model.name.contains("Reasoning", ignoreCase = true)
        
        val job = launch(Dispatchers.IO) {
            try {
                if (isVlmLoaded && vlmWrapper != null) {
                    // === VLM path: use VlmChatMessage + VlmContent for images ===
                    val vlm = vlmWrapper!!
                    
                    // Extract the actual user text from the prompt
                    val userText = extractUserText(prompt, imagePaths.isNotEmpty())
                    Log.d(TAG, "VLM: User text: $userText")
                    
                    // Build VLM content list: images first, then text
                    val contents = mutableListOf<VlmContent>()
                    for (path in imagePaths) {
                        if (File(path).exists()) {
                            contents.add(VlmContent("image", path))
                            Log.d(TAG, "VLM: Added image content: $path")
                        } else {
                            Log.w(TAG, "VLM: Image file not found, skipping: $path")
                        }
                    }
                    contents.add(VlmContent("text", userText))
                    
                    val vlmMessages = arrayOf(
                        VlmChatMessage(role = "user", contents = contents)
                    )
                    
                    // Build base generation config
                    val baseConfig = GenerationConfig().apply {
                        try {
                            val cls = this::class.java
                            val fields = mapOf(
                                "maxTokens" to maxTokensVal,
                                "max_tokens" to maxTokensVal,
                                "temperature" to temperatureVal,
                                "topP" to topPVal,
                                "top_p" to topPVal,
                                "topK" to topKVal,
                                "top_k" to topKVal
                            )
                            for ((fname, value) in fields) {
                                try {
                                    cls.getDeclaredField(fname).apply { isAccessible = true }.set(this, value)
                                } catch (_: Exception) {}
                            }
                        } catch (_: Exception) {}
                    }
                    
                    // APPLY: time the template + inject + generate steps so we can measure bottlenecks
                    val tStart = System.currentTimeMillis()

                    val templateResult = vlm.applyChatTemplate(vlmMessages, null, isThinkingModel)
                    val tAfterTemplate = System.currentTimeMillis()
                    val formattedPrompt = if (templateResult.isSuccess) {
                        templateResult.getOrNull()?.formattedText?.takeIf { it.isNotEmpty() } ?: userText
                    } else {
                        Log.w(TAG, "VLM: applyChatTemplate failed, using raw text")
                        userText
                    }
                    Log.d(TAG, "VLM: Formatted prompt length: ${formattedPrompt.length} (template=${tAfterTemplate - tStart}ms)")

                    val configWithMedia = vlm.injectMediaPathsToConfig(vlmMessages, baseConfig)
                    val tAfterInject = System.currentTimeMillis()
                    Log.d(TAG, "VLM: Config has ${configWithMedia.imageCount} images (inject=${tAfterInject - tAfterTemplate}ms)")

                    // Generate using the SDK-formatted prompt and track time-to-first-token
                    val vlmStart = System.currentTimeMillis()
                    var firstTokenAt = 0L
                    vlm.generateStreamFlow(formattedPrompt, configWithMedia)
                        .collect { streamResult ->
                            if (isActive) {
                                if (streamResult is com.nexa.sdk.bean.LlmStreamResult.Token) {
                                    if (firstTokenAt == 0L) {
                                        firstTokenAt = System.currentTimeMillis()
                                        Log.i(TAG, "VLM: time-to-first-token=${firstTokenAt - vlmStart}ms (total prefill=${firstTokenAt - tStart}ms)")
                                    }
                                }
                                handleStreamResult(streamResult, isThinkingModel)
                            }
                        }
                } else {
                    // === LLM path: text-only generation ===
                    val wrapper = llmWrapper!!
                    val formattedPrompt = formatPrompt(prompt, model)
                    
                    val genConfig = GenerationConfig().apply {
                        try {
                            val cls = this::class.java
                            val fields = mapOf(
                                "maxTokens" to maxTokensVal,
                                "max_tokens" to maxTokensVal,
                                "temperature" to temperatureVal,
                                "topP" to topPVal,
                                "top_p" to topPVal,
                                "topK" to topKVal,
                                "top_k" to topKVal
                            )
                            for ((fname, value) in fields) {
                                try {
                                    cls.getDeclaredField(fname).apply { isAccessible = true }.set(this, value)
                                } catch (_: Exception) {}
                            }
                        } catch (_: Exception) {}
                    }
                    
                    wrapper.generateStreamFlow(formattedPrompt, genConfig)
                        .collect { streamResult ->
                            if (isActive) {
                                handleStreamResult(streamResult, isThinkingModel)
                            }
                        }
                }
                close()
            } catch (e: Exception) {
                Log.e(TAG, "Generation error", e)
                close(e)
            } finally {
                // Delay cleanup of temp images to avoid deleting them
                // before an auto-retry can re-use them
                CoroutineScope(Dispatchers.IO).launch {
                    kotlinx.coroutines.delay(5000)
                    imagePaths.forEach { path ->
                        try { File(path).delete() } catch (_: Exception) {}
                    }
                }
            }
        }
        
        awaitClose {
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    if (isVlmLoaded) vlmWrapper?.stopStream()
                    else llmWrapper?.stopStream()
                } catch (_: Exception) {}
            }
            job.cancel()
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Handle a single stream result token, applying thinking tag normalization.
     */
    private fun kotlinx.coroutines.channels.ProducerScope<String>.handleStreamResult(
        streamResult: LlmStreamResult,
        isThinkingModel: Boolean
    ) {
        when (streamResult) {
            is LlmStreamResult.Token -> {
                var text = streamResult.text
                if (isThinkingModel) {
                    if (text.contains("<think>")) {
                        text = text.replace("<think>", "\u200B\u200BTHINK\u200B\u200B")
                    }
                    if (text.contains("</think>")) {
                        text = text.replace("</think>", "\u200B\u200BENDTHINK\u200B\u200B")
                    }
                }
                trySend(text)
            }
            is LlmStreamResult.Completed -> close()
            is LlmStreamResult.Error -> {
                // Log detailed error information for debugging
                try {
                    val errorCode = streamResult::class.java.getDeclaredField("errorCode").apply { isAccessible = true }.get(streamResult)
                    Log.e(TAG, "VLM/LLM SDK Error - Code: $errorCode")
                } catch (e: Exception) {
                    Log.e(TAG, "VLM/LLM SDK Error (unable to extract error code)")
                }
                close(Exception("SDK Error"))
            }
        }
    }

    /**
     * Extract the actual user text from a formatted prompt.
     * Strips prompt scaffolding ("user: ", "assistant:") and replaces
     * placeholder / filename-only text with a proper VLM description request.
     */
    private fun extractUserText(prompt: String, hasImages: Boolean = true): String {
        val cleanPrompt = if (prompt.trimEnd().endsWith("assistant:")) {
            prompt.substringBeforeLast("assistant:").trimEnd()
        } else prompt
        
        // Try to find the last "user: " segment
        var result = cleanPrompt.trim()
        if (cleanPrompt.contains("user: ")) {
            val segments = cleanPrompt.split("\n\n").filter { it.isNotBlank() }
            // Find the last user segment that has real content (not just a filename)
            val meaningfulUserSegment = segments.findLast { seg ->
                val text = seg.trimStart().removePrefix("user: ").trim()
                seg.trimStart().startsWith("user: ") && !isPlaceholderText(text)
            }
            val lastUserSegment = meaningfulUserSegment 
                ?: segments.findLast { it.trimStart().startsWith("user: ") }
            if (lastUserSegment != null) {
                result = lastUserSegment.removePrefix("user: ").trim()
            }
        }
        
        // Replace placeholder / filename text with a real image prompt when images are attached
        if (hasImages && isPlaceholderText(result)) {
            result = "Describe what you see in this image in detail."
        }
        
        return result
    }

    /** Check if text is a placeholder like "Shared a file" or just a filename like "📄 photo.png" */
    private fun isPlaceholderText(text: String): Boolean {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) return true
        if (cleaned.equals("Shared a file", ignoreCase = true)) return true
        if (cleaned.contains("Shared a file", ignoreCase = true) && cleaned.length < 40) return true
        // Matches emoji + filename patterns like "📄 1000004995.png"
        val withoutEmoji = cleaned.replace(Regex("^[\\p{So}\\p{Sc}\\s]+"), "").trim()
        if (withoutEmoji.matches(Regex("^[\\w._-]+\\.(png|jpg|jpeg|gif|webp|bmp|svg)$", RegexOption.IGNORE_CASE))) return true
        return false
    }

    private suspend fun formatPrompt(prompt: String, model: LLMModel): String {
        val wrapper = llmWrapper ?: vlmWrapper
        if (wrapper == null) return prompt
        val llmWrap = llmWrapper ?: return prompt  // formatPrompt only works with LlmWrapper
        
        // 1. Parse into structured messages
        var messages = mutableListOf<ChatMessage>()
        
        // Remove trailing assistant marker if present (fix for "appended assistant" issue)
        val cleanPrompt = if (prompt.trimEnd().endsWith("assistant:")) {
             prompt.substringBeforeLast("assistant:").trimEnd()
        } else prompt

        if (cleanPrompt.contains("user: ") || cleanPrompt.contains("assistant: ")) {
            try {
                val segments = cleanPrompt.split("\n\n").filter { it.isNotBlank() }
                for (segment in segments) {
                    when {
                        segment.startsWith("user: ") -> messages.add(ChatMessage("user", segment.removePrefix("user: ").trim()))
                        segment.startsWith("assistant: ") -> messages.add(ChatMessage("assistant", segment.removePrefix("assistant: ").trim()))
                        else -> {
                            if (messages.isEmpty()) messages.add(ChatMessage("system", segment.trim()))
                            else {
                                // Append to last message if unsure
                                val last = messages.last()
                                val role = try { last::class.java.getDeclaredField("role").apply { isAccessible = true }.get(last) as String } catch(e:Exception) { "user" }
                                val content = try { last::class.java.getDeclaredField("content").apply { isAccessible = true }.get(last) as String } catch(e:Exception) { "" }
                                messages[messages.size - 1] = ChatMessage(role, content + "\n\n" + segment.trim())
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                // Parsing failed, proceed with empty messages
            }
        }

        // 2. Try SDK Template
        if (messages.isNotEmpty()) {
            try {
                val result = llmWrap.applyChatTemplate(messages.toTypedArray(), null, false)
                if (result.isSuccess) {
                    result.getOrNull()?.formattedText?.let { 
                        if (it.isNotEmpty()) return it 
                    }
                }
            } catch (e: Exception) {}
            
            // 3. Ministral/Mistral handling (Prioritize [INST] format over ChatML)
            if (model.name.contains("Ministral", ignoreCase = true) || model.name.contains("Mistral", ignoreCase = true)) {
                val sb = StringBuilder("<s>")
                val isReasoning = model.name.contains("Reasoning", ignoreCase = true) || model.name.contains("Thinking", ignoreCase = true)
                
                var systemInstr = if (isReasoning) "You are a reasoning model. Always output your internal thought process within <think> and </think> tags before your final answer.\n\n" else ""
                
                // Pre-scan for system messages to merge
                for (msg in messages) {
                    val role = try { msg::class.java.getDeclaredField("role").apply { isAccessible = true }.get(msg) as String } catch(e:Exception) { "user" }
                    val content = try { msg::class.java.getDeclaredField("content").apply { isAccessible = true }.get(msg) as String } catch(e:Exception) { "" }
                    if (role == "system") systemInstr += content + "\n\n"
                }

                var isFirstUser = true
                for (msg in messages) {
                    val role = try { msg::class.java.getDeclaredField("role").apply { isAccessible = true }.get(msg) as String } catch(e:Exception) { "user" }
                    val content = try { msg::class.java.getDeclaredField("content").apply { isAccessible = true }.get(msg) as String } catch(e:Exception) { "" }
                    
                    if (role == "system") continue 
                    
                    if (role == "user") {
                        if (!isFirstUser) sb.append(" ") 
                        sb.append("[INST] ")
                        if (isFirstUser && systemInstr.isNotEmpty()) {
                            sb.append(systemInstr)
                            isFirstUser = false
                        }
                        sb.append(content)
                        sb.append(" [/INST]")
                    } else if (role == "assistant") {
                        if (msg === messages.last() && content.isEmpty()) continue
                        sb.append(" $content</s>")
                    }
                }
                return sb.toString()
            }

            // 4. Fallback: Manual ChatML construction from parsed messages (Robust)
            val sb = StringBuilder()
            val isThinkingModel = model.name.contains("Thinking", ignoreCase = true) || 
                                  model.name.contains("Reasoning", ignoreCase = true)
            
            sb.append("<|im_start|>system\n")
            if (isThinkingModel) {
                sb.append("You are a reasoning model. Always output your internal thought process within <think> and </think> tags before your final answer.\n")
            } else {
                sb.append("You are a helpful assistant.\n")
            }
            sb.append("<|im_end|>\n")

            for (msg in messages) {
                val role = try { msg::class.java.getDeclaredField("role").apply { isAccessible = true }.get(msg) as String } catch(e:Exception) { "user" }
                val content = try { msg::class.java.getDeclaredField("content").apply { isAccessible = true }.get(msg) as String } catch(e:Exception) { "" }
                
                // Skip empty assistant trailing message (it's just a hook)
                if (role == "assistant" && content.isEmpty() && msg === messages.last()) continue
                
                sb.append("<|im_start|>$role\n$content<|im_end|>\n")
            }
            sb.append("<|im_start|>assistant\n")
            return sb.toString()
        }

        // 4. Fallback: Raw String Replacement (Legacy/Backup)
        return if (model.name.contains("Ministral", ignoreCase = true) || model.name.contains("Mistral", ignoreCase = true)) {
            val lastUser = prompt.substringAfterLast("user: ").substringBefore("\nassistant:").trim()
            val systemInstr = if (model.name.contains("Reasoning", ignoreCase = true)) 
                "You are a reasoning model. Output your thoughts in <think> tags." 
            else ""
            
            // Ministral uses [INST], let's try to inject system prompt if possible, or just prepend to user
            if (systemInstr.isNotEmpty()) {
                "[INST] $systemInstr\n$lastUser [/INST]\n"
            } else {
                "[INST]\n$lastUser\n[/INST]\n"
            }
        } else {
            // Generic ChatML-like fallback
            var p = prompt
            p = p.replaceFirst("user: ", "<|im_start|>user\n")
            p = p.replace("\n\nuser: ", "<|im_end|>\n<|im_start|>user\n")
            p = p.replace(Regex("\nassistant: ?"), "<|im_end|>\n<|im_start|>assistant\n")
            
            var result = "<|startoftext|>" + p
            if (!result.endsWith("<|im_start|>assistant\n")) {
                result += "<|im_end|>\n<|im_start|>assistant\n"
            }
            result
        }
    }

    override suspend fun resetChatSession(chatId: String) {
        // Clear KV cache by destroying and reloading the model wrapper.
        // The Nexa SDK has no explicit KV-cache-clear API, so a full
        // destroy + rebuild cycle is the only reliable way to reset state.
        try {
            val modelToReload = currentModel
            val backendToUse = currentPreferredBackend
            
            if (isVlmLoaded && vlmWrapper != null) {
                Log.d(TAG, "VLM: Destroying wrapper to clear vision state for new chat")
                vlmWrapper?.stopStream()
                vlmWrapper?.destroy()
                vlmWrapper = null
                
                // Reload the model to get fresh vision encoder state
                if (modelToReload != null) {
                    Log.d(TAG, "VLM: Reloading model ${modelToReload.name} for fresh state (visionDisabled=$currentVisionDisabled)")
                    loadModelInternal(modelToReload, backendToUse, currentVisionDisabled)
                }
            } else if (llmWrapper != null) {
                Log.d(TAG, "LLM: Destroying wrapper to clear KV cache")
                llmWrapper?.stopStream()
                llmWrapper?.destroy()
                llmWrapper = null
                
                // Reload the model to get a fresh KV cache
                if (modelToReload != null) {
                    Log.d(TAG, "LLM: Reloading model ${modelToReload.name} for fresh KV cache")
                    loadModelInternal(modelToReload, backendToUse, currentVisionDisabled)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error resetting chat session: ${e.message}")
        }
    }
    override suspend fun onCleared() { unloadModel() }

    override fun getCurrentlyLoadedModel(): LLMModel? = currentModel
    override fun getCurrentlyLoadedBackend(): LlmInference.Backend? = currentPreferredBackend
    override fun getMemoryWarningForImages(images: List<Bitmap>): String? = null
    override fun wasSessionRecentlyReset(chatId: String): Boolean = false
    override fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?) {
        overrideMaxTokens = maxTokens
        overrideTopK = topK
        overrideTopP = topP
        overrideTemperature = temperature
    }
    override fun isVisionCurrentlyDisabled(): Boolean = false
    override fun isAudioCurrentlyDisabled(): Boolean = false
    override fun isGpuBackendEnabled(): Boolean = currentPreferredBackend == LlmInference.Backend.GPU
    override fun getEffectiveMaxTokens(model: LLMModel): Int = overrideMaxTokens ?: model.contextWindowSize
}
