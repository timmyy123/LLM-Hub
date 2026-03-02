package com.llmhub.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.localFileName
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
import java.util.UUID
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
import com.llmhub.llmhub.R
import com.llmhub.llmhub.websearch.WebSearchService
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector

/** State machine states for parsing GPT-OSS Harmony format output. */
private enum class HarmonyState {
    BEFORE_HEADER,   // buffering until <|channel|>analysis<|message|>
    IN_ANALYSIS,     // streaming thinking content; watching for <|end|>
    IN_TRANSITION,   // silently consuming <|start|>assistant<|channel|>final<|message|>
    IN_FINAL         // streaming final answer directly
}

@Singleton
class NexaInferenceService @Inject constructor(
    private val context: Context
) : InferenceService {

    private val TAG = "NexaInferenceService"
    private val webSearchService: WebSearchService = DuckDuckGoSearchService()
    private var llmWrapper: LlmWrapper? = null
    private var vlmWrapper: VlmWrapper? = null
    private var isVlmLoaded: Boolean = false
    
    private var currentModel: LLMModel? = null
    private var currentPreferredBackend: LlmInference.Backend? = null
    private var currentVisionDisabled: Boolean = false
    private var currentAudioDisabled: Boolean = false
    
    private var overrideMaxTokens: Int? = null
    private var overrideTopK: Int? = null
    private var overrideTopP: Float? = null
    private var overrideTemperature: Float? = null
    private var overrideNGpuLayers: Int? = null
    private var overrideEnableThinking: Boolean? = null  // null = follow model defaults

    // Thinking sentinel tokens (same values as OnnxInferenceService)
    private val SENTINEL_THINK = "\u200B\u200BTHINK\u200B\u200B"
    private val SENTINEL_ENDTHINK = "\u200B\u200BENDTHINK\u200B\u200B"

    // Harmony format state machine (GPT-OSS models: <|channel|>analysis<|message|>...<|end|>...final)
    private val harmonyBuffer = StringBuilder()
    private var harmonyState = HarmonyState.BEFORE_HEADER
    
    // Whether the Nexa native SDK and its JNI bindings successfully initialized on this device.
    // UnsatisfiedLinkError is an Error (not Exception), so catch Throwable to avoid crashing the app
    // on platforms that do not provide Nexa native libraries (emulators, non‑NPU devices).
    private var nexaAvailable: Boolean = true

    init {
        try {
            NexaSdk.getInstance().init(context)
            // Clean up any stale VLM cache files from previous sessions
            cleanupStaleCacheFiles()
            nexaAvailable = true
        } catch (t: Throwable) {
            // Catch Throwable to include UnsatisfiedLinkError / LinkageError
            Log.e(TAG, "Nexa SDK unavailable on this device — disabling Nexa backend", t)
            nexaAvailable = false
        }
    }

    /**
     * Expose availability so callers (UnifiedInferenceService) can fall back when needed.
     */
    fun isAvailable(): Boolean = nexaAvailable

    /**
     * Lightweight runtime probe to verify whether Hexagon/HTP device access is usable.
     * - Attempts to build a tiny Nexa LlmWrapper using `plugin_id = "cpu_gpu"`,
     *   `device_id = deviceId`, and `nGpuLayers = 1` against any local GGUF model.
     * - Returns Pair(allowed:Boolean, reason:String?) where reason is non-null when denied.
     * - This is defensive: it catches native/SELinux failures and returns a human-friendly reason.
     */
    suspend fun probeNpuAvailability(deviceId: String = "dev0"): Pair<Boolean, String?> {
        if (!nexaAvailable) {
            Log.w(TAG, "NPU probe: Nexa SDK unavailable")
            return Pair(false, "Nexa SDK unavailable on device")
        }

        // Find any local GGUF model to use as a harmless probe target (do not change files)
        val modelsDir = File(context.filesDir, "models")
        val ggufFile = modelsDir.listFiles { _, name -> name.endsWith(".gguf", ignoreCase = true) && !name.contains("mmproj", ignoreCase = true) }?.firstOrNull()
        if (ggufFile == null) {
            Log.w(TAG, "NPU probe: no local GGUF model available to probe")
            return Pair(false, "no local GGUF model available to probe")
        }

        return try {
            val probeConfig = ModelConfig(nCtx = 8, nGpuLayers = 1)
            val createInput = LlmCreateInput(
                model_name = "",
                model_path = ggufFile.absolutePath,
                tokenizer_path = null,
                config = probeConfig,
                plugin_id = "cpu_gpu",
                device_id = deviceId
            )

            val buildResult = withContext(Dispatchers.IO) {
                LlmWrapper.builder()
                    .llmCreateInput(createInput)
                    .build()
            }

            if (buildResult.isSuccess) {
                // Close immediately — this was only a probe
                try { buildResult.getOrNull()?.close() } catch (_: Exception) {}
                Log.i(TAG, "NPU probe: allowed — device responded successfully (deviceId=$deviceId)")
                Pair(true, null)
            } else {
                val err = buildResult.exceptionOrNull()
                val reason = err?.message ?: "unknown error"
                val friendly = when {
                    reason.contains("Permission denied", ignoreCase = true) -> "Permission denied (fastrpc/SELinux)"
                    reason.contains("open_shell failed", ignoreCase = true) -> "vendor fastRPC open_shell failed"
                    reason.contains("Bad address", ignoreCase = true) -> "kernel FastRPC optimization failure"
                    else -> reason
                }
                Log.w(TAG, "NPU probe: denied — $friendly", err)
                Pair(false, friendly)
            }
        } catch (t: Throwable) {
            val reason = t.message ?: t.toString()
            val friendly = when {
                reason.contains("Permission denied", ignoreCase = true) -> "Permission denied (fastrpc/SELinux)"
                reason.contains("open_shell failed", ignoreCase = true) -> "vendor fastRPC open_shell failed"
                reason.contains("Bad address", ignoreCase = true) -> "kernel FastRPC optimization failure"
                else -> reason
            }
            Log.w(TAG, "NPU probe: denied — $friendly", t)
            Pair(false, friendly)
        }
    }

    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?, deviceId: String?): Boolean {
        // Default to vision enabled for the two-arg load; clear any previous override
        currentVisionDisabled = false
        currentAudioDisabled = false
        return loadModelInternal(model, preferredBackend, false, deviceId)
    }

    override suspend fun loadModel(
        model: LLMModel,
        preferredBackend: LlmInference.Backend?,
        disableVision: Boolean,
        disableAudio: Boolean,
        deviceId: String?
    ): Boolean {
         // Respect the caller's disableVision flag so we can load as text-only if requested
         currentVisionDisabled = disableVision
         currentAudioDisabled = disableAudio
         return loadModelInternal(model, preferredBackend, disableVision, deviceId)
    }
    
    private suspend fun loadModelInternal(model: LLMModel, preferredBackend: LlmInference.Backend?, disableVision: Boolean = false, deviceId: String? = null): Boolean {
        // If Nexa is unavailable (emulator / missing native libs), bail out immediately so the
        // UnifiedInferenceService can choose a different backend instead of crashing the app.
        if (!nexaAvailable) {
            Log.w(TAG, "Nexa backend not available on this device — refusing to load model: ${model.name}")
            return false
        }

        if (currentModel?.name == model.name && (llmWrapper != null || vlmWrapper != null)) {
            return true
        }

        unloadModel()

        val modelFile: File
        val modelDir = getModelDirectory(model)
        // If caller passed a deviceId but model is not GGUF, ignore it and log (NPU only supported for GGUF-on-Hexagon)
        if (!deviceId.isNullOrBlank() && model.modelFormat != "gguf") {
            Log.w(TAG, "DeviceId requested but model is not GGUF; ignoring deviceId=$deviceId for model.format=${model.modelFormat}")
        }

        // Handle imported models with content:// URIs (same as InferenceService)
        if (model.source == "Custom" && model.url.startsWith("content://")) {
            Log.d(TAG, "Loading imported GGUF model from URI: ${model.url}")
            val targetFile = File(context.filesDir, "models/${model.localFileName()}")
            targetFile.parentFile?.mkdirs()

            if (!targetFile.exists()) {
                try {
                    context.contentResolver.openInputStream(android.net.Uri.parse(model.url))?.use { input ->
                        targetFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.d(TAG, "Copied imported model to: ${targetFile.absolutePath}")
                } catch (e: SecurityException) {
                    Log.e(TAG, "Permission denied for URI: ${model.url}")
                    return false
                }
            } else {
                Log.d(TAG, "Using existing copied model: ${targetFile.absolutePath}")
            }

            if (!targetFile.exists()) {
                Log.e(TAG, "Failed to copy imported model from URI: ${model.url}")
                return false
            }
            modelFile = targetFile
        } else {
            modelFile = findGGUFModelFile(modelDir, model)

            if (!modelFile.exists()) {
                Log.e(TAG, "Model file not found: ${modelFile.absolutePath}")
                return false
            }
        }

        val backendsToTry = mutableListOf<String>()
        if (preferredBackend == LlmInference.Backend.GPU || preferredBackend == null) {
            backendsToTry.add("gpu")
        }
        backendsToTry.add("CPU")

        for (backendId in backendsToTry) {
            try {
                Log.d(TAG, "Attempting load with $backendId...")
                
                // Cap context size to prevent OOM on mobile devices.
                // GGUF models allocate KV cache proportional to nCtx.
                // VLM models' memory cost grows with nCtx. Cap to 8192 for vision-enabled GGUF to keep allocations reasonable
                // For text models, determine maximum safe context based on total device RAM.
                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
                val memoryInfo = android.app.ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memoryInfo)
                val totalRamGb = memoryInfo.totalMem.toDouble() / (1024 * 1024 * 1024)
                
                val MAX_SAFE_CTX = if (model.supportsVision && !disableVision) {
                    8192
                } else {
                    when {
                        totalRamGb >= 20.0 -> 131072 // e.g. 24GB devices (true heavyweights)
                        totalRamGb >= 14.0 -> 65536  // e.g. 16GB devices
                        totalRamGb >= 10.0 -> 32768  // e.g. 12GB devices
                        totalRamGb >= 6.0 -> 16384   // e.g. 8GB devices
                        else -> 8192                 // < 6GB devices
                    }
                }
                
                val rawCtx = overrideMaxTokens ?: model.contextWindowSize
                val nCtx = rawCtx.coerceAtMost(MAX_SAFE_CTX)
                if (rawCtx != nCtx) {
                    Log.w(TAG, "Capped nCtx from $rawCtx to $nCtx to prevent OOM (Device RAM ~${String.format("%.1f", totalRamGb)}GB)")
                }
                // Determine device/plugin to use. If caller provided an explicit deviceId (e.g. "HTP0") prefer it
                val userRequestedNpu = !deviceId.isNullOrBlank() && deviceId.startsWith("dev", ignoreCase = true)
                var isNpuRequested = userRequestedNpu

                // If a dev (NPU/Hexagon) device is requested (either explicitly by user or by auto-selection),
                // run a lightweight probe. Behavior differs based on intent:
                // - explicit user request: fail fast (return false) if probe denies — do NOT fall back
                // - automatic request: fall back to gpu on probe denial
                if (isNpuRequested) {
                    val (allowed, reason) = probeNpuAvailability(deviceId ?: "dev0")
                    if (!allowed) {
                        if (userRequestedNpu) {
                            // User explicitly asked for NPU — do not silently fall back. Return failure
                            Log.e(TAG, "NPU probe denied for explicit request deviceId=$deviceId; reason=$reason — refusing to load model")
                            return false
                        } else {
                            Log.w(TAG, "NPU probe denied for deviceId=$deviceId; reason=$reason — falling back to gpu")
                            isNpuRequested = false
                        }
                    } else {
                        Log.i(TAG, "NPU probe allowed for deviceId=$deviceId — proceeding with Hexagon")
                    }
                }

                val deviceToUse = when {
                    backendId == "CPU" -> null
                    isNpuRequested -> deviceId // explicit Hexagon device like dev0
                    else -> "gpu"
                }

                // nGpuLayers > 0 is required to enable offloading to GPU/Hexagon (GGUF cpu_gpu path).
                // Nexa docs: device_id="gpu" for GPU, "dev0" for NPU/Hexagon.
                // When the user has set a custom layer count via the slider, honour it; otherwise default 999.
                val gpuLayers = when {
                    backendId == "CPU" -> 0
                    else -> overrideNGpuLayers ?: 999
                }

                Log.i(TAG, "Load config: backend=$backendId deviceId=$deviceId nGpuLayers=$gpuLayers (override=$overrideNGpuLayers) nCtx=$nCtx enableThinking=$overrideEnableThinking")

                if (isNpuRequested) {
                    Log.i(TAG, "NPU requested: deviceId=$deviceId -> nGpuLayers=$gpuLayers plugin=cpu_gpu (per Nexa docs)")
                }

                val isThinkingModelForConfig = model.name.contains("Thinking", ignoreCase = true) ||
                    model.name.contains("Reasoning", ignoreCase = true)

                val modelConfig = ModelConfig(
                    nCtx = nCtx,
                    nGpuLayers = gpuLayers,
                    enable_thinking = isThinkingModelForConfig
                )

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
                        val resolvedBackend = if (!deviceToUse.isNullOrBlank() && deviceToUse.startsWith("dev", ignoreCase = true)) {
                            "NPU($deviceToUse)"
                        } else {
                            backendId
                        }
                        Log.i(TAG, "✓ Successfully loaded VLM with $resolvedBackend backend")
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
                        val resolvedBackend = if (!deviceToUse.isNullOrBlank() && deviceToUse.startsWith("dev", ignoreCase = true)) {
                            "NPU($deviceToUse)"
                        } else {
                            backendId
                        }
                        Log.i(TAG, "✓ Successfully loaded LLM with $resolvedBackend backend")
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
        
        val modelName = modelFile.nameWithoutExtension
        val modelLower = modelName.lowercase()

        // 1) Prefer exact variant projector match first (BF16/F16/Q*)
        val variantRegex = Regex("""(?:q\d(?:_[a-z0-9]+)?|bf16|f16)""", RegexOption.IGNORE_CASE)
        val modelVariantRaw = variantRegex.find(modelLower)?.value?.lowercase()
        val modelVariant = when {
            modelVariantRaw == "f16" || modelVariantRaw == "bf16" -> "bf16"
            else -> modelVariantRaw
        }
        if (modelVariant != null) {
            val exactVariant = allMmproj.firstOrNull { candidate ->
                val candidateVariantRaw = variantRegex.find(candidate.nameWithoutExtension.lowercase())
                    ?.value
                    ?.lowercase()
                val candidateVariant = when {
                    candidateVariantRaw == "f16" || candidateVariantRaw == "bf16" -> "bf16"
                    else -> candidateVariantRaw
                }
                candidateVariant == modelVariant
            }
            if (exactVariant != null) return exactVariant
        }

        // 2) Try to match the model base name/family
        val modelBaseName = modelFile.nameWithoutExtension
            .replace(Regex("[-_]Q[0-9].*"), "") // Strip quant suffix
        
        val matched = allMmproj.firstOrNull { candidate ->
            val candidateBase = candidate.nameWithoutExtension
                .replace(Regex("[-_]BF16.*|[-_]mmproj.*", RegexOption.IGNORE_CASE), "")
            candidateBase.equals(modelBaseName, ignoreCase = true)
        }

        // 3) Fallback to first available projector if we couldn't resolve a better match
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
         if (!nexaAvailable) {
             // Nothing to do when Nexa isn't present
             llmWrapper = null
             vlmWrapper = null
             isVlmLoaded = false
             currentModel = null
             currentPreferredBackend = null
             return
         }

         try {
             if (isVlmLoaded) {
                 vlmWrapper?.stopStream()
                 vlmWrapper?.destroy()
             } else {
                 llmWrapper?.stopStream()
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
        val prepStart = System.currentTimeMillis()
        val imagePaths = if (images.isNotEmpty()) {
            images.mapIndexed { index, bitmap ->
                val oneImageStart = System.currentTimeMillis()
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
                Log.d(TAG, "VLM timing: image[$index] prep=${System.currentTimeMillis() - oneImageStart}ms path=${file.name}")
                file.absolutePath
            }
        } else {
            emptyList()
        }
        val imagePrepMs = System.currentTimeMillis() - prepStart
        if (images.isNotEmpty()) {
            Log.i(TAG, "VLM timing: image_prep_total=${imagePrepMs}ms images=${images.size}")
        }
        
        return generateResponseStreamInternal(prompt, model, imagePaths, webSearchEnabled, chatId, imagePrepMs)
    }

    private suspend fun generateResponseStreamInternal(
        prompt: String, 
        model: LLMModel, 
        imagePaths: List<String> = emptyList(),
        webSearchEnabled: Boolean = false,
        chatId: String = "",
        imagePrepMs: Long = 0L
    ): Flow<String> = callbackFlow {
        val requestId = UUID.randomUUID().toString().take(8)
        val requestStart = System.currentTimeMillis()
        Log.i(
            TAG,
            "GEN[$requestId] start model=${model.name} mode=${if (isVlmLoaded) "VLM" else "LLM"} images=${imagePaths.size} chatId=${if (chatId.isBlank()) "none" else chatId}"
        )

        if (llmWrapper == null && vlmWrapper == null) {
            close(IllegalStateException("Model not loaded"))
            return@callbackFlow
        }

        // --- Web Search ---
        val currentUserMessage = extractUserTextForSearch(prompt)
        val needsWebSearch = webSearchEnabled && SearchIntentDetector.needsWebSearch(currentUserMessage)
        var effectivePrompt = prompt

        if (needsWebSearch) {
            Log.d(TAG, "Web search detected for chat $chatId. Current message: '$currentUserMessage'")
            trySend(context.getString(R.string.web_searching))

            try {
                val searchQuery = SearchIntentDetector.extractSearchQuery(currentUserMessage)
                Log.d(TAG, "Extracted search query: '$searchQuery'")

                val searchResults = webSearchService.search(searchQuery, maxResults = 5)

                if (searchResults.isNotEmpty()) {
                    Log.d(TAG, "Found ${searchResults.size} search results")
                    trySend(context.getString(R.string.web_search_found_results, searchResults.size))

                    val resultsText = searchResults.joinToString("\n\n") { result ->
                        "SOURCE: ${result.source}\nTITLE: ${result.title}\nCONTENT: ${result.snippet}\n---"
                    }

                    effectivePrompt = """
                        CURRENT WEB SEARCH RESULTS:
                        $resultsText
                        
                        Based on the above current web search results, please answer the user's question: "$currentUserMessage"
                        
                        IMPORTANT INSTRUCTIONS:
                        - Use ONLY the information from the web search results above
                        - If the search results contain the answer, provide a clear and specific response
                        - If the search results don't contain enough information, say so clearly
                        - For dates and events, be specific based on what you find in the results
                        - Do not make up information not found in the search results
                        
                        Answer the question directly and clearly:
                    """.trimIndent()

                    Log.d(TAG, "Enhanced prompt created with ${searchResults.size} search results")
                } else {
                    Log.w(TAG, "No search results found for query: '$searchQuery'")
                    trySend(context.getString(R.string.web_search_no_results) + "\n\n")
                }
            } catch (searchException: Exception) {
                Log.e(TAG, "Web search failed for chat $chatId", searchException)
                trySend(context.getString(R.string.web_search_failed, searchException.message ?: "Unknown error") + "\n\n")
            }
        }
        
        val baseMaxTokens = overrideMaxTokens ?: model.contextWindowSize
        val maxTokensVal = if (isVlmLoaded && !currentVisionDisabled) baseMaxTokens.coerceAtMost(8192) else baseMaxTokens
        val temperatureVal = overrideTemperature ?: 0.7f
        val topKVal = overrideTopK ?: 40
        val topPVal = overrideTopP ?: 0.9f
        
        val isThinkingModel = model.name.contains("Thinking", ignoreCase = true) ||
                              model.name.contains("Reasoning", ignoreCase = true)
        val isHarmonyModel = model.name.contains("gpt-oss", ignoreCase = true) ||
                             model.name.contains("gpt_oss", ignoreCase = true)

        // Thinking toggle:
        // - LFM-Thinking: /no_think is injected by formatPrompt into the formatted string.
        // - GPT-OSS Harmony: an empty analysis prefill is injected after formatting in the
        //   LLM generation path so template processing cannot strip it.
        val thinkingEnabled = overrideEnableThinking ?: true

        // Reset per-generation Harmony state
        harmonyBuffer.clear()
        harmonyState = if (!thinkingEnabled && isHarmonyModel) HarmonyState.IN_FINAL else HarmonyState.BEFORE_HEADER

        val job = launch(Dispatchers.IO) {
            try {
                if (isVlmLoaded && vlmWrapper != null) {
                    // === VLM path: use VlmChatMessage + VlmContent for images ===
                    val vlm = vlmWrapper!!
                    
                    // Extract the actual user text from the prompt
                    val userText = extractUserText(effectivePrompt, imagePaths.isNotEmpty())
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
                    Log.d(TAG, "GEN[$requestId] VLM template=${tAfterTemplate - tStart}ms prompt_len=${formattedPrompt.length}")

                    val configWithMedia = vlm.injectMediaPathsToConfig(vlmMessages, baseConfig)
                    val tAfterInject = System.currentTimeMillis()
                    Log.d(TAG, "GEN[$requestId] VLM inject_media=${tAfterInject - tAfterTemplate}ms image_count=${configWithMedia.imageCount}")

                    // Generate using the SDK-formatted prompt and track time-to-first-token
                    val vlmStart = System.currentTimeMillis()
                    var firstTokenAt = 0L
                    var tokenCount = 0L
                    try {
                        vlm.generateStreamFlow(formattedPrompt, configWithMedia)
                            .collect { streamResult ->
                                if (isActive) {
                                    if (streamResult is com.nexa.sdk.bean.LlmStreamResult.Token) {
                                        tokenCount++
                                        if (firstTokenAt == 0L) {
                                            firstTokenAt = System.currentTimeMillis()
                                            val prefillOnlyMs = firstTokenAt - vlmStart
                                            val totalToFirstTokenMs = firstTokenAt - requestStart
                                            Log.i(
                                                TAG,
                                                "GEN[$requestId] VLM first_token prefill=${prefillOnlyMs}ms total_to_first_token=${totalToFirstTokenMs}ms image_prep=${imagePrepMs}ms"
                                            )
                                        }
                                    } else if (streamResult is com.nexa.sdk.bean.LlmStreamResult.Completed) {
                                        val end = System.currentTimeMillis()
                                        val decodeMs = if (firstTokenAt > 0L) end - firstTokenAt else 0L
                                        val totalMs = end - requestStart
                                        Log.i(TAG, "GEN[$requestId] VLM completed total=${totalMs}ms decode=${decodeMs}ms tokens=$tokenCount")
                                    }
                                    handleStreamResult(streamResult, isThinkingModel, isHarmonyModel)
                                }
                            }
                    } catch (t: Throwable) {
                        if (t is kotlinx.coroutines.CancellationException || t is java.util.concurrent.CancellationException) {
                            Log.d(TAG, "Nexa VLM generation cancelled; keeping Nexa backend available")
                            try {
                                vlmWrapper?.stopStream()
                            } catch (e: Exception) {
                                Log.w(TAG, "Failed to stop VLM stream on cancellation: ${e.message}")
                            }
                            close()
                            return@launch
                        }
                        // Defensive: mark Nexa unavailable on severe native/SDK failures and surface an error
                        Log.e(TAG, "Fatal error during Nexa VLM generation; disabling Nexa backend", t)
                        nexaAvailable = false
                        close(Exception("Nexa backend fatal error: ${t.message}"))
                        return@launch
                    }
                } else {
                    // === LLM path: text-only generation ===
                    val wrapper = llmWrapper!!
                    var formattedPrompt = formatPrompt(effectivePrompt, model, thinkingEnabled)
                    if (!thinkingEnabled && isHarmonyModel) {
                        formattedPrompt = formattedPrompt.trimEnd() +
                            "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
                    }
                    
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
                    
                    val llmStart = System.currentTimeMillis()
                    var firstTokenAt = 0L
                    var tokenCount = 0L
                    try {
                        wrapper.generateStreamFlow(formattedPrompt, genConfig)
                            .collect { streamResult ->
                                if (isActive) {
                                    if (streamResult is com.nexa.sdk.bean.LlmStreamResult.Token) {
                                        tokenCount++
                                        if (firstTokenAt == 0L) {
                                            firstTokenAt = System.currentTimeMillis()
                                            Log.i(
                                                TAG,
                                                "GEN[$requestId] LLM first_token prefill=${firstTokenAt - llmStart}ms total_to_first_token=${firstTokenAt - requestStart}ms"
                                            )
                                        }
                                    } else if (streamResult is com.nexa.sdk.bean.LlmStreamResult.Completed) {
                                        val end = System.currentTimeMillis()
                                        val decodeMs = if (firstTokenAt > 0L) end - firstTokenAt else 0L
                                        val totalMs = end - requestStart
                                        Log.i(TAG, "GEN[$requestId] LLM completed total=${totalMs}ms decode=${decodeMs}ms tokens=$tokenCount")
                                    }
                                    handleStreamResult(streamResult, isThinkingModel, isHarmonyModel)
                                }
                            }
                    } catch (t: Throwable) {
                        if (t is kotlinx.coroutines.CancellationException || t is java.util.concurrent.CancellationException) {
                            Log.d(TAG, "Nexa LLM generation cancelled; keeping Nexa backend available")
                            try {
                                llmWrapper?.stopStream()
                            } catch (e: Exception) {
                                Log.w(TAG, "Failed to stop LLM stream on cancellation: ${e.message}")
                            }
                            close()
                            return@launch
                        }
                        Log.e(TAG, "Fatal error during Nexa LLM generation; disabling Nexa backend", t)
                        nexaAvailable = false
                        close(Exception("Nexa backend fatal error: ${t.message}"))
                        return@launch
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
     * For Harmony-format models (GPT-OSS), routes through [emitTokenForHarmony].
     */
    private fun kotlinx.coroutines.channels.ProducerScope<String>.handleStreamResult(
        streamResult: LlmStreamResult,
        isThinkingModel: Boolean,
        isHarmonyModel: Boolean = false
    ) {
        when (streamResult) {
            is LlmStreamResult.Token -> {
                val text = streamResult.text
                when {
                    isHarmonyModel -> emitTokenForHarmony(text) { trySend(it) }
                    isThinkingModel -> {
                        var t = text
                        if (t.contains("<think>")) t = t.replace("<think>", SENTINEL_THINK)
                        if (t.contains("</think>")) t = t.replace("</think>", SENTINEL_ENDTHINK)
                        trySend(t)
                    }
                    else -> trySend(text)
                }
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
     * State-machine parser for GPT-OSS Harmony format output.
     *
     * Harmony output format:
     *   <|channel|>analysis<|message|>THINKING_CONTENT<|end|><|start|>assistant<|channel|>final<|message|>FINAL_ANSWER
     *
     * States (harmonyState):
     *   BEFORE_HEADER  — buffer silently until full analysis header arrives
     *   IN_ANALYSIS    — emit SENTINEL_THINK once, stream analysis chars, hold `endTag.length-1`
     *                    tail bytes to guard against partial tag splits across tokens
     *   IN_TRANSITION  — buffer silently until the full final-answer header is consumed
     *   IN_FINAL       — pass every new char straight through to the UI
     */
    private fun emitTokenForHarmony(tokenText: String, send: (String) -> Unit) {
        harmonyBuffer.append(tokenText)

        val analysisHeader = "<|channel|>analysis<|message|>"
        val endTag         = "<|end|>"
        val finalHeader    = "<|start|>assistant<|channel|>final<|message|>"

        when (harmonyState) {
            HarmonyState.BEFORE_HEADER -> {
                val headerIdx = harmonyBuffer.indexOf(analysisHeader)
                if (headerIdx >= 0) {
                    // Discard everything up-to-and-including the header, keep the rest
                    val afterHeader = harmonyBuffer.substring(headerIdx + analysisHeader.length)
                    harmonyBuffer.setLength(0)
                    harmonyBuffer.append(afterHeader)
                    harmonyState = HarmonyState.IN_ANALYSIS
                    // Emit SENTINEL_THINK immediately so the UI shows the thinking section
                    send(SENTINEL_THINK)
                    // Process any chars that arrived after the header in this same token
                    if (harmonyBuffer.isNotEmpty()) emitTokenForHarmony("", send)
                }
                // else: header not yet complete — keep buffering
            }

            HarmonyState.IN_ANALYSIS -> {
                val buf = harmonyBuffer.toString()
                val endIdx = buf.indexOf(endTag)
                if (endIdx >= 0) {
                    // Flush content before <|end|>, then close the thinking section
                    val chunk = buf.substring(0, endIdx)
                    if (chunk.isNotEmpty()) send(chunk)
                    send(SENTINEL_ENDTHINK)
                    val remainder = buf.substring(endIdx + endTag.length)
                    harmonyBuffer.setLength(0)
                    harmonyBuffer.append(remainder)
                    harmonyState = HarmonyState.IN_TRANSITION
                    if (harmonyBuffer.isNotEmpty()) emitTokenForHarmony("", send)
                } else {
                    // No <|end|> yet — safely flush all but the last (endTag.length - 1) chars
                    // so a tag split across token boundaries is never emitted prematurely.
                    val safeLen = (buf.length - (endTag.length - 1)).coerceAtLeast(0)
                    if (safeLen > 0) {
                        send(buf.substring(0, safeLen))
                        harmonyBuffer.delete(0, safeLen)
                    }
                }
            }

            HarmonyState.IN_TRANSITION -> {
                val buf = harmonyBuffer.toString()
                val finalIdx = buf.indexOf(finalHeader)
                if (finalIdx >= 0) {
                    val afterFinal = buf.substring(finalIdx + finalHeader.length)
                    harmonyBuffer.setLength(0)
                    harmonyBuffer.append(afterFinal)
                    harmonyState = HarmonyState.IN_FINAL
                    if (harmonyBuffer.isNotEmpty()) emitTokenForHarmony("", send)
                }
                // else: final-answer header not yet complete — keep buffering
            }

            HarmonyState.IN_FINAL -> {
                // Emit everything in the buffer directly to the UI
                if (harmonyBuffer.isNotEmpty()) {
                    send(harmonyBuffer.toString())
                    harmonyBuffer.setLength(0)
                }
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

    /**
     * Extract the current user message from the prompt for web search intent detection.
     * Similar to extractCurrentUserMessage in OnnxInferenceService.
     */
    private fun extractUserTextForSearch(prompt: String): String {
        val lines = prompt.trim().split('\n')

        // Look for the last user message in the conversation
        for (i in lines.lastIndex downTo 0) {
            val line = lines[i].trim()
            if (line.startsWith("user:")) {
                return line.removePrefix("user:").trim()
            }
        }

        // If no "user:" prefix found, check if the entire prompt is just a user message
        if (!prompt.contains("assistant:") && !prompt.contains("user:")) {
            return prompt.trim()
        }

        // Fallback: return the last non-empty line that doesn't start with "assistant:"
        for (i in lines.lastIndex downTo 0) {
            val line = lines[i].trim()
            if (line.isNotEmpty() && !line.startsWith("assistant:")) {
                return line
            }
        }
        return prompt.trim()
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

    /**
     * Injects "/no_think" at the start of the last user turn in an already-formatted
     * prompt string. Handles ChatML (<|im_start|>user\n) and INST ([INST]) formats.
     * Falls back to prepending "/no_think " to the whole string if neither is found.
     */
    private fun injectNoThinkIntoFormatted(formatted: String): String {
        // ChatML: <|im_start|>user\nCONTENT<|im_end|>
        val chatMlMarker = "<|im_start|>user\n"
        val lastChatMl = formatted.lastIndexOf(chatMlMarker)
        if (lastChatMl >= 0) {
            val insert = lastChatMl + chatMlMarker.length
            return formatted.substring(0, insert) + "/no_think " + formatted.substring(insert)
        }
        // INST: [INST] CONTENT [/INST]
        val lastInst = formatted.lastIndexOf("[INST]")
        if (lastInst >= 0) {
            val insert = lastInst + "[INST]".length + 1   // +1 for the space after [INST]
            return formatted.substring(0, insert) + "/no_think " + formatted.substring(insert)
        }
        return "/no_think $formatted"
    }

    private suspend fun formatPrompt(prompt: String, model: LLMModel, thinkingEnabled: Boolean = true): String {
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
                var systemPromptText = ""
                val segments = cleanPrompt.split("\n\n").filter { it.isNotBlank() }
                
                for (segment in segments) {
                    when {
                        segment.startsWith("system: ") -> {
                            val content = segment.removePrefix("system: ").trim()
                            if (content.isNotEmpty()) {
                                systemPromptText += content + "\n\n"
                            }
                        }
                        segment.startsWith("user: ") -> {
                            val content = segment.removePrefix("user: ").trim()
                            if (messages.isEmpty() && systemPromptText.isNotEmpty()) {
                                // Inject system prompt into the first user turn.
                                // This solves issues with Gemma models and others that don't support a dedicated system role.
                                messages.add(ChatMessage("user", systemPromptText + content))
                                systemPromptText = "" // Clear it so we don't inject it again
                            } else {
                                messages.add(ChatMessage("user", content))
                            }
                        }
                        segment.startsWith("assistant: ") -> {
                            messages.add(ChatMessage("assistant", segment.removePrefix("assistant: ").trim()))
                        }
                        else -> {
                            // If it doesn't have a marker, consider it a system prompt if at the very beginning
                            if (messages.isEmpty()) {
                                systemPromptText += segment.trim() + "\n\n"
                            } else {
                                // Append to the last message
                                val last = messages.last()
                                val role = try { last::class.java.getDeclaredField("role").apply { isAccessible = true }.get(last) as String } catch(e:Exception) { "user" }
                                val content = try { last::class.java.getDeclaredField("content").apply { isAccessible = true }.get(last) as String } catch(e:Exception) { "" }
                                messages[messages.size - 1] = ChatMessage(role, content + "\n\n" + segment.trim())
                            }
                        }
                    }
                }
                // If there's STILL a system prompt but no user turn was found to attach it to, add it
                if (systemPromptText.isNotEmpty()) {
                    messages.add(0, ChatMessage("system", systemPromptText.trimEnd()))
                }
            } catch (e: Exception) {
                // Parsing failed, proceed with empty messages
            }
        }

        // If no conversation structure found (no "user:"/"assistant:" markers), treat
        // the prompt as either a system+user split (feature screens) or a bare user message.
        // Feature screens (WritingAid, ScamDetector, Translator, etc.) build prompts as
        // "instruction block\n\nX to process:\n{userInput}". Without splitting, GPT-OSS
        // receives the entire instruction block as a user message and treats it as a
        // meta-request rather than an instruction to execute — outputting the prompt text
        // instead of the actual result. We detect common separators and split so that
        // the instructions go into the system role and the user input goes into the user role.
        if (messages.isEmpty()) {
            // Vibe Coder style prompts often contain a large instruction block plus a
            // dedicated USER REQUEST field. Split those into system+user roles so
            // Harmony models execute the request instead of echoing meta-instructions.
            val quotedReqRegex = Regex("""(?is)\bUSER REQUEST\s*:\s*\"([\s\S]*?)\"""")
            val plainReqRegex = Regex("""(?im)^\s*User request\s*:\s*(.+)$""")
            val quotedReqMatch = quotedReqRegex.find(cleanPrompt)
            val plainReqMatch = plainReqRegex.find(cleanPrompt)
            val reqMatch = quotedReqMatch ?: plainReqMatch

            if (reqMatch != null) {
                val userContent = reqMatch.groupValues.getOrNull(1)?.trim().orEmpty()
                val systemContent = cleanPrompt.removeRange(reqMatch.range).trim()
                if (userContent.isNotEmpty() && systemContent.isNotEmpty()) {
                    messages.add(ChatMessage("system", systemContent))
                    messages.add(ChatMessage("user", userContent))
                }
            }

            // Creator prompt pattern:
            //   User Description: "..."
            //   Structure your response EXACTLY...
            // Split this into system instructions + user description to avoid generic outputs.
            if (messages.isEmpty()) {
                val creatorDescRegex = Regex(
                    """(?is)\bUser Description\s*:\s*"([\s\S]*?)"\s*(?=\n+\s*Structure your response EXACTLY)"""
                )
                val creatorMatch = creatorDescRegex.find(cleanPrompt)
                if (creatorMatch != null) {
                    val userContent = creatorMatch.groupValues.getOrNull(1)?.trim().orEmpty()
                    val systemContent = cleanPrompt.removeRange(creatorMatch.range).trim()
                    if (userContent.isNotEmpty() && systemContent.isNotEmpty()) {
                        messages.add(ChatMessage("system", systemContent))
                        messages.add(ChatMessage("user", userContent))
                    }
                }
            }

            val featureSeparators = listOf(
                "Text to rewrite:\n",
                "Content to analyze:\n",
                "Text to translate:\n",
                "Text to transcribe:\n",
                "Text to process:\n"
            )
            val sep = featureSeparators.firstOrNull { cleanPrompt.contains(it) }
            if (sep != null) {
                val idx = cleanPrompt.indexOf(sep)
                val instructions = cleanPrompt.substring(0, idx).trimEnd()
                val userContent = (sep.trim() + " " + cleanPrompt.substring(idx + sep.length)).trim()
                if (instructions.isNotEmpty() && userContent.isNotEmpty()) {
                    messages.add(ChatMessage("system", instructions))
                    messages.add(ChatMessage("user", userContent))
                } else {
                    messages.add(ChatMessage("user", cleanPrompt.trim()))
                }
            } else {
                messages.add(ChatMessage("user", cleanPrompt.trim()))
            }
        }

        if (messages.isNotEmpty()) {
            try {
                val result = llmWrap.applyChatTemplate(messages.toTypedArray(), null, false)
                if (result.isSuccess) {
                    result.getOrNull()?.formattedText?.let {
                        if (it.isNotEmpty()) {
                            // Inject /no_think for LFM-Thinking models — done on the formatted
                            // string after applyChatTemplate so it always lands in the user turn
                            // regardless of template format, with no reflection needed.
                            val isThinkingModelFmt = model.name.contains("Thinking", ignoreCase = true) ||
                                                      model.name.contains("Reasoning", ignoreCase = true)
                            val isHarmonyModelFmt  = model.name.contains("gpt-oss", ignoreCase = true) ||
                                                     model.name.contains("gpt_oss", ignoreCase = true)
                            if (!thinkingEnabled && isThinkingModelFmt && !isHarmonyModelFmt) {
                                return injectNoThinkIntoFormatted(it)
                            }
                            return it
                        }
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
    override fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?, nGpuLayers: Int?, enableThinking: Boolean?) {
        overrideMaxTokens = maxTokens
        overrideTopK = topK
        overrideTopP = topP
        overrideTemperature = temperature
        overrideNGpuLayers = nGpuLayers
        overrideEnableThinking = enableThinking
    }
    override fun isVisionCurrentlyDisabled(): Boolean = currentVisionDisabled
    override fun isAudioCurrentlyDisabled(): Boolean = currentAudioDisabled
    override fun isGpuBackendEnabled(): Boolean = currentPreferredBackend == LlmInference.Backend.GPU
    override fun getEffectiveMaxTokens(model: LLMModel): Int = overrideMaxTokens ?: model.contextWindowSize
}
