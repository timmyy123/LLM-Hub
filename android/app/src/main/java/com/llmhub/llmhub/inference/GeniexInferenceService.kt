package com.llmhub.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.play.core.assetpacks.AssetPackManagerFactory
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
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton
import java.util.UUID
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

// Correct GenieX SDK Imports
import com.geniex.sdk.GenieXSdk
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.VlmWrapper
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.VlmCreateInput
import com.geniex.sdk.bean.VlmChatMessage
import com.geniex.sdk.bean.VlmContent
import com.geniex.sdk.bean.ModelConfig
import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.LlmApplyChatTemplateOutput
import com.geniex.sdk.bean.SamplerConfig
import com.llmhub.llmhub.R
import com.llmhub.llmhub.websearch.WebSearchService
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
import com.llmhub.llmhub.websearch.WebSearchCitationStore

/** State machine states for parsing GPT-OSS Harmony format output. */
private enum class HarmonyState {
    BEFORE_HEADER,   // buffering until <|channel|>analysis<|message|>
    IN_ANALYSIS,     // streaming thinking content; watching for <|end|>
    IN_TRANSITION,   // silently consuming <|start|>assistant<|channel|>final<|message|>
    IN_FINAL         // streaming final answer directly
}

/** State machine states for parsing Meta Muse Glimmer ATEM format output. */
private enum class MuseGlimmerState {
    BEFORE_HEADER,   // buffering until to=self<|message|> or to=user<|message|>
    IN_REASONING,    // streaming thinking; watching for <|eom|> or to=user<|message|>
    IN_TRANSITION,   // silently consuming to=user header / <|start|>assistant to=user<|message|>
    IN_FINAL         // streaming final user answer directly
}

@Singleton
class GeniexInferenceService @Inject constructor(
    private val context: Context
) : InferenceService {

    private val TAG = "GeniexInferenceService"
    private val webSearchService: WebSearchService = DuckDuckGoSearchService()
    private var llmWrapper: LlmWrapper? = null
    private var vlmWrapper: VlmWrapper? = null
    private var isVlmLoaded: Boolean = false

    private var currentModel: LLMModel? = null
    private var currentPreferredBackend: LlmInference.Backend? = null
    private var currentIsNpu: Boolean = false
    private var currentDeviceId: String? = null
    private var lastDecodeSpeedTokPerSec: Double? = null
    private var currentVisionDisabled: Boolean = false
    private var currentAudioDisabled: Boolean = false
    // True once at least one token has been generated since the last model load/reset.
    // resetChatSession only needs to destroy+reload when this is true (stale recurrent state).
    private var hasGeneratedTokensSinceLoad: Boolean = false

    private var overrideMaxTokens: Int? = null
    private var overrideContextWindow: Int? = null
    private var overrideTopK: Int? = null
    private var overrideTopP: Float? = null
    private var overrideTemperature: Float? = null
    private var overrideNGpuLayers: Int? = null
    private var overrideEnableThinking: Boolean? = null  // null = follow model defaults

    // Thinking sentinel tokens (same values as OnnxInferenceService)
    private val SENTINEL_THINK = "​​THINK​​"
    private val SENTINEL_ENDTHINK = "​​ENDTHINK​​"

    // Harmony format state machine (GPT-OSS models: <|channel|>analysis<|message|>...<|end|>...final)
    private val harmonyBuffer = StringBuilder()
    private var harmonyState = HarmonyState.BEFORE_HEADER

    // Muse Glimmer format state machine (Meta Muse Glimmer models: to=self<|message|>...<|eom|>...to=user<|message|>)
    private val museBuffer = StringBuilder()
    private var museGlimmerState = MuseGlimmerState.BEFORE_HEADER

    // Whether the GenieX native SDK and its JNI bindings successfully initialized on this device.
    private var geniexAvailable: Boolean = false
    @Volatile
    private var isInitialized: Boolean = false

    private fun tryLoadOpenCL() {
        val candidates = listOf(
            "OpenCL",
            "/vendor/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/system/lib64/libOpenCL.so"
        )
        for (candidate in candidates) {
            try {
                if (candidate.startsWith("/")) {
                    val f = File(candidate)
                    if (f.exists()) {
                        System.load(candidate)
                        Log.i(TAG, "Loaded OpenCL vendor library from $candidate")
                        break
                    }
                } else {
                    System.loadLibrary(candidate)
                    Log.i(TAG, "Loaded OpenCL library: $candidate")
                    break
                }
            } catch (t: Throwable) {
                // Ignore candidate paths where OpenCL is unavailable
            }
        }
    }

    @Synchronized
    private fun ensureGeniexInitialized(): Boolean {
        if (isInitialized) return geniexAvailable
        try {
            extractQnnHtpLibsFromAssetPack()
            tryLoadOpenCL()
            val nativeLibDir = context.applicationInfo.nativeLibraryDir
            Log.i(TAG, "nativeLibraryDir=$nativeLibDir")
            Log.i(TAG, "libgeniex_plugin_llama_cpp.so exists=${File(nativeLibDir, "libgeniex_plugin_llama_cpp.so").exists()}")
            val sdk = GenieXSdk.getInstance()
            sdk.init(context)
            val rcLlama = sdk.registerPlugin("llama_cpp")
            Log.i(TAG, "registerPlugin(llama_cpp) rc=$rcLlama")
            val rcQairt = sdk.registerPlugin("qairt")
            Log.i(TAG, "registerPlugin(qairt) rc=$rcQairt")
            val llamaVer = try { sdk.getPluginVersion("llama_cpp") } catch (_: Throwable) { null }
            val qairtVer = try { sdk.getPluginVersion("qairt") } catch (_: Throwable) { null }
            geniexAvailable = (!llamaVer.isNullOrBlank() || !qairtVer.isNullOrBlank() || rcLlama == 0 || rcQairt == 0)
        } catch (t: Throwable) {
            Log.e(TAG, "GenieX SDK unavailable on this device — disabling GenieX backend", t)
            geniexAvailable = false
        } finally {
            isInitialized = true
        }
        return geniexAvailable
    }

    init {
        // Lightweight constructor — native GenieX JNI plugins are registered lazily in ensureGeniexInitialized()
    }

    private fun extractQnnHtpLibsFromAssetPack() {
        try {
            val assetPackManager = AssetPackManagerFactory.getInstance(context)
            val packLocations = assetPackManager.packLocations
            val npuPackPath = packLocations["geniex_npu_pack"]?.assetsPath() ?: return
            val npuDir = File(npuPackPath, "npu")
            if (!npuDir.exists() || !npuDir.isDirectory) return

            val targetDir = File(context.applicationInfo.nativeLibraryDir)
            val libs = npuDir.listFiles { f -> f.extension == "so" } ?: return
            var count = 0
            for (lib in libs) {
                val target = File(targetDir, lib.name)
                if (!target.exists() || target.length() != lib.length()) {
                    lib.inputStream().use { input ->
                        target.outputStream().use { output -> input.copyTo(output) }
                    }
                    target.setReadable(true, true)
                    target.setExecutable(true, true)
                    count++
                }
            }
            if (count > 0) Log.i(TAG, "Extracted $count QNN HTP libs from geniex_npu_pack")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract QNN HTP libs from asset pack (NPU may be unavailable)", e)
        }
    }

    /**
     * Expose availability so callers (UnifiedInferenceService) can fall back when needed.
     */
    fun isAvailable(): Boolean {
        // DO NOT call ensureGeniexInitialized() here — that triggers sdk.init() which calls
        // ggml_backend_load_all_from_path and crashes via SIGILL on incompatible Hexagon hardware.
        // If already initialized, return the cached result.
        if (isInitialized) return geniexAvailable
        // Otherwise do a cheap disk check: if the plugin SO doesn't exist, GenieX can't work.
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        return File(nativeLibDir, "libgeniex_plugin_llama_cpp.so").exists()
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
        // If GenieX is unavailable (emulator / missing native libs), bail out immediately so the
        // UnifiedInferenceService can choose a different backend instead of crashing the app.
        if (!ensureGeniexInitialized()) {
            Log.w(TAG, "GenieX backend not available on this device — refusing to load model: ${model.name}")
            return false
        }

        if (currentModel?.name == model.name && (llmWrapper != null || vlmWrapper != null)) {
            return true
        }

        hasGeneratedTokensSinceLoad = false
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

                val nCtx = overrideContextWindow ?: overrideMaxTokens ?: model.contextWindowSize
                // Determine compute_unit: "npu" for Hexagon/HTP, "gpu" for OpenCL, "cpu" for CPU
                val isNpuRequested = !deviceId.isNullOrBlank() && (
                    deviceId.startsWith("dev", ignoreCase = true) ||
                    deviceId.startsWith("htp", ignoreCase = true)
                )

                val deviceToUse = when {
                    backendId == "CPU" -> "cpu"
                    isNpuRequested -> "npu"
                    else -> "gpu"
                }
                val pluginToUse = "llama_cpp"

                // nGpuLayers > 0 is required to enable offloading to GPU/Hexagon (GGUF llama_cpp path).
                // GenieX compute_unit values: "cpu", "gpu", "npu", "hybrid" (from ComputeUnitValue enum).
                // When the user has set a custom layer count via the slider, honour it; otherwise default 999.
                val gpuLayers = when {
                    backendId == "CPU" -> 0
                    else -> overrideNGpuLayers ?: 999
                }

                val effectiveBackendLabel = if (isNpuRequested) "npu" else backendId
                Log.i(TAG, "Load config: backend=$effectiveBackendLabel deviceId=$deviceId nGpuLayers=$gpuLayers (override=$overrideNGpuLayers) nCtx=$nCtx enableThinking=$overrideEnableThinking")

                val isThinkingModelForConfig = model.name.contains("Thinking", ignoreCase = true) ||
                    model.name.contains("Reasoning", ignoreCase = true) ||
                    model.name.contains("LFM2.5-8B-A1B", ignoreCase = true) ||
                    model.name.contains("LFM-2.5 2.6B", ignoreCase = true)

                val modelConfig = ModelConfig(
                    nCtx = nCtx,
                    nGpuLayers = gpuLayers
                )
                Log.i(
                    TAG,
                    "GenieX create config: backend=$effectiveBackendLabel plugin=$pluginToUse device=$deviceToUse requestedNGpuLayers=${overrideNGpuLayers ?: 999} appliedNGpuLayers=${modelConfig.nGpuLayers}"
                )

                // Find mmproj path for VLM models (only when vision is enabled)
                val mmprojPath = if (model.supportsVision && !disableVision) {
                    findMmprojFile(modelDir, modelFile, model)?.absolutePath
                } else null

                // Use VlmWrapper for vision-capable models, LlmWrapper for text-only
                if (model.supportsVision && !disableVision && mmprojPath != null) {
                    Log.i(TAG, "Loading as VLM with mmproj: $mmprojPath")
                    val vlmCreateInput = VlmCreateInput(
                        model_path = modelFile.absolutePath,
                        mmproj_path = mmprojPath,
                        config = modelConfig,
                        runtime_id = pluginToUse,
                        compute_unit = deviceToUse
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
                        currentIsNpu = deviceToUse == "npu"
                        currentDeviceId = deviceId
                        currentVisionDisabled = disableVision
                        Log.i(
                            TAG,
                            "GenieX applied config (VLM): backend=${if (deviceToUse == "npu") "npu" else backendId} plugin=$pluginToUse device=$deviceToUse appliedNGpuLayers=${modelConfig.nGpuLayers}"
                        )
                        val resolvedBackend = if (deviceToUse == "npu") "NPU" else backendId
                        Log.i(TAG, "✓ Successfully loaded VLM with $resolvedBackend backend")
                        return true
                    } else {
                        val err = buildResult.exceptionOrNull()
                        Log.w(TAG, "VLM Failed $backendId: ${err?.message}")
                    }
                } else {
                    val createInput = LlmCreateInput(
                        model_path = modelFile.absolutePath,
                        tokenizer_path = null,
                        config = modelConfig,
                        runtime_id = pluginToUse,
                        compute_unit = deviceToUse
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
                        currentIsNpu = deviceToUse == "npu"
                        currentDeviceId = deviceId
                        currentVisionDisabled = disableVision
                        Log.i(
                            TAG,
                            "GenieX applied config (LLM): backend=${if (deviceToUse == "npu") "npu" else backendId} plugin=$pluginToUse device=$deviceToUse appliedNGpuLayers=${modelConfig.nGpuLayers}"
                        )
                        val resolvedBackend = if (deviceToUse == "npu") "NPU" else backendId
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
     * Find the mmproj file for vision models, supporting model-specific directories,
     * additionalFiles (for HuggingFace imports), and generic name/variant matching.
     */
    private fun findMmprojFile(modelDir: File, modelFile: File, model: LLMModel? = null): File? {
        val modelsDir = File(context.filesDir, "models")

        // 1) First check model.additionalFiles if specified (e.g. HuggingFace search imports)
        if (model != null && model.additionalFiles.isNotEmpty()) {
            for (urlOrPath in model.additionalFiles) {
                val fileName = urlOrPath.substringAfterLast("/").substringBefore("?").substringBefore("#")
                val inModelDir = File(modelDir, fileName)
                if (inModelDir.exists() && inModelDir.isFile) return inModelDir
                val inModelsDir = File(modelsDir, fileName)
                if (inModelsDir.exists() && inModelsDir.isFile) return inModelsDir
                val rawInModelDir = File(modelDir, urlOrPath)
                if (rawInModelDir.exists() && rawInModelDir.isFile) return rawInModelDir
                val rawInModelsDir = File(modelsDir, urlOrPath)
                if (rawInModelsDir.exists() && rawInModelsDir.isFile) return rawInModelsDir
            }
        }

        // 2) If model is in a dedicated folder (modelDir != modelsDir), look for any mmproj / projector in modelDir
        if (modelDir.exists() && modelDir.absolutePath != modelsDir.absolutePath) {
            val dirProjectors = modelDir.listFiles { _, name ->
                name.endsWith(".gguf", ignoreCase = true) &&
                    (name.contains("mmproj", ignoreCase = true) || name.contains("projector", ignoreCase = true))
            }
            if (!dirProjectors.isNullOrEmpty()) {
                if (dirProjectors.size == 1) return dirProjectors.first()
                val variantRegex = Regex("""(?:q\d(?:_[a-z0-9]+)?|bf16|f16|f32)""", RegexOption.IGNORE_CASE)
                val modelVariantRaw = variantRegex.find(modelFile.nameWithoutExtension.lowercase())?.value?.lowercase()
                val modelVariant = when {
                    modelVariantRaw == "f16" || modelVariantRaw == "bf16" -> "bf16"
                    else -> modelVariantRaw
                }
                if (modelVariant != null) {
                    val exact = dirProjectors.firstOrNull { candidate ->
                        val candVariantRaw = variantRegex.find(candidate.nameWithoutExtension.lowercase())?.value?.lowercase()
                        val candVariant = when {
                            candVariantRaw == "f16" || candVariantRaw == "bf16" -> "bf16"
                            else -> candVariantRaw
                        }
                        candVariant == modelVariant
                    }
                    if (exact != null) return exact
                }
                val bf16 = dirProjectors.firstOrNull { it.name.contains("bf16", ignoreCase = true) || it.name.contains("f16", ignoreCase = true) }
                if (bf16 != null) return bf16
                return dirProjectors.first()
            }
        }

        // 3) Scan modelDir and modelsDir for all mmproj/projector files
        val searchDirs = if (modelDir.absolutePath != modelsDir.absolutePath) listOf(modelDir, modelsDir) else listOf(modelsDir)
        val allMmproj = searchDirs.flatMap { dir ->
            dir.listFiles { _, name ->
                name.endsWith(".gguf", ignoreCase = true) &&
                    (name.contains("mmproj", ignoreCase = true) || name.contains("projector", ignoreCase = true))
            }?.toList() ?: emptyList()
        }.distinctBy { it.absolutePath }

        if (allMmproj.isEmpty()) return null

        if (allMmproj.size == 1) {
            return allMmproj.first()
        }

        val modelName = modelFile.nameWithoutExtension.lowercase()

        fun cleanName(name: String): String {
            return name.lowercase()
                .replace("mmproj", "")
                .replace("projector", "")
                .replace("vision", "")
                .replace(Regex("""[-_](?:q\d(?:_[a-z0-9]+)?|bf16|f16|f32|ud-[a-z0-9_]+|instruct|it|preview)"""), "")
                .replace(Regex("""[^a-z0-9]"""), "")
        }

        val modelCore = cleanName(modelName)

        val candidates = allMmproj.filter { candidate ->
            val candCore = cleanName(candidate.nameWithoutExtension)
            candCore.isNotEmpty() && modelCore.isNotEmpty() &&
                (modelCore.contains(candCore) || candCore.contains(modelCore))
        }.ifEmpty { allMmproj }

        // Match quantization variant
        val variantRegex = Regex("""(?:q\d(?:_[a-z0-9]+)?|bf16|f16|f32)""", RegexOption.IGNORE_CASE)
        val modelVariantRaw = variantRegex.find(modelName)?.value?.lowercase()
        val modelVariant = when {
            modelVariantRaw == "f16" || modelVariantRaw == "bf16" -> "bf16"
            else -> modelVariantRaw
        }
        if (modelVariant != null) {
            val exactVariant = candidates.firstOrNull { candidate ->
                val candVariantRaw = variantRegex.find(candidate.nameWithoutExtension.lowercase())?.value?.lowercase()
                val candVariant = when {
                    candVariantRaw == "f16" || candVariantRaw == "bf16" -> "bf16"
                    else -> candVariantRaw
                }
                candVariant == modelVariant
            }
            if (exactVariant != null) return exactVariant
        }

        // Prefer BF16/F16 projector as fallback
        val bf16Candidate = candidates.firstOrNull {
            it.name.contains("bf16", ignoreCase = true) || it.name.contains("f16", ignoreCase = true)
        }
        if (bf16Candidate != null) return bf16Candidate

        return candidates.firstOrNull()
    }

    /**
     * Clean up stale VLM cache files from previous sessions to prevent accumulation
     */
    private fun cleanupStaleCacheFiles() {
        try {
            val cacheDir = context.cacheDir
            val vlmFiles = cacheDir.listFiles { file ->
                file.name.startsWith("geniex_vlm_") && file.name.endsWith(".jpg")
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
         if (!geniexAvailable) {
             // Nothing to do when GenieX isn't present
             llmWrapper = null
             vlmWrapper = null
             isVlmLoaded = false
             currentModel = null
             currentPreferredBackend = null
             currentDeviceId = null
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
             Log.w(TAG, "Error closing GenieX model: ${e.message}")
         } finally {
             llmWrapper = null
             vlmWrapper = null
             isVlmLoaded = false
             currentModel = null
             currentPreferredBackend = null
             currentDeviceId = null
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

                val file = File(context.cacheDir, "geniex_vlm_${System.currentTimeMillis()}_$index.jpg")
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
        val needsWebSearch = webSearchEnabled
        var effectivePrompt = prompt

        if (needsWebSearch) {
            Log.d(TAG, "Web search detected for chat $chatId. Current message: '$currentUserMessage'")
            trySend(context.getString(R.string.web_searching))

            try {
                val searchQuery = SearchIntentDetector.extractSearchQuery(currentUserMessage)
                Log.d(TAG, "Extracted search query: '$searchQuery'")

                val searchResults = webSearchService.search(searchQuery, maxResults = 5)
                WebSearchCitationStore.put(chatId, searchResults)

                if (searchResults.isNotEmpty()) {
                    Log.d(TAG, "Found ${searchResults.size} search results")
                    trySend(context.getString(R.string.web_search_found_results, searchResults.size))

                    val resultsText = searchResults.joinToString("\n\n") { result ->
                        "SOURCE: ${result.source}\nTITLE: ${result.title}\nURL: ${result.url}\nCONTENT: ${result.snippet}\n---"
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
                        - Cite factual claims with the provided source URLs when possible

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

        val baseMaxTokens = overrideContextWindow ?: overrideMaxTokens ?: model.contextWindowSize
        val maxTokensVal = if (isVlmLoaded && !currentVisionDisabled) baseMaxTokens.coerceAtMost(8192) else baseMaxTokens
        val temperatureVal = overrideTemperature ?: 0.7f
        val topKVal = overrideTopK ?: 40
        val topPVal = overrideTopP ?: 0.9f

        val isThinkingModel = (model.name.contains("Thinking", ignoreCase = true) ||
                              model.name.contains("Reasoning", ignoreCase = true) ||
                              model.name.contains("LFM2.5-8B-A1B", ignoreCase = true) ||
                              model.name.contains("LFM-2.5 2.6B", ignoreCase = true)) &&
                              !model.name.contains("muse", ignoreCase = true)
        val isHarmonyModel = model.name.contains("gpt-oss", ignoreCase = true) ||
                             model.name.contains("gpt_oss", ignoreCase = true)
        val isMuseGlimmerModel = model.name.contains("muse glimmer", ignoreCase = true) ||
                                 model.name.contains("muse-glimmer", ignoreCase = true)

        // Thinking toggle:
        // - LFM-Thinking: /no_think is injected by formatPrompt into the formatted string.
        // - GPT-OSS Harmony: an empty analysis prefill is injected after formatting in the
        //   LLM generation path so template processing cannot strip it.
        // - Muse Glimmer: # Valid recipients and assistant to=user prefill control thinking.
        val thinkingEnabled = overrideEnableThinking ?: true

        // Reset per-generation Harmony state
        harmonyBuffer.clear()
        harmonyState = if (!thinkingEnabled && isHarmonyModel) HarmonyState.IN_FINAL else HarmonyState.BEFORE_HEADER

        // Reset per-generation Muse Glimmer state
        museBuffer.clear()
        museGlimmerState = if (!thinkingEnabled && isMuseGlimmerModel) MuseGlimmerState.IN_FINAL else MuseGlimmerState.BEFORE_HEADER

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
                    val vlmSampler = SamplerConfig(
                        temperatureVal, topPVal, topKVal,
                        0.05f, 1.1f, 0f, 0f, 0, null, null
                    )
                    val baseConfig = GenerationConfig().apply {
                        maxTokens = maxTokensVal
                        samplerConfig = vlmSampler
                    }

                    // APPLY: time the template + inject + generate steps so we can measure bottlenecks
                    val tStart = System.currentTimeMillis()

                    val templateResult = vlm.applyChatTemplate(vlmMessages, null, isThinkingModel && thinkingEnabled)
                    val tAfterTemplate = System.currentTimeMillis()
                    var formattedPrompt = if (templateResult.isSuccess) {
                        templateResult.getOrNull()?.formattedText?.takeIf { it.isNotEmpty() } ?: userText
                    } else {
                        Log.w(TAG, "VLM: applyChatTemplate failed, using raw text")
                        userText
                    }
                    if (isMuseGlimmerModel) {
                        if (!thinkingEnabled) {
                            formattedPrompt = formattedPrompt.replace("# Valid recipients: \"self\", \"user\".", "# Valid recipients: \"user\".")
                                .replace("# Valid recipients: 'self', 'user'.", "# Valid recipients: \"user\".")
                            if (formattedPrompt.endsWith("<|start|>assistant")) {
                                formattedPrompt = formattedPrompt.removeSuffix("<|start|>assistant") + "<|start|>assistant to=user<|message|>"
                            } else if (!formattedPrompt.contains("<|start|>assistant to=user<|message|>")) {
                                formattedPrompt = formattedPrompt.trimEnd() + "<|start|>assistant to=user<|message|>"
                            }
                        }
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
                                    if (streamResult is com.geniex.sdk.bean.LlmStreamResult.Token) {
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
                                    } else if (streamResult is com.geniex.sdk.bean.LlmStreamResult.Completed) {
                                        val end = System.currentTimeMillis()
                                        val decodeMs = if (firstTokenAt > 0L) end - firstTokenAt else 0L
                                        val totalMs = end - requestStart
                                        if (decodeMs > 0 && tokenCount > 0) {
                                            lastDecodeSpeedTokPerSec = tokenCount * 1000.0 / decodeMs
                                        }
                                        Log.i(TAG, "GEN[$requestId] VLM completed total=${totalMs}ms decode=${decodeMs}ms tokens=$tokenCount")
                                    }
                                    handleStreamResult(streamResult, isThinkingModel, isHarmonyModel, isMuseGlimmerModel, thinkingEnabled)
                                }
                            }
                    } catch (t: Throwable) {
                        if (t is kotlinx.coroutines.CancellationException || t is java.util.concurrent.CancellationException) {
                            Log.d(TAG, "GenieX VLM generation cancelled; keeping GenieX backend available")
                            try {
                                vlmWrapper?.stopStream()
                            } catch (e: Exception) {
                                Log.w(TAG, "Failed to stop VLM stream on cancellation: ${e.message}")
                            }
                            close()
                            return@launch
                        }
                        // Defensive: mark GenieX unavailable on severe native/SDK failures and surface an error
                        Log.e(TAG, "Fatal error during GenieX VLM generation; disabling GenieX backend", t)
                        geniexAvailable = false
                        close(Exception("GenieX backend fatal error: ${t.message}"))
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
                    if (!thinkingEnabled && isMuseGlimmerModel) {
                        if (!formattedPrompt.endsWith("<|start|>assistant to=user<|message|>")) {
                            formattedPrompt = formattedPrompt.trimEnd()
                            if (formattedPrompt.endsWith("<|start|>assistant")) {
                                formattedPrompt = formattedPrompt.removeSuffix("<|start|>assistant") + "<|start|>assistant to=user<|message|>"
                            }
                        }
                    }

                    val sampler = SamplerConfig(
                        temperatureVal, topPVal, topKVal,
                        0.05f, 1.1f, 0f, 0f, 0, null, null
                    )
                    val genConfig = GenerationConfig().apply {
                        maxTokens = maxTokensVal
                        samplerConfig = sampler
                    }

                    val llmStart = System.currentTimeMillis()
                    sentInitialThinkingSentinel = false
                    var firstTokenAt = 0L
                    var tokenCount = 0L
                    try {
                        wrapper.generateStreamFlow(formattedPrompt, genConfig)
                            .collect { streamResult ->
                                if (isActive) {
                                    if (streamResult is com.geniex.sdk.bean.LlmStreamResult.Token) {
                                        tokenCount++
                                        hasGeneratedTokensSinceLoad = true
                                        if (firstTokenAt == 0L) {
                                            firstTokenAt = System.currentTimeMillis()
                                            Log.i(
                                                TAG,
                                                "GEN[$requestId] LLM first_token prefill=${firstTokenAt - llmStart}ms total_to_first_token=${firstTokenAt - requestStart}ms"
                                            )
                                        }
                                    } else if (streamResult is com.geniex.sdk.bean.LlmStreamResult.Completed) {
                                        val end = System.currentTimeMillis()
                                        val decodeMs = if (firstTokenAt > 0L) end - firstTokenAt else 0L
                                        val totalMs = end - requestStart
                                        if (decodeMs > 0 && tokenCount > 0) {
                                            val sdkSpeed = runCatching {
                                                val cls = streamResult::class.java
                                                cls.declaredFields.firstOrNull { it.name.contains("decoding", true) || it.name.contains("speed", true) }?.let {
                                                    it.isAccessible = true
                                                    (it.get(streamResult) as? Number)?.toDouble()
                                                }
                                            }.getOrNull()
                                            lastDecodeSpeedTokPerSec = sdkSpeed ?: (tokenCount * 1000.0 / decodeMs)
                                        }
                                        Log.i(TAG, "GEN[$requestId] LLM completed total=${totalMs}ms decode=${decodeMs}ms tokens=$tokenCount")
                                    }
                                    handleStreamResult(streamResult, isThinkingModel, isHarmonyModel, isMuseGlimmerModel, thinkingEnabled)
                                }
                            }
                    } catch (t: Throwable) {
                        if (t is kotlinx.coroutines.CancellationException || t is java.util.concurrent.CancellationException) {
                            Log.d(TAG, "GenieX LLM generation cancelled; keeping GenieX backend available")
                            try {
                                llmWrapper?.stopStream()
                            } catch (e: Exception) {
                                Log.w(TAG, "Failed to stop LLM stream on cancellation: ${e.message}")
                            }
                            close()
                            return@launch
                        }
                        Log.e(TAG, "Fatal error during GenieX LLM generation; disabling GenieX backend", t)
                        geniexAvailable = false
                        close(Exception("GenieX backend fatal error: ${t.message}"))
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
    private var sentInitialThinkingSentinel = false

    private fun kotlinx.coroutines.channels.ProducerScope<String>.handleStreamResult(
        streamResult: LlmStreamResult,
        isThinkingModel: Boolean,
        isHarmonyModel: Boolean = false,
        isMuseGlimmerModel: Boolean = false,
        thinkingEnabled: Boolean = true
    ) {
        when (streamResult) {
            is LlmStreamResult.Token -> {
                val text = streamResult.text
                when {
                    isMuseGlimmerModel -> emitTokenForMuseGlimmer(text, thinkingEnabled) { trySend(it) }
                    isHarmonyModel -> emitTokenForHarmony(text) { trySend(it) }
                    isThinkingModel -> {
                        var t = text
                        if (!sentInitialThinkingSentinel && !t.contains("<think>")) {
                            sentInitialThinkingSentinel = true
                            t = SENTINEL_THINK + t
                        }
                        if (t.contains("<think>")) {
                            sentInitialThinkingSentinel = true
                            t = t.replace("<think>", SENTINEL_THINK)
                        }
                        if (t.contains("</think>")) t = t.replace("</think>", SENTINEL_ENDTHINK)
                        trySend(t)
                    }
                    else -> trySend(text)
                }
            }
            is LlmStreamResult.Completed -> {
                sentInitialThinkingSentinel = false
                if (isMuseGlimmerModel) {
                    if (museGlimmerState == MuseGlimmerState.IN_REASONING) {
                        var remaining = museBuffer.toString().replace("<|eot|>", "").replace("<|end_of_text|>", "")
                        val userIdx = findMuseUserHeaderIndex(remaining)
                        if (userIdx >= 0) {
                            remaining = remaining.substring(0, userIdx)
                        }
                        if (remaining.isNotEmpty()) trySend(remaining)
                        trySend(SENTINEL_ENDTHINK)
                    } else if (museGlimmerState == MuseGlimmerState.IN_FINAL || museGlimmerState == MuseGlimmerState.IN_TRANSITION) {
                        var remaining = museBuffer.toString().replace("<|eot|>", "").replace("<|end_of_text|>", "")
                        val cleaned = stripLeadingMuseUserHeader(remaining)
                        if (cleaned != null) {
                            remaining = cleaned
                        }
                        if (remaining.startsWith("<|message|>")) {
                            remaining = remaining.removePrefix("<|message|>").trimStart('\n', '\r', ' ')
                        }
                        if (remaining.isNotEmpty()) trySend(remaining)
                    }
                    museBuffer.setLength(0)
                    museGlimmerState = MuseGlimmerState.BEFORE_HEADER
                }
                close()
            }
            is LlmStreamResult.Error -> {
                // Log detailed SDK error fields (field names vary by GenieX SDK builds)
                val cls = streamResult::class.java
                val code = runCatching {
                    cls.declaredFields.firstOrNull {
                        it.name.equals("errorCode", true) ||
                            it.name.equals("code", true) ||
                            it.name.equals("errCode", true)
                    }?.let {
                        it.isAccessible = true
                        it.get(streamResult)?.toString()
                    }
                }.getOrNull()
                val message = runCatching {
                    cls.declaredFields.firstOrNull {
                        it.name.equals("message", true) ||
                            it.name.equals("errorMessage", true) ||
                            it.name.equals("msg", true)
                    }?.let {
                        it.isAccessible = true
                        it.get(streamResult)?.toString()
                    }
                }.getOrNull()
                Log.e(TAG, "VLM/LLM SDK Error - code=${code ?: "unknown"} message=${message ?: "unknown"} class=${cls.simpleName}")
                close(Exception("SDK Error code=${code ?: "unknown"} message=${message ?: "unknown"}"))
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
     * State-machine parser for Meta Muse Glimmer ATEM format output.
     *
     * Format when thinking enabled:
     *   to=self<|message|>REASONING_CONTENT<|eom|><|start|>assistant to=user<|message|>FINAL_ANSWER<|eot|>
     *   (or to=self<|message|>...to=user<|message|>FINAL_ANSWER<|eot|>)
     *
     * Format when thinking disabled:
     *   FINAL_ANSWER<|eot|> (prompt ends with prefill <|start|>assistant to=user<|message|>)
     */
    private fun emitTokenForMuseGlimmer(tokenText: String, thinkingEnabled: Boolean, send: (String) -> Unit) {
        museBuffer.append(tokenText)

        when (museGlimmerState) {
            MuseGlimmerState.BEFORE_HEADER -> {
                val buf = museBuffer.toString()
                if (!thinkingEnabled) {
                    val cleaned = stripLeadingMuseUserHeader(buf)
                    if (cleaned != null) {
                        museBuffer.setLength(0)
                        museBuffer.append(cleaned)
                        museGlimmerState = MuseGlimmerState.IN_FINAL
                        if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    } else if (isPotentialMuseHeaderPrefix(buf)) {
                        return
                    } else {
                        museGlimmerState = MuseGlimmerState.IN_FINAL
                        if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    }
                    return
                }

                // Thinking is enabled:
                val cleanedSelf = stripLeadingMuseSelfHeader(buf)
                if (cleanedSelf != null) {
                    museBuffer.setLength(0)
                    museBuffer.append(cleanedSelf)
                    museGlimmerState = MuseGlimmerState.IN_REASONING
                    send(SENTINEL_THINK)
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }

                val cleanedUser = stripLeadingMuseUserHeader(buf)
                if (cleanedUser != null) {
                    museBuffer.setLength(0)
                    museBuffer.append(cleanedUser)
                    museGlimmerState = MuseGlimmerState.IN_FINAL
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }

                if (isPotentialMuseHeaderPrefix(buf)) {
                    return
                }

                if (buf.length >= 40) {
                    museGlimmerState = MuseGlimmerState.IN_REASONING
                    send(SENTINEL_THINK)
                    val safeLen = (museBuffer.length - 20).coerceAtLeast(0)
                    if (safeLen > 0) {
                        send(museBuffer.substring(0, safeLen))
                        museBuffer.delete(0, safeLen)
                    }
                }
            }

            MuseGlimmerState.IN_REASONING -> {
                val buf = museBuffer.toString()
                val eomMarker = "<|eom|>"
                val eomIdx = buf.indexOf(eomMarker)
                if (eomIdx >= 0) {
                    val reasoningChunk = buf.substring(0, eomIdx)
                    if (reasoningChunk.isNotEmpty()) send(reasoningChunk)
                    send(SENTINEL_ENDTHINK)

                    val remainder = buf.substring(eomIdx + eomMarker.length)
                    museBuffer.setLength(0)
                    museBuffer.append(remainder)
                    museGlimmerState = MuseGlimmerState.IN_TRANSITION
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }

                val userHeaderIdx = findMuseUserHeaderIndex(buf)
                if (userHeaderIdx >= 0) {
                    val reasoningChunk = buf.substring(0, userHeaderIdx)
                    if (reasoningChunk.isNotEmpty()) send(reasoningChunk)
                    send(SENTINEL_ENDTHINK)

                    val remainder = buf.substring(userHeaderIdx)
                    museBuffer.setLength(0)
                    museBuffer.append(remainder)
                    museGlimmerState = MuseGlimmerState.IN_TRANSITION
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }

                val eotIdx = buf.indexOf("<|eot|>")
                if (eotIdx >= 0) {
                    val reasoningChunk = buf.substring(0, eotIdx)
                    if (reasoningChunk.isNotEmpty()) send(reasoningChunk)
                    send(SENTINEL_ENDTHINK)
                    museBuffer.setLength(0)
                    museGlimmerState = MuseGlimmerState.IN_FINAL
                    return
                }

                val maxMarkerLen = 45
                val safeLen = (buf.length - maxMarkerLen).coerceAtLeast(0)
                if (safeLen > 0) {
                    send(buf.substring(0, safeLen))
                    museBuffer.delete(0, safeLen)
                }
            }

            MuseGlimmerState.IN_TRANSITION -> {
                val buf = museBuffer.toString()
                val cleaned = stripLeadingMuseUserHeader(buf)
                if (cleaned != null) {
                    museBuffer.setLength(0)
                    museBuffer.append(cleaned)
                    museGlimmerState = MuseGlimmerState.IN_FINAL
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }

                val msgIdx = buf.indexOf("<|message|>")
                if (msgIdx >= 0) {
                    val remainder = buf.substring(msgIdx + "<|message|>".length).trimStart('\n', '\r', ' ')
                    museBuffer.setLength(0)
                    museBuffer.append(remainder)
                    museGlimmerState = MuseGlimmerState.IN_FINAL
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }

                if (isPotentialMuseUserHeaderPrefix(buf)) {
                    return
                }

                val remainder = buf.trimStart('\n', '\r', ' ')
                museBuffer.setLength(0)
                museBuffer.append(remainder)
                museGlimmerState = MuseGlimmerState.IN_FINAL
                if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
            }

            MuseGlimmerState.IN_FINAL -> {
                val buf = museBuffer.toString()
                val cleanedBuf = stripLeadingMuseUserHeader(buf)
                if (cleanedBuf != null) {
                    museBuffer.setLength(0)
                    museBuffer.append(cleanedBuf)
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }
                if (buf.startsWith("<|message|>")) {
                    val afterMsg = buf.removePrefix("<|message|>").trimStart('\n', '\r', ' ')
                    museBuffer.setLength(0)
                    museBuffer.append(afterMsg)
                    if (museBuffer.isNotEmpty()) emitTokenForMuseGlimmer("", thinkingEnabled, send)
                    return
                }

                val eotIdx = buf.indexOf("<|eot|>")
                val endOfTextIdx = buf.indexOf("<|end_of_text|>")
                val stopIdx = when {
                    eotIdx >= 0 && endOfTextIdx >= 0 -> minOf(eotIdx, endOfTextIdx)
                    eotIdx >= 0 -> eotIdx
                    endOfTextIdx >= 0 -> endOfTextIdx
                    else -> -1
                }

                if (stopIdx >= 0) {
                    val finalChunk = buf.substring(0, stopIdx)
                    if (finalChunk.isNotEmpty()) send(finalChunk)
                    museBuffer.setLength(0)
                    return
                }

                val safeLen = (buf.length - 15).coerceAtLeast(0)
                if (safeLen > 0) {
                    send(buf.substring(0, safeLen))
                    museBuffer.delete(0, safeLen)
                }
            }
        }
    }

    private fun stripLeadingMuseUserHeader(text: String): String? {
        val trimmed = text.trimStart('\n', '\r', ' ')
        if (trimmed.startsWith("<|message|>")) {
            return trimmed.substring("<|message|>".length).trimStart('\n', '\r', ' ')
        }
        val fullHeaders = listOf(
            "<|start|>assistant to=user<|message|>",
            "<|start|>assistant to=user\n<|message|>",
            "<|start|>assistant\nto=user<|message|>",
            "<|start|>assistant\nto=user\n<|message|>",
            "assistant to=user<|message|>",
            "assistant to=user\n<|message|>",
            "to=user<|message|>",
            "to=user\n<|message|>"
        )
        for (h in fullHeaders) {
            if (trimmed.startsWith(h)) {
                return trimmed.substring(h.length).trimStart('\n', '\r', ' ')
            }
        }
        val toUserIdx = trimmed.indexOf("to=user")
        if (toUserIdx in 0..25) {
            val msgIdx = trimmed.indexOf("<|message|>", toUserIdx)
            if (msgIdx >= 0 && msgIdx - toUserIdx < 30) {
                return trimmed.substring(msgIdx + "<|message|>".length).trimStart('\n', '\r', ' ')
            }
        }
        return null
    }

    private fun stripLeadingMuseSelfHeader(text: String): String? {
        val trimmed = text.trimStart('\n', '\r', ' ')
        val fullHeaders = listOf(
            "<|start|>assistant to=self<|message|>",
            "<|start|>assistant to=self\n<|message|>",
            "<|start|>assistant\nto=self<|message|>",
            "<|start|>assistant\nto=self\n<|message|>",
            "assistant to=self<|message|>",
            "assistant to=self\n<|message|>",
            "to=self<|message|>",
            "to=self\n<|message|>"
        )
        for (h in fullHeaders) {
            if (trimmed.startsWith(h)) {
                return trimmed.substring(h.length).trimStart('\n', '\r', ' ')
            }
        }
        val toSelfIdx = trimmed.indexOf("to=self")
        if (toSelfIdx in 0..25) {
            val msgIdx = trimmed.indexOf("<|message|>", toSelfIdx)
            if (msgIdx >= 0 && msgIdx - toSelfIdx < 30) {
                return trimmed.substring(msgIdx + "<|message|>".length).trimStart('\n', '\r', ' ')
            }
        }
        return null
    }

    private fun isPotentialMuseUserHeaderPrefix(text: String): Boolean {
        val trimmed = text.trimStart('\n', '\r', ' ')
        if (trimmed.isEmpty()) return true
        val candidatePrefixes = listOf(
            "<|start|>assistant to=user<|message|>",
            "assistant to=user<|message|>",
            "to=user<|message|>",
            "<|message|>"
        )
        return candidatePrefixes.any { it.startsWith(trimmed) }
    }

    private fun isPotentialMuseHeaderPrefix(text: String): Boolean {
        val trimmed = text.trimStart('\n', '\r', ' ')
        if (trimmed.isEmpty()) return true
        val candidatePrefixes = listOf(
            "<|start|>assistant to=self<|message|>",
            "<|start|>assistant to=user<|message|>",
            "assistant to=self<|message|>",
            "assistant to=user<|message|>",
            "to=self<|message|>",
            "to=user<|message|>",
            "<|message|>"
        )
        return candidatePrefixes.any { it.startsWith(trimmed) }
    }

    private fun findMuseUserHeaderIndex(text: String): Int {
        val fullHeaders = listOf(
            "<|start|>assistant to=user<|message|>",
            "assistant to=user<|message|>",
            "to=user<|message|>",
            "<|start|>assistant to=user",
            "assistant to=user",
            "to=user"
        )
        for (h in fullHeaders) {
            val idx = text.indexOf(h)
            if (idx >= 0) return idx
        }
        return -1
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
            // VibeCoder and other feature prompts often contain a large instruction block plus a
            // dedicated USER REQUEST field. Split those into system+user roles so the model
            // executes the request instead of echoing meta-instructions.
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
                val userContent = cleanPrompt.substring(idx + sep.length).trim()
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

        val isMuseGlimmer = model.name.contains("muse glimmer", ignoreCase = true) ||
                            model.name.contains("muse-glimmer", ignoreCase = true)
        if (isMuseGlimmer) {
            return buildMuseGlimmerPrompt(messages, cleanPrompt, thinkingEnabled)
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
                                                      model.name.contains("Reasoning", ignoreCase = true) ||
                                                      model.name.contains("LFM2.5-8B-A1B", ignoreCase = true)
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
                                  model.name.contains("Reasoning", ignoreCase = true) ||
                                  model.name.contains("LFM2.5-8B-A1B", ignoreCase = true)

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

    private fun buildMuseGlimmerPrompt(
        messages: List<ChatMessage>,
        cleanPrompt: String,
        thinkingEnabled: Boolean
    ): String {
        var systemContent = ""
        val historyTurns = mutableListOf<Pair<String, String>>()

        for (msg in messages) {
            val role = try { msg::class.java.getDeclaredField("role").apply { isAccessible = true }.get(msg) as String } catch (e: Exception) { "user" }
            val content = try { msg::class.java.getDeclaredField("content").apply { isAccessible = true }.get(msg) as String } catch (e: Exception) { "" }
            if (role == "system") {
                systemContent = if (systemContent.isEmpty()) content else "$systemContent\n\n$content"
            } else {
                historyTurns.add(role to content)
            }
        }

        val effectiveSystem = if (systemContent.isNotBlank()) systemContent.trim() else "You are a helpful AI assistant."
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        val currentDate = sdf.format(Date())
        val validRecipients = if (thinkingEnabled) "\"self\", \"user\"" else "\"user\""

        val sb = StringBuilder()
        sb.append("<|start|>system<|message|>$effectiveSystem\nKnowledge cutoff: 2026-01-04.\nCurrent date: $currentDate.\n\nReasoning strength: high.\n\n# Valid recipients: $validRecipients.<|eot|>")

        if (historyTurns.isEmpty()) {
            val userText = cleanPrompt.trim()
            sb.append("<|start|>user<|message|>$userText<|eot|>")
        } else {
            for ((role, content) in historyTurns) {
                val trimmed = content.trim()
                if (trimmed.isEmpty()) continue
                if (role == "user") {
                    sb.append("<|start|>user<|message|>$trimmed<|eot|>")
                } else {
                    sb.append("<|start|>assistant to=user<|message|>$trimmed<|eot|>")
                }
            }
        }

        if (thinkingEnabled) {
            sb.append("<|start|>assistant")
        } else {
            sb.append("<|start|>assistant to=user<|message|>")
        }
        return sb.toString()
    }

    override suspend fun resetChatSession(chatId: String) {
        try {
            val modelToReload = currentModel
            val backendToUse = currentPreferredBackend

            val deviceToUse = currentDeviceId

            if (isVlmLoaded && vlmWrapper != null) {
                // VLM always needs a full reload to reset vision encoder state
                Log.d(TAG, "VLM: Destroying wrapper to clear vision state for new chat")
                vlmWrapper?.stopStream()
                vlmWrapper?.destroy()
                vlmWrapper = null
                if (modelToReload != null) {
                    Log.d(TAG, "VLM: Reloading model ${modelToReload.name} for fresh state (visionDisabled=$currentVisionDisabled)")
                    loadModelInternal(modelToReload, backendToUse, currentVisionDisabled, deviceToUse)
                }
            } else if (llmWrapper != null) {
                llmWrapper?.stopStream()
                if (hasGeneratedTokensSinceLoad && modelToReload != null) {
                    Log.d(TAG, "LLM: Destroying wrapper to clear KV cache for chat $chatId")
                    llmWrapper?.destroy()
                    llmWrapper = null
                    hasGeneratedTokensSinceLoad = false
                    Log.d(TAG, "LLM: Reloading model ${modelToReload.name} for fresh state")
                    loadModelInternal(modelToReload, backendToUse, currentVisionDisabled, deviceToUse)
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
    override fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?, nGpuLayers: Int?, enableThinking: Boolean?, contextWindow: Int?) {
        overrideMaxTokens = maxTokens
        overrideContextWindow = contextWindow
        overrideTopK = topK
        overrideTopP = topP
        overrideTemperature = temperature
        overrideNGpuLayers = nGpuLayers
        overrideEnableThinking = enableThinking
    }
    override fun isVisionCurrentlyDisabled(): Boolean = currentVisionDisabled
    override fun isAudioCurrentlyDisabled(): Boolean = currentAudioDisabled
    override fun isGpuBackendEnabled(): Boolean = currentPreferredBackend == LlmInference.Backend.GPU
    override fun isNpuBackendEnabled(): Boolean = currentIsNpu
    override fun getLastDecodeSpeedTokPerSec(): Double? = lastDecodeSpeedTokPerSec
    override fun getEffectiveMaxTokens(model: LLMModel): Int = overrideContextWindow ?: overrideMaxTokens ?: model.contextWindowSize
}
