package com.llmhub.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.isActive
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet
import javax.inject.Inject
import ai.onnxruntime.providers.NNAPIFlags
import javax.inject.Singleton
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.llmhub.llmhub.websearch.WebSearchService
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
import com.llmhub.llmhub.R

@Singleton
class OnnxInferenceService @Inject constructor(
    private val context: Context
) : InferenceService {

    private val TAG = "OnnxInferenceService"
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    /** Separate embedding session for decoder-only ONNX that expects inputs_embeds (e.g. Ministral-3). */
    private var ortEmbedSession: OrtSession? = null
    /** Vision encoder session for Ministral-3 ONNX (image → features). */
    private var ortVisionSession: OrtSession? = null
    /** Image token id for vision (from config.json text_config.image_token_index). */
    private var imageTokenIndex: Long? = null
    private var currentModel: LLMModel? = null
    private var tokenizer: OnnxTokenizer? = null
    private var currentPreferredBackend: LlmInference.Backend? = null
    
    private var overrideMaxTokens: Int? = null
    private var overrideTopK: Int? = 40
    private var overrideTopP: Float? = 0.95f
    private var overrideTemperature: Float? = 0.7f
    
    // Web search service for enhanced responses
    private val webSearchService: WebSearchService = DuckDuckGoSearchService()

    @Volatile
    private var shouldStop = false
    private val mutex = Mutex()
    // Buffers streamed output for LFM Thinking to detect think tags
    private val thinkingBuffer = StringBuilder()
    private var lastEmittedThinkingLength = 0
    private var thinkSentinelEmitted = false
    /** After we see </think> we stream answer tokens as plain text (no THINK sentinel). */
    private var inAnswerPhase = false
    private val SENTINEL_THINK = "\u200B\u200BTHINK\u200B\u200B"
    private val SENTINEL_ENDTHINK = "\u200B\u200BENDTHINK\u200B\u200B"

    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?): Boolean {
        return loadModelInternal(model, preferredBackend)
    }

    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?, disableVision: Boolean, disableAudio: Boolean): Boolean {
        return loadModelInternal(model, preferredBackend)
    }

    private suspend fun loadModelInternal(model: LLMModel, preferredBackend: LlmInference.Backend?): Boolean {
        mutex.withLock {
            if (currentModel?.name == model.name && ortSession != null &&
                (preferredBackend == null || preferredBackend == currentPreferredBackend)) {
                return true
            }
        }
        
        unloadModel()

        return try {
            val modelDir = getModelDirectory(model)
            val modelFile = findOnnxModelFile(modelDir, model)
            
            if (!modelFile.exists()) {
                Log.e(TAG, "Model file not found: ${modelFile.absolutePath}")
                return false
            }

            val useGpu = preferredBackend == LlmInference.Backend.GPU
            Log.d(TAG, "Loading ONNX model from: ${modelFile.absolutePath} (backend: ${preferredBackend?.name ?: "default"}, useNnapi: $useGpu)")
            
            withContext(Dispatchers.IO) {
                mutex.withLock {
                    ortEnvironment = OrtEnvironment.getEnvironment()
                    
                    if (useGpu) {
                        // Try NNAPI-only first (no CPU fallback) so full graph runs on GPU/NPU if driver supports it.
                        try {
                            val nnapiOnlyOptions = OrtSession.SessionOptions().apply {
                                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                                try { addConfigEntry("ep.nnapi.partitioning_stop_ops", "") } catch (_: Exception) { }
                                addConfigEntry("session.disable_cpu_ep_fallback", "1")
                                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
                            }
                            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, nnapiOnlyOptions)
                            Log.i(TAG, "ONNX session created with NNAPI only (full graph on device GPU/NPU)")
                        } catch (e: Exception) {
                            Log.w(TAG, "NNAPI-only failed (model has ops NNAPI cannot run on this device), using NNAPI+CPU: $e")
                            val sessionOptions = OrtSession.SessionOptions().apply {
                                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                                try { addConfigEntry("ep.nnapi.partitioning_stop_ops", "") } catch (_: Exception) { }
                                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
                                addCPU(true)
                            }
                            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)
                            Log.i(TAG, "ONNX session created with NNAPI (GPU/accelerator) + CPU fallback — ops supported by NNAPI run on device accelerator; rest on CPU")
                        }
                    } else {
                        val sessionOptions = OrtSession.SessionOptions().apply {
                            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                        }
                        ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)
                        Log.d(TAG, "ONNX session created (CPU only — GPU not selected)")
                    }
                    tokenizer = loadTokenizer(modelDir)
                    currentModel = model
                    currentPreferredBackend = preferredBackend
                    
                    // If decoder expects inputs_embeds (e.g. Ministral-3 ONNX), load embed_tokens per model docs
                    val session = ortSession!!
                    if ("inputs_embeds" in session.inputNames && "input_ids" !in session.inputNames) {
                        val embedFile = File(modelDir, "embed_tokens_fp16.onnx")
                        if (embedFile.exists()) {
                            val embedOptions = if (useGpu) {
                                try {
                                    OrtSession.SessionOptions().apply {
                                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                                        setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                                        try { addConfigEntry("ep.nnapi.partitioning_stop_ops", "") } catch (_: Exception) { }
                                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
                                        addCPU(true)
                                    }
                                } catch (_: Exception) {
                                    OrtSession.SessionOptions().apply {
                                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                                        setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                                    }
                                }
                            } else {
                                OrtSession.SessionOptions().apply {
                                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                                    setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                                }
                            }
                            ortEmbedSession = ortEnvironment?.createSession(embedFile.absolutePath, embedOptions)
                            Log.d(TAG, "Loaded embedding session: ${embedFile.name} (NNAPI: $useGpu)")
                        } else {
                            Log.w(TAG, "Decoder expects inputs_embeds but embed_tokens_fp16.onnx not found in ${modelDir.absolutePath}")
                        }
                    } else {
                        ortEmbedSession = null
                    }
                    
                    // Ministral-3 vision: load vision encoder and config (image_token_index)
                    ortVisionSession = null
                    imageTokenIndex = null
                    if (model.name.contains("Ministral", ignoreCase = true) || model.name.contains("Mistral", ignoreCase = true)) {
                        val visionFile = File(modelDir, "vision_encoder_q4.onnx")
                        if (visionFile.exists()) {
                            try {
                                val visionOptions = OrtSession.SessionOptions().apply {
                                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                                    setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                                }
                                ortVisionSession = ortEnvironment?.createSession(visionFile.absolutePath, visionOptions)
                                Log.d(TAG, "Loaded vision encoder: ${visionFile.name}")
                                val configFile = File(modelDir, "config.json")
                                if (configFile.exists()) {
                                    val configJson = configFile.readText()
                                    val config = Gson().fromJson<Map<String, Any>>(configJson, object : TypeToken<Map<String, Any>>() {}.type)
                                    val textConfig = config["text_config"] as? Map<*, *>
                                    val idx = (textConfig?.get("image_token_index") as? Number)?.toLong()
                                        ?: (config["image_token_index"] as? Number)?.toLong()
                                    if (idx != null) {
                                        imageTokenIndex = idx
                                        Log.d(TAG, "Image token index: $idx")
                                    } else Log.w(TAG, "config.json has no image_token_index")
                                } else Log.w(TAG, "config.json not found for vision")
                            } catch (e: Exception) {
                                Log.w(TAG, "Failed to load vision encoder: ${e.message}")
                            }
                        }
                    }
                    
                    Log.d(TAG, "ONNX model loaded successfully")
                    Log.d(TAG, "Input names: ${ortSession?.inputNames}")
                    Log.d(TAG, "Output names: ${ortSession?.outputNames}")
                }
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX model", e)
            false
        }
    }

    private fun getModelDirectory(model: LLMModel): File {
        val modelsDir = File(context.filesDir, "models")
        // Use same naming convention as downloadOnnxModel
        val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        val modelDir = File(modelsDir, modelDirName)
        return if (modelDir.exists()) modelDir else modelsDir
    }

    private fun findOnnxModelFile(modelDir: File, model: LLMModel): File {
        // Extract clean filename from URL (strip query params)
        val localName = model.url.substringAfterLast("/").substringBefore("?")
        
        var modelFile = File(modelDir, localName)
        if (modelFile.exists()) return modelFile
        
        val modelsDir = File(context.filesDir, "models")
        modelFile = File(modelsDir, localName)
        if (modelFile.exists()) return modelFile
        
        // Fallback: find any .onnx file in the directory
        val onnxFiles = modelDir.listFiles { _, name -> name.endsWith(".onnx") }
        if (onnxFiles?.isNotEmpty() == true) return onnxFiles.first()
        
        return File(modelDir, localName)
    }

    private fun loadTokenizer(modelDir: File): OnnxTokenizer? {
        val locations = listOf(
            File(modelDir, "tokenizer.json"),
            File(File(context.filesDir, "models"), "tokenizer.json"),
            modelDir.parentFile?.let { File(it, "tokenizer.json") }
        ).filterNotNull()
        
        for (tokenizerFile in locations) {
            if (tokenizerFile.exists()) {
                return try {
                    OnnxTokenizer(tokenizerFile)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to load tokenizer: ${e.message}")
                    null
                }
            }
        }
        Log.w(TAG, "No tokenizer.json found")
        return null
    }

    override suspend fun unloadModel() {
        mutex.withLock {
            withContext(Dispatchers.IO) {
                try {
                    shouldStop = true
                    ortEmbedSession?.close()
                } catch (e: Exception) {
                    Log.w(TAG, "Error releasing embed session: ${e.message}")
                }
                ortEmbedSession = null
                try {
                    ortVisionSession?.close()
                } catch (e: Exception) {
                    Log.w(TAG, "Error releasing vision session: ${e.message}")
                }
                ortVisionSession = null
                imageTokenIndex = null
                try {
                    ortSession?.close()
                } catch (e: Exception) {
                    Log.w(TAG, "Error releasing ONNX session: ${e.message}")
                }
                ortSession = null
                tokenizer = null
                currentModel = null
                currentPreferredBackend = null
                shouldStop = false
            }
        }
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        ensureModelLoaded(model)
        val result = StringBuilder()
        generateResponseStream(prompt, model).collect { result.append(it) }
        return result.toString()
    }

    override suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String> = callbackFlow {
        ensureModelLoaded(model)
        shouldStop = false
        thinkingBuffer.setLength(0)
        lastEmittedThinkingLength = 0
        thinkSentinelEmitted = false
        inAnswerPhase = false

        val backend = currentPreferredBackend
        Log.d(TAG, "Starting streaming generation... (backend: ${backend?.name ?: "default"} — ${if (backend == LlmInference.Backend.GPU) "NNAPI accelerator + CPU fallback" else "CPU only"})")

        val generatorJob = launch(Dispatchers.IO) {
            try {
                val session = ortSession ?: throw Exception("ONNX session not initialized")
                val env = ortEnvironment ?: throw Exception("ONNX environment not initialized")
                
                // Chat template: Ministral/Mistral use [INST]...[/INST]; LFM and others use ChatML
                // For Ministral we must put ONLY the user message in [INST], not "user: X\nassistant:" — otherwise
                // the model sees "X" and "assistant" adjacent and interprets it as "Xassistant".
                val formattedPrompt = if (model.name.contains("Ministral", ignoreCase = true) || model.name.contains("Mistral", ignoreCase = true)) {
                    val lastUser = prompt.substringAfterLast("user: ").substringBefore("\nassistant:").trim()
                    val instBody = if (lastUser.isNotEmpty()) lastUser else prompt.trim()
                    "[INST]\n$instBody\n[/INST]\n"
                } else {
                    // ChatML formatting. If prompt uses app's internal transcript format, convert to proper ChatML turns.
                    // This prevents models from getting confused by "user:"/"assistant:" markers inside a single user turn.
                    if (prompt.contains("user: ") && (prompt.contains("\nassistant:") || prompt.endsWith("assistant:"))) {
                        var p = prompt
                        // 1. Start interaction (First User Turn)
                        if (p.startsWith("user: ")) {
                            p = p.replaceFirst("user: ", "<|im_start|>user\n")
                        }
                        // 2. Middle Interactions (User Turn starts after Assistant Turn ends)
                        // The \n\n separator from ChatViewModel indicates the end of the previous assistant response.
                        // We close the previous Assistant turn and open the new User turn.
                        p = p.replace("\n\nuser: ", "<|im_end|>\n<|im_start|>user\n")
                        
                        // 3. Assistant Turns (and closing of User Turns)
                        // Matches "\nassistant: " (with space) or "\nassistant:" (no space/end of prompt)
                        // We close the current User turn and open the new Assistant turn.
                        p = p.replace(Regex("\nassistant: ?"), "<|im_end|>\n<|im_start|>assistant\n")
                        
                        "<|startoftext|>" + p
                    } else {
                        // Fallback: Wrap entire prompt in a single user turn (old behavior)
                        "<|startoftext|><|im_start|>user\n$prompt<|im_end|>\n<|im_start|>assistant\n"
                    }
                }
                
                // CRITICAL: Check if tokenizer loaded. If not, ONNX models cannot work properly
                val currentTokenizer = tokenizer
                if (currentTokenizer == null) {
                    val error = "Tokenizer failed to load! ONNX model requires tokenizer.json in model directory. " +
                            "Current model dir: ${getModelDirectory(model).absolutePath}. " +
                            "Please ensure tokenizer.json and tokenizer_config.json were downloaded."
                    Log.e(TAG, error)
                    throw Exception(error)
                }
                
                val inputIds = currentTokenizer.encode(formattedPrompt)
                Log.d(TAG, "Tokenized prompt: ${inputIds.size} tokens")
                
                val maxNewTokens = overrideMaxTokens ?: 1024
                val eosTokenId = currentTokenizer.getEosTokenId()
                
                // Initialize KV cache - create zero tensors for all past_* inputs
                val cache = mutableMapOf<String, OnnxTensor>()
                val inputNames = session.inputNames.toSet()
                val usePositionIds = "position_ids" in inputNames
                
                // Build cache for all past_* inputs (Python: for inp in session.get_inputs())
                for ((inputName, nodeInfo) in session.inputInfo) {
                    // Skip standard inputs
                    if (inputName in setOf("input_ids", "attention_mask", "position_ids")) {
                        continue
                    }
                    
                    // STRICT FILTER: Only treat inputs starting with "past_" as KV cache
                    // This avoids initializing optional inputs that have default values (initializers)
                    if (!inputName.startsWith("past_")) {
                        Log.w(TAG, "Skipping non-standard input: $inputName")
                        continue
                    }
                    
                    // Get shape from node info
                    val tensorInfo = nodeInfo.info as ai.onnxruntime.TensorInfo
                    val inputShape = tensorInfo.shape  // long[]
                    
                    // Build shape: first dim is batch (1), symbolic dims become 0 or keep fixed values
                    // Python: shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                    //         then set sequence dims to 0
                    val onnxShape = inputShape.mapIndexed { index, dim ->
                        when {
                            index == 0 -> 1L  // First dimension is always batch_size = 1
                            dim > 0 -> dim     // Fixed dimension (e.g., num_heads, head_dim)
                            else -> 0L         // Symbolic dimension (sequence length) starts at 0
                        }
                    }.toLongArray()
                    
                    // Create zero float array based on shape
                    // If ANY dimension is 0, total size is 0
                    val totalSize = if (onnxShape.contains(0L)) {
                        0
                    } else {
                        onnxShape.fold(1L) { acc, d -> acc * d }.toInt()
                    }
                    val zeroData = FloatArray(totalSize) { 0f }
                    
                    // Create zero tensor  
                    val zeroTensor = OnnxTensor.createTensor(env, 
                        java.nio.FloatBuffer.wrap(zeroData), onnxShape)
                    cache[inputName] = zeroTensor
                }
                
                Log.d(TAG, "Initialized ${cache.size} cache tensors")
                
                var seqLen = inputIds.size
                val generatedTokens = mutableListOf<Long>()
                var step = 0
                
                while (generatedTokens.size < maxNewTokens && !shouldStop && isActive) {
                    // Prepare input IDs
                    val ids = if (step == 0) {
                        inputIds.toLongArray()
                    } else {
                        longArrayOf(generatedTokens.last())
                    }
                    
                    // Prepare position IDs if needed
                    val posIds = if (usePositionIds) {
                        if (step == 0) {
                            LongArray(seqLen) { it.toLong() }
                        } else {
                            longArrayOf((seqLen + generatedTokens.size - 1).toLong())
                        }
                    } else null
                    
                    // Prepare attention mask
                    val attnMask = LongArray(seqLen + generatedTokens.size) { 1L }
                    
                    // Create input tensors
                    val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong()))
                    val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attnMask), longArrayOf(1, attnMask.size.toLong()))
                    
                    // Build feed dict
                    val feed = mutableMapOf<String, OnnxTensor>()
                    
                    // Handle input_ids vs inputs_embeds (Ministral-3: embed_tokens_fp16.onnx -> decoder)
                    var embedTensorToClose: OnnxTensor? = null
                    if ("input_ids" in inputNames) {
                        feed["input_ids"] = idsTensor
                    } else if ("inputs_embeds" in inputNames) {
                        val embedSession = ortEmbedSession
                        if (embedSession != null) {
                            val embedOutputs = embedSession.run(mapOf("input_ids" to idsTensor))
                            val embeds = embedOutputs[0] as OnnxTensor
                            embedTensorToClose = embeds
                            feed["inputs_embeds"] = embeds
                        } else {
                            throw IllegalStateException("Decoder expects inputs_embeds but embedding session not loaded. Ensure embed_tokens_fp16.onnx is in the model directory.")
                        }
                    }
                    
                    feed["attention_mask"] = maskTensor
                    
                    if (usePositionIds && posIds != null) {
                        val posTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(posIds), longArrayOf(1, posIds.size.toLong()))
                        feed["position_ids"] = posTensor
                    }
                    
                    // Add cache to feed
                    feed.putAll(cache)
                    
                    // Run inference
                    val outputs = session.run(feed)
                    
                    // Release tensors we created for this step (embed output; idsTensor is not in feed when using inputs_embeds)
                    embedTensorToClose?.close()
                    if ("inputs_embeds" in inputNames) idsTensor.close()
                    
                    // Use iterator to safely traverse outputs
                    val outputIterator = outputs.iterator()
                    
                    // Get logits (first output)
                    if (!outputIterator.hasNext()) throw Exception("No outputs from model")
                    val logitsEntry = outputIterator.next()
                    val logitsTensor = logitsEntry.value as OnnxTensor
                    val logits = logitsTensor.floatBuffer
                    
                    // Sample next token from last position  
                    val vocabSize = logits.remaining() / ids.size
                    val lastTokenLogits = FloatArray(vocabSize)
                    logits.position((ids.size - 1) * vocabSize)
                    logits.get(lastTokenLogits)
                    
                    val nextToken = sampleToken(lastTokenLogits, overrideTemperature ?: 0.7f, 
                        overrideTopK ?: 40, (overrideTopP ?: 0.95f).toDouble())
                    
                    generatedTokens.add(nextToken)
                    
                    // Updates cache from remaining outputs (present_*)
                    // Clear old cache tensors first
                    cache.values.forEach { it.close() }
                    cache.clear()
                    
                    while (outputIterator.hasNext()) {
                        val entry = outputIterator.next()
                        val outputName = entry.key
                        val outputValue = entry.value as OnnxTensor // Explicit cast on value
                        
                        // Convert present_conv -> past_conv, present.X.key -> past_key_values.X.key
                        val cacheName = outputName
                            .replace("present_conv", "past_conv")
                            .replace("present.", "past_key_values.")
                        
                        if (cacheName in session.inputNames) {
                            // Keep this tensor for next step (don't close it yet)
                            cache[cacheName] = outputValue
                        } else {
                            // Close unused outputs to prevent leaks
                            outputValue.close()
                        }
                    }
                    
                    // We must NOT close 'outputs' collection yet as it closes all tensors inside it.
                    // But we DO need to close the logits tensor as we are done with it.
                    // The cache tensors are preserved in 'cache' map for next iteration.
                    // Note: 'outputs' itself holds references. We rely on the fact that we extracted specific tensors.
                    // Ideally we should close 'outputs' but detach tensors? 
                    // ONNX Runtime Java doesn't easily support detach. Use caution.
                    // Current strategy: We keep references in 'cache'. 'outputs' object will be GC'd.
                    // However, 'outputs' might be AutoCloseable. If we don't close it, it might trigger finalizer warnings.
                    // But if we close it, it closes the tensors we want to keep.
                    // So we relying on cache holding the tensors open? No, OnnxValue is AutoCloseable.
                    // If 'outputs' is closed, it iterates and closes children.
                    // WE CANNOT CLOSE outputs.
                    
                    // Close logits specifically as we consumed it
                    // (Actually we can't easily close just one if we don't own it standalone? 
                    // OnnxTensor.close() works.)
                    // logitsTensor.close() // Safe to close logits as we read the data
                    
                    // Check for EOS before emitting (do not show EOS/special token to user)
                    if (nextToken == eosTokenId) {
                        Log.d(TAG, "EOS token generated, stopping")
                        logitsTensor.close()
                        if ("input_ids" in inputNames) idsTensor.close()
                        maskTensor.close()
                        if (usePositionIds && posIds != null) feed["position_ids"]?.close()
                        break
                    }
                    
                    // Decode and emit token (skip known special tokens so they never appear in chat)
                    val tokenText = tokenizer?.decode(listOf(nextToken.toInt()))
                        ?: decodeWithFallback(nextToken.toInt())
                    val trimmed = tokenText.trim()
                    val isSpecial = trimmed.isEmpty() ||
                        trimmed == "</s>" || trimmed == "<|im_end|>" || trimmed == "<|endoftext|>" ||
                        trimmed == "<|startoftext|>" || (trimmed.startsWith("<|") && trimmed.endsWith("|>")) ||
                        trimmed == "[INST]" || trimmed == "[/INST]" ||
                        trimmed.contains("\"prompt\"") || trimmed.contains("\"max_tokens\"") ||
                        trimmed.contains("lendoftext") || trimmed.contains("lim_end")
                    if (tokenText.isNotEmpty() && !isSpecial) {
                        emitTokenForThinking(model, tokenText, this@callbackFlow::trySend)
                    }
                    
                    // Clean up input tensors (idsTensor already closed when using inputs_embeds path)
                    if ("input_ids" in inputNames) idsTensor.close()
                    maskTensor.close()
                    if (usePositionIds && posIds != null) {
                        feed["position_ids"]?.close()
                    }
                    logitsTensor.close()
                    
                    step++
                }
                
                // Clean up cache tensors
                cache.values.forEach { it.close() }
                
                Log.d(TAG, "Generation complete. Generated ${generatedTokens.size} tokens")
                close()
            } catch (e: Exception) {
                Log.e(TAG, "Error during ONNX generation", e)
                close(e)
            }
        }
        
        awaitClose {
            shouldStop = true
            generatorJob.cancel()
        }
    }.flowOn(Dispatchers.IO)

    private fun sampleToken(logits: FloatArray, temperature: Float, topK: Int, topP: Double): Long {
        if (temperature <= 0) return logits.indices.maxByOrNull { logits[it] }?.toLong() ?: 0L
        
        val scaledLogits = logits.map { it / temperature }.toFloatArray()
        val maxLogit = scaledLogits.maxOrNull() ?: 0f
        val expLogits = scaledLogits.map { kotlin.math.exp((it - maxLogit).toDouble()) }
        val sumExp = expLogits.sum()
        val probs = expLogits.map { (it / sumExp).toFloat() }
        
        val sortedIndices = probs.indices.sortedByDescending { probs[it] }
        val topKIndices = sortedIndices.take(topK)
        
        var cumProb = 0.0
        val nucleusIndices = mutableListOf<Int>()
        for (idx in topKIndices) {
            cumProb += probs[idx]
            nucleusIndices.add(idx)
            if (cumProb >= topP) break
        }
        
        val nucleusProbs = nucleusIndices.map { probs[it] }
        val probSum = nucleusProbs.sum()
        val normalizedProbs = nucleusProbs.map { it / probSum }
        
        val rand = Math.random()
        var cumSum = 0.0
        for ((i, idx) in nucleusIndices.withIndex()) {
            cumSum += normalizedProbs[i]
            if (rand < cumSum) return idx.toLong()
        }
        return nucleusIndices.lastOrNull()?.toLong() ?: 0L
    }

    private fun encodeWithFallback(text: String): List<Long> {
        Log.w(TAG, "Using fallback tokenizer")
        return text.toCharArray().map { it.code.toLong() }
    }

    private fun decodeWithFallback(tokenId: Int): String {
        return if (tokenId in 32..126) tokenId.toChar().toString() else ""
    }

    /** For LFM Thinking: stream all content; use </think> as boundary. Everything before it = thinking (with THINK sentinel), after = answer (with ENDTHINK). */
    private fun emitTokenForThinking(model: LLMModel, tokenText: String, send: (String) -> Unit) {
        if (tokenText.isEmpty()) return
        val isThinkingModel = model.name.contains("Thinking", ignoreCase = true)
        if (!isThinkingModel || inAnswerPhase) {
            send(tokenText)
            return
        }
        thinkingBuffer.append(tokenText)
        val s = thinkingBuffer.toString()
        val endTag = "<" + "/think>"
        if (s.contains(endTag)) {
            val thinkingPart = s.substringBefore(endTag)
            val after = s.substringAfter(endTag)
            if (!thinkSentinelEmitted && thinkingPart.isNotEmpty()) {
                send(SENTINEL_THINK + thinkingPart)
                thinkSentinelEmitted = true
            } else if (thinkSentinelEmitted && thinkingPart.length > lastEmittedThinkingLength) {
                send(thinkingPart.drop(lastEmittedThinkingLength))
            }
            send(SENTINEL_ENDTHINK + after)
            thinkingBuffer.setLength(0)
            thinkingBuffer.append(after)
            lastEmittedThinkingLength = 0
            thinkSentinelEmitted = false
            inAnswerPhase = true
            return
        }
        // No </think> yet: stream as thinking (do not require <think> to be present)
        if (!thinkSentinelEmitted) {
            send(SENTINEL_THINK + s)
            thinkSentinelEmitted = true
        } else {
            val newPart = s.drop(lastEmittedThinkingLength)
            if (newPart.isNotEmpty()) send(newPart)
        }
        lastEmittedThinkingLength = s.length
    }

    override suspend fun generateResponseStreamWithSession(
        prompt: String, model: LLMModel, chatId: String,
        images: List<Bitmap>, audioData: ByteArray?, webSearchEnabled: Boolean
    ): Flow<String> = callbackFlow {
        ensureModelLoaded(model)
        
        // Extract the current user message from the prompt for web search detection
        val currentUserMessage = extractCurrentUserMessage(prompt)
        val needsWebSearch = webSearchEnabled && SearchIntentDetector.needsWebSearch(currentUserMessage)
        var enhancedPrompt = prompt
        
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
                    
                    // Create enhanced prompt with search results
                    val resultsText = searchResults.joinToString("\n\n") { result ->
                        "SOURCE: ${result.source}\nTITLE: ${result.title}\nCONTENT: ${result.snippet}\n---"
                    }
                    
                    // Extract just the current user question for better clarity
                    enhancedPrompt = """
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
                    // Continue with original prompt
                }
            } catch (searchException: Exception) {
                Log.e(TAG, "Web search failed for chat $chatId", searchException)
                trySend(context.getString(R.string.web_search_failed, searchException.message ?: "Unknown error") + "\n\n")
                // Continue with original prompt
            }
        }
        
        // Determine if we should use vision or standard text generation
        // Vision pipeline is used if:
        // 1. Any images are provided
        // 2. Vision session is loaded (means model supports vision and it initialized correctly)
        val useVision = images.isNotEmpty() && ortVisionSession != null
        
        val generatorJob = launch(Dispatchers.IO) {
            try {
                if (useVision) {
                    runGenerationWithVision(enhancedPrompt, model, images, this@callbackFlow::trySend)
                } else {
                    // Collect form the standard flow logic which yields tokens
                    generateResponseStream(enhancedPrompt, model).collect { token ->
                        trySend(token)
                    }
                }
                close()
            } catch (e: Exception) {
                Log.e(TAG, "Generation error", e)
                close(e)
            }
        }
        
        awaitClose {
            shouldStop = true
            generatorJob.cancel()
        }
    }.flowOn(Dispatchers.IO)

    private fun extractCurrentUserMessage(prompt: String): String {
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

    /**
     * Preprocess bitmap to NCHW float [1, 3, H, W] for vision encoder. Resizes to targetH x targetW, normalizes to [0,1].
     */
    private fun bitmapToPixelValues(bitmap: Bitmap, targetH: Int, targetW: Int): FloatArray {
        val scaled = if (bitmap.width != targetW || bitmap.height != targetH) {
            Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
        } else bitmap
        val w = scaled.width
        val h = scaled.height
        val pixels = IntArray(w * h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = FloatArray(1 * 3 * h * w)
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i] = ((p shr 16) and 0xFF) / 255f
            out[h * w + i] = ((p shr 8) and 0xFF) / 255f
            out[2 * h * w + i] = (p and 0xFF) / 255f
        }
        if (scaled != bitmap) scaled.recycle()
        return out
    }

    private suspend fun runGenerationWithVision(
        prompt: String,
        model: LLMModel,
        images: List<Bitmap>,
        send: (String) -> Unit
    ) {
        thinkingBuffer.setLength(0)
        lastEmittedThinkingLength = 0
        thinkSentinelEmitted = false
        inAnswerPhase = false
        val session = ortSession ?: throw Exception("ONNX session not initialized")
        val env = ortEnvironment ?: throw Exception("ONNX environment not initialized")
        val embedSession = ortEmbedSession ?: throw Exception("Vision path requires embed session")
        val visionSession = ortVisionSession ?: throw Exception("Vision session not loaded")
        val tok = tokenizer ?: throw Exception("Tokenizer not loaded")
        val imgTokenId = imageTokenIndex ?: throw Exception("image_token_index not set")
        val modelDir = getModelDirectory(model)
        val lastUser = prompt.substringAfterLast("user: ").substringBefore("\nassistant:").trim()
        val userText = if (lastUser.isNotEmpty()) lastUser else prompt.trim()
        val prefixIds = tok.encode("[INST]\n")
        val suffixIds = tok.encode("$userText\n[/INST]\n")
        // Ministral/Pixtral vision encoder patch_merger expects exactly 26x26 patches (reshape 1,13,2,13,2,1024). patch_size=14 => input must be 364x364. 384 gives 27x27 and Reshape fails.
        val vH = 364
        val vW = 364
        Log.d(TAG, "Vision input size: ${vH}x${vW}")
        val pixelValues = bitmapToPixelValues(images.first(), vH, vW)
        val pixelTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pixelValues), longArrayOf(1L, 3L, vH.toLong(), vW.toLong()))
        val visionOut = visionSession.run(mapOf("pixel_values" to pixelTensor))
        pixelTensor.close()
        val imageFeatures = visionOut[0] as OnnxTensor
        val featShape = imageFeatures.info.shape
        val numImageTokens = if (featShape.size >= 2) {
            var n = 1L
            for (i in 1 until featShape.size - 1) n *= featShape[i]
            n.toInt()
        } else 1
        val hiddenDim = if (featShape.isNotEmpty()) featShape.last().toInt() else 2048
        visionOut.forEach { if (it !== imageFeatures) (it as? OnnxTensor)?.close() }
        val inputIds = prefixIds + List(numImageTokens) { imgTokenId } + suffixIds
        imageFeatures.use {
            val featBuf = imageFeatures.floatBuffer
            val featFloats = FloatArray(featBuf.remaining())
            featBuf.get(featFloats)
            val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds.toLongArray()), longArrayOf(1, inputIds.size.toLong()))
            val embedOut = embedSession.run(mapOf("input_ids" to idsTensor))
            idsTensor.close()
            val embeds = embedOut[0] as OnnxTensor
            val embedShape = embeds.info.shape
            val seqLen = embedShape[0].toInt() * embedShape[1].toInt()
            val embedDim = embedShape[2].toInt()
            val merged = FloatArray(seqLen * embedDim)
            embeds.floatBuffer.get(merged)
            embeds.close()
            var imageOffset = 0
            for (i in inputIds.indices) {
                if (inputIds[i] == imgTokenId) {
                    for (d in 0 until minOf(embedDim, hiddenDim)) {
                        merged[i * embedDim + d] = featFloats.getOrElse(imageOffset * hiddenDim + d) { 0f }
                    }
                    imageOffset++
                }
            }
            val mergedTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(merged), longArrayOf(1, inputIds.size.toLong(), embedDim.toLong()))
            runDecoderLoop(env, session, model, inputIds, mergedTensor, tok, send)
        }
    }

    private fun runDecoderLoop(
        env: OrtEnvironment,
        session: OrtSession,
        model: LLMModel,
        initialInputIds: List<Long>,
        initialInputsEmbeds: OnnxTensor,
        tok: OnnxTokenizer,
        send: (String) -> Unit
    ) {
        val inputNames = session.inputNames.toSet()
        val usePositionIds = "position_ids" in inputNames
        val eosTokenId = tok.getEosTokenId()
        val maxNewTokens = overrideMaxTokens ?: 1024
        val cache = mutableMapOf<String, OnnxTensor>()
        for ((inputName, nodeInfo) in session.inputInfo) {
            if (inputName in setOf("input_ids", "attention_mask", "position_ids")) continue
            if (!inputName.startsWith("past_")) continue
            val tensorInfo = nodeInfo.info as ai.onnxruntime.TensorInfo
            val inputShape = tensorInfo.shape
            val onnxShape = inputShape.mapIndexed { index, dim ->
                when {
                    index == 0 -> 1L
                    dim > 0 -> dim
                    else -> 0L
                }
            }.toLongArray()
            val totalSize = if (onnxShape.contains(0L)) 0 else onnxShape.fold(1L) { acc, d -> acc * d }.toInt()
            val zeroData = FloatArray(totalSize) { 0f }
            cache[inputName] = OnnxTensor.createTensor(env, FloatBuffer.wrap(zeroData), onnxShape)
        }
        var seqLen = initialInputIds.size
        val generatedTokens = mutableListOf<Long>()
        var step = 0
        while (generatedTokens.size < maxNewTokens && !shouldStop) {
            val ids = if (step == 0) initialInputIds.toLongArray() else longArrayOf(generatedTokens.last())
            val posIds = if (usePositionIds) {
                if (step == 0) LongArray(seqLen) { it.toLong() } else longArrayOf((seqLen + generatedTokens.size - 1).toLong())
            } else null
            val attnMask = LongArray(seqLen + generatedTokens.size) { 1L }
            val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong()))
            val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attnMask), longArrayOf(1, attnMask.size.toLong()))
            val feed = mutableMapOf<String, OnnxTensor>()
            val embedSession = ortEmbedSession
            if ("inputs_embeds" in inputNames) {
                val embeds = if (step == 0) {
                    initialInputsEmbeds
                } else {
                    val eo = embedSession!!.run(mapOf("input_ids" to idsTensor))
                    eo[0] as OnnxTensor
                }
                feed["inputs_embeds"] = embeds
            }
            feed["attention_mask"] = maskTensor
            if (usePositionIds && posIds != null) {
                feed["position_ids"] = OnnxTensor.createTensor(env, LongBuffer.wrap(posIds), longArrayOf(1, posIds.size.toLong()))
            }
            feed.putAll(cache)
            val outputs = session.run(feed)
            if (step > 0 && "inputs_embeds" in inputNames) (feed["inputs_embeds"] as? OnnxTensor)?.close()
            if ("inputs_embeds" in inputNames) idsTensor.close()
            maskTensor.close()
            if (usePositionIds && posIds != null) feed["position_ids"]?.close()
            val outputIterator = outputs.iterator()
            if (!outputIterator.hasNext()) throw Exception("No outputs from model")
            val logitsEntry = outputIterator.next()
            val logitsTensor = logitsEntry.value as OnnxTensor
            val logits = logitsTensor.floatBuffer
            val vocabSize = logits.remaining() / ids.size
            val lastTokenLogits = FloatArray(vocabSize)
            logits.position((ids.size - 1) * vocabSize)
            logits.get(lastTokenLogits)
            val nextToken = sampleToken(lastTokenLogits, overrideTemperature ?: 0.7f, overrideTopK ?: 40, (overrideTopP ?: 0.95f).toDouble())
            generatedTokens.add(nextToken)
            cache.values.forEach { it.close() }
            cache.clear()
            while (outputIterator.hasNext()) {
                val entry = outputIterator.next()
                val outputValue = entry.value as OnnxTensor
                val cacheName = entry.key.replace("present_conv", "past_conv").replace("present.", "past_key_values.")
                if (cacheName in session.inputNames) cache[cacheName] = outputValue else outputValue.close()
            }
            if (nextToken == eosTokenId) break
            val tokenText = tok.decode(listOf(nextToken.toInt()))
            val trimmed = tokenText.trim()
            val isSpecial = trimmed.isEmpty() || trimmed == "</s>" || trimmed == "<|im_end|>" || trimmed == "<|endoftext|>" ||
                trimmed == "<|startoftext|>" || (trimmed.startsWith("<|") && trimmed.endsWith("|>")) ||
                trimmed == "[INST]" || trimmed == "[/INST]" || trimmed.contains("\"prompt\"") || trimmed.contains("\"max_tokens\"") ||
                trimmed.contains("lendoftext") || trimmed.contains("lim_end")
            if (tokenText.isNotEmpty() && !isSpecial) emitTokenForThinking(model, tokenText, send)
            logitsTensor.close()
            step++
        }
        initialInputsEmbeds.close()
        cache.values.forEach { it.close() }
        Log.d(TAG, "Vision generation complete. Generated ${generatedTokens.size} tokens")
    }

    private suspend fun ensureModelLoaded(model: LLMModel) {
        if (currentModel?.name != model.name || ortSession == null) {
            if (!loadModelInternal(model, currentPreferredBackend)) throw Exception("Failed to load model ${model.name}")
        }
    }

    override suspend fun resetChatSession(chatId: String) { shouldStop = true }
    override suspend fun onCleared() { unloadModel() }
    override fun getCurrentlyLoadedModel(): LLMModel? = currentModel

    override fun getCurrentlyLoadedBackend(): LlmInference.Backend? = currentPreferredBackend
    override fun getMemoryWarningForImages(images: List<Bitmap>): String? =
        if (images.isNotEmpty() && ortVisionSession != null)
            "Vision processing may use significant memory."
        else null
    override fun wasSessionRecentlyReset(chatId: String): Boolean = false
    
    override fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?) {
        overrideMaxTokens = maxTokens
        overrideTopK = topK
        overrideTopP = topP
        overrideTemperature = temperature
    }
    
    override fun isVisionCurrentlyDisabled(): Boolean = (ortVisionSession == null)
    override fun isAudioCurrentlyDisabled(): Boolean = true
    override fun isGpuBackendEnabled(): Boolean = currentPreferredBackend == LlmInference.Backend.GPU
    override fun getEffectiveMaxTokens(model: LLMModel): Int = overrideMaxTokens ?: model.contextWindowSize
}

class OnnxTokenizer(tokenizerFile: File) {
    private val vocab: Map<String, Int>
    private val reverseVocab: Map<Int, String>
    private val eosTokenId: Int
    
    init {
        val json = tokenizerFile.readText()
        val gson = Gson()
        val tokenizerData = gson.fromJson<Map<String, Any>>(json, object : TypeToken<Map<String, Any>>() {}.type)
        
        val modelData = tokenizerData["model"] as? Map<*, *>
        val vocabData = modelData?.get("vocab") as? Map<*, *>
        
        val baseVocab = vocabData?.mapNotNull { (key, value) ->
            val k = key as? String
            val v = (value as? Number)?.toInt()
            if (k != null && v != null) k to v else null
        }?.toMap() ?: emptyMap()
        
        // Load added_tokens to ensure special tokens (like <|im_start|>) are available
        val addedTokens = tokenizerData["added_tokens"] as? List<*>
        val addedVocab = addedTokens?.filterIsInstance<Map<*, *>>()?.mapNotNull { tokenMap ->
            val content = tokenMap["content"] as? String
            val id = (tokenMap["id"] as? Number)?.toInt()
            if (content != null && id != null) content to id else null
        }?.toMap() ?: emptyMap()
        
        // added_tokens supplement the base vocab
        vocab = baseVocab + addedVocab
        
        reverseVocab = vocab.entries.associate { (k, v) -> v to k }
        
        // Try to load tokenizer_config.json for accurate EOS token
        var configEosId: Int? = null
        try {
            val configFile = File(tokenizerFile.parentFile, "tokenizer_config.json")
            if (configFile.exists()) {
                val configJson = configFile.readText()
                val config = gson.fromJson<Map<String, Any>>(configJson, object : TypeToken<Map<String, Any>>() {}.type)
                val eosAny = config["eos_token"]
                when (eosAny) {
                    is String -> configEosId = vocab[eosAny]
                    is Map<*, *> -> {
                        val id = (eosAny["id"] as? Number)?.toInt()
                        if (id != null) configEosId = id
                        else (eosAny["content"] as? String)?.let { configEosId = vocab[it] }
                    }
                    else -> { }
                }
            }
        } catch (e: Exception) {
            Log.w("OnnxTokenizer", "Failed to load tokenizer_config.json: ${e.message}")
        }
        
        // Fallback: added_tokens or common names (Mistral uses </s>, LFM uses <|im_end|>)
        if (configEosId == null) {
            val eosEntry = addedTokens?.filterIsInstance<Map<*, *>>()?.find {
                val content = it["content"] as? String ?: ""
                content == "<|im_end|>" || content == "</s>" || content.contains("eos") || content.contains("end")
            }
            configEosId = (eosEntry?.get("id") as? Number)?.toInt()
        }
        
        eosTokenId = configEosId ?: vocab["<|im_end|>"] ?: vocab["</s>"] ?: vocab["<|endoftext|>"] ?: vocab["[EOS]"] ?: vocab["eos"] ?: 2
        Log.i("OnnxTokenizer", "Resolved EOS token ID: $eosTokenId")
    }
    
    fun encode(text: String): List<Long> {
        // Pre-process: Replace spaces with BPE space marker (Ġ/U+0120)
        // This is a simplified BPE encoding that assumes space boundaries
        val processedText = text.replace(" ", "\u0120")
        
        val tokens = mutableListOf<Long>()
        var remaining = processedText
        
        while (remaining.isNotEmpty()) {
            var matched = false
            // Try to find longest matching token
            for (len in minOf(remaining.length, 60) downTo 1) {
                val candidate = remaining.substring(0, len)
                val tokenId = vocab[candidate]
                if (tokenId != null) {
                    tokens.add(tokenId.toLong())
                    remaining = remaining.substring(len)
                    matched = true
                    break
                }
            }
            if (!matched) {
                // Unknown character - use byte fallback or skip
                val char = remaining[0]
                val byteToken = vocab["<0x${char.code.toString(16).uppercase().padStart(2, '0')}>"]
                if (byteToken != null) {
                    tokens.add(byteToken.toLong())
                }
                remaining = remaining.substring(1)
            }
        }
        return tokens
    }
    
    // ByteLevel BPE character mapping (matches GPT-2/Llama implementation)
    private val charToByte: Map<Char, Byte> by lazy {
        val bs = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)).toMutableList()
        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        cs.zip(bs).associate { (c, b) -> c.toChar() to b.toByte() }
    }

    fun decode(tokenIds: List<Int>): String {
        // Collect all bytes
        val bytes = java.io.ByteArrayOutputStream()
        
        for (id in tokenIds) {
            val token = reverseVocab[id] ?: continue
            
            // Skip special tokens in output if they look like <|...|>
            if (token.startsWith("<|") && token.endsWith("|>")) {
                continue 
            }
            
            for (char in token) {
                val b = charToByte[char]
                if (b != null) {
                    bytes.write(b.toInt())
                } else {
                    // Fallback for non-mapped chars
                    // Write strictly valid UTF-8 bytes for basic chars.
                    val charBytes = char.toString().toByteArray(Charsets.UTF_8)
                    bytes.write(charBytes)
                }
            }
        }
        
        // Use CharsetDecoder to IGNORE malformed input instead of replacing with 
        try {
            val decoder = java.nio.charset.Charset.forName("UTF-8").newDecoder()
                .onMalformedInput(java.nio.charset.CodingErrorAction.IGNORE)
                .onUnmappableCharacter(java.nio.charset.CodingErrorAction.IGNORE)
            
            val byteBuffer = java.nio.ByteBuffer.wrap(bytes.toByteArray())
            val charBuffer = decoder.decode(byteBuffer)
            
            // Further cleanup: remove NULL bytes and Replacement Characters if any slipped through
            return charBuffer.toString()
                .replace("\uFFFD", "")
                .replace("\u0000", "")
        } catch (e: Exception) {
            // Fallback
            return bytes.toString("UTF-8").replace("\uFFFD", "")
        }
    }
    
    fun getEosTokenId(): Long = eosTokenId.toLong()
}
