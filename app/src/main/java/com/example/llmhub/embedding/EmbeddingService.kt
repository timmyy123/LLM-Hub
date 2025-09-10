package com.llmhub.llmhub.embedding

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Optional
import com.llmhub.llmhub.data.localFileName
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import com.google.ai.edge.localagents.rag.models.EmbeddingRequest
import com.google.ai.edge.localagents.rag.models.EmbedData
import com.google.common.collect.ImmutableList

interface EmbeddingService {
    suspend fun generateEmbedding(text: String): FloatArray?
    suspend fun isInitialized(): Boolean
    suspend fun initialize(): Boolean
    fun cleanup()
}

/**
 * AI Edge RAG SDK-based text embedding service using Gecko text embedding models.
 * 
 * This service can use various Gecko models downloaded by the user:
 * - Gecko-110m-en with different dimensions (64, 256, 512, 1024)
 * - Both quantized and float32 versions
 * 
 * Uses the proper GeckoEmbeddingModel from AI Edge RAG SDK which handles
 * the TFLite models and tokenizer correctly.
 */
class MediaPipeEmbeddingService(
    private val context: Context,
    private val selectedModelName: String? = null
) : EmbeddingService {
    
    private var geckoEmbedder: GeckoEmbeddingModel? = null
    private var isInitialized = false
    private val initMutex = Mutex()
    
    companion object {
        private const val TAG = "GeckoEmbedding"
        private const val USE_GPU_FOR_EMBEDDINGS = true
        private const val LEGACY_EMBEDDING_MODEL_PATH = "text-embed/embeddinggemma-300M_seq2048_mixed-precision.tflite"
        private const val LEGACY_VOCAB_MODEL_PATH = "text-embed/sentencepiece.model"
    }
    
    override suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        initMutex.withLock {
            if (isInitialized) {
                return@withLock true
            }
            
            try {
                Log.d(TAG, "Initializing Gecko Embedding Model...")
                
                val (modelPath, tokenizerPath) = getGeckoModelPaths()
                if (modelPath == null) {
                    Log.e(TAG, "No Gecko embedding model available")
                    return@withLock false
                }
                
                Log.d(TAG, "Using Gecko model: $modelPath")
                Log.d(TAG, "Using tokenizer: ${tokenizerPath ?: "built-in (none provided)"}")
                
                // Create the Gecko embedding model
                geckoEmbedder = GeckoEmbeddingModel(
                    modelPath,
                    if (tokenizerPath != null) Optional.of(tokenizerPath) else Optional.empty(),
                    USE_GPU_FOR_EMBEDDINGS
                )
                
                Log.d(TAG, "Gecko embedding model created successfully")
                
                // Test with a simple embedding to ensure it works
                try {
                    val request = EmbeddingRequest.create(
                        listOf(EmbedData.create("test", EmbedData.TaskType.RETRIEVAL_QUERY))
                    )
                    val future = geckoEmbedder?.getEmbeddings(request)
                    val testEmbedding: ImmutableList<Float>? = future?.get()
                    
                    if (testEmbedding != null && testEmbedding.isNotEmpty()) {
                        Log.d(TAG, "Gecko embedder initialized successfully. Embedding dimension: ${testEmbedding.size}")
                        isInitialized = true
                        return@withLock true
                    } else {
                        Log.e(TAG, "Gecko embedder test failed - no embeddings returned")
                        geckoEmbedder = null
                        return@withLock false
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Gecko embedder test failed with error: ${e.message}")
                    geckoEmbedder = null
                    return@withLock false
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing Gecko embedder: ${e.message}", e)
                geckoEmbedder = null
                return@withLock false
            }
        }
    }

    private fun getGeckoModelPaths(): Pair<String?, String?> {
        // Get embedding models from ModelData
        val embeddingModels = com.llmhub.llmhub.data.ModelData.models.filter { it.category == "embedding" }
        val modelsDir = File(context.filesDir, "models")
        
        // Check for tokenizer (optional for some Gecko models)
        val tokenizerModel = embeddingModels.find { it.name.contains("Tokenizer") || it.name.contains("SentencePiece") }
        val tokenizerPath = if (tokenizerModel != null) {
            val tokenizerFile = File(modelsDir, tokenizerModel.localFileName())
            if (tokenizerFile.exists()) {
                Log.d(TAG, "Found separate tokenizer: ${tokenizerFile.absolutePath}")
                tokenizerFile.absolutePath
            } else {
                Log.d(TAG, "Separate tokenizer not found, will try built-in tokenizer")
                null
            }
        } else {
            Log.d(TAG, "No separate tokenizer model configured, will try built-in tokenizer")
            null
        }
        
        // First check for user-selected model
        if (selectedModelName != null) {
            val model = embeddingModels.find { it.name == selectedModelName && !it.name.contains("Tokenizer") }
            if (model != null) {
                val modelFile = File(modelsDir, model.localFileName())
                if (modelFile.exists()) {
                    return Pair(modelFile.absolutePath, tokenizerPath)
                }
            }
        }

        // Fallback to any available downloaded Gecko model (exclude tokenizer)
        if (modelsDir.exists()) {
            embeddingModels
                .filter { !it.name.contains("Tokenizer") && !it.name.contains("SentencePiece") }
                .forEach { model ->
                    val modelFile = File(modelsDir, model.localFileName())
                    if (modelFile.exists()) {
                        Log.d(TAG, "Using fallback Gecko model: ${model.name}")
                        return Pair(modelFile.absolutePath, tokenizerPath)
                    }
                }
        }

        Log.e(TAG, "No Gecko embedding models found")
        return Pair(null, null)
    }
    
    override suspend fun generateEmbedding(text: String): FloatArray? = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            Log.w(TAG, "Gecko embedder not initialized, attempting to initialize...")
            if (!initialize()) {
                Log.e(TAG, "Failed to initialize Gecko embedder")
                return@withContext null
            }
        }
        
        try {
            val embedder = geckoEmbedder ?: return@withContext null
            
            // Clean and prepare text for embedding
            val cleanText = text.trim().take(1024) // Gecko models support up to 1024 tokens
            if (cleanText.isEmpty()) {
                Log.w(TAG, "Empty text provided for embedding")
                return@withContext null
            }
            
            Log.d(TAG, "Generating embedding for text: '${cleanText.take(100)}${if (cleanText.length > 100) "..." else ""}'")
            
            // Generate embedding using Gecko model with proper API
            val request = EmbeddingRequest.create(
                listOf(EmbedData.create(cleanText, EmbedData.TaskType.RETRIEVAL_QUERY))
            )
            val future = embedder.getEmbeddings(request)
            val embedding: ImmutableList<Float>? = future?.get()
            
            if (embedding != null && embedding.isNotEmpty()) {
                Log.d(TAG, "Generated embedding with ${embedding.size} dimensions")
                return@withContext embedding.toFloatArray()
            } else {
                Log.w(TAG, "No embedding returned for text: '${cleanText.take(50)}...'")
                return@withContext null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error generating embedding: ${e.message}", e)
            return@withContext null
        }
    }
    
    override suspend fun isInitialized(): Boolean {
        return isInitialized && geckoEmbedder != null
    }
    
    override fun cleanup() {
        try {
            // GeckoEmbeddingModel doesn't have a close() method
            // Just set to null and let GC handle it
            geckoEmbedder = null
            isInitialized = false
            Log.d(TAG, "Gecko embedder cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up Gecko embedder", e)
        }
    }
}
