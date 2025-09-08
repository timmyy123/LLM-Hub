package com.llmhub.llmhub.embedding

import android.content.Context
import android.util.Log
import com.google.mediapipe.framework.MediaPipeException
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import com.google.mediapipe.tasks.text.textembedder.TextEmbedderResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

interface EmbeddingService {
    suspend fun generateEmbedding(text: String): FloatArray?
    suspend fun isInitialized(): Boolean
    suspend fun initialize(): Boolean
    fun cleanup()
}

/**
 * MediaPipe-based text embedding service using the Gemma text embedding model.
 * 
 * This service uses the text embedding model files you have in the text-embed folder:
 * - embeddinggemma-300M_seq2048_mixed-precision.tflite
 * - sentencepiece.model
 * 
 * Following Google's AI Edge RAG guide for text embeddings.
 */
class MediaPipeEmbeddingService(private val context: Context) : EmbeddingService {
    
    private var textEmbedder: TextEmbedder? = null
    private var isInitialized = false
    private val initMutex = Mutex()
    
    companion object {
        private const val TAG = "MediaPipeEmbedding"
        private const val EMBEDDING_MODEL_PATH = "text-embed/embeddinggemma-300M_seq2048_mixed-precision.tflite"
        private const val VOCAB_MODEL_PATH = "text-embed/sentencepiece.model"
    }
    
    override suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        initMutex.withLock {
            if (isInitialized) {
                return@withLock true
            }
            
            try {
                Log.d(TAG, "Initializing MediaPipe Text Embedder...")
                
                // Check if model files exist
                val embeddingModelPath = EMBEDDING_MODEL_PATH
                val vocabModelPath = VOCAB_MODEL_PATH
                
                Log.d(TAG, "Looking for embedding model at: $embeddingModelPath")
                Log.d(TAG, "Looking for vocab model at: $vocabModelPath")
                
                // Test if assets exist
                try {
                    context.assets.open(embeddingModelPath).use { 
                        Log.d(TAG, "Found embedding model in assets")
                    }
                    context.assets.open(vocabModelPath).use { 
                        Log.d(TAG, "Found vocab model in assets") 
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Model files not found in assets: ${e.message}")
                    return@withLock false
                }
                
                // Create the text embedder directly from asset file
                textEmbedder = TextEmbedder.createFromFile(context, embeddingModelPath)
                
                // Test with a simple embedding to ensure it works
                val testResult = textEmbedder?.embed("test")
                if (testResult?.embeddingResult()?.embeddings()?.isNotEmpty() == true) {
                    val embeddingSize = testResult.embeddingResult().embeddings()[0].floatEmbedding().size
                    Log.d(TAG, "TextEmbedder initialized successfully. Embedding dimension: $embeddingSize")
                    isInitialized = true
                    return@withLock true
                } else {
                    Log.e(TAG, "TextEmbedder test failed - no embeddings returned")
                    textEmbedder?.close()
                    textEmbedder = null
                    return@withLock false
                }
                
            } catch (e: MediaPipeException) {
                Log.e(TAG, "MediaPipe error initializing text embedder", e)
                textEmbedder?.close()
                textEmbedder = null
                return@withLock false
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing text embedder", e)
                textEmbedder?.close()
                textEmbedder = null
                return@withLock false
            }
        }
    }
    
    override suspend fun generateEmbedding(text: String): FloatArray? = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            Log.w(TAG, "Text embedder not initialized, attempting to initialize...")
            if (!initialize()) {
                Log.e(TAG, "Failed to initialize text embedder")
                return@withContext null
            }
        }
        
        try {
            val embedder = textEmbedder ?: return@withContext null
            
            // Clean and prepare text for embedding
            val cleanText = text.trim().take(2048) // Limit to model's sequence length
            if (cleanText.isEmpty()) {
                Log.w(TAG, "Empty text provided for embedding")
                return@withContext null
            }
            
            Log.d(TAG, "Generating embedding for text: '${cleanText.take(100)}${if (cleanText.length > 100) "..." else ""}'")
            
            // Generate embedding
            val result: TextEmbedderResult = embedder.embed(cleanText)
            val embeddings = result.embeddingResult().embeddings()
            
            if (embeddings.isNotEmpty()) {
                val embedding = embeddings[0].floatEmbedding()
                Log.d(TAG, "Generated embedding with ${embedding.size} dimensions")
                return@withContext embedding
            } else {
                Log.w(TAG, "No embeddings returned for text")
                return@withContext null
            }
            
        } catch (e: MediaPipeException) {
            Log.e(TAG, "MediaPipe error generating embedding", e)
            return@withContext null
        } catch (e: Exception) {
            Log.e(TAG, "Error generating embedding", e)
            return@withContext null
        }
    }
    
    override suspend fun isInitialized(): Boolean {
        return isInitialized && textEmbedder != null
    }
    
    override fun cleanup() {
        try {
            textEmbedder?.close()
            textEmbedder = null
            isInitialized = false
            Log.d(TAG, "TextEmbedder cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up TextEmbedder", e)
        }
    }
}
