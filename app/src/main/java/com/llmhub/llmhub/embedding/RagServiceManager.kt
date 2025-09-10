package com.llmhub.llmhub.embedding

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.first
import com.llmhub.llmhub.data.ThemePreferences

/**
 * Manages the RAG service lifecycle and provides easy access to document embeddings.
 * Handles initialization of the MediaPipe embedding service and RAG functionality.
 */
class RagServiceManager(
    private val context: Context
) {
    
    private var embeddingService: EmbeddingService? = null
    private var ragService: RagService? = null
    private var isInitialized = false
    private val initMutex = Mutex()
    private val TAG = "RagServiceManager"
    private var initializationJob: Job? = null
    
    /**
     * Initialize the RAG service asynchronously in the background
     */
    fun initializeAsync(): Job {
        if (initializationJob?.isActive == true) {
            return initializationJob!!
        }
        
        initializationJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                initMutex.withLock {
                    if (isInitialized) return@withLock
                    
                    // Get selected embedding model from user preferences
                    val themePreferences = ThemePreferences(context)
                    val selectedEmbeddingModel = themePreferences.selectedEmbeddingModel.first()
                    
                    Log.d(TAG, "Initializing RAG service with embedding model: ${selectedEmbeddingModel ?: "disabled"}")
                    
                    // Check if embeddings are disabled (selectedEmbeddingModel is null)
                    if (selectedEmbeddingModel == null) {
                        Log.i(TAG, "🚫 Embeddings are DISABLED by user preference - RAG service not initialized")
                        Log.i(TAG, "📝 Documents can still be uploaded but won't be used for semantic search")
                        isInitialized = false
                        return@withLock
                    }
                    
                    // Initialize embedding service with selected model
                    Log.i(TAG, "🔧 Initializing RAG service with embedding model: $selectedEmbeddingModel")
                    val embeddingService = MediaPipeEmbeddingService(context, selectedEmbeddingModel)
                    if (embeddingService.initialize()) {
                        this@RagServiceManager.embeddingService = embeddingService
                        this@RagServiceManager.ragService = InMemoryRagService(embeddingService)
                        isInitialized = true
                        Log.i(TAG, "✅ RAG service initialized successfully with embedding model: $selectedEmbeddingModel")
                        Log.i(TAG, "🔍 Document uploads will now use embeddings for semantic search")
                    } else {
                        Log.e(TAG, "❌ Failed to initialize embedding service with model: $selectedEmbeddingModel")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing RAG service", e)
            }
        }
        
        return initializationJob!!
    }
    
    /**
     * Get the RAG service if initialized, otherwise null
     */
    suspend fun getRagService(): RagService? {
        initMutex.withLock {
            return ragService
        }
    }
    
    /**
     * Check if RAG service is ready to use
     */
    suspend fun isReady(): Boolean {
        initMutex.withLock {
            return isInitialized && ragService != null
        }
    }
    
    /**
     * Get embedding status for debugging
     */
    suspend fun getEmbeddingStatus(): String {
        val themePreferences = ThemePreferences(context)
        val selectedEmbeddingModel = themePreferences.selectedEmbeddingModel.first()
        val ready = isReady()
        
        return when {
            selectedEmbeddingModel == null -> "Embeddings DISABLED by user"
            !ready -> "Embeddings ENABLED but service not ready (model: $selectedEmbeddingModel)"
            else -> "Embeddings ACTIVE (model: $selectedEmbeddingModel)"
        }
    }
    
    /**
     * Add a document to the RAG system for a specific chat
     */
    suspend fun addDocument(chatId: String, content: String, fileName: String, metadata: String = ""): Boolean {
        // Check if embeddings are disabled
        if (!isReady()) {
            Log.d(TAG, "RAG service not ready or embeddings disabled - skipping document '$fileName'")
            return false
        }
        
        val service = getRagService()
        return if (service != null) {
            try {
                service.addDocument(chatId, content, metadata, fileName)
                Log.d(TAG, "Added document '$fileName' to chat $chatId")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to add document to RAG", e)
                false
            }
        } else {
            Log.w(TAG, "RAG service not ready, cannot add document")
            false
        }
    }
    
    /**
     * Search for relevant context in a chat's documents
     */
    suspend fun searchRelevantContext(chatId: String, query: String, maxResults: Int = 3): List<ContextChunk> {
        // Check if embeddings are disabled
        if (!isReady()) {
            Log.d(TAG, "RAG service not ready or embeddings disabled - returning empty context")
            return emptyList()
        }
        
        val service = getRagService()
        return if (service != null) {
            try {
                service.searchRelevantContext(chatId, query, maxResults)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to search relevant context", e)
                emptyList()
            }
        } else {
            Log.w(TAG, "RAG service not ready, cannot search context")
            emptyList()
        }
    }
    
    /**
     * Check if a chat has any documents
     */
    suspend fun hasDocuments(chatId: String): Boolean {
        val service = getRagService()
        return service?.hasDocuments(chatId) ?: false
    }
    
    /**
     * Get document count for a chat
     */
    suspend fun getDocumentCount(chatId: String): Int {
        val service = getRagService()
        return service?.getDocumentCount(chatId) ?: 0
    }
    
    /**
     * Clear all documents for a chat
     */
    suspend fun clearChatDocuments(chatId: String) {
        val service = getRagService()
        if (service != null) {
            try {
                service.clearChatDocuments(chatId)
                Log.d(TAG, "Cleared documents for chat $chatId")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear documents for chat $chatId", e)
            }
        }
    }
    
    // getDocumentCount is defined above; keep a single definition.
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        try {
            initializationJob?.cancel()
            embeddingService?.cleanup()
            embeddingService = null
            ragService = null
            isInitialized = false
            Log.d(TAG, "RAG service cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
}
