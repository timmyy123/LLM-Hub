package com.llmhub.llmhub.embedding

import android.util.Log
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.*

interface RagService {
    suspend fun addDocument(chatId: String, content: String, metadata: String = "", fileName: String = "")
    suspend fun searchRelevantContext(chatId: String, query: String, maxResults: Int = 3): List<ContextChunk>
    suspend fun clearChatDocuments(chatId: String)
    suspend fun hasDocuments(chatId: String): Boolean
    suspend fun getDocumentCount(chatId: String): Int
}

// Public data class for context chunks used by UI/ViewModel
data class ContextChunk(
    val content: String,
    val metadata: String,
    val fileName: String,
    val similarity: Float,
    val chunkIndex: Int
)

/**
 * Enhanced RAG service with better context management and metadata support.
 * 
 * Features:
 * - Semantic chunking of documents
 * - Cosine similarity search
 * - Metadata preservation
 * - Per-chat document isolation
 * - Context ranking and filtering
 */
class InMemoryRagService(private val embeddingService: EmbeddingService) : RagService {
    
    private val documents = mutableMapOf<String, MutableList<DocumentChunk>>()
    private val mutex = Mutex()
    private val TAG = "InMemoryRagService"
    
    data class DocumentChunk(
        val content: String,
        val metadata: String,
        val fileName: String,
        val chunkIndex: Int,
        val embedding: FloatArray?
    )
    
    // Use top-level ContextChunk
    
    override suspend fun addDocument(chatId: String, content: String, metadata: String, fileName: String) = mutex.withLock {
        try {
            if (content.trim().isEmpty()) {
                Log.w(TAG, "Empty content provided for document")
                return@withLock
            }
            
            Log.d(TAG, "Adding document '$fileName' to chat $chatId (${content.length} chars)")
            
            // Enhanced chunking with overlap for better context preservation
            val chunks = createSmartChunks(content, maxChunkSize = 800, overlapSize = 100)
            
            val documentChunks = documents.getOrPut(chatId) { mutableListOf() }
            
            // Generate embeddings for each chunk
            var addedChunks = 0
            for ((index, chunk) in chunks.withIndex()) {
                if (chunk.trim().length < 50) { // Skip very short chunks
                    continue
                }
                
                val embedding = embeddingService.generateEmbedding(chunk)
                if (embedding != null) {
                    val documentChunk = DocumentChunk(
                        content = chunk,
                        metadata = metadata,
                        fileName = fileName,
                        chunkIndex = index,
                        embedding = embedding
                    )
                    documentChunks.add(documentChunk)
                    addedChunks++
                } else {
                    Log.w(TAG, "Failed to generate embedding for chunk $index of $fileName")
                }
            }
            
            Log.d(TAG, "Added $addedChunks chunks from '$fileName' to chat $chatId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add document to RAG", e)
        }
    }
    
    override suspend fun searchRelevantContext(chatId: String, query: String, maxResults: Int): List<ContextChunk> = mutex.withLock {
        try {
            val chatDocuments = documents[chatId] ?: return emptyList()
            if (chatDocuments.isEmpty()) return emptyList()
            
            Log.d(TAG, "Searching for relevant context in ${chatDocuments.size} chunks for query: '${query.take(50)}...'")
            
            val queryEmbedding = embeddingService.generateEmbedding(query) ?: return emptyList()
            
            val similarities = mutableListOf<ContextChunk>()
            
            for (doc in chatDocuments) {
                if (doc.embedding != null) {
                    val similarity = cosineSimilarity(queryEmbedding, doc.embedding)
                    similarities.add(
                        ContextChunk(
                            content = doc.content,
                            metadata = doc.metadata,
                            fileName = doc.fileName,
                            similarity = similarity,
                            chunkIndex = doc.chunkIndex
                        )
                    )
                }
            }
            
            // Return top similar chunks with more nuanced filtering
            val results = similarities
                .sortedByDescending { it.similarity }
                .take(maxResults * 2) // Get more candidates for diversity
                .filter { it.similarity > 0.3f } // Lower threshold for more inclusive results
                .distinctBy { it.content.take(100) } // Remove near-duplicates
                .take(maxResults)
            
            Log.d(TAG, "Found ${results.size} relevant chunks with similarities: ${results.map { "%.3f".format(it.similarity) }}")
            
            return results
                
        } catch (e: Exception) {
            Log.e(TAG, "Failed to search relevant context", e)
            emptyList()
        }
    }
    
    override suspend fun hasDocuments(chatId: String): Boolean = mutex.withLock {
        documents[chatId]?.isNotEmpty() ?: false
    }
    
    override suspend fun getDocumentCount(chatId: String): Int = mutex.withLock {
        documents[chatId]?.size ?: 0
    }
    
    override suspend fun clearChatDocuments(chatId: String) {
        mutex.withLock {
            val count = documents[chatId]?.size ?: 0
            documents.remove(chatId)
            Log.d(TAG, "Cleared $count document chunks for chat $chatId")
        }
    }
    
    /**
     * Create smart chunks with semantic boundaries and overlap for better context preservation
     */
    private fun createSmartChunks(text: String, maxChunkSize: Int, overlapSize: Int): List<String> {
        if (text.length <= maxChunkSize) {
            return listOf(text)
        }
        
        val chunks = mutableListOf<String>()
        
        // Try to split on paragraph boundaries first
    val paragraphs = text.split(Regex("""\n\s*\n""")).filter { it.trim().isNotEmpty() }
        
        if (paragraphs.size > 1) {
            // Handle paragraph-based chunking
            var currentChunk = StringBuilder()
            
            for (paragraph in paragraphs) {
                val trimmedParagraph = paragraph.trim()
                
                // If adding this paragraph would exceed the limit
                if (currentChunk.length + trimmedParagraph.length > maxChunkSize && currentChunk.isNotEmpty()) {
                    chunks.add(currentChunk.toString().trim())
                    
                    // Start new chunk with overlap from previous chunk
                    val overlapText = getOverlapText(currentChunk.toString(), overlapSize)
                    currentChunk = StringBuilder(overlapText)
                    
                    // Add separator if we have overlap
                    if (overlapText.isNotEmpty()) {
                        currentChunk.append("\n\n")
                    }
                }
                
                // If paragraph itself is too long, split it
                if (trimmedParagraph.length > maxChunkSize) {
                    val subChunks = splitLongText(trimmedParagraph, maxChunkSize, overlapSize)
                    for ((index, subChunk) in subChunks.withIndex()) {
                        if (index == 0 && currentChunk.isNotEmpty()) {
                            currentChunk.append(subChunk)
                            chunks.add(currentChunk.toString().trim())
                            currentChunk = StringBuilder()
                        } else {
                            chunks.add(subChunk)
                        }
                    }
                } else {
                    currentChunk.append(trimmedParagraph).append("\n\n")
                }
            }
            
            // Add remaining content
            if (currentChunk.isNotEmpty()) {
                chunks.add(currentChunk.toString().trim())
            }
        } else {
            // Fallback to sentence-based chunking
            return splitLongText(text, maxChunkSize, overlapSize)
        }
        
        return chunks.filter { it.trim().isNotEmpty() }
    }
    
    /**
     * Split long text by sentences with overlap
     */
    private fun splitLongText(text: String, maxChunkSize: Int, overlapSize: Int): List<String> {
        val sentences = text.split(Regex("[.!?]+")).filter { it.trim().isNotEmpty() }
        val chunks = mutableListOf<String>()
        var currentChunk = StringBuilder()
        
        for (sentence in sentences) {
            val trimmedSentence = sentence.trim()
            if (currentChunk.length + trimmedSentence.length > maxChunkSize && currentChunk.isNotEmpty()) {
                chunks.add(currentChunk.toString().trim())
                
                // Create overlap
                val overlapText = getOverlapText(currentChunk.toString(), overlapSize)
                currentChunk = StringBuilder(overlapText)
                if (overlapText.isNotEmpty()) {
                    currentChunk.append(". ")
                }
            }
            currentChunk.append(trimmedSentence).append(". ")
        }
        
        if (currentChunk.isNotEmpty()) {
            chunks.add(currentChunk.toString().trim())
        }
        
        return chunks.ifEmpty { listOf(text) }
    }
    
    /**
     * Get overlap text from the end of a chunk
     */
    private fun getOverlapText(text: String, overlapSize: Int): String {
        if (text.length <= overlapSize) return text
        
        val overlapStart = text.length - overlapSize
        val overlapText = text.substring(overlapStart)
        
        // Try to start overlap at sentence boundary
        val sentenceStart = overlapText.lastIndexOf(". ")
        return if (sentenceStart > 0) {
            overlapText.substring(sentenceStart + 2)
        } else {
            overlapText
        }
    }
    
    /**
     * Compute cosine similarity between two embedding vectors
     */
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        val denominator = sqrt(normA) * sqrt(normB)
        return if (denominator == 0f) 0f else dotProduct / denominator
    }
}
