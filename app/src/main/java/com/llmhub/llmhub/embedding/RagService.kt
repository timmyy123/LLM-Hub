package com.llmhub.llmhub.embedding

import android.util.Log
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.*

interface RagService {
    suspend fun addDocument(chatId: String, content: String, metadata: String = "", fileName: String = "")
    // Add a precomputed document chunk with supplied embedding (avoid re-embedding)
    suspend fun addDocumentWithEmbedding(chatId: String, content: String, fileName: String, chunkIndex: Int, embedding: FloatArray, metadata: String = "")
    // relaxedLexicalFallback: when true, the implementation should be more permissive
    // with lexical fallbacks (useful for explicit "what do you remember" style queries)
    // queryEmbedding: optional precomputed embedding for the query. When supplied,
    // implementations should use it instead of calling the embedder again.
    suspend fun searchRelevantContext(chatId: String, query: String, maxResults: Int = 3, relaxedLexicalFallback: Boolean = false, queryEmbedding: FloatArray? = null): List<ContextChunk>
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
                val trimmedChunk = chunk.trim()
                // Skip very short chunks only when the document produced multiple chunks.
                // If the document yields a single short chunk (e.g., a short pasted memory),
                // still attempt embedding so short memories are preserved.
                if (trimmedChunk.length < 50 && chunks.size > 1) {
                    Log.d(TAG, "Skipping very short chunk $index of '$fileName' (len=${trimmedChunk.length})")
                    continue
                }
                val embedding = embeddingService.generateEmbedding(chunk)
                if (embedding != null) {
                    // Defensive copy: ensure we store our own copy of the embedding so
                    // downstream calls can't be affected if the embedder reuses internal buffers.
                    val documentChunk = DocumentChunk(
                        content = chunk,
                        metadata = metadata,
                        fileName = fileName,
                        chunkIndex = index,
                        embedding = embedding.copyOf()
                    )
                    documentChunks.add(documentChunk)
                    addedChunks++
                    Log.d(TAG, "Generated embedding for chunk $index of '$fileName' (len=${chunk.length}) -> totalChunksNow=${documentChunks.size}")
                    Log.d(TAG, "Created document chunk with metadata='$metadata' for fileName='$fileName'")
                } else {
                    Log.w(TAG, "Failed to generate embedding for chunk $index of $fileName")
                }
            }
            
            Log.d(TAG, "Added $addedChunks chunks from '$fileName' to chat $chatId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add document to RAG", e)
        }
    }

    override suspend fun addDocumentWithEmbedding(chatId: String, content: String, fileName: String, chunkIndex: Int, embedding: FloatArray, metadata: String) {
        mutex.withLock {
            try {
                val documentChunks = documents.getOrPut(chatId) { mutableListOf() }
                val documentChunk = DocumentChunk(
                    content = content,
                    metadata = metadata,
                    fileName = fileName,
                    chunkIndex = chunkIndex,
                    embedding = embedding.copyOf()
                )
                documentChunks.add(documentChunk)
                Log.d(TAG, "Added precomputed embedding chunk $chunkIndex for '$fileName' to chat $chatId -> totalChunksNow=${documentChunks.size}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to add precomputed document chunk to RAG", e)
            }
        }
    }
    
    override suspend fun searchRelevantContext(chatId: String, query: String, maxResults: Int, relaxedLexicalFallback: Boolean, queryEmbedding: FloatArray?): List<ContextChunk> = mutex.withLock {
        try {
            val chatDocuments = documents[chatId] ?: return emptyList()
            if (chatDocuments.isEmpty()) {
                Log.d(TAG, "No documents found for chatId=$chatId during search")
                return emptyList()
            }

            Log.d(TAG, "Searching for relevant context in ${chatDocuments.size} chunks for query: '${query.take(50)}...'")
            // Debug: log metadata of all chunks
            for (chunk in chatDocuments) {
                Log.d(TAG, "Available chunk metadata='${chunk.metadata}', fileName='${chunk.fileName}'")
            }
            
            // Filter out replicated global chunks when memory is disabled
            // This ensures we only search chat-specific documents
            val filteredDocuments = chatDocuments.filter { 
                chunk -> chunk.metadata == "uploaded" || !chunk.metadata.startsWith("replicated_global:")
            }
            Log.d(TAG, "Filtered to ${filteredDocuments.size} chat-specific chunks (removed ${chatDocuments.size - filteredDocuments.size} replicated global chunks)")

            // Use the provided embedding if available to avoid re-embedding the same query
            val queryEmbCopy = queryEmbedding ?: run {
                val generated = embeddingService.generateEmbedding(query)
                if (generated == null) {
                    Log.w(TAG, "Failed to generate embedding for search query: '${query.take(80)}'")
                    return emptyList()
                }
                generated.copyOf()
            }

            val similarities = mutableListOf<ContextChunk>()

            for (doc in filteredDocuments) {
                if (doc.embedding != null) {
                    val similarity = cosineSimilarity(queryEmbCopy, doc.embedding)
                    similarities.add(
                        ContextChunk(
                            content = doc.content,
                            metadata = doc.metadata,
                            fileName = doc.fileName,
                            similarity = similarity,
                            chunkIndex = doc.chunkIndex
                        )
                    )
                } else {
                    Log.d(TAG, "Skipping doc chunk ${doc.chunkIndex} of '${doc.fileName}' due to missing embedding")
                }
            }

            // Simplified similarity filtering with clear thresholds
            val results = filterSimilarityCandidates(similarities, query, maxResults, relaxedLexicalFallback)

            Log.d(TAG, "Similarity candidates count=${similarities.size}; returning ${results.size} results -> scores=${results.map { "%.3f".format(it.similarity) }}")
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
     * Simplified similarity filtering with clear, maintainable logic
     */
    private fun filterSimilarityCandidates(
        similarities: List<ContextChunk>,
        query: String,
        maxResults: Int,
        relaxedLexicalFallback: Boolean
    ): List<ContextChunk> {
        // Clear thresholds
        val primaryThreshold = 0.60f      // High confidence semantic match
        val fallbackThreshold = 0.35f     // Moderate semantic match
        val lexicalThreshold = 0.15       // Minimum word overlap for fallback
        
        // Helper: compute Jaccard word overlap
        fun wordJaccard(a: String, b: String): Double {
            val wordsA = a.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }.toSet()
            val wordsB = b.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }.toSet()
            if (wordsA.isEmpty() || wordsB.isEmpty()) return 0.0
            val intersection = wordsA.intersect(wordsB).size.toDouble()
            val union = wordsA.union(wordsB).size.toDouble()
            return if (union == 0.0) 0.0 else intersection / union
        }
        
        // Simple acceptance logic
        fun shouldAccept(similarity: Float, overlap: Double, isShortMemory: Boolean): Boolean {
            return when {
                // High semantic similarity - always accept
                similarity >= primaryThreshold -> true
                
                // Moderate semantic similarity with lexical overlap
                similarity >= fallbackThreshold && overlap >= lexicalThreshold -> true
                
                // Short memories need higher standards to avoid noise
                isShortMemory -> (overlap >= 0.25) || (similarity >= 0.95f)
                
                else -> false
            }
        }
        
        // Get candidates sorted by similarity
        val candidates = similarities.sortedByDescending { it.similarity }
        
        // Filter candidates
        val filtered = mutableListOf<ContextChunk>()
        for (candidate in candidates) {
            if (filtered.size >= maxResults) break
            
            val similarity = candidate.similarity
            val overlap = wordJaccard(query, candidate.content)
            val isShortMemory = candidate.content.trim().length < 40
            
            if (shouldAccept(similarity, overlap, isShortMemory)) {
                filtered.add(candidate)
            }
        }
        
        // If no semantic matches and relaxed fallback is enabled, try lexical-only
        if (filtered.isEmpty() && relaxedLexicalFallback) {
            val lexicalCandidates = candidates.map { candidate ->
                val overlap = wordJaccard(query, candidate.content)
                Pair(candidate, overlap)
            }.filter { it.second > 0.05 } // Very low threshold for relaxed mode
              .sortedByDescending { it.second }
              .take(maxResults)
            
            return lexicalCandidates.map { it.first }
        }
        
        return filtered.distinctBy { it.content.take(100) }
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
