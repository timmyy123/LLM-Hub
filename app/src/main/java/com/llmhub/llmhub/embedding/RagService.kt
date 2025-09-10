package com.llmhub.llmhub.embedding

import android.util.Log
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.*

interface RagService {
    suspend fun addDocument(chatId: String, content: String, metadata: String = "", fileName: String = "")
    // relaxedLexicalFallback: when true, the implementation should be more permissive
    // with lexical fallbacks (useful for explicit "what do you remember" style queries)
    suspend fun searchRelevantContext(chatId: String, query: String, maxResults: Int = 3, relaxedLexicalFallback: Boolean = false): List<ContextChunk>
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
                } else {
                    Log.w(TAG, "Failed to generate embedding for chunk $index of $fileName")
                }
            }
            
            Log.d(TAG, "Added $addedChunks chunks from '$fileName' to chat $chatId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add document to RAG", e)
        }
    }
    
    override suspend fun searchRelevantContext(chatId: String, query: String, maxResults: Int, relaxedLexicalFallback: Boolean): List<ContextChunk> = mutex.withLock {
        try {
            val chatDocuments = documents[chatId] ?: return emptyList()
            if (chatDocuments.isEmpty()) {
                Log.d(TAG, "No documents found for chatId=$chatId during search")
                return emptyList()
            }

            Log.d(TAG, "Searching for relevant context in ${chatDocuments.size} chunks for query: '${query.take(50)}...'")

            val queryEmbedding = embeddingService.generateEmbedding(query)
            if (queryEmbedding == null) {
                Log.w(TAG, "Failed to generate embedding for search query: '${query.take(80)}'")
                return emptyList()
            }

            // Copy query embedding defensively as well to avoid accidental mutation from
            // underlying embedding service implementations that might reuse buffers.
            val queryEmbCopy = queryEmbedding.copyOf()

            val similarities = mutableListOf<ContextChunk>()
            // Keep a map from (fileName, chunkIndex) -> embedding so we can log diagnostics
            val embeddingMap = mutableMapOf<Pair<String, Int>, FloatArray>()

            for (doc in chatDocuments) {
                if (doc.embedding != null) {
                    // Use the defensive copy for comparisons
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
                    // Store a copy in the diagnostic map as well
                    embeddingMap[Pair(doc.fileName, doc.chunkIndex)] = doc.embedding.copyOf()
                } else {
                    Log.d(TAG, "Skipping doc chunk ${doc.chunkIndex} of '${doc.fileName}' due to missing embedding")
                }
            }

            // Return top similar chunks with more nuanced filtering to avoid false positives
            val primarySimilarityThreshold = 0.60f // high-confidence semantic match
            val fallbackSimilarityThreshold = 0.35f // moderate semantic match
            // When caller requests a relaxed lexical fallback (explicit memory queries),
            // be more permissive when falling back due to embedder unreliability
            // and consider a larger candidate pool so short/pasted memories aren't
            // lost behind large documents like resumes.
            val lexicalFallbackMinOverlap = if (relaxedLexicalFallback) 0.05 else 0.15
            val shortMemoryOverlapReq = if (relaxedLexicalFallback) 0.10 else 0.25
            val candidateMultiplier = if (relaxedLexicalFallback) 10 else 3

            // Helper: compute simple Jaccard word overlap between query and chunk content
            fun wordJaccard(a: String, b: String): Double {
                val wa = a.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }.toSet()
                val wb = b.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }.toSet()
                if (wa.isEmpty() || wb.isEmpty()) return 0.0
                val inter = wa.intersect(wb).size.toDouble()
                val union = wa.union(wb).size.toDouble()
                return if (union == 0.0) 0.0 else inter / union
            }

            // Evaluate candidates: require either a high similarity OR a moderate similarity paired
            // with non-trivial word overlap to avoid returning very short memories for unrelated queries.
            // Expand candidate set for relaxed lexical fallback to ensure short
            // explicit memories (e.g., pasted "your name is David") are considered.
            val candidates = similarities.sortedByDescending { it.similarity }.take(maxResults * candidateMultiplier)

            // Small diagnostic preview for debugging: list filenames and short previews
            try {
                val preview = candidates.map { c -> "${c.fileName}:${c.content.trim().take(80).replace("\n"," ")}" }
                Log.d(TAG, "RAG diag: candidate previews=${preview}")
            } catch (_: Exception) {
                // ignore preview failures
            }
            val filtered = mutableListOf<ContextChunk>()
            var suspiciousExactEqualCount = 0
            for (candidate in candidates) {
                val sim = candidate.similarity
                val overlap = wordJaccard(query, candidate.content)

                // Heuristic: avoid injecting very short memories (e.g., "I am David.") for
                // unrelated queries. For short chunks, require a higher lexical overlap OR
                // an extremely high semantic similarity to accept them.
                val isShortMemory = candidate.content.trim().length < 40
                val accept = if (isShortMemory) {
                    // For short memories require either decent overlap or near-exact semantic match
                    (overlap >= 0.25) || (sim >= 0.95f)
                } else {
                    when {
                        sim >= primarySimilarityThreshold -> true
                        sim >= fallbackSimilarityThreshold && overlap >= 0.15 -> true
                        else -> false
                    }
                }

                Log.d(TAG, "RAG candidate filter: sim=${"%.3f".format(sim)}, overlap=${"%.3f".format(overlap)}, short=$isShortMemory, accept=$accept, file=${candidate.fileName}")

                // Diagnostics: if similarity is very high, log candidate preview and embedding comparisons
                val DIAG_SIM_THRESHOLD = 0.95f
                // Track whether the query embedding equals the document embedding exactly.
                // Declared here so it can be referenced later for fallback logic.
                var exactEqual = false
                if (sim >= DIAG_SIM_THRESHOLD) {
                    try {
                        val key = Pair(candidate.fileName, candidate.chunkIndex)
                        val docEmb = embeddingMap[key]
                        Log.d(TAG, "RAG diag: high-sim candidate preview='${candidate.content.take(120).replace("\n", " ")}' file=${candidate.fileName} chunkIndex=${candidate.chunkIndex}")
                        if (docEmb != null) {
                            // Print first few components of each vector for quick eyeballing
                            val previewN = min(6, min(queryEmbedding.size, docEmb.size))
                            val qPreview = (0 until previewN).joinToString(",") { i -> "${"%.6f".format(queryEmbCopy[i])}" }
                            val dPreview = (0 until previewN).joinToString(",") { i -> "${"%.6f".format(docEmb[i])}" }
                            // Compute L2 distance
                            var sumSq = 0.0f
                            val minLen = min(queryEmbedding.size, docEmb.size)
                            for (i in 0 until minLen) {
                                val diff = queryEmbCopy[i] - docEmb[i]
                                sumSq += diff * diff
                            }
                            val l2 = sqrt(sumSq.toDouble())
                            exactEqual = if (queryEmbCopy.size == docEmb.size) queryEmbCopy.contentEquals(docEmb) else false
                            if (exactEqual && candidate.content.trim() != query.trim()) {
                                Log.w(TAG, "RAG diag: exact-equal embeddings for different texts - possible embedder buffer reuse; attempting re-embed for query and falling back to lexical overlap if necessary. file=${candidate.fileName} idx=${candidate.chunkIndex}")

                                // Try re-embedding the query with a tiny perturbation to avoid
                                // embedder buffer-reuse / caching issues. If re-embed yields a
                                // different vector, use that to recompute similarity.
                                try {
                                    val perturbed = query + "\u200B" // zero-width space
                                    val retryEmb = embeddingService.generateEmbedding(perturbed)
                                    if (retryEmb != null) {
                                        val retryEmbCopy = retryEmb.copyOf()
                                        val minLen2 = min(retryEmbCopy.size, docEmb.size)
                                        var sumSq2 = 0.0f
                                        var dot2 = 0f
                                        var normA2 = 0f
                                        var normB2 = 0f
                                        for (i in 0 until minLen2) {
                                            dot2 += retryEmbCopy[i] * docEmb[i]
                                            normA2 += retryEmbCopy[i] * retryEmbCopy[i]
                                            normB2 += docEmb[i] * docEmb[i]
                                        }
                                        val denom2 = sqrt(normA2) * sqrt(normB2)
                                        val newSim = if (denom2 == 0f) 0f else dot2 / denom2
                                        Log.d(TAG, "RAG diag: re-embed similarity for perturbed query vs doc=${"%.6f".format(newSim)}")

                                        // If the re-embed produced a different vector (not exact equal),
                                        // use that similarity for the acceptance heuristics below.
                                        val newExact = if (retryEmbCopy.size == docEmb.size) retryEmbCopy.contentEquals(docEmb) else false
                                        if (!newExact) {
                                            // Replace queryEmbCopy for subsequent diagnostics/formatting
                                            // Note: don't mutate original queryEmbedding variable
                                            Log.d(TAG, "RAG diag: re-embed produced different vector; using new similarity")
                                            // recompute qPreview for logging
                                            val previewN2 = min(6, min(retryEmbCopy.size, docEmb.size))
                                            val qPreview2 = (0 until previewN2).joinToString(",") { i -> "${"%.6f".format(retryEmbCopy[i])}" }
                                            Log.d(TAG, "RAG diag: qPreview=[$qPreview2] dPreview=[$dPreview] l2=${"%.6f".format(sqrt(sumSq2.toDouble()))} exactEqual=$newExact")
                                            // Use newSim as sim for decision; override sim variable by shadowing
                                            // (we'll use local variable below when evaluating accept)
                                            // To keep changes small, set a flag on candidate via metadata map
                                            // We'll recompute overlap and accept below using newSim via an auxiliary map
                                            // For simplicity, attach newSim to candidate.metadata temporarily by encoding
                                            // but to avoid mutating candidate, skip and instead set a local variable
                                            // We'll recompute acceptance here directly and add to filtered if accepted.
                                            val overlapNew = wordJaccard(query, candidate.content)
                                            val isShortMemoryNew = candidate.content.trim().length < 40
                                                                    val acceptNew = if (isShortMemoryNew) {
                                                                        (overlapNew >= shortMemoryOverlapReq) || (newSim >= 0.95f)
                                                                    } else {
                                                                        when {
                                                                            newSim >= primarySimilarityThreshold -> true
                                                                            newSim >= fallbackSimilarityThreshold && overlapNew >= lexicalFallbackMinOverlap -> true
                                                                            else -> false
                                                                        }
                                                                    }
                                            if (acceptNew) {
                                                filtered.add(candidate)
                                            }
                                            if (filtered.size >= maxResults) break
                                            // We've handled this candidate via re-embed path; continue outer loop
                                            continue
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.w(TAG, "RAG diag: re-embed attempt failed: ${e.message}")
                                }

                                // If we reach here the re-embed either failed or produced no change.
                                // Fall back to a lexical-only decision to avoid silently dropping
                                // potentially relevant memories when the embedder is unreliable.
                                val fallbackAccept = if (isShortMemory) {
                                    overlap >= shortMemoryOverlapReq
                                } else {
                                    overlap >= lexicalFallbackMinOverlap
                                }
                                Log.w(TAG, "RAG diag: embedder unreliable; falling back to lexical overlap (overlap=${"%.3f".format(overlap)}) -> accept=$fallbackAccept relaxed=$relaxedLexicalFallback")
                                if (fallbackAccept) filtered.add(candidate)
                                if (filtered.size >= maxResults) break
                                // continue to next candidate
                                continue
                            }
                            Log.d(TAG, "RAG diag: qPreview=[$qPreview] dPreview=[$dPreview] l2=${"%.6f".format(l2)} exactEqual=$exactEqual")
                        } else {
                            Log.d(TAG, "RAG diag: no embedding available for candidate (file=${candidate.fileName} idx=${candidate.chunkIndex})")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "RAG diag: failed to log diagnostics for candidate", e)
                    }
                }
                if (accept) filtered.add(candidate)
                else if (exactEqual && candidate.content.trim() != query.trim()) {
                    // Track suspicious exact-equal cases for potential lexical fallback
                    suspiciousExactEqualCount++
                }
                if (filtered.size >= maxResults) break
            }

            // If we skipped many exact-equal candidates (embedder likely unreliable) and
            // ended up with no results, perform a lexical-only fallback based on word overlap
            // to avoid silently dropping short but relevant memories.
            var results = filtered.distinctBy { it.content.take(100) }.take(maxResults)
            if (results.isEmpty() && suspiciousExactEqualCount > 0 && candidates.isNotEmpty()) {
                Log.w(TAG, "RAG diag: embedder unreliable detected (suspiciousExactEqualCount=$suspiciousExactEqualCount). Falling back to lexical selection.")
                val minimalOverlap = if (relaxedLexicalFallback) 0.0 else 0.10
                // Compute lexical candidates (use >= so equal scores at threshold are considered)
                val lexicalCandidates = candidates.map { cand ->
                    val overlapScore = wordJaccard(query, cand.content)
                    Pair(cand, overlapScore)
                }.filter { it.second >= minimalOverlap }
                    .sortedByDescending { it.second }

                // When relaxed fallback is requested, be permissive: if no candidates pass the
                // minimal overlap filter, fall back to selecting the top candidates by overlap
                // regardless of whether they meet the minimal threshold. This ensures explicit
                // user queries ("what's in my resume") still return candidate memories.
                // If no lexical candidates pass the minimal overlap threshold, be more
                // permissive when either the caller requested a relaxed fallback or when
                // the embedder appears unreliable (suspiciousExactEqualCount>0). In those
                // cases select the top candidates by overlap regardless of the minimal
                // threshold so explicit user queries still get context.
                val finalLexical = if (lexicalCandidates.isEmpty()) {
                    if (relaxedLexicalFallback || suspiciousExactEqualCount > 0) {
                        Log.w(TAG, "RAG diag: selecting top candidates by overlap due to relaxed fallback=${relaxedLexicalFallback} or embedder-unreliable (suspiciousExactEqualCount=$suspiciousExactEqualCount)")
                        candidates.map { cand -> Pair(cand, wordJaccard(query, cand.content)) }
                            .sortedByDescending { it.second }
                    } else {
                        lexicalCandidates
                    }
                } else lexicalCandidates

                results = finalLexical.map { it.first }.distinctBy { it.content.take(100) }.take(maxResults)
                Log.d(TAG, "RAG diag: lexical fallback selected ${results.size} candidates -> overlaps=${finalLexical.take(maxResults).map { "${"%.3f".format(it.second)}" }}")
            }

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
