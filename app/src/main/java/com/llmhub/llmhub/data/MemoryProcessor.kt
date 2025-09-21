package com.llmhub.llmhub.data

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.MutableStateFlow
import com.llmhub.llmhub.embedding.RagServiceManager

class MemoryProcessor(private val context: Context, private val db: LlmHubDatabase) {
    private val TAG = "MemoryProcessor"
    private val ragManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)
    
    /**
     * Create smart chunks with semantic boundaries and overlap for better context preservation
     * Same logic as used in RagService for consistency
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
            // Single paragraph or no clear paragraph breaks - use sentence-based chunking
            val sentences = text.split(Regex("""(?<=[.!?])\s+""")).filter { it.trim().isNotEmpty() }
            var currentChunk = StringBuilder()
            
            for (sentence in sentences) {
                val trimmedSentence = sentence.trim()
                
                if (currentChunk.length + trimmedSentence.length > maxChunkSize && currentChunk.isNotEmpty()) {
                    chunks.add(currentChunk.toString().trim())
                    
                    // Start new chunk with overlap
                    val overlapText = getOverlapText(currentChunk.toString(), overlapSize)
                    currentChunk = StringBuilder(overlapText)
                    
                    if (overlapText.isNotEmpty()) {
                        currentChunk.append(" ")
                    }
                }
                
                if (trimmedSentence.length > maxChunkSize) {
                    // Very long sentence - split by words
                    val subChunks = splitLongText(trimmedSentence, maxChunkSize, overlapSize)
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
                    currentChunk.append(trimmedSentence).append(" ")
                }
            }
            
            if (currentChunk.isNotEmpty()) {
                chunks.add(currentChunk.toString().trim())
            }
        }
        
        return chunks.filter { it.trim().isNotEmpty() }
    }
    
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

    companion object {
        // Public state that composables can observe to know when embedding work is active
        val processing = MutableStateFlow(false)
    }

    fun processPending() {
        CoroutineScope(Dispatchers.IO).launch {
            // mark processor as running
            processing.value = true
            try {
                // Ensure RAG initialization attempts
                val job = ragManager.initializeAsync()
                job.join()

                val list = db.memoryDao().getAllMemory().first()
                for (doc in list.filter { it.status == "PENDING" || it.status == "FAILED" }) {
                    try {
                        // mark in-progress
                        db.memoryDao().update(doc.copy(status = "EMBEDDING_IN_PROGRESS"))
                            Log.d(TAG, "Processing memory doc id=${doc.id} file='${doc.fileName}' status=${doc.status}")
                        var totalChunksCreated = 0
                        var embeddingFailures = 0
                        // Ensure RAG/embedding initialized
                        val mgr = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)
                        
                        // Use the same smart chunking as chat documents for consistency
                        val chunks = createSmartChunks(doc.content, maxChunkSize = 800, overlapSize = 100)
                        val initJob = mgr.initializeAsync()
                        initJob.join()

                                for ((index, chunkText) in chunks.withIndex()) {
                            try {
                                        // Re-check that the document still exists in DB. If the user deleted
                                        // or replaced the memory while processing was running, abort to avoid
                                        // re-inserting chunks or re-populating the in-memory RAG index.
                                        val stillExists = db.memoryDao().getById(doc.id)
                                        if (stillExists == null) {
                                            Log.w(TAG, "Skipping processing for deleted doc ${doc.id}")
                                            break
                                        }
                                val emb = mgr.generateEmbedding(chunkText)
                                if (emb == null) {
                                    Log.w(TAG, "Failed to generate embedding for chunk $index of doc ${doc.id}")
                                    embeddingFailures++
                                    continue
                                }

                                // Persist chunk embedding to DB
                                val chunkId = "${doc.id}_$index"
                                val embeddingBytes = com.llmhub.llmhub.data.floatArrayToByteArray(emb)
                                val chunkEntity = com.llmhub.llmhub.data.MemoryChunkEmbedding(
                                    id = chunkId,
                                    docId = doc.id,
                                    fileName = doc.fileName,
                                    chunkIndex = index,
                                    content = chunkText,
                                    embedding = embeddingBytes,
                                    embeddingModel = try { com.llmhub.llmhub.data.ThemePreferences(context).selectedEmbeddingModel.first() } catch (_: Exception) { null },
                                    createdAt = System.currentTimeMillis()
                                )
                                db.memoryDao().insertChunk(chunkEntity)

                                // Add precomputed chunk to RAG in-memory index
                                // If adding fails due to RAG not ready or doc deleted concurrently,
                                // count as a failure but continue processing other chunks.
                                val added = try {
                                    // Double-check doc presence once more before adding embedding
                                    val stillThere = db.memoryDao().getById(doc.id)
                                    if (stillThere == null) {
                                        Log.w(TAG, "Doc ${doc.id} deleted before adding chunk $index to RAG; skipping remaining chunks")
                                        false
                                    } else {
                                        ragManager.addGlobalDocumentChunk(doc.id, chunkText, doc.fileName, index, emb, chunkEntity.embeddingModel, doc.metadata)
                                    }
                                } catch (e: Exception) {
                                    Log.w(TAG, "Error adding chunk $index for doc ${doc.id} to RAG: ${e.message}")
                                    false
                                }
                                if (added) totalChunksCreated++ else embeddingFailures++

                            } catch (e: Exception) {
                                Log.e(TAG, "Error embedding/persisting chunk $index for doc ${doc.id}: ${e.message}")
                                embeddingFailures++
                            }
                        }

                        if (embeddingFailures == 0 && totalChunksCreated > 0) {
                            db.memoryDao().update(doc.copy(status = "EMBEDDED", chunkCount = totalChunksCreated))
                            Log.d(TAG, "MemoryProcessor: successfully embedded doc ${doc.id}; global chunkCount=${ragManager.getDocumentCount("__global_memory__")}")
                        } else if (totalChunksCreated > 0) {
                            db.memoryDao().update(doc.copy(status = "EMBEDDED", chunkCount = totalChunksCreated))
                            Log.w(TAG, "MemoryProcessor: partially embedded doc ${doc.id}; created=$totalChunksCreated failed=$embeddingFailures")
                        } else {
                            Log.i(TAG, "MemoryProcessor: embedding failed for doc ${doc.id} - leaving as PENDING for retry")
                            db.memoryDao().update(doc.copy(status = "PENDING"))
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed processing doc ${doc.id}", e)
                        db.memoryDao().update(doc.copy(status = "FAILED"))
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Memory processing failed", e)
            } finally {
                // mark processor as stopped
                processing.value = false
            }
        }
    }
}
