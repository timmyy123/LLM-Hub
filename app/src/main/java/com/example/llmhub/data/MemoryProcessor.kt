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
                        // Chunk the document into smaller chunks for embeddings.
                        val chunks = doc.content.split(Regex("\\n\\s*\\n")).filter { it.isNotBlank() }.ifEmpty { listOf(doc.content) }

                        var totalChunksCreated = 0
                        var embeddingFailures = 0
                        // Ensure RAG/embedding initialized
                        val mgr = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)
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
