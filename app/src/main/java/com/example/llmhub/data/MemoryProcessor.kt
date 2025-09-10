package com.llmhub.llmhub.data

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.first
import com.llmhub.llmhub.embedding.RagServiceManager

class MemoryProcessor(private val context: Context, private val db: LlmHubDatabase) {
    private val TAG = "MemoryProcessor"
    private val ragManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)

    fun processPending() {
        CoroutineScope(Dispatchers.IO).launch {
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
                        val success = ragManager.addGlobalDocument(doc.content, doc.fileName, doc.metadata)
                        if (success) {
                                val count = ragManager.getDocumentCount("__global_memory__")
                                Log.d(TAG, "MemoryProcessor: successfully embedded doc ${doc.id}; global chunkCount=$count")
                                db.memoryDao().update(doc.copy(status = "EMBEDDED", chunkCount = count))
                            Log.d(TAG, "Embedded memory doc ${doc.id}")
                        } else {
                            // If RAG service isn't ready (embeddings disabled), leave as PENDING so it can retry later
                            Log.i(TAG, "RAG service not ready or embedding failed for doc ${doc.id} - leaving as PENDING for retry")
                            // Keep as PENDING (don't mark FAILED) so UI/processor can retry when embeddings are available
                            db.memoryDao().update(doc.copy(status = "PENDING"))
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed processing doc ${doc.id}", e)
                        db.memoryDao().update(doc.copy(status = "FAILED"))
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Memory processing failed", e)
            }
        }
    }
}
