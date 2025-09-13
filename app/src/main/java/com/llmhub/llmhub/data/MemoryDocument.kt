package com.llmhub.llmhub.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "memory_documents")
data class MemoryDocument(
    @PrimaryKey val id: String,
    val fileName: String,
    val content: String,
    val metadata: String,
    val createdAt: Long,
    val status: String, // PENDING, EMBEDDING_IN_PROGRESS, EMBEDDED, FAILED
    val chunkCount: Int = 0
)
