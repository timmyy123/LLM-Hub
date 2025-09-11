package com.llmhub.llmhub.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface MemoryDao {
    @Query("SELECT * FROM memory_documents ORDER BY createdAt DESC")
    fun getAllMemory(): Flow<List<MemoryDocument>>

    @Query("SELECT * FROM memory_documents WHERE id = :id")
    suspend fun getById(id: String): MemoryDocument?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(doc: MemoryDocument)

    @Update
    suspend fun update(doc: MemoryDocument)

    @Delete
    suspend fun delete(doc: MemoryDocument)

    @Query("DELETE FROM memory_documents")
    suspend fun deleteAll()

    // Chunk-level embedding persistence
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertChunk(chunk: MemoryChunkEmbedding)

    @Query("SELECT * FROM memory_chunk_embeddings WHERE docId = :docId ORDER BY chunkIndex ASC")
    suspend fun getChunksForDoc(docId: String): List<MemoryChunkEmbedding>

    @Query("SELECT * FROM memory_chunk_embeddings ORDER BY createdAt ASC")
    suspend fun getAllChunks(): List<MemoryChunkEmbedding>

    @Query("DELETE FROM memory_chunk_embeddings WHERE docId = :docId")
    suspend fun deleteChunksForDoc(docId: String)

    @Query("DELETE FROM memory_chunk_embeddings")
    suspend fun deleteAllChunks()
}
