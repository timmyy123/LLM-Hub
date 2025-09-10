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
}
