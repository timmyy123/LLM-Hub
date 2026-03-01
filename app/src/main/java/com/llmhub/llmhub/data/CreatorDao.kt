package com.llmhub.llmhub.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface CreatorDao {
    @Query("SELECT * FROM creators ORDER BY createdAt DESC")
    fun getAllCreators(): Flow<List<CreatorEntity>>

    @Query("SELECT * FROM creators WHERE id = :id")
    suspend fun getCreatorById(id: String): CreatorEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCreator(creator: CreatorEntity)

    @Delete
    suspend fun deleteCreator(creator: CreatorEntity)
    
    @Query("DELETE FROM creators WHERE id = :id")
    suspend fun deleteCreatorById(id: String)
}
