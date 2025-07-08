package com.example.llmhub.data

import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import android.content.Context

@Database(
    entities = [ChatEntity::class, MessageEntity::class],
    version = 1,
    exportSchema = false
)
abstract class LlmHubDatabase : RoomDatabase() {
    
    abstract fun chatDao(): ChatDao
    abstract fun messageDao(): MessageDao
    
    companion object {
        @Volatile
        private var INSTANCE: LlmHubDatabase? = null
        
        fun getDatabase(context: Context): LlmHubDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    LlmHubDatabase::class.java,
                    "llmhub_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
} 