package com.llmhub.llmhub.data

import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import android.content.Context

@Database(
    entities = [ChatEntity::class, MessageEntity::class],
    version = 2,
    exportSchema = false
)
abstract class LlmHubDatabase : RoomDatabase() {
    
    abstract fun chatDao(): ChatDao
    abstract fun messageDao(): MessageDao
    
    companion object {
        @Volatile
        private var INSTANCE: LlmHubDatabase? = null
        
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                // Add the new columns for attachment file info
                database.execSQL("ALTER TABLE MessageEntity ADD COLUMN attachmentFileName TEXT")
                database.execSQL("ALTER TABLE MessageEntity ADD COLUMN attachmentFileSize INTEGER")
            }
        }
        
        fun getDatabase(context: Context): LlmHubDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    LlmHubDatabase::class.java,
                    "llmhub_database"
                ).addMigrations(MIGRATION_1_2).build()
                INSTANCE = instance
                instance
            }
        }
    }
} 