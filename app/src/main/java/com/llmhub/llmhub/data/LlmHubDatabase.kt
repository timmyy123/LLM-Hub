package com.llmhub.llmhub.data

import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import android.content.Context

@Database(
    entities = [ChatEntity::class, MessageEntity::class, MemoryDocument::class, com.llmhub.llmhub.data.MemoryChunkEmbedding::class, CreatorEntity::class],
    version = 5,
    exportSchema = false
)
abstract class LlmHubDatabase : RoomDatabase() {
    
    abstract fun chatDao(): ChatDao
    abstract fun messageDao(): MessageDao
    abstract fun memoryDao(): MemoryDao
    abstract fun creatorDao(): CreatorDao
    
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

        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(database: SupportSQLiteDatabase) {
                // Create the memory_documents table
                database.execSQL(
                    "CREATE TABLE IF NOT EXISTS `memory_documents` (`id` TEXT NOT NULL, `fileName` TEXT NOT NULL, `content` TEXT NOT NULL, `metadata` TEXT NOT NULL, `createdAt` INTEGER NOT NULL, `status` TEXT NOT NULL, `chunkCount` INTEGER NOT NULL, PRIMARY KEY(`id`))"
                )
            }
        }

        private val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(database: SupportSQLiteDatabase) {
                // Create the memory_chunk_embeddings table
                database.execSQL(
                    "CREATE TABLE IF NOT EXISTS `memory_chunk_embeddings` (`id` TEXT NOT NULL, `docId` TEXT NOT NULL, `fileName` TEXT NOT NULL, `chunkIndex` INTEGER NOT NULL, `content` TEXT NOT NULL, `embedding` BLOB NOT NULL, `embeddingModel` TEXT, `createdAt` INTEGER NOT NULL, PRIMARY KEY(`id`))"
                )
            }
        }

        private val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(database: SupportSQLiteDatabase) {
                // Create creators table
                database.execSQL(
                    "CREATE TABLE IF NOT EXISTS `creators` (`id` TEXT NOT NULL, `name` TEXT NOT NULL, `pctfPrompt` TEXT NOT NULL, `description` TEXT NOT NULL, `icon` TEXT NOT NULL, `createdAt` INTEGER NOT NULL, PRIMARY KEY(`id`))"
                )
                // Add creatorId to ChatEntity
                database.execSQL("ALTER TABLE chats ADD COLUMN creatorId TEXT")
            }
        }
        
        fun getDatabase(context: Context): LlmHubDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    LlmHubDatabase::class.java,
                    "llmhub_database"
                )
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5)
                .fallbackToDestructiveMigration() // Optional: useful for dev, but we have migrations now
                .build()
                INSTANCE = instance
                instance
            }
        }
    }
} 