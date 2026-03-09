package com.llmhub.llmhub.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

@Entity(tableName = "creators")
data class CreatorEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val pctfPrompt: String,
    val description: String,
    val icon: String, // Emoji or Icon name
    val createdAt: Long = System.currentTimeMillis()
)
