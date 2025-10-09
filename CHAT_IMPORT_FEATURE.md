# Chat History Import to Memory Feature

## Overview
Users can now import their chat history directly into global memory. This allows the AI to reference past conversations as context when answering questions.

## How It Works

### User Flow
1. Open Settings
2. Enable RAG/Embeddings and select an embedding model
3. Click "Manage Memory"
4. Click "Import Chat History" button
5. Select which chats to import (checkboxes)
6. Click "Import Selected"
7. Chat history is converted to text and added to memory

### Technical Implementation

#### 1. UI Components (`SettingsScreen.kt`)

**Import Button**
- Added after "Save to Memory" button
- Only enabled when embeddings are active
- Opens the `ChatImportDialog`

**ChatImportDialog Composable**
- Displays list of all non-empty chats
- Checkbox selection for each chat
- Shows chat name and last updated date
- "Import Selected" button (disabled if no chats selected)

#### 2. Data Flow

**Chat Text Extraction**
```kotlin
val messages = db.messageDao().getMessagesForChatSync(chat.id)
val chatText = messages.joinToString("\n\n") { msg ->
    val role = if (msg.isUser) "User" else "Assistant"
    "$role: ${msg.content}"
}
```

**Memory Document Creation**
```kotlin
val doc = MemoryDocument(
    id = "mem_chat_${chat.id}_${timestamp}",
    fileName = "Chat: ${chat.name}",
    content = chatText,  // Full conversation text
    metadata = "chat_import",
    createdAt = timestamp,
    status = "PENDING",
    chunkCount = 0
)
```

#### 3. Processing
- Uses existing `MemoryProcessor` to chunk and embed the chat text
- Chat conversations are treated like any other memory document
- Chunks are stored and indexed for RAG retrieval

## Format Example

Imported chat text format:
```
User: What's the weather today?