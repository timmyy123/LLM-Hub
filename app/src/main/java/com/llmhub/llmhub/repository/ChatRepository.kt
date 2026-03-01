package com.llmhub.llmhub.repository

import com.llmhub.llmhub.data.*
import kotlinx.coroutines.flow.Flow

data class ChatWithLastMessage(
    val chat: ChatEntity,
    val lastMessage: MessageEntity?
)

class ChatRepository(
    private val chatDao: ChatDao,
    private val messageDao: MessageDao,
    private val creatorDao: CreatorDao
) {
    
    fun getAllChats(): Flow<List<ChatEntity>> {
        return chatDao.getAllChats()
    }

    /** Chats that have at least one message. Use for the drawer. */
    fun getActiveChats(): Flow<List<ChatEntity>> {
        return chatDao.getNonEmptyChats()
    }

    fun getAllCreators(): Flow<List<CreatorEntity>> {
        return creatorDao.getAllCreators()
    }
    
    suspend fun getCreatorById(id: String): CreatorEntity? {
        return creatorDao.getCreatorById(id)
    }

    suspend fun insertCreator(creator: CreatorEntity) {
        creatorDao.insertCreator(creator)
    }

    suspend fun deleteCreator(creator: CreatorEntity) {
        creatorDao.deleteCreator(creator)
    }
    
    suspend fun getAllChatsWithLastMessage(): List<ChatWithLastMessage> {
        val chats = chatDao.getAllChats()
        // For now, return a simple flow that emits once
        return emptyList() // Will implement proper flow later
    }
    
    fun getMessagesForChat(chatId: String): Flow<List<MessageEntity>> {
        return messageDao.getMessagesForChat(chatId)
    }
    
    suspend fun getMessagesForChatSync(chatId: String): List<MessageEntity> {
        return messageDao.getMessagesForChatSync(chatId)
    }
    
    suspend fun createNewChat(title: String, modelName: String, creatorId: String? = null): String {
        val chat = ChatEntity(
            title = title,
            modelName = modelName,
            creatorId = creatorId
        )
        chatDao.insertChat(chat)
        return chat.id
    }
    
    suspend fun updateChatTitle(chatId: String, newTitle: String) {
        val chat = chatDao.getChatById(chatId)
        chat?.let {
            chatDao.updateChat(it.copy(title = newTitle, updatedAt = System.currentTimeMillis()))
        }
    }
    
    suspend fun updateChatModel(chatId: String, modelName: String) {
        val chat = chatDao.getChatById(chatId)
        chat?.let {
            chatDao.updateChat(it.copy(modelName = modelName, updatedAt = System.currentTimeMillis()))
        }
    }
    
    suspend fun deleteChat(chatId: String) {
        chatDao.deleteChatById(chatId)
        messageDao.deleteMessagesForChat(chatId)
    }
    
    suspend fun clearMessagesForChat(chatId: String) {
        messageDao.deleteMessagesForChat(chatId)
    }
    
    suspend fun deleteAllChats() {
        chatDao.deleteAllChats()
    }
    
    suspend fun addMessage(
        chatId: String, 
        content: String, 
        isFromUser: Boolean, 
        attachmentPath: String? = null, 
        attachmentType: String? = null,
        attachmentFileName: String? = null,
        attachmentFileSize: Long? = null
    ): String {
        // Normalize content by trimming trailing whitespace/newlines to avoid large empty selection areas
        val safeContent = content.trimEnd()

        val message = MessageEntity(
            chatId = chatId,
            content = safeContent,
            isFromUser = isFromUser,
            attachmentPath = attachmentPath,
            attachmentType = attachmentType,
            attachmentFileName = attachmentFileName,
            attachmentFileSize = attachmentFileSize
        )
        messageDao.insertMessage(message)
        
        // Update chat's updatedAt timestamp
        val chat = chatDao.getChatById(chatId)
        chat?.let {
            chatDao.updateChat(it.copy(updatedAt = System.currentTimeMillis()))
        }
        return message.id
    }

    suspend fun updateMessageContent(messageId: String, newContent: String) {
    val msg = messageDao.getMessageById(messageId) ?: return
    val safeContent = newContent.trimEnd()
    if (msg.content == safeContent) return
    messageDao.updateMessage(msg.copy(content = safeContent))
    }
    
    suspend fun updateMessageStats(messageId: String, tokenCount: Int, tokensPerSecond: Double) {
        val msg = messageDao.getMessageById(messageId) ?: return
        messageDao.updateMessage(msg.copy(tokenCount = tokenCount, tokensPerSecond = tokensPerSecond))
    }
    
    suspend fun getChatById(chatId: String): ChatEntity? {
        return chatDao.getChatById(chatId)
    }
    
    suspend fun deleteMessagesAfter(chatId: String, afterTimestamp: Long) {
        messageDao.deleteMessagesAfter(chatId, afterTimestamp)
    }
    
    suspend fun deleteMessageById(messageId: String) {
        val message = messageDao.getMessageById(messageId)
        message?.let {
            messageDao.deleteMessage(it)
        }
    }
}

