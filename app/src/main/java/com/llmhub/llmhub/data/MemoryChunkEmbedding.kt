package com.llmhub.llmhub.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.nio.ByteBuffer
import java.nio.ByteOrder

@Entity(tableName = "memory_chunk_embeddings")
data class MemoryChunkEmbedding(
    @PrimaryKey val id: String,
    val docId: String,
    val fileName: String,
    val chunkIndex: Int,
    val content: String,
    val embedding: ByteArray,
    val embeddingModel: String?,
    val createdAt: Long
)

// Helpers for serializing FloatArray <-> ByteArray
fun floatArrayToByteArray(arr: FloatArray): ByteArray {
    val bb = ByteBuffer.allocate(arr.size * 4)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    for (f in arr) bb.putFloat(f)
    return bb.array()
}

fun byteArrayToFloatArray(bytes: ByteArray): FloatArray {
    val fb = ByteBuffer.wrap(bytes)
    fb.order(ByteOrder.LITTLE_ENDIAN)
    val out = FloatArray(bytes.size / 4)
    var i = 0
    while (fb.remaining() >= 4 && i < out.size) {
        out[i++] = fb.getFloat()
    }
    return out
}
