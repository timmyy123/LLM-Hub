package com.llmhub.llmhub.data

import android.content.Context
import java.io.File
import java.io.RandomAccessFile

/** Reads GGUF metadata only; model tensors are never loaded. Includes llama.cpp's output layer. */
object GgufLayerLimits {
    const val UNKNOWN = 999

    fun forModel(context: Context, model: LLMModel): Int? {
        if (model.modelFormat != "gguf") return null
        val root = File(context.filesDir, "models")
        val folderName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        val folder = File(root, folderName).takeIf { it.isDirectory } ?: root
        val file = sequenceOf(File(folder, model.localFileName()), File(root, model.localFileName()))
            .firstOrNull { it.isFile }
            ?: folder.listFiles()?.firstOrNull {
                it.isFile && it.extension.equals("gguf", true) &&
                    !it.name.contains("mmproj", true) && !it.name.contains("projector", true)
            }
            ?: return null
        return runCatching { readLimit(file) }.getOrNull()
    }

    private fun readLimit(file: File): Int? = RandomAccessFile(file, "r").use { input ->
        fun u32(): Long {
            val a = input.readUnsignedByte().toLong()
            val b = input.readUnsignedByte().toLong()
            val c = input.readUnsignedByte().toLong()
            val d = input.readUnsignedByte().toLong()
            return a or (b shl 8) or (c shl 16) or (d shl 24)
        }
        fun u64(): Long {
            val low = u32()
            val high = u32()
            require(high <= Int.MAX_VALUE) { "GGUF length too large" }
            return low or (high shl 32)
        }
        fun skip(bytes: Long) {
            require(bytes >= 0 && bytes <= input.length() - input.filePointer) { "Invalid GGUF length" }
            input.seek(input.filePointer + bytes)
        }
        fun string(): String {
            val length = u64()
            require(length <= 1_048_576 && length <= input.length() - input.filePointer)
            val bytes = ByteArray(length.toInt())
            input.readFully(bytes)
            return bytes.toString(Charsets.UTF_8)
        }
        fun skipValue(type: Long) {
            when (type.toInt()) {
                0, 1, 7 -> skip(1)
                2, 3 -> skip(2)
                4, 5, 6 -> skip(4)
                8 -> skip(u64())
                10, 11, 12 -> skip(8)
                9 -> {
                    val itemType = u32().toInt()
                    val count = u64()
                    val width = when (itemType) {
                        0, 1, 7 -> 1L
                        2, 3 -> 2L
                        4, 5, 6 -> 4L
                        10, 11, 12 -> 8L
                        else -> 0L
                    }
                    if (width > 0) {
                        require(count <= (input.length() - input.filePointer) / width)
                        skip(count * width)
                    } else {
                        require(itemType == 8 && count <= 1_000_000)
                        repeat(count.toInt()) { skip(u64()) }
                    }
                }
                else -> error("Unsupported GGUF metadata type $type")
            }
        }
        val magic = ByteArray(4)
        input.readFully(magic)
        if (!magic.contentEquals(byteArrayOf(71, 71, 85, 70))) return@use null
        if (u32() !in 2L..3L) return@use null
        u64() // tensor count
        val keyCount = u64()
        require(keyCount <= 1_000_000)
        var architecture: String? = null
        val blockCounts = mutableMapOf<String, Long>()
        repeat(keyCount.toInt()) {
            val key = string()
            val type = u32()
            when {
                key == "general.architecture" && type == 8L -> architecture = string()
                key.endsWith(".block_count") && type in 4L..5L -> blockCounts[key] = u32()
                key.endsWith(".block_count") && type in 10L..11L -> blockCounts[key] = u64()
                else -> skipValue(type)
            }
            val count = architecture?.let { blockCounts["$it.block_count"] }
            if (count != null && count in 1..998) {
                // Large tokenizer arrays often follow the block count. No need to scan them.
                return@use count.toInt() + 1
            }
        }
        val count = architecture?.let { blockCounts["$it.block_count"] }
            ?: blockCounts.values.singleOrNull()
            ?: return@use null
        count.takeIf { it in 1..998 }?.toInt()?.plus(1)
    }
}
