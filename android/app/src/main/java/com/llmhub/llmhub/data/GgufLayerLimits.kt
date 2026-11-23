package com.llmhub.llmhub.data

import android.content.Context
import java.io.File
import java.io.RandomAccessFile

/** Reads GGUF metadata only; model tensors are never loaded. Layer limits include the output layer. */
object GgufLayerLimits {
    const val UNKNOWN = 999

    fun forModel(context: Context, model: LLMModel): Int? {
        val file = modelFile(context, model) ?: return null
        return runCatching { readLimit(file, "block_count", 1) }.getOrNull()
    }

    fun contextForModel(context: Context, model: LLMModel): Int? {
        val file = modelFile(context, model) ?: return null
        return runCatching { readLimit(file, "context_length", 0) }.getOrNull()
    }

    /** True for a mixture-of-experts GGUF: its header lists more than one expert per layer. */
    fun isMoe(context: Context, model: LLMModel): Boolean {
        val file = modelFile(context, model) ?: return false
        return (runCatching { readLimit(file, "expert_count", 0) }.getOrNull() ?: 0) > 1
    }

    /**
     * Weight types the Hexagon NPU multiplies by in MUL_MAT_ID, the op flash streaming reads the prompt with
     * (ggml-hexagon htp-kparams.cpp). Values are ggml_type: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
     * Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ4_NL, BF16, MXFP4, NVFP4, Q1_0, Q2_0. Keep in step with that list on a
     * llama.cpp bump.
     */
    private val NPU_EXPERT_TYPES = setOf(0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 20, 30, 39, 40, 41, 42)

    /**
     * True when flash streaming cannot read the prompt with this MoE GGUF on the NPU: an expert weight type
     * (for example Gemma 4's IQ2_XS and IQ3_XXS experts) is one the NPU cannot multiply by, so the engine
     * refuses the NPU and streams on the CPU. Reads only the GGUF header.
     */
    fun npuCannotStreamExperts(context: Context, model: LLMModel): Boolean =
        modelFile(context, model)?.let { npuCannotStreamExperts(it) } ?: false

    internal fun npuCannotStreamExperts(file: File): Boolean {
        val types = runCatching { readExpertTypes(file) }.getOrNull() ?: return false
        return types.isNotEmpty() && !NPU_EXPERT_TYPES.containsAll(types)
    }

    /** The NPU is left out for this model: flash streaming is on (its saved config) and its experts cannot run there. */
    suspend fun npuHiddenForFlashStreaming(context: Context, model: LLMModel): Boolean =
        isMoe(context, model) &&
            (ModelPreferences(context).getModelConfig(model.name)?.flashStreaming ?: true) &&
            npuCannotStreamExperts(context, model)

    private fun modelFile(context: Context, model: LLMModel): File? {
        if (model.modelFormat != "gguf") return null
        val root = File(context.filesDir, "models")
        val folderName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        val folder = File(root, folderName).takeIf { it.isDirectory } ?: root
        return sequenceOf(File(folder, model.localFileName()), File(root, model.localFileName()))
            .firstOrNull { it.isFile }
            ?: folder.listFiles()?.firstOrNull {
                it.isFile && it.extension.equals("gguf", true) &&
                    !it.name.contains("mmproj", true) && !it.name.contains("projector", true)
            }
            ?: return null
    }

    /** ggml types of the expert weight tensors (`*_exps.weight`), read from the header's tensor list. */
    private fun readExpertTypes(file: File): Set<Int>? = java.io.BufferedInputStream(file.inputStream(), 1 shl 16).use { input ->
        val length = file.length()
        var position = 0L
        fun byte(): Int {
            val value = input.read()
            require(value >= 0) { "GGUF ended early" }
            position++
            return value
        }
        fun u32(): Long = (0 until 4).fold(0L) { acc, i -> acc or (byte().toLong() shl (8 * i)) }
        fun u64(): Long {
            val low = u32()
            val high = u32()
            require(high <= Int.MAX_VALUE) { "GGUF length too large" }
            return low or (high shl 32)
        }
        fun skip(bytes: Long) {
            require(bytes >= 0 && bytes <= length - position) { "Invalid GGUF length" }
            var left = bytes
            while (left > 0) {
                val skipped = input.skip(left)
                if (skipped > 0) left -= skipped else { byte(); left--; position-- }
            }
            position += bytes
        }
        fun string(): String {
            val size = u64()
            require(size <= 1_048_576 && size <= length - position)
            val bytes = ByteArray(size.toInt())
            var read = 0
            while (read < bytes.size) {
                val n = input.read(bytes, read, bytes.size - read)
                require(n > 0) { "GGUF ended early" }
                read += n
            }
            position += size
            return bytes.toString(Charsets.UTF_8)
        }
        fun width(type: Int): Long = when (type) {
            0, 1, 7 -> 1L
            2, 3 -> 2L
            4, 5, 6 -> 4L
            10, 11, 12 -> 8L
            else -> 0L
        }
        fun skipValue(type: Int) {
            when (type) {
                8 -> skip(u64())
                9 -> {
                    val itemType = u32().toInt()
                    val count = u64()
                    val itemWidth = width(itemType)
                    if (itemWidth > 0) {
                        require(count <= (length - position) / itemWidth)
                        skip(count * itemWidth)
                    } else {
                        require(itemType == 8 && count <= 10_000_000)
                        repeat(count.toInt()) { skip(u64()) }
                    }
                }
                else -> width(type).takeIf { it > 0 }?.let { skip(it) } ?: error("Unsupported GGUF metadata type $type")
            }
        }
        val magic = ByteArray(4) { byte().toByte() }
        if (!magic.contentEquals(byteArrayOf(71, 71, 85, 70))) return@use null
        if (u32() !in 2L..3L) return@use null
        val tensorCount = u64()
        val keyCount = u64()
        require(keyCount <= 1_000_000 && tensorCount <= 1_000_000)
        repeat(keyCount.toInt()) {
            skip(u64()) // key
            skipValue(u32().toInt())
        }
        val types = mutableSetOf<Int>()
        repeat(tensorCount.toInt()) {
            val name = string()
            val dims = u32()
            require(dims <= 8)
            skip(dims * 8)
            val type = u32().toInt()
            skip(8) // data offset
            if (name.endsWith("_exps.weight")) types += type
        }
        types
    }

    private fun readLimit(file: File, keySuffix: String, offset: Int): Int? = RandomAccessFile(file, "r").use { input ->
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
        val metadataValues = mutableMapOf<String, Long>()
        repeat(keyCount.toInt()) {
            val key = string()
            val type = u32()
            when {
                key == "general.architecture" && type == 8L -> architecture = string()
                key.endsWith(".$keySuffix") && type in 4L..5L -> metadataValues[key] = u32()
                key.endsWith(".$keySuffix") && type in 10L..11L -> metadataValues[key] = u64()
                else -> skipValue(type)
            }
            val count = architecture?.let { metadataValues["$it.$keySuffix"] }
            if (count != null && count in 1..(Int.MAX_VALUE.toLong() - offset)) {
                // Large tokenizer arrays often follow these model limits. No need to scan them.
                return@use count.toInt() + offset
            }
        }
        val count = architecture?.let { metadataValues["$it.$keySuffix"] }
            ?: metadataValues.values.singleOrNull()
            ?: return@use null
        count.takeIf { it in 1..(Int.MAX_VALUE.toLong() - offset) }?.toInt()?.plus(offset)
    }
}

/** Use the GGUF header for downloaded GGUFs; keep catalog limits for every other format. */
fun LLMModel.effectiveContextWindow(context: Context): Int =
    GgufLayerLimits.contextForModel(context, this) ?: contextWindowSize.coerceAtLeast(1)
