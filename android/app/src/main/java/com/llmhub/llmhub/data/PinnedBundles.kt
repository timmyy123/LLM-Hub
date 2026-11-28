package com.llmhub.llmhub.data

import android.content.Context
import java.io.File

/**
 * Parakeet Ultra (moondream's post-trained NVIDIA Parakeet TDT 0.6B v3, 25 European languages) as litert-community's
 * plain LiteRT graphs: log-mel front end, encoder, TDT decoder, tokenizer. Two downloads: the int8 encoders for the CPU
 * (10 s and 30 s pairs: each holds the whole encoder, the 10 s pair only saves the work of padding short audio) and the
 * 30 s float16 encoder for the GPU.
 */
object ParakeetUltraModel {
    const val NAME = "Parakeet Ultra 0.6B (LiteRT)"
    const val GPU_NAME = "Parakeet Ultra 0.6B GPU (LiteRT)"
    private const val BASE = "https://huggingface.co/litert-community/parakeet-ultra/resolve/36d0fe9b069a7f205b6f08d4d9639ba67481abea"

    /** Our builds of the float16 encoders compiled for one chip's NPU (android/tools/parakeet_npu/compile_npu.py). */
    private const val NPU_BASE = "https://huggingface.co/timmyboy26/parakeet-ultra-LiteRT-NPU/resolve/a126af8f248721f367644322e43f44492bcbe4ac"

    private val shared = linkedMapOf(
        "parakeet_ultra_mel_30s.tflite" to 2896076L,
        "parakeet_ultra_mel_10s.tflite" to 1608068L,
        "parakeet_ultra_decoder_i8.tflite" to 17866768L,
        "tokenizer.json" to 1159960L,
    )

    /** File to size; the first is the catalog entry's main file. */
    private val cpuFiles = linkedMapOf(
        "parakeet_ultra_encoder_30s_i8.tflite" to 665458528L,
        "parakeet_ultra_encoder_10s_i8.tflite" to 616306528L,
    ).apply { putAll(shared) }
    /** Only the 30 s float16 encoder: two of them do not fit in a 12 GB tablet's memory on the GPU. */
    private val gpuFiles = linkedMapOf(
        "parakeet_ultra_encoder_30s_f16.tflite" to 1243832704L,
    ).apply { putAll(shared) }

    /** One chip's NPU build: the SoC tag in its file names (as [DeviceInfo.getEmbeddingGemma2NpuTag]), the chip, sizes. */
    data class NpuBuild(val tag: String, val chip: String, val size10: Long, val size30: Long) {
        val name get() = "Parakeet Ultra 0.6B (NPU $chip)"
        fun encoder(seconds: Int) = "parakeet_ultra_encoder_${seconds}s_f16_$tag.tflite"
    }

    val npuBuilds = listOf(
        NpuBuild("Qualcomm_SM8550", "Snapdragon 8 Gen 2", 1213415744L, 1246028096L),
        NpuBuild("Qualcomm_SM8650", "Snapdragon 8 Gen 3", 1213374784L, 1246269760L),
        NpuBuild("Qualcomm_SM8750", "Snapdragon 8 Elite", 1210507584L, 1246540096L),
        NpuBuild("Qualcomm_SM8850", "Snapdragon 8 Elite Gen 5", 1214112064L, 1243877696L),
        NpuBuild("MediaTek_MT6991", "MediaTek Dimensity 9400", 1184894796L, 1210382796L),
        NpuBuild("MediaTek_MT6993", "MediaTek Dimensity 9500", 1186607696L, 1211953168L),
        NpuBuild("Google_Tensor_G5", "Google Tensor G5", 1187793184L, 1212718176L),
        NpuBuild("Google_Tensor_G6", "Google Tensor G6", 1193532384L, 1213787936L),
    )

    fun npuBuild(name: String?): NpuBuild? = npuBuilds.firstOrNull { it.name == name }

    fun isModel(name: String?) = name == NAME || name == GPU_NAME || npuBuild(name) != null
    fun isGpu(name: String?) = name == GPU_NAME
    fun isNpu(name: String?) = npuBuild(name) != null

    /** Local file name to download URL, the catalog entry's main file first. */
    private fun sources(name: String?): Map<String, String> {
        val npu = npuBuild(name)
        val encoders = when {
            npu != null -> listOf(30, 10).associate { npu.encoder(it) to "$NPU_BASE/${npu.encoder(it)}" }
            else -> (if (isGpu(name)) gpuFiles else cpuFiles).keys.filter { "encoder" in it }.associateWith { "$BASE/$it" }
        }
        return encoders + shared.keys.associateWith { "$BASE/$it" }
    }

    fun files(name: String?): Map<String, Long> {
        val npu = npuBuild(name) ?: return if (isGpu(name)) gpuFiles else cpuFiles
        return linkedMapOf(npu.encoder(30) to npu.size30, npu.encoder(10) to npu.size10).apply { putAll(shared) }
    }

    /** Window lengths with their own encoder, shortest first. */
    fun encoderSeconds(name: String?): List<Int> = if (isGpu(name)) listOf(30) else listOf(10, 30)

    /** Encoder file for windows up to 10 s or 30 s. */
    fun encoder(name: String?, seconds: Int): String = npuBuild(name)?.encoder(seconds)
        ?: "parakeet_ultra_encoder_${seconds}s_${if (isGpu(name)) "f16" else "i8"}.tflite"

    fun urls(name: String): List<String> = sources(name).values.toList()
    fun totalBytes(name: String): Long = files(name).values.sum()
}

/**
 * Multi-file models downloaded file by file (the "onnx" multi-file path) whose every file has a pinned size (Parakeet
 * Ultra). A bundle counts as downloaded only when every file has exactly its size.
 */
object PinnedBundles {
    fun isPinned(name: String?) = ParakeetUltraModel.isModel(name)

    /** Local file name to size. */
    fun localFiles(name: String?): Map<String, Long> = if (ParakeetUltraModel.isModel(name)) ParakeetUltraModel.files(name) else emptyMap()

    /** models/<sanitized name>, where the downloader puts multi-file models. */
    fun directory(context: Context, name: String): File =
        File(File(context.filesDir, "models"), name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), ""))

    fun isComplete(context: Context, name: String?): Boolean {
        if (name == null || !isPinned(name)) return false
        val dir = directory(context, name)
        return localFiles(name).all { (file, size) -> File(dir, file).let { it.isFile && it.length() == size } }
    }
}
