package com.llmhub.llmhub.data

import android.content.Context
import java.io.File

/** English b6369a24 export, compatible with PocketTTS.cpp's original graphs. */
object PocketTtsModel {
    const val NAME = "Pocket TTS (English, ONNX)"
    const val REVISION = "58a6d00cf13d239b6748cb0769f35c580a8f606c"
    const val BASE_URL = "https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/$REVISION"
    val coreFiles = linkedMapOf(
        "flow_lm_main_int8.onnx" to 76341627L,
        "flow_lm_flow_int8.onnx" to 9962530L,
        "mimi_decoder_int8.onnx" to 22684077L,
        "mimi_encoder.onnx" to 73165554L,
        "text_conditioner.onnx" to 16388363L,
        "tokenizer.model" to 59339L,
        "LICENSE" to 18655L
    )
    fun url(name: String) = "$BASE_URL/${if (name == "tokenizer.model") "" else "onnx/"}$name"
    fun directory(context: Context) = File(context.filesDir, "models/Pocket_TTS_English_ONNX")
    fun isComplete(context: Context) = coreFiles.all { (name, size) ->
        File(directory(context), name).let { it.isFile && it.length() == size }
    }
    fun voicesDirectory(context: Context) = File(context.filesDir, "pocket-voices").apply { mkdirs() }
    // Only app-created UUID filenames can be selected. Reference clips stay private.
    private val voiceName = Regex("[0-9a-f-]{36}\\.(wav|mp3|flac)")
    fun voiceFile(context: Context, id: String): File? =
        if (!voiceName.matches(id)) null else File(voicesDirectory(context), id).takeIf { it.isFile }
    fun voices(context: Context) = voicesDirectory(context).listFiles()?.filter {
        voiceName.matches(it.name) && it.isFile && File(it.path + ".txt").isFile
    }?.sortedBy { it.name } ?: emptyList()
    fun label(file: File): String = File(file.path + ".txt").takeIf { it.isFile }?.readText()?.take(80) ?: file.name
    internal val operationLock = Any()
    fun renameVoice(context: Context, id: String, label: String) = synchronized(operationLock) {
        val name = label.trim()
        require(name.isNotEmpty() && name.length <= 80)
        val voice = requireNotNull(voiceFile(context, id))
        val labelFile = File(voice.path + ".txt")
        check(labelFile.isFile)
        val atomic = android.util.AtomicFile(labelFile)
        val stream = atomic.startWrite()
        try {
            stream.write(name.toByteArray(Charsets.UTF_8))
            atomic.finishWrite(stream)
        } catch (e: Exception) {
            atomic.failWrite(stream)
            throw e
        }
    }
    fun deleteVoice(context: Context, id: String) = synchronized(operationLock) {
        require(voiceName.matches(id))
        voiceFile(context, id)?.let { it.delete(); File(it.path + ".txt").delete() }
        // Cached embeddings/KV contain the voice too; remove them on deletion.
        listOf("emb", "kv").forEach { extension ->
            File(voicesDirectory(context), ".cache/${id.substringBeforeLast('.')}.$extension").delete()
        }
    }
}
