package com.llmhub.llmhub.data

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.llmhub.llmhub.utils.AudioConversionUtils
import com.llmhub.llmhub.utils.loadInferenceBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * Image and audio memories for multimodal embedding models (EmbeddingGemma 2).
 *
 * A media memory is a [MemoryDocument] whose metadata is [TYPE_IMAGE] or [TYPE_AUDIO]; the media
 * bytes live in [mediaFile] and [MemoryDocument.content] holds a text label plus the user's note,
 * which is what gets injected into chat prompts when the memory is retrieved.
 */
object MemoryMedia {
    const val TYPE_IMAGE = "image"
    const val TYPE_AUDIO = "audio"

    private const val IMAGE_LABEL = "[Image memory"
    private const val AUDIO_LABEL = "[Audio memory"

    /** Cross-modal (text query vs. image/audio) cosine scores run lower than text-to-text ones. */
    const val MEDIA_SIMILARITY_THRESHOLD = 0.30f

    /** 16 kHz mono float32 WAV: 44-byte header + 4 bytes per sample. */
    private const val MAX_AUDIO_SECONDS = 120
    private const val WAV_HEADER_BYTES = 44
    private const val MAX_IMAGE_EDGE = 1024

    fun isMediaType(metadata: String?): Boolean = metadata == TYPE_IMAGE || metadata == TYPE_AUDIO

    fun isMediaContent(content: String): Boolean =
        content.startsWith(IMAGE_LABEL) || content.startsWith(AUDIO_LABEL)

    fun buildContent(type: String, fileName: String, note: String?): String {
        val label = if (type == TYPE_IMAGE) IMAGE_LABEL else AUDIO_LABEL
        val trimmedNote = note?.trim().orEmpty()
        return if (trimmedNote.isEmpty()) "$label: $fileName]" else "$label: $fileName]\n$trimmedNote"
    }

    /** The user's note without the generated label line. */
    fun noteFromContent(content: String): String? =
        content.substringAfter('\n', "").trim().takeIf { isMediaContent(content) && it.isNotEmpty() }

    /** An image/audio memory shown under the assistant reply that used it. */
    data class MediaReference(val type: String, val docId: String, val fileName: String)

    fun encodeReferences(refs: List<MediaReference>): String? =
        refs.distinctBy { it.docId }.takeIf { it.isNotEmpty() }
            ?.joinToString("\n") { "${it.type}|${it.docId}|${it.fileName}" }

    fun decodeReferences(encoded: String?): List<MediaReference> =
        encoded.orEmpty().lines().mapNotNull { line ->
            val parts = line.split('|', limit = 3)
            if (parts.size == 3 && isMediaType(parts[0])) MediaReference(parts[0], parts[1], parts[2]) else null
        }

    /** Map retrieved chunk texts back to their media memories (a media memory is a single chunk). */
    suspend fun referencesForChunks(db: LlmHubDatabase, chunkContents: List<String>): List<MediaReference> {
        val mediaContents = chunkContents.map { it.trim() }.filter { isMediaContent(it) }.toSet()
        if (mediaContents.isEmpty()) return emptyList()
        return db.memoryDao().getAllMemory().first()
            .filter { isMediaType(it.metadata) && it.content.trim() in mediaContents }
            .map { MediaReference(it.metadata, it.id, it.fileName) }
    }

    fun mediaFile(context: Context, docId: String): File =
        File(File(context.filesDir, "memory_media").apply { mkdirs() }, docId)

    fun deleteMedia(context: Context, docId: String) {
        mediaFile(context, docId).delete()
    }

    fun deleteAllMedia(context: Context) {
        File(context.filesDir, "memory_media").deleteRecursively()
    }

    /** Decode any supported image (HEIC, WebP, ...) and re-encode as a bounded JPEG for LiteRT-LM. */
    suspend fun loadImageAsJpeg(context: Context, uri: Uri): ByteArray? {
        val bitmap = loadInferenceBitmap(context, uri) ?: return null
        return withContext(Dispatchers.Default) {
            val scale = MAX_IMAGE_EDGE.toFloat() / maxOf(bitmap.width, bitmap.height)
            val scaled = if (scale < 1f) {
                Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
            } else bitmap
            ByteArrayOutputStream().use { out ->
                scaled.compress(Bitmap.CompressFormat.JPEG, 90, out)
                out.toByteArray()
            }
        }
    }

    /** Convert any audio file to 16 kHz mono WAV, the format EmbeddingGemma 2 expects. */
    suspend fun loadAudioAsWav(context: Context, uri: Uri): ByteArray? =
        AudioConversionUtils.convertUriToFloat32Wav(context, uri)?.let { trimWav(it) }

    /** Keep the first [MAX_AUDIO_SECONDS] of a 16 kHz mono float32 WAV produced by the app. */
    fun trimWav(wav: ByteArray): ByteArray {
        val maxBytes = WAV_HEADER_BYTES + MAX_AUDIO_SECONDS * 16000 * 4
        if (wav.size <= maxBytes) return wav
        if (String(wav, 36, 4, Charsets.US_ASCII) != "data") return wav
        val trimmed = wav.copyOf(maxBytes)
        val dataSize = maxBytes - WAV_HEADER_BYTES
        java.nio.ByteBuffer.wrap(trimmed).order(java.nio.ByteOrder.LITTLE_ENDIAN).apply {
            putInt(4, 36 + dataSize)
            putInt(40, dataSize)
        }
        return trimmed
    }
}
