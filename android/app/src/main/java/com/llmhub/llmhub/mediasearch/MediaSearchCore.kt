package com.llmhub.llmhub.mediasearch

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaMetadataRetriever
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import androidx.core.graphics.drawable.toBitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import coil.imageLoader
import coil.request.ImageRequest
import coil.request.SuccessResult
import coil.size.Scale
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.hasCompleteDownloadedBundle
import com.llmhub.llmhub.embedding.LiteRtLmEmbeddingService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "MediaSearch"

/** EmbeddingGemma 2 settings used by AI Edge Gallery's Instant Media Search. */
internal const val MEDIA_SEARCH_VISION_TOKENS = 70
internal const val MEDIA_SEARCH_MAX_INPUT_TOKENS = 256

/** Video Moment Finder windows interleave frames and audio, so they need a longer context. */
internal const val VIDEO_MOMENT_MAX_INPUT_TOKENS = 1024

/** One embedded item: a photo, or an audio moment covering [startMs]..[endMs] of a file. */
class MediaVector(val id: String, val startMs: Int, val endMs: Int, val vector: FloatArray)

/** Binary on-disk store of [MediaVector]s, keyed by id + start. Not thread-safe; guard with a Mutex. */
class MediaIndexStore(private val file: File) {
    private val records = LinkedHashMap<String, MediaVector>()
    private var dirty = false

    val all: Collection<MediaVector> get() = records.values
    val ids: Set<String> get() = records.values.mapTo(HashSet()) { it.id }

    fun load() {
        records.clear()
        if (!file.exists()) return
        try {
            DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
                if (input.readInt() != MAGIC) return
                repeat(input.readInt()) {
                    val id = input.readUTF()
                    val start = input.readInt()
                    val end = input.readInt()
                    val vector = FloatArray(input.readInt()) { input.readFloat() }
                    records[key(id, start)] = MediaVector(id, start, end, vector)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Index ${file.name} unreadable, starting fresh: ${e.message}")
            records.clear()
        }
    }

    fun put(item: MediaVector) {
        records[key(item.id, item.startMs)] = item
        dirty = true
    }

    fun removeIds(ids: Set<String>) {
        if (records.values.removeAll { it.id in ids }) dirty = true
    }

    fun vectorsFor(id: String): List<MediaVector> = records.values.filter { it.id == id }

    fun clear() {
        records.clear()
        file.delete()
        dirty = false
    }

    fun saveIfDirty() {
        if (!dirty) return
        file.parentFile?.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        DataOutputStream(BufferedOutputStream(tmp.outputStream())).use { out ->
            out.writeInt(MAGIC)
            out.writeInt(records.size)
            for (r in records.values) {
                out.writeUTF(r.id)
                out.writeInt(r.startMs)
                out.writeInt(r.endMs)
                out.writeInt(r.vector.size)
                r.vector.forEach { out.writeFloat(it) }
            }
        }
        tmp.renameTo(file)
        dirty = false
    }

    private fun key(id: String, start: Int) = "$id#$start"

    companion object {
        private const val MAGIC = 0x4C484D31 // "LHM1"

        fun forFeature(context: Context, name: String) =
            MediaIndexStore(File(File(context.filesDir, "media_search"), "$name.idx"))
    }
}

/** Embeddings are L2-normalized, so the dot product is the cosine similarity. */
internal fun dot(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return -1f
    var sum = 0f
    for (i in a.indices) sum += a[i] * b[i]
    return sum
}

/** Decode an image at a bounded size and encode it as JPEG for the embedder. */
internal suspend fun loadJpegForEmbedding(context: Context, uri: Uri, maxEdge: Int = 512): ByteArray? =
    withContext(Dispatchers.IO) {
        val request = ImageRequest.Builder(context)
            .data(uri)
            .size(maxEdge, maxEdge)
            .scale(Scale.FIT)
            .allowHardware(false)
            .build()
        val bitmap = (context.imageLoader.execute(request) as? SuccessResult)?.drawable?.toBitmap()
            ?: return@withContext null
        ByteArrayOutputStream().use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            out.toByteArray()
        }
    }

/**
 * First frame and, for clips longer than a second, the frame near the end. Same two-keyframe
 * approach AI Edge Gallery uses so a video fits the 256-token input (70 vision tokens each).
 */
internal suspend fun loadVideoKeyframes(context: Context, uri: Uri, maxEdge: Int = 512): List<ByteArray> =
    withContext(Dispatchers.IO) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val timesUs = mutableListOf(0L)
            if (durationMs > 1_000) timesUs.add((durationMs - 500).coerceAtLeast(0) * 1_000)
            timesUs.mapNotNull { timeUs ->
                val frame = retriever.getScaledFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC, maxEdge, maxEdge)
                    ?: return@mapNotNull null
                val bytes = ByteArrayOutputStream().use { out ->
                    frame.compress(Bitmap.CompressFormat.JPEG, 90, out)
                    out.toByteArray()
                }
                frame.recycle()
                bytes
            }
        } catch (e: Exception) {
            Log.w(TAG, "Video frames failed for $uri: ${e.message}")
            emptyList()
        } finally {
            try { retriever.release() } catch (_: Exception) { }
        }
    }

internal fun frameJpegAt(context: Context, uri: Uri, timeMs: Long, maxEdge: Int = 512): ByteArray? =
    try {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val frame = retriever.getScaledFrameAtTime(
                timeMs * 1000L,
                MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                maxEdge,
                maxEdge
            ) ?: return null
            val bytes = ByteArrayOutputStream().use { out ->
                frame.compress(Bitmap.CompressFormat.JPEG, 80, out)
                out.toByteArray()
            }
            frame.recycle()
            bytes
        } finally {
            retriever.release()
        }
    } catch (e: Exception) {
        Log.w(TAG, "Frame at $timeMs failed for $uri: ${e.message}")
        null
    }

internal const val AUDIO_SAMPLE_RATE = 16_000

/**
 * Decode the first [maxSeconds] of any audio file to 16 kHz mono floats, streaming so long
 * recordings never need to fit in memory whole.
 */
internal suspend fun decodeAudio16kMono(context: Context, uri: Uri, maxSeconds: Int): FloatArray? =
    withContext(Dispatchers.IO) {
        val extractor = MediaExtractor()
        var decoder: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: return@withContext null
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            decoder = codec
            codec.configure(format, null, null, 0)
            codec.start()

            var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var floatPcm = false
            val out = FloatList(AUDIO_SAMPLE_RATE * 60)
            var resampleCursor = 0.0
            val info = MediaCodec.BufferInfo()
            var inputDone = false
            val maxOut = maxSeconds * AUDIO_SAMPLE_RATE

            while (out.size < maxOut) {
                if (!inputDone) {
                    val inIndex = codec.dequeueInputBuffer(10_000)
                    if (inIndex >= 0) {
                        val size = extractor.readSampleData(codec.getInputBuffer(inIndex)!!, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIndex = codec.dequeueOutputBuffer(info, 10_000)
                if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val f = codec.outputFormat
                    sampleRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    floatPcm = f.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                        f.getInteger(MediaFormat.KEY_PCM_ENCODING) == android.media.AudioFormat.ENCODING_PCM_FLOAT
                } else if (outIndex >= 0) {
                    val buf = codec.getOutputBuffer(outIndex)!!.order(ByteOrder.LITTLE_ENDIAN)
                    buf.position(info.offset)
                    buf.limit(info.offset + info.size)
                    val step = sampleRate.toDouble() / AUDIO_SAMPLE_RATE
                    val bytesPerSample = if (floatPcm) 4 else 2
                    val frames = info.size / (bytesPerSample * channels)
                    // Linear-skip resampling to 16 kHz while downmixing; adequate for embeddings.
                    while (resampleCursor < frames && out.size < maxOut) {
                        val frame = resampleCursor.toInt()
                        var sum = 0f
                        for (c in 0 until channels) {
                            val pos = info.offset + (frame * channels + c) * bytesPerSample
                            sum += if (floatPcm) buf.getFloat(pos) else buf.getShort(pos) / 32768f
                        }
                        out.add(sum / channels)
                        resampleCursor += step
                    }
                    resampleCursor -= frames
                    codec.releaseOutputBuffer(outIndex, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
            out.toArray().takeIf { it.isNotEmpty() }
        } catch (e: Exception) {
            Log.w(TAG, "Audio decode failed for $uri: ${e.message}")
            null
        } finally {
            try { decoder?.stop() } catch (_: Exception) { }
            try { decoder?.release() } catch (_: Exception) { }
            extractor.release()
        }
    }

/** 16 kHz mono PCM16 WAV of samples[from, to). */
internal fun pcm16Wav(samples: FloatArray, from: Int, to: Int): ByteArray {
    val count = (to - from).coerceAtLeast(0)
    val dataSize = count * 2
    val buf = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
    buf.put("RIFF".toByteArray()); buf.putInt(36 + dataSize); buf.put("WAVE".toByteArray())
    buf.put("fmt ".toByteArray()); buf.putInt(16); buf.putShort(1); buf.putShort(1)
    buf.putInt(AUDIO_SAMPLE_RATE); buf.putInt(AUDIO_SAMPLE_RATE * 2); buf.putShort(2); buf.putShort(16)
    buf.put("data".toByteArray()); buf.putInt(dataSize)
    for (i in from until to) buf.putShort((samples[i].coerceIn(-1f, 1f) * 32767f).toInt().toShort())
    return buf.array()
}

internal fun rms(samples: FloatArray, from: Int, to: Int): Float {
    if (to <= from) return 0f
    var sum = 0.0
    for (i in from until to) sum += samples[i] * samples[i]
    return kotlin.math.sqrt(sum / (to - from)).toFloat()
}

private class FloatList(initial: Int) {
    private var data = FloatArray(initial)
    var size = 0
        private set

    fun add(v: Float) {
        if (size == data.size) data = data.copyOf(data.size * 2)
        data[size++] = v
    }

    fun toArray() = data.copyOf(size)
}

/**
 * Model selection + EmbeddingGemma 2 loading shared by Photo Search and Audio Search. The engine
 * is configured like AI Edge Gallery's media search (70 vision tokens, 256 max input tokens).
 */
abstract class MediaSearchViewModel(application: Application, private val prefsName: String) :
    AndroidViewModel(application) {

    protected val context: Context get() = getApplication()
    private val prefs = application.getSharedPreferences(prefsName, Context.MODE_PRIVATE)

    private val _downloadedModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val downloadedModels: StateFlow<List<LLMModel>> = _downloadedModels.asStateFlow()

    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()

    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()

    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()

    private val _modelError = MutableStateFlow(false)
    val modelError: StateFlow<Boolean> = _modelError.asStateFlow()

    protected var embedder: LiteRtLmEmbeddingService? = null
        private set
    private val modelMutex = Mutex()

    fun refreshModels() {
        val models = ModelData.embeddingGemma2Models.filter { it.hasCompleteDownloadedBundle(context) }
        _downloadedModels.value = models
        val saved = prefs.getString(KEY_MODEL, null)
        val current = _selectedModel.value
        _selectedModel.value = models.firstOrNull { it.name == current?.name }
            ?: models.firstOrNull { it.name == saved }
            // Prefer the chip-specific NPU build when it is downloaded.
            ?: models.firstOrNull { ModelData.isEmbeddingGemma2NpuModel(it) }
            ?: models.firstOrNull()
    }

    fun selectModel(model: LLMModel) {
        if (_selectedModel.value?.name == model.name) return
        prefs.edit().putString(KEY_MODEL, model.name).apply()
        _selectedModel.value = model
        viewModelScope.launch {
            unloadModelInternal()
            loadModel()
        }
    }

    fun loadModel() {
        viewModelScope.launch { ensureModelLoaded() }
    }

    fun unloadModel() {
        viewModelScope.launch {
            onModelUnloading()
            unloadModelInternal()
        }
    }

    protected suspend fun ensureModelLoaded(): LiteRtLmEmbeddingService? = modelMutex.withLock {
        embedder?.let { return@withLock it }
        val model = _selectedModel.value ?: return@withLock null
        _isLoadingModel.value = true
        _modelError.value = false
        val service = LiteRtLmEmbeddingService(
            context, model,
            enableMedia = true,
            visionTokensPerImage = MEDIA_SEARCH_VISION_TOKENS,
            maxInputLength = maxInputTokens
        )
        val ok = service.initialize()
        _isLoadingModel.value = false
        if (ok) {
            embedder = service
            _isModelLoaded.value = true
            onModelLoaded()
            service
        } else {
            _modelError.value = true
            null
        }
    }

    private suspend fun unloadModelInternal() = modelMutex.withLock {
        embedder?.close()
        embedder = null
        _isModelLoaded.value = false
    }

    /** Instant Media Search uses 256. Video Moment Finder overrides this with 1024. */
    protected open val maxInputTokens: Int get() = MEDIA_SEARCH_MAX_INPUT_TOKENS

    /** Called after the engine becomes ready (e.g. to resume indexing). */
    protected open fun onModelLoaded() {}

    /** Called before the engine is released (e.g. to stop indexing). */
    protected open suspend fun onModelUnloading() {}

    override fun onCleared() {
        val service = embedder
        embedder = null
        kotlinx.coroutines.CoroutineScope(Dispatchers.IO + NonCancellable).launch { service?.close() }
        super.onCleared()
    }

    protected fun prefString(key: String): String? = prefs.getString(key, null)
    protected fun putPrefString(key: String, value: String?) = prefs.edit().putString(key, value).apply()

    private companion object {
        const val KEY_MODEL = "selected_model_name"
    }
}
