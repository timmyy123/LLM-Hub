package com.llmhub.llmhub.mediasearch

import android.app.Application
import android.content.ContentUris
import android.content.Intent
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.litertlm.InputData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class MomentVideo(val id: String, val uri: Uri, val name: String, val durationMs: Long)

data class VideoMoment(val video: MomentVideo, val startMs: Int, val endMs: Int, val score: Float)

/**
 * AI Edge Gallery's Video Moment Finder: each video is split into 2 second windows.
 * A window is two frames plus the audio around those frames, embedded together, then
 * ranked against a text query so the result is a timestamp inside the video.
 */
class VideoMomentViewModel(application: Application) : MediaSearchViewModel(application, "video_moment_prefs") {

    override val maxInputTokens: Int get() = VIDEO_MOMENT_MAX_INPUT_TOKENS

    private val store = MediaIndexStore.forFeature(application, "video_moments")
    private val storeMutex = Mutex()

    private val _videos = MutableStateFlow<List<MomentVideo>>(emptyList())
    val videos: StateFlow<List<MomentVideo>> = _videos.asStateFlow()

    private val _progress = MutableStateFlow(IndexingProgress())
    val progress: StateFlow<IndexingProgress> = _progress.asStateFlow()

    private val _isPaused = MutableStateFlow(false)
    val isPaused: StateFlow<Boolean> = _isPaused.asStateFlow()

    private val _open = MutableStateFlow<MomentVideo?>(null)
    val open: StateFlow<MomentVideo?> = _open.asStateFlow()

    private val _query = MutableStateFlow("")
    val query: StateFlow<String> = _query.asStateFlow()

    private val _results = MutableStateFlow<List<VideoMoment>?>(null)
    val results: StateFlow<List<VideoMoment>?> = _results.asStateFlow()

    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching.asStateFlow()

    private var indexJob: Job? = null
    private var searchJob: Job? = null
    private val failedIds = HashSet<String>()

    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { storeMutex.withLock { store.load() } }
            refreshModels()
            loadSavedVideos()
            if (selectedModel.value != null) loadModel()
        }
    }

    fun addVideos(uris: List<Uri>) {
        val resolver = context.contentResolver
        for (uri in uris) {
            try {
                resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: SecurityException) { }
        }
        val current = _videos.value.toMutableList()
        for (uri in uris) {
            val video = describe(uri) ?: continue
            if (current.none { it.id == video.id }) current.add(0, video)
        }
        _videos.value = current
        saveVideos()
        viewModelScope.launch { updateProgress(); startIndexing() }
    }

    fun useAllVideos() {
        _videos.value = queryDeviceVideos()
        saveVideos()
        viewModelScope.launch { updateProgress(); startIndexing() }
    }

    fun open(video: MomentVideo) {
        _open.value = video
        clearSearch()
    }

    fun closeVideo() {
        _open.value = null
        clearSearch()
    }

    fun onQueryChange(text: String) {
        _query.value = text
        searchJob?.cancel()
        _results.value = null
        _isSearching.value = false
    }

    /** Gallery searches when the query is submitted, not on each keystroke. */
    fun submitSearch() {
        val text = _query.value.trim()
        searchJob?.cancel()
        if (text.isEmpty()) {
            _results.value = null
            _isSearching.value = false
            return
        }
        _isSearching.value = true
        searchJob = viewModelScope.launch {
            val vector = ensureModelLoaded()?.generateEmbedding(text, isQuery = true)
            _results.value = if (vector == null) emptyList() else rank(vector)
            _isSearching.value = false
        }
    }

    fun clearSearch() {
        searchJob?.cancel()
        _query.value = ""
        _results.value = null
        _isSearching.value = false
    }

    fun pause() { _isPaused.value = true }

    fun resume() {
        _isPaused.value = false
        startIndexing()
    }

    fun clearAll() {
        indexJob?.cancel()
        clearSearch()
        viewModelScope.launch(Dispatchers.IO) { storeMutex.withLock { store.clear() } }
        _videos.value = emptyList()
        failedIds.clear()
        saveVideos()
        _progress.value = IndexingProgress()
    }

    fun refresh() {
        refreshModels()
        loadSavedVideos()
    }

    override fun onModelLoaded() = startIndexing()

    override suspend fun onModelUnloading() {
        indexJob?.cancel()
        indexJob?.join()
    }

    private fun loadSavedVideos() {
        val saved = prefString(KEY_URIS)?.split("\n")?.filter { it.isNotBlank() }.orEmpty()
        _videos.value = saved.mapNotNull { describe(Uri.parse(it)) }
        viewModelScope.launch { updateProgress() }
    }

    private fun saveVideos() {
        putPrefString(KEY_URIS, _videos.value.joinToString("\n") { it.uri.toString() })
    }

    private fun describe(uri: Uri): MomentVideo? = try {
        var name = uri.lastPathSegment ?: "video"
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) name = c.getString(0) ?: name
        }
        val duration = MediaMetadataRetriever().run {
            try {
                setDataSource(context, uri)
                extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            } finally { release() }
        }
        if (duration < MIN_DURATION_MS) null else MomentVideo(uri.toString(), uri, name, duration)
    } catch (e: Exception) {
        Log.w(TAG, "Video unavailable: $uri (${e.message})")
        null
    }

    private fun queryDeviceVideos(): List<MomentVideo> {
        val collection = MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        val result = ArrayList<MomentVideo>()
        try {
            context.contentResolver.query(
                collection,
                arrayOf(MediaStore.Video.Media._ID, MediaStore.Video.Media.DISPLAY_NAME, MediaStore.Video.Media.DURATION),
                "${MediaStore.Video.Media.DURATION} >= ?",
                arrayOf(MIN_DURATION_MS.toString()),
                "${MediaStore.Video.Media.DATE_ADDED} DESC"
            )?.use { c ->
                val idCol = c.getColumnIndexOrThrow(MediaStore.Video.Media._ID)
                val nameCol = c.getColumnIndexOrThrow(MediaStore.Video.Media.DISPLAY_NAME)
                val durCol = c.getColumnIndexOrThrow(MediaStore.Video.Media.DURATION)
                while (c.moveToNext()) {
                    val uri = ContentUris.withAppendedId(collection, c.getLong(idCol))
                    result.add(MomentVideo(uri.toString(), uri, c.getString(nameCol) ?: "video", c.getLong(durCol)))
                }
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "No video library access: ${e.message}")
        }
        return result
    }

    private fun completedIds(): Set<String> = store.all.filter { it.startMs == DONE_MARKER }.mapTo(HashSet()) { it.id }

    private suspend fun updateProgress() {
        val done = storeMutex.withLock { completedIds() }
        val videos = _videos.value
        _progress.value = IndexingProgress(videos.count { it.id in done || it.id in failedIds }, videos.size)
    }

    private fun startIndexing() {
        if (indexJob?.isActive == true || _isPaused.value) return
        indexJob = viewModelScope.launch(Dispatchers.Default) {
            val service = ensureModelLoaded() ?: return@launch
            while (isActive && !_isPaused.value) {
                val done = storeMutex.withLock { completedIds() }
                val pending = _videos.value.filter { it.id !in done && it.id !in failedIds }
                if (pending.isEmpty()) break
                for (video in pending) {
                    if (!isActive || _isPaused.value) break
                    indexVideo(video, service)
                    _progress.value = _progress.value.copy(processed = (_progress.value.processed + 1).coerceAtMost(_progress.value.total))
                }
            }
            withContext(Dispatchers.IO) { storeMutex.withLock { store.saveIfDirty() } }
            updateProgress()
            val active = _query.value
            if (active.isNotBlank()) onQueryChange(active)
        }
    }

    private suspend fun indexVideo(video: MomentVideo, service: com.llmhub.llmhub.embedding.LiteRtLmEmbeddingService) {
        val samples = decodeAudio16kMono(context, video.uri, MAX_SECONDS)
        storeMutex.withLock { store.removeIds(setOf(video.id)) }
        val limitMs = minOf(video.durationMs, MAX_SECONDS * 1000L)
        var start = 0L
        var stored = false
        while (start < limitMs) {
            if (_isPaused.value || !currentCoroutineContext().isActive) return
            val end = minOf(start + WINDOW_MS, limitMs)
            if (end - start >= 500) {
                val vector = service.generateMixedEmbedding(windowParts(video.uri, samples, start, end))
                if (vector != null) {
                    storeMutex.withLock { store.put(MediaVector(video.id, start.toInt(), end.toInt(), vector)) }
                    stored = true
                }
            }
            start = end
        }
        if (!stored && samples == null) failedIds.add(video.id)
        storeMutex.withLock {
            store.put(MediaVector(video.id, DONE_MARKER, DONE_MARKER, FloatArray(0)))
            store.saveIfDirty()
        }
    }

    /** Gallery's TAV window: timestamp, audio slice, timestamp, frame, repeated for two frames. */
    private fun windowParts(uri: Uri, samples: FloatArray?, startMs: Long, endMs: Long): List<InputData> {
        val times = listOf(startMs, (endMs - 1).coerceAtLeast(startMs))
        val frames = times.mapNotNull { frameJpegAt(context, uri, it) }
        if (frames.isEmpty()) return emptyList()
        val slices = audioSlices(samples, startMs, endMs, frames.size)
        val hasAudio = slices.any { it != null }
        if (!hasAudio) return frames.map { InputData.Image(it) }
        val parts = ArrayList<InputData>(frames.size * 4)
        frames.forEachIndexed { index, jpeg ->
            val stamp = timestamp(times[index])
            slices.getOrNull(index)?.let { wav ->
                parts.add(InputData.Text(stamp))
                parts.add(InputData.Audio(wav))
            }
            parts.add(InputData.Text(stamp))
            parts.add(InputData.Image(jpeg))
        }
        return parts
    }

    private fun audioSlices(samples: FloatArray?, startMs: Long, endMs: Long, count: Int): List<ByteArray?> {
        if (samples == null || count <= 0) return emptyList()
        val duration = (endMs - startMs).coerceAtLeast(1)
        return (0 until count).map { index ->
            val fromMs = startMs + index * duration / count
            val toMs = startMs + (index + 1) * duration / count
            val from = (fromMs * AUDIO_SAMPLE_RATE / 1000).toInt().coerceIn(0, samples.size)
            val to = (toMs * AUDIO_SAMPLE_RATE / 1000).toInt().coerceIn(from, samples.size)
            if (to - from < AUDIO_SAMPLE_RATE / 4) null else pcm16Wav(samples, from, to)
        }
    }

    private fun timestamp(ms: Long): String {
        val seconds = (ms / 1000).toInt()
        return "%d:%02d".format(seconds / 60, seconds % 60)
    }

    private suspend fun rank(query: FloatArray): List<VideoMoment> = withContext(Dispatchers.Default) {
        val byId = _videos.value.associateBy { it.id }
        val only = _open.value?.id
        storeMutex.withLock {
            store.all.mapNotNull { v ->
                if (v.vector.isEmpty() || (only != null && v.id != only)) return@mapNotNull null
                val video = byId[v.id] ?: return@mapNotNull null
                VideoMoment(video, v.startMs, v.endMs, dot(query, v.vector))
            }
        }.sortedByDescending { it.score }.take(5)
    }

    override fun onCleared() {
        indexJob?.cancel()
        kotlinx.coroutines.runBlocking { storeMutex.withLock { store.saveIfDirty() } }
        super.onCleared()
    }

    private companion object {
        const val TAG = "VideoMoment"
        const val KEY_URIS = "picked_video_uris"
        const val WINDOW_MS = 2_000L
        const val MAX_SECONDS = 600
        const val MIN_DURATION_MS = 1_000L
        const val DONE_MARKER = -1
    }
}
