package com.llmhub.llmhub.mediasearch

import android.app.Application
import android.content.ContentUris
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class AudioAsset(val id: String, val uri: Uri, val name: String, val durationMs: Long)

data class AudioMoment(val startMs: Int, val endMs: Int, val score: Float)

/** [timeline] holds one score per analyzed moment in time order, for the waveform-style strip. */
data class AudioMatch(val asset: AudioAsset, val score: Float, val moments: List<AudioMoment>, val timeline: List<AudioMoment>)

enum class AudioSource { NONE, DEVICE, IMPORTED }

/**
 * Audio Search: finds moments inside audio files from a text description. Like AI Edge Gallery's
 * Video Moment Finder, each file is split into short windows that are embedded separately.
 */
class AudioSearchViewModel(application: Application) : MediaSearchViewModel(application, "audio_search_prefs") {

    private val store = MediaIndexStore.forFeature(application, "audio")
    private val storeMutex = Mutex()

    private val _source = MutableStateFlow(AudioSource.NONE)
    val source: StateFlow<AudioSource> = _source.asStateFlow()

    private val _assets = MutableStateFlow<List<AudioAsset>>(emptyList())
    val assets: StateFlow<List<AudioAsset>> = _assets.asStateFlow()

    private val _progress = MutableStateFlow(IndexingProgress())
    val progress: StateFlow<IndexingProgress> = _progress.asStateFlow()

    private val _isPaused = MutableStateFlow(false)
    val isPaused: StateFlow<Boolean> = _isPaused.asStateFlow()

    private val _query = MutableStateFlow("")
    val query: StateFlow<String> = _query.asStateFlow()

    private val _results = MutableStateFlow<List<AudioMatch>?>(null)
    val results: StateFlow<List<AudioMatch>?> = _results.asStateFlow()

    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching.asStateFlow()

    private var indexJob: Job? = null
    private var searchJob: Job? = null
    private val failedIds = HashSet<String>()

    init {
        _source.value = prefString(KEY_SOURCE)?.let { runCatching { AudioSource.valueOf(it) }.getOrNull() } ?: AudioSource.NONE
        viewModelScope.launch {
            withContext(Dispatchers.IO) { storeMutex.withLock { store.load() } }
            refreshModels()
            refreshAssets()
            if (selectedModel.value != null) loadModel()
        }
    }

    fun useDeviceAudio() {
        setSource(AudioSource.DEVICE)
        viewModelScope.launch { refreshAssets() }
    }

    fun addImportedAudio(uris: List<Uri>) {
        if (uris.isEmpty()) return
        uris.forEach { uri ->
            try {
                context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: SecurityException) { }
        }
        putPrefString(KEY_IMPORTED, (importedUris() + uris.map { it.toString() }).distinct().joinToString("\n"))
        if (_source.value != AudioSource.DEVICE) setSource(AudioSource.IMPORTED)
        viewModelScope.launch { refreshAssets() }
    }

    fun clearAll() {
        viewModelScope.launch {
            indexJob?.cancel()
            indexJob?.join()
            withContext(Dispatchers.IO) { storeMutex.withLock { store.clear() } }
            putPrefString(KEY_IMPORTED, null)
            setSource(AudioSource.NONE)
            failedIds.clear()
            clearSearch()
            _assets.value = emptyList()
            _progress.value = IndexingProgress()
        }
    }

    fun pauseIndexing() { _isPaused.value = true }

    fun resumeIndexing() {
        _isPaused.value = false
        startIndexing()
    }

    fun refresh() {
        refreshModels()
        viewModelScope.launch { refreshAssets() }
    }

    fun onQueryChange(text: String) {
        _query.value = text
        searchJob?.cancel()
        if (text.isBlank()) {
            _results.value = null
            _isSearching.value = false
            return
        }
        _isSearching.value = true
        searchJob = viewModelScope.launch {
            delay(300)
            val queryVector = ensureModelLoaded()?.generateEmbedding(text.trim(), isQuery = true)
            _results.value = if (queryVector == null) emptyList() else rank(queryVector)
            _isSearching.value = false
        }
    }

    fun clearSearch() {
        searchJob?.cancel()
        _query.value = ""
        _results.value = null
        _isSearching.value = false
    }

    override fun onModelLoaded() = startIndexing()

    override suspend fun onModelUnloading() {
        indexJob?.cancel()
        indexJob?.join()
    }

    private suspend fun rank(query: FloatArray): List<AudioMatch> = withContext(Dispatchers.Default) {
        val byId = _assets.value.associateBy { it.id }
        val grouped = storeMutex.withLock {
            store.all.filter { it.vector.isNotEmpty() && it.id in byId }
                .groupBy({ it.id }, { AudioMoment(it.startMs, it.endMs, dot(query, it.vector)) })
        }
        grouped.mapNotNull { (id, moments) ->
            val asset = byId[id] ?: return@mapNotNull null
            val best = moments.sortedByDescending { it.score }
            AudioMatch(asset, best.first().score, best.take(MOMENTS_PER_FILE), moments.sortedBy { it.startMs })
        }.sortedByDescending { it.score }.take(MAX_RESULTS)
    }

    private fun setSource(value: AudioSource) {
        _source.value = value
        putPrefString(KEY_SOURCE, value.name)
    }

    private fun importedUris(): List<String> =
        prefString(KEY_IMPORTED)?.split("\n")?.filter { it.isNotBlank() } ?: emptyList()

    private suspend fun refreshAssets() {
        val assets = withContext(Dispatchers.IO) {
            val imported = importedUris().mapNotNull { describeImported(Uri.parse(it)) }
            when (_source.value) {
                AudioSource.NONE -> emptyList()
                AudioSource.DEVICE -> queryDeviceAudio() + imported
                AudioSource.IMPORTED -> imported
            }.distinctBy { it.id }
        }
        _assets.value = assets
        val liveIds = assets.mapTo(HashSet()) { it.id }
        withContext(Dispatchers.IO) {
            storeMutex.withLock {
                store.removeIds(store.ids - liveIds)
                store.saveIfDirty()
            }
        }
        updateProgress()
        startIndexing()
    }

    private fun describeImported(uri: Uri): AudioAsset? = try {
        var name = uri.lastPathSegment ?: "audio"
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) name = c.getString(0) ?: name
        }
        val duration = android.media.MediaMetadataRetriever().run {
            try {
                setDataSource(context, uri)
                extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            } finally {
                release()
            }
        }
        AudioAsset(uri.toString(), uri, name, duration)
    } catch (e: Exception) {
        Log.w(TAG, "Imported audio unavailable: $uri (${e.message})")
        null
    }

    private fun queryDeviceAudio(): List<AudioAsset> {
        val collection = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
        val result = ArrayList<AudioAsset>()
        try {
            context.contentResolver.query(
                collection,
                arrayOf(MediaStore.Audio.Media._ID, MediaStore.Audio.Media.DISPLAY_NAME, MediaStore.Audio.Media.DURATION),
                "${MediaStore.Audio.Media.DURATION} >= ?",
                arrayOf(MIN_DURATION_MS.toString()),
                "${MediaStore.Audio.Media.DATE_ADDED} DESC"
            )?.use { c ->
                val idCol = c.getColumnIndexOrThrow(MediaStore.Audio.Media._ID)
                val nameCol = c.getColumnIndexOrThrow(MediaStore.Audio.Media.DISPLAY_NAME)
                val durCol = c.getColumnIndexOrThrow(MediaStore.Audio.Media.DURATION)
                while (c.moveToNext()) {
                    val uri = ContentUris.withAppendedId(collection, c.getLong(idCol))
                    result.add(AudioAsset(uri.toString(), uri, c.getString(nameCol) ?: "audio", c.getLong(durCol)))
                }
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "No audio library access: ${e.message}")
        }
        return result
    }

    /** A file counts as analyzed once its completion marker (start = -1) is stored. */
    private fun completedIds(): Set<String> = store.all.filter { it.startMs == DONE_MARKER }.mapTo(HashSet()) { it.id }

    private suspend fun updateProgress() {
        val done = storeMutex.withLock { completedIds() }
        val assets = _assets.value
        _progress.value = IndexingProgress(assets.count { it.id in done || it.id in failedIds }, assets.size)
    }

    private fun startIndexing() {
        if (indexJob?.isActive == true || _isPaused.value) return
        indexJob = viewModelScope.launch(Dispatchers.Default) {
            val service = ensureModelLoaded() ?: return@launch
            while (isActive && !_isPaused.value) {
                val done = storeMutex.withLock { completedIds() }
                val pending = _assets.value.filter { it.id !in done && it.id !in failedIds }
                if (pending.isEmpty()) break
                for (asset in pending) {
                    if (!isActive || _isPaused.value) break
                    indexFile(asset, service)
                    _progress.value = _progress.value.copy(processed = (_progress.value.processed + 1).coerceAtMost(_progress.value.total))
                }
            }
            withContext(Dispatchers.IO) { storeMutex.withLock { store.saveIfDirty() } }
            updateProgress()
            val activeQuery = _query.value
            if (activeQuery.isNotBlank()) onQueryChange(activeQuery)
        }
    }

    private suspend fun indexFile(asset: AudioAsset, service: com.llmhub.llmhub.embedding.LiteRtLmEmbeddingService) {
        val samples = decodeAudio16kMono(context, asset.uri, MAX_SECONDS_PER_FILE)
        if (samples == null) {
            failedIds.add(asset.id)
            return
        }
        // Drop any partial windows from an interrupted run before re-analyzing the file.
        storeMutex.withLock { store.removeIds(setOf(asset.id)) }
        val window = AUDIO_SAMPLE_RATE * WINDOW_MS / 1000
        var start = 0
        while (start < samples.size) {
            if (_isPaused.value || !kotlinx.coroutines.currentCoroutineContext().isActive) return
            val end = minOf(start + window, samples.size)
            if (end - start >= AUDIO_SAMPLE_RATE / 2 && rms(samples, start, end) >= SILENCE_RMS) {
                val vector = service.generateAudioEmbedding(pcm16Wav(samples, start, end))
                if (vector != null) {
                    val startMs = start * 1000 / AUDIO_SAMPLE_RATE
                    val endMs = end * 1000 / AUDIO_SAMPLE_RATE
                    storeMutex.withLock { store.put(MediaVector(asset.id, startMs, endMs, vector)) }
                }
            }
            start = end
        }
        storeMutex.withLock {
            store.put(MediaVector(asset.id, DONE_MARKER, DONE_MARKER, FloatArray(0)))
            store.saveIfDirty()
        }
    }

    override fun onCleared() {
        indexJob?.cancel()
        kotlinx.coroutines.runBlocking { storeMutex.withLock { store.saveIfDirty() } }
        super.onCleared()
    }

    private companion object {
        const val TAG = "AudioSearch"
        const val KEY_SOURCE = "source"
        const val KEY_IMPORTED = "imported_uris"
        const val WINDOW_MS = 5_000
        const val MAX_SECONDS_PER_FILE = 600
        const val MIN_DURATION_MS = 1_000
        const val SILENCE_RMS = 0.003f
        const val DONE_MARKER = -1
        const val MOMENTS_PER_FILE = 3
        const val MAX_RESULTS = 40
    }
}
