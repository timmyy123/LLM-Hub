package com.llmhub.llmhub.mediasearch

import android.app.Application
import android.content.ContentUris
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class PhotoAsset(val id: String, val uri: Uri, val isVideo: Boolean = false)

data class PhotoMatch(val asset: PhotoAsset, val score: Float)

enum class PhotoSource { NONE, ALL_PHOTOS, SELECTED }

data class IndexingProgress(val processed: Int = 0, val total: Int = 0) {
    val isComplete: Boolean get() = processed >= total
    val percent: Int get() = if (total == 0) 100 else processed * 100 / total
}

/** Photo Search: text and photo-to-photo semantic search over the user's photos (EmbeddingGemma 2). */
class PhotoSearchViewModel(application: Application) : MediaSearchViewModel(application, "photo_search_prefs") {

    private val store = MediaIndexStore.forFeature(application, "photos")
    private val storeMutex = Mutex()

    private val _source = MutableStateFlow(PhotoSource.NONE)
    val source: StateFlow<PhotoSource> = _source.asStateFlow()

    private val _assets = MutableStateFlow<List<PhotoAsset>>(emptyList())
    val assets: StateFlow<List<PhotoAsset>> = _assets.asStateFlow()

    private val _progress = MutableStateFlow(IndexingProgress())
    val progress: StateFlow<IndexingProgress> = _progress.asStateFlow()

    private val _isPaused = MutableStateFlow(false)
    val isPaused: StateFlow<Boolean> = _isPaused.asStateFlow()

    private val _query = MutableStateFlow("")
    val query: StateFlow<String> = _query.asStateFlow()

    /** Null when there is no active text or similar-photo search. */
    private val _results = MutableStateFlow<List<PhotoMatch>?>(null)
    val results: StateFlow<List<PhotoMatch>?> = _results.asStateFlow()

    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching.asStateFlow()

    private val _similarTo = MutableStateFlow<PhotoAsset?>(null)
    val similarTo: StateFlow<PhotoAsset?> = _similarTo.asStateFlow()

    private var indexJob: Job? = null
    private var searchJob: Job? = null
    private val failedIds = HashSet<String>()

    init {
        _source.value = prefString(KEY_SOURCE)?.let { runCatching { PhotoSource.valueOf(it) }.getOrNull() } ?: PhotoSource.NONE
        viewModelScope.launch {
            withContext(Dispatchers.IO) { storeMutex.withLock { store.load() } }
            refreshModels()
            refreshAssets()
            if (selectedModel.value != null) loadModel()
        }
    }

    fun useAllPhotos() {
        setSource(PhotoSource.ALL_PHOTOS)
        viewModelScope.launch { refreshAssets() }
    }

    fun addSelectedPhotos(uris: List<Uri>) {
        if (uris.isEmpty()) return
        uris.forEach { uri ->
            try {
                context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: SecurityException) { }
        }
        val existing = selectedUris()
        putPrefString(KEY_SELECTED, (existing + uris.map { it.toString() }).distinct().joinToString("\n"))
        if (_source.value != PhotoSource.ALL_PHOTOS) setSource(PhotoSource.SELECTED)
        viewModelScope.launch { refreshAssets() }
    }

    fun clearAll() {
        viewModelScope.launch {
            indexJob?.cancel()
            indexJob?.join()
            withContext(Dispatchers.IO) { storeMutex.withLock { store.clear() } }
            putPrefString(KEY_SELECTED, null)
            setSource(PhotoSource.NONE)
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

    fun onQueryChange(text: String) {
        _query.value = text
        _similarTo.value = null
        searchJob?.cancel()
        if (text.isBlank()) {
            _results.value = null
            _isSearching.value = false
            return
        }
        _isSearching.value = true
        searchJob = viewModelScope.launch {
            delay(300)
            val service = ensureModelLoaded()
            val queryVector = service?.generateEmbedding(text.trim(), isQuery = true)
            _results.value = if (queryVector == null) emptyList() else rank(queryVector, exclude = null)
            _isSearching.value = false
        }
    }

    fun findSimilar(asset: PhotoAsset) {
        searchJob?.cancel()
        _query.value = ""
        _similarTo.value = asset
        _isSearching.value = true
        searchJob = viewModelScope.launch {
            val vector = storeMutex.withLock { store.vectorsFor(asset.id).firstOrNull()?.vector }
                ?: ensureModelLoaded()?.let { service ->
                    val frames = if (asset.isVideo) loadVideoKeyframes(context, asset.uri, 1024) else listOfNotNull(loadJpegForEmbedding(context, asset.uri, 1024))
                    service.generateImagesEmbedding(frames)
                }
            _results.value = if (vector == null) emptyList() else rank(vector, exclude = asset.id)
            _isSearching.value = false
        }
    }

    fun clearSearch() {
        searchJob?.cancel()
        _query.value = ""
        _similarTo.value = null
        _results.value = null
        _isSearching.value = false
    }

    /** Re-read the photo list (e.g. after a permission grant) and analyze anything new. */
    fun refresh() {
        refreshModels()
        viewModelScope.launch { refreshAssets() }
    }

    override fun onModelLoaded() = startIndexing()

    override suspend fun onModelUnloading() {
        indexJob?.cancel()
        indexJob?.join()
    }

    private suspend fun rank(query: FloatArray, exclude: String?): List<PhotoMatch> = withContext(Dispatchers.Default) {
        val byId = _assets.value.associateBy { it.id }
        val scored = storeMutex.withLock {
            store.all.mapNotNull { v ->
                if (v.id == exclude) return@mapNotNull null
                byId[v.id]?.let { PhotoMatch(it, dot(query, v.vector)) }
            }
        }
        scored.sortedByDescending { it.score }.take(MAX_RESULTS)
    }

    private fun setSource(value: PhotoSource) {
        _source.value = value
        putPrefString(KEY_SOURCE, value.name)
    }

    private fun selectedUris(): List<String> =
        prefString(KEY_SELECTED)?.split("\n")?.filter { it.isNotBlank() } ?: emptyList()

    private suspend fun refreshAssets() {
        val assets = withContext(Dispatchers.IO) {
            when (_source.value) {
                PhotoSource.NONE -> emptyList()
                PhotoSource.ALL_PHOTOS -> queryDeviceMedia() + selectedUris().map { describeSelected(Uri.parse(it)) }
                PhotoSource.SELECTED -> selectedUris().map { describeSelected(Uri.parse(it)) }
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

    /** Photos and videos, newest first, matching Gallery's Instant Media Search. */
    private fun queryDeviceMedia(): List<PhotoAsset> {
        val result = ArrayList<PhotoAsset>()
        result.addAll(queryCollection(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, isVideo = false))
        result.addAll(queryCollection(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, isVideo = true))
        return result
    }

    private fun queryCollection(collection: Uri, isVideo: Boolean): List<PhotoAsset> {
        val result = ArrayList<PhotoAsset>()
        try {
            context.contentResolver.query(
                collection,
                arrayOf(MediaStore.MediaColumns._ID, MediaStore.MediaColumns.DATE_ADDED),
                null, null,
                "${MediaStore.MediaColumns.DATE_ADDED} DESC"
            )?.use { cursor ->
                val idCol = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns._ID)
                while (cursor.moveToNext()) {
                    val uri = ContentUris.withAppendedId(collection, cursor.getLong(idCol))
                    result.add(PhotoAsset(uri.toString(), uri, isVideo))
                }
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "No library access for $collection: ${e.message}")
        }
        return result
    }

    private fun describeSelected(uri: Uri): PhotoAsset {
        val type = context.contentResolver.getType(uri).orEmpty()
        return PhotoAsset(uri.toString(), uri, type.startsWith("video"))
    }

    private suspend fun updateProgress() {
        val indexed = storeMutex.withLock { store.ids }
        val assets = _assets.value
        _progress.value = IndexingProgress(
            processed = assets.count { it.id in indexed || it.id in failedIds },
            total = assets.size
        )
    }

    private fun startIndexing() {
        if (indexJob?.isActive == true || _isPaused.value) return
        indexJob = viewModelScope.launch(Dispatchers.Default) {
            val service = ensureModelLoaded() ?: return@launch
            var sinceSave = 0
            while (isActive && !_isPaused.value) {
                // New photos can arrive mid-pass (picker, permission grant), so loop until none are pending.
                val indexed = storeMutex.withLock { store.ids }
                val pending = _assets.value.filter { it.id !in indexed && it.id !in failedIds }
                if (pending.isEmpty()) break
                for (next in pending) {
                    if (!isActive || _isPaused.value) break
                    val frames = if (next.isVideo) loadVideoKeyframes(context, next.uri) else listOfNotNull(loadJpegForEmbedding(context, next.uri))
                    val vector = service.generateImagesEmbedding(frames)
                    storeMutex.withLock {
                        if (vector != null) store.put(MediaVector(next.id, 0, 0, vector)) else failedIds.add(next.id)
                        if (++sinceSave >= SAVE_EVERY) {
                            store.saveIfDirty()
                            sinceSave = 0
                        }
                    }
                    _progress.value = _progress.value.copy(processed = (_progress.value.processed + 1).coerceAtMost(_progress.value.total))
                }
            }
            withContext(Dispatchers.IO) { storeMutex.withLock { store.saveIfDirty() } }
            updateProgress()
            // A search typed while photos were still being analyzed may now have more matches.
            val activeQuery = _query.first()
            if (activeQuery.isNotBlank()) onQueryChange(activeQuery)
        }
    }

    override fun onCleared() {
        indexJob?.cancel()
        kotlinx.coroutines.runBlocking { storeMutex.withLock { store.saveIfDirty() } }
        super.onCleared()
    }

    private companion object {
        const val TAG = "PhotoSearch"
        const val KEY_SOURCE = "source"
        const val KEY_SELECTED = "selected_uris"
        const val SAVE_EVERY = 20
        const val MAX_RESULTS = 120
    }
}
