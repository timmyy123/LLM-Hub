package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.InpaintKind
import com.llmhub.llmhub.data.InpaintModels
import com.llmhub.llmhub.imagegen.BoundedBitmapDecoder
import com.llmhub.llmhub.imagegen.LiteRtInpainter
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer

/** One brush stroke in photo pixels. */
data class BrushStroke(val points: List<Offset>, val radius: Float)

/** What the eraser is doing: loading the model, or erasing region [done] of [total]. */
sealed class InpaintStatus {
    object Idle : InpaintStatus()
    object Loading : InpaintStatus()
    data class Erasing(val done: Int, val total: Int) : InpaintStatus()
}

/**
 * Object Eraser state: the photo, the brush strokes, the chosen model and the running erase. The model stays
 * loaded between erases; [release] (screen left) and [onCleared] stop a running erase and close it.
 */
class ImageInpaintViewModel(application: Application) : AndroidViewModel(application) {

    private val appContext: Context = application.applicationContext
    private val prefs = appContext.getSharedPreferences("image_inpaint_prefs", Context.MODE_PRIVATE)

    var installed by mutableStateOf<Map<InpaintKind, File>>(emptyMap())
        private set
    var selectedKind by mutableStateOf(
        runCatching { InpaintKind.valueOf(prefs.getString(KEY_KIND, null) ?: "") }.getOrDefault(InpaintKind.MIGAN)
    )
        private set

    /** The photo being edited (the latest erase result). */
    var photo by mutableStateOf<Bitmap?>(null)
        private set
    /** The photo before the latest erase, for compare and undo. */
    var previous by mutableStateOf<Bitmap?>(null)
        private set
    var isLoadingPhoto by mutableStateOf(false)
        private set
    /** True once the current photo has been erased at least once (Save and Share show then). */
    var hasEdits by mutableStateOf(false)
        private set
    /** False until the first check of which models are downloaded has finished. */
    var installedChecked by mutableStateOf(false)
        private set

    val strokes = mutableStateListOf<BrushStroke>()
    var brushSizeDp by mutableStateOf(prefs.getFloat(KEY_BRUSH, DEFAULT_BRUSH_DP))
        private set

    private val _status = MutableStateFlow<InpaintStatus>(InpaintStatus.Idle)
    val status: StateFlow<InpaintStatus> = _status
    private val _onCpu = MutableStateFlow(false)
    /** True while the loaded model runs on the CPU (the GPU could not run it), which is much slower. */
    val onCpu: StateFlow<Boolean> = _onCpu

    private val _messages = MutableSharedFlow<String>(extraBufferCapacity = 4)
    val messages: SharedFlow<String> = _messages

    private val engineLock = Mutex()
    private var engine: LiteRtInpainter? = null
    private var job: Job? = null

    val isBusy: Boolean get() = job?.isActive == true

    init {
        refreshInstalled()
    }

    /** Re-reads which models are downloaded (on resume, e.g. back from the download screen). */
    fun refreshInstalled() {
        viewModelScope.launch {
            val found = withContext(Dispatchers.IO) { InpaintModels.installed(appContext) }
            installed = found
            installedChecked = true
            if (selectedKind !in found) found.keys.firstOrNull()?.let { selectedKind = it }
        }
    }

    fun selectKind(kind: InpaintKind) {
        if (kind !in installed || isBusy) return
        selectedKind = kind
        prefs.edit().putString(KEY_KIND, kind.name).apply()
    }

    fun setBrushSize(dp: Float) {
        brushSizeDp = dp
        prefs.edit().putFloat(KEY_BRUSH, dp).apply()
    }

    fun loadPhoto(uri: Uri) {
        if (isBusy) return
        isLoadingPhoto = true
        viewModelScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                runCatching { decode(uri) }.onFailure { Log.e(TAG, "Could not open the photo", it) }.getOrNull()
            }
            isLoadingPhoto = false
            if (bitmap == null) {
                _messages.tryEmit(appContext.getString(R.string.image_inpainting_open_failed))
                return@launch
            }
            photo = bitmap
            previous = null
            hasEdits = false
            strokes.clear()
        }
    }

    fun closePhoto() {
        if (isBusy) return
        photo = null
        previous = null
        hasEdits = false
        strokes.clear()
    }

    fun addStroke(stroke: BrushStroke) {
        if (!isBusy && stroke.points.isNotEmpty()) strokes.add(stroke)
    }

    /** Removes the last stroke; with no strokes left, goes back to the photo before the latest erase. */
    fun undo() {
        if (isBusy) return
        if (strokes.isNotEmpty()) {
            strokes.removeAt(strokes.lastIndex)
        } else {
            previous?.let { photo = it; previous = null }
        }
    }

    val canUndo: Boolean get() = strokes.isNotEmpty() || previous != null

    fun clearStrokes() {
        if (!isBusy) strokes.clear()
    }

    fun erase() {
        if (isBusy) return
        val source = photo ?: return
        val painted = strokes.toList().takeIf { it.isNotEmpty() } ?: return
        val kind = selectedKind
        val file = installed[kind] ?: return
        job = viewModelScope.launch {
            try {
                _status.value = InpaintStatus.Loading
                val result = withContext(Dispatchers.Default) {
                    engineLock.withLock {
                        val inpainter = engine?.takeIf { it.kind == kind } ?: run {
                            engine?.close()
                            engine = null
                            LiteRtInpainter.create(file, kind).also { engine = it }
                        }
                        _onCpu.value = !inpainter.isGpu
                        val mask = rasterize(painted, source.width, source.height)
                        inpainter.inpaint(source, mask) { done, total ->
                            _status.value = InpaintStatus.Erasing(done, total)
                            _onCpu.value = !inpainter.isGpu
                        }
                    }
                }
                previous = source
                photo = result
                hasEdits = true
                strokes.clear()
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                Log.e(TAG, "Erase failed", t)
                _messages.tryEmit(appContext.getString(R.string.image_inpainting_failed, t.message ?: t.javaClass.simpleName))
            } finally {
                _status.value = InpaintStatus.Idle
            }
        }
    }

    /** Stops a running erase and closes the model once its current pass returns. */
    fun release() {
        val running = job
        job = null
        running?.cancel()
        _status.value = InpaintStatus.Idle
        cleanupScope.launch {
            running?.join()
            engineLock.withLock {
                engine?.close()
                engine = null
            }
        }
    }

    override fun onCleared() {
        release()
        super.onCleared()
    }

    fun saveToGallery() {
        val bitmap = photo ?: return
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) { save(bitmap) }
            _messages.tryEmit(appContext.getString(if (saved) R.string.image_inpainting_saved else R.string.image_inpainting_save_failed))
        }
    }

    /** Writes the photo to the cache and returns a share chooser, or null if that failed. */
    suspend fun shareIntent(): Intent? {
        val bitmap = photo ?: return null
        return withContext(Dispatchers.IO) {
            try {
                val dir = File(appContext.cacheDir, "inpaint").apply { mkdirs() }
                dir.listFiles()?.forEach { it.delete() }
                val file = File(dir, "erased_${System.currentTimeMillis()}.jpg")
                file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                val uri = FileProvider.getUriForFile(appContext, "${appContext.packageName}.fileprovider", file)
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "image/jpeg"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                Intent.createChooser(send, appContext.getString(R.string.image_inpainting_share))
            } catch (e: Exception) {
                Log.e(TAG, "Failed to share", e)
                null
            }
        }
    }

    private fun save(bitmap: Bitmap): Boolean {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "LLMHub_Erased_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/LLM Hub")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        val resolver = appContext.contentResolver
        val uri = try {
            resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create MediaStore entry", e)
            null
        } ?: return false
        return try {
            resolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                ?: error("Could not open MediaStore output")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
            true
        } catch (e: Throwable) {
            resolver.delete(uri, null, null)
            Log.e(TAG, "Failed to save", e)
            false
        }
    }

    /** Decodes a picked photo upright (EXIF orientation applied) as a software ARGB_8888 bitmap. */
    private fun decode(uri: Uri): Bitmap {
        // ImageDecoder (Android 9+) applies the EXIF orientation itself; BitmapFactory does not.
        var bitmap = BoundedBitmapDecoder.decodeUri(appContext, uri) ?: error("Could not decode $uri")
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
            val orientation = appContext.contentResolver.openInputStream(uri)?.use {
                android.media.ExifInterface(it).getAttributeInt(
                    android.media.ExifInterface.TAG_ORIENTATION, android.media.ExifInterface.ORIENTATION_NORMAL
                )
            } ?: android.media.ExifInterface.ORIENTATION_NORMAL
            val matrix = Matrix()
            when (orientation) {
                android.media.ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                android.media.ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                android.media.ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                android.media.ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
                android.media.ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
                android.media.ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.postRotate(90f); matrix.postScale(-1f, 1f) }
                android.media.ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.postRotate(270f); matrix.postScale(-1f, 1f) }
            }
            if (!matrix.isIdentity) {
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }
        }
        if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        return bitmap
    }

    /** Draws the strokes into a w*h byte mask, non-zero = erase. */
    private fun rasterize(strokes: List<BrushStroke>, w: Int, h: Int): ByteArray {
        val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8)
        val canvas = android.graphics.Canvas(maskBitmap)
        val line = Paint().apply {
            color = android.graphics.Color.WHITE
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = false
        }
        val dot = Paint().apply {
            color = android.graphics.Color.WHITE
            style = Paint.Style.FILL
            isAntiAlias = false
        }
        for (stroke in strokes) {
            if (stroke.points.size == 1) {
                canvas.drawCircle(stroke.points[0].x, stroke.points[0].y, stroke.radius, dot)
            } else {
                val path = Path()
                path.moveTo(stroke.points[0].x, stroke.points[0].y)
                for (i in 1 until stroke.points.size) path.lineTo(stroke.points[i].x, stroke.points[i].y)
                line.strokeWidth = stroke.radius * 2f
                canvas.drawPath(path, line)
            }
        }
        val rowBytes = maskBitmap.rowBytes
        val buffer = ByteBuffer.allocate(rowBytes * h)
        maskBitmap.copyPixelsToBuffer(buffer)
        maskBitmap.recycle()
        val raw = buffer.array()
        if (rowBytes == w) return raw
        val mask = ByteArray(w * h)
        for (y in 0 until h) System.arraycopy(raw, y * rowBytes, mask, y * w, w)
        return mask
    }

    companion object {
        private const val TAG = "ImageInpaintVM"
        private const val KEY_KIND = "kind"
        private const val KEY_BRUSH = "brush_dp"
        const val DEFAULT_BRUSH_DP = 28f
        const val MIN_BRUSH_DP = 8f
        const val MAX_BRUSH_DP = 80f
        private val cleanupScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    }
}
