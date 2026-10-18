package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.hasCompleteDownloadedBundle
import com.llmhub.llmhub.data.isVideoGenerationModel
import com.llmhub.llmhub.videogen.VideoGenBackend
import com.llmhub.llmhub.videogen.VideoGenEngine
import com.llmhub.llmhub.videogen.VideoGenProgress
import com.llmhub.llmhub.videogen.VideoGenRequest
import com.llmhub.llmhub.videogen.VideoGenResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.coroutines.cancellation.CancellationException
import kotlin.math.roundToInt

/**
 * State holder for the Video Generator screen. Owns the [VideoGenEngine] so a loaded model survives
 * configuration changes; the model is unloaded when the screen leaves the back stack (onCleared).
 */
class VideoGeneratorViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "VideoGeneratorVM"
        const val MIN_STEPS = 4
        // LTX-Video 0.9.8 distilled has 8 trained timesteps; fewer steps pick a subset of them.
        const val MAX_STEPS = 8
        const val MIN_DIMENSION = 320
        const val MAX_DIMENSION = 768
        const val DIMENSION_STEP = 32
        const val MIN_FRAMES = 9
        const val MAX_FRAMES = 97
        const val MIN_FPS = 8
        const val MAX_FPS = 30

        /** LTX-Video needs 8n+1 frames. */
        fun snapFrames(value: Int): Int {
            val n = ((value - 1) / 8f).roundToInt().coerceIn(1, (MAX_FRAMES - 1) / 8)
            return n * 8 + 1
        }

        fun snapDimension(value: Int): Int =
            ((value / DIMENSION_STEP.toFloat()).roundToInt() * DIMENSION_STEP).coerceIn(MIN_DIMENSION, MAX_DIMENSION)
    }

    private val appContext: Context get() = getApplication<Application>()
    private val prefs = application.getSharedPreferences("video_generator_prefs", Context.MODE_PRIVATE)

    val engine = VideoGenEngine(application)
    val loadedModel: StateFlow<LLMModel?> = engine.loadedModel
    val loadedBackend: StateFlow<VideoGenBackend?> = engine.loadedBackend

    /** null until the first scan of downloaded models has finished. */
    private val _availableModels = MutableStateFlow<List<LLMModel>?>(null)
    val availableModels: StateFlow<List<LLMModel>?> = _availableModels.asStateFlow()

    private val _progress = MutableStateFlow<VideoGenProgress?>(null)
    val progress: StateFlow<VideoGenProgress?> = _progress.asStateFlow()

    private val _messages = MutableSharedFlow<String>(extraBufferCapacity = 4)
    val messages: SharedFlow<String> = _messages.asSharedFlow()

    var selectedModel by mutableStateOf<LLMModel?>(null)
        private set
    var selectedBackend by mutableStateOf<VideoGenBackend?>(null)
        private set

    var prompt by mutableStateOf("")
    var negativePrompt by mutableStateOf(prefs.getString("negative_prompt", "") ?: "")
        private set
    var steps by mutableIntStateOf(prefs.getInt("steps", 8).coerceIn(MIN_STEPS, MAX_STEPS))
        private set
    var width by mutableIntStateOf(snapDimension(prefs.getInt("width", 512)))
        private set
    var height by mutableIntStateOf(snapDimension(prefs.getInt("height", 320)))
        private set
    var numFrames by mutableIntStateOf(snapFrames(prefs.getInt("num_frames", 25)))
        private set
    var fps by mutableIntStateOf(prefs.getInt("fps", 24).coerceIn(MIN_FPS, MAX_FPS))
        private set
    var seed by mutableLongStateOf(prefs.getLong("seed", (0..999999).random().toLong()))
        private set

    var useInitImage by mutableStateOf(false)
    var initImage by mutableStateOf<Bitmap?>(null)

    var isLoading by mutableStateOf(false)
        private set
    var isGenerating by mutableStateOf(false)
        private set
    var loadError by mutableStateOf<String?>(null)
        private set
    var result by mutableStateOf<VideoGenResult?>(null)
        private set
    /** Width/height of the request that produced [result], used for the player aspect ratio. */
    var resultSize by mutableStateOf(512 to 320)
        private set

    private var workJob: Job? = null
    private var cancelRequested = false

    val isBusy: Boolean get() = isLoading || isGenerating

    /** NPU models run graphs compiled for a single resolution and length. */
    val sizeLocked: Boolean get() = selectedModel?.let(VideoGenEngine::fixedShape) != null

    fun refreshModels() {
        viewModelScope.launch {
            val models = withContext(Dispatchers.IO) {
                ModelData.models.filter { it.isVideoGenerationModel() && it.hasCompleteDownloadedBundle(appContext) }
            }
            _availableModels.value = models
            val current = selectedModel
            if (current == null || models.none { it.name == current.name }) {
                val lastName = prefs.getString("last_model", null)
                selectModel(models.find { it.name == lastName } ?: models.firstOrNull())
            }
            // A deleted model must not stay loaded
            val loaded = loadedModel.value
            if (loaded != null && models.none { it.name == loaded.name } && !isBusy) {
                engine.unload()
            }
        }
    }

    fun selectModel(model: LLMModel?) {
        selectedModel = model
        loadError = null
        if (model == null) {
            selectedBackend = null
            return
        }
        prefs.edit().putString("last_model", model.name).apply()
        val fixed = VideoGenEngine.fixedShape(model)
        width = fixed?.first ?: snapDimension(prefs.getInt("width", 512))
        height = fixed?.second ?: snapDimension(prefs.getInt("height", 320))
        numFrames = fixed?.third ?: snapFrames(prefs.getInt("num_frames", 25))
        val supported = VideoGenEngine.supportedBackends(model)
        val saved = prefs.getString("backend_${model.name}", null)
            ?.let { name -> VideoGenBackend.entries.firstOrNull { it.name == name } }
        selectedBackend = saved?.takeIf { it in supported } ?: supported.first()
    }

    fun selectBackend(backend: VideoGenBackend) {
        val model = selectedModel ?: return
        if (backend !in VideoGenEngine.supportedBackends(model)) return
        selectedBackend = backend
        loadError = null
        prefs.edit().putString("backend_${model.name}", backend.name).apply()
    }

    fun updateSteps(value: Int) {
        steps = value.coerceIn(MIN_STEPS, MAX_STEPS)
        prefs.edit().putInt("steps", steps).apply()
    }

    fun updateWidth(value: Int) {
        if (sizeLocked) return
        width = snapDimension(value)
        prefs.edit().putInt("width", width).apply()
    }

    fun updateHeight(value: Int) {
        if (sizeLocked) return
        height = snapDimension(value)
        prefs.edit().putInt("height", height).apply()
    }

    fun updateFrames(value: Int) {
        if (sizeLocked) return
        numFrames = snapFrames(value)
        prefs.edit().putInt("num_frames", numFrames).apply()
    }

    fun updateFps(value: Int) {
        fps = value.coerceIn(MIN_FPS, MAX_FPS)
        prefs.edit().putInt("fps", fps).apply()
    }

    fun updateSeed(value: Long) {
        seed = value.coerceAtLeast(0L)
        prefs.edit().putLong("seed", seed).apply()
    }

    fun randomizeSeed() = updateSeed((0..999999).random().toLong())

    fun updateNegativePrompt(value: String) {
        negativePrompt = value
        prefs.edit().putString("negative_prompt", value).apply()
    }

    fun loadModel() {
        val model = selectedModel ?: return
        val backend = selectedBackend ?: VideoGenEngine.supportedBackends(model).first()
        if (isBusy) return
        cancelRequested = false
        loadError = null
        isLoading = true
        _progress.value = VideoGenProgress.LoadingModel(0f)
        workJob = viewModelScope.launch {
            try {
                val outcome = withContext(Dispatchers.IO) {
                    engine.load(model, backend) { _progress.value = it }
                }
                outcome.onFailure { error ->
                    if (error is CancellationException) throw error
                    Log.e(TAG, "Failed to load ${model.name} on $backend", error)
                    val message = if (cancelRequested) {
                        appContext.getString(R.string.video_generator_cancelled)
                    } else {
                        appContext.getString(R.string.video_generator_load_failed, error.message ?: error.toString())
                    }
                    if (!cancelRequested) loadError = message
                    _messages.tryEmit(message)
                }
            } finally {
                isLoading = false
                _progress.value = null
            }
        }
    }

    fun unloadModel() {
        if (isBusy) cancel()
        engine.unload()
        loadError = null
    }

    fun canGenerate(): Boolean =
        loadedModel.value != null && !isBusy && prompt.isNotBlank() && (!useInitImage || initImage != null)

    fun generate() {
        if (!canGenerate()) return
        val request = VideoGenRequest(
            prompt = prompt.trim(),
            negativePrompt = negativePrompt.trim(),
            width = snapDimension(width),
            height = snapDimension(height),
            numFrames = snapFrames(numFrames),
            fps = fps,
            steps = steps,
            seed = seed,
            initImage = if (useInitImage) initImage else null
        )
        cancelRequested = false
        isGenerating = true
        _progress.value = VideoGenProgress.EncodingPrompt
        workJob = viewModelScope.launch {
            try {
                val outcome = withContext(Dispatchers.IO) {
                    engine.generate(request) { _progress.value = it }
                }
                outcome.onSuccess {
                    result = it
                    resultSize = request.width to request.height
                }.onFailure { error ->
                    if (error is CancellationException) throw error
                    Log.e(TAG, "Video generation failed", error)
                    _messages.tryEmit(
                        if (cancelRequested) appContext.getString(R.string.video_generator_cancelled)
                        else appContext.getString(R.string.video_generator_generate_failed, error.message ?: error.toString())
                    )
                }
            } finally {
                isGenerating = false
                _progress.value = null
            }
        }
    }

    fun cancel() {
        if (!isBusy) return
        cancelRequested = true
        engine.cancel()
    }

    fun saveToGallery() {
        val file = result?.videoFile ?: return
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) { saveVideoToGallery(file) }
            _messages.tryEmit(
                appContext.getString(if (saved) R.string.video_generator_saved else R.string.video_generator_save_failed)
            )
        }
    }

    fun shareIntent(): Intent? {
        val file = result?.videoFile?.takeIf { it.isFile } ?: return null
        return try {
            val uri = FileProvider.getUriForFile(appContext, "${appContext.packageName}.fileprovider", file)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "video/mp4"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            Intent.createChooser(send, appContext.getString(R.string.video_generator_share))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to share video", e)
            null
        }
    }

    private fun saveVideoToGallery(source: java.io.File): Boolean {
        if (!source.isFile) return false
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, "LLMHub_Video_${System.currentTimeMillis()}.mp4")
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/LLM Hub")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
        }
        val resolver = appContext.contentResolver
        val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        }
        val uri = try {
            resolver.insert(collection, values)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create MediaStore entry", e)
            null
        } ?: return false
        return try {
            resolver.openOutputStream(uri)?.use { output -> source.inputStream().use { it.copyTo(output) } }
                ?: error("Could not open MediaStore output")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Video.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
            true
        } catch (e: Throwable) {
            resolver.delete(uri, null, null)
            Log.e(TAG, "Failed to save video", e)
            false
        }
    }

    override fun onCleared() {
        val running = workJob?.takeIf { it.isActive }
        if (running != null) {
            // Native work cannot be interrupted mid-call: signal cancel, then unload once it returns.
            engine.cancel()
            CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
                running.join()
                engine.unload()
            }
        } else {
            engine.unload()
        }
        super.onCleared()
    }
}
