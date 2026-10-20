package com.llmhub.llmhub.videogen

import android.content.Context
import android.graphics.Bitmap
import com.llmhub.llmhub.data.LLMModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * Runtime used for LTX-Video inference.
 * NPU runs QNN context binaries on the Qualcomm Hexagon HTP ("ltxv_qnn" models).
 * GPU runs stable-diffusion.cpp through Vulkan and CPU runs the same GGUF files on the CPU ("ltxv_gguf" models).
 */
enum class VideoGenBackend { NPU, GPU, CPU }

data class VideoGenRequest(
    val prompt: String,
    val negativePrompt: String = "",
    val width: Int = 512,
    val height: Int = 320,
    val numFrames: Int = 25,
    val fps: Int = 24,
    val steps: Int = 8,
    val seed: Long = -1L,
    val initImage: Bitmap? = null
)

sealed interface VideoGenProgress {
    data class LoadingModel(val fraction: Float) : VideoGenProgress
    data object EncodingPrompt : VideoGenProgress
    data class Denoising(val step: Int, val totalSteps: Int) : VideoGenProgress
    data object DecodingFrames : VideoGenProgress
    data object WritingVideo : VideoGenProgress
}

data class VideoGenResult(
    val videoFile: File,
    val thumbnail: Bitmap?,
    val frameCount: Int,
    val fps: Int,
    val elapsedMs: Long
)

object VideoGenFormats {
    const val CATEGORY = "video_generation"
    const val GGUF = "ltxv_gguf"
    const val QNN = "ltxv_qnn"
}

class VideoGenEngine(context: Context) {
    private val appContext = context.applicationContext
    private val _loadedModel = MutableStateFlow<LLMModel?>(null)
    val loadedModel: StateFlow<LLMModel?> = _loadedModel.asStateFlow()
    private val _loadedBackend = MutableStateFlow<VideoGenBackend?>(null)
    val loadedBackend: StateFlow<VideoGenBackend?> = _loadedBackend.asStateFlow()

    suspend fun load(
        model: LLMModel,
        backend: VideoGenBackend,
        onProgress: (VideoGenProgress) -> Unit = {}
    ): Result<Unit> {
        unload()
        val runtime = when (model.modelFormat) {
            VideoGenFormats.QNN -> QnnLtxVideoRuntime(appContext)
            else -> SdCppLtxVideoRuntime(appContext)
        }
        return runtime.load(modelDir(appContext, model), model, backend, onProgress).onSuccess {
            this.runtime = runtime
            _loadedModel.value = model
            _loadedBackend.value = backend
        }
    }

    suspend fun generate(
        request: VideoGenRequest,
        onProgress: (VideoGenProgress) -> Unit
    ): Result<VideoGenResult> {
        val active = runtime ?: return Result.failure(IllegalStateException("No video model loaded"))
        val outDir = File(appContext.filesDir, "generated_videos").apply { mkdirs() }
        return active.generate(request, outDir, onProgress)
    }

    fun cancel() {
        runtime?.cancel()
    }

    fun unload() {
        runtime?.release()
        runtime = null
        _loadedModel.value = null
        _loadedBackend.value = null
    }

    private var runtime: LtxVideoRuntime? = null

    companion object {
        /** NPU graphs are compiled for one shape: (width, height, frames); null means any size. */
        fun fixedShape(model: LLMModel): Triple<Int, Int, Int>? {
            if (model.modelFormat != VideoGenFormats.QNN) return null
            // The shape is encoded in the NPU VAE decoder file: ltxv_vae_<width>x<height>x<frames>.bin
            val match = (model.additionalFiles + model.url).firstNotNullOfOrNull { NPU_VAE_FILE.find(it) }
                ?: return Triple(512, 320, 25)
            val (w, h, f) = match.destructured
            return Triple(w.toInt(), h.toInt(), f.toInt())
        }

        private val NPU_VAE_FILE = Regex("""ltxv_vae_(\d+)x(\d+)x(\d+)\.bin""")

        fun supportedBackends(model: LLMModel): List<VideoGenBackend> = when (model.modelFormat) {
            VideoGenFormats.QNN -> listOf(VideoGenBackend.NPU)
            // Adreno's Vulkan driver crashes compiling ggml's video shaders; Snapdragon devices use
            // the NPU models instead, so GGUF models run on the CPU there. The Tensor G5's PowerVR
            // D-Series driver runs the transformer but computes noise (with or without ggml's
            // integer-dot, cooperative-matrix and FP16 paths), so it gets the CPU too.
            else -> if (isAdreno() || isTensorG5()) listOf(VideoGenBackend.CPU) else listOf(VideoGenBackend.GPU, VideoGenBackend.CPU)
        }

        private fun isTensorG5(): Boolean =
            android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S &&
                android.os.Build.SOC_MODEL.equals("Tensor G5", ignoreCase = true)

        private fun isAdreno(): Boolean {
            val soc = com.llmhub.llmhub.data.DeviceInfo.getDeviceSoc()
            val manufacturer = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) android.os.Build.SOC_MANUFACTURER else ""
            return manufacturer.equals("QTI", ignoreCase = true) || soc.startsWith("SM") || soc.startsWith("QCS") || soc.startsWith("QCM")
        }

        /** All files of a video model live in filesDir/models/<sanitized model name>/. */
        fun modelDir(context: Context, model: LLMModel): File {
            val dirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
            return File(File(context.filesDir, "models"), dirName)
        }
    }
}

internal interface LtxVideoRuntime {
    suspend fun load(
        modelDir: File,
        model: LLMModel,
        backend: VideoGenBackend,
        onProgress: (VideoGenProgress) -> Unit
    ): Result<Unit>

    suspend fun generate(
        request: VideoGenRequest,
        outputDir: File,
        onProgress: (VideoGenProgress) -> Unit
    ): Result<VideoGenResult>

    fun cancel()
    fun release()
}
