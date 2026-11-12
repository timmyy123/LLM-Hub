package com.llmhub.llmhub.embedding

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.EmbeddingEngine
import com.google.ai.edge.litertlm.EmbeddingEngineConfig
import com.google.ai.edge.litertlm.EmbeddingOptions
import com.google.ai.edge.litertlm.InputData
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * EmbeddingGemma 2 (.litertlm) embeddings through the LiteRT-LM [EmbeddingEngine].
 *
 * Text uses the EmbeddingGemma 2 task prefixes. With [enableMedia], images (JPEG/PNG) and audio
 * (WAV) are embedded without a prefix into the same 768-d space, so text queries retrieve them.
 */
class LiteRtLmEmbeddingService(
    private val context: Context,
    private val model: LLMModel,
    private val enableMedia: Boolean = false,
    private val visionTokensPerImage: Int? = null,
    private val maxInputLength: Int? = null
) : EmbeddingService {

    private var engine: EmbeddingEngine? = null
    private val mutex = Mutex()

    companion object {
        private const val TAG = "LiteRtLmEmbedding"
        private const val MAX_TEXT_CHARS = 6000
        private const val QUERY_PREFIX = "task: search result | query: "
        private const val DOCUMENT_PREFIX = "title: none | text: "
        private const val NPU = "NPU"
        private const val NPU_GPU_VISION = "NPU + GPU vision"
        private const val NPU_CPU_VISION = "NPU + CPU vision"
    }

    val supportsImageEmbedding: Boolean get() = enableMedia && model.supportsVision
    val supportsAudioEmbedding: Boolean get() = enableMedia && model.supportsAudio

    /** Short label of the backend the engine was initialized with, e.g. "GPU". */
    var activeBackendLabel: String? = null
        private set

    /** True when [initialize] failed because this device's NPU can't run this chip-specific build. */
    var npuUnavailable = false
        private set

    override suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (engine != null) return@withLock true

            val modelFile = File(File(context.filesDir, "models"), model.localFileName())
            if (!modelFile.exists() || modelFile.length() == 0L) {
                Log.e(TAG, "Model file missing: ${modelFile.absolutePath}")
                return@withLock false
            }

            for ((label, config) in candidateConfigs(modelFile.absolutePath)) {
                var candidate: EmbeddingEngine? = null
                try {
                    candidate = EmbeddingEngine(config).also { it.initialize() }
                    val probe = candidate.computeEmbedding(listOf(InputData.Text(QUERY_PREFIX + "test")), EmbeddingOptions(normalize = true))
                    if (probe.embedding.isEmpty()) throw IllegalStateException("empty probe embedding")
                    engine = candidate
                    activeBackendLabel = label
                    Log.i(TAG, "Initialized ${model.name} with $label (dim=${probe.embedding.size})")
                    return@withLock true
                } catch (e: Throwable) {
                    Log.w(TAG, "Init with $label failed: ${e.message}")
                    try { candidate?.close() } catch (_: Throwable) { }
                }
            }
            Log.e(TAG, "All backend configurations failed for ${model.name}")
            npuUnavailable = ModelData.isEmbeddingGemma2NpuModel(model)
            false
        }
    }

    private fun candidateConfigs(modelPath: String): List<Pair<String, EmbeddingEngineConfig>> {
        // Compiled GPU programs depend on the signature config, so each config gets its own cache.
        val cacheDir = File(context.cacheDir, "litertlm_embed_m${if (enableMedia) 1 else 0}_v${visionTokensPerImage ?: 0}_t${maxInputLength ?: 0}")
            .apply { mkdirs() }.path
        fun config(main: Backend, visionBackend: Backend, audioBackend: Backend) = EmbeddingEngineConfig(
            modelPath = modelPath,
            backend = main,
            visionBackend = if (supportsImageEmbedding) visionBackend else null,
            audioBackend = if (supportsAudioEmbedding) audioBackend else null,
            cacheDir = cacheDir,
            maxInputLength = maxInputLength,
            visionTokensPerImage = visionTokensPerImage
        )

        if (ModelData.isEmbeddingGemma2NpuModel(model)) {
            val libDir = prepareNpuLibraryDir()
            return listOf(
                NPU to config(Backend.NPU(libDir), Backend.NPU(libDir), Backend.CPU()),
                NPU_GPU_VISION to config(Backend.NPU(libDir), Backend.GPU(), Backend.CPU()),
                NPU_CPU_VISION to config(Backend.NPU(libDir), Backend.CPU(), Backend.CPU()),
            )
        }
        return buildList {
            if (model.supportsGpu) add("GPU" to config(Backend.GPU(), Backend.GPU(), Backend.CPU()))
            add("CPU" to config(Backend.CPU(), Backend.CPU(), Backend.CPU()))
        }
    }

    /**
     * The folder LiteRT-LM loads its NPU dispatch library from.
     *
     * LiteRT-LM 0.19 asks the dispatch library for LiteRtDispatchQueryInterface. The litert-npu-runtime
     * 2.3.x libraries in the APK's native library folder (used by the CompiledModel features) only export
     * the older LiteRtDispatchGetApi, so LiteRT-LM gets its own libraries, built from the LiteRT revision it
     * pins (tools/litertlm_npu) and shipped in assets/litertlm_npu. Each vendor has a folder with only its
     * own dispatch library, because LiteRT loads the first one it finds.
     */
    private fun prepareNpuLibraryDir(): String {
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val tag = DeviceInfo.getEmbeddingGemma2NpuTag() ?: return nativeDir
        val vendor = when {
            tag.startsWith("Qualcomm") -> "Qualcomm"
            tag.startsWith("MediaTek") -> "MediaTek"
            tag.startsWith("Google") -> "GoogleTensor"
            else -> return nativeDir
        }
        return try {
            val library = "libLiteRtDispatch_$vendor.so"
            val dir = File(context.filesDir, "litertlm_npu/$vendor").apply { mkdirs() }
            val bytes = context.assets.open("litertlm_npu/$library").use { it.readBytes() }
            File(dir, library).let { if (it.length() != bytes.size.toLong()) it.writeBytes(bytes) }
            if (vendor == "Qualcomm") linkQnnRuntime(dir)
            dir.absolutePath
        } catch (e: Exception) {
            Log.w(TAG, "Failed to stage $vendor NPU libraries: ${e.message}")
            nativeDir
        }
    }

    /**
     * The Qualcomm dispatch library loads QNN from its own folder and needs QAIRT 2.50 (QNN system API 1.14,
     * assets/qnnlibs_litert). Those 100 MB are kept once in files/litert_npu_qnn_250, where the CompiledModel
     * features stage them too, and linked into [dir]. The Stable Diffusion pack (assets/qnnlibs, system API
     * 1.5) is a different runtime and must not end up here.
     */
    private fun linkQnnRuntime(dir: File) {
        val names = context.assets.list("qnnlibs_litert").orEmpty()
        check(names.isNotEmpty()) { "QAIRT 2.50 libraries missing from assets/qnnlibs_litert" }
        val store = File(context.filesDir, "litert_npu_qnn_250").apply { mkdirs() }
        for (name in names) {
            val stored = File(store, name)
            if (stored.length() == 0L) {
                context.assets.open("qnnlibs_litert/$name").use { input -> stored.outputStream().use { input.copyTo(it) } }
                stored.setReadable(true, false)
                stored.setExecutable(true, false)
            }
            val link = File(dir, name)
            if (link.exists()) continue
            runCatching { android.system.Os.symlink(stored.path, link.path) }.onFailure { stored.copyTo(link, overwrite = true) }
        }
    }

    private suspend fun embed(contents: List<InputData>): FloatArray? {
        if (engine == null && !initialize()) return null
        return withContext(Dispatchers.Default) {
            mutex.withLock {
                val current = engine ?: return@withLock null
                try {
                    current.computeEmbedding(contents, EmbeddingOptions(normalize = true, visionTokensPerImage = visionTokensPerImage))
                        .embedding
                        .takeIf { it.isNotEmpty() }
                } catch (e: Throwable) {
                    Log.e(TAG, "computeEmbedding failed: ${e.message}", e)
                    null
                }
            }
        }
    }

    override suspend fun generateEmbedding(text: String, isQuery: Boolean): FloatArray? {
        val clean = text.trim().take(MAX_TEXT_CHARS)
        if (clean.isEmpty()) return null
        val prefix = if (isQuery) QUERY_PREFIX else DOCUMENT_PREFIX
        return embed(listOf(InputData.Text(prefix + clean)))
    }

    /** [jpegOrPng] must be JPEG or PNG bytes. */
    suspend fun generateImageEmbedding(jpegOrPng: ByteArray): FloatArray? =
        generateImagesEmbedding(listOf(jpegOrPng))

    /** One embedding for several frames, so a video's first and last frame share a vector. */
    suspend fun generateImagesEmbedding(jpegOrPngs: List<ByteArray>): FloatArray? =
        if (supportsImageEmbedding && jpegOrPngs.isNotEmpty()) embed(jpegOrPngs.map { InputData.Image(it) }) else null

    /** [wav] must be a 16 kHz mono WAV file. */
    suspend fun generateAudioEmbedding(wav: ByteArray): FloatArray? =
        if (supportsAudioEmbedding) embed(listOf(InputData.Audio(wav))) else null

    /** One embedding for a Video Moment Finder window: timestamps, audio slices, and frames. */
    suspend fun generateMixedEmbedding(parts: List<InputData>): FloatArray? =
        if (parts.isNotEmpty()) embed(parts) else null

    override suspend fun isInitialized(): Boolean = engine != null

    override fun cleanup() {
        try { engine?.close() } catch (e: Throwable) { Log.w(TAG, "close failed: ${e.message}") }
        engine = null
        activeBackendLabel = null
    }

    /** Like [cleanup], but waits for an in-flight embedding to finish first. */
    suspend fun close() = mutex.withLock { cleanup() }

    override fun getCurrentModelName(): String = "EmbeddingGemma 2"
}
