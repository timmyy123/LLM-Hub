package com.llmhub.llmhub.embedding

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.EmbeddingEngine
import com.google.ai.edge.litertlm.EmbeddingEngineConfig
import com.google.ai.edge.litertlm.EmbeddingOptions
import com.google.ai.edge.litertlm.InputData
import com.google.android.play.core.assetpacks.AssetPackManagerFactory
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
    }

    val supportsImageEmbedding: Boolean get() = enableMedia && model.supportsVision
    val supportsAudioEmbedding: Boolean get() = enableMedia && model.supportsAudio

    /** Short label of the backend the engine was initialized with, e.g. "GPU". */
    var activeBackendLabel: String? = null
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
                "NPU" to config(Backend.NPU(libDir), Backend.NPU(libDir), Backend.CPU()),
                "NPU + GPU vision" to config(Backend.NPU(libDir), Backend.GPU(), Backend.CPU()),
                "NPU + CPU vision" to config(Backend.NPU(libDir), Backend.CPU(), Backend.CPU()),
            )
        }
        return buildList {
            if (model.supportsGpu) add("GPU" to config(Backend.GPU(), Backend.GPU(), Backend.CPU()))
            add("CPU" to config(Backend.CPU(), Backend.CPU(), Backend.CPU()))
        }
    }

    /**
     * Qualcomm dispatch loads the QNN HTP libraries from the directory it is given, but those
     * libraries ship in the qnn_pack asset pack rather than the APK, so stage both together.
     */
    private fun prepareNpuLibraryDir(): String {
        val nativeDir = context.applicationInfo.nativeLibraryDir
        if (DeviceInfo.getEmbeddingGemma2NpuTag()?.startsWith("Qualcomm") != true) return nativeDir

        val dispatchLib = File(nativeDir, "libLiteRtDispatch_Qualcomm.so")
        if (!dispatchLib.exists()) return nativeDir
        return try {
            val dir = File(context.filesDir, "litert_npu_qnn").apply { mkdirs() }
            copyIfChanged(dispatchLib, File(dir, dispatchLib.name))

            val packAssets = AssetPackManagerFactory.getInstance(context)
                .packLocations["qnn_pack"]?.assetsPath()?.let { File(it, "qnnlibs") }
            if (packAssets?.isDirectory == true) {
                packAssets.listFiles()?.forEach { copyIfChanged(it, File(dir, it.name)) }
            } else {
                context.assets.list("qnnlibs")?.forEach { name ->
                    val target = File(dir, name)
                    if (!target.exists()) {
                        context.assets.open("qnnlibs/$name").use { input ->
                            target.outputStream().use { input.copyTo(it) }
                        }
                    }
                }
            }
            dir.absolutePath
        } catch (e: Exception) {
            Log.w(TAG, "Failed to stage QNN libraries: ${e.message}")
            nativeDir
        }
    }

    private fun copyIfChanged(source: File, target: File) {
        if (target.exists() && target.length() == source.length()) return
        source.inputStream().use { input -> target.outputStream().use { input.copyTo(it) } }
        target.setReadable(true, true)
        target.setExecutable(true, true)
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
