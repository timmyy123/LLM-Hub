package com.llmhub.llmhub.inference

import android.content.Context
import android.util.Log
import com.argmaxinc.whisperkit.ExperimentalWhisperKit
import com.argmaxinc.whisperkit.WhisperKit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

enum class WhisperBackend { CPU, GPU, NPU }

@OptIn(ExperimentalWhisperKit::class)
class WhisperKitService(private val context: Context) {

    companion object {
        private const val TAG = "WhisperKitService"
        private val MODEL_NAME_TO_VARIANT = mapOf(
            "Whisper_Tiny_English-only" to WhisperKit.Builder.OPENAI_TINY_EN,
            "Whisper_Tiny_Multilingual" to WhisperKit.Builder.OPENAI_TINY,
            "Whisper_Base_English-only" to WhisperKit.Builder.OPENAI_BASE_EN,
            "Whisper_Base_Multilingual" to WhisperKit.Builder.OPENAI_BASE
        )
        // Files the JNI layer requires in the model directory
        private val REQUIRED_FILES = listOf(
            "AudioEncoder.tflite", "MelSpectrogram.tflite", "TextDecoder.tflite",
            "tokenizer.json", "config.json"
        )
    }

    private var whisperKit: WhisperKit? = null
    @Volatile private var lastTranscript: String = ""
    @Volatile private var isInitialized = false
    @Volatile private var isStreaming = false

    val isLoaded: Boolean get() = whisperKit != null && isInitialized

    /**
     * Load a WhisperKit model fully offline — bypasses the SDK's internal HuggingFace download
     * flow via reflection and calls the JNI loadModels() directly with our own config JSON.
     * All 5 required files (3 TFLite + tokenizer.json + config.json) must already be on disk.
     */
    suspend fun loadModel(modelDirPath: String, backend: WhisperBackend = WhisperBackend.NPU): Boolean = withContext(Dispatchers.IO) {
        try {
            release()
            lastTranscript = ""

            val modelDir = File(modelDirPath)
            if (!modelDir.exists() || !modelDir.isDirectory) {
                Log.e(TAG, "Model directory does not exist: $modelDirPath")
                return@withContext false
            }

            for (fileName in REQUIRED_FILES) {
                val file = File(modelDir, fileName)
                if (!file.exists() || file.length() == 0L) {
                    Log.e(TAG, "Missing or empty model file: ${file.absolutePath}")
                    return@withContext false
                }
            }

            val dirName = modelDir.name
            val modelVariant = MODEL_NAME_TO_VARIANT[dirName]
            if (modelVariant == null) {
                Log.e(TAG, "Unknown model directory name: $dirName")
                return@withContext false
            }

            // Copy all 5 files to WhisperKit SDK's expected cache location:
            // context.filesDir/argmaxinc/models/{modelName}
            val modelName = modelVariant.substringAfterLast("/")
            val sdkCacheDir = File(context.filesDir, "argmaxinc/models/$modelName")
            sdkCacheDir.mkdirs()
            for (fileName in REQUIRED_FILES) {
                val src = File(modelDir, fileName)
                val dst = File(sdkCacheDir, fileName)
                if (!dst.exists() || dst.length() != src.length()) {
                    src.copyTo(dst, overwrite = true)
                    Log.d(TAG, "Copied $fileName to SDK cache")
                }
            }

            val wkBackend = when (backend) {
                WhisperBackend.CPU -> WhisperKit.Builder.CPU_ONLY
                WhisperBackend.GPU -> WhisperKit.Builder.CPU_AND_GPU
                WhisperBackend.NPU -> WhisperKit.Builder.CPU_AND_NPU
            }

            Log.i(TAG, "Loading WhisperKit model: $modelVariant, backend: $backend")

            val kit = WhisperKit.Builder()
                .setModel(modelVariant)
                .setApplicationContext(context)
                .setEncoderBackend(wkBackend)
                .setDecoderBackend(wkBackend)
                .setCallback { what, result ->
                    when (what) {
                        WhisperKit.TextOutputCallback.MSG_INIT -> {
                            isInitialized = true
                            Log.i(TAG, "WhisperKit initialized")
                        }
                        WhisperKit.TextOutputCallback.MSG_TEXT_OUT -> {
                            // Use segments (tags stripped) if available, fall back to regex strip
                            lastTranscript = if (result.segments.isNotEmpty()) {
                                result.segments.joinToString(" ") { it.text }.trim()
                            } else {
                                (result.text ?: "")
                                    .replace(Regex("<\\|[^>]*\\|>"), "")
                                    .trim()
                            }
                        }
                        WhisperKit.TextOutputCallback.MSG_CLOSE -> {
                            Log.i(TAG, "WhisperKit closed")
                        }
                    }
                }
                .build()

            whisperKit = kit

            // Bypass kit.loadModel() (which always makes HuggingFace network calls) and call
            // the JNI loadModels() directly via reflection. The SDK cache dir is already populated.
            val libDir = context.applicationInfo.nativeLibraryDir
            val cacheDir = context.cacheDir.path
            val loadConfig = """
                {
                    "lib": "$libDir",
                    "cache": "$cacheDir",
                    "size": "dummy",
                    "encoder_backend": $wkBackend,
                    "decoder_backend": $wkBackend,
                    "model_path": "${sdkCacheDir.absolutePath}",
                    "report_path": "${context.filesDir.path}/argmaxinc/reports"
                }
            """.trimIndent()

            File(context.filesDir, "argmaxinc/reports").mkdirs()

            val implClass = Class.forName("com.argmaxinc.whisperkit.WhisperKitImpl")

            val loadModelsMethod = implClass.getDeclaredMethod("loadModels", String::class.java)
            loadModelsMethod.isAccessible = true
            val result = loadModelsMethod.invoke(kit, loadConfig) as Int
            if (result != 0) {
                Log.e(TAG, "JNI loadModels returned error: $result")
                return@withContext false
            }

            // Set isModelLoaded = true — only set inside loadModel() flow which we bypassed
            val isModelLoadedField = implClass.getDeclaredField("isModelLoaded")
            isModelLoadedField.isAccessible = true
            isModelLoadedField.set(kit, true)

            isInitialized = true
            Log.i(TAG, "WhisperKit model loaded successfully: $modelVariant")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model from: $modelDirPath", e)
            false
        }
    }

    suspend fun transcribe(pcm16Bytes: ByteArray, language: String = "en"): String? = withContext(Dispatchers.IO) {
        val kit = whisperKit ?: return@withContext null
        if (!isInitialized) return@withContext null

        try {
            lastTranscript = ""

            isStreaming = true
            kit.init(frequency = 16000, channels = 1, duration = 0)

            val chunkSize = 3200
            var offset = 0
            while (offset < pcm16Bytes.size) {
                val end = minOf(offset + chunkSize, pcm16Bytes.size)
                kit.transcribe(pcm16Bytes.copyOfRange(offset, end))
                offset = end
            }

            // Pad to 30s frame boundary (960000 bytes = 16kHz mono 16-bit 30s)
            val minFrameBytes = 960_000
            if (pcm16Bytes.size < minFrameBytes) {
                kit.transcribe(ByteArray(minFrameBytes - pcm16Bytes.size))
            }

            kit.deinitialize()
            isStreaming = false

            val result = lastTranscript.trim()
            Log.i(TAG, "Transcription result: '$result'")
            result.ifEmpty { null }
        } catch (e: Exception) {
            Log.e(TAG, "Transcription failed", e)
            if (isStreaming) {
                try { kit.deinitialize() } catch (_: Exception) {}
                isStreaming = false
            }
            null
        }
    }

    fun release() {
        if (isStreaming) {
            try { whisperKit?.deinitialize() } catch (_: Exception) {}
            isStreaming = false
        }
        whisperKit = null
        isInitialized = false
        lastTranscript = ""
    }
}
