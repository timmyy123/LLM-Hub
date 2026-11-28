package com.llmhub.llmhub.inference

import android.content.Context
import android.util.Log
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/** JNI of the whisper.cpp libraries built from android/app/src/main/cpp/whisper_jni.cpp. */
internal interface WhisperRuntime {
    /** Registers the CPU backend and the [accelerator] module ("Vulkan", "OpenCL", "Hexagon" or null). */
    fun nativeInit(libraryDir: String, accelerator: String?, htpDir: String?): Boolean
    /** Comma-separated names of the GPU-type devices registered in this runtime's ggml. */
    fun nativeGpuDevices(): String
    /** A whisper.cpp context for the ggml model at [modelPath] on [device] (null: CPU), or 0. */
    fun nativeLoad(modelPath: String, device: String?, flashAttn: Boolean): Long
    /** The transcript of 16 kHz mono [samples] as UTF-8, or null on failure or abort. */
    fun nativeTranscribe(handle: Long, samples: FloatArray, language: String, threads: Int): ByteArray?
    fun nativeAbort()
    fun nativeFree(handle: Long)
}

/** libllmhub_whisper: the app's own ggml (per-chip CPU variants, Vulkan). */
internal object WhisperCppNative : WhisperRuntime {
    init { System.loadLibrary("llmhub_whisper") }
    external override fun nativeInit(libraryDir: String, accelerator: String?, htpDir: String?): Boolean
    external override fun nativeGpuDevices(): String
    external override fun nativeLoad(modelPath: String, device: String?, flashAttn: Boolean): Long
    external override fun nativeTranscribe(handle: Long, samples: FloatArray, language: String, threads: Int): ByteArray?
    external override fun nativeAbort()
    external override fun nativeFree(handle: Long)
}

/** libllmhub_whisper_snapdragon: the official llama.cpp Snapdragon release's ggml (Adreno OpenCL, Hexagon NPU). */
internal object WhisperCppSnapdragonNative : WhisperRuntime {
    init { System.loadLibrary("llmhub_whisper_snapdragon") }
    external override fun nativeInit(libraryDir: String, accelerator: String?, htpDir: String?): Boolean
    external override fun nativeGpuDevices(): String
    external override fun nativeLoad(modelPath: String, device: String?, flashAttn: Boolean): Long
    external override fun nativeTranscribe(handle: Long, samples: FloatArray, language: String, threads: Int): ByteArray?
    external override fun nativeAbort()
    external override fun nativeFree(handle: Long)
}

/** Where whisper.cpp runs the model. */
enum class WhisperAccelerator(internal val module: String?, internal val devicePrefix: String?) {
    CPU(null, null),
    /** The app's Vulkan build (any Vulkan GPU). */
    VULKAN("Vulkan", "Vulkan"),
    /** Adreno GPU through the Snapdragon release's OpenCL backend. */
    OPENCL("OpenCL", "GPUOpenCL"),
    /** Hexagon NPU through the Snapdragon release's HTP backend. */
    HEXAGON("Hexagon", "HTP");

    internal val runtime: WhisperRuntime
        get() = if (this == OPENCL || this == HEXAGON) WhisperCppSnapdragonNative else WhisperCppNative
}

/**
 * Whisper speech recognition on whisper.cpp (the engine the Nexa SDK's whisper_cpp plugin wrapped), with ggml
 * models from ggerganov/whisper.cpp. Every native call runs on one thread, in order.
 */
class WhisperCppService(private val context: Context) {
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "whisper-cpp").apply { isDaemon = true } }
    private val dispatcher = executor.asCoroutineDispatcher()
    @Volatile private var handle = 0L
    @Volatile private var runtime: WhisperRuntime = WhisperCppNative

    val isLoaded: Boolean get() = handle != 0L

    /** Loads the ggml model [file] on [accelerator]. False when the file is missing or whisper.cpp rejects it. */
    suspend fun load(file: File, accelerator: WhisperAccelerator = WhisperAccelerator.CPU): Boolean =
        withContext(NonCancellable + dispatcher) {
            release()
            if (!file.isFile) {
                Log.e(TAG, "Model file not found: ${file.absolutePath}")
                return@withContext false
            }
            val native = accelerator.runtime
            val htpDir = if (accelerator == WhisperAccelerator.HEXAGON) LlamaHtpLibraries.prepare(context) else null
            if (!native.nativeInit(context.applicationInfo.nativeLibraryDir, accelerator.module, htpDir)) {
                Log.e(TAG, "whisper.cpp could not initialize $accelerator")
                return@withContext false
            }
            val device = accelerator.devicePrefix?.let { prefix ->
                native.nativeGpuDevices().split(',').firstOrNull { it.startsWith(prefix) }
                    ?: run {
                        Log.e(TAG, "No $prefix device among [${native.nativeGpuDevices()}]")
                        return@withContext false
                    }
            }
            val started = System.currentTimeMillis()
            handle = native.nativeLoad(file.absolutePath, device, flashAttn)
            runtime = native
            Log.i(TAG, "Loaded ${file.name} on ${device ?: "CPU"} in ${System.currentTimeMillis() - started} ms: ${handle != 0L}")
            handle != 0L
        }

    /**
     * Transcribes 16 kHz mono PCM16. [language] is a Whisper language code ("en") or "auto" to detect it.
     * Returns the text ("" when nothing was said) or null when no model is loaded or whisper.cpp fails.
     */
    suspend fun transcribe(pcm16Bytes: ByteArray, language: String): String? = withContext(dispatcher) {
        val current = handle
        if (current == 0L) return@withContext null
        val shorts = ByteBuffer.wrap(pcm16Bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val samples = FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
        runtime.nativeTranscribe(current, samples, language, threads)?.toString(Charsets.UTF_8)?.trim()
    }

    /** Stops a transcription in progress, from any thread. */
    fun abort() = runtime.nativeAbort()

    fun release() {
        val current = handle
        val native = runtime
        handle = 0L
        if (current != 0L) {
            native.nativeAbort()
            try {
                executor.execute { native.nativeFree(current) }
            } catch (_: java.util.concurrent.RejectedExecutionException) {
                native.nativeFree(current)
            }
        }
    }

    companion object {
        private const val TAG = "WhisperCppService"

        /** whisper.cpp's own default; 6 threads were slower than 4 on the Tab S9+ (more of the work on slow cores). */
        @Volatile internal var threads = 4

        @Volatile internal var flashAttn = true
    }
}
