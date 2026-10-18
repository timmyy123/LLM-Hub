package com.llmhub.llmhub.videogen

/** JNI surface of libllmhub_sdcpp.so (app/src/main/cpp/sdcpp/sdcpp_jni.cpp). */
internal object SdCppNative {
    const val STAGE_ENCODING_PROMPT = 1
    const val STAGE_DENOISING = 2
    const val STAGE_DECODING = 3

    interface Callback {
        fun onProgress(step: Int, steps: Int)
        fun onStage(stage: Int)
    }

    @Volatile
    private var loaded = false

    @Synchronized
    fun ensureLoaded() {
        if (!loaded) {
            // libllmhub_sdcpp is built for armv8.2-a with dotprod/fp16 (see :sdcpp); older cores would SIGILL.
            val features = runCatching { java.io.File("/proc/cpuinfo").readText() }.getOrDefault("")
            if (!features.contains("asimddp")) {
                throw UnsupportedOperationException("Video generation needs an ARMv8.2 CPU with dot-product support")
            }
            System.loadLibrary("llmhub_sdcpp")
            loaded = true
        }
    }

    external fun nativeListDevices(): String
    external fun nativeLastError(): String

    external fun nativeCreate(
        diffusionPath: String,
        t5Path: String,
        vaePath: String,
        backend: String,
        threads: Int,
        flashAttn: Boolean,
        modelArgs: String
    ): Long

    /**
     * Returns null on failure. Otherwise element 0 is a 12-byte little-endian header
     * (width, height, fps) and every following element is one packed RGB frame.
     */
    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        frames: Int,
        fps: Int,
        steps: Int,
        cfgScale: Float,
        seed: Long,
        sigmas: FloatArray?,
        initRgb: ByteArray?,
        initWidth: Int,
        initHeight: Int,
        callback: Callback
    ): Array<ByteArray>?

    external fun nativeCancel(handle: Long)
    external fun nativeFree(handle: Long)
}
