package com.llmhub.llmhub.videogen

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.llmhub.llmhub.data.LLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

private const val TAG = "LtxVideoRuntime"

/**
 * Cores clocked within 75% of the fastest one. ggml splits every op evenly across its threads, so
 * scheduling work on the efficiency cores makes the big cores wait for them.
 */
private fun performanceCoreCount(): Int {
    val maxFreqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { cpu ->
        runCatching { File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }.getOrNull()
    }
    val fastest = maxFreqs.maxOrNull() ?: return Runtime.getRuntime().availableProcessors()
    return maxFreqs.count { it >= fastest * 3 / 4 }
}

/** Sigma schedule of LTX-Video 0.9.8 distilled ("allowed_inference_steps" in the checkpoint config). */
internal object LtxvSchedule {
    private val distilled = floatArrayOf(1.0f, 0.9937f, 0.9875f, 0.9812f, 0.975f, 0.9094f, 0.725f, 0.4219f)

    /** Returns steps+1 sigmas ending in 0; fewer steps pick evenly spaced entries of the distilled grid. */
    fun sigmas(steps: Int): FloatArray {
        val n = steps.coerceIn(1, distilled.size)
        val picked = if (n == distilled.size) {
            distilled.toList()
        } else {
            (0 until n).map { i -> distilled[(i * (distilled.size - 1).toFloat() / (n - 1).coerceAtLeast(1)).toInt()] }
        }
        return (picked + 0f).toFloatArray()
    }
}

/**
 * LTX-Video through stable-diffusion.cpp. With [npu] the transformer runs from QNN context binaries
 * (ltxv_chunk0..3.bin in the model directory) on the Hexagon HTP; T5 and the VAE stay on the CPU.
 */
internal open class SdCppLtxVideoRuntime(
    private val context: Context,
    private val npu: Boolean = false
) : LtxVideoRuntime {
    @Volatile
    private var handle = 0L

    override suspend fun load(
        modelDir: File,
        model: LLMModel,
        backend: VideoGenBackend,
        onProgress: (VideoGenProgress) -> Unit
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            // Debug builds: `touch files/debug_dump_latent` makes the engine save the final latent
            // (cache/ltxv_latent.raw) so it can be decoded with the reference VAE on a desktop.
            if (com.llmhub.llmhub.BuildConfig.DEBUG && File(context.filesDir, "debug_dump_latent").exists()) {
                android.system.Os.setenv("SD_DUMP_LATENT_PATH", File(context.cacheDir, "ltxv_latent.raw").absolutePath, true)
            }
            SdCppNative.ensureLoaded()
            val files = modelDir.listFiles()?.filter { it.length() > 0 } ?: emptyList()
            val t5 = files.firstOrNull { it.name.endsWith(".gguf") && it.name.contains("t5", ignoreCase = true) }
                ?: error("T5 text encoder file is missing in ${modelDir.name}")
            // NPU models ship a small stub with the transformer's shape tensors instead of the full GGUF.
            val dit = files.firstOrNull { it.name.endsWith(".gguf") && it != t5 }
                ?: files.firstOrNull { it.name.contains("stub", ignoreCase = true) && it.name.endsWith(".safetensors") }
                ?: error("LTX-Video diffusion model file is missing in ${modelDir.name}")
            // NPU models also contain ltxv_vae_<shape>.bin (the compiled decoder); stable-diffusion.cpp needs
            // the safetensors VAE (latent statistics + image-to-video encoder).
            val vae = files.firstOrNull { it.name.contains("vae", ignoreCase = true) && it.name.endsWith(".safetensors") }
                ?: error("LTX-Video VAE file is missing in ${modelDir.name}")
            onProgress(VideoGenProgress.LoadingModel(0f))
            // T5 runs once per prompt; keep it on the CPU because some mobile Vulkan drivers (Adreno)
            // fail to build compute pipelines for its K-quant matmuls.
            // Debug builds: `touch files/debug_vae_vulkan` decodes on the GPU in NPU mode (Adreno test).
            val vaeOnGpu = com.llmhub.llmhub.BuildConfig.DEBUG && File(context.filesDir, "debug_vae_vulkan").exists()
            val gpus = if (npu || backend != VideoGenBackend.GPU) "" else SdCppNative.nativeListDevices()
            // PowerVR (Tensor G5/G6) runs the transformer but its Vulkan driver rejects the VAE decoder's
            // largest conv weights, which ggml cannot move off the GPU once allocated: decode on the CPU.
            val vaeDevice = if (gpus.contains("PowerVR", ignoreCase = true)) "CPU" else "Vulkan0"
            val device = when {
                npu && vaeOnGpu -> "te=CPU,diffusion=CPU,vae=Vulkan0"
                npu -> "CPU"
                backend == VideoGenBackend.GPU -> "te=CPU,diffusion=Vulkan0,vae=$vaeDevice"
                else -> "CPU"
            }
            val modelArgs = if (npu) {
                val missing = (0 until QNN_CHUNKS).map { "ltxv_chunk$it.bin" }.filterNot { File(modelDir, it).length() > 0 }
                check(missing.isEmpty()) { "NPU model files are missing: ${missing.joinToString()}" }
                "ltxv_qnn_dir=${modelDir.absolutePath},ltxv_qnn_lib_dir=${stageQnnLibraries()}"
            } else {
                ""
            }
            Log.i(TAG, "devices: ${SdCppNative.nativeListDevices().trim()}; using $device ${if (npu) "+ Hexagon NPU" else ""}")
            val threads = performanceCoreCount().coerceIn(2, 8)
            val created = SdCppNative.nativeCreate(dit.absolutePath, t5.absolutePath, vae.absolutePath, device, threads, !npu && backend == VideoGenBackend.CPU, modelArgs)
            if (created == 0L) {
                error(SdCppNative.nativeLastError().ifBlank { "Failed to load LTX-Video" })
            }
            handle = created
            onProgress(VideoGenProgress.LoadingModel(1f))
        }
    }

    override suspend fun generate(
        request: VideoGenRequest,
        outputDir: File,
        onProgress: (VideoGenProgress) -> Unit
    ): Result<VideoGenResult> = withContext(Dispatchers.IO) {
        runCatching {
            val session = handle
            check(session != 0L) { "No video model loaded" }
            val started = System.currentTimeMillis()
            val init = request.initImage?.let { fitToSize(it, request.width, request.height) }
            val callback = object : SdCppNative.Callback {
                override fun onProgress(step: Int, steps: Int) {
                    onProgress(VideoGenProgress.Denoising(step.coerceAtMost(steps), steps))
                }

                override fun onStage(stage: Int) {
                    when (stage) {
                        SdCppNative.STAGE_ENCODING_PROMPT -> onProgress(VideoGenProgress.EncodingPrompt)
                        SdCppNative.STAGE_DECODING -> onProgress(VideoGenProgress.DecodingFrames)
                    }
                }
            }
            val sigmas = LtxvSchedule.sigmas(request.steps)
            val seed = if (request.seed >= 0) request.seed else Random.nextLong(0, Int.MAX_VALUE.toLong())
            val result = SdCppNative.nativeGenerate(
                session,
                request.prompt,
                request.negativePrompt,
                request.width,
                request.height,
                request.numFrames,
                request.fps,
                sigmas.size - 1,
                1.0f,
                seed,
                sigmas,
                init?.let { Mp4Writer.bitmapToRgb(it) },
                init?.width ?: 0,
                init?.height ?: 0,
                callback
            ) ?: error(SdCppNative.nativeLastError().ifBlank { "Video generation failed" })

            val header = ByteBuffer.wrap(result[0]).order(ByteOrder.LITTLE_ENDIAN)
            val width = header.int
            val height = header.int
            val fps = header.int.takeIf { it > 0 } ?: request.fps
            val frames = result.drop(1)
            onProgress(VideoGenProgress.WritingVideo)
            val file = File(outputDir, "ltxv_${System.currentTimeMillis()}.mp4")
            Mp4Writer.write(frames, width, height, fps, file)
            VideoGenResult(
                videoFile = file,
                thumbnail = Mp4Writer.toBitmap(frames.first(), width, height),
                frameCount = frames.size,
                fps = fps,
                elapsedMs = System.currentTimeMillis() - started
            )
        }
    }

    override fun cancel() {
        val session = handle
        if (session != 0L) {
            SdCppNative.nativeCancel(session)
        }
    }

    override fun release() {
        val session = handle
        handle = 0L
        if (session != 0L) {
            SdCppNative.nativeFree(session)
        }
    }

    /**
     * QNN needs libQnnHtp/libQnnSystem and the Hexagon skel libraries in one readable directory.
     * Reuses the QAIRT 2.50 runtime staged for LiteRT (assets/qnnlibs_litert -> litert_npu_qnn_250).
     */
    private fun stageQnnLibraries(): String {
        val dir = File(context.filesDir, "litert_npu_qnn_250").apply { mkdirs() }
        val names = context.assets.list("qnnlibs_litert").orEmpty()
        check(names.isNotEmpty()) { "QAIRT runtime libraries are missing from the app" }
        for (name in names) {
            val target = File(dir, name)
            if (target.exists() && target.length() > 0L) continue
            context.assets.open("qnnlibs_litert/$name").use { input ->
                target.outputStream().use { input.copyTo(it) }
            }
            target.setReadable(true, false)
            target.setExecutable(true, false)
        }
        return dir.absolutePath
    }

    // Center-crop the conditioning image to the output aspect ratio, then scale it to the output size.
    private fun fitToSize(source: Bitmap, width: Int, height: Int): Bitmap {
        val targetRatio = width.toFloat() / height
        val sourceRatio = source.width.toFloat() / source.height
        val (cropW, cropH) = if (sourceRatio > targetRatio) {
            (source.height * targetRatio).toInt() to source.height
        } else {
            source.width to (source.width / targetRatio).toInt()
        }
        val cropped = Bitmap.createBitmap(source, (source.width - cropW) / 2, (source.height - cropH) / 2, cropW, cropH)
        return Bitmap.createScaledBitmap(cropped, width, height, true).copy(Bitmap.Config.ARGB_8888, false)
    }
}

/** LTX-Video with the transformer on the Qualcomm Hexagon NPU ("ltxv_qnn" models). */
internal class QnnLtxVideoRuntime(context: Context) : SdCppLtxVideoRuntime(context, npu = true)

private const val QNN_CHUNKS = 4
