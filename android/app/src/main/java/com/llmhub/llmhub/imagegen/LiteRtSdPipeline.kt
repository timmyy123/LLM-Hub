package com.llmhub.llmhub.imagegen

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.Random
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Stable Diffusion 1.5 on LiteRT for the per-SoC NPU builds in timmyboy26/stable-diffusion-v1-5-LiteRT-NPU.
 * The app runs it in [LiteRtSdService]'s process, since a vendor NPU driver can crash instead of failing.
 *
 * Mirrors litert_torch's Stable Diffusion example pipeline (k-Euler sampler, classifier-free guidance with the
 * conditional context first), which the models were converted from. The models have static 512x512 shapes.
 */
internal class LiteRtSdPipeline(
    context: Context,
    modelDir: File,
    private val accelerators: Set<Accelerator> = DEFAULT_ACCELERATORS
) : AutoCloseable {
    private val environment = vendorLibraryDir(context)?.let { dir ->
        Environment.create(mapOf(Environment.Option.DispatchLibraryDir to dir.path, Environment.Option.CompilerPluginLibraryDir to dir.path))
    } ?: Environment.create(BuiltinNpuAcceleratorProvider(context))
    private val tokenizer = ClipBpeTokenizer(File(modelDir, "vocab.json"), File(modelDir, "merges.txt"))
    private val options = CompiledModel.Options(accelerators)

    /** Model load times in milliseconds; on-device NPU compilation of uncompiled files happens here. */
    val loadTimings = linkedMapOf<String, Long>()
    private val clip = load(File(modelDir, "clip.tflite"))
    private val unet = load(File(modelDir, "diffusion.tflite"))
    private val decoder = load(File(modelDir, "decoder.tflite"))

    /** Accelerators LiteRT can use on this device; NPU is absent when no dispatch library matches the SoC. */
    val availableAccelerators: Set<Accelerator> get() = environment.getAvailableAccelerators()

    /** Wall-clock milliseconds of the last generation's stages. */
    var lastTimings: Map<String, Long> = emptyMap()
        private set

    private fun load(file: File): CompiledModel {
        val started = SystemClock.elapsedRealtime()
        return CompiledModel.create(file.path, options, environment).also {
            val elapsed = SystemClock.elapsedRealtime() - started
            loadTimings[file.nameWithoutExtension] = elapsed
            Log.i(TAG, "Loaded ${file.name} in $elapsed ms ($accelerators)")
        }
    }

    fun generate(
        prompt: String,
        negativePrompt: String,
        steps: Int,
        guidance: Float,
        seed: Long,
        onStep: (step: Int, total: Int) -> Unit = { _, _ -> }
    ): Bitmap {
        val timings = linkedMapOf<String, Long>()
        var started = SystemClock.elapsedRealtime()
        val context = encode(prompt) + encode(negativePrompt)
        timings["text"] = SystemClock.elapsedRealtime() - started

        val sampler = KEulerSampler(steps)
        val random = Random(seed)
        val latents = FloatArray(LATENT_SIZE) { (random.nextGaussian() * sampler.initialScale).toFloat() }
        started = SystemClock.elapsedRealtime()
        withBuffers(unet, UNET, UNET_INPUTS) { inputs, output ->
            inputs.getValue(CONTEXT).writeFloat(context)
            for (step in 0 until steps) {
                val scale = sampler.inputScale(step).toFloat()
                inputs.getValue(LATENTS).writeFloat(FloatArray(LATENT_SIZE * 2) { latents[it % LATENT_SIZE] * scale })
                inputs.getValue(TIME).writeFloat(timeEmbedding(sampler.timesteps[step]))
                unet.run(inputs, mapOf(OUTPUT to output), UNET)
                val noise = output.readFloat()
                val delta = (sampler.sigmas[step + 1] - sampler.sigmas[step]).toFloat()
                for (i in 0 until LATENT_SIZE) {
                    val uncond = noise[LATENT_SIZE + i]
                    latents[i] += (guidance * (noise[i] - uncond) + uncond) * delta
                }
                onStep(step + 1, steps)
            }
        }
        timings["denoise"] = SystemClock.elapsedRealtime() - started

        started = SystemClock.elapsedRealtime()
        val image = withBuffers(decoder, DECODE, listOf(LATENTS)) { inputs, output ->
            inputs.getValue(LATENTS).writeFloat(latents)
            decoder.run(inputs, mapOf(OUTPUT to output), DECODE)
            toBitmap(output.readFloat())
        }
        timings["decode"] = SystemClock.elapsedRealtime() - started
        lastTimings = timings
        Log.i(TAG, "Generated $steps steps: $timings")
        return image
    }

    private fun encode(text: String): FloatArray = withBuffers(clip, ENCODE, listOf(LATENTS)) { inputs, output ->
        inputs.getValue(LATENTS).writeInt(tokenizer.encode(text))
        clip.run(inputs, mapOf(OUTPUT to output), ENCODE)
        output.readFloat()
    }

    /** Binds inputs and the single output by tensor name, and releases the native buffers afterwards. */
    private fun <T> withBuffers(
        model: CompiledModel,
        signature: String,
        inputNames: List<String>,
        block: (Map<String, TensorBuffer>, TensorBuffer) -> T
    ): T {
        val inputs = inputNames.associateWith { model.createInputBuffer(it, signature) }
        val output = model.createOutputBuffer(OUTPUT, signature)
        try {
            return block(inputs, output)
        } finally {
            inputs.values.forEach(TensorBuffer::close)
            output.close()
        }
    }

    /** NCHW in [-1, 1] to an ARGB bitmap. */
    private fun toBitmap(chw: FloatArray): Bitmap {
        val plane = IMAGE_SIZE * IMAGE_SIZE
        fun channel(value: Float) = ((value + 1f) * 127.5f).toInt().coerceIn(0, 255)
        val pixels = IntArray(plane) { i ->
            (0xff shl 24) or (channel(chw[i]) shl 16) or (channel(chw[plane + i]) shl 8) or channel(chw[2 * plane + i])
        }
        return Bitmap.createBitmap(pixels, IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        clip.close()
        unet.close()
        decoder.close()
        environment.close()
    }

    /** k-Euler over SD1.x's scaled-linear beta schedule, as in litert_torch's KEulerSampler. */
    private class KEulerSampler(steps: Int, trainingSteps: Int = 1000) {
        val timesteps = DoubleArray(steps) { (trainingSteps - 1) * (1.0 - if (steps == 1) 0.0 else it.toDouble() / (steps - 1)) }
        val sigmas: DoubleArray
        val initialScale: Double

        init {
            val betaStart = sqrt(0.00085f)
            val betaEnd = sqrt(0.012f)
            var cumulative = 1f
            val logSigmas = DoubleArray(trainingSteps) { i ->
                val beta = (betaStart + (betaEnd - betaStart) * i / (trainingSteps - 1)).pow(2)
                cumulative *= 1f - beta
                ln(sqrt(((1f - cumulative) / cumulative).toDouble()))
            }
            sigmas = DoubleArray(steps + 1) { i ->
                if (i == steps) 0.0 else {
                    val t = timesteps[i]
                    val low = t.toInt().coerceAtMost(trainingSteps - 2)
                    exp(logSigmas[low] + (logSigmas[low + 1] - logSigmas[low]) * (t - low))
                }
            }
            initialScale = sigmas.max()
        }

        fun inputScale(step: Int) = 1.0 / sqrt(sigmas[step] * sigmas[step] + 1.0)
    }

    companion object {
        private const val TAG = "LiteRtSdPipeline"
        private const val IMAGE_SIZE = 512
        private const val LATENT_SIZE = 4 * 64 * 64
        private const val ENCODE = "encode"
        private const val UNET = "diffusion"
        private const val DECODE = "decode"
        // Tensor names from litert_torch signatures: positional arguments args_N, one output_0.
        private const val LATENTS = "args_0"
        private const val CONTEXT = "args_1"
        private const val TIME = "args_2"
        private const val OUTPUT = "output_0"
        private val UNET_INPUTS = listOf(LATENTS, CONTEXT, TIME)
        val DEFAULT_ACCELERATORS = setOf(Accelerator.NPU, Accelerator.CPU)

        /**
         * The app bundles the Google Tensor, Qualcomm and MediaTek NPU libraries in one native library folder. Given
         * that folder, LiteRT 2.3.0 loads the first dispatch library (Google Tensor's) whatever the SoC, so the SoC
         * vendor's dispatch library and compiler plugin are copied into a folder of their own.
         */
        fun vendorLibraryDir(context: Context): File? {
            val manufacturer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MANUFACTURER else ""
            val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else ""
            val vendor = when {
                manufacturer.equals("Mediatek", ignoreCase = true) || soc.startsWith("MT", ignoreCase = true) -> "MediaTek"
                soc.startsWith("Tensor", ignoreCase = true) -> "GoogleTensor"
                com.llmhub.llmhub.data.DeviceInfo.isSnapdragon() -> "Qualcomm"
                else -> return null
            }
            val plugin = if (vendor == "GoogleTensor") "google_tensor" else vendor
            val libraries = listOf("libLiteRtDispatch_$vendor.so", "libLiteRtCompilerPlugin_$plugin.so")
                .map { File(context.applicationInfo.nativeLibraryDir, it) }
            if (!libraries[0].isFile) return null
            val dir = File(context.noBackupFilesDir, "litert_npu/$vendor").apply { mkdirs() }
            for (library in libraries.filter { it.isFile }) {
                val copy = File(dir, library.name)
                if (copy.length() != library.length() || copy.lastModified() < library.lastModified()) library.copyTo(copy, overwrite = true)
            }
            return dir
        }

        /** Sinusoidal SD1.x time embedding: 160 cosines then 160 sines. */
        fun timeEmbedding(timestep: Double): FloatArray {
            val embedding = FloatArray(320)
            for (i in 0 until 160) {
                val angle = timestep * 10000.0.pow(-i / 160.0)
                embedding[i] = cos(angle).toFloat()
                embedding[160 + i] = sin(angle).toFloat()
            }
            return embedding
        }
    }
}
