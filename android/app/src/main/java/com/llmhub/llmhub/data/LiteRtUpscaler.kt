package com.llmhub.llmhub.data

import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * On-device image super-resolution engine using Google LiteRT (TensorFlow Lite).
 * Executes Real-ESRGAN x4 models (such as realesr_general_x4v3.tflite) on GPU
 * with automatic CPU fallback if GPU compilation or execution fails.
 *
 * Model specifications:
 * - Input: [1, 128, 128, 3] NHWC, RGB, normalized float [0.0, 1.0]
 * - Output: [1, 3, 512, 512] NCHW, RGB, normalized float [0.0, 1.0]
 * - Scale factor: 4x
 */
class LiteRtUpscaler private constructor(
    private var model: CompiledModel,
    private var isGpu: Boolean,
    private val modelFile: File
) : AutoCloseable {

    companion object {
        private const val TAG = "LiteRtUpscaler"
        private const val TILE_SIZE = 128
        private const val SCALE = 4
        private const val INNER_STEP = 112 // 8px boundary padding on each side for artifact-free convolution
        private const val MAX_DIMENSION = 1024 // Keep output within 4096px (GL texture & memory limit)

        /**
         * Creates a LiteRtUpscaler instance.
         * Tries GPU acceleration first, falling back to CPU if compilation fails.
         */
        fun create(modelFile: File): LiteRtUpscaler {
            require(modelFile.exists()) { "Model file does not exist: ${modelFile.absolutePath}" }

            var isGpu = true
            val model = try {
                Log.i(TAG, "Attempting to create LiteRT CompiledModel with GPU: ${modelFile.name}")
                CompiledModel.create(
                    modelFile.absolutePath,
                    CompiledModel.Options(Accelerator.GPU)
                ).also { Log.i(TAG, "LiteRT GPU model compilation successful") }
            } catch (t: Throwable) {
                Log.w(TAG, "LiteRT GPU compilation failed for ${modelFile.name}, falling back to CPU", t)
                isGpu = false
                CompiledModel.create(
                    modelFile.absolutePath,
                    CompiledModel.Options(Accelerator.CPU)
                ).also { Log.i(TAG, "LiteRT CPU model compilation successful") }
            }

            return LiteRtUpscaler(model, isGpu, modelFile)
        }
    }

    /**
     * Upscales the provided bitmap 4x using overlapping patch tiling.
     * Includes seamless GPU-to-CPU runtime fallback if execution throws on GPU.
     */
    fun upscale(
        inputBitmap: Bitmap,
        onProgress: ((Float) -> Unit)? = null
    ): Bitmap {
        // Ensure bitmap can be read
        val readableBitmap = if (inputBitmap.config == Bitmap.Config.HARDWARE) {
            inputBitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            inputBitmap
        }

        // Downscale large input to prevent exceeding GL texture size or OutOfMemory
        val maxDim = max(readableBitmap.width, readableBitmap.height)
        val srcBitmap = if (maxDim > MAX_DIMENSION) {
            val scaleFactor = MAX_DIMENSION.toFloat() / maxDim.toFloat()
            val scaledW = (readableBitmap.width * scaleFactor).roundToInt().coerceAtLeast(1)
            val scaledH = (readableBitmap.height * scaleFactor).roundToInt().coerceAtLeast(1)
            Bitmap.createScaledBitmap(readableBitmap, scaledW, scaledH, true)
        } else {
            readableBitmap
        }

        val inWidth = srcBitmap.width
        val inHeight = srcBitmap.height
        val outWidth = inWidth * SCALE
        val outHeight = inHeight * SCALE

        // Read all input pixels
        val inPixels = IntArray(inWidth * inHeight)
        srcBitmap.getPixels(inPixels, 0, inWidth, 0, 0, inWidth, inHeight)

        // Compute tile start positions
        val xStarts = mutableListOf<Int>()
        var currX = 0
        while (currX < inWidth) {
            xStarts.add(currX)
            currX += INNER_STEP
        }

        val yStarts = mutableListOf<Int>()
        var currY = 0
        while (currY < inHeight) {
            yStarts.add(currY)
            currY += INNER_STEP
        }

        val totalTiles = xStarts.size * yStarts.size
        var completedTiles = 0

        // Allocate buffers for tile inference
        val inputFloatBuffer = FloatArray(TILE_SIZE * TILE_SIZE * 3)
        val outPixels = IntArray(outWidth * outHeight)

        var inputs = model.createInputBuffers()
        var outputs = model.createOutputBuffers()

        try {
            for (y0 in yStarts) {
                val innerH = min(INNER_STEP, inHeight - y0)
                val tileY0 = (y0 - 8).coerceIn(0, max(0, inHeight - TILE_SIZE))
                val localY0 = y0 - tileY0

                for (x0 in xStarts) {
                    val innerW = min(INNER_STEP, inWidth - x0)
                    val tileX0 = (x0 - 8).coerceIn(0, max(0, inWidth - TILE_SIZE))
                    val localX0 = x0 - tileX0

                    // Fill 128x128 input tile (NHWC RGB float [0.0, 1.0])
                    for (ty in 0 until TILE_SIZE) {
                        val srcY = (tileY0 + ty).coerceIn(0, inHeight - 1)
                        val rowOffset = srcY * inWidth
                        val bufRowOffset = ty * TILE_SIZE * 3

                        for (tx in 0 until TILE_SIZE) {
                            val srcX = (tileX0 + tx).coerceIn(0, inWidth - 1)
                            val pixel = inPixels[rowOffset + srcX]

                            val r = ((pixel shr 16) and 0xFF) / 255.0f
                            val g = ((pixel shr 8) and 0xFF) / 255.0f
                            val b = (pixel and 0xFF) / 255.0f

                            val bufIdx = bufRowOffset + tx * 3
                            inputFloatBuffer[bufIdx] = r
                            inputFloatBuffer[bufIdx + 1] = g
                            inputFloatBuffer[bufIdx + 2] = b
                        }
                    }

                    // Run model inference with GPU-to-CPU runtime fallback
                    inputs[0].writeFloat(inputFloatBuffer)
                    try {
                        model.run(inputs, outputs)
                    } catch (t: Throwable) {
                        if (isGpu) {
                            Log.w(TAG, "Tile inference failed on GPU, falling back to CPU", t)
                            inputs.forEach { runCatching { it.close() } }
                            outputs.forEach { runCatching { it.close() } }
                            runCatching { model.close() }

                            isGpu = false
                            model = CompiledModel.create(
                                modelFile.absolutePath,
                                CompiledModel.Options(Accelerator.CPU)
                            )
                            inputs = model.createInputBuffers()
                            outputs = model.createOutputBuffers()

                            inputs[0].writeFloat(inputFloatBuffer)
                            model.run(inputs, outputs)
                        } else {
                            throw t
                        }
                    }

                    // Read 512x512 NCHW output (size: 3 * 512 * 512 = 786,432)
                    val outputPatch = outputs[0].readFloat()

                    // Copy the valid inner region to destination canvas
                    val outLocalX0 = localX0 * SCALE
                    val outLocalY0 = localY0 * SCALE
                    val outInnerW = innerW * SCALE
                    val outInnerH = innerH * SCALE

                    val dstX0 = x0 * SCALE
                    val dstY0 = y0 * SCALE

                    val planeSize = 512 * 512
                    for (oy in 0 until outInnerH) {
                        val patchY = outLocalY0 + oy
                        val dstY = dstY0 + oy
                        val dstRowOffset = dstY * outWidth

                        val patchROffset = patchY * 512
                        val patchGOffset = planeSize + patchY * 512
                        val patchBOffset = 2 * planeSize + patchY * 512

                        for (ox in 0 until outInnerW) {
                            val patchX = outLocalX0 + ox
                            val dstX = dstX0 + ox

                            val rFloat = outputPatch[patchROffset + patchX]
                            val gFloat = outputPatch[patchGOffset + patchX]
                            val bFloat = outputPatch[patchBOffset + patchX]

                            val r = (rFloat.coerceIn(0f, 1f) * 255.0f).roundToInt().coerceIn(0, 255)
                            val g = (gFloat.coerceIn(0f, 1f) * 255.0f).roundToInt().coerceIn(0, 255)
                            val b = (bFloat.coerceIn(0f, 1f) * 255.0f).roundToInt().coerceIn(0, 255)

                            outPixels[dstRowOffset + dstX] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                        }
                    }

                    completedTiles++
                    onProgress?.invoke(completedTiles.toFloat() / totalTiles.toFloat())
                }
            }
        } finally {
            inputs.forEach { runCatching { it.close() } }
            outputs.forEach { runCatching { it.close() } }
        }

        val outBitmap = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888)
        outBitmap.setPixels(outPixels, 0, outWidth, 0, 0, outWidth, outHeight)
        return outBitmap
    }

    override fun close() {
        runCatching { model.close() }
    }
}
