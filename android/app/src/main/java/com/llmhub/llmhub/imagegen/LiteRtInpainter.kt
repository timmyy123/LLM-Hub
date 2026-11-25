package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.llmhub.llmhub.data.InpaintKind
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Object Eraser: on-device inpainting with LiteRT CompiledModel, GPU first with automatic CPU fallback
 * (at load and, like LiteRtUpscaler, if a GPU run throws). No NPU.
 *
 * - MI-GAN 512 (Places2): one input [1,4,512,512] = concat(mask - 0.5, rgb * mask), rgb in [-1,1],
 *   mask 1 = keep, 0 = erase. Output [1,3,512,512] in [-1,1].
 * - Big LaMa 512: input 0 image [1,3,512,512] RGB 0..1 with the hole zeroed, input 1 mask [1,1,512,512]
 *   1 = fill. Output [1,3,512,512] RGB 0..1.
 *
 * Each separately painted region (4-connected) is erased on its own: a square crop around it (MI-GAN 1.4x the
 * region, LaMa 2x, at least 512 px, clamped to the photo) is scaled to 512, filled, scaled back, and only that
 * region's painted pixels are replaced in the full-resolution photo, so the rest keeps its full quality.
 * Not thread-safe: run [inpaint] and [close] from one caller at a time.
 */
class LiteRtInpainter private constructor(
    val kind: InpaintKind,
    private val modelFile: File,
    private var model: CompiledModel,
    isGpu: Boolean,
) : AutoCloseable {

    /** False once the model runs on the CPU (GPU compile or run failed). */
    var isGpu: Boolean = isGpu
        private set

    private var inputs: List<TensorBuffer> = model.createInputBuffers()
    private var outputs: List<TensorBuffer> = model.createOutputBuffers()

    companion object {
        private const val TAG = "LiteRtInpainter"
        const val N = 512
        private const val PLANE = N * N
        /** Painted pixels the 512 mask grows by, so object edges blended in by the downscale are filled too. */
        private const val MASK_GROW = 2

        /** Loads the model on the GPU, or on the CPU if the GPU cannot compile it. Slow: call off the main thread. */
        fun create(modelFile: File, kind: InpaintKind): LiteRtInpainter {
            require(modelFile.isFile) { "Model file does not exist: ${modelFile.absolutePath}" }
            return try {
                val gpu = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(Accelerator.GPU))
                Log.i(TAG, "${modelFile.parentFile?.name}: GPU")
                LiteRtInpainter(kind, modelFile, gpu, true)
            } catch (t: Throwable) {
                Log.w(TAG, "GPU compile failed for ${modelFile.parentFile?.name}, using the CPU", t)
                val cpu = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(Accelerator.CPU))
                LiteRtInpainter(kind, modelFile, cpu, false)
            }
        }
    }

    /**
     * Erases the painted pixels of [mask] (w*h bytes, non-zero = erase) from [source] and returns a new bitmap.
     * [onProgress] gets (regions done, regions in all). Checks for cancellation between regions.
     */
    suspend fun inpaint(source: Bitmap, mask: ByteArray, onProgress: (Int, Int) -> Unit = { _, _ -> }): Bitmap {
        val w = source.width
        val h = source.height
        require(mask.size == w * h) { "Mask size ${mask.size} does not match ${w}x$h" }
        val result = source.copy(Bitmap.Config.ARGB_8888, true)

        val regions = labelRegions(mask, w, h)
        onProgress(0, regions.boxes.size)
        val factor = if (kind == InpaintKind.LAMA) 2f else 1.4f
        regions.boxes.forEachIndexed { index, box ->
            currentCoroutineContext().ensureActive()
            val regionId = index + 1
            val extent = max(box.right - box.left, box.bottom - box.top) + 1
            val side = max(N, (extent * factor).toInt())
            // Square when the photo allows it; a region wider than the photo is short gets a rectangle instead.
            val cropW = min(side, w)
            val cropH = min(side, h)
            val cx = ((box.left + box.right) / 2 - cropW / 2).coerceIn(0, w - cropW)
            val cy = ((box.top + box.bottom) / 2 - cropH / 2).coerceIn(0, h - cropH)

            val cropPx = IntArray(cropW * cropH)
            result.getPixels(cropPx, 0, cropW, cx, cy, cropW, cropH)
            val cropMask = IntArray(cropW * cropH)
            for (yy in 0 until cropH) {
                val row = (cy + yy) * w + cx
                for (xx in 0 until cropW) {
                    cropMask[yy * cropW + xx] = if (regions.label[row + xx] == regionId) -1 else 0xFF000000.toInt()
                }
            }
            val small = scaledPixels(cropPx, cropW, cropH)
            val hole = grow(scaledPixels(cropMask, cropW, cropH).let { px -> BooleanArray(PLANE) { (px[it] and 0xFF) > 127 } })

            val out = run(small, hole)
            val generated = Bitmap.createBitmap(N, N, Bitmap.Config.ARGB_8888)
            generated.setPixels(out, 0, N, 0, 0, N, N)
            val back = IntArray(cropW * cropH)
            val scaledBack = Bitmap.createScaledBitmap(generated, cropW, cropH, true)
            scaledBack.getPixels(back, 0, cropW, 0, 0, cropW, cropH)
            if (scaledBack !== generated) scaledBack.recycle()
            generated.recycle()

            for (yy in 0 until cropH) {
                val row = (cy + yy) * w + cx
                for (xx in 0 until cropW) {
                    if (regions.label[row + xx] == regionId) cropPx[yy * cropW + xx] = back[yy * cropW + xx]
                }
            }
            result.setPixels(cropPx, 0, cropW, cx, cy, cropW, cropH)
            onProgress(index + 1, regions.boxes.size)
        }
        return result
    }

    /** One 512 pass: [pixels] ARGB, [hole] true = fill. Returns ARGB pixels. */
    private fun run(pixels: IntArray, hole: BooleanArray): IntArray {
        val feeds: List<FloatArray> = when (kind) {
            InpaintKind.MIGAN -> {
                val x = FloatArray(4 * PLANE)
                for (i in 0 until PLANE) {
                    val keep = if (hole[i]) 0f else 1f
                    val p = pixels[i]
                    x[i] = keep - 0.5f
                    x[PLANE + i] = (((p shr 16) and 0xFF) / 127.5f - 1f) * keep
                    x[2 * PLANE + i] = (((p shr 8) and 0xFF) / 127.5f - 1f) * keep
                    x[3 * PLANE + i] = ((p and 0xFF) / 127.5f - 1f) * keep
                }
                listOf(x)
            }
            InpaintKind.LAMA -> {
                val image = FloatArray(3 * PLANE)
                val m = FloatArray(PLANE)
                for (i in 0 until PLANE) {
                    val fill = if (hole[i]) 1f else 0f
                    val keep = 1f - fill
                    val p = pixels[i]
                    m[i] = fill
                    image[i] = ((p shr 16) and 0xFF) / 255f * keep
                    image[PLANE + i] = ((p shr 8) and 0xFF) / 255f * keep
                    image[2 * PLANE + i] = (p and 0xFF) / 255f * keep
                }
                listOf(image, m)
            }
        }
        val out = try {
            invoke(feeds)
        } catch (t: Throwable) {
            if (!isGpu) throw t
            Log.w(TAG, "GPU run failed, reloading on the CPU", t)
            reloadOnCpu()
            invoke(feeds)
        }
        return IntArray(PLANE) { i ->
            val r: Int; val g: Int; val b: Int
            if (kind == InpaintKind.MIGAN) {
                r = ((out[i] + 1f) * 127.5f).roundToInt().coerceIn(0, 255)
                g = ((out[PLANE + i] + 1f) * 127.5f).roundToInt().coerceIn(0, 255)
                b = ((out[2 * PLANE + i] + 1f) * 127.5f).roundToInt().coerceIn(0, 255)
            } else {
                r = (out[i] * 255f).roundToInt().coerceIn(0, 255)
                g = (out[PLANE + i] * 255f).roundToInt().coerceIn(0, 255)
                b = (out[2 * PLANE + i] * 255f).roundToInt().coerceIn(0, 255)
            }
            (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }

    private fun invoke(feeds: List<FloatArray>): FloatArray {
        feeds.forEachIndexed { index, data -> inputs[index].writeFloat(data) }
        model.run(inputs, outputs)
        return outputs[0].readFloat()
    }

    private fun reloadOnCpu() {
        releaseBuffers()
        runCatching { model.close() }
        isGpu = false
        model = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(Accelerator.CPU))
        inputs = model.createInputBuffers()
        outputs = model.createOutputBuffers()
    }

    private fun scaledPixels(px: IntArray, width: Int, height: Int): IntArray {
        val bitmap = Bitmap.createBitmap(px, width, height, Bitmap.Config.ARGB_8888)
        val scaled = Bitmap.createScaledBitmap(bitmap, N, N, true)
        val out = IntArray(PLANE)
        scaled.getPixels(out, 0, N, 0, 0, N, N)
        if (scaled !== bitmap) scaled.recycle()
        bitmap.recycle()
        return out
    }

    /** Grows the hole by [MASK_GROW] pixels (square neighbourhood). */
    private fun grow(hole: BooleanArray): BooleanArray {
        val grown = hole.copyOf()
        for (y in 0 until N) for (x in 0 until N) {
            if (!hole[y * N + x]) continue
            for (dy in -MASK_GROW..MASK_GROW) {
                val yy = y + dy
                if (yy !in 0 until N) continue
                for (dx in -MASK_GROW..MASK_GROW) {
                    val xx = x + dx
                    if (xx in 0 until N) grown[yy * N + xx] = true
                }
            }
        }
        return grown
    }

    private class Box(var left: Int, var top: Int, var right: Int, var bottom: Int)
    private class Regions(val label: IntArray, val boxes: List<Box>)

    /** Labels the painted pixels into 4-connected regions (1, 2, ...) with their bounding boxes. */
    private fun labelRegions(mask: ByteArray, w: Int, h: Int): Regions {
        val label = IntArray(w * h)
        val boxes = ArrayList<Box>()
        var stack = IntArray(1024)
        for (start in 0 until w * h) {
            if (mask[start].toInt() == 0 || label[start] != 0) continue
            val id = boxes.size + 1
            val box = Box(start % w, start / w, start % w, start / w)
            boxes += box
            var top = 0
            stack[top++] = start
            label[start] = id
            while (top > 0) {
                val q = stack[--top]
                val qx = q % w
                val qy = q / w
                if (qx < box.left) box.left = qx
                if (qx > box.right) box.right = qx
                if (qy < box.top) box.top = qy
                if (qy > box.bottom) box.bottom = qy
                if (top + 4 > stack.size) stack = stack.copyOf(stack.size * 2)
                if (qx > 0) { val n = q - 1; if (label[n] == 0 && mask[n].toInt() != 0) { label[n] = id; stack[top++] = n } }
                if (qx < w - 1) { val n = q + 1; if (label[n] == 0 && mask[n].toInt() != 0) { label[n] = id; stack[top++] = n } }
                if (qy > 0) { val n = q - w; if (label[n] == 0 && mask[n].toInt() != 0) { label[n] = id; stack[top++] = n } }
                if (qy < h - 1) { val n = q + w; if (label[n] == 0 && mask[n].toInt() != 0) { label[n] = id; stack[top++] = n } }
            }
        }
        return Regions(label, boxes)
    }

    private fun releaseBuffers() {
        inputs.forEach { runCatching { it.close() } }
        outputs.forEach { runCatching { it.close() } }
    }

    override fun close() {
        releaseBuffers()
        runCatching { model.close() }
    }
}
