package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs
import kotlin.math.sin

/**
 * Opt-in physical-device check of MI-GAN 512 Places2 (litert-community LiteRT build) on each LiteRT accelerator.
 * The phone downloads the pinned file itself. A synthetic photo (gradient plus stripes) has a 160x160 centre
 * square erased; each run logs load and inference time, output range and how well the fill matches its
 * surroundings, and writes the composite to files/migan-test/ for a visual check. Tag: MiganDevice.
 * The model card says XNNPACK declines this fp16 graph; on LiteRT 2.4.0 CompiledModel it does not (Tab S9+:
 * 495 of 509 nodes on XNNPACK, about 1.2 s per run; GPU about 47 ms, same output).
 */
class MiganDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun fp16Gpu() = run(FP16, Accelerator.GPU)
    @Test fun fp16Cpu() = run(FP16, Accelerator.CPU)
    /** Hexagon (or another vendor NPU): LiteRT compiles the published file for the NPU on the device at load. */
    @Test fun fp16Npu() = run(FP16, Accelerator.NPU)

    private fun run(fixture: Fixture, accelerator: Accelerator) {
        val label = "${fixture.file} ${accelerator.name}"
        Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} soc=${Build.SOC_MODEL} sdk=${Build.VERSION.SDK_INT} case=$label")
        val dir = File(context.filesDir, "models").apply { mkdirs() }
        val file = File(dir, fixture.file)
        OnDeviceFixtures.download(file, fixture.url, fixture.sha256)

        val (rgb, mask) = syntheticInput()
        val x = FloatArray(4 * N * N)
        for (i in 0 until N * N) {
            x[i] = mask[i] - 0.5f
            for (c in 0 until 3) x[(c + 1) * N * N + i] = rgb[c * N * N + i] * mask[i]
        }

        // The NPU needs the vendor dispatch and compiler plugin libraries, set up the way LiteRtSdPipeline does.
        // On Snapdragon, compiling on the device needs the QNN runtime next to the dispatch and compiler plugin
        // libraries, so all three go in the QAIRT runtime folder (as Parakeet and the v85 upscaler do).
        val environment = if (accelerator == Accelerator.NPU) {
            val dir = if (com.llmhub.llmhub.data.DeviceInfo.isSnapdragon()) {
                com.llmhub.llmhub.data.QairtRuntime.stage(context).also { dir ->
                    for (name in listOf("libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so")) {
                        val library = File(context.applicationInfo.nativeLibraryDir, name)
                        val copy = File(dir, name)
                        if (library.isFile && copy.length() != library.length()) library.copyTo(copy, overwrite = true)
                    }
                }
            } else LiteRtSdPipeline.vendorLibraryDir(context)
            Log.i(TAG, "NPU library dir ${dir?.path}: ${dir?.list()?.sorted()}")
            dir?.let { Environment.create(mapOf(Environment.Option.DispatchLibraryDir to it.path, Environment.Option.CompilerPluginLibraryDir to it.path)) }
                ?: Environment.create(BuiltinNpuAcceleratorProvider(context))
        } else null
        fun load() = if (environment != null) {
            CompiledModel.create(file.absolutePath, CompiledModel.Options(setOf(accelerator)), environment)
        } else {
            CompiledModel.create(file.absolutePath, CompiledModel.Options(accelerator))
        }
        val loadStart = SystemClock.elapsedRealtime()
        var model = try {
            load()
        } catch (t: Throwable) {
            Log.e(TAG, "RESULT $label LOAD FAILED: ${t.javaClass.simpleName}: ${t.message}", t)
            environment?.close()
            throw t
        }
        var loadMs = SystemClock.elapsedRealtime() - loadStart
        if (environment != null) {
            // The first NPU load includes on-device compilation; a second load shows what a later launch pays.
            model.close()
            val again = SystemClock.elapsedRealtime()
            model = load()
            Log.i(TAG, "LOAD $label first=${loadMs}ms second=${SystemClock.elapsedRealtime() - again}ms")
            loadMs = SystemClock.elapsedRealtime() - again
        }
        try {
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            val times = mutableListOf<Long>()
            var out = FloatArray(0)
            repeat(6) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                inputs[0].writeFloat(x)
                model.run(inputs, outputs)
                out = outputs[0].readFloat()
                times += (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000
            }
            assertTrue("$label output size ${out.size}", out.size == 3 * N * N)
            val finite = out.all { it.isFinite() }
            val min = out.min(); val max = out.max()
            // Fill quality: mean abs error between the inpainted hole and the known ground truth there.
            var err = 0.0; var holes = 0
            for (i in 0 until N * N) if (mask[i] == 0f) {
                for (c in 0 until 3) err += abs(out[c * N * N + i] - rgb[c * N * N + i]); holes++
            }
            val holeMae = err / (holes * 3)
            val warm = times.drop(1).sorted()
            Log.i(TAG, "RESULT $label ok load=${loadMs}ms first=${times[0]}ms median=${warm[warm.size / 2]}ms " +
                "finite=$finite range=[${"%.3f".format(min)}, ${"%.3f".format(max)}] holeMAE=${"%.4f".format(holeMae)}")
            save(rgb, mask, out, File(context.filesDir, "migan-test/${fixture.file}-${accelerator.name}.png"))
            assertTrue("$label non-finite output", finite)
        } finally {
            model.close()
            environment?.close()
        }
    }

    /**
     * A real photo: files/migan-photo/photo.jpg with masks mask_*.png (white = erase), each erased on its own.
     * Crops a square around the mask (512 px, or larger and scaled down to 512), inpaints on the GPU, scales the
     * result back and replaces only the masked pixels, then writes out_*.jpg next to the inputs.
     */
    @Test fun photoGpu() {
        val dir = File(context.filesDir, "migan-photo")
        val photo = BitmapFactory.decodeFile(File(dir, "photo.jpg").path)
        val file = File(File(context.filesDir, "models").apply { mkdirs() }, FP16.file)
        OnDeviceFixtures.download(file, FP16.url, FP16.sha256)
        val model = CompiledModel.create(file.absolutePath, CompiledModel.Options(Accelerator.GPU))
        try {
            val inputs = model.createInputBuffers(); val outputs = model.createOutputBuffers()
            for (maskFile in dir.listFiles { f -> f.name.startsWith("mask_") }.orEmpty().sortedBy { it.name }) {
                val t0 = SystemClock.elapsedRealtime()
                val mask = BitmapFactory.decodeFile(maskFile.path)
                val w = photo.width; val h = photo.height
                val mp = IntArray(w * h).also { mask.getPixels(it, 0, w, 0, 0, w, h) }
                var x0 = w; var y0 = h; var x1 = 0; var y1 = 0
                for (y in 0 until h) for (x in 0 until w) if ((mp[y * w + x] and 0xFF) > 127) {
                    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y
                }
                val side = maxOf(N, ((maxOf(x1 - x0, y1 - y0) + 1) * 1.4f).toInt()).coerceAtMost(minOf(w, h))
                val cx = ((x0 + x1) / 2 - side / 2).coerceIn(0, w - side)
                val cy = ((y0 + y1) / 2 - side / 2).coerceIn(0, h - side)
                val crop = Bitmap.createScaledBitmap(Bitmap.createBitmap(photo, cx, cy, side, side), N, N, true)
                val cropMask = Bitmap.createScaledBitmap(Bitmap.createBitmap(mask, cx, cy, side, side), N, N, true)
                val cp = IntArray(N * N).also { crop.getPixels(it, 0, N, 0, 0, N, N) }
                val km = IntArray(N * N).also { cropMask.getPixels(it, 0, N, 0, 0, N, N) }
                val x = FloatArray(4 * N * N)
                for (i in 0 until N * N) {
                    val keep = if ((km[i] and 0xFF) > 127) 0f else 1f
                    x[i] = keep - 0.5f
                    x[N * N + i] = (((cp[i] shr 16) and 0xFF) / 127.5f - 1f) * keep
                    x[2 * N * N + i] = (((cp[i] shr 8) and 0xFF) / 127.5f - 1f) * keep
                    x[3 * N * N + i] = ((cp[i] and 0xFF) / 127.5f - 1f) * keep
                }
                val t1 = SystemClock.elapsedRealtime()
                inputs[0].writeFloat(x); model.run(inputs, outputs); val out = outputs[0].readFloat()
                val inferMs = SystemClock.elapsedRealtime() - t1
                fun c(v: Float) = ((v + 1f) * 127.5f).toInt().coerceIn(0, 255)
                val gen = Bitmap.createBitmap(IntArray(N * N) { i ->
                    (0xFF shl 24) or (c(out[i]) shl 16) or (c(out[N * N + i]) shl 8) or c(out[2 * N * N + i])
                }, N, N, Bitmap.Config.ARGB_8888)
                val back = Bitmap.createScaledBitmap(gen, side, side, true)
                val bp = IntArray(side * side).also { back.getPixels(it, 0, side, 0, 0, side, side) }
                val result = photo.copy(Bitmap.Config.ARGB_8888, true)
                val rp = IntArray(w * h).also { result.getPixels(it, 0, w, 0, 0, w, h) }
                for (yy in 0 until side) for (xx in 0 until side) {
                    val gi = (cy + yy) * w + (cx + xx)
                    if ((mp[gi] and 0xFF) > 127) rp[gi] = bp[yy * side + xx]
                }
                result.setPixels(rp, 0, w, 0, 0, w, h)
                val target = File(dir, "out_" + maskFile.nameWithoutExtension.removePrefix("mask_") + ".jpg")
                target.outputStream().use { result.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                Log.i(TAG, "RESULT photo ${maskFile.name} crop=${side}px at ($cx,$cy) inference=${inferMs}ms total=${SystemClock.elapsedRealtime() - t0}ms")
            }
        } finally {
            model.close()
        }
    }

    private fun syntheticInput(): Pair<FloatArray, FloatArray> {
        val rgb = FloatArray(3 * N * N); val mask = FloatArray(N * N) { 1f }
        for (y in 0 until N) for (x in 0 until N) {
            val i = y * N + x
            rgb[i] = x / 255.5f - 1f
            rgb[N * N + i] = y / 255.5f - 1f
            rgb[2 * N * N + i] = 0.6f * sin(x / 12f) * sin(y / 17f)
            if (x in 176 until 336 && y in 176 until 336) mask[i] = 0f
        }
        return rgb to mask
    }

    private fun save(rgb: FloatArray, mask: FloatArray, out: FloatArray, target: File) {
        target.parentFile?.mkdirs()
        val bmp = Bitmap.createBitmap(N, N, Bitmap.Config.ARGB_8888)
        val px = IntArray(N * N) { i ->
            fun ch(c: Int): Int {
                val v = rgb[c * N * N + i] * mask[i] + out[c * N * N + i] * (1 - mask[i])
                return ((v + 1f) * 127.5f).toInt().coerceIn(0, 255)
            }
            (0xFF shl 24) or (ch(0) shl 16) or (ch(1) shl 8) or ch(2)
        }
        bmp.setPixels(px, 0, N, 0, 0, N, N)
        target.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    private data class Fixture(val file: String, val url: String, val sha256: String)

    private companion object {
        const val TAG = "MiganDevice"
        const val N = 512
        val FP16 = Fixture(
            "migan_fp16.tflite",
            "https://huggingface.co/litert-community/MI-GAN-512-Places2-LiteRT/resolve/179a35d8e8a518f6253f9c42fc9b412c6438cf7c/migan_fp16.tflite",
            "ef53f8dca69e5ce29629441128322c4d9a0a527b2a043703e0ab497f2463c25f",
        )
    }
}
