package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
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

        val loadStart = SystemClock.elapsedRealtime()
        val model = try {
            CompiledModel.create(file.absolutePath, CompiledModel.Options(accelerator))
        } catch (t: Throwable) {
            Log.e(TAG, "RESULT $label LOAD FAILED: ${t.javaClass.simpleName}: ${t.message}", t)
            throw t
        }
        val loadMs = SystemClock.elapsedRealtime() - loadStart
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
