package com.llmhub.llmhub.soundgen

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.CompiledModel.GpuOptions
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Random
import kotlin.math.abs

/**
 * Opt-in physical-device comparison of SoundGen Quick decoders on the LiteRT 2.3.0 GPU: the original (which does not
 * compile for the GPU) and the undilated rewrites in timmyboy26/soundgen-quick-litert-gpu, each checked against the
 * original's CPU output. Instrumentation arguments: `downloadFixtures 1` fetches the decoders on the phone (e.g.
 * BrowserStack); `decoders` (comma-separated: original, 1g, 256m, all), `backend` (opencl, opengl, webgpu),
 * `precision` (fp16, fp32) and `storage` (buffer, texture) select the GPU runs. Results go to logcat.
 */
class SoundGenGpuOptionsDeviceTest {
    @Test fun quickDecoderGpuOptions() {
        val arguments = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "soundgen-gpu-test-fixtures").apply { mkdirs() }
        val decoders = (arguments.getString("decoders") ?: "original,1g,256m,all").split(",").map { it.trim() }
        val files = (listOf("original") + decoders).distinct().associateWith { name ->
            val (url, sha256) = DECODERS.getValue(name)
            File(directory, url.substringAfterLast('/')).also {
                if (arguments.getString("downloadFixtures") == "1") OnDeviceFixtures.download(it, url, sha256)
                assertTrue("Missing fixture ${it.name}", it.isFile)
            }
        }
        val gpuOptions = GpuOptions(
            backend = when (arguments.getString("backend")) {
                "opencl" -> GpuOptions.Backend.OPENCL
                "opengl" -> GpuOptions.Backend.OPENGL
                "webgpu" -> GpuOptions.Backend.WEBGPU
                else -> null
            },
            precision = when (arguments.getString("precision")) {
                "fp16" -> GpuOptions.Precision.FP16
                "fp32" -> GpuOptions.Precision.FP32
                else -> null
            },
            bufferStorageType = when (arguments.getString("storage")) {
                "buffer" -> GpuOptions.BufferStorageType.BUFFER
                "texture" -> GpuOptions.BufferStorageType.TEXTURE_2D
                else -> null
            },
        )
        val report = JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL).put("android", Build.VERSION.RELEASE)
            .put("gpuOptions", gpuOptions.toString())
        Log.i(TAG, "device: $report")

        val latent = FloatArray(16_384).also { values -> Random(7).let { r -> values.indices.forEach { values[it] = r.nextGaussian().toFloat() } } }
        var reference: FloatArray? = null
        val runs = listOf("original cpu" to (files.getValue("original") to CompiledModel.Options(Accelerator.CPU))) +
            decoders.map { "$it gpu" to (files.getValue(it) to CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply { this.gpuOptions = gpuOptions }) }
        for ((label, run) in runs) {
            val (file, options) = run
            val entry = JSONObject()
            report.put(label, entry)
            Log.i(TAG, "starting $label")
            try {
                var started = SystemClock.elapsedRealtime()
                CompiledModel.create(file.path, options).use { model ->
                    entry.put("compileMs", SystemClock.elapsedRealtime() - started)
                    val inputs = model.createInputBuffers()
                    val outputs = model.createOutputBuffers()
                    try {
                        inputs[0].writeFloat(latent)
                        started = SystemClock.elapsedRealtime()
                        model.run(inputs, outputs)
                        outputs[0].readFloat()
                        entry.put("firstRunMs", SystemClock.elapsedRealtime() - started)
                        // GPU runs are asynchronous; reading the output waits for them.
                        started = SystemClock.elapsedRealtime()
                        model.run(inputs, outputs)
                        val audio = outputs[0].readFloat()
                        entry.put("runMs", SystemClock.elapsedRealtime() - started)
                        val base = reference ?: audio.also { reference = it }
                        var maxDiff = 0.0
                        var sumDiff = 0.0
                        for (i in audio.indices) {
                            val diff = abs(audio[i].toDouble() - base[i])
                            maxDiff = maxOf(maxDiff, diff)
                            sumDiff += diff
                        }
                        entry.put("finite", audio.all { it.isFinite() }).put("maxAbsDiff", maxDiff).put("meanAbsDiff", sumDiff / audio.size)
                            .put("absMean", audio.map { abs(it.toDouble()) }.average())
                    } finally {
                        inputs.forEach { it.close() }
                        outputs.forEach { it.close() }
                    }
                }
            } catch (e: Throwable) {
                entry.put("error", e.message ?: e.toString())
            }
            Log.i(TAG, "$label: $entry")
        }
        Log.i(TAG, "RESULT $report")
    }

    private companion object {
        const val TAG = "SoundGenGpuOptions"
        const val UNDILATED = "https://huggingface.co/timmyboy26/soundgen-quick-litert-gpu/resolve/610d2ad5b7058f53da8465b584d042b6ee8ed6f9/"
        val DECODERS = mapOf(
            "original" to ("https://huggingface.co/jegly/audio/resolve/e50bf9dff759c8622371a9c43ede989ea2d04b32/autoencoder_model.tflite" to
                "e5483ee50915af1ac419fce5206f14b55572176fcb9cf327ef48432d4c5feba2"),
            "1g" to (UNDILATED + "autoencoder_model_undilated_1g.tflite" to "15c9c8236a48c64075124cad0737e5d8cb35c154f78cb40756db5d882076ec47"),
            "256m" to (UNDILATED + "autoencoder_model_undilated_256m.tflite" to "f3e47c496633fe256992c8672e0411a05cb1a8b9af8bf15574d39a34818d20e1"),
            "all" to (UNDILATED + "autoencoder_model_undilated_all.tflite" to "a0659c851cdf02e1418be9ebd7adeeccbe52face3815863f4f95ee638b2ffb49"),
        )
    }
}
