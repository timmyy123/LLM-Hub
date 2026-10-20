package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check of SD1.5 LiteRT NPU builds (timmyboy26/stable-diffusion-v1-5-LiteRT-NPU).
 * Stage an AOT build in files/litert_sd-test-fixtures/aot and the uncompiled models in .../source, each with
 * vocab.json and merges.txt. Results go to files/litert_sd-test-results.
 */
class LiteRtSdDeviceTest {
    @Test fun npuBuildGenerates() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = File(context.filesDir, "litert_sd-test-fixtures")
        val results = File(context.filesDir, "litert_sd-test-results").apply { mkdirs() }
        val report = JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL).put("android", Build.VERSION.RELEASE)
        fun save() = File(results, "results.json").writeText(report.toString(2))
        val prompt = "a red ceramic teapot on a wooden table, studio photograph"

        fun run(label: String, dir: String, accelerators: Set<Accelerator>, steps: Int) {
            val modelDir = File(fixtures, dir)
            assertTrue("Missing fixtures in $modelDir", File(modelDir, "diffusion.tflite").isFile)
            LiteRtSdPipeline(context, modelDir, accelerators).use { pipeline ->
                val image = pipeline.generate(prompt, "", steps, 7.5f, 12345)
                File(results, "$label.png").outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
                report.put(label, JSONObject()
                    .put("available", pipeline.availableAccelerators.map { it.name })
                    .put("load", JSONObject(pipeline.loadTimings as Map<*, *>))
                    .put("steps", steps)
                    .put("timings", JSONObject(pipeline.lastTimings as Map<*, *>)))
                save()
                Log.i("LiteRtSdDeviceTest", "$label: ${report.getJSONObject(label)}")
                val pixels = IntArray(image.width * image.height).also { image.getPixels(it, 0, image.width, 0, 0, image.width, image.height) }
                assertTrue("$label must produce an image, not a flat field", pixels.distinct().size > 1000)
            }
        }

        run("npu-aot-20-steps", "aot", setOf(Accelerator.NPU, Accelerator.CPU), 20)
        run("cpu-2-steps", "source", setOf(Accelerator.CPU), 2)
        val npuStep = report.getJSONObject("npu-aot-20-steps").getJSONObject("timings").getLong("denoise") / 20.0
        val cpuStep = report.getJSONObject("cpu-2-steps").getJSONObject("timings").getLong("denoise") / 2.0
        report.put("npu-speedup-per-step", cpuStep / npuStep)
        save()
    }
}
