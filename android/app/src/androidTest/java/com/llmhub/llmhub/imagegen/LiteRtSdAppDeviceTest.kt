package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.ModelData
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check of the in-app SD1.5 LiteRT NPU path: this SoC's model entry, files laid out as the
 * downloader writes them, discovery, then load and generate through [ImageGeneratorHelper] and the `:litert_sd`
 * process. Stage the build's five files in files/litert_sd_app-test-fixtures/app; results go to
 * files/litert_sd_app-test-results.
 */
class LiteRtSdAppDeviceTest {
    @Test fun appGeneratesOrFailsWithoutCrashing() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = File(context.filesDir, "litert_sd_app-test-fixtures/app")
        val results = File(context.filesDir, "litert_sd_app-test-results").apply { mkdirs() }
        val report = JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL).put("android", Build.VERSION.RELEASE)
        fun save() = File(results, "results.json").writeText(report.toString(2))

        val entry = ModelData.models.firstOrNull { it.modelFormat == "litert_npu" }
        assertNotNull("No LiteRT Stable Diffusion build is offered for ${Build.SOC_MODEL}", entry)
        val modelDir = File(File(context.filesDir, "sd_models"), entry!!.name.replace(" ", "_")).apply { mkdirs() }
        for (url in listOf(entry.url) + entry.additionalFiles) {
            val name = url.substringAfterLast('/').substringBefore('?')
            val source = File(fixtures, name)
            assertTrue("Missing fixture $name", source.isFile)
            val target = File(modelDir, name)
            if (target.length() != source.length()) source.copyTo(target, overwrite = true)
        }
        assertEquals("Staged files must match the entry's download size", entry.sizeBytes, modelDir.listFiles()!!.sumOf { it.length() })

        val info = StableDiffusionHelper(context).listModels().firstOrNull { it.path == modelDir.absolutePath }
        assertEquals(ModelType.LITERT_NPU, info?.type)
        report.put("entry", entry.name)

        val helper = ImageGeneratorHelper(context)
        try {
            var started = SystemClock.elapsedRealtime()
            assertTrue("The LiteRT build must load", helper.initialize(info!!.path, info.type))
            report.put("loadMs", SystemClock.elapsedRealtime() - started)
            save()

            started = SystemClock.elapsedRealtime()
            val outcome = runCatching { helper.generateImage("a red ceramic teapot on a wooden table, studio photograph", iterations = 20, seed = 12345) }
            report.put("generateMs", SystemClock.elapsedRealtime() - started)
            val image = outcome.getOrNull()
            if (image != null) {
                File(results, "app-20-steps.png").outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
                val pixels = IntArray(image.width * image.height).also { image.getPixels(it, 0, image.width, 0, 0, image.width, image.height) }
                report.put("size", "${image.width}x${image.height}").put("distinctColors", pixels.distinct().size)
            } else {
                report.put("error", outcome.exceptionOrNull()?.message ?: "no image")
            }
            save()
            Log.i("LiteRtSdAppDeviceTest", report.toString())
            // Reaching this line means a crash in the NPU driver did not take down the app's process.
            if (image == null) {
                assertEquals("A failed generation must report the NPU crash", context.getString(com.llmhub.llmhub.R.string.image_generator_npu_crashed), report.getString("error"))
            } else {
                assertEquals("512x512", report.getString("size"))
                assertTrue("The image must not be a flat field", report.getInt("distinctColors") > 1000)
            }
        } finally {
            helper.close()
            modelDir.deleteRecursively()
        }
    }
}
