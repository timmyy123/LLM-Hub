package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.ModelDownloader
import com.llmhub.llmhub.service.SDBackendService
import io.ktor.client.HttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Opt-in sweep over every QNN NPU image model the catalog offers this device: download through the app's
 * ModelDownloader, load and generate through ImageGeneratorHelper, save the image, then delete the model.
 * The next model downloads while the current one is tested, so the phone never waits for a download it could overlap.
 *
 * Arguments (am instrument -e): `models` comma-separated name substrings (default: all), `keep` true to keep
 * downloads, `prompt`, `compare` true to also render each model with the pre-fix settings (DPM++, the guidance
 * slider's CFG, 8 steps for DMD2/Anima instead of its config.json) into `<name>_legacy.jpg`.
 * Results go to files/npu-sweep-results: results.json and one JPEG per model.
 */
class NpuImageModelSweepDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val args = InstrumentationRegistry.getArguments()
    private val compare = args.getString("compare") == "true"
    private val results = File(context.filesDir, "npu-sweep-results").apply { mkdirs() }

    @Test fun sweep() = runBlocking {
        val filters = args.getString("models")?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }.orEmpty()
        val keep = args.getString("keep") == "true"
        val prompt = args.getString("prompt") ?: "a red fox sitting in fresh snow at sunrise, detailed photograph"
        val catalog = ModelData.models.filter { it.category == "image_generation" && it.modelFormat == "qnn_npu" }
            .filter { m -> filters.isEmpty() || filters.any { m.name.contains(it, ignoreCase = true) } }
            // Models run in the order of the first substring they match, so earlier substrings go first.
            .sortedBy { m -> filters.indexOfFirst { m.name.contains(it, ignoreCase = true) } }
        Log.i(TAG, "Sweeping ${catalog.size} models on ${Build.MODEL} ${Build.SOC_MODEL}")
        val summary = File(results, "results.json")
        val records = summary.takeIf { it.isFile }?.let { JSONObject(it.readText()).optJSONArray("models") } ?: JSONArray()
        // A rerun resumes: models that already passed in results.json, or are named in `skip` ('|'-separated,
        // e.g. passed on an earlier device rental), are skipped.
        val passed = (0 until records.length()).map { records.getJSONObject(it) }
            .filter { it.optString("result") == "PASS" && (!compare || it.has("legacyGenerateMs")) }
            .map { it.getString("name") }.toSet() +
            args.getString("skip").orEmpty().split('|').map { it.trim() }.filter { it.isNotEmpty() }
        val models = catalog.filter { it.name !in passed }
        Log.i(TAG, "${passed.size} already passed; ${models.size} to run")
        val downloader = ModelDownloader(HttpClient(Android), context, null)
        val failures = mutableListOf<String>()

        // Each download returns its duration in ms, or the error it hit.
        fun download(index: Int): Deferred<Result<Long>>? = models.getOrNull(index)?.let { model ->
            async(Dispatchers.IO) { runCatching {
                val started = SystemClock.elapsedRealtime()
                downloader.downloadModel(model).collect { }
                SystemClock.elapsedRealtime() - started
            } }
        }
        var pending = download(0)

        for ((index, model) in models.withIndex()) {
            val record = JSONObject().put("name", model.name).put("url", model.url)
            val dir = File(File(context.filesDir, "sd_models"), model.name.replace(" ", "_"))
            val current = pending
            pending = null
            try {
                record.put("downloadMs", current!!.await().getOrThrow())
                pending = download(index + 1)
                var started: Long
                val info = StableDiffusionHelper(context).listModels().firstOrNull { File(it.path) == dir }
                    ?: error("Downloaded model not found under ${dir.path}")
                record.put("type", info.type.name)

                val helper = ImageGeneratorHelper(context)
                try {
                    started = SystemClock.elapsedRealtime()
                    check(helper.initialize(info.path, info.type, false)) { "initialize() returned false" }
                    record.put("loadMs", SystemClock.elapsedRealtime() - started)
                    // Distilled models (DMD2, Anima Turbo) run with the scheduler/steps/CFG from their own config.json.
                    val extras = ImageExtrasStore(context).load()
                    val distilled = ModelData.isAnimaModel(info.path) || model.name.contains("DMD2")
                    val bundled = helper.bundledDefaults
                    val steps = bundled?.steps ?: 20
                    started = SystemClock.elapsedRealtime()
                    val bitmap = helper.generateImage(prompt, iterations = steps, seed = 42)
                        ?: error("generateImage() returned null")
                    if (compare) {
                        // The request the app sent before the config.json fix: DPM++ and the guidance slider's CFG.
                        val size = if (ModelData.isSdxlModel(info.path) || ModelData.isAnimaModel(info.path)) 1024 else 512
                        val legacyStart = SystemClock.elapsedRealtime()
                        val old = StableDiffusionHelper(context).generateImage(
                            prompt = extras.translatedPrompt(prompt), negativePrompt = extras.translatedPrompt(extras.negativePrompt),
                            steps = if (distilled) 8 else 20, cfg = extras.guidance, seed = 42L, width = size, height = size,
                            scheduler = "dpm"
                        ) ?: error("legacy generateImage() returned null")
                        record.put("legacyGenerateMs", SystemClock.elapsedRealtime() - legacyStart)
                        File(results, model.name.replace(Regex("[^A-Za-z0-9._-]"), "_") + "_legacy.jpg").outputStream()
                            .use { old.compress(Bitmap.CompressFormat.JPEG, 85, it) }
                        started += SystemClock.elapsedRealtime() - legacyStart
                    }
                    record.put("steps", steps).put("scheduler", bundled?.scheduler ?: "dpm")
                        .put("cfg", (bundled?.cfg ?: extras.guidance).toDouble())
                        .put("generateMs", SystemClock.elapsedRealtime() - started)
                        .put("width", bitmap.width).put("height", bitmap.height).put("pssKb", Debug.getPss())
                    val pixels = IntArray(bitmap.width * bitmap.height)
                    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                    val distinct = pixels.asSequence().map { it and 0xffffff }.distinct().take(1000).count()
                    record.put("distinctColors", distinct)
                    File(results, model.name.replace(Regex("[^A-Za-z0-9._-]"), "_") + ".jpg").outputStream()
                        .use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
                    check(distinct >= 256) { "Image is flat ($distinct colors)" }
                    record.put("result", "PASS")
                    Log.i(TAG, "PASS ${model.name} $record")
                } finally {
                    helper.close()
                    SDBackendService.stop(context)
                }
            } catch (t: Throwable) {
                record.put("result", "FAIL").put("error", t.toString())
                failures += "${model.name}: $t"
                Log.e(TAG, "FAIL ${model.name}", t)
            } finally {
                if (pending == null) pending = download(index + 1)
                if (!keep) dir.deleteRecursively()
                records.put(record)
                summary.writeText(JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL)
                    .put("models", records).toString(2))
            }
        }
        assertTrue("Failures:\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    private companion object { const val TAG = "NpuImageSweep" }
}
