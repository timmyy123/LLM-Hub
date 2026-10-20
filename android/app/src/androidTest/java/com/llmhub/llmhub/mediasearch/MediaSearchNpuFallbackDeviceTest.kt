package com.llmhub.llmhub.mediasearch

import android.app.Application
import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.embedding.EmbeddingNpuProbe
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device check that Video Moment Finder opens with the chip-specific EmbeddingGemma 2 NPU build
 * downloaded: the NPU build loads where the NPU can run it, and elsewhere the app falls back to the standard build
 * instead of crashing. Stage both 740M files in files/embedgemma_fallback-test-fixtures, or pass
 * `-e downloadFixtures 1` to fetch them on the phone (no Device Farm host, e.g. BrowserStack); results go to
 * files/embedgemma_fallback-test-results and logcat.
 */
class MediaSearchNpuFallbackDeviceTest {
    @Test fun videoMomentFinderLoads() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = File(context.filesDir, "embedgemma_fallback-test-fixtures").apply { mkdirs() }
        val results = File(context.filesDir, "embedgemma_fallback-test-results").apply { mkdirs() }
        val npu = ModelData.embeddingGemma2Models.firstOrNull { ModelData.isEmbeddingGemma2NpuModel(it) && it.name.startsWith("EmbeddingGemma 2 740M") }
        assertNotNull("No EmbeddingGemma 2 NPU build is offered for ${Build.SOC_MODEL}", npu)
        val standard = ModelData.embeddingGemma2Models.first { it.name == "EmbeddingGemma 2 740M" }
        val models = File(context.filesDir, "models").apply { mkdirs() }
        val download = InstrumentationRegistry.getArguments().getString("downloadFixtures") == "1"
        val staged = listOf(npu!!, standard).map { model ->
            val source = File(fixtures, model.url.substringAfterLast('/').substringBefore('?'))
            if (download) {
                val sha256 = SHA256[source.name]
                assertNotNull("No pinned SHA-256 for ${source.name}", sha256)
                OnDeviceFixtures.download(source, model.url.substringBefore('?'), sha256!!)
            }
            assertTrue("Missing fixture ${source.name}", source.isFile)
            File(models, model.localFileName()).also { if (it.length() != source.length()) source.copyTo(it, overwrite = true) }
        }
        // Open the feature as on first launch: no saved model choice and no cached NPU probe.
        for (name in listOf("video_moment_prefs", "embedding_npu_probe")) {
            context.getSharedPreferences(name, Context.MODE_PRIVATE).edit().clear().commit()
        }

        val report = JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL).put("android", Build.VERSION.RELEASE)
        val started = SystemClock.elapsedRealtime()
        val viewModel = withContext(Dispatchers.Main) { VideoMomentViewModel(context.applicationContext as Application) }
        try {
            val (loaded, failed) = withTimeout(10 * 60_000L) {
                combine(viewModel.isModelLoaded, viewModel.modelError) { loaded, failed -> loaded to failed }.first { it.first || it.second }
            }
            val npuWorks = !EmbeddingNpuProbe.knownUnsupported(context, staged[0])
            report.put("loaded", loaded).put("failed", failed).put("npuWorks", npuWorks)
                .put("selected", viewModel.selectedModel.value?.name).put("loadMs", SystemClock.elapsedRealtime() - started)
            File(results, "results.json").writeText(report.toString(2))
            Log.i("MediaSearchNpuFallback", report.toString())
            assertTrue("Video Moment Finder failed to load a model", loaded)
            assertEquals(if (npuWorks) npu.name else standard.name, viewModel.selectedModel.value?.name)
        } finally {
            withContext(Dispatchers.Main) { viewModel.unloadModel() }
            withTimeout(60_000L) { viewModel.isModelLoaded.first { !it } }
            staged.forEach { it.delete() }
        }
    }

    private companion object {
        /** SHA-256 of the 740M files at the commit ModelData pins, as in tools/imagegen_devicefarm/prepare_fixtures.py. */
        val SHA256 = mapOf(
            "embeddinggemma-2-740m.litertlm" to "e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c",
            "embeddinggemma-2-740m_Google_Tensor_G5.litertlm" to "5b53d76c2c7c66d2dc6230ff158ea0033ac5939ac7cfcf827c3a22cdfc742a5e",
            "embeddinggemma-2-740m_Google_Tensor_G6.litertlm" to "ed0e4f413e705946eda544078d8bf5acf10fe5aee2946451c81919a1f59665e7",
            "embeddinggemma-2-740m_MediaTek_MT6991.litertlm" to "99e6a35b824b4f7ad4f7c7f20c59ba380aa90e0b4442aadfee3674230c6c7cca",
            "embeddinggemma-2-740m_MediaTek_MT6993.litertlm" to "ed2a2554bfb6d7abcafd0e9e36ee2f8d7de82888e7ed0fc52dd144e65b171742",
            "embeddinggemma-2-740m_Qualcomm_SM8550.litertlm" to "802686a865fd7350f2ce4b6fa17ce02a411f56b5b508ec2bd8cb7310b2db82e6",
            "embeddinggemma-2-740m_Qualcomm_SM8650.litertlm" to "3b8a18c6d043b8132c4eda45611296e1ae455ada82cda8b591ec0ca141e041c5",
            "embeddinggemma-2-740m_Qualcomm_SM8750.litertlm" to "ac1973c0195dc6e28a621651f2f5e04d35fdc7a7b11cdbbe94f29f5dccb57996",
            "embeddinggemma-2-740m_Qualcomm_SM8850.litertlm" to "19f6e8db6aa74b116d1c558c862ee131f60ce093576d3c8dd1898f656ca8080b",
        )
    }
}
