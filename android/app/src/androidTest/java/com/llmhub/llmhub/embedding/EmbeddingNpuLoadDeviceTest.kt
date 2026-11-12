package com.llmhub.llmhub.embedding

import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Physical-device check that the EmbeddingGemma 2 NPU build already downloaded in the app loads on
 * the NPU and embeds text. Run it after changing litertlm-android or the litert-npu-runtime
 * libraries: a mismatch between the two only shows up when a model is loaded.
 */
class EmbeddingNpuLoadDeviceTest {
    /**
     * LiteRT-LM and the CompiledModel runtime load different builds of the vendor dispatch library. With an
     * NPU embedding model and Parakeet on the NPU both loaded in one process, each must still work.
     */
    @Test fun npuEmbeddingAndParakeetNpuWorkTogether() = runBlocking<Unit> {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val models = File(context.filesDir, "models")
        val embedding = ModelData.embeddingGemma2Models.firstOrNull {
            ModelData.isEmbeddingGemma2NpuModel(it) && File(models, it.localFileName()).length() >= it.sizeBytes
        }
        val parakeet = ModelData.models.firstOrNull {
            com.llmhub.llmhub.data.AsrModels.isParakeetNpu(it) && File(models, it.localFileName()).length() >= it.sizeBytes
        }
        assumeTrue("Needs a downloaded NPU embedding model and Parakeet NPU build", embedding != null && parakeet != null)
        val embedder = LiteRtLmEmbeddingService(context, embedding!!)
        val recognizer = com.llmhub.llmhub.inference.SpeechRecognitionService(context)
        try {
            assertTrue("Embedding model failed to load", embedder.initialize())
            assertTrue("Parakeet NPU failed to load next to it", recognizer.loadModel(parakeet!!))
            val first = embedder.generateEmbedding("Bread is baked from flour, water and yeast.", isQuery = false)
            val heard = recognizer.transcribe(ByteArray(16000 * 2 * 2)) // two seconds of silence
            val second = embedder.generateEmbedding("The stock market fell sharply on Monday.", isQuery = false)
            Log.i("EmbeddingNpuLoadTest", "RESULT together: backend=${embedder.activeBackendLabel} dims=${first?.size},${second?.size} silence='$heard'")
            assertEquals(768, first?.size)
            assertEquals(768, second?.size)
            assertEquals("", heard)
            assertTrue(embedder.activeBackendLabel?.startsWith("NPU") == true)
        } finally {
            recognizer.release()
            embedder.close()
        }
    }

    @Test fun downloadedNpuModelEmbeds() = runBlocking<Unit> {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val models = File(context.filesDir, "models").apply { mkdirs() }
        // -e sha256 HEX downloads this chip's 270M NPU build first, for a device that has none.
        InstrumentationRegistry.getArguments().getString("sha256")?.let { sha256 ->
            val small = ModelData.embeddingGemma2Models.first { ModelData.isEmbeddingGemma2NpuModel(it) && it.name.contains("270M") }
            com.llmhub.llmhub.imagegen.OnDeviceFixtures.download(File(models, small.localFileName()), small.url.substringBefore('?'), sha256)
        }
        val model = ModelData.embeddingGemma2Models.firstOrNull {
            ModelData.isEmbeddingGemma2NpuModel(it) && File(models, it.localFileName()).let { f -> f.isFile && f.length() >= it.sizeBytes }
        }
        assumeTrue("No EmbeddingGemma 2 NPU build is downloaded for ${Build.SOC_MODEL}", model != null)
        val report = JSONObject().put("model", model!!.name).put("soc", Build.SOC_MODEL)
        val service = LiteRtLmEmbeddingService(context, model)
        try {
            val started = SystemClock.elapsedRealtime()
            val ready = service.initialize()
            report.put("initialized", ready).put("backend", service.activeBackendLabel).put("initMs", SystemClock.elapsedRealtime() - started)
            if (ready) {
                val vector = service.generateEmbedding("Bread is baked from flour, water and yeast in a hot oven.", isQuery = false)
                report.put("dimension", vector?.size ?: 0)
            }
            Log.i("EmbeddingNpuLoadTest", "RESULT $report")
            assertTrue("${model.name} failed to load on the NPU: $report", ready)
            assertTrue("${model.name} ran on ${service.activeBackendLabel}", service.activeBackendLabel?.startsWith("NPU") == true)
            assertTrue("No embedding came back", report.optInt("dimension") > 0)
        } finally {
            Log.i("EmbeddingNpuLoadTest", "REPORT $report")
            service.close()
        }
    }
}
