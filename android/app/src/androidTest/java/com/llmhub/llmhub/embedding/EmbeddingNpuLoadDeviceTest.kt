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
        // -e sha256 HEX downloads this chip's NPU build first, for a device that has none.
        // -e size 270M|440M|740M picks which one (default 270M, the smallest download).
        val size = InstrumentationRegistry.getArguments().getString("size") ?: "270M"
        InstrumentationRegistry.getArguments().getString("sha256")?.let { sha256 ->
            val wanted = ModelData.embeddingGemma2Models.first { ModelData.isEmbeddingGemma2NpuModel(it) && it.name.contains(size) }
            com.llmhub.llmhub.imagegen.OnDeviceFixtures.download(File(models, wanted.localFileName()), wanted.url.substringBefore('?'), sha256)
        }
        val model = ModelData.embeddingGemma2Models.firstOrNull {
            ModelData.isEmbeddingGemma2NpuModel(it) && it.name.contains(size) && File(models, it.localFileName()).let { f -> f.isFile && f.length() >= it.sizeBytes }
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

    /**
     * A document is embedded in chunks of very different lengths, and each length range runs a different
     * fixed-size graph on the NPU (128, 256, 512 and 1024 tokens). A short sentence only proves the smallest
     * one. Every length must give a finite, normalized vector, and a question must land closer to the text
     * that answers it than to unrelated text. -e size picks the model as above.
     */
    @Test fun documentChunksOfEveryLengthEmbed() = runBlocking<Unit> {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val models = File(context.filesDir, "models")
        val size = InstrumentationRegistry.getArguments().getString("size") ?: "270M"
        val model = ModelData.embeddingGemma2Models.firstOrNull {
            ModelData.isEmbeddingGemma2NpuModel(it) && it.name.contains(size) && File(models, it.localFileName()).let { f -> f.isFile && f.length() >= it.sizeBytes }
        }
        assumeTrue("No EmbeddingGemma 2 $size NPU build is downloaded for ${Build.SOC_MODEL}", model != null)
        val pricing = "The voice agent costs 18 to 56 cents per call before telephone charges. The Nova speech recognition line is ten cents. " +
            "Voice Live Lite is billed at 2.5 to 7 cents per minute and the Pro tier at 7 to 15 cents per minute, depending on token usage. "
        val baking = "Bread is baked from flour, water, salt and yeast. The dough rests for an hour, is shaped into loaves and goes into a hot oven until the crust is brown. "
        val service = LiteRtLmEmbeddingService(context, model!!)
        try {
            assertTrue("${model.name} failed to load", service.initialize())
            val bad = mutableListOf<String>()
            for (chars in listOf(60, 300, 700, 1500, 3000, 6000, 12000)) {
                val text = pricing.repeat(chars / pricing.length + 1).take(chars)
                val started = SystemClock.elapsedRealtime()
                val v = service.generateEmbedding(text, isQuery = false)
                val ms = SystemClock.elapsedRealtime() - started
                val finite = v != null && v.all { it.isFinite() }
                val norm = v?.let { kotlin.math.sqrt(it.sumOf { x -> (x * x).toDouble() }) } ?: 0.0
                Log.i("EmbeddingNpuLoadTest", "RESULT chunk chars=$chars ms=$ms dim=${v?.size} finite=$finite norm=${"%.4f".format(norm)} head=${v?.take(3)}")
                if (v == null || !finite || norm < 0.9 || norm > 1.1) bad += "$chars chars: dim=${v?.size} finite=$finite norm=$norm"
            }
            fun cos(a: FloatArray, b: FloatArray) = a.indices.sumOf { (a[it] * b[it]).toDouble() }
            val question = service.generateEmbedding("How much does the voice agent cost per call?", isQuery = true)
            val onTopic = service.generateEmbedding(pricing.repeat(4), isQuery = false)
            val offTopic = service.generateEmbedding(baking.repeat(6), isQuery = false)
            assertTrue("Query or document embedding missing", question != null && onTopic != null && offTopic != null)
            val near = cos(question!!, onTopic!!); val far = cos(question, offTopic!!)
            Log.i("EmbeddingNpuLoadTest", "RESULT similarity backend=${service.activeBackendLabel} question-vs-pricing=${"%.4f".format(near)} question-vs-baking=${"%.4f".format(far)}")
            assertTrue("Chunks with a bad vector: $bad", bad.isEmpty())
            assertTrue("The question is not closer to its answer ($near) than to unrelated text ($far)", near > far + 0.1)
        } finally {
            service.close()
        }
    }
}
