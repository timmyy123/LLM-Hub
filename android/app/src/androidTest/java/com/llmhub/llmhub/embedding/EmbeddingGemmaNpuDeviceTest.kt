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
import org.junit.Test
import java.io.File
import kotlin.math.sqrt

/**
 * Opt-in physical-device check of Google's SoC-specific EmbeddingGemma 2 NPU builds, using the app's model entries
 * for this device's SoC. Stage the .litertlm files in files/embedgemma-test-fixtures; results go to
 * files/embedgemma-test-results.
 */
class EmbeddingGemmaNpuDeviceTest {
    @Test fun npuEmbeddings() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = File(context.filesDir, "embedgemma-test-fixtures")
        val results = File(context.filesDir, "embedgemma-test-results").apply { mkdirs() }
        val report = JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL).put("android", Build.VERSION.RELEASE)
        fun save() = File(results, "results.json").writeText(report.toString(2))
        val npuModels = ModelData.embeddingGemma2Models.filter { ModelData.isEmbeddingGemma2NpuModel(it) }
        assertTrue("No EmbeddingGemma 2 NPU build is offered for ${Build.SOC_MODEL}", npuModels.isNotEmpty())

        for (size in listOf("270M", "740M")) {
            val model = npuModels.firstOrNull { it.name.startsWith("EmbeddingGemma 2 $size (NPU") } ?: continue
            val source = File(fixtures, model.url.substringAfterLast('/').substringBefore('?'))
            assertTrue("Missing fixture ${source.name}", source.isFile)
            val target = File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName())
            if (target.length() != source.length()) source.copyTo(target, overwrite = true)

            val service = LiteRtLmEmbeddingService(context, model)
            val entry = JSONObject().put("name", model.name)
            report.put(size, entry)
            try {
                var started = SystemClock.elapsedRealtime()
                val ready = service.initialize()
                entry.put("initialized", ready).put("backend", service.activeBackendLabel)
                    .put("initMs", SystemClock.elapsedRealtime() - started)
                save()
                assertTrue("${model.name} failed to initialize", ready)
                assertTrue("${model.name} fell back to ${service.activeBackendLabel}", service.activeBackendLabel?.startsWith("NPU") == true)

                val query = service.generateEmbedding("How do I bake bread at home?", isQuery = true)!!
                val related = service.generateEmbedding("Bread is baked from flour, water and yeast in a hot oven.", isQuery = false)!!
                val unrelated = service.generateEmbedding("The stock market fell sharply on Monday.", isQuery = false)!!
                val relatedScore = cosine(query, related)
                val unrelatedScore = cosine(query, unrelated)
                started = SystemClock.elapsedRealtime()
                repeat(10) { service.generateEmbedding("Document number $it about cooking and gardening.", isQuery = false) }
                entry.put("dimension", query.size).put("relatedCosine", relatedScore).put("unrelatedCosine", unrelatedScore)
                    .put("msPerEmbedding", (SystemClock.elapsedRealtime() - started) / 10.0)
                save()
                Log.i("EmbeddingGemmaNpuTest", "$size: $entry")
                assertTrue("${model.name} must rank the related document higher", relatedScore > unrelatedScore)
            } finally {
                service.close()
                target.delete()
            }
        }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrt(na) * sqrt(nb))
    }
}
