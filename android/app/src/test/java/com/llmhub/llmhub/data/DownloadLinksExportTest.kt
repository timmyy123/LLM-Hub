package com.llmhub.llmhub.data

import org.junit.Test
import java.io.File
import java.net.URLClassLoader

/**
 * Writes every model download link the Android app can offer, for every chip, to the JSON file named by the
 * DOWNLOAD_LINKS_OUT environment variable (the release kit's collect_download_links.py sets it). Without the
 * variable the test does nothing.
 *
 * ModelData builds its catalog once, for the device's chip, so each chip gets its own class loader with
 * DeviceInfo.socOverride set before ModelData loads. The chips are the SoC names the app's sources match on.
 */
class DownloadLinksExportTest {

    @Test
    fun exportDownloadLinks() {
        val out = System.getenv("DOWNLOAD_LINKS_OUT") ?: return
        val socs = (chipNamesInSources() + listOf("UNKNOWN", "SM6375")).sorted()
        val classPath = System.getProperty("java.class.path").split(File.pathSeparator)
            .map { File(it).toURI().toURL() }.toTypedArray()

        // url -> (model names, chips)
        val links = sortedMapOf<String, Pair<MutableSet<String>, MutableSet<String>>>()
        fun add(url: String, model: String, soc: String) {
            if (!url.startsWith("http")) return
            val entry = links.getOrPut(url) { mutableSetOf<String>() to mutableSetOf() }
            entry.first += model
            entry.second += soc
        }

        for (soc in socs) {
            URLClassLoader(classPath, ClassLoader.getSystemClassLoader().parent).use { loader ->
                loader.loadClass("com.llmhub.llmhub.data.DeviceInfo").getField("socOverride").set(null, soc)
                val modelData = loader.loadClass("com.llmhub.llmhub.data.ModelData")
                val models = modelData.getMethod("getModels").invoke(modelData.getField("INSTANCE").get(null)) as List<*>
                for (model in models) {
                    val type = model!!.javaClass
                    val name = type.getMethod("getName").invoke(model) as String
                    add(type.getMethod("getUrl").invoke(model) as String, name, soc)
                    (type.getMethod("getAdditionalFiles").invoke(model) as List<*>).forEach { add(it as String, name, soc) }
                }
            }
        }
        // Supertonic voice styles, downloaded one by one from the picker.
        SupertonicModel.voices.keys.forEach { add("${SupertonicModel.BASE_URL}/voice_styles/$it.json", "${SupertonicModel.NAME} voice $it", "all") }

        val json = buildString {
            append("{\"chips\":").append(socs.joinToString(",", "[", "]") { quote(it) }).append(",\"links\":[")
            links.entries.forEachIndexed { i, (url, value) ->
                if (i > 0) append(',')
                val chips = if (value.second.containsAll(socs) || "all" in value.second) listOf("all") else value.second.sorted()
                append("{\"url\":").append(quote(url))
                append(",\"models\":").append(value.first.sorted().joinToString(",", "[", "]") { quote(it) })
                append(",\"chips\":").append(chips.joinToString(",", "[", "]") { quote(it) }).append('}')
            }
            append("]}")
        }
        File(out).writeText(json)
        println("DownloadLinksExportTest: ${links.size} links for ${socs.size} chips -> $out")
    }

    private fun chipNamesInSources(): Set<String> {
        val pattern = Regex("\"((?:SM|QCS|QCM|MT)\\d{4}P?|TENSOR ?G\\d+|Tensor G\\d+)\"")
        return File("src/main/java").walkTopDown().filter { it.extension == "kt" }
            .flatMap { file -> pattern.findAll(file.readText()).map { it.groupValues[1].uppercase() } }
            .toSet()
    }

    private fun quote(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
}
