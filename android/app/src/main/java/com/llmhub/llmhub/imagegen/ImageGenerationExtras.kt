package com.llmhub.llmhub.imagegen

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

// Prompt resource files use the same portable format on Android and iOS.
data class ImageExtra(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val path: String = "",
    val kind: String,
    val weight: Float = 1f,
    val enabled: Boolean = true,
    val tags: List<String> = emptyList(),
    val mappings: Map<String, String> = emptyMap()
) {
    fun json() = JSONObject().apply {
        put("id", id); put("name", name); put("path", path); put("kind", kind)
        put("weight", weight); put("enabled", enabled)
        put("tags", JSONArray(tags)); put("mappings", JSONObject(mappings))
    }
}

data class ImageGenerationOptions(
    val negativePrompt: String = StableDiffusionHelper.DEFAULT_NEGATIVE_PROMPT,
    val guidance: Float = 7f,
    val clipSkip: Int = 1,
    val assets: List<ImageExtra> = emptyList()
) {
    fun json() = JSONObject().apply {
        put("negativePrompt", translatedPrompt(negativePrompt)); put("guidance", guidance); put("clipSkip", clipSkip)
        put("assets", JSONArray().apply { assets.forEach { put(it.json()) } })
    }
    fun translatedPrompt(prompt: String): String {
        val mappings = assets.filter { it.enabled && it.kind == "translations" }.flatMap { it.mappings.entries }.associate { it.toPair() }
        if (mappings.isEmpty()) return prompt
        val keys = mappings.keys.filter { it.isNotEmpty() }.sortedByDescending { it.length }
        if (keys.isEmpty()) return prompt
        val regex = Regex("(?<![\\p{L}\\p{N}_])(?:" + keys.joinToString("|") { Regex.escape(it) } + ")(?![\\p{L}\\p{N}_])")
        return regex.replace(prompt) { mappings[it.value] ?: it.value }
    }
    fun promptItems(): List<Pair<String, String>> = assets.filter { it.enabled }.flatMap {
        when (it.kind) {
            "tags" -> it.tags.map { tag -> tag to tag }
            "definitions" -> it.mappings.entries.sortedBy { entry -> entry.key }.map { entry -> entry.key to entry.value }
            "embedding" -> listOf(it.name to it.name)
            else -> emptyList()
        }
    }
}

class ImageExtrasStore(context: Context) {
    companion object {
        /** Import types offered in the model import dialog, in menu order. */
        val kinds = listOf("checkpoint", "lora", "clipL", "clipG", "textEncoder", "vae", "embedding", "tags", "definitions", "translations")
        /** Imports that work on their own (Import External Model): only a full checkpoint. */
        val modelKinds = listOf("checkpoint")
        /** Add-ons, imported in Image Generator's Image modifiers panel. */
        val addOnKinds = kinds - "checkpoint"
        /** Add-ons that need an imported checkpoint selected; prompt lists work with every model. */
        val checkpointOnlyKinds = setOf("lora", "clipL", "clipG", "textEncoder", "vae", "embedding")
        private val encoderKinds = setOf("clipL", "clipG", "textEncoder", "vae")
        fun extensions(kind: String): Set<String> = when (kind) {
            "tags", "definitions", "translations" -> setOf("txt", "json")
            // stable-diffusion.cpp detects the format from the file contents (ModelLoader).
            else -> setOf("safetensors", "gguf", "ckpt", "pt", "pth", "bin")
        }
    }
    private val app = context.applicationContext
    private val prefs = app.getSharedPreferences("image_generation_extras_v1", Context.MODE_PRIVATE)
    fun load(): ImageGenerationOptions = runCatching {
        val objectValue = JSONObject(prefs.getString("options", "{}") ?: "{}")
        val values = objectValue.optJSONArray("assets") ?: JSONArray()
        val assets = (0 until values.length()).map { index ->
            val item = values.getJSONObject(index)
            val tags = item.optJSONArray("tags") ?: JSONArray()
            val mappings = item.optJSONObject("mappings") ?: JSONObject()
            ImageExtra(item.getString("id"), item.getString("name"), item.optString("path"), item.getString("kind"),
                item.optDouble("weight", 1.0).toFloat().coerceIn(-2f, 2f), item.optBoolean("enabled", true),
                (0 until tags.length()).map { tags.getString(it) }, mappings.keys().asSequence().associateWith { mappings.getString(it) })
        }
        ImageGenerationOptions(objectValue.optString("negativePrompt", StableDiffusionHelper.DEFAULT_NEGATIVE_PROMPT),
            objectValue.optDouble("guidance", 7.0).toFloat().coerceIn(0f,20f), objectValue.optInt("clipSkip", 1).coerceIn(1,12), assets)
    }.getOrDefault(ImageGenerationOptions())
    fun save(options: ImageGenerationOptions) {
        // Persist the source prompt, rather than applying translation mappings twice.
        val json = options.json().put("negativePrompt", options.negativePrompt)
        prefs.edit().putString("options", json.toString()).apply()
    }
    fun remove(asset: ImageExtra) {
        val file = File(asset.path)
        val directory = File(app.filesDir, "image_extras")
        if (file.parentFile?.canonicalFile == directory.canonicalFile && file.name.startsWith(asset.id + ".")) file.delete()
    }
    fun checkpoints(): List<ModelInfo> = load().assets.filter { it.kind == "checkpoint" && File(it.path).isFile }
        .map { ModelInfo(it.name, ModelType.SDCPP, it.path, true) }

    /** Records an imported file; a new encoder or VAE replaces the active one of the same slot. */
    fun add(asset: ImageExtra) {
        val previous = load()
        val assets = if (asset.kind in encoderKinds) previous.assets.map { if (it.kind == asset.kind) it.copy(enabled = false) else it } else previous.assets
        if (asset.kind == "checkpoint") app.getSharedPreferences("image_generator_prefs", 0).edit().putString("last_model_path", asset.path).apply()
        save(previous.copy(assets = assets + asset))
    }

    fun importFile(uri: Uri, kind: String, displayNameOverride: String? = null): ImageExtra {
        val displayName = app.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        } ?: "model"
        val name = displayNameOverride?.trim()?.takeIf { it.isNotEmpty() } ?: displayName.substringBeforeLast('.')
        val id = UUID.randomUUID().toString()
        if (kind in listOf("tags", "definitions", "translations")) {
            val bytes = app.contentResolver.openInputStream(uri)?.use { input ->
                val output = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(8192)
                while (output.size() <= 2_000_000) {
                    val count = input.read(buffer, 0, minOf(buffer.size, 2_000_001 - output.size()))
                    if (count < 0) break
                    output.write(buffer, 0, count)
                }
                output.toByteArray()
            } ?: error("File cannot be read")
            require(bytes.size <= 2_000_000)
            val text = bytes.toString(Charsets.UTF_8)
            if (kind == "tags") {
                val tags = if (text.trimStart().startsWith("[")) {
                    val array = JSONArray(text); (0 until array.length()).map { array.getString(it) }
                } else text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.toList()
                require(tags.isNotEmpty())
                return ImageExtra(id, name, kind = kind, tags = tags)
            }
            val mappings = if (text.trimStart().startsWith("{")) {
                val values = JSONObject(text); values.keys().asSequence().associateWith { values.getString(it) }
            } else text.lineSequence().filter { it.isNotBlank() }.associate {
                require(it.contains('=')); it.substringBefore('=').trim() to it.substringAfter('=').trim()
            }
            require(mappings.isNotEmpty() && mappings.keys.none { it.isEmpty() })
            return ImageExtra(id, name, kind = kind, mappings = mappings)
        }
        val extension = displayName.substringAfterLast('.', "").lowercase()
        require(extension in extensions(kind))
        val directory = File(app.filesDir, "image_extras").apply { mkdirs() }
        val destination = File(directory, "$id.$extension")
        val temporary = File(directory, "$id.part")
        try {
            app.contentResolver.openInputStream(uri)?.use { input -> temporary.outputStream().use { input.copyTo(it) } } ?: error("File cannot be read")
            require(temporary.length() > 0 && temporary.renameTo(destination))
            return ImageExtra(id, name, destination.path, kind)
        } finally { temporary.delete() }
    }
}
