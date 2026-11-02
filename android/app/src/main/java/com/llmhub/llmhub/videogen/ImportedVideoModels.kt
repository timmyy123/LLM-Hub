package com.llmhub.llmhub.videogen

import android.content.Context
import android.net.Uri
import com.google.gson.Gson
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.bundleDirectory
import com.llmhub.llmhub.data.hasCompleteDownloadedBundle
import java.io.File

/**
 * LTX-Video transformer GGUFs the user imported. They are saved apart from imported chat models ("imported_models"),
 * so they only appear in Download Models' video tab and in Video Generator. Each one downloads the catalog's T5 text
 * encoder and VAE (its url and additional file), then the transformer is copied in next to them.
 */
object ImportedVideoModels {
    const val SOURCE = "Imported video"
    private const val PREFS = "model_prefs"
    private const val MODELS_KEY = "imported_video_models"
    private const val SOURCES_KEY = "imported_video_transformers"
    private val gson = Gson()

    fun load(context: Context): List<LLMModel> = runCatching {
        val json = prefs(context).getString(MODELS_KEY, null) ?: return emptyList()
        gson.fromJson(json, Array<LLMModel>::class.java).toList()
    }.getOrDefault(emptyList())

    fun add(context: Context, model: LLMModel, transformer: Uri) {
        save(context, load(context).filterNot { it.name == model.name } + model)
        prefs(context).edit().putString(sourceKey(model), transformer.toString()).apply()
    }

    fun remove(context: Context, model: LLMModel) {
        save(context, load(context).filterNot { it.name == model.name })
        prefs(context).edit().remove(sourceKey(model)).apply()
        model.bundleDirectory(context).deleteRecursively()
    }

    fun transformerFile(context: Context, model: LLMModel) = File(model.bundleDirectory(context), "ltxv-transformer.gguf")

    /** The encoder and VAE are downloaded and the transformer is in place. */
    fun isComplete(context: Context, model: LLMModel) =
        model.hasCompleteDownloadedBundle(context) && transformerFile(context, model).length() > 0

    /** Copies the imported transformer into the model's folder (the runtime tells it from the encoder by "t5"). */
    fun copyTransformer(context: Context, model: LLMModel) {
        val source = Uri.parse(prefs(context).getString(sourceKey(model), null) ?: error("No transformer for ${model.name}"))
        val target = transformerFile(context, model)
        val partial = File(target.path + ".part")
        partial.outputStream().use { output ->
            context.contentResolver.openInputStream(source)!!.use { it.copyTo(output, 1 shl 20) }
        }
        check(partial.renameTo(target)) { "Could not place the transformer for ${model.name}" }
    }

    private fun save(context: Context, models: List<LLMModel>) =
        prefs(context).edit().putString(MODELS_KEY, gson.toJson(models)).apply()

    private fun sourceKey(model: LLMModel) = "$SOURCES_KEY:${model.name}"

    private fun prefs(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
}
