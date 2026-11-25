package com.llmhub.llmhub.data

import android.content.Context
import java.io.File

/** Which network an Object Eraser model file holds; each has its own input layout. */
enum class InpaintKind { MIGAN, LAMA }

/**
 * Object Eraser (image inpainting) models: catalog names and where their files live.
 * Each model is one .tflite file in files/inpaint_models/<safe name>/model.tflite. The download
 * goes to files/models/<file name> first and is moved there once it is complete.
 */
object InpaintModels {
    const val CATEGORY = "image_inpainting"
    const val DIR = "inpaint_models"
    const val FILE_NAME = "model.tflite"

    const val MIGAN_NAME = "MI-GAN 512 (Quick)"
    const val LAMA_NAME = "Big LaMa 512 (HD)"

    fun safeName(name: String): String = name.replace(Regex("[^a-zA-Z0-9_\\-]"), "_")

    fun modelDir(context: Context, model: LLMModel): File =
        File(File(context.filesDir, DIR), safeName(model.name))

    fun modelFile(context: Context, model: LLMModel): File = File(modelDir(context, model), FILE_NAME)

    /** The partial or finished download before it is moved into [modelDir]. */
    fun downloadFile(context: Context, model: LLMModel): File =
        File(File(context.filesDir, "models"), model.localFileName())

    fun kindOf(model: LLMModel): InpaintKind =
        if (model.name == LAMA_NAME) InpaintKind.LAMA else InpaintKind.MIGAN

    fun isComplete(file: File, model: LLMModel): Boolean =
        file.isFile && (if (model.sizeBytes > 0) file.length() >= model.sizeBytes else file.length() > 0)

    /** Downloaded models by kind, in catalog order. Does file I/O. */
    fun installed(context: Context): Map<InpaintKind, File> =
        ModelData.inpaintingModels.mapNotNull { model ->
            modelFile(context, model).takeIf { isComplete(it, model) }?.let { kindOf(model) to it }
        }.toMap()
}
