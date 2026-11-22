package com.llmhub.llmhub.data

import android.content.Context
import java.io.File

/**
 * The QAIRT (QNN) runtime shipped in assets/qnnlibs_litert, from `com.qualcomm.qti:qnn-runtime` (see
 * app/build.gradle.kts). LiteRT's Qualcomm dispatch, LTX-Video on the NPU and, on Hexagon v85 chips, Stable
 * Diffusion and the upscaler load it from one folder in files/. The folder is named after the runtime version:
 * the libraries are copied only when missing, so a new version needs a new folder, and the old one is removed.
 */
object QairtRuntime {
    const val VERSION = "2.52"
    const val ASSET_DIR = "qnnlibs_litert"
    private const val DIR_NAME = "litert_npu_qnn_252"
    private val OLD_DIR_NAMES = listOf("litert_npu_qnn_250")

    /** The runtime's folder in files/ (created; folders of older runtime versions are deleted). */
    fun dir(context: Context): File {
        for (old in OLD_DIR_NAMES) File(context.filesDir, old).takeIf { it.exists() }?.deleteRecursively()
        return File(context.filesDir, DIR_NAME).apply { mkdirs() }
    }

    /** [dir] with every runtime library copied in (readable and executable). */
    fun stage(context: Context): File {
        val dir = dir(context)
        val names = context.assets.list(ASSET_DIR).orEmpty()
        check(names.isNotEmpty()) { "QAIRT $VERSION libraries missing from assets/$ASSET_DIR" }
        for (name in names) {
            val target = File(dir, name)
            if (target.length() > 0L) continue
            context.assets.open("$ASSET_DIR/$name").use { input -> target.outputStream().use { input.copyTo(it) } }
            target.setReadable(true, false)
            target.setExecutable(true, false)
        }
        return dir
    }
}
