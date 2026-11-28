package com.llmhub.llmhub.inference

import android.content.Context
import android.util.Log
import com.llmhub.llmhub.data.DeviceInfo
import java.io.File

/** NPU dispatch libraries for LiteRT-LM engines (EmbeddingGemma 2 and Gemma 4 E2B NPU builds). */
object LiteRtLmNpu {
    private const val TAG = "LiteRtLmNpu"

    /**
     * The folder LiteRT-LM loads its NPU dispatch library from.
     *
     * From LiteRT-LM 0.20 the engine runs on the app's libLiteRt.so (litert 2.4), so it uses the same
     * litert-npu-runtime dispatch libraries as the CompiledModel features, from the APK's native library
     * folder. Each vendor gets a folder with only its own dispatch library, because LiteRT loads the first
     * one it finds.
     */
    fun libraryDir(context: Context): String {
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val soc = DeviceInfo.getDeviceSoc().uppercase()
        val tag = DeviceInfo.getEmbeddingGemma2NpuTag()
        val vendor = when {
            tag?.startsWith("Qualcomm") == true -> "Qualcomm"
            tag?.startsWith("MediaTek") == true || soc.startsWith("MT") -> "MediaTek"
            tag?.startsWith("Google") == true -> "GoogleTensor"
            else -> return nativeDir
        }
        return try {
            val library = "libLiteRtDispatch_$vendor.so"
            val dir = File(context.filesDir, "litertlm_npu/$vendor").apply { mkdirs() }
            val source = File(nativeDir, library)
            File(dir, library).let { if (it.length() != source.length()) source.copyTo(it, overwrite = true) }
            if (vendor == "Qualcomm") linkQnnRuntime(context, dir)
            dir.absolutePath
        } catch (e: Exception) {
            Log.w(TAG, "Failed to stage $vendor NPU libraries: ${e.message}")
            nativeDir
        }
    }

    /**
     * The Qualcomm dispatch library loads QNN from its own folder and needs the QAIRT runtime ([QairtRuntime],
     * assets/qnnlibs_litert). Those 100 MB are kept once in [QairtRuntime.dir], where the CompiledModel
     * features stage them too, and linked into [dir]. The Stable Diffusion pack (assets/qnnlibs, system API
     * 1.5) is a different runtime and must not end up here.
     */
    private fun linkQnnRuntime(context: Context, dir: File) {
        val store = com.llmhub.llmhub.data.QairtRuntime.stage(context)
        for (name in context.assets.list(com.llmhub.llmhub.data.QairtRuntime.ASSET_DIR).orEmpty()) {
            val stored = File(store, name)
            val link = File(dir, name)
            // A link into an older runtime's folder dangles once QairtRuntime removes that folder (exists() is then
            // false but the link still blocks symlink()); replace anything that is not this runtime's file.
            if (link.exists() && link.canonicalPath == stored.canonicalPath) continue
            link.delete()
            runCatching { android.system.Os.symlink(stored.path, link.path) }.onFailure { stored.copyTo(link, overwrite = true) }
        }
    }
}
