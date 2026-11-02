package com.llmhub.llmhub.data

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import java.io.File

/**
 * Imported models are read where the user keeps them (an SD card, Downloads) instead of being copied into the app's
 * internal storage first. The document stays open for the life of the process, and engines that take a path open it
 * through /proc/self/fd, which works for any document the app can read.
 */
object ImportedModelFiles {
    private val open = mutableMapOf<String, ParcelFileDescriptor>()

    /** An earlier version's internal copy when there is one, else the document itself; null when it cannot be opened. */
    @Synchronized
    fun pathFor(context: Context, model: LLMModel): File? {
        File(File(context.filesDir, "models"), model.localFileName()).takeIf { it.isFile && it.length() > 0 }?.let { return it }
        val descriptor = open[model.url] ?: runCatching {
            context.contentResolver.openFileDescriptor(Uri.parse(model.url), "r")
        }.getOrNull()?.also { open[model.url] = it } ?: return null
        return File("/proc/self/fd/${descriptor.fd}")
    }
}
