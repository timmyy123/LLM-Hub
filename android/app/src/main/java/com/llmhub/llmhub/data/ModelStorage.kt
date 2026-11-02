package com.llmhub.llmhub.data

import android.content.Context
import android.os.Environment
import android.system.Os
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.file.Files
import kotlin.coroutines.coroutineContext

/**
 * Where downloaded models live: the app's internal storage, or its folder on a removable SD card
 * (Android/data/<package>/files/models, which needs no storage permission and is removed with the app).
 * On the SD card, files/models is a link to that folder, so every part of the app that opens files/models
 * keeps working unchanged.
 */
object ModelStorage {
    private fun internal(context: Context) = File(context.filesDir, "models")

    /** The SD card's app folder, when a removable card is mounted. */
    fun sdCardRoot(context: Context): File? = context.getExternalFilesDirs(null).drop(1).firstOrNull { dir ->
        dir != null && runCatching {
            Environment.isExternalStorageRemovable(dir) && Environment.getExternalStorageState(dir) == Environment.MEDIA_MOUNTED
        }.getOrDefault(false)
    }

    fun isOnSdCard(context: Context): Boolean = Files.isSymbolicLink(internal(context).toPath())

    /** Free bytes where models are stored now ([sdCard] false) or would be on the SD card ([sdCard] true). */
    fun freeBytes(context: Context, sdCard: Boolean): Long =
        (if (sdCard) sdCardRoot(context) else context.filesDir)?.usableSpace ?: 0L

    fun usedBytes(context: Context): Long = internal(context).walkTopDown().filter { it.isFile }.sumOf { it.length() }

    /**
     * Moves every model to the SD card ([toSdCard] true) or back to internal storage. Files are copied and checked
     * before the originals are removed, so an interrupted move leaves the models where they were.
     */
    suspend fun move(context: Context, toSdCard: Boolean, onProgress: (Float) -> Unit): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            if (toSdCard == isOnSdCard(context)) return@runCatching
            val link = internal(context)
            val card = File(sdCardRoot(context) ?: error("No SD card"), "models")
            val source = if (toSdCard) link.canonicalFile.also { it.mkdirs() } else card
            val target = if (toSdCard) card else File(context.filesDir, "models.moving")
            val total = usedBytes(context)
            check(freeBytes(context, toSdCard) > total) { "Not enough free space" }
            target.mkdirs()
            var copied = 0L
            source.walkTopDown().filter { it.isFile }.forEach { file ->
                coroutineContext.ensureActive()
                val destination = File(target, file.relativeTo(source).path).apply { parentFile?.mkdirs() }
                if (!(destination.isFile && destination.length() == file.length())) {
                    file.inputStream().use { input -> destination.outputStream().use { input.copyTo(it, 1 shl 20) } }
                }
                check(destination.length() == file.length()) { "Copy of ${file.name} is incomplete" }
                copied += file.length()
                onProgress(if (total > 0) copied.toFloat() / total else 1f)
            }
            if (toSdCard) {
                link.deleteRecursively()
                Os.symlink(card.absolutePath, link.absolutePath)
            } else {
                link.delete() // the link only
                check(target.renameTo(link)) { "Could not finish the move" }
                card.deleteRecursively()
            }
        }
    }
}
