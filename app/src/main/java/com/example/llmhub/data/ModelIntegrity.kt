package com.llmhub.llmhub.data

import java.io.File
import java.io.RandomAccessFile
import java.util.zip.ZipFile

/**
 * Lightweight integrity checks for downloaded model files.
 * - For MediaPipe .task files: verify ZIP structure opens and central directory is readable
 * - For GGUF files: verify magic header "GGUF" and minimal size
 */
fun isModelFileValid(file: File, modelFormat: String): Boolean {
    if (!file.exists() || file.length() < 10L * 1024 * 1024) return false // <10MB is suspicious

    return try {
        when (modelFormat.lowercase()) {
            "task" -> isZipValid(file)
            "gguf" -> isGgufValid(file)
            else -> true
        }
    } catch (_: Exception) {
        false
    }
}

private fun isZipValid(file: File): Boolean {
    // Opening ZipFile will validate central directory. Iterate first entry to force some reads.
    return try {
        ZipFile(file).use { zip ->
            val entries = zip.entries()
            if (!entries.hasMoreElements()) return false
            // Touch a couple of entries to ensure directory is sane
            var count = 0
            while (entries.hasMoreElements() && count < 3) {
                val e = entries.nextElement()
                if (e.size < 0 && e.compressedSize < 0) return false
                count++
            }
            true
        }
    } catch (_: Exception) {
        false
    }
}

private fun isGgufValid(file: File): Boolean {
    return try {
        RandomAccessFile(file, "r").use { raf ->
            if (raf.length() < 1024) return false
            val magic = ByteArray(4)
            raf.readFully(magic)
            val magicStr = String(magic)
            magicStr == "GGUF"
        }
    } catch (_: Exception) {
        false
    }
}
