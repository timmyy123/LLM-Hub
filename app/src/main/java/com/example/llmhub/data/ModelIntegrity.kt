package com.llmhub.llmhub.data

import java.io.File
import java.io.RandomAccessFile
import java.util.zip.ZipFile

/**
 * Lightweight integrity checks for downloaded model files.
 * - For MediaPipe .task files: tolerant validation. Try opening as ZIP; if that fails, check for "PK"; if both fail,
 *   still accept if the file is reasonably large (>=10MB). Many .task bundles are plain FlatBuffers or Zip64 variants.
 * - For GGUF files: verify magic header "GGUF" and minimal size
 */
fun isModelFileValid(file: File, modelFormat: String): Boolean {
    if (!file.exists() || file.length() < 10L * 1024 * 1024) return false // <10MB is suspicious

    return try {
        when (modelFormat.lowercase()) {
            "task" -> isTaskLikelyValid(file)
            "gguf" -> isGgufValid(file)
            else -> true
        }
    } catch (_: Exception) {
        false
    }
}

private fun isTaskLikelyValid(file: File): Boolean {
    // 1) Try as ZIP
    try {
        ZipFile(file).use { return true }
    } catch (_: Exception) { /* ignore */ }

    // 2) Try checking ZIP magic ("PK")
    try {
        RandomAccessFile(file, "r").use { raf ->
            if (raf.length() >= 2) {
                val sig = ByteArray(2)
                raf.readFully(sig)
                if (sig[0] == 'P'.code.toByte() && sig[1] == 'K'.code.toByte()) return true
            }
        }
    } catch (_: Exception) { /* ignore */ }

    // 3) Fallback: accept by size threshold (>=10MB) to avoid false negatives
    return file.length() >= 10L * 1024 * 1024
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
