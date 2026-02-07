package com.llmhub.llmhub.data

import java.io.File
import java.io.RandomAccessFile
import java.util.zip.ZipFile

/**
 * Lightweight integrity checks for downloaded model files.
 * For now, we just verify the file exists and has a reasonable size.
 * Future versions could add more sophisticated validation like checksum verification.
 */
fun isModelFileValid(file: File, modelFormat: String): Boolean {
    android.util.Log.d("ModelIntegrity", "Validating file: ${file.absolutePath}, format: $modelFormat, exists: ${file.exists()}, size: ${file.length()}")
    
    if (!file.exists()) {
        android.util.Log.d("ModelIntegrity", "File does not exist")
        return false
    }
    
    // Check if file is at least 1MB (very basic check)
    if (file.length() < 1L * 1024 * 1024) {
        android.util.Log.d("ModelIntegrity", "File too small: ${file.length()} bytes")
        return false
    }

    // Perform format-specific validation
    val valid = when (modelFormat) {
        "task", "litertlm" -> isTaskLikelyValid(file)
        "gguf", "bin" -> isGgufValid(file) // 'bin' might be raw but often GGUF in this context? Or maybe just size check.
        "onnx" -> true // ONNX validation is complex (protobuf), we rely on size check in caller or basic existence
        else -> true // Fallback for unknown formats
    }
    
    android.util.Log.d("ModelIntegrity", "File validation result: $valid for ${file.name}")
    return valid
}

private fun isTaskLikelyValid(file: File): Boolean {
    android.util.Log.d("ModelIntegrity", "Validating .task file: ${file.absolutePath}")
    
    // 1) Try as ZIP
    try {
        ZipFile(file).use { 
            android.util.Log.d("ModelIntegrity", "File is valid ZIP")
            return true 
        }
    } catch (e: Exception) { 
        android.util.Log.d("ModelIntegrity", "ZIP validation failed: ${e.message}")
    }

    // 2) Try checking ZIP magic ("PK")
    try {
        RandomAccessFile(file, "r").use { raf ->
            if (raf.length() >= 2) {
                val sig = ByteArray(2)
                raf.readFully(sig)
                if (sig[0] == 'P'.code.toByte() && sig[1] == 'K'.code.toByte()) {
                    android.util.Log.d("ModelIntegrity", "File has ZIP magic signature")
                    return true
                }
                android.util.Log.d("ModelIntegrity", "File signature: ${sig[0]}, ${sig[1]} (not ZIP)")
            }
        }
    } catch (e: Exception) { 
        android.util.Log.d("ModelIntegrity", "Magic signature check failed: ${e.message}")
    }

    // 3) Fallback: accept by size threshold (>=10MB) to avoid false negatives
    val result = file.length() >= 10L * 1024 * 1024
    android.util.Log.d("ModelIntegrity", "Size fallback validation: $result (${file.length()} bytes)")
    return result
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
