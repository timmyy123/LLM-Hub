package com.llmhub.llmhub.utils

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

/**
 * Utility for extracting ZIP archives
 * Used for extracting Stable Diffusion model ZIPs
 */
object ZipExtractor {
    private const val TAG = "ZipExtractor"
    private const val BUFFER_SIZE = 8192
    
    /**
     * Extract a ZIP file to a target directory
     * 
     * @param zipFile The ZIP file to extract
     * @param targetDir The directory to extract files into
     * @param onProgress Optional callback for extraction progress (0.0 to 1.0)
     * @return true if extraction succeeded, false otherwise
     */
    suspend fun extractZip(
        zipFile: File,
        targetDir: File,
        onProgress: ((Float) -> Unit)? = null
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!zipFile.exists()) {
                Log.e(TAG, "ZIP file does not exist: ${zipFile.absolutePath}")
                return@withContext false
            }
            
            if (!targetDir.exists()) {
                targetDir.mkdirs()
            }
            
            Log.i(TAG, "Extracting ${zipFile.name} to ${targetDir.absolutePath}")
            
            // First pass: count entries for progress tracking
            var totalEntries = 0
            ZipInputStream(BufferedInputStream(FileInputStream(zipFile))).use { zis ->
                while (zis.nextEntry != null) {
                    totalEntries++
                    zis.closeEntry()
                }
            }
            
            Log.d(TAG, "ZIP contains $totalEntries entries")
            
            // Second pass: extract files
            var extractedEntries = 0
            ZipInputStream(BufferedInputStream(FileInputStream(zipFile))).use { zis ->
                var entry: ZipEntry? = zis.nextEntry
                
                while (entry != null) {
                    val entryName = entry.name
                    val outputFile = File(targetDir, entryName)
                    
                    // Security check: prevent path traversal attacks
                    if (!outputFile.canonicalPath.startsWith(targetDir.canonicalPath)) {
                        Log.w(TAG, "Skipping potentially malicious entry: $entryName")
                        zis.closeEntry()
                        entry = zis.nextEntry
                        continue
                    }
                    
                    if (entry.isDirectory) {
                        // Create directory
                        outputFile.mkdirs()
                        Log.d(TAG, "Created directory: $entryName")
                    } else {
                        // Create parent directories
                        outputFile.parentFile?.mkdirs()
                        
                        // Extract file
                        BufferedOutputStream(FileOutputStream(outputFile)).use { bos ->
                            val buffer = ByteArray(BUFFER_SIZE)
                            var bytesRead: Int
                            while (zis.read(buffer).also { bytesRead = it } != -1) {
                                bos.write(buffer, 0, bytesRead)
                            }
                        }
                        
                        Log.d(TAG, "Extracted: $entryName (${entry.size} bytes)")
                    }
                    
                    zis.closeEntry()
                    extractedEntries++
                    
                    // Report progress
                    if (totalEntries > 0) {
                        val progress = extractedEntries.toFloat() / totalEntries
                        onProgress?.invoke(progress)
                    }
                    
                    entry = zis.nextEntry
                }
            }
            
            Log.i(TAG, "Successfully extracted $extractedEntries entries")
            onProgress?.invoke(1.0f)
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting ZIP: ${e.message}", e)
            false
        }
    }
    
    /**
     * Check if a directory contains extracted model files
     * 
     * @param dir Directory to check
     * @param modelType "qnn" or "mnn"
     * @return true if model files are present
     */
    fun isModelExtracted(dir: File, modelType: String): Boolean {
        if (!dir.exists() || !dir.isDirectory) {
            return false
        }
        
        // Check for key model files based on type
        return when (modelType.lowercase()) {
            "qnn" -> {
                // QNN models have .bin files
                File(dir, "unet.bin").exists() &&
                File(dir, "clip.bin").exists() &&
                File(dir, "vae_decoder.bin").exists()
            }
            "mnn" -> {
                // MNN models have .mnn files
                File(dir, "unet.mnn").exists() &&
                File(dir, "clip.mnn").exists() &&
                File(dir, "vae_decoder.mnn").exists()
            }
            else -> {
                Log.w(TAG, "Unknown model type: $modelType")
                false
            }
        }
    }
}
