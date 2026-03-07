package com.llmhub.llmhub.imagegen

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import com.llmhub.llmhub.service.SDBackendService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.OutputStream

/**
 * Image Generator Helper - Wrapper around StableDiffusionHelper
 * Maintains API compatibility while using the new SD backend
 * 
 * Based on local-dream architecture with QNN/MNN backends
 */
class ImageGeneratorHelper(private val context: Context) {
    private val TAG = "ImageGeneratorHelper"
    private val sdHelper = StableDiffusionHelper(context)
    
    suspend fun initialize(modelPath: String? = null, useGpu: Boolean = false): Boolean = withContext(Dispatchers.IO) {
        try {
            // Check if model is available
            val modelInfo = if (modelPath != null) {
                // If path provided, verify it
                val file = File(modelPath)
                if (file.exists()) {
                    ModelInfo(file.name, ModelType.QNN_NPU, file.absolutePath, true) // Type is guessed, backend will detect
                } else {
                    null
                }
            } else {
                sdHelper.getModelInfo()
            }

            if (modelInfo == null) {
                Log.e(TAG, "No SD model found. Please download a model first.")
                return@withContext false
            }
            
            Log.i(TAG, "Found model: ${modelInfo.name} (${modelInfo.type})")
            
            // Start backend service
            SDBackendService.start(context, modelInfo.path, useGpu)
            
            // Poll for service health (up to 30 seconds)
            var attempts = 0
            val maxAttempts = 30
            var isHealthy = false
            
            while (attempts < maxAttempts) {
                if (sdHelper.checkBackendHealth()) {
                    isHealthy = true
                    Log.i(TAG, "Backend service is healthy")
                    break
                }
                Log.d(TAG, "Waiting for backend service... ($attempts/$maxAttempts)")
                kotlinx.coroutines.delay(1000)
                attempts++
            }
            
            if (!isHealthy) {
                Log.e(TAG, "Backend service failed to start after ${maxAttempts}s")
                // Try to stop it since it failed to initialize properly
                SDBackendService.stop(context)
                return@withContext false
            }
            
            Log.i(TAG, "ImageGenerator initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ImageGenerator: ${e.message}", e)
            false
        }
    }
    
    suspend fun generateImage(
        prompt: String,
        iterations: Int = 28,  // Renamed from 'steps' for API compatibility
        seed: Int = 0,
        inputImage: Bitmap? = null,
        denoiseStrength: Float = 0.7f,
        useGpu: Boolean = false
    ): Bitmap? = withContext(Dispatchers.IO) {
        try {
            val img2imgInfo = if (inputImage != null) " [img2img, denoise: $denoiseStrength]" else ""
            Log.i(TAG, "Generating image: prompt='$prompt', steps=$iterations, seed=$seed$img2imgInfo")
            
            val bitmap = sdHelper.generateImage(
                prompt = prompt,
                steps = iterations,
                seed = if (seed == 0) null else seed.toLong(),
                inputImage = inputImage,
                denoiseStrength = denoiseStrength,
                useOpenCL = useGpu,
                onProgress = { current, total ->
                    Log.d(TAG, "Generation progress: $current/$total")
                }
            )
            
            if (bitmap != null) {
                Log.i(TAG, "Image generated successfully: ${bitmap.width}x${bitmap.height}")
            } else {
                Log.e(TAG, "Generated image is null")
            }
            
            bitmap
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate image: ${e.message}", e)
            null
        }
    }

    /**
     * Save bitmap to gallery
     */
    suspend fun saveImageToGallery(bitmap: Bitmap, title: String): Uri? = withContext(Dispatchers.IO) {
        try {
            val filename = "IMG_${System.currentTimeMillis()}.png"
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/LLM-Hub")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val resolver = context.contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                resolver.openOutputStream(uri)?.use { stream ->
                    if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                        throw Exception("Failed to compress bitmap")
                    }
                }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                    resolver.update(uri, contentValues, null, null)
                }
                
                Log.i(TAG, "Image saved to gallery: $uri")
                return@withContext uri
            } else {
                Log.e(TAG, "Failed to create MediaStore entry")
                return@withContext null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save image: ${e.message}", e)
            return@withContext null
        }
    }
    
    fun close() {
        try {
            SDBackendService.stop(context)
            Log.i(TAG, "ImageGenerator closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ImageGenerator: ${e.message}", e)
        }
    }
}
