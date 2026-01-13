package com.llmhub.llmhub.imagegen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Stable Diffusion Helper based on local-dream architecture
 * Communicates with backend HTTP server for image generation
 * 
 * Based on: https://github.com/xororz/local-dream
 */
class StableDiffusionHelper(private val context: Context) {
    private val TAG = "StableDiffusionHelper"
    private val BASE_URL = "http://127.0.0.1:8081"

    private val MAX_MODEL_SEARCH_DEPTH = 6

    private fun detectModelTypeInDir(dir: File): ModelType? {
        val files = dir.listFiles()?.filter { it.isFile } ?: return null

        val hasQnnModels = files.any {
            it.name.equals("unet.bin", ignoreCase = true) ||
                (it.name.startsWith("unet", ignoreCase = true) && it.name.endsWith(".bin", ignoreCase = true))
        }
        val hasMnnModels = files.any {
            it.name.equals("unet.mnn", ignoreCase = true) ||
                (it.name.contains("unet", ignoreCase = true) && it.name.endsWith(".mnn", ignoreCase = true))
        }

        return when {
            hasQnnModels -> ModelType.QNN_NPU
            hasMnnModels -> ModelType.MNN_CPU
            else -> null
        }
    }

    private fun findModelRoot(foundDir: File, baseDir: File): File {
        var current = foundDir

        // Walk up until we reach the first-level directory under baseDir.
        while (current != baseDir && current.parentFile != null && current.parentFile != baseDir) {
            current = current.parentFile!!
        }

        return current
    }
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS) // 5 minutes for generation
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private var isBackendRunning = false
    
    /**
     * Check if backend service is running
     */
    suspend fun checkBackendHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$BASE_URL/health")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            val healthy = response.isSuccessful
            isBackendRunning = healthy
            
            Log.i(TAG, "Backend health check: ${if (healthy) "OK" else "Failed"}")
            healthy
        } catch (e: IOException) {
            Log.w(TAG, "Backend not reachable: ${e.message}")
            isBackendRunning = false
            false
        }
    }
    
    /**
     * List all available models
     */
    fun listModels(): List<ModelInfo> {
        val modelDir = getModelDirectory()
        if (!modelDir.exists()) return emptyList()
        
        val models = mutableListOf<ModelInfo>()
        val seenPaths = HashSet<String>()
        
        // Recursive search for model files
        fun searchModels(dir: File, depth: Int) {
            if (depth > MAX_MODEL_SEARCH_DEPTH || !dir.exists() || !dir.isDirectory) return
            
            // Detect model files in this directory. If present in a nested folder
            // (e.g. sd_models/anything/unet/unet.bin), treat the top-level folder
            // under sd_models as the model root.
            detectModelTypeInDir(dir)?.let { detectedType ->
                val root = findModelRoot(dir, modelDir)
                val rootPath = root.absolutePath
                if (seenPaths.add(rootPath)) {
                    models.add(
                        ModelInfo(
                            name = root.name,
                            type = detectedType,
                            path = rootPath,
                            isReady = true
                        )
                    )
                }
            }
            
            // Check subdirectories
            dir.listFiles()?.forEach { subDir ->
                if (subDir.isDirectory) {
                    searchModels(subDir, depth + 1)
                }
            }
        }
        
        searchModels(modelDir, 0)
        return models
    }

    /**
     * Get model information
     */
    fun getModelInfo(): ModelInfo? {
        val modelDir = getModelDirectory()
        if (!modelDir.exists()) return null
        
        // Recursive search for model files
        fun findModelFiles(dir: File, depth: Int): Pair<File?, String>? {
            if (depth > MAX_MODEL_SEARCH_DEPTH || !dir.exists() || !dir.isDirectory) return null
            
            detectModelTypeInDir(dir)?.let { detectedType ->
                val root = findModelRoot(dir, modelDir)
                Log.i(TAG, "Found ${detectedType.name} model in: ${root.absolutePath}")
                return Pair(root, detectedType.name)
            }
            
            // Check subdirectories
            dir.listFiles()?.forEach { subDir ->
                if (subDir.isDirectory) {
                    findModelFiles(subDir, depth + 1)?.let { return it }
                }
            }
            return null
        }
        
        val (foundDir, modelType) = findModelFiles(modelDir, 0) ?: return null
        
        // foundDir should not be null here, but add explicit check for type safety
        val modelPath = foundDir?.absolutePath ?: return null
        
        return when (modelType) {
            "QNN" -> ModelInfo(
                name = "Absolute Reality",
                type = ModelType.QNN_NPU,
                path = modelPath,
                isReady = true
            )
            "MNN" -> ModelInfo(
                name = "Absolute Reality",
                type = ModelType.MNN_CPU,
                path = modelPath,
                isReady = true
            )
            else -> null
        }
    }
    
    /**
     * Generate image from text prompt with optional image-to-image support
     * 
     * @param prompt Text description of the image
     * @param negativePrompt Things to avoid in the image
     * @param steps Number of denoising steps (default: 28)
     * @param cfg Classifier-free guidance scale (default: 7.0)
     * @param seed Random seed (null for random)
     * @param width Image width (default: 512)
     * @param height Image height (default: 512)
     * @param inputImage Optional input image for image-to-image generation
     * @param denoiseStrength How much to modify the input image (0.0-1.0, default: 0.7)
     * @param onProgress Callback for generation progress
     */
    suspend fun generateImage(
        prompt: String,
        negativePrompt: String = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        steps: Int = 28,
        cfg: Float = 7f,
        seed: Long? = null,
        width: Int = 512,
        height: Int = 512,
        inputImage: Bitmap? = null,
        denoiseStrength: Float = 0.7f,
        useOpenCL: Boolean = false,
        onProgress: ((Int, Int) -> Unit)? = null
    ): Bitmap? = withContext(Dispatchers.IO) {
        try {
            if (!isBackendRunning) {
                Log.w(TAG, "Backend flag is false, checking health...")
                if (!checkBackendHealth()) {
                    Log.e(TAG, "Backend is not running. Please start the service first.")
                    return@withContext null
                }
            }
            
            // Build JSON request body
            val json = JSONObject().apply {
                put("prompt", prompt)
                put("negative_prompt", negativePrompt)
                put("steps", steps)
                put("cfg", cfg.toDouble())
                if (seed != null) put("seed", seed)
                put("width", width)
                put("height", height)
                put("scheduler", "dpm") // DPM++ 2M Karras scheduler
                put("use_opencl", useOpenCL) // GPU acceleration (OpenCL/Vulkan) vs CPU
                put("stream", false) // Disable streaming to get a single JSON response
                
                // Add img2img support
                if (inputImage != null) {
                    put("denoise_strength", denoiseStrength.toDouble())
                    
                    // Encode input image to base64
                    val resizedImage = Bitmap.createScaledBitmap(inputImage, width, height, true)
                    val outputStream = java.io.ByteArrayOutputStream()
                    resizedImage.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
                    val imageBytes = outputStream.toByteArray()
                    val imageBase64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
                    put("image", imageBase64)
                }
            }
            
            Log.i(TAG, "Generating image: $prompt (${width}x${height}, $steps steps)" + 
                if (inputImage != null) " [img2img, denoise: $denoiseStrength]" else "")
            
            val requestBody = json.toString()
                .toRequestBody("application/json; charset=utf-8".toMediaType())
            
            val request = Request.Builder()
                .url("$BASE_URL/generate")
                .post(requestBody)
                .build()
            
            val response = client.newCall(request).execute()
            
            if (!response.isSuccessful) {
                Log.e(TAG, "Generation failed: ${response.code} - ${response.message}")
                return@withContext null
            }
            
            val responseBody = response.body?.string()
            if (responseBody == null) {
                Log.e(TAG, "Empty response from backend")
                return@withContext null
            }
            
            // Parse response - handle potential SSE stream or raw JSON
            val resultJson = try {
                JSONObject(responseBody)
            } catch (e: Exception) {
                // If direct parsing fails, try to find the last data line from SSE stream
                // Format: data: {...}
                val lines = responseBody.split("\n")
                var lastJson: JSONObject? = null
                
                for (line in lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            val jsonStr = line.substring(6)
                            val jsonObj = JSONObject(jsonStr)
                            // Check if this is the final result (has "image" field)
                            if (jsonObj.has("image")) {
                                lastJson = jsonObj
                            }
                        } catch (ignore: Exception) {}
                    }
                }
                
                lastJson ?: throw e // Rethrow original exception if no valid JSON found
            }
            
            val imageBase64 = resultJson.optString("image", "")
            
            if (imageBase64.isEmpty()) {
                Log.e(TAG, "No image data in response")
                return@withContext null
            }
            
            Log.i(TAG, "Received image data: ${imageBase64.length} chars, prefix: ${imageBase64.take(20)}...")
            
            // Decode base64 to bitmap
            // The backend returns raw RGB/RGBA bytes, not a PNG/JPEG file
            // We need to create a bitmap from the raw pixels
            val imageBytes = Base64.decode(imageBase64, Base64.DEFAULT)
            
            // Create mutable bitmap
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            
            // Copy pixels
            try {
                // Check if we have RGB data (3 bytes per pixel)
                // 512 * 512 * 3 = 786,432 bytes
                if (imageBytes.size == width * height * 3) {
                    Log.i(TAG, "Detected RGB data (3 channels), converting to ARGB")
                    // Convert RGB to ARGB
                    val argbBytes = ByteArray(width * height * 4)
                    for (i in 0 until width * height) {
                        val pixelIndex = i * 3
                        val argbIndex = i * 4
                        argbBytes[argbIndex] = imageBytes[pixelIndex]         // R
                        argbBytes[argbIndex + 1] = imageBytes[pixelIndex + 1] // G
                        argbBytes[argbIndex + 2] = imageBytes[pixelIndex + 2] // B
                        argbBytes[argbIndex + 3] = 255.toByte()               // A (0xFF)
                    }
                    val buffer = java.nio.ByteBuffer.wrap(argbBytes)
                    bitmap.copyPixelsFromBuffer(buffer)
                } else {
                    // Assume RGBA or try direct copy
                    Log.i(TAG, "Assuming RGBA data (4 channels) or other format")
                    val buffer = java.nio.ByteBuffer.wrap(imageBytes)
                    bitmap.copyPixelsFromBuffer(buffer)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy pixels: ${e.message}")
                // Fallback: try standard decoding just in case it WAS an encoded image
                val decodedBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                if (decodedBitmap != null) return@withContext decodedBitmap
                return@withContext null
            }
            
            if (bitmap != null) {
                Log.i(TAG, "Image generated successfully: ${bitmap.width}x${bitmap.height}")
            } else {
                Log.e(TAG, "Failed to decode bitmap from response")
            }
            
            bitmap
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate image: ${e.message}", e)
            null
        }
    }
    
    /**
     * Stop generation (send stop signal to backend)
     */
    suspend fun stopGeneration() = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$BASE_URL/stop")
                .post("".toRequestBody())
                .build()
            
            client.newCall(request).execute()
            Log.i(TAG, "Stop signal sent")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to send stop signal: ${e.message}")
        }
    }
    
    private fun getModelDirectory(): File {
        return File(context.filesDir, "sd_models")
    }
    
    companion object {
        const val DEFAULT_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    }
}

/**
 * Model information
 */
data class ModelInfo(
    val name: String,
    val type: ModelType,
    val path: String,
    val isReady: Boolean
)

/**
 * Model backend type
 */
enum class ModelType {
    QNN_NPU,    // Qualcomm NPU acceleration
    MNN_CPU     // CPU inference
}
