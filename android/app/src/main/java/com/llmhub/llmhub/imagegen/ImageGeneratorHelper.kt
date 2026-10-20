package com.llmhub.llmhub.imagegen

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import com.llmhub.llmhub.R
import com.llmhub.llmhub.service.SDBackendService
import com.llmhub.llmhub.data.ModelData
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
    private var currentModelType: ModelType? = null
    private var checkpointRuntime: SdCppImageRuntime? = null
    private var liteRt: LiteRtSdClient? = null
    private var isSdxlModel = false
    private var isAnimaModel = false
    
    suspend fun initialize(modelPath: String? = null, modelType: ModelType? = null, useGpu: Boolean = false): Boolean = withContext(Dispatchers.IO) {
        try {
            // Check if model is available
            val modelInfo = if (modelPath != null && modelType != null) {
                // If path provided, verify it
                val file = File(modelPath)
                if (file.exists()) {
                    ModelInfo(file.name, modelType, file.absolutePath, true) // Type is guessed, backend will detect
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

            checkpointRuntime?.close()
            checkpointRuntime = null
            liteRt?.close()
            liteRt = null
            if (modelInfo.type == ModelType.LITERT_NPU) {
                SDBackendService.stop(context)
                val client = LiteRtSdClient(context)
                if (!client.load(File(modelInfo.path))) {
                    client.close()
                    return@withContext false
                }
                liteRt = client
                currentModelType = ModelType.LITERT_NPU
                return@withContext true
            }
            if (modelInfo.type == ModelType.SDCPP) {
                SDBackendService.stop(context)
                val runtime = SdCppImageRuntime()
                runtime.load(modelInfo.path, ImageExtrasStore(context).load(), useGpu)
                checkpointRuntime = runtime
                currentModelType = ModelType.SDCPP
                return@withContext true
            }
            currentModelType = modelInfo.type
            
            // Check if SDXL using ModelData centralized detection
            isSdxlModel = ModelData.isSdxlModel(modelInfo.name) || ModelData.isSdxlModel(modelInfo.path)

            // Anima format uses split unet (unet_part1.bin + unet_part2.bin), requires 1024x1024
            isAnimaModel = currentModelType == ModelType.QNN_NPU && !isSdxlModel &&
                    (ModelData.isAnimaModel(modelInfo.name) || ModelData.isAnimaModel(modelInfo.path))

            Log.i(TAG, "Found model: ${modelInfo.name} (${modelInfo.type}), isSdxl=$isSdxlModel, isAnima=$isAnimaModel")
            
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
        width: Int = 256,
        height: Int = 256,
        inputImage: Bitmap? = null,
        denoiseStrength: Float = 0.7f,
        useGpu: Boolean = false
    ): Bitmap? = withContext(Dispatchers.IO) {
        liteRt?.let { return@withContext generateLiteRt(it, prompt, iterations, seed, inputImage) }
        try {
            val extras = ImageExtrasStore(context).load()
            checkpointRuntime?.let { runtime ->
                return@withContext runtime.generate(prompt, extras, width, height, iterations, if (seed == 0) -1L else seed.toLong(), inputImage, denoiseStrength)
            }
            val img2imgInfo = if (inputImage != null) " [img2img, denoise: $denoiseStrength]" else ""
            val targetWidth = if (currentModelType == ModelType.QNN_NPU) {
                if (isSdxlModel || isAnimaModel) 1024 else 512
            } else width
            val targetHeight = if (currentModelType == ModelType.QNN_NPU) {
                if (isSdxlModel || isAnimaModel) 1024 else 512
            } else height

            Log.i(
                TAG,
                "Generating image: prompt='$prompt', steps=$iterations, seed=$seed, size=${targetWidth}x${targetHeight}$img2imgInfo"
            )
            
            val bitmap = sdHelper.generateImage(
                prompt = extras.translatedPrompt(prompt),
                negativePrompt = extras.translatedPrompt(extras.negativePrompt),
                cfg = extras.guidance,
                steps = iterations,
                seed = if (seed == 0) null else seed.toLong(),
                inputImage = inputImage,
                denoiseStrength = denoiseStrength,
                width = targetWidth,
                height = targetHeight,
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
     * Fixed 512x512 text-to-image on a LiteRT NPU build; the screen hides image-to-image for these models.
     * Throws with a readable message when the NPU process crashes, which the screen shows as the error.
     */
    private suspend fun generateLiteRt(client: LiteRtSdClient, prompt: String, steps: Int, seed: Int, inputImage: Bitmap?): Bitmap? {
        if (inputImage != null) Log.w(TAG, "LiteRT NPU models have no image encoder; ignoring the input image")
        val extras = ImageExtrasStore(context).load()
        val output = File(context.cacheDir, "litert_sd_output.png").apply { delete() }
        val ok = client.generate(
            extras.translatedPrompt(prompt), extras.translatedPrompt(extras.negativePrompt),
            steps, extras.guidance, if (seed == 0) System.nanoTime() else seed.toLong(), output
        ) { step, total -> Log.d(TAG, "Generation progress: $step/$total") }
        if (!ok && client.crashed) throw IllegalStateException(context.getString(R.string.image_generator_npu_crashed))
        return if (ok) BitmapFactory.decodeFile(output.path).also { output.delete() } else null
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
            checkpointRuntime?.close()
            checkpointRuntime = null
            liteRt?.close()
            liteRt = null
            currentModelType = null
            isSdxlModel = false
            isAnimaModel = false
            Log.i(TAG, "ImageGenerator closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ImageGenerator: ${e.message}", e)
        }
    }
}
