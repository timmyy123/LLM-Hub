package com.example.llmhub.inference

import android.content.Context
import com.example.llmhub.data.LLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import androidx.annotation.Keep
import java.io.File

/**
 * Interface for a service that can run model inference.
 */
interface InferenceService {
    suspend fun generateResponse(prompt: String, model: LLMModel): String
}

/**
 * A placeholder implementation of the InferenceService.
 *
 * This class is designed to be replaced with a real implementation that uses a native
 * inference library (e.g., llama.cpp) to run GGUF models locally.
 *
 * @param context The application context.
 */
class LlamaInferenceService(private val context: Context) : InferenceService {

    private var isModelLoaded = false

    init {
        System.loadLibrary("llm-hub-jni")
    }

    private external fun initLlama(modelPath: String): Int
    private external fun generateResponse(prompt: String): String
    private external fun releaseLlama()

    companion object {
        // Shared flow emitting each token sent from native side
        private val _tokens = MutableSharedFlow<String>(extraBufferCapacity = 64)
        val tokens: SharedFlow<String> = _tokens

        /** Called from native (JNI) for every generated token */
        @JvmStatic
        @Keep
        fun onNativeToken(text: String) {
            _tokens.tryEmit(text)
        }
    }

    // Make flow accessible for callers
    fun tokenStream(): SharedFlow<String> = tokens

    suspend fun loadModel(model: LLMModel): Boolean = withContext(Dispatchers.IO) {
        if (isModelLoaded) {
            releaseModel()
        }
        val modelsDir = File(context.filesDir, "models")
        val modelFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

        if (!modelFile.exists()) return@withContext false

        val result = initLlama(modelFile.absolutePath)
        isModelLoaded = result == 0
        isModelLoaded
    }

    suspend fun releaseModel() = withContext(Dispatchers.IO) {
        if (isModelLoaded) {
            releaseLlama()
            isModelLoaded = false
        }
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        return withContext(Dispatchers.IO) {
            if (!isModelLoaded) {
                if (!loadModel(model)) {
                    return@withContext "Error: Could not load model ${model.name}"
                }
            }
            generateResponse(prompt)
        }
    }
} 