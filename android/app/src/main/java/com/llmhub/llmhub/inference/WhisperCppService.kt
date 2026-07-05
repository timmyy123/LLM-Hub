package com.llmhub.llmhub.inference

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Wrapper around whisper.cpp JNI for on-device speech-to-text.
 * Replaces the Nexa SDK AsrWrapper with direct whisper.cpp integration.
 *
 * whisper.cpp uses GGML .bin model files (same format the app already downloads
 * from huggingface.co/ggerganov/whisper.cpp).
 */
class WhisperCppService(private val context: Context) {

    companion object {
        private const val TAG = "WhisperCppService"

        init {
            try {
                System.loadLibrary("whisper_jni")
                Log.i(TAG, "Loaded libwhisper_jni.so")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load libwhisper_jni.so", e)
            }
        }
    }

    private var contextPtr: Long = 0L

    val isLoaded: Boolean get() = contextPtr != 0L

    suspend fun loadModel(modelPath: String): Boolean = withContext(Dispatchers.IO) {
        if (contextPtr != 0L) {
            release()
        }
        val file = File(modelPath)
        if (!file.exists()) {
            Log.e(TAG, "Model file not found: $modelPath")
            return@withContext false
        }
        contextPtr = nativeInitContext(modelPath)
        if (contextPtr == 0L) {
            Log.e(TAG, "Failed to init whisper context from: $modelPath")
            return@withContext false
        }
        Log.i(TAG, "Model loaded: $modelPath")
        true
    }

    suspend fun transcribe(
        wavPath: String,
        language: String = "en"
    ): String? = withContext(Dispatchers.IO) {
        if (contextPtr == 0L) {
            Log.e(TAG, "Whisper context not initialized")
            return@withContext null
        }
        val langCode = when {
            language == "auto" -> ""
            language == "english" -> "en"
            else -> language
        }
        val result = nativeTranscribe(contextPtr, wavPath, langCode)
        result?.trim()
    }

    fun release() {
        if (contextPtr != 0L) {
            nativeFreeContext(contextPtr)
            contextPtr = 0L
            Log.i(TAG, "Whisper context released")
        }
    }

    private external fun nativeInitContext(modelPath: String): Long
    private external fun nativeFreeContext(contextPtr: Long)
    private external fun nativeTranscribe(contextPtr: Long, wavPath: String, language: String): String?
}
