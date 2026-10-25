package com.llmhub.llmhub.data

import android.util.Log
import com.llmhub.llmhub.BuildConfig
import java.net.HttpURLConnection
import java.net.URL

/**
 * The bundled Hugging Face tokens (local.properties HF_TOKEN, then HF_TOKEN_FALLBACK). The primary is
 * used until Hugging Face rejects it (expired or revoked), then the fallback.
 */
object HfDefaultToken {
    private const val TAG = "HfDefaultToken"
    private val tokens = listOf(BuildConfig.HF_TOKEN, BuildConfig.HF_TOKEN_FALLBACK).filter { it.isNotBlank() }.distinct()

    @Volatile var current: String? = tokens.firstOrNull()
        private set

    fun isBundled(token: String?) = token != null && token in tokens

    /** Checks the tokens in order (blocking network call) and keeps the first one Hugging Face accepts. */
    @Synchronized fun verify() {
        for ((index, token) in tokens.withIndex()) {
            val code = runCatching {
                val connection = (URL("https://huggingface.co/api/whoami-v2").openConnection() as HttpURLConnection).apply {
                    connectTimeout = 10_000
                    readTimeout = 10_000
                    setRequestProperty("Authorization", "Bearer $token")
                }
                connection.responseCode.also { connection.disconnect() }
            }.onFailure { Log.w(TAG, "Token check failed: $it") }.getOrNull()
            when (code) {
                200 -> { current = token; Log.i(TAG, "Using bundled token #${index + 1}"); return }
                401 -> Log.w(TAG, "Bundled token #${index + 1} rejected (HTTP 401)")
                else -> return // offline or server error: keep the current token
            }
        }
    }

    /** After [token] got HTTP 401: re-checks the bundled tokens and returns true if another one is now in use. */
    fun replaceRejected(token: String?): Boolean {
        if (!isBundled(token)) return false
        if (token == current) verify()
        return current != token
    }
}
