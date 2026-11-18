package com.llmhub.llmhub.embedding

import android.content.Context

/**
 * User-set similarity cut-off for RAG matching (global memory and documents in chats),
 * saved per embedding model. Unset means the built-in default, which keeps the original rules:
 * EmbeddingGemma: sim > 0.65, or sim > 0.50 with word overlap > 0.035.
 * Gecko: sim > 0.80 with overlap > 0.05, or sim > 0.95 with overlap > 0.005.
 * A custom value moves those cut-offs by the same amount.
 */
object RagMatchThreshold {
    private const val PREFS = "rag_prefs"
    const val GEMMA_DEFAULT = 0.65f
    const val GECKO_DEFAULT = 0.80f
    const val MIN = 0.20f
    const val MAX = 0.95f

    @Volatile private var appContext: Context? = null

    fun init(context: Context) {
        appContext = context.applicationContext
    }

    fun isEmbeddingGemma(modelName: String?): Boolean =
        modelName?.contains("EmbeddingGemma", ignoreCase = true) == true

    fun defaultFor(isEmbeddingGemma: Boolean): Float = if (isEmbeddingGemma) GEMMA_DEFAULT else GECKO_DEFAULT

    private fun key(isEmbeddingGemma: Boolean) = if (isEmbeddingGemma) "threshold_embeddinggemma" else "threshold_gecko"

    fun get(context: Context?, isEmbeddingGemma: Boolean): Float {
        val ctx = context ?: appContext ?: return defaultFor(isEmbeddingGemma)
        val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return prefs.getFloat(key(isEmbeddingGemma), defaultFor(isEmbeddingGemma))
    }

    /** null resets to the default. */
    fun set(context: Context, isEmbeddingGemma: Boolean, value: Float?) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().apply {
            if (value == null) remove(key(isEmbeddingGemma)) else putFloat(key(isEmbeddingGemma), value.coerceIn(MIN, MAX))
            apply()
        }
    }

    fun shouldAccept(isEmbeddingGemma: Boolean, threshold: Float, similarity: Float, overlap: Double): Boolean {
        val shift = threshold - defaultFor(isEmbeddingGemma)
        return if (isEmbeddingGemma) {
            (similarity > 0.65f + shift) ||
            (similarity > 0.50f + shift && overlap > 0.035)
        } else {
            (similarity > 0.80f + shift && overlap > 0.05) ||
            (similarity > 0.95f + shift && overlap > 0.005)
        }
    }
}
