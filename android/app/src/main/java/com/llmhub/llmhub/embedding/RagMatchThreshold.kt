package com.llmhub.llmhub.embedding

import android.content.Context

/**
 * User-set RAG matching cut-offs (global memory and documents in chats). Every number in the
 * matching rules has its own value, saved per embedding model. An unset value is the built-in
 * default, so the rules stay exactly as before unless the user moves a slider:
 * EmbeddingGemma: sim > 0.65, or sim > 0.50 with word overlap > 0.035.
 * Gecko: sim > 0.80 with overlap > 0.05, or sim > 0.95 with overlap > 0.005.
 */
object RagMatchThreshold {
    private const val PREFS = "rag_prefs"

    enum class Kind { SIMILARITY, OVERLAP }

    enum class Param(val isEmbeddingGemma: Boolean, val key: String, val kind: Kind, private val defaultValue: Double) {
        GEMMA_SIMILARITY(true, "gemma_similarity", Kind.SIMILARITY, 0.65),
        GEMMA_SIMILARITY_WITH_OVERLAP(true, "gemma_similarity_with_overlap", Kind.SIMILARITY, 0.50),
        GEMMA_OVERLAP(true, "gemma_overlap", Kind.OVERLAP, 0.035),
        GECKO_SIMILARITY_WITH_OVERLAP(false, "gecko_similarity_with_overlap", Kind.SIMILARITY, 0.80),
        GECKO_OVERLAP(false, "gecko_overlap", Kind.OVERLAP, 0.05),
        GECKO_HIGH_SIMILARITY(false, "gecko_high_similarity", Kind.SIMILARITY, 0.95),
        GECKO_HIGH_OVERLAP(false, "gecko_high_overlap", Kind.OVERLAP, 0.005);

        val default: Double get() = defaultValue
        val min: Float get() = 0f
        val max: Float get() = if (kind == Kind.SIMILARITY) 1f else 0.30f
    }

    fun paramsFor(isEmbeddingGemma: Boolean): List<Param> = Param.values().filter { it.isEmbeddingGemma == isEmbeddingGemma }

    @Volatile private var appContext: Context? = null

    fun init(context: Context) {
        appContext = context.applicationContext
    }

    fun isEmbeddingGemma(modelName: String?): Boolean =
        modelName?.contains("EmbeddingGemma", ignoreCase = true) == true

    /** The saved value, or the exact built-in default when the user never changed it. */
    fun get(context: Context?, param: Param): Double {
        val ctx = context ?: appContext ?: return param.default
        val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return if (prefs.contains(param.key)) prefs.getFloat(param.key, 0f).toDouble() else param.default
    }

    fun set(context: Context, param: Param, value: Float) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putFloat(param.key, value.coerceIn(param.min, param.max)).apply()
    }

    /** Back to the built-in defaults for this embedding model. */
    fun reset(context: Context, isEmbeddingGemma: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().apply {
            paramsFor(isEmbeddingGemma).forEach { remove(it.key) }
            apply()
        }
    }

    fun isCustom(context: Context, isEmbeddingGemma: Boolean): Boolean {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return paramsFor(isEmbeddingGemma).any { prefs.contains(it.key) }
    }

    /** Short text of the values in use, for logs. */
    fun describe(context: Context?, isEmbeddingGemma: Boolean): String =
        paramsFor(isEmbeddingGemma).joinToString(", ") { "${it.key}=${get(context, it)}" }

    fun shouldAccept(context: Context?, isEmbeddingGemma: Boolean, similarity: Float, overlap: Double): Boolean {
        val sim = similarity.toDouble()
        // Compare as the original rules did: similarity against the Float literal, overlap as Double
        fun simOver(p: Param): Boolean {
            val v = get(context, p)
            return if (v == p.default) similarity > p.default.toFloat() else sim > v
        }
        fun overlapOver(p: Param): Boolean = overlap > get(context, p)
        return if (isEmbeddingGemma) {
            simOver(Param.GEMMA_SIMILARITY) ||
            (simOver(Param.GEMMA_SIMILARITY_WITH_OVERLAP) && overlapOver(Param.GEMMA_OVERLAP))
        } else {
            (simOver(Param.GECKO_SIMILARITY_WITH_OVERLAP) && overlapOver(Param.GECKO_OVERLAP)) ||
            (simOver(Param.GECKO_HIGH_SIMILARITY) && overlapOver(Param.GECKO_HIGH_OVERLAP))
        }
    }
}
