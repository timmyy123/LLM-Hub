package com.llmhub.llmhub.data

import com.llmhub.llmhub.R
import kotlin.math.abs

/** One entry of a thinking-level picker: its label and the effort value saved in ModelConfig.thinkingEffort. */
data class ThinkingLevel(val labelRes: Int, val effort: Float)

/**
 * Models whose chat template takes a thinking level, with the levels the template names. The first entry is the
 * lowest choice: Off (thinking off, as the model's on/off switch did) for models that can stop thinking, and None for
 * Inkling 2, which has no off: its lowest level ("none", effort 0) still thinks, and that thought is shown.
 * The other entries turn thinking on at that level.
 *  - Inkling 2: "Thinking effort level: <0..0.99>" (InklingFormat.EFFORT_LEVELS).
 *  - Muse Glimmer: "Reasoning strength: low / medium / high / xhigh" (template default high).
 *  - GPT-OSS: "Reasoning: low / medium / high" (Harmony; template default medium).
 *  - Granite 4.2: low effort appends "{reasoning effort: low}" to the last user message; otherwise normal (high).
 * Effort values other than Inkling's are codes 1..4 (low, medium, high, xhigh).
 */
class ThinkingLevels(val levels: List<ThinkingLevel>, val default: Float) {
    /** The saved effort if it is one of this model's levels, else the model's default. */
    fun resolve(saved: Float?): Float =
        saved?.takeIf { s -> levels.drop(1).any { abs(it.effort - s) < 0.001f } } ?: default

    companion object {
        const val LOW = 1f
        const val MEDIUM = 2f
        const val HIGH = 3f
        const val XHIGH = 4f
        private val OFF = ThinkingLevel(R.string.thinking_level_off, 0f)

        private val INKLING = ThinkingLevels(
            listOf(
                ThinkingLevel(R.string.thinking_level_none, 0f),
                ThinkingLevel(R.string.thinking_level_minimal, 0.1f),
                ThinkingLevel(R.string.thinking_level_low, 0.2f),
                ThinkingLevel(R.string.thinking_level_medium, 0.7f),
                ThinkingLevel(R.string.thinking_level_high, 0.9f),
                ThinkingLevel(R.string.thinking_level_max, 0.99f),
            ),
            INKLING_DEFAULT_EFFORT,
        )
        private val MUSE = ThinkingLevels(
            listOf(
                OFF,
                ThinkingLevel(R.string.thinking_level_low, LOW),
                ThinkingLevel(R.string.thinking_level_medium, MEDIUM),
                ThinkingLevel(R.string.thinking_level_high, HIGH),
                ThinkingLevel(R.string.thinking_level_max, XHIGH),
            ),
            HIGH,
        )
        private val GPT_OSS = ThinkingLevels(
            listOf(
                OFF,
                ThinkingLevel(R.string.thinking_level_low, LOW),
                ThinkingLevel(R.string.thinking_level_medium, MEDIUM),
                ThinkingLevel(R.string.thinking_level_high, HIGH),
            ),
            MEDIUM,
        )
        private val GRANITE = ThinkingLevels(
            listOf(
                OFF,
                ThinkingLevel(R.string.thinking_level_low, LOW),
                ThinkingLevel(R.string.thinking_level_high, HIGH),
            ),
            HIGH,
        )

        fun forModel(model: LLMModel?): ThinkingLevels? {
            if (model == null || model.name.contains("Vision Projector", ignoreCase = true)) return null
            val name = model.name.lowercase()
            return when {
                model.isInkling2() -> INKLING
                name.contains("muse glimmer") || name.contains("muse-glimmer") -> MUSE
                name.contains("gpt-oss") || name.contains("gpt_oss") -> GPT_OSS
                name.contains("granite 4.2") || name.contains("granite-4.2") -> GRANITE
                else -> null
            }
        }

        /** The words the Muse Glimmer and GPT-OSS templates use for an effort code. */
        fun word(effort: Float): String = when {
            effort <= LOW -> "low"
            effort <= MEDIUM -> "medium"
            effort <= HIGH -> "high"
            else -> "xhigh"
        }
    }
}
