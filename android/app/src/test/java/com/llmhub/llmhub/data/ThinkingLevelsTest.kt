package com.llmhub.llmhub.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ThinkingLevelsTest {
    private fun model(name: String) = ModelData.models.first { it.name == name }
    private fun levels(name: String) = ThinkingLevels.forModel(model(name))!!

    @Test fun inklingHasNoOffItsLowestIsNone() {
        val first = levels("Inkling-2-Mini (UD-Q4_K_XL, GGUF)").levels.first()
        assertEquals(com.llmhub.llmhub.R.string.thinking_level_none, first.labelRes)
        assertEquals(0f, first.effort)
    }

    @Test fun theOtherPickersStartWithOff() {
        for (name in listOf("Muse Glimmer 30B (UD-IQ2_XXS)", "GPT-OSS 20B (Q4_K_M)", "Granite 4.2 3B (Q4_0)")) {
            val spec = levels(name)
            assertEquals(name, com.llmhub.llmhub.R.string.thinking_level_off, spec.levels.first().labelRes)
        }
    }

    @Test fun eachModelGetsItsOwnLevelsAndDefault() {
        assertEquals(listOf(0f, 0.1f, 0.2f, 0.7f, 0.9f, 0.99f), levels("Inkling-2-Mini (UD-Q4_K_XL, GGUF)").levels.map { it.effort })
        assertEquals(0.7f, levels("Inkling-2-Mini (UD-Q4_K_XL, GGUF)").default)
        assertEquals(ThinkingLevels.MEDIUM, levels("GPT-OSS 20B (Q4_K_M)").default)
        assertEquals(4, levels("GPT-OSS 20B (Q4_K_M)").levels.size)
        assertEquals(5, levels("Muse Glimmer 30B (UD-IQ2_XXS)").levels.size)
        assertEquals(3, levels("Granite 4.2 3B (Q4_0)").levels.size)
    }

    @Test fun anEffortSavedForAnotherModelFallsBackToTheDefault() {
        // ModelConfig's default (Inkling's 0.7) is not a Muse or Granite level.
        val muse = ThinkingLevels.forModel(ModelData.models.first { it.name.contains("Muse Glimmer") })!!
        assertEquals(ThinkingLevels.HIGH, muse.resolve(0.7f))
        assertEquals(ThinkingLevels.LOW, muse.resolve(ThinkingLevels.LOW))
        assertEquals(ThinkingLevels.HIGH, muse.resolve(null))
    }

    @Test fun modelsWithoutLevelsKeepTheSwitch() {
        assertNull(ThinkingLevels.forModel(ModelData.models.first { it.name.startsWith("Gemma-4 26B-A4B") && !it.name.contains("Projector") }))
        assertNull(ThinkingLevels.forModel(ModelData.models.first { it.name.startsWith("LFM3 1.7B") }))
    }

    @Test fun templateWords() {
        assertEquals("low", ThinkingLevels.word(ThinkingLevels.LOW))
        assertEquals("medium", ThinkingLevels.word(ThinkingLevels.MEDIUM))
        assertEquals("high", ThinkingLevels.word(ThinkingLevels.HIGH))
        assertEquals("xhigh", ThinkingLevels.word(ThinkingLevels.XHIGH))
    }
}
