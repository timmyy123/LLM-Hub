package com.llmhub.llmhub.imagegen

import org.junit.Assert.*
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import java.io.File

/**
 * Opt-in physical-device SD3.5 Medium coverage for imported CLIP-L, CLIP-G and T5-XXL encoders.
 * The checkpoint has no text encoders, so every encoder comes from the app's import path.
 */
class Sd3GenerationDeviceTest {
    /** One load and render; run with -e vkEnv to compare ggml-vulkan driver workarounds. */
    @Test fun vulkanProbe() { DeviceGenerationHarness("sd3", "a red ceramic teapot on a wooden table, studio photograph").use { h ->
        val encoders = listOf(h.import("clip-l.safetensors", "clipL"), h.import("clip-g.safetensors", "clipG"), h.import("t5.gguf", "textEncoder"))
        h.load(h.import("sd3.gguf", "checkpoint"), ImageGenerationOptions(negativePrompt = "", guidance = 4.5f, assets = encoders))
        h.render("probe")
    } }

    @Test fun sd3TextEncodersAffectGeneration() {
        // -e downloadFixtures 1: fetch and derive the fixtures on the phone (no Device Farm host, e.g. Remote Test Lab).
        if (InstrumentationRegistry.getArguments().getString("downloadFixtures") == "1") {
            OnDeviceFixtures.prepareSd3(File(InstrumentationRegistry.getInstrumentation().targetContext.filesDir, "sd3-test-fixtures"))
        }
        DeviceGenerationHarness("sd3",
        "a red ceramic teapot on a wooden table, studio photograph").use { h ->
        val model = h.import("sd3.gguf", "checkpoint")
        val clipL = h.import("clip-l.safetensors", "clipL")
        val clipG = h.import("clip-g.safetensors", "clipG")
        val t5 = h.import("t5.gguf", "textEncoder")
        val alteredT5 = h.import("t5-negated-norm.gguf", "textEncoder")
        val defaults = ImageGenerationOptions(negativePrompt = "", guidance = 4.5f)
        val encoders = defaults.copy(assets = listOf(clipL, clipG, t5))
        h.load(model, encoders)
        val baseline = h.render("baseline")
        assertArrayEquals("Repeated SD3 generation must be deterministic", baseline, h.render("baseline-repeat"))
        h.options = encoders.copy(guidance = 1f)
        assertFalse("Guidance must affect SD3", baseline.contentEquals(h.render("guidance-1")))
        h.options = encoders.copy(negativePrompt = "red, teapot, wooden table")
        assertFalse("Negative prompt must affect SD3", baseline.contentEquals(h.render("negative-prompt")))
        h.load(model, defaults.copy(assets = listOf(clipL, clipG)))
        val withoutT5 = h.render("without-t5")
        assertFalse("Imported T5-XXL must contribute to SD3 conditioning", baseline.contentEquals(withoutT5))
        h.load(model, defaults.copy(assets = listOf(clipL, clipG, alteredT5)))
        val altered = h.render("altered-t5")
        assertFalse("Changing imported T5-XXL weights must change output", baseline.contentEquals(altered))
        assertFalse("An altered T5-XXL must not behave as a missing encoder", withoutT5.contentEquals(altered))
        h.load(model, defaults.copy(assets = listOf(clipL, t5)))
        assertFalse("Imported CLIP-G must contribute to SD3 conditioning", baseline.contentEquals(h.render("without-clip-g")))
        h.load(model, encoders)
        h.assertReloadMatches("Reloading the same encoders must reproduce the baseline", baseline, h.render("reloaded"), withoutT5)
        val low = h.render("image-strength-03", strength = 0.3f, input = baseline)
        val high = h.render("image-strength-08", strength = 0.8f, input = baseline)
        assertFalse("Image strength must affect SD3", low.contentEquals(high))
        h.render("preview-12-steps", steps = 12)
    } }
}
