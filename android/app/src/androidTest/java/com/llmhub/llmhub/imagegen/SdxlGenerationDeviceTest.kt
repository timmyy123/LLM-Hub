package com.llmhub.llmhub.imagegen

import org.junit.Assert.*
import org.junit.Test

/** Opt-in physical-device SDXL coverage. Missing fixtures fail rather than silently skip. */
class SdxlGenerationDeviceTest {
    /** One load and render; run with -e vkEnv to compare ggml-vulkan driver workarounds. */
    @Test fun vulkanProbe() { DeviceGenerationHarness("sdxl", "a red ceramic teapot on a wooden table, studio photograph").use { h ->
        h.load(h.import("sdxl.safetensors", "checkpoint"), ImageGenerationOptions(negativePrompt = "", guidance = 5f, clipSkip = 2))
        h.render("probe")
    } }

    @Test fun sdxlEncodersAndModifiersAffectGeneration() { DeviceGenerationHarness("sdxl",
        "a red ceramic teapot on a wooden table, studio photograph").use { h ->
        val model = h.import("sdxl.safetensors", "checkpoint")
        val clipG = h.import("clip-g.safetensors", "clipG")
        val alteredG = h.import("clip-g-zero-projection.safetensors", "clipG")
        val lora = h.import("lora.safetensors", "lora")
        val defaults = ImageGenerationOptions(negativePrompt = "", guidance = 5f, clipSkip = 2)
        h.load(model, defaults)
        val baseline = h.render("baseline")
        h.options = defaults.copy(guidance = 1f)
        assertFalse("Guidance must affect SDXL", baseline.contentEquals(h.render("guidance-1")))
        h.options = defaults.copy(negativePrompt = "red, teapot, wooden table")
        assertFalse("Negative prompt must affect SDXL", baseline.contentEquals(h.render("negative-prompt")))
        h.options = defaults.copy(clipSkip = 3)
        assertFalse("CLIP skip must affect SDXL", baseline.contentEquals(h.render("clip-skip-3")))
        h.options = defaults.copy(assets = listOf(lora.copy(weight = 0.8f)))
        val highWeight = h.render("lora-08")
        assertFalse("SDXL LoRA must affect pixels", baseline.contentEquals(highWeight))
        h.options = defaults.copy(assets = listOf(lora.copy(weight = 0.2f)))
        assertFalse("SDXL LoRA weight must affect pixels", highWeight.contentEquals(h.render("lora-02")))
        h.options = defaults.copy(assets = listOf(lora.copy(enabled = false)))
        assertArrayEquals("Disabling LoRA must restore the baseline", baseline, h.render("lora-disabled"))
        h.load(model, defaults.copy(assets = listOf(clipG)))
        val external = h.render("external-clip-g")
        h.load(model, defaults.copy(assets = listOf(alteredG)))
        val altered = h.render("altered-clip-g")
        assertFalse("Changing imported CLIP-G must change conditioning", baseline.contentEquals(altered))
        h.assertReloadMatches("An identical external CLIP-G must preserve output", baseline, external, altered)
        h.load(model, defaults.copy(assets = listOf(clipG)))
        val low = h.render("image-strength-03", strength = 0.3f, input = baseline)
        val high = h.render("image-strength-08", strength = 0.8f, input = baseline)
        assertFalse("Image strength must affect SDXL", low.contentEquals(high))
        h.render("preview-8-steps", steps = 8)
    } }
}
