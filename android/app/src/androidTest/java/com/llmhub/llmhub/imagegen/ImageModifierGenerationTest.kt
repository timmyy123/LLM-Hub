package com.llmhub.llmhub.imagegen

import android.util.Log
import android.content.Context
import androidx.core.content.FileProvider
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.videogen.SdCppNative
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/** Real-device native inference coverage; fixtures are staged explicitly in app-private files. */
class ImageModifierGenerationTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun importedWeightsAndGenerationSettingsAffectOutput() {
        val fixtures = File(context.filesDir, "modifier-test-fixtures")
        val checkpoint = File(fixtures, "sd15.safetensors")
        assumeTrue("Stage a compatible SD1.5 checkpoint first", checkpoint.isFile)
        val phase = InstrumentationRegistry.getArguments().getString("modifierPhase")
        val componentsOnly = phase == "components"
        val testSteps = InstrumentationRegistry.getArguments().getString("modifierSteps")?.toIntOrNull()?.coerceIn(1, 4)
            ?: if (componentsOnly || phase == "strengths") 2 else 4
        val store = ImageExtrasStore(context)
        val original = store.load()
        val imported = mutableListOf<ImageExtra>()
        var handle = 0L
        try {
            fun import(name: String, kind: String): ImageExtra = store.importFile(
                FileProvider.getUriForFile(context, context.packageName + ".fileprovider", File(fixtures, name)), kind
            ).also { imported += it }
            val model = import("sd15.safetensors", "checkpoint")
            val lora = import("lora.safetensors", "lora")
            val embedding = import("embedding.safetensors", "embedding")
            val clip = import("clip.safetensors", "clipL")
            val vae = import("vae.safetensors", "vae")
            store.save(ImageGenerationOptions(assets = listOf(model)))
            assertTrue(store.checkpoints().any { it.path == model.path && it.type == ModelType.SDCPP })
            // Restore preferences before entering native code, including if native inference aborts.
            store.save(original)
            context.getSharedPreferences("image_generation_extras_v1", Context.MODE_PRIVATE).edit()
                .putString("options", original.json().put("negativePrompt", original.negativePrompt).toString()).commit()
            SdCppNative.ensureLoaded()
            var options = ImageGenerationOptions()
            fun load(value: ImageGenerationOptions) {
                if (handle != 0L) { SdCppImageNative.close(handle); handle = 0 }
                handle = SdCppImageNative.create(model.path, value.json().toString(), SdCppImageRuntime.usesGpu(true))
                assertTrue(handle != 0L)
                options = value
            }
            fun render(label: String, prompt: String = "a red ceramic teapot on a wooden table", strength: Float = 1f, input: ByteArray? = null): ByteArray {
                val output = SdCppImageNative.generate(handle, options.translatedPrompt(prompt), options.json().toString(), 256, 256, testSteps, 12345, strength, input)
                assertNotNull(label, output)
                val pixels = output!![1]
                assertEquals(256 * 256 * 3, pixels.size)
                assertTrue("$label must have image variation", pixels.distinct().size > 16)
                Log.i("ModifierGenerationTest", "PASS $label hash=${pixels.contentHashCode()}")
                return pixels
            }
            if (phase == "strengths") {
                load(ImageGenerationOptions(assets = listOf(embedding, clip, vae)))
                val reference = render("baseline with imported encoder, VAE and embedding")
                val low = render("image-to-image strength 0.3", strength = 0.3f, input = reference)
                val high = render("image-to-image strength 0.8", strength = 0.8f, input = reference)
                assertFalse(low.contentEquals(high))
                return
            }
            load(options)
            val baseline = render("baseline")
            if (!componentsOnly) {
                options = options.copy(guidance = 1f)
                assertFalse(baseline.contentEquals(render("guidance 1")))
                options = options.copy(guidance = 7f, clipSkip = 2)
                assertFalse(baseline.contentEquals(render("CLIP skip 2")))
                options = options.copy(clipSkip = 1, negativePrompt = "red, teapot, wood")
                assertFalse(baseline.contentEquals(render("negative prompt")))
                options = ImageGenerationOptions(assets = listOf(lora.copy(weight = 0.8f)))
                val weighted = render("LoRA weight 0.8")
                assertFalse(baseline.contentEquals(weighted))
                options = options.copy(assets = listOf(lora.copy(weight = 0.2f)))
                assertFalse(weighted.contentEquals(render("LoRA weight 0.2")))
                options = options.copy(assets = listOf(lora.copy(enabled = false)))
                assertFalse(weighted.contentEquals(render("LoRA disabled")))
            }
            load(ImageGenerationOptions(assets = listOf(clip, vae)))
            val withoutEmbedding = render("imported CLIP-L and VAE", "${embedding.name} a red ceramic teapot")
            load(ImageGenerationOptions(assets = listOf(embedding, clip, vae)))
            val withEmbedding = render("imported embedding", "${embedding.name} a red ceramic teapot")
            assertFalse("Embedding must change conditioning for the same prompt", withoutEmbedding.contentEquals(withEmbedding))
            val low = render("image-to-image strength 0.3", strength = 0.3f, input = baseline)
            val high = render("image-to-image strength 0.8", strength = 0.8f, input = baseline)
            assertFalse(low.contentEquals(high))
        } finally {
            if (handle != 0L) SdCppImageNative.close(handle)
            imported.forEach(store::remove)
            store.save(original)
        }
    }
}
