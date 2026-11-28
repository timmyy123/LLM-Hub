package com.llmhub.llmhub.imagegen

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.nio.file.Files

class BundledModelDefaultsTest {
    @Test
    fun parsesDmd2Config() {
        val d = BundledModelDefaults.parse("""{"default_scheduler":"lcm","default_steps":8,"default_cfg":1}""")
        assertEquals(BundledModelDefaults("lcm", 8, 1f), d)
    }

    @Test
    fun parsesAnimaConfigWithPrompts() {
        val d = BundledModelDefaults.parse(
            """{"default_prompt":"x","default_negative_prompt":"y","default_scheduler":"euler","default_steps":10,"default_cfg":1}"""
        )
        assertEquals(BundledModelDefaults("euler", 10, 1f), d)
    }

    @Test
    fun ignoresUnknownSchedulerAndOtherConfigs() {
        assertNull(BundledModelDefaults.parse("""{"default_scheduler":"ddim","default_steps":8}"""))
        assertNull(BundledModelDefaults.parse("""{"_name_or_path":"openai/clip"}"""))
    }

    @Test
    fun findsNestedConfigNextToUnet() {
        val root = Files.createTempDirectory("sd").toFile()
        try {
            val dir = root.resolve("output/qnn_models_sdxl_8gen3").apply { mkdirs() }
            dir.resolve("unet.bin").writeText("")
            dir.resolve("config.json").writeText("""{"default_scheduler":"lcm","default_steps":8,"default_cfg":1}""")
            assertEquals(BundledModelDefaults("lcm", 8, 1f), BundledModelDefaults.read(root.path))
            assertNull(BundledModelDefaults.read(root.resolve("missing").path))
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun noConfigMeansNull() {
        val root = Files.createTempDirectory("sd").toFile()
        try {
            root.resolve("unet.bin").writeText("")
            assertNull(BundledModelDefaults.read(root.path))
        } finally {
            root.deleteRecursively()
        }
    }
}
