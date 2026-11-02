package com.llmhub.llmhub.data

import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.random.Random

/**
 * Moves a small models folder to a removable SD card and back. Needs a card; on a device without one,
 * `adb shell sm set-virtual-disk true` and `sm partition disk:<id> public` make a 512 MB virtual one.
 * The device's real models are set aside during the test and put back after.
 */
class ModelStorageDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val models = File(context.filesDir, "models")
    private val parked = File(context.filesDir, "models.parked-by-test")

    @Before fun setUp() {
        assumeTrue("Needs a removable SD card", ModelStorage.sdCardRoot(context) != null)
        assertFalse(ModelStorage.isOnSdCard(context))
        if (models.exists()) assertTrue(models.renameTo(parked))
    }

    @After fun tearDown() {
        if (ModelStorage.isOnSdCard(context)) runBlocking { ModelStorage.move(context, toSdCard = false) {} }
        models.deleteRecursively()
        if (parked.exists()) parked.renameTo(models)
    }

    @Test fun movesToSdCardAndBack() = runBlocking {
        val weights = Random(7).nextBytes(64 shl 20)
        File(models, "bundle").mkdirs()
        File(models, "model.gguf").writeBytes(weights)
        File(models, "bundle/vae.safetensors").writeText("vae")

        val progress = mutableListOf<Float>()
        ModelStorage.move(context, toSdCard = true) { progress += it }.getOrThrow()
        assertTrue(ModelStorage.isOnSdCard(context))
        assertEquals(1f, progress.last())
        val card = File(ModelStorage.sdCardRoot(context), "models")
        assertEquals(weights.size.toLong(), File(card, "model.gguf").length())
        // The app keeps opening files/models; the link leads to the card.
        assertArrayEquals(weights, File(models, "model.gguf").readBytes())
        assertEquals("vae", File(models, "bundle/vae.safetensors").readText())
        assertEquals(weights.size + 3L, ModelStorage.usedBytes(context))

        ModelStorage.move(context, toSdCard = false) {}.getOrThrow()
        assertFalse(ModelStorage.isOnSdCard(context))
        assertFalse(card.exists())
        assertTrue(models.isDirectory)
        assertArrayEquals(weights, File(models, "model.gguf").readBytes())
        assertEquals("vae", File(models, "bundle/vae.safetensors").readText())
    }

    @Test fun refusesWhenTheCardIsTooSmall() = runBlocking {
        val free = ModelStorage.freeBytes(context, sdCard = true)
        assumeTrue("Card too large to fill in a test", free < 1L shl 30)
        // A sparse file as large as the card's free space.
        java.io.RandomAccessFile(File(models.apply { mkdirs() }, "big.gguf"), "rw").use { it.setLength(free + (1 shl 20)) }
        assertTrue(ModelStorage.move(context, toSdCard = true) {}.isFailure)
        assertFalse(ModelStorage.isOnSdCard(context))
        assertTrue(File(models, "big.gguf").isFile)
    }
}
