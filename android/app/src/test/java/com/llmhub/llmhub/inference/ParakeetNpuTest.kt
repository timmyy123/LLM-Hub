package com.llmhub.llmhub.inference

import com.llmhub.llmhub.data.AsrModels
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class ParakeetNpuTest {
    @Test fun picksTheBuildCompiledForTheChip() {
        assertEquals("Qualcomm_SM8550", AsrModels.parakeetNpuBuild("SM8550")?.tag)
        assertEquals("Qualcomm_SM8650", AsrModels.parakeetNpuBuild("SM8650P")?.tag)
        assertEquals("Google_Tensor_G5", AsrModels.parakeetNpuBuild("Tensor G5")?.tag)
        assertEquals("Google_Tensor_G6", AsrModels.parakeetNpuBuild("Tensor G6")?.tag)
        assertEquals("MediaTek_MT6993", AsrModels.parakeetNpuBuild("MT6993Z/CZA")?.tag)
        // 8 Gen 1 has no build that works; 8+ Gen 1 and Exynos have none at all.
        for (soc in listOf("SM8450", "SM8475", "s5e9945", "MT6989")) assertNull(soc, AsrModels.parakeetNpuBuild(soc))
        assertTrue(AsrModels.parakeetNpuBuilds.all { it.url.endsWith("_stateful_${it.tag}.tflite") })
    }

    @Test fun chipsGoogleDidNotGetWorkingUseOurBuilds() {
        val ours = AsrModels.parakeetNpuBuilds.filter { "timmyboy26" in it.url }.map { it.tag }
        assertEquals(listOf("Qualcomm_SM8750", "Qualcomm_SM8850", "MediaTek_MT6991", "MediaTek_MT6993"), ours)
        assertTrue(AsrModels.parakeetNpuBuild("SM8550")!!.url.contains("litert-community"))
    }

    @Test fun lastWindowEndsWithTheAudio() {
        val second = ParakeetFeatures.SAMPLE_RATE
        assertEquals(listOf(0), ParakeetNpuEngine.windowStarts(2 * second))
        assertEquals(listOf(0), ParakeetNpuEngine.windowStarts(5 * second))
        assertEquals(listOf(0, second / 2), ParakeetNpuEngine.windowStarts(5 * second + second / 2))
        assertEquals(listOf(0, 3 * second, 6 * second, 7 * second), ParakeetNpuEngine.windowStarts(12 * second))
    }

    @Test fun melBandsFollowTheTone() {
        val tone = FloatArray(ParakeetFeatures.WINDOW_SAMPLES) { 0.5f * sin(2 * PI * 1000 * it / ParakeetFeatures.SAMPLE_RATE).toFloat() }
        val half = ParakeetFeatures.WINDOW_SAMPLES / 2
        // Silence then tone: the band holding 1 kHz must rise more than any other.
        val features = ParakeetFeatures.logMel(FloatArray(ParakeetFeatures.WINDOW_SAMPLES) { if (it < half) 0f else tone[it] })
        assertEquals(ParakeetFeatures.MELS * ParakeetFeatures.FRAMES, features.size)
        fun rise(band: Int) = features[band * ParakeetFeatures.FRAMES + 400] - features[band * ParakeetFeatures.FRAMES + 100]
        val loudest = (0 until ParakeetFeatures.MELS).maxByOrNull { rise(it) }!!
        // Slaney mel is linear below 1 kHz: 15 of 128 bands spanning 0 to ~45.2 mel.
        assertTrue("1 kHz landed in band $loudest", loudest in 40..44)
        for (band in 0 until ParakeetFeatures.MELS) {
            val row = features.copyOfRange(band * ParakeetFeatures.FRAMES, (band + 1) * ParakeetFeatures.FRAMES)
            assertTrue(abs(row.average()) < 1e-3)
        }
    }

    @Test fun paddingStaysZeroAndOutOfTheStatistics() {
        val audio = FloatArray(ParakeetFeatures.WINDOW_SAMPLES) { if (it < 16000) sin(it * 0.3).toFloat() * (it % 97) / 97f else 0f }
        val features = ParakeetFeatures.logMel(audio, validSamples = 16000)
        for (band in 0 until ParakeetFeatures.MELS) {
            val row = band * ParakeetFeatures.FRAMES
            assertTrue((100 until ParakeetFeatures.FRAMES).all { features[row + it] == 0f })
            assertTrue(abs((0 until 100).sumOf { features[row + it].toDouble() } / 100) < 1e-3)
        }
    }

    @Test fun buildsWordsFromPiecesAndSkipsControlTokens() {
        val vocabulary = listOf("", "<|nospeech|>", "▁he", "llo", "world", ".", "▁")
        val tokens = listOf(1 to 0, 2 to 3, 3 to 4, 6 to 9, 4 to 10, 5 to 12)
        assertEquals(
            listOf(ParakeetText.Word("hello", 1240), ParakeetText.Word("world.", 1720)),
            ParakeetText.words(tokens, vocabulary, offsetMs = 1000)
        )
    }

    @Test fun stitchesOverlappingWindowsWithoutRepeatingWords() {
        fun words(vararg pairs: Pair<String, Int>) = pairs.map { ParakeetText.Word(it.first, it.second) }
        val merged = words("the" to 3000, "quick" to 3400, "brown" to 3920, "fo" to 4700).toMutableList()
        // The next window starts at 3 s; it timed "brown" one frame later and heard the cut-off word in full.
        ParakeetText.merge(merged, words("quick" to 3400, "Brown," to 4000, "fox" to 4640, "jumps" to 5200), cutMs = 4000)
        assertEquals(listOf("the", "quick", "brown", "fox", "jumps"), merged.map { it.text })
    }
}
