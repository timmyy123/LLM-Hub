package com.llmhub.llmhub.tts

import com.llmhub.llmhub.data.AsrModels
import com.llmhub.llmhub.data.LiteRtTtsModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.ui.components.LiteRtTtsEngine
import org.junit.Assert.*
import org.junit.Test

class LiteRtSpeechModelsTest {
    @Test fun catalogListsTheLiteRtSpeechModels() {
        val kokoro = ModelData.ttsModels.single { it.name == LiteRtTtsModel.KOKORO_NAME }
        val pocket = ModelData.ttsModels.single { it.name == LiteRtTtsModel.POCKET_NAME }
        val parakeet = ModelData.models.single { it.name == AsrModels.PARAKEET_NAME }
        for (model in listOf(kokoro, pocket, parakeet)) {
            assertEquals("litertlm", model.modelFormat)
            assertTrue(model.url.endsWith(".litertlm"))
            assertTrue(model.additionalFiles.isEmpty())
        }
        assertEquals("tts", kokoro.category)
        assertEquals("asr", parakeet.category)
        assertTrue(AsrModels.isParakeet(parakeet))
        assertFalse(AsrModels.isParakeet(kokoro))
    }

    @Test fun mlKitRecognizersAreSystemManagedAsrModels() {
        val mlKit = ModelData.models.filter { AsrModels.isMlKit(it) }
        assertEquals(listOf(AsrModels.MLKIT_BASIC_NAME, AsrModels.MLKIT_ADVANCED_NAME), mlKit.map { it.name })
        assertTrue(mlKit.all { it.category == "asr" && it.sizeBytes == 0L })
        assertEquals(listOf(false, true), mlKit.map(AsrModels::isMlKitAdvanced))
    }

    @Test fun keepsAStoredVoiceOnlyWhenTheModelPackagesIt() {
        assertEquals("af_sky", LiteRtTtsModel.resolveVoice(LiteRtTtsModel.KOKORO_NAME, "af_sky"))
        // "af" is an ONNX-only voice file; "af_sky" is not a PocketTTS voice.
        assertEquals("af_heart", LiteRtTtsModel.resolveVoice(LiteRtTtsModel.KOKORO_NAME, "af"))
        assertEquals("alba", LiteRtTtsModel.resolveVoice(LiteRtTtsModel.POCKET_NAME, "af_sky"))
        assertEquals("lola", LiteRtTtsModel.resolveVoice(LiteRtTtsModel.POCKET_NAME, "lola"))
    }

    @Test fun onlyPocketSendsALocaleWithTheVoice() {
        assertNull(LiteRtTtsModel.synthesisLocale(LiteRtTtsModel.KOKORO_NAME, "ef_dora"))
        assertEquals("es-ES", LiteRtTtsModel.synthesisLocale(LiteRtTtsModel.POCKET_NAME, "lola"))
        assertEquals("en-US", LiteRtTtsModel.synthesisLocale(LiteRtTtsModel.POCKET_NAME, "alba"))
    }

    @Test fun labelsEveryVoice() {
        val kokoro = LiteRtTtsModel.voices(LiteRtTtsModel.KOKORO_NAME).toMap()
        assertEquals(54, kokoro.size)
        assertEquals("American Female — Heart", kokoro["af_heart"])
        assertEquals("Chinese Male — Yunxi", kokoro["zm_yunxi"])
        val pocket = LiteRtTtsModel.voices(LiteRtTtsModel.POCKET_NAME).toMap()
        assertEquals(14, pocket.size)
        assertEquals("German — Juergen", pocket["juergen"])
    }

    @Test fun dropsQuotationMarksButKeepsContractions() {
        assertEquals("She said don’t go.", LiteRtTtsEngine.sanitize("She said “don’t go.”"))
        assertEquals("It's fine", LiteRtTtsEngine.sanitize("'It's fine'"))
    }
}
