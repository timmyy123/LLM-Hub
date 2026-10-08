package com.llmhub.llmhub.tts

import com.llmhub.llmhub.data.SupertonicModel
import com.llmhub.llmhub.ui.components.SupertonicEngine
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class SupertonicTest {
    private fun voice(): JSONObject = JSONObject().apply {
        put("style_ttl", JSONObject().put("dims", JSONArray(listOf(1, 50, 256)))
            .put("data", JSONArray(List(12800) { 0.25 })))
        put("style_dp", JSONObject().put("dims", JSONArray(listOf(1, 8, 16)))
            .put("data", JSONArray(List(128) { -0.5 })))
    }

    @Test fun importsExportedTensorData() {
        val parsed = SupertonicModel.parseVoice(voice().toString())
        assertEquals(12800, parsed.ttl.size)
        assertEquals(128, parsed.dp.size)
        assertEquals(0.25f, parsed.ttl.last(), 0f)
        assertEquals(-0.5f, parsed.dp.first(), 0f)
    }

    @Test fun importsNestedTensorData() {
        val json = voice()
        json.getJSONObject("style_dp").put("data", JSONArray().put(JSONArray(List(8) { JSONArray(List(16) { 0.5 }) })))
        assertEquals(0.5f, SupertonicModel.parseVoice(json.toString()).dp.last(), 0f)
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsWrongTensorDimensions() {
        val json = voice()
        json.getJSONObject("style_ttl").put("dims", JSONArray(listOf(1, 49, 256)))
        SupertonicModel.parseVoice(json.toString())
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsTruncatedVoiceData() {
        val json = voice()
        json.getJSONObject("style_dp").put("data", JSONArray(listOf(0.5)))
        SupertonicModel.parseVoice(json.toString())
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsNonNumericVoiceData() {
        val json = voice()
        json.getJSONObject("style_dp").getJSONArray("data").put(0, "0.5")
        SupertonicModel.parseVoice(json.toString())
    }

    @Test fun normalizesTextAndAppliesLanguageTags() {
        assertEquals("<en>Hello - world.</en>", SupertonicEngine.preprocess("Hello — world 😀", "en"))
        assertEquals("<ja>こんにちは。</ja>", SupertonicEngine.preprocess("こんにちは。", "ja"))
        assertEquals("<en><laugh> Hello!</en>", SupertonicEngine.preprocess("<laugh> Hello!", "en"))
    }

    @Test fun chunksUnbrokenMultilingualTextWithoutSplittingSurrogates() {
        val original = "あ𠮷".repeat(301)
        val chunks = SupertonicEngine.chunks(original, 120)
        assertEquals(original, chunks.joinToString(""))
        assertTrue(chunks.all { it.codePointCount(0, it.length) <= 120 })
        assertTrue(chunks.all { !Character.isLowSurrogate(it.first()) && !Character.isHighSurrogate(it.last()) })
    }

    @Test fun chunkingKeepsWordBoundariesWherePossible() {
        assertEquals(listOf("Hello world", "next words"), SupertonicEngine.chunks("Hello world next words", 16))
        assertTrue(SupertonicEngine.chunks("   ", 2).isEmpty())
    }
}
