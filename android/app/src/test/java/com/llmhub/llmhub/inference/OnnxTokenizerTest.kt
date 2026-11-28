package com.llmhub.llmhub.inference

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.zip.GZIPInputStream

/**
 * Checks the ONNX engine's tokenizer against HuggingFace `tokenizers` on the real tokenizer.json files of the
 * catalog's ONNX chat models (LFM-2.5 1.2B and Ministral-3 3B). Expected ids, decodes and rendered chat
 * templates come from `src/test/resources/onnx_tokenizer/generate_fixture.py`.
 */
class OnnxTokenizerTest {

    private class Model(val key: String, val tokenizer: OnnxTokenizer, val fixture: JsonObject)

    companion object {
        private val fixtureRoot: JsonObject by lazy {
            val stream = OnnxTokenizerTest::class.java.classLoader!!.getResourceAsStream("onnx_tokenizer/fixture.json")
                ?: error("fixture.json missing from test resources")
            stream.reader(Charsets.UTF_8).use { JsonParser.parseReader(it).asJsonObject }
        }

        private fun load(key: String): Model {
            val fx = fixtureRoot[key].asJsonObject
            val gz = OnnxTokenizerTest::class.java.classLoader!!.getResourceAsStream("onnx_tokenizer/" + fx["tokenizer"].asString)
                ?: error("tokenizer for $key missing from test resources")
            val tok = GZIPInputStream(gz).bufferedReader(Charsets.UTF_8).use { reader ->
                OnnxTokenizer(reader, fx["tokenizer_config"].toString(), fx["generation_config"].toString())
            }
            return Model(key, tok, fx)
        }

        private val lfm by lazy { load("lfm") }
        private val ministral by lazy { load("ministral") }
    }

    /** Hand-picked cases plus 400 seeded random mixes (white space kinds, scripts, marks, special-token fragments). */
    private fun Model.cases() = (fixture["cases"].asJsonArray + fixture["fuzz"].asJsonArray).map { it.asJsonObject }

    private fun JsonObject.ids(): List<Long> = this["ids"].asJsonArray.map { it.asLong }

    private fun checkEncode(m: Model) {
        val cases = m.cases()
        val failures = mutableListOf<String>()
        var exact = 0
        for (c in cases) {
            val text = c["text"].asString
            val expected = c.ids()
            val actual = m.tokenizer.encode(text)
            if (actual == expected) exact++ else failures += "${m.key}: ${text.take(60).replace("\n", "\\n")}\n  expected $expected\n  actual   $actual"
        }
        println("OnnxTokenizerTest ${m.key}: $exact/${cases.size} encodings identical to HuggingFace tokenizers")
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private fun checkDecode(m: Model) {
        for (c in m.cases()) {
            val ids = c.ids().map { it.toInt() }
            assertEquals("${m.key} decode", c["decoded"].asString, m.tokenizer.decode(ids, skipSpecialTokens = false))
            assertEquals("${m.key} decode skip special", c["decoded_skip_special"].asString, m.tokenizer.decode(ids, skipSpecialTokens = true))
            // Round trip: our own ids decode back to the exact input text.
            val text = c["text"].asString
            assertEquals("${m.key} round trip", text, m.tokenizer.decode(m.tokenizer.encode(text).map { it.toInt() }, skipSpecialTokens = false))
        }
    }

    private fun checkStreaming(m: Model) {
        for (c in m.cases()) {
            val ids = c.ids().map { it.toInt() }
            val stream = m.tokenizer.newStreamDecoder()
            val chunks = ids.map { stream.push(it) } + stream.flush()
            val streamed = chunks.joinToString("")
            assertEquals("${m.key} streamed text", c["decoded_skip_special"].asString, streamed)
            chunks.forEach { assertFalse("${m.key} chunk with a broken character: $it", it.contains('�')) }
        }
    }

    @Test fun lfmEncodeMatchesHuggingFace() = checkEncode(lfm)
    @Test fun ministralEncodeMatchesHuggingFace() = checkEncode(ministral)
    @Test fun lfmDecodeMatchesHuggingFace() = checkDecode(lfm)
    @Test fun ministralDecodeMatchesHuggingFace() = checkDecode(ministral)
    @Test fun lfmStreamingDecodeKeepsMultiByteCharacters() = checkStreaming(lfm)
    @Test fun ministralStreamingDecodeKeepsMultiByteCharacters() = checkStreaming(ministral)

    @Test
    fun streamingSplitsEmojiAcrossTokensWithoutLosingIt() {
        val text = "Feier 🎉 und 日本語"
        val ids = lfm.tokenizer.encode(text).map { it.toInt() }
        // The emoji is several byte-level tokens; none of them alone is a whole character.
        assertTrue(ids.size > text.split(" ").size)
        val stream = lfm.tokenizer.newStreamDecoder()
        assertEquals(text, (ids.map { stream.push(it) } + stream.flush()).joinToString(""))
    }

    @Test
    fun chatTemplatesMatchModelTemplates() {
        lfm.fixture["templates"].asJsonArray.map { it.asJsonObject }.forEach { t ->
            assertEquals(t["rendered"].asString, OnnxChatPrompt.chatMl(t["app_prompt"].asString, lfm.tokenizer.bosToken!!))
        }
        ministral.fixture["templates"].asJsonArray.map { it.asJsonObject }.forEach { t ->
            assertEquals(
                t["rendered"].asString,
                OnnxChatPrompt.mistral(t["app_prompt"].asString, ministral.tokenizer.bosToken!!, ministral.tokenizer.eosToken!!),
            )
        }
    }

    @Test
    fun specialTokensAndEos() {
        val l = lfm.tokenizer
        assertEquals(7L, l.getEosTokenId()) // <|im_end|>
        assertTrue(l.getEosTokenIds().containsAll(listOf(7L, 2L)))
        assertEquals("<|startoftext|>", l.bosToken)
        assertEquals(listOf(1L, 6L), l.encode("<|startoftext|><|im_start|>"))
        assertTrue(l.isHiddenControlToken(7))
        assertTrue(l.isHiddenControlToken(10)) // <|tool_call_start|> is added as non-special
        assertFalse(l.isHiddenControlToken(l.tokenToId("</think>")!!))

        val m = ministral.tokenizer
        assertEquals(2L, m.getEosTokenId()) // </s>
        assertEquals("<s>", m.bosToken)
        assertEquals(listOf(1L, 3L), m.encode("<s>[INST]"))
        assertEquals(12, m.tokenToId("[IMG_BREAK]"))
        assertEquals(13, m.tokenToId("[IMG_END]"))
    }

    @Test
    fun thinkingOffAddsNoThinkToLastUserMessage() {
        val p = OnnxChatPrompt.chatMl("user: a\nassistant: <think>x</think>b\n\nuser: c\nassistant:", noThink = true)
        assertEquals(
            "<|startoftext|><|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think>x</think>b<|im_end|>\n" +
                "<|im_start|>user\n/no_think c<|im_end|>\n<|im_start|>assistant\n",
            p,
        )
    }

    @Test
    fun visionPromptPutsImageFirstInLastUserMessage() {
        val p = OnnxChatPrompt.mistral("system: S\n\nuser: What is this?\nassistant:", withImage = true)
        assertEquals("<s>[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][INST]${OnnxChatPrompt.IMAGE_PLACEHOLDER}What is this?[/INST]", p)
    }
}
