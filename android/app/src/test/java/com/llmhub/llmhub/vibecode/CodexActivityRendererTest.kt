package com.llmhub.llmhub.vibecode

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CodexActivityRendererTest {
    @Test fun nativeFinalSummaryIsNotDuplicatedAsThinking() {
        val renderer = CodexActivityRenderer()
        renderer.thinking("message", "The edit is verified.")
        renderer.thinking("message", "")
        val result = renderer.render("item/completed", JSONObject("""{"item":{"id":"message","type":"agentMessage","text":"The edit is verified."}}"""))!!
        assertEquals("The edit is verified.", result.text)
    }
    @Test fun modelMessageCompletionIsNotReportedAsSuccessfulFileExecution() {
        val result = CodexActivityRenderer().render("item/completed",
            JSONObject("""{"item":{"id":"message","type":"agentMessage","text":"Updated the file"}}"""))!!
        assertEquals("running", result.state)
    }
    @Test fun liveThinkingAndValidatedAnswerUseOneCard() {
        val render = CodexActivityRenderer()
        val cards = mutableMapOf<String, CodexActivity>()
        render.thinking("msg-1", "Inspecting files")!!.let { cards[it.key] = it }
        render.render("item/agentMessage/delta", JSONObject("""{"itemId":"msg-1","delta":"Done"}"""))!!.let { cards[it.key] = it }
        render.render("item/completed", JSONObject("""{"item":{"id":"msg-1","type":"agentMessage","text":"Done"}}"""))!!.let { cards[it.key] = it }
        assertEquals(1, cards.size)
        assertEquals("<think>Inspecting files</think>\n\nDone", cards.getValue("msg-1").text)
    }
    @Test fun commandOutputAccumulatesLiveAndRetainsCommandHeader() {
        val render = CodexActivityRenderer()
        val started = render.render("item/started", JSONObject("""{"item":{"type":"commandExecution","id":"1","command":"npm test"}}"""))!!
        assertEquals("$ npm test\n", started.text)
        assertEquals("running", started.state)
        val first = render.render("item/commandExecution/outputDelta", JSONObject("""{"itemId":"1","delta":"test started\n"}"""))!!
        assertEquals("$ npm test\ntest started\n", first.text)
        val second = render.render("item/commandExecution/outputDelta", JSONObject("""{"itemId":"1","delta":"assertion failed\n"}"""))!!
        assertTrue(second.text.endsWith("test started\nassertion failed\n"))
        assertEquals("terminal", second.role)
        val end = render.render("item/completed", JSONObject("""{"item":{"type":"commandExecution","id":"1","command":"npm test","status":"failed","exitCode":1}}"""))!!
        assertEquals(second.text, end.text)
        assertEquals("failed", end.state)
    }
    @Test fun finalAggregateReplacesStreamInsteadOfDuplicatingIt() {
        val render = CodexActivityRenderer()
        render.render("item/commandExecution/outputDelta", JSONObject("""{"itemId":"1","delta":"ok\n"}"""))
        val end = render.render("item/completed", JSONObject("""{"item":{"type":"commandExecution","id":"1","command":"test","exitCode":0,"aggregatedOutput":"ok\n"}}"""))!!
        assertEquals("$ test\nok\n", end.text)
        assertEquals("succeeded", end.state)
    }
    @Test fun tomlUrlLiteralHasNoAndroidJsonSlashEscapes() {
        val url = "http://127.0.0.1:35405/token/v1"
        assertEquals("'$url'", CodexConfig.literal(url))
        assertFalse(CodexConfig.literal(url).contains("\\/"))
        val arg = "model_providers.llmhub_local.base_url=" + CodexConfig.literal(url)
        val process = ProcessBuilder("bash", "-c", "printf '%s' ${CodexConfig.shellQuote(arg)}").start()
        assertEquals(arg, process.inputStream.bufferedReader().readText())
        assertEquals(0, process.waitFor())
    }
}
