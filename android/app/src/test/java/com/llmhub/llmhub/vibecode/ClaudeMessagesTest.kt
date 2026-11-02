package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.net.HttpURLConnection
import java.net.URL

class ClaudeMessagesTest {
    private val tools = JSONArray("""[
        {"name":"Task","description":"Launch a new agent","input_schema":{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"]}},
        {"name":"Read","description":"Reads a file.\n\nUsage: long text","input_schema":{"type":"object","properties":{"file_path":{"type":"string","description":"abs path"},"limit":{"type":"number"}},"required":["file_path"]}},
        {"name":"Write","description":"Writes a file","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}]""")
    /** A request in which the model already read a file, so finishing without a change is a real choice. */
    private val afterRead = """[{"role":"user","content":[{"type":"text","text":"hi"}]},
        {"role":"assistant","content":[{"type":"tool_use","id":"r0","name":"Read","input":{"file_path":"/storage/emulated/0/Code/dog.html"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"r0","content":"<h1>Dogs</h1>"}]}]"""
    private fun request(messages: String) = JSONObject().put("model", "llmhub-local").put("tools", tools)
        .put("system", JSONArray().put(JSONObject().put("type", "text").put("text", "You are Claude Code.\n<env>\nWorking directory: /storage/emulated/0/Code\nPlatform: linux\n</env>")))
        .put("messages", JSONArray(messages))

    @Test fun promptIsCompactAndShowsExecutedCallsWithTheirResults() {
        val prompt = ClaudeMessages.prompt(request("""[
            {"role":"user","content":[{"type":"text","text":"<system-reminder>huge CLAUDE.md</system-reminder>make a dog site"}]},
            {"role":"assistant","content":[{"type":"text","text":"I'll write it"},{"type":"tool_use","id":"t1","name":"Write","input":{"file_path":"/storage/emulated/0/Code/dog.html","content":"<h1>Dogs</h1>"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"File created successfully"}]}]"""))
        assertTrue(prompt.contains("working in /storage/emulated/0/Code"))
        assertTrue(prompt.contains("user: make a dog site"))
        assertFalse(prompt.contains("CLAUDE.md"))
        assertFalse(prompt.contains("I'll write it"))
        assertFalse(prompt.contains("<h1>Dogs</h1>"))
        assertTrue(prompt.contains("Executed tool result for Write to /storage/emulated/0/Code/dog.html (13 bytes): File created successfully"))
        assertFalse(prompt.contains("_bytes"))
        assertTrue(prompt.contains("The last file change above succeeded"))
        assertFalse(prompt.contains("\"name\":\"Task\""))
        assertFalse(prompt.contains("abs path"))
        assertFalse(prompt.contains("long text"))
    }

    @Test fun historyKeepsOnlyTheLatestCopyOfAFile() {
        val read = """{"type":"tool_use","id":"%s","name":"Read","input":{"file_path":"/storage/emulated/0/Code/dog.html"}}"""
        val prompt = ClaudeMessages.prompt(request("""[
            {"role":"user","content":[{"type":"text","text":"make a dog site"}]},
            {"role":"assistant","content":[${read.format("r0")}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"r0","content":"FIRST-TURN-COPY"}]},
            {"role":"user","content":[{"type":"text","text":"add a button"}]},
            {"role":"assistant","content":[${read.format("r1")}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"r1","content":"OLD-COPY"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"e1","name":"Edit","input":{"file_path":"/storage/emulated/0/Code/dog.html","old_string":"a","new_string":"b"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"e1","content":"updated"}]},
            {"role":"assistant","content":[${read.format("r2")}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"r2","content":"LATEST-COPY"}]}]"""))
        assertFalse(prompt.contains("FIRST-TURN-COPY"))
        assertFalse(prompt.contains("OLD-COPY"))
        assertTrue(prompt.contains("LATEST-COPY"))
        assertTrue(prompt.contains("user: add a button"))
    }

    @Test fun replyBecomesToolUseBlocksWithAbsolutePaths() {
        val content = ClaudeMessages.content("""{"text":"Writing","tool_calls":[{"name":"write","arguments":{"file_path":"dog.html","content":"<h1>Dogs</h1>"}}]}""", request("[]"))
        assertEquals("text", content.getJSONObject(0).getString("type"))
        val use = content.getJSONObject(1)
        assertEquals("Write", use.getString("name"))
        assertEquals("/storage/emulated/0/Code/dog.html", use.getJSONObject("input").getString("file_path"))
        assertEquals("tool_use", ClaudeMessages.stopReason(content))
        assertEquals("end_turn", ClaudeMessages.stopReason(ClaudeMessages.content("""{"text":"Done","tool_calls":[]}""", request("[]"))))
    }

    @Test fun aDamagedReplyIsReadAsItsCallOrRetriedNeverEndedEmpty() {
        val damaged = "```json\n\n\":\"Adding\",\"tool_calls\":[{\"name\":\"Write\",\"arguments\":{\"file_path\":\"/a/b.html\",\"content\":\"x\"}}]}"
        assertEquals("tool_use", ClaudeMessages.content(damaged, request("[]")).getJSONObject(0).getString("type"))
        assertTrue(runCatching { ClaudeMessages.content("""{"summary":"done"}""", request("[]")) }.exceptionOrNull() is org.json.JSONException)
    }

    @Test fun aClaimWithoutAnyToolIsRetriedUnderAGrammarThatRequiresACall() {
        val grammars = mutableListOf<String?>()
        val replies = mutableListOf("""{"text":"I've created dog.html.","tool_calls":[]}""",
            """{"text":"Writing","tool_calls":[{"name":"Write","arguments":{"file_path":"dog.html","content":"<h1>Dogs</h1>"}}]}""")
        LocalClaudeServer(infer = { _, g, _ -> grammars.add(g); replies.removeAt(0) }).use { server ->
            val (_, body) = post("${server.baseUrl}/v1/messages", request("""[{"role":"user","content":"make a dog site"}]""").toString())
            assertEquals("tool_use", JSONObject(body).getString("stop_reason"))
            assertTrue(grammars[0].orEmpty().contains("(call (ws \",\" ws call){0,2})?"))
            assertTrue(grammars[1].orEmpty().contains("\"[\" ws call (ws \",\" ws call){0,2} ws"))
        }
    }

    @Test fun misspelledToolCallsKeyStillNamesTheCalls() {
        val content = ClaudeMessages.content("""{"text":"Adding","toolcalls":[{"name":"Write","arguments":{"file_path":"/a/b.html","content":"x"}}]}""", request("[]"))
        assertEquals("tool_use", ClaudeMessages.stopReason(content))
    }

    @Test fun repairsAMissingTextValueAndMisspelledArgumentNames() {
        val missing = ClaudeMessages.content("""{"text": "tool_calls": [{"name": "Write", "arguments": {"file_path": "/a/b.html", "content": "x"}}]}""", request("[]"))
        assertEquals("tool_use", ClaudeMessages.stopReason(missing))
        val misspelled = ClaudeMessages.content("""{"text":"Adding","toolcalls":[{"name":"Write","arguments":{"filePath":"/a/b.html","Content":"x"}}]}""", request("[]"))
        val input = misspelled.getJSONObject(1).getJSONObject("input")
        assertEquals("/a/b.html", input.getString("file_path"))
        assertEquals("x", input.getString("content"))
    }

    @Test fun aFinishWithoutAnyChangeIsQuestionedOnce() {
        val replies = mutableListOf("""{"text":"I will add a button.","tool_calls":[]}""", """{"text":"No change is needed: the button exists.","tool_calls":[]}""")
        var seen = ""
        LocalClaudeServer(infer = { prompt, _, _ -> seen = prompt; replies.removeAt(0) }).use { server ->
            val (_, body) = post("${server.baseUrl}/v1/messages", request(afterRead).toString())
            assertTrue(seen.contains("Nothing has been changed for this request yet"))
            assertTrue(JSONObject(body).toString().contains("the button exists"))
        }
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsToolsThatWereNotOffered() {
        ClaudeMessages.content("""{"text":"","tool_calls":[{"name":"Task","arguments":{"prompt":"x"}}]}""", request("[]"))
    }

    @Test fun aRepeatedSuccessfulReadIsRejectedUntilSomethingChanges() {
        val read = """{"text":"Reading","tool_calls":[{"name":"Read","arguments":{"file_path":"/storage/emulated/0/Code/dog.html"}}]}"""
        val history = """[{"role":"user","content":[{"type":"text","text":"add a paragraph"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"r1","name":"Read","input":{"file_path":"/storage/emulated/0/Code/dog.html"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"r1","content":"1  <h1>Dogs</h1>"}]}]"""
        val error = runCatching { ClaudeMessages.content(read, request(history)) }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException && error.message!!.contains("Repeated Read"))
        // A later request, or a change to the file, makes reading it again legitimate.
        val edited = history.removeSuffix("]") + """,{"role":"assistant","content":[{"type":"tool_use","id":"e1","name":"Write","input":{"file_path":"/storage/emulated/0/Code/dog.html","content":"x"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"e1","content":"ok"}]}]"""
        assertEquals("tool_use", ClaudeMessages.content(read, request(edited)).getJSONObject(1).getString("type"))
    }

    @Test fun streamsAnthropicEventsInOrder() {
        val content = ClaudeMessages.content("""{"text":"Writing","tool_calls":[{"name":"Write","arguments":{"file_path":"/a/b.html","content":"x"}}]}""", request("[]"))
        val types = ClaudeMessages.events(ClaudeMessages.message("msg_1", "llmhub-local", content, 100, 10)).map { it.getString("type") }
        assertEquals(listOf("message_start", "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"), types)
    }

    @Test fun serverAnswersMessagesFromTheConstrainedLocalModel() {
        var grammar: String? = null
        var calls = 0
        LocalClaudeServer(infer = { _, g, _ -> calls++; grammar = g
            """{"text":"Writing","tool_calls":[{"name":"Write","arguments":{"file_path":"dog.html","content":"<h1>Dogs</h1>"}}]}""" }).use { server ->
            val body = request("""[{"role":"user","content":"make a dog site"}]""").put("stream", true).put("max_tokens", 4096)
            val (status, stream) = post("${server.baseUrl}/v1/messages", body.toString())
            assertEquals(200, status)
            assertTrue(stream.contains("event: message_start"))
            assertTrue(stream.contains("\"type\":\"tool_use\""))
            assertTrue(stream.contains("\"stop_reason\":\"tool_use\""))
            assertTrue(stream.contains("event: message_stop"))
            assertTrue(grammar.orEmpty().contains("\"\\\"Write\\\"\""))
            assertFalse(grammar.orEmpty().contains("Task"))
            // Side requests without tools never reach the model.
            post("${server.baseUrl}/v1/messages", JSONObject().put("max_tokens", 1).put("messages", JSONArray("""[{"role":"user","content":"quota"}]""")).toString())
            assertEquals(1, calls)
            val (_, count) = post("${server.baseUrl}/v1/messages/count_tokens", body.toString())
            assertTrue(JSONObject(count).getInt("input_tokens") > 0)
            assertEquals(404, post(server.baseUrl.replace(Regex("/[a-f0-9-]+$"), "/wrong") + "/v1/messages", body.toString()).first)
        }
    }

    @Test fun generationStopsOnceTheReplyIsComplete() {
        var emitted = 0
        LocalClaudeServer(infer = { _, _, emit ->
            for (piece in listOf("{\"text\":\"Done\",", "\"tool_calls\":[]", "}", " and the model keeps going", " forever")) { emitted++; emit(piece) }
            error("generation should have been stopped")
        }).use { server ->
            val (status, body) = post("${server.baseUrl}/v1/messages", request(afterRead).toString())
            assertEquals(200, status)
            // Three pieces per attempt, and two attempts: a finish without any change is questioned once.
            assertEquals(6, emitted)
            assertEquals("end_turn", JSONObject(body).getString("stop_reason"))
        }
    }

    @Test fun aReplyThatNeverEndsIsCutOffAndRetried() {
        var attempts = 0
        LocalClaudeServer(infer = { _, _, emit ->
            if (++attempts == 1) { repeat(10_000) { emit("{\"text\":\"aaaa") }; error("generation should have been stopped") }
            """{"text":"Done","tool_calls":[]}"""
        }).use { server ->
            val (status, body) = post("${server.baseUrl}/v1/messages", request(afterRead).toString())
            assertEquals(200, status)
            assertEquals(2, attempts)
            assertEquals("end_turn", JSONObject(body).getString("stop_reason"))
        }
    }

    private fun post(url: String, body: String): Pair<Int, String> {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json")
        connection.outputStream.use { it.write(body.toByteArray()) }
        val status = connection.responseCode
        val text = (if (status < 400) connection.inputStream else connection.errorStream)?.bufferedReader()?.readText().orEmpty()
        return status to text
    }
}
