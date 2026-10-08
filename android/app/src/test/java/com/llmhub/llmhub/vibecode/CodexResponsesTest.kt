package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CodexResponsesTest {
    private val request = JSONObject("""{"tools":[
        {"type":"function","name":"shell","parameters":{"required":["command"]}},
        {"type":"custom","name":"apply_patch"},
        {"type":"namespace","name":"functions","tools":[{"type":"function","name":"exec","parameters":{}}]}
    ],"input":[{"type":"function_call_output","call_id":"old","output":"test failed"}]}""")

    @Test fun convertsValidatedFunctionCallToCodexWireFormat() {
        val output = CodexResponses.output("""{"text":"","tool_calls":[{"name":"shell","arguments":{"command":"ls"}}]}""", request)
        val call = output.getJSONObject(0)
        assertEquals("function_call", call.getString("type"))
        assertEquals("ls", JSONObject(call.getString("arguments")).getString("command"))
        assertTrue(call.getString("call_id").startsWith("call_"))
    }
    @Test fun preservesRawPatchInput() {
        val raw = JSONObject().put("text", "").put("tool_calls", JSONArray().put(JSONObject()
            .put("name", "apply_patch").put("input", "*** Begin Patch\n*** End Patch")))
        val item = CodexResponses.output(raw.toString(), request).getJSONObject(0)
        assertEquals("custom_tool_call", item.getString("type"))
        assertEquals("*** Begin Patch\n*** End Patch", item.getString("input"))
    }
    @Test fun supportsNamespacedTools() {
        val item = CodexResponses.output("""{"text":"","tool_calls":[{"name":"functions.exec","arguments":{}}]}""", request).getJSONObject(0)
        assertEquals("functions", item.getString("namespace"))
        assertEquals("exec", item.getString("name"))
    }
    @Test fun includesPriorToolResultsForNextAgentStep() {
        assertTrue(CodexResponses.prompt(request).contains("test failed"))
        assertTrue(CodexResponses.prompt(request).contains("apply_patch"))
    }
    @Test(expected = IllegalArgumentException::class) fun rejectsInventedTool() {
        CodexResponses.output("""{"text":"","tool_calls":[{"name":"invented","arguments":{}}]}""", request)
    }
    @Test(expected = IllegalArgumentException::class) fun rejectsMissingRequiredArgument() {
        CodexResponses.output("""{"text":"","tool_calls":[{"name":"shell","arguments":{}}]}""", request)
    }
    @Test(expected = org.json.JSONException::class) fun rejectsPretendCodeEdits() {
        CodexResponses.output("I edited your files!", request)
    }
    @Test fun convertsFinalAnswerAndRemovesThinking() {
        val item = CodexResponses.output("""<think>private</think>{"text":"Done","tool_calls":[]}""", request).getJSONObject(0)
        assertEquals("message", item.getString("type"))
        assertEquals("Done", item.getJSONArray("content").getJSONObject(0).getString("text"))
    }
    @Test fun previewsPartialTextWithoutRevealingToolArgumentsOrThinking() {
        assertEquals("Hello", CodexResponses.partialText("""{"text":"Hello"""))
        assertEquals("Hello\nworld", CodexResponses.partialText("""{"text":"Hello\nworld","tool_calls":[{"arguments":{"secret":1}}]}"""))
        assertEquals("", CodexResponses.partialText("<think>private reasoning"))
        assertEquals("Ready", CodexResponses.partialText("""<think>private</think>{"text":"Ready"""))
    }
    @Test fun partialUnicodeWaitsForCompleteEscapeAndSurrogatePair() {
        assertEquals("a", CodexResponses.partialText("""{"text":"a\u26"""))
        assertEquals("a", CodexResponses.partialText("""{"text":"a\ud83d"""))
        assertEquals("a😀", CodexResponses.partialText("""{"text":"a\ud83d\ude00"""))
    }

}
