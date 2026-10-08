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
    @Test fun stripsSentinelThinkingFromLlamaCppAndLFM() {
        val raw = "${CodexResponses.SENTINEL_THINK}I should check files${CodexResponses.SENTINEL_ENDTHINK}" +
            """{"text":"Files checked","tool_calls":[{"name":"shell","arguments":{"command":"ls"}}]}"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(1)
        assertEquals("function_call", call.getString("type"))
        assertEquals("ls", JSONObject(call.getString("arguments")).getString("command"))
    }

    @Test fun extractsJsonFromMarkdownCodeBlockWithPreambleAndTrailing() {
        val raw = """
            I will run ls to check current directory:
            ```json
            {"text":"Listing files","tool_calls":[{"name":"shell","arguments":{"command":"ls"}}]}
            ```
            Hope this helps!
        """.trimIndent()
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(1)
        assertEquals("function_call", call.getString("type"))
        assertEquals("ls", JSONObject(call.getString("arguments")).getString("command"))
    }

    @Test fun handlesDirectSingleToolCallWithoutWrapper() {
        val raw = """{"name":"shell","arguments":{"command":"pwd"}}"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(0)
        assertEquals("function_call", call.getString("type"))
        assertEquals("pwd", JSONObject(call.getString("arguments")).getString("command"))
    }

    @Test fun resolvesShellAliasesAndCmdParameter() {
        val raw = """{"text":"","tool_calls":[{"name":"bash","arguments":{"cmd":"whoami"}}]}"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(0)
        assertEquals("function_call", call.getString("type"))
        assertEquals("whoami", JSONObject(call.getString("arguments")).getString("command"))
    }

    @Test fun supportsTaggedToolCallsFromAgentModels() {
        val raw = """<tool_call>{"name":"shell","arguments":{"command":"git status"}}</tool_call>"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(0)
        assertEquals("function_call", call.getString("type"))
        assertEquals("git status", JSONObject(call.getString("arguments")).getString("command"))
    }

    @Test fun supportsPlainTextFallbackWhenAllowed() {
        val output = CodexResponses.output("Just answering your question directly.", request, allowPlainText = true)
        val msg = output.getJSONObject(0)
        assertEquals("message", msg.getString("type"))
        assertEquals("Just answering your question directly.", msg.getJSONArray("content").getJSONObject(0).getString("text"))
    }

    @Test fun adaptsWriteFileToShellCommand() {
        val raw = """{"text":"creating file","tool_calls":[{"name":"write_file","arguments":{"path":"index.html","content":"<h1>hello</h1>"}}]}"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(1)
        assertEquals("function_call", call.getString("type"))
        val cmd = JSONObject(call.getString("arguments")).getString("command")
        assertTrue(cmd.contains("cat << 'EOF_CODE' > \"index.html\""))
        assertTrue(cmd.contains("<h1>hello</h1>"))
    }

    @Test fun adaptsReadFileToShellCommand() {
        val raw = """{"text":"","tool_calls":[{"name":"read_file","arguments":{"path":"main.py"}}]}"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(0)
        assertEquals("function_call", call.getString("type"))
        val cmd = JSONObject(call.getString("arguments")).getString("command")
        assertEquals("cat \"main.py\"", cmd)
    }

    @Test fun cleansBracketedToolNames() {
        val raw = """{"text":"","tool_calls":[{"name":"[read_file","arguments":{"path":"gay.html"}}]}"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(0)
        assertEquals("function_call", call.getString("type"))
        val cmd = JSONObject(call.getString("arguments")).getString("command")
        assertEquals("cat \"gay.html\"", cmd)
    }

    @Test fun handlesBracketSyntaxToolCalls() {
        val raw = """I will read the file now: [read_file(path="gay.html")]"""
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(1)
        assertEquals("function_call", call.getString("type"))
        val cmd = JSONObject(call.getString("arguments")).getString("command")
        assertEquals("cat \"gay.html\"", cmd)
    }

    @Test fun formatDisplayMessagePreservesThinkingAndExtractsText() {
        val raw = "<think>I need to inspect the project</think>{\"text\":\"Inspecting files\",\"tool_calls\":[]}"
        val formatted = CodexResponses.formatDisplayMessage(raw)
        assertTrue(formatted.startsWith("<think>I need to inspect the project</think>"))
        assertTrue(formatted.contains("Inspecting files"))
    }

    @Test fun handlesTaggedBracketExecCommandWithWorkdir() {
        val raw = "<|tool_call_start|>[exec_command(command='cat gay.html', workdir='/data/data/com.termux/files/home/.llmhub-codex/workspaces/test')]<|tool_call_end|>"
        val output = CodexResponses.output(raw, request)
        val call = output.getJSONObject(0)
        assertEquals("function_call", call.getString("type"))
        val args = JSONObject(call.getString("arguments"))
        assertEquals("cat gay.html", args.getString("command"))
    }
}



