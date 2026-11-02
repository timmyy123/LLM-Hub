package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CodexResponsesTest {
    @Test fun successfulEmptyReadTellsModelToWriteInsteadOfSearchingAgain() {
        fun prompt(result: String) = CodexResponses.prompt(JSONObject().put("input", JSONArray()
            .put(JSONObject().put("type", "function_call").put("call_id", "read").put("name", "exec_command")
                .put("arguments", "{\"cmd\":\"cat \\\"pig.html\\\"\"}"))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "read").put("output", result))))
        assertTrue(prompt("Process exited with code 0\nOutput:\n").contains("NEXT action is write_file"))
        assertFalse(prompt("Process exited with code 1\nOutput:\nNo such file").contains("NEXT action is write_file"))
        assertFalse(prompt("Process exited with code 0\nOutput:\n<html>").contains("NEXT action is write_file"))
    }
    @Test fun successfulWriteTellsModelNotToWriteTheSameFileAgain() {
        val write = "printf '%s' '" + java.util.Base64.getEncoder().encodeToString("<h1>Dogs</h1>".toByteArray()) + "' | base64 -d > 'dog.html'"
        fun prompt(result: String) = CodexResponses.prompt(JSONObject().put("input", JSONArray()
            .put(JSONObject().put("type", "function_call").put("call_id", "w").put("name", "exec_command")
                .put("arguments", JSONObject().put("cmd", write).toString()))
            .put(JSONObject().put("type", "function_call_output").put("call_id", "w").put("output", result))))
        assertTrue(prompt("Process exited with code 0\nOutput:\n").contains("Do not write it again"))
        assertFalse(prompt("Process exited with code 1\nOutput:\nPermission denied").contains("Do not write it again"))
        assertFalse(prompt("Process exited with code 0\nOutput:\n").contains(write))
    }
    @Test fun nativeFinishEndsWithOneSummaryAndNoShellCommand() {
        val output = CodexResponses.output("Finished. <tool_call>[finish(summary='Verified the file change')]</tool_call>", request)
        assertEquals(1, output.length())
        assertEquals("message", output.getJSONObject(0).getString("type"))
        assertEquals("Verified the file change", output.getJSONObject(0).getJSONArray("content").getJSONObject(0).getString("text"))
    }
    @Test(expected = IllegalArgumentException::class) fun cannotFinishWithPendingCommands() {
        CodexResponses.output("""{"tool_calls":[{"name":"finish","arguments":{"summary":"Done"}},{"name":"shell","arguments":{"command":"echo pending"}}]}""",request)
    }
    @Test fun jsonReplyIsNotDuplicatedInsideAnUnclosedThinkingPrefix() {
        val raw = CodexResponses.SENTINEL_THINK + "Verified the edit." + """{"text":"Fixed the typo.","tool_calls":[]}"""
        assertEquals("Verified the edit.", CodexResponses.parseThinking(raw).first)
        val display = CodexResponses.formatDisplayMessage(raw)
        assertEquals(1, Regex("Fixed the typo\\.").findAll(display).count())
        assertFalse(display.contains("tool_calls"))
    }
    @Test fun nativeToolCallEndsAnUnclosedThinkingPrefix() {
        val offered = JSONObject("""{"tools":[{"type":"function","name":"exec_command","parameters":{"properties":{"cmd":{"type":"string"}},"required":["cmd"]}}]}""")
        for (prefix in listOf(CodexResponses.SENTINEL_THINK, "<think>")) {
            val raw = prefix + "Reading now<|tool_call_start|>[read_file(path='/cwd/index.html')]<|tool_call_end|>"
            val call = CodexResponses.output(raw, offered).getJSONObject(0)
            assertEquals("cat \"index.html\"", JSONObject(call.getString("arguments")).getString("cmd"))
            assertEquals("Reading now", CodexResponses.parseThinking(raw).first)
            assertFalse(CodexResponses.formatDisplayMessage(raw).contains("tool_call"))
        }
    }
    @Test fun adaptsDeviceReadFileResponseToExecCommandSchema() {
        val offered = JSONObject("""{"tools":[{"type":"function","name":"exec_command","parameters":{"properties":{"cmd":{"type":"string"}},"required":["cmd"]}}]}""")
        val raw = "${CodexResponses.SENTINEL_THINK}Read before editing${CodexResponses.SENTINEL_ENDTHINK}" +
            "<|tool_call_start|>[read_file(path='/cwd/gay.html')]<|tool_call_end|>"
        val call = CodexResponses.output(raw, offered).getJSONObject(0)
        assertEquals("exec_command", call.getString("name"))
        val args = JSONObject(call.getString("arguments"))
        assertEquals("cat \"gay.html\"", args.getString("cmd"))
        assertFalse(args.has("command"))
        val display = CodexResponses.formatDisplayMessage(raw)
        assertTrue(display.contains("Read before editing"))
        assertFalse(display.contains("tool_call"))
        assertFalse(display.contains("read_file("))
    }

    @Test fun adaptsWriteAndShellAliasesToOfferedCmdParameter() {
        val offered = JSONObject("""{"tools":[{"type":"function","name":"exec_command","parameters":{"required":["cmd"]}}]}""")
        for (raw in listOf(
            """{"name":"write_file","arguments":{"path":"index.html","content":"hello"}}""",
            """{"name":"shell_command","arguments":{"command":"ls -la"}}"""
        )) {
            val call = CodexResponses.output(raw, offered).getJSONObject(0)
            assertTrue(JSONObject(call.getString("arguments")).getString("cmd").isNotBlank())
        }
    }
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
        assertTrue(cmd.contains("base64 -d > 'index.html'"))
        assertTrue(cmd.contains(java.util.Base64.getEncoder().encodeToString("<h1>hello</h1>".toByteArray())))
    }

    @Test fun repairsToolCallMissingItsClosingBrace() {
        // LFM3 VL 2B's reply to "make a dog site": the tool-call object is never closed before "]".
        val raw = """{"text":"Creating the site","tool_calls":[{"name":"write_file","arguments":{"path":"dog.html","content":"<h1>Dogs {}</h1>"}]}"""
        val output = CodexResponses.output(raw, request)
        val cmd = JSONObject(output.getJSONObject(1).getString("arguments")).getString("command")
        assertTrue(cmd.contains("base64 -d > 'dog.html'"))
        assertTrue(cmd.contains(java.util.Base64.getEncoder().encodeToString("<h1>Dogs {}</h1>".toByteArray())))
    }
    @Test fun repairsRepliesThatStopBeforeTheirFinalBraces() {
        val output = CodexResponses.output("""{"text":"Listing","tool_calls":[{"name":"shell","arguments":{"command":"ls"}""", request)
        assertEquals("function_call", output.getJSONObject(1).getString("type"))
    }
    @Test fun replaceInFileUsesTheProjectFileForAPathIntoCodexHome() {
        val raw = """{"text":"","tool_calls":[{"name":"replace_in_file","arguments":{"path":"/storage/emulated/0/Code/data/data/com.termux/files/home/.llmhub-codex/skills/.system/dog.html","old_text":"blue","new_text":"red"}}]}"""
        val cmd = JSONObject(CodexResponses.output(raw, request).getJSONObject(0).getString("arguments")).getString("command")
        assertEquals("replace_in_file({\"path\":\"dog.html\",\"old_text_bytes\":4,\"replacement_bytes\":3})", CodexFileEdits.commandSummary(cmd))
    }
    @Test fun repairsAnArrayLeftOpenBeforeAnObjectKey() {
        // LFM3 VL 2B: `"tool_calls":[{...},"summary":"..."}` never closes tool_calls.
        val raw = """{"text":"Added","tool_calls":[{"name":"write_file","arguments":{"path":"dog.html","content":"<p>Goldens</p>"}},"summary":"Added a paragraph"}"""
        val output = CodexResponses.output(raw, request)
        val cmd = JSONObject(output.getJSONObject(1).getString("arguments")).getString("command")
        assertTrue(cmd.contains(java.util.Base64.getEncoder().encodeToString("<p>Goldens</p>".toByteArray())))
    }
    @Test fun repairsAKeyMissingItsOpeningQuote() {
        val raw = """{"text":"Red","tool_calls":[{"name":"replace_in_file","arguments":{"path":"dog.html","old_text":"<h1>",new_text":"<h1 style='color:red'>"}}]}"""
        val cmd = JSONObject(CodexResponses.output(raw, request).getJSONObject(1).getString("arguments")).getString("command")
        assertEquals("replace_in_file({\"path\":\"dog.html\",\"old_text_bytes\":4,\"replacement_bytes\":22})", CodexFileEdits.commandSummary(cmd))
    }
    @Test fun readsGemmasNativeToolCallFormat() {
        val q = "<|\"|>"
        val output = CodexResponses.output("Checking it.<|tool_call>call:exec{command:${q}cat dog.html${q},max_output_tokens:2048}<tool_call|>", request)
        assertEquals("message", output.getJSONObject(0).getString("type"))
        val args = JSONObject(output.getJSONObject(1).getString("arguments"))
        assertEquals("cat dog.html", args.getString("command"))
        val (_, calls) = CodexResponses.gemmaToolCalls("<|tool_call>call:Read{file_path:${q}/a/b, c.html${q}}")!!
        assertEquals("/a/b, c.html", calls.getJSONObject(0).getJSONObject("arguments").getString("file_path"))
        val (_, write) = CodexResponses.gemmaToolCalls("<|tool_call>call:write_file{path:${q}a.css${q},content:${q}body { color: red; }${q}}<tool_call|>")!!
        assertEquals("body { color: red; }", write.getJSONObject(0).getJSONObject("arguments").getString("content"))
    }

    @Test fun recognisesOutputThatCollapsedIntoRepetition() {
        val start = "```json\n{\n  \"text\": \"Adding a button\",\n  \"tool_calls\": [\n    {\n      \"name\"\": \"\"\",\n"
        val collapsed = start + (1..200).joinToString("") { if (it % 3 == 0) "          \n" else "          \"\n" }
        assertTrue(CodexResponses.degenerate(collapsed))
        assertTrue(CodexResponses.degenerate(start + "abcabcabc".repeat(150)))
        val page = (1..60).joinToString("\n") { "<p class=\"item-$it\">Golden retriever fact number $it: they love to fetch and swim.</p>" }
        assertFalse(CodexResponses.degenerate(page))
        assertFalse(CodexResponses.degenerate(java.util.Base64.getEncoder().encodeToString(page.toByteArray())))
    }

    @Test fun knowsWhenAStreamedReplyIsComplete() {
        assertFalse(CodexResponses.replyComplete("""{"text":"a}","tool_calls":[{"name":"x","arguments":{}}"""))
        assertTrue(CodexResponses.replyComplete("```json\n{\"text\":\"a\",\"tool_calls\":[]}\n``` and then more chatter {"))
        assertFalse(CodexResponses.replyComplete("<think>{\"text\":1}"))
        assertTrue(CodexResponses.replyComplete("<think>plan {x}</think>{\"text\":\"done\",\"tool_calls\":[]}"))
    }

    @Test(expected = org.json.JSONException::class) fun doesNotSaveAFileCutOffMidContent() {
        CodexResponses.output("""{"text":"","tool_calls":[{"name":"write_file","arguments":{"path":"dog.html","content":"<html><body>""", request)
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
