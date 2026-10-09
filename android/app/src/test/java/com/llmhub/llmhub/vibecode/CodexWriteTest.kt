package com.llmhub.llmhub.vibecode

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class CodexWriteTest {
    @Test fun exactHtmlEditWritesUrlsAndQuotesWithoutShellEscaping() {
        val dir = Files.createTempDirectory("codex-exact-edit-").toFile()
        val before = "<!DOCTYPE html>\n<head>\n</head>\n"
        val insertion = "<head>\n<script src=\"https://cdn.tailwindcss.com\"></script>"
        try {
            dir.resolve("index.html").writeText(before)
            val raw = JSONObject().put("name", "replace_in_file").put("arguments", JSONObject()
                .put("path", "index.html").put("old_text", "<head>").put("new_text", insertion)).toString()
            val cmd = command(raw).getString("cmd")
            fun execute(): Int = ProcessBuilder("bash", "-c", cmd).directory(dir).redirectErrorStream(true).start().let {
                it.inputStream.bufferedReader().readText(); it.waitFor()
            }
            assertEquals(0, execute())
            assertEquals(before.replace("<head>", insertion), dir.resolve("index.html").readText())
            dir.resolve("index.html").writeText("No matching fragment")
            assertNotEquals(0, execute())
            assertEquals("No matching fragment", dir.resolve("index.html").readText())
        } finally { dir.deleteRecursively() }
    }
    private val request = JSONObject("""{"tools":[{"type":"function","name":"exec_command","parameters":{"properties":{"cmd":{"type":"string"}},"required":["cmd"]}}]}""")
    private fun python(value: String) = "'" + value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"
    private fun command(raw: String): JSONObject {
        val output = CodexResponses.output(raw, request)
        val call = (0 until output.length()).map { output.getJSONObject(it) }
            .first { it.optString("type") == "function_call" }
        return JSONObject(call.getString("arguments"))
    }
    @Test fun fileVerificationCannotRequestOnlyFiftyTokens() {
        assertEquals(2048, command("""{"name":"exec_command","arguments":{"cmd":"cat index.html","max_output_tokens":50}}""").getInt("max_output_tokens"))
    }

    @Test fun quotedHtmlHeredocSurvivesParsingAndWritesCompleteFile() {
        val content = """
            <!DOCTYPE html>
            <html lang="en"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script>const values = ["a,b", ")]"]; document.getElementById('button').textContent = 'It works';</script>
            </html>
        """.trimIndent() + "\n"
        val script = "cat << 'EOF_CODE' > result.html\n${content}EOF_CODE"
        val raw = "<|tool_call_start|>[exec_command(command=${python(script)}, justification='Update the file')]<|tool_call_end|>"
        val args = command(raw)
        assertEquals(script, args.getString("cmd"))
        assertFalse(args.has("justification"))
        assertFalse(args.has("lang"))
        assertFalse(args.has("charset"))
        val dir = Files.createTempDirectory("codex-write-").toFile()
        try {
            val process = ProcessBuilder("bash", "-c", args.getString("cmd")).directory(dir).redirectErrorStream(true).start()
            val output = process.inputStream.bufferedReader().readText()
            assertEquals(output, 0, process.waitFor())
            assertEquals(content, dir.resolve("result.html").readText())
        } finally { dir.deleteRecursively() }
    }

    @Test fun bracketsAndCommasInsideScriptDoNotEndToolCall() {
        val script = "printf '%s' ')] , name=\"viewport\"' > result.txt"
        assertEquals(script, command("Working: [exec_command(command=${python(script)})]").getString("cmd"))
    }

    @Test fun structuredWritePreservesContentAndLiteralFileNameExactly() {
        val content = "EOF_CODE\n'quoted', \"double\", \\backslash\nUnicode: 😀\nEOF_CODE"
        val raw = JSONObject().put("name", "write_file").put("arguments", JSONObject()
            .put("path", "/cwd/sub dir/file' name.html").put("content", content)).toString()
        val args = command(raw)
        val dir = Files.createTempDirectory("codex-file-").toFile()
        try {
            val process = ProcessBuilder("bash", "-c", args.getString("cmd")).directory(dir).redirectErrorStream(true).start()
            val output = process.inputStream.bufferedReader().readText()
            assertEquals(output, 0, process.waitFor())
            assertEquals(content, dir.resolve("sub dir/file' name.html").readText())
        } finally { dir.deleteRecursively() }
    }

    @Test fun fullAccessCommandsOmitIncompatibleEscalationArguments() {
        val args = command("""{"name":"exec_command","arguments":{"cmd":"cat index.html","justification":"Read","sandbox_permissions":"require_escalated","prefix_rule":["cat"]}}""")
        assertEquals("cat index.html", args.getString("cmd"))
        for (key in listOf("justification", "sandbox_permissions", "prefix_rule")) assertFalse(args.has(key))
    }

    @Test(expected = IllegalArgumentException::class)
    fun malformedQuoteIsRejectedInsteadOfExecutingTruncatedCommand() {
        command("<tool_call>[exec_command(command='cat << \\)]</tool_call>")
    }

    @Test fun tripleQuotedAndNestedArgumentsPreserveScript() {
        val args = CodexToolArguments.parse("command='''line 'one'\nline \"two\"''', options={\"values\":[1,2]}, login=False")
        assertEquals("line 'one'\nline \"two\"", args.getString("command"))
        assertEquals(2, args.getJSONObject("options").getJSONArray("values").length())
        assertFalse(args.getBoolean("login"))
    }
}
