package com.llmhub.llmhub.vibecode

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.agent.TermuxStreamingCommand
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.net.InetAddress
import java.net.ServerSocket
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class CodexTermuxInstrumentedTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun terminalBytesArriveBeforeTermuxCommandFinishes() = runBlocking {
        val early = CompletableDeferred<Unit>()
        val output = StringBuffer()
        val command = async(Dispatchers.IO) {
            TermuxStreamingCommand.run(context, "printf early; sleep 2; printf 'late\\n' >&2", 20_000) {
                output.append(it)
                if (output.contains("early")) early.complete(Unit)
            }
        }
        withTimeout(10_000) { early.await() }
        assertFalse(command.isCompleted)
        command.await()
        assertEquals("earlylate\n", output.toString())
    }

    @Test fun installedCodexUsesLocalEndpointAndStreamsItsRealCommandOutput() = runCodexScenario(false)

    @Test fun malformedCompletionRecoversAndProducesOneAssistantMessage() = runCodexScenario(true)

    @Test fun emptyEditorFileExistsInTermuxBeforeAgentReadsIt() = runBlocking {
        val local = LocalCodexServer(infer = { "{\"text\":\"ready\",\"tool_calls\":[]}" },
            modelError = "Invalid local test response")
        val port = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1")).use { it.localPort }
        val token = UUID.randomUUID().toString()
        val sha = java.security.MessageDigest.getInstance("SHA-256").digest(token.toByteArray())
            .joinToString("") { "%02x".format(it) }
        val home = "/data/data/com.termux/files/home/.llmhub-codex/empty-file-${UUID.randomUUID()}"
        val project = "/storage/emulated/0/Download/llmhub-empty-file-${UUID.randomUUID()}"
        val logs = StringBuffer()
        val server = async(Dispatchers.IO) {
            TermuxStreamingCommand.run(context,
                "export PATH=/data/data/com.termux/files/usr/bin:\$PATH; mkdir -p ${CodexConfig.shellQuote(home)}; " +
                    "CODEX_HOME=${CodexConfig.shellQuote(home)} codex app-server --listen ws://127.0.0.1:$port " +
                    "--ws-auth capability-token --ws-token-sha256 $sha ${CodexConfig.arguments(local.baseUrl, 8192)}",
                120_000) { logs.append(it) }
        }
        var client: CodexClient? = null
        try {
            for (i in 0 until 40) {
                val candidate = CodexClient()
                try { candidate.connect(port, token); client = candidate; break }
                catch (e: CancellationException) { candidate.close(); throw e }
                catch (_: Exception) { candidate.close(); delay(250) }
            }
            val rpc = checkNotNull(client) { logs.toString() }
            rpc.request("fs/createDirectory", JSONObject().put("path", project))
            val workspace = CodexWorkspace(context, android.net.Uri.parse(project), home)
            val path = "$project/girl.html"
            // Reproduce the clean tab pointing at a file that failed to be created.
            workspace.stage(rpc, android.net.Uri.fromFile(java.io.File(path)).toString(), "girl.html", "", false)
            assertEquals("", rpc.request("fs/readFile", JSONObject().put("path", path)).getString("dataBase64"))
            // A clean empty editor must never erase content subsequently written on disk.
            val saved = android.util.Base64.encodeToString("saved on disk".toByteArray(), android.util.Base64.NO_WRAP)
            rpc.request("fs/writeFile", JSONObject().put("path", path).put("dataBase64", saved))
            workspace.stage(rpc, null, "girl.html", "", false)
            assertEquals(saved, rpc.request("fs/readFile", JSONObject().put("path", path)).getString("dataBase64"))
            // An intentional unsaved deletion does publish an empty file.
            workspace.stage(rpc, null, "girl.html", "", true)
            assertEquals("", rpc.request("fs/readFile", JSONObject().put("path", path)).getString("dataBase64"))
        } finally {
            client?.close()
            withContext(NonCancellable) {
                server.cancelAndJoin(); local.shutdown()
                runCatching { TermuxStreamingCommand.run(context,
                    "rm -rf -- ${CodexConfig.shellQuote(home)} ${CodexConfig.shellQuote(project)}", 10_000) {} }
            }
        }
    }

    private fun runCodexScenario(repairMalformed: Boolean) = runBlocking {
        var request = JSONObject()
        var calls = if (repairMalformed) -1 else 0
        val renderer = CodexActivityRenderer()
        val activities = java.util.concurrent.ConcurrentHashMap<String, CodexActivity>()
        val sharedScratch = "/storage/emulated/0/Download/llmhub-codex-write-${UUID.randomUUID()}"
        val writtenContent = """
            <!DOCTYPE html>
            <html lang="en"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script>const values = ["a,b", ")]"]; document.getElementById('button').textContent = 'It works 😀';</script>
            </html>
        """.trimIndent() + "\n"
        val local = LocalCodexServer(infer = { prompt ->
            calls++
            val step = calls - if (repairMalformed) 1 else 0
            if (step == 0) {
                """{"text":"I've finished updating the project!"]}"""
            } else if (step == 1) {
                if (repairMalformed) check(prompt.contains("were NOT executed"))
                "${CodexResponses.SENTINEL_THINK}Read the file${CodexResponses.SENTINEL_ENDTHINK}" +
                    "<|tool_call_start|>[read_file(path='/cwd/smoke.txt')]<|tool_call_end|>"
            } else if (step == 2 && repairMalformed) {
                val command = "cat << 'EOF_CODE' > smoke.txt\n${writtenContent}EOF_CODE"
                val literal = "'" + command.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"
                "<|tool_call_start|>[exec_command(command=$literal, justification='Update the website')]<|tool_call_end|>"
            } else if (step == 3 && repairMalformed) {
                JSONObject().put("text", "Writing the shared-storage file").put("tool_calls", JSONArray().put(
                    JSONObject().put("name", "write_file").put("arguments", JSONObject()
                        .put("path", "$sharedScratch/nested/file' name.html").put("content", writtenContent)))).toString()
            } else if (step == 2) {
                fun find(tools: JSONArray, prefix: String = ""): String? {
                    for (i in 0 until tools.length()) {
                        val tool = tools.getJSONObject(i)
                        if (tool.optString("type") == "namespace") {
                            find(tool.getJSONArray("tools"), prefix + tool.getString("name") + ".")?.let { return it }
                        } else if (tool.optString("name") in setOf("shell_command", "exec_command")) return prefix + tool.getString("name")
                    }
                    return null
                }
                val name = checkNotNull(find(request.getJSONArray("tools"))) { request.toString() }
                JSONObject().put("text", "").put("tool_calls", JSONArray().put(JSONObject().put("name", name)
                    .put("arguments", JSONObject().put("command", "printf 'codex-early\\n'; sleep 2; printf 'codex-late\\n' >&2")
                        .put("timeout_ms", 10_000).put("login", false)))).toString()
            } else """<think>Final check</think>{"text":"ready","tool_calls":[]}"""
        }, modelError = "Invalid local test response", onRequest = { request = it },
            onProgress = { id, _, _, raw ->
                renderer.thinking(id, CodexResponses.parseThinking(raw).first)?.let { activities[it.key] = it }
            })
        val port = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1")).use { it.localPort }
        val token = UUID.randomUUID().toString()
        val sha = java.security.MessageDigest.getInstance("SHA-256").digest(token.toByteArray()).joinToString("") { "%02x".format(it) }
        val home = "/data/data/com.termux/files/home/.llmhub-codex/smoke-${UUID.randomUUID()}"
        val logs = StringBuffer()
        val server = async(Dispatchers.IO) {
            TermuxStreamingCommand.run(context,
                "export PATH=/data/data/com.termux/files/usr/bin:\$PATH; mkdir -p ${CodexConfig.shellQuote(home)}; " +
                    "CODEX_HOME=${CodexConfig.shellQuote(home)} codex app-server --listen ws://127.0.0.1:$port " +
                    "--ws-auth capability-token --ws-token-sha256 $sha ${CodexConfig.arguments(local.baseUrl, 8192)}", 120_000) { logs.append(it) }
        }
        var client: CodexClient? = null
        try {
            for (i in 0 until 40) {
                val candidate = CodexClient()
                try { candidate.connect(port, token); client = candidate; break }
                catch (e: CancellationException) { candidate.close(); throw e }
                catch (_: Exception) { candidate.close(); delay(250) }
            }
            val rpc = checkNotNull(client) { logs.toString() }
            rpc.request("fs/createDirectory", JSONObject().put("path", "$home/project"))
            rpc.request("fs/writeFile", JSONObject().put("path", "$home/project/smoke.txt")
                .put("dataBase64", android.util.Base64.encodeToString("read-file-passed\n".toByteArray(), android.util.Base64.NO_WRAP)))
            val thread = rpc.request("thread/start", JSONObject().put("model", "llmhub-local")
                .put("modelProvider", "llmhub_local").put("cwd", "$home/project")
                .put("sandbox", "danger-full-access").put("approvalPolicy", if (repairMalformed) "untrusted" else CodexApprovals.POLICY))
                .getJSONObject("thread").getString("id")
            if (repairMalformed) {
                // Codex persists a resumable rollout only after the first turn.
                rpc.request("turn/start", JSONObject().put("threadId", thread).put("input", JSONArray().put(
                    JSONObject().put("type", "text").put("text", "Prepare this test session").put("text_elements", JSONArray()))))
                withTimeout(15_000) {
                    for (event in rpc.events) {
                        if (event.optString("method") == "turn/completed") break
                    }
                }
                activities.clear()
                // Existing chats must also lose the previous interactive approval policy.
                rpc.request("thread/resume", JSONObject().put("threadId", thread)
                    .put("model", "llmhub-local").put("modelProvider", "llmhub_local")
                    .put("cwd", "$home/project").put("sandbox", "danger-full-access")
                    .put("approvalPolicy", CodexApprovals.POLICY))
            }
            rpc.request("turn/start", JSONObject().put("threadId", thread).put("approvalPolicy", CodexApprovals.POLICY).put("input", JSONArray().put(
                JSONObject().put("type", "text").put("text", "Run the smoke command then reply ready").put("text_elements", JSONArray()))))
            var early = false
            val finishedCommands = mutableSetOf<String>()
            var fileRead = false
            var completed = false
            var approvals = 0
            withTimeout(40_000) {
                for (event in rpc.events) {
                    logs.append("\n").append(event.toString())
                    val method = event.optString("method")
                    val params = event.optJSONObject("params") ?: JSONObject()
                    renderer.render(method, params)?.let { activities[it.key] = it }
                    when {
                        event.has("id") && CodexApprovals.response(method, params) != null -> {
                            approvals++
                            rpc.respond(event.get("id"), CodexApprovals.response(method, params)!!)
                        }
                        event.has("id") -> rpc.reject(event.get("id"))
                        method == "item/commandExecution/outputDelta" -> if (params.optString("delta").contains("codex-early")) {
                            assertFalse("Output must precede the completed event", params.optString("itemId") in finishedCommands); early = true
                        }
                        method == "item/completed" && params.optJSONObject("item")?.optString("type") == "commandExecution" -> {
                            val item = params.getJSONObject("item")
                            assertEquals(item.toString() + logs, 0, item.optInt("exitCode", -1))
                            finishedCommands.add(item.getString("id"))
                            if (item.optString("aggregatedOutput").contains("read-file-passed")) fileRead = true
                        }
                        method == "turn/completed" -> {
                            assertEquals(params.toString() + logs, "completed", params.getJSONObject("turn").getString("status"))
                            completed = true; break
                        }
                        method == "error" -> fail(params.toString() + logs)
                    }
                }
            }
            assertTrue("Local provider did not continue after tools", calls >= 3)
            assertTrue("Tagged read_file was not executed: $logs", fileRead)
            if (!repairMalformed) assertTrue("No streamed command output: $logs", early)
            else {
                assertTrue("Malformed response was not retried", calls >= 4)
                val assistant = activities.values.filter { it.role == "assistant" && it.text.endsWith("ready") }
                assertEquals(1, assistant.size)
                assertEquals("<think>Final check</think>\n\nready", assistant.single().text)
                assertFalse(activities.values.any { it.text.contains("I've finished updating") })
                for (path in listOf("$home/project/smoke.txt", "$sharedScratch/nested/file' name.html")) {
                    val result = rpc.request("fs/readFile", JSONObject().put("path", path))
                    val actual = String(android.util.Base64.decode(result.getString("dataBase64"), android.util.Base64.DEFAULT), Charsets.UTF_8)
                    assertEquals("Actual file contents differ at $path", writtenContent, actual)
                }
                val workspace = CodexWorkspace(context, android.net.Uri.parse("$home/project"), home)
                workspace.stage(rpc, null, "smoke.txt", "stale editor snapshot", editorDirty = false)
                val preserved = rpc.request("fs/readFile", JSONObject().put("path", "$home/project/smoke.txt"))
                assertEquals(writtenContent, String(android.util.Base64.decode(preserved.getString("dataBase64"), android.util.Base64.DEFAULT), Charsets.UTF_8))
                workspace.stage(rpc, null, "smoke.txt", "", editorDirty = true)
                assertEquals("", rpc.request("fs/readFile", JSONObject().put("path", "$home/project/smoke.txt")).getString("dataBase64"))
            }
            assertTrue(completed)
            assertEquals("Unattended mode must not request approvals", 0, approvals)
        } finally {
            client?.close()
            withContext(NonCancellable) {
                server.cancelAndJoin(); local.shutdown()
                runCatching { TermuxStreamingCommand.run(context, "rm -rf -- ${CodexConfig.shellQuote(home)} ${CodexConfig.shellQuote(sharedScratch)}", 10_000) {} }
            }
        }
    }
}
