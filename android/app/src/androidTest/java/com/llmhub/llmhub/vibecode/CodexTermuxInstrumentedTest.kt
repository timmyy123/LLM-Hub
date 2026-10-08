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

    @Test fun installedCodexUsesLocalEndpointAndStreamsItsRealCommandOutput() = runBlocking {
        var request = JSONObject()
        var calls = 0
        val local = LocalCodexServer(infer = {
            calls++
            if (calls == 1) {
                fun find(tools: JSONArray, prefix: String = ""): String? {
                    for (i in 0 until tools.length()) {
                        val tool = tools.getJSONObject(i)
                        if (tool.optString("type") == "namespace") {
                            find(tool.getJSONArray("tools"), prefix + tool.getString("name") + ".")?.let { return it }
                        } else if (tool.optString("name") == "shell_command") return prefix + "shell_command"
                    }
                    return null
                }
                val name = checkNotNull(find(request.getJSONArray("tools"))) { request.toString() }
                JSONObject().put("text", "").put("tool_calls", JSONArray().put(JSONObject().put("name", name)
                    .put("arguments", JSONObject().put("command", "printf 'codex-early\\n'; sleep 2; printf 'codex-late\\n' >&2")
                        .put("timeout_ms", 10_000).put("login", false)))).toString()
            } else """{"text":"ready","tool_calls":[]}"""
        }, modelError = "Invalid local test response", onRequest = { request = it })
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
            val thread = rpc.request("thread/start", JSONObject().put("model", "llmhub-local")
                .put("modelProvider", "llmhub_local").put("cwd", "$home/project")
                .put("sandbox", "danger-full-access").put("approvalPolicy", "untrusted"))
                .getJSONObject("thread").getString("id")
            rpc.request("turn/start", JSONObject().put("threadId", thread).put("input", JSONArray().put(
                JSONObject().put("type", "text").put("text", "Run the smoke command then reply ready").put("text_elements", JSONArray()))))
            var early = false
            var commandFinished = false
            var completed = false
            withTimeout(40_000) {
                for (event in rpc.events) {
                    logs.append("\n").append(event.toString())
                    val method = event.optString("method")
                    val params = event.optJSONObject("params") ?: JSONObject()
                    when {
                        event.has("id") && method.endsWith("requestApproval") -> rpc.respond(event.get("id"), JSONObject().put("decision", "accept"))
                        event.has("id") -> rpc.reject(event.get("id"))
                        method == "item/commandExecution/outputDelta" -> if (params.optString("delta").contains("codex-early")) {
                            assertFalse("Output must precede the completed event", commandFinished); early = true
                        }
                        method == "item/completed" && params.optJSONObject("item")?.optString("type") == "commandExecution" -> commandFinished = true
                        method == "turn/completed" -> {
                            assertEquals(params.toString() + logs, "completed", params.getJSONObject("turn").getString("status"))
                            completed = true; break
                        }
                        method == "error" -> fail(params.toString() + logs)
                    }
                }
            }
            assertTrue("Local provider was not called", calls >= 2)
            assertTrue("No streamed command output: $logs", early)
            assertTrue(completed)
        } finally {
            client?.close()
            withContext(NonCancellable) {
                server.cancelAndJoin(); local.shutdown()
                runCatching { TermuxStreamingCommand.run(context, "rm -rf -- ${CodexConfig.shellQuote(home)}", 10_000) {} }
            }
        }
    }
}
