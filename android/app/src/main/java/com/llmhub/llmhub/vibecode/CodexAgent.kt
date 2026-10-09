package com.llmhub.llmhub.vibecode

import android.content.Context
import android.net.Uri
import android.os.Build
import com.llmhub.llmhub.R
import com.llmhub.llmhub.agent.TermuxStreamingCommand
import com.llmhub.llmhub.agent.TerminalOutputBuffer
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.net.InetAddress
import java.net.ServerSocket
import java.util.UUID

internal class CodexAgent(private val context: Context) {
    companion object {
        const val VERSION = "0.160.0-termux.3"
        // Explicitly invoked from the setup button; never run merely by enabling the toggle.
        const val INSTALL = "export PATH=/data/data/com.termux/files/usr/bin:\$PATH TERM=dumb npm_config_progress=false; " +
            "termux-setup-storage 2>/dev/null || true; " +
            "pkg update -y && pkg install nodejs-lts -y && npm install -g @mmmbuto/codex-cli-termux@$VERSION --allow-scripts=@mmmbuto/codex-cli-termux && codex --version"
        fun quote(value: String) = CodexConfig.shellQuote(value)
    }

    suspend fun run(
        prompt: String, folder: String, session: String, thread: String?,
        editorUri: String?, editorName: String?, editorCode: String, contextWindow: Int,
        infer: suspend (String, (String) -> Unit) -> String,
        onThread: (String) -> Unit,
        onMessage: (CodexActivity) -> Unit,
        editorDirty: Boolean = false
    ): Map<String, Uri> = coroutineScope {
        check(Build.VERSION.SDK_INT >= 29 && Build.SUPPORTED_ABIS.contains("arm64-v8a")) {
            context.getString(R.string.vibe_codex_requirements)
        }
        val port = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1")).use { it.localPort }
        val token = UUID.randomUUID().toString()
        val runId = UUID.randomUUID().toString()
        val home = "/data/data/com.termux/files/home/.llmhub-codex"
        val workspace = CodexWorkspace(context, Uri.parse(folder), home)
        val renderer = CodexActivityRenderer()
        val server = LocalCodexServer({ infer(it) {} }, context.getString(R.string.vibe_codex_model_error), streamInfer = infer,
            onProgress = { id, step, attempt, raw ->
                onMessage(CodexActivity("progress-$id", context.getString(R.string.vibe_codex_step_progress,
                    step, attempt, raw.length), "status", "running"))
                renderer.thinking(id, CodexResponses.parseThinking(raw).first)?.let(onMessage)
            })
        var client: CodexClient? = null
        var serverJob: Job? = null
        val serverOutput = TerminalOutputBuffer()
        fun status(resource: Int) = onMessage(CodexActivity("status", context.getString(resource), "status", "running"))
        try {
            val config = CodexConfig.arguments(server.baseUrl, contextWindow)
            status(R.string.vibe_codex_starting)
            val command = """
                export PATH=/data/data/com.termux/files/usr/bin:${'$'}PATH
                command -v codex >/dev/null || { echo ${quote(context.getString(R.string.vibe_codex_install_help))} >&2; exit 1; }
                [ -d /storage/emulated/0 ] || termux-setup-storage 2>/dev/null || true
                umask 022
                mkdir -p ${quote(home)}
                CODEX_HOME=${quote(home)} codex app-server --listen ws://127.0.0.1:$port --ws-auth capability-token --ws-token-sha256 ${quote(java.security.MessageDigest.getInstance("SHA-256").digest(token.toByteArray()).joinToString("") { "%02x".format(it) })} $config
            """.trimIndent()
            onMessage(CodexActivity("codex-server", "$ codex app-server --listen ws://127.0.0.1:$port\n", "terminal", "running"))
            serverJob = launch(Dispatchers.IO) {
                TermuxStreamingCommand.run(context, command, 35 * 60_000L) { chunk ->
                    onMessage(CodexActivity("codex-server", "$ codex app-server --listen ws://127.0.0.1:$port\n" + serverOutput.append(chunk), "terminal", "running"))
                }
            }
            status(R.string.vibe_codex_connecting)
            var lastFailure: Exception? = null
            for (attempt in 0 until 30) {
                currentCoroutineContext().ensureActive()
                val candidate = CodexClient()
                try { candidate.connect(port, token); client = candidate; break }
                catch (e: CancellationException) { candidate.close(); throw e }
                catch (e: Exception) { candidate.close(); lastFailure = e; delay(500) }
            }
            val rpc = client ?: throw IllegalStateException(context.getString(R.string.vibe_codex_connection_error), lastFailure)
            status(R.string.vibe_codex_staging)
            workspace.stage(rpc, editorUri, editorName, editorCode, editorDirty)
            if (workspace.isDirect) {
                onMessage(CodexActivity("workspace-mode", "Working directly in folder: ${workspace.remote}", "status", "running"))
            } else {
                onMessage(CodexActivity("workspace-mode", "Working in staged workspace: ${workspace.remote}", "status", "running"))
            }
            val threadParams = JSONObject().put("model", "llmhub-local").put("modelProvider", "llmhub_local")
                .put("cwd", workspace.remote).put("approvalPolicy", CodexApprovals.POLICY)
                .put("baseInstructions", "You are a coding agent working in the current project. " +
                    "Read relevant files before editing. Use the supplied tools to edit files and run commands. " +
                    "Work only in the project directory. Follow AGENTS.md instructions. " +
                    "Project commands and file edits are pre-authorized. Execute them without asking for permission. " +
                    "Run relevant checks and use their output to fix failures. " +
                    "Do not claim changes or successful tests without tool evidence. " +
                    "Finish with a concise account of changes and validation.")
                // Android cannot provide Codex's desktop Linux sandbox. Termux's Android UID is the boundary.
                .put("sandbox", "danger-full-access")
            val resumed = if (thread != null) rpc.request("thread/resume", threadParams.put("threadId", thread))
                else rpc.request("thread/start", threadParams)
            val threadId = resumed.getJSONObject("thread").getString("id")
            onThread(threadId)
            status(R.string.vibe_codex_generating)
            val turnPrompt = buildString {
                if (!editorName.isNullOrBlank()) {
                    append("Active editor file: ").append(editorName).append("\n")
                    append("Read the current file from disk using tools. Its contents may have changed since earlier turns.\n")
                    append("\nUser request:\n")
                }
                append(prompt)
            }
            val turn = rpc.request("turn/start", JSONObject().put("threadId", threadId).put("cwd", workspace.remote)
                .put("approvalPolicy", CodexApprovals.POLICY)
                .put("input", JSONArray().put(JSONObject().put("type", "text").put("text", turnPrompt)
                    .put("text_elements", JSONArray())))).getJSONObject("turn").getString("id")
            var completedSuccessfully = false
            try {
                withTimeout(30 * 60_000L) {
                    for (event in rpc.events) {
                        val method = event.optString("method")
                        val params = event.optJSONObject("params") ?: JSONObject()
                        if (params.has("threadId") && params.optString("threadId") != threadId) continue
                        if (event.has("id")) {
                            val accepted = CodexApprovals.response(method, params)
                            if (accepted != null) {
                                android.util.Log.d("CodexAgent", "Auto-approved $method: $params")
                                rpc.respond(event.get("id"), accepted)
                            } else rpc.reject(event.get("id"))
                        } else when (method) {
                            "item/agentMessage/delta", "item/commandExecution/outputDelta", "item/fileChange/outputDelta",
                            "item/started", "item/completed" -> renderer.render(method, params)?.let(onMessage)
                            "error" -> if (!params.optBoolean("willRetry")) throw IllegalStateException(
                                params.optJSONObject("error")?.optString("message") ?: context.getString(R.string.vibe_codex_connection_error))
                            "turn/completed" -> {
                                val completed = params.getJSONObject("turn")
                                if (completed.getString("id") != turn) continue
                                check(completed.optString("status") == "completed") {
                                    completed.optJSONObject("error")?.optString("message") ?: context.getString(R.string.vibe_codex_connection_error)
                                }
                                completedSuccessfully = true
                                break
                            }
                        }
                    }
                }
            } catch (e: CancellationException) {
                withContext(NonCancellable) {
                    runCatching { withTimeout(3000) { rpc.request("turn/interrupt", JSONObject().put("threadId", threadId).put("turnId", turn)) } }
                }
                throw e
            }
            check(completedSuccessfully) { context.getString(R.string.vibe_codex_connection_error) }
            status(R.string.vibe_codex_syncing)
            workspace.publish(rpc)
        } finally {
            client?.close()
            withContext(NonCancellable) { withTimeoutOrNull(10_000) { server.shutdown() } }
            withContext(NonCancellable) { serverJob?.cancelAndJoin() }
            onMessage(CodexActivity("codex-server", "$ codex app-server --listen ws://127.0.0.1:$port\n" + serverOutput.append(""), "terminal", "stopped"))
        }
    }
}
