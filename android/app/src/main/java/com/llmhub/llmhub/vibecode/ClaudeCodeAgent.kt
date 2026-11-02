package com.llmhub.llmhub.vibecode

import android.content.Context
import android.net.Uri
import android.os.Build
import com.llmhub.llmhub.R
import com.llmhub.llmhub.agent.TermuxStreamingCommand
import com.llmhub.llmhub.agent.TerminalOutputBuffer
import kotlinx.coroutines.*
import org.json.JSONObject

/**
 * Claude Code in Termux, answered by the local model through [LocalClaudeServer] (ANTHROPIC_BASE_URL); no Anthropic
 * account. Runs `claude -p` with stream-json output and renders its messages and tool calls like Codex mode.
 */
internal class ClaudeCodeAgent(private val context: Context) {
    companion object {
        /** LLM Hub's own copy, apart from any Claude Code the user installed in Termux themselves. */
        private const val HOME = "/data/data/com.termux/files/home/.llmhub-claude"

        /**
         * Anthropic's official linux-arm64 binary, made runnable on Android the way claude-code-android does it
         * (https://github.com/ferrumclaudepilgrim/claude-code-android, MIT, Ferrum_Flux_Fenice): Termux's glibc
         * runtime, the binary's checksum checked against Anthropic's release manifest, and its ELF interpreter
         * pointed at that glibc. Explicitly invoked from the setup button; re-running it updates Claude Code.
         */
        val INSTALL = """
            set -e
            export PATH=/data/data/com.termux/files/usr/bin:${'$'}PATH TERM=dumb DEBIAN_FRONTEND=noninteractive
            [ "${'$'}(uname -m)" = aarch64 ] || { echo "Claude Code needs a 64-bit (aarch64) device." >&2; exit 1; }
            termux-setup-storage 2>/dev/null || true
            [ -e ${'$'}PREFIX/etc/termux/chosen_mirrors ] || ln -sf ${'$'}PREFIX/etc/termux/mirrors/default ${'$'}PREFIX/etc/termux/chosen_mirrors 2>/dev/null || true
            APT="-y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold"
            apt-get update ${'$'}APT
            apt-get install ${'$'}APT curl jq glibc-repo
            apt-get update ${'$'}APT
            apt-get install ${'$'}APT glibc-runner patchelf-glibc
            V=${'$'}(curl -fsSL --max-time 20 https://registry.npmjs.org/@anthropic-ai/claude-code/latest | jq -r .version)
            echo "${'$'}V" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+${'$'}' || { echo "Could not find the latest Claude Code version." >&2; exit 1; }
            mkdir -p $HOME/config
            R=https://downloads.claude.ai/claude-code-releases/${'$'}V
            echo "Downloading Claude Code ${'$'}V (about 230 MB)"
            curl -fSL --max-time 900 "${'$'}R/linux-arm64/claude" -o $HOME/claude.tmp
            E=${'$'}(curl -fsSL --max-time 20 "${'$'}R/manifest.json" | jq -er '.platforms["linux-arm64"].checksum')
            [ "${'$'}(sha256sum $HOME/claude.tmp | cut -d' ' -f1)" = "${'$'}E" ] || { rm -f $HOME/claude.tmp; echo "Checksum mismatch; nothing was installed." >&2; exit 1; }
            chmod +x $HOME/claude.tmp
            LD_PRELOAD= ${'$'}PREFIX/glibc/bin/patchelf --set-interpreter ${'$'}PREFIX/glibc/lib/ld-linux-aarch64.so.1 $HOME/claude.tmp
            mv $HOME/claude.tmp $HOME/claude
            LD_PRELOAD= CLAUDE_CONFIG_DIR=$HOME/config $HOME/claude --version
        """.trimIndent()

        /** The project folder as a path Termux can open, or null for folders only reachable through Android's picker. */
        fun projectPath(context: Context, folder: String): String? =
            CodexWorkspace.resolveRealPath(context, Uri.parse(folder))?.takeIf { it.startsWith("/storage/") || it.startsWith("/sdcard") }
    }

    suspend fun run(
        prompt: String, folder: String, session: String?,
        editorUri: String?, editorName: String?, editorCode: String, editorDirty: Boolean, contextWindow: Int,
        infer: suspend (prompt: String, grammar: String?, onDelta: (String) -> Unit) -> String,
        onSession: (String) -> Unit,
        onMessage: (CodexActivity) -> Unit,
        /** Normal mode's rule for the file's language, so agent output meets the same bar. */
        target: String? = null
    ) = coroutineScope {
        check(Build.VERSION.SDK_INT >= 29 && Build.SUPPORTED_ABIS.contains("arm64-v8a")) {
            context.getString(R.string.vibe_codex_requirements)
        }
        val path = projectPath(context, folder) ?: throw IllegalStateException(context.getString(R.string.vibe_claude_code_folder_required))
        if (editorDirty && editorUri != null) withContext(Dispatchers.IO) {
            // Claude Code reads the file from disk; unsaved editor text would otherwise be lost under its edits.
            context.contentResolver.openOutputStream(Uri.parse(editorUri), "wt")?.use { it.write(editorCode.toByteArray()) }
        }
        val renderer = ClaudeStreamRenderer()
        val server = LocalClaudeServer(infer, onProgress = { id, step, attempt, raw ->
            onMessage(CodexActivity("progress-$id", context.getString(R.string.vibe_codex_step_progress, step, attempt, raw.length), "status", "running"))
        })
        val terminal = TerminalOutputBuffer()
        try {
            onMessage(CodexActivity("status", context.getString(R.string.vibe_claude_code_starting), "status", "running"))
            val turnPrompt = AgentTurnPrompt.build(prompt, editorName?.let { path.trimEnd('/') + "/" + it }, editorName, editorCode,
                contextWindow, target, writeTool = "Write", editTool = "Edit")
            val encoded = java.util.Base64.getEncoder().encodeToString(turnPrompt.toByteArray())
            val quote = CodexConfig::shellQuote
            fun command(resume: String?) = """
                export PATH=/data/data/com.termux/files/usr/bin:${'$'}PATH
                [ -x $HOME/claude ] || { echo ${quote(context.getString(R.string.vibe_claude_code_install_help))} >&2; exit 1; }
                mkdir -p ${quote(path)} 2>/dev/null
                cd ${quote(path)} || { echo ${quote(context.getString(R.string.vibe_claude_code_folder_required))} >&2; exit 1; }
                printf '%s' ${quote(encoded)} | base64 -d | env -u LD_PRELOAD CLAUDE_CONFIG_DIR=$HOME/config \
                  ANTHROPIC_BASE_URL=${quote(server.baseUrl)} ANTHROPIC_AUTH_TOKEN=llmhub-local \
                  ANTHROPIC_MODEL=llmhub-local ANTHROPIC_SMALL_FAST_MODEL=llmhub-local ANTHROPIC_DEFAULT_HAIKU_MODEL=llmhub-local \
                  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 \
                  API_TIMEOUT_MS=1800000 CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096 MAX_THINKING_TOKENS=0 \
                  $HOME/claude -p --output-format stream-json --verbose --permission-mode bypassPermissions \
                  --allowedTools Read,Write,Edit,Bash,Glob,Grep,LS${resume?.let { " --resume " + quote(it) }.orEmpty()}
            """.trimIndent()
            // A saved session Claude Code no longer has (its data was cleared) would fail every turn: start afresh once.
            for (resume in listOf(session?.takeIf { it.isNotBlank() }, null).distinct()) {
                val lines = StringBuilder()
                var result: JSONObject? = null
                val output = TerminalOutputBuffer()
                val failure = try {
                    TermuxStreamingCommand.run(context, command(resume), 35 * 60_000L) { chunk ->
                        terminal.append(chunk)
                        output.append(chunk)
                        lines.append(chunk)
                        while (true) {
                            val end = lines.indexOf("\n").takeIf { it >= 0 } ?: break
                            val line = lines.substring(0, end).trim()
                            lines.delete(0, end + 1)
                            val event = line.takeIf { it.startsWith("{") }?.let { runCatching { JSONObject(it) }.getOrNull() } ?: continue
                            event.optString("session_id").takeIf { it.isNotEmpty() && event.optString("type") == "system" }?.let(onSession)
                            if (event.optString("type") == "result") result = event
                            renderer.render(event).forEach(onMessage)
                        }
                    }
                    null
                } catch (e: CancellationException) { throw e } catch (e: Exception) { e }
                // Claude Code's own words (install help, a missing folder, an API error) say more than an exit code.
                val said = output.append("").trim().takeLast(2000)
                if (resume != null && said.contains("No conversation found")) { onSession(""); continue }
                val done = result
                if (done != null && !done.optBoolean("is_error") && done.optString("subtype") == "success") return@coroutineScope
                throw IllegalStateException(done?.optString("result")?.takeIf { it.isNotBlank() }
                    ?: said.ifBlank { failure?.message ?: context.getString(R.string.vibe_claude_code_failed) }, failure)
            }
        } finally {
            withContext(NonCancellable) { withTimeoutOrNull(10_000) { server.shutdown() } }
        }
    }
}

/** Claude Code's stream-json events as chat activities: replies, and one terminal card per tool call with its result. */
internal class ClaudeStreamRenderer {
    private val headers = mutableMapOf<String, String>()

    fun render(event: JSONObject): List<CodexActivity> {
        val message = event.optJSONObject("message") ?: return emptyList()
        val content = message.optJSONArray("content") ?: return emptyList()
        return (0 until content.length()).mapNotNull { content.optJSONObject(it) }.mapNotNull { block ->
            when (block.optString("type")) {
                "text" -> block.optString("text").takeIf { it.isNotBlank() && event.optString("type") == "assistant" }
                    ?.let { CodexActivity("text-${message.optString("id")}", it, state = "running") }
                "tool_use" -> {
                    val id = block.optString("id")
                    headers[id] = "$ " + summary(block.optString("name"), block.optJSONObject("input") ?: JSONObject()) + "\n"
                    CodexActivity(id, headers.getValue(id), "terminal", "running")
                }
                "tool_result" -> {
                    val id = block.optString("tool_use_id")
                    val output = block.opt("content").let { value ->
                        if (value is org.json.JSONArray) (0 until value.length()).joinToString("\n") { value.optJSONObject(it)?.optString("text").orEmpty() }
                        else value?.toString().orEmpty()
                    }
                    CodexActivity(id, headers[id].orEmpty() + TerminalOutputBuffer().append(output),
                        "terminal", if (block.optBoolean("is_error")) "failed" else "succeeded")
                }
                else -> null
            }
        }
    }

    private fun summary(name: String, input: JSONObject): String = when (name) {
        "Bash" -> input.optString("command")
        "Write" -> "Write ${input.optString("file_path")} (${input.optString("content").toByteArray().size} bytes)"
        "Edit" -> "Edit ${input.optString("file_path")}"
        "Read" -> "Read ${input.optString("file_path")}"
        else -> "$name ${input.toString().take(200)}"
    }
}
