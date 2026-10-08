package com.llmhub.llmhub.vibecode

import com.llmhub.llmhub.agent.TerminalOutputBuffer
import org.json.JSONObject

internal data class CodexActivity(val key: String, val text: String, val role: String = "assistant", val state: String? = null)

/** Keep the command header, streamed bytes, and final exit status in one terminal card. */
internal class CodexActivityRenderer {
    private val commands = mutableMapOf<String, String>()
    private val outputs = mutableMapOf<String, TerminalOutputBuffer>()
    private val text = mutableMapOf<String, String>()
    fun render(method: String, params: JSONObject): CodexActivity? {
        val key = params.optString("itemId")
        return when (method) {
            "item/agentMessage/delta" -> {
                val value = (text[key].orEmpty() + params.optString("delta")).takeLast(100_000)
                text[key] = value
                CodexActivity(key, value)
            }
            "item/commandExecution/outputDelta", "item/fileChange/outputDelta" -> {
                val body = outputs.getOrPut(key) { TerminalOutputBuffer() }.append(params.optString("delta"))
                CodexActivity(key, commands[key].orEmpty() + body, "terminal", "running")
            }
            "item/started", "item/completed" -> {
                val item = params.optJSONObject("item") ?: return null
                val id = item.optString("id")
                val done = method == "item/completed"
                when (item.optString("type")) {
                    "commandExecution" -> {
                        commands[id] = "$ ${item.optString("command")}\n"
                        val buffer = outputs.getOrPut(id) { TerminalOutputBuffer() }
                        val body = if (done && item.has("aggregatedOutput") && !item.isNull("aggregatedOutput")) {
                            TerminalOutputBuffer().append(item.optString("aggregatedOutput"))
                        } else buffer.append("")
                        CodexActivity(id, commands.getValue(id) + body, "terminal",
                            if (!done) "running" else if (item.optString("status") == "failed" || item.optInt("exitCode", 0) != 0) "failed" else "succeeded")
                    }
                    "fileChange" -> CodexActivity(id, item.optJSONArray("changes")?.toString(2).orEmpty(), "terminal",
                        if (!done) "running" else if (item.optString("status") == "failed") "failed" else "succeeded")
                    "agentMessage" -> if (done) CodexActivity(id, item.optString("text")) else null
                    else -> null
                }
            }
            else -> null
        }
    }
}
