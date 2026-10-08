package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/** Responses wire adapter. Inference stays in LLM Hub; Codex owns tool execution. */
internal object CodexResponses {
    fun prompt(request: JSONObject): String = """
        You are the model for a coding agent. Follow the instructions and conversation below.
        Return exactly one JSON object, without markdown or thinking tags:
        {"text":"reply to the user","tool_calls":[{"name":"exact tool name","arguments":{}}]}
        To act, use tool_calls with the supplied tool names and JSON parameters. Do not claim
        to have edited or executed anything without calling a tool. For a custom tool, use
        {"name":"exact tool name","input":"raw tool input"} instead of arguments.
        Namespace tools use the qualified name namespace.tool. Tool results are in input.
        Use an empty tool_calls array only when finished or asking the user a question.
        Instructions:
        ${request.optString("instructions")}
        Available tools:
        ${request.optJSONArray("tools") ?: JSONArray()}
        Conversation (including executed tool results):
        ${request.opt("input") ?: ""}
    """.trimIndent()

    private fun tools(array: JSONArray, prefix: String = ""): Map<String, JSONObject> = buildMap {
        for (i in 0 until array.length()) {
            val tool = array.getJSONObject(i)
            if (tool.optString("type") == "namespace") {
                putAll(tools(tool.optJSONArray("tools") ?: JSONArray(), prefix + tool.getString("name") + "."))
            } else if (tool.has("name")) put(prefix + tool.getString("name"), tool)
        }
    }

    fun output(raw: String, request: JSONObject): JSONArray {
        val clean = raw.replace(Regex("(?s)<think>.*?</think>"), "").trim()
            .removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        val reply = JSONObject(clean) // Malformed tool calls must fail, never become pretend edits.
        val calls = reply.getJSONArray("tool_calls")
        val available = tools(request.optJSONArray("tools") ?: JSONArray())
        val result = JSONArray()
        val text = reply.optString("text")
        if (text.isNotBlank()) result.put(JSONObject().put("type", "message")
            .put("id", "msg_${UUID.randomUUID()}").put("role", "assistant").put("status", "completed")
            .put("content", JSONArray().put(JSONObject().put("type", "output_text").put("text", text)
                .put("annotations", JSONArray()))))
        for (i in 0 until calls.length()) {
            val call = calls.getJSONObject(i)
            val qualified = call.getString("name")
            val spec = requireNotNull(available[qualified]) { "Unknown tool: $qualified" }
            val custom = spec.optString("type") == "custom"
            val item = JSONObject().put("type", if (custom) "custom_tool_call" else "function_call")
                .put("id", "fc_${UUID.randomUUID()}").put("call_id", "call_${UUID.randomUUID()}")
                .put("name", qualified.substringAfterLast('.')).put("status", "completed")
            if ('.' in qualified) item.put("namespace", qualified.substringBeforeLast('.'))
            if (custom) item.put("input", call.getString("input"))
            else {
                val args = call.get("arguments")
                val parsed = if (args is JSONObject) args else JSONObject(args.toString())
                val required = spec.optJSONObject("parameters")?.optJSONArray("required") ?: JSONArray()
                for (n in 0 until required.length()) require(parsed.has(required.getString(n))) {
                    "Missing tool argument: ${required.getString(n)}"
                }
                item.put("arguments", parsed.toString())
            }
            result.put(item)
        }
        require(result.length() > 0) { "Empty model response" }
        return result
    }

    /** Decode only a leading assistant text field; never execute partially generated tools. */
    fun partialText(raw: String): String {
        var clean = raw.trimStart().removePrefix("```json").removePrefix("```").trimStart()
        if (clean.startsWith("<think>")) {
            if (!clean.contains("</think>")) return ""
            clean = clean.substringAfter("</think>").trimStart()
        }
        val start = Regex("^\\{\\s*\"text\"\\s*:\\s*\"").find(clean)?.range?.last?.plus(1) ?: return ""
        val result = StringBuilder()
        var i = start
        while (i < clean.length) {
            val c = clean[i++]
            if (c == '"') break
            if (c != '\\') { result.append(c); continue }
            if (i == clean.length) break
            when (val escaped = clean[i++]) {
                '"', '\\', '/' -> result.append(escaped)
                'n' -> result.append('\n')
                'r' -> result.append('\r')
                't' -> result.append('\t')
                'b' -> result.append('\b')
                'f' -> result.append('\u000C')
                'u' -> {
                    if (i + 4 > clean.length) break
                    val value = clean.substring(i, i + 4).toIntOrNull(16) ?: return ""
                    result.append(value.toChar()); i += 4
                }
                else -> return ""
            }
        }
        if (result.isNotEmpty() && result.last().isHighSurrogate()) result.setLength(result.length - 1)
        return result.toString()
    }

    fun event(type: String, vararg fields: Pair<String, Any>): JSONObject =
        JSONObject().put("type", type).apply { fields.forEach { put(it.first, it.second) } }
}
