package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/** Responses wire adapter. Inference stays in LLM Hub; Codex owns tool execution. */
internal object CodexResponses {
    const val SENTINEL_THINK = "\u200B\u200BTHINK\u200B\u200B"
    const val SENTINEL_ENDTHINK = "\u200B\u200BENDTHINK\u200B\u200B"

    fun prompt(request: JSONObject): String {
        val instructions = request.optString("instructions").trim()
        val toolsArray = request.optJSONArray("tools") ?: JSONArray()
        val conversation = formatConversationInput(request.opt("input"))

        return buildString {
            if (instructions.isNotEmpty()) {
                append("Instructions:\n").append(instructions).append("\n\n")
            }
            append("Available tools:\n").append(toolsArray.toString(2)).append("\n\n")
            if (conversation.isNotEmpty()) {
                append("Conversation (including executed tool results):\n").append(conversation).append("\n\n")
            }
            append("""
                You are an expert coding agent. Follow the instructions and conversation above.
                Return exactly one JSON object, without markdown or thinking tags:
                {"text":"brief reply to user","tool_calls":[{"name":"exact tool name","arguments":{}}]}

                CRITICAL INSTRUCTIONS:
                - All project files are in the current working directory. Always access files by simple relative names (e.g. cat gay.html or cat index.html). Never use old absolute paths from prior chat history.
                - If the user asks to create, modify, inspect, or run anything, you MUST call a tool in tool_calls.
                - NEVER claim that you have finished, created, or edited files without calling tools first.
                - To inspect or read a file, use shell_command with cat:
                  {"name":"shell_command","arguments":{"command":"cat filename.ext"}}
                - To write or create a file, use shell_command with cat:
                  {"name":"shell_command","arguments":{"command":"cat << 'EOF' > filename.ext\n<file content>\nEOF"}}
                - To inspect files or directory contents, use shell_command (e.g. ls -la).
                - Use an empty tool_calls array ([]) ONLY when asking the user a clarifying question or answering a non-coding general question.

                Example of reading a file:
                {"text":"Reading file","tool_calls":[{"name":"shell_command","arguments":{"command":"cat gay.html"}}]}

                Example of creating or editing a file:
                {"text":"Writing index.html","tool_calls":[{"name":"shell_command","arguments":{"command":"cat << 'EOF' > index.html\n<!DOCTYPE html>\n<html>\n<head><title>App</title></head>\n<body><h1>Hello World</h1></body>\n</html>\nEOF"}}]}

                Example of inspecting files:
                {"text":"Checking workspace files","tool_calls":[{"name":"shell_command","arguments":{"command":"ls -la"}}]}

                Example of asking clarification:
                {"text":"Which file would you like me to update?","tool_calls":[]}
            """.trimIndent())
        }
    }

    private fun formatConversationInput(inputObj: Any?): String {
        if (inputObj !is JSONArray) return inputObj?.toString().orEmpty()
        val sb = StringBuilder()
        for (i in 0 until inputObj.length()) {
            val item = inputObj.optJSONObject(i)
            if (item == null) {
                inputObj.opt(i)?.toString()?.let { sb.append(it).append("\n\n") }
                continue
            }
            when (item.optString("type")) {
                "message" -> {
                    val role = item.optString("role").replaceFirstChar { it.uppercase() }
                    val contentArr = item.optJSONArray("content")
                    val text = if (contentArr != null) {
                        (0 until contentArr.length()).mapNotNull { idx ->
                            val c = contentArr.optJSONObject(idx)
                            c?.optString("text")?.ifBlank { c.optString("input_text") }
                        }.joinToString("\n")
                    } else item.optString("content")
                    if (text.isNotBlank()) sb.append("$role: $text\n\n")
                }
                "function_call" -> {
                    val name = item.optString("name")
                    val args = item.opt("arguments")
                    sb.append("Assistant (Tool Call): $name($args)\n\n")
                }
                "function_call_output" -> {
                    val output = item.optString("output")
                    sb.append("Tool Output: $output\n\n")
                }
                else -> {
                    val text = item.optString("text")
                    if (text.isNotBlank()) sb.append("$text\n\n")
                    else sb.append(item.toString()).append("\n\n")
                }
            }
        }
        return sb.toString().trim()
            .replace(Regex("/data/data/com\\.termux/files/home/\\.llmhub-codex/workspaces/[^/'\"\\s]+/[^/'\"\\s]+"), ".")
            .replace(Regex("/data/data/com\\.termux/files/home/\\.llmhub-codex/workspaces/[^/'\"\\s]+"), ".")
    }

    private fun tools(array: JSONArray, prefix: String = ""): Map<String, JSONObject> = buildMap {
        for (i in 0 until array.length()) {
            val tool = array.getJSONObject(i)
            if (tool.optString("type") == "namespace") {
                putAll(tools(tool.optJSONArray("tools") ?: JSONArray(), prefix + tool.getString("name") + "."))
            } else if (tool.has("name")) put(prefix + tool.getString("name"), tool)
        }
    }

    fun stripThinking(raw: String): String {
        var clean = raw
        // 1. Strip LLM-Hub sentinel thinking: \u200B\u200BTHINK\u200B\u200B ... \u200B\u200BENDTHINK\u200B\u200B
        clean = clean.replace(Regex("(?s)\u200B\u200BTHINK\u200B\u200B.*?\u200B\u200BENDTHINK\u200B\u200B"), "")
        // Handle unclosed sentinel thinking (e.g. truncated generation)
        if (clean.contains(SENTINEL_THINK)) {
            val idx = clean.indexOf(SENTINEL_THINK)
            val after = clean.substring(idx)
            val jsonStart = after.indexOf('{')
            clean = if (jsonStart >= 0) clean.substring(0, idx) + after.substring(jsonStart)
                    else clean.substring(0, idx)
        }
        // 2. Strip standard thinking tags: <think>...</think>, <thought>...</thought>, <reasoning>...</reasoning>
        clean = clean.replace(Regex("(?s)<think>.*?</think>"), "")
        clean = clean.replace(Regex("(?s)<thought>.*?</thought>"), "")
        clean = clean.replace(Regex("(?s)<reasoning>.*?</reasoning>"), "")
        clean = clean.replace(Regex("(?s)<\\|thought\\|>.*?<\\|/thought\\|>"), "")
        // Handle unclosed tags
        for (tag in listOf("<think>", "<thought>", "<reasoning>", "<|thought|>")) {
            if (clean.contains(tag)) {
                val idx = clean.indexOf(tag)
                val after = clean.substring(idx)
                val jsonStart = after.indexOf('{')
                clean = if (jsonStart >= 0) clean.substring(0, idx) + after.substring(jsonStart)
                        else clean.substring(0, idx)
            }
        }
        // Strip zero-width spaces
        return clean.replace("\u200B", "").trim()
    }

    private fun extractJson(text: String): String? {
        val trimmed = text.trim()
        // 1. Try markdown code fences first: ```json ... ```
        val fenceMatch = Regex("(?s)```(?:json|JSON)?\\s*(\\{.*?\\})\\s*```").find(trimmed)
        if (fenceMatch != null) return fenceMatch.groupValues[1].trim()

        // 2. If already starts and ends with brace, try parse
        if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
            try { JSONObject(trimmed); return trimmed } catch (_: Exception) {}
        }

        // 3. Find first brace and match balanced depth
        val start = trimmed.indexOf('{')
        if (start < 0) return null
        var depth = 0
        var inString = false
        var escape = false
        for (i in start until trimmed.length) {
            val c = trimmed[i]
            if (escape) { escape = false; continue }
            if (c == '\\') { escape = true; continue }
            if (c == '"') { inString = !inString; continue }
            if (!inString) {
                if (c == '{') depth++
                else if (c == '}') {
                    depth--
                    if (depth == 0) {
                        val candidate = trimmed.substring(start, i + 1)
                        try { JSONObject(candidate); return candidate } catch (_: Exception) {}
                    }
                }
            }
        }
        return null
    }

    fun cleanToolName(name: String): String {
        var clean = name.trim()
            .trim('`', '"', '\'', '[', ']', '(', ')', '{', '}', ' ', ':')
        clean = clean.removePrefix("tool:")
            .removePrefix("tool_")
            .removePrefix("call:")
            .removePrefix("function:")
            .substringBefore('(')
            .trim('`', '"', '\'', '[', ']', '(', ')', '{', '}', ' ', ':')
        return clean
    }

    private fun extractTaggedToolCalls(text: String): Pair<String, JSONArray>? {
        // 1. Explicit tags: <tool_call>...</tool_call> or <|tool_call_start|>...<|tool_call_end|>
        val tagRegex = Regex("(?s)(?:<\\|tool_call_start\\|>|<tool_call>)(.*?)(?:<\\|tool_call_end\\|>|</tool_call>)")
        val tagMatches = tagRegex.findAll(text).toList()
        if (tagMatches.isNotEmpty()) {
            val calls = JSONArray()
            var messageText = text
            for (m in tagMatches) {
                val inner = m.groupValues[1].trim()
                messageText = messageText.replace(m.value, "").trim()
                val parsedCall = parseToolCallString(inner)
                if (parsedCall != null) calls.put(parsedCall)
            }
            if (calls.length() > 0) return messageText to calls
        }

        // 2. Bracket tool calls: [read_file(...)] or [read_file, {...}]
        val bracketRegex = Regex("(?s)\\[([a-zA-Z0-9_.-]+(?:\\(.*?\\)|\\s*,\\s*\\{.*?\\}))\\]")
        val bracketMatches = bracketRegex.findAll(text).toList()
        if (bracketMatches.isNotEmpty()) {
            val calls = JSONArray()
            var messageText = text
            for (m in bracketMatches) {
                val inner = m.groupValues[1].trim()
                val parsedCall = parseToolCallString(inner)
                if (parsedCall != null) {
                    messageText = messageText.replace(m.value, "").trim()
                    calls.put(parsedCall)
                }
            }
            if (calls.length() > 0) return messageText to calls
        }

        return null
    }

    private fun parseToolCallString(raw: String): JSONObject? {
        var str = raw.trim().trim('`')
        if (str.startsWith('{') && str.endsWith('}')) {
            return runCatching { JSONObject(str) }.getOrNull()
        }
        // If wrapped in [ ... ], strip outer brackets first (e.g. [exec_command(command=...)])
        if (str.startsWith('[') && str.endsWith(']')) {
            str = str.substring(1, str.length - 1).trim()
        }
        if (str.startsWith('{') && str.endsWith('}')) {
            return runCatching { JSONObject(str) }.getOrNull()
        }

        // 1. Python style function call: tool_name(key=value, ...)
        val parenStart = str.indexOf('(')
        val parenEnd = str.lastIndexOf(')')
        if (parenStart > 0 && parenEnd > parenStart) {
            val name = cleanToolName(str.substring(0, parenStart).trim())
            val argsStr = str.substring(parenStart + 1, parenEnd).trim()
            val args = parsePythonKwargs(argsStr)
            return JSONObject().put("name", name).put("arguments", args)
        }

        // 2. Comma separated [tool_name, {args}]
        val firstComma = str.indexOf(',')
        if (firstComma > 0) {
            val name = cleanToolName(str.substring(0, firstComma))
            val rest = str.substring(firstComma + 1).trim()
            val args = if (rest.startsWith('{') && rest.endsWith('}')) {
                runCatching { JSONObject(rest) }.getOrNull() ?: JSONObject()
            } else parsePythonKwargs(rest)
            return JSONObject().put("name", name).put("arguments", args)
        }

        return null
    }

    private fun parsePythonKwargs(str: String): JSONObject {
        val obj = JSONObject()
        val trimmed = str.trim()
        if (trimmed.isBlank()) return obj
        val pattern = Regex("([a-zA-Z0-9_]+)\\s*=\\s*(\"[^\"]*\"|'[^']*'|[^,]+)")
        val matches = pattern.findAll(trimmed).toList()
        for (m in matches) {
            val k = m.groupValues[1]
            var v = m.groupValues[2].trim()
            if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith('\'') && v.endsWith('\''))) {
                v = v.substring(1, v.length - 1)
                obj.put(k, v)
            } else if (v.equals("true", ignoreCase = true)) obj.put(k, true)
            else if (v.equals("false", ignoreCase = true)) obj.put(k, false)
            else if (v.toIntOrNull() != null) obj.put(k, v.toInt())
            else obj.put(k, v)
        }
        if (obj.length() == 0 && trimmed.isNotBlank()) {
            val cleanVal = trimmed.trim('"', '\'')
            obj.put("path", cleanVal)
            obj.put("command", cleanVal)
        }
        return obj
    }

    fun resolveTool(name: String, available: Map<String, JSONObject>): Pair<String, JSONObject>? {
        val cleaned = cleanToolName(name)
        if (cleaned.isEmpty()) return available.entries.firstOrNull()?.toPair()

        // 1. Direct match
        available[cleaned]?.let { return cleaned to it }

        // 2. Unqualified match (model gave "shell_command", available has "functions.shell_command")
        for ((k, v) in available) {
            if (cleanToolName(k.substringAfterLast('.')) == cleaned) return k to v
        }

        // 3. Qualified match (model gave "functions.shell_command", available has "shell_command")
        val shortName = cleanToolName(cleaned.substringAfterLast('.'))
        available[shortName]?.let { return shortName to it }

        // 4. Any shell or execution tool in available
        for ((k, v) in available) {
            val s = cleanToolName(k.substringAfterLast('.')).lowercase()
            if (s in setOf("shell_command", "shell", "exec", "terminal", "bash", "command", "run", "execute") ||
                s.contains("shell") || s.contains("exec") || s.contains("cmd") || s.contains("command")) {
                return k to v
            }
        }

        // 5. Case-insensitive match
        for ((k, v) in available) {
            val kShort = cleanToolName(k.substringAfterLast('.'))
            if (k.equals(cleaned, ignoreCase = true) || kShort.equals(shortName, ignoreCase = true)) {
                return k to v
            }
        }

        // 6. Universal fallback: map to the first available non-custom function or first tool
        available.entries.firstOrNull { it.value.optString("type") != "custom" }?.let { return it.toPair() }
        return available.entries.firstOrNull()?.toPair()
    }

    private fun isShellTool(qualified: String): Boolean {
        val last = cleanToolName(qualified.substringAfterLast('.')).lowercase()
        return last in setOf("shell_command", "shell", "exec", "terminal", "bash", "command", "run", "execute") ||
            last.contains("shell") || last.contains("exec") || last.contains("cmd") || last.contains("command")
    }

    private fun normalizeToolCall(call: JSONObject, available: Map<String, JSONObject>): JSONObject {
        val rawName = cleanToolName(call.optString("name").ifBlank { call.optString("tool").ifBlank { call.optString("function") } })
        val (qualified, spec) = requireNotNull(resolveTool(rawName, available)) { "Unknown tool: $rawName" }
        val custom = spec.optString("type") == "custom"
        val item = JSONObject().put("type", if (custom) "custom_tool_call" else "function_call")
            .put("id", "fc_${UUID.randomUUID()}").put("call_id", "call_${UUID.randomUUID()}")
            .put("name", qualified.substringAfterLast('.')).put("status", "completed")
        if ('.' in qualified) item.put("namespace", qualified.substringBeforeLast('.'))

        if (custom) {
            val input = call.optString("input").ifBlank {
                call.opt("arguments")?.toString() ?: ""
            }
            item.put("input", input)
        } else {
            val argsRaw = call.opt("arguments") ?: call.opt("parameters")
            val args = when (argsRaw) {
                is JSONObject -> argsRaw
                is String -> runCatching { JSONObject(argsRaw) }.getOrElse {
                    if (isShellTool(qualified)) JSONObject().put("command", argsRaw) else JSONObject(argsRaw)
                }
                else -> {
                    val inlined = JSONObject()
                    val reserved = setOf("name", "tool", "function", "type", "id", "call_id", "status")
                    for (k in call.keys()) {
                        if (k !in reserved) inlined.put(k, call.get(k))
                    }
                    inlined
                }
            }

            val lowerRaw = cleanToolName(rawName).lowercase()
            if (isShellTool(qualified)) {
                if (lowerRaw in setOf("write_file", "create_file", "edit_file", "save_file", "file_write", "update_file")) {
                    var path = args.optString("path").ifBlank { args.optString("file").ifBlank { args.optString("filename").ifBlank { args.optString("target_file").ifBlank { args.optString("name") } } } }
                    if (path.contains(".llmhub-codex")) path = path.substringAfterLast('/')
                    val content = args.optString("content").ifBlank { args.optString("code").ifBlank { args.optString("text").ifBlank { args.optString("body").ifBlank { args.optString("data") } } } }
                    if (path.isNotBlank()) {
                        val delimiter = "EOF_CODE"
                        args.put("command", "cat << '$delimiter' > \"$path\"\n$content\n$delimiter")
                    }
                } else if (lowerRaw in setOf("read_file", "view_file", "cat", "open_file", "file_read")) {
                    var path = args.optString("path").ifBlank { args.optString("file").ifBlank { args.optString("filename").ifBlank { args.optString("name") } } }
                    if (path.contains(".llmhub-codex")) {
                        path = if (path.endsWith(".html") || path.endsWith(".js") || path.endsWith(".css") || path.endsWith(".txt") || path.contains('.')) {
                            path.substringAfterLast('/')
                        } else ""
                    }
                    if (path.isNotBlank()) args.put("command", "cat \"$path\"")
                    else args.put("command", "ls -la")
                } else if (lowerRaw in setOf("list_files", "list_dir", "ls", "dir", "tree")) {
                    var path = args.optString("path").ifBlank { args.optString("dir").ifBlank { args.optString("directory").ifBlank { args.optString("folder") } } }
                    if (path.contains(".llmhub-codex")) path = ""
                    args.put("command", if (path.isNotBlank()) "ls -la \"$path\"" else "ls -la")
                } else if (lowerRaw in setOf("delete_file", "remove_file", "rm")) {
                    var path = args.optString("path").ifBlank { args.optString("file").ifBlank { args.optString("filename") } }
                    if (path.contains(".llmhub-codex")) path = path.substringAfterLast('/')
                    if (path.isNotBlank()) args.put("command", "rm -f \"$path\"")
                }

                if (!args.has("command")) {
                    val cmd = args.optString("cmd").ifBlank {
                        args.optString("script").ifBlank {
                            args.optString("code").ifBlank {
                                args.optString("command_line").ifBlank {
                                    args.optString("commandLine")
                                }
                            }
                        }
                    }
                    if (cmd.isNotBlank()) args.put("command", cmd)
                }
            }

            val required = spec.optJSONObject("parameters")?.optJSONArray("required") ?: JSONArray()
            for (n in 0 until required.length()) {
                val reqKey = required.getString(n)
                require(args.has(reqKey)) { "Missing tool argument: $reqKey" }
            }
            item.put("arguments", args.toString())
        }
        return item
    }

    fun output(raw: String, request: JSONObject, allowPlainText: Boolean = false): JSONArray {
        val clean = stripThinking(raw)
        val available = tools(request.optJSONArray("tools") ?: JSONArray())
        val result = JSONArray()

        // 1. Tagged tool calls
        val tagged = extractTaggedToolCalls(clean)
        if (tagged != null) {
            val (msgText, calls) = tagged
            if (msgText.isNotBlank()) {
                result.put(JSONObject().put("type", "message")
                    .put("id", "msg_${UUID.randomUUID()}").put("role", "assistant").put("status", "completed")
                    .put("content", JSONArray().put(JSONObject().put("type", "output_text").put("text", msgText)
                        .put("annotations", JSONArray()))))
            }
            for (i in 0 until calls.length()) {
                val item = normalizeToolCall(calls.getJSONObject(i), available)
                result.put(item)
            }
            if (result.length() > 0) return result
        }

        // 2. JSON extraction
        val jsonStr = extractJson(clean)
        if (jsonStr != null) {
            val reply = JSONObject(jsonStr)
            val text = reply.optString("text")
            if (text.isNotBlank()) {
                result.put(JSONObject().put("type", "message")
                    .put("id", "msg_${UUID.randomUUID()}").put("role", "assistant").put("status", "completed")
                    .put("content", JSONArray().put(JSONObject().put("type", "output_text").put("text", text)
                        .put("annotations", JSONArray()))))
            }

            val callsList = mutableListOf<JSONObject>()
            if (reply.has("tool_calls")) {
                val arr = reply.optJSONArray("tool_calls")
                if (arr != null) {
                    for (i in 0 until arr.length()) arr.optJSONObject(i)?.let { callsList.add(it) }
                }
            } else if (reply.has("name") && (reply.has("arguments") || reply.has("input") || reply.has("command") || reply.has("cmd"))) {
                callsList.add(reply)
            } else if (reply.has("tool") || reply.has("action")) {
                val name = reply.optString("tool").ifBlank { reply.optString("action") }
                val args = reply.optJSONObject("parameters") ?: reply.optJSONObject("action_input") ?: JSONObject()
                callsList.add(JSONObject().put("name", name).put("arguments", args))
            } else if (reply.has("function_call")) {
                reply.optJSONObject("function_call")?.let { callsList.add(it) }
            }

            for (call in callsList) {
                val item = normalizeToolCall(call, available)
                result.put(item)
            }

            if (result.length() > 0) return result
        }

        // 3. Fallback to plain text message if allowed
        if (allowPlainText && clean.isNotBlank()) {
            result.put(JSONObject().put("type", "message")
                .put("id", "msg_${UUID.randomUUID()}").put("role", "assistant").put("status", "completed")
                .put("content", JSONArray().put(JSONObject().put("type", "output_text").put("text", clean)
                    .put("annotations", JSONArray()))))
            return result
        }

        throw org.json.JSONException("Malformed local model response: no valid tool call or JSON found in: $clean")
    }

    /** Decode only a leading assistant text field; never execute partially generated tools. */
    fun partialText(raw: String): String {
        // While still actively in a thinking block, don't preview as text
        if (raw.contains(SENTINEL_THINK) && !raw.contains(SENTINEL_ENDTHINK)) return ""
        if (raw.contains("<think>") && !raw.contains("</think>")) return ""
        if (raw.contains("<thought>") && !raw.contains("</thought>")) return ""
        if (raw.contains("<reasoning>") && !raw.contains("</reasoning>")) return ""

        var clean = stripThinking(raw)
        clean = clean.trimStart().removePrefix("```json").removePrefix("```").trimStart()

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

    fun parseThinking(content: String): Pair<String, String> {
        if (content.contains(SENTINEL_THINK)) {
            val after = content.substringAfter(SENTINEL_THINK)
            if (after.contains(SENTINEL_ENDTHINK)) {
                return after.substringBefore(SENTINEL_ENDTHINK).trim() to after.substringAfter(SENTINEL_ENDTHINK).trim()
            }
            return after.trim() to ""
        }
        if (content.contains("<think>")) {
            val after = content.substringAfter("<think>")
            if (after.contains("</think>")) {
                return after.substringBefore("</think>").trim() to after.substringAfter("</think>").trim()
            }
            return after.trim() to ""
        }
        if (content.contains("<thought>")) {
            val after = content.substringAfter("<thought>")
            if (after.contains("</thought>")) {
                return after.substringBefore("</thought>").trim() to after.substringAfter("</thought>").trim()
            }
            return after.trim() to ""
        }
        return "" to content
    }

    fun formatDisplayMessage(raw: String): String {
        val (thinking, answer) = parseThinking(raw)
        val cleanAnswer = answer.trim()
        val jsonStr = extractJson(cleanAnswer)
        val displayText = if (jsonStr != null) {
            val obj = runCatching { JSONObject(jsonStr) }.getOrNull()
            val text = obj?.optString("text").orEmpty()
            if (text.isNotBlank()) text else {
                val calls = obj?.optJSONArray("tool_calls")
                if (calls != null && calls.length() > 0) {
                    val firstName = cleanToolName(calls.optJSONObject(0)?.optString("name") ?: "tool")
                    "Calling $firstName..."
                } else ""
            }
        } else {
            val partial = partialText(cleanAnswer)
            if (partial.isNotBlank()) partial
            else if (cleanAnswer.startsWith('{')) ""
            else cleanAnswer.removePrefix("```json").removePrefix("```").trim()
        }

        return buildString {
            if (thinking.isNotBlank()) {
                append("<think>").append(thinking).append("</think>\n\n")
            }
            append(displayText)
        }.trim()
    }

    fun event(type: String, vararg fields: Pair<String, Any>): JSONObject =
        JSONObject().put("type", type).apply { fields.forEach { put(it.first, it.second) } }
}
