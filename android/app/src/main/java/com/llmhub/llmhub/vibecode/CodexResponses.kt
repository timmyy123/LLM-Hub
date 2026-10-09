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
        val conversation = formatConversationInput(request.opt("input"), CodexProgressGuard.supersededResults(request))

        return buildString {
            // The inference backend parses lowercase role labels into the model's chat template.
            // Tool results must be subsequent turns, not a transcript inside the original request.
            append("system: You are a coding agent continuing a tool-driven conversation.\n")
            if (instructions.isNotEmpty()) {
                append("Instructions:\n").append(instructions).append("\n\n")
            }
            val running = CodexProgressGuard.latestResult(request)?.second?.contains("Process running with session ID") == true
            append("Available tools:\n").append(toolManifest(toolsArray, running).toString()).append("\n\n")
            tools(toolsArray).entries.firstOrNull { isShellTool(it.key) }?.let { (name, _) ->
                append("Use ").append(name).append(" for reading files and running checks. Use replace_in_file for exact edits and write_file for complete files. Example:\n")
                append(JSONObject().put("text", "Applying a targeted edit").put("tool_calls", JSONArray().put(JSONObject()
                    .put("name", "replace_in_file").put("arguments", JSONObject().put("path", "path/to/file")
                        .put("old_text", "exact existing text").put("new_text", "replacement text"))))).append("\n")
                append("write_stdin only sends bytes to an already running process. It does not edit files. A Chunk ID is not a session ID.\n\n")
            }
            if (conversation.isNotEmpty()) {
                append(conversation).append("\n\n")
            }
            append("user: Continue after the executed tool results above.\n")
            append("""
                Act on the latest user request using the tools above. Tool results are real executed results.
                Return one JSON object:
                {"text":"brief progress or final reply","tool_calls":[{"name":"exact tool name","arguments":{}}]}
                For custom tools, use "input" containing the raw tool input instead of "arguments".
                Commands and edits are already authorized; execute them without asking permission.
                Read a relevant file once, then edit it using its returned contents. Do not keep rereading an unchanged file.
                The latest successful read is the current file. Older snapshots may contain bugs that have already been fixed.
                If a command failed, change the command to address its actual error.
                For HTML, URLs, quotes and multiline changes, use replace_in_file or write_file with literal text. Do not put them in sed substitutions or shell quoting.
                Fix the user's requested change, not unrelated issues from earlier assistant reasoning.
                Use relative paths in the current project. Use the exact argument names in the tool schema.
                For a small typo, perform a targeted replacement rather than rewriting the entire file.
                After editing, run a check to verify the change. After successful verification, finish with tool_calls: [].
                You can instead call finish(summary="factual result") to end the turn. Do not call a read tool to finish.
                When adding a framework, verify its dependency or script/import is actually present. Custom CSS and utility class names alone do not load a CSS framework.
                An empty tool_calls list is also valid for a general question or a necessary clarification.
                Never claim that a file changed without successful tool evidence.
                Keep reasoning brief. Emit the next tool call instead of repeatedly describing a plan.
            """.trimIndent())
            if (conversation.contains("tailwind", ignoreCase = true)) {
                append("\nTailwind reference for standalone HTML previews without a build pipeline: ")
                append("load <script src=\"https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4\"></script> in <head>, then apply utility classes. ")
                append("Normal <style> tags do not compile @tailwind or @apply directives. Do not replace working CSS with uncompiled directives. ")
                append("Existing build-based projects should keep their installed version and produce compiled CSS. ")
                append("Reference: https://tailwindcss.com/docs/installation/play-cdn\n")
            }
            CodexProgressGuard.latestResult(request)?.let { (call, result) ->
                append("\n\nMost recent executed tool: ").append(executedCall(call)).append("\n")
                append("Most recent tool result (use this to choose the NEXT action):\n").append(result)
                append("\nThe tool above already ran. Use its result; do not restart the same inspection.\n")
            }

        }
    }

    private fun toolManifest(array: JSONArray, running: Boolean): JSONArray = JSONArray().apply {
        for ((name, spec) in tools(array)) {
            if (name.substringAfterLast('.') in setOf("get_goal", "create_goal", "update_goal")) continue
            if (name.substringAfterLast('.') == "write_stdin" && !running) continue
            put(JSONObject().put("name", name).put("type", spec.optString("type"))
                .put("description", spec.optString("description").take(320)).apply {
                    spec.optJSONObject("parameters")?.let { parameters ->
                        val compact = JSONObject(parameters.toString())
                        compact.optJSONObject("properties")?.let { properties ->
                            for (key in properties.keys()) properties.optJSONObject(key)?.remove("description")
                        }
                        put("parameters", compact)
                    }
                })
        }
        if (tools(array).keys.any(::isShellTool)) {
            for ((name, fields) in listOf("replace_in_file" to listOf("path", "old_text", "new_text"),
                "write_file" to listOf("path", "content"))) {
                val properties = JSONObject()
                fields.forEach { properties.put(it, JSONObject().put("type", "string")) }
                put(JSONObject().put("name", name).put("type", "function")
                    .put("description", if (name == "replace_in_file") "Replace one unique fragment in a UTF-8 file. Copy old_text from the returned file. Indentation-only differences are accepted for HTML and brace-based code. Writes the actual file."
                        else "Write complete literal UTF-8 content to the actual file, creating parent directories.")
                    .put("parameters", JSONObject().put("type", "object").put("properties", properties).put("required", JSONArray(fields))))
            }
        }
        put(JSONObject().put("name", "finish").put("type", "function")
            .put("description", "End the agent turn with a factual summary after successful edits and verification. Does not execute a command.")
            .put("parameters", JSONObject().put("type", "object")
                .put("properties", JSONObject().put("summary", JSONObject().put("type", "string")))
                .put("required", JSONArray().put("summary"))))
    }

    fun recoveryPrompt(request: JSONObject, error: Exception): String {
        val input = request.optJSONArray("input") ?: JSONArray().put(JSONObject().put("type", "message")
            .put("role", "user").put("content", request.optString("input")))
        val condensed = JSONArray()
        val latestId = CodexProgressGuard.latestResult(request)?.first?.optString("call_id")
        val superseded = CodexProgressGuard.supersededResults(request)
        val inspections = mutableSetOf<String>()
        val retained = mutableSetOf<String>()
        var lastSuccessfulAction: String? = null
        latestId?.let(retained::add)
        for (i in 0 until input.length()) {
            val item = input.optJSONObject(i) ?: continue
            if (item.optString("type") == "function_call" && CodexProgressGuard.inspectionKey(item) != null) {
                inspections.add(item.optString("call_id"))
            }
        }
        for (i in 0 until input.length()) {
            val item = input.optJSONObject(i) ?: continue
            val id = item.optString("call_id")
            if (item.optString("type") == "function_call_output" && id in inspections && id !in superseded &&
                item.optString("output").contains("Process exited with code 0")) retained.add(id)
            if (item.optString("type") == "function_call_output" && id !in inspections &&
                item.optString("output").contains("Process exited with code 0")) lastSuccessfulAction = id
        }
        lastSuccessfulAction?.let(retained::add)
        for (i in 0 until input.length()) {
            val original = input.optJSONObject(i) ?: continue
            val item = JSONObject(original.toString())
            when (item.optString("type")) {
                "function_call_output", "custom_tool_call_output", "function_call", "custom_tool_call" ->
                    if (item.optString("call_id") !in retained) continue
                "message" -> if (item.optString("role") == "assistant") continue
            }
            condensed.put(item)
        }
        val recovery = JSONObject(request.toString()).put("input", condensed)
        return prompt(recovery) + "\n\nRecovery instruction: ${error.message?.take(2000)}\n" +
            "The earlier commands and edits listed above already executed. You are continuing after them. " +
            "Use the latest result to determine what remains. If it verifies the requested change, finish now with " +
            "{\"text\":\"concise factual result\",\"tool_calls\":[]}. Otherwise call a different necessary tool. " +
            "Your rejected tool calls were NOT executed. Return complete valid JSON with text and tool_calls."
    }

    private fun formatConversationInput(inputObj: Any?, superseded: Set<String> = emptySet()): String {
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
                    // Executed results are evidence; earlier model plans and success claims are not.
                    if (item.optString("role") == "assistant") continue
                    val role = when (item.optString("role")) {
                        "assistant" -> "assistant"
                        "developer", "system" -> "system"
                        else -> "user"
                    }
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
                    sb.append("assistant: Tool call already submitted: ${executedCall(item)}\n\n")
                }
                "function_call_output" -> {
                    val output = if (item.optString("call_id") in superseded)
                        "Earlier inspection superseded by a successful edit or later read. Use the current file contents, not this old snapshot."
                    else item.optString("output")
                    sb.append("user: Executed tool result: $output\n\n")
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

    private fun executedCall(call: JSONObject): String = CodexProgressGuard.command(call)?.let(CodexFileEdits::commandSummary)
        ?: "${call.optString("name")}(${call.optString("arguments")})"

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
            val actionStart = responseStart(after)
            clean = if (actionStart >= 0) clean.substring(0, idx) + after.substring(actionStart)
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
                val actionStart = responseStart(after)
                clean = if (actionStart >= 0) clean.substring(0, idx) + after.substring(actionStart)
                        else clean.substring(0, idx)
            }
        }
        // Strip zero-width spaces
        return clean.replace("\u200B", "").trim()
    }

    private fun responseStart(text: String): Int = listOf(text.indexOf('{'), text.indexOf("<|tool_call_start|>"),
        text.indexOf("<tool_call>")).filter { it >= 0 }.minOrNull() ?: -1

    private fun openThinking(after: String): Pair<String, String> {
        val json = Regex("\\{\\s*\"(?:text|tool_calls)\"\\s*:").find(after)?.range?.first ?: -1
        val marker = listOf(after.indexOf("<|tool_call_start|>"), after.indexOf("<tool_call>"), json)
            .filter { it >= 0 }.minOrNull() ?: return after.trim() to ""
        return after.substring(0, marker).trim() to after.substring(marker).trim()
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
        val bracketMatches = CodexToolArguments.bracketRanges(text)
        if (bracketMatches.isNotEmpty()) {
            val calls = JSONArray()
            var messageText = text
            for (range in bracketMatches) {
                val rawCall = text.substring(range)
                val parsedCall = parseToolCallString(rawCall)
                if (parsedCall != null) {
                    messageText = messageText.replace(rawCall, "").trim()
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

    private fun parsePythonKwargs(str: String): JSONObject = CodexToolArguments.parse(str)

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
        if (rawName.substringAfterLast('.') == "finish") {
            val value = call.opt("arguments") ?: call.opt("parameters")
            val args = if (value is JSONObject) value else JSONObject(value?.toString() ?: "{}")
            val summary = args.getString("summary")
            require(summary.isNotBlank()) { "finish requires a factual summary" }
            return JSONObject().put("type", "message").put("id", "msg_${UUID.randomUUID()}")
                .put("role", "assistant").put("status", "completed")
                .put("content", JSONArray().put(JSONObject().put("type", "output_text").put("text", summary).put("annotations", JSONArray())))
        }
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

            val lowerRaw = cleanToolName(rawName).substringAfterLast('.').lowercase()
            if (isShellTool(qualified)) {
                // Full-access, unattended execution does not need escalation metadata.
                args.remove("justification")
                args.remove("sandbox_permissions")
                args.remove("prefix_rule")
                if (lowerRaw == "replace_in_file") {
                    val path = args.getString("path").removePrefix("/cwd/")
                    val old = args.getString("old_text")
                    args.put("command", CodexFileEdits.replaceCommand(path, old, args.getString("new_text")))
                    listOf("path", "old_text", "new_text").forEach(args::remove)
                } else if (lowerRaw in setOf("write_file", "create_file", "edit_file", "save_file", "file_write", "update_file")) {
                    var path = args.optString("path").ifBlank { args.optString("file").ifBlank { args.optString("filename").ifBlank { args.optString("target_file").ifBlank { args.optString("name") } } } }
                    if (path.contains(".llmhub-codex")) path = path.substringAfterLast('/')
                    path = path.removePrefix("/cwd/")
                    require(path.isNotBlank()) { "Missing file path" }
                    val contentKey = listOf("content", "code", "text", "body", "data").firstOrNull { args.has(it) && !args.isNull(it) }
                    require(contentKey != null) { "Missing file content" }
                    val content = args.getString(contentKey)
                    val encoded = java.util.Base64.getEncoder().encodeToString(content.toByteArray(Charsets.UTF_8))
                    val parent = java.io.File(path).parent
                    val prepare = if (parent != null) "mkdir -p -- ${CodexConfig.shellQuote(parent)} && " else ""
                    args.put("command", prepare + "printf '%s' ${CodexConfig.shellQuote(encoded)} | base64 -d > ${CodexConfig.shellQuote(path)}")
                } else if (lowerRaw in setOf("read_file", "view_file", "cat", "open_file", "file_read")) {
                    var path = args.optString("path").ifBlank { args.optString("file").ifBlank { args.optString("filename").ifBlank { args.optString("name") } } }
                    if (path.contains(".llmhub-codex")) {
                        path = if (path.endsWith(".html") || path.endsWith(".js") || path.endsWith(".css") || path.endsWith(".txt") || path.contains('.')) {
                            path.substringAfterLast('/')
                        } else ""
                    }
                    path = path.removePrefix("/cwd/")
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
                // shell_command accepts `command`; exec_command accepts `cmd`.
                // Adapt aliases to the offered schema rather than assuming one executor.
                val parameters = spec.optJSONObject("parameters")
                val required = parameters?.optJSONArray("required") ?: JSONArray()
                val usesCmd = parameters?.optJSONObject("properties")?.has("cmd") == true ||
                    (0 until required.length()).any { required.optString(it) == "cmd" }
                if (usesCmd && args.has("command")) {
                    args.put("cmd", args.remove("command"))
                }
                val commandKey = if (usesCmd) "cmd" else "command"
                if (Regex("^cat\\s").containsMatchIn(args.optString(commandKey).trim()) &&
                    args.has("max_output_tokens") && args.optInt("max_output_tokens") < 2048) {
                    // Tiny model-selected budgets hide the edited file and provoke repeated reads.
                    args.put("max_output_tokens", 2048)
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

    private fun appendCalls(result: JSONArray, calls: List<JSONObject>, available: Map<String, JSONObject>) {
        val finish = calls.any { cleanToolName(it.optString("name")).substringAfterLast('.') == "finish" }
        if (finish) {
            require(calls.size == 1) { "finish must be the only action; pending commands must execute first" }
            // One final summary, rather than a progress message plus the same summary again.
            while (result.length() > 0) result.remove(result.length() - 1)
        }
        calls.forEach { result.put(normalizeToolCall(it, available)) }
    }

    /** The native backend can wrap an ordinary final answer in an unclosed thinking sentinel. */
    fun nativeCompletion(raw: String): String? {
        if (!raw.startsWith(SENTINEL_THINK)) return null
        var body = stripThinking(raw).trim()
        if (body.isBlank() && !raw.contains(SENTINEL_ENDTHINK) && !raw.contains("<think>")) {
            body = raw.removePrefix(SENTINEL_THINK).trim()
        }
        if (body.isBlank() || body.startsWith('{') || body.contains("```json") ||
            body.contains("tool_call") || CodexToolArguments.bracketRanges(body).isNotEmpty()) return null
        return body
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
            appendCalls(result, (0 until calls.length()).map { calls.getJSONObject(it) }, available)
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

            appendCalls(result, callsList, available)

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
            return openThinking(after)
        }
        if (content.contains("<think>")) {
            val after = content.substringAfter("<think>")
            if (after.contains("</think>")) {
                return after.substringBefore("</think>").trim() to after.substringAfter("</think>").trim()
            }
            return openThinking(after)
        }
        if (content.contains("<thought>")) {
            val after = content.substringAfter("<thought>")
            if (after.contains("</thought>")) {
                return after.substringBefore("</thought>").trim() to after.substringAfter("</thought>").trim()
            }
            return openThinking(after)
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
            else cleanAnswer.removePrefix("```json").removePrefix("```")
                .substringBefore("<|tool_call_start|>").substringBefore("<tool_call>").trim()
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
