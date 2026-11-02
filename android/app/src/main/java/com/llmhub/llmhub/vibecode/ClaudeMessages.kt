package com.llmhub.llmhub.vibecode

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * Anthropic Messages wire adapter for Claude Code. Inference stays in LLM Hub; Claude Code owns tool execution.
 * Claude Code's own system prompt and tool descriptions run to tens of thousands of tokens, more than a phone model's
 * context, so the model sees a compact prompt instead: the working directory, the offered tools and the executed
 * results, with replies in the same JSON shape as Codex mode ([CodexGrammar] constrains them on llama.cpp).
 */
internal object ClaudeMessages {
    /** The tools a small model can drive. Claude Code's others (agents, web, notebooks, plans) are not offered. */
    private val CORE = listOf("Read", "Write", "Edit", "Bash", "Glob", "Grep", "LS")
    private val PATH_KEYS = setOf("file_path", "path")

    fun offeredTools(request: JSONObject): JSONArray {
        val all = request.optJSONArray("tools") ?: JSONArray()
        val tools = (0 until all.length()).mapNotNull { all.optJSONObject(it) }.filter { it.has("name") }
        val core = tools.filter { it.getString("name") in CORE }.ifEmpty { tools }
        return JSONArray().apply {
            for (tool in core) {
                val schema = JSONObject((tool.optJSONObject("input_schema") ?: JSONObject()).toString())
                schema.optJSONObject("properties")?.let { properties ->
                    for (key in properties.keys()) properties.optJSONObject(key)?.remove("description")
                }
                put(JSONObject().put("name", tool.getString("name")).put("type", "function")
                    .put("description", tool.optString("description").substringBefore("\n\n").take(240))
                    .put("parameters", schema))
            }
        }
    }

    /** The project folder Claude Code reports in its environment section of the system prompt. */
    fun workingDirectory(request: JSONObject): String? =
        Regex("(?m)^\\s*(?:Primary )?[Ww]orking directory:\\s*(\\S.*?)\\s*$").find(text(request.opt("system")))?.groupValues?.get(1)

    fun prompt(request: JSONObject): String = buildString {
        val cwd = workingDirectory(request)
        append("system: You are Claude Code, a coding agent").append(cwd?.let { " working in $it" } ?: "").append(".\n")
        append("Available tools:\n").append(offeredTools(request).toString()).append("\n\n")
        val conversation = conversation(request)
        if (conversation.isNotEmpty()) append(conversation).append("\n\n")
        if (lastEditSucceeded(request)) append("user: The last file change above succeeded and is saved. Do not make it again. ")
            .append("If it fulfils the request, finish now with tool_calls: [] and a one-sentence summary.\n\n")
        append("user: Continue after the executed tool results above.\n")
        append("""
            Act on the latest user request using the tools above. Tool results are real executed results.
            Return one JSON object:
            {"text":"short progress note or final reply","tool_calls":[{"name":"exact tool name","arguments":{}}]}
            Use absolute paths inside the working directory.
            To create a file or replace all of it, use Write with the complete content. For a small change to an
            existing file, Read it first, then Edit with old_string copied exactly from what Read returned.
            Keep everything the request does not change. Write complete, working code with real content: no placeholders or stubs.
            If a tool failed, change the call to fix its actual error instead of repeating it.
            When the request is done, return tool_calls: [] and a one-sentence summary of what changed.
            Never claim a change without a successful tool result.
        """.trimIndent())
    }

    /**
     * The request history for the model, compacted like Codex mode's: Claude Code resumes whole sessions, and repeated
     * copies of the same file push a small model into degenerate output. Results from earlier requests become a short
     * note, and in the current request a Read result is dropped once that file was read again or changed.
     */
    private fun conversation(request: JSONObject): String {
        val messages = request.optJSONArray("messages") ?: return ""
        val all = (0 until messages.length()).mapNotNull { messages.optJSONObject(it) }
        fun blocks(message: JSONObject): List<JSONObject> = message.opt("content").let { content ->
            if (content is JSONArray) (0 until content.length()).mapNotNull { content.optJSONObject(it) }
            else listOf(JSONObject().put("type", "text").put("text", content?.toString().orEmpty()))
        }
        fun request(text: String) = text.replace(Regex("(?s)<system-reminder>.*?</system-reminder>"), "").trim()
        val latest = all.indexOfLast { m -> m.optString("role") == "user" && blocks(m).any { it.optString("type") == "text" && request(it.optString("text")).isNotEmpty() } }
        val uses = mutableMapOf<String, JSONObject>()
        all.forEach { m -> blocks(m).filter { it.optString("type") == "tool_use" }.forEach { uses[it.optString("id")] = it } }
        // The last result per file in the current request; earlier reads of that file are stale.
        val lastTouch = mutableMapOf<String, String>()
        all.drop(maxOf(latest, 0)).forEach { m -> blocks(m).filter { it.optString("type") == "tool_result" }.forEach { result ->
            uses[result.optString("tool_use_id")]?.optJSONObject("input")?.optString("file_path")?.takeIf { it.isNotEmpty() }
                ?.let { lastTouch[it] = result.optString("tool_use_id") }
        } }
        val sb = StringBuilder()
        all.forEachIndexed { index, message ->
            for (block in blocks(message)) when (block.optString("type")) {
                // Earlier model text is not evidence, and small models copy their own past lines; calls show with results.
                "tool_result" -> {
                    val id = block.optString("tool_use_id")
                    val use = uses[id]
                    val failed = if (block.optBoolean("is_error")) " (failed)" else ""
                    val call = use?.let { " for " + call(it.optString("name"), it.optJSONObject("input") ?: JSONObject()) }.orEmpty()
                    val path = use?.optJSONObject("input")?.optString("file_path").orEmpty()
                    val output = when {
                        index < latest -> "(result from an earlier request omitted)"
                        use?.optString("name") == "Read" && path.isNotEmpty() && lastTouch[path] != id ->
                            "(superseded by a later read or change of this file; use the latest one)"
                        else -> text(block.opt("content")).take(12_000)
                    }
                    sb.append("user: Executed tool result$call$failed: ").append(output).append("\n\n")
                }
                "text" -> if (message.optString("role") == "user") {
                    val text = request(block.optString("text"))
                    if (text.isNotEmpty()) sb.append("user: ").append(text).append("\n\n")
                }
            }
        }
        return sb.toString().trim()
    }

    /**
     * A past call in words, without file contents. A JSON form such as {"old_string_bytes":32} gets copied by small
     * models as the arguments of their next call.
     */
    private fun call(name: String, input: JSONObject): String {
        fun bytes(key: String) = input.optString(key).toByteArray(Charsets.UTF_8).size
        return when (name) {
            "Write" -> "Write to ${input.optString("file_path")} (${bytes("content")} bytes)"
            "Edit" -> "Edit of ${input.optString("file_path")} (replaced ${bytes("old_string")} bytes with ${bytes("new_string")})"
            "Read" -> "Read of ${input.optString("file_path")}"
            "Bash" -> "Bash: ${input.optString("command").take(300)}"
            else -> "$name: ${input.toString().take(300)}"
        }
    }

    private fun text(value: Any?): String = when (value) {
        is JSONArray -> (0 until value.length()).joinToString("\n") { value.optJSONObject(it)?.optString("text") ?: value.optString(it) }
        null, JSONObject.NULL -> ""
        else -> value.toString()
    }

    /** The model's reply as Messages content blocks: an optional text block, then one tool_use block per call. */
    /**
     * Read-only calls that already succeeded since the latest request or file change. Repeating one only returns the
     * same text (Claude Code even answers an unchanged Read with a stub), and small models then read forever.
     */
    private fun completedReads(request: JSONObject): Set<String> {
        val messages = request.optJSONArray("messages") ?: return emptySet()
        val calls = mutableMapOf<String, String>()
        val done = mutableSetOf<String>()
        for (i in 0 until messages.length()) {
            val message = messages.optJSONObject(i) ?: continue
            val blocks = message.optJSONArray("content") ?: run { if (message.optString("role") == "user") done.clear(); continue }
            for (j in 0 until blocks.length()) {
                val block = blocks.optJSONObject(j) ?: continue
                when (block.optString("type")) {
                    "text" -> if (message.optString("role") == "user" &&
                        block.optString("text").replace(Regex("(?s)<system-reminder>.*?</system-reminder>"), "").isNotBlank()) done.clear()
                    "tool_use" -> calls[block.optString("id")] = key(block.optString("name"), block.optJSONObject("input") ?: JSONObject())
                    "tool_result" -> if (!block.optBoolean("is_error")) calls[block.optString("tool_use_id")]?.let { call ->
                        if (call.substringBefore('(') in READ_ONLY) done.add(call) else done.clear()
                    }
                }
            }
        }
        return done
    }
    private val READ_ONLY = setOf("Read", "Glob", "Grep", "LS")

    /** Whether a Write, Edit or Bash call has succeeded since the latest user request. */
    fun changedSinceRequest(request: JSONObject): Boolean = resultsSinceRequest(request, setOf("Write", "Edit", "Bash"))

    /** Whether any tool has run since the latest user request. */
    fun toolUsedSinceRequest(request: JSONObject): Boolean = resultsSinceRequest(request, null, successOnly = false)

    private fun resultsSinceRequest(request: JSONObject, names: Set<String>?, successOnly: Boolean = true): Boolean {
        val messages = request.optJSONArray("messages") ?: return false
        val calls = mutableMapOf<String, String>()
        var found = false
        for (i in 0 until messages.length()) {
            val message = messages.optJSONObject(i) ?: continue
            val blocks = message.optJSONArray("content") ?: run { if (message.optString("role") == "user") found = false; continue }
            for (j in 0 until blocks.length()) {
                val block = blocks.optJSONObject(j) ?: continue
                when (block.optString("type")) {
                    "text" -> if (message.optString("role") == "user" &&
                        block.optString("text").replace(Regex("(?s)<system-reminder>.*?</system-reminder>"), "").isNotBlank()) found = false
                    "tool_use" -> calls[block.optString("id")] = block.optString("name")
                    "tool_result" -> if ((!successOnly || !block.optBoolean("is_error")) &&
                        (names == null || calls[block.optString("tool_use_id")] in names)) found = true
                }
            }
        }
        return found
    }

    /** Whether the most recent tool result is a successful Write or Edit. */
    private fun lastEditSucceeded(request: JSONObject): Boolean {
        val messages = request.optJSONArray("messages") ?: return false
        val names = mutableMapOf<String, String>()
        var last = false
        for (i in 0 until messages.length()) {
            val blocks = messages.optJSONObject(i)?.optJSONArray("content") ?: continue
            for (j in 0 until blocks.length()) {
                val block = blocks.optJSONObject(j) ?: continue
                when (block.optString("type")) {
                    "tool_use" -> names[block.optString("id")] = block.optString("name")
                    "tool_result" -> last = !block.optBoolean("is_error") && names[block.optString("tool_use_id")] in setOf("Write", "Edit")
                    "text" -> if (messages.optJSONObject(i).optString("role") == "user") last = false
                }
            }
        }
        return last
    }
    private fun key(name: String, input: JSONObject) = "$name(" + input.keys().asSequence().sorted()
        .joinToString(",") { "$it=${input.opt(it)}" } + ")"

    fun content(raw: String, request: JSONObject): JSONArray {
        val found = CodexResponses.replyJson(raw)
            ?: CodexResponses.gemmaToolCalls(CodexResponses.stripThinking(raw))?.let { (text, calls) -> JSONObject().put("text", text).put("tool_calls", calls) }
            ?: throw org.json.JSONException("No reply JSON found in: ${raw.take(400)}")
        // A damaged reply can leave only its inner call object; read that as the call, and refuse anything else
        // without the reply fields rather than end the turn as an empty answer.
        // Misspelled list keys ("toolcalls", "tool_call") still name the calls.
        found.keys().asSequence().toList().firstOrNull { it != "tool_calls" && it.lowercase().filter(Char::isLetter) in setOf("toolcalls", "toolcall") }
            ?.let { if (!found.has("tool_calls")) found.put("tool_calls", found.get(it)) }
        val reply = when {
            found.has("tool_calls") || found.has("text") -> found
            found.has("name") && found.has("arguments") -> JSONObject().put("text", "").put("tool_calls", JSONArray().put(found))
            else -> throw org.json.JSONException("Reply has neither text nor tool_calls: ${raw.take(400)}")
        }
        val completed = completedReads(request)
        val offered = offeredTools(request)
        val byName = (0 until offered.length()).map { offered.getJSONObject(it) }.associateBy { it.getString("name").lowercase() }
        val cwd = workingDirectory(request)
        val blocks = JSONArray()
        reply.optString("text").takeIf { it.isNotBlank() }?.let { blocks.put(JSONObject().put("type", "text").put("text", it)) }
        val calls = reply.optJSONArray("tool_calls") ?: reply.optJSONObject("tool_calls")?.let { JSONArray().put(it) } ?: JSONArray()
        for (i in 0 until calls.length()) {
            val call = calls.optJSONObject(i) ?: continue
            val name = call.optString("name").substringAfterLast('.')
            val tool = byName[name.lowercase()] ?: throw IllegalArgumentException("Unknown tool: $name")
            val input = call.optJSONObject("arguments") ?: JSONObject()
            // Argument names the model misspelled ("newstring", "filePath") become the schema's own names.
            val properties = tool.optJSONObject("parameters")?.optJSONObject("properties")?.keys()?.asSequence()?.toList().orEmpty()
            fun plain(key: String) = key.lowercase().filter(Char::isLetterOrDigit)
            for (key in input.keys().asSequence().toList()) if (key !in properties) {
                properties.firstOrNull { plain(it) == plain(key) && !input.has(it) }?.let { input.put(it, input.remove(key)) }
            }
            val required = tool.optJSONObject("parameters")?.optJSONArray("required") ?: JSONArray()
            for (n in 0 until required.length()) require(input.has(required.getString(n))) { "Missing tool argument: ${required.getString(n)}" }
            // Claude Code's file tools take absolute paths; small models often give them relative to the project.
            if (cwd != null) for (key in PATH_KEYS) input.optString(key).takeIf { it.isNotEmpty() && !it.startsWith("/") }?.let {
                input.put(key, cwd.trimEnd('/') + "/" + it.removePrefix("./"))
            }
            require(key(tool.getString("name"), input) !in completed) {
                "Repeated ${tool.getString("name")}: this exact call already succeeded and nothing changed since; its result is above. " +
                    "Use it: call Edit with old_string copied from that result, Write the complete file, or finish."
            }
            blocks.put(JSONObject().put("type", "tool_use").put("id", "toolu_${UUID.randomUUID().toString().replace("-", "")}")
                .put("name", tool.getString("name")).put("input", input))
        }
        if (blocks.length() == 0) blocks.put(JSONObject().put("type", "text").put("text", ""))
        return blocks
    }

    fun message(id: String, model: String, content: JSONArray, inputTokens: Int, outputTokens: Int): JSONObject =
        JSONObject().put("id", id).put("type", "message").put("role", "assistant").put("model", model)
            .put("content", content).put("stop_reason", stopReason(content)).put("stop_sequence", JSONObject.NULL)
            .put("usage", JSONObject().put("input_tokens", inputTokens).put("output_tokens", outputTokens))

    fun stopReason(content: JSONArray): String =
        if ((0 until content.length()).any { content.getJSONObject(it).optString("type") == "tool_use" }) "tool_use" else "end_turn"

    /** The streaming events for an already complete message, in the order Anthropic's API sends them. */
    fun events(message: JSONObject): List<JSONObject> = buildList {
        val content = message.getJSONArray("content")
        val usage = message.getJSONObject("usage")
        add(JSONObject().put("type", "message_start").put("message", JSONObject(message.toString())
            .put("content", JSONArray()).put("stop_reason", JSONObject.NULL)
            .put("usage", JSONObject().put("input_tokens", usage.getInt("input_tokens")).put("output_tokens", 1))))
        for (i in 0 until content.length()) {
            val block = content.getJSONObject(i)
            if (block.optString("type") == "tool_use") {
                add(JSONObject().put("type", "content_block_start").put("index", i).put("content_block", JSONObject()
                    .put("type", "tool_use").put("id", block.getString("id")).put("name", block.getString("name")).put("input", JSONObject())))
                add(JSONObject().put("type", "content_block_delta").put("index", i).put("delta", JSONObject()
                    .put("type", "input_json_delta").put("partial_json", block.getJSONObject("input").toString())))
            } else {
                add(JSONObject().put("type", "content_block_start").put("index", i)
                    .put("content_block", JSONObject().put("type", "text").put("text", "")))
                if (block.optString("text").isNotEmpty()) add(JSONObject().put("type", "content_block_delta").put("index", i)
                    .put("delta", JSONObject().put("type", "text_delta").put("text", block.getString("text"))))
            }
            add(JSONObject().put("type", "content_block_stop").put("index", i))
        }
        add(JSONObject().put("type", "message_delta").put("delta", JSONObject().put("stop_reason", message.getString("stop_reason"))
            .put("stop_sequence", JSONObject.NULL)).put("usage", JSONObject().put("output_tokens", usage.getInt("output_tokens"))))
        add(JSONObject().put("type", "message_stop"))
    }
}
